"""Unit and integration-contract tests for the parameter optimizer."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from analysis.optimizer import (
    SearchSettings,
    accelerator_status,
    build_non_overlapping_windows,
    evaluate_candidates,
    exhaustive_combinations,
    rank_evaluations,
    search,
    values_for_space,
)


START = datetime(2020, 1, 1, tzinfo=timezone.utc)
END = datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_non_overlapping_windows_cover_range_exactly():
    windows = build_non_overlapping_windows(START, END, 4)
    assert len(windows) == 4
    assert windows[0][0] == START
    assert windows[-1][1] == END
    for left, right in zip(windows, windows[1:]):
        assert left[1] == right[0]
        assert left[0] < left[1]


@pytest.mark.parametrize(
    ("start", "end", "count"),
    [(START.replace(tzinfo=None), END, 2), (START, START, 2), (START, END, 0)],
)
def test_non_overlapping_windows_reject_invalid_inputs(start, end, count):
    with pytest.raises(ValueError):
        build_non_overlapping_windows(start, end, count)


def test_finite_spaces_include_endpoints_without_float_overshoot():
    assert values_for_space({"type": "int", "low": 1, "high": 5, "step": 2}) == [1, 3, 5]
    assert values_for_space({"type": "float", "low": 0.1, "high": 0.3, "step": 0.1}) == pytest.approx([0.1, 0.2, 0.3])
    assert values_for_space({"type": "categorical", "choices": ["a", "b"]}) == ["a", "b"]


def test_exhaustive_combinations_enumerates_full_cross_product():
    combos = exhaustive_combinations({
        "a": {"type": "int", "low": 1, "high": 2},
        "b": {"type": "categorical", "choices": ["x", "y", "z"]},
    })
    assert len(combos) == 6
    assert {tuple(row.values()) for row in combos} == {
        (a, b) for a in (1, 2) for b in ("x", "y", "z")
    }


def _metric(sharpe, cagr, drawdown=-0.1, trades=10):
    return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": drawdown, "n_trades": trades}


def test_ranking_prefers_stable_candidate_and_marks_pareto_frontier():
    frame = pd.DataFrame([
        {"p": 1, "window_idx": 0, **_metric(1.20, 0.15, -0.20)},
        {"p": 1, "window_idx": 1, **_metric(0.20, 0.04, -0.30)},
        {"p": 2, "window_idx": 0, **_metric(0.85, 0.12, -0.10)},
        {"p": 2, "window_idx": 1, **_metric(0.80, 0.11, -0.11)},
    ])
    ranked = rank_evaluations(frame, ["p"], SearchSettings(algorithm="grid"))
    assert ranked.iloc[0]["p"] == 2
    assert bool(ranked.iloc[0]["pareto_optimal"])
    assert ranked.iloc[0]["worst_sharpe"] == pytest.approx(0.80)


def test_no_trade_or_nan_candidate_is_ineligible():
    frame = pd.DataFrame([
        {"p": 1, "window_idx": 0, **_metric(np.nan, 0.1, trades=0)},
        {"p": 2, "window_idx": 0, **_metric(0.5, 0.1, trades=2)},
    ])
    ranked = rank_evaluations(frame, ["p"], SearchSettings(algorithm="grid"))
    assert ranked.iloc[0]["p"] == 2
    bad = ranked[ranked["p"] == 1].iloc[0]
    assert not bool(bad["eligible"])
    assert bad["robust_score"] == float("-inf")


def test_stability_gates_require_positive_windows_and_worst_sharpe():
    frame = pd.DataFrame([
        {"p": 1, "window_idx": 0, **_metric(1.0, 0.10)},
        {"p": 1, "window_idx": 1, **_metric(-0.1, -0.01)},
        {"p": 2, "window_idx": 0, **_metric(0.8, 0.08)},
        {"p": 2, "window_idx": 1, **_metric(0.6, 0.06)},
    ])
    settings = SearchSettings(
        algorithm="grid", min_positive_windows=2, min_worst_sharpe=0.0,
    )
    ranked = rank_evaluations(frame, ["p"], settings)

    assert ranked.iloc[0]["p"] == 2
    assert bool(ranked.iloc[0]["eligible"])
    rejected = ranked[ranked["p"] == 1].iloc[0]
    assert not bool(rejected["eligible"])
    assert rejected["positive_windows"] == 1


def test_grid_search_evaluates_every_combo_in_every_window():
    calls = []
    windows = build_non_overlapping_windows(START, END, 2)

    def evaluator(params, window):
        calls.append((dict(params), window))
        p = params["p"]
        return _metric(1.0 - abs(p - 3) * 0.2, 0.10 - abs(p - 3) * 0.01)

    ranked, rows = search(
        base_params={"fixed": 9},
        param_space={"p": {"type": "int", "low": 1, "high": 5}},
        windows=windows,
        evaluator=evaluator,
        settings=SearchSettings(algorithm="grid"),
    )
    assert len(calls) == 10
    assert len(rows) == 10
    assert len(ranked) == 5
    assert ranked.iloc[0]["p"] == 3
    assert all(params["fixed"] == 9 for params, _ in calls)


def test_explicit_finalists_are_all_validated_in_every_window():
    windows = build_non_overlapping_windows(START, END, 3)
    calls = []

    def evaluator(params, window):
        calls.append((params["p"], window))
        return _metric(params["p"] / 10, params["p"] / 100)

    ranked, rows = evaluate_candidates(
        base_params={"fixed": True},
        candidates=[{"p": 2}, {"p": 5}],
        windows=windows,
        evaluator=evaluator,
        settings=SearchSettings(algorithm="grid"),
    )
    assert len(calls) == 6
    assert len(rows) == 6
    assert len(ranked) == 2
    assert ranked.iloc[0]["p"] == 5


def test_seeded_tpe_finds_good_region_and_deduplicates_discrete_trials():
    windows = build_non_overlapping_windows(START, END, 2)

    def evaluator(params, _window):
        score = 1.0 - abs(params["p"] - 4) * 0.25
        return _metric(score, score / 10)

    settings = SearchSettings(algorithm="tpe", n_trials=20, seed=7)
    first, first_rows = search(
        base_params={},
        param_space={"p": {"type": "int", "low": 1, "high": 6}},
        windows=windows,
        evaluator=evaluator,
        settings=settings,
    )
    second, _ = search(
        base_params={},
        param_space={"p": {"type": "int", "low": 1, "high": 6}},
        windows=windows,
        evaluator=evaluator,
        settings=settings,
    )
    assert first.iloc[0]["p"] == 4
    assert second.iloc[0]["p"] == 4
    assert not first_rows.duplicated(["p", "window_idx"]).any()


def test_accelerator_status_is_explicit_about_cpu_event_engine():
    status = accelerator_status()
    assert status["event_backtester"] == "cpu"
    assert isinstance(status["cuda_available"], bool)
    assert status["gpu_scope"] in {"batched vectorized pre-screening", "unavailable"}
