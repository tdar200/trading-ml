"""Tests for selective long/short opening-range discovery."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.cuda_scalping import simulate_scalping_candidates_cuda
from analysis.opening_range import (
    OpeningRangeGrammar,
    add_pooled_context_features,
    block_bootstrap_return_confidence,
    correlation_aware_weights,
    expand_opening_range_exits,
    generate_opening_range_entries,
    opening_range_signal_matrix,
    parameter_neighborhood_support,
    portfolio_metrics_from_curves,
    prepare_opening_range_features,
    replay_opening_range_candidate,
    trailing_performance_gate,
    split_opening_range_nested,
)
from analysis.scalping import prepare_intraday_bars


def _bars(sessions: int = 10, bars_per_session: int = 8) -> pd.DataFrame:
    rows = []
    for session_index, date in enumerate(
        pd.bdate_range("2024-01-02", periods=sessions)
    ):
        start = (
            pd.Timestamp(date.date(), tz="America/New_York")
            + pd.Timedelta(hours=9, minutes=30)
        )
        base = 100.0 + session_index
        for bar in range(bars_per_session):
            close = base + 0.2 * bar
            rows.append({
                "ts": (start + pd.Timedelta(minutes=5 * bar)).tz_convert("UTC"),
                "open": close - 0.05,
                "high": close + 0.10,
                "low": close - 0.10,
                "close": close,
                "volume": 1_000.0 + 100 * bar + 10 * session_index,
            })
    return prepare_intraday_bars(pd.DataFrame(rows))


def test_trailing_performance_gate_is_causal():
    prior = np.full(30, 0.001)
    prospective = np.asarray([0.001, -0.001, 0.002, -0.50])
    original = trailing_performance_gate(prior, prospective, lookback=20)
    changed = trailing_performance_gate(
        prior, np.asarray([0.001, -0.001, 0.002, 0.50]), lookback=20,
    )
    np.testing.assert_array_equal(original, changed)
    assert original.dtype == bool


def test_default_grammar_is_finite_and_expands_all_exits():
    grammar = OpeningRangeGrammar()
    entries = generate_opening_range_entries(grammar)
    assert len(entries) == 324
    assert len(expand_opening_range_exits(entries, grammar)) == 17_496
    assert {entry["direction"] for entry in entries} == {"long", "short", "both"}


def test_pooled_context_is_causal_and_uses_completed_bars():
    traded = _bars(sessions=10)
    context = _bars(sessions=10)
    original = add_pooled_context_features(traded, {"CTX": context})
    changed = context.copy()
    changed.loc[changed.index[-1], "close"] *= 10
    mutated = add_pooled_context_features(traded, {"CTX": changed})
    np.testing.assert_allclose(
        original["context_direction"].iloc[:-1],
        mutated["context_direction"].iloc[:-1],
        equal_nan=True,
    )
    assert original["context_direction"].iloc[-1] != mutated["context_direction"].iloc[-1]


def test_nested_split_separates_orientation_purge_validation_and_holdout():
    features = _bars(sessions=50)
    orientation, purged, windows, holdout = split_opening_range_nested(
        features, holdout_fraction=0.2, orientation_sessions=20,
        purge_sessions=4, validation_windows=4,
    )
    groups = [orientation, purged, *windows, holdout]
    session_sets = [set(frame["session"]) for frame in groups]
    assert len(session_sets[0]) == 20
    assert len(session_sets[1]) == 4
    assert len(session_sets[-1]) == 10
    assert all(left.isdisjoint(right) for i, left in enumerate(session_sets) for right in session_sets[i + 1:])
    assert max(session_sets[0]) < min(session_sets[1]) < min(session_sets[2])


def test_features_are_causal_before_a_future_mutation():
    bars = _bars()
    original = prepare_opening_range_features(bars, bars)
    changed_bars = bars.copy()
    changed_bars.loc[changed_bars.index[-1], ["high", "volume"]] *= 100
    changed = prepare_opening_range_features(changed_bars, bars)
    columns = [
        "atr_fraction", "relative_volume", "market_direction",
        "regime_strength", "regime_direction",
    ]
    pd.testing.assert_frame_equal(
        original.loc[:-2, columns],
        changed.loc[:-2, columns],
    )


def test_opening_range_signal_matrix_emits_requested_directions():
    features = _bars()
    features["bar_in_session"] = (
        (features["minute_of_day"] - (9 * 60 + 30)) // 5
    )
    features["atr_fraction"] = 0.001
    features["relative_volume"] = 2.0
    features["market_direction"] = 0.01
    features["regime_strength"] = 1.0
    grammar = OpeningRangeGrammar(
        opening_range_bars=(3,),
        breakout_atr=(0.0,),
        min_relative_volume=(1.0,),
        market_alignment=(False,),
        fresh_breakout=(False, True),
        directions=("long", "short", "both"),
        max_hold_bars=(6,),
        take_profit_atr=(1.0,),
        stop_loss_atr=(1.0,),
    )
    entries = generate_opening_range_entries(grammar)
    matrix = opening_range_signal_matrix(
        features, entries, device="cpu",
    ).numpy()
    assert set(np.unique(matrix[:, 0])).issubset({0, 1})
    assert 1 in matrix[:, 0]
    assert set(np.unique(matrix[:, 1])).issubset({-1, 0})
    assert set(np.unique(matrix[:, 2])).issubset({-1, 0, 1})
    assert matrix[:, 3].sum() < matrix[:, 0].sum()


def test_cuda_signed_atr_simulation_obeys_one_trade_daily_cap():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    bars = _bars(sessions=2, bars_per_session=6)
    signals = torch.zeros((len(bars), 1), dtype=torch.int8, device="cuda")
    signals[0:5, 0] = 1
    signals[6:11, 0] = -1
    candidate = {
        "family": "opening_range_breakout",
        "opening_range_bars": 1,
        "breakout_atr": 0.0,
        "min_relative_volume": 1.0,
        "market_alignment": False,
        "direction": "both",
        "entry_rule_id": "test",
        "max_hold_bars": 20,
        "take_profit_atr": 100.0,
        "stop_loss_atr": 100.0,
        "max_trades_per_session": 1,
    }
    result, curves = simulate_scalping_candidates_cuda(
        bars,
        [candidate],
        cost_per_side_bps=0.0,
        benchmark_bars=bars,
        signal_matrix=signals,
        risk_fraction=np.full(len(bars), 0.01),
        volatility_exits=True,
        max_trades_per_session=1,
        return_curves=True,
    )
    assert result.iloc[0]["n_trades"] == 2
    long_return = bars.iloc[5]["close"] / bars.iloc[3]["open"] - 1.0
    short_return = bars.iloc[9]["open"] / bars.iloc[11]["close"] - 1.0
    assert result.iloc[0]["total_return"] == pytest.approx(
        (1.0 + long_return) * (1.0 + short_return) - 1.0,
    )
    assert curves.shape == (2, 1)


def test_trade_ledger_matches_cuda_winner_replay():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    bars = _bars(sessions=10, bars_per_session=8)
    features = prepare_opening_range_features(bars, bars)
    grammar = OpeningRangeGrammar(
        opening_range_bars=(3,), breakout_atr=(0.0,),
        min_relative_volume=(1.0,), market_alignment=(False,),
        fresh_breakout=(False,),
        directions=("long",), max_hold_bars=(6,),
        take_profit_atr=(100.0,), stop_loss_atr=(100.0,),
    )
    candidate = expand_opening_range_exits(
        generate_opening_range_entries(grammar), grammar,
    )[0]
    signals = opening_range_signal_matrix(features, [candidate], device="cuda")
    result, _ = simulate_scalping_candidates_cuda(
        features, [candidate], cost_per_side_bps=2.5,
        benchmark_bars=bars, signal_matrix=signals,
        risk_fraction=features["atr_fraction"].to_numpy(dtype=float),
        volatility_exits=True, max_trades_per_session=1, return_curves=True,
    )
    ledger = replay_opening_range_candidate(
        features, candidate, cost_per_side_bps=2.5,
    )

    assert len(ledger) == result.iloc[0]["n_trades"]
    assert ((1.0 + ledger["net_return"]).prod() - 1.0) == pytest.approx(
        result.iloc[0]["total_return"], rel=1e-10, abs=1e-12,
    )
    assert set(ledger["exit_reason"]) <= {
        "target", "stop", "time", "session_boundary", "session_close",
    }


def test_block_bootstrap_is_deterministic_and_reports_interval():
    returns = np.array([0.01, -0.005, 0.007, 0.002, -0.001] * 10)
    first = block_bootstrap_return_confidence(
        returns, block_size=5, samples=200, seed=7,
    )
    second = block_bootstrap_return_confidence(
        returns, block_size=5, samples=200, seed=7,
    )
    assert first == second
    assert first["lower_95"] <= first["median"] <= first["upper_95"]
    assert 0 <= first["probability_positive"] <= 1


def test_correlation_weights_are_capped_and_normalized():
    frames = {f"T{index}": _bars(sessions=30) for index in range(5)}
    weights = correlation_aware_weights(frames, maximum_weight=0.25)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert max(weights.values()) <= 0.25
    assert set(weights) == set(frames)


def test_parameter_neighborhood_support_requires_adjacent_eligible_rules():
    ranked = pd.DataFrame([
        {"eligible": True, "breakout_atr": 0.0, "min_relative_volume": 1.0, "max_hold_bars": 1, "take_profit_atr": 1.0, "stop_loss_atr": 1.0},
        {"eligible": True, "breakout_atr": 0.0, "min_relative_volume": 1.0, "max_hold_bars": 2, "take_profit_atr": 1.0, "stop_loss_atr": 1.0},
        {"eligible": False, "breakout_atr": 0.5, "min_relative_volume": 2.0, "max_hold_bars": 6, "take_profit_atr": 1.5, "stop_loss_atr": 0.5},
    ])
    assert parameter_neighborhood_support(ranked).tolist() == [1, 1, 0]


def test_weighted_portfolio_metrics_use_supplied_capital_weights():
    candidate = {"p": 1}
    frame = pd.DataFrame([{"n_trades": 1}])
    metrics = portfolio_metrics_from_curves(
        [candidate], [frame, frame],
        [np.array([[1.10], [1.10]]), np.array([[0.90], [0.90]])],
        weights=[0.75, 0.25],
    ).iloc[0]
    assert metrics["total_return"] == pytest.approx(0.05)


def test_portfolio_metrics_include_first_session_return():
    candidate = {"p": 1}
    frame = pd.DataFrame([{"n_trades": 1}])
    metrics = portfolio_metrics_from_curves(
        [candidate],
        [frame, frame],
        [np.array([[1.10], [1.10]]), np.array([[0.90], [0.90]])],
    ).iloc[0]
    assert metrics["total_return"] == pytest.approx(0.0)
    assert metrics["n_trades"] == 2
