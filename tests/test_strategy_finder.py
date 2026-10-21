"""Tests for constrained structural strategy discovery."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from analysis.optimizer import SearchSettings
from analysis.strategy_finder import (
    FinderSettings,
    StrategyGrammar,
    StrategyStructure,
    apply_complexity_penalty,
    describe_strategy,
    find_strategies,
    generate_candidates,
    generate_structures,
    select_structural_finalists,
)
from analysis.structural_screen import screen_rule_candidates
from strategies.rule_rotation import score_rule


def test_rule_scores_are_distinct_and_validate():
    closes = [100.0, 105.0, 110.0]
    assert score_rule(closes, signal_family="momentum", lookback_days=2, skip_days=0) == pytest.approx(0.10)
    assert score_rule(closes, signal_family="mean_reversion", lookback_days=2, skip_days=0) == pytest.approx(-0.10)
    assert score_rule([100.0], signal_family="momentum", lookback_days=2, skip_days=0) is None
    with pytest.raises(ValueError, match="unsupported"):
        score_rule(closes, signal_family="invented", lookback_days=2, skip_days=0)


def test_default_grammar_generates_twelve_unique_structures():
    structures = generate_structures()
    assert len(structures) == 12
    assert len({item.structure_id for item in structures}) == 12
    assert {item.signal_family for item in structures} == {"momentum", "mean_reversion"}


def test_complexity_limit_removes_richer_structures():
    grammar = StrategyGrammar(max_complexity=1)
    structures = generate_structures(grammar)
    assert len(structures) == 2
    assert all(item.trend_filter_days == 0 and not item.require_positive for item in structures)


def test_candidate_generation_crosses_structure_and_parameter_spaces():
    structures = [
        StrategyStructure("momentum", 0, False),
        StrategyStructure("mean_reversion", 0, False),
    ]
    candidates = generate_candidates(
        structures,
        {"top_n": {"type": "int", "low": 1, "high": 2, "step": 1}},
    )
    assert len(candidates) == 4
    assert all({"structure_id", "complexity", "signal_family", "top_n"} <= set(row) for row in candidates)


def test_structural_screen_runs_distinct_families_on_cpu():
    index = pd.bdate_range("2020-01-01", periods=320)
    x = np.arange(len(index), dtype=float)
    closes = pd.DataFrame({
        "UP": 100.0 + x * 0.30,
        "DOWN": 200.0 - x * 0.20,
        "FLAT": 100.0 + np.sin(x / 10.0),
    }, index=index)
    base = {
        "trend_filter_days": 0,
        "require_positive": False,
        "top_n": 1,
        "lookback_days": 42,
        "skip_days": 0,
        "complexity": 1,
    }
    candidates = [
        {**base, "signal_family": "momentum", "structure_id": "momentum"},
        {**base, "signal_family": "mean_reversion", "structure_id": "mean_reversion"},
    ]
    screened = screen_rule_candidates(closes, candidates, device="cpu")
    assert set(screened["signal_family"]) == {"momentum", "mean_reversion"}
    assert set(screened["screen_device"]) == {"cpu"}
    assert screened["screen_trades"].gt(0).all()
    assert screened["screen_final_equity"].nunique() == 2


def test_structural_finalists_preserve_distinct_rule_families():
    screened = pd.DataFrame([
        {"structure_id": "a", "screen_score": 3.0, "p": 1},
        {"structure_id": "a", "screen_score": 2.9, "p": 2},
        {"structure_id": "b", "screen_score": 2.0, "p": 1},
    ])
    finalists = select_structural_finalists(screened, 2)
    assert set(finalists["structure_id"]) == {"a", "b"}


def test_complexity_penalty_can_prefer_simpler_equal_performer():
    ranked = pd.DataFrame([
        {"rank": 1, "eligible": True, "robust_score": 1.0, "median_sharpe": 1.0, "median_cagr": 0.1, "complexity": 3},
        {"rank": 2, "eligible": True, "robust_score": 1.0, "median_sharpe": 1.0, "median_cagr": 0.1, "complexity": 1},
    ])
    result = apply_complexity_penalty(ranked, 0.05)
    assert result.iloc[0]["complexity"] == 1
    assert result.iloc[0]["unpenalized_robust_score"] == 1.0


def test_finder_orchestration_returns_explainable_rankings():
    index = pd.bdate_range("2020-01-01", periods=300)
    closes = pd.DataFrame({
        "A": np.linspace(100, 180, len(index)),
        "B": np.linspace(100, 80, len(index)),
    }, index=index)
    structures = generate_structures(StrategyGrammar(
        trend_filter_days=(0,), positive_gates=(False,),
    ))
    windows = [
        (datetime(2020, 1, 1, tzinfo=timezone.utc), datetime(2020, 7, 1, tzinfo=timezone.utc)),
        (datetime(2020, 7, 1, tzinfo=timezone.utc), datetime(2021, 1, 1, tzinfo=timezone.utc)),
    ]

    def evaluator(params, window):
        favorable = params["signal_family"] == "momentum"
        return {
            "sharpe": 1.2 if favorable else 0.4,
            "cagr": 0.12 if favorable else 0.04,
            "max_drawdown": -0.10,
            "n_trades": 8,
        }

    ranked, evaluations, screened = find_strategies(
        closes=closes,
        structures=structures,
        parameter_space={
            "top_n": {"type": "int", "low": 1, "high": 1, "step": 1},
            "lookback_days": {"type": "int", "low": 42, "high": 42, "step": 1},
            "skip_days": {"type": "int", "low": 0, "high": 0, "step": 1},
        },
        windows=windows,
        evaluator=evaluator,
        settings=FinderSettings(
            finalists=2,
            device="cpu",
            search=SearchSettings(algorithm="grid"),
        ),
    )
    assert ranked.iloc[0]["signal_family"] == "momentum"
    assert len(evaluations) == 4
    assert len(screened) == 2
    assert "highest returns" in describe_strategy(ranked.iloc[0])
