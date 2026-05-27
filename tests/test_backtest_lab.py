"""Tests for dashboard.views.backtest_lab pure helpers."""
from __future__ import annotations

import pytest

from dashboard.views.backtest_lab import (
    discover_strategies,
    make_param_grid,
    field_default,
)


def test_discover_strategies_finds_momentum_topn_and_meta():
    """The two strategies we ship (Stage-2 baseline + Stage-5 meta) discover."""
    strategies = discover_strategies()
    assert "momentum_topn" in strategies
    assert "momentum_meta" in strategies
    s, c = strategies["momentum_topn"]
    assert s.__name__ == "MomentumTopNStrategy"
    assert c.__name__ == "MomentumTopNConfig"


def test_discover_strategies_returns_pairs():
    strategies = discover_strategies()
    for name, (strategy_cls, config_cls) in strategies.items():
        # Each pair must be class types; verified above with subclass checks
        # in the implementation. Here we just check the shape.
        assert isinstance(strategy_cls, type)
        assert isinstance(config_cls, type)


def test_make_param_grid_single_param():
    grid = make_param_grid({"top_n": [1, 2, 3]})
    assert len(grid) == 3
    assert grid[0] == {"top_n": 1}
    assert grid[-1] == {"top_n": 3}


def test_make_param_grid_combinatorial():
    """2 params × 3 levels × 2 levels = 6 cells."""
    grid = make_param_grid({"top_n": [1, 2, 3], "lookback_days": [60, 90]})
    assert len(grid) == 6
    # Verify the cross-product hits every combination
    expected = {(t, l) for t in (1, 2, 3) for l in (60, 90)}
    actual = {(g["top_n"], g["lookback_days"]) for g in grid}
    assert actual == expected


def test_make_param_grid_empty_returns_one_empty_dict():
    """Empty ranges = no-op = run with base params only."""
    assert make_param_grid({}) == [{}]


def test_field_default_returns_msgspec_default():
    """field_default should return the explicit default for a msgspec-Struct field."""
    from strategies.momentum_topn import MomentumTopNConfig
    # top_n has default 3, lookback_days has default 126
    assert field_default("top_n", MomentumTopNConfig) == 3
    assert field_default("lookback_days", MomentumTopNConfig) == 126


def test_config_fields_excludes_inherited():
    """config_fields should only return the strategy's own params, not the
    16+ inherited fields from StrategyConfig / NautilusConfig / Struct."""
    from dashboard.views.backtest_lab import config_fields
    from strategies.momentum_topn import MomentumTopNConfig
    fields = config_fields(MomentumTopNConfig)
    # MomentumTopNConfig defines: universe, benchmark, top_n, lookback_days, skip_days
    assert set(fields.keys()) == {"universe", "benchmark", "top_n", "lookback_days", "skip_days"}
    # And NOT the inherited ones
    assert "strategy_id" not in fields
    assert "oms_type" not in fields
