"""Tests for dashboard.views.backtest_lab pure helpers."""
from __future__ import annotations

import pytest

from dashboard.views.backtest_lab import (
    discover_strategies,
    make_param_grid,
    field_default,
    valid_numeric_space,
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


def test_valid_numeric_space_accepts_enumerable_ranges():
    space = {
        "top_n": {"low": 1, "high": 5, "step": 1},
        "lookback_days": {"low": 21, "high": 126, "step": 21},
    }
    assert valid_numeric_space(space)


@pytest.mark.parametrize(
    "space",
    [
        {},
        {"top_n": {"low": 5, "high": 1, "step": 1}},
        {"top_n": {"low": 1, "high": 5, "step": 0}},
    ],
)
def test_valid_numeric_space_rejects_transient_form_states(space):
    assert not valid_numeric_space(space)


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


def test_ann_kind_handles_pep563_string_annotations():
    """Regression for the 'can only concatenate str (not int) to str' TypeError:
    the strategy file uses ``from __future__ import annotations`` so
    ``__annotations__`` returns string literals ('int', 'float', 'str', ...)
    not type objects. _ann_kind must classify both shapes."""
    from dashboard.views.backtest_lab import _ann_kind
    # PEP-563 string form
    assert _ann_kind("int") == "int"
    assert _ann_kind("float") == "float"
    assert _ann_kind("bool") == "bool"
    assert _ann_kind("str") == "str"
    assert _ann_kind("tuple[str, ...]") == "tuple-str"
    # Resolved-type form (some files don't use future-annotations)
    assert _ann_kind(int) == "int"
    assert _ann_kind(float) == "float"
    assert _ann_kind(bool) == "bool"
    assert _ann_kind(str) == "str"
    # Unknown
    assert _ann_kind("dict[str, int]") == "unknown"
    assert _ann_kind(list) == "unknown"


def test_ann_kind_matches_real_strategy_configs():
    """Verifies the bug doesn't return — every numeric field in our two
    strategies must classify as int or float, not 'unknown'."""
    from dashboard.views.backtest_lab import config_fields, _ann_kind
    from strategies.momentum_topn import MomentumTopNConfig
    fields = config_fields(MomentumTopNConfig)
    # top_n, lookback_days, skip_days are ints; benchmark is str; universe is tuple-str
    assert _ann_kind(fields["top_n"][0]) == "int"
    assert _ann_kind(fields["lookback_days"][0]) == "int"
    assert _ann_kind(fields["skip_days"][0]) == "int"
    assert _ann_kind(fields["benchmark"][0]) == "str"
    assert _ann_kind(fields["universe"][0]) == "tuple-str"
