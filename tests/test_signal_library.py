from collections import Counter

import numpy as np
import pandas as pd

from analysis.signal_library import (
    PARAMETRIC_INDICATORS, SIGNAL_NAMES, IndicatorVariant,
    build_indicator_variants, compute_parameterized_semantic_signal_library,
    compute_parameterized_signal_library, compute_semantic_signal_library,
    compute_signal_library, signal_category,
)


def _bars(rows=160):
    rng = np.random.default_rng(12)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.002, rows))
    return pd.DataFrame({
        "ts": pd.date_range("2026-01-02 14:30", periods=rows, freq="h", tz="UTC"),
        "open": close * 0.999, "high": close * 1.002,
        "low": close * 0.998, "close": close,
        "volume": rng.integers(1000, 10000, rows),
    })


def test_signal_registry_contains_exactly_sixty_unique_signals():
    assert len(SIGNAL_NAMES) == len(set(SIGNAL_NAMES)) == 60
    assert {"rsi_14", "stochastic_k_14", "sma_cross_5_20", "donchian_position_20"} <= set(SIGNAL_NAMES)


def test_all_signals_are_causal():
    bars = _bars()
    baseline = compute_signal_library(bars)
    changed = bars.copy()
    changed.loc[130:, ["open", "high", "low", "close", "volume"]] *= 10
    altered = compute_signal_library(changed)
    pd.testing.assert_frame_equal(baseline.iloc[:130], altered.iloc[:130])


def test_semantic_signals_use_natural_states_and_remain_causal():
    bars = _bars()
    semantic = compute_semantic_signal_library(bars)
    assert semantic.shape[1] == 60
    assert set(np.unique(semantic.to_numpy())) <= {-1.0, 0.0, 1.0}
    assert {signal_category(name) for name in SIGNAL_NAMES} == {
        "trend", "momentum", "volatility", "volume", "structure",
    }
    assert Counter(signal_category(name) for name in SIGNAL_NAMES) == {
        "trend": 21,
        "momentum": 21,
        "volatility": 6,
        "volume": 6,
        "structure": 6,
    }


def test_parameterized_indicators_compute_arbitrary_values_and_remain_causal():
    bars = _bars()
    variants = (
        IndicatorVariant("sma", 37),
        IndicatorVariant("rsi", 11),
        IndicatorVariant("macd", 26),
        IndicatorVariant("directional_volume", 9),
    )
    baseline = compute_parameterized_signal_library(bars, variants)
    assert list(baseline) == [
        "sma_distance_37", "rsi_11", "macd_13_26", "directional_volume_z_9",
    ]
    changed = bars.copy()
    changed.loc[130:, ["open", "high", "low", "close", "volume"]] *= 10
    altered = compute_parameterized_signal_library(changed, variants)
    pd.testing.assert_frame_equal(baseline.iloc[:130], altered.iloc[:130])
    semantic = compute_parameterized_semantic_signal_library(bars, variants)
    assert set(np.unique(semantic.to_numpy())) <= {-1.0, 0.0, 1.0}


def test_indicator_value_grid_has_one_value_parameter_per_indicator():
    variants = build_indicator_variants(value_min=3, value_max=50, value_step=1)
    assert len(variants) == len(PARAMETRIC_INDICATORS) * 48
    assert len({variant.key for variant in variants}) == len(variants)
