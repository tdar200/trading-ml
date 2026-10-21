"""Tests for causal cross-sectional portfolio discovery."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.cross_sectional import (
    CrossSectionalGrammar,
    PARAM_FIELDS,
    combinatorial_probability_overfit,
    generate_cross_sectional_candidates,
    prepare_cross_sectional_panel,
    simulate_cross_sectional_candidates,
    split_panel_discovery_holdout,
)


def _bars(multiplier: float, sessions: int = 10, bars_per_session: int = 12) -> pd.DataFrame:
    rows = []
    for day, date in enumerate(pd.bdate_range("2024-01-02", periods=sessions)):
        start = pd.Timestamp(date.date(), tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
        for bar in range(bars_per_session):
            price = 100 + day + multiplier * bar
            rows.append({
                "ts": (start + pd.Timedelta(minutes=5 * bar)).tz_convert("UTC"),
                "open": price, "high": price + 0.1, "low": price - 0.1,
                "close": price + multiplier * 0.5,
                "volume": 1000 + day * 20 + bar * 5,
            })
    return pd.DataFrame(rows)


def _panel():
    stocks = {
        "FAST": _bars(0.30), "UP": _bars(0.15),
        "DOWN": _bars(-0.10), "SLOW": _bars(-0.20),
    }
    return prepare_cross_sectional_panel(stocks, _bars(0.02))


def test_candidate_grammar_expands_finite_space():
    candidates = generate_cross_sectional_candidates(CrossSectionalGrammar())
    assert len(candidates) == 972
    assert {row["signal_family"] for row in candidates} == {"momentum", "reversal"}


def test_panel_features_are_causal_before_future_mutation():
    stocks = {name: _bars(move) for name, move in {
        "A": 0.3, "B": 0.2, "C": -0.1, "D": -0.2,
    }.items()}
    original = prepare_cross_sectional_panel(stocks, _bars(0.02))
    changed = {name: frame.copy() for name, frame in stocks.items()}
    changed["A"].loc[changed["A"].index[-1], ["close", "volume"]] *= 100
    mutated = prepare_cross_sectional_panel(changed, _bars(0.02))
    np.testing.assert_allclose(original.beta[:-1], mutated.beta[:-1])
    np.testing.assert_allclose(
        original.relative_volume[:-1], mutated.relative_volume[:-1], equal_nan=True,
    )


def test_momentum_ranking_is_profitable_before_costs_and_costs_reduce_return():
    panel = _panel()
    candidate = {
        "signal_family": "momentum", "decision_bar": 6,
        "lookback_bars": 1, "hold_bars": 1,
        "top_n": 2, "min_dispersion_bps": 1.0,
        "min_relative_volume": 0.5, "max_market_move_bps": 100.0,
    }
    free = simulate_cross_sectional_candidates(
        panel, [candidate], cost_per_side_bps=0.0, device="cpu",
    ).iloc[0]
    costly = simulate_cross_sectional_candidates(
        panel, [candidate], cost_per_side_bps=5.0, device="cpu",
    ).iloc[0]
    assert free["total_return"] > 0
    assert costly["total_return"] < free["total_return"]
    assert free["portfolio_events"] > 0
    assert free["n_trades"] == free["portfolio_events"] * 4


def test_session_split_is_disjoint_and_keeps_holdout_sealed():
    windows, holdout = split_panel_discovery_holdout(
        _panel(), discovery_fraction=0.8, n_windows=4,
    )
    window_dates = [set(window.session_dates) for window in windows]
    assert all(left.isdisjoint(right) for i, left in enumerate(window_dates) for right in window_dates[i + 1:])
    discovery_dates = set().union(*window_dates)
    assert discovery_dates.isdisjoint(set(holdout.session_dates))


def test_cuda_and_cpu_match_when_cuda_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    candidate = {
        "signal_family": "reversal", "decision_bar": 6,
        "lookback_bars": 3, "hold_bars": 3,
        "top_n": 2, "min_dispersion_bps": 1.0,
        "min_relative_volume": 0.5, "max_market_move_bps": 100.0,
    }
    cpu = simulate_cross_sectional_candidates(
        _panel(), [candidate], cost_per_side_bps=1.0, device="cpu",
    ).iloc[0]
    cuda = simulate_cross_sectional_candidates(
        _panel(), [candidate], cost_per_side_bps=1.0, device="cuda",
    ).iloc[0]
    for field in ("total_return", "sharpe", "max_drawdown", "mean_abs_beta_exposure"):
        assert cuda[field] == pytest.approx(cpu[field], abs=1e-12)
    assert cuda["n_trades"] == cpu["n_trades"]


def test_combinatorial_overfit_detects_unstable_selection():
    rows = []
    for candidate, sharpes in enumerate(((4.0, 4.0, -4.0, -4.0), (0.5, 0.5, 0.5, 0.5))):
        params = {
            "signal_family": "momentum", "decision_bar": 6,
            "lookback_bars": 3, "hold_bars": 3, "top_n": 2,
            "min_dispersion_bps": 10.0 + candidate,
            "min_relative_volume": 1.0, "max_market_move_bps": 50.0,
        }
        rows.extend({**params, "window_idx": window, "sharpe": value} for window, value in enumerate(sharpes))
    result = combinatorial_probability_overfit(pd.DataFrame(rows), PARAM_FIELDS)
    assert result["split_count"] == 6
    assert 0.0 <= result["probability_overfit"] <= 1.0
    assert result["probability_overfit"] > 0


def test_combinatorial_overfit_ignores_ineligible_nan_candidate():
    rows = []
    for candidate, sharpes in enumerate(((1.0, 1.0, 1.0, 1.0), (0.5, 0.5, 0.5, 0.5), (float("nan"),) * 4)):
        params = {
            "signal_family": "momentum", "decision_bar": 6,
            "lookback_bars": 3, "hold_bars": 3, "top_n": 2,
            "min_dispersion_bps": 10.0 + candidate,
            "min_relative_volume": 1.0, "max_market_move_bps": 50.0,
        }
        rows.extend({**params, "window_idx": window, "sharpe": value} for window, value in enumerate(sharpes))
    result = combinatorial_probability_overfit(pd.DataFrame(rows), PARAM_FIELDS)
    assert result["split_count"] == 6
