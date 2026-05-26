"""Pure metric-function tests."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analysis.metrics import cagr, max_drawdown, sharpe


def test_cagr_empty_returns_nan():
    assert math.isnan(cagr(pd.Series([], dtype=float)))


def test_cagr_one_year_doubling():
    """1 year of daily returns that compound to 2× → CAGR = 1.0."""
    daily = pd.Series([(2.0) ** (1.0 / 252) - 1.0] * 252)
    assert cagr(daily) == pytest.approx(1.0, abs=1e-6)


def test_sharpe_zero_std_returns_nan():
    assert math.isnan(sharpe(pd.Series([0.001] * 100)))


def test_sharpe_constant_positive_returns_high():
    """Nearly-constant positive returns → very high Sharpe."""
    rng = np.random.default_rng(42)
    daily = pd.Series([0.001 + rng.normal(0, 1e-9) for _ in range(252)])
    s = sharpe(daily)
    assert math.isfinite(s)
    assert s > 100


def test_max_drawdown_monotone_growth():
    """Equity that only grows has max-DD = 0."""
    equity = pd.Series([100.0, 101.0, 102.0, 103.0])
    assert max_drawdown(equity) == pytest.approx(0.0)


def test_max_drawdown_50pct_peak_to_trough():
    """Equity peaks at 200, troughs at 100 → -50% drawdown."""
    equity = pd.Series([100.0, 200.0, 150.0, 100.0])
    assert max_drawdown(equity) == pytest.approx(-0.5)
