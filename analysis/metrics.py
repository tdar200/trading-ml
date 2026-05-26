"""
Pure metric functions: CAGR, Sharpe, max drawdown.

Used by analysis.tearsheet for individual-run summaries and by analysis.runner
for backtest result metrics. No IO; deterministic given inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate from a series of period returns."""
    if len(returns) == 0:
        return float("nan")
    total = (1.0 + returns).prod()
    years = len(returns) / periods_per_year
    if years <= 0 or total <= 0:
        return float("nan")
    return float(total ** (1.0 / years) - 1.0)


def sharpe(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio (ddof=0 for population std — consistent across the project)."""
    if len(returns) == 0 or np.isclose(returns.std(ddof=0), 0, atol=1e-14):
        return float("nan")
    excess = returns - rf / periods_per_year
    return float(excess.mean() / excess.std(ddof=0) * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (a negative number, or 0 for monotone growth)."""
    if len(equity) == 0:
        return float("nan")
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())
