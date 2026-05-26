"""Shared pytest fixtures for trading-ml tests."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_alpaca_bars_df() -> pd.DataFrame:
    """
    Three days of daily bars for SPY in the shape alpaca-py returns when
    .df is called on a StockBarsRequest result (MultiIndex on symbol+timestamp).
    """
    idx = pd.MultiIndex.from_tuples(
        [
            ("SPY", pd.Timestamp("2024-01-02 05:00:00+00:00")),
            ("SPY", pd.Timestamp("2024-01-03 05:00:00+00:00")),
            ("SPY", pd.Timestamp("2024-01-04 05:00:00+00:00")),
        ],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {
            "open":   [470.50, 471.20, 469.10],
            "high":   [472.00, 472.10, 470.80],
            "low":    [469.80, 470.30, 467.50],
            "close":  [471.10, 470.40, 468.90],
            "volume": [50_000_000, 48_000_000, 52_000_000],
            "trade_count": [400_000, 380_000, 410_000],
            "vwap":   [471.00, 471.10, 469.20],
        },
        index=idx,
    )
