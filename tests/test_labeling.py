"""Tests for models.labeling — pure pandas, no Nautilus."""
from __future__ import annotations

import pandas as pd

from models.labeling import make_labels


def _close_series(start_ts: str, prices: list[float]) -> pd.Series:
    return pd.Series(prices, index=pd.date_range(start_ts, periods=len(prices), freq="B"))


def test_positive_return_labels_one():
    """Price rises over the holding period → label=1."""
    closes = _close_series("2024-01-02", [100.0] + [100.0 + i * 0.1 for i in range(1, 30)])
    bars_by_ticker = {"A": closes}
    rebalance_dates = pd.DatetimeIndex(["2024-01-02"])
    df = make_labels(bars_by_ticker, rebalance_dates, horizon_days=21)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "A"
    assert df.iloc[0]["label"] == 1


def test_negative_return_labels_zero():
    """Price falls over the holding period → label=0."""
    closes = _close_series("2024-01-02", [100.0] + [100.0 - i * 0.1 for i in range(1, 30)])
    df = make_labels({"A": closes}, pd.DatetimeIndex(["2024-01-02"]), horizon_days=21)
    assert df.iloc[0]["label"] == 0


def test_zero_return_labels_zero():
    """Flat price → label=0 (not strictly positive)."""
    closes = _close_series("2024-01-02", [100.0] * 30)
    df = make_labels({"A": closes}, pd.DatetimeIndex(["2024-01-02"]), horizon_days=21)
    assert df.iloc[0]["label"] == 0


def test_insufficient_forward_data_dropped():
    """Rebalance dates near the end of the series are dropped (no forward bar)."""
    closes = _close_series("2024-01-02", [100.0] * 10)  # only 10 days
    df = make_labels({"A": closes}, pd.DatetimeIndex(["2024-01-02"]), horizon_days=21)
    assert df.empty


def test_multi_ticker_one_row_per_ticker_per_date():
    """Multiple tickers + multiple dates → cross-product of valid rows."""
    a = _close_series("2024-01-02", [100.0] + [100.0 + i * 0.1 for i in range(1, 60)])
    b = _close_series("2024-01-02", [100.0] + [100.0 - i * 0.05 for i in range(1, 60)])
    dates = pd.DatetimeIndex(["2024-01-02", "2024-02-01"])
    df = make_labels({"A": a, "B": b}, dates, horizon_days=21)
    assert len(df) == 4  # 2 tickers × 2 dates
    assert set(df["ticker"]) == {"A", "B"}
    assert set(df["date"]) == {pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-01")}
