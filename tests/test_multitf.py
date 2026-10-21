"""Tests for data.multitf — multi-timeframe loader."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.multitf import (
    INTRADAY_TIMEFRAMES,
    RESAMPLE_TIMEFRAMES,
    SUPPORTED_TIMEFRAMES,
    get_bars,
    storage_symbol,
    _resample_ohlcv,
    _slice_window,
    _empty_bars,
)


def _write_intraday_shard(root: Path, ticker: str, tf: str, df: pd.DataFrame) -> Path:
    """Helper: write a per-(ticker, tf, month) parquet shard with given df."""
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    month = df["ts"].iloc[0].strftime("%Y-%m")
    path = root / ticker / tf / f"{month}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["ticker", "ts", "open", "high", "low", "close", "volume"]].to_parquet(path, index=False)
    return path


def test_supported_timeframes_includes_all_branches():
    assert "1m" in SUPPORTED_TIMEFRAMES
    assert "1d" in SUPPORTED_TIMEFRAMES
    assert "1w" in SUPPORTED_TIMEFRAMES
    assert "1Y" in SUPPORTED_TIMEFRAMES


def test_get_bars_unknown_timeframe_raises():
    with pytest.raises(ValueError, match="unknown timeframe"):
        get_bars("SPY", "tick")  # tick is live-only, not in the historical loader


def test_get_bars_intraday_loads_from_shard(tmp_path: Path):
    intra = tmp_path / "intraday"
    df = pd.DataFrame({
        "ticker": ["SPY"] * 3,
        "ts": pd.to_datetime([
            "2024-03-15T14:30:00Z", "2024-03-15T14:35:00Z", "2024-03-15T14:40:00Z",
        ], utc=True),
        "open": [510.0, 511.0, 512.0],
        "high": [511.0, 512.0, 513.0],
        "low": [509.5, 510.5, 511.5],
        "close": [511.0, 512.0, 512.5],
        "volume": [1000, 2000, 1500],
    })
    _write_intraday_shard(intra, "SPY", "5m", df)
    result = get_bars("SPY", "5m", intraday_root=intra, catalog_path=tmp_path / "nonexistent")
    assert len(result) == 3
    assert list(result.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert result["close"].tolist() == [511.0, 512.0, 512.5]


def test_get_bars_loads_crypto_pair_from_filesystem_safe_path(tmp_path: Path):
    intra = tmp_path / "intraday"
    df = pd.DataFrame({
        "ticker": ["BTC/USD"],
        "ts": pd.to_datetime(["2024-03-15T14:30:00Z"], utc=True),
        "open": [68000.0], "high": [68100.0], "low": [67900.0],
        "close": [68050.0], "volume": [2.0],
    })
    _write_intraday_shard(intra, "BTC-USD", "5m", df)
    result = get_bars(
        "BTC/USD", "5m", intraday_root=intra,
        catalog_path=tmp_path / "nonexistent",
    )
    assert storage_symbol("BTC/USD") == "BTC-USD"
    assert result["close"].tolist() == [68050.0]


def test_get_bars_intraday_returns_empty_when_no_data(tmp_path: Path):
    result = get_bars("XYZ", "15m",
                      intraday_root=tmp_path / "no_intraday",
                      catalog_path=tmp_path / "no_catalog")
    assert result.empty
    assert list(result.columns) == ["ts", "open", "high", "low", "close", "volume"]


def test_resample_ohlcv_weekly_aggregates_correctly():
    """5 daily bars Mon-Fri → 1 weekly bar with OHLC aggregation."""
    df = pd.DataFrame({
        "ts": pd.to_datetime([
            "2024-03-04",  # Mon
            "2024-03-05",  # Tue
            "2024-03-06",  # Wed
            "2024-03-07",  # Thu
            "2024-03-08",  # Fri
        ], utc=True),
        "open":   [100.0, 101.0, 102.0, 103.0, 104.0],
        "high":   [105.0, 106.0, 107.0, 108.0, 109.0],
        "low":    [ 99.0, 100.0, 101.0, 102.0, 103.0],
        "close":  [101.0, 102.0, 103.0, 104.0, 105.0],
        "volume": [1000, 2000, 3000, 4000, 5000],
    })
    weekly = _resample_ohlcv(df, "W")
    assert len(weekly) == 1
    assert weekly["open"].iloc[0] == 100.0   # first
    assert weekly["high"].iloc[0] == 109.0   # max
    assert weekly["low"].iloc[0] == 99.0     # min
    assert weekly["close"].iloc[0] == 105.0  # last
    assert weekly["volume"].iloc[0] == 15000  # sum


def test_slice_window_filters_inclusive_exclusive():
    df = pd.DataFrame({
        "ts": pd.to_datetime([
            "2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15",
        ], utc=True),
        "open": [1.0, 2.0, 3.0, 4.0],
        "high": [1.0, 2.0, 3.0, 4.0],
        "low": [1.0, 2.0, 3.0, 4.0],
        "close": [1.0, 2.0, 3.0, 4.0],
        "volume": [1, 2, 3, 4],
    })
    # [Feb 1, Feb 15) should include only Feb 1 (Feb 15 is the exclusive end).
    result = _slice_window(df,
                           start=pd.Timestamp("2024-02-01", tz="UTC"),
                           end=pd.Timestamp("2024-02-15", tz="UTC"))
    assert len(result) == 1
    assert result["close"].iloc[0] == 3.0
