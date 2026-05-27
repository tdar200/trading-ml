"""Tests for data.ingest_alpaca_intraday pure helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.ingest_alpaca_intraday import parse_timeframe, shard_path


def test_parse_timeframe_maps_canonical_strings():
    """1m / 5m / 15m / 30m / 1h / 4h all parse without raising."""
    for tf in ("1m", "5m", "15m", "30m", "1h", "4h"):
        # Just needs to not raise — the returned object is an Alpaca TimeFrame.
        result = parse_timeframe(tf)
        assert result is not None


def test_parse_timeframe_rejects_unknown():
    for bad in ("1d", "1w", "1M", "tick", "10m", ""):
        with pytest.raises(ValueError, match="unknown intraday timeframe"):
            parse_timeframe(bad)


def test_shard_path_includes_timeframe(tmp_path: Path):
    """Different timeframes for the same ticker land in different files."""
    p_5m = shard_path(tmp_path, "SPY", "5m", pd.Timestamp("2024-03-15T10:00:00Z"))
    p_1h = shard_path(tmp_path, "SPY", "1h", pd.Timestamp("2024-03-15T10:00:00Z"))
    assert p_5m == tmp_path / "SPY" / "5m" / "2024-03.parquet"
    assert p_1h == tmp_path / "SPY" / "1h" / "2024-03.parquet"
    assert p_5m != p_1h


def test_shard_path_normalizes_naive_timestamp(tmp_path: Path):
    """Naive timestamps are treated as UTC (matches Stage 1's pattern)."""
    p = shard_path(tmp_path, "XLK", "15m", pd.Timestamp("2024-12-01"))
    assert p == tmp_path / "XLK" / "15m" / "2024-12.parquet"


def test_shard_path_rejects_unknown_timeframe(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown intraday timeframe"):
        shard_path(tmp_path, "SPY", "1d", pd.Timestamp("2024-03-15", tz="UTC"))
