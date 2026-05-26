"""
Tests for the ingest_alpaca orchestrator using a mocked AlpacaClient.

Live Alpaca calls are covered by the integration test (test_catalog_integration.py)
and the cross-check script (scripts/cross_check_yfinance.py).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from data.ingest_alpaca import ingest, latest_ingested_date


@pytest.fixture
def fake_client(sample_alpaca_bars_df):
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[str, ...], datetime, datetime]] = []

        def fetch_daily_bars(self, tickers, start, end):
            self.calls.append((tuple(tickers), start, end))
            frames = []
            for t in tickers:
                df = sample_alpaca_bars_df.copy()
                df.index = df.index.set_levels([t], level=0)
                frames.append(df)
            return pd.concat(frames) if frames else pd.DataFrame()

    return FakeClient()


def test_ingest_writes_bars_to_fresh_catalog(tmp_path: Path, fake_client) -> None:
    catalog_path = tmp_path / "catalog"
    ingest(
        client=fake_client,
        tickers=("SPY",),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        catalog_path=catalog_path,
    )
    assert catalog_path.exists()
    assert any(catalog_path.rglob("*.parquet")), "no parquet files written"
    assert len(fake_client.calls) == 1


def test_ingest_is_idempotent_skips_already_ingested(tmp_path: Path, fake_client) -> None:
    catalog_path = tmp_path / "catalog"
    args = dict(
        client=fake_client,
        tickers=("SPY",),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        catalog_path=catalog_path,
    )
    ingest(**args)
    ingest(**args)
    assert len(fake_client.calls) == 1 or fake_client.calls[1][1] > fake_client.calls[1][2]


def test_latest_ingested_date_none_when_empty(tmp_path: Path) -> None:
    assert latest_ingested_date(tmp_path / "catalog", "SPY") is None


def test_latest_ingested_date_after_ingest(tmp_path: Path, fake_client) -> None:
    catalog_path = tmp_path / "catalog"
    ingest(
        client=fake_client,
        tickers=("SPY",),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        catalog_path=catalog_path,
    )
    latest = latest_ingested_date(catalog_path, "SPY")
    assert latest is not None
    assert latest.date() == datetime(2024, 1, 4).date()


def test_ingest_partial_overlap_refetch_does_not_raise_or_duplicate(tmp_path, sample_alpaca_bars_df):
    """A re-fetch returning some already-stored days plus a new day must not
    crash write_data (non-disjoint intervals) and must not duplicate bars."""
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

    # Phase-2 frame: Jan 3,4,5 (Jan 3,4 overlap the stored Jan 2,3,4; Jan 5 is new).
    overlap_df = sample_alpaca_bars_df.copy()
    overlap_df.index = pd.MultiIndex.from_tuples(
        [
            ("SPY", pd.Timestamp("2024-01-03 05:00:00+00:00")),
            ("SPY", pd.Timestamp("2024-01-04 05:00:00+00:00")),
            ("SPY", pd.Timestamp("2024-01-05 05:00:00+00:00")),
        ],
        names=["symbol", "timestamp"],
    )

    class TwoPhaseClient:
        def __init__(self):
            self.phase = 0

        def fetch_daily_bars(self, tickers, start, end):
            self.phase += 1
            src = sample_alpaca_bars_df if self.phase == 1 else overlap_df
            frames = []
            for t in tickers:
                df = src.copy()
                df.index = df.index.set_levels([t], level=0)
                frames.append(df)
            return pd.concat(frames)

    client = TwoPhaseClient()
    catalog_path = tmp_path / "catalog"

    # Phase 1: store Jan 2,3,4.
    ingest(
        client=client,
        tickers=("SPY",),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 4, 12, tzinfo=timezone.utc),
        catalog_path=catalog_path,
    )

    # Phase 2: client returns Jan 3,4,5 (partial overlap). Must not raise or duplicate.
    ingest(
        client=client,
        tickers=("SPY",),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        catalog_path=catalog_path,
    )

    cat = ParquetDataCatalog(path=str(catalog_path.resolve()))
    bars = cat.bars(instrument_ids=["SPY.ARCX"])
    ts = [b.ts_event for b in bars]
    assert len(ts) == len(set(ts)), "no duplicate ts_event allowed"
    assert len(ts) == 4, f"expected 4 unique bars (Jan 2,3,4,5), got {len(ts)}"
