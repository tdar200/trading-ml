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
