"""
Orchestrate ingestion: fetch from Alpaca → convert → write to Parquet catalog.

Idempotency strategy: before fetching, ask the catalog for the latest bar
timestamp we already have for each ticker. Only fetch (latest + 1 day) → end.
If the catalog already has data through `end`, the ticker is skipped entirely.

CLI entry point fetches the full universe from 2016-01-01 → today and writes to
`data/catalog/`.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

from data.bar_converter import alpaca_df_to_bars
from data.instruments import build_universe_instruments
from data.tickers import ALL_TICKERS

DEFAULT_CATALOG_PATH = Path("data/catalog")
DEFAULT_START = datetime(2016, 1, 1, tzinfo=timezone.utc)


class AlpacaLike(Protocol):
    """Minimal interface the ingester needs — keeps tests mock-friendly."""

    def fetch_daily_bars(
        self, tickers: Iterable[str], start: datetime, end: datetime
    ) -> pd.DataFrame: ...


def _catalog(path: Path) -> ParquetDataCatalog:
    path.mkdir(parents=True, exist_ok=True)
    # §7: ParquetDataCatalog needs an absolute path that exists.
    return ParquetDataCatalog(path=str(path.resolve()))


def latest_ingested_date(catalog_path: Path, ticker: str) -> datetime | None:
    """Return the timestamp of the last bar in the catalog for `ticker`,
    or None if no data exists for that ticker yet."""
    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        return None
    catalog = _catalog(catalog_path)
    instrument_id = f"{ticker}.ARCX"
    try:
        bars = catalog.bars(instrument_ids=[instrument_id])
    except Exception:
        return None
    if not bars:
        return None
    last_ts_ns = max(b.ts_event for b in bars)
    return datetime.fromtimestamp(last_ts_ns / 1e9, tz=timezone.utc)


def ingest(
    client: AlpacaLike,
    tickers: Iterable[str],
    start: datetime,
    end: datetime,
    catalog_path: Path,
) -> None:
    """Fetch and write bars for `tickers` over [start, end]; idempotent."""
    catalog_path = Path(catalog_path)
    catalog_path.mkdir(parents=True, exist_ok=True)
    catalog = _catalog(catalog_path)

    tickers = tuple(tickers)

    effective_start = start
    for t in tickers:
        latest = latest_ingested_date(catalog_path, t)
        if latest is not None and latest + timedelta(days=1) > effective_start:
            effective_start = max(effective_start, latest + timedelta(days=1))

    if effective_start > end:
        return

    df = client.fetch_daily_bars(tickers, effective_start, end)
    if df.empty:
        return

    all_bars = []
    for t in tickers:
        all_bars.extend(alpaca_df_to_bars(df, t))
    if all_bars:
        catalog.write_data(all_bars)


def main() -> None:
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        description="Ingest Alpaca daily bars into Nautilus catalog"
    )
    parser.add_argument("--start", default=DEFAULT_START.isoformat())
    parser.add_argument("--end", default=datetime.now(timezone.utc).isoformat())
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH))
    args = parser.parse_args()

    load_dotenv()
    from data.alpaca_client import AlpacaClient

    client = AlpacaClient.from_env()
    catalog_path = Path(args.catalog)

    _catalog(catalog_path).write_data(build_universe_instruments())

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    ingest(
        client=client,
        tickers=ALL_TICKERS,
        start=start,
        end=end,
        catalog_path=catalog_path,
    )
    print(f"ingest complete → {catalog_path}")


if __name__ == "__main__":
    main()
