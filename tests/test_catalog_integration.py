"""
Integration test: ingest a small real slice from Alpaca, replay through
BacktestEngine. Skipped automatically when credentials are absent so CI on
machines without keys still passes.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from dotenv import load_dotenv

from data.alpaca_client import AlpacaClient
from data.ingest_alpaca import ingest, latest_ingested_date

load_dotenv()
needs_keys = pytest.mark.skipif(
    not (os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_API_SECRET")),
    reason="Alpaca credentials not in environment (.env)",
)


@needs_keys
def test_real_ingest_one_ticker_60_days(tmp_path: Path) -> None:
    client = AlpacaClient.from_env()
    catalog_path = tmp_path / "catalog"
    end = datetime(2024, 6, 1, tzinfo=timezone.utc)
    start = datetime(2024, 4, 1, tzinfo=timezone.utc)

    ingest(
        client=client,
        tickers=("SPY",),
        start=start,
        end=end,
        catalog_path=catalog_path,
    )

    latest = latest_ingested_date(catalog_path, "SPY")
    assert latest is not None
    assert start <= latest <= end


@needs_keys
def test_real_ingest_then_backtest_engine_replays(tmp_path: Path) -> None:
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.data import BarType
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import TraderId, Venue
    from nautilus_trader.model.objects import Money
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

    from data.instruments import build_equity

    client = AlpacaClient.from_env()
    catalog_path = tmp_path / "catalog"
    start = datetime(2024, 4, 1, tzinfo=timezone.utc)
    end = datetime(2024, 6, 1, tzinfo=timezone.utc)
    ingest(
        client=client,
        tickers=("SPY",),
        start=start,
        end=end,
        catalog_path=catalog_path,
    )

    catalog = ParquetDataCatalog(path=str(catalog_path))
    bars = catalog.bars(instrument_ids=["SPY.ARCX"])
    assert len(bars) > 35  # ≈42 trading days in Apr–May 2024

    ts_list = [b.ts_event for b in bars]
    assert ts_list == sorted(ts_list)
    for b in bars:
        assert b.open > 0
        assert b.high >= b.open
        assert b.low <= b.open
        assert b.close > 0
        assert int(b.volume) > 0

    config = BacktestEngineConfig(
        trader_id=TraderId("IT-001"),
        logging=LoggingConfig(log_level="WARNING"),
    )
    engine = BacktestEngine(config=config)
    engine.add_venue(
        venue=Venue("ARCX"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD,
        starting_balances=[Money(100_000, USD)],
    )
    engine.add_instrument(build_equity("SPY"))
    engine.add_data(bars)
    engine.run()
