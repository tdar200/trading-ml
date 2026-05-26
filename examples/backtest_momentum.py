"""
Stage-2 acceptance backtest: MomentumTopN over the full Stage-1 catalog.

Run:
    uv run python examples/backtest_momentum.py

Loads daily bars for the 12 tickers from data/catalog/, runs the engine with
the project's default backtest venue (~0.5 bps stochastic slippage, $100k cash),
writes equity curve + SPY close series to results/momentum_topn_<timestamp>/,
and produces a quantstats tearsheet (tearsheet.html) comparing the strategy
to SPY buy-and-hold.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Make project root importable when running as a script (mirrors smoke.py).
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

from analysis.tearsheet import report as tearsheet_report
from data.instruments import VENUE_MIC, build_universe_instruments
from data.tickers import ALL_TICKERS, BENCHMARK, UNIVERSE
from strategies.backtest_setup import add_default_backtest_venue
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy


CATALOG_PATH = _PROJECT_ROOT / "data" / "catalog"
RESULTS_ROOT = _PROJECT_ROOT / "results"


def _load_bars_for(catalog: ParquetDataCatalog, ticker: str):
    return catalog.bars(instrument_ids=[f"{ticker}.{VENUE_MIC}"])


def _spy_close_series(catalog: ParquetDataCatalog) -> pd.Series:
    """Build SPY's daily close series, tz-naive normalized DatetimeIndex."""
    bars = _load_bars_for(catalog, BENCHMARK)
    rows = [
        (pd.Timestamp(b.ts_event, unit="ns", tz="UTC").tz_convert(None).normalize(),
         float(b.close))
        for b in bars
    ]
    s = pd.Series(dict(rows), name="spy_close").sort_index()
    return s


def _equity_curve_from_engine(engine: BacktestEngine, venue, strategy) -> pd.Series:
    """Equity time series — strategy's per-day mark-to-market snapshot.

    The Strategy snapshots `portfolio.equity(venue)` once per benchmark bar (one
    per calendar trading day), which gives a proper daily mark-to-market equity
    curve. Falls back to account.events (cash-only — only meaningful pre-first-rebalance)
    if the strategy's history is somehow empty.
    """
    if strategy._equity_history:
        rows = [
            (pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert(None).normalize(), eq)
            for ts_ns, eq in strategy._equity_history
        ]
        df = pd.DataFrame(rows, columns=["date", "equity"]).groupby("date").last()
        return df["equity"].sort_index()

    # Fallback (shouldn't happen if benchmark bars are subscribed).
    account = engine.portfolio.account(venue)
    final = account.balance_total(account.base_currency)
    return pd.Series(
        {pd.Timestamp.utcnow().tz_localize(None).normalize(): float(final.as_decimal())},
        name="equity",
    )


def main() -> int:
    # 1. Output directory.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS_ROOT / f"momentum_topn_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")

    # 2. Engine + venue.
    config = BacktestEngineConfig(
        trader_id=TraderId("STAGE2-001"),
        logging=LoggingConfig(log_level="WARNING"),
    )
    engine = BacktestEngine(config=config)
    venue = add_default_backtest_venue(engine, starting_cash=100_000)

    # 3. Instruments + bars from the catalog.
    catalog = ParquetDataCatalog(path=str(CATALOG_PATH.resolve()))
    for instrument in build_universe_instruments():
        engine.add_instrument(instrument)
    total_bars = 0
    for ticker in ALL_TICKERS:
        bars = _load_bars_for(catalog, ticker)
        engine.add_data(bars)
        total_bars += len(bars)
        print(f"  {ticker}: {len(bars)} bars")
    print(f"total bars loaded: {total_bars}")
    if total_bars == 0:
        print("ERROR: catalog is empty. Run data/ingest_alpaca.py first.")
        return 1

    # 4. Strategy.
    strategy = MomentumTopNStrategy(config=MomentumTopNConfig(
        universe=UNIVERSE,
        benchmark=BENCHMARK,
    ))
    engine.add_strategy(strategy)

    # 5. Run.
    print("running backtest...")
    engine.run()
    print("backtest complete")

    # 6. Equity curve + SPY close series → tearsheet.
    equity = _equity_curve_from_engine(engine, venue, strategy)
    spy = _spy_close_series(catalog)
    equity.to_csv(out_dir / "equity.csv", header=True)
    spy.to_csv(out_dir / "spy_close.csv", header=True)
    summary = tearsheet_report(equity_curve=equity, spy_closes=spy, out_dir=out_dir)

    # 7. Final account balance for sanity.
    account = engine.portfolio.account(venue)
    final = account.balance_total(account.base_currency)
    print(f"final balance: {final}")
    print(f"final equity (mark-to-market): {engine.portfolio.equity(venue)[account.base_currency]}")
    print(f"tearsheet: {out_dir / 'tearsheet.html'}")
    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
