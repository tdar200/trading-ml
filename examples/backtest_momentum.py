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


def _equity_curve_from_engine(engine: BacktestEngine, venue) -> pd.Series:
    """Reconstruct the strategy's equity curve from the account-event log.

    Per docs/nautilus_api_notes.md §9.7: NautilusTrader v1.227 has no built-in
    time-series equity API, but `account.events` gives one AccountState per
    balance change (initial funding + every fill). Each event has ts_event
    (int ns) and balances (list of AccountBalance with .total Money). The
    verified attribute path (§9.7 Option A) is `b.currency == account.base_currency`,
    then `b.total.as_decimal()`.
    """
    account = engine.portfolio.account(venue)
    events = list(account.events)
    if not events:
        # Fallback: just a single end-of-backtest point.
        final = account.balance_total(account.base_currency)
        return pd.Series(
            {pd.Timestamp.utcnow().tz_localize(None).normalize(): float(final.as_decimal())},
            name="equity",
        )

    rows: list[tuple[pd.Timestamp, float]] = []
    for ev in events:
        ts = pd.Timestamp(ev.ts_event, unit="ns", tz="UTC").tz_convert(None).normalize()
        # `balances` is list[AccountBalance]; pick the entry in the account's base currency.
        total = None
        for bal in ev.balances:
            if bal.currency == account.base_currency:
                total = float(bal.total.as_decimal())
                break
        if total is None:
            continue
        rows.append((ts, total))
    if not rows:
        return pd.Series(name="equity", dtype=float)

    # Multiple events on the same day collapse to the LAST value for that day.
    df = pd.DataFrame(rows, columns=["date", "equity"]).groupby("date").last()
    return df["equity"].sort_index()


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
    engine.add_strategy(MomentumTopNStrategy(config=MomentumTopNConfig(
        universe=UNIVERSE,
        benchmark=BENCHMARK,
    )))

    # 5. Run.
    print("running backtest...")
    engine.run()
    print("backtest complete")

    # 6. Equity curve + SPY close series → tearsheet.
    equity = _equity_curve_from_engine(engine, venue)
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
