"""Stage-4.5 real-catalog parity regression.

Regression gate for the parity gap diagnosed and fixed in Stage-4.5:

- Cause 1: ``LiveRiskEngineConfig.max_order_submit_rate`` defaulted to
  "100/00:00:01" which throttled ~2/3 of orders under fast-replay
  (wallclock LiveClock).
- Cause 2: livesim filled at next-bar OPEN; BacktestEngine fills at
  same-bar CLOSE (Nautilus daily-bar SimulatedExchange semantic). The
  overnight gap drove ~5% systematic drift.

With both fixes in place, the TradingNode fast-replay equity should match
BacktestEngine equity within 2% over the real ETF catalog. Skipped if the
catalog is absent.

To keep test runtime reasonable, the parity test runs over a 1-year
sub-window of the full ~5.5-year catalog (enough to trigger multiple
rebalances and rotations).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import Venue

from analysis.runner import run as backtest_run
from analysis.runspec import RunSpec
from data.instruments import VENUE_MIC, build_universe_instruments
from data.tickers import ALL_TICKERS, BENCHMARK, UNIVERSE
from livesim.factories import (
    CatalogDataClientFactory,
    SimulatedExecClientFactory,
    get_last_data_client,
    get_last_exec_client,
    reset_client_registry,
)
from livesim.node_config import (
    ExecClientParams,
    build_trading_node_config,
    register_bar_provider,
    register_exec_client_params,
    reset_registries,
)
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy


# Window: 1 year of the real catalog (2021-01-01 → 2022-01-01).
# Catalog covers ~2020-07-27 → ~2026-05; this window has enough warm-up data
# behind it (sector ETFs start 2020-07-27) for the strategy's 63-day lookback
# to be valid from the start of the window. 1 year gives ~12 monthly
# rebalances — enough to surface cross-ticker rotation behaviour.
_WINDOW_START = datetime(2021, 1, 1, tzinfo=timezone.utc)
_WINDOW_END = datetime(2022, 1, 1, tzinfo=timezone.utc)

_CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog"

# Same Stage-3 winner combo as examples/run_trading_node.py uses.
_STRATEGY_KWARGS = {
    "universe": UNIVERSE,
    "benchmark": BENCHMARK,
    "top_n": 3,
    "lookback_days": 63,
    "skip_days": 0,
}

# Parity tolerance for the Stage-4.5 regression. Post-fix drift should be
# well under 1%; gating at 2% leaves headroom for minor day-counting
# differences while still catching any reintroduction of the throttle or
# fill-semantic bugs.
_DRIFT_TOL = 0.02


def _catalog_available() -> bool:
    """Return True if the catalog has bar parquet files for every ticker."""
    bar_dir = _CATALOG_PATH / "data" / "bar"
    if not bar_dir.exists():
        return False
    for t in ALL_TICKERS:
        if not (bar_dir / f"{t}.{VENUE_MIC}-1-DAY-LAST-EXTERNAL").exists():
            return False
    return True


def _load_catalog_bars():
    """Load all bars per ticker from the real Parquet catalog."""
    # Local import so importing the test module doesn't drag in the catalog
    # reader when the catalog is absent.
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

    catalog = ParquetDataCatalog(path=str(_CATALOG_PATH.resolve()))
    bars_by_ticker: dict[str, list] = {}
    for ticker in ALL_TICKERS:
        bars = catalog.bars(instrument_ids=[f"{ticker}.{VENUE_MIC}"])
        bars_by_ticker[ticker] = bars
    return bars_by_ticker


@pytest.mark.asyncio
async def test_real_catalog_parity_within_2pct():
    """TradingNode fast-replay equity must match BacktestEngine within 2%
    over a 1-year real-catalog window (Stage-4.5 regression gate)."""
    if not _catalog_available():
        pytest.skip(
            "real catalog absent — run data/ingest_alpaca.py to populate "
            f"{_CATALOG_PATH}"
        )

    reset_registries()
    reset_client_registry()
    bars_by_ticker = _load_catalog_bars()
    instruments = build_universe_instruments()

    # 1. BacktestEngine reference run (slippage=0 for deterministic parity).
    backtest_spec = RunSpec(
        label="parity-bt",
        strategy_class=MomentumTopNStrategy,
        strategy_config=MomentumTopNConfig(**_STRATEGY_KWARGS),
        start=_WINDOW_START,
        end=_WINDOW_END,
        slippage_prob=0.0,
    )
    backtest_result = backtest_run(
        backtest_spec, bars_by_ticker=bars_by_ticker, instruments=instruments,
    )
    backtest_equity = backtest_result.metrics["final_equity"]
    assert backtest_equity > 0, "BacktestEngine produced zero equity"

    # 2. TradingNode fast-replay path.
    # Filter bars to the same window so the data client sees identical data.
    start_ns = int(_WINDOW_START.timestamp() * 1e9)
    end_ns = int(_WINDOW_END.timestamp() * 1e9)
    windowed_bars = {
        t: [b for b in bars if start_ns <= b.ts_event < end_ns]
        for t, bars in bars_by_ticker.items()
    }

    register_bar_provider(VENUE_MIC, lambda: windowed_bars)
    register_exec_client_params(VENUE_MIC, ExecClientParams(slippage_prob=0.0))

    config = build_trading_node_config(
        strategy_path="strategies.momentum_topn:MomentumTopNStrategy",
        strategy_config_path="strategies.momentum_topn:MomentumTopNConfig",
        strategy_config_kwargs=dict(_STRATEGY_KWARGS),
    )

    node = TradingNode(config=config)
    node.add_data_client_factory(VENUE_MIC, CatalogDataClientFactory)
    node.add_exec_client_factory(VENUE_MIC, SimulatedExecClientFactory)
    node.build()

    for instrument in instruments:
        node.kernel.cache.add_instrument(instrument)

    diagnostics = ""
    live_equity: float | None = None
    node_run_task = None
    try:
        node_run_task = asyncio.create_task(node.run_async())

        data_client = get_last_data_client(VENUE_MIC)
        try:
            await asyncio.wait_for(data_client.done(), timeout=120.0)
        except asyncio.TimeoutError:
            pytest.fail("TradingNode fast-replay did not complete within 120s")

        # Settle for trailing fills/equity updates.
        await asyncio.sleep(1.0)

        venue = Venue(VENUE_MIC)
        account = node.kernel.portfolio.account(venue)
        live_equity_money = node.kernel.portfolio.equity(venue)[account.base_currency]
        live_equity = float(live_equity_money.as_decimal())

        from nautilus_trader.model.enums import OrderStatus
        exec_client = get_last_exec_client(VENUE_MIC)
        all_orders = list(node.kernel.cache.orders())
        n_filled = sum(1 for o in all_orders if o.status == OrderStatus.FILLED)
        n_denied = sum(1 for o in all_orders if o.status == OrderStatus.DENIED)
        n_pending = sum(len(v) for v in exec_client._pending.values())
        diagnostics = (
            f"orders={len(all_orders)} filled={n_filled} denied={n_denied} "
            f"pending={n_pending} "
            f"exec_bars_received={exec_client._bars_received} "
            f"exec_fills_emitted={exec_client._fills_emitted} "
            f"backtest_trades={backtest_result.metrics.get('n_trades')}"
        )
        assert n_filled > 0, f"No orders filled on TradingNode path — {diagnostics}"
        assert n_denied == 0, (
            f"TradingNode denied {n_denied} orders — Stage-4.5 throttle fix "
            f"regressed. {diagnostics}"
        )
        # Cause-3 regression gate: if msgbus subscriber order ever flips so
        # the exec client's _on_bar fires AFTER strategy.on_bar, orders would
        # sit in _pending instead of buffering on submit, and equity would
        # silently degrade by one bar's gap. This catches it directly.
        assert n_pending == 0, (
            f"TradingNode left {n_pending} orders in _pending — Stage-4.5 "
            f"submit-time buffering regressed (likely msgbus subscriber-order "
            f"flip). {diagnostics}"
        )
    finally:
        try:
            await node.stop_async()
        except Exception:
            pass
        if node_run_task is not None:
            node_run_task.cancel()
            try:
                await node_run_task
            except (asyncio.CancelledError, Exception):
                pass

    assert live_equity is not None
    drift = abs(live_equity - backtest_equity) / backtest_equity
    print(
        f"\nBacktestEngine final_equity = {backtest_equity:.2f}\n"
        f"TradingNode    final_equity = {live_equity:.2f}\n"
        f"drift                       = {drift:.6f}\n"
        f"diag                        = {diagnostics}"
    )
    assert drift < _DRIFT_TOL, (
        f"Real-catalog parity drift {drift:.4f} exceeds {_DRIFT_TOL:.2f} "
        f"gate. BT={backtest_equity:.2f}, ND={live_equity:.2f}. "
        f"{diagnostics}"
    )
