"""End-to-end integration: TradingNode replays synthetic catalog through
MomentumTopNStrategy via livesim's data + exec clients. Final equity must
match `analysis.runner.run` (BacktestEngine) within 1% tolerance.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity

from analysis.runner import run as backtest_run
from analysis.runspec import RunSpec
from data.instruments import VENUE_MIC
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


# ---- synthetic data helpers (mirror analysis.runner / Stage 2/3) ----

def _instrument(ticker: str) -> Equity:
    return Equity(
        instrument_id=InstrumentId.from_str(f"{ticker}.{VENUE_MIC}"),
        raw_symbol=Symbol(ticker),
        currency=USD,
        price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
    )


def _bars(ticker: str, closes: list[float], start_ts_ns: int) -> list[Bar]:
    bar_type = BarType.from_str(f"{ticker}.{VENUE_MIC}-1-DAY-LAST-EXTERNAL")
    one_day_ns = 86_400_000_000_000
    out: list[Bar] = []
    for i, c in enumerate(closes):
        ts = start_ts_ns + i * one_day_ns
        p = Price.from_str(f"{c:.2f}")
        out.append(
            Bar(
                bar_type=bar_type,
                open=p,
                high=p,
                low=p,
                close=p,
                volume=Quantity.from_int(1_000_000),
                ts_event=ts,
                ts_init=ts,
            )
        )
    return out


def _grow(rate: float, n: int) -> list[float]:
    out = [100.0]
    for _ in range(n - 1):
        out.append(out[-1] * (1.0 + rate))
    return out


def _synthetic_data():
    n = 380
    start_ts_ns = 1_577_836_800_000_000_000  # 2020-01-01
    closes_by_ticker = {
        "A": _grow(0.0020, n),
        "B": _grow(0.0010, n),
        "BENCH": _grow(0.0005, n),
    }
    bars_by_ticker = {t: _bars(t, c, start_ts_ns) for t, c in closes_by_ticker.items()}
    instruments = [_instrument(t) for t in bars_by_ticker]
    return bars_by_ticker, instruments


@pytest.mark.asyncio
async def test_trading_node_equity_within_1pct_of_backtest():
    """TradingNode fast-replay equity must match BacktestEngine within 1%."""
    reset_registries()
    reset_client_registry()
    bars_by_ticker, instruments = _synthetic_data()

    # 1. Reference equity via BacktestEngine.
    backtest_spec = RunSpec(
        label="bt-baseline",
        strategy_class=MomentumTopNStrategy,
        strategy_config=MomentumTopNConfig(
            universe=("A", "B"),
            benchmark="BENCH",
            top_n=1,
            lookback_days=63,
            skip_days=21,
        ),
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2021, 1, 17, tzinfo=timezone.utc),
        slippage_prob=0.0,  # deterministic
    )
    backtest_result = backtest_run(
        backtest_spec, bars_by_ticker=bars_by_ticker, instruments=instruments,
    )
    backtest_equity = backtest_result.metrics["final_equity"]
    assert backtest_equity > 0, "BacktestEngine produced zero equity — fix Stage 2/3 first"

    # 2. TradingNode path.
    register_bar_provider(VENUE_MIC, lambda: bars_by_ticker)
    register_exec_client_params(VENUE_MIC, ExecClientParams(slippage_prob=0.0))  # deterministic

    config = build_trading_node_config(
        strategy_path="strategies.momentum_topn:MomentumTopNStrategy",
        strategy_config_path="strategies.momentum_topn:MomentumTopNConfig",
        strategy_config_kwargs={
            "universe": ("A", "B"),
            "benchmark": "BENCH",
            "top_n": 1,
            "lookback_days": 63,
            "skip_days": 21,
        },
    )

    node = TradingNode(config=config)
    node.add_data_client_factory(VENUE_MIC, CatalogDataClientFactory)
    node.add_exec_client_factory(VENUE_MIC, SimulatedExecClientFactory)
    node.build()

    # Register instruments in the cache so the strategy can resolve them.
    for instrument in instruments:
        node.kernel.cache.add_instrument(instrument)

    node_run_task = None
    try:
        # Start the node (async).
        node_run_task = asyncio.create_task(node.run_async())

        # Wait for the data client to signal "fast-replay done".
        data_client = get_last_data_client(VENUE_MIC)
        try:
            await asyncio.wait_for(data_client.done(), timeout=60.0)
        except asyncio.TimeoutError:
            pytest.fail("TradingNode fast-replay did not complete within 60 seconds")

        # Give the engine a moment to process trailing fills and equity updates.
        await asyncio.sleep(1.0)

        # 3. Compare final equity (before tearing the node down).
        venue = Venue(VENUE_MIC)
        account = node.kernel.portfolio.account(venue)
        live_equity_money = node.kernel.portfolio.equity(venue)[account.base_currency]
        live_equity = float(live_equity_money.as_decimal())

        # Sanity-check: at least one fill happened on each side. If the
        # diagnostic numbers are off, the assert below will print them too.
        from nautilus_trader.model.enums import OrderStatus
        exec_client = get_last_exec_client(VENUE_MIC)
        all_orders = list(node.kernel.cache.orders())
        n_filled = sum(1 for o in all_orders if o.status == OrderStatus.FILLED)
        n_pending = sum(len(v) for v in exec_client._pending.values())
        diagnostics = (
            f"orders={len(all_orders)} filled={n_filled} pending={n_pending} "
            f"exec_bars_received={exec_client._bars_received} "
            f"exec_fills_emitted={exec_client._fills_emitted} "
            f"backtest_trades={backtest_result.metrics.get('n_trades')}"
        )
        assert n_filled > 0, f"No orders filled on TradingNode path — {diagnostics}"
    finally:
        # Stop the node gracefully. We deliberately do NOT call node.dispose()
        # here — dispose() tears down the process-global Rust logging system
        # and (synchronously) stops the asyncio loop pytest-asyncio is using
        # under us, which surfaces as a RuntimeWarning on test teardown.
        # ``analysis.runner.run`` follows the same pattern for the same
        # reason; GC handles cleanup.
        await node.stop_async()
        if node_run_task is not None:
            node_run_task.cancel()
            try:
                await node_run_task
            except (asyncio.CancelledError, Exception):
                pass

    drift = abs(live_equity - backtest_equity) / backtest_equity
    print(
        f"\nBacktestEngine final_equity = {backtest_equity:.2f}\n"
        f"TradingNode    final_equity = {live_equity:.2f}\n"
        f"drift                       = {drift:.6f}\n"
        f"diag                        = {diagnostics}"
    )
    assert drift < 0.01, (
        f"TradingNode equity {live_equity:.2f} drifted {drift:.4f} from "
        f"BacktestEngine {backtest_equity:.2f} (>1% tolerance)\n{diagnostics}"
    )
