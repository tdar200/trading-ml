"""
Stage-4 runnable: start a TradingNode running MomentumTopNStrategy against
the Stage-1 catalog via livesim's CatalogLiveDataClient + SimulatedLiveExecutionClient.

Usage:
    uv run python examples/run_trading_node.py --fast-replay
        Emit all catalog bars as fast as possible, then exit. Prints final equity.

    uv run python examples/run_trading_node.py --poll-seconds 60
        Continuous mode: emit historical bars, then poll the catalog every 60s
        for newly-appeared bars and emit them. Runs until Ctrl-C / SIGTERM.

    uv run python examples/run_trading_node.py
        Same as --poll-seconds 86400 (one day; matches the daily ingest cadence).

    uv run python examples/run_trading_node.py --fast-replay --slippage-prob 0.0
        Run fast-replay with deterministic (no-slippage) fills -- useful for the
        Stage-4 acceptance parity check vs BacktestEngine.
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

from data.instruments import VENUE_MIC, build_universe_instruments
from data.tickers import ALL_TICKERS, BENCHMARK, UNIVERSE
from livesim.factories import (
    CatalogDataClientFactory,
    SimulatedExecClientFactory,
    get_last_data_client,
)
from livesim.node_config import (
    DataClientParams,
    ExecClientParams,
    build_trading_node_config,
    register_data_client_params,
    register_exec_client_params,
    reset_registries,
)


CATALOG_PATH = _PROJECT_ROOT / "data" / "catalog"


def make_catalog_bar_provider(catalog_path: Path):
    """Closure that reads bars from the Parquet catalog on each call.

    For continuous mode, this is invoked on each poll tick by the data
    client so newly-ingested bars are picked up automatically.
    """
    catalog = ParquetDataCatalog(path=str(catalog_path.resolve()))

    def provider():
        out = {}
        for ticker in ALL_TICKERS:
            bars = catalog.bars(instrument_ids=[f"{ticker}.{VENUE_MIC}"])
            out[ticker] = bars
        return out

    return provider


async def _run(fast_replay: bool, poll_seconds: float | None, slippage_prob: float) -> int:
    reset_registries()

    # 1. Bar provider from the real catalog.
    bar_provider = make_catalog_bar_provider(CATALOG_PATH)
    snapshot = bar_provider()
    total_bars = sum(len(b) for b in snapshot.values())
    print(f"loaded {total_bars} bars across {len(snapshot)} tickers")
    if total_bars == 0:
        print("ERROR: catalog is empty. Run data/ingest_alpaca.py first.")
        return 1

    # 2. Register livesim params.
    # In fast-replay mode, poll_seconds=None signals one-shot emission.
    # In continuous mode, poll_seconds drives the polling cadence.
    effective_poll = None if fast_replay else poll_seconds
    register_data_client_params(
        VENUE_MIC,
        DataClientParams(bar_provider=bar_provider, poll_seconds=effective_poll),
    )
    register_exec_client_params(VENUE_MIC, ExecClientParams(slippage_prob=slippage_prob))

    # 3. Build the TradingNodeConfig with the Stage-3 winner combo.
    config = build_trading_node_config(
        strategy_path="strategies.momentum_topn:MomentumTopNStrategy",
        strategy_config_path="strategies.momentum_topn:MomentumTopNConfig",
        strategy_config_kwargs={
            "universe": UNIVERSE,
            "benchmark": BENCHMARK,
            "top_n": 2,
            "lookback_days": 63,
            "skip_days": 0,  # Stage-3 winner (re-baselined to top_n=2 post Stage-4.5 followup)
        },
    )

    # 4. Construct + register factories + build.
    node = TradingNode(config=config)
    node.add_data_client_factory(VENUE_MIC, CatalogDataClientFactory)
    node.add_exec_client_factory(VENUE_MIC, SimulatedExecClientFactory)
    node.build()

    # 5. Add instruments to the cache so the strategy can resolve them.
    for instrument in build_universe_instruments():
        node.kernel.cache.add_instrument(instrument)

    # 6. Fetch the data client via the factory registry (same pattern the
    #    integration test uses — avoids groping through Cython-private
    #    DataEngine._clients).
    data_client = get_last_data_client(VENUE_MIC)

    # 7. SIGINT/SIGTERM handler so continuous mode shuts down cleanly.
    stop_event = asyncio.Event()

    def _on_stop(*_args):
        print("\n(stop signal received; shutting down)")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_stop)

    # 8. Run the node. Read the final equity BEFORE stop+dispose so we
    #    don't race with the cache being torn down (integration test does
    #    the same thing).
    final_equity_str: str | None = None
    run_task: asyncio.Task | None = None
    try:
        run_task = asyncio.create_task(node.run_async())

        if fast_replay:
            print("fast-replay mode; waiting for emission to complete...")
            try:
                await asyncio.wait_for(data_client.done(), timeout=600.0)
                print("fast-replay complete")
            except asyncio.TimeoutError:
                print("ERROR: fast-replay did not complete within 600 seconds")
                return 2
            # Brief settle for trailing fills/equity updates.
            await asyncio.sleep(1.0)
        else:
            print(f"continuous mode; polling every {poll_seconds}s. Ctrl+C to stop.")
            await stop_event.wait()

        # 9. Read final account state BEFORE stop+dispose.
        venue = Venue(VENUE_MIC)
        account = node.kernel.portfolio.account(venue)
        if account is not None:
            equity = node.kernel.portfolio.equity(venue)[account.base_currency]
            final_equity_str = str(equity)
    finally:
        try:
            await node.stop_async()
        except Exception as e:
            print(f"warn: stop_async raised: {e}")
        try:
            node.dispose()
        except Exception:
            pass
        if run_task is not None and not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except (asyncio.CancelledError, Exception):
                pass

    if final_equity_str is not None:
        print(f"final equity: {final_equity_str}")
    else:
        print("warn: could not read final equity (no account registered)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fast-replay",
        action="store_true",
        help="Emit all bars then exit (no polling).",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=86400.0,
        help="Polling cadence in continuous mode (default 86400 = 1 day).",
    )
    parser.add_argument(
        "--slippage-prob",
        type=float,
        default=0.25,
        help="SimulatedLiveExecutionClient slippage probability (0.0-1.0, default 0.25).",
    )
    args = parser.parse_args()
    # uvloop + Nautilus TradingNode.run_async() interact poorly on shutdown:
    # the kernel stops its own event loop while asyncio.run() is still
    # holding a reference, producing "Event loop stopped before Future
    # completed". The main work has already completed by then (final equity
    # was already printed); suppress the cosmetic teardown error so callers
    # see exit code 0 on success.
    try:
        return asyncio.run(
            _run(
                fast_replay=args.fast_replay,
                poll_seconds=args.poll_seconds,
                slippage_prob=args.slippage_prob,
            )
        )
    except RuntimeError as e:
        if "Event loop stopped before Future completed" in str(e):
            return 0
        raise


if __name__ == "__main__":
    raise SystemExit(main())
