"""
CatalogLiveDataClient — replays bars from a callable provider into Nautilus's
TradingNode.

CRITICAL: bars emitted from inside ``_connect()`` are DROPPED because the
strategy hasn't subscribed yet at that point in the lifecycle. The
implementation defers emission via ``self.create_task(self._emit_after_delay(...))``
so the bars land after ``Trader.start()`` has fired the strategy subscriptions.
See §12.4 / §12.9 of ``docs/nautilus_api_notes.md`` for the verified pattern.

The data client is decoupled from any catalog/filesystem concerns by taking a
callable ``bar_provider() -> dict[str, list[Bar]]``. The runnable in
``examples/run_trading_node.py`` will wrap a ``ParquetDataCatalog`` in a
closure; tests pass a dict-returning lambda.

Per §12.4 of the API notes:

- Base class is ``LiveMarketDataClient`` (NOT ``LiveDataClient`` — that is for
  custom data types only).
- ``_connect`` MUST call ``self._set_connected(True)`` or the node hangs in
  the "Awaiting engine connections and initializations" phase until
  ``timeout_connection`` fires.
- ``_subscribe_bars`` / ``_unsubscribe_bars`` are async coroutines that take a
  ``SubscribeBars`` / ``UnsubscribeBars`` command — not a bare ``bar_type``.
"""
from __future__ import annotations

import asyncio
from typing import Callable

from nautilus_trader.live.data_client import LiveMarketDataClient
from nautilus_trader.model.data import Bar


BarProvider = Callable[[], dict[str, list[Bar]]]

# Small delay between _connect and first bar emission to ensure strategies
# have completed their subscription handshake (Trader.start() fires after the
# connection step in the kernel lifecycle).
_INITIAL_EMIT_DELAY_SECONDS: float = 0.5

# Number of asyncio.sleep(0) calls used to drain the exec engine's command
# queue after a timestamp group's bars are emitted. Each sleep gives the
# event loop one scheduling cycle; the command queue drains one SubmitOrder
# per cycle. A top_n=3 rebalance generates up to 6 SubmitOrder commands
# (3 SELLs + 3 BUYs on a full rotation). Each command also generates
# submitted+accepted events on the evt queue, so we need enough yields for
# both. 50 drains is conservative overkill to ensure all commands drain.
_CMD_DRAIN_YIELDS: int = 50

# Number of asyncio.sleep(0) calls used to drain the exec engine's event
# queue after ``flush_buffered_fills()`` emits fill events. Each fill
# generates an OrderFilled event + a PositionOpened/Changed/Closed event
# = 2 events per fill; top_n=3 with rotation has ~6 fills = 12 events.
# 50 drains is conservative overkill.
_EVT_DRAIN_YIELDS: int = 50


def replay_bars(bars: list[Bar], emit_fn: Callable[[Bar], None]) -> None:
    """Pure helper: emit each bar in ascending ``ts_event`` order.

    Decoupled from any Nautilus runtime so it is trivially unit-testable.
    """
    for bar in sorted(bars, key=lambda b: b.ts_event):
        emit_fn(bar)


class CatalogLiveDataClient(LiveMarketDataClient):
    """Replays bars from a callable provider into the engine.

    - **fast-replay mode** (``poll_seconds=None``): emit every bar the provider
      returns once, then resolve ``done()`` so the caller can shut the node
      down cleanly.
    - **continuous mode** (``poll_seconds > 0``): emit all currently-known bars,
      then poll the provider every ``poll_seconds`` for newly-appeared bars
      and emit any deltas. ``done()`` only resolves on ``_disconnect``.
    """

    def __init__(
        self,
        *,
        bar_provider: BarProvider,
        poll_seconds: float | None = None,
        # The remaining kwargs are forwarded to ``LiveMarketDataClient`` — see
        # §12.4 for the verified signature
        # (loop, client_id, venue, msgbus, cache, clock, instrument_provider,
        #  config=None, is_sync=False).
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._bar_provider = bar_provider
        self._poll_seconds = poll_seconds
        # Per-ticker set of ts_event values we've already emitted, used so
        # continuous-mode polling only emits new bars.
        self._emitted_ts_per_ticker: dict[str, set[int]] = {}
        self._done: asyncio.Future = asyncio.Future()
        self._poll_task: asyncio.Task | None = None
        # Optional reference to the paired exec client — set lazily at first
        # use by _replay_once by looking up factories._LAST_EXEC_CLIENT.
        # When present, flush_buffered_fills() is called after each timestamp
        # group so that SELL fills are emitted before BUY fills, preventing
        # AccountBalanceNegative on cross-ticker rotation days.
        self._exec_client = None  # type: ignore[assignment]

    def done(self) -> asyncio.Future:
        """Resolves to ``True`` when fast-replay completes (or ``False`` on
        disconnect before completion). Never resolves on its own in
        continuous mode."""
        return self._done

    # ------------------------------------------------------------------
    # LiveMarketDataClient required overrides (§12.4)
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Per §12.4: MUST call ``_set_connected(True)``. Bars emitted inline
        here would be dropped because strategies haven't subscribed yet — so
        defer emission to a task that wakes up after ``Trader.start()``.
        """
        self._set_connected(True)
        self.create_task(
            self._emit_after_delay(_INITIAL_EMIT_DELAY_SECONDS),
            log_msg="catalog_replay_emit",
        )

    async def _disconnect(self) -> None:
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
        if not self._done.done():
            self._done.set_result(False)
        self._set_connected(False)

    async def _subscribe_bars(self, command) -> None:
        """No-op: this client emits bars internally via the provider; the
        strategy's ``subscribe_bars`` call still needs this hook to exist
        (otherwise the engine logs an error per §12.4). The actual delivery
        of bars to subscribed strategies happens via ``self._handle_data``.
        """
        # command is a SubscribeBars message; we intentionally don't inspect it.
        return None

    async def _unsubscribe_bars(self, command) -> None:
        return None

    # ------------------------------------------------------------------
    # Internal replay / polling machinery
    # ------------------------------------------------------------------

    async def _emit_after_delay(self, delay: float) -> None:
        """Wait briefly so strategy subscriptions complete, then emit bars."""
        await asyncio.sleep(delay)
        await self._replay_once()
        if self._poll_seconds is not None:
            self._poll_task = asyncio.create_task(self._poll_loop())
        else:
            if not self._done.done():
                self._done.set_result(True)

    def _get_exec_client(self):
        """Lazy-fetch the paired exec client from the factories module.

        Uses a late import to avoid a circular dependency (factories imports
        data_client; data_client must not import factories at module level).
        Returns None if not found (e.g., in unit tests that don't use the
        factory).
        """
        if self._exec_client is not None:
            return self._exec_client
        try:
            from livesim.factories import get_last_exec_client  # late import
            venue_name = self.id.value  # client_id == venue MIC (e.g. "ARCX")
            self._exec_client = get_last_exec_client(venue_name)
        except (ImportError, KeyError):
            self._exec_client = None  # type: ignore[assignment]
        return self._exec_client

    async def _flush_day(self) -> None:
        """Drain cmd queue, flush fills (SELLs first), then drain evt queue.

        Called between timestamp groups in ``_replay_once`` to ensure:
        1. All SubmitOrder commands are processed (orders land in _pending).
        2. Buffered fills are emitted in SELL-before-BUY order.
        3. All resulting fill events are processed (balances + positions updated).

        Without this, cross-ticker rotation days overdraw the account:
        BUY ticker-A (low index) fills before SELL ticker-B (high index) on
        the same day, debiting cash before the corresponding credit arrives.
        """
        # Step 1: drain exec engine command queue so all orders are in _pending.
        for _ in range(_CMD_DRAIN_YIELDS):
            await asyncio.sleep(0)

        # Step 2: flush buffered fills (SELL before BUY).
        exec_client = self._get_exec_client()
        if exec_client is not None:
            exec_client.flush_buffered_fills()

        # Step 3: drain exec engine event queue so fills and position events
        # are processed before the next day's bars arrive.
        for _ in range(_EVT_DRAIN_YIELDS):
            await asyncio.sleep(0)

    async def _replay_once(self) -> None:
        """Read current bar set from provider; emit any not-yet-emitted bars.

        Bars are emitted in **interleaved ts_event order** across tickers (not
        ticker-by-ticker). After each *timestamp group* (all bars sharing the
        same ts_event — one trading day), ``_flush_day()`` drains the command
        queue, emits buffered fills in SELL-before-BUY order, then drains the
        event queue. This prevents ``AccountBalanceNegative`` on rotation days
        where the SELL ticker appears later in the daily bar sequence than the
        BUY ticker.
        """
        bars_by_ticker = self._bar_provider()
        merged: list[Bar] = []
        for ticker, bars in bars_by_ticker.items():
            emitted = self._emitted_ts_per_ticker.setdefault(ticker, set())
            for b in bars:
                if b.ts_event not in emitted:
                    merged.append(b)
                    emitted.add(b.ts_event)
        merged.sort(key=lambda b: b.ts_event)
        prev_ts: int | None = None
        for bar in merged:
            # At timestamp group boundary: flush the previous day's fills
            # before emitting the new day's first bar.
            if prev_ts is not None and bar.ts_event != prev_ts:
                await self._flush_day()
            self._handle_data(bar)
            # Yield once per bar so the exec engine can process the SubmitOrder
            # commands the strategy just enqueued (needed before the next bar
            # fires _on_bar, which checks _pending for fills).
            await asyncio.sleep(0)
            prev_ts = bar.ts_event
        # Final flush for the last day's fills.
        await self._flush_day()

    async def _poll_loop(self) -> None:
        """Continuous-mode polling. Re-reads from provider every poll_seconds."""
        assert self._poll_seconds is not None  # narrowing for type checkers
        while True:
            try:
                await asyncio.sleep(self._poll_seconds)
                await self._replay_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — log and continue
                # Log via Nautilus's logger; don't crash the loop.
                self._log.error(f"CatalogLiveDataClient poll error: {e}")
