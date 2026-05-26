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

    async def _replay_once(self) -> None:
        """Read current bar set from provider; emit any not-yet-emitted bars.

        Bars are emitted in **interleaved ts_event order** across tickers (not
        ticker-by-ticker), and the coroutine yields with ``asyncio.sleep(0)``
        after every bar so the live exec engine's async command queue can
        drain (submitted orders need to land in the exec client's ``_pending``
        bucket *before* the next bar arrives, otherwise no fills are ever
        emitted in fast-replay).
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
        for bar in merged:
            self._handle_data(bar)
            # Yield to the event loop so the exec engine can process the
            # SubmitOrder command(s) the strategy just enqueued before the
            # next bar fires the exec client's _on_bar (which is when those
            # newly-pending orders should fill).
            await asyncio.sleep(0)

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
