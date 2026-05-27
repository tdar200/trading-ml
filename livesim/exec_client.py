"""
SimulatedLiveExecutionClient — accepts orders and emits fills at same-bar close.

Subscribes (via the message bus) to bar events emitted by the data client.
On each bar for an instrument we have a pending order for, this client
buffers the fill (order, price, bar) rather than emitting it immediately.
The data client calls ``flush_buffered_fills()`` at the end of each
timestamp group (trading day), which emits the buffered fills in
SELL-before-BUY order. This mirrors BacktestEngine behaviour where fills
within the same rebalance always clear the SELL side first so that freed
cash is available before the BUY side debits the account.

Stage-4.5 fix — fill semantic matches BacktestEngine:
    Fills are computed against the **decision bar's close** (the bar whose
    ``on_bar`` invocation produced the order). This matches Nautilus's
    daily-bar ``SimulatedExchange`` semantic: without tick data the
    exchange fills market orders at the bar's close (the only
    "post-decision" price available within the same bar). The earlier
    livesim implementation deferred fills to the NEXT bar's open, which
    drifted from BacktestEngine by ~5% on the real catalog because of
    overnight gap risk between the decision bar close and the next bar
    open. We now: (a) drop the next-bar defer guard, (b) fill at
    ``bar.close``, (c) keep the SELL-before-BUY buffering — the buffering
    is still required because cross-ticker rotation days submit BUY and
    SELL orders on the same bar's processing, and on a CASH account the
    BUY must not debit before the SELL credit arrives.

Background (why buffering is needed): in the livesim's async architecture,
the exec client's ``_on_bar`` handler fires synchronously on every bar in
the order bars arrive (sorted by ticker index in ALL_TICKERS). When a
rebalance fires on ticker A's bar and submits a SELL for ticker B (later
in the sequence) and a BUY for ticker C (earlier in the sequence), C's
bar arrives before B's bar, so the BUY fills before the SELL. On a CASH
account the BUY debit can overdraw the account before the SELL credit
arrives, raising ``AccountBalanceNegative``. Buffering fills and flushing
in SELL-then-BUY order eliminates this race.

Per §12.5 of ``docs/nautilus_api_notes.md``:

- Base class is ``LiveExecutionClient``.
- ``_connect`` must call ``self._set_connected(True)`` and set an account id
  whose **issuer (part before the ``-``) equals ``client_id.value``** —
  ``Condition.equal`` enforces this; mismatch raises ``ValueError``.
- Required abstract overrides: ``_connect``, ``_disconnect``,
  ``_submit_order``, ``_cancel_order``, ``_cancel_all_orders``,
  ``generate_order_status_report``, ``generate_order_status_reports``,
  ``generate_fill_reports``, ``generate_position_status_reports``,
  ``generate_mass_status``.
- Fills are emitted via the base-class helper
  ``self.generate_order_filled(strategy_id, instrument_id, client_order_id,
  venue_order_id, venue_position_id, trade_id, order_side, order_type,
  last_qty, last_px, quote_currency, commission, liquidity_side, ts_event,
  info=None)``.

To produce equity that mirrors ``BacktestEngine``'s cash-account behaviour
we additionally:

- Register the venue as a "calculated account" with ``AccountFactory`` so
  the engine recomputes balances from OrderFilled events (see
  ``BacktestExecClient.__init__`` for the equivalent in the backtest path).
- Publish an initial ``AccountState`` with the configured starting balance
  at connect time (``BacktestEngine`` does this in
  ``SimulatedExchange.initialize_account`` via the same
  ``generate_account_state`` helper).
- Subscribe to ``data.bars.*`` on the message bus so the same bars the
  strategy sees also drive fills here.
- For every submitted order, generate the full lifecycle: submitted →
  accepted → filled. The exec engine refuses to fill an order that wasn't
  previously accepted (it logs "Cannot apply event to any order").
"""
from __future__ import annotations

import random
from decimal import Decimal

from nautilus_trader.accounting.factory import AccountFactory
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.live.execution_client import LiveExecutionClient
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import LiquiditySide, OrderSide, OrderType
from nautilus_trader.model.identifiers import (
    AccountId,
    TradeId,
    VenueOrderId,
)
from nautilus_trader.model.objects import AccountBalance, Money


def compute_fill_price(
    bar: Bar,
    side: OrderSide,
    slippage_ticks: int,
    tick_size: Decimal,
) -> Decimal:
    """Pure helper: compute the simulated fill price for an order against a bar.

    Returns the **same-bar close** (Stage-4.5 fix). Nautilus's
    ``SimulatedExchange`` fills daily-bar market orders against the bar's
    close — the only "post-decision" price available without tick data —
    so BacktestEngine uses ``bar.close``. The livesim previously used
    ``bar.open`` against the NEXT bar, which produced systematic drift
    via overnight gaps; aligning to ``bar.close`` of the decision bar
    eliminates that gap.

    Buys slip UP by ``slippage_ticks * tick_size`` (worse price for the buyer);
    sells slip DOWN by the same amount (worse price for the seller). With
    ``slippage_ticks == 0`` the bar's close price is returned unchanged.

    This function has no Nautilus-runtime dependencies and is trivially
    unit-testable.
    """
    base = Decimal(str(bar.close))
    delta = Decimal(slippage_ticks) * tick_size
    if side == OrderSide.BUY:
        return base + delta
    return base - delta


# Default tick size for the Stage-4 ETF universe (2dp, $0.01 increment).
_DEFAULT_TICK_SIZE: Decimal = Decimal("0.01")

# Type alias for a buffered fill: (order, venue_order_id, fill_price, bar).
_FillEntry = tuple  # (order, VenueOrderId, Decimal, Bar)


class SimulatedLiveExecutionClient(LiveExecutionClient):
    """Simulated live exec client — fills pending orders at same-bar close.

    Stage-4.5 semantic: orders are filled at the close of the bar that the
    strategy decided on (matches BacktestEngine's daily-bar SimulatedExchange).
    Fills are buffered per-day and flushed in SELL-before-BUY order so cash
    accounts never overdraw on rotation days.

    Constructor accepts Stage-4-specific kwargs (``slippage_prob``,
    ``random_seed``, ``starting_balance_usd``) plus the base-class kwargs
    forwarded via ``**kwargs`` (``loop``, ``client_id``, ``venue``,
    ``oms_type``, ``account_type``, ``base_currency``, ``instrument_provider``,
    ``msgbus``, ``cache``, ``clock``, ``config``) — see §12.5 for the
    verified signature.
    """

    def __init__(
        self,
        *,
        slippage_prob: float = 0.25,
        random_seed: int = 42,
        starting_balance_usd: int = 100_000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._slippage_prob = slippage_prob
        self._rng = random.Random(random_seed)
        self._starting_balance_usd = starting_balance_usd
        # ticker → most-recently-seen Bar. Updated in _on_bar; read in
        # _submit_order so fills land at the SAME bar's close that the
        # strategy decided on — even though SubmitOrder reaches us
        # asynchronously, AFTER the strategy's on_bar handler returned and
        # AFTER our own _on_bar has already processed the bar (both
        # subscribers fire from the same msgbus dispatch; the exec client
        # observes the bar BEFORE the strategy submits orders against it).
        # Without this dict, an order submitted on ticker T's bar misses
        # filling at T's close: by the time SubmitOrder arrives, T's bar
        # is already in the past and _on_bar will not fire for T again
        # until the NEXT day — the SELL never flushes in time and the
        # rotation's BUYs (which DO get filled because they happen to be
        # later in the day's bar sequence) overdraw the account.
        self._last_bar_by_ticker: dict[str, Bar] = {}
        # ticker → list of (order, venue_order_id) tuples. Holds orders
        # submitted before any bar for that ticker has been seen (e.g. an
        # order routed before the data client has emitted any data). With
        # buffer-at-submit-time semantics, this should normally stay empty
        # post-warmup; any non-empty residue at the end of fast-replay
        # surfaces in the integration test's diagnostics.
        self._pending: dict[str, list] = {}
        # Buffered fills waiting for flush_buffered_fills() — split by side so
        # SELLs are always emitted before BUYs within a day (mirrors BacktestEngine).
        self._buffered_fills_sell: list[_FillEntry] = []
        self._buffered_fills_buy: list[_FillEntry] = []
        # Monotonic counter for generating unique VenueOrderId / TradeId values.
        self._venue_order_seq: int = 0
        self._trade_seq: int = 0
        # Diagnostic counters (read by Task 8 integration test on failure).
        self._bars_received: int = 0
        self._fills_emitted: int = 0
        # Per §12.5: issuer of AccountId MUST equal ``client_id.value`` —
        # ``self.id`` is the ClientId.
        self._set_account_id(AccountId(f"{self.id.value}-001"))
        # Register the account as "calculated" — the engine recomputes
        # balances from OrderFilled events instead of waiting for an
        # external venue report. Mirrors ``BacktestExecClient.__init__``.
        # Idempotent guard: this is a module-global registry and re-registering
        # the same issuer raises ``KeyError`` (see ``AccountFactory``).
        try:
            AccountFactory.register_calculated_account(self.id.value)
        except KeyError:
            # Already registered in an earlier test in the same process.
            pass

    # ------------------------------------------------------------------
    # LiveExecutionClient required overrides (§12.5)
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        # Publish the initial AccountState so the portfolio can register the
        # cash account before any fills arrive. Mirrors
        # ``SimulatedExchange._generate_fresh_account_state``.
        starting = Money(self._starting_balance_usd, self.base_currency)
        zero = Money(0, self.base_currency)
        self.generate_account_state(
            balances=[AccountBalance(total=starting, locked=zero, free=starting)],
            margins=[],
            reported=True,
            ts_event=self._clock.timestamp_ns(),
        )
        # Subscribe to all bar events on the msgbus — see §12.4: bars are
        # published to ``data.bars.<bar_type>`` after the data engine routes
        # them through. Equivalent to ``SandboxExecutionClient.connect``'s
        # ``data.*.{venue}.*`` subscription, but bar-typed.
        self._msgbus.subscribe(topic="data.bars.*", handler=self._on_bar)
        self._set_connected(True)

    async def _disconnect(self) -> None:
        self._set_connected(False)

    async def _submit_order(self, command) -> None:
        """Record the order as pending and emit the submitted/accepted events.

        The exec engine only accepts an OrderFilled against an order that is
        already in the ACCEPTED state — without these two prior events the
        fill is dropped with "Cannot apply event to any order". So we emit
        them inline at submit time, then buffer the fill itself for
        end-of-day flush (Stage-4.5: same-bar close semantic).

        ``command`` is a ``SubmitOrder`` message (§12.5) with an ``.order``
        attribute. The fall-through ``else`` branch tolerates being handed a
        bare ``Order`` for ease of unit-testing.
        """
        order = command.order if hasattr(command, "order") else command
        ts_event = self._clock.timestamp_ns()
        self._venue_order_seq += 1
        venue_order_id = VenueOrderId(f"V-{self._venue_order_seq}")

        self.generate_order_submitted(
            strategy_id=order.strategy_id,
            instrument_id=order.instrument_id,
            client_order_id=order.client_order_id,
            ts_event=ts_event,
        )
        self.generate_order_accepted(
            strategy_id=order.strategy_id,
            instrument_id=order.instrument_id,
            client_order_id=order.client_order_id,
            venue_order_id=venue_order_id,
            ts_event=ts_event,
        )

        ticker = str(order.instrument_id).split(".")[0]
        # Stage-4.5: buffer the fill immediately against the most-recently-
        # seen bar's close for this ticker. The exec client subscribes to
        # data.bars.* and its _on_bar fires BEFORE the strategy's on_bar
        # (both are msgbus subscribers on the same topic), so by the time
        # SubmitOrder arrives here the bar that "produced" this order is
        # the LAST bar we saw for this ticker. Filling against its close
        # matches BacktestEngine's same-bar close semantic.
        #
        # If no bar has been seen yet (rare — only if SubmitOrder somehow
        # routes before any data), park the order in _pending; _on_bar's
        # residual-flush path will handle it when the first bar arrives.
        last_bar = self._last_bar_by_ticker.get(ticker)
        if last_bar is not None:
            self._buffer_fill(order=order, venue_order_id=venue_order_id, bar=last_bar)
        else:
            self._pending.setdefault(ticker, []).append((order, venue_order_id))

    async def _cancel_order(self, command) -> None:
        # No-op for the sim: fills happen on the same bar's close (or are
        # buffered for end-of-day flush), so a real cancellation race is not
        # exercised by this strategy.
        return None

    async def _cancel_all_orders(self, command) -> None:
        # No-op for the sim.
        return None

    # ------------------------------------------------------------------
    # Reconciliation report stubs — sim returns empty per §12.5.
    # ------------------------------------------------------------------

    async def generate_order_status_report(self, command):
        return None

    async def generate_order_status_reports(self, command) -> list:
        return []

    async def generate_fill_reports(self, command) -> list:
        return []

    async def generate_position_status_reports(self, command) -> list:
        return []

    async def generate_mass_status(self, lookback_mins: int | None = None):
        return None

    # ------------------------------------------------------------------
    # Bar handler — wired to ``data.bars.*`` in ``_connect``.
    # ------------------------------------------------------------------

    def _on_bar(self, bar: Bar) -> None:
        """Record the latest bar per ticker; flush any pre-bar-routed pending.

        Wired in ``_connect`` to the ``data.bars.*`` msgbus topic; the data
        client emits via ``self._handle_data`` which the data engine then
        publishes to this topic.

        Stage-4.5 semantic: this handler fires BEFORE the strategy's
        on_bar handler for the same bar (both are subscribers to the same
        topic; the exec client subscribes first). So when this handler
        runs, the strategy has NOT yet decided to submit any orders for
        this bar. We just record the bar for later use by ``_submit_order``,
        which will be invoked later (asynchronously, via the exec engine's
        command queue) and will buffer the fill against this bar's close.

        The residual ``_pending`` path handles the rare case where a
        SubmitOrder arrived before any bar for the ticker was ever seen
        (e.g. cold-start ordering races); those orders are flushed at the
        first bar that does arrive.
        """
        if not isinstance(bar, Bar):
            return
        self._bars_received += 1
        ticker = str(bar.bar_type).split(".")[0]
        self._last_bar_by_ticker[ticker] = bar
        if ticker in self._pending:
            for order, venue_order_id in self._pending[ticker]:
                self._buffer_fill(order=order, venue_order_id=venue_order_id, bar=bar)
            del self._pending[ticker]

    def _buffer_fill(self, order, venue_order_id: VenueOrderId, bar: Bar) -> None:
        """Buffer one fill into the SELL or BUY queue at the given bar's close.

        Splits by side so ``flush_buffered_fills()`` can emit SELLs first
        (frees cash) before BUYs (debits cash) — required on CASH accounts
        to avoid ``AccountBalanceNegative`` on rotation days.
        """
        slip_ticks = 1 if self._rng.random() < self._slippage_prob else 0
        fill_price = compute_fill_price(
            bar=bar,
            side=order.side,
            slippage_ticks=slip_ticks,
            tick_size=_DEFAULT_TICK_SIZE,
        )
        if order.side == OrderSide.SELL:
            self._buffered_fills_sell.append((order, venue_order_id, fill_price, bar))
        else:
            self._buffered_fills_buy.append((order, venue_order_id, fill_price, bar))

    def flush_buffered_fills(self) -> None:
        """Emit all buffered fills: SELLs first, then BUYs.

        Called by ``CatalogLiveDataClient._replay_once`` at the end of each
        timestamp group (trading day) after draining the exec-engine command
        queue. The SELL-first ordering ensures freed cash is credited before
        BUY debits, mirroring BacktestEngine fill sequencing.
        """
        sells = self._buffered_fills_sell
        buys = self._buffered_fills_buy
        self._buffered_fills_sell = []
        self._buffered_fills_buy = []
        for order, venue_order_id, fill_price, bar in sells:
            self._fills_emitted += 1
            self._emit_fill(order=order, venue_order_id=venue_order_id,
                            fill_price=fill_price, bar=bar)
        for order, venue_order_id, fill_price, bar in buys:
            self._fills_emitted += 1
            self._emit_fill(order=order, venue_order_id=venue_order_id,
                            fill_price=fill_price, bar=bar)

    def _emit_fill(
        self,
        order,
        venue_order_id: VenueOrderId,
        fill_price: Decimal,
        bar: Bar,
    ) -> None:
        """Emit OrderFilled via ``self.generate_order_filled`` per §12.5.

        Commission is zero (Equity instruments default to maker/taker fee = 0;
        the backtest path uses ``MakerTakerFeeModel`` which also returns 0 for
        equities). Liquidity side is TAKER (market orders cross the spread).
        """
        instrument = self._cache.instrument(order.instrument_id)
        quote_currency = instrument.quote_currency if instrument is not None else self.base_currency
        # Quantize the fill price to the instrument's price increment so the
        # ``Price`` constructor doesn't raise on extra precision.
        if instrument is not None:
            last_px = instrument.make_price(fill_price)
        else:
            from nautilus_trader.model.objects import Price  # local import to avoid top-level dep
            last_px = Price(fill_price, precision=2)

        self._trade_seq += 1
        trade_id = TradeId(f"T-{self._trade_seq}-{UUID4().value[:8]}")

        self.generate_order_filled(
            strategy_id=order.strategy_id,
            instrument_id=order.instrument_id,
            client_order_id=order.client_order_id,
            venue_order_id=venue_order_id,
            venue_position_id=None,
            trade_id=trade_id,
            order_side=order.side,
            order_type=OrderType.MARKET,
            last_qty=order.quantity,
            last_px=last_px,
            quote_currency=quote_currency,
            commission=Money(0, quote_currency),
            liquidity_side=LiquiditySide.TAKER,
            ts_event=bar.ts_event,
        )
