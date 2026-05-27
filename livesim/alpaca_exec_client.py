"""
AlpacaLiveExecutionClient — Nautilus LiveExecutionClient backed by Alpaca paper trading.

Translates Nautilus market orders into Alpaca submit_order calls; a polling task
fetches new fills from Alpaca and emits them as Nautilus OrderFilled events.

Scope (per Stage 8 design): MARKET orders only, paper account only. Real-money
rollout is gated by the 3-month observation in
``docs/superpowers/specs/2026-05-27-stage-8-deploy-checklist.md`` and is a
human decision, not a code change of this module.

Boundaries: this is the **only** module in the project that imports
``alpaca.trading``. The data side stays catalog-driven (``CatalogLiveDataClient``)
because the strategy is daily-bar; live tick feeds are out of scope.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from nautilus_trader.accounting.factory import AccountFactory
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.live.execution_client import LiveExecutionClient
from nautilus_trader.model.enums import LiquiditySide, OrderSide, OrderType
from nautilus_trader.model.identifiers import (
    AccountId,
    TradeId,
    VenueOrderId,
)
from nautilus_trader.model.objects import AccountBalance, Money


# Default poll cadence for fill detection. 1 Hz is plenty for daily-bar
# strategies (typically ≤6 orders per rebalance, monthly) and well inside
# Alpaca's 200/min free-paper limit.
_DEFAULT_FILL_POLL_SECONDS: float = 1.0

# Delay between SELL submissions and BUY submissions on the same rebalance.
# Even on Alpaca paper, market-order SELLs settle near-instantly into the cash
# balance, but a small gap removes any race against an immediate BUY for the
# rotation pair.
_SELL_TO_BUY_DELAY_SECONDS: float = 0.5


def _translate_alpaca_status(status: str) -> str:
    """Reduce Alpaca order statuses to one of the categories this module cares
    about: 'filled', 'partial', 'rejected', 'pending', 'cancelled'.

    Pure helper — testable without any Alpaca client.
    """
    status = (status or "").lower()
    if status == "filled":
        return "filled"
    if status == "partially_filled":
        return "partial"
    if status in {"rejected", "expired"}:
        return "rejected"
    if status in {"canceled", "cancelled"}:
        return "cancelled"
    return "pending"


def _build_market_order_request(
    symbol: str,
    qty: int,
    side: OrderSide,
    *,
    client_order_id: str | None = None,
):
    """Translate a Nautilus market order into an Alpaca MarketOrderRequest.

    Pulled into a module-level helper so the test suite can verify request
    shape without needing a TradingClient.
    """
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    alpaca_side = AlpacaOrderSide.BUY if side == OrderSide.BUY else AlpacaOrderSide.SELL
    return MarketOrderRequest(
        symbol=symbol,
        qty=int(qty),
        side=alpaca_side,
        time_in_force=TimeInForce.DAY,
        client_order_id=client_order_id,
    )


class AlpacaLiveExecutionClient(LiveExecutionClient):
    """Live execution client that routes orders to an Alpaca paper account.

    Constructor extras (beyond the base class kwargs):
        api_key, secret_key: paper-account credentials. If None, falls back to
            env vars ``ALPACA_API_KEY`` / ``ALPACA_API_SECRET``.
        paper: must be True for now (real-money is gated by the deploy
            checklist; a future Stage 8.1 flips this).
        fill_poll_seconds: how often to poll Alpaca for new fills.
        sell_to_buy_delay_seconds: cash-account safety gap between SELL
            submissions and BUY submissions on the same rebalance.
        starting_balance_usd: published as the initial Nautilus AccountState
            on _connect. Real balance is fetched from Alpaca and OVERRIDES
            this on first poll.

    Designed around the same shape as livesim/exec_client.py
    (SimulatedLiveExecutionClient) so the rest of the runtime
    (factories, node config) treats them interchangeably.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
        fill_poll_seconds: float = _DEFAULT_FILL_POLL_SECONDS,
        sell_to_buy_delay_seconds: float = _SELL_TO_BUY_DELAY_SECONDS,
        starting_balance_usd: int = 100_000,
        trading_client=None,  # injectable for tests; otherwise built from credentials
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._paper = paper
        self._fill_poll_seconds = fill_poll_seconds
        self._sell_to_buy_delay_seconds = sell_to_buy_delay_seconds
        self._starting_balance_usd = starting_balance_usd

        # Lazy / injected TradingClient — keeps unit tests offline.
        self._trading_client = trading_client
        self._api_key = api_key or os.getenv("ALPACA_API_KEY")
        self._secret_key = secret_key or os.getenv("ALPACA_API_SECRET")

        # Maps: Nautilus client_order_id (str) ↔ Alpaca order id (str)
        self._client_to_alpaca: dict[str, str] = {}
        self._alpaca_to_nautilus: dict[str, Any] = {}  # alpaca_id → original Nautilus order

        # Fills already emitted (Alpaca order id) so the polling loop is idempotent.
        self._emitted_fills: set[str] = set()

        # Poll loop bookkeeping.
        self._last_poll_after: datetime | None = None
        self._poll_task: asyncio.Task | None = None
        # Tracks when the last SELL was submitted, so a BUY arriving in the
        # same async cycle waits the configured gap before going out.
        self._last_sell_submit_ts: float = 0.0

        # Diagnostic counters (read by tests / dashboard).
        self._orders_submitted: int = 0
        self._fills_emitted: int = 0
        self._poll_ticks: int = 0

        # Per §12.5 of the Nautilus API notes: issuer of AccountId MUST equal
        # the ClientId's value.
        self._set_account_id(AccountId(f"{self.id.value}-001"))
        try:
            AccountFactory.register_calculated_account(self.id.value)
        except KeyError:
            pass  # already registered in this process

    # ------------------------------------------------------------------
    # LiveExecutionClient required overrides
    # ------------------------------------------------------------------

    def _build_trading_client(self):
        """Lazy-build the Alpaca TradingClient from credentials. Tests can
        inject ``trading_client=...`` to skip this entirely."""
        if self._trading_client is not None:
            return self._trading_client
        if not self._api_key or not self._secret_key:
            raise RuntimeError(
                "Alpaca credentials missing: set ALPACA_API_KEY / ALPACA_API_SECRET "
                "in .env or pass api_key/secret_key explicitly."
            )
        from alpaca.trading.client import TradingClient
        self._trading_client = TradingClient(
            api_key=self._api_key, secret_key=self._secret_key, paper=self._paper,
        )
        return self._trading_client

    async def _connect(self) -> None:
        # Publish initial AccountState (overridden on first poll once we know
        # the real Alpaca balance).
        starting = Money(self._starting_balance_usd, self.base_currency)
        zero = Money(0, self.base_currency)
        self.generate_account_state(
            balances=[AccountBalance(total=starting, locked=zero, free=starting)],
            margins=[],
            reported=True,
            ts_event=self._clock.timestamp_ns(),
        )
        self._set_connected(True)
        # Kick off the fill-polling loop.
        self._last_poll_after = datetime.now(tz=timezone.utc)
        self._poll_task = asyncio.create_task(self._poll_fills_loop())

    async def _disconnect(self) -> None:
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
        self._set_connected(False)

    async def _submit_order(self, command) -> None:
        """Translate command.order into an Alpaca MarketOrderRequest and submit."""
        order = command.order if hasattr(command, "order") else command
        ticker = str(order.instrument_id).split(".")[0]
        client_oid = str(order.client_order_id)

        # SELL/BUY pacing: if this is a BUY and we just submitted a SELL,
        # wait so the SELL credit settles in Alpaca's account view.
        if order.side == OrderSide.BUY and self._sell_to_buy_delay_seconds > 0:
            loop_time = asyncio.get_event_loop().time()
            wait_remaining = self._last_sell_submit_ts + self._sell_to_buy_delay_seconds - loop_time
            if wait_remaining > 0:
                await asyncio.sleep(wait_remaining)

        request = _build_market_order_request(
            ticker, int(order.quantity), order.side, client_order_id=client_oid,
        )
        client = self._build_trading_client()

        # The actual submit is sync in alpaca-py; run in default executor so
        # we don't block the event loop.
        loop = asyncio.get_event_loop()
        alpaca_order = await loop.run_in_executor(None, client.submit_order, request)
        self._orders_submitted += 1

        alpaca_id = str(alpaca_order.id)
        self._client_to_alpaca[client_oid] = alpaca_id
        self._alpaca_to_nautilus[alpaca_id] = order

        if order.side == OrderSide.SELL:
            self._last_sell_submit_ts = asyncio.get_event_loop().time()

        # Mirror simulated client: emit Nautilus 'submitted' + 'accepted' so the
        # exec engine can later apply OrderFilled to a known order.
        ts_event = self._clock.timestamp_ns()
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
            venue_order_id=VenueOrderId(alpaca_id),
            ts_event=ts_event,
        )

    async def _cancel_order(self, command) -> None:
        client = self._build_trading_client()
        client_oid = str(command.client_order_id)
        alpaca_id = self._client_to_alpaca.get(client_oid)
        if alpaca_id is None:
            return None
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, client.cancel_order_by_id, alpaca_id)
        except Exception as exc:
            self._log.warning(f"cancel_order failed for {alpaca_id}: {exc}")

    async def _cancel_all_orders(self, command) -> None:
        client = self._build_trading_client()
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, client.cancel_orders)
        except Exception as exc:
            self._log.warning(f"cancel_all_orders failed: {exc}")

    # ------------------------------------------------------------------
    # Reconciliation report stubs — paper smoke run accepts empty.
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
    # Polling loop
    # ------------------------------------------------------------------

    async def _poll_fills_loop(self) -> None:
        """Periodically ask Alpaca for new order state and emit Nautilus
        events for any newly-filled / rejected / cancelled orders.

        Cancellations and rejections are logged via ``self._log`` rather than
        emitted as OrderEvents — keeps the strategy / risk engine unaware of
        adapter-internal lifecycle noise.
        """
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        client = self._build_trading_client()
        loop = asyncio.get_event_loop()
        while True:
            try:
                await asyncio.sleep(self._fill_poll_seconds)
                self._poll_ticks += 1
                req = GetOrdersRequest(
                    status=QueryOrderStatus.ALL,
                    after=self._last_poll_after,
                    limit=500,
                )
                orders = await loop.run_in_executor(None, client.get_orders, req)
                self._last_poll_after = datetime.now(tz=timezone.utc)
                for o in orders or []:
                    self._handle_alpaca_order_update(o)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._log.error(f"AlpacaLiveExecutionClient poll error: {exc}")

    def _handle_alpaca_order_update(self, alpaca_order) -> None:
        """Inspect one Alpaca order and emit a Nautilus OrderFilled if new."""
        alpaca_id = str(alpaca_order.id)
        status = _translate_alpaca_status(getattr(alpaca_order, "status", ""))
        if status in {"rejected", "cancelled"}:
            self._log.warning(
                f"Alpaca order {alpaca_id} status={status}: {getattr(alpaca_order, 'symbol', '?')} "
                f"{getattr(alpaca_order, 'side', '?')} {getattr(alpaca_order, 'qty', '?')}"
            )
            return
        if status != "filled":
            return  # partial / pending: wait for full fill
        if alpaca_id in self._emitted_fills:
            return  # already emitted

        order = self._alpaca_to_nautilus.get(alpaca_id)
        if order is None:
            # Filled order we don't recognise (e.g., placed via Alpaca UI
            # during reconciliation). Skip — strategy doesn't own it.
            return

        self._emit_fill_from_alpaca(order, alpaca_order)
        self._emitted_fills.add(alpaca_id)
        self._fills_emitted += 1

    def _emit_fill_from_alpaca(self, nautilus_order, alpaca_order) -> None:
        """Emit a Nautilus OrderFilled from an Alpaca filled order."""
        instrument = self._cache.instrument(nautilus_order.instrument_id)
        quote_currency = (
            instrument.quote_currency if instrument is not None else self.base_currency
        )
        fill_qty = Decimal(str(getattr(alpaca_order, "filled_qty", nautilus_order.quantity)))
        fill_price_decimal = Decimal(str(getattr(alpaca_order, "filled_avg_price", "0")))
        if instrument is not None:
            last_px = instrument.make_price(fill_price_decimal)
        else:
            from nautilus_trader.model.objects import Price
            last_px = Price(fill_price_decimal, precision=2)

        from nautilus_trader.model.objects import Quantity
        last_qty = Quantity.from_int(int(fill_qty))

        ts_event = self._clock.timestamp_ns()
        venue_order_id = VenueOrderId(str(alpaca_order.id))
        trade_id = TradeId(f"T-{UUID4().value[:8]}")

        self.generate_order_filled(
            strategy_id=nautilus_order.strategy_id,
            instrument_id=nautilus_order.instrument_id,
            client_order_id=nautilus_order.client_order_id,
            venue_order_id=venue_order_id,
            venue_position_id=None,
            trade_id=trade_id,
            order_side=nautilus_order.side,
            order_type=OrderType.MARKET,
            last_qty=last_qty,
            last_px=last_px,
            quote_currency=quote_currency,
            commission=Money(0, quote_currency),
            liquidity_side=LiquiditySide.TAKER,
            ts_event=ts_event,
        )
