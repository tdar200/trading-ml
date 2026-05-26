"""
SimulatedLiveExecutionClient — accepts orders and emits fills at the next bar.

Subscribes (via the message bus) to bar events emitted by the data client.
On each bar for an instrument we have a pending order for, this client
generates a fill at the bar's open price + optional one-tick slippage gated
by a probability (seeded for determinism).

The fill-price decision is extracted into the pure ``compute_fill_price``
helper for unit testability. The full async order-fill cycle is exercised
in Task 8's TradingNode integration test (not here).

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
  info=None)``. Task 8 wires this end-to-end; Task 5 leaves it stubbed.
"""
from __future__ import annotations

import random
from decimal import Decimal

from nautilus_trader.live.execution_client import LiveExecutionClient
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import AccountId


def compute_fill_price(
    bar: Bar,
    side: OrderSide,
    slippage_ticks: int,
    tick_size: Decimal,
) -> Decimal:
    """Pure helper: compute the simulated fill price for an order against a bar.

    Buys slip UP by ``slippage_ticks * tick_size`` (worse price for the buyer);
    sells slip DOWN by the same amount (worse price for the seller). With
    ``slippage_ticks == 0`` the bar's open price is returned unchanged.

    This function has no Nautilus-runtime dependencies and is trivially
    unit-testable.
    """
    base = Decimal(str(bar.open))
    delta = Decimal(slippage_ticks) * tick_size
    if side == OrderSide.BUY:
        return base + delta
    return base - delta


# Default tick size for the Stage-4 ETF universe (2dp, $0.01 increment).
_DEFAULT_TICK_SIZE: Decimal = Decimal("0.01")


class SimulatedLiveExecutionClient(LiveExecutionClient):
    """Simulated live exec client — fills pending orders at the next bar's open.

    Constructor accepts Stage-4-specific kwargs (``slippage_prob``,
    ``random_seed``) plus the base-class kwargs forwarded via ``**kwargs``
    (``loop``, ``client_id``, ``venue``, ``oms_type``, ``account_type``,
    ``base_currency``, ``instrument_provider``, ``msgbus``, ``cache``,
    ``clock``, ``config``) — see §12.5 for the verified signature.
    """

    def __init__(
        self,
        *,
        slippage_prob: float = 0.25,
        random_seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._slippage_prob = slippage_prob
        self._rng = random.Random(random_seed)
        # ticker (string prefix of the InstrumentId, e.g. "SPY") → pending orders
        self._pending: dict[str, list] = {}

    # ------------------------------------------------------------------
    # LiveExecutionClient required overrides (§12.5)
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        # Per §12.5: issuer of AccountId MUST equal ``client_id.value`` —
        # ``self.id`` is the ClientId (verified via dir(LiveExecutionClient)).
        account_id = AccountId(f"{self.id.value}-001")
        self._set_account_id(account_id)
        self._set_connected(True)

    async def _disconnect(self) -> None:
        self._set_connected(False)

    async def _submit_order(self, command) -> None:
        """Record the order as pending; fill is emitted on the next bar.

        The command is a ``SubmitOrder`` message (§12.5) with an ``.order``
        attribute. The fall-through ``else`` branch tolerates being handed a
        bare ``Order`` for ease of unit-testing.
        """
        order = command.order if hasattr(command, "order") else command
        ticker = str(order.instrument_id).split(".")[0]
        self._pending.setdefault(ticker, []).append(order)

    async def _cancel_order(self, command) -> None:
        # No-op for the sim: fills happen on the very next bar so a real
        # cancellation race is not exercised.
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
    # Bar handler — wired to the data client's bar stream in Task 8.
    # ------------------------------------------------------------------

    def _on_bar(self, bar: Bar) -> None:
        """Fill any pending orders for this bar's instrument at the bar's open.

        Wired by Task 8's integration test once the TradingNode + data + exec
        + strategy graph is in place. See ``_emit_fill`` below for the
        deferred ``generate_order_filled`` wiring.
        """
        ticker = str(bar.bar_type).split(".")[0]
        if ticker not in self._pending:
            return
        pending_orders = self._pending.pop(ticker)
        for order in pending_orders:
            slip_ticks = 1 if self._rng.random() < self._slippage_prob else 0
            fill_price = compute_fill_price(
                bar=bar,
                side=order.side,
                slippage_ticks=slip_ticks,
                tick_size=_DEFAULT_TICK_SIZE,
            )
            self._emit_fill(order, fill_price=fill_price, bar=bar)

    def _emit_fill(self, order, fill_price: Decimal, bar: Bar) -> None:
        """Emit OrderFilled via ``self.generate_order_filled`` per §12.5.

        Implementation deferred to Task 8 integration — the full 15-arg
        ``generate_order_filled`` signature needs the right command-level
        info (venue_order_id, trade_id, commission, liquidity_side, …) which
        is most naturally wired when the integration test exercises it
        end-to-end.
        """
        raise NotImplementedError(
            "Wire to self.generate_order_filled(strategy_id, instrument_id, "
            "client_order_id, venue_order_id, venue_position_id, trade_id, "
            "order_side, order_type, last_qty, last_px, quote_currency, "
            "commission, liquidity_side, ts_event, info=None) per §12.5"
        )
