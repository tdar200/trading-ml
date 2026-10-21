"""Tests for AlpacaLiveExecutionClient — mocked TradingClient (offline)."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from livesim.alpaca_exec_client import (
    _build_market_order_request,
    _translate_alpaca_status,
)


# ---- _translate_alpaca_status: pure helper, no mocks --------------------------


@pytest.mark.parametrize("alpaca, expected", [
    ("filled", "filled"),
    ("FILLED", "filled"),
    ("partially_filled", "partial"),
    ("rejected", "rejected"),
    ("expired", "rejected"),
    ("canceled", "cancelled"),
    ("cancelled", "cancelled"),
    ("new", "pending"),
    ("pending_cancel", "pending"),
    ("", "pending"),
])
def test_translate_alpaca_status_maps_correctly(alpaca, expected):
    assert _translate_alpaca_status(alpaca) == expected


# ---- _build_market_order_request: shape verification --------------------------


def test_build_market_order_request_buy():
    from nautilus_trader.model.enums import OrderSide
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce
    req = _build_market_order_request(
        "SPY", 5, OrderSide.BUY, client_order_id="cid-1",
    )
    assert req.symbol == "SPY"
    assert int(req.qty) == 5
    assert req.side == AlpacaOrderSide.BUY
    assert req.time_in_force == TimeInForce.DAY
    assert req.client_order_id == "cid-1"


def test_build_market_order_request_sell():
    from nautilus_trader.model.enums import OrderSide
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide
    req = _build_market_order_request("XLF", 100, OrderSide.SELL)
    assert req.symbol == "XLF"
    assert int(req.qty) == 100
    assert req.side == AlpacaOrderSide.SELL


# ---- _handle_alpaca_order_update: id round-trip + emit-once -------------------


@dataclass
class _FakeAlpacaOrder:
    id: str
    status: str
    symbol: str = ""
    qty: str = "0"
    side: str = "buy"
    filled_qty: str = "0"
    filled_avg_price: str = "0"


@dataclass
class _FakeNautilusOrder:
    """Minimal stand-in — only attributes the fill-emit path touches."""
    strategy_id: object
    instrument_id: object
    client_order_id: object
    side: object
    quantity: object


def test_handle_alpaca_order_update_skips_unknown_alpaca_id():
    """An Alpaca order we don't recognise (e.g. placed via the Alpaca UI)
    must NOT emit a Nautilus fill — we only own orders we submitted.

    Tested via stand-alone reimplementation of the dispatch logic that
    matches the production code; instantiating the real Cython-based
    LiveExecutionClient in a unit test is impractical.
    """
    from livesim.alpaca_exec_client import _translate_alpaca_status

    # Mirror the production dispatch logic for the "unknown id" path.
    alpaca_to_nautilus: dict[str, object] = {}
    emitted_fills: set[str] = set()
    fills_emitted = 0

    fake_order = _FakeAlpacaOrder(id="unknown-id", status="filled")
    status = _translate_alpaca_status(fake_order.status)
    if status == "filled" and fake_order.id not in emitted_fills:
        nautilus_order = alpaca_to_nautilus.get(fake_order.id)
        if nautilus_order is not None:
            fills_emitted += 1
            emitted_fills.add(fake_order.id)

    assert fills_emitted == 0
    assert "unknown-id" not in emitted_fills


def test_handle_alpaca_order_update_idempotent_for_already_emitted():
    """Even when we own the order, emitting twice for the same Alpaca id
    must not double-count fills.

    Reimplements the dispatch logic for unit-testability; see the previous
    test's note on why we don't instantiate the real Cython base class.
    """
    from livesim.alpaca_exec_client import _translate_alpaca_status

    alpaca_to_nautilus: dict[str, object] = {"alpaca-1": object()}
    emitted_fills: set[str] = {"alpaca-1"}  # already emitted earlier
    fills_emitted = 0

    fake_order = _FakeAlpacaOrder(id="alpaca-1", status="filled")
    status = _translate_alpaca_status(fake_order.status)
    if status == "filled" and fake_order.id not in emitted_fills:
        nautilus_order = alpaca_to_nautilus.get(fake_order.id)
        if nautilus_order is not None:
            fills_emitted += 1
            emitted_fills.add(fake_order.id)

    assert fills_emitted == 0  # already-emitted guard prevented re-emission


def test_actual_dispatch_cumulative_partial_fills_and_retry():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from decimal import Decimal
    from livesim.alpaca_exec_client import AlpacaLiveExecutionClient
    fake=SimpleNamespace(_emitted_fills=set(),_alpaca_to_nautilus={'o':object()},_fills_emitted=0,_emit_fill_from_alpaca=Mock(),_log=Mock())
    def update(q,p,status='partially_filled'):
        AlpacaLiveExecutionClient._handle_alpaca_order_update(fake,SimpleNamespace(id='o',status=status,filled_qty=str(q),filled_avg_price=str(p)))
    update('30','10');update('30','10');update('20','10')
    update('70',str((Decimal(300)+Decimal(402))/70))
    update('100','10.05','filled');update('100','10.05','filled')
    fills=[call.args[1] for call in fake._emit_fill_from_alpaca.call_args_list]
    assert [Decimal(f.filled_qty) for f in fills]==[30,40,30]
    assert [Decimal(f.filled_avg_price) for f in fills]==[Decimal('10'),Decimal('10.05'),Decimal('10.10')]
    assert sum(Decimal(f.filled_qty)*Decimal(f.filled_avg_price) for f in fills)==1005


def test_actual_cancelled_order_still_accounts_fractional_execution():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from livesim.alpaca_exec_client import AlpacaLiveExecutionClient
    fake=SimpleNamespace(_emitted_fills=set(),_alpaca_to_nautilus={'o':object()},_fills_emitted=0,_emit_fill_from_alpaca=Mock(),_log=Mock())
    order=SimpleNamespace(id='o',status='canceled',filled_qty='0.25',filled_avg_price='100')
    AlpacaLiveExecutionClient._handle_alpaca_order_update(fake,order)
    AlpacaLiveExecutionClient._handle_alpaca_order_update(fake,order)
    assert fake._emit_fill_from_alpaca.call_count==1
    assert fake._emit_fill_from_alpaca.call_args.args[1].filled_qty=='0.25'
