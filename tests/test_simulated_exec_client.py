"""Tests for SimulatedLiveExecutionClient.

The exec client's order-intake + fill-emission cycle is integration-tested
via Task 8. This task unit-tests the pure compute_fill_price helper.
"""
from __future__ import annotations

from decimal import Decimal

from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity

from data.instruments import VENUE_MIC
from livesim.exec_client import compute_fill_price


def _bar(ticker: str, open_price: float) -> Bar:
    bar_type = BarType.from_str(f"{ticker}.{VENUE_MIC}-1-DAY-LAST-EXTERNAL")
    p = Price.from_str(f"{open_price:.2f}")
    return Bar(bar_type=bar_type, open=p, high=p, low=p, close=p,
               volume=Quantity.from_int(1_000_000),
               ts_event=1_700_000_000_000_000_000, ts_init=1_700_000_000_000_000_000)


def test_compute_fill_price_no_slippage_returns_bar_open():
    bar = _bar("A", 100.00)
    px = compute_fill_price(bar, side=OrderSide.BUY, slippage_ticks=0, tick_size=Decimal("0.01"))
    assert px == Decimal("100.00")


def test_compute_fill_price_buy_with_one_tick_slippage_adds_tick():
    """Buy side slips UP by one tick (worse price for buyer)."""
    bar = _bar("A", 100.00)
    px = compute_fill_price(bar, side=OrderSide.BUY, slippage_ticks=1, tick_size=Decimal("0.01"))
    assert px == Decimal("100.01")


def test_compute_fill_price_sell_with_one_tick_slippage_subtracts_tick():
    """Sell side slips DOWN by one tick (worse price for seller)."""
    bar = _bar("A", 100.00)
    px = compute_fill_price(bar, side=OrderSide.SELL, slippage_ticks=1, tick_size=Decimal("0.01"))
    assert px == Decimal("99.99")


def test_compute_fill_price_three_tick_slippage():
    bar = _bar("A", 50.20)
    buy_px = compute_fill_price(bar, side=OrderSide.BUY, slippage_ticks=3, tick_size=Decimal("0.01"))
    assert buy_px == Decimal("50.23")
    sell_px = compute_fill_price(bar, side=OrderSide.SELL, slippage_ticks=3, tick_size=Decimal("0.01"))
    assert sell_px == Decimal("50.17")
