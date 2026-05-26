"""Tests for CatalogLiveDataClient in fast-replay mode."""
from __future__ import annotations

import asyncio
import inspect
from typing import Callable

import pytest

from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity

from data.instruments import VENUE_MIC
from livesim.data_client import CatalogLiveDataClient, replay_bars


def _instrument(ticker: str) -> Equity:
    return Equity(
        instrument_id=InstrumentId.from_str(f"{ticker}.{VENUE_MIC}"),
        raw_symbol=Symbol(ticker), currency=USD, price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        ts_event=0, ts_init=0,
    )


def _bars(ticker: str, n: int, start_ts_ns: int = 1_700_000_000_000_000_000) -> list[Bar]:
    bar_type = BarType.from_str(f"{ticker}.{VENUE_MIC}-1-DAY-LAST-EXTERNAL")
    one_day_ns = 86_400_000_000_000
    out: list[Bar] = []
    for i in range(n):
        ts = start_ts_ns + i * one_day_ns
        p = Price.from_str(f"{100.0 + i:.2f}")
        out.append(Bar(bar_type=bar_type, open=p, high=p, low=p, close=p,
                       volume=Quantity.from_int(1_000_000),
                       ts_event=ts, ts_init=ts))
    return out


def test_constructor_exposes_bar_provider_and_poll_seconds():
    """CatalogLiveDataClient.__init__ must accept bar_provider + poll_seconds."""
    sig = inspect.signature(CatalogLiveDataClient.__init__)
    assert "bar_provider" in sig.parameters
    assert "poll_seconds" in sig.parameters
    assert sig.parameters["poll_seconds"].default is None


def test_replay_bars_sorts_by_ts_event():
    """Pure helper: emits bars in ascending ts_event order regardless of input order."""
    bars = _bars("A", 4)
    shuffled = [bars[2], bars[0], bars[3], bars[1]]
    emitted: list[Bar] = []
    replay_bars(shuffled, emit_fn=emitted.append)
    assert [b.ts_event for b in emitted] == sorted(b.ts_event for b in bars)


def test_replay_bars_emits_each_bar_once():
    """Pure helper: each input bar is emitted exactly once."""
    bars = _bars("A", 3)
    emitted: list[Bar] = []
    replay_bars(bars, emit_fn=emitted.append)
    assert len(emitted) == 3
    assert emitted == bars


def test_replay_once_emits_only_new_bars_per_ticker():
    """Calling replay logic twice with the SAME bars should not double-emit."""
    bars = _bars("A", 3)
    emitted: list[Bar] = []
    emitted_ts: set[int] = set()

    def emit_if_new(bar: Bar) -> None:
        if bar.ts_event not in emitted_ts:
            emitted.append(bar)
            emitted_ts.add(bar.ts_event)

    replay_bars(bars, emit_fn=emit_if_new)
    replay_bars(bars, emit_fn=emit_if_new)  # same bars, second time
    assert len(emitted) == 3


def test_replay_once_picks_up_newly_appeared_bars():
    """If the provider returns more bars on the second call, only the deltas are emitted."""
    bars_initial = _bars("A", 3)
    bars_extended = _bars("A", 5)  # bars 4,5 are new (indices 3,4)
    emitted: list[Bar] = []
    emitted_ts: set[int] = set()

    def emit_if_new(bar: Bar) -> None:
        if bar.ts_event not in emitted_ts:
            emitted.append(bar)
            emitted_ts.add(bar.ts_event)

    replay_bars(bars_initial, emit_fn=emit_if_new)
    assert len(emitted) == 3
    replay_bars(bars_extended, emit_fn=emit_if_new)
    assert len(emitted) == 5
    # Last two emitted are the deltas (indices 3, 4 of bars_extended).
    assert emitted[-2:] == bars_extended[-2:]
