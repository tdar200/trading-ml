"""
Integration tests for MomentumTopNStrategy via a synthetic BacktestEngine.

These tests build a tiny engine with fake instruments and hand-crafted bars
so the strategy logic can be exercised end-to-end without depending on the
real Stage-1 catalog or real Alpaca data.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Money, Price, Quantity

from data.instruments import VENUE_MIC
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy


# ---- synthetic data helpers ----

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
    """Build daily bars: one bar per day, OHLC = close, volume = 1_000_000."""
    bar_type = BarType.from_str(f"{ticker}.{VENUE_MIC}-1-DAY-LAST-EXTERNAL")
    one_day_ns = 86_400_000_000_000
    bars: list[Bar] = []
    for i, c in enumerate(closes):
        ts = start_ts_ns + i * one_day_ns
        p = Price.from_str(f"{c:.2f}")
        bars.append(Bar(
            bar_type=bar_type, open=p, high=p, low=p, close=p,
            volume=Quantity.from_int(1_000_000),
            ts_event=ts, ts_init=ts,
        ))
    return bars


@dataclass
class SyntheticBacktest:
    engine: BacktestEngine
    strategy: MomentumTopNStrategy
    venue: Venue


def _run_synthetic_backtest(
    closes_by_ticker: dict[str, list[float]],
    benchmark_closes: list[float],
    *,
    top_n: int = 3,
    lookback_days: int = 126,
    skip_days: int = 21,
    starting_cash: int = 100_000,
    start_ts_ns: int = 1_700_000_000_000_000_000,
) -> SyntheticBacktest:
    """Build + run a BacktestEngine with the given synthetic data; return engine + strategy."""
    universe = tuple(closes_by_ticker.keys())
    config = BacktestEngineConfig(
        trader_id=TraderId("TEST-001"),
        logging=LoggingConfig(log_level="WARNING"),
    )
    engine = BacktestEngine(config=config)
    venue = Venue(VENUE_MIC)
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD,
        starting_balances=[Money(starting_cash, USD)],
        fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=42),
    )
    # Instruments + bars for tradeable universe + benchmark
    for t in universe:
        engine.add_instrument(_instrument(t))
        engine.add_data(_bars(t, closes_by_ticker[t], start_ts_ns))
    engine.add_instrument(_instrument("BENCH"))
    engine.add_data(_bars("BENCH", benchmark_closes, start_ts_ns))

    strat = MomentumTopNStrategy(config=MomentumTopNConfig(
        universe=universe,
        benchmark="BENCH",
        top_n=top_n,
        lookback_days=lookback_days,
        skip_days=skip_days,
    ))
    engine.add_strategy(strat)
    engine.run()
    return SyntheticBacktest(engine=engine, strategy=strat, venue=venue)


# ---- buffer tests (Task 5) ----

def test_ring_buffer_appends_closes_per_ticker():
    closes = {"A": [100.0 + i for i in range(30)],
              "B": [50.0 + i for i in range(30)]}
    bt = _run_synthetic_backtest(
        closes_by_ticker=closes,
        benchmark_closes=[200.0] * 30,
        top_n=1, lookback_days=5, skip_days=1,
    )
    assert "A" in bt.strategy._closes
    assert "B" in bt.strategy._closes
    assert "BENCH" in bt.strategy._closes
    assert bt.strategy._closes["A"][-1] == pytest.approx(100.0 + 29)
    assert bt.strategy._closes["B"][-1] == pytest.approx(50.0 + 29)


def test_ring_buffer_is_size_bounded():
    # With lookback=5, skip=1, the buffer cap should be ~11 entries; well below 30.
    closes = {"A": [100.0 + i for i in range(30)]}
    bt = _run_synthetic_backtest(
        closes_by_ticker=closes,
        benchmark_closes=[200.0] * 30,
        top_n=1, lookback_days=5, skip_days=1,
    )
    # The buffer must keep at least the (lookback + skip + 1) most recent closes;
    # it must NOT grow unbounded across many bars.
    assert len(bt.strategy._closes["A"]) <= 5 + 1 + 5  # generous upper bound (=11)
    assert bt.strategy._closes["A"][-1] == pytest.approx(129.0)


# ---- rebalance trigger tests (Task 6) ----

def _build_warmup_passing_closes(n_days: int) -> list[float]:
    """Trivial closes long enough to pass any reasonable warm-up window."""
    return [100.0 + 0.01 * i for i in range(n_days)]


def test_no_rebalance_during_warmup():
    # 100 days is < lookback(126) + skip(21) = 147 — never warmed up.
    closes = {"A": _build_warmup_passing_closes(100),
              "B": _build_warmup_passing_closes(100)}
    bt = _run_synthetic_backtest(
        closes_by_ticker=closes,
        benchmark_closes=[200.0] * 100,
        top_n=1,  # default lookback_days=126, skip_days=21
    )
    assert bt.strategy._rebalance_count == 0


def test_rebalance_fires_monthly_after_warmup():
    # ~380 days of daily bars ensures we pass warm-up by day ~147 and then see
    # multiple month rollovers in the remaining window.
    closes = {"A": _build_warmup_passing_closes(380),
              "B": _build_warmup_passing_closes(380),
              "C": _build_warmup_passing_closes(380)}
    bt = _run_synthetic_backtest(
        closes_by_ticker=closes,
        benchmark_closes=[200.0] * 380,
        top_n=2,
    )
    # We expect rebalances to fire roughly once per calendar month after warm-up.
    # Synthetic timestamps advance 1 calendar day per bar.
    # 380 calendar days ≈ 12-13 month rollovers; first ~147 days are warm-up.
    # Loose bounds — testing "happens monthly", not exact count.
    assert 6 <= bt.strategy._rebalance_count <= 12, (
        f"expected ~monthly rebalances after warm-up, got {bt.strategy._rebalance_count}"
    )


def test_warmup_requires_every_tradeable_ticker():
    # A has full history, B is short. Warm-up should NEVER complete — no rebalance fires.
    closes = {"A": _build_warmup_passing_closes(380),
              "B": _build_warmup_passing_closes(80)}   # B is short → no warmup
    bt = _run_synthetic_backtest(
        closes_by_ticker=closes,
        benchmark_closes=[200.0] * 380,
        top_n=1,
    )
    assert bt.strategy._rebalance_count == 0


# ---- position sizing tests (Task 7) ----

def _held_tickers(bt) -> set[str]:
    """Return the set of tickers with a nonzero open position at end of backtest."""
    held: set[str] = set()
    for t in bt.strategy.config.universe:
        instr_id = InstrumentId.from_str(f"{t}.{VENUE_MIC}")
        positions = bt.engine.cache.positions_open(instrument_id=instr_id)
        if positions and any(p.signed_qty != 0 for p in positions):
            held.add(t)
    return held


def test_top_n_positions_held_after_first_rebalance():
    """With known-ranked synthetic data, the strategy should hold the top-N tickers."""
    # 380 daily closes so warm-up completes by ~day 147 and we get ~8 monthly rebalances.
    # Build closes such that A > B > C > D > E by 6-month-skip-1 return.
    # Different daily growth rates produce a clear, stable ranking.
    def grow(rate: float) -> list[float]:
        out = [100.0]
        for _ in range(379):
            out.append(out[-1] * (1.0 + rate))
        return out

    closes = {
        "A": grow(0.0020),  # fastest
        "B": grow(0.0015),
        "C": grow(0.0010),
        "D": grow(0.0005),
        "E": grow(0.0000),  # slowest
    }
    bt = _run_synthetic_backtest(
        closes_by_ticker=closes,
        benchmark_closes=[200.0] * 380,
        top_n=3,
    )

    # After the final rebalance, the strategy should hold A, B, C (the top 3).
    held = _held_tickers(bt)
    assert held == {"A", "B", "C"}, f"expected top-3 {{A,B,C}}, got {held}"


def test_holdings_rotate_when_ranks_flip():
    """When the leading ticker changes after warm-up, holdings should rotate."""
    # Build 380 daily closes for 5 tickers. First half: C leads; second half: E leads.
    n = 380
    half = n // 2

    def build(early_rate: float, late_rate: float) -> list[float]:
        out = [100.0]
        for i in range(n - 1):
            r = early_rate if i < half else late_rate
            out.append(out[-1] * (1.0 + r))
        return out

    closes = {
        "A": build(0.0005, 0.0005),
        "B": build(0.0005, 0.0005),
        "C": build(0.0020, 0.0000),   # leader early, flat late
        "D": build(0.0005, 0.0005),
        "E": build(0.0000, 0.0030),   # flat early, leader late
    }
    bt = _run_synthetic_backtest(
        closes_by_ticker=closes,
        benchmark_closes=[200.0] * n,
        top_n=1,  # narrow the test — we only care which single name leads
    )

    held = _held_tickers(bt)
    assert "E" in held, f"expected new leader E to be held, got {held}"
    assert "C" not in held, f"expected old leader C to be dropped, got {held}"
