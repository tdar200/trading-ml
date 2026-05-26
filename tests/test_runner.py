"""Integration test for runner.run against synthetic bar data."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity

from analysis.runner import run
from analysis.runspec import RunResult, RunSpec
from data.instruments import VENUE_MIC
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy


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


def _grow(rate: float, n: int) -> list[float]:
    out = [100.0]
    for _ in range(n - 1):
        out.append(out[-1] * (1.0 + rate))
    return out


def test_runner_returns_RunResult_with_metrics_and_equity():
    n = 380
    start_ts_ns = 1_577_836_800_000_000_000  # 2020-01-01 UTC
    closes_by_ticker = {"A": _grow(0.0020, n), "B": _grow(0.0010, n), "BENCH": _grow(0.0005, n)}
    bars_by_ticker = {t: _bars(t, c, start_ts_ns) for t, c in closes_by_ticker.items()}
    instruments = [_instrument(t) for t in bars_by_ticker]

    spec = RunSpec(
        label="runner-test",
        strategy_class=MomentumTopNStrategy,
        strategy_config=MomentumTopNConfig(
            universe=("A", "B"), benchmark="BENCH",
            top_n=1, lookback_days=63, skip_days=21,
        ),
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2021, 1, 17, tzinfo=timezone.utc),  # ~380 days
        slippage_prob=0.0,  # deterministic for the test
    )

    result = run(spec, bars_by_ticker=bars_by_ticker, instruments=instruments)
    assert isinstance(result, RunResult)
    for key in ("cagr", "sharpe", "max_drawdown", "total_return",
                "final_equity", "n_days", "n_trades"):
        assert key in result.metrics, f"missing metric: {key}"
    assert len(result.equity_curve) > 0
    assert result.metrics["final_equity"] > 0
    # With top_n=1 and ticker A clearly the strongest momentum, the strategy held
    # A and grew with it → final_equity should be above 100k.
    assert result.metrics["final_equity"] > 100_000


def test_runner_filters_bars_by_date_window():
    """Bars outside [spec.start, spec.end) are excluded — different windows on
    the SAME bars produce different n_days."""
    n = 380
    start_ts_ns = 1_577_836_800_000_000_000  # 2020-01-01
    closes_by_ticker = {"A": _grow(0.0020, n), "B": _grow(0.0010, n), "BENCH": _grow(0.0005, n)}
    bars_by_ticker = {t: _bars(t, c, start_ts_ns) for t, c in closes_by_ticker.items()}
    instruments = [_instrument(t) for t in bars_by_ticker]

    config = MomentumTopNConfig(universe=("A", "B"), benchmark="BENCH",
                                top_n=1, lookback_days=63, skip_days=21)
    short_spec = RunSpec(
        label="short", strategy_class=MomentumTopNStrategy, strategy_config=config,
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2020, 7, 1, tzinfo=timezone.utc),
        slippage_prob=0.0,
    )
    long_spec = RunSpec(
        label="long", strategy_class=MomentumTopNStrategy, strategy_config=config,
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2021, 1, 17, tzinfo=timezone.utc),
        slippage_prob=0.0,
    )
    r_short = run(short_spec, bars_by_ticker=bars_by_ticker, instruments=instruments)
    r_long = run(long_spec, bars_by_ticker=bars_by_ticker, instruments=instruments)
    assert r_long.metrics["n_days"] > r_short.metrics["n_days"]
