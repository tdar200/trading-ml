"""Cost stress tests on synthetic data."""
from __future__ import annotations

from datetime import datetime, timezone

from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity

from analysis.costs import cost_stress
from analysis.runspec import RunSpec
from data.instruments import VENUE_MIC
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy


def _instrument(ticker: str) -> Equity:
    return Equity(
        instrument_id=InstrumentId.from_str(f"{ticker}.{VENUE_MIC}"),
        raw_symbol=Symbol(ticker), currency=USD, price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        ts_event=0, ts_init=0,
    )


def _bars(ticker: str, closes: list[float], start_ts_ns: int) -> list[Bar]:
    bar_type = BarType.from_str(f"{ticker}.{VENUE_MIC}-1-DAY-LAST-EXTERNAL")
    one_day_ns = 86_400_000_000_000
    out: list[Bar] = []
    for i, c in enumerate(closes):
        ts = start_ts_ns + i * one_day_ns
        p = Price.from_str(f"{c:.2f}")
        out.append(Bar(bar_type=bar_type, open=p, high=p, low=p, close=p,
                       volume=Quantity.from_int(1_000_000),
                       ts_event=ts, ts_init=ts))
    return out


def _grow(rate: float, n: int) -> list[float]:
    out = [100.0]
    for _ in range(n - 1):
        out.append(out[-1] * (1.0 + rate))
    return out


def test_cost_stress_returns_one_row_per_slippage_level():
    n = 380
    start_ts_ns = 1_577_836_800_000_000_000  # 2020-01-01
    closes_by_ticker = {"A": _grow(0.002, n), "B": _grow(0.001, n), "BENCH": _grow(0.0005, n)}
    bars_by_ticker = {t: _bars(t, c, start_ts_ns) for t, c in closes_by_ticker.items()}
    instruments = [_instrument(t) for t in bars_by_ticker]

    base_spec = RunSpec(
        label="winner",
        strategy_class=MomentumTopNStrategy,
        strategy_config=MomentumTopNConfig(
            universe=("A", "B"), benchmark="BENCH",
            top_n=1, lookback_days=63, skip_days=21,
        ),
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2021, 1, 17, tzinfo=timezone.utc),
    )

    df = cost_stress(
        base_spec=base_spec,
        slippage_probs=[0.0, 0.25, 0.5, 1.0],
        bars_by_ticker=bars_by_ticker,
        instruments=instruments,
    )
    assert len(df) == 4
    assert list(df["slippage_prob"]) == [0.0, 0.25, 0.5, 1.0]
    for col in ("slippage_prob", "cagr", "sharpe", "max_drawdown",
                "total_return", "final_equity", "n_days", "n_trades"):
        assert col in df.columns
