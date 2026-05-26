"""Walk-forward harness tests on synthetic data."""
from __future__ import annotations

from datetime import datetime, timezone

from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity

from analysis.walk_forward import expand_param_grid, walk_forward
from data.instruments import VENUE_MIC
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy


def test_expand_param_grid_cross_product():
    grid = {"top_n": [2, 3], "lookback_days": [63, 126], "skip_days": [21]}
    combos = expand_param_grid(grid)
    assert len(combos) == 4
    assert {"top_n": 2, "lookback_days": 63, "skip_days": 21} in combos
    assert {"top_n": 3, "lookback_days": 126, "skip_days": 21} in combos


def test_expand_param_grid_empty_grid_returns_one_empty_combo():
    assert expand_param_grid({}) == [{}]


# ---- synthetic-data helpers ----

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


def test_walk_forward_produces_one_row_per_combo_per_window():
    n = 380
    start_ts_ns = 1_577_836_800_000_000_000  # 2020-01-01
    closes_by_ticker = {"A": _grow(0.002, n), "B": _grow(0.001, n), "BENCH": _grow(0.0005, n)}
    bars_by_ticker = {t: _bars(t, c, start_ts_ns) for t, c in closes_by_ticker.items()}
    instruments = [_instrument(t) for t in bars_by_ticker]

    grid = {"top_n": [1, 2], "lookback_days": [63], "skip_days": [21]}
    windows = [
        (datetime(2020, 1, 1, tzinfo=timezone.utc),
         datetime(2020, 7, 1, tzinfo=timezone.utc)),
        (datetime(2020, 7, 1, tzinfo=timezone.utc),
         datetime(2021, 1, 17, tzinfo=timezone.utc)),
    ]
    base = dict(universe=("A", "B"), benchmark="BENCH")

    df = walk_forward(
        strategy_class=MomentumTopNStrategy,
        config_class=MomentumTopNConfig,
        base_config_kwargs=base,
        param_grid=grid,
        windows=windows,
        bars_by_ticker=bars_by_ticker,
        instruments=instruments,
        slippage_prob=0.0,
    )

    assert len(df) == 4
    for col in ("window_idx", "start", "end", "top_n", "lookback_days",
                "skip_days", "cagr", "sharpe", "max_drawdown",
                "n_days", "n_trades", "final_equity"):
        assert col in df.columns
    assert set(df["window_idx"]) == {0, 1}
