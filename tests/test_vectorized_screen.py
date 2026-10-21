"""GPU/CPU vector pre-screen correctness and canonical parity tests."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity

from analysis.runner import run
from analysis.runspec import RunSpec
from analysis.vectorized_screen import (
    bars_to_close_frame,
    resolve_device,
    screen_momentum,
    select_diverse_finalists,
)
from data.instruments import VENUE_MIC
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy


def _closes(n=520):
    index = pd.bdate_range("2020-01-01", periods=n)
    phase = np.arange(n)
    a_returns = 0.0015 + 0.0008 * np.sin(phase / 11)
    b_returns = 0.0004 + 0.0005 * np.cos(phase / 17)
    return pd.DataFrame({
        "A": 100 * np.cumprod(1 + a_returns),
        "B": 100 * np.cumprod(1 + b_returns),
    }, index=index)


CANDIDATES = [
    {"top_n": 1, "lookback_days": 63, "skip_days": 0},
    {"top_n": 2, "lookback_days": 63, "skip_days": 0},
]


def test_screen_batches_candidates_and_prefers_stronger_single_asset():
    result = screen_momentum(_closes(), CANDIDATES, device="cpu", turnover_cost_bps=0)
    assert len(result) == 2
    top1 = result[result["top_n"] == 1].iloc[0]
    top2 = result[result["top_n"] == 2].iloc[0]
    assert top1["screen_final_equity"] > top2["screen_final_equity"]
    assert set(result["screen_device"]) == {"cpu"}
    assert (result["screen_trades"] > 0).all()


def test_turnover_cost_never_improves_equity():
    free = screen_momentum(_closes(), CANDIDATES, device="cpu", turnover_cost_bps=0)
    costly = screen_momentum(_closes(), CANDIDATES, device="cpu", turnover_cost_bps=50)
    by_params = ["top_n", "lookback_days", "skip_days"]
    merged = free.merge(costly, on=by_params, suffixes=("_free", "_costly"))
    assert (merged["screen_final_equity_costly"] <= merged["screen_final_equity_free"]).all()


def test_candidate_without_warmup_is_reported_not_silently_scored():
    candidate = [{"top_n": 1, "lookback_days": 999, "skip_days": 0}]
    result = screen_momentum(_closes(100), candidate, device="cpu")
    assert np.isnan(result.iloc[0]["screen_sharpe"])
    assert result.iloc[0]["screen_trades"] == 0


def test_invalid_candidate_and_forced_unavailable_cuda_fail_closed():
    with pytest.raises(ValueError, match="invalid top_n"):
        screen_momentum(_closes(), [{"top_n": 3, "lookback_days": 20, "skip_days": 0}], device="cpu")
    if resolve_device("auto") == "cpu":
        with pytest.raises(RuntimeError, match="CUDA requested"):
            resolve_device("cuda")


def test_diverse_finalists_keep_best_and_distant_regions():
    screened = pd.DataFrame({
        "top_n": [1, 2, 3, 4, 5],
        "lookback_days": [42, 63, 84, 126, 252],
        "skip_days": [0, 0, 21, 21, 42],
        "screen_score": [1.00, 0.99, 0.98, 0.97, 0.96],
    })
    finalists = select_diverse_finalists(
        screened, ["top_n", "lookback_days", "skip_days"], 3,
    )
    assert len(finalists) == 3
    assert 1 in set(finalists["top_n"])
    assert finalists[["top_n", "lookback_days", "skip_days"]].drop_duplicates().shape[0] == 3


class _FakeBar:
    def __init__(self, ts, close):
        self.ts_event = ts
        self.close = close


def test_bars_to_close_frame_inner_aligns_dates_and_preserves_order():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True)
    ns = [int(item.value) for item in dates]
    bars = {
        "B": [_FakeBar(ns[0], 20), _FakeBar(ns[1], 21)],
        "A": [_FakeBar(ns[0], 10), _FakeBar(ns[1], 11)],
    }
    frame = bars_to_close_frame(bars, ("A", "B"))
    assert list(frame.columns) == ["A", "B"]
    assert frame.iloc[-1].tolist() == [11.0, 21.0]


def _instrument(ticker):
    return Equity(
        instrument_id=InstrumentId.from_str(f"{ticker}.{VENUE_MIC}"),
        raw_symbol=Symbol(ticker), currency=USD, price_precision=2,
        price_increment=Price.from_str("0.01"), lot_size=Quantity.from_int(1),
        ts_event=0, ts_init=0,
    )


def _bars(ticker, series):
    bar_type = BarType.from_str(f"{ticker}.{VENUE_MIC}-1-DAY-LAST-EXTERNAL")
    output = []
    for timestamp, close in series.items():
        ts = int(timestamp.tz_localize("UTC").value)
        price = Price.from_str(f"{close:.2f}")
        output.append(Bar(
            bar_type=bar_type, open=price, high=price, low=price, close=price,
            volume=Quantity.from_int(1_000_000), ts_event=ts, ts_init=ts,
        ))
    return output


def test_screen_winner_matches_canonical_nautilus_on_synthetic_market():
    closes = _closes(520)
    screened = screen_momentum(closes, CANDIDATES, device="cpu", turnover_cost_bps=0)
    benchmark = pd.Series(100 * np.cumprod(1 + np.full(len(closes), 0.0002)), index=closes.index)
    bars = {"A": _bars("A", closes["A"]), "B": _bars("B", closes["B"]), "BENCH": _bars("BENCH", benchmark)}
    instruments = [_instrument(ticker) for ticker in bars]
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2022, 1, 1, tzinfo=timezone.utc)
    canonical = []
    for candidate in CANDIDATES:
        result = run(
            RunSpec(
                label=f"screen-parity-{candidate['top_n']}",
                strategy_class=MomentumTopNStrategy,
                strategy_config=MomentumTopNConfig(
                    universe=("A", "B"), benchmark="BENCH", **candidate,
                ),
                start=start, end=end, slippage_prob=0, log_level="ERROR",
            ),
            bars_by_ticker=bars,
            instruments=instruments,
        )
        canonical.append((candidate["top_n"], result.metrics["final_equity"]))
    canonical_winner = max(canonical, key=lambda item: item[1])[0]
    screen_winner = screened.loc[screened["screen_final_equity"].idxmax(), "top_n"]
    assert screen_winner == canonical_winner == 1
