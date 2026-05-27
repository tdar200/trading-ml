"""Integration tests for MomentumMetaStrategy on synthetic catalogs.

Uses a stub ModelStore that returns a controllable mock model so the
filter+sizer logic can be exercised without training a real LightGBM.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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
from strategies.momentum_meta import MomentumMetaConfig, MomentumMetaStrategy


# ---- synthetic-data helpers (mirror tests/test_momentum_strategy.py) -------

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


@dataclass
class _StubModel:
    """Mock LGBMClassifier: returns a fixed P(win) for all rows.

    The strategy calls `predict_proba(X)` which returns shape (n, 2) where
    column 1 = P(y=1). We mimic that.
    """
    p_win: float

    def predict_proba(self, X) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.p_win), np.full(n, self.p_win)])


def _write_stub_store(tmp_path: Path, *, label: str, cutoff: pd.Timestamp,
                     feature_names: list[str], p_win: float) -> Path:
    """Write a joblib pickle the ModelStore can load."""
    import joblib
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    path = cache / f"{label}_{cutoff.strftime('%Y-%m-%d')}.joblib"
    joblib.dump({"model": _StubModel(p_win=p_win),
                 "feature_names": list(feature_names)}, path)
    return cache


def _run_meta_backtest(*, closes_by_ticker, benchmark_closes, p_win, filter_threshold,
                      tmp_path, top_n=3):
    """Run a synthetic backtest of MomentumMetaStrategy with a stub model."""
    universe = tuple(closes_by_ticker.keys())
    start_ts_ns = 1_577_836_800_000_000_000

    cache_dir = _write_stub_store(
        tmp_path, label="default", cutoff=pd.Timestamp("2020-01-01"),
        feature_names=[
            "mom_6m", "mom_1m", "mom_3m_skip_1m",
            "vol_20d", "vol_60d",
            "rank_mom_6m", "rank_vol_60d",
            "spy_mom_6m", "spy_vol_60d",
            "month_sin", "month_cos",
        ],
        p_win=p_win,
    )

    config = BacktestEngineConfig(
        trader_id=TraderId("META-TEST-001"),
        logging=LoggingConfig(log_level="WARNING"),
    )
    engine = BacktestEngine(config=config)
    venue = Venue(VENUE_MIC)
    engine.add_venue(
        venue=venue, oms_type=OmsType.NETTING, account_type=AccountType.CASH,
        base_currency=USD, starting_balances=[Money(100_000, USD)],
        fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=42),
    )
    for t in universe:
        engine.add_instrument(_instrument(t))
        engine.add_data(_bars(t, closes_by_ticker[t], start_ts_ns))
    engine.add_instrument(_instrument("BENCH"))
    engine.add_data(_bars("BENCH", benchmark_closes, start_ts_ns))

    strat = MomentumMetaStrategy(config=MomentumMetaConfig(
        universe=universe, benchmark="BENCH",
        top_n=top_n, lookback_days=63, skip_days=0,
        model_store_path=str(cache_dir),
        model_store_label="default",
        filter_threshold=filter_threshold,
    ))
    engine.add_strategy(strat)
    engine.run()
    return strat, engine


def _held_count(strat, engine, universe) -> int:
    """Count of tickers with nonzero open position at end of backtest."""
    held = 0
    for t in universe:
        instr_id = InstrumentId.from_str(f"{t}.{VENUE_MIC}")
        positions = engine.cache.positions_open(instrument_id=instr_id)
        if positions and any(float(p.signed_qty) != 0 for p in positions):
            held += 1
    return held


# ---- tests -----------------------------------------------------------------

def test_filter_threshold_zero_with_pwin_one_holds_top_n(tmp_path: Path):
    """With p_win=1.0 (model always confident) and threshold=0, behavior is equivalent
    to baseline: holds top-N positions."""
    n = 380
    closes = {
        "A": _grow(0.0020, n), "B": _grow(0.0015, n),
        "C": _grow(0.0010, n), "D": _grow(0.0005, n), "E": _grow(0.0000, n),
    }
    strat, engine = _run_meta_backtest(
        closes_by_ticker=closes, benchmark_closes=[200.0] * n,
        p_win=1.0, filter_threshold=0.0, tmp_path=tmp_path, top_n=3,
    )
    held = _held_count(strat, engine, closes.keys())
    assert held == 3


def test_filter_threshold_above_pwin_holds_nothing(tmp_path: Path):
    """When the model's p_win is BELOW the filter threshold, all positions filtered."""
    n = 380
    closes = {
        "A": _grow(0.0020, n), "B": _grow(0.0015, n),
        "C": _grow(0.0010, n), "D": _grow(0.0005, n), "E": _grow(0.0000, n),
    }
    strat, engine = _run_meta_backtest(
        closes_by_ticker=closes, benchmark_closes=[200.0] * n,
        p_win=0.3, filter_threshold=0.5, tmp_path=tmp_path, top_n=3,
    )
    held = _held_count(strat, engine, closes.keys())
    assert held == 0


def test_no_model_falls_back_to_baseline(tmp_path: Path):
    """If the ModelStore has no model strictly-before the rebalance date, the
    strategy holds the baseline top-N (no filtering)."""
    n = 380
    closes = {
        "A": _grow(0.0020, n), "B": _grow(0.0015, n),
        "C": _grow(0.0010, n), "D": _grow(0.0005, n), "E": _grow(0.0000, n),
    }
    # Don't create any stub artifacts; pass an empty cache_dir.
    cache_dir = tmp_path / "empty_cache"
    cache_dir.mkdir()

    start_ts_ns = 1_577_836_800_000_000_000
    config = BacktestEngineConfig(
        trader_id=TraderId("META-TEST-002"),
        logging=LoggingConfig(log_level="WARNING"),
    )
    engine = BacktestEngine(config=config)
    venue = Venue(VENUE_MIC)
    engine.add_venue(
        venue=venue, oms_type=OmsType.NETTING, account_type=AccountType.CASH,
        base_currency=USD, starting_balances=[Money(100_000, USD)],
        fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=42),
    )
    universe = tuple(closes.keys())
    for t in universe:
        engine.add_instrument(_instrument(t))
        engine.add_data(_bars(t, closes[t], start_ts_ns))
    engine.add_instrument(_instrument("BENCH"))
    engine.add_data(_bars("BENCH", [200.0] * n, start_ts_ns))

    strat = MomentumMetaStrategy(config=MomentumMetaConfig(
        universe=universe, benchmark="BENCH",
        top_n=3, lookback_days=63, skip_days=0,
        model_store_path=str(cache_dir),
        model_store_label="default",
        filter_threshold=0.55,
    ))
    engine.add_strategy(strat)
    engine.run()

    held = _held_count(strat, engine, universe)
    # No model available → baseline behavior → hold top-3.
    assert held == 3


def test_non_midnight_timestamps_still_trade(tmp_path: Path):
    """Real catalog bars have ts_event at NYSE close (non-midnight UTC). The
    strategy must still extract features and trade — regression test for the
    timestamp-normalization bug found in Stage-5 final review."""
    n = 380
    closes = {
        "A": _grow(0.0020, n), "B": _grow(0.0015, n),
        "C": _grow(0.0010, n), "D": _grow(0.0005, n), "E": _grow(0.0000, n),
    }

    # NYSE close ≈ 21:00 EST = 02:00 UTC (winter). Pick 04:00 UTC for clear
    # non-midnight alignment.
    start_ts_ns = 1_577_851_200_000_000_000  # 2020-01-01 04:00:00 UTC

    cache_dir = _write_stub_store(
        tmp_path, label="default", cutoff=pd.Timestamp("2020-01-01"),
        feature_names=[
            "mom_6m", "mom_1m", "mom_3m_skip_1m",
            "vol_20d", "vol_60d",
            "rank_mom_6m", "rank_vol_60d",
            "spy_mom_6m", "spy_vol_60d",
            "month_sin", "month_cos",
        ],
        p_win=1.0,  # always confident
    )

    config = BacktestEngineConfig(
        trader_id=TraderId("META-TEST-003"),
        logging=LoggingConfig(log_level="WARNING"),
    )
    engine = BacktestEngine(config=config)
    venue = Venue(VENUE_MIC)
    engine.add_venue(
        venue=venue, oms_type=OmsType.NETTING, account_type=AccountType.CASH,
        base_currency=USD, starting_balances=[Money(100_000, USD)],
        fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=42),
    )
    universe = tuple(closes.keys())
    for t in universe:
        engine.add_instrument(_instrument(t))
        engine.add_data(_bars(t, closes[t], start_ts_ns))
    engine.add_instrument(_instrument("BENCH"))
    engine.add_data(_bars("BENCH", [200.0] * n, start_ts_ns))

    strat = MomentumMetaStrategy(config=MomentumMetaConfig(
        universe=universe, benchmark="BENCH",
        top_n=3, lookback_days=63, skip_days=0,
        model_store_path=str(cache_dir),
        model_store_label="default",
        filter_threshold=0.0,  # don't filter — exercise the model path
    ))
    engine.add_strategy(strat)
    engine.run()

    held = _held_count(strat, engine, universe)
    # With p_win=1.0 and threshold=0.0, the meta strategy should trade
    # exactly like the baseline (top-3). If it doesn't, the model path is
    # silently failing (likely a timestamp-alignment bug like the one
    # found in Stage-5 final review).
    assert held == 3, f"meta strategy not trading on non-midnight timestamps; held={held}"
