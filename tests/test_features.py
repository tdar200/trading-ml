"""Tests for models.features — pure pandas, no Nautilus."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from models.features import (
    extract_features_for_date,
    make_features,
)


def _const_series(start_ts: str, n: int, val: float = 100.0) -> pd.Series:
    return pd.Series([val] * n, index=pd.date_range(start_ts, periods=n, freq="B"))


def _ramp_series(start_ts: str, n: int, daily_rate: float) -> pd.Series:
    """Compounding series: close[i] = 100 * (1 + daily_rate)**i."""
    vals = [100.0]
    for _ in range(n - 1):
        vals.append(vals[-1] * (1.0 + daily_rate))
    return pd.Series(vals, index=pd.date_range(start_ts, periods=n, freq="B"))


def _bars_universe(start_ts: str, n: int):
    """Build a 3-ticker universe + SPY with distinct daily rates."""
    return {
        "A": _ramp_series(start_ts, n, 0.0020),
        "B": _ramp_series(start_ts, n, 0.0010),
        "C": _ramp_series(start_ts, n, 0.0005),
        "SPY": _ramp_series(start_ts, n, 0.0007),
    }


# ---- mom_6m ----------------------------------------------------------------

def test_mom_6m_value_matches_hand_compute():
    bars = _bars_universe("2024-01-02", 200)
    date = bars["A"].index[150]  # day 150, has > 126 days of history
    feats = extract_features_for_date(bars, ticker="A", date=date, benchmark="SPY")
    expected = bars["A"].iloc[150] / bars["A"].iloc[150 - 126] - 1.0
    assert feats["mom_6m"] == pytest.approx(expected, rel=1e-6)


def test_mom_1m_value_matches_hand_compute():
    bars = _bars_universe("2024-01-02", 200)
    date = bars["A"].index[150]
    feats = extract_features_for_date(bars, ticker="A", date=date, benchmark="SPY")
    expected = bars["A"].iloc[150] / bars["A"].iloc[150 - 21] - 1.0
    assert feats["mom_1m"] == pytest.approx(expected, rel=1e-6)


def test_mom_3m_skip_1m_value_matches_hand_compute():
    bars = _bars_universe("2024-01-02", 200)
    date = bars["A"].index[150]
    feats = extract_features_for_date(bars, ticker="A", date=date, benchmark="SPY")
    # close[150-21] / close[150-21-63] - 1
    expected = bars["A"].iloc[150 - 21] / bars["A"].iloc[150 - 21 - 63] - 1.0
    assert feats["mom_3m_skip_1m"] == pytest.approx(expected, rel=1e-6)


# ---- vol_20d / vol_60d -----------------------------------------------------

def test_vol_20d_annualized_constant_series_is_zero():
    bars = {"A": _const_series("2024-01-02", 200), "SPY": _const_series("2024-01-02", 200)}
    date = bars["A"].index[150]
    feats = extract_features_for_date(bars, ticker="A", date=date, benchmark="SPY")
    assert feats["vol_20d"] == pytest.approx(0.0, abs=1e-10)


def test_vol_60d_positive_on_growth_series():
    bars = _bars_universe("2024-01-02", 200)
    date = bars["A"].index[150]
    feats = extract_features_for_date(bars, ticker="A", date=date, benchmark="SPY")
    assert feats["vol_60d"] > 0.0


# ---- cross-sectional ranks --------------------------------------------------

def test_rank_mom_6m_orders_universe_correctly():
    """A grows fastest → rank highest (1.0); C grows slowest → rank lowest (0.0)."""
    bars = _bars_universe("2024-01-02", 200)
    date = bars["A"].index[150]
    fa = extract_features_for_date(bars, ticker="A", date=date, benchmark="SPY")
    fb = extract_features_for_date(bars, ticker="B", date=date, benchmark="SPY")
    fc = extract_features_for_date(bars, ticker="C", date=date, benchmark="SPY")
    # Ranks should be ordered A > B > C; with 3 tradeable tickers + SPY in
    # universe (SPY excluded from rank): A=1.0, B=0.5, C=0.0.
    assert fa["rank_mom_6m"] > fb["rank_mom_6m"] > fc["rank_mom_6m"]
    assert fa["rank_mom_6m"] == pytest.approx(1.0)
    assert fc["rank_mom_6m"] == pytest.approx(0.0)


def test_rank_vol_60d_in_unit_interval():
    bars = _bars_universe("2024-01-02", 200)
    date = bars["A"].index[150]
    feats = extract_features_for_date(bars, ticker="A", date=date, benchmark="SPY")
    assert 0.0 <= feats["rank_vol_60d"] <= 1.0


# ---- SPY regime features ----------------------------------------------------

def test_spy_mom_6m_matches_spy_series():
    bars = _bars_universe("2024-01-02", 200)
    date = bars["A"].index[150]
    feats = extract_features_for_date(bars, ticker="A", date=date, benchmark="SPY")
    expected = bars["SPY"].iloc[150] / bars["SPY"].iloc[150 - 126] - 1.0
    assert feats["spy_mom_6m"] == pytest.approx(expected, rel=1e-6)


def test_spy_vol_60d_is_positive_on_growth_series():
    bars = _bars_universe("2024-01-02", 200)
    date = bars["A"].index[150]
    feats = extract_features_for_date(bars, ticker="A", date=date, benchmark="SPY")
    assert feats["spy_vol_60d"] > 0.0


# ---- calendar ---------------------------------------------------------------

def test_month_sin_and_month_cos_in_unit_interval():
    bars = _bars_universe("2024-01-02", 200)
    date = bars["A"].index[150]
    feats = extract_features_for_date(bars, ticker="A", date=date, benchmark="SPY")
    assert -1.0 <= feats["month_sin"] <= 1.0
    assert -1.0 <= feats["month_cos"] <= 1.0
    # Pythagorean identity: sin² + cos² = 1.
    assert feats["month_sin"] ** 2 + feats["month_cos"] ** 2 == pytest.approx(1.0, abs=1e-9)


# ---- insufficient history → row excluded -----------------------------------

def test_make_features_excludes_rows_without_history():
    bars = _bars_universe("2024-01-02", 50)  # too short for 6-month lookback (126d)
    dates_per_ticker = {"A": [bars["A"].index[40]], "B": [bars["B"].index[40]],
                        "C": [bars["C"].index[40]]}
    df = make_features(bars, dates_per_ticker, benchmark="SPY")
    assert df.empty


def test_make_features_returns_dataframe_with_expected_columns():
    bars = _bars_universe("2024-01-02", 200)
    dates_per_ticker = {
        "A": [bars["A"].index[150]],
        "B": [bars["B"].index[150]],
        "C": [bars["C"].index[150]],
    }
    df = make_features(bars, dates_per_ticker, benchmark="SPY")
    assert len(df) == 3
    expected_cols = {
        "ticker", "date",
        "mom_6m", "mom_1m", "mom_3m_skip_1m",
        "vol_20d", "vol_60d",
        "rank_mom_6m", "rank_vol_60d",
        "spy_vol_60d", "spy_mom_6m",
        "month_sin", "month_cos",
    }
    assert expected_cols.issubset(df.columns)
