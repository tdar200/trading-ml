"""
Feature extraction for the meta-labeling model.

10 features (one encoded as 2 columns) per (ticker, date), all computed
from data strictly BEFORE the date. Two public entry points:

- ``make_features(...)`` — batch, used by the offline trainer.
- ``extract_features_for_date(...)`` — single-row, used online by the strategy
  at rebalance time. Same feature definitions; one source of truth.

The benchmark series (default ``SPY``) is required for the market-regime
features (``spy_mom_6m`` / ``spy_vol_60d``) and is excluded from the
cross-sectional ranks (which are over the *tradeable* universe only).

Pure pandas + numpy; no Nautilus.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


_LOOKBACK_6M = 126  # ~6 months of business days
_LOOKBACK_1M = 21
_LOOKBACK_3M = 63
_LOOKBACK_VOL_20D = 20
_LOOKBACK_VOL_60D = 60


def _pos_or_none(series: pd.Series, date: pd.Timestamp) -> int | None:
    try:
        return series.index.get_loc(date)
    except KeyError:
        return None


def _annualized_vol(closes: pd.Series, end_pos: int, window: int) -> float:
    """Annualized stdev of daily log returns over the trailing window ending at end_pos (inclusive)."""
    if end_pos - window < 0:
        return float("nan")
    seg = closes.iloc[end_pos - window : end_pos + 1].to_numpy(dtype=float)
    rets = np.diff(np.log(seg))
    if rets.size == 0 or np.std(rets, ddof=0) == 0:
        return 0.0
    return float(np.std(rets, ddof=0) * math.sqrt(252))


def _momentum(closes: pd.Series, end_pos: int, lookback: int, skip: int = 0) -> float:
    """`close[end_pos - skip] / close[end_pos - skip - lookback] - 1`."""
    start = end_pos - skip - lookback
    end = end_pos - skip
    if start < 0:
        return float("nan")
    return float(closes.iloc[end]) / float(closes.iloc[start]) - 1.0


def _rank_in_universe(values: dict[str, float], ticker: str) -> float:
    """0..1 rank of `values[ticker]` among finite values; NaN if ticker missing or all NaN."""
    finite = {t: v for t, v in values.items() if not math.isnan(v)}
    if ticker not in finite:
        return float("nan")
    if len(finite) == 1:
        return 0.5
    sorted_vals = sorted(finite.values())
    rank = sorted_vals.index(finite[ticker])
    return rank / (len(finite) - 1)


def extract_features_for_date(
    bars_by_ticker: dict[str, pd.Series],
    *,
    ticker: str,
    date: pd.Timestamp,
    benchmark: str = "SPY",
) -> dict[str, float]:
    """Compute features for a single (ticker, date). Returns dict; values may be NaN
    when there's insufficient history (caller decides whether to drop)."""
    closes = bars_by_ticker.get(ticker)
    spy = bars_by_ticker.get(benchmark)
    if closes is None or spy is None:
        return {}

    pos = _pos_or_none(closes, date)
    spy_pos = _pos_or_none(spy, date)
    if pos is None or spy_pos is None:
        return {}

    mom_6m = _momentum(closes, pos, _LOOKBACK_6M)
    mom_1m = _momentum(closes, pos, _LOOKBACK_1M)
    mom_3m_skip_1m = _momentum(closes, pos, _LOOKBACK_3M, skip=_LOOKBACK_1M)
    vol_20d = _annualized_vol(closes, pos, _LOOKBACK_VOL_20D)
    vol_60d = _annualized_vol(closes, pos, _LOOKBACK_VOL_60D)

    spy_mom_6m = _momentum(spy, spy_pos, _LOOKBACK_6M)
    spy_vol_60d = _annualized_vol(spy, spy_pos, _LOOKBACK_VOL_60D)

    # Cross-sectional ranks across the TRADEABLE universe (exclude benchmark).
    universe_mom_6m: dict[str, float] = {}
    universe_vol_60d: dict[str, float] = {}
    for t, c in bars_by_ticker.items():
        if t == benchmark:
            continue
        t_pos = _pos_or_none(c, date)
        if t_pos is None:
            continue
        universe_mom_6m[t] = _momentum(c, t_pos, _LOOKBACK_6M)
        universe_vol_60d[t] = _annualized_vol(c, t_pos, _LOOKBACK_VOL_60D)

    rank_mom_6m = _rank_in_universe(universe_mom_6m, ticker)
    rank_vol_60d = _rank_in_universe(universe_vol_60d, ticker)

    month = date.month
    month_sin = math.sin(2.0 * math.pi * month / 12.0)
    month_cos = math.cos(2.0 * math.pi * month / 12.0)

    return {
        "mom_6m": mom_6m,
        "mom_1m": mom_1m,
        "mom_3m_skip_1m": mom_3m_skip_1m,
        "vol_20d": vol_20d,
        "vol_60d": vol_60d,
        "rank_mom_6m": rank_mom_6m,
        "rank_vol_60d": rank_vol_60d,
        "spy_mom_6m": spy_mom_6m,
        "spy_vol_60d": spy_vol_60d,
        "month_sin": month_sin,
        "month_cos": month_cos,
    }


def make_features(
    bars_by_ticker: dict[str, pd.Series],
    dates_per_ticker: dict[str, list[pd.Timestamp]],
    *,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """Batch entry: build a feature matrix for the given (ticker, date) pairs.

    Rows whose features cannot all be computed (insufficient history) are dropped.
    """
    rows: list[dict] = []
    for ticker, dates in dates_per_ticker.items():
        for date in dates:
            feats = extract_features_for_date(
                bars_by_ticker, ticker=ticker, date=date, benchmark=benchmark
            )
            if not feats:
                continue
            if any((isinstance(v, float) and math.isnan(v)) for v in feats.values()):
                continue
            rows.append({"ticker": ticker, "date": date, **feats})
    return pd.DataFrame(rows)


FEATURE_COLUMNS: tuple[str, ...] = (
    "mom_6m", "mom_1m", "mom_3m_skip_1m",
    "vol_20d", "vol_60d",
    "rank_mom_6m", "rank_vol_60d",
    "spy_mom_6m", "spy_vol_60d",
    "month_sin", "month_cos",
)
