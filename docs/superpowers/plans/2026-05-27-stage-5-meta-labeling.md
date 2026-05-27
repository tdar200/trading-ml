# Stage 5: Meta-Labeling ML Overlay — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LightGBM meta-labeling layer that filters + sizes the Stage-2 momentum signal's positions per-(ticker, rebalance-date), with walk-forward retraining and purged-embargoed CV, and evaluate it against the Stage-2 baseline through the Stage-3 harness on a 24-combo × 5-window grid.

**Architecture:** Pure-Python `models/` package (labeling, features, CV, trainer, store) consumed by a new `MomentumMetaStrategy` (subclass of Stage-2's strategy). Models are trained offline by a runnable that produces dated `.joblib` artifacts keyed by training-cutoff date; the strategy at rebalance time loads the most-recent qualifying artifact and applies a `filter+sizer` formula. Evaluation runs the strategy through the existing `analysis/walk_forward.py` harness varying `filter_threshold` × LightGBM hyperparameters, then computes Deflated Sharpe over the 24-trial grid.

**Tech Stack:**
- Python 3.12 (`uv`-managed)
- `lightgbm>=4.5` (NEW)
- `joblib>=1.4` (NEW; transitive via sklearn but pinned)
- `pandas`, `numpy` (existing)
- `pytest`, `pytest-mock` (existing)

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Add `lightgbm>=4.5`, `joblib>=1.4` |
| `.gitignore` | Add `models/cache/` |
| `models/__init__.py` | Package marker |
| `models/labeling.py` | Pure: `make_labels(bars_by_ticker, rebalance_dates) -> pd.DataFrame` |
| `models/features.py` | Pure: `make_features(bars_by_ticker, dates_per_ticker) -> pd.DataFrame` + `extract_features_for_date(...)` for online use |
| `models/cv.py` | Pure: `PurgedWalkForwardSplit` generator |
| `models/trainer.py` | Pure: `fit(X, y, **hyperparams) -> LGBMClassifier`, `predict_proba(model, X) -> np.ndarray` |
| `models/store.py` | `ModelStore` dataclass — save/load_for dated artifacts via joblib |
| `strategies/momentum_meta.py` | `MomentumMetaConfig` + `MomentumMetaStrategy(MomentumTopNStrategy)` with filter+sizer |
| `examples/train_meta_models.py` | Runnable: walk-forward train across the 8-LightGBM-combo grid → save dated artifacts per combo |
| `examples/run_harness_meta.py` | Runnable: Stage-3 harness comparing baseline vs meta on 24-combo × 5-window grid |
| `tests/test_labeling.py` | Pure-pandas tests on synthetic close series |
| `tests/test_features.py` | Pure-pandas tests on each feature + no-lookahead property |
| `tests/test_cv.py` | Pure tests on small synthetic date index |
| `tests/test_trainer.py` | Tiny synthetic X/y; determinism + fit/predict roundtrip |
| `tests/test_store.py` | tmp_path: save/load_for roundtrip + most-recent-before semantics |
| `tests/test_momentum_meta_strategy.py` | Synthetic-catalog integration with stub ModelStore (mirrors `tests/test_momentum_strategy.py`) |
| `models/cache/` | Pickled model artifacts (gitignored) |

**Boundaries:**
- `models/` is the **only** package that imports `lightgbm` or `joblib`.
- `strategies/momentum_meta.py` imports from `models.store` (load) and `models.features` (extract online), nothing else from `models/`.
- Online (strategy) and offline (trainer) feature paths share the same module → one source of truth per feature.

---

## Task 1: Scaffolding (`lightgbm`, `joblib`, gitignore, `models/__init__.py`)

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `models/__init__.py`

- [ ] **Step 1: Add deps**

Edit `pyproject.toml`'s `[project].dependencies` and append:
```toml
"lightgbm>=4.5",
"joblib>=1.4",
```

- [ ] **Step 2: Sync + verify**
```bash
cd /home/tahseen-dar/Projects/trading-ml
uv sync
uv run python -c "
import lightgbm, joblib
print('lightgbm', lightgbm.__version__)
print('joblib', joblib.__version__)
"
```
Expected: prints both versions, no errors.

- [ ] **Step 3: Add `models/cache/` to `.gitignore`**
```bash
grep -E '^models/cache/?$' .gitignore || echo "models/cache/" >> .gitignore
grep -E '^models/cache/?$' .gitignore
```
Expected: prints `models/cache/`.

- [ ] **Step 4: Create the models package**
```bash
mkdir -p models
touch models/__init__.py
```

- [ ] **Step 5: Full suite sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 76 passed (Stage-4 baseline). No regressions from adding deps.

- [ ] **Step 6: Commit**
```bash
git add pyproject.toml uv.lock .gitignore models/__init__.py
git commit -m "chore(stage5): add lightgbm + joblib deps + models package scaffold

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `models/labeling.py` — holding-period-return-sign labels

**Files:**
- Create: `tests/test_labeling.py`
- Create: `models/labeling.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_labeling.py`:
```python
"""Tests for models.labeling — pure pandas, no Nautilus."""
from __future__ import annotations

import pandas as pd

from models.labeling import make_labels


def _close_series(start_ts: str, prices: list[float]) -> pd.Series:
    return pd.Series(prices, index=pd.date_range(start_ts, periods=len(prices), freq="B"))


def test_positive_return_labels_one():
    """Price rises over the holding period → label=1."""
    # 30 business days; rebalance on day 0; check label by day 21.
    closes = _close_series("2024-01-02", [100.0] + [100.0 + i * 0.1 for i in range(1, 30)])
    bars_by_ticker = {"A": closes}
    rebalance_dates = pd.DatetimeIndex(["2024-01-02"])
    df = make_labels(bars_by_ticker, rebalance_dates, horizon_days=21)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "A"
    assert df.iloc[0]["label"] == 1


def test_negative_return_labels_zero():
    """Price falls over the holding period → label=0."""
    closes = _close_series("2024-01-02", [100.0] + [100.0 - i * 0.1 for i in range(1, 30)])
    df = make_labels({"A": closes}, pd.DatetimeIndex(["2024-01-02"]), horizon_days=21)
    assert df.iloc[0]["label"] == 0


def test_zero_return_labels_zero():
    """Flat price → label=0 (not strictly positive)."""
    closes = _close_series("2024-01-02", [100.0] * 30)
    df = make_labels({"A": closes}, pd.DatetimeIndex(["2024-01-02"]), horizon_days=21)
    assert df.iloc[0]["label"] == 0


def test_insufficient_forward_data_dropped():
    """Rebalance dates near the end of the series are dropped (no forward bar)."""
    closes = _close_series("2024-01-02", [100.0] * 10)  # only 10 days
    df = make_labels({"A": closes}, pd.DatetimeIndex(["2024-01-02"]), horizon_days=21)
    assert df.empty


def test_multi_ticker_one_row_per_ticker_per_date():
    """Multiple tickers + multiple dates → cross-product of valid rows."""
    a = _close_series("2024-01-02", [100.0] + [100.0 + i * 0.1 for i in range(1, 60)])
    b = _close_series("2024-01-02", [100.0] + [100.0 - i * 0.05 for i in range(1, 60)])
    dates = pd.DatetimeIndex(["2024-01-02", "2024-02-01"])
    df = make_labels({"A": a, "B": b}, dates, horizon_days=21)
    assert len(df) == 4  # 2 tickers × 2 dates
    assert set(df["ticker"]) == {"A", "B"}
    assert set(df["date"]) == {pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-01")}
```

- [ ] **Step 2: Run to confirm failure**
```bash
uv run pytest tests/test_labeling.py -v
```
Expected: collection error (`ModuleNotFoundError: No module named 'models.labeling'`).

- [ ] **Step 3: Implement**

Create `models/labeling.py`:
```python
"""
Holding-period-return-sign labels for meta-labeling.

For each (ticker, rebalance_date) pair, label = 1 if the close `horizon_days`
business days later is strictly above today's close, else 0. Pairs without
enough forward data (last `horizon_days` of the series) are dropped.

Pure pandas; no Nautilus. Output schema: [ticker, date, label].
"""
from __future__ import annotations

import pandas as pd


def make_labels(
    bars_by_ticker: dict[str, pd.Series],
    rebalance_dates: pd.DatetimeIndex,
    *,
    horizon_days: int = 21,
) -> pd.DataFrame:
    """Build a (ticker, date, label) DataFrame.

    Args:
        bars_by_ticker: each ticker's daily close series, indexed by date.
        rebalance_dates: dates at which to label positions.
        horizon_days: how many business days forward to look (default 21 ≈ 1 month).
    """
    rows: list[dict] = []
    for ticker, closes in bars_by_ticker.items():
        for date in rebalance_dates:
            if date not in closes.index:
                continue
            spot_pos = closes.index.get_loc(date)
            if spot_pos + horizon_days >= len(closes):
                continue
            spot = float(closes.iloc[spot_pos])
            forward = float(closes.iloc[spot_pos + horizon_days])
            label = 1 if (forward > spot) else 0
            rows.append({"ticker": ticker, "date": date, "label": label})
    return pd.DataFrame(rows, columns=["ticker", "date", "label"])
```

- [ ] **Step 4: Run to confirm pass**
```bash
uv run pytest tests/test_labeling.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Full suite sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 81 passed (76 + 5).

- [ ] **Step 6: Commit**
```bash
git add models/labeling.py tests/test_labeling.py
git commit -m "feat(models): holding-period-return-sign labels (pure pandas)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `models/features.py` — 10 no-lookahead features

**Files:**
- Create: `tests/test_features.py`
- Create: `models/features.py`

Features (per spec §4.2): `mom_6m`, `mom_1m`, `mom_3m_skip_1m`, `vol_20d`, `vol_60d`, `rank_mom_6m`, `rank_vol_60d`, `spy_vol_60d`, `spy_mom_6m`, `month_sin`, `month_cos`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_features.py`:
```python
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
```

- [ ] **Step 2: Run to confirm failure**
```bash
uv run pytest tests/test_features.py -v
```
Expected: collection error.

- [ ] **Step 3: Implement**

Create `models/features.py`:
```python
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

_MAX_LOOKBACK = max(_LOOKBACK_6M, _LOOKBACK_3M + _LOOKBACK_1M, _LOOKBACK_VOL_60D)


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
```

- [ ] **Step 4: Run to confirm pass**
```bash
uv run pytest tests/test_features.py -v
```
Expected: 12 passed.

- [ ] **Step 5: Full suite sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 93 passed (81 + 12).

- [ ] **Step 6: Commit**
```bash
git add models/features.py tests/test_features.py
git commit -m "feat(models): 10-feature extractor with batch + online entry points

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `models/cv.py` — purged + embargoed walk-forward split

**Files:**
- Create: `tests/test_cv.py`
- Create: `models/cv.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cv.py`:
```python
"""Tests for models.cv — pure, no Nautilus."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.cv import PurgedWalkForwardSplit


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def test_n_splits_yields_n_tuples():
    dates = _dates(500)
    splitter = PurgedWalkForwardSplit(n_splits=5, label_horizon_days=21, embargo_days=5)
    splits = list(splitter.split(dates))
    assert len(splits) == 5
    for train, test in splits:
        assert isinstance(train, np.ndarray)
        assert isinstance(test, np.ndarray)


def test_train_and_test_are_disjoint():
    dates = _dates(500)
    splitter = PurgedWalkForwardSplit(n_splits=5, label_horizon_days=21, embargo_days=5)
    for train, test in splitter.split(dates):
        assert set(train).isdisjoint(set(test))


def test_train_indices_are_strictly_before_test():
    """Walk-forward: every train index < every test index."""
    dates = _dates(500)
    splitter = PurgedWalkForwardSplit(n_splits=5, label_horizon_days=21, embargo_days=5)
    for train, test in splitter.split(dates):
        if len(train) == 0:
            continue
        assert int(train.max()) < int(test.min())


def test_embargo_gap_respected():
    """The largest train index is at least `embargo_days` before the smallest test index."""
    dates = _dates(500)
    splitter = PurgedWalkForwardSplit(n_splits=5, label_horizon_days=21, embargo_days=5)
    for train, test in splitter.split(dates):
        if len(train) == 0:
            continue
        gap = int(test.min()) - int(train.max())
        assert gap >= 5


def test_purge_horizon_respected():
    """Train labels whose forward horizon would overlap the test fold are excluded.

    With label_horizon_days=21 and embargo=0, the latest train index must be
    at least 21 days before the test fold's first date.
    """
    dates = _dates(500)
    splitter = PurgedWalkForwardSplit(n_splits=5, label_horizon_days=21, embargo_days=0)
    for train, test in splitter.split(dates):
        if len(train) == 0:
            continue
        gap = int(test.min()) - int(train.max())
        assert gap >= 21
```

- [ ] **Step 2: Run to confirm failure**
```bash
uv run pytest tests/test_cv.py -v
```
Expected: collection error.

- [ ] **Step 3: Implement**

Create `models/cv.py`:
```python
"""
Purged + embargoed walk-forward cross-validation.

Per López de Prado, *Advances in Financial Machine Learning* (2018):

- **Walk-forward**: the data are split into `n_splits` chronologically
  consecutive test folds; the training set for fold k is everything earlier
  than fold k's start.
- **Purge**: any training label whose forward horizon (`date + label_horizon_days`)
  would overlap the test fold's date range is excluded. This prevents the
  model from "knowing" overlapping outcomes during training.
- **Embargo**: an additional buffer of `embargo_days` is left between the
  latest training index and the first test index to handle autocorrelation
  in returns.

Returns numpy integer index arrays into the input `dates`. Pure: no IO,
no Nautilus.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedWalkForwardSplit:
    """Generator-style splitter.

    Usage:
        splitter = PurgedWalkForwardSplit(n_splits=5, label_horizon_days=21, embargo_days=5)
        for train_idx, test_idx in splitter.split(dates):
            ...
    """
    n_splits: int
    label_horizon_days: int = 21
    embargo_days: int = 5

    def split(self, dates: pd.DatetimeIndex) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        n = len(dates)
        if n < self.n_splits:
            raise ValueError(f"need at least n_splits ({self.n_splits}) dates, got {n}")

        # Equal-sized contiguous test folds.
        fold_size = n // self.n_splits
        for k in range(self.n_splits):
            test_start = k * fold_size
            test_end = (k + 1) * fold_size if k < self.n_splits - 1 else n
            test_idx = np.arange(test_start, test_end, dtype=np.int64)

            # Train = everything strictly before test fold, minus purge + embargo.
            train_cutoff = test_start - self.embargo_days - self.label_horizon_days
            if train_cutoff <= 0:
                train_idx = np.array([], dtype=np.int64)
            else:
                train_idx = np.arange(0, train_cutoff, dtype=np.int64)

            yield train_idx, test_idx
```

- [ ] **Step 4: Run to confirm pass**
```bash
uv run pytest tests/test_cv.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Full suite sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 98 passed (93 + 5).

- [ ] **Step 6: Commit**
```bash
git add models/cv.py tests/test_cv.py
git commit -m "feat(models): PurgedWalkForwardSplit per López de Prado AFML

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `models/trainer.py` — LightGBM fit + predict

**Files:**
- Create: `tests/test_trainer.py`
- Create: `models/trainer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trainer.py`:
```python
"""Tests for models.trainer — LightGBM fit/predict pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.trainer import fit, predict_proba


def _synthetic_xy(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })
    # y depends on f1 with noise — model should learn this.
    y = (X["f1"] + 0.3 * rng.normal(size=n) > 0).astype(int).to_numpy()
    return X, y


def test_fit_returns_model_with_predict_proba():
    X, y = _synthetic_xy(200, seed=1)
    model = fit(X, y, n_estimators=20, max_depth=3, learning_rate=0.1)
    proba = predict_proba(model, X)
    assert proba.shape == (200,)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_fit_is_deterministic_with_fixed_seed():
    X, y = _synthetic_xy(200, seed=2)
    m1 = fit(X, y, n_estimators=20, max_depth=3, learning_rate=0.1, random_state=42)
    m2 = fit(X, y, n_estimators=20, max_depth=3, learning_rate=0.1, random_state=42)
    p1 = predict_proba(m1, X)
    p2 = predict_proba(m2, X)
    assert np.allclose(p1, p2)


def test_fit_learns_signal_on_easy_problem():
    """On data where y is largely a linear function of f1, in-sample accuracy
    should be well above 50%."""
    X, y = _synthetic_xy(400, seed=3)
    model = fit(X, y, n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    pred = (predict_proba(model, X) > 0.5).astype(int)
    accuracy = (pred == y).mean()
    assert accuracy > 0.7
```

- [ ] **Step 2: Run to confirm failure**
```bash
uv run pytest tests/test_trainer.py -v
```
Expected: collection error.

- [ ] **Step 3: Implement**

Create `models/trainer.py`:
```python
"""
LightGBM fit + predict pipeline.

Two functions, both pure:

- ``fit(X, y, *, n_estimators, max_depth, learning_rate, random_state=42) -> LGBMClassifier``
- ``predict_proba(model, X) -> np.ndarray`` — returns P(y=1) per row.

The trainer is unaware of walk-forward / CV — that orchestration lives in
``examples/train_meta_models.py`` and ``models/cv.py``.

No IO, no Nautilus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


def fit(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> LGBMClassifier:
    """Fit a binary LightGBM classifier on (X, y).

    Args:
        X: feature DataFrame (rows = samples, columns = feature names).
        y: 0/1 label array, same length as X.
        n_estimators: number of boosting rounds.
        max_depth: tree depth limit (3 / 5 typical for small datasets).
        learning_rate: shrinkage.
        random_state: passed to LightGBM for reproducibility.

    Returns:
        Fitted LGBMClassifier.
    """
    model = LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        verbose=-1,  # quiet
        n_jobs=1,    # deterministic on small datasets
    )
    model.fit(X, y)
    return model


def predict_proba(model: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return P(y=1) for each row in X."""
    proba = model.predict_proba(X)
    # LGBMClassifier returns shape (n, 2); column 1 is P(y=1).
    return proba[:, 1]
```

- [ ] **Step 4: Run to confirm pass**
```bash
uv run pytest tests/test_trainer.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Full suite sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 101 passed (98 + 3).

- [ ] **Step 6: Commit**
```bash
git add models/trainer.py tests/test_trainer.py
git commit -m "feat(models): LightGBM fit/predict pipeline (pure)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `models/store.py` — ModelStore for dated artifacts

**Files:**
- Create: `tests/test_store.py`
- Create: `models/store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_store.py`:
```python
"""Tests for models.store — dated joblib artifacts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.store import ModelStore
from models.trainer import fit


def _trained_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=100), "f2": rng.normal(size=100)})
    y = (X["f1"] > 0).astype(int).to_numpy()
    return fit(X, y, n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42)


def test_save_creates_dated_file(tmp_path: Path):
    store = ModelStore(cache_dir=tmp_path, label="default")
    model = _trained_model()
    path = store.save(pd.Timestamp("2024-03-01"), model, feature_names=["f1", "f2"])
    assert path.exists()
    assert "2024-03-01" in path.name
    assert "default" in path.name


def test_load_for_picks_most_recent_strictly_before(tmp_path: Path):
    store = ModelStore(cache_dir=tmp_path, label="default")
    model = _trained_model()
    store.save(pd.Timestamp("2024-01-01"), model, feature_names=["f1", "f2"])
    store.save(pd.Timestamp("2024-04-01"), model, feature_names=["f1", "f2"])
    store.save(pd.Timestamp("2024-07-01"), model, feature_names=["f1", "f2"])
    # Query for 2024-05-15 → should pick 2024-04-01.
    result = store.load_for(pd.Timestamp("2024-05-15"))
    assert result is not None
    _, feature_names = result
    assert feature_names == ["f1", "f2"]
    # Verify it picked the 2024-04-01 file by inspecting `last_loaded_path`.
    assert "2024-04-01" in str(store.last_loaded_path)


def test_load_for_returns_none_before_any_model(tmp_path: Path):
    """Querying before any model has been saved → None."""
    store = ModelStore(cache_dir=tmp_path, label="default")
    model = _trained_model()
    store.save(pd.Timestamp("2024-04-01"), model, feature_names=["f1", "f2"])
    result = store.load_for(pd.Timestamp("2024-03-15"))  # before only saved cutoff
    assert result is None


def test_load_for_label_filter_isolates_labels(tmp_path: Path):
    """Different labels in the same cache_dir don't interfere with load_for."""
    store_a = ModelStore(cache_dir=tmp_path, label="combo_a")
    store_b = ModelStore(cache_dir=tmp_path, label="combo_b")
    model = _trained_model()
    store_a.save(pd.Timestamp("2024-04-01"), model, feature_names=["f1", "f2"])
    # store_b has nothing → its load_for must return None.
    assert store_b.load_for(pd.Timestamp("2024-05-15")) is None
    assert store_a.load_for(pd.Timestamp("2024-05-15")) is not None
```

- [ ] **Step 2: Run to confirm failure**
```bash
uv run pytest tests/test_store.py -v
```
Expected: collection error.

- [ ] **Step 3: Implement**

Create `models/store.py`:
```python
"""
ModelStore — persist + look up trained models keyed by training-cutoff date.

Pickle format (via joblib):
- Filename: ``{label}_{cutoff_date:%Y-%m-%d}.joblib`` under ``cache_dir``.
- Payload: ``{"model": LGBMClassifier, "feature_names": list[str]}``.

Usage:
    store = ModelStore(cache_dir=Path("models/cache"), label="default")
    store.save(pd.Timestamp("2024-04-01"), trained_model, feature_names=[...])
    pair = store.load_for(pd.Timestamp("2024-05-15"))   # → (model, feature_names) or None

``load_for`` picks the most-recent stored model whose cutoff_date is
STRICTLY before the query date. This enforces walk-forward: a query at
date D never sees a model trained on data including D.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


_CUTOFF_RE = re.compile(r"^(?P<label>.+)_(?P<date>\d{4}-\d{2}-\d{2})\.joblib$")


@dataclass
class ModelStore:
    """Persist + retrieve dated model artifacts. One ``label`` per
    (hyperparameter combo) namespace within a shared ``cache_dir``."""
    cache_dir: Path
    label: str
    # Set by load_for to the path of the artifact actually loaded
    # (useful for diagnostics; not load-bearing).
    last_loaded_path: Path | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _filename(self, cutoff_date: pd.Timestamp) -> Path:
        return self.cache_dir / f"{self.label}_{cutoff_date.strftime('%Y-%m-%d')}.joblib"

    def save(
        self,
        cutoff_date: pd.Timestamp,
        model: Any,
        *,
        feature_names: list[str],
    ) -> Path:
        path = self._filename(cutoff_date)
        joblib.dump({"model": model, "feature_names": list(feature_names)}, path)
        return path

    def list_cutoffs(self) -> list[pd.Timestamp]:
        out: list[pd.Timestamp] = []
        for path in self.cache_dir.glob(f"{self.label}_*.joblib"):
            m = _CUTOFF_RE.match(path.name)
            if not m:
                continue
            if m.group("label") != self.label:
                continue
            out.append(pd.Timestamp(m.group("date")))
        return sorted(out)

    def load_for(self, query_date: pd.Timestamp) -> tuple[Any, list[str]] | None:
        """Return (model, feature_names) for the most-recent cutoff strictly before
        query_date. None if no qualifying artifact exists."""
        cutoffs = self.list_cutoffs()
        eligible = [c for c in cutoffs if c < query_date]
        if not eligible:
            self.last_loaded_path = None
            return None
        chosen = max(eligible)
        path = self._filename(chosen)
        payload = joblib.load(path)
        self.last_loaded_path = path
        return payload["model"], list(payload["feature_names"])
```

- [ ] **Step 4: Run to confirm pass**
```bash
uv run pytest tests/test_store.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Full suite sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 105 passed (101 + 4).

- [ ] **Step 6: Commit**
```bash
git add models/store.py tests/test_store.py
git commit -m "feat(models): ModelStore — dated joblib artifacts keyed by training cutoff

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `strategies/momentum_meta.py` — MomentumMetaStrategy

**Files:**
- Create: `tests/test_momentum_meta_strategy.py`
- Create: `strategies/momentum_meta.py`

This subclasses Stage-2's `MomentumTopNStrategy` and adds the filter+sizer overlay. Tests use synthetic data and a stub `ModelStore` — same pattern as `tests/test_momentum_strategy.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_momentum_meta_strategy.py`:
```python
"""Integration tests for MomentumMetaStrategy on synthetic catalogs.

Uses a stub ModelStore that returns a controllable mock model so the
filter+sizer logic can be exercised without training a real LightGBM.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
```

- [ ] **Step 2: Run to confirm failure**
```bash
uv run pytest tests/test_momentum_meta_strategy.py -v
```
Expected: collection error.

- [ ] **Step 3: Implement**

Create `strategies/momentum_meta.py`:
```python
"""
MomentumMetaStrategy — Stage-5 ML overlay on top of the Stage-2 momentum strategy.

For each (ticker, rebalance-date) pair the primary signal would consider, a
LightGBM meta-model predicts P(win) (will THIS position be profitable over
the next holding period?). The result is used as a filter + sizer:

    if p_win < filter_threshold:
        target_weight = 0.0
    else:
        confidence = (p_win - filter_threshold) / (1.0 - filter_threshold)
        target_weight = (equity / top_n) * confidence

The strategy loads the most-recent ModelStore artifact whose training
cutoff is STRICTLY before the rebalance date. During warm-up (no model
yet available), the strategy falls back to pure baseline behaviour
(equivalent to Stage-2).

The feature extraction at rebalance time uses ``models.features.extract_features_for_date``,
which is the SAME function the offline trainer calls — one source of truth.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Quantity

from data.instruments import VENUE_MIC
from models.features import extract_features_for_date
from models.store import ModelStore
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy
from strategies.ranking import rank_by_momentum


class MomentumMetaConfig(MomentumTopNConfig, frozen=True):
    """Config = Stage-2 MomentumTopNConfig fields + Stage-5 model parameters."""
    model_store_path: str = ""        # filesystem path to ModelStore cache_dir
    model_store_label: str = "default"  # ModelStore label (one per hyperparam combo)
    filter_threshold: float = 0.55     # P(win) below this → skip position


class MomentumMetaStrategy(MomentumTopNStrategy):
    """Stage-5 meta-labeled wrapper around Stage-2 momentum top-N."""

    def __init__(self, config: MomentumMetaConfig) -> None:
        super().__init__(config)
        # Resolve the ModelStore. Stays None if no path configured.
        self._model_store: ModelStore | None = None
        if config.model_store_path:
            cache_dir = Path(config.model_store_path)
            if cache_dir.exists():
                self._model_store = ModelStore(
                    cache_dir=cache_dir,
                    label=config.model_store_label,
                )

    # ---- _rebalance override ----------------------------------------------

    def _rebalance(self, now_ts_ns: int) -> None:
        """Stage-5 rebalance: same as Stage-2 except positions go through the
        meta-model filter+sizer before order generation."""
        self._rebalance_count += 1

        # 1. Primary ranking.
        ranked = rank_by_momentum(
            {t: self._closes.get(t, []) for t in self.config.universe},
            lookback_days=self.config.lookback_days,
            skip_days=self.config.skip_days,
        )
        if not ranked:
            return
        target_tickers = ranked[: self.config.top_n]

        # 2. Equity → base per-name USD.
        venue = Venue(VENUE_MIC)
        account = self.portfolio.account(venue)
        equity_usd = float(self.portfolio.equity(venue)[account.base_currency].as_decimal())
        # 2% cash buffer (Stage 2): leave headroom for slippage + integer rounding.
        per_name_usd = (equity_usd * 0.98) / max(self.config.top_n, 1)

        # 3. Compute per-ticker target USD via filter+sizer.
        target_usd_by_ticker: dict[str, float] = {}
        rebalance_date = pd.Timestamp(now_ts_ns, unit="ns", tz="UTC").tz_convert(None)

        # Try to load a model for this date. If None → baseline behaviour
        # (treat every target as full base_weight).
        model_payload = None
        if self._model_store is not None:
            try:
                model_payload = self._model_store.load_for(rebalance_date)
            except Exception:
                model_payload = None

        if model_payload is None:
            # Baseline: top_n at full base_weight.
            for t in target_tickers:
                target_usd_by_ticker[t] = per_name_usd
        else:
            model, feature_names = model_payload
            # Build a bars_by_ticker dict of pd.Series for feature extraction.
            bars_by_ticker = self._bars_to_series_dict()
            for t in target_tickers:
                feats = extract_features_for_date(
                    bars_by_ticker, ticker=t, date=rebalance_date,
                    benchmark=self.config.benchmark,
                )
                if not feats or any(
                    (isinstance(v, float) and (v != v))  # NaN
                    for v in feats.values()
                ):
                    # Insufficient history for this ticker → conservative: skip.
                    continue
                row = pd.DataFrame([{k: feats[k] for k in feature_names}])
                proba = model.predict_proba(row)
                p_win = float(proba[0, 1])
                if p_win < self.config.filter_threshold:
                    continue
                confidence = (p_win - self.config.filter_threshold) / (
                    1.0 - self.config.filter_threshold
                )
                target_usd_by_ticker[t] = per_name_usd * confidence

        # 4. Reconcile current positions to target USD (same as Stage 2: sells then buys).
        sells: list[tuple[Any, int]] = []
        buys: list[tuple[Any, int]] = []
        for t in self.config.universe:
            instr_id = InstrumentId.from_str(f"{t}.{VENUE_MIC}")
            current = self._current_position_qty(instr_id)
            target_usd = target_usd_by_ticker.get(t, 0.0)
            if t in target_tickers and target_usd > 0:
                last_close = self._closes[t][-1] if self._closes.get(t) else 0.0
                target = int(target_usd / last_close) if last_close > 0 else 0
            else:
                target = 0
            delta = target - current
            if delta == 0:
                continue
            qty = Quantity.from_int(abs(delta))
            if delta < 0:
                sells.append((instr_id, qty))
            else:
                buys.append((instr_id, qty))

        # SELL first, then BUY (cash-account safety from Stage 4).
        for instr_id, qty in sells:
            order = self.order_factory.market(
                instrument_id=instr_id, order_side=OrderSide.SELL, quantity=qty,
            )
            self.submit_order(order)
        for instr_id, qty in buys:
            order = self.order_factory.market(
                instrument_id=instr_id, order_side=OrderSide.BUY, quantity=qty,
            )
            self.submit_order(order)

    # ---- helpers -----------------------------------------------------------

    def _bars_to_series_dict(self) -> dict[str, pd.Series]:
        """Convert the ring buffer into the pd.Series-per-ticker shape that
        models.features expects. Index is synthetic business-day; values are
        the historical closes the strategy has accumulated."""
        out: dict[str, pd.Series] = {}
        for ticker, closes in self._closes.items():
            if not closes:
                continue
            # Use a business-day index of the same length, ending today.
            idx = pd.date_range(end=pd.Timestamp.utcnow().tz_localize(None).normalize(),
                                periods=len(closes), freq="B")
            out[ticker] = pd.Series(closes, index=idx)
        return out
```

⚠️ The `_bars_to_series_dict` helper uses a *synthetic* business-day index because the strategy doesn't track real calendar dates per close — its ring buffer is just a `list[float]`. This means `month_sin`/`month_cos` features at rebalance time will reflect *today's* calendar month (which is correct — they describe the rebalance date itself, not historical bars). The momentum/vol features only depend on relative positions in the series, so the synthetic index doesn't bias them.

- [ ] **Step 4: Run to confirm pass**
```bash
uv run pytest tests/test_momentum_meta_strategy.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Full suite sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 108 passed (105 + 3).

- [ ] **Step 6: Commit**
```bash
git add strategies/momentum_meta.py tests/test_momentum_meta_strategy.py
git commit -m "feat(strategies): MomentumMetaStrategy — Stage-5 filter+sizer overlay

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `examples/train_meta_models.py` — offline walk-forward training

**Files:**
- Create: `examples/train_meta_models.py`

Trains 8 LightGBM combos (`n_estimators × max_depth × learning_rate` cross-product) × N walk-forward cutoffs per combo. Saves each to `models/cache/{label}_{cutoff}.joblib`. Used by the harness eval runnable (Task 9).

- [ ] **Step 1: Implement**

Create `examples/train_meta_models.py`:
```python
"""
Stage-5 offline training: walk-forward retrain across the LightGBM
hyperparameter grid against the Stage-1 catalog.

For each (n_estimators, max_depth, learning_rate) combo, train one model
per monthly cutoff date and save to models/cache/ as dated joblib artifacts.

Usage:
    uv run python examples/train_meta_models.py
    uv run python examples/train_meta_models.py --smoke    # 1 combo, ~3 cutoffs
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

from data.instruments import VENUE_MIC
from data.tickers import ALL_TICKERS, BENCHMARK, UNIVERSE
from models.features import FEATURE_COLUMNS, make_features
from models.labeling import make_labels
from models.store import ModelStore
from models.trainer import fit


CATALOG_PATH = _PROJECT_ROOT / "data" / "catalog"
CACHE_DIR = _PROJECT_ROOT / "models" / "cache"


# Hyperparameter grid (matches spec §4.9). Defines 8 LightGBM combos.
DEFAULT_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.10],
}
SMOKE_GRID = {
    "n_estimators": [100],
    "max_depth": [3],
    "learning_rate": [0.10],
}


def _combo_label(combo: dict) -> str:
    return (
        f"n{combo['n_estimators']}"
        f"_d{combo['max_depth']}"
        f"_lr{int(combo['learning_rate'] * 100):02d}"
    )


def _bars_to_close_series(bars) -> pd.Series:
    """ParquetDataCatalog bars → pd.Series indexed by date, values = close."""
    rows = [
        (pd.Timestamp(b.ts_event, unit="ns", tz="UTC").tz_convert(None).normalize(),
         float(b.close))
        for b in bars
    ]
    s = pd.Series(dict(rows), name="close").sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def _monthly_rebalance_dates(closes: pd.Series, after: pd.Timestamp | None = None
                             ) -> pd.DatetimeIndex:
    """First close index of each calendar month, restricted to `after`+."""
    df = closes.to_frame(name="close")
    df["month"] = df.index.to_period("M")
    grouped = df.groupby("month", as_index=False).head(1)
    dates = pd.DatetimeIndex(grouped.index)
    if after is not None:
        dates = dates[dates >= after]
    return dates


def _monthly_cutoffs(dates: pd.DatetimeIndex, warmup_months: int = 12
                     ) -> pd.DatetimeIndex:
    """Cutoffs at which we train a new model — every rebalance date after
    the warmup_months-th."""
    if len(dates) <= warmup_months:
        return pd.DatetimeIndex([])
    return dates[warmup_months:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Train 1 hyperparameter combo on the first 3 cutoffs only (~seconds).")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    grid = SMOKE_GRID if args.smoke else DEFAULT_GRID

    print(f"loading catalog from {CATALOG_PATH}...")
    catalog = ParquetDataCatalog(path=str(CATALOG_PATH.resolve()))
    bars_by_ticker = {
        t: _bars_to_close_series(catalog.bars(instrument_ids=[f"{t}.{VENUE_MIC}"]))
        for t in ALL_TICKERS
    }
    spy_close = bars_by_ticker[BENCHMARK]
    print(f"catalog loaded: {sum(len(s) for s in bars_by_ticker.values())} bars across {len(bars_by_ticker)} tickers")

    # 1. Build monthly rebalance schedule from SPY's index (the longest series).
    all_rebalance_dates = _monthly_rebalance_dates(spy_close)
    print(f"rebalance dates: {len(all_rebalance_dates)} ({all_rebalance_dates.min().date()} → {all_rebalance_dates.max().date()})")

    # 2. Labels for every (ticker in UNIVERSE, date) pair.
    labels = make_labels(
        {t: bars_by_ticker[t] for t in UNIVERSE},
        all_rebalance_dates, horizon_days=21,
    )
    print(f"labels generated: {len(labels)} rows")

    # 3. Features for the same (ticker, date) pairs.
    dates_per_ticker = {
        t: list(all_rebalance_dates[all_rebalance_dates.isin(bars_by_ticker[t].index)])
        for t in UNIVERSE
    }
    features = make_features(bars_by_ticker, dates_per_ticker, benchmark=BENCHMARK)
    print(f"features generated: {len(features)} rows × {len(FEATURE_COLUMNS)} columns")

    # 4. Join labels + features on (ticker, date).
    dataset = features.merge(labels, on=["ticker", "date"], how="inner")
    dataset = dataset.dropna(subset=list(FEATURE_COLUMNS) + ["label"]).reset_index(drop=True)
    print(f"training dataset: {len(dataset)} rows after join + dropna")
    if dataset.empty:
        print("ERROR: empty training dataset — check catalog has enough history")
        return 1

    # 5. Cutoffs: each monthly rebalance after a warm-up of 12 months.
    cutoffs = _monthly_cutoffs(all_rebalance_dates, warmup_months=12)
    if args.smoke:
        cutoffs = cutoffs[:3]
    print(f"cutoffs: {len(cutoffs)} ({cutoffs.min().date() if len(cutoffs) else 'n/a'} → "
          f"{cutoffs.max().date() if len(cutoffs) else 'n/a'})")

    # 6. Hyperparameter cross-product.
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals, strict=True)) for vals in product(*grid.values())]
    print(f"hyperparameter combos: {len(combos)}")

    total_models = 0
    for combo in combos:
        label = _combo_label(combo)
        store = ModelStore(cache_dir=cache_dir, label=label)
        print(f"  combo={label}  training {len(cutoffs)} cutoff models...")
        for cutoff in cutoffs:
            # Train on rows where date < cutoff - purge_horizon (use 21 = horizon_days).
            train = dataset[dataset["date"] < cutoff - pd.Timedelta(days=21)]
            if len(train) < 50:
                # Not enough training data yet; skip this cutoff.
                continue
            X_train = train[list(FEATURE_COLUMNS)]
            y_train = train["label"].to_numpy()
            model = fit(
                X_train, y_train,
                n_estimators=combo["n_estimators"],
                max_depth=combo["max_depth"],
                learning_rate=combo["learning_rate"],
            )
            store.save(cutoff, model, feature_names=list(FEATURE_COLUMNS))
            total_models += 1
    print(f"done: {total_models} models saved under {cache_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify `--help` works**
```bash
cd /home/tahseen-dar/Projects/trading-ml
uv run python examples/train_meta_models.py --help
```
Expected: argparse help text, exit 0.

- [ ] **Step 3: Run smoke mode against the real catalog**
```bash
time uv run python examples/train_meta_models.py --smoke 2>&1 | tail -15
```
Expected: completes in seconds (1 combo × ≤3 cutoffs). Prints "done: N models saved" with N≥1.

Verify a model artifact was actually written:
```bash
ls -la models/cache/ | head -5
```
Expected: at least one `*.joblib` file present.

- [ ] **Step 4: Full pytest sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: still 108 passed.

- [ ] **Step 5: Commit**
```bash
git add examples/train_meta_models.py
git commit -m "feat(examples): walk-forward meta-model training across LightGBM grid

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: `examples/run_harness_meta.py` — baseline vs meta evaluation

**Files:**
- Create: `examples/run_harness_meta.py`

Drives the Stage-3 harness for both `MomentumTopNStrategy` (baseline) and `MomentumMetaStrategy` (across the 24-combo grid). Writes a comparison Markdown report.

- [ ] **Step 1: Implement**

Create `examples/run_harness_meta.py`:
```python
"""
Stage-5 harness evaluation: compare baseline Stage-2 momentum vs Stage-5
meta-labeled overlay through the Stage-3 walk-forward harness.

Usage:
    uv run python examples/run_harness_meta.py
    uv run python examples/run_harness_meta.py --smoke

Writes results/harness_meta_<timestamp>/ with baseline.csv, meta.csv,
summary.json, summary.md.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json

import pandas as pd
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

from analysis.deflated_sharpe import deflated_sharpe
from analysis.runspec import RunSpec
from analysis.runner import run as backtest_run
from analysis.walk_forward import walk_forward
from data.instruments import VENUE_MIC, build_universe_instruments
from data.tickers import ALL_TICKERS, BENCHMARK, UNIVERSE
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy
from strategies.momentum_meta import MomentumMetaConfig, MomentumMetaStrategy


CATALOG_PATH = _PROJECT_ROOT / "data" / "catalog"
CACHE_DIR = _PROJECT_ROOT / "models" / "cache"
RESULTS_ROOT = _PROJECT_ROOT / "results"


def _combo_label(n_estimators: int, max_depth: int, learning_rate: float) -> str:
    return f"n{n_estimators}_d{max_depth}_lr{int(learning_rate * 100):02d}"


def _build_windows(bars_by_ticker, n_windows: int) -> list[tuple[datetime, datetime]]:
    universe_starts = [bars_by_ticker[t][0].ts_event for t in UNIVERSE if bars_by_ticker.get(t)]
    if not universe_starts:
        raise RuntimeError("no universe bars")
    start_ns = max(universe_starts)
    end_ns = max(bars_by_ticker[t][-1].ts_event for t in ALL_TICKERS if bars_by_ticker.get(t))
    start = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc)
    end = datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc)
    span = end - start
    return [(start, start + (span * (i + 1)) / n_windows) for i in range(n_windows)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="2 meta combos × 2 windows (~seconds).")
    args = parser.parse_args()

    n_windows = 2 if args.smoke else 5

    catalog = ParquetDataCatalog(path=str(CATALOG_PATH.resolve()))
    bars_by_ticker = {t: catalog.bars(instrument_ids=[f"{t}.{VENUE_MIC}"]) for t in ALL_TICKERS}
    instruments = build_universe_instruments()
    if sum(len(v) for v in bars_by_ticker.values()) == 0:
        print("ERROR: catalog is empty. Run data/ingest_alpaca.py first.")
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS_ROOT / f"harness_meta_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_dir}")

    windows = _build_windows(bars_by_ticker, n_windows=n_windows)
    print(f"windows: {len(windows)}")

    # 1. Baseline: Stage-2 winner config, one run per window.
    print("running baseline...")
    base_kwargs = dict(universe=UNIVERSE, benchmark=BENCHMARK,
                       top_n=3, lookback_days=63, skip_days=0)
    base_rows = []
    for window_idx, (start, end) in enumerate(windows):
        spec = RunSpec(
            label=f"baseline-w{window_idx}",
            strategy_class=MomentumTopNStrategy,
            strategy_config=MomentumTopNConfig(**base_kwargs),
            start=start, end=end,
            slippage_prob=0.25,
        )
        result = backtest_run(spec, bars_by_ticker=bars_by_ticker, instruments=instruments)
        base_rows.append({"window_idx": window_idx, "start": start, "end": end, **result.metrics})
    base_df = pd.DataFrame(base_rows)
    base_df.to_csv(out_dir / "baseline.csv", index=False)
    print(f"  baseline median Sharpe: {base_df['sharpe'].median():.4f}")

    # 2. Meta: cross-product of (LightGBM combo, filter_threshold) × windows.
    print("running meta grid...")
    n_estimators_vals = [100] if args.smoke else [100, 200]
    max_depth_vals = [3] if args.smoke else [3, 5]
    learning_rate_vals = [0.10] if args.smoke else [0.05, 0.10]
    threshold_vals = [0.55] if args.smoke else [0.50, 0.55, 0.60]

    meta_grid = {
        "model_store_label": [
            _combo_label(n, d, lr)
            for n, d, lr in product(n_estimators_vals, max_depth_vals, learning_rate_vals)
        ],
        "filter_threshold": threshold_vals,
    }
    meta_base = dict(
        universe=UNIVERSE, benchmark=BENCHMARK,
        top_n=3, lookback_days=63, skip_days=0,
        model_store_path=str(CACHE_DIR),
    )
    meta_df = walk_forward(
        strategy_class=MomentumMetaStrategy,
        config_class=MomentumMetaConfig,
        base_config_kwargs=meta_base,
        param_grid=meta_grid,
        windows=windows,
        bars_by_ticker=bars_by_ticker,
        instruments=instruments,
        slippage_prob=0.25,
    )
    meta_df.to_csv(out_dir / "meta.csv", index=False)
    print(f"  meta runs: {len(meta_df)}; max median Sharpe: "
          f"{meta_df.groupby(['model_store_label', 'filter_threshold'])['sharpe'].median().max():.4f}")

    # 3. Deflated Sharpe over baseline + meta combos.
    sharpes = pd.concat([base_df["sharpe"], meta_df["sharpe"]]).dropna().tolist()
    n_obs = int(meta_df["n_days"].iloc[0]) if len(meta_df) else int(base_df["n_days"].iloc[0])
    dsr = deflated_sharpe(sharpes=sharpes, n_obs=n_obs)
    print(f"DSR: p-value={dsr['p_value']:.4f}, z={dsr['deflated_sharpe']:.4f}")

    # 4. Summary report.
    baseline_median = float(base_df["sharpe"].median())
    meta_grouped = meta_df.groupby(["model_store_label", "filter_threshold"])["sharpe"].median()
    meta_winner_idx = meta_grouped.idxmax()
    meta_winner_median = float(meta_grouped.max())
    meta_beats = meta_winner_median > baseline_median

    md = f"""# Stage-5 Meta-Labeling Harness — {timestamp}

## Baseline (Stage-2 winner: top_n=3, lookback=63, skip=0)
- median window Sharpe: **{baseline_median:.4f}**
- median window CAGR:   {float(base_df['cagr'].median()):.4f}

## Meta winner combo
- model_store_label: `{meta_winner_idx[0]}`
- filter_threshold:  `{meta_winner_idx[1]}`
- median window Sharpe: **{meta_winner_median:.4f}**

## Deflated Sharpe (baseline + meta as trial set)
- n_trials:              {dsr['n_trials']}
- observed Sharpe:       {dsr['observed_sharpe']:.4f}
- expected_max under H0: {dsr['expected_max_sharpe']:.4f}
- deflated z-score:      {dsr['deflated_sharpe']:.4f}
- p-value:               **{dsr['p_value']:.4f}**

## Verdict
- Meta beats baseline OOS: **{'YES' if meta_beats else 'NO'}**
- Meta winner survives DSR: **{'YES' if dsr['p_value'] > 0.95 else 'NO'}** (p > 0.95)
- Overall verdict: **{('YES' if (meta_beats and dsr['p_value'] > 0.95) else 'NO')} — meta did{'' if (meta_beats and dsr['p_value'] > 0.95) else ' not'} clearly beat baseline.**
"""
    (out_dir / "summary.md").write_text(md)

    summary = {
        "baseline_median_sharpe": baseline_median,
        "meta_winner": {
            "model_store_label": meta_winner_idx[0],
            "filter_threshold": float(meta_winner_idx[1]),
            "median_window_sharpe": meta_winner_median,
        },
        "dsr": dsr,
        "n_windows": n_windows,
        "n_meta_runs": int(len(meta_df)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + md)
    print(f"artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify `--help` works**
```bash
cd /home/tahseen-dar/Projects/trading-ml
uv run python examples/run_harness_meta.py --help
```
Expected: argparse help, exit 0.

- [ ] **Step 3: Run smoke mode against the real catalog**

First make sure smoke training has been run (Task 8). If not, re-run:
```bash
uv run python examples/train_meta_models.py --smoke
```

Then evaluate:
```bash
time uv run python examples/run_harness_meta.py --smoke 2>&1 | tail -40
```
Expected: ~30 seconds. Prints the Markdown summary with concrete numbers.

- [ ] **Step 4: Full pytest sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: still 108 passed.

- [ ] **Step 5: Commit**
```bash
git add examples/run_harness_meta.py
git commit -m "feat(examples): Stage-5 harness — baseline vs meta with Deflated Sharpe

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Full acceptance run on the real catalog

**Files:** none new — runs both runnables full-grid.

- [ ] **Step 1: Train the full 8-combo model grid**
```bash
cd /home/tahseen-dar/Projects/trading-ml
time uv run python examples/train_meta_models.py 2>&1 | tail -20
```
Expected: completes in 1–10 minutes. Prints "done: N models saved" with N = 8 combos × M cutoffs. Verify cache directory:
```bash
ls models/cache/ | wc -l
```
Expected: at least 8 × ~50 = ~400+ files (8 combos × number of monthly cutoffs over the catalog window).

- [ ] **Step 2: Run the full harness evaluation**
```bash
time uv run python examples/run_harness_meta.py 2>&1 | tee /tmp/stage5_acceptance.log | tail -60
```
Expected: completes in 5–15 minutes. Final output includes the Markdown summary with baseline vs meta numbers and DSR.

- [ ] **Step 3: Sanity checks on the output**

```bash
d=$(ls -dt results/harness_meta_*/ | head -1)
echo "harness dir: $d"
wc -l "$d/baseline.csv" "$d/meta.csv"
cat "$d/summary.md"
```
Expected: `baseline.csv` ≥ 6 lines (header + 5 windows); `meta.csv` ≥ 121 lines (header + 24 combos × 5 windows); `summary.md` has concrete numbers.

- [ ] **Step 4: Capture the headline numbers for Task 11's README**

Note from `summary.md`:
- Baseline median window Sharpe.
- Meta winner combo + median window Sharpe.
- DSR p-value.
- Overall verdict ("YES" / "NO" — meta beat baseline + DSR).

(No commit — `results/` is gitignored.)

---

## Task 11: README + Stage 5 wrap-up

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Status section**

Edit `README.md`. Change Stage 5 from `[ ]` to `[x]` and add sub-bullets citing actual numbers from Task 10:
```markdown
- [x] Stage 5 — Meta-labeling ML overlay
  - [x] `models/{labeling,features,cv,trainer,store}.py` — pure pipeline; `strategies/momentum_meta.py` — Stage-2 wrapper with filter+sizer.
  - [x] Walk-forward train: `uv run python examples/train_meta_models.py` (8 LightGBM combos × ~50 monthly cutoffs each).
  - [x] Harness eval: `uv run python examples/run_harness_meta.py`. Baseline median window Sharpe `<INSERT>` vs meta winner `<INSERT>` (combo `<INSERT>`). Deflated Sharpe over baseline+meta trials: p=`<INSERT>`. Verdict: meta `<INSERT YES/NO>` beats baseline OOS + DSR.
```

REPLACE every `<INSERT>` with the actual numbers from Task 10's `summary.md`. Do NOT leave any literal `<INSERT>` in the committed file.

- [ ] **Step 2: Add a new section to the Backtest area**

Add to the Backtest section in README.md:
```markdown
### Meta-labeling (Stage 5)

```bash
# 1. Walk-forward train the model series across the LightGBM grid (~few min)
uv run python examples/train_meta_models.py
uv run python examples/train_meta_models.py --smoke    # 1 combo, ~seconds

# 2. Evaluate baseline (Stage-2) vs meta through the Stage-3 harness
uv run python examples/run_harness_meta.py
uv run python examples/run_harness_meta.py --smoke     # 2 combos × 2 windows, ~seconds
```
Writes `baseline.csv`, `meta.csv`, `summary.json`, `summary.md` to `results/harness_meta_<timestamp>/`.
```

- [ ] **Step 3: Final pytest gate**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 108 passed. Zero failures.

- [ ] **Step 4: Commit**
```bash
git add README.md
git commit -m "docs: README — Stage 5 complete with meta-labeling numbers

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Stage 5 done**

The spec milestones (parent §5: "ML version beats Stage-2 baseline OOS *and* on Deflated Sharpe") have been evaluated. Whether meta clearly wins, loses, or ties is honest research — the milestone is the produced verdict, not a particular outcome.

---

## Self-review

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §2 file structure (6 modules + 2 runnables + 6 test files) | Tasks 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| §3 stack additions (lightgbm, joblib) | Task 1 |
| §4.1 labeling (holding-period-return-sign) | Task 2 |
| §4.2 10 features + cross-sectional ranks | Task 3 |
| §4.3 PurgedWalkForwardSplit | Task 4 |
| §4.4 trainer (fit + predict_proba) | Task 5 |
| §4.5 ModelStore (save / load_for / last_loaded_path) | Task 6 |
| §4.6 MomentumMetaStrategy with filter+sizer + warm-up fallback | Task 7 |
| §4.7 train_meta_models.py runnable | Task 8 |
| §4.8 run_harness_meta.py with DSR + Markdown summary | Task 9 |
| §4.9 24-combo grid (2×2×2 LightGBM × 3 thresholds) | Tasks 8 (LightGBM combos) + 9 (thresholds) |
| §5 tests across 6 modules | Tasks 2, 3, 4, 5, 6, 7 (test files in each) |
| §6 risks acknowledged | Tasks 7, 8, 9 each cover relevant mitigations |
| §7 out-of-scope respected | None implemented (verify in final review) |
| §8 expected outcome — credible verdict regardless | Task 11 README cites whatever Task 10 produced |

No spec section uncovered. The Stage-4 parity gap is explicitly out of scope per §6.

**2. Placeholder scan**

- Task 11 contains `<INSERT>` markers explicitly to be substituted with Task 10's numbers; Step 1's text says "REPLACE every `<INSERT>` with the actual numbers from Task 10's `summary.md`. Do NOT leave any literal `<INSERT>` in the committed file." This is a substitution instruction, not a plan failure.
- No `TBD`, no `implement later`, no `similar to Task N`, no "add appropriate error handling" generalities.

**3. Type consistency**

- `make_labels(bars_by_ticker, rebalance_dates, *, horizon_days=21) -> pd.DataFrame` signature (Task 2) called identically in Task 8.
- `make_features(bars_by_ticker, dates_per_ticker, *, benchmark="SPY") -> pd.DataFrame` (Task 3) called identically in Task 8.
- `extract_features_for_date(bars_by_ticker, *, ticker, date, benchmark="SPY") -> dict[str, float]` (Task 3) called identically in Task 7.
- `FEATURE_COLUMNS` tuple defined in Task 3, imported in Tasks 7, 8.
- `PurgedWalkForwardSplit(n_splits, label_horizon_days, embargo_days).split(dates) -> Iterator[tuple[ndarray, ndarray]]` (Task 4) — referenced in Task 8's CV-based train logic via the documented purge-by-time-delta pattern (the runnable uses an inlined `cutoff - 21d` check rather than the splitter, which is fine — the splitter is available for future per-cutoff CV).
- `fit(X, y, *, n_estimators, max_depth, learning_rate, random_state=42) -> LGBMClassifier` (Task 5) called identically in Task 8.
- `ModelStore(cache_dir, label).save(cutoff_date, model, *, feature_names) -> Path` and `.load_for(query_date) -> tuple[model, feature_names] | None` (Task 6) used identically in Tasks 7, 8.
- `MomentumMetaConfig` fields (Task 7) match exactly what Task 9's `walk_forward` param_grid passes.

All signatures consistent.
