# Stage 5 — Meta-Labeling ML Overlay: Design Spec

**Date:** 2026-05-27
**Status:** Approved through brainstorming; pending user spec-review before implementation planning.
**Parent spec:** `2026-05-26-trading-ml-design.md` (architecture, philosophy, full roadmap)
**Stage:** 5 of 8 — ML as a *filter + sizer* on the Stage-2 momentum signal, evaluated through the Stage-3 harness.

## 1. Goal & milestone

Build a LightGBM meta-labeling layer that decides, for each (ticker, rebalance-date) pair the Stage-2 primary signal would consider, whether *this specific position* is worth taking — and at what size. The meta-model does **not** re-optimize the primary signal; it filters and scales its decisions.

**Milestone (parent spec §5):** "ML version beats Stage-2 baseline OOS *and* on Deflated Sharpe."

Honest framing per the project's discipline: this is a **bar to be tested, not assumed**. The Stage-3 harness already confirmed the primary momentum signal has DSR p=1.00 / z=23.3 with n_trials=90 — a strong, real signal. Meta-labeling typically yields 5–20% Sharpe improvement when it works and zero when it doesn't. The Stage-5 deliverable is "a credible ML pipeline run through the harness with honest verdict," not "ML beats baseline."

## 2. Architecture

```
models/                              # NEW package
  __init__.py
  labeling.py                # (ticker, date) → binary label (next-rebalance return sign)
  features.py                # ~10 no-lookahead features per (ticker, date)
  cv.py                      # PurgedWalkForwardSplit (López de Prado canonical)
  trainer.py                 # LightGBM fit/predict pipeline
  store.py                   # ModelStore: pickled artifacts keyed by training-cutoff date

strategies/
  momentum_meta.py           # MomentumMetaStrategy: filter+sizer wrapping Stage-2

examples/
  train_meta_models.py       # OFFLINE: walk-forward train → save dated artifacts
  run_harness_meta.py        # ONLINE: harness evaluation of baseline vs meta

tests/                       # 6 new files: pure unit tests + synthetic integration
docs/                        # Stage-5 spec (this) + plan

models/cache/                # Pickled model artifacts, gitignored
```

### Architectural commitments

1. **`models/` is the only package that touches LightGBM, joblib, or sklearn.** `strategies/`, `analysis/`, `livesim/`, `data/` stay ML-free. The `MomentumMetaStrategy` consumes a trained `ModelStore` instance — it doesn't import LightGBM directly.

2. **Walk-forward is real, not aspirational.** Each model artifact is named with its training-cutoff date (e.g. `models/cache/meta_2024-03-01.joblib`). The strategy at rebalance time loads the most-recent artifact whose cutoff is strictly *before* today. No accidental peeking at the future.

3. **Purged + embargoed CV gates training.** When training a model with cutoff `T`, any label whose forward horizon `[t, t+21d]` overlaps `(T - 21d, T]` is purged; an additional 5-trading-day embargo separates train from test fold. Implemented in `models/cv.py` as a generator.

4. **Filter+sizer is one combined formula.** Pure filter (size always 1.0) and pure sizer (no skips) are special cases. The combined form is the canonical mode.

5. **Evaluation is exclusively through the Stage-3 harness.** Walk-forward over time windows × meta-hyperparameter grid is exactly what `analysis/walk_forward.py` was built for. We add one new `RunSpec` field (`model_store_path`) and one new harness runnable that compares baseline vs meta.

## 3. Stack additions

| Concern | Choice | Notes |
|---|---|---|
| Tree model | `lightgbm>=4.5` | NEW dependency added in Task 1 |
| Model persistence | `joblib>=1.4` | Already transitive via scikit-learn; pinned explicitly |
| Pure-math stack | `numpy`, `pandas` | Already present |

`pyproject.toml` `[project].dependencies` gains:
```toml
"lightgbm>=4.5",
"joblib>=1.4",
```

## 4. Key behaviors

### 4.1 Labeling (`models/labeling.py`)

For every (ticker, rebalance-date) pair in the catalog window:

```python
spot_close   = close[t]
next_t       = next_rebalance_date_after(t)   # ~21 trading days later
forward_close = close[next_t]
label = 1 if (forward_close / spot_close - 1.0) > 0 else 0
```

Edge cases:
- Skip pairs without 21 trading days of forward bars (last 21 days of catalog).
- Skip pairs without enough lookback history for the features (first ~252 trading days, set by the longest lookback in §4.2).

Public API: `make_labels(bars_by_ticker, rebalance_dates) -> pd.DataFrame` with columns `[ticker, date, label]`.

The benchmark `SPY` is in the labeling input (its bars are used for market-regime features) but **NOT labeled** — only the 11 tradeable tickers get rows in the label frame.

### 4.2 Features (`models/features.py`) — 10 features

| Feature | Formula | Lookback |
|---|---|---|
| `mom_6m` | `close[t] / close[t-126d] - 1.0` | 126d |
| `mom_1m` | `close[t] / close[t-21d] - 1.0` | 21d |
| `mom_3m_skip_1m` | `close[t-21d] / close[t-84d] - 1.0` | 84d |
| `vol_20d` | std of daily log returns over 20d, annualized | 20d |
| `vol_60d` | std of daily log returns over 60d, annualized | 60d |
| `rank_mom_6m` | rank of `mom_6m` among universe at date `t`, scaled 0..1 | 126d |
| `rank_vol_60d` | rank of `vol_60d` among universe at date `t`, scaled 0..1 | 60d |
| `spy_vol_60d` | SPY's `vol_60d` | 60d |
| `spy_mom_6m` | SPY's `mom_6m` | 126d |
| `month_sin` / `month_cos` | `sin(2π·month/12)` and `cos(2π·month/12)` | n/a |

Total = 11 columns (month encoded as 2 features). All computed from data strictly before `t`.

Public API: `make_features(bars_by_ticker, dates_per_ticker) -> pd.DataFrame` with columns `[ticker, date, mom_6m, mom_1m, ..., month_cos]`. Joined with labels on `(ticker, date)` to form the training matrix.

### 4.3 Purged-embargoed walk-forward CV (`models/cv.py`)

```python
class PurgedWalkForwardSplit:
    def __init__(self, n_splits: int, label_horizon_days: int = 21, embargo_days: int = 5): ...
    def split(self, dates: pd.DatetimeIndex) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) pairs with purging + embargo applied."""
```

Splits ordered by date; for each split:
- `train` = all labels whose `(date + label_horizon_days)` is strictly before `test_start - embargo_days`
- `test`  = labels whose `date` is in the test fold's date range

Used internally by `trainer.py` for cross-validated hyperparameter selection and by `train_meta_models.py` for rolling fit-and-save.

### 4.4 Training (`models/trainer.py`)

```python
def fit(X_train, y_train, *, n_estimators, max_depth, learning_rate, random_state=42) -> LGBMClassifier:
    """Fit a LightGBM binary classifier with the given hyperparams."""

def predict_proba(model, X) -> np.ndarray:
    """Return P(win) array for each row in X."""
```

Pure-function-style API. No side effects. The trainer is unaware of walk-forward; that's `train_meta_models.py`'s job.

### 4.5 Model store (`models/store.py`)

```python
@dataclass(frozen=True)
class ModelStore:
    cache_dir: Path
    label: str   # human-readable suffix in filenames, e.g. "default"

    def save(self, cutoff_date: pd.Timestamp, model, feature_names: list[str]) -> Path:
        """Pickle model + feature_names to cache_dir/{label}_{cutoff_date:%Y-%m-%d}.joblib."""

    def load_for(self, query_date: pd.Timestamp) -> tuple[Any, list[str]] | None:
        """Return the (model, feature_names) whose cutoff_date is the most recent
        strictly before query_date. None if no such model exists yet (warm-up)."""
```

Strategy-side helper: pick the right historical model for a given decision date. If `load_for` returns `None` (warm-up period before first model trained), the strategy falls back to pure baseline behaviour for that rebalance.

### 4.6 Strategy (`strategies/momentum_meta.py`)

```python
class MomentumMetaConfig(StrategyConfig, frozen=True):
    # Inherited from Stage-2:
    universe: tuple[str, ...]
    benchmark: str = "SPY"
    top_n: int = 3
    lookback_days: int = 63   # Stage-3 winner
    skip_days: int = 0        # Stage-3 winner
    # New for Stage 5:
    model_store_path: str               # filesystem path to models/cache subdirectory
    filter_threshold: float = 0.55

class MomentumMetaStrategy(MomentumTopNStrategy):
    """Stage-2 primary + Stage-5 ML filter+sizer overlay."""
```

`_rebalance` override:
1. Compute the primary ranking via `rank_by_momentum` (unchanged Stage-2 logic).
2. Take the top-N candidate tickers.
3. Load `ModelStore.load_for(current_date)`. If `None`: pure-baseline rebalance (warm-up).
4. For each candidate, extract features online from `_closes` history → `predict_proba` → `p_win`.
5. Apply filter+sizer:
   ```python
   if p_win < filter_threshold:
       target_weight = 0.0
   else:
       confidence = (p_win - filter_threshold) / (1.0 - filter_threshold)
       target_weight = (equity / top_n) * confidence
   ```
6. Reconcile current positions to new target weights (Stage-2's existing sells-before-buys + integer-shares + 0.98 cash buffer all unchanged).

Crucially: **the feature extraction at rebalance time uses the same `models/features.py` code as training**, but reads from the strategy's in-memory `_closes` ring buffer (already maintained per Stage 2). No catalog reads from inside the engine.

### 4.7 Offline training runnable (`examples/train_meta_models.py`)

```bash
uv run python examples/train_meta_models.py --cache-dir models/cache --label default
```

1. Load catalog → `bars_by_ticker`.
2. Build rebalance-date schedule (first business day of each month from 2021-02 to today).
3. Generate labels via `models/labeling.py`.
4. Generate features via `models/features.py`.
5. Walk-forward train: for each monthly cutoff date (starting after ~12 months of training data):
   - Take all (features, labels) with `date < cutoff_date - purge_horizon`.
   - Fit LightGBM with default hyperparams.
   - Save via `ModelStore.save(cutoff, model)`.
6. Print summary: number of models trained, OOS accuracy per model.

This is the offline preprocessing step. Run once per catalog refresh.

### 4.8 Harness evaluation runnable (`examples/run_harness_meta.py`)

```bash
uv run python examples/run_harness_meta.py
uv run python examples/run_harness_meta.py --smoke
```

1. Run the Stage-3 harness against `MomentumTopNStrategy` (baseline, single config) → capture baseline metrics per window.
2. Run the Stage-3 harness against `MomentumMetaStrategy` with the hyperparam grid (see §4.9 below) → capture meta metrics per (combo, window).
3. Compute Deflated Sharpe with `n_trials = 1 + grid_size` (1 baseline + 24 meta combos).
4. Write a comparison Markdown report under `results/harness_meta_<timestamp>/summary.md` with sections:
   - Baseline (Stage-2 winner) summary
   - Meta winner summary
   - DSR for the meta winner
   - Verdict: did meta beat baseline OOS *and* on DSR?

### 4.9 Hyperparameter grid

| Param | Values | Combo count |
|---|---|---|
| `n_estimators` | {100, 200} | 2 |
| `max_depth` | {3, 5} | 2 |
| `learning_rate` | {0.05, 0.1} | 2 |
| `filter_threshold` | {0.50, 0.55, 0.60} | 3 |
| **Total** | | **24** |

Each combo × 5 walk-forward windows = 120 backtests + 1 baseline reference (×5 windows = 5) = **125 harness runs**. With Nautilus caching this should complete in ~5–10 minutes.

Smoke mode: 2 combos (one minimal + one maximal) × 2 windows = 4 runs ≈ ~30 seconds.

## 5. Testing

| Test file | Coverage | Approach |
|---|---|---|
| `tests/test_labeling.py` | label correctness on synthetic series | pure-pandas, no catalog |
| `tests/test_features.py` | each feature's no-lookahead property | pure-pandas, known synthetic series |
| `tests/test_cv.py` | purge+embargo correctness, fold shapes | small synthetic date index |
| `tests/test_trainer.py` | fit+predict round-trip; determinism with `random_state` | tiny synthetic X/y |
| `tests/test_store.py` | save/load round-trip; `load_for` picks most-recent-before-date | tmp_path |
| `tests/test_momentum_meta_strategy.py` | integration: synthetic catalog + stub ModelStore + baseline path equivalence when threshold=0 | mirrors `tests/test_momentum_strategy.py` synthetic pattern |

**Total new tests:** ~15–20.

## 6. Risks & open items

1. **Tiny training set.** ~660 labeled examples is small for ML. Tree models with limited `max_depth=3` mitigate but don't eliminate overfitting risk. Walk-forward + DSR are the honest guards.

2. **Stage-3 harness Cache-vs-LightGBM compatibility.** The harness creates many BacktestEngines; each strategy instance needs to load a `ModelStore` independently. Path passing via `MomentumMetaConfig.model_store_path` is the cleanest mechanism — verify in Task 6 (integration test) that no shared mutable state leaks across runs.

3. **Feature-vs-strategy state drift.** The strategy's `_closes` ring buffer holds raw closes; `models/features.py` must compute features from raw closes in *exactly the same way* training did. Risk of subtle mismatch (e.g. log vs linear returns). Mitigated by sharing the feature module — there's only one implementation.

4. **Walk-forward retrain cadence is monthly** (matches the rebalance). More frequent retraining risks overfit; less frequent risks drift. Hyperparameter; not exposed in this stage's grid to keep DSR's n_trials manageable.

5. **`filter_threshold` interacts with the warm-up period.** Until the first model is trained, the strategy is pure baseline. Some walk-forward windows may have very little post-warmup data; their metrics will be unstable. Documented in the harness summary's `n_days` field.

6. **The Stage-4 real-catalog parity gap** (TradingNode vs BacktestEngine ~29% drift) is **not** in scope here. Stage 5 evaluates exclusively through `BacktestEngine` via the Stage-3 harness — same engine the Stage-3 results came from, so the comparison is apples-to-apples.

7. **LightGBM serialization compatibility.** Models pickled with one LightGBM version may not load with another. We pin `lightgbm>=4.5` and document that the cache must be regenerated on minor-version upgrades. Acceptable for a research project.

## 7. Out of scope (deliberately, do not add)

- **Deep learning, transformer, sequence models.** LightGBM only.
- **Online/incremental learning.** Periodic batch retrain only.
- **Probability calibration** (Platt, isotonic). Raw `predict_proba` suffices at this scale.
- **NLP / sentiment / news features.** Stage 6.
- **SEC EDGAR / fundamental features.** Stage 6 follow-on.
- **Re-training the primary signal.** Meta-model only layers on top.
- **Hyperparameter optimization beyond grid search.** No Bayesian opt, no Optuna.
- **Combinatorial purged CV.** Single walk-forward direction is enough at this data scale.
- **Live execution on Alpaca paper.** Stage 8.
- **Dashboard / monitoring UI.** Stage 7.
- **Stage-4 real-catalog parity fix.** Tracked separately.

## 8. Bottom-line expected outcome

Stage 5 ships:
- A pure-Python `models/` package (labeling + features + CV + trainer + store) with ~15-20 tests.
- A new `MomentumMetaStrategy` that uses the model store as a filter+sizer overlay.
- An offline training runnable that produces a series of dated model artifacts.
- A harness runnable that evaluates baseline vs meta across the 24-combo × 5-window grid and prints a Markdown verdict with DSR.

The **honest verdict** will most likely fall into one of three buckets:
1. Meta produces ≈baseline Sharpe with marginally better max-DD → "ML filter avoids some losers; gains within DSR noise."
2. Meta degrades baseline → "overfit on small training set; signal washed out by DSR penalty."
3. Meta clearly beats baseline OOS *and* on DSR → "real, statistically significant improvement; worth investigating further in Stage 6."

Whichever outcome lands, the milestone is met as long as the pipeline produces a credible, reproducible Markdown verdict with concrete numbers.

---

*This spec is the Stage-5 contract. Implementation plan to be written next via `superpowers:writing-plans`.*
