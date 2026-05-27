"""
Stage-6 ablation: compare price-only (v1) vs price+sentiment (v2) meta-models.

For each held-out walk-forward window:
  1. Score the test fold with both v1 and v2 models.
  2. Compute per-window Spearman IC between meta P(win) and label.
  3. Compute ΔIC = IC_v2 - IC_v1.

Then run a permutation test on the sentiment features within the test set
(re-predict with shuffled sentiment columns) to build a null distribution
of ΔIC under H0 (sentiment is noise).

Verdict per spec §6:
    median ΔIC > 0 AND permutation p < 0.05  → sentiment adds real signal
    otherwise                                → clean rejection

Writes results/sentiment_ablation_<timestamp>/summary.md + summary.json.

Usage:
    uv run python examples/ablate_sentiment.py
    uv run python examples/ablate_sentiment.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data.instruments import VENUE_MIC
from data.tickers import ALL_TICKERS, BENCHMARK, UNIVERSE
from models.features import FEATURE_COLUMNS, SENTIMENT_FEATURE_COLUMNS, make_features
from models.labeling import make_labels
from models.store import ModelStore
from nlp.sentiment_store import SentimentStore
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


RESULTS_ROOT = _PROJECT_ROOT / "results"
CATALOG_PATH = _PROJECT_ROOT / "data" / "catalog"
CACHE_V1 = _PROJECT_ROOT / "models" / "cache"
CACHE_V2 = _PROJECT_ROOT / "models" / "cache_v2"
SENTIMENT_CACHE = _PROJECT_ROOT / "nlp" / "cache"
SENTIMENT_MODEL_VERSION = "finbert-v1"


# ---- pure utilities (tested in tests/test_ablation.py) ---------------------

def spearman_ic(preds: np.ndarray, labels: np.ndarray) -> float:
    """Spearman rank correlation between preds and labels."""
    if len(preds) < 2:
        return float("nan")
    rho, _ = spearmanr(preds, labels)
    return float(rho) if not np.isnan(rho) else 0.0


def permutation_p_value(observed: float, null_dist: np.ndarray) -> float:
    """One-sided p-value: fraction of null samples >= observed."""
    return float(np.mean(np.asarray(null_dist) >= observed))


# ---- orchestration ---------------------------------------------------------

def _bars_to_series(bars) -> pd.Series:
    rows = [
        (pd.Timestamp(b.ts_event, unit="ns", tz="UTC").tz_convert(None).normalize(),
         float(b.close))
        for b in bars
    ]
    s = pd.Series(dict(rows), name="close").sort_index()
    return s[~s.index.duplicated(keep="last")]


def _ic_for_combo(
    dataset: pd.DataFrame,
    store: ModelStore,
    feature_cols: list[str],
    cutoff: pd.Timestamp,
    test_end: pd.Timestamp,
) -> float | None:
    """Score the test fold with the model that was trained at this cutoff."""
    test = dataset[(dataset["date"] >= cutoff) & (dataset["date"] < test_end)]
    if test.empty:
        return None
    loaded = store.load_for(cutoff)
    if loaded is None:
        return None
    model, _ = loaded
    X = test[feature_cols]
    proba = model.predict_proba(X)
    preds = proba[:, 1] if proba.ndim == 2 else proba
    return spearman_ic(preds, test["label"].to_numpy())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="3 cutoffs only, 100 perm draws.")
    parser.add_argument("--n-perm", type=int, default=1000)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS_ROOT / f"sentiment_ablation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_dir}")

    # 1. Load catalog + build features + labels (v2 features so we have both sets).
    catalog = ParquetDataCatalog(path=str(CATALOG_PATH.resolve()))
    bars_by_ticker = {
        t: _bars_to_series(catalog.bars(instrument_ids=[f"{t}.{VENUE_MIC}"]))
        for t in ALL_TICKERS
    }
    spy = bars_by_ticker[BENCHMARK]
    df = spy.to_frame(name="close")
    df["month"] = df.index.to_period("M")
    grouped = df.groupby("month", as_index=False).head(1)
    rebalance_dates = pd.DatetimeIndex(grouped.index)
    labels = make_labels(
        {t: bars_by_ticker[t] for t in UNIVERSE},
        rebalance_dates, horizon_days=21,
    )

    sentiment_store = SentimentStore(
        cache_dir=SENTIMENT_CACHE, model_version=SENTIMENT_MODEL_VERSION,
    )
    dates_per_ticker = {
        t: list(rebalance_dates[rebalance_dates.isin(bars_by_ticker[t].index)])
        for t in UNIVERSE
    }
    features = make_features(
        bars_by_ticker, dates_per_ticker, benchmark=BENCHMARK,
        news_provider=sentiment_store,
    )
    dataset = features.merge(labels, on=["ticker", "date"], how="inner")
    v2_cols = list(FEATURE_COLUMNS) + list(SENTIMENT_FEATURE_COLUMNS)
    dataset = dataset.dropna(subset=v2_cols + ["label"]).reset_index(drop=True)
    print(f"dataset: {len(dataset)} rows × ({len(v2_cols)} feats + label)")

    # 2. Choose cutoffs (every monthly rebalance after warmup_months=12).
    cutoffs = rebalance_dates[12:]
    if args.smoke:
        cutoffs = cutoffs[:3]
    print(f"cutoffs: {len(cutoffs)}")

    # 3. For each cutoff, compute v1 and v2 IC on the test fold.
    # Pick a single combo per side as the "representative" model (the one
    # whose label name is alphabetically first — matches train_meta_models'
    # iteration order; the harness winner is a separate concern).
    combo_label = "n100_d3_lr05"  # first combo in DEFAULT_GRID order
    store_v1 = ModelStore(cache_dir=CACHE_V1, label=combo_label)
    store_v2 = ModelStore(cache_dir=CACHE_V2, label=combo_label)

    per_fold = []
    for i, cutoff in enumerate(cutoffs):
        # Test fold = next 21 days after the cutoff.
        test_end = cutoff + pd.Timedelta(days=21)
        ic_v1 = _ic_for_combo(dataset, store_v1, list(FEATURE_COLUMNS), cutoff, test_end)
        ic_v2 = _ic_for_combo(dataset, store_v2, v2_cols, cutoff, test_end)
        if ic_v1 is None or ic_v2 is None:
            continue
        per_fold.append({"cutoff": cutoff, "ic_v1": ic_v1, "ic_v2": ic_v2,
                         "delta_ic": ic_v2 - ic_v1})
    fold_df = pd.DataFrame(per_fold)
    if fold_df.empty:
        print("ERROR: no folds produced usable IC values.")
        return 1
    median_delta = float(fold_df["delta_ic"].median())
    print(f"\nper-fold ΔIC (median = {median_delta:+.4f}):")
    print(fold_df.to_string(index=False))

    # 4. Permutation null: for each fold, re-predict v2 with shuffled sentiment
    # columns; compute the median ΔIC across folds for each draw.
    rng = np.random.default_rng(42)
    n_perm = 100 if args.smoke else args.n_perm
    null_medians = np.zeros(n_perm)
    for k in range(n_perm):
        fold_deltas = []
        for cutoff in fold_df["cutoff"]:
            test_end = cutoff + pd.Timedelta(days=21)
            test = dataset[(dataset["date"] >= cutoff) & (dataset["date"] < test_end)]
            if test.empty:
                continue
            loaded_v2 = store_v2.load_for(cutoff)
            loaded_v1 = store_v1.load_for(cutoff)
            if loaded_v2 is None or loaded_v1 is None:
                continue
            model_v2, _ = loaded_v2
            model_v1, _ = loaded_v1
            X = test[v2_cols].copy()
            for col in SENTIMENT_FEATURE_COLUMNS:
                X[col] = rng.permutation(X[col].to_numpy())
            proba_v2 = model_v2.predict_proba(X)
            preds_v2 = proba_v2[:, 1]
            X_v1 = test[list(FEATURE_COLUMNS)]
            proba_v1 = model_v1.predict_proba(X_v1)
            preds_v1 = proba_v1[:, 1]
            labels = test["label"].to_numpy()
            fold_deltas.append(spearman_ic(preds_v2, labels) - spearman_ic(preds_v1, labels))
        if fold_deltas:
            null_medians[k] = float(np.median(fold_deltas))
    p_value = permutation_p_value(median_delta, null_medians)
    print(f"\npermutation null (n={n_perm}): mean={null_medians.mean():.4f}, "
          f"std={null_medians.std():.4f}, p_value={p_value:.4f}")

    # 5. Verdict + report.
    if median_delta > 0 and p_value < 0.05:
        verdict = "REAL — sentiment adds incremental signal"
    elif median_delta < 0 and p_value > 0.95:
        verdict = "HURTS — sentiment degrades meta-model"
    else:
        verdict = "REJECT — sentiment is noise within permutation tolerance"

    md = f"""# Stage-6 Sentiment Ablation — {timestamp}

## Methodology
- Combo under test: `{combo_label}`
- Folds (cutoffs): {len(fold_df)}
- Permutation draws: {n_perm}

## Per-fold IC
{fold_df.to_markdown(index=False)}

## Summary
- Median ΔIC: **{median_delta:+.4f}**
- Permutation p-value (one-sided, ΔIC ≥ observed under H0): **{p_value:.4f}**

## Verdict: **{verdict}**
"""
    (out_dir / "summary.md").write_text(md)
    (out_dir / "summary.json").write_text(json.dumps({
        "median_delta_ic": median_delta,
        "p_value": p_value,
        "n_folds": int(len(fold_df)),
        "n_perm": int(n_perm),
        "combo_label": combo_label,
        "verdict": verdict,
    }, indent=2, default=str))
    fold_df.to_csv(out_dir / "per_fold.csv", index=False)
    print(f"\n{md}")
    print(f"artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
