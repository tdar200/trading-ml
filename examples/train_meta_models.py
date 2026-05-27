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
