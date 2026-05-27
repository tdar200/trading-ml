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

    # Pre-flight: without trained models the meta strategy silently falls
    # back to baseline on every rebalance, producing a meaningless verdict.
    if not any(CACHE_DIR.glob("*.joblib")):
        print(f"WARNING: {CACHE_DIR} contains no model artifacts. "
              f"Run examples/train_meta_models.py first, otherwise meta "
              f"will fall back to baseline on every rebalance and the "
              f"DSR comparison is meaningless.")

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
    if meta_df.empty:
        print("ERROR: meta walk-forward produced no rows. "
              "Check models/cache/ has artifacts and that the harness "
              "didn't filter out every window.")
        return 1
    print(f"  meta runs: {len(meta_df)}; max median Sharpe: "
          f"{meta_df.groupby(['model_store_label', 'filter_threshold'])['sharpe'].median().max():.4f}")

    # 3. Deflated Sharpe over baseline + meta combos.
    sharpes = pd.concat([base_df["sharpe"], meta_df["sharpe"]]).dropna().tolist()
    # Conservative n_obs: shortest (first) window's day count. Inflates
    # σ_SR_obs and biases the deflated z toward "NO" — appropriate for a
    # meta-overlay where false positives cost more than false negatives.
    n_obs = int(meta_df["n_days"].iloc[0]) if len(meta_df) else int(base_df["n_days"].iloc[0])
    dsr = deflated_sharpe(sharpes=sharpes, n_obs=n_obs)
    print(f"DSR: p-value={dsr['p_value']:.4f}, z={dsr['deflated_sharpe']:.4f}")

    # 4. Summary report.
    baseline_median = float(base_df["sharpe"].median())
    meta_grouped = meta_df.groupby(["model_store_label", "filter_threshold"])["sharpe"].median()
    meta_winner_idx = meta_grouped.idxmax()
    meta_winner_median = float(meta_grouped.max())
    meta_beats = meta_winner_median > baseline_median
    verdict = meta_beats and dsr["p_value"] > 0.95

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
- Overall verdict: **{'YES' if verdict else 'NO'} — meta did{'' if verdict else ' not'} clearly beat baseline.**
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
