"""
Stage-3 honesty harness: walk-forward × parameter grid + DSR + cost stress
on the Stage-2 momentum strategy, against the Stage-1 catalog.

Usage:
    uv run python examples/run_harness.py            # full battery, ~45-50 min
    uv run python examples/run_harness.py --smoke    # tiny grid, ~6 min

Writes results/harness_<timestamp>/ with walk_forward.csv, cost_stress.csv,
summary.json, summary.md. Prints summary.md to stdout.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

from analysis.costs import cost_stress
from analysis.deflated_sharpe import deflated_sharpe
from analysis.harness_report import write_report
from analysis.runspec import RunSpec
from analysis.walk_forward import walk_forward
from data.instruments import VENUE_MIC, build_universe_instruments
from data.tickers import ALL_TICKERS, BENCHMARK, UNIVERSE
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy


CATALOG_PATH = _PROJECT_ROOT / "data" / "catalog"
RESULTS_ROOT = _PROJECT_ROOT / "results"


# Stage-3 default param grid (spec §3): 3 × 3 × 2 = 18 combos.
DEFAULT_PARAM_GRID = {
    "top_n": [2, 3, 5],
    "lookback_days": [63, 126, 252],
    "skip_days": [0, 21],
}
SMOKE_PARAM_GRID = {"top_n": [3], "lookback_days": [126], "skip_days": [21]}

DEFAULT_SLIPPAGE_PROBS = [0.0, 0.25, 0.5, 1.0]
SMOKE_SLIPPAGE_PROBS = [0.0, 0.25]


def _load_catalog_data(catalog: ParquetDataCatalog) -> tuple[dict, list]:
    bars_by_ticker: dict = {}
    for ticker in ALL_TICKERS:
        bars = catalog.bars(instrument_ids=[f"{ticker}.{VENUE_MIC}"])
        bars_by_ticker[ticker] = bars
        print(f"  {ticker}: {len(bars)} bars")
    instruments = build_universe_instruments()
    return bars_by_ticker, instruments


def _build_windows(bars_by_ticker: dict, n_windows: int) -> list[tuple[datetime, datetime]]:
    """Build n_windows expanding test windows over the universe-bar period."""
    universe_starts = [
        bars_by_ticker[t][0].ts_event
        for t in UNIVERSE if bars_by_ticker.get(t)
    ]
    if not universe_starts:
        raise RuntimeError("no universe bars to slice")
    start_ns = max(universe_starts)
    all_ends = [bars_by_ticker[t][-1].ts_event for t in ALL_TICKERS if bars_by_ticker.get(t)]
    end_ns = max(all_ends)

    start = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc)
    end = datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc)
    span = end - start
    out: list[tuple[datetime, datetime]] = []
    for i in range(1, n_windows + 1):
        window_end = start + (span * i) / n_windows
        out.append((start, window_end))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Tiny grid + few windows for dev iteration (~6 min)")
    args = parser.parse_args()

    grid = SMOKE_PARAM_GRID if args.smoke else DEFAULT_PARAM_GRID
    slippage_probs = SMOKE_SLIPPAGE_PROBS if args.smoke else DEFAULT_SLIPPAGE_PROBS
    n_windows = 2 if args.smoke else 5

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS_ROOT / f"harness_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")
    print(f"mode: {'SMOKE' if args.smoke else 'FULL'}")

    catalog = ParquetDataCatalog(path=str(CATALOG_PATH.resolve()))
    print("loading catalog...")
    bars_by_ticker, instruments = _load_catalog_data(catalog)
    total_bars = sum(len(v) for v in bars_by_ticker.values())
    print(f"total bars: {total_bars}")
    if total_bars == 0:
        print("ERROR: catalog is empty. Run data/ingest_alpaca.py first.")
        return 1

    windows = _build_windows(bars_by_ticker, n_windows=n_windows)
    print(f"windows ({len(windows)}):")
    for i, (s, e) in enumerate(windows):
        print(f"  [{i}] {s.date()} -> {e.date()}")

    base_kwargs = dict(universe=UNIVERSE, benchmark=BENCHMARK)
    print("running walk-forward...")
    wf_df = walk_forward(
        strategy_class=MomentumTopNStrategy,
        config_class=MomentumTopNConfig,
        base_config_kwargs=base_kwargs,
        param_grid=grid,
        windows=windows,
        bars_by_ticker=bars_by_ticker,
        instruments=instruments,
        slippage_prob=0.25,
    )
    print(f"walk-forward done: {len(wf_df)} runs")

    # Pick winner: highest median window Sharpe.
    param_cols = list(grid.keys())
    medians = wf_df.groupby(param_cols)["sharpe"].median().sort_values(ascending=False)
    winner_tuple = medians.index[0]
    if not isinstance(winner_tuple, tuple):
        winner_tuple = (winner_tuple,)
    winner_kwargs = {**base_kwargs, **dict(zip(param_cols, winner_tuple, strict=True))}
    print(f"winner: {dict(zip(param_cols, winner_tuple, strict=True))}")

    # Full-catalog winner spec for cost stress + DSR n_obs.
    earliest_start = min(b.ts_event for b in bars_by_ticker[UNIVERSE[0]])
    latest_end = max(b.ts_event for b in bars_by_ticker[BENCHMARK])
    full_start = datetime.fromtimestamp(earliest_start / 1e9, tz=timezone.utc)
    full_end = datetime.fromtimestamp(latest_end / 1e9, tz=timezone.utc)
    winner_spec = RunSpec(
        label="winner-fullcatalog",
        strategy_class=MomentumTopNStrategy,
        strategy_config=MomentumTopNConfig(**winner_kwargs),
        start=full_start,
        end=full_end,
        slippage_prob=0.25,
    )

    print("running cost stress...")
    cost_df = cost_stress(
        base_spec=winner_spec,
        slippage_probs=slippage_probs,
        bars_by_ticker=bars_by_ticker,
        instruments=instruments,
    )
    print(f"cost stress done: {len(cost_df)} runs")

    # DSR: all per-(combo, window) Sharpes as the trial set; n_obs = baseline-slippage n_days.
    sharpes = wf_df["sharpe"].dropna().tolist()
    matching = cost_df[cost_df["slippage_prob"] == 0.25]
    n_obs = int(matching.iloc[0]["n_days"]) if len(matching) else int(cost_df.iloc[0]["n_days"])
    dsr = deflated_sharpe(sharpes=sharpes, n_obs=n_obs)
    print(f"DSR: p-value={dsr['p_value']:.4f}, deflated_z={dsr['deflated_sharpe']:.4f}")

    write_report(out_dir=out_dir, wf_df=wf_df, cost_df=cost_df, dsr=dsr,
                 strategy_name="MomentumTopN")
    md = (out_dir / "summary.md").read_text()
    print("\n" + md)
    print(f"\nartifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
