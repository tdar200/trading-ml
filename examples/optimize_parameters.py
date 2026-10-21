"""Find robust momentum parameters across independent market windows.

Examples:
    uv run python examples/optimize_parameters.py --algorithm tpe --trials 100
    uv run python examples/optimize_parameters.py --algorithm grid
    uv run python examples/optimize_parameters.py --smoke

The event-driven Nautilus backtest is the source of truth. Search output keeps
every candidate and its per-window evidence under ``results/parameter_optimizer_*``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

from analysis.optimizer import (
    SearchSettings, accelerator_status, build_non_overlapping_windows,
    evaluate_candidates, exhaustive_combinations, search,
)
from analysis.runner import run
from analysis.runspec import RunSpec
from analysis.vectorized_screen import (
    bars_to_close_frame,
    screen_momentum,
    select_diverse_finalists,
)
from data.instruments import VENUE_MIC, build_universe_instruments
from data.tickers import ALL_TICKERS, BENCHMARK, UNIVERSE
from strategies.momentum_topn import MomentumTopNConfig, MomentumTopNStrategy


PARAM_SPACE = {
    "top_n": {"type": "int", "low": 1, "high": 5, "step": 1},
    "lookback_days": {"type": "int", "low": 42, "high": 252, "step": 21},
    "skip_days": {"type": "int", "low": 0, "high": 42, "step": 21},
}
SMOKE_SPACE = {
    "top_n": {"type": "int", "low": 2, "high": 3, "step": 1},
    "lookback_days": {"type": "categorical", "choices": [63]},
    "skip_days": {"type": "categorical", "choices": [0]},
}


def load_data():
    catalog = ParquetDataCatalog(path=str((ROOT / "data" / "catalog").resolve()))
    bars = {
        ticker: catalog.bars(instrument_ids=[f"{ticker}.{VENUE_MIC}"])
        for ticker in ALL_TICKERS
    }
    missing = [ticker for ticker in ALL_TICKERS if not bars[ticker]]
    if missing:
        raise RuntimeError(f"catalog has no bars for {missing}; run data/ingest_alpaca.py")
    return bars, build_universe_instruments()


def catalog_range(bars):
    start_ns = max(bars[ticker][0].ts_event for ticker in UNIVERSE)
    end_ns = min(bars[ticker][-1].ts_event for ticker in ALL_TICKERS)
    start = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc)
    # RunSpec end is exclusive, so move beyond the final available day.
    end = datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc) + timedelta(days=1)
    return start, end


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("hybrid", "grid", "tpe"), default="hybrid")
    parser.add_argument("--trials", type=int, default=100, help="TPE trial budget")
    parser.add_argument("--windows", type=int, default=4, help="non-overlapping evaluation windows")
    parser.add_argument("--finalists", type=int, default=20, help="hybrid candidates sent to Nautilus")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    bars, instruments = load_data()
    start, end = catalog_range(bars)
    n_windows = 2 if args.smoke else args.windows
    windows = build_non_overlapping_windows(start, end, n_windows)
    space = SMOKE_SPACE if args.smoke else PARAM_SPACE
    algorithm = args.algorithm
    settings = SearchSettings(
        algorithm="grid" if algorithm in {"grid", "hybrid"} else "tpe",
        n_trials=2 if args.smoke else args.trials,
        seed=args.seed,
    )
    if algorithm in {"grid", "hybrid"}:
        combos = len(exhaustive_combinations(space))
    else:
        combos = settings.n_trials
    print(f"parameter optimizer: {algorithm}; up to {combos} candidates × {len(windows)} windows")
    print(f"accelerator: {accelerator_status()}")

    def evaluator(params, window):
        result = run(
            RunSpec(
                label=f"optimizer-{window[0].date()}-{window[1].date()}",
                strategy_class=MomentumTopNStrategy,
                strategy_config=MomentumTopNConfig(
                    universe=UNIVERSE, benchmark=BENCHMARK, **params,
                ),
                start=window[0], end=window[1], slippage_prob=0.25,
                random_seed=args.seed,
                log_level="ERROR",
            ),
            bars_by_ticker=bars,
            instruments=instruments,
        )
        return result.metrics

    screened = None
    if algorithm == "hybrid":
        candidates = exhaustive_combinations(space)
        close_frame = bars_to_close_frame(bars, UNIVERSE)
        screened = screen_momentum(close_frame, candidates, device=args.device)
        finalist_count = min(len(candidates), 2 if args.smoke else args.finalists)
        finalists = select_diverse_finalists(screened, list(space), finalist_count)
        finalist_params = finalists[list(space)].to_dict(orient="records")
        print(f"pre-screen: {len(screened)} on {screened.iloc[0]['screen_device']}; validating {len(finalist_params)} finalists")
        ranked, evaluations = evaluate_candidates(
            base_params={}, candidates=finalist_params, windows=windows,
            evaluator=evaluator, settings=settings, param_names=list(space),
        )
    else:
        ranked, evaluations = search(
            base_params={}, param_space=space, windows=windows,
            evaluator=evaluator, settings=settings,
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"parameter_optimizer_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out / "rankings.csv", index=False)
    evaluations.to_csv(out / "window_evaluations.csv", index=False)
    if screened is not None:
        screened.to_csv(out / "prescreen.csv", index=False)
    best = ranked[ranked["eligible"]].iloc[0].to_dict() if ranked["eligible"].any() else None
    summary = {
        "algorithm": algorithm,
        "candidate_count": len(ranked),
        "prescreen_count": len(screened) if screened is not None else None,
        "window_count": len(windows),
        "accelerator": accelerator_status(),
        "best": best,
        "pareto_candidates": int(ranked["pareto_optimal"].sum()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(ranked.head(20).to_string(index=False))
    print(f"artifacts: {out}")
    return 0 if best is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
