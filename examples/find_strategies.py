"""Discover rule structures, tune their parameters, and validate finalists.

Examples:
    .venv/bin/python examples/find_strategies.py --smoke --device cuda
    .venv/bin/python examples/find_strategies.py --finalists 24 --windows 4
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.optimizer import SearchSettings, accelerator_status, build_non_overlapping_windows
from analysis.runner import run
from analysis.runspec import RunSpec
from analysis.strategy_finder import (
    FinderSettings,
    StrategyGrammar,
    describe_strategy,
    find_strategies,
    generate_candidates,
    generate_structures,
)
from analysis.vectorized_screen import bars_to_close_frame
from data.tickers import BENCHMARK, UNIVERSE
from examples.optimize_parameters import catalog_range, load_data
from strategies.rule_rotation import RuleRotationConfig, RuleRotationStrategy


PARAMETER_SPACE = {
    "top_n": {"type": "int", "low": 1, "high": 5, "step": 1},
    "lookback_days": {"type": "int", "low": 21, "high": 252, "step": 21},
    "skip_days": {"type": "int", "low": 0, "high": 42, "step": 21},
}
SMOKE_SPACE = {
    "top_n": {"type": "int", "low": 2, "high": 3, "step": 1},
    "lookback_days": {"type": "int", "low": 63, "high": 63, "step": 1},
    "skip_days": {"type": "int", "low": 0, "high": 0, "step": 1},
}
CONFIG_FIELDS = {
    "signal_family", "trend_filter_days", "require_positive",
    "top_n", "lookback_days", "skip_days",
}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--finalists", type=int, default=24)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--complexity-penalty", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    bars, instruments = load_data()
    start, end = catalog_range(bars)
    n_windows = 2 if args.smoke else args.windows
    windows = build_non_overlapping_windows(start, end, n_windows)
    grammar = (
        StrategyGrammar(
            signal_families=("momentum", "mean_reversion"),
            trend_filter_days=(0,),
            positive_gates=(False,),
        )
        if args.smoke else StrategyGrammar()
    )
    structures = generate_structures(grammar)
    parameter_space = SMOKE_SPACE if args.smoke else PARAMETER_SPACE
    candidate_count = len(generate_candidates(structures, parameter_space))
    finalist_count = min(4 if args.smoke else args.finalists, candidate_count)
    closes = bars_to_close_frame(bars, UNIVERSE)
    print(
        f"strategy finder: {len(structures)} structures, {candidate_count} "
        f"structure/parameter candidates, {finalist_count} finalists × {n_windows} windows"
    )
    print(f"accelerator: {accelerator_status()}")

    def evaluator(params, window):
        config_params = {key: params[key] for key in CONFIG_FIELDS}
        result = run(
            RunSpec(
                label=f"strategy-{params['structure_id']}-{window[0].date()}",
                strategy_class=RuleRotationStrategy,
                strategy_config=RuleRotationConfig(
                    universe=UNIVERSE,
                    benchmark=BENCHMARK,
                    **config_params,
                ),
                start=window[0],
                end=window[1],
                slippage_prob=0.25,
                random_seed=args.seed,
                log_level="ERROR",
            ),
            bars_by_ticker=bars,
            instruments=instruments,
        )
        return result.metrics

    ranked, evaluations, screened = find_strategies(
        closes=closes,
        structures=structures,
        parameter_space=parameter_space,
        windows=windows,
        evaluator=evaluator,
        settings=FinderSettings(
            finalists=finalist_count,
            complexity_penalty=args.complexity_penalty,
            device=args.device,
            search=SearchSettings(algorithm="grid", seed=args.seed),
        ),
    )
    ranked.insert(1, "description", [
        describe_strategy(row) for _, row in ranked.iterrows()
    ])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"strategy_finder_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out / "rankings.csv", index=False)
    evaluations.to_csv(out / "window_evaluations.csv", index=False)
    screened.to_csv(out / "prescreen.csv", index=False)
    best = ranked[ranked["eligible"]].iloc[0].to_dict() if ranked["eligible"].any() else None
    summary = {
        "structure_count": len(structures),
        "candidate_count": candidate_count,
        "finalist_count": len(ranked),
        "window_count": n_windows,
        "complexity_penalty": args.complexity_penalty,
        "accelerator": accelerator_status(),
        "best": best,
    }
    (out / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8",
    )

    columns = [
        "rank", "structure_id", "top_n", "lookback_days", "skip_days",
        "eligible", "robust_score", "median_sharpe", "worst_drawdown",
        "complexity", "pareto_optimal",
    ]
    print(ranked[columns].to_string(index=False))
    if best:
        print(f"best structure: {best['description']}")
    print(f"artifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
