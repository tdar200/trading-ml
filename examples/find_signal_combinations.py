"""Discover 1–4 primitive-signal strategies and run the complete honesty pipeline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.cross_sectional import combinatorial_probability_overfit
from analysis.deflated_sharpe import deflated_sharpe
from analysis.deployment_gate import candidate_fingerprint, historical_qualification
from analysis.dynamic_profitability import prepare_dynamic_decision_dataset
from analysis.optimizer import SearchSettings, rank_evaluations
from analysis.profitability import align_decision_datasets, slice_decision_dataset
from analysis.profitability_v2 import evaluate_v2_candidate
from analysis.signal_combinations import (
    combined_score, generate_combinations, gpu_screen_combinations,
    orient_and_select_signals, sample_signal_tensor,
)
from analysis.signal_library import PARAMETRIC_INDICATORS, build_indicator_variants
from analysis.sleeves import require_equity_replay_sleeve
from analysis.vectorized_screen import resolve_device
from data.multitf import get_bars


PARAM_FIELDS = [
    "signal_key", "decision_bar", "exit_policy",
    "max_positions_per_side", "min_relative_volume",
]


def _progress(percent: int, message: str) -> None:
    """Emit a line-oriented progress event for CLI and dashboard consumers."""
    print(f"PROGRESS|{percent}|{message}", flush=True)


def _safe(value):
    if isinstance(value, dict): return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if isinstance(value, np.ndarray): return [_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
    return value


def _replay_candidate(row: dict, definition: dict) -> dict:
    return {
        "family": "signal_consensus", "timeframe": "1h",
        "decision_bar": int(row["decision_bar"]),
        "exit_policy": row["exit_policy"],
        "max_positions_per_side": int(row["max_positions_per_side"]),
        "min_edge_bps": 0.0, "edge_cost_multiple": 0.0,
        "min_relative_volume": float(row["min_relative_volume"]),
        "event_policy": "all", "maximum_spread_bps": 8.0,
        "execution_buffer_bps": 0.75, "signal_key": row["signal_key"],
        "signals": definition["signals"], "directions": definition["directions"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sleeve", default="individual_stock")
    parser.add_argument("--intraday-root", type=Path, default=ROOT / "data" / "free_intraday")
    parser.add_argument("--cost-bps", type=float, default=2.5)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--top-signals", type=int, default=60)
    parser.add_argument("--max-combination-size", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--finalists", type=int, default=48)
    parser.add_argument(
        "--indicators", nargs="+", choices=PARAMETRIC_INDICATORS,
        default=list(PARAMETRIC_INDICATORS),
    )
    parser.add_argument("--value-min", type=int, default=3)
    parser.add_argument("--value-max", type=int, default=50)
    parser.add_argument("--value-step", type=int, default=1)
    args = parser.parse_args()
    if not 4 <= args.top_signals <= 60 or not 8 <= args.finalists <= 50:
        parser.error("top-signals must be 4–60 and finalists must be 8–50")
    variants = build_indicator_variants(
        tuple(args.indicators), value_min=args.value_min,
        value_max=args.value_max, value_step=args.value_step,
    )
    if args.top_signals > len(variants):
        parser.error("top-signals cannot exceed the indicator/value grid size")
    signal_names = tuple(variant.key for variant in variants)
    variant_by_key = {variant.key: variant for variant in variants}
    _progress(2, f"Prepared {len(args.indicators)} indicators and {len(variants)} values")
    device = resolve_device(args.device)
    sleeve = require_equity_replay_sleeve(args.sleeve)
    tickers = sleeve.symbols
    kwargs = {"intraday_root": args.intraday_root}
    _progress(5, f"Loading hourly bars for {len(tickers)} instruments")
    bars = {ticker: get_bars(ticker, "1h", **kwargs) for ticker in tickers}
    benchmark = get_bars(sleeve.benchmark, "1h", **kwargs)
    if missing := [ticker for ticker, frame in {**bars, sleeve.benchmark: benchmark}.items() if frame.empty]:
        raise SystemExit("missing 1h bars: " + ", ".join(missing))
    datasets = align_decision_datasets({
        decision: prepare_dynamic_decision_dataset(
            bars, benchmark, decision, min_assets=15, bar_minutes=60,
            exit_after_bars=1 if decision == 1 else None,
        ) for decision in (1, 3)
    })
    _progress(15, "Aligned market data and forward-return targets")
    sessions = len(datasets[1].session_dates)
    warmup, cut = 60, sessions - 252
    if cut - warmup < 120:
        raise SystemExit("at least 432 common sessions are required")
    nested_validation_sessions, purge_sessions = 126, 5
    validation_start = cut - nested_validation_sessions
    orientation = np.arange(warmup, validation_start - purge_sessions)
    discovery = np.arange(validation_start, cut)
    if len(orientation) < 120:
        raise SystemExit("not enough sessions for nested orientation and validation")
    catalog: dict[str, dict] = {}
    screen_frames = []
    per_decision = max(1, args.finalists // 4)
    finalists = []
    tensors = {}
    research_trials = 0
    live_top: list[dict] = []
    for decision_index, (decision, dataset) in enumerate(datasets.items()):
        stage_start = 20 + decision_index * 28
        _progress(stage_start, f"Computing indicator/value grid for decision bar {decision}")
        tensor = sample_signal_tensor(
            bars, dataset.session_dates, decision, bar_minutes=60, variants=variants,
            device=device,
        )
        tensors[decision] = tensor
        oriented = orient_and_select_signals(
            tensor, dataset.forward_return, orientation, top_k=args.top_signals,
            device=device,
        )
        _progress(stage_start + 12, f"Selected {len(oriented)} oriented indicator/value variants")
        combinations = generate_combinations(
            oriented, max_size=args.max_combination_size, signal_names=signal_names,
        )
        _progress(stage_start + 15, f"GPU screening {len(combinations):,} unrestricted combinations")
        batch_updates = 0

        def publish_batch(batch_rows: list[dict], processed: int, total: int) -> None:
            nonlocal batch_updates, live_top
            batch_updates += 1
            for row in batch_rows:
                names = row["signals"].split("+")
                live_top.append({
                    "combination": " + ".join(
                        f"{variant_by_key[name].indicator}({variant_by_key[name].value})"
                        for name in names
                    ),
                    "decision_bar": decision,
                    "signal_count": row["signal_count"],
                    "provisional_sharpe": row["screen_sharpe"],
                    "provisional_return": row["screen_total_return"],
                    "events": row["screen_events"],
                })
            live_top = sorted(
                live_top,
                key=lambda row: (row["provisional_sharpe"], row["provisional_return"]),
                reverse=True,
            )[:25]
            if batch_updates % 8 == 0 or processed == total:
                print("RESULTS|" + json.dumps(_safe(live_top)), flush=True)
                fraction = processed / total if total else 1.0
                _progress(
                    stage_start + 15 + int(11 * fraction),
                    f"GPU screen {processed:,}/{total:,}; live top {len(live_top)}",
                )

        screened = gpu_screen_combinations(
            tensor, dataset.forward_return, dataset.active_mask, combinations,
            discovery, cost_per_side_bps=args.cost_bps, device=device,
            batch_callback=publish_batch,
        )
        _progress(stage_start + 27, f"Completed CUDA screen for decision bar {decision}")
        screened.insert(0, "decision_bar", decision)
        screen_frames.append(screened)
        research_trials += len(screened)
        for _, item in screened.head(per_decision).iterrows():
            definition = combinations[int(item.candidate_index)]
            key = f"d{decision}:" + "+".join(
                ("+" if direction > 0 else "-") + signal
                for signal, direction in zip(definition["signals"], definition["directions"], strict=True)
            )
            catalog[key] = definition
            for positions in (3, 5):
                finalists.append({
                    "signal_key": key, "decision_bar": decision,
                    "exit_policy": "one_bar" if decision == 1 else "session_close",
                    "max_positions_per_side": positions, "min_relative_volume": 0.75,
                })
    finalists = finalists[:args.finalists]
    windows = np.array_split(discovery, 4)
    rows = []
    score_cache = {
        key: combined_score(tensors[int(row["decision_bar"])], catalog[key])
        for row in finalists for key in (row["signal_key"],)
    }
    total_replays = len(windows) * len(finalists)
    completed_replays = 0
    _progress(77, f"Canonical validation of {len(finalists)} finalists across {len(windows)} windows")
    for window_index, indices in enumerate(windows):
        start, stop = int(indices[0]), int(indices[-1]) + 1
        for row in finalists:
            decision = int(row["decision_bar"])
            candidate = _replay_candidate(row, catalog[row["signal_key"]])
            result, _ = evaluate_v2_candidate(
                slice_decision_dataset(datasets[decision], start, stop), candidate,
                cost_per_side_bps=args.cost_bps,
                forecast_cache={"signal_consensus": score_cache[row["signal_key"]][start:stop]},
                device=device,
            )
            rows.append({"window_idx": window_index, **row, **{
                name: result[name] for name in ("sharpe", "cagr", "max_drawdown", "n_trades", "portfolio_events")
            }})
            completed_replays += 1
            if completed_replays % max(1, total_replays // 10) == 0:
                _progress(
                    77 + int(13 * completed_replays / total_replays),
                    f"Canonical validation {completed_replays}/{total_replays}",
                )
    evaluations = pd.DataFrame(rows)
    ranked = rank_evaluations(
        evaluations, PARAM_FIELDS,
        SearchSettings(algorithm="grid", min_trades_per_window=20, stability_penalty=0.75, drawdown_penalty=0.3),
    )
    eligible = ranked[
        ranked.eligible & (ranked.robust_score > 0) &
        (ranked.median_sharpe > 0) & (ranked.worst_sharpe > 0) &
        (ranked.median_cagr > 0)
    ]
    if eligible.empty:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = ROOT / "results" / f"signal_strategy_discovery_reject_{stamp}"
        out.mkdir(parents=True)
        pd.concat(screen_frames, ignore_index=True).to_csv(out / "gpu_research_screen.csv", index=False)
        ranked.to_csv(out / "canonical_discovery_rankings.csv", index=False)
        evaluations.to_csv(out / "nested_validation_windows.csv", index=False)
        (out / "summary.json").write_text(json.dumps({
            "strategy_version": 5, "verdict": "DISCOVERY_REJECT",
            "deployment_approved": False, "holdout_opened": False,
            "indicator_count": len(args.indicators),
            "indicator_variant_count": len(variants),
            "indicator_value_range": {
                "min": args.value_min, "max": args.value_max, "step": args.value_step,
            },
            "signal_primitive_count": len(variants), "research_trial_count": research_trials,
            "canonical_finalist_count": len(finalists),
            "orientation_sessions": len(orientation),
            "purge_sessions": purge_sessions,
            "nested_validation_sessions": len(discovery),
            "reason": "no finalist had positive robust score, median/worst Sharpe, and median CAGR",
        }, indent=2))
        print(
            f"{len(args.indicators)} indicators / {len(variants)} values -> "
            f"{research_trials} unrestricted GPU combinations"
        )
        print("DISCOVERY_REJECT: no stable positive finalist; holdout_opened=false")
        print(f"artifacts: {out}")
        _progress(100, "Complete: discovery rejected; holdout remained sealed")
        return 2
    winner = eligible.iloc[0].to_dict()
    definition = catalog[winner["signal_key"]]
    candidate = _replay_candidate(winner, definition)
    decision = int(candidate["decision_bar"])
    evaluation = slice_decision_dataset(datasets[decision], cut, sessions)
    _progress(92, "Evaluating the discovery winner on the sealed holdout")
    score = score_cache[winner["signal_key"]][cut:]
    cost_rows, contribution_rows, ledger = [], [], pd.DataFrame()
    for cost in (1.0, 2.5, 5.0):
        replay = evaluate_v2_candidate(
            evaluation, candidate, cost_per_side_bps=cost,
            forecast_cache={"signal_consensus": score}, device=device,
            return_ledger=bool(np.isclose(cost, args.cost_bps)),
        )
        result, contribution = replay[:2]
        if len(replay) == 3: ledger = replay[2]
        cost_rows.append(result)
        contribution_rows.extend({
            "cost_scenario_bps_per_side": cost, "ticker": ticker,
            "net_additive_contribution": value,
        } for ticker, value in zip(tickers, contribution, strict=True))
    costs = pd.DataFrame(cost_rows)
    contributions = pd.DataFrame(contribution_rows)
    # Evaluator echoes candidate, so add the scenario column explicitly and resolve rows.
    costs.insert(0, "cost_scenario_bps_per_side", (1.0, 2.5, 5.0))
    base = costs[np.isclose(costs.cost_scenario_bps_per_side, args.cost_bps)].iloc[0]
    harsh = costs[np.isclose(costs.cost_scenario_bps_per_side, 5.0)].iloc[0]
    winner_rows = evaluations.copy()
    for field in PARAM_FIELDS: winner_rows = winner_rows[winner_rows[field] == winner[field]]
    pbo = combinatorial_probability_overfit(evaluations, PARAM_FIELDS)
    all_screen = pd.concat(screen_frames, ignore_index=True)
    dsr = deflated_sharpe(all_screen.screen_sharpe.fillna(0).tolist(), n_obs=cut - warmup)
    dsr["scope"] = "research_screen"
    base_contrib = contributions[np.isclose(contributions.cost_scenario_bps_per_side, args.cost_bps)].net_additive_contribution
    concentration = float(base_contrib.abs().max() / base_contrib.abs().sum()) if base_contrib.abs().sum() else float("inf")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"signal_strategy_finder_{stamp}"
    out.mkdir(parents=True)
    all_screen.to_csv(out / "gpu_research_screen.csv", index=False)
    ranked.to_csv(out / "canonical_discovery_rankings.csv", index=False)
    ranked.to_csv(out / "discovery_rankings.csv", index=False)
    evaluations.to_csv(out / "discovery_windows.csv", index=False)
    costs.to_csv(out / "evaluation_cost_stress.csv", index=False)
    contributions.to_csv(out / "evaluation_contributions.csv", index=False)
    ledger.to_csv(out / "evaluation_ledger.csv", index=False)
    summary = {
        "strategy_version": 4, "verdict": "REJECT", "deployment_approved": False,
        "sleeve": args.sleeve, "session_model": sleeve.session_model, "timeframe": "1h",
        "bar_data_root": str(args.intraday_root), "research_only_free_data": True,
        "tickers": list(tickers), "benchmark": sleeve.benchmark,
        "indicator_count": len(args.indicators),
        "indicator_variant_count": len(variants),
        "indicator_value_range": {
            "min": args.value_min, "max": args.value_max, "step": args.value_step,
        },
        "signal_primitive_count": len(variants), "research_trial_count": research_trials,
        "candidate_count": len(finalists), "common_session_count": sessions,
        "warmup_sessions_unscored": warmup, "orientation_sessions": len(orientation),
        "purge_sessions": purge_sessions, "discovery_sessions": len(discovery),
        "nested_validation": True, "holdout_opened": True,
        "evaluation_sessions": sessions - cut, "portfolio_window_count": 4,
        "simulation_device": f"hybrid-{device}", "sip_quote_ready": False,
        "point_in_time_membership": False, "event_calendar_ready": False,
        "historical_interval_unexposed": False,
        "winner": {**winner, **_safe(candidate), "description": winner["signal_key"]},
        "discovery_positive_windows": int((winner_rows.total_return if "total_return" in winner_rows else winner_rows.sharpe > 0).sum()),
        "evaluation_base_cost": base.to_dict(), "evaluation_harsh_cost": harsh.to_dict(),
        "largest_absolute_contribution_share": concentration,
        "probability_backtest_overfit": pbo, "deflated_sharpe": dsr,
    }
    summary["historical_deployment_qualification"] = historical_qualification(summary, costs)
    fingerprint = candidate_fingerprint(candidate)
    summary["candidate_fingerprint"] = fingerprint
    (out / "summary.json").write_text(json.dumps(_safe(summary), indent=2))
    (out / "frozen_candidate.json").write_text(json.dumps({
        "candidate": _safe(candidate), "sha256": fingerprint,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "data_end": str(evaluation.session_dates[-1]), "deployment_approved": False,
        "strategy_version": 4,
    }, indent=2))
    print(
        f"{len(args.indicators)} indicators / {len(variants)} values -> "
        f"{research_trials} GPU combinations -> {len(finalists)} canonical finalists"
    )
    print(f"winner: {winner['signal_key']}")
    print(f"evaluation: {base.total_return:.2%}, Sharpe {base.sharpe:.3f}, events {int(base.portfolio_events)}")
    print(f"qualification: {sum(summary['historical_deployment_qualification']['gates'].values())}/14; REJECT")
    print(f"artifacts: {out}")
    _progress(100, "Complete: holdout evaluation and deployment gates written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
