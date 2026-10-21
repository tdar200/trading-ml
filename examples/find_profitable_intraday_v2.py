"""Run the preregistered, structure-aware profitability V3 search on CUDA."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.cross_sectional import combinatorial_probability_overfit
from analysis.deflated_sharpe import deflated_sharpe
from analysis.deployment_gate import candidate_fingerprint, file_sha256, historical_qualification
from analysis.dynamic_profitability import prepare_dynamic_decision_dataset
from analysis.optimizer import SearchSettings, rank_evaluations
from analysis.profitability import align_decision_datasets, slice_decision_dataset
from analysis.profitability_v2 import (
    V2_PARAM_FIELDS, ProfitabilityV2Grammar, build_v2_forecasts, describe_v2,
    evaluate_v2_candidate, generate_v2_candidates,
)
from analysis.sleeves import SLEEVES, require_equity_replay_sleeve
from analysis.vectorized_screen import resolve_device
from data.event_calendar import load_event_calendar
from data.multitf import get_bars
from data.quotes import load_quotes
from data.universe_membership import load_membership

EXPOSED_THROUGH = date(2026, 9, 12)


def _json_safe(value):
    if isinstance(value, dict): return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray): return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)): return value.isoformat()
    return value


def _params(row):
    return {field: row[field] for field in V2_PARAM_FIELDS}


def _data_manifest(
    bars: dict[str, pd.DataFrame], *, membership: Path | None, events: Path | None,
    quote_ready: bool,
) -> dict:
    coverage = {
        ticker: {
            "rows": int(len(frame)),
            "start": frame.ts.min().isoformat() if len(frame) else None,
            "end": frame.ts.max().isoformat() if len(frame) else None,
        }
        for ticker, frame in bars.items()
    }
    return {
        "bars": coverage,
        "sip_nbbo_complete": quote_ready,
        "point_in_time_membership": (
            {"path": str(membership), "sha256": file_sha256(membership)} if membership else None
        ),
        "event_calendar": (
            {"path": str(events), "sha256": file_sha256(events)} if events else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sleeve", choices=tuple(SLEEVES), default="individual_stock")
    parser.add_argument("--tickers", help="Optional comma-separated override within one sleeve")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--cost-bps", type=float, default=2.5)
    parser.add_argument("--timeframe", choices=("5m", "1h"), default="5m")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--membership", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--min-assets", type=int, default=15)
    parser.add_argument(
        "--intraday-root", type=Path,
        help="Explicit bar store, e.g. data/free_intraday for keyless research-only Yahoo data.",
    )
    parser.add_argument("--preregistered-forward-start")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    try:
        sleeve = require_equity_replay_sleeve(args.sleeve)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    used_device = resolve_device(args.device)
    tickers = sleeve.symbols
    if args.tickers:
        tickers = tuple(dict.fromkeys(x.strip().upper() for x in args.tickers.split(",") if x.strip()))
    if args.smoke:
        tickers = tickers[:6]
    if len(tickers) < 4:
        raise SystemExit("at least four instruments are required")
    membership = load_membership(args.membership) if args.membership else None
    events = load_event_calendar(args.events) if args.events else None
    bar_kwargs = {"intraday_root": args.intraday_root} if args.intraday_root else {}
    bars = {ticker: get_bars(ticker, args.timeframe, **bar_kwargs) for ticker in tickers}
    benchmark = get_bars(sleeve.benchmark, args.timeframe, **bar_kwargs)
    if missing := [ticker for ticker, frame in {**bars, sleeve.benchmark: benchmark}.items() if frame.empty]:
        raise SystemExit(f"missing {args.timeframe} bars for {args.sleeve}: {', '.join(missing)}")
    quotes = {ticker: load_quotes(ticker) for ticker in tickers}
    quote_input_ready = all(not frame.empty for frame in quotes.values())
    bar_minutes = 60 if args.timeframe == "1h" else 5
    decisions = ((1,) if args.smoke else (1, 3)) if args.timeframe == "1h" else ((23,) if args.smoke else (23, 47))
    min_assets = 4 if args.smoke else min(args.min_assets, len(tickers))
    try:
        datasets = align_decision_datasets({
            decision: prepare_dynamic_decision_dataset(
                bars, benchmark, decision, min_assets=min_assets,
                bar_minutes=bar_minutes,
                quotes_by_ticker=quotes if quote_input_ready else None,
                membership=membership, event_calendar=events,
            ) for decision in decisions
        })
    except ValueError as exc:
        raise SystemExit(f"{args.sleeve} data is not research-ready: {exc}") from exc
    quote_ready = quote_input_ready and all(item.spread_bps is not None for item in datasets.values())
    grammar = ProfitabilityV2Grammar(
        decision_bars=decisions,
        max_positions_per_side=(2, 3) if args.smoke else (3, 5),
        minimum_relative_volume=(0.75, 1.0, 1.25, 1.5) if args.smoke else (0.75, 1.0),
        timeframe=args.timeframe,
    )
    candidates = generate_v2_candidates(
        quote_ready=quote_ready, event_ready=events is not None, grammar=grammar,
    )
    sessions = len(next(iter(datasets.values())).session_dates)
    warmup = 60
    interval_unexposed = False
    if args.preregistered_forward_start:
        start = pd.Timestamp(args.preregistered_forward_start).date()
        if start <= EXPOSED_THROUGH:
            raise SystemExit(f"forward start must be after {EXPOSED_THROUGH}")
        cut = int(np.searchsorted(next(iter(datasets.values())).session_dates, start))
        interval_unexposed = True
    else:
        cut = sessions - 252 if sessions - warmup >= 292 else warmup + int((sessions - warmup) * 0.8)
    if sessions - cut < 10 or cut - warmup < args.windows * 10:
        raise SystemExit("not enough sessions for discovery and evaluation")
    windows = np.array_split(np.arange(warmup, cut), args.windows)
    forecasts = {decision: build_v2_forecasts(dataset, device=used_device) for decision, dataset in datasets.items()}
    evaluations = []
    for window_index, indices in enumerate(windows):
        start, stop = int(indices[0]), int(indices[-1]) + 1
        for candidate in candidates:
            decision = int(candidate["decision_bar"])
            cache = {name: values[start:stop] for name, values in forecasts[decision].items()}
            result, _ = evaluate_v2_candidate(
                slice_decision_dataset(datasets[decision], start, stop), candidate,
                cost_per_side_bps=args.cost_bps, forecast_cache=cache, device=used_device,
            )
            evaluations.append({"window_idx": window_index, **result})
    frame = pd.DataFrame(evaluations)
    ranked = rank_evaluations(
        frame, V2_PARAM_FIELDS,
        SearchSettings(
            algorithm="grid", min_trades_per_window=8,
            stability_penalty=0.75, drawdown_penalty=0.3,
        ),
    )
    eligible = ranked[ranked.eligible]
    if eligible.empty:
        raise SystemExit("no V3 candidate met the discovery activity gate")
    winner = eligible.iloc[0]
    params = _params(winner)
    decision = int(params["decision_bar"])
    evaluation = slice_decision_dataset(datasets[decision], cut, sessions)
    cache = {name: values[cut:] for name, values in forecasts[decision].items()}
    costs, contribution_rows = [], []
    evaluation_ledger = pd.DataFrame()
    for cost in sorted({1.0, args.cost_bps, 2.5, 5.0}):
        evaluated = evaluate_v2_candidate(
            evaluation, params, cost_per_side_bps=cost,
            forecast_cache=cache, device=used_device,
            return_ledger=bool(np.isclose(cost, args.cost_bps)),
        )
        result, contribution = evaluated[:2]
        if len(evaluated) == 3:
            evaluation_ledger = evaluated[2]
        costs.append({"cost_scenario_bps_per_side": cost, **result})
        contribution_rows.extend({
            "cost_scenario_bps_per_side": cost, "ticker": ticker,
            "net_additive_contribution": value,
        } for ticker, value in zip(tickers, contribution, strict=True))
    cost_frame = pd.DataFrame(costs)
    contributions = pd.DataFrame(contribution_rows)
    base = cost_frame.loc[np.isclose(cost_frame.cost_scenario_bps_per_side, args.cost_bps)].iloc[0]
    harsh = cost_frame.loc[np.isclose(cost_frame.cost_scenario_bps_per_side, 5.0)].iloc[0]
    winner_windows = frame.copy()
    for field in V2_PARAM_FIELDS:
        winner_windows = winner_windows[winner_windows[field] == winner[field]]
    positive_windows = int((winner_windows.total_return > 0).sum())
    pbo = combinatorial_probability_overfit(frame, V2_PARAM_FIELDS)
    # Every attempted configuration counts toward multiple-testing deflation,
    # including inactive/invalid trials (conservatively represented as zero).
    dsr = deflated_sharpe(ranked.median_sharpe.fillna(0.0).tolist(), n_obs=cut - warmup)
    dsr["scope"] = "discovery"
    base_contrib = contributions.loc[
        np.isclose(contributions.cost_scenario_bps_per_side, args.cost_bps),
        "net_additive_contribution",
    ]
    concentration = float(base_contrib.abs().max() / base_contrib.abs().sum()) if base_contrib.abs().sum() else float("inf")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"profitability_v3_{args.sleeve}_{args.timeframe}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    ranked.insert(1, "description", [describe_v2(row) for _, row in ranked.iterrows()])
    ranked.to_csv(out / "discovery_rankings.csv", index=False)
    frame.to_csv(out / "discovery_windows.csv", index=False)
    cost_frame.to_csv(out / "evaluation_cost_stress.csv", index=False)
    contributions.to_csv(out / "evaluation_contributions.csv", index=False)
    evaluation_ledger.to_csv(out / "evaluation_ledger.csv", index=False)
    manifest = _data_manifest(
        bars, membership=args.membership, events=args.events, quote_ready=quote_ready,
    )
    (out / "data_readiness.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "strategy_version": 3, "verdict": "REJECT",
        "deployment_approved": False, "historical_interval_unexposed": interval_unexposed,
        "preregistered_forward_start": args.preregistered_forward_start,
        "sleeve": args.sleeve, "session_model": sleeve.session_model,
        "timeframe": args.timeframe,
        "bar_data_root": str(args.intraday_root) if args.intraday_root else "data/intraday",
        "research_only_free_data": bool(args.intraday_root and args.intraday_root.name == "free_intraday"),
        "tickers": list(tickers), "benchmark": sleeve.benchmark,
        "candidate_count": len(candidates), "candidate_budget": "20-50 preregistered trials",
        "simulation_device": f"hybrid-{used_device}", "common_session_count": sessions,
        "warmup_sessions_unscored": warmup, "discovery_sessions": cut - warmup,
        "evaluation_sessions": sessions - cut, "portfolio_window_count": args.windows,
        "sip_quote_ready": quote_ready, "point_in_time_membership": membership is not None,
        "event_calendar_ready": events is not None,
        "winner": {**winner.to_dict(), "description": describe_v2(winner)},
        "discovery_positive_windows": positive_windows,
        "evaluation_base_cost": base.to_dict(), "evaluation_harsh_cost": harsh.to_dict(),
        "largest_absolute_contribution_share": concentration,
        "probability_backtest_overfit": pbo, "deflated_sharpe": dsr,
    }
    fingerprint = candidate_fingerprint(params)
    summary["candidate_fingerprint"] = fingerprint
    summary["historical_deployment_qualification"] = historical_qualification(summary, cost_frame)
    passed = summary["historical_deployment_qualification"]["qualified"]
    summary["verdict"] = "HISTORICAL_PASS_PAPER_REQUIRED" if passed else "REJECT"
    (out / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    frozen = {
        "candidate": _json_safe(params), "sha256": fingerprint,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "data_end": str(evaluation.session_dates[-1]), "mutable_parameters": False,
        "deployment_approved": False, "strategy_version": 3, "sleeve": args.sleeve,
    }
    (out / "frozen_candidate.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    gates = summary["historical_deployment_qualification"]
    print(f"V3: {args.sleeve}, {len(tickers)} instruments, {len(candidates)} preregistered candidates")
    print(f"winner: {describe_v2(winner)}")
    print(f"evaluation @ {args.cost_bps:g} bps/side: {base.total_return:.2%}, Sharpe {base.sharpe:.3f}, events {int(base.portfolio_events)}")
    print(f"5 bps/side: {harsh.total_return:.2%}; PBO {pbo['probability_overfit']:.1%}; discovery DSR {dsr['p_value']:.1%}")
    print(f"historical gates: {sum(gates['gates'].values())}/{len(gates['gates'])}; {summary['verdict']}")
    print(f"artifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
