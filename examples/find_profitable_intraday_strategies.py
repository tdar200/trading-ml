"""GPU-first, cost-gated research over pre-registered intraday strategy families."""
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
from analysis.deployment_gate import candidate_fingerprint, historical_qualification
from analysis.deflated_sharpe import deflated_sharpe
from analysis.dynamic_profitability import prepare_dynamic_decision_dataset
from analysis.optimizer import SearchSettings, rank_evaluations
from analysis.profitability import (
    PARAM_FIELDS, ProfitabilityGrammar, align_decision_datasets,
    causal_momentum_forecast, confidence_adjusted_forecast, describe_profitability,
    evaluate_profitability_candidate,
    generate_profitability_candidates, pooled_ridge_predictions,
    pooled_elastic_net_predictions,
    slice_decision_dataset,
)
from analysis.vectorized_screen import resolve_device
from data.event_calendar import load_event_calendar
from data.multitf import get_bars
from data.quotes import load_quotes
from data.tickers import BENCHMARK, SCALPING_STOCKS
from data.universe_membership import load_membership

EXPOSED_THROUGH = date(2026, 9, 12)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
    return value


def _params(row):
    return {field: row[field] for field in PARAM_FIELDS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=",".join(SCALPING_STOCKS))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--membership", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--min-assets", type=int, default=15)
    parser.add_argument(
        "--preregistered-forward-start",
        help="Lock selection before this future date; must be after 2026-09-12",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    used_device = resolve_device(args.device)  # Explicit CUDA: unavailable hardware fails here.
    tickers = tuple(dict.fromkeys(x.strip().upper() for x in args.tickers.split(",") if x.strip()))
    if args.smoke:
        tickers = tickers[:6]
    if len(tickers) < 4:
        raise SystemExit("at least four stocks are required")
    membership = load_membership(args.membership) if args.membership else None
    event_calendar = load_event_calendar(args.events) if args.events else None
    bars = {ticker: get_bars(ticker, "5m") for ticker in tickers}
    benchmark = get_bars(BENCHMARK, "5m")
    if missing := [ticker for ticker, frame in {**bars, BENCHMARK: benchmark}.items() if frame.empty]:
        raise SystemExit(f"missing 5m bars for: {', '.join(missing)}")
    quote_frames = {ticker: load_quotes(ticker) for ticker in tickers}
    quote_input_ready = all(not frame.empty for frame in quote_frames.values())
    decision_bars = (5, 23) if args.smoke else (5, 23, 47, 71)
    min_assets = 4 if args.smoke else args.min_assets
    datasets = align_decision_datasets({
        decision: prepare_dynamic_decision_dataset(
            bars, benchmark, decision, min_assets=min_assets,
            quotes_by_ticker=quote_frames if quote_input_ready else None,
            membership=membership,
            event_calendar=event_calendar,
        )
        for decision in decision_bars
    })
    quote_ready = quote_input_ready and all(item.spread_bps is not None for item in datasets.values())
    session_count = len(next(iter(datasets.values())).session_dates)
    warmup_sessions = 60
    interval_unexposed = False
    if args.preregistered_forward_start:
        forward_start = pd.Timestamp(args.preregistered_forward_start).date()
        if forward_start <= EXPOSED_THROUGH:
            raise SystemExit(f"forward start must be after exposed cutoff {EXPOSED_THROUGH}")
        cut = int(np.searchsorted(next(iter(datasets.values())).session_dates, forward_start))
        interval_unexposed = True
    else:
        scored_sessions = session_count - warmup_sessions
        cut = warmup_sessions + int(scored_sessions * 0.8)
    if session_count - cut < 10 or cut - warmup_sessions < args.windows * 10:
        raise SystemExit("not enough common sessions for discovery and evaluation")
    chunks = np.array_split(np.arange(warmup_sessions, cut), args.windows)

    grammar = ProfitabilityGrammar(
        decision_bars=decision_bars,
        max_positions_per_side=(2,) if args.smoke else (2, 3, 5),
        min_edge_bps=(0.0, 5.0) if args.smoke else (0.0, 5.0, 10.0),
        edge_cost_multiple=(2.0,) if args.smoke else (2.0, 3.0),
        confidence_z=(0.0,) if args.smoke else (0.0, 0.5, 1.0),
        regimes=("all",) if args.smoke else ("all", "high_information", "avoid_panic_rebound"),
        seasonality_days=(20,) if args.smoke else (20, 40),
    )
    candidates = generate_profitability_candidates(
        quote_ready=quote_ready, event_ready=event_calendar is not None, grammar=grammar,
    )
    ridge = {
        decision: {
            days: pooled_ridge_predictions(dataset, days, device=used_device)
            for days in grammar.seasonality_days
        }
        for decision, dataset in datasets.items()
    }
    elastic = {
        decision: {
            days: pooled_elastic_net_predictions(dataset, days, device=used_device)
            for days in grammar.seasonality_days
        }
        for decision, dataset in datasets.items()
    }
    forecasts = {
        decision: {
            ("first", 0): causal_momentum_forecast(
                dataset.sector_residual if dataset.sector_residual is not None else dataset.first_residual,
                dataset.forward_return, device=used_device,
            ),
            **{
                ("seasonal", days): causal_momentum_forecast(
                    (dataset.sector_residual if dataset.sector_residual is not None else dataset.first_residual)
                    + dataset.seasonality[days],
                    dataset.forward_return, device=used_device,
                )
                for days in grammar.seasonality_days
            },
        }
        for decision, dataset in datasets.items()
    }
    confidence_forecasts = {}
    for decision, dataset in datasets.items():
        raw = {
            ("first", 0): forecasts[decision][("first", 0)],
            **{("seasonal", days): forecasts[decision][("seasonal", days)] for days in grammar.seasonality_days},
            **{("ridge", days): ridge[decision][days] for days in grammar.seasonality_days},
            **{("elastic", days): elastic[decision][days] for days in grammar.seasonality_days},
        }
        confidence_forecasts[decision] = {
            (family, days, z): confidence_adjusted_forecast(values, dataset.forward_return, z)
            for (family, days), values in raw.items() for z in grammar.confidence_z
        }
    evaluations = []
    for window_index, indices in enumerate(chunks):
        start, stop = int(indices[0]), int(indices[-1]) + 1
        for candidate in candidates:
            decision = int(candidate["decision_bar"])
            cache = {days: values[start:stop] for days, values in ridge[decision].items()}
            elastic_cache = {days: values[start:stop] for days, values in elastic[decision].items()}
            forecast_cache = {key: values[start:stop] for key, values in forecasts[decision].items()}
            confidence_cache = {
                key: values[start:stop] for key, values in confidence_forecasts[decision].items()
            }
            result, _ = evaluate_profitability_candidate(
                slice_decision_dataset(datasets[decision], start, stop), candidate,
                cost_per_side_bps=args.cost_bps, ridge_cache=cache,
                forecast_cache=forecast_cache, device=used_device,
                confidence_cache=confidence_cache,
                elastic_cache=elastic_cache,
            )
            evaluations.append({"window_idx": window_index, **result})
    evaluation_frame = pd.DataFrame(evaluations)
    ranked = rank_evaluations(
        evaluation_frame, PARAM_FIELDS,
        SearchSettings(
            algorithm="grid", min_trades_per_window=max(10, len(tickers)),
            stability_penalty=0.5, drawdown_penalty=0.2,
        ),
    )
    eligible = ranked[ranked["eligible"]]
    if eligible.empty:
        raise SystemExit("no candidate met the discovery activity gate")
    winner = eligible.iloc[0]
    winner_params = _params(winner)
    decision = int(winner["decision_bar"])
    evaluation_dataset = slice_decision_dataset(datasets[decision], cut, session_count)
    evaluation_cache = {days: values[cut:] for days, values in ridge[decision].items()}
    evaluation_elastic = {days: values[cut:] for days, values in elastic[decision].items()}
    evaluation_forecasts = {key: values[cut:] for key, values in forecasts[decision].items()}
    evaluation_confidence = {key: values[cut:] for key, values in confidence_forecasts[decision].items()}
    cost_results, contribution_rows = [], []
    for cost in sorted(set((args.cost_bps, 2.5, 5.0))):
        result, contribution = evaluate_profitability_candidate(
            evaluation_dataset, winner_params, cost_per_side_bps=cost,
            ridge_cache=evaluation_cache, forecast_cache=evaluation_forecasts, device=used_device,
            confidence_cache=evaluation_confidence,
            elastic_cache=evaluation_elastic,
        )
        cost_results.append({"cost_scenario_bps_per_side": cost, **result})
        contribution_rows.extend({
            "cost_scenario_bps_per_side": cost, "ticker": ticker,
            "net_additive_contribution": value,
        } for ticker, value in zip(tickers, contribution, strict=True))
    costs = pd.DataFrame(cost_results)
    contributions = pd.DataFrame(contribution_rows)
    base = costs.loc[costs.cost_scenario_bps_per_side == args.cost_bps].iloc[0]
    harsh = costs.loc[costs.cost_scenario_bps_per_side == 5.0].iloc[0]
    base_contrib = contributions.loc[
        contributions.cost_scenario_bps_per_side == args.cost_bps, "net_additive_contribution"
    ]
    concentration = float(base_contrib.abs().max() / base_contrib.abs().sum()) if base_contrib.abs().sum() else float("inf")
    winner_windows = evaluation_frame
    for field in PARAM_FIELDS:
        winner_windows = winner_windows[winner_windows[field] == winner[field]]
    positive_windows = int((winner_windows.total_return > 0).sum())
    pbo = combinatorial_probability_overfit(evaluation_frame, PARAM_FIELDS)
    dsr = deflated_sharpe(
        ranked.median_sharpe.fillna(0.0).tolist(), n_obs=cut - warmup_sessions,
    )
    research_pass = bool(
        positive_windows >= int(np.ceil(0.75 * args.windows))
        and base.total_return > 0 and base.sharpe > 1.0 and base.max_drawdown > -0.10
        and base.portfolio_events >= 20 and harsh.total_return > 0
        and concentration <= 0.25 and base.mean_abs_beta_exposure <= 0.02
        and pbo["probability_overfit"] <= 0.20 and dsr["p_value"] >= 0.95
    )
    # Historical evaluation dates have already been inspected in earlier research runs.
    # A new forward period and quote-aware replay remain mandatory for deployment.
    verdict = "RESEARCH_PASS_FORWARD_REQUIRED" if research_pass else "REJECT"
    ranked.insert(1, "description", [describe_profitability(row) for _, row in ranked.iterrows()])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"profitability_finder_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out / "discovery_rankings.csv", index=False)
    evaluation_frame.to_csv(out / "discovery_windows.csv", index=False)
    costs.to_csv(out / "evaluation_cost_stress.csv", index=False)
    contributions.to_csv(out / "evaluation_contributions.csv", index=False)
    summary = {
        "verdict": verdict, "research_gates_passed": research_pass,
        "deployment_approved": False,
        "historical_interval_unexposed": interval_unexposed,
        "preregistered_forward_start": args.preregistered_forward_start,
        "deployment_blockers": [
            "historical evaluation interval was exposed by earlier project runs"
            if not interval_unexposed else "pre-registered forward interval supplied",
            "new forward paper-trading period not completed",
            "SIP/NBBO quote-aware fill replay not completed" if not quote_ready else "live fill calibration not completed",
            "point-in-time universe membership not supplied" if membership is None else "membership supplied",
            "earnings/macro event calendar not supplied" if event_calendar is None else "event calendar supplied",
        ],
        "tickers": list(tickers), "benchmark": BENCHMARK,
        "source_bar_count": int(sum(len(frame) for frame in bars.values())),
        "common_session_count": session_count, "minimum_assets_per_session": min_assets,
        "warmup_sessions_unscored": warmup_sessions,
        "discovery_sessions": cut - warmup_sessions,
        "evaluation_sessions": session_count - cut,
        "candidate_count": len(candidates), "simulation_device": f"hybrid-{used_device}",
        "portfolio_window_count": args.windows,
        "sip_quote_ready": quote_ready, "point_in_time_membership": membership is not None,
        "event_calendar_ready": event_calendar is not None,
        "winner": {**winner.to_dict(), "description": describe_profitability(winner)},
        "discovery_positive_windows": positive_windows,
        "evaluation_base_cost": base.to_dict(), "evaluation_harsh_cost": harsh.to_dict(),
        "largest_absolute_contribution_share": concentration,
        "probability_backtest_overfit": pbo, "deflated_sharpe": dsr,
    }
    fingerprint = candidate_fingerprint(winner_params)
    summary["candidate_fingerprint"] = fingerprint
    summary["historical_deployment_qualification"] = historical_qualification(summary, costs)
    (out / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    frozen = {
        "candidate": _json_safe(winner_params), "sha256": fingerprint,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "data_end": str(evaluation_dataset.session_dates[-1]),
        "mutable_parameters": False, "deployment_approved": False,
    }
    (out / "frozen_candidate.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"profitability finder: {len(tickers)} stocks, {len(candidates)} candidates, {args.windows} windows")
    print(f"device: hybrid-{used_device}; common sessions: {session_count}; SIP/NBBO: {quote_ready}")
    print(ranked[["rank", *PARAM_FIELDS, "robust_score", "median_sharpe", "worst_sharpe", "total_trades"]].head(15).to_string(index=False))
    print(f"winner: {describe_profitability(winner)}")
    print(f"evaluation @ {args.cost_bps:g} bps/side: return {base.total_return:.2%}, Sharpe {base.sharpe:.3f}, max DD {base.max_drawdown:.2%}, events {int(base.portfolio_events)}")
    print(f"evaluation @ 5 bps/side: {harsh.total_return:.2%}; PBO {pbo['probability_overfit']:.1%}; DSR confidence {dsr['p_value']:.1%}")
    print(f"verdict: {verdict}; artifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
