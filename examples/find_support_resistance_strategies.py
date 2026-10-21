"""Search mechanical support/resistance confluence portfolios on CUDA."""
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
from analysis.optimizer import SearchSettings, rank_evaluations
from analysis.support_resistance import (
    PARAM_FIELDS,
    SupportResistanceGrammar,
    describe_support_resistance,
    generate_support_resistance_candidates,
    prepare_support_resistance_panel,
    simulate_support_resistance_candidates,
    split_support_resistance_panel,
)
from data.multitf import get_bars
from data.tickers import BENCHMARK, SCALPING_STOCKS


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=",".join(SCALPING_STOCKS))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tickers = tuple(dict.fromkeys(
        ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()
    ))
    if args.smoke:
        tickers = tickers[:6]
    if len(tickers) < 4:
        raise SystemExit("Support/resistance finder requires at least four stocks.")
    bars = {ticker: get_bars(ticker, "5m") for ticker in tickers}
    benchmark = get_bars(BENCHMARK, "5m")
    if missing := [ticker for ticker, frame in {**bars, BENCHMARK: benchmark}.items() if frame.empty]:
        raise SystemExit(f"No 5m data for {', '.join(missing)}")

    panel = prepare_support_resistance_panel(bars, benchmark)
    window_count = 4 if args.smoke else args.windows
    windows, evaluation_period = split_support_resistance_panel(
        panel, discovery_fraction=0.8, n_windows=window_count,
    )
    grammar = (
        SupportResistanceGrammar(
            level_types=("opening_range", "vwap_band"),
            behaviors=("rejection", "failed_breakout"),
            signal_families=("momentum", "reversal"),
            decision_bars=(24,), lookback_bars=(12,), hold_bars=(12,),
            top_n=(1,), zone_atr=(0.10,), stop_atr=(0.50,),
            reward_risk=(2.0,), min_relative_volume=(1.0,),
            max_market_move_bps=(50.0,),
        ) if args.smoke else SupportResistanceGrammar()
    )
    candidates = generate_support_resistance_candidates(grammar)
    discovery_rows = []
    for window_index, window in enumerate(windows):
        result, _ = simulate_support_resistance_candidates(
            window, candidates, cost_per_side_bps=args.cost_bps, device=args.device,
        )
        result.insert(0, "window_idx", window_index)
        discovery_rows.append(result)
    evaluations = pd.concat(discovery_rows, ignore_index=True)
    ranked = rank_evaluations(
        evaluations, PARAM_FIELDS,
        SearchSettings(
            algorithm="grid", min_trades_per_window=4 if args.smoke else 10,
            stability_penalty=0.5, drawdown_penalty=0.2,
        ),
    )
    eligible = ranked[ranked["eligible"]]
    if eligible.empty:
        raise SystemExit("No support/resistance candidate was eligible in every window.")
    winner = eligible.iloc[0]
    winner_params = {field: winner[field] for field in PARAM_FIELDS}

    cost_rows, contribution_rows = [], []
    for cost in sorted(set([args.cost_bps, 2.5, 5.0])):
        result, contribution = simulate_support_resistance_candidates(
            evaluation_period, [winner_params], cost_per_side_bps=cost,
            device=args.device, contribution_candidate=winner_params,
        )
        result.insert(0, "cost_scenario_bps_per_side", cost)
        contribution.insert(0, "cost_scenario_bps_per_side", cost)
        cost_rows.append(result)
        contribution_rows.append(contribution)
    cost_stress = pd.concat(cost_rows, ignore_index=True)
    contributions = pd.concat(contribution_rows, ignore_index=True)
    base = cost_stress[cost_stress["cost_scenario_bps_per_side"] == args.cost_bps].iloc[0]
    harsh = cost_stress[cost_stress["cost_scenario_bps_per_side"] == 5.0].iloc[0]
    base_contribution = contributions[
        contributions["cost_scenario_bps_per_side"] == args.cost_bps
    ]["net_additive_contribution"]
    concentration = float(
        base_contribution.abs().max() / base_contribution.abs().sum()
    ) if base_contribution.abs().sum() else float("inf")
    winner_windows = evaluations.copy()
    for field in PARAM_FIELDS:
        winner_windows = winner_windows[winner_windows[field] == winner[field]]
    positive_windows = int((winner_windows["total_return"] > 0).sum())
    try:
        pbo = combinatorial_probability_overfit(evaluations, PARAM_FIELDS)
    except ValueError as exc:
        pbo = {"probability_overfit": None, "reason": str(exc)}
    discovery_sessions = sum(len(np.unique(window.base.sessions)) for window in windows)
    dsr = deflated_sharpe(
        eligible["median_sharpe"].dropna().tolist(), n_obs=discovery_sessions,
    )
    statistical_pass = bool(
        positive_windows >= int(np.ceil(0.75 * window_count))
        and base["total_return"] > 0 and base["sharpe"] > 1.0
        and base["max_drawdown"] > -0.10 and base["n_trades"] >= 100
        and harsh["total_return"] > 0 and concentration <= 0.25
        and base["mean_abs_beta_exposure"] <= 0.02
        and pbo.get("probability_overfit") is not None
        and pbo["probability_overfit"] <= 0.20 and dsr["p_value"] >= 0.95
    )
    # This date range was exposed by earlier finder runs. It cannot become a
    # deployment PASS regardless of its return; new future data is mandatory.
    deployment_verdict = "REJECT"
    research_verdict = "PASS" if statistical_pass else "REJECT"
    ranked.insert(1, "description", [
        describe_support_resistance(row) for _, row in ranked.iterrows()
    ])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"support_resistance_finder_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out / "discovery_rankings.csv", index=False)
    evaluations.to_csv(out / "discovery_portfolio_windows.csv", index=False)
    winner_windows.to_csv(out / "winner_discovery_windows.csv", index=False)
    cost_stress.to_csv(out / "winner_evaluation_cost_stress.csv", index=False)
    contributions.to_csv(out / "winner_evaluation_contributions.csv", index=False)
    summary = {
        "deployment_verdict": deployment_verdict,
        "research_verdict": research_verdict,
        "evaluation_status": "reused_post_selection_period_not_untouched",
        "required_next_gate": "new_future_data_then_quote_aware_replay_and_paper_trading",
        "tickers": list(tickers), "benchmark": BENCHMARK,
        "instrument_count": len(tickers), "candidate_count": len(candidates),
        "aligned_bar_count": len(panel.base.timestamps),
        "session_count": int(len(np.unique(panel.base.sessions))),
        "portfolio_window_count": window_count,
        "winner": {**winner.to_dict(), "description": describe_support_resistance(winner)},
        "discovery_positive_windows": positive_windows,
        "evaluation_base_cost": base.to_dict(),
        "evaluation_harsh_cost": harsh.to_dict(),
        "largest_absolute_contribution_share": concentration,
        "probability_backtest_overfit": pbo, "deflated_sharpe": dsr,
        "premarket_limitation": "IEX cache begins around 08:00 ET and is not a complete 04:00-09:30 premarket feed",
    }
    (out / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8",
    )
    print(
        f"support/resistance finder: {len(tickers)} stocks x {len(candidates)} "
        f"candidates x {window_count} discovery windows + reused evaluation period"
    )
    print(ranked[[
        "rank", *PARAM_FIELDS, "robust_score", "median_sharpe", "worst_sharpe", "total_trades",
    ]].head(15).to_string(index=False))
    print(f"winner: {describe_support_resistance(winner)}")
    print(
        f"evaluation @ {args.cost_bps:g} bps/side: {base['total_return']:.2%}, "
        f"Sharpe {base['sharpe']:.3f}, max DD {base['max_drawdown']:.2%}, "
        f"trades {int(base['n_trades'])}; @ 5 bps/side: {harsh['total_return']:.2%}"
    )
    print(
        f"research verdict: {research_verdict}; deployment verdict: REJECT "
        f"(evaluation dates were previously exposed); artifacts: {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
