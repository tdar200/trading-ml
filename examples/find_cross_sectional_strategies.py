"""GPU search and untouched-holdout test for market-neutral intraday portfolios."""
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

from analysis.cross_sectional import (
    PARAM_FIELDS,
    CrossSectionalGrammar,
    combinatorial_probability_overfit,
    describe_cross_sectional,
    generate_cross_sectional_candidates,
    prepare_cross_sectional_panel,
    simulate_cross_sectional_candidates,
    simulate_cross_sectional_winner,
    split_panel_discovery_holdout,
)
from analysis.optimizer import SearchSettings, rank_evaluations
from analysis.deflated_sharpe import deflated_sharpe
from data.multitf import get_bars
from data.tickers import BENCHMARK, SCALPING_STOCKS


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
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _winner_params(row: pd.Series) -> dict:
    return {field: row[field] for field in PARAM_FIELDS}


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
        raise SystemExit("Cross-sectional finder requires at least four stocks.")
    bars = {ticker: get_bars(ticker, "5m") for ticker in tickers}
    benchmark = get_bars(BENCHMARK, "5m")
    if missing := [ticker for ticker, frame in {**bars, BENCHMARK: benchmark}.items() if frame.empty]:
        raise SystemExit(f"No 5m data for {', '.join(missing)}")

    panel = prepare_cross_sectional_panel(bars, benchmark)
    window_count = 4 if args.smoke else args.windows
    windows, holdout = split_panel_discovery_holdout(
        panel, discovery_fraction=0.8, n_windows=window_count,
    )
    grammar = (
        CrossSectionalGrammar(
            signal_families=("momentum", "reversal"),
            decision_bars=(24,),
            lookback_bars=(3,), hold_bars=(3,), top_n=(2,),
            min_dispersion_bps=(5.0,), min_relative_volume=(0.75,),
            max_market_move_bps=(50.0,),
        )
        if args.smoke else CrossSectionalGrammar()
    )
    candidates = generate_cross_sectional_candidates(grammar)
    evaluations = []
    for window_index, window in enumerate(windows):
        result = simulate_cross_sectional_candidates(
            window, candidates, cost_per_side_bps=args.cost_bps, device=args.device,
        )
        result.insert(0, "window_idx", window_index)
        evaluations.append(result)
    evaluation_frame = pd.concat(evaluations, ignore_index=True)
    ranked = rank_evaluations(
        evaluation_frame,
        PARAM_FIELDS,
        SearchSettings(
            algorithm="grid", min_trades_per_window=max(10, len(tickers) * 2),
            stability_penalty=0.5, drawdown_penalty=0.2,
        ),
    )
    eligible = ranked[ranked["eligible"]]
    if eligible.empty:
        raise SystemExit("No candidate was eligible in every discovery window.")
    winner = eligible.iloc[0]
    winner_params = _winner_params(winner)

    cost_rows = []
    contribution_rows = []
    for cost in sorted(set([args.cost_bps, 2.5, 5.0])):
        result, contribution = simulate_cross_sectional_winner(
            holdout, winner_params, cost_per_side_bps=cost, device=args.device,
        )
        result.insert(0, "cost_scenario_bps_per_side", cost)
        contribution.insert(0, "cost_scenario_bps_per_side", cost)
        cost_rows.append(result)
        contribution_rows.append(contribution)
    holdout_costs = pd.concat(cost_rows, ignore_index=True)
    contributions = pd.concat(contribution_rows, ignore_index=True)
    base = holdout_costs.loc[
        holdout_costs["cost_scenario_bps_per_side"] == args.cost_bps
    ].iloc[0]
    harsh = holdout_costs.loc[
        holdout_costs["cost_scenario_bps_per_side"] == 5.0
    ].iloc[0]
    base_contributions = contributions[
        contributions["cost_scenario_bps_per_side"] == args.cost_bps
    ]["net_additive_contribution"]
    concentration = float(
        base_contributions.abs().max() / base_contributions.abs().sum()
    ) if base_contributions.abs().sum() > 0 else float("inf")
    winner_windows = evaluation_frame.copy()
    for field in PARAM_FIELDS:
        winner_windows = winner_windows[winner_windows[field] == winner[field]]
    positive_windows = int((winner_windows["total_return"] > 0).sum())
    required_positive = int(np.ceil(0.75 * window_count))
    pbo = combinatorial_probability_overfit(evaluation_frame)
    discovery_sessions = sum(len(np.unique(window.sessions)) for window in windows)
    dsr = deflated_sharpe(
        ranked.loc[ranked["eligible"], "median_sharpe"].dropna().tolist(),
        n_obs=discovery_sessions,
    )
    verdict_pass = bool(
        positive_windows >= required_positive
        and base["total_return"] > 0
        and base["sharpe"] > 1.0
        and base["max_drawdown"] > -0.10
        and base["n_trades"] >= 200
        and harsh["total_return"] > 0
        and concentration <= 0.25
        and base["mean_abs_beta_exposure"] <= 0.02
        and pbo["probability_overfit"] <= 0.20
        and dsr["p_value"] >= 0.95
    )
    verdict = "PASS" if verdict_pass else "REJECT"

    ranked.insert(1, "description", [
        describe_cross_sectional(row) for _, row in ranked.iterrows()
    ])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"cross_sectional_finder_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out / "discovery_rankings.csv", index=False)
    evaluation_frame.to_csv(out / "discovery_portfolio_windows.csv", index=False)
    holdout_costs.to_csv(out / "winner_holdout_cost_stress.csv", index=False)
    contributions.to_csv(out / "winner_holdout_contributions.csv", index=False)
    summary = {
        "verdict": verdict,
        "tickers": list(tickers),
        "benchmark": BENCHMARK,
        "instrument_count": len(tickers),
        "aligned_bar_count": len(panel.timestamps),
        "session_count": int(len(np.unique(panel.sessions))),
        "candidate_count": len(candidates),
        "portfolio_window_count": window_count,
        "simulation_device": str(base["simulation_device"]),
        "winner": {**winner.to_dict(), "description": describe_cross_sectional(winner)},
        "discovery_positive_windows": positive_windows,
        "holdout_base_cost": base.to_dict(),
        "holdout_harsh_cost": harsh.to_dict(),
        "largest_absolute_contribution_share": concentration,
        "probability_backtest_overfit": pbo,
        "deflated_sharpe": dsr,
        "gates": {
            "positive_discovery_windows_required": required_positive,
            "minimum_holdout_sharpe": 1.0,
            "minimum_holdout_trades": 200,
            "maximum_drawdown": -0.10,
            "maximum_contribution_share": 0.25,
            "maximum_mean_absolute_beta_exposure": 0.02,
            "must_profit_at_5_bps_per_side": True,
            "maximum_probability_backtest_overfit": 0.20,
            "minimum_deflated_sharpe_confidence": 0.95,
        },
    }
    (out / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8",
    )

    print(
        f"cross-sectional finder: {len(tickers)} stocks x {len(candidates)} "
        f"candidates x {window_count} discovery windows + untouched holdout"
    )
    print(f"device: {base['simulation_device']}; aligned bars: {len(panel.timestamps):,}")
    print(ranked[[
        "rank", *PARAM_FIELDS, "robust_score", "median_sharpe",
        "worst_sharpe", "total_trades",
    ]].head(15).to_string(index=False))
    print(f"winner: {describe_cross_sectional(winner)}")
    print(
        f"holdout @ {args.cost_bps:g} bps/side: return {base['total_return']:.2%}, "
        f"Sharpe {base['sharpe']:.3f}, max drawdown {base['max_drawdown']:.2%}, "
        f"trades {int(base['n_trades'])}, mean |beta| "
        f"{base['mean_abs_beta_exposure']:.3f}"
    )
    print(
        f"holdout @ 5 bps/side: {harsh['total_return']:.2%}; "
        f"largest contribution share: {concentration:.1%}; verdict: {verdict}"
    )
    print(
        f"PBO: {pbo['probability_overfit']:.1%}; Deflated Sharpe confidence: "
        f"{dsr['p_value']:.1%}"
    )
    print(f"artifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
