"""Find, optimize, and holdout-test generic 5-minute scalping strategies."""
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

from analysis.optimizer import SearchSettings, rank_evaluations
from analysis.scalping import (
    ScalpingGrammar,
    aggregate_entry_screens,
    aggregate_group_holdout,
    aggregate_instrument_holdout,
    align_common_sessions,
    describe_scalper,
    expand_exit_rules,
    generate_entry_rules,
    prepare_intraday_bars,
    screen_entry_rules,
    select_entry_finalists,
    simulate_scalping_candidates,
    split_discovery_holdout,
)
from data.multitf import get_bars


from data.tickers import BENCHMARK, SCALPING_GROUPS, SCALPING_UNIVERSE, scalping_asset_group
PARAM_NAMES = [
    "family", "lookback_bars", "entry_threshold", "max_hold_bars",
    "entry_rule_id", "take_profit_bps", "stop_loss_bps",
]


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=",".join(SCALPING_UNIVERSE))
    parser.add_argument("--timeframe", default="5m", choices=("5m",))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--entry-finalists", type=int, default=18)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--cost-bps", type=float, default=1.0, help="spread/slippage cost per side")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    tickers = tuple(dict.fromkeys(t.strip().upper() for t in args.tickers.split(",") if t.strip()))
    if args.smoke:
        tickers = tickers[:2]
    if not tickers:
        raise SystemExit("Scalping Finder requires at least one traded instrument.")
    raw_by_ticker = {
        ticker: get_bars(ticker, args.timeframe)
        for ticker in (*tickers, BENCHMARK)
    }
    missing = [ticker for ticker, raw in raw_by_ticker.items() if raw.empty]
    if missing:
        raise SystemExit(
            f"No {args.timeframe} data for {', '.join(missing)}. Run "
            f"data/ingest_alpaca_intraday.py first."
        )
    prepared = align_common_sessions({
        ticker: prepare_intraday_bars(raw)
        for ticker, raw in raw_by_ticker.items()
    })
    benchmark_bars = prepared[BENCHMARK]
    bars_by_ticker = {ticker: prepared[ticker] for ticker in tickers}
    window_count = 2 if args.smoke else args.windows
    discovery_by_ticker = {}
    holdout_by_ticker = {}
    for ticker, bars in bars_by_ticker.items():
        windows, holdout = split_discovery_holdout(
            bars, discovery_fraction=0.8, n_windows=window_count,
        )
        discovery_by_ticker[ticker] = windows
        holdout_by_ticker[ticker] = holdout
    grammar = (
        ScalpingGrammar(
            families=("mean_reversion", "momentum"),
            lookback_bars=(6,),
            mean_reversion_thresholds=(1.0,),
            return_threshold_bps=(5.0,),
            max_hold_bars=(3,),
            take_profit_bps=(10.0,),
            stop_loss_bps=(10.0,),
        )
        if args.smoke else ScalpingGrammar()
    )
    entry_rules = generate_entry_rules(grammar)
    instrument_screens = []
    for ticker, windows in discovery_by_ticker.items():
        discovery = pd.concat(windows, ignore_index=True)
        screen = screen_entry_rules(
            discovery, entry_rules,
            cost_per_side_bps=args.cost_bps,
            device=args.device,
        )
        screen.insert(0, "ticker", ticker)
        screen.insert(1, "asset_group", scalping_asset_group(ticker))
        instrument_screens.append(screen)
    screen_by_instrument = pd.concat(instrument_screens, ignore_index=True)
    screened = aggregate_entry_screens(screen_by_instrument)
    entry_finalists = select_entry_finalists(
        screened, min(args.entry_finalists, len(screened)),
        min_screen_trades=30 * len(tickers),
    )
    candidates = expand_exit_rules(entry_finalists, grammar)
    if not candidates:
        raise SystemExit("No entry rule passed the cross-instrument screen gate.")

    total_bars = sum(len(bars) for bars in bars_by_ticker.values())
    common_sessions = benchmark_bars["session"].nunique()
    print(
        f"scalping finder: {len(tickers)} instruments x {len(entry_rules)} entry rules -> "
        f"{len(entry_finalists)} entry finalists -> {len(candidates)} full candidates; "
        f"{window_count} discovery windows each + untouched holdouts"
    )
    print(
        f"data: {total_bars:,} traded regular-session bars, "
        f"{common_sessions} common sessions, {min(tickers)}..{max(tickers)} alphabetically"
    )
    print(f"entry screen device: {screened.iloc[0]['screen_device']}")

    evaluations: list[pd.DataFrame] = []
    for ticker, windows in discovery_by_ticker.items():
        for window_idx, window in enumerate(windows):
            result = simulate_scalping_candidates(
                window, candidates, cost_per_side_bps=args.cost_bps,
                benchmark_bars=benchmark_bars,
                device=args.device,
            )
            result.insert(0, "ticker", ticker)
            result.insert(1, "asset_group", scalping_asset_group(ticker))
            result.insert(1, "window_idx", window_idx)
            result.insert(2, "start", window["ts"].min())
            result.insert(3, "end", window["ts"].max())
            evaluations.append(result)
    evaluation_frame = pd.concat(evaluations, ignore_index=True)
    ranked = rank_evaluations(
        evaluation_frame,
        PARAM_NAMES,
        SearchSettings(
            algorithm="grid",
            min_trades_per_window=5,
            stability_penalty=0.35,
            drawdown_penalty=0.15,
        ),
    )
    eligible = ranked[ranked["eligible"]]
    if eligible.empty:
        raise SystemExit("No scalping candidate was eligible across every discovery window.")

    winner = eligible.iloc[0]
    winner_params = {name: winner[name] for name in PARAM_NAMES}
    cost_levels = sorted(set([args.cost_bps, 2.5, 5.0]))
    holdout_rows: list[pd.DataFrame] = []
    for ticker, holdout in holdout_by_ticker.items():
        for cost in cost_levels:
            row = simulate_scalping_candidates(
                holdout, [winner_params], cost_per_side_bps=cost,
                benchmark_bars=benchmark_bars,
                device=args.device,
            )
            row.insert(0, "ticker", ticker)
            row.insert(1, "asset_group", scalping_asset_group(ticker))
            row.insert(1, "cost_scenario_bps_per_side", cost)
            holdout_rows.append(row)
    holdout_result = pd.concat(holdout_rows, ignore_index=True)
    holdout_aggregate = aggregate_instrument_holdout(holdout_result)
    holdout_by_group = aggregate_group_holdout(holdout_result)
    base_holdout = holdout_aggregate.loc[
        holdout_aggregate["cost_scenario_bps_per_side"] == args.cost_bps
    ].iloc[0]
    harsh = holdout_aggregate.loc[
        holdout_aggregate["cost_scenario_bps_per_side"] == max(cost_levels)
    ].iloc[0]
    required_majority = int(np.ceil(0.6 * len(tickers)))
    verdict_pass = bool(
        base_holdout["sharpe"] > base_holdout["benchmark_sharpe"]
        and base_holdout["excess_total_return"] > 0
        and base_holdout["n_trades"] >= 20 * len(tickers)
        and base_holdout["positive_instruments"] >= required_majority
        and base_holdout["beat_spy_instruments"] >= required_majority
        and harsh["median_total_return"] > 0
    )
    verdict = "PASS" if verdict_pass else "REJECT"

    ranked.insert(1, "description", [
        describe_scalper(row) for _, row in ranked.iterrows()
    ])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"scalping_finder_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    screened.to_csv(out / "entry_prescreen.csv", index=False)
    screen_by_instrument.to_csv(out / "entry_prescreen_by_instrument.csv", index=False)
    ranked.to_csv(out / "discovery_rankings.csv", index=False)
    evaluation_frame.to_csv(out / "discovery_windows.csv", index=False)
    holdout_result.to_csv(out / "winner_holdout_by_instrument.csv", index=False)
    holdout_aggregate.to_csv(out / "winner_holdout_cost_stress.csv", index=False)
    holdout_by_group.to_csv(out / "winner_holdout_by_group.csv", index=False)
    base_group_holdout = holdout_by_group.loc[
        holdout_by_group["cost_scenario_bps_per_side"] == args.cost_bps
    ]
    summary = {
        "verdict": verdict,
        "tickers": list(tickers),
        "benchmark": BENCHMARK,
        "asset_groups": {
            name: len(symbols)
            for name, symbols in SCALPING_GROUPS.items()
        },
        "instrument_count": len(tickers),
        "timeframe": args.timeframe,
        "bar_count": total_bars,
        "session_count": int(common_sessions),
        "discovery_fraction": 0.8,
        "entry_rule_count": len(entry_rules),
        "entry_finalist_count": len(entry_finalists),
        "candidate_count": len(candidates),
        "instrument_window_cells": len(tickers) * window_count,
        "winner": {**winner.to_dict(), "description": describe_scalper(winner)},
        "holdout_base_cost": base_holdout.to_dict(),
        "holdout_harsh_cost": harsh.to_dict(),
        "holdout_base_by_group": base_group_holdout.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8",
    )

    print(ranked[[
        "rank", "family", "lookback_bars", "entry_threshold",
        "take_profit_bps", "stop_loss_bps", "max_hold_bars",
        "robust_score", "median_sharpe", "worst_sharpe", "total_trades",
    ]].head(15).to_string(index=False))
    print(f"winner: {describe_scalper(winner)}")
    print(
        f"holdout @ {args.cost_bps:g} bps/side: average instrument return "
        f"{base_holdout['total_return']:.2%}, median Sharpe {base_holdout['sharpe']:.3f}; "
        f"SPY return {base_holdout['benchmark_total_return']:.2%}, "
        f"Sharpe {base_holdout['benchmark_sharpe']:.3f}; median excess "
        f"{base_holdout['excess_total_return']:.2%}; positive "
        f"{int(base_holdout['positive_instruments'])}/{len(tickers)}"
    )
    print(
        f"holdout @ {max(cost_levels):g} bps/side: "
        f"median instrument return {harsh['median_total_return']:.2%}"
    )
    print("holdout by asset group @ base cost:")
    print(base_group_holdout[[
        "asset_group", "instrument_count", "total_return", "sharpe",
        "positive_instruments", "beat_spy_instruments", "n_trades",
    ]].to_string(index=False))
    print(f"verdict: {verdict}")
    print(f"artifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
