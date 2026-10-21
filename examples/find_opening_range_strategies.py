"""Find and holdout-test selective long/short opening-range portfolios."""
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
from analysis.cuda_scalping import simulate_scalping_candidates_cuda
from analysis.opening_range import (
    ENTRY_FIELDS,
    add_pooled_context_features,
    block_bootstrap_return_confidence,
    correlation_aware_weights,
    PARAM_FIELDS,
    OpeningRangeGrammar,
    describe_opening_range,
    expand_opening_range_exits,
    generate_opening_range_entries,
    opening_range_signal_matrix,
    parameter_neighborhood_support,
    portfolio_metrics_from_curves,
    prepare_opening_range_features,
    replay_opening_range_candidate,
    screen_opening_range_entries,
    select_opening_range_finalists,
    split_opening_range_nested,
    trailing_performance_gate,
)
from analysis.optimizer import SearchSettings, rank_evaluations
from analysis.scalping import (
    aggregate_instrument_holdout,
    align_common_sessions,
    prepare_intraday_bars,
)
from data.event_calendar import load_event_calendar
from data.multitf import get_bars
from data.quotes import load_quotes, quote_coverage, quote_features_at
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


def _aggregate_screens(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(ENTRY_FIELDS, as_index=False, dropna=False)
        .agg(
            screen_trades=("screen_trades", "sum"),
            screen_edge_bps=("screen_edge_bps", "median"),
            screen_win_rate=("screen_win_rate", "median"),
            screen_t_stat=("screen_t_stat", "median"),
            screen_score=("screen_score", "median"),
            instrument_screen_count=("ticker", "nunique"),
            screen_device=("screen_device", "first"),
        )
        .sort_values("screen_score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def _simulate(
    features: pd.DataFrame,
    candidates: list[dict],
    benchmark: pd.DataFrame,
    cost_bps: float,
    session_gate: np.ndarray | None = None,
    maximum_spread_bps: float | None = None,
):
    signals = opening_range_signal_matrix(features, candidates, device="cuda")
    if session_gate is not None:
        import torch
        ordered_sessions = pd.unique(features["session"])
        if len(session_gate) != len(ordered_sessions):
            raise ValueError("session gate must match feature sessions")
        mapping = dict(zip(ordered_sessions, session_gate, strict=True))
        bar_gate = features["session"].map(mapping).to_numpy(dtype=bool)
        signals = signals * torch.as_tensor(bar_gate[:, None], device=signals.device)
    if maximum_spread_bps is not None:
        import torch
        spreads = features["spread_bps"].to_numpy(dtype=float)
        quote_gate = np.isfinite(spreads) & (spreads <= maximum_spread_bps)
        signals = signals * torch.as_tensor(quote_gate[:, None], device=signals.device)
    return simulate_scalping_candidates_cuda(
        features,
        candidates,
        cost_per_side_bps=cost_bps,
        benchmark_bars=benchmark,
        signal_matrix=signals,
        risk_fraction=features["atr_fraction"].to_numpy(dtype=float),
        volatility_exits=True,
        max_trades_per_session=[
            int(candidate["max_trades_per_session"])
            for candidate in candidates
        ],
        return_curves=True,
    )


def _portfolio_row_with_benchmark(
    portfolio: pd.DataFrame,
    instrument_frame: pd.DataFrame,
) -> pd.DataFrame:
    result = portfolio.copy()
    for column in (
        "benchmark_total_return", "benchmark_sharpe", "benchmark_max_drawdown",
    ):
        result[column] = float(instrument_frame[column].iloc[0])
    result["excess_total_return"] = (
        result["total_return"] - result["benchmark_total_return"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=",".join(SCALPING_STOCKS))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--entry-finalists", type=int, default=48)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--cost-bps", type=float, default=2.5)
    parser.add_argument("--context-tickers", default="")
    parser.add_argument("--events", type=Path)
    parser.add_argument("--min-regime-strength", type=float, default=0.0)
    parser.add_argument("--max-entry-spread-bps", type=float, default=8.0)
    parser.add_argument("--max-instrument-weight", type=float, default=0.25)
    parser.add_argument("--min-neighbor-support", type=int, default=2)
    parser.add_argument("--skip-leave-one-out", action="store_true")
    parser.add_argument("--circuit-breaker-lookback", type=int, default=40)
    parser.add_argument("--disable-circuit-breaker", action="store_true")
    parser.add_argument(
        "--context-relation", choices=("none", "same", "inverse"), default="none",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    tickers = tuple(dict.fromkeys(
        ticker.strip().upper()
        for ticker in args.tickers.split(",")
        if ticker.strip()
    ))
    if args.smoke:
        tickers = tickers[:3]
    if not tickers:
        raise SystemExit("Opening-range finder requires at least one instrument.")
    context_tickers = tuple(dict.fromkeys(
        ticker.strip().upper() for ticker in args.context_tickers.split(",")
        if ticker.strip()
    ))
    if args.min_regime_strength < 0:
        raise SystemExit("--min-regime-strength must be non-negative")
    if not 0 < args.max_instrument_weight <= 1:
        raise SystemExit("--max-instrument-weight must be in (0, 1]")
    if args.min_neighbor_support < 0:
        raise SystemExit("--min-neighbor-support must be non-negative")
    if args.max_entry_spread_bps <= 0:
        raise SystemExit("--max-entry-spread-bps must be positive")
    if args.circuit_breaker_lookback < 20:
        raise SystemExit("--circuit-breaker-lookback must be at least 20")
    events = load_event_calendar(args.events) if args.events else None
    if args.context_relation != "none" and not context_tickers:
        raise SystemExit("a context relationship requires --context-tickers")

    raw = {
        ticker: get_bars(ticker, "5m")
        for ticker in (*tickers, BENCHMARK)
    }
    if missing := [ticker for ticker, bars in raw.items() if bars.empty]:
        raise SystemExit(f"No 5m data for {', '.join(missing)}")
    prepared = align_common_sessions({
        ticker: prepare_intraday_bars(bars)
        for ticker, bars in raw.items()
    })
    benchmark = prepared[BENCHMARK]
    context_prepared = {
        ticker: prepare_intraday_bars(get_bars(ticker, "5m"))
        for ticker in context_tickers
    }
    missing_context = [
        ticker for ticker, frame in context_prepared.items() if frame.empty
    ]
    if missing_context:
        raise SystemExit("No 5m context data for " + ", ".join(missing_context))
    features = {
        ticker: prepare_opening_range_features(prepared[ticker], benchmark)
        for ticker in tickers
    }
    if args.context_relation != "none":
        features = {
            ticker: add_pooled_context_features(frame, context_prepared)
            for ticker, frame in features.items()
        }
    if events is not None:
        for ticker, frame in features.items():
            blocked = events[
                events["ticker"].isin([ticker, "*"])
            ]["date"]
            frame.loc[frame["session_date"].isin(blocked), "relative_volume"] = np.nan
    quotes_by_ticker = {ticker: load_quotes(ticker) for ticker in tickers}
    quote_coverage_by_ticker = {}
    for ticker, frame in features.items():
        quotes = quotes_by_ticker[ticker]
        if quotes.empty:
            aligned_quotes = pd.DataFrame({
                column: np.full(len(frame), np.nan)
                for column in ("spread_bps", "quote_imbalance", "microprice_edge_bps")
            })
        else:
            aligned_quotes = quote_features_at(
                quotes, pd.DatetimeIndex(frame["ts"]),
            )
        for quote_column in aligned_quotes:
            frame[quote_column] = aligned_quotes[quote_column].to_numpy()
        quote_coverage_by_ticker[ticker] = quote_coverage(aligned_quotes)
    sip_quote_ready = bool(
        quote_coverage_by_ticker
        and min(quote_coverage_by_ticker.values()) >= 0.80
    )
    window_count = 2 if args.smoke else args.windows
    orientation = {}
    purged = {}
    discovery = {}
    holdouts = {}
    for ticker, frame in features.items():
        orient, purge, windows, holdout = split_opening_range_nested(
            frame, validation_windows=window_count,
        )
        orientation[ticker] = orient
        purged[ticker] = purge
        discovery[ticker] = windows
        holdouts[ticker] = holdout

    effective_maximum_weight = max(
        args.max_instrument_weight, 1.0 / len(tickers),
    )
    portfolio_weight_map = correlation_aware_weights(
        orientation, maximum_weight=effective_maximum_weight,
    )
    portfolio_weights = np.asarray(
        [portfolio_weight_map[ticker] for ticker in tickers], dtype=float,
    )

    grammar = (
        OpeningRangeGrammar(
            opening_range_bars=(3,),
            breakout_atr=(0.0,),
            min_relative_volume=(1.0,),
            market_alignment=(False,), fresh_breakout=(False,),
            context_relation=(args.context_relation,),
            min_regime_strength=(args.min_regime_strength,),
            directions=("long", "short", "both"),
            max_hold_bars=(6,),
            take_profit_atr=(1.0,),
            stop_loss_atr=(1.0,),
        )
        if args.smoke else OpeningRangeGrammar(
            context_relation=(args.context_relation,),
            min_regime_strength=(args.min_regime_strength,),
        )
    )
    entries = generate_opening_range_entries(grammar)
    instrument_screens = []
    for ticker, windows in discovery.items():
        screen = screen_opening_range_entries(
            orientation[ticker],
            entries,
            cost_per_side_bps=args.cost_bps,
            device=args.device,
        )
        screen.insert(0, "ticker", ticker)
        instrument_screens.append(screen)
    screen_by_instrument = pd.concat(instrument_screens, ignore_index=True)
    screened = _aggregate_screens(screen_by_instrument)
    finalists = select_opening_range_finalists(
        screened,
        min(args.entry_finalists, len(screened)),
        min_screen_trades=10 * len(tickers),
    )
    if not finalists:
        raise SystemExit("No opening-range entry passed the screen gate.")
    candidates = expand_opening_range_exits(finalists, grammar)

    evaluation_rows = []
    instrument_rows = []
    discovery_frames_by_window = []
    discovery_curves_by_window = []
    for window_index in range(window_count):
        frames = []
        curves = []
        for ticker in tickers:
            result, equity = _simulate(
                discovery[ticker][window_index], candidates, benchmark,
                args.cost_bps,
                maximum_spread_bps=(
                    args.max_entry_spread_bps if sip_quote_ready else None
                ),
            )
            result.insert(0, "ticker", ticker)
            result.insert(1, "window_idx", window_index)
            frames.append(result)
            curves.append(equity)
            instrument_rows.append(result)
        discovery_frames_by_window.append(frames)
        discovery_curves_by_window.append(curves)
        portfolio = portfolio_metrics_from_curves(
            candidates, frames, curves, weights=portfolio_weights,
        )
        portfolio = _portfolio_row_with_benchmark(portfolio, frames[0])
        portfolio.insert(0, "window_idx", window_index)
        evaluation_rows.append(portfolio)

    evaluations = pd.concat(evaluation_rows, ignore_index=True)
    ranked = rank_evaluations(
        evaluations,
        PARAM_FIELDS,
        SearchSettings(
            algorithm="grid",
            min_trades_per_window=max(15, 2 * len(tickers)),
            stability_penalty=0.50,
            drawdown_penalty=0.20,
            min_positive_windows=window_count,
            min_worst_sharpe=0.0,
        ),
    )
    leave_one_out_rows = []
    leave_one_out_enabled = len(tickers) >= 3 and not args.skip_leave_one_out
    if leave_one_out_enabled:
        for window_index, (window_frames, window_curves) in enumerate(zip(
            discovery_frames_by_window, discovery_curves_by_window, strict=True,
        )):
            for omitted_index, omitted_ticker in enumerate(tickers):
                kept = [index for index in range(len(tickers)) if index != omitted_index]
                omitted_weights = portfolio_weights[kept]
                omitted_weights = omitted_weights / omitted_weights.sum()
                portfolio = portfolio_metrics_from_curves(
                    candidates,
                    [window_frames[index] for index in kept],
                    [window_curves[index] for index in kept],
                    weights=omitted_weights,
                )
                portfolio = _portfolio_row_with_benchmark(
                    portfolio, window_frames[kept[0]],
                )
                portfolio.insert(0, "omitted_ticker", omitted_ticker)
                portfolio.insert(0, "window_idx", window_index * len(tickers) + omitted_index)
                leave_one_out_rows.append(portfolio)
        leave_one_out_evaluations = pd.concat(leave_one_out_rows, ignore_index=True)
        leave_one_out_ranked = rank_evaluations(
            leave_one_out_evaluations, PARAM_FIELDS,
            SearchSettings(
                algorithm="grid",
                min_trades_per_window=max(15, 2 * (len(tickers) - 1)),
                stability_penalty=0.50, drawdown_penalty=0.20,
                min_positive_windows=window_count * len(tickers),
                min_worst_sharpe=0.0,
            ),
        )
        loo_columns = [
            *PARAM_FIELDS, "eligible", "robust_score", "worst_sharpe",
            "positive_windows",
        ]
        ranked = ranked.merge(
            leave_one_out_ranked[loo_columns], on=PARAM_FIELDS, how="left",
            suffixes=("", "_leave_one_out"),
        )
        ranked["eligible"] = (
            ranked["eligible"] & ranked["eligible_leave_one_out"].fillna(False)
        )
        ranked["robust_score"] = np.minimum(
            ranked["robust_score"], ranked["robust_score_leave_one_out"],
        )
    else:
        leave_one_out_evaluations = pd.DataFrame()
        ranked["eligible_leave_one_out"] = ranked["eligible"]
    ranked["neighbor_support"] = parameter_neighborhood_support(ranked)
    ranked["eligible"] = (
        ranked["eligible"] & (ranked["neighbor_support"] >= args.min_neighbor_support)
    )
    ranked = ranked.sort_values(
        ["eligible", "robust_score", "median_sharpe"],
        ascending=[False, False, False], kind="stable",
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    try:
        pbo = combinatorial_probability_overfit(evaluations, PARAM_FIELDS)
    except ValueError as exc:
        pbo = {"probability_overfit": None, "reason": str(exc)}
    trial_sharpes = (
        evaluations.groupby(PARAM_FIELDS, dropna=False)["sharpe"]
        .median().replace([np.inf, -np.inf], np.nan).dropna().tolist()
    )
    validation_session_count = int(sum(
        frame["session"].nunique() for frame in discovery[tickers[0]]
    ))
    dsr = deflated_sharpe(
        trial_sharpes or [0.0], n_obs=validation_session_count,
    )
    eligible = ranked[ranked["eligible"]]
    if eligible.empty:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = ROOT / "results" / f"opening_range_discovery_reject_{stamp}"
        out.mkdir(parents=True, exist_ok=True)
        screened.to_csv(out / "entry_prescreen.csv", index=False)
        screen_by_instrument.to_csv(
            out / "entry_prescreen_by_instrument.csv", index=False,
        )
        ranked.to_csv(out / "discovery_rankings.csv", index=False)
        evaluations.to_csv(out / "discovery_portfolio_windows.csv", index=False)
        if not leave_one_out_evaluations.empty:
            leave_one_out_evaluations.to_csv(
                out / "discovery_leave_one_out.csv", index=False,
            )
        (out / "summary.json").write_text(json.dumps(_json_safe({
            "verdict": "DISCOVERY_REJECT",
            "deployment_approved": False,
            "holdout_opened": False,
            "tickers": list(tickers),
            "benchmark": BENCHMARK,
            "selection_cost_bps_per_side": args.cost_bps,
            "orientation_sessions": int(orientation[tickers[0]]["session"].nunique()),
            "purge_sessions": int(purged[tickers[0]]["session"].nunique()),
            "nested_validation_sessions": int(sum(
                frame["session"].nunique() for frame in discovery[tickers[0]]
            )),
            "required_positive_windows": window_count,
            "leave_one_out_enabled": leave_one_out_enabled,
            "minimum_neighbor_support": args.min_neighbor_support,
            "portfolio_weights": portfolio_weight_map,
            "minimum_worst_window_sharpe": 0.0,
            "entry_rule_count": len(entries),
            "entry_finalist_count": len(finalists),
            "candidate_count": len(candidates),
            "probability_backtest_overfit": pbo,
            "deflated_sharpe": dsr,
            "sip_quote_ready": sip_quote_ready,
            "maximum_entry_spread_bps": args.max_entry_spread_bps,
            "quote_coverage_by_ticker": quote_coverage_by_ticker,
            "event_calendar_ready": events is not None,
            "reason": (
                "no candidate passed every discovery and leave-one-out window "
                "with positive worst-window Sharpe and neighborhood support"
            ),
        }), indent=2), encoding="utf-8")
        print(
            "DISCOVERY_REJECT: no all-window positive candidate at "
            f"{args.cost_bps:g} bps/side; holdout_opened=false"
        )
        print(f"artifacts: {out}")
        return 2
    winner = eligible.iloc[0]
    winner_params = {field: winner[field] for field in PARAM_FIELDS}

    circuit_gates = {}
    circuit_diagnostics = {}
    for ticker in tickers:
        session_count = int(holdouts[ticker]["session"].nunique())
        if args.disable_circuit_breaker:
            circuit_gates[ticker] = np.ones(session_count, dtype=bool)
        else:
            prior_returns = []
            for validation_frame in discovery[ticker]:
                _, validation_curve = _simulate(
                    validation_frame, [winner_params], benchmark, args.cost_bps,
                    maximum_spread_bps=(
                        args.max_entry_spread_bps if sip_quote_ready else None
                    ),
                )
                prior_returns.extend(np.concatenate([
                    validation_curve[:1, 0] - 1.0,
                    validation_curve[1:, 0] / validation_curve[:-1, 0] - 1.0,
                ]).tolist())
            _, shadow_curve = _simulate(
                holdouts[ticker], [winner_params], benchmark, args.cost_bps,
                maximum_spread_bps=(
                    args.max_entry_spread_bps if sip_quote_ready else None
                ),
            )
            shadow_returns = np.concatenate([
                shadow_curve[:1, 0] - 1.0,
                shadow_curve[1:, 0] / shadow_curve[:-1, 0] - 1.0,
            ])
            circuit_gates[ticker] = trailing_performance_gate(
                np.asarray(prior_returns), shadow_returns,
                lookback=args.circuit_breaker_lookback,
                min_observations=20, minimum_sharpe=0.0,
            )
        circuit_diagnostics[ticker] = {
            "enabled_sessions": int(circuit_gates[ticker].sum()),
            "total_sessions": session_count,
            "enabled_fraction": float(circuit_gates[ticker].mean()),
        }

    holdout_instruments = []
    holdout_portfolios = []
    holdout_trade_ledgers = []
    for cost in sorted(set([args.cost_bps, 2.5, 5.0])):
        frames = []
        curves = []
        for ticker in tickers:
            result, equity = _simulate(
                holdouts[ticker], [winner_params], benchmark, cost,
                session_gate=circuit_gates[ticker],
                maximum_spread_bps=(
                    args.max_entry_spread_bps if sip_quote_ready else None
                ),
            )
            result.insert(0, "ticker", ticker)
            result.insert(1, "cost_scenario_bps_per_side", cost)
            frames.append(result)
            curves.append(equity)
            holdout_instruments.append(result)
            if np.isclose(cost, args.cost_bps):
                ledger = replay_opening_range_candidate(
                    holdouts[ticker], winner_params,
                    cost_per_side_bps=cost,
                )
                enabled_dates = pd.unique(holdouts[ticker]["session_date"])[
                    circuit_gates[ticker]
                ]
                ledger = ledger[ledger["session_date"].isin(enabled_dates)].copy()
                if not ledger.empty and not quotes_by_ticker[ticker].empty:
                    quote_features = quote_features_at(
                        quotes_by_ticker[ticker],
                        pd.DatetimeIndex(pd.to_datetime(ledger["entry_ts"], utc=True)),
                    )
                    for quote_column in quote_features:
                        ledger[quote_column] = quote_features[quote_column].to_numpy()
                else:
                    for quote_column in (
                        "spread_bps", "quote_imbalance", "microprice_edge_bps",
                    ):
                        ledger[quote_column] = np.nan
                ledger.insert(0, "ticker", ticker)
                holdout_trade_ledgers.append(ledger)
        portfolio = portfolio_metrics_from_curves(
            [winner_params], frames, curves, weights=portfolio_weights,
        )
        portfolio = _portfolio_row_with_benchmark(portfolio, frames[0])
        portfolio.insert(0, "cost_scenario_bps_per_side", cost)
        holdout_portfolios.append(portfolio)

    by_instrument = pd.concat(holdout_instruments, ignore_index=True)
    portfolio_holdout = pd.concat(holdout_portfolios, ignore_index=True)
    breadth = aggregate_instrument_holdout(by_instrument)
    trade_ledger = (
        pd.concat(holdout_trade_ledgers, ignore_index=True)
        if holdout_trade_ledgers else pd.DataFrame()
    )
    trade_attribution = {
        "trade_count": int(len(trade_ledger)),
        "win_rate": (
            float((trade_ledger["net_return"] > 0).mean())
            if not trade_ledger.empty else None
        ),
        "gross_return_sum": (
            float(trade_ledger["gross_return"].sum())
            if not trade_ledger.empty else 0.0
        ),
        "cost_drag_sum": (
            float(trade_ledger["cost_drag"].sum())
            if not trade_ledger.empty else 0.0
        ),
        "net_return_sum": (
            float(trade_ledger["net_return"].sum())
            if not trade_ledger.empty else 0.0
        ),
        "exit_reason_counts": (
            trade_ledger["exit_reason"].value_counts().to_dict()
            if not trade_ledger.empty else {}
        ),
    }
    all_holdout_dates = pd.Index(pd.unique(holdouts[tickers[0]]["session_date"]))
    if trade_ledger.empty:
        holdout_daily_returns = np.zeros(len(all_holdout_dates), dtype=float)
    else:
        daily_by_ticker = trade_ledger.pivot_table(
            index="session_date", columns="ticker", values="net_return",
            aggfunc="sum", fill_value=0.0,
        ).reindex(all_holdout_dates, fill_value=0.0)
        daily_by_ticker = daily_by_ticker.reindex(columns=tickers, fill_value=0.0)
        holdout_daily_returns = daily_by_ticker.mul(
            pd.Series(portfolio_weight_map), axis=1,
        ).sum(axis=1).to_numpy(dtype=float)
    bootstrap = block_bootstrap_return_confidence(holdout_daily_returns)
    contributions = (
        trade_ledger.groupby("ticker")["net_return"].sum().mul(
            pd.Series(portfolio_weight_map), fill_value=0.0,
        )
        if not trade_ledger.empty else pd.Series(dtype=float)
    )
    largest_contribution_share = (
        float(contributions.abs().max() / contributions.abs().sum())
        if not contributions.empty and contributions.abs().sum() > 0 else 1.0
    )

    base = portfolio_holdout.loc[
        portfolio_holdout["cost_scenario_bps_per_side"] == args.cost_bps
    ].iloc[0]
    harsh = portfolio_holdout.loc[
        portfolio_holdout["cost_scenario_bps_per_side"] == 5.0
    ].iloc[0]
    winner_windows = evaluations
    for field in PARAM_FIELDS:
        winner_windows = winner_windows[winner_windows[field] == winner[field]]
    discovery_positive = int((winner_windows["total_return"] > 0).sum())
    research_pass = bool(
        pbo.get("probability_overfit") is not None
        and float(pbo["probability_overfit"]) <= 0.20
        and float(dsr["p_value"]) >= 0.95
        and float(bootstrap["probability_positive"]) >= 0.95
        and 
        base["total_return"] > 0
        and base["excess_total_return"] > 0
        and base["sharpe"] > 1.0
        and base["max_drawdown"] > -0.10
        and harsh["total_return"] > 0
        and discovery_positive == window_count
    )
    research_verdict = "PASS" if research_pass else "REJECT"

    ranked.insert(1, "description", [
        describe_opening_range(row) for _, row in ranked.iterrows()
    ])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"opening_range_finder_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    screened.to_csv(out / "entry_prescreen.csv", index=False)
    screen_by_instrument.to_csv(
        out / "entry_prescreen_by_instrument.csv", index=False,
    )
    ranked.to_csv(out / "discovery_rankings.csv", index=False)
    evaluations.to_csv(out / "discovery_portfolio_windows.csv", index=False)
    if not leave_one_out_evaluations.empty:
        leave_one_out_evaluations.to_csv(
            out / "discovery_leave_one_out.csv", index=False,
        )
    pd.concat(instrument_rows, ignore_index=True).to_csv(
        out / "discovery_by_instrument.csv", index=False,
    )
    by_instrument.to_csv(out / "winner_holdout_by_instrument.csv", index=False)
    portfolio_holdout.to_csv(out / "winner_portfolio_cost_stress.csv", index=False)
    qualification_costs = portfolio_holdout.rename(
        columns={"n_trades": "portfolio_events"},
    ).copy()
    qualification_costs.to_csv(out / "evaluation_cost_stress.csv", index=False)
    breadth.to_csv(out / "winner_breadth_cost_stress.csv", index=False)
    trade_ledger.to_csv(out / "winner_trade_ledger.csv", index=False)
    evaluation_base_cost = base.to_dict()
    evaluation_base_cost["portfolio_events"] = int(base["n_trades"])
    candidate_sha256 = candidate_fingerprint(winner_params)
    summary = {
        "verdict": "REJECT",
        "research_verdict": research_verdict,
        "deployment_approved": False,
        "holdout_opened": True,
        "tickers": list(tickers),
        "benchmark": BENCHMARK,
        "instrument_count": len(tickers),
        "portfolio_weights": portfolio_weight_map,
        "maximum_instrument_weight": effective_maximum_weight,
        "leave_one_out_enabled": leave_one_out_enabled,
        "minimum_neighbor_support": args.min_neighbor_support,
        "bar_count": int(sum(len(frame) for frame in features.values())),
        "session_count": int(benchmark["session"].nunique()),
        "entry_rule_count": len(entries),
        "entry_finalist_count": len(finalists),
        "candidate_count": len(candidates),
        "portfolio_window_count": window_count,
        "orientation_sessions": int(orientation[tickers[0]]["session"].nunique()),
        "purge_sessions": int(purged[tickers[0]]["session"].nunique()),
        "nested_validation_sessions": int(sum(
            frame["session"].nunique() for frame in discovery[tickers[0]]
        )),
        "holdout_sessions": int(holdouts[tickers[0]]["session"].nunique()),
        "evaluation_sessions": int(holdouts[tickers[0]]["session"].nunique()),
        "evaluation_base_cost": evaluation_base_cost,
        "probability_backtest_overfit": pbo,
        "deflated_sharpe": dsr,
        "block_bootstrap": bootstrap,
        "largest_absolute_contribution_share": largest_contribution_share,
        "discovery_positive_windows": discovery_positive,
        "sip_quote_ready": sip_quote_ready,
        "maximum_entry_spread_bps": args.max_entry_spread_bps,
        "quote_coverage_by_ticker": quote_coverage_by_ticker,
        "point_in_time_membership": False,
        "event_calendar_ready": events is not None,
        "historical_interval_unexposed": False,
        "candidate_sha256": candidate_sha256,
        "circuit_breaker": {
            "enabled": not args.disable_circuit_breaker,
            "lookback_sessions": args.circuit_breaker_lookback,
            "diagnostics": circuit_diagnostics,
        },
        "screen_device": str(screened.iloc[0]["screen_device"]),
        "simulation_device": "cuda",
        "selection_cost_bps_per_side": args.cost_bps,
        "discovery_stability_gates": {
            "required_positive_windows": window_count,
            "minimum_worst_window_sharpe": 0.0,
        },
        "winner": {
            **winner.to_dict(),
            "description": describe_opening_range(winner),
        },
        "holdout_base_cost": base.to_dict(),
        "holdout_harsh_cost": harsh.to_dict(),
        "holdout_trade_attribution": trade_attribution,
        "holdout_breadth_base": breadth.loc[
            breadth["cost_scenario_bps_per_side"] == args.cost_bps
        ].iloc[0].to_dict(),
    }
    historical = historical_qualification(summary, qualification_costs)
    summary["historical_qualification"] = historical
    summary["verdict"] = (
        "HISTORICAL_PASS_PAPER_REQUIRED" if historical["qualified"] else "REJECT"
    )
    (out / "frozen_candidate.json").write_text(json.dumps(_json_safe({
        "candidate": winner_params,
        "sha256": candidate_sha256,
        "data_end": str(max(frame["ts"].max() for frame in holdouts.values())),
        "mutable": False,
        "deployment_approved": False,
    }), indent=2), encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8",
    )

    print(
        f"opening-range finder: {len(tickers)} stocks x {len(entries)} entries "
        f"-> {len(finalists)} finalists -> {len(candidates)} candidates; "
        f"{window_count} shared-capital discovery portfolios + holdout"
    )
    print(f"screen/validation device: cuda; bars: {summary['bar_count']:,}")
    print(ranked[[
        "rank", "opening_range_bars", "breakout_atr", "min_relative_volume",
        "market_alignment", "fresh_breakout", "direction",
        "take_profit_atr", "stop_loss_atr",
        "max_hold_bars", "robust_score", "median_sharpe", "total_trades",
    ]].head(15).to_string(index=False))
    print(f"winner: {describe_opening_range(winner)}")
    print(
        f"portfolio holdout @ {args.cost_bps:g} bps/side: "
        f"return {base['total_return']:.2%}, Sharpe {base['sharpe']:.3f}, "
        f"max drawdown {base['max_drawdown']:.2%}, "
        f"SPY {base['benchmark_total_return']:.2%}, "
        f"excess {base['excess_total_return']:.2%}, "
        f"trades {int(base['n_trades'])}"
    )
    print(
        f"portfolio holdout @ 5 bps/side: {harsh['total_return']:.2%}; "
        f"research verdict: {research_verdict}; qualification: {summary['verdict']}"
    )
    print(f"artifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
