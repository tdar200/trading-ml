"""Tests for the execution-aware intraday scalping finder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.scalping import (
    ScalpingGrammar,
    aggregate_entry_screens,
    aggregate_group_holdout,
    aggregate_instrument_holdout,
    align_common_sessions,
    expand_exit_rules,
    generate_entry_rules,
    prepare_intraday_bars,
    screen_entry_rules,
    select_entry_finalists,
    simulate_scalping_candidates,
    split_discovery_holdout,
)


def _intraday_frame(sessions: int = 30, bars_per_session: int = 12) -> pd.DataFrame:
    rows = []
    price = 100.0
    for date in pd.bdate_range("2024-01-02", periods=sessions):
        start = pd.Timestamp(date.date(), tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
        for bar in range(bars_per_session):
            price *= 1.0 + (0.0008 if bar % 4 < 2 else -0.0003)
            rows.append({
                "ts": (start + pd.Timedelta(minutes=5 * bar)).tz_convert("UTC"),
                "open": price * 0.9999,
                "high": price * 1.0005,
                "low": price * 0.9995,
                "close": price,
                "volume": 10_000.0,
            })
    rows.append({
        "ts": pd.Timestamp("2024-01-02T08:00:00Z"),
        "open": 99.0, "high": 99.0, "low": 99.0, "close": 99.0, "volume": 1.0,
    })
    return pd.DataFrame(rows)


def _candidate() -> dict:
    return {
        "family": "momentum",
        "lookback_bars": 3,
        "entry_threshold": 1.0,
        "max_hold_bars": 3,
        "entry_rule_id": "momentum-test",
        "take_profit_bps": 20.0,
        "stop_loss_bps": 20.0,
    }


def test_prepare_filters_extended_hours_and_assigns_sessions():
    bars = prepare_intraday_bars(_intraday_frame())
    assert len(bars) == 30 * 12
    assert bars["session"].nunique() == 30
    assert bars.groupby("session")["last_bar"].sum().eq(1).all()
    assert bars["minute_of_day"].between(570, 960).all()


def test_prepare_rejects_duplicate_timestamps():
    frame = _intraday_frame()
    frame = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_intraday_bars(frame)


def test_split_keeps_untouched_whole_session_holdout():
    bars = prepare_intraday_bars(_intraday_frame())
    windows, holdout = split_discovery_holdout(bars, discovery_fraction=0.8, n_windows=4)
    discovery_sessions = set(pd.concat(windows)["session"])
    holdout_sessions = set(holdout["session"])
    assert len(windows) == 4
    assert discovery_sessions.isdisjoint(holdout_sessions)
    assert len(holdout_sessions) == 6


def test_default_grammar_has_108_entry_rules_and_972_full_candidates():
    grammar = ScalpingGrammar()
    entries = generate_entry_rules(grammar)
    assert len(entries) == 108
    assert len(expand_exit_rules(entries, grammar)) == 972


def test_entry_screen_runs_on_cpu_and_preserves_families():
    bars = prepare_intraday_bars(_intraday_frame())
    grammar = ScalpingGrammar(
        families=("momentum", "mean_reversion"),
        lookback_bars=(3,),
        mean_reversion_thresholds=(0.5,),
        return_threshold_bps=(1.0,),
        max_hold_bars=(1,),
        take_profit_bps=(10.0,),
        stop_loss_bps=(10.0,),
    )
    screened = screen_entry_rules(bars, generate_entry_rules(grammar), device="cpu")
    assert set(screened["family"]) == {"momentum", "mean_reversion"}
    assert set(screened["screen_device"]) == {"cpu"}
    assert screened["screen_trades"].gt(0).all()
    screened["screen_score"] = [2.0, 1.0]
    finalists = select_entry_finalists(screened, 2, min_screen_trades=1)
    assert {row["family"] for row in finalists} == {"momentum", "mean_reversion"}


def test_cuda_entry_screen_matches_cpu_reference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    bars = prepare_intraday_bars(_intraday_frame())
    grammar = ScalpingGrammar(
        families=("momentum", "mean_reversion"),
        lookback_bars=(3,),
        mean_reversion_thresholds=(0.5,),
        return_threshold_bps=(1.0,),
        max_hold_bars=(1,),
        take_profit_bps=(10.0,),
        stop_loss_bps=(10.0,),
    )
    rules = generate_entry_rules(grammar)
    cpu = screen_entry_rules(bars, rules, device="cpu").sort_values("entry_rule_id")
    cuda = screen_entry_rules(bars, rules, device="cuda").sort_values("entry_rule_id")

    assert cuda["screen_device"].eq("cuda").all()
    assert cuda["screen_trades"].tolist() == cpu["screen_trades"].tolist()
    np.testing.assert_allclose(
        cuda[["screen_edge_bps", "screen_win_rate", "screen_t_stat", "screen_score"]],
        cpu[["screen_edge_bps", "screen_win_rate", "screen_t_stat", "screen_score"]],
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


def test_costs_reduce_returns_and_market_comparison_is_reported():
    bars = prepare_intraday_bars(_intraday_frame())
    low_cost = simulate_scalping_candidates(
        bars, [_candidate()], cost_per_side_bps=0.0, device="cpu",
    ).iloc[0]
    high_cost = simulate_scalping_candidates(
        bars, [_candidate()], cost_per_side_bps=5.0, device="cpu",
    ).iloc[0]
    assert low_cost["n_trades"] > 0
    assert high_cost["total_return"] < low_cost["total_return"]
    assert high_cost["max_drawdown"] <= low_cost["max_drawdown"]
    assert np.isfinite(low_cost["benchmark_total_return"])
    assert low_cost["excess_total_return"] == pytest.approx(
        low_cost["total_return"] - low_cost["benchmark_total_return"]
    )
    assert low_cost["simulation_device"] == "cpu"


def test_cuda_simulator_matches_cpu_reference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    bars = prepare_intraday_bars(_intraday_frame())
    candidates = [
        _candidate(),
        {
            **_candidate(),
            "family": "mean_reversion",
            "entry_rule_id": "mean-reversion-test",
            "entry_threshold": 0.5,
            "take_profit_bps": 10.0,
            "stop_loss_bps": 5.0,
        },
    ]
    cpu = simulate_scalping_candidates(
        bars, candidates, cost_per_side_bps=1.0, device="cpu",
    )
    cuda = simulate_scalping_candidates(
        bars, candidates, cost_per_side_bps=1.0, device="cuda",
    )
    assert cuda["simulation_device"].eq("cuda").all()
    assert cuda["n_trades"].tolist() == cpu["n_trades"].tolist()
    numeric = [
        "sharpe", "cagr", "max_drawdown", "total_return", "win_rate",
        "profit_factor", "excess_total_return",
    ]
    np.testing.assert_allclose(
        cuda[numeric].to_numpy(), cpu[numeric].to_numpy(),
        rtol=1e-10, atol=1e-12, equal_nan=True,
    )


def test_common_session_alignment_removes_non_shared_dates():
    first = prepare_intraday_bars(_intraday_frame(sessions=10))
    second = prepare_intraday_bars(_intraday_frame(sessions=10))
    second = second[second["session_date"] != second["session_date"].iloc[-1]]
    aligned = align_common_sessions({"XLK": first, "XLE": second})
    assert aligned["XLK"]["session"].nunique() == 9
    assert set(aligned["XLK"]["session_date"]) == set(aligned["XLE"]["session_date"])
    assert aligned["XLK"].groupby("session")["last_bar"].sum().eq(1).all()


def test_entry_screens_aggregate_across_instruments():
    rows = []
    for ticker, score in (("XLK", 3.0), ("XLE", 1.0)):
        row = {**_candidate(), "ticker": ticker, "screen_trades": 40,
               "screen_edge_bps": score, "screen_win_rate": 0.55,
               "screen_t_stat": score, "screen_score": score,
               "screen_device": "cpu"}
        rows.append(row)
    aggregate = aggregate_entry_screens(pd.DataFrame(rows)).iloc[0]
    assert aggregate["screen_score"] == pytest.approx(2.0)
    assert aggregate["screen_trades"] == 80
    assert aggregate["instrument_screen_count"] == 2


def test_holdout_aggregation_reports_breadth():
    holdout = pd.DataFrame([
        {"ticker": "XLK", "cost_scenario_bps_per_side": 1.0,
         "total_return": 0.04, "sharpe": 1.0, "n_trades": 20,
         "excess_total_return": 0.01, "benchmark_total_return": 0.03,
         "benchmark_sharpe": 0.8},
        {"ticker": "XLE", "cost_scenario_bps_per_side": 1.0,
         "total_return": -0.02, "sharpe": -0.5, "n_trades": 30,
         "excess_total_return": -0.05, "benchmark_total_return": 0.03,
         "benchmark_sharpe": 0.8},
    ])
    aggregate = aggregate_instrument_holdout(holdout).iloc[0]
    assert aggregate["total_return"] == pytest.approx(0.01)
    assert aggregate["n_trades"] == 50
    assert aggregate["positive_instruments"] == 1
    assert aggregate["beat_spy_instruments"] == 1


def test_holdout_aggregation_reports_each_asset_group():
    holdout = pd.DataFrame([
        {"ticker": "NVDA", "asset_group": "individual_stock",
         "cost_scenario_bps_per_side": 1.0, "total_return": 0.04,
         "sharpe": 1.0, "n_trades": 20, "excess_total_return": 0.01,
         "benchmark_total_return": 0.03, "benchmark_sharpe": 0.8},
        {"ticker": "BTC/USD", "asset_group": "crypto",
         "cost_scenario_bps_per_side": 1.0, "total_return": -0.02,
         "sharpe": -0.5, "n_trades": 30, "excess_total_return": -0.05,
         "benchmark_total_return": 0.03, "benchmark_sharpe": 0.8},
    ])
    aggregate = aggregate_group_holdout(holdout)
    assert set(aggregate["asset_group"]) == {"crypto", "individual_stock"}
    assert aggregate.set_index("asset_group").loc["crypto", "instrument_count"] == 1
