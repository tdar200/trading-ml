"""Accounting, causality, chronological selection and immutable replay tests."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.discovery_jobs import demo_bars, read_state, run_job
from analysis.rule_discovery import (
    Cancelled, Rule, candidate_grid, discover, load_run, replay_run,
    save_run, signals, simulate, validate_bars,
)


def bars(closes=(100, 99, 98, 105, 110, 111, 112)):
    close = np.asarray(closes, dtype=float)
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"ts": pd.date_range("2024-01-02", periods=len(close), freq="B", tz="UTC"),
                         "open": opening, "high": np.maximum(opening, close) + .1,
                         "low": np.minimum(opening, close) - .1, "close": close, "volume": 1000.0})


def rule(**kwargs):
    return replace(Rule("AAPL", "1d", "ma_cross", period=2, hold_bars=1,
                        stop_pct=0, target_pct=0), **kwargs)


def replay(frame, candidate=None, cost=0):
    return simulate(frame, candidate or rule(), frame.ts.min(), frame.ts.max() + pd.Timedelta(days=1), cost)


def test_next_open_entry_and_hand_calculated_cash_costs():
    frame = bars()
    result = replay(frame, cost=3)
    trade = result["trades"][0]
    quantity = 9500 / (105 * 1.0003)
    expected_net = quantity * (110 - 105) - quantity * (105 + 110) * .0003
    assert trade["signal_time"] == frame.ts.iloc[3].isoformat()
    assert trade["entry_time"] == frame.ts.iloc[4].isoformat()
    assert trade["entry_price"] == 105
    assert trade["exit_price"] == 110
    assert trade["exit_reason"] == "holding_period"
    assert trade["quantity"] == pytest.approx(quantity)
    assert trade["net_pnl"] == pytest.approx(expected_net)
    assert result["equity"][-1]["equity"] == pytest.approx(10000 + expected_net)
    assert all(row["cash"] >= -1e-8 for row in result["equity"])
    assert result["equity"][-1]["quantity"] == 0


def test_ambiguous_intrabar_exit_is_stop_first():
    frame = bars()
    frame.loc[4, ["high", "low"]] = [120, 80]
    trade = replay(frame, rule(stop_pct=.1, target_pct=.1))["trades"][0]
    assert trade["exit_reason"] == "stop"
    assert trade["exit_price"] == pytest.approx(94.5)


def test_stop_gap_fills_at_worse_open():
    frame = bars()
    frame.loc[5, ["open", "low"]] = [80, 79]
    trade = replay(frame, rule(hold_bars=4, stop_pct=.1))["trades"][0]
    assert trade["exit_reason"] == "stop_gap"
    assert trade["exit_price"] == 80


def test_final_liquidation_reconciles_equity():
    result = replay(bars(), rule(hold_bars=20), cost=5)
    assert result["trades"][-1]["exit_reason"] == "evaluation_end"
    assert result["equity"][-1]["equity"] == pytest.approx(10000 + sum(t["net_pnl"] for t in result["trades"]))


@pytest.mark.parametrize("trigger", ["rsi_cross", "ma_cross", "breakout"])
def test_future_changes_cannot_change_prior_signals_or_replay(trigger):
    frame = demo_bars("AAPL", "1h")
    changed = frame.copy()
    changed.loc[400:, ["open", "high", "low", "close"]] *= 7
    candidate = Rule("AAPL", "1h", trigger, relative_volume=1, trend_period=14)
    pd.testing.assert_series_equal(signals(frame, candidate).iloc[:400], signals(changed, candidate).iloc[:400])
    assert simulate(frame, candidate, frame.ts.iloc[100], frame.ts.iloc[400]) == simulate(
        changed, candidate, frame.ts.iloc[100], frame.ts.iloc[400])


def test_flat_prices_produce_no_trades_and_finite_metrics():
    result = replay(bars([100] * 90))
    assert result["trades"] == []
    assert result["metrics"]["sharpe"] == result["metrics"]["total_return"] == 0


def test_rsi_cross_and_breakout_have_explicit_trigger_semantics():
    rsi = signals(bars([100, 98, 96, 95, 98, 99]), rule(trigger="rsi_cross"))
    assert list(np.flatnonzero(rsi)) == [4]
    breakout = signals(bars(), rule(trigger="breakout"))
    assert list(np.flatnonzero(breakout)) == [3, 4, 5, 6]


def test_positive_outer_selection_opens_only_frozen_final_candidate(tmp_path):
    data = {("AAPL", "1h"): demo_bars("AAPL", "1h")}
    rules = [Rule("AAPL", "1h", trigger) for trigger in ("rsi_cross", "ma_cross")]
    report = discover(data, rules, min_trades=1)
    assert report["holdout_opened"] is True
    assert report["evaluation"] is not None
    assert report["deployment_approved"] is False
    assert all(t["entry_time"] >= report["holdout_start"] for t in report["evaluation"]["trades"])
    save_run(tmp_path, report, data, rules)
    assert replay_run(tmp_path) == report["evaluation"]


@pytest.mark.parametrize("damage", ["duplicate", "nan", "negative", "bounds"])
def test_bad_market_data_fails_preflight(damage):
    frame = bars()
    if damage == "duplicate": frame.loc[1, "ts"] = frame.ts.iloc[0]
    if damage == "nan": frame.loc[1, "close"] = np.nan
    if damage == "negative": frame.loc[1, "open"] = -1
    if damage == "bounds": frame.loc[1, "high"] = 1
    with pytest.raises(ValueError): validate_bars(frame)


def test_missing_bar_uses_next_observed_open_without_filling_data():
    frame = bars().drop(index=4).reset_index(drop=True)
    result = replay(frame)
    assert result["trades"][0]["entry_time"] == frame.ts.iloc[4].isoformat()
    assert result["trades"][0]["entry_price"] == 110


def test_grid_searches_instrument_timeframe_and_indicator_roles():
    rules = candidate_grid(["AAPL", "MSFT"], ["1h", "1d"], ["rsi_cross", "ma_cross"],
                           (7,), (0, 20), (0, 1), (4,))
    assert len(rules) == 32
    assert len({r.key for r in rules}) == 32
    with pytest.raises(ValueError): candidate_grid([], ["1h"], ["ma_cross"])


def test_no_activity_rejects_without_opening_holdout(tmp_path):
    data = {("AAPL", "1d"): bars([100] * 100)}
    report = discover(data, [rule()])
    assert report["verdict"] == "REJECT"
    assert report["holdout_opened"] is False
    assert report["evaluation"] is None
    save_run(tmp_path, report, data, [rule()])
    assert replay_run(tmp_path) == report["preview"]


def test_later_data_cannot_change_selection_and_folds_are_chronological():
    frame = demo_bars("AAPL", "1h")
    rules = [Rule("AAPL", "1h", trigger) for trigger in ("ma_cross", "rsi_cross")]
    baseline = discover({("AAPL", "1h"): frame}, rules, min_trades=1)
    changed = frame.copy()
    changed.loc[changed.ts >= baseline["holdout_start"], ["open", "high", "low", "close"]] *= 2
    result = discover({("AAPL", "1h"): changed}, rules, min_trades=1)
    assert result["rankings"] == baseline["rankings"]
    assert result["folds"] == baseline["folds"]
    assert result["candidate_id"] == baseline["candidate_id"]
    for fold in result["folds"]:
        assert fold["train_end"] == fold["test_start"] < fold["test_end"] <= result["holdout_start"]


def test_snapshot_replay_ignores_live_data_and_detects_tampering(tmp_path):
    frame = demo_bars("AAPL", "1d")
    rules = [rule()]
    report = discover({("AAPL", "1d"): frame}, rules, cost_bps=3, min_trades=1)
    save_run(tmp_path, report, {("AAPL", "1d"): frame}, rules)
    frame.loc[:, "close"] = 1.0
    assert replay_run(tmp_path) == (report["evaluation"] or report["preview"])
    snapshot = tmp_path / "bars_0.parquet"
    with snapshot.open("ab") as handle: handle.write(b"changed")
    with pytest.raises(ValueError, match="fingerprint"): replay_run(tmp_path)


def test_report_tampering_rejected(tmp_path):
    data = {("AAPL", "1d"): bars([100] * 100)}
    save_run(tmp_path, discover(data, [rule()]), data, [rule()])
    with (tmp_path / "report.json").open("a") as handle: handle.write(" ")
    with pytest.raises(ValueError, match="fingerprint"): load_run(tmp_path)


def test_cancellation_and_insufficient_data():
    with pytest.raises(Cancelled):
        discover({("AAPL", "1d"): bars([100] * 100)}, [rule()], cancelled=lambda: True)
    with pytest.raises(ValueError, match="80 common"):
        discover({("AAPL", "1d"): bars()}, [rule()])
    with pytest.raises(ValueError, match="Missing data"):
        discover({}, [rule()])


def test_failed_and_cancelled_jobs_never_publish_report(tmp_path):
    run_job(tmp_path, [rule(timeframe="1h")], 3, "Local market data", tmp_path / "missing", 3)
    assert read_state(tmp_path)["status"] == "failed"
    assert not (tmp_path / "report.json").exists()
    (tmp_path / "cancel").touch()
    run_job(tmp_path, [rule()], 3, "Synthetic demo", tmp_path, 3)
    assert read_state(tmp_path)["status"] == "cancelled"
    assert not (tmp_path / "report.json").exists()
