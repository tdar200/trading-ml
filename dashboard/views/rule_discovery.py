"""Full research workflow for instrument/timeframe/entry-rule discovery."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis.discovery_jobs import ROOT, job_alive, read_state, start_job
from analysis.rule_discovery import TIMEFRAMES, TRIGGERS, Rule, candidate_grid, load_run, replay_run
from data.tickers import SCALPING_STOCKS


def results_root():
    return Path(os.environ.get("TRADING_DISCOVERY_RESULTS", str(ROOT / "results" / "rule_discovery")))


def _integers(text):
    return tuple(dict.fromkeys(int(value.strip()) for value in text.split(",")))


def render(_sources=None):
    st.subheader("Rule Strategy Finder")
    st.write("Search instruments, timeframes and indicator rules; validate chronologically and replay exact saved trades.")
    st.caption("Each candidate trades one instrument, long-only, with one position. Instruments compete inside each training window.")
    source = st.selectbox("Data source", ["Local market data", "Synthetic demo"], key="rd_source")
    if source == "Synthetic demo":
        st.warning("Synthetic demo — exercises the software only. Results are not market evidence.")
    columns = st.columns(3)
    instruments = columns[0].multiselect("Instruments", SCALPING_STOCKS, default=["AAPL"], key="rd_instruments")
    timeframes = columns[1].multiselect("Timeframes", TIMEFRAMES, default=["1h"], key="rd_timeframes")
    triggers = columns[2].multiselect("Entry triggers", TRIGGERS, default=["rsi_cross", "ma_cross"], key="rd_triggers")
    columns = st.columns(3)
    periods_text = columns[0].text_input("Indicator periods (comma-separated)", "7,14", key="rd_periods")
    trends_text = columns[1].text_input("Trend MA periods (0 disables)", "0,50", key="rd_trends")
    holds_text = columns[2].text_input("Holding periods in bars", "4,8", key="rd_holds")
    columns = st.columns(3)
    volume = columns[0].checkbox("Also search relative volume ≥ 1", value=False, key="rd_volume")
    stop = columns[1].number_input("Stop loss (%)", min_value=0.0, max_value=50.0, value=2.0, key="rd_stop")
    target = columns[2].number_input("Profit target (%)", min_value=0.0, max_value=50.0, value=4.0, key="rd_target")
    columns = st.columns(3)
    cost = columns[0].number_input("Execution cost per side (bps)", min_value=0.0, max_value=100.0, value=2.5, step=0.5, key="rd_cost")
    allocation = columns[1].number_input("Maximum allocation (%)", min_value=1.0, max_value=100.0, value=95.0, key="rd_allocation")
    min_trades = columns[2].number_input("Minimum trades per validation window", min_value=1, max_value=1000, value=3, key="rd_min_trades")
    intraday_root = st.text_input("Local intraday data directory", str(ROOT / "data" / "free_intraday"), key="rd_data_root")
    st.caption("Local data only; no downloads. Daily bars use data/catalog. At least 80 common sessions are required. Final 20% reserved for evaluation.")
    st.caption("The final period is reserved within this run. Previously examined history is not fresh out-of-sample evidence.")
    rules = []
    try:
        rules = candidate_grid(instruments, timeframes, triggers, _integers(periods_text),
                               _integers(trends_text), (0.0, 1.0) if volume else (0.0,),
                               _integers(holds_text), stop_pct=stop / 100, target_pct=target / 100,
                               allocation=allocation / 100)
        st.caption(f"{len(rules)} candidates · 3 chronological outer windows · costs included in every simulation")
    except ValueError as exc:
        st.warning(f"Invalid search: {exc}")
    if st.button("Start discovery", type="primary", disabled=not rules, key="rd_start"):
        try:
            run = start_job(results_root(), rules, cost, source, Path(intraday_root), int(min_trades))
            st.session_state["rd_selected_run"] = run.name
        except ValueError as exc:
            st.error(str(exc))
    _runs_panel()


@st.fragment(run_every="1s")
def _runs_panel():
    root = results_root()
    runs = sorted((p.name for p in root.glob("*") if (p / "state.json").exists()), reverse=True)
    if not runs:
        st.info("No discovery runs yet.")
        return
    if st.session_state.get("rd_selected_run") not in runs:
        st.session_state["rd_selected_run"] = runs[0]
    selected = st.selectbox("Saved run", runs, key="rd_selected_run")
    directory = root / selected
    state = read_state(directory)
    st.caption(f"Run: {selected}")
    if state["status"] == "running":
        if not job_alive(directory):
            st.error("Search interrupted by a server restart. Start a new run; no completed result is available.")
            return
        st.progress(state["percent"], text=state["message"])
        if st.button("Cancel search", key=f"cancel_{selected}"):
            (directory / "cancel").touch()
            st.info("Cancellation requested")
        return
    if state["status"] != "complete":
        st.error(state["message"])
        return
    try:
        report = load_run(directory)
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return
    st.success("Search complete")
    st.write(f"Verdict: {report['verdict']}")
    st.write(report["reason"])
    st.caption(f"Source: {report['data_source']} · Final period {'opened' if report['holdout_opened'] else 'sealed'} · Deployment not approved")
    st.write(Rule(**report["candidate"]).description)
    st.subheader("Chronological selection evidence")
    st.dataframe(pd.DataFrame(report["folds"]), hide_index=True)
    st.caption("Each row evaluates a candidate selected only on earlier data. This evaluates the selection process; fold winners may differ.")
    st.subheader("Development rankings")
    st.dataframe(pd.DataFrame(report["rankings"]), hide_index=True)
    evidence = report["evaluation"] if report["holdout_opened"] else report["preview"]
    st.subheader("Final evaluation" if report["holdout_opened"] else "Development replay — final period sealed")
    m = evidence["metrics"]
    columns = st.columns(4)
    columns[0].metric("Net return", f"{m['total_return']:.2%}")
    columns[1].metric("Sharpe", f"{m['sharpe']:.2f}")
    columns[2].metric("Maximum drawdown", f"{m['max_drawdown']:.2%}")
    columns[3].metric("Completed trades", m["trades"])
    st.caption(f"Gross P&L USD {m['gross_pnl']:.2f} · execution costs USD {m['costs']:.2f} · initial cash USD 10,000")
    curve = pd.DataFrame(evidence["equity"])
    curve["ts"] = pd.to_datetime(curve.ts, utc=True)
    st.line_chart(curve.set_index("ts")[["equity"]])
    st.subheader("Exact trade ledger")
    st.dataframe(pd.DataFrame(evidence["trades"]).drop(columns=["entry_index"], errors="ignore"), hide_index=True)
    st.caption("Timestamps label bars. Intrabar exits identify the exit bar, not an inferred exact fill time. Stop-first when both stop and target are touched.")
    if st.button("Replay saved candidate", key=f"replay_{selected}"):
        try:
            replay_run(directory)
            st.session_state[f"verified_{selected}"] = True
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    if st.session_state.get(f"verified_{selected}"):
        st.success("Replay verified: trades, costs and equity exactly match saved evidence")
    st.download_button("Download run JSON", json.dumps(report, indent=2), f"{selected}.json", "application/json", key=f"export_{selected}")
    st.download_button("Download trades CSV", pd.DataFrame(evidence["trades"]).to_csv(index=False), f"{selected}_trades.csv", "text/csv", key=f"trades_{selected}")
