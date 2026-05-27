"""Harness Verdict tab: Stage-3 winner + Stage-5 baseline-vs-meta cards."""
from __future__ import annotations

import streamlit as st


def _meta_card(payload: dict, label: str) -> None:
    st.markdown(f"**{label}**")
    bm = payload.get("baseline_median_sharpe")
    mw = (payload.get("meta_winner") or {}).get("median_window_sharpe")
    if bm is None or mw is None:
        st.json(payload)
        return
    delta = ((mw - bm) / bm) * 100 if bm else float("nan")
    cols = st.columns(2)
    cols[0].metric("Baseline Sharpe", f"{bm:.4f}")
    cols[1].metric("Meta winner Sharpe", f"{mw:.4f}", delta=f"{delta:+.1f}%")
    dsr = payload.get("dsr") or {}
    if dsr:
        st.caption(
            f"DSR n_trials={dsr.get('n_trials')}, "
            f"z={dsr.get('deflated_sharpe', float('nan')):.3f}, "
            f"p={dsr.get('p_value', float('nan')):.4f}"
        )


def render(data_sources) -> None:
    st.header("Harness Verdicts")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stage 3 — Walk-forward winner")
        s3 = data_sources.load_latest_harness()
        if s3 is None:
            st.info("No Stage-3 harness yet. Run `uv run python examples/run_harness.py`.")
        else:
            st.json(s3)
    with col2:
        st.subheader("Stage 5 — Baseline vs Meta")
        v1 = data_sources.load_latest_meta_harness("v1")
        v2 = data_sources.load_latest_meta_harness("v2")
        if v1 is None and v2 is None:
            st.info(
                "No Stage-5/6 meta harness yet. Run "
                "`uv run python examples/run_harness_meta.py`."
            )
        else:
            if v1 is not None:
                _meta_card(v1, "v1 (price-only)")
            if v2 is not None:
                _meta_card(v2, "v2 (price + sentiment)")
