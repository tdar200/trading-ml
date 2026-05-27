"""
Strategy Runner — Stage 9.

Form-driven launcher for ``examples/run_trading_node.py``. The user picks
exec mode (simulated / alpaca-paper) and poll cadence; we shell out and
stream stdout into the page. No subprocess is auto-spawned without an
explicit button click.

Deliberately minimal: this is a research-time convenience, not a
production scheduler. For the actual 3-month paper observation gated
by ``docs/superpowers/specs/2026-05-27-stage-8-deploy-checklist.md``
the user runs ``run_trading_node.py`` directly in a terminal.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st


_PROJECT_ROOT = Path(__file__).parent.parent.parent
_RUN_SCRIPT = _PROJECT_ROOT / "examples" / "run_trading_node.py"


def _stream_subprocess(cmd: list[str]):
    """Generator yielding the subprocess' stdout line-by-line."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(_PROJECT_ROOT),
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            yield line
    finally:
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.terminate()


def render(data_sources) -> None:
    st.header("Strategy Runner")
    st.caption(
        "Wraps `examples/run_trading_node.py`. For continuous daily "
        "paper-trade ops (Stage-8 observation), run directly in a terminal "
        "instead — the in-dashboard launch is for quick smoke checks."
    )

    cols = st.columns(3)
    with cols[0]:
        exec_mode = st.selectbox(
            "Execution mode",
            options=("simulated", "alpaca-paper"),
            index=0,
            help="alpaca-paper routes orders to your live Alpaca paper account "
                 "(real broker; market hours apply).",
        )
    with cols[1]:
        fast_replay = st.checkbox(
            "Fast replay", value=True,
            help="One-shot replay of the full catalog through the strategy. "
                 "Uncheck for continuous mode (--poll-seconds).",
        )
    with cols[2]:
        slippage = st.number_input(
            "Slippage prob",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            help="Ignored in alpaca-paper mode.",
        )

    poll_seconds = None
    if not fast_replay:
        poll_seconds = st.number_input(
            "Poll seconds (continuous mode)",
            min_value=1, value=60, step=1,
        )

    cmd = [
        "uv", "run", "python", str(_RUN_SCRIPT.relative_to(_PROJECT_ROOT)),
        "--exec-mode", exec_mode,
        "--slippage-prob", str(slippage),
    ]
    if fast_replay:
        cmd.append("--fast-replay")
    elif poll_seconds is not None:
        cmd.extend(["--poll-seconds", str(int(poll_seconds))])

    st.subheader("Launch command")
    st.code(" ".join(cmd), language="bash")

    if st.button("Run", type="primary"):
        st.subheader("Output")
        # st.write_stream renders a streaming text area as the generator yields.
        st.write_stream(_stream_subprocess(cmd))
