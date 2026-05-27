"""Equity tab: equity curves + rolling Sharpe + drawdown."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def _rolling_sharpe(returns: pd.Series, window: int = 60) -> pd.Series:
    return (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)


def _drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def render(data_sources) -> None:
    st.header("Equity")
    curves = data_sources.load_equity_curves()
    if not curves:
        st.info("No equity curves yet. Run `uv run python examples/backtest_momentum.py`.")
        return
    df = pd.DataFrame(curves).sort_index()
    st.subheader("Equity curves")
    st.line_chart(df)
    rets = df.pct_change().dropna()
    if not rets.empty:
        st.subheader("Rolling 60-day Sharpe")
        st.line_chart(rets.apply(_rolling_sharpe))
        st.subheader("Drawdown")
        st.line_chart(df.apply(_drawdown))
