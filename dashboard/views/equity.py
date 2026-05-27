"""
Equity tab — Stage 9 rewrite.

Renders TradingView-quality candlestick charts via
``streamlit-lightweight-charts`` (a Streamlit-native component wrapping
TradingView's open-source Lightweight Charts JS library, Apache 2.0).
Pulls multi-timeframe OHLCV via ``data.multitf.get_bars``: 1m / 5m / 15m /
30m / 1h / 4h come from ``data/intraday/`` (Stage-9 ingest), 1d from the
existing Stage-1 catalog, and 1w / 1M / 1Y are resampled from 1d.

Falls back to the legacy backtest-equity overlay (Stage-2 equity vs SPY)
when the user hasn't run the intraday ingest yet.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts

from data.multitf import (
    INTRADAY_TIMEFRAMES,
    SUPPORTED_TIMEFRAMES,
    get_bars,
)
from data.tickers import ALL_TICKERS, BENCHMARK


# TradingView-style timeframe lineup. Ordered shortest → longest.
_TIMEFRAME_OPTIONS: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "4h",
                                        "1d", "1w", "1M", "1Y")

# Default lookback per timeframe — coarser bars deserve more history.
_DEFAULT_LOOKBACK_DAYS: dict[str, int] = {
    "1m": 5, "5m": 7, "15m": 14, "30m": 30, "1h": 60, "4h": 90,
    "1d": 730, "1w": 365 * 5, "1M": 365 * 10, "1Y": 365 * 20,
}


def _df_to_lwc_candles(df: pd.DataFrame) -> list[dict]:
    """Convert OHLCV DataFrame → list-of-dicts in LWC candlestick shape."""
    return [
        {
            "time": int(row["ts"].timestamp()),  # UNIX seconds
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for row in df.to_dict("records")
    ]


def _df_to_lwc_volume(df: pd.DataFrame) -> list[dict]:
    """OHLCV → list-of-dicts in LWC histogram shape (color by direction)."""
    out = []
    for _, row in df.iterrows():
        color = "rgba(38, 166, 154, 0.5)" if row["close"] >= row["open"] else "rgba(239, 83, 80, 0.5)"
        out.append({
            "time": int(row["ts"].timestamp()),
            "value": float(row["volume"]),
            "color": color,
        })
    return out


def _render_tv_chart(df: pd.DataFrame, ticker: str, timeframe: str) -> None:
    """Render a candlestick + volume LWC chart for the given OHLCV frame."""
    candles = _df_to_lwc_candles(df)
    volume = _df_to_lwc_volume(df)
    charts = [
        {
            "chart": {
                "height": 500,
                "layout": {
                    "background": {"type": "solid", "color": "#1e1e1e"},
                    "textColor": "#d1d4dc",
                },
                "grid": {
                    "vertLines": {"color": "#363c4e"},
                    "horzLines": {"color": "#363c4e"},
                },
                "timeScale": {"timeVisible": True, "secondsVisible": False},
                "watermark": {
                    "visible": True,
                    "fontSize": 36,
                    "horzAlign": "center",
                    "vertAlign": "center",
                    "color": "rgba(180, 180, 200, 0.15)",
                    "text": f"{ticker}  {timeframe}",
                },
            },
            "series": [
                {
                    "type": "Candlestick",
                    "data": candles,
                    "options": {
                        "upColor": "#26a69a",
                        "downColor": "#ef5350",
                        "wickUpColor": "#26a69a",
                        "wickDownColor": "#ef5350",
                        "borderVisible": False,
                    },
                },
                {
                    "type": "Histogram",
                    "data": volume,
                    "options": {
                        "color": "#26a69a",
                        "priceFormat": {"type": "volume"},
                        "priceScaleId": "volume",
                        "scaleMargins": {"top": 0.8, "bottom": 0.0},
                    },
                },
            ],
        }
    ]
    renderLightweightCharts(charts, key=f"chart_{ticker}_{timeframe}")


def _legacy_equity_overlay(data_sources) -> None:
    """Stage-7 legacy view: Stage-2 momentum equity curve vs SPY (normalized).

    Used when no intraday data exists yet (e.g., fresh clone) — keeps the
    Equity tab useful without requiring the user to first run the
    Stage-9 intraday ingest.
    """
    st.subheader("Stage-2 momentum equity vs SPY (legacy view)")
    curves = data_sources.load_equity_curves()
    if not curves:
        st.info(
            "No equity curves yet. Run `uv run python examples/backtest_momentum.py` "
            "or `uv run python data/ingest_alpaca_intraday.py` to populate this tab."
        )
        return
    df = pd.DataFrame(curves).sort_index()
    st.line_chart(df)
    rets = df.pct_change().dropna()
    if not rets.empty:
        st.subheader("Rolling 60-day Sharpe")
        st.line_chart(
            (rets.rolling(60).mean() / rets.rolling(60).std()) * np.sqrt(252)
        )
        st.subheader("Drawdown")
        st.line_chart(df.apply(lambda s: s / s.cummax() - 1.0))


def render(data_sources) -> None:
    st.header("Equity")

    # --- selector row ---
    col_ticker, col_tf, col_lookback = st.columns([2, 3, 2])
    with col_ticker:
        ticker = st.selectbox("Ticker", list(ALL_TICKERS), index=list(ALL_TICKERS).index(BENCHMARK))
    with col_tf:
        timeframe = st.segmented_control(
            "Timeframe",
            options=_TIMEFRAME_OPTIONS,
            default="1d",
            selection_mode="single",
        )
        if timeframe is None:
            timeframe = "1d"
    with col_lookback:
        lookback = st.number_input(
            "Lookback (days)",
            min_value=1,
            value=_DEFAULT_LOOKBACK_DAYS.get(timeframe, 730),
            step=1,
        )

    # --- fetch + render ---
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=int(lookback))
    df = get_bars(ticker, timeframe, start=start, end=end)
    if df.empty:
        if timeframe in INTRADAY_TIMEFRAMES:
            st.warning(
                f"No {timeframe} bars for {ticker} in the last {lookback} days. "
                f"Run `uv run python data/ingest_alpaca_intraday.py --timeframe {timeframe}` "
                "to populate the intraday cache (Alpaca free-tier IEX feed)."
            )
        else:
            st.warning(
                f"No {timeframe} bars for {ticker}. "
                "Run `uv run python data/ingest_alpaca.py` to populate the daily catalog."
            )
        # Fall back to the legacy equity overlay so the tab still has content.
        st.divider()
        _legacy_equity_overlay(data_sources)
        return

    st.caption(f"{len(df)} bars · {df['ts'].min().date()} → {df['ts'].max().date()}")
    _render_tv_chart(df, ticker, timeframe)

    # Keep the legacy equity overlay below as a "long context" view —
    # useful for cross-checking against the deployed strategy's performance.
    with st.expander("Stage-2 momentum equity vs SPY (legacy overlay)", expanded=False):
        _legacy_equity_overlay(data_sources)
