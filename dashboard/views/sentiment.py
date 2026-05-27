"""Sentiment tab: Stage-6 ablation verdict + per-fold ΔIC + per-ticker timeseries."""
from __future__ import annotations

import streamlit as st


_UNIVERSE = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
             "XLP", "XLU", "XLB", "XLRE", "XLC"]


def render(data_sources) -> None:
    st.header("Sentiment Ablation (Stage 6)")
    a = data_sources.load_latest_ablation()
    if a is None:
        st.info("No ablation run yet. Run `uv run python examples/ablate_sentiment.py`.")
        return
    cols = st.columns(3)
    cols[0].metric("Median ΔIC", f"{a.get('median_delta_ic', float('nan')):+.4f}")
    cols[1].metric("Permutation p", f"{a.get('p_value', float('nan')):.4f}")
    verdict_full = str(a.get("verdict", ""))
    verdict_label = verdict_full.split(" — ")[0] if verdict_full else "—"
    cols[2].metric("Verdict", verdict_label)
    if verdict_full:
        st.caption(verdict_full)

    per_fold = a.get("per_fold")
    if per_fold is not None and len(per_fold) > 0:
        st.subheader("Per-fold ΔIC")
        st.bar_chart(per_fold.set_index("cutoff")["delta_ic"])
        st.subheader("Per-fold IC table")
        st.dataframe(per_fold)

    st.subheader("Per-ticker sentiment (last 180 days)")
    ticker = st.selectbox("Ticker", _UNIVERSE)
    news = data_sources.load_news_for_ticker(ticker, window_days=180)
    if news.empty:
        st.info(f"No news for {ticker} in last 180 days.")
    else:
        st.write(f"{len(news)} articles")
        st.dataframe(news[["created_at", "headline", "score"]].head(50))
