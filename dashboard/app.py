"""Stage-7 Streamlit dashboard entry.

Usage:
    uv run streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from dashboard import data_sources as ds
from dashboard.views import (
    backtest_lab,
    equity,
    harness_verdict,
    journal,
    models,
    sentiment,
    strategy_runner,
)


# Wrap every loader with Streamlit's cache so the dashboard isn't
# re-reading parquet/json on every interaction. TTL 5 min — refresh by
# hitting "Rerun" in the Streamlit UI when a fresh acceptance run
# completes.
class _CachedDataSources:
    load_latest_harness = staticmethod(st.cache_data(ttl=300)(ds.load_latest_harness))
    load_latest_meta_harness = staticmethod(st.cache_data(ttl=300)(ds.load_latest_meta_harness))
    load_latest_ablation = staticmethod(st.cache_data(ttl=300)(ds.load_latest_ablation))
    load_equity_curves = staticmethod(st.cache_data(ttl=300)(ds.load_equity_curves))
    load_news_for_ticker = staticmethod(st.cache_data(ttl=300)(ds.load_news_for_ticker))
    load_model_coverage = staticmethod(st.cache_data(ttl=300)(ds.load_model_coverage))
    load_feature_importances = staticmethod(st.cache_data(ttl=300)(ds.load_feature_importances))
    load_journal_entries = staticmethod(st.cache_data(ttl=300)(ds.load_journal_entries))


def main() -> None:
    st.set_page_config(page_title="trading-ml", layout="wide")
    st.title("trading-ml — Research Dashboard")
    sources = _CachedDataSources

    tabs = st.tabs([
        "Equity",
        "Backtest Lab",
        "Strategy Runner",
        "Harness Verdicts",
        "Sentiment",
        "Models",
        "Journal",
    ])
    with tabs[0]:
        equity.render(sources)
    with tabs[1]:
        backtest_lab.render(sources)
    with tabs[2]:
        strategy_runner.render(sources)
    with tabs[3]:
        harness_verdict.render(sources)
    with tabs[4]:
        sentiment.render(sources)
    with tabs[5]:
        models.render(sources)
    with tabs[6]:
        journal.render(sources)


if __name__ == "__main__":
    main()
