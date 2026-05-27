"""Models tab: cache coverage maps + per-combo feature importances."""
from __future__ import annotations

import streamlit as st

from dashboard.data_sources import DEFAULT_CACHE_V1, DEFAULT_CACHE_V2


def render(data_sources) -> None:
    st.header("Model Cache Coverage")
    for label, cache in (("v1 (price-only)", DEFAULT_CACHE_V1),
                         ("v2 (price + sentiment)", DEFAULT_CACHE_V2)):
        st.subheader(label)
        df = data_sources.load_model_coverage(cache_dir=cache)
        if df.empty:
            st.info(f"No models in `{cache}`.")
            continue
        st.write(
            f"{len(df)} models across {df['combo'].nunique()} combos × "
            f"{df['cutoff'].nunique()} cutoffs"
        )
        pivot = df.assign(present=1).pivot_table(
            index="combo", columns="cutoff", values="present", fill_value=0
        )
        st.dataframe(pivot)
        # Feature importances for the latest cutoff of the first combo.
        latest_cutoff = df["cutoff"].max()
        first_combo = sorted(df["combo"].unique())[0]
        try:
            imp = data_sources.load_feature_importances(
                cache_dir=cache, combo=first_combo, cutoff=latest_cutoff,
            )
        except Exception as exc:
            st.warning(f"Could not load feature importances: {exc}")
            imp = None
        if imp is not None:
            st.markdown(
                f"**Feature importances for `{first_combo}` @ {latest_cutoff.date()}**"
            )
            st.bar_chart(imp)
