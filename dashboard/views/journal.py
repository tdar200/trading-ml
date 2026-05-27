"""Journal tab: render all spec/plan/summary markdown, newest first."""
from __future__ import annotations

import streamlit as st


def render(data_sources) -> None:
    st.header("Research Journal")
    entries = data_sources.load_journal_entries()
    if not entries:
        st.info("No journal entries found.")
        return
    st.write(f"{len(entries)} entries, newest first")
    options = [p.name for p in entries]
    selected = st.selectbox("Document", options)
    chosen = next(p for p in entries if p.name == selected)
    st.markdown(chosen.read_text())
