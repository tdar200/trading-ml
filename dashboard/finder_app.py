"""Standalone entrypoint: streamlit run dashboard/finder_app.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from dashboard.views.rule_discovery import render

st.set_page_config(page_title="Rule Strategy Finder", layout="wide")
render()
