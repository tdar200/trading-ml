from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
from dashboard.views.fundamentals import render
st.set_page_config(page_title='Fundamental research',layout='wide')
render()
