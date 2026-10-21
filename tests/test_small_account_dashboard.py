import json
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest
from dashboard.views.small_account import load_monitor
from livesim.small_account import PaperEngine


def test_dashboard_empty_directory(tmp_path):
    report,state,equity,journal,mode,status=load_monitor(tmp_path)
    assert state['equity']==100 and journal.empty


def test_dashboard_reads_transactional_paper_journal(tmp_path):
    e=PaperEngine(database=tmp_path/'paper.sqlite',shadow=True,fee_bps=0,slip_bps=0,spread_bps=0)
    prev={'ts':'2026-01-01T00:00:00Z','signal':True,'distance':2.,'regime':'trend_up'}
    e.step({'ts':'2026-01-02T00:00:00Z','open':100,'high':101,'low':99,'close':100},prev)
    _,state,eq,journal,mode,_=load_monitor(tmp_path)
    assert state['position']['qty']==.25 and len(eq)==1
    assert 'ENTRY' in set(journal.event) and mode=='Local shadow account'


def test_streamlit_page_renders():
    app=AppTest.from_file('dashboard/small_account_app.py',default_timeout=20).run()
    assert not app.exception
    assert any('NO_TRADE' in x.value for x in app.info)
    assert len(app.metric)==8
