from datetime import datetime,timezone
import fcntl,json
from pathlib import Path
import pytest
from dashboard.sweep_status import snapshot


def test_progress_excludes_failures_from_backtests(tmp_path):
    s=snapshot(tmp_path,{'states':{'PENDING':60,'READY':10,'GPU_TESTED':20,'EXCLUDED':7,'FETCH_ERROR':3}})
    assert s['attempted_fraction']==.4
    assert s['tested_fraction']==.2
    assert s['pending']==60 and s['ready']==10
    assert s['state']=='Worker not running'
    assert snapshot(tmp_path,{})['attempted_fraction']==0


def test_paused_reason_and_worker_lock(tmp_path):
    (tmp_path/'STOP').write_text('Provider rate limit: yahoo')
    state={'updated_at':'2026-09-30T12:00:00+00:00'}
    now=datetime(2026,9,30,12,5,tzinfo=timezone.utc)
    result=snapshot(tmp_path,state,now)
    assert result['state']=='Paused' and result['age_seconds']==300
    assert result['reason']=='Provider rate limit: yahoo'
    with (tmp_path/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert snapshot(tmp_path,state,now)['state']=='Stopping after current batch'
        (tmp_path/'STOP').unlink()
        assert snapshot(tmp_path,state,now)['state']=='Running — update overdue'


def test_dashboard_displays_exact_progress_and_pause(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    (tmp_path/'coverage.json').write_text(json.dumps({'instruments':100,'coverage':[]}))
    (tmp_path/'status.json').write_text(json.dumps({'states':{'PENDING':60,'READY':10,'GPU_TESTED':20,'EXCLUDED':7,'FETCH_ERROR':3},'histories':100,'stage':'STOPPED'}))
    (tmp_path/'STOP').write_text('Provider rate limit: yahoo')
    monkeypatch.setenv('TRADING_SWEEP_ROOT',str(tmp_path))
    app=AppTest.from_file(str(Path('dashboard/market_sweep_app.py').resolve())).run(timeout=20)
    assert not app.exception
    assert 'Paused · Provider rate limit: yahoo' in app.warning[0].value
    assert app.get('progress')[0].proto.text=='History requests attempted: 40 / 100 (40.00%)'
    assert app.get('progress')[1].proto.text=='Histories GPU-backtested: 20 / 100 (20.00%)'
    assert {m.label:m.value for m in app.metric}['Waiting for data']=='60'
