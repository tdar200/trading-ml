from dataclasses import replace
import json
import sqlite3
import numpy as np
import pandas as pd
import pytest
from livesim.small_account_risk import RiskEngine,RiskLimits,RiskSnapshot,sizing_comparison
from livesim.small_account import PaperEngine
from strategies.small_account import Setup,features
from analysis.small_account_mc import requirement,binary_paths,exact_terminal_probability,block_paths
from analysis.small_account_research import metrics,replay


def snapshot(**kwargs):
    return replace(RiskSnapshot(100,100,100,98,0,0,0,research_shadow=True),**kwargs)


def bar(day=1,open=100,high=101,low=99,close=100):
    return {'ts':pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(days=day),
            'open':open,'high':high,'low':low,'close':close}


def signal(day=0):
    return {**bar(day),'signal':True,'distance':2.,'regime':'trend_up'}


@pytest.mark.parametrize('kwargs',[
    {'equity':float('nan')},{'price':float('inf')},{'cash':-1},{'cash':101},
    {'stop':101},{'fee_rate':-1},{'spread_bps':21},{'slippage_bps':21},
    {'data_age_seconds':100},{'data_age_seconds':-10},{'clock_skew_seconds':6},
    {'healthy':False},{'consistent':False},{'database_ok':False},{'order_status_known':False},
    {'research_shadow':False},{'edge_degraded':True},{'duplicate':True},{'kill':True},
    {'daily_return':-.02},{'weekly_return':-.04},{'drawdown':.10},{'positions':1},
    {'gross_exposure':.25},{'correlated_exposure':.25}])
def test_risk_fails_closed(kwargs):
    qty,why=RiskEngine().approve(snapshot(**kwargs)); assert qty==0 and 'NO_TRADE' in why


def test_risk_budgets_costs_cash_and_minimum():
    q,_=RiskEngine().approve(snapshot(fee_rate=.008,spread_bps=10,slippage_bps=5))
    assert 0<q<.25
    loss=q*(100*1.008-98*(1-.001)*.992)
    assert loss<=.5 and q*100<=25
    assert RiskEngine(RiskLimits(min_notional=30)).approve(snapshot())[0]==0


def test_live_and_kill_cannot_be_overridden(monkeypatch):
    monkeypatch.setenv('TRADING_MODE','LIVE'); monkeypatch.setenv('LIVE_TRADING_CONFIRMED','true')
    assert RiskEngine().approve(snapshot())[0]==0
    monkeypatch.setenv('TRADING_MODE','PAPER'); monkeypatch.setenv('STOP_TRADING','true')
    assert RiskEngine().approve(snapshot())[0]==0


def test_no_martingale():
    a=RiskEngine().approve(snapshot())[0]; b=RiskEngine().approve(snapshot(losing_streak=3))[0]
    assert b<=a/2


def test_hand_calculated_accounting_and_ambiguous_stop():
    e=PaperEngine(fee_bps=0,spread_bps=0,slip_bps=0,shadow=True)
    e.step(bar(),signal()); assert e.state['cash']==75 and e.state['position']['qty']==.25
    e.step(bar(2,100,105,97,100))
    assert e.state['equity']==99.5
    assert [x for x in e.journal if x['event']=='EXIT'][0]['reason']=='stop'


def test_gap_not_guaranteed_stop_and_fees():
    e=PaperEngine(fee_bps=10,spread_bps=0,slip_bps=0,shadow=True)
    e.step(bar(),signal()); cost=e.state['position']['cost']; qty=e.state['position']['qty']
    e.step(bar(2,90,91,89,90))
    expected=100-cost+qty*90*.999
    assert e.state['equity']==pytest.approx(expected)
    assert e.journal[-1]['reason']=='gap_stop'


def test_entry_bar_stop_and_target():
    e=PaperEngine(fee_bps=0,spread_bps=0,slip_bps=0,shadow=True)
    e.step(bar(1,100,105,97,100),signal())
    assert e.state['equity']==99.5
    assert e.journal[-1]['reason']=='entry_bar_stop'


def test_default_no_trade():
    e=PaperEngine(); e.step(bar(),signal())
    assert e.state['position'] is None and e.state['equity']==100


def test_duplicate_restart_and_atomic_database(tmp_path):
    db=tmp_path/'paper.sqlite'; kwargs=dict(fee_bps=0,spread_bps=0,slip_bps=0,shadow=True,database=db)
    e=PaperEngine(**kwargs); e.step(bar(),signal()); state=e.state.copy()
    e2=PaperEngine(**kwargs); assert e2.state==state
    assert e2.step(bar(),signal())=='DUPLICATE_OR_OLD'
    e2.step(bar(2,100,105,99,104)); assert e2.state['equity']==101
    assert sqlite3.connect(db).execute('select count(*) from equity').fetchone()[0]==2
    with pytest.raises(RuntimeError,match='concurrent'): e.step(bar(3))
    assert e.state==state


def test_db_failure_rolls_back_without_order(tmp_path):
    e=PaperEngine(shadow=True,database=tmp_path/'a.db'); before=json.dumps(e.state)
    e.db.execute('DROP TABLE events')
    with pytest.raises(sqlite3.OperationalError): e.step(bar(),signal())
    assert json.dumps(e.state)==before and not e.journal


@pytest.mark.parametrize('health', [{'healthy':False},{'consistent':False},{'status_known':False},{'age_seconds':200},{'clock_skew':10}])
def test_failed_connection_unknown_partial_or_rejected_state_no_entry(health):
    e=PaperEngine(shadow=True); e.step(bar(),signal(),**health); assert e.state['position'] is None


def test_nonfinite_bar_no_mutation():
    e=PaperEngine(shadow=True); b=bar(); b['close']=float('nan')
    with pytest.raises(ValueError): e.step(b,signal())
    assert e.state['last_ts'] is None


def fixture_frame(n=200):
    x=np.arange(n); close=100+np.sin(x/6)*6+x*.06
    return pd.DataFrame({'ts':pd.date_range('2024-01-01',periods=n,tz='UTC'),
                         'open':close,'high':close+2,'low':close-2,'close':close,'volume':1000.})


@pytest.mark.parametrize('family',['donchian','ma','rsi','bollinger','volume','gap'])
def test_feature_future_isolation(family):
    f=fixture_frame(); s=Setup(family); before=features(f,s).iloc[:150]
    f.loc[150:,['open','high','low','close']]*=3
    pd.testing.assert_frame_equal(before,features(f,s).iloc[:150])


def test_historical_to_journal_end_to_end(tmp_path):
    f=fixture_frame(); e=replay(f,Setup('rsi'),symbol='TEST',begin=60,end=len(f),fee_bps=10,database=tmp_path/'e2e.db')
    m=metrics(e); assert m['trades']>0 and e.state['position'] is None
    assert e.state['equity']==pytest.approx(100+sum(x['pnl'] for x in e.journal if x['event']=='EXIT'))
    assert e.db.execute('select count(*) from equity').fetchone()[0]==140
    assert m['fees_gbp']>0


def test_monte_carlo_boundaries_reproducibility_and_exact_tail():
    assert requirement()['calendar_daily']==pytest.approx(.0797751623)
    a=binary_paths(.45,2,.01,paths=1000,horizons={21:'30_days'})
    assert a==binary_paths(.45,2,.01,paths=1000,horizons={21:'30_days'})
    h=a['horizons']['30_days']; assert h['hit_1000']==0 and h['loss_100']==0
    assert exact_terminal_probability(.45,2,.01,21)['probability']==0
    z=block_paths(np.zeros(40),paths=1000); assert z['horizons']['12_months']['p50']==100


def test_partial_fill_only_books_executed_quantity():
    e=PaperEngine(fee_bps=0,spread_bps=0,slip_bps=0,shadow=True)
    e.step(bar(),signal(),fill_fraction=.5)
    assert e.state['position']['qty']==.125
    assert e.state['cash']==87.5
    assert any(x['event']=='PARTIAL_IOC' for x in e.journal)
    e.step(bar(2,100,105,99,104))
    assert e.state['equity']==100.5


def test_order_rejection_has_no_fill_or_position():
    e=PaperEngine(shadow=True); e.step(bar(),signal(),order_rejected=True)
    assert e.state['cash']==100 and e.state['position'] is None
    assert e.journal[-1]['event']=='ORDER_REJECTED'


def test_minimum_economical_notional():
    from analysis.small_account_costs import minimum_economic_notional
    assert np.isinf(minimum_economic_notional(10,38,0))
    assert minimum_economic_notional(100,20,1)==pytest.approx(1000/3)


def test_runtime_file_kill_switch(tmp_path,monkeypatch):
    path=tmp_path/'STOP'; monkeypatch.setenv('STOP_TRADING_FILE',str(path))
    engine=RiskEngine(); assert engine.approve(snapshot())[0]>0
    path.touch(); assert engine.approve(snapshot())[0]==0


def test_future_signal_and_unavailable_risk_engine_are_rejected():
    e=PaperEngine(shadow=True)
    with pytest.raises(ValueError,match='signal must precede'): e.step(bar(),signal(2))
    e.risk=None
    with pytest.raises(AttributeError): e.step(bar(),signal())
    assert e.state['last_ts'] is None and not e.journal


def test_reducing_orders_cannot_reverse_position_and_survive_entry_kill(monkeypatch):
    risk=RiskEngine()
    assert not risk.approve_reduction(2,1,100)
    assert not risk.approve_reduction(float('nan'),1,100)
    e=PaperEngine(fee_bps=0,spread_bps=0,slip_bps=0,shadow=True)
    e.step(bar(),signal())
    monkeypatch.setenv('STOP_TRADING','true')
    e.step(bar(2,100,105,99,104))
    assert e.state['position'] is None and e.state['equity']==101
