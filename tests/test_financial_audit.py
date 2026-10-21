"""Independent regression audit. Oracle never imports production code."""
import json, math, sqlite3
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from tests.financial_oracle import D, Ledger, size, statistics, indicators, terminal_probability, hit_probability
from livesim.small_account import PaperEngine
from livesim.small_account_risk import RiskEngine, RiskLimits, RiskSnapshot
from strategies.small_account import Setup, features
from analysis.small_account_research import metrics, replay
from analysis.small_account_mc import binary_paths, block_paths, requirement, exact_terminal_probability


def bar(day, o=100, h=101, l=99, c=100):
    return dict(ts=pd.Timestamp('2025-01-01',tz='UTC')+pd.Timedelta(days=day),open=o,high=h,low=l,close=c)


def signal(day=0, **kw):
    return dict(**bar(day),signal=True,distance=kw.get('distance',2),regime='range_or_down')


def engine(**kw):
    return PaperEngine(shadow=True,fee_bps=0,spread_bps=0,slip_bps=0,**kw)


def test_time_exit_precedes_intrabar_future():
    e=engine(setup=Setup(hold=1)); e.step(bar(1),signal())
    e.step(bar(2,100,110,90,105))
    assert e.journal[-1]['reason']=='time'
    assert e.state['equity']==100


def test_prior_completed_trailing_stop_active_at_open():
    e=engine(setup=Setup(stop_kind='trailing')); e.step(bar(1),signal())
    prev=dict(signal(1),close=103,distance=2,signal=False)
    e.step(bar(2,102,103,100,102),prev)
    assert e.state['position'] is None
    assert e.journal[-1]['price']==101
    assert e.state['equity']==100.25


def test_breakeven_not_in_average_loss():
    e=SimpleNamespace(curve=[dict(equity=x,exposure=0) for x in (110,100,100)],
                      journal=[dict(event='EXIT',pnl=x) for x in (10,-10,0)])
    expected=statistics([100,110,100,100],[10,-10,0]); actual=metrics(e)
    assert actual['average_loss']==expected['average_loss']
    assert actual['reward_risk_realized']==expected['reward_risk_realized']


def test_open_risk_includes_liquidation_costs():
    e=PaperEngine(shadow=True,fee_bps=10,spread_bps=2,slip_bps=3)
    e.step(bar(1),signal()); p=e.state['position']
    expected=D(p['qty'])*(D(100)-D(p['stop'])*D('.9996')*D('.999'))
    assert e.curve[-1]['risk']==pytest.approx(float(expected),abs=1e-10)


def test_restart_rejects_corrupted_cash(tmp_path):
    db=tmp_path/'paper.sqlite'; e=engine(database=db); e.step(bar(1),signal()); e.db.close()
    con=sqlite3.connect(db); s=json.loads(con.execute('select payload from state').fetchone()[0])
    s['cash']+=1; s['equity']+=1
    con.execute('update state set payload=?',(json.dumps(s),)); con.commit(); con.close()
    with pytest.raises(ValueError,match='reconcil'): engine(database=db)


@pytest.mark.parametrize('cost',[3.,float('inf'),float('nan')])
def test_mc_invalid_or_bankrupt_cost_handled(cost):
    if math.isfinite(cost):
        r=binary_paths(.5,1,1,cost_r=cost,paths=100,horizons={1:'one'})
        assert r['horizons']['one']['p50']==0
    else:
        with pytest.raises(ValueError): binary_paths(.5,1,.1,cost_r=cost,paths=100,horizons={1:'one'})


def test_oracle_manual_accounting():
    a=Ledger(); a.buy(10,5,'.1'); assert a.cash==D('49.9'); assert a.mark(6)==D('109.9')
    assert a.sell(10,6,'.1')==D('9.8'); assert a.cash==D('109.8')
    assert a.initial+sum(x for _,x in a.entries)==a.cash
    b=Ledger(2000)
    for q,p in [(30,10),(40,'10.05'),(30,'10.10')]: b.buy(q,p)
    assert b.basis/b.quantity==D('10.05')
    c=Ledger(200); c.buy(10,5); c.buy(10,6)
    assert c.basis/c.quantity==D('5.5'); assert c.sell(5,7)==D('7.5'); assert c.quantity==15


@pytest.mark.parametrize('fee',[0,10,80])
@pytest.mark.parametrize('exit_price',[90,104])
def test_execution_ledger(fee,exit_price):
    e=PaperEngine(shadow=True,fee_bps=fee,spread_bps=2,slip_bps=3)
    entry=D(100)*D('1.0004'); stop=entry-2
    qty=size(100,100,entry,stop,D(fee)/10000,'.0004')
    a=Ledger(); a.buy(qty,entry,qty*entry*D(fee)/10000)
    e.step(bar(1),signal())
    assert e.state['cash']==pytest.approx(float(a.cash),abs=1e-8)
    assert e.state['equity']==pytest.approx(float(a.mark(100)),abs=1e-8)
    e.step(bar(2,exit_price,exit_price,exit_price,exit_price),close_final=True)
    raw=D(exit_price) if exit_price==90 else D(exit_price) # below target 104.04: final exit
    price=raw*D('.9996'); pnl=a.sell(qty,price,qty*price*D(fee)/10000)
    assert e.state['cash']==pytest.approx(float(a.cash),abs=1e-8)
    assert e.journal[-1]['pnl']==pytest.approx(float(pnl),abs=1e-8)


@pytest.mark.parametrize('risk',[.0025,.005,.01])
@pytest.mark.parametrize('fee',[0,.001,.008])
def test_independent_sizing(risk,fee):
    s=RiskSnapshot(100,100,100,98,fee,2,3,research_shadow=True)
    actual,_=RiskEngine(RiskLimits(risk_per_trade=risk)).approve(s)
    expected=size(100,100,100,98,fee,'.0004',risk=risk)
    assert actual==pytest.approx(float(expected),abs=1e-12)
    assert D(str(actual))*(D(100)*(1+D(fee))-D(98)*D('.9996')*(1-D(fee)))<=D(100)*D(risk)+D('1e-12')


@pytest.mark.parametrize('risk',[.02,.05,.10])
def test_unsupported_risk_rejected(risk):
    with pytest.raises(ValueError): RiskLimits(risk_per_trade=risk)


@pytest.mark.parametrize('family',['donchian','ma','rsi','bollinger','volume','gap'])
def test_all_indicators_signals_and_prefix_invariance(family):
    rng=np.random.default_rng(451); c=100+np.cumsum(rng.normal(0,2,180)); o=c-rng.uniform(-2,2,180)
    f=pd.DataFrame(dict(ts=pd.date_range('2025-01-01',periods=180,tz='UTC'),open=o,
                        high=np.maximum(o,c)+1,low=np.minimum(o,c)-1,close=c,volume=rng.integers(100,1000,180)))
    oracle=indicators(f.to_dict('records')); actual=features(f,Setup(family))
    for i,row in enumerate(oracle):
        if row['atr'] is not None: assert actual.atr.iloc[i]==pytest.approx(row['atr'],abs=1e-11)
        assert bool(actual.signal.iloc[i])==row['signals'][family]
    for cut in (25,50,100,150):
        pd.testing.assert_frame_equal(actual.iloc[:cut],features(f.iloc[:cut],Setup(family)))


def test_manual_statistics_and_compounding():
    expected=statistics([100,120,110,90,95,130],[10,20,-5,-10])
    assert expected['max_drawdown']==.25 and expected['profit_factor']==2
    assert statistics([100,110,121],[10,11])['total_return']==pytest.approx(.21)
    e=SimpleNamespace(curve=[dict(equity=x,exposure=0) for x in [120,110,90,95,130]],journal=[dict(event='EXIT',pnl=x) for x in [10,20,-5,-10]])
    for key,value in expected.items(): assert metrics(e)[key]==pytest.approx(value) if value is not None else metrics(e)[key] is None


def test_required_growth_and_exact_probability():
    from decimal import localcontext
    with localcontext() as ctx:
        ctx.prec=40
        assert requirement()['calendar_daily']==pytest.approx(float(D(10)**(D(1)/30)-1),abs=1e-15)
    assert requirement()['total_return']==9
    for n in (21,63):
        assert exact_terminal_probability(.45,2,.1,n)['probability']==pytest.approx(terminal_probability(.45,2,.1,n),abs=1e-13)


def test_seeded_paths_individually_and_probabilities():
    n=21; count=200; seed=81; p=.45; risk=.1
    actual=binary_paths(p,2,risk,paths=count,seed=seed,horizons={n:'test'})['horizons']['test']
    uniforms=np.random.default_rng(seed).random((n,count)); ends=[]; highs=[]; lows=[]; dds=[]
    for j in range(count):
        curve=[D(100)]; peak=D(100); dd=D(0)
        for t in range(n):
            curve.append(curve[-1]*(1+D(risk)*(D(2) if uniforms[t,j]<p else D(-1))-D(risk)*D('.1')))
            peak=max(peak,curve[-1]); dd=max(dd,1-curve[-1]/peak)
        ends.append(float(curve[-1])); highs.append(float(max(curve))); lows.append(float(min(curve))); dds.append(float(dd))
    assert actual['mean']==pytest.approx(sum(ends)/count,abs=1e-10)
    for target in (125,150,200,250,500,1000):
        assert actual[f'hit_{target}']==sum(v>=target for v in highs)/count
        assert actual[f'end_above_{target}']==sum(v>=target for v in ends)/count
    for level in (25,50,75,100): assert actual[f'loss_{level}']==sum(v<=100-level for v in lows)/count
    for level in (20,25,50,75): assert actual[f'drawdown_{level}']==sum(v>=level/100 for v in dds)/count
    for q in (5,25,50,75,95): assert actual[f'p{q}']==pytest.approx(np.percentile(ends,q),abs=1e-10)
    assert binary_paths(p,2,risk,paths=count,seed=seed,horizons={n:'test'})['horizons']['test']==actual
    assert binary_paths(p,2,risk,paths=count,seed=seed+1,horizons={n:'test'})['horizons']['test']!=actual


def test_100000_mc_paths_against_exact_dp():
    import time
    results=[]
    exact=hit_probability(.45,2,.1,21,200)
    for count in (10000,50000,100000):
        start=time.perf_counter(); result=binary_paths(.45,2,.1,paths=count,horizons={21:'test'})
        actual=result['horizons']['test']; assert actual['paths']==count
        # Fixed 6-sigma binomial bound, not an arbitrary relative tolerance.
        assert abs(actual['hit_200']-exact)<=6*math.sqrt(exact*(1-exact)/count)
        results.append(dict(paths=count,seconds=time.perf_counter()-start,expected=exact,actual=actual['hit_200']))
    output=Path('results/qa_audit_20260930'); output.mkdir(exist_ok=True)
    (output/'mc_convergence.json').write_text(json.dumps(results,indent=2))
    zero=block_paths(np.zeros(40),paths=100000)
    assert all(h['paths']==100000 and h['mean']==100 and h['drawdown_20']==0 for h in zero['horizons'].values())


def golden_run(database=None):
    fixture=json.loads(Path('tests/fixtures/financial_golden.json').read_text())
    e=PaperEngine(shadow=True,fee_bps=10,spread_bps=2,slip_bps=3,database=database)
    for row in fixture['bars']:
        b=bar(row['day'],row['open'],row['high'],row['low'],row['close'])
        e.step(b,signal(row['signal_day']) if 'signal_day' in row else None)
    return e,fixture


def test_golden_30_trade_ledger_database_read_model(tmp_path):
    from dashboard.views.small_account import load_monitor
    e,f=golden_run(tmp_path/'paper.sqlite'); actual=metrics(e)
    for key,value in f['expected'].items():
        if key in ('net_pnl','gross_pnl'): continue
        assert actual[key]==pytest.approx(value,abs=1e-8),key
    exits=[x for x in e.journal if x['event']=='EXIT']; assert len(exits)==30
    for expected,trade in zip(f['trades'],exits):
        for key in ('quantity','pnl'): assert trade[key]==pytest.approx(float(expected[key]),abs=1e-8)
        assert trade['price']==pytest.approx(float(expected['exit']),abs=1e-8)
    assert [x['equity'] for x in e.curve]==pytest.approx(f['equity'][1:],abs=1e-8)
    _,state,eq,journal,_,_=load_monitor(tmp_path)
    assert state==e.state
    assert eq.equity.tolist()==pytest.approx(f['equity'][1:],abs=1e-8)
    assert journal[journal.event=='EXIT'].pnl.sum()==pytest.approx(f['expected']['net_pnl'],abs=1e-8)
    restored=PaperEngine(shadow=True,fee_bps=10,spread_bps=2,slip_bps=3,database=tmp_path/'paper.sqlite')
    assert metrics(restored)==metrics(e)


def test_state_tamper_fails_closed():
    e=engine(); e.state['cash']=101; e.state['equity']=101
    with pytest.raises(ValueError,match='reconcil'): e.step(bar(1),signal())
    assert not e.journal


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf'),1e308])
def test_bad_ohlc_does_not_mutate(value):
    e=engine(); b=bar(1); b['open']=value; original=json.dumps(e.state)
    with pytest.raises(ValueError): e.step(b,signal())
    assert json.dumps(e.state)==original


def test_duplicate_and_concurrent_writers(tmp_path):
    a=engine(database=tmp_path/'paper.sqlite'); b=engine(database=tmp_path/'paper.sqlite')
    a.step(bar(1),signal()); before=json.dumps(a.state)
    assert a.step(bar(1),signal())=='DUPLICATE_OR_OLD'
    assert a.step(bar(0),signal(-1))=='DUPLICATE_OR_OLD'
    assert json.dumps(a.state)==before
    with pytest.raises(RuntimeError,match='concurrent'): b.step(bar(1),signal())
    assert b.state['cash']==100 and not b.journal


def test_fee_and_slippage_monotonicity_randomized():
    # Fixed quantity isolates costs from the separate position-sizing effect.
    rng=np.random.default_rng(11)
    for _ in range(250):
        entry=D(rng.uniform(1,100)); exit=D(rng.uniform(1,100)); q=D('.1')
        base=Ledger(); base.buy(q,entry); gross=base.sell(q,exit)
        costly=Ledger(); costly.buy(q,entry*D('1.001'),q*entry*D('.001'))
        net=costly.sell(q,exit*D('.999'),q*exit*D('.001'))
        assert net<gross; assert costly.initial+sum(x for _,x in costly.entries)==costly.cash


def test_kill_pending_signal_and_gap_extreme():
    e=engine(); e.step(bar(1),signal(),kill=True); assert e.state['position'] is None
    e.step(bar(2),signal(1)); q=e.state['position']['qty']
    e.step(bar(3,50,51,49,50)); assert e.state['equity']==100-50*q
    assert e.state['halted']; e.step(bar(4),signal(3)); assert e.state['position'] is None


def test_legacy_rsi_all_gain_and_cci_first_valid_window():
    from analysis.signal_library import _rsi,compute_signal_library
    c=pd.Series(np.arange(100.,150.)); assert _rsi(c,14).iloc[-1]==1
    f=pd.DataFrame(dict(ts=pd.date_range('2025-01-01',periods=50,tz='UTC'),open=c,high=c+1,low=c-1,close=c,volume=100))
    actual=compute_signal_library(f); window=list(range(100,110)); avg=sum(window)/10
    expected=(109-avg)/(.015*sum(abs(x-avg) for x in window)/10)
    assert actual.cci_10.iloc[9]==pytest.approx(min(5,max(-5,expected)))
    f.loc[9,'close']=104.001; f.loc[9,'high']=105.001; f.loc[9,'low']=103.001
    window=list(range(100,109))+[104.001]; avg=sum(window)/10
    expected=(window[-1]-avg)/(.015*sum(abs(x-avg) for x in window)/10)
    assert compute_signal_library(f).cci_10.iloc[9]==pytest.approx(expected)


def test_dashboard_rolling_annualization_and_initial_return():
    from analysis.small_account_monitor import rolling_performance
    eq=pd.DataFrame(dict(ts=pd.date_range('2025-01-01',periods=3,tz='UTC'),equity=[110,100,120]))
    expected=statistics([100,110,100,120],[])['sharpe']
    assert rolling_performance(eq,pd.DataFrame(),annual=252)['sharpe']==pytest.approx(expected)


def test_legacy_ema_macd_vwap_against_loops():
    from analysis.signal_library import compute_signal_library
    n=80; closes=[100+math.sin(i/3)*3+i*.1 for i in range(n)]
    f=pd.DataFrame(dict(ts=pd.date_range('2025-01-01T10:00Z',periods=n,freq='min'),open=closes,
                        high=np.array(closes)+1,low=np.array(closes)-1,close=closes,volume=np.arange(1,n+1)))
    result=compute_signal_library(f)
    ema={}
    for period in (5,12,26):
        v=closes[0]; values=[]
        for price in closes:
            v=2/(period+1)*price+(1-2/(period+1))*v; values.append(v)
        ema[period]=values
    for i in range(26,n):
        assert result.ema_distance_5.iloc[i]==pytest.approx(closes[i]/ema[5][i]-1,abs=1e-12)
        assert result.macd_12_26.iloc[i]==pytest.approx((ema[12][i]-ema[26][i])/closes[i],abs=1e-12)
        vwap=sum(closes[j]*(j+1) for j in range(i+1))/sum(range(1,i+2))
        assert result.session_vwap_deviation.iloc[i]==pytest.approx(closes[i]/vwap-1,abs=1e-12)


def test_replay_and_step_identical_with_database(tmp_path):
    c=100+np.sin(np.arange(200)/6)*6+np.arange(200)*.06
    f=pd.DataFrame(dict(ts=pd.date_range('2025-01-01',periods=200,tz='UTC'),open=c,high=c+2,low=c-2,close=c,volume=1000))
    setup=Setup('rsi'); historical=replay(f,setup,symbol='TEST',begin=60,end=200,fee_bps=10)
    live=PaperEngine(setup,symbol='TEST',fee_bps=10,shadow=True,database=tmp_path/'paper.sqlite')
    records=features(f,setup).to_dict('records')
    for i in range(60,200): live.step(records[i],records[i-1],close_final=i==199)
    assert historical.state==live.state and historical.journal==live.journal
    ledger=Ledger()
    for e in live.journal:
        if e['event']=='ENTRY': ledger.buy(e['quantity'],e['price'],e['fee'])
        elif e['event']=='EXIT': assert float(ledger.sell(e['quantity'],e['price'],e['fee']))==pytest.approx(e['pnl'],abs=1e-9)
    assert live.state['cash']==pytest.approx(float(ledger.cash),abs=1e-8)


def test_legacy_actual_partial_fill_dispatch():
    from unittest.mock import Mock
    from livesim.alpaca_exec_client import AlpacaLiveExecutionClient
    fake=SimpleNamespace(_emitted_fills=set(),_alpaca_to_nautilus={'o':object()},_fills_emitted=0,
                         _emit_fill_from_alpaca=Mock(),_log=Mock())
    order=SimpleNamespace(id='o',status='partially_filled',filled_qty='30',filled_avg_price='10')
    AlpacaLiveExecutionClient._handle_alpaca_order_update(fake,order)
    assert fake._emit_fill_from_alpaca.call_count==1


def test_legacy_analytics_documented_conventions():
    from analysis.metrics import sharpe,cagr,max_drawdown
    from statistics import pstdev,mean
    r=[.1,-.05,.03,.01]
    assert sharpe(pd.Series(r))==pytest.approx(mean(r)/pstdev(r)*math.sqrt(252))
    assert cagr(pd.Series(r))==pytest.approx(math.prod(1+x for x in r)**(252/4)-1)
    assert max_drawdown(pd.Series([100,120,110,90,95,130]))==-.25


def test_monitor_ignores_corrupt_db(tmp_path):
    from dashboard.views.small_account import load_monitor
    e=engine(database=tmp_path/'paper.sqlite'); e.step(bar(1),signal())
    e.db.execute("update equity set value=999")
    with pytest.raises(ValueError,match='reconcil'): load_monitor(tmp_path)


def test_data_invalid_and_dst():
    from analysis.rule_discovery import validate_bars
    rows=[dict(bar(i),volume=100) for i in range(20)]
    f=pd.DataFrame(rows)
    for broken in (pd.concat([f,f.iloc[[0]]]),f.assign(close=-1),f.assign(volume=-1)):
        with pytest.raises(ValueError): validate_bars(broken)
    # UTC events straddling UK fallback cannot be mistaken for duplicates.
    e=engine()
    a=dict(bar(1),ts=pd.Timestamp('2025-10-26T00:30Z'))
    b=dict(bar(1),ts=pd.Timestamp('2025-10-26T01:30Z'))
    e.step(a); e.step(b); assert len(e.curve)==2


def test_cv_purge_groups_sessions_across_assets():
    from models.cv import PurgedWalkForwardSplit
    dates=pd.date_range('2025-01-01',periods=60,freq='B').repeat(4)
    unique=dates.unique()
    for train,test in PurgedWalkForwardSplit(5,5,2).split(dates):
        assert not set(dates[train]) & set(dates[test])
        if len(train): assert unique.get_loc(dates[test[0]])-unique.get_loc(dates[train[-1]])>=8


def test_no_fractional_truncation_in_legacy_order_request():
    from livesim.alpaca_exec_client import _build_market_order_request
    from nautilus_trader.model.enums import OrderSide
    with pytest.raises(ValueError): _build_market_order_request('SPY',1.7,OrderSide.BUY)


def test_chart_points_progress_and_calendar_cutoff(tmp_path):
    import pyarrow as pa
    from streamlit.testing.v1 import AppTest
    e,f=golden_run(tmp_path/'paper.sqlite')
    app=AppTest.from_file('dashboard/small_account_app.py').run()
    app.text_input[0].set_value(str(tmp_path)).run(); assert not app.exception
    chart=app.get('vega_lite_chart')[0].proto
    points=pa.ipc.open_stream(chart.datasets[0].data.data).read_all().to_pandas()
    assert points.equity.tolist()==pytest.approx(f['equity'][1:],abs=1e-8)
    assert pd.to_datetime(points.ts,utc=True).tolist()==[pd.Timestamp(x['ts']) for x in e.curve]
    milestone=app.dataframe[0].value
    assert milestone.Progress.iloc[-1]==pytest.approx((f['expected']['final_equity']-100)/900)
    daily=app.dataframe[1].value['Daily P&L'].iloc[0]
    assert daily==pytest.approx(f['equity'][-1]-f['equity'][-2],abs=1e-8)


def test_bootstrap_nonzero_paths_independent():
    returns=[.01,-.02,.005,0]*10; count=100; seed=987; block=5
    actual=block_paths(returns,paths=count,seed=seed,block=block)
    rng=np.random.default_rng(seed); curves=[[100.] for _ in range(count)]
    for start in range(0,252,block):
        offsets=rng.integers(0,len(returns),count)
        for j,offset in enumerate(offsets):
            for k in range(min(block,252-start)): curves[j].append(curves[j][-1]*(1+returns[(offset+k)%len(returns)]))
    for n,label in [(21,'30_days'),(63,'90_days'),(126,'6_months'),(252,'12_months')]:
        ends=[c[n] for c in curves]; expected=sum(ends)/count; summary=actual['horizons'][label]
        assert summary['mean']==pytest.approx(expected,abs=1e-10)
        assert summary['hit_125']==sum(max(c[:n+1])>=125 for c in curves)/count
        assert summary['loss_25']==sum(min(c[:n+1])<=75 for c in curves)/count
    assert actual['seed']==seed and actual['paths']==count and actual['annual']==252


def test_restart_in_new_process(tmp_path):
    import subprocess,sys
    e,f=golden_run(tmp_path/'paper.sqlite'); e.db.close()
    command="from livesim.small_account import PaperEngine; import json,sys; e=PaperEngine(shadow=True,fee_bps=10,spread_bps=2,slip_bps=3,database=sys.argv[1]); print(json.dumps(e.state))"
    result=subprocess.run([sys.executable,'-c',command,str(tmp_path/'paper.sqlite')],capture_output=True,text=True,check=True)
    state=json.loads(result.stdout); assert state['cash']==pytest.approx(f['expected']['final_equity'],abs=1e-8)
    assert len(state['closed_pnl'])==30


@pytest.mark.parametrize('price',[float('inf'),float('nan'),0,-1])
def test_bad_quote_never_reaches_execution(price):
    import time
    from data.small_account_feed import live_snapshot
    def request(endpoint,**kw):
        if endpoint=='Time': return {'unixtime':time.time()}
        if endpoint=='AssetPairs': return {'pair':{'status':'online','costmin':'1','ordermin':'.0001','lot_decimals':8}}
        return {'pair':{'b':[price],'a':[price]}}
    with pytest.raises(ValueError): live_snapshot(request=request)


def test_scipy_second_statistics_crosscheck():
    from scipy.stats import tstd
    returns=[.1,-.05,.03,.01]
    e=[100.]
    for r in returns: e.append(e[-1]*(1+r))
    oracle=statistics(e,[])
    assert oracle['sharpe']==pytest.approx(np.mean(returns)/tstd(returns)*math.sqrt(252))


@pytest.mark.parametrize('n',[5,21,63,252])
def test_exact_tail_includes_all_wins_boundary(n):
    actual=exact_terminal_probability(.5,1,.01,n,cost_r=0,target=1.01**n)
    assert actual['wins_required']==n
    assert actual['probability']==pytest.approx(.5**n,rel=1e-12,abs=0)


def test_metrics_refuse_nonfinite_scenario_parameters():
    for kwargs in [dict(paths=0),dict(paths=1.1),dict(block=0),dict(annual=360)]:
        with pytest.raises(ValueError): block_paths([0.]*40,**kwargs)


def test_random_seed_is_persisted_and_replayable():
    a=binary_paths(.45,2,.1,seed=None,paths=50,horizons={21:'test'})
    b=binary_paths(.45,2,.1,seed=None,paths=50,horizons={21:'test'})
    assert a['seed']!=b['seed']
    assert a==binary_paths(.45,2,.1,seed=a['seed'],paths=50,horizons={21:'test'})
