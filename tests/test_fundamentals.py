"""Point-in-time, independent arithmetic and complete risk/paper boundary tests."""
from dataclasses import replace
import json,math,sqlite3
from pathlib import Path
from decimal import Decimal
import numpy as np
import pandas as pd
import pytest
from fundamentals.store import Fact,PointInTimeFundamentalStore
from fundamentals.features import accounting_features,FundamentalFeatureEngine,valuation_features,normalize_cross_section,quarters,annual_cagr
from fundamentals.models import fit_model,FundamentalScoreEngine,CombinedSignalEngine,MacroRegimeService,EventCatalystEngine
from fundamentals.providers import sec_facts,alfred_snapshot
from fundamentals.adapters import sector_features,crypto_features,estimate_revisions,historical_fx,apply_known_splits
from fundamentals.api import FundamentalAPI
from livesim.small_account import PaperEngine


def fact(metric='eps',value=1.2,end='2024-03-31',reported='2024-05-10T12:00:00Z',revision='original',**kwargs):
    currency='' if metric in ('diluted_shares','shares_outstanding') else 'USD'
    units='shares' if not currency else 'USD/shares' if metric=='eps' else 'USD'
    return Fact('TEST',metric,value,end,reported,reported,'fixture',revision,currency,units,**kwargs)


def populated_store(path=':memory:'):
    s=PointInTimeFundamentalStore(path); rows=[]
    for year,revenue in [(2020,80),(2021,90),(2022,100),(2023,120)]:
        vals=dict(revenue=revenue,gross_profit=.4*revenue,operating_income=.2*revenue,net_income=.1*revenue,
                  ocf=.15*revenue,capex=.05*revenue,eps=revenue/100,depreciation=2,interest_expense=1,pretax_income=15,tax_expense=3,
                  assets=200,equity=100,cash=20,current_assets=60,current_liabilities=30,short_debt=10,long_debt=40,inventory=10,
                  receivables=12,payables=8,prepaids=0,cogs=.6*revenue,diluted_shares=10,rd=5,buybacks=1)
        instant={'assets','equity','cash','current_assets','current_liabilities','short_debt','long_debt','inventory','receivables','payables','prepaids'}
        for metric,value in vals.items():
            rows.append(fact(metric,value,f'{year}-12-31',f'{year+1}-05-10T12:00:00Z',revision=str(year),period_start='' if metric in instant else f'{year}-01-01'))
    s.ingest(rows);return s


def test_quarter_not_known_at_period_end_or_before_release():
    s=PointInTimeFundamentalStore();s.ingest([fact()])
    assert not s.as_of('TEST','2024-04-15T00:00:00Z')
    assert s.as_of('TEST','2024-05-11T00:00:00Z')[0]['value']==1.2
    assert not s.as_of('TEST','2024-05-11T00:00:00Z',observed_only=True)


def test_restatement_and_duplicate_ingestion():
    s=PointInTimeFundamentalStore();a=fact();b=fact(value=1.05,reported='2024-06-10T00:00:00Z',revision='amended')
    s.ingest([a,b,a]);assert len(s.history('TEST'))==2
    assert s.as_of('TEST','2024-06-01T00:00:00Z')[0]['value']==1.2
    assert s.as_of('TEST','2024-06-11T00:00:00Z')[0]['value']==1.05
    s.ingest([replace(b,value=None,reported_at='2024-07-01T00:00:00Z',available_at='2024-07-01T00:00:00Z',revision='withdrawn')])
    assert s.as_of('TEST','2024-07-02T00:00:00Z')[0]['value'] is None


def test_batch_validation_atomic_and_timestamps():
    s=PointInTimeFundamentalStore()
    with pytest.raises(ValueError):s.ingest([fact(),fact(value=float('nan'))])
    assert s.history('TEST')==[]
    with pytest.raises(ValueError):s.ingest([replace(fact(),available_at='2024-04-01T00:00:00Z')])
    with pytest.raises(ValueError):s.ingest([replace(fact(),available_at='2024-05-10')])


def test_hand_calculated_growth_margins_cashflows_health():
    v=dict(revenue=120,gross_profit=48,operating_income=24,net_income=12,ocf=18,capex=6,
           assets=200,equity=100,cash=20,current_assets=60,current_liabilities=30,inventory=10,prepaids=0,short_debt=10,long_debt=40,
           depreciation=2,interest_expense=1,pretax_income=15,tax_expense=3)
    p={**v,'revenue':100,'assets':180,'equity':90};f,_,_=accounting_features(v,p)
    assert f['revenue_yoy']==pytest.approx(float(Decimal(120)/100-1),abs=1e-14)
    assert f['gross_margin']==.4 and f['fcf']==12 and f['current_ratio']==2
    assert f['quick_ratio']==pytest.approx(50/30);assert f['debt_equity']==.5 and f['net_debt']==30
    assert f['roe']==pytest.approx(12/95);assert f['roa']==pytest.approx(12/190)
    assert f['roic']==pytest.approx(24*.8/125)
    assert f['accrual_ratio']==pytest.approx(-6/190)
    assert accounting_features({})[0]['revenue_yoy'] is None
    assert accounting_features(v,p,sector='banks')[0]['debt_equity'] is None


def test_feature_snapshot_filing_boundary_lineage_and_missing():
    s=populated_store();e=FundamentalFeatureEngine(s)
    a=e.snapshot('TEST','2024-04-15T00:00:00Z');b=e.snapshot('TEST','2024-05-11T00:00:00Z',persist=True)
    assert a['period_end']=='2022-12-31' and b['period_end']=='2023-12-31'
    assert b['features']['revenue_yoy']==pytest.approx(.2)
    assert b['features']['fcf']==12 and b['confidence'] is None
    assert all(pd.Timestamp(r['available_at'])<=pd.Timestamp(b['as_of']) for r in s.as_of('TEST',b['as_of']) if r['id'] in b['lineage'])
    assert e.snapshot('TEST','2026-01-01T00:00:00Z')['status']=='STALE'
    assert e.snapshot('XLE','2024-06-01T00:00:00Z',asset_class='etf')['status']=='REQUIRES_DATED_HOLDINGS'


def test_future_facts_cannot_change_historical_features():
    s=populated_store();e=FundamentalFeatureEngine(s);a=e.snapshot('TEST','2024-06-01T00:00:00Z')
    s.ingest([fact('revenue',999,'2023-12-31','2025-01-01T00:00:00Z','later',period_start='2023-01-01')])
    assert e.snapshot('TEST','2024-06-01T00:00:00Z')==a


def test_ytd_differencing_not_sum_and_no_future_revision():
    s=PointInTimeFundamentalStore()
    s.ingest([fact('ocf',10,'2024-03-31','2024-05-01T00:00:00Z',period_start='2024-01-01'),
              fact('ocf',25,'2024-06-30','2024-08-01T00:00:00Z',period_start='2024-01-01'),
              fact('ocf',45,'2024-09-30','2024-11-01T00:00:00Z',period_start='2024-01-01')])
    q=quarters(s.as_of('TEST','2024-11-02T00:00:00Z'),'ocf')
    assert [q[k]['value'] for k in sorted(q)]==[10,15,20]
    assert '2024-09-30' not in quarters(s.as_of('TEST','2024-08-02T00:00:00Z'),'ocf')


def test_valuation_currency_split_and_negative_earnings():
    values=dict(net_income=10,revenue=100,equity=50,short_debt=2,long_debt=8,preferred_equity=0,minority_interest=0,cash=5,ocf=15,capex=5,operating_income=15,depreciation=5)
    out=valuation_features(5,20,values,price_currency='GBP',statement_currency='GBP',price_basis='A',share_basis='A')
    assert out['pe']==10 and out['ps']==1 and out['pb']==2 and out['fcf_yield']==.1
    assert out['enterprise_value']==105 and out['ev_ebitda']==5.25
    for kw in ({'price_currency':'USD','statement_currency':'GBP','price_basis':'A','share_basis':'A'},
               {'price_currency':'USD','statement_currency':'USD','price_basis':'A','share_basis':'B'}):
        with pytest.raises(ValueError):valuation_features(5,20,values,**kw)


def test_sector_normalization_no_missing_zero():
    x=pd.DataFrame({'sector':['energy']*5+['banks'],'v':[1,2,3,4,None,100]})
    out=normalize_cross_section(x,'v',minimum=4)
    assert out.v_percentile.iloc[0]==.125 and out.v_percentile.iloc[3]==.875
    assert pd.isna(out.v_percentile.iloc[4]) and pd.isna(out.v_percentile.iloc[5])


def test_macro_vintage_revision_not_retroactive():
    s=PointInTimeFundamentalStore()
    a=b'observation_date,CPIAUCSL_20200601\n2020-01-01,100\n'
    b=b'observation_date,CPIAUCSL_20200701\n2020-01-01,105\n'
    s.ingest(alfred_snapshot(a,'CPIAUCSL','2020-06-01'));s.ingest(alfred_snapshot(b,'CPIAUCSL','2020-07-01'))
    assert not s.as_of('MACRO:CPIAUCSL','2020-06-01T12:00:00Z')
    assert s.as_of('MACRO:CPIAUCSL','2020-06-15T00:00:00Z')[0]['value']==100
    assert s.as_of('MACRO:CPIAUCSL','2020-07-02T00:00:00Z')[0]['value']==105
    with pytest.raises(ValueError):alfred_snapshot(a,'CPIAUCSL','2020-07-01')


def test_sec_acceptance_and_date_only_conservative_delay():
    payload={'cik':1,'facts':{'us-gaap':{'Revenues':{'units':{'USD':[{'val':120,'end':'2023-12-31','start':'2023-01-01','filed':'2024-05-10','form':'10-K','accn':'A'}]}}}}}
    sub={'filings':{'recent':{'accessionNumber':['A'],'acceptanceDateTime':['2024-05-10T20:00:00Z']}}}
    row=sec_facts(payload,sub,'TEST')[0]
    assert pd.Timestamp(row.available_at)==pd.Timestamp('2024-05-10T20:05:00Z')
    fallback=sec_facts(payload,{},'TEST')[0]
    assert pd.Timestamp(fallback.available_at)==pd.Timestamp('2024-05-11T04:00:00Z')


def trained_artifact():
    dates=pd.date_range('2023-01-01',periods=40,tz='UTC')
    panel=pd.DataFrame(dict(as_of=dates,label_end=dates+pd.Timedelta(days=5),target=.02,revenue_yoy=np.linspace(.1,.3,40)))
    return fit_model(panel,['revenue_yoy'],fit_at='2024-01-01T00:00:00Z')


def test_model_only_matured_labels_and_train_normalization():
    dates=pd.date_range('2023-01-01',periods=40,tz='UTC');p=pd.DataFrame(dict(as_of=dates,label_end=dates+pd.Timedelta(days=20),target=.01,x=np.arange(40)))
    with pytest.raises(ValueError):fit_model(p,['x'],fit_at='2023-01-15T00:00:00Z')
    a=trained_artifact();e=FundamentalScoreEngine(a)
    assert e.score({'revenue_yoy':.2},'2024-02-01T00:00:00Z')['score']==pytest.approx(.02)
    with pytest.raises(ValueError):e.score({'revenue_yoy':.2},'2023-02-01T00:00:00Z')
    assert e.score({},'2024-02-01T00:00:00Z')['score'] is None
    with pytest.raises(ValueError):replace(a,deployment_approved=True)


def test_ingest_features_model_filter_risk_paper_position_exit(tmp_path):
    s=populated_store(tmp_path/'fund.sqlite');combo=CombinedSignalEngine(FundamentalFeatureEngine(s),FundamentalScoreEngine(trained_artifact()),policy='filter',research_shadow=True)
    e=PaperEngine(symbol='TEST',shadow=True,fee_bps=0,spread_bps=0,slip_bps=0,database=tmp_path/'paper.sqlite',fundamental_filter=combo)
    prev={'ts':'2024-05-11T00:00:00Z','signal':True,'distance':2.,'regime':'fixture'}
    e.step(dict(ts='2024-05-12T00:00:00Z',open=100,high=101,low=99,close=100),prev)
    assert e.state['position']['qty']==.25 and e.state['cash']==75
    d=next(x['decision'] for x in e.journal if x['event']=='FUNDAMENTAL_DECISION')
    assert d['fundamental']['features']['revenue_yoy']==pytest.approx(.2) and d['risk']['status']=='APPROVED_SHADOW'
    e.step(dict(ts='2024-05-13T00:00:00Z',open=100,high=105,low=99,close=104))
    assert e.state['cash']==101 and e.state['position'] is None
    assert e.db.execute("select count(*) from events where payload like '%FUNDAMENTAL_DECISION%'").fetchone()[0]==1


@pytest.mark.parametrize('policy',['filter','selection','weighted','regime'])
def test_unvalidated_fundamentals_never_create_order(policy):
    s=populated_store();combo=CombinedSignalEngine(FundamentalFeatureEngine(s),policy=policy)
    e=PaperEngine(symbol='TEST',shadow=True,fundamental_filter=combo)
    e.step(dict(ts='2024-05-12T00:00:00Z',open=100,high=101,low=99,close=100),dict(ts='2024-05-11T00:00:00Z',signal=True,distance=2,regime='test'))
    assert e.state['position'] is None and e.state['cash']==100


def test_high_fundamental_score_cannot_bypass_kill_or_risk(monkeypatch):
    s=populated_store();combo=CombinedSignalEngine(FundamentalFeatureEngine(s),FundamentalScoreEngine(trained_artifact()),policy='filter',research_shadow=True)
    monkeypatch.setenv('STOP_TRADING','true');e=PaperEngine(symbol='TEST',shadow=True,fundamental_filter=combo)
    e.step(dict(ts='2024-05-12T00:00:00Z',open=100,high=101,low=99,close=100),dict(ts='2024-05-11T00:00:00Z',signal=True,distance=2,regime='test'))
    assert e.state['cash']==100 and e.state['position'] is None


def test_event_consensus_must_precede_earnings():
    s=PointInTimeFundamentalStore()
    s.add_event(symbol='TEST',event_type='earnings',event_at='2024-05-10T12:00:00Z',reported_at='2024-05-10T12:00:00Z',available_at='2024-05-10T12:01:00Z',source='fixture',revision='A',payload={'reported_eps':1.2,'consensus_eps':1.,'consensus_available_at':'2024-05-11T00:00:00Z'})
    out=EventCatalystEngine(s).state('TEST','2024-05-12T00:00:00Z')
    assert out['events'][0]['eps_surprise'] is None


def test_sector_and_crypto_distinct():
    assert sector_features('banks',dict(cet1_capital=12,risk_weighted_assets=100))['cet1_ratio']==.12
    assert sector_features('energy',dict(realized_price=80,production_cost_per_unit=25))['unit_margin']==55
    assert 'roe' not in crypto_features(dict(active_addresses=110),dict(active_addresses=100))


def test_splits_and_fx_use_historical_availability():
    s=PointInTimeFundamentalStore();s.add_event(symbol='TEST',event_type='split',event_at='2024-06-01T00:00:00Z',reported_at='2024-05-01T00:00:00Z',available_at='2024-05-01T00:00:00Z',source='fixture',revision='A',payload={'ratio':2})
    before=apply_known_splits(s,'TEST',10,100,'2024-01-01T00:00:00Z','2024-05-10T00:00:00Z');after=apply_known_splits(s,'TEST',10,100,'2024-01-01T00:00:00Z','2024-06-02T00:00:00Z')
    assert before['quantity']==10 and after['quantity']==20 and after['price']==50
    assert after['quantity']*after['price']==before['quantity']*before['price']
    s.ingest([Fact('FX:USDGBP','rate',.8,'2024-06-01','2024-06-01T00:00:00Z','2024-06-01T00:00:00Z','fixture','A','GBP','GBP/USD',kind='fx')])
    assert historical_fx(s,100,'USD','GBP','2024-05-01T00:00:00Z')['value'] is None
    assert historical_fx(s,100,'USD','GBP','2024-06-02T00:00:00Z')['value']==80


def test_future_classification_not_used_early():
    s=populated_store();s.ingest([replace(fact('sector',1,'2024-07-01','2024-05-01T00:00:00Z',kind='classification',metadata={'sector':'banks'}),currency='',units='category')])
    assert FundamentalFeatureEngine(s).snapshot('TEST','2024-06-01T00:00:00Z')['sector'] is None
    assert FundamentalFeatureEngine(s).snapshot('TEST','2024-07-02T00:00:00Z')['sector']=='banks'


def test_api_exact_values_and_no_score_fabrication():
    s=populated_store();api=FundamentalAPI(s);q={'as_of':'2024-06-01T00:00:00Z'}
    assert api.get('/fundamentals/TEST',q)['features']['fcf']==12
    assert api.get('/fundamentals/TEST/score',q)['score'] is None
    assert not api.get('/watchlist/fundamental',q)['ranked']
    with pytest.raises(KeyError):api.get('/orders',q)
    assert all(pd.Timestamp(r['available_at'])<=pd.Timestamp(q['as_of']) for r in api.get('/fundamentals/TEST/history',q)['records'])


def test_feature_dictionary_covers_every_snapshot_feature():
    from fundamentals.dictionary import feature_dictionary
    s=populated_store();snap=FundamentalFeatureEngine(s).snapshot('TEST','2024-06-01T00:00:00Z')
    definitions=feature_dictionary();assert set(snap['features'])<=set(definitions)
    assert all(definitions[k]['production_weight'] is None for k in snap['features'])


def test_cross_sectional_ic_and_future_membership_guard():
    from fundamentals.research import cross_sectional_ic
    f=pd.DataFrame({'symbol':[str(i) for i in range(20)],'as_of':'2024-01-01T00:00:00Z','label_end':'2024-02-01T00:00:00Z',
                    'member_known_at':'2023-12-01T00:00:00Z','score':np.arange(20),'target':np.arange(20)/100})
    assert cross_sectional_ic(f)[0]['rank_ic']==1
    f.loc[0,'member_known_at']='2024-01-02T00:00:00Z'
    with pytest.raises(ValueError):cross_sectional_ic(f)


def test_estimate_revisions_fixed_forecast_period():
    s=PointInTimeFundamentalStore()
    a=replace(fact('consensus_eps',1,'2024-12-31','2024-01-01T00:00:00Z','a',kind='estimate'),units='USD/shares')
    b=replace(a,value=1.2,reported_at='2024-02-01T00:00:00Z',available_at='2024-02-01T00:00:00Z',revision='b')
    s.ingest([a,b]);r=estimate_revisions(s,'TEST','2024-02-02T00:00:00Z','consensus_eps','2024-12-31')
    assert r['30d']==pytest.approx(.2) and r['90d'] is None


def test_http_api_actual_requests(tmp_path):
    import threading
    from urllib.request import urlopen
    from urllib.error import HTTPError
    from fundamentals.api import serve
    s=populated_store(tmp_path/'f.sqlite');s.close();server=serve(tmp_path/'f.sqlite',port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    root=f'http://127.0.0.1:{server.server_address[1]}'
    try:
        for endpoint in ('/fundamentals/TEST','/fundamentals/TEST/history','/fundamentals/TEST/score','/fundamentals/TEST/events','/macro/regime','/watchlist/fundamental','/strategies/bollinger/explanation'):
            with urlopen(root+endpoint+'?as_of=2024-06-01T00%3A00%3A00Z') as r:
                payload=json.load(r);assert r.status==200
                if endpoint=='/fundamentals/TEST':assert payload['features']['fcf']==12
        with pytest.raises(HTTPError):urlopen(root+'/orders')
        with pytest.raises(HTTPError):urlopen(root+'/fundamentals/TEST?as_of=2024-06-01')
    finally:server.shutdown();server.server_close();thread.join()


def test_observation_mode_preserves_technical_execution():
    s=populated_store();combo=CombinedSignalEngine(FundamentalFeatureEngine(s),policy='observe')
    engines=[PaperEngine(symbol='TEST',shadow=True,fee_bps=10,spread_bps=2,slip_bps=3,fundamental_filter=f) for f in (None,combo)]
    for e in engines:
        e.step(dict(ts='2024-05-12T00:00:00Z',open=100,high=101,low=99,close=100),dict(ts='2024-05-11T00:00:00Z',signal=True,distance=2,regime='test'))
        e.step(dict(ts='2024-05-13T00:00:00Z',open=90,high=91,low=89,close=90))
    assert engines[0].state==engines[1].state


def test_missing_constituents_not_reweighted_to_survivors():
    s=populated_store();rows=[]
    for symbol,weight in [('TEST',.5),('DELISTED',.5)]:
        rows.append(Fact('ETF','weight:'+symbol,weight,'2024-03-31','2024-05-01T00:00:00Z','2024-05-01T00:00:00Z','fixture','holdings1','','fraction',kind='holding',metadata={'constituent':symbol}))
    s.ingest(rows);snap=FundamentalFeatureEngine(s).etf_snapshot('ETF','2024-06-01T00:00:00Z','net_margin')
    assert snap['coverage']==.5 and snap['value'] is None


def test_mutable_news_not_backdated_and_evidence_required():
    from fundamentals.catalysts import ingest_document_annotations
    s=PointInTimeFundamentalStore();doc='Management reduced guidance.'
    annotation={'tag':'guidance_down','start':0,'end':len(doc),'quote':doc,'model_version':'fixture-v1'}
    ingest_document_annotations(s,symbol='TEST',document=doc,source='fixture:news',published_at='2024-01-01T00:00:00Z',observed_at='2024-02-01T00:00:00Z',annotations=[annotation])
    assert not s.events('TEST','2024-01-15T00:00:00Z')
    event=s.events('TEST','2024-02-02T00:00:00Z')[0];assert event['payload']['execution_authority'] is False
    with pytest.raises(ValueError):
        ingest_document_annotations(s,symbol='TEST',document=doc,source='fixture:news',published_at='2024-01-01T00:00:00Z',observed_at='2024-02-01T00:00:00Z',annotations=[{**annotation,'quote':'invented'}])


def test_append_only_database():
    s=PointInTimeFundamentalStore();s.ingest([fact()])
    with pytest.raises(sqlite3.IntegrityError):s.db.execute('UPDATE observations SET value=999')
    with pytest.raises(sqlite3.IntegrityError):s.db.execute('DELETE FROM observations')


def test_fx_units_and_insider_currencies_cannot_mix():
    from fundamentals.adapters import insider_activity
    s=PointInTimeFundamentalStore()
    assert insider_activity(s,'TEST','2024-06-02T00:00:00Z')['discretionary_purchase_value'] is None
    s.ingest([Fact('FX:USDGBP','rate',.8,'2024-06-01','2024-06-01T00:00:00Z','2024-06-01T00:00:00Z','fixture','A','USD','USD/GBP',kind='fx')])
    with pytest.raises(ValueError,match='quote'):historical_fx(s,100,'USD','GBP','2024-06-02T00:00:00Z')
    a=replace(fact(metric='transactionA',value=100),kind='ownership',metadata={'transaction_code':'P','discretionary':True})
    b=replace(a,metric='transactionB',currency='GBP',units='GBP')
    s.ingest([a,b])
    with pytest.raises(ValueError,match='currency'):insider_activity(s,'TEST','2024-06-02T00:00:00Z')


def test_fundamental_monte_carlo_paths_against_independent_scalar_oracle():
    from fundamentals.research import account_bootstrap
    r=np.array([.01,-.02,0.,.03,-.005]*8);paths=100000;days=21;block=5;seed=431
    actual=account_bootstrap(r,paths=paths,days=days,block=block,seed=seed)
    # Separate matrix construction and cumulative-product computation.
    gen=np.random.default_rng(seed);sample=np.empty((paths,days))
    for start in range(0,days,block):
        indexes=gen.integers(0,len(r),paths)
        for offset in range(min(block,days-start)):sample[:,start+offset]=r[(indexes+offset)%len(r)]
    curves=np.column_stack([np.full(paths,100.),100*np.cumprod(1+sample,axis=1)])
    peaks=np.maximum.accumulate(curves,axis=1);maxdd=np.max(1-curves/peaks,axis=1)
    assert actual==account_bootstrap(r,paths=paths,days=days,block=block,seed=seed)
    assert actual['mean']==pytest.approx(curves[:,-1].mean(),abs=1e-11)
    assert actual['median']==pytest.approx(np.median(curves[:,-1]),abs=1e-11)
    assert actual['mean_max_drawdown']==pytest.approx(maxdd.mean(),abs=1e-13)
    assert actual['end_at_or_above_100']==np.count_nonzero(curves[:,-1]>=100)/paths
    assert actual['hit_200']==np.count_nonzero(curves.max(axis=1)>=200)/paths
    for bad in (0,-1,True,1.2):
        with pytest.raises(ValueError):account_bootstrap(r,paths=bad)


def test_etf_missing_weights_not_zero_and_negative_weights_rejected():
    s=populated_store();a=replace(fact('holding',None,'2024-05-01','2024-05-02T00:00:00Z'),symbol='ETF',kind='holding',currency='',units='weight',metadata={'constituent':'TEST'})
    s.ingest([a]);e=FundamentalFeatureEngine(s)
    assert e.etf_snapshot('ETF','2024-06-01T00:00:00Z','revenue_yoy')['status']=='INCOMPLETE_HOLDING_WEIGHTS'
    s.ingest([replace(a,value=-.5,available_at='2024-05-03T00:00:00Z',reported_at='2024-05-03T00:00:00Z',revision='corrected')])
    with pytest.raises(ValueError,match='negative'):e.etf_snapshot('ETF','2024-06-01T00:00:00Z','revenue_yoy')


def test_old_fx_observation_does_not_become_fresh_on_late_publication():
    s=PointInTimeFundamentalStore()
    s.ingest([Fact('FX:USDGBP','rate',.8,'2024-01-01','2024-06-01T00:00:00Z','2024-06-01T00:00:00Z','fixture','A','GBP','GBP/USD',kind='fx')])
    assert historical_fx(s,100,'USD','GBP','2024-06-02T00:00:00Z')['value'] is None
