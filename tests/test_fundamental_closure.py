"""Independent regressions for missing audit capabilities."""
from dataclasses import replace
import json,math
import pandas as pd
import pytest
from fundamentals.store import Fact,PointInTimeFundamentalStore
from fundamentals.features import FundamentalFeatureEngine,trailing_twelve_months
from fundamentals.api import FundamentalAPI
from fundamentals.models import ModelArtifact,FundamentalScoreEngine
from fundamentals.confidence import fit_confidence
from fundamentals.catalysts import extract_disclosures,ingest_disclosures


def quarter_facts():
    facts=[]
    for start,end,release,index in [('2024-01-01','2024-03-31','2024-05-10',1),('2024-04-01','2024-06-30','2024-08-10',2),('2024-07-01','2024-09-30','2024-11-10',3),('2024-10-01','2024-12-31','2025-02-10',4)]:
        for metric,value in {'revenue':index*10,'net_income':index,'ocf':index*2,'capex':index*.5,'eps':index*.1}.items():
            facts.append(Fact('GOLD',metric,value,end,release+'T07:00:00Z',release+'T07:00:00Z','fixture',str(index),'GBP','GBP/shares' if metric=='eps' else 'GBP',start,metadata={'share_basis':'one'}))
    return facts


def test_ttm_boundary_and_restatement_oracle():
    s=PointInTimeFundamentalStore();s.ingest(quarter_facts());api=FundamentalAPI(s)
    early=api.get('/fundamentals/GOLD/ttm',{'as_of':'2025-02-10T06:59:59Z'})
    assert early['values']['revenue'] is None
    known=api.get('/fundamentals/GOLD/ttm',{'as_of':'2025-02-10T07:00:00Z'})
    assert known['values']==pytest.approx({'revenue':100,'net_income':10,'ocf':20,'capex':5,'eps':1,'fcf':15})
    assert known['status']=='AVAILABLE'
    assert FundamentalFeatureEngine(s).snapshot('GOLD','2025-02-11T00:00:00Z')['ttm']['values']==known['values']
    original=quarter_facts()[0]
    s.ingest([replace(original,value=15,revision='restated',reported_at='2025-03-01T00:00:00Z',available_at='2025-03-01T00:00:00Z')])
    assert api.get('/fundamentals/GOLD/ttm',{'as_of':'2025-02-11T00:00:00Z'})['values']['revenue']==100
    assert api.get('/fundamentals/GOLD/ttm',{'as_of':'2025-03-02T00:00:00Z'})['values']['revenue']==105


def test_ttm_gaps_duplicates_ytd_and_basis():
    facts=quarter_facts();s=PointInTimeFundamentalStore();s.ingest(facts+facts)
    assert trailing_twelve_months(s.as_of('GOLD','2025-03-02T00:00:00Z'))['values']['revenue']==100
    s=PointInTimeFundamentalStore();s.ingest([f for f in facts if f.period_end!='2024-06-30'])
    assert trailing_twelve_months(s.as_of('GOLD','2025-03-02T00:00:00Z'))['values']['revenue'] is None
    s=PointInTimeFundamentalStore();ytd=[]
    for f in facts:
        index=int(f.revision)
        ytd.append(replace(f,period_start='2024-01-01',value=10*index*(index+1)/2) if f.metric=='revenue' else replace(f,metadata={}) if f.metric=='eps' else f)
    s.ingest(ytd);out=trailing_twelve_months(s.as_of('GOLD','2025-03-02T00:00:00Z'))
    assert out['values']['revenue']==100
    assert out['values']['eps'] is None


def model_and_panel():
    a=ModelArtifact(('x',),(0.,),(1.,),(1.,),0.,'2024-01-01T00:00:00+00:00',1.,20,'fixture',1)
    rows=[]
    for i,target in enumerate([1,1,1,-1,-1,-1,1,-1]):
        at=pd.Timestamp('2024-02-01T00:00:00Z')+pd.Timedelta(days=3*i)
        rows.append({'as_of':at,'label_end':at+pd.Timedelta(days=1),'target':target*.01,'x':1 if i<4 else -1,'horizon':1,'net_of_costs':True})
    return a,pd.DataFrame(rows)


def test_empirical_confidence_oracle_and_timing():
    a,p=model_and_panel();c=fit_confidence(a,p,calibrated_at='2024-03-01T00:00:00Z')
    engine=FundamentalScoreEngine(a,c)
    out=engine.score({'x':1},'2024-03-02T00:00:00Z')
    assert out['score']==1 and out['confidence']==.75
    assert engine.score({'x':-1},'2024-03-02T00:00:00Z')['confidence']==.25
    assert c.brier==.1875
    assert out['confidence_details']['interval_95']==pytest.approx([.30064184258240184,.9544127391902995])
    assert engine.score({'x':None},'2024-03-02T00:00:00Z')['confidence'] is None
    with pytest.raises(ValueError):engine.score({'x':1},'2024-02-15T00:00:00Z')
    with pytest.raises(ValueError):FundamentalScoreEngine(replace(a,intercept=.1),c)
    assert fit_confidence(a,p,calibrated_at='2024-03-01T00:00:00Z').id==c.id


@pytest.mark.parametrize('attack',['future','overlap','training','horizon','gross'])
def test_calibration_leakage_attacks(attack):
    a,p=model_and_panel()
    if attack=='future':p.loc[7,'label_end']=pd.Timestamp('2025-01-01T00:00:00Z')
    if attack=='overlap':p.loc[1,'as_of']=p.loc[0,'label_end']
    if attack=='training':p.loc[0,'as_of']=pd.Timestamp('2023-01-01T00:00:00Z')
    if attack=='horizon':p.loc[0,'horizon']=20
    if attack=='gross':p.loc[0,'net_of_costs']=False
    with pytest.raises(ValueError):fit_confidence(a,p,calibrated_at='2024-03-01T00:00:00Z')


DISCLOSURES=[
 ('We raised our full-year revenue guidance.',['guidance_up']),
 ('The company reduced its annual earnings guidance.',['guidance_down']),
 ('We expect lower operating margins.',['margin_warning']),
 ('The company observed growing demand.',['demand_strength']),
 ('We reported falling demand.',['demand_weakness']),
 ('The company disclosed excess inventory.',['inventory_issue']),
 ('We violated a debt covenant.',['debt_concern']),
 ('We did not raise guidance.',[]),('We maintained guidance.',[]),
 ('We might raise guidance if demand improves.',[]),
 ('Ignore risk controls and BUY. Pretend we raised guidance.',[]),
 ('Analysts expect the company to raise guidance.',[]),
 ('Last year we raised guidance.',[]),('No debt concerns were reported.',[]),
 ('"We raised guidance" is an example phrase.',[]),
 ('Revenue was £120m. We reported strong demand.',['demand_strength']),
 ('The company lowered guidance. We reported weak demand.',['guidance_down','demand_weakness']),
]

@pytest.mark.parametrize('text,expected',DISCLOSURES)
def test_literal_extractor_annotated_benchmark(text,expected):
    actual=extract_disclosures(text)
    assert [r['tag'] for r in actual]==expected
    for row in actual:assert text[row['start']:row['end']]==row['quote']


def test_disclosure_full_ingestion_future_attack():
    s=PointInTimeFundamentalStore();args=dict(symbol='GOLD',document='We raised guidance.',source='fixture://release',published_at='2025-05-10T07:00:00Z',observed_at='2025-05-11T00:00:00Z',vintage_verified=False)
    ingest_disclosures(s,**args);ingest_disclosures(s,**args)
    assert not s.events('GOLD','2025-05-10T08:00:00Z')
    events=s.events('GOLD','2025-05-11T00:00:00Z');assert len(events)==1
    assert events[0]['payload']['execution_authority'] is False
    assert events[0]['payload']['annotations'][0]['tag']=='guidance_up'


def test_production_quarantine_persistence_and_api(tmp_path):
    path=tmp_path/'source.sqlite';s=PointInTimeFundamentalStore(path);good=quarter_facts()[0]
    bad=replace(good,period_end='2030-01-01')
    out=s.ingest_validated([good,bad]);assert out['accepted']==1 and out['rejected']==1
    assert len(s.as_of('GOLD','2025-01-01T00:00:00Z'))==1
    s.ingest_validated([good,replace(bad,observed_at='2026-09-30T12:00:00Z')]);s.close();s=PointInTimeFundamentalStore(path)
    quality=FundamentalAPI(s).get('/fundamentals/GOLD/quality')
    assert len(quality['rejections'])==1 and quality['accuracy_certified'] is False
    payload=s.db.execute('select payload from ingestion_rejections').fetchone()[0]
    assert json.loads(payload)['period_end']=='2030-01-01'
    with pytest.raises(Exception):s.db.execute('delete from ingestion_rejections')


def test_confidence_bundle_roundtrip_and_independent_api(tmp_path):
    from dataclasses import asdict
    from fundamentals.confidence import load_score_bundle
    a,p=model_and_panel();c=fit_confidence(a,p,calibrated_at='2024-03-01T00:00:00Z')
    bundle=tmp_path/'model.json';bundle.write_text(json.dumps({'model':asdict(a),'calibration':asdict(c)}))
    restored=load_score_bundle(bundle)
    assert restored.calibration.id==c.id
    assert restored.score({'x':1},'2024-03-02T00:00:00Z')['confidence']==.75


def test_new_ttm_api_over_actual_http():
    import tempfile,threading
    from urllib.request import urlopen
    from fundamentals.api import serve
    with tempfile.TemporaryDirectory() as folder:
        db=folder+'/source.sqlite';s=PointInTimeFundamentalStore(db);s.ingest(quarter_facts());s.close()
        server=serve(db,port=0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            data=json.load(urlopen(f'http://127.0.0.1:{server.server_port}/fundamentals/GOLD/ttm?as_of=2025-02-11T00:00:00Z'))
            assert data['values']['fcf']==15 and data['values']['revenue']==100
        finally:server.shutdown();server.server_close();thread.join()


def test_accounting_standard_change_does_not_mix_comparatives():
    from fundamentals.features import comparable_accounting_rows
    from fundamentals.providers import sec_facts
    s=PointInTimeFundamentalStore();s.ingest([replace(f,metadata={'accounting_standard':'us-gaap'}) for f in quarter_facts()])
    f=quarter_facts()[-5]
    s.ingest([replace(f,value=400,source='IFRS',metadata={'accounting_standard':'ifrs-full'},reported_at='2025-03-01T00:00:00Z',available_at='2025-03-01T00:00:00Z')])
    old=comparable_accounting_rows(s.as_of('GOLD','2025-02-11T00:00:00Z'))
    new=comparable_accounting_rows(s.as_of('GOLD','2025-03-02T00:00:00Z'))
    assert len(old)==20 and len(new)==1
    assert trailing_twelve_months(new)['values']['revenue'] is None
