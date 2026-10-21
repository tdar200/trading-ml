"""Repeatable real-source quarantine and held-out calibration diagnostics."""
import sys,json
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from fundamentals.store import PointInTimeFundamentalStore
from fundamentals.providers import ingest_sec
from fundamentals.models import fit_model,FundamentalScoreEngine
from fundamentals.confidence import fit_confidence
from fundamentals.research import label_panel,MACRO_FEATURES,purged_training
from strategies.small_account import features,Setup

out=Path('results/fundamental_closure_20260930');out.mkdir(exist_ok=True,parents=True)
raw=Path('results/fundamental_audit_20260930/raw')
s=PointInTimeFundamentalStore(out/'real_ingestion_v3.sqlite');summary=[]
for symbol in ('MSFT','JPM','WMT','JNJ','CAT','XOM','BA','SONY'):
    cik=json.loads((raw/f'{symbol}_facts.json').read_text())['cik']
    count=ingest_sec(s,symbol,cik,raw)
    q=s.quality(symbol)
    summary.append({'symbol':symbol,'accepted_input_records':count,'quarantined':len(q['rejections'])})
(out/'real_ingestion.json').write_text(json.dumps(summary,indent=2))
# Independently compare all 37 previously checked issuer numbers to the newly
# production-ingested database, using the transcribed expectations as oracle.
checks=pd.read_csv('results/fundamental_audit_20260930/official_reconciliation.csv');diffs=[]
from fundamentals.features import quarters
for r in checks.to_dict('records'):
    rows=s.as_of(r['symbol'],'2021-07-01T00:00:00Z' if r['symbol']=='SONY' else '2025-06-01T00:00:00Z',kind='financial')
    if r['metric'].endswith('_quarter'):
        value=quarters(rows,r['metric'][:-8])[r['period_end']]['value']
    else:
        matches=[x for x in rows if x['metric']==r['metric'] and x['period_end']==r['period_end'] and (not x['period_start'] or x['period_start']==r.get('period_start'))]
        value=matches[0]['value'] if len(matches)==1 else None
    diffs.append({'symbol':r['symbol'],'metric':r['metric'],'expected':r['source_value'],'actual':value,'pass':value==r['source_value']})
assert all(x['pass'] for x in diffs)
# New IFRS samples transcribed from Sony's original FY2025 issuer release.
rows=s.as_of('SONY','2025-07-01T00:00:00Z',kind='financial')
for metric,want in {'revenue':12957064000000,'net_income':1141600000000,'eps':187.92}.items():
    matches=[r for r in rows if r['metric']==metric and r['period_end']=='2025-03-31' and r['period_start']=='2024-04-01']
    actual=matches[0]['value'] if len(matches)==1 else None
    diffs.append({'symbol':'SONY','metric':metric,'expected':want,'actual':actual,'pass':actual==want,'source':'https://www.sony.com/en/SonyInfo/IR/library/presen/er/pdf/24q4_sony.pdf'})
assert all(x['pass'] for x in diffs)
(out/'real_reconciliation.json').write_text(json.dumps(diffs,indent=2));s.close()
# New fixed alpha=1 diagnostic, not reuse of alpha chosen on calibration data.
frame=features(pd.read_parquet('results/small_account/XLE.parquet'),Setup('bollinger',20))
states=json.loads(Path('results/fundamental_audit_20260930/reproduction/macro_states.json').read_text())
panel=label_panel(frame,states);fit_at=frame.ts.iloc[int(len(frame)*.5)];test_at=frame.ts.iloc[int(len(frame)*.75)]
model=fit_model(purged_training(panel,fit_at).dropna(subset=list(MACRO_FEATURES)),MACRO_FEATURES,fit_at=fit_at,alpha=1.,horizon=5)
def disjoint(sample):
    selected=[];end=None
    for _,row in sample.sort_values('as_of').iterrows():
        if end is None or row.as_of>end:selected.append(row);end=row.label_end
    p=pd.DataFrame(selected);p['horizon']=5;p['net_of_costs']=True;return p
cal=disjoint(panel[(panel.as_of>fit_at)&(panel.label_end<test_at)].dropna(subset=list(MACRO_FEATURES)))
artifact=fit_confidence(model,cal,calibrated_at=test_at)
engine=FundamentalScoreEngine(model,artifact)
test=disjoint(panel[panel.as_of>test_at].dropna(subset=list(MACRO_FEATURES)))
records=[]
for row in test.to_dict('records'):
    value=engine.score(row,row['as_of']);records.append({'as_of':str(row['as_of']),'probability':value['confidence'],'outcome':int(row['target']>0)})
valid=[r for r in records if r['probability'] is not None]
brier=sum((r['probability']-r['outcome'])**2 for r in valid)/len(valid)
report={'model':asdict(model),'calibration':asdict(artifact),'calibration_id':artifact.id,'evaluation':records,'evaluation_brier':brier,'evaluation_n':len(valid),'scope':'XLE macro model diagnostic, previously researched history; not corporate alpha validation','deployment_approved':False}
(out/'confidence_diagnostic.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'source_checks':len(diffs),'source_checks_pass':True,'ingestion':summary,'calibration_n':artifact.observations,'evaluation_n':len(valid),'evaluation_brier':brier},indent=2))
