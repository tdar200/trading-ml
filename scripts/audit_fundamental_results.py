"""Independent numerical audit of recorded fundamental research artifacts."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from tests import fundamental_oracle as oracle
from fundamentals.features import accounting_features,valuation_features
root=Path('results/fundamental_audit_20260930');rep=root/'reproduction'
rows=[];gold=json.loads(Path('tests/fixtures/fundamental_companies.json').read_text())['companies']
for name,c in gold.items():
    expected=oracle.calculate(c['current'],c['prior']);actual=accounting_features(c['current'],c['prior'])[0]
    expected.update(oracle.valuation(c['current'],c['price'],c['shares'],c['forward_eps']))
    actual.update(valuation_features(c['price'],c['shares'],c['current'],price_currency='GBP',statement_currency='GBP',price_basis='A',share_basis='A',forward_eps=c['forward_eps']))
    for k,want in expected.items():
        if k=='revenue_acceleration':continue
        got=actual[k];delta=None if want is None or got is None else got-want
        good=got is None if want is None else got is not None and abs(delta)<=max(1e-12,abs(want)*1e-12)
        rows.append({'company':name,'metric':k,'oracle':want,'actual':got,'difference':delta,'status':'PASS' if good else 'FAIL'})
pd.DataFrame(rows).to_csv(root/'formula_reconciliation.csv',index=False)
original=json.loads(Path('results/fundamentals/experiment/research.json').read_text());current=json.loads((rep/'research.json').read_text());checks=[]
for name,record in current['selected_account_experiment'].items():
    curve=pd.read_csv(rep/f'{name}_equity.csv');journal=json.loads((rep/f'{name}_journal.json').read_text())
    calculated=oracle.performance(curve.equity.tolist(),journal)
    for metric,value in calculated.items():
        reported=original['selected_account_experiment'][name]['metrics'][metric]
        difference=None if value is None or reported is None else value-reported
        good=value is None and reported is None or difference is not None and abs(difference)<=1e-10
        checks.append({'system':name,'metric':metric,'reported':reported,'independent':value,'difference':difference,'status':'PASS' if good else 'FAIL'})
pd.DataFrame(checks).to_csv(root/'headline_claim_reconciliation.csv',index=False)
comparison=[]
for a,b in zip(original['results'],current['results']):
    assert (a['symbol'],a['system'])==(b['symbol'],b['system'])
    numeric={k:abs(a[k]-b[k]) for k in a if isinstance(a[k],(float,int)) and not isinstance(a[k],bool) and b.get(k) is not None}
    comparison.append({'symbol':a['symbol'],'system':a['system'],'maximum_numeric_difference':max(numeric.values()),'PASS':max(numeric.values())<1e-10})
(root/'all_ablation_reproduction.json').write_text(json.dumps(comparison,indent=2))
# Verify frozen model labels against the actual interval, not descriptive names.
intervals=[]
for m in current['models']:
    good=pd.Timestamp(m['train_last_label'])<pd.Timestamp(m['trained_through'])<pd.Timestamp(m['validation_last_label'])<pd.Timestamp(m['test_start'])
    intervals.append({'symbol':m['symbol'],'model':m['model'],'train_last_label':m['train_last_label'],'fit_at':m['trained_through'],'validation_last_label':m['validation_last_label'],'diagnostic_start':m['test_start'],'PASS':good})
(root/'model_interval_audit.json').write_text(json.dumps(intervals,indent=2))
# Randomly chosen actual multi-year macro decision times: every dependency must already exist.
from fundamentals.store import PointInTimeFundamentalStore
states=json.loads((rep/'macro_states.json').read_text());s=PointInTimeFundamentalStore('results/fundamentals/fundamentals.sqlite',readonly=True)
import random
samples=[]
for at in random.Random(2481).sample(list(states),100):
    dep=states[at]['lineage'];rows_=list(s.db.execute('SELECT id,available_at FROM observations WHERE id IN ('+','.join('?' for _ in dep)+')',dep))
    good=len(rows_)==len(set(dep)) and all(pd.Timestamp(r['available_at'])<=pd.Timestamp(at) for r in rows_)
    samples.append({'as_of':at,'dependency_count':len(rows_),'PASS':good})
(root/'macro_lineage_samples.json').write_text(json.dumps(samples,indent=2));s.close()
assert all(r['status']=='PASS' for r in rows+checks)
assert all(r['PASS'] for r in intervals+comparison+samples)
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*Path('fundamentals').glob('*.py'),Path('tests/fundamental_oracle.py'),Path('tests/fixtures/fundamental_companies.json')]}
(root/'audited_code_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'formula_checks':len(rows),'independent_headline_checks':len(checks),'reproduced_configurations':len(comparison),'model_splits':len(intervals),'historical_macro_samples':len(samples),'all_passed':True},indent=2))
