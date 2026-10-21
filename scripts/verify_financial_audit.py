"""Recompute archived claims with an independent oracle and rerun frozen data.

No downloads, credentials, or broker orders. Outputs stay separate from originals.
"""
import ast,hashlib,json,sys,time,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from tests.financial_oracle import Ledger,statistics
from analysis.small_account_research import experiment

ROOT=Path('results/qa_audit_20260930'); ROOT.mkdir(exist_ok=True)
original=Path('results/small_account')
inventory=[]
for p in sorted(Path('.').rglob('*.py')):
    if any(x.startswith('.') or x in ('results','__pycache__') for x in p.parts): continue
    source=p.read_text(); tree=ast.parse(source)
    inventory.append(dict(path=str(p),lines=len(source.splitlines()),sha256=hashlib.sha256(source.encode()).hexdigest(),
                          imports=sorted({n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}|{a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}),
                          functions=[n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]))
(ROOT/'dependency_inventory.json').write_text(json.dumps(inventory,indent=2))
report=json.loads((original/'research.json').read_text()); eq=pd.read_csv(original/'equity.csv'); journal=pd.read_csv(original/'journal.csv')
ledger=Ledger(); ledgerrows=[]
for event in journal.to_dict('records'):
    if event['event']=='ENTRY': ledger.buy(event['quantity'],event['price'],event['fee'])
    if event['event']=='EXIT':
        expected=float(ledger.sell(event['quantity'],event['price'],event['fee']))
        ledgerrows.append(dict(ts=event['timestamp'],expected_pnl=expected,reported_pnl=event['pnl'],difference=event['pnl']-expected,cash=float(ledger.cash)))
expected=statistics([100,*eq.equity],journal[journal.event=='EXIT'].pnl)
claims=[]
for key,value in expected.items():
    actual=report['selected']['diagnostic'][key]
    difference=actual-value if actual is not None and value is not None else None
    claims.append(dict(metric=key,reported=actual,oracle=value,difference=difference,result='PASS' if difference is None and actual==value or difference is not None and abs(difference)<1e-8 else 'FAIL'))
pd.DataFrame(claims).to_csv(ROOT/'original_claims.csv',index=False)
pd.DataFrame(ledgerrows).to_csv(ROOT/'original_ledger.csv',index=False)
assert abs(float(ledger.cash)-eq.equity.iloc[-1])<1e-8
print('Original accounting claims recomputed; now rerunning same eight snapshots.',flush=True)
start=time.perf_counter()
data={p.stem:pd.read_parquet(p) for p in sorted(original.glob('*.parquet'))}
revised=experiment(data,ROOT/'corrected_research',paths=100000)
(ROOT/'research_runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,candidates=len(revised['candidates']),selected=revised['selected']['id'],metrics=revised['selected']['diagnostic']),indent=2))
print(json.dumps(dict(selected=revised['selected']['id'],eligible=revised['eligible_count'],metrics=revised['selected']['diagnostic']),indent=2))
