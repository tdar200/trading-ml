"""Verify saved data/code, replay the frozen selection, and optionally journal it."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,hashlib,json
import numpy as np
import pandas as pd
from analysis.small_account_research import replay
from strategies.small_account import Setup


def main():
    p=argparse.ArgumentParser(); p.add_argument('--directory',type=Path,default=Path('results/small_account'))
    p.add_argument('--database',type=Path); a=p.parse_args()
    r=json.loads((a.directory/'research.json').read_text())
    for file,digest in r['code_hashes'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=digest: raise SystemExit('Code drift: '+file)
    s=r['selected']; symbol=s['symbol']; path=a.directory/(symbol.replace('/','-')+'.parquet')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=r['data'][symbol]['sha256']: raise SystemExit('Data drift')
    if a.database and a.database.exists(): raise SystemExit('Replay needs a new database; existing journal preserved')
    f=pd.read_parquet(path); crypto=symbol.endswith('GBP')
    e=replay(f,Setup(**s['setup']),symbol=symbol,begin=int(len(f)*.75),end=len(f),
             fee_bps=80. if crypto else 15.,spread_bps=10. if crypto else 2.,slip_bps=5. if crypto else 3.,database=a.database)
    saved=pd.read_csv(a.directory/'equity.csv')
    np.testing.assert_allclose([x['equity'] for x in e.curve],saved.equity,rtol=0,atol=1e-10)
    print(json.dumps({'verified':True,'candidate':s['id'],'final_equity':e.state['equity'],
                      'closed_trades':len(e.state['closed_pnl']),'database':str(a.database)}))

if __name__=='__main__': main()
