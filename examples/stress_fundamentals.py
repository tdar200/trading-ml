"""Frozen-model cost sensitivity and paired block-bootstrap increments, no tuning."""
import json,sys,argparse
from dataclasses import fields
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from fundamentals.store import PointInTimeFundamentalStore
from fundamentals.models import ModelArtifact,FundamentalScoreEngine,CombinedSignalEngine,MacroRegimeService
from fundamentals.features import FundamentalFeatureEngine
from fundamentals.research import replay_with_fundamentals
from strategies.small_account import Setup
from analysis.small_account_research import metrics

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/fundamentals/experiment');args=parser.parse_args()
    root=Path(args.output);artifacts=json.loads((root/'frozen_models.json').read_text())
    store=PointInTimeFundamentalStore('results/fundamentals/fundamentals.sqlite',readonly=True)
    rows=[];paired=[]
    for symbol in ('SPY','XLE','XLF','XLK','XLP','XLV'):
        frame=pd.read_parquet(Path('results/small_account')/(symbol+'.parquet'));begin=int(len(frame)*.75)
        models={a['model']:ModelArtifact(**{f.name:a[f.name] for f in fields(ModelArtifact)}) for a in artifacts if a['symbol']==symbol}
        for cost in (1,2):
            curves={}
            for name in ('technical','macro_filter_A_B','joint_C'):
                combo=None if name=='technical' else CombinedSignalEngine(FundamentalFeatureEngine(store),FundamentalScoreEngine(models[name]),MacroRegimeService(store),policy='filter' if name=='macro_filter_A_B' else 'weighted',research_shadow=True)
                e=replay_with_fundamentals(frame,Setup('bollinger',20),symbol,begin,len(frame),combo,cost_multiple=cost)
                rows.append({'symbol':symbol,'system':name,'cost_multiple':cost,**metrics(e)})
                curves[name]=np.diff(np.array([100.,*[p['equity'] for p in e.curve]]))
            # Paired blocks preserve same dates for technical and extended portfolio P&L.
            delta=curves['macro_filter_A_B']-curves['technical'];rng=np.random.default_rng(947);sums=np.zeros(10000)
            for start in range(0,len(delta),20):
                offsets=rng.integers(0,len(delta),len(sums))
                for j in range(min(20,len(delta)-start)):sums+=delta[(offsets+j)%len(delta)]
            paired.append({'symbol':symbol,'cost_multiple':cost,'observed_increment_gbp':float(delta.sum()),
                           'paired_block_95_interval_gbp':np.quantile(sums,[.025,.975]).tolist(),'paths':10000,'block_sessions':20,'seed':947,
                           'note':'Conditional descriptive interval on reused holdout; not corrected for strategy search or multiple tests'})
        print(symbol,flush=True)
    pd.DataFrame(rows).to_csv(root/'cost_sensitivity.csv',index=False)
    (root/'paired_incremental_intervals.json').write_text(json.dumps(paired,indent=2,allow_nan=False));store.close()
