"""Run preregistered research, conditional Monte Carlo, and save artifacts."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json
from data.multitf import get_bars
from data.small_account_feed import daily_bars
from analysis.small_account_mc import binary_paths,requirement,exact_terminal_probability
from analysis.small_account_research import experiment


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path('results/small_account'))
    p.add_argument('--paths',type=int,default=100_000); p.add_argument('--skip-crypto',action='store_true')
    p.add_argument('--mc-only',action='store_true'); p.add_argument('--research-only',action='store_true')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    if not a.mc_only:
        data={s:get_bars(s,'1d') for s in ('SPY','XLK','XLE','XLV','XLF','XLP')}
        # Align equities on common observed sessions; never fill the SPY catalog's 634-day hole.
        common=set.intersection(*(set(f.ts) for f in data.values()))
        raw_coverage={s:{'rows':len(f),'max_gap_days':float(f.ts.diff().dt.total_seconds().max()/86400)} for s,f in data.items()}
        data={s:f[f.ts.isin(common)].reset_index(drop=True) for s,f in data.items()}
        (a.output/'data_quality.json').write_text(json.dumps({'raw':raw_coverage,'common_equity_sessions':len(common),'policy':'intersection of observed ETF sessions; no filling; early isolated SPY bar excluded'},indent=2))
        failures={}
        if not a.skip_crypto:
            for s in ('XBTGBP','ETHGBP'):
                try: data[s]=daily_bars(s)
                except Exception as e: failures[s]=type(e).__name__+': '+str(e)
        (a.output/'feed_failures.json').write_text(json.dumps(failures,indent=2))
        report=experiment(data,a.output,a.paths)
        print('Selected research candidate:',report['selected']['id'],'Decision:',report['decision'],flush=True)
    if not a.research_only:
        scenarios=[]
        for pwin,reward in ((.4,2),(.45,2),(.5,1.5),(.55,1.5),(.6,1),(.65,1)):
            for risk in (.0025,.005,.01,.02,.03,.05,.1,.2):
                r=binary_paths(pwin,reward,risk,paths=a.paths)
                r['exact_30day_terminal_10x']=exact_terminal_probability(pwin,reward,risk,21)
                scenarios.append(r)
            print('Simulated win/reward',pwin,reward,flush=True)
        extremes=[binary_paths(.45,2,risk,paths=a.paths,horizons={60:'30_days_60_trades'},stress=stress)
                  for risk in (.05,.1,.2,.5) for stress in (False,True)]
        cost_sensitivity=[binary_paths(.45,2,.01,cost_r=c,paths=a.paths,horizons={21:'30_days'}) for c in (0,.1,.4,.8,1.6)]
        (a.output/'monte_carlo.json').write_text(json.dumps({'requirements':requirement(),'seed':20260930,
            'interpretation':'Assumed stable win rates are NOT estimated edges. 21 sessions per 30 days; 1 trade/session except extreme scenarios. Fixed fractional losses cannot reach zero unless a >=100% loss occurs.',
            'scenarios':scenarios,'extreme_60trades':extremes,'cost_sensitivity':cost_sensitivity},indent=2,allow_nan=False))
    print('Saved',a.output)

if __name__=='__main__': main()
