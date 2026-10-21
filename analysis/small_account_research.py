"""Preregistered family comparison on reused history; never true unseen OOS."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from strategies.small_account import Setup,features
from livesim.small_account import PaperEngine
from analysis.small_account_mc import block_paths


def metrics(engine, annual=252):
    f=pd.DataFrame(engine.curve)
    if f.empty: raise ValueError('empty evaluation')
    eq=np.r_[100.,f.equity.to_numpy()]; r=eq[1:]/eq[:-1]-1
    dd=1-eq/np.maximum.accumulate(eq); n=len(r); std=r.std(ddof=1) if n>1 else 0
    downside=np.sqrt(np.mean(np.minimum(r,0)**2)); exits=[e for e in engine.journal if e['event']=='EXIT']
    pnl=np.array([e['pnl'] for e in exits]); wins=pnl[pnl>0]; losses=pnl[pnl<0]
    cagr=float((eq[-1]/100)**(annual/n)-1)
    duration=0; recovery=0
    for d in dd:
        duration=duration+1 if d>1e-10 else 0; recovery=max(recovery,duration)
    turnover=sum(e.get('price',0)*e.get('quantity',0) for e in engine.journal if e['event'] in ('ENTRY','EXIT'))/100
    return {'annualization_periods':annual,'annualization_convention':'per observed return; not elapsed-calendar CAGR','final_equity':float(eq[-1]),'total_return':float(eq[-1]/100-1),'cagr':cagr,
        'monthly_geometric':float((eq[-1]/100)**(annual/(12*n))-1),'daily_mean':float(r.mean()),
        'sharpe':float(r.mean()/std*np.sqrt(annual)) if std>0 else None,
        'sortino':float(r.mean()/downside*np.sqrt(annual)) if downside>0 else None,
        'calmar':cagr/float(dd.max()) if dd.max()>0 else None,
        'profit_factor':float(wins.sum()/-losses.sum()) if losses.sum()<0 else None,
        'expectancy_gbp':float(pnl.mean()) if len(pnl) else None,
        'win_rate':float(np.mean(pnl>0)) if len(pnl) else None,
        'average_win':float(wins.mean()) if len(wins) else None,
        'average_loss':float(losses.mean()) if len(losses) else None,
        'reward_risk_realized':float(wins.mean()/-losses.mean()) if len(wins) and losses.sum()<0 else None,
        'trades':len(exits),'exposure':float(f.exposure.mean()),'turnover':turnover,
        'fees_gbp':sum(e.get('fee',0) for e in engine.journal),
        'max_drawdown':float(dd.max()),'average_drawdown':float(dd.mean()),
        'longest_underwater_observations':recovery,'observations':n}


def replay(frame,setup,*,symbol,begin,end,fee_bps=15.,slip_bps=3.,spread_bps=2.,database=None,delay=1):
    if delay<1: raise ValueError('execution must follow completed signal')
    engine=PaperEngine(setup,symbol=symbol,fee_bps=fee_bps,slip_bps=slip_bps,spread_bps=spread_bps,shadow=True,database=database)
    records=features(frame,setup,annual=365 if symbol.endswith('GBP') else 252).to_dict('records')
    for i in range(begin,end):
        engine.step(records[i],records[i-delay] if i>=delay else None,close_final=i==end-1)
    return engine


def confidence_lower(returns, seed=3, block=10, paths=4000):
    r=np.asarray(returns); rng=np.random.default_rng(seed); means=np.zeros(paths)
    for t in range(0,len(r),block):
        starts=rng.integers(0,len(r),paths)
        for j in range(min(block,len(r)-t)): means+=r[(starts+j)%len(r)]
    return float(np.quantile(means/len(r),.01))


def experiment(data,output:Path,paths=100_000):
    output.mkdir(parents=True,exist_ok=True)
    setups=[Setup(family=f,period=p) for f in ('donchian','ma','rsi','bollinger','volume','gap') for p in (18,20,22)]
    prereg={'setups':[asdict(x) for x in setups], 'split':'50% train, next 25% validation (3 chronological windows), final25% diagnostic only',
        'selection':'positive in all validation windows, >=12 trades, positive 99% block mean lower bound; rank validation Sharpe minus drawdown',
        'warning':'previously researched historical samples; no true unseen OOS; all deployment disabled',
        'starting_gbp':100,'base_equity_cost':'15bp FX/side + 2bp spread + 3bp slippage/side; constant GBPUSD proxy',
        'base_crypto_cost':'80bp taker/side + 10bp spread + 5bp slippage/side; actual GBP spot', 'paths':paths}
    (output/'preregistration.json').write_text(json.dumps(prereg,indent=2))
    rows=[]; validation_returns={}; engines={}; manifests={}
    for symbol,frame in data.items():
        frame=frame.sort_values('ts').reset_index(drop=True)
        snap=output/(symbol.replace('/','-')+'.parquet'); frame.to_parquet(snap,index=False)
        manifests[symbol]={'sha256':hashlib.sha256(snap.read_bytes()).hexdigest(),'rows':len(frame),'start':str(frame.ts.min()),'end':str(frame.ts.max())}
        n=len(frame); train_end=int(n*.5); val_end=int(n*.75)
        if train_end<100: continue
        crypto=symbol.endswith('GBP'); kwargs={'fee_bps':80. if crypto else 15.,'spread_bps':10. if crypto else 2.,'slip_bps':5. if crypto else 3.}
        annual=365 if crypto else 252
        for setup in setups:
            key=symbol+'_'+setup.family+'_'+str(setup.period)
            train=replay(frame,setup,symbol=symbol,begin=60,end=train_end,**kwargs)
            val=replay(frame,setup,symbol=symbol,begin=train_end,end=val_end,**kwargs)
            m=metrics(val,annual); vr=pd.Series(np.r_[100.,[c['equity'] for c in val.curve]]).pct_change().dropna().to_numpy()
            cuts=np.linspace(train_end,val_end,4,dtype=int); folds=[]
            for a,b in zip(cuts[:-1],cuts[1:]):
                folds.append(metrics(replay(frame,setup,symbol=symbol,begin=int(a),end=int(b),**kwargs),annual)['total_return'])
            lower=confidence_lower(vr)
            eligible=min(folds)>0 and m['trades']>=12 and lower>0
            row={'id':key,'symbol':symbol,'setup':asdict(setup),'train':metrics(train,annual),'validation':m,
                 'fold_returns':folds,'mean_return_99pct_lower':lower,'eligible':eligible,
                 'score':(m['sharpe'] if m['sharpe'] is not None else -100)-m['max_drawdown']}
            rows.append(row); validation_returns[key]=vr
        print('Compared',symbol,len(frame),'bars',flush=True)
    if not rows: raise ValueError('no usable historical datasets')
    ordered=sorted(rows,key=lambda x:x['score'],reverse=True)
    eligible=[r for r in ordered if r['eligible']]
    selected=(eligible or ordered)[0]
    # Freeze selection before opening any final-quarter diagnostics.
    (output/'selection.json').write_text(json.dumps({'candidate':selected,'deployment_approved':False,'selected_before_diagnostic':True},indent=2))
    for row in rows:
        symbol=row['symbol']; frame=data[symbol]; n=len(frame); crypto=symbol.endswith('GBP')
        e=replay(frame,Setup(**row['setup']),symbol=symbol,begin=int(n*.75),end=n,
                 fee_bps=80. if crypto else 15.,spread_bps=10. if crypto else 2.,slip_bps=5. if crypto else 3.)
        row['diagnostic']=metrics(e,365 if crypto else 252)
        if row['id']==selected['id']: engines['selected']=e
    best=engines['selected']; eq=pd.DataFrame(best.curve); eq.to_csv(output/'equity.csv',index=False)
    pd.DataFrame(best.journal).to_csv(output/'journal.csv',index=False)
    (output/'state.json').write_text(json.dumps(best.state,indent=2))
    symbol=selected['symbol']; frame=data[symbol]; n=len(frame); setup=Setup(**selected['setup']); crypto=symbol.endswith('GBP')
    stress=[]
    for multiple,delay in ((1,1),(2,1),(4,1),(1,2)):
        e=replay(frame,setup,symbol=symbol,begin=int(n*.75),end=n,fee_bps=(80. if crypto else 15.)*multiple,
                 spread_bps=10. if crypto else 2.,slip_bps=min(20.,(5. if crypto else 3.)*multiple),delay=delay)
        stress.append({'cost_multiple':multiple,'delay_bars':delay,'metrics':metrics(e,365 if crypto else 252)})
    exit_comparison=[]
    # Exit experiments stay on development data, never tune selected diagnostic.
    for stop in ('fixed','atr','trailing','structure'):
        for rr in (1.,1.5,2.,3.):
            variant=Setup(setup.family,setup.period,setup.hold,stop,rr)
            e=replay(frame,variant,symbol=symbol,begin=int(n*.5),end=int(n*.75),fee_bps=80. if crypto else 15.,spread_bps=10. if crypto else 2.,slip_bps=5. if crypto else 3.)
            exit_comparison.append({'stop':stop,'reward':rr,'metrics':metrics(e,365 if crypto else 252)})
    returns=pd.Series(np.r_[100.,eq.equity.to_numpy()]).pct_change().dropna().to_numpy()
    mc={'base':block_paths(returns,paths=paths,annual=365 if crypto else 252),
        'stress':block_paths(returns,paths=paths,annual=365 if crypto else 252,stress=True)}
    code_files=['strategies/small_account.py','livesim/small_account.py','livesim/small_account_risk.py','analysis/small_account_research.py','analysis/small_account_mc.py','livesim/small_account_accounting.py','analysis/rule_discovery.py']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in code_files}
    report={'code_hashes':hashes,'preregistration':prereg,'data':manifests,'candidates':rows,'selected':selected,'cost_stress':stress,
        'exit_development_only':exit_comparison,'monte_carlo':mc,'deployment_approved':False,
        'decision':'NO_TRADE','reason':'No independent forward evidence; historical proxies and selection uncertainty remain',
        'eligible_count':len(eligible)}
    (output/'research.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report
