"""Timestamp-purged research, IC and common-cost paper ablations.

The pre-existing ETF history is reused: temporal holdout is not pristine OOS.
Corporate ETF ablations are unavailable without dated constituent coverage.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib,json,math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from fundamentals.models import fit_model,FundamentalScoreEngine,CombinedSignalEngine,MacroRegimeService
from fundamentals.features import FundamentalFeatureEngine
from strategies.small_account import features,Setup
from livesim.small_account import PaperEngine
from analysis.small_account_research import metrics

HORIZONS=(1,5,20,60,120)
MACRO_FEATURES=('macro_inflation_yoy','macro_unemployment_change_3m','macro_policy_rate_change_3m')


def _finite_or_none(value):
    """Recursively map NaN/inf to None so the journal is strict JSON (what pandas to_json used to do)."""
    if isinstance(value,float):return value if math.isfinite(value) else None
    if isinstance(value,dict):return {k:_finite_or_none(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_finite_or_none(v) for v in value]
    return value


def label_panel(frame,macro_states,horizon=5,cost=.0038):
    f=frame.reset_index(drop=True);rows=[]
    for i in range(len(f)-horizon-1):
        # Decision at next opening timestamp; technical signal uses previous close.
        at=pd.Timestamp(f.ts.iloc[i+1]); state=macro_states[str(at)]
        end=i+1+horizon
        rows.append({'as_of':at,'label_end':pd.Timestamp(f.ts.iloc[end]),
                     'target':float(f.open.iloc[end]/f.open.iloc[i+1]-1-cost),
                     'technical_signal':float(f.signal.iloc[i]),
                     **{'macro_'+k:v for k,v in state['features'].items()}})
    return pd.DataFrame(rows)


def purged_training(panel,cutoff):
    return panel[pd.to_datetime(panel.label_end,utc=True)<pd.Timestamp(cutoff)].copy()


def information_coefficients(panel,names,start,end):
    sample=panel[(panel.as_of>=pd.Timestamp(start))&(panel.label_end<pd.Timestamp(end))]
    rows=[]
    for name in names:
        f=sample[[name,'target']].dropna() if name in sample else pd.DataFrame()
        ic=float(spearmanr(f[name],f.target).statistic) if len(f)>=20 and f[name].nunique()>1 else None
        rows.append({'feature':name,'observations':len(f),'time_series_spearman_ic':ic,
                     'cross_sectional_ic':None,'note':'Overlapping targets; descriptive IC, no IID significance claim'})
    return rows


def cross_sectional_ic(panel,score='score',minimum=20):
    """PIT membership must be supplied. Deciles require at least 20 names/date."""
    required={'as_of','symbol','label_end','target',score,'member_known_at'}
    if not required.issubset(panel):raise ValueError('PIT universe and labels required')
    if (pd.to_datetime(panel.member_known_at,utc=True)>pd.to_datetime(panel.as_of,utc=True)).any():raise ValueError('future universe membership')
    out=[]
    for timestamp,frame in panel.groupby('as_of'):
        f=frame.dropna(subset=[score,'target'])
        if f.symbol.duplicated().any():raise ValueError('duplicate security in cross-section')
        if len(f)<minimum or f[score].nunique()<2:continue
        n=max(1,len(f)//10);rank=f.sort_values(score)
        out.append({'as_of':str(timestamp),'count':len(f),'rank_ic':float(spearmanr(f[score],f.target).statistic),
                    'top_minus_bottom_decile':float(rank.target.iloc[-n:].mean()-rank.target.iloc[:n].mean())})
    return out


def account_bootstrap(returns,paths=100000,seed=1941,days=21,block=5):
    if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in (paths,days,block)):
        raise ValueError('positive integer paths, days and block required')
    r=np.asarray(returns,float)
    if r.ndim!=1:raise ValueError('one-dimensional returns required')
    if len(r)<30 or not np.isfinite(r).all() or (r < -1).any():raise ValueError('invalid net returns')
    rng=np.random.default_rng(seed);eq=np.full(paths,100.);peak=eq.copy();low=eq.copy();dd=np.zeros(paths)
    for i in range(days):
        if i%block==0:offset=rng.integers(0,len(r),paths)
        eq*=1+r[(offset+i%block)%len(r)];peak=np.maximum(peak,eq);low=np.minimum(low,eq);dd=np.maximum(dd,1-eq/peak)
    return {'paths':paths,'seed':seed,'sessions':days,'median':float(np.median(eq)),'mean':float(eq.mean()),
            'mean_max_drawdown':float(dd.mean()),'p95_max_drawdown':float(np.quantile(dd,.95)),
            'ruin_exact_zero':float(np.mean(low<=0)),'operational_ruin_below_1':float(np.mean(low<1)),
            'end_at_or_below_50':float(np.mean(eq<=50)),'end_at_or_above_100':float(np.mean(eq>=100)),
            **{f'hit_{x}':float(np.mean(peak>=x)) for x in (200,500,1000)},
            'zero_event_one_sided_95_upper':float(1-.05**(1/paths)),
            'interpretation':'Conditional block bootstrap of sparse reused-history net returns; not a forward forecast'}


def replay_with_fundamentals(frame,setup,symbol,begin,end,combo=None,database=None,cost_multiple=1):
    records=features(frame,setup).to_dict('records')
    e=PaperEngine(setup,symbol=symbol,fee_bps=15.*cost_multiple,spread_bps=2.,slip_bps=3.*cost_multiple,
                  shadow=True,database=database,fundamental_filter=combo)
    for i in range(begin,end):
        previous={**records[i-1],'asset_class':'etf'} if i else None
        e.step(records[i],previous,close_final=i==end-1)
    return e


def run_etf_experiment(store,data,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    prereg={'setup':asdict(Setup('bollinger',20)), 'horizon_sessions':5,'alpha_grid':[.1,1.,10.,100.],
        'threshold':'predicted forward net return > 0; threshold is economic break-even, not optimized to account target',
        'split':'first 50% training with label-end purge; next25% validation MSE for alpha; final25% temporal diagnostic',
        'models':['technical','fundamental_observe','macro_filter_A_B','joint_C','joint_regime_D'],
        'corporate_etf_model':'UNAVAILABLE without fully covered dated holdings; observation-only parity is not evidence',
        'costs':'15bp per-side fee/FX proxy, 2bp full spread, 3bp per-side slippage; all shared risk limits unchanged',
        'data_reuse':'Previously researched price history; no pristine unseen or forward test',
        'deployment_approved':False,'minimum_feature_history':'20 complete rows and 5 per coefficient; research fitting floor only'}
    (output/'preregistration.json').write_text(json.dumps(prereg,indent=2))
    macro=MacroRegimeService(store); feature_engine=FundamentalFeatureEngine(store)
    # One shared macro snapshot per timestamp, not separately adjusted per asset.
    dates=sorted({str(pd.Timestamp(t)) for frame in data.values() for t in frame.ts})
    states={t:macro.state(t) for t in dates};results=[];ic=[];artifacts=[];selected_outputs={}
    (output/'macro_states.json').write_text(json.dumps(states,indent=2,allow_nan=False))
    for symbol,frame in data.items():
        f=features(frame,Setup('bollinger',20));n=len(f);train_end=int(n*.5);test_start=int(n*.75)
        fit_at=pd.Timestamp(f.ts.iloc[train_end]);test_at=pd.Timestamp(f.ts.iloc[test_start])
        panel=label_panel(f,states);train=purged_training(panel,fit_at)
        validation=panel[(panel.as_of>fit_at)&(panel.label_end<test_at)]
        if len(validation)<30:raise ValueError('insufficient validation interval')
        models={}
        for name,inputs in [('macro_filter_A_B',MACRO_FEATURES),('joint_C',(*MACRO_FEATURES,'technical_signal'))]:
            candidates=[]
            complete_train=train.dropna(subset=list(inputs));complete_val=validation.dropna(subset=list(inputs))
            for alpha in prereg['alpha_grid']:
                a=fit_model(complete_train,inputs,fit_at=fit_at,alpha=alpha,horizon=5)
                x=complete_val[list(inputs)].to_numpy();pred=(x-np.array(a.means))/np.array(a.scales)@np.array(a.coefficients)+a.intercept
                candidates.append((float(np.mean((pred-complete_val.target.to_numpy())**2)),a))
            mse,artifact=min(candidates,key=lambda x:x[0]);models[name]=artifact
            artifact_record={**asdict(artifact),'id':artifact.id,'symbol':symbol,'model':name,'validation_mse':mse,
                'validation_constant_mse':float(np.mean((complete_val.target-complete_train.target.mean())**2)),
                'train_last_label':str(complete_train.label_end.max()),'validation_last_label':str(complete_val.label_end.max()),'test_start':str(test_at)}
            artifacts.append(artifact_record);store.save('model_artifacts',artifact_record)
        # Selection is frozen before any diagnostic execution or metric is inspected.
        (output/'frozen_models.json').write_text(json.dumps(artifacts,indent=2,allow_nan=False))
        for name in prereg['models']:
            combo=None
            if name=='fundamental_observe':combo=CombinedSignalEngine(feature_engine,policy='observe')
            elif name!='technical':
                a=models['joint_C' if name=='joint_regime_D' else name]
                combo=CombinedSignalEngine(feature_engine,FundamentalScoreEngine(a),macro,
                    policy={'macro_filter_A_B':'filter','joint_C':'weighted','joint_regime_D':'regime'}[name],research_shadow=True)
            e=replay_with_fundamentals(f,Setup('bollinger',20),symbol,test_start,n,combo)
            m=metrics(e);r=pd.Series([100.,*[c['equity'] for c in e.curve]]).pct_change().dropna().to_numpy()
            row={'symbol':symbol,'system':name,**m,'independent_forward_evidence':False,'corporate_fundamental_coverage':0.,
                 'fundamental_influence':'none (observation parity)' if name=='fundamental_observe' else 'macro only' if name!='technical' else 'technical only'}
            results.append(row)
            if symbol=='XLE':
                selected_outputs[name]={'metrics':m,'bootstrap_30_days':account_bootstrap(r),'equity':[c['equity'] for c in e.curve]}
                # json.dumps keeps full float precision (pandas to_json rounds to 10 decimals, so metrics rebuilt from the file drifted ~1e-11).
                # Same record layout as before (every record carries every column, null where absent), minus the rounding.
                columns=list(dict.fromkeys(key for record in e.journal for key in record))
                records=[{key:record.get(key) for key in columns} for record in e.journal]
                (output/f'{name}_journal.json').write_text(json.dumps(_finite_or_none(records),indent=2,default=lambda o:o.item() if hasattr(o,'item') else str(o)))
                pd.DataFrame(e.curve).to_csv(output/f'{name}_equity.csv',index=False)
            if name=='fundamental_observe':
                base=next(x for x in results if x['symbol']==symbol and x['system']=='technical')
                assert abs(row['final_equity']-base['final_equity'])<1e-10
        for horizon in HORIZONS:
            p=label_panel(f,states,horizon=horizon)
            for row in information_coefficients(p,MACRO_FEATURES,test_at,pd.Timestamp(f.ts.iloc[-1])+pd.Timedelta(seconds=1)):
                ic.append({'symbol':symbol,'horizon':horizon,**row})
        print('Completed PIT ablations',symbol,flush=True)
    comparisons=[]
    for symbol in data:
        rows=[r for r in results if r['symbol']==symbol];base=next(r for r in rows if r['system']=='technical')
        for row in rows:
            comparisons.append({'symbol':symbol,'system':row['system'],'incremental_return':row['total_return']-base['total_return'],
                'incremental_sharpe':row['sharpe']-base['sharpe'] if row['sharpe'] is not None and base['sharpe'] is not None else None,
                'drawdown_reduction':base['max_drawdown']-row['max_drawdown'],'incremental_trades':row['trades']-base['trades']})
    code_paths=[*Path('fundamentals').glob('*.py'),Path('livesim/small_account.py'),Path('livesim/small_account_risk.py'),Path('strategies/small_account.py'),Path('analysis/small_account_research.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    inputs={'price_frames_sha256':{symbol:hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() for symbol,frame in data.items()},
            'observation_ids_sha256':hashlib.sha256('\n'.join(r[0] for r in store.db.execute('SELECT id FROM observations ORDER BY id')).encode()).hexdigest(),
            'raw_documents':[dict(r) for r in store.db.execute('SELECT sha256,source,fetched_at FROM raw_documents ORDER BY sha256')]}
    report={'preregistration':prereg,'results':results,'incremental':comparisons,'ic':ic,'models':artifacts,
        'selected_account_experiment':selected_outputs,'code_hashes':hashes,'input_manifest':inputs,
        'corporate_fundamental_ablation':{'status':'NOT_ESTIMABLE','reason':'No historically dated ETF holdings with complete constituent fundamental coverage'},
        'production_configuration':{'fundamental_policy':'observe','deployment_approved':False,'strategy_decision':'NO_TRADE'},
        'conclusion':'No deployable incremental edge established; corporate effects unidentifiable and macro tests use reused temporal holdout'}
    pd.DataFrame(results).to_csv(output/'comparison.csv',index=False);pd.DataFrame(comparisons).to_csv(output/'incremental.csv',index=False)
    pd.DataFrame(ic).to_csv(output/'information_coefficients.csv',index=False)
    (output/'research.json').write_text(json.dumps(report,indent=2,allow_nan=False));return report
