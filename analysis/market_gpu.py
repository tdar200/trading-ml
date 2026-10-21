"""Batched CUDA validation backtests with CPU equivalence and canonical replay.

Long-only, one position, ATR stops, conservative collisions, costs, cash/exposure
caps and loss halts. Research output is never authorization to submit an order.
The final 20% of each input is held out and is not screened or ranked.
"""
from __future__ import annotations
from itertools import product
import time
import numpy as np
import pandas as pd
import torch
from analysis.vectorized_screen import resolve_device
from analysis.rule_discovery import validate_bars

FAMILIES=('donchian','ma','rsi','bollinger','volume','gap')
PERIODS=(10,15,20,30,50)
HOLDS=(5,10,20)


def rolling(x,n,method='mean'):
    windows=x.unfold(-1,n,1)
    v=windows.mean(-1) if method=='mean' else windows.std(-1,correction=1) if method=='std' else windows.max(-1).values
    return torch.cat((torch.full((*x.shape[:-1],n-1),float('nan'),device=x.device,dtype=x.dtype),v),-1)


def signal_tensor(values,periods=PERIODS):
    o,h,l,c,v=values.unbind(-1)
    previous=torch.cat((c[:,:1],c[:,:-1]),1)
    tr=torch.maximum(h-l,torch.maximum((h-previous).abs(),(l-previous).abs()))
    atr=rolling(tr,14)
    delta=c-previous;delta[:,0]=float('nan')
    shift=lambda x:torch.cat((torch.full_like(x[:,:1],float('nan')),x[:,:-1]),1)
    signals=[];specs=[]
    for p in periods:
        mean=rolling(c,p);sd=rolling(c,p,'std');g=rolling(delta.clamp(min=0),p);loss=rolling((-delta).clamp(min=0),p);rsi=100*g/(g+loss)
        mapping={'donchian':c>rolling(shift(h),p,'max'),'ma':(c>mean)&(shift(c)<=shift(mean)),
            'rsi':(rsi>30)&(shift(rsi)<=30),'bollinger':c<mean-2*sd,
            'volume':(c>previous)&(v>1.5*rolling(shift(v),p))&(c>mean),
            'gap':(o>previous*1.01)&(c>o)&(c>mean)}
        for family in FAMILIES:
            signals.append(mapping[family]&torch.isfinite(atr));specs.append((family,p))
    return torch.stack(signals,1),atr,specs


def screen_batch(frames,*,device='cuda',periods=PERIODS,holds=HOLDS,costs=None,keep_curves=False):
    """Each asset has its own calendar and 60/20/20 split. No cross-asset fill."""
    if not frames:raise ValueError('no instruments')
    device=resolve_device(device);names=list(frames)
    validated=[validate_bars(frames[n]) for n in names]
    if any(len(f)<300 for f in validated):raise ValueError('at least 300 bars required')
    if any(p<2 or p>100 for p in periods) or any(h<1 for h in holds):raise ValueError('invalid grid')
    # Physically truncate before tensorization: the reserved holdout cannot leak.
    lengths=[int(len(f)*.8) for f in validated];starts=[int(len(f)*.6) for f in validated]
    t=max(lengths);b=len(names);array=np.full((b,t,5),np.nan);days=np.zeros((b,t),dtype=np.int64);weeks=np.zeros_like(days);seconds=np.zeros_like(days)
    for i,(f,end) in enumerate(zip(validated,lengths)):
        array[i,:end]=f[['open','high','low','close','volume']].iloc[:end].to_numpy(float)
        ts=pd.to_datetime(f.ts.iloc[:end],utc=True);days[i,:end]=ts.astype('int64').to_numpy()//86400000000000;seconds[i,:end]=ts.astype('int64').to_numpy()//1000000000
        iso=ts.dt.isocalendar();weeks[i,:end]=(iso.year*100+iso.week).to_numpy()
    values=torch.tensor(array,dtype=torch.float64,device=device)
    # Exact canonical CPU features avoid backend-dependent decisions at a
    # strict threshold (notably zero-variance Bollinger windows). Financial
    # paths, fills, sizing and metrics remain batched on the selected GPU.
    from strategies.small_account import features,Setup
    base_specs=[(family,p) for p in periods for family in FAMILIES]
    signal_values=np.zeros((b,len(base_specs),t),dtype=bool)
    atr_values=np.full((b,t),np.nan)
    for i,(frame,length) in enumerate(zip(validated,lengths)):
        for j,(family,period) in enumerate(base_specs):
            canonical=features(frame.iloc[:length],Setup(family,period))
            signal_values[i,j,:length]=canonical.signal.to_numpy()
        atr_values[i,:length]=canonical.atr.to_numpy()
    sig=torch.tensor(signal_values,device=device)
    atr=torch.tensor(atr_values,dtype=torch.float64,device=device)
    sig=sig.repeat_interleave(len(holds),dim=1)
    specs=[(family,p,h) for family,p in base_specs for h in holds];k=len(specs)
    expand=lambda x:torch.tensor(x,device=device,dtype=torch.float64).reshape(b,1).expand(b,k)
    start,end=expand(starts),expand(lengths)
    day=torch.tensor(days,device=device);week=torch.tensor(weeks,device=device);second=torch.tensor(seconds,device=device)
    hold=torch.tensor([s[2] for s in specs],device=device).reshape(1,k)
    costs=costs or {n:(15.,2.,3.) for n in names}
    fee=expand([costs[n][0]/10000 for n in names]);impact=expand([(costs[n][1]/2+costs[n][2])/10000 for n in names])
    if not torch.isfinite(fee).all() or not torch.isfinite(impact).all() or bool(((fee<0)|(fee>=1)|(impact<0)|(impact>=1)).any()):raise ValueError('invalid costs')
    cash=torch.full((b,k),100.,device=device,dtype=torch.float64);eq=cash.clone();peak=eq.clone();daily=eq.clone();weekly=eq.clone()
    qty=torch.zeros_like(eq);stop=qty.clone();target=qty.clone();entry_cost=qty.clone();entry_day=qty.clone();streak=qty.clone();trades=qty.clone();ddmax=qty.clone();fees=qty.clone()
    halted=torch.zeros((b,k),dtype=torch.bool,device=device);ret_sum=qty.clone();ret_sq=qty.clone();observations=qty.clone();curve=[]
    timer=time.perf_counter()
    for j in range(min(starts),t):
        active=(j>=start)&(j<end);final=j==end-1
        op,hi,lo,cl=[torch.nan_to_num(values[:,j,z],nan=1.).reshape(b,1).expand(b,k) for z in range(4)]
        daily=torch.where((day[:,j]!=day[:,j-1]).reshape(b,1),eq,daily)
        weekly=torch.where((week[:,j]!=week[:,j-1]).reshape(b,1),eq,weekly)
        had=qty>0
        gap=op<=stop;timed=second[:,j].reshape(b,1)-entry_day>=hold*86400;stop_hit=lo<=stop;target_hit=hi>=target
        exited=active&had&(gap|timed|stop_hit|target_hit|final)
        raw=torch.where(gap|timed,op,torch.where(stop_hit,stop,torch.where(target_hit,target,cl)))
        proceeds=qty*raw*(1-impact)*(1-fee);pnl=proceeds-entry_cost
        cash+=torch.where(exited,proceeds,0.);fees+=torch.where(exited,qty*raw*(1-impact)*fee,0.)
        streak=torch.where(exited,torch.where(pnl<=0,streak+1,0.),streak);trades+=exited
        qty=torch.where(exited,0.,qty)
        mark=cash+qty*op;daily_return=mark/daily-1;weekly_return=mark/weekly-1
        price=op*(1+impact);distance=2*atr[:,j-1].reshape(b,1);newstop=price-distance
        loss=price*(1+fee)-newstop*(1-impact)*(1-fee)
        scale=torch.where((streak>=3)|(weekly_return<=-.02),.5,1.)
        requested=torch.minimum(mark*.005*scale/loss,torch.minimum(cash/(price*(1+fee)),mark*.25/price))
        requested=torch.floor(requested*1e6)/1e6
        enter=active&(~final)&(~exited)&(qty==0)&sig[:,:,j-1]&(~halted)&(newstop>0)&(newstop<price)&(daily_return>-.02)&(weekly_return>-.04)&(1-mark/peak<.1)&(requested*price>=1)
        newqty=torch.where(enter,requested,0.)
        cost=newqty*price*(1+fee);cash-=cost;fees+=newqty*price*fee
        qty+=newqty;entry_cost=torch.where(enter,cost,entry_cost);stop=torch.where(enter,newstop,stop);target=torch.where(enter,price+2*distance,target);entry_day=torch.where(enter,second[:,j].reshape(b,1),entry_day)
        entry_exit=enter&((lo<=stop)|(hi>=target))
        raw=torch.where(lo<=stop,torch.minimum(op,stop),target);proceeds=qty*raw*(1-impact)*(1-fee)
        pnl=proceeds-entry_cost;cash+=torch.where(entry_exit,proceeds,0.);fees+=torch.where(entry_exit,qty*raw*(1-impact)*fee,0.)
        streak=torch.where(entry_exit,torch.where(pnl<=0,streak+1,0.),streak);trades+=entry_exit;qty=torch.where(entry_exit,0.,qty)
        neweq=cash+qty*cl;r=torch.where(active,neweq/eq-1,0.)
        eq=torch.where(active,neweq,eq);peak=torch.maximum(peak,eq);dd=1-eq/peak;ddmax=torch.maximum(ddmax,dd);halted|=dd>=.1
        ret_sum+=r;ret_sq+=r*r;observations+=active
        if keep_curves:curve.append(eq.detach().cpu().numpy().copy())
    if device=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-timer
    mean=ret_sum/observations;var=torch.clamp((ret_sq-ret_sum**2/observations)/(observations-1),min=0)
    # Compare within asset class; calendars differ, and FX translation is absent.
    annual=expand([365 if names[i].startswith(('kraken:','alpaca_crypto:')) else 252 for i in range(b)])
    sharpe=torch.where(var>1e-20,mean/torch.sqrt(var)*torch.sqrt(annual),float('nan'))
    arrays=[x.detach().cpu().numpy() for x in (eq,ddmax,trades,fees,sharpe)]
    rows=[]
    for i,name in enumerate(names):
        for c,(family,p,h) in enumerate(specs):
            rows.append(dict(instrument=name,family=family,period=p,hold=h,final_equity=arrays[0][i,c],total_return=arrays[0][i,c]/100-1,
                max_drawdown=arrays[1][i,c],trades=int(arrays[2][i,c]),fees=arrays[3][i,c],sharpe=arrays[4][i,c],device=device,
                validation_start=str(validated[i].ts.iloc[starts[i]]),validation_end=str(validated[i].ts.iloc[lengths[i]-1]),holdout_opened=False,deployment_approved=False))
    return pd.DataFrame(rows),dict(device=device,gpu_name=torch.cuda.get_device_name() if device=='cuda' else None,
        cases=b*k,seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,curves=curve if keep_curves else None)
