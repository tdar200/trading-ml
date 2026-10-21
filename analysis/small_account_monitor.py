"""Descriptive rolling paper performance; sparse samples remain unqualified."""
import numpy as np
import pandas as pd


def rolling_performance(equity:pd.DataFrame,journal:pd.DataFrame,window=30, *, annual=252):
    if annual not in (252,365): raise ValueError('annual must be 252 or 365')
    exits=journal[journal.event=='EXIT'].tail(window) if not journal.empty and 'event' in journal else pd.DataFrame()
    pnl=exits.pnl.astype(float).to_numpy() if not exits.empty else np.array([])
    result={'closed_trades':len(pnl),'win_rate':float((pnl>0).mean()) if len(pnl) else None,
            'expectancy_gbp':float(pnl.mean()) if len(pnl) else None,
            'profit_factor':float(pnl[pnl>0].sum()/-pnl[pnl<0].sum()) if np.any(pnl<0) else None,
            'sharpe':None,'drawdown':0.,'evidence_state':'INSUFFICIENT_SAMPLE' if len(pnl)<30 else 'DESCRIPTIVE_ONLY'}
    if not equity.empty:
        f=equity.copy(); f['ts']=pd.to_datetime(f.ts,utc=True)
        e=f.set_index('ts').equity.resample('1D').last().dropna(); r=pd.Series(np.r_[100.,e.to_numpy()]).pct_change().dropna().tail(window)
        if len(r)>1 and r.std()>0: result['sharpe']=float(r.mean()/r.std()*np.sqrt(annual))
        result['drawdown']=float(1-e.iloc[-1]/max(100.,e.max()))
    return result
