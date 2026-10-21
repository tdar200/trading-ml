"""Read-only £100 research and paper monitoring. Never places orders."""
from pathlib import Path
import json,sqlite3
import numpy as np
import pandas as pd
import streamlit as st
from analysis.small_account_monitor import rolling_performance
from livesim.small_account_accounting import reconcile


def load_monitor(root:Path):
    research=json.loads((root/'research.json').read_text()) if (root/'research.json').exists() else None
    db=root/'paper.sqlite'
    if db.exists():
        with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as con:
            con.execute('BEGIN')
            state=json.loads(con.execute('SELECT payload FROM state WHERE id=1').fetchone()[0])
            eq=pd.read_sql_query('SELECT ts,value AS equity,cash,risk FROM equity ORDER BY ts',con)
            events=[json.loads(r[0]) for r in con.execute('SELECT payload FROM events ORDER BY seq')]
        reconcile(state,events,eq.to_dict('records'))
        mode='Local shadow account' if any(e.get('mode')=='SHADOW' for e in events) else 'Local paper account'
    else:
        state=json.loads((root/'state.json').read_text()) if (root/'state.json').exists() else {'cash':100.,'equity':100.,'position':None,'max_drawdown':0.,'last_ts':None}
        eq=pd.read_csv(root/'equity.csv') if (root/'equity.csv').exists() else pd.DataFrame()
        events=pd.read_csv(root/'journal.csv').replace({np.nan:None}).to_dict('records') if (root/'journal.csv').exists() else []
        mode='Historical diagnostic replay — not forward performance'
    status=json.loads((root/'paper.status.json').read_text()) if (root/'paper.status.json').exists() else {}
    return research,state,eq,pd.DataFrame(events),mode,status


def render(data_sources=None):
    st.subheader('£100 Research & Paper Monitor')
    st.info('NO_TRADE is the deployment decision. Live trading is disabled. Targets never change position risk.')
    root=Path(st.text_input('Results directory',value='results/small_account',key='small_account_results'))
    try: research,state,eq,journal,mode,status=load_monitor(root)
    except Exception as e:
        st.error('Monitoring unavailable; do not trade: '+str(e)); return
    st.caption(mode)
    cols=st.columns(4)
    for col,label,value in zip(cols,['Equity','Available cash','Cumulative P&L','Maximum drawdown'],
                              [f"£{state['equity']:.2f}",f"£{state['cash']:.2f}",f"£{state['equity']-100:.2f}",f"{state.get('max_drawdown',0):.2%}"]):
        col.metric(label,value)
    accounting=reconcile(state,journal.to_dict('records'),eq.to_dict('records'))
    cols=st.columns(4)
    for col,label,key in zip(cols,['Realized balance','Realized P&L','Unrealized P&L','Total fees'],['balance','realized_pnl','unrealized_pnl','fees']):
        col.metric(label,f"£{accounting[key]:.2f}")
    st.write('Starting balance: £100.00 · cash account · milestones do not change risk')
    st.caption('Milestone progress = (equity − £100) / (milestone − £100), clamped to 0–100%.')
    st.download_button('Download account JSON',json.dumps(accounting,indent=2),'small-account-account.json','application/json')
    st.dataframe(pd.DataFrame([{'Milestone':f'£{t}','Progress':max(0.,min((state['equity']-100)/(t-100),1.))} for t in (125,150,200,250,500,1000)]),hide_index=True)
    if not eq.empty:
        eq['ts']=pd.to_datetime(eq.ts,utc=True); st.line_chart(eq.set_index('ts')[['equity']])
        now=eq.ts.iloc[-1]; values={}
        for label,delta in [('Daily',1),('Weekly',7),('Monthly',30)]:
            before=eq[eq.ts<=now-pd.Timedelta(days=delta)]
            base=float(before.equity.iloc[-1]) if len(before) else 100.
            values[label+' P&L']=float(eq.equity.iloc[-1])-base
        st.dataframe(pd.DataFrame([values]),hide_index=True)
        st.download_button('Download equity CSV',eq.to_csv(index=False),'small-account-equity.csv','text/csv')
        st.caption('P&L windows are trailing calendar windows anchored to the latest observation.')
    st.write('Rolling paper/replay performance (last 30 closed trades)')
    st.dataframe(pd.DataFrame([rolling_performance(eq,journal,annual=365 if not journal.empty and str(journal.instrument.iloc[-1]).endswith('GBP') else 252)]),hide_index=True)
    st.write('Open position',state.get('position') or 'None')
    st.write('Current risk deployed',float(eq.risk.iloc[-1]) if not eq.empty and 'risk' in eq else 0.)
    if status:
        st.write('Feed / execution state',status.get('state')); st.write('Current signal',status.get('signal','unavailable'))
        st.write('Market regime',status.get('regime','unavailable')); st.caption('Last feed check: '+status.get('checked_at','unknown'))
        checked=pd.Timestamp(status['checked_at']); age=(pd.Timestamp.now(tz='UTC')-checked).total_seconds()
        if age>120: st.warning('Feed observation is stale; this is not an active live-data session.')
        if status.get('reason'): st.error(status['reason'])
    st.write('Next setup: a new completed-bar signal with healthy data and risk approval. No predicted setup probability is asserted.')
    if research:
        selected=research['selected']; st.write('Historical research candidate',selected['id'])
        st.caption('This candidate failed qualification; results are conditional diagnostics, not return forecasts.')
        st.dataframe(pd.DataFrame([selected['diagnostic']]),hide_index=True)
        probabilities=[]
        for horizon,m in research['monte_carlo']['base']['horizons'].items():
            probabilities.append({'horizon':horizon,'median_GBP':m['p50'],**{f'P(hit £{t})':m[f'hit_{t}'] for t in (125,150,200,250,500,1000)}})
        st.dataframe(pd.DataFrame(probabilities),hide_index=True)
        st.caption('100,000-path block bootstrap of sparse historical returns; zero observations are not proof of zero real-world risk.')
    st.subheader('Trade journal and risk decisions')
    if not journal.empty:
        st.dataframe(journal,hide_index=True)
        st.download_button('Download journal CSV',journal.to_csv(index=False),'small-account-journal.csv','text/csv')
    else: st.write('No trades recorded.')
