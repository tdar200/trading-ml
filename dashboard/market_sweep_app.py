"""Local monitoring for the broad, resumable GPU research run."""
from pathlib import Path
import sys,json,sqlite3,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import streamlit as st
import pandas as pd
st.set_page_config(page_title='Market-wide GPU research',page_icon='🌐',layout='wide')
st.title('Market-wide GPU research')
st.caption('Candidate catalogues → validated history → 90 CUDA backtests per history → canonical replay. Research only; no live orders.')
from dashboard.sweep_status import snapshot
root=Path(os.environ.get('TRADING_SWEEP_ROOT',str(ROOT/'results/market_sweep')))
@st.fragment(run_every='10s')
def monitor():
    if not (root/'coverage.json').exists():
        st.info('Discovering catalogues…')
        if (root/'catalogue_progress.json').exists():st.json(json.loads((root/'catalogue_progress.json').read_text()),expanded=False)
        return
    coverage=json.loads((root/'coverage.json').read_text());status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    # Read one live ledger snapshot; status.json can lag while downloads run.
    if (root/'sweep.sqlite').exists():
        with sqlite3.connect((root/'sweep.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as ledger:
            ledger.execute('BEGIN')
            status['states']=dict(ledger.execute('SELECT status,count(*) FROM histories GROUP BY status'))
            status['histories']=sum(status['states'].values())
            status['gpu_backtest_configurations']=status['states'].get('GPU_TESTED',0)*90
            status['canonical_replays']=ledger.execute('SELECT count(*) FROM canonical').fetchone()[0]
    view=snapshot(root,status)
    st.subheader('Sweep progress')
    message=view['state']+' · '+(view['reason'] or status.get('stage','Waiting for worker'))
    if view['active']:st.info(message)
    elif view['state']=='Finished available queue':st.success(message)
    else:st.warning(message)
    st.progress(view['attempted_fraction'],text=f"History requests attempted: {view['attempted']:,} / {view['total']:,} ({view['attempted_fraction']:.2%})")
    st.progress(view['tested_fraction'],text=f"Histories GPU-backtested: {view['tested']:,} / {view['total']:,} ({view['tested_fraction']:.2%})")
    st.caption('Attempted includes rejected data and download errors. Only successful GPU runs count as backtested. Some catalogued histories will never qualify for testing.')
    counts=st.columns(4)
    counts[0].metric('Waiting for data',f"{view['pending']:,}")
    counts[1].metric('Ready for GPU',f"{view['ready']:,}")
    counts[2].metric('Excluded histories',f"{view['excluded']:,}")
    counts[3].metric('Download errors',f"{view['errors']:,}")
    age='unknown' if view['age_seconds'] is None else f"{int(view['age_seconds'])//60:,} minutes ago"
    st.caption(f"Dashboard refreshes every 10 seconds · Last worker update: {age} · ETA unavailable while paused or without a measured download rate.")
    cols=st.columns(4)
    cols[0].metric('Discovered venue listings',f"{coverage['instruments']:,}")
    cols[1].metric('Unique history requests',f"{status.get('histories',0):,}")
    cols[2].metric('GPU backtest configurations',f"{status.get('gpu_backtest_configurations',0):,}")
    cols[3].metric('Canonical finalist replays',f"{status.get('canonical_replays',0):,}")
    st.write('Run state:',status.get('stage','Preparing data queue'),' · Updated:',status.get('updated_at','pending'))
    st.info('Final 20% reserved and unopened. Catalogue coverage is broad, not the entire global market. Cross-listings are counted separately; no strategy is approved for deployment.')
    left,right=st.columns(2)
    states=status.get('states',{})
    if states:left.bar_chart(pd.DataFrame({'Histories':states}))
    if (root/'gpu_latest.json').exists():
        gpu=json.loads((root/'gpu_latest.json').read_text());right.metric('GPU',gpu.get('gpu_name','unknown'));right.metric('Last GPU batch',f"{gpu['cases']:,} configurations",f"{gpu['seconds']:.2f} seconds",delta_color='off')
        right.caption(f"Peak tensor allocation: {(gpu.get('peak_allocated_bytes') or 0)/1024**2:.1f} MiB. Costs and conservative stop fills are included.")
    st.subheader('Coverage and exclusions')
    st.dataframe(pd.DataFrame(coverage['coverage']),hide_index=True,use_container_width=True)
    if (root/'sweep.sqlite').exists():
        db=sqlite3.connect((root/'sweep.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        try:
            st.subheader('Data processing by asset type')
            st.dataframe(pd.read_sql_query('SELECT asset_class,status,count(*) AS histories FROM histories GROUP BY asset_class,status',db),hide_index=True)
            st.subheader('Recent GPU candidates — provisional validation results')
            records=[]
            for payload in db.execute('SELECT payload FROM screening ORDER BY rowid DESC LIMIT 100'):
                candidates=json.loads(payload[0]);best=max(candidates,key=lambda x:x['total_return']);records.append(best)
            if records:
                best=pd.DataFrame(records).sort_values('total_return',ascending=False)
                st.dataframe(best[['instrument','family','period','hold','total_return','max_drawdown','trades','device']],hide_index=True)
                st.caption('Best configuration per recently processed history. This is not a global winner list. Returns use normalized quote-currency capital, not a reconstructed GBP account.')
            st.subheader('Failures are retained, not hidden')
            st.dataframe(pd.read_sql_query("SELECT provider,symbol,status,reason FROM histories WHERE status IN ('EXCLUDED','FETCH_ERROR') ORDER BY rowid DESC LIMIT 100",db),hide_index=True)
        finally:db.close()
    st.caption('Known gaps: unavailable global derivatives contracts, incomplete delisting history and historical memberships, uncertain broker/UK access, missing dated fundamentals, and unverified FX/borrow/margin economics. GPU speed does not resolve these gaps.')
monitor()
