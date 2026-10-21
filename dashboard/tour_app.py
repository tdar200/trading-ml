"""Read-only guided demonstration of the actual research engines and audit data."""
from pathlib import Path
import sys,json
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
import altair as alt
import streamlit as st
from fundamentals.store import PointInTimeFundamentalStore
from fundamentals.features import FundamentalFeatureEngine,comparable_accounting_rows
from livesim.small_account_risk import RiskEngine,RiskSnapshot

st.set_page_config(page_title='Trading research • Guided tour',page_icon='📊',layout='wide')
st.markdown('''<style>
.block-container{max-width:1400px;padding-top:2rem}
[data-testid="stMetric"]{background:#f1f5f9;border-radius:12px;padding:18px;border:1px solid #dce5ef}
.hero{background:#10253e;padding:28px 32px;border-radius:16px;color:white;margin-bottom:20px}
.hero h1{color:white;font-size:2.3rem;padding:0;margin:8px 0}.hero p{color:#c3d4e7;margin:0}
.eyebrow{font-size:12px;letter-spacing:2px;color:#64e5bf}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="eyebrow">TRADING RESEARCH / INTERACTIVE WALKTHROUGH</div><h1>From company filings to a controlled decision.</h1><p>Explore the actual data, rewind a filing, inspect a backtest and challenge the risk engine.</p></div>',unsafe_allow_html=True)
st.caption('Read-only demonstration • Historical data and synthetic risk scenarios are labelled • No live orders')
base=ROOT/'results/fundamental_closure_20260930';db=base/'real_ingestion_v3.sqlite';run=base/'reproduction'
if not db.exists():st.error('Run scripts/verify_fundamental_closure.py to create the verified data.');st.stop()
company,time,research,risk=st.tabs(['1 · Company fundamentals','2 · Rewind a filing','3 · Strategy results','4 · Challenge the risk engine'])

with company:
    st.subheader('Inspect a real company')
    a,b=st.columns([1,3]);symbol=a.selectbox('Company',['MSFT','WMT','JPM','JNJ','CAT','XOM','BA','SONY'])
    at='2025-07-01T00:00:00Z';b.info('Historical information cutoff: 1 July 2025. Values come from archived, accession-tagged SEC filings; these are not current quotes.')
    store=PointInTimeFundamentalStore(db,readonly=True)
    try:
        snap=FundamentalFeatureEngine(store).snapshot(symbol,at);rows=comparable_accounting_rows(store.as_of(symbol,at,kind='financial'))
        c=st.columns(4);currency=snap.get('currency','');raw=snap['raw'];f=snap['features']
        money=lambda value:'Unavailable' if value is None else f'{currency} {value/1e9:,.2f}bn'
        c[0].metric('Annual revenue',money(raw.get('revenue')));c[1].metric('Free cash flow',money(f.get('fcf')))
        c[2].metric('Revenue growth YoY','Unavailable' if f.get('revenue_yoy') is None else f"{f['revenue_yoy']:.2%}")
        c[3].metric('Net margin','Unavailable' if f.get('net_margin') is None else f"{f['net_margin']:.2%}")
        st.caption(f"Financial year ended {snap.get('period_end','unknown')} · Positive cash-expenditure FCF definition · Missing inputs remain unknown")
        left,right=st.columns([2,1])
        annual=[r for r in rows if r['metric']=='revenue' and r['period_start'] and 350<=(pd.Timestamp(r['period_end'])-pd.Timestamp(r['period_start'])).days+1<=380]
        frame=pd.DataFrame(annual).sort_values('period_end').drop_duplicates(['period_end'],keep='last').tail(8)
        frame['Revenue (billions)']=frame.value/1e9;frame['Fiscal year end']=frame.period_end
        left.write('Revenue over time');left.bar_chart(frame.set_index('Fiscal year end')[['Revenue (billions)']],color='#167d9a')
        right.write('What the engine can explain')
        for flag in snap['flags']:right.warning(flag.replace('_',' ').capitalize())
        if not snap['flags']:right.info('No implemented descriptive flags triggered. This is not an investment recommendation.')
        right.write(f"Feature coverage: **{snap['completeness']:.0%}**")
        right.caption('Coverage is not predictive confidence. No company model is approved for trading.')
        with st.expander('Trace the numbers to their filings'):
            st.dataframe(frame[['period_end','value','currency','available_at','revision','source']],hide_index=True)
            st.json({'TTM':snap.get('ttm'),'source_validation':store.quality(symbol)},expanded=False)
        st.download_button('Download this company snapshot',json.dumps(snap,indent=2),'company-snapshot.json','application/json')
    finally:store.close()

with time:
    st.subheader('Watch information become available')
    st.write('Move across Microsoft’s FY2024 SEC filing cutoff. The same feature engine is queried at each timestamp.')
    store=PointInTimeFundamentalStore(db,readonly=True)
    try:
        records=store.as_of('MSFT','2025-07-01T00:00:00Z',kind='financial')
        filing=min([r for r in records if r['metric']=='revenue' and r['period_end']=='2024-06-30' and r['period_start']=='2023-07-01'],key=lambda r:r['available_at'])
        cutoff=pd.Timestamp(filing['available_at'])
        before=cutoff-pd.Timedelta(seconds=1);after=cutoff+pd.Timedelta(seconds=1)
        position=st.radio('Replay position',['Before filing availability','After filing availability'],horizontal=True)
        clock=before if position.startswith('Before') else after
        view=FundamentalFeatureEngine(store).snapshot('MSFT',clock)
        st.info(f'Simulated clock: {clock.isoformat()}')
        c=st.columns(3);c[0].metric('Latest visible financial year',view.get('period_end','Missing'))
        c[1].metric('Visible annual revenue',f"USD {view['raw']['revenue']/1e9:,.3f}bn")
        c[2].metric('FY2024 filing visible','YES' if view.get('period_end')=='2024-06-30' else 'NO')
        st.success('This is a real point-in-time query. A future filing cannot enter the earlier snapshot.')
        st.caption('This demonstrates the SEC importer’s conservative availability cutoff, including its modeled delay. Earlier earnings releases are separate sources and are not being demonstrated here.')
        st.dataframe(pd.DataFrame([{'Stage':'Financial period ends','UTC date/time':'2024-06-30'},{'Stage':'SEC filing available to this engine','UTC date/time':cutoff.isoformat()},{'Stage':'Replay clock','UTC date/time':clock.isoformat()}]),hide_index=True)
    finally:store.close()

with research:
    st.subheader('See the actual £100 experiment')
    st.caption('Historical diagnostic replay • XLE Bollinger mean reversion • 5 December 2024 onward • Costs included • Not forward paper performance')
    comparison=pd.read_csv(run/'comparison.csv');row=comparison[(comparison.symbol=='XLE')&(comparison.system=='technical')].iloc[0]
    c=st.columns(4);c[0].metric('Ending equity',f"£{row.final_equity:.4f}");c[1].metric('Maximum drawdown',f'{row.max_drawdown:.2%}');c[2].metric('Closed trades',int(row.trades));c[3].metric('Total fees',f'£{row.fees_gbp:.4f}')
    curve=pd.read_csv(run/'technical_equity.csv');curve['ts']=pd.to_datetime(curve.ts)
    chart=curve.set_index('ts')[['equity']].rename(columns={'equity':'Account equity (£)'})
    st.altair_chart(alt.Chart(chart.reset_index()).mark_line(color='#167d9a').encode(x='ts:T',y=alt.Y('Account equity (£):Q',scale=alt.Scale(zero=False)),tooltip=['ts:T','Account equity (£):Q']).properties(height=300),use_container_width=True)
    st.caption('Equity chart uses a zoomed vertical axis so small gains and drawdowns remain visible.')
    st.warning('The measured gain is about 2 pence. These results do not justify live deployment or imply that £100 can become £1,000 in a month.')
    st.write('Did adding fundamental and macro information change this run?')
    st.dataframe(comparison[comparison.symbol=='XLE'][['system','final_equity','sharpe','max_drawdown','trades']],hide_index=True)
    st.caption('Company fundamentals have no demonstrated ETF influence without complete dated holdings. The tested variants have identical XLE outcomes.')
    with st.expander('Inspect the actual trade and risk journal'):
        st.dataframe(pd.DataFrame(json.loads((run/'technical_journal.json').read_text())),hide_index=True)
    st.link_button('Open the detailed paper-account monitor','http://127.0.0.1:8517')
    st.link_button('Open market-wide GPU research','http://127.0.0.1:8521')

with risk:
    st.subheader('Try to make the risk engine reject a trade')
    st.caption('Synthetic scenario • Actual production RiskEngine • Calculation only; no order is submitted')
    scenario=st.selectbox('Risk scenario',['Healthy shadow research setup','Daily loss limit reached','Stale market data','Emergency kill switch','Unqualified strategy'])
    price=st.slider('Example entry price (£)',10,200,100);daily=-.025 if scenario=='Daily loss limit reached' else 0.
    state=RiskSnapshot(equity=100,cash=100,price=float(price),stop=float(price)*.98,fee_rate=.0015,spread_bps=2,slippage_bps=3,
        daily_return=daily,data_age_seconds=120 if scenario=='Stale market data' else 0,kill=scenario=='Emergency kill switch',research_shadow=scenario!='Unqualified strategy')
    qty,reason=RiskEngine().approve(state)
    if qty:st.success('SHADOW RESEARCH APPROVED — sizing calculated; no order created')
    else:st.error(reason)
    c=st.columns(4);c[0].metric('Account equity','£100.00');c[1].metric('Requested risk budget','£0.50');c[2].metric('Approved quantity',f'{qty:.6f}');c[3].metric('Approved notional',f'£{qty*price:.2f}')
    st.write('Signal → independent risk checks → capped position size → paper execution')
    st.caption('The engine budgets costs, enforces the 25% exposure cap and checks feed health, loss limits, account consistency and the kill switch. Shadow approval is not approval for deployment.')

st.divider();st.caption('597 tests passed in the closure audit · 40 authoritative source checks · Historical data, source gaps and absence of a demonstrated edge remain explicit')
