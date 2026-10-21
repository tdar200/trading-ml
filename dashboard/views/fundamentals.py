"""Source-grounded, read-only fundamental research interface."""
from pathlib import Path
import re
import pandas as pd
import streamlit as st
from fundamentals.store import PointInTimeFundamentalStore
from fundamentals.api import FundamentalAPI


def _no_negative_zero(text):
    """A value that rounds to zero must not display as '-0.00' / '-0.00%'."""
    return text[1:] if text.startswith('-') and not any(ch in '123456789' for ch in text) else text


def _plain(text):
    """Escape markdown so user-typed text echoed into a message is shown literally (no live links or images)."""
    return re.sub(r'([\\`*_{}\[\]()#+\-.!<>|~])',r'\\\1',str(text))


def render(data_sources=None):
    st.subheader('Point-in-time fundamental research')
    st.info('Research only. No fundamental model is approved for deployment. Missing scores are not zero scores.')
    database=st.text_input('Fundamental database','results/fundamentals/fundamentals.sqlite',key='fund_db')
    symbol=st.text_input('Fundamental symbol','XOM',key='fund_symbol').upper()
    asof=st.text_input('Information available as of (UTC)','2024-06-01T00:00:00Z',key='fund_asof')
    bundle=st.text_input('Research model bundle (optional)','',key='fund_model_bundle')
    asset=st.selectbox('Instrument class',['equity','etf','forex','crypto'],key='fund_asset')
    if not Path(database).exists():st.warning('No fundamental database. Run examples/ingest_fundamentals.py.');return
    store=None
    try:
        from fundamentals.confidence import load_score_bundle
        store=PointInTimeFundamentalStore(database,readonly=True);api=FundamentalAPI(store,load_score_bundle(bundle) if bundle else None)
        snapshot=api.get('/fundamentals/'+symbol,{'as_of':asof,'asset_class':asset})
        score=api.get('/fundamentals/'+symbol+'/score',{'as_of':asof,'asset_class':asset})
        cols=st.columns(4)
        cols[0].metric('Fundamental score','Unvalidated' if score['score'] is None else f"{score['score']:.2%}")
        cols[1].metric('Data coverage',f"{snapshot['completeness']:.0%}")
        cols[2].metric('Data status',snapshot['status']);cols[3].metric('Period age',str(snapshot.get('age_days','—')))
        st.caption('Coverage measures populated features, not predictive confidence. Scores, when trained, are predicted returns rather than probabilities.')
        features=snapshot['features'];columns=st.columns(3)
        for col,name in zip(columns,('revenue_yoy','gross_margin','fcf')):
            value=features.get(name);col.metric(name,'Missing' if value is None else _no_negative_zero(f'{value:,.2f}' if name=='fcf' else f'{value:.2%}'))
        st.write('Reported financial period',snapshot.get('period_end','Unavailable'))
        st.write('Information available at',snapshot.get('available_at','Unavailable'));st.write('Currency',snapshot.get('currency','Unavailable'))
        st.write('Descriptive flags',snapshot.get('flags',[]))
        st.write('Trailing twelve months',snapshot.get('ttm',{'status':'UNAVAILABLE'}))
        st.write('Source validation',api.get('/fundamentals/'+symbol+'/quality',{'as_of':asof}))
        st.write('Predictive confidence',score.get('confidence_details') or 'No held-out calibration artifact loaded')
        st.dataframe(pd.DataFrame([{'Feature':k,'Value':v,'Dimension':snapshot['dimensions'].get(k),'Local predictive evidence':'Not established'} for k,v in features.items()]),hide_index=True)
        if features:
            metric=st.selectbox('Financial history',['revenue','eps','ocf','capex','gross_profit','equity','long_debt'],key='fund_history')
            rows=store.as_of(symbol,asof,kind='financial'); rows=[r for r in rows if r['metric']==metric and r['value'] is not None and (not r['period_start'] or 300<=(pd.Timestamp(r['period_end'])-pd.Timestamp(r['period_start'])).days+1<=380)]
            if rows:
                f=pd.DataFrame(rows);f['period_end']=pd.to_datetime(f.period_end)
                st.line_chart(f.set_index('period_end')[['value']]);st.caption('Annual flow / instant balance values from versions available at the chosen timestamp.')
                st.dataframe(f[['period_end','value','reported_at','available_at','source','revision','currency','units']],hide_index=True)
            else:st.info('No data: '+metric+' history is not available at this timestamp.')
        macro=api.get('/macro/regime',{'as_of':asof});st.write('Macro information set',macro)
        events=api.get('/fundamentals/'+symbol+'/events',{'as_of':asof});st.write('Disclosed events and catalysts',events)
        st.write('Fundamental watchlist',api.get('/watchlist/fundamental',{'as_of':asof}))
        st.download_button('Download PIT snapshot',__import__('json').dumps(snapshot,indent=2,allow_nan=False),'fundamental-snapshot.json','application/json')
    except Exception as e:st.error('Fundamental information unavailable: '+_plain(e))
    finally:
        if store:store.close()
