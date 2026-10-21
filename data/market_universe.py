"""Dated market catalogues. Discovery is not historical membership or permission.

Every provider response is retained. No order endpoint is used. Global completeness
is never inferred from a successful response from one provider.
"""
from __future__ import annotations
import hashlib,io,json,time
from pathlib import Path
from datetime import datetime,timezone
from itertools import combinations
import pandas as pd
import requests


def stamp():return datetime.now(timezone.utc).isoformat()

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=str,allow_nan=False));tmp.replace(path)

def instrument(source,symbol,asset_class,**kwargs):
    return dict(id=hashlib.sha256((source+':'+symbol).encode()).hexdigest()[:20],source=source,symbol=symbol,
                asset_class=asset_class,discovered_at=stamp(),historical_membership_verified=False,
                deployment_approved=False,**kwargs)


def nasdaq_frame(text):
    # "NA" is a legitimate ticker; pandas' default NA inference corrupts it.
    frame=pd.read_csv(io.StringIO(text),sep='|',dtype=str,keep_default_na=False)
    required={'Symbol','Test Issue','ETF','Security Name','Listing Exchange'}
    if not required.issubset(frame):raise ValueError('invalid Nasdaq directory schema')
    return frame[(frame['Test Issue']=='N')&(frame['Symbol']!='')].copy()


def discover(root,progress=lambda *args:None):
    """Resume independently cached catalogue pages; persist failures explicitly."""
    root=Path(root);raw=root/'catalogues';raw.mkdir(parents=True,exist_ok=True);items=[];coverage=[]
    def fetch(name,url):
        p=raw/name
        if p.exists():return p.read_bytes()
        r=requests.get(url,timeout=40);r.raise_for_status();p.write_bytes(r.content);return r.content
    # Broker current AND inactive US assets; preserves rather than drops failures.
    try:
        p=raw/'alpaca.json'
        if p.exists():assets=json.loads(p.read_text())
        else:
            import os
            from alpaca.trading.client import TradingClient
            from dotenv import load_dotenv
            load_dotenv('.env')
            client=TradingClient(os.environ['ALPACA_API_KEY'],os.environ['ALPACA_API_SECRET'],paper=True)
            assets=[x.model_dump(mode='json') for x in client.get_all_assets()];save_json(p,assets)
        for a in assets:
            kind='crypto' if a.get('asset_class',a.get('class'))=='crypto' else 'equity_or_fund'
            items.append(instrument('alpaca',a['symbol'],kind,name=a['name'],exchange=a['exchange'],currency='USD',status=a['status'],broker_tradable=a['tradable'],fractionable=a.get('fractionable'),history_provider='alpaca',history_symbol=a['symbol']))
        coverage.append(dict(source='alpaca',status='COMPLETE_RESPONSE',count=len(assets),scope='Broker-listed active and inactive assets, not all historical listings'))
    except Exception as e:coverage.append(dict(source='alpaca',status='ERROR',reason=type(e).__name__))
    progress('broker catalogue',len(items))
    try:
        text=fetch('nasdaqtraded.txt','https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt').decode()
        frame=nasdaq_frame(text)
        for r in frame.to_dict('records'):
            items.append(instrument('nasdaq',r['Symbol'],'etf' if r['ETF']=='Y' else 'equity_or_security',name=r['Security Name'],exchange=r['Listing Exchange'],currency='USD',status='listed',history_provider='yahoo',history_symbol=r['Symbol'].replace('.','-')))
        coverage.append(dict(source='nasdaq',status='COMPLETE_RESPONSE',count=len(frame),scope='Current Nasdaq-traded listed issues; dated snapshot only'))
    except Exception as e:coverage.append(dict(source='nasdaq',status='ERROR',reason=type(e).__name__))
    try:
        pairs=json.loads(fetch('kraken.json','https://api.kraken.com/0/public/AssetPairs'))
        if pairs.get('error'):raise ValueError('exchange rejected catalogue')
        for key,a in pairs['result'].items():
            items.append(instrument('kraken',key,'crypto_spot',name=a.get('wsname',key),exchange='Kraken',currency=a['quote'],status=a.get('status','unknown'),minimum_quantity=a.get('ordermin'),minimum_notional=a.get('costmin'),history_provider='kraken',history_symbol=key))
        coverage.append(dict(source='kraken',status='COMPLETE_RESPONSE',count=len(pairs['result']),scope='Kraken spot pairs; regional access not verified'))
    except Exception as e:coverage.append(dict(source='kraken',status='ERROR',reason=type(e).__name__))
    try:
        currencies=json.loads(fetch('currencies.json','https://api.frankfurter.dev/v1/currencies'))
        for base,quote in combinations(sorted(currencies),2):
            items.append(instrument('fx_crosses',base+quote,'forex',name=base+'/'+quote,exchange='OTC',currency=quote,status='REFERENCE_CROSS_NOT_BROKER_LISTING',history_provider='yahoo',history_symbol=base+quote+'=X'))
        coverage.append(dict(source='fx_crosses',status='GENERATED_CROSSES',count=len(currencies)*(len(currencies)-1)//2,scope='Unordered crosses of published currencies; availability and executable quotes must be verified'))
    except Exception as e:coverage.append(dict(source='fx_crosses',status='ERROR',reason=type(e).__name__))
    # International equity directories via documented yfinance screener wrapper.
    import yfinance as yf
    from yfinance import EquityQuery
    exchanges=EquityQuery('eq',['region','us']).valid_values['exchange']
    for region,venues in sorted(exchanges.items()):
        if region=='us':continue # US official + broker directories already included.
        for venue in sorted(venues):
            count=0;total=None;error=None;venue_symbols=set();venue_symbols=set()
            try:
                offset=0
                while True:
                    p=raw/f'yahoo_{region}_{venue}_{offset}.json'
                    if p.exists():page=json.loads(p.read_text())
                    else:
                        page=yf.screen(EquityQuery('and',[EquityQuery('eq',['region',region]),EquityQuery('eq',['exchange',venue])]),size=250,offset=offset)
                        save_json(p,page);time.sleep(.15)
                    total=page['total'];quotes=page.get('quotes',[])
                    if not quotes:break
                    for q in quotes:
                        venue_symbols.add(q['symbol'])
                        venue_symbols.add(q['symbol'])
                        items.append(instrument('yahoo',q['symbol'],'equity',name=q.get('shortName',q['symbol']),exchange=venue,region=region,currency=q.get('currency','UNKNOWN'),status='current_listing',history_provider='yahoo',history_symbol=q['symbol']))
                    count+=len(quotes);offset+=len(quotes)
                    if offset>=total:break
            except Exception as e:error=type(e).__name__+': '+str(e)[:160]
            coverage.append(dict(source='yahoo',region=region,exchange=venue,status='ERROR_OR_PARTIAL' if error or len(venue_symbols)!=total else 'COMPLETE_RESPONSE',count=count,unique_symbols=len(venue_symbols),duplicate_rows=count-len(venue_symbols),total=total,reason=error or ('PAGINATION_COUNT_MISMATCH' if len(venue_symbols)!=total else None)))
            progress(region+':'+venue,len(items))
            save_json(root/'catalogue_progress.json',{'updated_at':stamp(),'records':len(items),'coverage':coverage})
    # Public continuous futures proxies are clearly distinct from actual contracts.
    for ticker in ('GC','SI','HG','PL','PA','CL','BZ','NG','HO','RB','ZC','ZW','ZS','ZM','ZL','KC','CC','CT','SB','OJ','LE','HE','GF','ES','NQ','YM','RTY','ZB','ZN','ZF','ZT','DX'):
        items.append(instrument('continuous_proxy',ticker+'=F','futures_proxy',name=ticker+' continuous proxy',exchange='UNVERIFIED',currency='UNKNOWN',status='CONTRACT_MODEL_REQUIRED',history_provider='yahoo',history_symbol=ticker+'=F'))
    coverage.append(dict(source='derivatives',status='INCOMPLETE_ENTITLEMENTS',scope='No complete global futures/options/bond/CFD catalogue or contract history; continuous proxies are not contracts'))
    # Exact venue/provider identities remain separate. Mark duplicate price histories.
    seen={};unique={}
    for r in items:
        key=(r['history_provider'],r['history_symbol']);r['duplicate_history_of']=seen.get(key)
        if key not in seen:seen[key]=r['id']
        unique[r['id']]=r
    frame=pd.DataFrame(unique.values());frame.to_parquet(root/'universe.parquet',index=False)
    save_json(root/'coverage.json',{'as_of':stamp(),'global_complete':False,'coverage':coverage,'instruments':len(frame),'survivorship_free':False})
    return frame
