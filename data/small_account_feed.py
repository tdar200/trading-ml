"""Public Kraken GBP spot data only: never authenticated, never submits orders."""
from __future__ import annotations
import json
import time
import math
from urllib.parse import urlencode
from urllib.request import Request,urlopen
import pandas as pd
from analysis.rule_discovery import validate_bars

BASE='https://api.kraken.com/0/public/'


def public_request(endpoint, **params):
    req=Request(BASE+endpoint+'?'+urlencode(params),headers={'User-Agent':'trading-ml-research/1.0','Cache-Control':'no-cache'})
    with urlopen(req,timeout=15) as response: payload=json.load(response)
    if payload.get('error'): raise RuntimeError('Kraken public API: '+str(payload['error']))
    return payload['result']


def daily_bars(pair='XBTGBP', request=public_request):
    if pair not in ('XBTGBP','ETHGBP'): raise ValueError('only GBP BTC/ETH spot supported')
    server=request('Time')['unixtime']; result=request('OHLC',pair=pair,interval=1440)
    key=next(k for k in result if k!='last')
    frame=pd.DataFrame(result[key],columns=['ts','open','high','low','close','vwap','volume','count'])
    frame['ts']=pd.to_datetime(frame.ts,unit='s',utc=True)
    for c in ('open','high','low','close','volume'): frame[c]=frame[c].astype(float)
    # Drop all incomplete periods, not just the last row by convention.
    frame=frame[frame.ts+pd.Timedelta(days=1)<=pd.Timestamp(server,unit='s',tz='UTC')]
    frame=validate_bars(frame)
    if len(frame)<60: raise ValueError('insufficient completed history')
    if (pd.Timestamp(server,unit='s',tz='UTC')-(frame.ts.iloc[-1]+pd.Timedelta(days=1))).total_seconds()>86400:
        raise ValueError('stale daily candles')
    return frame


def live_snapshot(pair='XBTGBP', request=public_request):
    if pair not in ('XBTGBP','ETHGBP'): raise ValueError('only GBP BTC/ETH spot supported')
    start=time.time(); server=request('Time')['unixtime']; elapsed=time.time()-start
    skew=server-(start+elapsed/2)
    if abs(skew)>5 or elapsed>5: raise ValueError('clock skew or excessive API latency')
    meta=next(iter(request('AssetPairs',pair=pair).values()))
    if meta.get('status')!='online': raise ValueError('market not online')
    begin=time.time(); quote=next(iter(request('Ticker',pair=pair).values())); end=time.time()
    bid,ask=float(quote['b'][0]),float(quote['a'][0])
    if not all(math.isfinite(x) for x in (bid,ask)) or not 0<bid<=ask or end-begin>5: raise ValueError('invalid/stale quote response')
    minimum=float(meta.get('costmin',1)); units=float(meta['ordermin']); precision=int(meta['lot_decimals'])
    if not all(math.isfinite(x) and x>0 for x in (minimum,units)) or not 0<=precision<=18:
        raise ValueError('invalid instrument precision/minimum')
    mid=bid/2+ask/2
    return {'ts':pd.Timestamp(end,unit='s',tz='UTC'),'mid':mid,'bid':bid,'ask':ask,
            'spread_bps':(ask-bid)/mid*10000,'age_seconds':end-begin,'clock_skew':skew,
            'min_notional':max(minimum,units*ask),
            'quantity_step':10**-precision,
            'provenance':'Kraken REST response receipt time; no exchange quote timestamp'}
