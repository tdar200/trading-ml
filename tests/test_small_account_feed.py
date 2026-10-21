import pandas as pd
import pytest
from data.small_account_feed import daily_bars,live_snapshot


def test_incomplete_candle_removed():
    start=pd.Timestamp('2025-01-01',tz='UTC').timestamp()
    rows=[[start+i*86400,100,101,99,100,100,10,2] for i in range(81)]
    def request(endpoint,**kwargs):
        return {'unixtime':start+80*86400+10} if endpoint=='Time' else {'XXBTZGBP':rows,'last':1}
    f=daily_bars(request=request); assert len(f)==80
    assert f.ts.max()==pd.Timestamp('2025-03-21',tz='UTC')


def test_feed_timeout_propagates():
    def request(*a,**kw): raise TimeoutError('simulated')
    with pytest.raises(TimeoutError): daily_bars(request=request)


def test_exchange_clock_skew_rejected():
    with pytest.raises(ValueError,match='clock skew'):
        live_snapshot(request=lambda *a,**kw:{'unixtime':0})


def test_no_arbitrary_market_or_endpoint():
    with pytest.raises(ValueError): daily_bars('DOGEGBP')
