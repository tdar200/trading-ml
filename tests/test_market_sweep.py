import json,sqlite3
import pandas as pd,pytest
from data.market_universe import nasdaq_frame,instrument
from analysis.market_sweep import connect,register,normalize,persist
from tests.test_market_gpu import fixture


def test_literal_na_ticker_and_test_issues():
    f=nasdaq_frame('Symbol|Test Issue|ETF|Security Name|Listing Exchange\nNA|N|N|Nano Labs|Q\nTEST|Y|N|Test|Q\n')
    assert list(f.Symbol)==['NA']


def test_provider_identity_and_duplicate_price_history(tmp_path):
    db=connect(tmp_path)
    records=[instrument('alpaca','NA','equity_or_fund',currency='USD',history_provider='alpaca',history_symbol='NA'),instrument('nasdaq','NA','equity_or_security',currency='USD',history_provider='yahoo',history_symbol='NA'),instrument('yahoo','NA.L','equity',currency='GBP',history_provider='yahoo',history_symbol='NA.L')]
    assert register(db,pd.DataFrame(records))==2
    assert register(db,pd.DataFrame(records))==2
    assert db.execute('select count(*) from histories').fetchone()[0]==2


def test_history_validation_does_not_invent_missing_bars(tmp_path):
    db=connect(tmp_path);r=instrument('alpaca','A','equity',currency='USD',history_provider='alpaca',history_symbol='A');register(db,pd.DataFrame([r]))
    row=dict(db.execute('select * from histories').fetchone())
    persist(db,row,fixture(),tmp_path)
    actual=dict(db.execute('select * from histories').fetchone());assert actual['status']=='READY' and actual['rows']==400 and len(actual['sha256'])==64
    with pytest.raises(ValueError):normalize(fixture(299))
    bad=fixture();bad.loc[2,'high']=bad.loc[2,'low']-1
    with pytest.raises(ValueError):normalize(bad)


def test_conflicting_duplicate_rejected():
    f=fixture();duplicate=f.iloc[[10]].copy();duplicate['volume']=123
    with pytest.raises(ValueError,match='CONFLICTING'):normalize(pd.concat([f,duplicate]))
    assert len(normalize(pd.concat([f,f.iloc[[10]]])))==400


def test_invalid_broker_symbol_does_not_poison_batch(tmp_path,monkeypatch):
    from analysis.market_sweep import history_round
    from data.alpaca_client import AlpacaClient
    db=connect(tmp_path)
    register(db,pd.DataFrame([instrument('alpaca',s,'equity',currency='USD',history_provider='alpaca',history_symbol=s) for s in ('BAD','GOOD')]))
    attempts=[]
    class Client:
        def fetch_daily_bars(self,symbols,start,end):
            attempts.append(symbols)
            if 'BAD' in symbols:raise ValueError('{"message":"invalid symbol: BAD"}')
            f=fixture().rename(columns={'ts':'timestamp'});f['symbol']='GOOD'
            return f.set_index(['symbol','timestamp'])
    monkeypatch.setattr(AlpacaClient,'from_env',lambda:Client())
    history_round(db,tmp_path)
    states={r['symbol']:r['status'] for r in db.execute('select * from histories')}
    assert states=={'BAD':'EXCLUDED','GOOD':'READY'}
    assert attempts==[['BAD','GOOD'],['GOOD']]


def test_throttling_stops_new_provider_requests(tmp_path,monkeypatch):
    import analysis.market_sweep as sweep
    db=connect(tmp_path)
    register(db,pd.DataFrame([instrument('yahoo','A','equity',history_provider='yahoo',history_symbol='A'),instrument('kraken','BTCUSD','crypto_spot',history_provider='kraken',history_symbol='BTCUSD')]))
    seen=[]
    def fetch(row):
        seen.append(row['provider']);raise RuntimeError('HTTP 429 Too Many Requests')
    monkeypatch.setattr(sweep,'fetch_other',fetch)
    sweep.history_round(db,tmp_path)
    assert seen==['yahoo']
    assert (tmp_path/'STOP').exists()
    assert db.execute("select status from histories where provider='kraken'").fetchone()[0]=='PENDING'


def test_numeric_ticker_is_not_http_rate_limit():
    from analysis.market_sweep import is_rate_limit
    from requests import HTTPError, Response
    assert not is_rate_limit(RuntimeError('$073429.TW: possibly delisted; no price data found'))
    assert not is_rate_limit(RuntimeError('429 records missing'))
    assert is_rate_limit(RuntimeError('HTTP 429'))
    assert is_rate_limit(RuntimeError('status code: 429'))
    response=Response();response.status_code=429
    assert is_rate_limit(HTTPError('request failed',response=response))


def test_rate_limit_leaves_unstarted_batch_members_pending(tmp_path,monkeypatch):
    import analysis.market_sweep as sweep
    db=connect(tmp_path)
    register(db,pd.DataFrame([instrument('yahoo',symbol,'equity',history_provider='yahoo',history_symbol=symbol) for symbol in ('A','B','C')]))
    called=[]
    def fetch(row):
        called.append(row['symbol']);raise RuntimeError('Too Many Requests')
    monkeypatch.setattr(sweep,'fetch_other',fetch)
    monkeypatch.setenv('MARKET_YAHOO_DELAY_SECONDS','0')
    sweep.history_round(db,tmp_path,workers=1)
    assert called==['A']
    assert dict(db.execute('SELECT symbol,status FROM histories'))=={'A':'FETCH_ERROR','B':'PENDING','C':'PENDING'}
