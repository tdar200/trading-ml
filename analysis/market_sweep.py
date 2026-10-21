"""Resumable catalogue → historical data → GPU backtests → canonical finalists.

Worker is read-only with respect to brokers. No live execution path is imported.
"""
from __future__ import annotations
import argparse,hashlib,json,logging,sqlite3,time,os,fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed,CancelledError
from threading import Event
from datetime import datetime,timezone
import pandas as pd
import numpy as np
from data.market_universe import save_json,stamp
from analysis.rule_discovery import validate_bars
from analysis.market_gpu import screen_batch,FAMILIES,PERIODS,HOLDS


def connect(root):
    db=sqlite3.connect(Path(root)/'sweep.sqlite',timeout=30);db.row_factory=sqlite3.Row
    db.executescript('''PRAGMA journal_mode=WAL;
      CREATE TABLE IF NOT EXISTS histories(id TEXT PRIMARY KEY,provider TEXT,symbol TEXT,asset_class TEXT,currency TEXT,status TEXT,reason TEXT,path TEXT,rows INTEGER,sha256 TEXT);
      CREATE TABLE IF NOT EXISTS screening(history_id TEXT PRIMARY KEY,payload TEXT,telemetry TEXT);
      CREATE TABLE IF NOT EXISTS canonical(history_id TEXT PRIMARY KEY,payload TEXT);
    ''');return db


def register(db,universe):
    # Nasdaq and Alpaca aliases share one US price history where the broker lists it.
    us=set(universe.loc[(universe.source=='alpaca')&(universe.asset_class!='crypto'),'symbol'])
    count=0
    with db:
        for row in universe.to_dict('records'):
            provider,symbol=row['history_provider'],row['history_symbol']
            if row['source']=='nasdaq' and row['symbol'] in us:provider,symbol='alpaca',row['symbol']
            if provider=='alpaca' and row['asset_class']=='crypto':provider='alpaca_crypto'
            key=hashlib.sha256((provider+':'+symbol).encode()).hexdigest()[:20]
            db.execute('INSERT OR IGNORE INTO histories VALUES (?,?,?,?,?,?,?,?,?,?)',(key,provider,symbol,row['asset_class'],row.get('currency','UNKNOWN'),'PENDING',None,None,None,None));count+=1
    return db.execute('select count(*) from histories').fetchone()[0]


def normalize(frame):
    if frame.empty:raise ValueError('NO_HISTORY')
    f=frame.copy();f['ts']=pd.to_datetime(f.ts,utc=True)
    # The current UTC day may contain an unfinished daily candle.
    f=f[f.ts<pd.Timestamp.now(tz='UTC').normalize()]
    if f.ts.duplicated().any():
        f=f.drop_duplicates()
        if f.ts.duplicated().any():raise ValueError('CONFLICTING_DUPLICATE_BARS')
    f=validate_bars(f)
    if len(f)<300:raise ValueError('INSUFFICIENT_HISTORY:'+str(len(f)))
    return f


def persist(db,row,frame,root):
    try:
        f=normalize(frame);path=Path(root)/'bars'/(row['id']+'.parquet');path.parent.mkdir(exist_ok=True)
        tmp=path.with_suffix('.tmp');f.to_parquet(tmp,index=False);tmp.replace(path)
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        with db:db.execute('UPDATE histories SET status=?,reason=NULL,path=?,rows=?,sha256=? WHERE id=?',('READY',str(path),len(f),sha,row['id']))
    except Exception as e:
        with db:db.execute('UPDATE histories SET status=?,reason=? WHERE id=?',('EXCLUDED',str(e)[:200],row['id']))


def fetch_other(row):
    if row['provider']=='kraken':
        import requests
        r=requests.get('https://api.kraken.com/0/public/OHLC',params={'pair':row['symbol'],'interval':1440},timeout=30);r.raise_for_status();data=r.json()
        if data.get('error'):raise ValueError('KRAKEN:'+str(data['error']))
        values=next(v for k,v in data['result'].items() if k!='last')
        f=pd.DataFrame(values,columns=['ts','open','high','low','close','vwap','volume','trades']);f['ts']=pd.to_datetime(f.ts,unit='s',utc=True)
        for col in ['open','high','low','close','volume']:f[col]=pd.to_numeric(f[col],errors='raise')
        return f[['ts','open','high','low','close','volume']]
    import yfinance as yf
    f=yf.Ticker(row['symbol']).history(start='2020-01-01',end=pd.Timestamp.now(tz='UTC').date().isoformat(),auto_adjust=True,raise_errors=True)
    if f.empty:return pd.DataFrame()
    f=f.reset_index().rename(columns={f.index.name or 'Date':'ts','Datetime':'ts','Date':'ts','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    return f[['ts','open','high','low','close','volume']]


def is_rate_limit(error):
    import re
    response=getattr(error,'response',None)
    status=getattr(response,'status_code',None)
    if status is None:status=getattr(error,'status_code',None)
    if status==429:return True
    message=str(error).lower()
    # A numeric ticker such as 073429.TW is not an HTTP status.
    return (type(error).__name__=='YFRateLimitError'
        or any(x in message for x in ('rate limit','too many requests','ratelimit'))
        or bool(re.search(r'\b(?:http(?:/\d(?:\.\d)?)?|status(?: code)?)\s*[:=]?\s*429\b',message)))


def history_round(db,root,workers=6,batch=64):
    from dotenv import load_dotenv
    load_dotenv('.env');logging.getLogger('yfinance').setLevel(logging.CRITICAL)
    # Alternate providers so a huge equity catalogue cannot starve FX/crypto.
    for provider in ('alpaca','alpaca_crypto','yahoo','kraken'):
        if (Path(root)/'STOP').exists():break
        limited=False
        if provider=='yahoo':
            # Give each asset class a turn; all equities remain queued.
            rows=[]
            for category in ('forex','futures_proxy','equity'):
                rows.extend(dict(r) for r in db.execute('SELECT * FROM histories WHERE status=? AND provider=? AND asset_class=? ORDER BY symbol LIMIT ?',('PENDING',provider,category,16 if category!='equity' else batch-32)))
            if not rows:
                rows=[dict(r) for r in db.execute('SELECT * FROM histories WHERE status=? AND provider=? LIMIT ?',('PENDING',provider,batch))]
        else:
            rows=[dict(r) for r in db.execute("SELECT * FROM histories WHERE status=? AND provider=? ORDER BY (symbol GLOB '[A-Za-z]*') DESC,symbol LIMIT ?",('PENDING',provider,batch if provider!='kraken' else 4))]
        if not rows:continue
        if provider.startswith('alpaca'):
            try:
                from datetime import datetime,timezone
                start=datetime(2020,1,1,tzinfo=timezone.utc);end=datetime.now(timezone.utc)
                if provider=='alpaca':
                    from data.alpaca_client import AlpacaClient
                    remaining=list(rows);bars=pd.DataFrame();client=AlpacaClient.from_env()
                    while remaining:
                        try:
                            bars=client.fetch_daily_bars([r['symbol'] for r in remaining],start,end);break
                        except Exception as error:
                            import re
                            match=re.search(r'invalid symbol: ([^"\\\s]+)',str(error))
                            bad=match.group(1) if match else None
                            if bad not in {r['symbol'] for r in remaining}:raise
                            with db:
                                db.execute('UPDATE histories SET status=?,reason=? WHERE provider=? AND symbol=?',('EXCLUDED','PROVIDER_REJECTED_SYMBOL','alpaca',bad))
                            remaining=[r for r in remaining if r['symbol']!=bad]
                    rows=remaining
                else:
                    from alpaca.data.historical import CryptoHistoricalDataClient
                    from alpaca.data.requests import CryptoBarsRequest
                    from alpaca.data.timeframe import TimeFrame
                    bars=CryptoHistoricalDataClient().get_crypto_bars(CryptoBarsRequest(symbol_or_symbols=[r['symbol'] for r in rows],timeframe=TimeFrame.Day,start=start,end=end)).df
                for row in rows:
                    f=bars.xs(row['symbol'],level='symbol').reset_index().rename(columns={'timestamp':'ts'}) if not bars.empty and row['symbol'] in bars.index.get_level_values('symbol') else pd.DataFrame()
                    persist(db,row,f,root)
            except Exception as e:
                limited=is_rate_limit(e)
                with db:
                    for row in rows:db.execute('UPDATE histories SET status=?,reason=? WHERE id=?',('FETCH_ERROR',type(e).__name__+': '+str(e)[:100],row['id']))
        else:
            stopped=Event()
            delay=float(os.environ.get('MARKET_YAHOO_DELAY_SECONDS','0')) if provider=='yahoo' else 0.
            if not np.isfinite(delay) or delay<0:raise ValueError('invalid provider delay')
            def guarded_fetch(row):
                if stopped.wait(delay):raise CancelledError('Provider paused before request')
                try:return fetch_other(row)
                except Exception as error:
                    if is_rate_limit(error):stopped.set()
                    raise
            with ThreadPoolExecutor(max_workers=1 if provider=='kraken' else workers) as pool:
                tasks={pool.submit(guarded_fetch,row):row for row in rows}
                for future in as_completed(tasks):
                    row=tasks[future]
                    try:persist(db,row,future.result(),root)
                    except CancelledError:continue
                    except Exception as e:
                        limited=limited or is_rate_limit(e)
                        with db:db.execute('UPDATE histories SET status=?,reason=? WHERE id=?',('FETCH_ERROR',type(e).__name__+': '+str(e)[:140],row['id']))
            if provider=='kraken':time.sleep(1)
        if limited:
            # Stop the whole worker: no automatic repeated requests after throttling.
            (Path(root)/'STOP').write_text('Provider rate limit: '+provider+'; inspect FETCH_ERROR rows before resuming.\n')
            progress(db,root,'RATE_LIMIT_PAUSE')
            break
        progress(db,root,'FETCHING')


def cost_profile(row):
    if 'crypto' in row['asset_class']:return (80.,10.,5.)
    if row['asset_class']=='forex':return (2.,10.,3.)
    return (15.,2.,3.)


def gpu_round(db,root,device='cuda',batch=64):
    rows=[dict(r) for r in db.execute('SELECT * FROM histories WHERE status=? ORDER BY id LIMIT ?',('READY',batch))]
    if not rows:return 0
    frames={r['provider']+':'+r['symbol']:pd.read_parquet(r['path']) for r in rows};costs={r['provider']+':'+r['symbol']:cost_profile(r) for r in rows}
    results,telemetry=screen_batch(frames,device=device,costs=costs)
    with db:
        for r in rows:
            part=results[results.instrument==r['provider']+':'+r['symbol']].replace({np.nan:None})
            db.execute('INSERT OR REPLACE INTO screening VALUES (?,?,?)',(r['id'],json.dumps(part.to_dict('records'),allow_nan=False),json.dumps(telemetry,allow_nan=False)))
            db.execute('UPDATE histories SET status=? WHERE id=?',('GPU_TESTED',r['id']))
    save_json(Path(root)/'gpu_latest.json',telemetry);progress(db,root,'GPU_SCREENING');return len(rows)


def canonical_round(db,root,maximum=3):
    from analysis.small_account_research import replay,metrics
    from strategies.small_account import Setup
    rows=db.execute("SELECT h.*,s.payload FROM histories h JOIN screening s ON s.history_id=h.id LEFT JOIN canonical c ON c.history_id=h.id WHERE c.history_id IS NULL LIMIT ?",(maximum,)).fetchall()
    for raw in rows:
        row=dict(raw);screened=json.loads(row['payload']);best=max(screened,key=lambda x:x['total_return'])
        f=pd.read_parquet(row['path']);fee,spread,slip=cost_profile(row)
        e=replay(f,Setup(best['family'],best['period'],best['hold']),symbol=row['symbol'],begin=int(len(f)*.6),end=int(len(f)*.8),fee_bps=fee,spread_bps=spread,slip_bps=slip)
        m=metrics(e);difference=abs(m['final_equity']-best['final_equity'])
        result={'candidate':best,'canonical':m,'difference':difference,'parity_pass':difference<=.000001,'holdout_opened':False,'deployment_approved':False,
                'interpretation':'normalized 100 quote-currency units; broker size/FX/margin/borrow not certified'}
        if not result['parity_pass']:raise RuntimeError('GPU/canonical mismatch '+row['id']+' '+str(difference))
        directory=Path(root)/'canonical'/row['id'];directory.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(e.curve).to_csv(directory/'equity.csv',index=False);save_json(directory/'journal.json',e.journal)
        with db:db.execute('INSERT INTO canonical VALUES (?,?)',(row['id'],json.dumps(result,allow_nan=False)))
    return len(rows)


def progress(db,root,stage):
    states={r[0]:r[1] for r in db.execute('SELECT status,count(*) FROM histories GROUP BY status')}
    result={'updated_at':stamp(),'stage':stage,'histories':sum(states.values()),'states':states,
        'gpu_backtest_configurations':states.get('GPU_TESTED',0)*len(FAMILIES)*len(PERIODS)*len(HOLDS),
        'canonical_replays':db.execute('SELECT count(*) FROM canonical').fetchone()[0],
        'global_coverage_complete':False,'deployment_approved':False,'holdout_opened':False}
    save_json(Path(root)/'status.json',result);print(json.dumps(result),flush=True)


def run(root,device='cuda',workers=6):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'worker.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A market sweep worker is already running')
    lock.write(str(os.getpid()));lock.flush()
    torch_threads=__import__('torch');torch_threads.set_num_threads(2)
    db=connect(root)
    universe=pd.read_parquet(root/'universe.parquet');register(db,universe)
    config={'created_at':stamp(),'families':FAMILIES,'periods':PERIODS,'hold_calendar_days':HOLDS,'reward_risk':2,
       'training_fraction':.6,'validation_fraction':.2,'reserved_holdout_fraction':.2,'device':device,'download_workers':workers,'yahoo_delay_seconds':float(os.environ.get('MARKET_YAHOO_DELAY_SECONDS','0')),
       'fees_bps_per_side':{'stocks':15,'crypto':80,'forex':2},'screening_only':True,'fx_translation':'not applied; normalized quote currency',
       'survivorship_warning':'dated current listings plus inactive broker assets do not reconstruct complete historical membership',
       'fundamentals':'not backfilled; missing point-in-time coverage cannot generate a fundamental score',
       'case_count':len(FAMILIES)*len(PERIODS)*len(HOLDS),'code_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('analysis/market_gpu.py'),Path('data/market_universe.py')]}}
    config_path=root/'run_config.json'
    if config_path.exists():
        previous=json.loads(config_path.read_text())
        for field in ('families','periods','hold_calendar_days','fees_bps_per_side'):
            if json.dumps(previous[field],sort_keys=True)!=json.dumps(config[field],sort_keys=True):raise RuntimeError('Research configuration changed; use a new results root: '+field)
        if previous['code_hashes']['analysis/market_gpu.py']!=config['code_hashes']['analysis/market_gpu.py']:
            raise RuntimeError('GPU model changed; use a new results root')
        save_json(root/('resume_'+str(time.time_ns())+'.json'),config)
    else:save_json(config_path,config)
    try:
        while not (root/'STOP').exists():
            pending=db.execute("select count(*) from histories where status='PENDING'").fetchone()[0]
            if pending:history_round(db,root,workers)
            while gpu_round(db,root,device):
                canonical_round(db,root,3)
                if (root/'STOP').exists():break
            if not pending:break
        progress(db,root,'STOPPED' if (root/'STOP').exists() else 'AVAILABLE_DATA_SWEEP_COMPLETE')
    except Exception:
        progress(db,root,'FAILED')
        raise
    finally:
        db.close()
        lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/market_sweep');p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();run(args.root,args.device,args.workers)
