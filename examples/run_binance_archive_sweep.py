"""Separate resumable free-archive research run. No broker execution."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,fcntl,hashlib,json,os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from data.binance_archive import symbols,listing,fetch_month
from data.market_universe import instrument,save_json,stamp
from analysis.market_sweep import connect,register,persist,progress,cost_profile
from analysis.market_gpu import screen_batch
from analysis.small_account_research import replay,metrics
from strategies.small_account import Setup

def run(root,limit=0):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
        (root/'worker.pid').write_text(str(os.getpid()));torch.set_num_threads(2)
        db=connect(root)
        try:
            up=root/'universe.parquet'
            if not up.exists():
                names=symbols()
                pd.DataFrame([instrument('binance',s,'crypto_spot',history_provider='binance',history_symbol=s,currency='UNKNOWN',status='ARCHIVE_LISTED_NOT_BROKER_ELIGIBILITY') for s in names]).to_parquet(up,index=False)
            universe=pd.read_parquet(up);register(db,universe)
            save_json(root/'coverage.json',dict(instruments=len(universe),global_complete=False,coverage=[dict(source='binance_archive',count=len(universe),scope='Archived spot pairs including inactive; monthly files since 2020; quote currency not mapped')]))
            cfg=dict(source='Binance public monthly spot archives',start='2020-01',annualization=365,costs_bps=[80,10,5],cost_policy='conservative existing crypto assumptions, not Binance fee quotation',holdout_opened=False,created_at=stamp(),code_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('data/binance_archive.py','analysis/market_gpu.py',__file__)})
            cp=root/'run_config.json'
            if cp.exists() and json.loads(cp.read_text())['code_hashes']!=cfg['code_hashes']:raise ValueError('code changed; use new run root')
            if not cp.exists():save_json(cp,cfg)
            completed=0
            while not (root/'STOP').exists():
                rows=[dict(r) for r in db.execute("SELECT * FROM histories WHERE status='PENDING' ORDER BY CASE symbol WHEN 'BTCUSDT' THEN 0 WHEN 'ETHUSDT' THEN 1 ELSE 2 END,symbol LIMIT 8")]
                if limit:rows=rows[:max(0,limit-completed)]
                for row in rows:
                    if (root/'STOP').exists():break
                    progress(db,root,'ARCHIVE_DOWNLOAD:'+row['symbol'])
                    try:
                        keys=[k for k in listing('data/spot/monthly/klines/'+row['symbol']+'/1d/') if k.endswith('.zip') and '2020-01'<=k[-11:-4]<pd.Timestamp.now(tz='UTC').strftime('%Y-%m')]
                        if not keys:raise ValueError('NO_MONTHLY_ARCHIVES')
                        with ThreadPoolExecutor(max_workers=4) as pool:frames=list(pool.map(lambda k:fetch_month(k,root/'raw'),keys))
                        save_json(root/('sources_'+row['id']+'.json'),dict(keys=keys,retrieved_at=stamp(),checksums='raw/*.CHECKSUM',latest_complete_month=max(k[-11:-4] for k in keys)))
                        persist(db,row,pd.concat(frames).sort_values('ts').reset_index(drop=True),root)
                    except Exception as e:
                        with db:db.execute('UPDATE histories SET status=?,reason=? WHERE id=?',('FETCH_ERROR',type(e).__name__+': '+str(e)[:200],row['id']))
                        if isinstance(e,__import__('requests').HTTPError) and e.response.status_code in (403,418,429):
                            (root/'STOP').write_text('Archive access/rate restriction; inspect before resume');break
                    completed+=1
                ready=[dict(r) for r in db.execute("SELECT * FROM histories WHERE status='READY' LIMIT 8")]
                if ready:
                    frames={r['provider']+':'+r['symbol']:pd.read_parquet(r['path']) for r in ready}
                    results,telemetry=screen_batch(frames,device='cuda',costs={k:(80,10,5) for k in frames})
                    # Existing GPU core uses 252 for unknown providers. Only Sharpe
                    # depends on this factor: explicitly convert Binance to 365.
                    results['sharpe']*=np.sqrt(365/252);results['annualization_periods']=365
                    for r in ready:
                        part=results[results.instrument=='binance:'+r['symbol']].replace({np.nan:None});best=max(part.to_dict('records'),key=lambda x:x['total_return'])
                        f=frames['binance:'+r['symbol']];e=replay(f,Setup(best['family'],best['period'],best['hold']),symbol=r['symbol'],begin=int(len(f)*.6),end=int(len(f)*.8),fee_bps=80,spread_bps=10,slip_bps=5)
                        m=metrics(e,annual=365);diff=abs(m['final_equity']-best['final_equity'])
                        if diff>1e-6:raise RuntimeError('GPU/canonical mismatch '+r['symbol'])
                        with db:
                            db.execute('INSERT OR REPLACE INTO screening VALUES (?,?,?)',(r['id'],json.dumps(part.to_dict('records'),allow_nan=False),json.dumps(telemetry)))
                            db.execute('INSERT OR REPLACE INTO canonical VALUES (?,?)',(r['id'],json.dumps(dict(candidate=best,canonical=m,difference=diff,parity_pass=True,holdout_opened=False),allow_nan=False)))
                            db.execute("UPDATE histories SET status='GPU_TESTED' WHERE id=?",(r['id'],))
                    save_json(root/'gpu_latest.json',telemetry);progress(db,root,'GPU_SCREENING')
                if not rows or (limit and completed>=limit):break
            progress(db,root,'STOPPED' if (root/'STOP').exists() else 'PILOT_COMPLETE' if limit else 'AVAILABLE_DATA_SWEEP_COMPLETE')
        except Exception:
            progress(db,root,'FAILED');raise
        finally:db.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/binance_archive_sweep');p.add_argument('--limit',type=int,default=0);a=p.parse_args();run(a.root,a.limit)
