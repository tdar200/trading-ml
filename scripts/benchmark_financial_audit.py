"""Offline local paper throughput and permanent golden audit artifacts."""
import sys,json,time,resource,sqlite3,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from livesim.small_account import PaperEngine
from tests.test_financial_audit import golden_run
from dashboard.views.small_account import load_monitor
root=Path('results/qa_audit_20260930'); out=root/'golden'; out.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory() as d:
    e,fixture=golden_run(Path(d)/'paper.sqlite')
    con=sqlite3.connect(out/'paper.sqlite'); e.db.backup(con); con.close(); e.db.close()
    pd.DataFrame(fixture['ledger']).to_csv(root/'golden_ledger.csv',index=False)
    pd.DataFrame(fixture['trades']).to_csv(root/'golden_trades.csv',index=False)
    e=PaperEngine(database=Path(d)/'benchmark.sqlite'); timings=[]
    start=pd.Timestamp('2026-01-01',tz='UTC')
    for i in range(10000):
        bar=dict(ts=start+pd.Timedelta(milliseconds=i*10),open=100,high=100,low=100,close=100)
        t=time.perf_counter();e.step(bar);timings.append(time.perf_counter()-t)
    t=time.perf_counter(); load_monitor(Path(d)); read=time.perf_counter()-t # empty default path; separate actual below
    e.db.close()
t=time.perf_counter(); load_monitor(out); golden_read=time.perf_counter()-t
v=sorted(timings)
summary=dict(events=10000,simulated_feed_hz=100,seconds=sum(timings),mean_ms=1000*sum(timings)/len(timings),p95_ms=1000*v[int(len(v)*.95)],max_ms=1000*max(v),golden_read_ms=golden_read*1000,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='one instrument, no signals, durable SQLite; not a multi-asset or broker latency guarantee')
(root/'performance.json').write_text(json.dumps(summary,indent=2)); print(summary)
