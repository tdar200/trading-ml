"""Reproduce the free quarterly ALFRED research archive; never backdate latest data."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from fundamentals.store import PointInTimeFundamentalStore
from fundamentals.providers import ingest_alfred,ingest_sec

def main():
    p=argparse.ArgumentParser();p.add_argument('--database',default='results/fundamentals/fundamentals.sqlite')
    p.add_argument('--cache',default='results/fundamentals/raw');p.add_argument('--start',default='2020-04-01');p.add_argument('--end',default='2026-07-01')
    p.add_argument('--include-xom',action='store_true');a=p.parse_args()
    if pd.Timestamp(a.end)>pd.Timestamp.now().normalize():p.error('future vintages forbidden')
    s=PointInTimeFundamentalStore(a.database);failures=[];completed=0
    for vintage in pd.date_range(a.start,a.end,freq='QS'):
        for series,units in [('CPIAUCSL','index'),('UNRATE','percent'),('FEDFUNDS','percent')]:
            try:
                n=ingest_alfred(s,series,str(vintage.date()),a.cache,units);completed+=1
                print(json.dumps({'series':series,'vintage':str(vintage.date()),'candidate_records':n}),flush=True)
            except Exception as e:failures.append({'series':series,'vintage':str(vintage.date()),'error':str(e)})
    if a.include_xom:ingest_sec(s,'XOM','34088',a.cache)
    report={'completed_snapshots':completed,'failures':failures}
    Path(a.database).with_suffix('.ingestion.json').write_text(json.dumps(report,indent=2));s.close()
    if failures:raise SystemExit('Incomplete archive: see ingestion JSON; do not claim complete coverage')
if __name__=='__main__':main()
