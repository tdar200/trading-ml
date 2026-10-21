"""Read-only public data fetch and versioned local ingestion; no broker orders."""
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fundamentals.store import PointInTimeFundamentalStore
from fundamentals.providers import ingest_sec,ingest_alfred,ingest_json_records

def main():
 p=argparse.ArgumentParser();p.add_argument('--database',default='results/fundamentals/fundamentals.sqlite')
 p.add_argument('--cache',default='results/fundamentals/raw');p.add_argument('--symbol');p.add_argument('--cik')
 p.add_argument('--units');p.add_argument('--macro-series');p.add_argument('--vintage');p.add_argument('--import-json')
 a=p.parse_args();s=PointInTimeFundamentalStore(a.database)
 if a.symbol and a.cik:n=ingest_sec(s,a.symbol,a.cik,a.cache)
 elif a.macro_series and a.vintage:n=ingest_alfred(s,a.macro_series,a.vintage,a.cache,units=a.units)
 elif a.import_json:n=ingest_json_records(s,a.import_json)
 else:p.error('specify symbol+cik, macro-series+vintage, or import-json')
 print(json.dumps({'ingested_candidates':n,'database':a.database,'quality':s.quality(a.symbol) if a.symbol else None}));s.close()
if __name__=='__main__':main()
