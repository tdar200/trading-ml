"""Run frozen-data, publication-time-aware ETF macro ablations."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from fundamentals.store import PointInTimeFundamentalStore
from fundamentals.research import run_etf_experiment
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--database',default='results/fundamentals/fundamentals.sqlite')
 p.add_argument('--prices',default='results/small_account');p.add_argument('--output',default='results/fundamentals/experiment')
 a=p.parse_args();s=PointInTimeFundamentalStore(a.database)
 data={symbol:pd.read_parquet(Path(a.prices)/(symbol+'.parquet')) for symbol in ('SPY','XLE','XLF','XLK','XLP','XLV')}
 run_etf_experiment(s,data,a.output);s.close()
