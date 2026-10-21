"""Discover broad catalogues, then run actual GPU research without live orders."""
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.market_universe import discover
from analysis.market_sweep import run
p=argparse.ArgumentParser();p.add_argument('--root',default='results/market_sweep');p.add_argument('--discover',action='store_true');p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--workers',type=int,default=6)
a=p.parse_args()
if a.discover or not (Path(a.root)/'universe.parquet').exists():discover(a.root,lambda venue,n:print(venue,n,flush=True))
run(a.root,a.device,a.workers)
