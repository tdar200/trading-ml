"""Public live quotes -> deterministic risk -> local paper ledger; no broker orders."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,os,time
from dataclasses import replace
from datetime import datetime,timezone
from data.small_account_feed import daily_bars,live_snapshot
from strategies.small_account import Setup,features
from livesim.small_account import PaperEngine
from livesim.small_account_risk import RiskEngine


def main():
    p=argparse.ArgumentParser(); p.add_argument('--pair',choices=['XBTGBP','ETHGBP'],default='XBTGBP')
    p.add_argument('--database',type=Path,default=Path('results/small_account/paper.sqlite'))
    p.add_argument('--shadow',action='store_true',help='Hypothetical research only; bypasses evidence gate, never risk limits')
    p.add_argument('--once',action='store_true'); p.add_argument('--interval',type=float,default=60)
    p.add_argument('--family',choices=['donchian','ma','rsi','bollinger','volume','gap'],default='donchian')
    p.add_argument('--period',type=int,default=20)
    p.add_argument('--baseline',type=Path,help='Frozen research.json; matching validation needs >=30 trades for a degradation reference')
    p.add_argument('--fundamental-database',type=Path)
    p.add_argument('--fundamental-policy',choices=['observe','filter'],default='observe')
    a=p.parse_args()
    if os.getenv('TRADING_MODE','PAPER').upper()!='PAPER': raise SystemExit('LIVE DISABLED: no real-money adapter is qualified')
    if a.interval<10: raise SystemExit('poll interval must be at least 10 seconds')
    os.environ.setdefault('STOP_TRADING_FILE',str(a.database.parent/'STOP_TRADING'))
    fundamental_filter=None
    if a.fundamental_database:
        from fundamentals.store import PointInTimeFundamentalStore
        from fundamentals.features import FundamentalFeatureEngine
        from fundamentals.models import CombinedSignalEngine,MacroRegimeService,EventCatalystEngine
        store=PointInTimeFundamentalStore(a.fundamental_database,readonly=True)
        fundamental_filter=CombinedSignalEngine(FundamentalFeatureEngine(store),macro=MacroRegimeService(store),catalysts=EventCatalystEngine(store),policy=a.fundamental_policy)
    engine=PaperEngine(Setup(a.family,a.period),symbol=a.pair,fee_bps=80.,spread_bps=10.,slip_bps=5.,shadow=a.shadow,database=a.database,fundamental_filter=fundamental_filter)
    expected=None
    if a.baseline:
        report=json.loads(a.baseline.read_text())
        matches=[r for r in report['candidates'] if r['symbol']==a.pair and r['setup']==engine.config['setup'] and r['validation']['trades']>=30]
        if len(matches)!=1: raise SystemExit('No uniquely matching baseline with at least 30 validation trades')
        expected=matches[0]['validation']['expectancy_gbp']
    status_path=a.database.with_suffix('.status.json')
    while True:
        status={'checked_at':datetime.now(timezone.utc).isoformat(),'mode':'SHADOW' if a.shadow else 'PAPER','deployment_approved':False,'degradation_baseline':expected}
        try:
            bars=daily_bars(a.pair); prev=features(bars,engine.setup,annual=365).iloc[-1].to_dict()
            prev['asset_class']='crypto'
            quote=live_snapshot(a.pair)
            # Each actual order uses the current venue minimum and observed spread.
            engine.spread=quote['spread_bps']
            engine.risk=RiskEngine(replace(engine.risk.limits,min_notional=max(1.,quote['min_notional']),quantity_step=quote['quantity_step']))
            bar={'ts':quote['ts'],**{k:quote['mid'] for k in ('open','high','low','close')}}
            action=engine.step(bar,prev,age_seconds=quote['age_seconds'],clock_skew=quote['clock_skew'],expected_pnl=expected)
            status.update(state='OK',action=action,signal=bool(prev['signal']),regime=prev['regime'],quote={k:str(v) for k,v in quote.items()},account=engine.state)
        except Exception as e:
            status.update(state='NO_TRADE',reason=type(e).__name__+': '+str(e))
        tmp=status_path.with_suffix('.tmp'); tmp.write_text(json.dumps(status,indent=2,allow_nan=False)); tmp.replace(status_path)
        print(json.dumps(status,allow_nan=False),flush=True)
        if a.once: break
        time.sleep(a.interval)

if __name__=='__main__': main()
