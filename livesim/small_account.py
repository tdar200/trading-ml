"""Transactional local paper execution. No network order placement exists here.

The same deterministic event path drives historical replay and fresh-quote paper
execution. Historical OHLC uses conservative stop-first ordering; quote paper
uses only the current quote and never fills retrospectively at a prior close.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import math
import numpy as np
import pandas as pd
from livesim.small_account_risk import RiskEngine, RiskLimits, RiskSnapshot
from strategies.small_account import Setup
from livesim.small_account_accounting import reconcile
from decimal import Decimal, ROUND_FLOOR


class PaperEngine:
    def __init__(self, setup=Setup(), *, symbol='SPY', fee_bps=0., spread_bps=2., slip_bps=3.,
                 limits=RiskLimits(), shadow=False, database=None, fundamental_filter=None):
        if any(not math.isfinite(x) or x<0 for x in (fee_bps,spread_bps,slip_bps)) or fee_bps>=10000:
            raise ValueError('invalid execution costs')
        self.setup=setup; self.symbol=symbol; self.fee=fee_bps/10000
        self.spread=spread_bps; self.slip=slip_bps; self.risk=RiskEngine(limits); self.shadow=shadow
        self.state={'cash':100.,'equity':100.,'peak':100.,'position':None,'day':None,'week':None,
                    'day_start':100.,'week_start':100.,'last_ts':None,'streak':0,'halted':False,
                    'closed_pnl':[], 'processed_signals':[], 'max_drawdown':0.}
        self.journal=[]; self.curve=[]; self.db=None
        self.fundamental_filter=fundamental_filter
        self.config={'setup':asdict(setup),'symbol':symbol,'fee_bps':fee_bps,'spread_bps':spread_bps,
                     'slip_bps':slip_bps,'limits':asdict(limits),'shadow':shadow}
        if fundamental_filter is not None:
            artifact=fundamental_filter.score.artifact
            self.config['fundamental_filter']={'policy':fundamental_filter.policy,'threshold':fundamental_filter.threshold,
                'calibration_id':fundamental_filter.score.calibration.id if fundamental_filter.score.calibration else None,
                'model_id':artifact.id if artifact else None,'research_shadow':fundamental_filter.shadow,
                'feature_version':__import__('fundamentals.features',fromlist=['VERSION']).VERSION,'max_period_age_days':fundamental_filter.features.max_age,
                'macro_max_age':fundamental_filter.macro.max_age if fundamental_filter.macro else None,
                'catalyst_half_life':fundamental_filter.catalysts.half_life if fundamental_filter.catalysts else None}
        self.fingerprint=hashlib.sha256(json.dumps(self.config,sort_keys=True).encode()).hexdigest()
        if database:
            p=Path(database); p.parent.mkdir(parents=True,exist_ok=True)
            self.db=sqlite3.connect(str(p),timeout=5,isolation_level=None)
            self.db.execute('PRAGMA journal_mode=WAL'); self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), fingerprint TEXT, payload TEXT)')
            self.db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, ts TEXT, payload TEXT)')
            self.db.execute('CREATE TABLE IF NOT EXISTS equity (ts TEXT PRIMARY KEY, value REAL, cash REAL, risk REAL)')
            row=self.db.execute('SELECT fingerprint,payload FROM state WHERE id=1').fetchone()
            if row:
                if row[0]!=self.fingerprint: raise ValueError('database configuration differs; use a new journal')
                self.state=json.loads(row[1])
                self.journal=[json.loads(x[0]) for x in self.db.execute('SELECT payload FROM events ORDER BY seq')]
                self.curve=[dict(ts=x[0],equity=x[1],cash=x[2],risk=x[3],exposure=(x[1]-x[2])/x[1] if x[1] else 0.)
                            for x in self.db.execute('SELECT ts,value,cash,risk FROM equity ORDER BY ts')]
                try: reconcile(self.state,self.journal,self.curve)
                except Exception:
                    self.db.close()
                    raise
            else:
                self.db.execute('INSERT INTO state VALUES (1,?,?)',(self.fingerprint,json.dumps(self.state)))

    def event(self, ts, kind, **kwargs):
        self.journal.append({'timestamp':str(ts),'instrument':self.symbol,'event':kind,
                             'strategy':self.setup.family,'mode':'SHADOW' if self.shadow else 'PAPER',**kwargs})

    def _exit(self, ts, raw, reason):
        s=self.state; p=s['position']; price=raw*(1-(self.spread/2+self.slip)/10000)
        if not self.risk.approve_reduction(p['qty'],p['qty'],price):
            raise RuntimeError('risk engine rejected invalid reducing execution')
        fee=price*p['qty']*self.fee; proceeds=price*p['qty']-fee
        pnl=proceeds-p['cost']; s['cash']+=proceeds; s['position']=None
        s['streak']=s['streak']+1 if pnl<=0 else 0; s['closed_pnl'].append(pnl)
        self.event(ts,'EXIT',price=price,quantity=p['qty'],fee=fee,pnl=pnl,reason=reason,
                   entry=p['entry'],stop=p['stop'],target=p['target'],regime=p['regime'],
                   entry_time=p['ts'],entry_cost=p['cost'],rules_followed=True)

    def step(self, bar, previous=None, *, healthy=True, age_seconds=0., clock_skew=0.,
             consistent=True, status_known=True, kill=False, expected_pnl=None, close_final=False,
             fill_fraction=1., order_rejected=False):
        if not math.isfinite(fill_fraction) or not 0 <= fill_fraction <= 1:
            raise ValueError('fill fraction must be finite in [0,1]')
        ts=pd.Timestamp(bar['ts'])
        if ts.tzinfo is None: raise ValueError('timezone-aware events required')
        if previous is not None and pd.Timestamp(previous['ts'])>=ts:
            raise ValueError('signal must precede execution event')
        if self.state['last_ts'] and ts<=pd.Timestamp(self.state['last_ts']): return 'DUPLICATE_OR_OLD'
        values=[float(bar[k]) for k in ('open','high','low','close')]
        if not all(math.isfinite(x) and x>0 for x in values) or values[1]<max(values) or values[2]>min(values):
            raise ValueError('invalid market event; no orders')
        reconcile(self.state,self.journal,self.curve)
        before=json.dumps(self.state); j0=len(self.journal); c0=len(self.curve)
        try:
            if self.db:
                self.db.execute('BEGIN IMMEDIATE')
                current=self.db.execute('SELECT payload FROM state WHERE id=1').fetchone()[0]
                if json.loads(current)!=self.state: raise RuntimeError('concurrent account changed; reload required')
            self._step(bar,previous,healthy,age_seconds,clock_skew,consistent,status_known,kill,expected_pnl,close_final,fill_fraction,order_rejected)
            reconcile(self.state,self.journal,self.curve)
            if self.db:
                self.db.execute('UPDATE state SET payload=? WHERE id=1',(json.dumps(self.state,allow_nan=False),))
                self.db.executemany('INSERT INTO events(ts,payload) VALUES (?,?)',[(e['timestamp'],json.dumps(e,allow_nan=False)) for e in self.journal[j0:]])
                c=self.curve[-1]
                self.db.execute('INSERT INTO equity VALUES (?,?,?,?)',(c['ts'],c['equity'],c['cash'],c['risk']))
                self.db.execute('COMMIT')
        except Exception:
            if self.db and self.db.in_transaction: self.db.execute('ROLLBACK')
            self.state=json.loads(before); del self.journal[j0:]; del self.curve[c0:]
            raise
        return self.journal[-1]['event'] if len(self.journal)>j0 else 'HOLD'

    def _step(self,b,prev,healthy,age,clock,consistent,known,kill,expected,final,fill_fraction,rejected):
        s=self.state; ts=pd.Timestamp(b['ts']).tz_convert('UTC'); day=str(ts.date()); iso=ts.isocalendar(); week=f'{iso.year}-{iso.week}'
        if day!=s['day']: s['day']=day; s['day_start']=s['equity']
        if week!=s['week']: s['week']=week; s['week_start']=s['equity']
        degraded=s['halted']
        if expected is not None and len(s['closed_pnl'])>=30:
            r=np.array(s['closed_pnl'][-30:]); se=r.std(ddof=1)/np.sqrt(len(r))
            # 99% one-sided divergence from a supplied predeployment mean; no fitted default.
            degraded |= bool(r.mean()+2.326*se < expected)
        # Unknown/disconnected state never invents a fill, including an exit.
        actionable=healthy and consistent and known and -self.risk.limits.max_clock_skew_seconds<=age<=self.risk.limits.max_data_age_seconds and abs(clock)<=self.risk.limits.max_clock_skew_seconds
        p=s['position']; exited=False
        if p and actionable:
            raw=None; reason=''
            if self.setup.stop_kind=='trailing' and prev:
                candidate=float(prev['close'])-float(prev['distance'])
                if not math.isfinite(candidate): raise ValueError('invalid trailing stop')
                p['stop']=max(p['stop'],candidate)
            if b['open']<=p['stop']: raw=b['open']; reason='gap_stop'
            elif (ts-pd.Timestamp(p['ts'])).total_seconds()>=self.setup.hold*86400: raw=b['open']; reason='time'
            elif b['low']<=p['stop']: raw=p['stop']; reason='stop'
            elif b['high']>=p['target']: raw=p['target']; reason='target'
            elif final: raw=b['close']; reason='evaluation_end'
            if raw is not None: self._exit(ts,raw,reason); exited=True
        mark=s['cash']+(s['position']['qty']*b['open'] if s['position'] else 0)
        if prev and bool(prev.get('signal',False)) and not s['position'] and not exited and not final:
            signal_id=str(prev['ts']); entry=float(Decimal(str(b['open']))*(1+(Decimal(str(self.spread))/2+Decimal(str(self.slip)))/10000))
            distance=float(prev['distance']); stop=entry-distance
            snapshot=RiskSnapshot(equity=mark,cash=s['cash'],price=entry,stop=stop,fee_rate=self.fee,
                spread_bps=self.spread,slippage_bps=self.slip,
                daily_return=mark/s['day_start']-1,weekly_return=mark/s['week_start']-1,
                drawdown=max(0,1-mark/s['peak']),data_age_seconds=age,clock_skew_seconds=clock,
                healthy=healthy,consistent=consistent,order_status_known=known,
                research_shadow=self.shadow,edge_degraded=degraded,losing_streak=s['streak'],
                duplicate=signal_id in s['processed_signals'],kill=kill)
            decision=None
            if self.fundamental_filter is not None:
                decision=self.fundamental_filter.decide(self.symbol,ts,prev,
                    asset_class=prev.get('asset_class','equity'))
            qty,reason=self.risk.approve(snapshot)
            if decision is not None:
                if not decision['permit_signal']:
                    # A hard-risk refusal (kill switch, loss limit...) must stay visible in the audit trail when the gate also says no.
                    reason='NO_TRADE: fundamental decision gate' if qty else reason+'; NO_TRADE: fundamental decision gate'
                    qty=0.
                decision['risk']={'status':reason,'quantity':qty,'flags':decision['fundamental']['flags']}
                decision['positionSize']=qty
                self.event(ts,'FUNDAMENTAL_DECISION',decision=decision)

            self.event(ts,'RISK',reason=reason,signal_time=signal_id,quantity=qty)
            requested_qty=qty
            if qty and rejected:
                self.event(ts,'ORDER_REJECTED',reason='simulated venue rejection; no fill')
                qty=0.
            if qty:
                if fill_fraction < 1:
                    tick=Decimal(str(self.risk.limits.quantity_step))
                    qty=float((Decimal(str(qty))*Decimal(str(fill_fraction))/tick).to_integral_value(rounding=ROUND_FLOOR)*tick)
                if qty<requested_qty:
                    self.event(ts,'PARTIAL_IOC',requested_quantity=requested_qty,filled_quantity=qty,reason='unfilled remainder cancelled')
            if qty:
                fee=qty*entry*self.fee; cost=qty*entry+fee; s['cash']-=cost
                s['position']={'qty':qty,'entry':entry,'cost':cost,'stop':stop,
                    'target':entry+self.setup.reward*distance,'ts':str(ts),'regime':str(prev['regime'])}
                s['processed_signals'].append(signal_id)
                self.event(ts,'ENTRY',price=entry,quantity=qty,fee=fee,stop=stop,
                    target=s['position']['target'],regime=str(prev['regime']),confidence=None,
                    reason=self.setup.family+' completed-bar signal',
                    expected_risk=qty*(entry*(1+self.fee)-stop*(1-(self.spread/2+self.slip)/10000)*(1-self.fee)),
                    expected_reward=qty*distance*self.setup.reward,spread_bps=self.spread,
                    slippage_bps=self.slip,signal_time=signal_id,rules_followed=True)
                # Entry-bar stops are executable; ambiguous OHLC always stop-first.
                p=s['position']
                if b['low']<=p['stop']: self._exit(ts,min(b['open'],p['stop']),'entry_bar_stop')
                elif b['high']>=p['target']: self._exit(ts,p['target'],'entry_bar_target')
        p=s['position']; s['equity']=s['cash']+(p['qty']*b['close'] if p else 0)
        s['peak']=max(s['peak'],s['equity']); dd=1-s['equity']/s['peak']
        s['max_drawdown']=max(s['max_drawdown'],dd)
        s['halted'] |= dd>=self.risk.limits.max_drawdown or degraded
        s['last_ts']=str(ts)
        self.curve.append({'ts':str(ts),'equity':s['equity'],'cash':s['cash'],
                           'risk':p['qty']*max(0,b['close']-p['stop']*(1-(self.spread/2+self.slip)/10000)*(1-self.fee)) if p else 0.,
                           'exposure':p['qty']*b['close']/s['equity'] if p else 0.})
        if s['cash']< -1e-8: raise RuntimeError('cash invariant violated')
