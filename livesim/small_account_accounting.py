"""Read-model reconciliation for the single-position cash ledger.

Sub-penny values are retained internally. 1e-8 GBP tolerance accommodates
binary storage only; it is far below the displayed currency precision.
"""
import math
from decimal import Decimal


def reconcile(state, events, curve):
    d=lambda x: Decimal(str(x))
    cash=d(100); fees=d(0); realized=d(0); position=None; pnls=[]
    def same(a,b):
        return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(d(a)-d(b))<=d('0.00000001')
    for event in events:
        kind=event['event']
        if kind not in ('ENTRY','EXIT'): continue
        q,p,f=map(d,(event['quantity'],event['price'],event['fee']))
        if not all(x.is_finite() for x in (q,p,f)) or q<=0 or p<=0 or f<0:
            raise ValueError('ledger reconciliation: invalid execution')
        fees+=f
        if kind=='ENTRY':
            if position: raise ValueError('ledger reconciliation: overlapping entries')
            cost=q*p+f; cash-=cost; position=(q,p,cost)
        else:
            if position is None or not same(q,position[0]): raise ValueError('ledger reconciliation: unmatched exit')
            pnl=q*p-f-position[2]
            if not same(pnl,event['pnl']): raise ValueError('ledger reconciliation: exit P&L')
            cash+=q*p-f; realized+=pnl; pnls.append(float(pnl)); position=None
    if not same(cash,state['cash']): raise ValueError('ledger reconciliation: cash')
    actual=state.get('position')
    if bool(position)!=bool(actual): raise ValueError('ledger reconciliation: position presence')
    if position and any(not same(actual[k],v) for k,v in zip(('qty','entry','cost'),position)):
        raise ValueError('ledger reconciliation: position basis')
    if len(pnls)!=len(state.get('closed_pnl',[])) or any(not same(a,b) for a,b in zip(pnls,state.get('closed_pnl',[]))):
        raise ValueError('ledger reconciliation: realized history')
    if curve:
        last=curve[-1]
        if not same(last['cash'],cash) or not same(last['equity'],state['equity']) or str(last['ts'])!=state['last_ts']:
            raise ValueError('ledger reconciliation: latest snapshot')
    elif state['last_ts'] is not None: raise ValueError('ledger reconciliation: missing snapshots')
    if not position and not same(state['equity'],cash): raise ValueError('ledger reconciliation: flat equity')
    unrealized=d(state['equity'])-d(state['cash'])-d(actual['cost']) if actual else d(0)
    return {'cash':float(cash),'balance':float(d(100)+realized),'equity':float(state['equity']),
            'realized_pnl':float(realized),'unrealized_pnl':float(unrealized),'fees':float(fees),
            'market_value':float(d(state['equity'])-cash)}
