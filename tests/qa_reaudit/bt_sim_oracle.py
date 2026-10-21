"""Track G (prefix bt): independent event-driven simulator of the documented small-account paper rules.

Written from the rules (docs/small-account-architecture.md and the RiskEngine limits), not from the engine code:
  * one long position, fills at the open of the bar after the signal bar, entry = open*(1+(spread/2+slip)bp),
    exit = raw*(1-(spread/2+slip)bp), fee = fee_rate * notional on both legs (cash account, no leverage)
  * exit order inside a bar: gap through stop (open<=stop) at the open; calendar-time exit at the open when held >= hold days;
    stop touched -> stop; target touched -> target; ambiguity resolves stop-first; end of evaluation -> close
  * entry after the exits of the same bar are processed, never on the exit bar or on the last bar
  * size = min(0.5% equity risk / loss-per-share, cash/(price*(1+fee)), 25% equity / price) floored to 1e-6
    where loss-per-share budgets entry+exit fees and adverse exit spread/slippage
  * refuse: daily return <= -2%, weekly <= -4%, drawdown >= 10%, halted (drawdown >= 10% at a close), duplicate signal id
  * half size after 3 consecutive non-positive trades or weekly return <= -2%
Standard library only (Decimal for money).
"""
from __future__ import annotations
from decimal import Decimal as D, getcontext
import datetime as dt

getcontext().prec = 50


def simulate(bars, signals, *, fee_bps=15., spread_bps=2., slip_bps=3., hold_days=10, stop_dist_key='distance', reward=2.0, final_index=None):
    """bars: list of dict(ts: datetime, open, high, low, close); signals[i] = dict(signal, distance, ts) describing bar i-1 (or None).

    Returns (journal, equity_curve) where journal is a list of ('ENTRY'|'EXIT', ts, price, qty, fee, pnl, reason).
    """
    fee = D(str(fee_bps)) / 10000
    adverse = (D(str(spread_bps)) / 2 + D(str(slip_bps))) / 10000
    cash = D(100)
    qty = D(0)
    pos = None
    peak = D(100)
    equity = D(100)
    day = None
    week = None
    day_start = D(100)
    week_start = D(100)
    streak = 0
    halted = False
    processed = set()
    journal = []
    curve = []
    n = len(bars)
    final_index = n - 1 if final_index is None else final_index

    def exit_(ts, raw, reason):
        nonlocal cash, pos, streak
        price = D(str(raw)) * (1 - adverse)
        f = price * pos['qty'] * fee
        proceeds = price * pos['qty'] - f
        pnl = proceeds - pos['cost']
        cash += proceeds
        streak = streak + 1 if pnl <= 0 else 0
        journal.append(('EXIT', ts, price, pos['qty'], f, pnl, reason))
        pos = None

    for i, b in enumerate(bars):
        ts = b['ts']
        o, h, l, c = (D(str(b[k])) for k in ('open', 'high', 'low', 'close'))
        iso = ts.isocalendar()
        d_key, w_key = ts.date(), (iso[0], iso[1])
        if d_key != day:
            day, day_start = d_key, equity
        if w_key != week:
            week, week_start = w_key, equity
        degraded = halted
        exited = False
        prev = signals[i]
        if pos is not None:
            raw, reason = None, ''
            if o <= pos['stop']:
                raw, reason = o, 'gap_stop'
            elif (ts - pos['ts']).total_seconds() >= hold_days * 86400:
                raw, reason = o, 'time'
            elif l <= pos['stop']:
                raw, reason = pos['stop'], 'stop'
            elif h >= pos['target']:
                raw, reason = pos['target'], 'target'
            elif i == final_index:
                raw, reason = c, 'evaluation_end'
            if raw is not None:
                exit_(ts, raw, reason)
                exited = True
        mark = cash + (pos['qty'] * o if pos else 0)
        if prev is not None and prev['signal'] and pos is None and not exited and i != final_index:
            entry = o * (1 + adverse)
            dist = D(str(prev['distance']))
            stop = entry - dist
            sig_id = str(prev['ts'])
            daily = mark / day_start - 1
            weekly = mark / week_start - 1
            dd = max(D(0), 1 - mark / peak)
            allowed = True
            if sig_id in processed or degraded:
                allowed = False
            elif stop <= 0 or entry <= 0:
                allowed = False
            elif daily <= D('-0.02') or weekly <= D('-0.04') or dd >= D('0.10'):
                allowed = False
            q = D(0)
            if allowed:
                exit_price = stop * (1 - adverse)
                loss = entry * (1 + fee) - exit_price * (1 - fee)
                scale = D('0.5') if streak >= 3 or weekly <= D('-0.02') else D(1)
                q0 = min(mark * D('0.005') * scale / loss, cash / (entry * (1 + fee)), mark * D('0.25') / entry)
                step = D('0.000001')
                q = (q0 / step).to_integral_value(rounding='ROUND_FLOOR') * step
                if q * entry < 1:
                    q = D(0)
            if q > 0:
                f = q * entry * fee
                cost = q * entry + f
                cash -= cost
                pos = {'qty': q, 'entry': entry, 'cost': cost, 'stop': stop, 'target': entry + D(str(reward)) * dist, 'ts': ts}
                processed.add(sig_id)
                journal.append(('ENTRY', ts, entry, q, f, None, ''))
                if l <= pos['stop']:
                    exit_(ts, min(o, pos['stop']), 'entry_bar_stop')
                elif h >= pos['target']:
                    exit_(ts, pos['target'], 'entry_bar_target')
        equity = cash + (pos['qty'] * c if pos else 0)
        peak = max(peak, equity)
        dd = 1 - equity / peak
        halted = halted or dd >= D('0.10')
        curve.append(equity)
    return journal, curve
