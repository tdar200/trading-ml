"""Conditional scenarios, not forecasts. Seeded streaming Monte Carlo in GBP."""
from __future__ import annotations
import math
import secrets
import numpy as np
from scipy.stats import binom

TARGETS = (125, 150, 200, 250, 500, 1000)
HORIZONS = {21: '30_days', 63: '90_days', 126: '6_months', 252: '12_months'}


def requirement():
    return {'total_return': 9., 'calendar_daily': 10**(1/30)-1,
            'trading_daily_21': 10**(1/21)-1, 'weekly_30_days': 10**(7/30)-1,
            'weekly_four_weeks': 10**.25-1}


def summarize(equity, peak, trough, maxdd, maximum, initial=100.):
    n = len(equity)
    return {'paths': n, 'mean': float(equity.mean()),
            **{f'p{p}': float(np.percentile(equity, p)) for p in (5,25,50,75,95)},
            'mean_max_drawdown': float(maxdd.mean()), 'p95_max_drawdown': float(np.quantile(maxdd,.95)),
            **{f'hit_{t}': float(np.mean(maximum >= t)) for t in TARGETS},
            **{f'end_above_{t}': float(np.mean(equity >= t)) for t in TARGETS},
            **{f'loss_{p}': float(np.mean(trough <= initial*(1-p/100))) for p in (25,50,75,100)},
            **{f'drawdown_{p}': float(np.mean(maxdd >= p/100)) for p in (20,25,50,75)},
            'operational_failure_below_1': float(np.mean(trough < 1)),
            'zero_event_95pct_upper_bound': float(1-.05**(1/n))}


def binary_paths(p, reward, risk, *, cost_r=.1, paths=100_000, seed=20260930,
                 horizons=HORIZONS, stress=False):
    """One opportunity/session. Stress has persistent edge decay and rare 5R gaps.

    Loss is a fraction of CURRENT equity. Exact zero cannot occur in bounded
    fractional IID scenarios; operational failure is equity < £1. No margin or
    counterparty default model is claimed. Stress assumptions are hypothetical.
    """
    if not all(math.isfinite(x) for x in (p,reward,risk,cost_r)) or type(paths) is not int or not (0 < p < 1 and reward > 0 and 0 < risk <= 1 and cost_r >= 0 and paths > 0):
        raise ValueError('invalid scenario')
    if not horizons or any(type(k) is not int or k<1 for k in horizons) or len(set(horizons.values()))!=len(horizons):
        raise ValueError('invalid horizons')
    if seed is None: seed=secrets.randbits(64)
    rng=np.random.default_rng(seed)
    equity=np.full(paths,100.); peak=equity.copy(); trough=equity.copy(); maximum=equity.copy()
    maxdd=np.zeros(paths); bad=np.zeros(paths,dtype=bool); result={}
    for step in range(1,max(horizons)+1):
        if stress:
            bad=np.where(bad, rng.random(paths) < .9, rng.random(paths) < .05)
            win=rng.random(paths) < np.maximum(.01,p-.15*bad)
            cost=cost_r*rng.lognormal(0,.5,paths)
            loss=np.where(rng.random(paths)<.005,5.,1.)
        else:
            win=rng.random(paths)<p; cost=cost_r; loss=1.
        r=np.where(win,reward,-loss)-cost
        equity*=np.maximum(0,1+risk*r)
        maximum=np.maximum(maximum,equity); peak=np.maximum(peak,equity)
        trough=np.minimum(trough,equity); maxdd=np.maximum(maxdd,1-equity/peak)
        if step in horizons:
            result[horizons[step]]=summarize(equity,peak,trough,maxdd,maximum)
    a=reward-cost_r; b=1+cost_r
    logg=p*math.log1p(risk*a)+(1-p)*math.log1p(-risk*b) if risk*b<1 and 1+risk*a>0 else -math.inf
    k=max(0.,(p*a-(1-p)*b)/(a*b)) if a>0 else 0.
    return {'seed':seed,'paths':paths,'horizon_steps':horizons,'p':p,'reward_risk':reward,'risk':risk,'cost_r':cost_r,'stress':stress,
            'analytic_basis':'unstressed IID assumptions; not stressed-process expectation',
            'expectancy_r':p*reward-(1-p)-cost_r,
            'variance_r':p*(1-p)*(reward+1)**2,'kelly_analytic':k,
            'expected_log_growth':logg if math.isfinite(logg) else None,
            'trades_to_10x_at_log_mean': math.ceil(math.log(10)/logg) if logg>0 else None,
            'horizons':result}


def exact_terminal_probability(p,reward,risk,n,cost_r=.1,target=10.):
    if not all(math.isfinite(x) for x in (p,reward,risk,cost_r,target)) or not (0<p<1 and reward>0 and 0<risk<=1 and cost_r>=0 and target>0) or type(n) is not int or n<1:
        raise ValueError('invalid exact scenario')
    up=1+risk*(reward-cost_r); down=1-risk*(1+cost_r)
    if down<=0: return None
    boundary=(math.log(target)-n*math.log(down))/(math.log(up)-math.log(down))
    # Inclusive threshold: logarithm/division rounding must not exclude an
    # exact all-wins outcome. Tolerance is bounded by floating-point ULPs.
    nearest=round(boundary)
    if abs(boundary-nearest)<=32*math.ulp(max(1.,abs(boundary))): boundary=float(nearest)
    wins=math.ceil(boundary)
    return {'wins_required':wins,'trades':n,'probability':float(binom.sf(wins-1,n,p))}


def block_paths(returns, *, paths=100_000, seed=420, block=5, stress=False, annual=252):
    """Circular block bootstrap preserves local dependence; no claim of stationarity.

    Net daily returns already include backtest costs. Stress adds a persistent
    adverse regime and execution drag on nonzero-return days, not reconstructed
    order-book fills. It deliberately does not pretend to simulate queue delay.
    """
    if type(paths) is not int or paths<1 or type(block) is not int or block<1 or annual not in (252,365):
        raise ValueError('invalid bootstrap configuration')
    r=np.asarray(returns,float)
    if r.ndim!=1 or len(r)<30 or not np.isfinite(r).all() or (r < -1).any():
        raise ValueError('need >=30 finite daily returns >= -1')
    horizons=HORIZONS if annual==252 else {30:'30_days',90:'90_days',182:'6_months',365:'12_months'}
    if seed is None: seed=secrets.randbits(64)
    rng=np.random.default_rng(seed); eq=np.full(paths,100.); peak=eq.copy(); low=eq.copy(); high=eq.copy()
    dd=np.zeros(paths); idx=np.zeros(paths,dtype=int); bad=np.zeros(paths,bool); out={}
    for t in range(max(horizons)):
        if t%block==0: idx=rng.integers(0,len(r),paths)
        sampled=r[(idx+t%block)%len(r)].copy()
        if stress:
            bad=np.where(bad,rng.random(paths)<.9,rng.random(paths)<.05)
            active=sampled!=0
            sampled=np.where(bad,sampled*1.5-.001*active,sampled)
            sampled-=active*rng.exponential(.0002,paths)
        eq*=np.maximum(0,1+sampled); peak=np.maximum(peak,eq); low=np.minimum(low,eq); high=np.maximum(high,eq)
        dd=np.maximum(dd,1-eq/peak)
        if t+1 in horizons: out[horizons[t+1]]=summarize(eq,peak,low,dd,high)
    return {'seed':seed,'paths':paths,'annual':annual,'horizon_steps':horizons,'method':'circular_block_bootstrap','block':block,'stress':stress,'observations':len(r),'horizons':out}
