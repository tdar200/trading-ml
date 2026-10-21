"""Track G (prefix bt) independent oracles. Standard library only (Decimal/fractions).

Nothing here imports fundamentals.*, analysis.small_account_research, tests/fundamental_oracle.py
or tests/financial_oracle.py. Conventions are INFERRED from the reported numbers and then
re-implemented from their textbook definitions:

  returns      r_t = E_t / E_{t-1} - 1, E_0 = 100 (starting capital), one observation per bar
               (flat days included).
  Sharpe       mean(r) / sample_std(r, ddof=1) * sqrt(252); risk-free = 0.
  Sortino      mean(r) / sqrt(sum(min(r,0)^2) / N) * sqrt(252) (target semideviation, MAR 0, divisor N).
  CAGR         (E_n/100)^(252/n) - 1 with n = number of observed bars (NOT elapsed calendar time).
  max DD       max_t (1 - E_t / max_{s<=t} E_s), the running peak includes E_0 = 100.
  PF/win/exp   per EXIT event pnl (net of both fees): PF = sum(wins)/-sum(losses), expectancy = mean pnl.
  turnover     sum(price*qty over ENTRY and EXIT events) / 100 (initial capital, both sides).
"""
from __future__ import annotations
from decimal import Decimal, getcontext
from fractions import Fraction
import math
import json
import csv
from pathlib import Path

getcontext().prec = 60
D = lambda x: Decimal(repr(float(x))) if not isinstance(x, Decimal) else x


def load_curve(path):
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    return rows


def load_journal(path):
    return json.loads(Path(path).read_text())


def _mean(xs):
    return sum(xs, Decimal(0)) / Decimal(len(xs))


def _sample_std(xs):
    n = len(xs)
    if n < 2:
        return Decimal(0)
    m = _mean(xs)
    return (sum(((x - m) ** 2 for x in xs), Decimal(0)) / Decimal(n - 1)).sqrt()


def perf_metrics(equity, journal, annual=252, start=100):
    """equity: list of end-of-bar equity values (floats), journal: list of event dicts."""
    e = [D(start)] + [D(x) for x in equity]
    n = len(e) - 1
    r = [e[i] / e[i - 1] - 1 for i in range(1, n + 1)]
    mean = _mean(r)
    std = _sample_std(r)
    sq = Decimal(annual).sqrt()
    downside = (sum((min(x, Decimal(0)) ** 2 for x in r), Decimal(0)) / Decimal(n)).sqrt()
    peak = e[0]
    dd = []
    for x in e:
        peak = max(peak, x)
        dd.append(1 - x / peak)
    maxdd = max(dd)
    growth = e[-1] / Decimal(start)
    cagr = Decimal(math.pow(float(growth), annual / n) - 1)
    monthly = Decimal(math.pow(float(growth), annual / (12 * n)) - 1)
    exits = [ev for ev in journal if ev['event'] == 'EXIT']
    pnl = [D(ev['pnl']) for ev in exits]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p < 0]
    longest = 0
    run = 0
    for d in dd:
        run = run + 1 if d > Decimal('1e-10') else 0
        longest = max(longest, run)
    turnover = sum((D(ev['price']) * D(ev['quantity']) for ev in journal if ev['event'] in ('ENTRY', 'EXIT')), Decimal(0)) / Decimal(start)
    fees = sum((D(ev['fee']) for ev in journal if 'fee' in ev and ev['event'] in ('ENTRY', 'EXIT')), Decimal(0))
    out = {
        'final_equity': e[-1], 'total_return': growth - 1, 'cagr': cagr, 'monthly_geometric': monthly,
        'daily_mean': mean,
        'sharpe': mean / std * sq if std > 0 else None,
        'sortino': mean / downside * sq if downside > 0 else None,
        'max_drawdown': maxdd, 'average_drawdown': _mean(dd),
        'calmar': cagr / maxdd if maxdd > 0 else None,
        'profit_factor': sum(wins, Decimal(0)) / -sum(losses, Decimal(0)) if losses else None,
        'expectancy_gbp': _mean(pnl) if pnl else None,
        'win_rate': Decimal(len(wins)) / Decimal(len(pnl)) if pnl else None,
        'average_win': _mean(wins) if wins else None,
        'average_loss': _mean(losses) if losses else None,
        'reward_risk_realized': (_mean(wins) / -_mean(losses)) if wins and losses else None,
        'trades': len(exits), 'turnover': turnover, 'fees_gbp': fees,
        'longest_underwater_observations': longest, 'observations': n,
    }
    return {k: (float(v) if isinstance(v, Decimal) else v) for k, v in out.items()}


def reconstruct_equity(journal, bars, begin_ts, end_ts, start=100, exposure=None):
    """Rebuild the end-of-bar equity from ENTRY/EXIT events and close prices only.

    bars: list of (ts_str, close). Processes events with timestamp <= bar ts. Position valued at the
    bar CLOSE after all of that bar's events (the engine marks equity at close).
    Returns list of Decimal equities for bars with begin_ts <= ts <= end_ts.
    """
    import datetime as dt

    def parse(s):
        s = str(s)
        return dt.datetime.fromisoformat(s)

    evs = [ev for ev in journal if ev['event'] in ('ENTRY', 'EXIT')]
    cash = D(start)
    qty = Decimal(0)
    i = 0
    out = []
    if exposure is not None:
        exposure.clear()
    for ts, close in bars:
        t = parse(ts)
        while i < len(evs) and parse(evs[i]['timestamp']) <= t:
            ev = evs[i]
            q, p, f = D(ev['quantity']), D(ev['price']), D(ev['fee'])
            if ev['event'] == 'ENTRY':
                cash -= q * p + f
                qty += q
            else:
                cash += q * p - f
                qty -= q
            i += 1
        out.append(cash + qty * D(close))
        if exposure is not None:
            exposure.append(float(qty * D(close) / out[-1]))
    return out


# --------------------------------------------------------------------------- DSR (Bailey & Lopez de Prado)
from statistics import NormalDist
_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def expected_max_sr(n_trials, sigma_trials):
    """E[max of N iid N(0, sigma^2)] approximation of Bailey-LdP (2014), same units as sigma."""
    if n_trials <= 1 or sigma_trials == 0:
        return 0.0
    return sigma_trials * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials) + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * math.e)))


def dsr_per_period(sharpes_annual, n_obs, periods=252, sr_observed_annual=None):
    """Deflated Sharpe z in PER-PERIOD units. Everything is converted to per-period first.

    SE of a per-period Sharpe under Gaussian IID: sqrt((1 + 0.5 SR^2)/(n-1)).
    """
    k = math.sqrt(periods)
    per = [s / k for s in sharpes_annual]
    mu = sum(per) / len(per)
    sigma_pop = math.sqrt(sum((x - mu) ** 2 for x in per) / len(per))
    sr = (max(sharpes_annual) if sr_observed_annual is None else sr_observed_annual) / k
    em = expected_max_sr(len(per), sigma_pop)
    se = math.sqrt((1 + 0.5 * sr * sr) / max(n_obs - 1, 1))
    z = (sr - em) / se
    return {'sr_per_period': sr, 'expected_max_per_period': em, 'se': se, 'z': z, 'p': _N.cdf(z)}


def dsr_mixed_units_as_published(sharpes_annual, n_obs, periods=252):
    """The pre-fix behaviour documented in the README banner: annualised SR vs per-period SE."""
    sr = max(sharpes_annual)
    n = len(sharpes_annual)
    mu = sum(sharpes_annual) / n
    sigma = math.sqrt(sum((x - mu) ** 2 for x in sharpes_annual) / n)
    em = expected_max_sr(n, sigma)
    se = math.sqrt((1 + 0.5 * sr * sr) / max(n_obs - 1, 1))
    return {'z': (sr - em) / se, 'expected_max': em}


# --------------------------------------------------------------------------- block bootstrap (matrix form)
def block_bootstrap_matrix(returns, paths=100000, seed=1941, days=21, block=5):
    """Independent matrix implementation of the circular block bootstrap used for the 30-day account study.

    The documented algorithm: a fresh uniform start offset for every block of `block` sessions,
    circular indexing into the return vector, equity multiplicatively compounded from 100.
    RNG: numpy default_rng(seed).integers(0, len(r), paths), one draw per block.
    """
    import numpy as np
    r = np.asarray(returns, float)
    rng = np.random.default_rng(seed)
    nblocks = -(-days // block)
    offsets = [rng.integers(0, len(r), paths) for _ in range(nblocks)]
    idx = np.stack([(offsets[i // block] + (i % block)) % len(r) for i in range(days)], axis=1)  # paths x days
    growth = 1.0 + r[idx]
    path = 100.0 * np.cumprod(growth, axis=1)
    path0 = np.concatenate([np.full((paths, 1), 100.0), path], axis=1)
    peak = np.maximum.accumulate(path0, axis=1)
    dd = (1 - path0 / peak).max(axis=1)
    end = path[:, -1]
    low = path0.min(axis=1)
    top = path0.max(axis=1)
    return {
        'median': float(np.median(end)), 'mean': float(end.mean()),
        'mean_max_drawdown': float(dd.mean()), 'p95_max_drawdown': float(np.quantile(dd, .95)),
        'end_at_or_above_100': float((end >= 100).mean()), 'end_at_or_below_50': float((end <= 50).mean()),
        'ruin_exact_zero': float((low <= 0).mean()), 'operational_ruin_below_1': float((low < 1).mean()),
        **{f'hit_{x}': float((top >= x).mean()) for x in (200, 500, 1000)},
    }


# --------------------------------------------------------------------------- misc
def required_daily_growth(target_multiple, periods):
    """Compound growth PER PERIOD needed to multiply capital by target_multiple over `periods` periods."""
    return target_multiple ** (1.0 / periods) - 1.0


def exact_equal(a, b, tol=0.0):
    if a is None or b is None:
        return a is None and b is None
    return abs(Fraction(a) - Fraction(b)) <= Fraction(tol)


# --------------------------------------------------------------------------- macro state oracle (raw SQL, no production code)
def _ym(period_end):
    y, m = int(period_end[:4]), int(period_end[5:7])
    return y * 12 + (m - 1)


def macro_state_oracle(db, t_iso, max_age_days=100):
    """Independent PIT macro state from the observations table.

    Visible rows: kind='macro', available_at <= T (string compare is exact because every stored timestamp is a
    zero-offset ISO string; verified separately). Per (metric, period_end) the LAST published vintage wins.
    Features follow the documented definitions: inflation_yoy = CPI_t / CPI_{t-12m} - 1, inflation_change_3m =
    inflation_yoy - (CPI_{t-3m} / CPI_{t-15m} - 1), unemployment/policy rate level and 3-month change.
    """
    out = {}
    status = 'AVAILABLE'
    for series in ('CPIAUCSL', 'UNRATE', 'FEDFUNDS'):
        rows = db.execute("SELECT period_end, value, available_at, reported_at, id FROM observations WHERE symbol=? AND kind='macro' "
                          "AND available_at<=? AND value IS NOT NULL ORDER BY available_at, reported_at, id", ('MACRO:' + series, t_iso)).fetchall()
        latest_by_period = {}
        for pe, v, av, rep, ident in rows:
            latest_by_period[pe] = (v, av)
        if not latest_by_period:
            if status != 'STALE':
                status = 'INCOMPLETE'
            continue
        last_pe = max(latest_by_period)
        cur_v, cur_av = latest_by_period[last_pe]
        age_days = (_parse(t_iso) - _parse(cur_av)).days
        if age_days > max_age_days:
            status = 'STALE'
            continue
        by_ym = {}
        for pe, (v, av) in latest_by_period.items():
            by_ym[_ym(pe)] = v
        base = _ym(last_pe)
        lag = lambda k: by_ym.get(base - k)
        if series == 'CPIAUCSL':
            p12, p3, p15 = lag(12), lag(3), lag(15)
            infl = cur_v / p12 - 1 if p12 and p12 > 0 else None
            prev = p3 / p15 - 1 if p3 is not None and p15 and p15 > 0 else None
            out['inflation_yoy'] = infl
            out['inflation_change_3m'] = infl - prev if infl is not None and prev is not None else None
        else:
            name = 'unemployment' if series == 'UNRATE' else 'policy_rate'
            p3 = lag(3)
            out[name] = cur_v
            out[name + '_change_3m'] = cur_v - p3 if p3 is not None else None
    if status != 'STALE' and (len(out) != 6 or any(v is None for v in out.values())):
        status = 'INCOMPLETE'
    return {'features': out, 'status': status}


def _parse(s):
    import datetime as dt
    return dt.datetime.fromisoformat(str(s))


# --------------------------------------------------------------------------- Spearman IC oracle
def _avg_ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(xs, ys):
    rx, ry = _avg_ranks(list(xs)), _avg_ranks(list(ys))
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None
