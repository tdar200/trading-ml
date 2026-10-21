"""Independent point-in-time oracle and seeded generators (QA track A, prefix pit).

INDEPENDENCE: nothing here imports a calculation helper from fundamentals.*.
Only the public data classes needed to *feed* the system under test (Fact) are
imported lazily inside builder functions. All arithmetic is stdlib (float/int,
datetime, calendar, json), written from the documented semantics:

  * a fact is public at T iff available_at <= T (UTC instant comparison)
  * for one (metric, period_start, period_end) the visible version is the one
    with the greatest (available_at, reported_at)
  * withdrawn (None) versions replace earlier values
  * annual snapshot = latest visible annual revenue period + its same-period
    facts; prior/older periods are the annual revenue periods within 10 days
    of one/two calendar years earlier.
"""
from __future__ import annotations
import calendar
import json
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

UTC = timezone.utc
CANARY = 123456789123.0
CANARY_TEXT = '123456789123'
OBSERVED = '2030-01-01T00:00:00Z'

INSTANT_METRICS = ('assets', 'equity', 'cash', 'current_assets', 'current_liabilities', 'short_debt', 'long_debt')
FLOW_METRICS = ('revenue', 'gross_profit', 'operating_income', 'net_income', 'ocf', 'capex', 'depreciation')


def canon(obj) -> str:
    """Canonical JSON text: sorted keys, no NaN, compact."""
    return json.dumps(obj, sort_keys=True, allow_nan=False, separators=(',', ':'), default=str)


# ----------------------------------------------------------------------------
# calendar helpers (independent of pandas)
# ----------------------------------------------------------------------------
def month_end(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def months_back_end(d: date, k: int) -> date:
    """Month-end k months before month-end d."""
    idx = d.year * 12 + (d.month - 1) - k
    return month_end(idx // 12, idx % 12 + 1)


def years_back(d: date, n: int) -> date:
    try:
        return d.replace(year=d.year - n)
    except ValueError:  # Feb 29
        return d.replace(year=d.year - n, day=28)


def iso(d: date) -> str:
    return d.isoformat()


def D(s: str) -> date:
    return date.fromisoformat(s)


def dt_utc(s: str) -> datetime:
    """Parse '...+00:00' / '...Z' ISO text produced by this module or the store."""
    return datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(UTC)


# ----------------------------------------------------------------------------
# oracle fact model
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class OFact:
    metric: str
    start: str          # '' for instants
    end: str
    value: float | None
    reported: datetime  # UTC aware
    available: datetime  # UTC aware
    revision: str
    source: str = 'SRC'
    basis: str | None = None  # share_basis metadata for per-share facts

    @property
    def key(self):
        return (self.metric, self.start, self.end)


def visible_winners(facts, T: datetime):
    """The visible version per (metric,start,end) at instant T (None if no conflict)."""
    win = {}
    for f in facts:
        if f.available > T:
            continue
        cur = win.get(f.key)
        if cur is None or (f.available, f.reported) > (cur.available, cur.reported):
            win[f.key] = f
        elif (f.available, f.reported) == (cur.available, cur.reported) and f.value != cur.value:
            raise AssertionError('generator produced same-instant conflict for %r' % (f.key,))
    return win


# ----------------------------------------------------------------------------
# oracle arithmetic (documented conventions: denominators must be > 0, else None)
# ----------------------------------------------------------------------------
def o_ratio(a, b):
    if a is None or b is None or b <= 0:
        return None
    v = a / b
    return v if math.isfinite(v) else None


def o_growth(a, b):
    v = o_ratio(a, b)
    return None if v is None else v - 1


def o_sub(a, b):
    return None if a is None or b is None else a - b


def o_add(a, b):
    return None if a is None or b is None else a + b


def duration_days(start: str, end: str) -> int:
    return (D(end) - D(start)).days + 1


def annual_revenue_rows(win):
    out = []
    for (m, s, e), f in win.items():
        if m == 'revenue' and s and 300 <= duration_days(s, e) <= 380 and f.value is not None:
            out.append(f)
    return out


def period_values(win, annual, target: date):
    """(values-by-metric, chosen annual fact) for the annual period closest to target (+-10d)."""
    match = [f for f in annual if abs((D(f.end) - target).days) <= 10]
    if not match:
        return {}, None
    period = min(match, key=lambda f: abs((D(f.end) - target).days))
    vals = {}
    for (m, s, e), f in win.items():
        if e != period.end:
            continue
        if s and s != period.start:
            continue
        assert m not in vals, 'generator ambiguity'
        vals[m] = f.value
    return vals, period


def quarters_oracle(win, metric):
    """Discrete quarters {end: (value, start)} from visible versions (direct 60-120d, else YTD difference)."""
    cands = [f for (m, s, e), f in win.items() if m == metric and s and f.value is not None]
    out = {}
    for f in cands:
        if 60 <= duration_days(f.start, f.end) <= 120:
            out[f.end] = (f.value, f.start)
    if metric in ('eps', 'diluted_shares', 'shares_outstanding'):
        return out
    for f in cands:
        if f.end in out:
            continue
        earlier = [p for p in cands if p.start == f.start and p.end < f.end]
        if earlier:
            p = max(earlier, key=lambda x: x.end)
            gap = (D(f.end) - D(p.end)).days
            if 60 <= gap <= 120:
                out[f.end] = (f.value - p.value, iso(D(p.end) + timedelta(days=1)))
    return out


def ttm_from_totals(win, metric, end):
    """Published form (lead re-audit, QA-REAL-02): TTM = FY + current YTD - prior-year YTD; at a fiscal year end it
    IS the reported FY figure. Returns (value, start) or None when a needed total is not visible."""
    flows = [f for (m, s, e), f in win.items() if m == metric and s and f.value is not None]
    cur = [f for f in flows if f.end == end and 55 <= duration_days(f.start, f.end) <= 380]
    if not cur:
        return None
    c = max(cur, key=lambda f: duration_days(f.start, f.end))
    if duration_days(c.start, c.end) >= 350:
        return c.value, c.start
    before = iso(D(c.start) - timedelta(days=1))
    annual = [f for f in flows if f.end == before and 350 <= duration_days(f.start, f.end) <= 380]
    if len(annual) != 1:
        return None
    fy = annual[0]
    e = D(end)
    try:
        target = e.replace(year=e.year - 1)
    except ValueError:
        target = e.replace(year=e.year - 1, day=28)
    prior = [f for f in flows if f.start == fy.start and abs((D(f.end) - target).days) <= 10
             and abs(duration_days(f.start, f.end) - duration_days(c.start, c.end)) <= 10]
    if not prior:
        return None
    p = min(prior, key=lambda f: abs((D(f.end) - target).days))
    return fy.value + c.value - p.value, iso(D(p.end) + timedelta(days=1))


def ttm_oracle(win):
    metrics = ('revenue', 'net_income', 'ocf', 'capex', 'eps')
    res = {'values': {k: None for k in (*metrics, 'fcf')}, 'status': 'INCOMPLETE',
           'period_start': None, 'period_end': None, 'reasons': {}}
    rev = quarters_oracle(win, 'revenue')
    if not rev:
        return res
    end = max(rev)
    res['period_end'] = end
    for m in metrics:
        totals = None if m == 'eps' else ttm_from_totals(win, m, end)
        if totals is not None:
            value, start = totals
            if res['period_start'] is not None and start != res['period_start']:
                res['reasons'][m] = 'PERIOD_MISMATCH'
                continue
            res['values'][m] = value
            res['period_start'] = start
            continue
        q = quarters_oracle(win, m)
        dates = sorted(d for d in q if d <= end)[-4:]
        if len(dates) != 4 or dates[-1] != end:
            res['reasons'][m] = 'MISSING_QUARTERS'
            continue
        if any(D(q[b][1]) != D(a) + timedelta(days=1) for a, b in zip(dates, dates[1:])):
            res['reasons'][m] = 'NONCONTIGUOUS_QUARTERS'
            continue
        start = q[dates[0]][1]
        if not 350 <= (D(end) - D(start)).days + 1 <= 380:
            res['reasons'][m] = 'NON_ANNUAL_SPAN'
            continue
        if res['period_start'] is not None and start != res['period_start']:
            res['reasons'][m] = 'PERIOD_MISMATCH'
            continue
        if m == 'eps':
            bases = {win[(m, q[d][1], d)].basis for d in dates if (m, q[d][1], d) in win}
            if len(bases) != 1 or None in bases:
                res['reasons'][m] = 'UNVERIFIED_SPLIT_BASIS'
                continue
        res['values'][m] = sum(q[d][0] for d in dates)
        res['period_start'] = start
    res['values']['fcf'] = o_sub(res['values']['ocf'], res['values']['capex'])
    res['status'] = 'AVAILABLE' if all(v is not None for v in res['values'].values()) else 'INCOMPLETE'
    return res


def snapshot_oracle(facts, T: datetime, max_age=550):
    """Expected annual snapshot at instant T. Returns None if no annual revenue is visible."""
    win = visible_winners(facts, T)
    annual = annual_revenue_rows(win)
    out = {'ttm': ttm_oracle(win)}
    # lead re-audit (QA-CALC-06): a TTM older than the freshness limit is STALE whatever its completeness
    if out['ttm']['period_end'] is not None and (T.astimezone(timezone.utc).date() - D(out['ttm']['period_end'])).days > max_age:
        out['ttm']['status'] = 'STALE'
    if not annual:
        out['status'] = 'MISSING'
        return out
    latest = max(annual, key=lambda f: f.end)
    end = D(latest.end)
    cur, _ = period_values(win, annual, end)
    pri, pper = period_values(win, annual, years_back(end, 1))
    old, oper = period_values(win, annual, years_back(end, 2))
    g, p, o = cur.get, pri.get, old.get
    debt = o_add(g('short_debt'), g('long_debt'))
    fcf = o_sub(g('ocf'), g('capex'))
    feats = {
        'fcf': fcf,
        'gross_margin': o_ratio(g('gross_profit'), g('revenue')),
        'operating_margin': o_ratio(g('operating_income'), g('revenue')),
        'net_margin': o_ratio(g('net_income'), g('revenue')),
        'fcf_margin': o_ratio(fcf, g('revenue')),
        'debt': debt,
        'current_ratio': o_ratio(g('current_assets'), g('current_liabilities')),
        'debt_equity': o_ratio(debt, g('equity')),
        'revenue_yoy': o_growth(g('revenue'), p('revenue')),
        'operating_income_yoy': o_growth(g('operating_income'), p('operating_income')),
        'fcf_yoy': o_growth(fcf, o_sub(p('ocf'), p('capex'))),
        'revenue_acceleration': o_sub(o_growth(g('revenue'), p('revenue')), o_growth(p('revenue'), o('revenue'))),
        'roe': o_ratio(g('net_income'), o_ratio(o_add(g('equity'), p('equity')), 2)),
        'eps_yoy': o_growth(g('eps'), p('eps')),
    }
    for years in (3, 5):
        earlier, _ = period_values(win, annual, years_back(end, years))
        a, b = g('revenue'), earlier.get('revenue')
        if a is None or b is None or a <= 0 or b <= 0:
            feats[f'revenue_cagr_{years}y'] = None
        else:
            feats[f'revenue_cagr_{years}y'] = (a / b) ** (1 / years) - 1
    for m in ('revenue', 'ocf', 'operating_income'):
        q = quarters_oracle(win, m)
        dates = sorted(q)
        v = None
        if len(dates) >= 2:
            a, b = dates[-1], dates[-2]
            if 60 <= (D(a) - D(b)).days <= 120:
                v = o_growth(q[a][0], q[b][0])
        feats[m + '_qoq'] = v
    age = (T.astimezone(UTC).date() - end).days
    # lead re-audit (QA-REAL-03): when other annual flows are already public for a LATER year (revenue missing for it),
    # the revenue-anchored snapshot is a year behind and must not be served as current
    behind = any(f.value is not None and m in ('net_income', 'ocf', 'operating_income') and s
                 and 300 <= duration_days(s, e) <= 380 and D(e) > end + timedelta(days=30) for (m, s, e), f in win.items())
    out.update(status='STALE' if age > max_age or behind else 'AVAILABLE', period_end=latest.end, period_start=latest.start,
               raw=cur, features=feats, age_days=age, annual_prior=pper, annual_older=oper)
    return out


# ----------------------------------------------------------------------------
# seeded company generator
# ----------------------------------------------------------------------------
def _stamp(rng, day: date, lag_days: int) -> datetime:
    base = datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=lag_days)
    sec = rng.randint(6 * 3600, 22 * 3600 - 1)
    micro = rng.choice([0, 0, 0, rng.randint(1, 999999)])
    return base + timedelta(seconds=sec, microseconds=micro)


@dataclass
class Company:
    seed: int
    symbol: str
    facts: list
    scale_label: str | None  # None -> plain 'USD'; else 'USD millions'
    style: str
    fiscal_month: int


def gen_company(seed: int, symbol: str = 'ACME', n_years: int | None = None) -> Company:
    rng = random.Random(seed)
    fm = rng.choice([12, 12, 12, 12, 3, 6, 9, 1, 2])
    n = n_years or rng.randint(5, 9)
    y0 = rng.randint(2012, 2017)
    style = rng.choice(['ytd', 'full', 'discrete_only', 'ytd'])
    scaled = rng.random() < 0.3
    step = 1_000_000 if scaled else 1000
    rnd = lambda x: int(round(x / step)) * step  # noqa: E731

    ends = []
    for k in range(n):
        # fiscal year k ends in calendar year y0+k (month fm)
        ends.append(month_end(y0 + k, fm))
    starts = [months_back_end(e, 12) + timedelta(days=1) for e in ends]

    # truth tables
    truth = {}
    rev = rnd(rng.randint(40, 4000) * 1_000_000)
    qvals = {}  # (metric, k) -> [q1..q4]
    inst = {}
    for k in range(n):
        if k:
            rev = rnd(rev * (1 + rng.uniform(-0.25, 0.6)))
        rev = max(rev, step * 10)
        vals = {
            'revenue': rev,
            'gross_profit': rnd(rev * rng.uniform(0.15, 0.7)),
            'operating_income': rnd(rev * rng.uniform(-0.1, 0.35)),
            'net_income': rnd(rev * rng.uniform(-0.12, 0.28)),
            'ocf': rnd(rev * rng.uniform(-0.05, 0.3)),
            'capex': rnd(rev * rng.uniform(0.01, 0.15)),
            'depreciation': rnd(rev * rng.uniform(0.01, 0.08)),
        }
        for m, v in vals.items():
            truth[(m, k)] = v
            w = [rng.uniform(0.15, 0.4) for _ in range(4)]
            tot = sum(w)
            q = [rnd(v * x / tot) for x in w[:3]]
            q.append(v - sum(q))
            qvals[(m, k)] = q
        eps = round(rng.uniform(-2, 6), 2)
        truth[('eps', k)] = eps
        qeps = [round(eps * x, 2) for x in (0.2, 0.25, 0.25, 0.3)]
        qvals[('eps', k)] = qeps
        truth[('diluted_shares', k)] = rnd(rng.randint(10, 900) * 1_000_000)
        for m in INSTANT_METRICS:
            truth[(m, k)] = rnd(rev * rng.uniform(0.02, 1.5))
        for j in range(3):  # quarter-end instants
            for m in INSTANT_METRICS:
                inst[(m, k, j)] = rnd(truth[(m, k)] * rng.uniform(0.8, 1.2))

    facts: list[OFact] = []
    counter = [0]

    def acc():
        counter[0] += 1
        return f'ACC-{seed}-{counter[0]}'

    def mk(metric, start, end, value, rep, avail, rev_id, basis=None):
        facts.append(OFact(metric, start, end, None if value is None else float(value), rep, avail, rev_id, 'SRC', basis))

    def stamps(day, lag):
        rep = _stamp(rng, day, lag)
        avail = rep + timedelta(minutes=5) if rng.random() < 0.6 else rep + timedelta(seconds=rng.randint(0, 600))
        return rep, avail

    def restated(v):
        return rnd(v * (1 + rng.uniform(-0.12, 0.12)) + rng.choice([-1, 1]) * step * rng.randint(0, 3))

    for k in range(n):
        end, start = ends[k], starts[k]
        # ---- 10-K ----
        r = rng.random()
        lag = rng.randint(25, 60) if r < .35 else rng.randint(60, 95) if r < .80 else rng.randint(95, 150) if r < .95 else rng.randint(150, 420)
        lag_k = lag
        rep, avail = stamps(end, lag)
        a = acc()
        for m in FLOW_METRICS:
            mk(m, iso(start), iso(end), truth[(m, k)], rep, avail, a)
        mk('eps', iso(start), iso(end), truth[('eps', k)], rep, avail, a, basis='b1')
        mk('diluted_shares', iso(start), iso(end), truth[('diluted_shares', k)], rep, avail, a, basis='b1')
        for m in INSTANT_METRICS:
            mk(m, '', iso(end), truth[(m, k)], rep, avail, a)
        if style == 'discrete_only':
            q4s = months_back_end(end, 3) + timedelta(days=1)
            for m in ('revenue', 'net_income', 'ocf', 'capex'):
                mk(m, iso(q4s), iso(end), qvals[(m, k)][3], rep, avail, a)
            mk('eps', iso(q4s), iso(end), qvals[('eps', k)][3], rep, avail, a, basis='b1')
        # comparatives for k-1 and k-2 (possibly restated; truth evolves)
        for back, prob in ((1, 0.16), (2, 0.06)):
            kk = k - back
            if kk < 0:
                continue
            for m in (*FLOW_METRICS, 'eps', 'diluted_shares'):
                if rng.random() < prob and m not in ('eps', 'diluted_shares'):
                    truth[(m, kk)] = restated(truth[(m, kk)])
                mk(m, iso(starts[kk]), iso(ends[kk]), truth[(m, kk)], rep, avail, a,
                   basis='b1' if m in ('eps', 'diluted_shares') else None)
            if back == 1:
                for m in INSTANT_METRICS:
                    if rng.random() < prob:
                        truth[(m, kk)] = restated(truth[(m, kk)])
                    mk(m, '', iso(ends[kk]), truth[(m, kk)], rep, avail, a)
        # ---- 10-Q for quarters 1..3 ----
        for j in range(3):
            qend = months_back_end(end, 9 - 3 * j)
            if rng.random() < 0.08:
                continue
            qs = months_back_end(qend, 3) + timedelta(days=1)
            r = rng.random()
            lag = rng.randint(18, 60) if r < .93 else rng.randint(60, 200)
            rep, avail = stamps(qend, lag)
            a = acc()
            for m in INSTANT_METRICS:
                mk(m, '', iso(qend), inst[(m, k, j)], rep, avail, a)
            for m in ('revenue', 'gross_profit', 'operating_income', 'net_income', 'ocf', 'capex', 'depreciation'):
                cum = sum(qvals[(m, k)][:j + 1])
                disc = qvals[(m, k)][j]
                if style == 'ytd':
                    mk(m, iso(start), iso(qend), cum, rep, avail, a)
                elif style == 'full':
                    mk(m, iso(qs), iso(qend), disc, rep, avail, a)
                    if j > 0:
                        mk(m, iso(start), iso(qend), cum, rep, avail, a)
                else:
                    mk(m, iso(qs), iso(qend), disc, rep, avail, a)
            if style in ('full', 'ytd'):
                mk('eps', iso(qs), iso(qend), qvals[('eps', k)][j], rep, avail, a, basis='b1')
            else:
                mk('eps', iso(qs), iso(qend), qvals[('eps', k)][j], rep, avail, a, basis='b1')
            if style == 'ytd' and j == 0:
                # ytd style: Q1 YTD is 3 months = a direct quarter already
                pass
        # ---- 10-K/A restatement / withdrawal ----
        if rng.random() < 0.28:
            rep2, avail2 = stamps(end, lag_k + rng.randint(10, 220))
            if avail2 > avail:
                a = acc()
                for m in rng.sample(list(FLOW_METRICS), rng.randint(1, 4)):
                    if rng.random() < 0.18:
                        mk(m, iso(start), iso(end), None, rep2, avail2, a)  # withdrawal
                    else:
                        truth[(m, k)] = restated(truth[(m, k)])
                        mk(m, iso(start), iso(end), truth[(m, k)], rep2, avail2, a)
    # duplicates (exactly the same version published again, e.g. re-download) are not added here
    return Company(seed, symbol, facts, 'USD millions' if scaled else None, style, fm)


# ----------------------------------------------------------------------------
# rendering oracle facts into production Fact objects (random timestamp formats)
# ----------------------------------------------------------------------------
_TZ = [0, 1, -4, 5.5, -7, 9, -3.5, 2]


def render_ts(dt: datetime, rng: random.Random):
    r = rng.random()
    if r < 0.30:
        return dt.astimezone(UTC).isoformat().replace('+00:00', 'Z')
    if r < 0.45:
        return dt.astimezone(UTC).isoformat()
    if r < 0.85:
        tz = timezone(timedelta(hours=rng.choice(_TZ)))
        return dt.astimezone(tz).isoformat()
    if r < 0.93:
        return dt.astimezone(timezone(timedelta(hours=rng.choice(_TZ))))  # aware datetime object
    import pandas as pd
    return pd.Timestamp(dt).tz_convert(rng.choice(['UTC', 'Europe/London', 'America/New_York', 'Asia/Karachi']))


def to_fact(of: OFact, symbol: str, scale_label, rng: random.Random, observed=OBSERVED):
    from fundamentals.store import Fact
    metric = of.metric
    if metric == 'eps':
        cur, units, val = 'USD', 'USD/shares', of.value
    elif metric in ('diluted_shares', 'shares_outstanding'):
        cur, units, val = '', 'shares', of.value
    else:
        cur = 'USD'
        if scale_label:
            units, val = scale_label, (None if of.value is None else of.value / 1_000_000)
        else:
            units, val = 'USD', of.value
    meta = {'share_basis': of.basis} if of.basis else {}
    return Fact(symbol, metric, val, of.end, render_ts(of.reported, rng), render_ts(of.available, rng), of.source,
                of.revision, cur, units, of.start, observed, 'financial', meta)


def build_store(facts, symbol, scale_label, seed, path=':memory:', extra=(), dup_rate=0.1, batches=4):
    """Ingest oracle facts out of order, in several batches, with duplicates and varying observed_at."""
    from fundamentals.store import PointInTimeFundamentalStore
    rng = random.Random(seed * 7919 + 13)
    store = PointInTimeFundamentalStore(path)
    rows = [to_fact(f, symbol, scale_label, rng) for f in facts]
    rows += list(extra)
    dups = [to_fact(f, symbol, scale_label, rng, observed='2031-02-03T04:05:06Z') for f in facts if rng.random() < dup_rate]
    rows += dups
    rng.shuffle(rows)
    k = max(1, batches)
    for i in range(k):
        chunk = rows[i::k]
        if chunk:
            store.ingest(chunk)
    return store


# ----------------------------------------------------------------------------
# timestamps to probe
# ----------------------------------------------------------------------------
def pick_times(rng: random.Random, facts, n=6):
    avs = sorted({f.available for f in facts})
    lo, hi = avs[0] - timedelta(days=30), avs[-1] + timedelta(days=620)
    ts = []
    for _ in range(n):
        r = rng.random()
        if r < 0.35:
            ts.append(lo + timedelta(seconds=rng.random() * (hi - lo).total_seconds()))
        else:
            a = rng.choice(avs)
            ts.append(a + rng.choice([timedelta(0), timedelta(microseconds=-1), timedelta(microseconds=1),
                                      timedelta(seconds=-1), timedelta(seconds=1), timedelta(minutes=-5), timedelta(hours=1)]))
    return ts


# ----------------------------------------------------------------------------
# macro + event + model generators (Phase 98)
# ----------------------------------------------------------------------------
def gen_macro(seed: int, first_year=2012, last_year=2024):
    """List of dicts describing macro vintages (CPI index; unemployment, fed funds percent)."""
    rng = random.Random(seed * 31 + 5)
    out = []
    series = {'CPIAUCSL': ('index', 230.0, 0.002), 'UNRATE': ('percent', 6.0, 0.0), 'FEDFUNDS': ('percent', 1.0, 0.0)}
    for name, (units, level, drift) in series.items():
        v = level
        for y in range(first_year, last_year + 1):
            for m in range(1, 13):
                v = v * (1 + drift + rng.uniform(-0.001, 0.003)) if name == 'CPIAUCSL' else max(0.05, v + rng.uniform(-0.3, 0.3))
                period = date(y, m, 1)
                rel = datetime(y, m, 1, tzinfo=UTC) + timedelta(days=rng.randint(30, 55), hours=rng.randint(0, 23), minutes=rng.randint(0, 59))
                rel = rel.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)  # vintage day + 1 UTC
                out.append(dict(series=name, period=iso(period), value=round(v, 4), available=rel, units=units, revision='v1'))
                if rng.random() < 0.35:  # later revision of the same period
                    rel2 = rel + timedelta(days=rng.randint(20, 200))
                    out.append(dict(series=name, period=iso(period), value=round(v + rng.uniform(-0.2, 0.2) * (1 if name != 'CPIAUCSL' else 3), 4),
                                    available=rel2, units=units, revision='v2'))
    return out


def macro_facts(rows, canary_after=None):
    from fundamentals.store import Fact
    facts = []
    for r in rows:
        value = r['value']
        if canary_after is not None and r['available'] > canary_after and r['revision'] == 'v2':
            value = CANARY
        facts.append(Fact('MACRO:' + r['series'], r['series'], value, r['period'], r['available'].isoformat(), r['available'].isoformat(),
                          'ALFRED/graph', r['revision'], '', r['units'], '', OBSERVED, 'macro', {}))
    return facts


def gen_events(seed: int, facts, symbol='ACME'):
    rng = random.Random(seed * 17 + 3)
    avs = sorted(f.available for f in facts)
    ev = []
    for i in range(rng.randint(3, 10)):
        a = rng.choice(avs) + timedelta(minutes=rng.randint(-600, 600))
        event_at = a - timedelta(hours=rng.randint(0, 5))
        eps = round(rng.uniform(0.2, 3), 2)
        cons = round(rng.uniform(0.2, 3), 2)
        ev.append(dict(symbol=symbol, event_type=rng.choice(['earnings', 'filing', 'document_annotation', 'split']), event_at=event_at,
                       reported_at=event_at, available_at=a, source=rng.choice(['src1', 'src2']), revision=str(i),
                       payload={'event_key': f'k{i % 4}', 'reported_eps': eps, 'consensus_eps': cons,
                                'consensus_available_at': (event_at - timedelta(days=3)).isoformat(), 'ratio': 2, 'n': i}))
    return ev
