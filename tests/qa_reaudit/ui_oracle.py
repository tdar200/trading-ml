"""TRACK F (ui) independent oracle. Reads ONLY the truth tables (ui_fixture), never the DB and never
fundamentals.* calculators. Decimal arithmetic. Conventions restated from the documented contract:
a ratio/growth with a missing or non-positive denominator is None (never 0/inf/NaN)."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal as D, getcontext, ROUND_HALF_UP, ROUND_HALF_EVEN
from tests.qa_reaudit import ui_fixture as fx

getcontext().prec = 60


def ts(s) -> datetime:
    if isinstance(s, datetime):
        return s.astimezone(timezone.utc)
    s = s.strip()
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    d = datetime.fromisoformat(s)
    if d.tzinfo is None:
        raise ValueError('naive')
    return d.astimezone(timezone.utc)


def pd_(s) -> date:
    return date.fromisoformat(s)


def visible(obs, T):
    """Latest published version per (metric,start,end,currency,source) with published <= T."""
    T = ts(T)
    sel = {}
    for o in obs:
        if o.kind != 'financial' or ts(o.published) > T:
            continue
        k = (o.metric, o.start, o.end, o.currency, o.source)
        if k not in sel or ts(o.published) >= ts(sel[k].published):
            sel[k] = o
    return list(sel.values())


def ratio(a, b):
    if a is None or b is None or b <= 0:
        return None
    return a / b


def growth(a, b):
    r = ratio(a, b)
    return None if r is None else r - 1


def sub(a, b):
    return None if a is None or b is None else a - b


def add(a, b):
    return None if a is None or b is None else a + b


def days(s, e):
    return (pd_(e) - pd_(s)).days + 1


def _val(facts, metric, start, end):
    for o in facts:
        if o.metric == metric and o.end == end and o.start in (start, ''):
            return o.value
    return None


def pick_period(annual, target: date):
    best = None
    for o in annual:
        d = abs((pd_(o.end) - target).days)
        if d <= 10 and (best is None or d < best[0]):
            best = (d, o)
    return None if best is None else best[1]


def add_years(d: date, n: int) -> date:
    try:
        return d.replace(year=d.year + n)
    except ValueError:      # Feb 29
        return d.replace(year=d.year + n, day=28)


def period_vals(facts, anchor):
    if anchor is None:
        return {}
    out = {}
    for o in facts:
        if o.end != anchor.end or o.currency not in (anchor.currency, ''):
            continue
        if o.start and o.start != anchor.start:
            continue
        out[o.metric] = o.value
    return out


def snapshot(obs, T, max_age=550):
    """Oracle for the annual feature snapshot of one symbol at T."""
    T = ts(T)
    facts = visible(obs, T)
    annual = [o for o in facts if o.metric == 'revenue' and o.start and 300 <= days(o.start, o.end) <= 380 and o.value is not None]
    res = {'status': 'MISSING', 'features': {}, 'flags': [], 'period_end': None}
    if not annual:
        return res
    latest = max(annual, key=lambda o: o.end)
    end = pd_(latest.end)
    cur = pick_period(annual, end)
    pri = pick_period(annual, add_years(end, -1))
    old = pick_period(annual, add_years(end, -2))
    c, p, o2 = period_vals(facts, cur), period_vals(facts, pri), period_vals(facts, old)
    g, pv, ov = c.get, p.get, o2.get
    f = {}
    debt = add(g('short_debt'), g('long_debt'))
    fcf = sub(g('ocf'), g('capex'))
    f['fcf'] = fcf
    f['debt'] = debt
    f['net_debt'] = sub(debt, g('cash'))
    ebitda = add(g('operating_income'), g('depreciation'))
    for name, num in (('gross_margin', g('gross_profit')), ('operating_margin', g('operating_income')), ('net_margin', g('net_income')),
                      ('fcf_margin', fcf), ('ebitda_margin', ebitda)):
        f[name] = ratio(num, g('revenue'))
    avg_eq = None if g('equity') is None or pv('equity') is None else (g('equity') + pv('equity')) / 2
    avg_as = None if g('assets') is None or pv('assets') is None else (g('assets') + pv('assets')) / 2
    f['roe'] = ratio(g('net_income'), avg_eq)
    f['roa'] = ratio(g('net_income'), avg_as)
    inv_cur = sub(add(debt, g('equity')), g('cash'))
    pdebt = add(pv('short_debt'), pv('long_debt'))
    inv_pri = sub(add(pdebt, pv('equity')), pv('cash'))
    tax = ratio(g('tax_expense'), g('pretax_income'))
    nopat = None
    if tax is not None and 0 <= tax <= 1 and g('operating_income') is not None:
        nopat = g('operating_income') * (1 - tax)
    avg_inv = None if inv_cur is None or inv_pri is None else (inv_cur + inv_pri) / 2
    f['roic'] = ratio(nopat, avg_inv)
    f['current_ratio'] = ratio(g('current_assets'), g('current_liabilities'))
    f['debt_equity'] = ratio(debt, g('equity'))
    f['net_debt_ebitda'] = ratio(f['net_debt'], ebitda)
    f['interest_coverage'] = ratio(g('operating_income'), g('interest_expense'))
    f['cash_debt'] = ratio(g('cash'), debt)
    f['ocf_net_income'] = ratio(g('ocf'), g('net_income'))
    f['accrual_ratio'] = ratio(sub(g('net_income'), g('ocf')), avg_as)
    f['revenue_yoy'] = growth(g('revenue'), pv('revenue'))
    f['operating_income_yoy'] = growth(g('operating_income'), pv('operating_income'))
    f['receivables_yoy'] = growth(g('receivables'), pv('receivables'))
    f['diluted_shares_yoy'] = growth(g('diluted_shares'), pv('diluted_shares'))
    f['eps_yoy'] = growth(g('eps'), pv('eps'))
    pfcf = sub(pv('ocf'), pv('capex'))
    f['fcf_yoy'] = growth(fcf, pfcf)
    f['revenue_acceleration'] = sub(f['revenue_yoy'], growth(pv('revenue'), ov('revenue')))
    pgm = ratio(pv('gross_profit'), pv('revenue'))
    f['gross_margin_change'] = sub(f['gross_margin'], pgm)
    f['receivables_vs_sales'] = sub(f['receivables_yoy'], f['revenue_yoy'])
    for yrs in (3, 5):
        earlier = pick_period(annual, add_years(end, -yrs))
        ev = period_vals(facts, earlier).get('revenue') if earlier else None
        cg = None
        if g('revenue') is not None and ev is not None and g('revenue') > 0 and ev > 0:
            cg = (g('revenue') / ev) ** (D(1) / D(yrs)) - 1
        f[f'revenue_cagr_{yrs}y'] = cg
    flags = set()
    if fcf is not None and fcf < 0:
        flags.add('NEGATIVE_FCF')
    noise = D('1e-12')   # lead re-audit (QA-CALC-02): differences of ratios below 1e-12 are float noise, not a direction
    if f['gross_margin_change'] is not None and f['gross_margin_change'] < -noise:
        flags.add('FALLING_MARGINS')
    if f['revenue_acceleration'] is not None and f['revenue_acceleration'] < -noise:
        flags.add('REVENUE_DECELERATION')
    if f['diluted_shares_yoy'] is not None and f['diluted_shares_yoy'] > 0:
        flags.add('DILUTION')
    if f['receivables_vs_sales'] is not None and f['receivables_vs_sales'] > noise:
        flags.add('RECEIVABLES_OUTPACE_SALES')
    age = (T.date() - end).days
    used = [o for o in facts if (cur and o.end == cur.end) or (pri and o.end == pri.end) or (old and o.end == old.end)]
    res.update(features=f, flags=sorted(flags), period_end=latest.end, period_start=latest.start, currency=latest.currency,
               age_days=age, status='STALE' if age > max_age else 'AVAILABLE', raw=c,
               used_available_max=max(ts(o.published) for o in used) if used else None)
    return res


# ---------------------------------------------------------------- TTM
def ttm_alpha(obs, T):
    """Analyst formula for YTD reporters: TTM = FY(n) + YTD(n+1, latest) - YTD(n, same end)."""
    facts = visible(obs, T)
    out = {}
    fys = [o for o in facts if o.metric == 'revenue' and o.start and 360 <= days(o.start, o.end) <= 371]
    if not fys:
        return None
    fy_end = max(o.end for o in fys)
    y = int(fy_end[:4])
    res = {}
    period_end = fy_end
    for m in ('revenue', 'net_income', 'ocf', 'capex'):
        fyv = next((o.value for o in facts if o.metric == m and o.end == fy_end and o.start == f'{y}-01-01'), None)
        ytd_now = [o for o in facts if o.metric == m and o.start == f'{y + 1}-01-01']
        if fyv is None:
            res[m] = None
            continue
        if not ytd_now:
            res[m] = fyv
            continue
        latest = max(ytd_now, key=lambda o: o.end)
        prior = next((o for o in facts if o.metric == m and o.start == f'{y}-01-01' and o.end == f'{y}-' + latest.end[5:]), None)
        res[m] = None if prior is None else fyv + latest.value - prior.value
        if m == 'revenue':
            period_end = latest.end
    # annual-only EPS is not additive; ALPHA publishes no discrete Q4 EPS, so TTM EPS is unavailable by construction
    res['eps'] = None
    res['fcf'] = sub(res['ocf'], res['capex'])
    return res, period_end


def ttm_sum_quarters(obs, T, symbol_metrics=('revenue', 'net_income', 'ocf', 'capex', 'eps')):
    """Sum of the last four contiguous discrete quarters (13-week reporters)."""
    facts = visible(obs, T)
    qs = {}
    for o in facts:
        if o.metric in symbol_metrics and o.start and 60 <= days(o.start, o.end) <= 120:
            qs.setdefault(o.metric, {})[o.end] = o
    rev = qs.get('revenue', {})
    if not rev:
        return None
    end = max(rev)
    res = {}
    for m in symbol_metrics:
        q = sorted(e for e in qs.get(m, {}) if e <= end)[-4:]
        if len(q) != 4 or q[-1] != end:
            res[m] = None
            continue
        ok = all(pd_(qs[m][b].start) == pd_(a) + timedelta(days=1) for a, b in zip(q, q[1:]))
        res[m] = sum(qs[m][e].value for e in q) if ok else None
    res['fcf'] = sub(res['ocf'], res['capex'])
    start = qs['revenue'][sorted(e for e in rev if e <= end)[-4]].start if len(rev) >= 4 else None
    return res, end, start


# ---------------------------------------------------------------- macro
def macro_from_seen(seen_by_series, T):
    """seen_by_series[series] = {obs_index: (value, available_at datetime)} already filtered to available_at <= T."""
    T = ts(T)
    out = {'features': {}, 'status': None}
    stale = incomplete = False
    for series in ('CPIAUCSL', 'UNRATE', 'FEDFUNDS'):
        seen = seen_by_series.get(series) or {}
        if not seen:
            incomplete = True
            continue
        last = max(seen)
        val, av = seen[last]
        age = (T - av).days
        if age > 100:
            stale = True
            continue
        if series == 'CPIAUCSL':
            def g(k):
                return seen[k][0] if k in seen else None
            infl = growth(val, g(last - 12))
            prev = growth(g(last - 3), g(last - 15))
            out['features']['inflation_yoy'] = infl
            out['features']['inflation_change_3m'] = sub(infl, prev)
        else:
            name = 'unemployment' if series == 'UNRATE' else 'policy_rate'
            out['features'][name] = val
            out['features'][name + '_change_3m'] = sub(val, seen[last - 3][0] if (last - 3) in seen else None)
    if stale:
        out['status'] = 'STALE'
    elif incomplete or len(out['features']) != 6 or any(v is None for v in out['features'].values()):
        out['status'] = 'INCOMPLETE'
    else:
        out['status'] = 'AVAILABLE'
    return out


def macro_state(T):
    """Vintage oracle from the TRUTH tables: a vintage is usable once its day + 1 (00:00 UTC) <= T."""
    T = ts(T)
    by = {}
    for series in ('CPIAUCSL', 'UNRATE', 'FEDFUNDS'):
        seen = {}
        for vintage in sorted(fx.MACRO_VINTAGES):
            avail = ts(vintage + 'T00:00:00Z') + timedelta(days=1)
            if avail > T:
                continue
            for i in range(0, fx.MACRO_VINTAGES[vintage] + 1):
                seen[i] = (fx.macro_value(series, i, vintage), avail)
        by[series] = seen
    return macro_from_seen(by, T)


def macro_state_from_db(db_path, T):
    """Same oracle logic fed from raw SQLite macro rows (independent SQL + Python datetime filter)."""
    import sqlite3
    T = ts(T)
    by = {}
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    for series in ('CPIAUCSL', 'UNRATE', 'FEDFUNDS'):
        seen = {}
        rows = con.execute("SELECT period_end,value,available_at FROM observations WHERE symbol=? AND kind='macro' ORDER BY available_at", ('MACRO:' + series,)).fetchall()
        for pe, val, av in rows:
            if ts(av) > T or val is None:
                continue
            idx = (int(pe[:4]) - 2023) * 12 + int(pe[5:7]) - 1
            seen[idx] = (D(repr(val)), ts(av))
        by[series] = seen
    con.close()
    return macro_from_seen(by, T)


# ---------------------------------------------------------------- score
def model_score(features, model=fx.MODEL):
    """Hand arithmetic of the frozen ridge model: intercept + sum((x-mean)/scale*coef) in Decimal."""
    total = D(repr(model['intercept']))
    contrib = {}
    for name, mean, scale, coef in zip(model['features'], model['means'], model['scales'], model['coefficients']):
        x = features.get(name)
        if x is None:
            return None, None
        c = (D(x) - D(repr(mean))) / D(repr(scale)) * D(repr(coef))
        contrib[name] = c
        total += c
    return total, contrib


# ---------------------------------------------------------------- display helpers (exact formatting)
def pct2(x):
    """Python-format-independent rendering of f'{x:.2%}' (accept half-up or half-even at an exact tie)."""
    v = (D(x) * 100)
    a = v.quantize(D('0.01'), rounding=ROUND_HALF_EVEN)
    b = v.quantize(D('0.01'), rounding=ROUND_HALF_UP)
    return {_no_negative_zero(f'{a}%'), _no_negative_zero(f'{b}%')}


def _no_negative_zero(text):
    """Lead re-audit (QA-UI-09 fixed): a value that rounds to zero is displayed without a minus sign."""
    return text[1:] if text.startswith('-') and not any(ch in '123456789' for ch in text) else text


def thousands2(x):
    v = D(x)
    out = set()
    for r in (ROUND_HALF_EVEN, ROUND_HALF_UP):
        q = v.quantize(D('0.01'), rounding=r)
        out.add(_no_negative_zero(f'{q:,.2f}'))
    return out


def close(a, b, rel=1e-12, abs_=1e-15):
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def diff_snapshot(api, orc, *, rel=1e-12, abs_=1e-15):
    """List human-readable mismatches between an API snapshot dict and the oracle snapshot."""
    out = []
    if orc['status'] == 'MISSING':
        if api['status'] != 'MISSING' or api['features']:
            out.append(f"status {api['status']} features {len(api['features'])} but oracle MISSING")
        return out
    for k, v in orc['features'].items():
        pv = api['features'].get(k, 'ABSENT')
        tol = 1e-9 if k == 'fcf' and abs(float(v or 0)) < 1e-2 else abs_   # cancellation of two ~1e2 doubles
        if pv == 'ABSENT' or not close(pv, v, rel, tol):
            out.append(f'{k}: api={pv!r} oracle={None if v is None else float(v)!r}')
    if set(api['flags']) != set(orc['flags']):
        out.append(f"flags api={api['flags']} oracle={orc['flags']}")
    for k in ('period_end', 'age_days', 'status', 'currency'):
        if api[k] != orc[k]:
            out.append(f'{k}: api={api[k]!r} oracle={orc[k]!r}')
    return out


def chart_series(obs, T, metric):
    """What the history chart must plot: latest-visible version per period; annual flows (300-380 days) or instants."""
    rows = [x for x in visible(obs, T) if x.metric == metric and x.value is not None and (not x.start or 300 <= days(x.start, x.end) <= 380)]
    return sorted((x.end, x.value) for x in rows)


def expect_pct(v):
    return {'Missing'} if v is None else pct2(v)


def expect_money(v):
    return {'Missing'} if v is None else thousands2(v)


def db_obs(db_path, symbol, kind='financial'):
    """Raw SQLite rows -> Obs (independent SQL: no store/as_of code). Values come back as Decimal(repr(float))."""
    import sqlite3
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    out = []
    for r in con.execute('SELECT * FROM observations WHERE symbol=? AND kind=?', (symbol, kind)):
        meta = __import__('json').loads(r['metadata'])
        out.append(fx.Obs(r['symbol'], r['metric'], r['period_start'], r['period_end'], None if r['value'] is None else D(repr(r['value'])),
                          r['available_at'], r['revision'], r['currency'], '', r['source'], meta.get('share_basis'), r['kind']))
    con.close()
    return out


def coverage_report(db_path):
    """Phase 113: coverage by symbol x fiscal year (annual revenue period) plus sector and year roll-ups, from raw SQLite only."""
    import sqlite3, json
    HEAD = ('revenue', 'gross_profit', 'operating_income', 'net_income', 'eps', 'ocf', 'capex', 'assets', 'equity', 'cash', 'short_debt', 'long_debt', 'diluted_shares')
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT symbol,metric,period_start,period_end,value,available_at FROM observations WHERE kind='financial' AND symbol NOT LIKE 'MACRO:%'")]
    sectors = {}
    for r in con.execute("SELECT symbol,metadata FROM observations WHERE kind='classification'"):
        sectors[r['symbol']] = json.loads(r['metadata']).get('sector')
    con.close()
    table = []
    annual = {(r['symbol'], r['period_end']): r for r in rows if r['metric'] == 'revenue' and r['period_start'] and 300 <= days(r['period_start'], r['period_end']) <= 380}
    for (sym, end), rev in sorted(annual.items()):
        present = {}
        first = None
        for r in rows:
            if r['symbol'] == sym and r['period_end'] == end and r['value'] is not None and (not r['period_start'] or r['period_start'] == rev['period_start']):
                present[r['metric']] = True
                a = ts(r['available_at'])
                first = a if first is None or a < first else first
        table.append({'symbol': sym, 'sector': sectors.get(sym) or 'UNCLASSIFIED', 'fiscal_period_end': end, 'year': end[:4],
                      'present': sum(1 for m in HEAD if m in present), 'expected': len(HEAD), 'missing': [m for m in HEAD if m not in present],
                      'first_available_at': None if first is None else first.isoformat()})
    return table
