"""TRACK F (ui) fixture: truth tables + builder for a multi-company PIT fixture DB.

The TRUTH is a list of Obs (Decimal values in whole currency units, with the real
publication instant). The SQLite fixture is generated FROM the truth, and the oracle
(ui_oracle.py) reads ONLY the truth, never the database or fundamentals.* calculators.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from decimal import Decimal as D
from pathlib import Path
from fundamentals.store import Fact, PointInTimeFundamentalStore
from fundamentals.providers import alfred_snapshot

CANARY = D('123456789123')
INSTANT = ('assets', 'equity', 'cash', 'current_assets', 'current_liabilities', 'short_debt', 'long_debt',
           'inventory', 'receivables', 'payables', 'prepaids')
SHARES = ('diluted_shares', 'shares_outstanding')
MULT = {'': 1, 'thousands': 1000, 'millions': 10 ** 6, 'billions': 10 ** 9}


@dataclass(frozen=True)
class Obs:
    symbol: str
    metric: str
    start: str
    end: str
    value: object          # Decimal or None
    published: str         # ISO UTC; available_at == reported_at == published
    revision: str
    currency: str = 'USD'
    scale: str = ''
    source: str = 'fixture'
    basis: object = 'basis-1'   # share_basis metadata for eps / diluted_shares
    kind: str = 'financial'


def to_fact(o: Obs) -> Fact:
    if o.metric in SHARES:
        cur, units, val = '', 'shares', o.value
    elif o.metric == 'eps':
        cur, units, val = o.currency, o.currency + '/shares', o.value
    else:
        cur = o.currency
        units = f'{cur} {o.scale}' if o.scale else cur
        val = None if o.value is None else o.value / MULT[o.scale]
    meta = {'accounting_standard': 'us-gaap'}
    if o.metric in ('eps', 'diluted_shares') and o.basis is not None:
        meta['share_basis'] = o.basis
    return Fact(o.symbol, o.metric, None if val is None else float(val), o.end, o.published, o.published,
                o.source, o.revision, cur, units, o.start, '2026-10-01T00:00:00Z', kind=o.kind, metadata=meta)


def fy(symbol, start, end, published, revision, vals, **kw):
    """One fiscal-year filing: flows carry [start,end], instants carry period_start ''."""
    out = []
    for m, v in vals.items():
        out.append(Obs(symbol, m, '' if m in INSTANT else start, end, None if v is None else D(str(v)), published, revision, **kw))
    return out


def M(x):  # millions -> whole units, exact Decimal
    return D(str(x)) * D(10 ** 6)


def mm(d):
    return {k: (None if v is None else M(v)) for k, v in d.items()}


# ------------------------------------------------------------------ ALPHA (calendar FY, USD, restatements, quarters)
A_FY = {
    2021: dict(revenue=80, gross_profit=32, cogs=48, operating_income=14, net_income=7, ocf=11, capex=4, depreciation=2,
               interest_expense=1, pretax_income=9, tax_expense=2, rd=3, buybacks=1, assets=150, equity=80, cash=15,
               current_assets=50, current_liabilities=28, short_debt=8, long_debt=45, inventory=8, receivables=10,
               payables=7, prepaids=1),
    2022: dict(revenue=90, gross_profit=37, cogs=53, operating_income=17, net_income=8, ocf=12.5, capex=4.5, depreciation=2,
               interest_expense=1, pretax_income=10, tax_expense=2, rd=3, buybacks=1, assets=170, equity=88, cash=18,
               current_assets=55, current_liabilities=30, short_debt=9, long_debt=42, inventory=9, receivables=11,
               payables=7.5, prepaids=1),
    2023: dict(revenue=100, gross_profit=40, cogs=60, operating_income=20, net_income=10, ocf=15, capex=5, depreciation=2.5,
               interest_expense=1, pretax_income=12.5, tax_expense=2.5, rd=3.5, buybacks=1, assets=190, equity=100, cash=20,
               current_assets=60, current_liabilities=30, short_debt=10, long_debt=40, inventory=10, receivables=12,
               payables=8, prepaids=1),
    2024: dict(revenue=120, gross_profit=48, cogs=72, operating_income=24, net_income=12, ocf=18, capex=6, depreciation=3,
               interest_expense=1.2, pretax_income=15, tax_expense=3, rd=4, buybacks=1.5, assets=210, equity=112, cash=24,
               current_assets=70, current_liabilities=35, short_debt=10, long_debt=38, inventory=11, receivables=15,
               payables=9, prepaids=1),
}
A_PUB = {2021: '2022-03-01T07:00:00Z', 2022: '2023-03-01T07:00:00Z', 2023: '2024-03-01T07:00:00Z', 2024: '2025-03-03T07:00:00Z'}
A_SHARES = {2021: 10, 2022: 10, 2023: 10, 2024: 10}
A_EPS = {2021: '0.70', 2022: '0.80', 2023: '1.00', 2024: '1.20'}
# discrete quarterly truth (millions) for FY2023 / FY2024
A_Q = {
    2023: dict(revenue=(22, 24, 26, 28), net_income=(2, 2.4, 2.6, 3), ocf=(3, 3.5, 4, 4.5), capex=(1, 1.2, 1.3, 1.5)),
    2024: dict(revenue=(27, 29, 31, 33), net_income=(2.6, 2.9, 3.1, 3.4), ocf=(4, 4.4, 4.6, 5), capex=(1.4, 1.5, 1.6, 1.5)),
}
A_QEPS = {2023: ('0.20', '0.24', '0.26'), 2024: ('0.26', '0.29', '0.31')}   # Q4 EPS never published discretely
A_QPUB = {2023: ('2023-05-04T20:30:00Z', '2023-08-03T20:30:00Z', '2023-11-02T20:30:00Z'),
          2024: ('2024-05-02T20:30:00Z', '2024-08-01T20:30:00Z', '2024-10-31T20:30:00Z')}
A_QEND = {1: ('01-01', '03-31'), 2: ('04-01', '06-30'), 3: ('07-01', '09-30'), 4: ('10-01', '12-31')}


def alpha_truth():
    out = []
    for y, vals in A_FY.items():
        v = dict(vals)
        v['diluted_shares'] = M(A_SHARES[y]); v['eps'] = A_EPS[y]
        v = {k: (x if k in ('diluted_shares', 'eps') else (None if x is None else M(x))) for k, x in v.items()}
        out += fy('ALPHA', f'{y}-01-01', f'{y}-12-31', A_PUB[y], f'0001-{y}-10K', v)
    # quarterly 10-Q style: revenue/NI discrete + YTD, OCF/CapEx YTD only, EPS discrete (Q1-Q3 only)
    for y in (2023, 2024):
        for q in (1, 2, 3):
            pub = A_QPUB[y][q - 1]; rev = f'0001-{y}-Q{q}'
            s, e = f'{y}-{A_QEND[q][0]}', f'{y}-{A_QEND[q][1]}'
            for m in ('revenue', 'net_income'):
                out.append(Obs('ALPHA', m, s, e, M(A_Q[y][m][q - 1]), pub, rev))
                if q > 1:
                    out.append(Obs('ALPHA', m, f'{y}-01-01', e, M(sum(D(str(x)) for x in A_Q[y][m][:q])), pub, rev))
            for m in ('ocf', 'capex'):
                out.append(Obs('ALPHA', m, f'{y}-01-01', e, M(sum(D(str(x)) for x in A_Q[y][m][:q])), pub, rev))
            out.append(Obs('ALPHA', 'eps', s, e, D(A_QEPS[y][q - 1]), pub, rev))
            if y == 2024:   # quarter-end balance sheets (instants)
                out.append(Obs('ALPHA', 'equity', '', e, M(100 + 4 * q), pub, rev))
                out.append(Obs('ALPHA', 'long_debt', '', e, M(40 - q), pub, rev))
    # restated FY2023 revenue (98 instead of 100) inside the FY2024 filing
    out.append(Obs('ALPHA', 'revenue', '2023-01-01', '2023-12-31', M(98), A_PUB[2024], '0001-2024-10K-comparative'))
    # 10-K/A: FY2024 EPS 1.20 -> 1.05 and net income 12 -> 10.5
    out.append(Obs('ALPHA', 'eps', '2024-01-01', '2024-12-31', D('1.05'), '2025-04-15T07:00:00Z', '0001-2024-10KA'))
    out.append(Obs('ALPHA', 'net_income', '2024-01-01', '2024-12-31', M('10.5'), '2025-04-15T07:00:00Z', '0001-2024-10KA'))
    # FUTURE CANARY: FY2025 filed 2026-03-02; every value is the canary
    canary = {m: CANARY for m in A_FY[2024]}
    canary['eps'] = CANARY; canary['diluted_shares'] = CANARY
    out += fy('ALPHA', '2025-01-01', '2025-12-31', '2026-03-02T07:00:00Z', '0001-2025-10K', canary)
    return out


# ------------------------------------------------------------------ BETA (52/53-week FY ending late Sep, USD millions)
B_FY = {
    2022: ('2021-10-03', '2022-10-01', '2022-11-18T21:00:00Z', dict(revenue=500, gross_profit=200, cogs=300, operating_income=75, net_income=55, ocf=80, capex=30, depreciation=12, interest_expense=4, pretax_income=70, tax_expense=15, rd=20, buybacks=5, assets=900, equity=400, cash=100, current_assets=300, current_liabilities=150, short_debt=20, long_debt=200, inventory=60, receivables=70, payables=50, prepaids=5), 100, '0.55'),
    2023: ('2022-10-02', '2023-09-30', '2023-11-17T21:00:00Z', dict(revenue=560, gross_profit=224, cogs=336, operating_income=84, net_income=60, ocf=85, capex=32, depreciation=13, interest_expense=4.5, pretax_income=76, tax_expense=16, rd=22, buybacks=6, assets=960, equity=430, cash=110, current_assets=320, current_liabilities=160, short_debt=22, long_debt=210, inventory=63, receivables=76, payables=54, prepaids=5), 100, '0.60'),
    2024: ('2023-10-01', '2024-09-28', '2024-11-15T21:00:00Z', dict(revenue=650, gross_profit=247, cogs=403, operating_income=91, net_income=70, ocf=95, capex=40, depreciation=15, interest_expense=5, pretax_income=88, tax_expense=18, rd=25, buybacks=8, assets=1050, equity=470, cash=95, current_assets=350, current_liabilities=190, short_debt=30, long_debt=240, inventory=80, receivables=90, payables=60, prepaids=6), 104, '0.67'),
}
B_Q = {  # discrete 13-week quarters (start, end, pub, revenue, ni, ocf, capex, eps)
    2023: [('2022-10-02', '2023-01-01', '2023-02-03T21:00:00Z', 130, 14, 20, 7, '0.14'), ('2023-01-02', '2023-04-02', '2023-05-05T21:00:00Z', 135, 15, 21, 8, '0.15'),
           ('2023-04-03', '2023-07-02', '2023-08-04T21:00:00Z', 145, 15, 22, 8, '0.15'), ('2023-07-03', '2023-09-30', '2023-11-17T21:00:00Z', 150, 16, 22, 9, '0.16')],
    2024: [('2023-10-01', '2023-12-30', '2024-02-02T21:00:00Z', 150, 16, 22, 9, '0.15'), ('2023-12-31', '2024-03-30', '2024-05-03T21:00:00Z', 155, 17, 23, 10, '0.16'),
           ('2024-03-31', '2024-06-29', '2024-08-02T21:00:00Z', 165, 18, 24, 10, '0.17'), ('2024-06-30', '2024-09-28', '2024-11-15T21:00:00Z', 180, 19, 26, 11, '0.19')],
}


def beta_truth():
    out = []
    for y, (s, e, pub, vals, sh, eps) in B_FY.items():
        v = {k: M(x) for k, x in vals.items()}; v['diluted_shares'] = M(sh); v['eps'] = D(eps)
        # BETA reports in 'USD millions': the DB stores millions, truth holds whole USD
        out += fy('BETA', s, e, pub, f'B-{y}', v, scale='millions')
    for y, qs in B_Q.items():
        for i, (s, e, pub, rev, ni, ocf, capex, eps) in enumerate(qs, 1):
            rv = f'B-{y}-Q{i}'
            for m, x in (('revenue', rev), ('net_income', ni), ('ocf', ocf), ('capex', capex)):
                out.append(Obs('BETA', m, s, e, M(x), pub, rv, scale='millions'))
            out.append(Obs('BETA', 'eps', s, e, D(eps), pub, rv))
    return out


# ------------------------------------------------------------------ GAMMA (June FY, loss making, no quarters)
G_FY = {
    2022: ('2021-07-01', '2022-06-30', '2022-08-26T11:00:00Z', dict(revenue=30, gross_profit=12, cogs=18, operating_income=-8, net_income=-9, ocf=-5, capex=2, depreciation=1, interest_expense=0.5, pretax_income=-8.9, tax_expense=0.1, assets=60, equity=20, cash=10, current_assets=25, current_liabilities=14, short_debt=2, long_debt=5, inventory=3, receivables=4, payables=3, prepaids=1), 10, '-0.90'),
    2023: ('2022-07-01', '2023-06-30', '2023-08-25T11:00:00Z', dict(revenue=42, gross_profit=17, cogs=25, operating_income=-9, net_income=-10.4, ocf=-6, capex=2.5, depreciation=1, interest_expense=0.6, pretax_income=-10.3, tax_expense=0.1, assets=62, equity=12, cash=8, current_assets=24, current_liabilities=16, short_debt=2, long_debt=6, inventory=4, receivables=5, payables=3.5, prepaids=1), 10, '-1.04'),
    2024: ('2023-07-01', '2024-06-30', '2024-08-23T11:00:00Z', dict(revenue=60, gross_profit=21, cogs=39, operating_income=-6.5, net_income=-7.8, ocf=-3, capex=3, depreciation=1.2, interest_expense=0.7, pretax_income=-7.7, tax_expense=0.1, assets=70, equity=9, cash=5, current_assets=26, current_liabilities=20, short_debt=3, long_debt=7, inventory=5, receivables=8, payables=4, prepaids=1), 10, '-0.78'),
}


def gamma_truth():
    out = []
    for y, (s, e, pub, vals, sh, eps) in G_FY.items():
        v = mm(vals); v['diluted_shares'] = M(sh); v['eps'] = D(eps)
        out += fy('GAMMA', s, e, pub, f'G-{y}', v)
    return out


# ------------------------------------------------------------------ DELTA (GBP reporter, 'GBP millions', fractional EPS)
D_FY = {
    2022: dict(revenue='1250.5', gross_profit='500.25', cogs='750.25', operating_income='187.5', net_income='125.125', ocf='200.5', capex='80.25', depreciation='30.5', interest_expense='10.25', pretax_income='160.5', tax_expense='35.375', assets='2400', equity='1000.5', cash='250.25', current_assets='700', current_liabilities='350.5', short_debt='50', long_debt='500', inventory='120', receivables='150', payables='110', prepaids='10'),
    2023: dict(revenue='1402.25', gross_profit='575', cogs='827.25', operating_income='210.75', net_income='140.3', ocf='230.7', capex='90.1', depreciation='33', interest_expense='11', pretax_income='180.2', tax_expense='39.9', assets='2550', equity='1080', cash='270', current_assets='760', current_liabilities='380', short_debt='55', long_debt='520', inventory='130', receivables='160', payables='118', prepaids='11'),
    2024: dict(revenue='1589.75', gross_profit='630.5', cogs='959.25', operating_income='240.25', net_income='160.45', ocf='260.2', capex='101.1', depreciation='36', interest_expense='12', pretax_income='205.1', tax_expense='44.65', assets='2700', equity='1190', cash='300', current_assets='830', current_liabilities='410', short_debt='60', long_debt='540', inventory='141', receivables='171', payables='125', prepaids='12'),
}
D_PUB = {2022: '2023-03-23T07:00:00Z', 2023: '2024-03-21T07:00:00Z', 2024: '2025-03-20T07:00:00Z'}
D_SHARES = {2022: '812.5', 2023: '815', 2024: '820'}
D_EPS = {2022: '0.1540', 2023: '0.1721', 2024: '0.1957'}


def delta_truth():
    out = []
    for y, vals in D_FY.items():
        v = {k: M(x) for k, x in vals.items()}; v['diluted_shares'] = M(D_SHARES[y]); v['eps'] = D(D_EPS[y])
        out += fy('DELTA', f'{y}-01-01', f'{y}-12-31', D_PUB[y], f'D-{y}', v, currency='GBP', scale='millions')
    return out


# ------------------------------------------------------------------ LATE filers (Dec FY)
def late_truth(symbol, pub24, rev=(70, 77, 84)):
    out = []
    for y, r, pub in zip((2022, 2023, 2024), rev, ('2023-03-10T21:00:00Z', '2024-03-12T21:00:00Z', pub24)):
        vals = mm(dict(revenue=r, gross_profit=r * 0.3, operating_income=r * 0.1, net_income=r * 0.05, ocf=r * 0.08, capex=r * 0.02,
                       assets=100 + 5 * (y - 2022), equity=40 + 2 * (y - 2022), cash=10, short_debt=5, long_debt=20,
                       current_assets=30, current_liabilities=15))
        vals['eps'] = D(str(round(r * 0.005, 4))); vals['diluted_shares'] = M(10)
        out += fy(symbol, f'{y}-01-01', f'{y}-12-31', pub, f'{symbol}-{y}', vals)
    return out


# ------------------------------------------------------------------ GIANT (6.8e11 revenue, 'USD billions')
def giant_truth():
    out = []
    for y, r, pub in ((2023, 600, '2024-02-01T21:00:00Z'), (2024, 680, '2025-02-03T21:00:00Z')):
        b = lambda x: D(str(x)) * D(10 ** 9)
        vals = {k: b(v) for k, v in dict(revenue=r, gross_profit=r * 0.4, cogs=r * 0.6, operating_income=r * 0.25, net_income=r * 0.2, ocf=r * 0.22 - 12 + 0,
                                         capex=35, depreciation=20, interest_expense=3, pretax_income=r * 0.23, tax_expense=r * 0.03, assets=1500, equity=600,
                                         cash=150, current_assets=400, current_liabilities=300, short_debt=30, long_debt=110).items()}
        vals['eps'] = D('9.5') if y == 2023 else D('11.25'); vals['diluted_shares'] = D(10 ** 10)
        out += fy('GIANT', f'{y}-01-01', f'{y}-12-31', pub, f'GI-{y}', vals, scale='billions')
    return out


# ------------------------------------------------------------------ NEGEQ (zero prior revenue, negative equity)
def negeq_truth():
    out = []
    for y, r, eq, pub in ((2023, 0, -3, '2024-03-05T21:00:00Z'), (2024, 25, -5, '2025-03-05T21:00:00Z')):
        vals = mm(dict(revenue=r, gross_profit=r * 0.5, operating_income=-4, net_income=-5, ocf=3, capex=1, assets=40, equity=eq, cash=5,
                       short_debt=4, long_debt=30, current_assets=15, current_liabilities=12, pretax_income=-5, tax_expense=0))
        vals['eps'] = D('-0.50'); vals['diluted_shares'] = M(10)
        out += fy('NEGEQ', f'{y}-01-01', f'{y}-12-31', pub, f'N-{y}', vals)
    return out


# ------------------------------------------------------------------ FLATCO (negative-zero display)
def flat_truth():
    out = []
    for y, r, ocf, pub in ((2023, 1_000_000_000, 100.004, '2024-03-05T21:00:00Z'), (2024, 999_999_999, 100.004, '2025-03-05T21:00:00Z')):
        vals = {'revenue': D(r), 'gross_profit': D(r) * D('0.5'), 'ocf': D(str(ocf)), 'capex': D('100.005'), 'net_income': D(r) / 10, 'operating_income': D(r) / 5,
                'diluted_shares': M(100), 'eps': D('0.1')}
        out += fy('FLATCO', f'{y}-01-01', f'{y}-12-31', pub, f'F-{y}', vals)
    return out


# ------------------------------------------------------------------ XSSCO (hostile strings in source/revision/metadata/symbol)
XSS_SOURCE = '<script>window.__xss_src=1</script>'
XSS_REV = '<img src=x onerror=window.__xss_rev=1>'


def xss_truth():
    out = []
    for y, r, pub in ((2023, 10, '2024-03-05T21:00:00Z'), (2024, 12, '2025-03-05T21:00:00Z')):
        vals = mm(dict(revenue=r, gross_profit=r * 0.5, operating_income=1, net_income=1, ocf=2, capex=1, assets=20, equity=10, cash=2, short_debt=1, long_debt=2))
        vals['eps'] = D('0.10'); vals['diluted_shares'] = M(10)
        out += fy('XSSCO', f'{y}-01-01', f'{y}-12-31', pub, XSS_REV + str(y), vals, source=XSS_SOURCE)
    return out


TRUTH_BUILDERS = {'ALPHA': alpha_truth, 'BETA': beta_truth, 'GAMMA': gamma_truth, 'DELTA': delta_truth,
                  'LATE': lambda: late_truth('LATE', '2025-06-30T20:00:00Z'), 'LATE2': lambda: late_truth('LATE2', '2025-07-20T20:00:00Z'),
                  'GIANT': giant_truth, 'NEGEQ': negeq_truth, 'FLATCO': flat_truth, 'XSSCO': xss_truth}


def all_truth():
    return {s: b() for s, b in TRUTH_BUILDERS.items()}


# ------------------------------------------------------------------ macro (ALFRED graph vintages)
def _series(first_idx, last_idx, base, step_fn):
    return {i: step_fn(i) for i in range(first_idx, last_idx + 1)}


def month_date(i):   # i=0 -> 2023-01-01
    y, m = divmod(i, 12)
    return f'{2023 + y}-{m + 1:02d}-01'


# index i: 2023-01 .. 2025-02 => 0..25
CPI = {i: D('300') + D('0.7') * i for i in range(26)}          # monotone index
UNRATE = {i: D('3.5') + D('0.05') * i for i in range(26)}      # rising
FEDFUNDS = {i: D('5.25') if i < 12 else D('5.25') - D('0.0625') * (i - 11) for i in range(26)}   # falling after 2024-01
MACRO_SERIES = {'CPIAUCSL': ('index', CPI), 'UNRATE': ('percent', UNRATE), 'FEDFUNDS': ('percent', FEDFUNDS)}
# vintage day -> last observation index included; the later vintage also revises the last two observations
MACRO_VINTAGES = {'2025-01-15': 23, '2025-03-14': 25}
MACRO_REVISION = D('0.35')   # added to the revised observations in the second vintage (indices 23, 24)


def macro_value(series, i, vintage):
    base = MACRO_SERIES[series][1][i]
    if vintage == '2025-03-14' and i in (23, 24):
        base = base + MACRO_REVISION
    return base


def macro_csv(series, vintage):
    last = MACRO_VINTAGES[vintage]
    col = f'{series}_{vintage.replace("-", "")}'
    lines = [f'observation_date,{col}']
    for i in range(0, last + 1):
        lines.append(f'{month_date(i)},{macro_value(series, i, vintage)}')
    return ('\n'.join(lines) + '\n').encode()


# ------------------------------------------------------------------ builder
def build_main_db(path, *, include=None, with_macro=True, with_events=True):
    path = Path(path)
    if path.exists():
        path.unlink()
    store = PointInTimeFundamentalStore(path)
    truth = all_truth()
    for sym, obs in truth.items():
        if include and sym not in include:
            continue
        res = store.ingest_validated([to_fact(o) for o in obs])
        assert res['status'] == 'VALIDATED', (sym, res)
    # classification fact for ALPHA (software) and DELTA (banks => debt ratios suppressed? no: keep DELTA unclassified)
    store.ingest([Fact('ALPHA', 'sector', 1, '2020-01-01', '2020-01-02T00:00:00Z', '2020-01-02T00:00:00Z', 'fixture', 'cls1', '', 'category',
                       kind='classification', metadata={'sector': 'software'}, observed_at='2026-10-01T00:00:00Z')])
    # quarantined (invalid) record: negative capex is rejected by the canonical-capex contract
    store.ingest_validated([Fact('GAMMA', 'capex', -3e6, '2024-06-30', '2024-08-23T11:00:00Z', '2024-08-23T11:00:00Z', 'vendor-x', 'bad-1', 'USD', 'USD',
                                 '2023-07-01', '2026-10-01T00:00:00Z')])
    # a hostile-looking symbol only to exercise watchlist rendering
    store.ingest([Fact('<B>BAD</B>', 'revenue', 5e6, '2024-12-31', '2025-03-05T21:00:00Z', '2025-03-05T21:00:00Z', 'fixture', 'x1', 'USD', 'USD',
                       '2024-01-01', '2026-10-01T00:00:00Z')])
    if with_macro:
        for series, (units, _) in MACRO_SERIES.items():
            for vintage in MACRO_VINTAGES:
                rows = alfred_snapshot(macro_csv(series, vintage), series, vintage, units=units, observed_at='2026-10-01T00:00:00Z')
                store.ingest(rows)
    if with_events:
        store.add_event(symbol='ALPHA', event_type='earnings', event_at='2025-03-03T07:00:00Z', reported_at='2025-03-03T07:00:00Z',
                        available_at='2025-03-03T07:00:00Z', source='fixture-8k', revision='1',
                        payload={'event_key': 'ALPHA-FY2024', 'reported_eps': 1.10, 'consensus_eps': 1.00,
                                 'consensus_available_at': '2025-02-01T00:00:00Z'})
        store.add_event(symbol='*', event_type='macro_calendar', event_at='2025-03-19T18:00:00Z', reported_at='2025-01-10T00:00:00Z',
                        available_at='2025-01-10T00:00:00Z', source='fixture-cal', revision='1', payload={'event_key': 'FOMC-2025-03', 'note': 'FOMC decision'})
        store.add_event(symbol='XSSCO', event_type='document_annotation', event_at='2025-03-05T21:00:00Z', reported_at='2025-03-05T21:00:00Z',
                        available_at='2025-03-05T21:00:00Z', source=XSS_SOURCE, revision='1',
                        payload={'event_key': 'xss-1', 'tag': 'guidance_raise',
                                 'note': '<script>window.__xss_evt=1</script> **bold** [x](javascript:window.__xss_link=1)'})
    store.close()
    return path


# frozen research model used by score/watchlist tests: hand computable
MODEL = dict(features=['revenue_yoy', 'gross_margin', 'fcf_margin'], means=[0.1, 0.4, 0.1], scales=[0.05, 0.1, 0.05],
             coefficients=[0.01, 0.02, -0.005], intercept=0.001, trained_through='2024-01-01T00:00:00+00:00', alpha=1.0,
             training_rows=100, training_sha256='ui-fixture', horizon=5)


def write_bundle(path):
    Path(path).write_text(json.dumps({'model': MODEL}))
    return Path(path)


def frozen_engine():
    from fundamentals.models import ModelArtifact, FundamentalScoreEngine
    return FundamentalScoreEngine(ModelArtifact(**{**MODEL, 'features': tuple(MODEL['features']), 'means': tuple(MODEL['means']),
                                                    'scales': tuple(MODEL['scales']), 'coefficients': tuple(MODEL['coefficients'])}))


# ------------------------------------------------------------------ edge DB: ambiguity, currency switch, overflow
def ambig_truth():
    out = []
    for y, r, pub in ((2022, 90, '2023-03-01T07:00:00Z'), (2023, 100, '2024-03-01T07:00:00Z'), (2024, 110, '2025-03-01T07:00:00Z')):
        for src, rr in (('srcA', r), ('srcB', r + 5)):     # two vendors disagree on revenue (and only revenue)
            out += fy('AMBIG', f'{y}-01-01', f'{y}-12-31', pub, f'{src}-{y}', {'revenue': M(rr)}, source=src)
        out += fy('AMBIG', f'{y}-01-01', f'{y}-12-31', pub, f'srcA-{y}', {'gross_profit': M(rr * 0.4)}, source='srcA')
    return out


def switch_truth():
    """Reporting currency changes GBP (FY2022, FY2023) -> USD (FY2024)."""
    out = []
    for y, r, cur, pub in ((2022, 1000, 'GBP', '2023-03-01T07:00:00Z'), (2023, 1100, 'GBP', '2024-03-01T07:00:00Z'), (2024, 1500, 'USD', '2025-03-01T07:00:00Z')):
        out += fy('SWITCH', f'{y}-01-01', f'{y}-12-31', pub, f'S-{y}', {'revenue': M(r), 'gross_profit': M(r * 0.4), 'ocf': M(r * 0.1), 'capex': M(r * 0.03), 'equity': M(r * 0.5)},
                  currency=cur, scale='millions')
    return out


def overflow_truth():
    out = []
    for y, pub in ((2022, '2023-03-01T07:00:00Z'), (2023, '2024-03-01T07:00:00Z'), (2024, '2025-03-01T07:00:00Z')):
        vals = {'revenue': D('1.0E+308'), 'gross_profit': D('1.0E+308'), 'net_income': D('1.5E+308') if y != 2023 else D('1.7E+308')}
        out += fy('HUGE', f'{y}-01-01', f'{y}-12-31', pub, f'H-{y}', vals)
    return out


def build_edge_db(path, which=('ALPHA', 'AMBIG', 'SWITCH', 'HUGE')):
    path = Path(path)
    if path.exists():
        path.unlink()
    store = PointInTimeFundamentalStore(path)
    builders = {'ALPHA': alpha_truth, 'AMBIG': ambig_truth, 'SWITCH': switch_truth, 'HUGE': overflow_truth}
    for sym in which:
        res = store.ingest_validated([to_fact(o) for o in builders[sym]()])
        assert res['status'] == 'VALIDATED', (sym, res)
    store.close()
    return path
