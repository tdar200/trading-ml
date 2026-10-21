"""Shared real-data database builder + production-vs-oracle comparison helpers (QA track B).

Databases are built ONCE per process in a private temp directory (never results/fundamentals/*.sqlite) by calling
the system under test's public entry points: providers.sec_facts -> store.ingest_validated.
"""
from __future__ import annotations
import atexit
import datetime as dt
import json
import shutil
import tempfile
from pathlib import Path

from fundamentals.features import FundamentalFeatureEngine
from fundamentals.providers import sec_facts
from fundamentals.store import PointInTimeFundamentalStore
from tests.qa_reaudit.real_oracle import RAW8, RAWNEW, UTC, iso_utc, load_company, days_inclusive

GROUP8 = ['MSFT', 'JPM', 'WMT', 'JNJ', 'CAT', 'XOM', 'BA', 'SONY']
GROUPNEW = ['AAPL', 'PFE', 'F', 'KO', 'NEE', 'AMZN', 'PLUG', 'NKE', 'GSK']
IFRS_ONLY = {'GSK'}          # no us-gaap facts at all
IFRS_SWITCH = {'SONY'}       # us-gaap until FY2020, ifrs-full afterwards

_SCRATCH = Path(__file__).resolve().parents[2] / 'results' / 'qa_reaudit_20261001' / 'real' / 'tmp'
_SCRATCH.mkdir(parents=True, exist_ok=True)
_TMP = Path(tempfile.mkdtemp(prefix='real_qa_', dir=_SCRATCH))     # private scratch SQLite DBs, removed at exit
atexit.register(lambda: shutil.rmtree(_TMP, ignore_errors=True))
_CACHE: dict = {}
INGEST_SUMMARY: dict = {}


def have_new_raw() -> bool:
    return all((RAWNEW / f'{s}_facts.json').exists() and (RAWNEW / f'{s}_submissions.json').exists() for s in GROUPNEW)


def build(symbols, rawdir: Path, name: str) -> Path:
    if name in _CACHE:
        return _CACHE[name]
    path = _TMP / f'{name}.sqlite'
    store = PointInTimeFundamentalStore(str(path))
    for symbol in symbols:
        payload = json.loads((rawdir / f'{symbol}_facts.json').read_text())
        subs = json.loads((rawdir / f'{symbol}_submissions.json').read_text())
        facts = sec_facts(payload, subs, symbol)
        result = store.ingest_validated(facts)
        INGEST_SUMMARY[symbol] = dict(candidates=len(facts), **result)
    store.close()
    _CACHE[name] = path
    return path


def db8() -> Path:
    return build(GROUP8, RAW8, 'real8')


def dbnew() -> Path:
    return build(GROUPNEW, RAWNEW, 'realnew')


def open_ro(path: Path) -> PointInTimeFundamentalStore:
    return PointInTimeFundamentalStore(str(path), readonly=True)


def standard_for(co, symbol: str, T) -> str:
    """Accounting standard whose latest annual revenue is the most recent (oracle-side choice)."""
    if symbol in IFRS_ONLY:
        return 'ifrs-full'
    if symbol in IFRS_SWITCH:
        cands = []
        for std in ('us-gaap', 'ifrs-full'):
            a = co.annual(T, std)
            if a:
                cands.append((a['end'], a['pub'], std))
        return max(cands)[2] if cands else 'us-gaap'
    return 'us-gaap'


def close(a, b, rel=1e-9) -> bool:
    return abs(a - b) <= rel * max(1.0, abs(b))


def compare(engine: FundamentalFeatureEngine, co, symbol: str, T: dt.datetime):
    """Return a list of discrepancy strings between production snapshot(T) and the independent oracle."""
    issues = []
    snap = engine.snapshot(symbol, iso_utc(T))
    std = standard_for(co, symbol, T)
    a = co.annual(T, std)
    if a is None:
        if snap['status'] != 'MISSING':
            issues.append(f'prod has period {snap.get("period_end")} but oracle sees no annual revenue')
        return issues, snap
    if snap['status'] == 'MISSING':
        issues.append(f'oracle sees annual {a["end"]} but production MISSING')
        return issues, snap
    if snap.get('period_end') != a['end']:
        issues.append(f'period_end prod={snap.get("period_end")} oracle={a["end"]}')
    if snap.get('period_start') != a['start']:
        issues.append(f'period_start prod={snap.get("period_start")} oracle={a["start"]}')
    raw = snap['raw']
    for metric, value in a['values'].items():
        if metric not in ('revenue', 'net_income', 'ocf', 'capex', 'eps', 'assets', 'equity', 'cash', 'diluted_shares'):
            continue
        pv = raw.get(metric)
        if pv is None:
            issues.append(f'{metric}: prod None, oracle {value}')
        elif not close(pv, value):
            issues.append(f'{metric}: prod {pv} != oracle {value}')
    for metric in ('revenue', 'net_income', 'ocf', 'capex', 'eps', 'assets', 'equity', 'cash', 'diluted_shares'):
        if raw.get(metric) is not None and metric not in a['values']:
            issues.append(f'{metric}: prod {raw[metric]} but oracle None')
    # trailing twelve months
    o = co.ttm(T, std)
    p = snap['ttm']
    if p['period_end'] != o['end']:
        issues.append(f'ttm period_end prod={p["period_end"]} oracle={o["end"]}')
    for metric in ('revenue', 'net_income', 'ocf', 'capex'):
        pv, ov = p['values'][metric], o['values'].get(metric)
        if pv is None and ov is not None:
            issues.append(f'ttm {metric}: prod None({p["reasons"].get(metric)}) oracle {ov}')
        elif pv is not None and ov is None:
            issues.append(f'ttm {metric}: prod {pv} oracle None({o["reasons"].get(metric)})')
        elif pv is not None and not close(pv, ov):
            issues.append(f'ttm {metric}: prod {pv} != oracle {ov}')
    if p['values']['eps'] is not None:
        issues.append(f'ttm eps unexpectedly available {p["values"]["eps"]} (SEC facts carry no share_basis)')
    # quarter-over-quarter growth from the last two discrete quarters (revenue, operating cash flow)
    known = co.known(T, std)
    for metric in ('revenue', 'ocf'):
        q = co.discrete_quarters(known.get(metric, {}))
        dates = sorted(q)
        want = None
        if len(dates) >= 2:
            a, b = dates[-1], dates[-2]
            gap = (dt.date.fromisoformat(a) - dt.date.fromisoformat(b)).days
            if 60 <= gap <= 120 and q[b][1] > 0:
                want = q[a][1] / q[b][1] - 1
        got = snap['features'].get(metric + '_qoq')
        if (want is None) != (got is None) or (want is not None and not close(got, want, 1e-9)):
            issues.append(f'{metric}_qoq prod={got} oracle={want}')
    return issues, snap


def random_instant(rng, lo=dt.datetime(2010, 1, 1, tzinfo=UTC), hi=dt.datetime(2026, 9, 30, tzinfo=UTC)):
    span = int((hi - lo).total_seconds())
    return lo + dt.timedelta(seconds=rng.randrange(span), microseconds=rng.randrange(1_000_000))


def fixed_dates():
    out = []
    for year in range(2018, 2026):
        for m, d in ((1, 15), (3, 1), (4, 30), (6, 15), (8, 20), (10, 31), (12, 20)):
            out.append(dt.datetime(year, m, d, 4, 0, 0, tzinfo=UTC))   # 04:00 UTC = daily bar timestamp
    return out


def periodic_filings(co, standards=('us-gaap',), forms=('10-K', '10-Q', '10-K/A', '10-Q/A', '20-F')):
    """{accn: filed} of periodic filings that carry at least one mapped fact."""
    out = {}
    for std in standards:
        for r in co.rows(std):
            if r['form'] in forms:
                out.setdefault(r['accn'], r['filed'])
    return out


def rounding_unit(values) -> float:
    """Largest power-of-1000 that divides every value (statements are rounded to this unit)."""
    for unit in (1e9, 1e6, 1e3):
        if all(v is None or float(v) % unit == 0 for v in values):
            return unit
    return 1.0


def cleanup():
    shutil.rmtree(_TMP, ignore_errors=True)
