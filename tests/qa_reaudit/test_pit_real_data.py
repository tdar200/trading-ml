"""PIT checks on the REAL SEC cache of the 8 audited companies (read-only evidence under results/fundamental_audit_20260930/raw).

Independent of production: availability is re-derived here from the raw submissions JSON (acceptance + 5 minutes, else
the start of the next New York calendar day after the filing date) and compared with providers.sec_facts for EVERY fact;
then random instants are probed (full database vs a database that physically lacks everything published after T) and every
real restatement is probed one microsecond either side of its publication instant.
"""
from __future__ import annotations
import collections
import json
import random
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tests.qa_reaudit import pit_helpers as H
from tests.qa_reaudit import pit_oracle as O

from fundamentals.features import FundamentalFeatureEngine
from fundamentals.providers import sec_facts
from fundamentals.store import PointInTimeFundamentalStore

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'results' / 'fundamental_audit_20260930' / 'raw'
SYMBOLS = ['BA', 'CAT', 'JNJ', 'JPM', 'MSFT', 'SONY', 'WMT', 'XOM']
NY = ZoneInfo('America/New_York')
UTC = timezone.utc

pytestmark = pytest.mark.skipif(not RAW.exists(), reason='raw SEC cache not present')

_cache = {}


def load(symbol):
    if symbol not in _cache:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            payload = json.loads((RAW / f'{symbol}_facts.json').read_text())
            subs = json.loads((RAW / f'{symbol}_submissions.json').read_text())
            facts = sec_facts(payload, subs, symbol, observed_at='2030-01-01T00:00:00Z')
        recent = subs.get('filings', {}).get('recent', {})
        acc = dict(zip(recent.get('accessionNumber', []), recent.get('acceptanceDateTime', [])))
        _cache[symbol] = (facts, acc)
    return _cache[symbol]


def indep_available(accn, filed, acc_map):
    acc = acc_map.get(accn)
    if acc:
        t = datetime.fromisoformat(acc.replace('Z', '+00:00'))
        if t.tzinfo is None:
            t = t.replace(tzinfo=NY)
        return t.astimezone(UTC) + timedelta(minutes=5), 'acceptance'
    y, m, d = map(int, filed.split('-'))
    nxt = datetime(y, m, d, tzinfo=NY) + timedelta(days=1)
    return datetime(nxt.year, nxt.month, nxt.day, tzinfo=NY).astimezone(UTC), 'date-only'


def instant(dt):
    return int(dt.timestamp()) * 10**9 + dt.microsecond * 1000


@pytest.mark.parametrize('symbol', SYMBOLS)
def test_independent_availability_equals_production_for_every_real_fact(symbol):
    facts, acc = load(symbol)
    kinds = collections.Counter()
    bad = []
    for f in facts:
        exp, kind = indep_available(f.revision, f.metadata['filing_date'], acc)
        kinds[kind] += 1
        if H.ns_instant(f.available_at) != instant(exp):
            bad.append((f.revision, f.metric, f.available_at, exp.isoformat()))
        # never earlier than the end of the measured period, never before the reported instant
        assert H.ns_instant(f.available_at) >= H.ns_instant(f.reported_at)
    assert not bad, bad[:5]
    assert len(facts) > 500 and kinds['acceptance'] + kinds['date-only'] == len(facts)


def versions(facts):
    """key -> list of (available_ns, value) per distinct accession (my own grouping from the raw facts)."""
    g = collections.defaultdict(list)
    for f in facts:
        g[(f.metric, f.period_start, f.period_end, f.currency, f.units, f.source)].append(f)
    return g


@pytest.mark.parametrize('symbol', SYMBOLS)
def test_every_real_restatement_flips_exactly_at_the_later_filing_instant(symbol):
    facts, acc = load(symbol)
    store = PointInTimeFundamentalStore(':memory:')
    store.ingest_validated(facts)
    restated = 0
    probed = 0
    rng = random.Random(symbol)
    keys = []
    for key, fs in versions(facts).items():
        by_time = sorted(((H.ns_instant(f.available_at), f.value, f.revision) for f in fs))
        # collapse equal consecutive values: a "restatement" is a change of the visible value
        changes = [(t, v) for (t, v, _), prev in zip(by_time, [None] + by_time[:-1]) if prev is None or prev[1] != v]
        if len(changes) >= 2:
            keys.append((key, changes))
    restated = len(keys)
    for key, changes in rng.sample(keys, min(25, len(keys))):
        for (t0, v0), (t1, v1) in zip(changes, changes[1:]):
            def val_at(ns):
                text = datetime.fromtimestamp(ns // 10**9, UTC).isoformat()
                micro = (ns % 10**9) // 1000
                if micro:
                    text = datetime.fromtimestamp(ns // 10**9, UTC).replace(microsecond=micro).isoformat()
                rows = [r for r in store.as_of(symbol, text, kind='financial') if (r['metric'], r['period_start'], r['period_end'], r['currency'], r['units'], r['source']) == key]
                return rows[0]['value'] if rows else 'ABSENT'
            before = val_at(t1 - 1000)  # one microsecond earlier
            at = val_at(t1)
            assert before == v0, f'{symbol} {key}: at t-1us expected old value {v0}, got {before}'
            assert at == v1, f'{symbol} {key}: at t expected restated {v1}, got {at}'
            probed += 1
    assert restated > 20, f'{symbol}: only {restated} restated keys in real data'
    assert probed > 20


@pytest.mark.parametrize('symbol', SYMBOLS)
def test_real_data_phase98_truncated_database_gives_identical_snapshot_and_ttm(symbol):
    facts, acc = load(symbol)
    rng = random.Random(f'p98-{symbol}')
    avail = sorted({H.ns_instant(f.available_at) for f in facts})
    probes = []
    for _ in range(3):
        probes.append(rng.randint(avail[len(avail) // 10], avail[-1]))
    # plus exact publication instants of random filings
    for t in rng.sample(avail, 2):
        probes += [t, t - 1000]
    full = PointInTimeFundamentalStore(':memory:')
    full.ingest_validated(facts)
    engine_f = FundamentalFeatureEngine(full)
    for ns in probes:
        T = datetime.fromtimestamp(ns // 10**9, UTC).replace(microsecond=(ns % 10**9) // 1000)
        keep = [f for f in facts if H.ns_instant(f.available_at) <= ns]
        trunc = PointInTimeFundamentalStore(':memory:')
        trunc.ingest_validated(keep)
        a = engine_f.snapshot(symbol, T.isoformat())
        b = FundamentalFeatureEngine(trunc).snapshot(symbol, T.isoformat())
        assert O.canon(a) == O.canon(b), f'{symbol} T={T.isoformat()}: snapshot differs between full and truncated real database'
        if a.get('available_at'):
            assert H.ns_instant(a['available_at']) <= ns
        for r in full.as_of(symbol, T.isoformat()):
            assert H.ns_instant(r['available_at']) <= ns


# ---------------------------------------------------------------------------
# the SHIPPED database (XOM + ALFRED macro + filing events): PHASE 98 by physical truncation through ATTACH/INSERT..SELECT
# ---------------------------------------------------------------------------
SHIPPED = ROOT / 'results' / 'fundamentals' / 'fundamentals.sqlite'


@pytest.mark.skipif(not SHIPPED.exists(), reason='shipped database not present')
def test_shipped_database_phase98_truncation_identity_for_random_instants(tmp_path):
    import shutil
    import sqlite3
    from tests.qa_reaudit.pit_phase98 import make_model
    from fundamentals.api import FundamentalAPI
    from fundamentals.models import CombinedSignalEngine, EventCatalystEngine, FundamentalScoreEngine, MacroRegimeService
    src = tmp_path / 'full.sqlite'
    shutil.copy(SHIPPED, src)
    for suffix in ('-wal', '-shm'):
        p = Path(str(SHIPPED) + suffix)
        if p.exists() and p.stat().st_size:
            shutil.copy(p, str(src) + suffix)
    full = PointInTimeFundamentalStore(src)
    assert full.db.execute("select count(*) from sqlite_master where type='trigger'").fetchone()[0] >= 14
    n_obs = full.db.execute('select count(*) from observations').fetchone()[0]
    assert n_obs > 5000
    ts = [r[0] for r in full.db.execute("select distinct available_at from observations where symbol='XOM'")]
    rng = random.Random(98)
    ns = sorted(H.ns_instant(t) for t in ts)
    instants = [rng.randint(ns[len(ns) // 5], ns[-1] + 10**15) for _ in range(5)] + [ns[len(ns) // 2], ns[len(ns) // 2] - 1000, ns[-3]]
    for k, t_ns in enumerate(instants):
        T = datetime.fromtimestamp(t_ns // 10**9, UTC).replace(microsecond=(t_ns % 10**9) // 1000)
        tq = T.isoformat()
        trunc_path = tmp_path / f'trunc{k}.sqlite'
        trunc = PointInTimeFundamentalStore(trunc_path)
        trunc.db.execute('ATTACH DATABASE ? AS src', (str(src),))
        trunc.db.execute('INSERT INTO observations SELECT * FROM src.observations WHERE available_at<=?', (tq,))
        trunc.db.execute('INSERT INTO corporate_events SELECT * FROM src.corporate_events WHERE available_at<=?', (tq,))
        trunc.db.commit()
        trunc.db.execute('DETACH DATABASE src')
        assert trunc.db.execute('select count(*) from observations where available_at>?', (tq,)).fetchone()[0] == 0
        outs = []
        for st in (full, trunc):
            combo = CombinedSignalEngine(FundamentalFeatureEngine(st), FundamentalScoreEngine(make_model()), MacroRegimeService(st), EventCatalystEngine(st, 30.),
                                         policy='filter', research_shadow=True)
            api = FundamentalAPI(st)
            outs.append({'decision': combo.decide('XOM', tq, {'signal': True}), 'snapshot': api.get('/fundamentals/XOM', {'as_of': tq}),
                         'macro': api.get('/macro/regime', {'as_of': tq}), 'events': api.get('/fundamentals/XOM/events', {'as_of': tq}),
                         'history': api.get('/fundamentals/XOM/history', {'as_of': tq}), 'watch': api.get('/watchlist/fundamental', {'as_of': tq})})
        for name in outs[0]:
            assert O.canon(outs[0][name]) == O.canon(outs[1][name]), f'shipped DB T={tq}: {name} differs between full and truncated database'
        trunc.close()
