"""Exact publication-boundary, timestamp-format, timezone/DST and SQL string-comparison attacks (QA track A).

Expected visibility is decided by an independent integer-nanosecond parser in pit_helpers.ns_instant,
never by pandas or by the production SQL.
"""
from __future__ import annotations
import json
import random
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tests.qa_reaudit import pit_helpers as H
from tests.qa_reaudit import pit_oracle as O

from fundamentals.api import FundamentalAPI
from fundamentals.features import FundamentalFeatureEngine
from fundamentals.store import Fact, PointInTimeFundamentalStore

PUB = '2025-05-10T07:00:00Z'
PUB_NS = H.ns_instant(PUB)

QUERIES = [
    '2025-04-15T00:00:00Z', '2025-05-09T00:00:00Z', '2025-05-09T23:59:59Z', '2025-05-10T06:59:59Z',
    '2025-05-10T06:59:59.999Z', '2025-05-10T06:59:59.999999Z', '2025-05-10T06:59:59.999999999Z',
    '2025-05-10T07:00:00Z', '2025-05-10T07:00:00+00:00', '2025-05-10T07:00:00.000Z', '2025-05-10T07:00:00.000000000Z',
    '2025-05-10T07:00:00.000000001Z', '2025-05-10T07:00:00.000001Z', '2025-05-10T07:00:00.5Z', '2025-05-10T07:01:00Z',
    '2025-05-10T08:00:00+01:00', '2025-05-10T07:59:59+01:00', '2025-05-10T07:59:59.999999999+01:00', '2025-05-10T08:00:00.000000001+01:00',
    '2025-05-10T03:00:00-04:00', '2025-05-10T02:59:59.999999-04:00', '2025-05-10T02:59:59-04:00',
    '2025-05-10T12:30:00+05:30', '2025-05-10T12:29:59+05:30', '2025-05-09T23:00:00-08:00', '2025-05-09T22:59:59-08:00',
    '2025-05-10T16:00:00+09:00', '2025-05-10T15:59:59+09:00', '2025-05-10T00:00:00-07:00', '2025-05-10T00:00:01-07:00',
]


@pytest.fixture(scope='module')
def acme_db(tmp_path_factory):
    path = tmp_path_factory.mktemp('pit') / 'acme.sqlite'
    s = PointInTimeFundamentalStore(path)
    s.ingest(H.acme_nov_fy(PUB))
    s.close()
    return path


@pytest.mark.parametrize('query', QUERIES)
def test_publication_boundary_store_snapshot_api_and_http(acme_db, query):
    visible = H.ns_instant(query) >= PUB_NS
    s = PointInTimeFundamentalStore(acme_db)
    got = {r['period_end'] for r in s.as_of('ACME', query, kind='financial') if r['metric'] == 'revenue'}
    assert ('2025-03-31' in got) == visible, f'store.as_of {query}'
    snap = FundamentalFeatureEngine(s).snapshot('ACME', query)
    assert snap['period_end'] == ('2025-03-31' if visible else '2024-03-31'), f'snapshot {query}'
    assert snap['features']['revenue_yoy'] == pytest.approx(0.25 if visible else 800 / 700 - 1, rel=1e-12)
    assert snap['features']['fcf'] == pytest.approx(100e6 if visible else 80e6, rel=1e-12)
    assert snap['available_at'] == ('2025-05-10T07:00:00+00:00' if visible else '2024-05-12T07:00:00+00:00')
    api = FundamentalAPI(s).get('/fundamentals/ACME', {'as_of': query})
    assert api['period_end'] == snap['period_end']
    hist = FundamentalAPI(s).get('/fundamentals/ACME/history', {'as_of': query, 'metric': 'revenue'})['records']
    assert any(r['period_end'] == '2025-03-31' for r in hist) == visible, f'history {query}'
    ttm = FundamentalAPI(s).get('/fundamentals/ACME/ttm', {'as_of': query})
    assert ttm['as_of'] == snap['as_of']
    s.close()


@pytest.mark.parametrize('query', QUERIES)
def test_publication_boundary_over_real_http(acme_db, query):
    visible = H.ns_instant(query) >= PUB_NS
    with H.http_server(acme_db) as get:
        code, body, headers, raw = get('/fundamentals/ACME?as_of=' + quote(query, safe=''))
        assert code == 200
        assert body['period_end'] == ('2025-03-31' if visible else '2024-03-31')
        assert headers['Cache-Control'] == 'no-store'
        assert b'NaN' not in raw and b'Infinity' not in raw
        # the response echoes the instant normalised to UTC with explicit offset
        assert re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d{6,9})?\+00:00', body['as_of'])
        assert H.ns_instant(body['as_of']) == H.ns_instant(query)


# ---------------------------------------------------------------------------
# random mixed-precision / mixed-offset ordering attack on the SQL string comparison
# ---------------------------------------------------------------------------
_OFFS = [('+00:00', 0), ('Z', 0), ('+01:00', 60), ('-04:00', -240), ('+05:30', 330), ('-08:00', -480), ('+09:00', 540), ('-03:30', -210), ('+13:00', 780)]
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def render_ns(ns: int, rng: random.Random) -> str:
    secs, frac = divmod(ns, 10**9)
    digits_ok = [d for d in range(0, 10) if d == 0 and frac == 0 or d > 0 and frac % 10 ** (9 - d) == 0]
    d = rng.choice(digits_ok)
    label, minutes = rng.choice(_OFFS)
    local = EPOCH + timedelta(seconds=secs) + timedelta(minutes=minutes)
    text = f'{local.year:04d}-{local.month:02d}-{local.day:02d}T{local.hour:02d}:{local.minute:02d}:{local.second:02d}'
    if d:
        text += '.' + f'{frac:09d}'[:d]
    if label == 'Z':
        return text + 'Z'
    return text + label


@pytest.mark.parametrize('seed', range(200), ids=lambda s: f'seed{s}')
def test_sql_string_comparison_mixed_precision_offsets_years(seed):
    rng = random.Random(seed)
    s = PointInTimeFundamentalStore(':memory:')
    n = rng.randint(6, 14)
    anchors = [rng.choice([rng.randint(-2_000_000_000, 4_100_000_000), rng.randint(1_700_000_000, 1_800_000_000)]) * 10**9 + rng.randint(0, 999_999_999) for _ in range(3)]
    inst = []
    for i in range(n):
        base = rng.choice(anchors)
        delta = rng.choice([0, 1, -1, 999, 1000, 1001, 10**6, -10**6, 10**9, -10**9, 12345678, rng.randint(-10**11, 10**11)])
        ns = base + delta
        if rng.random() < 0.5:
            ns -= ns % rng.choice([1, 1000, 10**6, 10**9])  # trims to coarser precision
        inst.append(ns)
    facts = []
    for i, ns in enumerate(inst):
        txt = render_ns(ns, rng)
        # period_end must not be after the publication day: use 1900-01-01
        facts.append(Fact('ZZ', f'zz{i}', float(i + 1), '1900-01-01', txt, txt, 'SRC', 'r', 'USD', 'USD', '', H.OBS, 'financial', {}))
    s.ingest(facts)
    # stored text is canonical UTC (the premise of the SQL string comparison)
    for r in s.db.execute('select reported_at,available_at from observations'):
        for v in r:
            assert re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d{6}|\.\d{9})?\+00:00', v), v
    probes = []
    for ns in inst:
        for d in (-10**9, -1000, -1, 0, 1, 1000, 10**9):
            probes.append(ns + d)
    probes += [rng.randint(-2_000_000_000, 4_100_000_000) * 10**9 for _ in range(5)]
    for t_ns in probes:
        tq = render_ns(t_ns, rng)
        expected = {f'zz{i}' for i, ns in enumerate(inst) if ns <= t_ns}
        got = {r['metric'] for r in s.as_of('ZZ', tq, kind='financial')}
        assert got == expected, f'seed={seed} T={tq} (ns={t_ns})'
        hist = {r['metric'] for r in s.history('ZZ', as_of=tq)}
        assert hist == expected, f'seed={seed} history T={tq}'
    s.close()


# ---------------------------------------------------------------------------
# DST transition days: US 2025-03-09 / 2025-11-02, UK 2025-03-30 / 2025-10-26
# ---------------------------------------------------------------------------
DST_CASES = [
    ('America/New_York', datetime(2025, 3, 9, 7, 0, tzinfo=timezone.utc)),    # EST->EDT, gap 02:00-03:00 local
    ('America/New_York', datetime(2025, 11, 2, 6, 0, tzinfo=timezone.utc)),   # EDT->EST, 01:00-02:00 repeated
    ('Europe/London', datetime(2025, 3, 30, 1, 0, tzinfo=timezone.utc)),      # GMT->BST, gap 01:00-02:00 local
    ('Europe/London', datetime(2025, 10, 26, 1, 0, tzinfo=timezone.utc)),     # BST->GMT, 01:00-02:00 repeated
]


@pytest.mark.parametrize('zone,transition', DST_CASES, ids=['us-spring', 'us-fall', 'uk-spring', 'uk-fall'])
def test_dst_transition_days_visibility_follows_utc_instant(zone, transition):
    tz = ZoneInfo(zone)
    offsets = [timedelta(seconds=-1), timedelta(0), timedelta(microseconds=-1), timedelta(microseconds=1), timedelta(minutes=-30),
               timedelta(minutes=30), timedelta(hours=-1), timedelta(hours=1), timedelta(seconds=1)]
    pub = transition  # fact published exactly at the transition instant
    s = PointInTimeFundamentalStore(':memory:')
    local_pub = pub.astimezone(tz)
    # published with the *local* offset text (the form a local provider would send)
    s.ingest([Fact('D', 'zz', 1.0, '2025-01-01', local_pub.isoformat(), local_pub.isoformat(), 'SRC', 'r', 'USD', 'USD', '', H.OBS, 'financial', {})])
    for off in offsets:
        t_utc = pub + off
        for form in ('utc_text', 'local_text', 'zoneinfo_obj', 'pandas_tz'):
            if form == 'utc_text':
                q = t_utc.isoformat()
            elif form == 'local_text':
                q = t_utc.astimezone(tz).isoformat()
            elif form == 'zoneinfo_obj':
                q = t_utc.astimezone(tz)
            else:
                q = pd.Timestamp(t_utc).tz_convert(zone)
            expected = off >= timedelta(0)
            got = bool(s.as_of('D', q, kind='financial'))
            assert got == expected, f'{zone} {transition} off={off} form={form} q={q!r}'
    # snapshots/APIs use the same instant semantics
    s.close()


def test_local_wall_clock_without_offset_is_rejected_everywhere_even_on_dst_days():
    s = PointInTimeFundamentalStore(':memory:')
    for naive in ('2025-03-09T03:00:00', '2025-11-02T01:30:00', '2025-03-30T01:30:00', '2025-10-26T01:30:00', '2025-05-10', '2025-05-10 07:00'):
        with pytest.raises(ValueError):
            s.as_of('A', naive)
        with pytest.raises(ValueError):
            s.history('A', as_of=naive)
        with pytest.raises(ValueError):
            FundamentalAPI(s).get('/fundamentals/A', {'as_of': naive})
        with pytest.raises(ValueError):
            FundamentalFeatureEngine(s).snapshot('A', naive)
        with pytest.raises(ValueError):
            s.ingest([Fact('A', 'zz', 1., '2020-01-01', naive, '2025-12-31T00:00:00Z', 'S', 'r', 'USD', 'USD', '', H.OBS, 'financial', {})])
        with pytest.raises(ValueError):
            s.ingest([Fact('A', 'zz', 1., '2020-01-01', '2020-01-02T00:00:00Z', naive, 'S', 'r', 'USD', 'USD', '', H.OBS, 'financial', {})])
        with pytest.raises(ValueError):
            s.ingest([Fact('A', 'zz', 1., '2020-01-01', '2020-01-02T00:00:00Z', '2020-01-02T00:00:00Z', 'S', 'r', 'USD', 'USD', '', naive, 'financial', {})])
        with pytest.raises(ValueError):
            s.add_event(symbol='A', event_type='x', event_at=naive, reported_at='2020-01-01T00:00:00Z', available_at='2020-01-01T00:00:00Z', source='s', revision='1', payload={})
    # nothing was written by any of the rejected calls
    assert s.db.execute('select count(*) from observations').fetchone()[0] == 0
    assert s.db.execute('select count(*) from corporate_events').fetchone()[0] == 0
    for bad in (pd.NaT, None, '', 'garbage', float('nan')):
        with pytest.raises((ValueError, TypeError)):
            s.as_of('A', bad)


def test_ingest_validated_quarantines_naive_and_backdated_records_instead_of_storing():
    s = PointInTimeFundamentalStore(':memory:')
    good = Fact('A', 'zz', 1., '2020-01-01', '2020-02-01T00:00:00Z', '2020-02-01T00:00:00Z', 'S', 'r', 'USD', 'USD', '', H.OBS, 'financial', {})
    naive = Fact('A', 'zz2', 1., '2020-01-01', '2020-02-01T00:00:00', '2020-02-01T00:00:00Z', 'S', 'r', 'USD', 'USD', '', H.OBS, 'financial', {})
    before_period = Fact('A', 'zz3', 1., '2020-06-30', '2020-02-01T00:00:00Z', '2020-02-01T00:00:00Z', 'S', 'r', 'USD', 'USD', '', H.OBS, 'financial', {})
    avail_before_reported = Fact('A', 'zz4', 1., '2020-01-01', '2020-02-01T00:00:00Z', '2020-01-31T23:59:59Z', 'S', 'r', 'USD', 'USD', '', H.OBS, 'financial', {})
    out = s.ingest_validated([good, naive, before_period, avail_before_reported])
    assert out == {'accepted': 1, 'rejected': 3, 'status': 'QUARANTINED'}
    assert [r['metric'] for r in s.as_of('A', '2021-01-01T00:00:00Z')] == ['zz']
    reasons = sorted(r['reason'] for r in s.quality('A')['rejections'])
    assert reasons == ['available_at precedes publication', 'explicit timezone required', 'observation precedes measured period']


def test_ingest_json_records_non_utc_text_is_canonicalised_before_storage(tmp_path):
    from fundamentals.providers import ingest_json_records
    recs = [
        dict(symbol='J', metric='revenue', value=5.0, period_end='2024-12-31', reported_at='2025-02-20T08:00:00+01:00', available_at='2025-02-20T08:00:00+01:00',
             source='x', revision='1', currency='USD', units='USD', period_start='2024-01-01'),
        dict(symbol='J', metric='revenue', value=6.0, period_end='2024-12-31', reported_at='2025-03-20T03:00:00-04:00', available_at='2025-03-20T03:00:00-04:00',
             source='x', revision='2', currency='USD', units='USD', period_start='2024-01-01'),
        dict(symbol='J', metric='ocf', value=1.0, period_end='2024-12-31', reported_at='2025-02-20T12:30:00+05:30', available_at='2025-02-20T12:30:00+05:30',
             source='x', revision='1', currency='USD', units='USD', period_start='2024-01-01'),
    ]
    p = tmp_path / 'recs.json'
    p.write_text(json.dumps(recs))
    s = PointInTimeFundamentalStore(':memory:')
    ingest_json_records(s, p)
    stored = {r['available_at'] for r in s.db.execute('select available_at from observations')}
    assert stored == {'2025-02-20T07:00:00+00:00', '2025-03-20T07:00:00+00:00'}
    assert s.as_of('J', '2025-02-20T06:59:59.999999Z') == []
    assert [r['value'] for r in s.as_of('J', '2025-02-20T07:00:00Z') if r['metric'] == 'revenue'] == [5.0]
    assert [r['value'] for r in s.as_of('J', '2025-03-20T06:59:59.999999999Z') if r['metric'] == 'revenue'] == [5.0]
    assert [r['value'] for r in s.as_of('J', '2025-03-20T07:00:00Z') if r['metric'] == 'revenue'] == [6.0]
    # a record with a naive timestamp rejects the entire file, nothing partial is stored
    bad = recs + [dict(recs[0], metric='cash', reported_at='2025-02-20T08:00:00', available_at='2025-02-20T08:00:00')]
    q = tmp_path / 'bad.json'
    q.write_text(json.dumps(bad))
    s2 = PointInTimeFundamentalStore(':memory:')
    with pytest.raises(ValueError):
        ingest_json_records(s2, q)
    assert s2.db.execute('select count(*) from observations').fetchone()[0] == 0


@pytest.mark.parametrize('seed', range(12), ids=lambda s: f'seed{s}')
def test_every_stored_timestamp_column_is_canonical_utc(seed, tmp_path):
    """The premise of all string comparisons: every time column in every table is canonical UTC text."""
    company = O.gen_company(seed)
    store = O.build_store(company.facts, company.symbol, company.scale_label, seed)
    macro = O.macro_facts(O.gen_macro(seed)[:30])
    store.ingest(macro)
    for e in O.gen_events(seed, company.facts):
        store.add_event(symbol=e['symbol'], event_type=e['event_type'], event_at=e['event_at'].isoformat(), reported_at=e['reported_at'].isoformat(),
                        available_at=e['available_at'].isoformat(), source=e['source'], revision=e['revision'], payload=e['payload'])
    store.ingest_validated([Fact('X', 'revenue', 1., '2020-12-31', '2021-01-01T00:00:00+02:00', '2021-01-01T00:00:00+02:00', 'S', 'r', 'XXX', 'XXX', '', H.OBS, 'financial', {})])
    store.archive('src', 'body', '2021-01-01T00:00:00+05:00')
    store.save('feature_snapshots', FundamentalFeatureEngine(store).snapshot(company.symbol, '2030-01-01T00:00:00-05:00'))
    pat = re.compile(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d{6}|\.\d{9})?\+00:00')
    cols = {'observations': ('reported_at', 'available_at', 'observed_at'), 'corporate_events': ('event_at', 'reported_at', 'available_at'),
            'ingestion_rejections': ('detected_at',), 'raw_documents': ('fetched_at',), 'feature_snapshots': ('as_of',)}
    for table, names in cols.items():
        for name in names:
            for (v,) in store.db.execute(f'select {name} from {table}'):
                assert pat.fullmatch(v), f'{table}.{name}={v!r}'


# ---------------------------------------------------------------------------
# freshness: max_period_age_days = 550 boundary (phases 46/155)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('age,status', [(1, 'AVAILABLE'), (2, 'AVAILABLE'), (549, 'AVAILABLE'), (550, 'AVAILABLE'), (551, 'STALE'), (552, 'STALE'), (2000, 'STALE')])
def test_stale_threshold_exact_day_boundary(age, status):
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest(H.fy_facts('fy', '2023-01-01', '2023-12-31', '2024-01-01T00:00:00Z', 100 * H.M))
    base = datetime(2023, 12, 31, tzinfo=timezone.utc) + timedelta(days=age)
    engine = FundamentalFeatureEngine(s)
    for delta in (timedelta(0), timedelta(hours=23, minutes=59, seconds=59, microseconds=999999)):
        snap = engine.snapshot('ACME', (base + delta).isoformat())
        assert snap['age_days'] == age
        assert snap['status'] == status, f'age={age} +{delta}'
    # the day counter flips at 00:00 UTC, not at local midnight: 23:30 at -04:00 on day 550 is already day 551 UTC
    if age == 550:
        local = (base + timedelta(hours=23, minutes=30)).astimezone(timezone(timedelta(hours=-4)))
        assert local.day == base.day  # still the same local calendar day...
        snap = engine.snapshot('ACME', (base + timedelta(days=1, minutes=-30)).astimezone(timezone(timedelta(hours=-4))).isoformat())
        assert snap['age_days'] == 550  # 23:30Z day 550
        snap = engine.snapshot('ACME', (base + timedelta(days=1, minutes=30)).astimezone(timezone(timedelta(hours=-4))).isoformat())
        assert snap['age_days'] == 551 and snap['status'] == 'STALE'


def test_stale_company_and_status_values_and_no_permit_when_stale():
    from fundamentals.models import CombinedSignalEngine, FundamentalScoreEngine, ModelArtifact
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest(H.fy_facts('fy', '2020-01-01', '2020-12-31', '2021-02-01T00:00:00Z', 100 * H.M))
    engine = FundamentalFeatureEngine(s)
    statuses = {engine.snapshot('ACME', t)['status'] for t in ('2020-06-01T00:00:00Z', '2021-02-01T00:00:00Z', '2022-06-01T00:00:00Z', '2022-07-15T00:00:00Z', '2030-01-01T00:00:00Z')}
    assert statuses == {'MISSING', 'AVAILABLE', 'STALE'}
    assert engine.snapshot('NOSUCH', '2030-01-01T00:00:00Z')['status'] == 'MISSING'
    art = ModelArtifact(('revenue_yoy', 'technical_signal'), (0., 0.), (1., 1.), (0., 1.), 0.1, '2000-01-01T00:00:00Z', 1., 100, 'x', 5)
    combo = CombinedSignalEngine(engine, FundamentalScoreEngine(art), policy='filter', research_shadow=True)
    # revenue_yoy is missing (single year) so even a fresh company cannot be scored; add a second year to get a real score
    s.ingest(H.fy_facts('fy19', '2019-01-01', '2019-12-31', '2020-02-01T00:00:00Z', 80 * H.M))
    fresh = combo.decide('ACME', '2021-03-01T00:00:00Z', {'signal': True})
    stale = combo.decide('ACME', '2030-01-01T00:00:00Z', {'signal': True})
    assert fresh['fundamental']['data_status'] == 'AVAILABLE' and fresh['permit_signal'] is True
    assert stale['fundamental']['data_status'] == 'STALE' and stale['permit_signal'] is False
    assert 'Fundamentals missing/stale or wrong instrument type' in stale['explanation']


def test_custom_stale_threshold_is_respected():
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest(H.fy_facts('fy', '2023-01-01', '2023-12-31', '2024-02-01T00:00:00Z', 100 * H.M))
    at = '2024-06-01T00:00:00Z'  # 153 days after period end
    assert FundamentalFeatureEngine(s, 153).snapshot('ACME', at)['status'] == 'AVAILABLE'
    assert FundamentalFeatureEngine(s, 152).snapshot('ACME', at)['status'] == 'STALE'


def test_nan_stale_threshold_must_be_rejected():
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest(H.fy_facts('fy', '2020-01-01', '2020-12-31', '2021-02-01T00:00:00Z', 100 * H.M))
    try:
        engine = FundamentalFeatureEngine(s, float('nan'))
    except ValueError:
        return
    # constructor accepted NaN: the 10-year-old statement must still be flagged STALE
    assert engine.snapshot('ACME', '2030-01-01T00:00:00Z')['status'] == 'STALE'


# ---------------------------------------------------------------------------
# HTTP-level input handling of the as_of parameter
# ---------------------------------------------------------------------------
def test_http_as_of_unencoded_plus_offset_fails_closed_not_shifted(acme_db):
    with H.http_server(acme_db) as get:
        code, body, _, _ = get('/fundamentals/ACME?as_of=2025-05-10T08:00:00+01:00')  # '+' decodes to a space
        assert code == 400 and 'timezone' in body['error']
        code, body, _, _ = get('/fundamentals/ACME?as_of=2025-05-10T08:00:00%2B01:00')
        assert code == 200 and body['period_end'] == '2025-03-31'
        for bad in ('garbage', '2025-05-10', '2025-05-10T07:00:00', '2025-13-45T00:00:00Z', '9999-12-31T00:00:00Z', "'; DROP TABLE observations;--"):
            code, body, _, raw = get('/fundamentals/ACME?as_of=' + quote(bad, safe=''))
            assert code in (200, 400), (bad, code)
            if code == 400:
                assert 'Traceback' not in json.dumps(body)


def test_http_blank_as_of_must_not_silently_mean_now(acme_db):
    with H.http_server(acme_db) as get:
        code, body, _, _ = get('/fundamentals/ACME?as_of=')
        assert code == 400, f'blank as_of answered {code} at {body.get("as_of")}'


# ---------------------------------------------------------------------------
# DST transition days through snapshot AND real HTTP (not only store.as_of)
# ---------------------------------------------------------------------------
DST_PUBS = [
    # (zone, publication instant as local-offset text, queries as local-offset text)
    ('us-spring', '2025-03-09T03:00:00-04:00', ['2025-03-09T01:59:59-05:00', '2025-03-09T03:00:00-04:00', '2025-03-09T02:59:59-04:00', '2025-03-09T07:00:00Z', '2025-03-09T06:59:59.999999999Z']),
    ('us-fall-first', '2025-11-02T01:30:00-04:00', ['2025-11-02T01:29:59-04:00', '2025-11-02T01:30:00-04:00', '2025-11-02T01:30:00-05:00', '2025-11-02T00:59:59-05:00', '2025-11-02T05:29:59.999999Z', '2025-11-02T05:30:00Z']),
    ('us-fall-second', '2025-11-02T01:30:00-05:00', ['2025-11-02T01:59:59-04:00', '2025-11-02T01:29:59-05:00', '2025-11-02T01:30:00-05:00', '2025-11-02T06:29:59Z', '2025-11-02T06:30:00Z']),
    ('uk-spring', '2025-03-30T02:00:00+01:00', ['2025-03-30T00:59:59+00:00', '2025-03-30T01:00:00+00:00', '2025-03-30T01:59:59+01:00', '2025-03-30T02:00:00+01:00', '2025-03-30T01:00:00Z']),
    ('uk-fall-first', '2025-10-26T01:30:00+01:00', ['2025-10-26T01:29:59+01:00', '2025-10-26T01:30:00+01:00', '2025-10-26T00:30:00Z', '2025-10-26T00:29:59.999999Z', '2025-10-26T01:30:00+00:00']),
    ('uk-fall-second', '2025-10-26T01:30:00+00:00', ['2025-10-26T01:29:59+00:00', '2025-10-26T01:30:00+00:00', '2025-10-26T01:59:59+01:00', '2025-10-26T02:30:00+01:00']),
]


@pytest.mark.parametrize('name,pub,queries', DST_PUBS, ids=[c[0] for c in DST_PUBS])
def test_dst_days_snapshot_and_http_flip_exactly_at_the_published_instant(tmp_path, name, pub, queries):
    pub_ns = H.ns_instant(pub)
    path = tmp_path / f'{name}.sqlite'
    s = PointInTimeFundamentalStore(path)
    # fiscal years ending in February so that the results can legally be published on the March/October/November DST days
    s.ingest(H.fy_facts('fy23', '2022-03-01', '2023-02-28', '2023-05-13T07:00:00Z', 700 * H.M))
    s.ingest(H.fy_facts('fy24', '2023-03-01', '2024-02-29', '2024-05-12T07:00:00Z', 800 * H.M))
    s.ingest(H.fy_facts('fy25', '2024-03-01', '2025-02-28', pub, 1000 * H.M))
    s.close()
    ro = PointInTimeFundamentalStore(path, readonly=True)
    eng = FundamentalFeatureEngine(ro)
    with H.http_server(path) as get:
        for q in queries:
            visible = H.ns_instant(q) >= pub_ns
            # FY2025 (period end 2025-03-31) is only a legal observation once the publication instant has passed
            snap = eng.snapshot('ACME', q)
            assert snap['period_end'] == ('2025-02-28' if visible else '2024-02-29'), f'{name} snapshot {q}'
            assert snap['features']['revenue_yoy'] == pytest.approx(0.25 if visible else 800 / 700 - 1, rel=1e-12)
            code, body, _, _ = get('/fundamentals/ACME?as_of=' + quote(q, safe=''))
            assert code == 200 and body['period_end'] == snap['period_end'], f'{name} http {q}'
    ro.close()


def test_extreme_years_sort_correctly_in_sql_string_comparison():
    years = ['1677-09-22T00:00:00Z', '1900-01-01T00:00:00Z', '1969-12-31T23:59:59.999999Z', '1970-01-01T00:00:00Z', '2025-05-10T07:00:00Z', '2262-04-11T23:47:16Z', '2262-04-12T00:00:00Z', '9999-12-31T23:59:59Z']
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest([Fact('Y', f'zz{i}', 1.0, '1600-01-01', t, t, 'S', 'r', 'USD', 'USD', '', H.OBS, 'financial', {}) for i, t in enumerate(years)])
    for q in years + ['1677-09-21T23:59:59Z', '2262-04-11T23:47:15.999999999Z', '9999-12-31T23:59:58Z']:
        got = {r['metric'] for r in s.as_of('Y', q, kind='financial')}
        want = {f'zz{i}' for i, t in enumerate(years) if H.ns_instant(t) <= H.ns_instant(q)}
        assert got == want, q


def test_leap_second_and_calendar_impossible_instants_are_rejected_not_rounded():
    s = PointInTimeFundamentalStore(':memory:')
    for bad in ('2016-12-31T23:59:60Z', '2025-02-29T00:00:00Z', '2025-04-31T00:00:00Z', '2025-13-01T00:00:00Z', '2025-05-10T24:00:01Z'):
        with pytest.raises(ValueError):
            s.as_of('A', bad)
