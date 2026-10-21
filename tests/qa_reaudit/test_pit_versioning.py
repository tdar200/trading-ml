"""Restatements, withdrawals, late filers, duplicates, corrected filings, append-only enforcement,
same-timestamp conflicts, mixed-vintage snapshots, event versioning and 'latest by period_end' audit (QA track A)."""
from __future__ import annotations
import json
import re
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from tests.qa_reaudit import pit_helpers as H
from tests.qa_reaudit.pit_helpers import M, fact

from fundamentals.api import FundamentalAPI
from fundamentals.features import FundamentalFeatureEngine
from fundamentals.models import CombinedSignalEngine, EventCatalystEngine, FundamentalScoreEngine
from fundamentals.store import Fact, PointInTimeFundamentalStore

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# restatement (EPS 1.20 -> 1.05), exact instants, audit history preserved
# ---------------------------------------------------------------------------
def test_restatement_eps_exact_instants_and_history_preserved():
    orig = fact('eps', 1.20, '2025-03-31', '2025-01-01', '2025-05-10T07:00:00Z', rev='original', basis='b1')
    amended = fact('eps', 1.05, '2025-03-31', '2025-01-01', '2025-08-01T12:00:00Z', rev='amended', basis='b1')
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest([amended, orig])  # out of order on purpose
    assert [r['value'] for r in s.as_of('ACME', '2025-05-10T06:59:59.999999Z')] == []
    assert [r['value'] for r in s.as_of('ACME', '2025-05-10T07:00:00Z')] == [1.20]
    assert [r['value'] for r in s.as_of('ACME', '2025-08-01T11:59:59.999999Z')] == [1.20]
    assert [r['value'] for r in s.as_of('ACME', '2025-08-01T12:00:00Z')] == [1.05]
    assert [r['value'] for r in s.as_of('ACME', '2031-01-01T00:00:00Z')] == [1.05]
    hist = s.history('ACME', 'eps')
    assert [(r['value'], r['revision']) for r in hist] == [(1.20, 'original'), (1.05, 'amended')]  # nothing overwritten
    assert [r['revision'] for r in s.history('ACME', 'eps', '2025-06-01T00:00:00Z')] == ['original']


def test_restated_annual_eps_flows_into_snapshot_only_after_publication():
    facts = H.fy_facts('fy24', '2024-01-01', '2024-12-31', '2025-02-20T12:00:00Z', 100 * M,
                       extra=[fact('eps', 1.20, '2024-12-31', '2024-01-01', '2025-02-20T12:00:00Z', rev='o', basis='b1')])
    facts += H.fy_facts('fy23', '2023-01-01', '2023-12-31', '2024-02-20T12:00:00Z', 80 * M,
                        extra=[fact('eps', 1.00, '2023-12-31', '2023-01-01', '2024-02-20T12:00:00Z', rev='o', basis='b1')])
    facts.append(fact('eps', 1.05, '2024-12-31', '2024-01-01', '2025-04-01T09:30:00Z', rev='amend', basis='b1'))
    facts.append(fact('revenue', 98 * M, '2024-12-31', '2024-01-01', '2025-04-01T09:30:00Z', rev='amend'))
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest(facts)
    e = FundamentalFeatureEngine(s)
    before = e.snapshot('ACME', '2025-04-01T09:29:59.999999Z')
    after = e.snapshot('ACME', '2025-04-01T09:30:00Z')
    assert before['raw']['eps'] == 1.20 and before['raw']['revenue'] == 100 * M
    assert after['raw']['eps'] == 1.05 and after['raw']['revenue'] == 98 * M
    assert before['features']['eps_yoy'] == pytest.approx(1.20 / 1.00 - 1, rel=1e-12)
    assert after['features']['eps_yoy'] == pytest.approx(1.05 / 1.00 - 1, rel=1e-12)
    assert before['features']['revenue_yoy'] == pytest.approx(100 / 80 - 1, rel=1e-12)
    assert after['features']['revenue_yoy'] == pytest.approx(98 / 80 - 1, rel=1e-12)
    # restated version ids differ from the originals and both remain in the audit trail
    assert set(before['lineage']) != set(after['lineage'])
    assert len(s.history('ACME', 'eps')) == 3


# ---------------------------------------------------------------------------
# withdrawals (null)
# ---------------------------------------------------------------------------
def test_withdrawal_replaces_value_and_never_resurrects_old_number():
    s = PointInTimeFundamentalStore(':memory:')
    facts = H.fy_facts('fy24', '2024-01-01', '2024-12-31', '2025-02-20T12:00:00Z', 100 * M)
    facts += H.fy_facts('fy23', '2023-01-01', '2023-12-31', '2024-02-20T12:00:00Z', 80 * M)
    facts.append(fact('ocf', None, '2024-12-31', '2024-01-01', '2025-03-15T00:00:00Z', rev='withdrawn'))
    s.ingest(facts)
    e = FundamentalFeatureEngine(s)
    a = e.snapshot('ACME', '2025-03-14T23:59:59Z')
    b = e.snapshot('ACME', '2025-03-15T00:00:00Z')
    assert a['features']['fcf'] == pytest.approx(10 * M) and a['raw']['ocf'] == 15 * M
    assert b['raw']['ocf'] is None and b['features']['fcf'] is None and b['features']['fcf_margin'] is None
    assert b['features']['fcf_yoy'] is None
    # other metrics of the same filing are untouched
    assert b['features']['gross_margin'] == pytest.approx(.4)
    assert [r['value'] for r in s.history('ACME', 'ocf')] == [12 * M, 15 * M, None]


def test_withdrawn_revenue_of_only_year_makes_snapshot_missing_not_old_value():
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest(H.fy_facts('fy24', '2024-01-01', '2024-12-31', '2025-02-20T12:00:00Z', 100 * M))
    s.ingest([fact('revenue', None, '2024-12-31', '2024-01-01', '2025-03-01T00:00:00Z', rev='withdrawn')])
    snap = FundamentalFeatureEngine(s).snapshot('ACME', '2025-03-02T00:00:00Z')
    assert snap['status'] == 'MISSING' and snap['features'] == {}


def test_withdrawn_quarter_makes_ttm_incomplete_never_uses_withdrawn_number():
    s = PointInTimeFundamentalStore(':memory:')
    qs = [('2024-01-01', '2024-03-31', '2024-05-10'), ('2024-04-01', '2024-06-30', '2024-08-10'),
          ('2024-07-01', '2024-09-30', '2024-11-10'), ('2024-10-01', '2024-12-31', '2025-02-10')]
    facts = []
    for i, (st, en, pub) in enumerate(qs):
        for m in ('revenue', 'net_income', 'ocf', 'capex'):
            facts.append(fact(m, (i + 1) * 10 * M, en, st, pub + 'T07:00:00Z'))
        facts.append(fact('eps', 0.1 * (i + 1), en, st, pub + 'T07:00:00Z', basis='b1'))
    s.ingest(facts)
    e = FundamentalFeatureEngine(s)
    full = e.snapshot('ACME', '2025-02-11T00:00:00Z')['ttm']
    assert full['status'] == 'AVAILABLE' and full['values']['revenue'] == 100 * M and full['values']['eps'] == pytest.approx(1.0)
    s.ingest([fact('revenue', None, '2024-06-30', '2024-04-01', '2025-03-01T00:00:00Z', rev='withdrawn')])
    t = e.snapshot('ACME', '2025-03-02T00:00:00Z')['ttm']
    assert t['status'] == 'INCOMPLETE' and t['values']['revenue'] is None and t['reasons']['revenue'] == 'MISSING_QUARTERS'


# ---------------------------------------------------------------------------
# late filers: previous quarter / year continues to be used until publication
# ---------------------------------------------------------------------------
def _late_company(p25):
    q = [(100, '2024-01-01', '2024-03-31', '2024-05-12T07:00:00Z'), (110, '2024-04-01', '2024-06-30', '2024-08-10T07:00:00Z'),
         (120, '2024-07-01', '2024-09-30', '2024-11-10T07:00:00Z'), (130, '2024-10-01', '2024-12-31', '2025-02-10T07:00:00Z'),
         (140, '2025-01-01', '2025-03-31', p25)]
    facts = H.acme_nov_fy(p25)
    for v, st, en, pub in q:
        for m in ('revenue', 'net_income', 'ocf', 'capex'):
            facts.append(fact(m, v * M * (0.1 if m == 'net_income' else .15 if m == 'ocf' else .05 if m == 'capex' else 1), en, st, pub, rev='q' + en))
        facts.append(fact('eps', v / 100, en, st, pub, rev='q' + en, basis='b1'))
    return facts


@pytest.mark.parametrize('t,annual_end,ttm_end,ttm_rev', [
    ('2025-03-31T23:59:59Z', '2024-03-31', '2024-12-31', 460 * M),   # quarter has ended but nothing published
    ('2025-05-31T00:00:00Z', '2024-03-31', '2024-12-31', 460 * M),   # 2 months after quarter end, still unpublished
    ('2025-06-30T19:59:59.999999Z', '2024-03-31', '2024-12-31', 460 * M),
    # filed Jun 30: annual and TTM roll forward only now. Lead re-audit (QA-REAL-02): at a fiscal year end TTM IS the
    # reported FY total. This fixture's FY row says 1000M while its quarters sum to 500M (the generators are
    # independent), so TTM now follows the reported total instead of summing the quarters.
    ('2025-06-30T20:00:00Z', '2025-03-31', '2025-03-31', 1000 * M),
])
def test_late_filer_previous_period_continues_to_be_used(t, annual_end, ttm_end, ttm_rev):
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest(_late_company('2025-06-30T20:00:00Z'))
    snap = FundamentalFeatureEngine(s).snapshot('ACME', t)
    assert snap['period_end'] == annual_end
    assert snap['ttm']['period_end'] == ttm_end and snap['ttm']['values']['revenue'] == ttm_rev
    assert snap['ttm']['status'] == 'AVAILABLE'
    if annual_end == '2025-03-31':
        assert snap['features']['revenue_yoy'] == pytest.approx(.25)
    else:
        assert snap['features']['revenue_yoy'] == pytest.approx(800 / 700 - 1)


# ---------------------------------------------------------------------------
# duplicates / idempotency
# ---------------------------------------------------------------------------
def test_duplicate_ingestion_is_idempotent_even_with_different_observed_at():
    a1 = fact('revenue', 5.0, '2024-12-31', '2024-01-01', '2025-02-20T12:00:00Z', observed='2025-02-21T00:00:00Z')
    a2 = replace(a1, observed_at='2026-01-01T00:00:00Z')
    s = PointInTimeFundamentalStore(':memory:')
    for batch in ([a1, a1], [a1], [a2], [a2, a1]):
        s.ingest(batch)
    rows = s.history('ACME')
    assert len(rows) == 1
    assert rows[0]['observed_at'] == '2025-02-21T00:00:00+00:00'  # first observation retained, later re-download ignored
    out = s.ingest_validated([a1, a2])
    assert out['accepted'] == 2 and len(s.history('ACME')) == 1
    # observed_only semantics: invisible before the first local observation, visible afterwards
    assert s.as_of('ACME', '2025-02-20T13:00:00Z', observed_only=True) == []
    assert len(s.as_of('ACME', '2025-02-20T13:00:00Z')) == 1
    assert len(s.as_of('ACME', '2025-02-21T00:00:00Z', observed_only=True)) == 1


def test_same_filing_republished_with_different_availability_keeps_both_versions_and_time_correct():
    a = fact('revenue', 5.0, '2024-12-31', '2024-01-01', '2025-02-20T12:00:00Z', rev='acc1')
    later = replace(a, available_at='2025-02-21T00:00:00Z')  # provider re-derived a later (date-only) availability
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest([a, later])
    assert len(s.history('ACME')) == 2
    assert [r['value'] for r in s.as_of('ACME', '2025-02-20T12:00:00Z')] == [5.0]
    assert [r['available_at'] for r in s.as_of('ACME', '2025-02-20T12:00:00Z')] == ['2025-02-20T12:00:00+00:00']
    assert [r['available_at'] for r in s.as_of('ACME', '2025-02-21T00:00:00Z')] == ['2025-02-21T00:00:00+00:00']
    assert len(s.as_of('ACME', '2025-02-21T00:00:00Z')) == 1  # never double counted


def test_no_duplicate_features_events_or_scores_when_everything_is_run_twice():
    s = PointInTimeFundamentalStore(':memory:')
    facts = H.fy_facts('fy24', '2024-01-01', '2024-12-31', '2025-02-20T12:00:00Z', 100 * M) + H.fy_facts('fy23', '2023-01-01', '2023-12-31', '2024-02-20T12:00:00Z', 80 * M)
    s.ingest(facts)
    s.ingest(facts)
    for _ in range(2):
        s.add_event(symbol='ACME', event_type='earnings', event_at='2025-02-20T12:00:00Z', reported_at='2025-02-20T12:00:00Z',
                    available_at='2025-02-20T12:05:00Z', source='x', revision='1', payload={'event_key': 'q4', 'reported_eps': 1.0, 'consensus_eps': .9, 'consensus_available_at': '2025-01-01T00:00:00Z'})
    e = FundamentalFeatureEngine(s)
    ids = {e.snapshot('ACME', '2025-03-01T00:00:00Z', persist=True)['lineage'].__repr__() for _ in range(3)}
    assert len(ids) == 1
    assert s.db.execute('select count(*) from feature_snapshots').fetchone()[0] == 1
    assert s.db.execute('select count(*) from corporate_events').fetchone()[0] == 1
    assert len(EventCatalystEngine(s).state('ACME', '2025-03-01T00:00:00Z')['events']) == 1
    combo = CombinedSignalEngine(e, policy='filter')
    d1, d2 = combo.decide('ACME', '2025-03-01T00:00:00Z', {'signal': True}), combo.decide('ACME', '2025-03-01T00:00:00Z', {'signal': True})
    assert d1 == d2
    s.save('decisions', d1)
    s.save('decisions', d2)
    assert s.db.execute('select count(*) from decisions').fetchone()[0] == 1
    # a restatement published afterwards is a NEW persisted snapshot, the old one is retained
    s.ingest([fact('revenue', 99 * M, '2024-12-31', '2024-01-01', '2025-04-01T00:00:00Z', rev='amend')])
    e.snapshot('ACME', '2025-05-01T00:00:00Z', persist=True)
    e.snapshot('ACME', '2025-03-01T00:00:00Z', persist=True)  # identical to the first one
    assert s.db.execute('select count(*) from feature_snapshots').fetchone()[0] == 2
    assert s.db.execute("select count(distinct as_of) from FundamentalFeatures").fetchone()[0] == 2
    first = json.loads(s.db.execute("select payload from feature_snapshots where as_of='2025-03-01T00:00:00+00:00'").fetchone()[0])
    assert first['raw']['revenue'] == 100 * M  # historical persisted snapshot still shows the number known at that time


# ---------------------------------------------------------------------------
# append-only enforcement
# ---------------------------------------------------------------------------
def _populate_all_tables(path):
    from fundamentals.models import ModelArtifact
    s = PointInTimeFundamentalStore(path)
    s.ingest(H.fy_facts('fy24', '2024-01-01', '2024-12-31', '2025-02-20T12:00:00Z', 100 * M))
    s.ingest_validated([fact('revenue', 1., '2024-12-31', '2024-01-01', '2025-02-20T12:00:00Z', currency='XXX', units='XXX')])
    s.archive('src', 'body')
    s.add_event(symbol='ACME', event_type='x', event_at='2025-01-01T00:00:00Z', reported_at='2025-01-01T00:00:00Z', available_at='2025-01-01T00:00:00Z', source='s', revision='1', payload={'a': 1})
    FundamentalFeatureEngine(s).snapshot('ACME', '2025-03-01T00:00:00Z', persist=True)
    art = ModelArtifact(('revenue_yoy',), (0.,), (1.,), (1.,), 0., '2000-01-01T00:00:00Z', 1., 10, 'x', 5)
    s.save('model_artifacts', art.__dict__)
    s.save('decisions', {'symbol': 'ACME', 'timestamp': '2025-03-01T00:00:00Z', 'x': 1})
    return s


TABLE_COLUMNS = {'raw_documents': 'body', 'ingestion_rejections': 'reason', 'observations': 'value', 'corporate_events': 'payload',
                 'feature_snapshots': 'payload', 'model_artifacts': 'payload', 'decisions': 'payload'}


@pytest.mark.parametrize('table', sorted(TABLE_COLUMNS))
def test_append_only_triggers_block_update_and_delete_on_every_table_from_a_second_connection(tmp_path, table):
    path = tmp_path / 'ao.sqlite'
    s = _populate_all_tables(path)
    s.close()
    raw = sqlite3.connect(str(path))  # a foreign connection that is not the store object
    assert raw.execute(f'select count(*) from {table}').fetchone()[0] >= 1
    col = TABLE_COLUMNS[table]
    with pytest.raises(sqlite3.DatabaseError, match='append-only'):
        raw.execute(f'UPDATE {table} SET {col}={col}')
    with pytest.raises(sqlite3.DatabaseError, match='append-only'):
        raw.execute(f'DELETE FROM {table}')
    with pytest.raises(sqlite3.DatabaseError, match='append-only'):
        raw.execute(f'DELETE FROM {table} WHERE 1=0 OR rowid>0')
    raw.close()


def test_append_only_blocks_updates_through_the_views_and_store_api_never_rewrites():
    s = _populate_all_tables(':memory:')
    for view in ('FinancialStatements', 'FundamentalFeatures', 'FundamentalScores', 'EarningsReports'):
        with pytest.raises(sqlite3.DatabaseError):
            s.db.execute(f'DELETE FROM {view}')
    before = s.db.execute('select count(*) from observations').fetchone()[0]
    s.ingest(H.fy_facts('fy24', '2024-01-01', '2024-12-31', '2025-02-20T12:00:00Z', 100 * M))  # exact duplicate batch
    assert s.db.execute('select count(*) from observations').fetchone()[0] == before


@pytest.mark.xfail(strict=True, reason='QA-PIT-01 LOW: INSERT OR REPLACE overwrites an observation in place; the BEFORE DELETE trigger does not fire for REPLACE (recursive_triggers off), so the append-only guarantee is bypassable with plain SQL')
def test_append_only_must_also_block_insert_or_replace(tmp_path):
    s = PointInTimeFundamentalStore(tmp_path / 'r.sqlite')
    s.ingest([fact('revenue', 100., '2024-12-31', '2024-01-01', '2025-02-20T12:00:00Z')])
    row = list(tuple(s.db.execute('select * from observations').fetchone()))
    row[3] = 999.0  # value
    with pytest.raises(sqlite3.DatabaseError):
        s.db.execute('INSERT OR REPLACE INTO observations VALUES (' + ','.join('?' * 15) + ')', row)
        s.db.commit()


# ---------------------------------------------------------------------------
# same-timestamp conflicting facts
# ---------------------------------------------------------------------------
def _conflict_store():
    s = PointInTimeFundamentalStore(':memory:')
    facts = H.fy_facts('fy24', '2024-01-01', '2024-12-31', '2025-02-20T12:00:00Z', 100 * M)
    a = fact('revenue', 120 * M, '2024-12-31', '2024-01-01', '2025-03-01T00:00:00Z', rev='A')
    b = fact('revenue', 130 * M, '2024-12-31', '2024-01-01', '2025-03-01T00:00:00Z', rev='B')
    return s, facts, a, b


@pytest.mark.parametrize('order', ['ab', 'ba', 'split'])
def test_same_instant_conflicting_values_fail_closed_everywhere_not_arbitrarily_resolved(order, tmp_path):
    s, facts, a, b = _conflict_store()
    s.ingest(facts)
    if order == 'ab':
        s.ingest([a, b])
    elif order == 'ba':
        s.ingest([b, a])
    else:
        s.ingest([b])
        s.ingest([a])
    # before the conflicting instant nothing is wrong
    assert FundamentalFeatureEngine(s).snapshot('ACME', '2025-02-28T23:59:59.999999Z')['raw']['revenue'] == 100 * M
    for T in ('2025-03-01T00:00:00Z', '2026-01-01T00:00:00Z'):
        with pytest.raises(ValueError, match='ambiguous conflicting'):
            s.as_of('ACME', T)
        with pytest.raises(ValueError, match='ambiguous conflicting'):
            FundamentalFeatureEngine(s).snapshot('ACME', T)
        with pytest.raises(ValueError, match='ambiguous conflicting'):
            CombinedSignalEngine(FundamentalFeatureEngine(s), policy='filter').decide('ACME', T, {'signal': True})
        with pytest.raises(ValueError, match='ambiguous conflicting'):
            FundamentalAPI(s).get('/fundamentals/ACME', {'as_of': T})
    # history (audit) still shows both competing versions
    assert sorted(r['revision'] for r in s.history('ACME', 'revenue', '2025-03-01T00:00:00Z') if r['revision'] in 'AB' and r['revision']) == ['A', 'B']


def test_conflicting_same_instant_facts_return_http_400_not_a_value(tmp_path):
    path = tmp_path / 'c.sqlite'
    s, facts, a, b = _conflict_store()
    s.close()
    s = PointInTimeFundamentalStore(path)
    s.ingest(facts + [a, b])
    s.close()
    with H.http_server(path) as get:
        code, body, _, raw = get('/fundamentals/ACME?as_of=2025-03-02T00:00:00Z')
        assert code == 400 and 'ambiguous' in body['error']
        code, body, _, _ = get('/fundamentals/ACME?as_of=2025-02-28T00:00:00Z')
        assert code == 200 and body['raw']['revenue'] == 100 * M
        code, body, _, _ = get('/watchlist/fundamental?as_of=2025-03-02T00:00:00Z')
        # Lead re-audit (QA-UI-03): one ambiguous company no longer hides every healthy one; it is listed unscored.
        assert code == 200
        row = [r for r in body['candidates'] if r['symbol'] == 'ACME'][0]
        assert row['data_status'] == 'AMBIGUOUS_SOURCE' and row['score'] is None and 'ambiguous' in row['reason']


def test_conflict_rules_equal_values_and_different_reported_at_and_share_basis():
    s = PointInTimeFundamentalStore(':memory:')
    base = fact('revenue', 100., '2024-12-31', '2024-01-01', '2025-03-01T00:00:00Z', rev='A')
    s.ingest([base, replace(base, revision='B')])  # same value, other revision: harmless duplicate
    assert [r['value'] for r in s.as_of('ACME', '2025-03-01T00:00:00Z')] == [100.]
    s2 = PointInTimeFundamentalStore(':memory:')
    first = fact('revenue', 100., '2024-12-31', '2024-01-01', '2025-03-01T00:00:00Z', avail='2025-03-01T01:00:00Z', rev='A')
    second = replace(first, value=90., reported_at='2025-03-01T00:30:00Z', revision='B')  # same available_at, later reported_at wins
    s2.ingest([first, second])
    assert [r['value'] for r in s2.as_of('ACME', '2025-03-01T01:00:00Z')] == [90.]
    s3 = PointInTimeFundamentalStore(':memory:')
    s3.ingest([fact('eps', 1.0, '2024-12-31', '2024-01-01', '2025-03-01T00:00:00Z', basis='b1', rev='A'),
               fact('eps', 1.0, '2024-12-31', '2024-01-01', '2025-03-01T00:00:00Z', basis='b2', rev='B')])
    with pytest.raises(ValueError, match='ambiguous'):
        s3.as_of('ACME', '2025-03-01T00:00:00Z')
    s4 = PointInTimeFundamentalStore(':memory:')
    s4.ingest([fact('revenue', 100., '2024-12-31', '2024-01-01', '2025-03-01T00:00:00Z', rev='A'),
               fact('revenue', None, '2024-12-31', '2024-01-01', '2025-03-01T00:00:00Z', rev='B')])
    with pytest.raises(ValueError, match='ambiguous'):
        s4.as_of('ACME', '2025-03-01T00:00:00Z')


def test_two_sources_same_period_never_silently_merged_in_snapshot():
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest(H.fy_facts('fy24', '2024-01-01', '2024-12-31', '2025-02-20T12:00:00Z', 100 * M))
    s.ingest([fact('revenue', 110 * M, '2024-12-31', '2024-01-01', '2025-02-21T00:00:00Z', source='VENDOR')])
    with pytest.raises(ValueError, match='ambiguous'):
        FundamentalFeatureEngine(s).snapshot('ACME', '2025-03-01T00:00:00Z')
    # before the vendor feed existed, the SRC-only view is fine
    assert FundamentalFeatureEngine(s).snapshot('ACME', '2025-02-20T12:00:00Z')['raw']['revenue'] == 100 * M


# ---------------------------------------------------------------------------
# restatement of one metric but not the others: mixed-vintage snapshot
# ---------------------------------------------------------------------------
def test_mixed_vintage_snapshot_uses_latest_public_version_per_metric_and_records_both_ids():
    s = PointInTimeFundamentalStore(':memory:')
    orig = H.fy_facts('fy24', '2024-01-01', '2024-12-31', '2025-02-20T12:00:00Z', 100 * M)
    s.ingest(orig)
    s.ingest([fact('net_income', 5 * M, '2024-12-31', '2024-01-01', '2025-06-01T08:00:00Z', rev='10-K/A')])  # only NI restated (10 -> 5)
    e = FundamentalFeatureEngine(s)
    before = e.snapshot('ACME', '2025-05-31T00:00:00Z')
    after = e.snapshot('ACME', '2025-06-01T08:00:00Z')
    assert before['features']['net_margin'] == pytest.approx(.10)
    assert after['features']['net_margin'] == pytest.approx(.05)
    assert after['raw']['revenue'] == 100 * M and before['raw']['revenue'] == 100 * M  # untouched metric keeps its original filing
    revisions = {r['revision'] for r in s.db.execute('select revision from observations where id in (%s)' % ','.join('?' * len(after['lineage'])), after['lineage'])}
    assert revisions == {'fy24', '10-K/A'}  # lineage honestly exposes the two vintages
    assert after['available_at'] == '2025-06-01T08:00:00+00:00' and after['reported_at'] == '2025-06-01T08:00:00+00:00'
    assert before['available_at'] == '2025-02-20T12:00:00+00:00'


# ---------------------------------------------------------------------------
# events: versioning and dedupe
# ---------------------------------------------------------------------------
def _ev(s, rev, avail, payload, source='src', etype='earnings', event_at='2025-05-01T00:00:00Z'):
    return s.add_event(symbol='ACME', event_type=etype, event_at=event_at, reported_at=avail, available_at=avail, source=source, revision=rev, payload=payload)


def test_event_versions_follow_availability_and_latest_version_wins_per_key():
    s = PointInTimeFundamentalStore(':memory:')
    _ev(s, '1', '2025-05-01T10:00:00Z', {'event_key': 'q1', 'reported_eps': 1.0})
    _ev(s, '2', '2025-05-02T10:00:00Z', {'event_key': 'q1', 'reported_eps': 0.9})  # corrected
    _ev(s, '1', '2025-05-01T10:00:00Z', {'event_key': 'q1', 'reported_eps': 1.0})  # duplicate delivery
    assert s.events('ACME', '2025-05-01T09:59:59.999999Z') == []
    assert [e['payload']['reported_eps'] for e in s.events('ACME', '2025-05-01T10:00:00Z')] == [1.0]
    assert [e['payload']['reported_eps'] for e in s.events('ACME', '2025-05-02T09:59:59Z')] == [1.0]
    assert [e['payload']['reported_eps'] for e in s.events('ACME', '2025-05-02T10:00:00Z')] == [0.9]
    assert s.db.execute('select count(*) from corporate_events').fetchone()[0] == 2  # both versions retained for audit


def test_wildcard_symbol_events_visible_only_after_publication():
    s = PointInTimeFundamentalStore(':memory:')
    s.add_event(symbol='*', event_type='macro', event_at='2025-05-01T00:00:00Z', reported_at='2025-05-01T12:30:00Z', available_at='2025-05-01T12:35:00Z', source='x', revision='1', payload={'event_key': 'cpi'})
    assert s.events('ACME', '2025-05-01T12:34:59.999999Z') == []
    assert len(s.events('ACME', '2025-05-01T12:35:00Z')) == 1


def test_same_instant_conflicting_event_versions_must_not_be_resolved_by_hash_order():
    wrong = 0
    for k in range(12):
        s = PointInTimeFundamentalStore(':memory:')
        for rev in (str(2 * k + 1), str(2 * k + 2)):
            _ev(s, rev, '2025-05-01T10:00:00Z', {'event_key': 'q1', 'rev': rev})
        try:
            got = s.events('ACME', '2025-06-01T00:00:00Z')
        except ValueError:
            continue  # consistent with store.as_of: ambiguity is an error
        assert len(got) == 1
        wrong += got[0]['revision'] != str(2 * k + 2)
    assert wrong == 0, f'{wrong}/12 trials silently picked the lower revision'


# ---------------------------------------------------------------------------
# ingest_json_records re-export duplicates (QA-PIT-05)
# ---------------------------------------------------------------------------
def test_reimporting_identical_records_from_a_reformatted_file_must_not_change_lineage(tmp_path):
    from fundamentals.providers import ingest_json_records
    recs = [dict(symbol='A', metric='revenue', value=100., period_end='2020-12-31', reported_at='2021-02-01T00:00:00Z', available_at='2021-02-01T00:00:00Z',
                 source='s', revision='r', currency='USD', units='USD', period_start='2020-01-01')]
    (tmp_path / 'a.json').write_text(json.dumps(recs))
    (tmp_path / 'b.json').write_text(json.dumps(recs, indent=2))
    s = PointInTimeFundamentalStore(':memory:')
    e = FundamentalFeatureEngine(s)
    ingest_json_records(s, tmp_path / 'a.json')
    before = e.snapshot('A', '2021-03-01T00:00:00Z')['lineage']
    ingest_json_records(s, tmp_path / 'b.json')
    after = e.snapshot('A', '2021-03-01T00:00:00Z')['lineage']
    assert len(s.history('A')) == 1 and before == after


def test_reimporting_the_byte_identical_file_is_idempotent(tmp_path):
    from fundamentals.providers import ingest_json_records
    recs = [dict(symbol='A', metric='revenue', value=100., period_end='2020-12-31', reported_at='2021-02-01T00:00:00Z', available_at='2021-02-01T00:00:00Z',
                 source='s', revision='r', currency='USD', units='USD', period_start='2020-01-01')]
    (tmp_path / 'a.json').write_text(json.dumps(recs))
    s = PointInTimeFundamentalStore(':memory:')
    ingest_json_records(s, tmp_path / 'a.json')
    ingest_json_records(s, tmp_path / 'a.json')
    assert len(s.history('A')) == 1
    assert s.db.execute('select count(*) from raw_documents').fetchone()[0] == 1


# ---------------------------------------------------------------------------
# audit of every "latest" selection in the package: all operate on rows already filtered by available_at
# ---------------------------------------------------------------------------
REVIEWED_LATEST_SITES = {
    # file -> number of max/min/sorted(key=...period_end|available_at|reported_at|event_at...) sites reviewed on 2026-10-01
    'fundamentals/adapters.py': 3,   # historical_fx max(period_end) over store.as_of rows; apply_known_splits sorted(event_at) over store.events rows; own_history_percentile newest period (as_of rows)
    'fundamentals/features.py': 6,   # quarters p; comparable_accounting_rows; snapshot latest/min; etf_snapshot latest holdings; _ttm_from_reported_totals closest prior YTD (as_of rows)
    'fundamentals/store.py': 1,      # as_of() final presentation sort (after the available_at<=T filter)
    'fundamentals/models.py': 2,     # macro latest period_end over store.as_of rows + lag() max(available_at)
}


def test_every_latest_by_period_end_site_is_reviewed():
    pat = re.compile(r'(max|min|sorted)\(.*key=lambda.*(period_end|reported_at|available_at|event_at)')
    found = {}
    for p in sorted((ROOT / 'fundamentals').glob('*.py')):
        n = sum(1 for line in p.read_text().splitlines() if pat.search(line))
        if n:
            found[f'fundamentals/{p.name}'] = n
    assert found == REVIEWED_LATEST_SITES, ('a new max/min/sorted selection by period_end/available_at appeared; review it for look-ahead '
                                            f'(it must only see rows already filtered by store.as_of/events): {found}')


def test_withdrawn_latest_revenue_rolls_back_to_previous_fiscal_year_but_is_marked_stale():
    """Lead re-audit (QA-REAL-03): a null revenue for the newest fiscal year removes that year from the annual anchor, but the
    snapshot is no longer served as AVAILABLE/current: other annual flows are public for the later year, so it is STALE and says why."""
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest(H.fy_facts('fy23', '2023-01-01', '2023-12-31', '2024-02-20T12:00:00Z', 80 * M) + H.fy_facts('fy24', '2024-01-01', '2024-12-31', '2025-02-20T12:00:00Z', 100 * M))
    s.ingest([fact('revenue', None, '2024-12-31', '2024-01-01', '2025-03-01T00:00:00Z', rev='withdrawn')])
    e = FundamentalFeatureEngine(s)
    before, after = e.snapshot('ACME', '2025-02-28T00:00:00Z'), e.snapshot('ACME', '2025-03-02T00:00:00Z')
    assert before['period_end'] == '2024-12-31' and after['period_end'] == '2023-12-31' and after['status'] == 'STALE'
    assert after['period_anchor_warning']['later_annual_period_end'] == '2024-12-31'
    assert after['raw']['revenue'] == 80 * M and after['age_days'] == 427
