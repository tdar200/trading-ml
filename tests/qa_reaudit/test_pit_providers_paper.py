"""Phases 141-144: filing-time semantics of providers.sec_facts and what a daily-bar PaperEngine can see (QA track A)."""
from __future__ import annotations
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from tests.qa_reaudit import pit_helpers as H
from tests.qa_reaudit import pit_oracle as O
from tests.qa_reaudit.pit_helpers import M, fact

from fundamentals.features import FundamentalFeatureEngine
from fundamentals.models import CombinedSignalEngine, FundamentalScoreEngine, ModelArtifact
from fundamentals.providers import sec_facts
from fundamentals.store import PointInTimeFundamentalStore
from livesim.small_account import PaperEngine

UTC = timezone.utc
NY = ZoneInfo('America/New_York')


def ny_midnight_utc(y, m, d):
    return datetime(y, m, d, tzinfo=NY).astimezone(UTC)


def sec_payload(rows, cik=1234):
    """rows: (accn, form, filed, start, end, val)"""
    units = [{'accn': a, 'form': f, 'filed': filed, 'start': s, 'end': e, 'val': v, 'fy': 2024, 'fp': 'FY'} for a, f, filed, s, e, v in rows]
    return {'cik': cik, 'facts': {'us-gaap': {'Revenues': {'units': {'USD': units}}}}}


def submissions(pairs):
    return {'filings': {'recent': {'accessionNumber': [a for a, _ in pairs], 'acceptanceDateTime': [t for _, t in pairs]}}}


def avail_of(facts, accn):
    got = [f for f in facts if f.revision == accn]
    assert len(got) == 1, accn
    return got[0]


# ---------------------------------------------------------------------------
# available_at semantics, from crafted submissions JSON
# ---------------------------------------------------------------------------
def test_acceptance_timestamp_gives_acceptance_plus_five_minutes_utc_after_hours_and_premarket():
    rows = [('A-after', '10-K', '2025-02-14', '2024-01-01', '2024-12-31', 100.), ('A-pre', '10-Q', '2025-02-14', '2024-10-01', '2024-12-31', 25.)]
    subs = submissions([('A-after', '2025-02-14T21:30:12.000Z'), ('A-pre', '2025-02-14T11:00:00.000Z')])  # 16:30 ET and 06:00 ET
    facts = sec_facts(sec_payload(rows), subs, 'X')
    a, p = avail_of(facts, 'A-after'), avail_of(facts, 'A-pre')
    assert H.ns_instant(a.available_at) == H.ns_instant('2025-02-14T21:35:12Z') and H.ns_instant(a.reported_at) == H.ns_instant('2025-02-14T21:30:12Z')
    assert H.ns_instant(p.available_at) == H.ns_instant('2025-02-14T11:05:00Z')
    assert a.metadata['timing_precision'] == 'acceptance+5min'
    # nothing leaks before acceptance + 5 minutes
    s = PointInTimeFundamentalStore(':memory:')
    assert s.ingest_validated(facts)['accepted'] == 2
    assert s.as_of('X', '2025-02-14T21:35:11.999999Z', kind='financial') == [r for r in s.as_of('X', '2025-02-14T21:35:11.999999Z', kind='financial') if r['revision'] == 'A-pre']
    assert {r['revision'] for r in s.as_of('X', '2025-02-14T21:35:12Z', kind='financial')} == {'A-after', 'A-pre'}
    assert s.as_of('X', '2025-02-14T11:04:59.999999Z', kind='financial') == []


@pytest.mark.parametrize('filed', ['2025-02-14', '2025-03-08', '2025-03-09', '2025-03-10', '2025-11-01', '2025-11-02', '2025-11-03', '2025-12-31', '2024-02-29'])
def test_date_only_filing_becomes_available_at_start_of_the_following_new_york_local_day(filed):
    y, m, d = map(int, filed.split('-'))
    start_next = datetime(y, m, d, tzinfo=NY) + timedelta(days=1)  # wall-clock midnight of the next local day
    expected = datetime(start_next.year, start_next.month, start_next.day, tzinfo=NY).astimezone(UTC)
    facts = sec_facts(sec_payload([('B', '10-K', filed, '2023-01-01', '2023-12-31', 7.)]), {'filings': {'recent': {}}}, 'X')
    f = avail_of(facts, 'B')
    assert H.ns_instant(f.available_at) == int(expected.timestamp()) * 10**9, f'{filed}: {f.available_at} vs {expected}'
    assert f.metadata['timing_precision'] == 'following_local_day'
    # must be strictly after the end of the filing day in New York (never earlier than the filing day's close)
    assert expected >= datetime(y, m, d, 23, 59, 59, tzinfo=NY).astimezone(UTC)


def test_us_dst_transition_days_use_calendar_days_not_24_hour_blocks():
    """filed on the 25-hour fall-back Sunday: a +24h rule would publish at 23:00 local the same day."""
    facts = sec_facts(sec_payload([('F', '10-K', '2025-11-02', '2024-01-01', '2024-12-31', 1.), ('S', '10-K', '2025-03-09', '2024-01-01', '2024-12-31', 1.)]), {'filings': {'recent': {}}}, 'X')
    assert avail_of(facts, 'F').available_at == '2025-11-03T05:00:00+00:00'   # 00:00 EST Monday
    assert avail_of(facts, 'S').available_at == '2025-03-10T04:00:00+00:00'   # 00:00 EDT Monday after the 23-hour day


def test_forms_filter_and_amendments_and_missing_acceptance_fallback():
    rows = [('K', '10-K', '2025-02-14', '2024-01-01', '2024-12-31', 1.), ('KA', '10-K/A', '2025-03-14', '2024-01-01', '2024-12-31', 2.),
            ('EK', '8-K', '2025-02-14', '2024-01-01', '2024-12-31', 3.), ('Q', '10-Q/A', '2025-05-14', '2025-01-01', '2025-03-31', 4.),
            ('F20', '20-F', '2025-04-14', '2024-01-01', '2024-12-31', 5.), ('X6', '6-K', '2025-04-14', '2024-01-01', '2024-12-31', 6.)]
    facts = sec_facts(sec_payload(rows), submissions([('K', ''), ('KA', '2025-03-14T20:00:00.000Z')]), 'X')  # empty acceptance => date-only rule
    assert {f.revision for f in facts} == {'K', 'KA', 'Q', 'F20'}
    assert avail_of(facts, 'K').metadata['timing_precision'] == 'following_local_day'
    assert avail_of(facts, 'KA').metadata['timing_precision'] == 'acceptance+5min'
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest_validated(facts)
    # original 10-K number until the 10-K/A is public
    assert [r['value'] for r in s.as_of('X', '2025-03-14T20:04:59Z', kind='financial') if r['period_end'] == '2024-12-31' and r['revision'] in ('K', 'KA')] == [1.]
    assert [r['value'] for r in s.as_of('X', '2025-03-14T20:05:00Z', kind='financial') if r['period_end'] == '2024-12-31' and r['revision'] in ('K', 'KA')] == [2.]


def test_acceptance_before_filing_date_after_hours_submission_is_available_at_acceptance_not_filing_date():
    """EDGAR deems an after-5:30pm submission filed on the next business day but disseminates it at acceptance.
    The code follows acceptance (+5 min). Documented behaviour: available_at may precede the nominal filing date."""
    rows = [('AH', '10-K', '2025-02-18', '2024-01-01', '2024-12-31', 9.)]  # Tuesday filing date after the Presidents' Day holiday
    facts = sec_facts(sec_payload(rows), submissions([('AH', '2025-02-14T22:59:59.000Z')]), 'X')  # Friday 17:59:59 ET
    f = avail_of(facts, 'AH')
    assert f.available_at == '2025-02-14T23:04:59+00:00' and f.metadata['filing_date'] == '2025-02-18'


def test_naive_acceptance_is_assumed_new_york_and_dst_ambiguous_naive_values_fail_loudly():
    facts = sec_facts(sec_payload([('N', '10-K', '2025-02-14', '2024-01-01', '2024-12-31', 1.)]), submissions([('N', '2025-02-14T16:30:00')]), 'X')
    assert avail_of(facts, 'N').available_at == '2025-02-14T21:35:00+00:00'  # 16:30 EST + 5 min
    facts = sec_facts(sec_payload([('N', '10-K', '2025-02-14', '2024-01-01', '2024-12-31', 1.)]), submissions([('N', '2025-02-14T16:30:00-05:00')]), 'X')
    assert avail_of(facts, 'N').available_at == '2025-02-14T21:35:00+00:00'
    facts = sec_facts(sec_payload([('N', '10-K', '2025-02-14', '2024-01-01', '2024-12-31', 1.)]), submissions([('N', '2025-02-14T21:30:00+05:30')]), 'X')
    assert avail_of(facts, 'N').available_at == '2025-02-14T16:05:00+00:00'
    for naive in ('2025-03-09T02:30:00', '2025-11-02T01:30:00'):  # nonexistent / ambiguous local wall-clock times
        with pytest.raises(Exception) as err:
            sec_facts(sec_payload([('N', '10-K', '2025-03-09', '2024-01-01', '2024-12-31', 1.)]), submissions([('N', naive)]), 'X')
        assert 'pytz' in type(err.value).__module__  # LIMITATION: not a ValueError, but it fails loudly and nothing is ingested


def test_sec_redownload_is_idempotent_even_with_different_observed_at():
    rows = [('K', '10-K', '2025-02-14', '2024-01-01', '2024-12-31', 1.), ('Q', '10-Q', '2025-05-14', '2025-01-01', '2025-03-31', 2.)]
    s = PointInTimeFundamentalStore(':memory:')
    for obs in ('2025-06-01T00:00:00Z', '2026-01-01T00:00:00Z'):
        s.ingest_validated(sec_facts(sec_payload(rows), submissions([('K', '2025-02-14T21:30:00.000Z')]), 'X', observed_at=obs))
    assert len(s.history('X')) == 2
    assert {r['observed_at'] for r in s.history('X')} == {'2025-06-01T00:00:00+00:00'}


def test_sec_restatement_in_a_later_filing_is_a_new_version_visible_only_after_that_filing():
    rows = [('K24', '10-K', '2025-02-14', '2024-01-01', '2024-12-31', 100.), ('K25', '10-K', '2026-02-13', '2024-01-01', '2024-12-31', 97.),
            ('K25', '10-K', '2026-02-13', '2025-01-01', '2025-12-31', 120.)]
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest_validated(sec_facts(sec_payload(rows), submissions([('K24', '2025-02-14T21:30:00.000Z'), ('K25', '2026-02-13T21:30:00.000Z')]), 'X'))
    q = lambda t: {r['period_end']: r['value'] for r in s.as_of('X', t, kind='financial') if r['metric'] == 'revenue'}  # noqa: E731
    assert q('2025-06-01T00:00:00Z') == {'2024-12-31': 100.}
    assert q('2026-02-13T21:34:59Z') == {'2024-12-31': 100.}
    assert q('2026-02-13T21:35:00Z') == {'2024-12-31': 97., '2025-12-31': 120.}


# ---------------------------------------------------------------------------
# PaperEngine, daily bars stamped 04:00 UTC: decision at ts sees exactly available_at <= ts
# ---------------------------------------------------------------------------
def bars(first='2025-05-07', n=6):
    d0 = datetime.fromisoformat(first).replace(tzinfo=UTC)
    out = []
    for i in range(n):
        ts = (d0 + timedelta(days=i)).replace(hour=4)
        out.append(dict(ts=ts.isoformat(), open=100., high=101., low=99., close=100.))
    return out


def run_engine(store, bar_list, *, policy='filter', model=None, shadow=True):
    combo = CombinedSignalEngine(FundamentalFeatureEngine(store), FundamentalScoreEngine(model) if model else None, policy=policy, research_shadow=True)
    e = PaperEngine(symbol='ACME', shadow=shadow, fee_bps=0, spread_bps=0, slip_bps=0, fundamental_filter=combo)
    prev = None
    for b in bar_list:
        if prev is not None:
            e.step(b, {'ts': prev['ts'], 'signal': True, 'distance': 2., 'regime': 'fx'})
        prev = b
    return e


def decisions(e):
    return [x['decision'] for x in e.journal if x['event'] == 'FUNDAMENTAL_DECISION']


@pytest.mark.parametrize('pub,visible_from', [
    ('2025-05-10T07:00:00Z', '2025-05-11'),            # published 3h after the 05-10 bar timestamp: next day's decision
    ('2025-05-10T04:00:00Z', '2025-05-10'),            # published exactly at the bar timestamp: visible (<=)
    ('2025-05-10T03:59:59.999999Z', '2025-05-10'),
    ('2025-05-10T04:00:00.000001Z', '2025-05-11'),     # one microsecond later: not visible to the 05-10 decision
    ('2025-05-10T11:05:00Z', '2025-05-11'),            # pre-market filing (acceptance 06:00 ET + 5 min): not seen by the same-day decision although public before the open fill
    ('2025-05-10T21:35:00Z', '2025-05-11'),            # after-hours: next day
    ('2025-05-08T05:00:00Z', '2025-05-09'),            # date-only filing in winter is 05:00Z > 04:00Z bar stamp
])
def test_daily_bar_decision_sees_only_what_was_public_at_the_bar_timestamp(pub, visible_from):
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest(H.acme_nov_fy(pub))
    e = run_engine(s, bars('2025-05-07', 7))
    got = {d['timestamp'][:10]: d for d in decisions(e)}
    assert len(got) == 6
    for day, d in got.items():
        ts = d['timestamp']
        # every decision is pinned to its bar timestamp and never cites data published after it
        assert d['fundamental']['available_at'] is None or H.ns_instant(d['fundamental']['available_at']) <= H.ns_instant(ts), (day, d['fundamental']['available_at'])
        new_visible = day >= visible_from
        assert d['fundamental']['period_end'] == ('2025-03-31' if new_visible else '2024-03-31'), (pub, day)
        # reproducibility: a database that physically lacks everything published after ts yields the same data_snapshot_id
        trunc = PointInTimeFundamentalStore(':memory:')
        trunc.ingest([f for f in H.acme_nov_fy(pub) if H.ns_instant(f.available_at) <= H.ns_instant(ts)])
        again = CombinedSignalEngine(FundamentalFeatureEngine(trunc), policy='filter', research_shadow=True).decide('ACME', ts, {'signal': True})
        assert again['data_snapshot_id'] == d['data_snapshot_id'], (pub, day)


def test_decision_with_model_never_trades_on_a_filing_published_after_the_decision_instant():
    """A very strong score exists only in the (future) FY2025 filing; the decision before it must stay NO_TRADE."""
    s = PointInTimeFundamentalStore(':memory:')
    facts = H.acme_nov_fy('2025-05-10T07:00:00Z')
    # make FY2025 spectacular and FY2024 poor so that the model score flips sign at publication
    facts = [f for f in facts if not (f.metric == 'revenue' and f.period_end == '2025-03-31')]
    facts.append(fact('revenue', 5000 * M, '2025-03-31', '2024-04-01', '2025-05-10T07:00:00Z', rev='fy25'))
    s.ingest(facts)
    art = ModelArtifact(('revenue_yoy',), (0.,), (1.,), (1.,), -1., '2000-01-01T00:00:00Z', 1., 100, 'x', 5)  # score = yoy - 1 (>0 only when yoy>1)
    e = run_engine(s, bars('2025-05-08', 5), model=art)
    by_day = {d['timestamp'][:10]: d for d in decisions(e)}
    assert by_day['2025-05-10']['permit_signal'] is False and by_day['2025-05-10']['fundamental']['score'] == pytest.approx(800 / 700 - 1 - 1)
    assert by_day['2025-05-11']['permit_signal'] is True and by_day['2025-05-11']['fundamental']['score'] == pytest.approx(5000 / 800 - 1 - 1)
    # the executed position (if any) in the earlier bars is zero: only the later decision may size a trade
    entries = [x for x in e.journal if x['event'] == 'ENTRY']
    assert all(x['timestamp'][:10] >= '2025-05-11' for x in entries)


def test_paper_engine_fails_closed_without_changing_state_on_same_instant_conflict():
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest(H.acme_nov_fy('2025-05-10T07:00:00Z'))
    s.ingest([fact('revenue', 1. * M, '2025-03-31', '2024-04-01', '2025-05-11T00:00:00Z', rev='A'), fact('revenue', 2. * M, '2025-03-31', '2024-04-01', '2025-05-11T00:00:00Z', rev='B')])
    combo = CombinedSignalEngine(FundamentalFeatureEngine(s), policy='filter', research_shadow=True)
    e = PaperEngine(symbol='ACME', shadow=True, fundamental_filter=combo)
    before = (json.dumps(e.state, sort_keys=True), len(e.journal), len(e.curve))
    with pytest.raises(ValueError, match='ambiguous'):
        e.step(dict(ts='2025-05-12T04:00:00+00:00', open=100., high=101., low=99., close=100.), {'ts': '2025-05-11T04:00:00+00:00', 'signal': True, 'distance': 2., 'regime': 'fx'})
    assert (json.dumps(e.state, sort_keys=True), len(e.journal), len(e.curve)) == before and e.state['position'] is None and e.state['cash'] == 100.


def test_intraday_same_instant_publication_is_visible_to_the_decision_at_that_instant_documented_limit():
    """LIMITATION: available_at == bar ts is visible (<=). With hourly bars the fill is at the same instant's open,
    so a fact published exactly at the bar open can influence a fill at that open price. SEC facts carry +5 min margin."""
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest(H.acme_nov_fy('2025-05-10T10:00:00Z'))
    hourly = [dict(ts=f'2025-05-10T0{h}:00:00+00:00' if h < 10 else f'2025-05-10T{h}:00:00+00:00', open=100., high=101., low=99., close=100.) for h in (8, 9, 10, 11)]
    e = run_engine(s, hourly)
    by = {d['timestamp'][11:13]: d['fundamental']['period_end'] for d in decisions(e)}
    assert by == {'09': '2024-03-31', '10': '2025-03-31', '11': '2025-03-31'}
