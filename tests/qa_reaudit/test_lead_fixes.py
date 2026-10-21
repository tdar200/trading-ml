"""Regression tests for behaviours introduced by the lead engineer's fixes in the 2026-10-01 re-audit.

The per-track suites prove each defect is fixed; these pin the new contracts that no track asserted directly.
"""
import pytest

from fundamentals.adapters import insider_activity
from fundamentals.catalysts import extract_disclosures
from fundamentals.store import Fact, PointInTimeFundamentalStore, iso_date, utc


def own(value, tid, available, insider, code='P', revision='r1'):
    return Fact('XYZ', 'insider_purchase', value, '2025-03-10', available, available, 'form4', revision, 'USD', 'USD', '', available,
                'ownership', {'transaction_code': code, 'discretionary': True, 'insider_id': insider, 'transaction_id': tid})


def test_insider_transactions_with_distinct_transaction_ids_are_all_counted():
    s = PointInTimeFundamentalStore()
    s.ingest([own(250_000.0, 'T1', '2025-03-11T20:00:00Z', 'alice'), own(100_000.0, 'T2', '2025-03-12T20:00:00Z', 'bob')])
    out = insider_activity(s, 'XYZ', '2025-03-20T00:00:00Z')
    assert out['discretionary_purchase_value'] == 350_000.0 and out['distinct_buyers'] == 2


def test_insider_amendment_reusing_the_transaction_id_supersedes_without_double_counting():
    s = PointInTimeFundamentalStore()
    s.ingest([own(250_000.0, 'T1', '2025-03-11T20:00:00Z', 'alice', revision='orig'),
              own(80_000.0, 'T1', '2025-03-14T20:00:00Z', 'alice', revision='amend')])
    assert insider_activity(s, 'XYZ', '2025-03-13T00:00:00Z')['discretionary_purchase_value'] == 250_000.0   # before the amendment
    assert insider_activity(s, 'XYZ', '2025-03-20T00:00:00Z')['discretionary_purchase_value'] == 80_000.0    # after: replaced, not added


@pytest.mark.parametrize('text,expected', [
    ('2025-05-10T07:00:00Z', '2025-05-10T07:00:00+00:00'),
    ('2025-05-10 07:00:00+00:00', '2025-05-10T07:00:00+00:00'),
    ('2025-05-10T08:00:00+01:00', '2025-05-10T07:00:00+00:00'),
    ('2025-05-10T03:00-04:00', '2025-05-10T07:00:00+00:00'),
    ('2025-05-10T12:30:00+0530', '2025-05-10T07:00:00+00:00'),
    ('2025-05-10T07:00:00.123456789Z', '2025-05-10T07:00:00.123456789+00:00'),
])
def test_utc_accepts_iso_8601_with_an_explicit_offset(text, expected):
    assert utc(text) == expected


@pytest.mark.parametrize('text', ['2025-05-10', '2025-05-10T07:00:00', '04/05/2025 07:00:00Z', '24-03-04', 'Sat, 10 May 2025 07:00:00 GMT',
                                  '2025-05-10T07:00:00 BST', '2025-05-10T07:00:00 01:00', '', '2025-13-40T00:00:00Z', 'now'])
def test_utc_rejects_everything_else(text):
    with pytest.raises(ValueError):
        utc(text)


def test_iso_date_is_strict():
    assert iso_date('2024-03-31') == '2024-03-31' and iso_date('2024-03-31T00:00:00') == '2024-03-31'
    for bad in ('03/31/2024', '31-03-2024', 'March 31 2024', None, ''):
        with pytest.raises(ValueError):
            iso_date(bad)


def test_extractor_treats_a_year_anchor_as_current_only_in_the_publication_year():
    sentence = 'In 2026, we raised our guidance.'
    assert extract_disclosures(sentence) == []                                           # no reference year: abstain
    assert extract_disclosures(sentence, reference_year=2027) == []                      # a different year is another period
    assert [a['tag'] for a in extract_disclosures(sentence, reference_year=2026)] == ['guidance_up']


def test_extractor_first_person_is_the_company_only_in_its_own_channels():
    text = 'We raised our guidance.'
    for channel in ('official_filing', 'official_news'):
        assert [a['tag'] for a in extract_disclosures(text, channel=channel)] == ['guidance_up']
    for channel in ('financial_news', 'social'):
        assert extract_disclosures(text, channel=channel) == []
    assert [a['tag'] for a in extract_disclosures('The company raised its guidance.', channel='social')] == ['guidance_up']
    with pytest.raises(ValueError):
        extract_disclosures(text, channel='telepathy')


def test_unserialisable_or_malformed_event_payload_is_rejected_at_write_time():
    s = PointInTimeFundamentalStore()
    base = dict(symbol='XYZ', event_type='earnings', event_at='2025-05-01T20:00:00Z', reported_at='2025-05-01T20:00:00Z',
                available_at='2025-05-01T20:05:00Z', source='feed', revision='1')
    for payload in ([1, 2], None, 'text', 5):
        with pytest.raises(ValueError):
            s.add_event(payload=payload, **base)
    with pytest.raises(ValueError):
        s.add_event(payload={'x': float('nan')}, **base)
    assert s.events('XYZ', '2025-06-01T00:00:00Z') == []
