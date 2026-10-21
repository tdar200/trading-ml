"""QA track B: units, scale, currency, CapEx sign, provider schema failures, fuzz and fetch behaviour.

Independent Decimal / stdlib oracles only; fundamentals.* is called only through public entry points
(Fact, PointInTimeFundamentalStore, FundamentalFeatureEngine.snapshot, providers.*, adapters.historical_fx, valuation_features).
Defect repros are strict xfail (QA-REAL-04 .. QA-REAL-09).
"""
from __future__ import annotations
import json
import math
import os
import random
import subprocess
import sys
import threading
from decimal import Decimal, getcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from fundamentals import providers
from fundamentals.adapters import historical_fx
from fundamentals.features import FundamentalFeatureEngine, accounting_features, valuation_features
from fundamentals.providers import alfred_snapshot, ingest_json_records, sec_facts
from fundamentals.store import Fact, PointInTimeFundamentalStore, utc

getcontext().prec = 60
REPO = Path(__file__).resolve().parents[2]
D = Decimal


def fact(metric='revenue', value=100.0, *, symbol='T', period_end='2024-12-31', period_start='2024-01-01', avail='2025-02-01T00:00:00+00:00',
         reported=None, currency='USD', units=None, source='S1', revision='r1', kind='financial', metadata=None, observed='2025-02-01T00:00:00+00:00'):
    if units is None:
        units = 'shares' if metric in ('diluted_shares', 'shares_outstanding') else (currency + '/shares' if metric == 'eps' else currency)
    if units == 'shares' and kind == 'financial':
        currency = ''
    return Fact(symbol, metric, value, period_end, reported or avail, avail, source, revision, currency, units, period_start, observed, kind, metadata or {})


def mem():
    return PointInTimeFundamentalStore(':memory:')


def stored_count(store):
    return store.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0]


# ===================================================================================================== A. units / scale
@pytest.mark.parametrize('units,value,expected', [
    ('USD thousands', 5.0, D(5000)), ('USD millions', 2.5, D(2500000)), ('USD billions', 1.5, D(1500000000)),
    ('USD millions', -3.0, D(-3000000)), ('USD thousands', 0.0, D(0)), ('USD billions', 0.001, D(1000000)),
])
def test_real_scale_suffixes_exact(units, value, expected):
    f = fact('revenue', value, units=units)
    n = f.normalized()
    assert n['units'] == 'USD' and D(n['value']) == expected
    assert n['metadata']['original_units'] == units and n['metadata']['original_value'] == value
    assert n['metadata']['unit_multiplier'] == {'thousands': 1000, 'millions': 1000000, 'billions': 1000000000}[units.split()[1]]


@pytest.mark.parametrize('currency', ['GBP', 'EUR', 'JPY', 'CHF', 'CAD'])
def test_real_scale_in_other_currencies_keeps_currency(currency):
    n = fact('revenue', 7.25, currency=currency, units=f'{currency} millions').normalized()
    assert n['currency'] == currency and n['units'] == currency and n['value'] == 7_250_000.0


@pytest.mark.parametrize('units', ['USD million', 'USD 1 million', 'USD in millions', 'USD mn', 'usd millions', 'USD Millions', 'USD M',
                                   'EUR millions', 'USD (millions)', 'USD 000s', 'millions'])
def test_real_unrecognised_scale_strings_rejected_not_guessed(units):
    with pytest.raises(ValueError):
        fact('revenue', 1.0, units=units).normalized()


@pytest.mark.parametrize('metric,units,currency', [
    ('eps', 'USD', 'USD'),                       # per-share value with a total-amount unit
    ('revenue', 'USD/shares', 'USD'),            # total amount with a per-share unit
    ('eps', 'USD millions', 'USD'),              # scale suffix on a per-share metric
    ('eps', 'USD/shares millions', 'USD'),
    ('diluted_shares', 'shares thousands', ''),  # shares can not be scaled through units
    ('diluted_shares', 'shares millions', ''),
    ('diluted_shares', 'USD', 'USD'),            # share count labelled as money
    ('revenue', 'shares', ''),                   # money labelled as shares
])
def test_real_per_share_vs_total_and_share_scale_contracts(metric, units, currency):
    with pytest.raises(ValueError):
        fact(metric, 1.0, units=units, currency=currency).normalized()


def test_real_one_million_dollars_equals_one_in_millions_and_never_conflicts():
    store = mem()
    a = fact('revenue', 1_000_000.0, units='USD', revision='a')
    b = fact('revenue', 1.0, units='USD millions', revision='b')
    store.ingest([a, b])
    rows = store.as_of('T', '2025-03-01T00:00:00+00:00', kind='financial')        # same key + same value: no ambiguity raised
    assert len(rows) == 1 and rows[0]['value'] == 1_000_000.0 and rows[0]['units'] == 'USD'


def test_real_scale_float_artifact_is_below_1e_12_relative():
    for raw in (1.1, 2.675, 0.07, 123.456, 9.99, 3.3):
        n = fact('revenue', raw, units='USD millions').normalized()
        assert abs(D(n['value']) - D(str(raw)) * 1000000) / (D(str(raw)) * 1000000) < D('1e-12')


def test_real_json_import_scale_and_shares_scale(tmp_path):
    records = [dict(symbol='T', metric='revenue', value=245.122, period_end='2024-06-30', reported_at='2024-07-30T20:11:22+00:00',
                    available_at='2024-07-30T20:16:22+00:00', source='x', revision='1', currency='USD', units='USD millions', period_start='2023-07-01')]
    path = tmp_path / 'ok.json'
    path.write_text(json.dumps(records))
    store = mem()
    assert ingest_json_records(store, path) == 1
    row = store.as_of('T', '2024-08-01T00:00:00+00:00', kind='financial')[0]
    assert row['value'] == pytest.approx(245_122_000.0, rel=1e-15) and json.loads(json.dumps(row['metadata']))['original_units'] == 'USD millions'
    # LIMITATION: share counts cannot be scaled; a provider quoting "shares in millions" is rejected as a whole file
    records[0].update(metric='diluted_shares', currency='', units='shares millions', value=7433.0)
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError):
        ingest_json_records(mem(), path)


# ===================================================================================================== B. currency
def _annual(store, symbol, currency, year=2024, avail='2025-03-01T12:00:00+00:00', scale='', mult=1.0, **kw):
    for metric, v in dict(revenue=1200, net_income=180, ocf=250, capex=50, assets=900, equity=450, cash=300, gross_profit=700, operating_income=300).items():
        flow = metric in ('revenue', 'net_income', 'ocf', 'capex', 'gross_profit', 'operating_income')
        units = f'{currency} {scale}' if scale else currency
        store.ingest([fact(metric, v * mult, symbol=symbol, currency=currency, units=units, period_start=f'{year}-01-01' if flow else '',
                           period_end=f'{year}-12-31', avail=avail, **kw)])


@pytest.mark.parametrize('currency', ['GBP', 'USD', 'EUR', 'JPY'])
def test_real_currency_fixtures_keep_original_currency_and_exact_ratios(currency):
    store = mem()
    _annual(store, 'X', currency, scale='millions')
    snap = FundamentalFeatureEngine(store).snapshot('X', '2025-03-02T00:00:00+00:00')
    assert snap['currency'] == currency and snap['status'] == 'AVAILABLE'
    rows = store.db.execute("SELECT DISTINCT currency FROM observations WHERE metric IN ('revenue','assets')").fetchall()
    assert [r[0] for r in rows] == [currency]                 # the stored rows keep the original currency, no conversion
    f, raw = snap['features'], snap['raw']
    assert raw['revenue'] == 1_200_000_000.0 and raw['capex'] == 50_000_000.0
    for name, num, den in (('net_margin', 180, 1200), ('gross_margin', 700, 1200), ('operating_margin', 300, 1200), ('fcf_margin', 200, 1200)):
        assert D(f[name]) == pytest.approx(D(num) / D(den), rel=D('1e-12'))
    assert f['fcf'] == 200_000_000.0 and f['ocf_net_income'] == pytest.approx(250 / 180, rel=1e-12)


def test_real_mixed_currency_history_is_never_converted_or_mixed():
    store = mem()
    _annual(store, 'M', 'USD', year=2023, avail='2024-03-01T12:00:00+00:00')
    _annual(store, 'M', 'GBP', year=2024, avail='2025-03-01T12:00:00+00:00')
    snap = FundamentalFeatureEngine(store).snapshot('M', '2024-06-01T00:00:00+00:00')
    assert snap['currency'] == 'USD'
    # after the reporting currency changes production fails closed (loud ValueError), it never blends USD and GBP
    with pytest.raises(ValueError, match='ambiguous quarter source/currency/units'):
        FundamentalFeatureEngine(store).snapshot('M', '2025-06-01T00:00:00+00:00')


def test_real_valuation_refuses_cross_currency_and_unmatched_split_basis():
    values = dict(net_income=10.0, revenue=100.0, equity=50.0, ocf=12.0, capex=2.0, cash=5.0)
    ok = valuation_features(20.0, 10.0, values, price_currency='GBP', statement_currency='GBP', price_basis='A', share_basis='A')
    assert ok['market_cap'] == 200.0 and ok['pe'] == 20.0
    with pytest.raises(ValueError, match='currency conversion'):
        valuation_features(20.0, 10.0, values, price_currency='USD', statement_currency='GBP', price_basis='A', share_basis='A')
    with pytest.raises(ValueError):
        valuation_features(20.0, 10.0, values, price_currency='GBP', statement_currency='GBP', price_basis='A', share_basis='B')


def _fx(store, pair, rate, period_end, avail, source='ECB', revision='q', cur=None, units=None):
    src, tgt = pair[:3], pair[3:]
    store.ingest([Fact('FX:' + pair, 'rate', rate, period_end, avail, avail, source, revision, cur or tgt, units or f'{tgt}/{src}', '', avail, kind='fx')])


def test_real_historical_fx_boundaries_staleness_direction_and_future_quotes():
    s = mem()
    _fx(s, 'USDGBP', 0.80, '2024-05-01', '2024-05-01T16:00:00+00:00')
    _fx(s, 'USDGBP', 0.90, '2024-05-20', '2024-05-20T16:00:00+00:00')
    q = lambda at, amt=100.0, a='USD', b='GBP': historical_fx(s, amt, a, b, at)
    assert q('2024-04-30T00:00:00+00:00')['value'] is None                              # nothing known yet
    assert q('2024-05-01T15:59:59.999999+00:00')['value'] is None                       # one microsecond before availability
    assert q('2024-05-01T16:00:00+00:00')['value'] == 80.0                              # exactly at availability
    assert q('2024-05-08T16:00:00+00:00')['value'] == 80.0                              # exactly 7 days after: still allowed
    assert q('2024-05-08T16:00:01+00:00')['value'] is None                              # stale beyond 7 days
    assert q('2024-05-19T00:00:00+00:00')['value'] is None                              # the 20 May quote is NOT visible, old one is stale
    assert q('2024-05-20T16:00:00+00:00')['value'] == 90.0 and q('2024-05-20T16:00:00+00:00')['rate'] == 0.9
    assert q('2024-05-02T00:00:00+00:00', 100.0, 'GBP', 'USD')['value'] is None         # no silent inversion
    assert q('2024-05-02T00:00:00+00:00', 100.0, 'GBP', 'GBP')['value'] == 100.0
    for bad in (float('nan'), float('inf')):
        with pytest.raises(ValueError):
            q('2024-05-02T00:00:00+00:00', bad)


def test_real_historical_fx_wrong_direction_and_nonpositive_rates():
    s = mem()
    _fx(s, 'USDGBP', 1.25, '2024-05-01', '2024-05-01T16:00:00+00:00', cur='USD', units='USD/GBP')      # quote stored the wrong way round
    with pytest.raises(ValueError, match='do not match'):
        historical_fx(s, 100.0, 'USD', 'GBP', '2024-05-02T00:00:00+00:00')
    for rate in (0.0, -0.8):
        t = mem()
        _fx(t, 'USDGBP', rate, '2024-05-01', '2024-05-01T16:00:00+00:00')
        assert historical_fx(t, 100.0, 'USD', 'GBP', '2024-05-02T00:00:00+00:00')['value'] is None


def test_real_pit_fx_valuation_uses_only_the_rate_known_at_T_and_matches_decimal_oracle():
    s = mem()
    _fx(s, 'USDGBP', 0.80, '2024-05-01', '2024-05-01T16:00:00+00:00')
    _fx(s, 'USDGBP', 0.50, '2024-05-03', '2024-05-06T08:00:00+00:00')                  # Friday quote published Monday (future at T)
    T = '2024-05-03T12:00:00+00:00'
    fx = historical_fx(s, 1_000_000_000.0, 'USD', 'GBP', T)
    assert fx['rate'] == 0.80
    net_income_gbp = fx['value'] * 0.1
    out = valuation_features(25.0, 400e6, dict(net_income=net_income_gbp), price_currency='GBP', statement_currency='GBP', price_basis='A', share_basis='A')
    assert out['market_cap'] == 25.0 * 400e6 and D(out['pe']) == pytest.approx(D(25) * D(400e6) / (D(1_000_000_000) * D('0.8') * D('0.1')), rel=D('1e-12'))
    later = historical_fx(s, 1_000_000_000.0, 'USD', 'GBP', '2024-05-06T08:00:00+00:00')
    assert later['rate'] == 0.50


def test_real_defect_fx_two_sources_silently_pick_one():
    s = mem()
    _fx(s, 'USDGBP', 0.80, '2024-05-01', '2024-05-01T16:00:00+00:00', source='ECB', revision='a')
    _fx(s, 'USDGBP', 0.85, '2024-05-01', '2024-05-01T17:00:00+00:00', source='BOE', revision='b')
    try:
        out = historical_fx(s, 100.0, 'USD', 'GBP', '2024-05-02T00:00:00+00:00')
    except ValueError:
        return                                  # failing closed is acceptable
    assert out['value'] is None                 # or refusing to convert; silently choosing 80.0 is the defect


# ===================================================================================================== C. CapEx sign and FCF
def test_real_signed_negative_capex_is_rejected_or_quarantined_never_flipped():
    store = mem()
    good = fact('capex', 30.0, revision='pos')
    bad = fact('capex', -30.0, revision='neg', period_end='2023-12-31', period_start='2023-01-01')
    result = store.ingest_validated([good, bad])
    assert (result['accepted'], result['rejected'], result['status']) == (1, 1, 'QUARANTINED')
    q = store.db.execute('SELECT reason,payload FROM ingestion_rejections').fetchone()
    assert 'capex' in q[0] and json.loads(q[1])['value'] == -30.0            # verbatim, not "repaired" to +30
    assert [r['value'] for r in store.as_of('T', '2025-03-01T00:00:00+00:00', kind='financial') if r['metric'] == 'capex'] == [30.0]
    with pytest.raises(ValueError):
        store.ingest([bad])
    with pytest.raises(ValueError, match='nonnegative'):
        accounting_features({'ocf': 100.0, 'capex': -30.0, 'revenue': 1000.0})


@pytest.mark.parametrize('ocf,capex,expected', [(100.0, 30.0, 70.0), (100.0, 0.0, 100.0), (-50.0, 10.0, -60.0), (0.0, 25.5, -25.5), (1e12, 4e11, 6e11)])
def test_real_fcf_is_ocf_minus_capex_exactly_once(ocf, capex, expected):
    store = mem()
    for metric, v in (('revenue', 1000.0), ('ocf', ocf), ('capex', capex), ('net_income', 80.0)):
        store.ingest([fact(metric, v)])
    snap = FundamentalFeatureEngine(store).snapshot('T', '2025-03-01T00:00:00+00:00')
    assert snap['features']['fcf'] == expected
    assert snap['features']['fcf_margin'] == pytest.approx(expected / 1000.0, rel=1e-12)
    assert snap['features']['fcf_net_income'] == pytest.approx(expected / 80.0, rel=1e-12)
    assert ('NEGATIVE_FCF' in snap['flags']) == (expected < 0)


def test_real_missing_capex_makes_fcf_missing_never_ocf():
    store = mem()
    for metric, v in (('revenue', 1000.0), ('ocf', 100.0)):
        store.ingest([fact(metric, v)])
    snap = FundamentalFeatureEngine(store).snapshot('T', '2025-03-01T00:00:00+00:00')
    assert snap['features']['fcf'] is None and snap['features']['fcf_margin'] is None and 'NEGATIVE_FCF' not in snap['flags']
    assert snap['ttm']['values']['fcf'] is None


def test_real_ttm_capex_from_ytd_cash_flow_differences_is_not_double_counted():
    """10-Q cash-flow style cumulative capex: Q1=10, 6M=25, 9M=45, FY=70 -> quarters 10,15,20,25 (sum 70, not 10+25+45+70)."""
    store = mem()
    spec = [('2024-01-01', '2024-03-31', 10.0), ('2024-01-01', '2024-06-30', 25.0), ('2024-01-01', '2024-09-30', 45.0), ('2024-01-01', '2024-12-31', 70.0)]
    for i, (s, e, v) in enumerate(spec):
        avail = ['2024-04-30', '2024-07-30', '2024-10-30', '2025-02-20'][i] + 'T12:00:00+00:00'
        for metric, val in (('capex', v), ('revenue', v * 10), ('ocf', v * 3), ('net_income', v)):
            store.ingest([fact(metric, val, period_start=s, period_end=e, avail=avail, revision=f'a{i}')])
    snap = FundamentalFeatureEngine(store).snapshot('T', '2025-02-21T00:00:00+00:00')
    assert snap['ttm']['values']['capex'] == 70.0 and snap['ttm']['values']['ocf'] == 210.0 and snap['ttm']['values']['fcf'] == 140.0
    assert snap['raw']['capex'] == 70.0


# ===================================================================================================== D. provider schema failures
ROW = dict(start='2024-01-01', end='2024-12-31', val=1000000, accn='0000000001-25-000001', fy=2024, fp='FY', form='10-K', filed='2025-02-10')


def payload(rows, tag='Revenues', unit='USD', taxonomy='us-gaap'):
    return {'cik': 1234, 'entityName': 'T', 'facts': {taxonomy: {tag: {'units': {unit: rows}}}}}


def run_sec(p, subs=None):
    facts = sec_facts(p, subs if subs is not None else {}, 'T')
    store = mem()
    result = store.ingest_validated(facts)
    return facts, store, result


def test_real_sec_baseline_and_acceptance_rules():
    facts, store, result = run_sec(payload([ROW]))
    assert (result['accepted'], stored_count(store)) == (1, 1)
    assert facts[0].available_at == '2025-02-11T05:00:00+00:00' and facts[0].metadata['timing_precision'] == 'following_local_day'
    sub = {'filings': {'recent': {'accessionNumber': [ROW['accn']], 'acceptanceDateTime': ['2025-02-10T22:30:00.000Z']}}}
    f2 = sec_facts(payload([ROW]), sub, 'T')[0]
    assert f2.available_at == '2025-02-10T22:35:00+00:00' and f2.metadata['timing_precision'] == 'acceptance+5min'
    # acceptance before the measured period is physically impossible -> quarantined, not trusted
    sub['filings']['recent']['acceptanceDateTime'] = ['2024-06-01T10:00:00.000Z']
    _, store3, result3 = run_sec(payload([ROW]), sub)
    assert (result3['accepted'], result3['rejected']) == (0, 1) and stored_count(store3) == 0


@pytest.mark.parametrize('label,mutate', [
    ('nan', dict(val=float('nan'))), ('inf', dict(val=float('inf'))), ('minus inf', dict(val=float('-inf'))),
    ('bad end date', dict(end='notadate')), ('end before start', dict(start='2025-01-01')),
])
def test_real_sec_bad_values_are_quarantined_verbatim(label, mutate):
    facts, store, result = run_sec(payload([{**ROW, **mutate}]))
    assert result['rejected'] == 1 and result['accepted'] == 0 and result['status'] == 'QUARANTINED' and stored_count(store) == 0
    assert store.db.execute('SELECT COUNT(*) FROM ingestion_rejections').fetchone()[0] == 1


@pytest.mark.parametrize('label,body', [
    ('renamed units key', {'cik': 1, 'facts': {'us-gaap': {'Revenues': {'unit': {'USD': [ROW]}}}}}),
    ('non-allowlisted currency CAD', payload([ROW], unit='CAD')), ('TWD', payload([ROW], unit='TWD')),
    ('unit with scale text', payload([ROW], unit='USD millions')), ('8-K row', payload([{**ROW, 'form': '8-K'}])),
    ('form missing', payload([{k: v for k, v in ROW.items() if k != 'form'}])), ('unmapped tag', payload([ROW], tag='SomethingElse')),
])
def test_real_sec_unknown_schema_yields_zero_facts_not_fabricated(label, body):
    """LIMITATION: schema drift is silent (no error, status VALIDATED) but never fabricates a value."""
    facts, store, result = run_sec(body)
    assert facts == [] and stored_count(store) == 0 and result == {'accepted': 0, 'rejected': 0, 'status': 'VALIDATED'}


def test_real_sec_wrong_units_for_metric_are_quarantined():
    _, _, r1 = run_sec(payload([{**ROW, 'val': 1.2}], tag='EarningsPerShareDiluted', unit='USD'))
    _, _, r2 = run_sec(payload([{**ROW, 'val': 1000}], tag='WeightedAverageNumberOfDilutedSharesOutstanding', unit='USD'))
    assert r1['rejected'] == 1 and r2['rejected'] == 1


def test_real_sec_conflicting_duplicate_contexts_become_explicit_missing():
    facts, store, _ = run_sec(payload([ROW, {**ROW, 'val': 2000000}]))
    assert len(facts) == 1 and facts[0].value is None and facts[0].metadata['quality_flag'] == 'CONFLICTING_CONTEXTS'
    row = store.as_of('T', '2025-03-01T00:00:00+00:00', kind='financial')[0]
    assert row['value'] is None


LEAKS = [   # (label, callable) -> every one raises a NON-ValueError today
    ('sec_facts val None', lambda: sec_facts(payload([{**ROW, 'val': None}]), {}, 'T')),
    ('sec_facts val missing', lambda: sec_facts(payload([{k: v for k, v in ROW.items() if k != 'val'}]), {}, 'T')),
    ('sec_facts accn missing', lambda: sec_facts(payload([{k: v for k, v in ROW.items() if k != 'accn'}]), {}, 'T')),
    ('sec_facts end missing', lambda: sec_facts(payload([{k: v for k, v in ROW.items() if k != 'end'}]), {}, 'T')),
    ('sec_facts huge int', lambda: sec_facts(payload([{**ROW, 'val': 10 ** 400}]), {}, 'T')),
    ('sec_facts facts missing', lambda: sec_facts({'cik': 1}, {}, 'T')),
    ('sec_facts payload not a dict', lambda: sec_facts([1, 2], {}, 'T')),
    ('sec_facts submissions not a dict', lambda: sec_facts(payload([ROW]), [1, 2], 'T')),
    ('sec_facts submissions.recent null', lambda: sec_facts(payload([ROW]), {'filings': {'recent': None}}, 'T')),
    ('Fact huge int', lambda: fact('revenue', 10 ** 400).normalized()),
    ('Fact int*scale overflow', lambda: fact('revenue', 10 ** 300, units='USD billions').normalized()),
    ('Fact period_end None', lambda: fact('revenue', 1.0, period_end=None).normalized()),
    ('Fact metadata set', lambda: mem().ingest([fact('revenue', 1.0, metadata={'a': {1}})])),
    ('alfred tz-aware vintage', lambda: alfred_snapshot(b'observation_date,CPIAUCSL_20240601\n2024-01-01,1\n', 'CPIAUCSL', '2024-06-01T00:00:00+00:00')),
]


@pytest.mark.parametrize('label', [l for l, _ in LEAKS])
def test_real_provider_failures_raise_only_the_controlled_value_error(label):
    """Lead re-audit (QA-REAL-05 fixed): every malformed input is a controlled ValueError, never TypeError/KeyError/OverflowError."""
    call = dict(LEAKS)[label]
    try:
        call()                      # a null/absent optional container may be tolerated; a malformed required one must not leak
    except ValueError:
        pass


@pytest.mark.parametrize('label', [l for l, _ in LEAKS])
def test_real_defect_provider_failures_must_be_controlled(label):
    call = dict(LEAKS)[label]
    try:
        call()
    except ValueError:
        return
    except Exception as err:             # noqa: BLE001
        pytest.fail(f'{label}: leaked {type(err).__name__}: {err}')


def test_real_one_bad_row_is_quarantined_not_poisoning_the_company_ingest():
    """Lead re-audit (QA-REAL-05 fixed): 1 malformed row out of 3 -> a controlled ValueError without `rejects`, and with a
    `rejects` list the two good rows are still produced and the bad row is returned verbatim for quarantine."""
    rows = [ROW, {**ROW, 'end': '2023-12-31', 'start': '2023-01-01', 'accn': '0000000001-24-000001'}, {**ROW, 'end': '2022-12-31', 'start': '2022-01-01', 'accn': 'x', 'val': None}]
    with pytest.raises(ValueError):
        sec_facts(payload(rows), {}, 'T')
    rejects = []
    facts = sec_facts(payload(rows), {}, 'T', rejects=rejects)
    assert len(facts) == 2 and len(rejects) == 1 and rejects[0]['row']['val'] is None and 'non-numeric' in rejects[0]['reason']
    store = mem()
    store.quarantine('T', 'malformed source row: ' + rejects[0]['reason'], rejects[0])
    assert store.quality('T')['status'] == 'QUARANTINED_RECORDS'


def test_real_boolean_source_value_is_a_corrupt_number_not_one():
    """Lead re-audit (QA-REAL-09 fixed): a JSON boolean (or string) is rejected as a non-numeric source value; with a `rejects`
    list it is returned verbatim for quarantine instead of becoming 1.0."""
    with pytest.raises(ValueError, match='non-numeric'):
        run_sec(payload([{**ROW, 'val': True}]))
    rejects = []
    assert sec_facts(payload([{**ROW, 'val': True}]), {}, 'T', rejects=rejects) == [] and rejects[0]['row']['val'] is True


# ---- silent data loss in ingest(): INSERT OR IGNORE swallows NOT NULL violations -------------------------------
NOT_NULL_CASES = [
    ('period_start None (balance-sheet item from a JSON export with null)', dict(metric='assets', period_start=None)),
    ('currency None on a macro observation', dict(kind='macro', metric='CPI', currency=None, units='index', period_start='')),
]


@pytest.mark.parametrize('label,kw', NOT_NULL_CASES, ids=[c[0][:20] for c in NOT_NULL_CASES])
def test_real_null_fields_are_stored_as_not_applicable_not_silently_dropped(label, kw):
    """Lead re-audit (QA-REAL-04 fixed): JSON-null period_start / currency means 'not applicable' ('') and the row IS stored;
    ingest()'s success report now matches what was written."""
    store = mem()
    f = Fact('T', kw.get('metric', 'revenue'), 100.0, '2024-12-31', '2025-02-01T00:00:00+00:00', '2025-02-01T00:00:00+00:00', 'x', 'r1',
             kw.get('currency', 'USD'), kw.get('units', 'USD'), kw.get('period_start', ''), '2025-02-01T00:00:00+00:00', kw.get('kind', 'financial'))
    n = store.ingest([f])
    result = mem().ingest_validated([f])
    assert n == 1 and result == {'accepted': 1, 'rejected': 0, 'status': 'VALIDATED'}
    assert stored_count(store) == 1
    row = store.db.execute('SELECT currency, period_start FROM observations').fetchone()
    assert row[0] == (kw.get('currency', 'USD') or '') and row[1] == (kw.get('period_start', '') or '')


@pytest.mark.parametrize('label,kw', NOT_NULL_CASES, ids=[c[0][:20] for c in NOT_NULL_CASES])
def test_real_defect_silent_drop_must_reject_or_quarantine(label, kw):
    store = mem()
    f = Fact('T', kw.get('metric', 'revenue'), 100.0, '2024-12-31', '2025-02-01T00:00:00+00:00', '2025-02-01T00:00:00+00:00', 'x', 'r1',
             kw.get('currency', 'USD'), kw.get('units', 'USD'), kw.get('period_start', ''), '2025-02-01T00:00:00+00:00', kw.get('kind', 'financial'))
    try:
        result = store.ingest_validated([f])
    except ValueError:
        return
    assert result['accepted'] == stored_count(store), result


def test_real_defect_json_import_null_period_start_silently_lost(tmp_path):
    rec = dict(symbol='T', metric='assets', value=900.0, period_end='2024-12-31', reported_at='2025-02-01T00:00:00+00:00', available_at='2025-02-01T00:00:00+00:00',
               source='x', revision='1', currency='USD', units='USD', period_start=None)
    path = tmp_path / 'r.json'
    path.write_text(json.dumps([rec]))
    store = mem()
    try:
        n = ingest_json_records(store, path)
    except ValueError:
        return
    assert n == stored_count(store)


def test_real_ingest_reports_attempted_not_inserted_on_idempotent_reload():
    """LIMITATION/LOW: 'accepted' counts rows offered to INSERT OR IGNORE, so re-ingesting identical data still says accepted=N."""
    store = mem()
    f = fact('revenue', 100.0)
    assert store.ingest_validated([f])['accepted'] == 1 and store.ingest_validated([f])['accepted'] == 1 and stored_count(store) == 1


# ---- alfred_snapshot and json import fail-safe behaviour --------------------------------------------------------
GOOD_CSV = 'observation_date,CPIAUCSL_20240601\n2024-01-01,310.3\n2024-02-01,311.1\n'


def test_real_alfred_wrong_schema_fails_closed():
    assert [f.value for f in alfred_snapshot(GOOD_CSV.encode(), 'CPIAUCSL', '2024-06-01')] == [310.3, 311.1]
    for body, vintage in ((GOOD_CSV, '2024-07-01'), (GOOD_CSV.replace('observation_date', 'DATE'), '2024-06-01'),
                          ('<html>rate limited</html>', '2024-06-01'), ('observation_date,CPIAUCSL_20240601\n2024-07-01,300\n', '2024-06-01')):
        with pytest.raises(ValueError):
            alfred_snapshot(body.encode(), 'CPIAUCSL', vintage)
    # ALFRED '.' / text values are missing observations, silently skipped (documented), never zero
    got = alfred_snapshot(b'observation_date,CPIAUCSL_20240601\n2024-01-01,.\n2024-02-01,abc\n2024-03-01,5\n', 'CPIAUCSL', '2024-06-01')
    assert [(f.period_end, f.value) for f in got] == [('2024-03-01', 5.0)]
    # nonfinite values reach the store's validator and the whole snapshot is rejected atomically
    store = mem()
    with pytest.raises(ValueError):
        store.ingest(alfred_snapshot(b'observation_date,CPIAUCSL_20240601\n2024-01-01,5\n2024-02-01,inf\n', 'CPIAUCSL', '2024-06-01'))
    assert stored_count(store) == 0
    with pytest.raises(ValueError, match='explicit units'):
        providers.ingest_alfred(mem(), 'UNMAPPED', '2024-06-01', 'unused-cache-dir')   # rejected before any fetch


def test_real_json_import_failures_reject_whole_file_and_archive_raw(tmp_path):
    base = dict(symbol='T', metric='revenue', value=100.0, period_end='2024-12-31', reported_at='2025-02-01T00:00:00+00:00',
                available_at='2025-02-01T00:00:00+00:00', source='x', revision='r', currency='USD', units='USD', period_start='2024-01-01')
    cases = {'nan': dict(value=float('nan')), 'str': dict(value='100'), 'bool': dict(value=True), 'currency': dict(currency='XXX'),
             'future period': dict(period_end='2026-01-01'), 'reversed': dict(period_start='2024-12-31', period_end='2024-01-01'),
             'naive ts': dict(available_at='2025-02-01T00:00:00'), 'avail<reported': dict(available_at='2025-01-31T00:00:00+00:00')}
    for name, patch in cases.items():
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps([base, {**base, 'revision': 'bad', **patch}], allow_nan=True))
        store = mem()
        with pytest.raises(ValueError):
            ingest_json_records(store, path)
        assert stored_count(store) == 0                      # atomic: the good record is not half-imported
        assert store.db.execute('SELECT COUNT(*) FROM raw_documents').fetchone()[0] == 1       # raw evidence archived first


# ===================================================================================================== E. timestamp parsing
def test_real_timestamp_normalisation_offsets_and_precision():
    assert utc('2025-03-09T03:00:00-04:00') == '2025-03-09T07:00:00+00:00'                   # US DST day
    assert utc('2025-11-02T01:30:00-05:00') == '2025-11-02T06:30:00+00:00'
    assert utc('2025-03-30T01:30:00+00:00') == utc('2025-03-30T01:30:00Z') == '2025-03-30T01:30:00+00:00'
    assert utc('2025-05-10T08:00:00+01:00') == '2025-05-10T07:00:00+00:00'
    assert utc('2025-05-10T07:00:00.123456789+00:00') == '2025-05-10T07:00:00.123456789+00:00'
    for naive in ('2025-05-10', '2025-05-10T07:00:00', 1715324400, 1715324400.0, None, float('nan'), ''):
        with pytest.raises(ValueError):
            utc(naive)


AMBIGUOUS = [('YY-MM-DD read as 2004-03-24', '24-03-04 09:00 +00:00', '2024-03-04'),
             ('dd/mm read as mm/dd (UK 3 April -> 4 March, a month early)', '03/04/2024 00:00 +00:00', '2024-04-03')]


@pytest.mark.parametrize('label,text,intended_day', AMBIGUOUS, ids=[a[0][:14] for a in AMBIGUOUS])
def test_real_defect_ambiguous_timestamp_formats_must_be_rejected(label, text, intended_day):
    try:
        got = utc(text)
    except ValueError:
        return
    assert got.startswith(intended_day), f'{text!r} silently became {got}'


def test_real_defect_tz_abbreviation_parse_depends_on_host_timezone():
    code = "import sys;sys.path.insert(0,%r);from fundamentals.store import utc\ntry: print(utc('2024-07-04 09:00 BST'))\nexcept ValueError: print('ValueError')" % str(REPO)
    outs = []
    for tz in ('UTC', 'Europe/London'):
        env = {**os.environ, 'TZ': tz}
        outs.append(subprocess.run([sys.executable, '-W', 'ignore', '-c', code], capture_output=True, text=True, env=env, cwd=REPO, timeout=60).stdout.strip())
    assert outs[0] == outs[1], outs


# ===================================================================================================== F. fuzz
LEAK_TYPES = (OverflowError, TypeError)


def _random_fact(rng):
    """90 % of the fields come from a valid pool so many records reach the deep validation branches; 10 % adversarial."""
    values = [0, 0.0, -1.0, 1e30, 1e-30, 1e300, -1e300, 5e-324, None, float('nan'), float('inf'), float('-inf'), 10 ** 400, 10 ** 18, True, '12', 'abc',
              123456789.123456789, 1.1, 2 ** 53 + 1]
    ends = ['2024-12-31', '2024-02-29', '2023-02-29', '2024-13-01', None, '', 'x', '2024-12-31T23:59:59-05:00', '9999-12-31', '1900-01-01']
    starts = ['', '2024-01-01', None, '2025-01-01', 'junk', '2024-12-31']
    stamps = ['2025-02-01T00:00:00+00:00', '2025-02-01T00:00:00', None, 'soon', '2025-02-01T00:00:00Z', '2025-02-01T05:00:00+05:00', '0000-00-00']
    units = ['USD', 'USD millions', 'USD thousands', 'usd', 'USD/shares', 'shares', 'EUR', '', 'USD million', 'USD in millions']

    def pick(good, bad):
        return rng.choice(good) if rng.random() < 0.9 else rng.choice(bad)
    metric = rng.choice(['revenue', 'eps', 'diluted_shares', 'capex', 'ocf', 'weird', ''])
    good_units = {'eps': ['USD/shares'], 'diluted_shares': ['shares']}.get(metric, ['USD', 'USD millions', 'USD thousands', 'USD billions'])
    good_cur = [''] if metric == 'diluted_shares' else ['USD']
    return dict(symbol=pick(['T'], ['T', 'x', '', ' ', None, 5]), metric=metric,
                value=pick([1.0, 100.5, 0.0, 12345678.9, -5.0], values), period_end=pick(['2024-12-31'], ends),
                reported_at=pick(['2025-02-01T00:00:00+00:00'], stamps), available_at=pick(['2025-02-01T00:00:00+00:00'], stamps),
                source=pick(['s'], ['s', '', None]), revision=pick(['r'], ['r', '', None]), currency=pick(good_cur, ['USD', 'EUR', 'JPY', 'GBP', '', None, 'XXX', 'usd']),
                units=pick(good_units, units), period_start=pick(['', '2024-01-01'], starts), observed_at=pick(['2025-02-01T00:00:00+00:00'], stamps),
                kind=pick(['financial'], ['financial', 'macro', 'weird']), metadata=pick([{}, {'a': 1}], [{}, {'a': 1}, None, {'a': float('nan')}, {'a': {1}}]))


@pytest.mark.parametrize('chunk', range(6))
def test_real_fuzz_facts_never_corrupt_silently_and_leaks_are_only_the_known_classes(chunk):
    """300 seeded random Facts per chunk x 6 chunks. Accepted facts must round-trip exactly; failures must be ValueError or the two
    known leak classes (QA-REAL-05); silent drops are only the two NOT NULL cases (QA-REAL-04)."""
    unexpected, silent, accepted_n = [], [], 0
    for seed in range(chunk * 300, chunk * 300 + 300):
        rng = random.Random(seed)
        spec = _random_fact(rng)
        try:
            f = Fact(**spec)
            n = f.normalized()
        except ValueError:
            continue
        except LEAK_TYPES:
            continue
        except Exception as err:     # noqa: BLE001
            unexpected.append((seed, type(err).__name__, str(err)[:80]))
            continue
        accepted_n += 1
        if n['value'] is not None:
            assert isinstance(n['value'], float) and math.isfinite(n['value']), (seed, n['value'])
        assert n['available_at'] >= n['reported_at'], seed
        assert len(n['period_end']) == 10, (seed, n['period_end'])
        if n['kind'] == 'financial':
            expected_units = 'shares' if n['metric'] in ('diluted_shares', 'shares_outstanding') else n['currency'] + ('/shares' if n['metric'] == 'eps' else '')
            assert n['units'] == expected_units, (seed, n['units'], expected_units)
            if n['metric'] == 'capex' and n['value'] is not None:
                assert n['value'] >= 0
        store = mem()
        try:
            count = store.ingest([f])
        except ValueError:
            continue
        except LEAK_TYPES:
            continue
        stored = stored_count(store)
        if stored != count:
            assert spec['period_start'] is None or spec['currency'] is None, (seed, spec)
            silent.append(seed)
        else:
            row = store.db.execute('SELECT value, units, period_end FROM observations').fetchone()
            if n['value'] is None:
                assert row['value'] is None
            else:
                assert row['value'] == n['value'] and row['period_end'] == n['period_end']
    assert not unexpected, unexpected[:5]
    assert accepted_n >= 40, accepted_n              # the generator really exercises the accepted branch


def test_real_fuzz_leaks_nothing_but_the_controlled_value_error():
    """Lead re-audit (QA-REAL-05 fixed): over the same 1800-seed generator no OverflowError/TypeError ever leaks."""
    seen = {}
    for seed in range(1800):
        rng = random.Random(seed)
        try:
            Fact(**_random_fact(rng)).normalized()
        except ValueError:
            pass
        except Exception as err:     # noqa: BLE001
            seen.setdefault(type(err).__name__, seed)
    assert seen == {}, seen


# numeric fuzz against a Decimal oracle through the PUBLIC path (Facts -> store -> snapshot)
def _decimal_ratio(a, b):
    return None if b <= 0 else D(a) / D(b)


@pytest.mark.parametrize('seed', range(60))
def test_real_fuzz_snapshot_ratios_match_decimal_oracle(seed):
    rng = random.Random(seed)
    scale = 10 ** rng.randint(3, 13)                                   # micro-cap .. trillion-scale statements
    cur = {k: round(rng.uniform(0.5, 5.0) * scale) * 1.0 for k in ('revenue', 'gross_profit', 'operating_income', 'net_income', 'ocf', 'capex', 'assets',
                                                                  'equity', 'cash', 'current_assets', 'current_liabilities')}
    prev = {k: round(rng.uniform(0.5, 5.0) * scale) * 1.0 for k in cur}
    if rng.random() < .3:
        cur['net_income'] = -cur['net_income']
    if rng.random() < .3:
        cur['ocf'] = -cur['ocf']
    store = mem()
    for year, vals, avail in ((2023, prev, '2024-03-01T12:00:00+00:00'), (2024, cur, '2025-03-01T12:00:00+00:00')):
        for metric, v in vals.items():
            flow = metric in ('revenue', 'gross_profit', 'operating_income', 'net_income', 'ocf', 'capex')
            store.ingest([fact(metric, v, period_start=f'{year}-01-01' if flow else '', period_end=f'{year}-12-31', avail=avail, revision=f'a{year}')])
    snap = FundamentalFeatureEngine(store).snapshot('T', '2025-03-02T00:00:00+00:00')
    f = snap['features']
    fcf = D(cur['ocf']) - D(cur['capex'])
    expect = {'fcf': fcf, 'gross_margin': _decimal_ratio(cur['gross_profit'], cur['revenue']), 'operating_margin': _decimal_ratio(cur['operating_income'], cur['revenue']),
              'net_margin': _decimal_ratio(cur['net_income'], cur['revenue']), 'fcf_margin': _decimal_ratio(fcf, cur['revenue']),
              'current_ratio': _decimal_ratio(cur['current_assets'], cur['current_liabilities']),
              'revenue_yoy': D(cur['revenue']) / D(prev['revenue']) - 1, 'ocf_net_income': _decimal_ratio(cur['ocf'], cur['net_income']),
              'roa': _decimal_ratio(cur['net_income'], (D(cur['assets']) + D(prev['assets'])) / 2),
              'roe': _decimal_ratio(cur['net_income'], (D(cur['equity']) + D(prev['equity'])) / 2),
              'capex_sales': _decimal_ratio(cur['capex'], cur['revenue'])}
    for name, want in expect.items():
        got = f[name]
        if want is None:
            assert got is None, (seed, name, got)
        else:
            assert got is not None and abs(D(got) - want) <= D('1e-12') * max(D(1), abs(want)), (seed, name, got, want)
    assert all(v is None or math.isfinite(v) for v in f.values())


@pytest.mark.parametrize('seed', range(80))
def test_real_fuzz_valuation_trillion_scale_and_micro_cap_vs_decimal(seed):
    rng = random.Random(1000 + seed)
    price = 10 ** rng.uniform(-3, 6)
    shares = 10 ** rng.uniform(3, 11.5)
    v = dict(net_income=rng.uniform(-1, 3) * 1e9, revenue=rng.uniform(.1, 5) * 1e9, equity=rng.uniform(-1, 4) * 1e9, ocf=rng.uniform(-1, 3) * 1e9,
             capex=rng.uniform(0, 1) * 1e9, cash=rng.uniform(0, 2) * 1e9, short_debt=rng.uniform(0, 1) * 1e9, long_debt=rng.uniform(0, 2) * 1e9,
             preferred_equity=0.0, minority_interest=0.0, operating_income=rng.uniform(-1, 2) * 1e9, depreciation=rng.uniform(0, 1) * 1e8)
    out = valuation_features(price, shares, v, price_currency='USD', statement_currency='USD', price_basis='A', share_basis='A', forward_eps=rng.choice([None, 2.5, -1.0, 0.0]))
    mc = D(price) * D(shares)
    ev = mc + D(v['short_debt']) + D(v['long_debt']) + 0 + 0 - D(v['cash'])
    fcf = D(v['ocf']) - D(v['capex'])
    ebitda = D(v['operating_income']) + D(v['depreciation'])
    want = {'market_cap': mc, 'enterprise_value': ev, 'pe': _decimal_ratio(v['net_income'], 1) and mc / D(v['net_income']) if v['net_income'] > 0 else None,
            'ps': mc / D(v['revenue']), 'pb': mc / D(v['equity']) if v['equity'] > 0 else None, 'ev_revenue': ev / D(v['revenue']),
            'ev_ebitda': ev / ebitda if ebitda > 0 else None, 'price_fcf': mc / fcf if fcf > 0 else None, 'fcf_yield': fcf / mc, 'earnings_yield': D(v['net_income']) / mc}
    for name, w in want.items():
        got = out[name]
        if w is None:
            assert got is None, (seed, name, got)
        else:
            assert got is not None and abs(D(got) - w) <= D('1e-12') * max(D(1), abs(w)), (seed, name, got, w)
    assert out['market_cap'] > 0 and all(x is None or math.isfinite(x) for x in out.values())


def test_real_valuation_overflow_and_invalid_inputs_fail_with_valueerror():
    ok = dict(net_income=1.0)
    for price, shares in ((1e308, 1e308), (0.0, 1.0), (-1.0, 1.0), (float('nan'), 1.0), (1.0, float('inf')), (1.0, 0.0)):
        with pytest.raises(ValueError):
            valuation_features(price, shares, ok, price_currency='USD', statement_currency='USD', price_basis='A', share_basis='A')
    big = valuation_features(190.0, 3.58e9, dict(net_income=1.0e11, revenue=4.0e11), price_currency='USD', statement_currency='USD', price_basis='A', share_basis='A')
    assert big['market_cap'] == pytest.approx(6.802e11, rel=1e-12) and big['pe'] == pytest.approx(6.802, rel=1e-12)


# ===================================================================================================== G. provider disagreement
def test_real_two_sources_one_billion_vs_one_point_one_billion():
    store = mem()
    for src, mult, minutes in (('A', 1.0, 0), ('B', 1.1, 60)):
        for metric, v in (('revenue', 1e9), ('net_income', 1e8), ('ocf', 1.5e8), ('capex', 5e7), ('assets', 2e9), ('equity', 1e9)):
            flow = metric in ('revenue', 'net_income', 'ocf', 'capex')
            store.ingest([fact(metric, v * mult, source=src, revision=src.lower(), period_start='2024-01-01' if flow else '',
                               avail=f'2025-03-01T{12 + minutes // 60:02d}:00:00+00:00')])
    eng = FundamentalFeatureEngine(store)
    # before the second provider publishes: one source, no disagreement visible
    before = eng.snapshot('T', '2025-03-01T12:30:00+00:00')
    assert before['raw']['revenue'] == 1e9
    # after: both stay in history and in as_of (distinct source keys); snapshot fails closed instead of choosing 1.0bn or 1.1bn
    assert {r['value'] for r in store.history('T', 'revenue')} == {1e9, 1.1e9}
    assert len([r for r in store.as_of('T', '2025-03-01T13:00:00+00:00', kind='financial') if r['metric'] == 'revenue']) == 2
    with pytest.raises(ValueError, match='ambiguous'):
        eng.snapshot('T', '2025-03-01T13:00:00+00:00')
    # NOT IMPLEMENTED: no discrepancy log table / quality flag records the disagreement
    assert store.quality('T')['status'] == 'NO_RECORDED_REJECTIONS'


def test_real_same_source_correction_is_a_new_version_not_a_conflict():
    store = mem()
    store.ingest([fact('revenue', 1e9, revision='a', avail='2025-03-01T12:00:00+00:00')])
    store.ingest([fact('revenue', 1.1e9, revision='b', avail='2025-03-01T13:00:00+00:00')])
    pick = lambda at: [r['value'] for r in store.as_of('T', at, kind='financial')]
    assert pick('2025-03-01T12:30:00+00:00') == [1e9] and pick('2025-03-01T13:00:00+00:00') == [1.1e9]
    assert len(store.history('T', 'revenue')) == 2                                  # audit trail keeps both versions


# ===================================================================================================== H. fetch / retry / cache / rate limit
class _Handler(BaseHTTPRequestHandler):
    plan: list = []
    hits: list = []

    def do_GET(self):
        type(self).hits.append(dict(path=self.path, ua=self.headers.get('User-Agent'), enc=self.headers.get('Accept-Encoding')))
        status, body = type(self).plan.pop(0) if type(self).plan else (200, b'{}')
        self.send_response(status)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    handler = type('H', (_Handler,), {'plan': [], 'hits': []})
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, handler
    httpd.shutdown()
    httpd.server_close()


def test_real_fetch_user_agent_cache_and_no_retry(server, tmp_path, monkeypatch):
    httpd, h = server
    url = f'http://127.0.0.1:{httpd.server_address[1]}/x.json'
    monkeypatch.delenv('SEC_USER_AGENT', raising=False)
    h.plan[:] = [(200, b'{"a":1}')]
    assert providers.fetch(url, tmp_path / 'c' / 'x.json') == b'{"a":1}'
    assert h.hits[0]['ua'] == 'trading-ml point-in-time research' and h.hits[0]['enc'] == 'identity'
    assert providers.fetch(url, tmp_path / 'c' / 'x.json') == b'{"a":1}' and len(h.hits) == 1          # served from cache, no second request
    monkeypatch.setenv('SEC_USER_AGENT', 'QA Audit qa@example.invalid')
    h.plan[:] = [(200, b'{"b":2}')]
    providers.fetch(url, tmp_path / 'c' / 'y.json')
    assert h.hits[-1]['ua'] == 'QA Audit qa@example.invalid'
    assert providers.fetch(url, tmp_path / 'c' / 'z.json', user_agent='explicit') or h.hits[-1]['ua'] == 'explicit'
    # LIMITATION: no retry/backoff - a 429 or 503 is raised straight away, once, and nothing is cached
    from urllib.error import HTTPError
    for status in (429, 503):
        n = len(h.hits)
        h.plan[:] = [(status, b'busy'), (200, b'{"ok":1}')]
        with pytest.raises(HTTPError):
            providers.fetch(url, tmp_path / 'c' / f'e{status}.json')
        assert len(h.hits) == n + 1 and not (tmp_path / 'c' / f'e{status}.json').exists()
        h.plan.clear()
    assert not list((tmp_path / 'c').glob('*.tmp'))                                                      # atomic write leaves no temp file


def test_real_ingest_sec_end_to_end_with_real_msft_payload(tmp_path, monkeypatch):
    """providers.ingest_sec on the REAL cached MSFT bodies served through a fake urlopen: counts, events, raw archive, rate spacing, idempotency."""
    raw = REPO / 'results' / 'fundamental_audit_20260930' / 'raw'
    bodies = {'facts': (raw / 'MSFT_facts.json').read_bytes(), 'submissions': (raw / 'MSFT_submissions.json').read_bytes()}
    calls, sleeps = [], []

    class _Resp:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.body

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return _Resp(bodies['facts'] if 'companyfacts' in req.full_url else bodies['submissions'])

    monkeypatch.setattr(providers, 'urlopen', fake_urlopen)
    monkeypatch.setattr(providers.time, 'sleep', lambda s: sleeps.append(s))
    store = PointInTimeFundamentalStore(str(tmp_path / 'm.sqlite'))
    count = providers.ingest_sec(store, 'MSFT', 789019, tmp_path / 'cache')
    assert count == 6035 and len(calls) == 2 and calls[0].endswith('CIK0000789019.json') and 'companyfacts' in calls[0] and 'submissions' in calls[1]
    assert all(s >= 0.1 for s in sleeps) and len(sleeps) == 2                    # 0.15 s spacing between SEC calls (limit is 10 req/s)
    assert store.db.execute('SELECT COUNT(*) FROM raw_documents').fetchone()[0] == 2
    import hashlib
    digests = {r[0] for r in store.db.execute('SELECT sha256 FROM raw_documents')}
    assert digests == {hashlib.sha256(b).hexdigest() for b in bodies.values()}
    recent = json.loads(bodies['submissions'])['filings']['recent']
    wanted = [i for i, f in enumerate(recent['form']) if f in ('8-K', '10-K', '10-Q', '10-K/A', '10-Q/A')]
    events = store.db.execute("SELECT * FROM corporate_events WHERE symbol='MSFT' AND event_type='filing'").fetchall()
    assert len(events) == len(wanted)
    sample = wanted[0]
    ev = [e for e in events if e['revision'] == recent['accessionNumber'][sample]][0]
    from tests.qa_reaudit.real_oracle import parse_utc
    import datetime as dt
    assert ev['available_at'] == (parse_utc(recent['acceptanceDateTime'][sample]) + dt.timedelta(minutes=5)).isoformat()
    assert json.loads(ev['payload'])['direction'] is None                       # a filing is never read as sentiment
    # idempotent: second run uses the cache (no new HTTP) and adds no rows or events
    before = (stored_count(store), len(events))
    providers.ingest_sec(store, 'MSFT', 789019, tmp_path / 'cache')
    assert len(calls) == 2
    assert (stored_count(store), store.db.execute("SELECT COUNT(*) FROM corporate_events").fetchone()[0]) == before
    store.close()


def test_real_defect_poisoned_cache_survives_provider_recovery(tmp_path, monkeypatch):
    good = (REPO / 'results' / 'fundamental_audit_20260930' / 'raw' / 'XOM_facts.json').read_bytes()
    subs = (REPO / 'results' / 'fundamental_audit_20260930' / 'raw' / 'XOM_submissions.json').read_bytes()
    state = {'broken': True}

    class _Resp:
        def __init__(self, b):
            self.b = b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.b

    def fake(req, timeout=None):
        if state['broken']:
            return _Resp(b'<html>Scheduled maintenance</html>')
        return _Resp(good if 'companyfacts' in req.full_url else subs)

    monkeypatch.setattr(providers, 'urlopen', fake)
    monkeypatch.setattr(providers.time, 'sleep', lambda s: None)
    store = PointInTimeFundamentalStore(str(tmp_path / 'x.sqlite'))
    with pytest.raises(ValueError):
        providers.ingest_sec(store, 'XOM', 34088, tmp_path / 'cache')
    state['broken'] = False
    assert providers.ingest_sec(store, 'XOM', 34088, tmp_path / 'cache') > 0


# ===================================================================================================== I. extra provider / currency facts
@pytest.mark.parametrize('currency,units', [('GBX', 'GBX'), ('GBp', 'GBp'), ('GBP', 'GBP pence'), ('GBP', 'GBX'), ('TWD', 'TWD'), ('INR', 'INR millions')])
def test_real_pence_and_unlisted_currencies_are_rejected_not_guessed(currency, units):
    """LIMITATION (fail-safe): no pence/GBX scale and no TWD/INR support; a UK quote in pence can never be mistaken for pounds."""
    with pytest.raises(ValueError):
        fact('revenue', 1.0, currency=currency, units=units).normalized()


@pytest.mark.parametrize('filed,available,reported', [
    ('2025-01-15', '2025-01-16T05:00:00+00:00', '2025-01-15T05:00:00+00:00'),     # winter: New York midnight = 05:00 UTC
    ('2025-07-15', '2025-07-16T04:00:00+00:00', '2025-07-15T04:00:00+00:00'),     # summer: 04:00 UTC
    ('2025-03-07', '2025-03-08T05:00:00+00:00', '2025-03-07T05:00:00+00:00'),     # Friday before US DST starts
    ('2025-03-10', '2025-03-11T04:00:00+00:00', '2025-03-10T04:00:00+00:00'),     # Monday after
    ('2025-10-31', '2025-11-01T04:00:00+00:00', '2025-10-31T04:00:00+00:00'),     # Friday before US DST ends
    ('2025-11-03', '2025-11-04T05:00:00+00:00', '2025-11-03T05:00:00+00:00'),     # Monday after
])
def test_real_date_only_filing_becomes_available_at_the_next_new_york_midnight(filed, available, reported):
    f = sec_facts(payload([{**ROW, 'filed': filed}]), {}, 'T')[0]
    assert f.available_at == available and f.metadata['timing_precision'] == 'following_local_day'
    # LIMITATION: reported_at for a date-only filing is 00:00 New York of the filing day, up to ~24h BEFORE the real acceptance
    assert f.reported_at == reported


def test_real_ingest_sec_partial_failure_writes_nothing_and_resumes_from_cache(tmp_path, monkeypatch):
    from urllib.error import HTTPError
    raw = REPO / 'results' / 'fundamental_audit_20260930' / 'raw'
    facts, subs = (raw / 'XOM_facts.json').read_bytes(), (raw / 'XOM_submissions.json').read_bytes()
    state, calls = {'fail': True}, []

    class _Resp:
        def __init__(self, b):
            self.b = b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.b

    def fake(req, timeout=None):
        calls.append(req.full_url)
        if 'companyfacts' in req.full_url:
            return _Resp(facts)
        if state['fail']:
            raise HTTPError(req.full_url, 503, 'busy', {}, None)
        return _Resp(subs)

    monkeypatch.setattr(providers, 'urlopen', fake)
    monkeypatch.setattr(providers.time, 'sleep', lambda s: None)
    store = PointInTimeFundamentalStore(str(tmp_path / 'p.sqlite'))
    with pytest.raises(HTTPError):
        providers.ingest_sec(store, 'XOM', 34088, tmp_path / 'c')
    assert stored_count(store) == 0 and sorted(p.name for p in (tmp_path / 'c').iterdir()) == ['XOM_facts.json']
    state['fail'] = False
    assert providers.ingest_sec(store, 'XOM', 34088, tmp_path / 'c') == 4020
    assert len(calls) == 3                                    # the cached facts body is not downloaded again
    store.close()


def test_real_ingest_alfred_url_parameters_vintage_and_stored_fields(tmp_path, monkeypatch):
    body = b'observation_date,CPIAUCSL_20240601\n2021-06-01,271.0\n2024-04-01,313.2\n2024-05-01,314.1\n'
    calls = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return body

    monkeypatch.setattr(providers, 'urlopen', lambda req, timeout=None: (calls.append(req.full_url), _Resp())[1])
    store = mem()
    assert providers.ingest_alfred(store, 'CPIAUCSL', '2024-06-01', tmp_path) == 3
    assert calls == ['https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=CPIAUCSL&cosd=2021-06-01&coed=2024-06-01&vintage_date=2024-06-01']
    rows = store.db.execute('SELECT value,period_end,reported_at,available_at,units,currency,kind FROM observations ORDER BY period_end').fetchall()
    assert [tuple(r) for r in rows][1] == (313.2, '2024-04-01', '2024-06-01T00:00:00+00:00', '2024-06-02T00:00:00+00:00', 'index', '', 'macro')
    providers.ingest_alfred(store, 'CPIAUCSL', '2024-06-01', tmp_path)
    assert len(calls) == 1 and stored_count(store) == 3        # cache hit, idempotent
    assert store.as_of('MACRO:CPIAUCSL', '2024-06-01T23:59:59+00:00', kind='macro') == []      # not visible during the vintage day itself
    assert len(store.as_of('MACRO:CPIAUCSL', '2024-06-02T00:00:00+00:00', kind='macro')) == 3


def test_real_ten_month_transition_period_is_accepted_as_annual():
    """LIMITATION: the 300-380 day window treats a 304-day stub (fiscal-year change filed on a 10-K) as the latest 'annual' period."""
    store = mem()
    for metric, v in (('revenue', 1200.0), ('net_income', 120.0)):
        store.ingest([fact(metric, v, period_start='2022-01-01', period_end='2022-12-31', avail='2023-03-01T12:00:00+00:00', revision='a')])
    for metric, v in (('revenue', 900.0), ('net_income', 90.0)):
        store.ingest([fact(metric, v, period_start='2023-01-01', period_end='2023-10-31', avail='2023-12-15T12:00:00+00:00', revision='b')])
    snap = FundamentalFeatureEngine(store).snapshot('T', '2023-12-20T00:00:00+00:00')
    assert (snap['period_start'], snap['period_end']) == ('2023-01-01', '2023-10-31') and snap['raw']['revenue'] == 900.0
    assert snap['features']['revenue_yoy'] is None and snap['status'] == 'AVAILABLE'      # no growth fabricated, but no stub flag either


def test_real_redownload_with_new_observed_at_creates_no_duplicate_economic_fact():
    """Same SEC rows re-ingested with a different local observation clock: identity ignores observed_at -> exactly one stored row."""
    store = mem()
    for observed in ('2026-01-01T00:00:00+00:00', '2026-06-01T12:34:56+00:00'):
        facts = sec_facts(payload([ROW]), {}, 'T', observed_at=observed)
        assert store.ingest_validated(facts)['accepted'] == 1
    assert stored_count(store) == 1
    assert store.db.execute('SELECT observed_at FROM observations').fetchone()[0] == '2026-01-01T00:00:00+00:00'      # first observation kept
