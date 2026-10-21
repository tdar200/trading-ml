"""TRACK F: HTTP API (fundamentals.api.serve) black-box audit against a fixture DB and an independent oracle.

Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_ui_api.py
Servers bind 127.0.0.1 ports in 8850-8899 and are stopped in fixtures' finalizers.
"""
import hashlib
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from decimal import Decimal as D
from urllib.parse import quote

import pytest

from tests.qa_reaudit import ui_fixture as fx
from tests.qa_reaudit import ui_harness as h
from tests.qa_reaudit import ui_oracle as o

T_B, T_0, T_A = '2025-03-03T06:59:59Z', '2025-03-03T07:00:00Z', '2025-03-03T07:01:00Z'
T_KA_B, T_KA = '2025-04-15T06:59:59Z', '2025-04-15T07:00:00Z'
ISO_UTC = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,9})?\+00:00$')
DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
TS_KEYS = {'as_of', 'timestamp', 'reported_at', 'available_at', 'observed_at', 'event_at', 'detected_at', 'trained_through', 'calibrated_at'}
CANARY_TXT = ('123456789123', '1.23456789123e+11')


@pytest.fixture(scope='module')
def main_db(tmp_path_factory):
    return fx.build_main_db(tmp_path_factory.mktemp('uiapi') / 'main.sqlite')


@pytest.fixture(scope='module')
def truth():
    return fx.all_truth()


@pytest.fixture(scope='module')
def port(main_db):
    with h.api_server(main_db) as p:
        yield p


@pytest.fixture(scope='module')
def model_port(main_db):
    with h.api_server(main_db, score=fx.frozen_engine()) as p:
        yield p


def J(port, path, expect=200):
    st, hd, body = h.http(port, path)
    assert st == expect, (path, st, body[:300])
    return h.strict_loads(body)


def walk(x, path=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from walk(v, f'{path}/{k}')
            if isinstance(v, str):
                yield (k, v, f'{path}/{k}')
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from walk(v, f'{path}[{i}]')


ENDPOINTS = ['/health', '/macro/regime', '/watchlist/fundamental', '/fundamentals/ALPHA', '/fundamentals/ALPHA/history', '/fundamentals/ALPHA/score',
             '/fundamentals/ALPHA/events', '/fundamentals/ALPHA/ttm', '/fundamentals/ALPHA/quality',
             '/strategies/demo/explanation?symbol=ALPHA&asset_class=equity&technical_signal=true']


def with_asof(path, asof):
    if asof is None:
        return path
    sep = '&' if '?' in path else '?'
    return f'{path}{sep}as_of={quote(asof, safe="")}'


# ============================================================================ schema / transport
@pytest.mark.parametrize('asof', [None, T_B, T_A, '2026-03-03T00:00:00Z'])
@pytest.mark.parametrize('path', ENDPOINTS)
def test_every_endpoint_status_headers_strict_json_and_timestamp_format(port, path, asof):
    st, hd, body = h.http(port, with_asof(path, asof))
    assert st == 200, body[:200]
    assert hd['content-type'] == 'application/json'
    assert hd['cache-control'] == 'no-store'
    assert int(hd['content-length']) == len(body)
    doc = h.strict_loads(body)                      # NaN/Infinity tokens -> failure
    assert not re.search(rb'\b(NaN|-?Infinity)\b', re.sub(rb'"(?:[^"\\]|\\.)*"', b'""', body))
    for key, val, where in walk(doc):
        if key in TS_KEYS:
            assert ISO_UTC.match(val), (where, val)
        if key in ('period_end', 'period_start') and val:
            assert DATE.match(val), (where, val)


def test_snapshot_schema_types(port):
    s = J(port, with_asof('/fundamentals/ALPHA', T_A))
    assert {'symbol', 'as_of', 'version', 'asset_class', 'features', 'raw', 'lineage', 'dimensions', 'flags', 'status', 'score', 'confidence',
            'completeness', 'sector', 'ttm', 'period_end', 'period_start', 'currency', 'reported_at', 'available_at', 'age_days'} <= set(s)
    assert s['symbol'] == 'ALPHA' and s['version'] == 'fundamentals-v4-audited' and s['asset_class'] == 'equity'
    assert all(isinstance(v, (float, type(None))) and not isinstance(v, bool) for v in s['features'].values())
    assert all(isinstance(v, (float, int)) and not isinstance(v, bool) for v in s['raw'].values())
    assert isinstance(s['flags'], list) and all(isinstance(x, str) for x in s['flags'])
    assert isinstance(s['age_days'], int) and isinstance(s['completeness'], float) and 0 <= s['completeness'] <= 1
    assert all(re.fullmatch(r'[0-9a-f]{64}', x) for x in s['lineage'])
    assert set(s['dimensions']) == set(s['features'])
    assert s['score'] is None and s['confidence'] is None        # no approved model: never a fabricated number
    assert s['sector'] == 'software' and s['currency'] == 'USD'
    n = sum(v is not None for v in s['features'].values())
    assert s['completeness'] == n / len(s['features'])           # completeness agrees with its own feature table


def test_health_payload(port):
    assert J(port, '/health') == {'status': 'OK', 'read_only': True, 'live_trading': False}


# ============================================================================ numeric reconciliation vs oracle
SNAP_TIMES = ['2022-02-01T00:00:00Z', '2023-06-01T00:00:00Z', '2024-02-29T23:59:59Z', '2024-03-01T07:00:00Z', '2024-12-01T00:00:00Z', T_B, T_0, T_A, T_KA_B, T_KA,
              '2025-07-04T00:00:00Z', '2025-07-05T00:00:00Z', '2025-07-21T00:00:00Z', '2026-03-01T00:00:00Z']


@pytest.mark.parametrize('symbol', ['ALPHA', 'BETA', 'GAMMA', 'DELTA', 'LATE', 'LATE2', 'GIANT', 'NEGEQ', 'FLATCO', 'XSSCO'])
def test_snapshot_matches_independent_oracle_at_every_boundary(port, truth, symbol):
    bad = []
    for T in SNAP_TIMES:
        api = J(port, with_asof(f'/fundamentals/{symbol}', T))
        orc = o.snapshot(truth[symbol], T)
        for m in o.diff_snapshot(api, orc):
            bad.append(f'{symbol}@{T}: {m}')
    assert not bad, '\n'.join(bad)


def test_alpha_golden_numbers_by_hand(port):
    """Hand derivation (millions): FY2024 rev 120, GP 48, OCF 18, CapEx 6, restated FY2023 rev 98 (orig 100)."""
    s = J(port, with_asof('/fundamentals/ALPHA', T_A))
    f = s['features']
    assert s['period_end'] == '2024-12-31' and s['age_days'] == 62
    assert f['fcf'] == 12_000_000.0
    assert f['gross_margin'] == pytest.approx(0.4, abs=1e-15)
    assert f['revenue_yoy'] == pytest.approx(120 / 98 - 1, rel=1e-13)            # 22.4489795918...%, restated base
    assert f['debt_equity'] == pytest.approx(48 / 112, rel=1e-13)
    assert f['current_ratio'] == pytest.approx(2.0, rel=1e-13)
    # ROIC = NOPAT / avg invested capital; NOPAT = 24*(1-3/15)=19.2; invested 2024 = 10+38+112-24 = 136; 2023 = 10+40+100-20 = 130
    assert f['roic'] == pytest.approx(19.2 / ((136 + 130) / 2), rel=1e-13)
    assert s['raw']['eps'] == 1.2 and s['raw']['revenue'] == 120_000_000.0


def test_snapshot_json_has_full_double_precision_not_rounded(port):
    st, hd, body = h.http(port, with_asof('/fundamentals/ALPHA', T_A))
    txt = body.decode()
    assert '"revenue_yoy": 0.22448979591836737' in txt or re.search(r'"revenue_yoy": 0\.224489795918367\d+', txt)
    m = re.search(r'"debt_equity": ([0-9.e+-]+)', txt)
    assert len(m.group(1).replace('.', '').lstrip('0')) >= 15


def test_scaled_units_end_to_end_gbp_millions_and_usd_billions(port, truth):
    """DELTA stored as 'GBP millions', GIANT as 'USD billions': API must return base-currency units."""
    d = J(port, with_asof('/fundamentals/DELTA', '2025-04-01T00:00:00Z'))
    assert d['currency'] == 'GBP'
    assert d['raw']['revenue'] == pytest.approx(1_589_750_000, rel=1e-14)
    assert d['features']['fcf'] == pytest.approx(260_200_000 - 101_100_000, rel=1e-12)
    g = J(port, with_asof('/fundamentals/GIANT', '2025-04-01T00:00:00Z'))
    assert g['raw']['revenue'] == 6.8e11 and g['currency'] == 'USD'
    assert g['features']['fcf'] == pytest.approx(680e9 * 0.22 - 12e9 - 35e9, rel=1e-12)


# ============================================================================ PIT behaviour through HTTP
def period(port, T, symbol='ALPHA'):
    return J(port, with_asof(f'/fundamentals/{symbol}', T))['period_end']


def test_boundary_second_resolution_and_offsets(port):
    assert period(port, '2025-03-03T06:59:59Z') == '2023-12-31'
    assert period(port, '2025-03-03T06:59:59.999999Z') == '2023-12-31'
    assert period(port, '2025-03-03T06:59:59.999999999Z') == '2023-12-31'
    assert period(port, '2025-03-03T07:00:00Z') == '2024-12-31'
    assert period(port, '2025-03-03T07:00:00.000000001Z') == '2024-12-31'
    assert period(port, '2025-03-03T07:01:00Z') == '2024-12-31'
    # same instants expressed with offsets (must be percent-encoded: '+' decodes to a space in query strings)
    assert period(port, '2025-03-03T02:59:59-04:00') == '2023-12-31'      # 06:59:59Z
    assert period(port, '2025-03-03T03:00:00-04:00') == '2024-12-31'      # 07:00:00Z
    assert period(port, '2025-03-03T07:59:59+01:00') == '2023-12-31'      # 06:59:59Z
    assert period(port, '2025-03-03T08:00:00+01:00') == '2024-12-31'      # 07:00:00Z
    assert period(port, '2025-03-03T12:30:00+05:30') == '2024-12-31'      # 07:00:00Z


@pytest.mark.parametrize('day,tzname', [('2025-03-09', 'US spring-forward'), ('2025-11-02', 'US fall-back'), ('2025-03-30', 'UK spring-forward'), ('2025-10-26', 'UK fall-back')])
def test_dst_transition_days_use_absolute_instants(port, day, tzname):
    """All timestamps are absolute UTC instants; DST days must not shift the visible-fact set."""
    for T in (f'{day}T00:00:00Z', f'{day}T12:00:00Z', f'{day}T23:59:59Z'):
        api = J(port, with_asof('/fundamentals/ALPHA', T))
        assert not o.diff_snapshot(api, o.snapshot(fx.alpha_truth(), T)), (tzname, T)
    # a wall-clock offset that exists only on the DST day
    api = J(port, with_asof('/fundamentals/ALPHA', f'{day}T03:30:00-04:00'))
    assert api['as_of'].endswith('+00:00') and api['as_of'].startswith(f'{day}T07:30:00')


def test_future_canary_never_in_any_endpoint_before_publication_and_visible_after(port):
    before = '2026-03-02T06:59:59Z'
    for path in ENDPOINTS + ['/fundamentals/ALPHA/history?metric=revenue', '/fundamentals/ALPHA/history?metric=eps', '/fundamentals/ALPHA/history?metric=diluted_shares']:
        st, hd, body = h.http(port, with_asof(path, before))
        assert st == 200
        txt = body.decode()
        assert not any(c in txt for c in CANARY_TXT), path
    # positive control: after publication the canary is public and must appear (proves the scan can see it)
    st, hd, body = h.http(port, with_asof('/fundamentals/ALPHA/history?metric=revenue', '2026-03-02T07:00:00Z'))
    assert '123456789123' in body.decode()
    snap = J(port, with_asof('/fundamentals/ALPHA', '2026-03-02T07:00:00Z'))
    assert snap['raw']['revenue'] == 123456789123.0 and snap['period_end'] == '2025-12-31'


def test_restatement_visibility_and_audit_trail_via_history(port):
    pre = J(port, with_asof('/fundamentals/ALPHA/history?metric=eps', T_KA_B))['records']
    post = J(port, with_asof('/fundamentals/ALPHA/history?metric=eps', T_KA))['records']
    fy24 = lambda rows: sorted((r['value'], r['revision']) for r in rows if r['period_end'] == '2024-12-31' and r['period_start'] == '2024-01-01')
    assert fy24(pre) == [(1.2, '0001-2024-10K')]
    assert fy24(post) == [(1.05, '0001-2024-10KA'), (1.2, '0001-2024-10K')]      # original retained: append-only audit trail
    # as-of view keeps only the latest version
    assert J(port, with_asof('/fundamentals/ALPHA', T_KA_B))['raw']['eps'] == 1.2
    assert J(port, with_asof('/fundamentals/ALPHA', T_KA))['raw']['eps'] == 1.05
    assert J(port, with_asof('/fundamentals/ALPHA', T_KA))['raw']['net_income'] == 10_500_000.0
    # restated FY2023 revenue only inside the later filing
    assert J(port, with_asof('/fundamentals/ALPHA', T_B))['features']['revenue_yoy'] == pytest.approx(100 / 90 - 1, rel=1e-13)
    assert J(port, with_asof('/fundamentals/ALPHA', T_0))['features']['revenue_yoy'] == pytest.approx(120 / 98 - 1, rel=1e-13)


@pytest.mark.parametrize('T', ['2023-01-01T00:00:00Z', '2024-03-01T06:59:59Z', '2024-03-01T07:00:00Z', T_B, T_0, T_KA, '2026-03-02T07:00:00Z', '2030-01-01T00:00:00Z'])
@pytest.mark.parametrize('metric', [None, 'revenue', 'eps'])
def test_history_equals_independent_sql_with_python_datetime_filter(port, main_db, T, metric):
    """Differential check of the string-compare SQL against Python datetime semantics on the raw SQLite file."""
    con = sqlite3.connect(f'file:{main_db}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute('SELECT * FROM observations WHERE symbol=?', ('ALPHA',))]
    con.close()
    want = {r['id'] for r in rows if o.ts(r['available_at']) <= o.ts(T) and (metric is None or r['metric'] == metric)}
    q = '/fundamentals/ALPHA/history' + (f'?metric={metric}' if metric else '')
    got = J(port, with_asof(q, T))['records']
    assert {r['id'] for r in got} == want
    assert all(o.ts(r['available_at']) <= o.ts(T) for r in got)
    byid = {r['id']: r for r in rows}
    for r in got:
        assert r['value'] == byid[r['id']]['value'] and r['units'] == byid[r['id']]['units'] and r['currency'] == byid[r['id']]['currency']


def test_history_rows_expose_currency_units_and_source_lineage(port):
    recs = J(port, with_asof('/fundamentals/DELTA/history?metric=revenue', '2025-04-01T00:00:00Z'))['records']
    assert len(recs) == 3
    for r in recs:
        assert r['currency'] == 'GBP' and r['units'] == 'GBP'                  # normalised from 'GBP millions'
        assert json.loads(r['metadata'])['original_units'] == 'GBP millions'
        assert json.loads(r['metadata'])['unit_multiplier'] == 1_000_000
    assert recs[0]['value'] == pytest.approx(1_250_500_000.0)


@pytest.mark.parametrize('symbol', ['ALPHA', 'BETA', 'GAMMA', 'DELTA', 'LATE', 'LATE2'])
def test_every_lineage_id_resolves_to_a_row_public_at_the_requested_time(port, main_db, symbol):
    """Source lineage must itself be point-in-time: no id may point at a row published after as_of (checked against raw SQLite)."""
    con = sqlite3.connect(f'file:{main_db}?mode=ro', uri=True)
    avail = {r[0]: r[1] for r in con.execute('SELECT id, available_at FROM observations')}
    con.close()
    for T in SNAP_TIMES:
        snap = J(port, with_asof(f'/fundamentals/{symbol}', T))
        ids = list(snap['lineage']) + list(snap.get('ttm', {}).get('lineage', []))
        for i in ids:
            assert i in avail, (symbol, T, i)
            assert o.ts(avail[i]) <= o.ts(T), (symbol, T, i, avail[i])
        if snap['status'] != 'MISSING':
            assert o.ts(snap['reported_at']) <= o.ts(snap['available_at']) <= o.ts(T), (symbol, T)
    macro = J(port, with_asof('/macro/regime', '2025-03-20T00:00:00Z'))
    con = sqlite3.connect(f'file:{main_db}?mode=ro', uri=True)
    avail = {r[0]: r[1] for r in con.execute('SELECT id, available_at FROM observations')}
    con.close()
    assert macro['lineage'] and all(o.ts(avail[i]) <= o.ts('2025-03-20T00:00:00Z') for i in macro['lineage'])


# ============================================================================ TTM / macro / events / score
TTM_TIMES = ['2024-03-01T07:00:00Z', '2024-06-01T00:00:00Z', '2024-12-01T00:00:00Z', T_B, T_0, T_KA_B, T_KA, '2025-12-31T00:00:00Z']


@pytest.mark.parametrize('T', TTM_TIMES)
def test_ttm_alpha_ytd_differencing_matches_analyst_formula(port, truth, T):
    api = J(port, with_asof('/fundamentals/ALPHA/ttm', T))
    vals, end = o.ttm_alpha(truth['ALPHA'], T)
    for k, v in vals.items():
        assert o.close(api['values'][k], v), (T, k, api['values'][k], v)
    assert api['period_end'] == end
    assert api['values']['eps'] is None and api['status'] == 'INCOMPLETE'        # no discrete Q4 EPS is published: never invented
    assert api['reasons']['eps'] in ('MISSING_QUARTERS', 'NONCONTIGUOUS_QUARTERS')


@pytest.mark.parametrize('T', ['2024-02-03T21:00:00Z', '2024-05-03T21:00:00Z', '2024-06-01T00:00:00Z', '2024-11-15T20:59:59Z', '2024-11-15T21:00:00Z', '2025-01-01T00:00:00Z'])
def test_ttm_beta_sum_of_four_discrete_quarters(port, truth, T):
    api = J(port, with_asof('/fundamentals/BETA/ttm', T))
    res = o.ttm_sum_quarters(truth['BETA'], T)
    vals, end, start = res
    for k, v in vals.items():
        assert o.close(api['values'][k], v), (T, k)
    assert (api['period_start'], api['period_end']) == (start, end)
    assert api['status'] == 'AVAILABLE' and api['values']['fcf'] == pytest.approx(api['values']['ocf'] - api['values']['capex'])


def test_ttm_beta_after_hours_filing_boundary(port):
    """FY2024 Q4 + FY filed 2024-11-15 21:00Z (4pm ET): TTM rolls only at the publication instant."""
    assert J(port, with_asof('/fundamentals/BETA/ttm', '2024-11-15T20:59:59Z'))['period_end'] == '2024-06-29'
    assert J(port, with_asof('/fundamentals/BETA/ttm', '2024-11-15T21:00:00Z'))['period_end'] == '2024-09-28'


def test_ttm_missing_symbol_is_empty_not_zero(port):
    t = J(port, '/fundamentals/NOSUCH/ttm')
    assert t['status'] == 'INCOMPLETE' and all(v is None for v in t['values'].values()) and t['period_end'] is None


@pytest.mark.parametrize('T', ['2025-01-15T23:59:59Z', '2025-01-16T00:00:00Z', '2025-02-01T00:00:00Z', '2025-03-14T23:59:59Z', '2025-03-15T00:00:00Z', '2025-06-23T00:00:00Z',
                               '2025-06-24T00:00:00Z', '2026-01-01T00:00:00Z'])
def test_macro_regime_vintage_boundaries_vs_oracle(port, T):
    api = J(port, with_asof('/macro/regime', T))
    orc = o.macro_state(T)
    assert api['status'] == orc['status'], (T, api['status'], orc['status'])
    for k, v in orc['features'].items():
        assert o.close(api['features'].get(k), v), (T, k, api['features'].get(k), v)
    assert set(api['features']) == set(orc['features'])
    assert api['score'] is None and api['confidence'] is None             # no invented composite


def test_macro_revision_visible_only_from_next_utc_day_of_the_vintage(port):
    """Vintage 2025-03-14 revises Dec/Jan CPI; ALFRED has no release time so availability = vintage day + 1 (00:00 UTC)."""
    a = J(port, with_asof('/macro/regime', '2025-03-14T23:59:59Z'))
    b = J(port, with_asof('/macro/regime', '2025-03-15T00:00:00Z'))
    assert a['features']['policy_rate'] == 4.5 and b['features']['policy_rate'] == 4.375        # Feb 2025 obs only exists in the new vintage
    assert a['features']['unemployment'] == pytest.approx(4.65) and b['features']['unemployment'] == pytest.approx(4.75)
    assert a['regime'] == {'inflation': 'FALLING', 'labour_market': 'WEAKENING', 'policy': 'EASING'}


def test_macro_status_labels_not_confused(port):
    assert J(port, with_asof('/macro/regime', '2024-06-01T00:00:00Z'))['status'] == 'INCOMPLETE'     # nothing published yet
    assert J(port, with_asof('/macro/regime', '2026-06-01T00:00:00Z'))['status'] == 'STALE'          # all vintages too old
    assert J(port, with_asof('/macro/regime', '2026-06-01T00:00:00Z'))['features'] == {}             # stale values are not shown


def test_events_visibility_surprise_and_wildcard(port):
    pre = J(port, with_asof('/fundamentals/ALPHA/events', '2025-03-03T06:59:59Z'))
    post = J(port, with_asof('/fundamentals/ALPHA/events', T_0))
    assert [e['event_type'] for e in pre['events']] == ['macro_calendar']             # '*' event announced 2025-01-10, event_at in the future
    assert pre['events'][0]['symbol'] == '*' and pre['events'][0]['event_at'] > '2025-03-03T06:59:59'
    assert 'as_of' not in pre and 'symbol' not in pre            # schema gap: /events echoes neither the symbol nor the as_of it answered
    earn = [e for e in post['events'] if e['event_type'] == 'earnings'][0]
    assert earn['eps_surprise'] == pytest.approx((1.10 - 1.00) / 1.00, abs=1e-12)
    assert earn['age_days'] == 0.0 and earn['decay'] is None and post['status'] == 'DESCRIPTIVE_ONLY'
    before_cal = J(port, with_asof('/fundamentals/ALPHA/events', '2025-01-09T23:59:59Z'))
    assert before_cal['events'] == [] and before_cal['status'] == 'MISSING'
    # age is measured from available_at, not event_at
    cal = J(port, with_asof('/fundamentals/ALPHA/events', '2025-01-11T00:00:00Z'))['events'][0]
    assert cal['age_days'] == pytest.approx(1.0)


def test_score_endpoint_without_model_is_null_unvalidated(port):
    s = J(port, with_asof('/fundamentals/ALPHA/score', T_A))
    assert s['score'] is None and s['status'] == 'UNVALIDATED' and s['components'] == {} and s['deployment_approved'] is False


def test_score_endpoint_with_frozen_model_equals_hand_arithmetic(model_port):
    s = J(model_port, with_asof('/fundamentals/ALPHA/score', T_A))
    f = {'revenue_yoy': D(120) / D(98) - 1, 'gross_margin': D('0.4'), 'fcf_margin': D('0.1')}
    expect, contrib = o.model_score(f)
    assert s['status'] == 'RESEARCH_MODEL' and o.close(s['score'], expect, 1e-12)
    assert o.close(sum(s['components'].values()) + s['intercept'], s['score'], 1e-13)      # contributions sum to the score
    for k, v in contrib.items():
        assert o.close(s['components'][k], v, 1e-10, 1e-15)
    assert s['deployment_approved'] is False and 'not probability' in s['score_units']
    # missing feature -> no score (never silently improved); NEGEQ has no revenue_yoy (zero prior revenue)
    n = J(model_port, with_asof('/fundamentals/NEGEQ/score', '2025-04-01T00:00:00Z'))
    assert n['score'] is None and n['status'] == 'MISSING_FEATURES' and n['missing'] == ['revenue_yoy']
    # a model trained through 2024-01-01 is not usable at or before that instant
    st, hd, b = h.http(model_port, with_asof('/fundamentals/ALPHA/score', '2024-01-01T00:00:00Z'))
    assert st == 400 and b'model unavailable at historical timestamp' in b
    assert J(model_port, with_asof('/fundamentals/ALPHA/score', '2024-01-01T00:00:01Z'))['status'] == 'RESEARCH_MODEL'


def test_watchlist_without_model_is_unranked_and_pit(port):
    w = J(port, with_asof('/watchlist/fundamental', T_A))
    syms = [r['symbol'] for r in w['candidates']]
    assert w['ranked'] is False and w['deployment_approved'] is False
    assert syms == sorted(syms)                                                  # deterministic tie order, no fake ranking
    assert all(r['score'] is None and r['reason'] == 'UNVALIDATED' for r in w['candidates'])
    assert not any(s.startswith('MACRO:') for s in syms)
    assert 'LATE2' in syms and '<B>BAD</B>' not in syms and 'XSSCO' in syms       # BAD/XSSCO FY2024 published 2025-03-05
    later = [r['symbol'] for r in J(port, with_asof('/watchlist/fundamental', '2025-04-01T00:00:00Z'))['candidates']]
    assert '<B>BAD</B>' in later
    early = J(port, with_asof('/watchlist/fundamental', '2023-06-01T00:00:00Z'))
    assert {r['symbol'] for r in early['candidates']} == {'ALPHA', 'BETA', 'DELTA', 'GAMMA', 'LATE', 'LATE2'}
    assert J(port, with_asof('/watchlist/fundamental', '2020-01-01T00:00:00Z'))['candidates'] == []


def test_watchlist_with_model_ranked_desc_none_last_and_coverage_consistent(model_port):
    w = J(model_port, with_asof('/watchlist/fundamental', T_A))
    scores = [r['score'] for r in w['candidates']]
    assert w['ranked'] is True
    numeric = [s for s in scores if s is not None]
    assert numeric == sorted(numeric, reverse=True)
    assert all(s is None for s in scores[len(numeric):])                         # missing scores sink to the bottom, never as zeros
    for r in w['candidates']:
        snap = J(model_port, with_asof(f"/fundamentals/{quote(r['symbol'], safe='')}", T_A)) if re.fullmatch(r'[A-Za-z0-9_.:-]{1,32}', r['symbol']) else None
        if snap:
            assert r['coverage'] == snap['completeness'] and r['data_status'] == snap['status'] and r['flags'] == snap['flags']
    by = {r['symbol']: r for r in w['candidates']}
    assert by['NEGEQ']['score'] is None and by['NEGEQ']['reason'] == 'MISSING_FEATURES'
    exp = o.model_score({'revenue_yoy': D(120) / D(98) - 1, 'gross_margin': D('0.4'), 'fcf_margin': D('0.1')})[0]
    assert o.close(by['ALPHA']['score'], exp, 1e-12)


def test_quality_endpoint_reports_quarantine_and_declares_it_not_historical(port):
    q = J(port, '/fundamentals/GAMMA/quality')
    assert q['status'] == 'QUARANTINED_RECORDS' and len(q['rejections']) == 1
    assert 'canonical capex must be positive' in q['rejections'][0]['reason']
    assert q['accuracy_certified'] is False and 'not historical signal input' in q['time_scope']
    assert J(port, '/fundamentals/ALPHA/quality')['status'] == 'NO_RECORDED_REJECTIONS'
    # as_of is accepted but (documented) ignored
    assert J(port, with_asof('/fundamentals/GAMMA/quality', '2000-01-01T00:00:00Z')) == q


def test_strategy_explanation_agrees_with_machine_state(port):
    base = '/strategies/demo/explanation?symbol=GAMMA&asset_class=equity&as_of=' + quote(T_A, safe='')
    off = J(port, base)
    on = J(port, base + '&technical_signal=true')
    snap = J(port, with_asof('/fundamentals/GAMMA', T_A))
    assert 'NEGATIVE_FCF' in snap['flags']
    for d in (off, on):
        assert d['fundamental']['flags'] == snap['flags'] and d['risk']['flags'] == snap['flags'] and d['risk']['status'] == 'NOT_EVALUATED'
        assert d['deployment_approved'] is False and d['positionSize'] is None and d['policy'] == 'observe'
        assert d['fundamental']['features'] == snap['features']
    assert off['permit_signal'] is False and off['decision'] == 'NO_TRADE'
    assert on['permit_signal'] is True and on['decision'] == 'TECHNICAL_CANDIDATE'
    assert on['explanation'] == ['Fundamental observation only; technical decision unchanged']
    assert off['data_snapshot_id'] == on['data_snapshot_id']            # snapshot id depends on data, not on the technical flag
    assert off['configuration']['model'] is None and off['fundamental']['status'] == 'UNVALIDATED'
    assert J(port, base)['data_snapshot_id'] == off['data_snapshot_id']  # reproducible


def test_decision_snapshot_id_changes_exactly_when_visible_data_changes(port):
    ids = {T: J(port, f'/strategies/x/explanation?symbol=ALPHA&asset_class=equity&as_of={quote(T, safe="")}')['data_snapshot_id']
           for T in (T_B, T_0, '2025-03-04T00:00:00Z', T_KA_B, T_KA)}
    assert ids[T_B] != ids[T_0]                                       # filing published at 07:00:00
    assert ids[T_KA_B] != ids[T_KA]                                   # 10-K/A restatement


def test_etf_and_other_asset_classes_fail_closed_no_numbers(port):
    for ac, status in (('etf', 'REQUIRES_DATED_HOLDINGS'), ('forex', 'REQUIRES_MACRO_MODEL'), ('crypto', 'REQUIRES_NATIVE_DATA'), ('commodity', 'REQUIRES_ASSET_ADAPTER')):
        s = J(port, with_asof('/fundamentals/ALPHA', T_A) + '&asset_class=' + ac)
        assert s['status'] == status and s['features'] == {} and s['completeness'] == 0.0 and s['score'] is None


def test_unknown_symbol_returns_missing_not_fabricated(port):
    s = J(port, with_asof('/fundamentals/NOSUCH', T_A))
    assert s['status'] == 'MISSING' and s['features'] == {} and s['score'] is None and s['completeness'] == 0.0
    assert J(port, with_asof('/fundamentals/NOSUCH/history', T_A))['records'] == []


# ============================================================================ errors, injection, methods, concurrency
@pytest.mark.parametrize('path,status', [
    ('/nope', 404), ('/', 404), ('/fundamentals', 404), ('/fundamentals/', 404), ('/fundamentals/ALPHA/nonsense', 404), ('/FUNDAMENTALS/ALPHA', 404), ('/macro', 404),
    ('/watchlist', 404), ('/health/x', 404), ('/strategies/x', 404), ('/strategies/x/other', 404), ('/../../etc/passwd', 404), ('/fundamentals/%2e%2e/%2e%2e/etc/passwd', 404),
    ('/fundamentals/A%20B', 400), ("/fundamentals/ALPHA'%20OR%201=1--", 400), ('/fundamentals/..%2f..%2fetc%2fpasswd', 400), ('/fundamentals/' + 'A' * 33, 400),
    ('/fundamentals/ALPHA%00', 400), ('/fundamentals/ALP%C3%81', 400), ('/fundamentals/%3Cscript%3E', 400),
    ('/fundamentals/ALPHA?as_of=garbage', 400), ('/fundamentals/ALPHA?as_of=2025-03-03', 400), ('/fundamentals/ALPHA?as_of=2025-03-03T07:00:00', 400),
    ('/fundamentals/ALPHA?as_of=NaT', 400), ('/fundamentals/ALPHA?as_of=2025-02-30T00:00:00Z', 400), ('/fundamentals/ALPHA?as_of=99999-01-01T00:00:00Z', 400),
    ('/fundamentals/ALPHA?as_of=1700000000', 400), ('/macro/regime?as_of=bad', 400), ('/watchlist/fundamental?as_of=bad', 400), ('/health?as_of=bad', 400),
    ('/fundamentals/ALPHA/events?as_of=bad', 400), ('/fundamentals/ALPHA/ttm?as_of=2025-03-03', 400), ('/fundamentals/ALPHA/history?as_of=2025', 400),
])
def test_status_code_matrix_and_json_error_bodies(port, path, status):
    st, hd, body = h.http(port, path)
    assert st == status, (path, st, body[:200])
    doc = h.strict_loads(body)
    assert set(doc) == {'error'} and isinstance(doc['error'], str) and 'Traceback' not in body.decode() and '/home/' not in body.decode()
    assert hd['cache-control'] == 'no-store'


def test_unencoded_plus_offset_is_rejected_not_misread(port):
    """'+' in a query string decodes to a space: '2025-03-03T07:00:00+00:00' becomes a naive string and is refused (fail-safe)."""
    st, hd, body = h.http(port, '/fundamentals/ALPHA?as_of=2025-03-03T07:00:00+00:00')
    assert st == 400 and b'explicit timezone required' in body


INJECTIONS = ["' OR 1=1--", "'; DROP TABLE observations;--", '" OR ""="', "1; ATTACH DATABASE '/tmp/x.db' AS x;--", '%', '_', '\\', 'x UNION SELECT sql FROM sqlite_master--',
              '../../etc/passwd', '..\\..\\windows\\system32', '${jndi:ldap://x/a}', '<script>alert(1)</script>', '\x00', 'A' * 5000, '\U0001f4a5', '‮', '%00', '*', ' ', '\t', '\r\nX-Injected: 1']


@pytest.mark.parametrize('param', ['symbol_path', 'as_of', 'metric', 'asset_class', 'x_symbol', 'technical_signal'])
def test_injection_and_garbage_in_every_parameter_is_inert(port, main_db, param):
    sha = hashlib.sha256(main_db.read_bytes()).hexdigest()
    for payload in INJECTIONS:
        q = quote(payload, safe='')
        path = {'symbol_path': f'/fundamentals/{q}', 'as_of': f'/fundamentals/ALPHA?as_of={q}', 'metric': f'/fundamentals/ALPHA/history?metric={q}',
                'asset_class': f'/fundamentals/ALPHA?as_of={quote(T_A, safe="")}&asset_class={q}',
                'x_symbol': f'/strategies/x/explanation?symbol={q}&as_of={quote(T_A, safe="")}',
                'technical_signal': f'/strategies/x/explanation?symbol=ALPHA&asset_class=equity&as_of={quote(T_A, safe="")}&technical_signal={q}'}[param]
        st, hd, body = h.http(port, path)
        assert st in (200, 400, 404), (param, payload[:30], st)
        h.strict_loads(body)
        assert b'Traceback' not in body and b'OperationalError' not in body and b'sqlite3.' not in body
    assert hashlib.sha256(main_db.read_bytes()).hexdigest() == sha            # read-only: file bytes unchanged
    con = sqlite3.connect(f'file:{main_db}?mode=ro', uri=True)
    assert con.execute('SELECT count(*) FROM observations').fetchone()[0] > 100
    assert con.execute("SELECT count(*) FROM sqlite_master WHERE name='observations'").fetchone()[0] == 1
    con.close()
    assert not os.path.exists('/tmp/x.db')


def test_long_inputs_and_unusual_paths(port):
    assert h.http(port, '/' + 'a' * 70000)[0] == 414
    assert h.http(port, '/fundamentals/ALPHA?as_of=' + '9' * 70000)[0] == 414
    assert h.http(port, '/fundamentals/ALPHA?' + '&'.join(f'k{i}=v' for i in range(3000)) + f'&as_of={quote(T_A, safe="")}')[0] == 200
    assert J(port, '/fundamentals//ALPHA?as_of=' + quote(T_A, safe=''))['symbol'] == 'ALPHA'      # empty path segments are ignored
    # repeated parameter: last one wins (documented parse_qs behaviour)
    p = f'/fundamentals/ALPHA?as_of={quote(T_B, safe="")}&as_of={quote(T_A, safe="")}'
    assert J(port, p)['period_end'] == '2024-12-31'


@pytest.mark.parametrize('method', ['HEAD', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
def test_non_get_methods_are_refused_and_never_mutate(port, main_db, method):
    sha = hashlib.sha256(main_db.read_bytes()).hexdigest()
    st, hd, body = h.http(port, '/health', method=method, body=b'x=1' if method in ('POST', 'PUT', 'PATCH') else None,
                          headers={'Content-Length': '3'} if method in ('POST', 'PUT', 'PATCH') else None)
    assert st == 501                                              # BaseHTTPRequestHandler: unsupported method (HTML body, documented)
    assert hashlib.sha256(main_db.read_bytes()).hexdigest() == sha
    assert J(port, '/health')['status'] == 'OK'                   # server still healthy


def test_concurrent_requests_20_threads_identical_bodies(port):
    paths = [with_asof(p, T_A) for p in ('/fundamentals/ALPHA', '/fundamentals/BETA/ttm', '/watchlist/fundamental', '/macro/regime',
                                         '/fundamentals/GAMMA/history', '/strategies/x/explanation?symbol=ALPHA&asset_class=equity')]
    base = {p: h.http(port, p)[2] for p in paths}

    def hit(i):
        p = paths[i % len(paths)]
        st, hd, body = h.http(port, p, timeout=60)
        return p, st, body

    with ThreadPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(hit, range(240)))
    bad = [(p, st) for p, st, body in results if st != 200 or body != base[p]]
    assert not bad, bad[:3]


def test_half_open_client_does_not_block_other_requests(port):
    """A client that sends an incomplete request must not stall the threaded server (no request timeout exists: documented limitation)."""
    s = socket.create_connection(('127.0.0.1', port))
    try:
        s.sendall(b'GET /health HTTP/1.1\r\nHost: x\r\n')
        for _ in range(5):
            assert J(port, '/health')['status'] == 'OK'
    finally:
        s.close()


# ============================================================================ database failure modes
def _status(path_db, endpoint='/fundamentals/ALPHA?as_of=' + quote(T_A, safe=''), timeout=30):
    with h.api_server(path_db) as p:
        st, hd, body = h.http(p, endpoint, timeout=timeout)
        return st, body


@pytest.mark.parametrize('kind', ['missing', 'corrupt', 'empty_file', 'directory', 'truncated', 'not_sqlite_but_big'])
def test_unreadable_database_gives_503_no_trade_never_data(tmp_path, main_db, kind):
    target = tmp_path / 'db.sqlite'
    if kind == 'corrupt':
        target.write_bytes(b'not a database ' * 200)
    elif kind == 'empty_file':
        target.write_bytes(b'')
    elif kind == 'directory':
        target = tmp_path
    elif kind == 'truncated':
        shutil.copy(main_db, target)
        with open(target, 'r+b') as f:
            f.truncate(20000)
    elif kind == 'not_sqlite_but_big':
        target.write_bytes(os.urandom(300_000))
    for ep in ('/fundamentals/ALPHA?as_of=' + quote(T_A, safe=''), '/watchlist/fundamental', '/macro/regime', '/fundamentals/ALPHA/history', '/fundamentals/ALPHA/events'):
        st, body = _status(target, ep)
        assert st == 503, (kind, ep, st, body[:100])
        assert h.strict_loads(body) == {'error': 'fundamental data unavailable; no trade'}


def test_locked_database_fails_closed_within_busy_timeout(tmp_path, main_db):
    target = tmp_path / 'locked.sqlite'
    shutil.copy(main_db, target)
    w = sqlite3.connect(target, isolation_level=None)
    w.execute('PRAGMA journal_mode=DELETE')
    w.execute('BEGIN EXCLUSIVE')
    try:
        t0 = time.time()
        st, body = _status(target, timeout=40)
        assert st == 503 and time.time() - t0 < 30
    finally:
        w.execute('ROLLBACK')
        w.close()


def test_wal_writer_in_flight_does_not_block_or_leak_uncommitted(tmp_path, main_db):
    target = tmp_path / 'wal.sqlite'
    shutil.copy(main_db, target)
    w = sqlite3.connect(target, isolation_level=None)
    assert w.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
    w.execute('BEGIN IMMEDIATE')
    w.execute("INSERT INTO observations VALUES ('zz','ALPHA','revenue',999.0,'2025-01-01','2025-12-31','2025-02-01T00:00:00+00:00','2025-02-01T00:00:00+00:00','2025-02-01T00:00:00+00:00','fixture','zz','USD','USD','financial','{}')")
    try:
        with h.api_server(target) as p:
            s = J(p, with_asof('/fundamentals/ALPHA', '2025-06-01T00:00:00Z'))
            assert s['period_end'] == '2024-12-31' and s['raw']['revenue'] == 120_000_000.0     # uncommitted row invisible
    finally:
        w.execute('ROLLBACK')
        w.close()


def test_read_only_file_permissions_still_served(tmp_path, main_db):
    target = tmp_path / 'ro.sqlite'
    shutil.copy(main_db, target)
    os.chmod(target, 0o444)
    st, body = _status(target)
    assert st == 200


def test_wal_database_in_read_only_directory_cannot_be_served_limitation(tmp_path, main_db):
    """LIMITATION (fail-closed): a checkpointed WAL-mode DB on read-only media needs a -shm file, so mode=ro open fails -> 503."""
    d = tmp_path / 'rodir'
    d.mkdir()
    target = d / 'db.sqlite'
    shutil.copy(main_db, target)
    c = sqlite3.connect(target)
    c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    c.close()
    os.chmod(target, 0o444)
    os.chmod(d, 0o555)
    try:
        st, body = _status(target)
        assert st == 503
    finally:
        os.chmod(d, 0o755)


def test_company_that_changed_reporting_currency_fails_closed_with_400_limitation(tmp_path):
    """LIMITATION: quarters() refuses any metric whose history mixes (source,currency,units), even annual-only data -> snapshot raises ValueError -> 400."""
    db = fx.build_edge_db(tmp_path / 'edge.sqlite', which=('SWITCH',))
    with h.api_server(db) as p:
        before = J(p, with_asof('/fundamentals/SWITCH', '2024-06-01T00:00:00Z'))
        assert before['currency'] == 'GBP' and before['status'] != 'MISSING'                 # GBP-only history works
        st, hd, body = h.http(p, with_asof('/fundamentals/SWITCH', '2025-06-01T00:00:00Z'))
        assert st == 400 and b'ambiguous quarter source/currency/units' in body              # after the USD filing: refused, never mixed or converted


def test_database_deleted_while_serving_flips_to_503(tmp_path, main_db):
    target = tmp_path / 'gone.sqlite'
    shutil.copy(main_db, target)
    with h.api_server(target) as p:
        assert h.http(p, '/fundamentals/ALPHA')[0] == 200
        target.unlink()
        for ext in ('-wal', '-shm'):
            (tmp_path / ('gone.sqlite' + ext)).unlink(missing_ok=True)
        st, hd, body = h.http(p, '/fundamentals/ALPHA')
        assert st == 503 and b'no trade' in body


def test_empty_valid_schema_db_returns_missing_everywhere(tmp_path):
    from fundamentals.store import PointInTimeFundamentalStore
    target = tmp_path / 'schema.sqlite'
    PointInTimeFundamentalStore(target).close()
    with h.api_server(target) as p:
        s = J(p, with_asof('/fundamentals/ALPHA', T_A))
        assert s['status'] == 'MISSING' and s['features'] == {}
        w = J(p, with_asof('/watchlist/fundamental', T_A))
        assert w['candidates'] == [] and w['ranked'] is False
        m = J(p, with_asof('/macro/regime', T_A))
        assert m['status'] == 'INCOMPLETE' and m['features'] == {} and set(m['regime'].values()) == {'UNKNOWN'}
        assert J(p, with_asof('/fundamentals/ALPHA/events', T_A))['status'] == 'MISSING'


# ============================================================================ loopback enforcement
@pytest.mark.parametrize('host', ['0.0.0.0', '', '::', '192.168.1.10', '8.8.8.8', 'example.com', '127.0.0.2', '10.0.0.1'])
def test_serve_refuses_non_loopback_hosts(main_db, host):
    from fundamentals.api import serve
    with pytest.raises(ValueError, match='loopback'):
        serve(str(main_db), host=host, port=h.free_port())


def test_server_binds_only_127_0_0_1_and_refuses_other_interfaces(main_db):
    with h.api_server(main_db) as p:
        assert h.http(p, '/health')[0] == 200
        ips = set()
        try:
            out = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=5).stdout.split()
            ips = {i for i in out if re.fullmatch(r'\d+\.\d+\.\d+\.\d+', i) and not i.startswith('127.')}
        except Exception:
            pass
        for ip in list(ips)[:3]:
            with socket.socket() as s:
                s.settimeout(3)
                assert s.connect_ex((ip, p)) != 0, f'bound on {ip}'


def test_localhost_hostname_is_accepted_and_binds_ipv4_loopback(main_db):
    from fundamentals.api import serve
    port = h.free_port()
    srv = serve(str(main_db), host='localhost', port=port)
    try:
        assert srv.server_address[0] == '127.0.0.1'
    finally:
        srv.server_close()


def test_host_header_is_not_validated_dns_rebinding_surface_is_documented_limitation(port):
    """The API is read-only but answers any Host header (DNS-rebinding exposure if a browser on this machine visits a hostile page)."""
    st, hd, body = h.http(port, '/health', headers={'Host': 'evil.example.com'})
    assert st == 200 and 'access-control-allow-origin' not in hd         # no CORS grant, so cross-origin reads stay blocked by the browser


# ============================================================================ DEFECT REPRODUCTIONS (strict xfail)
def test_defect_blank_as_of_must_be_rejected(port):
    st, hd, body = h.http(port, '/fundamentals/ALPHA?as_of=')
    assert st == 400


def test_defect_nonfinite_feature_must_not_drop_the_connection(tmp_path):
    db = fx.build_edge_db(tmp_path / 'edge.sqlite', which=('HUGE',))
    with h.api_server(db) as p:
        st, hd, body = h.http(p, '/fundamentals/HUGE?as_of=2025-04-01T00:00:00Z')
        assert st in (200, 400, 422, 500, 503)
        h.strict_loads(body)


def test_defect_watchlist_must_isolate_one_bad_symbol(tmp_path):
    db = fx.build_edge_db(tmp_path / 'edge.sqlite', which=('ALPHA', 'AMBIG'))
    with h.api_server(db) as p:
        st, hd, body = h.http(p, '/watchlist/fundamental?as_of=2025-04-01T00:00:00Z')
        assert st == 200
        assert 'ALPHA' in [r['symbol'] for r in h.strict_loads(body)['candidates']]


def test_defect_health_must_reflect_database_readability(tmp_path):
    bad = tmp_path / 'corrupt.sqlite'
    bad.write_bytes(b'not a database ' * 200)
    with h.api_server(bad) as p:
        assert h.http(p, '/health')[0] == 503


def test_defect_ipv6_loopback_is_whitelisted_but_unusable(main_db):
    from fundamentals.api import serve
    srv = serve(str(main_db), host='::1', port=h.free_port())
    srv.server_close()


def test_defect_tz_abbreviation_depends_on_host_timezone():
    """Lead re-audit (QA-UI-06 fixed): a zone abbreviation is never resolved through the host timezone. It is rejected, so the
    answer is identical on a London and a UTC host (previously 07:00 GMT became 06:00Z on a London host in summer)."""
    code = "from fundamentals.store import utc\ntry: print(utc('2025-07-03 07:00:00 GMT'))\nexcept ValueError: print('ValueError')"
    outs = [subprocess.run([sys.executable, '-W', 'ignore', '-c', code], capture_output=True, text=True, cwd=str(h.REPO), env={**os.environ, 'TZ': tz}).stdout.strip()
            for tz in ('Europe/London', 'UTC', 'America/New_York')]
    assert outs == ['ValueError'] * 3, outs


def test_defect_ambiguous_day_month_as_of_must_be_rejected(port):
    assert h.http(port, '/fundamentals/ALPHA?as_of=' + quote('04/05/2025 07:00:00Z', safe=''))[0] == 400


def test_defect_strategy_explanation_symbol_must_be_validated(port):
    st, hd, body = h.http(port, '/strategies/x/explanation?symbol=' + quote('<script>', safe=''))
    assert st == 400


def test_defect_stale_snapshot_must_not_be_scored_or_ranked(model_port):
    w = J(model_port, with_asof('/watchlist/fundamental', '2025-07-05T00:00:00Z'))
    stale = [r for r in w['candidates'] if r['data_status'] == 'STALE']
    assert stale and all(r['score'] is None for r in stale)


# ============================================================================ examples/serve_fundamentals.py CLI
def _wait_http(port, tries=100):
    for _ in range(tries):
        try:
            return h.http(port, '/health', timeout=2)
        except OSError:
            time.sleep(0.1)
    raise AssertionError('CLI server did not start')


def test_serve_cli_script_serves_loopback_and_honours_model_bundle(main_db, tmp_path):
    bundle = fx.write_bundle(tmp_path / 'bundle.json')
    port = h.free_port()
    log = (h.OUT / f'serve_cli_{port}.log').open('w')
    proc = subprocess.Popen([h.PY, 'examples/serve_fundamentals.py', '--database', str(main_db), '--port', str(port), '--model-bundle', str(bundle)],
                            cwd=str(h.REPO), stdout=log, stderr=subprocess.STDOUT)
    try:
        st, hd, body = _wait_http(port)
        assert st == 200
        s = J(port, with_asof('/fundamentals/ALPHA/score', T_A))
        assert s['status'] == 'RESEARCH_MODEL' and o.close(s['score'], o.model_score({'revenue_yoy': D(120) / D(98) - 1, 'gross_margin': D('0.4'), 'fcf_margin': D('0.1')})[0])
        out = subprocess.run(['ss', '-ltnH', f'sport = :{port}'], capture_output=True, text=True).stdout
        locals_ = [line.split()[3] for line in out.splitlines() if line.strip()]
        assert locals_ == ['127.0.0.1:%d' % port]                     # only the IPv4 loopback listener, no wildcard socket
    finally:
        proc.terminate()
        proc.wait(timeout=15)
        log.close()


def test_serve_cli_script_without_bundle_serves_unvalidated_and_bad_bundle_exits_nonzero(main_db, tmp_path):
    port = h.free_port()
    proc = subprocess.Popen([h.PY, 'examples/serve_fundamentals.py', '--database', str(main_db), '--port', str(port)], cwd=str(h.REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_http(port)
        assert J(port, with_asof('/fundamentals/ALPHA/score', T_A))['status'] == 'UNVALIDATED'
    finally:
        proc.terminate()
        proc.wait(timeout=15)
    bad = subprocess.run([h.PY, 'examples/serve_fundamentals.py', '--database', str(main_db), '--port', str(h.free_port()), '--model-bundle', str(tmp_path / 'missing.json')],
                         cwd=str(h.REPO), capture_output=True, text=True, timeout=60)
    assert bad.returncode != 0 and 'No such file' in bad.stderr                  # refuses to start rather than silently serving without the model


def test_serve_cli_with_missing_database_starts_but_every_endpoint_is_503(tmp_path):
    port = h.free_port()
    proc = subprocess.Popen([h.PY, 'examples/serve_fundamentals.py', '--database', str(tmp_path / 'nope.sqlite'), '--port', str(port)], cwd=str(h.REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        st, hd, body = _wait_http(port)
        assert st == 503 and b'no trade' in body
        assert h.http(port, '/watchlist/fundamental')[0] == 503
    finally:
        proc.terminate()
        proc.wait(timeout=15)


# ============================================================================ real SEC-derived data (copy of results/fundamentals)
REAL_DB = h.REPO / 'results' / 'fundamentals' / 'fundamentals.sqlite'


@pytest.fixture(scope='module')
def real_copy(tmp_path_factory):
    if not REAL_DB.exists():
        pytest.skip('results/fundamentals/fundamentals.sqlite not present')
    d = tmp_path_factory.mktemp('realcopy')
    shutil.copy(REAL_DB, d / 'real.sqlite')               # never open the original
    return d / 'real.sqlite'


def test_real_xom_api_equals_independent_oracle_on_60_random_dates_except_unverifiable_per_share(real_copy):
    """Differential on genuine SEC companyfacts (4,020 XOM observations): every feature except the per-share growth family must agree to 1e-12."""
    import random
    from datetime import datetime, timezone
    obs = o.db_obs(real_copy, 'XOM')
    assert len(obs) > 3000
    rng = random.Random(20261001)
    skip = ('eps_yoy', 'eps_acceleration', 'diluted_shares_yoy')
    bad = []
    with h.api_server(real_copy) as p:
        for i in range(60):
            T = datetime(2016, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=rng.randrange(0, int(10 * 365 * 86400)))
            iso = T.isoformat().replace('+00:00', 'Z')
            api = J(p, with_asof('/fundamentals/XOM', iso))
            orc = o.snapshot(obs, iso)
            for d in o.diff_snapshot(api, orc):
                if not d.startswith(skip) and not d.startswith('flags'):
                    bad.append(f'{iso}: {d}')
            assert api['per_share_comparability'] == 'UNVERIFIED_SPLIT_BASIS' if api['status'] != 'MISSING' else True
            assert o.ts(api.get('available_at') or '1970-01-01T00:00:00Z') <= o.ts(iso)
    assert not bad, bad[:5]


def test_real_sec_facts_never_carry_share_basis_so_eps_growth_is_always_unavailable_limitation(real_copy):
    """LIMITATION: sec_facts tags 'split_basis':'as_reported' but the feature engine requires metadata 'share_basis'; for SEC data EPS growth,
    EPS acceleration, dilution and TTM EPS are therefore permanently unavailable (fail-safe)."""
    con = sqlite3.connect(f'file:{real_copy}?mode=ro', uri=True)
    metas = [json.loads(r[0]) for r in con.execute("SELECT metadata FROM observations WHERE metric IN ('eps','diluted_shares') AND kind='financial'")]
    con.close()
    assert metas and all('share_basis' not in m and m.get('split_basis') == 'as_reported' for m in metas)
    with h.api_server(real_copy) as p:
        s = J(p, with_asof('/fundamentals/XOM', '2025-10-01T00:00:00Z'))
        assert s['features']['eps_yoy'] is None and s['features']['diluted_shares_yoy'] is None and s['per_share_comparability'] == 'UNVERIFIED_SPLIT_BASIS'
        assert s['ttm']['values']['eps'] is None
        assert J(p, with_asof('/fundamentals/XOM/ttm', '2025-10-01T00:00:00Z'))['status'] == 'INCOMPLETE'


def test_production_database_covers_exactly_one_company_phase_113(real_copy):
    con = sqlite3.connect(f'file:{real_copy}?mode=ro', uri=True)
    syms = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM observations WHERE kind='financial'")]
    con.close()
    assert syms == ['XOM']                                    # coverage gap: a one-company universe cannot support cross-sectional ranking
    with h.api_server(real_copy) as p:
        w = J(p, with_asof('/watchlist/fundamental', '2025-10-01T00:00:00Z'))
        assert [r['symbol'] for r in w['candidates']] == ['XOM'] and w['ranked'] is False
