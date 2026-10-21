"""Host-timezone independence of timestamp handling (QA track A, phases 141-144).

Every subprocess runs with TZ=<zone> so the *host* local timezone changes. Numeric-offset / 'Z' inputs must give the
same UTC instant everywhere. Timezone *abbreviations* ('GMT', 'EST', 'EDT', 'BST') are the attack surface: dateutil
resolves any abbreviation that equals the host's own tzname to the host's local zone (with the host's DST rule).
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ZONES = ['UTC', 'Europe/London', 'America/New_York', 'Asia/Karachi', 'Pacific/Auckland']

PROBE = textwrap.dedent('''
    import sys, json, warnings
    warnings.simplefilter('ignore')
    sys.path.insert(0, %r)
    from fundamentals.store import utc, Fact, PointInTimeFundamentalStore
    inputs = json.loads(sys.argv[1])
    out = {}
    for a in inputs:
        try:
            out[a] = utc(a)
        except BaseException as e:
            out[a] = 'ERR:' + type(e).__name__
    # facts carrying the strings as availability: which instant does the store believe?
    s = PointInTimeFundamentalStore(':memory:')
    vis = {}
    for i, a in enumerate(inputs):
        try:
            s.ingest([Fact('H', 'zz%%d' %% i, 1.0, '2000-01-01', '2000-01-02T00:00:00Z', a, 'S', 'r', 'USD', 'USD', '', '2030-01-01T00:00:00Z', 'financial', {})])
        except BaseException as e:
            vis[a] = 'ERR:' + type(e).__name__
    for r in s.db.execute('select metric, available_at from observations'):
        vis[inputs[int(r[0][2:])]] = r[1]
    print(json.dumps({'utc': out, 'fact_available_at': vis}))
''' % str(ROOT))


def run_in_zone(zone, inputs):
    env = dict(os.environ, TZ=zone)
    p = subprocess.run([sys.executable, '-c', PROBE, json.dumps(inputs)], capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


NUMERIC = {
    '2025-05-10T07:00:00Z': '2025-05-10T07:00:00+00:00',
    '2025-05-10T08:00:00+01:00': '2025-05-10T07:00:00+00:00',
    '2025-05-10T03:00:00-04:00': '2025-05-10T07:00:00+00:00',
    '2025-05-10T12:30:00+05:30': '2025-05-10T07:00:00+00:00',
    '2025-03-09T03:00:00-04:00': '2025-03-09T07:00:00+00:00',   # first instant after the US spring-forward gap
    '2025-03-09T01:59:59-05:00': '2025-03-09T06:59:59+00:00',
    '2025-11-02T01:30:00-04:00': '2025-11-02T05:30:00+00:00',   # first 01:30 of the repeated hour
    '2025-11-02T01:30:00-05:00': '2025-11-02T06:30:00+00:00',   # second 01:30
    '2025-03-30T02:00:00+01:00': '2025-03-30T01:00:00+00:00',   # UK spring forward
    '2025-10-26T01:30:00+01:00': '2025-10-26T00:30:00+00:00',   # UK first 01:30 (BST)
    '2025-10-26T01:30:00+00:00': '2025-10-26T01:30:00+00:00',   # UK second 01:30 (GMT)
    '2025-05-10T07:00:00.000000001Z': '2025-05-10T07:00:00.000000001+00:00',
}
NAIVE = ['2025-05-10T07:00:00', '2025-05-10', '2025-03-09T02:30:00', '2025-10-26T01:30:00']


@pytest.mark.parametrize('zone', ZONES)
def test_numeric_offset_inputs_give_the_same_utc_instant_whatever_the_host_timezone(zone):
    got = run_in_zone(zone, list(NUMERIC) + NAIVE)
    for k, v in NUMERIC.items():
        assert got['utc'][k] == v, f'TZ={zone} {k}'
        assert got['fact_available_at'][k] == v, f'TZ={zone} fact availability {k}'
    for k in NAIVE:
        assert got['utc'][k] == 'ERR:ValueError', f'TZ={zone}: naive {k} must be rejected, got {got["utc"][k]}'
        assert got['fact_available_at'][k] == 'ERR:ValueError'


ABBREV = {  # (input, true UTC instant)
    'Sat, 10 May 2025 07:00:00 GMT': '2025-05-10T07:00:00+00:00',     # RFC 2822 / HTTP-date, e.g. JavaScript toUTCString()
    '2025-05-10T07:00:00 GMT': '2025-05-10T07:00:00+00:00',
    '2025-05-10T07:00:00 EST': '2025-05-10T12:00:00+00:00',           # EST is UTC-5 whatever the date
    '2025-01-10T07:00:00 EDT': '2025-01-10T11:00:00+00:00',           # EDT is UTC-4 whatever the date
    '2025-05-10T07:00:00 UTC': '2025-05-10T07:00:00+00:00',
}


def test_timezone_abbreviations_are_never_resolved_through_the_host_local_zone():
    bad = {}
    for zone in ('Europe/London', 'America/New_York'):
        got = run_in_zone(zone, list(ABBREV))
        for k, v in ABBREV.items():
            r = got['fact_available_at'][k]
            if r != v and not r.startswith('ERR:'):
                bad[(zone, k)] = (r, v)
    assert not bad, bad


def test_sec_naive_acceptance_assumption_is_new_york_regardless_of_host_timezone():
    code = textwrap.dedent('''
        import sys, json
        sys.path.insert(0, %r)
        from fundamentals.providers import sec_facts
        pay = {'cik': 1, 'facts': {'us-gaap': {'Revenues': {'units': {'USD': [{'accn': 'A', 'form': '10-K', 'filed': '2025-02-14', 'start': '2024-01-01', 'end': '2024-12-31', 'val': 1.0}]}}}}}
        subs = {'filings': {'recent': {'accessionNumber': ['A'], 'acceptanceDateTime': ['2025-02-14T16:30:00']}}}
        print(json.dumps([f.available_at for f in sec_facts(pay, subs, 'X')]))
    ''' % str(ROOT))
    seen = set()
    for zone in ZONES:
        p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=dict(os.environ, TZ=zone), cwd=str(ROOT), timeout=120)
        assert p.returncode == 0, p.stderr[-1500:]
        seen.add(p.stdout.strip().splitlines()[-1])
    assert seen == {'["2025-02-14T21:35:00+00:00"]'}
