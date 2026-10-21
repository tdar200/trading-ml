"""'Latest' selection sites (period_end / available_at), classification PIT, limits of the version key, extreme values and
HTTP robustness (QA track A)."""
from __future__ import annotations
import json
import random
import threading
import urllib.error
import urllib.request
import warnings
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import pytest

from tests.qa_reaudit import pit_helpers as H
from tests.qa_reaudit.pit_helpers import M, fact

from fundamentals.adapters import historical_fx
from fundamentals.features import FundamentalFeatureEngine
from fundamentals.models import MacroRegimeService
from fundamentals.store import Fact, PointInTimeFundamentalStore


# ---------------------------------------------------------------------------
# selection by latest period_end happens only over rows already published at T
# ---------------------------------------------------------------------------
def fx_quote(date, rate, pub, rev='q'):
    return Fact('FX:GBPUSD', 'rate', rate, date, pub, pub, 'ECB', rev, 'USD', 'USD/GBP', '', H.OBS, 'fx', {})


def test_fx_latest_quote_by_period_end_never_picks_a_quote_published_after_T():
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest([fx_quote('2025-05-01', 1.30, '2025-05-01T17:00:00Z'), fx_quote('2025-05-08', 1.40, '2025-05-08T17:00:00Z')])
    assert historical_fx(s, 100., 'GBP', 'USD', '2025-05-08T16:59:59.999999Z')['rate'] == 1.30
    assert historical_fx(s, 100., 'GBP', 'USD', '2025-05-08T17:00:00Z')['rate'] == 1.40
    assert historical_fx(s, 100., 'GBP', 'USD', '2025-05-08T17:00:00Z')['value'] == pytest.approx(140.)
    # a quote dated after its own availability is refused at ingestion (never visible)
    out = s.ingest_validated([fx_quote('2025-05-20', 1.99, '2025-05-09T17:00:00Z', rev='future')])
    assert out['rejected'] == 1
    assert historical_fx(s, 100., 'GBP', 'USD', '2025-05-12T00:00:00Z')['rate'] == 1.40
    # staleness is judged by the quote date: 8 days later no rate is fabricated
    assert historical_fx(s, 100., 'GBP', 'USD', '2025-05-16T00:00:00Z')['rate'] is None


def macro_obs(series, period, value, avail_day, units, rev='v1'):
    t = f'{avail_day}T00:00:00Z'
    return Fact('MACRO:' + series, series, value, period, t, t, 'ALFRED/graph', rev, '', units, '', H.OBS, 'macro', {})


def test_macro_latest_period_is_chosen_among_published_vintages_only_and_revisions_follow_vintage_day():
    s = PointInTimeFundamentalStore(':memory:')
    facts = []
    for i in range(30):  # CPI monthly index 2023-01 .. 2025-06
        y, m = 2023 + i // 12, i % 12 + 1
        pub = datetime(y, m, 1, tzinfo=timezone.utc) + timedelta(days=45)
        facts.append(macro_obs('CPIAUCSL', f'{y}-{m:02d}-01', 300. + i, pub.date().isoformat(), 'index'))
    last_period = '2025-06-01'
    assert facts[-1].period_end == last_period
    for name, units, val in (('UNRATE', 'percent', 4.0), ('FEDFUNDS', 'percent', 5.0)):
        for i in range(30):
            y, m = 2023 + i // 12, i % 12 + 1
            pub = datetime(y, m, 1, tzinfo=timezone.utc) + timedelta(days=45)
            facts.append(macro_obs(name, f'{y}-{m:02d}-01', val + 0.01 * i, pub.date().isoformat(), units))
    s.ingest(facts)
    svc = MacroRegimeService(s)
    pub_last = datetime(2025, 6, 1, tzinfo=timezone.utc) + timedelta(days=45)  # 2025-07-16
    before = svc.state((pub_last - timedelta(microseconds=1)).isoformat())
    after = svc.state(pub_last.isoformat())
    assert before['features']['unemployment'] == pytest.approx(4.0 + 0.01 * 28)
    assert after['features']['unemployment'] == pytest.approx(4.0 + 0.01 * 29)
    # a vintage revision published later changes the *same* period only from its own instant
    s.ingest([macro_obs('UNRATE', last_period, 9.9, '2025-08-20', 'percent', rev='v2')])
    assert svc.state('2025-08-19T23:59:59Z')['features']['unemployment'] == pytest.approx(4.0 + 0.01 * 29)
    assert svc.state('2025-08-20T00:00:00Z')['features']['unemployment'] == 9.9


def test_classification_sector_change_is_point_in_time_and_changes_ratio_availability():
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest(H.fy_facts('fy24', '2024-01-01', '2024-12-31', '2025-02-20T12:00:00Z', 100 * M))
    cls = lambda sector, avail, rev: Fact('ACME', 'sector', None, '2024-12-31', avail, avail, 'GICS', rev, '', 'text', '', H.OBS, 'classification', {'sector': sector})  # noqa: E731
    s.ingest([cls('software', '2025-01-01T00:00:00Z', 'a'), cls('banks', '2025-04-01T00:00:00Z', 'b')])
    e = FundamentalFeatureEngine(s)
    a, b = e.snapshot('ACME', '2025-03-31T23:59:59.999999Z'), e.snapshot('ACME', '2025-04-01T00:00:00Z')
    assert a['sector'] == 'software' and a['features']['current_ratio'] == pytest.approx(2.0)
    assert b['sector'] == 'banks' and b['features']['current_ratio'] is None  # bank adapter drops industrial liquidity ratios
    assert e.snapshot('ACME', '2024-12-31T23:59:59Z')['sector'] is None


# ---------------------------------------------------------------------------
# limitation: the version key contains period_start, so a correction that changes period_start does not supersede
# ---------------------------------------------------------------------------
def test_limit_correction_with_changed_period_start_is_a_new_period_not_a_supersession():
    """LIMITATION (documented key = metric, period_start, period_end, currency, units, source): both vintages stay
    visible, the snapshot uses the OLDER one (first max by period_end). Real XBRL amendments keep period boundaries."""
    s = PointInTimeFundamentalStore(':memory:')
    s.ingest(H.fy_facts('fy24', '2024-01-01', '2024-12-31', '2025-02-20T12:00:00Z', 100 * M))
    s.ingest([fact('revenue', 90 * M, '2024-12-31', '2023-12-31', '2025-03-20T12:00:00Z', rev='amend-start-shifted')])
    rows = [r for r in s.as_of('ACME', '2025-04-01T00:00:00Z', kind='financial') if r['metric'] == 'revenue']
    assert len(rows) == 2  # two live versions for the "same" fiscal year
    snap = FundamentalFeatureEngine(s).snapshot('ACME', '2025-04-01T00:00:00Z')
    assert snap['raw']['revenue'] == 100 * M  # the superseded number still wins


# ---------------------------------------------------------------------------
# extreme but valid magnitudes: non-finite feature values and a dropped HTTP connection (QA-PIT-07)
# ---------------------------------------------------------------------------
def _extreme_db(path):
    s = PointInTimeFundamentalStore(path)
    facts = []
    for y, v in ((2021, 1e200), (2022, 2e200), (2023, 3e200)):
        for m, val in (('revenue', 5e200), ('net_income', v), ('gross_profit', v)):
            facts.append(fact(m, val, f'{y}-12-31', f'{y}-01-01', f'{y + 1}-02-01T00:00:00Z'))
    s.ingest(facts)
    s.close()


def test_snapshot_features_are_finite_or_none_for_extreme_magnitudes(tmp_path):
    p = tmp_path / 'x.sqlite'
    _extreme_db(p)
    s = PointInTimeFundamentalStore(p)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        snap = FundamentalFeatureEngine(s).snapshot('ACME', '2024-03-01T00:00:00Z')
    bad = {k: v for k, v in snap['features'].items() if v is not None and (v != v or abs(v) == float('inf'))}
    assert not bad, bad
    json.dumps(snap, allow_nan=False)


def test_http_answers_with_json_even_when_the_payload_is_not_serialisable(tmp_path):
    p = tmp_path / 'x.sqlite'
    _extreme_db(p)
    with H.http_server(p) as get:
        try:
            code, body, _, raw = get('/fundamentals/ACME?as_of=2024-03-01T00:00:00Z')
        except Exception as e:  # RemoteDisconnected / URLError
            pytest.fail(f'no HTTP response: {type(e).__name__}')
        assert code in (200, 400, 500, 503)
        assert b'Infinity' not in raw


# ---------------------------------------------------------------------------
# seeded HTTP fuzz of as_of / symbol / metric: always a JSON answer, never a dropped connection or a future-shifted value
# ---------------------------------------------------------------------------
import re
ISO = re.compile(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d{1,9})?(Z|[+-]\d\d:\d\d)')
FUZZ_AS_OF = ['2025-05-10T07:00:00Z', '2025-05-10T07:00:00.123456789Z', ' 2025-05-10T07:00:00Z', '2025-05-10T07:00:00Z ', '2025-05-10T07:00:00ZZ', '2025-05-10T07:00:00+0100',
              '2025-05-10T07:00:00+01', '2025-05-10T07:00:00 UTC', '2025-05-10 07:00:00Z', '20250510T070000Z', '2025-W19-6T07:00:00Z', '1e9', '0', '-1', 'NaN', 'now', 'today',
              '2025-02-30T00:00:00Z', '2025-05-10T25:00:00Z', '1677-09-21T00:00:00Z', '2262-04-12T00:00:00Z', '9999-12-31T23:59:59Z', '0001-01-01T00:00:00Z', '٢٠٢٥-05-10T07:00:00Z',
              'a' * 5000, '%00', '../../etc/passwd', "1'; DROP TABLE observations; --", '<script>alert(1)</script>', '2025-05-10T07:00:00Z&as_of=2099-01-01T00:00:00Z']


def test_http_fuzz_always_returns_json_and_never_silently_shifts_the_instant(tmp_path):
    p = tmp_path / 'f.sqlite'
    s = PointInTimeFundamentalStore(p)
    s.ingest(H.acme_nov_fy('2025-05-10T07:00:00Z'))
    s.close()
    rng = random.Random(20261001)
    with H.http_server(p) as get:
        for i in range(160):
            as_of = rng.choice(FUZZ_AS_OF)
            path = rng.choice(['/fundamentals/ACME', '/fundamentals/ACME/ttm', '/fundamentals/ACME/history', '/fundamentals/ACME/score', '/fundamentals/ACME/events',
                               '/fundamentals/ACME/quality', '/watchlist/fundamental', '/macro/regime', '/strategies/x/explanation'])
            extra = rng.choice(['', '&metric=revenue', '&metric=' + quote("x'--"), '&asset_class=etf', '&asset_class=%FF', '&symbol=' + quote('a b/c')])
            url = f'{path}?as_of={quote(as_of, safe="")}{extra}'
            try:
                code, body, headers, raw = get(url)
            except Exception as e:
                pytest.fail(f'dropped connection for {url[:120]!r}: {type(e).__name__}')
            assert code in (200, 400, 404, 503), (url[:120], code)
            assert headers['Content-Type'] == 'application/json' and headers['Cache-Control'] == 'no-store'
            assert b'NaN' not in raw and b'Infinity' not in raw
            if code == 200 and isinstance(body, dict) and 'as_of' in body and ISO.fullmatch(as_of):
                # whatever was accepted must be exactly the instant that was asked for (no silent shift to now or to a date)
                try:
                    want = H.ns_instant(as_of)
                except (OverflowError, ValueError):
                    continue
                assert H.ns_instant(body['as_of']) == want, (as_of, body['as_of'])
    # SQL/markup payloads never altered the database
    s = PointInTimeFundamentalStore(p)
    assert s.db.execute('select count(*) from observations').fetchone()[0] == len(H.acme_nov_fy())


def test_http_headers_methods_and_loopback_enforcement_for_the_pit_api(tmp_path):
    from fundamentals.api import serve
    p = tmp_path / 'h.sqlite'
    PointInTimeFundamentalStore(p).close()
    for host in ('0.0.0.0', '192.168.0.10', '::', 'example.com', ''):
        with pytest.raises(ValueError, match='loopback'):
            serve(str(p), host=host, port=0)
    httpd = serve(str(p), port=0)
    assert httpd.server_address[0] == '127.0.0.1'
    httpd.server_close()
