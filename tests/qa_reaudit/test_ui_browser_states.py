"""TRACK F (part 2): error/unavailable states, model bundle, XSS inertness and UI defect reproductions (real Streamlit + Chrome).

Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_ui_browser_states.py
The Streamlit server runs with an EMPTY cwd, so the page's relative default DB path never opens the real
results/fundamentals database. Charts are inspected from the exact protobuf messages Streamlit sends over the websocket.
"""
import json
import math
import zlib
import sqlite3
import urllib.parse
from datetime import timedelta
from decimal import Decimal as D

import pytest
from playwright.sync_api import sync_playwright

from tests.qa_reaudit import ui_fixture as fx
from tests.qa_reaudit import ui_harness as h
from tests.qa_reaudit import ui_oracle as o

SHOTS = h.OUT
T_B, T_0, T_A = '2025-03-03T06:59:00Z', '2025-03-03T07:00:00Z', '2025-03-03T07:01:00Z'
T_KA_B, T_KA = '2025-04-15T06:59:00Z', '2025-04-15T07:00:00Z'
INFO = 'Research only. No fundamental model is approved for deployment. Missing scores are not zero scores.'


@pytest.fixture(scope='module')
def dbs(tmp_path_factory):
    d = tmp_path_factory.mktemp('uibrowser')
    main = fx.build_main_db(d / 'main.sqlite')
    edge = fx.build_edge_db(d / 'edge.sqlite')
    from fundamentals.store import PointInTimeFundamentalStore
    PointInTimeFundamentalStore(d / 'schema_only.sqlite').close()
    (d / 'corrupt.sqlite').write_bytes(b'this is not sqlite ' * 100)
    (d / 'empty.sqlite').write_bytes(b'')
    bundle = fx.write_bundle(d / 'bundle.json')
    return dict(main=main, edge=edge, schema=d / 'schema_only.sqlite', corrupt=d / 'corrupt.sqlite', empty=d / 'empty.sqlite', bundle=bundle, dir=d)


@pytest.fixture(scope='module')
def truth():
    return fx.all_truth()


@pytest.fixture(scope='module')
def api(dbs):
    with h.api_server(dbs['main']) as p:
        yield p


@pytest.fixture(scope='module')
def st_port():
    with h.streamlit_app('browser') as p:
        yield p


@pytest.fixture(scope='module')
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True)
        yield b
        b.close()


@pytest.fixture
def ui(browser, st_port):
    u = h.Ui(browser, st_port)
    yield u
    u.close()


def api_json(port, path, asof):
    st, hd, body = h.http(port, path + ('&' if '?' in path else '?') + 'as_of=' + urllib.parse.quote(asof, safe=''))
    assert st == 200, body[:200]
    return h.strict_loads(body)


def shot(ui, name):
    ui.page.screenshot(path=str(SHOTS / f'{name}.png'), full_page=True)


# ============================================================================ error / unavailable states (phases 124/125/128)
def assert_no_values_on_screen(ui):
    assert ui.metrics() == {}, ui.metrics()
    b = ui.body()
    for stale in ('Unvalidated', '22.45%', '12,000,000.00', 'AVAILABLE', 'Reported financial period', 'Download PIT snapshot'):
        assert stale not in b, stale
    assert ui.page.get_by_role('button', name='Download PIT snapshot').count() == 0
    assert ui.chart()[1] is None
    assert not ui.exception_shown() and 'Traceback' not in b and 'File "' not in b


@pytest.mark.parametrize('field,value,message', [
    ('asof', 'garbage', 'unsupported timestamp format'), ('asof', '2025-03-03T07:01:00', 'explicit timezone required'),
    ('asof', '2025-03-03', 'explicit timezone required'), ('asof', '', 'Fundamental information unavailable'),
    ('symbol', 'A B', 'invalid symbol'), ('symbol', '../etc/passwd', 'unknown endpoint'), ('symbol', "ALPHA'; DROP TABLE observations;--", 'invalid symbol'),
    ('symbol', 'A' * 40, 'invalid symbol'), ('symbol', ' ALPHA ', 'invalid symbol'), ('asof', 'X' * 20000, 'unsupported timestamp format'), ('symbol', '', 'unknown endpoint'), ('bundle', '/nonexistent/bundle.json', 'No such file or directory'),
    ('bundle', '/etc/hostname', 'Fundamental information unavailable'),
])
def test_invalid_inputs_show_message_and_clear_previous_numbers(ui, dbs, field, value, message):
    ui.set(db=dbs['main'], symbol='ALPHA', asof=T_A)
    assert ui.metrics()['fcf'] == '12,000,000.00'
    if field == 'asof':
        ui.set(asof=value)
    elif field == 'symbol':
        ui.set(symbol=value)
    else:
        ui.set(bundle=value)
    alerts = ui.alerts()
    assert any(message in a for a in alerts), alerts
    assert any(a.startswith('Fundamental information unavailable:') for a in alerts)
    assert_no_values_on_screen(ui)
    shot(ui, f'error_{field}_{zlib.crc32(value.encode()) % 10000}')
    # recovery: the next valid input shows fresh numbers, not leftovers
    ui.set(symbol='ALPHA', asof=T_A, bundle='')
    assert ui.metrics()['revenue_yoy'] == '22.45%'


@pytest.mark.parametrize('kind', ['nonexistent', 'directory', 'corrupt', 'empty_file'])
def test_database_unavailable_states(ui, dbs, tmp_path, kind):
    ui.set(db=dbs['main'], symbol='ALPHA', asof=T_A)
    assert ui.metrics()['revenue_yoy'] == '22.45%'
    target = {'nonexistent': tmp_path / 'nope.sqlite', 'directory': dbs['dir'], 'corrupt': dbs['corrupt'], 'empty_file': dbs['empty']}[kind]
    ui.set(db=target)
    al = ui.alerts()
    if kind == 'nonexistent':
        assert any('No fundamental database' in a for a in al)
    else:
        assert any(a.startswith('Fundamental information unavailable:') for a in al), al
    assert_no_values_on_screen(ui)
    shot(ui, f'db_{kind}')


def test_schema_only_database_labels_everything_missing(ui, dbs):
    ui.set(db=dbs['schema'], symbol='ALPHA', asof=T_A)
    m = ui.metrics()
    assert m['Data status'] == 'MISSING' and m['Data coverage'] == '0%' and m['Period age'] == '\\u2014'.encode().decode('unicode_escape') and m['fcf'] == 'Missing' and m['Fundamental score'] == 'Unvalidated'
    j = ui.jsons()
    assert j['macro']['status'] == 'INCOMPLETE' and j['events']['status'] == 'MISSING' and j['watchlist']['candidates'] == []
    assert j['ttm']['status'] == 'INCOMPLETE' and ui.chart()[1] is None


def test_unknown_symbol_is_labelled_missing_not_fabricated(ui, dbs):
    ui.set(db=dbs['main'], symbol='NOSUCH', asof=T_A)
    m = ui.metrics()
    assert m['Data status'] == 'MISSING' and m['revenue_yoy'] == m['gross_margin'] == m['fcf'] == 'Missing' and m['Data coverage'] == '0%'
    assert ui.chart()[1] is None
    assert ui.page.get_by_role('button', name='Download PIT snapshot').count() == 1       # export of the MISSING snapshot itself (status MISSING, no numbers)
    assert ui.jsons()['ttm']['status'] == 'INCOMPLETE'


def test_stale_and_late_filer_labels_exact_550_day_boundary(ui, dbs, truth):
    end = o.pd_('2023-12-31')
    day550, day551 = end + timedelta(days=550), end + timedelta(days=551)
    ui.set(db=dbs['main'], symbol='LATE2', asof=f'{day550}T12:00:00Z')
    assert ui.metrics()['Data status'] == 'AVAILABLE' and ui.metrics()['Period age'] == '550'
    ui.set(asof=f'{day551}T12:00:00Z')
    assert ui.metrics()['Data status'] == 'STALE' and ui.metrics()['Period age'] == '551'
    assert 'Reported financial period 2023-12-31' in ui.body()                       # stale numbers are labelled, still the old period
    ui.set(symbol='LATE', asof='2025-06-30T19:59:00Z')
    assert 'Reported financial period 2023-12-31' in ui.body() and ui.metrics()['Data status'] == 'AVAILABLE'       # late filer: previous year until filed
    ui.set(asof='2025-06-30T20:01:00Z')
    assert 'Reported financial period 2024-12-31' in ui.body()
    shot(ui, 'late_filer')


def test_page_rereads_database_on_every_run_no_stale_cache(ui, dbs, tmp_path):
    import shutil
    from fundamentals.store import PointInTimeFundamentalStore
    copy = tmp_path / 'live.sqlite'
    shutil.copy(dbs['main'], copy)
    ui.set(db=copy, symbol='ALPHA', asof='2025-03-05T00:00:00Z')
    assert ui.metrics()['revenue_yoy'] == '22.45%'
    w = PointInTimeFundamentalStore(copy)                  # a new, later filing arrives while the page is open
    w.ingest([fx.to_fact(fx.Obs('ALPHA', 'revenue', '2024-01-01', '2024-12-31', D(125_000_000), '2025-03-04T12:00:00Z', 'late-amend'))])
    w.close()
    ui.set(asof='2025-03-05T00:01:00Z')
    assert ui.metrics()['revenue_yoy'] == f"{125 / 98 - 1:.2%}"                       # fresh read: 27.55%
    ui.set(asof='2025-03-04T11:59:00Z')
    assert ui.metrics()['revenue_yoy'] == '22.45%'                                    # and the older view is still reproducible


def test_quality_block_is_labelled_current_audit_not_historical(ui, dbs):
    """The rejection log is not point-in-time; the page must say so when a historical as_of is chosen."""
    ui.set(db=dbs['main'], symbol='GAMMA', asof='2023-01-01T00:00:00Z')
    q = ui.jsons()['quality']
    assert q['status'] == 'QUARANTINED_RECORDS' and q['accuracy_certified'] is False
    assert 'not historical signal input' in q['time_scope'] and 'not independent verification' in q['note']
    assert 'Source validation' in ui.body()


# ============================================================================ model bundle
def test_valid_bundle_score_matches_hand_arithmetic_and_invalid_boundary(ui, dbs):
    ui.set(db=dbs['main'], symbol='ALPHA', asof=T_A, bundle=str(dbs['bundle']))
    expect = o.model_score({'revenue_yoy': D(120) / D(98) - 1, 'gross_margin': D('0.4'), 'fcf_margin': D('0.1')})[0]
    assert ui.metrics()['Fundamental score'] in o.pct2(expect)                         # 2.59%: predicted 5-session return shown as percent
    assert 'No held-out calibration artifact loaded' in ui.body()
    ui.set(asof='2024-01-01T00:00:00Z')
    assert any('model unavailable at historical timestamp' in a for a in ui.alerts()) and ui.metrics() == {}
    shot(ui, 'bundle_error_before_training_cutoff')


def test_bundle_missing_feature_shows_unvalidated_and_never_zero_score(ui, dbs):
    ui.set(db=dbs['main'], symbol='NEGEQ', asof='2025-04-01T00:00:00Z', bundle=str(dbs['bundle']))
    s = ui.metrics()['Fundamental score']
    assert s == 'Unvalidated' and '0.00%' not in s


def test_calibrated_bundle_displays_heldout_probability_and_wilson_interval(ui, dbs, tmp_path):
    from fundamentals.models import ModelArtifact
    a = ModelArtifact(**{**fx.MODEL, 'features': tuple(fx.MODEL['features']), 'means': tuple(fx.MODEL['means']), 'scales': tuple(fx.MODEL['scales']), 'coefficients': tuple(fx.MODEL['coefficients'])})
    cal = dict(model_id=a.id, calibrated_at='2024-06-01T00:00:00+00:00', bins=[[10, 4], [20, 15]], sample_sha256='x', observations=30, brier=0.2)
    p = tmp_path / 'cal_bundle.json'
    p.write_text(json.dumps({'model': fx.MODEL, 'calibration': cal}))
    ui.set(db=dbs['main'], symbol='ALPHA', asof=T_A, bundle=str(p))
    body = ui.body()
    other = ui.jsons().get('other', [])
    conf = [d for d in other if 'probability' in d]
    assert conf and conf[0]['probability'] == 0.75 and conf[0]['observations'] == 20 and conf[0]['event'] == 'net forward return > 0'
    # independent Wilson 95% interval for k=15, n=20
    n, k, z = 20, 15, 1.959963984540054
    p_hat = k / n
    centre = (p_hat + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    assert conf[0]['interval_95'] == pytest.approx([centre - half, centre + half], rel=1e-12)
    assert 'Coverage measures populated features, not predictive confidence' in body


# ============================================================================ XSS / markup inertness
HOSTILE_ASOF = ['<script>window.__xss_asof=1</script>', '[click](javascript:window.__xss_link=1)', '<img src=x onerror=window.__xss_img=1>', '**bold** `code` ![i](http://127.0.0.1:1/x.png)',
                '</div><svg onload=window.__xss_svg=1>', '{{7*7}} ${7*7} <iframe src=javascript:window.__xss_if=1>']


def xss_probe(page):
    return page.evaluate("""() => ({
        vars: Object.keys(window).filter(k => k.startsWith('__xss')),
        onerror: document.querySelectorAll('[onerror],[onload],[onclick]').length,
        jsLinks: Array.from(document.querySelectorAll('a')).filter(a => (a.getAttribute('href')||'').toLowerCase().startsWith('javascript:')).length,
        iframes: document.querySelectorAll('iframe').length,
        scripts: Array.from(document.querySelectorAll('script')).filter(s => (s.textContent||'').includes('__xss')).length,
        svgs: Array.from(document.querySelectorAll('svg[onload]')).length })""")


@pytest.mark.parametrize('payload', HOSTILE_ASOF)
def test_hostile_as_of_text_never_executes(ui, dbs, payload):
    """Nothing typed into as_of may execute (script, handlers, iframes, javascript: navigation); clickable-markup rendering is QA-UI-12 below."""
    ui.set(db=dbs['main'], symbol='ALPHA', asof=payload)
    probe = xss_probe(ui.page)
    probe.pop('jsLinks')
    assert probe == {'vars': [], 'onerror': 0, 'iframes': 0, 'scripts': 0, 'svgs': 0}, probe
    for a in ui.page.locator('a').filter(has_text='click').all():            # a rendered javascript: link must not run when clicked
        with ui.ctx.expect_page() as newp:
            a.click()
        newp.value.wait_for_timeout(300)
        assert newp.value.evaluate("Object.keys(window).filter(k => k.startsWith('__xss')).length") == 0
    assert xss_probe(ui.page)['vars'] == []
    assert ui.dialogs == []
    assert any(a.startswith('Fundamental information unavailable:') for a in ui.alerts())
    assert not ui.exception_shown()
    shot(ui, f'xss_asof_{zlib.crc32(payload.encode()) % 10000}')


def test_hostile_source_revision_event_payload_and_symbol_are_inert(ui, dbs):
    ui.set(db=dbs['main'], symbol='XSSCO', asof='2025-04-01T00:00:00Z')
    ui.select('Financial history', 'eps')
    assert ui.metrics()['Data status'] == 'AVAILABLE'
    hist = ui.history_table()
    assert fx.XSS_SOURCE in set(hist['source']) and any(fx.XSS_REV in r for r in hist['revision'])       # hostile strings are real data...
    ev = ui.jsons()['events']['events']
    assert any('<script>' in str(e['payload'].get('note')) for e in ev)
    wl = ui.jsons()['watchlist']['candidates']
    assert '<B>BAD</B>' in [r['symbol'] for r in wl]                                                    # ...and the odd symbol is listed as data
    probe = xss_probe(ui.page)
    assert probe == {'vars': [], 'onerror': 0, 'jsLinks': 0, 'iframes': 0, 'scripts': 0, 'svgs': 0}, probe   # ...but nothing executes or becomes markup
    assert ui.dialogs == []
    # the hostile text appears only as inert text (or not at all) and never as an element
    assert ui.page.locator('script', has_text='__xss').count() == 0
    assert ui.page.locator('img[src="x"]').count() == 0
    shot(ui, 'xss_source_revision_payload')


def test_hostile_symbol_is_rejected_before_any_query_and_not_echoed(ui, dbs):
    ui.set(db=dbs['main'], symbol='<img src=x onerror=window.__xss_sym=1>', asof=T_A)
    assert xss_probe(ui.page)['vars'] == [] and ui.dialogs == []
    assert any('invalid symbol' in a for a in ui.alerts()) and '<img' not in ' '.join(ui.alerts())


# ============================================================================ DEFECT REPRODUCTIONS (strict xfail)
def test_defect_negative_zero_is_displayed(ui, dbs):
    ui.set(db=dbs['main'], symbol='FLATCO', asof='2025-04-01T00:00:00Z')
    m = ui.metrics()
    assert not m['fcf'].startswith('-0.00') and not m['revenue_yoy'].startswith('-0.00')


def test_defect_empty_history_selection_is_silent(ui, dbs):
    ui.set(db=dbs['main'], symbol='FLATCO', asof='2025-04-01T00:00:00Z')
    ui.select('Financial history', 'equity')
    assert ui.chart()[1] is not None or 'no data' in ui.body().lower() or 'not available' in ui.body().lower() or 'missing' in ui.body().lower().replace('missing scores', '')


def test_defect_partial_render_then_error_for_a_healthy_symbol(ui, dbs):
    ui.set(db=dbs['edge'], symbol='ALPHA', asof='2025-04-01T00:00:00Z')
    assert not any(a.startswith('Fundamental information unavailable:') for a in ui.alerts())


def test_score_card_says_unvalidated_for_missing_features_but_reason_is_in_the_watchlist_block(ui, dbs):
    """LIMITATION: the card reads 'Unvalidated' for no-model AND MISSING_FEATURES; the machine reason only appears inside the watchlist JSON."""
    ui.set(db=dbs['main'], symbol='NEGEQ', asof='2025-04-01T00:00:00Z', bundle=str(dbs['bundle']))
    assert ui.metrics()['Fundamental score'] == 'Unvalidated'
    rows = {r['symbol']: r for r in ui.jsons()['watchlist']['candidates']}
    assert rows['NEGEQ']['reason'] == 'MISSING_FEATURES' and rows['NEGEQ']['score'] is None


def test_defect_user_text_echoed_into_error_is_rendered_as_markdown_links(ui, dbs):
    ui.set(db=dbs['main'], symbol='ALPHA', asof='[Download the fix](https://example.invalid/x) ![pixel](http://127.0.0.1:1/p.png)')
    assert ui.page.get_by_test_id('stAlert').locator('a, img').count() == 0
