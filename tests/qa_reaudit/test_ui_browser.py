"""TRACK F: real Streamlit page (dashboard/fundamentals_app.py) driven by Playwright/Chrome against fixture DBs.

Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_ui_browser.py
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


# ============================================================================ initial / missing database states
def test_default_page_without_database_says_so_and_shows_no_numbers(ui):
    assert ui.page.get_by_role('heading', name='Point-in-time fundamental research').count() == 1
    al = ui.alerts()
    assert INFO in al and any('No fundamental database' in a for a in al)
    assert ui.metrics() == {}
    assert ui.page.get_by_role('button', name='Download PIT snapshot').count() == 0
    assert not ui.exception_shown() and 'Traceback' not in ui.body()
    shot(ui, 'default_no_db')


# ============================================================================ boundary: 06:59 vs 07:01 UTC
def test_boundary_before_and_after_filing_numbers_and_json_download_equal_api(ui, dbs, api, truth, tmp_path):
    ui.set(db=dbs['main'], symbol='alpha', asof=T_B)           # lower-case input is normalised to ALPHA
    for asof, name in ((T_B, 'before_release'), (T_A, 'after_release')):
        ui.set(asof=asof)
        orc = o.snapshot(truth['ALPHA'], asof)
        m = ui.metrics()
        assert m['revenue_yoy'] in o.expect_pct(orc['features']['revenue_yoy']), (asof, m)
        assert m['gross_margin'] in o.expect_pct(orc['features']['gross_margin']), (asof, m)
        assert m['fcf'] in o.expect_money(orc['features']['fcf']), (asof, m)
        assert m['Data status'] == orc['status'] and m['Period age'] == str(orc['age_days']) and m['Fundamental score'] == 'Unvalidated'
        body = ui.body()
        assert f"Reported financial period {orc['period_end']}" in body
        snap_api = api_json(api, '/fundamentals/ALPHA?asset_class=equity', asof)
        assert m['Data coverage'] == f"{snap_api['completeness']:.0%}"
        assert f"Information available at {snap_api['available_at']}" in body
        dl = ui.download(tmp_path / f'{name}.json')
        assert dl == snap_api                                    # download == HTTP API, field for field (full precision)
        assert dl['features']['revenue_yoy'] == snap_api['features']['revenue_yoy']
        shot(ui, name)
    # exact publication instant
    ui.set(asof=T_0)
    assert ui.metrics()['revenue_yoy'] in o.expect_pct(o.snapshot(truth['ALPHA'], T_0)['features']['revenue_yoy'])
    ui.set(asof='2025-03-03T06:59:59Z')
    assert 'Reported financial period 2023-12-31' in ui.body()


def test_hundred_percent_vs_fraction_consistency_between_cards_and_table(ui, dbs, api):
    ui.set(db=dbs['main'], symbol='ALPHA', asof=T_A)
    snap = api_json(api, '/fundamentals/ALPHA?asset_class=equity', T_A)
    table = ui.features_table().set_index('Feature')['Value']
    assert table['revenue_yoy'] == snap['features']['revenue_yoy']            # raw fraction in the table...
    assert 0.2 < table['revenue_yoy'] < 0.25 and ui.metrics()['revenue_yoy'] == '22.45%'   # ...percent on the card
    assert ui.metrics()['Data coverage'] == '92%' and 0.9 < snap['completeness'] < 0.93


# ============================================================================ per-company display vs oracle
@pytest.mark.parametrize('symbol,asof', [
    ('ALPHA', T_A), ('BETA', '2025-01-01T00:00:00Z'), ('GAMMA', '2025-01-01T00:00:00Z'), ('DELTA', '2025-04-01T00:00:00Z'), ('GIANT', '2025-04-01T00:00:00Z'),
    ('NEGEQ', '2025-04-01T00:00:00Z'), ('LATE', '2025-06-15T00:00:00Z'), ('LATE', '2025-07-01T00:00:00Z'), ('LATE2', '2025-07-05T00:00:00Z'), ('FLATCO', '2025-04-01T00:00:00Z'),
])
def test_metric_cards_match_oracle_for_every_company(ui, dbs, api, truth, symbol, asof):
    ui.set(db=dbs['main'], symbol=symbol, asof=asof)
    orc = o.snapshot(truth[symbol], asof)
    m = ui.metrics()
    f = orc['features']
    assert m['revenue_yoy'] in o.expect_pct(f['revenue_yoy']), (symbol, m)
    assert m['gross_margin'] in o.expect_pct(f['gross_margin']), (symbol, m)
    if symbol == 'FLATCO':
        assert m['fcf'] in ('-0.00', '0.00')                    # |fcf| < 0.005 (cancellation noise): formatting handled by QA-UI-09 test
    else:
        assert m['fcf'] in o.expect_money(f['fcf']), (symbol, m)
    assert m['Data status'] == orc['status'] and m['Period age'] == str(orc['age_days'])
    assert f"Reported financial period {orc['period_end']}" in ui.body()
    assert f"Currency {orc['currency']}" in ui.body()
    flags = ui.jsons()['flags']
    assert set(flags) == set(orc['flags']), (symbol, flags, orc['flags'])
    table = ui.features_table().set_index('Feature')['Value']
    api_features = api_json(api, '/fundamentals/' + symbol + '?asset_class=equity', asof)['features']
    for k, v in api_features.items():
        tv = table[k]
        assert (v is None and (tv is None or (isinstance(tv, float) and math.isnan(tv)))) or tv == v, (symbol, k, tv, v)
    shot(ui, f'metrics_{symbol}_{asof[:10]}')


def test_missing_values_render_as_missing_never_zero(ui, dbs, truth):
    ui.set(db=dbs['main'], symbol='NEGEQ', asof='2025-04-01T00:00:00Z')
    m = ui.metrics()
    assert m['revenue_yoy'] == 'Missing'                          # zero prior revenue: growth undefined
    table = ui.features_table().set_index('Feature')['Value']
    orc = o.snapshot(truth['NEGEQ'], '2025-04-01T00:00:00Z')['features']
    for k, v in orc.items():
        tv = table[k]
        if v is None:
            assert tv is None or (isinstance(tv, float) and math.isnan(tv)), (k, tv)     # never coerced to 0.0
    assert orc['debt_equity'] is None and orc['roe'] is None      # negative equity: ratios refuse rather than flip sign
    assert 'Missing' in ui.body()


def test_large_and_scaled_values_format_with_thousands_separators(ui, dbs, truth):
    ui.set(db=dbs['main'], symbol='GIANT', asof='2025-04-01T00:00:00Z')
    expect = o.snapshot(truth['GIANT'], '2025-04-01T00:00:00Z')['features']['fcf']
    assert ui.metrics()['fcf'] in o.thousands2(expect)
    assert ui.metrics()['fcf'] == '102,600,000,000.00'            # 680bn*0.22 - 12bn - 35bn: 6.8e11-scale revenue renders in full
    ui.set(symbol='DELTA', asof='2025-04-01T00:00:00Z')
    assert ui.metrics()['fcf'] == '159,100,000.00' and 'Currency GBP' in ui.body()
    ui.set(symbol='GAMMA', asof='2025-01-01T00:00:00Z')
    assert ui.metrics()['fcf'] == '-6,000,000.00'                 # negative amounts keep their sign
    assert 'NEGATIVE_FCF' in ui.jsons()['flags']


# ============================================================================ charts
CHART_METRICS = ['revenue', 'eps', 'ocf', 'capex', 'gross_profit', 'equity', 'long_debt']


def read_chart(ui):
    spec, df = ui.chart()
    return spec, df


def series_of(df):
    if df is None:
        return []
    return [(str(r.period_end.date()), float(r.value)) for r in df.itertuples()]


@pytest.mark.parametrize('asof', ['2024-02-29T23:59:00Z', T_A, T_KA, '2026-03-01T00:00:00Z'])
def test_history_chart_data_equals_visible_versions_for_every_metric(ui, dbs, truth, asof):
    ui.set(db=dbs['main'], symbol='ALPHA', asof=asof)
    for metric in CHART_METRICS:
        ui.select('Financial history', metric)
        spec, df = read_chart(ui)
        want = [(e, float(v)) for e, v in o.chart_series(truth['ALPHA'], asof, metric)]
        got = series_of(df)
        assert len(got) == len(want) and all(g[0] == w[0] and o.close(g[1], w[1]) for g, w in zip(got, want)), (asof, metric, got, want)
        assert not any(abs(v - 123456789123.0) < 1 for _, v in got)               # future canary
        if want:
            enc = spec['layer'][0]['encoding']['x'] if 'layer' in spec else spec['encoding']['x']
            assert enc['type'] == 'temporal' and enc['field'] == 'period_end' and 'timeUnit' not in enc
        # the table under the chart lists the same period_end/value pairs
        t = ui.history_table()
        assert [(str(pd_end)[:10], float(v)) for pd_end, v in zip(t['period_end'], t['value'])] == got
    shot(ui, f'chart_alpha_{asof[:10]}')


def test_restated_values_replace_originals_in_chart_but_future_filing_never_appears(ui, dbs):
    ui.set(db=dbs['main'], symbol='ALPHA', asof=T_B)
    assert series_of(ui.chart()[1]) == [('2021-12-31', 80e6), ('2022-12-31', 90e6), ('2023-12-31', 100e6)]
    ui.set(asof=T_A)
    assert series_of(ui.chart()[1]) == [('2021-12-31', 80e6), ('2022-12-31', 90e6), ('2023-12-31', 98e6), ('2024-12-31', 120e6)]       # restated FY2023
    ui.select('Financial history', 'eps')
    assert series_of(ui.chart()[1])[-1] == ('2024-12-31', 1.2)
    ui.set(asof=T_KA)
    assert series_of(ui.chart()[1])[-1] == ('2024-12-31', 1.05)                         # 10-K/A: 1.20 -> 1.05 at 07:00:00
    ui.set(asof='2026-03-02T06:59:00Z')
    assert ('2025-12-31', 123456789123.0) not in series_of(ui.chart()[1]) and len(series_of(ui.chart()[1])) == 4
    ui.set(asof='2026-03-02T07:01:00Z')
    assert series_of(ui.chart()[1])[-1][0] == '2025-12-31' and series_of(ui.chart()[1])[-1][1] == 123456789123.0   # positive control


@pytest.mark.parametrize('symbol,asof,dates,first,last', [
    ('BETA', '2025-01-01T00:00:00Z', ['2022-10-01', '2023-09-30', '2024-09-28'], 500e6, 650e6),
    ('BETA', '2024-11-15T20:59:00Z', ['2022-10-01', '2023-09-30'], 500e6, 560e6),
    ('GAMMA', '2025-01-01T00:00:00Z', ['2022-06-30', '2023-06-30', '2024-06-30'], 30e6, 60e6),
    ('GAMMA', '2024-08-23T10:59:00Z', ['2022-06-30', '2023-06-30'], 30e6, 42e6),
    ('DELTA', '2025-04-01T00:00:00Z', ['2022-12-31', '2023-12-31', '2024-12-31'], 1250.5e6, 1589.75e6),
])
def test_non_calendar_fiscal_period_axis_and_scaled_values(ui, dbs, symbol, asof, dates, first, last):
    ui.set(db=dbs['main'], symbol=symbol, asof=asof)
    spec, df = read_chart(ui)
    got = series_of(df)
    assert [d for d, _ in got] == dates                       # x-axis is the true fiscal period end (52/53-week FY, June FY), not a calendar year
    assert o.close(got[0][1], first) and o.close(got[-1][1], last)
    assert 'Annual flow / instant balance values from versions available at the chosen timestamp.' in ui.body()
    shot(ui, f'chart_{symbol}_{asof[:10]}')


def test_chart_is_not_drawn_for_missing_asset_class_and_no_stale_chart_remains(ui, dbs):
    ui.set(db=dbs['main'], symbol='ALPHA', asof=T_A)
    assert ui.chart()[1] is not None
    ui.select('Instrument class', 'etf')
    assert ui.chart()[1] is None and ui.metrics()['Data status'] == 'REQUIRES_DATED_HOLDINGS' and ui.metrics()['Data coverage'] == '0%'
    assert ui.metrics()['revenue_yoy'] == 'Missing'
    ft = ui.features_table()
    assert ft is None or len(ft) == 0
    assert ui.page.get_by_role('button', name='Download PIT snapshot').count() == 1       # snapshot of the refusal itself may be exported
    for ac, status in (('forex', 'REQUIRES_MACRO_MODEL'), ('crypto', 'REQUIRES_NATIVE_DATA')):
        ui.select('Instrument class', ac)
        assert ui.metrics()['Data status'] == status and ui.metrics()['fcf'] == 'Missing'
    ui.select('Instrument class', 'equity')
    assert ui.metrics()['fcf'] == '12,000,000.00' and ui.chart()[1] is not None
