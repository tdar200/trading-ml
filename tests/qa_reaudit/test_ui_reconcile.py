"""TRACK F: Metric | Oracle | Database | API | UI reconciliation, coverage report (phase 113), explanation/label consistency.

Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_ui_reconcile.py
Writes results/qa_reaudit_20261001/ui/reconciliation.csv and coverage_symbol_year.csv
"""
import csv
import json
import re
import urllib.parse
from decimal import Decimal as D

import pytest
from playwright.sync_api import sync_playwright

from tests.qa_reaudit import ui_fixture as fx
from tests.qa_reaudit import ui_harness as h
from tests.qa_reaudit import ui_oracle as o

T_B, T_A, T_KA = '2025-03-03T06:59:00Z', '2025-03-03T07:01:00Z', '2025-04-15T07:01:00Z'
CASES = [('ALPHA', T_B), ('ALPHA', T_A), ('ALPHA', T_KA), ('BETA', '2025-01-01T00:00:00Z'), ('GAMMA', '2025-01-01T00:00:00Z'), ('DELTA', '2025-04-01T00:00:00Z')]
FEATURES = ['fcf', 'gross_margin', 'operating_margin', 'net_margin', 'fcf_margin', 'roic', 'roe', 'debt_equity', 'net_debt', 'current_ratio', 'revenue_yoy', 'eps_yoy',
            'fcf_yoy', 'revenue_cagr_3y', 'gross_margin_change', 'interest_coverage']


@pytest.fixture(scope='module')
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp('uirecon')
    db = fx.build_main_db(d / 'main.sqlite')
    return dict(db=db, bundle=fx.write_bundle(d / 'bundle.json'), dir=d)


@pytest.fixture(scope='module')
def servers(env):
    with h.api_server(env['db']) as a, h.api_server(env['db'], score=fx.frozen_engine()) as am, h.streamlit_app('recon') as s:
        yield dict(api=a, api_model=am, st=s)


@pytest.fixture(scope='module')
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True)
        yield b
        b.close()


def api(port, path, asof):
    st, hd, body = h.http(port, path + ('&' if '?' in path else '?') + 'as_of=' + urllib.parse.quote(asof, safe=''))
    assert st == 200, body[:200]
    return h.strict_loads(body)


def fnum(x):
    if x is None or (isinstance(x, float) and x != x):
        return None
    return float(x)


def same(*vals, rel=1e-12):
    base = next((v for v in vals if v is not None), None)
    if base is None:
        return all(v is None for v in vals)
    return all(v is not None and o.close(base, v, rel, 1e-9 if abs(float(base)) < 1e-2 else 1e-15) for v in vals)


def test_reconciliation_table_oracle_database_api_ui(env, servers, browser):
    truth = fx.all_truth()
    rows = []
    bad = []
    ui = h.Ui(browser, servers['st'])
    try:
        ui.set(db=env['db'])
        for symbol, asof in CASES:
            case = f'{symbol}@{asof}'
            ui.set(symbol=symbol, asof=asof)
            orc = o.snapshot(truth[symbol], asof)
            dbs = o.snapshot(o.db_obs(env['db'], symbol), asof)
            snap = api(servers['api'], f'/fundamentals/{symbol}?asset_class=equity', asof)
            table = ui.features_table().set_index('Feature')['Value']
            cards = ui.metrics()

            def add(metric, oracle, database, api_v, ui_v):
                vals = [oracle, database, api_v, ui_v]
                ok = same(*[None if v is None or (isinstance(v, float) and v != v) else v for v in vals])
                rows.append({'Case': case, 'Metric': metric, 'Oracle': fnum(oracle), 'Database': fnum(database), 'API': fnum(api_v), 'UI': fnum(ui_v), 'Status': 'OK' if ok else 'MISMATCH'})
                if not ok:
                    bad.append((case, metric, oracle, database, api_v, ui_v))

            # raw statements through the charts (revenue, EPS): last plotted point is the current period
            for metric, key in (('revenue', 'revenue'), ('eps', 'eps')):
                ui.select('Financial history', metric)
                pts = ui.chart()[1]
                add(metric, orc['raw'].get(key), dbs['raw'].get(key), snap['raw'].get(key), None if pts is None else float(pts.value.iloc[-1]))
            for f in FEATURES:
                add(f, orc['features'].get(f), dbs['features'].get(f), snap['features'].get(f), table[f])
            # card strings vs oracle formatting
            for card, f, fmt in (('revenue_yoy', 'revenue_yoy', o.expect_pct), ('gross_margin', 'gross_margin', o.expect_pct), ('fcf', 'fcf', o.expect_money)):
                ok = cards[card] in fmt(orc['features'][f])
                rows.append({'Case': case, 'Metric': f'card:{card}', 'Oracle': sorted(fmt(orc['features'][f]))[0], 'Database': sorted(fmt(dbs['features'][f]))[0],
                             'API': snap['features'][f], 'UI': cards[card], 'Status': 'OK' if ok else 'MISMATCH'})
                if not ok:
                    bad.append((case, 'card:' + card, cards[card]))
            # TTM
            ttm_ui = ui.jsons()['ttm']
            ttm_api = api(servers['api'], f'/fundamentals/{symbol}/ttm', asof)
            if symbol == 'ALPHA':
                ov, _ = o.ttm_alpha(truth[symbol], asof); dv, _ = o.ttm_alpha(o.db_obs(env['db'], symbol), asof)
            elif symbol == 'BETA':
                ov = o.ttm_sum_quarters(truth[symbol], asof)[0]; dv = o.ttm_sum_quarters(o.db_obs(env['db'], symbol), asof)[0]
            else:
                ov = dv = None
            for k in ('revenue', 'net_income', 'ocf', 'capex', 'eps', 'fcf'):
                if ov is None:
                    add('ttm_' + k, None, None, ttm_api['values'][k], ttm_ui['values'][k])
                else:
                    add('ttm_' + k, ov[k], dv[k], ttm_api['values'][k], ttm_ui['values'][k])
            assert ttm_ui == {k: v for k, v in ttm_api.items() if k not in ('symbol', 'as_of')}, case          # UI block == /ttm endpoint
            # macro + catalysts (symbol independent, but shown on every page)
            macro_ui = ui.jsons()['macro']
            macro_api = api(servers['api'], '/macro/regime', asof)
            mo, md = o.macro_state(asof), o.macro_state_from_db(env['db'], asof)
            assert macro_ui['status'] == macro_api['status'] == mo['status'] == md['status'], case
            for k in sorted(set(mo['features']) | set(macro_api['features'])):
                add('macro_' + k, mo['features'].get(k), md['features'].get(k), macro_api['features'].get(k), macro_ui['features'].get(k))
        # catalysts: ALPHA earnings surprise (1.10 vs consensus 1.00 -> 10%)
        ui.set(symbol='ALPHA', asof=T_KA)
        ev_ui = [e for e in ui.jsons()['events']['events'] if e['event_type'] == 'earnings'][0]
        ev_api = [e for e in api(servers['api'], '/fundamentals/ALPHA/events', T_KA)['events'] if e['event_type'] == 'earnings'][0]
        import sqlite3
        con = sqlite3.connect(f"file:{env['db']}?mode=ro", uri=True)
        payload = json.loads(con.execute("SELECT payload FROM corporate_events WHERE event_type='earnings' AND symbol='ALPHA'").fetchone()[0])
        con.close()
        db_surprise = (D(repr(payload['reported_eps'])) - D(repr(payload['consensus_eps']))) / abs(D(repr(payload['consensus_eps'])))
        orc_surprise = (D('1.10') - D('1.00')) / D('1.00')
        ok = o.close(db_surprise, orc_surprise, 1e-12) and o.close(ev_api['eps_surprise'], orc_surprise, 1e-12) and o.close(ev_ui['eps_surprise'], orc_surprise, 1e-12)
        rows.append({'Case': 'ALPHA@' + T_KA, 'Metric': 'catalyst_eps_surprise', 'Oracle': float(orc_surprise), 'Database': float(db_surprise), 'API': ev_api['eps_surprise'],
                     'UI': ev_ui['eps_surprise'], 'Status': 'OK' if ok else 'MISMATCH'})
        if not ok:
            bad.append(('catalyst', db_surprise, ev_api['eps_surprise'], ev_ui['eps_surprise']))
        assert ev_ui == ev_api
        # score: model bundle in the UI vs model-serving API vs hand arithmetic
        ui.set(bundle=str(env['bundle']), symbol='ALPHA', asof=T_A)
        sc_api = api(servers['api_model'], '/fundamentals/ALPHA/score', T_A)
        f = o.snapshot(truth['ALPHA'], T_A)['features']
        fd = o.snapshot(o.db_obs(env['db'], 'ALPHA'), T_A)['features']
        so = o.model_score(f)[0]; sd = o.model_score(fd)[0]
        card = ui.metrics()['Fundamental score']
        ok = same(so, sd, sc_api['score']) and card in o.pct2(so)
        rows.append({'Case': 'ALPHA@' + T_A, 'Metric': 'score(frozen ridge)', 'Oracle': float(so), 'Database': float(sd), 'API': sc_api['score'], 'UI': card, 'Status': 'OK' if ok else 'MISMATCH'})
        if not ok:
            bad.append(('score', so, sd, sc_api['score'], card))
    finally:
        ui.close()
    out = h.OUT / 'reconciliation.csv'
    with out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['Case', 'Metric', 'Oracle', 'Database', 'API', 'UI', 'Status'])
        w.writeheader()
        w.writerows(rows)
    assert len(rows) > 150
    assert not bad, bad[:5]


# ============================================================================ phase 113 coverage
def test_coverage_report_by_symbol_sector_year_and_how_gaps_surface(env, servers, browser):
    report = o.coverage_report(env['db'])
    out = h.OUT / 'coverage_symbol_year.csv'
    with out.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['symbol', 'sector', 'fiscal_period_end', 'year', 'present', 'expected', 'missing', 'first_available_at'])
        for r in report:
            w.writerow([r['symbol'], r['sector'], r['fiscal_period_end'], r['year'], r['present'], r['expected'], '|'.join(r['missing']), r['first_available_at']])
    by = {(r['symbol'], r['fiscal_period_end']): r for r in report}
    # known truths from the fixture
    assert by[('ALPHA', '2024-12-31')]['present'] == 13 and by[('ALPHA', '2024-12-31')]['sector'] == 'software'
    assert by[('GAMMA', '2024-06-30')]['missing'] == []                       # GAMMA has every headline metric
    assert set(by[('FLATCO', '2024-12-31')]['missing']) == {'assets', 'equity', 'cash', 'short_debt', 'long_debt'}
    assert 'revenue' not in by[('NEGEQ', '2023-12-31')]['missing'] and by[('NEGEQ', '2023-12-31')]['present'] >= 9
    assert by[('ALPHA', '2025-12-31')]['first_available_at'].startswith('2026-03-02T07:00:00')
    # roll-ups by sector and by year
    sectors = {}
    years = {}
    for r in report:
        sectors.setdefault(r['sector'], []).append(r['present'] / r['expected'])
        years.setdefault(r['year'], []).append(r['present'] / r['expected'])
    assert set(sectors) == {'software', 'UNCLASSIFIED'} and set(years) == {'2021', '2022', '2023', '2024', '2025'}
    # does the product surface the gap? API completeness / UI card must be < 100% whenever any headline metric is missing in the latest FY
    ui = h.Ui(browser, servers['st'])
    try:
        ui.set(db=env['db'])
        for sym in ('FLATCO', 'NEGEQ', 'DELTA', 'GAMMA'):
            snap = api(servers['api'], f'/fundamentals/{sym}?asset_class=equity', '2025-12-31T00:00:00Z')
            latest = max(r['fiscal_period_end'] for r in report if r['symbol'] == sym)
            gaps = by[(sym, latest)]['missing']
            ui.set(symbol=sym, asof='2025-12-31T00:00:00Z')
            card = ui.metrics()['Data coverage']
            if gaps:
                assert snap['completeness'] < 1.0 and card != '100%', (sym, gaps)
            assert card == f"{snap['completeness']:.0%}"
    finally:
        ui.close()
    # there is no endpoint/page that reports coverage by symbol x sector x year: only per-symbol completeness (documented NOT IMPLEMENTED)
    for path in ('/coverage', '/coverage/by_year', '/watchlist/coverage', '/fundamentals/ALPHA/coverage'):
        assert h.http(servers['api'], path)[0] == 404


# ============================================================================ phases 80/121: explanations must agree with machine state
BANNED = re.compile(r'\b(strong|buy|sell|bullish|bearish|undervalued|overvalued|cheap|expensive|recommend\w*|outperform\w*|guarantee\w*)\b', re.I)


def strings(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)
    elif isinstance(x, str):
        yield x


@pytest.mark.parametrize('symbol,asof', [('GAMMA', '2025-01-01T00:00:00Z'), ('ALPHA', T_A), ('NEGEQ', '2025-04-01T00:00:00Z'), ('DELTA', '2025-04-01T00:00:00Z')])
def test_no_recommendation_language_and_flags_match_arithmetic_contradictory_data(env, servers, symbol, asof):
    """GAMMA: revenue +42.86% (growth) but net loss and negative FCF: the page/API may only state descriptive flags that follow from the numbers."""
    snap = api(servers['api'], f'/fundamentals/{symbol}?asset_class=equity', asof)
    dec = api(servers['api'], f'/strategies/x/explanation?symbol={symbol}&asset_class=equity&technical_signal=true', asof)
    orc = o.snapshot(fx.all_truth()[symbol], asof)
    assert set(snap['flags']) == set(orc['flags']) == set(dec['fundamental']['flags']) == set(dec['risk']['flags'])
    texts = list(strings(snap)) + list(strings(dec['explanation'])) + [dec['decision'], dec['risk']['status']]
    for t in texts:
        if t in snap['lineage'] or re.fullmatch(r'[0-9a-f]{64}', t):
            continue
        assert not BANNED.search(t), t
    # decision labels agree with the permit flag; policy 'observe' cannot create a research candidate
    assert dec['permit_signal'] is True and dec['decision'] == 'TECHNICAL_CANDIDATE' and dec['policy'] == 'observe'
    if symbol == 'GAMMA':
        f = orc['features']
        assert f['revenue_yoy'] > 0.4 and f['fcf'] < 0 and 'NEGATIVE_FCF' in snap['flags']
        assert 'DILUTION' not in snap['flags']                                  # share count flat: no spurious dilution flag
    if symbol == 'NEGEQ':
        assert orc['features']['debt_equity'] is None and snap['features']['debt_equity'] is None


def test_ui_text_has_no_recommendation_language_for_contradictory_company(env, servers, browser):
    ui = h.Ui(browser, servers['st'])
    try:
        ui.set(db=env['db'], symbol='GAMMA', asof='2025-01-01T00:00:00Z')
        body = ui.body()
        stripped = body.replace('Research only. No fundamental model is approved for deployment. Missing scores are not zero scores.', '')
        assert not BANNED.search(stripped), BANNED.search(stripped).group(0)
        assert 'NEGATIVE_FCF' in ui.jsons()['flags'] and ui.metrics()['fcf'] == '-6,000,000.00'
    finally:
        ui.close()


def test_snapshot_status_vs_completeness_label_is_independent(servers):
    """AVAILABLE only means 'not stale': a company with 10% completeness is still labelled AVAILABLE (documented in the feature docstring)."""
    s = api(servers['api'], '/fundamentals/FLATCO?asset_class=equity', '2025-04-01T00:00:00Z')
    assert s['status'] == 'AVAILABLE' and s['completeness'] < 0.25
    assert s['confidence'] is None and 'not calibrated predictive confidence' in s['confidence_note']


# ============================================================================ the real (copied) production DB through the UI
def test_real_xom_copy_ui_cards_equal_api_and_independent_oracle(servers, browser, tmp_path):
    import shutil
    real = h.REPO / 'results' / 'fundamentals' / 'fundamentals.sqlite'
    if not real.exists():
        pytest.skip('production fundamentals DB not present')
    copy = tmp_path / 'real.sqlite'
    shutil.copy(real, copy)
    obs = o.db_obs(copy, 'XOM')
    ui = h.Ui(browser, servers['st'])
    try:
        ui.set(db=copy, symbol='XOM', asof='2024-06-01T00:00:00Z')              # the page's own default as-of
        orc = o.snapshot(obs, '2024-06-01T00:00:00Z')
        m = ui.metrics()
        f = orc['features']
        assert m['revenue_yoy'] in o.expect_pct(f['revenue_yoy']) and m['gross_margin'] in o.expect_pct(f['gross_margin']) and m['fcf'] in o.expect_money(f['fcf'])
        assert m['Data status'] == orc['status'] and m['Period age'] == str(orc['age_days'])
        assert f"Reported financial period {orc['period_end']}" in ui.body()
        series = [(str(r.period_end.date()), float(r.value)) for r in ui.chart()[1].itertuples()]
        assert series == [(e, float(v)) for e, v in o.chart_series(obs, '2024-06-01T00:00:00Z', 'revenue')]
        assert series[-1][0] <= '2024-06-01' and len(series) >= 5
        shot = h.OUT / 'real_xom_default_asof.png'
        ui.page.screenshot(path=str(shot), full_page=True)
    finally:
        ui.close()
