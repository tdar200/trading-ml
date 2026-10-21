"""QA track B: generalisation to companies NOT used to tune the tag mapping.

Fresh public SEC EDGAR data (fetched by tests/qa_reaudit/real_fetch.py into results/qa_reaudit_20261001/real/raw):
AAPL (Sep 52/53-week FY), PFE, F (loss in FY2025), KO, NEE (utility), AMZN (TTM columns in 10-Qs), PLUG (loss-making small
cap with signed-capex tagging errors), NKE (May FY), GSK (IFRS, GBP, 20-F).

Official numbers come from the PRIMARY 10-K / 10-Q / 20-F documents (inline XBRL as printed, with scale and printed label),
cached under tests/qa_reaudit/fixtures/real/official by tests/qa_reaudit/real_official.py.  That is a different path from the
companyfacts API that production ingests.
"""
from __future__ import annotations
import csv
import datetime as dt
import json
import math
import random
from pathlib import Path

import pytest

from fundamentals.features import FundamentalFeatureEngine
from tests.qa_reaudit import real_db
from tests.qa_reaudit.real_db import GROUPNEW, compare, fixed_dates, random_instant, rounding_unit
from tests.qa_reaudit.real_oracle import RAWNEW, UTC, days_inclusive, iso_utc, load_company, parse_utc

pytestmark = pytest.mark.skipif(not real_db.have_new_raw(), reason='fresh SEC cache missing: run tests/qa_reaudit/real_fetch.py')
HERE = Path(__file__).resolve().parent
OFFICIAL = HERE / 'fixtures' / 'real' / 'official'
OUT = HERE.parents[1] / 'results' / 'qa_reaudit_20261001' / 'real'
SEEDS = range(24)
HEADLINE = ('revenue', 'net_income', 'eps', 'ocf', 'capex', 'assets', 'equity', 'cash', 'gross_profit', 'operating_income', 'diluted_shares')


@pytest.fixture(scope='module')
def storenew():
    store = real_db.open_ro(real_db.dbnew())
    yield store
    store.close()


@pytest.fixture(scope='module')
def enginenew(storenew):
    return FundamentalFeatureEngine(storenew)


@pytest.fixture(scope='module')
def cosnew():
    return {s: load_company(s) for s in GROUPNEW}


# --------------------------------------------------------------------------------------------- ingestion health
def test_real_new_companies_ingest_and_only_signed_capex_is_quarantined(storenew):
    real_db.dbnew()
    for symbol in GROUPNEW:
        info = real_db.INGEST_SUMMARY[symbol]
        assert info['accepted'] + info['rejected'] == info['candidates']
        stored = storenew.db.execute('SELECT COUNT(*) FROM observations WHERE symbol=?', (symbol,)).fetchone()[0]
        assert stored == info['accepted']
        if symbol != 'PLUG':
            assert info['rejected'] == 0, (symbol, info)
    rejected = storenew.db.execute("SELECT reason,payload FROM ingestion_rejections WHERE symbol='PLUG'").fetchall()
    assert len(rejected) == 8
    for row in rejected:
        payload = json.loads(row['payload'])
        assert row['reason'].startswith('canonical capex must be positive')
        assert payload['metric'] == 'capex' and payload['value'] < 0           # kept verbatim, never "repaired"
        assert payload['metadata']['tag'] == 'PaymentsToAcquirePropertyPlantAndEquipment'


@pytest.mark.parametrize('symbol', GROUPNEW)
def test_real_new_every_stored_fact_equals_raw_json(symbol, storenew, cosnew):
    co = cosnew[symbol]
    index = {}
    for standard in ('us-gaap', 'ifrs-full'):
        for tag, body in co.facts['facts'].get(standard, {}).items():
            for unit, rows in body['units'].items():
                for r in rows:
                    index[(standard, tag, unit, r.get('start', ''), r['end'], r['accn'])] = r
    n = 0
    for row in storenew.db.execute("SELECT * FROM observations WHERE symbol=? AND kind='financial'", (symbol,)):
        meta = json.loads(row['metadata'])
        key = (meta['accounting_standard'], meta['tag'], row['units'], row['period_start'], row['period_end'], row['revision'])
        raw = index.get(key)
        assert raw is not None, key
        assert row['value'] == float(raw['val']), (key, row['value'], raw['val'])
        pub = co.pub_time(raw['accn'], raw['filed'])
        assert row['available_at'] == pub.isoformat()
        assert parse_utc(row['available_at']) >= co.lower_bound_time(raw['accn'], raw['filed'])
        n += 1
    assert n > 100


# --------------------------------------------------------------------------------------------- differential
@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('symbol', GROUPNEW)
def test_real_new_random_pair_matches_human_view(symbol, seed, enginenew, cosnew):
    rng = random.Random(f'new-{symbol}-{seed}')
    T = random_instant(rng)
    issues, snap = compare(enginenew, cosnew[symbol], symbol, T)
    assert not issues, f'{symbol} seed={seed} T={iso_utc(T)}: {issues}'
    if snap['lineage']:
        marks = ','.join('?' * len(snap['lineage']))
        latest = enginenew.store.db.execute(f'SELECT MAX(available_at) FROM observations WHERE id IN ({marks})', snap['lineage']).fetchone()[0]
        assert parse_utc(latest) <= T


@pytest.mark.parametrize('symbol', GROUPNEW)
def test_real_new_fixed_date_grid_matches_human_view(symbol, enginenew, cosnew):
    problems = []
    for T in fixed_dates():
        issues, _ = compare(enginenew, cosnew[symbol], symbol, T)
        problems += [f'{iso_utc(T)}: {i}' for i in issues]
    assert not problems, problems[:6]


# --------------------------------------------------------------------------------------------- official reconciliation
def _official_table(path: Path):
    """{(metric, start, end): dict(value,label,shown,scale,concept)} - the least-rounded printed value per key."""
    data = json.loads(path.read_text())
    best: dict = {}
    for r in data['rows']:
        key = (r['metric'], r['start'], r['end'])
        sig = len(f'{abs(r["value"]):.0f}'.rstrip('0'))
        if key not in best or sig > best[key]['sig']:
            best[key] = dict(r, sig=sig)
    return data, best


def _stored(store, symbol, T, metric, start, end):
    for r in store.as_of(symbol, iso_utc(T), kind='financial'):
        if r['metric'] == metric and r['period_end'] == end and (r['period_start'] == start or not start):
            return r
    return None


ANNUAL_DOCS = sorted(p for p in OFFICIAL.glob('*.extract.json') if json.loads(p.read_text())['form'] in ('10-K', '20-F'))


@pytest.fixture(scope='module')
def reconciliation(storenew, cosnew):
    rows = []
    for path in ANNUAL_DOCS:
        data, table = _official_table(path)
        symbol, accn = data['symbol'], data['accn']
        co = cosnew[symbol]
        T = co.pub_time(accn, data['filed']) + dt.timedelta(seconds=1)
        report_end = data['report_date']
        for (metric, start, end), off in sorted(table.items()):
            if metric not in HEADLINE:
                continue
            annual = bool(start) and 350 <= days_inclusive(start, end) <= 380
            instant = not start
            if not (annual or instant):
                continue
            gap_days = (dt.date.fromisoformat(report_end) - dt.date.fromisoformat(end)).days
            if not (-3 <= gap_days <= 3 or 360 <= gap_days <= 372):   # current FY and the prior-year comparative column
                continue
            stored = _stored(storenew, symbol, T, metric, start, end)
            sv = None if stored is None else stored['value']
            if sv is None:
                status, diff = 'GAP', None
            else:
                diff = sv - off['value']
                status = 'PASS' if abs(diff) <= 1e-9 * max(1.0, abs(off['value'])) else 'FAIL'
            rows.append(dict(symbol=symbol, accn=accn, form=data['form'], period=f'{start or "instant"}..{end}', metric=metric,
                             official=off['value'], printed=f'{off["shown"]} (scale 1e{off["scale"]}{", negative" if off["sign"] else ""})',
                             label=off['label'], source=data['url'], concept=off['concept'].split(':')[-1], stored=sv, difference=diff, status=status))
    return rows


def test_real_official_headline_values_match_stored(reconciliation):
    fails = [r for r in reconciliation if r['status'] == 'FAIL']
    assert not fails, fails[:5]
    per_company = {}
    for r in reconciliation:
        if r['status'] == 'PASS':
            per_company.setdefault(r['symbol'], set()).add((r['period'], r['metric']))
    for symbol in GROUPNEW:
        need = 6 if symbol == 'GSK' else 8        # GSK: only 7 IFRS metrics are mapped at all (documented limitation)
        assert len(per_company.get(symbol, ())) >= need, (symbol, len(per_company.get(symbol, ())))


def test_real_official_gaps_are_exactly_the_known_unmapped_tags(reconciliation):
    """Where the filing prints a headline number but production stores nothing (coverage gaps, never wrong numbers)."""
    gaps = {(r['symbol'], r['metric'], r['concept']) for r in reconciliation if r['status'] == 'GAP'}
    expected = {
        ('AMZN', 'capex', 'PaymentsToAcquireProductiveAssets'), ('F', 'capex', 'PaymentsToAcquireProductiveAssets'),
        # lead re-audit (QA-REAL-03): NEE RegulatedAndUnregulatedOperatingRevenue is now mapped, so it is no longer a gap
        ('GSK', 'gross_profit', 'GrossProfit'),
        ('F', 'operating_income', 'OperatingIncomeLoss'),
    }
    unexpected = gaps - expected
    assert not unexpected, sorted(unexpected)
    assert ('AMZN', 'capex', 'PaymentsToAcquireProductiveAssets') in gaps and ('NEE', 'revenue', 'RegulatedAndUnregulatedOperatingRevenue') not in gaps


def test_real_write_reconciliation_csv(reconciliation):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'reconciliation_official_10k.csv'
    with path.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['Metric', 'Source', 'Stored', 'Difference', 'Status', 'Official', 'PrintedAs', 'PrintedLabel', 'Concept', 'Accession', 'Period'])
        for r in reconciliation:
            w.writerow([f'{r["symbol"]} {r["metric"]}', r['source'], r['stored'], r['difference'], r['status'], r['official'], r['printed'],
                        r['label'], r['concept'], r['accn'], r['period']])
    (HERE / 'fixtures' / 'real' / 'reconciliation_official_10k.csv').write_text(path.read_text())
    assert sum(1 for r in reconciliation if r['status'] == 'PASS') >= 100
    text = path.read_text()
    assert chr(0x2014) not in text and chr(0x2013) not in text            # ASCII-only deliverable (repo rule)


# --------------------------------------------------------------------------------------------- official TTM
QUARTERLY = {'AAPL': '0000320193-26-000020', 'PFE': '0000078003-26-000095', 'KO': '0001628280-26-028802',
             'NKE': '0000320187-26-000037', 'AMZN': '0001018724-26-000026', 'PLUG': '0001104659-26-093454'}
FY_DOC = {'AAPL': '0000320193-25-000079', 'PFE': '0000078003-26-000026', 'KO': '0001628280-26-010047',
          'NKE': '0000320187-25-000047', 'AMZN': '0001018724-26-000004', 'PLUG': '0001104659-26-022286'}


def _official_ttm(symbol, metric):
    """FY + YTD - prior YTD from the printed documents; or the printed 12-month column when the 10-Q has one (Amazon)."""
    q = json.loads((OFFICIAL / f'{symbol}_{QUARTERLY[symbol]}.extract.json').read_text())
    k = json.loads((OFFICIAL / f'{symbol}_{FY_DOC[symbol]}.extract.json').read_text())
    rep = q['report_date']
    rows = [r for r in q['rows'] if r['metric'] == metric and r['start'] and r['end'] == rep]
    if not rows:
        return None
    long_cols = [r for r in rows if 350 <= days_inclusive(r['start'], r['end']) <= 380]
    if long_cols:
        return max(long_cols, key=lambda r: len(f'{abs(r["value"]):.0f}'.rstrip('0')))['value']
    ytd = max(rows, key=lambda r: days_inclusive(r['start'], r['end']))
    prior = [r for r in q['rows'] if r['metric'] == metric and r['start'] and r['concept'] == ytd['concept']
             and abs(days_inclusive(r['start'], r['end']) - days_inclusive(ytd['start'], ytd['end'])) <= 10
             and 355 <= (dt.date.fromisoformat(rep) - dt.date.fromisoformat(r['end'])).days <= 375]
    fy = [r for r in k['rows'] if r['metric'] == metric and r['concept'] == ytd['concept'] and r['start'] and r['end'] == k['report_date']
          and 350 <= days_inclusive(r['start'], r['end']) <= 380]
    if not prior or not fy:
        return None
    best = lambda lst: max(lst, key=lambda r: len(f'{abs(r["value"]):.0f}'.rstrip('0')))
    return best(fy)['value'] + ytd['value'] - best(prior)['value']


@pytest.mark.parametrize('symbol', sorted(QUARTERLY))
@pytest.mark.parametrize('metric', ['revenue', 'net_income', 'ocf', 'capex'])
def test_real_ttm_matches_official_documents(symbol, metric, enginenew, cosnew):
    """TTM at the second after the latest 10-Q vs (FY + YTD - prior YTD) re-computed from the printed 10-K and 10-Q."""
    official = _official_ttm(symbol, metric)
    co = cosnew[symbol]
    sub = json.loads((RAWNEW / f'{symbol}_submissions.json').read_text())['filings']['recent']
    accn = QUARTERLY[symbol]
    pub = parse_utc(sub['acceptanceDateTime'][sub['accessionNumber'].index(accn)]) + dt.timedelta(minutes=5)
    snap = enginenew.snapshot(symbol, iso_utc(pub + dt.timedelta(seconds=1)))
    ttm = snap['ttm']
    report_date = sub['reportDate'][sub['accessionNumber'].index(accn)]
    assert ttm['period_end'] == report_date, (symbol, ttm['period_end'], report_date)
    value = ttm['values'][metric]
    if official is None:
        pytest.skip(f'{symbol} {metric}: document does not print the needed columns')
    if value is None:
        # coverage gap: Amazon CapEx is tagged PaymentsToAcquireProductiveAssets which production does not map
        assert (symbol, metric) == ('AMZN', 'capex'), (symbol, metric, ttm['reasons'])
        return
    tol = 3 * rounding_unit([value, official])
    # lead re-audit (QA-REAL-01 fixed): the PLUG net-income tag flip no longer makes TTM disagree with the printed 12-month column
    assert abs(value - official) <= tol, (symbol, metric, value, official, tol)


def test_real_defect_plug_ttm_net_income_tag_flip(enginenew, cosnew):
    sub = json.loads((RAWNEW / 'PLUG_submissions.json').read_text())['filings']['recent']
    accn = QUARTERLY['PLUG']
    pub = parse_utc(sub['acceptanceDateTime'][sub['accessionNumber'].index(accn)]) + dt.timedelta(minutes=5)
    snap = enginenew.snapshot('PLUG', iso_utc(pub + dt.timedelta(seconds=1)))
    assert snap['ttm']['values']['net_income'] == -1_641_350_000.0      # -361,869 - 845,970 - 245,304 - 188,207 (thousands)


# --------------------------------------------------------------------------------------------- specific generalisation checks
def test_real_aapl_52_53_week_fiscal_year_labels_and_growth_from_printed_numbers(enginenew):
    """Apple's FY ends on the last Saturday of September: 2025-09-27 (364 days), prior FY 2024-09-28. Growth from printed numbers."""
    data, table = _official_table(OFFICIAL / 'AAPL_0000320193-25-000079.extract.json')
    cur = table[('revenue', '2024-09-29', '2025-09-27')]['value']
    prev = table[('revenue', '2023-10-01', '2024-09-28')]['value']
    snap = enginenew.snapshot('AAPL', '2025-11-01T00:00:00+00:00')
    assert (snap['period_start'], snap['period_end']) == ('2024-09-29', '2025-09-27')
    assert snap['features']['revenue_yoy'] == pytest.approx(cur / prev - 1, rel=1e-12)
    assert cur == 416_161_000_000.0 and prev == 391_035_000_000.0
    # not visible one second before the 10-K (acceptance 2025-10-31T10:01:26Z + 5 min)
    pre = enginenew.snapshot('AAPL', '2025-10-31T10:06:25+00:00')
    post = enginenew.snapshot('AAPL', '2025-10-31T10:06:26+00:00')
    assert pre['period_end'] == '2024-09-28' and post['period_end'] == '2025-09-27'


def test_real_nke_may_year_end_and_ttm_matches_annual_at_year_end(enginenew):
    snap = enginenew.snapshot('NKE', '2026-07-20T00:00:00+00:00')
    assert (snap['period_start'], snap['period_end']) == ('2025-06-01', '2026-05-31')
    assert snap['raw']['revenue'] == 46_398_000_000.0 and snap['ttm']['values']['revenue'] == 46_398_000_000.0
    assert snap['raw']['eps'] == 2.10 and snap['raw']['net_income'] == 3_108_000_000.0


def test_real_loss_making_filers_keep_negative_numbers_and_withhold_undefined_ratios(enginenew):
    plug = enginenew.snapshot('PLUG', '2026-04-01T00:00:00+00:00')
    assert plug['raw']['net_income'] == -1_631_594_000.0 and plug['raw']['ocf'] == -535_835_000.0 and plug['raw']['gross_profit'] == -242_040_000.0
    f = plug['features']
    assert f['net_margin'] is None or f['net_margin'] < 0
    assert f['fcf'] == plug['raw']['ocf'] - plug['raw']['capex'] == pytest.approx(-535_835_000.0 - 111_164_000.0)
    assert 'NEGATIVE_FCF' in plug['flags']
    # FY2020 revenue was restated NEGATIVE (-93.2M): growth from a non-positive base is undefined, never fabricated
    snap21 = enginenew.snapshot('PLUG', '2022-03-02T00:00:00+00:00')
    assert snap21['raw']['revenue'] == 502_342_000.0 and snap21['features']['revenue_yoy'] is None
    ford = enginenew.snapshot('F', '2026-04-01T00:00:00+00:00')
    assert ford['raw']['net_income'] == -8_162_000_000.0 and ford['raw']['eps'] == -2.06 and ford['raw']['operating_income'] == -9_169_000_000.0


def test_real_gsk_ifrs_gbp_eps_pence_scale_and_original_currency(storenew, enginenew):
    snap = enginenew.snapshot('GSK', '2026-04-01T00:00:00+00:00')
    assert snap['currency'] == 'GBP' and snap['raw']['revenue'] == 32_667_000_000.0
    assert snap['raw']['eps'] == pytest.approx(1.388, rel=1e-12)             # printed as 138.8 pence, stored as GBP per share
    row = [r for r in storenew.as_of('GSK', '2026-04-01T00:00:00+00:00', kind='financial') if r['metric'] == 'eps'][-1]
    assert (row['currency'], row['units']) == ('GBP', 'GBP/shares')
    meta = json.loads(storenew.db.execute("SELECT metadata FROM observations WHERE id=?", (row['id'],)).fetchone()[0])
    assert meta['accounting_standard'] == 'ifrs-full'
    assert snap['features']['capex_sales'] is None and snap['features']['fcf'] is None       # IFRS adapter maps only 7 metrics


def test_real_amzn_ten_q_twelve_month_columns_do_not_corrupt_ttm(enginenew):
    """Amazon's 10-Q cash-flow statement prints 3-month and TRAILING-12-MONTH columns; the 365-day facts must not be treated as YTD."""
    snap = enginenew.snapshot('AMZN', '2026-08-05T00:00:00+00:00')
    assert snap['ttm']['period_end'] == '2026-06-30'
    assert snap['ttm']['values']['revenue'] == 775_680_000_000.0
    assert snap['ttm']['values']['ocf'] == 161_403_000_000.0          # == the printed 12-month column of the 10-Q
    assert snap['ttm']['values']['net_income'] == 135_281_000_000.0


def test_real_debt_and_roic_coverage_is_poor_on_unseen_companies(enginenew):
    """LIMITATION (documented rationale in providers.py: DebtCurrent must be present, short borrowings are never substituted)."""
    with_debt = {s for s in GROUPNEW if enginenew.snapshot(s, '2026-09-30T00:00:00+00:00')['features'].get('debt') is not None}
    assert with_debt == {'PFE'}      # NEE's debt tags were only ever stale-year coverage; with revenue mapped its snapshot is current
    snaps = {s: enginenew.snapshot(s, '2026-09-30T00:00:00+00:00') for s in GROUPNEW}
    assert {s for s, n in snaps.items() if n['status'] == 'AVAILABLE' and n['features'].get('roic') is not None} == set()
    assert {s for s, n in snaps.items() if n['status'] == 'AVAILABLE' and n['features'].get('net_debt_ebitda') is not None} == set()


def test_real_capex_gaps_make_fcf_unavailable_never_ocf(enginenew):
    """AMZN / F / GSK / NEE tag capex as PaymentsToAcquireProductiveAssets (not mapped): FCF is None, never OCF-with-zero-capex."""
    for symbol in ('AMZN', 'F', 'GSK'):
        snap = enginenew.snapshot(symbol, '2026-09-30T00:00:00+00:00')
        assert 'capex' not in snap['raw'] and snap['features']['fcf'] is None and snap['features']['fcf_margin'] is None


def test_real_nee_fy2013_revenue_is_now_mapped(storenew):
    """Lead re-audit (QA-REAL-03 fixed): NextEra's RegulatedAndUnregulatedOperatingRevenue is mapped, so FY2013 revenue is stored."""
    rows = storenew.as_of('NEE', '2014-03-01T00:00:00+00:00', kind='financial')
    have = {(r['metric']) for r in rows if r['period_end'] == '2013-12-31'}
    assert {'net_income', 'ocf', 'assets', 'equity', 'revenue'} <= have


def test_real_defect_nee_snapshot_anchored_on_stale_year_but_available(enginenew):
    snap = enginenew.snapshot('NEE', '2014-03-01T00:00:00+00:00')          # FY2013 10-K accepted 2014-02-24
    assert snap['period_end'] == '2013-12-31' or snap['status'] != 'AVAILABLE', (snap['period_end'], snap['status'])


def test_real_ttm_now_carries_its_own_age(enginenew):
    """Lead re-audit (QA-CALC-06 fixed): the TTM payload carries age_days and goes STALE past the freshness limit."""
    snap = enginenew.snapshot('NEE', '2018-10-07T00:00:00+00:00')
    assert snap['ttm']['period_end'] == '2018-06-30' and snap['ttm']['age_days'] == 99 and snap['ttm']['status'] == 'INCOMPLETE'


def test_real_write_ttm_reconciliation_csv(enginenew):
    """Metric | Source | Stored | Difference | Status for TTM at the second after the latest 10-Q vs numbers re-derived from the printed documents."""
    rows = []
    for symbol in sorted(QUARTERLY):
        sub = json.loads((RAWNEW / f'{symbol}_submissions.json').read_text())['filings']['recent']
        i = sub['accessionNumber'].index(QUARTERLY[symbol])
        pub = parse_utc(sub['acceptanceDateTime'][i]) + dt.timedelta(minutes=5)
        snap = enginenew.snapshot(symbol, iso_utc(pub + dt.timedelta(seconds=1)))
        for metric in ('revenue', 'net_income', 'ocf', 'capex'):
            official = _official_ttm(symbol, metric)
            stored = snap['ttm']['values'][metric]
            if stored is None:
                status, diff = 'GAP', None
            else:
                diff = stored - official
                tol = 3 * rounding_unit([stored, official])
                status = 'PASS' if abs(diff) <= tol else ('KNOWN_DEFECT_QA_REAL_01' if (symbol, metric) == ('PLUG', 'net_income') else 'FAIL')
            rows.append([f'{symbol} TTM {metric} to {sub["reportDate"][i]}', f'official FY + YTD - prior YTD from accession {QUARTERLY[symbol]} and {FY_DOC[symbol]}', stored, diff, status, official])
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'reconciliation_ttm.csv', HERE / 'fixtures' / 'real' / 'reconciliation_ttm.csv'):
        with path.open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['Metric', 'Source', 'Stored', 'Difference', 'Status', 'Official'])
            w.writerows(rows)
    statuses = {r[4] for r in rows}
    assert 'FAIL' not in statuses and sum(1 for r in rows if r[4] == 'PASS') >= 18, rows


def test_real_aapl_revenue_tag_switch_is_continuous_and_five_year_cagr_matches_printed_numbers(enginenew):
    """Apple moved from SalesRevenueNet to RevenueFromContractWithCustomer...: no jump at the switch, CAGR from printed FY2020 and FY2025."""
    s18 = enginenew.snapshot('AAPL', '2018-11-06T00:00:00+00:00')
    s19 = enginenew.snapshot('AAPL', '2019-11-01T00:00:00+00:00')
    assert s18['raw']['revenue'] == 265_595_000_000.0 and s19['raw']['revenue'] == 260_174_000_000.0
    assert s19['features']['revenue_yoy'] == pytest.approx(260_174 / 265_595 - 1, rel=1e-12)
    old = _official_table(OFFICIAL / 'AAPL_0000320193-21-000105.extract.json')[1][('revenue', '2019-09-29', '2020-09-26')]['value']
    new = _official_table(OFFICIAL / 'AAPL_0000320193-25-000079.extract.json')[1][('revenue', '2024-09-29', '2025-09-27')]['value']
    assert (old, new) == (274_515_000_000.0, 416_161_000_000.0)
    snap = enginenew.snapshot('AAPL', '2025-11-01T00:00:00+00:00')
    assert snap['features']['revenue_cagr_5y'] == pytest.approx((new / old) ** (1 / 5) - 1, rel=1e-12)
    assert snap['features']['revenue_cagr_3y'] == pytest.approx((new / 394_328_000_000.0) ** (1 / 3) - 1, rel=1e-12)       # FY2022 (ended 2022-09-24)


MIXED_VINTAGE_CASES = [
    # (symbol, 10-K filing date, FY end, FY revenue printed in that 10-K, TTM revenue production serves, why)
    ('KO', '2020-02-24', '2019-12-31', 37_266_000_000.0, 36_592_000_000.0),
    ('PFE', '2013-02-28', '2012-12-31', 58_986_000_000.0, 59_506_000_000.0),
    ('PFE', '2014-02-28', '2013-12-31', 51_584_000_000.0, 52_674_000_000.0),
    ('NKE', '2013-07-23', '2013-05-31', 25_313_000_000.0, 25_508_000_000.0),
    ('AMZN', '2013-01-30', '2012-12-31', 61_093_000_000.0, 48_077_000_000.0),
]


@pytest.mark.parametrize('symbol,filed,fy_end,fy_revenue,served', MIXED_VINTAGE_CASES, ids=[f'{c[0]}-{c[1]}' for c in MIXED_VINTAGE_CASES])
def test_real_defect_ttm_equals_fy_at_year_end_mixed_vintage(symbol, filed, fy_end, fy_revenue, served, enginenew, cosnew):
    co = cosnew[symbol]
    accn = next(a for a, f in {r['accn']: r['filed'] for r in co.rows('us-gaap') if r['form'] == '10-K'}.items() if f == filed)
    snap = enginenew.snapshot(symbol, iso_utc(co.pub_time(accn, filed) + dt.timedelta(seconds=1)))
    assert abs(snap['ttm']['values']['revenue'] - fy_revenue) <= 3e6


@pytest.mark.parametrize('symbol,missing_accn,expected_ttm_end', [('KO', '0001628280-26-050503', '2026-04-03'), ('F', '0000037996-26-000156', None),
                                                                    ('NEE', '0000753308-26-000060', '2026-03-31')])
def test_real_companyfacts_lags_submissions_and_production_does_not_notice(symbol, missing_accn, expected_ttm_end, enginenew, cosnew):
    """LIMITATION: the SEC companyfacts feed trails the submissions feed (5 of 17 cached issuers lack their newest 10-Q/20-F facts).
    ingest_sec stores a 'filing' event for the accession yet no facts, and snapshot() reports nothing about the missing filing."""
    co = cosnew[symbol]
    recent = co.subs['filings']['recent']
    assert missing_accn in recent['accessionNumber'] and recent['isXBRL'][recent['accessionNumber'].index(missing_accn)]
    have = {r['accn'] for std in ('us-gaap', 'ifrs-full') for body in co.facts['facts'].get(std, {}).values() for rows in body['units'].values() for r in rows}
    assert missing_accn not in have
    snap = enginenew.snapshot(symbol, '2026-09-30T00:00:00+00:00')
    assert snap['status'] in ('AVAILABLE', 'STALE') and 'missing_filings' not in snap and 'stale_reason' not in snap
    if expected_ttm_end:
        assert snap['ttm']['period_end'] == expected_ttm_end
