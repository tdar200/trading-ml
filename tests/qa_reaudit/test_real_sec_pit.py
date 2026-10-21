"""QA track B: real SEC companyfacts ingestion vs an independent human-view oracle (8 audited companies).

Raw evidence: results/fundamental_audit_20260930/raw/*_facts.json + *_submissions.json (read only).
Production path under test: providers.sec_facts -> store.ingest_validated -> store.as_of -> FundamentalFeatureEngine.snapshot
(and FundamentalAPI.get).  The oracle (tests/qa_reaudit/real_oracle.py) imports nothing from fundamentals.*.

Defect repros are strict xfail (see QA-REAL-01 / QA-REAL-02) so they flip loudly when fixed.
"""
from __future__ import annotations
import datetime as dt
import json
import random

import pytest

from fundamentals.api import FundamentalAPI
from fundamentals.features import FundamentalFeatureEngine
from tests.qa_reaudit import real_db
from tests.qa_reaudit.real_db import GROUP8, compare, fixed_dates, periodic_filings, random_instant, rounding_unit
from tests.qa_reaudit.real_oracle import RAW8, UTC, days_inclusive, iso_utc, load_company, parse_utc

SEEDS = range(30)


@pytest.fixture(scope='module')
def store8():
    path = real_db.db8()
    store = real_db.open_ro(path)
    yield store
    store.close()


@pytest.fixture(scope='module')
def engine8(store8):
    return FundamentalFeatureEngine(store8)


@pytest.fixture(scope='module')
def cos():
    return {s: load_company(s) for s in GROUP8}


# ---------------------------------------------------------------------------------------------------
# Ingestion fidelity (Phases 5-7, 90-91): every stored number is exactly the raw SEC number
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('symbol', GROUP8)
def test_real_ingest_counts_no_silent_drop(symbol, store8):
    """accepted + rejected == candidates and every accepted fact is physically stored (no silent loss)."""
    real_db.db8()
    info = real_db.INGEST_SUMMARY[symbol]
    assert info['accepted'] + info['rejected'] == info['candidates']
    stored = store8.db.execute('SELECT COUNT(*) FROM observations WHERE symbol=?', (symbol,)).fetchone()[0]
    assert stored == info['accepted'], f'{symbol}: stored {stored} != accepted {info["accepted"]}'
    rejected = store8.db.execute('SELECT reason, payload FROM ingestion_rejections WHERE symbol=?', (symbol,)).fetchall()
    assert len(rejected) == info['rejected']
    if symbol == 'WMT':
        # one mislabelled XBRL context (period_end 2012-12-31 inside a filing made 2012-03-27) is quarantined verbatim
        assert len(rejected) == 1 and rejected[0]['reason'] == 'observation precedes measured period'
        payload = json.loads(rejected[0]['payload'])
        assert payload['period_end'] == '2012-12-31' and payload['value'] == 6600000000.0
        assert payload['metric'] == 'cash' and payload['revision'] == '0001193125-12-134679'
    else:
        assert not rejected


@pytest.mark.parametrize('symbol', GROUP8)
def test_real_every_stored_fact_equals_raw_json(symbol, store8, cos):
    """Value, period, accession, unit, form, and availability time of EVERY stored observation re-derived from raw JSON."""
    co = cos[symbol]
    index = {}
    for standard in ('us-gaap', 'ifrs-full'):
        for tag, body in co.facts['facts'].get(standard, {}).items():
            for unit, rows in body['units'].items():
                for r in rows:
                    index[(standard, tag, unit, r.get('start', ''), r['end'], r['accn'])] = r
    n = 0
    for row in store8.db.execute("SELECT * FROM observations WHERE symbol=? AND kind='financial'", (symbol,)):
        meta = json.loads(row['metadata'])
        standard = meta['accounting_standard']
        key = (standard, meta['tag'], row['units'] if row['units'] != row['currency'] else row['units'], row['period_start'], row['period_end'], row['revision'])
        raw = index.get(key)
        assert raw is not None, f'stored fact with no raw source row: {key}'
        assert row['value'] == float(raw['val']), (key, row['value'], raw['val'])
        assert meta['form'] == raw['form'] and meta['filing_date'] == raw['filed']
        pub = co.pub_time(raw['accn'], raw['filed'])
        assert row['available_at'] == pub.isoformat(), (key, row['available_at'], pub)
        # never earlier than the true dissemination lower bound (acceptance time, or 00:00 New York of the filing date)
        assert parse_utc(row['available_at']) >= co.lower_bound_time(raw['accn'], raw['filed'])
        n += 1
    assert n > 500


@pytest.mark.parametrize('symbol', GROUP8)
def test_real_headline_metrics_complete_for_every_filing_period(symbol, store8, cos):
    """Every (accession, period) that has a headline revenue/net income/OCF/assets fact in the raw JSON has one stored."""
    co = cos[symbol]
    standards = ['us-gaap'] + (['ifrs-full'] if symbol == 'SONY' else [])
    expected = set()
    for standard in standards:
        for r in co.rows(standard):
            if r['metric'] in ('revenue', 'net_income', 'ocf', 'assets', 'equity') and r['unit'] in ('USD', 'JPY'):
                expected.add((r['metric'], r['accn'], r['start'], r['end']))
    got = {(r['metric'], r['revision'], r['period_start'], r['period_end'])
           for r in store8.db.execute("SELECT metric,revision,period_start,period_end FROM observations WHERE symbol=? AND value IS NOT NULL", (symbol,))}
    missing = sorted(expected - got)
    # the single WMT row quarantined for a mislabelled context is cash, not a headline metric here
    assert not missing, f'{symbol}: {len(missing)} raw headline facts missing, e.g. {missing[:3]}'


def test_real_non_periodic_forms_never_ingested(store8, cos):
    """8-K / 6-K / DEF 14A XBRL rows exist in companyfacts but production deliberately ignores them (documented limitation)."""
    forms = {r[0] for r in store8.db.execute("SELECT DISTINCT json_extract(metadata,'$.form') FROM observations")}
    assert forms <= {'10-K', '10-Q', '10-K/A', '10-Q/A', '20-F', '20-F/A'}, forms
    excluded = sum(1 for s in ('MSFT', 'JPM', 'CAT') for std in ('us-gaap',) for body in cos[s].facts['facts'][std].values()
                   for rows in body['units'].values() for r in rows if r['form'] == '8-K')
    assert excluded > 1000  # recast statements filed on 8-K are invisible to the pipeline


# ---------------------------------------------------------------------------------------------------
# Random differential vs the human view (>= 240 random (company, T) pairs + fixed 2018-2025 grid)
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('symbol', GROUP8)
def test_real_random_pair_matches_human_view(symbol, seed, engine8, cos):
    rng = random.Random(f'{symbol}-{seed}')
    T = random_instant(rng)
    issues, snap = compare(engine8, cos[symbol], symbol, T)
    assert not issues, f'{symbol} seed={seed} T={iso_utc(T)}: {issues}'
    # lineage leak check: nothing used by the snapshot was published after T
    ids = snap['lineage']
    if ids:
        marks = ','.join('?' * len(ids))
        latest = engine8.store.db.execute(f'SELECT MAX(available_at) FROM observations WHERE id IN ({marks})', ids).fetchone()[0]
        assert parse_utc(latest) <= T, f'LEAK {symbol} seed={seed}: lineage available_at {latest} > T {iso_utc(T)}'


@pytest.mark.parametrize('symbol', GROUP8)
def test_real_fixed_date_grid_matches_human_view(symbol, engine8, cos):
    problems = []
    for T in fixed_dates():
        issues, _ = compare(engine8, cos[symbol], symbol, T)
        problems += [f'{iso_utc(T)}: {i}' for i in issues]
    assert not problems, problems[:8]


# ---------------------------------------------------------------------------------------------------
# Publication boundary on real filings: second-exact, through store.as_of AND snapshot
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('symbol', GROUP8)
def test_real_publication_boundary_exact(symbol, store8, engine8, cos):
    co = cos[symbol]
    standards = ['us-gaap'] + (['ifrs-full'] if symbol == 'SONY' else [])
    filings = periodic_filings(co, standards, forms=('10-K', '10-Q', '20-F'))
    rng = random.Random(f'boundary-{symbol}')
    chosen = rng.sample(sorted(filings.items(), key=lambda kv: kv[1]), 12)
    for accn, filed in chosen:
        pub = co.pub_time(accn, filed)
        before, at = pub - dt.timedelta(microseconds=1), pub
        seen_before = {r['revision'] for r in store8.as_of(symbol, iso_utc(before), kind='financial')}
        seen_at = {r['revision'] for r in store8.as_of(symbol, iso_utc(at), kind='financial')}
        assert accn not in seen_before, f'{symbol} {accn} visible 1us before publication {iso_utc(pub)}'
        assert accn in seen_at, f'{symbol} {accn} invisible at publication {iso_utc(pub)}'
        # one microsecond before: snapshot must equal the oracle that excludes the filing; at pub: includes it
        for T in (before, at):
            issues, _ = compare(engine8, co, symbol, T)
            assert not issues, f'{symbol} {accn} T={iso_utc(T)}: {issues}'


def test_real_after_hours_acceptance_is_not_delayed_to_next_day(store8, cos):
    """Facts with a known EDGAR acceptance time become visible at acceptance+5min (not held to the SEC 'filing date' or the next
    local day); facts of older filings without an acceptance time fall back to the following New York midnight (documented)."""
    co = cos['BA']
    accn = next(a for a, f in periodic_filings(co, forms=('10-K',)).items() if a in co.acceptance and f >= '2023-01-01')
    filed = periodic_filings(co, forms=('10-K',))[accn]
    acc = parse_utc(co.acceptance[accn])
    rows = [r for r in store8.as_of('BA', iso_utc(acc + dt.timedelta(minutes=5)), kind='financial') if r['revision'] == accn]
    assert rows, 'accession with acceptance time must be usable exactly acceptance+5min'
    assert not [r for r in store8.as_of('BA', iso_utc(acc + dt.timedelta(minutes=5) - dt.timedelta(seconds=1)), kind='financial') if r['revision'] == accn]
    assert all(r['metadata']['timing_precision'] == 'acceptance+5min' for r in rows)
    # fallback rule for older filings: date only -> following local (New York) midnight
    old = next(a for a, f in periodic_filings(co, forms=('10-K',)).items() if a not in co.acceptance and f >= '2012-01-01')
    meta = next(json.loads(r['metadata']) for r in store8.db.execute('SELECT metadata FROM observations WHERE symbol=? AND revision=? LIMIT 1', ('BA', old)))
    assert meta['timing_precision'] == 'following_local_day'


# ---------------------------------------------------------------------------------------------------
# Self-consistency invariants of real data: TTM at fiscal-year-end must equal the 10-K's own FY number
# ---------------------------------------------------------------------------------------------------
def _ttm_vs_fy_mismatches(engine, co, symbol, standards):
    """[(filed, metric, ttm, fy)] where TTM at the 10-K publication second differs from FY beyond rounding."""
    out = []
    for accn, filed in sorted(periodic_filings(co, standards, forms=('10-K', '20-F')).items(), key=lambda kv: kv[1]):
        if filed < '2011-01-01':
            continue
        T = co.pub_time(accn, filed) + dt.timedelta(seconds=1)
        snap = engine.snapshot(symbol, iso_utc(T))
        if snap['status'] != 'AVAILABLE' or snap['ttm']['period_end'] != snap['period_end']:
            continue
        for metric in ('revenue', 'net_income', 'ocf', 'capex'):
            tv, fv = snap['ttm']['values'][metric], snap['raw'].get(metric)
            if tv is None or fv is None:
                continue
            tol = 3 * rounding_unit([tv, fv])
            if abs(tv - fv) > tol:
                out.append((filed, metric, tv, fv))
    return out


# (company, 10-K filing date, metric) already reported as QA-REAL-01 / 02; excluded from the clean-invariant sweep below
KNOWN_TTM_FY_BREAKS = set()   # lead re-audit: QA-REAL-01/02 fixed - TTM now equals the reported FY at every year end of the 8 companies
# (before the fix: WMT 2011/2012 revenue, WMT 2021-2024 net income, XOM 2020 revenue)


@pytest.mark.parametrize('symbol', GROUP8)
def test_real_ttm_equals_fiscal_year_at_10k_publication(symbol, engine8, cos):
    """TTM built from four discrete quarters (Q4 = FY - 9M) must reproduce the FY figure the 10-K itself prints."""
    standards = ['us-gaap'] + (['ifrs-full'] if symbol == 'SONY' else [])
    bad = [m for m in _ttm_vs_fy_mismatches(engine8, cos[symbol], symbol, standards) if (symbol, m[0], m[1]) not in KNOWN_TTM_FY_BREAKS]
    assert not bad, f'{symbol}: unexpected TTM != FY at 10-K publication: {bad[:5]}'


def _snapshot_after_10k(engine, co, symbol, filed):
    accn = next(a for a, f in periodic_filings(co, forms=('10-K',)).items() if f == filed)
    return engine.snapshot(symbol, iso_utc(co.pub_time(accn, filed) + dt.timedelta(seconds=1)))


def test_real_defect_ttm_net_income_mixes_parent_and_nci_definitions_wmt(engine8, cos):
    snap = _snapshot_after_10k(engine8, cos['WMT'], 'WMT', '2024-03-15')
    assert snap['raw']['net_income'] == 15_511_000_000.0          # FY2024 attributable to Walmart, as printed in the 10-K
    assert snap['ttm']['period_end'] == snap['period_end'] == '2024-01-31'
    assert snap['ttm']['values']['net_income'] == 15_511_000_000.0


def test_real_defect_ttm_revenue_mixes_revenue_tags_within_one_filing_wmt(engine8, cos):
    snap = _snapshot_after_10k(engine8, cos['WMT'], 'WMT', '2011-03-30')
    assert snap['raw']['revenue'] == 421_849_000_000.0
    assert snap['ttm']['values']['revenue'] == snap['raw']['revenue']


def test_real_defect_ttm_mixed_vintage_quarters_xom(engine8, cos):
    snap = _snapshot_after_10k(engine8, cos['XOM'], 'XOM', '2020-02-26')
    assert snap['raw']['revenue'] == 264_938_000_000.0
    assert snap['ttm']['values']['revenue'] == 264_938_000_000.0


def test_real_known_ttm_fy_breaks_are_exactly_the_documented_ones(engine8, cos):
    """The allow-list above matches reality (so the clean sweep is meaningful) and every entry is a genuine, non-rounding break."""
    found = set()
    for symbol in GROUP8:
        standards = ['us-gaap'] + (['ifrs-full'] if symbol == 'SONY' else [])
        for filed, metric, tv, fv in _ttm_vs_fy_mismatches(engine8, cos[symbol], symbol, standards):
            found.add((symbol, filed, metric))
    assert found == KNOWN_TTM_FY_BREAKS, (sorted(found - KNOWN_TTM_FY_BREAKS), sorted(KNOWN_TTM_FY_BREAKS - found))


def test_real_wmt_net_income_flip_is_a_pure_tag_swap(store8, engine8, cos):
    """Evidence for QA-REAL-01: the WMT quarter 2023-05-01..2023-07-31 is 7,891M (NetIncomeLoss) in every filing that tags it
    that way, but production serves 8,053M (ProfitLoss, includes non-controlling interests) because the 2023-09-01 and later
    accessions carry only ProfitLoss for that quarter."""
    rows = store8.history('WMT', 'net_income', '2024-03-16T00:00:00+00:00')
    q = [r for r in rows if r['period_start'] == '2023-05-01' and r['period_end'] == '2023-07-31']
    tags = {(json.loads(r['metadata'])['tag'], r['value']) for r in q}
    assert tags == {('NetIncomeLoss', 7_891_000_000.0), ('ProfitLoss', 8_053_000_000.0)}
    served = [r for r in store8.as_of('WMT', '2024-03-16T00:00:00+00:00', kind='financial')
              if r['metric'] == 'net_income' and r['period_start'] == '2023-05-01' and r['period_end'] == '2023-07-31']
    assert len(served) == 1 and served[0]['value'] == 8_053_000_000.0 and served[0]['metadata']['tag'] == 'ProfitLoss'


# ---------------------------------------------------------------------------------------------------
# Specific real-data scenarios
# ---------------------------------------------------------------------------------------------------
def test_real_sony_us_gaap_to_ifrs_switch_uses_correct_standard_and_original_currency(engine8, store8):
    before = engine8.snapshot('SONY', '2021-06-23T00:00:00+00:00')       # after the last US GAAP 20-F (2021-06-22)
    after = engine8.snapshot('SONY', '2022-06-29T00:00:00+00:00')        # after the first IFRS 20-F (2022-06-28)
    assert before['currency'] == after['currency'] == 'JPY'              # reported currency retained, never silently converted
    assert before['raw']['revenue'] == 8_999_360_000_000.0 and before['raw']['eps'] == 936.9      # US GAAP FY2020
    assert before['period_end'] == '2021-03-31'
    assert after['period_end'] == '2022-03-31'                                                    # first IFRS 20-F: FY2021
    assert after['raw']['revenue'] == 9_921_513_000_000.0
    # prior-year comparison after the switch uses the IFRS recast of FY2020 (8,998,661M), never the US GAAP 8,999,360M
    assert after['features']['revenue_yoy'] == pytest.approx(9_921_513_000_000.0 / 8_998_661_000_000.0 - 1, rel=1e-12)
    rows = store8.as_of('SONY', '2022-06-29T00:00:00+00:00', kind='financial')
    fy2020 = {r['metadata']['accounting_standard']: r['value'] for r in rows if r['metric'] == 'revenue' and r['period_end'] == '2021-03-31' and r['period_start'] == '2020-04-01'}
    assert fy2020 == {'us-gaap': 8_999_360_000_000.0, 'ifrs-full': 8_998_661_000_000.0}      # both vintages stored side by side, standard kept


def test_real_scrambled_fiscal_labels_do_not_change_results(tmp_path):
    """Fiscal-year labelling (fy/fp/frame fields are famously the filing's, not the period's) must never matter."""
    from fundamentals.providers import sec_facts
    from fundamentals.store import PointInTimeFundamentalStore
    payload = json.loads((RAW8 / 'MSFT_facts.json').read_text())
    subs = json.loads((RAW8 / 'MSFT_submissions.json').read_text())
    rng = random.Random(7)
    for body in payload['facts']['us-gaap'].values():
        for rows in body['units'].values():
            for r in rows:
                r['fy'] = rng.randrange(1990, 2040)
                r['fp'] = rng.choice(['Q1', 'Q2', 'Q3', 'Q4', 'FY'])
                r.pop('frame', None)
    store = PointInTimeFundamentalStore(str(tmp_path / 'scr.sqlite'))
    store.ingest_validated(sec_facts(payload, subs, 'MSFT'))
    base = real_db.open_ro(real_db.db8())
    eng_a, eng_b = FundamentalFeatureEngine(base), FundamentalFeatureEngine(store)
    for T in fixed_dates()[::3]:
        a, b = eng_a.snapshot('MSFT', iso_utc(T)), eng_b.snapshot('MSFT', iso_utc(T))
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), iso_utc(T)
    store.close()


def test_real_msft_non_calendar_year_and_q4_derivation_numbers(engine8):
    """MSFT FY ends 30 June. FY2024 revenue 245,122M; Q4 FY24 revenue 64,727M (earnings release) = FY - 9M."""
    snap = engine8.snapshot('MSFT', '2024-08-01T00:00:00+00:00')
    assert (snap['period_start'], snap['period_end']) == ('2023-07-01', '2024-06-30')
    assert snap['raw']['revenue'] == 245_122_000_000.0 and snap['raw']['net_income'] == 88_136_000_000.0 and snap['raw']['eps'] == 11.8
    assert snap['ttm']['period_end'] == '2024-06-30' and snap['ttm']['values']['revenue'] == 245_122_000_000.0
    from fundamentals.features import quarters
    rows = engine8.store.as_of('MSFT', '2024-08-01T00:00:00+00:00', kind='financial')
    q = quarters(rows, 'revenue')
    assert q['2024-06-30']['value'] == 64_727_000_000.0
    assert q['2024-06-30']['start'] == '2024-04-01'


def test_real_fcf_is_ocf_minus_capex_exactly_once(engine8):
    snap = engine8.snapshot('MSFT', '2024-08-01T00:00:00+00:00')
    ocf, capex = snap['raw']['ocf'], snap['raw']['capex']
    assert capex > 0 and snap['features']['fcf'] == ocf - capex
    assert snap['ttm']['values']['fcf'] == snap['ttm']['values']['ocf'] - snap['ttm']['values']['capex']
    assert snap['features']['fcf_margin'] == pytest.approx((ocf - capex) / snap['raw']['revenue'], rel=1e-12)


def test_real_sec_ttm_eps_never_available_and_per_share_growth_withheld(engine8):
    """LIMITATION (fail-safe): SEC facts carry no share_basis, so TTM EPS and per-share growth are withheld for ALL real data."""
    for symbol in GROUP8:
        snap = engine8.snapshot(symbol, '2025-06-01T00:00:00+00:00')
        assert snap['ttm']['values']['eps'] is None
        assert snap['ttm']['status'] in ('INCOMPLETE', 'STALE')       # never AVAILABLE; STALE when its last quarter is older than the freshness limit
        assert snap.get('per_share_comparability') == 'UNVERIFIED_SPLIT_BASIS' or snap['status'] == 'MISSING'
        assert snap['features'].get('eps_yoy') is None and snap['features'].get('diluted_shares_yoy') is None


def test_real_api_get_matches_snapshot_and_ttm_for_real_company(store8, engine8):
    api = FundamentalAPI(store8)
    for T in ('2019-06-10T04:00:00+00:00', '2023-11-02T04:00:00+00:00', '2025-03-01T04:00:00+00:00'):
        for symbol in ('MSFT', 'CAT', 'JNJ'):
            a = api.get(f'/fundamentals/{symbol}', {'as_of': T})
            assert json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(engine8.snapshot(symbol, T), sort_keys=True)
            t = api.get(f'/fundamentals/{symbol}/ttm', {'as_of': T})
            assert t['values'] == a['ttm']['values'] and t['period_end'] == a['ttm']['period_end']
