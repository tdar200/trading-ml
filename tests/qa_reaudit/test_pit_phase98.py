"""PHASE 98 (QA track A): for random T a SECOND database physically contains only rows with available_at <= T and every output
(snapshot, TTM, history, score, events, watchlist, macro, decision incl. data_snapshot_id, UI download JSON, the Streamlit page and
the real HTTP server) must be byte-identical (canonical JSON) to the output from the full database that also holds future filings,
restatements, macro vintages and events carrying the canary 123456789123, which must never appear in any output."""
from __future__ import annotations
import json
import random
from dataclasses import replace
from datetime import timedelta

import pytest

from tests.qa_reaudit import pit_oracle as O
from tests.qa_reaudit.pit_phase98 import build_pair, make_model, outputs

from fundamentals.api import FundamentalAPI
from fundamentals.models import FundamentalScoreEngine
from fundamentals.store import PointInTimeFundamentalStore

SEEDS = list(range(320))


@pytest.mark.parametrize('seed', SEEDS, ids=lambda s: f'seed{s}')
def test_phase98_truncated_database_gives_byte_identical_outputs_and_canary_never_appears(seed):
    rng = random.Random(seed ^ 0x98)
    company = O.gen_company(seed)
    T = O.pick_times(rng, company.facts, 1)[0]
    # choose T inside the data range most of the time so there IS a future to leak
    avs = sorted({f.available for f in company.facts})
    if rng.random() < 0.8:
        T = avs[rng.randint(len(avs) // 5, len(avs) - 2)] + rng.choice([timedelta(0), timedelta(microseconds=-1), timedelta(seconds=30)])
    company, full, trunc = build_pair(seed, T)
    n_future = trunc.db.execute('select count(*) from observations where available_at>?', (T.isoformat(),)).fetchone()[0]
    assert n_future == 0
    policy = rng.choice(['filter', 'regime', 'weighted', 'selection', 'observe'])
    a = outputs(full, company.symbol, T, policy)
    b = outputs(trunc, company.symbol, T, policy)
    for name in a:
        ta, tb = (a[name], b[name]) if name == 'download_json' else (O.canon(a[name]), O.canon(b[name]))
        assert O.CANARY_TEXT not in ta, f'seed={seed} T={T.isoformat()}: canary leaked into {name}'
        assert ta == tb, f'seed={seed} T={T.isoformat()} policy={policy}: {name} differs between full DB and truncated DB'
    assert a['decision']['data_snapshot_id'] == b['decision']['data_snapshot_id']
    # the quarantine audit of a future-dated bad record must not carry its value either
    assert O.CANARY_TEXT not in O.canon(full.quality(company.symbol))
    # control: the future data really exists in the full DB (the test is sensitive)
    n_future_full = full.db.execute('select count(*) from observations where available_at>?', (T.isoformat(),)).fetchone()[0]
    assert n_future_full > 0


# ---------------------------------------------------------------------------
# PHASE 98 over the real HTTP server: every endpoint, full DB vs truncated DB, raw bytes scanned for the canary
# ---------------------------------------------------------------------------
HTTP_SEEDS = list(range(500, 512))


@pytest.mark.parametrize('seed', HTTP_SEEDS, ids=lambda s: f'seed{s}')
def test_phase98_over_real_http_every_endpoint_matches_truncated_database(seed, tmp_path):
    from urllib.parse import quote
    from tests.qa_reaudit import pit_helpers as H
    rng = random.Random(seed ^ 0x77)
    company = O.gen_company(seed)
    avs = sorted({f.available for f in company.facts})
    T = avs[rng.randint(len(avs) // 4, len(avs) - 2)] + rng.choice([timedelta(0), timedelta(microseconds=-1), timedelta(hours=7)])
    full_path, trunc_path = tmp_path / 'full.sqlite', tmp_path / 'trunc.sqlite'
    company, full, trunc = build_pair(seed, T, str(full_path), str(trunc_path))
    full.close()
    trunc.close()
    sym = company.symbol
    q = '?as_of=' + quote(T.isoformat(), safe='')
    paths = [f'/fundamentals/{sym}{q}', f'/fundamentals/{sym}/ttm{q}', f'/fundamentals/{sym}/history{q}', f'/fundamentals/{sym}/history{q}&metric=revenue',
             f'/fundamentals/{sym}/score{q}', f'/fundamentals/{sym}/events{q}', f'/fundamentals/{sym}/quality{q}', f'/watchlist/fundamental{q}', f'/macro/regime{q}',
             f'/strategies/x/explanation{q}&symbol={sym}&asset_class=equity&technical_signal=true', f'/health']
    bodies = {}
    for label, path in (('full', full_path), ('trunc', trunc_path)):
        with H.http_server(path) as get:
            bodies[label] = {p: get(p) for p in paths}
    for p in paths:
        (c1, b1, h1, raw1), (c2, b2, h2, raw2) = bodies['full'][p], bodies['trunc'][p]
        assert c1 == c2 == 200, (seed, p, c1, c2, b1 if c1 != 200 else b2)
        assert O.CANARY_TEXT.encode() not in raw1, f'seed={seed} {p}: canary in HTTP response'
        assert h1['Cache-Control'] == 'no-store'
        if '/quality' in p:
            continue  # quarantine audit is a current-ingestion view by documented design (time_scope field); values never appear
        assert O.canon(b1) == O.canon(b2), f'seed={seed} T={T.isoformat()} {p}: HTTP output differs between full and truncated database'
    # the UI download button serialises exactly the API snapshot
    snap = bodies['full'][f'/fundamentals/{sym}{q}'][1]
    assert json.loads(json.dumps(snap, indent=2, allow_nan=False)) == snap


# ---------------------------------------------------------------------------
# multi-company universe: symbols that have not published yet must not exist at T
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('seed', list(range(40)), ids=lambda s: f'seed{s}')
def test_phase98_watchlist_universe_and_ranking_do_not_depend_on_future_filings(seed):
    rng = random.Random(seed * 13 + 1)
    companies = [O.gen_company(seed * 10 + k, symbol=f'CO{k}') for k in range(3)]
    all_av = sorted({f.available for c in companies for f in c.facts})
    T = all_av[rng.randint(len(all_av) // 6, len(all_av) - 2)] + rng.choice([timedelta(0), timedelta(microseconds=-1), timedelta(minutes=3)])
    # a fourth company that publishes everything in the future
    late = O.gen_company(seed * 10 + 9, symbol='LATE')
    shift = max(timedelta(0), (T - min(f.available for f in late.facts)) + timedelta(days=1, seconds=rng.randint(0, 5000)))
    late_facts = [replace(f, reported=f.reported + shift, available=f.available + shift) for f in late.facts]
    full = PointInTimeFundamentalStore(':memory:')
    trunc = PointInTimeFundamentalStore(':memory:')
    r = random.Random(seed)
    for c in companies:
        rows = [O.to_fact(f, c.symbol, c.scale_label, r) for f in c.facts]
        full.ingest(rows)
        trunc.ingest([O.to_fact(f, c.symbol, c.scale_label, r) for f in c.facts if f.available <= T])
    full.ingest([O.to_fact(f, 'LATE', late.scale_label, r) for f in late_facts])
    api_f, api_t = FundamentalAPI(full, FundamentalScoreEngine(make_model())), FundamentalAPI(trunc, FundamentalScoreEngine(make_model()))
    ts = T.isoformat()
    wf, wt = api_f.get('/watchlist/fundamental', {'as_of': ts}), api_t.get('/watchlist/fundamental', {'as_of': ts})
    assert O.canon(wf) == O.canon(wt), f'seed={seed} T={ts}'
    assert 'LATE' not in {x['symbol'] for x in wf['candidates']}
    assert full.symbols(ts) == trunc.symbols(ts) and 'LATE' not in full.symbols(ts) and 'LATE' in full.symbols()
    assert [x['symbol'] for x in wf['candidates']] == sorted([x['symbol'] for x in wf['candidates']], key=lambda s_: next((i for i, x in enumerate(wf['candidates']) if x['symbol'] == s_)))
    # LATE must be invisible at T by construction; after its first publication it appears
    assert min(f.available for f in late_facts) > T
    first_pub = min(f.available for f in late_facts) + timedelta(seconds=1)
    assert 'LATE' in {x['symbol'] for x in api_f.get('/watchlist/fundamental', {'as_of': first_pub.isoformat()})['candidates']}


# ---------------------------------------------------------------------------
# PHASE 98 at the UI level: the Streamlit page rendered headlessly (streamlit AppTest), full DB vs truncated DB
# ---------------------------------------------------------------------------
def _fund_app():
    from dashboard.views.fundamentals import render
    render()


def _ui_dump(db_path, symbol, asof):
    from google.protobuf.json_format import MessageToDict
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_fund_app, default_timeout=180)
    at.run()
    at.text_input(key='fund_db').set_value(str(db_path))
    at.text_input(key='fund_symbol').set_value(symbol)
    at.text_input(key='fund_asof').set_value(asof)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    out = []

    def walk(node):
        p = getattr(node, 'proto', None)
        if p is not None and type(node).__name__ != 'Block' and hasattr(p, 'DESCRIPTOR'):
            out.append((type(node).__name__, MessageToDict(p)))
        ch = getattr(node, 'children', None)
        if isinstance(ch, dict):
            for c in ch.values():
                walk(c)

    walk(at.main)
    return out


@pytest.mark.parametrize('seed', list(range(600, 608)), ids=lambda s: f'seed{s}')
def test_phase98_streamlit_page_is_identical_for_full_and_truncated_database_and_never_shows_the_canary(seed, tmp_path):
    rng = random.Random(seed ^ 0x55)
    company = O.gen_company(seed)
    avs = sorted({f.available for f in company.facts})
    T = avs[rng.randint(len(avs) // 3, len(avs) - 2)] + rng.choice([timedelta(0), timedelta(microseconds=-1), timedelta(hours=5)])
    company, full, trunc = build_pair(seed, T, str(tmp_path / 'full.sqlite'), str(tmp_path / 'trunc.sqlite'))
    full.close()
    trunc.close()
    a = _ui_dump(tmp_path / 'full.sqlite', company.symbol, T.isoformat())
    b = _ui_dump(tmp_path / 'trunc.sqlite', company.symbol, T.isoformat())
    assert O.CANARY_TEXT not in json.dumps(a), f'seed={seed}: canary rendered by the page'

    def comparable(items):
        keep = []
        for kind, body in items:
            if kind == 'TextInput':
                continue  # database path differs by construction
            if kind == 'Json' and 'time_scope' in json.dumps(body):
                continue  # 'Source validation' = current ingestion audit by documented design
            keep.append((kind, body))
        return keep

    ca, cb = comparable(a), comparable(b)
    assert len(ca) == len(cb) and len(ca) > 20
    for (ka, ba), (kb, bb) in zip(ca, cb):
        assert ka == kb and ba == bb, f'seed={seed} T={T.isoformat()}: UI element {ka} differs between full and truncated databases'
    # the downloadable snapshot content hash (mock media url) is the same too
    assert any(k == 'UnknownElement' and 'download' in json.dumps(bd) for k, bd in ca) or any('Download PIT snapshot' in json.dumps(bd) for k, bd in ca)
