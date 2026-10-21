"""Restart/recovery, snapshot consistency under a concurrent writer, decision reproducibility (QA track A)."""
from __future__ import annotations
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from tests.qa_reaudit import pit_helpers as H
from tests.qa_reaudit import pit_oracle as O
from tests.qa_reaudit.pit_helpers import M, fact
from tests.qa_reaudit.pit_phase98 import make_model

from fundamentals.api import FundamentalAPI
from fundamentals.features import FundamentalFeatureEngine
from fundamentals.models import (CombinedSignalEngine, EventCatalystEngine, FundamentalScoreEngine, MacroRegimeService,
                                 ModelArtifact)
from fundamentals.store import Fact, PointInTimeFundamentalStore

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
T0 = '2025-06-01T00:00:00Z'


def full_engine(store):
    return CombinedSignalEngine(FundamentalFeatureEngine(store), FundamentalScoreEngine(make_model()), MacroRegimeService(store),
                                EventCatalystEngine(store, 30.), policy='filter', research_shadow=True)


def macro_series(n_months=30, last=None):
    out = []
    import datetime as dt
    for i in range(n_months):
        y, m = 2022 + (i // 12), i % 12 + 1
        avail = dt.datetime(y, m, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=45)
        for name, units, base in (('CPIAUCSL', 'index', 300.0 + i), ('UNRATE', 'percent', 4.0 + 0.01 * i), ('FEDFUNDS', 'percent', 5.0)):
            out.append(Fact('MACRO:' + name, name, base, f'{y}-{m:02d}-01', avail.isoformat(), avail.isoformat(), 'ALFRED/graph', 'v1', '', units, '', H.OBS, 'macro', {}))
    return out


def populated(path):
    s = PointInTimeFundamentalStore(path)
    s.ingest(H.acme_nov_fy('2025-05-10T07:00:00Z'))
    s.ingest(macro_series())
    s.add_event(symbol='ACME', event_type='earnings', event_at='2025-05-10T06:00:00Z', reported_at='2025-05-10T06:00:00Z', available_at='2025-05-10T06:00:00Z',
                source='x', revision='1', payload={'event_key': 'q4', 'reported_eps': 1.1, 'consensus_eps': 1.0, 'consensus_available_at': '2025-04-01T00:00:00Z'})
    return s


# ---------------------------------------------------------------------------
# restart in a NEW PROCESS, different hash seeds
# ---------------------------------------------------------------------------
CHILD = textwrap.dedent('''
    import sys, json, hashlib
    sys.path.insert(0, %r)
    from tests.qa_reaudit.test_pit_recovery_concurrency import full_engine
    from fundamentals.store import PointInTimeFundamentalStore
    from fundamentals.api import FundamentalAPI
    s = PointInTimeFundamentalStore(sys.argv[1], readonly=True)
    d = full_engine(s).decide('ACME', %r, {'signal': True})
    api = FundamentalAPI(s).get('/fundamentals/ACME', {'as_of': %r})
    raw = json.dumps(api, allow_nan=False)
    print(json.dumps({'id': d['data_snapshot_id'], 'canon': hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest(),
                      'api_canon': hashlib.sha256(json.dumps(api, sort_keys=True).encode()).hexdigest(),
                      'api_raw': hashlib.sha256(raw.encode()).hexdigest()}))
''' % (str(ROOT), T0, T0))


def run_child(db, hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    out = subprocess.run([PY, '-c', CHILD, str(db)], capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_new_process_reopening_the_database_reproduces_decision_and_snapshot_id_across_hash_seeds(tmp_path):
    db = tmp_path / 'f.sqlite'
    s = populated(db)
    here = full_engine(s).decide('ACME', T0, {'signal': True})
    api_here = FundamentalAPI(s).get('/fundamentals/ACME', {'as_of': T0})
    s.close()
    results = [run_child(db, seed) for seed in (0, 1, 12345)]
    for r in results:
        assert r['id'] == here['data_snapshot_id']
        assert r['canon'] == hashlib.sha256(json.dumps(here, sort_keys=True).encode()).hexdigest()
        assert r['api_canon'] == hashlib.sha256(json.dumps(api_here, sort_keys=True).encode()).hexdigest()
    assert here['fundamental']['features']['revenue_yoy'] is not None and here['macro']['status'] in ('AVAILABLE', 'INCOMPLETE', 'STALE')


def test_http_and_download_json_raw_bytes_are_reproducible_across_processes(tmp_path):
    db = tmp_path / 'f.sqlite'
    s = populated(db)
    s.close()
    raws = {run_child(db, seed)['api_raw'] for seed in (0, 1, 2, 3)}
    assert len(raws) == 1


# ---------------------------------------------------------------------------
# crash / kill recovery
# ---------------------------------------------------------------------------
def test_committed_data_survives_a_hard_kill_and_uncommitted_data_is_absent(tmp_path):
    db = tmp_path / 'k.sqlite'
    s = PointInTimeFundamentalStore(db)
    s.ingest(H.acme_nov_fy('2025-05-10T07:00:00Z'))
    n_committed = s.db.execute('select count(*) from observations').fetchone()[0]
    s.close()
    child = textwrap.dedent(f'''
        import os, signal, sys
        sys.path.insert(0, {str(ROOT)!r})
        from fundamentals.store import PointInTimeFundamentalStore
        from tests.qa_reaudit.pit_helpers import fact
        s = PointInTimeFundamentalStore({str(db)!r})
        # committed batch (will survive SIGKILL even though the WAL is not checkpointed)
        s.ingest([fact('revenue', 111.0, '2025-03-31', '2024-04-01', '2025-06-01T00:00:00Z', rev='committed')])
        # an open, uncommitted write transaction at the moment of the kill
        s.db.execute('BEGIN IMMEDIATE')
        for i in range(500):
            s.db.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (f'uncommitted{{i}}', 'ACME', 'revenue', 9.9e9, '2024-04-01', '2025-03-31', '2025-06-02T00:00:00+00:00', '2025-06-02T00:00:00+00:00',
                          '2030-01-01T00:00:00+00:00', 'SRC', 'u', 'USD', 'USD', 'financial', '{{}}'))
        os.kill(os.getpid(), signal.SIGKILL)
    ''')
    proc = subprocess.run([PY, '-c', child], capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert proc.returncode == -signal.SIGKILL, proc.stderr[-1500:]
    s = PointInTimeFundamentalStore(db)
    assert s.db.execute('pragma integrity_check').fetchone()[0] == 'ok'
    assert s.db.execute('select count(*) from observations').fetchone()[0] == n_committed + 1
    assert s.db.execute("select count(*) from observations where id like 'uncommitted%'").fetchone()[0] == 0
    assert [r['value'] for r in s.as_of('ACME', '2025-06-03T00:00:00Z', kind='financial') if r['metric'] == 'revenue' and r['period_end'] == '2025-03-31'] == [111.0]
    # the append-only triggers are still in force after recovery
    with pytest.raises(sqlite3.DatabaseError, match='append-only'):
        s.db.execute('DELETE FROM observations')
    assert FundamentalFeatureEngine(s).snapshot('ACME', '2025-06-03T00:00:00Z')['status'] == 'AVAILABLE'


def test_reopening_is_idempotent_schema_and_data_unchanged(tmp_path):
    db = tmp_path / 'r.sqlite'
    s = populated(db)
    before = FundamentalFeatureEngine(s).snapshot('ACME', T0)
    ver = s.db.execute('pragma user_version').fetchone()[0]
    s.close()
    for _ in range(3):
        s = PointInTimeFundamentalStore(db)
        assert s.db.execute('pragma user_version').fetchone()[0] == ver
        assert O.canon(FundamentalFeatureEngine(s).snapshot('ACME', T0)) == O.canon(before)
        s.close()
    ro = PointInTimeFundamentalStore(db, readonly=True)
    assert O.canon(FundamentalFeatureEngine(ro).snapshot('ACME', T0)) == O.canon(before)
    with pytest.raises(sqlite3.OperationalError):
        ro.db.execute('DELETE FROM observations')
    with pytest.raises(sqlite3.OperationalError):
        ro.ingest([fact('revenue', 1., '2020-12-31', '2020-01-01', '2021-01-01T00:00:00Z')])


# ---------------------------------------------------------------------------
# concurrent writer: one decision == one consistent snapshot
# ---------------------------------------------------------------------------
def test_writer_committing_between_the_reads_of_one_decision_is_invisible_to_that_decision(tmp_path):
    db = tmp_path / 'c.sqlite'
    s = populated(db)
    baseline = full_engine(s).decide('ACME', T0, {'signal': True})
    writer = PointInTimeFundamentalStore(db)  # a second, real connection
    state = {'calls': 0}
    real = s.as_of

    def hooked(*a, **k):
        r = real(*a, **k)
        state['calls'] += 1
        if state['calls'] == 1:  # right after the financial read, before macro and event reads
            writer.ingest([fact('revenue', 7777 * M, '2025-03-31', '2024-04-01', '2025-05-20T00:00:00Z', rev='backfill-restatement')])
            writer.ingest([Fact('MACRO:CPIAUCSL', 'CPIAUCSL', 999.0, '2024-12-01', '2025-05-25T00:00:00Z', '2025-05-25T00:00:00Z', 'ALFRED/graph', 'v2', '', 'index', '', H.OBS, 'macro', {})])
            writer.add_event(symbol='ACME', event_type='earnings', event_at='2025-05-26T00:00:00Z', reported_at='2025-05-26T00:00:00Z', available_at='2025-05-26T00:00:00Z',
                             source='x', revision='9', payload={'event_key': 'q9', 'reported_eps': 5.0, 'consensus_eps': 1.0, 'consensus_available_at': '2025-05-01T00:00:00Z'})
        return r

    s.as_of = hooked
    during = full_engine(s).decide('ACME', T0, {'signal': True})
    assert state['calls'] >= 3  # financial, macro series, classification/... all read inside the one decision
    assert O.canon(during) == O.canon(baseline), 'a concurrent commit leaked into a decision that had already started reading'
    assert during['data_snapshot_id'] == baseline['data_snapshot_id']
    del s.as_of
    after = full_engine(s).decide('ACME', T0, {'signal': True})
    assert after['data_snapshot_id'] != baseline['data_snapshot_id']  # the write was real and visible to the next decision
    assert after['fundamental']['raw']['revenue'] == 7777 * M


def test_without_the_consistent_read_wrapper_the_same_interleaving_would_mix_versions_control_experiment(tmp_path):
    """Control: proves the test above is sensitive. Unwrapped (no read transaction) calls see different worlds."""
    db = tmp_path / 'c2.sqlite'
    s = populated(db)
    writer = PointInTimeFundamentalStore(db)
    feats = FundamentalFeatureEngine(s)
    macro = MacroRegimeService(s)
    raw_snapshot = type(feats).snapshot.__wrapped__  # unwrapped: no BEGIN
    raw_state = type(macro).state.__wrapped__
    fin_before = raw_snapshot(feats, 'ACME', T0)
    mac_before = raw_state(macro, T0)
    writer.ingest([fact('revenue', 7777 * M, '2025-03-31', '2024-04-01', '2025-05-20T00:00:00Z', rev='backfill'),
                   Fact('MACRO:CPIAUCSL', 'CPIAUCSL', 999.0, '2024-12-01', '2025-05-25T00:00:00Z', '2025-05-25T00:00:00Z', 'ALFRED/graph', 'v2', '', 'index', '', H.OBS, 'macro', {})])
    fin_after = raw_snapshot(feats, 'ACME', T0)
    mac_after = raw_state(macro, T0)
    assert fin_before['raw']['revenue'] == 1000 * M and fin_after['raw']['revenue'] == 7777 * M
    assert O.canon(mac_before) != O.canon(mac_after)


@pytest.mark.parametrize('rounds', [1])
def test_threaded_writer_never_produces_a_decision_mixing_two_versions(tmp_path, rounds):
    """Writer thread commits N atomic generations (revenue restatement + CPI vintage + UNRATE vintage carrying the same
    generation number). Every decision taken while it runs must show one single generation in all components."""
    db = tmp_path / 't.sqlite'
    s0 = populated(db)
    s0.close()
    N = 60
    stop = threading.Event()
    errors = []

    def writer():
        try:
            w = PointInTimeFundamentalStore(db)
            for n in range(1, N + 1):
                av = f'2025-05-{11 + n // 24:02d}T{n % 24:02d}:00:00Z'
                batch = [fact('revenue', 1000 * M + n * M, '2025-03-31', '2024-04-01', av, rev=f'g{n}'),
                         Fact('MACRO:CPIAUCSL', 'CPIAUCSL', 400.0 + n, '2024-12-01', av, av, 'ALFRED/graph', f'g{n}', '', 'index', '', H.OBS, 'macro', {}),
                         Fact('MACRO:UNRATE', 'UNRATE', 5.0 + n / 100, '2024-12-01', av, av, 'ALFRED/graph', f'g{n}', '', 'percent', '', H.OBS, 'macro', {})]
                w.ingest(batch)  # one transaction
                time.sleep(0.02)
            w.close()
        except Exception as e:  # pragma: no cover
            errors.append(e)
        finally:
            stop.set()

    seen = set()
    inconsistent = []

    def reader():
        try:
            r = PointInTimeFundamentalStore(db)
            eng = full_engine(r)
            while True:
                d = eng.decide('ACME', T0, {'signal': True})
                rev = d['fundamental']['raw'].get('revenue')
                gen_rev = None if rev is None else round((rev - 1000 * M) / M)
                feats = d['macro']['features']
                gens = set()
                if feats.get('policy_rate') is not None:
                    pass
                lineage = d['macro']['lineage']
                ids = r.db.execute('select revision from observations where id in (%s)' % ','.join('?' * len(lineage)), lineage).fetchall() if lineage else []
                macro_revs = {x[0] for x in ids if x[0].startswith('g')}
                fin_revs = {x[0] for x in r.db.execute('select revision from observations where id in (%s)' % ','.join('?' * len(d['fundamental']['lineage'])), d['fundamental']['lineage']) if x[0].startswith('g')}
                seen.add(gen_rev)
                if macro_revs and fin_revs and macro_revs != fin_revs:
                    inconsistent.append((fin_revs, macro_revs))
                if stop.is_set():
                    break
            r.close()
        except Exception as e:
            errors.append(e)

    tw, tr = threading.Thread(target=writer), threading.Thread(target=reader)
    tr.start()
    tw.start()
    tw.join(timeout=240)
    tr.join(timeout=240)
    assert not errors, errors
    assert not inconsistent, inconsistent[:3]
    assert len(seen) >= 3  # the reader really observed several generations while the writer ran


# ---------------------------------------------------------------------------
# decision reproducibility from recorded configuration + data_snapshot_id
# ---------------------------------------------------------------------------
def rebuild_engine_from_decision(store, d):
    cfg = d['configuration']
    m = cfg['model']
    art = ModelArtifact(tuple(m['features']), tuple(m['means']), tuple(m['scales']), tuple(m['coefficients']), m['intercept'], m['trained_through'],
                        m['alpha'], m['training_rows'], m['training_sha256'], m['horizon'])
    return CombinedSignalEngine(FundamentalFeatureEngine(store, cfg['max_period_age_days']), FundamentalScoreEngine(art),
                                MacroRegimeService(store, cfg['macro_max_age']), EventCatalystEngine(store, cfg['catalyst_half_life']),
                                policy=cfg['policy'], threshold=cfg['threshold'], research_shadow=cfg['research_shadow'])


def test_decision_is_reproducible_from_its_recorded_configuration_after_later_data_arrives(tmp_path):
    db = tmp_path / 'p.sqlite'
    s = populated(db)
    d = full_engine(s).decide('ACME', T0, {'signal': True})
    journal_text = json.dumps(d, allow_nan=False)  # what PaperEngine writes to the journal
    # a later filing, later restatement, later macro vintage and later event are ingested (all published after T0)
    s.ingest([fact('revenue', 4000 * M, '2025-03-31', '2024-04-01', '2025-09-01T00:00:00Z', rev='later')])
    s.ingest([Fact('MACRO:CPIAUCSL', 'CPIAUCSL', 999.0, '2025-02-01', '2025-09-02T00:00:00Z', '2025-09-02T00:00:00Z', 'ALFRED/graph', 'v9', '', 'index', '', H.OBS, 'macro', {})])
    s.add_event(symbol='ACME', event_type='earnings', event_at='2025-09-03T00:00:00Z', reported_at='2025-09-03T00:00:00Z', available_at='2025-09-03T00:00:00Z', source='x', revision='z', payload={'event_key': 'q1'})
    s.close()
    s2 = PointInTimeFundamentalStore(db)  # restart
    recorded = json.loads(journal_text)
    again = rebuild_engine_from_decision(s2, recorded).decide(recorded['symbol'], recorded['timestamp'], {'signal': recorded['technical']['signal']})
    assert again['data_snapshot_id'] == recorded['data_snapshot_id']
    assert O.canon(again) == O.canon(recorded)


def test_backfilled_fact_changes_the_data_snapshot_id_so_irreproducibility_is_detected_not_silent(tmp_path):
    """LIMITATION: decide()/snapshot() never use observed_only, so a decision cannot be replayed *as the live system saw it*
    after a backfill with available_at <= T; the changed data_snapshot_id is the only (and sufficient) alarm."""
    db = tmp_path / 'b.sqlite'
    s = populated(db)
    d = full_engine(s).decide('ACME', T0, {'signal': True})
    s.ingest([fact('revenue', 1234 * M, '2025-03-31', '2024-04-01', '2025-05-12T00:00:00Z', rev='found-later', observed='2026-01-01T00:00:00Z')])
    again = full_engine(s).decide('ACME', T0, {'signal': True})
    assert again['data_snapshot_id'] != d['data_snapshot_id']
    assert s.as_of('ACME', T0, observed_only=True, kind='financial') != s.as_of('ACME', T0, kind='financial')
    # observed_only is only reachable through the store API, not through the engines
    import inspect
    assert 'observed_only' not in inspect.signature(FundamentalFeatureEngine.snapshot).parameters
    assert 'observed_only' not in inspect.signature(CombinedSignalEngine.decide).parameters


def test_no_hidden_cache_new_data_is_visible_to_long_lived_objects_and_old_instants_stay_stable(tmp_path):
    db = tmp_path / 'n.sqlite'
    s = populated(db)
    api = FundamentalAPI(s)
    first = api.get('/fundamentals/ACME', {'as_of': T0})
    s.ingest([fact('revenue', 1500 * M, '2025-03-31', '2024-04-01', '2025-05-20T00:00:00Z', rev='amend')])
    second = api.get('/fundamentals/ACME', {'as_of': T0})
    earlier = api.get('/fundamentals/ACME', {'as_of': '2025-05-19T23:59:59Z'})
    assert first['raw']['revenue'] == 1000 * M and second['raw']['revenue'] == 1500 * M and earlier['raw']['revenue'] == 1000 * M
    assert O.canon(earlier) != O.canon(second)
    with H.http_server(db) as get:
        a = get('/fundamentals/ACME?as_of=2025-05-19T23:59:59Z')[1]
        b = get('/fundamentals/ACME?as_of=2025-05-20T00:00:00Z')[1]
        assert a['raw']['revenue'] == 1000 * M and b['raw']['revenue'] == 1500 * M


def test_no_cache_decorators_anywhere_in_fundamentals_or_the_view():
    src = (ROOT / 'dashboard' / 'views' / 'fundamentals.py').read_text()
    assert 'cache_data' not in src and 'cache_resource' not in src and 'lru_cache' not in src
    for p in (ROOT / 'fundamentals').glob('*.py'):
        t = p.read_text()
        assert 'lru_cache' not in t and 'functools.cache' not in t and '@cache' not in t, p
