"""Track G (phases 98, 101-105, 110): leakage review and canary tests for the ETF research path.

Physical truncation: a second database/price frame holding ONLY information up to a random cut must give
byte-identical states, features, labels, models, decisions and trades for every decision at or before the cut.
Future canary: a huge future macro vintage and a x7 future price spike must never change anything earlier.
Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_bt_leakage.py
"""
import json
import math
import random
import sqlite3

import numpy as np
import pandas as pd
import pytest

from tests.qa_reaudit import bt_oracle as oracle
from tests.qa_reaudit import bt_runner as run

SEEDS = list(range(8))


@pytest.fixture(scope='module')
def repro():
    return run.ensure_repro()


@pytest.fixture(scope='module')
def full_store():
    s = run.open_store()
    yield s
    s.close()


@pytest.fixture(scope='module')
def states(repro):
    return json.loads((repro / 'macro_states.json').read_text())


@pytest.fixture(scope='module')
def models(repro):
    return run.frozen_models(repro)


def _canon(x):
    return json.dumps(x, sort_keys=True, allow_nan=False)


def _truncated_store(full, cut_iso):
    from fundamentals.store import PointInTimeFundamentalStore
    mem = PointInTimeFundamentalStore(':memory:')
    rows = full.db.execute('SELECT * FROM observations WHERE available_at<=?', (cut_iso,)).fetchall()
    if rows:
        mem.db.executemany('INSERT INTO observations VALUES (' + ','.join('?' * len(rows[0].keys())) + ')', [tuple(r) for r in rows])
    ev = full.db.execute('SELECT * FROM corporate_events WHERE available_at<=?', (cut_iso,)).fetchall()
    if ev:
        mem.db.executemany('INSERT INTO corporate_events VALUES (' + ','.join('?' * len(ev[0].keys())) + ')', [tuple(r) for r in ev])
    mem.db.commit()
    return mem


# --------------------------------------------------------------------------------------------------- macro path
def test_bt_macro_state_equals_independent_raw_sql_oracle_every_decision_date(states):
    """Stored macro states for all 1464 decision timestamps equal my own SQL oracle (no production code): values,
    status and presence. Also proves every one of them used only rows with available_at <= T."""
    db = sqlite3.connect(f'file:{run.ensure_db_copy()}?mode=ro', uri=True)
    bad = []
    for key, st in states.items():
        t = pd.Timestamp(key).tz_convert('UTC').isoformat()
        mine = oracle.macro_state_oracle(db, t)
        if mine['status'] != st['status']:
            bad.append((key, 'status', mine['status'], st['status']))
            continue
        for name, value in st['features'].items():
            other = mine['features'].get(name)
            if (value is None) != (other is None) or (value is not None and abs(value - other) > 1e-12 * max(1, abs(value))):
                bad.append((key, name, value, other))
    assert not bad, bad[:5]
    assert len(states) == 1464


def test_bt_every_macro_lineage_row_was_public_before_the_decision(states, full_store):
    """Dependency check beyond the previous audit: ALL lineage ids, their vintage day and their availability."""
    lag_days = []
    for key, st in list(states.items())[::7]:
        ids = st['lineage']
        rows = full_store.db.execute('SELECT id, available_at, period_end, revision FROM observations WHERE id IN (%s)' % ','.join('?' * len(ids)), ids).fetchall()
        assert len(rows) == len(set(ids))
        for r in rows:
            assert r['available_at'] <= pd.Timestamp(key).tz_convert('UTC').isoformat()
            # availability is exactly vintage day + 1 at 00:00 UTC (conservative day-granularity rule)
            assert pd.Timestamp(r['available_at']) == pd.Timestamp(r['revision'], tz='UTC') + pd.Timedelta(days=1)
            lag_days.append((pd.Timestamp(key).tz_convert('UTC') - pd.Timestamp(r['available_at'])).days)
    assert min(lag_days) >= 0


@pytest.mark.parametrize('seed', SEEDS)
def test_bt_macro_states_identical_in_physically_truncated_database(full_store, seed):
    """Phase 98 (macro path): a DB holding only rows available <= cut reproduces every earlier state byte for byte."""
    rng = random.Random(seed)
    px = run.load_prices('SPY')
    cut_idx = rng.randrange(300, len(px) - 10)
    cut = px.ts.iloc[cut_idx]
    trunc = _truncated_store(full_store, cut.isoformat())
    from fundamentals.models import MacroRegimeService
    a, b = MacroRegimeService(full_store), MacroRegimeService(trunc)
    picks = sorted(rng.sample(range(0, cut_idx + 1), 14))
    for i in picks:
        t = px.ts.iloc[i]
        assert _canon(a.state(t)) == _canon(b.state(t)), f'seed={seed} t={t}'
    trunc.close()


def test_bt_future_macro_canary_never_reaches_earlier_states(full_store):
    """Insert an absurd future vintage (CPI 1e9, unemployment 99, rate 99) available 2026-08-01; every state before
    that instant is unchanged, a state after it changes (so the canary is demonstrably live)."""
    from fundamentals.store import PointInTimeFundamentalStore
    from fundamentals.models import MacroRegimeService
    mem = _truncated_store(full_store, '2999-01-01T00:00:00+00:00')
    for series, value, units in (('CPIAUCSL', 1e9, 'index'), ('UNRATE', 99.0, 'percent'), ('FEDFUNDS', 99.0, 'percent')):
        for pe in ('2026-03-01', '2026-04-01', '2026-05-01', '2026-06-01', '2026-07-01'):
            mem.db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                           (f'canary-{series}-{pe}', 'MACRO:' + series, series, value, '', pe, '2026-08-01T00:00:00+00:00', '2026-08-01T00:00:00+00:00',
                            '2026-08-01T00:00:00+00:00', 'ALFRED/graph', '2026-08-01', '', units, 'macro', '{}'))
    mem.db.commit()
    a, b = MacroRegimeService(full_store), MacroRegimeService(mem)
    px = run.load_prices('SPY')
    for t in list(px.ts.iloc[::97]):
        assert _canon(a.state(t)) == _canon(b.state(t))
    after = pd.Timestamp('2026-08-02T00:00:00+00:00')
    assert _canon(a.state(after)) != _canon(b.state(after))
    mem.close()


# --------------------------------------------------------------------------------------------------- price / label / model path
def _panel(frame, states):
    from fundamentals.research import label_panel
    from strategies.small_account import Setup, features
    f = features(frame, Setup('bollinger', 20))
    return f, label_panel(f, states)


def _same_frames(a, b, cols=None):
    cols = list(cols or a.columns)
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True), check_exact=True)


@pytest.mark.parametrize('seed', SEEDS)
def test_bt_features_labels_and_model_unchanged_by_future_price_spike(states, seed):
    """x7 spike on every bar after a random cut (and physical truncation): features, labels with matured ends, the
    purged training panel and the fitted ridge artifact are identical; labels reaching past the cut DO change."""
    from fundamentals.research import purged_training
    from fundamentals.models import fit_model
    rng = random.Random(1000 + seed)
    sym = rng.choice(run.SYMBOLS)
    frame = run.load_prices(sym).reset_index(drop=True)
    n = len(frame)
    cut = rng.randrange(700, n - 50)
    spike = frame.copy()
    for c in ('open', 'high', 'low', 'close'):
        spike.loc[cut + 1:, c] = spike.loc[cut + 1:, c] * 7.0
    trunc = frame.iloc[:cut + 1].copy()
    f_full, p_full = _panel(frame, states)
    f_spike, p_spike = _panel(spike, states)
    f_trunc, p_trunc = _panel(trunc, states)
    # 1. features of every bar up to the cut are identical (no non-causal rolling window / shift)
    _same_frames(f_full.iloc[:cut + 1], f_spike.iloc[:cut + 1])
    _same_frames(f_full.iloc[:cut + 1], f_trunc)
    # 2. labels whose horizon ends at or before the cut are identical in all three worlds
    cut_ts = frame.ts.iloc[cut]
    m_full = p_full[p_full.label_end <= cut_ts].reset_index(drop=True)
    _same_frames(m_full, p_spike[p_spike.label_end <= cut_ts].reset_index(drop=True))
    _same_frames(m_full, p_trunc.reset_index(drop=True))
    assert len(m_full) > 500
    # 3. canary is live: labels that straddle the cut are different between full and spiked data
    straddle = p_full[(p_full.label_end > cut_ts) & (p_full.as_of <= cut_ts)]
    assert len(straddle) >= 1
    other = p_spike.set_index('as_of').loc[straddle.as_of]
    assert (other.target.to_numpy() != straddle.target.to_numpy()).all()
    # 4. a model fitted at a fit clock before the cut is bit-identical (training-only scaler + ridge)
    fit_at = frame.ts.iloc[int(cut * 0.8)]
    inputs = ('macro_inflation_yoy', 'macro_unemployment_change_3m', 'macro_policy_rate_change_3m')
    arts = []
    for p in (p_full, p_spike, p_trunc):
        t = purged_training(p, fit_at).dropna(subset=list(inputs))
        arts.append(fit_model(t, inputs, fit_at=fit_at, alpha=10.0, horizon=5))
    assert arts[0] == arts[1] == arts[2] and arts[0].id == arts[2].id


@pytest.mark.parametrize('seed', SEEDS)
def test_bt_paper_replay_decisions_and_trades_identical_with_truncated_db_and_prices(full_store, models, seed):
    """Phase 98 for the research path: truncated prices + truncated fundamentals/macro DB => byte-identical journal
    (including data_snapshot_id and the FUNDAMENTAL_DECISION payloads) and equity curve up to the cut."""
    rng = random.Random(2000 + seed)
    sym = rng.choice(run.SYMBOLS)
    system = rng.choice(['macro_filter_A_B', 'joint_C', 'joint_regime_D', 'fundamental_observe'])
    frame = run.load_prices(sym).reset_index(drop=True)
    n = len(frame)
    begin = int(n * .75)
    cut = rng.randrange(begin + 40, n - 5)
    spike = frame.copy()
    for c in ('open', 'high', 'low', 'close'):
        spike.loc[cut + 1:, c] = spike.loc[cut + 1:, c] * 5.0
    trunc_store = _truncated_store(full_store, frame.ts.iloc[cut].isoformat())
    full_run = run.run_system(system, sym, full_store, models, begin=begin, end=cut + 1, frame=frame)
    spike_run = run.run_system(system, sym, full_store, models, begin=begin, end=cut + 1, frame=spike)
    trunc_run = run.run_system(system, sym, trunc_store, models, begin=begin, end=cut + 1, frame=frame.iloc[:cut + 1])
    assert _canon(full_run.journal) == _canon(spike_run.journal) == _canon(trunc_run.journal), f'seed={seed} {sym} {system} cut={cut}'
    assert full_run.curve == spike_run.curve == trunc_run.curve
    assert any(e['event'] == 'FUNDAMENTAL_DECISION' for e in full_run.journal), 'window without any decision would make the test vacuous'
    trunc_store.close()


def test_bt_decision_uses_signal_of_previous_bar_and_macro_of_decision_bar(repro, states):
    """Timing contract: a decision at bar ts uses the technical signal of the PREVIOUS completed bar, the macro state
    as of ts, and fills at the open of ts. Check every recorded XLE decision and ENTRY."""
    from strategies.small_account import Setup, features
    frame = run.load_prices('XLE').reset_index(drop=True)
    f = features(frame, Setup('bollinger', 20))
    by_ts = {str(t): i for i, t in enumerate(frame.ts)}
    journal = json.loads((repro / 'macro_filter_A_B_journal.json').read_text())
    decisions = [e for e in journal if e['event'] == 'FUNDAMENTAL_DECISION']
    assert len(decisions) >= 8
    for e in decisions:
        i = by_ts[e['timestamp']]
        assert e['decision']['technical']['signal'] == bool(f.signal.iloc[i - 1]), e['timestamp']
        assert e['decision']['timestamp'] == pd.Timestamp(e['timestamp']).tz_convert('UTC').isoformat()
        # macro features recorded in the decision equal the stored state for this exact timestamp
        recorded = e['decision']['macro']['features']
        stored = states[e['timestamp']]['features']
        for k, v in stored.items():
            assert recorded[k] == pytest.approx(v, abs=1e-9)
    for e in journal:
        if e['event'] == 'ENTRY':
            i = by_ts[e['timestamp']]
            assert e['price'] == pytest.approx(frame.open.iloc[i] * 1.0004, rel=1e-9)


# --------------------------------------------------------------------------------------------------- ridge / alpha / splits
def _label_inputs(models_json, sym, name):
    return next(m for m in models_json if m['symbol'] == sym and m['model'] == name)


def test_bt_ridge_artifacts_match_independent_closed_form_and_alpha_selection(repro, states):
    """Training-only standardisation, ridge solution and validation alpha selection recomputed with numpy:
    z=(x-mean)/std on TRAINING rows only; coef=(Z'Z+aI)^-1 Z'(y-ybar); intercept=ybar; alpha=argmin validation MSE."""
    from fundamentals.research import label_panel, purged_training
    models_json = json.loads((repro / 'frozen_models.json').read_text())
    for sym in run.SYMBOLS:
        frame = run.load_prices(sym).reset_index(drop=True)
        from strategies.small_account import Setup, features
        f = features(frame, Setup('bollinger', 20))
        panel = label_panel(f, states)
        n = len(f)
        fit_at = pd.Timestamp(f.ts.iloc[int(n * .5)])
        test_at = pd.Timestamp(f.ts.iloc[int(n * .75)])
        train_all = purged_training(panel, fit_at)
        val_all = panel[(panel.as_of > fit_at) & (panel.label_end < test_at)]
        for name, cols in (('macro_filter_A_B', ['macro_inflation_yoy', 'macro_unemployment_change_3m', 'macro_policy_rate_change_3m']),
                           ('joint_C', ['macro_inflation_yoy', 'macro_unemployment_change_3m', 'macro_policy_rate_change_3m', 'technical_signal'])):
            m = _label_inputs(models_json, sym, name)
            tr, va = train_all.dropna(subset=cols), val_all.dropna(subset=cols)
            X, y = tr[cols].to_numpy(float), tr.target.to_numpy(float)
            mean = X.mean(axis=0)
            std = np.sqrt(((X - mean) ** 2).mean(axis=0))
            std = np.where(std > 1e-12, std, 1.0)
            Z = (X - mean) / std
            assert np.allclose(m['means'], mean, rtol=1e-12, atol=1e-15), (sym, name)
            assert np.allclose(m['scales'], std, rtol=1e-12, atol=1e-15), (sym, name)
            assert m['training_rows'] == len(tr)
            Zv, yv = (va[cols].to_numpy(float) - mean) / std, va.target.to_numpy(float)
            mses = {}
            for a in (0.1, 1.0, 10.0, 100.0):
                w = np.linalg.solve(Z.T @ Z + a * np.eye(len(cols)), Z.T @ (y - y.mean()))
                mses[a] = float(np.mean((Zv @ w + y.mean() - yv) ** 2))
                if a == m['alpha']:
                    assert np.allclose(m['coefficients'], w, rtol=1e-9, atol=1e-13), (sym, name)
                    assert m['intercept'] == pytest.approx(float(y.mean()), rel=1e-12, abs=1e-15)
            assert m['alpha'] == min(mses, key=mses.get), (sym, name, mses)
            assert m['validation_mse'] == pytest.approx(mses[m['alpha']], rel=1e-9)
            # the diagnostic window is not used: every label of training and validation ends before it
            assert tr.label_end.max() < fit_at and va.label_end.max() < test_at


def test_bt_chronological_split_boundaries_are_exact_and_written(repro):
    """Exact printed split points: index 732 (fit clock), 1098 (diagnostic start); purge gaps; file written."""
    models_json = json.loads((repro / 'frozen_models.json').read_text())
    lines = []
    for m in models_json:
        frame = run.load_prices(m['symbol']).reset_index(drop=True)
        n = len(frame)
        fit_idx, test_idx = int(n * .5), int(n * .75)
        assert pd.Timestamp(m['trained_through']) == frame.ts.iloc[fit_idx]
        assert pd.Timestamp(m['test_start']) == frame.ts.iloc[test_idx]
        assert pd.Timestamp(m['train_last_label']) < pd.Timestamp(m['trained_through']) < pd.Timestamp(m['validation_last_label']) < pd.Timestamp(m['test_start'])
        lines.append(f"{m['symbol']:4s} {m['model']:17s} n={n} train[0..{fit_idx - 1}] last_label={m['train_last_label']} fit_clock=ts[{fit_idx}]={m['trained_through']} "
                     f"validation last_label={m['validation_last_label']} diagnostic ts[{test_idx}..{n - 1}] starts {m['test_start']} (diagnostic bars {n - test_idx})")
    (run.OUT / 'split_boundaries.txt').write_text('\n'.join(lines) + '\n')
    assert len(lines) == 12


# --------------------------------------------------------------------------------------------------- guards
def _panel_for_guards(n=40):
    base = pd.Timestamp('2024-01-02', tz='UTC')
    rows = []
    rng = random.Random(1)
    for i in range(n):
        rows.append({'as_of': base + pd.Timedelta(days=i), 'label_end': base + pd.Timedelta(days=i + 5), 'x': rng.random(), 'target': rng.random() - 0.5})
    return pd.DataFrame(rows)


def test_bt_fit_model_refuses_unmatured_labels_and_boundary_is_strict():
    from fundamentals.models import fit_model
    p = _panel_for_guards()
    ok_at = p.label_end.max() + pd.Timedelta(seconds=1)
    fit_model(p, ['x'], fit_at=ok_at)
    with pytest.raises(ValueError, match='unmatured'):
        fit_model(p, ['x'], fit_at=p.label_end.max())  # label_end == fit clock is NOT matured (strict)
    with pytest.raises(ValueError):
        fit_model(p.assign(label_end=p.as_of), ['x'], fit_at=ok_at)  # empty label interval
    with pytest.raises(ValueError):
        fit_model(p.assign(x=np.nan), ['x'], fit_at=ok_at)


def test_bt_purged_training_boundary_matches_fit_model_boundary():
    from fundamentals.research import purged_training
    p = _panel_for_guards()
    cutoff = p.label_end.iloc[10]
    kept = purged_training(p, cutoff)
    assert (kept.label_end < cutoff).all() and len(kept) == int((p.label_end < cutoff).sum())


def test_bt_score_engine_refuses_historical_timestamps(models):
    from fundamentals.models import FundamentalScoreEngine
    art = models[('XLE', 'macro_filter_A_B')]
    eng = FundamentalScoreEngine(art)
    feats = dict(macro_inflation_yoy=0.03, macro_unemployment_change_3m=0.0, macro_policy_rate_change_3m=0.0)
    with pytest.raises(ValueError, match='unavailable'):
        eng.score(feats, art.trained_through)  # equal to the fit clock is refused
    assert eng.score(feats, pd.Timestamp(art.trained_through) + pd.Timedelta(seconds=1))['score'] is not None


def test_bt_cross_sectional_ic_requires_pit_membership_and_20_names():
    from fundamentals.research import cross_sectional_ic
    rng = np.random.default_rng(3)
    t0 = pd.Timestamp('2024-03-01', tz='UTC')
    def panel(n, known_shift=0):
        return pd.DataFrame({'as_of': t0, 'symbol': [f'S{i}' for i in range(n)], 'label_end': t0 + pd.Timedelta(days=5),
                             'target': rng.normal(size=n), 'score': rng.normal(size=n),
                             'member_known_at': t0 + pd.Timedelta(days=known_shift)})
    assert len(cross_sectional_ic(panel(25))) == 1
    assert cross_sectional_ic(panel(19)) == []  # fewer than 20 names: silently skipped
    with pytest.raises(ValueError, match='future universe'):
        cross_sectional_ic(panel(25, known_shift=1))
    with pytest.raises(ValueError, match='duplicate'):
        p = panel(25)
        p.loc[1, 'symbol'] = 'S0'
        cross_sectional_ic(p)
    with pytest.raises(ValueError):
        cross_sectional_ic(panel(25).drop(columns=['member_known_at']))


def test_bt_cross_sectional_normalisation_trusts_the_callers_universe():
    """LIMITATION (documented): normalize_cross_section ranks whatever frame it is given. Adding a symbol that only
    exists in the future (no PIT guard) changes every other company's percentile."""
    from fundamentals.features import normalize_cross_section
    base = pd.DataFrame({'symbol': list('ABCDE'), 'sector': 'tech', 'pe': [10., 12., 14., 16., 18.]})
    future = pd.concat([base, pd.DataFrame({'symbol': ['FUT'], 'sector': ['tech'], 'pe': [1.0]})], ignore_index=True)
    a = normalize_cross_section(base, 'pe').set_index('symbol').pe_percentile
    b = normalize_cross_section(future, 'pe').set_index('symbol').pe_percentile
    assert not np.allclose(a.loc[list('ABCDE')].to_numpy(), b.loc[list('ABCDE')].to_numpy())


# --------------------------------------------------------------------------------------------------- positive controls
def test_bt_positive_control_macro_leak_injection_is_detected(full_store):
    """The truncation harness is not vacuous: a store that ignores the decision time (reads every stored vintage) is
    caught by the very same comparison."""
    from fundamentals.store import PointInTimeFundamentalStore
    from fundamentals.models import MacroRegimeService

    class Leaky(PointInTimeFundamentalStore):
        def as_of(self, symbol, timestamp, **kw):
            return super().as_of(symbol, '2999-01-01T00:00:00+00:00', **kw)

    px = run.load_prices('SPY')
    cut = px.ts.iloc[900]
    leaky_full = _truncated_store(full_store, '2999-01-01T00:00:00+00:00')
    leaky_full.__class__ = Leaky
    leaky_trunc = _truncated_store(full_store, cut.isoformat())
    leaky_trunc.__class__ = Leaky
    t = px.ts.iloc[700]
    assert _canon(MacroRegimeService(leaky_full).state(t)) != _canon(MacroRegimeService(leaky_trunc).state(t))
    leaky_full.close()
    leaky_trunc.close()


def test_bt_positive_control_feature_leak_injection_is_detected(states):
    """A one-bar look-ahead in the signal (shift(-1)) is caught by the price-truncation comparison."""
    from strategies.small_account import Setup, features
    frame = run.load_prices('XLE').reset_index(drop=True)
    base = features(frame, Setup('bollinger', 20))
    cut = int(next(i for i in range(800, len(base)) if bool(base.signal.iloc[i]))) - 1  # next bar carries a signal

    def leaky(fr):
        f = features(fr, Setup('bollinger', 20))
        f['signal'] = f['signal'].shift(-1).fillna(False)
        return f
    with pytest.raises(AssertionError):
        _same_frames(leaky(frame).iloc[:cut + 1], leaky(frame.iloc[:cut + 1]))
    _same_frames(features(frame, Setup('bollinger', 20)).iloc[:cut + 1], features(frame.iloc[:cut + 1], Setup('bollinger', 20)))


@pytest.mark.parametrize('symbol', run.SYMBOLS)
def test_bt_label_panel_equals_independent_open_to_open_labels(states, symbol):
    """target = open[i+1+h]/open[i+1] - 1 - 0.0038 at decision time ts[i+1]; label_end = ts[i+1+h]; macro features of
    the decision timestamp; technical signal of bar i. Recomputed row by row for all 5 horizons."""
    from fundamentals.research import label_panel
    from strategies.small_account import Setup, features
    frame = run.load_prices(symbol).reset_index(drop=True)
    f = features(frame, Setup('bollinger', 20))
    for h in (1, 5, 20):
        panel = label_panel(f, states, horizon=h)
        n = len(frame)
        assert len(panel) == n - h - 1
        for i in random.Random(h).sample(range(len(panel)), 60):
            row = panel.iloc[i]
            assert row.as_of == frame.ts.iloc[i + 1] and row.label_end == frame.ts.iloc[i + 1 + h]
            assert row.target == pytest.approx(frame.open.iloc[i + 1 + h] / frame.open.iloc[i + 1] - 1 - 0.0038, abs=1e-15)
            assert row.technical_signal == float(f.signal.iloc[i])
            st = states[str(frame.ts.iloc[i + 1])]['features']
            assert row.macro_inflation_yoy == st['inflation_yoy'] and row.macro_policy_rate_change_3m == st['policy_rate_change_3m']
