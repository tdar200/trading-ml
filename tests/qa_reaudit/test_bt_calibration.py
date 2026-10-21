"""Track G (phase 110): leakage guards and arithmetic of fundamentals/confidence.py (held-out sign calibration).

Independent Wilson interval / Brier / bin counts from the definitions; every temporal guard exercised with a minimal
violating panel. Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_bt_calibration.py
"""
import math
import random

import pandas as pd
import pytest

T0 = pd.Timestamp('2024-01-01', tz='UTC')


def _model():
    from fundamentals.models import ModelArtifact
    return ModelArtifact(('x',), (0.0,), (1.0,), (0.01,), -0.001, T0.isoformat(), 1.0, 100, 'a' * 64, 5)


def _panel(n=40, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a = pd.Timestamp('2024-02-01', tz='UTC') + pd.Timedelta(days=10 * i)
        rows.append({'as_of': a, 'label_end': a + pd.Timedelta(days=5), 'x': rng.gauss(0, 1), 'target': rng.gauss(0, 0.02), 'horizon': 5, 'net_of_costs': True})
    return pd.DataFrame(rows)


def _calibrated_at(p):
    return (p.label_end.max() + pd.Timedelta(days=1)).isoformat()


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


@pytest.mark.parametrize('seed', range(12))
def test_bt_calibration_counts_probability_wilson_and_brier_match_definitions(seed):
    from fundamentals.confidence import fit_confidence
    from fundamentals.models import FundamentalScoreEngine
    m, p = _model(), _panel(40, seed)
    cal = fit_confidence(m, p, calibrated_at=_calibrated_at(p))
    # independent: sign of score = sign(intercept + x*0.01/1); bins [score<=0, score>0]
    bins = [[0, 0], [0, 0]]
    pairs = []
    for r in p.itertuples():
        score = -0.001 + 0.01 * r.x
        g, y = int(score > 0), int(r.target > 0)
        bins[g][0] += 1
        bins[g][1] += y
        pairs.append((g, y))
    assert [list(b) for b in cal.bins] == bins and cal.observations == 40
    brier = sum((bins[g][1] / bins[g][0] - y) ** 2 for g, y in pairs) / len(pairs)
    assert cal.brier == pytest.approx(brier, abs=1e-12)
    at = (pd.Timestamp(cal.calibrated_at) + pd.Timedelta(days=1)).isoformat()
    for score, g in ((0.5, 1), (-0.5, 0)):
        out = cal.probability(score, at)
        n, k = bins[g]
        if n == 0:
            assert out['probability'] is None
            continue
        lo, hi = wilson(k, n)
        assert out['probability'] == pytest.approx(k / n, abs=1e-15)
        assert out['interval_95'] == pytest.approx([lo, hi], abs=1e-12)
        assert out['observations'] == n
    # engine wiring: confidence appears only through a matching calibration
    eng = FundamentalScoreEngine(m, cal)
    res = eng.score({'x': 1.0}, at)
    assert res['confidence'] == pytest.approx(bins[1][1] / bins[1][0], abs=1e-15)


@pytest.mark.parametrize('mutation,match', [
    ('before_training', 'follow training'),
    ('unmatured', 'follow training'),
    ('overlap', 'overlapping'),
    ('horizon', 'horizon'),
    ('gross', 'net-of-cost'),
    ('nan_target', 'invalid target'),
    ('missing_feature', 'incomplete'),
])
def test_bt_calibration_refuses_every_leaky_or_inconsistent_panel(mutation, match):
    from fundamentals.confidence import fit_confidence
    m, p = _model(), _panel(20, 1)
    at = _calibrated_at(p)
    if mutation == 'before_training':
        p.loc[0, 'as_of'] = T0 - pd.Timedelta(days=2)
        p.loc[0, 'label_end'] = T0 + pd.Timedelta(days=3)
    elif mutation == 'unmatured':
        at = p.label_end.max().isoformat()  # label ends exactly at the calibration clock: not matured
    elif mutation == 'overlap':
        p.loc[3, 'label_end'] = p.loc[4, 'as_of'] + pd.Timedelta(days=1)
    elif mutation == 'horizon':
        p['horizon'] = 20
    elif mutation == 'gross':
        p.loc[2, 'net_of_costs'] = False
    elif mutation == 'nan_target':
        p.loc[5, 'target'] = float('nan')
    elif mutation == 'missing_feature':
        p.loc[6, 'x'] = float('nan')
    with pytest.raises(ValueError, match=match):
        fit_confidence(m, p, calibrated_at=at)


def test_bt_calibration_probability_unavailable_before_calibration_time_and_validation_of_counts():
    from fundamentals.confidence import fit_confidence, ConfidenceCalibration
    m, p = _model(), _panel(20, 2)
    cal = fit_confidence(m, p, calibrated_at=_calibrated_at(p))
    with pytest.raises(ValueError, match='unavailable'):
        cal.probability(0.1, cal.calibrated_at)  # equal to the calibration instant is refused
    with pytest.raises(ValueError):
        cal.probability(float('nan'), (pd.Timestamp(cal.calibrated_at) + pd.Timedelta(days=1)).isoformat())
    with pytest.raises(ValueError):
        ConfidenceCalibration('m', cal.calibrated_at, ((5, 6), (5, 1)), 'h', 10, 0.2)  # successes > n
    with pytest.raises(ValueError):
        ConfidenceCalibration('m', cal.calibrated_at, ((5, 1), (5, 1)), 'h', 11, 0.2)  # count mismatch


def test_bt_unvalidated_engine_never_invents_a_score_or_confidence():
    from fundamentals.models import FundamentalScoreEngine
    r = FundamentalScoreEngine().score({'x': 1.0}, '2025-01-01T00:00:00+00:00')
    assert r['score'] is None and r['confidence'] is None and r['status'] == 'UNVALIDATED'
    art = _model()
    miss = FundamentalScoreEngine(art).score({'x': None}, '2025-01-01T00:00:00+00:00')
    assert miss['score'] is None and miss['status'] == 'MISSING_FEATURES'
    nonfinite = FundamentalScoreEngine(art).score({'x': float('inf')}, '2025-01-01T00:00:00+00:00')
    assert nonfinite['score'] is None
