"""QA track E: CombinedSignalEngine.decide truth table, score arithmetic, thresholds, catalyst/macro authority.

Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_dec_decision.py
"""
from __future__ import annotations
import itertools
import json
import math
from fractions import Fraction

import pytest

from fundamentals.catalysts import ingest_disclosures
from fundamentals.models import FundamentalScoreEngine, CombinedSignalEngine
from fundamentals.features import FundamentalFeatureEngine
from tests.qa_reaudit import dec_fixtures as F

POLICIES = ('observe', 'filter', 'selection', 'weighted', 'regime')
MODELS = {'none': None, 'fund': F.M_FUND, 'joint': F.M_JOINT, 'macro': F.M_MACRO, 'missing': F.M_MISSING}
FUNDS = ('GOOD', 'BAD', 'OLD', 'NONE')
MACROS = ('full', 'nofed', 'stale', 'nosvc')

_STORES = {}


def store(kind):
    if kind not in _STORES:
        _STORES[kind] = F.build_store(macro={'full': 'full', 'nofed': 'nofed', 'stale': 'full', 'nosvc': 'full'}[kind])
    return _STORES[kind]


# ---------------------------------------------------------------- independent specification
def spec_inputs(fund, macro, tech, model):
    """The feature values the model can see, from the fixture constants (hand-derived), or None when not computable."""
    vals = {'technical_signal': Fraction(int(tech))}
    if fund in ('GOOD', 'OLD'):
        vals.update(revenue_yoy=F.GOOD_INPUTS['revenue_yoy'], fcf_margin=F.GOOD_INPUTS['fcf_margin'])
    elif fund == 'BAD':
        vals.update(revenue_yoy=F.BAD_INPUTS['revenue_yoy'], fcf_margin=F.BAD_INPUTS['fcf_margin'])
    if macro in ('full', 'nofed'):
        vals['macro_unemployment'] = Fraction(5)
    return vals


def spec_expected(policy, shadow, model, fund, macro, tech, threshold=0.):
    inputs = spec_inputs(fund, macro, tech, model)
    score = None
    if model is not None and all(k in inputs for k in model.features):
        score = F.oracle_score(model, inputs)
    fund_status = {'GOOD': 'AVAILABLE', 'BAD': 'AVAILABLE', 'OLD': 'STALE', 'NONE': 'MISSING'}[fund]
    macro_status = {'full': 'AVAILABLE', 'nofed': 'INCOMPLETE', 'stale': 'STALE', 'nosvc': 'MISSING'}[macro]
    permit = False
    if policy == 'observe':
        permit = tech
        why = ['Fundamental observation only; technical decision unchanged']
    elif not shadow:
        why = ['No fundamental model approved for deployment']
    elif score is None:
        why = ['Missing or unvalidated model/feature coverage']
    elif fund_status != 'AVAILABLE' and any(not k.startswith(('macro_', 'technical_')) for k in model.features):
        why = ['Fundamentals missing/stale or wrong instrument type']
    elif policy == 'regime' and macro_status != 'AVAILABLE':
        why = ['Macro vintage coverage missing/stale']
    else:
        permit = tech and score > Fraction(threshold)
        vetoed = policy in ('weighted', 'regime') and model is not None and 'technical_signal' not in model.features
        # lead re-audit (QA-DEC-09): a vetoed decision is described by the veto sentence only
        why = [] if vetoed else ['Fitted research prediction exceeds frozen threshold' if permit else
                                 'Technical/fundamental contradiction or prediction below threshold']
    if policy in ('weighted', 'regime') and model is not None and 'technical_signal' not in model.features:
        permit = False
        why.append('Joint model must contain technical input; no arbitrary fusion weights')
    label = 'NO_TRADE' if not permit else ('TECHNICAL_CANDIDATE' if policy == 'observe' else 'RESEARCH_CANDIDATE')
    return dict(permit=permit, why=why, label=label, score=score)


def run_decide(policy, shadow, mname, fund, macro, tech, threshold=0.):
    as_of = F.T_MACRO_STALE if macro == 'stale' else F.T0
    eng = F.engines(store(macro), MODELS[mname], policy=policy, shadow=shadow, macro=macro != 'nosvc', threshold=threshold)
    return eng.decide(fund, as_of, {'signal': tech})


GRID = list(itertools.product(POLICIES, (True, False), MODELS, FUNDS, MACROS, (True, False)))


def test_truth_table_is_complete_and_matches_independent_spec():
    """1,600 combinations: policy x shadow x model x fundamentals x macro status x technical signal."""
    assert len(GRID) == 1600
    bad = []
    permits = 0
    for policy, shadow, mname, fund, macro, tech in GRID:
        model = MODELS[mname]
        d = run_decide(policy, shadow, mname, fund, macro, tech)
        e = spec_expected(policy, shadow, model, fund, macro, tech)
        key = (policy, shadow, mname, fund, macro, tech)
        got = (d['permit_signal'], d['decision'], d['explanation'])
        want = (e['permit'], e['label'], e['why'])
        if got != want:
            bad.append((key, got, want))
        # score arithmetic against the Fraction oracle
        s = d['fundamental']['score']
        if e['score'] is None:
            if s is not None:
                bad.append((key, 'score should be None', s))
        elif s is None or abs(Fraction(s) - e['score']) > Fraction(1, 10**12):
            bad.append((key, 'score', s, float(e['score'])))
        # invariants independent of the spec function
        if d['permit_signal']:
            permits += 1
            if not tech:
                bad.append((key, 'permit without technical signal'))
            if policy != 'observe' and (not shadow or model is None or s is None):
                bad.append((key, 'research permit without shadow/model/score'))
        if d['positionSize'] is not None or d['deployment_approved'] is not False or d['risk']['status'] != 'NOT_EVALUATED':
            bad.append((key, 'decision claims execution authority'))
        if d['decision'] == 'NO_TRADE' and d['permit_signal']:
            bad.append((key, 'label/permit mismatch'))
    assert not bad, f'{len(bad)} mismatches, first 5: {bad[:5]}'
    assert permits > 100


def test_explanation_for_weighted_without_technical_feature_describes_only_the_veto():
    """Lead re-audit (QA-DEC-09 fixed): the vetoed decision no longer also claims the prediction exceeded the threshold."""
    d = run_decide('weighted', True, 'fund', 'GOOD', 'full', True)
    assert d['permit_signal'] is False and d['decision'] == 'NO_TRADE'
    assert d['explanation'] == ['Joint model must contain technical input; no arbitrary fusion weights']


def test_explanation_must_not_claim_pass_when_vetoed():
    d = run_decide('weighted', True, 'fund', 'GOOD', 'full', True)
    assert not d['permit_signal']
    assert not any('exceeds' in line for line in d['explanation'])


# ---------------------------------------------------------------- hand-computed predictions
def test_hand_computed_scores_and_contributions():
    """GOOD: z=(1,1) -> .0625 + .5*1 + .25*1 = .8125 ; BAD: z=(-3,0) -> .0625 - 1.5 = -1.4375 (exact dyadic arithmetic)."""
    s = store('full')
    for sym, expected, parts in (('GOOD', .8125, {'revenue_yoy': .5, 'fcf_margin': .25}),
                                 ('BAD', -1.4375, {'revenue_yoy': -1.5, 'fcf_margin': 0.})):
        snap = FundamentalFeatureEngine(s).snapshot(sym, F.T0)
        out = FundamentalScoreEngine(F.M_FUND).score(snap['features'], F.T0)
        assert out['score'] == expected
        assert out['components'] == parts
        assert out['intercept'] + sum(out['components'].values()) == out['score']
        assert out['status'] == 'RESEARCH_MODEL' and out['deployment_approved'] is False
        assert 'not probability' in out['score_units'] and out['confidence'] is None


def test_missing_nan_inf_feature_gives_no_score_never_imputed():
    eng = FundamentalScoreEngine(F.M_FUND)
    for bad in (None, float('nan'), float('inf'), -float('inf')):
        out = eng.score({'revenue_yoy': .25, 'fcf_margin': bad}, F.T0)
        assert out['score'] is None and out['status'] == 'MISSING_FEATURES' and out['missing'] == ['fcf_margin']
    out = eng.score({'revenue_yoy': .25}, F.T0)
    assert out['score'] is None  # mean imputation would have produced .5625
    out = eng.score({}, F.T0)
    assert out['score'] is None


def test_overflow_fails_closed_and_blocks_decision():
    huge = F.artifact(('revenue_yoy',), (0.,), (1e-300,), (1e300,), 0.)
    out = FundamentalScoreEngine(huge).score({'revenue_yoy': .25}, F.T0)
    assert out['score'] is None and out['status'] == 'INVALID_MODEL_OUTPUT'
    d = F.engines(store('full'), huge, policy='filter').decide('GOOD', F.T0, {'signal': True})
    assert d['permit_signal'] is False and d['decision'] == 'NO_TRADE'


def test_score_unavailable_at_or_before_training_time_raises():
    eng = FundamentalScoreEngine(F.M_FUND)
    with pytest.raises(ValueError, match='historical'):
        eng.score({'revenue_yoy': .25, 'fcf_margin': .25}, '2023-12-31T00:00:00Z')    # == trained_through
    assert eng.score({'revenue_yoy': .25, 'fcf_margin': .25}, '2023-12-31T00:00:00.000001Z')['score'] == .8125
    # the combined decision propagates the error (fail loud, never a default score)
    with pytest.raises(ValueError):
        F.engines(store('full'), F.M_FUND, policy='observe').decide('GOOD', '2023-06-01T00:00:00Z', {'signal': True})


def test_monotonic_in_positive_coefficient_features_seeded():
    import random
    for seed in range(40):
        rng = random.Random(seed)
        n = rng.randint(1, 5)
        names = tuple(f'f{i}' for i in range(n))
        coefs = [rng.uniform(-2, 2) for _ in names]
        model = F.artifact(names, [rng.uniform(-1, 1) for _ in names], [rng.uniform(.1, 3) for _ in names], coefs, rng.uniform(-1, 1))
        base = {k: rng.uniform(-3, 3) for k in names}
        eng = FundamentalScoreEngine(model)
        s0 = eng.score(base, F.T0)['score']
        for k, c in zip(names, coefs):
            up = dict(base, **{k: base[k] + rng.uniform(.01, 2)})
            s1 = eng.score(up, F.T0)['score']
            assert (s1 > s0) == (c > 0) or c == 0, f'seed {seed} feature {k} coef {c}'
        assert abs(Fraction(s0) - F.oracle_score(model, base)) < Fraction(1, 10**12), f'seed {seed}'


# ---------------------------------------------------------------- thresholds
def _score_exact(threshold, *, tech=True):
    eng = F.engines(store('full'), F.M_FUND, policy='filter', threshold=threshold)
    return eng.decide('GOOD', F.T0, {'signal': tech})


def test_threshold_is_strictly_greater_than_at_ulp_resolution():
    s = .8125   # exactly representable
    up, down = math.nextafter(s, 2.), math.nextafter(s, 0.)
    assert _score_exact(s)['permit_signal'] is False                     # equality does not pass
    assert _score_exact(s)['explanation'] == ['Technical/fundamental contradiction or prediction below threshold']
    assert _score_exact(down)['permit_signal'] is True
    assert _score_exact(up)['permit_signal'] is False
    assert _score_exact(s - 1e-12)['permit_signal'] is True
    assert _score_exact(s + 1e-12)['permit_signal'] is False
    assert _score_exact(s, tech=False)['permit_signal'] is False and _score_exact(down, tech=False)['permit_signal'] is False


def test_threshold_point_zero_seven_style_boundaries():
    flat = F.artifact(('revenue_yoy',), (.25,), (.125,), (.5,), .07)       # feature equals mean -> score == intercept == .07
    for threshold, want in ((.069999, True), (.07, False), (.070001, False), (.0699999999999, True), (-1., True)):
        d = F.engines(store('full'), flat, policy='filter', threshold=threshold).decide('GOOD', F.T0, {'signal': True})
        assert d['fundamental']['score'] == .07
        assert d['permit_signal'] is want, threshold
    # negative score vs negative threshold
    neg = F.artifact(('revenue_yoy',), (.25,), (.125,), (.5,), -.07)
    for threshold, want in ((-.070001, True), (-.07, False), (-.069999, False)):
        d = F.engines(store('full'), neg, policy='filter', threshold=threshold).decide('GOOD', F.T0, {'signal': True})
        assert d['permit_signal'] is want, threshold


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_threshold_rejected(bad):
    with pytest.raises(ValueError):
        F.engines(store('full'), F.M_FUND, threshold=bad)


def test_unknown_policy_rejected_and_mixed_store_rejected():
    with pytest.raises(ValueError):
        F.engines(store('full'), F.M_FUND, policy='bogus')
    other = F.build_store(macro='none')
    from fundamentals.models import MacroRegimeService
    with pytest.raises(ValueError, match='consistent'):
        CombinedSignalEngine(FundamentalFeatureEngine(store('full')), macro=MacroRegimeService(other))


# ---------------------------------------------------------------- the user's named scenarios
def test_excellent_fundamentals_with_bearish_technicals_is_no_trade():
    d = F.engines(store('full'), F.M_FUND, policy='filter').decide('GOOD', F.T0, {'signal': False})
    assert d['fundamental']['score'] == .8125 and d['permit_signal'] is False and d['decision'] == 'NO_TRADE'


def test_poor_fundamentals_with_bullish_momentum_is_no_trade_in_filter_but_trade_in_observe():
    f = F.engines(store('full'), F.M_FUND, policy='filter').decide('BAD', F.T0, {'signal': True})
    assert f['fundamental']['score'] == -1.4375 and f['permit_signal'] is False
    o = F.engines(store('full'), F.M_FUND, policy='observe').decide('BAD', F.T0, {'signal': True})
    assert o['permit_signal'] is True and o['decision'] == 'TECHNICAL_CANDIDATE'     # observe never blocks (documented)


def test_regime_dependent_selection_follows_macro_status():
    fresh = F.engines(store('full'), F.M_JOINT, policy='regime').decide('GOOD', F.T0, {'signal': True})
    stale = F.engines(store('full'), F.M_JOINT, policy='regime').decide('GOOD', F.T_MACRO_STALE, {'signal': True})
    incomplete = F.engines(store('nofed'), F.M_JOINT, policy='regime').decide('GOOD', F.T0, {'signal': True})
    assert fresh['macro']['status'] == 'AVAILABLE' and fresh['permit_signal'] is True
    assert stale['macro']['status'] == 'STALE' and stale['permit_signal'] is False
    assert incomplete['macro']['status'] == 'INCOMPLETE' and incomplete['permit_signal'] is False
    assert stale['explanation'] == incomplete['explanation'] == ['Macro vintage coverage missing/stale']
    # the non-regime policy ignores macro status
    assert F.engines(store('nofed'), F.M_JOINT, policy='filter').decide('GOOD', F.T0, {'signal': True})['permit_signal'] is True


def test_macro_staleness_boundary_is_floor_days_strictly_over_100():
    """CPI/UNRATE/FEDFUNDS last obs 2024-01 available 2024-02-15T12:00Z: age = floor(days). 100 days -> AVAILABLE, 101 -> STALE."""
    from fundamentals.models import MacroRegimeService
    m = MacroRegimeService(store('full'))
    avail = __import__('pandas').Timestamp('2024-02-15T12:00:00Z')
    assert m.state(avail + __import__('pandas').Timedelta(days=100, hours=23, minutes=59))['status'] == 'AVAILABLE'
    assert m.state(avail + __import__('pandas').Timedelta(days=101))['status'] == 'STALE'
    assert m.state(avail + __import__('pandas').Timedelta(days=100, hours=23, minutes=59, seconds=59))['status'] == 'AVAILABLE'


def test_fundamental_staleness_boundary_550_days():
    """age = (as_of date - period_end).days; FY end 2023-12-31 -> 550 days = 2025-07-03 AVAILABLE, 551 days = 2025-07-04 STALE."""
    assert (__import__('datetime').date(2025, 7, 3) - __import__('datetime').date(2023, 12, 31)).days == 550
    eng = FundamentalFeatureEngine(store('full'))
    assert eng.snapshot('GOOD', '2025-07-03T23:59:59Z')['status'] == 'AVAILABLE'
    assert eng.snapshot('GOOD', '2025-07-04T00:00:00Z')['status'] == 'STALE'
    assert eng.snapshot('GOOD', '2025-07-03T23:59:59Z')['age_days'] == 550


# ---------------------------------------------------------------- catalysts and annotations never carry authority
def _clone(kind='full'):
    return F.build_store(macro='full')


def test_catalysts_and_annotations_do_not_change_permit_or_score():
    base = _clone()
    loud = _clone()
    F.add_earnings(loud, eps=9.99, consensus=1.0)                          # +899% surprise
    ingest_disclosures(loud, symbol='GOOD', document='We raised our guidance. We reported strong demand.', source='pr-1',
                       published_at='2024-02-21T00:00:00Z', observed_at='2024-02-21T00:00:00Z')
    quiet = _clone()
    F.add_earnings(quiet, eps=-5.0, consensus=1.0)                         # disastrous miss
    ingest_disclosures(quiet, symbol='GOOD', document='We lowered our guidance. We breached a debt covenant.', source='pr-2',
                       published_at='2024-02-21T00:00:00Z', observed_at='2024-02-21T00:00:00Z')
    outs = [F.engines(s, F.M_FUND, policy='filter', half_life=3.).decide('GOOD', F.T0, {'signal': True}) for s in (base, loud, quiet)]
    assert [o['permit_signal'] for o in outs] == [True, True, True]
    assert len({o['fundamental']['score'] for o in outs}) == 1
    # but the evidence is recorded in the snapshot id: changing catalysts changes data_snapshot_id
    assert len({o['data_snapshot_id'] for o in outs}) == 3
    assert [len(o['catalysts']['events']) for o in outs] == [0, 2, 2]
    for o in outs[1:]:
        assert o['catalysts']['score'] is None and o['catalysts']['confidence'] is None
        for ev in o['catalysts']['events']:
            assert ev['payload'].get('execution_authority', False) is False


def test_decision_is_deterministic_and_snapshot_id_binds_inputs():
    a = F.engines(store('full'), F.M_FUND, policy='filter').decide('GOOD', F.T0, {'signal': True})
    b = F.engines(store('full'), F.M_FUND, policy='filter').decide('GOOD', F.T0, {'signal': True})
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert len(a['data_snapshot_id']) == 64
    c = F.engines(store('full'), F.M_FUND, policy='filter').decide('GOOD', '2024-03-06T04:00:00Z', {'signal': True})
    assert c['data_snapshot_id'] != a['data_snapshot_id']           # as_of is part of the snapshot
    d = F.engines(store('full'), F.M_FUND, policy='filter').decide('BAD', F.T0, {'signal': True})
    assert d['data_snapshot_id'] != a['data_snapshot_id']


def test_technical_signal_nan_and_numpy_truthiness_documented():
    """bool(NaN) is True: a NaN 'signal' counts as a signal in decide() (and in PaperEngine). Documented, features() never emits NaN."""
    import numpy as np
    d = F.engines(store('full'), F.M_FUND, policy='filter').decide('GOOD', F.T0, {'signal': float('nan')})
    assert d['technical']['signal'] is True
    d = F.engines(store('full'), F.M_FUND, policy='filter').decide('GOOD', F.T0, {'signal': np.bool_(False)})
    assert d['technical']['signal'] is False and d['permit_signal'] is False
    d = F.engines(store('full'), F.M_FUND, policy='filter').decide('GOOD', F.T0, {})
    assert d['technical']['signal'] is False


def test_etf_asset_class_never_gets_fundamental_score():
    eng = F.engines(store('full'), F.M_FUND, policy='filter')
    d = eng.decide('GOOD', F.T0, {'signal': True}, asset_class='etf')
    assert d['fundamental']['data_status'] == 'REQUIRES_DATED_HOLDINGS' and d['fundamental']['score'] is None
    assert d['permit_signal'] is False
    # a macro+technical-only model is allowed for an ETF (documented joint macro model path)
    d2 = F.engines(store('full'), F.M_MACRO, policy='filter').decide('GOOD', F.T0, {'signal': True}, asset_class='etf')
    assert d2['permit_signal'] is True and d2['fundamental']['score'] == pytest.approx(.875)


def test_coverage_is_reported_separately_from_score():
    d = F.engines(store('full'), F.M_FUND, policy='filter').decide('GOOD', F.T0, {'signal': True})
    assert d['fundamental']['score'] == .8125 and 0 < d['fundamental']['coverage'] < .2      # score is high, completeness is low
    assert d['fundamental']['confidence'] is None
    assert d['permit_signal'] is True        # LIMITATION: low completeness does not gate (only the model inputs must exist)


def test_decision_object_has_full_audit_content():
    d = F.engines(store('full'), F.M_FUND, policy='filter', half_life=5.).decide('GOOD', F.T0, {'signal': True})
    for k in ('symbol', 'timestamp', 'data_snapshot_id', 'configuration', 'technical', 'fundamental', 'macro', 'catalysts', 'risk',
              'decision', 'permit_signal', 'policy', 'explanation'):
        assert k in d
    for k in ('score', 'confidence', 'features', 'raw', 'flags', 'coverage', 'lineage', 'feature_version', 'available_at', 'period_end', 'components'):
        assert k in d['fundamental']
    cfg = d['configuration']
    assert list(cfg['model']['coefficients']) == [.5, .25] and cfg['threshold'] == 0. and cfg['policy'] == 'filter'
    assert cfg['catalyst_half_life'] == 5. and cfg['macro_max_age'] == 100 and cfg['max_period_age_days'] == 550
    assert d['fundamental']['raw']['revenue'] == 160. and d['fundamental']['raw']['ocf'] == 48.
    assert d['fundamental']['available_at'] == '2024-02-20T21:00:00+00:00' and d['fundamental']['period_end'] == '2023-12-31'
    json.dumps(d, allow_nan=False)


# ---------------------------------------------------------------- seeded randomized end-to-end differential (store -> snapshot -> decide)
@pytest.mark.parametrize('seed', range(150))
def test_random_companies_models_thresholds_match_fraction_oracle(seed):
    import random
    rng = random.Random(seed)
    prior = rng.randint(50, 500)
    rev = max(1, int(prior * rng.uniform(.4, 2.2)))
    ocf, capex = rng.randint(0, rev), rng.randint(0, max(1, rev // 3))
    s = F.PointInTimeFundamentalStore()
    s.ingest(F.company_facts('RND', rev_prior=float(prior), rev=float(rev), ocf=float(ocf), capex=float(capex)))
    s.ingest(F.macro_facts())
    inputs = {'revenue_yoy': Fraction(rev, prior) - 1, 'fcf_margin': Fraction(ocf - capex, rev), 'macro_unemployment': Fraction(5)}
    tech = rng.random() < .6
    names = rng.sample(['revenue_yoy', 'fcf_margin', 'macro_unemployment', 'technical_signal'], rng.randint(1, 4))
    model = F.artifact(names, [rng.uniform(-1, 1) for _ in names], [rng.uniform(.05, 2) for _ in names],
                       [rng.uniform(-2, 2) for _ in names], rng.uniform(-1, 1))
    allin = dict(inputs, technical_signal=Fraction(int(tech)))
    exact = F.oracle_score(model, allin)
    threshold = rng.choice([0., float(exact) + rng.choice([-1, 1]) * rng.choice([1e-3, 1e-6, .5]), rng.uniform(-1, 1)])
    policy = rng.choice(['filter', 'selection', 'weighted', 'regime'])
    d = F.engines(s, model, policy=policy, threshold=threshold, shadow=True).decide('RND', F.T0, {'signal': tech})
    got = d['fundamental']['score']
    assert got is not None and abs(Fraction(got) - exact) < Fraction(1, 10**12), f'seed {seed}'
    needs_tech = policy in ('weighted', 'regime') and 'technical_signal' not in names
    want = (not needs_tech) and tech and exact > Fraction(threshold)
    if abs(exact - Fraction(threshold)) > Fraction(1, 10**9):
        assert d['permit_signal'] is want, f'seed {seed} policy {policy} score {got} threshold {threshold}'


# ---------------------------------------------------------------- confidence calibration inside the decision
def test_calibrated_confidence_matches_independent_wilson_interval_and_is_time_gated():
    from fundamentals.confidence import ConfidenceCalibration
    cal = ConfidenceCalibration(F.M_FUND.id, '2024-01-15T00:00:00Z', ((10, 4), (20, 15)), 'dec-sha', 30, .2)
    feat = FundamentalFeatureEngine(store('full'))
    eng = CombinedSignalEngine(feat, FundamentalScoreEngine(F.M_FUND, cal), policy='filter', research_shadow=True)
    d = eng.decide('GOOD', F.T0, {'signal': True})
    n, k = 20, 15
    p, z = k / n, 1.959963984540054
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    c = d['fundamental']['confidence_details']
    assert d['fundamental']['confidence'] == .75 and c['observations'] == 20 and c['status'] == 'HELDOUT_EMPIRICAL'
    assert c['interval_95'][0] == pytest.approx(centre - half, rel=1e-12) and c['interval_95'][1] == pytest.approx(centre + half, rel=1e-12)
    assert d['fundamental']['confidence'] != d['fundamental']['coverage']          # confidence is not data coverage
    # negative score uses the other bin: BAD score -1.4375 -> 4/10
    bad = eng.decide('BAD', F.T0, {'signal': True})
    assert bad['fundamental']['confidence'] == .4
    # a calibration made after the decision time is unusable (fail loud, no stale probability)
    late = ConfidenceCalibration(F.M_FUND.id, '2024-06-01T00:00:00Z', ((10, 4), (20, 15)), 'dec-sha', 30, .2)
    with pytest.raises(ValueError, match='unavailable'):
        CombinedSignalEngine(feat, FundamentalScoreEngine(F.M_FUND, late), policy='filter', research_shadow=True).decide('GOOD', F.T0, {'signal': True})
    # calibration for a different model is refused at construction
    with pytest.raises(ValueError, match='mismatch'):
        FundamentalScoreEngine(F.M_JOINT, cal)


def test_model_identity_is_recorded_in_the_decision_and_is_a_content_hash():
    d = F.engines(store('full'), F.M_FUND, policy='filter').decide('GOOD', F.T0, {'signal': True})
    assert d['fundamental']['model_id'] == F.M_FUND.id and d['fundamental']['trained_through'] == F.M_FUND.trained_through
    assert F.M_FUND.id != F.artifact(('revenue_yoy', 'fcf_margin'), (.125, .125), (.125, .125), (.5, .25), .0626).id


@pytest.mark.parametrize('asset_class,status', [('etf', 'REQUIRES_DATED_HOLDINGS'), ('forex', 'REQUIRES_MACRO_MODEL'), ('crypto', 'REQUIRES_NATIVE_DATA'),
                                                ('bond', 'REQUIRES_ASSET_ADAPTER'), ('', 'REQUIRES_ASSET_ADAPTER')])
def test_non_equity_asset_classes_get_no_corporate_features_and_cannot_pass_a_fundamental_model(asset_class, status):
    d = F.engines(store('full'), F.M_FUND, policy='filter').decide('GOOD', F.T0, {'signal': True}, asset_class=asset_class)
    assert d['fundamental']['data_status'] == status and d['fundamental']['features'] == {} and d['fundamental']['score'] is None
    assert d['permit_signal'] is False and d['explanation'] == ['Missing or unvalidated model/feature coverage']


def test_observe_with_a_model_trained_after_decision_time_raises_instead_of_passing_the_technical_signal_through():
    """Observation (not a defect): 'observe' still evaluates an attached model, so as_of <= trained_through raises ValueError."""
    late = F.artifact(('revenue_yoy',), (.25,), (.125,), (.5,), .5, trained_through='2025-01-01T00:00:00Z')
    with pytest.raises(ValueError, match='historical'):
        F.engines(store('full'), late, policy='observe').decide('GOOD', F.T0, {'signal': True})


# ---------------------------------------------------------------- malformed catalyst payloads (stored without validation)
MALFORMED_EARNINGS = [
    dict(reported_eps='1.1', consensus_eps=1.0, consensus_available_at='2024-02-01T00:00:00Z'),      # str EPS -> TypeError
    dict(reported_eps=[1.1], consensus_eps=1.0, consensus_available_at='2024-02-01T00:00:00Z'),       # list EPS -> TypeError
    dict(reported_eps=1.1, consensus_eps=1.0, consensus_available_at='2024-02-01'),                    # naive timestamp -> ValueError
    dict(reported_eps=1.1, consensus_eps=1.0, consensus_available_at='garbage'),                       # DateParseError
    dict(reported_eps=True, consensus_eps=1.0, consensus_available_at='2024-02-01T00:00:00Z'),         # bool silently treated as 1.0
]


def _store_with_bad_earnings(payload):
    s = F.build_store()
    s.add_event(symbol='GOOD', event_type='earnings', event_at='2024-02-20T21:00:00Z', reported_at='2024-02-20T21:00:00Z',
                available_at='2024-02-20T21:05:00Z', source='x', revision='A', payload=dict(payload, event_key='bad'))
    return s


def test_malformed_earnings_payload_is_flagged_and_never_aborts_a_later_decision():
    """Lead re-audit (QA-DEC-11 / QA-MAC-02 fixed): a malformed (string/list/bool/naive-timestamp) earnings payload is still
    stored, but it is flagged with no surprise value and every later decide() keeps working."""
    for p in MALFORMED_EARNINGS[:4]:
        s = _store_with_bad_earnings(p)
        for t in (F.T0, '2024-06-01T00:00:00Z'):
            d = F.engines(s, F.M_FUND, policy='filter').decide('GOOD', t, {'signal': True})
            event = d['catalysts']['events'][0]
            assert event.get('eps_surprise') is None and event.get('quality_flag') in ('INVALID_EARNINGS_PAYLOAD', 'INVALID_CONSENSUS_TIMESTAMP', 'NO_PRE_EVENT_CONSENSUS')


def test_qa_dec_11_malformed_catalyst_payload_must_be_flagged_not_abort_or_corrupt_the_decision():
    for p in MALFORMED_EARNINGS:
        s = _store_with_bad_earnings(p)
        d = F.engines(s, F.M_FUND, policy='filter').decide('GOOD', F.T0, {'signal': True})
        assert d['catalysts']['events'][0].get('eps_surprise') is None, p


def test_decide_uses_publication_time_only_there_is_no_observed_at_operational_mode():
    """LIMITATION (design: research replay reconstructs public knowledge). A filing published 2024-02-20 but only INGESTED on
    2024-04-25 is used by a decision at 2024-03-05, so re-running a live decision after a late backfill changes the snapshot.
    store.as_of(observed_only=True) exists but snapshot()/decide()/PaperEngine cannot request it."""
    from dataclasses import replace
    s = F.PointInTimeFundamentalStore()
    late = '2024-04-25T00:00:00Z'
    s.ingest([replace(f, observed_at=late) for f in F.company_facts('GOOD', rev_prior=128., rev=160., ocf=48., capex=8.)])
    assert s.as_of('GOOD', F.T0, observed_only=True) == []                               # operational view: nothing known yet
    d = F.engines(s, F.M_FUND, policy='filter', macro=False, catalysts=False).decide('GOOD', F.T0, {'signal': True})
    assert d['fundamental']['score'] == .8125 and d['permit_signal'] is True              # decide() ignores observed_at
    import inspect
    assert 'observed_only' not in inspect.signature(FundamentalFeatureEngine.snapshot).parameters


def test_selection_policy_is_identical_to_filter_there_is_no_ranked_selection():
    """LIMITATION / NOT IMPLEMENTED: 'selection' ("ranked selection") has no ranking code; it equals 'filter' on every input."""
    for mname, fund, macro, tech in itertools.product(MODELS, FUNDS, ('full', 'stale'), (True, False)):
        a = run_decide('filter', True, mname, fund, macro, tech)
        b = run_decide('selection', True, mname, fund, macro, tech)
        for k in ('permit_signal', 'decision', 'explanation'):
            assert a[k] == b[k], (mname, fund, macro, tech)
        assert a['fundamental']['score'] == b['fundamental']['score']


# ---------------------------------------------------------------- real HTTP: /strategies/{id}/explanation agrees with in-process decide()
def _serve(db_path):
    import threading
    from fundamentals.api import serve
    for port in range(8890, 8900):
        try:
            srv = serve(str(db_path), port=port)
            break
        except OSError:
            continue
    else:
        pytest.skip('no free port in 8890-8899')
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, port


def test_http_explanation_endpoint_matches_in_process_decision(tmp_path):
    import urllib.request
    import urllib.parse
    path = tmp_path / 'fund.sqlite'
    s = F.build_store(path)
    F.add_earnings(s)
    s.close()
    srv, port = _serve(path)
    try:
        def get(sym, signal, as_of=F.T0, asset='equity'):
            q = urllib.parse.urlencode(dict(symbol=sym, as_of=as_of, technical_signal=signal, asset_class=asset))
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/strategies/anything/explanation?{q}', timeout=20) as r:
                assert r.headers['Cache-Control'] == 'no-store'
                return json.loads(r.read(), parse_constant=lambda c: (_ for _ in ()).throw(AssertionError('non-finite token ' + c)))
        ro = F.PointInTimeFundamentalStore(path, readonly=True)
        for sym in ('GOOD', 'BAD', 'NONE'):
            for signal in ('true', 'false'):
                remote = get(sym, signal)
                mine = F.engines(ro, None, policy='observe', shadow=False).decide(sym, F.T0, {'signal': signal == 'true'})
                assert remote['data_snapshot_id'] == mine['data_snapshot_id'], (sym, signal)
                assert remote['permit_signal'] == mine['permit_signal'] == (signal == 'true')
                assert remote['decision'] == mine['decision'] == ('TECHNICAL_CANDIDATE' if signal == 'true' else 'NO_TRADE')
                assert remote['explanation'] == ['Fundamental observation only; technical decision unchanged']
                assert remote['positionSize'] is None and remote['deployment_approved'] is False and remote['risk']['status'] == 'NOT_EVALUATED'
                assert remote['fundamental']['score'] is None and remote['fundamental']['confidence'] is None
        # default asset class for this endpoint is 'etf' (explanation of a corporate symbol must say it has no fundamentals)
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/strategies/x/explanation?symbol=GOOD&as_of={F.T0}', timeout=20) as r:
            body = json.loads(r.read())
        assert body['fundamental']['data_status'] == 'REQUIRES_DATED_HOLDINGS'
        ro.close()
    finally:
        srv.shutdown()
        srv.server_close()
