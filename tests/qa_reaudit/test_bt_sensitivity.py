"""Track G (phases 110-111): turnover / score stability of decisions in a frozen model.

1e-9 feature perturbations, exact-threshold behaviour, +-0.5 percent price noise and the share of bars that sit close
to a Bollinger band. Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_bt_sensitivity.py
"""
import json
import math
import random

import numpy as np
import pandas as pd
import pytest

from tests.qa_reaudit import bt_claims as claims
from tests.qa_reaudit import bt_runner as run


@pytest.fixture(scope='module')
def repro():
    return run.ensure_repro()


@pytest.fixture(scope='module')
def states(repro):
    return json.loads((repro / 'macro_states.json').read_text())


@pytest.fixture(scope='module')
def models(repro):
    return run.frozen_models(repro)


def _score_oracle(m, values):
    return m.intercept + sum((x - mu) / s * c for x, mu, s, c in zip(values, m.means, m.scales, m.coefficients))


def test_bt_feature_perturbation_of_1e9_changes_no_decision_and_score_is_lipschitz(models, states):
    """Score = intercept + sum((x-mean)/scale*coef). For +-1e-9 absolute feature noise the score may move by at most
    1e-9 * sum|coef/scale|; count decision flips over ALL diagnostic dates of ALL 12 frozen models."""
    from fundamentals.models import FundamentalScoreEngine
    flips = 0
    checked = 0
    rng = random.Random(1)
    for (sym, name), art in models.items():
        eng = FundamentalScoreEngine(art)
        lipschitz = sum(abs(c / s) for c, s in zip(art.coefficients, art.scales))
        px = run.load_prices(sym)
        for t in px.ts.iloc[int(len(px) * .75):]:
            f = states[str(t)]['features']
            base = {'macro_inflation_yoy': f['inflation_yoy'], 'macro_unemployment_change_3m': f['unemployment_change_3m'],
                    'macro_policy_rate_change_3m': f['policy_rate_change_3m'], 'technical_signal': float(rng.random() < 0.1)}
            vals = [base[k] for k in art.features]
            at = pd.Timestamp(t) + pd.Timedelta(seconds=1)
            s0 = eng.score(base, at)['score']
            assert s0 == pytest.approx(_score_oracle(art, vals), abs=1e-15)  # independent arithmetic
            for k in art.features:
                for d in (-1e-9, 1e-9):
                    p = dict(base)
                    p[k] = base[k] + d
                    s1 = eng.score(p, at)['score']
                    assert abs(s1 - s0) <= 1e-9 * lipschitz + 1e-15
                    flips += (s1 > 0) != (s0 > 0)
            checked += 1
    assert checked == 12 * 366
    assert flips == 0


def test_bt_threshold_is_strict_greater_than_at_the_exact_boundary(repro, models):
    """permit_signal requires score > threshold: at threshold == score (to the last bit) the signal is refused,
    one ulp lower it is permitted; one ulp above the score it is refused."""
    from fundamentals.models import CombinedSignalEngine, FundamentalScoreEngine, MacroRegimeService
    from fundamentals.features import FundamentalFeatureEngine
    store = run.open_store()
    art = models[('XLE', 'macro_filter_A_B')]
    t = run.load_prices('XLE').ts.iloc[1200]
    def decide(th):
        c = CombinedSignalEngine(FundamentalFeatureEngine(store), FundamentalScoreEngine(art), MacroRegimeService(store), policy='filter',
                                 threshold=th, research_shadow=True)
        return c.decide('XLE', t, {'signal': True}, asset_class='etf')
    base = decide(0.0)
    score = base['fundamental']['score']
    assert score > 0 and base['permit_signal'] is True
    assert decide(score)['permit_signal'] is False
    assert decide(float(np.nextafter(score, -np.inf)))['permit_signal'] is True
    assert decide(float(np.nextafter(score, np.inf)))['permit_signal'] is False
    for bad in (float('nan'), float('inf'), -float('inf')):
        with pytest.raises(ValueError):
            CombinedSignalEngine(FundamentalFeatureEngine(store), policy='observe', threshold=bad)
    store.close()


def test_bt_signal_without_technical_confirmation_is_never_permitted_whatever_the_score(repro, models):
    from fundamentals.models import CombinedSignalEngine, FundamentalScoreEngine, MacroRegimeService
    from fundamentals.features import FundamentalFeatureEngine
    store = run.open_store()
    art = models[('XLE', 'macro_filter_A_B')]
    c = CombinedSignalEngine(FundamentalFeatureEngine(store), FundamentalScoreEngine(art), MacroRegimeService(store), policy='filter',
                             threshold=-10.0, research_shadow=True)
    d = c.decide('XLE', run.load_prices('XLE').ts.iloc[1200], {'signal': False}, asset_class='etf')
    assert d['permit_signal'] is False and d['decision'] == 'NO_TRADE'
    store.close()


def _noisy(frame, seed, pct):
    rng = np.random.default_rng(seed)
    f = frame.copy()
    k = 1 + rng.uniform(-pct, pct, len(f))
    for c in ('open', 'high', 'low', 'close'):
        f[c] = f[c] * k
    return f


def test_bt_half_percent_price_noise_moves_trades_and_pnl_by_more_than_the_reported_effects(states):
    """+-0.5 percent iid multiplicative noise per bar (a bar keeps a valid OHLC): trade counts and the final equity of the
    TECHNICAL system move by amounts comparable to the macro 'effects' reported, i.e. the 1.0 GBP-scale increments in the
    docs are within data-noise sensitivity."""
    store = run.open_store()
    models = run.frozen_models(run.ensure_repro())
    rows = []
    for sym in ('SPY', 'XLE', 'XLF'):
        frame = run.load_prices(sym).reset_index(drop=True)
        base = run.run_system('technical', sym, store, models, frame=frame)
        b_eq = base.curve[-1]['equity']
        b_tr = sum(1 for e in base.journal if e['event'] == 'EXIT')
        for seed in range(6):
            nz = run.run_system('technical', sym, store, models, frame=_noisy(frame, seed, 0.005))
            rows.append({'symbol': sym, 'seed': seed, 'delta_equity': nz.curve[-1]['equity'] - b_eq, 'delta_trades': sum(1 for e in nz.journal if e['event'] == 'EXIT') - b_tr})
    store.close()
    df = pd.DataFrame(rows)
    df.to_csv(run.OUT / 'sensitivity_price_noise.csv', index=False)
    assert df.delta_trades.abs().max() >= 1  # the trade list itself changes
    assert df.delta_equity.abs().max() > 0.3  # up to 0.3+ GBP, the size of the reported macro effects
    assert df.delta_equity.abs().median() > 0.05


def test_bt_share_of_bars_within_half_percent_of_the_bollinger_band():
    """Fragility of the technical gate: fraction of bars whose close is within 0.5 percent of the lower band (so a 0.5%
    price error would flip that bar's signal)."""
    out = {}
    for sym in run.SYMBOLS:
        f = run.load_prices(sym).reset_index(drop=True)
        mean = f.close.rolling(20).mean()
        sd = f.close.rolling(20).std()
        band = mean - 2 * sd
        near = ((f.close - band).abs() / f.close < 0.005)
        out[sym] = {'near_band_bars': int(near.sum()), 'bars': int(band.notna().sum()), 'signals': int((f.close < band).sum())}
    pd.DataFrame(out).T.to_csv(run.OUT / 'sensitivity_band_proximity.csv')
    assert all(v['near_band_bars'] >= 0.5 * v['signals'] for v in out.values())  # as many near-misses as actual signals


@pytest.mark.parametrize('policy', ['filter', 'selection', 'weighted', 'regime'])
def test_bt_non_observe_policy_without_research_shadow_never_permits_even_with_a_fitted_model(repro, models, policy):
    """Docs claim 'unapproved filter fails closed': a fitted artifact without research_shadow=True cannot turn a technical
    signal into a permit under any of the four influence policies; 'observe' is the only passthrough."""
    from fundamentals.models import CombinedSignalEngine, FundamentalScoreEngine, MacroRegimeService
    from fundamentals.features import FundamentalFeatureEngine
    store = run.open_store()
    art = models[('XLE', 'joint_C')]
    t = run.load_prices('XLE').ts.iloc[1200]
    c = CombinedSignalEngine(FundamentalFeatureEngine(store), FundamentalScoreEngine(art), MacroRegimeService(store), policy=policy, threshold=-10.0)
    d = c.decide('XLE', t, {'signal': True}, asset_class='etf')
    assert d['permit_signal'] is False and d['decision'] == 'NO_TRADE' and d['deployment_approved'] is False
    o = CombinedSignalEngine(FundamentalFeatureEngine(store), policy='observe').decide('XLE', t, {'signal': True}, asset_class='etf')
    assert o['permit_signal'] is True and o['decision'] == 'TECHNICAL_CANDIDATE'
    store.close()
