"""Track G (prefix bt): robustness runners (cost, split, alpha, threshold, setup) built from public SUT entry points."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from tests.qa_reaudit import bt_runner as run
from tests.qa_reaudit import bt_oracle as oracle

MACRO = ('macro_inflation_yoy', 'macro_unemployment_change_3m', 'macro_policy_rate_change_3m')


def load_states():
    return json.loads((run.ensure_repro() / 'macro_states.json').read_text())


def fit_models(frame, symbol, states, train_frac, val_frac, alphas=(.1, 1., 10., 100.), horizon=5, setup=None):
    """Replicates the preregistered fitting recipe at arbitrary split fractions; returns dict name->(artifact, val_mse)."""
    from fundamentals.research import label_panel, purged_training
    from fundamentals.models import fit_model
    from strategies.small_account import Setup, features
    f = features(frame, setup or Setup('bollinger', 20))
    n = len(f)
    fit_at = pd.Timestamp(f.ts.iloc[int(n * train_frac)])
    test_at = pd.Timestamp(f.ts.iloc[int(n * val_frac)])
    panel = label_panel(f, states, horizon=horizon)
    train = purged_training(panel, fit_at)
    val = panel[(panel.as_of > fit_at) & (panel.label_end < test_at)]
    out = {}
    for name, inputs in (('macro_filter_A_B', MACRO), ('joint_C', (*MACRO, 'technical_signal'))):
        tr, va = train.dropna(subset=list(inputs)), val.dropna(subset=list(inputs))
        cands = []
        for a in alphas:
            art = fit_model(tr, inputs, fit_at=fit_at, alpha=a, horizon=horizon)
            x = va[list(inputs)].to_numpy()
            pred = ((x - np.array(art.means)) / np.array(art.scales)) @ np.array(art.coefficients) + art.intercept
            cands.append((float(np.mean((pred - va.target.to_numpy()) ** 2)), a, art))
        mse, a, art = min(cands, key=lambda c: c[0])
        out[name] = (art, mse, a)
    return out


def replay_window(system, symbol, store, art_map, frame, begin, end, cost_multiple=1, threshold=0.0, setup=None):
    from fundamentals.models import CombinedSignalEngine, FundamentalScoreEngine, MacroRegimeService
    from fundamentals.features import FundamentalFeatureEngine
    from fundamentals.research import replay_with_fundamentals
    from strategies.small_account import Setup
    fe, macro = FundamentalFeatureEngine(store), MacroRegimeService(store)
    combo = None
    if system != 'technical':
        key = 'joint_C' if system == 'joint_regime_D' else system
        combo = CombinedSignalEngine(fe, FundamentalScoreEngine(art_map[key][0]), macro,
                                     policy={'macro_filter_A_B': 'filter', 'joint_C': 'weighted', 'joint_regime_D': 'regime'}[system],
                                     research_shadow=True, threshold=threshold)
    return replay_with_fundamentals(frame, setup or Setup('bollinger', 20), symbol, begin, end, combo, cost_multiple=cost_multiple)


def total_return(engine):
    return engine.curve[-1]['equity'] / 100 - 1
