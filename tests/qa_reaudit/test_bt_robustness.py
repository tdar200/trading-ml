"""Track G (phases 167-170): regime / cost / parameter / strategy-family robustness of the ETF macro conclusions.

Every run goes through the real machinery (label_panel, purged_training, fit_model, CombinedSignalEngine, PaperEngine).
Outputs: results/qa_reaudit_20261001/bt/robustness_*.csv.
Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_bt_robustness.py
"""
import numpy as np
import pandas as pd
import pytest

from tests.qa_reaudit import bt_claims as claims
from tests.qa_reaudit import bt_robustness as rb
from tests.qa_reaudit import bt_runner as run

SPLITS = [(.30, .45, .60), (.40, .55, .70), (.45, .60, .80), (.50, .75, 1.00), (.55, .70, .90), (.60, .75, .90)]


@pytest.fixture(scope='module')
def states():
    return rb.load_states()


@pytest.fixture(scope='module')
def store():
    s = run.open_store()
    yield s
    s.close()


def _trades(e):
    return sum(1 for x in e.journal if x['event'] == 'EXIT')


# ---------------------------------------------------------------------------------------------- cost multipliers
def test_bt_cost_multipliers_1x_2x_3x_flip_the_macro_verdict_and_erase_the_technical_edge():
    rows = []
    for sym in run.SYMBOLS:
        for cm in (1, 2, 3):
            t, m = claims.metrics(sym, 'technical', cm), claims.metrics(sym, 'macro_filter_A_B', cm)
            rows.append({'symbol': sym, 'cost_multiple': cm, 'technical_final': t['final_equity'], 'macro_final': m['final_equity'],
                         'increment': m['final_equity'] - t['final_equity'], 'technical_trades': t['trades'], 'macro_trades': m['trades'],
                         'technical_fees': t['fees_gbp'], 'technical_sharpe': t['sharpe']})
    df = pd.DataFrame(rows)
    df.to_csv(run.OUT / 'robustness_cost_multiples.csv', index=False)
    piv = df.pivot(index='symbol', columns='cost_multiple', values='technical_final')
    # technical result worsens monotonically with costs for every ETF (fees rise linearly with the multiple)
    for sym in run.SYMBOLS:
        assert piv.loc[sym, 1] >= piv.loc[sym, 2] >= piv.loc[sym, 3] - 1e-9, sym
    # the only ETFs that are net-profitable at 1x (XLE, XLF, XLP, XLV) lose that profit at 3x for XLE; count them
    assert (piv[1] > 100).sum() == 4 and piv.loc['XLE', 3] < 100
    # The macro increment is mostly a trade-less-pay-less effect: it GROWS with the cost multiple for every ETF whose filter
    # removes trades, and the sign of the conclusion flips for XLK and XLV between 1x and 3x costs.
    inc = df.pivot(index='symbol', columns='cost_multiple', values='increment')
    for sym in ('SPY', 'XLF', 'XLK', 'XLP', 'XLV'):
        assert inc.loc[sym, 1] < inc.loc[sym, 2] < inc.loc[sym, 3], (sym, inc.loc[sym].tolist())
    flipped = {sym for sym in run.SYMBOLS if np.sign(round(inc.loc[sym, 1], 9)) != np.sign(round(inc.loc[sym, 3], 9)) and inc.loc[sym, 1] != 0}
    assert flipped == {'XLK', 'XLV'}
    assert (inc[1] > 1e-9).sum() == 2 and (inc[1] < -1e-9).sum() == 3  # 1x: helps SPY/XLF, hurts XLK/XLP/XLV
    assert (inc[3] > 1e-9).sum() == 4 and (inc[3] < -1e-9).sum() == 1  # 3x: helps SPY/XLF/XLK/XLV, only XLP still negative, XLE zero
    f = df.pivot(index='symbol', columns='cost_multiple', values='technical_fees')
    assert (f[2] / f[1]).min() > 1.5


# ---------------------------------------------------------------------------------------------- sub-periods / regimes
@pytest.fixture(scope='module')
def rolling(states, store):
    rows = []
    for (tf, vf, ef) in SPLITS:
        for sym in run.SYMBOLS:
            frame = run.load_prices(sym).reset_index(drop=True)
            n = len(frame)
            arts = rb.fit_models(frame, sym, states, tf, vf)
            b, e_ = int(n * vf), (int(n * ef) if ef < 1 else n)
            res = {s: rb.replay_window(s, sym, store, arts, frame, b, e_) for s in ('technical', 'macro_filter_A_B', 'joint_C')}
            rows.append({'split': f'train<{tf}/val<{vf}/test<{ef}', 'symbol': sym, 'test_start': str(frame.ts.iloc[b].date()), 'test_end': str(frame.ts.iloc[e_ - 1].date()),
                         'alpha_selected': arts['macro_filter_A_B'][2], 'technical': rb.total_return(res['technical']),
                         'macro_increment': rb.total_return(res['macro_filter_A_B']) - rb.total_return(res['technical']),
                         'joint_increment': rb.total_return(res['joint_C']) - rb.total_return(res['technical']),
                         'technical_trades': _trades(res['technical']), 'macro_trades': _trades(res['macro_filter_A_B'])})
    df = pd.DataFrame(rows)
    df.to_csv(run.OUT / 'robustness_rolling_origin.csv', index=False)
    return df


def test_bt_macro_filter_conclusion_flips_with_the_test_period(rolling):
    """6 rolling-origin windows (rate-hike, plateau, cut regimes). The cross-ETF mean macro increment changes sign between
    windows, the best ETF in one window is the worst in another, and in every window the filter blocks all trades for >=1 ETF."""
    g = rolling.groupby('split').macro_increment
    means = g.mean()
    assert (means > 0).any() and (means < 0).any()
    wins = rolling.pivot(index='symbol', columns='split', values='macro_increment')
    assert ((wins > 1e-9).sum(axis=1) > 0).sum() >= 4 and ((wins < -1e-9).sum(axis=1) > 0).sum() >= 4  # most ETFs both win and lose
    blocked = rolling[rolling.macro_trades == 0].groupby('split').size()
    assert len(blocked) == len(SPLITS)  # 'filter blocks everything' happens in every window


def test_bt_joint_model_is_inert_in_almost_every_window(rolling):
    """+joint equals technical in 35 of 36 ETF-window runs: the joint ablation carries no information."""
    zero = (rolling.joint_increment.abs() < 1e-12).sum()
    assert zero >= 34 and len(rolling) == 36


def test_bt_pooled_macro_increment_not_significant_in_any_window(rolling):
    from scipy import stats
    out = []
    for split, g in rolling.groupby('split'):
        t = stats.ttest_1samp(g.macro_increment, 0.0)
        out.append((split, float(g.macro_increment.mean()), float(t.pvalue)))
    assert all(p > 0.05 / len(SPLITS) for _, _, p in out)
    assert min(p for _, _, p in out) > 0.05


# ---------------------------------------------------------------------------------------------- alpha / threshold
def test_bt_alpha_grid_truncation_all_models_pick_the_largest_alpha_and_collapse_to_the_intercept(states, store):
    """Extending the grid to 1e5 keeps selecting a larger alpha for most ETFs: the coefficients shrink toward zero and the
    'macro filter' becomes the sign of the training-mean net return (an on/off switch), not a regime model."""
    rows = []
    shrink = 0
    for sym in run.SYMBOLS:
        frame = run.load_prices(sym).reset_index(drop=True)
        base = rb.fit_models(frame, sym, states, .5, .75)
        wide = rb.fit_models(frame, sym, states, .5, .75, alphas=(.1, 1., 10., 100., 1e3, 1e4, 1e5))
        a0, a1 = base['macro_filter_A_B'][2], wide['macro_filter_A_B'][2]
        c0 = np.abs(base['macro_filter_A_B'][0].coefficients).sum()
        c1 = np.abs(wide['macro_filter_A_B'][0].coefficients).sum()
        rows.append({'symbol': sym, 'alpha_grid_4': a0, 'alpha_grid_7': a1, 'coef_l1_grid4': c0, 'coef_l1_grid7': c1,
                     'mse_grid4': base['macro_filter_A_B'][1], 'mse_grid7': wide['macro_filter_A_B'][1]})
        shrink += a1 > a0
    pd.DataFrame(rows).to_csv(run.OUT / 'robustness_alpha_grid.csv', index=False)
    assert all(r['alpha_grid_4'] == 100.0 for r in rows)
    assert shrink >= 3  # at least three ETFs prefer an even larger penalty than any value in the preregistered grid


def test_bt_threshold_perturbation_flips_decisions(states, store):
    """Phase 110-111: shifting the economic break-even threshold by +-0.1/0.2 percent (score units are 5-session
    returns, typical |score| ~0.1-0.5 percent) changes the trade set for most ETFs: decisions are fragile to the threshold."""
    rows = []
    for sym in run.SYMBOLS:
        frame = run.load_prices(sym).reset_index(drop=True)
        n = len(frame)
        arts = rb.fit_models(frame, sym, states, .5, .75)
        for th in (-0.002, -0.001, 0.0, 0.001, 0.002):
            e = rb.replay_window('macro_filter_A_B', sym, store, arts, frame, int(n * .75), n, threshold=th)
            rows.append({'symbol': sym, 'threshold': th, 'trades': _trades(e), 'return': rb.total_return(e)})
    df = pd.DataFrame(rows)
    df.to_csv(run.OUT / 'robustness_threshold.csv', index=False)
    spread = df.groupby('symbol').trades.agg(lambda s: s.max() - s.min())
    assert (spread > 0).sum() >= 5, spread.to_dict()
    # at threshold 0 the numbers equal the preregistered macro filter
    zero = df[df.threshold == 0.0].set_index('symbol')
    for sym in run.SYMBOLS:
        assert zero.loc[sym, 'return'] == pytest.approx(claims.metrics(sym, 'macro_filter_A_B')['total_return'], abs=1e-12)


# ---------------------------------------------------------------------------------------------- other strategy families
def test_bt_macro_filter_on_other_rule_families_is_not_consistently_helpful(states, store):
    """Same frozen macro models (they do not depend on the rule) applied to five other pre-registered setups."""
    from strategies.small_account import Setup
    rows = []
    for fam, per in (('bollinger', 18), ('bollinger', 22), ('donchian', 20), ('ma', 20), ('rsi', 20)):
        setup = Setup(fam, per)
        for sym in run.SYMBOLS:
            frame = run.load_prices(sym).reset_index(drop=True)
            n = len(frame)
            arts = rb.fit_models(frame, sym, states, .5, .75)
            t = rb.replay_window('technical', sym, store, arts, frame, int(n * .75), n, setup=setup)
            m = rb.replay_window('macro_filter_A_B', sym, store, arts, frame, int(n * .75), n, setup=setup)
            rows.append({'setup': f'{fam}_{per}', 'symbol': sym, 'technical': rb.total_return(t), 'increment': rb.total_return(m) - rb.total_return(t),
                         'technical_trades': _trades(t), 'macro_trades': _trades(m)})
    df = pd.DataFrame(rows)
    df.to_csv(run.OUT / 'robustness_other_setups.csv', index=False)
    per_setup = df.groupby('setup').increment.mean()
    assert (per_setup > 0).any() and (per_setup < 0).any()
    # the increment is just minus the base return: a gate that removes trades helps exactly when the base rule loses
    assert df.technical.corr(df.increment) < -0.6
    nonzero = df[df.increment.abs() > 1e-12]
    assert (nonzero.increment > 0).any() and (nonzero.increment < 0).any()
