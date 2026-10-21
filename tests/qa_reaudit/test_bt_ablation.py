"""Track G (phases 99-100, 105): ablations, significance and multiple-testing accounting for the ETF macro study.

Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_bt_ablation.py
Writes results/qa_reaudit_20261001/bt/ablation_significance.csv, ic_significance.csv, experiment_count.json
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from tests.qa_reaudit import bt_claims as claims
from tests.qa_reaudit import bt_oracle as oracle
from tests.qa_reaudit import bt_runner as run


@pytest.fixture(scope='module')
def repro():
    return run.ensure_repro()


@pytest.fixture(scope='module')
def comparison(repro):
    df = pd.read_csv(repro / 'comparison.csv')
    return {(r.symbol, r.system): r for r in df.itertuples(index=False)}


def _paired(symbol, cost=1, paths=20000, block=20, seed=947):
    a = np.diff(np.r_[100., [c['equity'] for c in claims.engine(symbol, 'technical', cost).curve]])
    b = np.diff(np.r_[100., [c['equity'] for c in claims.engine(symbol, 'macro_filter_A_B', cost).curve]])
    d = b - a
    n = len(d)
    rng = np.random.default_rng(seed)
    offs = [rng.integers(0, n, paths) for _ in range(-(-n // block))]
    tot = np.zeros(paths)
    for i in range(n):
        tot += d[(offs[i // block] + (i % block)) % n]
    lo, hi = np.quantile(tot, [.025, .975])
    p = min(1.0, 2 * min(float((tot <= 0).mean()), float((tot >= 0).mean())))
    sd = float(d.std(ddof=1))
    return {'increment_gbp': float(d.sum()), 'ci_lo': float(lo), 'ci_hi': float(hi), 'boot_p_two_sided': p,
            'daily_sd': sd, 'cohen_d_daily': float(d.mean() / sd) if sd > 0 else None, 'se_total': float(tot.std()), 'sessions': n}


def test_bt_incremental_numbers_and_paired_bootstrap_are_not_significant_after_multiple_testing(comparison):
    """Phase 99-100: technical vs +macro increments, CI, effect size and Holm/Bonferroni accounting over the 12 paired
    tests (6 ETFs x 2 cost levels). Nothing survives; the nonzero effects are < 1 GBP on a 100 GBP account."""
    rows = []
    for cost in (1, 2):
        for sym in run.SYMBOLS:
            r = _paired(sym, cost)
            r.update(symbol=sym, cost=cost,
                     technical_trades=claims.metrics(sym, 'technical', cost)['trades'], macro_trades=claims.metrics(sym, 'macro_filter_A_B', cost)['trades'])
            rows.append(r)
    df = pd.DataFrame(rows)
    m = len(df)
    order = df.boot_p_two_sided.sort_values().index
    holm = {}
    running = 0.0
    for k, idx in enumerate(order):
        running = max(running, min(1.0, (m - k) * df.boot_p_two_sided[idx]))
        holm[idx] = running
    df['p_bonferroni'] = np.minimum(1.0, df.boot_p_two_sided * m)
    df['p_holm'] = pd.Series(holm)
    df.to_csv(run.OUT / 'ablation_significance.csv', index=False)
    # reproduce the documented incremental numbers from the independent engines
    for sym in run.SYMBOLS:
        inc = claims.metrics(sym, 'macro_filter_A_B')['final_equity'] - claims.metrics(sym, 'technical')['final_equity']
        rep = comparison[(sym, 'macro_filter_A_B')].final_equity - comparison[(sym, 'technical')].final_equity
        assert inc == pytest.approx(rep, abs=1e-10)
    assert (df.p_holm > 0.05).all()
    assert (df[df.increment_gbp != 0].boot_p_two_sided > 0.01).all()
    assert df.increment_gbp.abs().max() < 2.0  # the largest effect is below 2 GBP on a 100 GBP account
    # trade counts are tiny: this is the sample behind every verdict
    assert df.technical_trades.max() <= 13 and df.macro_trades.min() == 0
    # no increment is even 2 bootstrap standard errors away from zero
    assert (df.increment_gbp.abs() <= 2 * df.se_total).all() and df.boot_p_two_sided.min() > 0.10


def test_bt_macro_filter_is_inert_blocking_or_gating_by_etf(repro):
    """Why the six-ETF 'macro ablation' has only three informative cases: the frozen filter score is >0 on EVERY
    diagnostic day for XLE (never filters), <0 on EVERY day for XLP and XLV (blocks everything, driven by the
    intercept = training-mean net-of-cost return, not by macro regime), and crosses zero only for SPY, XLF, XLK."""
    states = json.loads((repro / 'macro_states.json').read_text())
    models = {m['symbol']: m for m in json.loads((repro / 'frozen_models.json').read_text()) if m['model'] == 'macro_filter_A_B'}
    shape = {}
    for sym, m in models.items():
        px = run.load_prices(sym)
        sc = []
        for t in px.ts.iloc[int(len(px) * .75):]:
            f = states[str(t)]['features']
            x = [f['inflation_yoy'], f['unemployment_change_3m'], f['policy_rate_change_3m']]
            sc.append(m['intercept'] + sum((xi - mu) / s * c for xi, mu, s, c in zip(x, m['means'], m['scales'], m['coefficients'])))
        sc = np.array(sc)
        shape[sym] = ('always_go' if (sc > 0).all() else 'always_block' if (sc <= 0).all() else 'gates', int((np.diff((sc > 0).astype(int)) != 0).sum()))
    assert shape['XLE'][0] == 'always_go'
    assert shape['XLP'][0] == 'always_block' and shape['XLV'][0] == 'always_block'
    assert shape['SPY'][0] == shape['XLF'][0] == shape['XLK'][0] == 'gates'
    assert models['XLP']['intercept'] < 0 and models['XLV']['intercept'] < 0
    # joint models: the technical-signal term makes the score positive on every signal day, so +joint and +joint/regime
    # are identical to technical in all six ETFs by construction
    jm = {m['symbol']: m for m in json.loads((repro / 'frozen_models.json').read_text()) if m['model'] == 'joint_C'}
    for sym, m in jm.items():
        px = run.load_prices(sym)
        worst = None
        for t in px.ts.iloc[int(len(px) * .75):]:
            f = states[str(t)]['features']
            x = [f['inflation_yoy'], f['unemployment_change_3m'], f['policy_rate_change_3m'], 1.0]
            s = m['intercept'] + sum((xi - mu) / sc * c for xi, mu, sc, c in zip(x, m['means'], m['scales'], m['coefficients']))
            worst = s if worst is None else min(worst, s)
        assert worst > 0, (sym, worst)


def test_bt_regime_d_is_a_noop_because_macro_status_is_always_available(repro):
    states = json.loads((repro / 'macro_states.json').read_text())
    assert {v['status'] for v in states.values()} == {'AVAILABLE'}
    for sym in run.SYMBOLS:
        assert claims.engine(sym, 'joint_regime_D').curve == claims.engine(sym, 'joint_C').curve


def test_bt_ic_table_has_no_significant_features_once_overlap_and_quarterly_updates_are_respected(repro):
    """90 ICs. With IID p-values 52/90 look 'significant'; with an overlap-adjusted effective n = n/horizon only 8/90
    reach 5% unadjusted (4.5 expected by chance) and 0 survive Bonferroni. Macro features change once per quarter, so the
    real information content of a 366-day window is about 6 distinct macro states."""
    from scipy import stats
    ic = pd.read_csv(repro / 'information_coefficients.csv').dropna(subset=['time_series_spearman_ic'])
    out = []
    for r in ic.itertuples():
        n, h, rho = r.observations, r.horizon, r.time_series_spearman_ic
        neff = max(5.0, n / h)
        p_naive = 2 * (1 - stats.t.cdf(abs(rho * math.sqrt((n - 2) / (1 - rho * rho))), n - 2))
        p_eff = 2 * (1 - stats.t.cdf(abs(rho * math.sqrt((neff - 2) / (1 - rho * rho))), neff - 2))
        out.append({'symbol': r.symbol, 'horizon': h, 'feature': r.feature, 'ic': rho, 'n': n, 'p_iid': p_naive, 'p_effective_n': p_eff})
    d = pd.DataFrame(out)
    d.to_csv(run.OUT / 'ic_significance.csv', index=False)
    assert len(d) == 90
    assert (d.p_iid < .05).sum() >= 40
    assert 3 <= (d.p_effective_n < .05).sum() <= 12
    assert (d.p_effective_n < .05 / len(d)).sum() == 0
    # distinct macro feature vectors inside the 366-session diagnostic window
    states = json.loads((repro / 'macro_states.json').read_text())
    px = run.load_prices('SPY')
    vecs = {tuple(round(v, 12) for v in states[str(t)]['features'].values()) for t in px.ts.iloc[int(len(px) * .75):]}
    assert len(vecs) <= 8


def test_bt_experiment_count_for_multiple_testing_accounting(repro):
    """Count of configurations/looks behind the verdicts (phases 99-100/105), written to experiment_count.json."""
    research = json.loads((repro / 'research.json').read_text())
    earlier = json.loads((run.ROOT / 'results' / 'small_account' / 'research.json').read_text())
    n_candidates = len(earlier['candidates'])
    count = {
        'earlier_candidate_selection': {'candidates_18_setups_x_8_symbols': n_candidates, 'eligible_after_gates': earlier['eligible_count'],
                                        'exit_variants_development': len(earlier['exit_development_only']), 'cost_stress': len(earlier['cost_stress'])},
        'this_study': {'ridge_fits': 6 * 2 * 4,
                       'diagnostic_systems': len(research['results']), 'cost_stress_runs': 36, 'ic_values': len(research['ic']),
                       'paired_intervals': 12, 'xle_bootstrap_configs': 5, 'horizons': 5, 'features': 3, 'etfs': 6},
    }
    count['this_study']['hypothesis_like_outputs'] = (count['this_study']['diagnostic_systems'] + count['this_study']['cost_stress_runs']
                                                       + count['this_study']['ic_values'] + count['this_study']['paired_intervals'])
    count['bonferroni_alpha_for_ics'] = 0.05 / count['this_study']['ic_values']
    count['selection_note'] = ('the Bollinger-20 rule was chosen as the highest-ranked of %d candidates although eligible_count=%d; '
                               'the diagnostic window outcomes of all candidates were written to research.json before this study' % (n_candidates, earlier['eligible_count']))
    (run.OUT / 'experiment_count.json').write_text(json.dumps(count, indent=2))
    assert n_candidates == 144 and earlier['eligible_count'] == 0
    assert earlier['selected']['id'] == 'XLE_bollinger_20'
    assert count['this_study']['hypothesis_like_outputs'] == 30 + 36 + 90 + 12
    # the diagnostic return of the selected candidate was ALREADY visible in the earlier research file
    assert earlier['selected']['diagnostic']['final_equity'] == pytest.approx(100.01880867069762, abs=1e-12)


def test_bt_frozen_models_do_not_beat_a_constant_predictor_on_validation_for_half_the_etfs(repro):
    """No skill gate exists: research.json records validation_constant_mse but nothing requires beating it. Macro-only
    models are WORSE than the training-mean constant for XLK, XLP and XLV (and ~equal for XLE); joint models are worse for
    XLK and XLP. The models are frozen and used anyway."""
    models = json.loads((repro / 'frozen_models.json').read_text())
    worse = {(m['symbol'], m['model']) for m in models if m['validation_mse'] > m['validation_constant_mse']}
    assert {s for s, n in worse if n == 'macro_filter_A_B'} == {'XLK', 'XLP', 'XLV'}
    assert {s for s, n in worse if n == 'joint_C'} == {'XLK', 'XLP'}
    best_gain = max(1 - m['validation_mse'] / m['validation_constant_mse'] for m in models)
    assert best_gain < 0.03  # the best validation improvement over a constant is under 3 percent of MSE
    xle = next(m for m in models if m['symbol'] == 'XLE' and m['model'] == 'macro_filter_A_B')
    assert abs(1 - xle['validation_mse'] / xle['validation_constant_mse']) < 1e-4


def test_bt_training_label_horizon_is_not_the_horizon_of_the_traded_position():
    """LIMITATION (design): the ridge target is the 5-session open-to-open net return, but the engine exits at ATR stop,
    2R target or the 10-calendar-day time rule. Only 3 of 61 technical trades in the six ETF windows are held exactly 5
    sessions (median 7); 43 exit on the time rule, 18 by stop/target/gap before it."""
    holds = []
    for sym in run.SYMBOLS:
        px = run.load_prices(sym)
        idx = {str(t): i for i, t in enumerate(px.ts)}
        for ev in claims.engine(sym, 'technical').journal:
            if ev['event'] == 'EXIT':
                holds.append((ev['reason'], idx[ev['timestamp']] - idx[ev['entry_time']]))
    assert len(holds) == 61
    assert sum(1 for _, h in holds if h == 5) == 3
    assert np.median([h for _, h in holds]) == 7
    assert sum(1 for r, _ in holds if r == 'time') == 43


def test_bt_information_coefficients_match_an_independent_label_and_spearman_oracle(repro):
    """All 90 reported ICs (6 ETFs x 5 horizons x 3 macro features) from my own labels, sample window and rank correlation."""
    states = json.loads((repro / 'macro_states.json').read_text())
    ic = pd.read_csv(repro / 'information_coefficients.csv')
    feats = {'macro_inflation_yoy': 'inflation_yoy', 'macro_unemployment_change_3m': 'unemployment_change_3m', 'macro_policy_rate_change_3m': 'policy_rate_change_3m'}
    worst = 0.0
    for sym in run.SYMBOLS:
        px = run.load_prices(sym).reset_index(drop=True)
        n = len(px)
        test_at = px.ts.iloc[int(n * .75)]
        end = px.ts.iloc[-1] + pd.Timedelta(seconds=1)
        for h in (1, 5, 20, 60, 120):
            rows = []
            for i in range(n - h - 1):
                at, lab = px.ts.iloc[i + 1], px.ts.iloc[i + 1 + h]
                if at >= test_at and lab < end:
                    rows.append((states[str(at)]['features'], float(px.open.iloc[i + 1 + h] / px.open.iloc[i + 1] - 1 - 0.0038)))
            for name, key in feats.items():
                pairs = [(f[key], t) for f, t in rows if f[key] is not None]
                got = ic[(ic.symbol == sym) & (ic.horizon == h) & (ic.feature == name)].iloc[0]
                assert int(got.observations) == len(pairs), (sym, h, name)
                xs = [a for a, _ in pairs]
                if len(pairs) >= 20 and len(set(xs)) > 1:
                    mine = oracle.spearman(xs, [b for _, b in pairs])
                    worst = max(worst, abs(mine - got.time_series_spearman_ic))
                else:
                    assert pd.isna(got.time_series_spearman_ic)
    assert worst < 1e-9


def test_bt_buy_and_hold_benchmark_is_missing_from_the_docs_and_dwarfs_every_system():
    """The docs compare systems only with each other. Over the same 366-session diagnostic window buy-and-hold of the
    (split+dividend adjusted) ETF returned +24.9 percent SPY, +34.4 XLE, +5.6 XLF, +51.3 XLK, +7.0 XLP, +5.4 XLV while
    the technical rule returned -1.5..+1.9 percent with 1-3 percent average exposure."""
    rows = []
    for sym in run.SYMBOLS:
        px = run.load_prices(sym).reset_index(drop=True)
        b = int(len(px) * .75)
        bh = float(px.close.iloc[-1] / px.open.iloc[b] - 1)
        t = claims.metrics(sym, 'technical')
        rows.append({'symbol': sym, 'buy_and_hold_diagnostic': bh, 'technical_return': t['total_return'], 'macro_return': claims.metrics(sym, 'macro_filter_A_B')['total_return']})
        assert bh > max(t['total_return'], rows[-1]['macro_return'])
    pd.DataFrame(rows).to_csv(run.OUT / 'buy_and_hold_benchmark.csv', index=False)
    assert rows[0]['buy_and_hold_diagnostic'] == pytest.approx(0.2492, abs=5e-4)
