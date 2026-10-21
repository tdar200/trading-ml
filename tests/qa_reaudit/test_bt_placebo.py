"""Track G (phases 99-100, 111): is the macro filter's increment distinguishable from RANDOM trade skipping?

A placebo gate permits each technical entry attempt with the same probability the real macro filter permitted
(same number of expected trades) but ignores macro information. If the macro increments are not outside the placebo
distribution the 'macro effect' is a trade-less effect. Run:
.venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_bt_placebo.py
"""
import random

import numpy as np
import pandas as pd
import pytest

from tests.qa_reaudit import bt_claims as claims
from tests.qa_reaudit import bt_runner as run

SEEDS = 24


class RandomGate:
    """Delegates everything to a real CombinedSignalEngine but replaces permit_signal by a seeded coin flip."""
    def __init__(self, base, p, seed):
        self._base, self._p, self._rng = base, p, random.Random(seed)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def decide(self, symbol, as_of, technical, *, asset_class='equity'):
        d = dict(self._base.decide(symbol, as_of, technical, asset_class=asset_class))
        d['permit_signal'] = bool(technical.get('signal')) and self._rng.random() < self._p
        return d


def _placebo_total(sym, store, models, p, seed):
    from fundamentals.research import replay_with_fundamentals
    from strategies.small_account import Setup
    frame = run.load_prices(sym)
    base = run.build_combo('macro_filter_A_B', sym, store, models)
    n = len(frame)
    e = replay_with_fundamentals(frame, Setup('bollinger', 20), sym, int(n * .75), n, RandomGate(base, p, seed))
    return e.curve[-1]['equity'] / 100 - 1


def test_bt_macro_increments_are_not_better_than_random_trade_skipping():
    store, models = claims._store_models()
    rows = []
    for sym in ('SPY', 'XLF', 'XLK'):
        real = claims.engine(sym, 'macro_filter_A_B')
        dec = [e['decision'] for e in real.journal if e['event'] == 'FUNDAMENTAL_DECISION']
        p = sum(d['permit_signal'] for d in dec) / len(dec)
        tech = claims.metrics(sym, 'technical')['total_return']
        observed = claims.metrics(sym, 'macro_filter_A_B')['total_return'] - tech
        inc = [_placebo_total(sym, store, models, p, 100 + s) - tech for s in range(SEEDS)]
        rows.append({'symbol': sym, 'permit_rate': p, 'observed_increment': observed, 'placebo_mean': float(np.mean(inc)),
                     'placebo_sd': float(np.std(inc, ddof=1)), 'placebo_min': float(min(inc)), 'placebo_max': float(max(inc)),
                     'one_sided_p_observed_at_least_as_good': float((np.array(inc) >= observed).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(run.OUT / 'placebo_random_skip.csv', index=False)
    # no informative ETF beats random skipping at 5 percent (one-sided p of "placebo at least as good")
    for r in rows:
        assert r['one_sided_p_observed_at_least_as_good'] > 0.05, r
        assert r['placebo_min'] <= r['observed_increment'] <= r['placebo_max'], r  # inside the placebo range
    spy, xlf, xlk = rows
    # the macro gate is WORSE than random skipping on average for SPY and XLK and better only for XLF
    assert spy['observed_increment'] < spy['placebo_mean'] and xlk['observed_increment'] < xlk['placebo_mean']
    assert xlf['observed_increment'] > xlf['placebo_mean']
