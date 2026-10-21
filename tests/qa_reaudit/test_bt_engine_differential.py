"""Track G (phases 92-95): randomized differential test of the real PaperEngine against bt_sim_oracle (written from the rules).

Random OHLC paths with gaps, random signals/stop distances, DST-labelled timestamps; the journal (entries, exits, fees, pnl,
reasons) and the equity curve must agree to 1e-9. Seeds are in the test ids.
Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_bt_engine_differential.py
"""
import math
import random

import pandas as pd
import pytest

from tests.qa_reaudit import bt_sim_oracle as sim

SEEDS = list(range(150))


def _path(seed, n=160):
    rng = random.Random(seed)
    days = pd.bdate_range('2025-01-02', periods=n)
    ts = [pd.Timestamp(d).tz_localize('America/New_York').tz_convert('UTC') for d in days]  # midnight NY labels (DST aware)
    bars = []
    close = rng.uniform(20, 400)
    vol = rng.choice([0.004, 0.008, 0.015, 0.03])
    for t in ts:
        gap = rng.gauss(0, vol / 2) + (rng.choice([-1, 1]) * rng.uniform(0.03, 0.12) if rng.random() < 0.03 else 0.0)
        o = max(0.5, close * (1 + gap))
        c = max(0.5, o * (1 + rng.gauss(0, vol)))
        h = max(o, c) * (1 + abs(rng.gauss(0, vol / 2)))
        l = min(o, c) * (1 - abs(rng.gauss(0, vol / 2)))
        bars.append({'ts': t, 'open': round(o, 4), 'high': round(max(h, o, c), 4), 'low': round(min(l, o, c), 4), 'close': round(c, 4), 'volume': 1e6})
        bars[-1]['high'] = max(bars[-1]['high'], bars[-1]['open'], bars[-1]['close'])
        bars[-1]['low'] = min(bars[-1]['low'], bars[-1]['open'], bars[-1]['close'])
        close = bars[-1]['close']
    p = rng.choice([0.08, 0.2, 0.4])
    signals = [None]
    for i in range(1, n):
        b = bars[i - 1]
        signals.append({'signal': rng.random() < p, 'distance': round(b['close'] * rng.uniform(0.004, 0.03), 4), 'ts': b['ts'], 'close': b['close']})
    return bars, signals


@pytest.mark.parametrize('seed', SEEDS)
def test_bt_engine_matches_independent_rule_simulator(seed):
    from livesim.small_account import PaperEngine
    from strategies.small_account import Setup
    bars, signals = _path(seed)
    e = PaperEngine(Setup('donchian', 20), symbol='SIM', fee_bps=15., spread_bps=2., slip_bps=3., shadow=True)
    n = len(bars)
    for i, b in enumerate(bars):
        prev = None
        if signals[i] is not None:
            prev = {'ts': signals[i]['ts'], 'signal': signals[i]['signal'], 'distance': signals[i]['distance'], 'regime': 'x', 'close': signals[i]['close']}
        e.step(b, prev, close_final=i == n - 1)
    journal, curve = sim.simulate(bars, signals)
    mine = [(k, str(ts), p, q, f, pnl, r) for k, ts, p, q, f, pnl, r in journal]
    prod = [(x['event'], x['timestamp'], x['price'], x['quantity'], x['fee'], x.get('pnl'), x.get('reason', '')) for x in e.journal if x['event'] in ('ENTRY', 'EXIT')]
    assert len(mine) == len(prod), f'seed={seed}: {len(mine)} vs {len(prod)} events'
    for a, b in zip(mine, prod):
        assert a[0] == b[0] and a[1] == b[1], (seed, a, b)
        for x, y in zip(a[2:6], b[2:6]):
            if x is None:
                assert y is None
            else:
                assert float(x) == pytest.approx(y, rel=1e-9, abs=1e-9), (seed, a, b)
        if a[0] == 'EXIT':
            assert a[6] == b[6], (seed, a, b)
    assert len(curve) == len(e.curve) == n
    for x, c in zip(curve, e.curve):
        assert float(x) == pytest.approx(c['equity'], rel=1e-10, abs=1e-9), seed


def test_bt_differential_sample_is_not_vacuous():
    """The random paths exercise entries, all four exit reasons, partial sizing and a halt in a meaningful share of seeds."""
    reasons, entries, halts = set(), 0, 0
    for seed in SEEDS[:60]:
        bars, signals = _path(seed)
        journal, curve = sim.simulate(bars, signals)
        entries += sum(1 for j in journal if j[0] == 'ENTRY')
        reasons |= {j[6] for j in journal if j[0] == 'EXIT'}
        peak = 100.0
        for c in curve:
            peak = max(peak, float(c))
            if 1 - float(c) / peak >= 0.10:
                halts += 1
                break
    assert entries > 150
    assert {'stop', 'target', 'time', 'gap_stop', 'evaluation_end'} <= reasons
    assert halts >= 1
