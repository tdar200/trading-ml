"""Track G confirmed defects, each a minimal strict-xfail reproduction (flips loudly when fixed).

QA-BT-01 (claims wording) lives in test_bt_claims.py, QA-BT-03 (membership) in test_bt_survivorship.py.
Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_bt_defects.py
"""
import json
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from tests.qa_reaudit import bt_claims as claims
from tests.qa_reaudit import bt_runner as run


# --------------------------------------------------------------------------------------------------- QA-BT-02
def _series(seed, n=400):
    idx = pd.bdate_range('2021-01-04', periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


@pytest.mark.xfail(strict=True, reason='QA-BT-02 MEDIUM: models/features.py promises features "computed from data strictly BEFORE the '
                                       'date" but every feature (mom_6m, mom_1m, vol_*, ranks, spy_*) uses close[date]; the backtests '
                                       'fill at that same bar close (README Stage 4.5 fix 2): decision information == fill instant')
def test_bt_meta_features_must_not_depend_on_the_close_of_the_decision_date():
    from models.features import extract_features_for_date
    spy, ticker = _series(1), _series(2)
    date = ticker.index[250]
    base = extract_features_for_date({'SPY': spy, 'XLK': ticker}, ticker='XLK', date=date)
    spy2, tk2 = spy.copy(), ticker.copy()
    spy2.loc[date] *= 1.10
    tk2.loc[date] *= 1.10
    moved = extract_features_for_date({'SPY': spy2, 'XLK': tk2}, ticker='XLK', date=date)
    assert moved == base


def test_bt_meta_features_unaffected_by_bars_after_the_decision_date_canary():
    """Positive part of the review: appending/altering FUTURE bars changes nothing (pure function of data up to date)."""
    from models.features import extract_features_for_date
    spy, ticker = _series(3), _series(4)
    date = ticker.index[250]
    base = extract_features_for_date({'SPY': spy, 'XLK': ticker}, ticker='XLK', date=date)
    spy2, tk2 = spy.copy(), ticker.copy()
    spy2.loc[spy2.index > date] *= 9.0
    tk2.loc[tk2.index > date] *= 0.01
    assert extract_features_for_date({'SPY': spy2, 'XLK': tk2}, ticker='XLK', date=date) == base


# --------------------------------------------------------------------------------------------------- QA-BT-04
def _local_midnight_utc(day):
    return pd.Timestamp(day).tz_localize('America/New_York').tz_convert('UTC')


def _hold_exit_date(entry_day, days):
    from livesim.small_account import PaperEngine
    from strategies.small_account import Setup

    def bar(day, o):
        return {'ts': _local_midnight_utc(day), 'open': o, 'high': o + .3, 'low': o - .3, 'close': o, 'volume': 1e6}
    prev = bar(pd.Timestamp(entry_day) - pd.Timedelta(days=1), 100.)
    prev.update(signal=True, distance=2., regime='trend_up')
    e = PaperEngine(Setup('donchian', 20), symbol='XYZ', fee_bps=15, shadow=True)
    assert e.step(bar(entry_day, 100.), prev) == 'ENTRY'
    for d in days:
        if e.step(bar(d, 100.1), None) == 'EXIT':
            ex = [x for x in e.journal if x['event'] == 'EXIT'][-1]
            assert ex['reason'] == 'time'
            return pd.Timestamp(ex['timestamp']).tz_convert('America/New_York').date()
    raise AssertionError('no exit')


def test_bt_time_exit_control_without_dst_is_exactly_ten_calendar_days():
    assert _hold_exit_date('2025-02-18', [d.date() for d in pd.bdate_range('2025-02-19', '2025-03-07')]) == pd.Timestamp('2025-02-28').date()


@pytest.mark.xfail(strict=True, reason='QA-BT-04 LOW: livesim/small_account.py time exit compares elapsed SECONDS to hold*86400; '
                                       'across the US spring-forward the 10th calendar day is 9d23h away and the exit slips a session '
                                       '(entry 2025-03-04 exits 2025-03-17 not 2025-03-14); hits 2 reported XLP/XLV trades (entry 2026-03-06)')
def test_bt_time_exit_across_spring_dst_change_must_still_be_the_tenth_calendar_day():
    got = _hold_exit_date('2025-03-04', [d.date() for d in pd.bdate_range('2025-03-05', '2025-03-24')])
    assert got == pd.Timestamp('2025-03-14').date()


def test_bt_dst_delay_is_present_in_the_reproduced_reported_trades():
    late = set()
    for sym in run.SYMBOLS:
        dates = [t.tz_convert('America/New_York').date() for t in run.load_prices(sym).ts]
        for system in run.SYSTEMS:
            for ev in claims.engine(sym, system).journal:
                if ev['event'] == 'EXIT' and ev['reason'] == 'time':
                    entry = pd.Timestamp(ev['entry_time']).tz_convert('America/New_York').date()
                    got = pd.Timestamp(ev['timestamp']).tz_convert('America/New_York').date()
                    want = min(d for d in dates if d >= entry + pd.Timedelta(days=10))
                    if got != want:
                        late.add((sym, entry.isoformat(), got.isoformat(), want.isoformat()))
    assert late == {('XLP', '2026-03-06', '2026-03-17', '2026-03-16'), ('XLV', '2026-03-06', '2026-03-17', '2026-03-16')}


# --------------------------------------------------------------------------------------------------- QA-BT-05
def test_bt_saved_journal_must_round_trip_engine_values_exactly():
    saved = json.loads((run.ensure_repro() / 'technical_journal.json').read_text())
    live = claims.engine('XLE', 'technical').journal
    s = [e['pnl'] for e in saved if e['event'] == 'EXIT']
    l = [e['pnl'] for e in live if e['event'] == 'EXIT']
    assert s == l


# --------------------------------------------------------------------------------------------------- QA-BT-06
def test_bt_meta_training_rows_must_have_matured_labels_at_the_cutoff():
    from examples.train_meta_models import _monthly_rebalance_dates, _monthly_cutoffs
    from models.labeling import make_labels
    idx = pd.bdate_range('2020-07-27', periods=1400)
    closes = pd.Series(np.random.default_rng(1).normal(100, 1, len(idx)), index=idx)
    reb = _monthly_rebalance_dates(closes)
    dataset = make_labels({'T': closes}, reb, horizon_days=21)
    src = pathlib.Path(run.ROOT / 'examples' / 'train_meta_models.py').read_text()
    expr = re.search(r'train = (dataset\[dataset\["date"\] < cutoff - pd\.Timedelta\(days=\d+\)\])', src).group(1)  # the production line
    late = 0
    for cutoff in _monthly_cutoffs(reb, 12):
        train = eval(expr, {'dataset': dataset, 'cutoff': cutoff, 'pd': pd})
        ends = [closes.index[closes.index.get_loc(d) + 21] for d in train.date]
        late += bool(ends) and max(ends) > cutoff
    assert late == 0, f'{late} cutoffs use labels that mature after the cutoff'
