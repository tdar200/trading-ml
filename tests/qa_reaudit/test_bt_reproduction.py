"""Track G: reproduce the ETF macro/fundamental ablation from scratch and check metrics with an independent oracle.

Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_bt_reproduction.py
The module fixture runs the REAL CLI (examples/research_fundamentals.py) on a COPY of the fundamentals DB into
results/qa_reaudit_20261001/bt/repro unless that directory already exists.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.qa_reaudit import bt_oracle as oracle
from tests.qa_reaudit import bt_runner as run

NUMERIC_FILES = ('comparison.csv', 'incremental.csv', 'information_coefficients.csv')


@pytest.fixture(scope='module')
def repro():
    return run.ensure_repro()


@pytest.fixture(scope='module')
def store():
    s = run.open_store()
    yield s
    s.close()


@pytest.fixture(scope='module')
def models(repro):
    return run.frozen_models(repro)


def _numeric_diff(a: pd.DataFrame, b: pd.DataFrame):
    assert list(a.columns) == list(b.columns), 'column sets differ'
    assert len(a) == len(b), 'row counts differ'
    worst = 0.0
    cells = 0
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            x, y = a[c].to_numpy(float), b[c].to_numpy(float)
            assert (np.isnan(x) == np.isnan(y)).all(), f'NaN pattern differs in {c}'
            keep = ~np.isnan(x)
            if keep.any():
                worst = max(worst, float(np.abs(x[keep] - y[keep]).max()))
                cells += int(keep.sum())
        else:
            assert (a[c].astype(str) == b[c].astype(str)).all(), f'non-numeric column {c} differs'
    return worst, cells


@pytest.mark.parametrize('stored', sorted(run.STORED))
@pytest.mark.parametrize('name', NUMERIC_FILES)
def test_bt_every_numeric_cell_matches_stored_output(repro, stored, name):
    """Phase 92-93: every numeric column of the three result tables reproduces BIT-FOR-BIT (max abs diff 0)."""
    a = pd.read_csv(Path(repro) / name)
    b = pd.read_csv(run.STORED[stored] / name)
    worst, cells = _numeric_diff(a, b)
    assert cells > 100
    assert worst == 0.0, f'{stored}/{name}: max abs diff {worst} over {cells} cells'


@pytest.mark.parametrize('stored', sorted(run.STORED))
@pytest.mark.parametrize('system', run.SYSTEMS)
def test_bt_xle_equity_curve_and_trades_identical(repro, stored, system):
    a = pd.read_csv(Path(repro) / f'{system}_equity.csv')
    b = pd.read_csv(run.STORED[stored] / f'{system}_equity.csv')
    worst, _ = _numeric_diff(a, b)
    assert worst == 0.0
    ja = json.loads((Path(repro) / f'{system}_journal.json').read_text())
    jb = json.loads((run.STORED[stored] / f'{system}_journal.json').read_text())
    keep = lambda j: [e for e in j if e['event'] in ('ENTRY', 'EXIT', 'RISK')]
    ka, kb = keep(ja), keep(jb)
    assert len(ka) == len(kb), 'trade/risk event count differs'
    for ea, eb in zip(ka, kb):
        # the stored audit journals were written with pandas to_json (10 decimals); the repro keeps full precision (QA-BT-05 fix)
        assert ea.keys() == eb.keys()
        for key in ea:
            if isinstance(ea[key], float) and isinstance(eb[key], float):
                assert abs(ea[key] - eb[key]) <= 1e-9, (key, ea[key], eb[key])
            else:
                assert ea[key] == eb[key], (key, ea[key], eb[key])
    assert len(ja) == len(jb)


@pytest.mark.parametrize('stored', sorted(run.STORED))
def test_bt_frozen_models_and_macro_states_identical(repro, stored):
    for name in ('frozen_models.json', 'macro_states.json'):
        a = json.loads((Path(repro) / name).read_text())
        b = json.loads((run.STORED[stored] / name).read_text())
        assert a == b, name


def _bars(symbol):
    f = run.load_prices(symbol)
    return {str(t): float(c) for t, c in zip(f.ts, f.close)}


@pytest.fixture(scope='module')
def engines(store, models):
    """All 30 (symbol, system) diagnostic-window engines, in memory (unrounded journals)."""
    out = {}
    for sym in run.SYMBOLS:
        for sysn in run.SYSTEMS:
            out[(sym, sysn)] = run.run_system(sysn, sym, store, models)
    return out


@pytest.fixture(scope='module')
def reported(repro):
    df = pd.read_csv(Path(repro) / 'comparison.csv')
    return {(r.symbol, r.system): r for r in df.itertuples(index=False)}


METRIC_KEYS = ['final_equity', 'total_return', 'cagr', 'monthly_geometric', 'daily_mean', 'sharpe', 'sortino', 'calmar',
               'profit_factor', 'expectancy_gbp', 'win_rate', 'average_win', 'average_loss', 'reward_risk_realized', 'trades',
               'turnover', 'fees_gbp', 'max_drawdown', 'average_drawdown', 'longest_underwater_observations', 'observations']


@pytest.mark.parametrize('symbol', run.SYMBOLS)
@pytest.mark.parametrize('system', run.SYSTEMS)
def test_bt_independent_metric_oracle_matches_reported_all_30(engines, reported, symbol, system):
    """Phase 92/99: every reported metric of all 30 configurations from an independent Decimal oracle (<=1e-12 rel)."""
    e = engines[(symbol, system)]
    mine = oracle.perf_metrics(run.curve_values(e), e.journal)
    rep = reported[(symbol, system)]._asdict()
    bad = []
    for k in METRIC_KEYS:
        a, b = mine[k], rep[k]
        if a is None or (isinstance(b, float) and math.isnan(b)):
            if not (a is None and (b is None or (isinstance(b, float) and math.isnan(b)))):
                bad.append((k, a, b))
            continue
        if abs(a - b) > 1e-12 * max(1.0, abs(b)):
            bad.append((k, a, b))
    assert not bad, f'{symbol}/{system}: {bad}'


@pytest.mark.parametrize('symbol', run.SYMBOLS)
@pytest.mark.parametrize('system', run.SYSTEMS)
def test_bt_equity_curve_reconstructed_from_journal_and_closes(engines, reported, symbol, system):
    """Rebuild equity bar-by-bar from ENTRY/EXIT cash flows + closes only; must equal the engine curve (<1e-9)
    and the reported mean exposure (position value / equity at the close)."""
    e = engines[(symbol, system)]
    closes = _bars(symbol)
    bars = [(c['ts'], closes[c['ts']]) for c in e.curve]
    expo = []
    rebuilt = oracle.reconstruct_equity(e.journal, bars, None, None, exposure=expo)
    worst = max(abs(float(r) - c['equity']) for r, c in zip(rebuilt, e.curve))
    assert worst < 1e-9, worst
    assert sum(expo) / len(expo) == pytest.approx(reported[(symbol, system)].exposure, abs=1e-12)


def _expected_quantity(equity, cash, price, stop, fee, spread, slip, streak, weekly):
    # independent re-derivation of the documented sizing rule (risk 0.5% of equity, 25% exposure cap, cash cap)
    exit_price = stop * (1 - (spread / 2 + slip) / 10000)
    loss = price * (1 + fee) - exit_price * (1 - fee)
    scale = 0.5 if streak >= 3 or weekly <= -0.02 else 1.0
    cap = min(0.25, 0.25, 1.0)
    q = min(equity * 0.005 * scale / loss, cash / (price * (1 + fee)), equity * cap / price)
    return math.floor(q / 1e-6 + 1e-9) * 1e-6


@pytest.mark.parametrize('symbol', run.SYMBOLS)
def test_bt_entry_fills_costs_and_size_follow_the_documented_rules(engines, symbol):
    """Entry fill = open*(1+4bp); fee = 15bp of notional; size = min(0.5% risk, cash, 25% exposure) floored to 1e-6."""
    e = engines[(symbol, 'technical')]
    frame = run.load_prices(symbol)
    opens = {str(t): float(o) for t, o in zip(frame.ts, frame.open)}
    cash = 100.0
    streak = 0
    entries = 0
    for ev in e.journal:
        if ev['event'] == 'ENTRY':
            entries += 1
            assert ev['price'] == pytest.approx(opens[ev['timestamp']] * (1 + 4 / 10000), rel=1e-12)
            assert ev['fee'] == pytest.approx(ev['price'] * ev['quantity'] * 0.0015, rel=1e-12)
            stop = ev['stop']
            exp = _expected_quantity(cash, cash, ev['price'], stop, 0.0015, 2.0, 3.0, streak, 0.0)
            # (equity==cash when flat; weekly stress term is zero in this sample, asserted via equality)
            assert ev['quantity'] == pytest.approx(exp, abs=2e-6), (ev['timestamp'], ev['quantity'], exp)
            cash -= ev['price'] * ev['quantity'] + ev['fee']
        elif ev['event'] == 'EXIT':
            cash += ev['price'] * ev['quantity'] - ev['fee']
            streak = streak + 1 if ev['pnl'] <= 0 else 0
    assert entries >= 5


def test_bt_stored_journal_precision_is_ten_decimals(repro):
    """Surprise: examples/research_fundamentals.py writes journals via pandas.to_json (10-digit rounding).

    Metrics rebuilt from the saved XLE journal therefore differ from the reported ones by ~1e-11 (not a P&L error).
    """
    j = json.loads((Path(repro) / 'technical_journal.json').read_text())
    eq = [float(r['equity']) for r in oracle.load_curve(Path(repro) / 'technical_equity.csv')]
    m = oracle.perf_metrics(eq, j)
    rep = pd.read_csv(Path(repro) / 'comparison.csv')
    row = rep[(rep.symbol == 'XLE') & (rep.system == 'technical')].iloc[0]
    for k in ('final_equity', 'sharpe', 'sortino', 'max_drawdown'):
        assert abs(m[k] - row[k]) < 1e-13 * max(1, abs(row[k])) or abs(m[k] - row[k]) < 1e-12
    for k in ('profit_factor', 'expectancy_gbp', 'fees_gbp', 'average_win', 'average_loss'):
        # lead re-audit (QA-BT-05 fixed): the journal keeps full float precision, so metrics rebuilt from the file match the
        # reported ones to float round-off (they differed by ~1e-11 when pandas to_json rounded to 10 decimals)
        assert abs(m[k] - row[k]) < 1e-12 * max(1, abs(row[k])), k
    # and the stored (pre-fix) audit journals are the 10-decimal ones
    import re
    old = (run.STORED['original'] / 'technical_journal.json').read_text()
    assert max(len(x.split('.')[1]) for x in re.findall(r'-?\d+\.\d+', old) if 'e' not in x) <= 10


def test_bt_bootstrap_block_30_day_matches_matrix_oracle(repro):
    """Phase 94/109: the reported 100,000-path 21-session study from my own matrix implementation (same seed)."""
    res = json.loads((Path(repro) / 'research.json').read_text())
    eq = [100.0] + [c['equity'] for c in pd.read_csv(Path(repro) / 'technical_equity.csv').to_dict('records')]
    r = np.diff(eq) / np.array(eq[:-1])
    mine = oracle.block_bootstrap_matrix(r, paths=100000, seed=1941, days=21, block=5)
    rep = res['selected_account_experiment']['technical']['bootstrap_30_days']
    for k, v in mine.items():
        assert v == pytest.approx(rep[k], abs=1e-9), (k, v, rep[k])
    assert rep['zero_event_one_sided_95_upper'] == pytest.approx(1 - 0.05 ** (1 / 100000), rel=1e-12)


@pytest.mark.parametrize('kwargs', [
    {'paths': 0}, {'paths': True}, {'days': 0}, {'block': 0}, {'paths': 2.5}, {'days': -1},
])
def test_bt_account_bootstrap_rejects_invalid_dimensions(kwargs):
    from fundamentals.research import account_bootstrap
    r = np.r_[np.zeros(40), 0.01, -0.01]
    with pytest.raises(ValueError):
        account_bootstrap(r, **kwargs)


@pytest.mark.parametrize('returns', [np.r_[np.zeros(40), np.nan], np.r_[np.zeros(40), np.inf], np.r_[np.zeros(40), -1.5], np.zeros(10), np.zeros((6, 6))])
def test_bt_account_bootstrap_rejects_invalid_returns(returns):
    from fundamentals.research import account_bootstrap
    with pytest.raises(ValueError):
        account_bootstrap(returns, paths=100)


def test_bt_account_bootstrap_total_loss_is_counted_as_ruin_and_matches_matrix_oracle():
    from fundamentals.research import account_bootstrap
    r = np.r_[np.zeros(40), -1.0]
    got = account_bootstrap(r, paths=5000, seed=7)
    mine = oracle.block_bootstrap_matrix(r, paths=5000, seed=7)
    assert got['ruin_exact_zero'] == pytest.approx(mine['ruin_exact_zero'], abs=1e-12) and got['ruin_exact_zero'] > 0.3
    assert got['median'] == pytest.approx(mine['median'], abs=1e-9) and got['mean'] == pytest.approx(mine['mean'], abs=1e-9)


def test_bt_input_manifest_and_code_provenance(repro):
    """Inputs (price frames, observation ids, raw documents) are byte-identical to every stored run. Code hashes are
    compared and written to bt/provenance.json for information (they legitimately change when defects are fixed)."""
    new = json.loads((Path(repro) / 'research.json').read_text())
    report = {}
    for label, d in run.STORED.items():
        old = json.loads((d / 'research.json').read_text())
        assert new['input_manifest']['price_frames_sha256'] == old['input_manifest']['price_frames_sha256'], label
        assert new['input_manifest']['observation_ids_sha256'] == old['input_manifest']['observation_ids_sha256'], label
        assert new['input_manifest']['raw_documents'] == old['input_manifest']['raw_documents'], label
        report[label] = sorted(k for k in new['code_hashes'] if new['code_hashes'][k] != old['code_hashes'].get(k))
    (run.OUT / 'provenance.json').write_text(json.dumps({'code_files_changed_since_stored_run': report}, indent=2))
    # informational only: no assertion on code hashes
