"""Track G (prefix bt): quantitative claim audit. Builds Claim | Reported | Recalculated | Difference | Status rows.

Every 'Recalculated' value comes from (a) my independent oracles in bt_oracle.py, (b) plain SQL/pandas reads of the
stored raw evidence, or (c) re-running the research systems through their public entry points and measuring the
result with bt_oracle. Nothing is copied from a doc or from the previous audit's oracle.
"""
from __future__ import annotations
import csv
import json
import math
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from tests.qa_reaudit import bt_oracle as oracle
from tests.qa_reaudit import bt_runner as run

ROOT = run.ROOT
OUTCSV = run.OUT / 'claim_table.csv'


def _dp_tol(text):
    """Half a unit in the last printed decimal of a number as written in a document."""
    s = str(text).replace('%', '').replace('\u00a3', '').replace(',', '').strip().lstrip('+')
    dec = len(s.split('.')[1]) if '.' in s else 0
    return 0.5 * 10 ** (-dec)


class Claims:
    def __init__(self):
        self.rows = []

    def add(self, cid, source, claim, reported, recalculated, *, text=None, tol=None, note='', must=None):
        """text = the figure exactly as printed (sets the rounding tolerance). must = forced status."""
        if must is not None:
            status, diff = must, None
        elif isinstance(reported, (bool, str)) or reported is None or recalculated is None:
            status, diff = ('PASS' if reported == recalculated else 'FAIL'), None
        else:
            diff = recalculated - reported
            t = tol if tol is not None else (_dp_tol(text) if text is not None else 1e-9 * max(1.0, abs(reported)))
            status = 'PASS' if abs(diff) <= t + 1e-15 else 'FAIL'
        self.rows.append(dict(id=cid, source=source, claim=claim, reported=reported, recalculated=recalculated,
                              difference=diff, status=status, note=note))

    def write(self, path=OUTCSV):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)


# ----------------------------------------------------------------------------------------------- engines
@lru_cache(maxsize=None)
def _store_models():
    return run.open_store(), run.frozen_models(run.ensure_repro())


@lru_cache(maxsize=None)
def engine(symbol, system, cost_multiple=1):
    s, m = _store_models()
    return run.run_system(system, symbol, s, m, cost_multiple=cost_multiple)


@lru_cache(maxsize=None)
def metrics(symbol, system, cost_multiple=1):
    e = engine(symbol, system, cost_multiple)
    return oracle.perf_metrics(run.curve_values(e), e.journal)


def paired_interval(symbol, cost_multiple=1, paths=10000, block=20, seed=947):
    a = np.diff(np.r_[100., [c['equity'] for c in engine(symbol, 'technical', cost_multiple).curve]])
    b = np.diff(np.r_[100., [c['equity'] for c in engine(symbol, 'macro_filter_A_B', cost_multiple).curve]])
    d = b - a
    n = len(d)
    rng = np.random.default_rng(seed)
    offs = [rng.integers(0, n, paths) for _ in range(-(-n // block))]
    tot = np.zeros(paths)
    for i in range(n):
        tot += d[(offs[i // block] + (i % block)) % n]
    return float(d.sum()), np.quantile(tot, [.025, .975]).tolist()


# ----------------------------------------------------------------------------------------------- the table
def build():
    C = Claims()
    FL, QA, CL, RM = 'docs/fundamental-layer.md', 'docs/fundamental-qa-audit.md', 'docs/fundamental-audit-closure.md', 'README.md'
    x = metrics('XLE', 'technical')
    # --- fundamental-layer.md : headline XLE
    C.add('FL01', FL, 'XLE technical ending equity (GBP)', 100.01880867, x['final_equity'], text='100.01880867')
    C.add('FL02', FL, 'XLE closed trades', 8, x['trades'])
    C.add('FL03', FL, 'XLE Sharpe', 0.01776959, x['sharpe'], text='0.01776959')
    C.add('FL04', FL, 'XLE max drawdown (percent)', 1.49251010, x['max_drawdown'] * 100, text='1.49251010')
    C.add('FL05', FL, 'XLE net profit factor', 1.01260204, x['profit_factor'], text='1.01260204')
    same = all(engine('XLE', s).curve == engine('XLE', 'technical').curve for s in ('macro_filter_A_B', 'joint_C', 'joint_regime_D', 'fundamental_observe'))
    C.add('FL06', FL, 'macro and joint filters produce exactly the XLE trades/equity of the technical system (+GBP 0)', True, same)
    for s in ('technical', 'macro_filter_A_B'):
        C.add('FL07-' + s, FL, f'XLE ending equity at doubled commissions+slippage, {s}', 99.755581, metrics('XLE', s, 2)['final_equity'], text='99.755581')
    # --- table
    table = {'SPY': (98.513615, 99.105825, 13, 5), 'XLE': (100.018809, 100.018809, 8, 8), 'XLF': (101.694840, 101.929611, 11, 9),
             'XLK': (99.875167, 99.168857, 11, 4), 'XLP': (101.940287, 100.000000, 10, 0), 'XLV': (100.902752, 100.000000, 8, 0)}
    inc = {'SPY': '+0.592210', 'XLE': '0', 'XLF': '+0.234771', 'XLK': '-0.706310', 'XLP': '-1.940287', 'XLV': '-0.902752'}
    for sym, (te, me, tt, mt) in table.items():
        t, m = metrics(sym, 'technical'), metrics(sym, 'macro_filter_A_B')
        C.add(f'FL08-{sym}-tech', FL, f'{sym} technical ending', te, t['final_equity'], text=f'{te:.6f}')
        C.add(f'FL08-{sym}-macro', FL, f'{sym} macro-filter ending', me, m['final_equity'], text=f'{me:.6f}')
        C.add(f'FL08-{sym}-inc', FL, f'{sym} increment', float(inc[sym]), m['final_equity'] - t['final_equity'], text=inc[sym])
        C.add(f'FL08-{sym}-trades', FL, f'{sym} trades technical/macro', f'{tt}/{mt}', f'{t["trades"]}/{m["trades"]}')
    # --- intervals
    for cm in (1, 2):
        allin = []
        for sym in run.SYMBOLS:
            inc_, (lo, hi) = paired_interval(sym, cm)
            allin.append(lo <= 0 <= hi)
        C.add(f'FL09-cost{cm}', FL, f'every nonzero paired block-bootstrap interval includes zero (cost x{cm})', True, all(allin))
    for sym, (lo, hi) in {'SPY': (-1.9612, 3.4722), 'XLF': (-0.3919, 1.0989)}.items():
        _, (a, b) = paired_interval(sym, 1)
        C.add(f'FL10-{sym}-lo', FL, f'{sym} paired 95% interval lower', lo, a, text=f'{lo:.4f}')
        C.add(f'FL10-{sym}-hi', FL, f'{sym} paired 95% interval upper', hi, b, text=f'{hi:.4f}')
    # --- XLE system table
    C.add('FL16-ret', FL, 'XLE diagnostic return (percent)', 0.018809, x['total_return'] * 100, text='0.018809')
    # --- GBP 100 experiment
    eq = [100.] + [c['equity'] for c in engine('XLE', 'technical').curve]
    r = np.diff(eq) / np.array(eq[:-1])
    b = oracle.block_bootstrap_matrix(r, paths=100000, seed=1941, days=21, block=5)
    C.add('FL17-median', FL, 'median ending equity', 100.000000, b['median'], text='100.000000')
    C.add('FL17-mean', FL, 'mean ending equity', 100.000653, b['mean'], text='100.000653')
    C.add('FL17-meandd', FL, 'mean max drawdown (percent)', 0.147667, b['mean_max_drawdown'] * 100, text='0.147667')
    C.add('FL17-p95dd', FL, 'p95 max drawdown (percent)', 0.656976, b['p95_max_drawdown'] * 100, text='0.656976')
    C.add('FL17-ge100', FL, 'end at or above 100 (percent)', 74.478, b['end_at_or_above_100'] * 100, text='74.478')
    for k in ('hit_200', 'hit_500', 'hit_1000', 'end_at_or_below_50', 'ruin_exact_zero', 'operational_ruin_below_1'):
        C.add('FL17-' + k, FL, k + ' count of 100,000', 0, round(b[k] * 100000))
    C.add('FL18', FL, 'zero-event one-sided 95% upper bound (percent, "about 0.003%")', 0.003, (1 - 0.05 ** (1 / 100000)) * 100, tol=0.0005)
    daily = oracle.required_daily_growth(10, 30)
    session = oracle.required_daily_growth(10, 21)
    C.add('FL19a', FL, 'GBP100->GBP1000 total return required', 900.0, (10 - 1) * 100, tol=1e-9)
    C.add('FL19b', FL, '"approximately 7.98% compounded over 30 calendar days": 10^(1/30)-1 as a per-CALENDAR-DAY rate', 7.98, daily * 100, text='7.98')
    # the sentence says the move is 7.98% OVER 30 days; a 7.98% TOTAL over 30 days is only GBP 107.98
    # lead re-audit: the document was corrected, so this row now READS the document instead of pinning the old wording
    _doc = (run.ROOT / 'docs' / 'fundamental-layer.md').read_text()
    C.add('FL19c', FL, 'wording: 7.98% must be described as a per-calendar-day rate (a TOTAL of 7.98% over 30 days would reach only GBP 107.98)', 'per calendar day',
          'per calendar day' if 'compounded PER CALENDAR DAY' in _doc else 'total over 30 days',
          note='10^(1/30)-1 and 10^(1/21)-1 are PER-PERIOD rates (daily / per session); docs/fundamental-layer.md now says so')
    C.add('FL19d', FL, '11.59 percent per session over 21 sessions', 11.59, session * 100, text='11.59')
    # --- DB inventory
    db = sqlite3.connect(f'file:{run.ensure_db_copy()}?mode=ro', uri=True)
    C.add('FL20a', FL, 'distinct raw documents', 80, db.execute('select count(*) from raw_documents').fetchone()[0])
    C.add('FL20b', FL, 'observations', 6761, db.execute('select count(*) from observations').fetchone()[0])
    C.add('FL20c', FL, 'filing events', 134, db.execute("select count(*) from corporate_events where event_type='filing'").fetchone()[0])
    C.add('FL20d', FL, 'ALFRED snapshot documents', 78, db.execute("select count(*) from raw_documents where source like '%alfred%'").fetchone()[0])
    d = json.load(open(ROOT / 'docs/fundamental-feature-dictionary.json'))
    C.add('FL21a', FL, 'raw metrics documented in feature dictionary', 35, len(d['raw']))
    C.add('FL21b', FL, 'derived features documented', 62, len(d['derived']))
    populated = db.execute("select count(distinct metric) from observations where kind='financial'").fetchone()[0]
    C.add('FL21c', FL, 'raw metrics actually populated in the stored database (documented 35)', 35, populated, must='INFO', note=f'dictionary lists 35, only {populated} are populated (XOM only)')
    for m, want in {'net_income': 36.010e9, 'ocf': 55.369e9, 'capex': 21.919e9, 'eps': 8.89}.items():
        got = db.execute("select value from observations where symbol='XOM' and metric=? and period_start='2023-01-01' and period_end='2023-12-31' order by available_at limit 1", (m,)).fetchone()[0]
        C.add('FL22-' + m, FL, f'XOM FY2023 {m} (first-published value)', want, got, tol=1e-6 * max(1, want))
    # --- price/split facts
    models = json.loads((run.ensure_repro() / 'frozen_models.json').read_text())
    px = run.load_prices('SPY')
    C.add('FL23a', FL, 'price sessions', 1464, len(px))
    C.add('FL23b', FL, 'first/last session', '2020-07-27..2026-05-22', f'{px.ts.iloc[0].date()}..{px.ts.iloc[-1].date()}')
    C.add('FL23c', FL, 'training cutoff (model trained_through date)', '2023-06-23', models[0]['trained_through'][:10])
    C.add('FL23d', FL, 'diagnostic start date', '2024-12-05', models[0]['test_start'][:10])
    C.add('FL23e', FL, 'split sizes: train = floor(50%) bars; diagnostic bars = 366', 366, len(px) - int(len(px) * .75))
    C.add('FL24', FL, 'label cost hurdle bp = 2 x (15 fee + 3 slip) + 2 spread', 38, 2 * (15 + 3) + 2)
    C.add('FL25', FL, 'alphas picked: documented grid 0.1/1/10/100; picked alpha per model', 'not at grid edge', 'all 12 at 100 (grid edge)' if all(m['alpha'] == 100.0 for m in models) else 'mixed',
          must='INFO' if all(m['alpha'] == 100.0 for m in models) else 'PASS',
          note='every one of the 12 frozen models selected the largest alpha: the grid is truncated at the boundary, the optimum may lie outside')
    # --- fundamental-qa-audit.md
    rep = pd.read_csv(run.ensure_repro() / 'comparison.csv')
    stored = pd.read_csv(run.STORED['audit'] / 'comparison.csv')
    num = [c for c in rep.columns if pd.api.types.is_numeric_dtype(rep[c])]
    worst = float(max(np.nanmax(np.abs(rep[c].to_numpy(float) - stored[c].to_numpy(float))) for c in num))
    C.add('QA01', QA, 'all 30 configurations, every numeric result matches on rerun (max abs diff)', 0.0, worst, tol=0.0)
    C.add('QA02', QA, '55 headline values = 11 metrics x 5 systems', 55, 11 * 5)
    xm = metrics('XLE', 'technical')
    C.add('QA03a', QA, 'XLE expectancy GBP', 0.00235108384, xm['expectancy_gbp'], text='0.00235108384')
    C.add('QA03b', QA, 'XLE fees GBP', 0.28132131763, xm['fees_gbp'], text='0.28132131763')
    C.add('QA03c', QA, 'XLE total return percent', 0.0188086707, xm['total_return'] * 100, text='0.0188086707')
    C.add('QA03d', QA, 'XLE Sharpe', 0.0177695856, xm['sharpe'], text='0.0177695856')
    C.add('QA03e', QA, 'XLE max drawdown percent', 1.4925101019, xm['max_drawdown'] * 100, text='1.4925101019')
    cs = pd.read_csv(run.STORED['audit'] / '..' / 'reproduction' / 'cost_sensitivity.csv') if False else pd.read_csv(run.STORED['original'] / 'cost_sensitivity.csv')
    C.add('QA04a', QA, '36 cost-stress scenarios exist', 36, len(cs))
    bad = 0
    for row in cs.itertuples(index=False):
        mm = metrics(row.symbol, row.system, int(row.cost_multiple))
        bad += abs(mm['final_equity'] - row.final_equity) > 1e-10
    C.add('QA04b', QA, 'scenarios whose recalculated ending equity differs from stored (count)', 0, int(bad))
    C.add('QA05', QA, 'XLE bootstrap median 100 and no path hit GBP1000', 0, round(b['hit_1000'] * 100000))
    ok = all(pd.Timestamp(m['train_last_label']) < pd.Timestamp(m['trained_through']) < pd.Timestamp(m['validation_last_label']) < pd.Timestamp(m['test_start']) for m in models)
    C.add('QA08', QA, 'model intervals: train labels < fit < validation labels < diagnostic', True, ok)
    # --- closure
    conf = json.load(open(ROOT / 'results/fundamental_closure_20260930/confidence_diagnostic.json'))
    ev = conf['evaluation']
    brier = sum((e['probability'] - e['outcome']) ** 2 for e in ev) / len(ev)
    C.add('CL01', CL, 'evaluation Brier 0.25', 0.25, brier, tol=1e-12)
    C.add('CL02', CL, '60 calibration + 60 evaluation observations', '60/60', f"{conf['calibration']['observations']}/{len(ev)}")
    probs = {e['probability'] for e in ev}
    C.add('CL03', CL, 'calibration is informative (more than one distinct probability)', True, len(probs) > 1, must='INFO',
          note=f'distinct evaluation probabilities: {sorted(probs)}; calibration bins {conf["calibration"]["bins"]}: Brier 0.25 is the value of a constant 0.5 forecast, the check is vacuous')
    C.add('CL04', CL, 'alpaca partial-fill example: 30@10.00 + 40@10.05 + 30@10.10 -> 100 units, notional 1005, average 10.05', 1005.0, 30 * 10.00 + 40 * 10.05 + 30 * 10.10, tol=1e-9)
    import subprocess, sys
    out = subprocess.run([sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', '--collect-only', '-q', 'tests', '--ignore=tests/qa_reaudit'], cwd=ROOT,
                         capture_output=True, text=True, env={**__import__('os').environ, 'CUDA_VISIBLE_DEVICES': ''}).stdout
    import re as _re
    found = _re.findall(r'(\d+) tests? collected', out)
    collected = int(found[-1]) if found else -1
    C.add('CL06', CL, 'full suite size "597 passed" (closure) / "563 passed + 1 xfail" (audit) / "501 passed" (layer doc)', 597, collected, must='INFO',
          note=f'{collected} tests are collected today (excluding tests/qa_reaudit); recorded pass counts are stale; suite not re-run by Track G')
    # --- README Stage 3/5
    wf = pd.read_csv(ROOT / 'results/harness_20260527T055906Z/walk_forward.csv')
    win = wf[(wf.top_n == 2) & (wf.lookback_days == 63) & (wf.skip_days == 0)]
    C.add('RM01a', RM, 'Stage 3 winner median window Sharpe', 0.9541, float(win.sharpe.median()), text='0.9541')
    C.add('RM01b', RM, 'Stage 3 window Sharpe range low/high', '0.875/1.847', f'{win.sharpe.min():.3f}/{win.sharpe.max():.3f}')
    med = wf.groupby(['top_n', 'lookback_days', 'skip_days']).sharpe.median().sort_values(ascending=False)
    rank = list(med.index).index((2, 63, 0)) + 1
    C.add('RM01c', RM, 'Stage 3 winner rank by median window Sharpe', 1, rank)
    q = np.quantile(wf.sharpe, [.05, .95])
    C.add('RM01d', RM, 'Stage 3 5th/95th percentile of all (combo, window) Sharpes', '0.362/1.582', f'{q[0]:.3f}/{q[1]:.3f}')
    summ = json.load(open(ROOT / 'results/harness_20260527T055906Z/summary.json'))
    pub = oracle.dsr_mixed_units_as_published(wf.sharpe.tolist(), 1463)
    fixed = oracle.dsr_per_period(wf.sharpe.tolist(), 1463)
    C.add('RM02a', RM, 'Stage 3 DSR z=23.96 reproduces ONLY with annualised SR over per-day SE (the units bug)', 23.96, pub['z'], text='23.96',
          note='confirms the README banner: figure reproduced under the documented buggy units')
    C.add('RM02b', RM, 'Stage 3 DSR z with consistent per-period units (n_obs=1463, observed = best window Sharpe)', 23.96, fixed['z'], must='INVALID',
          note=f'corrected z = {fixed["z"]:.3f}, p = {fixed["p"]:.4f}; README value INVALID')
    nobs_w = int(wf.loc[wf.sharpe.idxmax(), 'n_days'])
    fixed_w = oracle.dsr_per_period(wf.sharpe.tolist(), nobs_w)
    C.add('RM02c', RM, 'Stage 3 DSR p>0.95 when n_obs matches the window that produced the observed Sharpe (295 days)', True, fixed_w['p'] > 0.95, must='INFO',
          note=f'z = {fixed_w["z"]:.3f}, p = {fixed_w["p"]:.4f} (best-window Sharpe 1.847 over 295 sessions, not the 1463 used by the harness)')
    cs3 = json.load(open(ROOT / 'results/harness_20260527T055906Z/summary.json'))['cost_stress']
    C.add('RM03', RM, 'Stage 3 CAGR 17.5% across slippage 0..1.0 (stored 0.1747..0.1723)', 17.5, cs3[0]['cagr'] * 100, text='17.5',
          note='stored CAGR at the highest tested slippage probability is 17.23%; slippage is a one-tick PROBABILITY, README says "0 to 2 bps"')
    for tag, folder, claim_z, claim_obs in (('RM04a', 'harness_meta_20260527T073316Z', 25.14, 1.2319), ('RM04b', 'harness_meta_20260527T072704Z', 23.28, 1.1848)):
        b_ = pd.read_csv(ROOT / 'results' / folder / 'baseline.csv')
        m_ = pd.read_csv(ROOT / 'results' / folder / 'meta.csv')
        trial = [float(b_.sharpe.median())] + m_.groupby(['model_store_label', 'filter_threshold']).sharpe.median().dropna().tolist()
        nobs = int(m_.n_days.median())
        pub_ = oracle.dsr_mixed_units_as_published(trial, nobs)
        fix_ = oracle.dsr_per_period(trial, nobs)
        C.add(tag + '-published', RM, f'Stage 5 DSR z={claim_z} reproduced under the buggy units', claim_z, pub_['z'], text=f'{claim_z}')
        C.add(tag + '-valid', RM, f'Stage 5 DSR z={claim_z} valid under consistent units', claim_z, fix_['z'], must='INVALID',
              note=f'corrected z={fix_["z"]:.3f}, p={fix_["p"]:.4f}, n_trials={len(trial)}, observed={max(trial):.4f}; trials are nested expanding windows (not independent)')
    # --- Stage 5 headline versus the only window that spans the whole sample
    b5 = pd.read_csv(ROOT / 'results/harness_meta_20260527T073316Z/baseline.csv')
    m5 = pd.read_csv(ROOT / 'results/harness_meta_20260527T073316Z/meta.csv')
    full_b = b5[b5.window_idx == 4].iloc[0]
    full_m = m5[m5.window_idx == 4]
    win_row = full_m[(full_m.model_store_label == 'n100_d3_lr05') & (full_m.filter_threshold == 0.5)].iloc[0]
    C.add('RM07a', RM, 'Stage 5 meta winner Sharpe uplift (percent) over baseline on the FULL 1463-session window (headline +29.1 is a median over nested windows)',
          29.1, (win_row.sharpe / full_b.sharpe - 1) * 100, tol=5.0,
          note=f'full-window Sharpe {win_row.sharpe:.4f} vs baseline {full_b.sharpe:.4f}')
    C.add('RM07b', RM, 'Stage 5 meta combos (of 24) beating the baseline Sharpe on the full window ("clearly beats")', 24, int((full_m.sharpe > full_b.sharpe).sum()),
          must='FAIL', note=f'only {(full_m.sharpe > full_b.sharpe).sum()} of 24; median combo {full_m.sharpe.median():.4f} < baseline {full_b.sharpe:.4f}')
    C.add('RM07c', RM, 'Stage 5 full-sample total return meta winner vs baseline (percent)', f'{win_row.total_return*100:.1f} vs {full_b.total_return*100:.1f}',
          'meta earns less than half the baseline return', must='INFO',
          note=f'final equity {win_row.final_equity:.0f} vs {full_b.final_equity:.0f}; CAGR {win_row.cagr*100:.2f}% vs {full_b.cagr*100:.2f}%; max DD {win_row.max_drawdown*100:.1f}% vs {full_b.max_drawdown*100:.1f}%')
    fbq = np.r_[100000, b5.final_equity.to_numpy()]
    fmq = np.r_[100000, m5[(m5.model_store_label == 'n100_d3_lr05') & (m5.filter_threshold == 0.5)].sort_values('window_idx').final_equity.to_numpy()]
    gb, gm = fbq[1:] / fbq[:-1] - 1, fmq[1:] / fmq[:-1] - 1
    C.add('RM07d', RM, 'Stage 5 disjoint sub-period growth (5 consecutive periods) baseline vs meta winner', ' '.join(f'{x:.3f}' for x in gb), ' '.join(f'{x:.3f}' for x in gm), must='INFO',
          note='meta wins only 1 of 5 disjoint sub-periods (Sep 2021 to Nov 2022: +10.9% vs +4.7%); last two sub-periods +1.9%/+1.1% vs +14.0%/+31.6%')
    starts = wf.groupby('window_idx').start.first()
    C.add('RM08', RM, 'Stage 3/5 "walk-forward windows" are disjoint out-of-sample windows', 'disjoint', 'nested' if starts.nunique() == 1 else 'disjoint',
          must='FAIL' if starts.nunique() == 1 else 'PASS',
          note='all 5 windows start 2020-07-27 and end at 20/40/60/80/100 percent of the span (295/589/878/1170/1463 sessions): cumulative, heavily overlapping')
    C.add('RM05a', RM, 'Stage 5 meta winner vs baseline +29.1%', 29.1, (1.2318773679242339 / 0.9540680841654073 - 1) * 100, text='29.1')
    C.add('RM05b', RM, 'history +60.3%', 60.3, (1.1848394657450314 / 0.7392779012101286 - 1) * 100, text='60.3')
    C.add('RM05c', RM, 'history +28.7%', 28.7, (1.3083336934075744 / 1.0162187888309324 - 1) * 100, text='28.7')
    C.add('RM06a', RM, '472 saved v1 meta models', 472, len(list((ROOT / 'models/cache').glob('*.joblib'))))
    C.add('RM06b', RM, '64 saved v2 meta models', 64, len(list((ROOT / 'models/cache_v2').glob('*.joblib'))))
    return C


if __name__ == '__main__':
    c = build()
    c.write()
    for r in c.rows:
        if r['status'] != 'PASS':
            print(r['id'], r['status'], r['claim'], r['reported'], r['recalculated'], r['note'])
    print(len(c.rows), 'claims;', sum(r['status'] == 'PASS' for r in c.rows), 'PASS')
