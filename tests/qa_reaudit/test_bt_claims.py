"""Track G: quantitative claim audit (docs/fundamental-layer.md, fundamental-qa-audit.md, closure, README Stage 3/5).

Writes results/qa_reaudit_20261001/bt/claim_table.csv.
Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_bt_claims.py
"""
import pytest

from tests.qa_reaudit import bt_claims as claims


@pytest.fixture(scope='module')
def table():
    c = claims.build()
    c.write()
    return {r['id']: r for r in c.rows}


def test_bt_claim_table_is_large_and_written(table):
    assert len(table) >= 100
    assert claims.OUTCSV.exists()


def test_bt_every_headline_number_in_the_docs_recalculates(table):
    """All XLE / six-ETF / cost-stress / bootstrap / inventory numbers in the three fundamental docs recalculate."""
    failing = {k: (r['reported'], r['recalculated']) for k, r in table.items()
               if r['status'] == 'FAIL' and k not in ('FL19c', 'RM07a', 'RM07b', 'RM08')}
    assert not failing, failing


def test_bt_readme_dsr_figures_reproduce_only_under_the_documented_units_bug(table):
    """README banner: DSR values are INVALID. 23.96 / 25.14 / 23.28 are reproduced exactly with annualised SR over a
    per-day standard error, and the units-consistent z is 2.5 / 2.1 / 1.9 (not 24)."""
    for k in ('RM02a', 'RM04a-published', 'RM04b-published'):
        assert table[k]['status'] == 'PASS', table[k]
    for k in ('RM02b', 'RM04a-valid', 'RM04b-valid'):
        assert table[k]['status'] == 'INVALID'
        assert table[k]['recalculated'] < 3.0
    # Stage 3 'survives DSR' also depends on pairing the 1.847 SR of the 295-session window with n_obs=1463
    assert table['RM02c']['recalculated'] is False


def test_bt_confidence_diagnostic_is_vacuous(table):
    """Closure doc 'Brier 0.25' is real arithmetic but equals the Brier of a constant 0.5 forecast."""
    assert table['CL01']['status'] == 'PASS'
    assert table['CL03']['recalculated'] is False


def test_bt_alpha_grid_selected_at_boundary(table):
    assert table['FL25']['recalculated'].startswith('all 12 at 100')


def test_bt_doc_wording_of_the_ten_x_requirement(table):
    assert table['FL19c']['status'] == 'PASS'


@pytest.mark.xfail(strict=True, reason='QA-BT-07 HIGH: README Stage 5 headline "meta clearly beats baseline OOS (+29.1% median window Sharpe)" is a '
                                       'median over NESTED expanding windows; on the full 1463-session window the winner is +0.5% Sharpe '
                                       '(0.9916 vs 0.9867), 23 of 24 combos are below baseline, total return 68.8% vs 153.7%')
def test_bt_stage5_headline_survives_the_full_sample_view(table):
    assert table['RM07a']['status'] == 'PASS' and table['RM07b']['status'] == 'PASS'


def test_bt_stage5_full_sample_evidence_is_recorded(table):
    assert table['RM07a']['recalculated'] == pytest.approx(0.5, abs=0.1)
    assert table['RM07b']['recalculated'] == 1
    assert table['RM07c']['status'] == 'INFO'


def test_bt_harness_windows_are_nested_not_disjoint(table):
    """README calls the Stage 3/5 harness walk-forward; its five windows all start on 2020-07-27 (nested, expanding)."""
    assert table['RM08']['recalculated'] == 'nested'
    assert table['RM08']['status'] == 'FAIL'
    assert table['CL06']['status'] == 'INFO' and table['CL06']['recalculated'] > 597


def test_bt_current_deflated_sharpe_function_equals_the_units_consistent_oracle():
    """README banner says the scalar calculation is now corrected: analysis.deflated_sharpe equals my per-period oracle
    on the Stage 3 and three Stage 5 trial sets (z 2.474 / 2.097 / 1.911 / 1.693) to 1e-12."""
    import pandas as pd
    from analysis.deflated_sharpe import deflated_sharpe
    from tests.qa_reaudit import bt_oracle as oracle
    wf = pd.read_csv(claims.ROOT / 'results/harness_20260527T055906Z/walk_forward.csv')
    assert deflated_sharpe(wf.sharpe.tolist(), 1463)['deflated_sharpe'] == pytest.approx(oracle.dsr_per_period(wf.sharpe.tolist(), 1463)['z'], abs=1e-12)
    for d in ('harness_meta_20260527T073316Z', 'harness_meta_20260527T072704Z', 'harness_meta_v2_20260527T105904Z'):
        b = pd.read_csv(claims.ROOT / 'results' / d / 'baseline.csv')
        m = pd.read_csv(claims.ROOT / 'results' / d / 'meta.csv')
        trial = [float(b.sharpe.median())] + m.groupby(['model_store_label', 'filter_threshold']).sharpe.median().dropna().tolist()
        n = int(m.n_days.median())
        assert deflated_sharpe(trial, n)['deflated_sharpe'] == pytest.approx(oracle.dsr_per_period(trial, n)['z'], abs=1e-12)
