"""Track C phase 31-33: cross-sectional normalisation (percentiles, winsorised z-scores, ties, outliers, grouping).

Oracle = exact Decimal mid-rank percentile + R-type-7 quantile winsorisation at 1%/99% + sample std.
Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_calc_normalization.py
"""
import math
import random
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from fundamentals.features import normalize_cross_section

from tests.qa_reaudit import calc_oracle as O


def near(a, b, rel=1e-10, abs_tol=1e-12):
    if b is None:
        return a is None or (isinstance(a, float) and math.isnan(a))
    return O.rel_close(float(a), b, rel, abs_tol)


def test_ties_get_average_rank_and_percentile_formula():
    f = pd.DataFrame({"sector": ["S"] * 5, "x": [1.0, 2.0, 2.0, 2.0, 5.0]})
    out = normalize_cross_section(f, "x")
    # average ranks 1,3,3,3,5 -> (rank - 0.5)/5
    assert list(out.x_percentile) == pytest.approx([0.1, 0.5, 0.5, 0.5, 0.9], abs=1e-15)
    assert out.x_percentile.mean() == pytest.approx(0.5, abs=1e-15)


def test_all_equal_values_have_no_zscore_not_zero_or_inf():
    f = pd.DataFrame({"sector": ["S"] * 6, "x": [3.0] * 6})
    out = normalize_cross_section(f, "x")
    assert out.x_zscore.isna().all()                          # undefined, not 0/inf
    assert list(out.x_percentile) == pytest.approx([0.5] * 6)  # all tied -> middle rank


def test_minimum_group_size_boundary_and_missing_values():
    base = {"sector": ["A"] * 4 + ["B"] * 5, "x": [1.0, 2, 3, 4, 10, 20, 30, 40, 50]}
    out = normalize_cross_section(pd.DataFrame(base), "x", minimum=5)
    assert out.loc[:3, "x_percentile"].isna().all() and out.loc[:3, "x_zscore"].isna().all()   # 4 < 5: never ranked
    assert out.loc[4:, "x_percentile"].notna().all()                                               # exactly 5: ranked
    out = normalize_cross_section(pd.DataFrame(base), "x", minimum=4)
    assert out.loc[:3, "x_percentile"].notna().all()
    # NaN metric / NaN group never count towards the minimum and never get a rank
    f = pd.DataFrame({"sector": ["A"] * 6, "x": [1.0, 2, 3, np.nan, 5, 6]})
    out = normalize_cross_section(f, "x", minimum=6)
    assert out.x_percentile.isna().all()
    out = normalize_cross_section(f, "x", minimum=5)
    assert out.x_percentile.isna().sum() == 1 and np.isnan(out.x_percentile[3])
    g = pd.DataFrame({"sector": ["A", "A", "A", "A", "A", None], "x": [1.0, 2, 3, 4, 5, 6]})
    out = normalize_cross_section(g, "x")
    assert np.isnan(out.x_percentile[5]) and out.x_percentile[:5].notna().all()


def test_input_frame_not_mutated_and_index_preserved():
    f = pd.DataFrame({"sector": ["A"] * 6, "x": [6.0, 5, 4, 3, 2, 1]}, index=list("fedcba"))
    before = f.copy(deep=True)
    out = normalize_cross_section(f, "x")
    pd.testing.assert_frame_equal(f, before)
    assert list(out.index) == list("fedcba")
    assert out.x_percentile["f"] > out.x_percentile["a"]          # higher metric -> higher percentile (no sign flip)


def test_group_isolation_banks_vs_software_by_industry_column():
    banks = [0.9, 1.0, 1.1, 1.2, 1.3]          # e.g. net interest margin type values
    soft = [10.0, 20.0, 30.0, 40.0, 50.0]
    f = pd.DataFrame({"sector": ["Financials"] * 5 + ["Technology"] * 5, "industry": ["Banks"] * 5 + ["Software"] * 5, "x": banks + soft})
    by_ind = normalize_cross_section(f, "x", group="industry")
    # a bank's rank/z is computed ONLY against banks
    assert by_ind.x_percentile[:5].tolist() == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.9])
    assert by_ind.x_percentile[5:].tolist() == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.9])
    assert by_ind.x_zscore[0] == pytest.approx(by_ind.x_zscore[5], rel=1e-12)        # same relative position
    # grouping by a coarser label that contains both would (by design) mix them: the CALLER owns the taxonomy
    mixed = f.assign(sector="All")
    coarse = normalize_cross_section(mixed, "x", group="sector", minimum=5)
    assert coarse.x_percentile[0] == pytest.approx(0.05)   # a bank at the 5th percentile of the mixed 10-name universe


def test_duplicate_index_does_not_misassign():
    f = pd.DataFrame({"sector": ["A"] * 5 + ["B"] * 5, "x": [1.0, 2, 3, 4, 5, 10, 20, 30, 40, 50]}, index=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    out = normalize_cross_section(f, "x")
    assert out.x_percentile.tolist() == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.9] * 2)


def test_infinite_metric_gives_no_zscore_not_nan_propagation_to_percentile():
    f = pd.DataFrame({"sector": ["A"] * 6, "x": [1.0, 2, 3, 4, 5, math.inf]})
    out = normalize_cross_section(f, "x")
    assert out.x_percentile.notna().all() and out.x_zscore.isna().all()    # fail-safe: no finite z from an infinite input


# -------------------------------------------------------------------------- randomized differential
def make_frame(rng):
    groups = rng.randint(1, 4)
    rows = []
    for g in range(groups):
        n = rng.choice([3, 4, 5, 6, 8, 12, 25, 60, 101])
        kind = rng.choice(["normal", "ties", "heavy", "outlier"])
        for _ in range(n):
            if kind == "normal":
                v = round(rng.gauss(10, 4), 3)
            elif kind == "ties":
                v = float(rng.choice([1, 2, 2, 3, 3, 3, 7]))
            elif kind == "heavy":
                v = round(math.exp(rng.gauss(0, 2)), 4)
            else:
                v = round(rng.gauss(10, 4), 3)
            rows.append(("G%d" % g, v))
        if kind == "outlier":
            rows.append(("G%d" % g, rng.choice([1e6, -1e6, 5e4])))
    # a few missing metrics / missing groups
    out = []
    for g, v in rows:
        r = rng.random()
        out.append((None if r < 0.02 else g, None if 0.02 <= r < 0.05 else v))
    return pd.DataFrame(out, columns=["sector", "x"]).astype({"x": float})


@pytest.mark.parametrize("seed", range(150), ids=lambda s: f"seed{s}")
def test_random_cross_sections_match_oracle(seed):
    rng = random.Random(seed)
    f = make_frame(rng)
    minimum = rng.choice([3, 4, 5, 5, 5])
    out = normalize_cross_section(f, "x", minimum=minimum)
    for g, rows in f.dropna(subset=["sector", "x"]).groupby("sector"):
        if len(rows) < minimum:
            assert out.loc[rows.index, ["x_percentile", "x_zscore"]].isna().all().all(), (seed, g)
            continue
        vals = [Decimal(float(v)) for v in rows.x]
        pct = O.mid_percentiles(vals)
        z = O.winsor_z(vals)
        for i, idx in enumerate(rows.index):
            assert near(out.loc[idx, "x_percentile"], pct[i], 1e-12, 1e-14), (seed, g, idx, out.loc[idx, "x_percentile"], pct[i])
            if z is None:
                assert math.isnan(out.loc[idx, "x_zscore"]), (seed, g, idx)
            else:
                assert near(out.loc[idx, "x_zscore"], z[i], 1e-8, 1e-9), (seed, g, idx, out.loc[idx, "x_zscore"], z[i])
    # untouched rows stay NaN
    skip = f[f.sector.isna() | f.x.isna()].index
    assert out.loc[skip, ["x_percentile", "x_zscore"]].isna().all().all()


@pytest.mark.parametrize("seed", range(40), ids=lambda s: f"seed{s}")
def test_percentile_properties(seed):
    rng = random.Random(1000 + seed)
    n = rng.randint(5, 60)
    xs = [round(rng.gauss(0, 3), 2) for _ in range(n)]
    f = pd.DataFrame({"sector": ["A"] * n, "x": xs})
    p = normalize_cross_section(f, "x").x_percentile
    assert p.mean() == pytest.approx(0.5, abs=1e-12)
    assert (p > 0).all() and (p < 1).all()
    order = np.argsort(xs, kind="stable")
    assert all(p.iloc[order[i]] <= p.iloc[order[i + 1]] + 1e-15 for i in range(n - 1))   # monotone in the metric
    # affine invariance: a positive affine transform of the metric changes neither ranks nor z-scores
    g = pd.DataFrame({"sector": ["A"] * n, "x": [3.0 * v + 7 for v in xs]})
    a, b = normalize_cross_section(f, "x"), normalize_cross_section(g, "x")
    assert np.allclose(a.x_percentile, b.x_percentile, atol=1e-12)
    assert np.allclose(a.x_zscore.fillna(0), b.x_zscore.fillna(0), atol=1e-9)


def test_outlier_changes_percentiles_by_at_most_rank_shift():
    """Rank percentiles are robust: one extreme name only rescales (r-0.5)/n, shift <= 1/(n+1)."""
    base = list(range(1, 21))
    f1 = pd.DataFrame({"sector": ["A"] * 20, "x": [float(v) for v in base]})
    f2 = pd.DataFrame({"sector": ["A"] * 21, "x": [float(v) for v in base] + [1e9]})
    p1 = normalize_cross_section(f1, "x").x_percentile.values
    p2 = normalize_cross_section(f2, "x").x_percentile.values[:20]
    assert np.max(np.abs(p1 - p2)) <= 1 / 21 + 1e-12


def test_QA_CALC_01_one_extreme_company_must_not_distort_the_others_zscores():
    base = [float(v) for v in range(1, 21)]
    f = pd.DataFrame({"sector": ["A"] * 21, "x": base + [1e9]})
    z = normalize_cross_section(f, "x").x_zscore.values[:20]
    # With robust winsorisation the 20 ordinary names keep a meaningful dispersion (>= 0.5 SD); today it is ~3e-8
    assert np.std(z) >= 0.5, f"dispersion of ordinary names' z-scores is {np.std(z):.3e}"
