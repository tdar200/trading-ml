"""Track C phases 40-45, 136: the frozen-ridge score, fail-closed behaviour, thresholds, validation.

The production "score" is a predicted H-session return from a standardised ridge model, not a 0-100 composite.
Oracle: exact rational ridge (Fractions) + Decimal linear score.
Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_calc_scoring.py
"""
import math
import random
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from fundamentals.features import FundamentalFeatureEngine
from fundamentals.models import CombinedSignalEngine, FundamentalScoreEngine, ModelArtifact, fit_model

from tests.qa_reaudit import calc_oracle as O
from tests.qa_reaudit.calc_fixtures import annual_facts, classification, new_store
from tests.qa_reaudit.calc_golden import GOLDEN

TRAINED = "2024-01-01T00:00:00Z"
AS_OF = "2024-06-01T00:00:00Z"


def art(features=("a", "b"), means=(10.0, 20.0), scales=(2.0, 5.0), coefs=(0.03, -0.01), intercept=0.02, **kw):
    d = dict(features=tuple(features), means=tuple(means), scales=tuple(scales), coefficients=tuple(coefs), intercept=intercept,
             trained_through=TRAINED, alpha=1.0, training_rows=100, training_sha256="x" * 8, horizon=5)
    d.update(kw)
    return ModelArtifact(**d)


# --------------------------------------------------------------------------------- arithmetic
def test_hand_computed_prediction_and_contribution_decomposition():
    a = art()
    out = FundamentalScoreEngine(a).score({"a": 14.0, "b": 25.0}, AS_OF)
    # 0.02 + ((14-10)/2)*0.03 + ((25-20)/5)*(-0.01) = 0.02 + 0.06 - 0.01 = 0.07
    assert out["score"] == pytest.approx(0.07, abs=1e-15)
    assert out["components"]["a"] == pytest.approx(0.06, abs=1e-15) and out["components"]["b"] == pytest.approx(-0.01, abs=1e-15)
    assert out["intercept"] + sum(out["components"].values()) == pytest.approx(out["score"], abs=1e-15)
    assert out["status"] == "RESEARCH_MODEL" and out["deployment_approved"] is False
    assert out["confidence"] is None                       # no calibration object -> no invented probability
    assert "not probability" in out["score_units"]


def test_unused_features_do_not_change_the_score_and_extra_keys_are_ignored():
    e = FundamentalScoreEngine(art())
    base = e.score({"a": 14.0, "b": 25.0}, AS_OF)["score"]
    assert e.score({"a": 14.0, "b": 25.0, "zzz": None, "other": float("nan")}, AS_OF)["score"] == base


@pytest.mark.parametrize("seed", range(200), ids=lambda s: f"seed{s}")
def test_random_models_match_decimal_oracle(seed):
    rng = random.Random(seed)
    k = rng.randint(1, 8)
    names = tuple(f"f{i}" for i in range(k))
    means = tuple(round(rng.uniform(-5, 5), 6) for _ in range(k))
    scales = tuple(round(rng.uniform(0.01, 10), 6) for _ in range(k))
    coefs = tuple(round(rng.uniform(-0.5, 0.5), 6) for _ in range(k))
    icpt = round(rng.uniform(-0.1, 0.1), 6)
    x = [round(rng.uniform(-20, 20), 6) for _ in range(k)]
    out = FundamentalScoreEngine(art(names, means, scales, coefs, icpt)).score(dict(zip(names, x)), AS_OF)
    want, comps = O.linear_score(x, means, scales, coefs, icpt)
    assert O.rel_close(out["score"], want, 1e-12, 1e-14), (seed, out["score"], want)
    for n, c in zip(names, comps):
        assert O.rel_close(out["components"][n], c, 1e-12, 1e-14), (seed, n)


@pytest.mark.parametrize("seed", range(100), ids=lambda s: f"seed{s}")
def test_monotonic_in_each_feature_by_sign_of_coefficient(seed):
    rng = random.Random(5000 + seed)
    k = rng.randint(1, 6)
    names = tuple(f"f{i}" for i in range(k))
    scales = tuple(rng.uniform(0.1, 5) for _ in range(k))
    coefs = tuple(rng.choice([-1, 1]) * rng.uniform(0.001, 0.2) for _ in range(k))
    means = tuple(rng.uniform(-3, 3) for _ in range(k))
    e = FundamentalScoreEngine(art(names, means, scales, coefs, 0.01))
    base = {n: rng.uniform(-5, 5) for n in names}
    s0 = e.score(base, AS_OF)["score"]
    for n, c in zip(names, coefs):
        up = {**base, n: base[n] + rng.uniform(0.5, 3)}
        s1 = e.score(up, AS_OF)["score"]
        assert (s1 > s0) if c > 0 else (s1 < s0), (seed, n, c, s0, s1)
    # a score never depends on the ORDER of the supplied dict
    shuffled = dict(reversed(list(base.items())))
    assert e.score(shuffled, AS_OF)["score"] == pytest.approx(s0, abs=1e-15)


# ------------------------------------------------------------------------- fail-closed behaviour
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf")])
def test_missing_or_nonfinite_input_gives_no_score_never_a_default(bad):
    e = FundamentalScoreEngine(art())
    out = e.score({"a": 14.0, "b": bad}, AS_OF)
    assert out["score"] is None and out["status"] == "MISSING_FEATURES" and out["missing"] == ["b"] and out["components"] == {}
    out = e.score({"a": 14.0}, AS_OF)                      # key absent
    assert out["score"] is None and out["missing"] == ["b"]
    assert e.score({"b": bad}, AS_OF)["missing"] == ["a", "b"]


def test_dropping_a_negative_contribution_feature_never_improves_the_score():
    """If 'b' contributes negatively, removing it must NOT raise the score (no imputation/zero-fill)."""
    e = FundamentalScoreEngine(art())
    with_b = e.score({"a": 14.0, "b": 40.0}, AS_OF)
    assert with_b["components"]["b"] < 0
    assert e.score({"a": 14.0, "b": None}, AS_OF)["score"] is None


def test_nonfinite_model_output_fails_closed():
    e = FundamentalScoreEngine(art(coefs=(1e300, 1.0), scales=(1e-300, 1.0)))
    out = e.score({"a": 1e300, "b": 0.0}, AS_OF)
    assert out["score"] is None and out["status"] == "INVALID_MODEL_OUTPUT" and out["components"] == {}
    # and an overflowing difference: (1e308 - (-1e308)) -> inf
    e2 = FundamentalScoreEngine(art(means=(-1e308, 20.0)))
    assert e2.score({"a": 1e308, "b": 20.0}, AS_OF)["status"] == "INVALID_MODEL_OUTPUT"


def test_no_artifact_means_unvalidated_not_zero():
    out = FundamentalScoreEngine().score({"a": 1.0}, AS_OF)
    assert out["score"] is None and out["status"] == "UNVALIDATED" and out["confidence"] is None


def test_model_cannot_score_at_or_before_its_training_clock():
    e = FundamentalScoreEngine(art())
    with pytest.raises(ValueError, match="historical"):
        e.score({"a": 1.0, "b": 1.0}, TRAINED)                         # equal: refused (strictly after required)
    with pytest.raises(ValueError):
        e.score({"a": 1.0, "b": 1.0}, "2023-12-31T23:59:59.999999Z")
    assert e.score({"a": 1.0, "b": 1.0}, "2024-01-01T00:00:00.000001Z")["score"] is not None
    with pytest.raises(ValueError, match="timezone"):
        e.score({"a": 1.0, "b": 1.0}, "2024-06-01T00:00:00")


# --------------------------------------------------------------------------------- validation
@pytest.mark.parametrize("kw", [
    dict(alpha=-0.1), dict(alpha=float("nan")), dict(alpha=float("inf")), dict(horizon=0), dict(horizon=-5), dict(training_rows=0),
    dict(scales=(0.0, 5.0)), dict(scales=(-1.0, 5.0)), dict(scales=(float("nan"), 5.0)), dict(scales=(float("inf"), 5.0)),
    dict(means=(float("nan"), 20.0)), dict(means=(float("inf"), 20.0)), dict(coefs=(float("nan"), 1.0)), dict(coefs=(float("inf"), 1.0)),
    dict(intercept=float("nan")), dict(intercept=float("inf")), dict(features=("a", "a")), dict(features=()),
    dict(means=(1.0,)), dict(coefs=(1.0, 2.0, 3.0)), dict(deployment_approved=True), dict(trained_through="2024-01-01"),
])
def test_model_artifact_rejects_invalid_specifications(kw):
    base = dict(features=("a", "b"), means=(10.0, 20.0), scales=(2.0, 5.0), coefs=(0.03, -0.01), intercept=0.02)
    base.update(kw)
    with pytest.raises(ValueError):
        art(**base)


def test_model_artifact_is_frozen_and_id_is_deterministic():
    a, b = art(), art()
    assert a.id == b.id and a.id != art(intercept=0.021).id
    with pytest.raises(Exception):
        a.intercept = 5.0


# ------------------------------------------------------------------------------------ fit_model
def panel(rng, n, k, fit_gap=True):
    rows = []
    start = pd.Timestamp("2022-01-03", tz="UTC")
    for i in range(n):
        t = start + pd.Timedelta(days=i)
        xs = [rng.gauss(0, 1) * (j + 1) + 3 * j for j in range(k)]
        y = sum(0.01 * (j + 1) * x for j, x in enumerate(xs)) + rng.gauss(0, 0.02)
        rows.append(dict(as_of=t.isoformat(), label_end=(t + pd.Timedelta(days=5)).isoformat(), target=y, **{f"x{j}": x for j, x in enumerate(xs)}))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("seed", range(60), ids=lambda s: f"seed{s}")
def test_fit_model_matches_exact_ridge_and_uses_training_statistics_only(seed):
    rng = random.Random(900 + seed)
    k = rng.randint(1, 5)
    n = rng.randint(max(20, 5 * k), 160)
    p = panel(rng, n, k)
    feats = [f"x{j}" for j in range(k)]
    alpha = rng.choice([0.1, 1.0, 10.0, 100.0])
    fit_at = pd.Timestamp(p.label_end.max()) + pd.Timedelta(days=1)
    m = fit_model(p, feats, fit_at=fit_at, alpha=alpha, horizon=5)
    rows = [[Decimal(float(v)) for v in r] for r in p[feats].to_numpy(float)]
    means, scales, coef, ybar = O.ridge_fit([[float(x) for x in r] for r in rows], list(p.target), alpha)
    for got, want in zip(m.means, means):
        assert O.rel_close(got, Decimal(want.numerator) / Decimal(want.denominator), 1e-12, 1e-12), (seed, "mean")
    for got, want in zip(m.scales, scales):
        assert O.rel_close(got, Decimal(want.numerator) / Decimal(want.denominator), 1e-9, 1e-12), (seed, "scale")
    for got, want in zip(m.coefficients, coef):
        assert O.rel_close(got, Decimal(want.numerator) / Decimal(want.denominator), 1e-7, 1e-10), (seed, "coef", got, float(want))
    assert O.rel_close(m.intercept, Decimal(ybar.numerator) / Decimal(ybar.denominator), 1e-9, 1e-12), (seed, "intercept")
    assert m.trained_through == fit_at.isoformat() and m.training_rows == n and m.alpha == alpha and m.horizon == 5
    # scoring a held-out row uses the TRAINING mean/scale, never statistics of the row being scored
    row = {f"x{j}": 7.5 + j for j in range(k)}
    s = FundamentalScoreEngine(m).score(row, (fit_at + pd.Timedelta(days=1)).isoformat())["score"]
    want, _ = O.linear_score([row[f] for f in feats], m.means, m.scales, m.coefficients, m.intercept)
    assert O.rel_close(s, want, 1e-12, 1e-14)


def test_fit_model_training_stats_ignore_future_rows_that_are_not_passed():
    rng = random.Random(77)
    p = panel(rng, 120, 2)
    early = p.iloc[:80]
    fit_at = pd.Timestamp(early.label_end.max()) + pd.Timedelta(days=1)
    m1 = fit_model(early, ["x0", "x1"], fit_at=fit_at)
    spiked = p.copy()
    spiked.loc[100:, "x0"] = 1e6          # a later regime spike lives only in rows that are not training rows
    m2 = fit_model(spiked.iloc[:80], ["x0", "x1"], fit_at=fit_at)
    assert m1 == m2


def test_constant_feature_gets_unit_scale_and_zero_weight():
    rng = random.Random(5)
    p = panel(rng, 60, 2)
    p["c"] = 4.2
    m = fit_model(p, ["x0", "x1", "c"], fit_at=pd.Timestamp(p.label_end.max()) + pd.Timedelta(days=1))
    assert m.scales[2] == 1.0 and m.means[2] == pytest.approx(4.2) and abs(m.coefficients[2]) < 1e-12


@pytest.mark.parametrize("mutate,match", [
    (lambda p: p.drop(columns=["target"]), "timestamped labels"),
    (lambda p: p.drop(columns=["label_end"]), "timestamped labels"),
    (lambda p: p.assign(label_end=p.as_of), "invalid label intervals"),
    (lambda p: p.iloc[:19], "insufficient"),
    (lambda p: p.assign(x0=[float("nan")] + list(p.x0[1:])), "insufficient"),
    (lambda p: p.assign(target=[float("inf")] + list(p.target[1:])), "insufficient"),
])
def test_fit_model_rejects_bad_panels(mutate, match):
    p = panel(random.Random(3), 60, 2)
    with pytest.raises(ValueError, match=match):
        fit_model(mutate(p), ["x0", "x1"], fit_at=pd.Timestamp(p.label_end.max()) + pd.Timedelta(days=1))


def test_fit_model_rejects_unmatured_labels_and_boundary_is_strict():
    p = panel(random.Random(4), 60, 2)
    last = pd.Timestamp(p.label_end.max())
    with pytest.raises(ValueError, match="unmatured"):
        fit_model(p, ["x0", "x1"], fit_at=last)                                   # label_end == fit_at -> refused
    fit_model(p, ["x0", "x1"], fit_at=last + pd.Timedelta(microseconds=1))        # one microsecond later -> allowed
    with pytest.raises(ValueError, match="timezone"):
        fit_model(p, ["x0", "x1"], fit_at="2030-01-01")


def test_fit_model_needs_five_rows_per_feature():
    p = panel(random.Random(8), 24, 5)
    with pytest.raises(ValueError, match="insufficient"):
        fit_model(p, [f"x{j}" for j in range(5)], fit_at=pd.Timestamp(p.label_end.max()) + pd.Timedelta(days=1))   # 24 < 25
    p = panel(random.Random(8), 25, 5)
    fit_model(p, [f"x{j}" for j in range(5)], fit_at=pd.Timestamp(p.label_end.max()) + pd.Timedelta(days=1))


# ------------------------------------------------------------------- combined decision + thresholds
def engine_for(name="A", features=("revenue_yoy", "gross_margin", "technical_signal"), means=(0.1, 0.4, 0.5), scales=(0.1, 0.05, 0.5),
               coefs=(0.01, 0.02, 0.03), intercept=0.001, policy="filter", threshold=0.0, shadow=True, years=None, sector=None):
    g = GOLDEN[name]
    facts = annual_facts(name, years or g["years"])
    if sector or g["sector"]:
        facts.append(classification(name, sector or g["sector"]))
    store = new_store(facts)
    model = art(features, means, scales, coefs, intercept)
    fe = FundamentalFeatureEngine(store)
    return CombinedSignalEngine(fe, FundamentalScoreEngine(model), policy=policy, threshold=threshold, research_shadow=shadow), store


def test_decision_score_matches_hand_arithmetic_on_company_a():
    eng, _ = engine_for()
    d = eng.decide("A", AS_OF, {"signal": True})
    # 0.001 + (0.2-0.1)/0.1*0.01 + (0.5-0.4)/0.05*0.02 + (1-0.5)/0.5*0.03 = 0.001+0.01+0.04+0.03 = 0.081
    assert d["fundamental"]["score"] == pytest.approx(0.081, abs=1e-15)
    comps = d["fundamental"]["components"]
    assert comps["revenue_yoy"] == pytest.approx(0.01, abs=1e-14) and comps["gross_margin"] == pytest.approx(0.04, abs=1e-14)
    assert comps["technical_signal"] == pytest.approx(0.03, abs=1e-15)
    assert d["permit_signal"] is True and d["decision"] == "RESEARCH_CANDIDATE" and d["positionSize"] is None
    assert d["deployment_approved"] is False and d["risk"]["status"] == "NOT_EVALUATED"
    off = eng.decide("A", AS_OF, {"signal": False})
    # without the technical signal that component is (0-0.5)/0.5*0.03 = -0.03 -> score 0.021, but NO entry is permitted
    assert off["fundamental"]["score"] == pytest.approx(0.021, abs=1e-15) and off["permit_signal"] is False and off["decision"] == "NO_TRADE"


@pytest.mark.parametrize("threshold,permit", [(0.080999, True), (0.081001, False), (-1.0, True), (0.0, True), (1.0, False)])
def test_threshold_boundary_away_from_the_score(threshold, permit):
    eng, _ = engine_for(threshold=threshold)
    d = eng.decide("A", AS_OF, {"signal": True})
    assert d["fundamental"]["score"] == pytest.approx(0.081, abs=1e-15)
    assert d["permit_signal"] is permit


def test_threshold_exact_equality_ulp_neighbours():
    eng0, _ = engine_for()
    s = eng0.decide("A", AS_OF, {"signal": True})["fundamental"]["score"]
    for thr, want in [(s, False), (np.nextafter(s, -np.inf), True), (np.nextafter(s, np.inf), False)]:
        eng, _ = engine_for(threshold=float(thr))
        d = eng.decide("A", AS_OF, {"signal": True})
        assert d["permit_signal"] is want, (thr, s)
        assert d["configuration"]["threshold"] == float(thr)
        assert ("exceeds" in d["explanation"][0]) == want


def test_clean_threshold_boundary_069999_07_070001():
    eng, _ = engine_for(features=("technical_signal",), means=(0.0,), scales=(1.0,), coefs=(0.0,), intercept=0.07)
    for thr, want in [(0.069999, True), (0.07, False), (0.070001, False)]:
        e = CombinedSignalEngine(eng.features, eng.score, policy="filter", threshold=thr, research_shadow=True)
        assert e.decide("A", AS_OF, {"signal": True})["permit_signal"] is want, thr


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_threshold_must_be_finite(bad):
    eng, _ = engine_for()
    with pytest.raises(ValueError):
        CombinedSignalEngine(eng.features, eng.score, policy="filter", threshold=bad)


def test_invalid_policy_rejected():
    eng, _ = engine_for()
    for policy in ("", "FILTER", "blend", None, "weighted2"):
        with pytest.raises(ValueError):
            CombinedSignalEngine(eng.features, eng.score, policy=policy)


def test_missing_model_feature_means_no_candidate_with_reason_not_a_degraded_score():
    # drop revenue_yoy by removing the prior year's revenue -> model cannot score
    years = {y: dict(v) for y, v in GOLDEN["A"]["years"].items()}
    del years[2022]["revenue"]
    eng, _ = engine_for(years=years)
    d = eng.decide("A", AS_OF, {"signal": True})
    assert d["fundamental"]["score"] is None and d["fundamental"]["status"] == "MISSING_FEATURES" and "revenue_yoy" in d["fundamental"]["missing"]
    assert d["permit_signal"] is False and d["decision"] == "NO_TRADE"
    assert "Missing or unvalidated" in d["explanation"][0]


def test_completeness_is_reported_separately_from_score_and_never_blended():
    """Two companies with IDENTICAL model features but 40% vs 100% completeness: identical score, different coverage."""
    full_years = {y: dict(v) for y, v in GOLDEN["A"]["years"].items()}
    thin_years = {y: {k: v for k, v in d.items() if k in ("revenue", "gross_profit")} for y, d in GOLDEN["A"]["years"].items()}
    eng_full, _ = engine_for(years=full_years)
    eng_thin, _ = engine_for(years=thin_years)
    a = eng_full.decide("A", AS_OF, {"signal": True})
    b = eng_thin.decide("A", AS_OF, {"signal": True})
    assert a["fundamental"]["score"] == pytest.approx(b["fundamental"]["score"], abs=1e-15)
    assert b["fundamental"]["coverage"] < 0.4 < 0.7 < a["fundamental"]["coverage"] <= 1.0, (a["fundamental"]["coverage"], b["fundamental"]["coverage"])
    assert a["fundamental"]["confidence"] is None and b["fundamental"]["confidence"] is None   # coverage is not a probability
    assert a["fundamental"]["coverage"] != a["fundamental"]["score"]


def test_policy_and_shadow_truth_table_slice():
    # observe: score ignored, technical decides
    eng, _ = engine_for(policy="observe", shadow=False)
    d = eng.decide("A", AS_OF, {"signal": True})
    assert d["permit_signal"] is True and d["decision"] == "TECHNICAL_CANDIDATE"
    assert eng.decide("A", AS_OF, {"signal": False})["decision"] == "NO_TRADE"
    # filter without shadow: nothing is ever permitted (no approved model)
    eng, _ = engine_for(policy="filter", shadow=False)
    d = eng.decide("A", AS_OF, {"signal": True})
    assert d["permit_signal"] is False and d["explanation"] == ["No fundamental model approved for deployment"]
    # weighted needs a technical input inside the model, else it never permits
    eng, _ = engine_for(policy="weighted", features=("revenue_yoy",), means=(0.1,), scales=(0.1,), coefs=(0.01,))
    d = eng.decide("A", AS_OF, {"signal": True})
    assert d["permit_signal"] is False and any("technical input" in x for x in d["explanation"])
    # regime needs AVAILABLE macro: no macro service -> status MISSING -> refuse
    eng, _ = engine_for(policy="regime")
    d = eng.decide("A", AS_OF, {"signal": True})
    assert d["permit_signal"] is False and "Macro vintage coverage missing/stale" in d["explanation"]


def test_decision_hash_is_sensitive_to_every_input_and_stable_across_calls():
    eng, store = engine_for()
    a = eng.decide("A", AS_OF, {"signal": True})
    b = eng.decide("A", AS_OF, {"signal": True})
    assert a["data_snapshot_id"] == b["data_snapshot_id"]
    c = eng.decide("A", "2024-06-02T00:00:00Z", {"signal": True})
    assert c["data_snapshot_id"] != a["data_snapshot_id"]     # as_of/age is part of the snapshot
    assert a["configuration"]["model"]["intercept"] == 0.001 and a["configuration"]["feature_version"] == "fundamentals-v4-audited"
