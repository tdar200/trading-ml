"""Track C: deterministic golden companies through the PUBLIC path.

Facts ('GBP millions') -> PointInTimeFundamentalStore.ingest -> FundamentalFeatureEngine.snapshot,
then valuation_features on the snapshot's raw statement values. Every returned feature is compared to
the independent Decimal oracle (tests/qa_reaudit/calc_oracle.py) at 1e-12 relative, and a subset is
compared to numbers derived by hand on paper (calc_golden.GOLDEN[...]['hand']).
Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_calc_golden.py
"""
import math
from decimal import Decimal

import pytest

from fundamentals.features import valuation_features

from tests.qa_reaudit import calc_oracle as O
from tests.qa_reaudit.calc_fixtures import annual_facts, base_dict, classification, golden_snapshot, new_store, snapshot
from tests.qa_reaudit.calc_golden import GOLDEN

AS_OF = "2024-06-01T00:00:00Z"
REL = 1e-12


def oracle_for(name, *, drop_current=(), drop_prior=(), sector="__spec__"):
    g = GOLDEN[name]
    yrs = g["years"]
    cur = base_dict(yrs[2023])
    prior = base_dict(yrs[2022])
    old = base_dict(yrs[2021])
    for k in drop_current:
        cur.pop(k, None)
    for k in drop_prior:
        prior.pop(k, None)
    sec = g["sector"] if sector == "__spec__" else sector
    feats, flags = O.accounting(cur, prior, old, sec)
    feats["revenue_cagr_3y"] = O.cagr(cur.get("revenue"), base_dict(yrs[2020]).get("revenue") if 2020 in yrs else None, 3)
    feats["revenue_cagr_5y"] = O.cagr(cur.get("revenue"), base_dict(yrs[2018]).get("revenue") if 2018 in yrs else None, 5)
    for m in ("revenue", "eps", "ocf", "operating_income"):
        feats[m + "_qoq"] = None
    return feats, flags, cur


def compare(actual, expected, label, rel=REL, abs_tol=1e-15):
    bad = []
    for key in sorted(set(actual) | set(expected)):
        if key not in actual:
            bad.append(f"{label}:{key} missing in production")
            continue
        if key not in expected:
            bad.append(f"{label}:{key} unexpected in production (value {actual[key]!r})")
            continue
        a, e = actual[key], expected[key]
        if a is not None and not isinstance(a, bool) and not math.isfinite(a):
            bad.append(f"{label}:{key} non-finite {a!r}")
        elif not O.rel_close(a, e, rel, abs_tol):
            bad.append(f"{label}:{key} production={a!r} oracle={e!r}")
    return bad


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_snapshot_every_feature_matches_oracle(name):
    g = GOLDEN[name]
    snap, _ = golden_snapshot(name, g["years"], AS_OF, g["sector"])
    assert snap["status"] == "AVAILABLE", snap["status"]
    assert snap["period_end"] == "2023-12-31" and snap["currency"] == "GBP"
    feats, flags, _ = oracle_for(name)
    bad = compare(snap["features"], feats, name)
    assert not bad, "\n".join(bad)
    assert sorted(snap["flags"]) == sorted(flags), (snap["flags"], flags)
    assert sorted(snap["flags"]) == sorted(g["hand_flags"]), ("hand-derived flags differ", snap["flags"], g["hand_flags"])
    # completeness is exactly the share of non-None features
    n = len(snap["features"])
    assert snap["completeness"] == sum(v is not None for v in snap["features"].values()) / n


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_hand_derived_values_match_production_and_oracle(name):
    g = GOLDEN[name]
    snap, _ = golden_snapshot(name, g["years"], AS_OF, g["sector"])
    feats, _, _ = oracle_for(name)
    for key, hand in g["hand"].items():
        prod, orc = snap["features"][key], feats[key]
        if hand is None:
            assert prod is None and orc is None, (name, key, prod, orc)
        else:
            assert O.rel_close(prod, Decimal(hand), 1e-12), (name, key, "production", prod, "hand", hand)
            assert O.rel_close(orc, Decimal(hand), 1e-18), (name, key, "ORACLE disagrees with hand", orc, hand)


def test_company_a_supplied_numbers():
    """The user's Company A, asserted against literal constants from the brief."""
    g = GOLDEN["A"]
    snap, _ = golden_snapshot("A", g["years"], AS_OF)
    f = snap["features"]
    assert f["revenue_yoy"] == pytest.approx(0.20, rel=1e-12)          # 20%
    assert f["gross_margin"] == pytest.approx(0.50, rel=1e-12)         # 50%
    assert f["operating_margin"] == pytest.approx(0.20, rel=1e-12)
    assert f["net_margin"] == pytest.approx(0.15, rel=1e-12)
    assert f["fcf"] == pytest.approx(20_000_000, rel=1e-12)            # GBP 20m
    assert f["fcf_margin"] == pytest.approx(0.166666666666667, rel=1e-12)
    assert f["current_ratio"] == pytest.approx(2.0, rel=1e-12)
    assert f["debt_equity"] == pytest.approx(0.2222222222222222, rel=1e-12)
    snap_raw = snap["raw"]
    val = valuation_features(20, 10_000_000, {**snap_raw, "preferred_equity": 0.0, "minority_interest": 0.0},
                             price_currency="GBP", statement_currency="GBP", price_basis="X", share_basis="X", forward_eps=2.0)
    assert val["market_cap"] == pytest.approx(200_000_000, rel=1e-12)  # 200m
    assert val["pe"] == pytest.approx(11.11111111111111, rel=1e-12)
    assert val["price_fcf"] == pytest.approx(10.0, rel=1e-12)
    assert val["fcf_yield"] == pytest.approx(0.10, rel=1e-12)
    # units: statements were entered in 'GBP millions' and are stored in GBP
    assert snap_raw["revenue"] == 120_000_000 and snap_raw["cash"] == 30_000_000


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_valuation_features_match_oracle_and_hand(name):
    g = GOLDEN[name]
    snap, _ = golden_snapshot(name, g["years"], AS_OF, g["sector"])
    raw = {**snap["raw"], "preferred_equity": 0.0, "minority_interest": 0.0}
    price, shares = g["price"], g["shares_out"]
    fwd = g["forward_eps"]
    got = valuation_features(float(price), float(shares), raw, price_currency="GBP", statement_currency="GBP",
                             price_basis="S", share_basis="S", forward_eps=float(fwd))
    cur = base_dict(g["years"][2023])
    cur["preferred_equity"] = Decimal(0)
    cur["minority_interest"] = Decimal(0)
    want = O.valuation(price, shares, cur, forward_eps=fwd)
    bad = compare(got, want, name)
    assert not bad, "\n".join(bad)
    for key, hand in g.get("hand_valuation", {}).items():
        if hand is None:
            assert got[key] is None, (name, key, got[key])
        else:
            assert O.rel_close(got[key], Decimal(hand), 1e-12), (name, key, got[key], hand)
    # negative / zero denominators never become 0, inf or NaN
    for key, value in got.items():
        assert value is None or math.isfinite(value), (name, key, value)


def test_negative_pe_is_undefined_not_cheap_and_yields_keep_sign():
    g = GOLDEN["D"]
    snap, _ = golden_snapshot("D", g["years"], AS_OF)
    raw = {**snap["raw"], "preferred_equity": 0.0, "minority_interest": 0.0}
    v = valuation_features(5.0, 40e6, raw, price_currency="GBP", statement_currency="GBP", price_basis="S", share_basis="S",
                           forward_eps=-0.1)
    assert v["pe"] is None and v["forward_pe"] is None and v["price_fcf"] is None
    assert v["earnings_yield"] == pytest.approx(-0.05, rel=1e-12)
    assert v["fcf_yield"] == pytest.approx(-0.04, rel=1e-12)


def test_ev_needs_explicit_preferred_and_minority_claims():
    """Documented fail-safe: missing preferred/minority -> EV, EV/Revenue, EV/EBITDA unavailable (not zero)."""
    snap, _ = golden_snapshot("A", GOLDEN["A"]["years"], AS_OF)
    v = valuation_features(20.0, 10e6, snap["raw"], price_currency="GBP", statement_currency="GBP", price_basis="S", share_basis="S")
    assert v["enterprise_value"] is None and v["ev_revenue"] is None and v["ev_ebitda"] is None
    assert v["pe"] == pytest.approx(200e6 / 18e6, rel=1e-12)


def test_percent_vs_fraction_convention_everywhere():
    """Every ratio is a FRACTION: 0.5 means 50 percent; day counts are days; nothing is multiplied by 100."""
    snap, _ = golden_snapshot("A", GOLDEN["A"]["years"], AS_OF)
    f = snap["features"]
    assert f["gross_margin"] == 0.5 and f["operating_margin"] == pytest.approx(0.2) and f["revenue_yoy"] == pytest.approx(0.2)
    ratios = [k for k in f if k.endswith(("margin", "_yoy", "_cagr_3y", "_cagr_5y", "roe", "roa", "roic"))]
    assert ratios
    for k in ratios:
        assert f[k] is None or abs(f[k]) < 5, (k, f[k])  # a percent-scaled value would be >= 10 for these fixtures
    # days are in days (receivable days ~ 365 * avg receivables / revenue = 365*13.5/120)
    assert f["receivable_days"] == pytest.approx(365 * 13.5 / 120, rel=1e-12)


def test_no_nan_inf_or_zero_in_missing_propagation_all_goldens():
    for name, g in GOLDEN.items():
        snap, _ = golden_snapshot(name, g["years"], AS_OF, g["sector"])
        for k, v in snap["features"].items():
            assert v is None or (isinstance(v, float) and math.isfinite(v)), (name, k, v)


# ----------------------------------------------------------------- missing-input ablation (None never 0)
CUR_METRICS = ["revenue", "gross_profit", "cogs", "operating_income", "net_income", "eps", "ocf", "capex", "interest_expense",
               "pretax_income", "tax_expense", "depreciation", "diluted_shares", "rd", "buybacks", "dividends", "issuance",
               "acquisitions", "cash", "current_assets", "assets", "current_liabilities", "short_debt", "long_debt", "equity",
               "inventory", "receivables", "payables", "prepaids"]


@pytest.mark.parametrize("metric", CUR_METRICS)
def test_dropping_a_current_input_matches_oracle_and_never_imputes_zero(metric):
    g = GOLDEN["A"]
    years = {y: dict(v) for y, v in g["years"].items()}
    if metric not in years[2023]:
        pytest.skip("not in fixture")
    del years[2023][metric]
    if metric == "revenue":
        pytest.skip("revenue anchors the annual period; covered by test_missing_revenue_means_no_annual_snapshot")
    snap, _ = golden_snapshot("A", years, AS_OF)
    feats, flags, _ = oracle_for("A", drop_current=(metric,))
    bad = compare(snap["features"], feats, f"A-minus-{metric}")
    assert not bad, "\n".join(bad)
    assert sorted(snap["flags"]) == sorted(flags)


@pytest.mark.parametrize("metric", CUR_METRICS)
def test_dropping_a_prior_year_input_matches_oracle(metric):
    g = GOLDEN["A"]
    years = {y: dict(v) for y, v in g["years"].items()}
    if metric not in years[2022] or metric == "revenue":
        pytest.skip("n/a")
    del years[2022][metric]
    snap, _ = golden_snapshot("A", years, AS_OF)
    feats, flags, _ = oracle_for("A", drop_prior=(metric,))
    bad = compare(snap["features"], feats, f"A-prior-minus-{metric}")
    assert not bad, "\n".join(bad)
    assert sorted(snap["flags"]) == sorted(flags)


def test_missing_revenue_means_no_annual_snapshot():
    years = {y: dict(v) for y, v in GOLDEN["A"]["years"].items()}
    del years[2023]["revenue"]
    snap, _ = golden_snapshot("A", years, AS_OF)
    # FY2022 becomes the latest annual period (never silently mixes FY2023 flows)
    assert snap["period_end"] == "2022-12-31"
    assert snap["features"]["revenue"] if "revenue" in snap["features"] else True


def test_explicit_none_fact_is_missing_not_zero():
    """A Fact with value=None (withdrawn/unknown) must behave like an absent input."""
    from fundamentals.store import Fact
    g = GOLDEN["A"]
    facts = annual_facts("A", g["years"])
    facts.append(Fact("A", "gross_profit", None, "2023-12-31", "2024-03-30T10:00:00Z", "2024-03-30T10:00:00Z", "fixture", "withdrawn",
                      "GBP", "GBP", "2023-01-01", metadata={"share_basis": "one", "accounting_standard": "us-gaap"}))
    store = new_store(facts)
    f = snapshot(store, "A", AS_OF)["features"]
    for k in ("gross_margin", "gross_margin_change", "gross_profitability", "gross_margin_stability"):
        assert f[k] is None, (k, f[k])
    assert f["operating_margin"] == pytest.approx(0.2)


# ----------------------------------------------------------------- industry gating
def test_bank_vs_software_not_mixed_in_ratio_gating():
    g = GOLDEN["K"]
    bank, _ = golden_snapshot("K", g["years"], AS_OF, "banks")
    plain, _ = golden_snapshot("K", g["years"], AS_OF, None)
    for k in ("current_ratio", "debt_equity", "net_debt_ebitda", "interest_coverage", "cash_debt", "quick_ratio"):
        assert bank["features"][k] is None, k
    assert plain["features"]["current_ratio"] == pytest.approx(1600 / 1500)
    assert plain["features"]["debt_equity"] == pytest.approx(1080 / 1100)
    for sector in ("saas", "financials", "insurance"):
        snap, _ = golden_snapshot("K", g["years"], AS_OF, sector)
        assert snap["features"]["receivable_days"] is None
    # sector is chosen from PIT classification, never from the symbol
    assert bank["sector"] == "banks" and plain["sector"] is None


# ----------------------------------------------------------------- unit scale invariance
@pytest.mark.parametrize("scale_unit,divisor", [("GBP millions", 1_000_000), ("GBP thousands", 1000), ("GBP billions", 1_000_000_000), ("GBP", 1)])
def test_scale_invariance_millions_thousands_billions(scale_unit, divisor):
    """Same economic statement entered at four scales must give identical snapshots."""
    from fundamentals.store import Fact
    from tests.qa_reaudit.calc_fixtures import FLOW, SHARE_METRICS
    g = GOLDEN["A"]
    base, _ = golden_snapshot("A", g["years"], AS_OF)
    facts = []
    for y, metrics in g["years"].items():
        for m, v in metrics.items():
            if m in SHARE_METRICS or m == "eps":
                continue
            val = float(Decimal(str(v)) * 1_000_000 / divisor)
            end, start = f"{y}-12-31", f"{y}-01-01"
            facts.append(Fact("A", m, val, end, f"{y + 1}-02-25T21:30:00Z", f"{y + 1}-02-25T21:30:00Z", "fixture", f"FY{y}", "GBP",
                              scale_unit, start if m in FLOW else "", metadata={"share_basis": "one", "accounting_standard": "us-gaap"}))
    from tests.qa_reaudit.calc_fixtures import fact as mkfact
    for y, metrics in g["years"].items():
        for m in ("eps", "diluted_shares"):
            if m in metrics:
                facts.append(mkfact("A", m, metrics[m], f"{y}-12-31", f"{y}-01-01", f"{y + 1}-02-25T21:30:00Z", revision=f"FY{y}"))
    snap = snapshot(new_store(facts), "A", AS_OF)
    bad = compare(snap["features"], {k: (None if v is None else Decimal(v)) for k, v in base["features"].items()}, scale_unit, rel=1e-12)
    assert not bad, "\n".join(bad)
    assert snap["raw"]["revenue"] == pytest.approx(120_000_000, rel=1e-12)


def test_one_million_string_vs_scale_word_cannot_be_confused():
    """'1 million' as text units is not silently accepted as a scale (fails loudly)."""
    from fundamentals.store import Fact
    f = Fact("A", "revenue", 1.0, "2023-12-31", "2024-02-25T00:00:00Z", "2024-02-25T00:00:00Z", "fixture", "r", "GBP", "1 million",
             "2023-01-01")
    with pytest.raises(ValueError):
        f.normalized()
