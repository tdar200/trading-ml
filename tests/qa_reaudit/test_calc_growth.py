"""Track C phases 14-30: growth, sign-change conventions, acceleration/momentum, CAGR, period matching, documentation.

Public path: Facts -> store -> FundamentalFeatureEngine.snapshot. Expected numbers are derived by hand.
Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_calc_growth.py
"""
import pytest

from fundamentals.dictionary import feature_dictionary
from fundamentals.store import Fact

from tests.qa_reaudit import calc_oracle as O
from tests.qa_reaudit.calc_fixtures import annual_facts, classification, golden_snapshot, new_store, snapshot
from tests.qa_reaudit.calc_golden import GOLDEN

AS_OF = "2024-06-01T00:00:00Z"
M = 1_000_000


def snap2(prior, current, metric="eps", extra=None):
    ys = {2022: {"revenue": "100", metric: prior, **(extra or {})}, 2023: {"revenue": "100", metric: current, **(extra or {})}}
    s, _ = golden_snapshot("G", ys, AS_OF)
    return s["features"]


@pytest.mark.parametrize("prior,current,expected", [
    ("1", "1.2", 0.2),            # +20%
    ("1", "1", 0.0),
    ("2", "0", -1.0),             # profit wiped out: -100%
    ("1", "-1", -2.0),            # +1 -> -1: current/prior - 1 = -2 (i.e. -200%)
    ("0.5", "5", 9.0),
    ("0.000001", "1", 999999.0),  # tiny positive base is NOT capped (convention: positive base only)
    ("-1", "1", None),            # sign change from a loss: growth undefined under the positive-base convention
    ("0", "1", None),             # zero base
    ("-1", "-0.5", None),         # narrowing loss: undefined, never reported as "improved +50%"
    ("-1", "-2", None),
    ("-2", "0", None),
])
def test_eps_growth_sign_change_matrix(prior, current, expected):
    got = snap2(prior, current)["eps_yoy"]
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("metric", ["operating_income", "cash", "assets", "receivables", "inventory"])
def test_other_yoy_metrics_use_the_same_positive_base_convention(metric):
    assert snap2("50", "60", metric)[metric + "_yoy"] == pytest.approx(0.2, rel=1e-12)
    assert snap2("-50", "60", metric)[metric + "_yoy"] is None
    assert snap2("0", "60", metric)[metric + "_yoy"] is None
    assert snap2("50", "-10", metric)[metric + "_yoy"] == pytest.approx(-1.2, rel=1e-12)


def test_fcf_growth_with_sign_changes_and_negative_base():
    def f(prior_ocf, prior_capex, ocf, capex):
        ys = {2022: {"revenue": "100", "ocf": prior_ocf, "capex": prior_capex}, 2023: {"revenue": "100", "ocf": ocf, "capex": capex}}
        return golden_snapshot("F", ys, AS_OF)[0]["features"]
    assert f("30", "10", "36", "10")["fcf_yoy"] == pytest.approx(0.3, rel=1e-12)       # 20 -> 26
    assert f("30", "10", "5", "10")["fcf_yoy"] == pytest.approx(-1.25, rel=1e-12)      # 20 -> -5
    assert f("5", "10", "30", "10")["fcf_yoy"] is None                                  # -5 -> 20: undefined, not +500%
    assert f("10", "10", "30", "10")["fcf_yoy"] is None                                 # 0 -> 20
    assert f("30", "10", None, None) ["fcf_yoy"] is None                                 # missing current -> None


def test_revenue_growth_5_10_15_20_vs_20_15_10_5_acceleration():
    up = [100, 105, 115.5, 132.825, 159.39]       # growth 5%,10%,15%,20%
    dn = [100, 120, 138, 151.8, 159.39]           # growth 20%,15%,10%,5%
    def build(path):
        ys = {2019 + i: {"revenue": str(v)} for i, v in enumerate(path)}
        facts = annual_facts("MOM", ys)
        return snapshot(new_store(facts), "MOM", "2025-06-01T00:00:00Z")
    a = build(up)
    assert a["period_end"] == "2023-12-31"
    assert a["features"]["revenue_yoy"] == pytest.approx(0.20, rel=1e-12)
    assert a["features"]["revenue_acceleration"] == pytest.approx(0.05, rel=1e-12)
    assert "REVENUE_DECELERATION" not in a["flags"]
    b = build(dn)
    assert b["features"]["revenue_yoy"] == pytest.approx(0.05, rel=1e-12)
    assert b["features"]["revenue_acceleration"] == pytest.approx(-0.05, rel=1e-12)
    assert "REVENUE_DECELERATION" in b["flags"]
    # 3y CAGR of the accelerating path: (159.39/105)^(1/3)-1 ; identity check without trusting any constant
    assert (1 + a["features"]["revenue_cagr_3y"]) ** 3 == pytest.approx(159.39 / 105, rel=1e-12)


def test_cagr_hand_values_and_guards():
    ys = {2018: {"revenue": "60"}, 2020: {"revenue": "80"}, 2022: {"revenue": "100"}, 2023: {"revenue": "120"}}
    f = golden_snapshot("C", ys, AS_OF)[0]["features"]
    assert f["revenue_cagr_3y"] == pytest.approx(1.5 ** (1 / 3) - 1, rel=1e-12)
    assert (1 + f["revenue_cagr_3y"]) ** 3 == pytest.approx(1.5, rel=1e-12)       # 80 -> 120 over three years
    assert f["revenue_cagr_5y"] == pytest.approx(2 ** 0.2 - 1, rel=1e-12)
    assert (1 + f["revenue_cagr_5y"]) ** 5 == pytest.approx(2.0, rel=1e-12)       # 60 -> 120 over five years
    # zero / negative start and missing start -> None, never an error or infinity
    for start in ("0", "-10"):
        g = golden_snapshot("C", {2020: {"revenue": start}, 2023: {"revenue": "120"}}, AS_OF)[0]["features"]
        assert g["revenue_cagr_3y"] is None
    assert golden_snapshot("C", {2023: {"revenue": "120"}}, AS_OF)[0]["features"]["revenue_cagr_3y"] is None
    # shrinking revenue gives a negative CAGR
    h = golden_snapshot("C", {2020: {"revenue": "200"}, 2023: {"revenue": "100"}}, AS_OF)[0]["features"]
    assert h["revenue_cagr_3y"] == pytest.approx(0.5 ** (1 / 3) - 1, rel=1e-12) and h["revenue_cagr_3y"] < 0


def test_eps_fcf_margin_acceleration_hand_values():
    ys = {2021: {"revenue": "100", "eps": "1.0", "ocf": "30", "capex": "10", "gross_profit": "40", "operating_income": "10"},
          2022: {"revenue": "110", "eps": "1.2", "ocf": "40", "capex": "10", "gross_profit": "46.2", "operating_income": "13.2"},
          2023: {"revenue": "121", "eps": "1.5", "ocf": "60", "capex": "10", "gross_profit": "54.45", "operating_income": "18.15"}}
    f = golden_snapshot("ACC", ys, AS_OF)[0]["features"]
    assert f["eps_yoy"] == pytest.approx(0.25, rel=1e-12) and f["eps_acceleration"] == pytest.approx(0.25 - 0.2, rel=1e-12)
    assert f["fcf_yoy"] == pytest.approx(50 / 30 - 1, rel=1e-12) and f["fcf_acceleration"] == pytest.approx((50 / 30 - 1) - (30 / 20 - 1), rel=1e-12)
    assert f["gross_margin"] == pytest.approx(0.45, rel=1e-12) and f["gross_margin_change"] == pytest.approx(0.45 - 0.42, rel=1e-12)
    assert f["operating_margin"] == pytest.approx(0.15, rel=1e-12) and f["operating_margin_change"] == pytest.approx(0.15 - 0.12, rel=1e-12)
    assert f["revenue_acceleration"] == pytest.approx(0.1 - 0.1, abs=1e-12)


def test_acceleration_needs_three_years_else_none():
    ys = {2022: {"revenue": "100", "eps": "1"}, 2023: {"revenue": "110", "eps": "1.1"}}
    f = golden_snapshot("TWO", ys, AS_OF)[0]["features"]
    assert f["revenue_yoy"] == pytest.approx(0.1) and f["revenue_acceleration"] is None and f["eps_acceleration"] is None and f["gross_margin_stability"] is None


# ---------------------------------------------------------------------------- period matching
def fy_facts(symbol, rows):
    """rows: list of (period_start, period_end, revenue). Always published 60 days after the end."""
    import datetime as dt
    out = []
    for start, end, rev in rows:
        pub = (dt.date.fromisoformat(end) + dt.timedelta(days=60)).isoformat() + "T21:00:00Z"
        out.append(Fact(symbol, "revenue", float(rev) * M, end, pub, pub, "fixture", "r" + end, "GBP", "GBP", start, metadata={"accounting_standard": "us-gaap"}))
    return out


@pytest.mark.parametrize("drift,matched", [(0, True), (3, True), (10, True), (11, False), (-10, True), (-11, False), (30, False)])
def test_prior_year_matching_tolerance_is_ten_days(drift, matched):
    import datetime as dt
    cur_end = dt.date(2023, 9, 30)
    prior_end = dt.date(2022, 9, 30) + dt.timedelta(days=drift)
    rows = [((prior_end - dt.timedelta(days=364)).isoformat(), prior_end.isoformat(), 100), ((cur_end - dt.timedelta(days=364)).isoformat(), cur_end.isoformat(), 110)]
    s = snapshot(new_store(fy_facts("P", rows)), "P", "2024-05-01T00:00:00Z")
    assert s["period_end"] == "2023-09-30"
    if matched:
        assert s["features"]["revenue_yoy"] == pytest.approx(0.1, rel=1e-12)
    else:
        assert s["features"]["revenue_yoy"] is None      # never compare against a period that is not ~one year earlier


def test_52_53_week_fiscal_year_ends_still_match():
    rows = [("2021-10-03", "2022-10-01", 100), ("2022-10-02", "2023-09-30", 105), ("2023-10-01", "2024-09-28", 120)]
    s = snapshot(new_store(fy_facts("W", rows)), "W", "2025-02-01T00:00:00Z")
    assert s["period_end"] == "2024-09-28" and s["features"]["revenue_yoy"] == pytest.approx(120 / 105 - 1, rel=1e-12)
    assert s["features"]["revenue_acceleration"] == pytest.approx((120 / 105 - 1) - 0.05, rel=1e-12)


def test_non_annual_durations_are_not_annual_periods():
    """A 6-month transition period and a 3-month quarter are never used as the 'annual' revenue."""
    rows = [("2023-01-01", "2023-06-30", 60), ("2023-04-01", "2023-06-30", 30)]
    s = snapshot(new_store(fy_facts("T", rows)), "T", "2024-05-01T00:00:00Z")
    assert s["status"] == "MISSING" and s["features"] == {}


def test_annual_period_duration_bounds_300_to_380_days():
    for days, ok in ((299, False), (300, True), (380, True), (381, False)):
        import datetime as dt
        end = dt.date(2023, 12, 31)
        start = end - dt.timedelta(days=days - 1)
        s = snapshot(new_store(fy_facts("D", [(start.isoformat(), end.isoformat(), 100)])), "D", "2024-05-01T00:00:00Z")
        assert (s["status"] == "AVAILABLE") is ok, days


def test_prior_year_in_other_currency_is_never_converted_it_fails_closed_or_is_ignored():
    """FY22 revenue in USD, FY23 in GBP. Behaviour pinned: the TTM/quarter builder raises (clear ValueError) for the
    whole snapshot - the annual block is not produced either (LIMITATION: all-or-nothing). Never a converted number."""
    facts = annual_facts("FX", {2023: {"revenue": "110"}}) + annual_facts("FX", {2022: {"revenue": "100"}}, currency="USD")
    with pytest.raises(ValueError, match="ambiguous"):
        snapshot(new_store(facts), "FX", AS_OF)
    # a non-flow (balance sheet) line in another currency for the prior year is simply not matched
    facts = annual_facts("FX", {2022: {"revenue": "100"}, 2023: {"revenue": "110", "cash": "5"}})
    facts += [Fact("FX", "cash", 7e6, "2022-12-31", "2023-02-25T21:30:00Z", "2023-02-25T21:30:00Z", "fixture", "usd-cash", "USD", "USD", "",
                   metadata={"accounting_standard": "us-gaap"})]
    s = snapshot(new_store(facts), "FX", AS_OF)
    assert s["features"]["cash_yoy"] is None and s["features"]["revenue_yoy"] == pytest.approx(0.1)


def test_two_sources_for_one_period_fail_closed():
    facts = annual_facts("DS", {2022: {"revenue": "100"}, 2023: {"revenue": "110"}})
    facts += annual_facts("DS", {2023: {"revenue": "111"}}, source="vendor-b")
    with pytest.raises(ValueError, match="ambiguous"):
        snapshot(new_store(facts), "DS", AS_OF)


def test_quarterly_rows_never_leak_into_annual_features():
    """A 3M revenue row sharing the FY end date must not replace the annual revenue."""
    facts = annual_facts("QL", {2022: {"revenue": "400"}, 2023: {"revenue": "440"}})
    facts.append(Fact("QL", "revenue", 120e6, "2023-12-31", "2024-02-25T21:30:00Z", "2024-02-25T21:30:00Z", "fixture", "Q4", "GBP", "GBP", "2023-10-01",
                      metadata={"accounting_standard": "us-gaap"}))
    s = snapshot(new_store(facts), "QL", AS_OF)
    assert s["raw"]["revenue"] == pytest.approx(440e6) and s["features"]["revenue_yoy"] == pytest.approx(0.1, rel=1e-12)


def test_missing_prior_year_leaves_growth_none_but_levels_present():
    s = golden_snapshot("L", {2023: GOLDEN["A"]["years"][2023]}, AS_OF)[0]["features"]
    assert s["gross_margin"] == pytest.approx(0.5) and s["revenue_yoy"] is None and s["roe"] is None and s["roic"] is None and s["roa"] is None


# --------------------------------------------------------------------------- ROE / ROA / ROIC hand
def test_roe_roa_use_average_balances_not_ending_balances():
    ys = {2022: {"revenue": "100", "equity": "80", "assets": "140"}, 2023: {"revenue": "120", "net_income": "18", "equity": "90", "assets": "150"}}
    f = golden_snapshot("R", ys, AS_OF)[0]["features"]
    assert f["roe"] == pytest.approx(18 / 85, rel=1e-12) != pytest.approx(18 / 90, rel=1e-6)
    assert f["roa"] == pytest.approx(18 / 145, rel=1e-12)


def test_roic_hand_computation_and_tax_guard():
    base = {2022: {"revenue": "100", "short_debt": "5", "long_debt": "20", "equity": "80", "cash": "20"},
            2023: {"revenue": "120", "operating_income": "24", "pretax_income": "22.5", "tax_expense": "4.5", "short_debt": "5", "long_debt": "15", "equity": "90", "cash": "30"}}
    f = golden_snapshot("RO", base, AS_OF)[0]["features"]
    # NOPAT = 24 * (1 - 4.5/22.5) = 19.2 ; avg invested = ((20+90-30) + (25+80-20))/2 = 82.5
    assert f["roic"] == pytest.approx(19.2 / 82.5, rel=1e-12)
    for tax, pretax in (("-1", "22.5"), ("25", "22.5"), ("1", "0"), ("1", "-5")):
        b2 = {**base, 2023: {**base[2023], "tax_expense": tax, "pretax_income": pretax}}
        assert golden_snapshot("RO", b2, AS_OF)[0]["features"]["roic"] is None, (tax, pretax)   # tax rate outside [0,1] / undefined
    b3 = {**base, 2023: {**base[2023], "tax_expense": "22.5", "pretax_income": "22.5"}}
    assert golden_snapshot("RO", b3, AS_OF)[0]["features"]["roic"] == pytest.approx(0.0, abs=1e-15)    # 100% tax: NOPAT 0, still defined
    b4 = {**base, 2023: {**base[2023], "tax_expense": "0", "pretax_income": "22.5"}}
    assert golden_snapshot("RO", b4, AS_OF)[0]["features"]["roic"] == pytest.approx(24 / 82.5, rel=1e-12)  # 0% tax boundary included


def test_roic_negative_invested_capital_is_undefined():
    ys = {2022: {"revenue": "100", "short_debt": "0", "long_debt": "0", "equity": "10", "cash": "50"},
          2023: {"revenue": "120", "operating_income": "10", "pretax_income": "10", "tax_expense": "2", "short_debt": "0", "long_debt": "0", "equity": "10", "cash": "50"}}
    assert golden_snapshot("N", ys, AS_OF)[0]["features"]["roic"] is None     # invested = 0+10-50 < 0


# --------------------------------------------------------------------------- documentation consistency
def test_feature_dictionary_covers_exactly_the_snapshot_features_and_dimensions():
    snap, _ = golden_snapshot("A", GOLDEN["A"]["years"], AS_OF)
    d = feature_dictionary()
    assert set(d) == set(snap["features"])
    for k in snap["features"]:
        assert d[k]["dimension"] == snap["dimensions"][k], k
        assert d[k]["formula"] and d[k]["missing_rule"].startswith("None")
        assert d[k]["production_weight"] is None and d[k]["local_predictive_usefulness"] == "NOT_ESTABLISHED"


def test_not_implemented_spec_items_are_really_absent():
    """Reported as NOT IMPLEMENTED in the audit: PEG, ROIC momentum, margin acceleration, composite 0-100 score."""
    snap, _ = golden_snapshot("A", GOLDEN["A"]["years"], AS_OF)
    names = set(snap["features"])
    assert not any("peg" in n for n in names)
    assert not {"roic_change", "roic_acceleration", "net_margin_change", "fcf_margin_change", "gross_margin_acceleration", "dividend_yield"} & names
    assert snap["score"] is None and snap["confidence"] is None
    assert not any(f in snap["flags"] for f in ("HIGH_DEBT", "DISTRESS", "ESTIMATE_CUTS"))
