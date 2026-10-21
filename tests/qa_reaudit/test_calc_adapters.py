"""Track C: valuation edge cases and fuzz, sector adapters, Altman Z, own-history percentile, ETF aggregation.

Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_calc_adapters.py
"""
import datetime as dt
import math
import random
from decimal import Decimal

import pytest

from fundamentals.adapters import altman_manufacturing_z, own_history_percentile, sector_features
from fundamentals.features import FundamentalFeatureEngine, valuation_features
from fundamentals.store import Fact, PointInTimeFundamentalStore

from tests.qa_reaudit import calc_oracle as O
from tests.qa_reaudit.calc_fixtures import annual_facts, classification, golden_snapshot, new_store, snapshot
from tests.qa_reaudit.calc_golden import GOLDEN

AS_OF = "2024-06-01T00:00:00Z"
KW = dict(price_currency="GBP", statement_currency="GBP", price_basis="B", share_basis="B")
RAW = {"net_income": 1e6, "ocf": 2e6, "capex": 1e6, "cash": 1e6, "short_debt": 1e6, "long_debt": 1e6, "preferred_equity": 0.0, "minority_interest": 0.0,
       "revenue": 1e7, "equity": 5e6, "operating_income": 2e6, "depreciation": 1e5}


# ---------------------------------------------------------------------------- valuation scale / fuzz
@pytest.mark.parametrize("label,price,shares,scale", [
    ("trillion-cap", 190.0, 15.5e9, 1e11), ("micro-cap", 0.35, 1.4e6, 1e3), ("fractional eps", 0.0123, 7.5e8, 1e6), ("huge price", 6.8e5, 3e6, 1e9),
])
def test_valuation_at_extreme_scales_matches_oracle(label, price, shares, scale):
    v = {k: val * scale / 1e6 for k, val in RAW.items()}
    got = valuation_features(price, shares, v, forward_eps=price / 17.0, **KW)
    want = O.valuation(Decimal(price), Decimal(shares), {k: Decimal(x) for k, x in v.items()}, forward_eps=Decimal(price / 17.0))
    for k, w in want.items():
        assert O.rel_close(got[k], w, 1e-12, 0), (label, k, got[k], w)


@pytest.mark.parametrize("seed", range(150), ids=lambda s: f"seed{s}")
def test_random_valuations_match_oracle_including_negative_and_zero_inputs(seed):
    rng = random.Random(seed)
    def pick(lo, hi, p_neg=0.2, p_zero=0.08, p_none=0.1):
        r = rng.random()
        if r < p_none:
            return None
        if r < p_none + p_zero:
            return 0.0
        x = rng.uniform(lo, hi)
        return -x if rng.random() < p_neg else x
    v = {"net_income": pick(1e5, 5e9), "ocf": pick(1e5, 5e9), "capex": abs(pick(1e4, 1e9, 0, 0.1) or 0) if rng.random() > 0.1 else None,
         "cash": pick(1e5, 5e9, 0), "short_debt": pick(1e5, 5e9, 0), "long_debt": pick(1e5, 5e9, 0), "preferred_equity": rng.choice([0.0, 0.0, 1e6, None]),
         "minority_interest": rng.choice([0.0, 0.0, 5e5, None]), "revenue": pick(1e6, 5e10, 0.05), "equity": pick(1e6, 1e10, 0.2),
         "operating_income": pick(1e5, 5e9), "depreciation": pick(1e4, 1e9, 0)}
    v = {k: x for k, x in v.items() if x is not None or rng.random() < 0.5}
    price, shares = rng.uniform(0.01, 5000), rng.uniform(1e5, 5e9)
    fwd = rng.choice([None, rng.uniform(0.01, 100), -1.0, 0.0])
    got = valuation_features(price, shares, v, forward_eps=fwd, **KW)
    want = O.valuation(Decimal(price), Decimal(shares), {k: (None if x is None else Decimal(x)) for k, x in v.items()}, forward_eps=fwd)
    for k, w in want.items():
        assert O.rel_close(got[k], w, 1e-11, 1e-6 if k in ("enterprise_value", "market_cap") else 0), (seed, k, got[k], w)
        assert got[k] is None or math.isfinite(got[k])


def test_valuation_guards_fail_loudly_for_invalid_market_cap_inputs():
    for price, shares in ((0.0, 1e6), (-1.0, 1e6), (20.0, 0.0), (20.0, -5.0), (float("nan"), 1e6), (20.0, float("inf"))):
        with pytest.raises(ValueError):
            valuation_features(price, shares, RAW, **KW)
    with pytest.raises(ValueError, match="overflow"):
        valuation_features(1e308, 1e308, RAW, **KW)
    with pytest.raises(ValueError, match="CapEx"):
        valuation_features(20.0, 1e6, {**RAW, "capex": -1.0}, **KW)
    with pytest.raises(ValueError, match="currency"):
        valuation_features(20.0, 1e6, RAW, **{**KW, "price_currency": "USD"})
    for pb, sb in (("", ""), (None, None), ("A", "B")):
        with pytest.raises(ValueError):
            valuation_features(20.0, 1e6, RAW, **{**KW, "price_basis": pb, "share_basis": sb})


def test_QA_CALC_04_valuation_features_must_validate_numeric_inputs():
    failures = []
    def must_raise_value_error(label, fn):
        try:
            fn()
        except ValueError:
            return
        except Exception as e:   # TypeError / OverflowError leak
            failures.append(f"{label}: leaked {type(e).__name__}")
            return
        failures.append(f"{label}: silently accepted")
    must_raise_value_error("bool price", lambda: valuation_features(True, 1e6, RAW, **KW))
    must_raise_value_error("None price", lambda: valuation_features(None, 1e6, RAW, **KW))
    must_raise_value_error("inf equity", lambda: valuation_features(20.0, 1e6, {**RAW, "equity": float("inf")}, **KW))
    must_raise_value_error("str net income", lambda: valuation_features(20.0, 1e6, {**RAW, "net_income": "abc"}, **KW))
    must_raise_value_error("bool cash", lambda: valuation_features(20.0, 1e6, {**RAW, "cash": True}, **KW))
    assert not failures, failures


def test_valuation_missing_inputs_are_none_never_zero():
    got = valuation_features(20.0, 1e6, {"revenue": 1e7}, **KW)
    assert got["market_cap"] == 20e6 and got["ps"] == pytest.approx(2.0)
    for k in ("pe", "pb", "price_fcf", "fcf_yield", "earnings_yield", "ev_revenue", "ev_ebitda", "enterprise_value", "forward_pe"):
        assert got[k] is None, k


# ---------------------------------------------------------------------------- sector adapters
def test_bank_adapter_hand_values():
    out = sector_features("banks", dict(net_interest_income=30, earning_assets=1000, cet1_capital=12, risk_weighted_assets=100, nonperforming_loans=5, loans=200, provisions=1),
                          dict(loans=160))
    assert out["net_interest_margin"] == pytest.approx(0.03) and out["cet1_ratio"] == pytest.approx(0.12)
    assert out["npl_ratio"] == pytest.approx(0.025) and out["loan_growth"] == pytest.approx(0.25)     # fractions, not percent
    z = sector_features("banks", dict(net_interest_income=30, earning_assets=0, cet1_capital=12, risk_weighted_assets=0, nonperforming_loans=5, loans=0), dict(loans=0))
    assert z["net_interest_margin"] is None and z["cet1_ratio"] is None and z["npl_ratio"] is None and z["loan_growth"] is None
    assert sector_features("banks", {})["cet1_ratio"] is None


def test_saas_rule_of_40_is_a_fraction_sum():
    out = sector_features("saas", dict(arr=130, revenue=130, ocf=30, capex=10, ltv=300, cac=100), dict(arr=100, revenue=100))
    assert out["arr_growth"] == pytest.approx(0.30) and out["ltv_cac"] == pytest.approx(3.0)
    assert out["rule_of_40"] == pytest.approx(0.30 + 20 / 130, rel=1e-12)          # 0.4538 == 45.38 percent (threshold in fraction units: 0.40)
    assert sector_features("saas", dict(arr=130, revenue=130, ocf=30), dict(arr=100, revenue=100))["rule_of_40"] is None    # capex missing -> None
    assert sector_features("saas", dict(arr=130, revenue=130, ocf=30, capex=10), dict(arr=0, revenue=0))["rule_of_40"] is None
    assert sector_features("saas", dict(ltv=1, cac=0))["ltv_cac"] is None


def test_retail_semis_energy_adapters():
    assert sector_features("retail", dict(store_count=110), dict(store_count=100))["store_growth"] == pytest.approx(0.1)
    assert sector_features("semiconductors", dict(inventory=90), dict(inventory=100))["inventory_growth"] == pytest.approx(-0.1)
    e = sector_features("energy", dict(production=120, realized_price=80, production_cost_per_unit=25), dict(production=100))
    assert e["production_growth"] == pytest.approx(0.2) and e["unit_margin"] == pytest.approx(55)
    assert sector_features("energy", dict(realized_price=80))["unit_margin"] is None


@pytest.mark.parametrize("name", ["Banks", "bank", "BANKS", "crypto", "", None, "Financials"])
def test_unknown_sector_label_is_refused_by_the_adapter(name):
    with pytest.raises(ValueError):
        sector_features(name, {})


def test_ratio_gating_label_is_case_insensitive():
    """Lead re-audit: the sector label is trimmed and lower-cased, so a vendor label like 'Financials' / ' Banks ' gates bank/insurer
    ratios exactly like 'financials' / 'banks'. Only labels that are not in the gate list ('bank') still compute them."""
    g = GOLDEN["K"]
    for label, gated in (("banks", True), ("financials", True), ("insurance", True), ("Financials", True), ("Banks", True), (" BANKS ", True), ("bank", False)):
        snap, _ = golden_snapshot("K", g["years"], AS_OF, label)
        assert (snap["features"]["debt_equity"] is None) is gated, label


def test_altman_z_hand_value_and_guards():
    v = dict(assets=100, liabilities=60, current_assets=50, current_liabilities=30, retained_earnings=30, operating_income=15, revenue=110)
    # 1.2*0.2 + 1.4*0.3 + 3.3*0.15 + 0.6*(120/60) + 1.0*1.1 = 0.24 + 0.42 + 0.495 + 1.2 + 1.1
    assert altman_manufacturing_z(v, 120, "manufacturing") == pytest.approx(3.455, rel=1e-12)
    assert altman_manufacturing_z(v, 120, "software") is None and altman_manufacturing_z(v, 120, None) is None
    assert altman_manufacturing_z({**v, "assets": 0}, 120, "manufacturing") is None
    assert altman_manufacturing_z({**v, "liabilities": 0}, 120, "manufacturing") is None
    assert altman_manufacturing_z({k: x for k, x in v.items() if k != "retained_earnings"}, 120, "manufacturing") is None
    assert altman_manufacturing_z(v, None, "manufacturing") is None
    # distress zone arithmetic: negative working capital and losses reduce Z below 1.81
    bad = dict(assets=100, liabilities=90, current_assets=20, current_liabilities=60, retained_earnings=-20, operating_income=-5, revenue=60)
    assert altman_manufacturing_z(bad, 10, "manufacturing") == pytest.approx(1.2 * -0.4 + 1.4 * -0.2 + 3.3 * -0.05 + 0.6 * 10 / 90 + 0.6, rel=1e-12)


# ---------------------------------------------------------------------------- own-history percentile
def instants(values, end0=dt.date(2021, 3, 31)):
    out = []
    for i, v in enumerate(values):
        end = (end0 + dt.timedelta(days=91 * i))
        pub = (end + dt.timedelta(days=30)).isoformat() + "T00:00:00Z"
        out.append(Fact("H", "cash", v, end.isoformat(), pub, pub, "x", f"r{i}", "GBP", "GBP"))
    return out


def test_own_history_percentile_hand_values_for_instants():
    s = PointInTimeFundamentalStore()
    s.ingest(instants([1e6 * k for k in range(1, 9)]))           # 8 quarter-end cash balances 1..8 million
    t = "2023-06-30T00:00:00Z"
    assert own_history_percentile(s, "H", "cash", 4.5e6, t, 3) == pytest.approx(4 / 8)
    assert own_history_percentile(s, "H", "cash", 4e6, t, 3) == pytest.approx((3 + 0.5) / 8)      # ties count half
    assert own_history_percentile(s, "H", "cash", 0.1e6, t, 3) == 0.0 and own_history_percentile(s, "H", "cash", 9e6, t, 3) == 1.0
    assert own_history_percentile(s, "H", "cash", None, t, 3) is None
    with pytest.raises(ValueError):
        own_history_percentile(s, "H", "cash", 1.0, t, 4)


def test_own_history_percentile_needs_eight_observations_and_one_unit():
    s = PointInTimeFundamentalStore()
    s.ingest(instants([1e6 * k for k in range(1, 8)]))
    assert own_history_percentile(s, "H", "cash", 4e6, "2023-06-30T00:00:00Z", 3) is None        # 7 < 8
    s2 = PointInTimeFundamentalStore()
    s2.ingest(instants([1e6 * k for k in range(1, 8)]) + [Fact("H", "cash", 1.0, "2022-06-30", "2022-07-30T00:00:00Z", "2022-07-30T00:00:00Z", "other", "z", "USD", "USD")])
    with pytest.raises(ValueError, match="mixes"):
        own_history_percentile(s2, "H", "cash", 4e6, "2023-06-30T00:00:00Z", 3)


def test_own_history_percentile_ignores_unpublished_future_rows():
    s = PointInTimeFundamentalStore()
    s.ingest(instants([1e6 * k for k in range(1, 9)]) + instants([1e12], end0=dt.date(2023, 5, 1)))
    t = "2023-06-30T00:00:00Z"            # the 1e12 row is published 2023-05-31 ... so move it later to test exclusion
    s2 = PointInTimeFundamentalStore()
    s2.ingest(instants([1e6 * k for k in range(1, 9)]))
    s2.ingest([Fact("H", "cash", 1e12, "2023-06-01", "2023-08-15T00:00:00Z", "2023-08-15T00:00:00Z", "x", "future", "GBP", "GBP")])
    assert own_history_percentile(s2, "H", "cash", 4.5e6, t, 3) == pytest.approx(4 / 8)   # period_end <= as_of but not yet published


def test_QA_CALC_05_flow_percentile_must_compare_like_durations():
    s = PointInTimeFundamentalStore()
    facts = []
    ends = [("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30"), ("10-01", "12-31")]
    for y, level in ((2021, 100), (2022, 105), (2023, 110)):
        cum = 0
        for k, (a, b) in enumerate(ends):
            cum += level
            pub = f"{y + 1}-02-20T00:00:00Z" if k == 3 else f"{y}-{['05', '08', '11'][k]}-05T00:00:00Z"
            facts.append(Fact("Z", "revenue", level * 1e6, f"{y}-{b}", pub, pub, "x", f"q{y}{k}", "GBP", "GBP", f"{y}-{a}"))
            if k >= 1:   # a 10-Q also prints the year-to-date row
                facts.append(Fact("Z", "revenue", cum * 1e6, f"{y}-{b}", pub, pub, "x", f"y{y}{k}", "GBP", "GBP", f"{y}-01-01"))
    s.ingest(facts)
    # a 112m quarter exceeds all twelve past 3-month quarters (100..110): own-history percentile must be 1.0
    got = own_history_percentile(s, "Z", "revenue", 112e6, "2024-03-01T00:00:00Z", 3)
    assert got == pytest.approx(1.0), got


# ---------------------------------------------------------------------------- ETF aggregation
def etf_store(weights, extra_constituents=(), holdings_end="2024-03-31", pub="2024-04-20T00:00:00Z"):
    facts = []
    gm = {"AAA": ("100", "50"), "BBB": ("100", "20"), "CCC": ("100", "35")}
    for sym, (rev, gp) in gm.items():
        facts += annual_facts(sym, {2022: {"revenue": rev, "gross_profit": gp}, 2023: {"revenue": rev, "gross_profit": gp}})
    for sym, w in weights.items():
        facts.append(Fact("FUND", "weight:" + sym, w, holdings_end, pub, pub, "issuer", "h1", "", "fraction", kind="holding", metadata={"constituent": sym}))
    return new_store(facts)


def test_etf_disclosed_weight_average_is_exact_and_not_reweighted():
    s = etf_store({"AAA": 0.6, "BBB": 0.4})
    r = FundamentalFeatureEngine(s).etf_snapshot("FUND", AS_OF, "gross_margin")
    assert r["status"] == "AVAILABLE" and r["value"] == pytest.approx(0.6 * 0.5 + 0.4 * 0.2, rel=1e-12) and r["coverage"] == pytest.approx(1.0)
    # a constituent without data: the aggregate is withheld (no re-weighting of survivors to 100%)
    s2 = etf_store({"AAA": 0.6, "ZZZ": 0.4})
    r2 = FundamentalFeatureEngine(s2).etf_snapshot("FUND", AS_OF, "gross_margin")
    assert r2["value"] is None and r2["status"] == "INCOMPLETE_HOLDINGS_COVERAGE" and r2["coverage"] == pytest.approx(0.6)
    # disclosed weights summing to less than 99.9 percent (top-N list) are also incomplete, never scaled up
    s3 = etf_store({"AAA": 0.5, "BBB": 0.4})
    assert FundamentalFeatureEngine(s3).etf_snapshot("FUND", AS_OF, "gross_margin")["value"] is None


def test_etf_holdings_guards():
    eng = lambda s: FundamentalFeatureEngine(s)
    assert eng(etf_store({"AAA": 0.6, "BBB": 0.4})).etf_snapshot("FUND", "2024-04-19T23:59:59Z", "gross_margin")["status"] == "MISSING_DATED_HOLDINGS"   # not yet published
    assert eng(etf_store({"AAA": 0.6, "BBB": 0.4})).etf_snapshot("FUND", "2024-12-01T00:00:00Z", "gross_margin")["status"] == "STALE_HOLDINGS"          # > 180 days
    with pytest.raises(ValueError):
        eng(etf_store({"AAA": 0.7, "BBB": 0.4})).etf_snapshot("FUND", AS_OF, "gross_margin")         # sums to 1.1
    with pytest.raises(ValueError):
        eng(etf_store({"AAA": 1.2, "BBB": -0.2})).etf_snapshot("FUND", AS_OF, "gross_margin")        # short weight
    assert eng(etf_store({"AAA": 0.6, "BBB": 0.4005})).etf_snapshot("FUND", AS_OF, "gross_margin")["status"] == "AVAILABLE"   # 1.0005 tolerated (<=1.001)
