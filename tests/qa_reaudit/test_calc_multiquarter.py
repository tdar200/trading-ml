"""Track C phase 135 (permanent multi-quarter golden company) and phases 148-153 (value trap, growth trap,
quality, deterioration, improvement fixtures), plus flag-trigger exactness and split/dilution adapters.

Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_calc_multiquarter.py
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction

import pytest

from fundamentals.adapters import apply_known_splits
from fundamentals.features import FundamentalFeatureEngine, valuation_features
from fundamentals.models import CombinedSignalEngine, FundamentalScoreEngine, ModelArtifact
from fundamentals.store import Fact, PointInTimeFundamentalStore

from tests.qa_reaudit import calc_oracle as O
from tests.qa_reaudit.calc_cmp import compare, oracle_from_years
from tests.qa_reaudit.calc_fixtures import annual_facts, base_dict, classification, golden_snapshot, new_store, snapshot
from tests.qa_reaudit.calc_golden import R, Y

M = 1_000_000
AS_OF = "2024-06-01T00:00:00Z"

# =====================================================================================================
# Phase 135: ONE permanent multi-quarter golden company "MQ" (GBP millions, calendar fiscal year)
# Q1'24 strong growth | Q2'24 margin decline | Q3'24 debt increase | Q4'24 earnings miss | Q1'25 recovery
# =====================================================================================================
Q = {  # discrete quarter values: revenue, operating income, net income, ocf, capex, eps
    "23Q1": (100, 20, 15, 18, 4, "0.15"), "23Q2": (105, 21, 16, 19, 4, "0.16"), "23Q3": (110, 22, 17, 20, 4, "0.17"), "23Q4": (115, 23, 18, 21, 4, "0.18"),
    "24Q1": (140, 30, 24, 28, 6, "0.24"), "24Q2": (150, 24, 18, 20, 6, "0.18"), "24Q3": (155, 25, 19, 18, 10, "0.19"), "24Q4": (150, 15, 8, 10, 6, "0.08"),
    "25Q1": (165, 32, 25, 30, 5, "0.25"),
}
QDATES = {"23Q1": ("2023-01-01", "2023-03-31"), "23Q2": ("2023-04-01", "2023-06-30"), "23Q3": ("2023-07-01", "2023-09-30"), "23Q4": ("2023-10-01", "2023-12-31"),
          "24Q1": ("2024-01-01", "2024-03-31"), "24Q2": ("2024-04-01", "2024-06-30"), "24Q3": ("2024-07-01", "2024-09-30"), "24Q4": ("2024-10-01", "2024-12-31"),
          "25Q1": ("2025-01-01", "2025-03-31")}
# acceptance time of each filing; available_at = acceptance + 5 minutes (SEC rule used by providers.sec_facts)
ACCEPT = {"23Q1": "2023-05-04T20:15:00Z", "23Q2": "2023-08-03T20:15:00Z", "23Q3": "2023-11-02T20:15:00Z", "23Q4": "2024-02-22T21:00:00Z",
          "24Q1": "2024-05-02T20:15:00Z", "24Q2": "2024-08-01T20:15:00Z", "24Q3": "2024-10-31T20:15:00Z", "24Q4": "2025-02-20T21:00:00Z",
          "25Q1": "2025-05-01T20:15:00Z"}
ANNUAL = {
    2021: Y(340, 136, 60, 45, 60, 14, 35, 20, 85, 130, 80, 280, 560, 100 * M, "0.45", receivables=40),
    2022: Y(380, 152, 70, 52, 70, 15, 40, 20, 90, 140, 85, 300, 600, 100 * M, "0.52", receivables=45),
    2023: Y(430, "176.3", 86, 66, 78, 16, 50, 20, 100, 150, 90, 340, 650, 100 * M, "0.66", receivables=50),
    2024: Y(595, "226.1", 94, 69, 76, 28, 30, 20, 160, 160, 120, 360, 800, 104 * M, "0.69", receivables=75),
}
ANNUAL_PUB = {2021: "2022-02-24T21:00:00Z", 2022: "2023-02-23T21:00:00Z"}


def avail(accept):
    return (datetime.fromisoformat(accept.replace("Z", "+00:00")) + timedelta(minutes=5)).isoformat()


def mq_fact(metric, value, start, end, accept, rev, basis="one"):
    cur = "" if metric == "diluted_shares" else "GBP"
    units = "shares" if metric == "diluted_shares" else "GBP/shares" if metric == "eps" else "GBP millions"
    return Fact("MQ", metric, float(Decimal(str(value))), end, accept, avail(accept), "fixture", rev, cur, units,
                start if metric not in ("cash", "short_debt", "long_debt", "equity", "assets", "current_assets", "current_liabilities", "receivables") else "",
                metadata={"share_basis": basis, "accounting_standard": "us-gaap"})


def mq_facts(through=None):
    out = []
    for y in (2021, 2022):
        for m, v in ANNUAL[y].items():
            out.append(mq_fact(m, v, f"{y}-01-01", f"{y}-12-31", ANNUAL_PUB[y], f"FY{y}"))
    for fy in (2023, 2024):
        ytd = {"revenue": 0, "operating_income": 0, "net_income": 0, "ocf": 0, "capex": 0}
        for n in (1, 2, 3, 4):
            key = f"{str(fy)[2:]}Q{n}"
            rev, oi, ni, ocf, capex, eps = Q[key]
            vals = dict(revenue=rev, operating_income=oi, net_income=ni, ocf=ocf, capex=capex)
            for k in ytd:
                ytd[k] += vals[k]
            s, e = QDATES[key]
            fy_start = f"{fy}-01-01"
            acc = ACCEPT[key]
            tag = f"{'10K' if n == 4 else '10Q'}-{key}"
            if n < 4:
                for m in ("revenue", "operating_income", "net_income"):
                    out.append(mq_fact(m, vals[m], s, e, acc, tag))
                    if n >= 2:
                        out.append(mq_fact(m, ytd[m], fy_start, e, acc, tag))
                out.append(mq_fact("eps", eps, s, e, acc, tag))
                for m in ("ocf", "capex"):
                    out.append(mq_fact(m, ytd[m], fy_start, e, acc, tag))      # cash flows exist only as YTD
                # quarter-end balance sheet (not used by the annual snapshot)
                debt_up = {"24Q3": 140}.get(key)
                if debt_up:
                    out.append(mq_fact("long_debt", debt_up, "", e, acc, tag))
            else:
                for m, v in ANNUAL[fy].items():
                    out.append(mq_fact(m, v, fy_start if m not in ("cash", "short_debt", "long_debt", "equity", "assets", "current_assets", "current_liabilities", "receivables") else "", e, acc, tag))
                out.append(mq_fact("eps", eps, s, e, acc, tag))                   # the 10-K quarterly data note prints Q4 EPS
    s, e = QDATES["25Q1"]
    rev, oi, ni, ocf, capex, eps = Q["25Q1"]
    for m, v in dict(revenue=rev, operating_income=oi, net_income=ni).items():
        out.append(mq_fact(m, v, s, e, ACCEPT["25Q1"], "10Q-25Q1"))
    out.append(mq_fact("eps", eps, s, e, ACCEPT["25Q1"], "10Q-25Q1"))
    out.append(mq_fact("ocf", ocf, s, e, ACCEPT["25Q1"], "10Q-25Q1"))
    out.append(mq_fact("capex", capex, s, e, ACCEPT["25Q1"], "10Q-25Q1"))
    return out


# hand-derived expectations at each publication boundary (paper arithmetic, Fractions)
HAND = {
    "B0": dict(label="10-K FY23 (Q4'23)", key="23Q4", rev_qoq=Fraction(115, 110) - 1, oi_qoq=Fraction(23, 22) - 1, ocf_qoq=Fraction(21, 20) - 1,
               ttm=dict(revenue=430, net_income=66, ocf=78, capex=16, eps="0.66", fcf=62), annual_fy=2023, flags=[], decision="NO_TRADE"),
    "B1": dict(label="Q1'24 strong growth", key="24Q1", rev_qoq=Fraction(140, 115) - 1, oi_qoq=Fraction(30, 23) - 1, ocf_qoq=Fraction(28, 21) - 1,
               ttm=dict(revenue=470, net_income=75, ocf=88, capex=18, eps="0.75", fcf=70), annual_fy=2023, flags=[], decision="RESEARCH_CANDIDATE"),
    "B2": dict(label="Q2'24 margin decline", key="24Q2", rev_qoq=Fraction(150, 140) - 1, oi_qoq=Fraction(24, 30) - 1, ocf_qoq=Fraction(20, 28) - 1,
               ttm=dict(revenue=515, net_income=77, ocf=89, capex=20, eps="0.77", fcf=69), annual_fy=2023, flags=[], decision="NO_TRADE"),
    "B3": dict(label="Q3'24 debt increase", key="24Q3", rev_qoq=Fraction(155, 150) - 1, oi_qoq=Fraction(25, 24) - 1, ocf_qoq=Fraction(18, 20) - 1,
               ttm=dict(revenue=560, net_income=79, ocf=87, capex=26, eps="0.79", fcf=61), annual_fy=2023, flags=[], decision="NO_TRADE"),
    "B4": dict(label="10-K FY24 (Q4'24 earnings miss)", key="24Q4", rev_qoq=Fraction(150, 155) - 1, oi_qoq=Fraction(15, 25) - 1, ocf_qoq=Fraction(10, 18) - 1,
               ttm=dict(revenue=595, net_income=69, ocf=76, capex=28, eps="0.69", fcf=48), annual_fy=2024,
               flags=["DILUTION", "FALLING_MARGINS", "RECEIVABLES_OUTPACE_SALES"], decision="NO_TRADE"),
    "B5": dict(label="Q1'25 recovery", key="25Q1", rev_qoq=Fraction(165, 150) - 1, oi_qoq=Fraction(32, 15) - 1, ocf_qoq=Fraction(30, 10) - 1,
               ttm=dict(revenue=620, net_income=70, ocf=78, capex=27, eps="0.70", fcf=51), annual_fy=2024,
               flags=["DILUTION", "FALLING_MARGINS", "RECEIVABLES_OUTPACE_SALES"], decision="RESEARCH_CANDIDATE"),
}


def mq_model():
    # score = 0.02*(rq-0.05)/0.1 + 0.01*(oq-0.05)/0.2 + 0.01*(cq-0.05)/0.2  (+ 0 * technical)
    return ModelArtifact(("revenue_qoq", "operating_income_qoq", "ocf_qoq", "technical_signal"), (0.05, 0.05, 0.05, 0.5), (0.1, 0.2, 0.2, 0.5),
                         (0.02, 0.01, 0.01, 0.0), 0.0, "2023-01-01T00:00:00Z", 1.0, 100, "hand", 5)


def hand_score(h):
    f = lambda x: Fraction(x)
    return float(Fraction(2, 100) * (h["rev_qoq"] - Fraction(1, 20)) / Fraction(1, 10) + Fraction(1, 100) * (h["oi_qoq"] - Fraction(1, 20)) / Fraction(1, 5)
                 + Fraction(1, 100) * (h["ocf_qoq"] - Fraction(1, 20)) / Fraction(1, 5))


def boundary_time(h):
    return avail(ACCEPT[h["key"]])


@pytest.fixture(scope="module")
def mq_store():
    return new_store(mq_facts())


@pytest.mark.parametrize("bname", sorted(HAND))
def test_phase135_each_publication_boundary(mq_store, bname):
    h = HAND[bname]
    T = boundary_time(h)
    snap = FundamentalFeatureEngine(mq_store).snapshot("MQ", T)
    assert snap["status"] == "AVAILABLE" and snap["period_end"] == f"{h['annual_fy']}-12-31", (bname, snap["period_end"])
    f = snap["features"]
    assert f["revenue_qoq"] == pytest.approx(float(h["rev_qoq"]), rel=1e-12), bname
    assert f["operating_income_qoq"] == pytest.approx(float(h["oi_qoq"]), rel=1e-12), bname
    assert f["ocf_qoq"] == pytest.approx(float(h["ocf_qoq"]), rel=1e-12), bname
    ttm = snap["ttm"]
    assert ttm["status"] == "AVAILABLE", (bname, ttm["reasons"], ttm["values"])
    for m, v in h["ttm"].items():
        want = float(Decimal(str(v))) * (1 if m == "eps" else M)
        assert ttm["values"][m] == pytest.approx(want, rel=1e-12), (bname, m, ttm["values"][m], want)
    assert sorted(snap["flags"]) == sorted(h["flags"]), (bname, snap["flags"])
    # the annual block reflects the latest 10-K only
    exp_feats, exp_flags, _ = oracle_from_years({y: dict(v) for y, v in ANNUAL.items()}, None, current=h["annual_fy"])
    assert f["revenue_yoy"] == pytest.approx(float(exp_feats["revenue_yoy"]), rel=1e-12)


@pytest.mark.parametrize("bname", sorted(HAND))
def test_phase135_decisions_and_boundary_exclusion(mq_store, bname):
    h = HAND[bname]
    T = boundary_time(h)
    eng = CombinedSignalEngine(FundamentalFeatureEngine(mq_store), FundamentalScoreEngine(mq_model()), policy="filter", threshold=0.0, research_shadow=True)
    d = eng.decide("MQ", T, {"signal": True})
    assert d["fundamental"]["score"] == pytest.approx(hand_score(h), abs=1e-13), (bname, d["fundamental"]["score"], hand_score(h))
    assert d["decision"] == h["decision"], (bname, d["decision"], d["fundamental"]["score"])
    assert d["permit_signal"] == (h["decision"] == "RESEARCH_CANDIDATE")
    # one second before the filing becomes available the previous boundary's numbers apply (nothing leaks early)
    before = datetime.fromisoformat(T.replace("Z", "+00:00")) - timedelta(seconds=1)
    snap_before = FundamentalFeatureEngine(mq_store).snapshot("MQ", before)
    snap_at = FundamentalFeatureEngine(mq_store).snapshot("MQ", T)
    assert snap_before["ttm"]["values"] != snap_at["ttm"]["values"] or bname == "B0"
    names = sorted(HAND)
    if bname != "B0":
        prev = HAND[names[names.index(bname) - 1]]
        assert snap_before["features"]["revenue_qoq"] == pytest.approx(float(prev["rev_qoq"]), rel=1e-12), bname
        assert snap_before["ttm"]["values"]["revenue"] == pytest.approx(prev["ttm"]["revenue"] * M, rel=1e-12), bname


def test_phase135_quarter_balance_sheet_debt_is_not_visible_until_fiscal_year_end(mq_store):
    """Limitation pinned: the Q3'24 debt increase (long_debt 100->140 at 2024-09-30) is not in any feature until the 10-K."""
    s3 = FundamentalFeatureEngine(mq_store).snapshot("MQ", boundary_time(HAND["B3"]))
    s4 = FundamentalFeatureEngine(mq_store).snapshot("MQ", boundary_time(HAND["B4"]))
    assert s3["features"]["debt_change"] == pytest.approx(10 * M)       # FY23 vs FY22: (20+100) - (20+90)
    assert s4["features"]["debt_change"] == pytest.approx(60 * M)       # FY24 vs FY23: (20+160) - (20+100)


def test_phase135_no_filing_after_T_changes_the_decision(mq_store):
    """Phase 98 in miniature: a store holding only rows available <= T gives the identical decision."""
    for bname in ("B1", "B3", "B5"):
        T = boundary_time(HAND[bname])
        trunc = PointInTimeFundamentalStore()
        trunc.ingest([f for f in mq_facts() if f.available_at <= T])
        mk = lambda st: CombinedSignalEngine(FundamentalFeatureEngine(st), FundamentalScoreEngine(mq_model()), policy="filter", threshold=0.0, research_shadow=True)
        a = mk(mq_store).decide("MQ", T, {"signal": True})
        b = mk(trunc).decide("MQ", T, {"signal": True})
        assert a == b, bname


# =====================================================================================================
# Phases 148-153: value trap, growth trap, quality, deterioration, improvement
# =====================================================================================================
SCEN = {}
SCEN["value_trap"] = dict(
    price="3", shares=100 * M, forward="0.3",
    years={2018: R(560), 2020: R(520),
           2021: Y(500, 200, 80, 60, 70, 20, 30, 10, 100, 150, 90, 250, 500, 100 * M, "0.6", depreciation=20, interest_expense=8, pretax_income=75, tax_expense=15, receivables=60, inventory=50, payables=40, cogs=300),
           2022: Y(470, "178.6", 60, 45, 40, 25, 25, 20, 120, 140, 100, 245, 520, 100 * M, "0.45", depreciation=20, interest_expense=9, pretax_income=56, tax_expense=11, receivables=62, inventory=55, payables=40, cogs="291.4"),
           2023: Y(430, "150.5", 40, 40, 15, 30, 20, 40, 140, 130, 110, 240, 540, 100 * M, "0.40", depreciation=20, interest_expense=12, pretax_income=34, tax_expense=-6, receivables=70, inventory=60, payables=38, cogs="279.5")},
    hand={"fcf": "-15000000", "gross_margin": "0.35", "revenue_yoy": "-0.08510638297872340426", "debt": "180000000", "debt_equity": "0.75",
          "net_margin": "0.09302325581395348837"},
    hand_flags=["NEGATIVE_FCF", "FALLING_MARGINS", "REVENUE_DECELERATION", "RECEIVABLES_OUTPACE_SALES"],
    hand_val={"pe": "7.5", "pb": "1.25", "ps": "0.69767441860465116279", "price_fcf": None, "fcf_yield": "-0.05", "earnings_yield": "0.13333333333333333333"},
)
SCEN["growth_trap"] = dict(
    price="20", shares=170 * M, forward="-0.2",
    years={2018: R(20), 2020: R(60),
           2021: Y(100, 60, -5, -10, -8, 10, 300, 0, 0, 320, 40, 300, 400, 100 * M, "-0.1", depreciation=5, interest_expense=0, pretax_income=-10, tax_expense=0, receivables=20, inventory=5, payables=10, cogs=40),
           2022: Y(160, 96, -10, -18, -20, 15, 220, 0, 0, 250, 60, 330, 420, 130 * M, "-0.1385", depreciation=8, interest_expense=0, pretax_income=-18, tax_expense=0, receivables=40, inventory=10, payables=15, cogs=64),
           2023: Y(250, 150, -20, -35, -40, 25, 150, 0, 0, 200, 90, 420, 520, 170 * M, "-0.2059", depreciation=12, interest_expense=0, pretax_income=-35, tax_expense=0, receivables=100, inventory=25, payables=30, cogs=100)},
    hand={"fcf": "-65000000", "revenue_yoy": "0.5625", "revenue_acceleration": "-0.0375", "gross_margin": "0.6", "gross_margin_change": "0",
          "diluted_shares_yoy": "0.30769230769230769231", "receivables_yoy": None, "roic": None, "pe": None},
    hand_flags=["NEGATIVE_FCF", "REVENUE_DECELERATION", "DILUTION", "RECEIVABLES_OUTPACE_SALES"],
    hand_val={"pe": None, "ps": "13.6", "market_cap": "3400000000", "ev_ebitda": None, "fcf_yield": "-0.01911764705882352941"},
)
SCEN["quality"] = dict(
    price="40", shares=98 * M, forward="4.5",
    years={2018: R(700), 2020: R(900),
           2021: Y(1000, 550, 250, 200, 260, 40, 300, 0, 50, 500, 150, 900, 1500, 100 * M, "2.0", depreciation=40, interest_expense=3, pretax_income=247, tax_expense="49.4", receivables=100, inventory=60, payables=50, cogs=450),
           2022: Y(1080, 604.8, 275, 220, 285, 42, 380, 0, 50, 600, 160, 1000, 1650, 100 * M, "2.2", depreciation=42, interest_expense=3, pretax_income=272, tax_expense="54.4", receivables=106, inventory=62, payables=54, cogs="475.2"),
           2023: Y(1170, "666.9", 310, 248, 320, 45, 480, 0, 50, 720, 170, 1130, 1800, 98 * M, "2.53", depreciation=45, interest_expense=3, pretax_income=307, tax_expense="59.0", receivables=111, inventory=64, payables=58, cogs="503.1")},
    hand={"fcf": "275000000", "gross_margin": "0.57", "gross_margin_change": "0.01", "diluted_shares_yoy": "-0.02", "net_debt": "-430000000",
          "current_ratio": "4.23529411764705882353", "revenue_yoy": "0.08333333333333333333", "ocf_net_income": "1.29032258064516129032"},
    hand_flags=[],
    hand_val={},
)
SCEN["deterioration"] = dict(
    price="10", shares=105 * M, forward="0.1",
    years={2018: R(900), 2020: R(950),
           2021: Y(1000, 500, 150, 100, 120, 50, 150, 20, 150, 400, 200, 600, 1200, 100 * M, "1.0", depreciation=40, interest_expense=15, pretax_income=135, tax_expense=27, receivables=100, inventory=100, payables=80, cogs=500),
           2022: Y(1050, 483, 120, 70, 80, 50, 120, 30, 170, 420, 250, 610, 1250, 100 * M, "0.7", depreciation=40, interest_expense=17, pretax_income=103, tax_expense=21, receivables=120, inventory=130, payables=80, cogs=567),
           2023: Y(1060, 424, 60, 20, 20, 50, 60, 60, 220, 380, 330, 580, 1300, 105 * M, "0.19", depreciation=40, interest_expense=22, pretax_income=38, tax_expense=8, receivables=160, inventory=170, payables=75, cogs=636)},
    hand={"fcf": "-30000000", "gross_margin": "0.4", "gross_margin_change": "-0.06", "fcf_yoy": "-2", "revenue_yoy": "0.00952380952380952381",
          "debt_change": "80000000", "diluted_shares_yoy": "0.05"},
    hand_flags=["NEGATIVE_FCF", "FALLING_MARGINS", "REVENUE_DECELERATION", "DILUTION", "RECEIVABLES_OUTPACE_SALES"],
    hand_val={},
)
SCEN["improvement"] = dict(
    price="15", shares=100 * M, forward="1.2",
    years={2018: R(700), 2020: R(780),
           2021: Y(800, 280, 40, 20, 20, 30, 40, 20, 100, 200, 150, 300, 700, 100 * M, "0.2", depreciation=30, interest_expense=10, pretax_income=30, tax_expense=6, receivables=80, inventory=60, payables=50, cogs=520),
           2022: Y(850, 323, 60, 35, 35, 30, 60, 20, 90, 230, 150, 330, 720, 100 * M, "0.35", depreciation=30, interest_expense=9, pretax_income=51, tax_expense="10.2", receivables=85, inventory=62, payables=52, cogs=527),
           2023: Y(950, 399, 110, 70, 80, 30, 120, 10, 80, 300, 160, 400, 760, 100 * M, "0.7", depreciation=30, interest_expense=8, pretax_income=102, tax_expense="20.4", receivables=92, inventory=60, payables=55, cogs=551)},
    hand={"fcf": "50000000", "gross_margin": "0.42", "gross_margin_change": "0.04", "fcf_yoy": "9", "fcf_acceleration": None, "revenue_yoy": "0.11764705882352941176",
          "revenue_acceleration": "0.05514705882352941176", "debt_change": "-20000000", "eps_yoy": "1"},
    hand_flags=[],
    hand_val={},
)


@pytest.mark.parametrize("name", sorted(SCEN))
def test_phases_148_153_fixtures_match_oracle_hand_values_and_flags(name):
    sc = SCEN[name]
    snap, _ = golden_snapshot(name.upper(), sc["years"], AS_OF)
    feats, flags, cur = oracle_from_years(sc["years"])
    bad = compare(snap["features"], feats, name)
    assert not bad, "\n".join(bad)
    assert sorted(snap["flags"]) == sorted(flags) == sorted(sc["hand_flags"]), (name, snap["flags"], flags, sc["hand_flags"])
    for k, hv in sc["hand"].items():
        got = snap["features"].get(k)
        if k in ("pe", "receivables_yoy"):
            continue
        if hv is None:
            assert got is None, (name, k, got)
        else:
            assert O.rel_close(got, Decimal(hv), 1e-12, 1e-14), (name, k, got, hv)
    raw = {**snap["raw"], "preferred_equity": 0.0, "minority_interest": 0.0}
    val = valuation_features(float(sc["price"]), float(sc["shares"]), raw, price_currency="GBP", statement_currency="GBP", price_basis="S", share_basis="S",
                             forward_eps=float(sc["forward"]))
    want = O.valuation(sc["price"], sc["shares"], {**cur, "preferred_equity": Decimal(0), "minority_interest": Decimal(0)}, forward_eps=sc["forward"])
    assert not compare(val, want, name + ":val", abs_tol=1e-12)
    for k, hv in sc["hand_val"].items():
        if hv is None:
            assert val[k] is None, (name, k, val[k])
        else:
            assert O.rel_close(val[k], Decimal(hv), 1e-12, 1e-14), (name, k, val[k], hv)


def test_value_trap_cheap_multiples_do_not_hide_risk_flags():
    snap, _ = golden_snapshot("VT", SCEN["value_trap"]["years"], AS_OF)
    raw = {**snap["raw"], "preferred_equity": 0.0, "minority_interest": 0.0}
    v = valuation_features(3.0, 100e6, raw, price_currency="GBP", statement_currency="GBP", price_basis="S", share_basis="S")
    assert v["pe"] == pytest.approx(7.5) and v["pb"] == pytest.approx(1.25)      # looks cheap ...
    assert v["price_fcf"] is None and v["fcf_yield"] < 0                          # ... but it burns cash: undefined, never "cheap"
    assert {"NEGATIVE_FCF", "FALLING_MARGINS", "REVENUE_DECELERATION"} <= set(snap["flags"])


def test_deterioration_vs_improvement_have_opposite_signs_on_every_trend_feature():
    det, _ = golden_snapshot("DT", SCEN["deterioration"]["years"], AS_OF)
    imp, _ = golden_snapshot("IM", SCEN["improvement"]["years"], AS_OF)
    for k in ("gross_margin_change", "operating_margin_change", "fcf_yoy", "revenue_acceleration"):
        assert (det["features"][k] or 0) < 0 < (imp["features"][k] or 0), (k, det["features"][k], imp["features"][k])
    assert det["flags"] and not imp["flags"]


# =====================================================================================================
# Flag trigger exactness (strict inequalities at the mathematical boundary)
# =====================================================================================================
def flags_for(cur, prior=None, old=None, years=None):
    ys = years or {2021: old or {}, 2022: prior or {}, 2023: cur}
    ys = {y: {**{"revenue": "100"}, **d} for y, d in ys.items() if d is not None}
    snap, _ = golden_snapshot("FLG", ys, AS_OF)
    return snap


def test_flag_boundaries_zero_is_not_a_flag():
    # FCF exactly zero: no NEGATIVE_FCF
    assert "NEGATIVE_FCF" not in flags_for({"ocf": "10", "capex": "10"})["flags"]
    assert "NEGATIVE_FCF" in flags_for({"ocf": "10", "capex": "10.01"})["flags"]
    # gross margin unchanged: no FALLING_MARGINS ; down by 0.01 pt: flag
    assert "FALLING_MARGINS" not in flags_for({"gross_profit": "40"}, {"gross_profit": "40"})["flags"]
    assert "FALLING_MARGINS" in flags_for({"gross_profit": "39.99"}, {"gross_profit": "40"})["flags"]
    # equal growth -> zero acceleration -> no flag; slower -> flag
    eq = {2021: {"revenue": "100"}, 2022: {"revenue": "110"}, 2023: {"revenue": "121"}}
    assert "REVENUE_DECELERATION" not in flags_for(None, years=eq)["flags"]
    slow = {2021: {"revenue": "100"}, 2022: {"revenue": "110"}, 2023: {"revenue": "120.99"}}
    assert "REVENUE_DECELERATION" in flags_for(None, years=slow)["flags"]
    # dilution: unchanged shares no flag; any increase flags (threshold-free by design)
    base = {"diluted_shares": 100 * M, "eps": "1"}
    assert "DILUTION" not in flags_for(dict(base), dict(base))["flags"]
    assert "DILUTION" in flags_for({**base, "diluted_shares": 100 * M + 1}, dict(base))["flags"]
    assert "DILUTION" not in flags_for({**base, "diluted_shares": 100 * M - 1}, dict(base))["flags"]
    # receivables growing exactly with sales: no flag
    ys = {2021: {"revenue": "100", "receivables": "10"}, 2022: {"revenue": "110", "receivables": "11"}, 2023: {"revenue": "121", "receivables": "12.1"}}
    assert "RECEIVABLES_OUTPACE_SALES" not in flags_for(None, years=ys)["flags"]
    ys[2023]["receivables"] = "12.2"
    assert "RECEIVABLES_OUTPACE_SALES" in flags_for(None, years=ys)["flags"]


def test_flags_absent_when_inputs_missing_never_defaulted():
    snap = flags_for({"gross_profit": "40"}, {})          # prior has no gross profit -> change unknown
    assert snap["flags"] == []
    snap = flags_for({"ocf": "10"}, {})                    # capex missing -> FCF unknown -> no NEGATIVE_FCF
    assert "NEGATIVE_FCF" not in snap["flags"] and snap["features"]["fcf"] is None


def test_falling_margins_flag_watches_GROSS_margin_only_LIMITATION():
    """Pinned limitation: operating margin collapse with flat/rising gross margin raises no FALLING_MARGINS."""
    cur = {"gross_profit": "50", "operating_income": "2"}
    prior = {"gross_profit": "50", "operating_income": "30"}
    snap = flags_for(cur, prior)
    assert snap["features"]["operating_margin_change"] == pytest.approx(-0.28, rel=1e-12)
    assert "FALLING_MARGINS" not in snap["flags"]


def test_QA_CALC_02_mathematically_equal_margins_must_not_raise_falling_margins():
    # prior: 2408.79 / 7192.06 ; current: both x7 -> identical ratio mathematically (gross_margin_change == 0 exactly)
    years = {2021: {"revenue": "7000"}, 2022: {"revenue": "7192.06", "gross_profit": "2408.79"}, 2023: {"revenue": "50344.42", "gross_profit": "16861.53"}}
    snap, _ = golden_snapshot("TIE", years, AS_OF)
    assert abs(snap["features"]["gross_margin_change"]) <= 1e-15
    assert "FALLING_MARGINS" not in snap["flags"], snap["features"]["gross_margin_change"]


# =====================================================================================================
# Stock splits and dilution (2-for-1 invariance, adapters.apply_known_splits)
# =====================================================================================================
def split_store(events):
    s = PointInTimeFundamentalStore()
    for ev in events:
        s.add_event(symbol="X", event_type="split", event_at=ev["at"], reported_at=ev["pub"], available_at=ev["pub"], source=ev.get("src", "vendorA"),
                    revision=ev.get("rev", "1"), payload={"ratio": ev["ratio"], **({"event_key": ev["key"]} if "key" in ev else {})})
    return s


def test_two_for_one_split_adjusts_quantity_price_and_keeps_value():
    s = split_store([dict(at="2024-03-01T00:00:00Z", pub="2024-02-15T12:00:00Z", ratio=2.0)])
    basis = "2024-01-01T00:00:00Z"
    for as_of, q, p in [("2024-02-10T00:00:00Z", 100, 50.0),   # not announced yet
                        ("2024-02-20T00:00:00Z", 100, 50.0),   # announced, not yet effective: must NOT adjust
                        ("2024-03-01T00:00:00Z", 200, 25.0),   # effective instant: adjusted (event_at <= as_of)
                        ("2024-03-02T00:00:00Z", 200, 25.0)]:
        out = apply_known_splits(s, "X", 100, 50.0, basis, as_of)
        assert (out["quantity"], out["price"]) == (q, p), as_of
        assert out["quantity"] * out["price"] == pytest.approx(5000.0, rel=1e-15)   # position value invariant
    # position opened on/after the effective date is already on the post-split basis (strict start < event_at)
    out = apply_known_splits(s, "X", 100, 50.0, "2024-03-01T00:00:00Z", "2024-03-05T00:00:00Z")
    assert (out["quantity"], out["price"]) == (100, 50.0)
    out = apply_known_splits(s, "X", 100, 50.0, "2024-02-29T23:59:59Z", "2024-03-05T00:00:00Z")
    assert (out["quantity"], out["price"]) == (200, 25.0)


def test_reverse_split_and_two_sequential_splits_compound():
    s = split_store([dict(at="2024-03-01T00:00:00Z", pub="2024-02-01T00:00:00Z", ratio=2.0, key="s1"),
                     dict(at="2024-06-03T00:00:00Z", pub="2024-05-01T00:00:00Z", ratio=3.0, key="s2"),
                     dict(at="2024-09-02T00:00:00Z", pub="2024-08-01T00:00:00Z", ratio=0.1, key="rs")])
    out = apply_known_splits(s, "X", 100, 60.0, "2024-01-01T00:00:00Z", "2024-12-01T00:00:00Z")
    assert out["quantity"] == pytest.approx(100 * 2 * 3 * 0.1, rel=1e-12) and out["price"] == pytest.approx(60.0 / (2 * 3 * 0.1), rel=1e-12)
    assert out["quantity"] * out["price"] == pytest.approx(6000.0, rel=1e-12)


@pytest.mark.parametrize("ratio", [0.0, -2.0, float("nan"), float("inf")])
def test_invalid_split_ratio_is_refused(ratio):
    import json
    s = PointInTimeFundamentalStore()
    try:
        s.add_event(symbol="X", event_type="split", event_at="2024-03-01T00:00:00Z", reported_at="2024-02-01T00:00:00Z",
                    available_at="2024-02-01T00:00:00Z", source="v", revision="1", payload={"ratio": ratio})
    except ValueError:
        return   # rejected at ingestion (allow_nan=False) is also fail-safe
    with pytest.raises(ValueError):
        apply_known_splits(s, "X", 100, 50.0, "2024-01-01T00:00:00Z", "2024-04-01T00:00:00Z")


def test_revised_split_ratio_from_same_source_uses_latest_version():
    s = split_store([dict(at="2024-03-01T00:00:00Z", pub="2024-02-15T12:00:00Z", ratio=2.0, rev="1"),
                     dict(at="2024-03-01T00:00:00Z", pub="2024-02-20T12:00:00Z", ratio=3.0, rev="2")])
    assert apply_known_splits(s, "X", 100, 50.0, "2024-01-01T00:00:00Z", "2024-03-02T00:00:00Z")["quantity"] == 300.0


def test_QA_CALC_03_same_split_from_two_sources_must_apply_once():
    s = split_store([dict(at="2024-03-01T00:00:00Z", pub="2024-02-15T12:00:00Z", ratio=2.0, src="vendorA"),
                     dict(at="2024-03-01T00:00:00Z", pub="2024-02-16T12:00:00Z", ratio=2.0, src="vendorB")])
    out = apply_known_splits(s, "X", 100, 50.0, "2024-01-01T00:00:00Z", "2024-03-02T00:00:00Z")
    assert (out["quantity"], out["price"]) == (200.0, 25.0), out


def test_split_invariance_of_valuation_ratios_and_forward_eps_hazard():
    snap, _ = golden_snapshot("A", {2022: {"revenue": "100"}, **{2023: {k: v for k, v in Y(120, 60, 24, 18, 25, 5, 30, 5, 15, 50, 25, 90, 150, 10 * M, "1.8", depreciation=6).items()}}}, AS_OF)
    raw = {**snap["raw"], "preferred_equity": 0.0, "minority_interest": 0.0}
    kw = dict(price_currency="GBP", statement_currency="GBP")
    pre = valuation_features(20.0, 10e6, raw, price_basis="pre", share_basis="pre", forward_eps=2.0, **kw)
    post = valuation_features(10.0, 20e6, raw, price_basis="post", share_basis="post", forward_eps=1.0, **kw)
    for k in pre:
        assert pre[k] == pytest.approx(post[k], rel=1e-12), k          # every ratio and market cap invariant
    # basis mismatch is refused
    with pytest.raises(ValueError):
        valuation_features(10.0, 10e6, raw, price_basis="post", share_basis="pre", **kw)
    # HAZARD (limitation): forward_eps carries no basis tag, so an unadjusted forward EPS silently halves forward P/E
    wrong = valuation_features(10.0, 20e6, raw, price_basis="post", share_basis="post", forward_eps=2.0, **kw)
    assert wrong["forward_pe"] == pytest.approx(5.0) and pre["forward_pe"] == pytest.approx(10.0)


def test_dilution_100m_to_120m_is_exactly_20_percent_and_flags():
    ys = {2022: {"revenue": "100", "diluted_shares": 100 * M, "eps": "1"}, 2023: {"revenue": "100", "diluted_shares": 120 * M, "eps": "0.8333"}}
    snap, _ = golden_snapshot("DIL", ys, AS_OF)
    assert snap["features"]["diluted_shares_yoy"] == pytest.approx(0.20, rel=1e-12) and "DILUTION" in snap["flags"]
    assert snap["features"]["eps_yoy"] == pytest.approx(0.8333 / 1 - 1, rel=1e-12)


def test_split_without_common_basis_cannot_emit_economic_dilution_or_eps_growth():
    """Pre-split FY22 (10m shares, EPS 1.8) vs post-split FY23 (20m shares, EPS 0.9): bases differ -> per-share growth withheld."""
    ys = {2022: {"revenue": "100"}, 2023: {"revenue": "110"}}
    facts = annual_facts("SPL", ys)
    facts += [Fact("SPL", "diluted_shares", 10e6, "2022-12-31", "2023-02-25T21:30:00Z", "2023-02-25T21:30:00Z", "fixture", "FY2022", "", "shares", "2022-01-01", metadata={"share_basis": "pre-split", "accounting_standard": "us-gaap"}),
              Fact("SPL", "eps", 1.8, "2022-12-31", "2023-02-25T21:30:00Z", "2023-02-25T21:30:00Z", "fixture", "FY2022", "GBP", "GBP/shares", "2022-01-01", metadata={"share_basis": "pre-split", "accounting_standard": "us-gaap"}),
              Fact("SPL", "diluted_shares", 20e6, "2023-12-31", "2024-02-25T21:30:00Z", "2024-02-25T21:30:00Z", "fixture", "FY2023", "", "shares", "2023-01-01", metadata={"share_basis": "post-split", "accounting_standard": "us-gaap"}),
              Fact("SPL", "eps", 0.9, "2023-12-31", "2024-02-25T21:30:00Z", "2024-02-25T21:30:00Z", "fixture", "FY2023", "GBP", "GBP/shares", "2023-01-01", metadata={"share_basis": "post-split", "accounting_standard": "us-gaap"})]
    snap = snapshot(new_store(facts), "SPL", AS_OF)
    assert snap["features"]["diluted_shares_yoy"] is None and snap["features"]["eps_yoy"] is None
    assert "DILUTION" not in snap["flags"] and snap["per_share_comparability"] == "UNVERIFIED_SPLIT_BASIS"
    # a later filing re-presents FY22 on the post-split basis (EPS 0.9, 20m shares): growth becomes computable and correct
    facts += [Fact("SPL", "diluted_shares", 20e6, "2022-12-31", "2024-02-25T21:30:00Z", "2024-02-25T21:30:00Z", "fixture", "FY2023-recast", "", "shares", "2022-01-01", metadata={"share_basis": "post-split", "accounting_standard": "us-gaap"}),
              Fact("SPL", "eps", 0.9, "2022-12-31", "2024-02-25T21:30:00Z", "2024-02-25T21:30:00Z", "fixture", "FY2023-recast", "GBP", "GBP/shares", "2022-01-01", metadata={"share_basis": "post-split", "accounting_standard": "us-gaap"})]
    snap = snapshot(new_store(facts), "SPL", AS_OF)
    assert snap["features"]["diluted_shares_yoy"] == pytest.approx(0.0, abs=1e-15) and snap["features"]["eps_yoy"] == pytest.approx(0.0, abs=1e-15)
    assert "DILUTION" not in snap["flags"]
