"""Track C phases 86-89: trailing-twelve-month and discrete-quarter logic.

Hand-built scenarios (numbers worked on paper) + a seeded randomized differential test (300 seeds) against a
fiscal-quarter-INDEX oracle (production works from day gaps, the oracle from quarter indices).
Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_calc_ttm.py
"""
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from fundamentals.features import FundamentalFeatureEngine, trailing_twelve_months
from fundamentals.store import Fact, PointInTimeFundamentalStore

from tests.qa_reaudit import calc_oracle as O
from tests.qa_reaudit.calc_qgen import METRICS, expected_ttm, facts_for_q, gen_qcompany, visible_rows

MD = {"accounting_standard": "us-gaap", "share_basis": "one"}


def F(metric, value, start, end, pub, rev="r", basis="one", source="fixture", currency="GBP", units=None, observed_at=None):
    if metric == "eps":
        cur, un = currency, currency + "/shares"
    else:
        cur, un = currency, units or currency + " millions"
    meta = {"accounting_standard": "us-gaap"}
    if basis:
        meta["share_basis"] = basis
    kw = {"observed_at": observed_at} if observed_at else {}
    return Fact("T", metric, value, end, pub, pub, source, rev, cur, un, start, metadata=meta, **kw)


def ttm_at(facts, T):
    s = PointInTimeFundamentalStore()
    s.ingest(facts)
    snap = FundamentalFeatureEngine(s).snapshot("T", T)
    return snap["ttm"], snap, s


# publication times (UTC) of the golden multi-filing company used by several scenarios
P = dict(q1="2023-05-04T20:00:00Z", q2="2023-08-03T20:00:00Z", q3="2023-11-02T20:00:00Z", k="2024-02-22T21:00:00Z", q1b="2024-05-02T20:00:00Z")
QS = {  # (start, end) of each fiscal quarter, calendar fiscal year
    "q1": ("2023-01-01", "2023-03-31"), "q2": ("2023-04-01", "2023-06-30"), "q3": ("2023-07-01", "2023-09-30"),
    "q4": ("2023-10-01", "2023-12-31"), "q1b": ("2024-01-01", "2024-03-31")}
REV = dict(q1=100, q2=110, q3=120, q4=130, q1b=140)
NI = dict(q1=10, q2=12, q3=14, q4=16, q1b=18)
OCF_YTD = dict(q1=20, q2=45, q3=75, k=110, q1b=30)      # discrete 20,25,30,35 | 30
CAPEX_YTD = dict(q1=5, q2=12, q3=20, k=30, q1b=8)       # discrete 5,7,8,10 | 8
EPS = dict(q1=0.10, q2=0.12, q3=0.14, q4=0.16, q1b=0.18)


def scenario_facts(skip=(), eps_q4=True, basis="one"):
    out = []
    fy_start = "2023-01-01"
    for key, pub in (("q1", P["q1"]), ("q2", P["q2"]), ("q3", P["q3"]), ("k", P["k"]), ("q1b", P["q1b"])):
        if key in skip:
            continue
        if key == "k":
            end = "2023-12-31"
            out += [F("revenue", sum(REV[q] for q in ("q1", "q2", "q3", "q4")), fy_start, end, pub, "K"),
                    F("net_income", sum(NI[q] for q in ("q1", "q2", "q3", "q4")), fy_start, end, pub, "K"),
                    F("ocf", OCF_YTD["k"], fy_start, end, pub, "K"), F("capex", CAPEX_YTD["k"], fy_start, end, pub, "K"),
                    F("eps", round(sum(EPS[q] for q in ("q1", "q2", "q3", "q4")), 4), fy_start, end, pub, "K", basis)]
            if eps_q4:
                out.append(F("eps", EPS["q4"], *QS["q4"], pub, "K", basis))
            continue
        s, e = QS[key]
        start_ytd = "2024-01-01" if key == "q1b" else fy_start
        out += [F("revenue", REV[key], s, e, pub, key), F("net_income", NI[key], s, e, pub, key),
                F("eps", EPS[key], s, e, pub, key, basis), F("ocf", OCF_YTD[key], start_ytd, e, pub, key),
                F("capex", CAPEX_YTD[key], start_ytd, e, pub, key)]
        if key in ("q2", "q3"):  # YTD income rows as printed in a 10-Q
            out += [F("revenue", sum(REV[q] for q in ("q1", "q2", "q3")[: (2 if key == "q2" else 3)]), fy_start, e, pub, key),
                    F("net_income", sum(NI[q] for q in ("q1", "q2", "q3")[: (2 if key == "q2" else 3)]), fy_start, e, pub, key)]
    return out


M = 1_000_000


def test_hand_worked_ttm_at_every_publication_boundary():
    facts = scenario_facts()
    cases = [
        # (T, expected end, revenue, net_income, ocf, capex, eps, status)
        ("2023-05-04T19:59:59Z", None, None, None, None, None, None, "INCOMPLETE"),     # nothing public yet
        ("2023-11-02T20:05:00Z", "2023-09-30", None, None, None, None, None, "INCOMPLETE"),  # only 3 quarters of FY2023
        ("2024-02-22T20:59:59Z", "2023-09-30", None, None, None, None, None, "INCOMPLETE"),  # one second before the 10-K is available
        ("2024-02-22T21:00:00Z", "2023-12-31", 460 * M, 52 * M, 110 * M, 30 * M, 0.52, "AVAILABLE"),    # Q1..Q4 2023
        ("2024-05-02T19:59:59Z", "2023-12-31", 460 * M, 52 * M, 110 * M, 30 * M, 0.52, "AVAILABLE"),
        ("2024-05-02T20:00:00Z", "2024-03-31", 500 * M, 60 * M, 120 * M, 33 * M, 0.60, "AVAILABLE"),    # Q2'23..Q1'24
    ]
    for T, end, rev, ni, ocf, capex, eps, status in cases:
        ttm, _, _ = ttm_at(facts, T)
        assert ttm["status"] == status, (T, ttm)
        assert ttm["period_end"] == end, (T, ttm["period_end"])
        v = ttm["values"]
        if status == "AVAILABLE":
            assert v["revenue"] == pytest.approx(rev, rel=1e-12) and v["net_income"] == pytest.approx(ni, rel=1e-12)
            assert v["ocf"] == pytest.approx(ocf, rel=1e-12) and v["capex"] == pytest.approx(capex, rel=1e-12)
            assert v["fcf"] == pytest.approx(ocf - capex, rel=1e-12) and v["eps"] == pytest.approx(eps, rel=1e-12)
        else:
            assert all(x is None for x in v.values()), (T, v)


def test_q4_is_fy_minus_nine_months_and_lineage_has_both_rows():
    ttm, snap, store = ttm_at(scenario_facts(eps_q4=False), "2024-02-23T00:00:00Z")
    assert ttm["values"]["revenue"] == pytest.approx(460e6)           # 110+120+... (Q4 = 460 - 330 = 130)
    assert ttm["values"]["ocf"] == pytest.approx(110e6)               # Q4 = 110 - 75 = 35
    ids = set(ttm["lineage"])
    fy_row = [r for r in store.as_of("T", "2024-02-23T00:00:00Z") if r["metric"] == "revenue" and r["period_end"] == "2023-12-31"][0]
    nine = [r for r in store.as_of("T", "2024-02-23T00:00:00Z") if r["metric"] == "revenue" and r["period_end"] == "2023-09-30" and r["period_start"] == "2023-01-01"][0]
    # Lead re-audit: at a fiscal year end TTM IS the reported FY row; the 9M row is no longer part of its lineage
    assert fy_row["id"] in ids and nine["id"] not in ids
    # EPS: Q4 is NOT derivable (never FY - 9M): documented fail-safe -> eps missing and TTM INCOMPLETE
    assert ttm["values"]["eps"] is None and ttm["reasons"]["eps"] == "MISSING_QUARTERS" and ttm["status"] == "INCOMPLETE"


def test_ttm_never_equals_fy_plus_quarter_and_annual_only_is_incomplete():
    only_annual = [f for f in scenario_facts() if f.revision == "K"]
    ttm, _, _ = ttm_at(only_annual, "2024-03-01T00:00:00Z")
    assert ttm["status"] == "INCOMPLETE" and ttm["values"]["revenue"] is None   # annual totals are never TTM
    # FY row + one later quarter must not be summed to fabricate 12 months
    plus_q1 = only_annual + [f for f in scenario_facts() if f.revision == "q1b"]
    ttm, _, _ = ttm_at(plus_q1, "2024-06-01T00:00:00Z")
    assert ttm["values"]["revenue"] is None


def test_missing_quarter_is_reported_not_papered_over():
    facts = [f for f in scenario_facts() if f.revision != "q2"]   # Q2 10-Q never ingested (3M and 6M rows both absent)
    ttm, _, _ = ttm_at(facts, "2024-05-03T00:00:00Z")
    assert ttm["status"] == "INCOMPLETE"
    v = ttm["values"]
    # Lead re-audit: revenue, net income, OCF and CapEx now follow from the PUBLISHED totals FY23 + Q1'24 - Q1'23,
    # which do not need the missing Q2 10-Q at all (the old four-quarter sum could not be formed).
    assert v["revenue"] == pytest.approx(500e6) and v["net_income"] == pytest.approx(60e6)
    assert v["ocf"] == pytest.approx(120e6) and v["capex"] == pytest.approx(33e6)
    # EPS has no published-totals form (per-share amounts are not additive across bases) and still needs 4 quarters
    assert v["eps"] is None and ttm["reasons"]["eps"] == "NONCONTIGUOUS_QUARTERS"


def test_duplicate_filing_and_duplicate_rows_do_not_double_count():
    facts = scenario_facts()
    doubled = facts + [Fact(**{**f.__dict__, "observed_at": "2030-01-01T00:00:00+00:00"}) for f in facts]  # same filing re-downloaded
    a, _, s1 = ttm_at(facts, "2024-06-01T00:00:00Z")
    b, _, s2 = ttm_at(doubled, "2024-06-01T00:00:00Z")
    assert a["values"] == b["values"] and a["lineage"] == b["lineage"]
    assert len(s2.history("T")) == len(s1.history("T"))  # byte-identical economic identity -> no new rows


def test_same_quarter_from_two_sources_fails_closed():
    facts = scenario_facts() + [F("revenue", 100, *QS["q1"], P["q1"], "dup", source="other-vendor")]
    s = PointInTimeFundamentalStore()
    s.ingest(facts)
    with pytest.raises(ValueError, match="ambiguous"):
        FundamentalFeatureEngine(s).snapshot("T", "2024-06-01T00:00:00Z")


def test_currency_mix_and_unit_mix_fail_closed_in_ttm():
    facts = scenario_facts() + [F("revenue", 1, *QS["q4"], P["k"], "usd", currency="USD", units="USD millions")]
    s = PointInTimeFundamentalStore()
    s.ingest(facts)
    with pytest.raises(ValueError):
        FundamentalFeatureEngine(s).snapshot("T", "2024-06-01T00:00:00Z")


def test_eps_ttm_requires_one_explicit_split_basis():
    for basis in (None,):
        ttm, _, _ = ttm_at(scenario_facts(basis=basis), "2024-06-01T00:00:00Z")
        assert ttm["values"]["eps"] is None and ttm["reasons"]["eps"] == "UNVERIFIED_SPLIT_BASIS" and ttm["status"] == "INCOMPLETE"
    f = scenario_facts()
    f = [Fact(**{**x.__dict__, "metadata": {**x.metadata, "share_basis": "post-split"}}) if (x.metric == "eps" and x.revision == "q3") else x for x in f]
    ttm, _, _ = ttm_at(f, "2024-06-01T00:00:00Z")
    assert ttm["values"]["eps"] is None and ttm["reasons"]["eps"] == "UNVERIFIED_SPLIT_BASIS"
    assert ttm["values"]["revenue"] == pytest.approx(500e6)   # other metrics unaffected


def test_ttm_overflow_is_reported_not_inf():
    big = 1.7e308 / 1e6  # millions; x1e6 -> 1.7e308, four of them overflow the sum
    facts = []
    for key in ("q1", "q2", "q3", "q4", "q1b"):
        pass
    quarters = [("2023-04-01", "2023-06-30", "a"), ("2023-07-01", "2023-09-30", "b"), ("2023-10-01", "2023-12-31", "c"), ("2024-01-01", "2024-03-31", "d")]
    facts = [F("revenue", big, s, e, "2024-05-02T20:00:00Z", r) for s, e, r in quarters]
    ttm, _, _ = ttm_at(facts, "2024-06-01T00:00:00Z")
    assert ttm["values"]["revenue"] is None and ttm["reasons"]["revenue"] == "OVERFLOW"


def test_fiscal_year_change_transition_period_ttm():
    """Dec FY -> Jun FY. 10-KT covers Jan-Jun 2024 (6M). Calendar-quarter revenue 150,160,170,180,190."""
    rev = {"q1": 150, "q2": 160, "q3": 170, "q4": 180, "q5": 190}
    f = []
    f.append(F("revenue", rev["q1"], "2024-01-01", "2024-03-31", "2024-05-02T20:00:00Z", "10Q-Mar24"))
    f.append(F("revenue", rev["q1"] + rev["q2"], "2024-01-01", "2024-06-30", "2024-09-10T20:00:00Z", "10KT"))        # transition 6M only
    f.append(F("revenue", rev["q3"], "2024-07-01", "2024-09-30", "2024-11-05T20:00:00Z", "10Q-Sep24"))
    f.append(F("revenue", rev["q4"], "2024-10-01", "2024-12-31", "2025-02-04T20:00:00Z", "10Q-Dec24"))
    f.append(F("revenue", rev["q3"] + rev["q4"], "2024-07-01", "2024-12-31", "2025-02-04T20:00:00Z", "10Q-Dec24"))
    f.append(F("revenue", rev["q5"], "2025-01-01", "2025-03-31", "2025-05-06T20:00:00Z", "10Q-Mar25"))
    f.append(F("revenue", rev["q3"] + rev["q4"] + rev["q5"], "2024-07-01", "2025-03-31", "2025-05-06T20:00:00Z", "10Q-Mar25"))
    ttm, snap, _ = ttm_at(f, "2025-06-01T00:00:00Z")
    # last four contiguous discrete quarters: Q2'24 (derived 6M-Q1), Q3'24, Q4'24, Q1'25 = 160+170+180+190 = 700
    assert ttm["values"]["revenue"] == pytest.approx(700e6)
    assert ttm["period_start"] == "2024-04-01" and ttm["period_end"] == "2025-03-31"   # window crosses the fiscal-year change
    ttm2, _, _ = ttm_at(f, "2025-02-05T00:00:00Z")
    assert ttm2["values"]["revenue"] == pytest.approx(660e6)  # Q1'24..Q4'24 = 150+160+170+180


def test_52_53_week_calendar_contiguity():
    """52/53-week year: quarter ends Dec-30, Mar-30, Jun-29, Sep-28 (starts the next day) are still contiguous."""
    q = [("2023-09-30", "2023-12-30"), ("2023-12-31", "2024-03-30"), ("2024-03-31", "2024-06-29"), ("2024-06-30", "2024-09-28")]
    f = [F("revenue", v, s, e, "2024-11-05T20:00:00Z", f"q{i}") for i, ((s, e), v) in enumerate(zip(q, (10, 20, 30, 40)))]
    ttm, _, _ = ttm_at(f, "2024-12-01T00:00:00Z")
    assert ttm["values"]["revenue"] == pytest.approx(100e6)
    # a one-day hole between quarters is NOT silently accepted
    g = [F("revenue", v, s, e, "2024-11-05T20:00:00Z", f"q{i}") for i, ((s, e), v) in enumerate(zip(q, (10, 20, 30, 40)))]
    g[2] = F("revenue", 30, "2024-04-02", "2024-06-29", "2024-11-05T20:00:00Z", "q2")
    ttm, _, _ = ttm_at(g, "2024-12-01T00:00:00Z")
    assert ttm["values"]["revenue"] is None and ttm["reasons"]["revenue"] == "NONCONTIGUOUS_QUARTERS"


def test_future_filing_does_not_change_ttm_at_earlier_time():
    facts = scenario_facts()
    base, _, _ = ttm_at([f for f in facts if f.revision != "q1b"], "2024-04-01T00:00:00Z")
    full, _, _ = ttm_at(facts, "2024-04-01T00:00:00Z")
    assert base == full


def test_restated_comparative_mixes_vintages_in_derived_quarter_LIMITATION():
    """Pinned behaviour (not asserted as correct): Q2'23 derived as 6M(original) - Q1(restated in a later filing)
    produces a number no single filing ever printed. Documented in the report as a limitation."""
    facts = [f for f in scenario_facts() if f.metric != "ocf"]
    facts += [F("ocf", 20, "2023-01-01", "2023-03-31", P["q1"], "q1"), F("ocf", 45, "2023-01-01", "2023-06-30", P["q2"], "q2"),
              F("ocf", 75, "2023-01-01", "2023-09-30", P["q3"], "q3"), F("ocf", 110, "2023-01-01", "2023-12-31", P["k"], "K"),
              F("ocf", 30, "2024-01-01", "2024-03-31", P["q1b"], "q1b"),
              F("ocf", 22, "2023-01-01", "2023-03-31", P["q1b"], "q1b-comparative")]   # Q1'23 restated 20 -> 22 inside the Q1'24 10-Q
    ttm, _, _ = ttm_at(facts, "2024-06-01T00:00:00Z")
    assert ttm["values"]["ocf"] == pytest.approx((45 - 22 + 30 + 35 + 30) * M)   # = 118 (mixed vintage), not 120
    assert ttm["values"]["ocf"] != pytest.approx(120 * M)


# ------------------------------------------------------------------ randomized differential (index oracle)
SEEDS = list(range(300))


def pick_T(c, seed):
    rng = random.Random(seed * 104729 + 7)
    pubs = sorted({r.published for r in c.rows})
    if rng.random() < 0.6:
        return rng.choice(pubs) + rng.choice([timedelta(seconds=-1), timedelta(0), timedelta(seconds=1), timedelta(days=1)])
    lo, hi = pubs[0] - timedelta(days=10), pubs[-1] + timedelta(days=400)
    return lo + (hi - lo) * rng.random()


@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed{s}")
def test_random_ttm_and_qoq_match_index_oracle(seed):
    c = gen_qcompany(seed)
    T = pick_T(c, seed)
    s = PointInTimeFundamentalStore()
    s.ingest(facts_for_q(c, "RQ"))
    snap = FundamentalFeatureEngine(s).snapshot("RQ", T)
    ttm = snap["ttm"]
    exp = expected_ttm(c, T)
    ctx = f"seed={seed} T={T.isoformat()} units={c.unit[0]}"
    assert ttm["status"] == exp["status"], (ctx, ttm["status"], exp["status"], ttm["reasons"], {k: (None if v is None else str(v)) for k, v in exp["values"].items()})
    assert ttm["period_end"] == exp["period_end"], (ctx, ttm["period_end"], exp["period_end"])
    for m, want in exp["values"].items():
        got = ttm["values"][m]
        want_base = None if want is None else (want if m == "eps" else want * M)
        assert O.rel_close(got, want_base, 1e-12, 1e-6), (ctx, m, got, want_base, ttm["reasons"].get(m))
    # lineage never cites a row published after T
    ids = set(ttm["lineage"])
    for r in s.history("RQ"):
        if r["id"] in ids:
            assert r["available_at"] <= T.isoformat().replace("+00:00", "+00:00"), (ctx, r)
    # QoQ (annual snapshot exists only once a FY revenue row is public)
    if snap["status"] in ("AVAILABLE", "STALE"):
        for m in ("revenue", "ocf"):
            want = exp["qoq"][m]
            got = snap["features"][m + "_qoq"]
            assert O.rel_close(got, want, 1e-12, 1e-12), (ctx, m + "_qoq", got, want)


def test_random_ttm_generator_hits_available_incomplete_and_gaps():
    seen = {"AVAILABLE": 0, "INCOMPLETE": 0, "STALE": 0, "restated": 0, "gap": 0}
    for seed in SEEDS:
        c = gen_qcompany(seed)
        T = pick_T(c, seed)
        e = expected_ttm(c, T)
        seen[e["status"]] += 1
        seen["restated"] += any(r.revision.endswith("A") and r.published <= T for r in c.rows)
        seen["gap"] += any(r.published <= T for r in c.rows) and e["status"] == "INCOMPLETE" and e["period_end"] is not None
    assert seen["AVAILABLE"] >= 30 and seen["INCOMPLETE"] >= 30 and seen["restated"] >= 10 and seen["gap"] >= 10, seen
