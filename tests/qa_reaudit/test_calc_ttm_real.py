"""Track C: TTM freshness and a read-only spot check of TTM on the 8-company real SEC cache.

Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_calc_ttm_real.py
"""
import json
from pathlib import Path

import pytest

from fundamentals.features import FundamentalFeatureEngine
from fundamentals.providers import sec_facts
from fundamentals.store import PointInTimeFundamentalStore

from tests.qa_reaudit.test_calc_ttm import F, ttm_at

RAW = Path(__file__).resolve().parents[2] / "results" / "fundamental_audit_20260930" / "raw"


def test_QA_CALC_06_ttm_must_not_be_available_when_its_last_quarter_is_years_old():
    quarters = [("2015-01-01", "2015-03-31"), ("2015-04-01", "2015-06-30"), ("2015-07-01", "2015-09-30"), ("2015-10-01", "2015-12-31")]
    facts = []
    for i, (s, e) in enumerate(quarters):
        pub = f"{e[:4]}-{int(e[5:7]) + 1:02d}-20T21:00:00Z" if e[5:7] != "12" else "2016-02-20T21:00:00Z"
        for m, v in (("revenue", 100), ("net_income", 10), ("ocf", 15), ("capex", 5), ("eps", 0.1)):
            facts.append(F(m, v, s, e, pub, f"r{i}"))
    facts.append(F("revenue", 400, "2015-01-01", "2015-12-31", "2016-02-20T21:00:00Z", "FY2015"))   # annual row so the annual block exists
    ttm, snap, _ = ttm_at(facts, "2024-06-01T00:00:00Z")
    assert ttm["period_end"] == "2015-12-31" and snap["status"] == "STALE"        # the annual block knows it is stale (age > 550 days)
    assert ttm["status"] != "AVAILABLE", ttm["status"]                              # ... but the TTM block says AVAILABLE today


def test_ttm_period_end_is_always_reported_so_staleness_is_at_least_visible():
    quarters = [("2015-01-01", "2015-03-31"), ("2015-04-01", "2015-06-30"), ("2015-07-01", "2015-09-30"), ("2015-10-01", "2015-12-31")]
    facts = [F("revenue", 100, s, e, "2016-02-20T21:00:00Z", f"r{i}") for i, (s, e) in enumerate(quarters)]
    ttm, _, _ = ttm_at(facts, "2024-06-01T00:00:00Z")
    assert ttm["period_end"] == "2015-12-31" and ttm["period_start"] == "2015-01-01" and ttm["values"]["revenue"] == pytest.approx(400e6)


needs_raw = pytest.mark.skipif(not (RAW / "MSFT_facts.json").exists(), reason="real SEC cache not present")


def load(sym):
    facts = json.load(open(RAW / f"{sym}_facts.json"))
    subs = json.load(open(RAW / f"{sym}_submissions.json"))
    s = PointInTimeFundamentalStore()
    s.ingest_validated(sec_facts(facts, subs, sym))
    return s


@needs_raw
def test_msft_ttm_matches_the_four_reported_quarters_from_press_releases():
    """MSFT (June fiscal year). At 2024-06-01 the latest 10-Q is Q3 FY24 (quarter ended 2024-03-31).
    TTM revenue = Q4 FY23 56.189 + Q1 FY24 56.517 + Q2 FY24 62.020 + Q3 FY24 61.858 = 236.584 bn (press-release figures);
    TTM net income = 20.081 + 22.291 + 21.870 + 21.939 = 86.181 bn. Q4 FY23 is DERIVED (FY - 9M) in production."""
    s = load("MSFT")
    t = FundamentalFeatureEngine(s).snapshot("MSFT", "2024-06-01T12:00:00Z")["ttm"]
    assert t["period_end"] == "2024-03-31" and t["period_start"] == "2023-04-01"
    assert t["values"]["revenue"] == pytest.approx(236.584e9, rel=1e-9)
    assert t["values"]["net_income"] == pytest.approx(86.181e9, rel=1e-9)
    assert t["values"]["fcf"] == pytest.approx(t["values"]["ocf"] - t["values"]["capex"], rel=1e-12)
    # a day before the Q3 FY24 10-Q was available the TTM still ends 2023-12-31 (nothing leaks early)
    before = FundamentalFeatureEngine(s).snapshot("MSFT", "2024-04-24T00:00:00Z")["ttm"]
    # Q3 FY23 52.857 + Q4 FY23 56.189 + Q1 FY24 56.517 + Q2 FY24 62.020 = 227.583 bn
    assert before["period_end"] == "2023-12-31" and before["values"]["revenue"] == pytest.approx(227.583e9, rel=1e-9)


@needs_raw
def test_real_sec_ttm_status_is_never_available_because_eps_cannot_be_completed_LIMITATION():
    """Pinned limitation (6 companies x 26 quarterly dates = 156 samples): revenue/NI/OCF/CapEx/FCF TTM exist but EPS TTM never does
    (Q4 EPS is not derivable and sec_facts writes 'split_basis' while the engine requires 'share_basis'), so status is INCOMPLETE."""
    import pandas as pd
    avail = rev = 0
    total = 0
    for sym in ("MSFT", "WMT", "JNJ", "XOM", "CAT", "BA"):
        eng = FundamentalFeatureEngine(load(sym))
        for T in pd.date_range("2019-01-15", "2025-06-15", freq="3MS"):
            t = eng.snapshot(sym, T.strftime("%Y-%m-%dT12:00:00Z"))["ttm"]
            total += 1
            avail += t["status"] == "AVAILABLE"
            rev += t["values"]["revenue"] is not None
            assert t["values"]["eps"] is None or t["status"] == "AVAILABLE"
    # lead re-audit: still never AVAILABLE; a TTM whose last quarter is older than the freshness limit is now STALE instead of INCOMPLETE
    assert total == 156 and rev >= 155 and avail == 0, (total, rev, avail)


@needs_raw
def test_real_jpm_ttm_block_is_ten_years_stale_while_annual_block_is_current():
    s = load("JPM")
    snap = FundamentalFeatureEngine(s).snapshot("JPM", "2024-06-01T12:00:00Z")
    assert snap["status"] == "AVAILABLE" and snap["period_end"] == "2023-12-31"
    assert snap["ttm"]["period_end"] == "2014-12-31"            # last quarterly 'revenue' row the tag map ever saw for JPM
