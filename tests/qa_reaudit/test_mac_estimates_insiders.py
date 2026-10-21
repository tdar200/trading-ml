"""QA track D: estimate revisions, forward P/E sourcing, insider activity, short interest (absent).

Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_mac_estimates_insiders.py
"""
from __future__ import annotations

import random
import re
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from fundamentals.adapters import estimate_revisions, insider_activity
from fundamentals.features import valuation_features
from fundamentals.store import Fact, PointInTimeFundamentalStore
from tests.qa_reaudit.mac_fixtures import CANARY, REPO, canonical, fact, iso, truncated_store
from tests.qa_reaudit.mac_oracle import UTC, dt, revision

PROD_DIRS = ("fundamentals", "livesim", "strategies", "dashboard", "analysis", "data", "models", "examples")
AS_OF = "2025-03-31T00:00:00Z"


def est(value, at, period="2025-12-31", source="IBES", units="USD/shares", currency="USD", metric="consensus_eps", **kw):
    d = dict(symbol="XYZ", metric=metric, value=value, period_end=period, reported_at=at, available_at=at, source=source,
             revision=at, currency=currency, units=units, kind="estimate")
    d.update(kw)
    return Fact(**d)


def rev(store, as_of=AS_OF, period="2025-12-31", metric="consensus_eps"):
    return estimate_revisions(store, "XYZ", as_of, metric, period)


# ============================================================================ estimate revisions
def test_consensus_2_00_to_2_20_over_30_days_is_plus_10_percent_only_for_30d():
    st = PointInTimeFundamentalStore()
    st.ingest([est(2.00, "2025-03-01T00:00:00Z"), est(2.20, "2025-03-31T00:00:00Z")])
    out = rev(st)
    assert out["30d"] == pytest.approx(0.10, rel=1e-12) and out["60d"] is None and out["90d"] is None
    assert out["30d"] == pytest.approx(revision(2.20, 2.00), rel=1e-12)


def test_revision_windows_60_and_90_days_use_the_snapshot_known_at_that_time():
    st = PointInTimeFundamentalStore()
    st.ingest([est(1.00, "2024-12-01T00:00:00Z"), est(1.50, "2025-01-30T00:00:00Z"), est(2.00, "2025-03-01T00:00:00Z"),
               est(2.20, "2025-03-31T00:00:00Z")])
    out = rev(st)
    assert out["30d"] == pytest.approx(0.10, rel=1e-12)           # 2.2 vs 2.0 (as of 2025-03-01)
    assert out["60d"] == pytest.approx(2.2 / 1.5 - 1, rel=1e-12)   # as of 2025-01-30 -> 1.5
    assert out["90d"] == pytest.approx(2.2 / 1.0 - 1, rel=1e-12)   # as of 2024-12-31 -> 1.0


def test_forecast_period_is_held_fixed():
    st = PointInTimeFundamentalStore()
    st.ingest([est(2.00, "2025-03-01T00:00:00Z", period="2025-12-31"), est(2.20, "2025-03-31T00:00:00Z", period="2025-12-31"),
               est(9.00, "2025-03-01T00:00:00Z", period="2026-12-31"), est(1.00, "2025-03-31T00:00:00Z", period="2026-12-31")])
    assert rev(st, period="2025-12-31")["30d"] == pytest.approx(0.10, rel=1e-12)
    assert rev(st, period="2026-12-31")["30d"] == pytest.approx(-8 / 9, rel=1e-12)
    assert rev(st, period="2027-12-31") == {"30d": None, "60d": None, "90d": None}


@pytest.mark.parametrize("old,new,expected", [
    (2.0, 2.2, 0.1), (2.0, 1.8, -0.1), (0.0, 1.0, None), (1.0, 0.0, -1.0), (-1.0, -0.5, 0.5), (-2.0, -3.0, -0.5),
    (1.0, -1.0, -2.0), (-0.5, 0.5, 2.0),
])
def test_zero_and_negative_bases(old, new, expected):
    st = PointInTimeFundamentalStore()
    st.ingest([est(old, "2025-03-01T00:00:00Z"), est(new, "2025-03-31T00:00:00Z")])
    got = rev(st)["30d"]
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-15) and got == pytest.approx(revision(new, old), rel=1e-12, abs=1e-15)


def test_withdrawn_consensus_is_missing_not_zero():
    st = PointInTimeFundamentalStore()
    st.ingest([est(2.0, "2025-03-01T00:00:00Z"), est(None, "2025-03-31T00:00:00Z")])
    assert rev(st)["30d"] is None


@pytest.mark.parametrize("seconds,visible", [(0, True), (1, False)])
def test_window_boundary_estimate_published_exactly_30_days_before_is_visible(seconds, visible):
    st = PointInTimeFundamentalStore()
    old_at = datetime(2025, 3, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    st.ingest([est(2.0, iso(old_at)), est(2.2, "2025-03-31T00:00:00Z")])
    assert (rev(st)["30d"] is not None) is visible


def test_revisions_published_after_T_are_never_visible_and_become_visible_exactly_at_publication():
    st = PointInTimeFundamentalStore()
    st.ingest([est(2.0, "2025-03-01T00:00:00Z"), est(2.2, "2025-03-31T00:00:00Z"),
               est(CANARY, "2025-03-31T00:00:01Z")])
    assert rev(st, "2025-03-31T00:00:00Z")["30d"] == pytest.approx(0.10, rel=1e-12)
    assert rev(st, "2025-03-31T00:00:00.999999Z")["30d"] == pytest.approx(0.10, rel=1e-12)
    assert rev(st, "2025-03-31T00:00:01Z")["30d"] == pytest.approx(CANARY / 2.0 - 1, rel=1e-12)


def test_unit_or_currency_change_between_snapshots_is_rejected_not_converted():
    for old_kw, new_kw in (({"currency": "USD", "units": "USD/shares"}, {"currency": "GBP", "units": "GBP/shares"}),
                           ({"units": "USD/shares"}, {"units": "USD/cents"})):
        st = PointInTimeFundamentalStore()
        st.ingest([est(2.0, "2025-03-01T00:00:00Z", **old_kw), est(2.2, "2025-03-31T00:00:00Z", **new_kw)])
        with pytest.raises(ValueError, match="estimate|ambiguous"):
            rev(st)


def test_two_sources_for_the_same_estimate_are_rejected_not_averaged():
    st = PointInTimeFundamentalStore()
    st.ingest([est(2.0, "2025-03-01T00:00:00Z", source="IBES"), est(2.1, "2025-03-01T00:00:00Z", source="Zacks"),
               est(2.2, "2025-03-31T00:00:00Z", source="IBES")])
    with pytest.raises(ValueError, match="ambiguous estimate source"):
        rev(st)


def test_estimate_kind_skips_financial_currency_and_unit_validation_LIMITATION():
    """kind='estimate' bypasses Fact's currency/unit contract: a bogus currency is stored as-is."""
    st = PointInTimeFundamentalStore()
    assert st.ingest([est(2.0, "2025-03-01T00:00:00Z", currency="ZZZ", units="whatever")]) == 1
    with pytest.raises(ValueError, match="unsupported financial currency"):
        fact(kind="financial", currency="ZZZ", units="ZZZ").normalized()


def test_estimates_cannot_be_selected_by_accident_as_financial_statement_facts():
    from fundamentals.features import FundamentalFeatureEngine
    st = PointInTimeFundamentalStore()
    st.ingest([est(2.0, "2025-03-01T00:00:00Z", metric="revenue", period="2025-12-31")])
    assert FundamentalFeatureEngine(st).snapshot("XYZ", AS_OF)["status"] == "MISSING"
    assert st.symbols(AS_OF) == []         # symbols() only counts kind='financial'


@pytest.mark.parametrize("seed", range(100))
def test_estimate_random_timelines_match_oracle_truncated_db_and_canary(seed):
    rng = random.Random(seed)
    base = datetime(2024, 6, 1, tzinfo=UTC)
    revs = {"2025-12-31": [], "2026-12-31": []}
    taken = set()
    for period in revs:
        t = base
        for _ in range(rng.randrange(2, 10)):
            t += timedelta(days=rng.randrange(1, 45), seconds=rng.randrange(0, 86400))
            while t in taken:
                t += timedelta(seconds=1)
            taken.add(t)
            revs[period].append((t, round(rng.uniform(-3, 4), 2)))
    st = PointInTimeFundamentalStore()
    st.ingest([est(v, iso(t), period=p) for p, items in revs.items() for t, v in items])
    for _ in range(4):
        T = base + timedelta(days=rng.randrange(20, 520), seconds=rng.randrange(0, 86400))
        period = rng.choice(list(revs))

        def known(at):
            vis = [(t, v) for t, v in revs[period] if t <= at]
            return max(vis)[1] if vis else None
        got = estimate_revisions(st, "XYZ", T, "consensus_eps", period)
        for days, key in ((30, "30d"), (60, "60d"), (90, "90d")):
            cur, old = known(T), known(T - timedelta(days=days))
            want = revision(cur, old) if cur is not None and old is not None else None
            if want is None:
                assert got[key] is None, f"seed={seed} {key}"
            else:
                assert got[key] == pytest.approx(want, rel=1e-12, abs=1e-15), f"seed={seed} {key}"
        trunc = truncated_store(st, T)
        assert estimate_revisions(trunc, "XYZ", T, "consensus_eps", period) == got
        canary = truncated_store(st, datetime(2100, 1, 1, tzinfo=UTC))
        canary.ingest([est(CANARY, iso(T + timedelta(seconds=1)), period=period)])
        assert estimate_revisions(canary, "XYZ", T, "consensus_eps", period) == got


# ============================================================================ forward P/E sourcing
def _py_files():
    for d in PROD_DIRS:
        for p in (REPO / d).rglob("*.py"):
            yield p


def test_no_live_code_path_computes_forward_pe_NOT_WIRED():
    """valuation_features(forward_eps=...) is only called by tests/scripts; no live path sources consensus,
    so no non-PIT consensus can reach a decision. (Also nothing calls estimate_revisions.)"""
    callers = []
    for p in _py_files():
        text = p.read_text()
        for pat in (r"valuation_features\(", r"estimate_revisions\("):
            for m in re.finditer(pat, text):
                line = text[: m.start()].count("\n")
                src_line = text.splitlines()[line]
                if src_line.lstrip().startswith("def "):
                    continue
                callers.append((str(p.relative_to(REPO)), src_line.strip()))
    assert callers == []
    engine_src = (REPO / "fundamentals/features.py").read_text().split("class FundamentalFeatureEngine")[1]
    assert "forward" not in engine_src and "valuation_features" not in engine_src


@pytest.mark.parametrize("price,forward_eps,expected", [(40.0, 2.0, 20.0), (40.0, 0.5, 80.0), (40.0, -2.0, None),
                                                         (40.0, 0.0, None), (40.0, None, None)])
def test_forward_pe_from_caller_supplied_consensus(price, forward_eps, expected):
    out = valuation_features(price, 1_000_000.0, {"net_income": 5e6, "revenue": 5e7}, price_currency="GBP",
                             statement_currency="GBP", price_basis="A", share_basis="A", forward_eps=forward_eps)
    if expected is None:
        assert out["forward_pe"] is None                  # negative forward EPS is never ranked 'cheap'
    else:
        assert out["forward_pe"] == pytest.approx(expected, rel=1e-12)


def test_forward_pe_refuses_currency_mismatch_and_split_basis_mismatch():
    kw = dict(price_currency="GBP", statement_currency="USD", price_basis="A", share_basis="A", forward_eps=2.0)
    with pytest.raises(ValueError, match="currency"):
        valuation_features(40.0, 1e6, {"net_income": 1.0}, **kw)
    kw.update(statement_currency="GBP", share_basis="B")
    with pytest.raises(ValueError, match="split basis"):
        valuation_features(40.0, 1e6, {"net_income": 1.0}, **kw)


# ============================================================================ insiders
def txn(metric="insider_txn", value=100_000.0, period_end="2025-03-10", at="2025-03-12T21:05:00Z", code="P",
        discretionary=True, insider="CEO", currency="USD", revision_="A1", **meta):
    return Fact("XYZ", metric, value, period_end, at, at, "SEC/form4", revision_, currency, currency, kind="ownership",
                metadata={"transaction_code": code, "discretionary": discretionary, "insider_id": insider, **meta})


def act(store, as_of="2025-04-01T00:00:00Z", **kw):
    return insider_activity(store, "XYZ", as_of, **kw)


@pytest.mark.parametrize("as_of,status", [
    ("2025-03-10T23:59:59Z", "MISSING"),        # trade day itself, not filed
    ("2025-03-11T12:00:00Z", "MISSING"),        # D+1
    ("2025-03-12T21:04:59Z", "MISSING"),        # one second before the filing is public
    ("2025-03-12T21:05:00Z", "AVAILABLE"),      # filed D+2: visible exactly at availability
])
def test_trade_on_day_D_filed_D_plus_2_invisible_until_filed(as_of, status):
    st = PointInTimeFundamentalStore()
    st.ingest([txn()])
    out = act(st, as_of)
    assert out["status"] == status
    if status == "MISSING":
        assert out["discretionary_purchase_value"] is None and out["reported_sale_value"] is None   # never 0 for absent data
    else:
        assert out["discretionary_purchase_value"] == 100_000.0


def test_classification_discretionary_purchases_vs_sales_vs_other():
    st = PointInTimeFundamentalStore()
    st.ingest([
        txn("t1", 100.0, code="P", discretionary=True, insider="A"),
        txn("t2", 200.0, code="P", discretionary=True, insider="B"),
        txn("t3", 400.0, code="P", discretionary=False, insider="C"),          # non-discretionary purchase
        txn("t4", 800.0, code="P", discretionary="true", insider="D"),         # truthy string is not True
        txn("t5", 1600.0, code="S", discretionary=False, insider="E"),
        txn("t6", 3200.0, code="S", discretionary=False, insider="F", plan="10b5-1", scheduled=True),
        txn("t7", 6400.0, code="A", discretionary=False, insider="G"),         # award
        txn("t8", 12800.0, code="M", discretionary=False, insider="H"),        # option exercise
        txn("t9", 25600.0, code="F", discretionary=False, insider="I"),        # tax withholding
    ])
    out = act(st)
    assert out["discretionary_purchase_value"] == 300.0 and out["distinct_buyers"] == 2
    assert out["reported_sale_value"] == 4800.0           # 10b5-1 scheduled sale is included as reported, not as a signal
    assert out["unclassified_transactions"] == 5          # t3, t4, t7, t8, t9
    assert "bearish" not in " ".join(map(str, out.values())).lower().replace("not automatically bearish", "")
    assert "not automatically bearish" in out["note"]
    assert out["currency"] == "USD" and out["status"] == "AVAILABLE"


@pytest.mark.parametrize("delta,included", [(timedelta(0), True), (timedelta(microseconds=-1), False), (timedelta(days=1), True)])
@pytest.mark.parametrize("days", [90, 30, 1])
def test_window_boundary_is_inclusive_on_publication_time(days, delta, included):
    as_of = datetime(2025, 6, 1, 12, tzinfo=UTC)
    at = as_of - timedelta(days=days) + delta
    st = PointInTimeFundamentalStore()
    st.ingest([txn(period_end=at.date().isoformat(), at=iso(at))])
    assert (act(st, as_of, days=days)["status"] == "AVAILABLE") is included


def test_window_uses_publication_age_not_transaction_date():
    """Trade 200 days ago but filed yesterday is in the 90-day window; trade 10 days ago filed 100 days ago is not."""
    as_of = datetime(2025, 9, 1, tzinfo=UTC)
    st = PointInTimeFundamentalStore()
    st.ingest([txn("late", 111.0, period_end="2025-02-10", at="2025-08-31T10:00:00Z"),
               txn("early", 222.0, period_end="2025-05-20", at="2025-05-22T10:00:00Z")])
    assert act(st, as_of, days=90)["discretionary_purchase_value"] == 111.0


@pytest.mark.parametrize("days", [0, -1, -90])
def test_window_must_be_positive(days):
    with pytest.raises(ValueError, match="positive window"):
        act(PointInTimeFundamentalStore(), days=days)


def test_currency_rules():
    st = PointInTimeFundamentalStore()
    st.ingest([txn("a", currency="USD"), txn("b", currency="GBP")])
    with pytest.raises(ValueError, match="currency"):
        act(st)
    st2 = PointInTimeFundamentalStore()
    st2.ingest([Fact("XYZ", "t", 5.0, "2025-03-10", "2025-03-12T00:00:00Z", "2025-03-12T00:00:00Z", "s", "r", "", "shares",
                     kind="ownership", metadata={"transaction_code": "P", "discretionary": True})])
    with pytest.raises(ValueError, match="explicit currency"):
        act(st2)
    st3 = PointInTimeFundamentalStore()
    st3.ingest([txn("a", currency="GBP", value=7.0)])
    assert act(st3)["currency"] == "GBP"


def test_duplicate_filing_ingested_twice_counts_once_even_with_different_observed_at():
    st = PointInTimeFundamentalStore()
    a = txn()
    st.ingest([replace(a, observed_at="2025-03-13T00:00:00Z")])
    st.ingest([replace(a, observed_at="2025-09-01T00:00:00Z")])
    assert st.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
    assert act(st)["discretionary_purchase_value"] == 100_000.0


def test_amended_filing_supersedes_original_for_same_transaction_no_double_count():
    st = PointInTimeFundamentalStore()
    st.ingest([txn(value=100_000.0, at="2025-03-12T21:05:00Z", revision_="A1"),
               txn(value=80_000.0, at="2025-03-20T09:00:00Z", revision_="A1/A")])
    assert act(st, "2025-03-15T00:00:00Z")["discretionary_purchase_value"] == 100_000.0
    assert act(st, "2025-03-21T00:00:00Z")["discretionary_purchase_value"] == 80_000.0


def test_QA_MAC_03_two_insiders_same_trade_date_both_counted():
    st = PointInTimeFundamentalStore()
    st.ingest([txn(value=250_000.0, insider="CEO", revision_="0001-A", at="2025-03-12T21:05:00Z"),
               txn(value=100_000.0, insider="CFO", revision_="0001-B", at="2025-03-13T21:05:00Z")])   # same metric, same period_end
    out = act(st)
    assert out["discretionary_purchase_value"] == 350_000.0 and out["distinct_buyers"] == 2


def test_same_timestamp_conflicting_ownership_rows_raise_instead_of_guessing():
    st = PointInTimeFundamentalStore()
    st.ingest([txn(value=250_000.0, insider="CEO", revision_="0001-A"), txn(value=100_000.0, insider="CFO", revision_="0001-B")])
    with pytest.raises(ValueError, match="ambiguous conflicting source facts"):
        act(st)


def test_missing_insider_id_inflates_distinct_buyers_LIMITATION():
    st = PointInTimeFundamentalStore()
    facts = []
    for i in range(3):
        f = txn(f"m{i}")
        facts.append(replace(f, metadata={"transaction_code": "P", "discretionary": True}))   # no insider_id
    st.ingest(facts)
    out = act(st)
    assert out["discretionary_purchase_value"] == 300_000.0
    assert out["distinct_buyers"] == 3        # each anonymous row counts as its own buyer (could be one person)


def test_insider_feed_normalisation_not_implemented():
    src = (REPO / "fundamentals/providers.py").read_text()
    assert not re.search(r"form ?4|insider|ownership|sec_form", src, re.I)


@pytest.mark.parametrize("seed", range(100))
def test_insider_random_filings_match_oracle_truncated_db_and_canary(seed):
    rng = random.Random(seed)
    base = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(rng.randrange(3, 14)):
        tdate = base + timedelta(days=rng.randrange(0, 200))
        at = tdate + timedelta(days=rng.randrange(0, 5), seconds=rng.randrange(0, 86400))
        code = rng.choice(["P", "P", "S", "A", "M"])
        rows.append(dict(i=i, tdate=tdate, at=at, code=code, disc=rng.random() < 0.7, value=float(rng.randrange(1, 500) * 1000),
                         insider=rng.choice(["a", "b", "c", None])))
    st = PointInTimeFundamentalStore()
    for r in rows:
        meta = {"transaction_code": r["code"], "discretionary": r["disc"]}
        if r["insider"]:
            meta["insider_id"] = r["insider"]
        st.ingest([Fact("XYZ", f"txn{r['i']}", r["value"], r["tdate"].date().isoformat(), iso(r["at"]), iso(r["at"]), "SEC/form4",
                        f"r{r['i']}", "GBP", "GBP", kind="ownership", metadata=meta)])
    for _ in range(4):
        T = base + timedelta(days=rng.randrange(5, 330), seconds=rng.randrange(0, 86400))
        days = rng.choice([30, 60, 90, 180])
        vis = [r for r in rows if r["at"] <= T and r["at"] >= T - timedelta(days=days)]
        got = insider_activity(st, "XYZ", T, days=days)
        if not vis:
            assert got["status"] == "MISSING", f"seed={seed}"
            continue
        buys = [r for r in vis if r["code"] == "P" and r["disc"]]
        want = dict(buy=sum(r["value"] for r in buys), sell=sum(r["value"] for r in vis if r["code"] == "S"),
                    other=sum(1 for r in vis if not (r["code"] == "P" and r["disc"]) and r["code"] != "S"))
        assert got["discretionary_purchase_value"] == want["buy"], f"seed={seed} T={T} days={days}"
        assert got["reported_sale_value"] == want["sell"] and got["unclassified_transactions"] == want["other"], f"seed={seed}"
        ids = {(r["insider"] or f"anon{r['i']}") for r in buys}
        # anonymous rows are counted per row (limitation); named insiders are de-duplicated
        assert got["distinct_buyers"] == len(ids), f"seed={seed}"
        trunc = truncated_store(st, T)
        assert insider_activity(trunc, "XYZ", T, days=days) == got
        canary = truncated_store(st, datetime(2100, 1, 1, tzinfo=UTC))
        canary.ingest([Fact("XYZ", "canary", float(CANARY), T.date().isoformat(), iso(T + timedelta(seconds=1)), iso(T + timedelta(seconds=1)),
                            "SEC/form4", "c", "GBP", "GBP", kind="ownership", metadata={"transaction_code": "P", "discretionary": True, "insider_id": "z"})])
        assert insider_activity(canary, "XYZ", T, days=days) == got
        assert insider_activity(canary, "XYZ", T + timedelta(seconds=1), days=days)["discretionary_purchase_value"] >= CANARY


# ============================================================================ short interest
def test_short_interest_not_implemented_anywhere():
    pat = re.compile(r"short[_ ]?interest|days[_ ]to[_ ]cover|shortinterest|short_ratio|finra", re.I)
    hits = [str(p.relative_to(REPO)) for p in _py_files() if pat.search(p.read_text())]
    assert hits == [], hits
