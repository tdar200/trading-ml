"""QA track D: macro regime, vintage correctness, availability convention, staleness, units.

Oracle: tests/qa_reaudit/mac_oracle.py (exact Fractions, no fundamentals.* helpers).
Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_mac_macro.py
"""
from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction

import pytest

from fundamentals.models import (CombinedSignalEngine, FundamentalScoreEngine, MacroRegimeService,
                                 ModelArtifact)
from fundamentals.features import FundamentalFeatureEngine
from fundamentals.providers import alfred_snapshot
from fundamentals.store import Fact, PointInTimeFundamentalStore
from tests.qa_reaudit.mac_fixtures import (CANARY, REPO, alfred_csv, canonical, clone_store, ingest_vintage, iso,
                                           month_starts, random_vintage_universe, simple_macro_store,
                                           store_from_universe, truncated_store)
from tests.qa_reaudit.mac_oracle import (SERIES, UTC, VintageArchive, dt, macro_oracle, rel_close)


def state(store, ts):
    return MacroRegimeService(store).state(ts)


def assert_state_matches_oracle(prod, orc, msg=""):
    assert prod["status"] == orc["status"], msg
    assert prod["regime"] == orc["regime"], msg
    assert set(prod["features"]) == set(orc["features"]), msg
    for k, v in orc["features"].items():
        assert rel_close(v, prod["features"][k]), f"{msg} {k}: oracle={v} prod={prod['features'][k]}"


# ====================================================================== hand-computed values
def test_macro_hand_computed_cpi_unrate_fedfunds():
    """Month index i=29 is 2025-06. Hand arithmetic:
    CPI yoy = 310/300-1 = 1/30 ; previous yoy (2025-03 vs 2024-03) = 306/297-1 = 1/33 ; change = 1/330.
    UNRATE 4.5 vs 5.0 three months earlier = -0.5 ; FEDFUNDS 4.25 vs 4.5 = -0.25.
    """
    cpi = {29: 310.0, 17: 300.0, 26: 306.0, 14: 297.0}
    un = {29: 4.5, 26: 5.0}
    ff = {29: 4.25, 26: 4.5}
    st = simple_macro_store(cpi=lambda i: cpi.get(i, 200.0 + i), unrate=lambda i: un.get(i, 4.0),
                            ff=lambda i: ff.get(i, 3.0))
    out = state(st, "2025-07-02T00:00:00Z")
    f = out["features"]
    assert out["status"] == "AVAILABLE"
    assert f["inflation_yoy"] == pytest.approx(1 / 30, rel=1e-12)
    assert f["inflation_change_3m"] == pytest.approx(1 / 330, rel=1e-12)
    assert f["unemployment"] == 4.5 and f["unemployment_change_3m"] == -0.5
    assert f["policy_rate"] == 4.25 and f["policy_rate_change_3m"] == -0.25
    assert out["regime"] == {"inflation": "RISING", "labour_market": "STRENGTHENING", "policy": "EASING"}
    # CPI latest + 3 lags, 2 + 2 for the two rate series = the 8 dependencies reported in the audit
    assert len(out["lineage"]) == 8
    assert out["score"] is None and out["confidence"] is None  # no composite regime score invented


def test_macro_only_known_features_are_published():
    out = state(simple_macro_store(), "2025-07-02T00:00:00Z")
    assert set(out["features"]) == {"inflation_yoy", "inflation_change_3m", "unemployment", "unemployment_change_3m",
                                    "policy_rate", "policy_rate_change_3m"}


# ====================================================================== zero boundary labels
def _tie_store(cur, prior, three, fifteen):
    vals = {29: cur, 17: prior, 26: three, 14: fifteen}
    return simple_macro_store(cpi=lambda i: vals.get(i, 150.0))


@pytest.mark.parametrize("delta,label", [(0.5, "WEAKENING"), (0.0, "UNCHANGED"), (-0.5, "STRENGTHENING")])
def test_unemployment_label_boundary_at_exact_zero(delta, label):
    st = simple_macro_store(unrate=lambda i: {29: 5.0 + delta}.get(i, 5.0))
    out = state(st, "2025-07-02T00:00:00Z")
    assert out["features"]["unemployment_change_3m"] == delta
    assert out["regime"]["labour_market"] == label


@pytest.mark.parametrize("delta,label", [(0.25, "TIGHTENING"), (0.0, "UNCHANGED"), (-0.25, "EASING")])
def test_policy_label_boundary_at_exact_zero(delta, label):
    st = simple_macro_store(ff=lambda i: {29: 4.0 + delta}.get(i, 4.0))
    out = state(st, "2025-07-02T00:00:00Z")
    assert out["regime"]["policy"] == label


def test_inflation_label_exact_tie_is_unchanged():
    # 110/100 and 220/200 are both exactly 1.1 -> change is exactly 0.0
    out = state(_tie_store(110.0, 100.0, 220.0, 200.0), "2025-07-02T00:00:00Z")
    assert out["features"]["inflation_change_3m"] == 0.0
    assert out["regime"]["inflation"] == "UNCHANGED"


def test_inflation_label_rising_and_falling_by_one_basis_point():
    up = state(_tie_store(110.1, 100.0, 220.0, 200.0), "2025-07-02T00:00:00Z")
    dn = state(_tie_store(109.9, 100.0, 220.0, 200.0), "2025-07-02T00:00:00Z")
    assert up["regime"]["inflation"] == "RISING" and dn["regime"]["inflation"] == "FALLING"


def test_QA_MAC_04_inflation_exact_decimal_tie_is_float_noise_rising():
    """CPI 351.961/322.9 and 175.381/160.9 are BOTH exactly 1.09 in exact decimal arithmetic
    (322.9*1.09 = 351.961 ; 160.9*1.09 = 175.381), so the 3-month change in inflation is exactly 0
    and the documented literal rule ('zero-change thresholds have literal meaning') says UNCHANGED."""
    out = state(_tie_store(351.961, 322.9, 175.381, 160.9), "2025-07-02T00:00:00Z")
    assert Fraction("351.961") / Fraction("322.9") == Fraction("175.381") / Fraction("160.9")  # oracle premise
    assert out["regime"]["inflation"] == "UNCHANGED", out["features"]["inflation_change_3m"]


# ====================================================================== staleness boundary
VINTAGE = "2025-07-01"
PUBLIC = datetime(2025, 7, 2, tzinfo=UTC)


@pytest.mark.parametrize("delta,expected", [
    (timedelta(seconds=-1), "INCOMPLETE"),                     # vintage not public yet -> no data at all
    (timedelta(0), "AVAILABLE"),
    (timedelta(days=100), "AVAILABLE"),                        # exactly 100d old
    (timedelta(days=100, hours=23, minutes=59, seconds=59, microseconds=999999), "AVAILABLE"),
    (timedelta(days=101) - timedelta(microseconds=1), "AVAILABLE"),
    (timedelta(days=101), "STALE"),
])
def test_staleness_boundary_is_whole_days_floor_gt_100(delta, expected):
    """max_vintage_age=100: age is floor((T-available_at)/1d) and STALE iff age > 100,
    so the effective cut-off is 101.000 days (a 100.99-day-old vintage is still AVAILABLE)."""
    st = simple_macro_store(vintage=VINTAGE)
    out = state(st, PUBLIC + delta)
    assert out["status"] == expected
    if expected == "STALE":
        assert out["features"] == {} and set(out["regime"].values()) == {"UNKNOWN"}


@pytest.mark.parametrize("max_age", [1, 30, 100, 365])
def test_staleness_threshold_is_parameterised_exactly(max_age):
    st = simple_macro_store(vintage=VINTAGE)
    svc = MacroRegimeService(st, max_vintage_age=max_age)
    assert svc.state(PUBLIC + timedelta(days=max_age))["status"] == "AVAILABLE"
    assert svc.state(PUBLIC + timedelta(days=max_age + 1))["status"] == "STALE"


STATES = ("ok", "stale", "missing")


@pytest.mark.parametrize("cpi_s", STATES)
@pytest.mark.parametrize("un_s", STATES)
@pytest.mark.parametrize("ff_s", STATES)
def test_stale_beats_incomplete_beats_available_matrix(cpi_s, un_s, ff_s):
    st = PointInTimeFundamentalStore()
    mode = {"CPIAUCSL": cpi_s, "UNRATE": un_s, "FEDFUNDS": ff_s}
    for series, m in mode.items():
        if m == "missing":
            continue
        vintage, first, n = ("2025-07-01", date(2023, 1, 1), 30) if m == "ok" else ("2025-02-01", date(2022, 7, 1), 31)
        ms = month_starts(first, n)
        rows = [(d.isoformat(), 100 + i if series == "CPIAUCSL" else 4.0 + (i % 3) * 0.1) for i, d in enumerate(ms)]
        ingest_vintage(st, series, vintage, rows)
    out = state(st, "2025-07-10T00:00:00Z")  # stale vintage is 158 days old, ok vintage 8 days old
    modes = set(mode.values())
    expected = "STALE" if "stale" in modes else "INCOMPLETE" if "missing" in modes else "AVAILABLE"
    assert out["status"] == expected, mode
    produced_series = {k.split("_")[0] for k in out["features"]}
    want = {"inflation" if s == "CPIAUCSL" else "unemployment" if s == "UNRATE" else "policy"
            for s, m in mode.items() if m == "ok"}
    assert produced_series == want, (mode, out["features"])
    for s, label in (("CPIAUCSL", "inflation"), ("UNRATE", "labour_market"), ("FEDFUNDS", "policy")):
        if mode[s] != "ok":
            assert out["regime"][label] == "UNKNOWN"


# ====================================================================== vintage correctness
def test_gdp_like_initial_value_visible_before_revision_vintage():
    """Initial 1.5 (vintage 2025-05-01), revised 2.0 (vintage 2025-06-01) for the SAME period 2025-01-01."""
    st = PointInTimeFundamentalStore()
    for vintage, value in (("2025-05-01", 1.5), ("2025-06-01", 2.0)):
        body = alfred_csv("A191RL1Q225SBEA", vintage, [("2025-01-01", value)])
        st.ingest(alfred_snapshot(body, "A191RL1Q225SBEA", vintage, units="percent"))
    sym = "MACRO:A191RL1Q225SBEA"
    assert st.as_of(sym, "2025-05-01T23:59:59Z", kind="macro") == []
    for ts, want in (("2025-05-02T00:00:00Z", 1.5), ("2025-06-01T23:59:59Z", 1.5), ("2025-06-02T00:00:00Z", 2.0),
                     ("2026-01-01T00:00:00Z", 2.0)):
        rows = st.as_of(sym, ts, kind="macro")
        assert len(rows) == 1 and rows[0]["value"] == want, (ts, rows)
    # both vintages are preserved (append-only), so the audit trail can show the revision
    assert [r["value"] for r in st.history(sym, "A191RL1Q225SBEA")] == [1.5, 2.0]


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_unrate_revision_changes_state_only_after_revision_vintage(order):
    st = PointInTimeFundamentalStore()
    steps = []
    base_months = month_starts(date(2024, 9, 1), 10)  # through 2025-06
    for vintage, latest_value in (("2025-07-01", 4.0), ("2025-08-01", 4.5)):
        rows = [(d.isoformat(), 4.0) for d in base_months[:-1]] + [(base_months[-1].isoformat(), latest_value)]
        steps.append((vintage, rows))
    if order == "reverse":
        steps.reverse()
    for vintage, rows in steps:
        ingest_vintage(st, "UNRATE", vintage, rows)
    assert state(st, "2025-07-15T00:00:00Z")["features"]["unemployment"] == 4.0
    assert state(st, "2025-08-01T23:59:59Z")["features"]["unemployment"] == 4.0
    assert state(st, "2025-08-02T00:00:00Z")["features"]["unemployment"] == 4.5
    # change_3m uses the vintage known at T for the lagged period too (all period values as known at T)
    assert state(st, "2025-08-02T00:00:00Z")["features"]["unemployment_change_3m"] == 0.5


def test_revised_prior_period_is_used_in_change_calculation_only_when_public():
    """A revision to the lagged month (not the latest) must also respect its vintage."""
    st = PointInTimeFundamentalStore()
    ms = month_starts(date(2024, 9, 1), 10)
    rows1 = [(d.isoformat(), 4.0) for d in ms]
    rows2 = [(d.isoformat(), 4.0 if d != ms[-4] else 3.0) for d in ms]   # 2025-03 revised 4.0 -> 3.0
    ingest_vintage(st, "UNRATE", "2025-07-01", rows1)
    ingest_vintage(st, "UNRATE", "2025-08-01", rows2)
    assert state(st, "2025-07-20T00:00:00Z")["features"]["unemployment_change_3m"] == 0.0
    assert state(st, "2025-08-03T00:00:00Z")["features"]["unemployment_change_3m"] == 1.0


# ====================================================================== day-granularity convention
def _release_store():
    """CPI for Feb 2025 released 2025-03-12 08:30 US Eastern = 12:30 UTC (EDT began 2025-03-09)."""
    st = PointInTimeFundamentalStore()
    ms = month_starts(date(2023, 12, 1), 15)   # 2023-12 .. 2025-02
    ingest_vintage(st, "CPIAUCSL", "2025-03-12", [(d.isoformat(), 100 + i) for i, d in enumerate(ms)])
    return st


@pytest.mark.parametrize("ts,visible", [
    ("2025-03-12T12:29:00Z", False),          # one minute BEFORE the real release: correctly invisible
    ("2025-03-12T12:31:00Z", False),          # one minute AFTER the real release: conservatively still invisible
    ("2025-03-12T13:29:00+00:00", False),     # 13:29 / 13:31 as quoted in the brief (UK clock, UTC offset)
    ("2025-03-12T13:31:00+00:00", False),
    ("2025-03-12T13:31:00+01:00", False),     # same instant 12:31Z in a +01:00 rendering
    ("2025-03-12T23:59:59.999999Z", False),   # last microsecond of the vintage day
    ("2025-03-13T00:00:00Z", True),           # vintage day + 1 at 00:00 UTC: first instant visible
    ("2025-03-13T01:00:00+01:00", True),      # the same instant expressed with a +01:00 offset
    ("2025-03-12T20:00:00-04:00", True),      # = 2025-03-13 00:00Z in EDT
    ("2025-03-12T19:59:59-04:00", False),
])
def test_alfred_vintage_becomes_public_at_vintage_day_plus_one_utc(ts, visible):
    """Conservative rule: availability = 00:00 UTC of vintage day + 1. It is never earlier than the real
    release (08:30 ET = 12:30 UTC on 2025-03-12) but up to ~11.5 hours later (plus any time-zone slack)."""
    st = _release_store()
    rows = st.as_of("MACRO:CPIAUCSL", ts, kind="macro")
    assert bool(rows) is visible


def test_macro_value_never_visible_before_real_release_instant():
    """No timestamp earlier than the real release (12:30Z) can ever see the vintage: scan the whole day."""
    st = _release_store()
    start = datetime(2025, 3, 12, tzinfo=UTC)
    release = datetime(2025, 3, 12, 12, 30, tzinfo=UTC)
    for minutes in range(0, 24 * 60, 5):
        ts = start + timedelta(minutes=minutes)
        seen = bool(st.as_of("MACRO:CPIAUCSL", ts, kind="macro"))
        assert not seen, ts
        if ts < release:
            assert not seen


def test_daily_bar_decision_sees_vintage_only_from_next_session():
    """Equity daily bar ts = 04:00 UTC. A vintage published during session day D is usable by bar D+1, not bar D."""
    st = simple_macro_store(vintage="2025-03-12", months=30, first=date(2022, 9, 1))
    assert state(st, "2025-03-12T04:00:00Z")["status"] == "INCOMPLETE"
    assert state(st, "2025-03-13T04:00:00Z")["status"] == "AVAILABLE"


def test_period_end_never_used_as_availability():
    """Observation for Feb 2025 (period_end 2025-02-01) is NOT visible just because period_end passed."""
    st = _release_store()
    assert st.as_of("MACRO:CPIAUCSL", "2025-03-05T00:00:00Z", kind="macro") == []
    # the store refuses facts whose period_end lies after availability
    with pytest.raises(ValueError, match="precedes measured period"):
        Fact("MACRO:X", "X", 1.0, "2025-03-15", "2025-03-01T00:00:00Z", "2025-03-02T00:00:00Z", "ALFRED/graph", "v", "", "index", kind="macro").normalized()


def test_alfred_snapshot_rejects_future_observations_and_wrong_header():
    body = alfred_csv("UNRATE", "2025-03-01", [("2025-04-01", 4.0)])
    with pytest.raises(ValueError, match="future macro observation"):
        alfred_snapshot(body, "UNRATE", "2025-03-01", units="percent")
    with pytest.raises(ValueError, match="wrong/missing ALFRED vintage column"):
        alfred_snapshot(alfred_csv("UNRATE", "2025-03-01", [("2025-02-01", 4.0)]), "UNRATE", "2025-03-02", units="percent")
    with pytest.raises(ValueError):
        alfred_snapshot(b"date,UNRATE_20250301\n2025-02-01,4.0\n", "UNRATE", "2025-03-01", units="percent")


# ====================================================================== units / contracts
@pytest.mark.parametrize("series,units,currency", [
    ("UNRATE", "bps", ""), ("UNRATE", "fraction", ""), ("UNRATE", "index", ""),
    ("FEDFUNDS", "bps", ""), ("FEDFUNDS", "percentage_points", ""),
    ("CPIAUCSL", "percent", ""), ("CPIAUCSL", "fraction", ""),
    ("UNRATE", "percent", "USD"),
])
def test_wrong_units_or_currency_rejected_loudly_at_read(series, units, currency):
    st = simple_macro_store()
    # add a second vintage with wrong units for one series; the service must refuse (not guess)
    body = alfred_csv(series, "2025-08-01", [(d.isoformat(), 1.0) for d in month_starts(date(2023, 1, 1), 31)])
    st.ingest(alfred_snapshot(body, series, "2025-08-01", units=units, currency=currency))
    assert state(st, "2025-07-10T00:00:00Z")["status"] == "AVAILABLE"      # before the bad vintage is public
    with pytest.raises(ValueError, match="incorrect units"):
        state(st, "2025-08-10T00:00:00Z")


def test_two_macro_sources_for_one_series_rejected():
    st = simple_macro_store()
    st.ingest([Fact("MACRO:UNRATE", "UNRATE", 9.9, "2025-06-01", "2025-07-05T00:00:00Z", "2025-07-05T00:00:00Z",
                    "OTHER/provider", "v", "", "percent", kind="macro")])
    with pytest.raises(ValueError, match="ambiguous macro source"):
        state(st, "2025-07-10T00:00:00Z")


def test_percent_vs_fraction_mislabel_is_not_detectable_LIMITATION():
    """FEDFUNDS 0.045 labelled 'percent' (really a fraction) is accepted silently: no plausibility range."""
    st = simple_macro_store(ff=lambda i: 0.045 - i * 0.0001)
    out = state(st, "2025-07-02T00:00:00Z")
    assert out["status"] == "AVAILABLE" and out["features"]["policy_rate"] < 0.05


def test_yield_curve_and_bps_conversions_NOT_IMPLEMENTED():
    src = "\n".join(p.read_text() for p in (REPO / "fundamentals").glob("*.py"))
    for token in ("T10Y2Y", "DGS10", "DGS2", "yield_curve", "term_spread", "bps_to", "to_bps"):
        assert token not in src, token
    out = state(simple_macro_store(), "2025-07-02T00:00:00Z")
    assert not [k for k in out["features"] if "curve" in k or "spread" in k or "10y" in k.lower()]


# ====================================================================== provider fail-safe
def test_alfred_missing_markers_skipped_never_filled():
    st = PointInTimeFundamentalStore()
    body = b"observation_date,UNRATE_20250701\n2025-04-01,4.1\n2025-05-01,.\n2025-06-01,\n"
    facts = alfred_snapshot(body, "UNRATE", "2025-07-01", units="percent")
    assert [f.period_end for f in facts] == ["2025-04-01"]
    assert st.ingest(facts) == 1


@pytest.mark.parametrize("cell", ["inf", "-inf", "Infinity"])
def test_alfred_nonfinite_value_aborts_whole_vintage_atomically(cell):
    st = PointInTimeFundamentalStore()
    body = f"observation_date,UNRATE_20250701\n2025-04-01,4.1\n2025-05-01,{cell}\n".encode()
    with pytest.raises(ValueError, match="finite"):
        st.ingest(alfred_snapshot(body, "UNRATE", "2025-07-01", units="percent"))
    assert st.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0


def test_alfred_text_cells_such_as_percent_sign_are_silently_dropped_LIMITATION():
    st = PointInTimeFundamentalStore()
    body = b"observation_date,UNRATE_20250701\n2025-04-01,4.1%\n2025-05-01,4.2%\n"
    assert alfred_snapshot(body, "UNRATE", "2025-07-01", units="percent") == []
    assert st.ingest([]) == 0     # silent zero rows: later reads show INCOMPLETE, never a fabricated value
    assert state(st, "2025-07-10T00:00:00Z")["status"] == "INCOMPLETE"


def test_alfred_duplicate_observation_dates_poison_reads_loudly_LIMITATION():
    st = PointInTimeFundamentalStore()
    body = b"observation_date,UNRATE_20250701\n2025-04-01,4.1\n2025-04-01,4.9\n"
    st.ingest(alfred_snapshot(body, "UNRATE", "2025-07-01", units="percent"))
    with pytest.raises(ValueError, match="ambiguous conflicting source facts"):
        st.as_of("MACRO:UNRATE", "2025-07-10T00:00:00Z", kind="macro")


def test_alfred_withdrawn_observation_cannot_be_expressed_LIMITATION():
    """A later vintage showing '.' for a period that an earlier vintage had does NOT withdraw it."""
    st = PointInTimeFundamentalStore()
    ms = month_starts(date(2024, 9, 1), 10)
    ingest_vintage(st, "UNRATE", "2025-07-01", [(d.isoformat(), 4.0) for d in ms])
    rows = [(d.isoformat(), "." if d == ms[-1] else 4.0) for d in ms]
    ingest_vintage(st, "UNRATE", "2025-08-01", rows)
    latest = max(st.as_of("MACRO:UNRATE", "2025-08-05T00:00:00Z", kind="macro"), key=lambda r: r["period_end"])
    assert latest["period_end"] == ms[-1].isoformat() and latest["value"] == 4.0   # stale value survives


def test_ingest_alfred_requires_explicit_units_for_unmapped_series(tmp_path):
    from fundamentals.providers import ingest_alfred
    st = PointInTimeFundamentalStore()
    with pytest.raises(ValueError, match="explicit units"):
        ingest_alfred(st, "GDPC1", "2025-07-01", tmp_path)


# ====================================================================== randomized differential + PHASE 98
SEEDS = list(range(120))


@pytest.mark.parametrize("seed", SEEDS)
def test_macro_random_vintages_match_oracle_truncated_db_and_future_canary(seed):
    universe = random_vintage_universe(seed)
    rng = random.Random(1000 + seed)
    archive = VintageArchive.from_memory(universe)
    full = store_from_universe(universe)
    first_public = min(VintageArchive.public_from(v) for s in SERIES for v, _ in universe[s])
    last_public = max(VintageArchive.public_from(v) for s in SERIES for v, _ in universe[s])
    span = int((last_public - first_public).total_seconds()) + 160 * 86400
    for k in range(4):
        ts = first_public - timedelta(days=2) + timedelta(seconds=rng.randrange(span))
        prod = state(full, ts)
        orc = macro_oracle({s: archive.known_at(s, ts) for s in SERIES}, ts)
        assert_state_matches_oracle(prod, orc, f"seed={seed} ts={ts}")
        # lineage: exactly the (series, period, vintage) rows the oracle used, all public at ts
        got = set()
        for rid in prod["lineage"]:
            r = full.db.execute("SELECT symbol,period_end,revision,available_at FROM observations WHERE id=?", (rid,)).fetchone()
            assert dt(r["available_at"]) <= ts, f"seed={seed} lineage row public after ts"
            got.add((r["symbol"].split(":")[1], date.fromisoformat(r["period_end"]), date.fromisoformat(r["revision"])))
        assert got == orc["used"], f"seed={seed} ts={ts}"
        # PHASE 98 for macro: physically truncated DB gives byte-identical canonical JSON
        trunc = truncated_store(full, ts)
        assert trunc.db.execute("SELECT MAX(available_at) FROM observations").fetchone()[0] is None or \
            trunc.db.execute("SELECT MAX(available_at) FROM observations").fetchone()[0] <= iso(ts)
        assert canonical(state(trunc, ts)) == canonical(prod), f"seed={seed} ts={ts} truncated DB differs"
        # future canary vintage (published AFTER ts) must never leak
        taken = {v for s_ in SERIES for v, _ in universe[s_]}
        v = ts.date()  # public from v+1 00:00 UTC > ts
        while v in taken:
            v += timedelta(days=1)
        months = month_starts(date(2018, 1, 1), 100)
        extra = [(s_, v.isoformat(), [(d.isoformat(), CANARY) for d in months if d <= v]) for s_ in SERIES]
        canary = clone_store(full)
        for s_, vint, rows in extra:
            ingest_vintage(canary, s_, vint, rows)
        out = canonical(state(canary, ts))
        assert out == canonical(prod), f"seed={seed} ts={ts} canary changed the state"
        assert str(CANARY) not in out
        # control: once public the canary IS visible, so the absence above is meaningful
        assert str(CANARY) in canonical(state(canary, VintageArchive.public_from(v))), f"seed={seed} canary not wired"


def test_macro_canary_actually_reaches_state_once_public():
    """Control for the canary tests above: the canary vintage DOES change the state once public."""
    st = simple_macro_store(vintage="2025-07-01")
    ms = [d for d in month_starts(date(2022, 1, 1), 60) if d <= date(2025, 7, 20)]
    ingest_vintage(st, "UNRATE", "2025-07-20", [(d.isoformat(), CANARY) for d in ms])
    before = state(st, "2025-07-20T23:59:59Z")["features"]["unemployment"]
    after = state(st, "2025-07-21T00:00:00Z")["features"]["unemployment"]
    assert before != CANARY and after == CANARY


# ====================================================================== policy interplay (macro status)
def _model(features, coefficients, intercept=0.25):
    n = len(features)
    return ModelArtifact(tuple(features), tuple([0.0] * n), tuple([1.0] * n), tuple(coefficients), intercept,
                         "2020-01-01T00:00:00+00:00", 1.0, 100, "x" * 8, 5)


def _engine(store, features, coefficients, policy, *, with_macro=True, shadow=True, threshold=0.0):
    fe = FundamentalFeatureEngine(store)
    macro = MacroRegimeService(store) if with_macro else None
    return CombinedSignalEngine(fe, FundamentalScoreEngine(_model(features, coefficients)), macro,
                                policy=policy, threshold=threshold, research_shadow=shadow)


MODEL_FEATURES = ("macro_unemployment_change_3m", "technical_signal")
MODEL_COEF = (2.0, 0.5)  # score = 0.25 + 2.0*0.5 + 0.5*1.0 = 1.75 exactly


def _policy_store(mode):
    """mode: dict series -> ok/stale/missing; UNRATE change fixed at +0.5."""
    st = PointInTimeFundamentalStore()
    for series, m in mode.items():
        if m == "missing":
            continue
        vintage, first, n = ("2025-07-01", date(2023, 1, 1), 30) if m == "ok" else ("2025-02-01", date(2022, 7, 1), 31)
        ms = month_starts(first, n)
        if series == "UNRATE":
            rows = [(d.isoformat(), 4.5 if i == n - 1 else 4.0) for i, d in enumerate(ms)]
        else:
            rows = [(d.isoformat(), 100 + i if series == "CPIAUCSL" else 4.0) for i, d in enumerate(ms)]
        ingest_vintage(st, series, vintage, rows)
    return st


AT = "2025-07-10T04:00:00Z"


@pytest.mark.parametrize("policy", ["filter", "selection", "weighted", "regime"])
def test_policy_all_macro_available_permits_with_hand_computed_score(policy):
    st = _policy_store({"CPIAUCSL": "ok", "UNRATE": "ok", "FEDFUNDS": "ok"})
    d = _engine(st, MODEL_FEATURES, MODEL_COEF, policy).decide("XYZ", AT, {"signal": True})
    assert d["macro"]["status"] == "AVAILABLE"
    assert d["fundamental"]["score"] == 1.75
    assert d["permit_signal"] is True and d["decision"] == "RESEARCH_CANDIDATE"
    assert d["explanation"] == ["Fitted research prediction exceeds frozen threshold"]
    # same snapshot without a technical signal: never a trade
    d2 = _engine(st, MODEL_FEATURES, MODEL_COEF, policy).decide("XYZ", AT, {"signal": False})
    assert d2["permit_signal"] is False and d2["decision"] == "NO_TRADE"


def test_policy_regime_requires_available_macro_but_filter_does_not_CHARACTERISATION():
    st = _policy_store({"CPIAUCSL": "missing", "UNRATE": "ok", "FEDFUNDS": "ok"})  # macro INCOMPLETE
    reg = _engine(st, MODEL_FEATURES, MODEL_COEF, "regime").decide("XYZ", AT, {"signal": True})
    assert reg["macro"]["status"] == "INCOMPLETE" and reg["permit_signal"] is False
    assert reg["explanation"] == ["Macro vintage coverage missing/stale"]
    flt = _engine(st, MODEL_FEATURES, MODEL_COEF, "filter").decide("XYZ", AT, {"signal": True})
    # documented behaviour: only the 'regime' policy gates on macro status; filter trusts per-feature coverage
    assert flt["macro"]["status"] == "INCOMPLETE" and flt["permit_signal"] is True


def test_policy_stale_series_needed_by_model_fails_closed_for_every_policy():
    st = _policy_store({"CPIAUCSL": "ok", "UNRATE": "stale", "FEDFUNDS": "ok"})
    for policy in ("filter", "selection", "weighted", "regime"):
        d = _engine(st, MODEL_FEATURES, MODEL_COEF, policy).decide("XYZ", AT, {"signal": True})
        assert d["macro"]["status"] == "STALE"
        assert d["fundamental"]["score"] is None and d["fundamental"]["status"] == "MISSING_FEATURES"
        assert d["fundamental"]["missing"] == ["macro_unemployment_change_3m"]
        assert d["permit_signal"] is False and d["decision"] == "NO_TRADE"


def test_policy_stale_series_not_needed_still_blocks_regime_only():
    st = _policy_store({"CPIAUCSL": "ok", "UNRATE": "ok", "FEDFUNDS": "stale"})
    assert _engine(st, MODEL_FEATURES, MODEL_COEF, "filter").decide("XYZ", AT, {"signal": True})["permit_signal"] is True
    reg = _engine(st, MODEL_FEATURES, MODEL_COEF, "regime").decide("XYZ", AT, {"signal": True})
    assert reg["permit_signal"] is False and reg["macro"]["status"] == "STALE"


def test_policy_without_macro_engine_never_invents_macro_features():
    st = _policy_store({"CPIAUCSL": "ok", "UNRATE": "ok", "FEDFUNDS": "ok"})
    for policy in ("filter", "regime"):
        d = _engine(st, MODEL_FEATURES, MODEL_COEF, policy, with_macro=False).decide("XYZ", AT, {"signal": True})
        assert d["macro"]["status"] == "MISSING" and d["permit_signal"] is False
        assert d["fundamental"]["score"] is None


def test_policy_observe_ignores_macro_completely():
    st = _policy_store({"CPIAUCSL": "missing", "UNRATE": "missing", "FEDFUNDS": "missing"})
    d = _engine(st, MODEL_FEATURES, MODEL_COEF, "observe").decide("XYZ", AT, {"signal": True})
    assert d["permit_signal"] is True and d["decision"] == "TECHNICAL_CANDIDATE"


def test_decision_data_snapshot_id_ignores_future_macro_vintage():
    st = _policy_store({"CPIAUCSL": "ok", "UNRATE": "ok", "FEDFUNDS": "ok"})
    eng = _engine(st, MODEL_FEATURES, MODEL_COEF, "filter")
    before = eng.decide("XYZ", AT, {"signal": True})
    ms = [d for d in month_starts(date(2022, 1, 1), 60) if d <= date(2025, 7, 10)]
    ingest_vintage(st, "UNRATE", "2025-07-10", [(d.isoformat(), CANARY) for d in ms])   # public 07-11 00:00 > AT
    after = eng.decide("XYZ", AT, {"signal": True})
    assert canonical(before) == canonical(after) and str(CANARY) not in canonical(after)
    assert after["data_snapshot_id"] == before["data_snapshot_id"]
    late = eng.decide("XYZ", "2025-07-11T04:00:00Z", {"signal": True})
    assert late["data_snapshot_id"] != before["data_snapshot_id"]


# ====================================================================== ingest_alfred (fake HTTP, no network)
class _Resp:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(bodies, calls):
    def fake(req, timeout=None):
        calls.append(req.full_url)
        return _Resp(bodies[min(len(calls) - 1, len(bodies) - 1)])
    return fake


GOOD_BODY = b"observation_date,UNRATE_20250701\n2025-04-01,4.1\n2025-05-01,4.2\n"


def test_ingest_alfred_request_cache_and_archive(monkeypatch, tmp_path):
    from fundamentals import providers
    calls = []
    monkeypatch.setattr(providers, "urlopen", _fake_urlopen([GOOD_BODY], calls))
    st = PointInTimeFundamentalStore()
    assert providers.ingest_alfred(st, "UNRATE", "2025-07-01", tmp_path) == 2
    assert calls == ["https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=UNRATE&cosd=2022-07-01&coed=2025-07-01&vintage_date=2025-07-01"]
    assert st.db.execute("SELECT COUNT(*) FROM raw_documents").fetchone()[0] == 1
    assert providers.ingest_alfred(st, "UNRATE", "2025-07-01", tmp_path) == 2     # served from cache, idempotent
    assert len(calls) == 1 and st.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 2


def test_QA_MAC_06_transient_bad_body_must_not_poison_the_vintage_cache(monkeypatch, tmp_path):
    from fundamentals import providers
    calls = []
    monkeypatch.setattr(providers, "urlopen", _fake_urlopen([b"<html>rate limited</html>", GOOD_BODY], calls))
    st = PointInTimeFundamentalStore()
    with pytest.raises(ValueError):
        providers.ingest_alfred(st, "UNRATE", "2025-07-01", tmp_path)
    assert providers.ingest_alfred(st, "UNRATE", "2025-07-01", tmp_path) == 2      # retry should refetch and succeed


def test_macro_withdrawal_is_expressible_through_null_fact_but_not_through_alfred_csv():
    """A later vintage with value=None (JSON/Fact route) removes the period from the macro state."""
    st = PointInTimeFundamentalStore()
    ms = month_starts(date(2024, 9, 1), 10)
    ingest_vintage(st, "UNRATE", "2025-07-01", [(d.isoformat(), 4.0) for d in ms])
    st.ingest([Fact("MACRO:UNRATE", "UNRATE", None, ms[-1].isoformat(), "2025-08-01T00:00:00Z", "2025-08-02T00:00:00Z",
                    "ALFRED/graph", "2025-08-01", "", "percent", kind="macro")])
    before = state(st, "2025-07-20T00:00:00Z")["features"]
    after = state(st, "2025-08-05T00:00:00Z")["features"]
    assert before["unemployment"] == 4.0 and after["unemployment"] == 4.0      # latest is now ms[-2], also 4.0
    latest = max((r for r in st.as_of("MACRO:UNRATE", "2025-08-05T00:00:00Z", kind="macro") if r["value"] is not None),
                 key=lambda r: r["period_end"])
    assert latest["period_end"] == ms[-2].isoformat()


# ====================================================================== provider-trust and config validation
def test_monthly_average_stamped_at_month_start_inside_its_own_month_is_accepted_LIMITATION():
    """The ingestion gate only checks observation_date <= vintage. A (wrong) vintage dated 2025-03-15 that already
    contains the March monthly average (stamped 2025-03-01) is accepted and visible on 2025-03-16, two weeks
    before March ends. Real ALFRED never does this (see test_real_data_never_appears_in_a_vintage_...)."""
    st = PointInTimeFundamentalStore()
    ms = month_starts(date(2024, 9, 1), 7)    # 2024-09 .. 2025-03
    ingest_vintage(st, "FEDFUNDS", "2025-03-15", [(d.isoformat(), 4.33) for d in ms])
    rows = st.as_of("MACRO:FEDFUNDS", "2025-03-16T00:00:00Z", kind="macro")
    assert "2025-03-01" in {r["period_end"] for r in rows}


def test_QA_MAC_07_nonfinite_max_vintage_age_must_be_rejected():
    st = simple_macro_store(vintage=VINTAGE)
    try:
        svc = MacroRegimeService(st, max_vintage_age=float("nan"))
    except ValueError:
        return
    # fail-open: a 5-year-old vintage is still reported AVAILABLE
    assert svc.state(PUBLIC + timedelta(days=5 * 365))["status"] != "AVAILABLE"


def test_paper_engine_daily_bar_sees_macro_vintage_and_news_only_after_availability():
    """PaperEngine decides at as_of = bar ts (04:00 UTC session start). A UNRATE vintage dated 2025-07-01 (public
    2025-07-02 00:00 UTC) and an article available 2025-07-01 14:00 UTC are invisible to the 2025-07-01 04:00 bar
    and visible to the 2025-07-02 04:00 bar."""
    from livesim.small_account import PaperEngine
    from fundamentals.models import EventCatalystEngine
    from tests.qa_reaudit.mac_fixtures import add_event

    def build():
        st = PointInTimeFundamentalStore()
        ms = month_starts(date(2023, 1, 1), 30)
        ingest_vintage(st, "UNRATE", "2025-07-01", [(d.isoformat(), 4.5 if i == 29 else 4.0) for i, d in enumerate(ms)])
        add_event(st, event_type="document_annotation", event_at="2025-07-01T13:55:00Z", available_at="2025-07-01T14:00:00Z",
                  payload={"event_key": "n1"})
        fe = FundamentalFeatureEngine(st)
        model = _model(("macro_unemployment_change_3m", "technical_signal"), MODEL_COEF)
        return CombinedSignalEngine(fe, FundamentalScoreEngine(model), MacroRegimeService(st), EventCatalystEngine(st),
                                    policy="weighted", threshold=0.0, research_shadow=True)

    def run(bar, prev):
        eng = PaperEngine(symbol="XYZ", shadow=True, fee_bps=0, spread_bps=0, slip_bps=0, fundamental_filter=build())
        eng.step(dict(ts=bar, open=100, high=101, low=99, close=100),
                 dict(ts=prev, signal=True, distance=2.0, regime="fixture"))
        decision = next(x["decision"] for x in eng.journal if x["event"] == "FUNDAMENTAL_DECISION")
        return eng, decision

    early, d1 = run("2025-07-01T04:00:00Z", "2025-06-30T04:00:00Z")
    assert d1["macro"]["features"] == {} and d1["catalysts"]["events"] == []
    assert d1["fundamental"]["score"] is None and d1["permit_signal"] is False
    assert early.state["position"] is None and early.state["cash"] == 100

    late, d2 = run("2025-07-02T04:00:00Z", "2025-07-01T04:00:00Z")
    assert d2["macro"]["features"]["unemployment_change_3m"] == 0.5
    assert [e["payload"]["event_key"] for e in d2["catalysts"]["events"]] == ["n1"]
    assert d2["fundamental"]["score"] == 1.75 and d2["permit_signal"] is True
    assert late.state["position"] is not None
    # the decision object records the snapshot id and configuration used for the order
    assert len(d2["data_snapshot_id"]) == 64 and d2["configuration"]["policy"] == "weighted"
