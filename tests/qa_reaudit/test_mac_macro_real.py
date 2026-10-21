"""QA track D: macro on the REAL ALFRED vintage archive (results/fundamentals/raw/*.csv + stored DB copy).

The stored DB is only ever opened through a sqlite backup copy in tmp_path. The oracle parses the raw CSVs
directly (tests/qa_reaudit/mac_oracle.py) and never looks at the DB.
Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_mac_macro_real.py
"""
from __future__ import annotations

import json
import random
import threading
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from fundamentals.api import FundamentalAPI, serve
from fundamentals.models import MacroRegimeService
from fundamentals.store import PointInTimeFundamentalStore
from tests.qa_reaudit.mac_fixtures import REPO, canonical, copy_real_db, iso
from tests.qa_reaudit.mac_oracle import (RAW_ALFRED_DIR, SERIES, UTC, VintageArchive, dt, macro_oracle, month_shift,
                                         rel_close)

OUT = REPO / "results/qa_reaudit_20261001/mac"
REAL_DB = REPO / "results/fundamentals/fundamentals.sqlite"

pytestmark = pytest.mark.skipif(not REAL_DB.exists() or not (REPO / RAW_ALFRED_DIR).exists(),
                                reason="real ALFRED archive not present")


@pytest.fixture(scope="module")
def archive():
    return VintageArchive.from_dir(REPO / RAW_ALFRED_DIR)


@pytest.fixture(scope="module")
def real_path(tmp_path_factory):
    return copy_real_db(tmp_path_factory.mktemp("macreal") / "real.sqlite")


@pytest.fixture(scope="module")
def real(real_path):
    st = PointInTimeFundamentalStore(real_path, readonly=True)
    yield st
    st.close()


def oracle_at(archive, ts):
    return macro_oracle({s: archive.known_at(s, ts) for s in SERIES}, ts)


# ---------------------------------------------------------------------------------- ingestion fidelity
def test_real_archive_is_26_quarterly_vintages_per_series(archive):
    for s in SERIES:
        vs = [v for v, _ in archive.v[s]]
        assert len(vs) == 26 and vs[0] == date(2020, 4, 1) and vs[-1] == date(2026, 7, 1)
        assert all(date(*month_shift(a, 3), 1) == b for a, b in zip(vs, vs[1:]))


def test_real_db_macro_rows_equal_raw_csv_cells(real, archive):
    rows = real.db.execute("SELECT symbol,period_end,revision,value,available_at,reported_at,units,currency,source,"
                           "kind,metadata FROM observations WHERE kind='macro'").fetchall()
    got = {}
    for r in rows:
        key = (r["symbol"].split(":")[1], r["period_end"], r["revision"])
        assert key not in got
        got[key] = r
    want = {}
    for s in SERIES:
        for vintage, data in archive.v[s]:
            for obs, val in data.items():
                want[(s, obs.isoformat(), vintage.isoformat())] = (float(val), vintage)
    assert set(got) == set(want), (len(got), len(want))
    for key, (val, vintage) in want.items():
        r = got[key]
        assert r["value"] == val, key
        assert r["available_at"] == iso(VintageArchive.public_from(vintage)), key
        assert r["reported_at"] == iso(datetime(vintage.year, vintage.month, vintage.day, tzinfo=UTC)), key
        assert r["units"] == ("index" if key[0] == "CPIAUCSL" else "percent") and r["currency"] == ""
        assert r["source"] == "ALFRED/graph"
        meta = json.loads(r["metadata"])
        assert meta["not_exact_release_time"] is True and meta["timing_precision"] == "vintage_day+1UTC"


def test_real_data_never_appears_in_a_vintage_before_the_measured_month_ends(archive):
    """Real-world release check: for every observation the first vintage containing it is dated after
    the end of the observation month, so stamping at month START never leaks a not-yet-complete month."""
    for s in SERIES:
        for obs, (val, vint) in archive.first_vintage_values(s).items():
            if vint == date(2020, 4, 1):
                continue
            y, m = month_shift(obs, 1)
            month_end = date(y, m, 1) - timedelta(days=1)
            assert vint > month_end, (s, obs, vint)


# ---------------------------------------------------------------------------------- state vs oracle
LO = datetime(2020, 3, 1, tzinfo=UTC)
HI = datetime(2026, 10, 2, tzinfo=UTC)


@pytest.mark.parametrize("seed", range(8))
def test_real_macro_state_matches_independent_oracle(real, archive, seed):
    rng = random.Random(seed)
    svc = MacroRegimeService(real)
    for k in range(50):
        ts = LO + timedelta(seconds=rng.randrange(int((HI - LO).total_seconds())))
        prod, orc = svc.state(ts), oracle_at(archive, ts)
        tag = f"seed={seed} k={k} ts={ts}"
        assert prod["status"] == orc["status"], tag
        assert prod["regime"] == orc["regime"], tag
        assert set(prod["features"]) == set(orc["features"]), tag
        for name, v in orc["features"].items():
            assert rel_close(v, prod["features"][name]), f"{tag} {name} {v} {prod['features'][name]}"
        if k % 5 == 0:
            got = set()
            for rid in prod["lineage"]:
                r = real.db.execute("SELECT symbol,period_end,revision,available_at FROM observations WHERE id=?", (rid,)).fetchone()
                assert dt(r["available_at"]) <= ts, tag
                got.add((r["symbol"].split(":")[1], date.fromisoformat(r["period_end"]), date.fromisoformat(r["revision"])))
            assert got == orc["used"], tag


def test_real_state_flips_exactly_at_each_vintage_availability_instant(real, archive):
    svc = MacroRegimeService(real)
    for vintage in [v for v, _ in archive.v["CPIAUCSL"]]:
        edge = VintageArchive.public_from(vintage)
        before, after = svc.state(edge - timedelta(microseconds=1)), svc.state(edge)
        for ts, prod in ((edge - timedelta(microseconds=1), before), (edge, after)):
            orc = oracle_at(archive, ts)
            assert prod["status"] == orc["status"] and prod["regime"] == orc["regime"], (vintage, ts)
            for name, v in orc["features"].items():
                assert rel_close(v, prod["features"][name]), (vintage, name)
        assert canonical(before) != canonical(after), vintage      # the new vintage is the only thing that changed


def test_real_vintage_staircase_every_observation_every_boundary(real, archive):
    """At each vintage boundary (+/- 1 microsecond) the store returns, for EVERY observation of EVERY series,
    exactly the value of the latest public vintage that contains it (so revisions are never retroactive)."""
    checked = 0
    for vintage in [v for v, _ in archive.v["UNRATE"]]:
        edge = VintageArchive.public_from(vintage)
        for ts in (edge - timedelta(microseconds=1), edge):
            for s in SERIES:
                known = archive.known_at(s, ts)
                rows = {r["period_end"]: r for r in real.as_of("MACRO:" + s, ts, kind="macro")}
                assert set(rows) == {o.isoformat() for o in known}, (s, ts)
                for obs, (val, vint) in known.items():
                    r = rows[obs.isoformat()]
                    assert r["value"] == float(val) and r["revision"] == vint.isoformat(), (s, obs, ts)
                    checked += 1
    assert checked > 10000


def test_real_cpi_revision_examples_show_original_value_until_revision_vintage(real, archive):
    first, final = archive.first_vintage_values("CPIAUCSL"), archive.final_vintage_values("CPIAUCSL")
    revised = sorted(((abs(float(final[o][0] - first[o][0])), o) for o in first
                      if final[o][0] != first[o][0] and first[o][1] > date(2020, 4, 1)), reverse=True)[:5]
    assert revised and revised[0][0] > 0.5            # largest seasonal-adjustment revision ~0.68 index points
    for mag, obs in revised:
        v0 = first[obs][1]
        at_first = real.as_of("MACRO:CPIAUCSL", VintageArchive.public_from(v0), kind="macro")
        old = [r for r in at_first if r["period_end"] == obs.isoformat()][0]["value"]
        assert old == float(first[obs][0])
        late = real.as_of("MACRO:CPIAUCSL", datetime(2026, 9, 1, tzinfo=UTC), kind="macro")
        new = [r for r in late if r["period_end"] == obs.isoformat()]
        if new:
            assert new[0]["value"] == float(final[obs][0]) and new[0]["value"] != old


# ---------------------------------------------------------------------------------- phase 106
def research_timestamps():
    import pandas as pd
    frame = pd.read_parquet(REPO / "results/small_account/SPY.parquet", columns=["ts"])
    return [pd.Timestamp(t).to_pydatetime() for t in frame.ts]


def test_phase106_research_path_uses_vintage_state_per_decision_timestamp(real, archive):
    """research.run_etf_experiment computes macro.state(t) per decision bar from the vintage store.
    Reproduce that on the real ETF bar timestamps: it equals the vintage-only oracle and differs from a
    latest-revised variant, so the research path is vintage-based (no revised-value leakage)."""
    src = (REPO / "fundamentals/research.py").read_text()
    assert "macro=MacroRegimeService(store)" in src and "states={t:macro.state(t)" in src
    assert "fred" not in src.lower() and "urlopen" not in src and "read_csv" not in src
    ts_all = research_timestamps()
    final = {s: archive.final_vintage_values(s) for s in SERIES}
    svc = MacroRegimeService(real)
    differ_from_revised = 0
    for i, ts in enumerate(ts_all[::24]):          # every 24th bar: ~61 research decision timestamps
        prod = svc.state(ts)
        orc = oracle_at(archive, ts)
        assert prod["status"] == orc["status"] and prod["regime"] == orc["regime"], ts
        kn = {s: archive.known_at(s, ts) for s in SERIES}
        revised = macro_oracle({s: {o: (final[s][o][0], kn[s][o][1]) for o in kn[s]} for s in SERIES}, ts)
        if any(not rel_close(revised["features"].get(k), prod["features"].get(k)) for k in prod["features"]):
            differ_from_revised += 1
    assert differ_from_revised > 10


def test_phase106_quantify_vintage_vs_latest_revised_and_staleness(archive):
    """Quantification from the stored data (written to results/qa_reaudit_20261001/mac/macro_leakage.json).
    NOT a leakage in the research path (it uses vintages); it quantifies what a latest-revised FRED pull would
    have changed and how coarse/stale the quarterly vintage grid is."""
    final = {s: archive.final_vintage_values(s) for s in SERIES}
    ts_list = [t for t in research_timestamps() if t >= datetime(2020, 4, 2, tzinfo=UTC)]
    flips = {"inflation": 0, "labour_market": 0, "policy": 0}
    max_dyoy = 0.0
    lag_days = {s: [] for s in SERIES}
    n = 0
    for ts in ts_list:
        kn = {s: archive.known_at(s, ts) for s in SERIES}
        a = macro_oracle(kn, ts)
        r = macro_oracle({s: {o: (final[s][o][0], kn[s][o][1]) for o in kn[s]} for s in SERIES}, ts)
        n += 1
        for k in flips:
            flips[k] += a["regime"][k] != r["regime"][k]
        if a["features"].get("inflation_yoy") is not None and r["features"].get("inflation_yoy") is not None:
            max_dyoy = max(max_dyoy, abs(float(a["features"]["inflation_yoy"] - r["features"]["inflation_yoy"])))
        for s in SERIES:
            lag_days[s].append((ts.date() - max(kn[s])).days)
    report = {
        "research_decision_timestamps": n,
        "label_flips_if_latest_revised_values_were_used": flips,
        "label_flip_rates": {k: v / n for k, v in flips.items()},
        "max_abs_inflation_yoy_difference": max_dyoy,
        "lag_days_between_decision_and_latest_period_start": {s: {"min": min(v), "max": max(v),
                                                                  "median": sorted(v)[len(v) // 2]} for s, v in lag_days.items()},
        "note": "Research path uses vintage snapshots (no leakage); quarterly grid makes macro state coarse and up to ~5 months old.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "macro_leakage.json").write_text(json.dumps(report, indent=2))
    assert flips["policy"] == 0                      # FEDFUNDS is never revised
    assert flips["labour_market"] > flips["inflation"] > 0
    assert max_dyoy < 0.002                          # CPI seasonal revisions move YoY by at most ~11 bp
    assert max(lag_days["CPIAUCSL"]) >= 120          # vintage grid is quarterly: macro can be 4+ months behind


# ---------------------------------------------------------------------------------- API
def test_macro_regime_api_and_http_equal_state_and_have_no_nan(real_path, real):
    ts = "2025-07-02T04:00:00Z"
    expected = json.loads(json.dumps(MacroRegimeService(real).state(ts), allow_nan=False))
    assert json.loads(json.dumps(FundamentalAPI(real).get("/macro/regime", {"as_of": ts}), allow_nan=False)) == expected
    srv = serve(str(real_path), port=0)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}/macro/regime?as_of={ts}", timeout=10)
        body = r.read().decode()
        assert r.status == 200 and r.headers["Cache-Control"] == "no-store"
        assert "NaN" not in body and "Infinity" not in body
        assert json.loads(body) == expected
        early = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/macro/regime?as_of=2020-04-01T23:59:59Z", timeout=10).read())
        assert early["status"] == "INCOMPLETE" and early["features"] == {}
    finally:
        srv.shutdown()
        srv.server_close()
