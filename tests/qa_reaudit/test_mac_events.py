"""QA track D: event/catalyst engine (surprise, decay, visibility, versioning, dedupe, wildcard, robustness).

Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_mac_events.py
"""
from __future__ import annotations

import json
import random
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta

import pytest

from fundamentals.api import FundamentalAPI, serve
from fundamentals.models import EventCatalystEngine
from fundamentals.store import PointInTimeFundamentalStore
from tests.qa_reaudit.mac_fixtures import CANARY, REPO, add_event, canonical, iso, truncated_store
from tests.qa_reaudit.mac_oracle import UTC, dt, decay, eps_surprise

EVENT_AT = "2025-05-01T13:00:00Z"
AVAIL = "2025-05-01T14:00:00Z"
AFTER = "2025-05-02T00:00:00Z"


def ev_state(store, ts=AFTER, symbol="XYZ", half_life=None):
    return EventCatalystEngine(store, half_life_days=half_life).state(symbol, ts)


def earnings(store, actual, consensus, consensus_at="2025-04-30T00:00:00Z", **kw):
    payload = {"reported_eps": actual, "consensus_eps": consensus}
    if consensus_at is not None:
        payload["consensus_available_at"] = consensus_at
    return add_event(store, event_at=EVENT_AT, available_at=AVAIL, payload=payload, **kw)


# ============================================================================ EPS surprise
@pytest.mark.parametrize("actual,consensus,expected", [
    (1.10, 1.00, 0.10), (0.90, 1.00, -0.10), (2.50, 2.50, 0.0), (0.5, 0.25, 1.0),
    (-0.05, -0.10, 0.5),      # loss narrower than expected -> positive surprise, |consensus| denominator
    (-0.15, -0.10, -0.5),     # loss wider -> negative
    (0.05, -0.10, 1.5),       # swing to profit
    (1.0, 0.0, None), (0.0, 0.0, None),    # zero consensus -> undefined, never inf/0
    (1e-9, 1e-12, 999.0),     # micro consensus stays finite
])
def test_eps_surprise_matches_exact_oracle(actual, consensus, expected):
    st = PointInTimeFundamentalStore()
    earnings(st, actual, consensus)
    e = ev_state(st)["events"][0]
    want = eps_surprise(actual, consensus)
    if expected is None:
        assert e["eps_surprise"] is None and want is None
    else:
        assert want == pytest.approx(expected, rel=1e-12)
        assert e["eps_surprise"] == pytest.approx(want, rel=1e-12, abs=1e-15)
    assert e["payload"]["reported_eps"] == actual and e["payload"]["consensus_eps"] == consensus  # raw inputs preserved


def test_missing_consensus_never_becomes_zero_surprise_and_is_not_flagged_LIMITATION():
    st = PointInTimeFundamentalStore()
    add_event(st, event_at=EVENT_AT, available_at=AVAIL, payload={"reported_eps": 1.1})
    e = ev_state(st)["events"][0]
    assert e.get("eps_surprise") is None            # absent key, never 0.0
    assert "quality_flag" not in e                  # silent: no flag explains why there is no surprise


def test_missing_actual_gives_no_surprise():
    st = PointInTimeFundamentalStore()
    add_event(st, event_at=EVENT_AT, available_at=AVAIL, payload={"consensus_eps": 1.0, "consensus_available_at": "2025-04-01T00:00:00Z"})
    assert ev_state(st)["events"][0].get("eps_surprise") is None


@pytest.mark.parametrize("consensus_at,usable", [
    ("2025-04-30T23:59:59Z", True),
    ("2025-05-01T12:59:59.999999Z", True),
    ("2025-05-01T13:00:00Z", False),          # published exactly at the event: not pre-event
    ("2025-05-01T13:00:00.000001Z", False),
    ("2025-05-02T00:00:00Z", False),          # consensus published AFTER the event: excluded
    ("2025-05-01T14:00:00+01:00", False),     # = 13:00Z, equal to event_at -> excluded
    ("2025-05-01T13:59:00+01:00", True),      # 12:59Z
    (None, False),                            # no consensus timestamp at all: cannot prove pre-event
    ("", False),
])
def test_consensus_must_precede_event(consensus_at, usable):
    st = PointInTimeFundamentalStore()
    earnings(st, 1.10, 1.00, consensus_at=consensus_at)
    e = ev_state(st)["events"][0]
    if usable:
        assert e["eps_surprise"] == pytest.approx(0.10, rel=1e-12) and "quality_flag" not in e
    else:
        assert e["eps_surprise"] is None and e["quality_flag"] == "NO_PRE_EVENT_CONSENSUS"


def test_naive_or_unparseable_consensus_timestamp_is_flagged_not_silently_used():
    """Lead re-audit (QA-MAC-02 / QA-DEC-11 fixed): a naive or unparseable consensus timestamp yields no surprise and an explicit
    INVALID_CONSENSUS_TIMESTAMP flag (it neither passes silently nor aborts the whole symbol's catalyst state)."""
    for bad in ("2025-04-30T00:00:00", "not-a-date"):
        st = PointInTimeFundamentalStore()
        earnings(st, 1.1, 1.0, consensus_at=bad)
        e = ev_state(st)["events"][0]
        assert e["eps_surprise"] is None and e["quality_flag"] == "INVALID_CONSENSUS_TIMESTAMP"


# ============================================================================ decay
HL_TABLE = {0: 1.0, 10: 0.5, 20: 0.25, 30: 0.125}


@pytest.mark.parametrize("age_days", [0, 1, 5, 20, 0.5, 365])
@pytest.mark.parametrize("half_life", [10, 5, 365, 0.25])
def test_decay_matches_exact_half_life_formula(age_days, half_life):
    st = PointInTimeFundamentalStore()
    earnings(st, 1.1, 1.0)
    ts = dt(AVAIL) + timedelta(days=age_days)
    e = ev_state(st, ts, half_life=half_life)["events"][0]
    assert e["age_days"] == pytest.approx(age_days, abs=1e-12)
    assert e["decay"] == pytest.approx(decay(age_days, half_life), rel=1e-12, abs=1e-300)


@pytest.mark.parametrize("age_days,expected", [(0, 1.0), (10, 0.5), (20, 0.25), (30, 0.125), (5, 2 ** -0.5)])
def test_decay_half_life_ten_hand_values(age_days, expected):
    st = PointInTimeFundamentalStore()
    earnings(st, 1.1, 1.0)
    e = ev_state(st, dt(AVAIL) + timedelta(days=age_days), half_life=10)["events"][0]
    assert e["decay"] == pytest.approx(expected, rel=1e-12)


def test_age_is_measured_from_available_at_not_event_at():
    st = PointInTimeFundamentalStore()
    # event happened 5 days before it became available (late publication)
    add_event(st, event_at="2025-04-26T14:00:00Z", reported_at="2025-04-26T14:00:00Z", available_at=AVAIL,
              payload={"reported_eps": 1.1})
    e = ev_state(st, dt(AVAIL) + timedelta(days=1), half_life=10)["events"][0]
    assert e["age_days"] == pytest.approx(1.0, abs=1e-12)           # not 6.0
    assert e["decay"] == pytest.approx(2 ** -0.1, rel=1e-12)


def test_decay_is_none_without_configured_half_life_and_validates_configuration():
    st = PointInTimeFundamentalStore()
    earnings(st, 1.1, 1.0)
    assert ev_state(st)["events"][0]["decay"] is None
    for bad in (0, -1, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="invalid decay"):
            EventCatalystEngine(st, half_life_days=bad)


def test_event_state_has_no_score_confidence_or_execution_authority():
    st = PointInTimeFundamentalStore()
    earnings(st, 1.1, 1.0)
    out = ev_state(st, half_life=10)
    assert out["score"] is None and out["confidence"] is None and out["status"] == "DESCRIPTIVE_ONLY"
    assert ev_state(PointInTimeFundamentalStore())["status"] == "MISSING"


# ============================================================================ visibility
@pytest.mark.parametrize("ts,visible", [
    ("2025-05-01T13:59:59Z", False), ("2025-05-01T13:59:59.999999Z", False),
    ("2025-05-01T14:00:00Z", True), ("2025-05-01T14:00:00.000001Z", True),
    ("2025-05-01T15:00:00+01:00", True), ("2025-05-01T14:59:59+01:00", False),
    ("2025-05-01T09:59:59-04:00", False), ("2025-05-01T10:00:00-04:00", True),
])
def test_event_announced_14_00_invisible_at_13_59(ts, visible):
    st = PointInTimeFundamentalStore()
    add_event(st, event_at="2025-05-01T13:30:00Z", reported_at="2025-05-01T13:30:00Z", available_at=AVAIL, payload={})
    assert bool(st.events("XYZ", ts)) is visible
    assert bool(ev_state(st, ts)["events"]) is visible


def test_naive_as_of_is_rejected():
    st = PointInTimeFundamentalStore()
    with pytest.raises(ValueError, match="timezone"):
        st.events("XYZ", "2025-05-01T14:00:00")
    with pytest.raises(ValueError, match="timezone"):
        EventCatalystEngine(st).state("XYZ", "2025-05-01T14:00:00")


def test_future_dated_calendar_event_visible_only_after_announcement():
    st = PointInTimeFundamentalStore()
    # earnings date (2025-06-10) announced on 2025-04-01
    add_event(st, event_at="2025-06-10T20:00:00Z", reported_at="2025-04-01T09:00:00Z",
              available_at="2025-04-01T09:00:00Z", payload={"event_key": "Q1-call", "calendar": True})
    assert ev_state(st, "2025-03-31T23:59:59Z")["events"] == []
    out = ev_state(st, "2025-04-02T00:00:00Z", half_life=30)["events"]
    assert len(out) == 1 and out[0]["event_at"] == "2025-06-10T20:00:00+00:00"
    # age counts from the announcement even though the event itself is in the future
    assert out[0]["age_days"] == pytest.approx(0.625, abs=1e-12)


def test_late_published_event_is_invisible_before_publication_even_if_it_already_happened():
    st = PointInTimeFundamentalStore()
    add_event(st, event_at="2025-04-01T10:00:00Z", reported_at="2025-04-20T10:00:00Z", available_at="2025-04-20T10:00:00Z", payload={})
    assert st.events("XYZ", "2025-04-19T23:59:59Z") == []
    assert len(st.events("XYZ", "2025-04-20T10:00:00Z")) == 1


def test_event_cannot_be_available_before_it_was_published():
    st = PointInTimeFundamentalStore()
    with pytest.raises(ValueError, match="available before publication"):
        add_event(st, event_at=EVENT_AT, reported_at="2025-05-01T15:00:00Z", available_at="2025-05-01T14:00:00Z")


# ============================================================================ versioning, dedupe
def test_corrected_event_replaces_only_after_correction_is_available_any_insertion_order():
    for order in ((0, 1), (1, 0)):
        st = PointInTimeFundamentalStore()
        versions = [dict(revision="1", available_at="2025-05-01T14:00:00Z", payload={"event_key": "Q1", "reported_eps": 1.0}),
                    dict(revision="2", available_at="2025-05-03T09:00:00Z", payload={"event_key": "Q1", "reported_eps": 1.05})]
        for i in order:
            add_event(st, event_at=EVENT_AT, **versions[i])
        assert [e["payload"]["reported_eps"] for e in st.events("XYZ", "2025-05-02T00:00:00Z")] == [1.0]
        assert [e["payload"]["reported_eps"] for e in st.events("XYZ", "2025-05-03T08:59:59Z")] == [1.0]
        assert [e["payload"]["reported_eps"] for e in st.events("XYZ", "2025-05-03T09:00:00Z")] == [1.05]
        assert st.db.execute("SELECT COUNT(*) FROM corporate_events").fetchone()[0] == 2     # history kept


def test_QA_MAC_01_equal_timestamp_conflicting_versions_must_not_depend_on_hash_order():
    wins_rev1 = 0
    for k, (eps1, eps2) in enumerate([(1.0, 1.1), (1.2, 1.3), (0.7, 0.8), (2.0, 2.1), (3.0, 3.1), (0.1, 0.2)]):
        st = PointInTimeFundamentalStore()
        for rev, eps in (("1", eps1), ("2", eps2)):
            add_event(st, event_at=EVENT_AT, available_at=AVAIL, revision=rev, payload={"event_key": "Q1", "reported_eps": eps})
        got = st.events("XYZ", AFTER)
        assert len(got) == 1
        wins_rev1 += got[0]["revision"] == "1"
    # acceptable outcomes: always the higher revision, or a ValueError conflict (never reached here). Hash order gives a mix.
    assert wins_rev1 == 0, f"older revision '1' silently won {wins_rev1}/6 equal-timestamp conflicts"


def test_same_real_event_from_five_sources_is_five_entries_LIMITATION():
    """No cross-source dedupe: the same earnings release from 5 feeds yields 5 events (each with its own surprise)."""
    st = PointInTimeFundamentalStore()
    for i in range(5):
        add_event(st, source=f"feed{i}", payload={"event_key": "Q1", "reported_eps": 1.1, "consensus_eps": 1.0,
                                                  "consensus_available_at": "2025-04-01T00:00:00Z"})
    out = ev_state(st)
    assert len(out["events"]) == 5 and {e["source"] for e in out["events"]} == {f"feed{i}" for i in range(5)}
    assert all(e["eps_surprise"] == pytest.approx(0.1, rel=1e-12) for e in out["events"])
    assert out["score"] is None            # no aggregation exists, so there is no double counting in any number


def test_dedupe_key_is_symbol_type_key_source():
    st = PointInTimeFundamentalStore()
    base = dict(event_at=EVENT_AT, available_at=AVAIL)
    add_event(st, payload={"event_key": "A"}, **base)
    add_event(st, payload={"event_key": "B"}, **base)                        # other key
    add_event(st, event_type="guidance", payload={"event_key": "A"}, **base)  # other type
    add_event(st, source="other", payload={"event_key": "A"}, **base)        # other source
    add_event(st, symbol="ABC", payload={"event_key": "A"}, **base)          # other symbol
    assert len(st.events("XYZ", AFTER)) == 4 and len(st.events("ABC", AFTER)) == 1


def test_events_without_event_key_collapse_on_event_at_LIMITATION():
    """Without event_key two DISTINCT same-source events with identical event_at overwrite each other."""
    st = PointInTimeFundamentalStore()
    add_event(st, available_at="2025-05-01T14:00:00Z", revision="a", payload={"headline": "A"})
    add_event(st, available_at="2025-05-01T14:01:00Z", revision="b", payload={"headline": "B"})
    got = st.events("XYZ", AFTER)
    assert len(got) == 1 and got[0]["payload"]["headline"] == "B"
    # and a correction that moves event_at is NOT a correction: both versions stay visible
    st2 = PointInTimeFundamentalStore()
    add_event(st2, event_at="2025-05-01T13:00:00Z", available_at="2025-05-01T14:00:00Z", revision="a")
    add_event(st2, event_at="2025-05-02T13:00:00Z", available_at="2025-05-03T14:00:00Z", revision="b")
    assert len(st2.events("XYZ", "2025-05-04T00:00:00Z")) == 2


@pytest.mark.parametrize("event_type", ["earnings", "merger_acquisition", "ceo_change", "capital_raise", "buyback", "dividend_change"])
def test_each_corporate_event_enters_once_even_if_ingested_repeatedly(event_type):
    st = PointInTimeFundamentalStore()
    for _ in range(3):
        ids = {add_event(st, event_type=event_type, payload={"event_key": "e1", "detail": event_type})}
    assert len(ids) == 1
    assert st.db.execute("SELECT COUNT(*) FROM corporate_events").fetchone()[0] == 1
    assert len(st.events("XYZ", AFTER)) == 1


def test_typed_corporate_event_extraction_and_revenue_surprise_NOT_IMPLEMENTED():
    src = "\n".join(p.read_text() for p in (REPO / "fundamentals").glob("*.py"))
    for token in ("revenue_surprise", "reported_revenue", "consensus_revenue", "ceo_change", "merger", "capital_raise"):
        assert token not in src, token
    st = PointInTimeFundamentalStore()
    add_event(st, payload={"reported_revenue": 120.0, "consensus_revenue": 100.0, "consensus_available_at": "2025-04-01T00:00:00Z"})
    assert "revenue_surprise" not in ev_state(st)["events"][0]


# ============================================================================ wildcard
def test_wildcard_events_visible_to_every_symbol_after_availability():
    st = PointInTimeFundamentalStore()
    add_event(st, symbol="*", event_type="macro_release", available_at=AVAIL, payload={"event_key": "CPI"})
    add_event(st, symbol="AAA", available_at=AVAIL, payload={"event_key": "CPI"})
    for sym in ("AAA", "BBB", "zzz", "MACRO:X"):
        before = st.events(sym, "2025-05-01T13:59:59Z")
        assert before == []
    assert {e["symbol"] for e in st.events("AAA", AFTER)} == {"*", "AAA"}      # distinct keys: both kept
    assert [e["symbol"] for e in st.events("bbb", AFTER)] == ["*"]              # lower-case query normalised
    assert [e["symbol"] for e in st.events("*", AFTER)] == ["*"]


# ============================================================================ malformed payloads
def test_QA_MAC_02_string_eps_payload_must_not_leak_TypeError():
    st = PointInTimeFundamentalStore()
    earnings(st, "1.1", "1.0")
    try:
        out = ev_state(st)
    except ValueError:
        return                                   # a clear ValueError is acceptable
    assert out["events"][0].get("eps_surprise") is None   # or a flagged None, never an exception of another type


def test_QA_MAC_02_http_events_endpoint_with_string_eps_returns_json_error(tmp_path):
    """Lead re-audit: the endpoint now answers 200 with the malformed record FLAGGED (no surprise value); any other failure
    must still be a JSON error, never a dropped connection (api.serve also has a catch-all 500 JSON branch)."""
    path = tmp_path / "ev.sqlite"
    st = PointInTimeFundamentalStore(path)
    earnings(st, "1.1", "1.0")
    st.close()
    srv = serve(str(path), port=0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/fundamentals/XYZ/events?as_of={AFTER}", timeout=10)
            status, body = r.status, r.read()
        except urllib.error.HTTPError as e:
            status, body = e.code, e.read()
        parsed = json.loads(body)
        if status == 200:
            assert parsed["events"][0]["eps_surprise"] is None and parsed["events"][0]["quality_flag"] == "INVALID_EARNINGS_PAYLOAD"
        else:
            assert status in (400, 422, 500, 503) and "error" in parsed
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.mark.parametrize("payload", [[1, 2], None, "text", 5])
def test_QA_MAC_02_non_dict_payload_must_be_rejected_at_write_or_tolerated_at_read(payload):
    st = PointInTimeFundamentalStore()
    try:
        st.add_event(symbol="XYZ", event_type="earnings", event_at=EVENT_AT, reported_at=EVENT_AT, available_at=AVAIL,
                     source="feed", revision="1", payload=payload)
    except ValueError:
        return                                  # rejected at write: acceptable
    st.events("XYZ", AFTER)                     # otherwise reads must keep working (currently AttributeError)


def test_QA_MAC_02_one_malformed_wildcard_event_must_not_break_every_symbol():
    st = PointInTimeFundamentalStore()
    add_event(st, symbol="*", payload={"reported_eps": "1", "consensus_eps": "2", "consensus_available_at": "2025-04-01T00:00:00Z"})
    for sym in ("AAA", "BBB", "CCC"):
        ev_state(st, symbol=sym)


def test_nan_and_infinity_cannot_enter_event_payloads():
    st = PointInTimeFundamentalStore()
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            add_event(st, payload={"reported_eps": bad})


# ============================================================================ API
def test_events_api_equals_engine_and_is_pit(tmp_path):
    st = PointInTimeFundamentalStore()
    earnings(st, 1.1, 1.0)
    api = FundamentalAPI(st)
    got = json.loads(json.dumps(api.get("/fundamentals/XYZ/events", {"as_of": AFTER}), allow_nan=False))
    assert got == json.loads(json.dumps(ev_state(st), allow_nan=False))
    assert api.get("/fundamentals/XYZ/events", {"as_of": "2025-05-01T13:59:59Z"})["events"] == []


# ============================================================================ randomized differential
def _random_events(seed):
    rng = random.Random(seed)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    events = []
    used_avail = set()
    for _ in range(rng.randrange(6, 20)):
        sym = rng.choice(["AAA", "BBB", "*"])
        etype = rng.choice(["earnings", "guidance", "filing"])
        source = rng.choice(["s1", "s2"])
        key = rng.choice(["k1", "k2", None])
        event_at = base + timedelta(days=rng.randrange(0, 300), hours=rng.randrange(0, 24))
        for v in range(rng.randrange(1, 4)):
            while True:
                lag = timedelta(days=rng.randrange(0, 200), seconds=rng.randrange(1, 86400))
                avail = event_at + lag
                if (sym, etype, source, key, event_at, avail) not in used_avail and avail not in used_avail:
                    used_avail.add(avail)
                    break
            payload = {"reported_eps": round(rng.uniform(-2, 3), 2), "consensus_eps": round(rng.uniform(-2, 3), 2),
                       "consensus_available_at": iso(event_at - timedelta(days=rng.choice([-1, 0, 1, 5])))}
            if key:
                payload["event_key"] = key
            events.append(dict(symbol=sym, event_type=etype, event_at=event_at, available_at=avail, source=source,
                               revision=str(v), payload=payload))
    return events


def _oracle_visible(events, symbol, ts):
    groups = {}
    for e in sorted((e for e in events if e["symbol"] in (symbol, "*") and e["available_at"] <= ts),
                    key=lambda e: e["available_at"]):
        key = (e["symbol"], e["event_type"], e["payload"].get("event_key", iso(e["event_at"])), e["source"])
        groups[key] = e
    return groups


@pytest.mark.parametrize("seed", range(100))
def test_events_random_timelines_match_oracle_truncated_db_and_canary(seed):
    events = _random_events(seed)
    st = PointInTimeFundamentalStore()
    for e in events:
        add_event(st, symbol=e["symbol"], event_type=e["event_type"], event_at=iso(e["event_at"]),
                  available_at=iso(e["available_at"]), source=e["source"], revision=e["revision"], payload=e["payload"])
    rng = random.Random(5000 + seed)
    half_life = 20.0
    for _ in range(4):
        ts = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=rng.randrange(0, 520 * 86400))
        symbol = rng.choice(["AAA", "BBB"])
        want = _oracle_visible(events, symbol, ts)
        out = EventCatalystEngine(st, half_life_days=half_life).state(symbol, ts)
        got = {(e["symbol"], e["event_type"], e["payload"].get("event_key", e["event_at"]), e["source"]): e for e in out["events"]}
        want_keyed = {(k[0], k[1], k[2] if "event_key" in v["payload"] else iso(v["event_at"]), k[3]): v for k, v in want.items()}
        assert set(got) == set(want_keyed), f"seed={seed} ts={ts}"
        for k, prod in got.items():
            o = want_keyed[k]
            assert prod["revision"] == o["revision"] and prod["available_at"] == iso(o["available_at"]), f"seed={seed} {k}"
            age = (ts - o["available_at"]).total_seconds() / 86400
            assert prod["age_days"] == pytest.approx(age, rel=1e-12, abs=1e-12)
            assert prod["decay"] == pytest.approx(decay(age, half_life), rel=1e-12)
            pre = dt(o["payload"]["consensus_available_at"]) < o["event_at"]
            exp = eps_surprise(o["payload"]["reported_eps"], o["payload"]["consensus_eps"]) if pre else None
            if exp is None:
                assert prod["eps_surprise"] is None, f"seed={seed} {k}"
            else:
                assert prod["eps_surprise"] == pytest.approx(exp, rel=1e-12, abs=1e-15), f"seed={seed} {k}"
        # PHASE 98: physical truncation to rows available at ts is byte-identical
        trunc = truncated_store(st, ts)
        assert canonical(EventCatalystEngine(trunc, half_life_days=half_life).state(symbol, ts)) == canonical(out), f"seed={seed}"
        # canary: new events AND corrected versions of visible events published after ts must not leak
        canary = truncated_store(st, datetime(2100, 1, 1, tzinfo=UTC))
        for j, e in enumerate(list(out["events"]) + [{"symbol": symbol, "event_type": "earnings", "event_at": ts, "source": "s1", "payload": {}}]):
            add_event(canary, symbol=e["symbol"], event_type=e["event_type"], event_at=e["event_at"] if isinstance(e["event_at"], str) else iso(e["event_at"]),
                      available_at=iso(ts + timedelta(seconds=1 + j)), source=e["source"], revision="canary",
                      payload={**e["payload"], "reported_eps": CANARY, "consensus_eps": 1.0, "consensus_available_at": "2000-01-01T00:00:00Z"})
        after = canonical(EventCatalystEngine(canary, half_life_days=half_life).state(symbol, ts))
        assert after == canonical(out), f"seed={seed} future canary changed the state"
        assert str(CANARY) not in after
        late = canonical(EventCatalystEngine(canary, half_life_days=half_life).state(symbol, ts + timedelta(days=1)))
        assert str(CANARY) in late, "canary control: it must be visible once public"
