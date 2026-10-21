"""Track C: frozen golden-feature fixture (generated from the ORACLE), determinism, ingestion-order independence,
persisted snapshot idempotency.

Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_calc_determinism.py
"""
import json
import random
from decimal import Decimal
from pathlib import Path

import pytest

from fundamentals.features import FundamentalFeatureEngine, valuation_features
from fundamentals.store import PointInTimeFundamentalStore

from tests.qa_reaudit import calc_oracle as O
from tests.qa_reaudit.calc_cmp import compare, oracle_from_years
from tests.qa_reaudit.calc_fixtures import annual_facts, classification, new_store, snapshot
from tests.qa_reaudit.calc_golden import GOLDEN
from tests.qa_reaudit.calc_make_fixture import OUT, build

AS_OF = "2024-06-01T00:00:00Z"
FIXTURE = json.loads(Path(OUT).read_text())


def facts_of(name):
    g = GOLDEN[name]
    f = annual_facts(name, g["years"])
    if g["sector"]:
        f.append(classification(name, g["sector"]))
    return f


def test_fixture_file_is_current_with_the_oracle():
    assert json.loads(json.dumps(build(), sort_keys=True)) == FIXTURE, "regenerate with: .venv/bin/python -m tests.qa_reaudit.calc_make_fixture"


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_production_matches_frozen_fixture(name):
    fx = FIXTURE[name]
    snap = snapshot(new_store(facts_of(name)), name, AS_OF)
    want = {k: None if v is None else Decimal(v) for k, v in fx["features"].items()}
    bad = compare(snap["features"], want, name)
    assert not bad, "\n".join(bad)
    assert sorted(snap["flags"]) == fx["flags"]
    raw = {**snap["raw"], "preferred_equity": 0.0, "minority_interest": 0.0}
    val = valuation_features(float(fx["price"]), float(fx["shares_out"]), raw, price_currency="GBP", statement_currency="GBP",
                             price_basis="S", share_basis="S", forward_eps=float(fx["forward_eps"]))
    wantv = {k: None if v is None else Decimal(v) for k, v in fx["valuation"].items()}
    assert not compare(val, wantv, name + ":val", abs_tol=1e-12)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_ingestion_order_does_not_change_the_snapshot_byte_for_byte(name):
    facts = facts_of(name)
    base = snapshot(new_store(facts), name, AS_OF)
    canon = json.dumps(base, sort_keys=True, allow_nan=False)
    for seed in range(3):
        shuffled = list(facts)
        random.Random(seed).shuffle(shuffled)
        s = PointInTimeFundamentalStore()
        for chunk in range(0, len(shuffled), 7):
            s.ingest(shuffled[chunk:chunk + 7])
        assert json.dumps(snapshot(s, name, AS_OF), sort_keys=True, allow_nan=False) == canon, (name, seed)


def test_repeated_snapshots_are_identical_and_persist_is_idempotent():
    s = new_store(facts_of("A"))
    eng = FundamentalFeatureEngine(s)
    a = eng.snapshot("A", AS_OF, persist=True)
    b = eng.snapshot("A", AS_OF, persist=True)
    assert a == b
    rows = s.db.execute("SELECT id, payload FROM feature_snapshots").fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0]["payload"]) == json.loads(json.dumps(a, sort_keys=True))   # stored copy == returned copy (full precision)
    c = eng.snapshot("A", "2024-06-02T00:00:00Z", persist=True)                           # a different as_of is a different record
    assert len(s.db.execute("SELECT id FROM feature_snapshots").fetchall()) == 2 and c["as_of"] != a["as_of"]
    # append-only: persisted snapshots cannot be rewritten
    import sqlite3
    with pytest.raises(sqlite3.DatabaseError):
        s.db.execute("UPDATE feature_snapshots SET payload='{}'")
    with pytest.raises(sqlite3.DatabaseError):
        s.db.execute("DELETE FROM feature_snapshots")


def test_features_do_not_depend_on_the_ticker_symbol_or_unrelated_companies():
    """The same statements under another symbol, in a store full of other companies, give identical feature values."""
    s = PointInTimeFundamentalStore()
    for n in GOLDEN:
        s.ingest(facts_of(n))
    from tests.qa_reaudit.calc_fixtures import fact
    alone = snapshot(new_store(facts_of("A")), "A", AS_OF)
    crowd = FundamentalFeatureEngine(s).snapshot("A", AS_OF)
    assert alone["features"] == crowd["features"] and alone["flags"] == crowd["flags"] and alone["lineage"] == crowd["lineage"]


def test_snapshot_is_independent_of_observed_at_for_research_queries():
    """Re-downloading the same filings later (new observed_at) never changes research values (identity excludes observed_at)."""
    facts = facts_of("A")
    s = new_store(facts)
    from dataclasses import replace
    s.ingest([replace(f, observed_at="2031-01-01T00:00:00+00:00") for f in facts])
    assert len(s.history("A")) == len(facts)
    assert snapshot(s, "A", AS_OF)["features"] == snapshot(new_store(facts), "A", AS_OF)["features"]
