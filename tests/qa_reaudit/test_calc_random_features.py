"""Track C: seeded randomized differential test of the annual snapshot (formulas, flags, growth, CAGR,
share-basis gating, restatement versions, fiscal-year drift, unit scales) against an independent Decimal oracle.

Run: .venv/bin/python -m pytest -p no:cacheprovider -q tests/qa_reaudit/test_calc_random_features.py
"""
import math
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from tests.qa_reaudit import calc_oracle as O
from tests.qa_reaudit.calc_fixtures import new_store, snapshot
from tests.qa_reaudit.calc_gen import expected, facts_for, gen_company, tolerance

SEEDS = list(range(400))


def pick_T(company, seed):
    rng = random.Random(seed * 7919 + 13)
    pubs = []
    for yt in company.years.values():
        pubs.append(yt.published)
        pubs += [p for p, _ in yt.restated.values()]
    pubs.sort()
    r = rng.random()
    if r < 0.5:
        return rng.choice(pubs) + rng.choice([timedelta(seconds=-1), timedelta(0), timedelta(seconds=1)])
    lo, hi = pubs[0] - timedelta(days=20), pubs[-1] + timedelta(days=800)
    return lo + (hi - lo) * rng.random()


@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed{s}")
def test_random_company_snapshot_matches_oracle(seed):
    company = gen_company(seed)
    T = pick_T(company, seed)
    store = new_store(facts_for(company))
    snap = snapshot(store, "RND", T)
    exp = expected(company, T)
    assert snap["status"] == exp["status"], (seed, T, snap["status"], exp["status"])
    if exp["status"] == "MISSING":
        assert snap["features"] == {} and snap["flags"] == [] and snap["score"] is None
        return
    assert snap["period_end"] == exp["period_end"], (seed, snap["period_end"], exp["period_end"])
    assert snap["age_days"] == exp["age"], (seed, snap["age_days"], exp["age"])
    got, want = snap["features"], exp["features"]
    assert set(got) == set(want), (seed, set(got) ^ set(want))
    bad = []
    for key in sorted(want):
        a, e = got[key], want[key]
        if a is not None and not math.isfinite(a):
            bad.append(f"{key}: non-finite {a!r}")
        elif not O.rel_close(a, e, 1e-12, tolerance(key)):
            bad.append(f"{key}: production={a!r} oracle={e!r}")
    assert not bad, f"seed={seed} T={T.isoformat()} sector={company.sector} units={company.unit_style[0]} blanked={exp['blanked']}\n" + "\n".join(bad)
    assert sorted(snap["flags"]) == sorted(exp["flags"]), (seed, snap["flags"], exp["flags"])
    assert snap["completeness"] == pytest.approx(sum(v is not None for v in want.values()) / len(want), abs=1e-15)
    # raw values are the base-currency statement lines visible at T
    for k, v in exp["cur"].items():
        assert snap["raw"].get(k) is not None and O.rel_close(snap["raw"][k], v, 1e-12, 1e-6), (seed, k, snap["raw"].get(k), v)


def test_generator_covers_the_interesting_space():
    """Meta test: the 400 seeds actually contain restatements, STALE, MISSING, blanked share basis, banks, negatives."""
    stats = dict(missing=0, stale=0, available=0, blanked=0, restated_visible=0, banks=0, neg_fcf=0, dilution=0, decel=0, falling=0, recv=0)
    for seed in SEEDS:
        c = gen_company(seed)
        T = pick_T(c, seed)
        e = expected(c, T)
        if e["status"] == "MISSING":
            stats["missing"] += 1
            continue
        stats["stale" if e["status"] == "STALE" else "available"] += 1
        stats["blanked"] += e["blanked"]
        stats["banks"] += c.sector == "banks"
        stats["restated_visible"] += any(p <= T for yt in c.years.values() for p, _ in yt.restated.values())
        for name, flag in (("neg_fcf", "NEGATIVE_FCF"), ("dilution", "DILUTION"), ("decel", "REVENUE_DECELERATION"),
                           ("falling", "FALLING_MARGINS"), ("recv", "RECEIVABLES_OUTPACE_SALES")):
            stats[name] += flag in e["flags"]
    for k, v in stats.items():
        assert v >= 5, (k, stats)
