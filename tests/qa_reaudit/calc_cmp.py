"""Track C shared comparison helpers (oracle expectation from a years-dict, feature-by-feature compare)."""
from __future__ import annotations
import math

from tests.qa_reaudit import calc_oracle as O
from tests.qa_reaudit.calc_fixtures import base_dict


def oracle_from_years(years, sector=None, current=2023):
    cur = base_dict(years[current])
    prior = base_dict(years.get(current - 1, {}))
    old = base_dict(years.get(current - 2, {}))
    feats, flags = O.accounting(cur, prior, old, sector)
    for n in (3, 5):
        feats[f"revenue_cagr_{n}y"] = O.cagr(cur.get("revenue"), base_dict(years.get(current - n, {})).get("revenue"), n)
    for m in ("revenue", "eps", "ocf", "operating_income"):
        feats[m + "_qoq"] = None
    return feats, flags, cur


def compare(actual, expected, label, rel=1e-12, abs_tol=1e-15):
    bad = []
    for key in sorted(set(actual) | set(expected)):
        if key not in actual:
            bad.append(f"{label}:{key} missing in production")
            continue
        if key not in expected:
            bad.append(f"{label}:{key} unexpected in production (value {actual[key]!r})")
            continue
        a, e = actual[key], expected[key]
        if a is not None and not isinstance(a, bool) and not math.isfinite(a):
            bad.append(f"{label}:{key} non-finite {a!r}")
        elif not O.rel_close(a, e, rel, abs_tol):
            bad.append(f"{label}:{key} production={a!r} oracle={e!r}")
    return bad
