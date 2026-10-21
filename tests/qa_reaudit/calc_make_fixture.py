"""Generate tests/qa_reaudit/fixtures/calc/golden_features.json from the ORACLE (never from production).

Run once (or after editing calc_golden.py):  .venv/bin/python -m tests.qa_reaudit.calc_make_fixture
"""
from __future__ import annotations
import json
from pathlib import Path

from tests.qa_reaudit import calc_oracle as O
from tests.qa_reaudit.calc_cmp import oracle_from_years
from tests.qa_reaudit.calc_golden import GOLDEN

OUT = Path(__file__).parent / "fixtures" / "calc" / "golden_features.json"


def build():
    doc = {}
    for name, g in sorted(GOLDEN.items()):
        feats, flags, cur = oracle_from_years(g["years"], g["sector"])
        cur = {**cur, "preferred_equity": O.D(0), "minority_interest": O.D(0)}
        val = O.valuation(g["price"], g["shares_out"], cur, forward_eps=g["forward_eps"])
        fmt = lambda x: None if x is None else format(x, ".30g")
        doc[name] = {"features": {k: fmt(v) for k, v in sorted(feats.items())}, "flags": sorted(flags),
                     "valuation": {k: fmt(v) for k, v in sorted(val.items())}, "price": g["price"], "shares_out": g["shares_out"],
                     "forward_eps": g["forward_eps"], "sector": g["sector"], "note": g["note"]}
    return doc


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1, sort_keys=True) + "\n")
    print("wrote", OUT)
