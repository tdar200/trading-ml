"""Create month, instrument, side and regime attribution for one rejected run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.failure_attribution import summarize_ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    ledger = pd.read_csv(args.run_dir / "evaluation_ledger.csv")
    outputs = {}
    for name, groups in {
        "monthly": ["month"], "instrument": ["ticker"],
        "side": ["side"], "regime": ["regime"],
    }.items():
        frame = summarize_ledger(ledger, groups)
        frame.to_csv(args.run_dir / f"failure_attribution_{name}.csv", index=False)
        outputs[name] = frame.to_dict(orient="records")
    report = {"approval_evidence": False, "research_diagnostic_only": True, "attribution": outputs}
    (args.run_dir / "failure_attribution.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(args.run_dir / "failure_attribution.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
