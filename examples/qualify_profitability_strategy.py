"""Evaluate a frozen profitability candidate against historical and paper gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.deployment_gate import (
    candidate_fingerprint, file_sha256, historical_qualification, paper_qualification,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--paper-log", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    frozen = json.loads((args.run_dir / "frozen_candidate.json").read_text(encoding="utf-8"))
    if candidate_fingerprint(frozen["candidate"]) != frozen["sha256"]:
        raise SystemExit("frozen candidate fingerprint mismatch")
    historical = historical_qualification(
        summary, pd.read_csv(args.run_dir / "evaluation_cost_stress.csv"),
    )
    if args.paper_log:
        if args.paper_log.resolve() != (args.run_dir / "paper_observations.csv").resolve():
            raise SystemExit("paper log must be RUN_DIR/paper_observations.csv so approval can bind it")
        paper = paper_qualification(
            pd.read_csv(args.paper_log), expected_fingerprint=frozen["sha256"],
            evidence_root=args.run_dir,
        )
    else:
        paper = {"qualified": False, "gates": {"paper_log_supplied": False}}
    approved = bool(historical["qualified"] and paper["qualified"])
    artifact_names = ["frozen_candidate.json", "evaluation_cost_stress.csv"]
    if args.paper_log:
        artifact_names.append("paper_observations.csv")
    report = {
        "deployment_approved": approved, "candidate_sha256": frozen["sha256"],
        "historical": historical, "paper": paper,
        "qualified_at": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": {
            name: file_sha256(args.run_dir / name) for name in artifact_names
        },
        "blockers": [
            f"historical:{name}" for name in historical.get("blockers", [])
        ] + [f"paper:{name}" for name in paper.get("blockers", ["paper_log_supplied"])],
    }
    (args.run_dir / "deployment_qualification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    summary["historical_deployment_qualification"] = historical
    summary["deployment_approved"] = approved
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0 if approved else 2


if __name__ == "__main__":
    raise SystemExit(main())
