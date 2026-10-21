"""Append one broker-reconciled session to a frozen strategy paper journal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.paper_observation import append_paper_session
from analysis.deployment_gate import file_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--date", required=True)
    parser.add_argument("--predicted-gross-bps", required=True, type=float)
    parser.add_argument("--realized-gross-bps", required=True, type=float)
    parser.add_argument("--modeled-cost-bps", required=True, type=float)
    parser.add_argument("--realized-cost-bps", required=True, type=float)
    parser.add_argument("--net-return", required=True, type=float)
    parser.add_argument("--orders-submitted", required=True, type=int)
    parser.add_argument("--orders-filled", required=True, type=int)
    parser.add_argument(
        "--broker-evidence", required=True, type=Path,
        help="Immutable broker order/fill export for this session; its SHA-256 is journaled.",
    )
    parser.add_argument(
        "--reconciliation-failed", action="store_true",
        help="Mark the session as not reconciled; this permanently fails the paper gate.",
    )
    parser.add_argument("--risk-breach", action="store_true")
    args = parser.parse_args()
    if not args.broker_evidence.is_file():
        parser.error(f"broker evidence file does not exist: {args.broker_evidence}")
    evidence_dir = args.run_dir / "paper_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_name = f"{args.date}{args.broker_evidence.suffix or '.bin'}"
    evidence_copy = evidence_dir / evidence_name
    source_hash = file_sha256(args.broker_evidence)
    if evidence_copy.exists() and file_sha256(evidence_copy) != source_hash:
        parser.error(f"different broker evidence already exists for {args.date}")
    if not evidence_copy.exists():
        shutil.copy2(args.broker_evidence, evidence_copy)
    frozen = json.loads((args.run_dir / "frozen_candidate.json").read_text(encoding="utf-8"))
    frame = append_paper_session(args.run_dir / "paper_observations.csv", {
        "session_date": args.date, "candidate_sha256": frozen["sha256"],
        "predicted_gross_bps": args.predicted_gross_bps,
        "realized_gross_bps": args.realized_gross_bps,
        "modeled_cost_bps": args.modeled_cost_bps,
        "realized_cost_bps": args.realized_cost_bps, "net_return": args.net_return,
        "orders_submitted": args.orders_submitted, "orders_filled": args.orders_filled,
        "broker_reconciled": not args.reconciliation_failed,
        "broker_evidence_path": evidence_copy.relative_to(args.run_dir).as_posix(),
        "broker_evidence_sha256": source_hash,
        "risk_breach": args.risk_breach,
    })
    print(f"paper journal now contains {len(frame)} sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
