"""Fail-closed static preflight for profitability paper or real deployment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.deployment_gate import (
    candidate_fingerprint, historical_qualification, verify_deployment_approval,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--mode", choices=("paper", "real"), default="paper")
    args = parser.parse_args()
    frozen = json.loads((args.run_dir / "frozen_candidate.json").read_text(encoding="utf-8"))
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    fingerprint_ok = candidate_fingerprint(frozen["candidate"]) == frozen["sha256"]
    historical = historical_qualification(
        summary, pd.read_csv(args.run_dir / "evaluation_cost_stress.csv"),
    )
    gates = {
        "candidate_fingerprint": fingerprint_ok,
        "historical_qualification": bool(historical["qualified"]),
        "alpaca_api_key": bool(os.getenv("ALPACA_API_KEY")),
        "alpaca_api_secret": bool(os.getenv("ALPACA_API_SECRET")),
        "kill_switch_clear": os.getenv("PROFITABILITY_KILL_SWITCH", "0") != "1",
    }
    if args.mode == "real":
        try:
            verify_deployment_approval(args.run_dir)
            gates["deployment_qualification"] = True
        except RuntimeError:
            gates["deployment_qualification"] = False
        # No real-money profitability adapter exists; never imply that it does.
        gates["real_money_adapter_enabled"] = False
    ready = all(gates.values())
    report = {
        "mode": args.mode, "ready": ready, "candidate_sha256": frozen["sha256"],
        "gates": gates, "historical_blockers": historical["blockers"],
        "next_step": (
            "run examples/preflight_alpaca.py, then start supervised paper observation"
            if ready and args.mode == "paper" else
            "resolve every failed gate; real-money routing remains disabled"
        ),
    }
    print(json.dumps(report, indent=2))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
