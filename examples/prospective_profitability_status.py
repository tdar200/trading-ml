"""Report evidence accumulated after a frozen candidate's data cutoff."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.prospective import prospective_progress
from data.multitf import get_bars


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--intraday-root", type=Path)
    args = parser.parse_args()
    frozen = json.loads((args.run_dir / "frozen_candidate.json").read_text())
    summary = json.loads((args.run_dir / "summary.json").read_text())
    kwargs = {"intraday_root": args.intraday_root} if args.intraday_root else {}
    bars = get_bars(summary["benchmark"], summary["timeframe"], **kwargs)
    report = prospective_progress(
        data_end=frozen["data_end"], benchmark_bars=bars,
        shadow_path=args.run_dir / "shadow_observations.csv",
        paper_path=args.run_dir / "paper_observations.csv",
    )
    report.update({
        "candidate_sha256": frozen["sha256"],
        "research_only_free_data": bool(summary.get("research_only_free_data", False)),
        "deployment_approved": False,
    })
    path = args.run_dir / "prospective_status.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
