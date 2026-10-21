"""Record one prospective, non-executed observation for execution calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.shadow_observation import append_shadow_observation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--observed-at", required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--side", required=True, choices=("long", "short"))
    parser.add_argument("--quantity", required=True, type=float)
    parser.add_argument("--predicted-edge-bps", required=True, type=float)
    parser.add_argument("--intended-price", required=True, type=float)
    parser.add_argument("--observed-bid", required=True, type=float)
    parser.add_argument("--observed-ask", required=True, type=float)
    parser.add_argument("--reference-exit-price", required=True, type=float)
    parser.add_argument("--data-source", required=True)
    args = parser.parse_args()
    frozen = json.loads((args.run_dir / "frozen_candidate.json").read_text())
    frame = append_shadow_observation(args.run_dir / "shadow_observations.csv", {
        "observed_at": args.observed_at, "candidate_sha256": frozen["sha256"],
        "ticker": args.ticker.upper(), "side": args.side, "quantity": args.quantity,
        "predicted_edge_bps": args.predicted_edge_bps,
        "intended_price": args.intended_price, "observed_bid": args.observed_bid,
        "observed_ask": args.observed_ask, "reference_exit_price": args.reference_exit_price,
        "data_source": args.data_source,
    })
    print(f"shadow journal now contains {len(frame)} observations; approval_evidence=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
