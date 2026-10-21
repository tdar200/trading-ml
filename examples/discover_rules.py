"""Run instrument/timeframe discovery, or verify a saved run, without the UI."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.discovery_jobs import read_state, start_job
from analysis.rule_discovery import TIMEFRAMES, TRIGGERS, candidate_grid, replay_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruments", nargs="+", default=["AAPL"])
    parser.add_argument("--timeframes", nargs="+", choices=TIMEFRAMES, default=["1h"])
    parser.add_argument("--triggers", nargs="+", choices=TRIGGERS, default=["rsi_cross", "ma_cross"])
    parser.add_argument("--periods", nargs="+", type=int, default=[7, 14])
    parser.add_argument("--trends", nargs="+", type=int, default=[0, 50])
    parser.add_argument("--holds", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--relative-volumes", nargs="+", type=float, default=[0.0])
    parser.add_argument("--stop-pct", type=float, default=.02)
    parser.add_argument("--target-pct", type=float, default=.04)
    parser.add_argument("--allocation", type=float, default=.95)
    parser.add_argument("--cost-bps", type=float, default=2.5)
    parser.add_argument("--min-trades", type=int, default=3)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--intraday-root", type=Path, default=ROOT / "data" / "free_intraday")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "rule_discovery")
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.replay:
        result = replay_run(args.replay)
        print(f"Replay verified: {result['metrics']}")
        return 0
    rules = candidate_grid(args.instruments, args.timeframes, args.triggers,
                           args.periods, args.trends, args.relative_volumes, args.holds,
                           stop_pct=args.stop_pct, target_pct=args.target_pct, allocation=args.allocation)
    directory = start_job(args.output_root, rules, args.cost_bps,
                          "Synthetic demo" if args.demo else "Local market data",
                          args.intraday_root, args.min_trades)
    print(f"Run: {directory}", flush=True)
    previous = None
    try:
        while True:
            state = read_state(directory)
            if state != previous:
                print(f"{state['percent']}% {state['status']}: {state['message']}", flush=True)
                previous = state
            if state["status"] != "running":
                return 0 if state["status"] == "complete" else 1
            time.sleep(.25)
    except KeyboardInterrupt:
        (directory / "cancel").touch()
        while read_state(directory)["status"] == "running":
            time.sleep(.1)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
