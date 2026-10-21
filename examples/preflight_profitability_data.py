"""Write a fail-closed readiness inventory for every profitability sleeve."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.deployment_gate import file_sha256
from analysis.sleeves import SLEEVES
from data.multitf import get_bars
from data.quotes import load_quotes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--intraday-root", type=Path)
    parser.add_argument("--timeframe", choices=("5m", "1h"), default="5m")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "profitability_data_readiness.json")
    args = parser.parse_args()
    if args.membership and not args.membership.is_file(): parser.error("membership file not found")
    if args.events and not args.events.is_file(): parser.error("event file not found")
    sleeves = {}
    for name, spec in SLEEVES.items():
        symbols = {}
        for symbol in spec.symbols:
            kwargs = {"intraday_root": args.intraday_root} if args.intraday_root else {}
            bars = get_bars(symbol, args.timeframe, **kwargs)
            quotes = load_quotes(symbol)
            symbols[symbol] = {
                "bar_rows": len(bars), "bar_sessions": int(bars.ts.dt.date.nunique()) if len(bars) else 0,
                "quote_rows": len(quotes),
            }
        sleeves[name] = {
            "session_model": spec.session_model,
            "simulator_ready": spec.deployable_with_equity_replay,
            "all_bars_present": all(item["bar_rows"] > 0 for item in symbols.values()),
            "sip_quotes_present": all(item["quote_rows"] > 0 for item in symbols.values()),
            "symbols": symbols,
        }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "sleeves": sleeves,
        "timeframe": args.timeframe,
        "intraday_root": str(args.intraday_root) if args.intraday_root else "data/intraday",
        "point_in_time_membership": (
            {"path": str(args.membership), "sha256": file_sha256(args.membership)} if args.membership else None
        ),
        "event_calendar": (
            {"path": str(args.events), "sha256": file_sha256(args.events)} if args.events else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
