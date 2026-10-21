"""Reconstruct and stream a canonical replay of one GPU-screened combination."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.dynamic_profitability import prepare_dynamic_decision_dataset
from analysis.profitability import align_decision_datasets, slice_decision_dataset
from analysis.profitability_v2 import evaluate_v2_candidate
from analysis.signal_combinations import (
    combined_score, orient_and_select_signals, sample_signal_tensor,
)
from analysis.signal_library import build_indicator_variants
from analysis.sleeves import require_equity_replay_sleeve
from analysis.vectorized_screen import resolve_device
from data.multitf import get_bars


def _safe(value):
    if isinstance(value, dict): return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if isinstance(value, np.ndarray): return [_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
    if hasattr(value, "isoformat"): return value.isoformat()
    return value


def _progress(percent: int, message: str) -> None:
    print(f"PROGRESS|{percent}|{message}", flush=True)


def _execution_lookup(
    bars: dict[str, pd.DataFrame], decision_bar: int,
) -> dict[tuple[str, str], dict]:
    """Resolve the precise next-bar entry, exit, and visible price path."""
    lookup: dict[tuple[str, str], dict] = {}
    for ticker, frame in bars.items():
        ordered = frame.sort_values("ts").reset_index(drop=True).copy()
        eastern = pd.to_datetime(ordered["ts"], utc=True).dt.tz_convert("America/New_York")
        ordered["session_date"] = eastern.dt.date
        ordered["bar"] = ((eastern.dt.hour * 60 + eastern.dt.minute - 570) // 60).astype(int)
        ordered["eastern_ts"] = eastern
        regular = ordered[(ordered.bar >= 0) & (ordered.bar <= 6)]
        for session_date, session in regular.groupby("session_date"):
            indexed = session.set_index("bar", drop=False)
            entry_bar = decision_bar + 1
            if entry_bar not in indexed.index:
                continue
            exit_bar = entry_bar if decision_bar == 1 else int(indexed.index.max())
            if exit_bar not in indexed.index:
                continue
            entry = indexed.loc[entry_bar]
            exit_row = indexed.loc[exit_bar]
            if isinstance(entry, pd.DataFrame): entry = entry.iloc[-1]
            if isinstance(exit_row, pd.DataFrame): exit_row = exit_row.iloc[-1]
            path = session[session.bar <= exit_bar][["eastern_ts", "open", "high", "low", "close"]]
            lookup[(str(session_date), ticker)] = {
                "entry_time": entry["eastern_ts"].isoformat(),
                "entry_price": float(entry["open"]),
                "exit_time": exit_row["eastern_ts"].isoformat(),
                "exit_price": float(exit_row["close"]),
                "price_bars": [
                    {
                        "ts": row.eastern_ts.isoformat(), "open": float(row.open),
                        "high": float(row.high), "low": float(row.low), "close": float(row.close),
                    }
                    for row in path.itertuples(index=False)
                ],
            }
    return lookup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", required=True)
    parser.add_argument("--decision-bar", type=int, choices=(1, 3), required=True)
    parser.add_argument("--period", choices=("discovery", "holdout", "full"), default="discovery")
    parser.add_argument("--cost-bps", type=float, default=2.5)
    parser.add_argument("--positions-per-side", type=int, choices=(1, 2, 3, 4, 5), default=3)
    parser.add_argument("--min-relative-volume", type=float, default=0.75)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--sleeve", default="individual_stock")
    parser.add_argument("--intraday-root", type=Path, default=ROOT / "data" / "free_intraday")
    args = parser.parse_args()

    device = resolve_device(args.device)
    all_variants = build_indicator_variants()
    by_key = {variant.key: variant for variant in all_variants}
    signal_keys = tuple(args.signals.split("+"))
    missing = [key for key in signal_keys if key not in by_key]
    if missing:
        raise SystemExit("unknown signal keys: " + ", ".join(missing))
    variants = tuple(by_key[key] for key in signal_keys)

    sleeve = require_equity_replay_sleeve(args.sleeve)
    kwargs = {"intraday_root": args.intraday_root}
    _progress(5, f"Loading hourly bars for {len(sleeve.symbols)} instruments")
    bars = {ticker: get_bars(ticker, "1h", **kwargs) for ticker in sleeve.symbols}
    benchmark = get_bars(sleeve.benchmark, "1h", **kwargs)
    if missing_bars := [ticker for ticker, frame in {**bars, sleeve.benchmark: benchmark}.items() if frame.empty]:
        raise SystemExit("missing 1h bars: " + ", ".join(missing_bars))

    datasets = align_decision_datasets({
        decision: prepare_dynamic_decision_dataset(
            bars, benchmark, decision, min_assets=15, bar_minutes=60,
            exit_after_bars=1 if decision == 1 else None,
        ) for decision in (1, 3)
    })
    dataset = datasets[args.decision_bar]
    execution_lookup = _execution_lookup(bars, args.decision_bar)
    _progress(20, "Aligned market data and execution targets")
    sessions = len(dataset.session_dates)
    warmup, cut = 60, sessions - 252
    validation_start, purge_sessions = cut - 126, 5
    orientation = np.arange(warmup, validation_start - purge_sessions)
    ranges = {
        "discovery": (validation_start, cut),
        "holdout": (cut, sessions),
        "full": (warmup, sessions),
    }
    start, stop = ranges[args.period]

    _progress(30, "Rebuilding selected indicator values on CUDA")
    tensor = sample_signal_tensor(
        bars, dataset.session_dates, args.decision_bar, bar_minutes=60,
        variants=variants, device=device,
    )
    oriented = orient_and_select_signals(
        tensor, dataset.forward_return, orientation, top_k=len(variants), device=device,
    )
    direction_by_index = {index: direction for index, direction, _ in oriented}
    definition = {
        "signal_indices": tuple(range(len(variants))),
        "directions": tuple(direction_by_index[index] for index in range(len(variants))),
        "signals": signal_keys,
    }
    scores = combined_score(tensor, definition)
    candidate = {
        "family": "signal_consensus", "timeframe": "1h",
        "decision_bar": args.decision_bar,
        "exit_policy": "one_bar" if args.decision_bar == 1 else "session_close",
        "max_positions_per_side": args.positions_per_side,
        "min_edge_bps": 0.0, "edge_cost_multiple": 0.0,
        "min_relative_volume": args.min_relative_volume,
        "event_policy": "all", "maximum_spread_bps": 8.0,
        "execution_buffer_bps": 0.75,
        "signal_key": f"d{args.decision_bar}:" + "+".join(signal_keys),
        "signals": signal_keys, "directions": definition["directions"],
    }
    replay_dataset = slice_decision_dataset(dataset, start, stop)
    equity_rows: list[dict] = []
    execution_charts: list[dict] = []

    def stream(event: dict) -> None:
        charts = []
        for trade in event["trades"]:
            detail = execution_lookup.get((str(trade["session_date"]), trade["ticker"]))
            if detail is None:
                continue
            trade.update({key: value for key, value in detail.items() if key != "price_bars"})
            chart = {
                "session_date": str(trade["session_date"]), "ticker": trade["ticker"],
                "side": trade["side"], **detail,
            }
            charts.append(chart)
            execution_charts.append(chart)
        event["trade_charts"] = charts
        event = _safe(event)
        equity_rows.append({
            "session_date": event["session_date"], "equity": event["equity"],
            "daily_return": event["daily_return"],
        })
        percent = 40 + int(55 * (event["session_index"] + 1) / event["session_count"])
        payload = {**event, "progress": percent}
        print("REPLAY|" + json.dumps(payload), flush=True)

    _progress(40, f"Replaying {len(replay_dataset.session_dates)} {args.period} sessions")
    result, _, ledger = evaluate_v2_candidate(
        replay_dataset, candidate, cost_per_side_bps=args.cost_bps,
        forecast_cache={"signal_consensus": scores[start:stop]}, device=device,
        return_ledger=True, stream_callback=stream,
    )
    if not ledger.empty:
        details = [
            execution_lookup.get((str(row.session_date), row.ticker), {})
            for row in ledger.itertuples(index=False)
        ]
        for name in ("entry_time", "entry_price", "exit_time", "exit_price"):
            ledger[name] = [item.get(name) for item in details]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / "signal_replays" / stamp
    out.mkdir(parents=True)
    pd.DataFrame(equity_rows).to_csv(out / "equity.csv", index=False)
    ledger.to_csv(out / "trades.csv", index=False)
    (out / "execution_charts.json").write_text(json.dumps(_safe(execution_charts)))
    summary = {
        **_safe(result), "period": args.period,
        "period_start": str(replay_dataset.session_dates[0]),
        "period_end": str(replay_dataset.session_dates[-1]),
        "artifact": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY|" + json.dumps(summary), flush=True)
    _progress(100, "Replay complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
