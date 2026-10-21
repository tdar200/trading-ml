"""Background discovery jobs with per-run state, cancellation and saved inputs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock, Thread
from uuid import uuid4

import numpy as np
import pandas as pd

from analysis.rule_discovery import Cancelled, Rule, discover, save_run
from data.multitf import get_bars

ROOT = Path(__file__).resolve().parents[1]
_lock = Lock()
_workers: dict[str, Thread] = {}


def demo_bars(instrument: str, timeframe: str) -> pd.DataFrame:
    """Deterministic synthetic fixture, explicitly not market evidence."""
    rows = []
    bars_per_day = {"15m": 26, "1h": 6, "1d": 1}[timeframe]
    minutes = {"15m": 15, "1h": 60, "1d": 1440}[timeframe]
    offset = sum(map(ord, instrument)) % 11
    previous = 100.0
    for day, session in enumerate(pd.bdate_range("2024-01-02", periods=120, tz="America/New_York")):
        for slot in range(bars_per_day):
            i = day * bars_per_day + slot
            close = 100 + .035 * i + 2.8 * np.sin((i + offset) / 4)
            opening = previous
            ts = session + pd.Timedelta(hours=9, minutes=30 + slot * minutes)
            rows.append({"ts": ts.tz_convert("UTC"), "open": opening,
                         "high": max(opening, close) + .15, "low": min(opening, close) - .15,
                         "close": close, "volume": 1000 + 200 * (1 + np.sin(i))})
            previous = close
    return pd.DataFrame(rows)


def write_state(directory: Path, **state):
    temporary = directory / "state.tmp"
    temporary.write_text(json.dumps(state, allow_nan=False))
    temporary.replace(directory / "state.json")


def read_state(directory: Path) -> dict:
    return json.loads((directory / "state.json").read_text())


def run_job(directory: Path, rules: list[Rule], cost_bps: float, source: str,
            intraday_root: Path, min_trades: int):
    def progress(percent, message):
        write_state(directory, status="running", percent=percent, message=message)

    try:
        data = {}
        for instrument, timeframe in sorted({(r.instrument, r.timeframe) for r in rules}):
            if (directory / "cancel").exists():
                raise Cancelled()
            progress(0, f"Loading {instrument} {timeframe}")
            frame = (demo_bars(instrument, timeframe) if source == "Synthetic demo" else
                     get_bars(instrument, timeframe, intraday_root=intraday_root))
            if frame.empty:
                raise ValueError(f"No local bars for {instrument} {timeframe}. Ingest this timeframe or change the selection.")
            data[(instrument, timeframe)] = frame
        report = discover(data, rules, cost_bps=cost_bps, min_trades=min_trades,
                          progress=progress, cancelled=lambda: (directory / "cancel").exists())
        report["data_source"] = source
        if source == "Synthetic demo":
            report["verdict"] = "DEMO_ONLY"
        if (directory / "cancel").exists():
            raise Cancelled()
        save_run(directory, report, data, rules)
        write_state(directory, status="complete", percent=100, message="Search complete")
    except Cancelled:
        write_state(directory, status="cancelled", percent=0, message="Search cancelled; no result published")
    except Exception as exc:
        write_state(directory, status="failed", percent=0, message=str(exc))


def start_job(root: Path, rules: list[Rule], cost_bps: float, source: str,
              intraday_root: Path, min_trades: int = 3) -> Path:
    if source not in ("Synthetic demo", "Local market data"):
        raise ValueError("Unknown data source")
    with _lock:
        if any(worker.is_alive() for worker in _workers.values()):
            raise ValueError("A search is already running; finish or cancel it first")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        directory = root / f"{stamp}_{uuid4().hex[:8]}"
        directory.mkdir(parents=True)
        write_state(directory, status="running", percent=0, message="Starting search")
        worker = Thread(target=run_job, args=(directory, rules, cost_bps, source, intraday_root, min_trades), daemon=True)
        _workers.clear()
        _workers[str(directory)] = worker
        worker.start()
    return directory


def job_alive(directory: Path) -> bool:
    worker = _workers.get(str(directory))
    return bool(worker and worker.is_alive())
