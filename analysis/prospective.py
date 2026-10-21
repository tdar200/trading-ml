"""Progress accounting for evidence accumulated after a candidate was frozen."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def prospective_progress(
    *, data_end: str, benchmark_bars: pd.DataFrame,
    shadow_path: Path | None = None, paper_path: Path | None = None,
) -> dict[str, Any]:
    if "ts" not in benchmark_bars:
        raise ValueError("benchmark bars require ts")
    timestamps = pd.to_datetime(benchmark_bars["ts"], utc=True)
    cutoff = pd.Timestamp(data_end)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    local_dates = timestamps.dt.tz_convert("America/New_York").dt.date
    future = local_dates[local_dates > cutoff.date()]
    sessions = int(future.nunique())
    shadow = pd.read_csv(shadow_path) if shadow_path and shadow_path.exists() else pd.DataFrame()
    paper = pd.read_csv(paper_path) if paper_path and paper_path.exists() else pd.DataFrame()
    paper_sessions = (
        int(pd.to_datetime(paper["session_date"], utc=True).dt.date.nunique())
        if len(paper) and "session_date" in paper else 0
    )
    paper_fills = int(paper["orders_filled"].sum()) if len(paper) and "orders_filled" in paper else 0
    return {
        "post_freeze_market_sessions": sessions,
        "shadow_observations": int(len(shadow)),
        "broker_verified_paper_sessions": paper_sessions,
        "broker_verified_fills": paper_fills,
        "minimum_required": {
            "unexposed_market_sessions": 252, "paper_sessions": 126, "paper_fills": 100,
        },
        "approval_ready_by_count_only": bool(
            sessions >= 252 and paper_sessions >= 126 and paper_fills >= 100
        ),
    }
