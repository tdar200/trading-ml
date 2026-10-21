"""Validated point-in-time earnings/macro event exclusions for intraday research."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("date", "ticker", "event")


def load_event_calendar(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"event calendar not found: {path}")
    frame = pd.read_csv(path)
    if missing := set(REQUIRED_COLUMNS) - set(frame.columns):
        raise ValueError(f"event calendar missing columns: {sorted(missing)}")
    result = frame[list(REQUIRED_COLUMNS)].copy()
    result["date"] = pd.to_datetime(result["date"], utc=True).dt.date
    result["ticker"] = result["ticker"].astype(str).str.upper().str.strip()
    result["event"] = result["event"].astype(str).str.strip()
    if result[["ticker", "event"]].eq("").any().any():
        raise ValueError("event calendar requires ticker and event; use * for market-wide events")
    return result.drop_duplicates().sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def event_clear_mask(frame: pd.DataFrame, dates: object, tickers: tuple[str, ...]):
    import numpy as np

    output = np.ones((len(dates), len(tickers)), dtype=bool)
    by_date = {date: set(rows.ticker) for date, rows in frame.groupby("date", sort=False)}
    for row, date in enumerate(dates):
        blocked = by_date.get(date, set())
        if "*" in blocked:
            output[row] = False
        else:
            output[row] = np.asarray([ticker not in blocked for ticker in tickers])
    return output
