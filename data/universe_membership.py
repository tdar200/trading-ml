"""Point-in-time universe membership; missing history fails closed."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("ticker", "start", "end")


def load_membership(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"point-in-time membership file not found: {path}; current constituents "
            "must not be backfilled into historical tests"
        )
    frame = pd.read_csv(path)
    if missing := set(REQUIRED_COLUMNS) - set(frame.columns):
        raise ValueError(f"membership missing columns: {sorted(missing)}")
    result = frame[list(REQUIRED_COLUMNS)].copy()
    result["ticker"] = result["ticker"].str.upper().str.strip()
    result["start"] = pd.to_datetime(result["start"], utc=True)
    raw_end = result["end"]
    result["end"] = pd.to_datetime(raw_end, utc=True, errors="coerce")
    # A blank end means "still a member"; a NON-blank end that cannot be parsed must not silently mean the same.
    garbled = raw_end.notna() & raw_end.astype(str).str.strip().ne("") & result["end"].isna()
    if garbled.any():
        raise ValueError(f"membership end date is not a valid date for: {sorted(result.loc[garbled, 'ticker'])}")
    if result["ticker"].eq("").any() or result["start"].isna().any():
        raise ValueError("membership requires ticker and start")
    if ((result["end"].notna()) & (result["end"] <= result["start"])).any():
        raise ValueError("membership end must be after start")
    return result.sort_values(["start", "ticker"], kind="stable").reset_index(drop=True)


def members_on(frame: pd.DataFrame, timestamp: pd.Timestamp) -> tuple[str, ...]:
    ts = pd.Timestamp(timestamp)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    active = frame[(frame["start"] <= ts) & (frame["end"].isna() | (frame["end"] > ts))]
    return tuple(active["ticker"].drop_duplicates())
