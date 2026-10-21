"""Validated SIP/NBBO quote storage and causal five-minute features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.multitf import storage_symbol

QUOTE_COLUMNS = ("ts", "bid_price", "ask_price", "bid_size", "ask_size")
DEFAULT_QUOTE_ROOT = Path(__file__).parent / "quotes"


def validate_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    if missing := set(QUOTE_COLUMNS) - set(frame.columns):
        raise ValueError(f"quotes missing columns: {sorted(missing)}")
    result = frame[list(QUOTE_COLUMNS)].copy()
    result["ts"] = pd.to_datetime(result["ts"], utc=True)
    result = result.sort_values("ts", kind="stable").drop_duplicates("ts", keep="last")
    numeric = list(QUOTE_COLUMNS[1:])
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all() or (result[numeric] < 0).any().any():
        raise ValueError("quote prices/sizes must be finite and non-negative")
    if (result["bid_price"] <= 0).any() or (result["ask_price"] < result["bid_price"]).any():
        raise ValueError("quotes require 0 < bid <= ask")
    return result.reset_index(drop=True)


def load_quotes(ticker: str, *, root: Path = DEFAULT_QUOTE_ROOT) -> pd.DataFrame:
    directory = root / storage_symbol(ticker)
    if not directory.exists():
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    parts = [pd.read_parquet(path) for path in sorted(directory.glob("*.parquet"))]
    return validate_quotes(pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame(columns=QUOTE_COLUMNS)


def quote_features_at(
    quotes: pd.DataFrame, timestamps: pd.DatetimeIndex, *, tolerance: str = "5min",
) -> pd.DataFrame:
    """Return last-known quote features without looking past each timestamp."""
    validated = validate_quotes(quotes)
    left = pd.DataFrame({"ts": pd.to_datetime(timestamps, utc=True)}).sort_values("ts")
    aligned = pd.merge_asof(
        left, validated, on="ts", direction="backward", tolerance=pd.Timedelta(tolerance),
    )
    midpoint = (aligned["bid_price"] + aligned["ask_price"]) / 2
    size = aligned["bid_size"] + aligned["ask_size"]
    return pd.DataFrame({
        "spread_bps": ((aligned["ask_price"] - aligned["bid_price"]) / midpoint * 10_000).to_numpy(),
        "quote_imbalance": ((aligned["bid_size"] - aligned["ask_size"]) / size.where(size > 0)).to_numpy(),
        "microprice_edge_bps": (
            (aligned["ask_price"] * aligned["bid_size"] + aligned["bid_price"] * aligned["ask_size"])
            / size.where(size > 0) / midpoint - 1
        ).to_numpy() * 10_000,
    }, index=timestamps)


def quote_coverage(features: pd.DataFrame) -> float:
    return float(np.isfinite(features["spread_bps"]).mean()) if len(features) else 0.0
