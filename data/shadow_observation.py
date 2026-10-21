"""Append-only research-only shadow fills; never accepted as broker evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SHADOW_COLUMNS = (
    "observed_at", "candidate_sha256", "ticker", "side", "quantity",
    "predicted_edge_bps", "intended_price", "observed_bid", "observed_ask",
    "reference_exit_price", "data_source", "previous_row_sha256", "row_sha256",
)


def shadow_row_fingerprint(row: dict[str, Any]) -> str:
    numeric = {
        "quantity", "predicted_edge_bps", "intended_price", "observed_bid",
        "observed_ask", "reference_exit_price",
    }
    payload = {}
    for key in SHADOW_COLUMNS[:-1]:
        value = row[key].item() if isinstance(row[key], np.generic) else row[key]
        if key == "observed_at":
            timestamp = pd.Timestamp(value)
            timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
            value = timestamp.isoformat()
        elif key in numeric:
            value = float(value)
        else:
            value = str(value)
        payload[key] = value
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_shadow_log(frame: pd.DataFrame) -> pd.DataFrame:
    if missing := set(SHADOW_COLUMNS) - set(frame.columns):
        raise ValueError(f"shadow log missing columns: {sorted(missing)}")
    result = frame[list(SHADOW_COLUMNS)].copy()
    result["observed_at"] = pd.to_datetime(result["observed_at"], utc=True)
    if not result["observed_at"].is_monotonic_increasing:
        raise ValueError("shadow observations must be chronological")
    numeric = [
        "quantity", "predicted_edge_bps", "intended_price", "observed_bid",
        "observed_ask", "reference_exit_price",
    ]
    if not np.isfinite(result[numeric].to_numpy(float)).all():
        raise ValueError("shadow numeric fields must be finite")
    if (result["quantity"] <= 0).any() or (result[["intended_price", "observed_bid", "observed_ask", "reference_exit_price"]] <= 0).any().any():
        raise ValueError("shadow quantities and prices must be positive")
    if (result["observed_ask"] < result["observed_bid"]).any():
        raise ValueError("shadow ask cannot be below bid")
    previous = "GENESIS"
    for index, row in result.iterrows():
        if str(row["previous_row_sha256"]) != previous:
            raise ValueError(f"shadow hash chain broken at row {index}")
        expected = shadow_row_fingerprint(row.to_dict())
        if str(row["row_sha256"]) != expected:
            raise ValueError(f"shadow row hash mismatch at row {index}")
        previous = expected
    return result


def append_shadow_observation(path: Path, row: dict[str, Any]) -> pd.DataFrame:
    supplied = set(SHADOW_COLUMNS) - {"previous_row_sha256", "row_sha256"}
    if missing := supplied - set(row):
        raise ValueError(f"shadow row missing columns: {sorted(missing)}")
    existing = validate_shadow_log(pd.read_csv(path)) if path.exists() else None
    previous = "GENESIS" if existing is None else str(existing.iloc[-1]["row_sha256"])
    chained = {key: row[key] for key in supplied}
    chained["previous_row_sha256"] = previous
    chained["row_sha256"] = shadow_row_fingerprint(chained)
    new = pd.DataFrame([{key: chained[key] for key in SHADOW_COLUMNS}])
    combined = new if existing is None else pd.concat([existing, new], ignore_index=True)
    combined = validate_shadow_log(combined)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined
