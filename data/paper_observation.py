"""Append-only, fingerprint-bound paper observation journal."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from analysis.deployment_gate import PAPER_COLUMNS, paper_row_fingerprint, validate_paper_log


def append_paper_session(path: Path, row: dict[str, Any]) -> pd.DataFrame:
    supplied_columns = set(PAPER_COLUMNS) - {"previous_row_sha256", "row_sha256"}
    if supplied_columns - set(row):
        raise ValueError(f"paper row missing columns: {sorted(supplied_columns - set(row))}")
    if path.exists():
        existing = validate_paper_log(pd.read_csv(path))
        if existing["candidate_sha256"].iloc[0] != str(row["candidate_sha256"]):
            raise ValueError("candidate fingerprint changed; start a new paper journal")
        previous_hash = str(existing["row_sha256"].iloc[-1])
    else:
        existing = None
        previous_hash = "GENESIS"
    chained = {column: row[column] for column in supplied_columns}
    chained["previous_row_sha256"] = previous_hash
    chained["row_sha256"] = paper_row_fingerprint(chained)
    new = pd.DataFrame([{column: chained[column] for column in PAPER_COLUMNS}])
    combined = new if existing is None else pd.concat([existing, new], ignore_index=True)
    combined = validate_paper_log(combined)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined
