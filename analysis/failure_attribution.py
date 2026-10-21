"""Transaction-level attribution for rejected profitability candidates."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "session_date", "ticker", "side", "gross_contribution",
    "cost_contribution", "net_contribution", "high_information", "panic_rebound",
}


def summarize_ledger(ledger: pd.DataFrame, groups: Iterable[str]) -> pd.DataFrame:
    if missing := REQUIRED_COLUMNS - set(ledger.columns):
        raise ValueError(f"evaluation ledger missing columns: {sorted(missing)}")
    data = ledger.copy()
    data["session_date"] = pd.to_datetime(data["session_date"])
    data["month"] = data["session_date"].dt.to_period("M").astype(str)
    data["regime"] = np.where(
        data["panic_rebound"], "panic_rebound",
        np.where(data["high_information"], "high_information", "normal"),
    )
    group_columns = list(groups)
    invalid = set(group_columns) - set(data.columns)
    if invalid:
        raise ValueError(f"unknown attribution groups: {sorted(invalid)}")
    grouped = data.groupby(group_columns, dropna=False, sort=True)
    summary = grouped.agg(
        trades=("net_contribution", "size"),
        gross_return=("gross_contribution", "sum"),
        cost_drag=("cost_contribution", "sum"),
        net_return=("net_contribution", "sum"),
        win_rate=("net_contribution", lambda values: float((values > 0).mean())),
        mean_score=("score", "mean"),
    ).reset_index()
    summary["cost_share_of_absolute_gross"] = np.divide(
        summary["cost_drag"], summary["gross_return"].abs(),
        out=np.full(len(summary), np.inf), where=summary["gross_return"].abs() > 0,
    )
    return summary
