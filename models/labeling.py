"""
Holding-period-return-sign labels for meta-labeling.

For each (ticker, rebalance_date) pair, label = 1 if the close `horizon_days`
business days later is strictly above today's close, else 0. Pairs without
enough forward data (last `horizon_days` of the series) are dropped.

Pure pandas; no Nautilus. Output schema: [ticker, date, label].
"""
from __future__ import annotations

import pandas as pd


def make_labels(
    bars_by_ticker: dict[str, pd.Series],
    rebalance_dates: pd.DatetimeIndex,
    *,
    horizon_days: int = 21,
) -> pd.DataFrame:
    """Build a (ticker, date, label) DataFrame.

    Args:
        bars_by_ticker: each ticker's daily close series, indexed by date.
        rebalance_dates: dates at which to label positions.
        horizon_days: how many business days forward to look (default 21 ≈ 1 month).
    """
    rows: list[dict] = []
    for ticker, closes in bars_by_ticker.items():
        for date in rebalance_dates:
            if date not in closes.index:
                continue
            spot_pos = closes.index.get_loc(date)
            if spot_pos + horizon_days >= len(closes):
                continue
            spot = float(closes.iloc[spot_pos])
            forward = float(closes.iloc[spot_pos + horizon_days])
            label = 1 if (forward > spot) else 0
            rows.append({"ticker": ticker, "date": date, "label": label})
    return pd.DataFrame(rows, columns=["ticker", "date", "label"])
