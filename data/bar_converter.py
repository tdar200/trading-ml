"""
Pure conversion from an Alpaca bars DataFrame to a list of Nautilus Bar objects.

Alpaca returns daily bars indexed on (symbol, timestamp) with the columns
open/high/low/close/volume (and trade_count, vwap). We slice by symbol, then
map each row to a Nautilus Bar with a fixed BarType of
"<TICKER>.ARCX-1-DAY-LAST-EXTERNAL".

No IO, no Alpaca SDK dependency — this stays unit-testable.
"""
from __future__ import annotations

import pandas as pd
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Price, Quantity

from data.instruments import VENUE_MIC


def _bar_type_for(ticker: str) -> BarType:
    return BarType.from_str(f"{ticker}.{VENUE_MIC}-1-DAY-LAST-EXTERNAL")


def _to_nanos(ts: pd.Timestamp) -> int:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.value)


def alpaca_df_to_bars(df: pd.DataFrame, ticker: str) -> list[Bar]:
    """Convert the (symbol, timestamp)-indexed Alpaca bars DataFrame for one
    ticker into a list of Nautilus Bar objects. Other symbols in the frame
    are ignored."""
    if df.empty:
        return []
    if ticker not in df.index.get_level_values(0):
        return []

    rows = df.xs(ticker, level=0).sort_index()
    bar_type = _bar_type_for(ticker)
    bars: list[Bar] = []
    for ts, row in rows.iterrows():
        ts_ns = _to_nanos(ts)
        bars.append(Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{float(row['open']):.2f}"),
            high=Price.from_str(f"{float(row['high']):.2f}"),
            low=Price.from_str(f"{float(row['low']):.2f}"),
            close=Price.from_str(f"{float(row['close']):.2f}"),
            volume=Quantity.from_int(int(row["volume"])),
            ts_event=ts_ns,
            ts_init=ts_ns,
        ))
    return bars
