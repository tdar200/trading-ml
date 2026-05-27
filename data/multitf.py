"""
Stage-9 multi-timeframe loader. One entry point: ``get_bars(ticker, timeframe, ...)``.

Branches:
- **intraday** (``1m, 5m, 15m, 30m, 1h, 4h``) — load from
  ``data/intraday/<ticker>/<tf>/<YYYY-MM>.parquet`` (Stage 9 ingest).
- **daily** (``1d``) — load from Stage 1's ``ParquetDataCatalog`` at
  ``data/catalog/``.
- **derived** (``1w, 1M, 1Y``) — resample from the daily series via pandas
  OHLC aggregation. Avoids storing pre-aggregated weekly/monthly/yearly
  parquet (which would drift from daily on data fixes).

Used by ``dashboard/views/equity.py`` to feed the TradingView-style
timeframe switcher.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_INTRADAY_ROOT = _PROJECT_ROOT / "data" / "intraday"
DEFAULT_CATALOG_PATH = _PROJECT_ROOT / "data" / "catalog"

INTRADAY_TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h", "4h")
RESAMPLE_TIMEFRAMES = ("1w", "1M", "1Y")
SUPPORTED_TIMEFRAMES = INTRADAY_TIMEFRAMES + ("1d",) + RESAMPLE_TIMEFRAMES

# Map our timeframe strings to pandas resample rules.
_RESAMPLE_RULE: dict[str, str] = {
    "1w": "W",
    "1M": "ME",   # pandas 2.x: 'ME' = month-end (deprecated 'M' still works)
    "1Y": "YE",
}


# ---- public API ----------------------------------------------------------

def get_bars(
    ticker: str,
    timeframe: str,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    intraday_root: Path = DEFAULT_INTRADAY_ROOT,
    catalog_path: Path = DEFAULT_CATALOG_PATH,
) -> pd.DataFrame:
    """Return OHLCV bars for `ticker` at `timeframe` in [start, end).

    Output columns: [ts, open, high, low, close, volume]. Empty DataFrame
    (correct columns, 0 rows) if no data covers the window.

    Daily / weekly / monthly / yearly loads delegate to the same Stage-1
    catalog so the daily-derived series stay consistent.
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"unknown timeframe '{timeframe}'; supported: {SUPPORTED_TIMEFRAMES}"
        )

    if timeframe in INTRADAY_TIMEFRAMES:
        df = _load_intraday(ticker, timeframe, intraday_root)
    elif timeframe == "1d":
        df = _load_daily(ticker, catalog_path)
    else:
        daily = _load_daily(ticker, catalog_path)
        df = _resample_ohlcv(daily, _RESAMPLE_RULE[timeframe])

    return _slice_window(df, start, end)


# ---- branch helpers -----------------------------------------------------

def _load_intraday(ticker: str, timeframe: str, root: Path) -> pd.DataFrame:
    """Concatenate all parquet shards under root/<ticker>/<timeframe>/."""
    tf_dir = root / ticker / timeframe
    if not tf_dir.exists():
        return _empty_bars()
    parts = [pd.read_parquet(p) for p in sorted(tf_dir.glob("*.parquet"))]
    if not parts:
        return _empty_bars()
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df[["ts", "open", "high", "low", "close", "volume"]].sort_values("ts").reset_index(drop=True)


def _load_daily(ticker: str, catalog_path: Path) -> pd.DataFrame:
    """Read daily bars from the Stage-1 ParquetDataCatalog. Returns one
    row per business day with [ts, open, high, low, close, volume]."""
    if not catalog_path.exists():
        return _empty_bars()
    # Lazy-imported so unit tests that don't exercise catalog loading
    # don't pay Nautilus's import cost.
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
    from data.instruments import VENUE_MIC

    catalog = ParquetDataCatalog(path=str(catalog_path.resolve()))
    try:
        bars = catalog.bars(instrument_ids=[f"{ticker}.{VENUE_MIC}"])
    except Exception:
        return _empty_bars()
    if not bars:
        return _empty_bars()
    rows = [{
        "ts": pd.Timestamp(b.ts_event, unit="ns", tz="UTC"),
        "open": float(b.open),
        "high": float(b.high),
        "low": float(b.low),
        "close": float(b.close),
        "volume": float(b.volume),
    } for b in bars]
    df = pd.DataFrame(rows)
    return df.sort_values("ts").reset_index(drop=True)


def _resample_ohlcv(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    """OHLC aggregate a daily bars frame to a coarser timeframe."""
    if daily.empty:
        return _empty_bars()
    df = daily.set_index("ts").sort_index()
    agg = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])
    return agg.reset_index()


def _slice_window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if df.empty:
        return df
    if start is not None:
        start_ts = pd.Timestamp(start)
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize("UTC")
        df = df[df["ts"] >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end)
        if end_ts.tz is None:
            end_ts = end_ts.tz_localize("UTC")
        df = df[df["ts"] < end_ts]
    return df.reset_index(drop=True)


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
