"""
Stage-9 intraday news ingest: pull Alpaca free-tier intraday bars (1m / 5m
/ 15m / 30m / 1h / 4h) for the universe, write per-(ticker, timeframe, month)
parquet shards.

Idempotent — re-runs only fetch days not already present. Free-tier IEX feed
gives ~5 years of intraday history, which is plenty for the Stage-9
dashboard's TradingView-style timeframe range without paying for Algo Trader
Plus or Polygon.

Layout: ``data/intraday/<ticker>/<timeframe>/<YYYY-MM>.parquet``
(sibling of Stage 1's ``data/catalog/`` for daily bars).

Usage:
    uv run python data/ingest_alpaca_intraday.py                    # default 5m, all tickers, last 7 days
    uv run python data/ingest_alpaca_intraday.py --timeframe 1m --since 2024-01-01
    uv run python data/ingest_alpaca_intraday.py --timeframe 1h --tickers SPY,XLK
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from dotenv import load_dotenv

from data.multitf import storage_symbol
from data.tickers import BENCHMARK, SCALPING_CRYPTO, SCALPING_SECURITIES

load_dotenv(_PROJECT_ROOT / ".env")

INTRADAY_DIR = _PROJECT_ROOT / "data" / "intraday"

# Canonical timeframe strings we accept on the CLI / from the dashboard.
# Maps to the alpaca-py TimeFrame constructor calls. Daily and above are
# served from data/catalog/ (Stage 1), not from here.
_TIMEFRAME_TO_ALPACA: dict[str, tuple[int, str]] = {
    # Maps to TimeFrameUnit *value* strings (Min/Hour), not names.
    "1m": (1, "Min"),
    "5m": (5, "Min"),
    "15m": (15, "Min"),
    "30m": (30, "Min"),
    "1h": (1, "Hour"),
    "4h": (4, "Hour"),
}


# ---- pure helpers (tested in tests/test_intraday_ingest.py) -----------------

def parse_timeframe(tf: str):
    """Map a canonical timeframe string (e.g. '5m', '1h') to an alpaca-py
    ``TimeFrame``. Raises ``ValueError`` for unknown strings.

    Daily / weekly / monthly / yearly are served from Stage 1's
    ``data/catalog/``, not from this module — passing '1d' raises.
    """
    if tf not in _TIMEFRAME_TO_ALPACA:
        raise ValueError(
            f"unknown intraday timeframe '{tf}'; supported: {list(_TIMEFRAME_TO_ALPACA)}"
        )
    amount, unit = _TIMEFRAME_TO_ALPACA[tf]
    # Local import keeps unit tests fast.
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    return TimeFrame(amount, TimeFrameUnit(unit))


def shard_path(base_dir: Path, ticker: str, timeframe: str, ts: pd.Timestamp) -> Path:
    """Return the per-(ticker, timeframe, month) parquet shard path.

    Different from Stage 1's daily shards — adds the timeframe level so
    1m and 1h for the same ticker land in separate files.
    """
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    if timeframe not in _TIMEFRAME_TO_ALPACA:
        raise ValueError(f"unknown intraday timeframe '{timeframe}'")
    return Path(base_dir) / storage_symbol(ticker) / timeframe / f"{ts.strftime('%Y-%m')}.parquet"


def _row_from_bar(ticker: str, ts: pd.Timestamp, ohlcv: dict) -> dict:
    """Normalize one bar into a flat dict row."""
    return {
        "ticker": ticker,
        "ts": pd.Timestamp(ts),
        "open": float(ohlcv["open"]),
        "high": float(ohlcv["high"]),
        "low": float(ohlcv["low"]),
        "close": float(ohlcv["close"]),
        "volume": float(ohlcv.get("volume", 0.0)),
    }


# ---- orchestration ---------------------------------------------------------

def _existing_shards(ticker: str, timeframe: str) -> pd.DataFrame:
    ticker_dir = INTRADAY_DIR / storage_symbol(ticker) / timeframe
    if not ticker_dir.exists():
        return pd.DataFrame(columns=["ticker", "ts", "open", "high", "low", "close", "volume"])
    parts = [pd.read_parquet(p) for p in sorted(ticker_dir.glob("*.parquet"))]
    if not parts:
        return pd.DataFrame(columns=["ticker", "ts", "open", "high", "low", "close", "volume"])
    return pd.concat(parts, ignore_index=True)


def _write_shards(rows: list[dict], timeframe: str) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["month"] = df["ts"].dt.strftime("%Y-%m")
    written = 0
    for (ticker, month), group in df.groupby(["ticker", "month"]):
        shard = INTRADAY_DIR / storage_symbol(ticker) / timeframe / f"{month}.parquet"
        shard.parent.mkdir(parents=True, exist_ok=True)
        if shard.exists():
            existing = pd.read_parquet(shard)
            combined = pd.concat([existing, group], ignore_index=True)
            combined = combined.drop_duplicates(subset=["ts"], keep="last")
        else:
            combined = group.drop_duplicates(subset=["ts"], keep="last")
        cols = ["ticker", "ts", "open", "high", "low", "close", "volume"]
        combined[cols].sort_values("ts").to_parquet(shard, index=False)
        written += len(group)
    return written


def _fetch_window(tickers: list[str], timeframe: str,
                  start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """Page through Alpaca's intraday bars for the given window."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest

    client = StockHistoricalDataClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_API_SECRET"),
    )
    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=parse_timeframe(timeframe),
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        adjustment=Adjustment.ALL,
        feed=DataFeed.IEX,  # free-tier
    )
    response = client.get_stock_bars(request)
    if response.df is None or response.df.empty:
        return []
    frame = response.df.reset_index()
    return [
        _row_from_bar(
            ticker=str(row.symbol),
            ts=row.timestamp,
            ohlcv={
                "open": row.open, "high": row.high, "low": row.low,
                "close": row.close, "volume": row.volume,
            },
        )
        for row in frame.itertuples(index=False)
    ]


def _fetch_crypto_window(
    tickers: list[str],
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict]:
    """Fetch Alpaca US crypto bars for a bounded time window."""
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest

    client = CryptoHistoricalDataClient()
    request = CryptoBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=parse_timeframe(timeframe),
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
    )
    response = client.get_crypto_bars(request)
    if response.df is None or response.df.empty:
        return []
    frame = response.df.reset_index()
    return [
        _row_from_bar(
            ticker=str(row.symbol),
            ts=row.timestamp,
            ohlcv={
                "open": row.open, "high": row.high, "low": row.low,
                "close": row.close, "volume": row.volume,
            },
        )
        for row in frame.itertuples(index=False)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-class", choices=("stock", "crypto"), default="stock",
        help="Alpaca market-data asset class (default: stock).",
    )
    parser.add_argument(
        "--timeframe", default="5m",
        choices=list(_TIMEFRAME_TO_ALPACA),
        help="Intraday bar size (default 5m).",
    )
    parser.add_argument(
        "--chunk-days", type=int, default=None,
        help="Override request chunk length; useful for large symbol baskets.",
    )
    parser.add_argument(
        "--since", default=None,
        help="ISO date (UTC). Default: 7 days ago for minute timeframes, "
             "60 days ago for hourly.",
    )
    parser.add_argument(
        "--until", default=None,
        help="ISO date (UTC). Default: now.",
    )
    parser.add_argument(
        "--tickers", default=None,
        help="Comma-separated symbols. Defaults to the scalping securities or crypto basket.",
    )
    args = parser.parse_args()

    INTRADAY_DIR.mkdir(parents=True, exist_ok=True)

    if args.tickers is None:
        tickers = list(
            SCALPING_CRYPTO
            if args.asset_class == "crypto"
            else (*SCALPING_SECURITIES, BENCHMARK)
        )
    else:
        tickers = [ticker.strip() for ticker in args.tickers.split(",") if ticker.strip()]

    if args.until:
        end = pd.Timestamp(args.until, tz="UTC")
    else:
        end = pd.Timestamp.now(tz="UTC")

    if args.since:
        start = pd.Timestamp(args.since, tz="UTC")
    else:
        # Sensible defaults: 7 days for sub-hour, 60 days for hourly+.
        days_back = 7 if args.timeframe.endswith("m") else 60
        start = end - pd.Timedelta(days=days_back)

    if start >= end:
        print(f"nothing to fetch: start ({start.date()}) >= end ({end.date()})")
        return 0

    print(f"fetching {args.asset_class} {args.timeframe} bars from {start.date()} "
          f"to {end.date()} for {len(tickers)} symbols...")

    # Paginate in 7-day chunks for minute timeframes, 30-day for hourly.
    # Keeps individual responses small (Alpaca caps page size).
    chunk_days = args.chunk_days or (7 if args.timeframe.endswith("m") else 30)
    chunk = pd.Timedelta(days=chunk_days)
    cursor = start
    total_written = 0
    while cursor < end:
        chunk_end = min(cursor + chunk, end)
        fetcher = (
            _fetch_crypto_window
            if args.asset_class == "crypto"
            else _fetch_window
        )
        rows = fetcher(tickers, args.timeframe, cursor, chunk_end)
        written = _write_shards(rows, args.timeframe)
        total_written += written
        print(f"  {cursor.date()}..{chunk_end.date()}: fetched {len(rows)}, wrote {written}")
        cursor = chunk_end

    print(f"done: wrote {total_written} bars under {INTRADAY_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
