"""Ingest historical Alpaca SIP quotes into validated monthly shards.

This intentionally requests the consolidated SIP feed.  It does not fall back
to IEX because an IEX quote is not an NBBO-equivalent execution-cost input.
An Alpaca subscription with historical SIP quote entitlement is required.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.multitf import storage_symbol
from data.quotes import DEFAULT_QUOTE_ROOT, QUOTE_COLUMNS, validate_quotes
from data.tickers import SCALPING_STOCKS

load_dotenv(ROOT / ".env")


def normalize_quote_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the multi-index/dataframe shape returned by alpaca-py."""
    source = frame.reset_index() if isinstance(frame.index, pd.MultiIndex) else frame.copy()
    aliases = {
        "timestamp": "ts", "bid_price": "bid_price", "ask_price": "ask_price",
        "bid_size": "bid_size", "ask_size": "ask_size",
    }
    source = source.rename(columns=aliases)
    if "symbol" not in source:
        raise ValueError("Alpaca quote response is missing symbol")
    normalized = source[["symbol", *QUOTE_COLUMNS]].copy()
    normalized["symbol"] = normalized["symbol"].astype(str).str.upper()
    normalized["ts"] = pd.to_datetime(normalized["ts"], utc=True)
    for ticker, group in normalized.groupby("symbol", sort=False):
        validate_quotes(group[list(QUOTE_COLUMNS)])
    return normalized.sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)


def write_quote_shards(frame: pd.DataFrame, *, root: Path = DEFAULT_QUOTE_ROOT) -> int:
    normalized = normalize_quote_frame(frame)
    normalized["month"] = normalized["ts"].dt.strftime("%Y-%m")
    written = 0
    for (ticker, month), group in normalized.groupby(["symbol", "month"], sort=False):
        path = root / storage_symbol(ticker) / f"{month}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        values = group[list(QUOTE_COLUMNS)]
        if path.exists():
            values = pd.concat([pd.read_parquet(path), values], ignore_index=True)
        values = validate_quotes(values)
        values.to_parquet(path, index=False)
        written += len(group)
    return written


def fetch_quotes(tickers: tuple[str, ...], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockQuotesRequest

    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_API_SECRET"):
        raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET are required")
    client = StockHistoricalDataClient(
        api_key=os.environ["ALPACA_API_KEY"], secret_key=os.environ["ALPACA_API_SECRET"],
    )
    request = StockQuotesRequest(
        symbol_or_symbols=list(tickers), start=start.to_pydatetime(), end=end.to_pydatetime(),
        feed=DataFeed.SIP,
    )
    response = client.get_stock_quotes(request)
    if response.df is None or response.df.empty:
        raise RuntimeError("SIP returned no quotes; no lower-quality feed was substituted")
    return response.df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=",".join(SCALPING_STOCKS))
    parser.add_argument("--since", required=True)
    parser.add_argument("--until", required=True)
    args = parser.parse_args()
    tickers = tuple(dict.fromkeys(x.strip().upper() for x in args.tickers.split(",") if x.strip()))
    start, end = pd.Timestamp(args.since, tz="UTC"), pd.Timestamp(args.until, tz="UTC")
    if not tickers or end <= start:
        raise SystemExit("tickers and a positive [since, until) interval are required")
    try:
        count = write_quote_shards(fetch_quotes(tickers, start, end))
    except Exception as exc:
        raise SystemExit(f"SIP quote ingest failed closed: {exc}") from exc
    print(f"wrote {count:,} SIP quotes for {len(tickers)} symbols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
