"""
Validate Alpaca-ingested daily closes against yfinance for one ticker.

Asserts max relative drift on adjusted close < tolerance over the overlap window.
Run after a successful ingest. Exits non-zero on failure for easy CI wiring.

Default tolerance is 1.5% (calibrated for free-tier DataFeed.IEX vs yfinance SIP):
  - Alpaca free tier uses IEX (single-exchange feed); yfinance uses the SIP-consolidated
    tape. Legitimate venue spread is ~1% worst-case on isolated days (e.g. SPDR
    ex-dividend days). Median drift is ~0.01%; p99 is ~0.06%; only ~0.3% of days
    exceed 0.1%. The 1.5% tolerance accepts this venue noise while still catching
    adjustment bugs (which would be many percent off, every day).
  - Pass --tolerance to override from the CLI (e.g. tighten to 0.005 when on a
    paid SIP feed where both sides reflect the consolidated tape).

Usage:
    uv run python scripts/cross_check_yfinance.py --ticker SPY --start 2024-01-01 --end 2024-06-30
    uv run python scripts/cross_check_yfinance.py --ticker SPY --start 2024-01-01 --tolerance 0.005
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on sys.path when the script is run directly
# (e.g. `uv run python scripts/cross_check_yfinance.py`).
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

from data.instruments import VENUE_MIC


def alpaca_closes_from_catalog(catalog_path: Path, ticker: str) -> pd.Series:
    catalog = ParquetDataCatalog(path=str(catalog_path))
    bars = catalog.bars(instrument_ids=[f"{ticker}.{VENUE_MIC}"])
    rows = [
        (pd.Timestamp(b.ts_event, unit="ns", tz="UTC").normalize(), float(b.close))
        for b in bars
    ]
    if not rows:
        return pd.Series(dtype=float, name="alpaca_close")
    s = pd.Series(dict(rows), name="alpaca_close").sort_index()
    s.index = s.index.tz_convert(None)
    return s


def yfinance_closes(ticker: str, start: str, end: str) -> pd.Series:
    # NOTE: Recent yfinance returns MultiIndex columns when downloading even a
    # single ticker (e.g. ("Close", "SPY"), ("High", "SPY"), ...).
    # Pass multi_level_index=False to get flat columns ("Close", "High", ...)
    # so df["Close"] is directly a 1-D Series.
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    if df.empty:
        return pd.Series(dtype=float, name="yfinance_close")
    s = df["Close"].rename("yfinance_close")
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


def cross_check(catalog_path: Path, ticker: str, start: str, end: str, tolerance: float = 0.015) -> int:
    a = alpaca_closes_from_catalog(catalog_path, ticker)
    y = yfinance_closes(ticker, start, end)
    if a.empty or y.empty:
        print(f"FAIL: no overlap (alpaca={len(a)}, yfinance={len(y)})")
        return 2

    joined = pd.concat([a, y], axis=1, join="inner").dropna()
    if joined.empty:
        print("FAIL: no overlapping dates between Alpaca catalog and yfinance")
        return 2

    rel_drift = (joined["alpaca_close"] - joined["yfinance_close"]).abs() / joined["yfinance_close"]
    max_drift = float(rel_drift.max())
    print(f"{ticker} compared days={len(joined)} max_rel_drift={max_drift:.5f}")
    if max_drift >= tolerance:
        print(f"FAIL: max drift {max_drift:.5%} >= tolerance {tolerance:.2%}")
        worst = rel_drift.sort_values(ascending=False).head(5)
        print("worst rows:\n", joined.loc[worst.index].assign(rel=worst))
        return 1
    print("OK")
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Cross-check Alpaca-ingested daily closes against yfinance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD (default: 2024-01-01)")
    parser.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"), help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--catalog", default="data/catalog", help="Path to Parquet data catalog (default: data/catalog)")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.015,
        help=(
            "Max allowed relative drift between Alpaca close and yfinance close "
            "(default: 0.015 = 1.5%%, calibrated for free-tier IEX vs SIP venue spread). "
            "Tighten to e.g. 0.005 when on a paid SIP feed."
        ),
    )
    args = parser.parse_args()
    return cross_check(Path(args.catalog), args.ticker, args.start, args.end, tolerance=args.tolerance)


if __name__ == "__main__":
    sys.exit(main())
