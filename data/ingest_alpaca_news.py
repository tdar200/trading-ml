"""
Stage-6 news ingest: pull Alpaca free-tier news headlines for the
universe + benchmark over the catalog window, write sharded parquet.

Idempotent: re-runs only fetch days not already present in the per-ticker
shard files. Headlines-only by design (free tier returns no body text
historically); body text is preserved when Alpaca returns it (recent days).

Usage:
    uv run python data/ingest_alpaca_news.py            # full window
    uv run python data/ingest_alpaca_news.py --since 2024-01-01
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from dotenv import load_dotenv

from data.tickers import ALL_TICKERS

load_dotenv(_PROJECT_ROOT / ".env")

NEWS_DIR = _PROJECT_ROOT / "data" / "news"
# Catalog window starts ~2020-07; news API supports this range on free tier.
DEFAULT_SINCE = pd.Timestamp("2020-07-01", tz="UTC")
# Alpaca paginates news; this is the max page size for the free endpoint.
PAGE_LIMIT = 50


# ---- pure helpers (tested in tests/test_news_ingest.py) -------------------

def article_to_row(article) -> dict:
    """Normalize an Alpaca news article object into a flat dict row.

    `symbols` is comma-joined for parquet stability (lists are awkward
    in pyarrow's default schema).
    """
    syms = getattr(article, "symbols", None) or []
    return {
        "id": int(getattr(article, "id", 0)),
        "headline": str(getattr(article, "headline", "") or ""),
        "symbols": ",".join(syms),
        "created_at": pd.Timestamp(getattr(article, "created_at", pd.NaT)),
        "source": str(getattr(article, "source", "") or ""),
        "summary": str(getattr(article, "summary", "") or ""),
    }


def dedup_by_id(rows: list[dict]) -> list[dict]:
    """Drop duplicate-id rows, keeping the first occurrence."""
    seen: set[int] = set()
    out: list[dict] = []
    for r in rows:
        rid = r.get("id")
        if rid in seen:
            continue
        seen.add(rid)
        out.append(r)
    return out


def shard_path(base_dir: Path, ticker: str, ts: pd.Timestamp) -> Path:
    """Return the per-(ticker, year-month) shard path."""
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return Path(base_dir) / ticker / f"{ts.strftime('%Y-%m')}.parquet"


# ---- orchestration ---------------------------------------------------------

def _load_existing(ticker: str) -> pd.DataFrame:
    """Read all existing shards for a ticker, return concatenated DataFrame."""
    ticker_dir = NEWS_DIR / ticker
    if not ticker_dir.exists():
        return pd.DataFrame(columns=["id", "headline", "symbols", "created_at", "source", "summary"])
    parts = []
    for shard in sorted(ticker_dir.glob("*.parquet")):
        parts.append(pd.read_parquet(shard))
    if not parts:
        return pd.DataFrame(columns=["id", "headline", "symbols", "created_at", "source", "summary"])
    return pd.concat(parts, ignore_index=True)


def _write_shards(rows: list[dict]) -> int:
    """Write rows to per-(ticker, month) shards. Returns count written."""
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["month"] = df["created_at"].dt.strftime("%Y-%m")
    # Each article tags multiple tickers; explode so each ticker's shard
    # gets a copy. This matches downstream lookups (sentiment_store
    # queries by ticker).
    expanded: list[dict] = []
    for _, row in df.iterrows():
        tags = [t for t in str(row["symbols"]).split(",") if t]
        # Restrict to tickers we trade — drop noise from unrelated tags.
        for t in tags:
            if t not in ALL_TICKERS:
                continue
            r = row.to_dict()
            r["ticker"] = t
            expanded.append(r)
    if not expanded:
        return 0
    exp_df = pd.DataFrame(expanded)
    written = 0
    for (ticker, month), group in exp_df.groupby(["ticker", "month"]):
        shard = NEWS_DIR / ticker / f"{month}.parquet"
        shard.parent.mkdir(parents=True, exist_ok=True)
        if shard.exists():
            existing = pd.read_parquet(shard)
            combined = pd.concat([existing, group], ignore_index=True)
            combined = combined.drop_duplicates(subset=["id"], keep="first")
        else:
            combined = group.drop_duplicates(subset=["id"], keep="first")
        cols = ["id", "headline", "symbols", "created_at", "source", "summary"]
        combined[cols].to_parquet(shard, index=False)
        written += len(group)
    return written


def _fetch_window(start: pd.Timestamp, end: pd.Timestamp) -> list:
    """Page through Alpaca's news endpoint for the universe in this window."""
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    client = NewsClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_API_SECRET"),
    )
    symbols_csv = ",".join(ALL_TICKERS)
    page_token = None
    all_articles: list = []
    while True:
        req = NewsRequest(
            start=start.to_pydatetime(),
            end=end.to_pydatetime(),
            symbols=symbols_csv,
            limit=PAGE_LIMIT,
            page_token=page_token,
            include_content=True,
        )
        resp = client.get_news(req)
        items = resp.data.get("news", []) if hasattr(resp, "data") else list(resp)
        all_articles.extend(items)
        page_token = (
            resp.data.get("next_page_token") if hasattr(resp, "data") else None
        )
        if not page_token or not items:
            break
    return all_articles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--since", type=str, default=None,
        help="ISO date (UTC) to start from. Default: 2020-07-01 or latest existing.",
    )
    args = parser.parse_args()

    NEWS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine effective start: max of (--since, latest existing across tickers, DEFAULT_SINCE).
    if args.since:
        start = pd.Timestamp(args.since, tz="UTC")
    else:
        existing_maxes = []
        for ticker in ALL_TICKERS:
            df = _load_existing(ticker)
            if not df.empty:
                existing_maxes.append(pd.to_datetime(df["created_at"], utc=True).max())
        if existing_maxes:
            # Re-fetch from the day AFTER the latest existing (avoid full re-pull).
            start = max(existing_maxes).normalize() + pd.Timedelta(days=1)
        else:
            start = DEFAULT_SINCE

    end = pd.Timestamp.now(tz="UTC")
    if start >= end:
        print(f"nothing to fetch: start ({start.date()}) >= end ({end.date()})")
        return 0

    print(f"fetching news from {start.date()} to {end.date()} for {len(ALL_TICKERS)} tickers...")

    # Paginate in 30-day chunks to stay under any per-call limits and to
    # surface progress.
    total_written = 0
    cursor = start
    chunk = pd.Timedelta(days=30)
    while cursor < end:
        chunk_end = min(cursor + chunk, end)
        articles = _fetch_window(cursor, chunk_end)
        rows = dedup_by_id([article_to_row(a) for a in articles])
        written = _write_shards(rows)
        total_written += written
        print(f"  {cursor.date()}..{chunk_end.date()}: fetched {len(articles)}, wrote {written}")
        cursor = chunk_end

    print(f"done: wrote {total_written} (ticker, article) rows under {NEWS_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
