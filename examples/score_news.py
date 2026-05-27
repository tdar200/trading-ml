"""
Stage-6 sentiment scoring runnable: read ingested news, score un-scored
headlines via the configured sentiment scorer (currently VADER per the
nlp/sentiment.py docstring's VADER-for-FinBERT substitution note;
restoring FinBERT is a drop-in change to ``nlp.sentiment._load_pipeline``),
persist via SentimentStore.

Usage:
    uv run python examples/score_news.py                  # full
    uv run python examples/score_news.py --smoke          # ~100 headlines
    uv run python examples/score_news.py --batch-size 64  # tuning (VADER ignores)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from data.tickers import ALL_TICKERS
from nlp.sentiment import score_headlines
from nlp.sentiment_store import SentimentStore


NEWS_DIR = _PROJECT_ROOT / "data" / "news"
CACHE_DIR = _PROJECT_ROOT / "nlp" / "cache"
MODEL_VERSION = "finbert-v1"


def _load_all_news() -> pd.DataFrame:
    """Concatenate every per-ticker shard into one DataFrame keyed by (id, ticker)."""
    rows = []
    for ticker_dir in sorted(NEWS_DIR.glob("*")):
        if not ticker_dir.is_dir():
            continue
        for shard in sorted(ticker_dir.glob("*.parquet")):
            df = pd.read_parquet(shard)
            df["ticker"] = ticker_dir.name
            rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["id", "ticker", "headline", "created_at"])
    df = pd.concat(rows, ignore_index=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


def _already_scored_ids(store: SentimentStore) -> set:
    """Collect (ticker, id) pairs that are already in the sentiment store."""
    ids = set()
    for ticker in ALL_TICKERS:
        try:
            df = store.get_scored(
                ticker,
                start=pd.Timestamp("1990-01-01", tz="UTC"),
                end=pd.Timestamp("2100-01-01", tz="UTC"),
            )
            for _id in df["id"].tolist():
                ids.add((ticker, int(_id)))
        except Exception:
            continue
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Score only the first ~100 headlines.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    store = SentimentStore(cache_dir=CACHE_DIR, model_version=MODEL_VERSION)

    news_df = _load_all_news()
    if news_df.empty:
        print(f"ERROR: no news under {NEWS_DIR}/. Run data/ingest_alpaca_news.py first.")
        return 1
    print(f"loaded {len(news_df)} (ticker, article) rows from {NEWS_DIR}/")

    already = _already_scored_ids(store)
    news_df["key"] = list(zip(news_df["ticker"], news_df["id"].astype(int)))
    needs_score = news_df[~news_df["key"].isin(already)].copy()
    print(f"already scored: {len(already)}; need to score: {len(needs_score)}")

    if args.smoke:
        needs_score = needs_score.head(100)
        print(f"smoke mode: scoring only first {len(needs_score)} rows")

    if needs_score.empty:
        print("nothing new to score; sentiment store is up to date.")
        return 0

    # Deduplicate by (id) so we don't re-score the same article for each ticker
    # tag — score once, then fan out per ticker.
    unique = needs_score.drop_duplicates(subset=["id"], keep="first")
    print(f"scoring {len(unique)} unique articles (will fan out across tickers)...")
    headlines = unique["headline"].astype(str).tolist()
    scores = score_headlines(headlines)
    print(f"scored {len(scores)} headlines; mean={scores.mean():.4f}, min={scores.min():.4f}, max={scores.max():.4f}")

    id_to_score = dict(zip(unique["id"].astype(int), scores))
    rows_to_save = []
    for _, r in needs_score.iterrows():
        rows_to_save.append({
            "id": int(r["id"]),
            "ticker": r["ticker"],
            "created_at": r["created_at"],
            "score": float(id_to_score[int(r["id"])]),
        })
    written = store.save_scores(rows_to_save)
    print(f"done: wrote {written} (ticker, article) score rows under {CACHE_DIR}/{MODEL_VERSION}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
