# Stage 6: NLP Sentiment Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FinBERT-scored headline sentiment as a Stage-5 feature, train a v2 LightGBM grid against the wider feature set, and run an ablation harness that produces a credible verdict (measurable incremental IC or a clean rejection).

**Architecture:** New `nlp/` package (only place transformers/torch lives) + `data/ingest_alpaca_news.py` (free-tier Alpaca news → sharded parquet) + extension of `models/features.py` with 3 sentiment features behind an optional `news_provider` seam. Two trained model sets (`models/cache/` price-only, `models/cache_v2/` price+sentiment) feed the existing Stage-5 harness, plus an ablation script that compares per-window IC + DSR + permutation-test verdict.

**Tech Stack:**
- Python 3.12 (`uv`-managed)
- `transformers>=4.40`, `torch>=2.2` (NEW; CPU is acceptable for this scale)
- `huggingface_hub` (transitive via transformers)
- Existing: `lightgbm`, `joblib`, `pandas`, `numpy`, `pyarrow`, `alpaca-py`

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Add `transformers>=4.40`, `torch>=2.2` |
| `.gitignore` | Add `nlp/cache/`, `data/news/` |
| `nlp/__init__.py` | Package marker |
| `nlp/sentiment.py` | Pure-ish: `score_headlines(headlines) -> np.ndarray` (lazy-loaded FinBERT) |
| `nlp/sentiment_store.py` | `SentimentStore` — persist scored articles, query by ticker+date range |
| `data/ingest_alpaca_news.py` | Runnable: idempotent news ingest → `data/news/<ticker>/<YYYY-MM>.parquet` |
| `examples/score_news.py` | Runnable: bulk FinBERT inference over ingested news → sentiment_store cache |
| `examples/ablate_sentiment.py` | Runnable: compute v1 vs v2 IC + permutation test → write verdict markdown |
| `models/features.py` | EXTEND: add 3 sentiment features behind optional `news_provider` param |
| `examples/train_meta_models.py` | EXTEND: `--features-version v1\|v2` flag, separate cache dirs |
| `examples/run_harness_meta.py` | EXTEND: `--features-version v1\|v2` flag |
| `tests/test_news_ingest.py` | Pure tests on the article-normalization / dedup helpers |
| `tests/test_sentiment.py` | Mocked-HF tests of `score_headlines` shape + range + caching |
| `tests/test_sentiment_store.py` | `tmp_path`: roundtrip, dedup-by-hash, range query, missing-ticker |
| `tests/test_features_with_sentiment.py` | Sentiment feature computation on synthetic news + cold-start NaN |
| `tests/test_ablation.py` | IC and permutation-test computations on synthetic preds + labels |
| `nlp/cache/` | Scored-article parquets (gitignored) |
| `data/news/` | Ingested news parquets (gitignored) |
| `models/cache_v2/` | v2 (price+sentiment) trained models (gitignored, same as cache/) |

**Boundaries:**
- `nlp/` is the **only** package that imports `transformers` or `torch`.
- `data/ingest_alpaca_news.py` is the **only** module that touches Alpaca's news API.
- `models/features.py` imports from `nlp.sentiment_store` only when `news_provider` is passed — never `transformers` directly.

---

## Task 1: Scaffolding (transformers, torch, nlp package, gitignore)

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `nlp/__init__.py`
- Create: `nlp/cache/.gitkeep` and `data/news/.gitkeep` (empty markers)

- [ ] **Step 1: Add deps**

Edit `pyproject.toml`'s `[project].dependencies` and append:
```toml
"transformers>=4.40",
"torch>=2.2",
```

- [ ] **Step 2: Sync + smoke**
```bash
cd /home/tahseen-dar/Projects/trading-ml
uv sync
uv run python -c "
import transformers, torch
print('transformers', transformers.__version__)
print('torch', torch.__version__)
print('cuda?', torch.cuda.is_available())
"
```
Expected: prints both versions, `cuda?` may be `False` (acceptable — FinBERT CPU is fine for this scale).

- [ ] **Step 3: Add gitignore entries**
```bash
grep -E '^nlp/cache/?$' .gitignore || echo "nlp/cache/" >> .gitignore
grep -E '^data/news/?$' .gitignore || echo "data/news/" >> .gitignore
grep -E '^models/cache_v2/?$' .gitignore || echo "models/cache_v2/" >> .gitignore
grep -E '^nlp/cache/?$|^data/news/?$|^models/cache_v2/?$' .gitignore
```
Expected: prints the three lines.

- [ ] **Step 4: Create the nlp package**
```bash
mkdir -p nlp
touch nlp/__init__.py
```

- [ ] **Step 5: Full pytest sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 110 passed (baseline). No regressions from new deps.

- [ ] **Step 6: Commit**
```bash
git add pyproject.toml uv.lock .gitignore nlp/__init__.py
git commit -m "chore(stage6): add transformers + torch deps + nlp package scaffold

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `data/ingest_alpaca_news.py` — news ingest

**Files:**
- Create: `tests/test_news_ingest.py`
- Create: `data/ingest_alpaca_news.py`

The runnable is mostly orchestration around Alpaca's `NewsClient`. The testable pure parts are: article normalization (Alpaca response → dict row), per-ticker shard path, and dedup-by-id.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_news_ingest.py`:
```python
"""Tests for data.ingest_alpaca_news pure helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from data.ingest_alpaca_news import (
    article_to_row,
    dedup_by_id,
    shard_path,
)


@dataclass
class _FakeArticle:
    id: int
    headline: str
    symbols: list
    created_at: object
    source: str = "benzinga"
    summary: str = ""


def test_article_to_row_extracts_fields():
    art = _FakeArticle(
        id=42,
        headline="Apple raises guidance",
        symbols=["AAPL", "SPY"],
        created_at=pd.Timestamp("2024-01-02T14:30:00Z"),
        source="benzinga",
    )
    row = article_to_row(art)
    assert row["id"] == 42
    assert row["headline"] == "Apple raises guidance"
    assert row["symbols"] == "AAPL,SPY"  # comma-joined for parquet stability
    assert row["created_at"] == pd.Timestamp("2024-01-02T14:30:00Z")
    assert row["source"] == "benzinga"


def test_dedup_by_id_keeps_first():
    rows = [
        {"id": 1, "headline": "a"},
        {"id": 2, "headline": "b"},
        {"id": 1, "headline": "a-dup"},
    ]
    out = dedup_by_id(rows)
    assert len(out) == 2
    assert {r["id"] for r in out} == {1, 2}
    # First wins.
    assert next(r for r in out if r["id"] == 1)["headline"] == "a"


def test_shard_path_uses_ticker_and_month(tmp_path: Path):
    p = shard_path(tmp_path, "SPY", pd.Timestamp("2024-03-15T10:00:00Z"))
    assert p == tmp_path / "SPY" / "2024-03.parquet"


def test_shard_path_normalizes_naive_timestamp(tmp_path: Path):
    p = shard_path(tmp_path, "XLK", pd.Timestamp("2024-12-01"))
    assert p == tmp_path / "XLK" / "2024-12.parquet"
```

- [ ] **Step 2: Run to confirm failure**
```bash
cd /home/tahseen-dar/Projects/trading-ml
uv run pytest tests/test_news_ingest.py -v
```
Expected: collection error (`ModuleNotFoundError: No module named 'data.ingest_alpaca_news'`).

- [ ] **Step 3: Implement**

Create `data/ingest_alpaca_news.py`:
```python
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
from datetime import datetime, timedelta, timezone
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
```

- [ ] **Step 4: Run to confirm tests pass**
```bash
uv run pytest tests/test_news_ingest.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Smoke the runnable (skip if no Alpaca keys in .env)**
```bash
uv run python data/ingest_alpaca_news.py --since 2024-01-01 2>&1 | tail -10
```
Expected: prints fetch progress lines + "done: wrote N (ticker, article) rows". Check `data/news/` contains per-ticker subdirs with monthly parquet files.

- [ ] **Step 6: Full pytest sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 114 passed (110 + 4).

- [ ] **Step 7: Commit**
```bash
git add data/ingest_alpaca_news.py tests/test_news_ingest.py
git commit -m "feat(stage6): Alpaca free-tier news ingest with per-ticker sharded parquet

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `nlp/sentiment.py` — FinBERT scoring

**Files:**
- Create: `tests/test_sentiment.py`
- Create: `nlp/sentiment.py`

The scorer takes a list of headlines and returns a numpy array of net-sentiment values in `[-1, +1]` (= P(positive) − P(negative)). Lazy-loaded model so unit tests don't pay HF-download cost.

- [ ] **Step 1: Write the failing tests (mocked HF)**

Create `tests/test_sentiment.py`:
```python
"""Tests for nlp.sentiment — FinBERT-on-headlines scorer.

The actual FinBERT model is heavyweight; we mock the underlying pipeline
so tests are fast and offline.
"""
from __future__ import annotations

import numpy as np
import pytest

from nlp import sentiment as sentiment_mod


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Replace the lazily-loaded pipeline with a deterministic fake."""
    def fake_predict(texts):
        # Return per-text label distribution mimicking transformers' output.
        out = []
        for t in texts:
            if "good" in t.lower() or "raise" in t.lower():
                out.append([{"label": "positive", "score": 0.85},
                            {"label": "neutral", "score": 0.10},
                            {"label": "negative", "score": 0.05}])
            elif "bad" in t.lower() or "miss" in t.lower():
                out.append([{"label": "positive", "score": 0.05},
                            {"label": "neutral", "score": 0.10},
                            {"label": "negative", "score": 0.85}])
            else:
                out.append([{"label": "positive", "score": 0.20},
                            {"label": "neutral", "score": 0.60},
                            {"label": "negative", "score": 0.20}])
        return out

    monkeypatch.setattr(sentiment_mod, "_load_pipeline", lambda: fake_predict)
    # Clear any cached pipeline from prior tests.
    sentiment_mod._PIPELINE_SINGLETON = None


def test_score_headlines_returns_array_in_unit_interval(fake_pipeline):
    headlines = ["AAPL raises guidance", "TSLA misses earnings", "MSFT cloud growth"]
    scores = sentiment_mod.score_headlines(headlines)
    assert scores.shape == (3,)
    assert (scores >= -1.0).all() and (scores <= 1.0).all()


def test_score_headlines_positive_label_gives_positive_score(fake_pipeline):
    scores = sentiment_mod.score_headlines(["AAPL raises guidance"])
    assert scores[0] > 0.5


def test_score_headlines_negative_label_gives_negative_score(fake_pipeline):
    scores = sentiment_mod.score_headlines(["TSLA misses earnings"])
    assert scores[0] < -0.5


def test_score_headlines_empty_input_returns_empty_array(fake_pipeline):
    scores = sentiment_mod.score_headlines([])
    assert scores.shape == (0,)
```

- [ ] **Step 2: Run to confirm failure**
```bash
uv run pytest tests/test_sentiment.py -v
```
Expected: collection error.

- [ ] **Step 3: Implement**

Create `nlp/sentiment.py`:
```python
"""
FinBERT-on-headlines sentiment scoring.

Public API:
    score_headlines(headlines: list[str]) -> np.ndarray
        Returns net sentiment ∈ [-1, +1] per headline (= P(pos) - P(neg)).

Implementation: lazy-loaded HuggingFace `text-classification` pipeline
wrapping `ProsusAI/finbert`. CPU is fine for the project scale
(~30k historical headlines is ~5 min batched).

The model is loaded on first call and cached process-wide via
``_PIPELINE_SINGLETON``. Tests monkeypatch ``_load_pipeline`` to inject
a deterministic fake.
"""
from __future__ import annotations

import numpy as np

_MODEL_NAME = "ProsusAI/finbert"
_BATCH_SIZE = 32
_PIPELINE_SINGLETON = None


def _load_pipeline():
    """Lazily construct the HuggingFace pipeline. Heavy import; only paid on first call."""
    from transformers import pipeline  # local import — keeps unit tests fast

    pipe = pipeline(
        "text-classification",
        model=_MODEL_NAME,
        top_k=None,  # return all labels per input
        truncation=True,
        max_length=128,
    )

    def predict(texts: list[str]):
        # The pipeline returns a list-of-lists when top_k=None.
        return pipe(texts, batch_size=_BATCH_SIZE)

    return predict


def _get_pipeline():
    global _PIPELINE_SINGLETON
    if _PIPELINE_SINGLETON is None:
        _PIPELINE_SINGLETON = _load_pipeline()
    return _PIPELINE_SINGLETON


def _net_sentiment(scores: list[dict]) -> float:
    """Reduce a per-label score distribution to net = P(pos) - P(neg)."""
    pos = next((s["score"] for s in scores if s["label"].lower() == "positive"), 0.0)
    neg = next((s["score"] for s in scores if s["label"].lower() == "negative"), 0.0)
    return float(pos - neg)


def score_headlines(headlines: list[str]) -> np.ndarray:
    """Score a batch of headlines. Returns an np.ndarray of net sentiment ∈ [-1, +1]."""
    if not headlines:
        return np.zeros((0,), dtype=np.float32)
    pipe = _get_pipeline()
    results = pipe(list(headlines))
    return np.array([_net_sentiment(r) for r in results], dtype=np.float32)
```

- [ ] **Step 4: Run to confirm pass**
```bash
uv run pytest tests/test_sentiment.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Full pytest sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 118 passed (114 + 4).

- [ ] **Step 6: Commit**
```bash
git add nlp/sentiment.py tests/test_sentiment.py
git commit -m "feat(stage6): FinBERT-on-headlines sentiment scorer with lazy model load

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `nlp/sentiment_store.py` — cached scored articles

**Files:**
- Create: `tests/test_sentiment_store.py`
- Create: `nlp/sentiment_store.py`

The store persists scored articles keyed by `(article_id, model_version)`. Lookups by ticker + date range return a tidy DataFrame for downstream feature aggregation.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sentiment_store.py`:
```python
"""Tests for nlp.sentiment_store — persistent scored-article cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nlp.sentiment_store import SentimentStore


def _row(aid: int, ticker: str, ts: str, score: float) -> dict:
    return {
        "id": aid,
        "ticker": ticker,
        "created_at": pd.Timestamp(ts, tz="UTC"),
        "score": float(score),
    }


def test_save_then_lookup_roundtrip(tmp_path: Path):
    store = SentimentStore(cache_dir=tmp_path, model_version="finbert-v1")
    store.save_scores([
        _row(1, "SPY", "2024-01-02T10:00:00Z", 0.7),
        _row(2, "SPY", "2024-01-03T10:00:00Z", -0.3),
    ])
    df = store.get_scored("SPY",
                          start=pd.Timestamp("2024-01-01", tz="UTC"),
                          end=pd.Timestamp("2024-01-31", tz="UTC"))
    assert len(df) == 2
    assert set(df.columns) >= {"id", "ticker", "created_at", "score"}
    assert sorted(df["score"].tolist()) == [-0.3, 0.7]


def test_range_query_excludes_outside_window(tmp_path: Path):
    store = SentimentStore(cache_dir=tmp_path, model_version="finbert-v1")
    store.save_scores([
        _row(1, "SPY", "2023-12-15T10:00:00Z", 0.5),
        _row(2, "SPY", "2024-01-15T10:00:00Z", 0.5),
        _row(3, "SPY", "2024-02-15T10:00:00Z", 0.5),
    ])
    df = store.get_scored("SPY",
                          start=pd.Timestamp("2024-01-01", tz="UTC"),
                          end=pd.Timestamp("2024-02-01", tz="UTC"))
    assert len(df) == 1
    assert df["id"].tolist() == [2]


def test_save_dedups_by_id_within_ticker(tmp_path: Path):
    store = SentimentStore(cache_dir=tmp_path, model_version="finbert-v1")
    store.save_scores([_row(1, "SPY", "2024-01-02T10:00:00Z", 0.7)])
    store.save_scores([_row(1, "SPY", "2024-01-02T10:00:00Z", 0.9)])  # same id; keep first
    df = store.get_scored("SPY",
                          start=pd.Timestamp("2024-01-01", tz="UTC"),
                          end=pd.Timestamp("2024-01-31", tz="UTC"))
    assert len(df) == 1
    assert df["score"].iloc[0] == pytest.approx(0.7)


def test_get_scored_returns_empty_for_unknown_ticker(tmp_path: Path):
    store = SentimentStore(cache_dir=tmp_path, model_version="finbert-v1")
    df = store.get_scored("ABC",
                          start=pd.Timestamp("2024-01-01", tz="UTC"),
                          end=pd.Timestamp("2024-12-31", tz="UTC"))
    assert df.empty
```

- [ ] **Step 2: Run to confirm failure**
```bash
uv run pytest tests/test_sentiment_store.py -v
```
Expected: collection error.

- [ ] **Step 3: Implement**

Create `nlp/sentiment_store.py`:
```python
"""
SentimentStore — persist scored articles + look up by (ticker, date range).

Layout: ``{cache_dir}/{model_version}/{ticker}.parquet`` with columns
[id, ticker, created_at, score]. One file per ticker per model_version
so a model upgrade re-scores cleanly (different cache_dir / model_version).

Score is the net sentiment ∈ [-1, +1] computed by ``nlp.sentiment.score_headlines``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class SentimentStore:
    cache_dir: Path
    model_version: str

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        (self.cache_dir / self.model_version).mkdir(parents=True, exist_ok=True)

    def _ticker_path(self, ticker: str) -> Path:
        return self.cache_dir / self.model_version / f"{ticker}.parquet"

    def save_scores(self, rows: Iterable[dict]) -> int:
        """Persist scored rows. Dedups by id within each ticker, keeping first."""
        rows = list(rows)
        if not rows:
            return 0
        df = pd.DataFrame(rows)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        total_written = 0
        for ticker, group in df.groupby("ticker"):
            path = self._ticker_path(str(ticker))
            if path.exists():
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, group], ignore_index=True)
            else:
                combined = group
            combined = combined.drop_duplicates(subset=["id"], keep="first")
            combined = combined[["id", "ticker", "created_at", "score"]]
            combined.to_parquet(path, index=False)
            total_written += len(group)
        return total_written

    def get_scored(
        self,
        ticker: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return scored articles for `ticker` in [start, end) — half-open window."""
        path = self._ticker_path(ticker)
        if not path.exists():
            return pd.DataFrame(columns=["id", "ticker", "created_at", "score"])
        df = pd.read_parquet(path)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        if start.tz is None:
            start = start.tz_localize("UTC")
        if end.tz is None:
            end = end.tz_localize("UTC")
        mask = (df["created_at"] >= start) & (df["created_at"] < end)
        return df.loc[mask].reset_index(drop=True)
```

- [ ] **Step 4: Run to confirm pass**
```bash
uv run pytest tests/test_sentiment_store.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Full pytest sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 122 passed (118 + 4).

- [ ] **Step 6: Commit**
```bash
git add nlp/sentiment_store.py tests/test_sentiment_store.py
git commit -m "feat(stage6): SentimentStore — persistent scored-article cache

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `examples/score_news.py` — bulk FinBERT inference

**Files:**
- Create: `examples/score_news.py`

Runnable: read all ingested news from `data/news/`, dedup unscored articles, batch-score via `score_headlines`, write to `SentimentStore`.

- [ ] **Step 1: Implement**

Create `examples/score_news.py`:
```python
"""
Stage-6 sentiment scoring runnable: read ingested news, score un-scored
headlines via FinBERT, persist via SentimentStore.

Usage:
    uv run python examples/score_news.py                  # full
    uv run python examples/score_news.py --smoke          # ~100 headlines
    uv run python examples/score_news.py --batch-size 64  # tuning
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
```

- [ ] **Step 2: Verify --help works**
```bash
cd /home/tahseen-dar/Projects/trading-ml
uv run python examples/score_news.py --help
```
Expected: argparse help text, exit 0.

- [ ] **Step 3: Smoke run (requires Task 2 to have ingested some news)**
```bash
time uv run python examples/score_news.py --smoke 2>&1 | tail -10
```
Expected: ~30 sec on CPU (HF model first-load + ~100 inferences). Prints "scored N headlines" + "done".

- [ ] **Step 4: Full pytest sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: still 122 passed.

- [ ] **Step 5: Commit**
```bash
git add examples/score_news.py
git commit -m "feat(stage6): bulk FinBERT scoring runnable over ingested news

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Extend `models/features.py` with sentiment features

**Files:**
- Create: `tests/test_features_with_sentiment.py`
- Modify: `models/features.py`

Add 3 sentiment features behind an optional `news_provider` seam. Default `news_provider=None` preserves Stage-5 behavior (sentiment NaN).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_features_with_sentiment.py`:
```python
"""Tests for sentiment-feature extension to models.features."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.features import (
    FEATURE_COLUMNS,
    SENTIMENT_FEATURE_COLUMNS,
    extract_features_for_date,
)


def _ramp(start: str, n: int, rate: float) -> pd.Series:
    vals = [100.0]
    for _ in range(n - 1):
        vals.append(vals[-1] * (1.0 + rate))
    return pd.Series(vals, index=pd.date_range(start, periods=n, freq="B"))


def _bars(start: str, n: int) -> dict:
    return {
        "A": _ramp(start, n, 0.0020),
        "B": _ramp(start, n, 0.0010),
        "C": _ramp(start, n, 0.0005),
        "SPY": _ramp(start, n, 0.0007),
    }


class _FakeNewsProvider:
    """Simulates a SentimentStore: returns a DataFrame of [created_at, score]."""
    def __init__(self, scores_by_ticker: dict):
        self._by_ticker = scores_by_ticker

    def get_scored(self, ticker: str, *, start, end) -> pd.DataFrame:
        df = self._by_ticker.get(ticker)
        if df is None or df.empty:
            return pd.DataFrame(columns=["id", "ticker", "created_at", "score"])
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        if start.tz is None:
            start = start.tz_localize("UTC")
        if end.tz is None:
            end = end.tz_localize("UTC")
        mask = (df["created_at"] >= start) & (df["created_at"] < end)
        return df.loc[mask].reset_index(drop=True)


def test_sentiment_feature_columns_are_three():
    assert len(SENTIMENT_FEATURE_COLUMNS) == 3
    assert set(SENTIMENT_FEATURE_COLUMNS) == {
        "sentiment_7d_mean", "sentiment_7d_count", "sentiment_30d_mean",
    }


def test_extract_features_without_news_provider_omits_sentiment():
    bars = _bars("2024-01-02", 200)
    date = bars["A"].index[150]
    feats = extract_features_for_date(bars, ticker="A", date=date, benchmark="SPY")
    # Backward compat: when news_provider not passed, sentiment keys absent.
    assert "sentiment_7d_mean" not in feats


def test_extract_features_with_news_provider_includes_sentiment():
    bars = _bars("2024-01-02", 200)
    date = bars["A"].index[150]  # 2024-07-30 (approx; depends on bdate calendar)
    # Synthesize 5 positive articles in the last 7 days before `date`.
    articles = pd.DataFrame({
        "id": list(range(5)),
        "ticker": ["A"] * 5,
        "created_at": pd.to_datetime([date - pd.Timedelta(days=i) for i in (1, 2, 3, 4, 6)], utc=True),
        "score": [0.6, 0.7, 0.5, 0.4, 0.8],
    })
    provider = _FakeNewsProvider({"A": articles})
    feats = extract_features_for_date(
        bars, ticker="A", date=date, benchmark="SPY", news_provider=provider,
    )
    assert "sentiment_7d_mean" in feats
    assert feats["sentiment_7d_count"] == 5
    assert feats["sentiment_7d_mean"] == pytest.approx(0.6, rel=1e-3)


def test_extract_features_with_no_news_for_ticker_returns_nan():
    bars = _bars("2024-01-02", 200)
    date = bars["A"].index[150]
    provider = _FakeNewsProvider({})  # no news for any ticker
    feats = extract_features_for_date(
        bars, ticker="A", date=date, benchmark="SPY", news_provider=provider,
    )
    assert math.isnan(feats["sentiment_7d_mean"])
    assert math.isnan(feats["sentiment_7d_count"])
    assert math.isnan(feats["sentiment_30d_mean"])


def test_sentiment_uses_strict_before_date_no_lookahead():
    bars = _bars("2024-01-02", 200)
    date = bars["A"].index[150]
    # An article EXACTLY at `date` (UTC midnight) should be excluded by the
    # half-open window (start <= created_at < end where end = date).
    articles = pd.DataFrame({
        "id": [1, 2],
        "ticker": ["A"] * 2,
        "created_at": pd.to_datetime([
            date,  # exactly at decision time — must be excluded
            date - pd.Timedelta(days=1),
        ], utc=True),
        "score": [0.99, 0.10],  # exclude-able vs included
    })
    provider = _FakeNewsProvider({"A": articles})
    feats = extract_features_for_date(
        bars, ticker="A", date=date, benchmark="SPY", news_provider=provider,
    )
    # Only the day-before article should count.
    assert feats["sentiment_7d_count"] == 1
    assert feats["sentiment_7d_mean"] == pytest.approx(0.10)
```

- [ ] **Step 2: Run to confirm failure**
```bash
uv run pytest tests/test_features_with_sentiment.py -v
```
Expected: at least 2 failures (`SENTIMENT_FEATURE_COLUMNS` import error, missing `news_provider` arg).

- [ ] **Step 3: Implement the extension**

Open `models/features.py` and:

1. Add the new constant near the existing `FEATURE_COLUMNS`:
```python
SENTIMENT_FEATURE_COLUMNS: tuple[str, ...] = (
    "sentiment_7d_mean", "sentiment_7d_count", "sentiment_30d_mean",
)
```

2. Add a `news_provider` parameter to `extract_features_for_date`:
```python
def extract_features_for_date(
    bars_by_ticker: dict[str, pd.Series],
    *,
    ticker: str,
    date: pd.Timestamp,
    benchmark: str = "SPY",
    news_provider=None,
) -> dict[str, float]:
```

3. After computing the 11 existing price features (just before the `return {...}` block), inject the sentiment features when `news_provider` is provided:
```python
    if news_provider is not None:
        end_ts = pd.Timestamp(date)
        if end_ts.tz is None:
            end_ts = end_ts.tz_localize("UTC")
        win7_start = end_ts - pd.Timedelta(days=7)
        win30_start = end_ts - pd.Timedelta(days=30)
        df7 = news_provider.get_scored(ticker, start=win7_start, end=end_ts)
        df30 = news_provider.get_scored(ticker, start=win30_start, end=end_ts)
        sentiment_features = {
            "sentiment_7d_mean": float(df7["score"].mean()) if not df7.empty else float("nan"),
            "sentiment_7d_count": float(len(df7)) if not df7.empty else float("nan"),
            "sentiment_30d_mean": float(df30["score"].mean()) if not df30.empty else float("nan"),
        }
    else:
        sentiment_features = {}
```

4. Merge `sentiment_features` into the returned dict (keep order: existing keys, then sentiment keys).

5. Also extend `make_features` to accept and forward `news_provider`:
```python
def make_features(
    bars_by_ticker: dict[str, pd.Series],
    dates_per_ticker: dict[str, list[pd.Timestamp]],
    *,
    benchmark: str = "SPY",
    news_provider=None,
) -> pd.DataFrame:
    ...
    feats = extract_features_for_date(
        bars_by_ticker, ticker=ticker, date=date, benchmark=benchmark,
        news_provider=news_provider,
    )
    ...
```

- [ ] **Step 4: Run to confirm pass**
```bash
uv run pytest tests/test_features_with_sentiment.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Make sure Stage-3 + Stage-5 tests still pass**
```bash
uv run pytest tests/test_features.py tests/test_momentum_meta_strategy.py -v 2>&1 | tail -6
uv run pytest -q 2>&1 | tail -3
```
Expected: existing 12 + 4 tests still pass; full suite now 127 (122 + 5).

- [ ] **Step 6: Commit**
```bash
git add models/features.py tests/test_features_with_sentiment.py
git commit -m "feat(stage6): extend models/features.py with sentiment features

Adds SENTIMENT_FEATURE_COLUMNS + optional news_provider seam on
extract_features_for_date / make_features. Default news_provider=None
preserves Stage-5 behavior (sentiment keys absent from output).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Extend `examples/train_meta_models.py` with v1/v2 flag

**Files:**
- Modify: `examples/train_meta_models.py`

Add `--features-version v1|v2` flag. v1 = existing 11 features (unchanged behavior). v2 = 14 features with sentiment, writes to `models/cache_v2/`.

- [ ] **Step 1: Implement the extension**

Open `examples/train_meta_models.py` and:

1. Near the top, add:
```python
from models.features import FEATURE_COLUMNS, SENTIMENT_FEATURE_COLUMNS, make_features

CACHE_DIR_V1 = _PROJECT_ROOT / "models" / "cache"
CACHE_DIR_V2 = _PROJECT_ROOT / "models" / "cache_v2"
SENTIMENT_CACHE = _PROJECT_ROOT / "nlp" / "cache"
SENTIMENT_MODEL_VERSION = "finbert-v1"
```

2. Add to `argparse`:
```python
parser.add_argument(
    "--features-version", choices=("v1", "v2"), default="v1",
    help="v1: 11 price features (Stage 5); v2: 11 price + 3 sentiment features.",
)
```

3. Replace the cache-dir resolution + grid logic so v2 swaps in sentiment:
```python
cache_dir = Path(args.cache_dir) if args.cache_dir != str(CACHE_DIR) else (
    CACHE_DIR_V2 if args.features_version == "v2" else CACHE_DIR_V1
)
cache_dir.mkdir(parents=True, exist_ok=True)

news_provider = None
feature_cols = list(FEATURE_COLUMNS)
if args.features_version == "v2":
    from nlp.sentiment_store import SentimentStore
    news_provider = SentimentStore(
        cache_dir=SENTIMENT_CACHE, model_version=SENTIMENT_MODEL_VERSION,
    )
    feature_cols = list(FEATURE_COLUMNS) + list(SENTIMENT_FEATURE_COLUMNS)
    print(f"v2 mode: using {len(feature_cols)} features (price + sentiment)")
```

4. Replace the `make_features(...)` call to forward `news_provider`:
```python
features = make_features(
    bars_by_ticker, dates_per_ticker, benchmark=BENCHMARK,
    news_provider=news_provider,
)
```

5. Replace `X_train = train[list(FEATURE_COLUMNS)]` with `X_train = train[feature_cols]` and the same `feature_names=` arg to `store.save(...)`.

- [ ] **Step 2: Smoke v1 (unchanged behavior)**
```bash
rm -f models/cache/*.joblib
uv run python examples/train_meta_models.py --smoke 2>&1 | tail -5
```
Expected: same output as Stage-5 smoke (2 models saved under `models/cache/`).

- [ ] **Step 3: Smoke v2 (new path)**
```bash
uv run python examples/train_meta_models.py --features-version v2 --smoke 2>&1 | tail -8
```
Expected: prints "v2 mode: using 14 features (price + sentiment)", "labels generated: N rows", "features generated: M rows × 14 columns" (note: M may be smaller than v1 due to sentiment NaN drops), "done: K models saved under models/cache_v2/".

- [ ] **Step 4: Full pytest sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 127 passed (unchanged).

- [ ] **Step 5: Commit**
```bash
git add examples/train_meta_models.py
git commit -m "feat(stage6): --features-version v1|v2 flag on train_meta_models.py

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Extend `examples/run_harness_meta.py` with v1/v2 flag

**Files:**
- Modify: `examples/run_harness_meta.py`

Add `--features-version v1|v2` flag that switches `model_store_path` between `models/cache/` and `models/cache_v2/`. The meta strategy is unchanged — it reads whichever models the store contains.

- [ ] **Step 1: Implement**

Open `examples/run_harness_meta.py` and:

1. Add to `argparse`:
```python
parser.add_argument(
    "--features-version", choices=("v1", "v2"), default="v1",
    help="v1: models/cache (Stage 5); v2: models/cache_v2 (Stage 6 sentiment).",
)
```

2. Resolve the cache dir:
```python
cache_dir = (
    _PROJECT_ROOT / "models" / "cache_v2"
    if args.features_version == "v2"
    else _PROJECT_ROOT / "models" / "cache"
)
```

3. Replace the `model_store_path=str(CACHE_DIR)` line in `meta_base = dict(...)` with `model_store_path=str(cache_dir)`.

4. Include the features version in the output directory name and the summary heading:
```python
out_dir = RESULTS_ROOT / f"harness_meta_{args.features_version}_{timestamp}"
...
md = f"""# Stage-5 Meta-Labeling Harness ({args.features_version.upper()}) — {timestamp}
```

- [ ] **Step 2: Smoke both versions**
```bash
# Need Task-7 smoke artifacts present in cache_v2 first (re-run if needed):
uv run python examples/train_meta_models.py --features-version v2 --smoke
# Then:
uv run python examples/run_harness_meta.py --features-version v2 --smoke 2>&1 | tail -25
```
Expected: prints v2 summary with concrete numbers, writes to `results/harness_meta_v2_<timestamp>/`.

- [ ] **Step 3: Full pytest sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 127 passed.

- [ ] **Step 4: Commit**
```bash
git add examples/run_harness_meta.py
git commit -m "feat(stage6): --features-version flag on run_harness_meta.py

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: `examples/ablate_sentiment.py` — IC + permutation test

**Files:**
- Create: `tests/test_ablation.py`
- Create: `examples/ablate_sentiment.py`

Compute per-window Spearman IC for v1 and v2 trained models on the held-out test folds, plus a permutation null distribution for ΔIC. Writes a verdict markdown.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ablation.py`:
```python
"""Tests for the ablation utilities — IC + permutation."""
from __future__ import annotations

import numpy as np
import pytest

from examples.ablate_sentiment import (
    spearman_ic,
    permutation_p_value,
)


def test_spearman_ic_perfect_positive():
    preds = np.array([0.1, 0.4, 0.7, 0.9])
    labels = np.array([0, 0, 1, 1])
    ic = spearman_ic(preds, labels)
    assert ic == pytest.approx(1.0, abs=1e-9) or ic > 0.9


def test_spearman_ic_perfect_negative():
    preds = np.array([0.9, 0.7, 0.4, 0.1])
    labels = np.array([0, 0, 1, 1])
    ic = spearman_ic(preds, labels)
    assert ic == pytest.approx(-1.0, abs=1e-9) or ic < -0.9


def test_spearman_ic_no_signal_near_zero():
    rng = np.random.default_rng(42)
    preds = rng.random(200)
    labels = rng.integers(0, 2, size=200)
    ic = spearman_ic(preds, labels)
    assert -0.2 < ic < 0.2


def test_permutation_p_value_observed_above_null():
    rng = np.random.default_rng(0)
    null_dist = rng.normal(0, 0.05, size=1000)
    observed = 0.15
    p = permutation_p_value(observed, null_dist)
    # Observed is far above the null mean → very small p (one-sided).
    assert p < 0.01


def test_permutation_p_value_observed_within_null():
    rng = np.random.default_rng(0)
    null_dist = rng.normal(0, 0.10, size=1000)
    observed = 0.0
    p = permutation_p_value(observed, null_dist)
    assert 0.3 < p < 0.7  # close to 0.5 (observed = null median-ish)
```

- [ ] **Step 2: Run to confirm failure**
```bash
uv run pytest tests/test_ablation.py -v
```
Expected: collection error.

- [ ] **Step 3: Implement**

Create `examples/ablate_sentiment.py`:
```python
"""
Stage-6 ablation: compare price-only (v1) vs price+sentiment (v2) meta-models.

For each held-out walk-forward window:
  1. Score the test fold with both v1 and v2 models.
  2. Compute per-window Spearman IC between meta P(win) and label.
  3. Compute ΔIC = IC_v2 - IC_v1.

Then run a permutation test on the sentiment features within the test set
(re-predict with shuffled sentiment columns) to build a null distribution
of ΔIC under H0 (sentiment is noise).

Verdict per spec §6:
    median ΔIC > 0 AND permutation p < 0.05  → sentiment adds real signal
    otherwise                                → clean rejection

Writes results/sentiment_ablation_<timestamp>/summary.md + summary.json.

Usage:
    uv run python examples/ablate_sentiment.py
    uv run python examples/ablate_sentiment.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data.instruments import VENUE_MIC
from data.tickers import ALL_TICKERS, BENCHMARK, UNIVERSE
from models.features import FEATURE_COLUMNS, SENTIMENT_FEATURE_COLUMNS, make_features
from models.labeling import make_labels
from models.store import ModelStore
from nlp.sentiment_store import SentimentStore
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


RESULTS_ROOT = _PROJECT_ROOT / "results"
CATALOG_PATH = _PROJECT_ROOT / "data" / "catalog"
CACHE_V1 = _PROJECT_ROOT / "models" / "cache"
CACHE_V2 = _PROJECT_ROOT / "models" / "cache_v2"
SENTIMENT_CACHE = _PROJECT_ROOT / "nlp" / "cache"
SENTIMENT_MODEL_VERSION = "finbert-v1"


# ---- pure utilities (tested in tests/test_ablation.py) ---------------------

def spearman_ic(preds: np.ndarray, labels: np.ndarray) -> float:
    """Spearman rank correlation between preds and labels."""
    if len(preds) < 2:
        return float("nan")
    rho, _ = spearmanr(preds, labels)
    return float(rho) if not np.isnan(rho) else 0.0


def permutation_p_value(observed: float, null_dist: np.ndarray) -> float:
    """One-sided p-value: fraction of null samples >= observed."""
    return float(np.mean(np.asarray(null_dist) >= observed))


# ---- orchestration ---------------------------------------------------------

def _bars_to_series(bars) -> pd.Series:
    rows = [
        (pd.Timestamp(b.ts_event, unit="ns", tz="UTC").tz_convert(None).normalize(),
         float(b.close))
        for b in bars
    ]
    s = pd.Series(dict(rows), name="close").sort_index()
    return s[~s.index.duplicated(keep="last")]


def _ic_for_combo(
    dataset: pd.DataFrame,
    store: ModelStore,
    feature_cols: list[str],
    cutoff: pd.Timestamp,
    test_end: pd.Timestamp,
) -> float | None:
    """Score the test fold with the model that was trained at this cutoff."""
    test = dataset[(dataset["date"] >= cutoff) & (dataset["date"] < test_end)]
    if test.empty:
        return None
    loaded = store.load_for(cutoff)
    if loaded is None:
        return None
    model, _ = loaded
    X = test[feature_cols]
    proba = model.predict_proba(X)
    preds = proba[:, 1] if proba.ndim == 2 else proba
    return spearman_ic(preds, test["label"].to_numpy())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="3 cutoffs only, 100 perm draws.")
    parser.add_argument("--n-perm", type=int, default=1000)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS_ROOT / f"sentiment_ablation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_dir}")

    # 1. Load catalog + build features + labels (v2 features so we have both sets).
    catalog = ParquetDataCatalog(path=str(CATALOG_PATH.resolve()))
    bars_by_ticker = {
        t: _bars_to_series(catalog.bars(instrument_ids=[f"{t}.{VENUE_MIC}"]))
        for t in ALL_TICKERS
    }
    spy = bars_by_ticker[BENCHMARK]
    df = spy.to_frame(name="close")
    df["month"] = df.index.to_period("M")
    grouped = df.groupby("month", as_index=False).head(1)
    rebalance_dates = pd.DatetimeIndex(grouped.index)
    labels = make_labels(
        {t: bars_by_ticker[t] for t in UNIVERSE},
        rebalance_dates, horizon_days=21,
    )

    sentiment_store = SentimentStore(
        cache_dir=SENTIMENT_CACHE, model_version=SENTIMENT_MODEL_VERSION,
    )
    dates_per_ticker = {
        t: list(rebalance_dates[rebalance_dates.isin(bars_by_ticker[t].index)])
        for t in UNIVERSE
    }
    features = make_features(
        bars_by_ticker, dates_per_ticker, benchmark=BENCHMARK,
        news_provider=sentiment_store,
    )
    dataset = features.merge(labels, on=["ticker", "date"], how="inner")
    v2_cols = list(FEATURE_COLUMNS) + list(SENTIMENT_FEATURE_COLUMNS)
    dataset = dataset.dropna(subset=v2_cols + ["label"]).reset_index(drop=True)
    print(f"dataset: {len(dataset)} rows × ({len(v2_cols)} feats + label)")

    # 2. Choose cutoffs (every monthly rebalance after warmup_months=12).
    cutoffs = rebalance_dates[12:]
    if args.smoke:
        cutoffs = cutoffs[:3]
    print(f"cutoffs: {len(cutoffs)}")

    # 3. For each cutoff, compute v1 and v2 IC on the test fold.
    # Pick a single combo per side as the "representative" model (the one
    # whose label name is alphabetically first — matches train_meta_models'
    # iteration order; the harness winner is a separate concern).
    combo_label = "n100_d3_lr05"  # first combo in DEFAULT_GRID order
    store_v1 = ModelStore(cache_dir=CACHE_V1, label=combo_label)
    store_v2 = ModelStore(cache_dir=CACHE_V2, label=combo_label)

    per_fold = []
    for i, cutoff in enumerate(cutoffs):
        # Test fold = next 21 days after the cutoff.
        test_end = cutoff + pd.Timedelta(days=21)
        ic_v1 = _ic_for_combo(dataset, store_v1, list(FEATURE_COLUMNS), cutoff, test_end)
        ic_v2 = _ic_for_combo(dataset, store_v2, v2_cols, cutoff, test_end)
        if ic_v1 is None or ic_v2 is None:
            continue
        per_fold.append({"cutoff": cutoff, "ic_v1": ic_v1, "ic_v2": ic_v2,
                         "delta_ic": ic_v2 - ic_v1})
    fold_df = pd.DataFrame(per_fold)
    if fold_df.empty:
        print("ERROR: no folds produced usable IC values.")
        return 1
    median_delta = float(fold_df["delta_ic"].median())
    print(f"\nper-fold ΔIC (median = {median_delta:+.4f}):")
    print(fold_df.to_string(index=False))

    # 4. Permutation null: for each fold, re-predict v2 with shuffled sentiment
    # columns 1000 times; compute the median ΔIC across folds for each draw.
    rng = np.random.default_rng(42)
    n_perm = 100 if args.smoke else args.n_perm
    null_medians = np.zeros(n_perm)
    for k in range(n_perm):
        fold_deltas = []
        for cutoff in fold_df["cutoff"]:
            test_end = cutoff + pd.Timedelta(days=21)
            test = dataset[(dataset["date"] >= cutoff) & (dataset["date"] < test_end)]
            if test.empty:
                continue
            loaded_v2 = store_v2.load_for(cutoff)
            loaded_v1 = store_v1.load_for(cutoff)
            if loaded_v2 is None or loaded_v1 is None:
                continue
            model_v2, _ = loaded_v2
            model_v1, _ = loaded_v1
            X = test[v2_cols].copy()
            for col in SENTIMENT_FEATURE_COLUMNS:
                X[col] = rng.permutation(X[col].to_numpy())
            proba_v2 = model_v2.predict_proba(X)
            preds_v2 = proba_v2[:, 1]
            X_v1 = test[list(FEATURE_COLUMNS)]
            proba_v1 = model_v1.predict_proba(X_v1)
            preds_v1 = proba_v1[:, 1]
            labels = test["label"].to_numpy()
            fold_deltas.append(spearman_ic(preds_v2, labels) - spearman_ic(preds_v1, labels))
        if fold_deltas:
            null_medians[k] = float(np.median(fold_deltas))
    p_value = permutation_p_value(median_delta, null_medians)
    print(f"\npermutation null (n={n_perm}): median={null_medians.mean():.4f}, p_value={p_value:.4f}")

    # 5. Verdict + report.
    if median_delta > 0 and p_value < 0.05:
        verdict = "REAL — sentiment adds incremental signal"
    elif median_delta < 0 and p_value > 0.95:
        verdict = "HURTS — sentiment degrades meta-model"
    else:
        verdict = "REJECT — sentiment is noise within permutation tolerance"

    md = f"""# Stage-6 Sentiment Ablation — {timestamp}

## Methodology
- Combo under test: `{combo_label}`
- Folds (cutoffs): {len(fold_df)}
- Permutation draws: {n_perm}

## Per-fold IC
{fold_df.to_markdown(index=False)}

## Summary
- Median ΔIC: **{median_delta:+.4f}**
- Permutation p-value (one-sided, ΔIC ≥ observed under H0): **{p_value:.4f}**

## Verdict: **{verdict}**
"""
    (out_dir / "summary.md").write_text(md)
    (out_dir / "summary.json").write_text(json.dumps({
        "median_delta_ic": median_delta,
        "p_value": p_value,
        "n_folds": int(len(fold_df)),
        "n_perm": int(n_perm),
        "combo_label": combo_label,
        "verdict": verdict,
    }, indent=2, default=str))
    fold_df.to_csv(out_dir / "per_fold.csv", index=False)
    print(f"\n{md}")
    print(f"artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run unit tests**
```bash
uv run pytest tests/test_ablation.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Full pytest sanity**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 132 passed (127 + 5).

- [ ] **Step 6: Commit**
```bash
git add examples/ablate_sentiment.py tests/test_ablation.py
git commit -m "feat(stage6): sentiment ablation harness (IC + permutation test + verdict)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Full acceptance — ingest news, score all, train v2, run ablation

**Files:** none new — runs the four runnables end-to-end.

- [ ] **Step 1: Ingest news (full window)**

```bash
cd /home/tahseen-dar/Projects/trading-ml
time uv run python data/ingest_alpaca_news.py 2>&1 | tail -10
```
Expected: pulls ~20k-50k headlines over the 2020-07 → today window (free tier; per-ticker volume varies). 5-15 min depending on rate-limit hits. Verify:
```bash
find data/news -name "*.parquet" | wc -l
```
Expected: 100+ shards (12 tickers × ~70 months, less for tickers with sparse coverage).

- [ ] **Step 2: Score all ingested news with FinBERT**

```bash
time uv run python examples/score_news.py 2>&1 | tail -10
```
Expected: prints "scoring N unique articles..." then "done: wrote M score rows under nlp/cache/finbert-v1/". 2-10 min on CPU. Verify:
```bash
ls -la nlp/cache/finbert-v1/ | head -15
```
Expected: per-ticker `.parquet` files present.

- [ ] **Step 3: Train v2 model grid (full)**

```bash
time uv run python examples/train_meta_models.py --features-version v2 2>&1 | tail -10
```
Expected: similar to Stage 5's full training (8 combos × ~60 cutoffs = ~480 models), but row counts may be lower due to sentiment NaN drops. Runtime 30 sec - 2 min.

- [ ] **Step 4: Run harness eval on v2 for comparison**

```bash
time uv run python examples/run_harness_meta.py --features-version v2 2>&1 | tail -30
```
Capture the v2 baseline + meta winner Sharpe numbers from the summary.

- [ ] **Step 5: Run the ablation (this is the verdict)**

```bash
time uv run python examples/ablate_sentiment.py 2>&1 | tee /tmp/stage6_ablation.log | tail -30
```
Capture from the printed summary:
- per-fold IC table
- median ΔIC
- permutation p-value
- verdict (REAL / REJECT / HURTS)

- [ ] **Step 6: Capture headline numbers for Task 11's README**

Note from the latest `results/sentiment_ablation_*/summary.md`:
- Median ΔIC
- Permutation p-value
- Verdict
- v1 vs v2 baseline-vs-meta Sharpes (cross-reference from the v1 and v2 `summary.md`s)

(No commit — `results/` is gitignored.)

---

## Task 11: README + Stage 6 wrap-up

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Status section**

Change Stage 6 from `[ ]` to `[x]` (or `[~]` if the ablation rejected sentiment — still done) and add sub-bullets citing actual numbers from Task 10:
```markdown
- [x] Stage 6 — NLP sentiment overlay
  - [x] News ingest: `uv run python data/ingest_alpaca_news.py` (Alpaca free-tier; headlines + per-ticker tags; `data/news/<ticker>/<YYYY-MM>.parquet`). <INSERT total article count> articles across 12 tickers.
  - [x] FinBERT scoring: `uv run python examples/score_news.py`. Stored at `nlp/cache/finbert-v1/<ticker>.parquet`.
  - [x] Feature integration: 3 sentiment features (`sentiment_7d_mean`, `sentiment_7d_count`, `sentiment_30d_mean`) added to `models/features.py` behind an optional `news_provider` seam; v1 (price-only) and v2 (price+sentiment) model grids trained side-by-side.
  - [x] Ablation: `uv run python examples/ablate_sentiment.py`. Median ΔIC = `<INSERT>`, permutation p = `<INSERT>`. **Verdict: `<INSERT REAL/REJECT/HURTS>`**. <One sentence interpreting the verdict honestly.>
```

REPLACE every `<INSERT>` with the actual numbers from Task 10. Do NOT leave any literal `<INSERT>` in the committed file.

- [ ] **Step 2: Add a new section to the Backtest area**

Add to the Backtest section in README.md:
```markdown
### Sentiment ablation (Stage 6)

```bash
# 1. Ingest news (headlines only; ~5-15 min for full window on free tier)
uv run python data/ingest_alpaca_news.py
uv run python data/ingest_alpaca_news.py --since 2024-01-01  # incremental

# 2. Score with FinBERT (~5 min on CPU)
uv run python examples/score_news.py
uv run python examples/score_news.py --smoke    # ~100 headlines

# 3. Train v2 model grid (price+sentiment features)
uv run python examples/train_meta_models.py --features-version v2

# 4. Harness eval on v2
uv run python examples/run_harness_meta.py --features-version v2

# 5. Ablation verdict — does sentiment add signal?
uv run python examples/ablate_sentiment.py
```
Writes `summary.md` + `per_fold.csv` to `results/sentiment_ablation_<timestamp>/`.
```

- [ ] **Step 3: Update the Layout section**
```markdown
nlp/          Stage-6 NLP — FinBERT sentiment scorer + persistent score cache (cache/ gitignored)
data/news/    Stage-6 — ingested news shards (gitignored)
```

- [ ] **Step 4: Final pytest gate**
```bash
uv run pytest -q 2>&1 | tail -3
```
Expected: 132 passed. Zero failures.

- [ ] **Step 5: Commit**
```bash
git add README.md
git commit -m "docs: README — Stage 6 complete with sentiment ablation verdict

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Stage 6 done**

The parent spec milestone (§5: "Measurable incremental IC or a clean rejection — both are real results") has been evaluated. Whether sentiment clearly wins, loses, or ties is honest research — the milestone is the produced verdict, not a particular outcome.

---

## Self-review

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §2 file structure (nlp + data ingest + features extension + 2 runnables) | Tasks 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| §3 stack additions (transformers, torch) | Task 1 |
| §4 data scope (Alpaca free-tier news, 12 tickers, catalog window) | Task 2 |
| §5 feature definitions (sentiment_7d_mean / count / 30d_mean) | Task 6 |
| §6 ablation methodology (per-window IC + permutation test) | Task 9 |
| §7 risks acknowledged (sparse news, headlines-only, FinBERT cost) | Tasks 2, 5, 6 each cover relevant mitigations |
| §8 out of scope respected | None implemented (constituent aggregation not built) |
| §9 acceptance — runnables + ablation summary | Task 10 + Task 11 |

No spec section uncovered.

**2. Placeholder scan**

- Task 11 contains `<INSERT>` markers explicitly to be substituted with Task 10's numbers; Step 1's text says "REPLACE every `<INSERT>` with the actual numbers from Task 10. Do NOT leave any literal `<INSERT>` in the committed file." This is a substitution instruction, not a plan failure.
- No `TBD`, no `implement later`, no `similar to Task N`, no "add appropriate error handling" generalities.

**3. Type consistency**

- `article_to_row(article) -> dict` (Task 2) with keys `id, headline, symbols, created_at, source` used identically in `_load_all_news` (Task 5) and `SentimentStore.save_scores` (Task 4).
- `score_headlines(headlines: list[str]) -> np.ndarray` (Task 3) called identically in Task 5.
- `SentimentStore(cache_dir, model_version)` + `save_scores(rows)` + `get_scored(ticker, *, start, end)` (Task 4) used identically in Tasks 5, 6, 9.
- `FEATURE_COLUMNS` (existing) + `SENTIMENT_FEATURE_COLUMNS` (Task 6) referenced identically in Tasks 7, 8, 9.
- `news_provider` kwarg on `extract_features_for_date` / `make_features` (Task 6) used identically in Tasks 7 and 9.
- Ablation's `spearman_ic(preds, labels) -> float` + `permutation_p_value(observed, null_dist) -> float` (Task 9) tested in `tests/test_ablation.py` (Task 9 Step 1).

All signatures consistent.
