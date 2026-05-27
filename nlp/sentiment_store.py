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
