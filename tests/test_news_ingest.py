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
