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
