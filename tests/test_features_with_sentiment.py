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
    date = bars["A"].index[150]
    # Synthesize 5 positive articles in the last 7 days before `date`.
    articles = pd.DataFrame({
        "id": list(range(5)),
        "ticker": ["A"] * 5,
        "created_at": pd.to_datetime(
            [pd.Timestamp(date) - pd.Timedelta(days=i) for i in (1, 2, 3, 4, 6)],
            utc=True,
        ),
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
            pd.Timestamp(date),  # exactly at decision time — must be excluded
            pd.Timestamp(date) - pd.Timedelta(days=1),
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
