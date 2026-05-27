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
