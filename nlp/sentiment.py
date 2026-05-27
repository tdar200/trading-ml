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
