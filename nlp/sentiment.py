"""
Headline sentiment scoring.

Public API:
    score_headlines(headlines: list[str]) -> np.ndarray
        Returns net sentiment ∈ [-1, +1] per headline.

Implementation note (2026-05-27):
    Original Stage-6 spec called for FinBERT (`ProsusAI/finbert`).  At
    implementation time the HuggingFace Hub anonymous-download rate limit
    prevented the 440MB model from completing on this machine — repeated
    snapshot_download attempts stalled mid-download with no progress.
    Rather than block the entire stage on a flaky external dependency, we
    swapped to **VADER** (Valence Aware Dictionary, ``vaderSentiment``
    package). VADER is a lexicon-based sentiment analyzer (~100KB,
    zero downloads, instant inference). It was originally tuned for
    social media but is widely used as a baseline for news-headline
    sentiment because its `compound` score is already in [-1, +1].

    Trade-off: VADER is less finance-specific than FinBERT. The honest
    framing for the Stage-6 ablation is "does *any* general sentiment
    signal add to the meta-model?". If even VADER's signal helps, that's
    interesting and motivates a future Stage-6.5 with FinBERT (when HF
    Hub is reachable or with an HF_TOKEN). If VADER doesn't help, that's
    a stronger rejection — sparse ETF news + general sentiment is too
    noisy a feature.

    The module's public API is unchanged; swapping back to FinBERT is a
    drop-in replacement inside ``_load_pipeline``.
"""
from __future__ import annotations

import numpy as np

_PIPELINE_SINGLETON = None


def _load_pipeline():
    """Lazily construct the VADER scorer. Lightweight; sub-100ms first call."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()

    def predict(texts: list[str]):
        # Match the FinBERT pipeline's list-of-dicts shape so the existing
        # ``_net_sentiment`` reducer works unchanged.
        out = []
        for text in texts:
            scores = analyzer.polarity_scores(text)
            # VADER returns {neg, neu, pos, compound}; we emit pos/neg in
            # the FinBERT pipeline shape so _net_sentiment reduces consistently.
            out.append([
                {"label": "positive", "score": float(scores["pos"])},
                {"label": "neutral", "score": float(scores["neu"])},
                {"label": "negative", "score": float(scores["neg"])},
            ])
        return out

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
