"""Tests for the ablation utilities — IC + permutation."""
from __future__ import annotations

import numpy as np
import pytest

from examples.ablate_sentiment import (
    spearman_ic,
    permutation_p_value,
)


def test_spearman_ic_strong_positive():
    # Binary labels with ties cap Spearman at ~0.894; with unique labels we'd get 1.0.
    preds = np.array([0.1, 0.4, 0.7, 0.9])
    labels = np.array([0, 0, 1, 1])
    ic = spearman_ic(preds, labels)
    assert ic > 0.8


def test_spearman_ic_strong_negative():
    preds = np.array([0.9, 0.7, 0.4, 0.1])
    labels = np.array([0, 0, 1, 1])
    ic = spearman_ic(preds, labels)
    assert ic < -0.8


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
