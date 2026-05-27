"""Tests for models.trainer — LightGBM fit/predict pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.trainer import fit, predict_proba


def _synthetic_xy(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })
    # y depends on f1 with noise — model should learn this.
    y = (X["f1"] + 0.3 * rng.normal(size=n) > 0).astype(int).to_numpy()
    return X, y


def test_fit_returns_model_with_predict_proba():
    X, y = _synthetic_xy(200, seed=1)
    model = fit(X, y, n_estimators=20, max_depth=3, learning_rate=0.1)
    proba = predict_proba(model, X)
    assert proba.shape == (200,)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_fit_is_deterministic_with_fixed_seed():
    X, y = _synthetic_xy(200, seed=2)
    m1 = fit(X, y, n_estimators=20, max_depth=3, learning_rate=0.1, random_state=42)
    m2 = fit(X, y, n_estimators=20, max_depth=3, learning_rate=0.1, random_state=42)
    p1 = predict_proba(m1, X)
    p2 = predict_proba(m2, X)
    assert np.allclose(p1, p2)


def test_fit_learns_signal_on_easy_problem():
    """On data where y is largely a linear function of f1, in-sample accuracy
    should be well above 50%."""
    X, y = _synthetic_xy(400, seed=3)
    model = fit(X, y, n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    pred = (predict_proba(model, X) > 0.5).astype(int)
    accuracy = (pred == y).mean()
    assert accuracy > 0.7
