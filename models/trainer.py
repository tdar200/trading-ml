"""
LightGBM fit + predict pipeline.

Two functions, both pure:

- ``fit(X, y, *, n_estimators, max_depth, learning_rate, random_state=42) -> LGBMClassifier``
- ``predict_proba(model, X) -> np.ndarray`` — returns P(y=1) per row.

The trainer is unaware of walk-forward / CV — that orchestration lives in
``examples/train_meta_models.py`` and ``models/cv.py``.

No IO, no Nautilus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


def fit(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> LGBMClassifier:
    """Fit a binary LightGBM classifier on (X, y).

    Args:
        X: feature DataFrame (rows = samples, columns = feature names).
        y: 0/1 label array, same length as X.
        n_estimators: number of boosting rounds.
        max_depth: tree depth limit (3 / 5 typical for small datasets).
        learning_rate: shrinkage.
        random_state: passed to LightGBM for reproducibility.

    Returns:
        Fitted LGBMClassifier.
    """
    model = LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        verbose=-1,  # quiet
        n_jobs=1,    # deterministic on small datasets
    )
    model.fit(X, y)
    return model


def predict_proba(model: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return P(y=1) for each row in X."""
    proba = model.predict_proba(X)
    # LGBMClassifier returns shape (n, 2); column 1 is P(y=1).
    return proba[:, 1]
