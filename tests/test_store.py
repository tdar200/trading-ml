"""Tests for models.store — dated joblib artifacts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.store import ModelStore
from models.trainer import fit


def _trained_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=100), "f2": rng.normal(size=100)})
    y = (X["f1"] > 0).astype(int).to_numpy()
    return fit(X, y, n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42)


def test_save_creates_dated_file(tmp_path: Path):
    store = ModelStore(cache_dir=tmp_path, label="default")
    model = _trained_model()
    path = store.save(pd.Timestamp("2024-03-01"), model, feature_names=["f1", "f2"])
    assert path.exists()
    assert "2024-03-01" in path.name
    assert "default" in path.name


def test_load_for_picks_most_recent_strictly_before(tmp_path: Path):
    store = ModelStore(cache_dir=tmp_path, label="default")
    model = _trained_model()
    store.save(pd.Timestamp("2024-01-01"), model, feature_names=["f1", "f2"])
    store.save(pd.Timestamp("2024-04-01"), model, feature_names=["f1", "f2"])
    store.save(pd.Timestamp("2024-07-01"), model, feature_names=["f1", "f2"])
    # Query for 2024-05-15 → should pick 2024-04-01.
    result = store.load_for(pd.Timestamp("2024-05-15"))
    assert result is not None
    _, feature_names = result
    assert feature_names == ["f1", "f2"]
    # Verify it picked the 2024-04-01 file by inspecting `last_loaded_path`.
    assert "2024-04-01" in str(store.last_loaded_path)


def test_load_for_returns_none_before_any_model(tmp_path: Path):
    """Querying before any model has been saved → None."""
    store = ModelStore(cache_dir=tmp_path, label="default")
    model = _trained_model()
    store.save(pd.Timestamp("2024-04-01"), model, feature_names=["f1", "f2"])
    result = store.load_for(pd.Timestamp("2024-03-15"))  # before only saved cutoff
    assert result is None


def test_load_for_label_filter_isolates_labels(tmp_path: Path):
    """Different labels in the same cache_dir don't interfere with load_for."""
    store_a = ModelStore(cache_dir=tmp_path, label="combo_a")
    store_b = ModelStore(cache_dir=tmp_path, label="combo_b")
    model = _trained_model()
    store_a.save(pd.Timestamp("2024-04-01"), model, feature_names=["f1", "f2"])
    # store_b has nothing → its load_for must return None.
    assert store_b.load_for(pd.Timestamp("2024-05-15")) is None
    assert store_a.load_for(pd.Timestamp("2024-05-15")) is not None
