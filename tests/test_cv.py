"""Tests for models.cv — pure, no Nautilus."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.cv import PurgedWalkForwardSplit


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def test_n_splits_yields_n_tuples():
    dates = _dates(500)
    splitter = PurgedWalkForwardSplit(n_splits=5, label_horizon_days=21, embargo_days=5)
    splits = list(splitter.split(dates))
    assert len(splits) == 5
    for train, test in splits:
        assert isinstance(train, np.ndarray)
        assert isinstance(test, np.ndarray)


def test_train_and_test_are_disjoint():
    dates = _dates(500)
    splitter = PurgedWalkForwardSplit(n_splits=5, label_horizon_days=21, embargo_days=5)
    for train, test in splitter.split(dates):
        assert set(train).isdisjoint(set(test))


def test_train_indices_are_strictly_before_test():
    """Walk-forward: every train index < every test index."""
    dates = _dates(500)
    splitter = PurgedWalkForwardSplit(n_splits=5, label_horizon_days=21, embargo_days=5)
    for train, test in splitter.split(dates):
        if len(train) == 0:
            continue
        assert int(train.max()) < int(test.min())


def test_embargo_gap_respected():
    """The largest train index is at least `embargo_days` before the smallest test index."""
    dates = _dates(500)
    splitter = PurgedWalkForwardSplit(n_splits=5, label_horizon_days=21, embargo_days=5)
    for train, test in splitter.split(dates):
        if len(train) == 0:
            continue
        gap = int(test.min()) - int(train.max())
        assert gap >= 5


def test_purge_horizon_respected():
    """Train labels with forward horizon overlapping test fold are excluded.

    With label_horizon_days=21 and embargo=0, the latest train index must be
    at least 21 days before the test fold's first date.
    """
    dates = _dates(500)
    splitter = PurgedWalkForwardSplit(n_splits=5, label_horizon_days=21, embargo_days=0)
    for train, test in splitter.split(dates):
        if len(train) == 0:
            continue
        gap = int(test.min()) - int(train.max())
        assert gap >= 21
