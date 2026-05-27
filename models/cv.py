"""
Purged + embargoed walk-forward cross-validation.

Per López de Prado, *Advances in Financial Machine Learning* (2018):

- **Walk-forward**: the data are split into `n_splits` chronologically
  consecutive test folds; the training set for fold k is everything earlier
  than fold k's start.
- **Purge**: any training label whose forward horizon (`date + label_horizon_days`)
  would overlap the test fold's date range is excluded. This prevents the
  model from "knowing" overlapping outcomes during training.
- **Embargo**: an additional buffer of `embargo_days` is left between the
  latest training index and the first test index to handle autocorrelation
  in returns.

Returns numpy integer index arrays into the input `dates`. Pure: no IO,
no Nautilus.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedWalkForwardSplit:
    """Generator-style splitter.

    Usage:
        splitter = PurgedWalkForwardSplit(n_splits=5, label_horizon_days=21, embargo_days=5)
        for train_idx, test_idx in splitter.split(dates):
            ...
    """
    n_splits: int
    label_horizon_days: int = 21
    embargo_days: int = 5

    def split(self, dates: pd.DatetimeIndex) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        n = len(dates)
        if n < self.n_splits:
            raise ValueError(f"need at least n_splits ({self.n_splits}) dates, got {n}")

        # Equal-sized contiguous test folds.
        fold_size = n // self.n_splits
        for k in range(self.n_splits):
            test_start = k * fold_size
            test_end = (k + 1) * fold_size if k < self.n_splits - 1 else n
            test_idx = np.arange(test_start, test_end, dtype=np.int64)

            # Train = everything strictly before test fold, minus purge + embargo.
            train_cutoff = test_start - self.embargo_days - self.label_horizon_days
            if train_cutoff <= 0:
                train_idx = np.array([], dtype=np.int64)
            else:
                train_idx = np.arange(0, train_cutoff, dtype=np.int64)

            yield train_idx, test_idx
