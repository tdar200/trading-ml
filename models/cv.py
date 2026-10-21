"""
Purged + embargoed walk-forward cross-validation.

Per López de Prado, *Advances in Financial Machine Learning* (2018):

- **Walk-forward**: the data are split into `n_splits` chronologically
  consecutive test folds; the training set for fold k is everything earlier
  than fold k's start.
- **Purge**: any training label whose forward horizon (measured in observed daily sessions)
  would overlap the test fold's date range is excluded. This prevents the
  model from "knowing" overlapping outcomes during training.
- **Embargo**: an additional buffer of `embargo_days` observed sessions is left between the
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
        if any(type(v) is not int or v<0 for v in (self.label_horizon_days,self.embargo_days)):
            raise ValueError("horizon and embargo must be nonnegative session counts")
        dates=pd.DatetimeIndex(dates)
        if dates.hasnans or not dates.is_monotonic_increasing:
            raise ValueError("dates must be valid and chronological")
        # These legacy *_days names mean observed daily sessions, not rows or
        # calendar days. Keep all assets of a session in the same fold.
        sessions=dates.normalize()
        unique=sessions.unique(); n=len(unique)
        if n < self.n_splits:
            raise ValueError(f"need at least n_splits ({self.n_splits}) sessions, got {n}")
        codes=unique.get_indexer(sessions)
        fold_size=n//self.n_splits
        for k in range(self.n_splits):
            start=k*fold_size
            end=(k+1)*fold_size if k<self.n_splits-1 else n
            cutoff=start-self.embargo_days-self.label_horizon_days
            yield np.flatnonzero(codes<cutoff), np.flatnonzero((codes>=start)&(codes<end))
