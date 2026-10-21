"""Chronological walk-forward splits with an explicit leakage purge."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PurgedSplit:
    train: np.ndarray
    validation: np.ndarray
    purged: np.ndarray


def purged_walk_forward_splits(
    n_sessions: int, *, min_train_sessions: int = 120,
    validation_sessions: int = 63, step_sessions: int = 63,
    purge_sessions: int = 5,
) -> list[PurgedSplit]:
    """Build expanding chronological folds; purged rows belong to neither side."""
    values = (n_sessions, min_train_sessions, validation_sessions, step_sessions)
    if any(value <= 0 for value in values) or purge_sessions < 0:
        raise ValueError("session and step sizes must be positive; purge cannot be negative")
    first_validation = min_train_sessions + purge_sessions
    if first_validation + validation_sessions > n_sessions:
        raise ValueError("not enough sessions for one purged walk-forward split")
    folds = []
    for start in range(first_validation, n_sessions - validation_sessions + 1, step_sessions):
        train_stop = start - purge_sessions
        folds.append(PurgedSplit(
            train=np.arange(0, train_stop, dtype=int),
            purged=np.arange(train_stop, start, dtype=int),
            validation=np.arange(start, start + validation_sessions, dtype=int),
        ))
    return folds
