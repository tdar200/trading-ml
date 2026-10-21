import numpy as np
import pytest

from analysis.purged_validation import purged_walk_forward_splits


def test_purged_splits_are_chronological_disjoint_and_embargoed():
    folds = purged_walk_forward_splits(
        320, min_train_sessions=120, validation_sessions=50,
        step_sessions=50, purge_sessions=5,
    )
    assert len(folds) == 3
    for fold in folds:
        assert fold.train[-1] < fold.purged[0] < fold.validation[0]
        assert len(fold.purged) == 5
        assert not np.intersect1d(fold.train, fold.validation).size


def test_purged_splits_require_enough_history():
    with pytest.raises(ValueError, match="not enough"):
        purged_walk_forward_splits(100)
