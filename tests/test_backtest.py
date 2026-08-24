from __future__ import annotations

import pandas as pd
import pytest

from urbanflow.evaluation.backtest import (
    assert_temporal_integrity,
    holdout_bounds,
    rolling_origin_folds,
)


@pytest.fixture
def hours():
    return pd.date_range("2025-01-01", periods=24 * 200, freq="h").to_numpy()


def test_holdout_is_strictly_later_than_train(hours):
    train, test, _ = holdout_bounds(hours, test_days=28)
    assert hours[train][-1] < hours[test][0]
    assert (train.stop - train.start) + (test.stop - test.start) == len(hours)


def test_folds_respect_the_embargo(hours):
    folds = rolling_origin_folds(hours, n_folds=4, validation_days=14, embargo_hours=336)
    assert len(folds) == 4
    assert_temporal_integrity(hours, folds, embargo_hours=336)


def test_embargo_violation_is_detected(hours):
    folds = rolling_origin_folds(hours, n_folds=2, validation_days=14, embargo_hours=24)
    with pytest.raises(ValueError, match="embargo violated"):
        assert_temporal_integrity(hours, folds, embargo_hours=336)


def test_train_windows_grow_monotonically(hours):
    folds = rolling_origin_folds(hours, n_folds=4, validation_days=14, embargo_hours=336)
    sizes = [f.train.stop - f.train.start for f in folds]
    assert sizes == sorted(sizes)


def test_validation_windows_do_not_overlap(hours):
    folds = rolling_origin_folds(hours, n_folds=4, validation_days=14, embargo_hours=336)
    spans = sorted((f.valid.start, f.valid.stop) for f in folds)
    for (_, end), (start, _) in zip(spans[:-1], spans[1:], strict=True):
        assert start >= end


def test_unsorted_input_is_rejected(hours):
    with pytest.raises(ValueError, match="sorted"):
        holdout_bounds(hours[::-1], test_days=28)
