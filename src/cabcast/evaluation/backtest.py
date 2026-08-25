from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cabcast.logging_utils import get_logger, log_event

log = get_logger(__name__)


@dataclass(frozen=True)
class Fold:
    name: str
    train: slice
    valid: slice
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp

    def describe(self) -> dict:
        return {
            "fold": self.name,
            "n_train": int(self.train.stop - self.train.start),
            "n_valid": int(self.valid.stop - self.valid.start),
            "train_end": str(self.train_end),
            "valid_start": str(self.valid_start),
            "valid_end": str(self.valid_end),
        }


def _require_sorted(hours: np.ndarray) -> None:
    if not np.all(hours[:-1] <= hours[1:]):
        raise ValueError("feature frame must be sorted by hour_ts before splitting")


def holdout_bounds(hours: np.ndarray, test_days: int) -> tuple[slice, slice, pd.Timestamp]:
    _require_sorted(hours)
    cutoff = pd.Timestamp(hours[-1]) - pd.Timedelta(days=test_days)
    split = int(np.searchsorted(hours, np.datetime64(cutoff), side="right"))
    log_event(
        log, "holdout split", cutoff=str(cutoff), n_train=split, n_test=len(hours) - split,
        test_days=test_days,
    )
    return slice(0, split), slice(split, len(hours)), cutoff


def rolling_origin_folds(
    hours: np.ndarray,
    n_folds: int,
    validation_days: int,
    embargo_hours: int,
) -> list[Fold]:
    _require_sorted(hours)
    end = pd.Timestamp(hours[-1])
    folds: list[Fold] = []

    for k in range(n_folds):
        valid_end = end - pd.Timedelta(days=validation_days * k)
        valid_start = valid_end - pd.Timedelta(days=validation_days)
        train_end = valid_start - pd.Timedelta(hours=embargo_hours)

        train_stop = int(np.searchsorted(hours, np.datetime64(train_end), side="right"))
        v0 = int(np.searchsorted(hours, np.datetime64(valid_start), side="right"))
        v1 = int(np.searchsorted(hours, np.datetime64(valid_end), side="right"))
        if train_stop == 0 or v1 <= v0:
            continue
        folds.append(
            Fold(
                name=f"fold_{n_folds - k}",
                train=slice(0, train_stop),
                valid=slice(v0, v1),
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )
        )

    folds = list(reversed(folds))
    for f in folds:
        log_event(log, "fold", **f.describe())
    return folds


def assert_temporal_integrity(hours: np.ndarray, folds: list[Fold], embargo_hours: int) -> None:
    for f in folds:
        max_train = hours[f.train][-1]
        min_valid = hours[f.valid][0]
        gap = (min_valid - max_train) / np.timedelta64(1, "h")
        if max_train >= min_valid:
            raise ValueError(f"{f.name}: training data overlaps validation window")
        if gap < embargo_hours:
            raise ValueError(
                f"{f.name}: embargo violated, gap {gap:.0f}h < required {embargo_hours}h"
            )
