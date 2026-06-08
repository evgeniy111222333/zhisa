"""Walk-forward and purged cross-validation splitters.

These splitters are designed for time series: there is **no random
sampling**, only contiguous windows, and an optional embargo gap
between train and test to avoid leakage from serial correlation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple


@dataclass
class SplitSpec:
    train_size: int
    test_size: int
    step: int = 0          # 0 = non-overlapping windows
    n_splits: Optional[int] = None
    embargo: int = 0       # gap (in bars) dropped between train and test
    min_train_size: Optional[int] = None


@dataclass
class Fold:
    train: Tuple[int, int]  # (start, end) inclusive-exclusive
    test: Tuple[int, int]


def walk_forward_splits(n: int, spec: SplitSpec) -> List[Fold]:
    """Return a list of walk-forward folds.

    For each fold, the training window ends at ``train_end`` and the test
    window starts at ``train_end + embargo``. The training window slides
    forward by ``step`` (or by ``test_size`` if ``step == 0``).
    """
    if n < spec.train_size + spec.test_size:
        raise ValueError("n is too small for the requested train/test sizes")
    step = spec.step if spec.step > 0 else spec.test_size
    min_train = spec.min_train_size or spec.train_size
    folds: List[Fold] = []
    train_end = spec.train_size
    while train_end + spec.embargo + spec.test_size <= n:
        train_start = max(0, train_end - spec.train_size)
        if train_end - train_start < min_train:
            break
        test_start = train_end + spec.embargo
        test_end = test_start + spec.test_size
        folds.append(Fold(
            train=(train_start, train_end),
            test=(test_start, test_end),
        ))
        train_end += step
        if spec.n_splits is not None and len(folds) >= spec.n_splits:
            break
    return folds


def purged_kfold_indices(
    n: int,
    n_splits: int = 5,
    embargo: int = 0,
) -> List[Fold]:
    """Purged k-fold (López de Prado) for time series.

    Each fold is a contiguous block; the embargo gap drops labels near
    the train/test boundary to avoid leakage. Train sets cover the
    complement of the test block.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    fold_size = n // n_splits
    folds: List[Fold] = []
    for k in range(n_splits):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k < n_splits - 1 else n
        train_ranges: List[Tuple[int, int]] = []
        # left
        if test_start - embargo > 0:
            train_ranges.append((0, max(0, test_start - embargo)))
        # right
        if test_end + embargo < n:
            train_ranges.append((min(n, test_end + embargo), n))
        # Combine ranges into a single Fold with tuple-of-tuples in train
        if not train_ranges:
            continue
        folds.append(Fold(train=train_ranges[0] if len(train_ranges) == 1 else tuple(train_ranges),  # type: ignore[arg-type]
                          test=(test_start, test_end)))
    return folds
