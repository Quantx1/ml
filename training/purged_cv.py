"""Date-aware purged walk-forward CV for cross-sectional panels.

Audit fix (docs/ML_DL_DEEP_AUDIT_2026_06_15.md): ml.training.wfcv measures
embargo in ROWS, which on a date-pooled panel (N symbols per date) purges a
fraction of one day. This module groups by DATE and sizes train/test/embargo
windows in TRADING DAYS, so a single date's symbols never straddle the
boundary and the embargo actually covers the label horizon.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PurgedCVConfig:
    n_folds: int = 5
    test_days: int = 21        # ~1 month test window
    embargo_days: int = 20     # >= label horizon (audit: day-sized, not rows)
    train_days: int = 252 * 2  # expanding floor for fold 0

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if self.test_days < 1 or self.embargo_days < 0:
            raise ValueError("test_days >= 1 and embargo_days >= 0 required")


def purged_walk_forward_by_date(
    dates: pd.Series,
    cfg: PurgedCVConfig,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) positional arrays per fold.

    `dates` is the per-row date column of the panel (length = n_rows, with
    repeats for multiple symbols per date). Folds expand: train grows, test
    slides forward by `test_days` of unique trading days.
    """
    dser = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    vals = dser.values

    # Map each unique trading day -> its row positions in O(n_rows log n_rows)
    # via argsort + split (a per-day np.where scan is O(n_days * n_rows) and
    # becomes a hot-path bottleneck inside Optuna loops on large panels).
    sort_idx = np.argsort(vals, kind="stable")
    uniq, first_occ = np.unique(vals[sort_idx], return_index=True)  # uniq sorted asc
    splits = np.split(sort_idx, first_occ[1:])
    pos_by_day = {uniq[i]: splits[i] for i in range(len(uniq))}
    n_days = len(uniq)

    need = cfg.train_days + cfg.embargo_days + cfg.n_folds * cfg.test_days
    if need > n_days:
        raise ValueError(
            f"purged CV needs {need} trading days "
            f"(train {cfg.train_days} + embargo {cfg.embargo_days} + "
            f"{cfg.n_folds}x test {cfg.test_days}); panel has {n_days}"
        )

    def rows_for(day_slice: np.ndarray) -> np.ndarray:
        if len(day_slice) == 0:
            return np.array([], dtype=int)
        return np.concatenate([pos_by_day[d] for d in day_slice])

    for i in range(cfg.n_folds):
        test_start = cfg.train_days + cfg.embargo_days + i * cfg.test_days
        test_end = min(test_start + cfg.test_days, n_days)
        if test_end - test_start < 1:
            break
        if test_end < test_start + cfg.test_days:
            logger.warning(
                "purged CV fold %d truncated: test window %d < requested %d days",
                i, test_end - test_start, cfg.test_days,
            )
        train_end = test_start - cfg.embargo_days          # purge embargo days
        train_day_slice = uniq[0:train_end]
        test_day_slice = uniq[test_start:test_end]
        yield rows_for(train_day_slice), rows_for(test_day_slice)


__all__ = ["PurgedCVConfig", "purged_walk_forward_by_date"]
