"""
PR 175 — Combinatorial Purged Cross-Validation (CPCV).

Walk-forward CV gives ONE estimate of OOS Sharpe. With Optuna trying
50-200 hyper-parameter variants, that single estimate has wide variance
and high probability of being curve-fit. CPCV (López de Prado AFML
Ch.7) splits T observations into N groups, then for every (N choose k)
combination uses those k groups as test and the rest as train (with
purge + embargo). The N-1 OOS "paths" you can stitch together from the
fold results give a *distribution* of OOS performance instead of a
point estimate.

Public surface:

    cfg = CPCVConfig(n_groups=10, n_test_groups=2, embargo=5, purge=10)
    for fold_idx, (train_idx, test_idx) in enumerate(combinatorial_purged_split(n, cfg)):
        ...

    # (n_groups choose n_test_groups) folds total. N=10, k=2 → 45 folds.

CPCV is significantly more compute-heavy than WFCV. For trainers where
walk-forward already shows clear regime sensitivity (regime_hmm,
intraday_lstm), keep using WFCV for diagnostics; switch to CPCV for the
promote-gate sample matrix that PBO needs.

References:
    López de Prado (2018), AFML Ch.7.
    skfolio.model_selection.CombinatorialPurgedCV.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CPCVConfig:
    """Combinatorial Purged CV configuration.

    n_groups:
        Number of contiguous chunks to split the time series into.
        López de Prado recommends 6-12. With n_groups=10, n_test_groups=2,
        we get C(10, 2) = 45 folds — plenty for stable PBO/DSR estimates.

    n_test_groups:
        How many of the N chunks are held out as test in each fold.
        AFML default: 2. Higher values reduce the train-size of each
        fold but increase fold-count combinatorially.

    embargo:
        Bars to skip on each side of every test chunk to prevent
        forward-label leakage. Default 5 (1 trading week of daily).

    purge:
        Bars to drop from train when their forward-label window overlaps
        a test chunk. Set equal to the longest forward window your label
        depends on (e.g. 10 for triple-barrier with vbd=10).
    """

    n_groups: int = 10
    n_test_groups: int = 2
    embargo: int = 5
    purge: int = 10


def _group_boundaries(n_samples: int, n_groups: int) -> List[Tuple[int, int]]:
    """Split [0, n_samples) into n_groups contiguous chunks. The last
    chunk absorbs any remainder."""
    chunk = n_samples // n_groups
    if chunk < 1:
        raise ValueError(
            f"n_samples={n_samples} too small for n_groups={n_groups}",
        )
    bounds: List[Tuple[int, int]] = []
    for i in range(n_groups):
        start = i * chunk
        end = (i + 1) * chunk if i < n_groups - 1 else n_samples
        bounds.append((start, end))
    return bounds


def _purge_train(
    train_mask: np.ndarray,
    test_starts: List[int],
    test_ends: List[int],
    embargo: int,
    purge: int,
    n_samples: int,
) -> np.ndarray:
    """Apply purge + embargo to a train mask in place."""
    for ts, te in zip(test_starts, test_ends):
        # Embargo on each side of the test chunk
        emb_start = max(0, ts - embargo)
        emb_end = min(n_samples, te + embargo)
        train_mask[emb_start:emb_end] = False
        # Purge: drop train rows whose forward window crosses test start
        if purge > 0:
            purge_start = max(0, ts - purge)
            train_mask[purge_start:ts] = False
    return train_mask


def combinatorial_purged_split(
    n_samples: int | pd.DataFrame | pd.Series | np.ndarray,
    cfg: CPCVConfig,
) -> Iterator[Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]]:
    """Yield (train_idx, test_idx, test_group_ids) for every CPCV fold.

    test_group_ids is the tuple of group indices held out in this fold —
    used by ``extract_backtest_paths`` to reconstruct N-1 OOS paths.
    """
    if isinstance(n_samples, (pd.DataFrame, pd.Series, np.ndarray)):
        n = len(n_samples)
    else:
        n = int(n_samples)

    if cfg.n_groups < cfg.n_test_groups + 1:
        raise ValueError("n_groups must exceed n_test_groups")
    if cfg.n_test_groups < 1:
        raise ValueError("n_test_groups must be >= 1")

    bounds = _group_boundaries(n, cfg.n_groups)
    for test_groups in combinations(range(cfg.n_groups), cfg.n_test_groups):
        test_idx_parts = [np.arange(*bounds[g]) for g in test_groups]
        test_idx = np.concatenate(test_idx_parts) if test_idx_parts else np.array([], dtype=int)

        train_mask = np.ones(n, dtype=bool)
        # Mark test rows out of train first
        for ts, te in (bounds[g] for g in test_groups):
            train_mask[ts:te] = False
        # Apply purge + embargo around each test chunk
        train_mask = _purge_train(
            train_mask,
            [bounds[g][0] for g in test_groups],
            [bounds[g][1] for g in test_groups],
            cfg.embargo,
            cfg.purge,
            n,
        )
        train_idx = np.where(train_mask)[0]
        yield train_idx, test_idx, test_groups


def n_paths(cfg: CPCVConfig) -> int:
    """Number of unique backtest paths CPCV produces.

    From AFML Ch.12: with N groups and k test groups per fold, the number
    of OOS paths a single observation appears in is C(N-1, k-1), and the
    total number of distinct paths through OOS is φ = k · C(N, k) / N.
    """
    from math import comb  # noqa: PLC0415
    return cfg.n_test_groups * comb(cfg.n_groups, cfg.n_test_groups) // cfg.n_groups


def extract_backtest_paths(
    per_fold_returns: List[Tuple[Tuple[int, ...], np.ndarray]],
    cfg: CPCVConfig,
    n_samples: int,
) -> np.ndarray:
    """Stitch CPCV fold returns into multiple OOS backtest paths.

    Each fold contributes a return series spanning its test groups. By
    sequencing folds so each group is visited the right number of times,
    we can construct ``n_paths(cfg)`` distinct end-to-end OOS paths that
    each cover the full timeline — providing a distribution of strategy
    Sharpes for DSR / PBO.

    Args:
        per_fold_returns: list of (test_groups, returns_array) tuples in
                          fold-iteration order.
        cfg: same CPCVConfig used to generate the folds.
        n_samples: total length of the original series.

    Returns:
        2-D array shape (n_paths, n_samples). Cell [p, t] is the strategy
        return on bar t along path p. Bars not covered by path p are 0.
    """
    bounds = _group_boundaries(n_samples, cfg.n_groups)
    paths_count = n_paths(cfg)
    paths = np.zeros((paths_count, n_samples), dtype=float)

    # For each group g, collect all folds that test it. Each path picks
    # one of those fold-occurrences for group g, distinct from other paths.
    group_to_folds: dict[int, List[int]] = {g: [] for g in range(cfg.n_groups)}
    for fold_idx, (test_groups, _ret) in enumerate(per_fold_returns):
        for g in test_groups:
            group_to_folds[g].append(fold_idx)

    # Round-robin assignment: path p uses the p-th fold occurrence of g
    for g in range(cfg.n_groups):
        fold_list = group_to_folds[g]
        if not fold_list:
            continue
        gs, ge = bounds[g]
        for p in range(paths_count):
            fold_idx = fold_list[p % len(fold_list)]
            test_groups, ret = per_fold_returns[fold_idx]
            # Slice the portion of `ret` that corresponds to group g.
            # Folds with multiple test groups: ret is concatenated in
            # group order, so we need offset within the fold.
            offset = 0
            for tg in test_groups:
                if tg == g:
                    seg = ret[offset : offset + (ge - gs)]
                    paths[p, gs:gs + len(seg)] = seg
                    break
                offset += bounds[tg][1] - bounds[tg][0]
    return paths


__all__ = [
    "CPCVConfig",
    "combinatorial_purged_split",
    "n_paths",
    "extract_backtest_paths",
]
