"""
PR 161 — Walk-forward cross-validation harness.

Replaces single-split chronological holdout with rolling/expanding-window
walk-forward CV across every trainer. This is the only honest way to
evaluate a time-series model: the CV folds match how the model will see
data in production (always train on past, test on a strictly later
window).

Two strategies supported:

    rolling   — fixed-width train + test window, slides forward
                Use when: regime shifts are real; we want to test
                stability across regimes (HMM, swing models)

    expanding — train window grows, test window stays small
                Use when: more data is always better and the latest
                test window is closest to production reality
                (LSTM intraday, RL, momentum)

Public surface:

    from ml.training.wfcv import walk_forward_split, WFCVConfig

    cfg = WFCVConfig(strategy="expanding", n_folds=5, test_size=252)
    for fold_idx, (train_idx, test_idx) in enumerate(walk_forward_split(df, cfg)):
        ...

Returns (train, test) index arrays per fold so the caller can train +
evaluate per fold and aggregate. Fold metrics come from
``ml.eval.backtest_eval`` (PR 162).

Anti-leakage rules built in:
  - test_idx is ALWAYS strictly after train_idx (no temporal overlap)
  - optional embargo gap (default 5 bars) between train end and test
    start, to handle the "label leakage" problem from López de Prado
    (Advances in Financial ML, ch.7) — predictions involving forward
    windows leak through if test starts immediately after train.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Literal, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


WFCVStrategy = Literal["rolling", "expanding"]


@dataclass
class WFCVConfig:
    """Walk-forward CV configuration.

    strategy:
        "rolling"   — fixed-width train, slides forward
        "expanding" — train grows, test stays fixed size

    n_folds:
        Number of folds. 5 is the conventional default for time-series CV.

    test_size:
        Number of bars (rows) in each test fold. For daily data: 252 = 1
        year, 63 = 1 quarter. For 5-min intraday: ~75 = 1 day, ~378 = 1
        week.

    train_size:
        Required for ``rolling``. For ``expanding``, the first fold
        uses ``train_size`` and subsequent folds extend by ``test_size``
        per fold.

    embargo:
        Number of bars to skip between train end and test start. Defaults
        to 5 (1 calendar week of daily). Prevents label-leakage when
        labels are computed over a forward window (e.g. 5-day forward
        return for the last training row depends on data inside the
        embargo, which would also appear in the next fold's training).

    purge:
        Number of bars to drop from the END of train when they overlap
        with the test fold's labeling window. Mirrors López de Prado's
        purging concept. Set equal to the longest forward-looking
        feature window in your label.
    """

    strategy: WFCVStrategy = "expanding"
    n_folds: int = 5
    test_size: int = 252
    train_size: int = 252 * 3   # used for "rolling" only
    embargo: int = 5
    purge: int = 0


def walk_forward_split(
    n_samples: int | pd.DataFrame | pd.Series | np.ndarray,
    cfg: WFCVConfig,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) integer arrays per fold.

    Accepts a length, a pandas frame/series, or a numpy array — anything
    we can call ``len()`` on. Indices are positional (0..n_samples-1).
    """
    if isinstance(n_samples, (pd.DataFrame, pd.Series, np.ndarray)):
        n = len(n_samples)
    else:
        n = int(n_samples)

    if cfg.n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if cfg.test_size < 1:
        raise ValueError("test_size must be >= 1")

    # Total bars consumed by ALL folds' tests.
    test_total = cfg.n_folds * cfg.test_size
    if cfg.strategy == "rolling":
        # Each fold needs train_size + embargo + test_size; folds slide
        # by test_size so the LAST fold's test ends at n.
        if cfg.train_size + cfg.embargo + cfg.test_size > n:
            raise ValueError(
                f"rolling WFCV requires at least "
                f"train_size + embargo + test_size = "
                f"{cfg.train_size + cfg.embargo + cfg.test_size} bars; got {n}",
            )
        # Last test ends at n. Walk back: each fold's test starts at
        # n - test_total + i * test_size.
        for i in range(cfg.n_folds):
            test_start = n - test_total + i * cfg.test_size
            test_end = test_start + cfg.test_size
            train_end = test_start - cfg.embargo
            train_start = train_end - cfg.train_size
            if train_start < 0:
                logger.warning(
                    "wfcv rolling fold %d would start before t=0; skipping",
                    i,
                )
                continue
            train_idx = np.arange(train_start, max(train_start, train_end - cfg.purge))
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx
        return

    # expanding (default)
    if cfg.train_size + cfg.embargo + cfg.test_size > n:
        raise ValueError(
            f"expanding WFCV requires at least "
            f"train_size + embargo + test_size = "
            f"{cfg.train_size + cfg.embargo + cfg.test_size} bars; got {n}",
        )
    # Each fold's test_start = train_size + embargo + i * test_size.
    # Each fold's train spans 0 → test_start - embargo.
    for i in range(cfg.n_folds):
        test_start = cfg.train_size + cfg.embargo + i * cfg.test_size
        test_end = test_start + cfg.test_size
        if test_end > n:
            logger.warning(
                "wfcv expanding fold %d test_end %d > n %d; truncating",
                i, test_end, n,
            )
            test_end = n
            if test_end - test_start < 1:
                break
        train_end = test_start - cfg.embargo
        train_idx = np.arange(0, max(0, train_end - cfg.purge))
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Compute mean + std across fold metric dicts.

    Returns ``{"<key>_mean": ..., "<key>_std": ..., "<key>_per_fold": [...]}``
    for every numeric key found in the input. Non-numeric keys (strings,
    nested dicts) are dropped from the aggregate to keep the
    ``model_versions.metrics`` JSON flat.
    """
    if not fold_metrics:
        return {}
    keys = set()
    for m in fold_metrics:
        for k, v in m.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(k)
    out: dict = {}
    for k in sorted(keys):
        vals = [m.get(k) for m in fold_metrics if isinstance(m.get(k), (int, float))]
        if not vals:
            continue
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
        out[f"{k}_per_fold"] = [round(float(v), 6) for v in vals]
    out["n_folds"] = len(fold_metrics)
    return out


__all__ = [
    "WFCVConfig",
    "WFCVStrategy",
    "walk_forward_split",
    "aggregate_fold_metrics",
]
