"""
LambdaRankIC custom objective + metric for LightGBM (arXiv:2605.00501).

Directly optimizes cross-sectional Spearman rank correlation (Rank IC)
between predictions and realized forward returns. Replaces MSE for
ranker-style models like qlib_alpha158.

Why it matters:
    MSE tells the model "predict the exact return value". Production
    cross-sectional alpha trades top-K vs bottom-K — we only care that
    the model's *ordering* matches the realized ordering. LambdaRankIC
    directly targets that ordering with pairwise lambda updates inside
    each cross-section (date).

Wire-up:
    The custom objective + metric integrate with raw LightGBM (not
    LGBMRegressor / LGBMClassifier). For Qlib's LGBModel, see the
    bypass path in qlib_alpha158_lambdarank.train().

    booster = lightgbm.train(
        params={"objective": "regression", ...},  # placeholder; fobj overrides
        train_set=lgb_dataset,                    # must have group info
        fobj=lambdarank_ic_objective,
        feval=rank_ic_metric,
        valid_sets=[lgb_dataset_val],
    )

Group structure:
    Each "group" = one cross-section = all stocks observed on the same date.
    The dataset's ``set_group([n1, n2, ...])`` lists group sizes in order.
    The trainer must sort rows by date before calling this.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-safe sigmoid. Clips to avoid overflow on extreme score gaps."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _rank_centered(values: np.ndarray) -> np.ndarray:
    """Return per-position ranks centered to mean zero.

    Centered ranks make the pairwise gradient direction-symmetric so the
    objective doesn't drift toward a constant prediction offset.
    """
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks - ranks.mean()


# ---------------------------------------------------------------------------
# Custom objective
# ---------------------------------------------------------------------------


def lambdarank_ic_objective(preds: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
    """LightGBM custom objective targeting cross-sectional Rank IC.

    For each group (date), forms all pairs (i, j) of stocks and applies
    a lambda update proportional to:

        λ_ij = sign(true_rank_i − true_rank_j)
              · |Δ_IC|
              · σ(score_diff) · (1 − σ(score_diff))

    The hessian uses the same σ(1−σ) magnitude. Aggregating across pairs
    yields per-row (grad, hess) that LightGBM uses for boosting.

    Parameters
    ----------
    preds : np.ndarray, shape (n_rows,)
        Current model predictions.
    dataset : lightgbm.Dataset
        Must have group info set via ``set_group`` (group sizes per date).

    Returns
    -------
    (grad, hess) : tuple of np.ndarray, shape (n_rows,)
        Per-row gradient and hessian for LightGBM's boosting step.
    """
    labels = dataset.get_label()
    groups = dataset.get_group()

    if groups is None or len(groups) == 0:
        # No group info — fall back to global rank treating all rows as
        # one cross-section. Useful for unit tests; production trainers
        # should always supply groups.
        groups = np.array([len(preds)], dtype=np.int64)

    grads = np.zeros_like(preds, dtype=np.float64)
    hesses = np.full_like(preds, fill_value=1e-6, dtype=np.float64)

    start = 0
    for g_size in groups:
        g_size = int(g_size)
        end = start + g_size
        if g_size < 2:
            start = end
            continue

        scores = preds[start:end].astype(np.float64)
        true = labels[start:end].astype(np.float64)
        true_ranks = _rank_centered(true)

        # Pairwise differences. shape: (g_size, g_size).
        score_diff = scores[:, None] - scores[None, :]
        rank_diff = true_ranks[:, None] - true_ranks[None, :]

        # PR 2026-05-12 — gradient scaling fix.
        # Previous: |rank_diff| / (g_size * (g_size² - 1) / 3). For g=200
        # this gives ~1e-5 per pair → ~1e-3 summed → too small for LightGBM
        # to learn from. Model stopped at iteration 1 with constant preds.
        #
        # Correct: normalize by g_size only. Each pair contributes
        # |rank_diff| / g_size (which is O(1) since centered ranks span
        # [-g/2, +g/2]). Summed over g_size pairs per row → O(g_size) per
        # row → proper LightGBM gradient magnitude.
        delta_ic = np.abs(rank_diff) / max(float(g_size), 1.0)

        sigma = _sigmoid(score_diff)
        # Negative gradient because LightGBM minimizes and we maximize IC.
        lam = -np.sign(rank_diff) * delta_ic * sigma * (1.0 - sigma)
        h_mag = delta_ic * sigma * (1.0 - sigma)

        # Zero the diagonal (i == j has no gradient contribution).
        np.fill_diagonal(lam, 0.0)
        np.fill_diagonal(h_mag, 0.0)

        grads[start:end] = lam.sum(axis=1)
        hesses[start:end] = np.maximum(h_mag.sum(axis=1), 1e-6)
        start = end

    return grads, hesses


# ---------------------------------------------------------------------------
# Evaluation metric
# ---------------------------------------------------------------------------


def rank_ic_metric(preds: np.ndarray, dataset):
    """LightGBM custom metric: mean per-group Spearman rank correlation.

    Returns the standard 3-tuple ``(eval_name, value, higher_is_better)``.
    Skips groups with fewer than 10 rows because Spearman is unstable on
    small cross-sections.
    """
    labels = dataset.get_label()
    groups = dataset.get_group()
    if groups is None or len(groups) == 0:
        groups = np.array([len(preds)], dtype=np.int64)

    ics: list[float] = []
    start = 0
    for g_size in groups:
        g_size = int(g_size)
        end = start + g_size
        if g_size >= 10:
            ic = _spearman_corr(preds[start:end], labels[start:end])
            if not np.isnan(ic):
                ics.append(ic)
        start = end

    mean_ic = float(np.mean(ics)) if ics else 0.0
    return ("rank_ic", mean_ic, True)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation without scipy — rank then Pearson.

    Hand-rolled so the module has no scipy dependency in the hot path.
    """
    if len(x) < 2:
        return float("nan")
    rx = _rank_centered(x.astype(np.float64))
    ry = _rank_centered(y.astype(np.float64))
    num = float((rx * ry).sum())
    den = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return num / den if den > 0 else float("nan")


__all__ = [
    "lambdarank_ic_objective",
    "rank_ic_metric",
]
