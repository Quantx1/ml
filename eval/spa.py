"""
PR 198 — Hansen Superior Predictive Ability (SPA) test.

When testing 11 trainers against the Nifty buy-and-hold benchmark,
the per-trainer t-test gives ~5 percent false positive rate per test.
Across 11 tests the family-wise false positive rate is
1 - 0.95**11 ≈ 43 percent — meaning we expect ~5 of 11 trainers to
"beat the benchmark" by chance alone.

White's Reality Check (2000) and Hansen's Superior Predictive
Ability test (2005) are the standard multiple-testing corrections
for trading-strategy evaluation. Hansen's SPA improves on White by:

  1. Centering each strategy's mean differential under the null on
     its own t-statistic (not the global mean) → more powerful
  2. Using a bootstrap resampling of the (T, K) loss-differential
     matrix → handles autocorrelation
  3. Reporting consistent / lower / upper p-values so users see the
     range of plausibility

Returns SPA p-value across the K trainers vs benchmark. Decision:

    p_spa < 0.05 → at least one strategy beats the benchmark in a
                    way that survives multiple-testing correction
    p_spa ≥ 0.05 → fail to reject null; the apparent winners may be
                    chance

Used to validate: "we're promoting the BEST of N tested models" rather
than "we're promoting the LUCKIEST of N tested models."

Public surface:

    from ml.eval.spa import hansen_spa_test, family_wise_t_correction

    p = hansen_spa_test(loss_differentials)
    # loss_differentials shape (T, K) where each column is a
    # strategy's per-period excess return over the benchmark.

References:
    Hansen (2005), "A Test for Superior Predictive Ability"
    White (2000), "A Reality Check for Data Snooping"
    Sullivan, Timmermann, White (1999), bootstrap resampling.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def family_wise_t_correction(
    p_values: list[float] | np.ndarray,
    *,
    method: str = "bonferroni",
) -> list[float]:
    """Adjust a list of per-test p-values for family-wise error rate.

    Args:
        p_values: per-trainer p-values vs benchmark.
        method: "bonferroni" (multiply by n_tests) or "holm" (step-down).

    Returns:
        list of adjusted p-values; ordering matches input.
    """
    p = np.asarray(list(p_values), dtype=float)
    n = p.size
    if n == 0:
        return []

    if method == "bonferroni":
        return [float(min(1.0, x * n)) for x in p]

    if method == "holm":
        # Sort ascending; multiply by (n - rank + 1)
        order = np.argsort(p)
        sorted_p = p[order]
        adj = np.empty_like(sorted_p)
        prev = 0.0
        for rank, raw in enumerate(sorted_p):
            scaled = raw * (n - rank)
            cur = max(prev, min(1.0, scaled))
            adj[rank] = cur
            prev = cur
        # Restore original order
        out = np.empty_like(adj)
        out[order] = adj
        return out.tolist()

    raise ValueError(f"unknown method: {method}")


def hansen_spa_test(
    loss_differentials: np.ndarray,
    *,
    n_bootstraps: int = 1000,
    block_length: int = 10,
    seed: int = 42,
) -> dict:
    """Hansen's SPA test on a (T, K) excess-return matrix.

    For each strategy k, d_k_t = strategy_k_return[t] - benchmark[t].
    Null hypothesis: max_k E[d_k] ≤ 0 (no strategy beats benchmark).
    SPA p-value: probability under H0 of observing a max sample mean
    at least as extreme as the realized one.

    Implementation:
      Stationary block bootstrap (Politis-Romano 1994) on the
      differential matrix to handle autocorrelation. For each
      bootstrap sample, recenter each column by its own mean × an
      indicator that mean ≤ 0 (Hansen's centering trick — drops
      strategies whose null mean is "too implausible" so their noise
      doesn't dominate).

    Args:
        loss_differentials: shape (T, K). Cell [t, k] = strategy k's
                            excess return over benchmark on day t.
        n_bootstraps: number of bootstrap iterations. 1000 is enough
                      for stable p-value; 10k for paper-grade tests.
        block_length: average block length in bootstrap. ~10 days for
                      daily data captures weekly autocorrelation.
        seed: RNG seed.

    Returns:
        {
          "spa_pvalue": float in [0, 1],
          "n_strategies": K,
          "n_periods": T,
          "best_strategy_idx": int,   # which column had the max mean
          "best_strategy_mean": float,
        }
    """
    arr = np.asarray(loss_differentials, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return {
            "spa_pvalue": 1.0, "n_strategies": 0, "n_periods": 0,
            "best_strategy_idx": -1, "best_strategy_mean": 0.0,
        }
    T, K = arr.shape
    if T < 30 or K < 1:
        return {
            "spa_pvalue": 1.0, "n_strategies": int(K), "n_periods": int(T),
            "best_strategy_idx": -1, "best_strategy_mean": 0.0,
        }

    # Realized per-strategy means + standard errors
    means = arr.mean(axis=0)
    stds = arr.std(axis=0, ddof=1) + 1e-12
    se = stds / np.sqrt(T)
    realized_t = means / se
    realized_max = float(realized_t.max())
    best_idx = int(realized_t.argmax())

    rng = np.random.default_rng(seed)

    # Stationary block bootstrap: random block starts + geometric
    # block lengths with mean = block_length.
    p_block = 1.0 / max(1, block_length)
    bootstrap_max_t: list[float] = []

    # Hansen centering: subtract the column mean only when it's
    # "rejectable" — when the t-stat is below a threshold sqrt(2 ln ln T)
    threshold = -np.sqrt(2.0 * np.log(max(2, np.log(max(2, T)))))
    centered_means = np.where(realized_t > threshold, means, 0.0)

    for _ in range(n_bootstraps):
        idx = np.empty(T, dtype=int)
        i = 0
        while i < T:
            start = rng.integers(0, T)
            length = max(1, rng.geometric(p_block))
            for j in range(length):
                if i >= T:
                    break
                idx[i] = (start + j) % T
                i += 1
        sample = arr[idx]
        boot_means = sample.mean(axis=0)
        # Recenter so the bootstrap distribution is under H0
        boot_t = (boot_means - centered_means) / se
        bootstrap_max_t.append(float(boot_t.max()))

    boots = np.asarray(bootstrap_max_t)
    p_value = float((boots >= realized_max).mean())
    return {
        "spa_pvalue": round(p_value, 4),
        "n_strategies": int(K),
        "n_periods": int(T),
        "best_strategy_idx": best_idx,
        "best_strategy_mean": round(float(means[best_idx]), 6),
    }


__all__ = ["family_wise_t_correction", "hansen_spa_test"]
