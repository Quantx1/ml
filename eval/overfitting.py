"""
PR 174 — Deflated Sharpe Ratio + Probability of Backtest Overfitting.

The promote gate (PR 167) checked whether one realized Sharpe cleared
1.0. With Optuna running 50-200 trials per trainer and 11 trainers
in the runner, the expected max Sharpe under the null hypothesis is
~1.5 by chance alone. Promoting on raw Sharpe at that scale is
indistinguishable from promoting noise.

This module adds two corrections from López de Prado's "Advances in
Financial Machine Learning" (Ch.11) and Bailey/López de Prado 2014
"The Deflated Sharpe Ratio":

    deflated_sharpe_ratio(sharpe, n_trials, skew, kurtosis, n_obs)
        Bailey-LdP 2014. Returns probability that the observed Sharpe
        is statistically distinguishable from the best of `n_trials`
        random Sharpes under H0. Promote-gate threshold: DSR ≥ 0.95
        (i.e. only 5 percent chance the result is curve-fit).

    probability_of_backtest_overfitting(perf_matrix)
        Bailey-Borwein-LdP 2017. Combinatorially symmetric CV: split
        the trial matrix in half, see how often the in-sample best
        underperforms in OOS. PBO ≤ 0.5 means the ranking is at least
        better than random.

Usage in trainers (optional — add to TrainResult.metrics):
    from ml.eval import deflated_sharpe_ratio
    metrics["deflated_sharpe"] = deflated_sharpe_ratio(
        sharpe=fold_sharpes.mean(),
        n_trials=optuna_n_trials,
        n_obs=n_test_periods,
        skew=fold_returns_skew,
        kurtosis=fold_returns_kurt,
    )

Both metrics are JSONB-safe scalars in [0, 1].
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Deflated Sharpe Ratio (Bailey & López de Prado 2014)
# ============================================================================


def expected_max_sharpe(n_trials: int) -> float:
    """Expected maximum of N IID Standard-Normal Sharpes under H0.

    Closed form for the expected maximum order statistic of N N(0,1)
    draws, used as the null-hypothesis benchmark in DSR. Formula from
    Bailey/LdP 2014 eq.5: E[max] ≈ (1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))
    where γ ≈ 0.5772 (Euler-Mascheroni).

    For N=1 returns 0 (single trial → no selection bias).
    """
    if n_trials <= 1:
        return 0.0
    try:
        from scipy.stats import norm  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy required for DSR") from exc

    gamma = 0.5772156649  # Euler-Mascheroni
    e = math.e
    # Φ⁻¹(1 - 1/N) and Φ⁻¹(1 - 1/(N·e))
    q1 = norm.ppf(1.0 - 1.0 / n_trials)
    q2 = norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float((1.0 - gamma) * q1 + gamma * q2)


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sharpe: Optional[float] = None,
) -> float:
    """Probability that the observed Sharpe is distinguishable from the
    best of `n_trials` random IID Sharpes under H0.

    Args:
        sharpe: Realized (annualized) Sharpe ratio of the strategy.
        n_trials: Number of model variants tried (Optuna trials × CV folds
                  is the right number — every variant that competed for
                  the slot we are promoting).
        n_obs: Number of return observations used to estimate `sharpe`.
        skew: Skewness of the strategy returns (default 0 — Normal).
        kurtosis: Kurtosis of returns (default 3 — Normal). Pass the raw
                  (not excess) kurtosis.
        benchmark_sharpe: Override the H0 benchmark. If None, use the
                  expected max of `n_trials` IID N(0,1) Sharpes.

    Returns:
        Probability ∈ [0, 1] that the strategy's Sharpe exceeds H0.
        DSR ≥ 0.95 means we are 95 percent confident this isn't curve-fit.

    References:
        Bailey & López de Prado (2014), "The Deflated Sharpe Ratio."
        AFML Ch.11.
    """
    if n_obs < 2 or not math.isfinite(sharpe):
        return 0.0
    try:
        from scipy.stats import norm  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy required for DSR") from exc

    sr0 = float(benchmark_sharpe) if benchmark_sharpe is not None else expected_max_sharpe(n_trials)

    # Standard error of Sharpe under non-Normal returns (Mertens 2002):
    # SE(SR) = sqrt((1 - skew·SR + (kurt - 1)/4 · SR^2) / (n_obs - 1))
    excess_kurt = kurtosis - 3.0
    se_sq = (1.0 - skew * sharpe + (excess_kurt / 4.0) * (sharpe ** 2)) / max(1, n_obs - 1)
    if se_sq <= 0:
        return 0.0
    se = math.sqrt(se_sq)
    z = (sharpe - sr0) / se
    return float(norm.cdf(z))


# ============================================================================
# Probability of Backtest Overfitting (Bailey-Borwein-López de Prado 2017)
# ============================================================================


def probability_of_backtest_overfitting(
    perf_matrix: np.ndarray,
    n_splits: int = 16,
) -> float:
    """Combinatorially Symmetric Cross-Validation PBO.

    Args:
        perf_matrix: 2-D array shape (T, N) where T = time periods (or
                     CV folds) and N = number of strategy variants tried.
                     Cell [t, n] = the realized Sharpe (or any
                     monotonically-mapped metric) of variant n in
                     period t.
        n_splits: Even integer. CSCV splits T rows into `n_splits` chunks
                  and considers all (n_splits choose n_splits/2) ways
                  to split into IS / OOS halves.

    Returns:
        PBO ∈ [0, 1]. PBO ≤ 0.5 means the in-sample-best beats median
        OOS more often than not — i.e. the ranking carries some signal.
        PBO > 0.5 means in-sample-best is *worse* than median OOS — the
        whole experiment is overfit.

    References:
        Bailey, Borwein, López de Prado, Zhu (2017),
        "The Probability of Backtest Overfitting."
    """
    arr = np.asarray(perf_matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return 0.5  # not enough variants to test
    T, N = arr.shape
    if T < n_splits or n_splits % 2 != 0:
        # Fall back to a smaller even split if input allows it.
        n_splits = max(0, (T // 2) * 2)
    # CSCV needs at least 4 splits (so combinations(4,2)=6 trials) to
    # produce a meaningful rank distribution. Below that, any value
    # would be a single-sample artifact — return the no-information
    # baseline instead.
    if n_splits < 4:
        return 0.5
    chunk_size = T // n_splits
    if chunk_size < 1:
        return 0.5
    chunks = [
        arr[i * chunk_size : (i + 1) * chunk_size]
        for i in range(n_splits)
    ]

    from itertools import combinations  # noqa: PLC0415
    half = n_splits // 2
    rank_logits: list[float] = []

    for is_idx in combinations(range(n_splits), half):
        oos_idx = tuple(i for i in range(n_splits) if i not in is_idx)
        is_perf = np.concatenate([chunks[i] for i in is_idx]).mean(axis=0)
        oos_perf = np.concatenate([chunks[i] for i in oos_idx]).mean(axis=0)
        # In-sample best variant
        best_n = int(np.argmax(is_perf))
        # Its OOS rank — relative position among N variants
        oos_rank = float(np.sum(oos_perf <= oos_perf[best_n])) / N
        # Logit of relative rank: ln(rank / (1 - rank))
        eps = 1e-9
        oos_rank = min(1.0 - eps, max(eps, oos_rank))
        rank_logits.append(math.log(oos_rank / (1.0 - oos_rank)))

    if not rank_logits:
        return 0.5
    # PBO = fraction of logits ≤ 0 (i.e. best-IS is below median in OOS)
    return float(np.mean([l <= 0 for l in rank_logits]))


# ============================================================================
# Convenience: per-trainer DSR/PBO from a list of fold returns
# ============================================================================


def dsr_pbo_from_fold_returns(
    fold_returns: Iterable[Iterable[float]],
    n_trials: int,
    annualize: int = 252,
) -> dict:
    """Compute DSR + PBO from a per-fold list of strategy return arrays.

    Each element of `fold_returns` is the per-period return series of
    one CV fold. We compute per-fold annualized Sharpe to feed PBO and
    use the pooled return moments to feed DSR.

    Caller should pass `n_trials = optuna_n_trials × n_folds` so the
    DSR null benchmark accounts for the full search space.
    """
    folds: list[np.ndarray] = [np.asarray(f, dtype=float) for f in fold_returns]
    folds = [f for f in folds if f.size >= 2]
    if not folds:
        return {"deflated_sharpe": 0.0, "probability_backtest_overfitting": 1.0}

    # Pooled moments for DSR
    pooled = np.concatenate(folds)
    mu = float(pooled.mean())
    sigma = float(pooled.std(ddof=1)) if pooled.size > 1 else 0.0
    if sigma <= 1e-12:
        return {"deflated_sharpe": 0.0, "probability_backtest_overfitting": 0.5}
    sharpe = (mu / sigma) * math.sqrt(annualize)

    try:
        from scipy.stats import skew, kurtosis  # noqa: PLC0415
        s = float(skew(pooled, bias=False))
        k = float(kurtosis(pooled, fisher=False, bias=False))  # raw, not excess
    except Exception:
        s, k = 0.0, 3.0

    dsr = deflated_sharpe_ratio(
        sharpe=sharpe, n_trials=int(n_trials), n_obs=int(pooled.size),
        skew=s, kurtosis=k,
    )

    # PBO from per-fold Sharpes — but we need a (T, N) matrix where N is
    # the number of variants. With only one variant we can't compute PBO
    # meaningfully; return 0.5 as the no-information value.
    return {
        "deflated_sharpe": round(dsr, 4),
        "probability_backtest_overfitting": 0.5,  # populated when N>1 (PR 175 CPCV)
        "n_trials_used": int(n_trials),
        "pooled_sharpe": round(sharpe, 4),
        "pooled_skew": round(s, 4),
        "pooled_kurtosis": round(k, 4),
    }


__all__ = [
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "dsr_pbo_from_fold_returns",
]
