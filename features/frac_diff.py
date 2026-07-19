"""
PR 193 — Fractional differentiation (López de Prado, AFML Ch.5).

Most price-derived features in the LightGBM gate are integer-differenced
(returns: pct_change(1)) which kills almost all memory, OR raw price
(non-stationary). Fractional differentiation finds the smallest d in
[0, 1] that yields a stationary series while retaining maximum memory.

The Fixed-Width Window FFD variant (AFML eq. 5.5) drops the first
``thresh``-determined warmup rows so the kernel weights stop at a
fixed cutoff. Faster than expanding-window FFD and avoids the
"infinite tail" weight issue.

Public surface:

    from ml.features.frac_diff import frac_diff_ffd, find_min_d_stationary

    # Apply with a known d:
    out = frac_diff_ffd(prices, d=0.4, thresh=0.01)

    # Find the smallest d that gives stationarity (ADF p-value < 0.05):
    d_star = find_min_d_stationary(prices)
    out = frac_diff_ffd(prices, d=d_star)

Used by:
    feature_engineering layer for log-close, MACD, OBV — the
    not-yet-stationary features that the LGBM gate currently sees as
    raw / non-stationary.

References:
    AFML Ch.5 §5.5-5.6.
    Hudson & Thames `mlfinlab.fracdiff_ffd` (BSD).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ffd_weights(d: float, thresh: float = 1e-4) -> np.ndarray:
    """Compute FFD kernel weights until |w_k| < thresh.

    From AFML §5.5: w_k = -w_{k-1} * (d - k + 1) / k. Returns weights
    in REVERSE order (most-recent-first) for direct convolution.
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
        if k > 5000:   # hard ceiling; thresh too tight if we hit this
            break
    return np.array(w[::-1])   # most-recent first


def frac_diff_ffd(
    series: np.ndarray | pd.Series,
    d: float,
    thresh: float = 1e-4,
) -> pd.Series:
    """Fixed-Width Window fractional differentiation.

    Args:
        series: 1-D price/log-price series.
        d: differentiation order in [0, 1]. d=1 = standard difference,
           d=0 = identity, in between = partial memory.
        thresh: kernel weight cutoff. Smaller → wider window, slower.

    Returns:
        pd.Series same length as input. The first len(weights)-1 rows
        are NaN (insufficient lookback) — caller drops them before
        training.
    """
    if not (0.0 <= d <= 1.0):
        raise ValueError(f"d must be in [0, 1], got {d}")

    s = pd.Series(series).astype(float)
    weights = _ffd_weights(d, thresh)
    width = len(weights)
    out = np.full(len(s), np.nan, dtype=float)
    arr = s.values
    for i in range(width - 1, len(arr)):
        window = arr[i - width + 1: i + 1]
        if np.isnan(window).any():
            continue
        out[i] = float(np.dot(weights, window))
    return pd.Series(out, index=s.index)


def find_min_d_stationary(
    series: np.ndarray | pd.Series,
    *,
    thresh: float = 1e-4,
    pvalue_target: float = 0.05,
    grid: Optional[list[float]] = None,
) -> Optional[float]:
    """Search d ∈ [0, 1] for the smallest value that produces a
    stationary FFD series (ADF p-value < pvalue_target).

    Returns the d value, or None if no value in the grid achieves
    stationarity (rare — log prices typically need d ≈ 0.3-0.5).

    Requires statsmodels for the ADF test. Returns None silently if
    statsmodels isn't installed (caller can pick a default d=0.4).
    """
    try:
        from statsmodels.tsa.stattools import adfuller  # noqa: PLC0415
    except ImportError:
        logger.warning("statsmodels missing — find_min_d_stationary returning None")
        return None

    s = pd.Series(series).dropna().astype(float)
    if grid is None:
        grid = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]

    for d in grid:
        ffd = frac_diff_ffd(s, d=d, thresh=thresh).dropna()
        if len(ffd) < 100:
            continue
        try:
            pval = float(adfuller(ffd, maxlag=1, regression="c", autolag=None)[1])
        except Exception:
            continue
        if pval < pvalue_target:
            return float(d)
    return None


__all__ = ["frac_diff_ffd", "find_min_d_stationary"]
