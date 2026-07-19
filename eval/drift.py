"""
PR 195 — feature distribution drift detection (PSI + KS).

Models trained on 2018-2024 data may quietly degrade in 2025+ live
trading because the feature distribution has shifted (regime change,
new market structure, sector rotation). The promote-gate checks
backtest performance, not input drift — so a model can pass the gate
on its training window then fail in production.

This module ships two detectors that admin tooling / a weekly cron
can call to flag drift early:

  Population Stability Index (PSI):
    Buckets the reference + current distributions into N quantiles,
    then sums (current_pct - ref_pct) * ln(current_pct / ref_pct)
    across buckets. Industry-standard threshold:
       PSI < 0.10 → no drift
       0.10 ≤ PSI < 0.25 → moderate drift, investigate
       PSI ≥ 0.25 → significant drift, retrain

  Kolmogorov-Smirnov (KS) test:
    Compares the empirical CDFs of reference vs current. Returns
    (statistic, pvalue). p < 0.05 → distributions differ.

Public surface:

    from ml.eval.drift import (
        psi,
        ks_drift,
        feature_drift_report,
    )

    rep = feature_drift_report(
        reference=train_features_df,
        current=live_features_df,
    )
    # → {feat: {"psi": ..., "ks_stat": ..., "ks_pvalue": ..., "alert": bool}}

References:
  Industry PSI standard from credit-risk model monitoring.
  Tecton drift docs: https://docs.tecton.ai/docs/detect-drift
  evidently AI library (open source).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DriftConfig:
    """Drift-detection thresholds.

    psi_warn: PSI value above which we emit a warning. 0.10 = standard.
    psi_alert: PSI value above which we mark as "drift confirmed". 0.25.
    ks_pvalue_alert: KS p-value below which distributions are flagged
                     as different. Standard 0.05.
    n_buckets: PSI bucket count. 10 = decile, the most common choice.
    epsilon: small value added to bucket counts to avoid log(0).
    """

    psi_warn: float = 0.10
    psi_alert: float = 0.25
    ks_pvalue_alert: float = 0.05
    n_buckets: int = 10
    epsilon: float = 1e-6


def psi(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    *,
    n_buckets: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Population Stability Index between two samples.

    Buckets are quantile-based on the REFERENCE distribution so the
    reference is by construction uniform across buckets. The current
    distribution gets bucketed using the same edges, then PSI is
    computed bucket-by-bucket.
    """
    ref = np.asarray(list(reference), dtype=float)
    cur = np.asarray(list(current), dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return 0.0

    # If reference is constant, PSI is undefined — treat as 0.
    if float(ref.std(ddof=0)) < 1e-12:
        return 0.0

    # Quantile-based bucket edges. Drop duplicate edges (constant tails)
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, n_buckets + 1)))
    if len(edges) < 3:
        return 0.0

    # Pad first and last edge so points outside the reference range still
    # bucket (assigned to first or last bucket).
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = ref_counts / max(1, ref_counts.sum())
    cur_pct = cur_counts / max(1, cur_counts.sum())
    # Avoid log(0) and div-by-zero
    ref_pct = np.where(ref_pct < epsilon, epsilon, ref_pct)
    cur_pct = np.where(cur_pct < epsilon, epsilon, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_drift(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """KS two-sample test. Returns (statistic, pvalue). Returns (0,1)
    if scipy unavailable so callers can skip silently."""
    try:
        from scipy.stats import ks_2samp  # noqa: PLC0415
    except ImportError:
        return 0.0, 1.0

    ref = np.asarray(list(reference), dtype=float)
    cur = np.asarray(list(current), dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size < 5 or cur.size < 5:
        return 0.0, 1.0
    try:
        stat, pval = ks_2samp(ref, cur)
    except Exception:
        return 0.0, 1.0
    return float(stat), float(pval)


def feature_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    cfg: Optional[DriftConfig] = None,
) -> Dict[str, Dict[str, float]]:
    """Per-feature drift report comparing two DataFrames with the same
    columns. Returns mapping feature → {psi, ks_stat, ks_pvalue, alert,
    severity ∈ {"none", "warn", "alert"}}.

    Caller persists into model_versions.metrics["drift_report"] or
    posts to admin dashboard.
    """
    cfg = cfg or DriftConfig()
    common = [c for c in reference.columns if c in current.columns]
    if not common:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for col in common:
        ps = psi(reference[col], current[col],
                  n_buckets=cfg.n_buckets, epsilon=cfg.epsilon)
        ks_stat, ks_p = ks_drift(reference[col], current[col])
        if ps >= cfg.psi_alert:
            severity = "alert"
        elif ps >= cfg.psi_warn or ks_p < cfg.ks_pvalue_alert:
            severity = "warn"
        else:
            severity = "none"
        out[col] = {
            "psi": round(ps, 4),
            "ks_stat": round(ks_stat, 4),
            "ks_pvalue": round(ks_p, 4),
            "alert": severity == "alert",
            "severity": severity,
        }
    return out


def summarize_report(report: Dict[str, Dict[str, float]]) -> dict:
    """One-line aggregate suitable for logging + admin display."""
    if not report:
        return {"n_features": 0, "n_alerts": 0, "n_warnings": 0, "max_psi": 0.0}
    n = len(report)
    n_alerts = sum(1 for v in report.values() if v["severity"] == "alert")
    n_warns = sum(1 for v in report.values() if v["severity"] == "warn")
    max_psi = max(v["psi"] for v in report.values())
    top_drifted = sorted(report.items(), key=lambda kv: -kv[1]["psi"])[:5]
    return {
        "n_features": int(n),
        "n_alerts": int(n_alerts),
        "n_warnings": int(n_warns),
        "max_psi": round(float(max_psi), 4),
        "top_drifted": [
            {"feature": k, "psi": v["psi"], "severity": v["severity"]}
            for k, v in top_drifted
        ],
    }


__all__ = [
    "DriftConfig",
    "feature_drift_report",
    "ks_drift",
    "psi",
    "summarize_report",
]
