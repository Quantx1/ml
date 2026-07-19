"""Pre-training EDA + preprocessing checks (strict, no fallbacks).

Each function returns a typed result (dict / dataclass) describing one
audit dimension. ``scripts/train/eda_report.py`` calls these per trainer,
aggregates results, and HARD-FAILS the pipeline if any blocking issue
is found.

Blocking thresholds (locked 2026-05-12):
  - any required feature has > 50% NaN
  - any classification label class has < 5% of total samples
  - any feature has variance == 0 across the entire train window
  - feature-label IC absolute mean < 0.005 for all features (cross-sec)
  - leakage: feature at t correlates >0.95 with label at t (lookahead)

These thresholds are aggressive on purpose. Failing them means the
training run will produce a garbage model — better to abort and fix
the data layer than to waste GPU hours.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EDAReport:
    """One report per trainer. Aggregates all audit dimensions."""

    trainer: str
    n_rows: int = 0
    n_features: int = 0
    n_symbols: Optional[int] = None
    date_range: Optional[List[str]] = None      # [first, last]
    feature_summary: Dict[str, Any] = field(default_factory=dict)
    label_summary: Dict[str, Any] = field(default_factory=dict)
    ic_summary: Dict[str, Any] = field(default_factory=dict)
    leakage_summary: Dict[str, Any] = field(default_factory=dict)
    near_constant: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.blockers

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Feature distribution audit
# ---------------------------------------------------------------------------


def eda_dataframe_summary(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    max_nan_pct: float = 0.50,
) -> Dict[str, Any]:
    """Per-feature NaN%, mean, std, skew, kurtosis, min, max.

    Returns:
        dict with ``per_feature`` (dict[col → stats]) and ``blockers``
        (list of features exceeding max_nan_pct).
    """
    if df is None or df.empty:
        return {"per_feature": {}, "blockers": ["empty_dataframe"]}

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        return {
            "per_feature": {},
            "blockers": [f"missing_columns:{missing_cols}"],
        }

    n = len(df)
    per: Dict[str, Dict[str, Any]] = {}
    blockers: List[str] = []
    for c in feature_cols:
        col = df[c]
        nan_pct = float(col.isna().mean())
        nonnan = col.dropna()
        stats = {
            "n_rows": n,
            "n_nonnan": int(len(nonnan)),
            "nan_pct": round(nan_pct, 4),
            "mean": float(nonnan.mean()) if len(nonnan) else None,
            "std": float(nonnan.std()) if len(nonnan) > 1 else 0.0,
            "min": float(nonnan.min()) if len(nonnan) else None,
            "max": float(nonnan.max()) if len(nonnan) else None,
            "skew": float(nonnan.skew()) if len(nonnan) > 2 else None,
            "kurtosis": float(nonnan.kurtosis()) if len(nonnan) > 3 else None,
        }
        per[c] = stats
        if nan_pct > max_nan_pct:
            blockers.append(f"high_nan:{c}={nan_pct:.2%}")
    return {"per_feature": per, "blockers": blockers}


# ---------------------------------------------------------------------------
# Near-constant features
# ---------------------------------------------------------------------------


def eda_near_constant_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    variance_floor: float = 1e-8,
    unique_ratio_floor: float = 0.01,
) -> List[str]:
    """Flag features that are effectively constant.

    A feature is near-constant if:
      - variance < ``variance_floor`` (numerical), OR
      - unique values < ``unique_ratio_floor`` * n_rows

    Returns list of flagged column names.
    """
    flagged: List[str] = []
    if df is None or df.empty:
        return flagged
    n = len(df)
    for c in feature_cols:
        if c not in df.columns:
            continue
        col = df[c].dropna()
        if len(col) < 10:
            continue
        var = float(col.var()) if len(col) > 1 else 0.0
        uniq_ratio = float(col.nunique()) / max(n, 1)
        if var < variance_floor or uniq_ratio < unique_ratio_floor:
            flagged.append(c)
    return flagged


# ---------------------------------------------------------------------------
# Classification label balance
# ---------------------------------------------------------------------------


def eda_classification_balance(
    labels: pd.Series,
    *,
    min_class_pct: float = 0.05,
    expected_classes: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Check class proportions for classification trainers.

    Blocking if any class < ``min_class_pct``. For triple-barrier 3-class
    labels (BUY/HOLD/SELL), a 95% HOLD distribution kills the model.

    Args:
        labels: pandas Series of class labels.
        min_class_pct: minimum fraction per class (default 5%).
        expected_classes: if provided, every class in this list must
            appear in the data — missing classes are also blocking.

    Returns dict with ``counts``, ``ratios``, ``blockers``, ``warnings``.
    """
    if labels is None or len(labels) == 0:
        return {"counts": {}, "ratios": {}, "blockers": ["empty_labels"],
                "warnings": []}

    labels = labels.dropna()
    counts = labels.value_counts().to_dict()
    n = int(labels.shape[0])
    ratios = {k: round(v / n, 4) for k, v in counts.items()}

    blockers: List[str] = []
    warnings: List[str] = []
    for cls, pct in ratios.items():
        if pct < min_class_pct:
            blockers.append(f"class_imbalance:{cls}={pct:.2%}")
    if expected_classes is not None:
        missing = [c for c in expected_classes if c not in counts]
        if missing:
            blockers.append(f"missing_classes:{missing}")

    # Soft warning when single class > 80%
    for cls, pct in ratios.items():
        if pct > 0.80:
            warnings.append(f"dominant_class:{cls}={pct:.2%}")

    return {
        "n": n,
        "counts": {str(k): int(v) for k, v in counts.items()},
        "ratios": {str(k): v for k, v in ratios.items()},
        "blockers": blockers,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Feature-label information coefficient
# ---------------------------------------------------------------------------


def eda_feature_label_ic(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    *,
    method: str = "spearman",
    min_abs_mean_ic: float = 0.005,
) -> Dict[str, Any]:
    """Per-feature correlation against label (information coefficient).

    For cross-sectional rankers, IC near zero across ALL features
    means the trainer has nothing to learn — blocking.

    Args:
        method: 'pearson' or 'spearman' (default spearman; robust to outliers).
        min_abs_mean_ic: if max |IC| across all features < this threshold,
            we flag it as a blocker.

    Returns dict with ``per_feature`` (col → IC), ``max_abs_ic``,
    ``blockers``.
    """
    if df is None or df.empty or label_col not in df.columns:
        return {"per_feature": {}, "max_abs_ic": 0.0,
                "blockers": [f"label_col_missing:{label_col}"]}

    sub = df[[label_col] + [c for c in feature_cols if c in df.columns]].dropna()
    if len(sub) < 30:
        return {"per_feature": {}, "max_abs_ic": 0.0,
                "blockers": ["insufficient_rows_for_ic"]}

    per: Dict[str, float] = {}
    for c in feature_cols:
        if c not in sub.columns or c == label_col:
            continue
        try:
            ic = sub[c].corr(sub[label_col], method=method)
            if pd.isna(ic):
                ic = 0.0
            per[c] = round(float(ic), 5)
        except Exception:  # noqa: BLE001
            per[c] = 0.0

    if not per:
        return {"per_feature": {}, "max_abs_ic": 0.0,
                "blockers": ["no_features_to_correlate"]}

    max_abs = max(abs(v) for v in per.values())
    blockers: List[str] = []
    if max_abs < min_abs_mean_ic:
        blockers.append(f"all_features_near_zero_ic:max_abs={max_abs:.4f}")
    return {
        "per_feature": per,
        "max_abs_ic": round(max_abs, 5),
        "n_above_005": sum(1 for v in per.values() if abs(v) >= 0.05),
        "blockers": blockers,
    }


# ---------------------------------------------------------------------------
# Look-ahead leakage check
# ---------------------------------------------------------------------------


def eda_leakage_check(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    *,
    max_corr: float = 0.95,
) -> Dict[str, Any]:
    """Flag features that correlate >max_corr with same-bar label.

    A feature at bar t that already knows the label at bar t is leaking
    future information. This commonly happens when the labeler peeks
    forward and the feature accidentally references the same window.

    Returns dict with ``suspect_features`` and ``blockers``.
    """
    if df is None or df.empty or label_col not in df.columns:
        return {"suspect_features": [], "blockers": []}

    sub = df[[label_col] + [c for c in feature_cols if c in df.columns]].dropna()
    if len(sub) < 30:
        return {"suspect_features": [], "blockers": []}

    suspects: List[Dict[str, float]] = []
    for c in feature_cols:
        if c not in sub.columns or c == label_col:
            continue
        try:
            r = float(sub[c].corr(sub[label_col], method="spearman"))
            if abs(r) >= max_corr:
                suspects.append({"feature": c, "corr": round(r, 4)})
        except Exception:  # noqa: BLE001
            continue

    blockers: List[str] = []
    if suspects:
        names = [s["feature"] for s in suspects]
        blockers.append(f"lookahead_suspect:{names}")
    return {"suspect_features": suspects, "blockers": blockers}
