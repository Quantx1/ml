"""Pre-training EDA + preprocessing audit (locked 2026-05-12).

Per-trainer audit run BEFORE training that surfaces:
    - feature distributions (NaN%, skew, kurtosis, near-constant flags)
    - label balance (for classification trainers)
    - feature-label IC (cross-sectional rankers)
    - leakage checks (feature timestamp vs label timestamp)

Used by ``scripts/train/eda_report.py`` and ``scripts/runpod/runpod_full_pipeline.sh``
Phase 8c. Hard-fails if any class balance is <5% or any feature has
>50% NaN — no fallbacks (per locked memory 2026-04-19).
"""

from .eda import (
    EDAReport,
    eda_classification_balance,
    eda_dataframe_summary,
    eda_feature_label_ic,
    eda_leakage_check,
    eda_near_constant_features,
)

__all__ = [
    "EDAReport",
    "eda_classification_balance",
    "eda_dataframe_summary",
    "eda_feature_label_ic",
    "eda_leakage_check",
    "eda_near_constant_features",
]
