"""Serve-smoke — promote precondition that round-trips a freshly-trained
artifact through the SERVING feature contract before it can go is_prod=TRUE.

Closes the audit's #1 systemic finding (train/serve skew): the live LGBM gate
expected 15 features while the trainer shipped 30; nothing checked that the
served booster and the serve-time feature builder agree, so promotions could
silently break inference. This module makes that mismatch a hard, named gate.

A LightGBM booster stores its own feature names, so the core check is exact:
the booster's feature_name() must equal the persisted feature_order.json, and
(optionally) a real serve-built feature frame must carry those exact columns
and score without error.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _load_feature_order(model_dir: Path) -> List[str]:
    p = Path(model_dir) / "feature_order.json"
    if not p.exists():
        raise FileNotFoundError(f"no feature_order.json in {model_dir} (serve contract missing)")
    return json.loads(p.read_text())


def check_lgbm_feature_contract(booster_path: Path, feature_order: List[str]) -> Tuple[bool, str]:
    """The booster's own feature names must equal feature_order (names + order).
    This is the exact train/serve-skew guard."""
    import lightgbm as lgb  # noqa: PLC0415

    booster = lgb.Booster(model_file=str(booster_path))
    names = list(booster.feature_name())
    if names != list(feature_order):
        missing = set(feature_order) - set(names)
        extra = set(names) - set(feature_order)
        return False, (
            f"feature mismatch: booster has {len(names)} features, "
            f"feature_order has {len(feature_order)}; "
            f"missing_from_booster={sorted(missing)} extra_in_booster={sorted(extra)}"
        )
    return True, f"feature contract OK ({len(names)} features)"


def round_trip(
    booster_path: Path,
    feature_order: List[str],
    serve_features: Optional[pd.DataFrame] = None,
) -> Tuple[bool, str]:
    """Full round-trip: feature-name contract + (if a serve-built frame is
    given) the frame carries every feature_order column and the booster scores
    it to finite values."""
    import lightgbm as lgb  # noqa: PLC0415

    ok, reason = check_lgbm_feature_contract(booster_path, feature_order)
    if not ok:
        return ok, reason
    if serve_features is not None:
        missing = [c for c in feature_order if c not in serve_features.columns]
        if missing:
            return False, f"serve feature builder is missing columns: {missing}"
        booster = lgb.Booster(model_file=str(booster_path))
        preds = booster.predict(serve_features[feature_order])
        if not np.all(np.isfinite(preds)):
            return False, "booster produced non-finite predictions on serve features"
        return True, f"round-trip OK ({len(serve_features)} rows scored, {len(feature_order)} features)"
    return ok, reason


def smoke_artifact(model_dir: Path, booster_name: str,
                   serve_features: Optional[pd.DataFrame] = None) -> Tuple[bool, str]:
    """Convenience: load feature_order.json from model_dir and round-trip the
    named booster. Returns (pass, reason) — a False here MUST block promotion."""
    feature_order = _load_feature_order(model_dir)
    return round_trip(Path(model_dir) / booster_name, feature_order, serve_features)


__all__ = ["check_lgbm_feature_contract", "round_trip", "smoke_artifact"]
