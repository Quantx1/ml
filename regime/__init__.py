"""Regime detection for gating cross-sectional momentum/swing books.

Successor to the legacy ``ml.training.trainers.regime_hmm`` prod model (left
untouched). Pipeline:

    build_regime_features (NIFTY + breadth, strictly trailing)
        -> RegimeEnsemble(JumpModel + RegimeHMM + rules, hysteresis)
        -> official state path 0=bear / 1=sideways / 2=bull

Design principles (full text in ``ml.regime.features``): regimes are LATENT —
we optimize hindsight agreement, turning-point lag and whipsaw, and the gated
books must beat ungated (utility judged in walk-forward backtests, not here).
STRICT NO-LOOKAHEAD in every online path; ``hindsight_labels`` is the one
deliberate lookahead and is EVALUATION ONLY — never a feature.
"""
from ml.regime.ensemble import (
    CORE_MODEL_FEATURES,
    STATE_NAMES,
    RegimeEnsemble,
    apply_hysteresis,
    rules_states,
)
from ml.regime.features import REGIME_FEATURES, REGIME_WARMUP_BARS, build_regime_features
from ml.regime.hindsight import agreement_report, hindsight_labels
from ml.regime.hmm_model import RegimeHMM
from ml.regime.jump_model import JumpModel

__all__ = [
    "CORE_MODEL_FEATURES",
    "REGIME_FEATURES",
    "REGIME_WARMUP_BARS",
    "STATE_NAMES",
    "JumpModel",
    "RegimeEnsemble",
    "RegimeHMM",
    "agreement_report",
    "apply_hysteresis",
    "build_regime_features",
    "hindsight_labels",
    "rules_states",
]
