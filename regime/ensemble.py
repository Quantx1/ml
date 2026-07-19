"""Regime ensemble: jump model + Gaussian HMM + transparent rules, with
majority voting and hysteresis.

WHY AN ENSEMBLE: regimes are latent — no single detector is right in real
time. The jump model is robust to fat tails, the HMM contributes calibrated
probabilistic structure, and the rules model is an interpretable prior that
can never silently drift. Majority voting + hysteresis targets the module's
objectives: hindsight agreement, low turning-point lag and — above all — NO
WHIPSAW, since every official regime flip re-gates live momentum/swing books.

STRICT NO-LOOKAHEAD: ``fit(features_df)`` trains jump + HMM on the given
history only (walk-forward refits re-estimate everything). ``run_online``
uses each model's filtered/causal path (``predict_online``), an
expanding-window volatility percentile for the rules, and a purely
sequential hysteresis pass — the official state at day t depends only on
information <= t.

State labels everywhere: 0 = bear, 1 = sideways, 2 = bull.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ml.regime.features import REGIME_FEATURES
from ml.regime.hmm_model import RegimeHMM
from ml.regime.jump_model import JumpModel

# Regime label axis: states are ordered bear/sideways/bull by their mean 21d
# return — NOT ret_1d, whose regime means are noise-dominated (caught on real
# data: label scrambling collapsed bear recall to 4.7%).
_RELABEL_IDX = REGIME_FEATURES.index("ret_21d")

STATE_NAMES = {0: "bear", 1: "sideways", 2: "bull"}

#: Model-input columns for jump + HMM. ``ret_21d`` MUST stay first — both
#: models relabel their states by ascending mean of feature 0, which is what
#: pins 0=bear / 1=sideways / 2=bull deterministically.
CORE_MODEL_FEATURES = [
    "ret_21d",
    "ret_5d",
    "realized_vol_21",
    "vol_of_vol_21",
    "drawdown",
    "trend_200",
    "trend_50",
    "sma50_slope_21",
]
_BREADTH_COLS = ["breadth_200", "breadth_change_21"]


# ---------------------------------------------------------------- rules model
def rules_states(features_df: pd.DataFrame) -> np.ndarray:
    """Transparent rule-based regime per row. Pure function, causal.

    * bull:  trend_200 > 0 AND realized_vol_21 below its trailing
      (expanding-window) 80th percentile — "uptrend, calm tape".
    * bear:  trend_200 < -0.02 OR (drawdown < -0.10 AND vol at/above the
      trailing 80th percentile) — "downtrend, or deep drawdown in panic vol".
    * else sideways.

    The vol percentile is an EXPANDING quantile (min 63 obs), so day t only
    ever compares against vol history <= t. NaN features (warmup) compare
    False on every branch and fall through to sideways — fail-soft neutral.
    """
    vol = features_df["realized_vol_21"]
    vol_q80 = vol.expanding(min_periods=63).quantile(0.8)
    vol_high = (vol >= vol_q80).fillna(False).to_numpy()
    vol_low = (vol < vol_q80).fillna(False).to_numpy()
    trend = features_df["trend_200"]
    dd = features_df["drawdown"]

    bear = (trend < -0.02).fillna(False).to_numpy() | (
        (dd < -0.10).fillna(False).to_numpy() & vol_high
    )
    bull = (trend > 0).fillna(False).to_numpy() & vol_low
    return np.where(bear, 0, np.where(bull, 2, 1)).astype(np.int64)


# ----------------------------------------------------------------- hysteresis
def apply_hysteresis(raw_states: np.ndarray, hysteresis_days: int = 3) -> np.ndarray:
    """Debounce the raw ensemble path: the OFFICIAL state switches only after
    ``hysteresis_days`` CONSECUTIVE days of one different raw state.

    A shorter excursion (or a broken streak) never flips the official state —
    this is the explicit anti-whipsaw stage. Purely sequential (causal); the
    official state at t depends only on raw states <= t.
    """
    raw = np.asarray(raw_states, dtype=np.int64)
    if raw.size == 0:
        return raw.copy()
    h = max(1, int(hysteresis_days))
    out = np.empty_like(raw)
    current = raw[0]
    candidate, streak = -1, 0
    out[0] = current
    for i in range(1, len(raw)):
        s = raw[i]
        if s == current:
            candidate, streak = -1, 0
        elif s == candidate:
            streak += 1
        else:
            candidate, streak = s, 1
        if streak >= h:
            current = candidate
            candidate, streak = -1, 0
        out[i] = current
    return out


def _majority_vote(votes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-day majority over (n_models, T) votes -> (raw_state, confidence).

    Strict majority (> half the models). A three-way split has no majority:
    the raw state then carries over from the previous day (persistence beats
    a coin flip for regime gating); day 0 falls back to sideways. Confidence
    is the fraction of models agreeing with the chosen raw state.
    """
    n_models, T = votes.shape
    raw = np.empty(T, dtype=np.int64)
    conf = np.empty(T, dtype=float)
    prev = 1  # sideways fallback at t=0
    for t in range(T):
        counts = np.bincount(votes[:, t], minlength=3)
        top = int(np.argmax(counts))
        if counts[top] * 2 > n_models:
            raw[t] = top
        else:
            raw[t] = prev
        conf[t] = counts[raw[t]] / n_models
        prev = raw[t]
    return raw, conf


# ------------------------------------------------------------------ ensemble
class RegimeEnsemble:
    """Majority-vote regime ensemble with hysteresis.

    Args:
        models: any subset of ('jump', 'hmm', 'rules').
        hysteresis_days: consecutive raw days required to flip the official
            state (see ``apply_hysteresis``).
        lambda_jump / n_states: forwarded to the underlying models.

    Usage (walk-forward-safe):
        >>> ens = RegimeEnsemble().fit(features_hist)     # history only
        >>> out = ens.run_online(features_hist_plus_new)  # causal replay
    """

    def __init__(
        self,
        models: Sequence[str] = ("jump", "hmm", "rules"),
        hysteresis_days: int = 3,
        lambda_jump: float = 50.0,
        n_states: int = 3,
    ) -> None:
        unknown = set(models) - {"jump", "hmm", "rules"}
        if unknown or not models:
            raise ValueError(f"models must be a non-empty subset of jump/hmm/rules, got {models}")
        self.models = tuple(models)
        self.hysteresis_days = int(hysteresis_days)
        self.lambda_jump = float(lambda_jump)
        self.n_states = int(n_states)
        self.jump_: Optional[JumpModel] = None
        self.hmm_: Optional[RegimeHMM] = None

    def _resolve_feature_cols(self, features_df: pd.DataFrame) -> list[str]:
        """Core features + breadth iff breadth is actually populated —
        fail-soft when no universe panel was available."""
        cols = list(CORE_MODEL_FEATURES)
        for c in _BREADTH_COLS:
            if c in features_df.columns and features_df[c].notna().any():
                cols.append(c)
        return cols

    def _matrix(self, features_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        df = features_df.dropna(subset=self.feature_cols_).reset_index(drop=True)
        if df.empty:
            raise ValueError("no rows left after dropping feature warmup NaNs")
        return df, df[self.feature_cols_].to_numpy(dtype=float)

    def fit(self, features_df: pd.DataFrame) -> "RegimeEnsemble":
        """Fit jump + HMM on the given HISTORY (output of
        ``build_regime_features``). Walk-forward-safe: uses only the rows
        passed in; refit on a schedule to roll the window forward."""
        self.feature_cols_ = self._resolve_feature_cols(features_df)
        _, X = self._matrix(features_df)
        if "jump" in self.models:
            self.jump_ = JumpModel(n_states=self.n_states, lambda_jump=self.lambda_jump, relabel_idx=_RELABEL_IDX).fit(X)
        if "hmm" in self.models:
            self.hmm_ = RegimeHMM(n_states=self.n_states, relabel_idx=_RELABEL_IDX).fit(X)
        return self

    def run_online(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Causal replay over ``features_df`` (warmup-NaN rows dropped).

        Returns DataFrame ``['date', 'state', 'state_name', 'confidence',
        'raw_state']`` where ``state`` is the hysteresis-debounced official
        regime, ``raw_state`` the per-day majority vote and ``confidence``
        the fraction of models agreeing with the raw state. Pass history +
        new rows together so the rules' expanding vol percentile has context;
        every stage is causal, so earlier rows are unaffected by later ones.
        """
        if not hasattr(self, "feature_cols_"):
            raise RuntimeError("RegimeEnsemble.run_online called before fit()")
        df, X = self._matrix(features_df)
        votes: list[np.ndarray] = []
        if "jump" in self.models:
            if self.jump_ is None:
                raise RuntimeError("jump model requested but not fitted")
            votes.append(self.jump_.predict_online(X))
        if "hmm" in self.models:
            if self.hmm_ is None:
                raise RuntimeError("hmm model requested but not fitted")
            votes.append(self.hmm_.predict_online(X))
        if "rules" in self.models:
            votes.append(rules_states(df))

        raw, conf = _majority_vote(np.stack(votes))
        official = apply_hysteresis(raw, self.hysteresis_days)
        return pd.DataFrame(
            {
                "date": df["date"].to_numpy(),
                "state": official,
                "state_name": [STATE_NAMES.get(int(s), str(s)) for s in official],
                "confidence": conf,
                "raw_state": raw,
            }
        )


__all__ = [
    "CORE_MODEL_FEATURES",
    "STATE_NAMES",
    "RegimeEnsemble",
    "apply_hysteresis",
    "rules_states",
]
