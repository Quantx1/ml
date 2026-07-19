"""
================================================================================
HMM MARKET REGIME DETECTOR
================================================================================
3-state Hidden Markov Model for detecting market regimes:
  - Bull (0): Strong uptrend, favor all strategies
  - Sideways (1): Range-bound, favor mean-reversion strategies
  - Bear (2): Downtrend, reduce exposure or skip trend-following

Trained on Nifty 50 daily returns + India VIX + market breadth proxy.
Used by SignalGenerator to weight strategy signals per regime.
================================================================================
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Strategy names (must match ml/scanner.py)
ALL_STRATEGIES = [
    "Consolidation_Breakout",
    "Trend_Pullback",
    "Reversal_Patterns",
    "Candle_Reversal",
    "BOS_Structure",
    "Volume_Reversal",
]

# Feature columns expected by the model
FEATURE_COLS = [
    "ret_5d",
    "ret_20d",
    "realized_vol_10d",
    "vix_level",
    "vix_5d_change",
]


class MarketRegimeDetector:
    """
    3-state Gaussian HMM: Bull (0), Sideways (1), Bear (2).
    Trained on Nifty 50 daily data + VIX + realized volatility.
    """

    REGIMES = {0: "bull", 1: "sideways", 2: "bear"}

    def __init__(self):
        self.model = None       # GaussianHMM
        self.scaler = None      # StandardScaler
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def train(self, features_df: pd.DataFrame, n_components: int = 3, n_iter: int = 200):
        """
        Train the HMM on a features DataFrame.

        Parameters
        ----------
        features_df : pd.DataFrame
            Must contain columns: ret_5d, ret_20d, realized_vol_10d, vix_level, vix_5d_change
        n_components : int
            Number of hidden states (default 3).
        n_iter : int
            EM iterations.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError(
                "hmmlearn is required for MarketRegimeDetector. "
                "Install with: pip install hmmlearn>=0.3.0"
            )

        df = features_df[FEATURE_COLS].dropna()
        if len(df) < 100:
            raise ValueError(f"Need at least 100 rows to train HMM, got {len(df)}")

        X = df.values.astype(np.float64)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit HMM
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42,
            verbose=False,
        )
        self.model.fit(X_scaled)

        # Decode states for the training data
        states = self.model.predict(X_scaled)

        # Sort states by mean return (ret_5d) so 0=bull, 2=bear
        state_returns = {}
        for s in range(n_components):
            mask = states == s
            state_returns[s] = df["ret_5d"].values[mask].mean()

        sorted_states = sorted(state_returns, key=state_returns.get, reverse=True)
        # sorted_states[0] = highest return (bull), sorted_states[-1] = lowest (bear)

        # Build mapping: old_state -> new_state
        mapping = {old: new for new, old in enumerate(sorted_states)}

        # Remap model parameters
        self._remap_states(mapping, n_components)

        self._is_trained = True
        logger.info(
            "HMM regime detector trained on %d observations. "
            "State mean returns: %s",
            len(df),
            {self.REGIMES[mapping[s]]: f"{ret:.4f}" for s, ret in state_returns.items()},
        )

    def _remap_states(self, mapping: Dict[int, int], n_components: int):
        """Remap HMM parameters so state 0=bull, 1=sideways, 2=bear."""
        order = [None] * n_components
        for old, new in mapping.items():
            order[new] = old

        # Reorder means, covars, startprob, transmat
        self.model.means_ = self.model.means_[order]
        self.model.covars_ = self.model.covars_[order]
        self.model.startprob_ = self.model.startprob_[order]

        # Reorder transition matrix rows and columns
        new_transmat = self.model.transmat_[order][:, order]
        self.model.transmat_ = new_transmat

    # ------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------

    def predict_regime(self, recent_features: pd.DataFrame) -> Dict:
        """
        Predict current market regime from recent feature observations.

        Parameters
        ----------
        recent_features : pd.DataFrame
            Last N rows (ideally 20-60 days) with FEATURE_COLS.

        Returns
        -------
        dict with keys: regime, regime_id, confidence, probabilities
        """
        if not self._is_trained:
            return self._default_regime()

        try:
            df = recent_features[FEATURE_COLS].dropna()
            if len(df) < 5:
                logger.warning("Not enough data for regime prediction, defaulting to bull")
                return self._default_regime()

            X = df.values.astype(np.float64)
            X_scaled = self.scaler.transform(X)

            # Get state probabilities for the last observation
            log_prob, state_sequence = self.model.decode(X_scaled)
            current_state = int(state_sequence[-1])

            # Posterior probabilities for the last observation
            posteriors = self.model.predict_proba(X_scaled)
            last_probs = posteriors[-1]

            confidence = float(last_probs[current_state])

            return {
                "regime": self.REGIMES.get(current_state, "unknown"),
                "regime_id": current_state,
                "confidence": round(confidence, 4),
                "probabilities": {
                    self.REGIMES[i]: round(float(last_probs[i]), 4)
                    for i in range(len(last_probs))
                },
            }
        except Exception as e:
            logger.error(f"Regime prediction failed: {e}")
            return self._default_regime()

    @staticmethod
    def _default_regime() -> Dict:
        """Fallback: assume bull regime (all strategies active)."""
        return {
            "regime": "bull",
            "regime_id": 0,
            "confidence": 0.0,
            "probabilities": {"bull": 1.0, "sideways": 0.0, "bear": 0.0},
        }

    # ------------------------------------------------------------------
    # STRATEGY WEIGHTING
    # ------------------------------------------------------------------

    def get_strategy_weights(self, regime_id: int) -> Dict[str, float]:
        """
        Return strategy weight multipliers for the given regime.

        Bull (0):     All strategies at full weight.
        Sideways (1): Mean-reversion strategies favored; trend-following reduced.
        Bear (2):     All reduced; Trend_Pullback disabled.
        """
        if regime_id == 0:
            # Bull — all strategies active
            return {s: 1.0 for s in ALL_STRATEGIES}

        elif regime_id == 1:
            # Sideways — favor structure/reversal, reduce trend-following
            return {
                "Consolidation_Breakout": 0.5,
                "Trend_Pullback": 0.5,
                "Reversal_Patterns": 1.0,
                "Candle_Reversal": 1.0,
                "BOS_Structure": 1.0,
                "Volume_Reversal": 1.0,
            }

        elif regime_id == 2:
            # Bear — reduce all, skip Trend_Pullback
            return {
                "Consolidation_Breakout": 0.5,
                "Trend_Pullback": 0.0,
                "Reversal_Patterns": 0.5,
                "Candle_Reversal": 0.5,
                "BOS_Structure": 0.5,
                "Volume_Reversal": 0.5,
            }

        # Unknown — treat as bull
        return {s: 1.0 for s in ALL_STRATEGIES}

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save trained model + scaler to pickle."""
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "is_trained": self._is_trained,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Regime detector saved to {path}")

    def load(self, path: str):
        """Load trained model + scaler from pickle."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self._is_trained = data.get("is_trained", True)
        logger.info(f"Regime detector loaded from {path}")


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING HELPERS
# ---------------------------------------------------------------------------

def compute_regime_features(nifty_df: pd.DataFrame, vix_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute regime features from Nifty 50 daily OHLCV data.

    Parameters
    ----------
    nifty_df : pd.DataFrame
        Nifty 50 daily data with 'close' column.
    vix_df : pd.DataFrame, optional
        India VIX daily data with 'close' column. If None, realized vol is used as proxy.

    Returns
    -------
    pd.DataFrame with FEATURE_COLS.
    """
    df = nifty_df.copy()
    close = df["close"] if "close" in df.columns else df["Close"]

    # Returns
    df["ret_5d"] = close.pct_change(5)
    df["ret_20d"] = close.pct_change(20)

    # Realized volatility (10-day rolling std of daily returns, annualized)
    daily_ret = close.pct_change()
    df["realized_vol_10d"] = daily_ret.rolling(10).std() * np.sqrt(252)

    # VIX level
    if vix_df is not None and len(vix_df) > 0:
        vix_close = vix_df["close"] if "close" in vix_df.columns else vix_df["Close"]
        # Align VIX to Nifty index
        vix_aligned = vix_close.reindex(df.index, method="ffill")
        df["vix_level"] = vix_aligned
        df["vix_5d_change"] = vix_aligned.pct_change(5)
    else:
        # Proxy: use realized vol scaled to typical VIX range (12-30)
        logger.info("VIX data not available, using realized vol as proxy")
        df["vix_level"] = df["realized_vol_10d"] * 100  # rough proxy
        df["vix_5d_change"] = df["vix_level"].pct_change(5)

    return df[FEATURE_COLS].copy()
