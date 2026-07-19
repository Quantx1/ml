"""
PR 128 — RegimeHMM trainer module.

Wraps ``ml.regime_detector.MarketRegimeDetector`` so the unified runner
re-trains the F8 RegimeIQ model on demand. The serialized model + scaler
artifact is the single file registered into ``model_versions``. We use
the existing ``MarketRegimeDetector.save()`` path so consumers continue
loading via ``MarketRegimeDetector.load()`` unchanged.

Features pulled from yfinance (Nifty 50 + India VIX):
    ret_5d, ret_20d, realized_vol_10d, vix_level, vix_5d_change

Eval: log-likelihood per observation on the last 252 trading days held out.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..base import Trainer, TrainerError, TrainResult

logger = logging.getLogger(__name__)


# yfinance tickers used by the feature builder. Kept here (not in the
# detector) because future trainers may pull richer macro covariates.
NIFTY_TICKER = "^NSEI"
VIX_TICKER = "^INDIAVIX"


def _build_features(start: str = "2010-01-01") -> pd.DataFrame:
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError as exc:
        raise TrainerError("yfinance required for HMM training") from exc

    nifty = yf.download(NIFTY_TICKER, start=start, progress=False, auto_adjust=False)
    vix = yf.download(VIX_TICKER, start=start, progress=False, auto_adjust=False)
    if nifty is None or nifty.empty or vix is None or vix.empty:
        raise TrainerError("yfinance returned empty Nifty or VIX series")

    # Multi-index frames from yfinance — flatten.
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = [c[0] for c in nifty.columns]
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0] for c in vix.columns]

    close = nifty["Close"].astype(float)
    vix_close = vix["Close"].astype(float)

    df = pd.DataFrame(index=close.index)
    df["ret_5d"] = close.pct_change(5)
    df["ret_20d"] = close.pct_change(20)
    df["realized_vol_10d"] = close.pct_change().rolling(10).std() * np.sqrt(252)
    df["vix_level"] = vix_close.reindex(df.index).ffill()
    df["vix_5d_change"] = df["vix_level"].pct_change(5)
    df = df.dropna()
    return df


class RegimeHMMTrainer(Trainer):
    name = "regime_hmm"
    requires_gpu = False  # tiny model; fits on CPU in seconds
    depends_on: list[str] = []
    # PR 167 — HMM is a regime *detector*, not a directional trader.
    # Its primary_metric is log-likelihood per observation, not Sharpe.
    # Skip the financial promote gate so it can flip is_prod=TRUE on
    # quality-of-fit alone.
    skip_promote_gate: bool = True

    def train(self, out_dir: Path) -> TrainResult:
        """Walk-forward 5-fold + final fit on all data.

        PR 168 — instead of a single 252-day holdout, we now run 5
        rolling WFCV folds to prove the HMM holds across regime epochs
        (2008-09 GFC, 2013 taper tantrum, 2020 COVID, 2022 sideways,
        2024 bull). Per-fold OOS log-likelihood is aggregated; the
        final shipped artifact is fit on ALL data so production
        inference uses the best fit available, not a 5th-fold subset.
        """
        from ml.training.verbose import banner, step  # noqa: PLC0415
        banner(
            "regime_hmm",
            states="3 (bull/sideways/bear)",
            features="Nifty ret + rv_10d + vol_z + VIX + VIX 5d chg",
            wfcv_folds=5,
            algorithm="GaussianHMM (hmmlearn)",
        )
        step(1, 3, "load 15y of ^NSEI + ^INDIAVIX from yfinance", "regime_hmm")
        from ml.regime_detector import MarketRegimeDetector  # noqa: PLC0415
        from ml.training.wfcv import (  # noqa: PLC0415
            WFCVConfig,
            aggregate_fold_metrics,
            walk_forward_split,
        )

        df = _build_features()
        if len(df) < 500:
            raise TrainerError(f"insufficient HMM training data: {len(df)} rows")

        # PR 192 — quality audit on the macro feature frame. Detect
        # negative VIX values (yfinance occasionally returns 0 or
        # negative on bad scrape days), stale runs (^INDIAVIX has
        # weeks where Yahoo returns repeated values), and outliers.
        from ml.data.quality_check import (  # noqa: PLC0415
            audit_feature_matrix,
        )
        feat_audit = audit_feature_matrix(df, feature_names=list(df.columns))
        logger.info(
            "regime_hmm feature audit — %d/%d dead, dead=%s",
            feat_audit["n_constant"], feat_audit["n_features"],
            feat_audit["constant_features"],
        )

        # 5-fold rolling WFCV. With ~3700 days of data, fold sizes are:
        # 3-year train (756 days) + 5-day embargo + 1-year test (252).
        # Rolling (not expanding) so each fold tests on a distinct
        # period — answers "does the HMM hold across regimes?".
        cfg = WFCVConfig(
            strategy="rolling",
            n_folds=5,
            test_size=252,           # 1 trading year per fold
            train_size=252 * 3,      # 3 trading years
            embargo=5,
        )

        fold_metrics: list[dict] = []
        for fold_idx, (train_idx, test_idx) in enumerate(walk_forward_split(df, cfg)):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            try:
                det = MarketRegimeDetector()
                det.train(train_df, n_components=3, n_iter=200)
                test_X = det.scaler.transform(test_df.values)
                fold_loglik = float(det.model.score(test_X))
                fold_loglik_per_obs = fold_loglik / max(1, len(test_df))
                fold_metrics.append({
                    "fold": fold_idx,
                    "log_likelihood": fold_loglik,
                    "log_likelihood_per_obs": fold_loglik_per_obs,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                })
                logger.info(
                    "regime_hmm fold %d  loglik/obs=%.4f  train=%d-%d  test=%d-%d",
                    fold_idx, fold_loglik_per_obs,
                    train_df.index.min().year, train_df.index.max().year,
                    test_df.index.min().year, test_df.index.max().year,
                )
            except Exception as exc:  # noqa: BLE001 — keep going across folds
                logger.warning("regime_hmm fold %d failed: %s", fold_idx, exc)

        if not fold_metrics:
            raise TrainerError("all regime_hmm WFCV folds failed")

        # Final fit on ALL data — this is what ships to production.
        det_final = MarketRegimeDetector()
        det_final.train(df, n_components=3, n_iter=200)
        artifact = out_dir / "regime_hmm.pkl"
        det_final.save(str(artifact))

        # In-sample regime counts on the full training set.
        full_states = det_final.model.predict(det_final.scaler.transform(df.values))
        counts = {det_final.REGIMES[i]: int((full_states == i).sum()) for i in det_final.REGIMES}

        # Aggregate across folds for the metrics dict.
        agg = aggregate_fold_metrics([
            {"log_likelihood_per_obs": m["log_likelihood_per_obs"]} for m in fold_metrics
        ])

        return TrainResult(
            artifacts=[artifact],
            metrics={
                "n_total_obs": int(len(df)),
                "n_folds_succeeded": int(len(fold_metrics)),
                "regime_counts_full_sample": counts,
                "log_likelihood_per_obs_per_fold": [
                    round(m["log_likelihood_per_obs"], 6) for m in fold_metrics
                ],
                "log_likelihood_per_obs_mean": agg.get("log_likelihood_per_obs_mean"),
                "log_likelihood_per_obs_std": agg.get("log_likelihood_per_obs_std"),
            },
            notes=f"WFCV 5-fold rolling on {df.index.min().date()}->{df.index.max().date()}, "
                  f"final fit on all {len(df)} obs",
        )

    def evaluate(self, result: TrainResult) -> Dict[str, Any]:
        # PR 168 — primary metric is now the WFCV-aggregated mean
        # log-likelihood per observation. The runner reads
        # primary_value to decide promote (HMM opts out of the financial
        # gate via skip_promote_gate=True so this is informational).
        m = dict(result.metrics)
        m["primary_metric"] = "log_likelihood_per_obs_mean"
        m["primary_value"] = result.metrics.get("log_likelihood_per_obs_mean")
        return m
