"""
PR 199 — Qlib Alpha158 + LightGBM trainer (real Microsoft pyqlib).

Per Step 2 §1.4 (locked) — F2/F3/F5 cross-sectional alpha spine.
Uses **real Microsoft Qlib** (`pip install pyqlib`), specifically:
  - qlib.contrib.data.handler.Alpha158  — 158 cross-sectional factors
  - qlib.contrib.model.gbdt.LGBModel    — Qlib's LightGBM wrapper

NO custom port, NO in-house re-implementation. We initialize Qlib
against an NSE provider directory (built once via
``scripts/data/ingest_nse_to_qlib.py``), then run Microsoft's standard
Alpha158 → LGBModel pipeline against our universe.

This is the trainer registration so ``python -m ml.training.runner --all``
picks it up. The heavy lifting still lives in the standalone
``scripts/train/train_qlib_alpha158.py`` (which Rishi can also run manually
on Colab Pro per Step 2 §5 retrain ritual).

Eval: rank-IC mean (primary), pearson IC, ICIR, long-short decile
spread. Promote-gate friendly via skip_promote_gate=True since IC is
the right metric for cross-sectional rank models, not Sharpe.

Provider directory:
    Default ``~/.qlib/qlib_data/nse_data`` — must exist before training.
    Bootstrap: ``python scripts/data/ingest_nse_to_qlib.py``
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..base import Trainer, TrainerError, TrainResult

logger = logging.getLogger(__name__)


# Default Qlib provider directory. Override via env QLIB_PROVIDER_URI
# when running on RunPod / Colab.
DEFAULT_PROVIDER_URI = os.path.expanduser("~/.qlib/qlib_data/nse_data")

# Universe + window match scripts/train/train_qlib_alpha158.py defaults.
# Smoke mode (SMOKE_MODE=1) uses a 10-symbol list so the Alpha158 build
# fits in seconds instead of minutes.
from ml.training.smoke import is_smoke_mode  # noqa: PLC0415, E402

DEFAULT_INSTRUMENTS = "nse_all"
DEFAULT_HORIZON = 5            # 5-day forward return

# Smoke mode: keep same instruments file but shrink to a recent window
# so the cross-sectional dataset stays representative without the full
# 8-year history. Cuts Alpha158 prep + LightGBM fit to ~60 seconds.
if is_smoke_mode():
    DEFAULT_TRAIN_START = "2023-01-01"
    DEFAULT_TRAIN_END = "2024-09-30"
    DEFAULT_VALID_START = "2024-10-01"
    DEFAULT_VALID_END = "2024-12-31"
    DEFAULT_OOS_START = "2025-01-01"
    DEFAULT_OOS_END = "2025-06-30"
else:
    DEFAULT_TRAIN_START = "2018-01-01"
    DEFAULT_TRAIN_END = "2024-06-30"
    DEFAULT_VALID_START = "2024-07-01"
    DEFAULT_VALID_END = "2024-12-31"
    DEFAULT_OOS_START = "2025-01-01"
    DEFAULT_OOS_END = "2026-04-18"

# LambdaRankIC swap toggle. When set, the trainer bypasses Qlib's
# LGBModel (MSE regression) and trains raw LightGBM with the
# cross-sectional LambdaRankIC objective from ``ml.eval.lambdarank_ic``.
# Cross-sectional Rank IC is the correct objective for cross-sectional
# rankers; MSE only happens to be a reasonable proxy. Per arXiv:2605.00501,
# LambdaRankIC lifts Rank IC / ICIR / Sharpe in OOS testing.
USE_LAMBDARANK_IC = os.environ.get("QLIB_USE_LAMBDARANK_IC", "1") == "1"

# LightGBM hyperparameters for the LambdaRankIC path. Mirror Qlib's
# CSI300 Alpha158 reference values where applicable; objective is set
# to "none" because fobj overrides the built-in objective.
# LightGBM 4.x: pass the custom objective as a callable in params; the
# deprecated fobj kwarg was removed. The objective callable is wired in
# at train-time inside _train_lambdarank_ic so this dict stays sans the
# closure-captured function.
from ml.training.smoke import lightgbm_device  # noqa: PLC0415, E402

LAMBDARANK_LGBM_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 210,
    "feature_fraction": 0.8879,
    "bagging_fraction": 0.8789,
    "bagging_freq": 5,
    "lambda_l1": 205.67,
    "lambda_l2": 580.96,
    "num_threads": 4,
    "verbose": -1,
    # GPU acceleration when a CUDA device is present.
    "device": lightgbm_device(),
}
LAMBDARANK_NUM_ROUNDS = 500
LAMBDARANK_EARLY_STOPPING = 30


class QlibAlpha158Trainer(Trainer):
    name = "qlib_alpha158"
    requires_gpu = False   # LightGBM CPU is fast on Alpha158 features
    depends_on: list[str] = []
    # Cross-sectional rank model — primary metric is rank-IC, not
    # Sharpe. Skip the financial promote gate; IC threshold is checked
    # in the metrics-evaluation step instead.
    skip_promote_gate: bool = True

    def train(self, out_dir: Path) -> TrainResult:
        from ml.training.verbose import banner, step  # noqa: PLC0415
        banner(
            "qlib_alpha158",
            handler="Microsoft Qlib Alpha158 (158 cross-sectional factors)",
            objective="LambdaRankIC (custom — optimizes Rank IC directly)",
            train=f"{DEFAULT_TRAIN_START}..{DEFAULT_TRAIN_END}",
            oos=f"{DEFAULT_OOS_START}..{DEFAULT_OOS_END}",
            label=f"Ref($close, -5)/$close - 1  (5d fwd return, ranked)",
        )
        step(1, 5, "verify pyqlib (real Microsoft library)", "qlib_alpha158")
        # --- 1. Verify pyqlib is installed (real Microsoft library) ---
        try:
            import qlib  # noqa: PLC0415
            from qlib.contrib.data.handler import Alpha158  # noqa: PLC0415
            from qlib.contrib.model.gbdt import LGBModel  # noqa: PLC0415
            from qlib.data.dataset import DatasetH  # noqa: PLC0415
        except ImportError as exc:
            raise TrainerError(
                "pyqlib not installed — pip install pyqlib (Microsoft Qlib)"
            ) from exc

        # --- 2. Verify provider directory exists ---
        provider_uri = os.environ.get("QLIB_PROVIDER_URI", DEFAULT_PROVIDER_URI)
        if not Path(provider_uri).expanduser().exists():
            raise TrainerError(
                f"Qlib provider directory missing: {provider_uri}. "
                f"Run: python scripts/data/ingest_nse_to_qlib.py first."
            )

        qlib.init(provider_uri=provider_uri, region="cn")
        logger.info("Qlib initialized: provider_uri=%s", provider_uri)

        # --- 3. Build Alpha158 handler exactly as Microsoft's CSI300 example ---
        horizon = DEFAULT_HORIZON
        instruments = DEFAULT_INSTRUMENTS
        label_expr = f"Ref($close, -{horizon}) / $close - 1"
        handler = Alpha158(
            instruments=instruments,
            start_time=DEFAULT_TRAIN_START,
            end_time=DEFAULT_OOS_END,
            fit_start_time=DEFAULT_TRAIN_START,
            fit_end_time=DEFAULT_TRAIN_END,
            label=([label_expr], ["LABEL0"]),
            infer_processors=[
                {"class": "RobustZScoreNorm",
                 "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            learn_processors=[
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
            ],
        )
        dataset = DatasetH(
            handler=handler,
            segments={
                "train": (DEFAULT_TRAIN_START, DEFAULT_TRAIN_END),
                "valid": (DEFAULT_VALID_START, DEFAULT_VALID_END),
                "test": (DEFAULT_OOS_START, DEFAULT_OOS_END),
            },
        )

        # --- 4. Train: LambdaRankIC (preferred) or Qlib LGBModel MSE ---
        if USE_LAMBDARANK_IC:
            logger.info("Training LightGBM with LambdaRankIC custom objective...")
            booster, pred_series, label_series = self._train_lambdarank_ic(
                dataset, handler,
            )
            pred_df = pred_series.to_frame("score")
            label_df = label_series.to_frame("LABEL0")
            objective_used = "lambdarank_ic"
        else:
            model = LGBModel(
                loss="mse", learning_rate=0.0421,
                max_depth=8, num_leaves=210,
                colsample_bytree=0.8879, subsample=0.8789,
                lambda_l1=205.67, lambda_l2=580.96,
                num_threads=4, early_stopping_rounds=30,
                num_boost_round=500,
            )
            logger.info("Fitting Qlib LGBModel on Alpha158 features (MSE)...")
            model.fit(dataset)
            booster = getattr(model, "model", None)
            objective_used = "mse"

            pred = model.predict(dataset, segment="test")
            if isinstance(pred, pd.DataFrame):
                pred_df = pred
            else:
                pred_df = pred.to_frame("score")
            if "score" not in pred_df.columns:
                pred_df.columns = ["score"]
            label_df = dataset.prepare(
                "test", col_set="label", data_key=handler.DK_R,
            )
            if isinstance(label_df, pd.DataFrame) and label_df.shape[1] == 1:
                label_df.columns = ["LABEL0"]

        merged = pred_df.join(label_df, how="inner").dropna()
        if merged.empty:
            raise TrainerError("Qlib OOS prediction merge produced empty frame")

        pearson, spearman = [], []
        for _, grp in merged.groupby(level="datetime"):
            if len(grp) < 20:
                continue
            p = grp["score"].corr(grp["LABEL0"])
            s = grp["score"].corr(grp["LABEL0"], method="spearman")
            if not np.isnan(p):
                pearson.append(float(p))
            if not np.isnan(s):
                spearman.append(float(s))
        merged["decile"] = merged.groupby(level="datetime")["score"].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"),
        )
        top_val = merged[merged["decile"] == 9]["LABEL0"].mean()
        bot_val = merged[merged["decile"] == 0]["LABEL0"].mean()
        top = float(top_val) if pd.notna(top_val) else 0.0
        bot = float(bot_val) if pd.notna(bot_val) else 0.0

        # NaN-safe metric aggregation: when all per-day IC values are NaN
        # (degenerate case — model produces constant predictions), report
        # 0.0 instead of NaN so JSON-encode + DB insert succeed. The
        # trainer's promote_gate then correctly flags rank_ic_mean=0.0
        # as below-threshold and refuses to promote.
        rank_ic_mean = float(np.mean(spearman)) if spearman else 0.0
        rank_ic_std = float(np.std(spearman)) if len(spearman) > 1 else 0.0
        # Phase 1.7 audit fix #4.6 — ICIR is undefined for single-day
        # OOS (std=0 forces division by 1e-9 to produce a meaningless
        # huge number that fooled the real-money gate). Require at
        # least 5 days of OOS to compute ICIR; below that, report 0.0
        # and let the gate fail loudly.
        if len(spearman) >= 5 and rank_ic_std > 1e-9:
            rank_icir = rank_ic_mean / rank_ic_std
        else:
            rank_icir = 0.0
        metrics = {
            "pearson_ic_mean": float(np.mean(pearson)) if pearson else 0.0,
            "rank_ic_mean": rank_ic_mean,
            "rank_ic_std": rank_ic_std,
            "rank_icir": float(rank_icir),
            "rank_ic_n_days": int(len(spearman)),
            "top_decile_mean_return": top,
            "bottom_decile_mean_return": bot,
            "long_short_spread": top - bot,
            "oos_rows": int(len(merged)),
            "oos_symbols": int(merged.index.get_level_values("instrument").nunique()),
            "qlib_version": getattr(qlib, "__version__", "unknown"),
            "handler_class": "qlib.contrib.data.handler.Alpha158",
            "model_class": "qlib.contrib.model.gbdt.LGBModel",
            "objective_used": objective_used,
        }

        # --- 6. Save artifact (native LightGBM booster + meta) ---
        out_dir.mkdir(parents=True, exist_ok=True)
        if booster is None:
            raise TrainerError("LGBModel.model is None — fit failed silently")
        artifact = out_dir / "qlib_alpha158.txt"
        booster.save_model(str(artifact))

        logger.info(
            "qlib_alpha158: rank_ic=%.4f icir=%.2f LS_spread=%.4f",
            metrics["rank_ic_mean"], metrics["rank_icir"],
            metrics["long_short_spread"],
        )

        return TrainResult(
            artifacts=[artifact],
            metrics=metrics,
            notes=(
                f"Real Microsoft Qlib Alpha158 + LGBModel on {instruments}, "
                f"{DEFAULT_TRAIN_START}->{DEFAULT_TRAIN_END} train, "
                f"OOS {DEFAULT_OOS_START}->{DEFAULT_OOS_END}"
            ),
        )

    def _train_lambdarank_ic(self, dataset, handler):
        """Train LightGBM with the cross-sectional LambdaRankIC objective.

        Uses Qlib's prepared Alpha158 features and label segmentation,
        but trains raw LightGBM with our custom fobj + feval. The output
        booster predicts cross-sectional rank scores directly.

        Returns
        -------
        (booster, pred_series, label_series) :
            booster — trained ``lightgbm.Booster``
            pred_series — test-segment predictions as a pandas Series
                (MultiIndex: instrument, datetime)
            label_series — test-segment realized labels (same index)
        """
        import lightgbm as lgb  # noqa: PLC0415
        from ml.eval.lambdarank_ic import (  # noqa: PLC0415
            lambdarank_ic_objective,
            rank_ic_metric,
        )

        # Prepare train + valid + test segments. Qlib hands us
        # (features, label) frames keyed by ("instrument", "datetime").
        train = dataset.prepare(
            "train", col_set=["feature", "label"], data_key=handler.DK_L,
        )
        valid = dataset.prepare(
            "valid", col_set=["feature", "label"], data_key=handler.DK_L,
        )
        test = dataset.prepare(
            "test", col_set=["feature", "label"], data_key=handler.DK_R,
        )

        def _split(df):
            # Qlib gives DataFrame with MultiIndex columns: feature/label.
            X = df["feature"].values.astype(np.float32)
            # Label column has shape (n, 1); flatten to (n,)
            y = df["label"].values.astype(np.float32).ravel()
            # Group sizes: count rows per datetime (cross-section size).
            dates = df.index.get_level_values("datetime")
            # Stable group order = dates in encounter order
            _, group_sizes = np.unique(dates.values, return_counts=True)
            # ``np.unique`` returns sorted unique values; we need ENCOUNTER
            # order to match X. Re-derive via pd.Series.
            grouped = pd.Series(dates).groupby(pd.Series(dates), sort=False).size()
            group_sizes = grouped.values.astype(np.int64)
            return X, y, group_sizes, df.index

        X_tr, y_tr, g_tr, _ = _split(train)
        X_val, y_val, g_val, _ = _split(valid)
        X_te, y_te, g_te, te_idx = _split(test)

        # LambdaRankIC needs the cross-section group structure. Sort
        # within-cross-section is preserved by Qlib's data prep.
        lgb_train = lgb.Dataset(X_tr, label=y_tr, group=g_tr)
        lgb_valid = lgb.Dataset(X_val, label=y_val, group=g_val,
                                reference=lgb_train)

        params = dict(LAMBDARANK_LGBM_PARAMS)
        params["objective"] = lambdarank_ic_objective
        booster = lgb.train(
            params=params,
            train_set=lgb_train,
            num_boost_round=LAMBDARANK_NUM_ROUNDS,
            valid_sets=[lgb_valid],
            valid_names=["valid"],
            feval=rank_ic_metric,
            callbacks=[
                lgb.early_stopping(LAMBDARANK_EARLY_STOPPING),
                lgb.log_evaluation(period=50),
            ],
        )

        preds_te = booster.predict(X_te)
        pred_series = pd.Series(preds_te, index=te_idx, name="score")
        label_series = pd.Series(y_te, index=te_idx, name="LABEL0")
        return booster, pred_series, label_series

    def evaluate(self, result: TrainResult) -> Dict[str, Any]:
        m = dict(result.metrics)
        m["primary_metric"] = "rank_ic_mean"
        m["primary_value"] = result.metrics.get("rank_ic_mean")
        # Phase 1.7 audit fix #11: industry-standard signal quality is
        # measured by IC Information Ratio (ICIR = IC_mean / IC_std),
        # not raw IC. ICIR >= 1.0 means the signal is at least 1σ above
        # zero consistently; ICIR < 0.5 means the signal is noise about
        # half the time. The legacy gate only checked rank_ic_mean >= 0.02
        # which is necessary but NOT sufficient for real-money use.
        rank_ic = float(result.metrics.get("rank_ic_mean", 0.0))
        # Phase 1.7 audit fix #4.6 — TRUST the trainer's rank_icir
        # (which now handles single-day OOS correctly). The legacy
        # evaluate() recomputed icir from mean/std with a 1e-9 floor,
        # which on single-day OOS produced 0.0 — overwriting the
        # trainer's value. That hid the actual root cause (insufficient
        # OOS depth) behind a falsely "safe" 0.0. We now read the
        # trainer-emitted value verbatim AND require min OOS days for
        # the real-money gate.
        icir = float(result.metrics.get("rank_icir", 0.0))
        n_oos_days = int(result.metrics.get("rank_ic_n_days", 0))
        m["rank_icir"] = round(icir, 4)
        # Two-tier quality:
        #   shadow_ok    — IC >= 0.02 → safe for shadow / display use
        #   prod_ok      — ICIR >= 1.0 AND IC >= 0.03 AND OOS depth → real-money
        # Key MUST be f"{trainer.name}_quality_pass" = "qlib_alpha158_quality_pass"
        # so the runner's skip_promote_gate quality check actually fires. The
        # audit found this was emitted as "qlib_quality_pass" → the runner read
        # "qlib_alpha158_quality_pass", never matched, and the gate was dead.
        m["qlib_alpha158_quality_pass"] = bool(rank_ic >= 0.02 and n_oos_days >= 5)
        m["qlib_realmoney_pass"] = bool(
            rank_ic >= 0.03 and icir >= 1.0 and n_oos_days >= 60
        )
        if rank_ic < 0.02:
            m["qlib_alpha158_quality_reason"] = (
                f"rank_ic_mean {rank_ic:.4f} < 0.02 — universe too narrow "
                f"or IC too weak; manual review before is_prod=TRUE"
            )
        elif not m["qlib_realmoney_pass"]:
            m["qlib_alpha158_quality_reason"] = (
                f"ICIR {icir:.2f} < 1.0 OR IC {rank_ic:.4f} < 0.03 "
                f"OR OOS depth {n_oos_days} < 60 days — "
                f"signal not consistent enough for real-money use. Safe "
                f"for shadow/display; not safe for autonomous trading."
            )
        return m
