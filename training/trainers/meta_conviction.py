"""Meta-labeling conviction trainer — P(signal wins) per style engine.

A PipelineTrainer over the walk-forward harness's per-name OOS fold
predictions (López de Prado meta-labeling): the primary engine decides WHAT
to surface, this model learns WHEN the primary is right. Claims no new alpha
— it grades the error structure of alpha already validated.

    load_panel        fold-preds parquet (scripts/eval/backtest_engines.py
                      --dump-preds) — raises with instructions when absent
    build_features    ml.features.meta_features (signal context + regime +
                      market + selected engine columns; ~20 PIT features)
    build_labels      meta_win = net H-day return > 0 (raw win, per spec §3)
    make_model        LGBMClassifier (small, depth-capped)
    gates             AUC ≥ .55 (no fold < .5), top-tercile lift ≥ +5pp,
                      Brier ≤ climatology  (EvalSpec, task="classification")

After the spine, ``train()`` fits ISOTONIC CALIBRATION on the pooled OOS
fold predictions and writes ``calibration.json`` + ``conviction_bands.json``
— ONLY when the quality gate passed (no-fallbacks: a failed gate ships
nothing). Serving consumes booster + calibration + bands.

Spec: docs/superpowers/specs/2026-07-07-meta-labeling-conviction-design.md.
Trains on CPU in minutes; training data comes free from the backtests.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.features.meta_features import build_meta_features
from ml.training.base import PipelineTrainer, TrainerError, TrainResult
from ml.training.optuna_search import SearchSpace
from ml.training.purged_cv import PurgedCVConfig
from ml.training.specs import CVSpec, EDASpec, EngineSpec, EvalSpec

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]  # ml/training/trainers/ -> repo root
if not (_ROOT / "data").exists():  # standalone ml checkout: package dir is the repo root
    _ROOT = Path(__file__).resolve().parents[2]

#: Primary-engine horizon (embargo + label window) — must match the engines.
_ENGINE_HORIZON = {"momentum": 20, "swing": 10}


@dataclass
class MetaConvictionConfig:
    engine: str = "momentum"          # "momentum" | "swing"
    cost_bps_side: float = 30.0       # per-side cost in the win/lose label
    # "excess" (default, Experiment 2): meta_win = beat the equal-weight
    # universe over the horizon — grades SELECTION, which the program proved
    # is regime-robust. "raw" (Experiment 1) = net absolute win; it FAILED
    # both gates because the label is dominated by the market's H-day
    # direction, so the model became a weak market-timer (~80% of importance
    # on date-level features, fold AUC 0.70 only in the regime-break fold).
    label_mode: str = "excess"        # "excess" | "raw"
    hpo_trials: int = 0               # Optuna over OOS AUC (0 = fixed params)
    preds_path: Optional[Path] = None  # default: artifacts/eval/fold_preds/<engine>_preds.parquet
    regime_path: Optional[Path] = None  # default: artifacts/regime/regime_series.parquet
    # Sized to the fold-preds panel (12 x 63d = 756 dates): 252 + 20 + 4x110
    # = 712 <= 756. Meta rows exist only where the PRIMARY was OOS, so the
    # meta panel is fixed-length — wider windows simply don't fit.
    cv: PurgedCVConfig = field(default_factory=lambda: PurgedCVConfig(
        n_folds=4, test_days=110, embargo_days=20, train_days=252,
    ))
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })

    def __post_init__(self) -> None:
        if self.engine not in _ENGINE_HORIZON:
            raise ValueError(f"unknown engine '{self.engine}' "
                             f"(available: {list(_ENGINE_HORIZON)})")
        if self.label_mode not in ("excess", "raw"):
            raise ValueError(f"label_mode must be 'excess' or 'raw', "
                             f"got '{self.label_mode}'")
        # embargo must cover the primary label window
        self.cv.embargo_days = max(self.cv.embargo_days, _ENGINE_HORIZON[self.engine])


class MetaConvictionTrainer(PipelineTrainer):
    """Conviction classifier for one style engine, on the canonical spine."""

    requires_gpu = False
    skip_promote_gate = True  # promote decision is manual (founder-gated feature)

    def __init__(self, cfg: Optional[MetaConvictionConfig] = None):
        self.cfg = cfg or MetaConvictionConfig()
        self.name = f"meta_conviction_{self.cfg.engine}"
        self._horizon = _ENGINE_HORIZON[self.cfg.engine]

    # ---- declarative contract ------------------------------------------
    def engine_spec(self) -> EngineSpec:
        return EngineSpec(
            name=self.name, horizon=self._horizon,
            label_col="meta_win", fwd_return_col="net_fwd_return",
            hpo_trials=self.cfg.hpo_trials,
            cv=CVSpec(n_folds=self.cfg.cv.n_folds, test_days=self.cfg.cv.test_days,
                      embargo_days=self.cfg.cv.embargo_days,
                      train_days=self.cfg.cv.train_days),
            # Pre-registered conviction gates (spec §6). All four must hold.
            eval=EvalSpec(task="classification", primary_metric="auc_mean",
                          min_auc=0.55, min_fold_auc=0.5, min_tercile_lift=0.05,
                          require_brier_beats_climatology=True),
            # score_pct/pred legitimately correlate with the outcome; the
            # leakage corr gate stays near-off like the other engines.
            eda=EDASpec(max_nan_pct=0.50, min_abs_ic=0.0, run_ic_leakage=True,
                        max_leakage_corr=0.999, check_class_balance=True,
                        min_class_pct=0.15, expected_classes=(0, 1),
                        max_constant_features=3),
        )

    # ---- data / model hooks --------------------------------------------
    def _preds_path(self) -> Path:
        if self.cfg.preds_path is not None:
            return Path(self.cfg.preds_path)
        return (_ROOT / "artifacts" / "eval" / "fold_preds"
                / f"{self.cfg.engine}_preds.parquet")

    def load_panel(self) -> pd.DataFrame:
        p = self._preds_path()
        if not p.exists():
            raise TrainerError(
                f"[{self.name}] fold predictions missing: {p}. Generate them "
                f"first: python3 scripts/eval/backtest_engines.py --engines "
                f"{self.cfg.engine} --dump-preds")
        preds = pd.read_parquet(p)
        preds["date"] = pd.to_datetime(preds["date"]).astype("datetime64[ns]")
        need = {"date", "symbol", "pred", "fwd_return"}
        if not need <= set(preds.columns):
            raise TrainerError(f"[{self.name}] {p} lacks {need - set(preds.columns)}")
        return preds

    def _engine_trainer(self):
        """The primary engine's trainer — its feature frame is the source of
        the per-name columns META_ENGINE_MAP joins (never recomputed here)."""
        if self.cfg.engine == "momentum":
            from ml.training.trainers.momentum_lambdarank import (  # noqa: PLC0415
                MomentumConfig, MomentumTrainer, cached_universe)
            return MomentumTrainer(cfg=MomentumConfig(with_forecasts=True),
                                   symbols=cached_universe())
        from ml.training.trainers.swing_lambdarank import (  # noqa: PLC0415
            SwingConfig, SwingTrainer)
        from ml.training.trainers.momentum_lambdarank import cached_universe  # noqa: PLC0415
        return SwingTrainer(cfg=SwingConfig(with_forecasts=True),
                            symbols=cached_universe())

    def build_features(self, panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        eng = self._engine_trainer()
        engine_feats, _ = eng.build_features(eng.load_panel())

        regime_path = Path(self.cfg.regime_path or
                           os.environ.get("META_REGIME_PATH",
                                          _ROOT / "artifacts" / "regime" / "regime_series.parquet"))
        if not regime_path.exists():
            raise TrainerError(f"[{self.name}] regime series missing: {regime_path} "
                               f"— run the ml/regime pipeline first")
        regime = pd.read_parquet(regime_path)

        from ml.data.benchmark import load_nifty_benchmark  # noqa: PLC0415
        start = (panel["date"].min() - pd.Timedelta(days=200)).date()
        end = panel["date"].max().date()
        bench = load_nifty_benchmark(start, end)
        if bench is None:
            raise TrainerError(f"[{self.name}] NIFTY benchmark unavailable — "
                               f"market-context features need it (no fallback)")

        df, cols = build_meta_features(panel, engine_feats, self.cfg.engine,
                                       regime, bench)
        return df[["date", "symbol", *cols]], cols

    def build_labels(self, panel: pd.DataFrame) -> pd.DataFrame:
        cost_rt = 2.0 * self.cfg.cost_bps_side / 10_000.0
        out = panel[["date", "symbol"]].copy()
        out["net_fwd_return"] = panel["fwd_return"] - cost_rt
        if self.cfg.label_mode == "excess":
            # Within-date excess win: beat the equal-weight cross-section over
            # the horizon. Costs cancel (every name pays the same); the label
            # uses the same-date peers' realized returns — future info is
            # legitimate LABEL material, never a feature.
            date_mean = panel.groupby("date")["fwd_return"].transform("mean")
            out["meta_win"] = (panel["fwd_return"] > date_mean).astype(int)
        else:
            out["meta_win"] = (out["net_fwd_return"] > 0).astype(int)
        return out

    def make_model(self, params: Dict[str, Any]):
        import lightgbm as lgb  # noqa: PLC0415
        p = dict(self.cfg.lgbm_params)
        p.update(params or {})
        return lgb.LGBMClassifier(**p)

    def search_space(self):
        if not self.cfg.hpo_trials:
            return None
        return SearchSpace(suggest=lambda tr: {
            "num_leaves": tr.suggest_int("num_leaves", 7, 31),
            "max_depth": tr.suggest_int("max_depth", 3, 6),
            "learning_rate": tr.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": tr.suggest_int("min_child_samples", 50, 300),
            "subsample": tr.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": tr.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": tr.suggest_float("reg_lambda", 0.0, 5.0),
        })

    def serve_smoke(self, out_dir: Path) -> tuple[bool, str]:
        from ml.training.serve_smoke import smoke_artifact  # noqa: PLC0415
        return smoke_artifact(out_dir, f"{self.name}.txt")

    # ---- calibration (post-spine; gate-conditional) ----------------------
    def train(self, out_dir: Path) -> TrainResult:
        result = super().train(out_dir)
        if result.metrics.get(f"{self.name}_quality_pass"):
            result.artifacts.extend(self._fit_calibration(Path(out_dir)))
        else:
            logger.warning("[%s] gate FAILED — no calibration artifact written "
                           "(the conviction feature does not ship)", self.name)
        return result

    def _fit_calibration(self, out_dir: Path) -> List[Path]:
        """Isotonic calibration + tercile bands from the pooled OOS fold preds
        the evaluation stage persisted. Probabilities shown to users and used
        for sizing MUST be calibrated — raw booster scores are not."""
        from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
        pooled_path = out_dir / "oos_fold_preds.parquet"
        if not pooled_path.exists():
            raise TrainerError(f"[{self.name}] {pooled_path} missing — the "
                               f"classification evaluation stage should have written it")
        pooled = pd.read_parquet(pooled_path)
        p = pooled["pred"].to_numpy(dtype=float)
        y = pooled["label"].to_numpy(dtype=float)
        if float(np.std(p)) < 1e-6:
            raise TrainerError(f"[{self.name}] degenerate OOS scores (std~0) — "
                               f"cannot calibrate")
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p, y)
        calib = {"x": [float(v) for v in iso.X_thresholds_],
                 "y": [float(v) for v in iso.y_thresholds_]}
        calibrated = iso.predict(p)
        lo, hi = (float(np.quantile(calibrated, q)) for q in (1 / 3, 2 / 3))
        bands = {"low_max": round(lo, 4), "medium_max": round(hi, 4),
                 "n_oos_rows": int(len(pooled))}
        calib_path = out_dir / "calibration.json"
        bands_path = out_dir / "conviction_bands.json"
        calib_path.write_text(json.dumps(calib))
        bands_path.write_text(json.dumps(bands, indent=2))
        logger.info("[%s] isotonic calibration on %d OOS rows; bands %.3f/%.3f",
                    self.name, len(pooled), lo, hi)
        return [calib_path, bands_path]


def train_meta_conviction(cfg: Optional[MetaConvictionConfig] = None,
                          out_dir: Optional[Path] = None) -> dict:
    cfg = cfg or MetaConvictionConfig()
    trainer = MetaConvictionTrainer(cfg=cfg)
    out_dir = out_dir or (_ROOT / "artifacts" / "models" / trainer.name)
    return trainer.train(out_dir).metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train the conviction meta-model")
    ap.add_argument("--engine", choices=sorted(_ENGINE_HORIZON), default="momentum")
    ap.add_argument("--hpo-trials", type=int, default=0)
    ap.add_argument("--preds-path", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = MetaConvictionConfig(engine=args.engine, hpo_trials=args.hpo_trials,
                               preds_path=args.preds_path)
    m = train_meta_conviction(cfg=cfg, out_dir=args.out_dir)
    print(json.dumps(m, indent=2, default=str))
