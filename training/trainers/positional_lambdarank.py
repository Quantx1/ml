"""Positional engine trainer — LightGBM LambdaRank cross-sectional ranker (60d).

A thin PipelineTrainer: it declares an EngineSpec + the data/model hooks and
delegates the 9-stage lifecycle (data → EDA → quality → label → feature →
purged-CV → fit → HPO → evaluation → report) to ``ml.training.pipeline``. The
spine owns the shared stages and the uniform metrics contract; this file only
says *what positional is*:

    load_panel        load_ohlcv (offline cache)              ml/data/data_loader
    build_features    build_positional_features + RS-vs-NIFTY  (+ optional
                      TimesFM/Kronos/Chronos forecast cols when with_forecasts)
    build_labels      forward_return_quantile_labels (h=60)    ml/labeling
    make_model        LGBMRanker (LambdaRank, NDCG)
    fit_args          per-date `group` query sizes
    search_space      LightGBM HPO space (opt-in via cfg.hpo_trials)

Forecast-cache ownership: positional OWNS NOTHING — it is a READ-ONLY consumer
of momentum's ``momentum_tsfm.parquet`` / ``momentum_kronos.parquet`` AND
swing's ``swing_chronos.parquet``. It never computes forecasts, never tops up,
never saves a cache. Run the momentum + swing backfills first (GPU) when
training with_forecasts. Trains on CPU.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ml.data.data_loader import load_ohlcv
from ml.features.positional_features import (
    POSITIONAL_FEATURE_ORDER,
    build_positional_features,
)
from ml.labeling.ranking_labels import forward_return_quantile_labels
from ml.training.base import PipelineTrainer
from ml.training.optuna_search import SearchSpace
from ml.training.purged_cv import PurgedCVConfig
from ml.training.specs import CVSpec, EDASpec, EngineSpec, EvalSpec
from ml.training.trainers.momentum_lambdarank import cached_universe  # same universe

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]  # ml/training/trainers/ -> repo root


@dataclass
class PositionalConfig:
    horizon: int = 60            # forward-return label horizon (trading days)
    n_quantiles: int = 10        # relevance buckets (deciles)
    # If set (W), rank labels on vol-scaled forward return (trailing W-day
    # realized vol) instead of raw — kills the vol-lottery decile failure
    # that inverted this engine OOS.
    vol_adjust_window: Optional[int] = None
    # If set (W), rank labels on the BETA-RESIDUALIZED forward return
    # (fwd_ret - beta_t * NIFTY fwd_ret, trailing W-day rolling beta) so the
    # ranker is graded on stock SELECTION, not the beta-dominated market
    # component that drowns raw 60d returns. Requires the NIFTY benchmark;
    # build_labels raises when it's unavailable (no silent fallback).
    beta_window: Optional[int] = None
    start: date = date(2020, 1, 1)
    # end defaults to TODAY — a fixed date here silently caps the training
    # window and defeats any data refresh (momentum lesson, 2026-07-06).
    end: date = field(default_factory=date.today)
    with_forecasts: bool = False  # add TimesFM + Kronos + Chronos forecast cols
    forecast_stride: int = 5      # kept for engine symmetry (positional computes nothing)
    hpo_trials: int = 0           # >0 enables Optuna HPO over OOS rank-IC (GPU run: 30)
    # Quarterly test windows + a 60d embargo (= label horizon) — positional
    # holds ~3 months, so folds are fewer/wider than swing's monthly cadence.
    cv: PurgedCVConfig = field(default_factory=lambda: PurgedCVConfig(
        n_folds=4, test_days=126, embargo_days=60, train_days=504,
    ))
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 20],
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })


def _groups_for(dates: pd.Series):
    """LightGBM ranking 'group' = consecutive row counts per date (query).
    Requires rows already sorted by date."""
    return dates.groupby(dates, sort=False).size().to_numpy()


class PositionalTrainer(PipelineTrainer):
    """Unified-runner adapter for the positional ranker, built on the canonical
    9-stage spine.

    Discoverable via ml.training.discovery so ``python -m ml.training.runner
    --only positional_lambdarank [--promote]`` works. skip_promote_gate=True
    (the promote signal is a ranking metric, not the financial backtest gate);
    the spine emits ``positional_lambdarank_quality_pass`` which the runner
    reads.
    """

    name = "positional_lambdarank"
    requires_gpu = False  # LGBM is CPU; the forecast caches are built by momentum/swing (GPU)
    skip_promote_gate = True

    def __init__(self, cfg: Optional[PositionalConfig] = None,
                 symbols: Optional[List[str]] = None):
        self.cfg = cfg or PositionalConfig()
        self.symbols = symbols or cached_universe()

    # ---- declarative contract ------------------------------------------
    def engine_spec(self) -> EngineSpec:
        return EngineSpec(
            name=self.name, horizon=self.cfg.horizon,
            label_col="relevance", fwd_return_col="fwd_return",
            hpo_trials=self.cfg.hpo_trials,
            cv=CVSpec(n_folds=self.cfg.cv.n_folds, test_days=self.cfg.cv.test_days,
                      embargo_days=self.cfg.cv.embargo_days, train_days=self.cfg.cv.train_days),
            # With beta-residual labels the gate must grade the model on the
            # residual target it ranks (ic_target_col); grading a selection
            # model on raw beta-dominated 60d returns makes the experiment
            # unfalsifiable. fwd_return stays raw for the money backtest.
            eval=EvalSpec(task="ranking", primary_metric="rank_ic_mean",
                          min_ic=0.02, min_icir=0.5,
                          ic_target_col=("resid_fwd_return"
                                         if self.cfg.beta_window else None)),
            # RS/forecast cols are legitimately sparse early; keep the IC/leakage
            # audit recorded but don't fail positional on a single near-zero-IC
            # col (min_abs_ic=0) or on benign same-bar rank correlation (max_corr=.999).
            eda=EDASpec(max_nan_pct=0.50, min_abs_ic=0.0, run_ic_leakage=True,
                        max_leakage_corr=0.999, max_constant_features=8),
        )

    # ---- data / model hooks --------------------------------------------
    def load_panel(self) -> pd.DataFrame:
        return load_ohlcv(self.symbols, self.cfg.start, self.cfg.end)

    def build_features(self, panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        from ml.data.benchmark import load_nifty_benchmark  # noqa: PLC0415
        bench = load_nifty_benchmark(self.cfg.start, self.cfg.end)
        if bench is None:
            logger.warning("NIFTY benchmark unavailable — RS-vs-index features will be NaN")
        feats = build_positional_features(panel, benchmark=bench)
        feature_cols = list(POSITIONAL_FEATURE_ORDER)
        if self.cfg.with_forecasts:
            # Foundation-model forecast features (TimesFM + Kronos + Chronos +
            # ensemble). Positional OWNS NOTHING: all three parquets are
            # consumed READ-ONLY from the caches their owners wrote — momentum
            # owns momentum_{tsfm,kronos}.parquet, swing owns
            # swing_chronos.parquet. Never computed, never topped up, never
            # saved here. FORECAST_CACHE_DIR overrides the location.
            from ml.features.forecast_features import (  # noqa: PLC0415
                CHRONOS_FEATURES, KRONOS_FEATURES, TIMESFM_FEATURES,
                merge_forecast_features,
            )
            import os  # noqa: PLC0415
            cache_dir = Path(os.environ.get(
                "FORECAST_CACHE_DIR", str(_ROOT / "artifacts" / "forecast_cache")))

            def _read(name: str):
                p = cache_dir / name
                if not p.exists():
                    return None
                try:
                    cached = pd.read_parquet(p)
                    # pod-written parquets may carry datetime64[us] — normalize
                    cached["date"] = pd.to_datetime(cached["date"]).astype("datetime64[ns]")
                    # POISONING GUARD (caught 2026-07-06): a cache covering only
                    # a subset of the panel's symbols (e.g. a smoke-run leftover)
                    # passes the max-date "current" check, silently skips the
                    # backfill, and NaN-collapses every uncovered symbol's rows.
                    # Coverage below 90% of the panel => treat as absent.
                    panel_syms = set(panel["symbol"].unique())
                    cov = len(set(cached["symbol"].unique()) & panel_syms) / max(len(panel_syms), 1)
                    if cov < 0.9:
                        logger.warning(
                            "forecast cache %s covers only %.0f%% of panel symbols "
                            "— treating as absent (full recompute)", p.name, cov * 100)
                        return None
                    # Second poison dimension: symbols present but with sparse
                    # HISTORY (e.g. a cache whose tail top-up added all symbols
                    # for a few weeks — 2026-07-06 incident). If the median
                    # per-symbol date SPAN is under half the cache's own span,
                    # most symbols are history-starved: treat as absent.
                    span = (cached["date"].max() - cached["date"].min()).days
                    if span > 0:
                        per_sym = cached.groupby("symbol")["date"].agg(lambda s: (s.max() - s.min()).days)
                        if per_sym.median() < 0.5 * span:
                            logger.warning(
                                "forecast cache %s: median per-symbol span %.0fd << cache span %.0fd "
                                "— history-starved cache, treating as absent", p.name,
                                per_sym.median(), span)
                            return None
                    return cached
                except Exception as exc:  # noqa: BLE001
                    logger.warning("forecast cache read failed (%s): %s", p, exc)
                    return None

            frames = []
            for fname, label, owner in (
                ("momentum_tsfm.parquet", "tsfm", "momentum"),
                ("momentum_kronos.parquet", "kronos", "momentum"),
                ("swing_chronos.parquet", "chronos", "swing"),
            ):
                f = _read(fname)
                if f is None:
                    # Fail-soft at merge time (cols come out NaN) but fail-LOUD
                    # downstream: the spine drops rows with NaN in ANY feature
                    # col, so an absent parquet wipes every row →
                    # PipelineError. Positional never computes forecasts — run
                    # the owning engine's backfill first.
                    logger.warning(
                        "positional: %s absent in %s — %s cols will be all-NaN and "
                        "the spine will drop every row (%s owns that cache; run "
                        "the %s forecast backfill first)",
                        fname, cache_dir, label, owner, owner)
                else:
                    frames.append(f)
            feats = merge_forecast_features(feats, frames)
            feature_cols += (list(TIMESFM_FEATURES) + list(KRONOS_FEATURES)
                             + list(CHRONOS_FEATURES) + ["ens_fwd_ret"])
        return feats[["date", "symbol", *feature_cols]], feature_cols

    def build_labels(self, panel: pd.DataFrame) -> pd.DataFrame:
        benchmark = None
        if self.cfg.beta_window:
            from ml.data.benchmark import load_nifty_benchmark  # noqa: PLC0415
            benchmark = load_nifty_benchmark(self.cfg.start, self.cfg.end)
            if benchmark is None:
                raise ValueError(
                    "beta-residual labels need the NIFTY benchmark but both the "
                    "offline cache (data/cache/NSEI_10y.csv) and yfinance ^NSEI "
                    "are unavailable — refusing to fall back to raw labels")
        labels = forward_return_quantile_labels(
            panel[["date", "symbol", "close"]], horizon=self.cfg.horizon,
            n_quantiles=self.cfg.n_quantiles,
            vol_adjust_window=self.cfg.vol_adjust_window,
            benchmark=benchmark, beta_window=self.cfg.beta_window,
        )
        cols = ["date", "symbol", "relevance", "fwd_return"]
        if self.cfg.beta_window:
            cols.append("resid_fwd_return")  # evaluation target (ic_target_col)
        return labels[cols]

    def make_model(self, params: Dict[str, Any]):
        import lightgbm as lgb  # noqa: PLC0415
        p = dict(self.cfg.lgbm_params)
        p.update(params or {})
        return lgb.LGBMRanker(**p)

    def fit_args(self, df_tr: pd.DataFrame) -> Dict[str, Any]:
        return {"group": _groups_for(df_tr["date"])}

    def search_space(self):
        if not self.cfg.hpo_trials:
            return None
        return SearchSpace(suggest=lambda tr: {
            "num_leaves": tr.suggest_int("num_leaves", 15, 63),
            "learning_rate": tr.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": tr.suggest_int("min_child_samples", 20, 100),
            "subsample": tr.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": tr.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": tr.suggest_float("reg_lambda", 0.0, 5.0),
        })

    def serve_smoke(self, out_dir: Path) -> tuple[bool, str]:
        from ml.training.serve_smoke import smoke_artifact  # noqa: PLC0415
        return smoke_artifact(out_dir, "positional_lambdarank.txt")


def train_positional(cfg: Optional[PositionalConfig] = None,
                     symbols: Optional[List[str]] = None,
                     out_dir: Optional[Path] = None) -> dict:
    """Back-compat convenience wrapper — runs the positional engine through the
    canonical spine and returns its metrics dict. (The local smoke imports
    this; the real lifecycle lives in PositionalTrainer/run_pipeline.)
    """
    out_dir = out_dir or (_ROOT / "artifacts" / "models" / "positional_lambdarank")
    trainer = PositionalTrainer(cfg=cfg, symbols=symbols)
    return trainer.train(out_dir).metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train the positional LambdaRank ranker")
    ap.add_argument("--with-forecasts", action="store_true",
                    help="add TimesFM + Kronos + Chronos forecast features "
                         "(READ-ONLY caches — run momentum + swing backfills first)")
    ap.add_argument("--stride", type=int, default=5,
                    help="kept for engine symmetry (positional computes no forecasts)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap universe size (smoke runs)")
    ap.add_argument("--hpo-trials", type=int, default=0,
                    help="Optuna HPO trials over OOS rank-IC (0 = fixed params)")
    ap.add_argument("--vol-adjust", type=int, default=None,
                    help="rank labels on vol-scaled forward return using a trailing "
                         "N-day realized vol (default: raw forward return)")
    ap.add_argument("--beta-window", type=int, default=None,
                    help="rank labels on the beta-residualized forward return "
                         "(trailing N-day rolling beta vs NIFTY; default: raw)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="artifact directory (default: artifacts/models/"
                         "positional_lambdarank) — use a distinct dir per experiment")
    args = ap.parse_args()

    cfg = PositionalConfig(with_forecasts=args.with_forecasts, forecast_stride=args.stride,
                           hpo_trials=args.hpo_trials, vol_adjust_window=args.vol_adjust,
                           beta_window=args.beta_window)
    m = train_positional(cfg=cfg, symbols=cached_universe(limit=args.limit),
                         out_dir=args.out_dir)
    print(json.dumps(m, indent=2, default=str))
