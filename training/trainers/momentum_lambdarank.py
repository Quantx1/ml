"""Momentum engine trainer — LightGBM LambdaRank cross-sectional ranker.

A thin PipelineTrainer: it declares an EngineSpec + the data/model hooks and
delegates the 9-stage lifecycle (data → EDA → quality → label → feature →
purged-CV → fit → HPO → evaluation → report) to ``ml.training.pipeline``. The
spine owns the shared stages and the uniform metrics contract; this file only
says *what momentum is*:

    load_panel        load_ohlcv (offline cache)            ml/data/data_loader
    build_features    build_momentum_features + RS-vs-NIFTY  (+ optional
                      TimesFM/Kronos forecast cols when with_forecasts)
    build_labels      forward_return_quantile_labels         ml/labeling
    make_model        LGBMRanker (LambdaRank, NDCG)
    fit_args          per-date `group` query sizes
    search_space      LightGBM HPO space (opt-in via cfg.hpo_trials)

The LGBM ranker IS the signal. Sector-RS is the one deferred feature family
(needs sector-index ingestion). Trains on CPU; forecast columns need a GPU.
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
from ml.features.momentum_features import (
    MOMENTUM_FEATURE_ORDER,
    build_momentum_features,
)
from ml.labeling.ranking_labels import forward_return_quantile_labels
from ml.training.base import PipelineTrainer, TrainResult
from ml.training.optuna_search import SearchSpace
from ml.training.purged_cv import PurgedCVConfig
from ml.training.specs import CVSpec, EDASpec, EngineSpec, EvalSpec

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]  # ml/training/trainers/ -> repo root
_CACHE_DIR = _ROOT / "data" / "cache"
_CACHE_SUFFIX = "_NS_10y.csv"


@dataclass
class MomentumConfig:
    horizon: int = 20            # forward-return label horizon (trading days)
    n_quantiles: int = 10        # relevance buckets (deciles)
    start: date = date(2020, 1, 1)
    # end defaults to TODAY — a fixed date here silently caps the training
    # window and defeats any data refresh (caught 2026-07-06: the default was
    # still 2026-03-01, so a run on July-fresh data trained only through Feb).
    end: date = field(default_factory=date.today)
    with_forecasts: bool = False  # add TimesFM + Kronos forecast cols (GPU)
    forecast_stride: int = 5      # weekly rebalance for forecast inference (cost)
    hpo_trials: int = 0           # >0 enables Optuna HPO over OOS rank-IC (GPU run: 30)
    cv: PurgedCVConfig = field(default_factory=lambda: PurgedCVConfig(
        n_folds=5, test_days=63, embargo_days=20, train_days=378,
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


def cached_universe(limit: Optional[int] = None) -> List[str]:
    """Training universe. Prefers local OHLCV cache files (offline / dev box);
    falls back to the tracked NSE-500 tier list (data/nse_tiers/) on a fresh
    pod where data/cache is gitignored — OHLCV is then fetched via yfinance by
    the data plane."""
    syms = sorted(
        f.name[: -len(_CACHE_SUFFIX)]
        for f in _CACHE_DIR.glob(f"*{_CACHE_SUFFIX}")
    )
    if not syms:
        for name in ("nifty500.txt", "nse_all.txt"):
            tier = _ROOT / "data" / "nse_tiers" / name
            if tier.exists():
                syms = sorted(
                    ln.strip().upper() for ln in tier.read_text().splitlines()
                    if ln.strip() and not ln.startswith("#")
                )
                logger.info("universe from %s: %d symbols (cache empty)", name, len(syms))
                break
    return syms[:limit] if limit else syms


def _groups_for(dates: pd.Series):
    """LightGBM ranking 'group' = consecutive row counts per date (query).
    Requires rows already sorted by date."""
    return dates.groupby(dates, sort=False).size().to_numpy()


class MomentumTrainer(PipelineTrainer):
    """Unified-runner adapter for the momentum ranker, built on the canonical
    9-stage spine.

    Discoverable via ml.training.discovery so ``python -m ml.training.runner
    --only momentum_lambdarank [--promote]`` works. skip_promote_gate=True (the
    promote signal is a ranking metric, not the financial backtest gate); the
    spine emits ``momentum_lambdarank_quality_pass`` which the runner reads.
    """

    name = "momentum_lambdarank"
    requires_gpu = False  # LGBM is CPU; forecast cols (with_forecasts) need GPU, off by default
    skip_promote_gate = True

    def __init__(self, cfg: Optional[MomentumConfig] = None,
                 symbols: Optional[List[str]] = None):
        self.cfg = cfg or MomentumConfig()
        self.symbols = symbols or cached_universe()

    # ---- declarative contract ------------------------------------------
    def engine_spec(self) -> EngineSpec:
        return EngineSpec(
            name=self.name, horizon=self.cfg.horizon,
            label_col="relevance", fwd_return_col="fwd_return",
            hpo_trials=self.cfg.hpo_trials,
            cv=CVSpec(n_folds=self.cfg.cv.n_folds, test_days=self.cfg.cv.test_days,
                      embargo_days=self.cfg.cv.embargo_days, train_days=self.cfg.cv.train_days),
            eval=EvalSpec(task="ranking", primary_metric="rank_ic_mean",
                          min_ic=0.02, min_icir=0.5),
            # RS/forecast cols are legitimately sparse early; keep the IC/leakage
            # audit recorded but don't fail momentum on a single near-zero-IC col
            # (min_abs_ic=0) or on benign same-bar rank correlation (max_corr=.999).
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
        feats = build_momentum_features(panel, benchmark=bench)
        feature_cols = list(MOMENTUM_FEATURE_ORDER)
        if self.cfg.with_forecasts:
            # GPU foundation-model forecast features (TimesFM + Kronos + ensemble).
            from ml.features.forecast_features import (  # noqa: PLC0415
                KRONOS_FEATURES, TIMESFM_FEATURES, kronos_forecast_features,
                merge_forecast_features, timesfm_forecast_features,
            )
            # Cache-first: the backfill is the expensive GPU phase (hours of
            # Kronos autoregression). Reuse the persisted parquets and compute
            # ONLY rebalance dates newer than the cache (incremental top-up),
            # then save the combined frames back. FORCE_FORECAST_BACKFILL=1
            # recomputes everything; FORECAST_CACHE_DIR overrides the location.
            import os  # noqa: PLC0415
            cache_dir = Path(os.environ.get(
                "FORECAST_CACHE_DIR", str(_ROOT / "artifacts" / "forecast_cache")))
            force = os.environ.get("FORCE_FORECAST_BACKFILL") == "1"
            panel_max = pd.Timestamp(panel["date"].max())

            def _cached(name: str):
                p = cache_dir / name
                if force or not p.exists():
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

            def _topup(cached, compute, label: str):
                """cached frame + compute(min_date) for the missing tail."""
                if cached is None or cached.empty:
                    return compute(None)
                cmax = pd.Timestamp(cached["date"].max())
                if cmax >= panel_max:
                    logger.info("forecast cache %s current through %s — skipping compute",
                                label, cmax.date())
                    return cached
                logger.info("forecast cache %s ends %s — topping up to %s",
                            label, cmax.date(), panel_max.date())
                new = compute(cmax + pd.Timedelta(days=1))
                out = pd.concat([cached, new], ignore_index=True)
                return (out.drop_duplicates(["date", "symbol"], keep="last")
                           .sort_values(["date", "symbol"]).reset_index(drop=True))

            tsfm = _topup(_cached("momentum_tsfm.parquet"),
                          lambda md: timesfm_forecast_features(
                              panel, horizon=self.cfg.horizon,
                              stride=self.cfg.forecast_stride, min_date=md),
                          "tsfm")
            kronos = _topup(_cached("momentum_kronos.parquet"),
                            lambda md: kronos_forecast_features(
                                panel, horizon=self.cfg.horizon,
                                stride=self.cfg.forecast_stride, min_date=md),
                            "kronos")
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                tsfm.to_parquet(cache_dir / "momentum_tsfm.parquet", index=False)
                kronos.to_parquet(cache_dir / "momentum_kronos.parquet", index=False)
                logger.info("forecast cache saved (tsfm=%d, kronos=%d rows)",
                            len(tsfm), len(kronos))
            except Exception as exc:  # noqa: BLE001 — never kill a run on a cache write
                logger.warning("forecast-cache persist failed (non-fatal): %s", exc)
            feats = merge_forecast_features(feats, [tsfm, kronos])
            # Momentum's serving contract is tsfm + kronos + ens ONLY — chronos
            # belongs to the swing engine. Momentum never computes chronos, so
            # its columns come back all-NaN from the merge; including them here
            # would wipe every row in the spine's dropna(feature_cols), and
            # adding chronos to momentum for real requires a retrain
            # (feature_order.json changes). Do NOT switch to FORECAST_FEATURES.
            feature_cols += list(TIMESFM_FEATURES) + list(KRONOS_FEATURES) + ["ens_fwd_ret"]
        return feats[["date", "symbol", *feature_cols]], feature_cols

    def build_labels(self, panel: pd.DataFrame) -> pd.DataFrame:
        labels = forward_return_quantile_labels(
            panel[["date", "symbol", "close"]], horizon=self.cfg.horizon,
            n_quantiles=self.cfg.n_quantiles,
        )
        return labels[["date", "symbol", "relevance", "fwd_return"]]

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
        return smoke_artifact(out_dir, "momentum_lambdarank.txt")


def train_momentum(cfg: Optional[MomentumConfig] = None,
                   symbols: Optional[List[str]] = None,
                   out_dir: Optional[Path] = None) -> dict:
    """Back-compat convenience wrapper — runs the momentum engine through the
    canonical spine and returns its metrics dict. (Existing callers + the local
    smoke import this; the real lifecycle lives in MomentumTrainer/run_pipeline.)
    """
    out_dir = out_dir or (_ROOT / "artifacts" / "models" / "momentum_lambdarank")
    trainer = MomentumTrainer(cfg=cfg, symbols=symbols)
    return trainer.train(out_dir).metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train the momentum LambdaRank ranker")
    ap.add_argument("--with-forecasts", action="store_true",
                    help="add TimesFM + Kronos forecast features (requires GPU)")
    ap.add_argument("--stride", type=int, default=5,
                    help="forecast rebalance stride in trading days (cost control)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap universe size (smoke runs)")
    ap.add_argument("--hpo-trials", type=int, default=0,
                    help="Optuna HPO trials over OOS rank-IC (0 = fixed params)")
    args = ap.parse_args()

    cfg = MomentumConfig(with_forecasts=args.with_forecasts, forecast_stride=args.stride,
                         hpo_trials=args.hpo_trials)
    m = train_momentum(cfg=cfg, symbols=cached_universe(limit=args.limit))
    print(json.dumps(m, indent=2, default=str))
