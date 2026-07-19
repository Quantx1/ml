"""Canonical 9-stage training spine (run_pipeline).

Fixed-order, fail-loud stages with a single namespaced metrics dict so
model_versions.metrics is uniform across engines. Shared stages live here;
per-engine behavior comes from PipelineTrainer hooks. Reuses M0 primitives
verbatim (eda, quality_check, purged_cv, optuna_search, eval/*).
"""
from __future__ import annotations

import enum
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.training.base import TrainResult
from ml.training.purged_cv import PurgedCVConfig, purged_walk_forward_by_date

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised when a stage gate fails (EDA blockers, fatal quality, etc.)."""


class Stage(enum.Enum):
    DATA = "data"
    EDA = "eda"
    QUALITY = "quality"
    LABEL = "label"
    FEATURE = "feature"
    CV = "cv"
    FIT = "fit"
    HPO = "hpo"
    EVALUATION = "evaluation"
    REPORT = "report"


@dataclass
class PipelineContext:
    trainer: Any
    spec: Any
    out_dir: Path
    panel: Optional[pd.DataFrame] = None
    feats: Optional[pd.DataFrame] = None
    feature_cols: List[str] = field(default_factory=list)
    df: Optional[pd.DataFrame] = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    n_hpo_trials: int = 1
    fold_preds: List[pd.DataFrame] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


def _rank_ic(frame: pd.DataFrame, target: str = "fwd_return") -> tuple[float, int]:
    from scipy.stats import spearmanr  # noqa: PLC0415
    ics = []
    for _, g in frame.groupby("date"):
        if len(g) >= 5:
            ic = spearmanr(g["pred"], g[target]).statistic
            if not np.isnan(ic):
                ics.append(ic)
    return (float(np.mean(ics)) if ics else float("nan")), len(ics)


def _decile_spread_frame(frame: pd.DataFrame, target: str = "fwd_return") -> list:
    per_date = []
    for _, g in frame.groupby("date"):
        if len(g) >= 20:
            g = g.sort_values("pred")
            k = max(1, len(g) // 10)
            per_date.append(g[target].iloc[-k:].mean() - g[target].iloc[:k].mean())
    return per_date


def _ic_target(spec) -> str:
    """The column the ranking metrics grade against: eval.ic_target_col when an
    engine labels a TRANSFORMED return (e.g. beta-residualized), else the raw
    fwd-return key. Fold frames name the raw column 'fwd_return' internally."""
    return getattr(spec.eval, "ic_target_col", None) or "fwd_return"


def _fold_scores(model, X, spec) -> np.ndarray:
    """Model scores for a fold's test rows: P(class 1) for classification
    (conviction is a probability), the raw score otherwise."""
    if spec.eval.task == "classification" and hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _classification_fold_stats(fp: pd.DataFrame) -> dict:
    """Per-fold classification stats: AUC, Brier, base rate, prediction-tercile
    win rates (the conviction-band contract: does the top tercile actually win
    more often?). Single-class folds yield NaN AUC (excluded from the mean)."""
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415
    y = fp["label"].to_numpy(dtype=float)
    p = fp["pred"].to_numpy(dtype=float)
    out = {"n": int(len(fp)),
           "base_rate": float(y.mean()) if len(y) else float("nan"),
           "brier": float(np.mean((p - y) ** 2)) if len(y) else float("nan"),
           "auc": float("nan")}
    if len(np.unique(y)) >= 2:
        out["auc"] = float(roc_auc_score(y, p))
    if len(fp) >= 30:
        terc = pd.qcut(fp["pred"].rank(method="first"), 3, labels=False)
        out["tercile_win_rates"] = [
            round(float(y[(terc == k).to_numpy()].mean()), 4) for k in (0, 1, 2)]
        out["tercile_lift"] = round(out["tercile_win_rates"][2] - out["base_rate"], 4)
    else:
        out["tercile_win_rates"] = None
        out["tercile_lift"] = float("nan")
    return out


def _crossfit_calibrated_brier(fold_preds: List[pd.DataFrame]) -> float:
    """Mean OOS Brier of CROSS-FITTED isotonic-calibrated probabilities: for
    each fold, fit isotonic on every OTHER fold's OOS preds and score this
    fold. The shipped artifact is booster + isotonic, so the calibration gate
    must judge that pipeline — raw-score Brier conflates ranking skill with a
    miscalibration the calibrator exists to remove. NaN when < 2 folds."""
    from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
    if len(fold_preds) < 2:
        return float("nan")
    briers = []
    for i, fp in enumerate(fold_preds):
        train = pd.concat([f for j, f in enumerate(fold_preds) if j != i],
                          ignore_index=True)
        if train["label"].nunique() < 2 or float(train["pred"].std()) < 1e-9:
            continue
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(train["pred"].to_numpy(float), train["label"].to_numpy(float))
        p = iso.predict(fp["pred"].to_numpy(float))
        briers.append(float(np.mean((p - fp["label"].to_numpy(float)) ** 2)))
    return float(np.mean(briers)) if briers else float("nan")


def _evaluate_classification(ctx: PipelineContext) -> None:
    """EVALUATION stage for task='classification': AUC/Brier/tercile-lift gates
    (EvalSpec.min_auc / min_fold_auc / min_tercile_lift /
    require_brier_beats_climatology — the Brier gate judges cross-fitted
    CALIBRATED probabilities, see _crossfit_calibrated_brier). DSR/PBO are
    ranking-return concepts and are not emitted here; the money verdict for
    meta-models is the conviction-weighted backtest diagnostic, not this gate."""
    per_fold = [_classification_fold_stats(fp) for fp in ctx.fold_preds]
    aucs = [s["auc"] for s in per_fold if s["auc"] == s["auc"]]
    pooled = pd.concat(ctx.fold_preds, ignore_index=True)
    pooled_stats = _classification_fold_stats(pooled)
    if ctx.out_dir is not None:
        # Pooled OOS predictions persist for post-hoc calibration (isotonic
        # must fit on OOS scores, which only exist here) and diagnostics.
        pooled_path = Path(ctx.out_dir) / "oos_fold_preds.parquet"
        try:
            pooled.to_parquet(pooled_path, index=False)
            ctx.metrics["oos_fold_preds_path"] = str(pooled_path)
        except Exception as exc:  # noqa: BLE001 — diagnostics, not a gate
            logger.warning("[%s] oos_fold_preds dump failed: %s", ctx.spec.name, exc)
    auc_mean = float(np.mean(aucs)) if aucs else float("nan")
    brier_mean = float(np.mean([s["brier"] for s in per_fold])) if per_fold else float("nan")
    brier_calibrated = _crossfit_calibrated_brier(ctx.fold_preds)
    base_rate = pooled_stats["base_rate"]
    brier_climatology = base_rate * (1.0 - base_rate) if base_rate == base_rate else float("nan")

    ctx.metrics.update({
        "model": ctx.spec.name,
        "primary_metric": "auc_mean",
        "auc_mean": round(auc_mean, 4),
        "auc_per_fold": [round(s["auc"], 4) for s in per_fold],
        "brier_mean": round(brier_mean, 4),
        "brier_calibrated_crossfit": round(brier_calibrated, 4),
        "brier_climatology": round(brier_climatology, 4),
        "base_rate": round(base_rate, 4),
        "tercile_win_rates": pooled_stats["tercile_win_rates"],
        "tercile_lift": pooled_stats["tercile_lift"],
        "per_fold_classification": per_fold,
        "primary_value": round(auc_mean, 4),
        "n_test_rows": int(len(pooled)),
    })
    ev = ctx.spec.eval
    lift = pooled_stats["tercile_lift"]
    passed = (
        auc_mean == auc_mean and auc_mean >= ev.min_auc
        and bool(aucs) and all(a >= ev.min_fold_auc for a in aucs)
        and lift == lift and lift >= ev.min_tercile_lift
        and (not ev.require_brier_beats_climatology
             or (brier_calibrated == brier_calibrated
                 and brier_calibrated <= brier_climatology))
    )
    ctx.metrics[f"{ctx.spec.name}_quality_pass"] = bool(passed)
    if not passed:
        ctx.metrics[f"{ctx.spec.name}_quality_reason"] = (
            f"auc={auc_mean:.4f} fold_aucs={[round(a, 3) for a in aucs]} "
            f"tercile_lift={lift} brier_calibrated={brier_calibrated:.4f} "
            f"(climatology {brier_climatology:.4f}) below gate"
        )


# --- stage seams filled by later tasks (no-ops now) -----------------------
def _stage_eda(ctx: PipelineContext) -> None:
    """Stage EDA — fail-loud pre-train audit (ml/preprocessing/eda.py)."""
    from ml.preprocessing.eda import (  # noqa: PLC0415
        EDAReport, eda_classification_balance, eda_dataframe_summary,
        eda_feature_label_ic, eda_leakage_check, eda_near_constant_features,
    )
    spec, df, cols = ctx.spec, ctx.df, ctx.feature_cols
    rep = EDAReport(trainer=spec.name, n_rows=len(df), n_features=len(cols),
                    n_symbols=int(df["symbol"].nunique()))
    fs = eda_dataframe_summary(df, cols, max_nan_pct=spec.eda.max_nan_pct)
    rep.feature_summary = fs.get("per_feature", {})
    rep.blockers.extend(fs.get("blockers", []))
    rep.near_constant = eda_near_constant_features(df, cols)
    if spec.eda.check_class_balance:
        bal = eda_classification_balance(
            df[spec.label_col], min_class_pct=spec.eda.min_class_pct,
            expected_classes=spec.eda.expected_classes)
        rep.label_summary = bal
        rep.blockers.extend(bal.get("blockers", []))
    if spec.eda.run_ic_leakage:
        eda_df = df[cols].copy()
        eda_df["_label"] = df[spec.fwd_return_col].to_numpy()
        ic = eda_feature_label_ic(eda_df, cols, "_label", min_abs_mean_ic=spec.eda.min_abs_ic)
        rep.ic_summary = ic
        rep.blockers.extend(ic.get("blockers", []))
        leak = eda_leakage_check(eda_df, cols, "_label", max_corr=spec.eda.max_leakage_corr)
        rep.leakage_summary = leak
        rep.blockers.extend(leak.get("blockers", []))
    ctx.metrics["eda"] = rep.to_dict()
    if not rep.ok:
        raise PipelineError(f"[{spec.name}] EDA gate FAILED: {rep.blockers}")


def _stage_quality(ctx: PipelineContext) -> None:
    """Stage QUALITY — dead/constant feature audit (catches un-ingested cols)."""
    from ml.data.quality_check import audit_feature_matrix  # noqa: PLC0415
    audit = audit_feature_matrix(
        ctx.df[ctx.feature_cols], feature_names=list(ctx.feature_cols),
        fatal_max_constant=ctx.spec.eda.max_constant_features,
    )
    ctx.metrics["feature_audit"] = audit
    if audit.get("fatal"):
        raise PipelineError(
            f"[{ctx.spec.name}] feature quality FATAL: {audit['n_constant']} dead "
            f"features {audit['constant_features']}"
        )


def _oos_rank_ic_for_params(ctx: PipelineContext, params: Dict[str, Any]) -> float:
    """Train each purged fold with `params`, return mean OOS rank-IC. The HPO
    objective — identical metric to the evaluation stage so we tune what we ship."""
    t, spec, df = ctx.trainer, ctx.spec, ctx.df
    cv = PurgedCVConfig(n_folds=spec.cv.n_folds, test_days=spec.cv.test_days,
                        embargo_days=spec.cv.embargo_days, train_days=spec.cv.train_days)
    ics = []
    for tr_idx, te_idx in purged_walk_forward_by_date(df["date"], cv):
        tr = df.iloc[tr_idx].sort_values("date")
        te = df.iloc[te_idx].sort_values("date")
        model = t.make_model(params)
        model.fit(tr[ctx.feature_cols], tr[spec.label_col], **t.fit_args(tr))
        if spec.eval.task == "classification":
            from sklearn.metrics import roc_auc_score  # noqa: PLC0415
            y = te[spec.label_col].to_numpy(dtype=float)
            if len(np.unique(y)) < 2:
                continue
            ics.append(float(roc_auc_score(
                y, _fold_scores(model, te[ctx.feature_cols], spec))))
            continue
        target = _ic_target(spec)
        fp = pd.DataFrame({"date": te["date"].to_numpy(),
                           "pred": np.asarray(model.predict(te[ctx.feature_cols]), float),
                           "fwd_return": te[spec.fwd_return_col].to_numpy()})
        if target != "fwd_return":
            fp[target] = te[target].to_numpy()
        ic, _ = _rank_ic(fp, target=target)
        if ic == ic:
            ics.append(ic)
    return float(np.mean(ics)) if ics else float("-inf")


def _stage_hpo(ctx: PipelineContext, n_trials: Optional[int] = None) -> None:
    """Stage HPO — optional Optuna TPE over OOS rank-IC. No space/budget -> skip."""
    budget = n_trials if n_trials is not None else getattr(ctx.spec, "hpo_trials", 0)
    space = ctx.trainer.search_space() if hasattr(ctx.trainer, "search_space") else None
    if space is None or not budget:
        ctx.best_params = {}; ctx.n_hpo_trials = 1
        ctx.metrics["hpo"] = {"optimized": False, "n_trials_run": 0,
                              "reason": "no search_space" if space is None else "hpo_trials=0"}
        return
    from ml.training.optuna_search import OptunaConfig, run_optuna_search  # noqa: PLC0415
    cfg = OptunaConfig(n_trials=int(budget), direction="maximize", n_jobs=1)
    result = run_optuna_search(
        objective=lambda params: _oos_rank_ic_for_params(ctx, params),
        space=space, cfg=cfg,
    )
    ctx.best_params = result.get("best_params", {}) or {}
    ctx.n_hpo_trials = int(result.get("n_trials_run", 1))
    ctx.metrics["hpo"] = {
        "optimized": result.get("optimized"),
        "n_trials_run": ctx.n_hpo_trials,
        "best_value": result.get("best_value"),
        "best_params": ctx.best_params,
    }


def _stage_report(ctx: PipelineContext) -> List[Path]:
    from ml.training.report import write_report  # noqa: PLC0415
    try:
        return write_report(ctx.metrics, ctx.out_dir, model_name=ctx.spec.name)
    except Exception as exc:  # noqa: BLE001 — report is non-fatal
        logger.warning("[%s] report stage failed (non-fatal): %s", ctx.spec.name, exc)
        return []
# --------------------------------------------------------------------------


def _build_dataset(ctx: PipelineContext) -> None:
    """Stages DATA + FEATURE + LABEL: panel -> merged df, warmup-dropped."""
    t = ctx.trainer
    ctx.panel = t.load_panel()
    if ctx.panel is None or ctx.panel.empty:
        raise PipelineError(f"[{ctx.spec.name}] load_panel returned no data")
    ctx.feats, ctx.feature_cols = t.build_features(ctx.panel)
    labels = t.build_labels(ctx.panel)
    df = ctx.feats.merge(labels, on=["date", "symbol"], how="inner")
    n_merged = len(df)
    required = ctx.feature_cols + [ctx.spec.label_col, ctx.spec.fwd_return_col]
    if _ic_target(ctx.spec) != "fwd_return":
        required = required + [_ic_target(ctx.spec)]
    df = df.dropna(subset=required)
    ctx.df = df.sort_values("date").reset_index(drop=True)
    if ctx.df.empty:
        raise PipelineError(f"[{ctx.spec.name}] dataset empty after warmup dropna")
    # DATASET-COLLAPSE GUARD (added after the 2026-07-06 swing incident): a
    # poisoned/sparse feature column NaN-collapses most rows while leaving a
    # plausible-looking remnant that can even pass the IC gate. Losing more
    # than 85% of merged rows is never legitimate warmup — fail loud with the
    # per-column NaN counts so the culprit is named.
    kept = len(ctx.df) / max(n_merged, 1)
    if kept < 0.15:
        nan_counts = df_nan = None
        try:
            merged = ctx.feats.merge(labels, on=["date", "symbol"], how="inner")
            nan_counts = merged[ctx.feature_cols].isna().sum().sort_values(ascending=False)
            df_nan = ", ".join(f"{c}={int(v)}" for c, v in nan_counts.head(5).items())
        except Exception:  # noqa: BLE001
            df_nan = "unavailable"
        raise PipelineError(
            f"[{ctx.spec.name}] dataset collapse: only {kept:.1%} of {n_merged} merged "
            f"rows survived the feature dropna — a feature column is poisoned/sparse. "
            f"Top NaN columns: {df_nan}"
        )


def _cv_and_fit(ctx: PipelineContext) -> None:
    """Stages CV + FIT + per-fold scoring (collect OOS preds)."""
    t, spec, df = ctx.trainer, ctx.spec, ctx.df
    cv = PurgedCVConfig(
        n_folds=spec.cv.n_folds, test_days=spec.cv.test_days,
        embargo_days=spec.cv.embargo_days, train_days=spec.cv.train_days,
    )
    folds = list(purged_walk_forward_by_date(df["date"], cv))
    if not folds:
        raise PipelineError(f"[{spec.name}] purged CV produced 0 folds (history too short)")
    for tr_idx, te_idx in folds:
        tr = df.iloc[tr_idx].sort_values("date")
        te = df.iloc[te_idx].sort_values("date")
        model = t.make_model(ctx.best_params)
        model.fit(tr[ctx.feature_cols], tr[spec.label_col], **t.fit_args(tr))
        fp = pd.DataFrame({
            "date": te["date"].to_numpy(), "symbol": te["symbol"].to_numpy(),
            "pred": _fold_scores(model, te[ctx.feature_cols], spec),
            "fwd_return": te[spec.fwd_return_col].to_numpy(),
        })
        if spec.eval.task == "classification":
            fp["label"] = te[spec.label_col].to_numpy(dtype=float)
        target = _ic_target(spec)
        if target != "fwd_return":
            fp[target] = te[target].to_numpy()
        ctx.fold_preds.append(fp)
    ctx.metrics["n_folds"] = len(folds)


def _stage_evaluation(ctx: PipelineContext) -> None:
    """Stage EVALUATION: rank-IC/ICIR/decile spread + DSR/PBO (uniform metrics).
    task='classification' routes to AUC/Brier/tercile-lift gates instead."""
    if ctx.spec.eval.task == "classification":
        _evaluate_classification(ctx)
        return
    target = _ic_target(ctx.spec)
    fold_ic, fold_ic_raw, fold_spread_means, n_dates, fold_returns = [], [], [], [], []
    for fp in ctx.fold_preds:
        ic, n = _rank_ic(fp, target=target)
        fold_ic.append(ic); n_dates.append(n)
        if target != "fwd_return":
            ic_raw, _ = _rank_ic(fp)  # raw-return IC, recorded for comparison
            fold_ic_raw.append(ic_raw)
        per_date = _decile_spread_frame(fp, target=target)
        fold_returns.append(per_date)
        fold_spread_means.append(float(np.mean(per_date)) if per_date else float("nan"))

    def _mean(xs):
        v = [x for x in xs if x == x]
        return float(np.mean(v)) if v else float("nan")

    ic_mean = _mean(fold_ic)
    ic_std = float(np.std([x for x in fold_ic if x == x])) if any(x == x for x in fold_ic) else float("nan")
    icir = ic_mean / (ic_std + 1e-9) if ic_std == ic_std else float("nan")

    try:
        from ml.eval.overfitting import dsr_pbo_from_fold_returns  # noqa: PLC0415
        dsr_pbo = dsr_pbo_from_fold_returns(
            fold_returns=fold_returns,
            n_trials=max(ctx.n_hpo_trials * max(len(ctx.fold_preds), 1), 1),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("DSR/PBO failed: %s", exc)
        dsr_pbo = {"deflated_sharpe": 0.0, "probability_backtest_overfitting": 0.5}

    ctx.metrics.update({
        "model": ctx.spec.name,
        "primary_metric": ctx.spec.eval.primary_metric,
        "rank_ic_mean": round(ic_mean, 4),
        "rank_ic_std": round(ic_std, 4),
        "rank_icir": round(icir, 4),
        "decile_spread_mean": round(_mean(fold_spread_means), 4),
        "rank_ic_per_fold": [round(x, 4) for x in fold_ic],
        "decile_spread_per_fold": [round(x, 4) for x in fold_spread_means],
        "deflated_sharpe": dsr_pbo.get("deflated_sharpe"),
        "probability_backtest_overfitting": dsr_pbo.get("probability_backtest_overfitting"),
        "primary_value": round(ic_mean, 4),
        "n_test_dates": int(sum(n_dates)),
    })
    if target != "fwd_return":
        ctx.metrics["ic_target_col"] = target
        ctx.metrics["rank_ic_raw_mean"] = round(_mean(fold_ic_raw), 4)
        ctx.metrics["rank_ic_raw_per_fold"] = [round(x, 4) for x in fold_ic_raw]
    ev = ctx.spec.eval
    passed = (
        ic_mean == ic_mean and ic_mean >= ev.min_ic
        and icir == icir and icir >= ev.min_icir
        and (ev.min_deflated_sharpe <= 0 or (dsr_pbo.get("deflated_sharpe") or 0) >= ev.min_deflated_sharpe)
        and (ev.max_pbo >= 1 or (dsr_pbo.get("probability_backtest_overfitting") or 1) <= ev.max_pbo)
    )
    ctx.metrics[f"{ctx.spec.name}_quality_pass"] = bool(passed)
    if not passed:
        ctx.metrics[f"{ctx.spec.name}_quality_reason"] = (
            f"ic={ic_mean} icir={icir} dsr={dsr_pbo.get('deflated_sharpe')} "
            f"pbo={dsr_pbo.get('probability_backtest_overfitting')} below gate"
        )


def _final_fit_and_save(ctx: PipelineContext) -> List[Path]:
    """Fit on ALL usable data; write booster + feature_order + drift baseline."""
    t, spec, df = ctx.trainer, ctx.spec, ctx.df
    model = t.make_model(ctx.best_params)
    model.fit(df[ctx.feature_cols], df[spec.label_col], **t.fit_args(df))
    booster = getattr(model, "booster_", model)
    model_path = ctx.out_dir / f"{spec.name}.txt"
    booster.save_model(str(model_path))
    try:
        imp = booster.feature_importance(importance_type="gain")
        ctx.feature_importance = {c: float(v) for c, v in zip(ctx.feature_cols, imp)}
    except Exception:  # noqa: BLE001
        ctx.feature_importance = {}
    fo_path = ctx.out_dir / "feature_order.json"
    fo_path.write_text(json.dumps(list(ctx.feature_cols), indent=2))
    ctx.metrics.update({
        "n_features": len(ctx.feature_cols),
        "n_rows": int(len(df)),
        "n_symbols": int(df["symbol"].nunique()),
        "n_dates": int(df["date"].nunique()),
        "horizon": spec.horizon,
        "best_params": ctx.best_params,
        "feature_importance": ctx.feature_importance,
    })
    paths = [model_path, fo_path]
    try:
        from ml.training.baseline_drift import write_baseline  # noqa: PLC0415
        paths.append(write_baseline(df, list(ctx.feature_cols), ctx.out_dir))
    except Exception as exc:  # noqa: BLE001 — non-fatal
        logger.warning("[%s] drift baseline failed: %s", spec.name, exc)
    return paths


def run_pipeline(trainer: Any, out_dir: Path) -> TrainResult:
    """Execute the 9 stages in fixed order and write artifacts to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = trainer.engine_spec()
    ctx = PipelineContext(trainer=trainer, spec=spec, out_dir=out_dir)
    t0 = time.time()

    _build_dataset(ctx)            # DATA + FEATURE + LABEL
    _stage_eda(ctx)                # EDA gate (Task 4)
    _stage_quality(ctx)            # QUALITY gate (Task 5)
    _stage_hpo(ctx, n_trials=spec.hpo_trials or None)  # HPO (Task 6)
    _cv_and_fit(ctx)               # CV + FIT + OOS preds
    _stage_evaluation(ctx)         # EVALUATION
    artifacts = [*_final_fit_and_save(ctx)]

    ctx.metrics["train_seconds"] = round(time.time() - t0, 1)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(ctx.metrics, indent=2, default=str))
    artifacts.append(metrics_path)
    artifacts.extend(_stage_report(ctx))   # REPORT (Task 7)

    logger.info("[%s] pipeline done: %s feats, %s folds, ic=%s",
                spec.name, ctx.metrics.get("n_features"),
                ctx.metrics.get("n_folds"), ctx.metrics.get("rank_ic_mean"))
    return TrainResult(artifacts=artifacts, metrics=ctx.metrics)


__all__ = ["Stage", "PipelineContext", "PipelineError", "run_pipeline"]
