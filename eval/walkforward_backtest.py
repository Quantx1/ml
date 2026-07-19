"""Walk-forward top-N PORTFOLIO backtest — the "does this predict real alpha,
net of costs?" verdict tool for the style engines (momentum/swing/positional).

The training spine's evaluation stage answers "does the ranker order stocks
correctly?" (rank-IC). This module answers the money question: if we actually
TRADED the ranking — buy the top-N names each day, hold for the engine's own
label horizon H — would the portfolio beat an equal-weight universe hold, after
transaction costs, out of sample?

Semantics (deliberately simple, and honest about every approximation):

1. DATASET — built from the trainer's own hooks (``load_panel`` →
   ``build_features`` → ``build_labels``, merged on (date, symbol), warmup
   dropna on features + relevance + fwd_return, sorted by date). This is
   byte-for-byte the spine's dataset (``ml.training.pipeline._build_dataset``)
   — reused, not reinvented. No backend imports.

2. FOLDS — ``purged_walk_forward_by_date`` with the trainer's own
   ``engine_spec().cv`` windows: the exact purged walk-forward refit pattern
   of ``ml.training.pipeline._cv_and_fit``. Each fold trains a fresh model
   (tuned ``params`` if given) on the purged train window and predicts the
   out-of-sample test window only.

3. PORTFOLIO — per test DATE d: select the top_n names by prediction; the
   portfolio's H-day period return is the mean REALIZED ``fwd_return`` of
   those names (the engine's own label horizon H). The internal benchmark is
   the equal-weight mean ``fwd_return`` of ALL names tradable that date —
   label columns only, no external index data.

4. PER-DAY CONVERSION — H-day returns are geometrically de-compounded to a
   per-day rate: ``daily = (1 + r_H) ** (1/H) - 1``. NOTE (the big honest
   caveat): consecutive test dates give OVERLAPPING H-day forward windows, so
   the daily series is an overlapping-window APPROXIMATION of a top-N
   strategy rebalanced daily with H-day holds (1/H of the book rolled each
   day). It is not a path-accurate equity curve; overlap induces
   autocorrelation that slightly flatters daily Sharpe. Treat the outputs as
   a relative verdict tool (engine vs engine vs benchmark, gross vs net),
   not as a brokerage-statement simulation.

5. COSTS — steady-state daily turnover of a top-N, H-day-hold book is ≈ 1/H
   of the book replaced per day, each replacement paying two sides:
   ``cost_daily = (1/H) * 2 * cost_bps_side / 10_000``. Flat per-side bps —
   no size-aware market impact (see ``ml.eval.impact_cost`` for the
   Almgren-Chriss model when position sizes are known).

6. METRICS — ``ml.eval.backtest_eval.metrics_from_returns`` on the gross,
   net (= gross - cost) and excess (= net - benchmark) daily series, plus
   ``ml.eval.overfitting.deflated_sharpe_ratio`` on the net Sharpe with
   ``n_trials_for_dsr`` accounting for the search that produced the params.

All returned values are JSONB-safe (float | int | str | list | dict).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.eval.backtest_eval import metrics_from_returns
from ml.eval.overfitting import deflated_sharpe_ratio
from ml.training.purged_cv import PurgedCVConfig, purged_walk_forward_by_date

logger = logging.getLogger(__name__)


def _h_to_daily(r_h: float, horizon: int) -> float:
    """Geometric per-day rate of an H-day simple return.

    Guarded at -99.9% per H-day window so a corrupt label can't produce a
    complex/NaN root; anything near the guard is upstream data corruption.
    """
    return float(np.power(max(1.0 + r_h, 1e-3), 1.0 / horizon) - 1.0)


def _daily_win_rate(returns: List[float]) -> float:
    """Fraction of positive days (metrics_from_returns has no daily win rate)."""
    if not returns:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    return round(float(np.mean(arr > 0)), 4)


def _iid_stats(h_returns, horizon: int) -> dict:
    """Annualized stats on NON-OVERLAPPING H-period returns (every H-th test
    date) — statistically independent samples, immune to the overlap-smoothing
    that inflates the daily-series Sharpe. The defensible headline numbers."""
    import numpy as _np
    arr = _np.asarray(list(h_returns), dtype=float)
    if arr.size < 3:
        return {"sharpe": 0.0, "n_periods": int(arr.size), "period_mean": 0.0,
                "cagr": 0.0, "max_drawdown": 0.0}
    periods_per_year = 252.0 / horizon
    mu, sd = float(arr.mean()), float(arr.std(ddof=1))
    sharpe = (mu / sd) * (periods_per_year ** 0.5) if sd > 1e-12 else 0.0
    equity = _np.cumprod(1.0 + _np.clip(arr, -0.999, None))
    years = arr.size / periods_per_year
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else 0.0
    dd = float((equity / _np.maximum.accumulate(equity) - 1.0).min())
    return {"sharpe": round(sharpe, 4), "n_periods": int(arr.size),
            "period_mean": round(mu, 5), "cagr": round(cagr, 4),
            "max_drawdown": round(dd, 4)}


def walkforward_portfolio_backtest(
    trainer: Any,
    *,
    top_n: int = 20,
    cost_bps_side: float = 30.0,
    params: Optional[Dict[str, Any]] = None,
    n_trials_for_dsr: int = 150,
    dump_preds_path: Optional[Path] = None,
) -> dict:
    """Run the walk-forward top-N portfolio backtest for one engine.

    Args:
        trainer: a PipelineTrainer-shaped object (duck-typed) exposing
            ``engine_spec() / load_panel() / build_features(panel) /
            build_labels(panel) / make_model(params) / fit_args(df)``.
        top_n: names held per test date (portfolio breadth).
        cost_bps_side: one-side transaction cost in basis points (30 = 0.30%
            per side ≈ conservative NSE retail incl. STT + slippage).
        params: model hyperparameters passed to ``trainer.make_model`` per
            fold (e.g. the tuned ``hpo.best_params`` from the training run).
            None/{} = the trainer's built-in defaults.
        n_trials_for_dsr: number of variants that competed for these params
            (HPO trials); feeds the Deflated Sharpe null benchmark.
        dump_preds_path: if set, write every fold's PER-NAME out-of-sample
            predictions to this parquet (columns date, symbol, fold, pred,
            fwd_return — fwd_return is the RAW realized H-day return). This
            is the meta-labeling training set: OOS primary-model predictions
            with realized outcomes, purged/embargoed by construction.

    Returns:
        dict — see module docstring; keys: engine, horizon, n_folds,
        n_test_dates, top_n, cost_bps_side, gross, net, excess,
        deflated_sharpe_net, per_fold_net_sharpe, avg_daily_turnover,
        cost_bps_per_day, notes.

    Raises:
        ValueError: empty dataset after warmup dropna, or history too short
            for the engine's purged-CV windows.
    """
    spec = trainer.engine_spec()
    horizon = int(spec.horizon)
    if horizon < 1:
        raise ValueError(f"[{spec.name}] horizon must be >= 1, got {horizon}")
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")

    # -- 1. dataset via the trainer's own hooks (the spine's dataset) --------
    panel = trainer.load_panel()
    if panel is None or panel.empty:
        raise ValueError(f"[{spec.name}] load_panel returned no data")
    feats, feature_cols = trainer.build_features(panel)
    labels = trainer.build_labels(panel)
    df = feats.merge(labels, on=["date", "symbol"], how="inner")
    df = df.dropna(subset=[*feature_cols, spec.label_col, spec.fwd_return_col])
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"[{spec.name}] dataset empty after warmup dropna")

    # -- 2. purged walk-forward folds (the spine's exact refit pattern) ------
    cv = PurgedCVConfig(
        n_folds=spec.cv.n_folds, test_days=spec.cv.test_days,
        embargo_days=spec.cv.embargo_days, train_days=spec.cv.train_days,
    )
    folds = list(purged_walk_forward_by_date(df["date"], cv))
    if not folds:
        raise ValueError(f"[{spec.name}] purged CV produced 0 folds (history too short)")

    cost_daily = (2.0 * float(cost_bps_side) / 10_000.0) / horizon

    gross_d: List[float] = []
    bench_d: List[float] = []
    iid_h: List[tuple] = []   # every H-th test date's (gross_H, bench_H): independent periods
    per_date: List[dict] = []  # dated series for downstream regime-gated evaluation
    net_d: List[float] = []
    excess_d: List[float] = []
    per_fold_net_sharpe: List[float] = []
    pred_frames: List[pd.DataFrame] = []  # per-name OOS preds (meta-labeling data)
    n_test_dates = 0

    # -- 3./4./5. per-fold refit -> per-date top-N portfolio series ----------
    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        tr = df.iloc[tr_idx].sort_values("date")
        te = df.iloc[te_idx].sort_values("date")
        model = trainer.make_model(dict(params) if params else {})
        model.fit(tr[feature_cols], tr[spec.label_col], **trainer.fit_args(tr))
        te = te.assign(pred=np.asarray(model.predict(te[feature_cols]), dtype=float))
        if dump_preds_path is not None:
            pred_frames.append(
                te[["date", "symbol", "pred", spec.fwd_return_col]]
                .rename(columns={spec.fwd_return_col: "fwd_return"})
                .assign(fold=fold_i)
            )

        fold_net: List[float] = []
        fold_h: List[tuple] = []
        for d, g in te.groupby("date", sort=True):
            top = g.nlargest(min(top_n, len(g)), "pred")
            gross_h = float(top[spec.fwd_return_col].mean())
            bench_h = float(g[spec.fwd_return_col].mean())
            gd = _h_to_daily(gross_h, horizon)
            bd = _h_to_daily(bench_h, horizon)
            nd = gd - cost_daily
            gross_d.append(gd)
            bench_d.append(bd)
            net_d.append(nd)
            excess_d.append(nd - bd)
            fold_net.append(nd)
            fold_h.append((gross_h, bench_h))
            per_date.append({"date": str(getattr(d, "date", lambda: d)()),
                             "fold": fold_i, "gross_h": round(gross_h, 6),
                             "bench_h": round(bench_h, 6)})
            n_test_dates += 1
        iid_h.extend(fold_h[::horizon])   # non-overlapping within the fold
        fold_sharpe = metrics_from_returns(fold_net)["sharpe"] if fold_net else 0.0
        per_fold_net_sharpe.append(float(fold_sharpe))
        logger.info("[%s] fold %d/%d: %d test dates, net sharpe %.2f",
                    spec.name, fold_i + 1, len(folds), len(fold_net), fold_sharpe)

    # -- 6. metrics on the pooled OOS daily series ----------------------------
    gross = metrics_from_returns(gross_d)
    net = metrics_from_returns(net_d, benchmark_returns=bench_d)
    excess = metrics_from_returns(excess_d)
    gross["daily_win_rate"] = _daily_win_rate(gross_d)
    net["daily_win_rate"] = _daily_win_rate(net_d)
    excess["daily_win_rate"] = _daily_win_rate(excess_d)

    net_arr = np.asarray(net_d, dtype=float)
    try:
        from scipy.stats import kurtosis, skew  # noqa: PLC0415
        s = float(skew(net_arr, bias=False)) if net_arr.size >= 3 else 0.0
        k = float(kurtosis(net_arr, fisher=False, bias=False)) if net_arr.size >= 4 else 3.0
    except Exception:  # noqa: BLE001 — moments are a refinement, not a gate
        s, k = 0.0, 3.0
    dsr_net = deflated_sharpe_ratio(
        sharpe=float(net["sharpe"]), n_trials=int(n_trials_for_dsr),
        n_obs=int(net_arr.size), skew=s, kurtosis=k,
    )

    cost_rt = 2.0 * float(cost_bps_side) / 10_000.0   # round-trip per H-period
    net_iid = _iid_stats((g - cost_rt for g, _ in iid_h), horizon)
    excess_iid = _iid_stats((g - cost_rt - b for g, b in iid_h), horizon)
    try:
        from scipy.stats import kurtosis as _kurt, skew as _skew  # noqa: PLC0415
        _ih = np.asarray([g - cost_rt for g, _ in iid_h], dtype=float)
        _s2 = float(_skew(_ih, bias=False)) if _ih.size >= 3 else 0.0
        _k2 = float(_kurt(_ih, fisher=False, bias=False)) if _ih.size >= 4 else 3.0
    except Exception:  # noqa: BLE001
        _s2, _k2 = 0.0, 3.0
    dsr_net_iid = deflated_sharpe_ratio(
        sharpe=float(net_iid["sharpe"]), n_trials=int(n_trials_for_dsr),
        n_obs=int(net_iid["n_periods"]), skew=_s2, kurtosis=_k2,
    )

    notes = [
        (f"net_iid/excess_iid: NON-OVERLAPPING evaluation — every {horizon}th test "
         f"date per fold gives independent {horizon}-day periods; these are the "
         f"statistically defensible headline numbers (the daily-series sharpe is "
         f"inflated by overlap autocorrelation up to ~sqrt(H))"),
        (f"overlapping-window approximation: consecutive test dates give overlapping "
         f"{horizon}-day forward windows; the daily series approximates a top-{top_n} "
         f"book rebalanced daily with {horizon}-day holds, NOT a path-accurate equity curve"),
        (f"per-day conversion is geometric: daily = (1 + r_H)^(1/{horizon}) - 1 "
         f"for both portfolio and benchmark"),
        (f"costs: steady-state turnover = 1/H = {1.0 / horizon:.3f} of the book/day, "
         f"2 sides x {float(cost_bps_side):.1f} bps -> {cost_daily * 1e4:.2f} bps/day; "
         f"flat per-side bps, no size-aware market impact (ml.eval.impact_cost exists "
         f"for that when position sizes are known)"),
        ("benchmark = equal-weight mean realized fwd_return of ALL tradable names each "
         "test date (internal, label-derived; no external index data)"),
        ("per-fold models refit on purged walk-forward train windows exactly as in "
         "ml.training.pipeline._cv_and_fit"),
        f"model params: {'tuned (explicit)' if params else 'trainer defaults'}",
    ]

    fold_preds_path = None
    if dump_preds_path is not None:
        out = pd.concat(pred_frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
        dump_preds_path = Path(dump_preds_path)
        dump_preds_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(dump_preds_path, index=False)
        fold_preds_path = str(dump_preds_path)
        logger.info("[%s] dumped %d per-name OOS fold predictions -> %s",
                    spec.name, len(out), dump_preds_path)

    return {
        "engine": str(spec.name),
        "horizon": horizon,
        "fold_preds_path": fold_preds_path,
        "n_folds": int(len(folds)),
        "n_test_dates": int(n_test_dates),
        "top_n": int(top_n),
        "cost_bps_side": float(cost_bps_side),
        "gross": gross,
        "net": net,
        "excess": excess,
        "deflated_sharpe_net": round(float(dsr_net), 4),
        "net_iid": net_iid,
        "excess_iid": excess_iid,
        "deflated_sharpe_net_iid": round(float(dsr_net_iid), 4),
        "per_fold_net_sharpe": [round(float(x), 4) for x in per_fold_net_sharpe],
        "avg_daily_turnover": round(1.0 / horizon, 4),
        "cost_bps_per_day": round(cost_daily * 1e4, 4),
        "notes": notes,
        "per_date": per_date,
    }


__all__ = ["walkforward_portfolio_backtest"]
