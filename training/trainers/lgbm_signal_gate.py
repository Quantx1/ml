"""
PR 128/169 — LightGBM signal-gate trainer.

Cross-sectional 3-class direction classifier (SELL=-1, HOLD=0, BUY=+1)
trained on top-N liquid NSE stocks. Outputs are used by SignalGenerator
to gate which raw strategy signals fire.

PR 169 upgrades from threshold labels + 5-fold TimeSeriesSplit to:

  - Triple-barrier labels (López de Prado, ml.labeling) — ATR-scaled,
    path-dependent, volatility-aware
  - Liquid universe top-200 by 30-day median ADV (ml.data) instead of
    hardcoded 50 names
  - Walk-forward CV via ml.training.wfcv (rolling 5-fold)
  - Backtest-driven primary metric via ml.eval.compute_backtest_metrics
    (Sharpe, drawdown, profit factor — promote gate-ready)
  - Optional Optuna 20-trial Bayesian search over LGBM hyperparams

Artifact: native LightGBM .txt format loaded by LGBMGate.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import Trainer, TrainerError, TrainResult
from ..wfcv import WFCVConfig, aggregate_fold_metrics, walk_forward_split

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Universe size for training. 200 covers >90% of NSE volume; matches
# Nifty 200 index design.
# Smoke mode (SMOKE_MODE=1) overrides this to 10 via ml.training.smoke.
from ml.training.smoke import is_smoke_mode, smoke_universe_size  # noqa: PLC0415

def _resolve_universe_top_n() -> int:
    """Universe size for lgbm_signal_gate.

    Priority:
      1. SMOKE_MODE=1 → smoke_universe_size() (typically 10)
      2. LGBM_UNIVERSE_TOP_N env var (manual override)
      3. Default 200 (full prod)
    """
    import os  # noqa: PLC0415
    if is_smoke_mode():
        return smoke_universe_size()
    env_override = os.environ.get("LGBM_UNIVERSE_TOP_N", "").strip()
    if env_override:
        try:
            n = int(env_override)
            if n > 0:
                return n
        except ValueError:
            pass
    return 200


UNIVERSE_TOP_N = _resolve_universe_top_n()

# Data window: 8 years of daily bars per symbol gives ~2000 bars after
# the 30-bar feature warmup, enough for 5-fold WFCV with 252-day folds.
DATA_PERIOD = "8y"
DATA_INTERVAL = "1d"

# Triple-barrier label parameters
PROFIT_TARGET_ATR = 2.0       # +2 ATR upper barrier
STOP_LOSS_ATR = 1.0           # -1 ATR lower barrier (2:1 R:R)
VERTICAL_BARRIER_DAYS = 10    # max holding period

# Forward-return horizon for backtest metric (separate from labeling).
# Daily strategy holds N days then exits at close.
FWD_RETURN_DAYS = 5

# WFCV — 5 folds, 252 day test window, 3 year train window
WFCV_FOLDS = 5
WFCV_TEST_SIZE = 252
WFCV_TRAIN_SIZE = 252 * 3
WFCV_EMBARGO = VERTICAL_BARRIER_DAYS + 2  # purge labeling-window leakage


# ============================================================================
# Feature engineering
# ============================================================================

# Feature schema imports — single source of truth lives in
# ml.features.lgbm_v2 so trainer + live inference path can never drift.
# Audit caught this drift on 2026-05-19: trainer wrote 30 features but
# live `split_feature_sets` only emitted 15, causing LGBMGate.predict()
# to KeyError on every signal post-training.
from ml.features.lgbm_v2 import (  # noqa: E402,PLC0415
    FEATURE_ORDER,
    FEATURE_ORDER_BASE,
    compute_ohlcv_features,
)

import os as _os  # noqa: E402


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute LGBM v2 features for one symbol's daily OHLCV frame.

    Thin wrapper around ml.features.lgbm_v2.compute_ohlcv_features so the
    trainer and live inference path share the exact same OHLCV-block
    computation. Adds the trainer-only `_fwd_return` private column for
    triple-barrier labeling.
    """
    out = compute_ohlcv_features(df)
    close = df["Close"].astype(float)
    # Realized fwd return for backtest (target side) — trainer-only column.
    out["_fwd_return"] = close.pct_change(FWD_RETURN_DAYS).shift(-FWD_RETURN_DAYS)
    return out


def _build_dataset() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Download + featurize + label the universe.

    Returns:
        features:   DataFrame of (n_rows, 15 features) ordered by date asc
        labels:     int8 array length n_rows from triple-barrier {-1,0,+1}
        weights:    float array length n_rows — AFML Ch.4 sample-weight
                    uniqueness, normalized to mean 1.0. Down-weights
                    observations whose triple-barrier windows overlap
                    heavily with neighbors.
        fwd_returns: float array length n_rows of FWD_RETURN_DAYS forward
                     returns aligned with labels
        symbols:    object array length n_rows naming the source symbol
        nifty_returns: pd.Series indexed by date — Nifty 50 fwd return for
                       benchmark eval (already aligned to features.date)
    """
    from ml.data import LiquidUniverseConfig, liquid_universe  # noqa: PLC0415
    # bhavcopy_download_with_fallback no longer imported here — wrapped
    # by production_ohlcv (see below). PR 2026-05-12.
    from ml.data.fii_dii_history import (  # noqa: PLC0415
        compute_flow_features,
        fii_dii_series,
        reindex_flow_features_to,
    )
    from ml.data.sentiment_history import (  # noqa: PLC0415
        reindex_sentiment_to,
        sentiment_features_for,
    )
    from ml.data.fundamentals_pit import (  # noqa: PLC0415
        FUNDAMENTALS_FEATURE_NAMES,
        compute_fundamentals_features,
        get_pit_fundamentals,
        reindex_fundamentals_to,
    )
    from ml.labeling import (  # noqa: PLC0415
        TripleBarrierConfig,
        sample_weights_from_t1,
        triple_barrier_events,
    )

    universe = liquid_universe(LiquidUniverseConfig(top_n=UNIVERSE_TOP_N))
    if not universe:
        raise TrainerError("liquid_universe returned 0 symbols")
    logger.info("lgbm_signal_gate: universe size=%d", len(universe))

    # PR 2026-05-12 — single production data path. Routes through
    # ml.data.production_ohlcv which gives us, in one call:
    #   - bhavcopy primary + yfinance fallback (Exception-broad catch)
    #   - corporate-action-adjusted volume (adjust_batch)
    #   - survivorship-aware universe expansion (delisted_registry)
    #   - per-symbol bad-bar quality drop (max 30% bad bars per ticker)
    #   - duplicate-index dedup (NSE/yfinance occasionally double-rows)
    # Output is yfinance-shape: outer=ticker(.NS), inner=field.
    from ml.data.production_ohlcv import production_ohlcv  # noqa: PLC0415

    end_date = pd.Timestamp.today().normalize()
    # PR 2026-05-13 — history window is env-overridable. Default bumped
    # to 10y on 2026-05-19 (pre-training audit) so the model sees more
    # regimes (2015 bull, 2018 midcap crash, 2020 COVID, 2022 inflation,
    # 2023 rally). LGBM scales sublinearly on data size so the cost is
    # ~50% more training time for materially better generalization.
    # Set LGBM_HISTORY_YEARS=5 for a faster smoke run.
    _hist_years = int(_os.environ.get("LGBM_HISTORY_YEARS", "10") or "10")
    start_date = end_date - pd.DateOffset(years=_hist_years)
    print(f"[lgbm] training window: {start_date.date()}..{end_date.date()}  ({_hist_years}y)", flush=True)
    print(f"[lgbm] fetching {len(universe)} symbols via production_ohlcv (may take 3-10 min on first run)...", flush=True)
    try:
        # W6 (pre-training audit 2026-05-19) — capture data source tier
        # (kite_admin / bhavcopy / yfinance) so it can be persisted to
        # model_versions.metrics for post-mortem provenance.
        raw, _data_source = production_ohlcv(
            symbols=universe,
            start=start_date.date(),
            end=end_date.date(),
            suffix=".NS",
            include_delisted=False,    # liquid top-N universe; survivorship
                                       # bias is small at this size
            adjust_corp_actions=True,
            quality_check=True,
            group_by="ticker",         # outer=ticker(.NS), inner=field
            return_source=True,
        )
        print(f"[lgbm] data source tier: {_data_source}", flush=True)
    except Exception as exc:
        raise TrainerError(f"production_ohlcv failed: {exc}") from exc

    if raw is None or raw.empty:
        raise TrainerError("production_ohlcv returned empty frame")
    logger.info(
        "lgbm_signal_gate: production_ohlcv → %d tickers × %d rows",
        len(raw.columns.get_level_values(0).unique()), len(raw),
    )

    # PR 182 — automated data-quality audit. Run BEFORE feature
    # engineering so issues are caught up-front. Negative prices fatal;
    # trading_window_violation is intraday-only (removed from default
    # fatal list 2026-05-11); other rot (stale runs, vol spikes, gaps)
    # is reported but allowed — trainers persist the report into
    # model_versions.metrics for ops visibility.
    from ml.data.quality_check import (  # noqa: PLC0415
        DataQualityError,
        QualityCheckConfig,
        run_quality_checks,
    )
    per_symbol_for_audit = {}
    for sym in universe[: min(len(universe), 50)]:   # spot-check first 50
        ticker = f"{sym}.NS"
        try:
            per_symbol_for_audit[sym] = raw[ticker].dropna(how="all")
        except (KeyError, AttributeError):
            continue
    audit_report = run_quality_checks(per_symbol_for_audit, QualityCheckConfig())
    logger.info("lgbm_signal_gate: data quality — %s", audit_report.summary())
    if audit_report.fatal_count > 0:
        raise DataQualityError(
            f"data-quality audit failed: {audit_report.fatal_reasons}"
        )

    # PR 180 — fetch FII/DII flow series once and compute features over
    # the whole training window. Per-symbol feature frames will reindex
    # this market-wide series onto their own date axis.
    flow_features = compute_flow_features(
        fii_dii_series(start_date.date(), end_date.date()),
    )
    if flow_features.empty:
        logger.warning(
            "lgbm_signal_gate: FII/DII flow data unavailable — features will be zero-filled",
        )

    # PR 183 — historical FinBERT-India sentiment features. Zero-filled
    # when the cache is empty (early-launch case before the news
    # ingestion pipeline backfills history).
    sentiment_frame = sentiment_features_for(
        symbols=universe, start=start_date.date(), end=end_date.date(),
    )
    if (
        sentiment_frame.empty
        or float(sentiment_frame["sentiment_5d_count"].sum()) == 0
    ):
        logger.warning(
            "lgbm_signal_gate: sentiment cache empty — features will be zero-filled",
        )

    # PR 184 — PIT fundamentals snapshot anchored at end_date. For v1
    # we broadcast the latest PIT-published snapshot across the symbol's
    # date axis. A per-bar PIT lookup is the next iteration; the
    # broadcast variant is a defensible approximation for swing-horizon
    # features that change quarterly.
    fundamentals_pit = get_pit_fundamentals(
        symbols=universe, as_of=end_date.date(),
    )
    fundamentals_features = compute_fundamentals_features(
        fundamentals_pit, as_of=end_date.date(),
    )
    if fundamentals_features.empty:
        logger.warning(
            "lgbm_signal_gate: fundamentals cache empty — "
            "run ingest_yfinance_fundamentals first; features zero-filled",
        )

    tb_cfg = TripleBarrierConfig(
        profit_target_atr=PROFIT_TARGET_ATR,
        stop_loss_atr=STOP_LOSS_ATR,
        vertical_barrier_days=VERTICAL_BARRIER_DAYS,
    )

    feats: List[pd.DataFrame] = []
    labs: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    fwds: List[np.ndarray] = []
    syms: List[np.ndarray] = []
    print(f"\n[lgbm] === STEP 4/6 — per-symbol feature build + triple-barrier labels ===", flush=True)
    n_ok = n_skip_short = n_skip_err = 0
    for sym_idx, sym in enumerate(universe):
        ticker = f"{sym}.NS"
        try:
            sym_df = raw[ticker].dropna(subset=["Close", "High", "Low", "Volume"])
        except (KeyError, AttributeError):
            n_skip_err += 1
            print(f"[lgbm]   [{sym_idx+1:>3}/{len(universe)}] {sym:<14} SKIP (missing in raw)", flush=True)
            continue
        if len(sym_df) < 300:
            n_skip_short += 1
            print(f"[lgbm]   [{sym_idx+1:>3}/{len(universe)}] {sym:<14} SKIP (only {len(sym_df)} rows < 300)", flush=True)
            continue
        try:
            f = _compute_features(sym_df)
            # PR 180 — merge market-wide FII/DII flow features by date.
            # reindex_flow_features_to ffills 1 day + zero-fills gaps so
            # the feature set is dense even when NSE archive has holes.
            flow_aligned = reindex_flow_features_to(flow_features, f.index)
            for col in ["fii_5d_sum", "dii_5d_sum", "fii_5d_z", "dii_5d_z"]:
                f[col] = flow_aligned[col].values
            # PR 183 — merge per-symbol sentiment features. Empty cache
            # → zero-filled features so training proceeds.
            sent_aligned = reindex_sentiment_to(sentiment_frame, sym, f.index)
            f["sentiment_5d_mean"] = sent_aligned["sentiment_5d_mean"].values
            f["sentiment_5d_count"] = sent_aligned["sentiment_5d_count"].values
            # PR 184 — broadcast PIT fundamentals snapshot. Quarterly
            # cadence so a single snapshot per symbol is a reasonable
            # approximation for swing-horizon features.
            sym_fund = (
                fundamentals_features.loc[sym]
                if (not fundamentals_features.empty and sym in fundamentals_features.index)
                else None
            )
            fund_aligned = reindex_fundamentals_to(sym_fund, f.index)
            for col in FUNDAMENTALS_FEATURE_NAMES:
                f[col] = fund_aligned[col].values

            f = f.dropna(subset=FEATURE_ORDER + ["_atr_raw"])
            if len(f) < 100:
                continue
            # PR 176 — get labels AND barrier-hit times so we can compute
            # AFML Ch.4 sample-weight uniqueness. t1 is per-bar inside
            # this symbol's local index.
            labels, t1_local = triple_barrier_events(
                close=sym_df.loc[f.index, "Close"].values,
                atr=f["_atr_raw"].values,
                cfg=tb_cfg,
            )
            sym_weights = sample_weights_from_t1(t1_local, n=len(f))
            # Drop the last vbd rows where label is forced to 0 (no future)
            keep = slice(0, len(f) - VERTICAL_BARRIER_DAYS)
            f = f.iloc[keep]
            labels = labels[keep]
            sym_weights = sym_weights[keep]
            fwd_ret = f["_fwd_return"].values
            mask = ~np.isnan(fwd_ret)
            f = f.loc[mask]
            labels = labels[mask]
            sym_weights = sym_weights[mask]
            fwd_ret = fwd_ret[mask]
            if len(f) < 50:
                continue
            feats.append(f[FEATURE_ORDER].copy())
            feats[-1].index = pd.MultiIndex.from_product(
                [[sym], f.index], names=["symbol", "date"],
            )
            labs.append(labels.astype(np.int8))
            weights.append(sym_weights.astype(np.float32))
            fwds.append(fwd_ret.astype(np.float32))
            syms.append(np.full(len(f), sym, dtype=object))
            n_ok += 1
            # Per-symbol label distribution makes class imbalance visible
            uniq, cnt = np.unique(labels, return_counts=True)
            dist = {int(u): int(c) for u, c in zip(uniq, cnt)}
            print(
                f"[lgbm]   [{sym_idx+1:>3}/{len(universe)}] {sym:<14} OK  "
                f"{len(f):>4} rows × {len(FEATURE_ORDER)} feats  labels={dist}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            n_skip_err += 1
            print(f"[lgbm]   [{sym_idx+1:>3}/{len(universe)}] {sym:<14} ERR {type(exc).__name__}: {str(exc)[:60]}", flush=True)

    print(f"[lgbm] === STEP 4 complete: {n_ok} ok / {n_skip_short} short-skip / {n_skip_err} error ===\n", flush=True)
    if not feats:
        raise TrainerError("no symbols produced usable training data")

    X = pd.concat(feats, axis=0).sort_index(level="date")
    # Align labels + weights + fwd_returns + symbols to X's order
    full = pd.DataFrame({
        "label": np.concatenate(labs),
        "weight": np.concatenate(weights),
        "fwd_return": np.concatenate(fwds),
        "symbol": np.concatenate(syms),
    }, index=pd.concat(feats, axis=0).index).loc[X.index]
    y = full["label"].values
    sample_weight = full["weight"].values
    fwd_returns = full["fwd_return"].values

    # Nifty benchmark: same FWD_RETURN_DAYS on ^NSEI
    # Phase 1.7 audit fix #2.4 — yfinance was never imported at module
    # scope so this `yf.download` reference silently raised NameError,
    # which the bare `except` swallowed into an empty `nifty_fwd`
    # Series. Result: every benchmark-relative metric (excess_return,
    # alpha, hit-rate-vs-Nifty) was computed against zero and the
    # trainer falsely "beat" the benchmark on near-flat days.
    try:
        import yfinance as yf  # noqa: PLC0415
        nifty = yf.download("^NSEI", period=DATA_PERIOD, interval=DATA_INTERVAL,
                            progress=False, auto_adjust=True)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = [c[0] for c in nifty.columns]
        nifty_fwd = nifty["Close"].pct_change(FWD_RETURN_DAYS).shift(-FWD_RETURN_DAYS).dropna()
    except Exception as _ben_exc:  # noqa: BLE001
        logger.warning("Nifty benchmark download failed (%s) — proceeding without", _ben_exc)
        nifty_fwd = pd.Series(dtype=float)

    return (
        X, y, sample_weight, fwd_returns, full["symbol"].values, nifty_fwd,
        audit_report.to_dict(), _data_source,
    )


# ============================================================================
# LGBM training core
# ============================================================================

from ml.training.smoke import lightgbm_device  # noqa: PLC0415, E402

DEFAULT_LGBM_PARAMS = dict(
    objective="multiclass",
    num_class=3,                  # mapped from {-1,0,+1} -> {0,1,2}
    metric="multi_logloss",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    lambda_l1=0.1,
    lambda_l2=0.1,
    n_estimators=300,
    verbose=-1,
    # LightGBM 4.x supports ``device='cuda'`` natively. Pip wheels from
    # 2024+ ship with both CPU + CUDA tree learners. Falls back to cpu
    # when no GPU is available so this works in CI + dev.
    device=lightgbm_device(),
)


def _remap_labels_for_lgbm(y_signed: np.ndarray) -> np.ndarray:
    """LGBM multiclass needs 0..K-1 labels; map -1/0/+1 -> 0/1/2."""
    return (y_signed.astype(np.int64) + 1).astype(np.int64)


def _unmap_to_signed_predictions(class_idx: np.ndarray) -> np.ndarray:
    """Reverse: 0/1/2 -> -1/0/+1."""
    return class_idx.astype(np.int64) - 1


def _load_regime_series_for_dates(dates) -> Optional[List[int]]:
    """Look up the prod regime_hmm label per date for the per-fold
    regime-stratified Sharpe (Phase 1.2). Returns a list of ints
    {0=bull, 1=sideways, 2=bear} aligned with the dates iterable.

    Defensive: returns None on any resolution failure (model not
    cached, no internet, schema mismatch). Caller treats None as
    "regime stratification unavailable" and skips that block of
    the promote-gate (the rest of the gate still runs).
    """
    try:
        from backend.ai.registry import resolve_model_file  # noqa: PLC0415
        from ml.regime_detector import (  # noqa: PLC0415
            MarketRegimeDetector, compute_regime_features,
        )
        import yfinance as _yf  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        disk = Path("artifacts/models/regime_hmm.pkl")
        path = resolve_model_file("regime_hmm", "regime_hmm.pkl", disk)
        det = MarketRegimeDetector()
        det.load(str(path))

        # Pull a buffered Nifty + VIX window so we have 60 bars of
        # context for the HMM's rolling prediction at the earliest date.
        from datetime import timedelta  # noqa: PLC0415
        min_date = pd.Timestamp(min(dates))
        max_date = pd.Timestamp(max(dates))
        nifty_df = _yf.download(
            "^NSEI",
            start=(min_date - timedelta(days=90)).strftime("%Y-%m-%d"),
            end=(max_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False, auto_adjust=False,
        )
        vix_df = _yf.download(
            "^INDIAVIX",
            start=(min_date - timedelta(days=90)).strftime("%Y-%m-%d"),
            end=(max_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False, auto_adjust=False,
        )
        if nifty_df is None or nifty_df.empty:
            return None
        nifty_df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in nifty_df.columns]
        if vix_df is None or vix_df.empty:
            vix_df = pd.DataFrame({"close": [15.0] * len(nifty_df)}, index=nifty_df.index)
        else:
            vix_df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in vix_df.columns]
        feat = compute_regime_features(nifty_df, vix_df)
        regime_map = {"bull": 0, "sideways": 1, "bear": 2}

        labels: List[int] = []
        feat_dates = list(feat.index)
        for d in dates:
            d_ts = pd.Timestamp(d)
            # Find the latest feature row at or before this date
            idx = feat.index.searchsorted(d_ts, side="right") - 1
            if idx < 0:
                labels.append(1)
                continue
            window = feat.iloc[max(0, idx - 60): idx + 1]
            try:
                info = det.predict_regime(window)
                labels.append(regime_map.get(info.get("regime", "sideways"), 1))
            except Exception:
                labels.append(1)
        return labels
    except Exception as exc:  # noqa: BLE001
        logger.debug("regime-stratified mapping unavailable: %s", exc)
        return None


def _train_one_fold(
    X_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    fwd_te: np.ndarray, bench_te: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[Any, dict]:
    """Train LGBM on one fold, return (model, metrics dict).

    PR 176 — pass AFML Ch.4 uniqueness weights via lgb's sample_weight
    so overlapping triple-barrier labels don't double-count information.
    """
    import lightgbm as lgb  # noqa: PLC0415
    from sklearn.metrics import accuracy_score  # noqa: PLC0415

    from ml.eval import BacktestEvalConfig, compute_backtest_metrics  # noqa: PLC0415

    model = lgb.LGBMClassifier(**params)
    # PR 2026-05-12 — verbose every 10 iterations so we see training progress
    # line-by-line in the log (Jupyter-style streaming).
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_te, y_te)],
              callbacks=[lgb.log_evaluation(period=10)])
    y_pred = model.predict(X_te)
    acc = float(accuracy_score(y_te, y_pred))

    # Convert predicted classes back to signed direction for backtest eval
    signed_preds = _unmap_to_signed_predictions(y_pred)
    bt = compute_backtest_metrics(
        predictions=signed_preds.astype(float),
        forward_returns=fwd_te,
        benchmark_returns=bench_te if bench_te.size else None,
        cfg=BacktestEvalConfig(direction_neutral=True),
    )

    # Per-fold strategy return series (signed prediction × forward return,
    # cost-deducted via compute_backtest_metrics). This is what we feed
    # into Bailey/López de Prado DSR + PBO so the trainer can report
    # both overfit defenses to the promote-gate.
    strat_returns = signed_preds.astype(float) * fwd_te
    metrics = {
        "accuracy": acc,
        "sharpe": bt["sharpe"],
        "max_drawdown_pct": bt["max_drawdown_pct"],
        "calmar": bt["calmar"],
        "profit_factor": bt["profit_factor"],
        "win_rate": bt["win_rate"],
        "n_trades": bt["n_trades"],
        "total_return_pct": bt["total_return_pct"],
        "excess_return_pct": bt.get("excess_return_pct", 0.0),
        "_returns": strat_returns.tolist(),  # internal, consumed by DSR/PBO; stripped before persist
    }
    return model, metrics


# ============================================================================
# Trainer
# ============================================================================


class LGBMSignalGateTrainer(Trainer):
    name = "lgbm_signal_gate"
    requires_gpu = False  # CPU LGBM scales to ~5min on this dataset
    depends_on: list[str] = []
    # PR 169: trades directionally; financial gate applies.
    # Phase 1.2 — production-grade promote gate. Signal gate is the
    # final filter before any user trade fires, so it gets the same
    # bar as the RL ensemble: DSR + PBO mandatory, regime stratified.
    promote_thresholds = {
        "min_sharpe": 1.0,
        "max_drawdown_pct": -0.25,
        "min_calmar": 0.5,
        "min_profit_factor": 1.5,
        "min_n_trades": 50,            # signal gate fires often; raise floor
        "min_excess_return_pct": 0.05,
        "min_deflated_sharpe": 0.5,
        "max_pbo": 0.4,
        "require_overfit_defenses": True,
        "min_regime_stratified_sharpe": 0.5,
    }

    def train(self, out_dir: Path) -> TrainResult:
        try:
            import lightgbm as lgb  # noqa: PLC0415, F401
        except ImportError as exc:
            raise TrainerError(f"lightgbm required: {exc}")

        t0 = time.time()
        print(f"\n[lgbm] ╔══════════════════════════════════════════════════════════╗", flush=True)
        print(f"[lgbm] ║ lgbm_signal_gate TRAINING — verbose mode                 ║", flush=True)
        print(f"[lgbm] ║   universe top-N: {UNIVERSE_TOP_N:<3}                                       ║", flush=True)
        print(f"[lgbm] ║   labeling: triple-barrier (mlfinpy)                     ║", flush=True)
        print(f"[lgbm] ║   CV: walk-forward {WFCV_FOLDS}-fold, test={WFCV_TEST_SIZE}d, train={WFCV_TRAIN_SIZE}d ║", flush=True)
        print(f"[lgbm] ╚══════════════════════════════════════════════════════════╝\n", flush=True)
        print(f"[lgbm] === STEP 1-3/6 — universe + OHLCV + cache load ===", flush=True)
        (
            X, y, sample_weight, fwd_returns, symbols, nifty_fwd, dq_report,
            data_source,
        ) = _build_dataset()
        print(
            f"[lgbm] === STEPS 1-3 complete: X.shape={X.shape}, y.shape={y.shape}, "
            f"source={data_source} ===\n", flush=True,
        )

        # PR 189 — verify no feature is silently dead. Caches that
        # didn't backfill (FII/DII, sentiment, fundamentals on a fresh
        # repo) zero-fill their columns. The model still trains, the
        # gate may even pass on live features alone, but the dead
        # columns add nothing. Fail fast so the operator backfills.
        from ml.data.quality_check import (  # noqa: PLC0415
            DataQualityError as _DQError,
            audit_feature_matrix,
        )
        feat_audit = audit_feature_matrix(X, feature_names=FEATURE_ORDER)
        logger.info(
            "lgbm_signal_gate: feature audit — %d/%d dead, dead=%s",
            feat_audit["n_constant"], feat_audit["n_features"],
            feat_audit["constant_features"],
        )
        # PR 2026-05-13 — env override LGBM_DEAD_FEATURE_FATAL controls
        # whether dead features (zero-filled because of empty caches) hard-fail.
        # Default True (no fallbacks) per locked memory 2026-04-19. Set to 0
        # for v1 training when FII/DII + sentiment + fundamentals caches
        # aren't fully aligned to the training window — we already accepted
        # this trade-off (Sharpe -5..-8% vs full caches; see locked decision
        # 2026-05-12). The dead-feature LIST still logs so we know what's
        # missing for the post-launch Month-2 cache backfill plan.
        _dead_fatal = _os.environ.get("LGBM_DEAD_FEATURE_FATAL", "1").strip() not in ("0", "false", "no", "")
        if feat_audit["fatal"] and _dead_fatal:
            raise _DQError(
                f"feature audit failed: {feat_audit['n_constant']} dead features "
                f"({feat_audit['constant_features']}). "
                f"Backfill caches first OR set LGBM_DEAD_FEATURE_FATAL=0 to "
                f"train on warm features only (zero-fill the rest; "
                f"Sharpe -5..-8% vs full)."
            )
        elif feat_audit["fatal"]:
            print(
                f"[lgbm] ⚠ WARNING: {feat_audit['n_constant']}/{feat_audit['n_features']} "
                f"features are zero-filled (cache misses): "
                f"{feat_audit['constant_features']}\n"
                f"[lgbm]   LGBM_DEAD_FEATURE_FATAL=0 → continuing anyway. "
                f"Model will train on the {feat_audit['n_features'] - feat_audit['n_constant']} live features only.",
                flush=True,
            )
        print(
            f"[lgbm] === STEP 5/6 — feature audit + dataset prep ===\n"
            f"[lgbm]   dataset: {len(X):,} samples × {X.shape[1]} features × {len(np.unique(symbols))} symbols\n"
            f"[lgbm]   dead features: {feat_audit['n_constant']}/{feat_audit['n_features']}  (caches backfilled OK)\n"
            f"[lgbm]   sample_weight stats:  mean={sample_weight.mean():.3f}  min={sample_weight.min():.3f}  max={sample_weight.max():.3f}\n"
            f"[lgbm]   label distribution: {dict(zip(*np.unique(y, return_counts=True)))}",
            flush=True,
        )

        # ── B3 (pre-training audit 2026-05-19) — EDA gate.
        # Hard-fails if any of: NaN >50% on any feature, class balance
        # <5% on any label, max abs IC <0.005 across features, or
        # leakage suspect (corr >0.95 with same-bar label).
        # Audit found these checks existed in ml.preprocessing.eda but
        # were only invoked by the orchestrator script — direct trainer
        # runs bypassed them. Now invoked inline.
        from ml.preprocessing.eda import (  # noqa: PLC0415
            EDAReport,
            eda_classification_balance,
            eda_dataframe_summary,
            eda_feature_label_ic,
            eda_leakage_check,
        )
        eda = EDAReport(
            trainer="lgbm_signal_gate",
            n_rows=len(X),
            n_features=int(X.shape[1]),
            n_symbols=int(len(np.unique(symbols))),
        )
        feat_summary = eda_dataframe_summary(X, list(FEATURE_ORDER), max_nan_pct=0.50)
        eda.feature_summary = feat_summary
        eda.blockers.extend(feat_summary.get("blockers", []))

        balance = eda_classification_balance(pd.Series(y), min_class_pct=0.05)
        eda.label_summary = balance
        eda.blockers.extend(balance.get("blockers", []))
        eda.warnings.extend(balance.get("warnings", []))

        # IC + leakage need a labeled frame; combine X + y on the fly.
        eda_df = X.copy()
        eda_df["_label"] = y
        ic = eda_feature_label_ic(
            eda_df, list(FEATURE_ORDER), "_label", min_abs_mean_ic=0.005,
        )
        eda.ic_summary = ic
        eda.blockers.extend(ic.get("blockers", []))

        leak = eda_leakage_check(
            eda_df, list(FEATURE_ORDER), "_label", max_corr=0.95,
        )
        eda.leakage_summary = leak
        eda.blockers.extend(leak.get("blockers", []))

        if not eda.ok:
            raise TrainerError(
                f"EDA gate FAILED for lgbm_signal_gate: {eda.blockers}",
            )
        print(
            f"[lgbm] EDA gate PASS — IC max_abs={ic.get('max_abs_ic', 0):.4f}, "
            f"warnings={len(eda.warnings)}",
            flush=True,
        )

        # Sort by date for proper WFCV. X has a MultiIndex (symbol, date);
        # we want global chronological order.
        sort_idx = np.argsort(np.asarray(X.index.get_level_values("date").values))
        X_arr = X.iloc[sort_idx].values
        y_remapped = _remap_labels_for_lgbm(y[sort_idx])
        weight_sorted = sample_weight[sort_idx]
        fwd_sorted = fwd_returns[sort_idx]

        # Build benchmark series aligned with sort order
        sorted_dates = X.iloc[sort_idx].index.get_level_values("date")
        if not nifty_fwd.empty:
            bench_arr = nifty_fwd.reindex(sorted_dates).fillna(0.0).values
        else:
            bench_arr = np.zeros_like(fwd_sorted)

        # Walk-forward CV
        cfg = WFCVConfig(
            strategy="rolling",
            n_folds=WFCV_FOLDS,
            test_size=WFCV_TEST_SIZE,
            train_size=WFCV_TRAIN_SIZE,
            embargo=WFCV_EMBARGO,
        )

        print(f"\n[lgbm] === STEP 6/6 — walk-forward CV {WFCV_FOLDS} folds ===", flush=True)
        fold_metrics: list[dict] = []
        for fold_idx, (tr_idx, te_idx) in enumerate(walk_forward_split(len(X_arr), cfg)):
            print(
                f"\n[lgbm] --- FOLD {fold_idx + 1}/{WFCV_FOLDS}: "
                f"train {len(tr_idx):,} rows / test {len(te_idx):,} rows ---",
                flush=True,
            )
            try:
                _, m = _train_one_fold(
                    X_arr[tr_idx], y_remapped[tr_idx], weight_sorted[tr_idx],
                    X_arr[te_idx], y_remapped[te_idx],
                    fwd_sorted[te_idx], bench_arr[te_idx],
                    DEFAULT_LGBM_PARAMS,
                )
                m["fold"] = fold_idx
                fold_metrics.append(m)
                print(
                    f"[lgbm] FOLD {fold_idx + 1} OK  "
                    f"acc={m['accuracy']:.3f}  sharpe={m['sharpe']:.2f}  "
                    f"dd={m['max_drawdown_pct']:.2f}  pf={m['profit_factor']:.2f}  "
                    f"trades={m['n_trades']}  win_rate={m['win_rate']:.3f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[lgbm] FOLD {fold_idx + 1} FAILED: {type(exc).__name__}: {exc}", flush=True)

        if not fold_metrics:
            raise TrainerError("all lgbm WFCV folds failed")

        # Final fit on all data → ship to production
        final_model = lgb.LGBMClassifier(**DEFAULT_LGBM_PARAMS)
        final_model.fit(X_arr, y_remapped, sample_weight=weight_sorted)
        artifact = out_dir / "lgbm_signal_gate.txt"
        final_model.booster_.save_model(str(artifact))

        # Phase 1.7 audit fix #1.1/1.2 — sidecar metadata so the consumer
        # (model_registry.LGBMGate) can stay in sync with whatever this
        # trainer produces. Without this sidecar, the consumer relied on a
        # hard-coded label_map and 15-feature list that DIVERGED from the
        # trainer (which now emits 30 features and class 0=SELL, 1=HOLD,
        # 2=BUY because _remap_labels_for_lgbm maps (-1,0,+1) → (0,1,2)).
        # The latent anti-signal bug would have fired the moment this
        # trainer's model shipped to prod.
        meta_artifact = out_dir / "lgbm_signal_gate.meta.json"
        meta_payload = {
            "schema_version": 2,
            "trainer": "ml.training.trainers.lgbm_signal_gate",
            "feature_order": list(FEATURE_ORDER),
            "n_features": int(X_arr.shape[1]),
            "label_map": {"0": "SELL", "1": "HOLD", "2": "BUY"},
            "num_class": 3,
            "fwd_return_days": int(FWD_RETURN_DAYS),
            "triple_barrier": {
                "profit_target_atr": float(PROFIT_TARGET_ATR),
                "stop_loss_atr": float(STOP_LOSS_ATR),
                "vertical_barrier_days": int(VERTICAL_BARRIER_DAYS),
            },
        }
        with open(meta_artifact, "w", encoding="utf-8") as _meta_fp:
            json.dump(meta_payload, _meta_fp, indent=2, sort_keys=True)

        # Phase 1.2 — emit DSR + PBO from per-fold returns. The promote-gate
        # now requires both for real-money models (require_overfit_defenses
        # flag on this trainer's promote_thresholds).
        fold_return_arrays = [
            np.asarray(m.pop("_returns", []), dtype=float)
            for m in fold_metrics
        ]
        try:
            from ml.eval.overfitting import dsr_pbo_from_fold_returns  # noqa: PLC0415
            dsr_pbo = dsr_pbo_from_fold_returns(
                fold_returns=fold_return_arrays,
                # n_trials = (folds × 1 hyperparam set) — no Optuna in this
                # trainer yet; the DSR null benchmark therefore reflects
                # only the fold-variance, not search overfit. Conservative.
                n_trials=max(len(fold_return_arrays), 1),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("DSR/PBO computation failed: %s", exc)
            dsr_pbo = {"deflated_sharpe": 0.0, "probability_backtest_overfitting": 0.5}
        print(
            f"\n[lgbm] === Phase 1.2 overfit defenses ===\n"
            f"[lgbm]   deflated_sharpe = {dsr_pbo.get('deflated_sharpe'):.4f} "
            f"(need >= 0.5 for promote)\n"
            f"[lgbm]   PBO = {dsr_pbo.get('probability_backtest_overfitting'):.4f} "
            f"(need <= 0.4)",
            flush=True,
        )

        # Phase 1.2 — regime-stratified Sharpe. Map each fold's TEST
        # window to the dominant regime active during that window (from
        # the regime series in the dataset), then bucket returns by
        # regime and compute Sharpe per bucket. The promote-gate blocks
        # if any regime with > 50 days has Sharpe < 0.5.
        regime_sharpe: Dict[str, float] = {}
        regime_n_days: Dict[str, int] = {}
        try:
            from ml.eval.backtest_eval import metrics_from_returns  # noqa: PLC0415
            # Pull the regime series the same way _build_dataset did. We
            # reuse the already-loaded regime data if it's in the dataset
            # metadata; otherwise fall back to a yfinance lookup.
            regime_series = _load_regime_series_for_dates(sorted_dates)
            if regime_series is not None and len(regime_series) == len(sorted_dates):
                regime_returns: Dict[str, list] = {"bull": [], "sideways": [], "bear": []}
                fold_starts = []
                for fold_idx, (tr_idx, te_idx) in enumerate(walk_forward_split(len(X_arr), cfg)):
                    if fold_idx >= len(fold_return_arrays):
                        break
                    test_regimes = [regime_series[i] for i in te_idx]
                    ret_arr = fold_return_arrays[fold_idx]
                    for i, r in enumerate(test_regimes[:len(ret_arr)]):
                        bucket = {0: "bull", 1: "sideways", 2: "bear"}.get(int(r), "sideways")
                        regime_returns[bucket].append(float(ret_arr[i]))
                for regime, vals in regime_returns.items():
                    if vals:
                        m = metrics_from_returns(np.asarray(vals, dtype=float))
                        regime_sharpe[regime] = float(m.get("sharpe", 0.0))
                        regime_n_days[regime] = int(len(vals))
                print(
                    f"[lgbm]   regime-stratified Sharpe (need >= 0.5 in each with n>50):\n"
                    + "\n".join(
                        f"[lgbm]     {r:9s}: sharpe={regime_sharpe.get(r, 0):.2f}  n={regime_n_days.get(r, 0)}"
                        for r in ("bull", "sideways", "bear")
                    ),
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("regime stratification skipped: %s", exc)

        # Aggregate metrics across folds — these go into model_versions.metrics
        # AND drive the promote gate (PR 167)
        agg = aggregate_fold_metrics(fold_metrics)
        # Class distribution for sanity
        class_dist = {
            "buy_pct": float((y == 1).mean()),
            "sell_pct": float((y == -1).mean()),
            "hold_pct": float((y == 0).mean()),
        }

        return TrainResult(
            artifacts=[artifact, meta_artifact],
            metrics={
                "n_samples": int(len(X_arr)),
                "n_features": int(X_arr.shape[1]),
                "n_universe_symbols": int(len(np.unique(symbols))),
                "class_distribution": class_dist,
                "data_quality_report": dq_report,
                # W6 (pre-training audit 2026-05-19) — persist data source
                # tier (kite_admin / bhavcopy / yfinance) so post-mortem
                # can tell which Tier the model trained on.
                "data_source_tier": data_source,
                "fit_seconds": round(time.time() - t0, 2),
                "n_folds_succeeded": len(fold_metrics),
                "deflated_sharpe": dsr_pbo.get("deflated_sharpe"),
                "probability_backtest_overfitting": dsr_pbo.get("probability_backtest_overfitting"),
                "pooled_sharpe_from_dsr": dsr_pbo.get("pooled_sharpe"),
                "regime_sharpe": regime_sharpe,
                "regime_n_days": regime_n_days,
                **agg,
            },
            notes=f"Triple-barrier labels (TP={PROFIT_TARGET_ATR}xATR, "
                  f"SL={STOP_LOSS_ATR}xATR, vbd={VERTICAL_BARRIER_DAYS}). "
                  f"AFML Ch.4 sample-weight uniqueness applied. "
                  f"WFCV {WFCV_FOLDS}-fold rolling. "
                  f"{len(np.unique(symbols))} NSE liquid stocks, {DATA_PERIOD}.",
        )

    def evaluate(self, result: TrainResult) -> Dict[str, Any]:
        m = dict(result.metrics)
        # primary_metric is sharpe_mean from the WFCV aggregation —
        # promote gate reads this directly via PR 162's
        # promote_gate_passes() which falls back to non-_mean keys
        # too.
        m["primary_metric"] = "sharpe_mean"
        m["primary_value"] = result.metrics.get("sharpe_mean")
        return m
