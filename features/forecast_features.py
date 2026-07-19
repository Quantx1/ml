"""Foundation-model forecast features for the ranker engines (GPU).

Adapters that turn a pretrained time-series foundation model into a small set
of FEATURE columns the LightGBM ranker consumes — the forecasters never decide
a trade, they only add a forward-looking view the price features can't express.

Three backends, all producing one row per (date, symbol):
  - TimesFM 2.5 (google/timesfm-2.5-200m-pytorch) — zero-shot, batched.
  - Kronos (NeoQuasar/Kronos-small) — finance K-line foundation model.
  - Chronos-2 (amazon/chronos-2) — zero-shot quantile forecaster, batched.

Output feature columns (absolute, no benchmark):
  tsfm_fwd_ret      forecast horizon-ahead return = point[-1]/last_close - 1
  tsfm_uncert       forecast dispersion = (q90-q10) at horizon, / last_close
  kronos_fwd_ret    Kronos predicted close[horizon]/last_close - 1
  chronos_fwd_ret   Chronos-2 q50 at horizon / last_close - 1
  chronos_uncert    Chronos-2 (q90-q10) at horizon / last_close

COST: rolling forecasts over universe x history are GPU-heavy. `stride` only
forecasts every Nth trading day (default 5 = weekly) and forward-fills to
daily, cutting inference ~stride x. Requires CUDA — these are GPU models.

NOTE: verified on the RunPod GPU pod (no local GPU to smoke-test). Import is
lazy + fail-loud so a missing model/dep never silently degrades a trainer.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TIMESFM_FEATURES = ["tsfm_fwd_ret", "tsfm_uncert"]
KRONOS_FEATURES = ["kronos_fwd_ret"]
CHRONOS_FEATURES = ["chronos_fwd_ret", "chronos_uncert"]
#: Full forecast feature contract consumed by the ranker — the three backends'
#: columns plus their ensemble blend. ``merge_forecast_features`` guarantees
#: every column here exists (NaN where a backend is absent).
FORECAST_FEATURES = TIMESFM_FEATURES + KRONOS_FEATURES + CHRONOS_FEATURES + ["ens_fwd_ret"]


def pick_device() -> str:
    """cuda → cpu, with a FORECAST_DEVICE env override. CUDA is the RunPod run;
    CPU is the stable local-debug path. Apple MPS is intentionally NOT auto-used
    for these foundation models — TimesFM/Kronos hit unsupported Metal ops and
    segfault the process; set FORECAST_DEVICE=mps only to experiment."""
    import os  # noqa: PLC0415
    import torch  # noqa: PLC0415
    override = os.environ.get("FORECAST_DEVICE", "").strip().lower()
    if override in ("cuda", "mps", "cpu"):
        return override
    if torch.cuda.is_available():
        return "cuda"
    logger.warning("CUDA unavailable — using CPU (local debug; slower than a GPU pod)")
    return "cpu"


def _rebalance_dates(all_dates: np.ndarray, stride: int, min_date=None) -> np.ndarray:
    """Every `stride`-th unique date (plus always the last).

    ``min_date`` filters AFTER the stride slice so an incremental top-up keeps
    the same phase alignment as the original full backfill (cache + top-up
    tile without gaps or duplicated off-phase dates).
    """
    uniq = np.array(sorted(pd.unique(all_dates)))
    picked = uniq[::stride]
    if uniq[-1] not in picked:
        picked = np.append(picked, uniq[-1])
    if min_date is not None:
        cutoff = pd.Timestamp(min_date)
        picked = np.array([d for d in picked if pd.Timestamp(d) >= cutoff])
    return picked


def timesfm_forecast_features(
    panel: pd.DataFrame,
    horizon: int = 20,
    context: int = 512,
    stride: int = 5,
    batch_size: int = 256,
    min_history: int = 252,
    min_date=None,
) -> pd.DataFrame:
    """Rolling TimesFM forecast features. panel=['date','symbol','close',...].

    Returns ['date','symbol', *TIMESFM_FEATURES] at each rebalance date,
    forward-filled to daily by the caller's merge. Fail-loud on missing dep.
    """
    rdates_probe = _rebalance_dates(panel["date"].to_numpy(), stride, min_date=min_date)
    if len(rdates_probe) == 0:
        logger.info("timesfm: no rebalance dates >= min_date — nothing to compute")
        return pd.DataFrame(columns=["date", "symbol", *TIMESFM_FEATURES])

    import timesfm  # noqa: PLC0415
    import torch  # noqa: PLC0415

    device = pick_device()
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(
        max_context=context, max_horizon=max(horizon, 32),
        normalize_inputs=True, use_continuous_quantile_head=True,
        force_flip_invariance=True, fix_quantile_crossing=True,
    ))

    df = panel.sort_values(["symbol", "date"]).copy()
    by_symbol = {s: g.reset_index(drop=True) for s, g in df.groupby("symbol")}
    rdates = _rebalance_dates(df["date"].to_numpy(), stride, min_date=min_date)
    rows: List[dict] = []

    for di, d in enumerate(rdates):
        inputs, meta = [], []
        for sym, g in by_symbol.items():
            hist = g[g["date"] <= d]
            if len(hist) < min_history:
                continue
            series = hist["close"].to_numpy(dtype=float)[-context:]
            inputs.append(series)
            meta.append((sym, series[-1]))
        if not inputs:
            continue
        # batch through the model
        for b in range(0, len(inputs), batch_size):
            chunk = inputs[b:b + batch_size]
            cmeta = meta[b:b + batch_size]
            point, quant = model.forecast(horizon=horizon, inputs=chunk)
            point = np.asarray(point); quant = np.asarray(quant)
            for k, (sym, last_close) in enumerate(cmeta):
                if last_close <= 0:
                    continue
                fwd = point[k, horizon - 1] / last_close - 1.0
                # quantile head: (.., horizon, 11) -> [mean,q10..q90]; spread q90-q10
                q = quant[k, horizon - 1]
                uncert = float((q[-1] - q[1]) / last_close) if q.shape[-1] >= 11 else 0.0
                rows.append({"date": d, "symbol": sym,
                             "tsfm_fwd_ret": float(fwd), "tsfm_uncert": uncert})
        if di % 20 == 0:
            logger.info("timesfm: %d/%d rebalance dates", di, len(rdates))

    out = pd.DataFrame(rows, columns=["date", "symbol", *TIMESFM_FEATURES])
    logger.info("timesfm features: %d rows over %d dates", len(out), len(rdates))
    return out


def kronos_forecast_features(
    panel: pd.DataFrame,
    horizon: int = 20,
    context: int = 400,
    stride: int = 5,
    min_history: int = 252,
    min_date=None,
    model_id: str = "NeoQuasar/Kronos-small",
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
) -> pd.DataFrame:
    """Rolling Kronos forecast features. panel needs OHLCV + 'date','symbol'.

    Requires the Kronos repo on PYTHONPATH (KRONOS_PATH). Fail-loud on missing.
    """
    rdates_probe = _rebalance_dates(panel["date"].to_numpy(), stride, min_date=min_date)
    if len(rdates_probe) == 0:
        logger.info("kronos: no rebalance dates >= min_date — nothing to compute")
        return pd.DataFrame(columns=["date", "symbol", *KRONOS_FEATURES])

    import torch  # noqa: PLC0415
    device = pick_device()
    from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: PLC0415

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
    kmodel = Kronos.from_pretrained(model_id)
    predictor = KronosPredictor(kmodel, tokenizer, device=device, max_context=context)

    df = panel.sort_values(["symbol", "date"]).copy()
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    by_symbol = {s: g.reset_index(drop=True) for s, g in df.groupby("symbol")}
    rdates = _rebalance_dates(df["date"].to_numpy(), stride, min_date=min_date)
    rows: List[dict] = []

    for di, d in enumerate(rdates):
        for sym, g in by_symbol.items():
            hist = g[g["date"] <= d]
            if len(hist) < min_history:
                continue
            ctx = hist.iloc[-context:]
            last_close = float(ctx["close"].iloc[-1])
            if last_close <= 0:
                continue
            x_ts = pd.Series(ctx["date"].to_numpy())
            # synthetic future business-day timestamps for the horizon
            y_ts = pd.Series(pd.bdate_range(
                start=pd.Timestamp(d) + pd.Timedelta(days=1), periods=horizon))
            try:
                pred = predictor.predict(
                    df=ctx[cols].reset_index(drop=True),
                    x_timestamp=x_ts, y_timestamp=y_ts,
                    pred_len=horizon, T=1.0, top_p=0.9, sample_count=1,
                )
                fwd = float(pred["close"].iloc[-1]) / last_close - 1.0
                rows.append({"date": d, "symbol": sym, "kronos_fwd_ret": fwd})
            except Exception as e:  # noqa: BLE001
                logger.debug("kronos predict failed for %s @ %s: %s", sym, d, e)
        if di % 20 == 0:
            logger.info("kronos: %d/%d rebalance dates", di, len(rdates))

    out = pd.DataFrame(rows, columns=["date", "symbol", *KRONOS_FEATURES])
    logger.info("kronos features: %d rows over %d dates", len(out), len(rdates))
    return out


def chronos_forecast_features(
    panel: pd.DataFrame,
    horizon: int = 10,
    context: int = 512,
    stride: int = 5,
    batch_size: int = 64,
    min_history: int = 252,
    min_date=None,
    model_id: str = "amazon/chronos-2",
) -> pd.DataFrame:
    """Rolling Chronos-2 zero-shot forecast features. panel=['date','symbol','close',...].

    Returns ['date','symbol', *CHRONOS_FEATURES] at each rebalance date,
    forward-filled to daily by the caller's merge. Fail-loud on missing dep.
    Verified API (chronos-forecasting 2.2.2): BaseChronosPipeline.from_pretrained
    + predict_quantiles(inputs=[tensor, ...], prediction_length, quantile_levels).
    """
    rdates_probe = _rebalance_dates(panel["date"].to_numpy(), stride, min_date=min_date)
    if len(rdates_probe) == 0:
        logger.info("chronos: no rebalance dates >= min_date — nothing to compute")
        return pd.DataFrame(columns=["date", "symbol", *CHRONOS_FEATURES])

    import torch  # noqa: PLC0415
    from chronos import BaseChronosPipeline  # noqa: PLC0415

    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map=pick_device())

    df = panel.sort_values(["symbol", "date"]).copy()
    by_symbol = {s: g.reset_index(drop=True) for s, g in df.groupby("symbol")}
    rdates = _rebalance_dates(df["date"].to_numpy(), stride, min_date=min_date)
    rows: List[dict] = []

    for di, d in enumerate(rdates):
        inputs, meta = [], []
        for sym, g in by_symbol.items():
            hist = g[g["date"] <= d]
            if len(hist) < min_history:
                continue
            series = hist["close"].to_numpy(dtype=float)[-context:]
            inputs.append(torch.tensor(series, dtype=torch.float32))
            meta.append((sym, float(series[-1])))
        if not inputs:
            continue
        # batch through the model
        for b in range(0, len(inputs), batch_size):
            chunk = inputs[b:b + batch_size]
            cmeta = meta[b:b + batch_size]
            quantiles, _mean = pipe.predict_quantiles(
                inputs=chunk, prediction_length=horizon,
                quantile_levels=[0.1, 0.5, 0.9],
            )
            for k, (sym, last_close) in enumerate(cmeta):
                if last_close <= 0:
                    continue
                # Chronos2Pipeline returns a list of per-series tensors with a
                # LEADING BATCH DIM — shape (1, horizon, 3) — while the classic
                # pipelines stack to (batch, horizon, 3). Live-probed on
                # chronos-forecasting 2.2.2 (the smoke caught the mismatch the
                # mocked unit test couldn't). Strip any leading unit dim to get
                # (horizon, 3) before indexing.
                q = np.asarray(quantiles[k])
                if q.ndim == 3 and q.shape[0] == 1:
                    q = q[0]
                q10, q50, q90 = (float(q[horizon - 1, 0]),
                                 float(q[horizon - 1, 1]),
                                 float(q[horizon - 1, 2]))
                rows.append({"date": d, "symbol": sym,
                             "chronos_fwd_ret": q50 / last_close - 1.0,
                             "chronos_uncert": (q90 - q10) / last_close})
        if di % 20 == 0:
            logger.info("chronos: %d/%d rebalance dates", di, len(rdates))

    out = pd.DataFrame(rows, columns=["date", "symbol", *CHRONOS_FEATURES])
    logger.info("chronos features: %d rows over %d dates", len(out), len(rdates))
    return out


def merge_forecast_features(
    feature_panel: pd.DataFrame,
    forecast_frames: List[pd.DataFrame],
) -> pd.DataFrame:
    """As-of merge each forecast frame (computed on rebalance dates) into the
    daily feature panel per symbol, forward-filling between rebalances."""
    # merge_asof requires BOTH frames sorted by the `on` key (date) globally
    # (the `by=symbol` grouping is applied on top); sorting by [symbol,date]
    # leaves date non-monotonic and raises "left keys must be sorted".
    out = feature_panel.sort_values("date").reset_index(drop=True)
    # Normalize datetime units on BOTH sides: parquet round-trips can yield
    # datetime64[us] (pod-written caches) while computed frames carry [ns] —
    # merge_asof hard-errors on mixed units (caught by the swing CPU smoke).
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    for ff in forecast_frames:
        if ff is None or ff.empty:
            continue
        ff = ff.sort_values("date").reset_index(drop=True)
        ff["date"] = pd.to_datetime(ff["date"]).astype("datetime64[ns]")
        out = pd.merge_asof(
            out, ff, on="date", by="symbol", direction="backward",
        )
    # Ensemble forecast = mean of the backends' forward-return views.
    # Only formed when >= 2 backends are present; otherwise NaN (fail-soft —
    # LightGBM tolerates it, and we never blend a single view with itself).
    # Per-row NaNs inside a present column are skipped by pandas' mean.
    present = [c for c in ("tsfm_fwd_ret", "kronos_fwd_ret", "chronos_fwd_ret")
               if c in out.columns]
    if len(present) >= 2:
        out["ens_fwd_ret"] = out[present].mean(axis=1)
    else:
        out["ens_fwd_ret"] = np.nan
    # Contract guarantee: every FORECAST_FEATURES column exists on the output
    # (NaN where a backend is absent) so downstream column selection never
    # KeyErrors on a missing backend.
    for col in FORECAST_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    return out
