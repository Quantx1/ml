"""Meta-labeling conviction features — context for "will THIS signal win?".

Consumes (a) per-name OOS fold predictions from the walk-forward harness
(``dump_preds_path``), (b) the engine's own PIT feature frame (selected
columns only — nothing recomputed), (c) the daily regime series
(``ml/regime`` ensemble output parquet) and (d) the NIFTY benchmark closes.
Emits ~20 point-in-time features per (date, symbol) signal row.

Everything is information available AT the signal date: engine features are
PIT by construction, the regime series is causal (filtered, not smoothed),
market context uses trailing windows on benchmark closes, and the signal
predictions themselves are out-of-sample by the harness's purged folds.
Rows whose date precedes the regime series or benchmark warmup carry NaNs
and are dropped by the spine — never filled.

Spec: docs/superpowers/specs/2026-07-07-meta-labeling-conviction-design.md §4.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

_EPS = 1e-9

#: Engine-specific columns pulled (read-only) from the engine's feature frame.
#: Names must exist in that engine's FEATURE_ORDER — validated, not assumed.
META_ENGINE_MAP: Dict[str, Dict[str, List[str]]] = {
    "momentum": {
        "name_features": ["realized_vol_63", "beta_index_63",
                          "amihud_illiq_21", "dist_high_252"],
        "forecast_features": ["tsfm_fwd_ret", "tsfm_uncert",
                              "kronos_fwd_ret", "ens_fwd_ret"],
    },
    "swing": {
        "name_features": ["realized_vol_21", "beta_index_63",
                          "rel_volume_21", "pullback_from_high_21"],
        "forecast_features": ["tsfm_fwd_ret", "tsfm_uncert", "kronos_fwd_ret",
                              "chronos_fwd_ret", "chronos_uncert", "ens_fwd_ret"],
    },
}

_SIGNAL_FEATURES = ["score_pct", "score_z", "score_dispersion", "score_gap"]
_REGIME_FEATURES = ["regime_bull", "regime_bear", "regime_confidence",
                    "days_since_switch"]
_MARKET_FEATURES = ["mkt_rv21", "mkt_ret_21", "mkt_dist_high_63"]
_SPREAD_FEATURE = "tsfm_kronos_spread"


def meta_feature_cols(engine: str) -> List[str]:
    """The ordered feature list for one engine's conviction model."""
    m = META_ENGINE_MAP[engine]
    return [*_SIGNAL_FEATURES, *_REGIME_FEATURES, *_MARKET_FEATURES,
            *m["name_features"], *m["forecast_features"], _SPREAD_FEATURE]


def _signal_context(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional context of the primary model's scores."""
    df = preds.sort_values(["date", "pred"], ascending=[True, False]).copy()
    g = df.groupby("date")["pred"]
    df["score_pct"] = g.rank(pct=True, method="average")
    mean, std = g.transform("mean"), g.transform("std")
    df["score_dispersion"] = std
    df["score_z"] = (df["pred"] - mean) / (std + _EPS)
    # gap to the NEXT-ranKED name below (descending order => diff to next row
    # within the date), scaled by that date's dispersion; the last name has no
    # gap below it -> 0.
    df["score_gap"] = (-df.groupby("date")["pred"].diff(-1)).fillna(0.0).abs() / (std + _EPS)
    return df


def _regime_context(regime: pd.DataFrame) -> pd.DataFrame:
    """One-hot state + confidence + days-since-switch, computed ON the regime
    calendar (causal cumcount) BEFORE any merge."""
    r = regime.sort_values("date").copy()
    r["date"] = pd.to_datetime(r["date"]).astype("datetime64[ns]")
    r["regime_bull"] = (r["state_name"] == "bull").astype(float)
    r["regime_bear"] = (r["state_name"] == "bear").astype(float)
    r["regime_confidence"] = r["confidence"].astype(float)
    switch = (r["state_name"] != r["state_name"].shift()).cumsum()
    r["days_since_switch"] = r.groupby(switch).cumcount().astype(float)
    return r[["date", *_REGIME_FEATURES]]


def _market_context(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Trailing market state from benchmark closes (all causal windows)."""
    b = benchmark.sort_values("date").copy()
    b["date"] = pd.to_datetime(b["date"]).astype("datetime64[ns]")
    ret = b["close"].pct_change(fill_method=None)
    b["mkt_rv21"] = ret.rolling(21).std()
    b["mkt_ret_21"] = b["close"].pct_change(21, fill_method=None)
    b["mkt_dist_high_63"] = b["close"] / b["close"].rolling(63).max() - 1.0
    return b[["date", *_MARKET_FEATURES]]


def build_meta_features(
    preds: pd.DataFrame,
    engine_feats: pd.DataFrame,
    engine: str,
    regime: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """Assemble the conviction feature frame for one engine.

    Args:
        preds: ['date','symbol','pred', ...] — per-name signal scores (OOS
            fold predictions at training time; the live cross-section at
            serving time). Extra columns pass through untouched.
        engine_feats: the engine's own feature frame (['date','symbol', ...]);
            only the META_ENGINE_MAP columns are read.
        engine: 'momentum' | 'swing' (KeyError on anything else).
        regime: ['date','state_name','confidence'] daily regime series.
        benchmark: ['date','close'] NIFTY closes.

    Returns:
        (df, feature_cols): preds columns + the ~20 meta features. Rows are
        NOT dropped here — NaN handling belongs to the consumer (the spine
        drops, serving degrades).

    Raises:
        KeyError: unknown engine.
        ValueError: a mapped column is missing from engine_feats (fail loud —
            a silently absent column would train a crippled model).
    """
    m = META_ENGINE_MAP[engine]
    join_cols = [*m["name_features"], *m["forecast_features"]]
    missing = [c for c in join_cols if c not in engine_feats.columns]
    if missing:
        raise ValueError(
            f"meta features for '{engine}': engine feature frame is missing "
            f"{missing} — engine FEATURE_ORDER changed? Update META_ENGINE_MAP.")

    df = _signal_context(preds)
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    df = df.merge(_regime_context(regime), on="date", how="left",
                  validate="many_to_one")
    df = df.merge(_market_context(benchmark), on="date", how="left",
                  validate="many_to_one")
    ef = engine_feats[["date", "symbol", *join_cols]].copy()
    ef["date"] = pd.to_datetime(ef["date"]).astype("datetime64[ns]")
    df = df.merge(ef, on=["date", "symbol"], how="left", validate="many_to_one")
    df[_SPREAD_FEATURE] = (df["tsfm_fwd_ret"] - df["kronos_fwd_ret"]).abs()
    return df, meta_feature_cols(engine)


__all__ = ["META_ENGINE_MAP", "build_meta_features", "meta_feature_cols"]
