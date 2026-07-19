"""Serving-side reader for the weekly forecast-feature cache (Phase 2).

The Phase-2 ranker artifacts (momentum/swing/positional) list foundation-model
forecast columns in their ``feature_order.json`` (tsfm/kronos for momentum,
plus chronos for swing/positional, plus the ``ens_fwd_ret`` blend). Those
columns are produced by GPU foundation models on a WEEKLY refresh
(``scripts/runpod/refresh_forecast_cache.sh``) and persisted as parquets in
``artifacts/forecast_cache/`` — serving is CPU-only and must never run the
forecasters. This module reads the parquets and returns the LATEST forecast
row per symbol so the serving engines can LEFT-merge them onto the live
feature panel before scoring.

Staleness is by design: forecasts are computed on stride-day rebalance dates
(default 5 = weekly) and the cache refreshes weekly, so served values can be
up to ``stride + refresh cadence`` trading days old. ``forecast_age_days``
is returned on every row so the caller decides its own staleness policy.

Poisoning guards: the trainers apply TWO guards to a cache parquet —
(1) symbol coverage vs the training panel, and (2) per-symbol history span vs
the cache's own span. Guard (1) needs a panel's symbol set, which serving does
not have (the universe is resolved inside the engine and a partial cache is
structurally harmless here: uncovered symbols merge as NaN, tolerated by
LightGBM at predict) — so only guard (2) is applied: a history-starved cache
(median per-symbol span under half the cache span — the 2026-07-06 incident
shape) is treated as absent.

Import contract: lives in ml/ and imports NOTHING from backend (import-linter:
ml must not depend on backend.api/services/platform). Pure pandas + stdlib.
"""
from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: ml/features/forecast_serving.py -> parents[2] == repo root
_ROOT = Path(__file__).resolve().parents[2]

#: Which cache parquets each engine's artifact consumes. Momentum owns
#: tsfm+kronos; swing adds its own chronos; positional consumes all three
#: read-only. Mirrors the trainers' with_forecasts blocks.
ENGINE_CACHE_FILES: Dict[str, Tuple[str, ...]] = {
    "momentum": ("momentum_tsfm.parquet", "momentum_kronos.parquet"),
    "swing": ("momentum_tsfm.parquet", "momentum_kronos.parquet",
              "swing_chronos.parquet"),
    "positional": ("momentum_tsfm.parquet", "momentum_kronos.parquet",
                   "swing_chronos.parquet"),
}


def _default_cache_dir() -> Path:
    return Path(os.environ.get(
        "FORECAST_CACHE_DIR", str(_ROOT / "artifacts" / "forecast_cache")))


def _read_latest(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.Timestamp]]:
    """Last-row-per-symbol from one cache parquet.

    Returns ``(frame, cache_max_date)`` where frame has ``symbol`` + the
    file's forecast columns, or ``(None, None)`` when the file is absent,
    unreadable, or fails the history-starvation poisoning guard (see module
    docstring for why the trainers' symbol-coverage guard is skipped here).
    """
    if not path.exists():
        return None, None
    try:
        cached = pd.read_parquet(path)
        # pod-written parquets may carry datetime64[us] — normalize
        cached["date"] = pd.to_datetime(cached["date"]).astype("datetime64[ns]")
        # POISONING GUARD (replicated from the trainers, 2026-07-06 incident):
        # symbols present but with sparse HISTORY (e.g. a cache whose tail
        # top-up added all symbols for a few weeks). If the median per-symbol
        # date SPAN is under half the cache's own span, most symbols are
        # history-starved: treat as absent.
        span = (cached["date"].max() - cached["date"].min()).days
        if span > 0:
            per_sym = cached.groupby("symbol")["date"].agg(
                lambda s: (s.max() - s.min()).days)
            if per_sym.median() < 0.5 * span:
                logger.warning(
                    "forecast cache %s: median per-symbol span %.0fd << cache "
                    "span %.0fd — history-starved cache, treating as absent",
                    path.name, per_sym.median(), span)
                return None, None
        cache_max = pd.Timestamp(cached["date"].max())
        latest = (cached.sort_values("date")
                        .groupby("symbol", as_index=False)
                        .tail(1)
                        .drop(columns=["date"])
                        .reset_index(drop=True))
        return latest, cache_max
    except Exception as exc:  # noqa: BLE001 — fail-soft, serving degrades
        logger.warning("forecast cache read failed (%s): %s", path, exc)
        return None, None


def latest_forecasts(engine: str, cache_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Latest cached forecast values per symbol for ``engine``'s artifact.

    Returns a frame keyed by ``symbol`` with the engine's forecast columns,
    an ``ens_fwd_ret`` blend (mean of the available ``*_fwd_ret`` backends,
    mirroring ``merge_forecast_features``: only formed when >= 2 backends are
    present, NaN otherwise), and ``forecast_age_days`` =
    ``(today - max cache date).days`` — the caller decides staleness policy.

    Fail-soft: returns ``None`` when NO cache parquet is readable (missing
    dir, missing files, or every file poisoned) — the serving engines then
    score without forecast features (degraded) instead of erroring.
    """
    files = ENGINE_CACHE_FILES.get(engine)
    if files is None:
        raise ValueError(
            f"unknown engine {engine!r}; expected one of {sorted(ENGINE_CACHE_FILES)}")
    cdir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()

    out: Optional[pd.DataFrame] = None
    max_date: Optional[pd.Timestamp] = None
    for name in files:
        latest, cache_max = _read_latest(cdir / name)
        if latest is None:
            continue
        out = latest if out is None else out.merge(latest, on="symbol", how="outer")
        max_date = cache_max if max_date is None else max(max_date, cache_max)
    if out is None or max_date is None:
        return None

    # Ensemble blend — mirror merge_forecast_features: mean of the backends'
    # forward-return views, only when >= 2 are present (never blend a single
    # view with itself). Per-row NaNs inside a present column are skipped.
    present = [c for c in ("tsfm_fwd_ret", "kronos_fwd_ret", "chronos_fwd_ret")
               if c in out.columns]
    if len(present) >= 2:
        out["ens_fwd_ret"] = out[present].mean(axis=1)
    else:
        out["ens_fwd_ret"] = np.nan

    out["forecast_age_days"] = int((pd.Timestamp(date.today()) - max_date).days)
    return out


__all__ = ["ENGINE_CACHE_FILES", "latest_forecasts"]
