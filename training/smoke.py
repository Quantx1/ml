"""
Smoke-mode controls for the unified training runner.

Purpose: shrink every universe-bound + epoch-bound knob so the entire
12-trainer pipeline can run on RunPod in ~30-60 min and ~$1-2 *before*
committing to the ~$6 / 8h full run.

Every trainer that touches a stock universe, training horizon, or
timesteps budget should read these helpers instead of hardcoding limits.

Env vars (all optional; default off = full run):

    SMOKE_MODE=1               Master toggle. When set, every helper below
                                returns a smoke-shrunk value.
    SMOKE_UNIVERSE_SIZE=10     Cap stock universe to N (default 10).
    SMOKE_TIMESTEPS=50000      Cap RL timesteps (default 50k vs 1M full).
    SMOKE_EPOCHS=2             Cap DL training epochs (default 2 vs 12).
    SMOKE_YFINANCE_PERIOD=2y   Shrink yfinance history (default 2y vs 8-10y).

The SMOKE_MODE master flag does NOT affect zero-shot trainers
(momentum_timesfm) — they don't have a meaningful "smaller" mode
beyond the calibration step they already run, which finishes in
seconds.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence


def is_smoke_mode() -> bool:
    """True when SMOKE_MODE=1 is set. Treated as falsy for any other value."""
    return os.environ.get("SMOKE_MODE", "").strip() in ("1", "true", "TRUE", "yes")


def smoke_universe_size() -> int:
    """Stock-universe cap when smoke mode is active. Default 10."""
    try:
        return max(2, int(os.environ.get("SMOKE_UNIVERSE_SIZE", "10")))
    except ValueError:
        return 10


def smoke_timesteps() -> int:
    """RL training timesteps when smoke mode is active. Default 50,000."""
    try:
        return max(1000, int(os.environ.get("SMOKE_TIMESTEPS", "50000")))
    except ValueError:
        return 50_000


def smoke_epochs() -> int:
    """DL training epoch cap when smoke mode is active. Default 2."""
    try:
        return max(1, int(os.environ.get("SMOKE_EPOCHS", "2")))
    except ValueError:
        return 2


def smoke_yf_period() -> str:
    """yfinance period string when smoke mode is active. Default '2y'."""
    return os.environ.get("SMOKE_YFINANCE_PERIOD", "2y")


def apply_universe_cap(
    symbols: Sequence[str], *, full_size: Optional[int] = None,
) -> list:
    """Return the universe to use given the current smoke setting.

    Parameters
    ----------
    symbols : sequence of stock tickers
        The trainer's full universe.
    full_size : int, optional
        When smoke mode is OFF and ``full_size`` is provided, cap the
        universe at this number. Useful for trainers whose "full" run
        is itself a top-N (e.g. tft_swing's top-100, lgbm_signal_gate's
        top-200).

    Returns
    -------
    list of str — the universe to actually train on.
    """
    syms = list(symbols)
    if is_smoke_mode():
        return syms[: smoke_universe_size()]
    if full_size is not None:
        return syms[: full_size]
    return syms


def smoke_label() -> str:
    """One-word label for log/metric output ('smoke' vs 'full')."""
    return "smoke" if is_smoke_mode() else "full"


# ---------------------------------------------------------------------------
# GPU detection — shared by every trainer that has an optional CUDA path
# ---------------------------------------------------------------------------


def cuda_available() -> bool:
    """True when a CUDA GPU is usable for training.

    Cached at first call. Trainers use this to choose between
    ``device='cuda'`` / ``device='cpu'`` on LightGBM, XGBoost, SB3, and
    PyTorch model placement.
    """
    global _cuda_cached
    if _cuda_cached is None:
        try:
            import torch  # noqa: PLC0415
            _cuda_cached = bool(torch.cuda.is_available())
        except Exception:
            _cuda_cached = False
    return _cuda_cached


_cuda_cached = None


def lightgbm_device() -> str:
    """LightGBM device string ('cuda' on GPU when wheel supports it, else 'cpu').

    LightGBM 4.x ships pip wheels WITHOUT the CUDA tree learner unless
    you build from source with USE_CUDA=ON. The 2026-05-11 RunPod smoke
    confirmed this: `[LightGBM] [Fatal] CUDA Tree Learner was not enabled
    in this build.` So we probe at first call — run a 1-row lgb.train with
    device='cuda' and catch the fatal. Cached result.

    Override via env:
        LGBM_DEVICE=cpu    — force CPU even on GPU machines
        LGBM_DEVICE=cuda   — force CUDA, skip the probe
    """
    if os.environ.get("LGBM_DEVICE"):
        return os.environ["LGBM_DEVICE"]
    if not cuda_available():
        return "cpu"

    # GPU is present; probe whether the LightGBM build supports CUDA.
    global _lgbm_device_cached
    if _lgbm_device_cached is not None:
        return _lgbm_device_cached
    try:
        import lightgbm as lgb  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        ds = lgb.Dataset(
            np.random.RandomState(0).randn(32, 4),
            label=np.random.RandomState(0).randn(32),
        )
        # silence=verbose=-1, only 1 boost round, just enough to exercise
        # the tree learner. If the CUDA build is missing, this fatal-fires.
        lgb.train(
            {"objective": "regression", "device": "cuda",
             "verbose": -1, "num_threads": 1},
            ds, num_boost_round=1,
        )
        _lgbm_device_cached = "cuda"
    except Exception as exc:  # noqa: BLE001
        import logging  # noqa: PLC0415
        logging.getLogger(__name__).info(
            "LightGBM CUDA probe failed (%s) — falling back to device='cpu'",
            str(exc)[:120],
        )
        _lgbm_device_cached = "cpu"
    return _lgbm_device_cached


_lgbm_device_cached = None


def xgboost_device() -> str:
    """XGBoost device string ('cuda' on GPU, 'cpu' otherwise).

    XGBoost 2.x uses ``device='cuda'`` + ``tree_method='hist'`` for GPU.
    """
    return "cuda" if cuda_available() else "cpu"


def torch_device() -> str:
    """PyTorch device string ('cuda' / 'cpu'), respected by SB3 + DL."""
    return "cuda" if cuda_available() else "cpu"


def maybe_cap(value: int, cap_fn) -> int:
    """Return ``cap_fn()`` when smoke mode active, else ``value``.

    Handy at trainer-class body for timesteps / epochs constants::

        TIMESTEPS = maybe_cap(1_000_000, smoke_timesteps)
    """
    return cap_fn() if is_smoke_mode() else value


__all__ = [
    "apply_universe_cap",
    "cuda_available",
    "is_smoke_mode",
    "lightgbm_device",
    "maybe_cap",
    "smoke_epochs",
    "smoke_label",
    "smoke_timesteps",
    "smoke_universe_size",
    "smoke_yf_period",
    "torch_device",
    "xgboost_device",
]
