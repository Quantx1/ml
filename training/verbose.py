"""Verbose Jupyter-style step printers for all trainers (locked 2026-05-12).

Every trainer uses these helpers so log output is uniformly readable
across trainers — banner, numbered steps, sub-progress lines. ``flush=True``
on every print so output streams real-time through ``python -u`` + ``tee``.

Usage::

    from ml.training.verbose import banner, step, sub, fold_header, fold_result

    banner("intraday_lstm", universe=10, epochs=12, batch_size=64, device="cuda")
    step(1, 5, "fetch 5-min OHLCV from yfinance")
    sub("RELIANCE.NS: 4680 bars (60d × 78 bars/day)")
    ...
    fold_header(1, 5, train_rows=18000, test_rows=4500)
    fold_result(1, accuracy=0.62, sharpe=0.84)
"""
from __future__ import annotations

from typing import Any


def banner(trainer_name: str, **config: Any) -> None:
    """Print the trainer's start banner with key configuration."""
    width = 62
    print(f"\n[{trainer_name}] ╔" + "═" * (width - 2) + "╗", flush=True)
    title = f" {trainer_name} TRAINING — verbose mode"
    print(f"[{trainer_name}] ║{title:<{width - 2}}║", flush=True)
    for k, v in config.items():
        line = f"   {k}: {v}"
        print(f"[{trainer_name}] ║{line:<{width - 2}}║", flush=True)
    print(f"[{trainer_name}] ╚" + "═" * (width - 2) + "╝\n", flush=True)


def step(n: int, total: int, description: str, trainer_name: str = "") -> None:
    """Print a major step marker like '=== STEP 3/8 — feature build ===.'"""
    prefix = f"[{trainer_name}] " if trainer_name else ""
    print(f"\n{prefix}=== STEP {n}/{total} — {description} ===", flush=True)


def step_done(n: int, total: int, summary: str = "", trainer_name: str = "") -> None:
    """Print a step completion line."""
    prefix = f"[{trainer_name}] " if trainer_name else ""
    msg = f"=== STEP {n}/{total} complete"
    if summary:
        msg += f" — {summary}"
    msg += " ==="
    print(f"{prefix}{msg}\n", flush=True)


def sub(msg: str, trainer_name: str = "") -> None:
    """Print an indented sub-line (one per symbol, per epoch, etc.)."""
    prefix = f"[{trainer_name}]   " if trainer_name else "  "
    print(f"{prefix}{msg}", flush=True)


def fold_header(fold_idx: int, total_folds: int, **fold_info: Any) -> None:
    """Print a fold separator with train/test sizes etc."""
    extras = " ".join(f"{k}={v}" for k, v in fold_info.items())
    print(f"\n--- FOLD {fold_idx + 1}/{total_folds}: {extras} ---", flush=True)


def fold_result(fold_idx: int, **metrics: Any) -> None:
    """Print fold completion with metrics summary."""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.3f}")
        else:
            parts.append(f"{k}={v}")
    print(f"FOLD {fold_idx + 1} OK  " + "  ".join(parts), flush=True)


def epoch_progress(
    epoch: int, total_epochs: int,
    train_loss: float, val_loss: float,
    extra: str = "",
) -> None:
    """Print one epoch-level training progress line."""
    msg = (
        f"epoch {epoch + 1:>3}/{total_epochs}  "
        f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
    )
    if extra:
        msg += f"  {extra}"
    print(msg, flush=True)
