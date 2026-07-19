"""
PR 128 — unified E2E training runner.

Single entry point for executing every registered trainer. Per the
locked memory directive, ML PRs add trainer modules but DO NOT train
inline. After the PR sequence finishes, run::

    python -m ml.training.runner --all

on the GPU box and every trainer's artifacts get uploaded + a
``model_versions`` row gets written. Phase H of the v1 plan calls this.

Subset usage::

    python -m ml.training.runner --only tft_swing,regime_hmm
    python -m ml.training.runner --list
    python -m ml.training.runner --skip-gpu       # CPU-only run
    python -m ml.training.runner --promote        # auto-promote on eval pass

The runner never silently swallows trainer failures — a failure in one
trainer is logged with full traceback and the loop continues so a 12-trainer
batch isn't aborted by one broken model.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .base import Trainer, TrainerError, TrainResult
from .discovery import discover_sorted

logger = logging.getLogger("ml.training.runner")


@dataclass
class RunReport:
    name: str
    status: str  # "ok" | "skipped" | "failed"
    duration_sec: float
    metrics: Dict[str, object]
    error: Optional[str] = None
    version: Optional[int] = None
    promoted: bool = False


def _has_cuda() -> bool:
    try:
        import torch  # noqa: PLC0415
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _git_sha() -> Optional[str]:
    sha = (
        os.environ.get("RAILWAY_GIT_COMMIT_SHA")
        or os.environ.get("VERCEL_GIT_COMMIT_SHA")
        or os.environ.get("GIT_SHA")
    )
    if sha:
        return sha[:12]
    try:
        import subprocess  # noqa: PLC0415
        r = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    except Exception:
        pass
    return None


def _trained_by() -> str:
    return (
        os.environ.get("TRAINED_BY")
        or os.environ.get("USER")
        or "ml-runner"
    )


def _filter_trainers(
    trainers: List[Trainer],
    only: Optional[List[str]],
    skip_gpu: bool,
    has_gpu: bool,
    *,
    force_cpu: bool = False,
) -> List[tuple[Trainer, Optional[str]]]:
    """Return list of (trainer, skip_reason). skip_reason=None means run.

    PR 152 — when ``--force`` is passed at the CLI we still surface the
    "no GPU" condition as a *warning* in the report rather than refusing
    to run, because some users (laptop dev) want the pipeline to attempt
    every trainer even on CPU. The trainer's own ``train()`` may still
    fail loudly if a CUDA-only op is hit; that's fine — it lands in the
    report as a normal failure with a clear traceback.
    """
    out: List[tuple[Trainer, Optional[str]]] = []
    only_set = {s.strip() for s in only} if only else None
    for t in trainers:
        if only_set is not None and t.name not in only_set:
            continue
        reason: Optional[str] = None
        if t.requires_gpu and (skip_gpu or not has_gpu):
            if force_cpu:
                # Caller knows the risk; let train() try and fail naturally.
                reason = None
            else:
                reason = "requires_gpu but no GPU available"
        out.append((t, reason))
    if only_set:
        unknown = only_set - {t.name for t, _ in out}
        if unknown:
            raise SystemExit(f"unknown trainer names: {sorted(unknown)}")
    return out


def run(
    *,
    only: Optional[List[str]] = None,
    skip_gpu: bool = False,
    promote: bool = False,
    out_root: Optional[Path] = None,
    dry_run: bool = False,
    force_cpu: bool = False,
) -> List[RunReport]:
    """Execute the unified training pipeline.

    Returns a list of RunReport entries — one per discovered trainer
    (including skipped). The caller (CLI or admin endpoint) renders them.
    """
    has_gpu = _has_cuda()
    trainers = discover_sorted()
    targets = _filter_trainers(trainers, only, skip_gpu, has_gpu, force_cpu=force_cpu)
    git_sha = _git_sha()
    trained_by = _trained_by()

    if out_root is None:
        out_root = Path(tempfile.mkdtemp(prefix="ml_train_"))
    else:
        out_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "runner start: %d trainers, gpu=%s, dry_run=%s, out_root=%s, git=%s",
        len(targets), has_gpu, dry_run, out_root, git_sha,
    )

    # PR 158 — annotate Sentry with the training-run release tag so any
    # exceptions raised inside trainers ship to the same release timeline
    # as the API. Best-effort: missing sentry_sdk is fine.
    try:
        import sentry_sdk  # noqa: PLC0415
        sentry_sdk.set_tag("ml.training_run", "true")
        if git_sha:
            sentry_sdk.set_tag("ml.git_sha", git_sha)
    except Exception:
        pass

    reports: List[RunReport] = []
    for trainer, skip_reason in targets:
        t0 = time.time()
        if skip_reason is not None:
            logger.info("[%s] SKIP — %s", trainer.name, skip_reason)
            reports.append(RunReport(
                name=trainer.name, status="skipped",
                duration_sec=0.0, metrics={}, error=skip_reason,
            ))
            continue

        out_dir = out_root / trainer.name
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[%s] TRAIN starting → %s", trainer.name, out_dir)
        try:
            train_result: TrainResult = trainer.train(out_dir)
            eval_metrics = trainer.evaluate(train_result)
            duration = time.time() - t0

            # PR 167 — promote gate: only allow is_prod=TRUE if metrics
            # pass the financial-eval thresholds. Trainers can override
            # via `promote_thresholds` class attr (e.g. AutoPilot's
            # tighter drawdown ceiling). Trainers whose primary_metric
            # isn't financial (regime_hmm log-likelihood, momentum
            # zero-shot pointer registration) opt out by setting
            # `skip_promote_gate=True`.
            promote_for_this = bool(promote)
            gate_reasons: List[str] = []
            if promote_for_this and not getattr(trainer, "skip_promote_gate", False):
                from ml.eval import promote_gate_passes  # noqa: PLC0415
                merged = {**train_result.metrics, **eval_metrics}
                trainer_thresholds = getattr(trainer, "promote_thresholds", None)
                passed, gate_reasons = promote_gate_passes(merged, trainer_thresholds)
                if not passed:
                    promote_for_this = False
                    logger.warning(
                        "[%s] promote-gate blocked: %s",
                        trainer.name, "; ".join(gate_reasons),
                    )

            # Forecaster / classifier quality gate. Trainers with
            # skip_promote_gate=True must still surface a custom quality
            # signal via `<trainer>_quality_pass` (bool) + `<trainer>_
            # quality_reason` (str). If False, block is_prod regardless
            # of --promote — otherwise a known-bad forecaster gets wired
            # into the consumer and silently degrades signal quality.
            if promote_for_this and getattr(trainer, "skip_promote_gate", False):
                merged = {**train_result.metrics, **eval_metrics}
                quality_key = f"{trainer.name}_quality_pass"
                if quality_key in merged and not merged[quality_key]:
                    promote_for_this = False
                    reason = merged.get(
                        f"{trainer.name}_quality_reason",
                        "quality gate failed (no reason supplied)",
                    )
                    gate_reasons = [f"{quality_key}=False: {reason}"]
                    logger.warning(
                        "[%s] quality-gate blocked: %s",
                        trainer.name, reason,
                    )

            # Safety net: trainers that emit None / missing primary_value
            # OR explicitly zero calibration counts produced no real signal.
            # Hard-block promotion regardless of skip_promote_gate. Catches
            # broken trainers (e.g. momentum_timesfm v2 where the foundation
            # model checkpoint failed to load → all forecasts errored →
            # n_calibrations=0 but no quality flag emitted).
            if promote_for_this:
                merged = {**train_result.metrics, **eval_metrics}
                primary_value = merged.get("primary_value")
                n_cal = merged.get("n_calibrations")
                n_wf = merged.get("n_walk_forward_windows")
                if (
                    primary_value is None
                    or (n_cal is not None and n_cal == 0)
                    or (n_wf is not None and n_wf == 0 and n_cal is None)
                ):
                    promote_for_this = False
                    reason = (
                        f"safety net: primary_value={primary_value} "
                        f"n_calibrations={n_cal} n_walk_forward_windows={n_wf} "
                        f"(no real eval signal — trainer likely broken)"
                    )
                    gate_reasons.append(reason)
                    logger.warning(
                        "[%s] safety-net blocked: %s",
                        trainer.name, reason,
                    )

            # Phase A — serve-smoke promote precondition: round-trip the
            # artifact through the serving feature contract. Closes the audit's
            # #1 train/serve-skew finding (LGBM 15-vs-30). Gates promote only.
            if promote_for_this:
                try:
                    smoke_ok, smoke_reason = trainer.serve_smoke(out_dir)
                except Exception as exc:  # noqa: BLE001
                    smoke_ok, smoke_reason = False, f"serve_smoke raised: {exc}"
                if not smoke_ok:
                    promote_for_this = False
                    gate_reasons.append(f"serve_smoke_failed: {smoke_reason}")
                    logger.warning(
                        "[%s] serve-smoke blocked promote: %s",
                        trainer.name, smoke_reason,
                    )

            # PR 197 — compute per-model Kelly fraction from backtest
            # win_rate + profit_factor. AutoPilot reads this at runtime
            # to scale per-signal position size. Skip-gate trainers
            # (regime_hmm, momentum_timesfm) emit Kelly=0.0 because
            # they aren't directional traders.
            from ml.eval.kelly import kelly_from_metrics  # noqa: PLC0415
            kelly = kelly_from_metrics({**train_result.metrics, **eval_metrics})

            row: Dict[str, object] = {}
            if not dry_run:
                row = trainer.register(
                    train_result,
                    {**eval_metrics, "kelly_fraction": round(kelly, 4)},
                    trained_by=trained_by,
                    git_sha=git_sha,
                    promote=promote_for_this,
                )
            reports.append(RunReport(
                name=trainer.name, status="ok",
                duration_sec=duration,
                metrics={
                    **train_result.metrics,
                    **eval_metrics,
                    "promote_gate_passed": len(gate_reasons) == 0,
                    "promote_gate_reasons": gate_reasons,
                    "kelly_fraction": round(kelly, 4),
                },
                version=int(row.get("version")) if row.get("version") is not None else None,
                promoted=bool(row.get("is_prod")) if row else False,
            ))
            logger.info(
                "[%s] OK in %.1fs (artifacts=%d, metrics=%s, promoted=%s)",
                trainer.name, duration, len(train_result.artifacts),
                eval_metrics, promote_for_this,
            )
        except Exception as exc:  # noqa: BLE001 — keep going across trainers
            duration = time.time() - t0
            tb = traceback.format_exc()
            logger.error("[%s] FAILED after %.1fs: %s\n%s",
                         trainer.name, duration, exc, tb)
            reports.append(RunReport(
                name=trainer.name, status="failed",
                duration_sec=duration, metrics={},
                error=f"{type(exc).__name__}: {exc}",
            ))

    logger.info("runner finished: %s", _summary(reports))
    return reports


def _summary(reports: List[RunReport]) -> str:
    counts = {"ok": 0, "failed": 0, "skipped": 0}
    for r in reports:
        counts[r.status] = counts.get(r.status, 0) + 1
    return f"ok={counts['ok']} failed={counts['failed']} skipped={counts['skipped']}"


def _print_report_human(reports: List[RunReport]) -> None:
    print()
    print("=" * 78)
    print(f"{'name':<32} {'status':<8} {'sec':>6}  {'version':<8} metrics")
    print("-" * 78)
    for r in reports:
        v = f"v{r.version}" if r.version is not None else ""
        promo = " *prod" if r.promoted else ""
        m = json.dumps(r.metrics) if r.metrics else ""
        if len(m) > 60:
            m = m[:57] + "..."
        print(f"{r.name:<32} {r.status:<8} {r.duration_sec:>6.1f}  {v:<6}{promo:<6} {m}")
        if r.error:
            print(f"   ↳ {r.error}")
    print("=" * 78)
    print(f"summary: {_summary(reports)}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Quant X unified training runner")
    parser.add_argument("--only", help="comma-separated trainer names")
    parser.add_argument("--list", action="store_true", help="list discovered trainers and exit")
    parser.add_argument("--skip-gpu", action="store_true", help="skip GPU-only trainers (CPU run)")
    parser.add_argument("--force-cpu", action="store_true",
                        help="attempt GPU trainers on CPU anyway (laptop dev only)")
    parser.add_argument("--promote", action="store_true", help="promote new versions to prod on eval pass")
    parser.add_argument("--dry-run", action="store_true", help="train + evaluate but skip B2 upload + DB write")
    parser.add_argument("--out", help="artifact root dir (default: temp)")
    parser.add_argument("--json", action="store_true", help="machine-readable JSON report")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list:
        for t in discover_sorted():
            gpu = "GPU" if t.requires_gpu else "CPU"
            deps = (",".join(t.depends_on) if t.depends_on else "-")
            print(f"  {t.name:<32} {gpu}   deps={deps}")
        return 0

    only = [s for s in (args.only.split(",") if args.only else []) if s]
    out_root = Path(args.out) if args.out else None
    reports = run(
        only=only or None,
        skip_gpu=args.skip_gpu,
        promote=args.promote,
        out_root=out_root,
        dry_run=args.dry_run,
        force_cpu=args.force_cpu,
    )
    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=2, default=str))
    else:
        _print_report_human(reports)
    failed = sum(1 for r in reports if r.status == "failed")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
