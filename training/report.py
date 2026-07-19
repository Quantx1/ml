"""Stage-9 training report: report.json + report.md + PNG plots.

report.json = the full metrics dict (verbatim, JSONB-safe). report.md = a
human one-pager (headline metrics + per-fold + a 'shippable?' verdict). PNGs
(rank-IC by fold, top feature importances) are best-effort — if matplotlib is
unavailable the report still writes, just without plots.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _verdict(m: Dict[str, Any], model_name: str) -> str:
    q = m.get(f"{model_name}_quality_pass")
    if q is True:
        return "SHIPPABLE — passed the quality gate."
    if q is False:
        return f"NOT shippable — {m.get(f'{model_name}_quality_reason', 'quality gate failed')}"
    return "UNKNOWN — no quality gate recorded."


def _plots(m: Dict[str, Any], out_dir: Path) -> List[Path]:
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        logger.info("matplotlib unavailable — skipping report PNGs: %s", exc)
        return []
    paths: List[Path] = []
    ic = m.get("rank_ic_per_fold") or []
    if ic:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(range(len(ic)), ic); ax.axhline(0, color="k", lw=0.5)
        ax.set_title("rank-IC by fold"); ax.set_xlabel("fold"); ax.set_ylabel("rank-IC")
        p = out_dir / "rank_ic_by_fold.png"; fig.tight_layout(); fig.savefig(p, dpi=90); plt.close(fig)
        paths.append(p)
    fi = m.get("feature_importance") or {}
    if fi:
        top = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)[:20]
        fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(top))))
        ax.barh([k for k, _ in reversed(top)], [v for _, v in reversed(top)])
        ax.set_title("top feature importance (gain)")
        p = out_dir / "feature_importance.png"; fig.tight_layout(); fig.savefig(p, dpi=90); plt.close(fig)
        paths.append(p)
    return paths


def write_report(metrics: Dict[str, Any], out_dir: Path, *, model_name: str) -> List[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    rj = out_dir / "report.json"
    rj.write_text(json.dumps(metrics, indent=2, default=str)); paths.append(rj)

    lines = [
        f"# Training report — {model_name}", "",
        f"**Verdict:** {_verdict(metrics, model_name)}", "",
        "## Headline metrics", "",
        "| metric | value |", "| --- | --- |",
    ]
    for k in ("rank_ic_mean", "rank_icir", "decile_spread_mean", "deflated_sharpe",
              "probability_backtest_overfitting", "n_features", "n_folds", "n_rows",
              "n_symbols", "n_dates", "train_seconds"):
        if k in metrics:
            lines.append(f"| {k} | {metrics[k]} |")
    if metrics.get("rank_ic_per_fold"):
        lines += ["", "## Per-fold rank-IC", "", f"`{metrics['rank_ic_per_fold']}`"]
    fi = metrics.get("feature_importance") or {}
    if fi:
        top = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)[:15]
        lines += ["", "## Top features (gain)", "", "| feature | gain |", "| --- | --- |"]
        lines += [f"| {k} | {round(float(v), 1)} |" for k, v in top]
    rm = out_dir / "report.md"; rm.write_text("\n".join(lines) + "\n"); paths.append(rm)

    paths.extend(_plots(metrics, out_dir))
    return paths
