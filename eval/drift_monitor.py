"""
Production drift monitoring + auto-rollback (Phase 1.6).

Real-money models without monitoring fail silently. This is the
piece that catches model decay BEFORE users lose capital.

Daily 16:00 IST scheduler runs ``run_daily_drift_check()`` which:

  1. Reads every prod model from ``model_versions`` (is_prod=true)
  2. Reads its rolling 30-day live performance from
     ``model_rolling_performance``
  3. Compares live Sharpe vs backtest Sharpe (from training metrics)
  4. If degraded > 50 percent: log + alert (no action)
  5. If degraded > 70 percent: auto-demote (is_prod=false), log,
     page admin, fall back to previous prod version

Champion-challenger:
  A retrained model deploys as ``is_shadow=true`` first. After 30
  days of shadow data, the auto-promote check fires:
    - shadow Sharpe > champion Sharpe + 0.3 → auto-promote
    - shadow Sharpe < champion Sharpe - 0.5 → auto-retire
    - else hold both for another 30 days

This module is the ML layer; the schedule + alert plumbing lives in
``backend/services/scheduler.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Thresholds — calibrated for real-money risk tolerance.
SOFT_DRIFT_RATIO = 0.50           # 50 percent of backtest sharpe → soft alert
HARD_DRIFT_RATIO = 0.30           # 30 percent of backtest sharpe → auto-demote
MIN_LIVE_DAYS_FOR_DEMOTE = 21     # need 21 live trading days before auto-demote fires
MIN_BACKTEST_SHARPE_FOR_DEMOTE = 0.5  # don't auto-demote a low-bar model on tiny moves
CHAMPION_CHALLENGER_WINDOW_DAYS = 30  # shadow eval period
CHALLENGER_PROMOTE_DELTA = 0.30       # shadow Sharpe must beat champion by 0.30
CHALLENGER_RETIRE_DELTA = -0.50       # shadow Sharpe < champion - 0.50 → retire


@dataclass
class DriftFinding:
    """One row of the drift report — one model, one decision."""

    model_name: str
    version: int
    backtest_sharpe: float
    live_sharpe: Optional[float]
    live_n_days: int
    drift_ratio: Optional[float]    # live / backtest (lower = worse)
    severity: str                    # 'ok' | 'soft' | 'hard'
    action: str                      # 'none' | 'alert' | 'demote'
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "backtest_sharpe": round(self.backtest_sharpe, 4),
            "live_sharpe": round(self.live_sharpe, 4) if self.live_sharpe is not None else None,
            "live_n_days": self.live_n_days,
            "drift_ratio": round(self.drift_ratio, 4) if self.drift_ratio is not None else None,
            "severity": self.severity,
            "action": self.action,
            "reason": self.reason,
        }


@dataclass
class ChampionChallengerDecision:
    """Verdict on a shadow model after the eval window."""

    model_name: str
    champion_version: int
    challenger_version: int
    champion_sharpe: float
    challenger_sharpe: float
    delta: float
    decision: str                    # 'promote' | 'retire' | 'hold'
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "champion_version": self.champion_version,
            "challenger_version": self.challenger_version,
            "champion_sharpe": round(self.champion_sharpe, 4),
            "challenger_sharpe": round(self.challenger_sharpe, 4),
            "delta": round(self.delta, 4),
            "decision": self.decision,
            "reason": self.reason,
        }


@dataclass
class DriftReport:
    """Full daily report emitted by run_daily_drift_check()."""

    computed_at: str
    n_prod_models: int
    findings: List[DriftFinding] = field(default_factory=list)
    champion_challenger: List[ChampionChallengerDecision] = field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "computed_at": self.computed_at,
            "n_prod_models": self.n_prod_models,
            "findings": [f.to_dict() for f in self.findings],
            "champion_challenger": [d.to_dict() for d in self.champion_challenger],
            "actions_taken": self.actions_taken,
            "errors": self.errors,
        }


def _extract_backtest_sharpe(metrics: Dict[str, Any]) -> Optional[float]:
    """Pull the 'right' Sharpe from a trainer's metrics dict.

    Prefer holdout_sharpe (held-out final test) → sharpe_mean (walk-forward
    pooled) → sharpe (single-fold). Returns None if no Sharpe-like metric
    exists (e.g. regime_hmm uses log-likelihood as primary).
    """
    for k in ("holdout_sharpe", "sharpe_mean", "sharpe"):
        v = metrics.get(k)
        if isinstance(v, (int, float)) and v != 0:
            return float(v)
    return None


def _assess_drift(
    backtest_sharpe: float,
    live_sharpe: Optional[float],
    live_n_days: int,
) -> Tuple[str, str, str, Optional[float]]:
    """Return (severity, action, reason, drift_ratio)."""
    if live_sharpe is None:
        return "ok", "none", "no live data yet", None
    if live_n_days < MIN_LIVE_DAYS_FOR_DEMOTE:
        ratio = live_sharpe / max(backtest_sharpe, 1e-3)
        return ("ok", "none",
                f"n_live={live_n_days} < {MIN_LIVE_DAYS_FOR_DEMOTE} min for demote",
                ratio)

    if backtest_sharpe < MIN_BACKTEST_SHARPE_FOR_DEMOTE:
        return ("ok", "none",
                f"backtest_sharpe {backtest_sharpe:.2f} below {MIN_BACKTEST_SHARPE_FOR_DEMOTE} — "
                "model wasn't strong to begin with, drift undefined", None)

    ratio = live_sharpe / backtest_sharpe
    if ratio < HARD_DRIFT_RATIO:
        return ("hard", "demote",
                f"live Sharpe {live_sharpe:.2f} is {ratio*100:.0f}% of backtest "
                f"{backtest_sharpe:.2f} — below {HARD_DRIFT_RATIO*100:.0f}% demote threshold",
                ratio)
    if ratio < SOFT_DRIFT_RATIO:
        return ("soft", "alert",
                f"live Sharpe {live_sharpe:.2f} is {ratio*100:.0f}% of backtest "
                f"{backtest_sharpe:.2f} — below {SOFT_DRIFT_RATIO*100:.0f}% soft floor",
                ratio)
    return ("ok", "none",
            f"live Sharpe {live_sharpe:.2f} is {ratio*100:.0f}% of backtest "
            f"{backtest_sharpe:.2f} — healthy",
            ratio)


def _assess_challenger(
    champion_sharpe: float,
    challenger_sharpe: float,
    challenger_days_live: int,
) -> Tuple[str, str]:
    """Return (decision, reason) for a shadow model after eval window."""
    if challenger_days_live < CHAMPION_CHALLENGER_WINDOW_DAYS:
        return ("hold",
                f"only {challenger_days_live} days of shadow data, need {CHAMPION_CHALLENGER_WINDOW_DAYS}")
    delta = challenger_sharpe - champion_sharpe
    if delta >= CHALLENGER_PROMOTE_DELTA:
        return ("promote",
                f"challenger {challenger_sharpe:.2f} beats champion {champion_sharpe:.2f} "
                f"by {delta:+.2f} (threshold +{CHALLENGER_PROMOTE_DELTA})")
    if delta <= CHALLENGER_RETIRE_DELTA:
        return ("retire",
                f"challenger {challenger_sharpe:.2f} trails champion {champion_sharpe:.2f} "
                f"by {delta:+.2f} (threshold {CHALLENGER_RETIRE_DELTA})")
    return ("hold",
            f"delta {delta:+.2f} within [{CHALLENGER_RETIRE_DELTA}, +{CHALLENGER_PROMOTE_DELTA}] "
            f"hold zone — continue shadow")


def _live_sharpe_from_rolling(
    rolling_rows: List[Dict[str, Any]],
) -> Tuple[Optional[float], int]:
    """Pull the most-recent 30-day rolling Sharpe + day count.

    ``rolling_rows`` comes from ``model_rolling_performance`` filtered
    to the target model. Newest-first ordering expected.
    """
    if not rolling_rows:
        return None, 0
    # Prefer the 30-day window row if present; otherwise newest available.
    win30 = next((r for r in rolling_rows if r.get("window_days") == 30), None)
    target = win30 or rolling_rows[0]
    sharpe = target.get("sharpe_ratio")
    n_days = int(target.get("signal_count") or 0)
    return (float(sharpe) if isinstance(sharpe, (int, float)) else None, n_days)


def assess_one_model(
    model_name: str,
    version: int,
    metrics: Dict[str, Any],
    rolling_rows: List[Dict[str, Any]],
) -> DriftFinding:
    """Pure function — no DB writes. The orchestrator handles persistence."""
    bt = _extract_backtest_sharpe(metrics)
    if bt is None:
        return DriftFinding(
            model_name=model_name, version=version,
            backtest_sharpe=0.0, live_sharpe=None, live_n_days=0,
            drift_ratio=None, severity="ok", action="none",
            reason="no Sharpe-like backtest metric (likely a forecaster, not a trader)",
        )
    live, n_days = _live_sharpe_from_rolling(rolling_rows)
    severity, action, reason, ratio = _assess_drift(bt, live, n_days)
    return DriftFinding(
        model_name=model_name, version=version,
        backtest_sharpe=bt, live_sharpe=live, live_n_days=n_days,
        drift_ratio=ratio, severity=severity, action=action, reason=reason,
    )


def assess_challenger(
    model_name: str,
    champion_version: int,
    challenger_version: int,
    champion_rolling: List[Dict[str, Any]],
    challenger_rolling: List[Dict[str, Any]],
) -> ChampionChallengerDecision:
    """Champion-challenger verdict from rolling performance rows."""
    champ_sharpe, _ = _live_sharpe_from_rolling(champion_rolling)
    chall_sharpe, chall_days = _live_sharpe_from_rolling(challenger_rolling)
    if champ_sharpe is None or chall_sharpe is None:
        return ChampionChallengerDecision(
            model_name=model_name,
            champion_version=champion_version,
            challenger_version=challenger_version,
            champion_sharpe=champ_sharpe or 0.0,
            challenger_sharpe=chall_sharpe or 0.0,
            delta=0.0,
            decision="hold",
            reason="insufficient rolling data on champion or challenger",
        )
    decision, reason = _assess_challenger(champ_sharpe, chall_sharpe, chall_days)
    return ChampionChallengerDecision(
        model_name=model_name,
        champion_version=champion_version,
        challenger_version=challenger_version,
        champion_sharpe=champ_sharpe,
        challenger_sharpe=chall_sharpe,
        delta=chall_sharpe - champ_sharpe,
        decision=decision,
        reason=reason,
    )


def run_daily_drift_check(
    *,
    supabase_admin,
    dry_run: bool = False,
) -> DriftReport:
    """Daily 16:00 IST orchestrator.

    Reads prod + shadow model_versions, reads rolling performance,
    assesses each, writes findings to ``model_drift_log``, demotes
    hard-drift models, fires alerts on soft-drift.

    Returns a DriftReport for the admin dashboard.
    """
    report = DriftReport(
        computed_at=datetime.now(timezone.utc).isoformat(),
        n_prod_models=0,
    )

    try:
        # 1. Read every prod + shadow model
        prod_rows = (
            supabase_admin.table("model_versions")
            .select("model_name, version, is_prod, is_shadow, metrics, trained_at")
            .or_("is_prod.eq.true,is_shadow.eq.true")
            .eq("is_retired", False)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        report.errors.append(f"prod model_versions read failed: {exc}")
        return report

    prod_by_name: Dict[str, Dict[str, Any]] = {}
    shadow_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for r in prod_rows:
        if r.get("is_prod"):
            prod_by_name[r["model_name"]] = r
        if r.get("is_shadow"):
            shadow_by_name.setdefault(r["model_name"], []).append(r)
    report.n_prod_models = len(prod_by_name)

    # 2. For each prod model, fetch rolling perf + assess
    for name, row in prod_by_name.items():
        try:
            rolling_rows = (
                supabase_admin.table("model_rolling_performance")
                .select("window_days, sharpe_ratio, signal_count, win_rate, computed_at")
                .eq("model_name", name)
                .order("computed_at", desc=True)
                .limit(10)
                .execute()
                .data
                or []
            )
        except Exception as exc:
            report.errors.append(f"rolling perf read failed for {name}: {exc}")
            continue
        finding = assess_one_model(
            model_name=name,
            version=int(row.get("version") or 0),
            metrics=row.get("metrics") or {},
            rolling_rows=rolling_rows,
        )
        report.findings.append(finding)

        # 3. Execute action
        if finding.action == "demote" and not dry_run:
            try:
                supabase_admin.table("model_versions").update({
                    "is_prod": False,
                    "is_retired": True,
                    "notes": f"AUTO-DEMOTE {datetime.now(timezone.utc).isoformat()}: {finding.reason}",
                }).eq("model_name", name).eq("version", finding.version).execute()
                report.actions_taken.append({
                    "action": "demote",
                    "model_name": name,
                    "version": finding.version,
                    "reason": finding.reason,
                })
                logger.warning(
                    "AUTO-DEMOTED %s v%d — %s",
                    name, finding.version, finding.reason,
                )
                # CRITICAL #3 (2026-05-31) — surface auto-demotes to ops.
                # Without this alert, drift goes silent and AutoPilot keeps
                # using a degraded model until someone notices manually.
                try:
                    # SCHEMA: type/message/data — verified 2026-05-31
                    supabase_admin.table("notifications").insert({
                        "user_id": None,    # admin broadcast (column is nullable)
                        "type": "cron_failed",
                        "priority": "critical",
                        "title": f"🚨 Model auto-demoted: {name} v{finding.version}",
                        "message": (
                            f"Drift monitor demoted {name} v{finding.version} "
                            f"(is_prod=false, is_retired=true). Reason: {finding.reason}. "
                            f"AutoPilot will fall back to the previous version. "
                            f"Investigate live IC + retrain ASAP."
                        ),
                        "channels": ["telegram", "email"],
                        "data": {
                            "model_name": name,
                            "version": finding.version,
                            "reason": finding.reason,
                            "live_sharpe": finding.live_sharpe,
                            "backtest_sharpe": finding.backtest_sharpe,
                            "url": "/admin/models",
                        },
                    }).execute()
                except Exception as alert_ex:
                    logger.debug("demote alert dispatch failed: %s", alert_ex)
            except Exception as exc:
                report.errors.append(f"demote {name} v{finding.version} failed: {exc}")

    # 4. Champion-challenger check for each shadow model
    for name, shadows in shadow_by_name.items():
        champion = prod_by_name.get(name)
        if champion is None:
            continue
        try:
            champion_rolling = (
                supabase_admin.table("model_rolling_performance")
                .select("window_days, sharpe_ratio, signal_count, computed_at")
                .eq("model_name", name)
                .order("computed_at", desc=True).limit(5).execute().data
                or []
            )
        except Exception:
            champion_rolling = []

        for shadow in shadows:
            try:
                challenger_rolling = (
                    supabase_admin.table("model_rolling_performance")
                    .select("window_days, sharpe_ratio, signal_count, computed_at")
                    .eq("model_name", f"{name}_shadow_v{shadow['version']}")
                    .order("computed_at", desc=True).limit(5).execute().data
                    or []
                )
            except Exception:
                challenger_rolling = []
            verdict = assess_challenger(
                model_name=name,
                champion_version=int(champion["version"]),
                challenger_version=int(shadow["version"]),
                champion_rolling=champion_rolling,
                challenger_rolling=challenger_rolling,
            )
            report.champion_challenger.append(verdict)

            if verdict.decision == "promote" and not dry_run:
                try:
                    # Promote shadow → prod, demote previous champion
                    supabase_admin.table("model_versions").update({
                        "is_prod": False,
                    }).eq("model_name", name).eq("version", verdict.champion_version).execute()
                    supabase_admin.table("model_versions").update({
                        "is_prod": True, "is_shadow": False,
                        "notes": f"AUTO-PROMOTE {datetime.now(timezone.utc).isoformat()}: {verdict.reason}",
                    }).eq("model_name", name).eq("version", verdict.challenger_version).execute()
                    report.actions_taken.append({"action": "promote", **verdict.to_dict()})
                    logger.info(
                        "AUTO-PROMOTED %s v%d (challenger) over v%d (champion) — %s",
                        name, verdict.challenger_version, verdict.champion_version, verdict.reason,
                    )
                except Exception as exc:
                    report.errors.append(f"promote {name} v{verdict.challenger_version}: {exc}")
            elif verdict.decision == "retire" and not dry_run:
                try:
                    supabase_admin.table("model_versions").update({
                        "is_shadow": False, "is_retired": True,
                        "notes": f"AUTO-RETIRE {datetime.now(timezone.utc).isoformat()}: {verdict.reason}",
                    }).eq("model_name", name).eq("version", verdict.challenger_version).execute()
                    report.actions_taken.append({"action": "retire", **verdict.to_dict()})
                except Exception as exc:
                    report.errors.append(f"retire {name} v{verdict.challenger_version}: {exc}")

    # 5. Persist the report to model_drift_log (if table exists)
    if not dry_run:
        try:
            supabase_admin.table("model_drift_log").insert({
                "computed_at": report.computed_at,
                "n_prod_models": report.n_prod_models,
                "findings": [f.to_dict() for f in report.findings],
                "champion_challenger": [d.to_dict() for d in report.champion_challenger],
                "actions_taken": report.actions_taken,
                "errors": report.errors,
            }).execute()
        except Exception as exc:
            # Table may not exist on first run — log + continue, dashboard
            # consumers can read the in-memory return value.
            logger.warning("model_drift_log insert failed (table may not exist yet): %s", exc)
            report.errors.append(f"drift log persist: {exc}")

    return report


__all__ = [
    "DriftFinding",
    "ChampionChallengerDecision",
    "DriftReport",
    "assess_one_model",
    "assess_challenger",
    "run_daily_drift_check",
    "SOFT_DRIFT_RATIO",
    "HARD_DRIFT_RATIO",
    "CHAMPION_CHALLENGER_WINDOW_DAYS",
]
