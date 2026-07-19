"""
PR 128 — Trainer protocol.

Every ML feature in the v1 plan registers a Trainer subclass here. The
runner discovers them via ``discover_trainers()`` and orchestrates a
single E2E pipeline run.

Contract:
    name              — unique slug, also used as ``model_versions.model_name``
    requires_gpu      — set True for RL / TFT-scale training; the runner
                        skips on CPU when this is True (unless --force-cpu)
    train(out_dir)    — produce artifact files in ``out_dir``; return
                        list of paths + a metrics dict
    evaluate(...)     — optional walk-forward / OOS metrics; return dict
    register(...)     — default impl uploads to B2 + writes
                        ``model_versions`` row via ModelRegistry
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ml.training.specs import EngineSpec

logger = logging.getLogger(__name__)

_run_pipeline = None  # module-level seam so tests can monkeypatch the spine


class TrainerError(RuntimeError):
    """Raised when a trainer cannot complete (data missing, GPU OOM, etc.)."""


@dataclass
class TrainResult:
    """Result returned by ``Trainer.train()`` and threaded through eval+register."""

    artifacts: List[Path]
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None


class Trainer(abc.ABC):
    """Base class every ML trainer module subclasses.

    Trainers are pure: ``train()`` writes artifacts under ``out_dir``,
    returns a TrainResult, and never touches the registry directly.
    The runner handles registration so a single eval+promote policy
    applies across every model.
    """

    #: Unique slug. Must match ``model_versions.model_name``.
    name: str = ""
    #: Skip this trainer when no GPU is available. RL + TFT-scale.
    requires_gpu: bool = False
    #: Optional list of other trainer names that must run first
    #: (e.g. the F2 ensemble waits on regime_hmm so it picks the
    #: just-trained version when re-evaluating).
    depends_on: List[str] = []

    @abc.abstractmethod
    def train(self, out_dir: Path) -> TrainResult:
        """Train the model and write artifact files into ``out_dir``."""

    def evaluate(self, result: TrainResult) -> Dict[str, Any]:
        """Walk-forward / OOS evaluation. Override per model. Default: noop."""
        return dict(result.metrics)

    def serve_smoke(self, out_dir: Path) -> tuple[bool, str]:
        """Round-trip the freshly-trained artifact through the SERVING feature
        contract before it may be promoted (closes the audit's train/serve-skew
        finding). Default: pass — trainers opt in by overriding (see
        ml.training.serve_smoke). Returning False blocks is_prod, never the run.
        """
        return True, "no serve-smoke defined"

    def register(
        self,
        result: TrainResult,
        eval_metrics: Dict[str, Any],
        *,
        trained_by: Optional[str] = None,
        git_sha: Optional[str] = None,
        promote: bool = False,
    ) -> Dict[str, Any]:
        """Upload artifacts and write a ``model_versions`` row.

        ``promote=True`` flips the new row to prod (and demotes the prior
        prod row inside ModelRegistry.promote — the runner only sets this
        when eval gates pass).
        """
        # Lazy import — base.py shouldn't pull the backend tree when used
        # by training scripts that just want to instantiate trainers.
        from backend.ai.registry import get_registry

        reg = get_registry()
        merged_metrics = {**result.metrics, **eval_metrics}
        row = reg.register(
            self.name,
            result.artifacts,
            metrics=merged_metrics,
            trained_by=trained_by,
            git_sha=git_sha,
            notes=result.notes,
        )
        if promote:
            row = reg.promote(self.name, int(row["version"]))
        return row


class PipelineTrainer(Trainer):
    """A Trainer whose train() delegates to the canonical 9-stage spine.

    Subclasses provide a declarative EngineSpec + the data/model hooks; the
    spine (ml.training.pipeline.run_pipeline) runs EDA, quality, purged-CV,
    HPO, evaluation, and report in fixed order. Default hooks raise — an engine
    MUST implement build_features/build_labels/make_model. fit_args/search_space
    are optional (ranking engines override fit_args to pass LightGBM `group`).
    """

    def engine_spec(self) -> "EngineSpec":
        raise NotImplementedError("engine_spec() must return an EngineSpec")

    def load_panel(self):
        """Return a tidy long OHLCV panel ['date','symbol',ohlcv]."""
        raise NotImplementedError("load_panel() must return an OHLCV panel")

    def build_features(self, panel):
        """panel -> (feats_df['date','symbol',*feature_cols], feature_cols)."""
        raise NotImplementedError

    def build_labels(self, panel):
        """panel -> labels_df['date','symbol', spec.label_col, spec.fwd_return_col]."""
        raise NotImplementedError

    def make_model(self, params: Dict[str, Any]):
        """Return a FRESH estimator built from hyperparams (LGBMRanker etc.)."""
        raise NotImplementedError

    def fit_args(self, df_tr) -> Dict[str, Any]:
        """Extra kwargs for model.fit (e.g. LightGBM ranking `group`)."""
        return {}

    def search_space(self):
        """Optional ml.training.optuna_search.SearchSpace for HPO. None = none."""
        return None

    def train(self, out_dir: Path) -> TrainResult:
        fn = _run_pipeline
        if fn is None:
            from ml.training.pipeline import run_pipeline as fn  # noqa: PLC0415
        return fn(self, out_dir)
