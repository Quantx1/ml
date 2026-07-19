"""
PR 163 / PR 176 — financial-ML labeling utilities.

Replaces naive sign-of-return labels (used by lgbm_signal_gate +
intraday_lstm) with the triple-barrier method from López de Prado's
"Advances in Financial Machine Learning" (chapter 3) and adds the
sample-weight uniqueness layer from chapter 4.

Public surface:

    from ml.labeling import (
        triple_barrier_labels,         # legacy: just labels
        triple_barrier_events,         # PR 176: labels + barrier-hit times
        TripleBarrierConfig,
        sample_weights_from_t1,        # PR 176: AFML Ch.4 weights
        average_uniqueness,
        num_concurrent_labels,
    )

    labels, t1 = triple_barrier_events(close, atr, cfg)
    weights = sample_weights_from_t1(t1, n=len(close))
"""

from .sample_weights import (
    average_uniqueness,
    num_concurrent_labels,
    sample_weights_from_t1,
    time_decay_weights,
)
from .triple_barrier import (
    TripleBarrierConfig,
    label_distribution,
    triple_barrier_events,
    triple_barrier_labels,
)

__all__ = [
    "TripleBarrierConfig",
    "average_uniqueness",
    "label_distribution",
    "num_concurrent_labels",
    "sample_weights_from_t1",
    "time_decay_weights",
    "triple_barrier_events",
    "triple_barrier_labels",
]
