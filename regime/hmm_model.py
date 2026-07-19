"""Multi-feature Gaussian HMM regime model with FILTERED (causal) inference.

Successor to the legacy prod ``ml.training.trainers.regime_hmm`` (which this
module deliberately does NOT touch): richer multi-feature inputs, robust
scaling, and — critically — filtered probabilities for the online path.

WHY FILTERED, NOT SMOOTHED: ``hmmlearn``'s ``predict``/``predict_proba`` run
the forward-BACKWARD algorithm, i.e. P(s_t | x_{1:T}) — the state at t is
informed by future observations, which is lookahead when used online. We
therefore run the forward pass ourselves from public model parameters
(``startprob_``, ``transmat_``, ``means_``/``covars_``) and normalize per
step, yielding the filtered posterior P(s_t | x_{1:t}). Appending future rows
never changes earlier filtered probabilities (truncation-invariant, tested).

Design principles (see ``ml.regime.features``): regimes are latent; this
model is one voter in ``ml.regime.ensemble`` — objectives are hindsight
agreement, low turning-point lag and no whipsaw, with STRICT NO-LOOKAHEAD:
``fit(history)`` freezes parameters + scaling; filtering at t uses info <= t.

States are relabeled deterministically by ascending state mean of feature 0
(``ret_21d`` by ensemble convention): 0 = bear, 1 = sideways, 2 = bull.
"""
from __future__ import annotations

import numpy as np

_TINY = 1e-300


class RegimeHMM:
    """Gaussian HMM (full covariance) over robust-scaled regime features.

    Args:
        n_states: number of regimes K (default 3).
        n_iter: EM iterations for ``hmmlearn``.
        covariance_type: passed to ``GaussianHMM`` (default "full").
        random_state: EM init seed.

    Attributes (after ``fit``):
        hmm_: the fitted ``hmmlearn`` model (raw state indexing).
        state_order_: raw state indices sorted by ascending mean of feature 0
            — ``state_order_[i]`` is the raw state behind public label ``i``
            (0 = bear, ..., K-1 = bull).
        center_, scale_: robust-scaling stats (median, IQR/1.349) frozen from
            the fit window.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        covariance_type: str = "full",
        random_state: int = 7,
        relabel_idx: int = 0,
    ) -> None:
        self.n_states = int(n_states)
        self.n_iter = int(n_iter)
        self.covariance_type = covariance_type
        self.random_state = int(random_state)
        self.relabel_idx = int(relabel_idx)

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray) -> "RegimeHMM":
        """Fit on history ``X`` (T, D). Scaling stats come from this window
        only and are frozen for online filtering (walk-forward refits
        re-estimate them)."""
        from hmmlearn.hmm import GaussianHMM  # lazy: keep import cost off hot paths

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < 5 * self.n_states:
            raise ValueError(f"X must be 2-D with >= {5 * self.n_states} rows, got {X.shape}")
        # Robust scaling: median / (IQR / 1.349) ~ unit std under normality,
        # but insensitive to the fat tails that wreck mean/std scaling in
        # crash regimes (exactly when regime detection matters most).
        self.center_ = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        self.scale_ = iqr / 1.349
        self.scale_[self.scale_ < 1e-12] = 1.0
        Z = (X - self.center_) / self.scale_

        self.hmm_ = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        ).fit(Z)
        # Deterministic bear/sideways/bull relabeling by state mean return.
        self.state_order_ = np.argsort(self.hmm_.means_[:, self.relabel_idx], kind="stable")
        return self

    # --------------------------------------------------------------- online
    def filtered_probs(self, X: np.ndarray) -> np.ndarray:
        """Causal filtered posteriors P(s_t | x_{1:t}) — (T, K), columns in
        public label order (0 = bear, ..., K-1 = bull).

        Forward algorithm in log space:
            log a_0 = log pi + log b_0                     (then normalize)
            log a_t = logsumexp_j(log a_{t-1,j} + log A_{j,k}) + log b_{t,k}
        with per-step normalization so each row is the filtered posterior.
        Strictly causal — never uses ``hmmlearn.predict_proba`` (smoothed).
        """
        from scipy.special import logsumexp  # noqa: PLC0415

        if not hasattr(self, "hmm_"):
            raise RuntimeError("RegimeHMM.filtered_probs called before fit()")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Z = (X - self.center_) / self.scale_
        log_b = self._state_loglik(Z)  # (T, K)
        log_pi = np.log(self.hmm_.startprob_ + _TINY)
        log_a_mat = np.log(self.hmm_.transmat_ + _TINY)

        out = np.empty_like(log_b)
        la = log_pi + log_b[0]
        la -= logsumexp(la)
        out[0] = np.exp(la)
        for t in range(1, len(Z)):
            la = logsumexp(la[:, None] + log_a_mat, axis=0) + log_b[t]
            la -= logsumexp(la)
            out[t] = np.exp(la)
        return out[:, self.state_order_]

    def predict_online(self, X: np.ndarray) -> np.ndarray:
        """Filtered MAP state per row (argmax of ``filtered_probs``) in public
        labels: 0 = bear, 1 = sideways, 2 = bull. Causal — no lookahead."""
        return np.argmax(self.filtered_probs(X), axis=1)

    # ------------------------------------------------------------- helpers
    def _state_loglik(self, Z: np.ndarray) -> np.ndarray:
        """(T, K) per-state Gaussian log-likelihoods from public params.

        ``hmmlearn``'s ``covars_`` getter returns full (K, D, D) matrices for
        every covariance_type, so this works uniformly. ``allow_singular``
        guards degenerate fits on near-deterministic inputs.
        """
        from scipy.stats import multivariate_normal  # noqa: PLC0415

        cols = [
            np.atleast_1d(
                multivariate_normal.logpdf(
                    Z, mean=self.hmm_.means_[k], cov=self.hmm_.covars_[k], allow_singular=True
                )
            )
            for k in range(self.n_states)
        ]
        return np.column_stack(cols)


__all__ = ["RegimeHMM"]
