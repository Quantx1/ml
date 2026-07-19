"""Statistical jump model for market-regime detection.

Statistical jump models (Nystrup et al. 2020-21; Nystrup & Boyd's "greedy
online classification" line of work) fit K regime centroids to standardized
features and assign a state path that minimizes

    sum_t || z_t - c_{s_t} ||^2  +  lambda_jump * (# of state changes)

The jump penalty is an explicit, interpretable persistence knob — unlike an
HMM's transition matrix it does not depend on Gaussian emission assumptions,
which makes jump models markedly more robust to fat tails and misspecified
volatility (the empirical reason they outperform HMMs on equity regimes).

Design principles (see ``ml.regime.features`` module docstring): regimes are
LATENT — the model optimizes hindsight agreement, turning-point lag and
whipsaw, never claiming real-time certainty. STRICT NO-LOOKAHEAD in the
online path: ``fit(history)`` freezes centroids + scaling; ``predict_online``
assigns each new row greedily against the PREVIOUS state only, so the state
for day t can never change when future rows are appended (tested).

Pure numpy. In-sample fitting alternates (a) an exact dynamic-programming
state assignment (Viterbi-style over T x K with the jump penalty) and (b)
centroid updates as state means, until the path stops changing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

_EPS = 1e-12


def _dp_assign(cost: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Exact minimum-cost state path for ``cost[t, k]`` + ``lam`` per jump.

    Standard O(T*K) recursion: min over the previous state is either "stay"
    (no penalty) or "jump from the globally cheapest previous state" (+lam);
    since lam >= 0, staying always beats jumping to yourself, so the global
    argmin shortcut is exact. Returns (path, total_objective).
    """
    T, K = cost.shape
    V = np.empty((T, K))
    ptr = np.zeros((T, K), dtype=np.int64)
    V[0] = cost[0]
    ks = np.arange(K)
    for t in range(1, T):
        prev = V[t - 1]
        j = int(np.argmin(prev))
        jump = prev[j] + lam
        stay_wins = prev <= jump
        V[t] = np.where(stay_wins, prev, jump) + cost[t]
        ptr[t] = np.where(stay_wins, ks, j)
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(np.argmin(V[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = ptr[t, path[t]]
    return path, float(V[-1].min())


class JumpModel:
    """K-state statistical jump model (Nystrup et al. 2020-21).

    Args:
        n_states: number of regimes K (default 3: bear/sideways/bull).
        lambda_jump: jump penalty — larger values force more persistent
            regimes (fewer switches, more lag). Expressed in units of squared
            standardized-feature distance.
        max_iter: coordinate-descent iterations (assign / re-center).
        n_init: restarts (1 deterministic quantile init + n_init-1 random);
            the fit with the lowest objective wins.
        random_state: seed for the random restarts.

    Attributes (after ``fit``):
        centroids_: (K, D) state centroids in standardized space, RELABELED so
            that index 0 has the lowest mean of feature 0 and index K-1 the
            highest. With ``ret_21d`` as feature 0 this makes labels
            deterministic: 0 = bear, 1 = sideways, 2 = bull.
        states_: in-sample state path (already relabeled).
        state_order_: permutation applied — ``state_order_[i]`` is the raw
            fitted state that became public label ``i``.
        objective_: final in-sample objective value.
    """

    def __init__(
        self,
        n_states: int = 3,
        lambda_jump: float = 50.0,
        max_iter: int = 30,
        n_init: int = 8,
        random_state: int = 0,
        relabel_idx: int = 0,
    ) -> None:
        self.n_states = int(n_states)
        self.lambda_jump = float(lambda_jump)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.random_state = int(random_state)
        self.relabel_idx = int(relabel_idx)

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray) -> "JumpModel":
        """Fit centroids + in-sample path on history ``X`` (T, D).

        Standardization stats come from THIS fit window only and are frozen
        for the online path (walk-forward refits re-estimate them — no
        lookahead into serving data).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < self.n_states:
            raise ValueError(f"X must be 2-D with >= {self.n_states} rows, got {X.shape}")
        self.mu_ = X.mean(axis=0)
        self.sigma_ = X.std(axis=0)
        self.sigma_[self.sigma_ < _EPS] = 1.0
        Z = (X - self.mu_) / self.sigma_

        rng = np.random.default_rng(self.random_state)
        inits = [self._quantile_init(Z)]
        for _ in range(max(0, self.n_init - 1)):
            inits.append(Z[rng.choice(len(Z), size=self.n_states, replace=False)])

        best: Optional[tuple[float, np.ndarray, np.ndarray]] = None
        for centroids in inits:
            centroids = centroids.copy()
            prev_path: Optional[np.ndarray] = None
            for _ in range(self.max_iter):
                cost = ((Z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
                path, obj = _dp_assign(cost, self.lambda_jump)
                if prev_path is not None and np.array_equal(path, prev_path):
                    break
                prev_path = path
                for k in range(self.n_states):
                    mask = path == k
                    if mask.any():  # empty state -> keep old centroid
                        centroids[k] = Z[mask].mean(axis=0)
            if best is None or obj < best[0]:
                best = (obj, centroids, path)

        assert best is not None
        obj, centroids, path = best
        # Relabel deterministically by ascending mean of feature 0 (ret_21d):
        # 0 = bear, ..., K-1 = bull. Standardization is a positive affine map,
        # so ordering centroid[:, 0] orders the raw feature means identically.
        order = np.argsort(centroids[:, self.relabel_idx], kind="stable")
        relabel = np.empty(self.n_states, dtype=np.int64)
        relabel[order] = np.arange(self.n_states)
        self.state_order_ = order
        self.centroids_ = centroids[order]
        self.states_ = relabel[path]
        self.objective_ = obj
        return self

    # --------------------------------------------------------------- online
    def predict_online(self, X_new: np.ndarray, prev_state: Optional[int] = None) -> np.ndarray:
        """Filtered (causal) state assignment for new rows — NO LOOKAHEAD.

        Runs the jump-model DP recursion FORWARD only (no backtracking),
        against frozen centroids, charging the jump penalty against the last
        state of the running best path:

            V_t[k] = min(V_{t-1}[k], min_j V_{t-1}[j] + lambda_jump)
                     + ||z_t - c_k||^2
            s_t    = argmin_k V_t[k]

        This is the filtered MAP state under the jump objective. Unlike a
        greedy one-step rule (which can never amortize the jump penalty and
        therefore freezes in its initial state), the running cost lets
        evidence for a new regime ACCUMULATE until it outweighs lambda_jump —
        persistence with bounded lag (~lambda / per-day distance gap, in
        days). V_t depends only on rows <= t, so appending future rows never
        changes earlier states (truncation-invariance is tested).

        ``prev_state`` seeds the recursion (e.g. the last in-sample state,
        penalizing an immediate departure from it); when None, the first row
        is costed by distance alone.
        """
        if not hasattr(self, "centroids_"):
            raise RuntimeError("JumpModel.predict_online called before fit()")
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        Z = (X_new - self.mu_) / self.sigma_
        ks = np.arange(self.n_states)
        out = np.empty(len(Z), dtype=np.int64)
        V: Optional[np.ndarray] = None
        for t, z in enumerate(Z):
            c = ((z - self.centroids_) ** 2).sum(axis=1)
            if V is None:
                V = c.copy()
                if prev_state is not None:
                    V += self.lambda_jump * (ks != int(prev_state))
            else:
                V = np.minimum(V, V.min() + self.lambda_jump) + c
            out[t] = int(np.argmin(V))
            V -= V.min()  # renormalize — argmin-invariant, avoids float drift
        return out

    # ------------------------------------------------------------- helpers
    def _quantile_init(self, Z: np.ndarray) -> np.ndarray:
        """Deterministic init: rows nearest the K quantiles of feature 0 —
        a sensible regime spread (low / mid / high mean return)."""
        qs = np.quantile(Z[:, 0], np.linspace(0.1, 0.9, self.n_states))
        idx = [int(np.argmin(np.abs(Z[:, 0] - q))) for q in qs]
        return Z[idx].copy()


__all__ = ["JumpModel"]
