"""Track-level fusion rules and linear consensus iterations.

Chapter 3, Section "Track-Level Fusion and Data Association":

* naive information fusion of independent tracks, eq. (independent-track-fusion-state);
* fusion with known cross-covariance, eqs. (correlated-track-fusion-state)--(correlated-track-fusion-cov);
* covariance intersection (CI), eqs. (ci-cov)--(ci-weight), with the scalar
  optimisation of ``omega`` for the trace, determinant or log-determinant cost.

The linear consensus iteration ``x(t+1) = W x(t)`` (with step-size / Perron
weights ``W = I - eps L`` or Metropolis weights) is the building block of the
distributed and multi-agent fusion schemes of the later chapters.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

Array = np.ndarray


# -----------------------------------------------------------------------------
# Track-to-track fusion
# -----------------------------------------------------------------------------
def information_fusion(x1: Array, P1: Array, x2: Array, P2: Array) -> Tuple[Array, Array]:
    """Naive (independent-errors) fusion: ``P_f^-1 = P_1^-1 + P_2^-1`` and
    ``x_f = P_f (P_1^-1 x_1 + P_2^-1 x_2)``, eq. (independent-track-fusion-state).

    Double-counts shared information when the tracks are correlated.
    """
    I1, I2 = np.linalg.inv(P1), np.linalg.inv(P2)
    Pf = np.linalg.inv(I1 + I2)
    return Pf @ (I1 @ x1 + I2 @ x2), Pf


def correlated_fusion(x1: Array, P1: Array, x2: Array, P2: Array, P12: Array) -> Tuple[Array, Array]:
    """Minimum-variance fusion with known cross-covariance ``P_12``,
    eqs. (correlated-track-fusion-state)--(correlated-track-fusion-cov):
    ``W = (P_1 - P_12)(P_1 + P_2 - P_12 - P_12^T)^-1``.
    """
    W = (P1 - P12) @ np.linalg.inv(P1 + P2 - P12 - P12.T)
    xf = x1 + W @ (x2 - x1)
    Pf = P1 - W @ (P1 - P12.T)
    return xf, Pf


def _ci_cost(name: str) -> Callable[[Array], float]:
    if name == "trace":
        return lambda P: float(np.trace(P))
    if name == "det":
        return lambda P: float(np.linalg.det(P))
    if name == "logdet":
        return lambda P: float(np.linalg.slogdet(P)[1])
    raise ValueError("criterion must be 'trace', 'det' or 'logdet'")


def covariance_intersection(x1: Array, P1: Array, x2: Array, P2: Array,
                            omega: Optional[float] = None,
                            criterion: str = "trace") -> Tuple[Array, Array, float]:
    """Covariance intersection, eqs. (ci-cov)--(ci-est).

    ``P_CI^-1 = w P_1^-1 + (1-w) P_2^-1`` and
    ``x_CI = P_CI (w P_1^-1 x_1 + (1-w) P_2^-1 x_2)``.  When ``omega`` is ``None``
    it is chosen by eq. (ci-weight), minimising ``J(P_CI(w))`` over ``[0, 1]``
    (``J`` = trace, det or logdet).  Returns ``(x_CI, P_CI, omega)``.

    CI is consistent for *any* unknown cross-covariance: ``P_CI`` upper-bounds
    the true error covariance of ``x_CI`` in the Loewner order.
    """
    I1, I2 = np.linalg.inv(P1), np.linalg.inv(P2)

    def fuse(w: float) -> Tuple[Array, Array]:
        P = np.linalg.inv(w * I1 + (1.0 - w) * I2)
        return P @ (w * I1 @ x1 + (1.0 - w) * I2 @ x2), P

    if omega is None:
        J = _ci_cost(criterion)
        res = minimize_scalar(lambda w: J(fuse(w)[1]), bounds=(0.0, 1.0), method="bounded",
                              options={"xatol": 1e-10})
        omega = float(res.x)
        # the optimum may sit on the boundary: compare with the end points explicitly
        cands = [(J(fuse(w)[1]), w) for w in (0.0, omega, 1.0)]
        omega = min(cands)[1]
    x, P = fuse(float(omega))
    return x, P, float(omega)


# -----------------------------------------------------------------------------
# Linear consensus
# -----------------------------------------------------------------------------
def laplacian(adjacency: Array) -> Array:
    """Graph Laplacian ``L = D - A`` of a symmetric adjacency matrix."""
    A = np.asarray(adjacency, dtype=float)
    return np.diag(A.sum(axis=1)) - A


def step_size_weights(adjacency: Array, eps: Optional[float] = None) -> Array:
    """Perron-type consensus matrix ``W = I - eps L``.

    With ``eps < 1 / d_max`` the matrix is doubly stochastic with non-negative
    entries, so the iteration converges to the average on a connected graph.
    ``eps`` defaults to ``1 / (d_max + 1)``.
    """
    A = np.asarray(adjacency, dtype=float)
    if eps is None:
        eps = 1.0 / (A.sum(axis=1).max() + 1.0)
    return np.eye(A.shape[0]) - eps * laplacian(A)


def metropolis_weights(adjacency: Array) -> Array:
    """Metropolis--Hastings weights ``W_ij = 1 / (1 + max(d_i, d_j))`` for ``(i, j)`` an edge,
    ``W_ii = 1 - sum_j W_ij``: doubly stochastic, needs only local degree information."""
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    deg = A.sum(axis=1)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] > 0:
                W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
        W[i, i] = 1.0 - W[i].sum()
    return W


def consensus_step(x: Array, W: Array) -> Array:
    """One linear consensus iteration ``x(t+1) = W x(t)`` (``x`` may be ``(n,)`` or ``(n, d)``)."""
    return np.asarray(W) @ np.asarray(x, dtype=float)


def run_consensus(x0: Array, W: Array, n_iter: int, return_history: bool = False):
    """Iterate :func:`consensus_step` ``n_iter`` times; optionally return all iterates."""
    x = np.asarray(x0, dtype=float)
    hist = [x]
    for _ in range(n_iter):
        x = consensus_step(x, W)
        hist.append(x)
    return (x, np.array(hist)) if return_history else x


def is_doubly_stochastic(W: Array, tol: float = 1e-10) -> bool:
    W = np.asarray(W, dtype=float)
    return bool(np.all(W >= -tol) and np.allclose(W.sum(0), 1, atol=tol) and np.allclose(W.sum(1), 1, atol=tol))
