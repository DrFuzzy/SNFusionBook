"""Analytic Hierarchy Process utilities, including incomplete comparisons.

Chapter 3, Section "Fusion based on Multi-Criteria Decision-Making":

* logarithmic least-squares (LLS) completion of an *incomplete* set of pairwise
  comparisons, eq. (lls-ahp), with the gauge ``sum_i y_i = 0`` (or ``y_1 = 0``)
  that removes the scaling ambiguity, followed by the normalisation
  ``x_i = exp(y_i) / sum_l exp(y_l)``;
* principal-eigenvector priorities of a complete reciprocal matrix;
* Saaty's consistency index and consistency ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Hashable, List, Sequence, Tuple

import numpy as np

Array = np.ndarray
Comparison = Tuple[Hashable, Hashable, float]   # (i, j, A_ij): "i is A_ij times as preferred as j"

#: Saaty's random consistency index RI(n) for n = 1 ... 10.
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


@dataclass
class LLSResult:
    """Output of :func:`incomplete_lls`."""

    labels: List[Hashable]      #: alternatives in index order
    log_priorities: Array       #: ``y*`` (gauge applied)
    priorities: Array           #: ``x* = exp(y*) / sum exp(y*)``
    residuals: Array            #: ``ln A_ij - (y_i - y_j)`` per observed comparison
    index: Dict[Hashable, int]  #: label -> index

    def fitted_ratio(self, i: Hashable, j: Hashable) -> float:
        """``exp(y_i - y_j)``: the ratio implied (or inferred) for the pair ``(i, j)``."""
        return float(np.exp(self.log_priorities[self.index[i]] - self.log_priorities[self.index[j]]))

    def completed_matrix(self) -> Array:
        """Fully consistent reciprocal matrix ``A_ij = x_i / x_j`` implied by the solution."""
        return completed_matrix(self.log_priorities)


def incomplete_lls(labels: Sequence[Hashable], comparisons: Sequence[Comparison],
                   gauge: str = "sum") -> LLSResult:
    """Solve the incomplete-AHP LLS problem, eq. (lls-ahp), for one or several experts.

    Parameters
    ----------
    labels : the ``r`` alternatives (or criteria) to rank.
    comparisons : ``(i, j, A_ij)`` tuples, possibly from several decision-makers;
                  the reciprocal ``A_ji = 1/A_ij`` is implicit.  The comparison
                  graph must be connected.
    gauge : ``"sum"`` imposes ``sum_i y_i = 0`` (the book's choice); ``"first"``
            imposes ``y_1 = 0``.  Both give the same normalised priorities.

    The problem ``min ||B y - b||^2`` with ``B`` the signed incidence matrix of the
    comparison graph is solved through the KKT system with the gauge as the
    single equality constraint, as in the authors' script.
    """
    labels = list(labels)
    index = {lab: k for k, lab in enumerate(labels)}
    n = len(labels)
    rows, target = [], []
    for left, right, ratio in comparisons:
        if ratio <= 0:
            raise ValueError("pairwise comparisons must be positive")
        row = np.zeros(n)
        row[index[left]] = 1.0
        row[index[right]] = -1.0
        rows.append(row)
        target.append(math.log(ratio))
    B = np.vstack(rows)
    b = np.array(target)

    normal = B.T @ B
    if gauge == "sum":
        g = np.ones((n, 1))
    elif gauge == "first":
        g = np.zeros((n, 1))
        g[0, 0] = 1.0
    else:
        raise ValueError("gauge must be 'sum' or 'first'")
    kkt = np.block([[normal, g], [g.T, np.zeros((1, 1))]])
    rhs = np.concatenate([B.T @ b, np.array([0.0])])
    try:
        sol = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError as exc:   # disconnected comparison graph
        raise ValueError("the comparison graph must be connected") from exc
    y = sol[:n]
    x = np.exp(y)
    x /= x.sum()
    return LLSResult(labels, y, x, b - B @ y, index)


def completed_matrix(log_priorities: Array) -> Array:
    """Consistent reciprocal matrix ``A_ij = exp(y_i - y_j)``."""
    y = np.asarray(log_priorities, dtype=float)
    return np.exp(y[:, None] - y[None, :])


def comparison_matrix(labels: Sequence[Hashable], comparisons: Sequence[Comparison]) -> Array:
    """Reciprocal matrix with ``nan`` where a comparison is missing."""
    index = {lab: k for k, lab in enumerate(labels)}
    n = len(labels)
    A = np.full((n, n), np.nan)
    np.fill_diagonal(A, 1.0)
    for i, j, a in comparisons:
        A[index[i], index[j]] = a
        A[index[j], index[i]] = 1.0 / a
    return A


def principal_eigenvector(A: Array) -> Tuple[Array, float]:
    """Priorities from the principal right eigenvector of a complete reciprocal matrix.

    Returns ``(w, lambda_max)`` with ``w`` normalised to sum to one.
    """
    A = np.asarray(A, dtype=float)
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    return w / w.sum(), float(vals[k].real)


def consistency_index(lambda_max: float, n: int) -> float:
    """Saaty's ``CI = (lambda_max - n) / (n - 1)``."""
    return 0.0 if n <= 2 else (lambda_max - n) / (n - 1)


def consistency_ratio(A: Array) -> float:
    """Saaty's consistency ratio ``CR = CI / RI(n)`` (``CR < 0.1`` is conventionally acceptable)."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n <= 2:
        return 0.0
    _, lam = principal_eigenvector(A)
    return consistency_index(lam, n) / RANDOM_INDEX.get(n, 1.49)


def geometric_mean_priorities(A: Array) -> Array:
    """Row geometric-mean priorities (the LLS solution of a complete matrix)."""
    A = np.asarray(A, dtype=float)
    g = np.exp(np.log(A).mean(axis=1))
    return g / g.sum()
