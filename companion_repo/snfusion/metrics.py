"""Estimation-error and filter-consistency metrics.

Chapter 3, "Bayesian Filtering", filter-consistency paragraph: the normalised
innovation squared (NIS), eq. (nis), the normalised estimation error squared
(NEES) and their chi-square acceptance intervals, plus the RMSE used to quote
the results of Examples ex:ekf-radar-cutin and ex:ukf-radar-cutin.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import chi2

Array = np.ndarray


def rmse(estimate: Array, truth: Array, axis=None) -> float:
    """Root-mean-square error ``sqrt(mean((estimate - truth)^2))``."""
    e = np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float)
    return np.sqrt(np.mean(e ** 2, axis=axis))


def position_rmse(est_xy: Array, true_xy: Array) -> float:
    """RMSE of the Euclidean position error (``hypot`` of the per-axis errors)."""
    d = np.hypot(est_xy[:, 0] - true_xy[:, 0], est_xy[:, 1] - true_xy[:, 1])
    return float(np.sqrt(np.mean(d ** 2)))


def nis(innovation: Array, S: Array) -> float:
    """Normalised innovation squared ``nu^T S^-1 nu``, eq. (nis); ``~ chi^2_{n_y}`` if consistent."""
    nu = np.asarray(innovation, dtype=float)
    return float(nu @ np.linalg.solve(S, nu))


def nees(error: Array, P: Array) -> float:
    """Normalised estimation error squared ``e^T P^-1 e``; ``~ chi^2_{n_x}`` if consistent."""
    e = np.asarray(error, dtype=float)
    return float(e @ np.linalg.solve(P, e))


def chi2_bounds(dof: int, prob: float = 0.95, two_sided: bool = True) -> Tuple[float, float]:
    """Acceptance interval of ``chi^2_dof`` containing probability ``prob``.

    Two-sided (default) returns ``(chi2_{(1-p)/2}, chi2_{(1+p)/2})``; e.g. for
    ``dof = 3, prob = 0.95`` this is ``(0.216, 9.348)``, the band drawn in
    Fig. fig:ekf-radar-cutin.  One-sided returns ``(0, chi2_p)``, e.g. ``7.81``
    for ``dof = 3``.
    """
    if two_sided:
        return float(chi2.ppf((1.0 - prob) / 2.0, dof)), float(chi2.ppf((1.0 + prob) / 2.0, dof))
    return 0.0, float(chi2.ppf(prob, dof))


def fraction_inside(values: Array, bounds: Tuple[float, float]) -> float:
    """Fraction of ``values`` inside ``[lo, hi]`` (consistency check for NIS/NEES sequences)."""
    v = np.asarray(values, dtype=float)
    lo, hi = bounds
    return float(np.mean((v >= lo) & (v <= hi)))


def average_nis_bounds(dof: int, n_runs: int, prob: float = 0.95) -> Tuple[float, float]:
    """Interval for the *time-averaged* NIS over ``n_runs`` independent samples."""
    lo, hi = chi2_bounds(dof * n_runs, prob)
    return lo / n_runs, hi / n_runs
