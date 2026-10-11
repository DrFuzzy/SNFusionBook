"""Kalman-type filters: linear KF, extended KF and unscented KF.

Implements the recursion of Chapter 3, Section "Bayesian Filtering":

* prediction, eqs. (kf_state_prediction)--(kf_cov_prediction);
* innovation, eq. (kf_innovation), and innovation covariance, eq.
  (kf_innovation_covariance);
* gain, eq. (kf_gain), and posterior update, eqs. (kf_state_update)--(kf_cov_update)
  (the covariance update is evaluated in Joseph form by default, which is the
  form used by the scripts of Example ex:ekf-radar-cutin);
* the normalised innovation squared (NIS), eq. (nis), returned with every update.

The floating-point operation order of :meth:`ExtendedKalmanFilter.update` and
:meth:`UnscentedKalmanFilter.update` is *exactly* the one of the authors' example
scripts, so that the companion examples reproduce the CSV files shipped with the
book bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray
StateFn = Callable[..., Array]


def wrap_angle(a):
    """Wrap an angle (or array of angles) to the interval [-pi, pi).

    Used for the azimuth component of the innovation in Example
    ex:ekf-radar-cutin: "the difference of two angles is not the angle of the
    difference".
    """
    return (a + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class KalmanUpdate:
    """Result of one measurement update."""

    x: Array          #: posterior mean  x_{k|k}
    P: Array          #: posterior covariance P_{k|k}
    innovation: Array  #: nu_k = y_k - g(x_{k|k-1}),  eq. (kf_innovation)
    S: Array          #: innovation covariance,         eq. (kf_innovation_covariance)
    K: Array          #: Kalman gain,                   eq. (kf_gain)
    nis: float        #: normalised innovation squared, eq. (nis)


def _linear_update(x: Array, P: Array, z: Array, H: Array, R: Array,
                   z_pred: Array, angle_idx: Sequence[int] = (),
                   joseph: bool = True) -> KalmanUpdate:
    """Shared update kernel (same operation order as the authors' scripts)."""
    nu = z - z_pred
    for j in angle_idx:
        nu[j] = wrap_angle(nu[j])
    S = H @ P @ H.T + R
    S_inv = np.linalg.inv(S)
    K = P @ H.T @ S_inv
    x_new = x + K @ nu
    I = np.eye(x.size)
    if joseph:
        P_new = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
    else:
        P_new = (I - K @ H) @ P
    nis = float(nu @ S_inv @ nu)
    return KalmanUpdate(x_new, P_new, nu, S, K, nis)


class KalmanFilter:
    """Linear Kalman filter for the model of eqs. (kf_linear_process)--(kf_linear_measurement).

    Parameters
    ----------
    F, H : state-transition and measurement matrices.
    Q, R : process- and measurement-noise covariances.
    B    : optional input matrix (``u`` is then passed to :meth:`predict`).
    joseph : evaluate the covariance update in Joseph form (numerically safer
             than eq. (kf_cov_update); identical in exact arithmetic).

    Examples
    --------
    >>> kf = KalmanFilter(F=np.eye(1), H=np.eye(1), Q=np.zeros((1, 1)), R=np.eye(1))
    >>> x, P = kf.predict(np.zeros(1), np.eye(1))
    >>> upd = kf.update(x, P, np.array([2.0]))
    >>> float(upd.x[0]), float(upd.P[0, 0]), upd.nis
    (1.0, 0.5, 2.0)
    """

    def __init__(self, F: Array, H: Array, Q: Array, R: Array,
                 B: Optional[Array] = None, joseph: bool = True) -> None:
        self.F = np.asarray(F, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.B = None if B is None else np.asarray(B, dtype=float)
        self.joseph = joseph

    def predict(self, x: Array, P: Array, u: Optional[Array] = None) -> Tuple[Array, Array]:
        """Time update, eqs. (kf_state_prediction)--(kf_cov_prediction)."""
        x_pred = self.F @ x
        if self.B is not None and u is not None:
            x_pred = x_pred + self.B @ u
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(self, x: Array, P: Array, z: Array) -> KalmanUpdate:
        """Measurement update, eqs. (kf_innovation)--(kf_cov_update), with the NIS."""
        z = np.asarray(z, dtype=float)
        return _linear_update(x, P, z, self.H, self.R, self.H @ x, joseph=self.joseph)

    def step(self, x: Array, P: Array, z: Array, u: Optional[Array] = None) -> KalmanUpdate:
        """One predict + update cycle."""
        x, P = self.predict(x, P, u)
        return self.update(x, P, z)


class ExtendedKalmanFilter:
    """Extended Kalman filter with user-supplied models and Jacobians.

    Parameters
    ----------
    f : process model ``f(x, u) -> x`` (``u`` may be ``None``).
    F_jac : Jacobian of ``f`` w.r.t. ``x``, ``F_jac(x, u) -> (n, n)``.
    h : measurement model ``h(x) -> z``.
    H_jac : Jacobian of ``h``, ``H_jac(x) -> (m, n)``, evaluated at the predicted
            state (Example ex:ekf-radar-cutin, eq. (ekf-radar-jacobian)).
    Q, R : noise covariances.
    angle_idx : indices of measurement components that are angles; the
                corresponding innovation entries are wrapped with :func:`wrap_angle`.
    """

    def __init__(self, f: StateFn, F_jac: StateFn, h: Callable[[Array], Array],
                 H_jac: Callable[[Array], Array], Q: Array, R: Array,
                 angle_idx: Sequence[int] = (), joseph: bool = True) -> None:
        self.f, self.F_jac, self.h, self.H_jac = f, F_jac, h, H_jac
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.angle_idx = tuple(angle_idx)
        self.joseph = joseph

    def predict(self, x: Array, P: Array, u: Optional[Array] = None) -> Tuple[Array, Array]:
        """EKF prediction: nonlinear mean propagation, linearised covariance."""
        F = self.F_jac(x, u)
        x_pred = self.f(x, u)
        P_pred = F @ P @ F.T + self.Q
        return x_pred, P_pred

    def update(self, x: Array, P: Array, z: Array) -> KalmanUpdate:
        """EKF update with the Jacobian evaluated at the predicted state."""
        z = np.asarray(z, dtype=float)
        H = self.H_jac(x)
        return _linear_update(x, P, z, H, self.R, self.h(x), self.angle_idx, self.joseph)

    def step(self, x: Array, P: Array, z: Array, u: Optional[Array] = None) -> KalmanUpdate:
        x, P = self.predict(x, P, u)
        return self.update(x, P, z)


class UnscentedKalmanFilter:
    """Unscented Kalman filter (Julier & Uhlmann) with scaled sigma points.

    The default sigma-point parameters are the ones used in Example
    ex:ukf-radar-cutin: ``alpha = 1, beta = 2, kappa = 1``, hence
    ``lambda = alpha^2 (n + kappa) - n = 1`` for ``n = 4`` and all weights are
    non-negative.  (With the frequently quoted ``alpha = 1e-3`` the sigma points
    collapse onto the mean and the centre weight becomes hugely negative.)

    Parameters
    ----------
    f, h : process and measurement models (``f(x, u)``, ``h(x)``).
    Q, R : noise covariances.
    alpha, beta, kappa : sigma-point parameters.
    angle_idx : measurement components that are angles.  They are averaged on
                the circle (``atan2`` of weighted sines/cosines) and their
                residuals are wrapped, as in the authors' UKF script.
    linear_predict : optional matrix ``F``; when given, the prediction is the
                exact linear Kalman prediction (the engineering choice of
                Example ex:ukf-radar-cutin: the unscented transform is applied
                only to the nonlinear measurement update).
    """

    def __init__(self, f: Optional[StateFn], h: Callable[[Array], Array], Q: Array, R: Array,
                 alpha: float = 1.0, beta: float = 2.0, kappa: float = 1.0,
                 angle_idx: Sequence[int] = (), linear_predict: Optional[Array] = None) -> None:
        self.f, self.h = f, h
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.alpha, self.beta, self.kappa = float(alpha), float(beta), float(kappa)
        self.angle_idx = tuple(angle_idx)
        self.F_lin = None if linear_predict is None else np.asarray(linear_predict, dtype=float)

    # ---- sigma points --------------------------------------------------------
    def weights(self, n: int) -> Tuple[float, Array, Array]:
        """Return ``(lambda, w_m, w_c)`` for an ``n``-dimensional state."""
        lam = self.alpha ** 2 * (n + self.kappa) - n
        wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = lam / (n + lam) + (1.0 - self.alpha ** 2 + self.beta)
        return lam, wm, wc

    def sigma_points(self, x: Array, P: Array) -> Array:
        """``(2n+1, n)`` array of scaled sigma points (Cholesky square root)."""
        n = x.size
        lam, _, _ = self.weights(n)
        S = np.linalg.cholesky((n + lam) * P)
        X = np.zeros((2 * n + 1, n))
        X[0] = x
        for i in range(n):
            X[1 + i] = x + S[:, i]
            X[1 + n + i] = x - S[:, i]
        return X

    # ---- recursion -----------------------------------------------------------
    def predict(self, x: Array, P: Array, u: Optional[Array] = None) -> Tuple[Array, Array]:
        """Unscented (or exact linear, if ``linear_predict`` was given) prediction."""
        if self.F_lin is not None:
            return self.F_lin @ x, self.F_lin @ P @ self.F_lin.T + self.Q
        if self.f is None:
            raise ValueError("either f or linear_predict must be provided")
        n = x.size
        _, wm, wc = self.weights(n)
        X = self.sigma_points(x, P)
        Xp = np.array([self.f(xi, u) for xi in X])
        x_pred = wm @ Xp
        dX = Xp - x_pred
        P_pred = (wc[:, None, None] * dX[:, :, None] * dX[:, None, :]).sum(0) + self.Q
        return x_pred, P_pred

    def update(self, x: Array, P: Array, z: Array) -> KalmanUpdate:
        """Unscented measurement update; the covariance update is ``P - K Pzz K^T``."""
        z = np.asarray(z, dtype=float)
        n = x.size
        _, wm, wc = self.weights(n)
        X = self.sigma_points(x, P)
        Z = np.array([self.h(xi) for xi in X])
        z_hat = wm @ Z
        for j in self.angle_idx:  # angles are averaged on the circle, not on the line
            z_hat[j] = np.arctan2(wm @ np.sin(Z[:, j]), wm @ np.cos(Z[:, j]))
        dZ = Z - z_hat
        for j in self.angle_idx:
            dZ[:, j] = wrap_angle(dZ[:, j])
        dX = X - x
        Pzz = (wc[:, None, None] * dZ[:, :, None] * dZ[:, None, :]).sum(0) + self.R
        Pxz = (wc[:, None, None] * dX[:, :, None] * dZ[:, None, :]).sum(0)
        K = Pxz @ np.linalg.inv(Pzz)
        nu = z - z_hat
        for j in self.angle_idx:
            nu[j] = wrap_angle(nu[j])
        x_new = x + K @ nu
        P_new = P - K @ Pzz @ K.T
        nis = float(nu @ np.linalg.inv(Pzz) @ nu)
        return KalmanUpdate(x_new, P_new, nu, Pzz, K, nis)

    def step(self, x: Array, P: Array, z: Array, u: Optional[Array] = None) -> KalmanUpdate:
        x, P = self.predict(x, P, u)
        return self.update(x, P, z)
