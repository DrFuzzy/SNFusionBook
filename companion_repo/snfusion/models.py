"""Discrete-time motion models used throughout the book.

State ordering is ``[positions..., velocities...]`` (and ``accelerations`` for
the constant-acceleration model), e.g. ``[px, py, vx, vy]`` in two dimensions,
which is the ordering of Example ex:ekf-radar-cutin.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def constant_velocity(dt: float, dim: int = 2) -> Array:
    """Constant-velocity transition matrix ``F`` (Example ex:ekf-radar-cutin).

    ``F = [[I, dt I], [0, I]]`` with ``dim``-dimensional position and velocity.
    """
    I = np.eye(dim)
    Z = np.zeros((dim, dim))
    return np.block([[I, dt * I], [Z, I]])


def white_noise_acceleration_Q(dt: float, sigma_a: float, dim: int = 2) -> Array:
    """Discretised white-noise-acceleration process covariance of the CV model.

    ``Q = sigma_a^2 [[T^4/4 I, T^3/2 I], [T^3/2 I, T^2 I]]`` -- the matrix quoted
    in Example ex:ekf-radar-cutin, where ``sigma_a`` "expresses how much
    unmodelled acceleration the target is permitted between updates".
    """
    q = sigma_a ** 2
    t2, t3, t4 = dt ** 2, dt ** 3, dt ** 4
    I = np.eye(dim)
    return q * np.block([[t4 / 4 * I, t3 / 2 * I], [t3 / 2 * I, t2 * I]])


def constant_acceleration(dt: float, dim: int = 2) -> Array:
    """Constant-acceleration (nearly-constant-acceleration) transition matrix.

    State ``[p, v, a]``: ``p+ = p + dt v + dt^2/2 a``, ``v+ = v + dt a``, ``a+ = a``.
    """
    I = np.eye(dim)
    Z = np.zeros((dim, dim))
    return np.block([[I, dt * I, 0.5 * dt ** 2 * I],
                     [Z, I, dt * I],
                     [Z, Z, I]])


def white_noise_jerk_Q(dt: float, sigma_j: float, dim: int = 2) -> Array:
    """Process covariance of the CA model driven by white-noise jerk ``sigma_j``."""
    q = sigma_j ** 2
    I = np.eye(dim)
    t = dt
    return q * np.block([[t ** 5 / 20 * I, t ** 4 / 8 * I, t ** 3 / 6 * I],
                         [t ** 4 / 8 * I, t ** 3 / 3 * I, t ** 2 / 2 * I],
                         [t ** 3 / 6 * I, t ** 2 / 2 * I, t * I]])


def coordinated_turn(dt: float, omega: float) -> Array:
    """Planar coordinated-turn transition for state ``[px, py, vx, vy]`` at known turn rate.

    For ``omega -> 0`` the matrix reduces to :func:`constant_velocity`.
    """
    if abs(omega) < 1e-12:
        return constant_velocity(dt, 2)
    s, c = np.sin(omega * dt), np.cos(omega * dt)
    return np.array([[1.0, 0.0, s / omega, -(1.0 - c) / omega],
                     [0.0, 1.0, (1.0 - c) / omega, s / omega],
                     [0.0, 0.0, c, -s],
                     [0.0, 0.0, s, c]])


def coordinated_turn_augmented(dt: float, x: Array) -> Array:
    """Nonlinear CT model with the turn rate in the state ``[px, py, vx, vy, omega]``.

    Returns the propagated state (for use as the ``f`` of an EKF/UKF).
    """
    px, py, vx, vy, w = x
    F = coordinated_turn(dt, w)
    p = F @ np.array([px, py, vx, vy])
    return np.array([p[0], p[1], p[2], p[3], w])


def coordinated_turn_Q(dt: float, sigma_a: float, sigma_w: float) -> Array:
    """Process covariance for the augmented CT model (WNA on both axes + turn-rate noise)."""
    Q = np.zeros((5, 5))
    Q[:4, :4] = white_noise_acceleration_Q(dt, sigma_a, 2)
    Q[4, 4] = (sigma_w * dt) ** 2
    return Q


def position_measurement(dim: int = 2, state_dim: int = 4) -> Array:
    """``H`` selecting the position block of the state (linear Cartesian sensor)."""
    H = np.zeros((dim, state_dim))
    H[:dim, :dim] = np.eye(dim)
    return H
