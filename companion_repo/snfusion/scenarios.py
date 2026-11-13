"""The automotive-radar cut-in scenario shared by three Chapter 3 examples.

Examples ex:ekf-radar-cutin (EKF), ex:ukf-radar-cutin (UKF comparison) and
ex:zono-radar-cutin (zonotopic enclosure) all use the same target, the same
constant-velocity model and the same radar.  This module holds the scenario
definition verbatim from the authors' ``ekf_radar_cutin.py`` so that the three
companion scripts share one source of truth.

Scenario (ego-vehicle frame, ISO 8855: x forward, y left, origin at the front
bumper radar):

* ego vehicle at 27 m/s in the right-hand lane;
* target at 22 m/s in the adjacent left-hand lane, initially 60 m ahead,
  so the relative longitudinal velocity is -5 m/s;
* between t = 5 s and t = 8 s the target changes lane into the ego lane
  (lateral offset 3.5 m -> 0 m), a manoeuvre the constant-velocity model does
  not contain.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .kalman import wrap_angle
from .models import constant_velocity, white_noise_acceleration_Q

Array = np.ndarray

# ----------------------------------------------------------------------------
# Scenario and filter parameters (the numbers quoted in Example ex:ekf-radar-cutin)
# ----------------------------------------------------------------------------
RNG_SEED = 20260727          #: fixed seed: the book figures are reproducible bit for bit

T = 0.05                     #: radar update period [s] (20 Hz)
T_END = 10.0                 #: simulation horizon [s]
N = int(round(T_END / T))    #: number of update intervals (N + 1 samples)

PX0, PY0 = 60.0, 3.5         #: initial relative position [m]
VX0, VY0 = -5.0, 0.0         #: initial relative velocity [m/s]

CUT_T0, CUT_T1 = 5.0, 8.0    #: lane-change window [s]
CUT_DY = -3.5                #: lateral displacement during the manoeuvre [m]

SIG_R = 0.30                 #: radar range noise std [m]
SIG_TH = np.deg2rad(0.5)     #: radar azimuth noise std [rad]
SIG_RD = 0.10                #: radar range-rate noise std [m/s]

SIG_A = 0.80                 #: process-noise acceleration std, both axes [m/s^2]


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def F_matrix(dt: float) -> Array:
    """Constant-velocity state transition (eq. (kf_linear_process) of the example)."""
    return constant_velocity(dt, 2)


def Q_matrix(dt: float, sigma_a: float) -> Array:
    """Discrete, independent piecewise-constant acceleration model (sigma_a in m/s^2)."""
    return white_noise_acceleration_Q(dt, sigma_a, 2)


def R_matrix(sig_r: float = SIG_R, sig_th: float = SIG_TH, sig_rd: float = SIG_RD) -> Array:
    """Radar measurement covariance ``diag(sigma_r^2, sigma_theta^2, sigma_rdot^2)``."""
    return np.diag([sig_r ** 2, sig_th ** 2, sig_rd ** 2])


def h(x: Array) -> Array:
    """Nonlinear radar measurement g(x): range, azimuth, range rate (eq. (ekf-radar-meas))."""
    px, py, vx, vy = x
    d2 = px * px + py * py
    d = np.sqrt(d2)
    return np.array([d,
                     np.arctan2(py, px),
                     (px * vx + py * vy) / d])


def H_jacobian(x: Array) -> Array:
    """Jacobian of :func:`h` evaluated at ``x`` (eq. (ekf-radar-jacobian)); singular at d = 0."""
    px, py, vx, vy = x
    d2 = px * px + py * py
    d = np.sqrt(d2)
    d3 = d2 * d
    return np.array([
        [px / d, py / d, 0.0, 0.0],
        [-py / d2, px / d2, 0.0, 0.0],
        [py * (vx * py - vy * px) / d3, px * (vy * px - vx * py) / d3, px / d, py / d],
    ])


wrap = wrap_angle  #: alias kept for readability of the ported scripts


# ----------------------------------------------------------------------------
# Ground truth: constant velocity plus a smooth lane change
# ----------------------------------------------------------------------------
def truth(t: float) -> Array:
    """True relative state ``[px, py, vx, vy]`` at time ``t`` (smoothstep lane change)."""
    px = PX0 + VX0 * t
    if t <= CUT_T0:
        s, sdot = 0.0, 0.0
    elif t >= CUT_T1:
        s, sdot = 1.0, 0.0
    else:
        tau = (t - CUT_T0) / (CUT_T1 - CUT_T0)
        # smoothstep: C^1, zero lateral velocity at both ends of the manoeuvre
        s = 3 * tau ** 2 - 2 * tau ** 3
        sdot = (6 * tau - 6 * tau ** 2) / (CUT_T1 - CUT_T0)
    py = PY0 + CUT_DY * s
    vy = CUT_DY * sdot
    return np.array([px, py, VX0, vy])


def simulate_measurements(Xt: Array, rng: np.random.Generator,
                          sig_r: float = SIG_R, sig_th: float = SIG_TH,
                          sig_rd: float = SIG_RD) -> Tuple[Array, Array]:
    """Noisy radar detections of the true trajectory ``Xt`` (``(N+1, 4)``).

    Returns ``(Z, err)``: the wrapped measurements and the realised noise, drawn
    in exactly the order of the authors' scripts (one ``rng.normal`` call with
    per-channel standard deviations).
    """
    Z = np.array([h(x) for x in Xt])
    err = rng.normal(0.0, [sig_r, sig_th, sig_rd], Z.shape)
    Z = Z + err
    Z[:, 1] = wrap(Z[:, 1])
    return Z, err


def nominal_run(seed: int = RNG_SEED) -> Tuple[Array, Array, Array, Array]:
    """Time grid, true states, measurements and noise of the nominal scenario."""
    rng = np.random.default_rng(seed)
    ts = np.arange(N + 1) * T
    Xt = np.array([truth(t) for t in ts])
    Z, err = simulate_measurements(Xt, rng)
    return ts, Xt, Z, err


def initial_estimate(z0: Array, p0_vel: float = 10.0,
                     sig_r: float = SIG_R, sig_th: float = SIG_TH) -> Tuple[Array, Array]:
    """Track initiation from the first detection (velocity essentially unknown)."""
    r0, th0, _ = z0
    x0 = np.array([r0 * np.cos(th0), r0 * np.sin(th0), 0.0, 0.0])
    J = np.array([[np.cos(th0), -r0 * np.sin(th0)],
                  [np.sin(th0), r0 * np.cos(th0)]])
    P0 = np.diag([0.0, 0.0, p0_vel ** 2, p0_vel ** 2])
    P0[:2, :2] = J @ np.diag([sig_r ** 2, sig_th ** 2]) @ J.T
    return x0, P0
