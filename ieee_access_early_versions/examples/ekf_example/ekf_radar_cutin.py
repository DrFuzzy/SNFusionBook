#!/usr/bin/env python3
"""
EKF tracking of a cut-in vehicle from automotive-radar detections.
=================================================================

Autonomous-driving counterpart of the classical "tracking an aircraft from radar
measurements" Kalman-filter case study: the constant-velocity target model is kept,
but the radar is modelled with its native polar + Doppler output (range, azimuth,
range rate) instead of Cartesian positions.  The measurement model is therefore
nonlinear and an EKF is required.

Scenario (ego-vehicle frame, ISO 8855: x forward, y left, origin at the front
bumper radar):

    - ego vehicle travels at 27 m/s in the right-hand lane;
    - target vehicle travels at 22 m/s in the adjacent left-hand lane,
      initially 60 m ahead, so the relative longitudinal velocity is -5 m/s;
    - between t = 5 s and t = 8 s the target changes lane into the ego lane
      (lateral offset 3.5 m -> 0 m), a manoeuvre the constant-velocity model
      does not contain.

Outputs `ekf_radar_cutin.csv`, consumed by the TikZ/pgfplots figure in
`ekf_radar_cutin_example.tex`, and prints the summary numbers quoted in the text.

Run:  python3 ekf_radar_cutin.py
Requires only numpy.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Scenario and filter parameters (these are the numbers quoted in the example)
# ----------------------------------------------------------------------------
RNG_SEED = 20260727

T        = 0.05          # radar update period [s]  (20 Hz)
T_END    = 10.0          # simulation horizon [s]
N        = int(round(T_END / T))

PX0, PY0 = 60.0, 3.5     # initial relative position [m]
VX0, VY0 = -5.0, 0.0     # initial relative velocity [m/s]

CUT_T0, CUT_T1 = 5.0, 8.0    # lane-change window [s]
CUT_DY         = -3.5        # lateral displacement during the manoeuvre [m]

SIG_R    = 0.30          # radar range noise std       [m]
SIG_TH   = np.deg2rad(0.5)   # radar azimuth noise std [rad]
SIG_RD   = 0.10          # radar range-rate noise std  [m/s]

SIG_A    = 0.80          # process-noise accel. std, both axes [m/s^2]

# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def F_matrix(dt):
    """Constant-velocity state transition (same structure as the aircraft example)."""
    return np.array([[1.0, 0.0,  dt, 0.0],
                     [0.0, 1.0, 0.0,  dt],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def Q_matrix(dt, sigma_a):
    """Discretised white-noise-acceleration process noise."""
    q = sigma_a ** 2
    t2, t3, t4 = dt**2, dt**3, dt**4
    return q * np.array([[t4/4, 0.0,  t3/2, 0.0 ],
                         [0.0,  t4/4, 0.0,  t3/2],
                         [t3/2, 0.0,  t2,   0.0 ],
                         [0.0,  t3/2, 0.0,  t2  ]])


def h(x):
    """Nonlinear radar measurement: range, azimuth, range rate."""
    px, py, vx, vy = x
    d2 = px*px + py*py
    d  = np.sqrt(d2)
    return np.array([d,
                     np.arctan2(py, px),
                     (px*vx + py*vy) / d])


def H_jacobian(x):
    """Jacobian of h evaluated at x (the EKF linearisation)."""
    px, py, vx, vy = x
    d2 = px*px + py*py
    d  = np.sqrt(d2)
    d3 = d2 * d
    return np.array([
        [px/d,                 py/d,                 0.0,  0.0 ],
        [-py/d2,               px/d2,                0.0,  0.0 ],
        [py*(vx*py - vy*px)/d3, px*(vy*px - vx*py)/d3, px/d, py/d],
    ])


def wrap(a):
    """Wrap an angle to (-pi, pi]; essential for the azimuth innovation."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ----------------------------------------------------------------------------
# Ground truth: constant velocity plus a smooth lane change
# ----------------------------------------------------------------------------
def truth(t):
    """Return the true relative state at time t."""
    px = PX0 + VX0 * t
    if t <= CUT_T0:
        s, sdot = 0.0, 0.0
    elif t >= CUT_T1:
        s, sdot = 1.0, 0.0
    else:
        tau  = (t - CUT_T0) / (CUT_T1 - CUT_T0)
        # smoothstep: C^1, zero lateral velocity at both ends of the manoeuvre
        s    = 3*tau**2 - 2*tau**3
        sdot = (6*tau - 6*tau**2) / (CUT_T1 - CUT_T0)
    py = PY0 + CUT_DY * s
    vy = CUT_DY * sdot
    return np.array([px, py, VX0, vy])


# ----------------------------------------------------------------------------
# Simulation + EKF
# ----------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(RNG_SEED)

    F = F_matrix(T)
    Q = Q_matrix(T, SIG_A)
    R = np.diag([SIG_R**2, SIG_TH**2, SIG_RD**2])

    ts   = np.arange(N + 1) * T
    Xt   = np.array([truth(t) for t in ts])                 # true states
    Z    = np.array([h(x) for x in Xt])                     # noise-free measurements
    Z    = Z + rng.normal(0.0, [SIG_R, SIG_TH, SIG_RD], Z.shape)
    Z[:, 1] = wrap(Z[:, 1])

    # --- initialisation from the first detection (standard track-initiation) ---
    r0, th0, rd0 = Z[0]
    x_hat = np.array([r0*np.cos(th0), r0*np.sin(th0), 0.0, 0.0])
    P = np.diag([SIG_R**2, SIG_R**2, 10.0**2, 10.0**2])     # velocity essentially unknown

    rows = []
    for k, t in enumerate(ts):
        if k > 0:
            # ---- prediction (linear here: the process model is CV) ----
            x_hat = F @ x_hat
            P     = F @ P @ F.T + Q

        # ---- update ----
        Hk  = H_jacobian(x_hat)
        nu  = Z[k] - h(x_hat)
        nu[1] = wrap(nu[1])                                  # azimuth innovation wrapping
        S   = Hk @ P @ Hk.T + R
        Sinv = np.linalg.inv(S)
        K   = P @ Hk.T @ Sinv
        x_hat = x_hat + K @ nu
        I = np.eye(4)
        P = (I - K @ Hk) @ P @ (I - K @ Hk).T + K @ R @ K.T  # Joseph form

        nis = float(nu @ Sinv @ nu)

        # raw detection expressed in Cartesian coordinates, for comparison
        zx, zy = Z[k, 0]*np.cos(Z[k, 1]), Z[k, 0]*np.sin(Z[k, 1])

        rows.append(dict(
            t=t,
            px_true=Xt[k, 0], py_true=Xt[k, 1],
            vx_true=Xt[k, 2], vy_true=Xt[k, 3],
            px_meas=zx,       py_meas=zy,
            px_est=x_hat[0],  py_est=x_hat[1],
            vx_est=x_hat[2],  vy_est=x_hat[3],
            sig_px=np.sqrt(P[0, 0]), sig_py=np.sqrt(P[1, 1]),
            nis=nis,
        ))

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    cols = list(rows[0].keys())
    with open("ekf_radar_cutin.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.6f}" for c in cols) + "\n")

    # ------------------------------------------------------------------
    # Summary numbers quoted in the example text
    # ------------------------------------------------------------------
    A = {c: np.array([r[c] for r in rows]) for c in cols}
    pos_err_meas = np.hypot(A["px_meas"] - A["px_true"], A["py_meas"] - A["py_true"])
    pos_err_est  = np.hypot(A["px_est"]  - A["px_true"], A["py_est"]  - A["py_true"])
    settle = A["t"] >= 1.0          # ignore the initial transient

    def rms(v):
        return float(np.sqrt(np.mean(v**2)))

    steady   = (A["t"] >= 1.0) & (A["t"] < CUT_T0)
    manoeuvr = (A["t"] >= CUT_T0) & (A["t"] <= CUT_T1)

    print(f"steps                         : {N+1}")
    print(f"position RMSE, raw detections : {rms(pos_err_meas[settle]):.3f} m")
    print(f"position RMSE, EKF            : {rms(pos_err_est[settle]):.3f} m")
    print(f"  -> reduction                : "
          f"{100*(1-rms(pos_err_est[settle])/rms(pos_err_meas[settle])):.0f} %")
    print(f"lateral RMSE, raw detections  : "
          f"{rms((A['py_meas']-A['py_true'])[settle]):.3f} m")
    print(f"lateral RMSE, EKF             : "
          f"{rms((A['py_est'] -A['py_true'])[settle]):.3f} m")
    print(f"vy RMSE, EKF (after 1 s)      : "
          f"{rms((A['vy_est'] -A['vy_true'])[settle]):.3f} m/s")
    print(f"mean NIS, constant-velocity   : {A['nis'][steady].mean():.2f}  (E=3)")
    print(f"mean NIS, during cut-in       : {A['nis'][manoeuvr].mean():.2f}  (E=3)")
    print(f"max  NIS, during cut-in       : {A['nis'][manoeuvr].max():.2f}")
    print(f"chi2_3 95% upper bound        : 7.81")
    frac = float(np.mean(A["nis"][steady] > 7.815))
    print(f"NIS above bound, CV segment   : {100*frac:.1f} %")
    frac_m = float(np.mean(A["nis"][manoeuvr] > 7.815))
    print(f"NIS above bound, cut-in       : {100*frac_m:.1f} %")
    print("\nwrote ekf_radar_cutin.csv")


if __name__ == "__main__":
    main()
