#!/usr/bin/env python3
"""
UKF companion to ekf_radar_cutin.py -- same scenario, same data, same tuning.
=============================================================================

Runs an unscented Kalman filter on exactly the measurement sequence the EKF
sees, so the two are directly comparable, and then repeats the comparison on a
stressed near-field variant in which the azimuth nonlinearity is no longer mild.

The process model is linear (constant velocity), so the prediction step is the
exact Kalman prediction; the unscented transform is applied only to the
nonlinear measurement update. This is the standard engineering choice and is
what the example states.

Sigma-point parameters: alpha = 1, beta = 2, kappa = 1, hence lambda = 1 and all
weights are non-negative (with the frequently quoted alpha = 1e-3 the sigma
points collapse onto the mean and the UKF degenerates numerically into a
central-difference version of the EKF, which is worth knowing but useless as a
demonstration).

Writes ukf_radar_cutin.csv and prints the numbers quoted in Example 2.
Run:  python3 ukf_radar_cutin.py       Requires only numpy.
"""

import numpy as np
from ekf_radar_cutin import (F_matrix, Q_matrix, h, H_jacobian, wrap,
                             truth, T, N, SIG_R, SIG_TH, SIG_RD, SIG_A,
                             CUT_T0, CUT_T1, RNG_SEED)

ALPHA, BETA, KAPPA = 1.0, 2.0, 1.0


def sigma_weights(n):
    lam = ALPHA**2 * (n + KAPPA) - n
    wm = np.full(2*n + 1, 1.0 / (2.0 * (n + lam)))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) + (1.0 - ALPHA**2 + BETA)
    return lam, wm, wc


def ukf_update(x, P, z, R):
    """Unscented measurement update; returns x+, P+, NIS."""
    n = x.size
    lam, wm, wc = sigma_weights(n)
    S = np.linalg.cholesky((n + lam) * P)
    X = np.zeros((2*n + 1, n))
    X[0] = x
    for i in range(n):
        X[1 + i]     = x + S[:, i]
        X[1 + n + i] = x - S[:, i]

    Z = np.array([h(xi) for xi in X])
    # the azimuth channel must be averaged on the circle, not on the line
    z_hat = wm @ Z
    z_hat[1] = np.arctan2(wm @ np.sin(Z[:, 1]), wm @ np.cos(Z[:, 1]))

    dZ = Z - z_hat
    dZ[:, 1] = wrap(dZ[:, 1])
    dX = X - x

    Pzz = (wc[:, None, None] * dZ[:, :, None] * dZ[:, None, :]).sum(0) + R
    Pxz = (wc[:, None, None] * dX[:, :, None] * dZ[:, None, :]).sum(0)

    K = Pxz @ np.linalg.inv(Pzz)
    nu = z - z_hat
    nu[1] = wrap(nu[1])
    return x + K @ nu, P - K @ Pzz @ K.T, float(nu @ np.linalg.inv(Pzz) @ nu)


def ekf_update(x, P, z, R):
    """First-order update, identical to the one in Example 1."""
    H = H_jacobian(x)
    nu = z - h(x)
    nu[1] = wrap(nu[1])
    S = H @ P @ H.T + R
    Sinv = np.linalg.inv(S)
    K = P @ H.T @ Sinv
    I = np.eye(x.size)
    P2 = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
    return x + K @ nu, P2, float(nu @ Sinv @ nu)


def run(px0, py0, vx0, sig_th, p0_vel, seed, n_steps, cut=True):
    """Simulate one scenario and filter it with both an EKF and a UKF."""
    rng = np.random.default_rng(seed)
    F, Q = F_matrix(T), Q_matrix(T, SIG_A)
    R = np.diag([SIG_R**2, sig_th**2, SIG_RD**2])
    ts = np.arange(n_steps + 1) * T

    def local_truth(t):
        if cut:
            x = truth(t)
            return np.array([px0 + vx0*t, x[1], vx0, x[3]])
        return np.array([px0 + vx0*t, py0, vx0, 0.0])

    Xt = np.array([local_truth(t) for t in ts])
    Z = np.array([h(x) for x in Xt])
    Z = Z + rng.normal(0.0, [SIG_R, sig_th, SIG_RD], Z.shape)
    Z[:, 1] = wrap(Z[:, 1])

    out = {}
    for name, upd in (("EKF", ekf_update), ("UKF", ukf_update)):
        r0, th0, _ = Z[0]
        x = np.array([r0*np.cos(th0), r0*np.sin(th0), 0.0, 0.0])
        P = np.diag([SIG_R**2, SIG_R**2, p0_vel**2, p0_vel**2])
        est, nis = [], []
        for k in range(len(ts)):
            if k > 0:
                x = F @ x
                P = F @ P @ F.T + Q
            x, P, d2 = upd(x, P, Z[k], R)
            P = 0.5 * (P + P.T)
            est.append(x.copy()); nis.append(d2)
        out[name] = (np.array(est), np.array(nis))
    return ts, Xt, Z, out


def rms(v):
    return float(np.sqrt(np.mean(np.asarray(v)**2)))


def report(tag, ts, Xt, out, t_from=1.0):
    m = ts >= t_from
    line = f"{tag:<34s}"
    for name in ("EKF", "UKF"):
        E, nis = out[name]
        pos = np.hypot(E[:, 0]-Xt[:, 0], E[:, 1]-Xt[:, 1])
        line += f"  {name} pos-RMSE {rms(pos[m]):6.3f} m  meanNIS {np.mean(nis[m]):5.2f}"
    print(line)
    return out


if __name__ == "__main__":
    print("=" * 108)
    print("A. NOMINAL SCENARIO  (identical to Example 1: 60 m ahead, sigma_theta = 0.5 deg)")
    ts, Xt, Z, outA = run(60.0, 3.5, -5.0, SIG_TH, 10.0, RNG_SEED, N)
    report("   nominal", ts, Xt, outA)

    print()
    print("B. STRESSED NEAR-FIELD VARIANT (12 m ahead, sigma_theta = 3 deg, poor initialisation)")
    # target starts 12 m ahead in the adjacent lane: azimuth ~16 deg and growing
    ts2, Xt2, Z2, outB = run(12.0, 3.5, -0.9, np.deg2rad(3.0), 25.0, RNG_SEED + 1, N)
    report("   stressed", ts2, Xt2, outB)

    print()
    print("Sigma-point weights (n = 4, alpha = %.1f, beta = %.1f, kappa = %.1f):" % (ALPHA, BETA, KAPPA))
    lam, wm, wc = sigma_weights(4)
    print(f"   lambda = {lam:.3f}   w_m0 = {wm[0]:.4f}   w_i = {wm[1]:.4f}   "
          f"spread = {np.sqrt(4+lam):.3f} sigma")

    # ---- CSV for the figure: nominal EKF vs UKF lateral error, plus stressed case
    EA, nisA = outA["EKF"]; UA, nisUA = outA["UKF"]
    EB, nisB = outB["EKF"]; UB, nisUB = outB["UKF"]
    cols = ["t", "ekf_ey_nom", "ukf_ey_nom", "ekf_ey_str", "ukf_ey_str",
            "ekf_ep_str", "ukf_ep_str"]
    with open("ukf_radar_cutin.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for k, t in enumerate(ts):
            row = [t,
                   EA[k, 1]-Xt[k, 1],   UA[k, 1]-Xt[k, 1],
                   EB[k, 1]-Xt2[k, 1],  UB[k, 1]-Xt2[k, 1],
                   np.hypot(EB[k, 0]-Xt2[k, 0], EB[k, 1]-Xt2[k, 1]),
                   np.hypot(UB[k, 0]-Xt2[k, 0], UB[k, 1]-Xt2[k, 1])]
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")
    print("\nwrote ukf_radar_cutin.csv")
