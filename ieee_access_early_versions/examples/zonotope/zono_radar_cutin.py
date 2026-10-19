#!/usr/bin/env python3
"""
Zonotopic (set-membership) counterpart of the EKF example.
==========================================================

Same target, same constant-velocity model F, same radar, same measurement
sequence as ekf_radar_cutin.py.  What changes is the description of the errors:
covariances are replaced by hard bounds, and the filter propagates a zonotope
X_k = p_k + H_k B_inf instead of a mean and a covariance.

Two constructions keep the example short and, more importantly, exact:

1. Bounded process noise.  A constant-velocity model with |a_x|,|a_y| <= a_bar
   contributes exactly two generators per step, a_bar*[T^2/2,0,T,0]^T and
   a_bar*[0,T^2/2,0,T]^T.  a_bar is chosen to cover the lane-change
   acceleration, which the script verifies.

2. Exact measurement strips.  Rather than linearising the polar measurement and
   bounding a Taylor remainder, the polar uncertainty is expressed exactly in
   the frame aligned with the measured bearing.  Writing u = (cos th, sin th)
   and v = (-sin th, cos th), the true position satisfies

       u.p  in  [(r-r_bar) cos th_bar,  r+r_bar]
       v.p  in  [-(r+r_bar) sin th_bar, (r+r_bar) sin th_bar]

   which are two exact strips: the annulus sector is contained in a rectangle in
   the rotated frame, with no linearisation error at all.  The range-rate
   channel gives a third strip with normal (0,0,u_x,u_y) and half-width
   rho_bar + th_bar * v_max, valid whenever the relative speed is bounded by
   v_max, since the bearing error rotates the projection direction by at most
   th_bar.

The strip gain minimising the Frobenius radius of the corrected generator matrix
has a closed form,  lambda* = H H^T c / (c^T H H^T c + phi_bar^2),  which is the
Kalman gain with H H^T in place of P.  That is the connection worth drawing.

Writes zono_radar_cutin.csv and prints the numbers quoted in Example 3.
Run:  python3 zono_radar_cutin.py       Requires only numpy.
"""

import numpy as np
from ekf_radar_cutin import (F_matrix, h, wrap, truth, T, N,
                             SIG_R, SIG_TH, SIG_RD, RNG_SEED,
                             CUT_T0, CUT_T1, CUT_DY)

# --- hard bounds, set at four standard deviations of the Example-1 noise ------
K_SIGMA = 4.0
R_BAR   = K_SIGMA * SIG_R          # 1.20 m
TH_BAR  = K_SIGMA * SIG_TH         # 2.0 deg
RD_BAR  = K_SIGMA * SIG_RD         # 0.40 m/s
V_MAX   = 15.0                     # bound on relative speed [m/s]
A_BAR   = 2.5                      # bound on relative acceleration [m/s^2]
M_MAX   = 20                       # generator budget (order reduction)


def strips(z):
    """Exact bounded-error strips from one polar detection z = (r, th, rdot)."""
    r, th, rdot = z
    u = np.array([np.cos(th), np.sin(th)])
    v = np.array([-np.sin(th), np.cos(th)])
    lo = (r - R_BAR) * np.cos(TH_BAR)
    hi = r + R_BAR
    out = [
        (np.array([u[0], u[1], 0.0, 0.0]), 0.5*(lo+hi),  0.5*(hi-lo)),
        (np.array([v[0], v[1], 0.0, 0.0]), 0.0,          (r + R_BAR)*np.sin(TH_BAR)),
        (np.array([0.0, 0.0, u[0], u[1]]), rdot,         RD_BAR + TH_BAR*V_MAX),
    ]
    return out


def correct(p, H, c, ybar, phibar):
    """Single-strip correction with the Frobenius-optimal gain."""
    HHt_c = H @ (H.T @ c)
    lam = HHt_c / (c @ HHt_c + phibar**2)
    p_new = p + lam * (ybar - c @ p)
    H_new = np.hstack([H - np.outer(lam, H.T @ c), (phibar*lam)[:, None]])
    return p_new, H_new


def reduce_order(H, m_max):
    """Interval-hull order reduction: keep the largest columns, box the rest."""
    n, m = H.shape
    if m <= m_max:
        return H
    order = np.argsort(-np.linalg.norm(H, axis=0))
    keep, red = H[:, order[:m_max - n]], H[:, order[m_max - n:]]
    return np.hstack([keep, np.diag(np.abs(red).sum(axis=1))])


def main():
    rng = np.random.default_rng(RNG_SEED)
    F = F_matrix(T)
    ts = np.arange(N + 1) * T
    Xt = np.array([truth(t) for t in ts])

    # identical measurement sequence to Example 1 (same seed, same draw order)
    Z = np.array([h(x) for x in Xt])
    err = rng.normal(0.0, [SIG_R, SIG_TH, SIG_RD], Z.shape)
    Z = Z + err
    Z[:, 1] = wrap(Z[:, 1])

    # --- sanity: are the assumed hard bounds actually valid on this run? ------
    viol = [(np.abs(err[:, 0]) > R_BAR).sum(),
            (np.abs(err[:, 1]) > TH_BAR).sum(),
            (np.abs(err[:, 2]) > RD_BAR).sum()]
    a_peak = A_BAR
    tau = np.linspace(0, 1, 2001)
    a_true = np.abs(CUT_DY * (6 - 12*tau) / (CUT_T1 - CUT_T0)**2).max()
    print(f"bound check: realised sensor errors exceeding the bound: "
          f"range {viol[0]}, azimuth {viol[1]}, range rate {viol[2]} (of {len(err)})")
    print(f"bound check: peak true lateral acceleration {a_true:.2f} m/s^2 "
          f"vs assumed bound {A_BAR:.2f} m/s^2 -> "
          f"{'VALID' if a_true <= A_BAR else 'VIOLATED'}")
    print(f"             relative speed max {np.abs(Xt[:, 2:]).max():.2f} m/s "
          f"vs assumed {V_MAX:.1f} m/s")

    # --- initialisation: the first detection's own strips define the box ------
    r0, th0, _ = Z[0]
    p = np.array([r0*np.cos(th0), r0*np.sin(th0), 0.0, 0.0])
    H = np.diag([R_BAR + r0*TH_BAR, R_BAR + r0*TH_BAR, V_MAX, V_MAX])

    Gq = A_BAR * np.array([[T**2/2, 0.0], [0.0, T**2/2], [T, 0.0], [0.0, T]])

    rows, contained, ngen = [], 0, []
    for k, t in enumerate(ts):
        if k > 0:
            p = F @ p
            H = np.hstack([F @ H, Gq])              # prediction: exact
        for c, ybar, phibar in strips(Z[k]):        # correction: three strips
            p, H = correct(p, H, c, ybar, phibar)
        H = reduce_order(H, M_MAX)
        ngen.append(H.shape[1])

        half = np.abs(H).sum(axis=1)                # interval hull half-widths
        inside = np.all(np.abs(Xt[k] - p) <= half + 1e-9)
        contained += int(inside)
        rows.append(dict(t=t,
                         py_true=Xt[k, 1],
                         zc=p[1], zlo=p[1]-half[1], zhi=p[1]+half[1],
                         vy_lo=p[3]-half[3], vy_hi=p[3]+half[3]))

    A = {c: np.array([r[c] for r in rows]) for c in rows[0]}
    m = A["t"] >= 1.0
    w_zono = (A["zhi"] - A["zlo"])

    # --- comparison against the EKF 3-sigma envelope from Example 1 -----------
    import csv
    E = list(csv.DictReader(open("ekf_radar_cutin.csv")))
    ek = {c: np.array([float(r[c]) for r in E]) for c in E[0]}
    ekf_lo = ek["py_est"] - 3*ek["sig_py"]
    ekf_hi = ek["py_est"] + 3*ek["sig_py"]
    ekf_in = (ek["py_true"] >= ekf_lo) & (ek["py_true"] <= ekf_hi)
    man = (A["t"] >= CUT_T0) & (A["t"] <= CUT_T1)

    print()
    print(f"zonotope: truth inside the enclosure on {contained}/{len(ts)} steps "
          f"({100*contained/len(ts):.1f} %)")
    print(f"EKF 3-sigma: truth inside on {ekf_in.sum()}/{len(ts)} steps "
          f"({100*ekf_in.mean():.1f} %); during the cut-in "
          f"{100*ekf_in[man].mean():.1f} %")
    print(f"mean lateral width  zonotope {w_zono[m].mean():.2f} m   "
          f"EKF 6-sigma {(6*ek['sig_py'])[m].mean():.2f} m   "
          f"ratio {w_zono[m].mean()/(6*ek['sig_py'])[m].mean():.1f}x")
    print(f"generators after reduction: {min(ngen)}--{max(ngen)} (budget {M_MAX})")

    with open("zono_radar_cutin.csv", "w") as f:
        cols = ["t", "py_true", "zlo", "zhi", "ekf_lo", "ekf_hi",
                "rz_lo", "rz_hi", "re_lo", "re_hi", "e_ekf"]
        f.write(",".join(cols) + "\n")
        for k in range(len(ts)):
            yt = A["py_true"][k]
            f.write(f"{A['t'][k]:.6f},{yt:.6f},{A['zlo'][k]:.6f},"
                    f"{A['zhi'][k]:.6f},{ekf_lo[k]:.6f},{ekf_hi[k]:.6f},"
                    f"{A['zlo'][k]-yt:.6f},{A['zhi'][k]-yt:.6f},"
                    f"{ekf_lo[k]-yt:.6f},{ekf_hi[k]-yt:.6f},"
                    f"{ek['py_est'][k]-yt:.6f}\n")
    print("\nwrote zono_radar_cutin.csv")


if __name__ == "__main__":
    main()
