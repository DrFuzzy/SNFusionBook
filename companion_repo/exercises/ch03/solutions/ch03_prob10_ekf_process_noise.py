#!/usr/bin/env python3
"""Chapter 3 - Problem 3.10: the radar cut-in EKF (Example 3.1 of the book) with different process-noise levels.

The scenario of Example 3.1 is regenerated here (ego 27 m/s, target 22 m/s in the
adjacent lane, initially 60 m ahead, lane change from 3.5 m to 0 m between t = 5 s
and t = 8 s with a smooth cubic profile, 77 GHz radar at 20 Hz with
sigma_r = 0.30 m, sigma_theta = 0.5 deg, sigma_rdot = 0.10 m/s).  The same
detection sequence (fixed seed) is then processed by the constant-velocity EKF
for several values of the per-interval random-acceleration standard deviation sigma_a, and the
normalised innovation squared (NIS) is analysed separately on the constant-velocity
segments and inside the lane-change window.

Self-contained (numpy/matplotlib only); deterministic (seed 1).  Convert with
``python3 tools/py2nb.py exercises/ch03/solutions/ch03_prob10_ekf_process_noise.py``.
"""
# %%
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 1
T, N = 0.05, 201                      # 20 Hz, 10 s
SIG_R, SIG_TH, SIG_RD = 0.30, np.deg2rad(0.5), 0.10
CHI2_3_LO, CHI2_3_HI = 0.216, 9.348    # two-sided 95 % interval of chi^2_3

# %% [markdown]
# ## Scenario and radar detections (identical for every filter)

# %%
def ground_truth():
    t = np.arange(N) * T
    px = 60.0 - 5.0 * t                                  # relative longitudinal position
    tau = np.clip((t - 5.0) / 3.0, 0.0, 1.0)
    s = 3 * tau**2 - 2 * tau**3                          # cubic smoothstep lane change
    py = 3.5 * (1.0 - s)
    vx = -5.0 * np.ones(N)
    ds = np.where((t > 5.0) & (t < 8.0), (6 * tau - 6 * tau**2) / 3.0, 0.0)
    vy = -3.5 * ds
    return t, np.column_stack([px, py, vx, vy])

def g(x):
    px, py, vx, vy = x
    d = np.hypot(px, py)
    return np.array([d, np.arctan2(py, px), (px * vx + py * vy) / d])

def jacobian(x):
    px, py, vx, vy = x
    d = np.hypot(px, py)
    return np.array([[px / d, py / d, 0, 0],
                     [-py / d**2, px / d**2, 0, 0],
                     [py * (vx * py - vy * px) / d**3, px * (vy * px - vx * py) / d**3, px / d, py / d]])

t, X = ground_truth()
rng = np.random.default_rng(SEED)
R = np.diag([SIG_R**2, SIG_TH**2, SIG_RD**2])
Z = np.array([g(x) for x in X]) + rng.standard_normal((N, 3)) * np.sqrt(np.diag(R))
manoeuvre = (t >= 5.0) & (t <= 8.0)
print("peak lateral acceleration of the manoeuvre: %.2f m/s^2" % np.abs(np.gradient(X[:, 3], T)).max())

# %% [markdown]
# ## Constant-velocity EKF

# %%
def run_ekf(sigma_a):
    F = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    Q = sigma_a**2 * np.array([[T**4 / 4, 0, T**3 / 2, 0], [0, T**4 / 4, 0, T**3 / 2],
                               [T**3 / 2, 0, T**2, 0], [0, T**3 / 2, 0, T**2]])
    r0, th0, _ = Z[0]
    x = np.array([r0 * np.cos(th0), r0 * np.sin(th0), 0.0, 0.0])
    J0 = np.array([[np.cos(th0), -r0 * np.sin(th0)], [np.sin(th0), r0 * np.cos(th0)]])
    P = np.diag([0.0, 0.0, 10.0**2, 10.0**2])
    P[:2, :2] = J0 @ R[:2, :2] @ J0.T  # propagate the polar initiation uncertainty
    est = np.zeros((N, 4)); nis = np.zeros(N); sig = np.zeros((N, 4))
    est[0] = x; sig[0] = np.sqrt(np.diag(P))
    for k in range(1, N):
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        H = jacobian(x)
        nu = Z[k] - g(x)
        nu[1] = (nu[1] + np.pi) % (2 * np.pi) - np.pi      # azimuth wrapping
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ nu
        IKH = np.eye(4) - K @ H
        P = IKH @ P @ IKH.T + K @ R @ K.T                 # Joseph form
        est[k] = x; nis[k] = nu @ np.linalg.solve(S, nu); sig[k] = np.sqrt(np.diag(P))
    return est, nis, sig

def summarise(sigma_a):
    est, nis, sig = run_ekf(sigma_a)
    err = est[:, :2] - X[:, :2]
    steady = (~manoeuvre) & (t >= 1.0)
    inside = (nis >= CHI2_3_LO) & (nis <= CHI2_3_HI)
    return dict(sigma_a=sigma_a,
                rmse_pos=np.sqrt((err**2).sum(axis=1).mean()),
                rmse_pos_steady=np.sqrt((err[steady]**2).sum(axis=1).mean()),
                rmse_pos_man=np.sqrt((err[manoeuvre]**2).sum(axis=1).mean()),
                peak_lat=np.abs(err[manoeuvre, 1]).max(),
                nis_steady=nis[steady].mean(), nis_man=nis[manoeuvre].mean(), nis_peak=nis.max(),
                in_steady=100 * inside[steady].mean(), in_man=100 * inside[manoeuvre].mean(),
                sig_y_steady=sig[steady][-20:, 1].mean(),
                nis=nis, est=est, sig=sig)

# %% [markdown]
# ## Sweep of the process-noise intensity

# %%
sweep = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
rows = [summarise(s) for s in sweep]
hdr = "sigma_a  RMSE_all  RMSE_cv  RMSE_lc  peak|e_y|_lc NIS_cv  NIS_lc  NIS_peak  in95_cv  in95_lc  sig_y_cv"
print(hdr)
for r in rows:
    print("%6.2f  %8.3f  %7.3f  %7.3f  %8.3f  %6.2f  %6.2f  %8.1f  %6.1f%%  %6.1f%%  %7.3f"
          % (r["sigma_a"], r["rmse_pos"], r["rmse_pos_steady"], r["rmse_pos_man"], r["peak_lat"],
             r["nis_steady"], r["nis_man"], r["nis_peak"], r["in_steady"], r["in_man"], r["sig_y_steady"]))
print("expected NIS under a consistent chi^2_3 model: 3.0 ; 95 %% interval [%.3f, %.3f]" % (CHI2_3_LO, CHI2_3_HI))

# %% [markdown]
# ## Figure: NIS traces for three representative values

# %%
fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.0), sharex=True)
for ax, s in zip(axes, [0.2, 0.8, 3.2]):
    r = rows[sweep.index(s)]
    ax.axvspan(5, 8, color="#f4b183", alpha=0.45, lw=0)
    ax.axhspan(CHI2_3_LO, CHI2_3_HI, color="#e1e1e1", lw=0)
    ax.axhline(3.0, color="grey", ls=":", lw=0.8)
    ax.plot(t, r["nis"], color="#1f569e", lw=0.8)
    ax.set_ylim(0, 30); ax.set_ylabel("NIS")
    ax.set_title(r"$\sigma_a=%.1f$ m/s$^2$: NIS %.2f (CV) / %.2f (lane change), RMSE %.3f m"
                 % (s, r["nis_steady"], r["nis_man"], r["rmse_pos"]), fontsize=9)
axes[-1].set_xlabel("time t [s]")
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".",
                   "ch03_prob10_nis_vs_sigma_a.png")
fig.savefig(out, dpi=150)
print("figure written to", out)
