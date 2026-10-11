"""Ego-localisation by GNSS--IMU--wheel-odometry fusion with a GNSS outage.

Chapter 5 ("Autonomous Driving as a Representative Application"), worked example
on planar dead reckoning corrected by GNSS position fixes.  A five-state EKF
(position, heading, gyroscope bias, wheel-speed scale error) is driven at 100 Hz
by the IMU yaw rate and the wheel speed and corrected at 2 Hz by GNSS fixes.
A 25 s GNSS outage (tunnel) is simulated.  A three-state filter that ignores the
sensor biases is run on the same data for comparison.

Outputs (written to ./output and copied by hand to data/ch05 and figures/ch05):
    gnss_imu_odometry.csv   -- one row every 0.5 s (240 rows), read by pgfplots
    gnss_imu_odometry.pdf   -- matplotlib version of the book figure
The script is deterministic (np.random.default_rng(5)).
"""
# %% [markdown]
# # Ego-localisation by GNSS--IMU--wheel-odometry fusion (Chapter 5)
#
# Planar unicycle model driven by the measured wheel speed $v_m$ and the measured
# yaw rate $\omega_m$:
#
# $$p_{x,k+1}=p_{x,k}+T\,\hat v\cos\psi_k,\quad p_{y,k+1}=p_{y,k}+T\,\hat v\sin\psi_k,\quad
#   \psi_{k+1}=\psi_k+T(\omega_m-b_k),\quad \hat v=\frac{v_m}{1+s_k}$$
#
# with gyroscope bias $b$ and wheel-speed scale error $s$ modelled as random walks.
# GNSS gives $\mathbf{y}=[p_x,p_y]^T+\mathbf{r}$ at 2 Hz, except during a tunnel.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # running inside a notebook
    HERE = os.getcwd()
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(5)

# %% [markdown]
# ## Scenario and sensor parameters (indicative automotive-grade values)

# %%
T = 0.01                 # IMU / wheel-odometry period [s] (100 Hz)
T_END = 120.0            # duration [s]
N = int(round(T_END / T))
GNSS_EVERY = 50          # one GNSS fix every 50 IMU samples -> 2 Hz
SIG_GNSS = 1.5           # GNSS position noise per axis [m] (single-frequency, open sky)
OUTAGE = (50.0, 75.0)    # tunnel: no GNSS fixes in this interval [s]

SIG_GYRO = 0.002         # gyroscope white noise per sample [rad/s]
B_TRUE = 0.010           # gyroscope bias [rad/s]  (about 0.57 deg/s)
SIG_WHEEL = 0.05         # wheel-speed white noise [m/s]
S_TRUE = 0.02            # wheel-speed scale error (+2 %, e.g. tyre-radius error)

# random-walk intensities assumed by the filter for the bias states
SIG_B_RW = 1e-4          # [rad/s / sqrt(s)]
SIG_S_RW = 1e-4          # [1 / sqrt(s)]
SIG_PSI_RW_5 = 1e-3      # heading random walk assumed by the 5-state filter [rad / sqrt(s)]
SIG_PSI_RW_3 = 5e-3      # heading random walk of the 3-state filter: inflated to absorb the unmodelled bias

# %% [markdown]
# ## Ground-truth trajectory: two curves, the second one inside the tunnel

# %%
t = np.arange(N) * T
v_true = 14.0 + 1.0 * np.sin(2 * np.pi * t / 40.0)          # 13--15 m/s
omega_true = np.zeros(N)
omega_true[(t >= 15) & (t < 35)] = +0.08                      # left curve (~92 deg)
omega_true[(t >= 55) & (t < 70)] = -0.06                      # right curve inside the tunnel (~52 deg)

px = np.zeros(N); py = np.zeros(N); psi = np.zeros(N)
for k in range(1, N):
    psi[k] = psi[k - 1] + T * omega_true[k - 1]
    px[k] = px[k - 1] + T * v_true[k - 1] * np.cos(psi[k - 1])
    py[k] = py[k - 1] + T * v_true[k - 1] * np.sin(psi[k - 1])

# sensor readings
omega_m = omega_true + B_TRUE + SIG_GYRO * rng.standard_normal(N)
v_m = v_true * (1.0 + S_TRUE) + SIG_WHEEL * rng.standard_normal(N)
gnss_avail = np.zeros(N, dtype=bool)
gnss_avail[::GNSS_EVERY] = True
gnss_avail &= ~((t >= OUTAGE[0]) & (t < OUTAGE[1]))
y_gnss = np.full((N, 2), np.nan)
y_gnss[gnss_avail, 0] = px[gnss_avail] + SIG_GNSS * rng.standard_normal(gnss_avail.sum())
y_gnss[gnss_avail, 1] = py[gnss_avail] + SIG_GNSS * rng.standard_normal(gnss_avail.sum())

# %% [markdown]
# ## EKF
#
# `estimate_bias=True` runs the five-state filter $[p_x,p_y,\psi,b,s]$;
# `estimate_bias=False` freezes $b=0$, $s=0$ (three-state filter, same data, with an
# inflated heading process noise so that it works while GNSS is available).

# %%
def run_ekf(estimate_bias=True):
    x = np.array([y_gnss[0, 0], y_gnss[0, 1], psi[0] + np.deg2rad(2.0), 0.0, 0.0])
    P = np.diag([SIG_GNSS**2, SIG_GNSS**2, np.deg2rad(5.0)**2, 0.02**2, 0.05**2])
    if not estimate_bias:
        P[3, 3] = P[4, 4] = 0.0
    H = np.zeros((2, 5)); H[0, 0] = 1.0; H[1, 1] = 1.0
    R = SIG_GNSS**2 * np.eye(2)
    W = np.diag([SIG_WHEEL**2, SIG_GYRO**2])         # input-noise covariance
    X = np.zeros((N, 5)); S3 = np.zeros((N, 5)); NIS = np.full(N, np.nan)
    X[0] = x; S3[0] = 3 * np.sqrt(np.diag(P))
    for k in range(1, N):
        # ---- prediction with the measured inputs of step k-1
        vm, wm = v_m[k - 1], omega_m[k - 1]
        s = x[4] if estimate_bias else 0.0
        b = x[3] if estimate_bias else 0.0
        vh = vm / (1.0 + s)
        c, sn = np.cos(x[2]), np.sin(x[2])
        x_pred = x.copy()
        x_pred[0] = x[0] + T * vh * c
        x_pred[1] = x[1] + T * vh * sn
        x_pred[2] = x[2] + T * (wm - b)
        F = np.eye(5)
        F[0, 2] = -T * vh * sn; F[1, 2] = T * vh * c
        if estimate_bias:
            F[2, 3] = -T
            F[0, 4] = -T * vm * c / (1.0 + s)**2
            F[1, 4] = -T * vm * sn / (1.0 + s)**2
        G = np.zeros((5, 2))
        G[0, 0] = T * c / (1.0 + s); G[1, 0] = T * sn / (1.0 + s); G[2, 1] = T
        Q = G @ W @ G.T
        if estimate_bias:
            Q[2, 2] += T * SIG_PSI_RW_5**2
            Q[3, 3] += T * SIG_B_RW**2
            Q[4, 4] += T * SIG_S_RW**2
        else:
            Q[2, 2] += T * SIG_PSI_RW_3**2
        x = x_pred
        P = F @ P @ F.T + Q
        # ---- GNSS update
        if gnss_avail[k]:
            nu = y_gnss[k] - H @ x
            S = H @ P @ H.T + R
            NIS[k] = nu @ np.linalg.solve(S, nu)
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ nu
            P = (np.eye(5) - K @ H) @ P
            P = 0.5 * (P + P.T)
        X[k] = x; S3[k] = 3 * np.sqrt(np.maximum(np.diag(P), 0.0))
    return X, S3, NIS

X5, S5, NIS5 = run_ekf(True)
X3, S3_, NIS3 = run_ekf(False)

# %% [markdown]
# ## Numbers quoted in the book

# %%
err5 = np.hypot(X5[:, 0] - px, X5[:, 1] - py)
err3 = np.hypot(X3[:, 0] - px, X3[:, 1] - py)
rad5 = np.sqrt(S5[:, 0]**2 + S5[:, 1]**2)      # 3-sigma radius sqrt(9 Pxx + 9 Pyy)
rad3 = np.sqrt(S3_[:, 0]**2 + S3_[:, 1]**2)
in_out = (t >= OUTAGE[0]) & (t < OUTAGE[1])
avail = (t >= 5) & ~in_out & (t < OUTAGE[0] - 0.01) | ((t >= OUTAGE[1] + 10) & (t < T_END))
k_end = np.searchsorted(t, OUTAGE[1]) - 1
k_re2 = np.searchsorted(t, OUTAGE[1] + 2.0)
k_re5 = np.searchsorted(t, OUTAGE[1] + 5.0)
nis_ok = NIS5[np.isfinite(NIS5)]
print(f"RMS position error with GNSS available, 5-state filter: {np.sqrt(np.mean(err5[avail]**2)):.2f} m")
print(f"RMS position error with GNSS available, 3-state filter: {np.sqrt(np.mean(err3[avail]**2)):.2f} m")
print(f"Position error at the end of the {OUTAGE[1]-OUTAGE[0]:.0f} s outage, 5-state: {err5[k_end]:.2f} m  (3-sigma radius {rad5[k_end]:.2f} m)")
print(f"Position error at the end of the outage, 3-state (no bias states): {err3[k_end]:.2f} m  (3-sigma radius {rad3[k_end]:.2f} m)")
print(f"Peak error during outage: 5-state {err5[in_out].max():.2f} m, 3-state {err3[in_out].max():.2f} m")
print(f"Error 2 s after re-acquisition: 5-state {err5[k_re2]:.2f} m, 3-state {err3[k_re2]:.2f} m")
print(f"Error 5 s after re-acquisition: 5-state {err5[k_re5]:.2f} m, 3-state {err3[k_re5]:.2f} m")
print(f"Gyro bias estimate at tunnel entry: {X5[np.searchsorted(t, OUTAGE[0]), 3]*1e3:.2f} mrad/s (true {B_TRUE*1e3:.2f}), 3-sigma {S5[np.searchsorted(t, OUTAGE[0]), 3]*1e3:.2f} mrad/s")
print(f"Scale error estimate at tunnel entry: {X5[np.searchsorted(t, OUTAGE[0]), 4]*100:.2f} % (true {S_TRUE*100:.2f} %), 3-sigma {S5[np.searchsorted(t, OUTAGE[0]), 4]*100:.2f} %")
print(f"Heading error at tunnel exit: 5-state {np.rad2deg(X5[k_end,2]-psi[k_end]):.2f} deg, 3-state {np.rad2deg(X3[k_end,2]-psi[k_end]):.2f} deg")
print(f"Fraction of samples with error inside the 3-sigma radius, 5-state: {np.mean(err5 <= rad5)*100:.1f} %, 3-state: {np.mean(err3 <= rad3)*100:.1f} %")
print(f"Mean NIS of the GNSS updates, 5-state: {nis_ok.mean():.2f} (expected 2), inside 95% chi2_2 interval: {np.mean((nis_ok>0.0506)&(nis_ok<7.378))*100:.1f} %")
print(f"Distance travelled inside the tunnel: {np.sum(v_true[in_out])*T:.0f} m")

# %% [markdown]
# ## CSV for the book figure (one row every 0.5 s) and a matplotlib version

# %%
idx = np.arange(0, N, GNSS_EVERY)
df = pd.DataFrame({
    "t": t[idx], "px_true": px[idx], "py_true": py[idx],
    "px_gnss": y_gnss[idx, 0], "py_gnss": y_gnss[idx, 1],
    "px_est": X5[idx, 0], "py_est": X5[idx, 1],
    "px_est3": X3[idx, 0], "py_est3": X3[idx, 1],
    "err": err5[idx], "rad3sig": rad5[idx], "err_nobias": err3[idx], "rad3sig_nobias": rad3[idx],
    "b_est_mrad": X5[idx, 3] * 1e3, "b_sig3_mrad": S5[idx, 3] * 1e3,
    "s_est_pct": X5[idx, 4] * 100, "s_sig3_pct": S5[idx, 4] * 100,
})
csv_path = os.path.join(OUT, "gnss_imu_odometry.csv")
df.to_csv(csv_path, index=False, float_format="%.4f", na_rep="nan")
print("rows written:", len(df), "->", csv_path)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ax[0].plot(px, py, "k", lw=1.2, label="truth")
ax[0].plot(y_gnss[:, 0], y_gnss[:, 1], ".", color="0.6", ms=3, label="GNSS fixes")
ax[0].plot(X5[:, 0], X5[:, 1], "--", color="#1f569e", lw=1, label="EKF (5 states)")
ax[0].plot(X3[:, 0], X3[:, 1], ":", color="#f4b183", lw=1.2, label="EKF (no bias states)")
ax[0].set_aspect("equal"); ax[0].set_xlabel("east [m]"); ax[0].set_ylabel("north [m]"); ax[0].legend(fontsize=7)
ax[1].axvspan(*OUTAGE, color="#f4b183", alpha=0.45)
ax[1].plot(t, rad5, color="0.7", lw=0.8, label=r"$3\sigma$ radius (5 states)")
ax[1].plot(t, err5, color="#1f569e", lw=1, label="error (5 states)")
ax[1].plot(t, err3, color="#c00000", lw=1, label="error (no bias states)")
ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("position error [m]"); ax[1].legend(fontsize=7)
ax[2].axvspan(*OUTAGE, color="#f4b183", alpha=0.45)
ax[2].fill_between(t, (X5[:, 3] - S5[:, 3]) * 1e3, (X5[:, 3] + S5[:, 3]) * 1e3, color="0.85")
ax[2].plot(t, X5[:, 3] * 1e3, color="#1f569e"); ax[2].axhline(B_TRUE * 1e3, color="k", lw=0.8)
ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("gyro bias [mrad/s]")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gnss_imu_odometry.pdf"))
print("figure written")
