#!/usr/bin/env python3
"""
EKF tracking of a cut-in vehicle from automotive-radar detections
=================================================================

Reproduces **Example ex:ekf-radar-cutin** ("Extended Kalman filtering of an
automotive-radar cut-in") and **Fig. fig:ekf-radar-cutin** of Chapter 3
(Bayesian Filtering) of *Towards a Unified View of Sensor Fusion*.

Autonomous-driving counterpart of the classical "tracking an aircraft from radar
measurements" Kalman-filter case study: the constant-velocity target model is
kept, but the radar is modelled with its native polar + Doppler output (range,
azimuth, range rate), so the measurement model is nonlinear and an EKF is
required.  Scenario, models and parameters live in ``snfusion.scenarios``; the
filter is ``snfusion.kalman.ExtendedKalmanFilter``, whose update is, operation
for operation, the one of the authors' original script (fixed seed 20260727).

Outputs
-------
``output/ekf_radar_cutin.csv``  -- the table read by the pgfplots figure
(``data/ch03/ekf_radar_cutin.csv`` in the book sources); columns
``t, px_true, py_true, vx_true, vy_true, px_meas, py_meas, px_est, py_est,
vx_est, vy_est, sig_px, sig_py, nis``.
``output/ekf_radar_cutin.png``  -- a matplotlib preview of the two panels.
The summary numbers quoted in the example text are printed.

Run:  python3 ekf_radar_cutin.py
"""

# %% [markdown]
# ## Setup
# The repository root is put on `sys.path` so that `snfusion` is importable
# without installation (`pip install -e .` makes this unnecessary).

# %%
import sys
from pathlib import Path

import numpy as np

try:
    HERE = Path(__file__).resolve().parent
except NameError:                     # running inside a notebook
    HERE = Path.cwd()
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

from snfusion.kalman import ExtendedKalmanFilter          # noqa: E402
from snfusion.metrics import chi2_bounds                   # noqa: E402
from snfusion.scenarios import (                           # noqa: E402
    CUT_T0, CUT_T1, N, RNG_SEED, SIG_A, T,
    F_matrix, H_jacobian, Q_matrix, R_matrix, h, initial_estimate, nominal_run)

# %% [markdown]
# ## Scenario and models
# Constant-velocity transition $\mathbf F$, white-noise-acceleration $\mathbf Q$
# ($\sigma_a = 0.8$ m/s$^2$) and radar covariance $\mathbf R$
# ($\sigma_r = 0.30$ m, $\sigma_\theta = 0.5^\circ$, $\sigma_{\dot r} = 0.10$ m/s).

# %%
F = F_matrix(T)
Q = Q_matrix(T, SIG_A)
R = R_matrix()

# ground truth and noisy detections (seeded: the book figure is reproducible bit for bit)
ts, Xt, Z, _ = nominal_run(RNG_SEED)

# %% [markdown]
# ## Extended Kalman filter
# The process model is linear, so the prediction is the exact Kalman prediction;
# the Jacobian of eq. (ekf-radar-jacobian) is evaluated at the predicted state and
# the azimuth innovation is wrapped to $(-\pi, \pi]$.  The track is initialised
# from the first detection with a deliberately large velocity variance.

# %%
ekf = ExtendedKalmanFilter(f=lambda x, u: F @ x, F_jac=lambda x, u: F,
                           h=h, H_jac=H_jacobian, Q=Q, R=R, angle_idx=(1,))

x_hat, P = initial_estimate(Z[0], p0_vel=10.0)

rows = []
for k, t in enumerate(ts):
    if k > 0:
        x_hat, P = ekf.predict(x_hat, P)          # eqs. (kf_state_prediction)-(kf_cov_prediction)
    upd = ekf.update(x_hat, P, Z[k])              # eqs. (kf_innovation)-(kf_cov_update), Joseph form
    x_hat, P, nis = upd.x, upd.P, upd.nis

    # raw detection expressed in Cartesian coordinates, for comparison
    zx, zy = Z[k, 0] * np.cos(Z[k, 1]), Z[k, 0] * np.sin(Z[k, 1])

    rows.append(dict(
        t=t,
        px_true=Xt[k, 0], py_true=Xt[k, 1],
        vx_true=Xt[k, 2], vy_true=Xt[k, 3],
        px_meas=zx, py_meas=zy,
        px_est=x_hat[0], py_est=x_hat[1],
        vx_est=x_hat[2], vy_est=x_hat[3],
        sig_px=np.sqrt(P[0, 0]), sig_py=np.sqrt(P[1, 1]),
        nis=nis,
    ))

# %% [markdown]
# ## CSV for the book figure

# %%
cols = list(rows[0].keys())
csv_path = OUT / "ekf_radar_cutin.csv"
with open(csv_path, "w") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join(f"{r[c]:.6f}" for c in cols) + "\n")
print(f"wrote {csv_path.relative_to(REPO)}")

# %% [markdown]
# ## Summary numbers quoted in the example text

# %%
A = {c: np.array([r[c] for r in rows]) for c in cols}
pos_err_meas = np.hypot(A["px_meas"] - A["px_true"], A["py_meas"] - A["py_true"])
pos_err_est = np.hypot(A["px_est"] - A["px_true"], A["py_est"] - A["py_true"])
settle = A["t"] >= 1.0          # ignore the initial transient


def rms(v):
    return float(np.sqrt(np.mean(v ** 2)))


steady = (A["t"] >= 1.0) & (A["t"] < CUT_T0)
manoeuvr = (A["t"] >= CUT_T0) & (A["t"] <= CUT_T1)
lo95, hi95 = chi2_bounds(3, 0.95)             # two-sided 95 % interval of chi^2_3

print(f"steps                         : {N+1}")
print(f"position RMSE, raw detections : {rms(pos_err_meas[settle]):.3f} m")
print(f"position RMSE, EKF            : {rms(pos_err_est[settle]):.3f} m")
print(f"  -> reduction                : "
      f"{100*(1-rms(pos_err_est[settle])/rms(pos_err_meas[settle])):.0f} %")
print(f"lateral RMSE, raw detections  : {rms((A['py_meas']-A['py_true'])[settle]):.3f} m")
print(f"lateral RMSE, EKF             : {rms((A['py_est']-A['py_true'])[settle]):.3f} m")
print(f"vy RMSE, EKF (after 1 s)      : {rms((A['vy_est']-A['vy_true'])[settle]):.3f} m/s")
print(f"peak lateral error, cut-in    : {np.abs(A['py_est']-A['py_true'])[manoeuvr].max():.2f} m")
print(f"mean NIS, constant-velocity   : {A['nis'][steady].mean():.2f}  (E=3)")
print(f"mean NIS, during cut-in       : {A['nis'][manoeuvr].mean():.2f}  (E=3)")
print(f"max  NIS, during cut-in       : {A['nis'][manoeuvr].max():.2f}")
print(f"chi2_3 95% two-sided interval : [{lo95:.3f}, {hi95:.3f}]")
inside = (A["nis"] >= lo95) & (A["nis"] <= hi95)
print(f"NIS inside interval, CV       : {100*inside[steady].mean():.1f} %")
print(f"NIS inside interval, cut-in   : {100*inside[manoeuvr].mean():.1f} %")
print(f"chi2_3 95% upper bound        : 7.81")
print(f"NIS above 7.815, CV segment   : {100*float(np.mean(A['nis'][steady] > 7.815)):.1f} %")
print(f"NIS above 7.815, cut-in       : {100*float(np.mean(A['nis'][manoeuvr] > 7.815)):.1f} %")

# %% [markdown]
# ## Preview figure (the book draws the same data with pgfplots)

# %%
import matplotlib                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.6))
ax1.axhline(1.75, color="0.6", ls="--", lw=0.6)
ax1.plot(A["px_meas"], A["py_meas"], ".", ms=2, color="0.6", label="radar detections")
ax1.plot(A["px_true"], A["py_true"], "k-", lw=1.0, label="ground truth")
ax1.plot(A["px_est"], A["py_est"], "--", color="#1f569e", lw=0.9, label="EKF")
ax1.set(xlim=(8, 63), ylim=(-1.4, 6.4), xlabel="longitudinal range $p_x$ [m]",
        ylabel="lateral offset $p_y$ [m]", title="(a) bird's-eye view in the ego-vehicle frame")
ax1.legend(loc="upper left", fontsize=8, ncol=3, frameon=False)
ax2.axhspan(lo95, hi95, color="0.88", lw=0)
ax2.axvspan(CUT_T0, CUT_T1, color="#f4b183", alpha=0.45, lw=0)
ax2.axhline(3, color="0.5", ls=":", lw=0.6)
ax2.plot(A["t"], A["nis"], color="#1f569e", lw=0.6)
ax2.set(xlim=(0, 10), ylim=(0, 15), xlabel="time $t$ [s]", ylabel="NIS $d_k^2$",
        title=r"(b) innovation consistency against the $\chi^2_3$ interval")
fig.tight_layout()
fig.savefig(OUT / "ekf_radar_cutin.png", dpi=120)
print(f"wrote {(OUT / 'ekf_radar_cutin.png').relative_to(REPO)}")
