#!/usr/bin/env python3
"""
UKF companion to the radar cut-in example -- same scenario, same data, same tuning
=================================================================================

Reproduces **Example ex:ukf-radar-cutin** ("Does the unscented transform help
here?") of Chapter 3.  An unscented Kalman filter is run on exactly the
measurement sequence the EKF of Example ex:ekf-radar-cutin sees, so the two are
directly comparable, and the comparison is then repeated on a stressed near-field
variant in which the azimuth nonlinearity is no longer mild.

The process model is linear (constant velocity), so the prediction step is the
exact Kalman prediction and the unscented transform is applied only to the
nonlinear measurement update (``UnscentedKalmanFilter(..., linear_predict=F)``).

Sigma-point parameters: ``alpha = 1, beta = 2, kappa = 1``, hence ``lambda = 1``
and all weights are non-negative (with the frequently quoted ``alpha = 1e-3`` the
sigma points collapse onto the mean and the UKF degenerates numerically into a
central-difference version of the EKF).

Outputs
-------
``output/ukf_radar_cutin.csv`` (``data/ch03/ukf_radar_cutin.csv`` in the book
sources): nominal and stressed lateral errors of both filters and the stressed
position errors; columns ``t, ekf_ey_nom, ukf_ey_nom, ekf_ey_str, ukf_ey_str,
ekf_ep_str, ukf_ep_str``.  ``output/ukf_radar_cutin.png`` is a preview.  The
numbers quoted in the example are printed. The optional --montecarlo flag runs
100 paired realisations per update rate, seeds 1000..1099, and saves
output/ukf_rate_sweep.csv. RMSE is the mean of per-run Euclidean position RMSEs
on t >= 1 s; sigma_y is the mean predicted lateral standard deviation on the
same samples. Both filters use the first detection only for initiation.
Run:  python3 ukf_radar_cutin.py [--montecarlo]
"""

# %% [markdown]
# ## Setup

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

from snfusion.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter   # noqa: E402
from snfusion.scenarios import (                                          # noqa: E402
    N, RNG_SEED, SIG_A, SIG_R, SIG_RD, SIG_TH, T,
    F_matrix, H_jacobian, Q_matrix, h, simulate_measurements, truth, initial_estimate)

ALPHA, BETA, KAPPA = 1.0, 2.0, 1.0

# %% [markdown]
# ## One scenario, two filters
# `run()` simulates a trajectory, draws one detection sequence and filters it
# with the EKF (first-order update, identical to Example ex:ekf-radar-cutin) and
# with the UKF.  Both covariances are symmetrised after each update, as in the
# authors' script.

# %%
def run(px0, py0, vx0, sig_th, p0_vel, seed, n_steps, cut=True, dt=T, continuous_q=None):
    """Simulate one scenario and filter it with both an EKF and a UKF."""
    rng = np.random.default_rng(seed)
    F, Q = F_matrix(dt), Q_matrix(dt, SIG_A)
    if continuous_q is not None:
        Q = continuous_q * np.array([[dt**3/3, 0, dt**2/2, 0],
                                    [0, dt**3/3, 0, dt**2/2],
                                    [dt**2/2, 0, dt, 0], [0, dt**2/2, 0, dt]])
    R = np.diag([SIG_R ** 2, sig_th ** 2, SIG_RD ** 2])
    ts = np.arange(n_steps + 1) * dt

    def local_truth(t):
        if cut:
            x = truth(t)
            return np.array([px0 + vx0 * t, x[1], vx0, x[3]])
        return np.array([px0 + vx0 * t, py0, vx0, 0.0])

    Xt = np.array([local_truth(t) for t in ts])
    Z, _ = simulate_measurements(Xt, rng, sig_th=sig_th)

    ekf = ExtendedKalmanFilter(f=lambda x, u: F @ x, F_jac=lambda x, u: F,
                               h=h, H_jac=H_jacobian, Q=Q, R=R, angle_idx=(1,))
    ukf = UnscentedKalmanFilter(f=None, h=h, Q=Q, R=R, alpha=ALPHA, beta=BETA, kappa=KAPPA,
                                angle_idx=(1,), linear_predict=F)

    out = {}
    for name, filt in (("EKF", ekf), ("UKF", ukf)):
        x, P = initial_estimate(Z[0], p0_vel, SIG_R, sig_th)
        est, nis, sy_pred = [], [], []
        for k in range(len(ts)):
            if k > 0:
                x, P = filt.predict(x, P)
                sy_pred.append(np.sqrt(P[1, 1]))
                upd = filt.update(x, P, Z[k])
                x, P = upd.x, 0.5 * (upd.P + upd.P.T)
            if k == 0:
                sy_pred.append(np.sqrt(P[1, 1]))
            est.append(x.copy())
            nis.append(upd.nis if k > 0 else 0.0)  # no re-use of initiation detection
        out[name] = (np.array(est), np.array(nis))
    run.last_sigma_y = np.asarray(sy_pred)  # same process model; final loop is UKF
    return ts, Xt, Z, out


def rms(v):
    return float(np.sqrt(np.mean(np.asarray(v) ** 2)))


def report(tag, ts, Xt, out, t_from=1.0):
    m = ts >= t_from
    line = f"{tag:<34s}"
    for name in ("EKF", "UKF"):
        E, nis = out[name]
        pos = np.hypot(E[:, 0] - Xt[:, 0], E[:, 1] - Xt[:, 1])
        line += f"  {name} pos-RMSE {rms(pos[m]):6.3f} m  meanNIS {np.mean(nis[m]):5.2f}"
    print(line)
    return out

# %% [markdown]
# ## A. Nominal scenario (identical to Example ex:ekf-radar-cutin)

# %%
print("=" * 108)
print("A. NOMINAL SCENARIO  (identical to Example ex:ekf-radar-cutin: 60 m ahead, sigma_theta = 0.5 deg)")
ts, Xt, Z, outA = run(60.0, 3.5, -5.0, SIG_TH, 10.0, RNG_SEED, N)
report("   nominal", ts, Xt, outA)

# %% [markdown]
# ## B. Stressed near-field variant
# The target starts 12 m ahead in the adjacent lane (azimuth about 16 degrees and
# growing), the azimuth noise is 3 degrees and the initialisation is poor.

# %%
print()
print("B. STRESSED NEAR-FIELD VARIANT (12 m ahead, sigma_theta = 3 deg, poor initialisation)")
ts2, Xt2, Z2, outB = run(12.0, 3.5, -0.9, np.deg2rad(3.0), 25.0, RNG_SEED + 1, N)
report("   stressed", ts2, Xt2, outB)

# %% [markdown]
# ## Sigma-point weights

# %%
print()
print("Sigma-point weights (n = 4, alpha = %.1f, beta = %.1f, kappa = %.1f):" % (ALPHA, BETA, KAPPA))
lam, wm, wc = UnscentedKalmanFilter(None, h, np.eye(4), np.eye(3), ALPHA, BETA, KAPPA).weights(4)
print(f"   lambda = {lam:.3f}   w_m0 = {wm[0]:.4f}   w_i = {wm[1]:.4f}   "
      f"spread = {np.sqrt(4 + lam):.3f} sigma")

# %% [markdown]
# ## CSV for the figure: nominal EKF vs UKF lateral error, plus the stressed case

# %%
EA, nisA = outA["EKF"]
UA, nisUA = outA["UKF"]
EB, nisB = outB["EKF"]
UB, nisUB = outB["UKF"]
cols = ["t", "ekf_ey_nom", "ukf_ey_nom", "ekf_ey_str", "ukf_ey_str", "ekf_ep_str", "ukf_ep_str"]
csv_path = OUT / "ukf_radar_cutin.csv"
with open(csv_path, "w") as f:
    f.write(",".join(cols) + "\n")
    for k, t in enumerate(ts):
        row = [t,
               EA[k, 1] - Xt[k, 1], UA[k, 1] - Xt[k, 1],
               EB[k, 1] - Xt2[k, 1], UB[k, 1] - Xt2[k, 1],
               np.hypot(EB[k, 0] - Xt2[k, 0], EB[k, 1] - Xt2[k, 1]),
               np.hypot(UB[k, 0] - Xt2[k, 0], UB[k, 1] - Xt2[k, 1])]
        f.write(",".join(f"{v:.6f}" for v in row) + "\n")
print(f"\nwrote {csv_path.relative_to(REPO)}")

# %% [markdown]
# ## Optional: Monte Carlo table tab:ekf-ukf-rate (update rate varied, 100 runs per row)
# Run the script with `--montecarlo` (or set `MONTE_CARLO = True` in a notebook).

# %%
MONTE_CARLO = "--montecarlo" in sys.argv
if MONTE_CARLO:
    print("\nTable tab:ekf-ukf-rate: EKF vs UKF position RMSE as the update rate is reduced")
    print(f"{'rate':>8s} {'EKF RMSE [m]':>14s} {'UKF RMSE [m]':>14s}")
    sweep_rows = []
    for rate in (20.0, 5.0, 1.0, 0.5):
        dt = 1.0 / rate
        n_steps = int(round(10.0 / dt))
        e_ekf, e_ukf, sigma_y = [], [], []
        for run_id in range(100):
            tsr, Xr, _, o = run(60.0, 3.5, -5.0, SIG_TH, 10.0, 1000 + run_id, n_steps, dt=dt, continuous_q=SIG_A**2*T)
            m = tsr >= 1.0
            sigma_y.append(np.mean(run.last_sigma_y[m]))
            for name, acc in (("EKF", e_ekf), ("UKF", e_ukf)):
                E, _ = o[name]
                acc.append(rms(np.hypot(E[m, 0] - Xr[m, 0], E[m, 1] - Xr[m, 1])))
        sweep_rows.append([rate, np.mean(sigma_y), np.mean(e_ekf), np.mean(e_ukf)])
        print(f"{rate:6.1f} Hz {np.mean(e_ekf):14.3f} {np.mean(e_ukf):14.3f} sigma_y={np.mean(sigma_y):.3f}")
    np.savetxt(OUT / "ukf_rate_sweep.csv", sweep_rows, delimiter=",",
               header="rate_hz,sigma_y,ekf_rmse,ukf_rmse", comments="", fmt="%.9f")

# %% [markdown]
# ## Preview figure

# %%
import matplotlib                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True)
ax1.plot(ts, EA[:, 1] - Xt[:, 1], color="#1f569e", lw=0.8, label="EKF")
ax1.plot(ts, UA[:, 1] - Xt[:, 1], color="#c0504d", lw=0.8, ls="--", label="UKF")
ax1.set(ylabel="lateral error [m]", title="nominal scenario: the two filters are indistinguishable")
ax1.legend(frameon=False, fontsize=8)
ax2.plot(ts, np.hypot(EB[:, 0] - Xt2[:, 0], EB[:, 1] - Xt2[:, 1]), color="#1f569e", lw=0.8, label="EKF")
ax2.plot(ts, np.hypot(UB[:, 0] - Xt2[:, 0], UB[:, 1] - Xt2[:, 1]), color="#c0504d", lw=0.8, ls="--", label="UKF")
ax2.set(xlabel="time $t$ [s]", ylabel="position error [m]", title="stressed near-field variant")
ax2.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "ukf_radar_cutin.png", dpi=120)
print(f"wrote {(OUT / 'ukf_radar_cutin.png').relative_to(REPO)}")
