"""Hybrid classical--learning fusion: a detector with learned heteroscedastic noise feeding a Kalman tracker.

Chapter 4 (Learning-Based and Hybrid Sensor Fusion), worked example of the section
"Hybrid Classical--Learning Architectures".

A neural-style detector reports the (x, y) position of a leading vehicle in the ego frame.
Its error standard deviation is NOT constant: it grows with range and degrades sharply when
visibility drops (a fog patch in the middle of the sequence).  A small "uncertainty head" is
fitted on a training set (heteroscedastic regression on the quality indicators range and
visibility) and its prediction is used as the measurement covariance R_k of a constant-velocity
Kalman filter.  The filter is compared with an identical filter that uses a fixed R.

Outputs (written to output/ and copied by hand into data/ch04 and figures/ch04 of the book):
    hybrid_hetero_kf.csv   time series used by the pgfplots figure of the chapter
    hybrid_hetero_kf.pdf   matplotlib preview of the same data
All numbers quoted in the chapter are printed by this script (seed fixed).
"""
# %% [markdown]
# # Learned heteroscedastic detector noise feeding a Kalman tracker
# Self-contained: only numpy / matplotlib are needed.

# %%
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(4)

# %% [markdown]
# ## 1. Scenario and true (unknown) noise model of the detector
# Constant-velocity target closing on the ego vehicle; the detector's error std depends on
# range r and visibility v in [0,1]:  sigma(r, v) = 0.15 + 0.010 r + 2.0 (1 - v)  [m].

# %%
dt, T = 0.1, 20.0
t = np.arange(0.0, T + 1e-9, dt)
N = len(t)


def true_sigma(r, v):
    return 0.15 + 0.010 * r + 2.0 * (1.0 - v)


def visibility(tt):
    """Clear (v=1) except a fog patch between 7 s and 13 s where v drops to 0.30."""
    v = np.ones_like(tt)
    fog = (tt >= 7.0) & (tt <= 13.0)
    v[fog] = 0.30
    return v


def simulate(tt, rng):
    px = 70.0 - 2.5 * tt          # closing at 2.5 m/s
    py = 3.5 - 0.05 * tt          # slow lateral drift
    vis = visibility(tt)
    r = np.hypot(px, py)
    sig = true_sigma(r, vis)
    meas = np.stack([px, py], 1) + sig[:, None] * rng.standard_normal((len(tt), 2))
    return px, py, vis, r, sig, meas


# %% [markdown]
# ## 2. "Training" the uncertainty head
# We simulate a training set (detections with known ground truth) and fit a linear model
# sigma_hat = theta_0 + theta_1 r + theta_2 (1 - v) by least squares on |e| sqrt(pi/2),
# which is an unbiased estimate of sigma for Gaussian errors.  This is the simplest possible
# heteroscedastic regression; a network minimising the Gaussian NLL would play the same role.

# %%
r_tr = rng.uniform(10, 75, 4000)
v_tr = rng.choice([1.0, 0.8, 0.6, 0.30], 4000)
sig_tr = true_sigma(r_tr, v_tr)
e_tr = sig_tr * rng.standard_normal(4000)
Phi = np.stack([np.ones_like(r_tr), r_tr, 1.0 - v_tr], 1)
theta, *_ = np.linalg.lstsq(Phi, np.abs(e_tr) * np.sqrt(np.pi / 2), rcond=None)
print("fitted uncertainty head theta = [%.3f, %.4f, %.3f]  (true [0.150, 0.0100, 2.000])" % tuple(theta))
sigma_fixed = np.sqrt(np.mean(e_tr ** 2))     # homoscedastic alternative: one std for everything
print("fixed-R alternative: pooled training std = %.3f m" % sigma_fixed)


def learned_sigma(r, v):
    return np.maximum(0.05, theta[0] + theta[1] * r + theta[2] * (1.0 - v))


# %% [markdown]
# ## 3. Constant-velocity Kalman filter with time-varying R_k

# %%
F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0.0]])
qa = 0.8  # acceleration noise std [m/s^2]
Qb = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * qa ** 2
Q = np.zeros((4, 4))
Q[np.ix_([0, 2], [0, 2])] = Qb
Q[np.ix_([1, 3], [1, 3])] = Qb


def run_kf(meas, sig_seq):
    """sig_seq[k] is the std used for R_k = sig^2 I at step k."""
    x = np.array([meas[0, 0], meas[0, 1], 0.0, 0.0])
    P = np.diag([sig_seq[0] ** 2, sig_seq[0] ** 2, 4.0, 4.0])
    est = np.zeros((len(meas), 4))
    nis = np.zeros(len(meas))
    est[0] = x
    nis[0] = np.nan
    for k in range(1, len(meas)):
        x = F @ x
        P = F @ P @ F.T + Q
        R = np.eye(2) * sig_seq[k] ** 2
        nu = meas[k] - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        nis[k] = float(nu @ np.linalg.solve(S, nu))
        x = x + K @ nu
        P = (np.eye(4) - K @ H) @ P
        est[k] = x
    return est, nis


px, py, vis, r, sig, meas = simulate(t, rng)
sig_hat = learned_sigma(r, vis)
est_fix, nis_fix = run_kf(meas, np.full(N, sigma_fixed))
est_ada, nis_ada = run_kf(meas, sig_hat)
est_orc, nis_orc = run_kf(meas, sig)          # oracle: true sigma (upper bound)

# %% [markdown]
# ## 4. Metrics: position RMSE, NIS consistency (chi^2 with 2 dof: 95 % interval [0.0506, 7.378])

# %%
truth = np.stack([px, py], 1)
fog = (t >= 7.0) & (t <= 13.0)


def rmse(e, mask=None):
    if mask is None:
        mask = np.ones(len(e), bool)
    return float(np.sqrt(np.mean(np.sum(e[mask] ** 2, 1))))


lo, hi = 0.0506, 7.378
res = {}
for name, est, nis in [("fixed R", est_fix, nis_fix), ("learned R", est_ada, nis_ada), ("oracle R", est_orc, nis_orc)]:
    e = est[:, :2] - truth
    ok = ~np.isnan(nis)
    res[name] = dict(rmse_all=rmse(e), rmse_clear=rmse(e, ~fog), rmse_fog=rmse(e, fog),
                     anis=float(np.nanmean(nis)), inside=float(np.mean((nis[ok] >= lo) & (nis[ok] <= hi))),
                     rmse_meas=rmse(meas - truth))
    print("%-10s  RMSE all %.3f m | clear %.3f | fog %.3f || ANIS %.2f  inside-95%% %.1f%%"
          % (name, res[name]["rmse_all"], res[name]["rmse_clear"], res[name]["rmse_fog"],
             res[name]["anis"], 100 * res[name]["inside"]))
print("raw detections: RMSE all %.3f m | clear %.3f | fog %.3f" % (rmse(meas - truth), rmse(meas - truth, ~fog), rmse(meas - truth, fog)))
print("ANIS fixed-R in fog %.2f, clear %.2f; learned-R in fog %.2f, clear %.2f" % (
    np.nanmean(nis_fix[fog]), np.nanmean(nis_fix[~fog]), np.nanmean(nis_ada[fog]), np.nanmean(nis_ada[~fog])))
print("sigma at t=0 (r=%.1f m): true %.2f, learned %.2f; in fog at t=10 s (r=%.1f m): true %.2f, learned %.2f"
      % (r[0], sig[0], sig_hat[0], r[100], sig[100], sig_hat[100]))

# %% [markdown]
# ## 5. Evidence-theoretic conflict monitor (camera vs LiDAR confidence for the same track)
# BBAs on Omega = {vehicle, clutter}: m_i({veh}) = a_i c_i, m_i({clutter}) = a_i (1 - c_i), m_i(Omega) = 1 - a_i,
# with discount a_i tied to the quality indicator.  Conflict K = m_c(veh) m_l(clu) + m_c(clu) m_l(veh).

# %%
def conflict(c_cam, a_cam, c_lid, a_lid):
    mc = (a_cam * c_cam, a_cam * (1 - c_cam))
    ml = (a_lid * c_lid, a_lid * (1 - c_lid))
    return mc[0] * ml[1] + mc[1] * ml[0]


for label, c_cam, a_cam, c_lid, a_lid in [("clear", 0.92, 0.95, 0.90, 0.95),
                                           ("fog, undiscounted camera", 0.35, 0.95, 0.88, 0.95),
                                           ("fog, discounted camera", 0.35, 0.40, 0.88, 0.95)]:
    print("conflict K (%s) = %.3f" % (label, conflict(c_cam, a_cam, c_lid, a_lid)))

# %% [markdown]
# ## 6. Export CSV for the book figure and a matplotlib preview

# %%
cols = dict(t=t, px_true=px, py_true=py, px_meas=meas[:, 0], py_meas=meas[:, 1],
            px_fix=est_fix[:, 0], py_fix=est_fix[:, 1], px_ada=est_ada[:, 0], py_ada=est_ada[:, 1],
            vis=vis, sig_true=sig, sig_hat=sig_hat,
            err_fix=np.hypot(*(est_fix[:, :2] - truth).T), err_ada=np.hypot(*(est_ada[:, :2] - truth).T),
            err_meas=np.hypot(*(meas - truth).T),
            nis_fix=np.nan_to_num(nis_fix, nan=0.0), nis_ada=np.nan_to_num(nis_ada, nan=0.0))
header = ",".join(cols)
data = np.stack(list(cols.values()), 1)
np.savetxt(os.path.join(OUT, "hybrid_hetero_kf.csv"), data, delimiter=",", header=header, comments="", fmt="%.6f")
print("rows written:", data.shape[0])

fig, ax = plt.subplots(3, 1, figsize=(6.5, 7), sharex=True)
ax[0].plot(t, sig, "k", label="true sigma"); ax[0].plot(t, sig_hat, "b--", label="learned sigma")
ax[0].axhline(sigma_fixed, color="grey", ls=":", label="fixed sigma"); ax[0].set_ylabel("sigma [m]"); ax[0].legend()
ax[1].plot(t, cols["err_meas"], color="0.6", lw=0.5, label="detector"); ax[1].plot(t, cols["err_fix"], "r", lw=0.8, label="KF fixed R")
ax[1].plot(t, cols["err_ada"], "b", lw=0.8, label="KF learned R"); ax[1].set_ylabel("position error [m]"); ax[1].legend()
ax[2].axhspan(lo, hi, color="0.9"); ax[2].plot(t, nis_fix, "r", lw=0.6, label="fixed R"); ax[2].plot(t, nis_ada, "b", lw=0.6, label="learned R")
ax[2].set_ylabel("NIS"); ax[2].set_xlabel("t [s]"); ax[2].set_ylim(0, 40); ax[2].legend()
for a in ax:
    a.axvspan(7, 13, color="orange", alpha=0.15)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "hybrid_hetero_kf.pdf"))
print("done")
