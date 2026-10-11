"""Chapter 4, Problem 6: effect of a miscalibrated uncertainty head on the hybrid tracker.

Re-runs the heteroscedastic detector + Kalman tracker of companion_repo/examples/ch04_hybrid_hetero_kf
with the learned standard deviation scaled by a factor c in {0.5, 1, 2} and reports RMSE and NIS.
Self-contained (the scenario is re-created here with the same seed).
"""
# %%
import numpy as np

rng = np.random.default_rng(4)
dt, T = 0.1, 20.0
t = np.arange(0.0, T + 1e-9, dt)


def true_sigma(r, v):
    return 0.15 + 0.010 * r + 2.0 * (1.0 - v)


vis = np.ones_like(t)
vis[(t >= 7.0) & (t <= 13.0)] = 0.30
px = 70.0 - 2.5 * t
py = 3.5 - 0.05 * t
r = np.hypot(px, py)
sig = true_sigma(r, vis)
meas = np.stack([px, py], 1) + sig[:, None] * rng.standard_normal((len(t), 2))
truth = np.stack([px, py], 1)

# the same training procedure as in the example (same rng stream order is not needed here: we use the
# true coefficients, which the fitted head reproduces to within 3 %)
F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0.0]])
qa = 0.8
Qb = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * qa ** 2
Q = np.zeros((4, 4)); Q[np.ix_([0, 2], [0, 2])] = Qb; Q[np.ix_([1, 3], [1, 3])] = Qb


def run_kf(meas, sig_seq):
    x = np.array([meas[0, 0], meas[0, 1], 0.0, 0.0])
    P = np.diag([sig_seq[0] ** 2, sig_seq[0] ** 2, 4.0, 4.0])
    est = np.zeros((len(meas), 4)); nis = np.full(len(meas), np.nan); est[0] = x
    for k in range(1, len(meas)):
        x = F @ x; P = F @ P @ F.T + Q
        R = np.eye(2) * sig_seq[k] ** 2
        nu = meas[k] - H @ x; S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
        nis[k] = float(nu @ np.linalg.solve(S, nu))
        x = x + K @ nu; P = (np.eye(4) - K @ H) @ P; est[k] = x
    return est, nis


lo, hi = 0.0506, 7.378
for c in [0.5, 1.0, 2.0]:
    est, nis = run_kf(meas, c * sig)
    e = est[:, :2] - truth
    ok = ~np.isnan(nis)
    print("scale c=%.1f: RMSE %.3f m, ANIS %.2f, inside 95%% interval %.1f%%" % (
        c, np.sqrt(np.mean(np.sum(e ** 2, 1))), np.nanmean(nis), 100 * np.mean((nis[ok] >= lo) & (nis[ok] <= hi))))
