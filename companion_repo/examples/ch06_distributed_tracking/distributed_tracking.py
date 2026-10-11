"""Distributed tracking of a 2-D constant-velocity target by a five-node sensor network.

Companion script for the worked example on distributed Kalman filtering of
Chapter 6 (Sensor Fusion in Multi-Agent Systems).  Five sensing nodes, connected
by a ring graph with one chord, observe a target that moves with (nearly)
constant velocity.  Every node observes the target only partially: nodes 1 and 4
measure the x coordinate only, nodes 2 and 5 the y coordinate only, and node 3
both coordinates but with a much larger noise.  The script compares

  * a centralised Kalman filter that processes all five measurements,
  * isolated local Kalman filters (no communication),
  * naive information fusion over the closed neighbourhood, which treats the
    neighbours' estimates as independent and therefore double-counts the
    common prior (overconfident),
  * the consensus-on-information filter of Battistelli and Chisci (2014) with
    Metropolis weights and L consensus steps per sampling interval,
  * the Kalman-consensus filter of Olfati-Saber (2007),
  * covariance-intersection fusion of the local posteriors (Julier and Uhlmann 1997).

Outputs (in output/ and copied into the book's data/ch06 and figures/ch06):
  distributed_tracking.csv   time series of run 0 (used by the pgfplots figure)
  distributed_tracking.pdf   matplotlib preview of the same quantities
Deterministic: numpy default_rng(seed) with a fixed seed.
"""
# %% [markdown]
# # Distributed tracking by a five-node sensor network
# Comparison of centralised, naive, consensus-on-information, Kalman-consensus and
# covariance-intersection fusion for a constant-velocity target observed partially
# by every node.  See Chapter 6 of the book.

# %%
import os
import shutil
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

SEED = 6
N_MC = 50          # Monte Carlo runs for the statistics quoted in the text
L_CONS = 3         # consensus steps per sampling interval (consensus on information)
EPS_KCF = 0.05     # consensus step of the Kalman-consensus filter

# %% [markdown]
# ## Target model and sensing nodes

# %%
T = 0.1            # sampling interval [s]
K = 100            # number of steps (10 s)
sigma_a = 0.5      # white-noise acceleration [m/s^2]
F = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1.0]])
Q = sigma_a**2 * np.array([[T**4 / 4, 0, T**3 / 2, 0],
                           [0, T**4 / 4, 0, T**3 / 2],
                           [T**3 / 2, 0, T**2, 0],
                           [0, T**3 / 2, 0, T**2]])
x0_true = np.array([0.0, 0.0, 8.0, 3.0])        # [px, py, vx, vy]
P0 = np.diag([25.0, 25.0, 4.0, 4.0])

# node i observes H_i x + noise with covariance R_i
H_nodes = [np.array([[1, 0, 0, 0.0]]),                        # node 1: x only
           np.array([[0, 1, 0, 0.0]]),                        # node 2: y only
           np.array([[1, 0, 0, 0.0], [0, 1, 0, 0.0]]),        # node 3: x and y, noisy
           np.array([[1, 0, 0, 0.0]]),                        # node 4: x only
           np.array([[0, 1, 0, 0.0]])]                        # node 5: y only
R_nodes = [np.array([[2.0**2]]), np.array([[2.0**2]]), np.diag([6.0**2, 6.0**2]),
           np.array([[3.0**2]]), np.array([[3.0**2]])]
N = len(H_nodes)
nx = 4

# communication graph: ring 1-2-3-4-5-1 plus the chord 1-3 (0-based indices)
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
A = np.zeros((N, N))
for i, j in edges:
    A[i, j] = A[j, i] = 1
deg = A.sum(axis=1)
Lap = np.diag(deg) - A
lam = np.sort(np.linalg.eigvalsh(Lap))
neigh = [np.flatnonzero(A[i]) for i in range(N)]
# Metropolis weights
W = np.zeros((N, N))
for i in range(N):
    for j in neigh[i]:
        W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
    W[i, i] = 1.0 - W[i].sum()
lamW = np.sort(np.abs(np.linalg.eigvals(W)))[::-1]
print("Laplacian eigenvalues:", np.round(lam, 4))
print("algebraic connectivity lambda_2 = %.4f, lambda_max = %.4f" % (lam[1], lam[-1]))
print("Metropolis weight matrix W =\n", np.round(W, 4))
print("second largest |eigenvalue| of W = %.4f" % lamW[1])

# %% [markdown]
# ## Filters

# %%
def kf_predict(x, P):
    return F @ x, F @ P @ F.T + Q


def kf_update(x, P, y, H, R):
    S = H @ P @ H.T + R
    Kg = P @ H.T @ np.linalg.inv(S)
    x = x + Kg @ (y - H @ x)
    P = (np.eye(nx) - Kg @ H) @ P
    P = 0.5 * (P + P.T)
    return x, P


def ci_pair(x1, P1, x2, P2):
    """Covariance intersection of two estimates, omega minimising the trace."""
    L1, L2 = np.linalg.inv(P1), np.linalg.inv(P2)

    def cost(w):
        return np.trace(np.linalg.inv(w * L1 + (1 - w) * L2))

    res = minimize_scalar(cost, bounds=(0.0, 1.0), method="bounded")
    w = res.x
    Lci = w * L1 + (1 - w) * L2
    Pci = np.linalg.inv(Lci)
    xci = Pci @ (w * L1 @ x1 + (1 - w) * L2 @ x2)
    return xci, 0.5 * (Pci + Pci.T), w


def simulate(rng):
    """One Monte Carlo run of all methods.  Returns truth and per-method estimates."""
    # truth and measurements
    X = np.zeros((K + 1, nx))
    X[0] = x0_true
    for k in range(K):
        X[k + 1] = F @ X[k] + rng.multivariate_normal(np.zeros(nx), Q)
    Y = [[H_nodes[i] @ X[k] + rng.multivariate_normal(np.zeros(H_nodes[i].shape[0]), R_nodes[i])
          for i in range(N)] for k in range(K + 1)]
    xhat0 = x0_true + rng.multivariate_normal(np.zeros(nx), P0)   # common prior

    methods = ["central", "local", "naive_nf", "naive", "coi", "kcf", "ci"]
    est = {m: np.zeros((K + 1, N, nx)) for m in methods}
    cov = {m: np.zeros((K + 1, N, nx, nx)) for m in methods}
    for m in methods:
        est[m][0] = xhat0
        cov[m][0] = P0

    # centralised filter
    Hc = np.vstack(H_nodes)
    Rc = np.zeros((Hc.shape[0], Hc.shape[0]))
    r0 = 0
    for R in R_nodes:
        d = R.shape[0]
        Rc[r0:r0 + d, r0:r0 + d] = R
        r0 += d
    x, P = xhat0.copy(), P0.copy()
    for k in range(1, K + 1):
        x, P = kf_predict(x, P)
        x, P = kf_update(x, P, np.concatenate(Y[k]), Hc, Rc)
        est["central"][k] = x
        cov["central"][k] = P

    # isolated local filters
    xs, Ps = [xhat0.copy() for _ in range(N)], [P0.copy() for _ in range(N)]
    for k in range(1, K + 1):
        for i in range(N):
            xs[i], Ps[i] = kf_predict(xs[i], Ps[i])
            xs[i], Ps[i] = kf_update(xs[i], Ps[i], Y[k][i], H_nodes[i], R_nodes[i])
            est["local"][k, i], cov["local"][k, i] = xs[i], Ps[i]

    # naive fusion WITHOUT feedback: the isolated local posteriors of the closed neighbourhood
    # are fused as if independent, but the local filters keep running on their own
    for k in range(1, K + 1):
        for i in range(N):
            Lam = np.zeros((nx, nx))
            q = np.zeros(nx)
            for j in list(neigh[i]) + [i]:
                Lj = np.linalg.inv(cov["local"][k, j])
                Lam += Lj
                q += Lj @ est["local"][k, j]
            cov["naive_nf"][k, i] = np.linalg.inv(Lam)
            est["naive_nf"][k, i] = cov["naive_nf"][k, i] @ q

    # naive neighbourhood information fusion WITH feedback (treats neighbours' posteriors as independent)
    xs, Ps = [xhat0.copy() for _ in range(N)], [P0.copy() for _ in range(N)]
    for k in range(1, K + 1):
        post = []
        for i in range(N):
            x, P = kf_predict(xs[i], Ps[i])
            post.append(kf_update(x, P, Y[k][i], H_nodes[i], R_nodes[i]))
        for i in range(N):
            Lam = np.zeros((nx, nx))
            q = np.zeros(nx)
            for j in list(neigh[i]) + [i]:
                Lj = np.linalg.inv(post[j][1])
                Lam += Lj
                q += Lj @ post[j][0]
            Ps[i] = np.linalg.inv(Lam)
            xs[i] = Ps[i] @ q
            est["naive"][k, i], cov["naive"][k, i] = xs[i], Ps[i]

    # consensus on information (Battistelli & Chisci 2014), L_CONS Metropolis steps
    xs, Ps = [xhat0.copy() for _ in range(N)], [P0.copy() for _ in range(N)]
    for k in range(1, K + 1):
        Lams, qs = [], []
        for i in range(N):
            x, P = kf_predict(xs[i], Ps[i])
            x, P = kf_update(x, P, Y[k][i], H_nodes[i], R_nodes[i])
            Lam = np.linalg.inv(P)
            Lams.append(Lam)
            qs.append(Lam @ x)
        for _ in range(L_CONS):
            Lams = [sum(W[i, j] * Lams[j] for j in range(N)) for i in range(N)]
            qs = [sum(W[i, j] * qs[j] for j in range(N)) for i in range(N)]
        for i in range(N):
            Ps[i] = np.linalg.inv(Lams[i])
            xs[i] = Ps[i] @ qs[i]
            est["coi"][k, i], cov["coi"][k, i] = xs[i], Ps[i]

    # Kalman-consensus filter (Olfati-Saber 2007)
    xbar, Pb = [xhat0.copy() for _ in range(N)], [P0.copy() for _ in range(N)]
    for k in range(1, K + 1):
        xpred = [F @ xbar[i] for i in range(N)]
        Ppred = [F @ Pb[i] @ F.T + Q for i in range(N)]
        u = [H_nodes[i].T @ np.linalg.solve(R_nodes[i], Y[k][i]) for i in range(N)]
        U = [H_nodes[i].T @ np.linalg.solve(R_nodes[i], H_nodes[i]) for i in range(N)]
        for i in range(N):
            J = list(neigh[i]) + [i]
            S = sum(U[j] for j in J)
            yv = sum(u[j] for j in J)
            M = np.linalg.inv(np.linalg.inv(Ppred[i]) + S)
            gamma = EPS_KCF / (1.0 + np.linalg.norm(M, "fro"))
            cons = sum(xpred[j] - xpred[i] for j in neigh[i])
            xi = xpred[i] + M @ (yv - S @ xpred[i]) + gamma * M @ cons
            est["kcf"][k, i], cov["kcf"][k, i] = xi, M
            xbar[i], Pb[i] = xi, M

    # covariance intersection of the local posteriors with the neighbours' posteriors
    xs, Ps = [xhat0.copy() for _ in range(N)], [P0.copy() for _ in range(N)]
    omegas = []
    for k in range(1, K + 1):
        post = []
        for i in range(N):
            x, P = kf_predict(xs[i], Ps[i])
            post.append(kf_update(x, P, Y[k][i], H_nodes[i], R_nodes[i]))
        for i in range(N):
            x, P = post[i]
            for j in neigh[i]:
                x, P, w = ci_pair(x, P, post[j][0], post[j][1])
                omegas.append(w)
            xs[i], Ps[i] = x, P
            est["ci"][k, i], cov["ci"][k, i] = xs[i], Ps[i]
    return X, Y, est, cov, float(np.mean(omegas))


# %% [markdown]
# ## Monte Carlo statistics: position RMSE and NEES consistency

# %%
rng = np.random.default_rng(SEED)
methods = ["central", "local", "naive_nf", "naive", "coi", "kcf", "ci"]
names = {"central": "centralised KF", "local": "isolated local KF", "naive_nf": "naive fusion, no feedback", "naive": "naive fusion with feedback",
         "coi": "consensus on information (L=%d)" % L_CONS, "kcf": "Kalman-consensus filter",
         "ci": "covariance intersection"}
sq_err = {m: [] for m in methods}
nees = {m: [] for m in methods}
first = None
for run in range(N_MC):
    X, Y, est, cov, wbar = simulate(rng)
    if first is None:
        first = (X, Y, est, cov, wbar)
    for m in methods:
        e = est[m][1:] - X[1:, None, :]                      # (K, N, nx)
        sq_err[m].append(np.sum(e[:, :, :2]**2, axis=2))     # position squared error
        Pinv = np.linalg.inv(cov[m][1:])
        nees[m].append(np.einsum("kni,knij,knj->kn", e, Pinv, e))

lo, hi = chi2.ppf(0.025, nx), chi2.ppf(0.975, nx)
print("\nchi^2_4 two-sided 95%% interval: [%.3f, %.3f]" % (lo, hi))
print("%-34s %10s %10s %10s %10s" % ("method", "pos RMSE", "mean NEES", "median", "in 95% band"))
summary = {}
for m in methods:
    se = np.concatenate([s.ravel() for s in sq_err[m]])
    ne = np.concatenate([n.ravel() for n in nees[m]])
    summary[m] = (np.sqrt(se.mean()), ne.mean(), np.median(ne), np.mean((ne >= lo) & (ne <= hi)))
    print("%-34s %10.3f %10.2f %10.2f %10.1f%%" % (names[m], summary[m][0], summary[m][1], summary[m][2],
                                                    100 * summary[m][3]))
# steady-state (last 5 s) statistics
print("\nsteady state (t >= 5 s):")
for m in methods:
    se = np.concatenate([s[50:].ravel() for s in sq_err[m]])
    ne = np.concatenate([n[50:].ravel() for n in nees[m]])
    print("%-34s %10.3f %10.2f" % (names[m], np.sqrt(se.mean()), ne.mean()))
print("\nmean CI weight omega over all pairwise fusions (run 0): %.3f" % first[4])

# per-node position RMSE for run statistics (all runs)
print("\nper-node position RMSE (all runs):")
for m in methods:
    per_node = np.sqrt(np.mean(np.stack(sq_err[m]), axis=(0, 1)))
    print("%-34s" % names[m], np.round(per_node, 3))

# sensitivity of consensus on information to the number of consensus steps (run statistics)
# (the loop below re-runs the simulation for L = 1 and L = 10 with fresh noise; same seed)
for Ltest in (1, 10):
    L_CONS_saved = L_CONS
    L_CONS = Ltest
    rng2 = np.random.default_rng(SEED)
    se_all, ne_all = [], []
    for run in range(N_MC):
        X2, Y2, est2, cov2, _ = simulate(rng2)
        e = est2["coi"][1:] - X2[1:, None, :]
        se_all.append(np.sum(e[:, :, :2]**2, axis=2))
        ne_all.append(np.einsum("kni,knij,knj->kn", e, np.linalg.inv(cov2["coi"][1:]), e))
    print("consensus on information with L=%2d: pos RMSE %.3f, mean NEES %.2f" %
          (Ltest, np.sqrt(np.concatenate([s.ravel() for s in se_all]).mean()),
           np.concatenate([n.ravel() for n in ne_all]).mean()))
    L_CONS = L_CONS_saved

# %% [markdown]
# ## Time series of run 0 for the figure

# %%
X, Y, est, cov, _ = first
t = np.arange(1, K + 1) * T
rows = {"t": t, "px_true": X[1:, 0], "py_true": X[1:, 1],
        "px_meas3": np.array([Y[k][2][0] for k in range(1, K + 1)]),
        "py_meas3": np.array([Y[k][2][1] for k in range(1, K + 1)]),
        "px_cen": est["central"][1:, 0, 0], "py_cen": est["central"][1:, 0, 1],
        "px_coi": est["coi"][1:, 0, 0], "py_coi": est["coi"][1:, 0, 1],
        "px_naive": est["naive"][1:, 0, 0], "py_naive": est["naive"][1:, 0, 1],
        "px_local": est["local"][1:, 0, 0], "py_local": est["local"][1:, 0, 1]}
for m in methods:
    e = est[m][1:] - X[1:, None, :]
    Pinv = np.linalg.inv(cov[m][1:])
    ne = np.einsum("kni,knij,knj->kn", e, Pinv, e)
    rows["nees1_" + m] = ne[:, 0]                      # node 1
    rows["err_" + m] = np.sqrt(np.mean(np.sum(e[:, :, :2]**2, axis=2), axis=1))   # node-averaged position error
df = pd.DataFrame(rows)
csv_path = os.path.join(OUT, "distributed_tracking.csv")
df.to_csv(csv_path, index=False, float_format="%.6f")
print("\nwrote", csv_path, df.shape)

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(6, 6))
ax[0].plot(df.px_meas3, df.py_meas3, ".", color="0.6", ms=3, label="node 3 detections")
ax[0].plot(df.px_true, df.py_true, "k-", label="truth")
ax[0].plot(df.px_cen, df.py_cen, "--", color="tab:blue", label="centralised")
ax[0].plot(df.px_coi, df.py_coi, "-", color="tab:green", label="node 1, consensus on information")
ax[0].legend(fontsize=7)
ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
for m, c in zip(["central", "naive_nf", "naive", "coi", "ci"], ["tab:blue", "tab:purple", "tab:red", "tab:green", "tab:orange"]):
    ax[1].semilogy(df.t, df["nees1_" + m], color=c, lw=0.8, label=names[m])
ax[1].axhspan(lo, hi, color="0.85")
ax[1].axhline(nx, color="0.4", ls=":")
ax[1].legend(fontsize=7); ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("NEES, node 1")
fig.tight_layout()
pdf_path = os.path.join(OUT, "distributed_tracking.pdf")
fig.savefig(pdf_path)
print("wrote", pdf_path)

# %% [markdown]
# ## Copy the data used by the book

# %%
BOOK = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for sub, src in (("data/ch06", csv_path), ("figures/ch06", pdf_path)):
    dst = os.path.join(BOOK, sub)
    if os.path.isdir(dst):
        shutil.copy(src, dst)
        print("copied", os.path.basename(src), "->", dst)
