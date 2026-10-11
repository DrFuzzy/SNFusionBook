#!/usr/bin/env python3
"""Chapter 3 - end-of-chapter problems 3.1-3.9: numerical verification of the hand computations.

Every number quoted in the problem statements and in the printed solutions of
Problems 3.1-3.9 is recomputed here.  The script is self-contained (numpy/scipy
only), deterministic, and can be converted to a notebook with
``python3 tools/py2nb.py exercises/ch03/solutions/ch03_hand_computations.py``.
"""
# %%
import numpy as np
from scipy.optimize import linprog, minimize_scalar
from scipy.stats import chi2

np.set_printoptions(precision=5, suppress=True)

# %% [markdown]
# ## Problem 3.1 - LS versus WLS with two sensors of different accuracy

# %%
y1, s1 = 41.6, 0.30   # radar range [m], std [m]
y2, s2 = 43.4, 1.50   # camera range [m], std [m]
x_ls = 0.5 * (y1 + y2)
var_ls = 0.25 * (s1**2 + s2**2)
lam = 1 / s1**2 + 1 / s2**2                      # information (scalar)
x_wls = (y1 / s1**2 + y2 / s2**2) / lam
var_wls = 1 / lam
print("P3.1  LS : x = %.4f  var = %.5f  std = %.4f" % (x_ls, var_ls, np.sqrt(var_ls)))
print("P3.1  WLS: x = %.4f  var = %.5f  std = %.4f  (weights %.4f, %.4f)"
      % (x_wls, var_wls, np.sqrt(var_wls), (1 / s1**2) / lam, (1 / s2**2) / lam))
print("P3.1  variance ratio LS/WLS = %.3f ; WLS vs radar alone: %.5f vs %.5f" % (var_ls / var_wls, var_wls, s1**2))

# %% [markdown]
# ## Problem 3.2 - MAP / regularised LS as prior-plus-measurement fusion

# %%
m = np.array([42.0, -5.0])                       # prior mean (range, range rate)
P = np.array([[1.0, 0.3], [0.3, 0.25]])          # prior covariance
G = np.array([[1.0, 0.0]])                       # radar range only
R = np.array([[0.09]])
Lam_post = G.T @ np.linalg.inv(R) @ G + np.linalg.inv(P)
P_post = np.linalg.inv(Lam_post)
x_map = P_post @ (G.T @ np.linalg.inv(R) @ np.array([y1]) + np.linalg.inv(P) @ m)
K = P @ G.T @ np.linalg.inv(G @ P @ G.T + R)
x_seq = m + (K @ (np.array([y1]) - G @ m))
P_seq = (np.eye(2) - K @ G) @ P
print("P3.2  P^-1 =\n", np.linalg.inv(P))
print("P3.2  Lambda_post =\n", Lam_post)
print("P3.2  P_post =\n", P_post)
print("P3.2  x_MAP =", x_map, "  via gain form:", x_seq, " K =", K.ravel())
print("P3.2  P_post via gain form =\n", P_seq)
# diagonal prior: the second component is not updated
P_d = np.diag([1.0, 0.25])
K_d = P_d @ G.T @ np.linalg.inv(G @ P_d @ G.T + R)
print("P3.2  diagonal prior: K =", K_d.ravel(), " x =", m + K_d @ (np.array([y1]) - G @ m))

# %% [markdown]
# ## Problem 3.3 - sequential LS equals batch WLS (numerical check on the data of 3.1)

# %%
x1, P1 = y1, s1**2                                # after the radar alone
K2 = P1 / (P1 + s2**2)
x2 = x1 + K2 * (y2 - x1)
P2 = (1 - K2) * P1
print("P3.3  K2 = %.5f  x2 = %.4f  P2 = %.5f   (batch WLS: %.4f, %.5f)" % (K2, x2, P2, x_wls, var_wls))
Gb = np.array([[1.0], [1.0]]); Rb = np.diag([s1**2, s2**2])
Pb = np.linalg.inv(Gb.T @ np.linalg.inv(Rb) @ Gb)
xb = Pb @ Gb.T @ np.linalg.inv(Rb) @ np.array([y1, y2])
print("P3.3  batch check: x = %.4f  P = %.5f" % (xb[0], Pb[0, 0]))

# %% [markdown]
# ## Problem 3.4 - one Kalman-filter step on a 1-D constant-velocity model

# %%
T, sa = 0.5, 0.8
F = np.array([[1, T], [0, 1]])
Q = sa**2 * np.array([[T**4 / 4, T**3 / 2], [T**3 / 2, T**2]])
xk = np.array([50.0, -4.0]); Pk = np.diag([0.25, 0.16])
x_pred = F @ xk
P_pred = F @ Pk @ F.T + Q
H = np.array([[1.0, 0.0]]); R = np.array([[0.09]]); y = 47.4
nu = y - H @ x_pred
S = H @ P_pred @ H.T + R
Kk = P_pred @ H.T @ np.linalg.inv(S)
x_upd = x_pred + (Kk @ nu)
P_upd = (np.eye(2) - Kk @ H) @ P_pred
P_jos = (np.eye(2) - Kk @ H) @ P_pred @ (np.eye(2) - Kk @ H).T + Kk @ R @ Kk.T
nis = float(nu @ np.linalg.inv(S) @ nu)
print("P3.4  Q =\n", Q)
print("P3.4  x_pred =", x_pred, "\n      P_pred =\n", P_pred)
print("P3.4  nu = %.4f  S = %.4f  K =" % (nu[0], S[0, 0]), Kk.ravel())
print("P3.4  x_upd =", x_upd, "\n      P_upd =\n", P_upd, "\n      Joseph =\n", P_jos)
print("P3.4  NIS = %.4f ; chi2_1 95%% two-sided interval = [%.5f, %.4f]" % (nis, chi2.ppf(0.025, 1), chi2.ppf(0.975, 1)))
print("P3.4  prior std of position: %.4f -> predicted %.4f -> posterior %.4f" % (np.sqrt(Pk[0, 0]), np.sqrt(P_pred[0, 0]), np.sqrt(P_upd[0, 0])))

# %% [markdown]
# ## Problem 3.5 - range-bearing Jacobian and azimuth wrapping

# %%
px, py, vx, vy = 40.0, 3.0, -5.0, 0.0
d = np.hypot(px, py)
Hrb = np.array([[px / d, py / d, 0, 0], [-py / d**2, px / d**2, 0, 0]])
print("P3.5  d = %.4f ; H =\n" % d, Hrb)
# finite-difference check
def g(x):
    return np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])
x0 = np.array([px, py, vx, vy]); eps = 1e-6
Hfd = np.column_stack([(g(x0 + eps * e) - g(x0 - eps * e)) / (2 * eps) for e in np.eye(4)])
print("P3.5  max |H - H_fd| = %.2e" % np.abs(Hrb - Hfd).max())
sig_th = np.deg2rad(0.5)
print("P3.5  lateral std implied by sigma_theta at 60 m: %.4f m ; at 10 m: %.4f m" % (60 * np.tan(sig_th), 10 * np.tan(sig_th)))
th_pred, th_meas = 3.135, -3.140
nu_naive = th_meas - th_pred
nu_wrap = (nu_naive + np.pi) % (2 * np.pi) - np.pi
print("P3.5  naive innovation = %.4f rad ; wrapped = %.5f rad" % (nu_naive, nu_wrap))
print("P3.5  NIS (S ~ sigma_theta^2): naive = %.3e ; wrapped = %.4f" % ((nu_naive / sig_th)**2, (nu_wrap / sig_th)**2))
print("P3.5  chi2_1 97.5%% quantile = %.3f" % chi2.ppf(0.975, 1))

# %% [markdown]
# ## Problem 3.6 - zonotope prediction and single-strip correction

# %%
p0 = np.array([1.0, 0.0]); H0 = np.array([[1.0, 0.5], [0.0, 1.0]])
Fz = np.array([[1.0, 1.0], [0.0, 1.0]]); abar = 0.2; Tz = 1.0
g_w = abar * np.array([Tz**2 / 2, Tz])
p_t = Fz @ p0
H_t = np.column_stack([Fz @ H0, g_w])
print("P3.6  predicted centre", p_t, " generators\n", H_t)
print("P3.6  predicted interval hull: x1 in [%.2f, %.2f], x2 in [%.2f, %.2f]"
      % (p_t[0] - np.abs(H_t[0]).sum(), p_t[0] + np.abs(H_t[0]).sum(), p_t[1] - np.abs(H_t[1]).sum(), p_t[1] + np.abs(H_t[1]).sum()))
c = np.array([1.0, 0.0]); yz = 1.4; phi = 0.3
M = H_t @ H_t.T
lam_opt = M @ c / (c @ M @ c + phi**2)
print("P3.6  H H^T =\n", M, "\n      lambda* =", lam_opt)
def strip_update(lam):
    ph = p_t + lam * (yz - c @ p_t)
    Hh = np.column_stack([(np.eye(2) - np.outer(lam, c)) @ H_t, phi * lam])
    return ph, Hh
for name, lam in [("lambda*", lam_opt), ("lambda=c", c)]:
    ph, Hh = strip_update(lam)
    half = np.abs(Hh).sum(axis=1)
    print("P3.6  [%s] centre" % name, ph, "\n      generators\n", Hh,
          "\n      hull x1 in [%.4f, %.4f], x2 in [%.4f, %.4f], ||H||_F^2 = %.4f"
          % (ph[0] - half[0], ph[0] + half[0], ph[1] - half[1], ph[1] + half[1], (Hh**2).sum()))
# exact intersection hull by linear programming over xi in [-1,1]^3
bounds = [(-1, 1)] * 3
A_ub = np.vstack([c @ H_t, -(c @ H_t)]); b_ub = np.array([yz + phi - c @ p_t, -(yz - phi - c @ p_t)])
for i, lab in enumerate(["x1", "x2"]):
    lo = linprog(H_t[i], A_ub=A_ub, b_ub=b_ub, bounds=bounds).fun + p_t[i]
    hi = -linprog(-H_t[i], A_ub=A_ub, b_ub=b_ub, bounds=bounds).fun + p_t[i]
    print("P3.6  exact intersection: %s in [%.4f, %.4f]" % (lab, lo, hi))

# %% [markdown]
# ## Problem 3.7 - Dempster and Smets combination of two BBAs

# %%
P_, V_, S_ = frozenset("P"), frozenset("V"), frozenset("S")
PV, OM = frozenset("PV"), frozenset("PVS")
m1 = {P_: 0.5, PV: 0.3, OM: 0.2}
m2 = {V_: 0.4, PV: 0.4, S_: 0.1, OM: 0.1}
def conj(m1, m2):
    out = {}
    for a, ma in m1.items():
        for b, mb in m2.items():
            out[a & b] = out.get(a & b, 0.0) + ma * mb
    return out
ms = conj(m1, m2)
Kc = ms.get(frozenset(), 0.0)
md = {a: v / (1 - Kc) for a, v in ms.items() if a}
name = lambda s: "{" + ",".join(sorted(s)) + "}"
print("P3.7  conflict K = %.4f" % Kc)
print("P3.7  Smets (unnormalised):", {name(a): round(v, 4) for a, v in ms.items()})
print("P3.7  Dempster:", {name(a): round(v, 4) for a, v in md.items()})
bel = lambda m, A: sum(v for B, v in m.items() if B <= A)
pl = lambda m, A: sum(v for B, v in m.items() if B & A)
for A in [P_, V_, S_]:
    print("P3.7  %s: Bel = %.4f  Pl = %.4f  BetP = %.4f" % (name(A), bel(md, A), pl(md, A),
          sum(v / len(B) for B, v in md.items() if A <= B)))
# Zadeh-type pathology
z1 = {P_: 0.99, S_: 0.01}; z2 = {V_: 0.99, S_: 0.01}
zs = conj(z1, z2); Kz = zs[frozenset()]
print("P3.7  Zadeh: K = %.4f ; Dempster m(S) = %.4f ; Smets m(S) = %.4f" % (Kz, zs[S_] / (1 - Kz), zs[S_]))

# %% [markdown]
# ## Problem 3.8 - covariance intersection: trace-optimal weight

# %%
P1c = np.diag([0.25, 1.0]); P2c = np.diag([1.0, 0.16])
trace_ci = lambda w: np.trace(np.linalg.inv(w * np.linalg.inv(P1c) + (1 - w) * np.linalg.inv(P2c)))
res = minimize_scalar(trace_ci, bounds=(0, 1), method="bounded", options={"xatol": 1e-12})
w = res.x
P_ci = np.linalg.inv(w * np.linalg.inv(P1c) + (1 - w) * np.linalg.inv(P2c))
# closed form from d/dw = 0 : (6.25 - 5.25 w)/(1 + 3 w) = sqrt(5.25/3)
r = np.sqrt(5.25 / 3); w_cf = (6.25 - r) / (5.25 + 3 * r)
P_naive = np.linalg.inv(np.linalg.inv(P1c) + np.linalg.inv(P2c))
print("P3.8  omega* (numerical) = %.5f ; closed form = %.5f ; sqrt(5.25/3) = %.5f" % (w, w_cf, r))
print("P3.8  P_CI = diag(%.4f, %.4f), trace = %.4f" % (P_ci[0, 0], P_ci[1, 1], np.trace(P_ci)))
print("P3.8  naive independent fusion: diag(%.4f, %.4f), trace = %.4f" % (P_naive[0, 0], P_naive[1, 1], np.trace(P_naive)))
print("P3.8  trace at omega = 0: %.4f, 0.5: %.4f, 1: %.4f" % (trace_ci(0), trace_ci(0.5), trace_ci(1)))
# consistency check against a worst-case correlation
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(2000):
    C = rng.standard_normal((2, 2)); rho = np.tanh(rng.standard_normal())
    # random cross-covariance with the joint matrix PSD
    L1 = np.linalg.cholesky(P1c); L2 = np.linalg.cholesky(P2c)
    U, _s, Vt = np.linalg.svd(C); Rm = U @ Vt
    P12 = L1 @ (rho * Rm) @ L2.T
    A = np.hstack([w * np.linalg.inv(P1c), (1 - w) * np.linalg.inv(P2c)])
    Sig = np.block([[P1c, P12], [P12.T, P2c]])
    true_cov = P_ci @ A @ Sig @ A.T @ P_ci
    worst = max(worst, np.linalg.eigvalsh(true_cov - P_ci).max())
print("P3.8  max eigenvalue of (true cov - P_CI) over random correlations: %.2e (<= 0 means consistent)" % worst)

# %% [markdown]
# ## Problem 3.9 - incomplete AHP: logarithmic least squares on a sparse graph

# %%
edges = {(1, 2): 3.0, (2, 3): 2.0, (1, 3): 4.0, (3, 4): 2.0}
n = 4
L = np.zeros((n, n)); b = np.zeros(n)
for (i, j), a in edges.items():
    i -= 1; j -= 1
    L[i, i] += 1; L[j, j] += 1; L[i, j] -= 1; L[j, i] -= 1
    b[i] += np.log(a); b[j] -= np.log(a)
print("P3.9  Laplacian L =\n", L, "\n      b =", b)
# gauge sum(y) = 0: add the constraint row
Aug = np.vstack([L, np.ones(n)]); rhs = np.append(b, 0.0)
y_lls = np.linalg.lstsq(Aug, rhs, rcond=None)[0]
x_lls = np.exp(y_lls) / np.exp(y_lls).sum()
print("P3.9  y* =", y_lls, "\n      x* =", x_lls)
for (i, j), a in edges.items():
    print("P3.9  edge (%d,%d): A = %.2f, fitted x_i/x_j = %.4f, residual ln A - (y_i - y_j) = %+.4f"
          % (i, j, a, np.exp(y_lls[i - 1] - y_lls[j - 1]), np.log(a) - (y_lls[i - 1] - y_lls[j - 1])))
print("P3.9  predicted missing ratios: A14 = %.4f, A24 = %.4f" % (np.exp(y_lls[0] - y_lls[3]), np.exp(y_lls[1] - y_lls[3])))
print("P3.9  cycle inconsistency: ln(A12 A23 / A13) = ln(%.2f) = %.4f" % (3 * 2 / 4, np.log(1.5)))
# tree check: dropping edge (1,3) gives an exact fit
L2 = L.copy(); b2 = b.copy()
L2[0, 0] -= 1; L2[2, 2] -= 1; L2[0, 2] += 1; L2[2, 0] += 1; b2[0] -= np.log(4); b2[2] += np.log(4)
y_tree = np.linalg.lstsq(np.vstack([L2, np.ones(n)]), np.append(b2, 0.0), rcond=None)[0]
print("P3.9  spanning tree (no edge 1-3): x* =", np.exp(y_tree) / np.exp(y_tree).sum(), " residuals all zero")
