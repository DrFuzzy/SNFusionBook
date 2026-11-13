"""Example B (Chapter 7): pre-curved cable, quadratic strain channels, EKF with second-order prediction.

A tensioned cable whose route is bent in plan is observed by strain channels whose reading is quadratic
in the state; an extended Kalman filter with second-order measurement prediction feeds per-step and
sliding-window NIS tests, and a lateral load is localised from the profile of the residuals.
Deterministic (seeded).
Writes output/cable_das_timeseries.csv and output/cable_das_profile.csv.

The parameters are those of the cable benchmark of Oliva et al., "Innovation-based detection and
localization of persistent disturbances in linear systems with quadratic outputs, motivated by
submarine-cable monitoring" (manuscript under review at the European Journal of Control), except
that the model is discretised exactly in time (zero-order hold) instead of by forward Euler.
"""
# %% [markdown]
# # Example B: model-based monitoring of a pre-curved cable span
# A 1-km span of a cable laid on the seabed is a damped tensioned string whose lateral displacement
# w(x,t) is discretised by finite differences on 40 interior nodes.  The route is bent in plan, so
# the axial strain read by a fibre over a gauge is *quadratic* in the state: a linear term
# proportional to the route slope plus half the square of the gauge slope (moderate-rotation
# kinematics).  An extended Kalman filter (EKF) linearises the reading at its prediction and adds
# the curvature offset beta_j = kappa_j g_j' P g_j / 2 to the predicted measurement, which centres
# the innovation.  A lateral load (an anchor dragging the cable) is detected with the per-step NIS
# and the sliding-window NIS sum and localised from the channel-wise residual energy.
# %%
import os
import numpy as np
import pandas as pd
from scipy.linalg import expm, solve_discrete_are, solve_discrete_lyapunov
from scipy.stats import chi2

rng = np.random.default_rng(7)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

# %% [markdown]
# ## Cable, channels and scenario
# %%
L = 1000.0            # span length [m]
T = 4.0e4             # tension [N]
rho = 15.0            # mass per unit length, structural plus added mass [kg/m]
c_d = 50.0            # lumped viscous damping [N s/m^2]
N = 40                # interior nodes of the finite-difference grid
dt = 0.02             # sampling period [s]
t_end = 30.0          # length of the record [s]
alpha_max = 0.10      # route slope at the ends: plan-view bend w_bar(x) = alpha_max x (1 - x/L)
n_ch = 10             # strain channels, one grid cell (24.4 m) each, every fourth cell
kappa = 0.78          # transduction factor of every channel
sigma_r = 1.0e-6      # strain noise per channel (about one microstrain)
q_c = 0.4             # white force-density intensity per node [(N/m)^2 s] (20 (N/m)^2 per 0.02-s sample)
node_e = 13           # load at node 14 (0-based index 13), x = 341.5 m
amp = 100.0           # lateral load density over one cell [N/m] (resultant amp * dx = 2.44 kN)
t_on, t_off = 5.0, 10.0
W = 50                # window of the NIS-sum test (1 s)
alpha_fa = 0.01       # false-alarm probability per test

dx = L / (N + 1)
xg = (np.arange(N) + 1) * dx
n = 2 * N
steps = int(round(t_end / dt))
t = np.arange(steps) * dt

# %% [markdown]
# ## Plant: finite-difference string, exact (zero-order-hold) discretisation
# State x = [u; du/dt] with u the nodal lateral displacements.  F and Q come from one matrix
# exponential (van Loan); Gam maps a force density held over one sample into the state.
# %%
D2 = (np.diag(-2.0 * np.ones(N)) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)) / dx**2
Ac = np.block([[np.zeros((N, N)), np.eye(N)], [T / rho * D2, -c_d / rho * np.eye(N)]])
Bc = np.vstack([np.zeros((N, N)), np.eye(N) / rho])
E = expm(np.block([[-Ac, Bc @ (q_c * np.eye(N)) @ Bc.T], [np.zeros((n, n)), Ac.T]]) * dt)
F = E[n:, n:].T
Q = F @ E[:n, n:]
Q = 0.5 * (Q + Q.T)
E_in = expm(np.block([[Ac, Bc], [np.zeros((N, n)), np.zeros((N, N))]]) * dt)
Gam = E_in[:n, n:]
omega1 = np.pi / L * np.sqrt(T / rho)
zeta = c_d / (2 * rho * omega1 * np.arange(1, N + 1))
tau_ol = -dt / np.log(np.max(np.abs(np.linalg.eigvals(F))))
print(f"zeta_1 = {zeta[0]:.1f}; overdamped modes (zeta_n > 1): n = 1..{np.sum(zeta > 1)}; "
      f"c_d L^2/(pi^2 T) = {c_d * L**2 / (np.pi**2 * T):.1f} s; slowest time constant of F: {tau_ol:.1f} s")

# %% [markdown]
# ## Quadratic strain channels
# Channel j spans the cell between nodes i_j and i_j + 1: its gauge slope is s_j = g_j' x and the
# route slope over the gauge is alpha_j.  The reading is y_j = kappa (alpha_j s_j + s_j^2 / 2) + r_j,
# with Jacobian row kappa (alpha_j + s_j) g_j' and Hessian kappa g_j g_j'.
# %%
idx = np.arange(n_ch) * (N // n_ch)                       # 0, 4, ..., 36
G = np.zeros((n_ch, n))
G[np.arange(n_ch), idx], G[np.arange(n_ch), idx + 1] = -1.0 / dx, 1.0 / dx
w_bar = alpha_max * xg * (1.0 - xg / L)
alpha = G[:, :N] @ w_bar
x_ch = (xg[idx] + xg[idx + 1]) / 2.0                     # gauge centres [m]
R = sigma_r**2 * np.eye(n_ch)


def h(x):
    s = G @ x
    return kappa * (alpha * s + 0.5 * s**2)


def jac(x):
    return (kappa * (alpha + G @ x))[:, None] * G


def beta(P):
    return 0.5 * kappa * np.einsum("ja,ab,jb->j", G, P, G)


print("gauge centres [m]:", np.round(x_ch, 1))
print("route slopes alpha_j:", np.round(alpha, 4))

# %% [markdown]
# ## Steady filter at rest: how large is the curvature offset?
# beta_j / sqrt(S_jj) is the offset in standard deviations; an uncentred innovation adds
# beta' S^-1 beta to the per-step NIS, so a window needs about M* = 1 / (beta' S^-1 beta) samples to
# accumulate one unit of non-centrality.
# %%
C0 = jac(np.zeros(n))
P_bar = solve_discrete_are(F.T, C0.T, Q, R)
S_bar = C0 @ P_bar @ C0.T + R
K_bar = np.linalg.solve(S_bar, C0 @ P_bar).T
b_bar = beta(P_bar)
ratio = b_bar / np.sqrt(np.diag(S_bar))
M_star = 1.0 / (b_bar @ np.linalg.solve(S_bar, b_bar))
tau_cl = -dt / np.log(np.max(np.abs(np.linalg.eigvals(F @ (np.eye(n) - K_bar @ C0)))))
state_share = np.trace(C0 @ P_bar @ C0.T) / np.trace(S_bar)
print(f"beta_j/sqrt(S_jj): median {np.median(ratio):.1e}, max {ratio.max():.1e}; "
      f"M* = {M_star:.2g} samples = {M_star * dt / 60:.0f} min; state share of S {state_share:.2f}; "
      f"closed-loop time constant {tau_cl:.1f} s")

# %% [markdown]
# ## Simulation: stationary start, lateral load between t_on and t_off
# The plant and the filter start from their stationary distributions (completed burn-in).  The
# filter knows the model and the noise levels but not the load.
# %%
def psd_sqrt(M):
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    # The symmetric principal root is invariant to eigenvector signs and rotations.
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


P_post = (np.eye(n) - K_bar @ C0) @ P_bar
x = psd_sqrt(solve_discrete_lyapunov(F, Q)) @ rng.standard_normal(n)
x_f = x - psd_sqrt(P_post) @ rng.standard_normal(n)
P_f = P_post.copy()
Lq = psd_sqrt(Q)
load = np.zeros(N)
load[node_e] = amp
xs, xps = np.zeros((steps, n)), np.zeros((steps, n))
nis, nu_n = np.zeros(steps), np.zeros((steps, n_ch))
for k in range(steps):
    if k > 0:
        f_k = load if t_on <= t[k - 1] < t_off else np.zeros(N)
        x = F @ x + Lq @ rng.standard_normal(n) + Gam @ f_k
    x_p, P_p = F @ x_f, F @ P_f @ F.T + Q                   # prediction
    y = h(x) + sigma_r * rng.standard_normal(n_ch)
    H = jac(x_p)
    nu = y - h(x_p) - beta(P_p)                             # centred innovation
    S = H @ P_p @ H.T + R
    K = np.linalg.solve(S, H @ P_p).T
    x_f = x_p + K @ nu
    IKH = np.eye(n) - K @ H
    P_f = IKH @ P_p @ IKH.T + K @ R @ K.T                   # Joseph form
    nis[k] = nu @ np.linalg.solve(S, nu)
    nu_n[k] = nu / np.sqrt(np.diag(S))
    xs[k], xps[k] = x, x_p

# %% [markdown]
# ## Per-step and sliding-window NIS tests
# %%
csum = np.concatenate([[0.0], np.cumsum(nis)])
nis_win = np.full(steps, np.nan)
nis_win[W - 1:] = csum[W:] - csum[:-W]
thr_step = chi2.ppf(1 - alpha_fa, n_ch)
thr_win = chi2.ppf(1 - alpha_fa, n_ch * W)
a_step, a_win = nis > thr_step, nis_win > thr_win
pre, ev = t < t_on, (t >= t_on) & (t < t_off)
episode = a_win & (t >= t_on)                             # alarm samples after the onset
k_first, k_last = np.flatnonzero(episode)[[0, -1]]
print(f"thresholds: per-step chi2_{n_ch}(0.99) = {thr_step:.1f}; window chi2_{n_ch * W}(0.99) = {thr_win:.1f}")
print(f"before the load: mean NIS {nis[pre].mean():.2f}; per-step alarms {a_step[pre].sum()} and window alarms "
      f"{a_win[pre].sum()} out of {pre.sum()} samples")
print(f"first alarm after the onset: per-step {t[np.flatnonzero(a_step & (t >= t_on))[0]] - t_on:.2f} s, "
      f"window {t[k_first] - t_on:.2f} s; per-step alarms during the load {a_step[ev].sum()} of {ev.sum()}")
print(f"window statistic: peak {np.nanmax(nis_win):.0f}; alarm from {t[k_first]:.2f} s to {t[k_last]:.2f} s "
      f"({t[k_last] - t[k_first]:.1f} s, i.e. {t[k_last] - t_off:.1f} s after the load is removed), "
      f"{episode.sum()} alarm samples; per-step alarms in the 5 s after the load {a_step[(t >= t_off) & (t < t_off + 5)].sum()}")

# %% [markdown]
# ## How quadratic are the reading and the innovation?
# In the reading the ratio of the quadratic to the linear part is |s_j| / (2 |alpha_j|); in the
# innovation, linearised at the prediction, it is |s_j - s_hat_j| / (2 |alpha_j + s_hat_j|).
# %%
j_e = int(np.argmin(np.abs(x_ch - xg[node_e])))
s_true, s_pred = xs[:, :N] @ G[:, :N].T, xps[:, :N] @ G[:, :N].T
quad_read = np.abs(s_true[ev, j_e]) / (2 * abs(alpha[j_e]))
quad_innov = np.abs(s_true[ev, j_e] - s_pred[ev, j_e]) / (2 * np.abs(alpha[j_e] + s_pred[ev, j_e]))
k_end = np.flatnonzero(ev)[-1]
gain = 1.0 + s_true[k_end] / alpha
print(f"largest gauge slope during the load {np.abs(s_true[ev]).max():.3f}; channel at {x_ch[j_e]:.0f} m: "
      f"quadratic/linear part of the reading up to {100 * quad_read.max():.0f}%, of the innovation up to "
      f"{100 * quad_innov.max():.1f}%")
print("channel gains 1 + s_j/alpha_j at the end of the load:", np.round(gain, 2))

# %% [markdown]
# ## Residual-energy profile over the alarm samples and its centroid
# %%
E_j = np.clip(np.sum(nu_n[episode] ** 2, axis=0) - episode.sum(), 0.0, None)
x_hat = np.sum(x_ch * E_j) / np.sum(E_j)
print("residual energy E_j:", np.round(E_j, 0))
print(f"centroid {x_hat:.0f} m against a true {xg[node_e]:.1f} m: error {abs(x_hat - xg[node_e]):.0f} m "
      f"(grid cell {dx:.1f} m, channel spacing {x_ch[1] - x_ch[0]:.1f} m)")

# %% [markdown]
# ## Export for the book figure
# %%
pd.DataFrame({"t": t, "nis": nis, "nis_win": nis_win, "alarm": a_win.astype(int)}).to_csv(
    os.path.join(OUT, "cable_das_timeseries.csv"), index=False, float_format="%.4f")
pd.DataFrame({"x": x_ch, "E": E_j, "alpha": alpha}).to_csv(
    os.path.join(OUT, "cable_das_profile.csv"), index=False, float_format="%.4f")
print("written: cable_das_timeseries.csv (", steps, "rows ) and cable_das_profile.csv")
