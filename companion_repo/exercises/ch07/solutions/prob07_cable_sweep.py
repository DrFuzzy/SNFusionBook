"""Problem 7.7: load position and mis-specified coupling factors in the quadratic-strain EKF of
Example B (self-contained; same model, channels and filter as examples/ch07_cable_das_kf)."""
# %%
import numpy as np
from scipy.linalg import expm, solve_discrete_are, solve_discrete_lyapunov
from scipy.stats import chi2

L, T, rho, c_d, N, dt = 1000.0, 4.0e4, 15.0, 50.0, 40, 0.02
alpha_max, n_ch, kappa0, sigma_r, q_c, W = 0.10, 10, 0.78, 1.0e-6, 0.4, 50
dx = L / (N + 1); xg = (np.arange(N) + 1) * dx; n = 2 * N
D2 = (np.diag(-2.0 * np.ones(N)) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)) / dx**2
Ac = np.block([[np.zeros((N, N)), np.eye(N)], [T / rho * D2, -c_d / rho * np.eye(N)]])
Bc = np.vstack([np.zeros((N, N)), np.eye(N) / rho])
E = expm(np.block([[-Ac, Bc @ (q_c * np.eye(N)) @ Bc.T], [np.zeros((n, n)), Ac.T]]) * dt)
F = E[n:, n:].T; Q = F @ E[:n, n:]; Q = 0.5 * (Q + Q.T)
Gam = expm(np.block([[Ac, Bc], [np.zeros((N, n)), np.zeros((N, N))]]) * dt)[:n, n:]
idx = np.arange(n_ch) * (N // n_ch)
G = np.zeros((n_ch, n)); G[np.arange(n_ch), idx], G[np.arange(n_ch), idx + 1] = -1.0 / dx, 1.0 / dx
alpha = G[:, :N] @ (alpha_max * xg * (1.0 - xg / L))
x_ch = (xg[idx] + xg[idx + 1]) / 2.0
R = sigma_r**2 * np.eye(n_ch)
P_ol = solve_discrete_lyapunov(F, Q)


def sq(M):
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    # The symmetric principal root is invariant to eigenvector signs and rotations.
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


def run(load=None, kappa_true=None, kappa_filt=None, t_end=30.0, t_on=5.0, t_off=10.0, seed=7):
    """One record; returns window-alarm delay [s], centroid [m], residual-energy profile, alarm fraction
    before t_on (whole record without a load) and per-channel mean squared normalised innovation."""
    kf = kappa0 * np.ones(n_ch) if kappa_filt is None else kappa_filt
    kt = kf if kappa_true is None else kappa_true
    h = lambda x, k: k * (alpha * (G @ x) + 0.5 * (G @ x) ** 2)
    jac = lambda x: (kf * (alpha + G @ x))[:, None] * G
    C0 = jac(np.zeros(n)); P = solve_discrete_are(F.T, C0.T, Q, R)
    K0 = np.linalg.solve(C0 @ P @ C0.T + R, C0 @ P).T
    rng = np.random.default_rng(seed); steps = int(round(t_end / dt)); t = np.arange(steps) * dt
    x = sq(P_ol) @ rng.standard_normal(n); P_f = (np.eye(n) - K0 @ C0) @ P
    x_f = x - sq(P_f) @ rng.standard_normal(n); Lq = sq(Q)
    load = np.zeros(N) if load is None else load
    nis, nu_n = np.zeros(steps), np.zeros((steps, n_ch))
    for k in range(steps):
        if k > 0:
            x = F @ x + Lq @ rng.standard_normal(n) + Gam @ (load if t_on <= t[k - 1] < t_off else 0 * load)
        x_p, P_p = F @ x_f, F @ P_f @ F.T + Q
        H = jac(x_p)
        nu = h(x, kt) + sigma_r * rng.standard_normal(n_ch) - h(x_p, kf) - 0.5 * kf * np.einsum("ja,ab,jb->j", G, P_p, G)
        S = H @ P_p @ H.T + R; K = np.linalg.solve(S, H @ P_p).T
        x_f = x_p + K @ nu; IKH = np.eye(n) - K @ H; P_f = IKH @ P_p @ IKH.T + K @ R @ K.T
        nis[k] = nu @ np.linalg.solve(S, nu); nu_n[k] = nu / np.sqrt(np.diag(S))
    c = np.concatenate([[0.0], np.cumsum(nis)]); win = np.full(steps, np.nan); win[W - 1:] = c[W:] - c[:-W]
    alarm = win > chi2.ppf(0.99, n_ch * W)
    ep = alarm & (t >= t_on)                                  # alarm samples after the onset
    delay, x_hat, Ej = np.nan, np.nan, np.zeros(n_ch)
    if ep.any() and load.any():
        Ej = np.clip(np.sum(nu_n[ep] ** 2, axis=0) - ep.sum(), 0.0, None)
        delay, x_hat = t[np.flatnonzero(ep)[0]] - t_on, np.sum(x_ch * Ej) / np.sum(Ej)
    quiet = t < t_on if load.any() else t >= 1.0
    return delay, x_hat, Ej, np.mean(alarm[quiet & (t >= (W - 1) * dt)]), np.mean(nu_n[quiet] ** 2, axis=0)


# %% [markdown]
# ## (a) Load position: 100 N/m over one cell for 5 s
# Nodes 8, 16, 21 and 33 (x = 195, 390, 512, 805 m) and the centre of the gauge at 329 m, where the
# load is split between the two nodes of that cell.
# %%
print("channel centres [m]:", np.round(x_ch, 0))
print("x_e [m]   delay [s]   centroid [m]   error [m]   residual energy E_j / 1000")
cases = [(xg[i], np.eye(N)[i]) for i in (7, 15, 20, 32)]
cases.append((0.5 * (xg[12] + xg[13]), 0.5 * (np.eye(N)[12] + np.eye(N)[13])))
for x_e, pattern in cases:
    d, xh, Ej, _, _ = run(load=100.0 * pattern)
    print(f"{x_e:7.1f}   {d:9.2f}   {xh:12.0f}   {abs(xh - x_e):9.0f}  ", np.round(Ej / 1000, 1))

# %% [markdown]
# ## (b) Mis-specified coupling of the channels at 232 and 329 m, no load (60-s records)
# %%
k_half = kappa0 * np.ones(n_ch); k_half[[2, 3]] = 0.5 * kappa0
for label, kt, kf in [("matched", None, None),
                      ("true coupling half the assumed", k_half, None),
                      ("true coupling twice the assumed", kappa0 * np.ones(n_ch), k_half)]:
    _, _, _, fa, ms = run(kappa_true=kt, kappa_filt=kf, t_end=60.0)
    print(f"{label:32s} window alarms {100 * fa:5.1f}% of the samples; mean squared normalised innovation:",
          np.round(ms, 2))
