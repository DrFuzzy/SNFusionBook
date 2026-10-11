"""Problem 7.7: strike position, number of modes and an unknown coupling factor in the modal
DAS Kalman filter of Example B (self-contained)."""
# %%
import numpy as np
from scipy.linalg import expm
from scipy.stats import chi2

def run(x_event=640.0, n_modes=5, kappa_filter_wrong=False, seed=7, N=280, dt=0.5, W=8, alpha=0.01):
    rng = np.random.default_rng(seed)
    L, c, rho, zeta, Lg, sig = 1000.0, 50.0, 20.0, 0.05, 10.0, 0.05
    x_ch = np.linspace(50, 950, 16); kappa = np.ones(16); kappa[[4, 5]] = 0.5
    t = np.arange(N + 1) * dt
    n = np.arange(1, n_modes + 1); om = n * np.pi * c / L
    phi = lambda x: np.sqrt(2 / L) * np.sin(np.outer(n, np.atleast_1d(x)) * np.pi / L)
    A = np.block([[np.zeros((n_modes, n_modes)), np.eye(n_modes)], [-np.diag(om**2), -np.diag(2 * zeta * om)]])
    B = np.vstack([np.zeros((n_modes, n_modes)), np.eye(n_modes) / rho])
    E = expm(np.block([[-A, B @ (3e-5 * np.eye(n_modes)) @ B.T], [np.zeros_like(A), A.T]]) * dt)
    F = E[2 * n_modes:, 2 * n_modes:].T; Q = F @ E[:2 * n_modes, 2 * n_modes:]; Q = 0.5 * (Q + Q.T)
    Hd = lambda kap: np.hstack([1e6 * (phi(x_ch + Lg / 2) - phi(x_ch - Lg / 2)).T / Lg * kap[:, None], np.zeros((16, n_modes))])
    H_true = Hd(kappa); H_filt = Hd(np.ones(16)) if kappa_filter_wrong else H_true
    R = sig**2 * np.eye(16)
    Lq = np.linalg.cholesky(Q + 1e-18 * np.eye(2 * n_modes)); x = np.zeros((N + 1, 2 * n_modes))
    for k in range(N):
        x[k + 1] = F @ x[k] + Lq @ rng.standard_normal(2 * n_modes)
        if abs(t[k + 1] - 60) < 1e-9: x[k + 1, n_modes:] += 0.7 / rho * phi(x_event)[:, 0]
        if 60 < t[k + 1] <= 75: x[k + 1] += (B * dt) @ (0.1 * phi(x_event)[:, 0])
    y = x @ H_true.T + sig * rng.standard_normal((N + 1, 16))
    xh = np.zeros(2 * n_modes); P = 10 * Q; nis = np.full(N + 1, np.nan); nun = np.full((N + 1, 16), np.nan)
    for k in range(1, N + 1):
        xp = F @ xh; Pp = F @ P @ F.T + Q; nu = y[k] - H_filt @ xp; S = H_filt @ Pp @ H_filt.T + R; Si = np.linalg.inv(S)
        nis[k] = nu @ Si @ nu; nun[k] = nu / np.sqrt(np.diag(S)); K = Pp @ H_filt.T @ Si
        xh = xp + K @ nu; P = (np.eye(2 * n_modes) - K @ H_filt) @ Pp
    win = np.array([np.nansum(nis[k - W + 1:k + 1]) if k >= W else np.nan for k in range(N + 1)])
    alarm = win > chi2.ppf(1 - alpha, 16 * W)
    i = np.where(alarm & (t >= 60))[0]; delay = t[i[0]] - 60 if len(i) else None
    sel = alarm & (t >= 60); Ej = np.clip(np.sum(nun[sel]**2, 0) - sel.sum(), 0, None)
    xc = np.sum(x_ch * Ej) / np.sum(Ej) if Ej.sum() > 0 else np.nan
    nominal_excess = np.clip(np.sum(nun[(t >= 5) & (t < 60)]**2, 0) - ((t >= 5) & (t < 60)).sum(), 0, None)
    return delay, xc, np.mean(alarm[(t >= 5) & (t < 60)]), nominal_excess

# %%
print("x_e   n_m   delay [s]   centroid [m]   error [m]")
for xe in [200, 400, 640, 800]:
    for nm in [3, 5, 8]:
        d, xc, _, _ = run(x_event=xe, n_modes=nm)
        print(f"{xe:4.0f}  {nm:3d}   {d}   {xc:8.0f}   {abs(xc - xe):6.0f}")
d, xc, fa, ex = run(kappa_filter_wrong=True)
print(f"wrong coupling in the filter: nominal window-alarm fraction {fa:.2f}; nominal residual excess per channel:")
print(np.round(ex, 0))
