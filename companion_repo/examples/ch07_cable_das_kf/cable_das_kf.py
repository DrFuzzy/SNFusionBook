"""Example B (Chapter 7): damped tensioned string (modal state space) observed by DAS-like strain
channels; Kalman filtering with sliding-window NIS detection and along-cable localisation of an
anchor-strike event from the spatial pattern of the residuals.  Deterministic (seeded).
Writes output/cable_das_timeseries.csv and output/cable_das_profile.csv.
"""
# %% [markdown]
# # Example B: model-based DAS monitoring of a submarine cable segment
# A 1-km cable span is modelled as a damped tensioned string truncated to its first five modes.
# Gauge-length-averaged strain is observed at 16 DAS channels with channel-dependent coupling.
# A Kalman filter tracks the modal state; an anchor strike (impulse followed by a short drag)
# is injected and detected with a sliding-window chi-square test on the innovations, and localised
# from the along-cable profile of normalised residual energy.
# %%
import os
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import chi2

rng = np.random.default_rng(7)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

# %% [markdown]
# ## Cable, DAS and filter parameters
# %%
L = 1000.0            # span length [m]
c = 50.0              # transverse wave speed sqrt(T/rho) [m/s]
rho = 20.0            # mass per unit length [kg/m]
zeta = 0.05           # modal damping ratio
n_modes = 5
dt = 0.5              # sampling period [s]
N = 280               # 140 s
t = np.arange(N + 1) * dt
x_ch = np.linspace(50.0, 950.0, 16)       # DAS channel centres [m]
L_g = 10.0                                # gauge length [m]
kappa = np.ones(len(x_ch)); kappa[[4, 5]] = 0.5     # two poorly coupled (free-span) channels
sigma_eps = 0.05                          # strain noise per channel [microstrain]
q_force = 3e-5                            # white modal forcing intensity [(N/m)^2 s] (environment)
t_event, x_event = 60.0, 640.0            # anchor strike time [s] and position [m]
impulse = 0.7                             # impulse of the strike [N s]
drag_force, t_drag = 0.1, 15.0            # sustained drag force [N] and duration [s]
W = 8                                     # window of the chi-square test (4 s)
alpha = 0.01

# %% [markdown]
# ## Modal state-space model and discretisation (van Loan)
# %%
n = np.arange(1, n_modes + 1)
omega = n * np.pi * c / L
phi = lambda x: np.sqrt(2.0 / L) * np.sin(np.outer(n, np.atleast_1d(x)) * np.pi / L)   # (n_modes, len(x))
A = np.block([[np.zeros((n_modes, n_modes)), np.eye(n_modes)],
              [-np.diag(omega**2), -np.diag(2 * zeta * omega)]])
B = np.vstack([np.zeros((n_modes, n_modes)), np.eye(n_modes) / rho])   # modal force -> modal acceleration
Qc = q_force * np.eye(n_modes)
M = np.block([[-A, B @ Qc @ B.T], [np.zeros_like(A), A.T]]) * dt
E = expm(M)
F = E[2 * n_modes:, 2 * n_modes:].T
Q = F @ E[:2 * n_modes, 2 * n_modes:]
Q = 0.5 * (Q + Q.T)
# measurement: gauge-length-averaged strain  eps_j = kappa_j [w(x_j + Lg/2) - w(x_j - Lg/2)] / Lg  (in microstrain)
Hdisp = 1e6 * (phi(x_ch + L_g / 2) - phi(x_ch - L_g / 2)).T / L_g * kappa[:, None]
H = np.hstack([Hdisp, np.zeros((len(x_ch), n_modes))])
R = sigma_eps**2 * np.eye(len(x_ch))
print("modal frequencies [Hz]:", np.round(omega / (2 * np.pi), 4))

# %% [markdown]
# ## Simulation with an anchor strike (impulse + short drag) at x_event
# %%
Lq = np.linalg.cholesky(Q + 1e-18 * np.eye(2 * n_modes))
x = np.zeros((N + 1, 2 * n_modes))
Gd = B * dt                                   # zero-order-hold approximation for the drag force
for k in range(N):
    x[k + 1] = F @ x[k] + Lq @ rng.standard_normal(2 * n_modes)
    if abs(t[k + 1] - t_event) < 1e-9:
        x[k + 1, n_modes:] += impulse / rho * phi(x_event)[:, 0]          # modal velocity jump
    if t_event < t[k + 1] <= t_event + t_drag:
        x[k + 1] += Gd @ (drag_force * phi(x_event)[:, 0])
y = x @ H.T + sigma_eps * rng.standard_normal((N + 1, len(x_ch)))
print(f"nominal strain RMS (t<{t_event:.0f} s): {np.sqrt(np.mean((x[t < t_event] @ H.T)**2)):.3f} microstrain; "
      f"peak |strain| after strike: {np.max(np.abs(x[t >= t_event] @ H.T)):.3f} microstrain")

# %% [markdown]
# ## Kalman filter, windowed NIS test and residual profile
# %%
xh = np.zeros(2 * n_modes); P = 10 * Q
nis = np.full(N + 1, np.nan); nu_n = np.full((N + 1, len(x_ch)), np.nan)
for k in range(1, N + 1):
    xp = F @ xh; Pp = F @ P @ F.T + Q
    nu = y[k] - H @ xp; S = H @ Pp @ H.T + R; Sinv = np.linalg.inv(S)
    nis[k] = nu @ Sinv @ nu
    nu_n[k] = nu / np.sqrt(np.diag(S))        # channel-wise normalised residuals
    K = Pp @ H.T @ Sinv
    xh = xp + K @ nu; P = (np.eye(2 * n_modes) - K @ H) @ Pp
n_y = len(x_ch)
nis_win = np.full(N + 1, np.nan)
for k in range(W, N + 1):
    nis_win[k] = np.nansum(nis[k - W + 1:k + 1])
thr_win = chi2.ppf(1 - alpha, n_y * W)
thr_step = chi2.ppf(1 - alpha, n_y)
alarm = nis_win > thr_win
idx_on = np.where(alarm & (t >= t_event))[0]
t_alarm = t[idx_on[0]] if len(idx_on) else None
print(f"thresholds: per-step chi2_{n_y} {thr_step:.1f}; window chi2_{n_y*W} {thr_win:.1f} (E = {n_y*W})")
print(f"nominal: mean NIS {np.nanmean(nis[(t>=5)&(t<t_event)]):.2f} (E = {n_y}); window alarm fraction {np.mean(alarm[(t>=5)&(t<t_event)]):.3f}")
print(f"first window alarm after strike at t = {t_alarm} s (delay {t_alarm - t_event:.1f} s); "
      f"max window statistic {np.nanmax(nis_win):.0f}; alarm duration {np.sum(alarm[t>=t_event])*dt:.1f} s")
print(f"per-step NIS at the strike sample: {nis[int(t_event/dt)]:.1f}")
# along-cable profile of residual energy over the alarm window(s)
sel = alarm & (t >= t_event)
E_j = np.sum(nu_n[sel]**2, axis=0) - sel.sum()        # excess over the H0 expectation (1 per sample)
E_j = np.clip(E_j, 0, None)
x_hat = np.sum(x_ch * E_j) / np.sum(E_j)
j_max = np.argmax(E_j)
print(f"residual-energy centroid: {x_hat:.0f} m; peak channel at {x_ch[j_max]:.0f} m; true event at {x_event:.0f} m "
      f"(spatial resolution ~ L/(2 n_modes) = {L/(2*n_modes):.0f} m)")

# %% [markdown]
# ## Export for the book figures
# %%
pd.DataFrame({"t": t, "nis": nis, "nis_win": nis_win, "alarm": alarm.astype(int),
              "strain_ch10": y[:, 9], "strain_ch4": y[:, 3]}).to_csv(
    os.path.join(OUT, "cable_das_timeseries.csv"), index=False, float_format="%.4f")
pd.DataFrame({"x": x_ch, "E": E_j, "kappa": kappa}).to_csv(
    os.path.join(OUT, "cable_das_profile.csv"), index=False, float_format="%.3f")
print("written: cable_das_timeseries.csv (", N + 1, "rows ) and cable_das_profile.csv")
