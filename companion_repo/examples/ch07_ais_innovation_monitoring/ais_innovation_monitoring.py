"""Example A (Chapter 7): constant-velocity Kalman fusion of coastal-radar plots and AIS position
reports with innovation monitoring (per-step NIS, sliding-window chi-square, Ljung-Box whiteness).

Scenario (30 min, 10-s period): nominal transit, then a slow quadratic spoofing ramp on the AIS
position, then a coloured (AR(1)) AIS disturbance with the nominal marginal variance.
Deterministic (seeded).  Writes output/ais_innovation_monitoring.csv.
"""
# %% [markdown]
# # Example A: innovation monitoring of an AIS/GNSS vessel track
# A constant-velocity Kalman filter fuses coastal-radar plots and AIS position reports (10-s period)
# in a local east-north frame, processing the two sensors sequentially at each step.  We inject (i) a slow spoofing ramp and (ii) a coloured disturbance and
# compare three innovation tests: per-step NIS, sliding-window chi-square, Ljung-Box whiteness.
# %%
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2

rng = np.random.default_rng(2026)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

# %% [markdown]
# ## Scenario parameters
# %%
T = 10.0                 # AIS report period [s] (class A, 0-14 kn)
N = 180                  # number of reports -> 30 min
t = np.arange(N + 1) * T
v_kn = 12.0              # speed over ground [kn]
v = v_kn * 1852.0 / 3600.0
heading_deg = 60.0       # course over ground, degrees clockwise from north
sigma_pos = 10.0         # AIS (GNSS-derived) position noise [m] (1-sigma per axis)
sigma_rad = 30.0         # coastal-radar plot noise [m] (1-sigma per axis)
sigma_a_true = 0.01      # true unmodelled acceleration [m/s^2]
sigma_a_filt = 0.02      # filter design value [m/s^2]
t_spoof = (1200.0, 1800.0)   # slow spoofing ramp window (runs to the end)
a_spoof = 0.001              # cross-track acceleration of the spoofed AIS position [m/s^2]
t_col = (600.0, 1200.0)      # coloured-disturbance window
phi_col = 0.85               # AR(1) coefficient of the coloured measurement noise
W = 12                       # window length of the sliding chi-square test (2 min)
M_lb, L_lb = 30, 5           # Ljung-Box: window of 30 innovations, 5 lags
alpha = 0.01                 # significance level of all alarms

# %% [markdown]
# ## Ground truth and measurements
# %%
F = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1.0]])
def Qmat(sa):
    return sa**2 * np.array([[T**4/4, 0, T**3/2, 0], [0, T**4/4, 0, T**3/2],
                             [T**3/2, 0, T**2, 0], [0, T**3/2, 0, T**2]])
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0.0]])
R_ais = sigma_pos**2 * np.eye(2)
R_rad = sigma_rad**2 * np.eye(2)

h = np.deg2rad(heading_deg)
along = np.array([np.sin(h), np.cos(h)])          # unit vector along the course (E, N)
cross = np.array([np.cos(h), -np.sin(h)])         # unit vector to starboard
x_true = np.zeros((N + 1, 4))
x_true[0] = [0, 0, v * along[0], v * along[1]]
G = np.array([[T**2/2, 0], [0, T**2/2], [T, 0], [0, T]])   # acceleration input matrix (Q = sigma_a^2 G G^T)
for k in range(N):
    x_true[k + 1] = F @ x_true[k] + G @ (sigma_a_true * rng.standard_normal(2))

# measurement noise: white, except AR(1) coloured in the coloured window (same marginal variance)
noise = sigma_pos * rng.standard_normal((N + 1, 2))
e = np.zeros(2)
for k in range(N + 1):
    if t_col[0] <= t[k] <= t_col[1]:
        e = phi_col * e + np.sqrt(1 - phi_col**2) * sigma_pos * rng.standard_normal(2)
        noise[k] = e
    else:
        e = noise[k]
# spoofing ramp: reported position drifts cross-track with a constant small acceleration
spoof = np.zeros((N + 1, 2))
for k in range(N + 1):
    if t[k] >= t_spoof[0]:
        tau = min(t[k], t_spoof[1]) - t_spoof[0]      # offset frozen after the ramp
        spoof[k] = 0.5 * a_spoof * tau**2 * cross
y = x_true[:, :2] + spoof + noise                      # AIS reports (spoofed / coloured)
y_rad = x_true[:, :2] + sigma_rad * rng.standard_normal((N + 1, 2))   # radar plots (trusted)

# %% [markdown]
# ## Kalman filter with innovation statistics
# %%
Qf = Qmat(sigma_a_filt)
xh = np.array([y[0, 0], y[0, 1], v * along[0], v * along[1]])
P = np.diag([sigma_pos**2, sigma_pos**2, 1.0, 1.0])
est = np.zeros((N + 1, 4)); est[0] = xh
nis = np.full(N + 1, np.nan)          # stacked NIS of the two sequential updates (chi^2_4 under H0)
nis_rad = np.full(N + 1, np.nan); nis_ais = np.full(N + 1, np.nan)
nu_w = np.full((N + 1, 4), np.nan)     # whitened innovations S^{-1/2} nu (radar, then AIS)
def update(xp, Pp, yk, R):
    nu = yk - H @ xp
    S = H @ Pp @ H.T + R
    Sinv = np.linalg.inv(S)
    K = Pp @ H.T @ Sinv
    return xp + K @ nu, (np.eye(4) - K @ H) @ Pp, nu @ Sinv @ nu, np.linalg.solve(np.linalg.cholesky(S), nu), S, K
for k in range(1, N + 1):
    xp = F @ xh
    Pp = F @ P @ F.T + Qf
    xh, P, nis_rad[k], nu_w[k, :2], S_r, K_r = update(xp, Pp, y_rad[k], R_rad)   # radar plot first
    xh, P, nis_ais[k], nu_w[k, 2:], S, K = update(xh, P, y[k], R_ais)            # then the AIS report
    nis[k] = nis_rad[k] + nis_ais[k]
    est[k] = xh
n_y = 4

# per-step test
thr_step = chi2.ppf(1 - alpha, n_y)
alarm_step = nis > thr_step
# sliding-window chi-square test (sum of W consecutive NIS values ~ chi^2_{n_y W})
nis_win = np.full(N + 1, np.nan)
for k in range(W, N + 1):
    nis_win[k] = np.nansum(nis[k - W + 1:k + 1])
thr_win = chi2.ppf(1 - alpha, n_y * W)
alarm_win = nis_win > thr_win
# two-sided per-sensor window test on the AIS channel alone (chi^2_{2W}): a coloured disturbance
# that the filter follows makes the AIS innovations *too small*, which the lower bound exposes
nis_win_ais = np.full(N + 1, np.nan)
for k in range(W, N + 1):
    nis_win_ais[k] = np.nansum(nis_ais[k - W + 1:k + 1])
lo_ais, hi_ais = chi2.ppf(alpha / 2, 2 * W), chi2.ppf(1 - alpha / 2, 2 * W)
alarm_low_ais = nis_win_ais < lo_ais
# Ljung-Box whiteness test on the whitened innovation components (sum over the 2 components)
def ljung_box(z, L):
    M = len(z); z = z - z.mean(); c0 = np.dot(z, z) / M
    Q = 0.0
    for l in range(1, L + 1):
        r = np.dot(z[l:], z[:-l]) / M / c0
        Q += r**2 / (M - l)
    return M * (M + 2) * Q
lb = np.full(N + 1, np.nan)
for k in range(M_lb, N + 1):
    seg = nu_w[k - M_lb + 1:k + 1]
    lb[k] = sum(ljung_box(seg[:, i], L_lb) for i in range(n_y))
thr_lb = chi2.ppf(1 - alpha, n_y * L_lb)
alarm_lb = lb > thr_lb

# %% [markdown]
# ## Summary of which test fires when
# %%
def first_alarm(flag, t0, t1):
    idx = np.where(flag & (t >= t0) & (t <= t1))[0]
    return None if len(idx) == 0 else t[idx[0]]
def frac(flag, t0, t1):
    m = (t >= t0) & (t <= t1) & ~np.isnan(np.where(flag, 1.0, 0.0))
    return flag[(t >= t0) & (t <= t1)].mean()
segs = {"nominal": (M_lb * T, t_col[0] - T), "coloured": (t_col[0] + 60, t_col[1] - T), "spoof": (t_spoof[0], t_spoof[1])}
print(f"thresholds (alpha={alpha}): per-step chi2_{n_y} {thr_step:.2f}, window chi2_{n_y*W} {thr_win:.2f}, Ljung-Box chi2_{n_y*L_lb} {thr_lb:.2f}")
print(f"steady-state sqrt(S_11): radar {np.sqrt(S_r[0,0]):.2f} m, AIS {np.sqrt(S[0,0]):.2f} m; position gains radar {K_r[0,0]:.3f}, AIS {K[0,0]:.3f}")
for name, (a, b) in segs.items():
    sel = (t >= a) & (t <= b)
    print(f"[{name:8s} {a:6.0f}-{b:6.0f} s] mean NIS {np.nanmean(nis[sel]):.2f} (radar {np.nanmean(nis_rad[sel]):.2f}, AIS {np.nanmean(nis_ais[sel]):.2f})  "
          f"alarm fraction: step {np.mean(alarm_step[sel]):.2f}  window {np.mean(alarm_win[sel]):.2f}  LB {np.mean(alarm_lb[sel]):.2f}")
print(f"two-sided AIS-only window test, chi2_{2*W} interval [{lo_ais:.2f}, {hi_ais:.2f}]: fraction below lower bound "
      f"nominal {np.mean(alarm_low_ais[(t>=segs['nominal'][0])&(t<=segs['nominal'][1])]):.2f}, "
      f"coloured {np.mean(alarm_low_ais[(t>=segs['coloured'][0])&(t<=segs['coloured'][1])]):.2f}; "
      f"first low alarm in coloured window: {first_alarm(alarm_low_ais, *t_col)}; "
      f"median AIS window statistic there {np.nanmedian(nis_win_ais[(t>=segs['coloured'][0])&(t<=segs['coloured'][1])]):.1f}")
for name, flag in [("per-step", alarm_step), ("window", alarm_win), ("Ljung-Box", alarm_lb)]:
    print(f"first {name:9s} alarm in spoofing window: {first_alarm(flag, *t_spoof)}   in coloured window: {first_alarm(flag, *t_col)}")
k_end = int(t_spoof[1] / T)
print(f"spoofing offset at end of ramp: {np.linalg.norm(spoof[k_end]):.1f} m; "
      f"position error of the filter then: {np.linalg.norm(est[k_end,:2]-x_true[k_end,:2]):.1f} m")
for name, flag in [("per-step", alarm_step), ("window", alarm_win), ("Ljung-Box", alarm_lb)]:
    ta = first_alarm(flag, *t_spoof)
    if ta is not None:
        print(f"   {name:9s}: delay after onset {ta - t_spoof[0]:.0f} s, spoofing offset then {np.linalg.norm(spoof[int(ta/T)]):.1f} m")
print(f"max per-step NIS in spoofing window: {np.nanmax(nis[(t>=t_spoof[0])&(t<=t_spoof[1])]):.2f}; "
      f"max window statistic: {np.nanmax(nis_win[(t>=t_spoof[0])&(t<=t_spoof[1])]):.1f}; "
      f"max LB in coloured window: {np.nanmax(lb[(t>=t_col[0])&(t<t_col[1])]):.1f}")
selc = (t >= t_col[0] + 60) & (t < t_col[1])
print("lag-1 autocorrelation of whitened innovations in coloured window (radar E,N, AIS E,N):",
      np.round([np.corrcoef(nu_w[selc, i][1:], nu_w[selc, i][:-1])[0, 1] for i in range(n_y)], 2))
print(f"RMS position error: nominal {np.sqrt(np.mean(np.sum((est[(t>=120)&(t<600),:2]-x_true[(t>=120)&(t<600),:2])**2,1))):.1f} m, "
      f"coloured {np.sqrt(np.mean(np.sum((est[selc,:2]-x_true[selc,:2])**2,1))):.1f} m")

# %% [markdown]
# ## Export for the book figure (pgfplots reads data/ch07/ais_innovation_monitoring.csv)
# %%
df = pd.DataFrame({"t": t, "e_true": x_true[:, 0], "n_true": x_true[:, 1],
                   "e_meas": y[:, 0], "n_meas": y[:, 1], "e_rad": y_rad[:, 0], "n_rad": y_rad[:, 1],
                   "e_est": est[:, 0], "n_est": est[:, 1],
                   "nis": nis, "nis_rad": nis_rad, "nis_ais": nis_ais, "nis_win": nis_win, "nis_win_ais": nis_win_ais, "lb": lb,
                   "alarm_step": alarm_step.astype(int), "alarm_win": alarm_win.astype(int), "alarm_lb": alarm_lb.astype(int)})
df.to_csv(os.path.join(OUT, "ais_innovation_monitoring.csv"), index=False, float_format="%.3f")
print("rows written:", len(df))
sel = (t >= t_col[0] + 60) & (t < t_col[1])
print("coloured window: NIS quantiles 50/90/99 %:", np.nanpercentile(nis[sel], [50, 90, 99]).round(2),
      " window-statistic quantiles:", np.nanpercentile(nis_win[sel], [50, 90, 99]).round(1))
sel = (t >= t_spoof[0]) & (t <= t_spoof[1])
print("spoof window: NIS quantiles 50/90/99 %:", np.nanpercentile(nis[sel], [50, 90, 99]).round(2),
      " window-statistic quantiles:", np.nanpercentile(nis_win[sel], [50, 90, 99]).round(1))
