"""Solution script for Problem 5.6 (Chapter 5): sensitivity of covariance intersection to the
camera range accuracy and the radar bearing accuracy (3 x 3 grid).  Self-contained."""
# %%
import numpy as np

R, TH = 60.0, np.deg2rad(10.0)

def polar_cov(r, th, sr, sth):
    J = np.array([[np.cos(th), -r*np.sin(th)], [np.sin(th), r*np.cos(th)]])
    return J @ np.diag([sr**2, sth**2]) @ J.T

def ci_trace(P1, P2, n=2001):
    I1, I2 = np.linalg.inv(P1), np.linalg.inv(P2)
    best = (np.inf, None)
    for w in np.linspace(0, 1, n):
        tr = np.trace(np.linalg.inv(w*I1 + (1-w)*I2))
        if tr < best[0]:
            best = (tr, w)
    return best

# %%
print("cam sr/r [%]  radar sth [deg]   omega*   tr P_CI [m^2]   tr P_naive [m^2]   tr P_cam   tr P_rad")
for frac in [0.05, 0.15, 0.30]:
    for sth_deg in [0.5, 1.5, 3.0]:
        Pc = polar_cov(R, TH, frac*R, np.deg2rad(0.2)); Pr = polar_cov(R, TH, 0.25, np.deg2rad(sth_deg))
        trci, w = ci_trace(Pc, Pr); trn = np.trace(np.linalg.inv(np.linalg.inv(Pc) + np.linalg.inv(Pr)))
        print(f"{frac*100:10.0f}  {sth_deg:14.1f}   {w:6.3f}   {trci:12.3f}   {trn:15.3f}   {np.trace(Pc):8.2f}   {np.trace(Pr):8.2f}")

# %% [markdown]
# Extra case: an imaging radar with 0.1 deg bearing accuracy dominates the camera in both
# directions, so CI puts all the weight on the radar (omega* -> 0) and returns the radar track.

# %%
Pc = polar_cov(R, TH, 0.15*R, np.deg2rad(0.2)); Pr = polar_cov(R, TH, 0.25, np.deg2rad(0.1))
trci, w = ci_trace(Pc, Pr); trn = np.trace(np.linalg.inv(np.linalg.inv(Pc) + np.linalg.inv(Pr)))
print(f"imaging radar 0.1 deg, camera 15 %: omega*={w:.3f}, tr P_CI={trci:.3f}, tr P_rad={np.trace(Pr):.3f}, tr P_naive={trn:.3f}")
