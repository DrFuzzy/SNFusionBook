"""Object-level camera--radar track fusion: naive independent fusion versus covariance intersection.

Chapter 5 ("Autonomous Driving as a Representative Application"), worked example.
A camera track has an accurate bearing but a poor (monocular) range; a radar track
has an accurate range but a poor bearing.  Both are expressed in the ego Cartesian
frame.  We fuse them (i) assuming independent errors (information-additive rule of
Chapter 3) and (ii) with covariance intersection (CI), and we show what happens when
the two tracks share a common error (the ego-heading error used to express both
tracks in the same frame): the naive rule becomes inconsistent, CI does not.

Outputs (written to ./output; copied by hand to data/ch05 and figures/ch05):
    camera_radar_ci_ellipses.csv  -- 95 % error ellipses (73 rows), read by pgfplots
    camera_radar_ci_sweep.csv     -- NEES / sigma versus shared heading error (41 rows)
    camera_radar_ci.pdf           -- matplotlib version of the book figure
Deterministic (np.random.default_rng(11) for the Monte-Carlo check).
"""
# %% [markdown]
# # Camera--radar track fusion with covariance intersection (Chapter 5)
#
# Local tracks $\hat{\mathbf x}_i = \mathbf x + \mathbf e_i$, $i\in\{\text{cam},\text{rad}\}$,
# with $\mathrm{cov}(\mathbf e_i)=\mathbf P_i$ obtained by linearising the polar-to-Cartesian map.
# Naive fusion: $\mathbf P_f^{-1}=\mathbf P_1^{-1}+\mathbf P_2^{-1}$.
# CI: $\mathbf P_{CI}^{-1}=\omega\mathbf P_1^{-1}+(1-\omega)\mathbf P_2^{-1}$, $\omega^\star=\arg\min\,\mathrm{tr}\,\mathbf P_{CI}$.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(11)

# %% [markdown]
# ## Geometry and sensor accuracies (indicative values)

# %%
R_TRUE, TH_TRUE = 60.0, np.deg2rad(10.0)            # target at 60 m, 10 deg to the left
x_true = np.array([R_TRUE * np.cos(TH_TRUE), R_TRUE * np.sin(TH_TRUE)])
SIG_R_CAM, SIG_TH_CAM = 0.15 * R_TRUE, np.deg2rad(0.2)   # monocular camera: 15 % range, 0.2 deg bearing
SIG_R_RAD, SIG_TH_RAD = 0.25, np.deg2rad(1.5)            # 77 GHz radar: 0.25 m range, 1.5 deg bearing

def polar_cov_to_cartesian(r, th, sig_r, sig_th):
    J = np.array([[np.cos(th), -r * np.sin(th)], [np.sin(th), r * np.cos(th)]])
    return J @ np.diag([sig_r**2, sig_th**2]) @ J.T

P_cam = polar_cov_to_cartesian(R_TRUE, TH_TRUE, SIG_R_CAM, SIG_TH_CAM)
P_rad = polar_cov_to_cartesian(R_TRUE, TH_TRUE, SIG_R_RAD, SIG_TH_RAD)

# %% [markdown]
# ## Fusion rules

# %%
def naive_fusion(x1, P1, x2, P2):
    I1, I2 = np.linalg.inv(P1), np.linalg.inv(P2)
    Pf = np.linalg.inv(I1 + I2)
    return Pf @ (I1 @ x1 + I2 @ x2), Pf

def ci_fusion(x1, P1, x2, P2, n_grid=2001):
    I1, I2 = np.linalg.inv(P1), np.linalg.inv(P2)
    best = None
    for w in np.linspace(0.0, 1.0, n_grid):
        Pci = np.linalg.inv(w * I1 + (1 - w) * I2)
        J = np.trace(Pci)
        if best is None or J < best[0]:
            best = (J, w, Pci)
    J, w, Pci = best
    xci = Pci @ (w * I1 @ x1 + (1 - w) * I2 @ x2)
    return xci, Pci, w

def sig_axes(P):
    """standard deviations along x (longitudinal) and y (lateral), and ellipse semi-axes"""
    ev = np.linalg.eigvalsh(P)
    return np.sqrt(P[0, 0]), np.sqrt(P[1, 1]), np.sqrt(ev[1]), np.sqrt(ev[0])

# one realisation (for the figure)
x_cam = x_true + np.linalg.cholesky(P_cam) @ rng.standard_normal(2)
x_rad = x_true + np.linalg.cholesky(P_rad) @ rng.standard_normal(2)
x_nf, P_nf = naive_fusion(x_cam, P_cam, x_rad, P_rad)
x_ci, P_ci, w_star = ci_fusion(x_cam, P_cam, x_rad, P_rad)

print("--- independent-error case: reported standard deviations [m] (sx, sy, major, minor) ---")
for name, P in [("camera", P_cam), ("radar", P_rad), ("naive fusion", P_nf), ("CI fusion", P_ci)]:
    sx, sy, a, b = sig_axes(P)
    print(f"{name:14s}: sx={sx:6.2f}  sy={sy:6.2f}  major={a:6.2f}  minor={b:6.2f}  tr={np.trace(P):8.3f}")
print(f"CI weight omega* = {w_star:.3f} (weight on the camera track)")

# %% [markdown]
# ## When is CI needed?  A shared ego-heading error correlates the two tracks
#
# Both tracks are rotated into the ego frame with the same heading estimate.  A heading
# error $\delta\psi$ moves both estimates by $\delta\psi\,[-p_y,\;p_x]^T$, which is a
# *common* error with cross-covariance $\mathbf P_{12}=\sigma_\psi^2\,\mathbf u\mathbf u^T$.

# %%
u = np.array([-x_true[1], x_true[0]])            # d(position)/d(heading) at the target

def actual_cov(Pf_gain1, Pf_gain2, P1, P2, P12):
    """true covariance of  e = G1 e1 + G2 e2  with cov(e1,e2) = P12"""
    return Pf_gain1 @ P1 @ Pf_gain1.T + Pf_gain2 @ P2 @ Pf_gain2.T \
        + Pf_gain1 @ P12 @ Pf_gain2.T + Pf_gain2 @ P12.T @ Pf_gain1.T

rows = []
for sig_psi_deg in np.linspace(0.0, 2.0, 41):
    sp = np.deg2rad(sig_psi_deg)
    P12 = sp**2 * np.outer(u, u)
    P1 = P_cam + P12; P2 = P_rad + P12          # each local covariance includes the shared term
    I1, I2 = np.linalg.inv(P1), np.linalg.inv(P2)
    # naive
    P_nf_rep = np.linalg.inv(I1 + I2)
    G1, G2 = P_nf_rep @ I1, P_nf_rep @ I2
    P_nf_act = actual_cov(G1, G2, P1, P2, P12)
    nees_nf = np.trace(np.linalg.solve(P_nf_rep, P_nf_act))
    # CI
    _, P_ci_rep, w = ci_fusion(x_cam, P1, x_rad, P2)
    G1, G2 = w * P_ci_rep @ I1, (1 - w) * P_ci_rep @ I2
    P_ci_act = actual_cov(G1, G2, P1, P2, P12)
    nees_ci = np.trace(np.linalg.solve(P_ci_rep, P_ci_act))
    rows.append(dict(sig_psi_deg=sig_psi_deg, omega=w,
                     nees_naive=nees_nf, nees_ci=nees_ci,
                     sy_naive_rep=np.sqrt(P_nf_rep[1, 1]), sy_naive_act=np.sqrt(P_nf_act[1, 1]),
                     sy_ci_rep=np.sqrt(P_ci_rep[1, 1]), sy_ci_act=np.sqrt(P_ci_act[1, 1]),
                     sx_naive_rep=np.sqrt(P_nf_rep[0, 0]), sx_naive_act=np.sqrt(P_nf_act[0, 0]),
                     sx_ci_rep=np.sqrt(P_ci_rep[0, 0]), sx_ci_act=np.sqrt(P_ci_act[0, 0])))
sweep = pd.DataFrame(rows)
sweep.to_csv(os.path.join(OUT, "camera_radar_ci_sweep.csv"), index=False, float_format="%.4f")

print("--- shared heading error sweep (analytic NEES = tr(P_reported^-1 P_actual); consistent iff <= 2) ---")
for s in [0.0, 0.25, 0.5, 1.0, 2.0]:
    r = sweep.iloc[(sweep.sig_psi_deg - s).abs().argmin()]
    print(f"sigma_psi={s:4.2f} deg: naive NEES={r.nees_naive:5.2f} (sy reported {r.sy_naive_rep:.2f} m, actual {r.sy_naive_act:.2f} m) | "
          f"CI NEES={r.nees_ci:5.2f}, omega={r.omega:.3f} (sy reported {r.sy_ci_rep:.2f} m, actual {r.sy_ci_act:.2f} m)")

# Monte-Carlo confirmation at sigma_psi = 0.5 deg
sp = np.deg2rad(0.5); P12 = sp**2 * np.outer(u, u); P1 = P_cam + P12; P2 = P_rad + P12
M = 20000
dpsi = sp * rng.standard_normal(M)
e1 = (np.linalg.cholesky(P_cam) @ rng.standard_normal((2, M))).T + dpsi[:, None] * u
e2 = (np.linalg.cholesky(P_rad) @ rng.standard_normal((2, M))).T + dpsi[:, None] * u
I1, I2 = np.linalg.inv(P1), np.linalg.inv(P2)
P_nf_rep = np.linalg.inv(I1 + I2)
_, P_ci_rep, w = ci_fusion(x_cam, P1, x_rad, P2)
e_nf = (P_nf_rep @ (I1 @ e1.T + I2 @ e2.T)).T
e_ci = (P_ci_rep @ (w * I1 @ e1.T + (1 - w) * I2 @ e2.T)).T
nees_nf_mc = np.mean(np.einsum("ij,jk,ik->i", e_nf, np.linalg.inv(P_nf_rep), e_nf))
nees_ci_mc = np.mean(np.einsum("ij,jk,ik->i", e_ci, np.linalg.inv(P_ci_rep), e_ci))
print(f"Monte Carlo ({M} trials, sigma_psi=0.5 deg): naive NEES {nees_nf_mc:.2f}, CI NEES {nees_ci_mc:.2f}; "
      f"RMS lateral error naive {np.sqrt(np.mean(e_nf[:,1]**2)):.2f} m (reported sy {np.sqrt(P_nf_rep[1,1]):.2f} m), "
      f"CI {np.sqrt(np.mean(e_ci[:,1]**2)):.2f} m (reported sy {np.sqrt(P_ci_rep[1,1]):.2f} m)")
print(f"Shared error 0.5 deg at 60 m moves the target laterally by {R_TRUE*np.tan(sp):.3f} m")

# %% [markdown]
# ## Ellipse CSV for the book figure

# %%
def ellipse(P, centre, n=72, conf=5.991):  # 95 % for chi2_2
    ang = np.linspace(0, 2 * np.pi, n + 1)
    L = np.linalg.cholesky(P)
    pts = centre[:, None] + np.sqrt(conf) * L @ np.vstack([np.cos(ang), np.sin(ang)])
    return pts
E = {}
E["cam"] = ellipse(P_cam, x_cam); E["rad"] = ellipse(P_rad, x_rad)
E["naive"] = ellipse(P_nf, x_nf); E["ci"] = ellipse(P_ci, x_ci)
E["naive_act"] = ellipse(P_nf_act if False else actual_cov(P_nf_rep @ I1, P_nf_rep @ I2, P1, P2, P12), x_nf)  # actual naive-error ellipse, sigma_psi=0.5 deg
df = pd.DataFrame({f"{k}_{c}": E[k][i] for k in E for i, c in enumerate("xy")})
df["true_x"] = x_true[0]; df["true_y"] = x_true[1]
df.to_csv(os.path.join(OUT, "camera_radar_ci_ellipses.csv"), index=False, float_format="%.4f")
print("ellipse rows:", len(df), " sweep rows:", len(sweep))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for k, col, lab in [("cam", "0.5", "camera"), ("rad", "#f4b183", "radar"), ("naive", "#c00000", "naive"), ("ci", "#1f569e", "CI")]:
    ax[0].plot(E[k][0], E[k][1], color=col, label=lab)
ax[0].plot(*x_true, "k+", ms=10); ax[0].set_aspect("equal"); ax[0].legend(fontsize=8)
ax[0].set_xlabel("longitudinal [m]"); ax[0].set_ylabel("lateral [m]")
ax[1].plot(sweep.sig_psi_deg, sweep.nees_naive, color="#c00000", label="naive"); ax[1].plot(sweep.sig_psi_deg, sweep.nees_ci, color="#1f569e", label="CI")
ax[1].axhline(2, color="k", lw=0.8); ax[1].set_xlabel("shared heading error [deg]"); ax[1].set_ylabel("NEES"); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "camera_radar_ci.pdf"))
print("figure written")
