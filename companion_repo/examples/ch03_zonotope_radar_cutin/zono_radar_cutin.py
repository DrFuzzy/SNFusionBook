#!/usr/bin/env python3
"""
Zonotopic (set-membership) counterpart of the EKF radar cut-in example
======================================================================

Reproduces **Example ex:zono-radar-cutin** and **Fig. fig:zono-radar-cutin** of
Chapter 3 (Set-Membership and Zonotopic Estimation).

Same target, same constant-velocity model F, same radar and the same detection
sequence as Example ex:ekf-radar-cutin.  What changes is the description of the
errors: covariances are replaced by hard bounds, and the filter propagates a
zonotope ``X_k = p_k (+) H_k B_inf`` (``snfusion.zonotope.Zonotope``) instead of
a mean and a covariance.

Two constructions keep the example short and exact:

1. Bounded process noise.  A constant-velocity model with |a_x|,|a_y| <= a_bar
   contributes exactly two generators per step, ``a_bar [T^2/2, 0, T, 0]^T`` and
   ``a_bar [0, T^2/2, 0, T]^T``.  ``a_bar`` is chosen to cover the lane-change
   acceleration, which the script verifies.
2. Exact measurement strips, eqs. (zono-strip-radial)--(zono-strip-tangential):
   in the frame aligned with the measured bearing the annulus sector of admissible
   positions is contained in a rectangle, with no linearisation error.  The
   range-rate channel gives a third strip with half-width
   ``rdot_bar + th_bar * v_max``.

The strip gain minimising the Frobenius radius of the corrected generator matrix,
eq. (zono-optimal-gain), is the Kalman gain with ``H H^T`` in place of ``P``.

Outputs
-------
``output/zono_radar_cutin.csv`` (``data/ch03/zono_radar_cutin.csv`` in the book
sources): the zonotopic lateral enclosure, the EKF 3-sigma envelope (read from
``../ch03_ekf_radar_cutin/output/ekf_radar_cutin.csv``, which is regenerated if
missing), both relative to the truth, and the EKF lateral error; columns
``t, py_true, zlo, zhi, ekf_lo, ekf_hi, rz_lo, rz_hi, re_lo, re_hi, e_ekf``.
``output/zono_radar_cutin.png`` is a preview.  The numbers quoted in the example
are printed.

Run:  python3 zono_radar_cutin.py
"""

# %% [markdown]
# ## Setup

# %%
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    HERE = Path(__file__).resolve().parent
except NameError:                     # running inside a notebook
    HERE = Path.cwd()
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

from snfusion.scenarios import (                      # noqa: E402
    CUT_DY, CUT_T0, CUT_T1, N, RNG_SEED, SIG_R, SIG_RD, SIG_TH, T,
    F_matrix, nominal_run)
from snfusion.zonotope import Zonotope                # noqa: E402

# %% [markdown]
# ## Hard bounds, set at four standard deviations of the Example ex:ekf-radar-cutin noise

# %%
K_SIGMA = 4.0
R_BAR = K_SIGMA * SIG_R          # 1.20 m
TH_BAR = K_SIGMA * SIG_TH        # 2.0 deg
RD_BAR = K_SIGMA * SIG_RD        # 0.40 m/s
V_MAX = 15.0                     # bound on relative speed [m/s]
A_BAR = 2.5                      # bound on relative acceleration [m/s^2]
M_MAX = 20                       # generator budget (order reduction)


def strips(z):
    """Exact bounded-error strips ``(c, y_bar, phi_bar)`` from one polar detection z = (r, th, rdot)."""
    r, th, rdot = z
    u = np.array([np.cos(th), np.sin(th)])
    v = np.array([-np.sin(th), np.cos(th)])
    lo = (r - R_BAR) * np.cos(TH_BAR)
    hi = r + R_BAR
    return [
        (np.array([u[0], u[1], 0.0, 0.0]), 0.5 * (lo + hi), 0.5 * (hi - lo)),          # radial, eq. (zono-strip-radial)
        (np.array([v[0], v[1], 0.0, 0.0]), 0.0, (r + R_BAR) * np.sin(TH_BAR)),        # tangential, eq. (zono-strip-tangential)
        (np.array([0.0, 0.0, u[0], u[1]]), rdot, RD_BAR + TH_BAR * V_MAX),             # range rate
    ]

# %% [markdown]
# ## Identical measurement sequence to Example ex:ekf-radar-cutin (same seed, same draw order)

# %%
F = F_matrix(T)
ts, Xt, Z, err = nominal_run(RNG_SEED)

# --- sanity: are the assumed hard bounds actually valid on this run? ------
viol = [(np.abs(err[:, 0]) > R_BAR).sum(),
        (np.abs(err[:, 1]) > TH_BAR).sum(),
        (np.abs(err[:, 2]) > RD_BAR).sum()]
tau = np.linspace(0, 1, 2001)
a_true = np.abs(CUT_DY * (6 - 12 * tau) / (CUT_T1 - CUT_T0) ** 2).max()
print(f"bound check: realised sensor errors exceeding the bound: "
      f"range {viol[0]}, azimuth {viol[1]}, range rate {viol[2]} (of {len(err)})")
print(f"bound check: peak true lateral acceleration {a_true:.2f} m/s^2 "
      f"vs assumed bound {A_BAR:.2f} m/s^2 -> {'VALID' if a_true <= A_BAR else 'VIOLATED'}")
print(f"             relative speed max {np.abs(Xt[:, 2:]).max():.2f} m/s vs assumed {V_MAX:.1f} m/s")

# %% [markdown]
# ## Zonotopic estimator: exact prediction, three strip corrections, order reduction

# %%
# initialisation: the first detection's own strips define the box
r0, th0, _ = Z[0]
p0 = np.array([r0 * np.cos(th0), r0 * np.sin(th0), 0.0, 0.0])
X = Zonotope.from_box(p0, [R_BAR + r0 * TH_BAR, R_BAR + r0 * TH_BAR, V_MAX, V_MAX])

Gq = A_BAR * np.array([[T ** 2 / 2, 0.0], [0.0, T ** 2 / 2], [T, 0.0], [0.0, T]])

rows, contained, ngen = [], 0, []
for k, t in enumerate(ts):
    if k > 0:
        X = X.predict(F, Gq)                     # prediction: exact (linear dynamics)
    X = X.intersect_strips(strips(Z[k]))         # correction: three strips, Frobenius-optimal gain
    X = X.reduce(M_MAX)                          # interval-hull order reduction
    ngen.append(X.order)

    p, half = X.center, X.half_widths()          # interval hull half-widths
    contained += int(X.contains(Xt[k]))
    rows.append(dict(t=t,
                     py_true=Xt[k, 1],
                     zc=p[1], zlo=p[1] - half[1], zhi=p[1] + half[1],
                     vy_lo=p[3] - half[3], vy_hi=p[3] + half[3]))

A = {c: np.array([r[c] for r in rows]) for c in rows[0]}
m = A["t"] >= 1.0
w_zono = (A["zhi"] - A["zlo"])

# %% [markdown]
# ## Comparison against the EKF 3-sigma envelope of Example ex:ekf-radar-cutin
# The EKF table is read from the sibling example's `output/` folder (and that
# example is run first if the file is missing).

# %%
ekf_csv = REPO / "examples" / "ch03_ekf_radar_cutin" / "output" / "ekf_radar_cutin.csv"
if not ekf_csv.exists():
    print("EKF output not found -> running examples/ch03_ekf_radar_cutin/ekf_radar_cutin.py")
    subprocess.run([sys.executable, "ekf_radar_cutin.py"], cwd=ekf_csv.parents[1], check=True)
with open(ekf_csv) as fh:
    E = list(csv.DictReader(fh))
ek = {c: np.array([float(r[c]) for r in E]) for c in E[0]}
ekf_lo = ek["py_est"] - 3 * ek["sig_py"]
ekf_hi = ek["py_est"] + 3 * ek["sig_py"]
ekf_in = (ek["py_true"] >= ekf_lo) & (ek["py_true"] <= ekf_hi)
man = (A["t"] >= CUT_T0) & (A["t"] <= CUT_T1)

print()
print(f"zonotope: truth inside the enclosure on {contained}/{len(ts)} steps "
      f"({100*contained/len(ts):.1f} %)")
print(f"EKF 3-sigma: truth inside on {ekf_in.sum()}/{len(ts)} steps "
      f"({100*ekf_in.mean():.1f} %); during the cut-in {100*ekf_in[man].mean():.1f} %")
print(f"largest EKF 3-sigma excursion: "
      f"{np.maximum(ek['py_true'] - ekf_hi, ekf_lo - ek['py_true']).max():.2f} m")
print(f"mean lateral width  zonotope {w_zono[m].mean():.2f} m   "
      f"EKF 6-sigma {(6*ek['sig_py'])[m].mean():.2f} m   "
      f"ratio {w_zono[m].mean()/(6*ek['sig_py'])[m].mean():.1f}x")
print(f"lateral width at t=1 s: {w_zono[A['t'] >= 1.0][0]:.1f} m,  at t=10 s: {w_zono[-1]:.1f} m")
print(f"generators after reduction: {min(ngen)}--{max(ngen)} (budget {M_MAX})")

# %% [markdown]
# ## CSV for the book figure

# %%
csv_path = OUT / "zono_radar_cutin.csv"
with open(csv_path, "w") as f:
    cols = ["t", "py_true", "zlo", "zhi", "ekf_lo", "ekf_hi",
            "rz_lo", "rz_hi", "re_lo", "re_hi", "e_ekf"]
    f.write(",".join(cols) + "\n")
    for k in range(len(ts)):
        yt = A["py_true"][k]
        f.write(f"{A['t'][k]:.6f},{yt:.6f},{A['zlo'][k]:.6f},"
                f"{A['zhi'][k]:.6f},{ekf_lo[k]:.6f},{ekf_hi[k]:.6f},"
                f"{A['zlo'][k]-yt:.6f},{A['zhi'][k]-yt:.6f},"
                f"{ekf_lo[k]-yt:.6f},{ekf_hi[k]-yt:.6f},"
                f"{ek['py_est'][k]-yt:.6f}\n")
print(f"\nwrote {csv_path.relative_to(REPO)}")

# %% [markdown]
# ## Preview figure

# %%
import matplotlib                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True)
ax1.axvspan(CUT_T0, CUT_T1, color="#f4b183", alpha=0.4, lw=0)
ax1.fill_between(A["t"], A["zlo"], A["zhi"], color="#4e9a4e", alpha=0.18, label="zonotope enclosure")
ax1.fill_between(A["t"], ekf_lo, ekf_hi, color="#1f569e", alpha=0.45, label=r"EKF $3\sigma$")
ax1.plot(A["t"], A["py_true"], "k-", lw=0.8, label="truth")
ax1.set(ylabel="lateral offset $p_y$ [m]", title="(a) bounded-error vs probabilistic envelopes")
ax1.legend(frameon=False, fontsize=8)
ax2.axvspan(CUT_T0, CUT_T1, color="#f4b183", alpha=0.4, lw=0)
ax2.fill_between(A["t"], ekf_lo - A["py_true"], ekf_hi - A["py_true"], color="#1f569e", alpha=0.35)
ax2.plot(A["t"], ek["py_est"] - A["py_true"], color="#1f569e", lw=0.6)
ax2.axhline(0, color="k", lw=0.6)
ax2.set(xlabel="time $t$ [s]", ylabel="error [m]", title=r"(b) EKF lateral error and $3\sigma$ band")
fig.tight_layout()
fig.savefig(OUT / "zono_radar_cutin.png", dpi=120)
print(f"wrote {(OUT / 'zono_radar_cutin.png').relative_to(REPO)}")
