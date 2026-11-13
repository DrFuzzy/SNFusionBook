"""Mean, variance, covariance ellipses and Gaussian densities (data for two figures of Chapter 3).

Chapter 3 (Classical Sensor Fusion Techniques), Sects. "Measurement uncertainty" and
"Gaussian measurement model".

Part 1. Two range sensors read the same distance (10 m) forty times each; the precise one has
        a standard deviation of 0.2 m, the coarse one of 0.6 m.  The figure shows the readings,
        their sample mean, and the interval of one sample standard deviation around it.
Part 2. Three hundred two-dimensional position errors are drawn from a zero-mean Gaussian with
        covariance R = [[1.0, 0.6], [0.6, 0.5]] m^2.  The eigen-decomposition R = U diag(l1, l2) U^T
        gives the principal axes; the ellipses (y - m)^T R^{-1} (y - m) = c^2 are drawn for c = 1, 2, 3.
        In two dimensions the probability inside the ellipse of "radius" c is 1 - exp(-c^2/2),
        i.e. 39.3 %, 86.5 % and 98.9 %; the script checks this on the samples.

Outputs (written to output/; `make export` copies the CSV files to data/ch03 of the book):
    mean_variance_1d.csv         readings of the two sensors (Part 1)
    covariance_2d_samples.csv    two-dimensional samples (Part 2)
    covariance_2d_ellipses.csv   ellipses for c = 1, 2, 3 (Part 2)
    mean_variance_gaussian.pdf   matplotlib preview of both parts
All numbers quoted in the chapter are printed by this script (seed fixed).
"""
# %% [markdown]
# # Mean, variance, covariance ellipses and Gaussian densities
# Self-contained: only numpy / matplotlib are needed.

# %%
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(3)

# %% [markdown]
# ## 1. Repeated readings of the same distance by two sensors

# %%
d_true, sigma_a, sigma_b, n_read = 10.0, 0.2, 0.6, 40
a = d_true + sigma_a * rng.standard_normal(n_read)
b = d_true + sigma_b * rng.standard_normal(n_read)
# vertical jitter, only to separate overlapping dots in the strip plot
ja = 1.0 + 0.12 * rng.uniform(-1.0, 1.0, n_read)
jb = 0.0 + 0.12 * rng.uniform(-1.0, 1.0, n_read)
for name, v, s in (("precise sensor", a, sigma_a), ("coarse sensor", b, sigma_b)):
    print(f"{name:15s}: true sigma {s:.2f} m, sample mean {v.mean():.3f} m, "
          f"sample std {v.std(ddof=1):.3f} m, range [{v.min():.2f}, {v.max():.2f}] m")
np.savetxt(os.path.join(OUT, "mean_variance_1d.csv"), np.column_stack([np.arange(n_read), a, ja, b, jb]),
           delimiter=",", header="k,a,ja,b,jb", comments="", fmt=["%d", "%.4f", "%.4f", "%.4f", "%.4f"])

# %% [markdown]
# ## 2. Correlated two-dimensional errors, eigen-decomposition and ellipses

# %%
R = np.array([[1.0, 0.6], [0.6, 0.5]])
lam, U = np.linalg.eigh(R)            # ascending order
lam, U = lam[::-1], U[:, ::-1]        # l1 >= l2
if U[0, 0] < 0:                       # orient the first principal axis to the right
    U[:, 0] *= -1
if U[1, 1] < 0:
    U[:, 1] *= -1
angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
print(f"eigenvalues l1 = {lam[0]:.4f}, l2 = {lam[1]:.4f} m^2; semi-axes sqrt(l) = "
      f"{np.sqrt(lam[0]):.3f}, {np.sqrt(lam[1]):.3f} m; first axis at {angle:.1f} deg")
print(f"u1 = ({U[0,0]:.4f}, {U[1,0]:.4f}), u2 = ({U[0,1]:.4f}, {U[1,1]:.4f})")
print(f"correlation coefficient = {R[0,1]/np.sqrt(R[0,0]*R[1,1]):.3f}")

n_samp = 300
L = np.linalg.cholesky(R)
Y = (L @ rng.standard_normal((2, n_samp))).T
np.savetxt(os.path.join(OUT, "covariance_2d_samples.csv"), Y, delimiter=",", header="x,y", comments="", fmt="%.4f")
print(f"sample covariance:\n{np.cov(Y.T)}")

th = np.linspace(0.0, 2.0 * np.pi, 121)
circle = np.vstack([np.cos(th), np.sin(th)])
A = U @ np.diag(np.sqrt(lam))          # maps the unit circle onto the c = 1 ellipse
cols, hdr = [], []
for c in (1, 2, 3):
    E = c * (A @ circle)
    cols += [E[0], E[1]]
    hdr += [f"x{c}", f"y{c}"]
np.savetxt(os.path.join(OUT, "covariance_2d_ellipses.csv"), np.column_stack(cols), delimiter=",",
           header=",".join(hdr), comments="", fmt="%.4f")

Rinv = np.linalg.inv(R)
d2 = np.einsum("ij,jk,ik->i", Y, Rinv, Y)   # squared Mahalanobis distance of each sample
for c in (1, 2, 3):
    print(f"c = {c}: theoretical mass 1 - exp(-c^2/2) = {1 - np.exp(-c**2 / 2):.3f}, "
          f"fraction of samples inside = {np.mean(d2 <= c**2):.3f}")
print("1-D comparison: P(|y - m| <= c sigma) = 0.683, 0.954, 0.997 for c = 1, 2, 3")

# %% [markdown]
# ## 3. Preview

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].plot(a, ja, "o", ms=3, label=f"precise, sigma = {sigma_a} m")
ax[0].plot(b, jb, "s", ms=3, label=f"coarse, sigma = {sigma_b} m")
for v, y0 in ((a, 1.0), (b, 0.0)):
    m, s = v.mean(), v.std(ddof=1)
    ax[0].errorbar(m, y0 + 0.3, xerr=s, fmt="k|", capsize=4)
ax[0].axvline(d_true, color="gray", lw=0.5)
ax[0].set_xlabel("reading [m]"); ax[0].set_yticks([]); ax[0].legend(fontsize=7)
ax[1].plot(Y[:, 0], Y[:, 1], ".", ms=2, color="gray")
for c in (1, 2, 3):
    E = c * (A @ circle)
    ax[1].plot(E[0], E[1], lw=1)
for i in range(2):
    ax[1].arrow(0, 0, *(np.sqrt(lam[i]) * U[:, i]), width=0.01, color="k", length_includes_head=True)
ax[1].set_aspect("equal"); ax[1].set_xlabel("error east [m]"); ax[1].set_ylabel("error north [m]")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "mean_variance_gaussian.pdf"))
print("outputs written to", OUT)
