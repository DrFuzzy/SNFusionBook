"""Chapter 8, Problem 3: effect of a timestamp offset on the innovation (NIS) test.

A radar measures the range to a lead vehicle (sigma = 0.3 m).  The radar clock is offset by
delta seconds, so a measurement taken at t - delta is interpreted at t.  For a target closing at
relative speed v this produces an innovation bias b = v * delta, and the expected NIS becomes
1 + b^2 / sigma^2.  The script reproduces the analytic numbers of the solution, estimates by
Monte Carlo the rejection rate of the single-sample 95 % gate and the detection probability of
a windowed test (N = 20), and plots the minimum detectable offset as a function of speed.

Run:  python3 ch08_p3_timestamp_offset.py   (prints the numbers, saves ch08_p3_nis_vs_offset.png)
"""
# %%
import numpy as np
from scipy.stats import chi2, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
sigma = 0.3          # radar range standard deviation [m]
N_win = 20           # window length of the averaged NIS test
gate_1 = chi2.ppf(0.95, df=1)                # single-sample 95 % gate (3.84)
gate_w = chi2.ppf(0.95, df=N_win) / N_win    # threshold on the windowed mean NIS (1.571)
print(f"single-sample gate = {gate_1:.3f};  windowed mean-NIS threshold (N={N_win}) = {gate_w:.3f}")


# %% [markdown]
# ## (a)-(b) Innovation bias and expected NIS
# The measurement is r(t_k - delta) + e = r(t_k) + v*delta + e for a target closing at speed v,
# so the innovation is nu = b + e with b = v*delta and E[nu^2 / sigma^2] = 1 + b^2 / sigma^2.

# %%
def expected_nis(v, delta):
    b = v * delta
    return b, 1.0 + (b / sigma) ** 2


for v, delta in [(15.0, 0.020), (15.0, 0.050)]:
    b, e_nis = expected_nis(v, delta)
    print(f"v = {v:4.1f} m/s, delta = {1e3*delta:4.0f} ms:  bias b = {b:.3f} m,  E[NIS] = {e_nis:.2f}")

# %% [markdown]
# ## (c) Rejection rate of a correct measurement by the single-sample gate
# nu / sigma ~ N(b/sigma, 1), so P(reject) = P(|Z + b/sigma| > sqrt(gate)).  Monte Carlo check.

# %%
def rejection_rate_analytic(v, delta):
    lam = v * delta / sigma
    thr = np.sqrt(gate_1)
    return norm.sf(thr - lam) + norm.cdf(-thr - lam)


def simulate_nis(v, delta, n_samples):
    b = v * delta
    e = rng.normal(0.0, sigma, size=n_samples)
    return ((b + e) / sigma) ** 2


for v, delta in [(15.0, 0.0), (15.0, 0.020), (15.0, 0.050)]:
    d2 = simulate_nis(v, delta, 200_000)
    print(f"delta = {1e3*delta:4.0f} ms: mean NIS = {d2.mean():.2f},  "
          f"P(reject | gate 3.84) = {np.mean(d2 > gate_1):.3f}  (analytic {rejection_rate_analytic(v, delta):.3f})")

# %% [markdown]
# ## (d) Minimum detectable offset with the windowed test
# The test fires on average when 1 + b^2/sigma^2 > gate_w, i.e. |b| > sigma*sqrt(gate_w - 1).
# Hence delta_min = sigma*sqrt(gate_w - 1) / v.

# %%
b_min = sigma * np.sqrt(gate_w - 1.0)
for v in (15.0, 2.0):
    print(f"v = {v:4.1f} m/s: minimum detectable bias {b_min:.3f} m  ->  delta_min = {1e3*b_min/v:6.1f} ms")

# Monte Carlo detection probability of the windowed test as a function of delta
deltas = np.linspace(0.0, 0.12, 25)
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
for v, style in ((15.0, "-"), (2.0, "--")):
    p_det = []
    for delta in deltas:
        d2 = simulate_nis(v, delta, 20_000 * N_win).reshape(-1, N_win).mean(axis=1)
        p_det.append(np.mean(d2 > gate_w))
    axes[0].plot(1e3 * deltas, [expected_nis(v, d)[1] for d in deltas], style, label=f"v = {v:g} m/s")
    axes[1].plot(1e3 * deltas, p_det, style, label=f"v = {v:g} m/s")
axes[0].axhline(gate_w, color="k", lw=0.8, ls=":", label="windowed threshold")
axes[0].axhline(gate_1, color="grey", lw=0.8, ls=":", label="single-sample gate")
axes[0].set_xlabel("timestamp offset $\\delta$ [ms]")
axes[0].set_ylabel("expected NIS")
axes[0].set_ylim(0, 10)
axes[0].legend(fontsize=7)
axes[1].axhline(0.05, color="k", lw=0.8, ls=":")
axes[1].set_xlabel("timestamp offset $\\delta$ [ms]")
axes[1].set_ylabel(f"P(detection), window N = {N_win}")
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig("ch08_p3_nis_vs_offset.png", dpi=150)
print("saved ch08_p3_nis_vs_offset.png")

# %% [markdown]
# ## (e) Interpretation
# At motorway closing speeds an offset of ~15 ms is already detectable and produces a bias
# comparable to the sensor noise; in slow urban traffic a 100 ms offset is invisible to the
# test and almost harmless (0.2 m).  The offset is therefore both most harmful and most
# detectable at speed.  Because d h / d delta = v, delta is observable whenever v != 0 and can
# be appended to the state vector (augmented-state estimation) instead of being merely flagged.
