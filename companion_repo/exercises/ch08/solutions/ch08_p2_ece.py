"""Chapter 8, Problem 2: expected calibration error (ECE), Brier score and temperature scaling.

Twelve detection confidences p_i with ground-truth correctness y_i are given.  We compute
the ECE with M = 5 equal-width bins, the Brier score, accuracy and mean confidence, and we
find the temperature T that minimises the negative log-likelihood of the temperature-scaled
confidences sigma(z_i / T), z_i = logit(p_i).  A reliability diagram is drawn.

Run:  python3 ch08_p2_ece.py          (prints the numbers, saves ch08_p2_reliability.png)
"""
# %%
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = np.array([0.95, 0.92, 0.88, 0.85, 0.78, 0.72, 0.65, 0.58, 0.52, 0.45, 0.35, 0.22])
y = np.array([1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0])
N = len(p)


# %% [markdown]
# ## (a) Expected calibration error with M equal-width bins
# ECE = sum_b (n_b / N) |acc_b - conf_b|.  Bin edges are [0, 0.2), [0.2, 0.4), ..., [0.8, 1.0].

# %%
def ece(p, y, M=5):
    edges = np.linspace(0.0, 1.0, M + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, M - 1)
    total, rows = 0.0, []
    for b in range(M):
        m = idx == b
        n_b = int(m.sum())
        if n_b == 0:
            rows.append((b, 0, np.nan, np.nan, 0.0))
            continue
        acc_b, conf_b = y[m].mean(), p[m].mean()
        total += n_b / len(p) * abs(acc_b - conf_b)
        rows.append((b, n_b, conf_b, acc_b, abs(acc_b - conf_b)))
    return total, rows


ECE, rows = ece(p, y, M=5)
print("bin  n   conf    acc     |gap|")
for b, n_b, conf_b, acc_b, gap in rows:
    print(f"{b:3d} {n_b:3d}  {conf_b:6.3f}  {acc_b:6.3f}  {gap:6.3f}")
print(f"\n(a) ECE (M=5) = {ECE:.4f}")

# %% [markdown]
# ## (b) Brier score, accuracy and mean confidence

# %%
brier = np.mean((p - y) ** 2)
acc, conf = y.mean(), p.mean()
print(f"(b) Brier = {brier:.4f},  accuracy = {acc:.3f},  mean confidence = {conf:.3f}")

# %% [markdown]
# ## (c) Temperature scaling
# Confidences are replaced by sigma(z/T) with z = logit(p).  T > 1 flattens the confidences
# towards 0.5 (reduces over-confidence); T < 1 sharpens them.  The detector above is
# over-confident on average (mean confidence 0.66 vs accuracy 0.50), so we expect T* > 1.

# %%
z = np.log(p / (1 - p))


def nll(T):
    q = 1.0 / (1.0 + np.exp(-z / T))
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))


res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
T_star = res.x
q_star = 1.0 / (1.0 + np.exp(-z / T_star))
ECE_T, _ = ece(q_star, y, M=5)
print(f"(c) NLL(T=1) = {nll(1.0):.4f};  T* = {T_star:.3f};  NLL(T*) = {nll(T_star):.4f}")
print(f"    ECE after temperature scaling = {ECE_T:.4f}  (mean confidence {q_star.mean():.3f})")
print("    T* > 1: the scores are flattened towards 0.5, which corrects over-confidence.")

# %% [markdown]
# ## (d) Reliability diagram (the text comments on the small-sample issue)

# %%
fig, ax = plt.subplots(figsize=(4.2, 4.0))
edges = np.linspace(0, 1, 6)
centres = 0.5 * (edges[:-1] + edges[1:])
accs = [r[3] if r[1] > 0 else np.nan for r in rows]
ax.bar(centres, accs, width=0.2, edgecolor="k", alpha=0.6, label="accuracy per bin (raw)")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
ax.set_xlabel("confidence")
ax.set_ylabel("empirical accuracy")
ax.set_title(f"Reliability diagram, ECE = {ECE:.3f}")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("ch08_p2_reliability.png", dpi=150)
print("saved ch08_p2_reliability.png")
