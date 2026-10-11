"""Resilient consensus with a misbehaving agent: plain averaging versus W-MSR.

Companion script for the worked example on resilience of Chapter 6 (Sensor Fusion
in Multi-Agent Systems).  Seven agents hold scalar range estimates to a common
obstacle and run the consensus iteration of the chapter on a circulant graph in
which every agent talks to its two nearest neighbours on each side.  Agent 7 is
faulty (or malicious): it ignores the protocol and keeps broadcasting a biased
value.  Plain consensus is dragged towards the bias; the trimmed W-MSR rule of
LeBlanc et al. (2013), which discards the F largest and F smallest received
values before averaging, stays inside the range of the correct initial values.
The script also verifies by brute force that the graph is 3-robust, which is the
sufficient condition for W-MSR with F = 1.

Outputs (in output/ and copied into the book's data/ch06 and figures/ch06):
  resilient_consensus.csv   iterates of all agents under the two rules
  resilient_consensus.pdf   matplotlib preview
Deterministic (no random numbers are used).
"""
# %% [markdown]
# # Resilient consensus: plain averaging versus W-MSR with one misbehaving agent

# %%
import os
import shutil
import itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

# %% [markdown]
# ## Graph, initial values and the misbehaving agent

# %%
N = 7
# circulant graph C_7(1,2) (agent i talks to i+-1 and i+-2 mod 7) plus the two
# chords 1-4 and 1-5: the smallest extension of C_7(1,2) that is 3-robust
A = np.zeros((N, N), dtype=int)
for i in range(N):
    for d in (1, 2):
        A[i, (i + d) % N] = A[(i + d) % N, i] = 1
for i, j in ((0, 3), (0, 4)):
    A[i, j] = A[j, i] = 1
neigh = [np.flatnonzero(A[i]) for i in range(N)]
deg = A.sum(axis=1)
Lap = np.diag(deg) - A
lam = np.sort(np.linalg.eigvalsh(Lap))
print("edges:", [(i + 1, j + 1) for i in range(N) for j in range(i + 1, N) if A[i, j]])
print("degrees:", deg, " Laplacian eigenvalues:", np.round(lam, 4))

x0 = np.array([10.0, 12.0, 11.0, 15.0, 13.0, 12.6, 12.4])   # initial range estimates [m]
BAD = 6                                                    # agent 7 (0-based index 6)
BIAS = 30.0                                                # value it keeps broadcasting [m]
F_TRIM = 1                                                 # W-MSR parameter
EPS = 0.15                                                 # step size, < 1/max degree = 1/6
K = 40                                                     # iterations
normal = [i for i in range(N) if i != BAD]
print("average of all seven initial values (honest case): %.4f" % x0.mean())
print("average of the six correct initial values: %.4f" % x0[normal].mean())


# %% [markdown]
# ## Brute-force check of r-robustness (LeBlanc et al. 2013)
# A graph is r-robust if for every pair of non-empty disjoint subsets S1, S2 at
# least one of them contains a node with at least r neighbours outside its own subset.

# %%
def is_r_robust(A, r):
    n = A.shape[0]
    nb = [set(np.flatnonzero(A[i])) for i in range(n)]
    for code in itertools.product(range(3), repeat=n):
        S1 = {v for v in range(n) if code[v] == 1}
        S2 = {v for v in range(n) if code[v] == 2}
        if not S1 or not S2:
            continue
        ok = False
        for S in (S1, S2):
            if any(len(nb[v] - S) >= r for v in S):
                ok = True
                break
        if not ok:
            return False
    return True


rmax = max(r for r in range(1, N) if is_r_robust(A, r))
print("the graph is r-robust for r <= %d (W-MSR with F=1 needs r >= 2F+1 = 3)" % rmax)

# path graph 1-2-3-4 of the earlier example, for comparison
Ap = np.zeros((4, 4), dtype=int)
for i in range(3):
    Ap[i, i + 1] = Ap[i + 1, i] = 1
print("path graph on 4 nodes: r-robust for r <= %d" % max(r for r in range(1, 4) if is_r_robust(Ap, r)))

# %% [markdown]
# ## Plain consensus and W-MSR

# %%
def step_plain(x):
    xn = x.copy()
    for i in normal:
        xn[i] = x[i] + EPS * np.sum(x[neigh[i]] - x[i])
    xn[BAD] = BIAS
    return xn


def step_wmsr(x):
    xn = x.copy()
    for i in normal:
        vals = x[neigh[i]]
        larger = np.sort(vals[vals > x[i]])[::-1]     # descending
        smaller = np.sort(vals[vals < x[i]])          # ascending
        removed = set()
        keep = list(vals)
        for v in larger[:F_TRIM]:
            keep.remove(v)
        for v in smaller[:F_TRIM]:
            keep.remove(v)
        keep = np.array(keep)
        xn[i] = x[i] + EPS * np.sum(keep - x[i])
    xn[BAD] = BIAS
    return xn


xp = np.zeros((K + 1, N))
xw = np.zeros((K + 1, N))
xp[0] = x0
xw[0] = x0
xp[0, BAD] = BIAS
xw[0, BAD] = BIAS
for k in range(K):
    xp[k + 1] = step_plain(xp[k])
    xw[k + 1] = step_wmsr(xw[k])

for k in (1, 5, 10, 20, 40):
    print("k=%2d  plain: %s   W-MSR: %s" % (k, np.round(xp[k, normal], 2), np.round(xw[k, normal], 2)))
print("plain consensus, mean of correct agents at k=%d: %.3f (spread %.3f)" %
      (K, xp[K, normal].mean(), xp[K, normal].max() - xp[K, normal].min()))
print("W-MSR, mean of correct agents at k=%d: %.4f (spread %.2e), inside [%.1f, %.1f]" %
      (K, xw[K, normal].mean(), xw[K, normal].max() - xw[K, normal].min(), x0[normal].min(), x0[normal].max()))
print("W-MSR limit versus average of the six correct values: %.4f vs %.4f" % (xw[K, normal].mean(), x0[normal].mean()))

# %% [markdown]
# ## Save the iterates

# %%
rows = {"k": np.arange(K + 1)}
for i in range(N):
    rows["plain%d" % (i + 1)] = xp[:, i]
for i in range(N):
    rows["wmsr%d" % (i + 1)] = xw[:, i]
df = pd.DataFrame(rows)
csv_path = os.path.join(OUT, "resilient_consensus.csv")
df.to_csv(csv_path, index=False, float_format="%.6f")
print("wrote", csv_path, df.shape)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
for a, X, ttl in ((ax[0], xp, "plain consensus"), (ax[1], xw, "W-MSR, F=1")):
    for i in normal:
        a.plot(X[:, i], color="tab:blue", lw=0.8)
    a.plot(X[:, BAD], color="tab:orange", lw=1.2, label="agent 7 (faulty)")
    a.axhline(x0[normal].mean(), color="k", ls=":", lw=0.8)
    a.set_title(ttl); a.set_xlabel("iteration k")
ax[0].set_ylabel("range estimate [m]"); ax[1].legend(fontsize=7)
fig.tight_layout()
pdf_path = os.path.join(OUT, "resilient_consensus.pdf")
fig.savefig(pdf_path)
print("wrote", pdf_path)

BOOK = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for sub, src in (("data/ch06", csv_path), ("figures/ch06", pdf_path)):
    dst = os.path.join(BOOK, sub)
    if os.path.isdir(dst):
        shutil.copy(src, dst)
        print("copied", os.path.basename(src), "->", dst)
