#!/usr/bin/env python3
"""
Incomplete AHP for a rain-night sensor policy
=============================================

Reproduces the **Example "Incomplete AHP for a rain-night sensor policy"**,
eq. (incomplete-ahp-example-weights) and
**Fig. fig:incomplete-ahp-autonomous-driving-example** of Chapter 3
(Fusion based on Multi-Criteria Decision-Making).

A fallback supervisor must rank four information streams for the same
adverse-weather scene: camera semantics C, LiDAR geometry L, radar range-rate R
and a map/IMU lane prior M.  Expert comparisons are available only for a sparse
connected graph (``A_LC = 2, A_RC = 5, A_MC = 3, A_RM = 1.5``), not for the full
4 x 4 matrix.  The logarithmic least-squares problem of eq. (lls-ahp) is solved
on the observed edges only, with the gauge ``sum_i y_i = 0``
(``snfusion.ahp.incomplete_lls``), giving the priority vector
``x* = [0.091, 0.183, 0.441, 0.284]`` and the inferred ratios
``A_RL ~ 2.41, A_ML ~ 1.55``.

Outputs
-------
``output/incomplete_ahp.csv``: code, label, priority of each stream.
``output/incomplete_ahp_ratios.csv``: observed and fitted ratio for each edge of
the comparison graph, plus the two inferred (missing) ratios.
``output/incomplete_ahp.png``: a preview of the priorities.

Run:  python3 incomplete_ahp.py
"""

# %% [markdown]
# ## Setup

# %%
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

from snfusion.ahp import consistency_ratio, incomplete_lls   # noqa: E402

# %% [markdown]
# ## Alternatives and the sparse comparison graph

# %%
ALTERNATIVES = [
    ("C", "camera semantics"),
    ("L", "LiDAR geometry"),
    ("R", "radar range-rate"),
    ("M", "map/IMU lane prior"),
]

# Each tuple means "left is ratio times as preferred as right".
COMPARISONS = [
    ("L", "C", 2.0),
    ("R", "C", 5.0),
    ("M", "C", 3.0),
    ("R", "M", 1.5),
]
MISSING = [("R", "L"), ("M", "L")]

# %% [markdown]
# ## Logarithmic least-squares completion, eq. (lls-ahp), gauge sum_i y_i = 0

# %%
res = incomplete_lls([code for code, _ in ALTERNATIVES], COMPARISONS, gauge="sum")

print("Incomplete AHP priorities for a rain-night autonomous-driving scene")
for (code, label), priority in zip(ALTERNATIVES, res.priorities):
    print(f"  {code:1s} ({label:20s}) = {priority:.4f}")
order = np.argsort(-res.priorities)
print("  ranking: " + " > ".join(ALTERNATIVES[i][0] for i in order))
print()

print("Observed comparisons and fitted ratios")
for left, right, ratio in COMPARISONS:
    print(f"  A_{{{left},{right}}}: observed {ratio:.3f}, fitted {res.fitted_ratio(left, right):.3f}")
print()

print("Ratios inferred from the incomplete graph")
for left, right in MISSING:
    print(f"  A_{{{left},{right}}} ~= {res.fitted_ratio(left, right):.3f}")
print()

A_hat = res.completed_matrix()
print("Completed (consistent) comparison matrix x_i / x_j, order C, L, R, M:")
print(np.array2string(A_hat, precision=3, suppress_small=True))
print(f"consistency ratio of the completed matrix: {consistency_ratio(A_hat):.2e} (consistent by construction)")
print(f"LLS residuals on the observed edges (ln A_ij - (y_i - y_j)): "
      f"{np.array2string(res.residuals, precision=3)}")

# %% [markdown]
# ## CSV outputs

# %%
csv_path = OUT / "incomplete_ahp.csv"
with open(csv_path, "w") as f:
    f.write("code,label,priority\n")
    for (code, label), priority in zip(ALTERNATIVES, res.priorities):
        f.write(f"{code},{label},{priority:.6f}\n")
print(f"wrote {csv_path.relative_to(REPO)}")

ratios_path = OUT / "incomplete_ahp_ratios.csv"
with open(ratios_path, "w") as f:
    f.write("left,right,observed,fitted\n")
    for left, right, ratio in COMPARISONS:
        f.write(f"{left},{right},{ratio:.6f},{res.fitted_ratio(left, right):.6f}\n")
    for left, right in MISSING:
        f.write(f"{left},{right},,{res.fitted_ratio(left, right):.6f}\n")
print(f"wrote {ratios_path.relative_to(REPO)}")

# %% [markdown]
# ## Preview figure

# %%
import matplotlib                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.bar([f"{c}\n{l}" for c, l in ALTERNATIVES], res.priorities, color="#1f569e")
for i, p in enumerate(res.priorities):
    ax.text(i, p + 0.01, f"{p:.3f}", ha="center", fontsize=8)
ax.set(ylabel="priority $x_i^*$", title="incomplete AHP: rain-night sensor-stream reliability")
fig.tight_layout()
fig.savefig(OUT / "incomplete_ahp.png", dpi=120)
print(f"wrote {(OUT / 'incomplete_ahp.png').relative_to(REPO)}")
