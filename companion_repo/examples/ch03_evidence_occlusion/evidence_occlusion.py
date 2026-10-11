#!/usr/bin/env python3
"""
Evidence-theory example: rain-night occlusion conflict
======================================================

Reproduces the **Example "Rain-night occlusion conflict"** and
**Fig. fig:evidence-theory-autonomous-driving-example** of Chapter 3
(Evidence Theory for Conflicting Sensor Information).

An autonomous vehicle approaches a poorly illuminated pedestrian crossing in heavy
rain.  The camera (glare, partial occlusion) leans toward "free lane"; the
radar--LiDAR track-level module returns a weak but persistent object hypothesis.
On the frame ``Omega = {F, O, U}`` (free lane / occupied by vehicle or obstacle /
unknown vulnerable road user) the two BBAs of eq. (evidence-example-bbas) are
combined with Smets' unnormalised rule, eq. (smets), which exposes the conflict
``K = 0.44`` as the empty-set mass, and with Dempster's rule, eq. (dempster).
Belief, plausibility and the pignistic probabilities of eq. (pignistic-transform)
are then reported.  Uses ``snfusion.evidence.MassFunction``.

Outputs
-------
``output/evidence_occlusion.csv``: one row per focal element with the two input
BBAs, the Smets and the Dempster combinations; the derived quantities are printed.
``output/evidence_occlusion.png``: a bar-chart preview.

Run:  python3 evidence_occlusion.py
"""

# %% [markdown]
# ## Setup

# %%
import sys
from pathlib import Path

try:
    HERE = Path(__file__).resolve().parent
except NameError:                     # running inside a notebook
    HERE = Path.cwd()
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

from snfusion.evidence import MassFunction, format_set   # noqa: E402

# %% [markdown]
# ## Frame of discernment and the two BBAs, eq. (evidence-example-bbas)

# %%
FRAME = ("F", "O", "U")
NAMES = {"F": "free lane", "O": "occupied by vehicle/obstacle", "U": "unknown vulnerable road user"}
OMEGA = frozenset(FRAME)

camera = MassFunction(FRAME, {"F": 0.55, ("O", "U"): 0.20, OMEGA: 0.25})
radar_lidar = MassFunction(FRAME, {"O": 0.45, ("O", "U"): 0.35, OMEGA: 0.20})

print("Frame of discernment")
for state in FRAME:
    print(f"  {state}: {NAMES[state]}")
print()
print("Camera BBA m_C\n" + camera.table() + "\n")
print("Radar-LiDAR BBA m_RL\n" + radar_lidar.table() + "\n")

# %% [markdown]
# ## Smets' conjunctive rule, eq. (smets): the empty-set mass is the conflict K

# %%
smets = camera.combine_conjunctive(radar_lidar)
print("Unnormalised Smets combination\n" + smets.table())
print(f"Conflict K = m_S(empty) = {smets.conflict():.4f}\n")

# %% [markdown]
# ## Dempster's rule, eq. (dempster), and the derived quantities

# %%
dempster, conflict = camera.combine_dempster(radar_lidar)
print("Dempster-normalised combination\n" + dempster.table() + "\n")
print(f"Bel(O) = {dempster.belief('O'):.4f}")
print(f"Pl(O)  = {dempster.plausibility('O'):.4f}\n")

betp = dempster.pignistic()
print("Pignistic probabilities, eq. (pignistic-transform)")
for state in FRAME:
    print(f"  BetP({state}) = {betp[state]:.4f}")

# %% [markdown]
# ## CSV output

# %%
focals = sorted(set(camera) | set(radar_lidar) | set(smets) | set(dempster),
                key=lambda s: (len(s), sorted(s)))
csv_path = OUT / "evidence_occlusion.csv"
with open(csv_path, "w") as f:
    f.write("focal,m_camera,m_radar_lidar,m_smets,m_dempster\n")
    for A in focals:
        f.write(f"{format_set(A, OMEGA)},{camera.get(A, 0.0):.6f},{radar_lidar.get(A, 0.0):.6f},"
                f"{smets.get(A, 0.0):.6f},{dempster.get(A, 0.0):.6f}\n")
print(f"\nwrote {csv_path.relative_to(REPO)}")

# %% [markdown]
# ## Preview figure

# %%
import matplotlib                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

labels = [format_set(A, OMEGA) for A in focals]
x = np.arange(len(focals))
fig, ax = plt.subplots(figsize=(6.4, 3.2))
for i, (name, mf) in enumerate((("camera $m_C$", camera), ("radar-LiDAR $m_{RL}$", radar_lidar),
                                ("Smets", smets), ("Dempster", dempster))):
    ax.bar(x + (i - 1.5) * 0.2, [mf.get(A, 0.0) for A in focals], width=0.2, label=name)
ax.set(xticks=x, xticklabels=labels, ylabel="mass", title="rain-night occlusion conflict: conflict K = %.2f" % conflict)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "evidence_occlusion.png", dpi=120)
print(f"wrote {(OUT / 'evidence_occlusion.png').relative_to(REPO)}")
