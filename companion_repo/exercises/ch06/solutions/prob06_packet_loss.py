"""Problem 6: packet loss in the consensus-on-information filter (weights renormalised over received messages)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _coi_common import run_coi

edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
for p in (0.0, 0.3, 0.6):
    rmse, nees = run_coi(edges, L=3, p_loss=p)
    print("loss probability p=%.1f: position RMSE %.3f m, mean NEES %.2f (consistent if <= 4)" % (p, rmse, nees))
