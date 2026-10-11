"""Problem 5: consensus rounds and topology for the consensus-on-information filter."""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _coi_common import run_coi, metropolis

graphs = {"ring + chord 1-3": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)],
          "path 1-2-3-4-5": [(0, 1), (1, 2), (2, 3), (3, 4)]}
for name, edges in graphs.items():
    A, W = metropolis(edges)
    slem = np.sort(np.abs(np.linalg.eigvals(W)))[-2]
    print("%s: second largest eigenvalue modulus of W = %.4f" % (name, slem))
    for L in (1, 3, 10):
        rmse, nees = run_coi(edges, L)
        print("   L=%2d: position RMSE %.3f m, mean NEES %.2f" % (L, rmse, nees))
