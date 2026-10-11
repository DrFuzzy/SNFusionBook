"""Pose-graph optimisation by hand: four 1-D poses, three odometry factors and one loop closure.

Chapter 4 (Learning-Based and Hybrid Sensor Fusion), worked example of the section
"Multi-Sensor Localisation, Odometry, and SLAM".  The problem is linear, so a single
weighted-least-squares solve (one Gauss--Newton step) is exact; the script also checks the
numbers quoted in the chapter and prints the marginal covariances with and without the
loop closure.  A second cell solves the same graph with a wrong (overconfident) loop closure
to illustrate the role of the information matrix.
"""
# %% [markdown]
# # 1-D pose graph with a loop closure (linear weighted least squares)

# %%
import numpy as np

# unknowns: x1, x2, x3 (x0 = 0 is anchored by a prior)
# odometry: x1 - x0 = 1.0, x2 - x1 = 1.0, x3 - x2 = 1.0   (std 0.10 m each)
# loop closure: x3 - x0 = 2.7                              (std 0.10 m)
A = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 1], [0, 0, 1.0]])
b = np.array([1.0, 1.0, 1.0, 2.7])
sig = np.array([0.1, 0.1, 0.1, 0.1])
W = np.diag(1.0 / sig ** 2)
Lam = A.T @ W @ A
xhat = np.linalg.solve(Lam, A.T @ W @ b)
P = np.linalg.inv(Lam)
print("information matrix Lambda =\n", Lam)
print("solution x1, x2, x3 =", np.round(xhat, 4))
print("residuals (odo1, odo2, odo3, loop) =", np.round(A @ xhat - b, 4))
print("cost sum of squares (weighted) = %.3f" % float((A @ xhat - b) @ W @ (A @ xhat - b)))
print("std of x3 with loop closure = %.4f m" % np.sqrt(P[2, 2]))
# without the loop closure (dead reckoning): covariance of x3 is the sum of the three odometry variances
P_odo = np.linalg.inv(A[:3].T @ W[:3, :3] @ A[:3])
print("std of x3 without loop closure = %.4f m" % np.sqrt(P_odo[2, 2]))

# %% [markdown]
# ## Overconfident loop closure (std 0.02 m instead of 0.10 m)

# %%
sig2 = np.array([0.1, 0.1, 0.1, 0.02])
W2 = np.diag(1.0 / sig2 ** 2)
x2 = np.linalg.solve(A.T @ W2 @ A, A.T @ W2 @ b)
print("solution with overconfident loop closure:", np.round(x2, 4), " residuals:", np.round(A @ x2 - b, 4))
