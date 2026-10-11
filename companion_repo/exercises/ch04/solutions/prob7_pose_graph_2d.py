"""Chapter 4, Problem 7: planar pose graph (SE(2)) with four poses, odometry factors and one loop
closure, solved by Gauss--Newton.  Self-contained.

Poses x_i = (x, y, theta).  Odometry factors measure the relative pose in the frame of the
previous node; the loop closure measures the relative pose between node 3 and node 0.
"""
# %%
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rel(xi, xj):
    """relative pose of j expressed in frame i"""
    c, s = np.cos(xi[2]), np.sin(xi[2])
    Rt = np.array([[c, s], [-s, c]])
    d = Rt @ (xj[:2] - xi[:2])
    return np.array([d[0], d[1], wrap(xj[2] - xi[2])])


# ground truth: a 10 m square driven counter-clockwise
truth = np.array([[0, 0, 0], [10, 0, np.pi / 2], [10, 10, np.pi], [0, 10, -np.pi / 2]], float)
rng = np.random.default_rng(7)
sig_odo = np.array([0.3, 0.3, np.deg2rad(2.0)])
sig_lc = np.array([0.1, 0.1, np.deg2rad(1.0)])
edges = []   # (i, j, z, sigma)
for i in range(3):
    z = rel(truth[i], truth[i + 1]) + sig_odo * rng.standard_normal(3)
    edges.append((i, i + 1, z, sig_odo))
z_lc = rel(truth[3], truth[0]) + sig_lc * rng.standard_normal(3)
edges.append((3, 0, z_lc, sig_lc))

# dead-reckoning initial guess (compose the odometry measurements)
x = np.zeros((4, 3))
for i, j, z, s in edges[:3]:
    c, sn = np.cos(x[i, 2]), np.sin(x[i, 2])
    x[j, :2] = x[i, :2] + np.array([[c, -sn], [sn, c]]) @ z[:2]
    x[j, 2] = wrap(x[i, 2] + z[2])
x0_dr = x.copy()


def residuals(x):
    r = []
    for i, j, z, s in edges:
        e = rel(x[i], x[j]) - z
        e[2] = wrap(e[2])
        r.append(e / s)
    return np.concatenate(r)


def jacobian(x, eps=1e-6):
    r0 = residuals(x)
    J = np.zeros((len(r0), 9))        # nodes 1..3 are free, node 0 is anchored
    for n in range(1, 4):
        for d in range(3):
            xp = x.copy(); xp[n, d] += eps
            J[:, 3 * (n - 1) + d] = (residuals(xp) - r0) / eps
    return J, r0


for it in range(10):
    J, r = jacobian(x)
    dx = np.linalg.solve(J.T @ J, -J.T @ r)
    x[1:] += dx.reshape(3, 3)
    x[:, 2] = wrap(x[:, 2])
    if np.linalg.norm(dx) < 1e-8:
        break
print("Gauss-Newton iterations:", it + 1)
print("dead-reckoning pose 3 :", np.round(x0_dr[3], 3), " error %.3f m" % np.linalg.norm(x0_dr[3, :2] - truth[3, :2]))
print("optimised pose 3      :", np.round(x[3], 3), " error %.3f m" % np.linalg.norm(x[3, :2] - truth[3, :2]))
print("weighted cost: dead reckoning %.2f -> optimised %.2f" % (float(residuals(x0_dr) @ residuals(x0_dr)), float(r @ r)))
J, r = jacobian(x)
P = np.linalg.inv(J.T @ J)
print("std of pose 3 (x, y, theta[deg]) :", np.round([np.sqrt(P[6, 6]), np.sqrt(P[7, 7]), np.rad2deg(np.sqrt(P[8, 8]))], 3))
