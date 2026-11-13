"""Shared helper for Problems 5 and 6: consensus-on-information filter on a given graph.
Self-contained compact version of the model of companion_repo/examples/ch06_distributed_tracking."""
import numpy as np

T, K, sigma_a = 0.1, 100, 0.5
F = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1.0]])
Q = sigma_a**2 * np.array([[T**4/4, 0, T**3/2, 0], [0, T**4/4, 0, T**3/2], [T**3/2, 0, T**2, 0], [0, T**3/2, 0, T**2]])
H = [np.array([[1, 0, 0, 0.]]), np.array([[0, 1, 0, 0.]]), np.array([[1, 0, 0, 0.], [0, 1, 0, 0]]),
     np.array([[1, 0, 0, 0.]]), np.array([[0, 1, 0, 0.]])]
R = [np.array([[4.]]), np.array([[4.]]), np.diag([36., 36.]), np.array([[9.]]), np.array([[9.]])]
x0, P0, N, nx = np.array([0., 0., 8., 3.]), np.diag([25., 25., 4., 4.]), 5, 4


def metropolis(edges):
    A = np.zeros((N, N))
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    d = A.sum(1)
    W = np.zeros((N, N))
    for i in range(N):
        for j in np.flatnonzero(A[i]):
            W[i, j] = 1 / (1 + max(d[i], d[j]))
        W[i, i] = 1 - W[i].sum()
    return A, W


def run_coi(edges, L, p_loss=0.0, n_mc=20, seed=6):
    """Returns (position RMSE, mean NEES) of the CoI filter, averaged over nodes, time and runs."""
    A, W = metropolis(edges)
    rng = np.random.default_rng(seed)
    se, ne = [], []
    for _ in range(n_mc):
        X = np.zeros((K + 1, nx)); X[0] = x0
        for k in range(K):
            X[k + 1] = F @ X[k] + rng.multivariate_normal(np.zeros(nx), Q)
        xh = x0 + rng.multivariate_normal(np.zeros(nx), P0)
        xs, Ps = [xh.copy() for _ in range(N)], [P0.copy() for _ in range(N)]
        for k in range(1, K + 1):
            Lams, qs = [], []
            for i in range(N):
                x, P = F @ xs[i], F @ Ps[i] @ F.T + Q
                y = H[i] @ X[k] + rng.multivariate_normal(np.zeros(H[i].shape[0]), R[i])
                Lam = np.linalg.inv(P) + H[i].T @ np.linalg.solve(R[i], H[i])
                q = np.linalg.inv(P) @ x + H[i].T @ np.linalg.solve(R[i], y)
                Lams.append(Lam); qs.append(q)
            for _ in range(L):
                Wk = W.copy()
                if p_loss > 0:            # independent losses, weights renormalised over received messages
                    for i in range(N):
                        for j in np.flatnonzero(A[i]):
                            if rng.random() < p_loss:
                                Wk[i, j] = 0
                        # Retain the original self-weight and renormalise all survivors.
                        Wk[i] /= Wk[i].sum()
                Lams = [sum(Wk[i, j] * Lams[j] for j in range(N)) for i in range(N)]
                qs = [sum(Wk[i, j] * qs[j] for j in range(N)) for i in range(N)]
            for i in range(N):
                Ps[i] = np.linalg.inv(Lams[i]); xs[i] = Ps[i] @ qs[i]
                e = xs[i] - X[k]
                se.append(e[:2] @ e[:2]); ne.append(e @ Lams[i] @ e)
    return float(np.sqrt(np.mean(se))), float(np.mean(ne))
