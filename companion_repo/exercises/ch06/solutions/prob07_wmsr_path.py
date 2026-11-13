"""Problem 7: robustness of the running network (ring 1-2-3-4-5-1 with chord 1-3) and W-MSR with F=1 on it,
with node 5 broadcasting 30 m (initial values of the consensus example of the chapter)."""
import itertools

import numpy as np

EDGES = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 3)]
N = 5
A = np.zeros((N, N), dtype=int)
for i, j in EDGES:
    A[i - 1, j - 1] = A[j - 1, i - 1] = 1
nb = [set(np.flatnonzero(A[i])) for i in range(N)]


def is_r_robust(r):
    for code in itertools.product(range(3), repeat=N):
        s1 = {v for v in range(N) if code[v] == 1}
        s2 = {v for v in range(N) if code[v] == 2}
        if s1 and s2 and not any(any(len(nb[v] - s) >= r for v in s) for s in (s1, s2)):
            return False, ([v + 1 for v in sorted(s1)], [v + 1 for v in sorted(s2)])
    return True, None


for r in (1, 2, 3):
    ok, wit = is_r_robust(r)
    print("running network %d-robust: %s" % (r, ok), "" if ok else "(counter-example S1=%s, S2=%s)" % wit)

x = np.array([10.0, 12.0, 11.0, 15.0, 30.0])
eps, f_trim, bad = 0.3, 1, 4
for k in range(60):
    xn = x.copy()
    for i in range(N):
        if i == bad:
            continue
        vals = x[sorted(nb[i])]
        keep = list(vals)
        for v in sorted(vals[vals > x[i]])[::-1][:f_trim]:
            keep.remove(v)
        for v in sorted(vals[vals < x[i]])[:f_trim]:
            keep.remove(v)
        xn[i] = x[i] + eps * sum(np.array(keep) - x[i])
    xn[bad] = 30.0
    x = xn
    if k in (0, 1, 4, 9, 29, 59):
        print("k=%2d: compliant nodes 1-4 %s" % (k + 1, np.round(x[:4], 3)))
print("spread of the compliant nodes after 60 rounds: %.3f m" % (x[:4].max() - x[:4].min()))
