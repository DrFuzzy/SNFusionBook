"""Problem 7: W-MSR with F=1 on the path graph 1-2-3-4 with agent 4 broadcasting 30 m."""
import itertools, numpy as np

A = np.zeros((4, 4), dtype=int)
for i in range(3):
    A[i, i + 1] = A[i + 1, i] = 1
nb = [set(np.flatnonzero(A[i])) for i in range(4)]


def is_r_robust(r):
    for code in itertools.product(range(3), repeat=4):
        S1 = {v for v in range(4) if code[v] == 1}; S2 = {v for v in range(4) if code[v] == 2}
        if S1 and S2 and not any(any(len(nb[v] - S) >= r for v in S) for S in (S1, S2)):
            return False, (sorted(S1), sorted(S2))
    return True, None


for r in (1, 2, 3):
    ok, wit = is_r_robust(r)
    print("path graph %d-robust: %s" % (r, ok), "" if ok else "(counter-example S1=%s, S2=%s, 0-based)" % wit)

x = np.array([10.0, 12.0, 11.0, 30.0]); eps, F = 0.3, 1
for k in range(30):
    xn = x.copy()
    for i in range(3):                     # compliant agents 1-3
        vals = x[list(nb[i])]
        keep = list(vals)
        for v in sorted(vals[vals > x[i]])[::-1][:F]: keep.remove(v)
        for v in sorted(vals[vals < x[i]])[:F]: keep.remove(v)
        xn[i] = x[i] + eps * sum(np.array(keep) - x[i])
    xn[3] = 30.0
    x = xn
    if k in (0, 1, 4, 9, 29):
        print("k=%2d: compliant agents %s" % (k + 1, np.round(x[:3], 3)))
print("The compliant agents freeze (every agent trims its only differing neighbour) and never agree: no guarantee without 3-robustness.")
