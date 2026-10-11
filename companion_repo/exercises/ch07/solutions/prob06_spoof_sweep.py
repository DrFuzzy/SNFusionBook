"""Problem 7.6: sweep the spoofing acceleration, the window length and the AR(1) coefficient of the
coloured AIS noise in the radar+AIS constant-velocity filter of Example A (self-contained)."""
# %%
import numpy as np
from scipy.stats import chi2

def run(a_spoof=0.001, W=12, phi=0.85, seed=2026, T=10.0, N=180, M_lb=30, L_lb=5, alpha=0.01):
    rng = np.random.default_rng(seed)
    t = np.arange(N + 1) * T
    v = 12 * 1852 / 3600; h = np.deg2rad(60.0)
    along = np.array([np.sin(h), np.cos(h)]); cross = np.array([np.cos(h), -np.sin(h)])
    F = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    G = np.array([[T**2/2, 0], [0, T**2/2], [T, 0], [0, T]])
    Qf = 0.02**2 * G @ G.T
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0.0]]); Ra, Rr = 100 * np.eye(2), 900 * np.eye(2)
    x = np.zeros((N + 1, 4)); x[0] = [0, 0, v * along[0], v * along[1]]
    for k in range(N):
        x[k + 1] = F @ x[k] + G @ (0.01 * rng.standard_normal(2))
    noise = 10 * rng.standard_normal((N + 1, 2)); e = np.zeros(2)
    for k in range(N + 1):
        if 600 <= t[k] <= 1200:
            e = phi * e + np.sqrt(1 - phi**2) * 10 * rng.standard_normal(2); noise[k] = e
        else:
            e = noise[k]
    spoof = np.array([0.5 * a_spoof * max(tk - 1200, 0)**2 * cross for tk in t])
    y = x[:, :2] + spoof + noise; yr = x[:, :2] + 30 * rng.standard_normal((N + 1, 2))
    xh = np.array([y[0, 0], y[0, 1], v * along[0], v * along[1]]); P = np.diag([100, 100, 1, 1.0])
    nis = np.full(N + 1, np.nan); nu_w = np.full((N + 1, 4), np.nan)
    def upd(xp, Pp, yk, R):
        nu = yk - H @ xp; S = H @ Pp @ H.T + R; Si = np.linalg.inv(S); K = Pp @ H.T @ Si
        return xp + K @ nu, (np.eye(4) - K @ H) @ Pp, nu @ Si @ nu, np.linalg.solve(np.linalg.cholesky(S), nu)
    for k in range(1, N + 1):
        xp = F @ xh; Pp = F @ P @ F.T + Qf
        xh, P, n1, nu_w[k, :2] = upd(xp, Pp, yr[k], Rr)
        xh, P, n2, nu_w[k, 2:] = upd(xh, P, y[k], Ra)
        nis[k] = n1 + n2
    win = np.array([np.nansum(nis[k - W + 1:k + 1]) if k >= W else np.nan for k in range(N + 1)])
    def lb(z):
        z = z - z.mean(); c0 = z @ z / len(z)
        return len(z) * (len(z) + 2) * sum((z[l:] @ z[:-l] / len(z) / c0)**2 / (len(z) - l) for l in range(1, L_lb + 1))
    Q = np.array([sum(lb(nu_w[k - M_lb + 1:k + 1, i]) for i in range(4)) if k >= M_lb else np.nan for k in range(N + 1)])
    a_step = nis > chi2.ppf(1 - alpha, 4); a_win = win > chi2.ppf(1 - alpha, 4 * W); a_lb = Q > chi2.ppf(1 - alpha, 4 * L_lb)
    def first(flag):
        i = np.where(flag & (t >= 1200))[0]; return (None, None) if len(i) == 0 else (float(t[i[0]] - 1200), round(float(np.linalg.norm(spoof[i[0]])), 1))
    return first(a_step), first(a_win), np.mean(a_lb[(t >= 660) & (t < 1200)])

# %%
print("a_spoof  W   step(delay s, offset m)   window(delay s, offset m)")
for a in [0.0005, 0.001, 0.002, 0.004]:
    for W in [6, 12, 24]:
        s, w, _ = run(a_spoof=a, W=W)
        print(f"{a:7.4f} {W:3d}   {s}   {w}")
print("phi  LB alarm fraction in coloured regime")
for phi in [0.5, 0.7, 0.85, 0.95]:
    print(f"{phi:.2f}  {run(phi=phi)[2]:.2f}")
