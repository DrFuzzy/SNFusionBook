"""Solution script for Problem 5.4 (Chapter 5): GNSS--IMU--wheel-odometry EKF with
(a) a 60 s outage and (b) 1 Hz GNSS fixes before a 25 s outage.
Compact, self-contained re-implementation of companion_repo/examples/ch05_gnss_imu_odometry_ekf.
Deterministic (np.random.default_rng(5))."""
# %%
import numpy as np

def simulate(outage=(50.0, 75.0), gnss_every=50, seed=5, T=0.01, t_end=120.0, sig_gnss=1.5,
             sig_gyro=0.002, b_true=0.010, sig_wheel=0.05, s_true=0.02):
    rng = np.random.default_rng(seed)
    N = int(round(t_end / T)); t = np.arange(N) * T
    v_true = 14.0 + 1.0 * np.sin(2 * np.pi * t / 40.0)
    om = np.zeros(N); om[(t >= 15) & (t < 35)] = 0.08; om[(t >= 55) & (t < 70)] = -0.06
    px = np.zeros(N); py = np.zeros(N); psi = np.zeros(N)
    for k in range(1, N):
        psi[k] = psi[k-1] + T*om[k-1]; px[k] = px[k-1] + T*v_true[k-1]*np.cos(psi[k-1]); py[k] = py[k-1] + T*v_true[k-1]*np.sin(psi[k-1])
    om_m = om + b_true + sig_gyro*rng.standard_normal(N)
    v_m = v_true*(1+s_true) + sig_wheel*rng.standard_normal(N)
    avail = np.zeros(N, bool); avail[::gnss_every] = True; avail &= ~((t >= outage[0]) & (t < outage[1]))
    y = np.full((N, 2), np.nan)
    y[avail, 0] = px[avail] + sig_gnss*rng.standard_normal(avail.sum()); y[avail, 1] = py[avail] + sig_gnss*rng.standard_normal(avail.sum())
    # five-state EKF
    x = np.array([y[0, 0], y[0, 1], psi[0] + np.deg2rad(2.0), 0.0, 0.0])
    P = np.diag([sig_gnss**2, sig_gnss**2, np.deg2rad(5.0)**2, 0.02**2, 0.05**2])
    H = np.zeros((2, 5)); H[0, 0] = H[1, 1] = 1.0; R = sig_gnss**2*np.eye(2); W = np.diag([sig_wheel**2, sig_gyro**2])
    X = np.zeros((N, 5)); S3 = np.zeros((N, 5)); X[0] = x; S3[0] = 3*np.sqrt(np.diag(P))
    for k in range(1, N):
        vm, wm = v_m[k-1], om_m[k-1]; s, b = x[4], x[3]; vh = vm/(1+s); c, sn = np.cos(x[2]), np.sin(x[2])
        xp = x.copy(); xp[0] += T*vh*c; xp[1] += T*vh*sn; xp[2] += T*(wm-b)
        F = np.eye(5); F[0, 2] = -T*vh*sn; F[1, 2] = T*vh*c; F[2, 3] = -T; F[0, 4] = -T*vm*c/(1+s)**2; F[1, 4] = -T*vm*sn/(1+s)**2
        G = np.zeros((5, 2)); G[0, 0] = T*c/(1+s); G[1, 0] = T*sn/(1+s); G[2, 1] = T
        Q = G @ W @ G.T; Q[2, 2] += T*1e-3**2; Q[3, 3] += T*1e-4**2; Q[4, 4] += T*1e-4**2
        x = xp; P = F @ P @ F.T + Q
        if avail[k]:
            nu = y[k] - H @ x; S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ nu; P = (np.eye(5) - K @ H) @ P; P = 0.5*(P + P.T)
        X[k] = x; S3[k] = 3*np.sqrt(np.maximum(np.diag(P), 0))
    err = np.hypot(X[:, 0]-px, X[:, 1]-py); rad = np.sqrt(S3[:, 0]**2 + S3[:, 1]**2)
    k_in = np.searchsorted(t, outage[0]); k_end = np.searchsorted(t, outage[1]) - 1
    return dict(err_end=err[k_end], rad_end=rad[k_end], inside=err[k_end] <= rad[k_end],
                b_sig3_mrad=S3[k_in, 3]*1e3, b_est_mrad=X[k_in, 3]*1e3, s_sig3_pct=S3[k_in, 4]*100,
                rms_avail=np.sqrt(np.mean(err[(t > 5) & (t < outage[0])]**2)))

# %%
for label, kw in [("reference (25 s outage, 2 Hz)", dict()),
                  ("(a) 60 s outage, 2 Hz", dict(outage=(50.0, 110.0))),
                  ("(b) 25 s outage, 1 Hz GNSS", dict(gnss_every=100))]:
    r = simulate(**kw)
    print(f"{label:32s}: error at end of outage {r['err_end']:6.2f} m, 3-sigma radius {r['rad_end']:6.2f} m, inside={r['inside']}, "
          f"bias {r['b_est_mrad']:.2f} +/- {r['b_sig3_mrad']:.2f} mrad/s (3 sigma), scale 3-sigma {r['s_sig3_pct']:.2f} %, RMS before outage {r['rms_avail']:.2f} m")
