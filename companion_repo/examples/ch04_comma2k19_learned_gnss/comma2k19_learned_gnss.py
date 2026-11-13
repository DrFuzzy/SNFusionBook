"""Learning the GNSS error model of a car from public driving data (comma2k19) and using it in an EKF.

Chapter 4 (Learning-Based and Hybrid Sensor Fusion), worked application of the section
"Hybrid Classical--Learning Architectures".

Data: comma2k19 (Schafer et al., 2018; MIT licence), u-blox GNSS fixes, phone gyroscope, CAN vehicle speed and the
reference poses of the dataset. Run download_comma2k19.py first (about 110 MB are transferred).

Steps
  1. alignment: fixes are time-stamped with their own GPS time, not with their arrival time (latency 0.1-0.2 s);
  2. baseline: EKF (position, heading, gyro bias, speed scale) with the receiver treated as white noise of 2.5 m;
  3. learning on the training routes of the Civic: constant along-track offset, then a Gaussian-process model of the
     remaining error with kernel  sigma_b^2 + sigma_g(u_i) sigma_g(u_j) exp(-|t_i - t_j| / T)  (a random constant per
     drive plus an Ornstein-Uhlenbeck component), fitted by maximising the marginal likelihood; sigma_g is either a
     constant or a function of the number of satellites and of the mean C/N0;
  4. hybrid filter: the EKF augmented with the state-space form of that kernel (two constant and two Gauss-Markov
     states), using the learned parameters;
  5. distribution shift: the same models on another car (RAV4, roof antenna), with and without clipping the
     features to their training range.

Outputs (written to output/; `make export` copies the CSV files to data/ch04 of the book):
    c2k_acf.csv        empirical autocovariance of the training errors and fitted kernel
    c2k_route.csv      errors and 2-sigma bounds along one test route (baseline and learned filter)
    c2k_coverage.csv   empirical vs nominal coverage of the error ellipses for each filter
    c2k_results.csv    the numbers of the results table
    c2k_preview.pdf    matplotlib preview
All numbers quoted in the chapter are printed by this script.

usage: python comma2k19_learned_gnss.py [--data DIR]   (default ../../data_cache/comma2k19)
"""
# %% [markdown]
# # Learning the GNSS error model from comma2k19 and using it in an EKF
# Needs numpy, scipy and matplotlib. The data files are numpy arrays, loaded with allow_pickle=False.

# %%
import argparse
import csv
import glob
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
A_WGS, E2_WGS = 6378137.0, 6.69437999014e-3
GPS_UTC_OFFSET = 315964800.0 - 18.0          # Unix epoch -> GPS epoch, 18 leap seconds in 2018
DATASHEET_SIGMA = 2.5                        # m, horizontal accuracy assumed by the baseline


# %% [markdown]
# ## 1. Loading and aligning the data

# %%
def lla2ecef(lat, lon, h):
    lat, lon = np.radians(lat), np.radians(lon)
    n = A_WGS / np.sqrt(1 - E2_WGS * np.sin(lat) ** 2)
    return np.stack([(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - E2_WGS) + h) * np.sin(lat)], 1)


def ecef2enu_matrix(x0):
    x, y, z = x0
    lon, p = np.arctan2(y, x), np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - E2_WGS))
    for _ in range(5):
        n = A_WGS / np.sqrt(1 - E2_WGS * np.sin(lat) ** 2)
        lat = np.arctan2(z + E2_WGS * n * np.sin(lat), p)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])


def _load(d, name):
    return np.load(os.path.join(d, name), allow_pickle=False)


def load_route(rdir):
    """Concatenate the one-minute segments of a route; all times are converted to GPS seconds."""
    acc = {}
    for seg in sorted((s for s in os.listdir(rdir) if s.isdigit()), key=int):
        p = os.path.join(rdir, seg)
        gp, pl = os.path.join(p, "global_pose"), os.path.join(p, "processed_log")
        fgt = _load(gp, "frame_gps_times")
        parts = dict(ft=_load(gp, "frame_times"), tref=fgt[:, 0] * 604800.0 + fgt[:, 1],
                     fp=_load(gp, "frame_positions"), fv=_load(gp, "frame_velocities"),
                     gt=_load(os.path.join(pl, "GNSS/live_gnss_ublox"), "t"),
                     gv=_load(os.path.join(pl, "GNSS/live_gnss_ublox"), "value"),
                     rv=_load(os.path.join(pl, "GNSS/raw_gnss_ublox"), "value"),
                     gyt=_load(os.path.join(pl, "IMU/gyro"), "t"), gyv=_load(os.path.join(pl, "IMU/gyro"), "value"),
                     cst=_load(os.path.join(pl, "CAN/speed"), "t"), csv=_load(os.path.join(pl, "CAN/speed"), "value")[:, 0])
        for k, v in parts.items():
            acc.setdefault(k, []).append(v)
    r = {k: np.concatenate(v) for k, v in acc.items()}
    for key, deps in (("ft", ["tref", "fp", "fv"]), ("gyt", ["gyv"]), ("cst", ["csv"]), ("gt", ["gv"])):
        o = np.argsort(r[key])
        r[key] = r[key][o]
        for k in deps:
            r[k] = r[k][o]
    boot2gps = np.median(r["tref"] - r["ft"])           # device boot time -> GPS time
    gv = r["gv"]                                          # lat, lon, speed, UTC [ms], altitude, bearing [deg]
    tfix = gv[:, 3] / 1000.0 - GPS_UTC_OFFSET             # GPS time of each fix (its own epoch)
    latency = (r["gt"] + boot2gps) - tfix                 # arrival time minus epoch
    m = ecef2enu_matrix(r["fp"][0])
    ref = (r["fp"] - r["fp"][0]) @ m.T
    vref = r["fv"] @ m.T
    fix = (lla2ecef(gv[:, 0], gv[:, 1], gv[:, 4]) - r["fp"][0]) @ m.T
    ix = lambda tq, x: np.stack([np.interp(tq, r["tref"], x[:, k], left=np.nan, right=np.nan) for k in range(2)], 1)
    pr, vr = ix(tfix, ref), ix(tfix, vref)
    # number of satellites with C/N0 >= 30 dB-Hz and their mean C/N0, at the epoch of each fix
    rv = r["rv"]
    ep = np.round(rv[:, 1] * 604800.0 + rv[:, 2], 3)
    u, inv = np.unique(ep, return_inverse=True)
    good = rv[:, 8] >= 30.0
    nsat_ep = np.bincount(inv, weights=good.astype(float), minlength=len(u))
    cno_ep = np.bincount(inv, weights=np.where(good, rv[:, 8], 0.0), minlength=len(u)) / np.maximum(nsat_ep, 1)
    j = np.clip(np.searchsorted(u, tfix), 1, len(u) - 1)
    j = np.where(np.abs(u[j - 1] - tfix) < np.abs(u[j] - tfix), j - 1, j)
    far = np.abs(u[j] - tfix) > 0.15
    nsat, cno = np.where(far, 0.0, nsat_ep[j]), np.where(far, 0.0, cno_ep[j])
    spd = np.hypot(vr[:, 0], vr[:, 1])
    keep = ~np.isnan(pr[:, 0]) & (spd > 3.0)
    return dict(name=os.path.basename(rdir), t=tfix[keep], fix=fix[keep, :2], ref=pr[keep], vref=vr[keep],
                nsat=nsat[keep], cno=cno[keep], vfix=gv[keep, 2], psi_fix=np.pi / 2 - np.radians(gv[keep, 5]),
                latency=latency[keep], gyt=r["gyt"] + boot2gps, wz=-r["gyv"][:, 2],
                cst=r["cst"] + boot2gps, csv=r["csv"], rate=(len(r["gt"]) - 1) / (r["gt"][-1] - r["gt"][0]))


def along_cross(d, e):
    th = d["vref"] / np.linalg.norm(d["vref"], axis=1, keepdims=True)
    return np.sum(e * th, 1), -e[:, 0] * th[:, 1] + e[:, 1] * th[:, 0]


# %% [markdown]
# ## 2. Learning the error model (maximum likelihood)

# %%
def features(nsat, cno):
    return np.stack([np.ones_like(nsat), (nsat - 10.0) / 5.0, (cno - 34.0) / 5.0], -1)


def error_series(d, b_al, dt=2.0):
    """Error of the fixes after the along-track offset, resampled every dt seconds (long gaps removed)."""
    th = d["vref"] / np.linalg.norm(d["vref"], axis=1, keepdims=True)
    e = d["fix"] - d["ref"] - b_al * th
    tt = np.arange(d["t"][0], d["t"][-1], dt)
    ok = np.interp(tt, d["t"], np.r_[0.0, (np.diff(d["t"]) < 2.0)]) > 0.5
    e_s = np.stack([np.interp(tt, d["t"], e[:, k]) for k in range(2)], 1)
    f_s = features(np.interp(tt, d["t"], d["nsat"]), np.interp(tt, d["t"], d["cno"]))
    return tt[ok], e_s[ok], f_s[ok]


def kernel_params(p, hetero):
    """p = (log sigma_b, log T, theta...) -> sigma_b, T, theta (theta has 3 entries; zeros beyond theta_0 if constant)."""
    theta = np.asarray(p[2:]) if hetero else np.array([p[2], 0.0, 0.0])
    return np.exp(p[0]), np.exp(p[1]), theta


def neg_log_marginal_likelihood(p, data, hetero, sig_w=0.1):
    """Average negative log marginal likelihood per sample of the GP error model (both horizontal axes)."""
    sb, tcorr, theta = kernel_params(p, hetero)
    total, n = 0.0, 0
    for tt, e, f in data:
        sg = np.exp(f @ theta)
        k = sb ** 2 + np.outer(sg, sg) * np.exp(-np.abs(tt[:, None] - tt[None, :]) / tcorr) + sig_w ** 2 * np.eye(len(tt))
        chol = np.linalg.cholesky(k)
        a = np.linalg.solve(chol, e)
        total += 0.5 * np.sum(a ** 2) + e.shape[1] * np.sum(np.log(np.diag(chol))) + 0.5 * e.size * np.log(2 * np.pi)
        n += e.size
    return total / n


# %% [markdown]
# ## 3. The EKF: position, heading, gyro bias, CAN speed scale (+ state-space form of the error kernel)

# %%
def run_filter(d, model, b_al=0.0, gp=None, hetero=False, clip=None, dt_grid=0.05, sig_white=DATASHEET_SIGMA,
               sig_meas=0.1):
    """model 'white': fix = position + white noise of std sig_white (states 0-4 only).
    model 'gp': fix = position + offset b_al along the heading + constant bias (states 5-6) + Gauss-Markov error
    (states 7-8) + white noise sig_meas, with the learned kernel parameters gp."""
    i0 = int(np.argmax(d["vfix"] > 10.0))
    t0, t1 = d["t"][i0], d["t"][-1]
    ev_t = np.concatenate([np.arange(t0, t1, dt_grid), d["t"][i0:]])
    ev_k = np.concatenate([-np.ones(len(ev_t) - (len(d["t"]) - i0), int), np.arange(i0, len(d["t"]))])
    o = np.argsort(ev_t, kind="stable")
    ev_t, ev_k = ev_t[o], ev_k[o]
    nsat, cno = d["nsat"], d["cno"]
    if clip is not None:
        nsat, cno = np.clip(nsat, *clip[0]), np.clip(cno, *clip[1])
    use_gp = model == "gp"
    n = 9
    psi0 = d["psi_fix"][i0]
    x = np.zeros(n)
    x[0:2], x[2], x[4] = d["fix"][i0] - b_al * np.array([np.cos(psi0), np.sin(psi0)]), psi0, 1.0
    p = np.diag([25.0, 25.0, 0.05 ** 2, 0.03 ** 2, 0.02 ** 2, 0.0, 0.0, 0.0, 0.0])
    if use_gp:
        sb, tcorr, theta = kernel_params(gp, hetero)
        sig = np.exp(features(nsat, cno) @ theta)
        p[0, 0] = p[1, 1] = 1.0
        p[5, 5] = p[6, 6] = sb ** 2
        p[7, 7] = p[8, 8] = sig[i0] ** 2
        s_cur = sig[i0]
    sig_v, sig_w, q_b, q_s = 0.2, 0.01, 1e-7, 1e-8
    tprev, rec = ev_t[0], []
    for tk, kk in zip(ev_t, ev_k):
        dt = tk - tprev
        if dt > 0:
            v = np.interp(tk - dt / 2, d["cst"], d["csv"])
            w = np.interp(tk - dt / 2, d["gyt"], d["wz"])
            c, s, sc = np.cos(x[2]), np.sin(x[2]), x[4]
            f = np.eye(n)
            f[0, 2], f[1, 2], f[2, 3] = -sc * v * s * dt, sc * v * c * dt, -dt
            f[0, 4], f[1, 4] = v * c * dt, v * s * dt
            x[0] += sc * v * c * dt
            x[1] += sc * v * s * dt
            x[2] += (w - x[3]) * dt
            g = np.zeros((n, 2))
            g[0, 0], g[1, 0], g[2, 1] = c * dt, s * dt, dt
            q = g @ np.diag([sig_v ** 2, sig_w ** 2]) @ g.T
            q[3, 3] += q_b * dt
            q[4, 4] += q_s * dt
            if use_gp:
                ph = np.exp(-dt / tcorr)
                f[7, 7] = f[8, 8] = ph
                x[7:9] *= ph
                q[7, 7] += s_cur ** 2 * (1 - ph ** 2)
                q[8, 8] += s_cur ** 2 * (1 - ph ** 2)
            p = f @ p @ f.T + q
            tprev = tk
        if kk < 0:
            continue
        c, s = np.cos(x[2]), np.sin(x[2])
        h_mat = np.zeros((2, n))
        h_mat[:, 0:2] = np.eye(2)
        h_mat[0, 2], h_mat[1, 2] = -b_al * s, b_al * c
        h = x[0:2] + b_al * np.array([c, s])
        if use_gp:
            s_cur = sig[kk]
            h = h + x[5:7] + x[7:9]
            h_mat[:, 5:9] = np.hstack([np.eye(2), np.eye(2)])
            r_mat = sig_meas ** 2 * np.eye(2)
        else:
            r_mat = sig_white ** 2 * np.eye(2)
        nu = d["fix"][kk] - h
        s_mat = h_mat @ p @ h_mat.T + r_mat
        k_gain = p @ h_mat.T @ np.linalg.inv(s_mat)
        x = x + k_gain @ nu
        ikh = np.eye(n) - k_gain @ h_mat
        p = ikh @ p @ ikh.T + k_gain @ r_mat @ k_gain.T
        rec.append((tk, *(x[0:2] - d["ref"][kk]), p[0, 0], p[1, 1], p[0, 1], nu @ np.linalg.solve(s_mat, nu)))
    a = np.array(rec)
    e, pp = a[:, 1:3], np.stack([np.stack([a[:, 3], a[:, 5]], 1), np.stack([a[:, 5], a[:, 4]], 1)], 1)
    nees = np.einsum("ki,kij,kj->k", e, np.linalg.inv(pp), e)
    return dict(t=a[:, 0], e=e, pp=pp, nees=nees, nis=a[:, 6])


def summary(results):
    e = np.concatenate([r["e"] for r in results])
    nees = np.concatenate([r["nees"] for r in results])
    nis = np.concatenate([r["nis"] for r in results])
    sd = np.sqrt(np.concatenate([r["pp"][:, 0, 0] + r["pp"][:, 1, 1] for r in results]))
    return dict(rmse=np.sqrt(np.mean(np.sum(e ** 2, 1))), nees=nees.mean(), cov2=np.mean(nees <= 4.0),
                nis=nis.mean(), sd=np.sqrt(np.mean(sd ** 2)), nees_all=nees)


# %% [markdown]
# ## 4. Run everything

# %%
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "..", "data_cache", "comma2k19"))
    args = ap.parse_args()
    if not os.path.isdir(os.path.join(args.data, "train")):
        print(f"comma2k19 data not found in {os.path.abspath(args.data)}: run download_comma2k19.py first "
              "(about 110 MB); the CSV files used by the book are already in output/.")
        return 2
    os.makedirs(OUT, exist_ok=True)
    routes = {role: [load_route(r) for r in sorted(glob.glob(os.path.join(args.data, role, "*")))]
              for role in ("train", "test", "rav4")}
    for role, rs in routes.items():
        mins = sum((r["t"][-1] - r["t"][0]) for r in rs) / 60
        rates = ", ".join(str(round(r["rate"])) for r in rs)
        print(f"{role:5s}: {len(rs)} routes, {sum(len(r['t']) for r in rs)} fixes, {mins:.0f} min, "
              f"fix rate {rates} Hz, "
              f"median satellites {np.median(np.concatenate([r['nsat'] for r in rs])):.0f}, "
              f"median C/N0 {np.median(np.concatenate([r['cno'][r['nsat'] > 0] for r in rs])):.1f} dB-Hz")
    lat = np.concatenate([r["latency"] for rs in routes.values() for r in rs])
    print(f"fix latency (arrival - epoch): median {np.median(lat) * 1e3:.0f} ms, 5-95% "
          f"{np.percentile(lat, 5) * 1e3:.0f}-{np.percentile(lat, 95) * 1e3:.0f} ms")

    # ---------------------------------------------------------------- raw errors and systematic offset
    for role, rs in routes.items():
        e = np.concatenate([r["fix"] - r["ref"] for r in rs])
        al, cr = along_cross({"vref": np.concatenate([r["vref"] for r in rs])}, e)
        print(f"{role:5s} raw fix error: horizontal RMS {np.sqrt(np.mean(np.sum(e ** 2, 1))):.2f} m, along-track mean "
              f"{al.mean():+.2f} m (sd {al.std():.2f}), cross-track mean {cr.mean():+.2f} m (sd {cr.std():.2f})")
    tr = routes["train"]
    al_tr, _ = along_cross({"vref": np.concatenate([r["vref"] for r in tr])},
                           np.concatenate([r["fix"] - r["ref"] for r in tr]))
    b_al = al_tr.mean()
    print(f"learned along-track offset b_al = {b_al:.2f} m")

    # ---------------------------------------------------------------- empirical autocovariance (pooled, 2-s grid)
    lags = np.arange(0, 181, 2.0)
    data_tr = [error_series(r, b_al) for r in tr]
    num, cnt = np.zeros(len(lags)), np.zeros(len(lags))
    for tt, e, _ in data_tr:
        for i, lag in enumerate(lags):
            m = int(round(lag / 2.0))
            if m < len(tt):
                num[i] += np.sum(e[:len(tt) - m] * e[m:])
                cnt[i] += e[m:].size
    autocov = num / cnt
    for lag in (0, 10, 30, 60, 120):
        print(f"empirical autocovariance at {lag:3d} s: {autocov[int(lag / 2)]:.2f} m^2")
    sig_white_learned = np.sqrt(autocov[0])
    print(f"learned white-noise std (ignoring correlation): {sig_white_learned:.2f} m")

    # ---------------------------------------------------------------- marginal-likelihood fits
    fits = {}
    for hetero in (False, True):
        x0 = np.r_[np.log(1.5), np.log(40.0), np.log(2.0), (0.0, 0.0) if hetero else ()]
        res = minimize(neg_log_marginal_likelihood, x0, args=(data_tr, hetero), method="L-BFGS-B")
        fits[hetero] = res.x
        sb, tcorr, theta = kernel_params(res.x, hetero)
        print(f"GP {'feature-based' if hetero else 'constant'}: sigma_b = {sb:.2f} m, T = {tcorr:.0f} s, "
              f"theta = {np.round(theta, 3)}, sigma_g at 10 satellites and 34 dB-Hz = {np.exp(theta[0]):.2f} m")
    nll = {}
    for role in ("train", "test", "rav4"):
        dd = [error_series(r, b_al) for r in routes[role]]
        nll[role] = (neg_log_marginal_likelihood(fits[False], dd, False), neg_log_marginal_likelihood(fits[True], dd, True))
        print(f"average negative log marginal likelihood per sample on {role}: constant {nll[role][0]:.3f}, "
              f"feature-based {nll[role][1]:.3f}")
    sb_c, t_c, th_c = kernel_params(fits[False], False)
    ns_tr = np.concatenate([r["nsat"] for r in tr])
    cn_tr = np.concatenate([r["cno"][r["nsat"] > 0] for r in tr])
    clip = ((np.percentile(ns_tr, 1), np.percentile(ns_tr, 99)), (np.percentile(cn_tr, 1), np.percentile(cn_tr, 99)))
    print(f"training range of the features (1-99 %): satellites {clip[0][0]:.0f}-{clip[0][1]:.0f}, "
          f"C/N0 {clip[1][0]:.1f}-{clip[1][1]:.1f} dB-Hz")

    # ---------------------------------------------------------------- filters
    configs = [("white, datasheet 2.5 m", dict(model="white")),
               ("white, tuned to NIS 0.3 m", dict(model="white", sig_white=0.3)),
               ("white, learned variance", dict(model="white", sig_white=sig_white_learned)),
               ("white, learned variance and offset", dict(model="white", sig_white=sig_white_learned, b_al=b_al)),
               ("learned kernel, constant", dict(model="gp", b_al=b_al, gp=fits[False], hetero=False)),
               ("learned kernel, feature-based", dict(model="gp", b_al=b_al, gp=fits[True], hetero=True)),
               ("learned kernel, features clipped", dict(model="gp", b_al=b_al, gp=fits[True], hetero=True, clip=clip))]
    rows, runs = [], {}
    for role in ("test", "rav4"):
        for name, kw in configs:
            if role == "test" and "clipped" in name:
                continue
            res = [run_filter(r, **kw) for r in routes[role]]
            sm = summary(res)
            runs[(role, name)] = (res, sm)
            rows.append([role, name, round(sm["rmse"], 2), round(sm["sd"], 2), round(sm["nees"], 2),
                         round(100 * sm["cov2"], 1), round(sm["nis"], 2)])
            print(f"{role:5s} {name:34s} RMSE {sm['rmse']:.2f} m | reported sd {sm['sd']:.2f} m | mean NEES "
                  f"{sm['nees']:8.2f} | inside c=2 ellipse {100 * sm['cov2']:5.1f} % | mean NIS {sm['nis']:.2f}")
    raw = {role: np.sqrt(np.mean(np.concatenate([np.sum((r["fix"] - r["ref"]) ** 2, 1) for r in routes[role]])))
           for role in ("test", "rav4")}
    print(f"raw fix RMSE: test {raw['test']:.2f} m, rav4 {raw['rav4']:.2f} m")

    # ---------------------------------------------------------------- CSV outputs
    with open(os.path.join(OUT, "c2k_results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "filter", "rmse", "sd", "nees", "cov2", "nis"])
        w.writerows(rows)
    fit_curve = sb_c ** 2 + np.exp(2 * th_c[0]) * np.exp(-lags / t_c)
    np.savetxt(os.path.join(OUT, "c2k_acf.csv"), np.column_stack([lags, autocov, fit_curve]),
               delimiter=",", header="lag,autocov,kernel", comments="", fmt="%.4f")
    cs = np.linspace(0.05, 4.0, 80)
    cov_cols, hdr = [1 - np.exp(-cs ** 2 / 2)], ["nominal"]
    for key, tag in ((("test", "white, datasheet 2.5 m"), "white"), (("test", "white, learned variance"), "whitelearned"),
                     (("test", "learned kernel, constant"), "const"), (("test", "learned kernel, feature-based"), "feat"),
                     (("rav4", "learned kernel, constant"), "constrav"), (("rav4", "learned kernel, feature-based"), "featrav")):
        nees = runs[key][1]["nees_all"]
        cov_cols.append(np.array([np.mean(nees <= c ** 2) for c in cs]))
        hdr.append(tag)
    np.savetxt(os.path.join(OUT, "c2k_coverage.csv"), np.column_stack(cov_cols), delimiter=",",
               header=",".join(hdr), comments="", fmt="%.4f")
    # one test route: east error and 2-sigma bounds of the baseline and of the learned filter, at 1 Hz
    k = 1
    rb, rl = runs[("test", "white, datasheet 2.5 m")][0][k], runs[("test", "learned kernel, constant")][0][k]
    tt = np.arange(max(rb["t"][0], rl["t"][0]), min(rb["t"][-1], rl["t"][-1]), 1.0)
    col = lambda r, x: np.interp(tt, r["t"], x)
    np.savetxt(os.path.join(OUT, "c2k_route.csv"),
               np.column_stack([(tt - tt[0]) / 60.0, col(rb, rb["e"][:, 0]), 2 * np.sqrt(col(rb, rb["pp"][:, 0, 0])),
                                col(rl, rl["e"][:, 0]), 2 * np.sqrt(col(rl, rl["pp"][:, 0, 0]))]),
               delimiter=",", header="tmin,ebase,sbase,elearn,slearn", comments="", fmt="%.4f")
    print(f"route used for the figure: {routes['test'][k]['name']}")

    # ---------------------------------------------------------------- preview
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(lags, autocov, "o", ms=3, label="training errors")
    ax[0].plot(lags, fit_curve, "-", label="learned kernel")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel("lag [s]"); ax[0].set_ylabel("autocovariance [m$^2$]"); ax[0].legend()
    route = np.loadtxt(os.path.join(OUT, "c2k_route.csv"), delimiter=",", skiprows=1)
    ax[1].plot(route[:, 0], route[:, 1], "C0", label="white, 2.5 m")
    ax[1].fill_between(route[:, 0], -route[:, 2], route[:, 2], color="C0", alpha=0.25)
    ax[1].plot(route[:, 0], route[:, 3], "C1", label="learned kernel")
    ax[1].plot(route[:, 0], route[:, 4], "C1--"); ax[1].plot(route[:, 0], -route[:, 4], "C1--")
    ax[1].set_xlabel("time [min]"); ax[1].set_ylabel("east error [m]"); ax[1].legend()
    cov = np.loadtxt(os.path.join(OUT, "c2k_coverage.csv"), delimiter=",", skiprows=1)
    for j, lab in enumerate(hdr[1:], 1):
        ax[2].plot(cov[:, 0], cov[:, j], label=lab)
    ax[2].plot([0, 1], [0, 1], "k:"); ax[2].set_xlabel("nominal probability"); ax[2].set_ylabel("empirical coverage")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "c2k_preview.pdf"))
    print("outputs written to", OUT)


if __name__ == "__main__":
    sys.exit(main() or 0)
