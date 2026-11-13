"""Problems 9 and 10 of Chapter 4: the comma2k19 GNSS error model (worked application of Sect. "Hybrid Classical--Learning
Architectures").

Problem 9 (computational): learn the error kernel on the four RAV4 drives and use it on the three Civic test drives.
Problem 10: effective number of independent fixes in one minute for the learned kernel.

usage: python prob9_10_comma2k19.py [--data DIR]   (default ../../../data_cache/comma2k19; run download_comma2k19.py first)
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "examples", "ch04_comma2k19_learned_gnss"))
import comma2k19_learned_gnss as c2k  # noqa: E402


def problem9(data_dir):
    routes = {role: [c2k.load_route(r) for r in sorted(glob.glob(os.path.join(data_dir, role, "*")))]
              for role in ("train", "test", "rav4")}
    src = routes["rav4"]
    al, _ = c2k.along_cross({"vref": np.concatenate([r["vref"] for r in src])},
                            np.concatenate([r["fix"] - r["ref"] for r in src]))
    b_al = al.mean()
    data = [c2k.error_series(r, b_al) for r in src]
    fit = minimize(c2k.neg_log_marginal_likelihood, np.r_[np.log(1.0), np.log(40.0), np.log(1.0)],
                   args=(data, False), method="L-BFGS-B").x
    sb, tcorr, theta = c2k.kernel_params(fit, False)
    print(f"Problem 9 - kernel learned on the RAV4: offset {b_al:.2f} m, sigma_b {sb:.2f} m, "
          f"sigma_g {np.exp(theta[0]):.2f} m, T {tcorr:.0f} s")
    for role in ("rav4", "test"):
        res = [c2k.run_filter(r, model="gp", b_al=b_al, gp=fit, hetero=False) for r in routes[role]]
        sm = c2k.summary(res)
        print(f"   on {role:5s}: RMSE {sm['rmse']:.2f} m, reported sd {sm['sd']:.2f} m, mean NEES {sm['nees']:.2f}, "
              f"inside c=2 ellipse {100 * sm['cov2']:.1f} %, mean NIS {sm['nis']:.2f}")


def problem10(sigma_b=2.16, sigma_g=1.37, tcorr=111.0, dt=0.1, n=600):
    """Variance of the mean of n consecutive errors e_k = b + g_k + w_k, with g an OU process."""
    k = np.arange(1, n)
    phi = np.exp(-dt / tcorr)
    var_mean_ou = sigma_g ** 2 / n * (1 + 2 * np.sum((1 - k / n) * phi ** k))
    var_mean = sigma_b ** 2 + var_mean_ou
    var_one = sigma_b ** 2 + sigma_g ** 2
    print(f"Problem 10 - {n} fixes at {1 / dt:.0f} Hz: single-fix variance {var_one:.2f} m^2, "
          f"variance of the mean {var_mean:.2f} m^2 (OU part {var_mean_ou:.2f}), effective number of independent "
          f"fixes {var_one / var_mean:.2f}; a white-noise model would claim {var_one / n:.4f} m^2")
    for t_tot in (60.0, 600.0, 3600.0):
        nn = int(t_tot / dt)
        kk = np.arange(1, nn)
        v_ou = sigma_g ** 2 / nn * (1 + 2 * np.sum((1 - kk / nn) * phi ** kk))
        print(f"   averaging over {t_tot:5.0f} s: std of the mean {np.sqrt(sigma_b ** 2 + v_ou):.2f} m "
              f"(constant part alone {sigma_b:.2f} m)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "..", "..", "data_cache", "comma2k19"))
    args = ap.parse_args()
    problem9(args.data)
    problem10()


if __name__ == "__main__":
    main()
