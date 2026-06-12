# -*- coding: ascii -*-
"""cost_addendum.py - clean asymptotic cost measurement for the Koopman operator.

The make-or-break: is preparing H and reading its dominant spectral feature poly(n)
or 2^n? The main run's automatic slope was overhead-dominated at small N. Here we
push to larger n, average many trials, and report per-step ratios (a ratio ~4 per
delta_n=2 means cost ~ N = 2^n). Also a robust point-gap winding-contour timing.
"""
import os, sys, time, json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import koopman_sensor as K
PHASE6 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PHASE6, "fold_audit"))
import construction as C


def robust_winding_time(P, N, K_pts=24):
    """Time K_pts sparse-LU solves on (P - E I) for E safely OUTSIDE the spectrum
    (|E|>1, never singular). Measures the per-contour-point cost of the literal
    argument-principle winding integral = K_pts * cost(one O(N)-ish LU)."""
    I = sp.identity(N, format="csc")
    t0 = time.perf_counter()
    for j in range(K_pts):
        E = 2.0 + 0.3 * np.exp(2j * np.pi * j / K_pts)
        lu = spla.splu((P - E * I).tocsc())
        _ = np.sum(np.log(lu.U.diagonal().astype(complex)))
    return time.perf_counter() - t0


def main():
    rng = np.random.default_rng(7777)
    ns = [10, 12, 14, 16, 18]
    rows = []
    for n in ns:
        N = 1 << n
        M = C.M_for(n)
        bt, ft = [], []
        trials = 9 if n <= 16 else 4
        P_last = None
        for _ in range(trials):
            d = C.sample_secret(N, rng)
            k, b = C.coset_samples(N, d, M, rng)
            t0 = time.perf_counter()
            s = K.score_all_x(k, b, N)              # FFT score at all x  (build core)
            thresh = len(b) / 4.0
            accept = s > thresh
            x = np.arange(N); f = (x + 1) % N; f[accept] = x[accept]
            t1 = time.perf_counter()
            fixed = np.flatnonzero(accept)
            _ = K._topfixed(fixed, s, N, 2)         # read pinning site (the answer)
            t2 = time.perf_counter()
            bt.append(t1 - t0); ft.append(t2 - t1)
            P_last = sp.csc_matrix((np.ones(N), (f, x)), shape=(N, N))
        wt = robust_winding_time(P_last, N, K_pts=24)
        rows.append({"n": n, "N": N, "build_s": float(np.median(bt)),
                     "readfp_s": float(np.median(ft)), "winding24_s": float(wt)})
    print("  n   N         build_s      readfp_s     winding24_s   build_ratio(/prev, dn=2)")
    for i, r in enumerate(rows):
        rr = "" if i == 0 else "  x%.2f (N x4)" % (r["build_s"] / rows[i-1]["build_s"])
        print("  %-3d %-9d %.6f     %.6f     %.5f%s"
              % (r["n"], r["N"], r["build_s"], r["readfp_s"], r["winding24_s"], rr))
    # asymptotic slope from the largest 3 points (overhead-free region)
    big = rows[-3:]
    nn = np.array([r["n"] for r in big], float)
    for key in ("build_s", "readfp_s", "winding24_s"):
        ys = np.array([r[key] for r in big], float)
        A = np.vstack([nn, np.ones(len(nn))]).T
        slope = np.linalg.lstsq(A, np.log2(ys), rcond=None)[0][0]
        print("  asymptotic log2(%s)/n = %.3f  (1.0 => ~2^n)" % (key, slope))
    with open(os.path.join(HERE, "cost_addendum_result.json"), "w") as fh:
        json.dump({"rows": rows}, fh, indent=2, default=float)
    print("  wrote cost_addendum_result.json")


if __name__ == "__main__":
    main()
