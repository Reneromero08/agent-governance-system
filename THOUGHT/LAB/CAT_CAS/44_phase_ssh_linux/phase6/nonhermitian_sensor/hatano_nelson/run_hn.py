"""
run_hn.py - drive the Hatano-Nelson non-Hermitian sensor against the REAL Exp 50.14
construction and MEASURE:
  (A) Does any non-Hermitian invariant read the ORIENTATION bit o = 1[d < N/2]?
      -> hardened_gate (random-private-fold + exact d-invariance), n=8,10.
  (B) Does it read the MAGNITUDE a = min(d, N-d)?  -> well-peak (folded) vs a.
  (C) COST scaling of building H + reading invariants across n=8,10,12,14
      -> make-or-break: poly(n) (crossing) or 2^n (wall relocated)?
Plus diagnostics: imaginary-gauge residual (g is a gauge) and winding constancy.

Honest outcome map: (i) reads o POLY -> CROSSING; (ii) reads o EXP -> wall relocated;
(iii) reads o via smuggle -> FAIL_SMUGGLE; (iv) H fold-invariant, cannot read o -> wall
confirmed for public-only H. ASCII only; seeds recorded.
"""
import os, sys, json, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PH6 = os.path.dirname(os.path.dirname(_HERE))
_FOLD = os.path.join(_PH6, "fold_audit")
_STAGE3 = os.path.join(_FOLD, "stage3")
for _p in (_HERE, _FOLD, _STAGE3):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import construction as C
import hardened_gate as HG
import hn_operator as HN

MASTER_SEED = 44061114
OUT = {"master_seed": MASTER_SEED}


def banner(t):
    print("=" * 92); print(t); print("=" * 92)


def sanity(n=10, seed=MASTER_SEED):
    banner("[0] SANITY: gauge residual (g is a gauge), winding constancy, well-peak vs fixed point (n=%d)" % n)
    rng = np.random.default_rng(seed)
    N = 1 << n
    # gauge residual must be demonstrated at SMALL n (e^{g N} overflow); use n=5
    ns_small = 5; Ns = 1 << ns_small
    drng = np.random.default_rng(seed + 1)
    d = C.sample_secret(Ns, drng); ks, bs = C.coset_samples(Ns, d, C.M_for(ns_small), drng)
    ss = HN.score_landscape(ks, bs, Ns)
    gres = HN.gauge_equivalence_residual(ss)
    print("  imaginary-gauge residual max|S^-1 H_obc S - H_herm| at n=%d : %.2e  (==0 => g is a pure gauge)" % (ns_small, gres))
    OUT["gauge_residual_n5"] = gres
    rows = []
    W0s = []
    for _ in range(8):
        d = C.sample_secret(N, rng); a = min(d, N - d); o = C.orientation_bit(d, N)
        inst = HG.G.make_instance(n, d, rng)
        s = HN.score_landscape(inst["k"], inst["b"], N)
        w0, w0r = HN.winding(s, 0.0, n_phi=48); W0s.append(w0)
        peak, pf, ipr, skin = HN.well_mode_position(s)
        rows.append((d, a, o, w0, peak, pf, ipr, skin))
        print("  d=%-6d a=%-6d o=%d | W(0)=%+d | peak=%7.0f peak_folded=%7.0f (a=%-6d err=%6.0f) ipr=%.3f skin_com=%7.1f"
              % (d, a, o, w0, peak, pf, a, abs(pf - a), ipr, skin))
    print("  winding W(0) across 8 instances: %s  (constant => reads the imposed +1, not the data)" % (W0s,))
    OUT["sanity"] = [{"d": int(r[0]), "a": int(r[1]), "o": int(r[2]), "W0": int(r[3]),
                      "peak": float(r[4]), "peak_folded": float(r[5]), "ipr": float(r[6]),
                      "skin_com": float(r[7])} for r in rows]


def magnitude_read(ns=(8, 10, 12), n_instances=60):
    banner("[B] MAGNITUDE READ: does well-peak (folded) track a = min(d, N-d)?")
    cells = []
    for n in ns:
        rng = np.random.default_rng((MASTER_SEED + 13 * n) & 0x7FFFFFFF)
        N = 1 << n
        A, PF = [], []
        for _ in range(n_instances):
            d = C.sample_secret(N, rng); a = min(d, N - d)
            inst = HG.G.make_instance(n, d, rng)
            s = HN.score_landscape(inst["k"], inst["b"], N)
            _, pf, _, _ = HN.well_mode_position(s)
            A.append(a); PF.append(pf)
        A = np.asarray(A, float); PF = np.asarray(PF, float)
        err = np.abs(PF - A) / N
        corr = float(np.corrcoef(A, PF)[0, 1]) if np.std(PF) > 0 else 0.0
        hit = float(np.mean(err < 0.01))
        print("  n=%2d N=%-6d  corr(a,peak_folded)=%+.3f  median|err|/N=%.4f  frac_exact(<1pctN)=%.2f"
              % (n, N, corr, float(np.median(err)), hit))
        cells.append({"n": n, "N": N, "corr": corr, "median_err_over_N": float(np.median(err)),
                      "frac_exact": hit, "n_instances": n_instances})
    OUT["magnitude_read"] = cells
    return cells


def orientation_gate(ns=(8, 10), n_instances=160, n_shuffles=16):
    banner("[A] ORIENTATION GATE (hardened: random-private-fold + exact d-invariance)")
    HN._N_PHI = 32
    cells = []
    for n in ns:
        seed = (MASTER_SEED + 7919 * n) & 0x7FFFFFFF
        t0 = time.time()
        res = HG.hardened_gate(HN.O_hatano_nelson, n, n_instances=n_instances, seed=seed, n_shuffles=n_shuffles)
        dt = time.time() - t0
        print("  n=%2d  verdict=%-13s  orient_auc=%.3f (null95=%.3f)  random_fold_auc=%.3f (null95=%.3f)  delta=%.2g  [%.1fs]"
              % (n, res["verdict"], res["auc"], res["shuffle_null_95"], res["random_fold_auc"],
                 res["random_fold_null_95"], res["max_fold_delta"], dt))
        res["wall_s"] = dt; res["seed"] = int(seed)
        cells.append(res)
    OUT["orientation_gate"] = cells
    HN._N_PHI = 48
    return cells


def cost_scaling(ns=(8, 10, 12, 14), reps=3):
    banner("[C] COST SCALING (make-or-break): build H + read invariants vs n (poly or 2^n?)")
    HN._N_PHI = 48
    rows = []
    for n in ns:
        rng = np.random.default_rng((MASTER_SEED + 101 * n) & 0x7FFFFFFF)
        N = 1 << n
        t_land = t_wind = t_well = t_tot = 0.0
        for _ in range(reps):
            d = C.sample_secret(N, rng); inst = HG.G.make_instance(n, d, rng)
            tic = time.time()
            s = HN.score_landscape(inst["k"], inst["b"], N); t1 = time.time()
            for E in HN._E_REFS:
                HN.winding(s, E, n_phi=HN._N_PHI)
            t2 = time.time()
            HN.well_mode_position(s); t3 = time.time()
            t_land += t1 - tic; t_wind += t2 - t1; t_well += t3 - t2; t_tot += t3 - tic
        t_land /= reps; t_wind /= reps; t_well /= reps; t_tot /= reps
        print("  n=%2d N=%-6d  landscape=%.3fs  winding=%.3fs  well=%.3fs  TOTAL=%.3fs"
              % (n, N, t_land, t_wind, t_well, t_tot))
        rows.append({"n": n, "N": N, "t_landscape": t_land, "t_winding": t_wind, "t_well": t_well, "t_total": t_tot})
    ns_a = np.array([r["n"] for r in rows], float)
    tt = np.array([max(r["t_total"], 1e-6) for r in rows], float)
    Ns = np.array([r["N"] for r in rows], float)
    slope_n = float(np.polyfit(ns_a, np.log2(tt), 1)[0])
    slope_N = float(np.polyfit(np.log2(Ns), np.log2(tt), 1)[0])
    print("  FIT: log2(total_time) ~ %.3f*n + c  ==>  time ~ 2^(%.3f n)  (EXP in n if slope ~ const > 0)" % (slope_n, slope_n))
    print("  FIT: time ~ N^%.3f  (N=2^n => exponential in n for any alpha>0)" % slope_N)
    OUT["cost_scaling"] = {"rows": rows, "slope_log2time_per_n": slope_n, "alpha_in_N": slope_N}
    return rows


def main():
    t0 = time.time()
    sanity(n=10)
    magnitude_read(ns=(8, 10, 12))
    orientation_gate(ns=(8, 10))
    cost_scaling(ns=(8, 10, 12, 14))
    OUT["elapsed_s"] = time.time() - t0
    with open(os.path.join(_HERE, "hn_result.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, default=float)
    banner("DONE  (%.1fs)  wrote hn_result.json" % OUT["elapsed_s"])


if __name__ == "__main__":
    main()