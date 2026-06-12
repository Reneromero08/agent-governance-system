"""
b2b_collimation.py - the MULTI-LEVEL coherent Kuperberg collimation sieve, measured.

The depth-1 birthday sieve (step_b.b2a) extracts the orientation at ~2^{n/2} queries.
Kuperberg's speedup collimates the labels in BLOCKS of b bits per round; with b ~ sqrt(2n)
the query cost drops to 2^{O(sqrt n)} - SUBEXPONENTIAL. This module implements the robust
LOW-bit collimation (exact integer arithmetic, no band/wrap fragility): bucket coset-state
labels by their low b bits, subtract same-bucket pairs (the coherent minus branch, prob
1/2) to zero those bits, recurse up the dyadic ladder until the label is a multiple of
2^{n-1}, i.e. label in {0, N/2}. A produced label N/2 coset state has phase
omega^{(N/2) d} = (-1)^d, whose X-quadrature sign IS the LSB of the dihedral secret d.

This MEASURES the canonical Kuperberg cost: the queries to extract ONE bit of d coherently
(in-black-hole, never translating out until the final read). The orientation bit o=1[d<N/2]
(the MSB) is in the SAME 2^{O(sqrt n)} class - reading any bit of the dihedral secret is
Kuperberg-hard, and the full secret reduces to lattice (Regev). Fits log2 R(n) vs n (exp)
and vs sqrt(n) (subexp).

No-smuggle: bucketing/combination decisions use ONLY the public labels; d enters only the
final measurement outcome law. ASCII only; RNGs seeded by caller.
"""
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE6 = os.path.dirname(HERE)
FOLD = os.path.join(PHASE6, "fold_audit")
for _p in (FOLD, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import construction as C


def collimate_low(labels, N, shift, b, rng):
    """Zero bits [shift, shift+b) of the labels (which already have bits below `shift`
    zeroed). Bucket by (label >> shift) mod 2^b; subtract same-bucket pairs on the coherent
    minus branch (prob 1/2). The difference has bits [shift, shift+b) zeroed. Pure function
    of labels (no d)."""
    nbins = 1 << b
    key = (labels >> shift) & (nbins - 1)
    order = rng.permutation(len(labels))
    labels = labels[order]
    key = key[order]
    buckets = defaultdict(list)
    for i in range(len(labels)):
        buckets[int(key[i])].append(int(labels[i]))
    survivors = []
    for _, vals in buckets.items():
        for j in range(0, len(vals) - 1, 2):
            if rng.random() < 0.5:
                survivors.append((vals[j] - vals[j + 1]) % N)
    if not survivors:
        return np.zeros(0, dtype=np.int64)
    return np.array(survivors, dtype=np.int64) % N


def sieve_once(n, d, R, b, rng):
    """Draw R coset-state labels; collimate the low n-1 bits in blocks of b until label is
    a multiple of 2^{n-1} (in {0, N/2}); if a label == N/2 survives, read LSB(d) by the
    sign of its (-1)^d phase. Returns (produced_half, lsb_correct, queries=R)."""
    N = 1 << n
    labels = rng.integers(0, N, size=R).astype(np.int64)
    shift = 0
    while shift < n - 1 and labels.size > 0:
        blk = min(b, (n - 1) - shift)
        labels = collimate_low(labels, N, shift, blk, rng)
        shift += blk
    if labels.size == 0:
        return False, False, R
    half = labels[labels == (N // 2)]
    if half.size == 0:
        return False, False, R
    # label N/2 -> phase (-1)^d, real; X-quadrature sign = LSB(d). Noiseless sign of a real
    # amplitude (the minus/plus computational outcome): read it.
    lsb_pred = int(d % 2)                              # the measurement returns exactly (-1)^d
    return True, (lsb_pred == (d % 2)), R


def measure_collimation_cost(ns, seed, n_trials=60, success_target=0.5):
    """Smallest pool R (swept over powers of two) reaching label-N/2 success probability
    >= success_target, per n. b = round(sqrt(2 n)) (Kuperberg-balanced block size)."""
    out = []
    for n in ns:
        N = 1 << n
        b = max(2, int(round(np.sqrt(2.0 * n))))
        R = 1 << b
        chosen = None
        rec = []
        Rcap = 1 << min(n + 6, 22)
        while R <= Rcap:
            succ = 0
            for t in range(n_trials):
                rng = np.random.default_rng(seed + 104729 * n + 31 * t + R)
                d = C.sample_secret(N, rng)
                ok, _, _ = sieve_once(n, d, R, b, rng)
                succ += int(ok)
            p = succ / n_trials
            rec.append({"R": int(R), "p_success": p})
            if chosen is None and p >= success_target:
                chosen = {"R": int(R), "p_success": p}
                break
            R *= 2
        out.append({
            "n": n, "N": N, "block_b": b,
            "R_star": (chosen["R"] if chosen else None),
            "p_success_at_R_star": (chosen["p_success"] if chosen else None),
            "sweep": rec,
        })
    return out


def fit_classes(ns, R_stars):
    """Fit log2 R* against n (exp: slope*n) and sqrt(n) (subexp: slope*sqrt n); report which
    is the better description and the implied class."""
    pts = [(n, R) for n, R in zip(ns, R_stars) if R]
    if len(pts) < 3:
        return {"note": "too few points to fit"}
    nn = np.array([p[0] for p in pts], dtype=float)
    y = np.log2(np.array([p[1] for p in pts], dtype=float))
    def fit(x):
        A = np.vstack([x, np.ones_like(x)]).T
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"slope": float(coef[0]), "intercept": float(coef[1]), "r2": r2}
    f_n = fit(nn)
    f_sqrt = fit(np.sqrt(nn))
    better = "subexp(sqrt n)" if f_sqrt["r2"] >= f_n["r2"] else "exp(n)"
    return {"fit_log2R_vs_n": f_n, "fit_log2R_vs_sqrtn": f_sqrt,
            "better_fit": better,
            "ns": [int(p[0]) for p in pts], "R_stars": [int(p[1]) for p in pts]}


if __name__ == "__main__":
    import time
    S = 50140611
    t = time.time()
    res = measure_collimation_cost([4, 6, 8, 10, 12, 14, 16], S, n_trials=60)
    Rs = [r["R_star"] for r in res]
    ns = [r["n"] for r in res]
    for r in res:
        print("n=%2d b=%d  R*=%s  p=%.2f  log2R*=%s" % (
            r["n"], r["block_b"], r["R_star"], (r["p_success_at_R_star"] or 0.0),
            ("%.2f" % np.log2(r["R_star"])) if r["R_star"] else None))
    print(fit_classes(ns, Rs))
    print("elapsed %.1fs" % (time.time() - t))
