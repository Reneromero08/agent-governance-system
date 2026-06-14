# -*- coding: ascii -*-
"""
koopman_sensor.py - Approach: Koopman / transfer operator of the public map f,
read as a non-Hermitian topological object (point-gap winding, directed-graph
spectrum, biorthogonal/basin structure, resolvent non-reciprocity).

CANDIDATE DIR: phase6/nonhermitian_sensor/koopman_transfer/
Builds on (does not re-litigate) the proven fold theorem. Reuses verbatim:
  fold_audit/construction.py            - the REAL Exp 50.14 construction
  fold_audit/stage3/hardened_gate.py    - random-private-fold no-smuggle gate
  fold_audit/no_smuggle_gate.py         - exact byte-equal d-invariance audit

THE OPERATOR
------------
f: Z_N -> Z_N,  f(x) = x if accept(x) else (x+1) mod N,  accept set A = {d, N-d}.
Perron-Frobenius / Koopman transfer operator P[y,x] = 1[y = f(x)]: a cyclic shift
S (pure +1 hopping) with the two columns x in A turned into self-loops. This is a
maximally non-reciprocal (Hatano-Nelson, t_left=0) directed chain with two pinning
sites. It is genuinely NON-HERMITIAN and NON-NORMAL: its point-gap spectral winding
of the homogeneous part is W=+1 (the directionality the hypothesis targets).

WHAT WE TEST
------------
(a) Does any non-Hermitian invariant of P read the orientation o = 1[d<N/2]?
    Measured as held-out AUC vs a random-private-fold null (hardened_gate).
(b) Cost scaling of building P and reading its dominant spectral feature, n=8..16.
(c) No-smuggle status (exact byte-equal d-invariance + random-fold), automatic
    because every honest feature is a pure function of public (k,b).

Controls prove the instrument is LIVE: a smuggle variant that orients the winding
by reading d (or the hidden sin channel) MUST be flagged FAIL_SMUGGLE.

ASCII only. All RNGs seeded. Claim ceiling L4-5.
"""
import os
import sys
import json
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
# phase6/nonhermitian_sensor/koopman_transfer/ -> phase6/fold_audit{,/stage3}
PHASE6 = os.path.dirname(os.path.dirname(HERE))
FOLD = os.path.join(PHASE6, "fold_audit")
STAGE3 = os.path.join(FOLD, "stage3")
for p in (FOLD, STAGE3):
    if p not in sys.path:
        sys.path.insert(0, p)

import construction as C          # the REAL construction
import no_smuggle_gate as G       # exact d-invariance audit + AUC harness
import hardened_gate as H         # random-private-fold gate


# ===========================================================================
# Operator construction (FFT-based: score at all x in O(N log N))
# ===========================================================================
def score_all_x(k, b, N):
    """score(x) = sum_i b_i cos(2 pi k_i x / N) for ALL x in Z_N, via one FFT.
    a[m] = sum_{i: k_i = m} b_i ; score(x) = Re( sum_m a_m e^{2 pi i m x / N} )
                                           = N * Re( ifft(a) )[x].
    Pure function of public (k,b,N)."""
    a = np.zeros(N, dtype=float)
    np.add.at(a, (k % N).astype(np.int64), b)
    return np.real(np.fft.ifft(a)) * N


def build_transfer(k, b, N, return_matrix=False):
    """Build f and its accepting (fixed-point) set from PUBLIC data only.
    Returns dict with f (length-N int array), fixed (sorted accept positions),
    score_all. Optionally the sparse N x N transfer operator P.
    Touches ONLY (k,b,N): no reference to d or to the range [1,N/2)."""
    s = score_all_x(k, b, N)
    thresh = len(b) / 4.0
    accept = s > thresh
    x = np.arange(N)
    f = (x + 1) % N
    f[accept] = x[accept]                  # self-loop on the accepting set
    fixed = np.flatnonzero(accept)
    out = {"f": f, "fixed": fixed, "score_all": s, "accept": accept, "N": N}
    if return_matrix:
        rows = f
        cols = x
        data = np.ones(N, dtype=float)
        P = sp.csc_matrix((data, (rows, cols)), shape=(N, N))
        out["P"] = P
    return out


def basin_sizes(fixed, N):
    """Sizes of the attracting basins of the (sorted) fixed points under the +1
    flow. Basin of fixed point p_j = arc from previous fixed point (exclusive) up
    to p_j (inclusive), going upward mod N. O(#fixed). Fold-orbit derived =>
    fold-invariant. Returns array aligned to sorted(fixed)."""
    fx = np.sort(fixed)
    m = len(fx)
    if m == 0:
        return np.array([], dtype=float)
    sizes = np.zeros(m, dtype=float)
    for j in range(m):
        prev = fx[j - 1]                   # wraps for j=0
        sizes[j] = (fx[j] - prev) % N
    sizes[sizes == 0] = N                  # single fixed point owns the whole ring
    return sizes


# ===========================================================================
# Non-Hermitian invariants of P
# ===========================================================================
def energy_winding_eigcount(P, E_ref, r, K=256):
    """Point-gap winding W(E_ref) = (1/2pi i) oint d log det(P - E) around the
    circle |E - E_ref| = r, via the Cauchy argument principle (= number of
    eigenvalues of P inside the contour). Computed by accumulating the phase of
    det(P - E) over K contour points using a sparse LU at each point. This is the
    DIRECT, faithful winding computation - each LU costs ~O(N) for this near-shift
    matrix, so the whole contour costs ~O(K N)."""
    N = P.shape[0]
    I = sp.identity(N, format="csc")
    thetas = np.linspace(0.0, 2.0 * np.pi, K, endpoint=False)
    logdets = np.empty(K, dtype=complex)
    for i, th in enumerate(thetas):
        E = E_ref + r * np.exp(1j * th)
        A = (P - E * I).tocsc()
        lu = spla.splu(A, permc_spec="NATURAL")
        diagU = lu.U.diagonal()
        diagL = lu.L.diagonal()
        # log det = sum log(diag U) + sum log(diag L); permutations are +-1 (real)
        logdets[i] = np.sum(np.log(diagU.astype(complex))) + np.sum(np.log(diagL.astype(complex)))
    # winding = total change in Im(log det) / 2pi
    ph = np.imag(logdets)
    dph = np.diff(np.concatenate([ph, ph[:1]]))
    dph = (dph + np.pi) % (2 * np.pi) - np.pi      # unwrap each step
    return float(np.sum(dph) / (2.0 * np.pi))


def bulk_bloch_winding(P=None):
    """Point-gap winding of the HOMOGENEOUS (translation-invariant) part of P:
    the pure shift S (h(theta)=e^{i theta}) winds the unit circle exactly once
    about E=0. This is the directionality the hypothesis targets; it is a constant
    +1 for the +1-increment map (and would be -1 for the decrement map f_rev that
    the fold sigma conjugates f to). Constant across instances => carries the
    direction but, being constant, zero bits about which orbit element is o."""
    return 1.0


def resolvent_nonreciprocity(P, fixed, N, E=2.0):
    """Directed (skin-effect) non-reciprocity probe. For a non-reciprocal operator
    the Green's function G=(P-E)^{-1} is asymmetric: G[y,x] != G[x,y]. We probe the
    asymmetry between the two basins by solving (P - E) g = e_src for a public
    source just-above-each fixed point and reading the directed transport. Pure
    function of P (public). Returns a small set of asymmetry scalars."""
    fx = np.sort(fixed)
    if len(fx) < 2:
        return np.array([0.0, 0.0, 0.0])
    I = sp.identity(N, format="csc")
    A = (P - E * I).tocsc()
    lu = spla.splu(A, permc_spec="NATURAL")
    p_lo, p_hi = int(fx[0]), int(fx[-1])
    # sources just downstream of each fixed point (start of the next basin)
    s1 = np.zeros(N); s1[(p_lo + 1) % N] = 1.0
    s2 = np.zeros(N); s2[(p_hi + 1) % N] = 1.0
    g1 = lu.solve(s1)
    g2 = lu.solve(s2)
    # directed transport magnitude reaching the OTHER fixed point
    t_lo_to_hi = abs(g1[p_hi])
    t_hi_to_lo = abs(g2[p_lo])
    asym = t_lo_to_hi - t_hi_to_lo
    tot = abs(g1).sum() - abs(g2).sum()
    return np.array([float(asym), float(tot), float(abs(g1[p_lo]) - abs(g2[p_hi]))])


# ===========================================================================
# THE FEATURE MAPS O(inst) -> 1D float vector  (gate contract)
# ===========================================================================
def _topfixed(fixed, score_all, N, kmax=2):
    """Return up to kmax accepting positions with the highest score, sorted
    ascending; pad with -1 if fewer. Public (positions + scores only)."""
    if len(fixed) == 0:
        return np.array([-1] * kmax, dtype=float)
    sc = score_all[fixed]
    order = fixed[np.argsort(-sc)][:kmax]
    order = np.sort(order)
    out = np.full(kmax, -1.0)
    out[:len(order)] = order
    return out


def O_koopman_winding(inst):
    """HONEST candidate. Build the Koopman transfer operator P from PUBLIC (k,b)
    and emit a battery of its non-Hermitian invariants:
      - bulk point-gap winding (the +1 directionality), constant +1
      - the two pinning-site (fixed-point) positions, sorted ascending
      - the inter-fixed-point gap and basin sizes (biorthogonal/left-eigvec scale)
      - a lower-vs-upper directed asymmetry across the public midpoint N/2
      - resolvent non-reciprocity between the two basins
    Every quantity is a deterministic function of (k,b,N); NONE reads inst['d'].
    Prediction (fold theorem): all are fold-invariant => orientation AUC = 0.5,
    exact d-invariance delta = 0 => FAIL_CHANCE."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]
    tr = build_transfer(k, b, N, return_matrix=True)
    fixed = tr["fixed"]; P = tr["P"]; s = tr["score_all"]
    topf = _topfixed(fixed, s, N, kmax=2)
    p_lo, p_hi = topf[0], topf[1]
    bs = basin_sizes(fixed, N)
    blo = float(np.min(bs)) if bs.size else 0.0
    bhi = float(np.max(bs)) if bs.size else 0.0
    # directed flux dipole across the PUBLIC midpoint N/2 (positions are public)
    mid = N / 2.0
    dipole = 0.0
    if p_lo >= 0 and p_hi >= 0:
        dipole = ((p_lo - mid) + (p_hi - mid)) / N      # even in the fold orbit
    nr = resolvent_nonreciprocity(P, fixed, N, E=2.0)
    feats = [
        bulk_bloch_winding(P),                 # +1, the directionality (constant)
        p_lo / N, p_hi / N,                    # pinning-site positions (fold orbit)
        (p_hi - p_lo) / N if (p_lo >= 0 and p_hi >= 0) else 0.0,
        blo / N, bhi / N, (bhi - blo) / N,     # basin scales (biorthogonal)
        float(len(fixed)),
        dipole,
        nr[0], nr[1], nr[2],                   # resolvent non-reciprocity
    ]
    return np.array(feats, dtype=float)


def O_koopman_winding_argprinciple(inst):
    """Same operator, but the orientation read is attempted via the genuine
    Cauchy ARGUMENT-PRINCIPLE winding numbers around E=0, E=1 and a contour that
    splits the two basins by their scores. (Expensive; small-n use.) Still a pure
    function of P => fold-invariant. Included to show the literal point-gap winding
    integral also lands at chance."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]
    tr = build_transfer(k, b, N, return_matrix=True)
    P = tr["P"]; fixed = tr["fixed"]; s = tr["score_all"]
    w0 = energy_winding_eigcount(P, 0.0, 0.5, K=96)     # encloses transient eigs
    w1 = energy_winding_eigcount(P, 1.0, 0.3, K=96)     # encloses fixed-point eigs
    topf = _topfixed(fixed, s, N, kmax=2)
    return np.array([w0 / N, w1, topf[0] / N, topf[1] / N], dtype=float)


# ---- SMUGGLE / ORACLE CONTROLS (prove the gate is live) -------------------
def O_koopman_SMUGGLE_orient(inst):
    """DESIGNATED SMUGGLE. Builds the same Koopman features but ORIENTS the
    directed winding using the sign of the hidden odd channel sin(2 pi d / N) -
    i.e. it reads inst['d']. This is exactly the move that would let a directed
    invariant pick which fixed point is the true d. The gate MUST flag FAIL_SMUGGLE
    (the orientation flips under d -> N-d at fixed public data => delta > 0)."""
    base = O_koopman_winding(inst)
    N = inst["N"]; d = inst["d"]
    orient = float(np.sign(np.sin(2 * np.pi * d / N)))   # <-- reads the secret
    return np.concatenate([base, [orient, orient * base[3]]])


def O_koopman_ORACLE_quadrature(inst):
    """ORACLE crossing (also a smuggle, by the no-d rule). Builds a NON-Hermitian
    operator using the QUADRATURE channel z_k = exp(-2 pi i k d / N) - the odd/sin
    component that is ABSENT from public data - by reading inst['d']. This DOES read
    o (phase estimation on the dyadic ladder), demonstrating the instrument can
    detect a genuine orientation read; but it is only available by smuggling d, so
    the gate MUST flag FAIL_SMUGGLE. This pins the boundary: the ONLY way to make
    the operator read o is to inject the absent odd channel."""
    N = inst["N"]; d = inst["d"]
    n = inst["n"]
    ladder = [(N >> j) for j in range(1, n + 1)]         # N/2,...,1 dyadic rungs
    feats = []
    for kk in ladder:
        z = np.exp(-2j * np.pi * kk * d / N)             # <-- reads the secret's phase
        feats.append(float(np.sin(np.angle(z))))         # odd channel = orientation
    return np.array(feats, dtype=float)


# ===========================================================================
# Cost scaling (the make-or-break)
# ===========================================================================
def cost_scaling(ns, seed=20260611, trials=5):
    """Measure the cost of PREPARING the Koopman operator and READING its dominant
    spectral feature (the fixed point / pinning site), across n. If this is poly(n)
    the invariant route would be a crossing candidate; if it is Theta(2^n) the wall
    is merely relocated to the operator. We measure:
      build_s   : time to build P (FFT score at all N points + accept + f)
      readfp_s  : time to locate the fixed point (the answer) from the operator
      winding_s : time for one argument-principle winding contour (sparse LU x K)
    H has dimension N = 2^n, so any global read is >= O(N)."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        N = 1 << n
        M = C.M_for(n)
        bt, rt, wt = [], [], []
        for _ in range(trials):
            d = C.sample_secret(N, rng)
            k, b = C.coset_samples(N, d, M, rng)
            t0 = time.perf_counter()
            tr = build_transfer(k, b, N, return_matrix=True)
            t1 = time.perf_counter()
            _ = _topfixed(tr["fixed"], tr["score_all"], N, kmax=2)   # locate fixed point
            t2 = time.perf_counter()
            bt.append(t1 - t0); rt.append(t2 - t1)
        # one winding contour timed once per n (sparse LU x K), can be slow at big n
        wt0 = time.perf_counter()
        try:
            _ = energy_winding_eigcount(tr["P"], 1.0, 0.3, K=24)
            wt.append(time.perf_counter() - wt0)
        except Exception as e:
            wt.append(float("nan"))
        rows.append({
            "n": n, "N": N, "M": M, "H_dim": N,
            "build_s": float(np.median(bt)),
            "readfp_s": float(np.median(rt)),
            "winding_contour_s": float(np.median(wt)),
        })
    return rows


def fit_loglog(ns, ys):
    """Fit log2(y) ~ slope * n + c; slope ~ 1 means cost ~ 2^n (exp in n),
    slope ~ 0 means poly/constant. Returns slope vs n."""
    ns = np.asarray(ns, dtype=float)
    ys = np.asarray(ys, dtype=float)
    good = (ys > 0) & np.isfinite(ys)
    if good.sum() < 2:
        return float("nan")
    A = np.vstack([ns[good], np.ones(good.sum())]).T
    slope, _ = np.linalg.lstsq(A, np.log2(ys[good]), rcond=None)[0]
    return float(slope)


# ===========================================================================
# Structural sanity check: spectrum + fold-invariance of P at small n
# ===========================================================================
def structural_check(n=6, seed=7):
    rng = np.random.default_rng(seed)
    N = 1 << n
    M = C.M_for(n)
    d = C.sample_secret(N, rng)
    k, b = C.coset_samples(N, d, M, rng)
    tr = build_transfer(k, b, N, return_matrix=True)
    P = tr["P"].toarray()
    eig = np.linalg.eigvals(P)
    eig_round = np.round(np.real(eig), 6) + 1j * np.round(np.imag(eig), 6)
    n_one = int(np.sum(np.abs(eig - 1.0) < 1e-6))
    n_zero = int(np.sum(np.abs(eig) < 1e-6))
    # validate FFT score against the brick's per-point score on a few x
    xs = rng.integers(0, N, size=5)
    ok_score = all(abs(tr["score_all"][int(x)] - C.score(k, b, float(x), N)) < 1e-6 for x in xs)
    # fold-invariance: same (k,b) => identical P (the bedrock). Build for folded d.
    df = C.fold(d, N)
    k2, b2 = k, b                       # public data identical; only label flips
    tr2 = build_transfer(k2, b2, N, return_matrix=True)
    P_identical = bool(np.array_equal(tr["P"].toarray(), tr2["P"].toarray()))
    # also: does the bulk winding flip if we actually reverse the map (decrement)?
    return {
        "n": n, "N": N, "d": int(d), "fold_d": int(df),
        "fixed_positions": [int(x) for x in np.sort(tr["fixed"])],
        "answer_min_d_Nmd": int(min(d % N, (N - d) % N)),
        "n_eig_at_1": n_one, "n_eig_at_0": n_zero,
        "spectrum_is_trivial_0_and_1": bool(n_one + n_zero == N),
        "fft_score_matches_brick": bool(ok_score),
        "P_identical_under_fold": P_identical,
    }


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    MASTER_SEED = 44_0611_50
    out = {"approach": "koopman_transfer", "master_seed": MASTER_SEED, "cells": []}
    print("=" * 90)
    print("KOOPMAN / TRANSFER-OPERATOR NON-HERMITIAN SENSOR  -  Exp 50.14 fold")
    print("=" * 90)

    # ---- (0) structural sanity: spectrum trivial, P fold-invariant ----
    print("\n[0] STRUCTURAL CHECK (operator spectrum + fold-invariance)")
    sc = structural_check(n=6, seed=MASTER_SEED & 0xFFFF)
    out["structural"] = sc
    for kk, vv in sc.items():
        print("    %-32s %s" % (kk, vv))

    # ---- (1) cost scaling: prepare H + read its dominant spectral feature ----
    print("\n[1] COST SCALING (make-or-break: poly(n) vs 2^n)")
    ns_cost = [8, 10, 12, 14, 16]
    cost = cost_scaling(ns_cost, seed=MASTER_SEED + 1, trials=5)
    out["cost_scaling"] = cost
    print("    n   N        H_dim    build_s     readfp_s    winding_contour_s")
    for r in cost:
        print("    %-3d %-8d %-8d %.6f    %.6f    %.6f"
              % (r["n"], r["N"], r["H_dim"], r["build_s"], r["readfp_s"], r["winding_contour_s"]))
    slope_build = fit_loglog([r["n"] for r in cost], [r["build_s"] for r in cost])
    slope_read = fit_loglog([r["n"] for r in cost], [r["readfp_s"] for r in cost])
    out["cost_slope_build_log2_per_n"] = slope_build
    out["cost_slope_readfp_log2_per_n"] = slope_read
    print("    log2(build_s) slope/n  = %.3f   (1.0 => ~2^n; 0 => poly/const)" % slope_build)
    print("    log2(readfp_s) slope/n = %.3f" % slope_read)

    # ---- (2) no-smuggle gate on the honest Koopman invariant + controls ----
    print("\n[2] HARDENED GATE: does any Koopman invariant read o? (vs random-fold null)")
    cases = [
        ("koopman_winding(PUBLIC)", O_koopman_winding, "FAIL_CHANCE"),
        ("koopman_SMUGGLE_orient(reads d)", O_koopman_SMUGGLE_orient, "FAIL_SMUGGLE"),
        ("koopman_ORACLE_quadrature(reads d)", O_koopman_ORACLE_quadrature, "FAIL_SMUGGLE"),
    ]
    gate_ns = [8, 10, 12, 14]
    N_INST = 300
    N_SHUF = 20
    for n in gate_ns:
        print("\n  ### n=%d (N=%d) ###" % (n, 1 << n))
        for ci, (name, O, expected) in enumerate(cases):
            seed = (MASTER_SEED + 1009 * n + 31 * ci) & 0x7FFFFFFF
            tic = time.time()
            res = H.hardened_gate(O, n, n_instances=N_INST, seed=seed, n_shuffles=N_SHUF)
            dt = time.time() - tic
            ok = (res["verdict"] == expected)
            print("    [%s] %-36s verdict=%-13s (exp %-13s)"
                  % ("OK " if ok else "!! ", name, res["verdict"], expected))
            print("          orient_auc=%.3f (null95=%.3f)  rfold_auc=%.3f (null95=%.3f)"
                  "  delta=%.3g  reason=%s  [%.1fs]"
                  % (res["auc"], res["shuffle_null_95"], res["random_fold_auc"],
                     res["random_fold_null_95"], res["max_fold_delta"],
                     res["smuggle_reason"], dt))
            out["cells"].append({
                "name": name, "n": n, "seed": int(seed), "expected": expected,
                "verdict": res["verdict"], "matches": bool(ok),
                "orientation_auc": res["auc"], "orientation_null95": res["shuffle_null_95"],
                "random_fold_auc": res["random_fold_auc"],
                "random_fold_null95": res["random_fold_null_95"],
                "invariance_delta": res["max_fold_delta"],
                "smuggle_reason": res["smuggle_reason"], "elapsed_s": dt,
            })

    # ---- (3) literal argument-principle winding at small n (sanity) ----
    print("\n[3] LITERAL argument-principle winding gate (small n)")
    for n in (8, 10):
        seed = (MASTER_SEED + 777 + n) & 0x7FFFFFFF
        res = H.hardened_gate(O_koopman_winding_argprinciple, n, n_instances=150,
                              seed=seed, n_shuffles=15)
        print("    n=%d  verdict=%s  orient_auc=%.3f (null95=%.3f)  delta=%.3g"
              % (n, res["verdict"], res["auc"], res["shuffle_null_95"], res["max_fold_delta"]))
        out["cells"].append({
            "name": "koopman_winding_argprinciple", "n": n, "seed": int(seed),
            "expected": "FAIL_CHANCE", "verdict": res["verdict"],
            "matches": bool(res["verdict"] == "FAIL_CHANCE"),
            "orientation_auc": res["auc"], "orientation_null95": res["shuffle_null_95"],
            "random_fold_auc": res["random_fold_auc"],
            "invariance_delta": res["max_fold_delta"],
        })

    # ---- summary ----
    honest = [c for c in out["cells"] if "PUBLIC" in c["name"] or "argprinciple" in c["name"]]
    smug = [c for c in out["cells"] if "SMUGGLE" in c["name"] or "ORACLE" in c["name"]]
    out["summary"] = {
        "honest_public_aucs": [round(c["orientation_auc"], 4) for c in honest],
        "honest_all_fail_chance": all(c["verdict"] == "FAIL_CHANCE" for c in honest),
        "smuggle_all_caught": all(c["verdict"] == "FAIL_SMUGGLE" for c in smug),
        "cost_verdict": "exp" if (slope_build > 0.6 or slope_read > 0.6) else "poly_or_unknown",
        "H_dimension": "2^n (N)",
    }
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("  honest public Koopman AUCs (o):", out["summary"]["honest_public_aucs"])
    print("  honest all FAIL_CHANCE (cannot read o): ", out["summary"]["honest_all_fail_chance"])
    print("  smuggle/oracle all caught FAIL_SMUGGLE:  ", out["summary"]["smuggle_all_caught"])
    print("  cost verdict: %s  (build slope/n=%.2f, readfp slope/n=%.2f; H dim = 2^n)"
          % (out["summary"]["cost_verdict"], slope_build, slope_read))
    print("=" * 90)

    with open(os.path.join(HERE, "koopman_result.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("wrote koopman_result.json")


if __name__ == "__main__":
    main()
