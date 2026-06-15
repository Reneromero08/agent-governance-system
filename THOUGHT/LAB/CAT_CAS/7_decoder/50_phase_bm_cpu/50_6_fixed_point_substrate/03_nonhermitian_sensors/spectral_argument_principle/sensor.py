"""
sensor.py - APPROACH: Cauchy argument principle on a public contour.

Non-Hermitian / topological readout attempt on the Exp 50.14 public fixed-point
map. The KEY HYPOTHESIS under test: the directionality of the map f (the +1
increment, a directed non-reciprocal flow) is exactly what non-Hermitian
operators encode, so the missing orientation bit o = 1[d < N/2] (which the cosine
fold destroys) might be readable as a non-Hermitian topological invariant - a
point-gap winding read by the Cauchy argument principle on a contour - rather than
as a function of the even cosine magnitudes.

We construct the public analytic function Phi(z) = score'(z) (the derivative of the
matched-filter score, analytically continued to complex z). Phi is entire; its real
zeros sit at the score peaks {d, N-d} and at the symmetry points {0, N/2}. The
argument-principle winding W(C) = (1/2 pi i) oint_C Phi'/Phi dz counts zeros of Phi
inside C. We then ask, sharply and with measurement:

  (1) READS o?   Does any PUBLIC (d-free) contour family give a winding feature that
                 predicts o above the random-private-fold null? (hardened_gate.)
  (2) COST?      THE MAKE-OR-BREAK. How many zeros does the public Phi have, and how
                 many quadrature points does a CONVERGED winding around [1, N/2)
                 need, as a function of N = 2^n? poly(n) would be a crossing; ~N is
                 the same forward O(N) wall relocated to the contour integral.
  (3) SMUGGLE?   The public winding is byte-identical under d<->N-d (delta=0). The
                 ONLY o-dependent contour is one PLACED using the true d (caught by
                 the exact d-invariance audit as FAIL_SMUGGLE).

Operator-level confirmation: the directed transition operator T of f (the literal
non-Hermitian object whose +1 hopping the hypothesis invokes) is built from PUBLIC
accept and shown fold-invariant - its fixed points are the symmetric pair {a, N-a}
and its two basins have sizes {2a, N-2a} that depend ONLY on a = min(d, N-d), never
on o. So every topological invariant of T (point-gap winding, skin position,
spectrum) is o-blind by construction, and building T is already O(N*M) = exp(n).

ASCII only. All RNGs seeded; seeds recorded. Reuses construction.py /
no_smuggle_gate.py / hardened_gate.py verbatim. Claim ceiling L4-5.
"""
import os
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FOLD_AUDIT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "02_fold_audit"))
STAGE3 = os.path.join(FOLD_AUDIT, "stage3")
for _p in (FOLD_AUDIT, STAGE3):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import construction as C            # the REAL Exp 50.14 construction (verbatim)
import no_smuggle_gate as G         # AUC harness + exact d-invariance audit
import hardened_gate as H           # random-private-fold smuggle detector


# ---------------------------------------------------------------------------
# The public analytic function Phi(z) = score'(z), continued to complex z.
# score(z)  = sum_i b_i cos(2 pi k_i z / N)              (entire in z)
# Phi(z)    = score'(z) = -(2 pi/N) sum_i b_i k_i sin(2 pi k_i z / N)
# We drop the constant -(2 pi/N) (irrelevant to zeros / winding).
# Phi is conjugation-symmetric (real coefficients b_i k_i): Phi(conj z)=conj Phi(z).
# Real zeros at the score peaks d, N-d and at 0, N/2. NO d is ever read here.
# ---------------------------------------------------------------------------
def phi(zz, k, b, N):
    """Vectorized Phi over an array of complex contour points zz. O(len(zz)*M)."""
    zz = np.asarray(zz, dtype=complex)
    ang = (2.0 * np.pi / N) * np.outer(zz, k.astype(float))   # (P, M)
    return (np.sin(ang) * (b * k.astype(float))).sum(axis=1)  # (P,)


def rect_contour(a, b, h, q_edge):
    """Counterclockwise rectangle in the complex z-plane enclosing the real
    interval [a, b] with half-height h. q_edge points per edge. a, b chosen as
    half-integers by callers so the contour never lands on an integer zero."""
    t = np.linspace(0.0, 1.0, q_edge, endpoint=False)
    bottom = (a + (b - a) * t) - 1j * h
    right = b + (-h + 2.0 * h * t) * 1j
    top = (b - (b - a) * t) + 1j * h
    left = a + (h - 2.0 * h * t) * 1j
    return np.concatenate([bottom, right, top, left])


def winding(zz, k, b, N):
    """Argument-principle winding = (1/2pi) * total change of arg(Phi) around the
    closed contour zz. Counts zeros minus poles inside; Phi is entire so this is the
    enclosed zero count. Robust angle-unwrap form (needs only Phi, not Phi')."""
    vals = phi(zz, k, b, N)
    vals = np.append(vals, vals[0])                 # close the loop
    inc = np.angle(vals[1:] / vals[:-1])            # principal-value increments
    return float(np.sum(inc) / (2.0 * np.pi))


# ---------------------------------------------------------------------------
# (1) candidate operations O(inst) for the no-smuggle / hardened gate
# ---------------------------------------------------------------------------
def O_winding_public(inst):
    """PUBLIC argument-principle features: windings around a fixed family of d-free
    contours. Contour edges are fixed fractions of N (the public modulus) only - the
    decisive one is the rectangle around the WHOLE range [1, N/2) that the prompt
    asks about, plus sub-interval rectangles. Reads ONLY k, b, N => fold-invariant
    => the gate's exact d-invariance audit must report delta = 0. Predicted verdict:
    FAIL_CHANCE (the winding of a conjugation-symmetric function carries no o)."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]
    h = 0.5
    qe = 40
    feats = []
    edges = np.linspace(0.5, N / 2.0 - 0.5, 4)      # 3 sub-intervals of [1, N/2)
    for j in range(len(edges) - 1):
        feats.append(winding(rect_contour(edges[j], edges[j + 1], h, qe), k, b, N))
    # the decisive [1, N/2) contour (the "range restriction as a public operation")
    feats.append(winding(rect_contour(0.5, N / 2.0 - 0.5, h, qe), k, b, N))
    return np.array(feats, dtype=float)


def O_winding_smuggle(inst):
    """DESIGNATED SMUGGLE. The only way to make a winding feature o-dependent is to
    PLACE the contour using the true d (or, equivalently here, to read the half d
    lives on). This appends 1[d < N/2] = o read straight from the hidden d. The gate
    MUST flag FAIL_SMUGGLE: under d<->N-d at fixed public data the second entry flips
    => max_fold_delta > 0. It marks precisely where the smuggle enters: the contour /
    half selection references inst['d']."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]; d = inst["d"]
    base = winding(rect_contour(0.5, N / 2.0 - 0.5, 0.5, 40), k, b, N)
    return np.array([base, float(int(d) < (N / 2.0))], dtype=float)   # <- reads d


# ---------------------------------------------------------------------------
# (2) COST scaling - the make-or-break
# ---------------------------------------------------------------------------
def dense_zero_count(k, b, N, oversample=8):
    """Count real zeros (sign changes) of Phi on [0, N) and on the range
    (0.5, N/2-0.5) by dense evaluation. A reference for how much structure the
    contour must resolve. Cost O(oversample*N*M)."""
    xs = np.linspace(0.0, N, oversample * N, endpoint=False)
    vals = np.real(phi(xs.astype(complex), k, b, N))
    s = np.sign(vals); s[s == 0] = 1.0
    total = int(np.sum(s[1:] != s[:-1]))
    in_range = (xs > 0.5) & (xs < (N / 2.0 - 0.5))
    xr = xs[in_range]
    sr = np.sign(np.real(phi(xr.astype(complex), k, b, N))); sr[sr == 0] = 1.0
    range_zeros = int(np.sum(sr[1:] != sr[:-1]))
    return total, range_zeros


def cost_scaling(k, b, N, h=0.5):
    """Measure the cost of a CONVERGED argument-principle winding around the public
    range [1, N/2). Method:
      - W_ref  : winding at a very dense contour (q_edge = max(2048, 6N)) -> the true
                 enclosed-zero count the integral must return.
      - q_star : smallest q_edge on a geometric grid at which round(winding) first
                 equals round(W_ref) AND stays there at the next (denser) grid point.
                 This is the quadrature the argument principle must pay to be correct.
      - W_poly : winding at a POLY budget q_edge = 32*log2(N) - shows the poly-budget
                 winding and its error vs W_ref (does poly quadrature suffice?).
    Returns the measured numbers. poly(n) q_star => crossing; q_star ~ N => exp wall."""
    a, bb = 0.5, N / 2.0 - 0.5
    n = int(round(np.log2(N)))
    q_ref = int(max(2048, 6 * N))
    W_ref = winding(rect_contour(a, bb, h, q_ref), k, b, N)
    target = round(W_ref)

    # poly-budget winding
    q_poly = 32 * n
    W_poly = winding(rect_contour(a, bb, h, q_poly), k, b, N)

    # geometric sweep for q_star
    grid = []
    q = 8
    while q <= 8 * N:
        grid.append(q); q *= 2
    q_star = None
    t_star = None
    prev_ok = False
    for qe in grid:
        t0 = time.perf_counter()
        w = winding(rect_contour(a, bb, h, qe), k, b, N)
        dt = time.perf_counter() - t0
        ok = (round(w) == target)
        if ok and prev_ok and q_star is None:
            q_star = qe; t_star = dt
            break
        prev_ok = ok
    if q_star is None:
        q_star = grid[-1]; t_star = float("nan")
    return {"W_ref": W_ref, "target_zeros": int(target),
            "q_ref": q_ref, "q_star": int(q_star),
            "q_star_over_N": q_star / float(N),
            "q_poly": int(q_poly), "W_poly": W_poly,
            "poly_abs_err": abs(W_poly - W_ref),
            "wall_s_at_qstar": t_star}


# ---------------------------------------------------------------------------
# Operator-level confirmation: the directed transition operator T of f.
# ---------------------------------------------------------------------------
def walk_to_fixed_point(x0, verify, N):
    """Follow f forward from x0 until it lands on a fixed point. O(N) worst case."""
    x = x0
    for _ in range(N + 1):
        if verify(x):
            return x
        x = (x + 1) % N
    return None


def directed_operator_fold_facts(n, seed):
    """Facts about the directed operator T of f from PUBLIC accept only: fixed points
    and the two basin sizes. Demonstrates T depends only on the fold-invariant accept
    SET {d, N-d}, hence on a = min(d, N-d), never on o. For small n we also
    diagonalize T (dense) to exhibit the fold-invariant point-gap spectrum. Building
    T / the basins is O(N*M) = exp(n)."""
    N = 1 << n
    rng = np.random.default_rng(seed)
    d = C.sample_secret(N, rng)
    M = C.M_for(n)
    k, b = C.coset_samples(N, d, M, rng)
    verify, _ = C.make_verify(k, b, N)

    fps = [x for x in range(N) if verify(x)]          # O(N*M)
    a = min(fps) if fps else None
    basin = {fp: 0 for fp in fps}
    for x0 in range(N):
        fp = walk_to_fixed_point(x0, verify, N)
        if fp is not None:
            basin[fp] = basin.get(fp, 0) + 1
    out = {"n": n, "N": N, "M": M, "d": int(d),
           "answer_a": int(a) if a is not None else None,
           "fixed_points": [int(x) for x in fps],
           "basin_sizes": {int(fp): int(sz) for fp, sz in basin.items()}}
    if a is not None and len(fps) == 2:
        out["predicted_fixed_points"] = sorted([int(a), int((N - a) % N)])
        out["predicted_basins"] = sorted([int(2 * a), int(N - 2 * a)])
        out["basins_match_prediction"] = (
            sorted(basin.values()) == sorted([2 * a, N - 2 * a]))
    if n <= 10:
        T = np.zeros((N, N))
        for x in range(N):
            fx = x if verify(x) else (x + 1) % N
            T[fx, x] = 1.0
        ev = np.linalg.eigvals(T)
        out["spectrum_abs_sorted"] = np.round(np.sort(np.abs(ev))[::-1][:6], 6).tolist()
        out["n_eigs_near_1"] = int(np.sum(np.abs(ev - 1.0) < 1e-6))
    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    MASTER_SEED = 44060611
    NS = [8, 10, 12, 14]
    report = {"master_seed": MASTER_SEED, "approach": "cauchy_argument_principle",
              "n_tested": NS}
    log = []

    def pr(s=""):
        print(s)
        log.append(str(s))

    pr("=" * 92)
    pr("NON-HERMITIAN SENSOR / Cauchy argument principle on a public contour")
    pr("Target: Exp 50.14 public fixed-point map (verbatim). master_seed=%d" % MASTER_SEED)
    pr("=" * 92)

    # ---- (1) READS o? hardened gate on the public winding (+ smuggle reference) ----
    pr("\n[1] DOES THE ARGUMENT-PRINCIPLE WINDING READ o?  (hardened random-fold gate)")
    pr("    candidate           n    verdict        orient_auc  rand_fold_auc(null95)  delta")
    gate_cells = []
    for n in NS:
        seed = (MASTER_SEED + 7919 * n) & 0x7FFFFFFF
        t0 = time.perf_counter()
        res = H.hardened_gate(O_winding_public, n, n_instances=150, seed=seed,
                              n_shuffles=20)
        dt = time.perf_counter() - t0
        pr("    %-18s  %-3d  %-13s  %.3f       %.3f (%.3f)        %.2g  [%.1fs]"
           % ("winding_PUBLIC", n, res["verdict"], res["auc"],
              res["random_fold_auc"], res["random_fold_null_95"],
              res["max_fold_delta"], dt))
        gate_cells.append({"candidate": "winding_PUBLIC", "n": n,
                           "verdict": res["verdict"], "orientation_auc": res["auc"],
                           "orientation_null95": res["shuffle_null_95"],
                           "random_fold_auc": res["random_fold_auc"],
                           "random_fold_null95": res["random_fold_null_95"],
                           "max_fold_delta": res["max_fold_delta"], "seed": int(seed)})
    for n in (8, 10):
        seed = (MASTER_SEED + 104729 * n) & 0x7FFFFFFF
        res = H.hardened_gate(O_winding_smuggle, n, n_instances=150, seed=seed,
                              n_shuffles=20)
        pr("    %-18s  %-3d  %-13s  %.3f       %.3f (%.3f)        %.2g"
           % ("winding_SMUGGLE", n, res["verdict"], res["auc"],
              res["random_fold_auc"], res["random_fold_null_95"],
              res["max_fold_delta"]))
        gate_cells.append({"candidate": "winding_SMUGGLE", "n": n,
                           "verdict": res["verdict"], "orientation_auc": res["auc"],
                           "random_fold_auc": res["random_fold_auc"],
                           "max_fold_delta": res["max_fold_delta"], "seed": int(seed)})
    report["gate"] = gate_cells

    # ---- (2) COST scaling - the make-or-break ----
    pr("\n[2] COST SCALING of a CONVERGED winding (make-or-break: poly(n) vs ~N)")
    pr("    n    N        M     zeros[0,N)  W_ref   q_star  q*/N    poly_q  poly_err")
    cost_cells = []
    for n in NS:
        seed = (MASTER_SEED + 31337 * n) & 0x7FFFFFFF
        rng = np.random.default_rng(seed)
        N = 1 << n
        d = C.sample_secret(N, rng)
        M = C.M_for(n)
        k, b = C.coset_samples(N, d, M, rng)
        total_z, range_z = dense_zero_count(k, b, N, oversample=8)
        cs = cost_scaling(k, b, N)
        pr("    %-3d  %-7d  %-4d  %-10d  %-6.0f  %-6d  %-6.3f  %-6d  %.0f"
           % (n, N, M, total_z, cs["W_ref"], cs["q_star"], cs["q_star_over_N"],
              cs["q_poly"], cs["poly_abs_err"]))
        cost_cells.append({"n": n, "N": N, "M": M, "zeros_full": total_z,
                           "zeros_range": range_z, "W_ref": cs["W_ref"],
                           "target_zeros": cs["target_zeros"],
                           "q_star": cs["q_star"], "q_star_over_N": cs["q_star_over_N"],
                           "q_poly": cs["q_poly"], "W_poly": cs["W_poly"],
                           "poly_abs_err": cs["poly_abs_err"],
                           "seed": int(seed), "d": int(d)})
    report["cost"] = cost_cells

    # ---- (3) operator-level confirmation: directed T is fold-invariant ----
    pr("\n[3] DIRECTED OPERATOR T of f: fold-invariant by construction (basins read a, not o)")
    pr("    n    N      d      a=min   fixed_points        basin_sizes       == {2a, N-2a}?")
    op_cells = []
    for n in (6, 8, 10):
        seed = (MASTER_SEED + 6151 * n) & 0x7FFFFFFF
        facts = directed_operator_fold_facts(n, seed)
        pr("    %-3d  %-5d  %-5d  %-6s  %-18s  %-16s  %s"
           % (n, facts["N"], facts["d"], str(facts.get("answer_a")),
              str(facts["fixed_points"]), str(list(facts["basin_sizes"].values())),
              str(facts.get("basins_match_prediction"))))
        op_cells.append(facts)
    report["operator"] = op_cells

    # ---- honest readout ----
    pr("\n" + "=" * 92)
    pr("WHAT THE DATA SAYS")
    public_all_chance = all(c["verdict"] == "FAIL_CHANCE"
                            for c in gate_cells if c["candidate"] == "winding_PUBLIC")
    smuggle_caught = all(c["verdict"] == "FAIL_SMUGGLE"
                         for c in gate_cells if c["candidate"] == "winding_SMUGGLE")
    qfrac = [c["q_star_over_N"] for c in cost_cells]
    # cost is ~ N if q_star/N stays bounded away from 0 (constant ratio = linear in N)
    cost_is_linear = float(np.min(qfrac)) > 0.05
    polyerr_grows = (cost_cells[-1]["poly_abs_err"] > cost_cells[0]["poly_abs_err"])
    pr("  (1) public winding reads o:  %s  (all FAIL_CHANCE => no public contour reads o)"
       % (not public_all_chance))
    pr("  (2) smuggle contour caught:  %s  (placing the contour by d => FAIL_SMUGGLE)"
       % smuggle_caught)
    pr("  (3) q_star/N = %s  (constant ratio => q_star ~ N => exp(n)): %s"
       % ([round(x, 3) for x in qfrac], cost_is_linear))
    pr("  (3b) poly-budget winding error grows with N: %s  (poly quadrature cannot converge)"
       % polyerr_grows)
    pr("  zeros_full ~ c*N: %s" % ([c["zeros_full"] for c in cost_cells]))
    pr("=" * 92)

    if public_all_chance and cost_is_linear:
        verdict = "WALL_HOLDS_FOR_PUBLIC_CONTOUR__COST_RELOCATED_TO_THE_INTEGRAL"
    elif not public_all_chance and not cost_is_linear:
        verdict = "APPARENT_CROSSING__DEMAND_REAUDIT"
    else:
        verdict = "MIXED__SEE_CELLS"
    pr("VERDICT: %s" % verdict)
    report["verdict"] = verdict
    report["public_winding_reads_o"] = bool(not public_all_chance)
    report["smuggle_caught"] = bool(smuggle_caught)
    report["cost_is_linear_in_N"] = bool(cost_is_linear)
    report["poly_budget_error_grows"] = bool(polyerr_grows)

    with open(os.path.join(HERE, "sensor_result.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    with open(os.path.join(HERE, "output_sensor.txt"), "w") as fh:
        fh.write("\n".join(log))
    pr("\nwrote sensor_result.json + output_sensor.txt")


if __name__ == "__main__":
    main()
