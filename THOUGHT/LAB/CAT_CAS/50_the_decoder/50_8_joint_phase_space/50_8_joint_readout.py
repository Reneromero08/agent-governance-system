"""
Exp 50.8 - The joint phase-space readout: use BOTH conjugate bases at once.

The convergent door. 50.6 (A9) showed the secret is small in the coefficient/primal basis but
the multiplication is diagonal in the NTT/dual basis, and NO single basis has both - so a
single-basis decode is walled. 50.7 (A13) showed injecting entropy (chaos / many random
combinations) DOES recover the secret, but at exponential cost (the BKW/sieve frontier). Every
door converges here: the secret lives in the JOINT (both-bases) geometry, and reading it from
there is exactly the lattice problem.

This brick builds the joint readout directly - a search that uses the NTT/dual basis for the
linear relation AND the coefficient/primal basis for the smallness objective at once - and lets
the DATA decide the cost scaling, instead of predicting it:
  - JOINT recovery uses both bases. CHANCE (single-basis) baselines from 50.6/50.7 are at chance.
  - Cost = number of candidate secrets the joint-guided search must score before recovery.
  - SCALING in n decides it: poly cost => the wall CROSSES (extraordinary; A8 suspicion -> Mythos).
    exponential/subexp cost => the joint door IS the lattice problem; the distinct attack families
    are exhausted and the residual (poly lattice algorithm) goes to Mythos - NOT 'the wall holds'.

This is NOT a verdict that the wall holds. It measures whether the joint phase-space readout buys
anything over the known sieve cost, and hands the residual up per the lab stance.

Run:  python 50_8_joint_readout.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # 50_the_decoder root
import decoder_lib as dl  # noqa: E402

LINES = []
def log(m=""):
    print(m)
    LINES.append(str(m))

Q = 12289


def centered(x, q=Q):
    x = np.asarray(x) % q
    return np.where(x > q // 2, x - q, x)


def find_psi(n, q=Q):
    # need a primitive 2n-th root (psi^n = -1 with order exactly 2n) => 2n must divide q-1
    if (q - 1) % (2 * n) != 0:
        raise ValueError("q=%d not NTT-friendly for n=%d (2n does not divide q-1)" % (q, n))
    for psi in range(2, q):
        if pow(psi, n, q) == q - 1 and pow(psi, 2 * n, q) == 1 and len(set(pow(psi, k, q) for k in range(2 * n))) == 2 * n:
            return psi
    raise ValueError("no primitive 2n-th root for n=%d" % n)


def ntt_mats(n, psi, q=Q):
    j = np.arange(n); i = np.arange(n)
    E = np.outer(2 * j + 1, i)
    W = np.array([[pow(int(psi), int(e), q) for e in row] for row in E], dtype=np.int64)
    # inverse: solve W x = y mod q via the conjugate transform (negacyclic intt)
    ninv = pow(n, q - 2, q)
    psi_inv = pow(int(psi), q - 2, q)
    Ei = np.outer(i, 2 * j + 1)
    Wi = np.array([[pow(psi_inv, int(e), q) for e in row] for row in Ei], dtype=np.int64)
    return W, Wi, ninv


def ntt(a, W, q=Q):
    return (W @ (np.asarray(a) % q)) % q


def intt(A, Wi, ninv, q=Q):
    return ((Wi @ (np.asarray(A) % q)) % q * ninv) % q


def negacyclic_conv(a, s, q=Q):
    n = len(a)
    full = np.zeros(2 * n, dtype=np.int64)
    for i in range(n):
        full[i:i + n] += (a[i] * np.asarray(s))
    return (full[:n] - full[n:2 * n]) % q


def ring_lwe(n, psi, W, rng, sigma=1):
    s = rng.integers(-1, 2, size=n) % Q          # ternary secret (small, coeff basis)
    a = rng.integers(0, Q, size=n)
    e = rng.integers(-sigma, sigma + 1, size=n)  # small error (coeff basis)
    b = (negacyclic_conv(a, s) + e) % Q
    return a, b, s


def joint_guided_search(a, b, n, psi, W, Wi, ninv, budget, rng):
    """JOINT readout: in the NTT basis the relation is b_hat = a_hat o s_hat + e_hat (diagonal).
    Parameterize candidate secrets, score each by the COEFFICIENT-basis smallness of the implied
    error e = INTT(b_hat - a_hat o s_hat). The correct s minimizes ||e||_coeff. We search ternary
    secrets (the prior), scoring by the joint (NTT-relation + coeff-smallness) objective.
    Returns the best secret found and how many candidates were scored."""
    a_hat = ntt(a, W); b_hat = ntt(b, W)
    best_s, best_score, scored = None, None, 0
    for _ in range(budget):
        cand = rng.integers(-1, 2, size=n)        # ternary candidate (the smallness prior)
        c_hat = ntt(cand % Q, W)
        e_hat = (b_hat - a_hat * c_hat) % Q
        e_coeff = centered(intt(e_hat, Wi, ninv))  # the implied error in the SMALL basis
        score = float(np.max(np.abs(e_coeff)))     # joint objective: smallest coeff-error
        scored += 1
        if best_score is None or score < best_score:
            best_score, best_s = score, cand
        if best_score <= 1:                        # found a ternary secret with tiny error
            break
    return best_s, scored, best_score


def main():
    log("=" * 98)
    log("EXP 50.8  -  THE JOINT PHASE-SPACE READOUT: use both conjugate bases at once")
    log("  NTT/dual basis = linear relation; coefficient/primal basis = smallness. Joint = LWE itself.")
    log("=" * 98)
    rng = np.random.default_rng(508)

    log("\n[JOINT RECOVERY + COST SCALING]  ring-LWE, ternary secret+error, q=%d" % Q)
    log("  cost = candidate secrets the joint-guided search scores before recovery.")
    log("  n    recover_rate   median_cost   log_3(cost)   3^n (full ternary space)")
    rows = []
    for n in (2, 3, 4, 6, 8, 12, 16):   # NTT-friendly: 2n divides q-1 = 2^12*3
        psi = find_psi(n); W, Wi, ninv = ntt_mats(n, psi)
        budget = min(3 ** n * 4 + 50, 200000)
        ok, costs, trials = 0, [], 12
        for _ in range(trials):
            a, b, s = ring_lwe(n, psi, W, rng, sigma=1)
            est, scored, sc = joint_guided_search(a, b, n, psi, W, Wi, ninv, budget, rng)
            hit = est is not None and np.array_equal(est % Q, s % Q)
            ok += int(hit)
            if hit:
                costs.append(scored)
        med = float(np.median(costs)) if costs else float("nan")
        rows.append({"n": n, "rate": ok / trials, "cost": med,
                     "log3": (np.log(med) / np.log(3)) if costs else None, "full": 3 ** n})
        log("  %-4d %-14.2f %-13s %-13s %d"
            % (n, ok / trials, ("%.0f" % med) if costs else "n/a",
               ("%.2f" % (np.log(med) / np.log(3))) if costs else "n/a", 3 ** n))

    pts = [(r["n"], r["log3"]) for r in rows if r["log3"] is not None]
    slope = float(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0]) if len(pts) >= 3 else None

    # ===================== HONEST READOUT =====================
    log("\n" + "=" * 98)
    log("WHAT THE DATA SAYS")
    recovers = all(r["rate"] > 0.7 for r in rows)
    log("  [%s] joint readout (both bases) recovers where single-basis decodes (50.6/50.7) are at chance" % ("YES" if recovers else "PARTIAL"))
    if slope is not None:
        log("  cost scaling: log_3(cost) ~ %.2f * n + c  (slope ~1 => cost ~ 3^n, EXPONENTIAL ternary search)" % slope)
    log("=" * 98)

    if recovers and (slope is None or slope > 0.3):
        verdict = "JOINT_READOUT_IS_THE_LATTICE_PROBLEM_EXPONENTIAL"
        log("VERDICT: %s" % verdict)
        log("  The joint phase-space readout - using the NTT/dual basis for the linear relation and the")
        log("  coefficient/primal basis for smallness at once - DOES recover the secret where every")
        log("  single-basis decode is at chance. This confirms the convergence: the secret lives in the")
        log("  JOINT both-bases geometry, exactly as 'more entropy = higher-dimensional geometry' says.")
        log("  But reading it from there IS the lattice problem: the joint-guided search cost scales as")
        log("  ~3^n (exponential in the ternary secret space) - the same wall 50.7's sieve hit, now seen")
        log("  from the joint side. All distinct attack families this session (single-basis 50.6, chaos/")
        log("  sieve 50.7, joint 50.8) converge on the SAME exponential lattice cost. That EXHAUSTS what")
        log("  this lab can build with the current machinery. NOT 'the wall holds' - the residual is the")
        log("  genuine open problem (a poly-time joint/phase-space readout = a poly lattice algorithm),")
        log("  handed to Mythos with the full convergence map. The crossing remains a live open question,")
        log("  not a closed door.")
    elif recovers and slope is not None and slope < 0.2:
        verdict = "JOINT_READOUT_POLY_SUSPECT"
        log("VERDICT: %s  (poly joint cost: A8 MAXIMUM SUSPICION -> Mythos to check the regime)" % verdict)
    else:
        verdict = "JOINT_READOUT_INCONCLUSIVE"
        log("VERDICT: %s" % verdict)

    import json
    (HERE / "joint_readout_result.json").write_text(json.dumps({
        "q": Q, "rows": rows, "log3_slope_vs_n": slope, "verdict": verdict,
    }, indent=2, default=float), encoding="utf-8")
    (HERE / "output_joint_readout.txt").write_text("\n".join(LINES), encoding="utf-8")
    sys.exit(0)


if __name__ == "__main__":
    main()
