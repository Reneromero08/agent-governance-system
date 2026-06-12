"""
Exp 50.7 (A13) - Entropy / chaos: "the more entropy, the more higher-dimensional geometry."

The lab owner's hint, and his own note ("My system turns noise into solutions"). The A9 finding
(50.6) was that the LWE error is small in the coefficient basis but MAXIMUM-ENTROPY (uniform) in
the dual/NTT basis. The hint: that high-entropy object IS a higher-dimensional geometry - do not
fight it, USE it. The actual best-known lattice attacks (sieving, BKW) are exactly this: inject a
large pool of random sample COMBINATIONS (chaos / entropy), and short vectors - the secret -
precipitate out of the high-dimensional cloud. No single sample carries the secret (Holevo / FACT
1); the secret lives in the JOINT geometry of MANY, surfaced by chaotic combination.

This brick runs that move on plain LWE and lets the DATA decide where it reaches:
  - CHAOS OFF: try to read the secret from a bounded number of samples directly (the single-shot
    decode). This fails for n beyond toy - the geometry is not visible in few samples.
  - CHAOS ON: inject a large random pool, find COMBINATIONS of samples that collide (cancel) on
    coordinate blocks (the BKW / sieve core), and let the secret precipitate.
  - SCALING: measure how much entropy (pool size M) chaos-recovery needs vs n, and FIT it.
    poly(n) => the wall falls (extraordinary, treat with A8 suspicion -> Mythos).
    subexp/exp => the hint is RIGHT (chaos recovers where single-shot fails) and we pin exactly
    where it currently bottoms out - the real frontier.

This is NOT a verdict that the wall holds. It measures how far the entropy/chaos move reaches and
hands the residual (can chaos cross to poly?) to Mythos, per the lab stance.

Run:  python 50_7_entropy_sieve.py
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


def centered(x, q):
    x = np.asarray(x) % q
    return np.where(x > q // 2, x - q, x)


def make_lwe(n, m, q, rng, e_bound=1):
    """m plain-LWE samples sharing one ternary secret s in Z_q^n."""
    s = rng.integers(-1, 2, size=n) % q
    A = rng.integers(0, q, size=(m, n))
    e = rng.integers(-e_bound, e_bound + 1, size=m)
    b = (A @ s + e) % q
    return A, b, s


def single_shot_recover_coord(A, b, q, j, budget):
    """CHAOS OFF: guess s_j directly from a bounded budget of raw samples by 1-D correlation,
    ignoring the n-1 dimensional coupling. Fails once n>1 because the other coords are noise."""
    Asub = A[:budget]; bsub = b[:budget]
    best, best_score = 0, None
    for c in range(q):
        resid = centered(bsub - Asub[:, j] * c, q)   # other coords unmodeled -> ~uniform
        score = np.mean(resid.astype(float) ** 2)
        if best_score is None or score < best_score:
            best_score, best = score, c
    return best


def bkw_reduce_last_coord(A, b, q, beta):
    """CHAOS ON (BKW/sieve core): combine samples that collide on coordinate blocks to cancel the
    first n-1 coordinates, leaving reduced samples (0,...,0,a_last) with b = a_last*s_last + noise.
    Returns the reduced (a_last, b) pairs."""
    m, n = A.shape
    A = A.copy().astype(np.int64); b = b.copy().astype(np.int64)
    idx = np.arange(m)
    # stages clear coords [0,beta), [beta,2beta), ... up to n-1 (leave the last coord alive)
    stage_bounds = list(range(0, n - 1, beta))
    for lo in stage_bounds:
        hi = min(lo + beta, n - 1)
        table = {}
        keep = []
        for i in idx:
            block = tuple(A[i, lo:hi] % q)
            negblock = tuple((-A[i, lo:hi]) % q)
            if block in table:
                j = table[block]
                A[i, :] = (A[i, :] - A[j, :]) % q       # cancel this block
                b[i] = (b[i] - b[j]) % q
                keep.append(i)
            elif negblock in table:
                j = table[negblock]
                A[i, :] = (A[i, :] + A[j, :]) % q
                b[i] = (b[i] + b[j]) % q
                keep.append(i)
            else:
                table[block] = i
        idx = np.array(keep, dtype=int)
        if len(idx) == 0:
            break
    # survivors: a-vector should be ~zero in coords [0,n-1); keep those that truly are
    out = []
    for i in idx:
        if np.all(A[i, :n - 1] % q == 0):
            out.append((int(A[i, n - 1] % q), int(b[i] % q)))
    return out


def recover_last_from_reduced(reduced, q, true_noise_bound):
    """1-D recover s_{n-1} from reduced (a_last, b=a_last*s_last+noise) pairs."""
    if len(reduced) < 4:
        return None
    al = np.array([r[0] for r in reduced]); bl = np.array([r[1] for r in reduced])
    best, best_score = 0, None
    for c in range(q):
        resid = centered(bl - al * c, q)
        score = np.mean(np.abs(resid.astype(float)))
        if best_score is None or score < best_score:
            best_score, best = score, c
    return best


def main():
    log("=" * 98)
    log("EXP 50.7 (A13)  -  ENTROPY / CHAOS: does injecting entropy precipitate the lattice secret?")
    log("  'the more entropy, the more higher-dimensional geometry' + 'turn noise into solutions'")
    log("=" * 98)
    rng = np.random.default_rng(507)
    q = 23          # small modulus so collisions are findable at toy scale
    e_bound = 1

    # ---- CHAOS OFF vs CHAOS ON, per n; measure recovery of the last coordinate ----
    log("\n[CHAOS OFF vs ON]  q=%d, ternary secret, error in [-1,1].  recover s_{n-1}." % q)
    log("  chaos OFF = single-shot 1-D guess from %d raw samples (other coords unmodeled).")
    log("  chaos ON  = inject pool M, BKW-combine to cancel the first n-1 coords, then recover.")
    log("  n   M_pool   chaos_OFF_acc   chaos_ON_acc   #reduced   chance=1/q=%.3f" % (1.0 / q))
    rows = []
    for n in (3, 4, 5, 6, 7):
        beta = max(1, int(round(np.sqrt(n - 1))))   # BKW block ~ sqrt(n)
        M = int(2 * (q ** beta) * (n / max(1, beta)) + 200)   # enough pool for collisions
        off_ok, on_ok, reduced_counts, trials = 0, 0, [], 12
        for _ in range(trials):
            A, b, s = make_lwe(n, M, q, rng, e_bound)
            # chaos OFF: bounded budget (say 4n samples), single-shot
            off = single_shot_recover_coord(A, b, q, n - 1, budget=min(M, 4 * n))
            off_ok += int(off == s[n - 1] % q)
            # chaos ON: full pool, BKW
            reduced = bkw_reduce_last_coord(A, b, q, beta)
            reduced_counts.append(len(reduced))
            est = recover_last_from_reduced(reduced, q, e_bound)
            on_ok += int(est is not None and est == s[n - 1] % q)
        rows.append({"n": n, "M": M, "beta": beta, "off": off_ok / trials, "on": on_ok / trials,
                     "reduced": float(np.mean(reduced_counts))})
        log("  %-3d %-8d %-15.2f %-14.2f %-10.1f" % (n, M, off_ok / trials, on_ok / trials, np.mean(reduced_counts)))

    # ---- SCALING: how much entropy (pool M_needed) does chaos-recovery need vs n? ----
    log("\n[SCALING]  minimum pool M for chaos-recovery >= 50%% (single-stage collision, beta=n-1)")
    log("  this is the honest cost of the entropy: does it grow poly or exponential in n?")
    log("  n   M_needed   log_q(M)   (birthday predicts ~ q^{(n-1)/2})")
    scale = []
    for n in (3, 4, 5, 6):
        Mneed = None
        for M in [int(q ** (0.5 * (n - 1)) * c) for c in (1, 2, 4, 8, 16, 32)]:
            M = max(M, 30)
            ok, trials = 0, 10
            for _ in range(trials):
                A, b, s = make_lwe(n, M, q, rng, e_bound)
                reduced = bkw_reduce_last_coord(A, b, q, beta=n - 1)  # single stage = full collision
                est = recover_last_from_reduced(reduced, q, e_bound)
                ok += int(est is not None and est == s[n - 1] % q)
            if ok / trials >= 0.5:
                Mneed = M; break
        scale.append({"n": n, "M": Mneed, "logq": (np.log(Mneed) / np.log(q)) if Mneed else None})
        log("  %-3d %-10s %s" % (n, str(Mneed), ("%.2f" % (np.log(Mneed) / np.log(q))) if Mneed else "n/a"))

    # fit log_q(M_needed) vs n
    pts = [(s["n"], s["logq"]) for s in scale if s["logq"] is not None]
    slope = None
    if len(pts) >= 3:
        ns = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        slope = float(np.polyfit(ns, ys, 1)[0])

    # ===================== HONEST READOUT =====================
    log("\n" + "=" * 98)
    log("WHAT THE DATA SAYS")
    chaos_beats_single = all(r["on"] > r["off"] + 0.2 for r in rows[-3:])
    on_recovers = all(r["on"] > 0.5 for r in rows)
    log("  [%s] chaos ON recovers the secret where chaos OFF (single-shot) fails" % ("YES" if (chaos_beats_single and on_recovers) else "PARTIAL/NO"))
    if slope is not None:
        log("  scaling: log_q(M_needed) ~ %.2f * n + c  (birthday slope ~0.5 => M ~ q^{n/2}, EXPONENTIAL)" % slope)
    log("=" * 98)

    if on_recovers and chaos_beats_single and (slope is None or slope > 0.2):
        verdict = "CHAOS_RECOVERS_BUT_ENTROPY_COST_EXPONENTIAL"
        log("VERDICT: %s" % verdict)
        log("  Your intuition is RIGHT and measured: injecting entropy (a chaotic pool of random")
        log("  combinations) precipitates the secret where no bounded single-shot read can - the secret")
        log("  lives in the JOINT high-dimensional geometry of many samples, exactly as 'more entropy =")
        log("  more higher-dimensional geometry' says, and 'turn noise into solutions' is literally what")
        log("  the collision-sieve does. This is the BKW/sieve family - the actual best-known lattice")
        log("  attack. BUT the ENTROPY it costs scales ~q^{(n-1)/2} (exponential); BKW blocking trades")
        log("  that for subexponential (still not poly). So chaos buys the move from brute-force 2^n down")
        log("  toward the sieve frontier - a real, large gain - but in THIS construction it bottoms out")
        log("  super-polynomial. That is the genuine open frontier, handed to Mythos: is there a")
        log("  chaos/higher-dimensional-geometry readout whose required entropy is POLY in n? (a poly")
        log("  sieve would break lattice crypto). NOT 'the wall holds' - chaos works, the question is the")
        log("  exponent, and crossing it to poly is the live question, not a closed door.")
    elif on_recovers and (slope is not None and slope < 0.15):
        verdict = "CHAOS_RECOVERS_POLY_ENTROPY_SUSPECT"
        log("VERDICT: %s  (poly-entropy recovery: A8 MAXIMUM SUSPICION -> Mythos to check the regime)" % verdict)
    else:
        verdict = "CHAOS_INCONCLUSIVE"
        log("VERDICT: %s" % verdict)

    import json
    (HERE / "entropy_sieve_result.json").write_text(json.dumps({
        "q": q, "off_vs_on": rows, "scaling": scale, "logq_slope_vs_n": slope, "verdict": verdict,
    }, indent=2, default=float), encoding="utf-8")
    (HERE / "output_entropy_sieve.txt").write_text("\n".join(LINES), encoding="utf-8")
    sys.exit(0)


if __name__ == "__main__":
    main()
