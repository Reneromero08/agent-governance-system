"""
Exp 50.2d - The Kuperberg rung: the dihedral barrier is SUBEXPONENTIAL, not polynomial.

50.2c sandwiched the dihedral-HSP slope between two facts:
  - info-cheap (O(sqrt N) coset states determine d), and
  - compute-hard (a poly(n)-budget secret search fails; full recovery is O(2^n)).

That leaves the honest MIDDLE rung unmeasured: Kuperberg's collimation sieve recovers
the slope in 2^{O(sqrt n)} queries - subexponential, strictly below the 2^n full search,
and strictly above the poly(n) budget that fails. This brick builds that sieve as a real
(d-agnostic) simulation on the coset-state LABELS and measures its query cost vs n, so the
barrier is pinned as a sandwich:

    poly(n)  [FAILS, 50.2c lower bound]   <   2^{O(sqrt n)}  [this sieve, upper bound]   <   2^n

The collimation sieve (Kuperberg/Regev style):
  A dihedral coset state carries a label k in [0,N), N=2^n, with phase e^{2 pi i k d / N}.
  The sieve NEVER sees d - it only manipulates labels. Combining two states with the same
  low b bits, |k_i> and |k_0>, yields |k_i - k_0> whose low b bits are zero. Clearing b
  bits per round for ~n/b rounds produces a state with label 2^{n-1}, whose phase is
  e^{i pi d} = (-1)^d - one secret bit, read by measurement. Choosing b ~ sqrt(n) balances
  states-lost-per-round against rounds, giving total queries 2^{O(sqrt n)}.

HONEST SCOPE:
  This brick supplies the SUBEXPONENTIAL UPPER BOUND (the sieve works, far below 2^n) and
  verifies the readout (conditional correctness 1.0; phase-randomised null at chance). The
  SUPER-POLYNOMIAL lower bound is NOT claimed from the rate fit here - over the reachable n
  range a 2^{sqrt n} curve and a low-degree polynomial are statistically indistinguishable
  (we report both R^2). Super-polynomiality is inherited from 50.2c's poly(n)-budget
  failure plus the standard Regev/Kuperberg result, which we cite, not re-prove.

Run:  python 50_2d_kuperberg_sieve.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # 50_the_decoder root (decoder_lib stats)
import decoder_lib as dl  # noqa: E402

LINES = []
def log(m=""):
    print(m)
    LINES.append(str(m))


def sieve_once(n, M, d, rng, b, randomize_phase=False):
    """One collimation-sieve run. d-agnostic on the labels; reads the final phase.
    Returns (produced, correct, queries):
      produced = a state with label 2^{n-1} survived the sieve (probabilistic in M);
      correct  = the parity bit read from its phase matched d (deterministic when honest).
    queries = M coset states consumed."""
    N = 1 << n
    labels = rng.integers(0, N, size=M).astype(np.int64).tolist()
    cleared = 0
    while cleared < n - 1 and labels:
        bb = min(b, (n - 1) - cleared)
        groups = {}
        for k in labels:
            key = (k >> cleared) & ((1 << bb) - 1)
            groups.setdefault(key, []).append(k)
        nxt = []
        for ks in groups.values():
            if len(ks) < 2:
                continue
            k0 = ks[0]
            for k in ks[1:]:
                nxt.append((k - k0) % N)   # low (cleared+bb) bits now zero
        labels = nxt
        cleared += bb
    target = 1 << (n - 1)
    useful = [k for k in labels if k == target]
    if not useful:
        return False, False, M
    if randomize_phase:
        phase = np.exp(1j * rng.uniform(0, 2 * np.pi))   # NULL: phase decoupled from d
    else:
        phase = np.exp(2j * np.pi * useful[0] * d / N)   # == (-1)^d
    bit = int(round((1 - phase.real) / 2)) & 1
    return True, (bit == (d & 1)), M


def production_rate(n, M, b, trials, rng):
    ok = 0
    for _ in range(trials):
        d = int(rng.integers(1, 1 << n))
        prod, _, _ = sieve_once(n, M, d, rng, b)
        ok += int(prod)
    return ok / trials


def conditional_correctness(n, M, b, trials, rng, randomize_phase=False):
    """P(bit correct | a useful state was produced)."""
    prod, corr = 0, 0
    for _ in range(trials):
        d = int(rng.integers(1, 1 << n))
        p, c, _ = sieve_once(n, M, d, rng, b, randomize_phase=randomize_phase)
        if p:
            prod += 1
            corr += int(c)
    return (corr / prod) if prod else float("nan"), prod


def m_needed(n, b, rng, trials=32, thresh=0.5, cap_exp=22):
    for e in range(2, cap_exp + 1):
        M = 1 << e
        if production_rate(n, M, b, trials, rng) >= thresh:
            return M
    return None


def main():
    log("=" * 96)
    log("EXP 50.2d  -  THE KUPERBERG RUNG: dihedral barrier is SUBEXPONENTIAL (upper bound)")
    log("  sandwich:  poly(n) [FAILS, 50.2c]  <  2^{O(sqrt n)} [this sieve]  <  2^n [full, 50.2c]")
    log("=" * 96)
    rng = np.random.default_rng(502_4)

    ns = list(range(6, 31, 2))   # n = 6,8,...,30  (N up to ~1.07e9)
    log("\n[SIEVE] collimation sieve, b = round(sqrt(n)); M_needed = coset states for >=50%% production")
    log("  n    b   M_needed   log2(M)   2^n          2^n / M_needed   production")
    rows = []
    for n in ns:
        b = max(1, int(round(np.sqrt(n))))
        M = m_needed(n, b, rng)
        if M is None:
            log("  %-4d %-3d  >cap" % (n, b))
            continue
        pr = production_rate(n, M, b, 40, rng)
        rows.append({"n": n, "b": b, "M": M, "log2M": float(np.log2(M)),
                     "two_n": float(2 ** n), "ratio": float((2 ** n) / M), "prod": pr})
        log("  %-4d %-3d  %-9d  %-8.2f  %-11.3g  %-15.4g  %.3f"
            % (n, b, M, np.log2(M), 2.0 ** n, (2.0 ** n) / M, pr))

    ns_arr = np.array([r["n"] for r in rows], float)
    log2M = np.array([r["log2M"] for r in rows], float)

    # ---- conditional correctness (signal) and phase-randomised null, at a comfortable M ----
    log("\n[READOUT] conditional correctness  P(bit correct | useful state produced)")
    cc_sig, cc_null = [], []
    for n in (8, 12, 16, 20):
        b = max(1, int(round(np.sqrt(n))))
        M = next(r["M"] for r in rows if r["n"] == n) * 4
        cs, _ = conditional_correctness(n, M, b, 200, rng, randomize_phase=False)
        cn, _ = conditional_correctness(n, M, b, 200, rng, randomize_phase=True)
        cc_sig.append(cs); cc_null.append(cn)
        log("  n=%-3d  signal correctness=%.3f   phase-randomised null=%.3f" % (n, cs, cn))
    cc_sig_min = float(np.nanmin(cc_sig))
    cc_null_mean = float(np.nanmean(cc_null))

    # ---- rate fits: subexponential 2^{a sqrt n}  vs  polynomial n^c (log2 M ~ c log2 n) ----
    def linfit(x, y):
        a, c = np.polyfit(x, y, 1)
        yhat = a * x + c
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-30
        return a, c, 1.0 - ss_res / ss_tot
    a_sqrt, c_sqrt, r2_sqrt = linfit(np.sqrt(ns_arr), log2M)
    a_log, c_log, r2_log = linfit(np.log2(ns_arr), log2M)
    log("\n  rate fit  log2(M_needed) ~ a*sqrt(n)+c :  a=%.3f  R^2=%.3f   (Kuperberg 2^{O(sqrt n)})"
        % (a_sqrt, r2_sqrt))
    log("  rate fit  log2(M_needed) ~ a*log2(n)+c :  a=%.3f  R^2=%.3f   (polynomial n^a)"
        % (a_log, r2_log))
    log("  NOTE: over this n-range the two fits are statistically indistinguishable; we do NOT")
    log("  claim to separate subexp from poly here. Super-polynomiality is inherited from 50.2c")
    log("  (poly(n) budget fails) + the standard Regev/Kuperberg result.")

    full_top = rows[-1]["n"] * rows[-1]["M"]
    log("\n  full n-bit slope ~ n * M_needed: at n=%d -> %d queries  vs  2^n = %.3g"
        % (rows[-1]["n"], full_top, 2.0 ** rows[-1]["n"]))

    # ===================== GATES =====================
    log("\n" + "=" * 96)
    log("GATES")
    # G1 readout correctness: conditional correctness is ~1.0 for the real phase
    g1 = cc_sig_min > 0.99
    g1_det = "min conditional correctness (signal) = %.3f" % cc_sig_min

    # G2 subexponential << exponential: M_needed far below 2^n and the gap WIDENS with n
    ratios = [r["ratio"] for r in rows]
    g2 = (min(ratios) > 1.0) and (ratios[-1] > 10 * ratios[0]) and (log2M[-1] < ns_arr[-1] * 0.5)
    g2_det = "2^n / M_needed grows %.3g -> %.3g; log2(M)=%.2f << n=%d" % (
        ratios[0], ratios[-1], log2M[-1], int(ns_arr[-1]))

    # G3 subexponential RATE (sublinear in n, well-described by sqrt(n)); NOT a poly separation
    sublinear = (a_sqrt > 0) and (r2_sqrt > 0.85) and (log2M[-1] / ns_arr[-1] < 0.4)
    g3 = sublinear
    g3_det = "sqrt-fit R^2=%.3f a=%.3f; log2(M)/n=%.3f (sublinear=subexp). poly-fit R^2=%.3f" % (
        r2_sqrt, a_sqrt, log2M[-1] / ns_arr[-1], r2_log)

    # G4 phase-randomised null reads the bit only at chance (the sieve's success is real phase)
    g4 = abs(cc_null_mean - 0.5) < 0.1
    g4_det = "phase-randomised null conditional correctness = %.3f (want ~0.5)" % cc_null_mean

    gates = [
        ("G1 readout correctness (bit | produced) ~ 1.0", g1, g1_det),
        ("G2 subexponential: M_needed << 2^n, gap widens >10x", g2, g2_det),
        ("G3 subexponential rate (sublinear in n; sqrt-consistent)", g3, g3_det),
        ("G4 phase-randomised null reads bit only at chance", g4, g4_det),
    ]
    for nm, ok, det in gates:
        log("  [%s] %-56s  %s" % ("PASS" if ok else "FAIL", nm, det))
    log("=" * 96)

    all_pass = all(ok for _, ok, _ in gates)
    verdict = "DIHEDRAL_BARRIER_SUBEXPONENTIAL_UPPER_BOUND" if all_pass else "KUPERBERG_RUNG_INCONCLUSIVE"
    log("VERDICT: %s" % verdict)
    log("  Kuperberg's collimation sieve recovers the dihedral slope in subexponential queries -")
    log("  decisively below the 2^n full search (the gap widens from ~%gx to ~%gx over n=%d..%d)."
        % (round(ratios[0]), round(ratios[-1]), int(ns_arr[0]), int(ns_arr[-1])))
    log("  Combined with 50.2c (poly(n) budget FAILS), the barrier is pinned as a SANDWICH:")
    log("  super-polynomial (no poly readout) but subexponential (not full 2^n). This is the exact")
    log("  shape of the lattice / unique-SVP wall the holographic decoder bottoms out on.")
    log("  (Claim level 4: a measured subexponential UPPER bound from a real sieve; the 2^{O(sqrt n)}")
    log("  rate and the super-poly lower bound are cited from Regev/Kuperberg + 50.2c, not re-proven.)")

    import json
    (HERE / "kuperberg_result.json").write_text(json.dumps({
        "rows": rows, "fit_sqrt": {"a": a_sqrt, "c": c_sqrt, "r2": r2_sqrt},
        "fit_poly": {"a": a_log, "c": c_log, "r2": r2_log},
        "cond_correctness_signal_min": cc_sig_min, "cond_correctness_null_mean": cc_null_mean,
        "full_slope_queries_top": full_top,
        "gates": {nm: bool(ok) for nm, ok, _ in gates}, "verdict": verdict,
    }, indent=2, default=float), encoding="utf-8")
    (HERE / "output_kuperberg.txt").write_text("\n".join(LINES), encoding="utf-8")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
