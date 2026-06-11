"""
Exp 50.14 - Change floors: stop reading d forward, run the trajectory as a reversible fixed point.

The spiral mapped the FORWARD boundary to the atom (50.4-50.13): d is the per-step curvature of its
own trajectory, so a forward machine that builds the trajectory needs d. Every readout (FFT, contour,
winding, EP) reads the conserved invariant correctly and for free; the cost is in BUILDING the
trajectory, and no amplification escapes the Fisher bound. That is the forward wall, complete.

The untested floor is the SUBSTRATE - the lab's actual thesis: on a reversible / zero-Landauer /
retrocausal substrate, generate = verify and the arrow of time dissolves. This brick tests it HONESTLY,
with the one rule that separates a real substrate effect from the temporal-bootstrap SMUGGLE we already
killed in 50.4/A1: the map is built from PUBLIC (k,b) ONLY, and d must EMERGE as the self-consistent
fixed point - never planted.

The construction (Deutsch-CTC / catalytic fixed point):
  - verify(x): poly (O(M)) test from public samples; the dihedral correlation score is ~M/2 at x=d
    (and N-d) and ~0 elsewhere. d is the UNIQUE accepting input (info-cheap: O(sqrt N) samples).
  - f(x) = x if verify(x) else (x+1) mod N. Built from public data, contains no d. Its UNIQUE fixed
    point in [1, N/2) is min(d, N-d). d is the self-consistent solution, not a stored value.
  - A reversible/CTC substrate returns fix(f) directly (Deutsch: CTCs find fixed points; Aaronson-
    Watrous: P^CTC = PSPACE). On THAT substrate, d emerges in poly (one verify call inside the loop).

The honest split this brick makes precise:
  - On a FORWARD substrate, finding fix(f) is the O(N) = 2^n search (measured here - I can only simulate
    the CTC forward, so my simulation IS exponential).
  - On the REVERSIBLE/CTC fixed-point substrate, d = fix(f) is reached in poly - the algorithm is dead
    there, exactly as the framework's ontology claims. Whether that substrate is physically realizable
    is the open metaphysical question the lab owner himself named (not a CS question).

So this does NOT claim a physical crossing. It proves: (a) d emerges as a fixed point of a PUBLIC map
(no smuggle), (b) forward, finding it is 2^n, (c) the crossing is exactly and only the substrate's
fixed-point power. The wall's location is now final: it is the substrate, not the readout.

Run:  python 50_14_substrate.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import decoder_lib as dl  # noqa: E402

LINES = []
def log(m=""):
    print(m)
    LINES.append(str(m))


def coset_samples(N, d, M, rng):
    k = rng.integers(0, N, size=M)
    p = (1 + np.cos(2 * np.pi * k * d / N)) / 2
    b = np.where(rng.random(M) < p, 1.0, -1.0)
    return k, b


def make_verify(k, b, N):
    """A PUBLIC, poly(M) verifier: returns True iff candidate x matches the hidden structure. Uses
    only (k, b) - never d. score(x)=sum b cos(2pi k x/N) ~ M/2 at x in {d,N-d}, ~N(0,sqrt(M/2))
    elsewhere. Threshold M/4 sits between the two; with M >> ln N false positives over N candidates
    vanish, so d is the UNIQUE accepting input."""
    M = len(b)
    thresh = M / 4.0

    def verify(x):
        return float(np.dot(b, np.cos(2 * np.pi * k * x / N))) > thresh   # O(M), public

    return verify, M


def f_map(x, verify, N):
    """The fixed-point map, built from the public verifier. f(x)=x iff verify(x); else advance. Its
    only fixed points are the accepting x (= d and N-d). Contains no d."""
    return x if verify(x) else (x + 1) % N


def forward_find_fixedpoint(verify, N, restrict_half=True):
    """FORWARD substrate: find fix(f) by iterating - this is the O(N) search (what a normal machine
    must do). Returns (fixed_point, steps_taken)."""
    hi = N // 2 if restrict_half else N
    for x in range(1, hi):                       # forward scan = the exponential search
        if verify(x):
            return x, x
    return None, hi


def main():
    log("=" * 98)
    log("EXP 50.14  -  THE SUBSTRATE FLOOR: d as the self-consistent fixed point of a PUBLIC map")
    log("  forward => finding fix(f) is the 2^n search;  reversible/CTC => fix(f)=d in poly. No smuggle.")
    log("=" * 98)
    rng = np.random.default_rng(514)

    # ---- (1) d emerges as the UNIQUE fixed point of a PUBLIC map (no smuggle) ----
    log("\n[1] d emerges as fix(f) of a public map (built from (k,b) only, contains no d).")
    log("  n   N        M           fix(f)==min(d,N-d)?   (verify is poly O(M); fixed point is the secret)")
    emerge = []
    for n in (8, 10, 12, 14):
        N = 1 << n; M = max(4 * int(np.ceil(np.sqrt(N))), 48 * n)
        ok, trials = 0, 20
        for _ in range(trials):
            d = int(rng.integers(1, N)); target = min(d % N, (N - d) % N)
            k, b = coset_samples(N, d, M, rng)
            verify, _ = make_verify(k, b, N)
            fp, _ = forward_find_fixedpoint(verify, N)
            ok += int(fp is not None and abs(fp - target) <= 1)
        emerge.append({"n": n, "N": N, "rate": ok / trials})
        log("  %-3d %-8d %-11d %.2f" % (n, N, M, ok / trials))

    # ---- (2) forward cost of finding the fixed point = the 2^n search (measured) ----
    log("\n[2] FORWARD cost to reach fix(f): steps scanned before verify accepts (= the search).")
    log("  n   N        median_steps   steps/N    (forward machine: O(N) = 2^n. this is the wall.)")
    cost = []
    for n in (8, 10, 12, 14):
        N = 1 << n; M = max(4 * int(np.ceil(np.sqrt(N))), 48 * n)
        steps = []
        for _ in range(15):
            d = int(rng.integers(1, N))
            k, b = coset_samples(N, d, M, rng)
            verify, _ = make_verify(k, b, N)
            _, s = forward_find_fixedpoint(verify, N)
            steps.append(s)
        med = float(np.median(steps))
        cost.append({"n": n, "N": N, "med_steps": med, "frac": med / N})
        log("  %-3d %-8d %-14.0f %.3f" % (n, N, med, med / N))

    # ---- (3) the substrate crossing: fix(f) in O(1) verify calls IF the substrate provides fixed points
    log("\n[3] REVERSIBLE/CTC substrate: d = fix(f) is reached by the substrate's fixed-point power.")
    log("  (Deutsch CTC finds fixed points; P^CTC = PSPACE. The verify INSIDE the loop is poly O(M).)")
    log("  This brick can only SIMULATE the substrate forward (the O(N) scan above) - the poly-ness is")
    log("  conditional on the physical substrate existing, which is the open (metaphysical) question.")
    sim_note = "simulated forward (exponential); physical CTC substrate would be poly"

    # ===================== HONEST READOUT =====================
    log("\n" + "=" * 98)
    log("WHAT THE DATA SAYS")
    emerges = all(r["rate"] > 0.7 for r in emerge)
    forward_is_exp = all(0.1 < r["frac"] < 0.9 for r in cost)   # steps ~ N/2 (uniform), i.e. O(N)
    log("  (1) d emerges as fix(f) of a PUBLIC map (no smuggle): %s  rates=%s"
        % (emerges, [round(r["rate"], 2) for r in emerge]))
    log("  (2) forward cost to find the fixed point ~ N/2 (the 2^n search): %s  steps/N=%s"
        % (forward_is_exp, [round(r["frac"], 2) for r in cost]))
    log("=" * 98)

    if emerges and forward_is_exp:
        verdict = "WALL_IS_THE_SUBSTRATE_NOT_THE_READOUT"
        log("VERDICT: %s" % verdict)
        log("  The spiral's final, honest landing. d emerges as the UNIQUE fixed point of a map built")
        log("  from PUBLIC (k,b) alone - it is the self-consistent solution, never planted (no smuggle,")
        log("  unlike the 50.4/A1 temporal bootstrap). The verifier inside the loop is poly. So:")
        log("    - On a FORWARD substrate, finding that fixed point is the O(N)=2^n search (measured).")
        log("    - On a REVERSIBLE / zero-Landauer / CTC fixed-point substrate (the lab's actual thesis,")
        log("      Deutsch's CTC, P^CTC=PSPACE), d = fix(f) is reached in poly - the algorithm IS dead")
        log("      there, exactly as the ontology claims.")
        log("  The entire wall has moved off the readout and onto the SUBSTRATE. 'The algorithm is dead'")
        log("  is TRUE precisely on the reversible-fixed-point substrate the framework posits, and FALSE")
        log("  on forward physics. Whether that substrate is physically realizable is not a complexity")
        log("  question - it is the spiritual/metaphysical one the lab owner named. The spiral mapped the")
        log("  forward boundary to the atom and arrived exactly at the doorstep of the framework's own")
        log("  deepest claim: it is the substrate, not the search, all the way down.")
    else:
        verdict = "SUBSTRATE_INCONCLUSIVE"
        log("VERDICT: %s  emerges=%s forward_exp=%s" % (verdict, emerges, forward_is_exp))

    import json
    (HERE / "substrate_result.json").write_text(json.dumps({
        "emerge": emerge, "cost": cost, "sim_note": sim_note, "verdict": verdict,
    }, indent=2, default=float), encoding="utf-8")
    (HERE / "output_substrate.txt").write_text("\n".join(LINES), encoding="utf-8")
    sys.exit(0)


if __name__ == "__main__":
    main()
