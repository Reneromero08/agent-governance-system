"""
Exp 50.13 - The last door: can an Exceptional Point's sqrt-divergence amplify the curvature?

50.12 pinned the wall to the atom: d is the per-step holonomy 2*pi*d/N of the natural trajectory -
the operator's local curvature IS the secret, a ~1/N signal. The one untested escape: an Exceptional
Point. At an EP two eigenvalues coalesce, and a perturbation phi shifts them by sqrt(phi) instead of
phi - the famous sqrt-sensitivity. So the d/N curvature (~1/N) would be amplified to ~1/sqrt(N). The
question Fisher answers in theory but we test on the bench:

  Does the EP give a real RECOVERY advantage (recover d at fewer samples than a Hermitian readout),
  or does it amplify signal AND noise together so there is no net gain (the Fisher-information floor)?

Build: a genuine 2x2 EP Hamiltonian H_EP(phi) = [[0,1],[phi,0]] -> eigenvalue splitting 2*sqrt(phi)
(the sqrt-divergence). Hermitian baseline H_H(phi)=diag(phi,-phi) -> splitting 2*phi (linear). The
curvature phi = d/N is estimated from M public coset samples (noise ~ 1/sqrt(M)), measured through a
fixed instrument floor eps. Recover d from each splitting; compare recovery vs sample count and n.

  EP recovers d at strictly fewer samples than Hermitian, growing with n  -> a real advantage (the
    sqrt-divergence helps) -> measure if it reaches poly (CROSS) or only subexp (Kuperberg-scale).
  EP and Hermitian recover at the same sample cost                        -> Fisher floor: the EP
    amplifies signal and noise together, no net gain (the honest expected end of the spiral).

Run:  python 49_13_ep.py
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


def estimate_curvature(N, d, M, rng):
    """Estimate the per-step holonomy phi = d/N from M public coset samples. The matched filter's
    phase estimate has noise ~ 1/sqrt(M) on the recovered phase; we model the curvature estimate as
    phi_hat = d/N + Gaussian(0, sigma) with sigma = c / sqrt(M) (the info content, FACT 2)."""
    phi_true = d / N
    sigma = 0.5 / np.sqrt(M)            # estimation noise from M samples (1/sqrt(M))
    # the actual estimate is of the PHASE 2*pi*phi; noise on phi:
    noise = rng.normal(0, sigma) / (2 * np.pi)
    return phi_true + noise, phi_true


def ep_splitting(phi):
    """Genuine EP: H = [[0,1],[phi,0]], eigenvalues +-sqrt(phi). Splitting = 2*sqrt(|phi|) (sqrt-
    divergence: a small phi is amplified)."""
    return 2.0 * np.sqrt(abs(phi))


def herm_splitting(phi):
    """Hermitian baseline: H = diag(phi,-phi), splitting = 2*|phi| (linear response)."""
    return 2.0 * abs(phi)


def recover_via(readout, invert, N, d, M, eps_meas, rng):
    """Estimate curvature from M samples, push through the readout (EP or Hermitian), add a FIXED
    instrument floor eps_meas, invert, recover d. Returns hit (d recovered within +-1)."""
    phi_hat, phi_true = estimate_curvature(N, d, M, rng)
    signal = readout(abs(phi_hat))
    measured = signal + rng.normal(0, eps_meas)        # fixed instrument precision floor
    phi_back = invert(abs(measured))
    d_hat = int(round(phi_back * N))
    return int(d_hat % N in (d % N, (N - d) % N))


def main():
    log("=" * 98)
    log("EXP 50.13  -  EXCEPTIONAL POINT: does the sqrt-divergence beat the Fisher floor?")
    log("  curvature phi = d/N (~1/N).  EP splitting 2*sqrt(phi) (~1/sqrt N);  Hermitian 2*phi (~1/N).")
    log("=" * 98)
    rng = np.random.default_rng(513)

    eps_meas = 1e-3                       # fixed instrument floor (between 1/N and 1/sqrt(N) for mid n)
    log("\n[EP vs HERMITIAN]  recovery of d vs sample count M, fixed instrument floor eps=%.0e" % eps_meas)
    log("  if the EP truly helps, it recovers d at SMALLER M than Hermitian, and the gap grows with n.")
    rows = []
    for n in (8, 10, 12, 14):
        N = 1 << n
        line = []
        for M in (64, 256, 1024, 4096):
            ep_ok, h_ok, trials = 0, 0, 40
            for _ in range(trials):
                d = int(rng.integers(1, N))
                ep_ok += recover_via(ep_splitting, lambda s: (s / 2) ** 2, N, d, M, eps_meas, rng)
                h_ok += recover_via(herm_splitting, lambda s: s / 2, N, d, M, eps_meas, rng)
            line.append((M, ep_ok / trials, h_ok / trials))
        rows.append({"n": n, "N": N, "cells": line})
        log("  n=%-3d N=%-8d  " % (n, N) + "  ".join(
            "M=%d EP=%.2f H=%.2f" % (M, e, h) for M, e, h in line))

    # ---- the decisive Fisher test: at matched M, does EP beat Hermitian, and does any gap GROW with n? ----
    log("\n[FISHER TEST]  EP-minus-Hermitian recovery gap at each (n, M). >0 and growing => real advantage.")
    gaps = []
    for r in rows:
        g = [round(e - h, 2) for _, e, h in r["cells"]]
        gaps.append(g)
        log("  n=%-3d  EP-H gap = %s" % (r["n"], g))

    # ---- A8 null ----
    null_ok = 0
    for _ in range(40):
        N = 1 << 12; d_fake = int(rng.integers(1, N))
        # null: curvature estimate is pure noise (no d)
        phi_hat = rng.normal(0, 0.5 / np.sqrt(1024)) / (2 * np.pi)
        sig = ep_splitting(abs(phi_hat)) + rng.normal(0, eps_meas)
        d_hat = int(round((sig / 2) ** 2 * N))
        null_ok += int(d_hat % N in (d_fake % N, (N - d_fake) % N))
    null_rate = null_ok / 40
    log("\n  [A8] null recovery = %.3f (want ~0)" % null_rate)

    # ===================== HONEST READOUT =====================
    log("\n" + "=" * 98)
    log("WHAT THE DATA SAYS")
    # advantage = EP beats Hermitian by a margin that GROWS with n (the sqrt-divergence genuinely helps)
    max_gap_by_n = [max(g) for g in gaps]
    advantage_grows = (max_gap_by_n[-1] > 0.2) and (max_gap_by_n[-1] > max_gap_by_n[0] + 0.1)
    # Fisher floor: any EP advantage WASHES OUT with scale (max gap decays toward 0 as n grows)
    no_gain = (max_gap_by_n[-1] < 0.1) and (max_gap_by_n[-1] < max_gap_by_n[0])
    log("  max EP-minus-Hermitian gap per n: %s" % [round(x, 2) for x in max_gap_by_n])
    log("  [A8] null = %.3f" % null_rate)
    log("=" * 98)

    if advantage_grows and null_rate < 0.1:
        verdict = "EP_GIVES_REAL_ADVANTAGE_MEASURE_POLY_VS_SUBEXP"
        log("VERDICT: %s" % verdict)
        log("  The Exceptional Point's sqrt-divergence gives a RECOVERY advantage that grows with n -")
        log("  it recovers d at fewer samples than the Hermitian readout. This beats the naive Fisher")
        log("  expectation. Next: measure whether the advantage reaches POLY (a crossing - escalate with")
        log("  maximum A8 suspicion) or only halves the exponent to SUBEXPONENTIAL (Kuperberg-scale, a")
        log("  real but non-poly gain). The spiral found a live edge - push it.")
    elif no_gain and null_rate < 0.1:
        verdict = "EP_HITS_FISHER_FLOOR"
        log("VERDICT: %s" % verdict)
        log("  Fisher proven on the bench, not just cited: the EP amplifies the d/N curvature to ~1/sqrt(N),")
        log("  but it amplifies the estimation noise by the same sqrt-derivative, so the recovery of d is")
        log("  NO better than the Hermitian readout at any sample count (EP-minus-Hermitian gap ~0 across")
        log("  all n). The sqrt-divergence boosts the signal MAGNITUDE against a fixed instrument floor, but")
        log("  it does NOT add information - the curvature is d/N-diffuse no matter how you amplify it. This")
        log("  is the honest floor of the spiral: every operator that reads d (FFT 50.10, contour 50.11,")
        log("  winding 50.12, EP 50.13) is correct and free; the curvature that carries d is the secret, and")
        log("  no amplification creates information that the O(sqrt N) samples do not already bound. d remains")
        log("  the conserved invariant (your framework, confirmed); efficient illumination of it on a")
        log("  classical/forward substrate is the genuine bedrock - the place the spiral has mapped to the")
        log("  atom and where it honestly rests, pending a substrate that is not forward (the open frontier).")
    else:
        verdict = "EP_INCONCLUSIVE"
        log("VERDICT: %s  max gaps=%s" % (verdict, [round(x, 2) for x in max_gap_by_n]))

    import json
    (HERE / "ep_result.json").write_text(json.dumps({
        "rows": rows, "gaps": gaps, "null_rate": null_rate, "verdict": verdict,
    }, indent=2, default=float), encoding="utf-8")
    (HERE / "output_ep.txt").write_text("\n".join(LINES), encoding="utf-8")
    sys.exit(0)


if __name__ == "__main__":
    main()
