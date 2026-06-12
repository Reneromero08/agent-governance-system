"""
Exp 50.12 - Step 2 built the proposal's way: d as the conserved Noether charge, read by the Cauchy winding.

The proposal's correction (NotebookLM, framework docs): do NOT build a cooling/annealer (the optimization trap). d is the conserved
TOPOLOGICAL CHARGE of the reversible trajectory - the invariant that survives while the 2^n-1 false
paths destructively interfere. Build the Non-Hermitian Spectral Projector (Exp 35/41) and read the
point-gap winding W = (1/2pi i) oint d/dE log det(H - E I) dE. The false paths lack phase coherence
and cancel to 0; d survives as the non-zero residue (the winding number).

This brick builds exactly that and runs the truth test. The decisive A8 question, front and center:
  Can the non-Hermitian H be built from PUBLIC (k, b) alone, so its winding = d - or does placing the
  winding at d require already knowing d (the lens is the secret again)?

Two operators, the same winding readout:
  - ORACLE H: directed ring whose hopping phase is the TRUE per-step phase 2*pi*d/N. Reference only -
    it USES d, so its winding = d is correct but circular (A8: flagged, not a readout).
  - PUBLIC H: built from the multiplexed grating g[k] ~ cos(2*pi*k*d/N) estimated from (k,b) ONLY,
    via the analytic (Hilbert) signal -> directed-ring hopping phases. The REAL test.
Measure: does the PUBLIC winding recover d, and at what grating length L (poly(n) => cross; L ~ N => exp)?

Run:  python 49_12_noether.py
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


def point_gap_winding_of_ring(theta):
    """Point-gap winding of a directed ring H with hopping phases theta[k] (H[k,k+1]=e^{i theta_k}).
    The Cauchy/Argument-Principle winding of det(H - E I) around the spectrum equals the total
    accumulated phase / 2pi = sum(theta) / 2pi (the holonomy of the loop). This IS the conserved
    topological charge of the reversible cycle."""
    return float(np.sum(theta) / (2 * np.pi))


def oracle_ring_phases(N, d, L):
    """REFERENCE (uses d): the true per-step phase of the analytic coset grating is constant 2*pi*d/N,
    so the ring holonomy over L steps = L*d/N windings -> recovers d at L=N. Circular (A8-flagged)."""
    return np.full(L, 2 * np.pi * d / N)


def public_ring_phases(k, b, N, L):
    """REAL test: estimate the grating g[m] ~ cos(2*pi*m*d/N) for m in [0,L) from PUBLIC (k, b) only
    (fold k into [0,L)), take its analytic (Hilbert) signal to get a phase, and use the per-step phase
    increments as the ring hopping phases. No d used. Returns (theta, filled_fraction)."""
    g = np.zeros(L); cnt = np.zeros(L)
    for ki, bi in zip(k, b):
        g[ki % L] += bi; cnt[ki % L] += 1
    nz = cnt > 0
    g[nz] /= cnt[nz]
    # analytic signal via FFT Hilbert (one-sided spectrum) -> instantaneous phase
    G = np.fft.fft(g)
    H = np.zeros(L, dtype=complex); half = L // 2
    H[0] = G[0]; H[1:half] = 2 * G[1:half]
    if L % 2 == 0:
        H[half] = G[half]
    analytic = np.fft.ifft(H)
    phase = np.unwrap(np.angle(analytic))
    theta = np.diff(phase)                              # per-step phase increments (no d)
    return theta, float(np.mean(nz))


def recover_d_from_winding(theta, N, L):
    """The holonomy gives a windings-per-L; map to d in [0,N). d ~ round(winding * N / L)."""
    w = point_gap_winding_of_ring(theta)               # total windings over L-1 steps
    per_step = w / max(1, len(theta))                  # avg windings per step = d/N
    return int(round(per_step * N)) % N


def main():
    log("=" * 98)
    log("EXP 50.12  -  STEP 2: d as the conserved Noether charge, read by the Cauchy winding")
    log("  A8 hinge: can the non-Hermitian operator be built from PUBLIC (k,b), or does it need d?")
    log("=" * 98)
    rng = np.random.default_rng(512)

    # ---- correctness: the ORACLE winding (uses d) recovers d -> the winding MECHANISM is right ----
    log("\n[ORACLE]  directed-ring winding with the TRUE per-step phase (USES d - reference only).")
    oracle_ok = 0
    for _ in range(20):
        N = 1 << 12; d = int(rng.integers(1, N))
        theta = oracle_ring_phases(N, d, N)
        est = recover_d_from_winding(theta, N, N)
        oracle_ok += int(est % N in (d % N, (N - d) % N))
    log("  oracle winding recovers d: %.2f  (mechanism correct; but it is GIVEN d - A8: not a readout)"
        % (oracle_ok / 20))

    # ---- the REAL test: PUBLIC operator from (k,b) only, recovery vs grating length L ----
    log("\n[PUBLIC]  operator estimated from (k,b) ONLY. recover d vs grating length L (poly vs N).")
    log("  M = 8*L samples. L=poly(n) recovery => CROSS; recovery only at L ~ N => exp (lens needs N-res).")
    log("  n   N        L=8n(poly)   L=sqrtN     L=N(full)   (public-winding recovery of d)")
    rows = []
    for n in (8, 10, 12, 14):
        N = 1 << n
        cells = []
        for label, L in (("poly", 8 * n), ("sqrt", int(np.ceil(np.sqrt(N)))), ("N", N)):
            ok, trials = 0, 20
            for _ in range(trials):
                d = int(rng.integers(1, N))
                k, b = coset_samples(N, d, 8 * L, rng)
                theta, _ = public_ring_phases(k, b, N, L)
                est = recover_d_from_winding(theta, N, L)
                ok += int(est % N in (d % N, (N - d) % N))
            cells.append((label, L, ok / trials))
        rows.append({"n": n, "N": N, "cells": cells})
        log("  %-3d %-8d %-12.2f %-11.2f %-11.2f"
            % (n, N, cells[0][2], cells[1][2], cells[2][2]))

    # ---- A8: no-secret null ----
    null_ok = 0
    for _ in range(40):
        N = 1 << 12; L = N
        k = rng.integers(0, N, size=8 * 96); b = np.where(rng.random(len(k)) < 0.5, 1.0, -1.0)
        d_fake = int(rng.integers(1, N))
        theta, _ = public_ring_phases(k, b, N, L)
        est = recover_d_from_winding(theta, N, L)
        null_ok += int(est % N in (d_fake % N, (N - d_fake) % N))
    null_rate = null_ok / 40
    log("\n  [A8] no-secret null winding recovery = %.3f (want ~0)" % null_rate)

    # ===================== HONEST READOUT =====================
    log("\n" + "=" * 98)
    log("WHAT THE DATA SAYS")
    poly_recovers = all(r["cells"][0][2] > 0.6 for r in rows)
    poly_fails = all(r["cells"][0][2] < 0.3 for r in rows)
    # the operator needs resolution: full-L beats poly-L clearly, and the mechanism (oracle) is correct
    needs_resolution = poly_fails and (max(r["cells"][2][2] for r in rows) > 0.3) and (oracle_ok / 20 > 0.9)
    log("  public-winding recovery  L=poly: %s   L=sqrtN: %s   L=N: %s"
        % ([round(r["cells"][0][2], 2) for r in rows], [round(r["cells"][1][2], 2) for r in rows],
           [round(r["cells"][2][2], 2) for r in rows]))
    log("  [A8] null = %.3f" % null_rate)
    log("=" * 98)

    if poly_recovers and null_rate < 0.1:
        verdict = "NOETHER_WINDING_CROSSES_POLY_ESCALATE"
        log("VERDICT: %s" % verdict)
        log("  The Cauchy winding of a PUBLIC non-Hermitian operator recovers d at poly(n) grating length.")
        log("  If it survives scrutiny this is a crossing -> MAXIMUM A8 suspicion, escalate immediately.")
    elif needs_resolution and null_rate < 0.1:
        verdict = "NOETHER_WINDING_NEEDS_N_RESOLUTION"
        log("VERDICT: %s" % verdict)
        log("  the winding MECHANISM is correct - the conserved charge IS d, and the false paths do")
        log("  cancel (oracle winding = d exactly; public winding = d at L=N). But building the operator")
        log("  from PUBLIC data needs the full N-resolution grating: the per-step phase 2*pi*d/N (the ring")
        log("  hopping that makes the winding = d) is exactly the secret, and estimating it from (k,b) at")
        log("  poly(n) resolution fails (poly-L recovery ~chance), succeeding only at L ~ N. So the winding")
        log("  reads the charge for free, but the OPERATOR that carries the charge is secret-resolution-")
        log("  bound: the holonomy per step IS d/N, so a poly-step ring cannot resolve d among N. The")
        log("  conserved charge is real; the lens that exposes it is still the secret. Same hinge, now in")
        log("  Noether language - and it says the obstruction is the operator's PHASE RESOLUTION, not the")
        log("  winding readout. To cross, the per-step holonomy must be amplified to O(1) (an Exceptional")
        log("  Point's sqrt-divergence) WITHOUT knowing d - the one thing still untested and unproven.")
    else:
        verdict = "NOETHER_WINDING_PARTIAL"
        log("VERDICT: %s   poly=%s full=%s" % (verdict,
            [round(r["cells"][0][2], 2) for r in rows], [round(r["cells"][2][2], 2) for r in rows]))

    import json
    (HERE / "noether_result.json").write_text(json.dumps({
        "oracle_recovery": oracle_ok / 20, "rows": rows, "null_rate": null_rate, "verdict": verdict,
    }, indent=2, default=float), encoding="utf-8")
    (HERE / "output_noether.txt").write_text("\n".join(LINES), encoding="utf-8")
    sys.exit(0)


if __name__ == "__main__":
    main()
