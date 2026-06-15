"""
hardened_gate.py - PART B: harden no_smuggle_gate with the RANDOM-PRIVATE-FOLD test
the frontier consult recommended.

The existing gate (no_smuggle_gate.gate) has TWO axes:
  AXIS 1  exact byte-equal d-invariance audit (O(inst) vs O(folded_instance(inst)))
  AXIS 2  held-out orientation AUC vs a per-dataset shuffle null

This module adds the consult's PRIMARY test, which is strictly sharper than AXIS 2 and
needs no orientation labels at all:

  RANDOM-PRIVATE-FOLD TEST
  ------------------------
  Per instance, draw a PRIVATE coin r in {0,1}. Present to the candidate T the public
  data generated for d (if r=0) or for N-d (if r=1), BOTH drawn from the SAME orbit
  {d, N-d}. Require T to predict r from PUBLIC data only.

  Because the two data laws are PROVABLY IDENTICAL (P(D|d)=P(D|N-d) pointwise; the
  bedrock fold), any honest public-only T scores AUC = 0.5 EXACTLY in expectation.
  ANYTHING above 0.5 (beyond the calibrated shuffle null) localizes a side-channel: T
  is reading something that is NOT a pure function of the public data -- i.e. it is
  reading the private label r (== the orientation) directly. This is the cleanest
  possible smuggle detector: it does not even need the orientation bit; it asks whether
  the candidate can distinguish two provably-identical data laws.

  Why this is sharper than the orientation-AUC axis: the orientation AUC can be confounded
  by the FIXED relationship between d's value and the public data across instances (a
  candidate could in principle exploit an even-but-instance-varying feature). The random
  fold removes ALL of that: within a single orbit, r is a fresh private coin independent
  of everything public, so the ONLY way to predict it is to read it. AUC>0.5 here is a
  pure smuggle, full stop.

Verdict logic (hardened):
  PASS_CROSSING : orientation AUC above chance AND random-fold AUC at chance AND exact
                  byte-equal d-invariance (delta==0). A genuine public quadrature synth.
                  (The random-fold test makes this verdict nearly impossible to fake.)
  FAIL_SMUGGLE  : random-fold AUC above its shuffle null, OR exact d-invariance fails
                  (delta>0). The candidate reads the private fold / the secret.
  FAIL_CHANCE   : neither orientation nor random-fold beats chance. An honest even
                  transform that manufactured no bit (the expected bedrock outcome).

ASCII only. All RNGs seeded by caller. Reuses construction.py and no_smuggle_gate.py
verbatim (added to sys.path from the parent fold_audit/ dir); reimplements nothing of
the construction. Claim ceiling L4-5.
"""
import os as _os
import sys as _sys

import numpy as np

# make the parent fold_audit/ instruments importable (construction, no_smuggle_gate)
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PARENT = _os.path.dirname(_HERE)
for _p in (_PARENT, _HERE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import construction as C
import no_smuggle_gate as G        # reuse the existing AUC harness + invariance audit


# ---------------------------------------------------------------------------
# THE RANDOM-PRIVATE-FOLD TEST
# ---------------------------------------------------------------------------
def random_fold_auc(O, n, n_instances=400, seed=0, n_shuffles=30):
    """The consult's random-private-fold test, at FULL strength.

    Per instance: draw a base secret d0 ALWAYS from the lower half [1, N/2) (the orbit
    REPRESENTATIVE), then flip a PRIVATE coin r in {0,1} and publish data for the lower
    half (d_used = d0, r=0) or the upper half (d_used = N - d0, r=1). The candidate O
    must predict r from PUBLIC data only.

    Pinning d0 to the representative makes r EXACTLY the orientation of the published
    secret (r == 1[d_used >= N/2]) -- it is no longer scrambled by d0's own random half.
    So this is a strict, full-strength orientation detector wearing the private-coin
    framing: the two branches are drawn from the SAME orbit {d0, N-d0} and (by the
    bedrock fold P(D|d0)=P(D|N-d0)) have IDENTICAL public laws. An honest public-only O
    therefore scores AUC = 0.5 in expectation; ANY lift above the calibrated shuffle null
    localizes a side-channel -- O is reading the private fold (== the orientation) it must
    not be able to see. Returns (auc, null95, null_mean, above_chance)."""
    rng = np.random.default_rng(seed)
    N = 1 << n
    feats, r_labels = [], []
    for _ in range(n_instances):
        d0 = int(rng.integers(1, N // 2))            # representative in the LOWER half
        r = int(rng.integers(0, 2))                  # PRIVATE coin
        d_used = d0 if r == 0 else (N - d0) % N       # same orbit; r sets the half exactly
        inst = G.make_instance(n, d_used, rng)        # public data for d_used
        f = np.asarray(O(inst), dtype=float).ravel()
        feats.append(f)
        r_labels.append(r)                            # r == orientation of d_used
    X = np.asarray(feats, dtype=float)
    y = np.asarray(r_labels, dtype=int)
    auc = G._held_out_auc(X, y, seed)
    null95, null_mean = G._shuffle_null_auc(X, y, seed, n_shuffles)
    return float(auc), float(null95), float(null_mean), bool(auc > null95)


# ---------------------------------------------------------------------------
# THE HARDENED GATE
# ---------------------------------------------------------------------------
def hardened_gate(O, n, n_instances=400, seed=0, invariance_tol=1e-9, n_shuffles=30,
                  rf_margin=0.05):
    """Run the original gate AND the random-private-fold test; render a hardened verdict.

    The random-fold test is the PRIMARY smuggle detector. The byte-equal d-invariance
    audit is retained as an exact (zero-false-positive) corroborator. A candidate must
    clear BOTH (random-fold at chance AND delta==0) to earn PASS_CROSSING.

    rf_margin: the random-fold AUC must beat its shuffle-null 95th pct by at least this
    EFFECT-SIZE margin to count as a smuggle. Empirically (12-seed sweep at n=8,10) honest
    public-only transforms have (rf_auc - null95) <= 0.016, while true orientation/sin
    smugglers sit at margin ~0.45 -- so 0.05 cleanly separates them with large headroom and
    removes the ~5% borderline false-positive rate of a bare percentile threshold.
    """
    base = G.gate(O, n, n_instances=n_instances, seed=seed,
                  invariance_tol=invariance_tol, n_shuffles=n_shuffles)
    rf_auc, rf_null95, rf_null_mean, _rf_above_bare = random_fold_auc(
        O, n, n_instances=n_instances, seed=seed + 101, n_shuffles=n_shuffles)
    rf_above = bool(rf_auc > rf_null95 + rf_margin)   # margin-robust above-chance

    reads_d = base["reads_d"]                     # exact byte-equal invariance failure
    # orientation lift must clear an effect-size margin (a bare percentile threshold fires
    # ~5% of the time by finite-sample noise -- see autocorr n=10, margin 0.002; over 8
    # seeds that candidate oscillates around chance with delta==0, i.e. it is NOT a
    # crossing). The same margin used for the random-fold axis is applied here.
    orient_above = bool(base["auc"] > base["shuffle_null_95"] + rf_margin)
    smuggle = reads_d or rf_above                 # EITHER exact-fail OR random-fold lift

    if smuggle:
        verdict = "FAIL_SMUGGLE"
    elif orient_above:
        # Orientation lifted by a real margin, random-fold at chance, AND exact byte-equal
        # d-invariance holds (delta==0). On THIS construction the bedrock proof forbids
        # this branch (any public orientation lift would also move the random-fold axis and
        # be caught as a smuggle); reaching it would be an EXTRAORDINARY result demanding a
        # hard re-audit before any claim.
        verdict = "PASS_CROSSING"
    else:
        verdict = "FAIL_CHANCE"

    out = dict(base)
    out.update({
        "verdict": verdict,
        "random_fold_auc": rf_auc,
        "random_fold_null_95": rf_null95,
        "random_fold_null_mean": rf_null_mean,
        "random_fold_above_chance": rf_above,
        "smuggle_flag": bool(smuggle),
        "smuggle_reason": ("exact_d_invariance_delta>0" if reads_d
                           else ("random_fold_AUC_above_chance" if rf_above else "none")),
    })
    return out


# ---------------------------------------------------------------------------
# Self-test when run directly: the hardened gate MUST flag the known cheats and the
# phase-retrieval support-constraint smuggle, and pass the useless-even transform.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import os
    import sys
    import time

    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _HERE)
    import candidates as Q

    MASTER_SEED = 44060611
    N_INST = 300
    N_SHUF = 25
    out = {"master_seed": MASTER_SEED, "n_instances": N_INST, "cells": []}
    print("=" * 88)
    print("PART B - HARDENED GATE (random-private-fold) self-test + phase-retrieval smuggle demo")
    print("=" * 88)

    # The decisive pairing: the PUBLIC phase-retrieval (honest, symmetric support) MUST
    # land FAIL_CHANCE; the SMUGGLE phase-retrieval (support mask reads d's half) MUST be
    # flagged FAIL_SMUGGLE by the hardened gate. Plus the gate's own cheats and useless-even.
    cases = [
        ("gate_cheat_reads_d", G.O_cheat_reads_d, "FAIL_SMUGGLE"),
        ("gate_cheat_reads_sin", G.O_cheat_reads_sin, "FAIL_SMUGGLE"),
        ("gate_useless_even", G.O_useless_even, "FAIL_CHANCE"),
        ("phase_retrieval_PUBLIC(symmetric_support)", Q.O_gerchberg_saxton_PUBLIC, "FAIL_CHANCE"),
        ("phase_retrieval_SMUGGLE(support=d_half)", Q.O_gerchberg_saxton_SMUGGLE, "FAIL_SMUGGLE"),
    ]
    t0 = time.time()
    for n in (8, 10):
        print("\n### n=%d ###" % n)
        for ci, (name, O, expected) in enumerate(cases):
            seed = (MASTER_SEED + 1009 * n + 31 * ci) & 0x7FFFFFFF
            tic = time.time()
            res = hardened_gate(O, n, n_instances=N_INST, seed=seed, n_shuffles=N_SHUF)
            dt = time.time() - tic
            ok = res["verdict"] == expected
            print("  [%s] %-42s verdict=%-13s (exp %-13s)" %
                  ("OK " if ok else "!! ", name, res["verdict"], expected))
            print("        orient_auc=%.3f (null95=%.3f)  random_fold_auc=%.3f (null95=%.3f)"
                  "  delta=%.3g  reason=%s  [%.1fs]" %
                  (res["auc"], res["shuffle_null_95"], res["random_fold_auc"],
                   res["random_fold_null_95"], res["max_fold_delta"],
                   res["smuggle_reason"], dt))
            out["cells"].append({
                "name": name, "n": n, "seed": int(seed), "expected": expected,
                "verdict": res["verdict"], "matches": bool(ok),
                "orientation_auc": res["auc"], "orientation_null95": res["shuffle_null_95"],
                "random_fold_auc": res["random_fold_auc"],
                "random_fold_null95": res["random_fold_null_95"],
                "random_fold_above_chance": res["random_fold_above_chance"],
                "invariance_delta": res["max_fold_delta"],
                "smuggle_reason": res["smuggle_reason"],
            })
    out["elapsed_s"] = time.time() - t0
    out["all_match"] = all(c["matches"] for c in out["cells"])
    with open(os.path.join(_HERE, "hardened_gate_result.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\n" + "=" * 88)
    print("ALL HARDENED-GATE VERDICTS MATCH EXPECTATION?  %s   (%.1fs)"
          % (out["all_match"], out["elapsed_s"]))
    print("wrote hardened_gate_result.json")
    print("=" * 88)
