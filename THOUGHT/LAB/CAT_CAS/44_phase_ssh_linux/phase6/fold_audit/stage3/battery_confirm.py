"""
battery_confirm.py - PART C: bounded empirical echo of the bedrock proof.

Run the EXISTING candidates.py battery (Hilbert/analytic-signal, double-angle ladder,
bispectrum, autocorrelation, half-angle Chebyshev, and both Gerchberg-Saxton variants)
through the HARDENED gate (hardened_gate.py, with the random-private-fold test) at
n = 8, 10 ONLY. Fast, foreground, seeded.

Expected (the bedrock prediction on the REAL construction):
  every honest public-only candidate -> FAIL_CHANCE  (manufactured no orientation bit)
  the designated smuggle (support = d's half) -> FAIL_SMUGGLE  (caught by the gate)

This is the empirical confirmation that Part A's analytic verdict (orbit-only public
interface) holds in practice: no transform of the public even data lifts orientation.

ASCII only; all RNGs seeded; master_seed recorded. Claim ceiling L4-5.
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in (_PARENT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import candidates as Q
import hardened_gate as H

MASTER_SEED = 44060611
N_LIST = [8, 10]
N_INSTANCES = 300
N_SHUFFLES = 25

CANDIDATES = [
    ("1_hilbert_analytic", Q.O_hilbert_analytic, "FAIL_CHANCE"),
    ("2_double_angle_ladder", Q.O_double_angle, "FAIL_CHANCE"),
    ("3_bispectrum", Q.O_bispectrum, "FAIL_CHANCE"),
    ("4_autocorr_asymmetry", Q.O_autocorr_asym, "FAIL_CHANCE"),
    ("5a_gerchberg_saxton_PUBLIC", Q.O_gerchberg_saxton_PUBLIC, "FAIL_CHANCE"),
    ("5b_gerchberg_saxton_SMUGGLE", Q.O_gerchberg_saxton_SMUGGLE, "FAIL_SMUGGLE"),
    ("6_halfangle_chebyshev", Q.O_halfangle_chebyshev, "FAIL_CHANCE"),
]


def main():
    t0 = time.time()
    out = {"master_seed": MASTER_SEED, "n_list": N_LIST, "n_instances": N_INSTANCES,
           "gate": "hardened (random-private-fold + exact d-invariance)", "cells": []}
    print("=" * 92)
    print("PART C - BOUNDED BATTERY CONFIRMATION through the HARDENED gate (n in {8,10})")
    print("master_seed=%d  n_instances=%d  n_shuffles=%d" % (MASTER_SEED, N_INSTANCES, N_SHUFFLES))
    print("=" * 92)
    for n in N_LIST:
        N = 1 << n
        import construction as C
        print("\n### n=%d  (N=%d, M=%d) ###" % (n, N, C.M_for(n)))
        for ci, (name, O, expected) in enumerate(CANDIDATES):
            seed = (MASTER_SEED + 1009 * n + 31 * ci) & 0x7FFFFFFF
            tic = time.time()
            res = H.hardened_gate(O, n, n_instances=N_INSTANCES, seed=seed,
                                  n_shuffles=N_SHUFFLES)
            dt = time.time() - tic
            ok = res["verdict"] == expected
            print("  [%s] %-28s verdict=%-13s (exp %-13s)" %
                  ("OK " if ok else "!! ", name, res["verdict"], expected))
            print("        orient_auc=%.3f(null95=%.3f) rf_auc=%.3f(null95=%.3f) "
                  "delta=%.3g reason=%s [%.1fs]" %
                  (res["auc"], res["shuffle_null_95"], res["random_fold_auc"],
                   res["random_fold_null_95"], res["max_fold_delta"],
                   res["smuggle_reason"], dt))
            out["cells"].append({
                "candidate": name, "n": n, "N": N, "seed": int(seed),
                "expected": expected, "verdict": res["verdict"], "matches": bool(ok),
                "orientation_auc": res["auc"], "orientation_null95": res["shuffle_null_95"],
                "random_fold_auc": res["random_fold_auc"],
                "random_fold_null95": res["random_fold_null_95"],
                "invariance_delta": res["max_fold_delta"],
                "smuggle_reason": res["smuggle_reason"],
            })
    out["elapsed_s"] = time.time() - t0
    out["all_match"] = all(c["matches"] for c in out["cells"])
    out["any_crossing"] = any(c["verdict"] == "PASS_CROSSING" for c in out["cells"])
    with open(os.path.join(_HERE, "battery_confirm_result.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\n" + "=" * 92)
    print("ANY PASS_CROSSING?  %s" % out["any_crossing"])
    print("ALL VERDICTS MATCH EXPECTATION?  %s   (%.1fs)" % (out["all_match"], out["elapsed_s"]))
    print("wrote battery_confirm_result.json")
    print("=" * 92)


if __name__ == "__main__":
    main()
