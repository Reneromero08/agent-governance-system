"""
run_stage3.py - Stage 3 driver: run every quadrature-synthesis candidate through
the EXISTING no_smuggle_gate.gate(O) at n = 8, 10, 12 and record the full verdict
battery.  Reuses construction.py and no_smuggle_gate.py verbatim (added to sys.path
from the parent fold_audit/ dir); reimplements NOTHING.

Output: stage3_result.json (all metrics + seeds) and a console log.

Discipline: ASCII only; all RNGs seeded; master_seed recorded.  Claim ceiling L4-5.
"""
import json
import os
import sys
import time

import numpy as np

# make the parent fold_audit/ instruments importable (construction, no_smuggle_gate)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in (_PARENT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import no_smuggle_gate as G          # the EXISTING gate (reused verbatim)
import candidates as Q               # this stage's attack battery


MASTER_SEED = 44060611               # same provenance as Stage 1
N_LIST = [8, 10, 12]
N_INSTANCES = 400
N_SHUFFLES = 30

# candidate name -> (callable, expected verdict, one-line mechanism)
CANDIDATES = [
    ("1_hilbert_analytic", Q.O_hilbert_analytic, "FAIL_CHANCE",
     "Hilbert transform of the even binned cos-spectrum yields an odd sequence with no absolute d-sign reference."),
    ("2_double_angle_ladder", Q.O_double_angle, "FAIL_CHANCE",
     "Double-angle identities give only |sin| and even consistency residuals; sin(2t) needs an absent odd channel."),
    ("3_bispectrum", Q.O_bispectrum, "FAIL_CHANCE",
     "Bispectrum of a real-even public spectrum is real (phase 0/pi); the orientation odd-sign is not encoded."),
    ("4_autocorr_asymmetry", Q.O_autocorr_asym, "FAIL_CHANCE",
     "score(x) is even in x (peaks at d and N-d), so its x->N-x antisymmetric part is ~0 by construction."),
    ("5a_gerchberg_saxton_PUBLIC", Q.O_gerchberg_saxton_PUBLIC, "FAIL_CHANCE",
     "Symmetric (public) support forces a real-even fixed point; synthesized phase is 0/pi, no orientation."),
    ("5b_gerchberg_saxton_SMUGGLE", Q.O_gerchberg_saxton_SMUGGLE, "FAIL_SMUGGLE",
     "Support mask reads inst['d'] half-plane (the orientation itself); flips under the fold -> caught."),
    ("6_halfangle_chebyshev", Q.O_halfangle_chebyshev, "FAIL_CHANCE",
     "Chebyshev/half-angle lift is invariant under theta->-theta (d->N-d); branch sign is unrecoverable."),
]


def main():
    t0 = time.time()
    results = {
        "master_seed": MASTER_SEED,
        "n_list": N_LIST,
        "n_instances": N_INSTANCES,
        "n_shuffles": N_SHUFFLES,
        "cells": [],
    }
    print("=" * 78)
    print("STAGE 3 QUADRATURE-SYNTHESIS ATTACK BATTERY")
    print("master_seed=%d  n in %s  n_instances=%d  n_shuffles=%d"
          % (MASTER_SEED, N_LIST, N_INSTANCES, N_SHUFFLES))
    print("=" * 78)

    for ni, n in enumerate(N_LIST):
        N = 1 << n
        M = __import__("construction").M_for(n)
        print("\n### n=%d  (N=%d, M=%d) " % (n, N, M) + "#" * 30)
        for ci, (name, O, expected, mech) in enumerate(CANDIDATES):
            # per-cell seed deterministically derived from master_seed, n, candidate
            seed = (MASTER_SEED + 1009 * n + 31 * ci) & 0x7FFFFFFF
            tic = time.time()
            res = G.gate(O, n, n_instances=N_INSTANCES, seed=seed,
                         n_shuffles=N_SHUFFLES)
            dt = time.time() - tic
            ok = (res["verdict"] == expected)
            print("  [%s] %-28s verdict=%-13s (exp %-13s) %s"
                  % ("OK " if ok else "!! ", name, res["verdict"], expected,
                     "" if ok else "<-- MISMATCH"))
            print("        auc=%.4f null95=%.4f null_mean=%.4f reads_d=%s "
                  "delta=%.3g above_chance=%s  [%.1fs]"
                  % (res["auc"], res["shuffle_null_95"], res["shuffle_null_mean"],
                     res["reads_d"], res["max_fold_delta"], res["above_chance"], dt))
            results["cells"].append({
                "candidate": name,
                "n": n,
                "N": N,
                "M": int(M),
                "seed": int(seed),
                "expected_verdict": expected,
                "verdict": res["verdict"],
                "verdict_matches_expectation": bool(ok),
                "auc": res["auc"],
                "shuffle_null_95": res["shuffle_null_95"],
                "shuffle_null_mean": res["shuffle_null_mean"],
                "invariance_delta": res["max_fold_delta"],
                "reads_d": res["reads_d"],
                "above_chance": res["above_chance"],
                "mechanism": mech,
            })

    # also re-run the gate's own three self-tests at n=10 as a sanity anchor
    print("\n### gate self-test anchor (n=10) " + "#" * 30)
    for name, O, expected in [
        ("selftest_cheat_reads_d", G.O_cheat_reads_d, "FAIL_SMUGGLE"),
        ("selftest_cheat_reads_sin", G.O_cheat_reads_sin, "FAIL_SMUGGLE"),
        ("selftest_useless_even", G.O_useless_even, "FAIL_CHANCE"),
    ]:
        seed = (MASTER_SEED + 9973) & 0x7FFFFFFF
        res = G.gate(O, 10, n_instances=N_INSTANCES, seed=seed, n_shuffles=N_SHUFFLES)
        ok = (res["verdict"] == expected)
        print("  [%s] %-26s verdict=%-13s (exp %s)  auc=%.3f delta=%.3g"
              % ("OK " if ok else "!! ", name, res["verdict"], expected,
                 res["auc"], res["max_fold_delta"]))
        results.setdefault("selftest_anchor", []).append({
            "name": name, "verdict": res["verdict"], "expected": expected,
            "matches": bool(ok), "auc": res["auc"],
            "invariance_delta": res["max_fold_delta"],
        })

    results["elapsed_s"] = time.time() - t0
    # crossing scan: did ANY candidate land PASS_CROSSING?
    crossed = [c for c in results["cells"] if c["verdict"] == "PASS_CROSSING"]
    results["any_crossing"] = len(crossed) > 0
    results["crossings"] = crossed
    all_match = all(c["verdict_matches_expectation"] for c in results["cells"])
    results["all_verdicts_match_expectation"] = all_match

    out = os.path.join(_HERE, "stage3_result.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n" + "=" * 78)
    print("ANY PUBLIC-ONLY CROSSING (PASS_CROSSING)?  %s" % results["any_crossing"])
    print("ALL VERDICTS MATCH EXPECTATION?            %s" % all_match)
    print("elapsed=%.1fs   wrote %s" % (results["elapsed_s"], out))
    print("=" * 78)


if __name__ == "__main__":
    main()
