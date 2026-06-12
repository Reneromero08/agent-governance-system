"""
run_kuramoto.py - drive the Kuramoto / CGLE resonance candidates through the AUDITED
hardened gate (random-private-fold + exact d-invariance + orientation AUC), measure the
cost scaling, and check unoriented-set recovery. Reuses construction.py,
no_smuggle_gate.py, hardened_gate.py VERBATIM (added to sys.path). Reimplements nothing
of the construction or the audit. ASCII only; all RNGs seeded; master_seed recorded.

Outputs kuramoto_result.json + console log. Claim ceiling L4-5.

Three measurements (the brief's a/b/c):
  (a) reads o?    held-out orientation AUC and the random-private-fold AUC vs null, via
                  hardened_gate. PASS_CROSSING only if a public-only O lifts AUC AND the
                  random-fold stays at chance AND delta==0.
  (b) cost?       full-O(N) resonance peak scan time + op-count, and Kuramoto integration
                  time, across n=8,10,12,14: poly(n) per network step vs the O(N) readout.
  (c) no-smuggle? exact d-invariance delta from the gate; smuggle control must FAIL.
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_FOLD_AUDIT = os.path.abspath(os.path.join(_HERE, "..", "..", "fold_audit"))
_STAGE3 = os.path.join(_FOLD_AUDIT, "stage3")
for _p in (_FOLD_AUDIT, _STAGE3, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import construction as C
import no_smuggle_gate as G
import hardened_gate as H
import kuramoto_op as OP


MASTER_SEED = 44060611
N_LIST = [8, 10, 12, 14]
N_INSTANCES = 240
N_SHUFFLES = 20

CANDIDATES = [
    ("A_resonance_phase", OP.O_resonance_phase, "FAIL_CHANCE",
     "Static order-parameter Z(x): |Z| peaks at {d,N-d} (even); arg Z is conjugation-antisymmetric across the fold."),
    ("B_kuramoto_orderparam", OP.O_kuramoto_orderparam, "FAIL_CHANCE",
     "Reciprocal mean-field Kuramoto; psi (order-param phase) tested vs R (magnitude). omega,b even in d -> fold-invariant flow."),
    ("C_nonreciprocal_chiral", OP.O_nonreciprocal_chiral, "FAIL_CHANCE",
     "Sakaguchi chiral phase-lag alpha (non-Hermitian). Chirality is global, orientation-independent: reads 'directed current', not which half."),
    ("D_phase_estimation_dyadic", OP.O_phase_estimation_dyadic, "FAIL_CHANCE",
     "SPEC-1C dyadic phase estimation on PUBLIC data: quadrature absent -> phases 0/pi -> information-empty."),
    ("Z_quadrature_SMUGGLE", OP.O_quadrature_smuggle, "FAIL_SMUGGLE",
     "Reads genuine sin(2 pi k d/N) from hidden d. Sensitivity control: must lift AUC and be caught (delta>0)."),
]


def run_gate_battery(out):
    print("=" * 92)
    print("KURAMOTO / CGLE RESONANCE - hardened-gate battery")
    print("master_seed=%d  n in %s  n_instances=%d  n_shuffles=%d"
          % (MASTER_SEED, N_LIST, N_INSTANCES, N_SHUFFLES))
    print("=" * 92)
    for n in N_LIST:
        N = 1 << n; M = C.M_for(n)
        print("\n### n=%d  (N=%d, M=%d) " % (n, N, M) + "#" * 40)
        for ci, (name, O, expected, mech) in enumerate(CANDIDATES):
            seed = (MASTER_SEED + 1009 * n + 31 * ci) & 0x7FFFFFFF
            tic = time.time()
            res = H.hardened_gate(O, n, n_instances=N_INSTANCES, seed=seed,
                                  n_shuffles=N_SHUFFLES)
            dt = time.time() - tic
            ok = (res["verdict"] == expected)
            print("  [%s] %-26s verdict=%-13s (exp %-13s) %s"
                  % ("OK " if ok else "!! ", name, res["verdict"], expected,
                     "" if ok else "<-- MISMATCH"))
            print("        orient_auc=%.4f (null95=%.4f)  rand_fold_auc=%.4f (null95=%.4f)"
                  "  delta=%.3g  reason=%s  [%.1fs]"
                  % (res["auc"], res["shuffle_null_95"], res["random_fold_auc"],
                     res["random_fold_null_95"], res["max_fold_delta"],
                     res["smuggle_reason"], dt))
            out["gate_cells"].append({
                "candidate": name, "n": n, "N": N, "M": int(M), "seed": int(seed),
                "expected_verdict": expected, "verdict": res["verdict"],
                "matches": bool(ok), "orientation_auc": res["auc"],
                "orientation_null95": res["shuffle_null_95"],
                "random_fold_auc": res["random_fold_auc"],
                "random_fold_null95": res["random_fold_null_95"],
                "random_fold_above_chance": res["random_fold_above_chance"],
                "invariance_delta": res["max_fold_delta"],
                "reads_d": res["reads_d"], "smuggle_reason": res["smuggle_reason"],
                "elapsed_s": dt, "mechanism": mech,
            })


def run_cost_scan(out, n_trials=12):
    """Price (b): the full-O(N) resonance peak readout vs the poly Kuramoto step. For each
    n, time the argmax_x |Z(x)| scan over the FULL Z_N grid (the only public route to the
    resonance peak, since the dyadic phase-estimation shortcut needs the absent
    quadrature) and the per-network Kuramoto integration. Report scaling."""
    print("\n" + "=" * 92)
    print("COST SCAN (b): full-O(N) resonance readout vs poly Kuramoto step")
    print("  n   N        M     scan_ops(M*N)   scan_s/trial   kuramoto_s/trial   peak_hits_set")
    print("-" * 92)
    rng = np.random.default_rng(MASTER_SEED + 555)
    for n in N_LIST:
        N = 1 << n; M = C.M_for(n)
        scan_times = []; kur_times = []; hits = 0
        for _ in range(n_trials):
            d = C.sample_secret(N, rng)
            inst = G.make_instance(n, d, rng)
            # full O(N) resonance scan -> peak (the readout-to-d cost)
            t0 = time.time()
            px, hit = OP.resonance_recovers_set(inst)
            scan_times.append(time.time() - t0)
            hits += int(hit)
            # poly Kuramoto integration (per-network step is O(M))
            omega = 2.0 * np.pi * inst["k"].astype(float) / N
            theta0 = np.random.default_rng(OP._public_seed(inst["k"], inst["b"], N)).uniform(-np.pi, np.pi, size=M)
            t1 = time.time()
            OP.kuramoto_meanfield(omega, inst["b"], K=2.5, alpha=0.0, T=120, dt=0.05, theta0=theta0)
            kur_times.append(time.time() - t1)
        st = float(np.median(scan_times)); kt = float(np.median(kur_times))
        print("  %-3d %-8d %-5d %-15d %-14.4g %-18.4g %d/%d"
              % (n, N, M, M * N, st, kt, hits, n_trials))
        out["cost_scan"].append({
            "n": n, "N": N, "M": int(M), "scan_ops_M_times_N": int(M * N),
            "scan_s_per_trial": st, "kuramoto_s_per_trial": kt,
            "peak_hits_set": hits, "n_trials": n_trials,
        })


def run_recovery_check(out, n_instances=200):
    """The 'alive' test: does the resonance MAGNITUDE recover the unoriented SET {d,N-d}?
    A high hit-rate proves the even channel works and the failure is SPECIFICALLY the
    orientation bit (not a dead instrument)."""
    print("\n" + "=" * 92)
    print("UNORIENTED-SET RECOVERY (alive test): argmax_x |Z(x)| in {d, N-d}?")
    print("  n   N        hit_rate (set recovered)   note")
    print("-" * 92)
    rng = np.random.default_rng(MASTER_SEED + 999)
    for n in N_LIST:
        N = 1 << n
        hits = 0
        for _ in range(n_instances):
            d = C.sample_secret(N, rng)
            inst = G.make_instance(n, d, rng)
            _, hit = OP.resonance_recovers_set(inst)
            hits += int(hit)
        rate = hits / n_instances
        print("  %-3d %-8d %.3f                      even channel recovers the SET, not o"
              % (n, N, rate))
        out["recovery"].append({"n": n, "N": N, "hit_rate": rate, "n_instances": n_instances})


def main():
    t0 = time.time()
    out = {"master_seed": MASTER_SEED, "n_list": N_LIST, "n_instances": N_INSTANCES,
           "n_shuffles": N_SHUFFLES, "gate_cells": [], "cost_scan": [], "recovery": []}
    run_gate_battery(out)
    run_recovery_check(out)
    run_cost_scan(out)

    out["elapsed_s"] = time.time() - t0
    honest = [c for c in out["gate_cells"] if not c["candidate"].startswith("Z_")]
    out["any_public_crossing"] = any(c["verdict"] == "PASS_CROSSING" for c in honest)
    out["all_match"] = all(c["matches"] for c in out["gate_cells"])
    out["smuggle_caught"] = all(c["verdict"] == "FAIL_SMUGGLE"
                                for c in out["gate_cells"] if c["candidate"].startswith("Z_"))
    with open(os.path.join(_HERE, "kuramoto_result.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)

    print("\n" + "=" * 92)
    print("ANY PUBLIC-ONLY CROSSING (PASS_CROSSING)?   %s" % out["any_public_crossing"])
    print("ALL VERDICTS MATCH EXPECTATION?             %s" % out["all_match"])
    print("SMUGGLE SENSITIVITY CONTROL CAUGHT?         %s" % out["smuggle_caught"])
    print("elapsed=%.1fs  wrote kuramoto_result.json" % out["elapsed_s"])
    print("=" * 92)


if __name__ == "__main__":
    main()
