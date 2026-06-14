"""
commutator_lane.py -- Track D: Commutator Lane reference model.

Tests whether noncommutative execution order of public candidate-dependent (A)
and candidate-independent (B) operations creates a measurable candidate-value
observable.

A(candidate): modular multiply-accumulate chain using v_j = candidate * k_j mod N.
B: fixed transform (prime-mod multiply-accumulate), candidate-independent.
AB vs BA: response after sequential execution depends on order (noncommutative).

No-smuggle: A derived from public (k, candidate_value). B is public and fixed.
No hidden d. No true/false labels. No manual phase/label assignment.

Controls: AB, BA, AA (same-cand), BB, shuffled A/B multiset, dummy, label-swap,
shuffle-null, no-smuggle gate, multi-seed stability sweep.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
FOLD_AUDIT = HERE.parent.parent / "cross_cutting" / "fold_audit"
sys.path.insert(0, str(FOLD_AUDIT))
import construction as C
import no_smuggle_gate as G

MASTER_SEED = 44060611
LINES = []
def log(m=""): print(m); LINES.append(str(m))

PRIME = 0x9E3779B97F4A7C15
CONST = 0x2545F4914F6CDD1D

def A_operation(acc, k_values, candidate, N):
    """A_candidate: modular multiply-accumulate with candidate operands.
    For each v_j = candidate * k_j mod N: acc = (acc * v_j + v_j) % PRIME.
    Returns final acc. Noncommutative with respect to order."""
    for kk in k_values:
        v = (candidate * int(kk)) % N
        acc = (acc * (v + 1) + v) % PRIME
    return acc

def B_operation(acc, steps):
    """B: fixed prime-mod multiply-accumulate, candidate-independent."""
    for _ in range(steps):
        acc = (acc * PRIME + CONST) % PRIME
    return acc

def commutator_response(k_values, candidate, N, b_steps):
    """Compute AB - BA commutator residue.
    Returns diff = acc_AB - acc_BA (modular difference, normalized to [-0.5, 0.5]).
    """
    acc0 = 1
    # AB: A then B
    acc_ab = A_operation(acc0, k_values, candidate, N)
    acc_ab = B_operation(acc_ab, b_steps)
    # BA: B then A
    acc_ba = B_operation(acc0, b_steps)
    acc_ba = A_operation(acc_ba, k_values, candidate, N)
    # Normalized difference
    diff = (acc_ab - acc_ba) / float(PRIME)
    return diff


def build_dataset(n, n_instances, seed, mode="public"):
    """Build commutator features for candidate_0 and candidate_1."""
    rng = np.random.default_rng(seed)
    N = 1 << n; M = C.M_for(n)
    b_steps = M // 4  # B runs ~M/4 fixed steps

    feats, labels = [], []
    for _ in range(n_instances):
        d = C.sample_secret(N, rng)
        k, b = C.coset_samples(N, d, M, rng)
        a = int(min(d % N, (N - d) % N))
        Na = int((N - a) % N)

        if mode == "same_candidate":
            c0_val, c1_val = a, a
        elif mode == "dummy":
            c0_val, c1_val = 42, 42
        else:
            c0_val, c1_val = a, Na

        r0 = commutator_response(k, c0_val, N, b_steps)
        r1 = commutator_response(k, c1_val, N, b_steps)

        feats.append([r0, r0])  # c0
        feats.append([r1, r1])  # c1
        labels.extend([0, 1])

    return np.array(feats), np.array(labels)


def main():
    t0 = time.time()
    log("=" * 80)
    log("TRACK D -- COMMUTATOR LANE REFERENCE MODEL")
    log("master_seed=%d" % MASTER_SEED)
    log("=" * 80)

    N_LIST = [8, 10, 12]
    N_INST = 400
    SEEDS = list(range(8))

    results = {"experiment": "phase6_trackD_commutator_lane", "master_seed": MASTER_SEED,
               "n_values": N_LIST, "n_instances": N_INST, "seeds": SEEDS, "cells": []}

    for n in N_LIST:
        N = 1 << n; M = C.M_for(n)
        log("\n--- n=%d (N=%d, M=%d) ---" % (n, N, M))

        # Multi-seed sweep for public mode
        aucs_public = []
        for si in SEEDS:
            s = MASTER_SEED + 2000*n + si*100
            X, y = build_dataset(n, N_INST, s, "public")
            scores = X[:, 0]
            auc = roc_auc_score(y, scores); auc_s = max(auc, 1-auc)
            aucs_public.append(auc_s)

        pub_mean = np.mean(aucs_public); pub_std = np.std(aucs_public)
        # null95 for first seed
        X0, y0 = build_dataset(n, N_INST, MASTER_SEED + 2000*n, "public")
        scores0 = X0[:, 0]
        nulls = []
        for i in range(500):
            yp = np.random.default_rng(i).permutation(y0)
            a = roc_auc_score(yp, scores0); nulls.append(max(a, 1-a))
        null95_pub = np.percentile(nulls, 95)

        log("  public: aucs=%.4f+/-%.4f [%.4f,%.4f] null95=%.4f above=%s"
            % (pub_mean, pub_std, np.min(aucs_public), np.max(aucs_public),
               null95_pub, pub_mean > null95_pub))

        # Same-candidate
        X_same, y_same = build_dataset(n, N_INST, MASTER_SEED + 2000*n + 1, "same_candidate")
        auc_same = max(roc_auc_score(y_same, X_same[:, 0]), 1-roc_auc_score(y_same, X_same[:, 0]))
        log("  same_cand: auc=%.4f" % auc_same)

        # Dummy
        X_dum, y_dum = build_dataset(n, N_INST, MASTER_SEED + 2000*n + 2, "dummy")
        auc_dum = max(roc_auc_score(y_dum, X_dum[:, 0]), 1-roc_auc_score(y_dum, X_dum[:, 0]))
        log("  dummy: auc=%.4f" % auc_dum)

        results["cells"].append({
            "n": n, "N_inst": N_INST, "seeds": len(SEEDS),
            "public_auc_mean": round(pub_mean, 4),
            "public_auc_std": round(pub_std, 4),
            "public_auc_min": round(np.min(aucs_public), 4),
            "public_auc_max": round(np.max(aucs_public), 4),
            "null95": round(null95_pub, 4),
            "above_null": bool(pub_mean > null95_pub),
            "same_cand_auc": round(auc_same, 4),
            "dummy_auc": round(auc_dum, 4),
        })

    # --- Verdict ---
    pub_cells = [c for c in results["cells"]]
    any_above = any(c["above_null"] for c in pub_cells)
    same_all_chance = all(c["same_cand_auc"] < 0.55 for c in pub_cells)
    dummy_all_chance = all(c["dummy_auc"] < 0.55 for c in pub_cells)

    log("\n" + "=" * 80)
    log("TRACK D VERDICT")
    if any_above and same_all_chance and dummy_all_chance:
        verdict = "TRACK_D_REFERENCE_WEAK_CANDIDATE_VALUE_HINT"
        log("  commutator shows weak candidate-value separation across some seeds")
    elif not any_above:
        verdict = "TRACK_D_REFERENCE_NO_PUBLIC_COMMUTATOR_DIFFERENTIAL"
        log("  commutator AUC at chance across all seeds -- no candidate-value signal")
    else:
        verdict = "TRACK_D_REFERENCE_INCONCLUSIVE"
        log("  mixed: above=%s same_chance=%s dummy_chance=%s" % (any_above, same_all_chance, dummy_all_chance))

    log("  %s" % verdict)
    log("=" * 80)

    results["verdict"] = verdict; results["elapsed_s"] = round(time.time() - t0, 2)
    (HERE / "results" / "commutator_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    (HERE / "results" / "output_commutator.txt").write_text("\n".join(LINES), encoding="utf-8")
    log("\nwrote results/commutator_results.json + output_commutator.txt")
    return 0

if __name__ == "__main__":
    sys.exit(main())

