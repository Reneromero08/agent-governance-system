"""
reaudit_C.py - A8 re-audit of the lone PASS_CROSSING (C_nonreciprocal_chiral, n=14).

The full battery (run_kuramoto.py) flagged ONE PASS_CROSSING out of 20 honest tests
(5 candidates x 4 n). The brief and hardened_gate both REQUIRE a hard re-audit before any
crossing claim. This script runs that re-audit and writes reaudit_C.json.

Decisive logic: C_nonreciprocal_chiral is a pure public-data function (exact d-invariance
delta==0 in every cell), so by the pointwise likelihood identity P(public|d)=P(public|N-d)
its TRUE orientation AUC is exactly 0.5. A real crossing would PERSIST and STRENGTHEN with
more instances/seeds; a finite-sample fluctuation REGRESSES to 0.5 with std ~ 1/sqrt(n).

ASCII only; seeds recorded. Claim ceiling L4-5.
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_FOLD_AUDIT = os.path.abspath(os.path.join(_HERE, "..", "..", "fold_audit"))
_STAGE3 = os.path.join(_FOLD_AUDIT, "stage3")
for _p in (_FOLD_AUDIT, _STAGE3, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hardened_gate as H
import kuramoto_op as OP

MASTER_SEED = 44060611
N = 14
out = {"candidate": "C_nonreciprocal_chiral", "n": N, "master_seed": MASTER_SEED,
       "sweeps": []}


def sweep(n_instances, seeds, tag):
    aucs = []; crosses = 0; rows = []
    for s in seeds:
        r = H.hardened_gate(OP.O_nonreciprocal_chiral, N, n_instances=n_instances,
                            seed=s, n_shuffles=20)
        aucs.append(r["auc"]); crosses += int(r["verdict"] == "PASS_CROSSING")
        rows.append({"seed": int(s), "orient_auc": r["auc"],
                     "orient_null95": r["shuffle_null_95"],
                     "random_fold_auc": r["random_fold_auc"],
                     "invariance_delta": r["max_fold_delta"], "verdict": r["verdict"]})
    rec = {"tag": tag, "n_instances": n_instances, "rows": rows,
           "mean_orient_auc": float(np.mean(aucs)), "std_orient_auc": float(np.std(aucs)),
           "pass_crossing_count": crosses, "n_seeds": len(seeds)}
    out["sweeps"].append(rec)
    print("[%s] n_inst=%d  mean_auc=%.4f std=%.4f  PASS_CROSSING=%d/%d"
          % (tag, n_instances, rec["mean_orient_auc"], rec["std_orient_auc"],
             crosses, len(seeds)))
    return rec


if __name__ == "__main__":
    s240 = [(MASTER_SEED + 1009 * N + 31 * 2 + 7919 * s) & 0x7FFFFFFF for s in range(8)]
    s900 = [(MASTER_SEED + 13 * s + 1) & 0x7FFFFFFF for s in range(5)]
    a = sweep(240, s240, "gate_setting_240")
    b = sweep(900, s900, "large_900")
    # verdict: noise iff mean regresses to ~0.5 AND std shrinks with n
    regressed = abs(b["mean_orient_auc"] - 0.5) < 0.02 and b["std_orient_auc"] < a["std_orient_auc"]
    out["conclusion"] = ("FINITE_SAMPLE_FALSE_POSITIVE: orientation AUC regresses to 0.5 "
                         "with std shrinking ~1/sqrt(n); no crossing."
                         if regressed else "PERSISTS: escalate, do not claim.")
    out["is_false_positive"] = bool(regressed)
    with open(os.path.join(_HERE, "reaudit_C.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("CONCLUSION:", out["conclusion"])
    print("wrote reaudit_C.json")
