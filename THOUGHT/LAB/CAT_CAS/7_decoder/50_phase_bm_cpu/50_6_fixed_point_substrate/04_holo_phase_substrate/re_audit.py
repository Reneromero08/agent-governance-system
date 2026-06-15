"""re_audit.py - A8 multi-seed re-audit of the conjugate-quadrature orientation operator
at n=12,14 (the largest cells, where a finite-sample flirt above the bare null appeared).
Mirrors the Kuramoto n=14 false-positive lesson: confirm the orientation AUC oscillates
around chance with delta==0 across many seeds and NEVER earns PASS_CROSSING."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import holo_phase_substrate as S   # configures sys.path for fold_audit + stage3 on import
Hg = S.Hg

SEEDS = [11, 23, 37, 51, 67, 83, 101, 119]
out = {"reaudit": "orientation_homodyne_quadrature", "seeds": SEEDS, "cells": []}
print("A8 MULTI-SEED RE-AUDIT - orientation operator (homodyne conjugate quadrature)")
for n in (12, 14):
    aucs, rfs, verdicts = [], [], []
    for s in SEEDS:
        r = Hg.hardened_gate(S.O_homodyne_quadrature, n, n_instances=200,
                             seed=(900001 + 777 * n + 13 * s) & 0x7FFFFFFF, n_shuffles=20)
        aucs.append(r["auc"]); rfs.append(r["random_fold_auc"]); verdicts.append(r["verdict"])
    rec = {"n": n, "orient_auc_mean": float(np.mean(aucs)), "orient_auc_std": float(np.std(aucs)),
           "orient_auc_min": float(np.min(aucs)), "orient_auc_max": float(np.max(aucs)),
           "rf_auc_mean": float(np.mean(rfs)), "rf_auc_max": float(np.max(rfs)),
           "all_fail_chance": all(v == "FAIL_CHANCE" for v in verdicts),
           "any_crossing": any(v == "PASS_CROSSING" for v in verdicts),
           "verdicts": verdicts, "aucs": [float(a) for a in aucs]}
    out["cells"].append(rec)
    print("  n=%2d  orient_auc mean=%.3f std=%.3f [min %.3f max %.3f]  rf mean=%.3f  all_FAIL_CHANCE=%s  any_crossing=%s"
          % (n, rec["orient_auc_mean"], rec["orient_auc_std"], rec["orient_auc_min"],
             rec["orient_auc_max"], rec["rf_auc_mean"], rec["all_fail_chance"], rec["any_crossing"]))
with open(os.path.join(HERE, "reaudit_result.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=float)
print("wrote reaudit_result.json")
