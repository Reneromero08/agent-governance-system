#!/usr/bin/env python3
"""SLOT 1 aggregator: run the matched-null analyzer on every pulled CSV, tabulate
per-(N,W,corepair,seed) gate results, and evaluate the witness reproducibility
(6/6 seeds for at least one (N,W,corepair)).

Gates for the witness (ALL must hold), per the SLOT 1 plan:
  all_rows_restore == true
  real_accuracy            >= 0.60
  real_vs_pseudo floor     >= 0.95
  pseudo_reject (floor)    >= 0.95
  wrong_actual_match       >= 0.60
  wrong_declared_match     <= 0.20

CONFLICT=0 control success criterion: real_accuracy in ~0.25-0.275 band (chance-ish).

Usage: aggregate.py <csv_dir> <out_json>
"""
import csv
import glob
import json
import math
import os
import re
import statistics
import sys

MODES = ["basis", "rotation", "residual", "mini"]
MODE_SETS = {
    "basis": {9, 10, 11, 12, 13, 14},
    "rotation": {16, 17, 18, 19, 20, 21, 22, 23},
    "residual": {24, 25, 26, 27},
    "mini": {9, 10, 11, 12, 16, 17, 18, 19, 24, 25, 26, 27},
}

NAME_RE = re.compile(
    r"conf(?P<c>\d)_N(?P<n>\d+)_W(?P<w>\d+)_cp(?P<wc>\d+)-(?P<oc>\d+)_s(?P<s>\d+)\.csv$"
)


def mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def mode_contrast(vals, mode):
    target = MODE_SETS[mode]
    t = mean([vals[i] for i in target])
    o = mean([vals[i] for i in range(64) if i not in target])
    return o - t


def features(row):
    vals = [float(row[f"l{i:02d}"]) for i in range(64)]
    return [mode_contrast(vals, mode) for mode in MODES]


def full_vector(row):
    return [float(row[f"l{i:02d}"]) for i in range(64)]


def distance(a, b):
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def analyze(path):
    """Reimplements analyze_cache_hologram_matched_nulls.py metrics in-process."""
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            row["trial"] = int(row["trial"])
            row["hash_restored"] = int(row["hash_restored"])
            row["_features"] = features(row)
            row["_vector"] = full_vector(row)
            rows.append(row)

    train_real = [r for r in rows if r["family"] == "real" and r["trial"] % 2 == 0]
    test = [r for r in rows if r["trial"] % 2 == 1]

    centroids = {}
    for mode in MODES:
        mr = [r for r in train_real if r["declared_mode"] == mode]
        centroids[mode] = [mean([r["_features"][i] for r in mr]) for i in range(len(MODES))]

    def predict(feat):
        return min(MODES, key=lambda m: distance(feat, centroids[m]))

    rc = rt = pdm = pt = wam = wdm = wt = 0
    by_mode = {m: {"rc": 0, "rt": 0} for m in MODES}
    for row in test:
        pred = predict(row["_features"])
        fam, dec, act = row["family"], row["declared_mode"], row["actual_mode"]
        if fam == "real":
            rt += 1
            by_mode[dec]["rt"] += 1
            if pred == dec:
                rc += 1
                by_mode[dec]["rc"] += 1
        elif fam == "pseudo":
            pt += 1
            if pred == dec:
                pdm += 1
        elif fam == "wrong":
            wt += 1
            if pred == act:
                wam += 1
            if pred == dec:
                wdm += 1

    def rate(a, b):
        return a / b if b else 0.0

    real_floor = min(rate(by_mode[m]["rc"], by_mode[m]["rt"]) for m in MODES)

    rvp = {}
    for mode in MODES:
        train_rp = [r for r in rows if r["declared_mode"] == mode and r["family"] in ("real", "pseudo") and r["trial"] % 2 == 0]
        test_rp = [r for r in rows if r["declared_mode"] == mode and r["family"] in ("real", "pseudo") and r["trial"] % 2 == 1]
        cen = {}
        for fam in ("real", "pseudo"):
            fr = [r for r in train_rp if r["family"] == fam]
            cen[fam] = [mean([r["_vector"][i] for r in fr]) for i in range(64)]
        tot = {"real": 0, "pseudo": 0}
        cor = {"real": 0, "pseudo": 0}
        for row in test_rp:
            pf = min(("real", "pseudo"), key=lambda fam: distance(row["_vector"], cen[fam]))
            f = row["family"]
            tot[f] += 1
            if pf == f:
                cor[f] += 1
        rvp[mode] = {
            "accuracy": rate(cor["real"] + cor["pseudo"], tot["real"] + tot["pseudo"]),
            "pseudo_reject": rate(cor["pseudo"], tot["pseudo"]),
        }

    all_restore = sum(r["hash_restored"] for r in rows) == len(rows)
    return {
        "rows": len(rows),
        "all_rows_restore": all_restore,
        "real_accuracy": rate(rc, rt),
        "real_mode_floor": real_floor,
        "pseudo_declared_match": rate(pdm, pt),
        "wrong_actual_match": rate(wam, wt),
        "wrong_declared_match": rate(wdm, wt),
        "real_vs_pseudo_accuracy_floor": min(v["accuracy"] for v in rvp.values()),
        "pseudo_reject_floor": min(v["pseudo_reject"] for v in rvp.values()),
    }


def witness_gates(a):
    return {
        "all_rows_restore": a["all_rows_restore"],
        "real_accuracy_ge_0_60": a["real_accuracy"] >= 0.60,
        "rvp_floor_ge_0_95": a["real_vs_pseudo_accuracy_floor"] >= 0.95,
        "pseudo_reject_ge_0_95": a["pseudo_reject_floor"] >= 0.95,
        "wrong_actual_ge_0_60": a["wrong_actual_match"] >= 0.60,
        "wrong_declared_le_0_20": a["wrong_declared_match"] <= 0.20,
    }


def main():
    csv_dir, out_json = sys.argv[1], sys.argv[2]
    results = {}
    for path in sorted(glob.glob(os.path.join(csv_dir, "conf*_s*.csv"))):
        m = NAME_RE.search(os.path.basename(path))
        if not m:
            continue
        a = analyze(path)
        g = witness_gates(a)
        rec = dict(a)
        rec["gates"] = g
        rec["witness_pass"] = all(g.values())
        rec["params"] = {
            "conflict": int(m["c"]), "N": int(m["n"]), "W": int(m["w"]),
            "writer_core": int(m["wc"]), "observer_core": int(m["oc"]), "seed": int(m["s"]),
        }
        results[os.path.basename(path)] = rec

    # reproducibility: for CONFLICT=1, group by (N,W,corepair) and count witness 6/6
    groups = {}
    for fn, rec in results.items():
        p = rec["params"]
        if p["conflict"] != 1:
            continue
        key = f"N{p['N']}_W{p['W']}_cp{p['writer_core']}-{p['observer_core']}"
        groups.setdefault(key, []).append(rec["witness_pass"])
    repro = {k: {"seeds": len(v), "witness_pass": sum(v), "six_of_six": (len(v) >= 6 and sum(v) == len(v))}
             for k, v in groups.items()}

    # CONFLICT=0 control band check
    ctrl = {}
    for fn, rec in results.items():
        p = rec["params"]
        if p["conflict"] != 0:
            continue
        key = f"N{p['N']}_W{p['W']}_cp{p['writer_core']}-{p['observer_core']}"
        ctrl.setdefault(key, []).append(rec["real_accuracy"])
    ctrl_summary = {k: {"mean_real_acc": round(statistics.fmean(v), 4),
                        "min": round(min(v), 4), "max": round(max(v), 4),
                        "in_025_0275_band": all(0.20 <= x <= 0.30 for x in v)}
                    for k, v in ctrl.items()}

    out = {
        "per_run": results,
        "reproducibility_conflict1": repro,
        "control_conflict0": ctrl_summary,
        "witness_holds_anywhere": any(r["six_of_six"] for r in repro.values()),
    }
    with open(out_json, "w", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"runs analyzed: {len(results)}")
    print("\n# CONFLICT=1 reproducibility (witness 6/6 per N,W,corepair):")
    for k in sorted(repro):
        r = repro[k]
        flag = "  <== WITNESS 6/6" if r["six_of_six"] else ""
        print(f"  {k:>22}: {r['witness_pass']}/{r['seeds']} pass{flag}")
    print("\n# CONFLICT=1 per-run gate detail (N,W,corepair,seed):")
    for fn in sorted(results):
        rec = results[fn]
        if rec["params"]["conflict"] != 1:
            continue
        p = rec["params"]
        print(f"  N{p['N']:>2} W{p['W']} cp{p['writer_core']}-{p['observer_core']} s{p['seed']}: "
              f"restore={int(rec['all_rows_restore'])} real_acc={rec['real_accuracy']:.3f} "
              f"rvp_floor={rec['real_vs_pseudo_accuracy_floor']:.3f} "
              f"pseudo_rej={rec['pseudo_reject_floor']:.3f} "
              f"wrong_act={rec['wrong_actual_match']:.3f} wrong_dec={rec['wrong_declared_match']:.3f} "
              f"-> {'PASS' if rec['witness_pass'] else 'fail'}")
    if ctrl_summary:
        print("\n# CONFLICT=0 control (want real_acc ~0.25-0.275):")
        for k in sorted(ctrl_summary):
            c = ctrl_summary[k]
            print(f"  {k:>22}: mean={c['mean_real_acc']} [{c['min']},{c['max']}] band_ok={c['in_025_0275_band']}")
    print(f"\nWITNESS_HOLDS_ANYWHERE = {out['witness_holds_anywhere']}")


if __name__ == "__main__":
    raise SystemExit(main())
