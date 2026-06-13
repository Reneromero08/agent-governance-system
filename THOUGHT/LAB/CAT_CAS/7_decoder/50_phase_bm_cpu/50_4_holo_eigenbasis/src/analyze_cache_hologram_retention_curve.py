#!/usr/bin/env python3
import csv
import json
import math
import statistics
import sys


MODES = ["basis", "rotation", "residual", "mini"]
MODE_SETS = {
    "basis": {9, 10, 11, 12, 13, 14},
    "rotation": {16, 17, 18, 19, 20, 21, 22, 23},
    "residual": {24, 25, 26, 27},
    "mini": {9, 10, 11, 12, 16, 17, 18, 19, 24, 25, 26, 27},
}


def mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def distance(a, b):
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def rate(num, den):
    return num / den if den else 0.0


def vector(row):
    return [float(row[f"l{i:02d}"]) for i in range(64)]


def mode_features(vals):
    out = []
    for mode in MODES:
        target = MODE_SETS[mode]
        t = mean([vals[i] for i in target])
        o = mean([vals[i] for i in range(64) if i not in target])
        out.append(o - t)
    return out


def predict(feat, centroids):
    return min(MODES, key=lambda mode: distance(feat, centroids[mode]))


def score_delay(rows):
    train = [r for r in rows if r["family"] == "real" and r["trial"] % 2 == 0]
    test = [r for r in rows if r["trial"] % 2 == 1]
    centroids = {}
    for mode in MODES:
        mode_rows = [r for r in train if r["declared_mode"] == mode]
        centroids[mode] = [
            mean([r["_features"][i] for r in mode_rows])
            for i in range(4)
        ]

    counts = {
        "real_total": 0,
        "real_correct": 0,
        "pseudo_total": 0,
        "pseudo_declared_match": 0,
        "wrong_total": 0,
        "wrong_actual_match": 0,
        "wrong_declared_match": 0,
    }
    by_mode = {m: {"real_total": 0, "real_correct": 0} for m in MODES}
    for row in test:
        pred = predict(row["_features"], centroids)
        family = row["family"]
        declared = row["declared_mode"]
        actual = row["actual_mode"]
        if family == "real":
            counts["real_total"] += 1
            by_mode[declared]["real_total"] += 1
            if pred == declared:
                counts["real_correct"] += 1
                by_mode[declared]["real_correct"] += 1
        elif family == "pseudo":
            counts["pseudo_total"] += 1
            if pred == declared:
                counts["pseudo_declared_match"] += 1
        else:
            counts["wrong_total"] += 1
            if pred == actual:
                counts["wrong_actual_match"] += 1
            if pred == declared:
                counts["wrong_declared_match"] += 1

    real_vs_pseudo = {}
    for mode in MODES:
        train_rp = [
            r for r in rows
            if r["declared_mode"] == mode and r["family"] in ("real", "pseudo") and r["trial"] % 2 == 0
        ]
        test_rp = [
            r for r in rows
            if r["declared_mode"] == mode and r["family"] in ("real", "pseudo") and r["trial"] % 2 == 1
        ]
        rp_centroids = {}
        for family in ("real", "pseudo"):
            family_rows = [r for r in train_rp if r["family"] == family]
            rp_centroids[family] = [
                mean([r["_vector"][i] for r in family_rows])
                for i in range(64)
            ]
        totals = {"real": 0, "pseudo": 0}
        corrects = {"real": 0, "pseudo": 0}
        for row in test_rp:
            pred = min(("real", "pseudo"), key=lambda f: distance(row["_vector"], rp_centroids[f]))
            family = row["family"]
            totals[family] += 1
            if pred == family:
                corrects[family] += 1
        real_vs_pseudo[mode] = {
            "accuracy": rate(corrects["real"] + corrects["pseudo"], totals["real"] + totals["pseudo"]),
            "pseudo_reject": rate(corrects["pseudo"], totals["pseudo"]),
        }

    real_floor = min(rate(v["real_correct"], v["real_total"]) for v in by_mode.values())
    real_acc = rate(counts["real_correct"], counts["real_total"])
    pseudo_decl = rate(counts["pseudo_declared_match"], counts["pseudo_total"])
    wrong_actual = rate(counts["wrong_actual_match"], counts["wrong_total"])
    wrong_decl = rate(counts["wrong_declared_match"], counts["wrong_total"])
    rp_floor = min(v["accuracy"] for v in real_vs_pseudo.values())
    pseudo_reject_floor = min(v["pseudo_reject"] for v in real_vs_pseudo.values())
    gates = {
        "real_accuracy_ge_0_60": real_acc >= 0.60,
        "real_floor_ge_0_45": real_floor >= 0.45,
        "pseudo_declared_le_0_35": pseudo_decl <= 0.35,
        "wrong_actual_ge_0_60": wrong_actual >= 0.60,
        "wrong_declared_le_0_20": wrong_decl <= 0.20,
        "real_vs_pseudo_floor_ge_0_90": rp_floor >= 0.90,
        "pseudo_reject_floor_ge_0_90": pseudo_reject_floor >= 0.90,
    }
    return {
        "rows": len(rows),
        "test_rows": len(test),
        "real_accuracy": real_acc,
        "real_mode_accuracy_floor": real_floor,
        "pseudo_declared_match": pseudo_decl,
        "wrong_actual_match": wrong_actual,
        "wrong_declared_match": wrong_decl,
        "real_vs_pseudo_accuracy_floor": rp_floor,
        "pseudo_reject_floor": pseudo_reject_floor,
        "gates": gates,
        "pass": all(gates.values()),
    }


def main():
    if len(sys.argv) != 3:
        print("usage: analyze_cache_hologram_retention_curve.py input.csv output.json", file=sys.stderr)
        return 2

    rows = []
    with open(sys.argv[1], newline="") as fh:
        for row in csv.DictReader(fh):
            row["delay_class"] = int(row["delay_class"])
            row["delay_pauses"] = int(row["delay_pauses"])
            row["trial"] = int(row["trial"])
            row["hash_restored"] = int(row["hash_restored"])
            row["_vector"] = vector(row)
            row["_features"] = mode_features(row["_vector"])
            rows.append(row)

    by_delay = {}
    for delay_class in sorted({r["delay_class"] for r in rows}):
        subset = [r for r in rows if r["delay_class"] == delay_class]
        scored = score_delay(subset)
        scored["delay_class"] = delay_class
        scored["delay_pauses"] = subset[0]["delay_pauses"]
        by_delay[str(delay_class)] = scored

    all_restore = sum(r["hash_restored"] for r in rows) == len(rows)
    passing = [d for d in by_delay.values() if d["pass"]]
    all_delay_pass = len(passing) == len(by_delay)
    verdict = (
        "PHASE4B_RETENTION_WINDOW_PASS"
        if all_restore and all_delay_pass
        else "PHASE4B_RETENTION_DECAY_OR_PARTIAL"
    )
    out = {
        "input": sys.argv[1],
        "rows": len(rows),
        "hash_restored": sum(r["hash_restored"] for r in rows),
        "hash_restore_total": len(rows),
        "all_restore": all_restore,
        "by_delay": by_delay,
        "all_delay_pass": all_delay_pass,
        "max_passing_delay_pauses": max([d["delay_pauses"] for d in passing], default=None),
        "verdict": verdict,
    }
    with open(sys.argv[2], "w", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(verdict)
    print("rows=%d restored=%d/%d all_delay_pass=%s max_passing_delay=%s" %
          (len(rows), out["hash_restored"], len(rows), all_delay_pass, out["max_passing_delay_pauses"]))
    for key, data in by_delay.items():
        print("delay%s pauses=%d pass=%s real=%.6f floor=%.6f pseudo=%.6f rp=%.6f preject=%.6f wactual=%.6f wdecl=%.6f" %
              (key, data["delay_pauses"], data["pass"], data["real_accuracy"],
               data["real_mode_accuracy_floor"], data["pseudo_declared_match"],
               data["real_vs_pseudo_accuracy_floor"], data["pseudo_reject_floor"],
               data["wrong_actual_match"], data["wrong_declared_match"]))
    return 0 if all_restore else 1


if __name__ == "__main__":
    raise SystemExit(main())
