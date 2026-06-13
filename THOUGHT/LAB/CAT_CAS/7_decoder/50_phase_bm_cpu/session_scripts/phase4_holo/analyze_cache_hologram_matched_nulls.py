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


def predict(feat, centroids):
    return min(MODES, key=lambda mode: distance(feat, centroids[mode]))


def rate(num, den):
    return num / den if den else 0.0


def main():
    if len(sys.argv) != 3:
        print("usage: analyze_cache_hologram_matched_nulls.py input.csv output.json", file=sys.stderr)
        return 2

    rows = []
    with open(sys.argv[1], newline="") as fh:
        for row in csv.DictReader(fh):
            row["trial"] = int(row["trial"])
            row["hash_restored"] = int(row["hash_restored"])
            row["_features"] = features(row)
            row["_vector"] = full_vector(row)
            rows.append(row)

    train_real = [
        r for r in rows
        if r["family"] == "real" and r["trial"] % 2 == 0
    ]
    test = [r for r in rows if r["trial"] % 2 == 1]

    centroids = {}
    for mode in MODES:
        mode_rows = [r for r in train_real if r["declared_mode"] == mode]
        centroids[mode] = [
            mean([r["_features"][i] for r in mode_rows])
            for i in range(len(MODES))
        ]

    families = ["real", "pseudo", "wrong"]
    confusion = {
        family: {actual: {pred: 0 for pred in MODES} for actual in MODES + ["pseudo"]}
        for family in families
    }
    counts = {
        "real_total": 0,
        "real_correct": 0,
        "pseudo_total": 0,
        "pseudo_declared_match": 0,
        "wrong_total": 0,
        "wrong_actual_match": 0,
        "wrong_declared_match": 0,
    }
    by_mode = {
        mode: {"real_total": 0, "real_correct": 0, "pseudo_total": 0,
               "pseudo_declared_match": 0, "wrong_total": 0,
               "wrong_actual_match": 0, "wrong_declared_match": 0}
        for mode in MODES
    }

    feature_means = {
        family: {
            mode: [0.0 for _ in MODES]
            for mode in MODES
        }
        for family in families
    }
    for family in families:
        for mode in MODES:
            subset = [r for r in rows if r["family"] == family and r["declared_mode"] == mode]
            if subset:
                feature_means[family][mode] = [
                    mean([r["_features"][i] for r in subset])
                    for i in range(len(MODES))
                ]

    for row in test:
        pred = predict(row["_features"], centroids)
        family = row["family"]
        declared = row["declared_mode"]
        actual = row["actual_mode"]
        confusion[family][actual][pred] += 1

        if family == "real":
            counts["real_total"] += 1
            by_mode[declared]["real_total"] += 1
            if pred == declared:
                counts["real_correct"] += 1
                by_mode[declared]["real_correct"] += 1
        elif family == "pseudo":
            counts["pseudo_total"] += 1
            by_mode[declared]["pseudo_total"] += 1
            if pred == declared:
                counts["pseudo_declared_match"] += 1
                by_mode[declared]["pseudo_declared_match"] += 1
        elif family == "wrong":
            counts["wrong_total"] += 1
            by_mode[declared]["wrong_total"] += 1
            if pred == actual:
                counts["wrong_actual_match"] += 1
                by_mode[declared]["wrong_actual_match"] += 1
            if pred == declared:
                counts["wrong_declared_match"] += 1
                by_mode[declared]["wrong_declared_match"] += 1

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
            pred_family = min(
                ("real", "pseudo"),
                key=lambda family: distance(row["_vector"], rp_centroids[family]),
            )
            family = row["family"]
            totals[family] += 1
            if pred_family == family:
                corrects[family] += 1
        real_vs_pseudo[mode] = {
            "accuracy": rate(corrects["real"] + corrects["pseudo"], totals["real"] + totals["pseudo"]),
            "real_accept": rate(corrects["real"], totals["real"]),
            "pseudo_reject": rate(corrects["pseudo"], totals["pseudo"]),
        }

    by_mode_rates = {}
    for mode, vals in by_mode.items():
        by_mode_rates[mode] = {
            "real_accuracy": rate(vals["real_correct"], vals["real_total"]),
            "pseudo_declared_match": rate(vals["pseudo_declared_match"], vals["pseudo_total"]),
            "wrong_actual_match": rate(vals["wrong_actual_match"], vals["wrong_total"]),
            "wrong_declared_match": rate(vals["wrong_declared_match"], vals["wrong_total"]),
        }

    real_accuracy = rate(counts["real_correct"], counts["real_total"])
    pseudo_declared_match = rate(counts["pseudo_declared_match"], counts["pseudo_total"])
    wrong_actual_match = rate(counts["wrong_actual_match"], counts["wrong_total"])
    wrong_declared_match = rate(counts["wrong_declared_match"], counts["wrong_total"])
    all_restore = sum(r["hash_restored"] for r in rows) == len(rows)
    real_floor = min(v["real_accuracy"] for v in by_mode_rates.values())
    real_vs_pseudo_accuracy_floor = min(v["accuracy"] for v in real_vs_pseudo.values())
    pseudo_reject_floor = min(v["pseudo_reject"] for v in real_vs_pseudo.values())

    gates = {
        "all_rows_restore": all_restore,
        "real_accuracy_ge_0_60": real_accuracy >= 0.60,
        "real_mode_floor_ge_0_45": real_floor >= 0.45,
        "pseudo_declared_match_le_0_35": pseudo_declared_match <= 0.35,
        "real_vs_pseudo_accuracy_floor_ge_0_95": real_vs_pseudo_accuracy_floor >= 0.95,
        "pseudo_reject_floor_ge_0_95": pseudo_reject_floor >= 0.95,
        "wrong_actual_match_ge_0_60": wrong_actual_match >= 0.60,
        "wrong_declared_match_le_0_20": wrong_declared_match <= 0.20,
    }
    verdict = (
        "PHASE4B_MATCHED_NULLS_PASS"
        if all(gates.values())
        else "PHASE4B_MATCHED_NULLS_PARTIAL"
    )

    out = {
        "input": sys.argv[1],
        "rows": len(rows),
        "test_rows": len(test),
        "train_real_rows": len(train_real),
        "hash_restored": sum(r["hash_restored"] for r in rows),
        "hash_restore_total": len(rows),
        "centroids": centroids,
        "counts": counts,
        "confusion": confusion,
        "feature_means": feature_means,
        "real_accuracy": real_accuracy,
        "real_mode_accuracy_floor": real_floor,
        "pseudo_declared_match": pseudo_declared_match,
        "wrong_actual_match": wrong_actual_match,
        "wrong_declared_match": wrong_declared_match,
        "by_mode_rates": by_mode_rates,
        "real_vs_pseudo_by_mode": real_vs_pseudo,
        "real_vs_pseudo_accuracy_floor": real_vs_pseudo_accuracy_floor,
        "pseudo_reject_floor": pseudo_reject_floor,
        "gates": gates,
        "verdict": verdict,
    }
    with open(sys.argv[2], "w", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(verdict)
    print(
        "rows=%d restored=%d/%d real_acc=%.6f real_floor=%.6f pseudo_declared=%.6f rp_floor=%.6f pseudo_reject_floor=%.6f wrong_actual=%.6f wrong_declared=%.6f"
        % (
            len(rows),
            out["hash_restored"],
            len(rows),
            real_accuracy,
            real_floor,
            pseudo_declared_match,
            real_vs_pseudo_accuracy_floor,
            pseudo_reject_floor,
            wrong_actual_match,
            wrong_declared_match,
        )
    )
    return 0 if all_restore else 1


if __name__ == "__main__":
    raise SystemExit(main())
