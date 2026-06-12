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


def map_line(layout, canonical):
    if layout == 0:
        return canonical & 63
    return ((canonical * 13) + 7) & 63


def physical_vector(row):
    return [float(row[f"l{i:02d}"]) for i in range(64)]


def canonical_vector(row):
    phys = physical_vector(row)
    layout = int(row["layout"])
    return [phys[map_line(layout, i)] for i in range(64)]


def mode_contrast(vals, mode):
    target = MODE_SETS[mode]
    t = mean([vals[i] for i in target])
    o = mean([vals[i] for i in range(64) if i not in target])
    return o - t


def features(vals):
    return [mode_contrast(vals, mode) for mode in MODES]


def distance(a, b):
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def rate(num, den):
    return num / den if den else 0.0


def train_centroids(rows, vector_key):
    out = {}
    for mode in MODES:
        mode_rows = [r for r in rows if r["family"] == "real" and r["declared_mode"] == mode]
        out[mode] = [
            mean([r[vector_key][i] for r in mode_rows])
            for i in range(4)
        ]
    return out


def predict(feat, centroids):
    return min(MODES, key=lambda mode: distance(feat, centroids[mode]))


def score_mode_classifier(rows, centroids, vector_key):
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
        mode: {
            "real_total": 0,
            "real_correct": 0,
            "wrong_total": 0,
            "wrong_actual_match": 0,
            "wrong_declared_match": 0,
            "pseudo_total": 0,
            "pseudo_declared_match": 0,
        }
        for mode in MODES
    }

    for row in rows:
        pred = predict(row[vector_key], centroids)
        declared = row["declared_mode"]
        actual = row["actual_mode"]
        family = row["family"]

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

    by_mode_rates = {}
    for mode, vals in by_mode.items():
        by_mode_rates[mode] = {
            "real_accuracy": rate(vals["real_correct"], vals["real_total"]),
            "pseudo_declared_match": rate(vals["pseudo_declared_match"], vals["pseudo_total"]),
            "wrong_actual_match": rate(vals["wrong_actual_match"], vals["wrong_total"]),
            "wrong_declared_match": rate(vals["wrong_declared_match"], vals["wrong_total"]),
        }

    return {
        "counts": counts,
        "real_accuracy": rate(counts["real_correct"], counts["real_total"]),
        "real_mode_accuracy_floor": min(v["real_accuracy"] for v in by_mode_rates.values()),
        "pseudo_declared_match": rate(counts["pseudo_declared_match"], counts["pseudo_total"]),
        "wrong_actual_match": rate(counts["wrong_actual_match"], counts["wrong_total"]),
        "wrong_declared_match": rate(counts["wrong_declared_match"], counts["wrong_total"]),
        "by_mode_rates": by_mode_rates,
    }


def score_real_vs_pseudo(train_rows, test_rows, raw_vector_key):
    out = {}
    for mode in MODES:
        train = [
            r for r in train_rows
            if r["declared_mode"] == mode and r["family"] in ("real", "pseudo")
        ]
        test = [
            r for r in test_rows
            if r["declared_mode"] == mode and r["family"] in ("real", "pseudo")
        ]
        centroids = {}
        for family in ("real", "pseudo"):
            family_rows = [r for r in train if r["family"] == family]
            centroids[family] = [
                mean([r[raw_vector_key][i] for r in family_rows])
                for i in range(64)
            ]
        totals = {"real": 0, "pseudo": 0}
        correct = {"real": 0, "pseudo": 0}
        for row in test:
            pred = min(("real", "pseudo"), key=lambda family: distance(row[raw_vector_key], centroids[family]))
            family = row["family"]
            totals[family] += 1
            if pred == family:
                correct[family] += 1
        out[mode] = {
            "accuracy": rate(correct["real"] + correct["pseudo"], totals["real"] + totals["pseudo"]),
            "real_accept": rate(correct["real"], totals["real"]),
            "pseudo_reject": rate(correct["pseudo"], totals["pseudo"]),
        }
    return out


def main():
    if len(sys.argv) != 3:
        print("usage: analyze_cache_hologram_layout_holdout.py input.csv output.json", file=sys.stderr)
        return 2

    rows = []
    with open(sys.argv[1], newline="") as fh:
        for row in csv.DictReader(fh):
            row["layout"] = int(row["layout"])
            row["trial"] = int(row["trial"])
            row["hash_restored"] = int(row["hash_restored"])
            row["_canonical_vector"] = canonical_vector(row)
            row["_physical_vector"] = physical_vector(row)
            row["_canonical_features"] = features(row["_canonical_vector"])
            row["_physical_features"] = features(row["_physical_vector"])
            rows.append(row)

    train = [r for r in rows if r["layout"] == 0]
    test = [r for r in rows if r["layout"] == 1]

    canonical_centroids = train_centroids(train, "_canonical_features")
    physical_centroids = train_centroids(train, "_physical_features")
    canonical = score_mode_classifier(test, canonical_centroids, "_canonical_features")
    physical_baseline = score_mode_classifier(test, physical_centroids, "_physical_features")
    real_vs_pseudo = score_real_vs_pseudo(train, test, "_canonical_vector")

    all_restore = sum(r["hash_restored"] for r in rows) == len(rows)
    rp_floor = min(v["accuracy"] for v in real_vs_pseudo.values())
    pseudo_reject_floor = min(v["pseudo_reject"] for v in real_vs_pseudo.values())
    gates = {
        "all_rows_restore": all_restore,
        "canonical_real_accuracy_ge_0_60": canonical["real_accuracy"] >= 0.60,
        "canonical_real_floor_ge_0_45": canonical["real_mode_accuracy_floor"] >= 0.45,
        "canonical_pseudo_declared_match_le_0_35": canonical["pseudo_declared_match"] <= 0.35,
        "canonical_wrong_actual_match_ge_0_60": canonical["wrong_actual_match"] >= 0.60,
        "canonical_wrong_declared_match_le_0_20": canonical["wrong_declared_match"] <= 0.20,
        "canonical_real_vs_pseudo_floor_ge_0_90": rp_floor >= 0.90,
        "canonical_pseudo_reject_floor_ge_0_90": pseudo_reject_floor >= 0.90,
    }
    verdict = (
        "PHASE4B_LAYOUT_HOLDOUT_PASS"
        if all(gates.values())
        else "PHASE4B_LAYOUT_HOLDOUT_PARTIAL"
    )

    out = {
        "input": sys.argv[1],
        "rows": len(rows),
        "train_layout": 0,
        "test_layout": 1,
        "train_rows": len(train),
        "test_rows": len(test),
        "hash_restored": sum(r["hash_restored"] for r in rows),
        "hash_restore_total": len(rows),
        "canonical": canonical,
        "physical_baseline": physical_baseline,
        "canonical_real_vs_pseudo_by_mode": real_vs_pseudo,
        "canonical_real_vs_pseudo_accuracy_floor": rp_floor,
        "canonical_pseudo_reject_floor": pseudo_reject_floor,
        "gates": gates,
        "verdict": verdict,
    }
    with open(sys.argv[2], "w", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(verdict)
    print(
        "rows=%d restored=%d/%d canonical_real=%.6f canonical_floor=%.6f canonical_pseudo=%.6f rp_floor=%.6f pseudo_reject=%.6f wrong_actual=%.6f wrong_declared=%.6f fixed_real=%.6f"
        % (
            len(rows),
            out["hash_restored"],
            len(rows),
            canonical["real_accuracy"],
            canonical["real_mode_accuracy_floor"],
            canonical["pseudo_declared_match"],
            rp_floor,
            pseudo_reject_floor,
            canonical["wrong_actual_match"],
            canonical["wrong_declared_match"],
            physical_baseline["real_accuracy"],
        )
    )
    return 0 if all_restore else 1


if __name__ == "__main__":
    raise SystemExit(main())
