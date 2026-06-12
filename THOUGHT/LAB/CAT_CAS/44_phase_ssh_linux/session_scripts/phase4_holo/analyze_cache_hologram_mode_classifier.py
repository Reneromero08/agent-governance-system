#!/usr/bin/env python3
import csv
import json
import statistics
import sys


MODES = ["basis", "rotation", "residual", "mini", "random_reversible"]
MODE_SETS = {
    "basis": {9, 10, 11, 12, 13, 14},
    "rotation": {16, 17, 18, 19, 20, 21, 22, 23},
    "residual": {24, 25, 26, 27},
    "mini": {9, 10, 11, 12, 16, 17, 18, 19, 24, 25, 26, 27},
}


def mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def feature(row, mode):
    vals = [float(row[f"l{i:02d}"]) for i in range(64)]
    if mode == "random_reversible":
        # Random schedule changes per trial, so use the generic cold/hot spread.
        return max(vals) - min(vals)
    target = MODE_SETS[mode]
    t = mean([vals[i] for i in target])
    o = mean([vals[i] for i in range(64) if i not in target])
    return o - t


def predict(row, centroids):
    feats = {m: feature(row, m) for m in MODES}
    # For real modes, larger contrast means stronger evidence. Random uses its own spread.
    scores = {m: -abs(feats[m] - centroids[m]) for m in MODES}
    return max(scores, key=scores.get), feats


def main():
    if len(sys.argv) != 3:
        print("usage: analyze_cache_hologram_mode_classifier.py input.csv output.json", file=sys.stderr)
        return 2

    rows = []
    with open(sys.argv[1], newline="") as fh:
        for row in csv.DictReader(fh):
            row["trial"] = int(row["trial"])
            row["hash_restored"] = int(row["hash_restored"])
            rows.append(row)

    train = [r for r in rows if r["trial"] % 2 == 0]
    test = [r for r in rows if r["trial"] % 2 == 1]

    centroids = {}
    for mode in MODES:
        centroids[mode] = mean([feature(r, mode) for r in train if r["mode"] == mode])

    confusion = {m: {n: 0 for n in MODES} for m in MODES}
    correct = 0
    by_mode_correct = {m: 0 for m in MODES}
    by_mode_total = {m: 0 for m in MODES}
    feature_means = {m: {} for m in MODES}

    for mode in MODES:
        mode_rows = [r for r in rows if r["mode"] == mode]
        for feat_mode in MODES:
            feature_means[mode][feat_mode] = mean([feature(r, feat_mode) for r in mode_rows])

    for row in test:
        pred, _ = predict(row, centroids)
        actual = row["mode"]
        confusion[actual][pred] += 1
        by_mode_total[actual] += 1
        if pred == actual:
            correct += 1
            by_mode_correct[actual] += 1

    accuracy = correct / len(test) if test else 0.0
    by_mode_accuracy = {
        m: (by_mode_correct[m] / by_mode_total[m] if by_mode_total[m] else 0.0)
        for m in MODES
    }
    all_restore = sum(r["hash_restored"] for r in rows) == len(rows)

    real_modes = [m for m in MODES if m != "random_reversible"]
    real_mode_floor = min(by_mode_accuracy[m] for m in real_modes)
    verdict = (
        "PHASE4B_CACHE_HOLOGRAM_WITNESS"
        if all_restore and accuracy >= 0.60 and real_mode_floor >= 0.50
        else "PHASE4B_CACHE_MODE_CLASSIFIER_WEAK"
    )

    out = {
        "input": sys.argv[1],
        "rows": len(rows),
        "train_rows": len(train),
        "test_rows": len(test),
        "hash_restored": sum(r["hash_restored"] for r in rows),
        "hash_restore_total": len(rows),
        "all_restore": all_restore,
        "centroids": centroids,
        "feature_means": feature_means,
        "confusion": confusion,
        "heldout_accuracy": accuracy,
        "by_mode_accuracy": by_mode_accuracy,
        "real_mode_accuracy_floor": real_mode_floor,
        "gates": {
            "all_rows_restore": all_restore,
            "heldout_accuracy_ge_0_60": accuracy >= 0.60,
            "real_mode_accuracy_floor_ge_0_50": real_mode_floor >= 0.50,
        },
        "verdict": verdict,
    }
    with open(sys.argv[2], "w", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(verdict)
    print("rows=%d restored=%d/%d heldout_accuracy=%.6f real_floor=%.6f" %
          (len(rows), out["hash_restored"], len(rows), accuracy, real_mode_floor))
    return 0 if all_restore else 1


if __name__ == "__main__":
    raise SystemExit(main())
