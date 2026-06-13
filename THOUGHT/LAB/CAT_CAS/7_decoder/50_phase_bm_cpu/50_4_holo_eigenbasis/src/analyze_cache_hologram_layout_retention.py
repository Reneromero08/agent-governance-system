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


def distance(a, b):
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def rate(num, den):
    return num / den if den else 0.0


def physical_vector(row):
    return [float(row[f"l{i:02d}"]) for i in range(64)]


def canonical_vector(row):
    phys = physical_vector(row)
    layout = row["layout"]
    return [phys[map_line(layout, i)] for i in range(64)]


def features(vals):
    out = []
    for mode in MODES:
        target = MODE_SETS[mode]
        t = mean([vals[i] for i in target])
        o = mean([vals[i] for i in range(64) if i not in target])
        out.append(o - t)
    return out


def train_centroids(rows, vector_key):
    centroids = {}
    for mode in MODES:
        mode_rows = [r for r in rows if r["family"] == "real" and r["declared_mode"] == mode]
        centroids[mode] = [
            mean([r[vector_key][i] for r in mode_rows])
            for i in range(4)
        ]
    return centroids


def predict(feat, centroids):
    return min(MODES, key=lambda mode: distance(feat, centroids[mode]))


def score_classifier(rows, centroids, vector_key):
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

    for row in rows:
        pred = predict(row[vector_key], centroids)
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
        elif family == "wrong":
            counts["wrong_total"] += 1
            if pred == actual:
                counts["wrong_actual_match"] += 1
            if pred == declared:
                counts["wrong_declared_match"] += 1

    return {
        "real_accuracy": rate(counts["real_correct"], counts["real_total"]),
        "real_mode_accuracy_floor": min(rate(v["real_correct"], v["real_total"]) for v in by_mode.values()),
        "pseudo_declared_match": rate(counts["pseudo_declared_match"], counts["pseudo_total"]),
        "wrong_actual_match": rate(counts["wrong_actual_match"], counts["wrong_total"]),
        "wrong_declared_match": rate(counts["wrong_declared_match"], counts["wrong_total"]),
        "counts": counts,
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
            fam_rows = [r for r in train if r["family"] == family]
            centroids[family] = [
                mean([r[raw_vector_key][i] for r in fam_rows])
                for i in range(64)
            ]
        totals = {"real": 0, "pseudo": 0}
        correct = {"real": 0, "pseudo": 0}
        for row in test:
            pred = min(("real", "pseudo"), key=lambda f: distance(row[raw_vector_key], centroids[f]))
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


def score_delay(rows):
    train = [r for r in rows if r["layout"] == 0]
    test = [r for r in rows if r["layout"] == 1]
    canonical_centroids = train_centroids(train, "_canonical_features")
    physical_centroids = train_centroids(train, "_physical_features")
    canonical = score_classifier(test, canonical_centroids, "_canonical_features")
    physical = score_classifier(test, physical_centroids, "_physical_features")
    rp = score_real_vs_pseudo(train, test, "_canonical_vector")
    rp_floor = min(v["accuracy"] for v in rp.values())
    pseudo_reject_floor = min(v["pseudo_reject"] for v in rp.values())
    gates = {
        "canonical_real_accuracy_ge_0_60": canonical["real_accuracy"] >= 0.60,
        "canonical_real_floor_ge_0_45": canonical["real_mode_accuracy_floor"] >= 0.45,
        "canonical_pseudo_declared_match_le_0_35": canonical["pseudo_declared_match"] <= 0.35,
        "canonical_wrong_actual_match_ge_0_60": canonical["wrong_actual_match"] >= 0.60,
        "canonical_wrong_declared_match_le_0_20": canonical["wrong_declared_match"] <= 0.20,
        "canonical_real_vs_pseudo_floor_ge_0_90": rp_floor >= 0.90,
        "canonical_pseudo_reject_floor_ge_0_90": pseudo_reject_floor >= 0.90,
    }
    return {
        "rows": len(rows),
        "train_rows": len(train),
        "test_rows": len(test),
        "canonical": canonical,
        "physical_baseline": physical,
        "canonical_real_vs_pseudo_by_mode": rp,
        "canonical_real_vs_pseudo_accuracy_floor": rp_floor,
        "canonical_pseudo_reject_floor": pseudo_reject_floor,
        "gates": gates,
        "pass": all(gates.values()),
    }


def main():
    if len(sys.argv) != 3:
        print("usage: analyze_cache_hologram_layout_retention.py input.csv output.json", file=sys.stderr)
        return 2

    rows = []
    with open(sys.argv[1], newline="") as fh:
        for row in csv.DictReader(fh):
            row["delay_class"] = int(row["delay_class"])
            row["delay_pauses"] = int(row["delay_pauses"])
            row["layout"] = int(row["layout"])
            row["trial"] = int(row["trial"])
            row["hash_restored"] = int(row["hash_restored"])
            row["_canonical_vector"] = canonical_vector(row)
            row["_physical_vector"] = physical_vector(row)
            row["_canonical_features"] = features(row["_canonical_vector"])
            row["_physical_features"] = features(row["_physical_vector"])
            rows.append(row)

    by_delay = {}
    for delay_class in sorted({r["delay_class"] for r in rows}):
        subset = [r for r in rows if r["delay_class"] == delay_class]
        scored = score_delay(subset)
        scored["delay_class"] = delay_class
        scored["delay_pauses"] = subset[0]["delay_pauses"]
        by_delay[str(delay_class)] = scored

    all_restore = sum(r["hash_restored"] for r in rows) == len(rows)
    all_delay_pass = all(v["pass"] for v in by_delay.values())
    core_gate_all = all(
        v["canonical"]["real_accuracy"] >= 0.60
        and v["canonical"]["real_mode_accuracy_floor"] >= 0.45
        and v["canonical"]["wrong_actual_match"] >= 0.60
        and v["canonical"]["wrong_declared_match"] <= 0.20
        for v in by_delay.values()
    )
    verdict = (
        "PHASE4B_LAYOUT_RETENTION_PASS"
        if all_restore and all_delay_pass
        else (
            "PHASE4B_LAYOUT_RETENTION_MODE_PASS_PSEUDO_PARTIAL"
            if all_restore and core_gate_all
            else "PHASE4B_LAYOUT_RETENTION_PARTIAL"
        )
    )

    out = {
        "input": sys.argv[1],
        "rows": len(rows),
        "hash_restored": sum(r["hash_restored"] for r in rows),
        "hash_restore_total": len(rows),
        "all_restore": all_restore,
        "by_delay": by_delay,
        "all_delay_pass": all_delay_pass,
        "core_mode_wrong_gates_all_delay": core_gate_all,
        "verdict": verdict,
    }
    with open(sys.argv[2], "w", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(verdict)
    print("rows=%d restored=%d/%d all_delay_pass=%s core_gate_all=%s" %
          (len(rows), out["hash_restored"], len(rows), all_delay_pass, core_gate_all))
    for key, data in by_delay.items():
        c = data["canonical"]
        p = data["physical_baseline"]
        print("delay%s pauses=%d pass=%s real=%.6f floor=%.6f pseudo=%.6f rp=%.6f preject=%.6f wactual=%.6f wdecl=%.6f fixed_real=%.6f" %
              (key, data["delay_pauses"], data["pass"], c["real_accuracy"],
               c["real_mode_accuracy_floor"], c["pseudo_declared_match"],
               data["canonical_real_vs_pseudo_accuracy_floor"], data["canonical_pseudo_reject_floor"],
               c["wrong_actual_match"], c["wrong_declared_match"], p["real_accuracy"]))
    return 0 if all_restore else 1


if __name__ == "__main__":
    raise SystemExit(main())
