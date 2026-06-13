#!/usr/bin/env python3
import csv
import json
import statistics
import sys


def mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def stdev(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def main():
    if len(sys.argv) != 3:
        print("usage: analyze_cache_hologram_afterimage.py input.csv output.json", file=sys.stderr)
        return 2

    rows = []
    with open(sys.argv[1], newline="") as fh:
        for row in csv.DictReader(fh):
            row["trial"] = int(row["trial"])
            row["hash_restored"] = int(row["hash_restored"])
            row["group_cycles"] = float(row["group_cycles"])
            row["other_cycles"] = float(row["other_cycles"])
            row["contrast_cycles"] = float(row["contrast_cycles"])
            rows.append(row)

    by_mode = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)

    modes = {}
    for mode, items in sorted(by_mode.items()):
        contrast = [r["contrast_cycles"] for r in items]
        modes[mode] = {
            "n": len(items),
            "restored": sum(r["hash_restored"] for r in items),
            "contrast_mean": mean(contrast),
            "contrast_stdev": stdev(contrast),
            "contrast_positive_frac": mean([1.0 if x > 0 else 0.0 for x in contrast]),
            "group_cycles_mean": mean([r["group_cycles"] for r in items]),
            "other_cycles_mean": mean([r["other_cycles"] for r in items]),
        }

    real_modes = [m for m in modes if m != "random_reversible"]
    random_mean = modes.get("random_reversible", {}).get("contrast_mean", 0.0)
    real_mean = mean([modes[m]["contrast_mean"] for m in real_modes])
    real_positive = mean([modes[m]["contrast_positive_frac"] for m in real_modes])
    real_over_random = {m: modes[m]["contrast_mean"] - random_mean for m in real_modes}

    restored = sum(r["hash_restored"] for r in rows)
    all_restore = restored == len(rows)
    real_beats_random = real_mean > random_mean
    real_positive_gate = real_positive >= 0.60
    mode_specific_gate = all(delta >= 10.0 for delta in real_over_random.values())
    if all_restore and mode_specific_gate and real_positive_gate:
        verdict = "PHASE4B_CACHE_HOLOGRAM_WITNESS"
    elif all_restore and real_beats_random and real_positive_gate:
        verdict = "PHASE4B_CACHE_AFTERIMAGE_PRESENT_GENERIC"
    else:
        verdict = "PHASE4B_CACHE_AFTERIMAGE_NULL_OR_WEAK"

    out = {
        "input": sys.argv[1],
        "rows": len(rows),
        "hash_restored": restored,
        "hash_restore_total": len(rows),
        "all_restore": all_restore,
        "modes": modes,
        "real_mode_contrast_mean": real_mean,
        "random_reversible_contrast_mean": random_mean,
        "real_over_random_contrast": real_over_random,
        "real_modes_positive_frac_mean": real_positive,
        "gates": {
            "all_rows_restore": all_restore,
            "real_mean_contrast_gt_random": real_beats_random,
            "each_real_mode_contrast_gt_random_plus_10_cycles": mode_specific_gate,
            "real_positive_frac_ge_0_60": real_positive_gate,
        },
        "verdict": verdict,
    }

    with open(sys.argv[2], "w", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(verdict)
    print("rows=%d restored=%d/%d real_mean=%.6f random_mean=%.6f real_positive=%.6f" %
          (len(rows), restored, len(rows), real_mean, random_mean, real_positive))
    return 0 if all_restore else 1


if __name__ == "__main__":
    raise SystemExit(main())
