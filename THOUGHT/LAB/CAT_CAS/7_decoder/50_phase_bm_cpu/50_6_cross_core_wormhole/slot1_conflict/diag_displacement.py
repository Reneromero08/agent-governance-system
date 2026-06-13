#!/usr/bin/env python3
"""Diagnostic: is the per-set conflict-displacement signal physically present?

For each real-family row, compute mean re-access latency on the MODE's targeted
sets vs the non-targeted sets. If conflict-displacement works, targeted sets
(the ones the sender hammered) should be SLOWER (positive targeted-minus-other).
If targeted ~ other, the eviction sets are NOT colliding in shared L3 and there
is no carrier -- a measurement/geometry issue, not a clean physics negative.

Also report the raw per-set value distribution to see absolute latency scale.
Statistics emitted: median, mean, min/max, per-mode n, and targeted-minus-other
effect size. This diagnostic is descriptive and does not compute p_value/CI.

Usage: diag_displacement.py <csv>
"""
import csv
import statistics
import sys

MODE_SETS = {
    "basis": {9, 10, 11, 12, 13, 14},
    "rotation": {16, 17, 18, 19, 20, 21, 22, 23},
    "residual": {24, 25, 26, 27},
    "mini": {9, 10, 11, 12, 16, 17, 18, 19, 24, 25, 26, 27},
}


def main():
    path = sys.argv[1]
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    all_vals = []
    by_mode_tgt = {m: [] for m in MODE_SETS}
    by_mode_oth = {m: [] for m in MODE_SETS}
    for r in rows:
        if r["family"] != "real":
            continue
        mode = r["declared_mode"]
        vals = [float(r[f"l{i:02d}"]) for i in range(64)]
        all_vals.extend(vals)
        tgt = MODE_SETS[mode]
        t = [vals[i] for i in tgt]
        o = [vals[i] for i in range(64) if i not in tgt]
        by_mode_tgt[mode].append(statistics.fmean(t))
        by_mode_oth[mode].append(statistics.fmean(o))
    print(f"file: {path}")
    print(f"global per-set cyc/touch: median={statistics.median(all_vals):.2f} "
          f"mean={statistics.fmean(all_vals):.2f} "
          f"min={min(all_vals):.2f} max={max(all_vals):.2f}")
    print(f"{'mode':>10} {'targeted':>10} {'other':>10} {'tgt-oth':>10} {'n':>5}")
    grand = []
    for m in MODE_SETS:
        if not by_mode_tgt[m]:
            continue
        mt = statistics.fmean(by_mode_tgt[m])
        mo = statistics.fmean(by_mode_oth[m])
        diff = mt - mo
        grand.append(diff)
        print(f"{m:>10} {mt:>10.2f} {mo:>10.2f} {diff:>10.3f} {len(by_mode_tgt[m]):>5}")
    if grand:
        print(f"mean targeted-minus-other displacement: {statistics.fmean(grand):.3f} cyc/touch")
        print("(5.10D measured ~37-49 cyc absolute; expect targeted SLOWER => positive)")


if __name__ == "__main__":
    raise SystemExit(main())
