#!/usr/bin/env bash
set -u

# Focused Phase 5.8 artifact-closure probe.
# Tests whether the large-tape CACHE anomaly survives fixed P-state and
# interleaved NONE/CACHE ordering.

MEAS_CORE="${MEAS_CORE:-3}"
WORKERS="${WORKERS:-0,1,2,4}"
ITERATIONS="${ITERATIONS:-30000}"
WINDOW_SIZE="${WINDOW_SIZE:-256}"
PSTATE="${PSTATE:-0}"
FREQ_LABEL="${FREQ_LABEL:-P0_LOCKED}"
OUTPUT_DIR="${OUTPUT_DIR:-../../phase5_8/results/freq_locked_cache_probe}"
BIN="./phase5_8_boundary_rdtsc"
MSR_PERF_CTL="0xC0010062"

mkdir -p "$OUTPUT_DIR"
make clean >/dev/null 2>&1 || true
make all

if [ ! -x "$BIN" ]; then
    echo "FATAL: $BIN not built"
    exit 1
fi

if ! command -v rdmsr >/dev/null 2>&1 || ! command -v wrmsr >/dev/null 2>&1; then
    echo "FATAL: rdmsr/wrmsr missing"
    exit 1
fi

modprobe msr 2>/dev/null || true
if ! rdmsr -p "$MEAS_CORE" "$MSR_PERF_CTL" >/dev/null 2>&1; then
    echo "FATAL: MSR device unavailable for core $MEAS_CORE; load msr or check /dev/cpu/$MEAS_CORE/msr"
    exit 1
fi

MSR_AUDIT="$OUTPUT_DIR/msr_lock_audit.csv"
ORDER="$OUTPUT_DIR/condition_order.csv"
SUMMARY="$OUTPUT_DIR/freq_locked_cache_probe_summary.csv"
REPORT="$OUTPUT_DIR/PHASE5_8_FREQ_LOCKED_CACHE_ARTIFACT_PROBE.md"

echo "run_id,tape_size,mode,repeat,pstate_requested,msr_before,msr_after,exit_code" > "$MSR_AUDIT"
echo "order,run_id,tape_size,mode,repeat" > "$ORDER"

run_one() {
    local order="$1" tape="$2" mode="$3" rep="$4"
    local run_id="FREQLOCK_${mode}_T${tape}_R${rep}"
    local dir="$OUTPUT_DIR/$run_id"
    local before after exit_code

    mkdir -p "$dir"
    echo "$order,$run_id,$tape,$mode,$rep" >> "$ORDER"

    before="$(rdmsr -p "$MEAS_CORE" "$MSR_PERF_CTL" 2>/dev/null || echo READ_FAIL)"
    if ! wrmsr -p "$MEAS_CORE" "$MSR_PERF_CTL" "$PSTATE"; then
        echo "FATAL: failed to write P-state $PSTATE on core $MEAS_CORE"
        return 1
    fi
    after="$(rdmsr -p "$MEAS_CORE" "$MSR_PERF_CTL" 2>/dev/null || echo READ_FAIL)"
    if [ "$after" = "READ_FAIL" ]; then
        echo "FATAL: failed to verify P-state after write on core $MEAS_CORE"
        return 1
    fi

    exit_code=0
    if [ "$mode" = "CACHE" ]; then
        "$BIN" \
            --core "$MEAS_CORE" \
            --workers "$WORKERS" \
            --worker-mode cache \
            --placement offcore \
            --iterations "$ITERATIONS" \
            --window-size "$WINDOW_SIZE" \
            --tape-size "$tape" \
            --freq-label "$FREQ_LABEL" \
            --vid-label unknown \
            --run-id "$run_id" \
            --output-dir "$dir" \
            --randomize-order || exit_code=$?
    else
        "$BIN" \
            --core "$MEAS_CORE" \
            --worker-mode none \
            --placement no_workers \
            --iterations "$ITERATIONS" \
            --window-size "$WINDOW_SIZE" \
            --tape-size "$tape" \
            --freq-label "$FREQ_LABEL" \
            --vid-label unknown \
            --run-id "$run_id" \
            --output-dir "$dir" \
            --randomize-order || exit_code=$?
    fi

    echo "$run_id,$tape,$mode,$rep,$PSTATE,$before,$after,$exit_code" >> "$MSR_AUDIT"
    return "$exit_code"
}

FAILED=0
order=0
for rep in 1 2 3; do
    for tape in 1024 4096; do
        run_one "$order" "$tape" "NONE" "$rep" || FAILED=1
        order=$((order + 1))
        run_one "$order" "$tape" "CACHE" "$rep" || FAILED=1
        order=$((order + 1))
    done
done

ANALYSIS_FAILED=0
for dir in "$OUTPUT_DIR"/FREQLOCK_*; do
    if [ -f "$dir/raw_cycles.csv" ]; then
        python3 analyze_phase5_8.py --input-dir "$dir" --window-size "$WINDOW_SIZE" || ANALYSIS_FAILED=1
    fi
done

python3 - "$OUTPUT_DIR" "$SUMMARY" "$REPORT" <<'PY'
import csv
import os
import statistics
import sys

outdir, summary_path, report_path = sys.argv[1:4]
rows = []
for name in sorted(os.listdir(outdir)):
    if not name.startswith("FREQLOCK_"):
        continue
    geo_path = os.path.join(outdir, name, "geometry_stats.csv")
    if not os.path.exists(geo_path):
        continue
    parts = name.split("_")
    mode = parts[1]
    tape = int(parts[2][1:])
    rep = int(parts[3][1:])
    with open(geo_path, newline="") as f:
        geo = next(csv.DictReader(f), None)
    if not geo:
        continue
    rows.append({
        "run_id": name,
        "tape_size": tape,
        "mode": mode,
        "repeat": rep,
        "boundary_thickness": float(geo.get("boundary_thickness_nn_mean", 0)),
        "mean_radius": float(geo.get("mean_radius", 0)),
        "effective_dimension": float(geo.get("effective_dimension", 0)),
        "spectral_entropy": float(geo.get("spectral_entropy", 0)),
    })

groups = {}
for r in rows:
    groups.setdefault((r["tape_size"], r["mode"]), []).append(r)

summary = []
for tape in sorted({r["tape_size"] for r in rows}):
    none = groups.get((tape, "NONE"), [])
    cache = groups.get((tape, "CACHE"), [])
    if not none or not cache:
        continue
    none_t = [r["boundary_thickness"] for r in none]
    cache_t = [r["boundary_thickness"] for r in cache]
    none_r = [r["mean_radius"] for r in none]
    cache_r = [r["mean_radius"] for r in cache]
    ratio = statistics.mean(cache_t) / statistics.mean(none_t) if statistics.mean(none_t) else 0
    summary.append({
        "tape_size": tape,
        "none_thickness_mean": statistics.mean(none_t),
        "cache_thickness_mean": statistics.mean(cache_t),
        "cache_over_none_thickness_ratio": ratio,
        "none_radius_mean": statistics.mean(none_r),
        "cache_radius_mean": statistics.mean(cache_r),
        "n_none": len(none),
        "n_cache": len(cache),
    })

with open(summary_path, "w", newline="") as f:
    fields = ["tape_size", "none_thickness_mean", "cache_thickness_mean",
              "cache_over_none_thickness_ratio", "none_radius_mean",
              "cache_radius_mean", "n_none", "n_cache"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(summary)

ratios = [s["cache_over_none_thickness_ratio"] for s in summary]
if len(ratios) >= 2 and all(r > 1.05 for r in ratios):
    verdict = "CACHE_ARTIFACT_CLEARED"
elif ratios and all(0.95 <= r <= 1.05 for r in ratios):
    verdict = "CACHE_EFFECT_FLAT_UNDER_FREQ_LOCK"
elif ratios and any(r < 0.95 for r in ratios):
    verdict = "CACHE_ARTIFACT_PERSISTS_OR_BOUNDARY_CONTRACTS"
else:
    verdict = "INCONCLUSIVE"

with open(report_path, "w", newline="") as f:
    f.write("# Phase 5.8 Frequency-Locked Cache Artifact Probe\n\n")
    f.write(f"Verdict: `{verdict}`\n\n")
    f.write("Objective: test whether the large-tape CACHE anomaly survives fixed P-state and interleaved NONE/CACHE ordering.\n\n")
    f.write("Acceptance:\n")
    f.write("- `CACHE_ARTIFACT_CLEARED`: CACHE/NONE thickness ratio > 1.05 for all tested large tapes.\n")
    f.write("- `CACHE_ARTIFACT_PERSISTS_OR_BOUNDARY_CONTRACTS`: any tested large tape has ratio < 0.95.\n")
    f.write("- `CACHE_EFFECT_FLAT_UNDER_FREQ_LOCK`: all ratios within +/-5%.\n\n")
    f.write("## Summary\n\n")
    f.write("| Tape | NONE thickness | CACHE thickness | CACHE/NONE | NONE radius | CACHE radius |\n")
    f.write("|------|----------------|-----------------|------------|-------------|--------------|\n")
    for s in summary:
        f.write(f"| {s['tape_size']} | {s['none_thickness_mean']:.6f} | {s['cache_thickness_mean']:.6f} | {s['cache_over_none_thickness_ratio']:.6f} | {s['none_radius_mean']:.6f} | {s['cache_radius_mean']:.6f} |\n")
    f.write("\nArtifacts:\n")
    f.write(f"- `{os.path.basename(summary_path)}`\n")
    f.write("- `msr_lock_audit.csv`\n")
    f.write("- `condition_order.csv`\n")
PY

if [ "$FAILED" -ne 0 ] || [ "$ANALYSIS_FAILED" -ne 0 ]; then
    exit 1
fi
