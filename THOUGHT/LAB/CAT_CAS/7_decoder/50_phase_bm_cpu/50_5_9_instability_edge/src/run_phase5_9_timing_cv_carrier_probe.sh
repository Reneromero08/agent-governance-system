#!/usr/bin/env bash
set -u

# Focused Phase 5.9 carrier probe.
# Tests whether boundary thickness tracks sustained timing CV under controlled
# P-state + worker-mode variation, without claiming failure-edge approach.

MEAS_CORE="${MEAS_CORE:-3}"
WORKER_CORES="${WORKER_CORES:-0,1,2,4}"
ALL_CORES="${ALL_CORES:-0 1 2 3 4 5}"
ITERATIONS="${ITERATIONS:-30000}"
WINDOW_SIZE="${WINDOW_SIZE:-256}"
FIXED_TAPE="${FIXED_TAPE:-2048}"
OUTPUT_DIR="${OUTPUT_DIR:-../../phase5_9/results/timing_cv_carrier_probe}"
TEMP_LIMIT="${TEMP_LIMIT:-65}"
BIN="./phase5_9_stress_ladder"
MSR_BASE="0xC0010062"

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
if ! rdmsr -p "$MEAS_CORE" "$MSR_BASE" >/dev/null 2>&1; then
    echo "FATAL: MSR device unavailable for core $MEAS_CORE"
    exit 1
fi

MSR_AUDIT="$OUTPUT_DIR/msr_all_core_audit.csv"
ORDER="$OUTPUT_DIR/condition_order.csv"
SUMMARY="$OUTPUT_DIR/timing_cv_carrier_probe_summary.csv"
REPORT="$OUTPUT_DIR/PHASE5_9_TIMING_CV_CARRIER_PROBE.md"

echo "run_id,repeat,pstate,label,mode,core,msr_before,msr_after,exit_code" > "$MSR_AUDIT"
echo "order,run_id,repeat,pstate,label,mode,stress_level" > "$ORDER"

set_pstate_all() {
    local pstate="$1"
    local label="$2"
    local run_id="$3"
    local rep="$4"
    local mode="$5"
    local exit_code=0

    for core in $ALL_CORES; do
        local before after
        before="$(rdmsr -p "$core" "$MSR_BASE" 2>/dev/null || echo READ_FAIL)"
        if ! wrmsr -p "$core" "$MSR_BASE" "$pstate"; then
            exit_code=1
        fi
        after="$(rdmsr -p "$core" "$MSR_BASE" 2>/dev/null || echo READ_FAIL)"
        echo "$run_id,$rep,$pstate,$label,$mode,$core,$before,$after,$exit_code" >> "$MSR_AUDIT"
        if [ "$after" = "READ_FAIL" ]; then
            exit_code=1
        fi
    done
    return "$exit_code"
}

run_point() {
    local order="$1" rep="$2" pstate="$3" label="$4" mode="$5" stress_level="$6"
    local run_id="CVCARRIER_${label}_${mode}_R${rep}"
    local dir="$OUTPUT_DIR/$run_id"
    local worker_args=""
    local exit_code=0

    mkdir -p "$dir"
    echo "$order,$run_id,$rep,$pstate,$label,$mode,$stress_level" >> "$ORDER"

    if ! set_pstate_all "$pstate" "$label" "$run_id" "$rep" "$mode"; then
        echo "FATAL: failed all-core P-state set for $run_id"
        return 1
    fi

    if [ "$mode" != "none" ]; then
        worker_args="--workers $WORKER_CORES --worker-mode $mode"
    else
        worker_args="--worker-mode none"
    fi

    "$BIN" --core "$MEAS_CORE" $worker_args \
        --iterations "$ITERATIONS" --tape-size "$FIXED_TAPE" \
        --freq-label "$label" --vid-label unknown \
        --run-id "$run_id" --output-dir "$dir" \
        --stress-id "CV_${label}_${mode}" \
        --stress-label "TIMING_CV_${label}_${mode}" \
        --stress-level "$stress_level" --temp-limit "$TEMP_LIMIT" || exit_code=$?

    return "$exit_code"
}

FAILED=0
order=0
for rep in 1 2; do
    for spec in "0,P0,0" "2,P2,1" "4,P4,2"; do
        IFS=',' read -r pstate label base_level <<< "$spec"
        for mode_spec in "none,0" "cache,1" "mixed,2"; do
            IFS=',' read -r mode mode_level <<< "$mode_spec"
            stress_level=$((base_level * 3 + mode_level))
            run_point "$order" "$rep" "$pstate" "$label" "$mode" "$stress_level" || FAILED=1
            order=$((order + 1))
        done
    done
done

ANALYSIS_FAILED=0
for dir in "$OUTPUT_DIR"/CVCARRIER_*; do
    if [ -f "$dir/raw_cycles.csv" ]; then
        python3 analyze_phase5_9c.py --input-dir "$dir" --window-size "$WINDOW_SIZE" || ANALYSIS_FAILED=1
    fi
done

python3 - "$OUTPUT_DIR" "$SUMMARY" "$REPORT" <<'PY'
import csv
import math
import os
import statistics
import sys

outdir, summary_path, report_path = sys.argv[1:4]

def corr(xs, ys):
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (denx * deny) if denx and deny else 0.0

rows = []
for name in sorted(os.listdir(outdir)):
    if not name.startswith("CVCARRIER_"):
        continue
    geo_path = os.path.join(outdir, name, "geometry_stats.csv")
    if not os.path.exists(geo_path):
        continue
    with open(geo_path, newline="") as f:
        geo = next(csv.DictReader(f), None)
    if not geo:
        continue
    parts = name.split("_")
    rows.append({
        "run_id": name,
        "pstate_label": parts[1],
        "mode": parts[2],
        "repeat": int(parts[3][1:]),
        "boundary_thickness": float(geo.get("boundary_thickness_nn_mean", 0)),
        "mean_radius": float(geo.get("mean_radius", 0)),
        "cycle_cv": float(geo.get("cycle_cv", 0)),
        "spike_rate": float(geo.get("spike_rate", 0)),
        "p99_p50_ratio": float(geo.get("p99_p50_ratio", 0)),
        "restoration_failures": int(float(geo.get("restoration_failures", 0))),
        "stable_thickness": float(geo.get("stable_thickness", 0)),
        "spike_free_thickness": float(geo.get("spike_free_thickness", 0)),
    })

with open(summary_path, "w", newline="") as f:
    fields = ["run_id", "pstate_label", "mode", "repeat", "boundary_thickness",
              "mean_radius", "cycle_cv", "spike_rate", "p99_p50_ratio",
              "restoration_failures", "stable_thickness", "spike_free_thickness"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

thickness = [r["boundary_thickness"] for r in rows]
cv = [r["cycle_cv"] for r in rows]
spike = [r["spike_rate"] for r in rows]
p99p50 = [r["p99_p50_ratio"] for r in rows]

r_cv = corr(cv, thickness)
r_spike = corr(spike, thickness)
r_tail = corr(p99p50, thickness)
restore_fails = sum(r["restoration_failures"] for r in rows)

if restore_fails:
    verdict = "RESTORATION_CONFOUNDED"
elif abs(r_cv) >= 0.5 and abs(r_cv) > abs(r_spike):
    verdict = "TIMING_CV_CARRIER_CONFIRMED"
elif abs(r_cv) >= 0.3:
    verdict = "TIMING_CV_CARRIER_CANDIDATE"
elif abs(r_spike) >= 0.3:
    verdict = "SPIKE_ARTIFACT_DOMINANT"
else:
    verdict = "NO_TIMING_CV_CARRIER"

with open(report_path, "w", newline="") as f:
    f.write("# Phase 5.9 Timing-CV Carrier Probe\n\n")
    f.write(f"Verdict: `{verdict}`\n\n")
    f.write("Objective: test whether boundary thickness tracks sustained timing CV under controlled P-state and worker-mode variation.\n\n")
    f.write("This is not a failure-edge claim. It tests the carrier thread left open by Phase 5.9C.\n\n")
    f.write("## Metrics\n\n")
    f.write(f"- Runs analyzed: {len(rows)}\n")
    f.write(f"- Restoration failures: {restore_fails}\n")
    f.write(f"- r(boundary_thickness, cycle_cv): {r_cv:.6f}\n")
    f.write(f"- r(boundary_thickness, spike_rate): {r_spike:.6f}\n")
    f.write(f"- r(boundary_thickness, p99_p50): {r_tail:.6f}\n\n")
    f.write("## Acceptance\n\n")
    f.write("- `TIMING_CV_CARRIER_CONFIRMED`: |r_cv| >= 0.5 and stronger than spike-rate correlation.\n")
    f.write("- `TIMING_CV_CARRIER_CANDIDATE`: |r_cv| >= 0.3.\n")
    f.write("- `SPIKE_ARTIFACT_DOMINANT`: spike-rate correlation dominates.\n")
    f.write("- `NO_TIMING_CV_CARRIER`: timing-CV relationship does not reproduce.\n\n")
    f.write("## Rows\n\n")
    f.write("| Run | Thickness | CV | Spike rate | p99/p50 |\n")
    f.write("|-----|-----------|----|------------|--------|\n")
    for r in rows:
        f.write(f"| {r['run_id']} | {r['boundary_thickness']:.6f} | {r['cycle_cv']:.6f} | {r['spike_rate']:.6f} | {r['p99_p50_ratio']:.6f} |\n")
PY

if [ "$FAILED" -ne 0 ] || [ "$ANALYSIS_FAILED" -ne 0 ]; then
    exit 1
fi
