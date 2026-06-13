#!/usr/bin/env bash
set -u

# Cyberpunk-safe boundary push: adversarial software substrate abuse without
# voltage writes, BIOS flashing, or destructive filesystem operations.

MEAS_CORE="${MEAS_CORE:-3}"
DISTURB_CORES="${DISTURB_CORES:-0 1 2 4}"
ALL_CORES="${ALL_CORES:-0 1 2 3 4 5}"
ITERATIONS="${ITERATIONS:-40000}"
WINDOW_SIZE="${WINDOW_SIZE:-256}"
FIXED_TAPE="${FIXED_TAPE:-2048}"
OUTPUT_DIR="${OUTPUT_DIR:-../results/boundary_abuse_probe}"
TEMP_LIMIT="${TEMP_LIMIT:-65}"
PSTATE="${PSTATE:-0}"
FREQ_LABEL="${FREQ_LABEL:-P0_ABUSE_LOCKED}"
BIN="./phase5_9_stress_ladder"
DISTURBER="./boundary_abuse_disturber"
MSR_BASE="0xC0010062"

mkdir -p "$OUTPUT_DIR"
make clean >/dev/null 2>&1 || true
make all
gcc -O2 -march=native -Wall -Wextra -std=c11 -o "$DISTURBER" boundary_abuse_disturber.c -lm

if [ ! -x "$BIN" ] || [ ! -x "$DISTURBER" ]; then
    echo "FATAL: harness or disturber not built"
    exit 1
fi

modprobe msr 2>/dev/null || true
if ! rdmsr -p "$MEAS_CORE" "$MSR_BASE" >/dev/null 2>&1; then
    echo "FATAL: MSR unavailable for P-state lock"
    exit 1
fi

ORDER="$OUTPUT_DIR/condition_order.csv"
MSR_AUDIT="$OUTPUT_DIR/msr_lock_audit.csv"
SUMMARY="$OUTPUT_DIR/boundary_abuse_probe_summary.csv"
REPORT="$OUTPUT_DIR/PHASE5_9_BOUNDARY_ABUSE_PROBE.md"

echo "order,run_id,repeat,abuse_mode,disturbers" > "$ORDER"
echo "run_id,core,msr_before,msr_after" > "$MSR_AUDIT"

PIDS=()
cleanup_disturbers() {
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" >/dev/null 2>&1 || true
    done
    wait >/dev/null 2>&1 || true
    PIDS=()
}
trap cleanup_disturbers EXIT

lock_pstate() {
    local run_id="$1"
    for core in $ALL_CORES; do
        local before after
        before="$(rdmsr -p "$core" "$MSR_BASE" 2>/dev/null || echo READ_FAIL)"
        wrmsr -p "$core" "$MSR_BASE" "$PSTATE"
        after="$(rdmsr -p "$core" "$MSR_BASE" 2>/dev/null || echo READ_FAIL)"
        echo "$run_id,$core,$before,$after" >> "$MSR_AUDIT"
        if [ "$after" = "READ_FAIL" ]; then
            return 1
        fi
    done
    return 0
}

start_abuse() {
    local mode="$1"
    cleanup_disturbers

    case "$mode" in
        quiet)
            ;;
        syscall)
            for core in $DISTURB_CORES; do "$DISTURBER" syscall "$core" 64 & PIDS+=("$!"); done
            ;;
        cache)
            for core in $DISTURB_CORES; do "$DISTURBER" cache "$core" 128 & PIDS+=("$!"); done
            ;;
        pagefault)
            for core in $DISTURB_CORES; do "$DISTURBER" pagefault "$core" 128 & PIDS+=("$!"); done
            ;;
        branch)
            for core in $DISTURB_CORES; do "$DISTURBER" branch "$core" 64 & PIDS+=("$!"); done
            ;;
        mixed)
            set -- $DISTURB_CORES
            "$DISTURBER" cache "${1:-0}" 128 & PIDS+=("$!")
            "$DISTURBER" pagefault "${2:-1}" 128 & PIDS+=("$!")
            "$DISTURBER" syscall "${3:-2}" 64 & PIDS+=("$!")
            "$DISTURBER" branch "${4:-4}" 64 & PIDS+=("$!")
            ;;
        *)
            echo "unknown abuse mode: $mode"
            return 1
            ;;
    esac

    sleep 0.25
}

run_one() {
    local order="$1" rep="$2" mode="$3"
    local run_id="ABUSE_${mode}_R${rep}"
    local dir="$OUTPUT_DIR/$run_id"
    local exit_code=0
    mkdir -p "$dir"

    echo "$order,$run_id,$rep,$mode,${#PIDS[@]}" >> "$ORDER"

    lock_pstate "$run_id" || return 1
    start_abuse "$mode" || return 1

    "$BIN" --core "$MEAS_CORE" --worker-mode none \
        --iterations "$ITERATIONS" --tape-size "$FIXED_TAPE" \
        --freq-label "$FREQ_LABEL" --vid-label unknown \
        --run-id "$run_id" --output-dir "$dir" \
        --stress-id "ABUSE_${mode}" \
        --stress-label "BOUNDARY_ABUSE_${mode}" \
        --stress-level "$order" --temp-limit "$TEMP_LIMIT" || exit_code=$?

    cleanup_disturbers
    return "$exit_code"
}

FAILED=0
order=0
for rep in 1 2; do
    for mode in quiet syscall cache pagefault branch mixed; do
        run_one "$order" "$rep" "$mode" || FAILED=1
        order=$((order + 1))
    done
done

ANALYSIS_FAILED=0
for dir in "$OUTPUT_DIR"/ABUSE_*; do
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
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (denx * deny) if denx and deny else 0.0

rows = []
for name in sorted(os.listdir(outdir)):
    if not name.startswith("ABUSE_"):
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
        "abuse_mode": parts[1],
        "repeat": int(parts[2][1:]),
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
    fields = ["run_id", "abuse_mode", "repeat", "boundary_thickness",
              "mean_radius", "cycle_cv", "spike_rate", "p99_p50_ratio",
              "restoration_failures", "stable_thickness", "spike_free_thickness"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

thickness = [r["boundary_thickness"] for r in rows]
cv = [r["cycle_cv"] for r in rows]
spike = [r["spike_rate"] for r in rows]
p99p50 = [r["p99_p50_ratio"] for r in rows]
fails = sum(r["restoration_failures"] for r in rows)

r_cv = corr(cv, thickness)
r_spike = corr(spike, thickness)
r_tail = corr(p99p50, thickness)
max_row = max(rows, key=lambda r: r["boundary_thickness"]) if rows else None
quiet = [r for r in rows if r["abuse_mode"] == "quiet"]
quiet_mean = statistics.mean([r["boundary_thickness"] for r in quiet]) if quiet else 0.0
max_ratio = (max_row["boundary_thickness"] / quiet_mean) if max_row and quiet_mean else 0.0

if fails:
    verdict = "RESTORATION_EDGE_REACHED"
elif max_ratio >= 3.0 and abs(r_cv) >= 0.4:
    verdict = "CARRIER_SATURATION_EDGE_ADVANCED"
elif abs(r_cv) >= 0.5 and abs(r_cv) > abs(r_spike):
    verdict = "TIMING_CV_CARRIER_RECONFIRMED_UNDER_ABUSE"
elif abs(r_spike) >= 0.4:
    verdict = "SPIKE_ARTIFACT_DOMINANT_UNDER_ABUSE"
else:
    verdict = "ABUSE_NO_NEW_EDGE"

with open(report_path, "w", newline="") as f:
    f.write("# Phase 5.9 Boundary Abuse Probe\n\n")
    f.write(f"Verdict: `{verdict}`\n\n")
    f.write("Objective: creatively push the carrier/failure boundary with software substrate abuse while avoiding voltage writes, BIOS flash, and destructive hardware actions.\n\n")
    f.write("## Metrics\n\n")
    f.write(f"- Runs analyzed: {len(rows)}\n")
    f.write(f"- Restoration failures: {fails}\n")
    f.write(f"- r(boundary_thickness, cycle_cv): {r_cv:.6f}\n")
    f.write(f"- r(boundary_thickness, spike_rate): {r_spike:.6f}\n")
    f.write(f"- r(boundary_thickness, p99_p50): {r_tail:.6f}\n")
    if max_row:
        f.write(f"- Max thickness run: {max_row['run_id']} = {max_row['boundary_thickness']:.6f}\n")
        f.write(f"- Max/quiet thickness ratio: {max_ratio:.6f}\n")
    f.write("\n## Rows\n\n")
    f.write("| Run | Mode | Thickness | CV | Spike rate | p99/p50 |\n")
    f.write("|-----|------|-----------|----|------------|--------|\n")
    for r in rows:
        f.write(f"| {r['run_id']} | {r['abuse_mode']} | {r['boundary_thickness']:.6f} | {r['cycle_cv']:.6f} | {r['spike_rate']:.6f} | {r['p99_p50_ratio']:.6f} |\n")
PY

rm -f "$DISTURBER"

if [ "$FAILED" -ne 0 ] || [ "$ANALYSIS_FAILED" -ne 0 ]; then
    exit 1
fi
