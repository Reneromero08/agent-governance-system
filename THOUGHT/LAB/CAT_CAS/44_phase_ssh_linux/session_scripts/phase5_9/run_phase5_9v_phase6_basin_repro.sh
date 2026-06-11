#!/usr/bin/env bash
set -u

# Phase 5.9V -> Phase 6 basin reproducibility matrix.
# Runs at P4 VID+5 and tests whether prelude classes reproducibly select a
# collapsed/mid/high carrier basin. This is a basin-control run, not a Mode C
# fixed-point crossing claim.

MEAS_CORE="${MEAS_CORE:-3}"
ALL_CORES="${ALL_CORES:-0 1 2 3 4 5}"
DEF_CORES="${DEF_CORES:-$MEAS_CORE}"
DISTURB_CORES="${DISTURB_CORES:-0 1 2 4}"
ITERATIONS="${ITERATIONS:-30000}"
WINDOW_SIZE="${WINDOW_SIZE:-256}"
FIXED_TAPE="${FIXED_TAPE:-2048}"
REPEATS="${REPEATS:-10}"
VID_OFFSET="${VID_OFFSET:-5}"
SELECTORS="${SELECTORS:-quiet syscall_prelude cache_prelude branch_prelude public_kb_prelude shuffled_kb_prelude d_oracle_prelude}"
PRELUDE_SECONDS="${PRELUDE_SECONDS:-1.0}"
TARGET_N="${TARGET_N:-12}"
TARGET_SEED="${TARGET_SEED:-6012}"
TARGET_M_FACTOR="${TARGET_M_FACTOR:-2.0}"
COUPLED_WORKLOAD="${COUPLED_WORKLOAD:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/exp44_k10_voltage_probe/p4_vid5_phase6_basin_repro}"
PHASE59_DIR="${PHASE59_DIR:-/root/exp44_phase5_9}"
TEMP_LIMIT="${TEMP_LIMIT:-65}"
PSTATE_CTL="0xC0010062"
P4_DEF="0xC0010068"
BIN="$PHASE59_DIR/phase5_9_stress_ladder"
ANALYZER="$PHASE59_DIR/analyze_phase5_9c.py"
DISTURBER="$PHASE59_DIR/boundary_abuse_disturber"

mkdir -p "$OUTPUT_DIR"

cd "$PHASE59_DIR" || exit 1
make all

if [ ! -x "$BIN" ]; then
    echo "FATAL: missing $BIN"
    exit 1
fi
if [ ! -f "$ANALYZER" ]; then
    echo "FATAL: missing $ANALYZER"
    exit 1
fi
if [ ! -x "$DISTURBER" ]; then
    if [ -f "$PHASE59_DIR/boundary_abuse_disturber.c" ]; then
        gcc -O2 -march=native -Wall -Wextra -std=c11 -o "$DISTURBER" "$PHASE59_DIR/boundary_abuse_disturber.c" -lm
    fi
fi
if [ ! -x "$DISTURBER" ]; then
    echo "FATAL: missing $DISTURBER"
    exit 1
fi

if ! command -v rdmsr >/dev/null 2>&1 || ! command -v wrmsr >/dev/null 2>&1; then
    echo "FATAL: rdmsr/wrmsr missing"
    exit 1
fi
modprobe msr 2>/dev/null || true
if ! rdmsr -p "$MEAS_CORE" "$P4_DEF" >/dev/null 2>&1; then
    echo "FATAL: MSR unavailable"
    exit 1
fi

ORIG_P4="$(rdmsr -p "$MEAS_CORE" "$P4_DEF")"
TARGET_P4="$(python3 - "$ORIG_P4" "$VID_OFFSET" <<'PY'
import sys
raw = int(sys.argv[1], 16)
offset = int(sys.argv[2])
vid = (raw >> 9) & 0x7f
new_vid = min(0x7f, vid + offset)
new_raw = (raw & ~(0x7f << 9)) | (new_vid << 9)
print(f"{new_raw:016x}")
PY
)"
TARGET_VCORE="$(python3 - "$TARGET_P4" <<'PY'
import sys
raw = int(sys.argv[1], 16)
vid = (raw >> 9) & 0x7f
print(f"{1.55 - 0.0125 * vid:.4f}")
PY
)"

PIDS=()
cleanup_disturbers() {
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" >/dev/null 2>&1 || true
    done
    wait >/dev/null 2>&1 || true
    PIDS=()
}

restore_p4() {
    cleanup_disturbers
    for core in $ALL_CORES; do
        wrmsr -p "$core" "$PSTATE_CTL" 0 2>/dev/null || true
    done
    for core in $DEF_CORES; do
        wrmsr -p "$core" "$P4_DEF" "0x$ORIG_P4" 2>/dev/null || true
    done
}
trap restore_p4 EXIT

set_vid5_p4() {
    local run_id="$1"
    for core in $DEF_CORES; do
        local before after ctl
        before="$(rdmsr -p "$core" "$P4_DEF" 2>/dev/null || echo READ_FAIL)"
        wrmsr -p "$core" "$PSTATE_CTL" 0 2>/dev/null || true
        wrmsr -p "$core" "$P4_DEF" "0x$TARGET_P4" 2>/dev/null || return 1
        wrmsr -p "$core" "$PSTATE_CTL" 4 2>/dev/null || true
        after="$(rdmsr -p "$core" "$P4_DEF" 2>/dev/null || echo READ_FAIL)"
        ctl="$(rdmsr -p "$core" "$PSTATE_CTL" 2>/dev/null || echo READ_FAIL)"
        echo "$run_id,$core,$before,$after,$ctl" >> "$OUTPUT_DIR/msr_audit.csv"
        [ "$after" = "$TARGET_P4" ] || return 1
    done
    return 0
}

start_disturbers() {
    local mode="$1"
    cleanup_disturbers
    case "$mode" in
        quiet)
            ;;
        syscall_prelude)
            for core in $DISTURB_CORES; do "$DISTURBER" syscall "$core" 64 & PIDS+=("$!"); done
            ;;
        cache_prelude)
            for core in $DISTURB_CORES; do "$DISTURBER" cache "$core" 128 & PIDS+=("$!"); done
            ;;
        branch_prelude)
            for core in $DISTURB_CORES; do "$DISTURBER" branch "$core" 64 & PIDS+=("$!"); done
            ;;
        public_kb_prelude|shuffled_kb_prelude|wrong_kb_prelude|d_oracle_prelude)
            run_logical_prelude "$mode"
            ;;
        public_kb_syscall_prelude|shuffled_kb_syscall_prelude|wrong_kb_syscall_prelude|d_oracle_syscall_prelude)
            run_logical_prelude "$mode"
            for core in $DISTURB_CORES; do "$DISTURBER" syscall "$core" 64 & PIDS+=("$!"); done
            ;;
        public_kb_cache_prelude|shuffled_kb_cache_prelude|wrong_kb_cache_prelude)
            run_logical_prelude "$mode"
            for core in $DISTURB_CORES; do "$DISTURBER" cache "$core" 128 & PIDS+=("$!"); done
            ;;
        public_kb_branch_prelude|shuffled_kb_branch_prelude|wrong_kb_branch_prelude)
            run_logical_prelude "$mode"
            for core in $DISTURB_CORES; do "$DISTURBER" branch "$core" 64 & PIDS+=("$!"); done
            ;;
        *)
            echo "unknown selector $mode"
            return 1
            ;;
    esac
    sleep "$PRELUDE_SECONDS"
}

target_payload() {
    local mode="$1"
    python3 - "$mode" "$TARGET_N" "$TARGET_SEED" "$TARGET_M_FACTOR" <<'PY'
import hashlib
import math
import random
import sys

mode = sys.argv[1]
n = int(sys.argv[2])
seed = int(sys.argv[3])
m_factor = float(sys.argv[4])
nspace = 1 << n
m = min(max(4, int(round(m_factor * math.sqrt(nspace)))), nspace // 2 - 1)
rng = random.Random(seed)
d = rng.randrange(1, nspace // 2)
ks = rng.sample(range(1, nspace // 2), min(m, nspace // 2 - 1))
bs = []
for k in ks:
    c = math.cos(2.0 * math.pi * k * d / nspace)
    bs.append(1 if c >= 0.0 else -1)

if mode.startswith("shuffled_kb"):
    local = random.Random(seed ^ 0x5A17)
    local.shuffle(bs)
elif mode.startswith("wrong_kb"):
    wrong_d = (d + max(1, nspace // 8)) % (nspace // 2)
    if wrong_d == 0:
        wrong_d = 1
    bs = [1 if math.cos(2.0 * math.pi * k * wrong_d / nspace) >= 0.0 else -1 for k in ks]
elif mode.startswith("d_oracle"):
    ks = [(k * d) % (nspace // 2) or 1 for k in ks]

payload = ";".join(f"{k}:{b}" for k, b in zip(ks, bs))
h = hashlib.sha256(payload.encode("ascii")).hexdigest()
weight = sum((idx + 1) * (1 if b > 0 else 3) * ((k % 17) + 1) for idx, (k, b) in enumerate(zip(ks, bs)))
tape = 1024 + 64 * (weight % 33)
iters_delta = 5000 * (weight % 5)
stress_level = weight % 997
print(f"{h},{tape},{iters_delta},{stress_level},{d},{m}")
PY
}

run_logical_prelude() {
    local mode="$1"
    python3 - "$mode" "$PRELUDE_SECONDS" "$TARGET_N" "$TARGET_SEED" "$TARGET_M_FACTOR" >/dev/null <<'PY'
import hashlib, math, random, sys, time
mode = sys.argv[1]
duration = float(sys.argv[2])
n = int(sys.argv[3])
seed0 = int(sys.argv[4])
m_factor = float(sys.argv[5])
nspace = 1 << n
m = min(max(4, int(round(m_factor * math.sqrt(nspace)))), nspace // 2 - 1)
rng = random.Random(seed0)
d = rng.randrange(1, nspace // 2)
ks = rng.sample(range(1, nspace // 2), min(m, nspace // 2 - 1))
bs = [1 if math.cos(2.0 * math.pi * k * d / nspace) >= 0.0 else -1 for k in ks]
if mode.startswith("shuffled_kb"):
    local = random.Random(seed0 ^ 0x5A17)
    local.shuffle(bs)
elif mode.startswith("wrong_kb"):
    wrong_d = (d + max(1, nspace // 8)) % (nspace // 2)
    if wrong_d == 0:
        wrong_d = 1
    bs = [1 if math.cos(2.0 * math.pi * k * wrong_d / nspace) >= 0.0 else -1 for k in ks]
elif mode.startswith("d_oracle"):
    ks = [(k * d) % (nspace // 2) or 1 for k in ks]
payload = ";".join(f"{k}:{b}" for k, b in zip(ks, bs))
seed = int(hashlib.sha256((mode + "|" + payload).encode("ascii")).hexdigest()[:8], 16)
x = seed | 1
acc = 0.0
end = time.time() + duration
while time.time() < end:
    x ^= (x << 13) & 0xffffffff
    x ^= (x >> 17)
    x ^= (x << 5) & 0xffffffff
    idx = x % len(ks)
    phase = (ks[idx] * (x & (nspace - 1))) / nspace
    if mode.startswith("public_kb") or mode.startswith("wrong_kb"):
        acc += bs[idx] * math.cos(2.0 * math.pi * phase)
    elif mode.startswith("shuffled_kb"):
        acc += bs[idx] * math.sin(2.0 * math.pi * phase)
    else:
        acc += math.sqrt(((x ^ (ks[idx] * max(1, d))) & 65535) + 1.0)
print(acc)
PY
}

echo "run_id,core,p4_before,p4_after,pstate_ctl_after" > "$OUTPUT_DIR/msr_audit.csv"
echo "order,run_id,selector,repeat,vid_offset,decoded_voltage,target_p4,target_hash,target_n,target_d,target_m,run_tape,run_iterations,coupled_workload" > "$OUTPUT_DIR/condition_order.csv"

FAILED=0
order=0
selectors="$SELECTORS"

for rep in $(seq 1 "$REPEATS"); do
    for selector in $selectors; do
        order=$((order + 1))
        run_id="P59V_${selector}_R${rep}"
        run_dir="$OUTPUT_DIR/$run_id"
        mkdir -p "$run_dir"
        payload="$(target_payload "$selector")"
        IFS=, read -r target_hash target_tape iters_delta target_stress target_d target_m <<EOF
$payload
EOF
        run_tape="$FIXED_TAPE"
        run_iterations="$ITERATIONS"
        run_stress="$order"
        if [ "$COUPLED_WORKLOAD" = "1" ]; then
            run_tape="$target_tape"
            run_iterations=$((ITERATIONS + iters_delta))
            run_stress="$target_stress"
        fi
        echo "$order,$run_id,$selector,$rep,$VID_OFFSET,$TARGET_VCORE,$TARGET_P4,$target_hash,$TARGET_N,$target_d,$target_m,$run_tape,$run_iterations,$COUPLED_WORKLOAD" >> "$OUTPUT_DIR/condition_order.csv"

        if ! set_vid5_p4 "$run_id"; then
            echo "WARN: MSR set failed for $run_id" | tee "$run_dir/status.txt"
            FAILED=1
            continue
        fi
        if ! start_disturbers "$selector"; then
            echo "WARN: prelude failed for $run_id" | tee "$run_dir/status.txt"
            FAILED=1
            continue
        fi

        "$BIN" --core "$MEAS_CORE" --worker-mode none \
            --iterations "$run_iterations" --tape-size "$run_tape" \
            --freq-label "P4" --vid-label "VID_PLUS_${VID_OFFSET}_${TARGET_VCORE}" \
            --run-id "$run_id" --output-dir "$run_dir" \
            --stress-id "P59V_${selector}" \
            --stress-label "PHASE5_9V_${selector}" \
            --stress-level "$run_stress" --temp-limit "$TEMP_LIMIT" || FAILED=1
        cleanup_disturbers
    done
done

restore_p4

ANALYSIS_FAILED=0
for dir in "$OUTPUT_DIR"/P59V_*; do
    if [ -f "$dir/raw_cycles.csv" ]; then
        python3 "$ANALYZER" --input-dir "$dir" --window-size "$WINDOW_SIZE" || ANALYSIS_FAILED=1
    fi
done

python3 - "$OUTPUT_DIR" "$REPEATS" "$VID_OFFSET" "$TARGET_VCORE" <<'PY'
import csv
import os
import statistics
import sys
from collections import Counter, defaultdict

outdir, repeats, vid_offset, vcore = sys.argv[1:5]
rows = []
for name in sorted(os.listdir(outdir)):
    if not name.startswith("P59V_"):
        continue
    geo_path = os.path.join(outdir, name, "geometry_stats.csv")
    if not os.path.exists(geo_path):
        continue
    with open(geo_path, newline="") as f:
        geo = next(csv.DictReader(f), None)
    if not geo:
        continue
    tail = name[len("P59V_"):]
    selector, rep = tail.rsplit("_R", 1)
    thickness = float(geo.get("boundary_thickness_nn_mean", 0))
    cv = float(geo.get("cycle_cv", 0))
    p99 = float(geo.get("p99_p50_ratio", 0))
    fails = int(float(geo.get("restoration_failures", 0)))
    if thickness < 100:
        basin = "collapsed"
    elif thickness < 5000:
        basin = "mid"
    else:
        basin = "high"
    rows.append({
        "run_id": name,
        "selector": selector,
        "repeat": int(rep),
        "vid_offset": int(vid_offset),
        "decoded_voltage": vcore,
        "boundary_thickness": thickness,
        "cycle_cv": cv,
        "p99_p50": p99,
        "restoration_failures": fails,
        "basin": basin,
    })

audit_path = os.path.join(outdir, "phase5_9v_phase6_basin_repro_audit.csv")
with open(audit_path, "w", newline="") as f:
    fields = ["run_id", "selector", "repeat", "vid_offset", "decoded_voltage",
              "boundary_thickness", "cycle_cv", "p99_p50", "restoration_failures", "basin"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

groups = defaultdict(list)
for row in rows:
    groups[row["selector"]].append(row)

summary = []
for selector in sorted(groups):
    group = groups[selector]
    counts = Counter(row["basin"] for row in group)
    n = len(group)
    top_basin, top_count = counts.most_common(1)[0]
    summary.append({
        "selector": selector,
        "n": n,
        "collapsed": counts.get("collapsed", 0),
        "mid": counts.get("mid", 0),
        "high": counts.get("high", 0),
        "top_basin": top_basin,
        "top_rate": top_count / n if n else 0.0,
        "noncollapse_rate": (n - counts.get("collapsed", 0)) / n if n else 0.0,
        "anti_high_rate": (n - counts.get("high", 0)) / n if n else 0.0,
        "mean_thickness": statistics.mean([r["boundary_thickness"] for r in group]) if group else 0.0,
        "max_thickness": max([r["boundary_thickness"] for r in group]) if group else 0.0,
        "restoration_failures": sum(r["restoration_failures"] for r in group),
    })

summary_path = os.path.join(outdir, "phase5_9v_phase6_basin_repro_summary.csv")
with open(summary_path, "w", newline="") as f:
    fields = ["selector", "n", "collapsed", "mid", "high", "top_basin", "top_rate",
              "noncollapse_rate", "anti_high_rate", "mean_thickness", "max_thickness",
              "restoration_failures"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(summary)

total_fails = sum(row["restoration_failures"] for row in rows)
best = max(summary, key=lambda r: r["top_rate"], default=None)
public = next((r for r in summary if r["selector"] == "public_kb_prelude"), None)
shuffled = next((r for r in summary if r["selector"] == "shuffled_kb_prelude"), None)
oracle = next((r for r in summary if r["selector"] == "d_oracle_prelude"), None)

if total_fails:
    verdict = "PHASE5_9V_REPRO_RESTORATION_CONFOUNDED"
elif public and public["top_rate"] >= 0.8 and (not shuffled or public["top_rate"] > shuffled["top_rate"]):
    verdict = "PHASE5_9V_PUBLIC_PRELUDE_BASIN_CANDIDATE"
elif best and best["top_rate"] >= 0.8:
    verdict = "PHASE5_9V_SELECTOR_REPRODUCIBLE_NONPUBLIC"
elif any(r["noncollapse_rate"] >= 0.9 or r["anti_high_rate"] >= 0.9 for r in summary):
    verdict = "PHASE5_9V_DIRECTIONAL_REPRODUCED_NOT_DETERMINISTIC"
else:
    verdict = "PHASE5_9V_SELECTOR_NOT_REPRODUCED"

report_path = os.path.join(outdir, "PHASE5_9V_PHASE6_BASIN_REPRO.md")
with open(report_path, "w") as f:
    f.write("# Phase 5.9V Phase 6 Basin Reproducibility Matrix\n\n")
    f.write(f"Verdict: `{verdict}`\n\n")
    f.write("Objective: hold P4 VID+5 and test whether Phase 6-relevant preludes reproducibly select collapsed/mid/high carrier basins.\n\n")
    f.write("This is not a Mode C fixed-point crossing claim. The public/shuffled/oracle preludes here are substrate-prelude probes; answer-predictive coupling still requires the Phase 6 fixed-point map integration.\n\n")
    f.write(f"- VID offset: +{vid_offset}\n")
    f.write(f"- Decoded voltage: {vcore}V\n")
    f.write(f"- Rows analyzed: {len(rows)}\n")
    f.write(f"- Restoration failures: {total_fails}\n")
    f.write(f"- Requested repeats per selector: {repeats}\n")
    f.write(f"- Coupled workload: {os.environ.get('COUPLED_WORKLOAD', '0')}\n\n")
    f.write("## Selector Summary\n\n")
    f.write("| Selector | n | Collapsed | Mid | High | Top basin | Top rate | Noncollapse | Anti-high | Mean thickness | Max thickness |\n")
    f.write("|----------|---|-----------|-----|------|-----------|----------|-------------|-----------|----------------|---------------|\n")
    for r in summary:
        f.write(f"| {r['selector']} | {r['n']} | {r['collapsed']} | {r['mid']} | {r['high']} | {r['top_basin']} | {r['top_rate']:.3f} | {r['noncollapse_rate']:.3f} | {r['anti_high_rate']:.3f} | {r['mean_thickness']:.6f} | {r['max_thickness']:.6f} |\n")
    f.write("\n## Gate Readout\n\n")
    f.write(f"- Restoration: {'PASS' if total_fails == 0 else 'FAIL'}.\n")
    if public:
        f.write(f"- Public-prelude top rate: {public['top_rate']:.3f}.\n")
    if shuffled:
        f.write(f"- Shuffled-prelude top rate: {shuffled['top_rate']:.3f}.\n")
    if oracle:
        f.write(f"- Oracle-control top rate: {oracle['top_rate']:.3f}; this is a smuggle detector, not crossing evidence.\n")
    f.write("- Mode C handoff requires public-prelude basin reproducibility plus answer-predictive invariant separation; this run tests basin reproducibility only.\n")
    if os.environ.get("COUPLED_WORKLOAD", "0") == "1":
        f.write("- This run used target-coupled workload shaping: the public/shuffled/wrong/oracle payload changed prelude dynamics plus tape/iteration/stress-level parameters while preserving restoration checks.\n")

print(verdict)
print(report_path)
PY

if [ "$FAILED" -ne 0 ] || [ "$ANALYSIS_FAILED" -ne 0 ]; then
    exit 1
fi
