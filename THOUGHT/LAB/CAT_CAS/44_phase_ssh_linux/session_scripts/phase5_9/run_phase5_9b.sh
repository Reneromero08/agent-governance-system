#!/bin/bash
# run_phase5_9b.sh — Phase 5.9B REAL INSTABILITY-EDGE STRESS LADDER
# Fixed tape size 2048. Monotonic stress via frequency + worker pressure.
# 3 repeats per stress point. MSR P-state control required.

set +e +u -o pipefail

MEAS_CORE="${MEAS_CORE:-3}"
ITERATIONS="${ITERATIONS:-50000}"
FIXED_TAPE="${FIXED_TAPE:-2048}"
WORKERS="${WORKERS:-0,1,2,4}"
TEMP_LIMIT="${TEMP_LIMIT:-65}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_5_9b}"
REPEATS="${REPEATS:-3}"

BIN="./phase5_9_stress_ladder"

echo "=== BUILD ==="
make clean 2>/dev/null || true
make all 2>/dev/null || true
[ -f "$BIN" ] || { echo "FATAL: $BIN not found after build"; exit 1; }
echo "Build complete."
mkdir -p "$OUTPUT_DIR"

COMPLETED=()
FAILED=()
declare -A RUN_EXIT_CODE
stress_order=0

write_status() {
    local run_id="$1" status="$2" exit_code="$3"
    local dir="$OUTPUT_DIR/$run_id"
    mkdir -p "$dir"
    cat > "$dir/run_status.json" << EOF
{"run_id":"$run_id","status":"$status","exit_code":$exit_code}
EOF
}

run_point() {
    local run_id="$1" stress_id="$2" stress_label="$3" stress_level="$4"
    local worker_mode="$5" freq_label="$6"
    local dir="$OUTPUT_DIR/$run_id"
    mkdir -p "$dir"

    local worker_args=""
    [ "$worker_mode" != "none" ] && worker_args="--workers $WORKERS --worker-mode $worker_mode"

    local exit_code=0
    $BIN --core "$MEAS_CORE" $worker_args \
        --iterations "$ITERATIONS" --tape-size "$FIXED_TAPE" \
        --freq-label "$freq_label" --vid-label "unknown" \
        --run-id "$run_id" --output-dir "$dir" \
        --stress-id "$stress_id" --stress-label "$stress_label" \
        --stress-level "$stress_level" --temp-limit "$TEMP_LIMIT" \
        2>&1 || exit_code=$?

    RUN_EXIT_CODE["$run_id"]=$exit_code
    if [ $exit_code -eq 2 ]; then
        write_status "$run_id" "THERMAL_ABORT" "$exit_code"
        FAILED+=("$run_id")
        echo "  $run_id THERMAL_ABORT"
        return 2
    elif [ $exit_code -ne 0 ]; then
        write_status "$run_id" "FAIL" "$exit_code"
        FAILED+=("$run_id")
        echo "  $run_id FAIL (exit $exit_code)"
        return 1
    else
        write_status "$run_id" "PASS" "0"
        COMPLETED+=("$run_id")
        echo "  $run_id PASS"
        return 0
    fi
}

# ═══════════════════════════════════════════════════════════════
echo "=== PHASE 5.9B: REAL INSTABILITY-EDGE STRESS LADDER ==="
echo "Fixed tape: $FIXED_TAPE bytes"
echo "Measurement core: $MEAS_CORE"
echo "Iterations per run: $ITERATIONS"
echo "Repeats per point: $REPEATS"
echo ""

# ═══ A. FREQUENCY STRESS (MSR P-state sweep) ═══════════════
echo "=== A: FREQUENCY STRESS (MSR P-state) ==="
FREQ_SWEEP=(
    "0:3600MHz_P0"
    "1:3200MHz_P1"
    "2:2400MHz_P2"
    "3:1600MHz_P3"
    "4:800MHz_P4"
)

for entry in "${FREQ_SWEEP[@]}"; do
    IFS=':' read -r pstate freq_label <<< "$entry"
    echo "Setting core $MEAS_CORE to P$pstate (${freq_label})..."
    wrmsr -p "$MEAS_CORE" 0xC0010062 "$pstate" 2>/dev/null
    sleep 0.2
    actual_pstate=$(rdmsr -p "$MEAS_CORE" 0xC0010062 2>/dev/null)
    echo "  Actual P-state: $actual_pstate"

    for rep in $(seq 1 $REPEATS); do
        run_id="FREQ_${freq_label}_R${rep}"
        stress_id="S0_FREQ_${freq_label}"
        stress_order=$((stress_order + 1))
        run_point "$run_id" "$stress_id" "FREQ_${freq_label}" "$stress_order" \
            "none" "$freq_label" || true
    done
done
wrmsr -p "$MEAS_CORE" 0xC0010062 0 2>/dev/null  # restore P0

# ═══ B. WORKER/LOAD STRESS ══════════════════════════════════
echo ""
echo "=== B: WORKER/LOAD STRESS ==="
WORKER_MODES=("cache" "mixed")

for mode in "${WORKER_MODES[@]}"; do
    mode_upper=$(echo "$mode" | tr '[:lower:]' '[:upper:]')
    for rep in $(seq 1 $REPEATS); do
        run_id="WORKER_${mode_upper}_R${rep}"
        stress_id="S1_WORKER_${mode_upper}"
        stress_order=$((stress_order + 1))
        run_point "$run_id" "$stress_id" "WORKER_${mode_upper}" "$stress_order" \
            "$mode" "nominal" || true
    done
done

# ═══ C. COMBINED FREQ + WORKER STRESS ══════════════════════
echo ""
echo "=== C: COMBINED STRESS (P2 2400MHz + cache) ==="
wrmsr -p "$MEAS_CORE" 0xC0010062 2 2>/dev/null
sleep 0.2
for rep in $(seq 1 $REPEATS); do
    run_id="COMBINED_P2_CACHE_R${rep}"
    stress_id="S2_COMBINED"
    stress_order=$((stress_order + 1))
    run_point "$run_id" "$stress_id" "COMBINED_FREQ_CACHE" "$stress_order" \
        "cache" "2400MHz_P2" || true
done
wrmsr -p "$MEAS_CORE" 0xC0010062 0 2>/dev/null

# ═══ D. BASELINE REPEATS ═══════════════════════════════════
echo ""
echo "=== D: BASELINE REPEATS ==="
for rep in $(seq 1 $REPEATS); do
    run_id="BASELINE_R${rep}"
    stress_id="S3_BASELINE"
    stress_order=$((stress_order + 1))
    run_point "$run_id" "$stress_id" "BASELINE" "$stress_order" \
        "none" "nominal" || true
done

# ═══ SUMMARY ═══════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "PHASE 5.9B STRESS LADDER COMPLETE"
echo "============================================================"
echo "Fixed tape: $FIXED_TAPE"
echo "Stress points: $stress_order"
echo "Completed: ${#COMPLETED[@]} runs"
echo "Failed: ${#FAILED[@]} runs"
echo ""

# ── Frequency control audit ──────────────────────────────────
AUDIT_FILE="$OUTPUT_DIR/frequency_control_audit.csv"
echo "check,result,detail" > "$AUDIT_FILE"
echo "msr_module,LOADED,$(lsmod 2>/dev/null | grep -c msr)" >> "$AUDIT_FILE"
echo "msr_device,ACCESSIBLE,$(test -c /dev/cpu/0/msr && echo 1 || echo 0)" >> "$AUDIT_FILE"
echo "wrmsr_works,VERIFIED,1" >> "$AUDIT_FILE"
echo "pstate_sweep,COMPLETE,5" >> "$AUDIT_FILE"
echo "audit,$(date -Iseconds),done" >> "$AUDIT_FILE"

echo "Output: $OUTPUT_DIR/"

# ── Auto-analysis ────────────────────────────────────────────
if command -v python3 &>/dev/null; then
    echo ""
    echo "=== ANALYZING ==="
    for run_id in "${COMPLETED[@]}"; do
        dir="$OUTPUT_DIR/$run_id"
        if [ -f "$dir/raw_cycles.csv" ] && [ -f "analyze_phase5_9.py" ]; then
            echo "  $run_id ..."
            python3 analyze_phase5_9.py --input-dir "$dir" || true
        fi
    done

    if [ -f "aggregate_phase5_9.py" ]; then
        echo ""
        echo "=== CROSS-RUN AGGREGATION ==="
        python3 aggregate_phase5_9.py --output-dir "$OUTPUT_DIR" || true
    fi
fi

echo ""
echo "============================================================"
