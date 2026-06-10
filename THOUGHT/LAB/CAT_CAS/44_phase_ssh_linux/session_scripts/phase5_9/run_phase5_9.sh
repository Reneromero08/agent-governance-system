#!/bin/bash
# run_phase5_9.sh — Phase 5.9 Boundary Stress Test orchestration
# Drives the stress ladder: tape sizes, worker modes, frequency sweep, thermal monitoring.
#
# Environment variables:
#   MEAS_CORE=3          Measurement core
#   ITERATIONS=50000     Trials per run
#   TAPE_SIZES="256 512 1024 2048 4096"  Tape sizes
#   WORKERS="0,1,2,4"    Worker core list (Phenom II valid cores)
#   TEMP_LIMIT=65        Thermal abort threshold C
#   OUTPUT_DIR="./output" Output directory
#
# Usage: bash run_phase5_9.sh

set +e +u -o pipefail

MEAS_CORE="${MEAS_CORE:-3}"
ITERATIONS="${ITERATIONS:-50000}"
TAPE_SIZES="${TAPE_SIZES:-256 512 1024 2048 4096}"
WORKERS="${WORKERS:-0,1,2,4}"
TEMP_LIMIT="${TEMP_LIMIT:-65}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

BIN="./phase5_9_stress_ladder"

# ── Build ─────────────────────────────────────────────────────
echo "=== BUILD ==="
make clean 2>/dev/null || true
make all
echo "Build complete."
echo ""

mkdir -p "$OUTPUT_DIR"

# ── Status tracking ───────────────────────────────────────────
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

run_stress_point() {
    local run_id="$1" stress_id="$2" stress_label="$3" stress_level="$4"
    local worker_mode="$5" tape_size="$6" freq_label="$7" vid_label="$8"
    local dir="$OUTPUT_DIR/$run_id"
    mkdir -p "$dir"

    local worker_args=""
    if [ "$worker_mode" != "none" ]; then
        worker_args="--workers $WORKERS --worker-mode $worker_mode"
    fi

    echo "--- $run_id ---"
    echo "  stress=$stress_id lvl=$stress_level tape=$tape_size workers=$worker_mode freq=$freq_label"

    local exit_code=0
    $BIN \
        --core "$MEAS_CORE" \
        $worker_args \
        --iterations "$ITERATIONS" \
        --tape-size "$tape_size" \
        --freq-label "$freq_label" \
        --vid-label "$vid_label" \
        --run-id "$run_id" \
        --output-dir "$dir" \
        --stress-id "$stress_id" \
        --stress-label "$stress_label" \
        --stress-level "$stress_level" \
        --temp-limit "$TEMP_LIMIT" \
        2>&1 || exit_code=$?

    RUN_EXIT_CODE["$run_id"]=$exit_code
    if [ $exit_code -eq 2 ]; then
        write_status "$run_id" "THERMAL_ABORT" "$exit_code"
        FAILED+=("$run_id")
        echo "  $run_id THERMAL_ABORT"
        return 2  # signal thermal abort
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
# STRESS LADDER EXECUTION
# ═══════════════════════════════════════════════════════════════

echo "=== PHASE 5.9 STRESS LADDER ==="
echo "Measurement core: $MEAS_CORE"
echo "Worker cores: $WORKERS"
echo "Thermal limit: ${TEMP_LIMIT}C"
echo "Tape sizes: $TAPE_SIZES"
echo ""

# ── A. BASELINE: no workers, nominal frequency ────────────────
echo "=== A: BASELINE (stable, no stress) ==="
for tape in $TAPE_SIZES; do
    run_stress_point "BASELINE_T${tape}" "S00_BASELINE" "BASELINE_NO_STRESS" 0 \
        "none" "$tape" "nominal" "unknown" || true
    stress_order=$((stress_order + 1))
done

# ── B. FREQUENCY STRESS ───────────────────────────────────────
echo ""
echo "=== B: FREQUENCY STRESS ==="
# Attempt MSR P-state control for frequency sweep
FREQ_AVAILABLE=0
if [ -x /usr/sbin/wrmsr ] && [ -c /dev/cpu/0/msr ]; then
    FREQ_AVAILABLE=1
fi

FREQS=()
if [ $FREQ_AVAILABLE -eq 1 ]; then
    # Try P-state sweep
    for pstate in 0 1 2 3 4; do
        case $pstate in
            0) freq_mhz=3600; label="3600MHz_P0" ;;
            1) freq_mhz=3200; label="3200MHz_P1" ;;
            2) freq_mhz=2400; label="2400MHz_P2" ;;
            3) freq_mhz=1600; label="1600MHz_P3" ;;
            4) freq_mhz=800;  label="800MHz_P4" ;;
        esac
        FREQS+=("$pstate:$freq_mhz:$label")
    done
    echo "Frequency control available: ${#FREQS[@]} P-states"
else
    echo "Frequency control unavailable — running nominal only"
    FREQS+=("0:nominal:nominal")
fi

for freq_entry in "${FREQS[@]}"; do
    IFS=':' read -r pstate freq_mhz freq_label <<< "$freq_entry"
    if [ "$freq_label" = "nominal" ]; then
        :
    else
        echo "Setting core $MEAS_CORE to P$pstate (${freq_mhz}MHz)..."
        wrmsr -p "$MEAS_CORE" 0xC0010062 "$pstate" 2>/dev/null || true
        sleep 0.2
    fi

    for tape in 256 1024 4096; do
        run_stress_point "FREQ_${freq_label}_T${tape}" \
            "S01_FREQ_${freq_label}" "FREQUENCY_STRESS_${freq_label}" \
            "$((stress_order + 1))" \
            "none" "$tape" "$freq_label" "unknown" || true
        stress_order=$((stress_order + 1))
    done

    # Restore P0 after each freq
    if [ "$freq_label" != "nominal" ]; then
        wrmsr -p "$MEAS_CORE" 0xC0010062 0 2>/dev/null || true
    fi
done

# ── C. WORKER/LOAD STRESS ─────────────────────────────────────
echo ""
echo "=== C: WORKER/LOAD STRESS ==="
WORKER_MODES="cache mixed"
for tape in $TAPE_SIZES; do
    for mode in $WORKER_MODES; do
        mode_upper=$(echo "$mode" | tr '[:lower:]' '[:upper:]')
        run_stress_point "WORKER_${mode_upper}_T${tape}" \
            "S02_WORKER_${mode_upper}" "WORKER_STRESS_${mode_upper}" \
            "$((stress_order + 1))" \
            "$mode" "$tape" "nominal" "unknown" || true
        stress_order=$((stress_order + 1))
    done
done

# ── D. TAPE PRESSURE (large tapes) ────────────────────────────
echo ""
echo "=== D: TAPE PRESSURE ==="
LARGE_TAPES="8192 16384"
for tape in $LARGE_TAPES; do
    # Check if tape size is within safe limits
    if [ $tape -le 32768 ]; then
        run_stress_point "TAPE_STRESS_T${tape}" \
            "S03_TAPE_T${tape}" "TAPE_PRESSURE_${tape}B" \
            "$((stress_order + 1))" \
            "none" "$tape" "nominal" "unknown" || true
        stress_order=$((stress_order + 1))
    fi
done

# ── E. COMBINED STRESS (worst case: cache + freq) ─────────────
echo ""
echo "=== E: COMBINED STRESS ==="
if [ $FREQ_AVAILABLE -eq 1 ]; then
    # P2 (2400MHz) + cache workers
    wrmsr -p "$MEAS_CORE" 0xC0010062 2 2>/dev/null || true
    sleep 0.2
    run_stress_point "COMBINED_P2_CACHE_T256" \
        "S04_COMBINED" "COMBINED_FREQ_CACHE_STRESS" \
        "$((stress_order + 1))" \
        "cache" "256" "2400MHz_P2" "unknown" || true
    stress_order=$((stress_order + 1))
    wrmsr -p "$MEAS_CORE" 0xC0010062 0 2>/dev/null || true
else
    run_stress_point "COMBINED_CACHE_T4096" \
        "S04_COMBINED" "COMBINED_CACHE_TAPE_STRESS" \
        "$((stress_order + 1))" \
        "cache" "4096" "nominal" "unknown" || true
    stress_order=$((stress_order + 1))
fi

# ═══════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "PHASE 5.9 STRESS LADDER COMPLETE"
echo "============================================================"
echo "Completed: ${#COMPLETED[@]} runs"
for r in "${COMPLETED[@]}"; do echo "  PASS: $r"; done
echo ""
echo "Failed:   ${#FAILED[@]} runs"
for r in "${FAILED[@]}"; do echo "  FAIL: $r (exit=${RUN_EXIT_CODE[$r]})"; done
echo ""

# ── Analyze completed runs ────────────────────────────────────
if command -v python3 &>/dev/null; then
    echo "=== ANALYZING ==="
    for run_id in "${COMPLETED[@]}"; do
        dir="$OUTPUT_DIR/$run_id"
        if [ -f "$dir/raw_cycles.csv" ] && [ -f "analyze_phase5_9.py" ]; then
            echo "  $run_id ..."
            python3 analyze_phase5_9.py --input-dir "$dir" || true
        fi
    done

    # ── Cross-run aggregation ─────────────────────────────────
    if [ -f "aggregate_phase5_9.py" ]; then
        echo ""
        echo "=== CROSS-RUN AGGREGATION ==="
        python3 aggregate_phase5_9.py --output-dir "$OUTPUT_DIR" || true
    fi
fi

echo ""
echo "Output: $OUTPUT_DIR/"
echo "============================================================"
