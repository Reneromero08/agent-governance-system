#!/bin/bash
# run_phase5_8.sh — Phase 5.8R hardened orchestration
# Nonfatal per-condition execution with randomized ordering and status files.
#
# Environment variables:
#   MEAS_CORE=3        Measurement core
#   ITERATIONS=100000  Trials per run (reduced for intermediate tape sizes if needed)
#   WINDOW_SIZE=256    Window size for analyzer
#   TAPE_SIZES="256 512 1024 2048 4096"  Tape sizes to sweep
#   WORKERS="0,1,2,4"  Worker core list (valid for Phenom II X6)
#   WORKER_MODES="none cache mixed"      Worker modes
#   FREQ_LABEL="nominal"  Frequency label
#   VID_LABEL="unknown"   VID label
#   OUTPUT_DIR="./output" Output directory
#   RANDOM_SEED=42        Seed for randomized ordering
#
# Usage:
#   bash run_phase5_8.sh

set +e +u -o pipefail  # NONFATAL — capture failures, allow unset vars

# ── Configuration ─────────────────────────────────────────────
MEAS_CORE="${MEAS_CORE:-3}"
ITERATIONS="${ITERATIONS:-100000}"
WINDOW_SIZE="${WINDOW_SIZE:-256}"
TAPE_SIZES="${TAPE_SIZES:-256 512 1024 2048 4096}"
WORKERS="${WORKERS:-0,1,2,4}"
WORKER_MODES="${WORKER_MODES:-none cache mixed}"
FREQ_LABEL="${FREQ_LABEL:-nominal}"
VID_LABEL="${VID_LABEL:-unknown}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
RANDOM_SEED="${RANDOM_SEED:-42}"
CONTROL_ITERS=10000
CTRL_TAPE=256

BIN="./phase5_8_boundary_rdtsc"

# ── Build ─────────────────────────────────────────────────────
echo "=== BUILD ==="
make clean 2>/dev/null || true
make all
echo "Build complete."
echo ""

# ── Create output directory ───────────────────────────────────
mkdir -p "$OUTPUT_DIR"

# ── Global status tracking ────────────────────────────────────
COMPLETED=()
FAILED=()
SKIPPED=()
declare -A RUN_STATUS
declare -A RUN_EXIT_CODE
declare -A RUN_SIGNAL

# ── Helper: write run status ──────────────────────────────────
write_status() {
    local run_id="$1" status="$2" exit_code="$3" signal="$4" reason="$5"
    local dir="$OUTPUT_DIR/$run_id"
    mkdir -p "$dir"

    # JSON status
    cat > "$dir/run_status.json" << EOF
{
  "run_id": "$run_id",
  "status": "$status",
  "exit_code": $exit_code,
  "signal": "$signal",
  "reason": "$reason",
  "raw_cycles_exists": $([ -f "$dir/raw_cycles.csv" ] && echo "true" || echo "false"),
  "restoration_integrity_exists": $([ -f "$dir/restoration_integrity.csv" ] && echo "true" || echo "false"),
  "window_features_exists": $([ -f "$dir/window_features.csv" ] && echo "true" || echo "false")
}
EOF

    # Text status
    echo "run_id=$run_id status=$status exit_code=$exit_code signal=$signal reason=$reason" > "$dir/run_status.txt"

    RUN_STATUS["$run_id"]="$status"
    RUN_EXIT_CODE["$run_id"]="$exit_code"
    RUN_SIGNAL["$run_id"]="$signal"
}

# ── Helper: run one configuration ─────────────────────────────
run_config() {
    local run_id="$1" worker_mode="$2" tape_size="$3"
    local freq="$4" vid="$5"
    local dir="$OUTPUT_DIR/$run_id"
    local iter="$ITERATIONS"

    mkdir -p "$dir"

    # Build worker args
    local worker_args=""
    if [ "$worker_mode" != "none" ]; then
        worker_args="--workers $WORKERS --worker-mode $worker_mode --placement offcore"
    fi

    echo "--- $run_id ---"
    echo "  tape=$tape_size  workers=$worker_mode  freq=$freq  vid=$vid"

    local exit_code=0 signal=0 reason=""

    # Run with extended timeout for large configurations
    $BIN \
        --core "$MEAS_CORE" \
        $worker_args \
        --iterations "$iter" \
        --window-size "$WINDOW_SIZE" \
        --tape-size "$tape_size" \
        --freq-label "$freq" \
        --vid-label "$vid" \
        --run-id "$run_id" \
        --output-dir "$dir" 2>&1 || exit_code=$?

    # Check for signal death
    if [ $exit_code -gt 128 ]; then
        signal=$((exit_code - 128))
        reason="SIGNAL_$signal"
        write_status "$run_id" "FAIL" "$exit_code" "$signal" "$reason"
        FAILED+=("$run_id")
        echo "  $run_id FAILED (signal $signal)"
    elif [ $exit_code -ne 0 ]; then
        reason="EXIT_$exit_code"
        write_status "$run_id" "FAIL" "$exit_code" "0" "$reason"
        FAILED+=("$run_id")
        echo "  $run_id FAILED (exit $exit_code)"
    else
        write_status "$run_id" "PASS" "0" "0" ""
        COMPLETED+=("$run_id")
        echo "  $run_id PASS"
    fi
    echo ""
}

# ── Control runs ──────────────────────────────────────────────
run_control() {
    local run_id="$1" control_mode="$2"
    local dir="$OUTPUT_DIR/$run_id"
    mkdir -p "$dir"

    echo "--- $run_id ---"
    local exit_code=0 signal=0

    $BIN --core "$MEAS_CORE" --iterations "$CONTROL_ITERS" --tape-size "$CTRL_TAPE" \
        --run-id "$run_id" --output-dir "$dir" \
        --freq-label "$FREQ_LABEL" --vid-label "$VID_LABEL" \
        --control "$control_mode" 2>&1 || exit_code=$?

    if [ $exit_code -gt 128 ]; then
        signal=$((exit_code - 128))
        write_status "$run_id" "FAIL" "$exit_code" "$signal" "SIGNAL_$signal"
        FAILED+=("$run_id")
        echo "  $run_id FAILED (signal $signal)"
    elif [ $exit_code -ne 0 ]; then
        write_status "$run_id" "FAIL" "$exit_code" "0" "EXIT_$exit_code"
        FAILED+=("$run_id")
        echo "  $run_id FAILED (exit $exit_code)"
    else
        write_status "$run_id" "PASS" "0" "0" ""
        COMPLETED+=("$run_id")
        echo "  $run_id PASS"
    fi
    echo ""
}

# ── Generate randomized condition order ───────────────────────
echo "=== GENERATING CONDITION ORDER ==="
CONDITIONS=()

# Build condition list: all tape × mode combinations, then controls
for tape in $TAPE_SIZES; do
    for mode in $WORKER_MODES; do
        label=$(echo "$mode" | tr '[:lower:]' '[:upper:]')
        CONDITIONS+=("TAPE=${tape},MODE=${mode},ID=${label}_T${tape}_NOMINAL")
    done
done

# Controls always at the end
CONTROL_LIST=(
    "CTRL=empty,ID=CONTROL_EMPTY_T256"
    "CTRL=nop,ID=CONTROL_NOP_T256"
    "CTRL=irreversible,ID=CONTROL_IRREVERSIBLE_T256"
    "CTRL=readonly,ID=CONTROL_READONLY_T256"
)

# Shuffle conditions using awk with fixed seed
ORDER_FILE="$OUTPUT_DIR/condition_order.csv"
echo "order_index,run_id,tape_size,worker_mode,control_mode" > "$ORDER_FILE"

ORDER_INDEX=0
# Simple deterministic shuffle: generate indices, sort by hash
for cond in "${CONDITIONS[@]}"; do
    echo "$ORDER_INDEX $cond"
    ORDER_INDEX=$((ORDER_INDEX + 1))
done | while read idx cond; do
    hash=$(echo "${RANDOM_SEED}_${cond}" | md5sum | cut -c1-8)
    echo "$hash $idx $cond"
done | sort | awk '{print NR-1, $3}' > "$OUTPUT_DIR/_shuffled.txt"

# Re-read shuffled conditions
SHUFFLED=()
while IFS=' ' read -r order cond; do
    SHUFFLED+=("$cond")
done < "$OUTPUT_DIR/_shuffled.txt"

# If shuffle failed (no md5sum etc), use original order
if [ ${#SHUFFLED[@]} -eq 0 ]; then
    SHUFFLED=("${CONDITIONS[@]}")
fi

# ── Execute shuffled matrix (nonfatal) ────────────────────────
echo "=== EXECUTING RUN MATRIX (${#SHUFFLED[@]} conditions + ${#CONTROL_LIST[@]} controls) ==="
echo ""

order_idx=0
for cond in "${SHUFFLED[@]}"; do
    # Parse condition
    tape=$(echo "$cond" | grep -oP 'TAPE=\K[^,]+')
    mode=$(echo "$cond" | grep -oP 'MODE=\K[^,]+')
    run_id=$(echo "$cond" | grep -oP 'ID=\K.*')

    echo "$order_idx,$run_id,$tape,$mode," >> "$ORDER_FILE"

    if [ "$mode" = "none" ]; then
        run_config "$run_id" "none" "$tape" "$FREQ_LABEL" "$VID_LABEL"
    elif [ "$mode" = "cache" ]; then
        run_config "$run_id" "cache" "$tape" "$FREQ_LABEL" "$VID_LABEL"
    elif [ "$mode" = "mixed" ]; then
        run_config "$run_id" "mixed" "$tape" "$FREQ_LABEL" "$VID_LABEL"
    fi
    order_idx=$((order_idx + 1))
done

# ── Execute controls ──────────────────────────────────────────
echo "=== CONTROLS ==="
for ctrl in "${CONTROL_LIST[@]}"; do
    cmode=$(echo "$ctrl" | grep -oP 'CTRL=\K[^,]+')
    run_id=$(echo "$ctrl" | grep -oP 'ID=\K.*')

    echo "$order_idx,$run_id,$CTRL_TAPE,,$cmode" >> "$ORDER_FILE"
    run_control "$run_id" "$cmode"
    order_idx=$((order_idx + 1))
done

# ── Analyze all completed runs ────────────────────────────────
echo "=== ANALYZING COMPLETED RUNS ==="
ANALYSIS_FAILED=0
for run_id in "${COMPLETED[@]}"; do
    dir="$OUTPUT_DIR/$run_id"
    if [ -f "$dir/raw_cycles.csv" ]; then
        echo "Analyzing $run_id ..."
        if ! python3 analyze_phase5_8.py --input-dir "$dir" --window-size "$WINDOW_SIZE"; then
            echo "  ANALYSIS FAIL: $run_id"
            ANALYSIS_FAILED=1
        fi
    fi
done
echo "Analysis complete."
echo ""

# ── Run cross-run aggregator if available ─────────────────────
if [ -f "aggregate_phase5_8.py" ]; then
    echo "=== CROSS-RUN AGGREGATION ==="
    if ! python3 aggregate_phase5_8.py --output-dir "$OUTPUT_DIR"; then
        echo "  AGGREGATION FAIL"
        ANALYSIS_FAILED=1
    fi
    echo ""
fi

# ── Summary ───────────────────────────────────────────────────
echo "============================================================"
echo "EXP50 PHASE 5.8R ORCHESTRATION COMPLETE"
echo "============================================================"
echo "Completed: ${#COMPLETED[@]} runs"
for r in "${COMPLETED[@]}"; do echo "  PASS: $r"; done
echo ""
echo "Failed:   ${#FAILED[@]} runs"
for r in "${FAILED[@]}"; do echo "  FAIL: $r (${RUN_STATUS[$r]}, exit=${RUN_EXIT_CODE[$r]}, sig=${RUN_SIGNAL[$r]})"; done
echo ""
echo "Skipped:  ${#SKIPPED[@]} runs"
echo ""

# Aggregate trial counts
total_trials=0
total_restore_failures=0
for run_id in "${COMPLETED[@]}"; do
    dir="$OUTPUT_DIR/$run_id"
    if [ -f "$dir/raw_cycles.csv" ]; then
        trials=$(($(wc -l < "$dir/raw_cycles.csv") - 1))
        total_trials=$((total_trials + trials))
    fi
done

echo "Total valid trials: $total_trials"
echo "Output: $OUTPUT_DIR/"
echo "Condition order: $ORDER_FILE"
echo "============================================================"
echo ""
echo "Next: run aggregate_phase5_8.py for cross-run verdict"
echo "============================================================"

if [ ${#FAILED[@]} -ne 0 ] || [ "$ANALYSIS_FAILED" -ne 0 ]; then
    exit 1
fi
