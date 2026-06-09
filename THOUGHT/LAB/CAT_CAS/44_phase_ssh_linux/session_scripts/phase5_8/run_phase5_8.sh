#!/bin/bash
# run_phase5_8.sh — Orchestrate Phase 5.8 bare-metal boundary probe
#
# Environment variables:
#   MEAS_CORE=3          Measurement core
#   ITERATIONS=100000    Trials per run
#   WINDOW_SIZE=256      Window size for analyzer
#   TAPE_SIZES="256 4096" Tape sizes to sweep
#   WORKERS="0 1 2 4 6 8 10 12"  Worker core list
#   FREQ_LABEL="nominal" Frequency label
#   VID_LABEL="unknown"   VID label
#   OUTPUT_DIR="./output" Output directory
#
# Usage:
#   ./run_phase5_8.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────
MEAS_CORE="${MEAS_CORE:-3}"
ITERATIONS="${ITERATIONS:-100000}"
WINDOW_SIZE="${WINDOW_SIZE:-256}"
TAPE_SIZES="${TAPE_SIZES:-256 4096}"
# Valid Phenom II X6 cores: 0,1,2,4 (skip 3=measurement, 5=phase master)
# Total 6 cores; 3 and 5 are isolated. Max usable workers: 4.
WORKERS="${WORKERS:-0,1,2,4}"
FREQ_LABEL="${FREQ_LABEL:-nominal}"
VID_LABEL="${VID_LABEL:-unknown}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

BIN="./phase5_8_boundary_rdtsc"

# ── Build ─────────────────────────────────────────────────────
echo "=== BUILD ==="
make clean 2>/dev/null || true
make all
echo "Build complete."
echo ""

# ── Create output directory ───────────────────────────────────
mkdir -p "$OUTPUT_DIR"

# ── Helper: run one configuration ─────────────────────────────
run_config() {
    local run_id="$1"
    local worker_mode="$2"
    local tape_size="$3"
    local freq_label="$4"
    local vid_label="$5"
    local subdir="$6"
    # Each run gets its own subdirectory (run_id) to prevent overwrite.
    # The parent category (subdir) groups runs for analysis.
    local run_dir="$OUTPUT_DIR/$subdir/$run_id"

    mkdir -p "$run_dir"

    local worker_args=""
    if [ "$worker_mode" != "none" ]; then
        worker_args="--workers $WORKERS --worker-mode $worker_mode --placement offcore"
    fi

    echo "--- $run_id ---"
    echo "  tape=$tape_size  workers=$worker_mode  freq=$freq_label  vid=$vid_label"

    $BIN \
        --core "$MEAS_CORE" \
        $worker_args \
        --iterations "$ITERATIONS" \
        --window-size "$WINDOW_SIZE" \
        --tape-size "$tape_size" \
        --freq-label "$freq_label" \
        --vid-label "$vid_label" \
        --run-id "$run_id" \
        --output-dir "$run_dir"

    echo "  $run_id complete."
    echo ""
}

# ── Baseline: no workers, nominal ────────────────────────────
echo "=== BASELINE (no workers) ==="
for TAPE in $TAPE_SIZES; do
    run_config "BASELINE_T${TAPE}_NOMINAL" "none" "$TAPE" "$FREQ_LABEL" "$VID_LABEL" "baseline"
done

# ── Cache pressure ────────────────────────────────────────────
echo "=== CACHE PRESSURE ==="
for TAPE in $TAPE_SIZES; do
    run_config "CACHE_T${TAPE}_NOMINAL" "cache" "$TAPE" "$FREQ_LABEL" "$VID_LABEL" "cache_pressure"
done

# ── Mixed pressure ────────────────────────────────────────────
echo "=== MIXED PRESSURE ==="
for TAPE in $TAPE_SIZES; do
    run_config "MIXED_T${TAPE}_NOMINAL" "mixed" "$TAPE" "$FREQ_LABEL" "$VID_LABEL" "mixed_pressure"
done

# ── Frequency-labeled trials (if provided) ─────────────────────
if [ -n "${FREQ_SWEEP:-}" ]; then
    echo "=== FREQUENCY SWEEP ==="
    for FREQ in $FREQ_SWEEP; do
        for TAPE in $TAPE_SIZES; do
            run_config "FREQ_${FREQ}_T${TAPE}" "cache" "$TAPE" "$FREQ" "$VID_LABEL" "freq_sweep"
        done
    done
fi

# ── Controls ──────────────────────────────────────────────────
echo "=== CONTROLS ==="
$BIN --core "$MEAS_CORE" --iterations 10000 --tape-size 256 \
    --run-id "CONTROL_EMPTY_T256" --output-dir "$OUTPUT_DIR/controls/CONTROL_EMPTY_T256" \
    --freq-label "$FREQ_LABEL" --vid-label "$VID_LABEL" \
    --control empty
echo "  Control A (empty timing): done."

$BIN --core "$MEAS_CORE" --iterations 10000 --tape-size 256 \
    --run-id "CONTROL_NOP_T256" --output-dir "$OUTPUT_DIR/controls/CONTROL_NOP_T256" \
    --freq-label "$FREQ_LABEL" --vid-label "$VID_LABEL" \
    --control nop
echo "  Control B (NOP loop): done."

$BIN --core "$MEAS_CORE" --iterations 10000 --tape-size 256 \
    --run-id "CONTROL_IRREVERSIBLE_T256" --output-dir "$OUTPUT_DIR/controls/CONTROL_IRREVERSIBLE_T256" \
    --freq-label "$FREQ_LABEL" --vid-label "$VID_LABEL" \
    --control irreversible
echo "  Control C (irreversible): done."

$BIN --core "$MEAS_CORE" --iterations 10000 --tape-size 256 \
    --run-id "CONTROL_READONLY_T256" --output-dir "$OUTPUT_DIR/controls/CONTROL_READONLY_T256" \
    --freq-label "$FREQ_LABEL" --vid-label "$VID_LABEL" \
    --control readonly
echo "  Control D (read-only): done."

# ── Analyze ───────────────────────────────────────────────────
echo "=== ANALYZE ==="
# Analyze each baseline run individually (each in its own subdir)
for run_dir in "$OUTPUT_DIR"/baseline/*/; do
    if [ -f "$run_dir/raw_cycles.csv" ]; then
        echo "Analyzing $run_dir ..."
        python3 analyze_phase5_8.py --input-dir "$run_dir" --window-size "$WINDOW_SIZE" || true
    fi
done
echo "Baseline analysis complete."

# ── Final summary ─────────────────────────────────────────────
echo ""
echo "============================================================"
echo "EXP44 PHASE 5.8 ORCHESTRATION COMPLETE"
echo "============================================================"
echo "Output: $OUTPUT_DIR/"
echo ""
echo "To view results per run:"
echo "  ls $OUTPUT_DIR/baseline/*/"
echo "  cat $OUTPUT_DIR/baseline/BASELINE_T4096_NOMINAL/TELEMETRY_PHASE5_8.txt"
echo "  cat $OUTPUT_DIR/baseline/BASELINE_T4096_NOMINAL/verdict_gate_audit.csv"
echo "============================================================"
echo ""
echo "Next action: review telemetry, check gates, assign verdict."
echo "If PASS: proceed to Phase 5.9 (Analog Silicon Boundary Entry)."
echo "If PARTIAL: proceed to Phase 5.8R (Artifact Removal)."
echo "If FAIL: document in PHASE5_8_FAILURE_ANALYSIS.md."
