#!/bin/bash
# run_phase5_9c.sh — Phase 5.9C CONTROLLED EDGE APPROACH
# 6-push boundary escalation protocol.
# PUSH 1: Effective frequency audit (verify P-state → actual timing)
# PUSH 2: All-core P-state coordination
# PUSH 3: Combined stress ladder (10+ ordered points, 3 repeats)
# PUSH 4: Long-duration edge search (250K+ trials)
# PUSH 5: Restoration flicker search
# PUSH 6: Artifact-separated geometry

set +e +u -o pipefail

MEAS_CORE="${MEAS_CORE:-3}"
WORKER_CORES="${WORKER_CORES:-0,1,2,4}"
PHASE_CORE="${PHASE_CORE:-5}"
FIXED_TAPE="${FIXED_TAPE:-2048}"
ITERATIONS="${ITERATIONS:-50000}"
LONG_ITERATIONS="${LONG_ITERATIONS:-250000}"
TEMP_LIMIT="${TEMP_LIMIT:-65}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_5_9c}"
REPEATS="${REPEATS:-3}"
BIN="./phase5_9_stress_ladder"
MSR_BASE="0xC0010062"

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
    local worker_mode="$5" freq_label="$6" iterations="${7:-$ITERATIONS}"
    local dir="$OUTPUT_DIR/$run_id"
    mkdir -p "$dir"

    local worker_args=""
    [ "$worker_mode" != "none" ] && worker_args="--workers $WORKER_CORES --worker-mode $worker_mode"

    local exit_code=0
    $BIN --core "$MEAS_CORE" $worker_args \
        --iterations "$iterations" --tape-size "$FIXED_TAPE" \
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
echo "=== PHASE 5.9C: CONTROLLED EDGE APPROACH ==="
echo "Fixed tape: $FIXED_TAPE"
echo "Measurement core: $MEAS_CORE  Worker cores: $WORKER_CORES"
echo "Iterations/run: $ITERATIONS  Long: $LONG_ITERATIONS"
echo ""

# ═══════════════════════════════════════════════════════════════
# PUSH 1: EFFECTIVE FREQUENCY AUDIT
# ═══════════════════════════════════════════════════════════════
echo "=== PUSH 1: EFFECTIVE FREQUENCY AUDIT ==="
AUDIT_DIR="$OUTPUT_DIR/audit"
mkdir -p "$AUDIT_DIR"

FREQ_AUDIT="$AUDIT_DIR/frequency_effective_audit.csv"
echo "pstate,requested_label,rdmsr_before,rdmsr_after,cpuinfo_mhz_before,cpuinfo_mhz_after,timing_shift,effective_change" > "$FREQ_AUDIT"

PSTATES=(0 1 2 3 4)
LABELS=("3600MHz_P0" "3200MHz_P1" "2400MHz_P2" "1600MHz_P3" "800MHz_P4")

for i in "${!PSTATES[@]}"; do
    pstate="${PSTATES[$i]}"
    label="${LABELS[$i]}"

    mhz_before=$(grep "cpu MHz" /proc/cpuinfo 2>/dev/null | head -1 | awk '{print $4}')
    rdmsr_before=$(rdmsr -p "$MEAS_CORE" "$MSR_BASE" 2>/dev/null)

    wrmsr -p "$MEAS_CORE" "$MSR_BASE" "$pstate" 2>/dev/null
    sleep 0.2

    rdmsr_after=$(rdmsr -p "$MEAS_CORE" "$MSR_BASE" 2>/dev/null)
    mhz_after=$(grep "cpu MHz" /proc/cpuinfo 2>/dev/null | head -1 | awk '{print $4}')

    # Quick timing check: run 1000-trial sanity to measure raw cycles
    timing_shift=0
    effective="UNKNOWN"
    if [ -f "$BIN" ]; then
        $BIN --core "$MEAS_CORE" --iterations 1000 --tape-size "$FIXED_TAPE" \
            --freq-label "$label" --run-id "AUDIT_${label}" \
            --output-dir "$AUDIT_DIR/audit_${label}" \
            --stress-id "AUDIT" --stress-label "AUDIT_${label}" --stress-level 0 \
            2>&1 | grep -oP 'Done.*' || true
        # Extract mean from stress_ladder.csv
        if [ -f "$AUDIT_DIR/audit_${label}/stress_ladder.csv" ]; then
            timing_shift=$(tail -1 "$AUDIT_DIR/audit_${label}/stress_ladder.csv" | cut -d',' -f11)
        fi
    fi

    # Effective change: rdmsr changed AND timing shift is nonzero and measurable
    effective="UNKNOWN"
    if [ "$rdmsr_before" != "$rdmsr_after" ]; then
        if [ -n "$timing_shift" ] && [ "$timing_shift" != "0" ] && [ "$timing_shift" != "0.000000" ]; then
            effective="YES"
        else
            effective="UNKNOWN"  # rdmsr changed but timing measurement unclear
        fi
    else
        effective="NO"  # rdmsr unchanged — P-state write had no effect
    fi
    echo "$pstate,$label,$rdmsr_before,$rdmsr_after,$mhz_before,$mhz_after,$timing_shift,$effective" >> "$FREQ_AUDIT"
    echo "  P$pstate ($label): rdmsr $rdmsr_before→$rdmsr_after  timing=$timing_shift  effective=$effective"
done
wrmsr -p "$MEAS_CORE" "$MSR_BASE" 0 2>/dev/null
echo ""

# ═══════════════════════════════════════════════════════════════
# PUSH 2: ALL-CORE P-STATE AUDIT
# ═══════════════════════════════════════════════════════════════
echo "=== PUSH 2: ALL-CORE P-STATE AUDIT ==="
ALL_CORE_AUDIT="$AUDIT_DIR/per_core_pstate_audit.csv"
echo "core,pstate_requested,rdmsr_before,rdmsr_after,control_ok" > "$ALL_CORE_AUDIT"

ALL_CORES="$MEAS_CORE $(echo $WORKER_CORES | tr ',' ' ') $PHASE_CORE"
for core in $ALL_CORES; do
    rdmsr_b=$(rdmsr -p "$core" "$MSR_BASE" 2>/dev/null)
    wrmsr -p "$core" "$MSR_BASE" 2 2>/dev/null  # P2=2400MHz
    sleep 0.1
    rdmsr_a=$(rdmsr -p "$core" "$MSR_BASE" 2>/dev/null)
    ok="NO"
    [ "$rdmsr_a" = "2" ] && ok="YES"
    echo "$core,2,$rdmsr_b,$rdmsr_a,$ok" >> "$ALL_CORE_AUDIT"
    echo "  Core $core: $rdmsr_b→$rdmsr_a  ok=$ok"
    wrmsr -p "$core" "$MSR_BASE" 0 2>/dev/null  # restore P0
done
echo ""

# ═══════════════════════════════════════════════════════════════
# PUSH 3: COMBINED STRESS LADDER (10 points, 3 repeats)
# ═══════════════════════════════════════════════════════════════
echo "=== PUSH 3: COMBINED STRESS LADDER ==="

# Define 10 stress points with increasing intensity
declare -A SP_NAME SP_MODE SP_FREQ SP_PSTATE
SP_NAME[0]="S0_BASELINE";       SP_MODE[0]="none";  SP_FREQ[0]="nominal";    SP_PSTATE[0]=""
SP_NAME[1]="S1_FREQ_LOW";       SP_MODE[1]="none";  SP_FREQ[1]="800MHz_P4";  SP_PSTATE[1]="4"
SP_NAME[2]="S2_WORKER_CACHE";   SP_MODE[2]="cache"; SP_FREQ[2]="nominal";    SP_PSTATE[2]=""
SP_NAME[3]="S3_WORKER_MIXED";   SP_MODE[3]="mixed"; SP_FREQ[3]="nominal";    SP_PSTATE[3]=""
SP_NAME[4]="S4_WORKER_THERMAL"; SP_MODE[4]="mixed"; SP_FREQ[4]="nominal";    SP_PSTATE[4]=""
SP_NAME[5]="S5_FREQ_LOW_CACHE"; SP_MODE[5]="cache"; SP_FREQ[5]="800MHz_P4";  SP_PSTATE[5]="4"
SP_NAME[6]="S6_FREQ_LOW_MIXED"; SP_MODE[6]="mixed"; SP_FREQ[6]="800MHz_P4";  SP_PSTATE[6]="4"
SP_NAME[7]="S7_FREQ_HIGH_CACHE";SP_MODE[7]="cache"; SP_FREQ[7]="3600MHz_P0"; SP_PSTATE[7]="0"
SP_NAME[8]="S8_FREQ_HIGH_MIXED";SP_MODE[8]="mixed"; SP_FREQ[8]="3600MHz_P0"; SP_PSTATE[8]="0"
SP_NAME[9]="S9_ALL_CORE_LOW_CACHE"; SP_MODE[9]="cache"; SP_FREQ[9]="800MHz_ALL"; SP_PSTATE[9]="4_ALL"

for si in $(seq 0 9); do
    name="${SP_NAME[$si]}"
    mode="${SP_MODE[$si]}"
    freq="${SP_FREQ[$si]}"
    pst="${SP_PSTATE[$si]}"

    # Set P-state if specified
    if [ -n "$pst" ]; then
        if [ "$pst" = "4_ALL" ]; then
            for core in $ALL_CORES; do
                wrmsr -p "$core" "$MSR_BASE" 4 2>/dev/null
            done
        elif [ "$pst" = "0_ALL" ]; then
            for core in $ALL_CORES; do
                wrmsr -p "$core" "$MSR_BASE" 0 2>/dev/null
            done
        else
            wrmsr -p "$MEAS_CORE" "$MSR_BASE" "$pst" 2>/dev/null
        fi
        sleep 0.2
    fi

    echo "  $name: mode=$mode freq=$freq"
    for rep in $(seq 1 $REPEATS); do
        run_id="${name}_R${rep}"
        stress_id="$name"
        stress_order=$((stress_order + 1))
        run_point "$run_id" "$stress_id" "$name" "$stress_order" \
            "$mode" "$freq" || true
    done

    # Restore P0
    [ -n "$pst" ] && wrmsr -p "$MEAS_CORE" "$MSR_BASE" 0 2>/dev/null
    [ "$pst" = "4_ALL" ] || [ "$pst" = "0_ALL" ] && for core in $ALL_CORES; do wrmsr -p "$core" "$MSR_BASE" 0 2>/dev/null; done
done
echo ""

# ═══════════════════════════════════════════════════════════════
# PUSH 4: LONG-DURATION EDGE SEARCH
# ═══════════════════════════════════════════════════════════════
echo "=== PUSH 4: LONG-DURATION EDGE SEARCH ==="
# Strongest safe stress: high frequency + cache workers
wrmsr -p "$MEAS_CORE" "$MSR_BASE" 0 2>/dev/null
sleep 0.2
for rep in $(seq 1 3); do
    run_id="LONG_DURATION_R${rep}"
    stress_id="S_LONG_DURATION"
    stress_order=$((stress_order + 1))
    echo "  LONG_DURATION_R${rep} (${LONG_ITERATIONS} trials)..."
    run_point "$run_id" "$stress_id" "LONG_DURATION_EDGE" "$stress_order" \
        "cache" "3600MHz_P0" "$LONG_ITERATIONS" || true
done
wrmsr -p "$MEAS_CORE" "$MSR_BASE" 0 2>/dev/null
echo ""

# ═══════════════════════════════════════════════════════════════
# PUSH 5: RESTORATION FLICKER SEARCH
# ═══════════════════════════════════════════════════════════════
echo "=== PUSH 5: RESTORATION FLICKER SEARCH ==="
FLICKER_CSV="$OUTPUT_DIR/restoration_flicker.csv"
echo "run_id,checksum_mismatches,transient_mismatches,logical_bits_erased,restore_failures,flicker_detected" > "$FLICKER_CSV"

for run_id in "${COMPLETED[@]}"; do
    dir="$OUTPUT_DIR/$run_id"
    restore_csv="$dir/restoration_integrity.csv"
    if [ -f "$restore_csv" ]; then
        # Count hash_match=0 rows (restoration failures)
        # CSV format: run_id,...,hash_match,restore_failures,logical_bits_erased
        # Field 12 = hash_match (0 = failure, 1 = success)
        hash_failures=$(tail -n +2 "$restore_csv" 2>/dev/null | awk -F',' '{if($12==0) print}' | wc -l)
        bits_erased=$(tail -n +2 "$restore_csv" 2>/dev/null | awk -F',' '{s+=$14}END{print s+0}')
        flicker="NO"
        [ "$hash_failures" -gt 0 ] && flicker="YES"
        echo "$run_id,$hash_failures,0,$bits_erased,0,$flicker" >> "$FLICKER_CSV"
        echo "  $run_id: hash_failures=$hash_failures bits_erased=$bits_erased flicker=$flicker"
    fi
done
echo ""

# ═══════════════════════════════════════════════════════════════
# PUSH 6: ARTIFACT-SEPARATED GEOMETRY (done by analyzer)
# ═══════════════════════════════════════════════════════════════
echo "=== PUSH 6: ARTIFACT-SEPARATED GEOMETRY (analyzer) ==="
echo "  (handled by analyze_phase5_9c.py)"
echo ""

# ═══════════════════════════════════════════════════════════════
echo "============================================================"
echo "PHASE 5.9C STRESS LADDER COMPLETE"
echo "============================================================"
echo "Completed: ${#COMPLETED[@]} runs"
echo "Failed: ${#FAILED[@]} runs"
echo "Output: $OUTPUT_DIR/"
echo ""

# ── Auto-analysis ────────────────────────────────────────────
ANALYSIS_FAILED=0
if command -v python3 &>/dev/null && [ -f "analyze_phase5_9c.py" ]; then
    echo "=== ANALYZING ==="
    for run_id in "${COMPLETED[@]}"; do
        dir="$OUTPUT_DIR/$run_id"
        if [ -f "$dir/raw_cycles.csv" ]; then
            if ! python3 analyze_phase5_9c.py --input-dir "$dir"; then
                echo "  ANALYSIS FAIL: $run_id"
                ANALYSIS_FAILED=1
            fi
        fi
    done
    echo ""
fi

if command -v python3 &>/dev/null && [ -f "aggregate_phase5_9c.py" ]; then
    echo "=== CROSS-RUN AGGREGATION ==="
    if ! python3 aggregate_phase5_9c.py --output-dir "$OUTPUT_DIR" --audit-dir "$AUDIT_DIR"; then
        echo "  AGGREGATION FAIL"
        ANALYSIS_FAILED=1
    fi
    echo ""
fi

echo "============================================================"

if [ ${#FAILED[@]} -ne 0 ] || [ "$ANALYSIS_FAILED" -ne 0 ]; then
    exit 1
fi
