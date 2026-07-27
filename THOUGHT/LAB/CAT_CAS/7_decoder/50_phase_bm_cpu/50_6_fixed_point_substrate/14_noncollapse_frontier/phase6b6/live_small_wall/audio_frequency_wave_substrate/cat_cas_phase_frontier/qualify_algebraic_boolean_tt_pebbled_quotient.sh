#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_algebraic_boolean_tt_pebbled_quotient.sh EVIDENCE_DIR" >&2
    exit 2
fi

frontier_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
evidence_dir=$1
mkdir -p "$evidence_dir"

phase_source="$frontier_dir/algebraic_boolean_tt_pebbled_quotient_phase.c"
retain_source="$frontier_dir/algebraic_boolean_tt_suffix_quotient_phase.c"
reference_source="$frontier_dir/algebraic_boolean_tt_suffix_quotient_reference.c"

for tool in gcc jq sha256sum cmp rg; do
    command -v "$tool" >/dev/null
done

common=(
    -std=c11
    -O2
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wconversion
    -Wshadow
)

gcc "${common[@]}" "$phase_source" -lm \
    -o "$evidence_dir/phase"
gcc "${common[@]}" "$retain_source" -lm \
    -o "$evidence_dir/retain_all"
gcc "${common[@]}" "$reference_source" \
    -o "$evidence_dir/reference"
gcc "${common[@]}" -fanalyzer "$phase_source" -lm \
    -o "$evidence_dir/phase_analyzer"
gcc "${common[@]}" -fstack-usage "$phase_source" -lm \
    -o "$evidence_dir/phase_stack"
gcc -std=c11 -O1 -g \
    -Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow \
    -fsanitize=address,undefined \
    -fno-omit-frame-pointer -fno-sanitize-recover=all \
    "$phase_source" -lm \
    -o "$evidence_dir/phase_sanitized"

cases=("4:4" "5:5" "8:8" "12:8" "16:8")
for case_spec in "${cases[@]}"; do
    width=${case_spec%%:*}
    depth=${case_spec#*:}
    label="w${width}_d${depth}"
    "$evidence_dir/phase" "$width" "$depth" \
        >"$evidence_dir/phase_$label.json"
    "$evidence_dir/phase" "$width" "$depth" \
        >"$evidence_dir/phase_${label}_replay.json"
    cmp \
        "$evidence_dir/phase_$label.json" \
        "$evidence_dir/phase_${label}_replay.json"
    "$evidence_dir/retain_all" "$width" "$depth" \
        >"$evidence_dir/retain_$label.json"
    "$evidence_dir/reference" "$width" "$depth" \
        >"$evidence_dir/reference_$label.json"
    jq -e \
        --slurpfile retain "$evidence_dir/retain_$label.json" \
        --slurpfile reference "$evidence_dir/reference_$label.json" '
        .claim
          == "TOPOLOGY_DERIVED_REVERSIBLE_BOOLEAN_TT_QUOTIENT_STAGE_PEBBLING_REDUCES_RETAINED_PHASE_HISTORY"
        and .primary_boundary_fnv1a64
          == $retain[0].primary_boundary_fnv1a64
        and .reuse_boundary_fnv1a64
          == $retain[0].reuse_boundary_fnv1a64
        and .primary_boundary_fnv1a64
          == $reference[0].and_boundary_fnv1a64
        and .reuse_boundary_fnv1a64
          == $reference[0].or_boundary_fnv1a64
        and .primary_boundary_ones
          == $reference[0].and_boundary_ones
        and .reuse_boundary_ones
          == $reference[0].or_boundary_ones
        and .boundary_cells == $reference[0].final_quotient_cells
        and .retain_all_carrier_cells == $retain[0].carrier_cells
        and .pebbled_carrier_cells < .retain_all_carrier_cells
        and .retain_all_carrier_bytes
          == (32 * .retain_all_carrier_cells)
        and .pebbled_carrier_bytes
          == (32 * .pebbled_carrier_cells)
        and .comparison_snapshot_bytes == .pebbled_carrier_bytes
        and .projection_bytes == .boundary_cells
        and .carrier_cell_reduction
          == (.retain_all_carrier_cells - .pebbled_carrier_cells)
        and .carrier_reduction_ratio > 1
        and .applicability_gate_reduces_carrier
        and .forward_schedule_moves > 0
        and .weighted_pebble_move_cells_per_transaction > 0
        and .stage_additions_per_transaction
          == .stage_removals_per_transaction
        and .reconstruction_additions_per_transaction > 0
        and .reconstruction_cells_per_transaction > 0
        and .logical_phase_ands_per_transaction
          == .quotient_member_terms_per_transaction
        and .phase_cell_updates_per_transaction
          > $retain[0].phase_cell_updates_per_transaction
        and .final_decodes_per_transaction == .boundary_cells
        and .decoded_intermediate_cells == 0
        and .serialized_intermediate_cells == 0
        and .materialized_raw_product_cells == 0
        and .schedule_derived_from_public_depth
        and .exact_slot_generation_custody
        and .actual_inverse_restoration
        and .actual_restored_carrier_reuse
        and .same_carrier_transactions == 18
        and .maximum_repeated_restoration_error <= 2e-12
        and .maximum_slot_clean_error <= 2e-12
        and .wrong_inverse_detected
        and .missing_inverse_detected
        and .reordered_noncommuting_inverse_detected
        and .dirty_slot_injection_detected
        and .generation_tag_tamper_detected
        and .schedule_hash_tamper_rejected
        and .snapshot_loaded_separately
        and .compact_public_recurrence_still_exists
        and (.computational_advantage | not)
        and (.small_wall_crossed | not)
        and (.terminal | not)
    ' "$evidence_dir/phase_$label.json" >/dev/null
done

jq -e '
    .pebbled_carrier_cells == 448
    and .retain_all_carrier_cells == 528
    and .phase_cell_updates_per_transaction == 1216
    and .slot_capacities == [140,120]
' "$evidence_dir/phase_w4_d4.json" >/dev/null
jq -e '
    .pebbled_carrier_cells == 1016
    and .retain_all_carrier_cells == 1132
    and .phase_cell_updates_per_transaction == 2496
    and .slot_capacities == [240,184,264]
' "$evidence_dir/phase_w5_d5.json" >/dev/null
jq -e '
    .pebbled_carrier_cells == 3640
    and .retain_all_carrier_cells == 5608
    and .phase_cell_updates_per_transaction == 16048
    and .slot_capacities == [956,920,696]
' "$evidence_dir/phase_w8_d8.json" >/dev/null
jq -e '
    .pebbled_carrier_cells == 7896
    and .retain_all_carrier_cells == 11448
    and .phase_cell_updates_per_transaction == 31472
    and .slot_capacities == [2252,1944,1272]
' "$evidence_dir/phase_w12_d8.json" >/dev/null
jq -e '
    .pebbled_carrier_cells == 12152
    and .retain_all_carrier_cells == 17288
    and .phase_cell_updates_per_transaction == 46896
    and .slot_capacities == [3548,2968,1848]
    and .pebble_slots == 3
    and .forward_schedule_moves == 13
    and .active_projection_mask == 104
    and .active_projection_cells == 8364
    and .weighted_pebble_move_cells_per_transaction == 39320
' "$evidence_dir/phase_w16_d8.json" >/dev/null

for case_spec in "4:4" "16:8"; do
    width=${case_spec%%:*}
    depth=${case_spec#*:}
    label="w${width}_d${depth}"
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        "$evidence_dir/phase_sanitized" "$width" "$depth" \
        >"$evidence_dir/sanitized_$label.json" \
        2>"$evidence_dir/sanitized_$label.stderr"
    [[ ! -s "$evidence_dir/sanitized_$label.stderr" ]]
    cmp \
        <(jq -S . "$evidence_dir/phase_$label.json") \
        <(jq -S . "$evidence_dir/sanitized_$label.json")
done

set +e
"$evidence_dir/phase" --project-intermediate 16 8 \
    >"$evidence_dir/intermediate.stdout" \
    2>"$evidence_dir/intermediate.stderr"
intermediate_rc=$?
set -e
[[ "$intermediate_rc" -eq 2 ]]
[[ ! -s "$evidence_dir/intermediate.stdout" ]]
rg -q '^only the final pebbled quotient boundary may be projected$' \
    "$evidence_dir/intermediate.stderr"

if rg -n \
    'truth_table|witness|candidate_set|expected_(hash|boundary)' \
    "$phase_source"
then
    echo "pebbled phase path contains forbidden extensional state" >&2
    exit 1
fi
rg -q 'qtp_generate_recursive' "$phase_source"
rg -q 'qtp_apply_move' "$phase_source"
rg -q 'qtp_slot_error' "$phase_source"
rg -q 'qtp_plan_hash' "$phase_source"

stack_file=$(
    find "$evidence_dir" -maxdepth 1 \
        -name 'phase_stack-*.su' -print -quit
)
[[ -n "$stack_file" ]]
maximum_stack_frame=$(
    awk -F '\t' '$2 > maximum {maximum=$2} END {print maximum+0}' \
        "$stack_file"
)
[[ "$maximum_stack_frame" =~ ^[0-9]+$ ]]

jq -s 'sort_by(.width,.depth)' \
    "$evidence_dir"/phase_w*_d[0-9].json \
    >"$evidence_dir/phase_results.json"
jq -s 'sort_by(.width,.depth)' \
    "$evidence_dir"/retain_w*_d[0-9].json \
    >"$evidence_dir/retain_results.json"
jq -s 'sort_by(.width,.depth)' \
    "$evidence_dir"/reference_w*_d[0-9].json \
    >"$evidence_dir/reference_results.json"

jq -n \
    --arg claim \
        "TOPOLOGY_DERIVED_REVERSIBLE_BOOLEAN_TT_QUOTIENT_STAGE_PEBBLING_REDUCES_RETAINED_PHASE_HISTORY" \
    --arg ceiling \
        "BOUNDED_LINUX_SOFTWARE_WIDTHS4_5_8_12_16_DEPTHS4_5_8_HOMOGENEOUS_NEIGHBOR_AND_OR_PHASE_PEBBLING_REFERENCE_ONLY" \
    --argjson maximum_stack_frame "$maximum_stack_frame" \
    --slurpfile phase "$evidence_dir/phase_results.json" \
    --slurpfile retain "$evidence_dir/retain_results.json" \
    --slurpfile reference "$evidence_dir/reference_results.json" '
    {
      result: "PASS",
      claim: $claim,
      claim_ceiling: $ceiling,
      cases: $phase[0],
      retain_all_comparison: $retain[0],
      independent_raw_reference: $reference[0],
      maximum_compiler_stack_frame_bytes: $maximum_stack_frame,
      width16_depth8: {
        retained_carrier_cells: $phase[0][4].retain_all_carrier_cells,
        pebbled_carrier_cells: $phase[0][4].pebbled_carrier_cells,
        carrier_reduction_cells: $phase[0][4].carrier_cell_reduction,
        carrier_reduction_percent:
          (
            100
            * $phase[0][4].carrier_cell_reduction
            / $phase[0][4].retain_all_carrier_cells
          ),
        retained_phase_updates:
          $retain[0][4].phase_cell_updates_per_transaction,
        pebbled_phase_updates:
          $phase[0][4].phase_cell_updates_per_transaction,
        phase_update_multiplier:
          (
            $phase[0][4].phase_cell_updates_per_transaction
            / $retain[0][4].phase_cell_updates_per_transaction
          )
      },
      controls: {
        exact_retain_all_parity: true,
        exact_raw_reference_parity: true,
        deterministic_replay: true,
        wrong_inverse_detected:
          (all($phase[0][]; .wrong_inverse_detected)),
        missing_inverse_detected: true,
        reordered_noncommuting_inverse_detected: true,
        schedule_hash_tamper_rejected: true,
        dirty_slot_injection_detected:
          (all($phase[0][]; .dirty_slot_injection_detected)),
        generation_tag_tamper_detected:
          (all($phase[0][]; .generation_tag_tamper_detected)),
        intermediate_projection_rejected: true,
        snapshot_separated_from_actual_inverse: true,
        actual_restored_carrier_reuse: true,
        analyzer: "PASS",
        asan_ubsan: "PASS"
      },
      primary_obstruction_removed:
        "RETAIN_ALL_PHASE_HISTORY_DEPTH_FACTOR_REDUCED_BY_PUBLIC_REVERSIBLE_PEBBLING",
      remaining_primary_obstruction:
        "PUBLIC_BOOLEAN_THRESHOLD_RECURRENCE_HAS_NO_DISTINCT_PHASE_RESOURCE",
      computational_advantage: false,
      small_wall_crossed: false,
      physical_execution: false,
      unlimited_catalytic_computation: false,
      terminal: false
    }
' >"$evidence_dir/summary.json"

jq -e '
    .result == "PASS"
    and (.cases | length) == 5
    and .width16_depth8.retained_carrier_cells == 17288
    and .width16_depth8.pebbled_carrier_cells == 12152
    and .width16_depth8.carrier_reduction_percent > 29
    and .width16_depth8.phase_update_multiplier > 1.35
    and .controls.exact_retain_all_parity
    and .controls.wrong_inverse_detected
    and .controls.dirty_slot_injection_detected
    and .controls.generation_tag_tamper_detected
    and .controls.actual_restored_carrier_reuse
    and (.computational_advantage | not)
    and (.small_wall_crossed | not)
    and (.terminal | not)
' "$evidence_dir/summary.json" >/dev/null

sha256sum \
    "$phase_source" \
    "$retain_source" \
    "$reference_source" \
    "$evidence_dir/phase" \
    "$evidence_dir/retain_all" \
    "$evidence_dir/reference" \
    "$evidence_dir/phase_results.json" \
    "$evidence_dir/summary.json" \
    >"$evidence_dir/SHA256SUMS"

printf '%s\n' \
    "TOPOLOGY_DERIVED_REVERSIBLE_BOOLEAN_TT_QUOTIENT_STAGE_PEBBLING_PASS"
