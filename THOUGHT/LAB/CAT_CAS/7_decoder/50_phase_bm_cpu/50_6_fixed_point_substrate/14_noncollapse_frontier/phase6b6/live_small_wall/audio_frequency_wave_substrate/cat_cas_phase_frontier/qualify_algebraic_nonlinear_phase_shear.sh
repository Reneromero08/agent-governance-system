#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_algebraic_nonlinear_phase_shear.sh EVIDENCE_DIR" >&2
    exit 2
fi

frontier_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
evidence_dir=$1
mkdir -p "$evidence_dir"

phase_source="$frontier_dir/algebraic_nonlinear_phase_shear.c"
reference_source="$frontier_dir/algebraic_nonlinear_phase_shear_reference.c"

for tool in gcc jq sha256sum cmp rg nice; do
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

nice -n 10 gcc "${common[@]}" "$phase_source" -lm \
    -o "$evidence_dir/phase"
nice -n 10 gcc "${common[@]}" "$reference_source" -lm \
    -o "$evidence_dir/reference"
nice -n 10 gcc "${common[@]}" -fanalyzer "$phase_source" -lm \
    -o "$evidence_dir/phase_analyzer"
nice -n 10 gcc "${common[@]}" -fstack-usage "$phase_source" -lm \
    -o "$evidence_dir/phase_stack"
nice -n 10 gcc -std=c11 -O1 -g \
    -Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow \
    -fsanitize=address,undefined \
    -fno-omit-frame-pointer -fno-sanitize-recover=all \
    "$phase_source" -lm \
    -o "$evidence_dir/phase_sanitized"

depths=(3 32 128 512 2048 4096)
for depth in "${depths[@]}"; do
    nice -n 10 "$evidence_dir/phase" "$depth" \
        >"$evidence_dir/phase_d${depth}.json"
    nice -n 10 "$evidence_dir/phase" "$depth" \
        >"$evidence_dir/phase_d${depth}_replay.json"
    cmp \
        "$evidence_dir/phase_d${depth}.json" \
        "$evidence_dir/phase_d${depth}_replay.json"
    nice -n 10 "$evidence_dir/reference" "$depth" \
        >"$evidence_dir/reference_d${depth}.json"
    jq -e \
        --slurpfile reference "$evidence_dir/reference_d${depth}.json" '
        .claim
          == "BOUNDED_NONLINEAR_UNIT_PHASE_TORUS_SHEAR_COMPOSITION_WITH_ACTUAL_RESTORATION_AND_INTERFERENCE_PROJECTION"
        and .primary_boundary_fnv1a64
          == $reference[0].primary_boundary_fnv1a64
        and .reuse_boundary_fnv1a64
          == $reference[0].reuse_boundary_fnv1a64
        and (
          .primary_interference_probability
          - $reference[0].primary_interference_probability
          | fabs
        ) <= 2e-10
        and .phase_rank == 2
        and .resident_phase_cells == 2
        and .boundary_phase_cells == 2
        and .carrier_cells == 4
        and .live_carrier_bytes == 128
        and .verification_snapshot_bytes == 128
        and .accepted_path_peak_carrier_bytes == 256
        and .projection_bytes == 48
        and .declared_local_shear_temporary_bytes == 80
        and .best_matched_classical_state_bytes
          == $reference[0].classical_state_bytes
        and .compiled_shear_list_bytes == 0
        and .primary_forward_shears == .depth
        and .primary_inverse_shears == .depth
        and .primary_phase_source_reads == (2 * .depth + 4)
        and .primary_native_phase_updates == (2 * .depth + 8)
        and .maximum_unit_modulus_error <= 2e-12
        and .maximum_repeated_restoration_error <= 2e-12
        and .wrong_boundary_restoration_residual >= 1e-3
        and .missing_inverse_restoration_residual >= 1e-3
        and .wrong_shear_restoration_residual >= 1e-3
        and .reordered_inverse_restoration_residual >= 1e-3
        and .snapshot_reload_restoration_residual <= 2e-12
        and .final_phase_decodes == 2
        and .projected_intermediate_phases == 0
        and .serialized_intermediate_phases == 0
        and .materialized_phase_paths == 0
        and .unit_modulus_carrier_invariant
        and .relative_phase_primitive_consumed
        and .multiply_cell_only_native_updates
        and .nonlinear_phase_coupling
        and .noncommuting_phase_shears
        and .fixed_carrier_across_depth
        and .actual_inverse_restoration
        and .actual_restored_carrier_reuse
        and .same_carrier_transactions == 18
        and .coupling_disabled_control_detected
        and .baseline_neutralization_control_detected
        and .null_carrier_execution_rejected
        and .wrong_boundary_inverse_detected
        and .wrong_shear_inverse_detected
        and .missing_shear_inverse_detected
        and .reordered_noncommuting_inverse_detected
        and .snapshot_loaded_separately
        and (.boolean_root_locked | not)
        and (.gf2_affine | not)
        and (.linear_u2_recurrence | not)
        and .compact_classical_two_angle_recurrence_exists
        and (.genuinely_distinct_phase_resource | not)
        and (.machine_enforced_hidden_intermediate | not)
        and (.computational_advantage | not)
        and (.small_wall_crossed | not)
        and (.terminal | not)
    ' "$evidence_dir/phase_d${depth}.json" >/dev/null
done

for depth in 3 4096; do
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        nice -n 10 "$evidence_dir/phase_sanitized" "$depth" \
        >"$evidence_dir/sanitized_d${depth}.json" \
        2>"$evidence_dir/sanitized_d${depth}.stderr"
    [[ ! -s "$evidence_dir/sanitized_d${depth}.stderr" ]]
    cmp \
        <(jq -S . "$evidence_dir/phase_d${depth}.json") \
        <(jq -S . "$evidence_dir/sanitized_d${depth}.json")
done

for denied in project-intermediate null-carrier; do
    set +e
    "$evidence_dir/phase" "--${denied}" \
        >"$evidence_dir/${denied}.stdout" \
        2>"$evidence_dir/${denied}.stderr"
    denied_rc=$?
    set -e
    [[ "$denied_rc" -eq 2 ]]
    [[ ! -s "$evidence_dir/${denied}.stdout" ]]
done
rg -q '^only the final phase-interference boundary may be projected$' \
    "$evidence_dir/project-intermediate.stderr"
rg -q '^invalid nonlinear phase-shear transaction$' \
    "$evidence_dir/null-carrier.stderr"

if rg -n \
    'truth_table|witness_list|candidate_set|assignment_expansion|expected_(hash|boundary)' \
    "$phase_source"
then
    echo "phase-shear path contains forbidden extensional state" >&2
    exit 1
fi
rg -q 'relative\(' "$phase_source"
rg -q 'multiply_cell\(' "$phase_source"
rg -q 'ps_apply_shear' "$phase_source"
if rg -q 'carrier->working\\[[^]]+\\][[:space:]]*[-+*/]?=' \
    "$phase_source"
then
    echo "phase-shear path bypasses native phase update primitive" >&2
    exit 1
fi

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

jq -s 'sort_by(.depth)' \
    "$evidence_dir/phase_d3.json" \
    "$evidence_dir/phase_d32.json" \
    "$evidence_dir/phase_d128.json" \
    "$evidence_dir/phase_d512.json" \
    "$evidence_dir/phase_d2048.json" \
    "$evidence_dir/phase_d4096.json" \
    >"$evidence_dir/phase_results.json"
jq -s 'sort_by(.depth)' \
    "$evidence_dir"/reference_d[0-9]*.json \
    >"$evidence_dir/reference_results.json"

jq -n \
    --arg claim \
        "BOUNDED_NONLINEAR_UNIT_PHASE_TORUS_SHEAR_COMPOSITION_WITH_ACTUAL_RESTORATION_AND_INTERFERENCE_PROJECTION" \
    --arg ceiling \
        "BOUNDED_LINUX_SOFTWARE_NONLINEAR_TORUS_SHEAR_DEPTHS3_32_128_512_2048_4096_DOUBLE_COMPLEX_PHASE_REFERENCE_ONLY" \
    --argjson maximum_stack_frame "$maximum_stack_frame" \
    --slurpfile phase "$evidence_dir/phase_results.json" \
    --slurpfile reference "$evidence_dir/reference_results.json" '
    {
      result: "PASS",
      claim: $claim,
      claim_ceiling: $ceiling,
      cases: $phase[0],
      direct_compact_reference: $reference[0],
      maximum_compiler_stack_frame_bytes: $maximum_stack_frame,
      resource_law: {
        phase_carrier_cells: 4,
        live_carrier_bytes: 128,
        verification_snapshot_bytes: 128,
        accepted_path_peak_carrier_bytes: 256,
        projection_bytes: 48,
        declared_local_shear_temporary_bytes: 80,
        direct_classical_state_bytes: 16,
        compiled_shear_list_bytes: 0,
        phase_time: "O_DEPTH",
        direct_classical_time: "O_DEPTH"
      },
      controls: {
        exact_quantized_reference_parity: true,
        deterministic_replay: true,
        unit_modulus_invariant: true,
        native_update_primitive_only: true,
        coupling_disabled_changes_boundary: true,
        baseline_neutralization_changes_boundary: true,
        null_carrier_execution_rejected: true,
        intermediate_projection_rejected: true,
        wrong_boundary_inverse_detected: true,
        wrong_shear_inverse_detected: true,
        missing_shear_inverse_detected: true,
        reordered_noncommuting_inverse_detected: true,
        snapshot_separated_from_actual_inverse: true,
        actual_restored_carrier_reuse: true,
        analyzer: "PASS",
        asan_ubsan: "PASS"
      },
      phase_owned_advance:
        "NONLINEAR_NONCOMMUTING_UNIT_PHASE_TORUS_SHEAR_COUPLING",
      remaining_primary_obstruction:
        "EQUIVALENT_FIXED_TWO_ANGLE_CLASSICAL_RECURRENCE",
      genuinely_distinct_phase_resource: false,
      computational_advantage: false,
      small_wall_crossed: false,
      physical_execution: false,
      unlimited_catalytic_computation: false,
      terminal: false
    }
  ' >"$evidence_dir/summary.json"

jq -e '
    .result == "PASS"
    and (.cases | length) == 6
    and all(.cases[];
      .carrier_cells == 4
      and .unit_modulus_carrier_invariant
      and .multiply_cell_only_native_updates
      and .nonlinear_phase_coupling
      and .actual_inverse_restoration
      and .actual_restored_carrier_reuse
    )
    and .controls.exact_quantized_reference_parity
    and .controls.coupling_disabled_changes_boundary
    and .controls.baseline_neutralization_changes_boundary
    and (.genuinely_distinct_phase_resource | not)
    and (.computational_advantage | not)
    and (.small_wall_crossed | not)
    and (.terminal | not)
' "$evidence_dir/summary.json" >/dev/null

sha256sum \
    "$phase_source" \
    "$reference_source" \
    "$evidence_dir/phase" \
    "$evidence_dir/reference" \
    "$evidence_dir/phase_results.json" \
    "$evidence_dir/reference_results.json" \
    "$evidence_dir/summary.json" \
    >"$evidence_dir/SHA256SUMS"

printf '%s\n' \
    "BOUNDED_NONLINEAR_UNIT_PHASE_TORUS_SHEAR_PASS"
