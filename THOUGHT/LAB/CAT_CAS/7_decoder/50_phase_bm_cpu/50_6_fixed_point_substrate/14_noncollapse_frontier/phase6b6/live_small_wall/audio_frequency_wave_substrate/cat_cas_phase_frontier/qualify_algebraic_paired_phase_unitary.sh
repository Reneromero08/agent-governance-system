#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_algebraic_paired_phase_unitary.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$1
mkdir -p "$out"
phase_source="$here/algebraic_paired_phase_unitary.c"
reference_source="$here/algebraic_paired_phase_unitary_reference.c"
common=(
    -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror
    -Wconversion -Wshadow
)
for tool in gcc jq cmp rg sha256sum nice; do
    command -v "$tool" >/dev/null
done

gcc "${common[@]}" "$phase_source" -lm -o "$out/phase"
gcc "${common[@]}" "$reference_source" -lm -o "$out/reference"
gcc "${common[@]}" -fanalyzer "$phase_source" -lm \
    -o "$out/phase_analyzer"
gcc "${common[@]}" -fstack-usage "$phase_source" -lm \
    -o "$out/phase_stack"
gcc \
    -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror \
    -Wconversion -Wshadow -fsanitize=address,undefined \
    -fno-omit-frame-pointer -fno-sanitize-recover=all \
    "$phase_source" -lm -o "$out/phase_sanitized"

nice -n 10 "$out/phase" >"$out/phase.json"
nice -n 10 "$out/phase" >"$out/phase_replay.json"
cmp "$out/phase.json" "$out/phase_replay.json"
"$out/reference" >"$out/reference.json"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    nice -n 10 "$out/phase_sanitized" \
    >"$out/sanitized.json" 2>"$out/sanitized.stderr"
[[ ! -s "$out/sanitized.stderr" ]]
cmp <(jq -S . "$out/phase.json") <(jq -S . "$out/sanitized.json")

jq -e \
    --slurpfile reference "$out/reference.json" '
    .result == "PASS"
    and .claim
      == "BOUNDED_GENERIC_5X5_UNITARY_PAIRED_UNIT_PHASOR_COHERENT_COMPOSITION_WITH_TWO_BLOCK_CUSTODY_RESTORATION_AND_REUSE"
    and .tested_depths == [1,2,4,8,16,32]
    and .tested_depths == $reference[0].tested_depths
    and .final_boundary_hashes == $reference[0].final_boundary_hashes
    and (
      [range(0;5) as $index |
        ((.deepest_probability[$index]
          - $reference[0].deepest_probability[$index]) | abs)
      ] | max
    ) <= 1e-8
    and .matrix_side == 5
    and .phase_cells_per_complex_entry == 2
    and .hidden_phase_blocks == 2
    and .hidden_phase_cells == 100
    and .retained_inverse_matrix_cells == 0
    and .resident_phase_storage_bytes == 1600
    and .hidden_baseline_storage_bytes == 1600
    and .borrowed_verification_copy_bytes == 3200
    and .explicit_product_temporaries_bytes == 1200
    and .dense_complex_recurrence_state_bytes == 400
    and .u5_coordinate_dimension_doubles == 25
    and .u5_coordinate_lower_bound_bytes_excluding_chart_metadata == 200
    and .native_phase_updates_at_depth32 == 6500
    and .resident_phase_reads_at_depth32 == 35250
    and .coherent_phasor_path_terms_at_depth32 == 64000
    and .hidden_matrix_decodes == 0
    and .final_matrix_decodes == 25
    and .maximum_split_reconstruction_error
      <= .matrix_reconstruction_tolerance
    and .maximum_matrix_reconstruction_error
      <= .matrix_reconstruction_tolerance
    and .maximum_phase_gauge_reconstruction_error
      <= .cell_restoration_tolerance
    and .maximum_entry_modulus_overshoot
      <= .matrix_reconstruction_tolerance
    and .minimum_nonzero_entry_modulus >= .declared_entry_floor
    and .maximum_repeated_restoration_error
      <= .cell_restoration_tolerance
    and .missing_inverse_residual >= 1e-6
    and .wrong_inverse_residual >= 1e-6
    and .reordered_inverse_residual >= 1e-6
    and .snapshot_residual <= .cell_restoration_tolerance
    and .nonunitary_control_error > .unitary_tolerance
    and .nonunitary_control_rejected
    and .expansive_entry_control_rejected
    and .unrelated_reuse_depth == 8
    and .following_reuse_cycles == 8
    and .actual_paired_phase_coherent_product
    and .actual_adjoint_source_uncompute
    and .generic_off_chirp_unitary_closure
    and .fixed_rank_across_tested_depth
    and .actual_inverse_restoration
    and .actual_restored_carrier_reuse
    and .amplitude_sensitive_host_split_required
    and (.phase_only_without_amplitude_computation | not)
    and (.globally_stable_arbitrary_unitary_closure | not)
    and .compact_complex_matrix_recurrence_exists
    and (.genuinely_distinct_phase_resource | not)
    and (.computational_advantage | not)
    and (.small_wall_crossed | not)
    and (.terminal | not)
    and $reference[0].dense_complex_live_state_bytes == 400
    and $reference[0].streaming_row_scratch_bytes == 80
    and $reference[0].dense_complex_multiply_adds_at_depth32 == 4000
    and $reference[0].retained_intermediate_matrices == 0
' "$out/phase.json" >/dev/null

for denied in \
    project-intermediate \
    null-carrier \
    reject-nonunitary \
    reject-expansive-entry
do
    set +e
    "$out/phase" "--${denied}" \
        >"$out/${denied}.stdout" \
        2>"$out/${denied}.stderr"
    rc=$?
    set -e
    [[ "$rc" -eq 2 ]]
    [[ ! -s "$out/${denied}.stdout" ]]
done
rg -q '^active paired-phase matrix projection denied$' \
    "$out/project-intermediate.stderr"
rg -q '^invalid paired-phase transaction$' \
    "$out/null-carrier.stderr"
rg -q '^paired-phase kernel is not unitary$' \
    "$out/reject-nonunitary.stderr"
rg -q '^paired-phase entry exceeds unit disk$' \
    "$out/reject-expansive-entry.stderr"

if rg -n \
    'witness|candidate_set|assignment_expansion|expected_(hash|boundary)|truth_table' \
    "$phase_source"
then
    echo "paired-phase path contains forbidden extensional state" >&2
    exit 1
fi
if rg -n \
    '"(intermediate|resident)_(matrix|phase|signature)(s|_values|_coefficients|_entries)?":[[:space:]]*[\[{]' \
    "$out/phase.json" "$out/reference.json"
then
    echo "paired-phase evidence serialized an intermediate" >&2
    exit 1
fi
rg -q 'pp_compute_product' "$phase_source"
rg -q 'pp_decode_entry' "$phase_source"
rg -q 'pp_multiply' "$phase_source"
working_assignments=$(
    rg 'working\[[^]]+\][[:space:]]*[-+*/]?=' "$phase_source"
)
if [[ $(wc -l <<<"$working_assignments") -ne 2 ]] \
    || ! rg -q \
        'carrier\.working\[cell\] = carrier\.baseline\[cell\];' \
        <<<"$working_assignments" \
    || ! rg -q \
        'carrier->working\[cell\] = pp_unit\(' \
        <<<"$working_assignments"
then
    echo "paired-phase path has an unauthorized working-cell assignment" >&2
    exit 1
fi

stack_file=$(
    find "$out" -maxdepth 1 -name 'phase_stack-*.su' -print -quit
)
maximum_stack=$(
    awk -F '\t' '$2 > maximum {maximum=$2} END {print maximum+0}' \
        "$stack_file"
)
jq -n \
    --argjson maximum_stack "$maximum_stack" \
    --slurpfile phase "$out/phase.json" \
    --slurpfile reference "$out/reference.json" '
    {
      result: "PASS",
      claim: $phase[0].claim,
      claim_ceiling:
        "BOUNDED_DOUBLE_COMPLEX_5X5_UNITARY_PUBLIC_CONSTANT_STATE_GENERATOR_PAIRED_PHASE_COORDINATE_EMBEDDING_DEPTHS1_2_4_8_16_32_ENTRY_FLOOR1E_6_SOFTWARE_REFERENCE_ONLY",
      accepted: $phase[0],
      dense_complex_reference: $reference[0],
      maximum_individual_compiler_stack_frame_bytes: $maximum_stack,
      phase_machine_advance:
        "GENERIC_OFF_CHIRP_UNITARY_MAGNITUDE_ENCODED_IN_PAIRED_UNIT_PHASORS",
      resource_adjudication:
        "PAIRED_PHASE_IS_TWICE_DENSE_COMPLEX_LIVE_COORDINATES_AND_U5_HAS_A_SMALLER_25_DOUBLE_CHART",
      remaining_primary_obstruction:
        "AMPLITUDE_SENSITIVE_HOST_SPLIT_GAUGE_INSTABILITY_AND_EQUIVALENT_COMPACT_COMPLEX_MATRIX_RECURRENCE",
      catvm_enforced: false,
      phase_only_without_amplitude_computation: false,
      globally_stable_arbitrary_unitary_closure: false,
      genuinely_distinct_phase_resource: false,
      computational_advantage: false,
      small_wall_crossed: false,
      physical_execution: false,
      unlimited_catalytic_computation: false,
      terminal: false
    }
' >"$out/summary.json"

sha256sum \
    "$phase_source" "$reference_source" \
    "$out/phase" "$out/reference" \
    "$out/phase.json" "$out/reference.json" "$out/summary.json" \
    >"$out/SHA256SUMS"
