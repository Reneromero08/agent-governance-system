#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_algebraic_symplectic_lie_relation.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
out=$1
mkdir -p "$out"

phase_source="$here/algebraic_symplectic_lie_relation_phase.c"
reference_source="$here/algebraic_symplectic_lie_relation_reference.py"
common=(
    -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror
    -Wconversion -Wshadow
)
for tool in gcc jq cmp rg sha256sum nice; do
    command -v "$tool" >/dev/null
done
[[ -x "$python" ]]

gcc "${common[@]}" "$phase_source" -lm -o "$out/phase"
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
"$python" "$reference_source" >"$out/reference.json"
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
      == "BOUNDED_DUAL_PRIME_PHASE_RESIDENT_NONLINEAR_SYMPLECTIC_LIE_SIGNATURE_GROWTH_WITH_ACTUAL_RESTORATION_AND_REUSE"
    and .port_type == "TWO_MODE_CANONICAL_WAVE_PORT"
    and .poisson_canonical_index_contraction
    and (.shared_wave_port_elimination_established | not)
    and .grades == 5
    and .maximum_degree == 12
    and .resident_dual_prime_coefficient_cells == 2050
    and .logical_packed_active_phase_payload_bytes == 32800
    and .logical_packed_cell_baseline_bytes == 32800
    and .actual_carrier_allocation_bytes >= 131200
    and .borrowed_verification_copy_bytes == 0
    and .retain_all_grade_blocks == 5
    and .retained_inverse_kernels == 0
    and .hidden_coefficient_decodes == 0
    and .final_coefficient_decodes == 2050
    and .maximum_unit_modulus_error <= 5e-10
    and .maximum_final_root_error <= 1e-3
    and .restoration_max_abs <= 5e-10
    and .maximum_reuse_restoration_error <= 5e-10
    and .successful_restoration_receipt == 1
    and .same_carrier_reuse_transactions == 8
    and .missing_inverse_residual >= 1e-6
    and .wrong_inverse_residual >= 1e-6
    and .reordered_inverse_residual >= 1e-6
    and .missing_inverse_modular_mismatch_cells > 0
    and .wrong_inverse_modular_mismatch_cells > 0
    and .reordered_inverse_modular_mismatch_cells > 0
    and .snapshot_residual <= 5e-10
    and .snapshot_restoration_receipt == 0
    and .identity_mixer_higher_grades_zero
    and .zero_kerr_all_grades_zero
    and .swapped_order_discriminated
    and .deterministic_replay
    and .actual_inverse_restoration
    and .actual_restored_carrier_reuse
    and .final_boundary_only_projection
    and .actual_resident_grade_consumed
    and .public_topology_inverse_rematerialization
    and .mathematically_exact_dual_prime_phase_algebra
    and .rational_oracle_required
    and .bounded_polynomial_lie_growth_obstruction
    and (.unbounded_growth_proved | not)
    and (.fixed_rank_closure_established | not)
    and (.genuinely_distinct_phase_resource | not)
    and (.computational_advantage | not)
    and (.small_wall_crossed | not)
    and (.physical_waveform_execution | not)
    and (.terminal | not)
    and $reference[0].result == "PASS"
    and $reference[0].successive_primary_degrees_nonzero
    and $reference[0].successive_primary_term_counts
      == [6, 32, 85, 126, 231]
    and $reference[0].identity_mixer_higher_grades_zero
    and $reference[0].swapped_first_bracket_negates_primary
    and $reference[0].compact_direct_wave_semantic_state_bytes == 64
    and $reference[0].retain_all_rational_grade_coefficient_cells == 1025
    and ($reference[0].unbounded_growth_proved | not)
    and ($reference[0].fixed_rank_reduction_found | not)
    and (
      [.primary_grades[]
        | {degree, coefficient_cells, nonzero_p17, nonzero_p19,
           hash_p17, hash_p19}]
      == [$reference[0].primary_grades[]
        | {degree, coefficient_cells, nonzero_p17, nonzero_p19,
           hash_p17, hash_p19}]
    )
    and (
      [.reuse_grades[]
        | {degree, coefficient_cells, nonzero_p17, nonzero_p19,
           hash_p17, hash_p19}]
      == [$reference[0].reuse_grades[]
        | {degree, coefficient_cells, nonzero_p17, nonzero_p19,
           hash_p17, hash_p19}]
    )
  ' "$out/phase.json" >/dev/null

for denied in project-shared-port null-carrier; do
    set +e
    "$out/phase" "--${denied}" \
        >"$out/${denied}.stdout" \
        2>"$out/${denied}.stderr"
    rc=$?
    set -e
    [[ "$rc" -eq 2 ]]
    [[ ! -s "$out/${denied}.stdout" ]]
done
rg -q '^resident shared wave port projection denied$' \
    "$out/project-shared-port.stderr"
rg -q '^invalid symplectic relation carrier$' \
    "$out/null-carrier.stderr"

if rg -n \
    'witness_list|candidate_set|truth_table|assignment_expansion|shared_port_coordinates|expected_(hash|boundary)' \
    "$phase_source" "$reference_source"
then
    echo "symplectic relation path contains forbidden extensional state" >&2
    exit 1
fi
rg -q 'sr_field_product' "$phase_source"
rg -q 'sr_accumulate_bracket' "$phase_source"
rg -q 'poisson' "$reference_source"
rg -q 'sympy.Rational' "$reference_source"

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
        "BOUNDED_TWO_MODE_RATIONAL_SU2_KERR_HAMILTONIAN_DUAL_PRIME_LIE_GRADES4_6_8_10_12_SOFTWARE_REFERENCE_ONLY",
      accepted: $phase[0],
      exact_rational_reference: $reference[0],
      maximum_compiler_stack_frame_bytes: $maximum_stack,
      phase_owned_result:
        "DUAL_PRIME_PHASE_RESIDENT_POISSON_CANONICAL_INDEX_CONTRACTION",
      adjudicated_obstruction:
        "TESTED_POLYNOMIAL_HAMILTONIAN_SIGNATURE_DEGREE_TERM_AND_RETAIN_ALL_CUSTODY_GROWTH",
      matched_compact_direct_wave_semantic_state_bytes: 64,
      exact_fixed_rank_reduction_found: false,
      unbounded_growth_proved: false,
      genuinely_distinct_phase_resource: false,
      computational_advantage: false,
      small_wall_crossed: false,
      physical_waveform_execution: false,
      terminal: false
    }
  ' >"$out/summary.json"

sha256sum \
    "$phase_source" "$reference_source" \
    "$out/phase" "$out/phase.json" "$out/reference.json" \
    "$out/summary.json" \
    >"$out/SHA256SUMS"
