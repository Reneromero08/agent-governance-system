#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
ulimit -c 0

if [[ $# -ne 1 ]]; then
  echo \
    "usage: qualify_wilczek_zee_nonabelian_phase_frame.sh EVIDENCE_DIR" \
    >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/wilczek_zee_nonabelian_phase_frame.c"
oracle_path="$frontier_dir/wilczek_zee_nonabelian_phase_frame_oracle.py"
qualifier_path="$frontier_dir/qualify_wilczek_zee_nonabelian_phase_frame.sh"
binary="$evidence_dir/wilczek_zee_nonabelian_phase_frame"
ubsan_binary="$evidence_dir/wilczek_zee_nonabelian_phase_frame_ubsan"
result="$evidence_dir/result.json"
replay="$evidence_dir/replay.json"
ubsan_result="$evidence_dir/ubsan.json"
oracle_result="$evidence_dir/oracle.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in gcc jq cmp sha256sum nice rg; do
  command -v "$tool" >/dev/null
done
test -x "$python"

gcc \
  -std=c11 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  "$source_path" \
  -lm \
  -o "$binary"

gcc \
  -std=c11 \
  -O1 \
  -g \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  -fsanitize=undefined \
  -fno-sanitize-recover=all \
  "$source_path" \
  -lm \
  -o "$ubsan_binary"

nice -n 10 "$binary" >"$result" 2>"$evidence_dir/result.stderr"
nice -n 10 "$binary" >"$replay" 2>"$evidence_dir/replay.stderr"
nice -n 10 "$ubsan_binary" \
  >"$ubsan_result" 2>"$evidence_dir/ubsan.stderr"

test ! -s "$evidence_dir/result.stderr"
test ! -s "$evidence_dir/replay.stderr"
test ! -s "$evidence_dir/ubsan.stderr"
cmp "$result" "$replay"
cmp "$result" "$ubsan_result"

"$python" -m py_compile "$oracle_path"
nice -n 10 "$python" -X dev "$oracle_path" "$result" \
  >"$oracle_result" 2>"$evidence_dir/oracle.stderr"
test ! -s "$evidence_dir/oracle.stderr"

jq -e '
  .result == "PASS"
  and .claim_candidate
    == "BOUNDED_NONABELIAN_WILCZEK_ZEE_SHARED_PHASE_FRAME_NONCOMMUTING_HOLONOMY_COMPOSITION_WITH_NUMERICAL_RESTORATION_AND_REUSE"
  and .carrier_geometry == "U2_DARK_FRAME_OVER_PUBLIC_CP2_BRIGHT_RAY"
  and ((.public_parameters.alpha - 1) | fabs) <= 1e-15
  and ((.public_parameters.beta - 0.7) | fabs) <= 1e-15
  and .loop_segments == 512
  and ((.numerical_tolerance_predeclared - 1e-9) | fabs) <= 1e-24
  and .primary.public_word
    == ["PHI1_POSITIVE_LOOP","PHI2_POSITIVE_LOOP"]
  and .primary.formula_error <= 1e-9
  and .primary.continuous_limit_error > 1e-6
  and .primary.continuous_limit_error < 1e-4
  and .primary.unitarity_error <= 1e-9
  and .primary.restoration_error <= 1e-9
  and .reordered_forward.public_word
    == ["PHI2_POSITIVE_LOOP","PHI1_POSITIVE_LOOP"]
  and .reordered_forward.formula_error <= 1e-9
  and .reordered_forward.boundary_difference_frobenius > 1
  and .reuse.public_word
    == [
      "PHI1_NEGATIVE_LOOP",
      "PHI2_POSITIVE_LOOP",
      "PHI1_POSITIVE_LOOP"
    ]
  and .reuse.formula_error <= 1e-9
  and .reuse.unitarity_error <= 1e-9
  and .reuse.restoration_error <= 1e-9
  and .fresh_restored_reuse_boundary_error <= 1e-9
  and .same_outer_carrier_variable
  and .actual_restored_carrier_reused
  and .restoration_generation_sequence == [1,2]
  and (.restoration_generation_lease_enforced | not)
  and .baseline_reload_bytes == 0
  and .stress_cycles == 100
  and .stress_maximum_restoration_error <= 1e-9
  and .restoration_classification == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and (.controls.missing_inverse_restored | not)
  and (.controls.wrong_inverse_restored | not)
  and .controls.reordered_inverse_applicable
  and (.controls.reordered_inverse_restored | not)
  and .controls.noncommuting_loop_order_changes_boundary
  and .controls.public_dark_frame_orthonormal
  and .controls.public_bright_dark_orthogonal
  and .controls.transported_edge_overlap_positive_hermitian
  and (.controls.null_carrier_accepted | not)
  and .resource_law.resident_phase_frame_complex128_cells == 6
  and .resource_law.resident_phase_frame_inline_bytes == 96
  and .resource_law.restoration_verification_baseline_inline_bytes == 96
  and .resource_law.final_boundary_inline_bytes == 64
  and .resource_law.public_loop_descriptor_inline_bytes_each == 32
  and .resource_law.declared_named_per_edge_object_subtotal_bytes == 448
  and .resource_law.o2_compiler_reported_transport_function_stack_bytes == 784
  and (.resource_law.complete_per_edge_peak_bounded | not)
  and .resource_law.retained_edge_history_bytes == 0
  and .resource_law.compiler_stack_padding_code_libm_allocator_and_whole_process_excluded
  and (.resource_law.whole_process_rss_claimed | not)
  and .matched_compact_classical.identical_2x2_holonomy_recurrence
  and .matched_compact_classical.matrix_state_inline_bytes == 64
  and (.matched_compact_classical.matrix_state_minimal_claimed | not)
  and .matched_compact_classical.closed_form_fixed_loop_modules_available
  and .matched_compact_classical.primary_boundary_error <= 1e-9
  and .matched_compact_classical.reuse_boundary_error <= 1e-9
  and (.matched_compact_classical.runtime_advantage_claimed | not)
  and .shared_phase_frame_consumed_by_multiple_modules
  and (.intermediate_frame_projected | not)
  and .final_boundary_projected_before_inverse
  and (.catvm_custody_established | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.physical_waveform_execution | not)
  and (.physical_bit_replacement | not)
  and (.unbounded_computation_established | not)
  and .claim_ceiling
    == "LINUX_X86_64_DIRECT_PROCESS_COMPLEX128_C3_DARK_TWO_FRAME_PUBLIC_CP2_PHI1_PHI2_LOOPS_SEGMENTS512_PRIMARY_TWO_MODULE_REUSE_THREE_MODULE_SOFTWARE_ONLY"
  and (.terminal | not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and .oracle
    == "INDEPENDENT_MPMATH80_CP2_DARK_FRAME_POLAR_TRANSPORT_AND_DISCRETE_LOOP_PRODUCT"
  and (.production_backend_imported | not)
  and .precision_decimal_digits == 80
  and .loop_segments == 512
  and .primary_boundary_matches
  and .reordered_boundary_matches
  and .reuse_boundary_matches
  and .all_discrete_loop_formulas_match_below_1e_65
  and .primary_restoration_below_1e_65
  and .reuse_restoration_below_1e_65
  and .fresh_restored_reuse_boundary_equal
  and .loop_order_noncommutator_frobenius > 1
  and .loop_order_noncommutes
  and (.reordered_inverse_restored | not)
  and .continuous_limit_distinct_from_finite_edge_product
  and .matched_compact_classical_2x2_recurrence_identical
  and .closed_form_fixed_loop_modules_available
  and (.distinct_phase_resource_established | not)
  and (.terminal | not)
' "$oracle_result" >/dev/null

if rg -n \
  'witness_list|candidate_set|truth_table|assignment_expansion|dense_operator' \
  "$source_path" "$oracle_path"
then
  echo "Wilczek-Zee phase-frame package contains forbidden extensional state" >&2
  exit 1
fi

jq -n \
  --slurpfile accepted "$result" \
  --slurpfile oracle "$oracle_result" '
  {
    result: "PASS",
    claim:
      "BOUNDED_NONABELIAN_WILCZEK_ZEE_SHARED_PHASE_FRAME_NONCOMMUTING_HOLONOMY_COMPOSITION_WITH_NUMERICAL_RESTORATION_AND_REUSE",
    claim_ceiling:
      "LINUX_X86_64_DIRECT_PROCESS_COMPLEX128_C3_DARK_TWO_FRAME_PUBLIC_CP2_PHI1_PHI2_LOOPS_SEGMENTS512_PRIMARY_TWO_MODULE_REUSE_THREE_MODULE_SOFTWARE_ONLY",
    verification_level: "INDEPENDENT_ORACLE_REEXECUTION",
    restoration_classification: "NUMERICAL_PHYSICAL_STATE_RESTORATION",
    accepted: $accepted[0],
    independent_oracle: $oracle[0],
    nonabelian_shared_phase_frame_established: true,
    catvm_custody_established: false,
    distinct_phase_resource_established: false,
    computational_advantage: false,
    small_wall_crossed: false,
    terminal: false
  }
' >"$evidence_dir/qualification.json"

hash_with_label() {
  local digest
  digest=$(sha256sum "$1")
  printf '%s  %s\n' "${digest%% *}" "$2"
}

{
  hash_with_label "$source_path" \
    "wilczek_zee_nonabelian_phase_frame.c"
  hash_with_label "$oracle_path" \
    "wilczek_zee_nonabelian_phase_frame_oracle.py"
  hash_with_label "$qualifier_path" \
    "qualify_wilczek_zee_nonabelian_phase_frame.sh"
  hash_with_label "$binary" \
    "wilczek_zee_nonabelian_phase_frame"
  hash_with_label "$ubsan_binary" \
    "wilczek_zee_nonabelian_phase_frame_ubsan"
  hash_with_label "$result" "result.json"
  hash_with_label "$oracle_result" "oracle.json"
  hash_with_label "$evidence_dir/qualification.json" "qualification.json"
} >"$evidence_dir/SHA256SUMS"

echo "Wilczek-Zee non-Abelian phase-frame qualification: PASS"
