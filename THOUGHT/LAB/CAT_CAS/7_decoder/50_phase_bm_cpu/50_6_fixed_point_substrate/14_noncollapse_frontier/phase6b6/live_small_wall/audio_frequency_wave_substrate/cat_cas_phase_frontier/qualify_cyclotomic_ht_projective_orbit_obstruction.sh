#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
ulimit -c 0

if [[ $# -ne 1 ]]; then
  echo \
    "usage: qualify_cyclotomic_ht_projective_orbit_obstruction.sh EVIDENCE_DIR" \
    >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/cyclotomic_ht_projective_orbit_obstruction.cpp"
oracle_path="$frontier_dir/cyclotomic_ht_projective_orbit_oracle.py"
qualifier_path="$frontier_dir/qualify_cyclotomic_ht_projective_orbit_obstruction.sh"
binary="$evidence_dir/cyclotomic_ht_projective_orbit_obstruction"
ubsan_binary="$evidence_dir/cyclotomic_ht_projective_orbit_obstruction_ubsan"
result="$evidence_dir/result.json"
replay="$evidence_dir/replay.json"
ubsan_result="$evidence_dir/ubsan.json"
oracle_result="$evidence_dir/oracle.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in g++ jq cmp sha256sum nice; do
  command -v "$tool" >/dev/null
done
test -x "$python"

g++ \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  "$source_path" \
  -o "$binary"

g++ \
  -std=c++20 \
  -O1 \
  -g \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  -fsanitize=undefined \
  -fno-sanitize-recover=all \
  "$source_path" \
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
    == "BOUNDED_EXACT_HT_ANALYTIC_INFINITE_PROJECTIVE_ORBIT_REJECTS_FIXED_FINITE_STATE_LOSSLESS_PHASE_QUOTIENT_WITH_EXACT_RESTORATION_AND_REUSE"
  and .gate_alphabet == "Z_ZETA8_DYADIC_H_AND_T"
  and .logical_phase_cells == 2
  and .analytic_projective_orbit_infinite
  and (.fixed_finite_state_lossless_quotient_possible | not)
  and .theorem_certificate.unitary == "U_EQUALS_H_T"
  and .theorem_certificate.q_plus_inverse_q
    == {
      "rational": -1,
      "sqrt2_numerator": -1,
      "sqrt2_denominator": 2
    }
  and .theorem_certificate.quadratic_integer_ring == "Z_SQRT2"
  and (.theorem_certificate.q_plus_inverse_q_is_algebraic_integer | not)
  and (.theorem_certificate.eigenvalue_ratio_is_root_of_unity | not)
  and .theorem_certificate.initial_basis_vector_is_cyclic
  and .theorem_certificate.projective_orbit_is_infinite
  and .tested_depths == [1,2,4,8,16,32,64]
  and [.depth_runs[].maximum_numerator_bits] == [1,1,2,3,5,9,17]
  and [.depth_runs[].maximum_denominator_power] == [1,1,2,3,5,9,17]
  and [.depth_runs[].logical_payload_bits] == [14,14,21,26,41,73,141]
  and ([.depth_runs[].star_norm_exactly_one] | all)
  and ([.depth_runs[].exact_algebraic_restoration] | all)
  and ([.depth_runs[].outer_carrier_backing_preserved] | all)
  and .sampled_projective_states_distinct
  and .primary.depth == 64
  and .primary.exact_algebraic_restoration
  and .reuse.depth == 23
  and .reuse.program == "H_THEN_T_CUBED"
  and .reuse.exact_algebraic_restoration
  and .fresh_restored_reuse_boundary_equal
  and .same_outer_carrier_backing
  and .restoration_generation_sequence == [1,2]
  and .baseline_reload_bytes == 0
  and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
  and (.controls.missing_inverse_restored | not)
  and (.controls.wrong_inverse_restored | not)
  and (.controls.reordered_inverse_restored | not)
  and .controls.phase_disabled_boundary_differs
  and .controls.demonstrated_nonzero_kernel_element_integer == 697
  and .controls.analytic_alias_pair_distinct
  and .controls.alias_pair_equal_mod_17_and_41
  and (.controls.alias_pair_is_normalized_transaction | not)
  and .resource_law.logical_phase_cells == 2
  and .resource_law.bounded_executor_signed64_integer_slots == 8
  and .resource_law.phase_object_bytes == 40
  and .resource_law.carrier_inline_bytes == 80
  and .resource_law.restoration_verification_baseline_inline_bytes == 80
  and .resource_law.retained_boundary_inline_bytes == 80
  and .resource_law.transaction_run_record_bytes == 120
  and .resource_law.retained_module_tape_bytes == 0
  and .resource_law.retained_inverse_history_bytes == 0
  and .resource_law.public_program_descriptor_bytes == 16
  and .resource_law.verification_depth_run_array_bytes == 840
  and .resource_law.compiler_binary_allocator_and_whole_process_excluded
  and (.resource_law.fixed_bit_payload_established | not)
  and (.resource_law.whole_process_rss_claimed | not)
  and .matched_compact_classical.identical_exact_two_cell_recurrence
  and .matched_compact_classical.boundary_error == 0
  and (.catvm_custody_established | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.physical_waveform_execution | not)
  and (.physical_bit_replacement | not)
  and (.unbounded_computation_established | not)
  and (.terminal | not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and .oracle
    == "INDEPENDENT_FRACTION_HT_RECURRENCE_AND_QUADRATIC_INTEGER_CERTIFICATE"
  and (.production_backend_imported | not)
  and .theorem_certificate.unitary == "U_EQUALS_H_T"
  and .theorem_certificate.q_plus_inverse_q
    == {
      "rational": -1,
      "sqrt2_denominator": 2,
      "sqrt2_numerator": -1
    }
  and .theorem_certificate.quadratic_integer_ring == "Z_SQRT2"
  and (.theorem_certificate.q_plus_inverse_q_is_algebraic_integer | not)
  and .theorem_certificate.root_of_unity_would_require_algebraic_integer_sum
  and (.theorem_certificate.eigenvalue_ratio_is_root_of_unity | not)
  and .theorem_certificate.initial_basis_vector_is_cyclic
  and .theorem_certificate.analytic_projective_orbit_infinite
  and .tested_depths == [1,2,4,8,16,32,64]
  and [.verified_depth_runs[].maximum_numerator_bits] == [1,1,2,3,5,9,17]
  and [.verified_depth_runs[].maximum_denominator_power] == [1,1,2,3,5,9,17]
  and [.verified_depth_runs[].logical_payload_bits] == [14,14,21,26,41,73,141]
  and .all_boundaries_match
  and .all_star_norms_exactly_one
  and .all_restorations_exact
  and .sampled_projective_states_distinct
  and .fresh_restored_reuse_boundary_equal
  and .inverse_controls_pass
  and .demonstrated_nonzero_kernel_element_integer == 697
  and .alias_pair_equal_mod_17_and_41
  and (.alias_pair_is_normalized_transaction | not)
  and (.fixed_finite_state_lossless_quotient_possible | not)
  and .matched_compact_classical_recurrence_identical
  and (.terminal | not)
' "$oracle_result" >/dev/null

if rg -n \
  'witness_list|candidate_set|truth_table|assignment_expansion|dense_operator' \
  "$source_path" "$oracle_path"
then
  echo "HT projective-orbit package contains forbidden extensional state" >&2
  exit 1
fi

jq -n \
  --slurpfile accepted "$result" \
  --slurpfile oracle "$oracle_result" '
  {
    result: "PASS",
    claim:
      "BOUNDED_EXACT_HT_ANALYTIC_INFINITE_PROJECTIVE_ORBIT_REJECTS_FIXED_FINITE_STATE_LOSSLESS_PHASE_QUOTIENT_WITH_EXACT_RESTORATION_AND_REUSE",
    claim_ceiling:
      "LINUX_X86_64_DIRECT_PROCESS_TWO_CELL_Z_ZETA8_DYADIC_ANALYTIC_HT_ORBIT_DEPTHS1_2_4_8_16_32_64_REUSE_DEPTH23_SOFTWARE_ONLY",
    verification_level: "INDEPENDENT_ORACLE_REEXECUTION",
    restoration_classification: "EXACT_ALGEBRAIC_RESTORATION",
    accepted: $accepted[0],
    independent_oracle: $oracle[0],
    fixed_finite_state_lossless_quotient_possible: false,
    unbounded_symbolic_state_ruled_out: false,
    approximation_ruled_out: false,
    physical_continuous_carrier_ruled_out: false,
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
    "cyclotomic_ht_projective_orbit_obstruction.cpp"
  hash_with_label "$oracle_path" \
    "cyclotomic_ht_projective_orbit_oracle.py"
  hash_with_label "$qualifier_path" \
    "qualify_cyclotomic_ht_projective_orbit_obstruction.sh"
  hash_with_label "$binary" \
    "cyclotomic_ht_projective_orbit_obstruction"
  hash_with_label "$ubsan_binary" \
    "cyclotomic_ht_projective_orbit_obstruction_ubsan"
  hash_with_label "$result" "result.json"
  hash_with_label "$oracle_result" "oracle.json"
  hash_with_label "$evidence_dir/qualification.json" "qualification.json"
} >"$evidence_dir/SHA256SUMS"

echo "cyclotomic H-T projective-orbit qualification: PASS"
