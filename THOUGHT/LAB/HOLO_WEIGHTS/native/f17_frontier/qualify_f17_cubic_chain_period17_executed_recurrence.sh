#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_cubic_chain_period17_executed_recurrence.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_cubic_chain_period17_executed_recurrence.py"
oracle_path="$frontier_dir/f17_cubic_chain_period17_executed_recurrence_oracle.py"
module_path="$frontier_dir/f17_cubic_chain_period17_cyclotomic_module.py"
adaptive_path="$frontier_dir/f17_cubic_chain_adaptive_gauge.py"
qualifier_path="$frontier_dir/qualify_f17_cubic_chain_period17_executed_recurrence.sh"
expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_EXECUTED_RECURRENCE_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_EXECUTED_RECURRENCE_ORACLE_RESULTS.json"
provenance_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_EXECUTED_RECURRENCE_PROVENANCE.json"
result="$evidence_dir/result.full.json"
oracle_result="$evidence_dir/oracle.full.json"
summary="$evidence_dir/result.summary.json"
oracle_summary="$evidence_dir/oracle.summary.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in jq cmp sha256sum nice rg; do
  command -v "$tool" >/dev/null
done
test -x "$python"
jq empty "$expected_path" "$oracle_expected_path" "$provenance_path"

for sealed_path in \
  "$source_path" \
  "$oracle_path" \
  "$module_path" \
  "$adaptive_path" \
  "$qualifier_path" \
  "$expected_path" \
  "$oracle_expected_path"
do
  sealed_name=$(basename "$sealed_path")
  sealed_expected=$(jq -r --arg name "$sealed_name" \
    '.files[$name] // empty' "$provenance_path")
  test -n "$sealed_expected"
  test "$(sha256sum "$sealed_path" | cut -d' ' -f1)" = "$sealed_expected"
done

rg -F 'import f17_cubic_chain_period17_cyclotomic_module as cyclo' \
  "$source_path" >/dev/null
if rg -n '^[[:space:]]*(from|import)[[:space:]]+f17_cubic_chain_' \
  "$oracle_path"; then
  echo "independent oracle imports a production module" >&2
  exit 1
fi

"$python" -m py_compile "$source_path" "$oracle_path"

nice -n 10 "$python" -X dev "$source_path" \
  >"$result" 2>"$evidence_dir/result.stderr"
test ! -s "$evidence_dir/result.stderr"
nice -n 10 "$python" -X dev "$oracle_path" "$result" \
  >"$oracle_result" 2>"$evidence_dir/oracle.stderr"
test ! -s "$evidence_dir/oracle.stderr"

jq -cS '{
  result,
  claim_candidate,
  claim_ceiling,
  classification_candidate,
  verification_level_candidate,
  restoration_class,
  coefficient_field,
  integral_carrier_ring,
  characteristic_order,
  characteristic_constant_coefficients_zero,
  native_k_monic_factor_degree,
  native_k_recurrence_basis_messages,
  native_k_monic_factor_constant_zero,
  native_k_monic_factor_integral_unit_certified,
  scalar_q_order16_established,
  reversible_integral_rolling_window_established,
  fixed_resident_basis_message_bank_executed,
  runtime_executes_characteristic_recurrence,
  runtime_dense_block_forward_applications_per_transaction,
  runtime_dense_block_inverse_applications_per_transaction,
  runtime_dense_block_total_applications_per_transaction,
  block_certificates:(.block_certificates|with_entries(
    .value |= del(.characteristic)
  )),
  tested_periods,
  cases:[.cases[]|del(.boundary)],
  all_dense_direct_boundaries_equal,
  all_cases_restored_exactly,
  restoration_reuse_case,
  controls,
  matched_classical,
  resource_law,
  not_established,
  next_obstruction,
  terminal
}' "$result" >"$summary"
jq -cS . "$expected_path" >"$evidence_dir/result.expected.json"
cmp "$summary" "$evidence_dir/result.expected.json"

jq -cS . "$oracle_result" >"$oracle_summary"
jq -cS . "$oracle_expected_path" \
  >"$evidence_dir/oracle.expected.json"
cmp "$oracle_summary" "$evidence_dir/oracle.expected.json"

jq -e '
  .result == "PASS"
  and .classification_candidate == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level_candidate == "SEPARATE_REFERENCE_PARITY"
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and .coefficient_field == "Q_ZETA17"
  and .integral_carrier_ring == "Z_ZETA17"
  and .characteristic_order == 17
  and .characteristic_constant_coefficients_zero
  and .native_k_monic_factor_degree == 16
  and .native_k_recurrence_basis_messages == 16
  and .native_k_monic_factor_constant_zero == {
    "primary":true,
    "reuse":false
  }
  and .native_k_monic_factor_integral_unit_certified == {
    "primary":false,
    "reuse":false
  }
  and (.scalar_q_order16_established|not)
  and (.reversible_integral_rolling_window_established|not)
  and .fixed_resident_basis_message_bank_executed
  and .runtime_executes_characteristic_recurrence
  and .runtime_dense_block_forward_applications_per_transaction == 16
  and .runtime_dense_block_inverse_applications_per_transaction == 16
  and .runtime_dense_block_total_applications_per_transaction == 32
  and .tested_periods == [1,4,16,64]
  and [.cases[].family] == [
    "PRIMARY","REUSE","PRIMARY","REUSE",
    "PRIMARY","REUSE","PRIMARY","REUSE"
  ]
  and [.cases[].periods] == [1,1,4,4,16,16,64,64]
  and .all_dense_direct_boundaries_equal
  and .all_cases_restored_exactly
  and all(.cases[];
    .dense_direct_boundary_equal
    and .restored_exactly
    and .same_backing
    and .message_slots == 18
    and .message_integer_cells == 4896
    and .coefficient_register_integer_cells == 256
    and .stats.basis_forward_block_applications == 16
    and .stats.basis_inverse_block_applications == 16
    and .canonical_restored_state == {
      "active":false,
      "all_state_zero":true,
      "coefficient_registers":16,
      "generation":1,
      "lease":1,
      "message_slots":18,
      "pending_operations":0,
      "phase":"RESTORED"
    }
  )
  and .restoration_reuse_case.primary_restored_exactly
  and .restoration_reuse_case.reuse_restored_exactly
  and .restoration_reuse_case.same_original_backing
  and .restoration_reuse_case.fresh_restored_reuse_boundary_equal
  and .restoration_reuse_case.generation == 2
  and .restoration_reuse_case.lease == 2
  and .restoration_reuse_case.canonical_restored_state == {
    "active":false,
    "all_state_zero":true,
    "coefficient_registers":16,
    "generation":2,
    "lease":2,
    "message_slots":18,
    "pending_operations":0,
    "phase":"RESTORED"
  }
  and .restoration_reuse_case.retained_forward_inverse_enabling_basis_integer_cells
    == 4352
  and .restoration_reuse_case.retained_seed_plus_basis_integer_cells
    == 4624
  and .restoration_reuse_case.separate_inverse_operation_log_bytes == 0
  and .restoration_reuse_case.baseline_reload_bytes == 0
  and all(.controls[]; .)
  and .matched_classical.identical_cyclotomic_characteristic_recurrence
  and .matched_classical.identical_fixed_basis_contraction
  and .matched_classical.identical_recurrence_lifecycle_and_restoration_available
  and .matched_classical.direct_streaming_dense_block_message_integer_cells == 544
  and .matched_classical.direct_streaming_dense_block_work_linear_in_periods
  and .matched_classical.direct_streaming_dense_block_comparison_is_forward_only
  and .matched_classical.direct_streaming_dense_block_restoration_classification
    == "NO_RESTORATION_CLAIM"
  and .matched_classical.polynomial_multiplication_count_is_logarithmic_in_periods_after_basis
  and (.matched_classical.growing_integer_bit_operation_work_bounded|not)
  and (.matched_classical.comparison_establishes_advantage|not)
  and (.matched_classical.strongest_family_specific_method_established|not)
  and .resource_law.accounting_scope
    == "COMPONENT_LEVEL_NAMED_LOGICAL_INTEGER_CELLS_NOT_EXACT_PROCESS_PEAK"
  and .resource_law.fixed_message_slots == 18
  and .resource_law.fixed_message_slots_are_not_fixed_total_footprint
  and .resource_law.fixed_message_bank_integer_cells == 4896
  and .resource_law.coefficient_register_integer_cells == 256
  and .resource_law.compiled_operator_integer_cells_per_family == 4624
  and .resource_law.compiled_operator_integer_cells_two_families == 9248
  and .resource_law.compiled_characteristic_integer_cells_per_family == 288
  and .resource_law.compiled_characteristic_integer_cells_two_families == 576
  and .resource_law.both_compiled_families_retained_during_full_run
  and .resource_law.retained_forward_inverse_enabling_basis_integer_cells_per_transaction
    == 4352
  and .resource_law.retained_seed_plus_basis_integer_cells_per_transaction
    == 4624
  and .resource_law.separate_inverse_operation_log_bytes == 0
  and .resource_law.baseline_reload_bytes == 0
  and .resource_law.integer_width_counted
  and .resource_law.dense_direct_baseline_width_and_payload_recorded
  and (.resource_law.named_logical_cell_accounting_is_exact_total|not)
  and (.resource_law.bit_operation_peak_bounded|not)
  and (.resource_law.whole_process_peak_bounded|not)
  and (.not_established|index("DISTINCT_PHASE_RESOURCE")) != null
  and (.not_established|index("COMPUTATIONAL_ADVANTAGE")) != null
  and (.not_established|index("SMALL_WALL_CROSSING")) != null
  and (.terminal|not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and .production_module_imported == false
  and .production_compiler_called == false
  and .production_binary_polynomial_power_called == false
  and .production_basis_builder_called == false
  and .production_inverse_called == false
  and .all_family_checks
  and .all_case_checks
  and .all_restored_exactly
  and .all_cross_family_reuse_checks
  and all(.family_checks[];
    .descriptor_sha256_equal
    and .operator_sha256_equal
    and .characteristic_sha256_equal
    and .characteristic_constant_zero
    and .whole_operator_annihilator_identity_exact
  )
  and [.case_checks[].periods] == [1,1,4,4,16,16,64,64]
  and all(.case_checks[];
    .sequential_recurrence_boundary_equal
    and .direct_dense_boundary_equal
    and .boundary_sha256_equal
    and .boundary_payload_bits_equal
    and .recurrence_maximum_carrier_payload_bits_equal
    and .recurrence_maximum_coefficient_signed_bits_equal
    and .recurrence_maximum_nonzero_message_slots_equal
    and .recurrence_maximum_nonzero_coefficient_registers_equal
    and .direct_maximum_two_message_payload_bits_equal
    and .direct_maximum_coefficient_signed_bits_equal
  )
  and all(.restoration_checks[];
    .restored_exactly
    and .same_message_and_coefficient_backing
    and .generation == 1
    and .lease == 1
    and .message_slots == 18
    and .message_integer_cells == 4896
    and .coefficient_register_integer_cells == 256
  )
  and .cross_family_restored_reuse_check.primary_restored_exactly
  and .cross_family_restored_reuse_check.reuse_restored_exactly
  and .cross_family_restored_reuse_check.same_original_message_and_coefficient_backing
  and .cross_family_restored_reuse_check.primary_boundary_sha256_equal
  and .cross_family_restored_reuse_check.reuse_boundary_sha256_equal
  and .cross_family_restored_reuse_check.fresh_restored_reuse_boundary_equal
  and .cross_family_restored_reuse_check.fresh_reuse_restored_exactly
  and .cross_family_restored_reuse_check.generation == 2
  and .cross_family_restored_reuse_check.lease == 2
  and .cross_family_restored_reuse_check.production_generation_equal
  and .cross_family_restored_reuse_check.production_lease_equal
  and .cross_family_restored_reuse_check.all_state_zero
  and .cross_family_restored_reuse_check.separate_inverse_operation_log_bytes == 0
  and .cross_family_restored_reuse_check.baseline_reload_bytes == 0
  and .independent_recurrence_method
    == "SEQUENTIAL_MULTIPLICATION_BY_X_MOD_MONIC_Q16"
  and .independent_exact_integer_payload_and_width_reexecution
  and .fixed_18_resident_message_slots_confirmed
  and .fixed_16_cyclotomic_coefficient_registers_confirmed
  and .fixed_slots_are_not_fixed_total_footprint
  and (.reversible_integral_rolling_window_established|not)
  and .identical_compact_classical_recurrence
  and (.distinct_phase_resource_established|not)
  and (.computational_advantage|not)
  and (.small_wall_crossed|not)
  and (.terminal|not)
' "$oracle_result" >/dev/null

echo "F17 period-17 executed recurrence qualification: PASS"
