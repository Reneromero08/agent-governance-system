#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_cubic_chain_period17_cyclotomic_module.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_cubic_chain_period17_cyclotomic_module.py"
oracle_path="$frontier_dir/f17_cubic_chain_period17_cyclotomic_module_oracle.py"
lift_path="$frontier_dir/f17_cubic_chain_period17_exact_lift.py"
adaptive_path="$frontier_dir/f17_cubic_chain_adaptive_gauge.py"
krylov_path="$frontier_dir/f17_cubic_chain_period17_krylov.py"
qualifier_path="$frontier_dir/qualify_f17_cubic_chain_period17_cyclotomic_module.sh"
expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_CYCLOTOMIC_MODULE_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_CYCLOTOMIC_MODULE_ORACLE_RESULTS.json"
lift_expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_EXACT_LIFT_CONTROL_RESULTS.json"
provenance_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_CYCLOTOMIC_MODULE_PROVENANCE.json"
result="$evidence_dir/result.full.json"
oracle_result="$evidence_dir/oracle.full.json"
lift_result="$evidence_dir/exact_lift.full.json"
summary="$evidence_dir/result.summary.json"
oracle_summary="$evidence_dir/oracle.summary.json"
lift_summary="$evidence_dir/exact_lift.summary.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in jq cmp sha256sum nice rg; do
  command -v "$tool" >/dev/null
done
test -x "$python"
jq empty \
  "$expected_path" \
  "$oracle_expected_path" \
  "$lift_expected_path" \
  "$provenance_path"

for sealed_path in \
  "$source_path" \
  "$oracle_path" \
  "$lift_path" \
  "$adaptive_path" \
  "$krylov_path" \
  "$qualifier_path" \
  "$expected_path" \
  "$oracle_expected_path" \
  "$lift_expected_path"
do
  sealed_name=$(basename "$sealed_path")
  sealed_expected=$(jq -r --arg name "$sealed_name" \
    '.files[$name] // empty' "$provenance_path")
  test -n "$sealed_expected"
  test "$(sha256sum "$sealed_path" | cut -d' ' -f1)" = "$sealed_expected"
done

rg -F 'import f17_cubic_chain_adaptive_gauge as adaptive' \
  "$source_path" "$lift_path" >/dev/null
rg -F 'import f17_cubic_chain_period17_krylov as period17' \
  "$lift_path" >/dev/null
if rg -n '^[[:space:]]*(from|import)[[:space:]]+f17_cubic_chain_' \
  "$oracle_path"; then
  echo "independent oracle imports a production module" >&2
  exit 1
fi

"$python" -m py_compile "$source_path" "$oracle_path" "$lift_path"

nice -n 10 "$python" -X dev "$lift_path" \
  >"$lift_result" 2>"$evidence_dir/exact_lift.stderr"
test ! -s "$evidence_dir/exact_lift.stderr"
jq -cS '{
  result,
  experiment,
  scope,
  crt_moduli,
  cases:[.cases[]|{
    family,
    crt_candidate:{
      degree:.crt_candidate.degree,
      combined_modulus_bits:.crt_candidate.combined_modulus_bits,
      maximum_candidate_coefficient_signed_bits:
        .crt_candidate.maximum_candidate_coefficient_signed_bits,
      nonzero_candidate_coefficients:
        .crt_candidate.nonzero_candidate_coefficients,
      candidate_sha256:.crt_candidate.candidate_sha256,
      crt_stages:[.crt_candidate.crt_stages[]|{
        combined_modulus_bits,
        maximum_candidate_coefficient_signed_bits,
        coefficients_unchanged_from_previous_stage,
        candidate_sha256
      }]
    },
    exact_verification
  }],
  all_degrees_match_prior_modular_dimensions,
  exact_z_zeta17_dependency_lift_established,
  dense_period_block_materialized,
  rational_matrix_materialized,
  assignment_table_materialized,
  relation_table_materialized,
  distinct_phase_resource_established,
  computational_advantage,
  small_wall_crossed,
  terminal
}' "$lift_result" >"$lift_summary"
jq -cS . "$lift_expected_path" \
  >"$evidence_dir/exact_lift.expected.json"
cmp "$lift_summary" "$evidence_dir/exact_lift.expected.json"

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
  native_module,
  blocks:(.blocks|with_entries(
    .value |= del(.operator,.characteristic)
  )),
  all_characteristic_identities_exact,
  exact_native_cyclotomic_recurrence_order_upper_bound,
  exact_native_minimal_order_established,
  native_cyclotomic_recurrence_certified_not_executed,
  runtime_executes_dense_17_by_17_cyclotomic_block,
  prior_modular_q_rank_lower_bounds,
  prior_modular_dependencies_lifted_to_q_recurrence,
  q_scalar_recurrence_order_upper_bound17_established,
  restriction_of_scalars_dimension_per_k_coefficient,
  native_k_coefficient_maps_expand_to_16_by_16_q_linear_operators,
  coefficient_ring_change_is_not_q_order_reduction,
  coefficient_ring_changed_to_native_phase_field,
  cases:[.cases[]|{
    periods,
    equivalent_edges,
    equivalent_nodes,
    family,
    boundary_sha256,
    boundary_payload_bits,
    message_slots,
    carrier_integer_cells,
    stats,
    restored_exactly,
    same_backing,
    canonical_restored_state
  }],
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
  and .crt_moduli == [41,73,65537,65539,65543,65551,65557,65563]
  and [.cases[].family] == ["PRIMARY","REUSE"]
  and [.cases[].crt_candidate.degree] == [241,256]
  and [.cases[].crt_candidate.combined_modulus_bits] == [108,108]
  and [.cases[].crt_candidate.maximum_candidate_coefficient_signed_bits]
    == [108,108]
  and [.cases[].exact_verification.completed] == [true,true]
  and [.cases[].exact_verification.resource_cap_reached]
    == [false,false]
  and [.cases[].exact_verification.candidate_residual_zero]
    == [false,false]
  and [.cases[].exact_verification.residual_nonzero_cells] == [272,272]
  and .all_degrees_match_prior_modular_dimensions
  and (.exact_z_zeta17_dependency_lift_established|not)
  and (.dense_period_block_materialized|not)
  and (.rational_matrix_materialized|not)
  and (.terminal|not)
' "$lift_result" >/dev/null

jq -e '
  .result == "PASS"
  and .classification_candidate == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level_candidate == "SEPARATE_REFERENCE_PARITY"
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and .native_module.coefficient_field == "Q_ZETA17"
  and .native_module.integral_carrier_ring == "Z_ZETA17"
  and .native_module.phase_states == 17
  and .native_module.integer_coefficients_per_phase_state == 16
  and .all_characteristic_identities_exact
  and .blocks.primary.characteristic_order == 17
  and .blocks.reuse.characteristic_order == 17
  and .exact_native_cyclotomic_recurrence_order_upper_bound == 17
  and (.exact_native_minimal_order_established|not)
  and .native_cyclotomic_recurrence_certified_not_executed
  and .runtime_executes_dense_17_by_17_cyclotomic_block
  and .prior_modular_q_rank_lower_bounds == {"primary":241,"reuse":256}
  and (.prior_modular_dependencies_lifted_to_q_recurrence|not)
  and (.q_scalar_recurrence_order_upper_bound17_established|not)
  and .restriction_of_scalars_dimension_per_k_coefficient == 16
  and .native_k_coefficient_maps_expand_to_16_by_16_q_linear_operators
  and .coefficient_ring_change_is_not_q_order_reduction
  and [.cases[].periods] == [1,1,2,2,4,4,8,8]
  and [.cases[].carrier_integer_cells]
    == [544,544,816,816,1088,1088,1360,1360]
  and [.cases[].stats.maximum_coefficient_signed_bits]
    == [36,37,71,72,141,144,283,289]
  and [.cases[].restored_exactly] == [true,true,true,true,true,true,true,true]
  and [.cases[].same_backing] == [true,true,true,true,true,true,true,true]
  and .all_cases_restored_exactly
  and .restoration_reuse_case.primary_restored_exactly
  and .restoration_reuse_case.reuse_restored_exactly
  and .restoration_reuse_case.same_original_backing
  and .restoration_reuse_case.fresh_restored_reuse_boundary_equal
  and .restoration_reuse_case.canonical_restored_state.all_messages_zero
  and .restoration_reuse_case.generation == 2
  and .restoration_reuse_case.lease == 2
  and .restoration_reuse_case.retained_inverse_history_bytes == 0
  and .restoration_reuse_case.baseline_reload_bytes == 0
  and ([.controls[]] | all)
  and .matched_classical.identical_17_state_cyclotomic_matrix_recurrence
  and .matched_classical.identical_characteristic_recurrence
  and (.matched_classical.strongest_family_specific_method_established|not)
  and .resource_law.accounting_scope
    == "COMPONENT_LEVEL_NAMED_LOGICAL_INTEGER_CELLS_NOT_EXACT_PROCESS_PEAK"
  and (.resource_law.named_logical_cell_accounting_is_exact_total|not)
  and (.resource_law.sympy_characteristic_internal_temporaries_bounded|not)
  and (.resource_law.whole_process_peak_bounded|not)
  and (.terminal|not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and (.production_module_imported|not)
  and (.production_compiler_called|not)
  and (.production_operator_builder_called|not)
  and (.production_characteristic_called|not)
  and (.production_inverse_called|not)
  and .all_family_checks
  and .all_boundaries_equal
  and .exact_subtractive_inverse_zero
  and .exact_native_cyclotomic_recurrence_order_upper_bound == 17
  and (.exact_native_minimal_order_established|not)
  and .native_cyclotomic_recurrence_certified_not_executed
  and .runtime_executes_dense_17_by_17_cyclotomic_block
  and (.prior_modular_dependencies_lifted_to_q_recurrence|not)
  and (.q_scalar_recurrence_order_upper_bound17_established|not)
  and .restriction_of_scalars_dimension_per_k_coefficient == 16
  and .coefficient_ring_change_is_not_q_order_reduction
  and .identical_compact_classical_cyclotomic_recurrence
  and (.resource_law.named_logical_cell_accounting_is_exact_total|not)
  and (.distinct_phase_resource_established|not)
  and (.computational_advantage|not)
  and (.small_wall_crossed|not)
  and (.terminal|not)
' "$oracle_result" >/dev/null

echo "F17 period-17 native cyclotomic-module qualification passed"
