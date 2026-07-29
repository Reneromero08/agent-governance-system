#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_cubic_chain_period17_unit_height_reduction.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_cubic_chain_period17_unit_height_reduction.py"
oracle_path="$frontier_dir/f17_cubic_chain_period17_unit_height_reduction_oracle.py"
module_path="$frontier_dir/f17_cubic_chain_period17_cyclotomic_module.py"
recurrence_path="$frontier_dir/f17_cubic_chain_period17_executed_recurrence.py"
adaptive_path="$frontier_dir/f17_cubic_chain_adaptive_gauge.py"
qualifier_path="$frontier_dir/qualify_f17_cubic_chain_period17_unit_height_reduction.sh"
expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_UNIT_HEIGHT_REDUCTION_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_UNIT_HEIGHT_REDUCTION_ORACLE_RESULTS.json"
provenance_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_UNIT_HEIGHT_REDUCTION_PROVENANCE.json"
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
  "$recurrence_path" \
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
rg -F 'import f17_cubic_chain_period17_executed_recurrence as recurrence' \
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

jq -cS 'del(.block_certificates[].characteristic, .cases[].boundary)' \
  "$result" >"$summary"
jq -cS . "$expected_path" >"$evidence_dir/result.expected.json"
cmp "$summary" "$evidence_dir/result.expected.json"

jq -cS . "$oracle_result" >"$oracle_summary"
jq -cS . "$oracle_expected_path" \
  >"$evidence_dir/oracle.expected.json"
cmp "$oracle_summary" "$evidence_dir/oracle.expected.json"

jq -e '
  .result == "PASS"
  and .classification_candidate == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level_candidate == "PACKAGE_SELF_REVIEW"
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and .coefficient_field == "Q_ZETA17"
  and .integral_carrier_ring == "Z_ZETA17"
  and .unit_generators == [2,3,4,5,6,7,8]
  and .unit_generator_count == 7
  and (.unit_generator_multiplicative_independence_certified|not)
  and .normalization_step_cap == 128
  and .tested_periods == [1,4,16,64,256]
  and .dense_direct_period_cap == 64
  and .all_recurrence_boundaries_equal
  and .all_applicable_dense_boundaries_equal
  and .all_cases_restore_exactly
  and .all_cases_reduce_carrier_payload
  and (.all_normalization_calls_below_step_cap|not)
  and .normalization_step_cap_hits_observed == 8
  and all(.cases[];
    .unnormalized_recurrence_boundary_equal
    and .carrier_payload_reduction_positive
    and .restored_exactly
    and .same_backing
    and .message_slots == 18
    and .message_integer_cells == 4896
    and .message_ledger_integer_cells == 126
    and .coefficient_register_integer_cells == 256
    and .coefficient_ledger_integer_cells == 7
    and .stats.normalization_calls == 36
    and .stats.basis_forward_block_applications == 16
    and .stats.basis_inverse_block_applications == 16
  )
  and .restoration_reuse_case.primary_restored_exactly
  and .restoration_reuse_case.reuse_restored_exactly
  and .restoration_reuse_case.same_original_backing
  and .restoration_reuse_case.fresh_restored_reuse_boundary_equal
  and .restoration_reuse_case.generation == 2
  and .restoration_reuse_case.lease == 2
  and all(.controls[]; .)
  and (.failure_atomic_restoration_established|not)
  and .matched_classical.identical_unit_normalized_recurrence_available
  and .matched_classical.identical_unit_ledger_available
  and (.matched_classical.comparison_establishes_advantage|not)
  and .resource_law.integer_width_counted
  and .resource_law.ledger_width_counted
  and .resource_law.normalization_live_named_message_vectors_minimum == 3
  and (.resource_law.normalization_live_message_vector_exact_peak_bounded|not)
  and (.resource_law.named_logical_cell_accounting_is_exact_total|not)
  and (.resource_law.whole_process_peak_bounded|not)
  and (.not_established|index("FIXED_INTEGER_WIDTH")) != null
  and (.not_established|index("DISTINCT_PHASE_RESOURCE")) != null
  and (.not_established|index("COMPUTATIONAL_ADVANTAGE")) != null
  and (.not_established|index("SMALL_WALL_CROSSING")) != null
  and (.not_established|index("FAILURE_ATOMIC_ROLLBACK_AFTER_REJECTED_INVERSE")) != null
  and (.terminal|not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and .classification_candidate == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level_candidate == "SEPARATE_REFERENCE_PARITY"
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and (.production_module_imported|not)
  and (.production_compiler_called|not)
  and (.production_normalizer_called|not)
  and (.production_recurrence_called|not)
  and (.production_inverse_called|not)
  and .unit_generator_inverse_identities_exact
  and .all_family_checks
  and .all_case_checks
  and .all_restoration_reuse_checks
  and .independent_coefficient_method
    == "SEQUENTIAL_MULTIPLICATION_BY_X_MOD_MONIC_Q16"
  and .independent_exact_unit_search_reexecution
  and .independent_exact_payload_and_width_reexecution
  and .fixed_slots_are_not_fixed_total_footprint
  and (.strict_local_minimum_for_every_call_established|not)
  and (.unit_generator_multiplicative_independence_certified|not)
  and .identical_compact_classical_normalizer
  and (.distinct_phase_resource_established|not)
  and (.computational_advantage|not)
  and (.small_wall_crossed|not)
  and (.failure_atomic_restoration_established|not)
  and [.case_checks[].periods] == [1,1,4,4,16,16,64,64,256,256]
  and all(.case_checks[]; .all_checks)
  and .case_checks[-2].exact_resource_tuple
    .normalized_carrier_payload_bits == 5767292
  and .case_checks[-2].exact_resource_tuple
    .normalized_coefficient_signed_bits == 8931
  and .case_checks[-1].exact_resource_tuple
    .normalized_carrier_payload_bits == 5935644
  and .case_checks[-1].exact_resource_tuple
    .normalized_coefficient_signed_bits == 8988
  and (.terminal|not)
' "$oracle_result" >/dev/null

echo "QUALIFIED_F17_CUBIC_CHAIN_PERIOD17_UNIT_HEIGHT_REDUCTION"
