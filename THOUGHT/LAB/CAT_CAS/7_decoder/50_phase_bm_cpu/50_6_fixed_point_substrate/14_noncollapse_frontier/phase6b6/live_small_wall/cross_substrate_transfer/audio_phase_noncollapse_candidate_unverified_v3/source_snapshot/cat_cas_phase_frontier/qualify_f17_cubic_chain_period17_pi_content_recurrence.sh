#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_cubic_chain_period17_pi_content_recurrence.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_cubic_chain_period17_pi_content_recurrence.py"
oracle_path="$frontier_dir/f17_cubic_chain_period17_pi_content_recurrence_oracle.py"
module_path="$frontier_dir/f17_cubic_chain_period17_cyclotomic_module.py"
recurrence_path="$frontier_dir/f17_cubic_chain_period17_executed_recurrence.py"
height_oracle_path="$frontier_dir/f17_cubic_chain_period17_height_lower_bound_oracle.py"
unit_oracle_path="$frontier_dir/f17_cubic_chain_period17_unit_height_reduction_oracle.py"
qualifier_path="$frontier_dir/qualify_f17_cubic_chain_period17_pi_content_recurrence.sh"
expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_PI_CONTENT_RECURRENCE_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_PI_CONTENT_RECURRENCE_ORACLE_RESULTS.json"
provenance_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_PI_CONTENT_RECURRENCE_PROVENANCE.json"
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
  "$height_oracle_path" \
  "$unit_oracle_path" \
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
rg -F 'import f17_cubic_chain_period17_height_lower_bound_oracle as height_reference' \
  "$oracle_path" >/dev/null
rg -F 'import f17_cubic_chain_period17_unit_height_reduction_oracle as reference' \
  "$oracle_path" >/dev/null
if rg -n '^[[:space:]]*(from|import)[[:space:]]+f17_cubic_chain_period17_pi_content_recurrence' \
  "$oracle_path"; then
  echo "independent oracle imports the production module" >&2
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
  and .uniformizer == "PI_EQUALS_1_MINUS_ZETA17"
  and .tested_periods == [1,4,16,64,256]
  and (.accepted_path_materializes_raw_recurrence_coefficients|not)
  and .accepted_path_factors_pi_content_during_polynomial_arithmetic
  and .all_raw_recurrence_boundaries_equal
  and .all_cases_restore_exactly
  and (.all_cases_reduce_carrier_payload|not)
  and .all_cases_worsen_carrier_payload
  and .all_cases_worsen_carrier_payload_in_pi_integral_basis
  and all(.cases[];
    .raw_recurrence_boundary_equal
    and .restored_exactly
    and .same_backing
    and (.carrier_payload_reduction_positive|not)
    and .message_slots == 18
    and .message_residual_integer_cells == 4896
    and .message_pi_exponent_ledger_cells == 18
    and .coefficient_residual_integer_cells == 256
    and .coefficient_pi_exponent_ledger_cells == 16
  )
  and .restoration_reuse_case.primary_restored_exactly
  and .restoration_reuse_case.reuse_restored_exactly
  and .restoration_reuse_case.same_original_backing
  and .restoration_reuse_case.fresh_restored_reuse_boundary_equal
  and .restoration_reuse_case.canonical_restored_state
    .all_payload_and_ledgers_zero
  and (.restoration_reuse_case.full_carrier_object_state_equal|not)
  and (.restoration_reuse_case.repeated_use_metadata_width_bounded|not)
  and .restoration_reuse_case.generation == 2
  and .restoration_reuse_case.lease == 2
  and all(.controls[]; .)
  and .pi_integral_basis_transform_roundtrip_exact
  and (.failure_atomic_restoration_established|not)
  and .matched_classical.identical_pi_content_normalized_recurrence_available
  and .matched_classical.identical_residual_and_exponent_ledger_available
  and (.matched_classical.comparison_establishes_advantage|not)
  and .resource_law.final_projection_materializes_exact_boundary
  and .resource_law.verification_baseline_materializes_raw_coefficients
  and (.resource_law.allocator_peak_bounded|not)
  and (.resource_law.whole_process_peak_bounded|not)
  and (.not_established|index("INTRINSIC_LOWER_BOUND_ACROSS_ALL_INTEGRAL_BASES")) != null
  and (.not_established|index("DISTINCT_PHASE_RESOURCE")) != null
  and (.not_established|index("COMPUTATIONAL_ADVANTAGE")) != null
  and (.not_established|index("SMALL_WALL_CROSSING")) != null
  and (.not_established|index("FAILURE_ATOMIC_ROLLBACK_AFTER_REJECTED_INVERSE")) != null
  and (.terminal|not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "SEPARATE_REFERENCE_PARITY"
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and (.oracle_imports_production_module|not)
  and .oracle_coefficient_method
    == "SEQUENTIAL_MULTIPLICATION_BY_X_MOD_Q"
  and .production_coefficient_method
    == "BINARY_POLYNOMIAL_POWERING_MOD_Q"
  and all(.family_checks[]; all(.[]; .))
  and [.case_checks[].periods] == [1,1,4,4,16,16,64,64,256,256]
  and all(.case_checks[];
    .sequential_scaled_coefficient_decompositions_equal
    and .boundary_sha256_equal
    and .boundary_payload_bits_equal
    and .boundary_pi_valuation_equal
    and .all_carrier_metrics_equal
    and .raw_baseline_payload_equal
    and .raw_baseline_width_equal
    and .zeta_basis_payload_worsens
    and .pi_basis_payload_worsens
  )
  and all(.restoration_checks[]; .)
  and all(.mutation_checks[]; .)
  and all(.production_scope_checks[]; .)
  and .case_checks[-2].exact_carrier_metrics
    .maximum_carrier_payload_bits == 27417447
  and .case_checks[-2].exact_carrier_metrics
    .maximum_pi_integral_basis_carrier_payload_bits == 27460355
  and .case_checks[-2].exact_carrier_metrics
    .maximum_residual_signed_bits == 42337
  and .case_checks[-2].exact_carrier_metrics
    .maximum_ledger_exponent_signed_bits == 16
  and .case_checks[-1].exact_carrier_metrics
    .maximum_carrier_payload_bits == 28372693
  and .case_checks[-1].exact_carrier_metrics
    .maximum_pi_integral_basis_carrier_payload_bits == 28415308
  and .case_checks[-1].exact_carrier_metrics
    .maximum_residual_signed_bits == 42774
  and .case_checks[-1].exact_carrier_metrics
    .maximum_ledger_exponent_signed_bits == 16
  and (.terminal|not)
' "$oracle_result" >/dev/null

echo "QUALIFIED_F17_CUBIC_CHAIN_PERIOD17_PI_CONTENT_RECURRENCE"
