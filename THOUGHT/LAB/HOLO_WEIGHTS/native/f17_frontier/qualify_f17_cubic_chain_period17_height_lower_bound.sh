#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_cubic_chain_period17_height_lower_bound.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_cubic_chain_period17_height_lower_bound.py"
oracle_path="$frontier_dir/f17_cubic_chain_period17_height_lower_bound_oracle.py"
reference_path="$frontier_dir/f17_cubic_chain_period17_unit_height_reduction_oracle.py"
module_path="$frontier_dir/f17_cubic_chain_period17_cyclotomic_module.py"
adaptive_path="$frontier_dir/f17_cubic_chain_adaptive_gauge.py"
qualifier_path="$frontier_dir/qualify_f17_cubic_chain_period17_height_lower_bound.sh"
expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_HEIGHT_LOWER_BOUND_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_HEIGHT_LOWER_BOUND_ORACLE_RESULTS.json"
provenance_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_HEIGHT_LOWER_BOUND_PROVENANCE.json"
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
  "$reference_path" \
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
rg -F \
  'import f17_cubic_chain_period17_unit_height_reduction_oracle as reference' \
  "$oracle_path" >/dev/null
if rg -n \
  '^[[:space:]]*(from|import)[[:space:]]+f17_cubic_chain_period17_(height_lower_bound|cyclotomic_module)' \
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

jq -cS 'del(.block_certificates[].characteristic)' \
  "$result" >"$summary"
jq -cS . "$expected_path" >"$evidence_dir/result.expected.json"
cmp "$summary" "$evidence_dir/result.expected.json"

jq -cS . "$oracle_result" >"$oracle_summary"
jq -cS . "$oracle_expected_path" \
  >"$evidence_dir/oracle.expected.json"
cmp "$oracle_summary" "$evidence_dir/oracle.expected.json"

jq -e '
  .result == "PASS"
  and .experiment
    == "EXACT_F17_PERIOD17_BOUNDARY_HEIGHT_LOWER_BOUND_FOR_LOSSLESS_EXACT_BOUNDARY_ENCODING"
  and .classification_candidate == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level_candidate == "PACKAGE_SELF_REVIEW"
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and .all_characteristic_identities_exact
  and .all_initial_boundaries_satisfy_bound
  and .all_coefficient_valuations_prove_induction
  and .all_normalized_cycles_have_nonzero_outputs
  and .infinitely_many_exact_nonzero_boundaries_certified
  and .infinitely_many_distinct_pi_valuations_certified
  and .fixed_finite_lossless_discrete_boundary_alphabet_rejected
  and .asymptotic_lossless_exact_boundary_encoding_horizon_lower_bound
    == "MAX_CODE_WIDTH_THROUGH_N_IS_OMEGA_LOG_N_WITHOUT_FREE_PERIOD_INDEX"
  and .lower_bound_scope
    == "WORST_CASE_INJECTIVE_EXACT_BOUNDARY_OR_VALUATION_ENCODING_THROUGH_PERIOD_N_DECODER_GETS_NO_FREE_N"
  and .exact_regauging_preserves_pi_valuation
  and .pi_content_factoring_transfers_valuation_to_exponent_ledger
  and (.cyclotomic_units_can_remove_pi_valuation|not)
  and .compact_exponent_ledger_upper_bound
    == "O_LOG_N_BITS_FOR_THE_PI_EXPONENT_ALONE"
  and .exact_cycle_nonzero_densities.primary
    == {"denominator":1632,"numerator":1555}
  and .exact_cycle_nonzero_densities.reuse
    == {"denominator":7344,"numerator":6913}
  and .families[0].family == "PRIMARY"
  and .families[0].cycle.prefix_length == 3
  and .families[0].cycle.cycle_length == 1632
  and .families[0].cycle.cycle_nonzero_outputs == 1555
  and .families[1].family == "REUSE"
  and .families[1].cycle.prefix_length == 0
  and .families[1].cycle.cycle_length == 14688
  and .families[1].cycle.cycle_nonzero_outputs == 13826
  and all(.families[];
    .characteristic_identity_exact
    and .period17_recurrence_residue_equal
    and .coefficient_valuations_prove_induction
    and .cycle.cycle_has_nonzero_output
    and .cycle.retained_cycle_dictionary_entries == 0
  )
  and .restoration_reuse_case.primary_restored_exactly
  and .restoration_reuse_case.reuse_restored_exactly
  and .restoration_reuse_case.same_original_backing
  and .restoration_reuse_case.fresh_restored_reuse_boundary_equal
  and .restoration_reuse_case.all_messages_zero
  and .restoration_reuse_case.generation == 2
  and .restoration_reuse_case.lease == 2
  and (.restoration_reuse_case.full_carrier_object_state_equal|not)
  and (.restoration_reuse_case.repeated_use_metadata_width_bounded|not)
  and .restoration_reuse_case.retained_inverse_history_bytes == 0
  and .restoration_reuse_case.baseline_reload_bytes == 0
  and .matched_classical.identical_characteristic_recurrence_available
  and .matched_classical.identical_pi_valuation_certificate_available
  and (.matched_classical.comparison_establishes_advantage|not)
  and .resource_law.accounting_scope
    == "NAMED_LOGICAL_COMPONENTS_NOT_AN_EXACT_TEMPORARY_OR_WHOLE_PROCESS_PEAK"
  and .resource_law.restoration_carrier_integer_cells == 1632
  and (.resource_law.exact_temporary_peak_bounded|not)
  and (.resource_law.whole_process_peak_bounded|not)
  and (.not_established|index("LINEAR_BIT_LOWER_BOUND")) != null
  and (.not_established|index("GENERIC_MACHINE_STATE_MEMORY_LOWER_BOUND")) != null
  and (.not_established|index("POINTWISE_OMEGA_LOG_N_BITS_AT_EVERY_PERIOD")) != null
  and (.not_established|index("DISTINCT_PHASE_RESOURCE")) != null
  and (.not_established|index("COMPUTATIONAL_ADVANTAGE")) != null
  and (.not_established|index("SMALL_WALL_CROSSING")) != null
  and (.terminal|not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "SEPARATE_REFERENCE_PARITY"
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and (.oracle_imports_production_module|not)
  and .oracle_cycle_algorithm_distinct
  and all(.family_checks[]; all(.[]; .))
  and all(.mutation_checks[]; .)
  and all(.restoration_checks[]; .)
  and all(.production_gate_checks[]; .)
  and .exact_cycle_nonzero_densities.primary
    == {"denominator":1632,"numerator":1555}
  and .exact_cycle_nonzero_densities.reuse
    == {"denominator":7344,"numerator":6913}
  and (.not_established|index("DISTINCT_PHASE_RESOURCE")) != null
  and (.not_established|index("COMPUTATIONAL_ADVANTAGE")) != null
  and (.not_established|index("SMALL_WALL_CROSSING")) != null
  and (.terminal|not)
' "$oracle_result" >/dev/null

echo "QUALIFIED_F17_CUBIC_CHAIN_PERIOD17_HEIGHT_LOWER_BOUND"
