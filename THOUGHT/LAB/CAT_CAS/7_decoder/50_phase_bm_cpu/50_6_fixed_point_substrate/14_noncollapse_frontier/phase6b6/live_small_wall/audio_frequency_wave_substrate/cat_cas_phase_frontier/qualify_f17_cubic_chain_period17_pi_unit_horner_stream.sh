#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_cubic_chain_period17_pi_unit_horner_stream.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_cubic_chain_period17_pi_unit_horner_stream.py"
oracle_path="$frontier_dir/f17_cubic_chain_period17_pi_unit_horner_stream_oracle.py"
deferred_path="$frontier_dir/f17_cubic_chain_period17_pi_unit_deferred_ledger_stream.py"
deferred_oracle_path="$frontier_dir/f17_cubic_chain_period17_pi_unit_deferred_ledger_stream_oracle.py"
exact_path="$frontier_dir/f17_cubic_chain_period17_pi_unit_exact_coordinate_descent.py"
exact_oracle_path="$frontier_dir/f17_cubic_chain_period17_pi_unit_exact_coordinate_descent_oracle.py"
balance_path="$frontier_dir/f17_cubic_chain_period17_pi_unit_embedding_balance.py"
balance_oracle_path="$frontier_dir/f17_cubic_chain_period17_pi_unit_embedding_balance_oracle.py"
cyclo_path="$frontier_dir/f17_cubic_chain_period17_cyclotomic_module.py"
recurrence_path="$frontier_dir/f17_cubic_chain_period17_executed_recurrence.py"
pi_path="$frontier_dir/f17_cubic_chain_period17_pi_content_recurrence.py"
pi_oracle_path="$frontier_dir/f17_cubic_chain_period17_pi_content_recurrence_oracle.py"
unit_path="$frontier_dir/f17_cubic_chain_period17_unit_height_reduction.py"
unit_oracle_path="$frontier_dir/f17_cubic_chain_period17_unit_height_reduction_oracle.py"
qualifier_path="$frontier_dir/qualify_f17_cubic_chain_period17_pi_unit_horner_stream.sh"
expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_PI_UNIT_HORNER_STREAM_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_PI_UNIT_HORNER_STREAM_ORACLE_RESULTS.json"
provenance_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_PI_UNIT_HORNER_STREAM_PROVENANCE.json"
review_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_PI_UNIT_HORNER_STREAM_INDEPENDENT_REVIEW.md"
result="$evidence_dir/result.full.json"
oracle_result="$evidence_dir/oracle.full.json"
summary="$evidence_dir/result.summary.json"
oracle_summary="$evidence_dir/oracle.summary.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in cmp jq nice rg sha256sum; do
  command -v "$tool" >/dev/null
done
test -x "$python"
jq empty "$expected_path" "$oracle_expected_path" "$provenance_path"

for sealed_path in \
  "$source_path" \
  "$oracle_path" \
  "$deferred_path" \
  "$deferred_oracle_path" \
  "$exact_path" \
  "$exact_oracle_path" \
  "$balance_path" \
  "$balance_oracle_path" \
  "$cyclo_path" \
  "$recurrence_path" \
  "$pi_path" \
  "$pi_oracle_path" \
  "$unit_path" \
  "$unit_oracle_path" \
  "$qualifier_path" \
  "$expected_path" \
  "$oracle_expected_path" \
  "$review_path"
do
  sealed_name=$(basename "$sealed_path")
  sealed_expected=$(jq -r --arg name "$sealed_name" \
    '.files[$name] // empty' "$provenance_path")
  test -n "$sealed_expected"
  test "$(sha256sum "$sealed_path" | cut -d' ' -f1)" = "$sealed_expected"
done

if rg -n '^[[:space:]]*(from|import)[[:space:]]+f17_cubic_chain_period17_pi_unit_horner_stream' \
  "$oracle_path"; then
  echo "independent oracle imports the production Horner successor" >&2
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
jq -cS . "$oracle_result" >"$oracle_summary"
jq -cS . "$expected_path" >"$evidence_dir/result.expected.json"
cmp "$summary" "$evidence_dir/result.expected.json"
jq -cS . "$oracle_expected_path" \
  >"$evidence_dir/oracle.expected.json"
cmp "$oracle_summary" "$evidence_dir/oracle.expected.json"

test "$(sha256sum "$result" | cut -d' ' -f1)" = \
  "$(jq -r '.reference_full_result_sha256' "$provenance_path")"
test "$(sha256sum "$oracle_result" | cut -d' ' -f1)" = \
  "$(jq -r '.reference_full_oracle_sha256' "$provenance_path")"

jq -e '
  .result == "PASS"
  and .classification_candidate == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level_candidate == "PACKAGE_SELF_REVIEW"
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and .tested_periods == [1,64]
  and .declared_exact_search_direction_count == 49
  and .carrier_resident_phase_vector_count == 1
  and .retained_inverse_history_bytes == 0
  and .public_topology_compilation_answer_independent
  and .all_raw_horner_boundaries_equal
  and .all_prior_raw_recurrence_boundaries_equal
  and .all_cases_restore_exactly
  and (.all_phase_named_payloads_beat_raw_horner|not)
  and all(.cases[];
    .restored_exactly
    and .same_backing
    and .raw_horner_boundary_equal
    and .prior_raw_recurrence_boundary_equal
    and .phase_stats.maximum_horner_named_checkpoint_vector_count == 6
    and .raw_horner_stats.maximum_named_checkpoint_vector_count == 5
    and .canonical_restored_state.all_payload_and_ledgers_zero
  )
  and all(.cases[] | select(.periods == 64);
    .phase_minus_raw_horner_named_payload_bits > 0
  )
  and .restoration_reuse_case.periods == 1
  and .restoration_reuse_case.primary_restored_exactly
  and .restoration_reuse_case.reuse_restored_exactly
  and .restoration_reuse_case.same_original_backing
  and .restoration_reuse_case.fresh_restored_reuse_boundary_equal
  and .restoration_reuse_case.canonical_restored_state
    .all_payload_and_ledgers_zero
  and all(.controls[]; .)
  and .matched_classical.matched_raw_horner_named_checkpoint_implemented
  and .matched_classical.identical_normalized_horner_available
  and .matched_classical.same_public_coefficients_operator_and_boundary
  and (.matched_classical.comparison_establishes_advantage|not)
  and .resource_law.public_coefficient_program_counted
  and .resource_law.named_seed_accumulator_raw_operator_term_and_next_checkpoints_counted
  and .resource_law.resident_plus_inverse_rematerialization_counted
  and .resource_law.search_power_trial_norm_energy_net_and_alignment_counted
  and .resource_law.raw_horner_named_vectors_and_coefficients_counted
  and (.resource_law.named_component_maxima_sum_is_simultaneous_peak|not)
  and (.resource_law.whole_process_peak_bounded|not)
  and (.not_established|index("DISTINCT_PHASE_RESOURCE")) != null
  and (.not_established|index("COMPUTATIONAL_ADVANTAGE")) != null
  and (.not_established|index("SMALL_WALL_CROSSING")) != null
  and (.not_established|index("MACHINE_ENFORCED_NO_SMUGGLE_OR_CATVM_CUSTODY")) != null
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
  and [.case_checks[].periods] == [1,1,64,64]
  and all(.case_checks[];
    .boundary_sha256_equal
    and .raw_horner_boundary_sha256_equal
    and .phase_resource_tuple_equal
    and .inverse_resource_tuple_equal
    and .raw_horner_resource_tuple_equal
    and .phase_named_checkpoint_equal
    and .named_search_temporary_sum_equal
    and .named_component_total_equal
    and .inverse_output_exactly_equal
  )
  and all(.restoration_checks[]; .)
  and all(.mutation_checks[]; .)
  and all(.production_scope_checks[]; .)
  and (.terminal|not)
' "$oracle_result" >/dev/null

echo "QUALIFIED_F17_CUBIC_CHAIN_PERIOD17_PI_UNIT_HORNER_STREAM"
