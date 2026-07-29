#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_cubic_chain_period17_pi_unit_embedding_balance.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_cubic_chain_period17_pi_unit_embedding_balance.py"
oracle_path="$frontier_dir/f17_cubic_chain_period17_pi_unit_embedding_balance_oracle.py"
cyclo_path="$frontier_dir/f17_cubic_chain_period17_cyclotomic_module.py"
recurrence_path="$frontier_dir/f17_cubic_chain_period17_executed_recurrence.py"
pi_path="$frontier_dir/f17_cubic_chain_period17_pi_content_recurrence.py"
pi_oracle_path="$frontier_dir/f17_cubic_chain_period17_pi_content_recurrence_oracle.py"
unit_path="$frontier_dir/f17_cubic_chain_period17_unit_height_reduction.py"
unit_oracle_path="$frontier_dir/f17_cubic_chain_period17_unit_height_reduction_oracle.py"
qualifier_path="$frontier_dir/qualify_f17_cubic_chain_period17_pi_unit_embedding_balance.sh"
expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_PI_UNIT_EMBEDDING_BALANCE_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_PI_UNIT_EMBEDDING_BALANCE_ORACLE_RESULTS.json"
provenance_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_PI_UNIT_EMBEDDING_BALANCE_PROVENANCE.json"
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
  "$cyclo_path" \
  "$recurrence_path" \
  "$pi_path" \
  "$pi_oracle_path" \
  "$unit_path" \
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

rg -F 'import f17_cubic_chain_period17_pi_content_recurrence as pi_content' \
  "$source_path" >/dev/null
rg -F 'import f17_cubic_chain_period17_unit_height_reduction as unit_reference' \
  "$source_path" >/dev/null
rg -F 'import f17_cubic_chain_period17_pi_content_recurrence_oracle as pi_reference' \
  "$oracle_path" >/dev/null
rg -F 'import f17_cubic_chain_period17_unit_height_reduction_oracle as reference' \
  "$oracle_path" >/dev/null
if rg -n '^[[:space:]]*(from|import)[[:space:]]+f17_cubic_chain_period17_pi_unit_embedding_balance' \
  "$oracle_path"; then
  echo "independent oracle imports the production successor" >&2
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
  and .tested_periods == [1,64]
  and .balance_step_cap == 128
  and (.unit_generator_multiplicative_independence_certified|not)
  and .all_raw_recurrence_boundaries_equal
  and .all_pi_content_boundaries_equal
  and .all_cases_restore_exactly
  and .all_cases_reduce_pi_content_payload
  and .all_declared_live_cases_reduce_pi_content_payload
  and (.all_cases_beat_raw_recurrence_payload|not)
  and .period64_both_families_remain_above_raw_recurrence
  and .period64_declared_live_both_families_remain_above_raw
  and all(.cases[];
    .restored_exactly
    and .same_backing
    and .raw_recurrence_boundary_equal
    and .pi_content_boundary_equal
    and .balanced_reduces_pi_content_payload
    and .balanced_declared_live_reduces_pi_content_payload
    and .balanced_stats.maximum_duplicate_state_payload_bits
      == .balanced_payload_bits
    and .balanced_declared_live_state_payload_bits
      == (2 * .balanced_payload_bits)
    and .balanced_stats.basis_operator_applications == 16
    and .balanced_stats.basis_operator_ring_multiply_accumulations == 4624
    and .balanced_stats.candidate_norm_ring_multiplications
      == .balanced_stats.balance_candidate_evaluations
  )
  and .cases[0].balanced_payload_bits == 601130
  and .cases[0].balanced_declared_live_state_payload_bits == 1202260
  and .cases[1].balanced_payload_bits == 603011
  and .cases[1].balanced_declared_live_state_payload_bits == 1206022
  and .cases[2].balanced_payload_bits == 5489878
  and .cases[2].balanced_declared_live_state_payload_bits == 10979756
  and .cases[3].balanced_payload_bits == 5699705
  and .cases[3].balanced_declared_live_state_payload_bits == 11399410
  and .cases[2].balanced_stats.balance_step_cap_hits == 53
  and .cases[3].balanced_stats.balance_step_cap_hits == 58
  and .restoration_reuse_case.periods == 1
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
  and all(.exact_embedding_controls[]; .)
  and .matched_classical.identical_pi_and_unit_balanced_recurrence_available
  and .matched_classical.identical_exact_trace_energy_objective_available
  and (.matched_classical.comparison_establishes_advantage|not)
  and .resource_law.duplicate_public_topology_state_payload_counted
  and .resource_law.declared_live_state_payload_uses_no_alias_discount
  and (.resource_law.whole_process_peak_bounded|not)
  and (.not_established|index("GLOBAL_CYCLOTOMIC_UNIT_OPTIMALITY")) != null
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
    and .all_exact_balance_and_resource_metrics_equal
    and .inverse_rematerialization_exact_resource_metrics_equal
    and .pi_payload_reduction_equal
    and .raw_payload_relation_equal
    and .declared_live_pi_payload_relation_equal
    and .declared_live_raw_payload_relation_equal
  )
  and all(.restoration_checks[]; .)
  and all(.mutation_checks[]; .)
  and all(.production_scope_checks[]; .)
  and (.terminal|not)
' "$oracle_result" >/dev/null

echo "QUALIFIED_F17_CUBIC_CHAIN_PERIOD17_PI_UNIT_EMBEDDING_BALANCE"
