#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_cubic_chain_period17_pi_unit_lattice_center.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_cubic_chain_period17_pi_unit_lattice_center.py"
oracle_path="$frontier_dir/f17_cubic_chain_period17_pi_unit_lattice_center_oracle.py"
cyclo_path="$frontier_dir/f17_cubic_chain_period17_cyclotomic_module.py"
recurrence_path="$frontier_dir/f17_cubic_chain_period17_executed_recurrence.py"
pi_path="$frontier_dir/f17_cubic_chain_period17_pi_content_recurrence.py"
pi_oracle_path="$frontier_dir/f17_cubic_chain_period17_pi_content_recurrence_oracle.py"
unit_path="$frontier_dir/f17_cubic_chain_period17_unit_height_reduction.py"
unit_oracle_path="$frontier_dir/f17_cubic_chain_period17_unit_height_reduction_oracle.py"
prior_oracle_path="$frontier_dir/f17_cubic_chain_period17_pi_unit_embedding_balance_oracle.py"
qualifier_path="$frontier_dir/qualify_f17_cubic_chain_period17_pi_unit_lattice_center.sh"
expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_PI_UNIT_LATTICE_CENTER_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_PI_UNIT_LATTICE_CENTER_ORACLE_RESULTS.json"
provenance_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_PI_UNIT_LATTICE_CENTER_PROVENANCE.json"
result="$evidence_dir/result.full.json"
oracle_result="$evidence_dir/oracle.full.json"
summary="$evidence_dir/result.summary.json"
oracle_summary="$evidence_dir/oracle.summary.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in jq nice rg; do
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
  "$prior_oracle_path" \
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

if rg -n '^[[:space:]]*(from|import)[[:space:]]+f17_cubic_chain_period17_pi_unit_lattice_center' \
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
jq -cS . "$oracle_result" >"$oracle_summary"
jq -cS . "$expected_path" >"$evidence_dir/result.expected.json"
cmp "$summary" "$evidence_dir/result.expected.json"
jq -cS . "$oracle_expected_path" \
  >"$evidence_dir/oracle.expected.json"
cmp "$oracle_summary" "$evidence_dir/oracle.expected.json"

jq -e '
  .result == "PASS"
  and .classification_candidate == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level_candidate == "PACKAGE_SELF_REVIEW"
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and .tested_periods == [1,64]
  and .embedding_proposal_precision_bits == 65536
  and (.embedding_proposal_is_exact|not)
  and (.embedding_proposal_is_certified_closest_vector|not)
  and .exact_trace_acceptance_controls_carrier_update
  and .all_raw_recurrence_boundaries_equal
  and .all_pi_content_boundaries_equal
  and .all_cases_restore_exactly
  and .all_cases_reduce_pi_content_payload
  and .all_cases_beat_raw_recurrence_payload
  and .all_declared_live_cases_reduce_pi_content_payload
  and .all_declared_live_cases_beat_raw_recurrence_payload
  and .all_fixed_precision_embedding_evaluations_positive
  and .all_named_component_maxima_sums_remain_above_raw
  and all(.cases[];
    .restored_exactly
    and .same_backing
    and .raw_recurrence_boundary_equal
    and .pi_content_boundary_equal
    and .balanced_reduces_pi_content_payload
    and .balanced_beats_raw_recurrence_payload
    and .balanced_declared_live_reduces_pi_content_payload
    and .balanced_declared_live_beats_raw_recurrence_payload
    and .balanced_stats.maximum_duplicate_state_payload_bits
      == .balanced_payload_bits
    and .balanced_declared_live_state_payload_bits
      == (2 * .balanced_payload_bits)
    and .balanced_stats.proposal_commitment_updates
      == .balanced_stats.lattice_center_proposals
    and .balanced_stats.basis_operator_applications == 16
    and .balanced_stats.basis_operator_ring_multiply_accumulations == 4624
  )
  and all(.cases[];
    (.named_component_maxima_sum_beats_raw_recurrence_payload|not)
  )
  and .restoration_reuse_case.primary_restored_exactly
  and .restoration_reuse_case.reuse_restored_exactly
  and .restoration_reuse_case.same_original_backing
  and .restoration_reuse_case.fresh_restored_reuse_boundary_equal
  and .restoration_reuse_case.canonical_restored_state
    .all_payload_and_ledgers_zero
  and .restoration_reuse_case.generation == 2
  and .restoration_reuse_case.lease == 2
  and all(.controls[]; .)
  and all(.exact_embedding_controls[]; .)
  and .matched_classical
    .identical_fixed_precision_proposal_recurrence_available
  and (.matched_classical.comparison_establishes_advantage|not)
  and (.resource_law.whole_process_peak_bounded|not)
  and (.not_established|index("EXACT_CLOSEST_VECTOR_SOLUTION")) != null
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
  and .oracle_center_solver == "FLOAT64_NORMAL_EQUATION_SOLVE"
  and .production_center_solver == "FLOAT64_LEAST_SQUARES"
  and .independent_exact_integer_precision_tuple_reexecution
  and all(.family_checks[]; all(.[]; .))
  and [.case_checks[].periods] == [1,1,64,64]
  and all(.case_checks[];
    .boundary_sha256_equal
    and .exact_integer_precision_and_resource_tuple_equal
    and .inverse_exact_integer_precision_and_resource_tuple_equal
    and .solver_residual_within_declared_tolerance
    and .proposal_transcript_equal
    and .named_component_maxima_sum_equal
    and .resident_payload_relation_equal
    and .declared_live_payload_relation_equal
    and .named_component_raw_relation_equal
  )
  and all(.restoration_checks[]; .)
  and all(.mutation_checks[]; .)
  and all(.production_scope_checks[]; .)
  and (.terminal|not)
' "$oracle_result" >/dev/null

echo "QUALIFIED_F17_CUBIC_CHAIN_PERIOD17_PI_UNIT_LATTICE_CENTER"
