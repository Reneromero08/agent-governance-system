#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_cubic_chain_period17_streamed_real_autocorrelation.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_cubic_chain_period17_streamed_real_autocorrelation.py"
oracle_path="$frontier_dir/f17_cubic_chain_period17_streamed_real_autocorrelation_oracle.py"
predecessor_path="$frontier_dir/f17_cubic_chain_period17_real_subfield_horner.py"
predecessor_oracle_path="$frontier_dir/f17_cubic_chain_period17_real_subfield_horner_oracle.py"
qualifier_path="$frontier_dir/qualify_f17_cubic_chain_period17_streamed_real_autocorrelation.sh"
expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_STREAMED_REAL_AUTOCORRELATION_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_STREAMED_REAL_AUTOCORRELATION_ORACLE_RESULTS.json"
provenance_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_STREAMED_REAL_AUTOCORRELATION_PROVENANCE.json"
review_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_STREAMED_REAL_AUTOCORRELATION_INDEPENDENT_REVIEW.md"
result="$evidence_dir/result.full.json"
oracle_result="$evidence_dir/oracle.full.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in cmp git jq nice rg sha256sum; do
  command -v "$tool" >/dev/null
done
test -x "$python"
jq empty "$expected_path" "$oracle_expected_path" "$provenance_path"

scientific_parent=$(jq -r '.scientific_source_parent' "$provenance_path")
test "$scientific_parent" = "319118cc4958304ee5c24d7a6dc9602adb277b1d"
git -C "$repo" cat-file -e "$scientific_parent^{commit}"
git -C "$repo" merge-base --is-ancestor "$scientific_parent" HEAD

for sealed_path in \
  "$source_path" \
  "$oracle_path" \
  "$predecessor_path" \
  "$predecessor_oracle_path" \
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

if rg -n '^[[:space:]]*(from|import)[[:space:]]+f17_cubic_chain_period17_streamed_real_autocorrelation([[:space:]]|$)' \
  "$oracle_path"; then
  echo "independent oracle imports production streamed successor" >&2
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
  "$result" >"$evidence_dir/result.summary.json"
jq -cS . "$oracle_result" >"$evidence_dir/oracle.summary.json"
cmp "$evidence_dir/result.summary.json" "$expected_path"
cmp "$evidence_dir/oracle.summary.json" "$oracle_expected_path"

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
  and (.full_cyclotomic_aggregate_norm_constructed|not)
  and .full_cyclotomic_per_element_norm_products_remain
  and .all_streamed_norm_term_counts_exact
  and .all_full_aggregate_norms_absent
  and .all_raw_horner_boundaries_equal
  and .all_prior_raw_recurrence_boundaries_equal
  and .all_cases_restore_exactly
  and (.all_phase_named_payloads_beat_raw_horner|not)
  and [.cases[].phase_named_component_maxima_sum_bits]
    == [96088,103778,3466223,3840024]
  and [.cases[].raw_horner_named_checkpoint_payload_bits]
    == [10005,10097,2790766,2901994]
  and [.cases[].phase_stats.streamed_real_norm_calls]
    == [5,5,93,98]
  and [.cases[].phase_stats.streamed_real_norm_input_cells]
    == [69,69,1101,1154]
  and [.cases[].phase_stats.streamed_real_norm_terms]
    == [69,69,1101,1154]
  and [.cases[].phase_stats.streamed_real_norm_singleton_calls]
    == [1,1,30,32]
  and [.cases[].phase_stats.streamed_real_norm_full_carrier_calls]
    == [4,4,63,66]
  and all(.cases[];
    .phase_stats.streamed_real_norm_unexpected_width_calls == 0
    and .restored_exactly
    and .same_backing
    and .raw_horner_boundary_equal
    and .prior_raw_recurrence_boundary_equal
    and .phase_minus_raw_horner_named_payload_bits > 0
  )
  and .restoration_reuse_case.primary_restored_exactly
  and .restoration_reuse_case.reuse_restored_exactly
  and .restoration_reuse_case.same_original_backing
  and .restoration_reuse_case.fresh_restored_reuse_boundary_equal
  and all(.carrier_controls[]; .)
  and all(.algebra_controls[]; .)
  and .matched_classical.identical_streamed_real_subfield_norm_available
  and (.matched_classical.comparison_establishes_advantage|not)
  and .resource_law.streamed_full_products_counted
  and .resource_law.streamed_real_terms_counted
  and .resource_law.streamed_real_accumulator_counted
  and .resource_law.streamed_conversion_live_payload_counted
  and .resource_law.streamed_addition_live_payload_counted
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
  and (.oracle_imports_production_streamed_module|not)
  and (.oracle_full_cyclotomic_aggregate_norm_constructed|not)
  and all(.family_checks[]; all(.[]; .))
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
  and all(.algebra_checks[]; .)
  and all(.compiled_table_checks[]; .)
  and all(.production_scope_checks[]; .)
  and all(.streamed_shape_checks[]; .)
  and (.terminal|not)
' "$oracle_result" >/dev/null

echo "QUALIFIED_F17_CUBIC_CHAIN_PERIOD17_STREAMED_REAL_AUTOCORRELATION"
