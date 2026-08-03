#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_cubic_chain_period17_krylov.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_cubic_chain_period17_krylov.py"
oracle_path="$frontier_dir/f17_cubic_chain_period17_krylov_oracle.py"
dependency_path="$frontier_dir/f17_cubic_chain_adaptive_gauge.py"
qualifier_path="$frontier_dir/qualify_f17_cubic_chain_period17_krylov.sh"
expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_KRYLOV_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_KRYLOV_ORACLE_RESULTS.json"
provenance_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_KRYLOV_PROVENANCE.json"
result="$evidence_dir/result.json"
replay="$evidence_dir/replay.json"
oracle_result="$evidence_dir/oracle.json"
summary="$evidence_dir/summary.json"
oracle_summary="$evidence_dir/oracle_summary.json"

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
  "$dependency_path" \
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

rg -F 'import f17_cubic_chain_adaptive_gauge as adaptive' \
  "$source_path" >/dev/null
if rg -n '^[[:space:]]*(from|import)[[:space:]]+f17_cubic_chain_' \
  "$oracle_path"; then
  echo "independent oracle imports a production module" >&2
  exit 1
fi

"$python" -m py_compile "$source_path" "$oracle_path"
nice -n 10 "$python" -X dev "$source_path" \
  >"$result" 2>"$evidence_dir/result.stderr"
nice -n 10 "$python" -X dev "$source_path" \
  >"$replay" 2>"$evidence_dir/replay.stderr"
test ! -s "$evidence_dir/result.stderr"
test ! -s "$evidence_dir/replay.stderr"
cmp "$result" "$replay"

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
  period,
  message_dimension,
  krylov_moduli,
  block_program_sha256,
  krylov,
  modular_dimensions_stable_across_tested_primes,
  observed_modular_krylov_dimensions,
  strictly_smaller_modular_images_observed,
  exact_rational_krylov_dimension_lower_bounds,
  exact_rational_krylov_dimensions_established,
  exact_rational_krylov_reduction_established,
  projective_cases:[.projective_cases[]|{
    periods,
    edges,
    nodes,
    primary:(.primary|del(
      .public_program_descriptor_bytes,
      .streaming_accounting,
      .metric_verification_canonical_quotient_integer_cells,
      .metric_verification_canonical_semantic_integer_cells,
      .metric_verification_combined_peak_integer_cells,
      .encoded_message_bytes
    )),
    reuse:(.reuse|del(
      .public_program_descriptor_bytes,
      .streaming_accounting,
      .metric_verification_canonical_quotient_integer_cells,
      .metric_verification_canonical_semantic_integer_cells,
      .metric_verification_combined_peak_integer_cells,
      .encoded_message_bytes
    ))
  }],
  all_quotient_coefficient_gcds_one,
  restoration_case:(.restoration_case|del(
    .message_slots,
    .message_integer_cells,
    .message_pivot_metadata_bits,
    .primary_program_descriptor_bytes,
    .reuse_program_descriptor_bytes,
    .concurrent_program_descriptor_bytes,
    .primary_stats,
    .reuse_stats,
    .fresh_reuse_stats
  )),
  matched_classical,
  observed_law,
  resource_law:(.resource_law|del(
    .projective_streaming_message_peak_integer_cells,
    .projective_streaming_pivot_metadata_peak_bits,
    .projective_streaming_scale_register_peak_bits,
    .projective_metric_verification_combined_peak_integer_cells,
    .maximum_projective_encoded_message_bytes,
    .maximum_projective_program_descriptor_bytes,
    .restoration_carrier_message_integer_cells,
    .restoration_carrier_pivot_metadata_bits,
    .restoration_verification_two_carrier_integer_cells,
    .temporary_seed_message_integer_cells,
    .temporary_seed_pivot_metadata_bits,
    .temporary_inverse_expected_message_integer_cells,
    .temporary_inverse_expected_pivot_metadata_bits,
    .restoration_concurrent_program_descriptor_bytes,
    .krylov_combined_explicit_peak_field_cells,
    .restoration_verification_transaction_count,
    .restoration_total_forward_transfer_applications,
    .restoration_total_inverse_transfer_applications,
    .restoration_total_transfer_scalar_accumulations,
    .restoration_total_projection_scalar_accumulations,
    .accounting_scope
  )),
  not_established,
  next_obstruction,
  terminal
}' "$result" >"$summary"

jq -cS '{
  result,
  oracle,
  production_module_imported,
  production_compiler_called,
  production_transfer_called,
  production_gauge_selector_called,
  production_inverse_called,
  descriptor_checks,
  rank_checks,
  projective_checks,
  all_public_descriptors_equal,
  all_modular_krylov_dimensions_equal,
  all_exact_projective_metrics_equal,
  exact_rational_krylov_dimensions_established,
  identical_compact_classical_block_map,
  distinct_phase_resource_established,
  computational_advantage,
  small_wall_crossed,
  terminal
}' "$oracle_result" >"$oracle_summary"

jq -cS . "$expected_path" >"$evidence_dir/expected.normalized.json"
jq -cS . "$oracle_expected_path" \
  >"$evidence_dir/oracle_expected.normalized.json"
cmp "$summary" "$evidence_dir/expected.normalized.json"
cmp "$oracle_summary" "$evidence_dir/oracle_expected.normalized.json"

jq -e '
  .result == "PASS"
  and .classification_candidate
    == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level_candidate == "SEPARATE_REFERENCE_PARITY"
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and .period == 17
  and .message_dimension == 272
  and .krylov_moduli == [41,73]
  and .modular_dimensions_stable_across_tested_primes
  and .observed_modular_krylov_dimensions
    == {"primary":241,"reuse":256}
  and .strictly_smaller_modular_images_observed
  and .exact_rational_krylov_dimension_lower_bounds
    == {"primary":241,"reuse":256}
  and (.exact_rational_krylov_dimensions_established|not)
  and (.exact_rational_krylov_reduction_established|not)
  and [.projective_cases[].periods] == [1,2,4,8]
  and [.projective_cases[].primary.adaptive_total_payload_bits]
    == [3462,6263,13205,26164]
  and [.projective_cases[].reuse.adaptive_total_payload_bits]
    == [3629,6567,13949,27689]
  and [.projective_cases[].primary
    .maximum_quotient_coefficient_signed_bits]
      == [15,25,52,99]
  and [.projective_cases[].reuse
    .maximum_quotient_coefficient_signed_bits]
      == [16,27,54,104]
  and .all_quotient_coefficient_gcds_one
  and .restoration_case.periods == 4
  and .restoration_case.primary_restored_exactly
  and .restoration_case.reuse_restored_exactly
  and .restoration_case.same_original_backing
  and .restoration_case.fresh_restored_reuse_boundary_equal
  and .restoration_case.canonical_restored_state.all_messages_zero
  and .restoration_case.message_slots == 9
  and .restoration_case.message_integer_cells == 2448
  and .restoration_case.retained_inverse_history_bytes == 0
  and .restoration_case.baseline_reload_bytes == 0
  and .matched_classical.identical_period_block_linear_map
  and .matched_classical.identical_modular_krylov_images
  and (.matched_classical
    .exact_recurrence_order_lower_than_272_established|not)
  and (.matched_classical
    .strongest_family_specific_implementation_established|not)
  and .resource_law.krylov_basis_peak_field_cells == 69632
  and .resource_law.krylov_combined_explicit_peak_field_cells == 70176
  and .resource_law
    .projective_streaming_message_peak_integer_cells == 544
  and .resource_law
    .projective_streaming_pivot_metadata_peak_bits == 170
  and .resource_law
    .projective_streaming_scale_register_peak_bits > 0
  and .resource_law
    .projective_metric_verification_combined_peak_integer_cells == 544
  and .resource_law.maximum_projective_encoded_message_bytes > 0
  and .resource_law
    .restoration_carrier_message_integer_cells == 2448
  and .resource_law
    .restoration_verification_two_carrier_integer_cells == 4896
  and .resource_law.temporary_seed_message_integer_cells == 272
  and .resource_law.temporary_seed_pivot_metadata_bits == 85
  and .resource_law
    .temporary_inverse_expected_message_integer_cells == 272
  and .resource_law
    .temporary_inverse_expected_pivot_metadata_bits == 85
  and .resource_law.restoration_verification_transaction_count == 3
  and .resource_law.restoration_total_forward_transfer_applications > 0
  and .resource_law.restoration_total_inverse_transfer_applications > 0
  and .resource_law.restoration_total_transfer_scalar_accumulations > 0
  and .resource_law.restoration_total_projection_scalar_accumulations > 0
  and (.resource_law.dense_block_materialized|not)
  and (.resource_law.assignment_table_materialized|not)
  and (.resource_law.relation_table_materialized|not)
  and (.terminal|not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and (.production_module_imported|not)
  and (.production_compiler_called|not)
  and (.production_transfer_called|not)
  and (.production_gauge_selector_called|not)
  and (.production_inverse_called|not)
  and .rank_checks
    == {"primary":{"41":241,"73":241},
        "reuse":{"41":256,"73":256}}
  and .all_public_descriptors_equal
  and .all_modular_krylov_dimensions_equal
  and .all_exact_projective_metrics_equal
  and (.exact_rational_krylov_dimensions_established|not)
  and .identical_compact_classical_block_map
  and .resource_law.modular_seed_field_cells == 272
  and .resource_law.modular_krylov_basis_peak_field_cells == 69632
  and .resource_law
    .modular_combined_explicit_peak_field_cells == 70720
  and .resource_law
    .exact_projective_transfer_two_message_peak_integer_cells == 544
  and .resource_law
    .exact_projective_final_fixed_adaptive_semantic_peak_integer_cells
      == 816
  and .resource_law.maximum_exact_projective_encoded_message_bytes > 0
  and .resource_law.gauge_candidate_metadata_integer_fields == 34
  and .resource_law
    .gauge_candidate_retained_coefficient_integer_cells == 32
  and .resource_law.gauge_candidate_redundant_row_integer_cells == 17
  and .resource_law.production_result_file_bytes > 0
  and (.distinct_phase_resource_established|not)
  and (.computational_advantage|not)
  and (.small_wall_crossed|not)
  and (.terminal|not)
' "$oracle_result" >/dev/null

echo "F17 cubic-chain period-17 Krylov qualification passed"
