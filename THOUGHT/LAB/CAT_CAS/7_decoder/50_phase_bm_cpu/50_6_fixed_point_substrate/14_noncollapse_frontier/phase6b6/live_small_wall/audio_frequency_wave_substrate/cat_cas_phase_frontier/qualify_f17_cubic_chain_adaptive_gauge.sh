#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_cubic_chain_adaptive_gauge.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_cubic_chain_adaptive_gauge.py"
oracle_path="$frontier_dir/f17_cubic_chain_adaptive_gauge_oracle.py"
qualifier_path="$frontier_dir/qualify_f17_cubic_chain_adaptive_gauge.sh"
expected_path="$frontier_dir/F17_CUBIC_CHAIN_ADAPTIVE_GAUGE_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_CUBIC_CHAIN_ADAPTIVE_GAUGE_ORACLE_RESULTS.json"
provenance_path="$frontier_dir/F17_CUBIC_CHAIN_ADAPTIVE_GAUGE_PROVENANCE.json"
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
if rg -n '\.clear\(' "$source_path"; then
  echo "destructive clear is forbidden in the accepted inverse path" >&2
  exit 1
fi
rg -F 'subtract_message_exact(target, expected)' "$source_path" \
  >/dev/null
rg -F 'subtract_message_exact(carrier.messages[0], seed)' "$source_path" \
  >/dev/null

for sealed_path in \
  "$source_path" \
  "$oracle_path" \
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
  restoration_mechanism,
  representation,
  tested_nodes:[.cases[].nodes],
  primary_program_sha256:[.cases[].primary_program_sha256],
  reuse_program_sha256:[.cases[].reuse_program_sha256],
  primary_program_descriptor_bytes:
    [.cases[].primary_program_descriptor_bytes],
  reuse_program_descriptor_bytes:
    [.cases[].reuse_program_descriptor_bytes],
  primary_boundary_descriptor_bytes:
    [.cases[].primary_boundary_descriptor_bytes],
  reuse_boundary_descriptor_bytes:
    [.cases[].reuse_boundary_descriptor_bytes],
  primary_boundary_content_17_exponents:
    [.cases[].primary_boundary.content_17_exponent],
  reuse_boundary_content_17_exponents:
    [.cases[].reuse_boundary.content_17_exponent],
  primary_effective_sqrt_powers:
    [.cases[].primary_boundary
      .effective_normalization_denominator_sqrt_power],
  reuse_effective_sqrt_powers:
    [.cases[].reuse_boundary
      .effective_normalization_denominator_sqrt_power],
  primary_boundary_adaptive_payload_bits:
    [.cases[].primary_boundary.adaptive_payload_bits],
  reuse_boundary_adaptive_payload_bits:
    [.cases[].reuse_boundary.adaptive_payload_bits],
  primary_boundary_unfactored_payload_bits:
    [.cases[].primary_boundary.unfactored_canonical_payload_bits],
  reuse_boundary_unfactored_payload_bits:
    [.cases[].reuse_boundary.unfactored_canonical_payload_bits],
  message_slots:[.cases[].message_slots],
  message_integer_cells:[.cases[].message_integer_cells],
  message_pivot_metadata_bits:[.cases[].message_pivot_metadata_bits],
  pebble_forward_applications:[.cases[].pebble_forward_applications],
  all_restored_exactly:([.cases[].restored_exactly]|all),
  all_same_original_backing:([.cases[].same_original_backing]|all),
  all_fresh_restored_reuse_boundary_equal:
    ([.cases[].fresh_restored_reuse_boundary_equal]|all),
  restoration_generations:[.cases[].restoration_generation],
  restoration_leases:[.cases[].restoration_lease],
  all_canonical_restored_states_zero:
    ([.cases[].canonical_restored_state.all_messages_zero]|all),
  periodic_complete_blocks:
    [.cases[].periodic_block_baseline.complete_blocks],
  periodic_tail_transfers:
    [.cases[].periodic_block_baseline.tail_transfers],
  periodic_dense_block_integer_cells:
    [.cases[].periodic_block_baseline.dense_block_integer_cells],
  periodic_dense_block_build_transfer_equivalents:
    [.cases[].periodic_block_baseline
      .dense_block_build_transfer_equivalents],
  periodic_dense_build_exceeds_streaming:
    ([.cases[].periodic_block_baseline
      .dense_block_build_exceeds_streaming_at_case]|all),
  periodic_powering_executed:
    ([.cases[].periodic_block_baseline.powering_executed]|any),
  controls,
  observed_law,
  matched_classical,
  resource_law,
  not_established,
  next_obstruction,
  terminal
}' "$result" >"$summary"

jq -cS '{
  result,
  oracle,
  tested_nodes,
  direct_enumeration_nodes,
  cases,
  all_semantic_boundaries_equal,
  all_factorized_boundaries_equal,
  all_adaptive_pivots_minimal,
  all_retain_all_inverses_restored_exactly,
  all_declared_descriptor_periods_equal_17,
  production_module_imported,
  production_compiler_called,
  production_transfer_called,
  production_gauge_selector_called,
  production_projector_called,
  production_inverse_called,
  public_descriptors_consumed,
  identical_compact_classical_recurrence,
  periodic_dense_block_powering_executed,
  periodic_dense_block_build_inapplicable_at_declared_cases,
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
  and .restoration_mechanism.transfer_inverse
    == "RECOMPUTE_EXPECTED_TRANSFER_THEN_EXACTLY_SUBTRACT_COEFFICIENTS_FROM_RESIDENT_TARGET"
  and .restoration_mechanism.pivot_metadata_inverse
    == "OFFSET_SUBTRACTION_MOD17_WITH_ZERO_IDENTITY_16"
  and .restoration_mechanism.content_exponent_inverse
    == "EXACT_INTEGER_SUBTRACTION"
  and .restoration_mechanism.seed_inverse
    == "EXACT_COEFFICIENT_PIVOT_AND_CONTENT_SUBTRACTION"
  and (.restoration_mechanism.validated_destructive_erasure_used|not)
  and .representation.ring == "Z[ZETA17]"
  and .representation.integer_cells_per_message == 272
  and .representation.pivot_metadata_bits_per_message == 85
  and (.representation.assignment_table_materialized|not)
  and (.representation.relation_table_materialized|not)
  and [.cases[].nodes] == [2,3,5,9,17,33,65]
  and [.cases[].message_slots] == [2,3,4,5,6,7,8]
  and [.cases[].message_integer_cells]
    == [544,816,1088,1360,1632,1904,2176]
  and [.cases[].pebble_forward_applications]
    == [1,3,9,27,81,243,729]
  and ([.cases[].restored_exactly]|all)
  and ([.cases[].same_original_backing]|all)
  and ([.cases[].fresh_restored_reuse_boundary_equal]|all)
  and ([.cases[].canonical_restored_state.all_messages_zero]|all)
  and ([.cases[].restoration_generation] == [2,2,2,2,2,2,2])
  and ([.cases[].restoration_lease] == [2,2,2,2,2,2,2])
  and .controls.missing_inverse_rejected
  and .controls.wrong_inverse_rejected
  and .controls.reordered_inverse_rejected
  and .controls.null_carrier_rejected
  and .controls.semantic_edge_perturbation_changes_boundary
  and .observed_law.adaptive_peak_payload_bits
    == [937,1885,2851,5580,12760,30652,76433]
  and .observed_law.fixed_basis_peak_payload_bits
    == [869,1837,4653,13432,35191,88872,216589]
  and .observed_law.adaptive_peak_coefficient_signed_bits
    == [3,5,5,9,14,26,50]
  and .observed_law.fixed_basis_peak_coefficient_signed_bits
    == [3,6,10,18,34,68,136]
  and .observed_law.final_stored_17_content_exponents
    == [0,0,1,2,5,10,21]
  and .observed_law.boundary_17_content_exponents
    == [0,1,1,3,5,11,21]
  and .observed_law
    .final_stored_17_content_exponents_match_floor_edges_over_3
  and .observed_law.content_quotient_reduces_depth65_peak_payload
  and (.observed_law.fixed_integer_width_established|not)
  and (.observed_law.constant_reversible_storage_established|not)
  and (.observed_law
    .one_minus_zeta_factorization_reduces_any_final_payload|not)
  and .matched_classical.identical_adaptive_gauge_and_content_recurrence
  and (.matched_classical.dense_block_powering_executed|not)
  and (.matched_classical
    .strongest_family_specific_method_established|not)
  and .resource_law.retained_inverse_history_bytes == 0
  and .resource_law.baseline_reload_bytes == 0
  and .resource_law.temporary_seed_message_integer_cells == 272
  and .resource_law
    .temporary_inverse_expected_message_integer_cells == 272
  and .resource_law.temporary_gauge_candidate_integer_cells == 16
  and .resource_law.temporary_regauge_combined_peak_integer_cells == 49
  and .resource_law
    .verification_content_diagnostic_source_and_list_integer_cells
      == 1088
  and (.resource_law.whole_process_peak_bounded|not)
  and (.terminal|not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and .tested_nodes == [2,3,5,9,17,33,65]
  and .direct_enumeration_nodes == [2,3]
  and .all_semantic_boundaries_equal
  and .all_factorized_boundaries_equal
  and .all_adaptive_pivots_minimal
  and .all_retain_all_inverses_restored_exactly
  and .all_declared_descriptor_periods_equal_17
  and (.production_module_imported|not)
  and (.production_compiler_called|not)
  and (.production_transfer_called|not)
  and (.production_gauge_selector_called|not)
  and (.production_projector_called|not)
  and (.production_inverse_called|not)
  and .public_descriptors_consumed
  and .identical_compact_classical_recurrence
  and (.periodic_dense_block_powering_executed|not)
  and .periodic_dense_block_build_inapplicable_at_declared_cases
  and (.distinct_phase_resource_established|not)
  and (.computational_advantage|not)
  and (.small_wall_crossed|not)
  and (.terminal|not)
' "$oracle_result" >/dev/null

echo "F17 cubic-chain adaptive gauge qualification passed"
