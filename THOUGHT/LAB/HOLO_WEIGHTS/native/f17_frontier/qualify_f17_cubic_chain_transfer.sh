#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_cubic_chain_transfer.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_cubic_chain_transfer_closure.py"
oracle_path="$frontier_dir/f17_cubic_chain_transfer_oracle.py"
qualifier_path="$frontier_dir/qualify_f17_cubic_chain_transfer.sh"
expected_path="$frontier_dir/F17_CUBIC_CHAIN_TRANSFER_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_CUBIC_CHAIN_TRANSFER_ORACLE_RESULTS.json"
provenance_path="$frontier_dir/F17_CUBIC_CHAIN_TRANSFER_PROVENANCE.json"
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
  carrier,
  tested_cases:[.cases[]|{nodes,edges}],
  primary_program_sha256:[.cases[].primary_program_sha256],
  reuse_program_sha256:[.cases[].reuse_program_sha256],
  primary_boundary_coefficients:
    [.cases[].primary_boundary.canonical_cyclotomic_coefficients],
  reuse_boundary_coefficients:
    [.cases[].reuse_boundary.canonical_cyclotomic_coefficients],
  reversible_pebble_message_slots:
    [.cases[].reversible_pebble_message_slots],
  reversible_pebble_message_integer_cells:
    [.cases[].reversible_pebble_message_integer_cells],
  retain_all_message_integer_cells:
    [.cases[].retain_all_message_integer_cells],
  strongest_compact_classical_message_integer_cells:
    [.cases[].strongest_compact_classical_message_integer_cells],
  generic_assignment_trace_count_avoided:
    [.cases[].generic_assignment_trace_count_avoided],
  maximum_message_integer_payload_bits:
    [.cases[]|[
      .primary_stats.maximum_message_integer_payload_bits,
      .reuse_stats.maximum_message_integer_payload_bits
    ]|max],
  maximum_single_coefficient_signed_bits:
    [.cases[]|[
      .primary_stats.maximum_single_coefficient_signed_bits,
      .reuse_stats.maximum_single_coefficient_signed_bits
    ]|max],
  pebble_forward_step_applications:
    [.cases[].pebble_forward_step_applications],
  primary_transfer_scalar_updates:
    [.cases[].primary_stats.forward_transfer_scalar_updates],
  reuse_transfer_scalar_updates:
    [.cases[].reuse_stats.forward_transfer_scalar_updates],
  primary_boundary_descriptor_bytes:
    [.cases[].primary_boundary_descriptor_bytes],
  reuse_boundary_descriptor_bytes:
    [.cases[].reuse_boundary_descriptor_bytes],
  all_primary_restored_exactly:
    ([.cases[].primary_restored_exactly]|all),
  all_reuse_restored_exactly:
    ([.cases[].reuse_restored_exactly]|all),
  all_fresh_restored_reuse_boundary_equal:
    ([.cases[].fresh_restored_reuse_boundary_equal]|all),
  all_same_original_carrier_backing:
    ([.cases[].same_original_carrier_backing_after_reuse]|all),
  restoration_generations:[.cases[].restoration_generation],
  restoration_leases:[.cases[].restoration_lease],
  controls,
  measured_repair,
  resource_law,
  matched_compact_classical,
  comparison_baselines,
  no_smuggle_scope,
  composition_scope,
  next_obstruction,
  catvm_custody_established,
  distinct_phase_resource_established,
  computational_advantage,
  small_wall_crossed,
  catalytic_inference_established,
  physical_waveform_execution,
  replacement_of_physical_bits_with_pi,
  unbounded_computation_established,
  terminal
}' "$result" >"$summary"

jq -cS '{
  result,
  oracle,
  tested_nodes,
  direct_enumeration_nodes,
  case_summary:[.cases[]|{
    nodes,
    two_message_integer_cells,
    retain_all_integer_cells,
    reported_reversible_pebble_integer_cells,
    independent_pebble_forward_applications,
    direct_enumeration_assignments
  }],
  primary_descriptor_sha256:
    [.cases[].programs[]|select(.family=="primary")|.descriptor_sha256],
  reuse_descriptor_sha256:
    [.cases[].programs[]|select(.family=="reuse")|.descriptor_sha256],
  production_module_imported,
  production_compiler_called,
  production_transfer_called,
  production_projection_called,
  public_descriptors_consumed,
  all_two_message_boundaries_equal,
  all_retain_all_boundaries_equal,
  all_retain_all_inverse_restorations_exact,
  small_direct_enumerations_equal,
  independent_pebble_resource_recurrence_equal,
  strongest_compact_classical_message_integer_cells,
  python_integer_object_overhead_bounded,
  python_container_allocator_peak_bounded,
  whole_process_peak_bounded,
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
  and .carrier.tested_nodes == [2,3,5,9,17,33,65]
  and .carrier.message_cells_per_slot == 272
  and .carrier.cyclotomic_basis_dimension == 16
  and (.carrier.assignment_table_materialized|not)
  and (.carrier.relation_table_materialized|not)
  and (.carrier.transfer_plan_table_materialized|not)
  and ([.cases[].reversible_pebble_message_slots]
    == [2,3,4,5,6,7,8])
  and ([.cases[].reversible_pebble_message_integer_cells]
    == [544,816,1088,1360,1632,1904,2176])
  and ([.cases[].pebble_forward_step_applications]
    == [1,3,9,27,81,243,729])
  and ([.cases[].maximum_recursive_pebble_call_frames]
    == [1,2,3,4,5,6,7])
  and ([.cases[].maximum_recursive_scratch_slot_references]
    == [0,1,2,3,4,5,6])
  and ([.cases[]|
    .strongest_compact_classical_message_integer_cells == 544
    and .generic_assignment_trace_count_avoided > 0
    and .primary_program_descriptor_bytes > 0
    and .reuse_program_descriptor_bytes > 0
    and .public_factor_f17_cells == 3*.nodes+3*(.nodes-1)
    and .public_factor_fixed_width_bits
      == 5*.public_factor_f17_cells
    and (.generic_assignment_table_materialized|not)
    and (.transfer_plan_table_materialized|not)
    and .retained_inverse_history_bytes == 0
    and .baseline_reload_bytes == 0
    and .primary_restored_exactly
    and .reuse_restored_exactly
    and .fresh_restored_reuse_boundary_equal
    and .same_original_carrier_backing_after_reuse
    and .restoration_generation == 2
    and .restoration_lease == 2
    and .canonical_restored_state.all_message_cells_zero
    and (.canonical_restored_state.active|not)
    and .canonical_restored_state.pending_operations == 0
    and .primary_boundary.normalization_denominator_base == 17
    and .primary_boundary.normalization_denominator_sqrt_power
      == .nodes
    and .reuse_boundary.normalization_denominator_base == 17
    and .reuse_boundary.normalization_denominator_sqrt_power
      == .nodes
    and .primary_boundary
      .preprojection_value_cyclotomic_message_cells == 272
    and .primary_boundary
      .projected_boundary_cyclotomic_coefficients == 16
    and .maximum_recursive_scratch_slot_references
      == .reversible_pebble_message_slots-2
    and .primary_record_peak_integer_cells_scanned > 0
    and .reuse_record_peak_integer_cells_scanned > 0
  ]|all)
  and .controls.missing_inverse_rejected
  and .controls.wrong_inverse_rejected
  and .controls.reordered_inverse_rejected
  and .controls.null_carrier_rejected
  and .controls.semantic_edge_perturbation_changes_boundary
  and .measured_repair
    .generic_17_to_k_final_trace_removed_for_declared_chain
  and .measured_repair.message_cell_count_logarithmic_in_chain_depth
  and (.measured_repair.integer_payload_width_fixed_in_chain_depth|not)
  and .measured_repair.retain_all_history_avoided
  and (.measured_repair.constant_message_storage_achieved|not)
  and .matched_compact_classical.same_exact_transfer_recurrence
  and .matched_compact_classical.same_integer_width_growth
  and .matched_compact_classical
    .strictly_less_message_storage_for_nodes_ge_3
  and .matched_compact_classical
    .strictly_less_transfer_work_for_nodes_ge_3
  and .matched_compact_classical.baseline_scope
    == "STRONGEST_IMPLEMENTED_MATCHED_STREAMING_BASELINE"
  and (.matched_compact_classical.periodic_block_transfer_powering_tested|not)
  and (.matched_compact_classical
    .strongest_family_specific_compact_baseline_established|not)
  and .resource_law.verification_peak_simultaneous_carriers == 2
  and .resource_law
    .fresh_reuse_additional_verification_carriers == 1
  and .resource_law.fixed_local_left_right_basis_domains_enumerated
  and (.resource_law
    .growing_integer_bit_operation_complexity_bounded|not)
  and (.resource_law.recursive_python_call_stack_bytes_bounded|not)
  and (.resource_law.scratch_tuple_control_bytes_bounded|not)
  and .comparison_baselines
    .occurrence_expanded_assignment_trace.accepted_as_matched_baseline
    == false
  and .composition_scope.native_exact_cyclotomic_transfer_composition
  and .composition_scope.nearest_neighbor_mixed_cubic_factors
  and .composition_scope.topology_derived_variable_elimination
  and (.composition_scope.arbitrary_graph_topology|not)
  and (.composition_scope.arbitrary_treewidth|not)
  and (.composition_scope.general_non_gaussian_composition|not)
  and (.no_smuggle_scope.machine_enforced_custody|not)
  and (.catvm_custody_established|not)
  and (.distinct_phase_resource_established|not)
  and (.computational_advantage|not)
  and (.small_wall_crossed|not)
  and (.catalytic_inference_established|not)
  and (.physical_waveform_execution|not)
  and (.replacement_of_physical_bits_with_pi|not)
  and (.unbounded_computation_established|not)
  and (.terminal|not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and .tested_nodes == [2,3,5,9,17,33,65]
  and .direct_enumeration_nodes == [2,3]
  and .all_two_message_boundaries_equal
  and .all_retain_all_boundaries_equal
  and .all_retain_all_inverse_restorations_exact
  and .small_direct_enumerations_equal
  and .independent_pebble_resource_recurrence_equal
  and .strongest_compact_classical_message_integer_cells == 544
  and (.production_module_imported|not)
  and (.production_compiler_called|not)
  and (.production_transfer_called|not)
  and (.production_projection_called|not)
  and .public_descriptors_consumed
  and ([.cases[].programs[]|
    .two_message_boundary_equal
    and .retain_all_boundary_equal
    and .retain_all_inverse_restored_exactly
  ]|all)
  and ([.cases[]|select(.nodes==2 or .nodes==3)|.programs[]|
    .direct_enumeration_boundary_equal
  ]|all)
  and (.python_integer_object_overhead_bounded|not)
  and (.python_container_allocator_peak_bounded|not)
  and (.whole_process_peak_bounded|not)
  and (.distinct_phase_resource_established|not)
  and (.computational_advantage|not)
  and (.small_wall_crossed|not)
  and (.terminal|not)
' "$oracle_result" >/dev/null

if rg -n 'import .*f17_cubic_chain_transfer_closure|from .*f17_cubic_chain' \
  "$oracle_path"
then
  echo "separate oracle imports the production package" >&2
  exit 1
fi

if rg -n 'itertools|product\(' "$source_path"; then
  echo "accepted transfer path contains assignment enumeration" >&2
  exit 1
fi

jq -n \
  --arg result_sha256 "$(sha256sum "$result" | cut -d' ' -f1)" \
  --arg oracle_sha256 "$(sha256sum "$oracle_result" | cut -d' ' -f1)" \
  --arg source_sha256 "$(sha256sum "$source_path" | cut -d' ' -f1)" \
  --arg oracle_source_sha256 \
    "$(sha256sum "$oracle_path" | cut -d' ' -f1)" \
  --arg qualifier_source_sha256 \
    "$(sha256sum "$qualifier_path" | cut -d' ' -f1)" \
  '{
    result:"PASS",
    verification_level:"SEPARATE_REFERENCE_PARITY",
    classification:"INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
    restoration_class:"EXACT_ALGEBRAIC_RESTORATION",
    tested_nodes:[2,3,5,9,17,33,65],
    direct_enumeration_nodes:[2,3],
    result_sha256:$result_sha256,
    oracle_sha256:$oracle_sha256,
    source_sha256:$source_sha256,
    oracle_source_sha256:$oracle_source_sha256,
    qualifier_source_sha256:$qualifier_source_sha256,
    compact_chain_transfer_closure_established:true,
    fixed_integer_payload_width_established:false,
    distinct_phase_resource_established:false,
    computational_advantage:false,
    small_wall_crossed:false,
    terminal:false
  }' >"$evidence_dir/qualification.json"

echo "F17 cubic-chain transfer qualification passed"
