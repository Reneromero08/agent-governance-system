#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 EVIDENCE_DIR" >&2
    exit 2
fi

evidence_dir=$1
frontier_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python_bin="$repo_root/.venv/bin/python"
production="$frontier_dir/f17_exact_c17_fiber_port_convolution.py"
oracle="$frontier_dir/f17_exact_c17_fiber_port_convolution_oracle.py"
predecessor="$frontier_dir/f17_anisotropic_radial_local_polar_givens_coupling.py"
reference_result="$frontier_dir/F17_EXACT_C17_FIBER_PORT_CONVOLUTION_RESULTS.json"
reference_oracle="$frontier_dir/F17_EXACT_C17_FIBER_PORT_CONVOLUTION_ORACLE_RESULTS.json"
result="$evidence_dir/F17_EXACT_C17_FIBER_PORT_CONVOLUTION_RESULTS.json"
oracle_result="$evidence_dir/F17_EXACT_C17_FIBER_PORT_CONVOLUTION_ORACLE_RESULTS.json"

mkdir -p "$evidence_dir"
test -x "$python_bin"
test -f "$production"
test -f "$oracle"
test -f "$predecessor"
test -f "$reference_result"
test -f "$reference_oracle"

PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 LC_ALL=C nice -n 10 \
    "$python_bin" -X dev "$production" --output "$result" \
    >"$evidence_dir/production.stdout.json"

PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 LC_ALL=C nice -n 10 \
    "$python_bin" -X dev "$oracle" \
    --package "$result" \
    --production "$production" \
    --predecessor "$predecessor" \
    --output "$oracle_result" \
    >"$evidence_dir/oracle.stdout.json"

jq -e '
  .schema == "CAT_CAS_F17_EXACT_C17_FIBER_PORT_CONVOLUTION_RESULT_V1"
  and .classification == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level == "PACKAGE_SELF_REVIEW"
  and .execution_scope.case_count == 21
  and .execution_scope.all_cases_exact
  and (.execution_scope.catvm_machine_boundary_used | not)
  and (.execution_scope.public_topology_compilation_reads_final_answers | not)
  and .carrier_law.logical_phase_factors == 51
  and .carrier_law.resident_phase_orbit_coordinates_per_factor == 17
  and .carrier_law.resident_phase_orbit_uint8_cells == 867
  and .carrier_law.shared_hub_port_consumers_per_out_layer == 16
  and (.carrier_law.resident_port_scalar_readout_per_update | not)
  and (.carrier_law.resident_port_exponent_decode_per_update | not)
  and .carrier_law.port_is_unprojected_until_final_boundary
  and .carrier_law.port_projection_api_rejects_resident_port
  and .carrier_law.out_and_in_layers_noncommute
  and .carrier_law.gauge_only_port_perturbation_changes_boundary
  and .carrier_law.no_relation_table_or_assignment_expansion
  and .carrier_law.phase_orbit_coordinate_expansion_is_counted
  and ([.controls[]] | all)
  and ([.port_causality_witness[] | select(type == "boolean")] | all)
  and .restoration.class == "EXACT_ALGEBRAIC_RESTORATION"
  and .restoration.transient_buffers == "NO_RESTORATION_CLAIM"
  and .restoration.all867_resident_phase_orbit_cells_compared
  and .restoration.same_backing
  and (.restoration.snapshot_reload_used | not)
  and .restoration.inverse_history_cells == 0
  and .restoration.retained_restoration_baseline_cells == 0
  and (.restoration.post_inverse_state_reset_or_canonical_reload_used | not)
  and .reuse.fresh_restored_boundary_equal
  and .reuse.same_original_backing
  and .repeated_reuse.exact_restoration_every_cycle
  and .repeated_reuse.same_backing
  and .resource_law.resident_phase_orbit_bytes == 867
  and .resource_law.forward_cyclic_convolutions_per_step == 179
  and .resource_law.forward_convolution_coordinate_multiplications_per_step == 51731
  and .resource_law.inverse_history_cells == 0
  and .matched_classical_recurrence.executed_in_every_case
  and .matched_classical_recurrence.final_expanded_phase_bytes_identical_in_every_case
  and .matched_classical_recurrence.boundary_identical_in_every_case
  and .matched_classical_recurrence.method == "IDENTICAL51_UINT8_C17_RESIDUE_TRIANGULAR_RECURRENCE"
  and .matched_classical_recurrence.resident_bytes == 51
  and .matched_classical_recurrence.phase_to_classical_resident_byte_ratio == 17
  and (.matched_classical_recurrence.comparison_establishes_distinct_phase_resource | not)
  and (.matched_classical_recurrence.comparison_establishes_computational_advantage | not)
  and (.matched_classical_recurrence.optimal_compact_classical_recurrence_claimed | not)
  and (.rejected_unrenormalized_complex_coordinate_attempt.accepted_path | not)
  and .rejected_unrenormalized_complex_coordinate_attempt.observed_first_unit_norm_failure_depth == 64
  and (.rejected_unrenormalized_complex_coordinate_attempt.normalization_silently_added | not)
  and (.claim_boundary.not_established | index("DISTINCT_PHASE_RESOURCE"))
  and (.claim_boundary.not_established | index("COMPUTATIONAL_ADVANTAGE"))
  and (.claim_boundary.not_established | index("SMALL_WALL_CROSSING"))
  and (.claim_boundary.not_established | index("PHYSICAL_BIT_REPLACEMENT"))
  and (.claim_boundary.not_established | index("UNBOUNDED_CATALYTIC_COMPUTATION"))
' "$result" >/dev/null

jq -e '
  .schema == "CAT_CAS_F17_EXACT_C17_FIBER_PORT_CONVOLUTION_ORACLE_V1"
  and .result == "PASS"
  and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "INDEPENDENT_ORACLE_REEXECUTION"
  and (.independence.imports_production_module | not)
  and (.independence.imports_public_schedule_dependency | not)
  and .independence.separate_public_schedule_compiler
  and .independence.separate_mask_convolution_forward_inverse
  and .independence.separate_compact51_residue_recurrence
  and .independence.reconstructs_production867_byte_commitments
  and .independence.separate_final_cyclotomic_boundary
  and .independence.separate_controls
  and .independence.separate_unrelated_and100_cycle_reuse
  and .independence.separate_unrenormalized_complex_coordinate_diagnostic
  and .case_checks.case_count == 21
  and .case_checks.all_mask_residue_states_equal
  and .case_checks.all_production_commitments_reconstructed
  and .case_checks.all_boundaries_equal
  and .case_checks.all_exact_restorations_reexecuted
  and ([.controls[]] | all)
  and .reuse.fresh_restored_state_equal
  and .reuse.fresh_restored_boundary_equal
  and .reuse.restored_exactly_after_reuse
  and .reuse.repeated_reuse_exact
  and .unrenormalized_complex_coordinate_diagnostic.first_failure_depth == 64
  and .observed_resource_law.accepted_resident_phase_orbit_bytes == 867
  and .observed_resource_law.matched_resident_residue_bytes == 51
  and .observed_resource_law.resident_byte_ratio == 17
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and .transient_restoration_class == "NO_RESTORATION_CLAIM"
  and .comparison_count >= 300
  and (.rejected_interpretations | index("RESOURCE_BEYOND51_RESIDUE_RECURRENCE"))
  and (.rejected_interpretations | index("COMPUTATIONAL_ADVANTAGE"))
  and (.rejected_interpretations | index("SMALL_WALL_CROSSING"))
  and (.rejected_interpretations | index("PHYSICAL_BIT_REPLACEMENT"))
  and (.rejected_interpretations | index("UNBOUNDED_CATALYTIC_COMPUTATION"))
' "$oracle_result" >/dev/null

if rg -n '^(from|import) f17_(exact_c17_fiber_port_convolution|anisotropic_radial_local_polar_givens_coupling)' "$oracle"; then
    echo "oracle imports production or public schedule dependency" >&2
    exit 1
fi

if rg -n 'np\.argmax|atan2' "$production"; then
    echo "production decodes a resident phase coordinate" >&2
    exit 1
fi

cmp "$reference_result" "$result"
cmp "$reference_oracle" "$oracle_result"
sha256sum \
    "$production" \
    "$oracle" \
    "$predecessor" \
    "$result" \
    "$oracle_result" \
    >"$evidence_dir/SHA256SUMS"

echo "QUALIFIED_F17_EXACT_C17_FIBER_PORT_CONVOLUTION_STRICT_SCOPE"
