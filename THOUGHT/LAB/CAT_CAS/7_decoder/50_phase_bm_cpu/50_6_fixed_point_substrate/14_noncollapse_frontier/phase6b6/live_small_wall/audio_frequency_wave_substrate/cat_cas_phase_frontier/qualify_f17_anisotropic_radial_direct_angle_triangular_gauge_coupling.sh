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
production="$frontier_dir/f17_anisotropic_radial_direct_angle_triangular_gauge_coupling.py"
oracle="$frontier_dir/f17_anisotropic_radial_direct_angle_triangular_gauge_coupling_oracle.py"
predecessor="$frontier_dir/f17_anisotropic_radial_stateful_gauge_phasor_lift.py"
reference_result="$frontier_dir/F17_ANISOTROPIC_RADIAL_DIRECT_ANGLE_TRIANGULAR_GAUGE_COUPLING_RESULTS.json"
reference_oracle="$frontier_dir/F17_ANISOTROPIC_RADIAL_DIRECT_ANGLE_TRIANGULAR_GAUGE_COUPLING_ORACLE_RESULTS.json"
result="$evidence_dir/F17_ANISOTROPIC_RADIAL_DIRECT_ANGLE_TRIANGULAR_GAUGE_COUPLING_RESULTS.json"
oracle_result="$evidence_dir/F17_ANISOTROPIC_RADIAL_DIRECT_ANGLE_TRIANGULAR_GAUGE_COUPLING_ORACLE_RESULTS.json"

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
  .schema == "CAT_CAS_F17_DIRECT_ANGLE_TRIANGULAR_GAUGE_COUPLING_RESULT_V1"
  and .classification == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level == "PACKAGE_SELF_REVIEW"
  and .execution_scope.case_count == 21
  and .execution_scope.all_cases_within_predeclared_tolerances
  and (.execution_scope.full_weighted_unit_sphere_supported | not)
  and (.execution_scope.public_topology_compilation_reads_final_answers | not)
  and .carrier_law.resident_phase_angle_cells == 51
  and .carrier_law.coupling_strength == 0.03125
  and .carrier_law.rejected_overstrong_coupling == 0.15625
  and .carrier_law.shared_hub_gauge_consumers_per_out_layer == 16
  and .carrier_law.phase_only_update
  and .carrier_law.out_and_in_layers_noncommute
  and .carrier_law.equal_base_different_gauge_boundary_distinguishable
  and .carrier_law.no_relation_table_or_assignment_expansion
  and .equal_base_different_gauge_witness.same_initial_base_state
  and .equal_base_different_gauge_witness.different_actual_gauge_state
  and .equal_base_different_gauge_witness.gauge_changes_final_boundary
  and .equal_base_different_gauge_witness.both_actual_carriers_restored
  and ([.controls[]] | all)
  and .inverse_conditioning_control.accepted_within_tolerance
  and .inverse_conditioning_control.rejected_overstrong_exceeds_tolerance
  and .inverse_conditioning_control.rejected_overstrong_restoration_error > 1
  and .restoration.class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and .restoration.transient_buffers == "NO_RESTORATION_CLAIM"
  and .restoration.all51_resident_phase_cells_compared
  and .restoration.same_backing
  and (.restoration.snapshot_reload_used | not)
  and .restoration.inverse_history_cells == 0
  and .restoration.retained_restoration_baseline_cells == 0
  and (.restoration.post_inverse_state_reset_or_canonical_reload_used | not)
  and .resource_law.resident_phase_angle_float64_cells == 51
  and .resource_law.resident_phase_angle_bytes == 408
  and .resource_law.retained_public_plan_bytes == 0
  and .resource_law.maximum_named_update_float64_scratch_cells == 4
  and .resource_law.maximum_named_warm_execution_live_bytes_including_program_json == 750
  and .resource_law.retained_complex_state_cells == 0
  and .resource_law.retained_dense_kernel_cells == 0
  and .resource_law.inverse_history_cells == 0
  and .resource_law.retained_restoration_baseline_cells == 0
  and .matched_classical_recurrence.executed_in_every_case
  and .matched_classical_recurrence.final_angle_bytes_identical_in_every_case
  and .matched_classical_recurrence.method == "IDENTICAL51_FLOAT64_ANGLE_TRIANGULAR_RECURRENCE"
  and .matched_classical_recurrence.resident_float64_cells == 51
  and .matched_classical_recurrence.maximum_named_warm_execution_live_bytes_including_program_json == 750
  and (.matched_classical_recurrence.comparison_establishes_distinct_phase_resource | not)
  and (.matched_classical_recurrence.comparison_establishes_computational_advantage | not)
  and (.matched_classical_recurrence.optimal_compact_classical_recurrence_claimed | not)
  and .maximum_errors.phase_cell_against_identical_angle_recurrence == 0
  and .maximum_errors.boundary_against_identical_angle_recurrence <= 2e-11
  and .maximum_errors.single_transaction_restoration <= 2e-11
  and (.claim_boundary.not_established | index("DISTINCT_PHASE_RESOURCE"))
  and (.claim_boundary.not_established | index("COMPUTATIONAL_ADVANTAGE"))
  and (.claim_boundary.not_established | index("SMALL_WALL_CROSSING"))
  and (.claim_boundary.not_established | index("PHYSICAL_BIT_REPLACEMENT"))
  and (.claim_boundary.not_established | index("UNBOUNDED_CATALYTIC_COMPUTATION"))
' "$result" >/dev/null

jq -e '
  .schema == "CAT_CAS_F17_DIRECT_ANGLE_TRIANGULAR_GAUGE_COUPLING_ORACLE_V1"
  and .result == "PASS"
  and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "INDEPENDENT_ORACLE_REEXECUTION"
  and (.independence.imports_production_module | not)
  and (.independence.imports_predecessor_module | not)
  and .independence.separate_weighted_three_phasor_seed_chart
  and .independence.separate51_angle_forward_inverse
  and .independence.separate_triangular_gauge_coupling
  and .independence.separate_equal_base_different_gauge_witness
  and .independence.separate_identical51_angle_recurrence
  and .independence.separate_unrelated_and100_cycle_reuse
  and .independence.separate_inverse_conditioning_attack
  and .case_checks.case_count == 21
  and .case_checks.all_reference_angle_bytes_equal
  and .case_checks.maximum_package_boundary_error <= 2e-11
  and .case_checks.maximum_restoration_error <= 2e-11
  and .equal_base_different_gauge_witness.final_boundary_separation > 1e-6
  and ([.mutation_controls[]] | all)
  and .reuse.fresh_restored_boundary_error <= 2e-11
  and .repeated_reuse.maximum_error <= 2e-11
  and .inverse_conditioning_control.accepted_restoration_error <= 2e-11
  and .inverse_conditioning_control.rejected_overstrong_restoration_error > 1
  and .comparison_count >= 120
  and .restoration_class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and (.not_established | index("DISTINCT_PHASE_RESOURCE"))
  and (.not_established | index("COMPUTATIONAL_ADVANTAGE"))
  and (.not_established | index("SMALL_WALL_CROSSING"))
  and (.not_established | index("PHYSICAL_BIT_REPLACEMENT"))
  and (.not_established | index("UNBOUNDED_CATALYTIC_COMPUTATION"))
' "$oracle_result" >/dev/null

if rg -n '^(from|import) f17_anisotropic_radial_(direct_angle_triangular_gauge_coupling|stateful_gauge_phasor_lift)' "$oracle"; then
    echo "oracle imports production or direct predecessor" >&2
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

echo "QUALIFIED_F17_ANISOTROPIC_RADIAL_DIRECT_ANGLE_TRIANGULAR_GAUGE_COUPLING_STRICT_SCOPE"
