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
production="$frontier_dir/f17_anisotropic_radial_stateful_gauge_phasor_lift.py"
oracle="$frontier_dir/f17_anisotropic_radial_stateful_gauge_phasor_lift_oracle.py"
predecessor="$frontier_dir/f17_anisotropic_radial_local_polar_givens_coupling.py"
reference_result="$frontier_dir/F17_ANISOTROPIC_RADIAL_STATEFUL_GAUGE_PHASOR_LIFT_RESULTS.json"
reference_oracle="$frontier_dir/F17_ANISOTROPIC_RADIAL_STATEFUL_GAUGE_PHASOR_LIFT_ORACLE_RESULTS.json"
result="$evidence_dir/F17_ANISOTROPIC_RADIAL_STATEFUL_GAUGE_PHASOR_LIFT_RESULTS.json"
oracle_result="$evidence_dir/F17_ANISOTROPIC_RADIAL_STATEFUL_GAUGE_PHASOR_LIFT_ORACLE_RESULTS.json"

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
  .schema == "CAT_CAS_F17_STATEFUL_GAUGE_PHASOR_LIFT_RESULT_V1"
  and .classification == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level == "PACKAGE_SELF_REVIEW"
  and .execution_scope.case_count == 21
  and .execution_scope.all_cases_within_predeclared_tolerances
  and (.execution_scope.full_weighted_unit_sphere_supported | not)
  and .execution_scope.supported_base_magnitude_ceiling == 0.9375
  and .carrier_law.resident_phase_angle_cells == 51
  and .carrier_law.gauge_weight == 0.03125
  and .carrier_law.base_zero_residual_radius == (1 / 31)
  and (.carrier_law.base_zero_requires_canonicalization | not)
  and (.carrier_law.residual_chart_zero_encountered_in_declared_cases | not)
  and .zero_fiber_obstruction.fiber_cardinality_mismatch_proves_any_injective_lift_impossible
  and (.zero_fiber_obstruction.fixed_four_angle_injective_lift_across_zero_possible | not)
  and .zero_fiber_obstruction.collision_detected
  and .stateful_zero_probe.both_actual_gauges_restored
  and .stateful_zero_probe.gauge_survives_while_base_output_is_identical
  and ([.controls[]] | all)
  and .restoration.class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and .restoration.all51_resident_phase_cells_compared
  and .restoration.same_backing
  and (.restoration.snapshot_reload_used | not)
  and .restoration.inverse_history_cells == 0
  and (.restoration.post_inverse_state_reset_or_canonical_reload_used | not)
  and .resource_law.resident_phase_angle_float64_cells == 51
  and .resource_law.added_gauge_phase_angle_cells_against_m145 == 17
  and .resource_law.retained_complex_state_cells == 0
  and .resource_law.retained_dense_kernel_cells == 0
  and .resource_law.inverse_history_cells == 0
  and .resource_law.retained_restoration_baseline_cells == 0
  and .resource_law.restoration_seed_rematerialized_one_coordinate_at_a_time
  and .resource_law.maximum_named_warm_execution_live_bytes_including_program_json == 3223
  and .resource_law.maximum_named_commitment_live_bytes_including_program_json == 3191
  and .resource_law.commitment_input_copy_bytes == 0
  and .resource_law.commitment_public_hexdigest_bytes == 64
  and .resource_law.commitment_logical_sha256_state_and_block_bytes == 96
  and .resource_law.phase_module_trigonometric_evaluations_per_step == 0
  and .resource_law.projection_phasor_cosine_evaluations == 51
  and .resource_law.projection_phasor_sine_evaluations == 51
  and .matched_classical_frontiers.identical_local_plan.method == "IDENTICAL_PUBLIC_PLAN17_COMPLEX_IN_PLACE_GIVENS_RECURRENCE"
  and .matched_classical_frontiers.identical_local_plan.maximum_named_warm_execution_live_bytes_including_program_json == 3055
  and .matched_classical_frontiers.work_frontier.method == "IDENTICAL_MATRIX_FREE_NORMALIZED17_COMPLEX_RADIAL_RECURRENCE"
  and .matched_classical_frontiers.work_frontier.maximum_named_warm_execution_live_bytes_including_program_json == 1767
  and .matched_classical_frontiers.memory_frontier.method == "STREAMED_REAL_KERNEL_NORMALIZED17_COMPLEX_RADIAL_RECURRENCE"
  and .matched_classical_frontiers.memory_frontier.maximum_named_warm_execution_live_bytes_including_program_json == 983
  and ([.cases[] | .boundary_error_against_matrix_free_complex] | max) <= 1e-10
  and ([.cases[] | .boundary_error_against_streamed_real_kernel_complex] | max) <= 1e-10
  and ([.cases[] | .maximum_state_error_against_matrix_free_complex] | max) <= 2e-11
  and ([.cases[] | .maximum_state_error_against_streamed_real_kernel_complex] | max) <= 2e-11
  and (.matched_classical_frontiers.comparison_establishes_distinct_phase_resource | not)
  and (.matched_classical_frontiers.comparison_establishes_computational_advantage | not)
  and (.claim_boundary.not_established | index("DISTINCT_PHASE_RESOURCE"))
  and (.claim_boundary.not_established | index("SMALL_WALL_CROSSING"))
  and (.claim_boundary.not_established | index("PHYSICAL_BIT_REPLACEMENT"))
  and (.claim_boundary.not_established | index("UNBOUNDED_CATALYTIC_COMPUTATION"))
' "$result" >/dev/null

jq -e '
  .schema == "CAT_CAS_F17_STATEFUL_GAUGE_PHASOR_LIFT_ORACLE_V1"
  and .result == "PASS"
  and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "INDEPENDENT_ORACLE_REEXECUTION"
  and .independence.imports_production_module == false
  and .independence.imports_predecessor_module == false
  and .independence.long_double_dense_operator_reconstruction
  and .independence.separate_float64_qr_compilation
  and .independence.separate51_angle_forward_inverse
  and .independence.separate_zero_fiber_collision_and_repair
  and .case_checks.case_count == 21
  and .case_checks.maximum_boundary_error_against_production <= 1e-10
  and .case_checks.maximum_boundary_error_against_long_double_dense <= 1e-10
  and .case_checks.maximum_state_error_against_long_double_dense <= 2e-11
  and .case_checks.maximum_restoration_error <= 3e-11
  and .zero_fiber_checks.legacy_collision_confirmed
  and .zero_fiber_checks.fiber_cardinality_mismatch_proves_any_injective_lift_impossible
  and .zero_fiber_checks.repaired_both_actual_gauges_restored
  and ([.mutation_controls[]] | all)
  and .comparison_count >= 137
  and .restoration_class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and (.not_established | index("DISTINCT_PHASE_RESOURCE"))
  and (.not_established | index("SMALL_WALL_CROSSING"))
  and (.not_established | index("PHYSICAL_BIT_REPLACEMENT"))
  and (.not_established | index("UNBOUNDED_CATALYTIC_COMPUTATION"))
' "$oracle_result" >/dev/null

if rg -n '^(from|import) f17_anisotropic_radial_(stateful_gauge_phasor_lift|local_polar_givens_coupling)' "$oracle"; then
    echo "oracle imports production or predecessor" >&2
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

echo "QUALIFIED_F17_ANISOTROPIC_RADIAL_STATEFUL_GAUGE_PHASOR_LIFT_STRICT_SCOPE"
