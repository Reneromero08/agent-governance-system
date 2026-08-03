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
production="$frontier_dir/f103_c17_quadratic_mode_mixing_no_go.py"
oracle="$frontier_dir/f103_c17_quadratic_mode_mixing_no_go_oracle.py"
dependency="$frontier_dir/f103_c17_superposition_interference_factor_no_go.py"
reference_result="$frontier_dir/F103_C17_QUADRATIC_MODE_MIXING_NO_GO_RESULTS.json"
reference_oracle="$frontier_dir/F103_C17_QUADRATIC_MODE_MIXING_NO_GO_ORACLE_RESULTS.json"
result="$evidence_dir/F103_C17_QUADRATIC_MODE_MIXING_NO_GO_RESULTS.json"
oracle_result="$evidence_dir/F103_C17_QUADRATIC_MODE_MIXING_NO_GO_ORACLE_RESULTS.json"

mkdir -p "$evidence_dir"
test -x "$python_bin"
test -f "$production"
test -f "$oracle"
test -f "$dependency"
test -f "$reference_result"
test -f "$reference_oracle"
test ! -e "$result"
test ! -e "$oracle_result"

PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 LC_ALL=C nice -n 10 \
    "$python_bin" -X dev "$production" --output "$result"

PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 LC_ALL=C nice -n 10 \
    "$python_bin" -X dev "$oracle" \
    --package "$result" \
    --production "$production" \
    --output "$oracle_result"

jq -e '
  .schema == "CAT_CAS_F103_C17_QUADRATIC_MODE_MIXING_NO_GO_RESULT_V1"
  and .classification == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level == "PACKAGE_SELF_REVIEW"
  and .execution_scope.case_count == 18
  and .execution_scope.all_cases_exact
  and (.execution_scope.catvm_machine_boundary_used | not)
  and (.execution_scope.public_topology_compilation_reads_final_answers | not)
  and .carrier_law.field_modulus == 103
  and .carrier_law.cycle_order == 17
  and .carrier_law.logical_phase_factors == 51
  and .carrier_law.resident_field_coordinates == 867
  and .carrier_law.shared_quadratic_port_consumers == 16
  and (.carrier_law.resident_port_scalar_or_exponent_readout | not)
  and .carrier_law.resident_port_unprojected_until_final_boundary
  and .carrier_law.relation_table_cells == 0
  and .carrier_law.assignment_expansion_cells == 0
  and .carrier_law.quadratic_destructive_interference_events_observed > 0
  and ([.controls[]] | all)
  and (.spectral_mode_mixing.independent17_mode_closure_preserved | not)
  and .spectral_mode_mixing.independent_mode_square_sham_rejected
  and .spectral_mode_mixing.coupled17_mode_recurrence_executed
  and .spectral_mode_mixing.full_state_and_boundary_match_every_case
  and .spectral_mode_mixing.input_modes_per_output_quadratic_term == 17
  and .matched_classical_recurrences.resident_field_coordinates_each == 867
  and .matched_classical_recurrences.maximum_phase_nonlinear_core_multiplications == 18957312
  and .matched_classical_recurrences.maximum_coupled_spectral_nonlinear_core_multiplications == 1410048
  and (.matched_classical_recurrences.optimal_compact_classical_recurrence_claimed | not)
  and .restoration.carrier_classification == "EXACT_ALGEBRAIC_RESTORATION"
  and .restoration.transient_buffers_classification == "NO_RESTORATION_CLAIM"
  and .restoration.same_backing
  and (.restoration.snapshot_reload_used | not)
  and .restoration.inverse_history_cells == 0
  and .restoration.retained_restoration_baseline_cells == 0
  and .restoration.unrelated_program_reuse.second_boundary_matches_fresh
  and .restoration.unrelated_program_reuse.resource_signature_matches_fresh
  and .restoration.repeated_reuse.exact_restoration
  and .restoration.repeated_reuse.cycles == 64
  and .resource_accounting.phase_resident_uint8_cells == 867
  and .resource_accounting.coupled_spectral_resident_uint8_cells == 867
  and .resource_accounting.phase_to_coupled_spectral_resident_dimension_ratio == 1
  and (.not_established | index("DISTINCT_PHASE_RESOURCE"))
  and (.not_established | index("COMPUTATIONAL_ADVANTAGE"))
  and (.not_established | index("SMALL_WALL_CROSSING"))
  and (.not_established | index("PHYSICAL_BIT_REPLACEMENT"))
  and (.not_established | index("UNBOUNDED_CATALYTIC_COMPUTATION"))
' "$result" >/dev/null

jq -e '
  .schema == "CAT_CAS_F103_C17_QUADRATIC_MODE_MIXING_NO_GO_ORACLE_V1"
  and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "INDEPENDENT_ORACLE_REEXECUTION"
  and (.independence.imports_production | not)
  and (.independence.imports_m150_dependency | not)
  and (.independence.uses_numpy | not)
  and .scope.case_count == 18
  and .scope.independent_comparison_count == 180
  and ([.controls[]] | all)
  and ([.cases[].checks[]] | all)
  and .repeated_reuse.exact_restoration
  and .repeated_reuse.cycles == 64
  and .observed_resource_law.coefficient_convolution_multiplications_per_shear == 289
  and .observed_resource_law.coefficient_quadratic_multiplications_per_shared_layer == 17
  and .observed_resource_law.spectral_convolution_multiplications_per_shear == 17
  and .observed_resource_law.spectral_quadratic_multiplications_per_shared_layer == 289
  and .observed_resource_law.resident_field_coordinates_all_paths == 867
  and (.observed_resource_law.independent_spectral_modes_close | not)
  and .observed_resource_law.identical_coefficient_classical_recurrence_executes
  and .observed_resource_law.coupled17_mode_classical_recurrence_executes
  and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
  and .transient_restoration_classification == "NO_RESTORATION_CLAIM"
  and (.rejected_interpretations | index("DISTINCT_PHASE_RESOURCE"))
  and (.rejected_interpretations | index("COMPUTATIONAL_ADVANTAGE"))
  and (.rejected_interpretations | index("SMALL_WALL_CROSSING"))
  and (.rejected_interpretations | index("PHYSICAL_BIT_REPLACEMENT"))
  and (.rejected_interpretations | index("UNBOUNDED_CATALYTIC_COMPUTATION"))
' "$oracle_result" >/dev/null

if rg -n '^(from|import) (f103_c17_quadratic_mode_mixing_no_go|f103_c17_superposition_interference_factor_no_go)' "$oracle"; then
    echo "oracle imports production or M150 dependency" >&2
    exit 1
fi

cmp "$reference_result" "$result"
cmp "$reference_oracle" "$oracle_result"
sha256sum \
    "$production" \
    "$oracle" \
    "$dependency" \
    "$result" \
    "$oracle_result" \
    >"$evidence_dir/SHA256SUMS"

echo "QUALIFIED_F103_C17_QUADRATIC_MODE_MIXING_NO_GO_STRICT_SCOPE"
