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
production="$frontier_dir/f103_c17_superposition_interference_factor_no_go.py"
oracle="$frontier_dir/f103_c17_superposition_interference_factor_no_go_oracle.py"
dependency="$frontier_dir/f17_exact_c17_fiber_port_convolution.py"
reference_result="$frontier_dir/F103_C17_SUPERPOSITION_INTERFERENCE_FACTOR_NO_GO_RESULTS.json"
reference_oracle="$frontier_dir/F103_C17_SUPERPOSITION_INTERFERENCE_FACTOR_NO_GO_ORACLE_RESULTS.json"
result="$evidence_dir/F103_C17_SUPERPOSITION_INTERFERENCE_FACTOR_NO_GO_RESULTS.json"
oracle_result="$evidence_dir/F103_C17_SUPERPOSITION_INTERFERENCE_FACTOR_NO_GO_ORACLE_RESULTS.json"

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
  .schema == "CAT_CAS_F103_C17_SUPERPOSITION_INTERFERENCE_FACTOR_NO_GO_RESULT_V1"
  and .classification == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level == "PACKAGE_SELF_REVIEW"
  and .execution_scope.case_count == 18
  and .execution_scope.all_cases_exact
  and (.execution_scope.catvm_machine_boundary_used | not)
  and (.execution_scope.public_topology_compilation_reads_final_answers | not)
  and .carrier_law.field_modulus == 103
  and .carrier_law.cycle_order == 17
  and .carrier_law.primitive_c17_root == 72
  and .carrier_law.logical_phase_factors == 51
  and .carrier_law.resident_field_coordinates == 867
  and .carrier_law.general_multi_coordinate_superposition
  and .carrier_law.destructive_interference_events_observed > 0
  and .carrier_law.shared_hub_consumers_per_out_layer == 16
  and .carrier_law.reciprocal_hub_consumers_per_in_layer == 16
  and (.carrier_law.resident_port_scalar_or_exponent_readout | not)
  and .carrier_law.resident_port_unprojected_until_final_boundary
  and ([.controls[]] | all)
  and .exact_spectral_factorization.full_state_and_boundary_match_every_case
  and .exact_spectral_factorization.method == "EXECUTED17_MODE_F103_NUMBER_THEORETIC_FACTOR_RECURRENCE"
  and .exact_spectral_factorization.resident_field_coordinates == 867
  and (.exact_spectral_factorization.optimal_compact_classical_recurrence_claimed | not)
  and .restoration.carrier_classification == "EXACT_ALGEBRAIC_RESTORATION"
  and .restoration.transient_buffers_classification == "NO_RESTORATION_CLAIM"
  and .restoration.same_backing
  and (.restoration.snapshot_reload_used | not)
  and .restoration.inverse_history_cells == 0
  and .restoration.retained_restoration_baseline_cells == 0
  and .restoration.unrelated_program_reuse.second_boundary_matches_fresh
  and .restoration.unrelated_program_reuse.resource_signature_matches_fresh
  and .restoration.repeated_reuse.exact_restoration
  and .restoration.repeated_reuse.cycles == 100
  and .resource_accounting.phase_resident_uint8_cells == 867
  and .resource_accounting.spectral_resident_uint8_cells == 867
  and .resource_accounting.phase_to_spectral_resident_dimension_ratio == 1
  and .resource_accounting.maximum_phase_forward_coefficient_multiplications
      == 17 * .resource_accounting.maximum_spectral_forward_modal_multiplications
  and (.not_established | index("DISTINCT_PHASE_RESOURCE"))
  and (.not_established | index("COMPUTATIONAL_ADVANTAGE"))
  and (.not_established | index("SMALL_WALL_CROSSING"))
  and (.not_established | index("PHYSICAL_BIT_REPLACEMENT"))
  and (.not_established | index("UNBOUNDED_CATALYTIC_COMPUTATION"))
' "$result" >/dev/null

jq -e '
  .schema == "CAT_CAS_F103_C17_SUPERPOSITION_INTERFERENCE_FACTOR_NO_GO_ORACLE_V1"
  and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "INDEPENDENT_ORACLE_REEXECUTION"
  and (.independence.imports_production | not)
  and (.independence.imports_m149_dependency | not)
  and .scope.case_count == 18
  and .scope.independent_comparison_count == 126
  and ([.controls[]] | all)
  and .repeated_reuse.exact_restoration
  and .repeated_reuse.cycles == 100
  and .observed_resource_law.phase_coefficient_convolution_multiplications_per_shear == 289
  and .observed_resource_law.spectral_modal_multiplications_per_shear == 17
  and .observed_resource_law.phase_to_spectral_convolution_work_ratio == 17
  and .observed_resource_law.resident_field_coordinates_both_paths == 867
  and .observed_resource_law.phase_to_spectral_resident_dimension_ratio == 1
  and (.rejected_interpretations | index("DISTINCT_PHASE_RESOURCE"))
  and (.rejected_interpretations | index("COMPUTATIONAL_ADVANTAGE"))
  and (.rejected_interpretations | index("SMALL_WALL_CROSSING"))
  and (.rejected_interpretations | index("PHYSICAL_BIT_REPLACEMENT"))
  and (.rejected_interpretations | index("UNBOUNDED_CATALYTIC_COMPUTATION"))
' "$oracle_result" >/dev/null

if rg -n '^(from|import) (f103_c17_superposition_interference_factor_no_go|f17_exact_c17_fiber_port_convolution)' "$oracle"; then
    echo "oracle imports production or M149 dependency" >&2
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

echo "QUALIFIED_F103_C17_SUPERPOSITION_INTERFERENCE_FACTOR_NO_GO_STRICT_SCOPE"
