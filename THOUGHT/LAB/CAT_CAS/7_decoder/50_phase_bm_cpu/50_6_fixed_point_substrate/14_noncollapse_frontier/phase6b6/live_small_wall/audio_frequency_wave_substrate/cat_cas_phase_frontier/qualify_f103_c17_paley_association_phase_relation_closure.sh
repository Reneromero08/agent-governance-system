#!/usr/bin/env bash
set -euo pipefail

frontier_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(git -C "${frontier_dir}" rev-parse --show-toplevel)"
python_bin="${repo_dir}/.venv/bin/python"
evidence_dir="$(mktemp -d /dev/shm/ags-audio-m163-qualifier-XXXXXX)"

production="f103_c17_paley_association_phase_relation_closure.py"
oracle="f103_c17_paley_association_phase_relation_closure_oracle.py"
result="F103_C17_PALEY_ASSOCIATION_PHASE_RELATION_CLOSURE_RESULTS.json"
oracle_result="F103_C17_PALEY_ASSOCIATION_PHASE_RELATION_CLOSURE_ORACLE_RESULTS.json"
review="F103_C17_PALEY_ASSOCIATION_PHASE_RELATION_CLOSURE_INDEPENDENT_REEXECUTION_REVIEW.md"

test -x "${python_bin}"
for artifact in "${production}" "${oracle}" "${result}" "${oracle_result}" "${review}"; do
  cp "${frontier_dir}/${artifact}" "${evidence_dir}/${artifact}"
done

test "$(sha256sum "${evidence_dir}/${production}" | cut -d' ' -f1)" = "908ebe0a6214a1fa30fa13ecff432d3a98c013a5fee1d893f308d1b8ba848ff5"
test "$(sha256sum "${evidence_dir}/${oracle}" | cut -d' ' -f1)" = "3052e933177e83c07d2c821eca631222deedd9248e689e50e4927115538dce81"
test "$(sha256sum "${evidence_dir}/${result}" | cut -d' ' -f1)" = "cccd8d218cdafcd5b632ced856210cdf36d1bcf70337d3ff52ce7ccf746ce06f"
test "$(sha256sum "${evidence_dir}/${oracle_result}" | cut -d' ' -f1)" = "a8dc5639baf943bbe096484baa2449505350f8905a6c3c1c519ae13bd76ce251"
test "$(sha256sum "${evidence_dir}/${review}" | cut -d' ' -f1)" = "ed7a326711d21729afb08db10959919758740f4a0d16cb985dd48125bef0e27f"

if rg -n '^\s*(import numpy|from numpy|from .*f103_c17_paley_association_phase_relation_closure import|import f103_c17_paley_association_phase_relation_closure([[:space:]]|$))' "${evidence_dir}/${oracle}"; then
  echo "oracle isolation violation" >&2
  exit 1
fi

"${python_bin}" "${evidence_dir}/${production}" --output "${evidence_dir}/production.rerun.json"
cmp "${evidence_dir}/${result}" "${evidence_dir}/production.rerun.json"
"${python_bin}" "${evidence_dir}/${oracle}" "${evidence_dir}/production.rerun.json" --output "${evidence_dir}/oracle.rerun.json"
cmp "${evidence_dir}/${oracle_result}" "${evidence_dir}/oracle.rerun.json"

jq -e '
  .classification == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level == "PACKAGE_SELF_REVIEW"
  and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
  and .experiment.case_count == 12
  and .experiment.resident_relation_coordinates_per_port == 3
  and .experiment.shared_unresolved_a_relation_consumers_per_step == 2
  and .experiment.shared_unresolved_b_relation_consumers_per_step == 2
  and .experiment.final_boundary_only == true
  and .experiment.hidden_a_serialized == false
  and .experiment.full17_relation_materialized_on_accepted_path == false
  and .experiment.dense17_by17_relation_table_materialized == false
  and ([.cases[] | .matches_identical_compact_classical_recurrence and .exact_cells_restored and .same_backing_restored] | all)
  and ([.controls[]] | all)
  and ([.algebra_checks[]] | all)
  and .restoration_and_reuse.same_backing_reused == true
  and .restoration_and_reuse.exact_cells_restored_after_reuse == true
  and .restoration_and_reuse.restoration_generation == 2
  and .restoration_and_reuse.unrelated_second_boundary_matches_fresh == true
  and .restoration_and_reuse.unrelated_second_commitment_matches_fresh == true
  and .restoration_and_reuse.snapshot_used == false
  and .resource_accounting.accepted_carrier_field_cells == 6
  and .resource_accounting.maximum_logical_working_field_cells_peak == 15
  and .resource_accounting.matched_compact_classical_carrier_field_cells == 6
  and .resource_accounting.accepted_over_matched_compact_classical_carrier_ratio == 1
  and .resource_accounting.phase_and_matched_classical_operation_law_identical == true
  and .resource_accounting.retained_inverse_history_field_cells == 0
  and .resource_accounting.advantage_claimed == false
' "${evidence_dir}/production.rerun.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "INDEPENDENT_ORACLE_REEXECUTION"
  and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
  and .imports_production == false
  and .imports_numpy == false
  and .oracle_state_law == "FULL17_F103_TRANSLATION_INVARIANT_RELATION_VALUES"
  and .algebra_oracle.class_sizes == [1,8,8]
  and .algebra_oracle.composition_checks == 9
  and .algebra_oracle.hadamard_checks == 9
  and .algebra_oracle.all_outputs_class_constant == true
  and .algebra_oracle.single_residue_delta_outside_family_rejected == true
  and .algebra_oracle.zeta_order17 == true
  and .production_comparison.cases == 12
  and .production_comparison.field_comparisons == 60
  and .production_comparison.all_match == true
  and ([.cases[] | .all_fields_match and .full34_cells_restore_exactly] | all)
  and .resource_scope.accepted_two_register_carrier_field_cells == 6
  and .resource_scope.oracle_full_two_register_carrier_field_cells == 34
  and .resource_scope.dense17_by17_relation_table_cells == 0
' "${evidence_dir}/oracle.rerun.json" >/dev/null

echo "QUALIFIED_F103_C17_PALEY_ASSOCIATION_PHASE_RELATION_CLOSURE_STRICT_SCOPE"
echo "evidence=${evidence_dir}"
