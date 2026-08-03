#!/usr/bin/env bash
set -euo pipefail

frontier_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(git -C "${frontier_dir}" rev-parse --show-toplevel)"
python_bin="${repo_dir}/.venv/bin/python"
evidence_dir="$(mktemp -d /dev/shm/ags-audio-m161-qualifier-XXXXXX)"

production="f17_c17_nilpotent_phase_jet_open_relation_closure.py"
oracle="f17_c17_nilpotent_phase_jet_open_relation_closure_oracle.py"
result="F17_C17_NILPOTENT_PHASE_JET_OPEN_RELATION_CLOSURE_RESULTS.json"
oracle_result="F17_C17_NILPOTENT_PHASE_JET_OPEN_RELATION_CLOSURE_ORACLE_RESULTS.json"
review="F17_C17_NILPOTENT_PHASE_JET_OPEN_RELATION_CLOSURE_INDEPENDENT_REEXECUTION_REVIEW.md"

test -x "${python_bin}"
for artifact in "${production}" "${oracle}" "${result}" "${oracle_result}" "${review}"; do
  cp "${frontier_dir}/${artifact}" "${evidence_dir}/${artifact}"
done

test "$(sha256sum "${evidence_dir}/${production}" | cut -d' ' -f1)" = "64238bf43a957a1dd497313914925495f0f53ec781a692ef50ad0caa2c935810"
test "$(sha256sum "${evidence_dir}/${oracle}" | cut -d' ' -f1)" = "816ae7a7a5106ccaaad5f7b813cd7ada263239b532896a570a5f9920b071257c"
test "$(sha256sum "${evidence_dir}/${result}" | cut -d' ' -f1)" = "986d3948c7a4b0e20e69d46d15f754dc6bb6b938a2b24358c459f4a5b1fa7587"
test "$(sha256sum "${evidence_dir}/${oracle_result}" | cut -d' ' -f1)" = "d6569c939af32d9263b7f218dc040bfd7b6e6ac5e04710a04b6c6930e873fdcc"
test "$(sha256sum "${evidence_dir}/${review}" | cut -d' ' -f1)" = "64e3339a157f2e65e02631eb1309e78e0538ea78135be7df87ff58182da009d7"

if rg -n '^\s*(import numpy|from numpy|from .*f17_c17_nilpotent_phase_jet_open_relation_closure import|import f17_c17_nilpotent_phase_jet_open_relation_closure([[:space:]]|$))' "${evidence_dir}/${oracle}"; then
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
  and .experiment.ranks == [2,4,8]
  and .experiment.depths == [1,4,16,64]
  and .experiment.case_count == 24
  and .experiment.final_boundary_only == true
  and .experiment.hidden_a_jet_serialized == false
  and .experiment.full17_coefficient_vector_materialized_on_accepted_path == false
  and .experiment.dense17_by17_relation_table_materialized == false
  and ([.cases[] | .matches_identical_compact_classical_recurrence and .exact_cells_restored and .same_backing_restored] | all)
  and ([.controls[]] | all)
  and ([.algebra_checks[]] | all)
  and .restoration_and_reuse.same_backing_reused == true
  and .restoration_and_reuse.restoration_generation == 2
  and .restoration_and_reuse.unrelated_second_boundary_matches_fresh == true
  and .restoration_and_reuse.exact_cells_restored_after_reuse == true
  and .restoration_and_reuse.snapshot_used == false
  and .resource_accounting.maximum_accepted_carrier_field_cells == 16
  and .resource_accounting.matched_compact_classical_carrier_field_cells == 16
  and .resource_accounting.accepted_over_matched_compact_classical_carrier_ratio == 1
  and .resource_accounting.maximum_logical_working_field_cells_peak == 56
  and .resource_accounting.matched_compact_classical_logical_working_field_cells_peak == 56
  and .resource_accounting.accepted_over_matched_compact_classical_logical_working_ratio == 1
  and .resource_accounting.phase_and_matched_classical_field_operation_law_identical == true
  and .resource_accounting.retained_inverse_history_field_cells == 0
  and .resource_accounting.advantage_claimed == false
' "${evidence_dir}/production.rerun.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "INDEPENDENT_ORACLE_REEXECUTION"
  and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
  and .imports_production == false
  and .imports_numpy == false
  and .oracle_state_law == "FULL17_COEFFICIENT_F17_C17_GROUP_ALGEBRA"
  and .production_comparison.cases == 24
  and .production_comparison.comparisons == 96
  and .production_comparison.all_match == true
  and ([.cases[] | .full17_coefficient_restoration and .all_fields_match] | all)
  and .quotient_law.comparison_count == 54
  and .quotient_law.all_pass == true
  and .quotient_law.all_scope_controls_pass == true
  and .resource_scope.oracle_full_coefficient_carrier_field_cells == 34
  and .resource_scope.accepted_maximum_jet_carrier_field_cells == 16
  and .resource_scope.oracle_dense17_by17_relation_table_cells == 0
' "${evidence_dir}/oracle.rerun.json" >/dev/null

echo "QUALIFIED_F17_C17_NILPOTENT_PHASE_JET_OPEN_RELATION_CLOSURE_STRICT_SCOPE"
echo "evidence=${evidence_dir}"
