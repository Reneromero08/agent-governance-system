#!/usr/bin/env bash
set -euo pipefail

frontier_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(git -C "${frontier_dir}" rev-parse --show-toplevel)"
python_bin="${repo_dir}/.venv/bin/python"
evidence_dir="$(mktemp -d /dev/shm/ags-audio-m162-qualifier-XXXXXX)"

production="f17_c17_common_congruence_hadamard_no_go.py"
oracle="f17_c17_common_congruence_hadamard_no_go_oracle.py"
result="F17_C17_COMMON_CONGRUENCE_HADAMARD_NO_GO_RESULTS.json"
oracle_result="F17_C17_COMMON_CONGRUENCE_HADAMARD_NO_GO_ORACLE_RESULTS.json"
review="F17_C17_COMMON_CONGRUENCE_HADAMARD_NO_GO_INDEPENDENT_REEXECUTION_REVIEW.md"

test -x "${python_bin}"
for artifact in "${production}" "${oracle}" "${result}" "${oracle_result}" "${review}"; do
  cp "${frontier_dir}/${artifact}" "${evidence_dir}/${artifact}"
done

test "$(sha256sum "${evidence_dir}/${production}" | cut -d' ' -f1)" = "9510107475457e630d78663f3ca9f4c9ad2c9361d64557b1d03d4bf9ab98b10e"
test "$(sha256sum "${evidence_dir}/${oracle}" | cut -d' ' -f1)" = "cad6c9cacedb8d418edad1f6655ecb4c45504274295660f53e54a55cc4cb71d9"
test "$(sha256sum "${evidence_dir}/${result}" | cut -d' ' -f1)" = "85f51b783f8c207757357f7d2aa0327fc94d461b8930fd7648bffb972eb3e964"
test "$(sha256sum "${evidence_dir}/${oracle_result}" | cut -d' ' -f1)" = "f07abfe62f6470a337fa0452f1f3293346636c828512bddcb29d22782cb4aff0"
test "$(sha256sum "${evidence_dir}/${review}" | cut -d' ' -f1)" = "3ae9bfb0ee403e48b774d4ec271f49f31151c0571468c87f0503c8d598d41b9b"

if rg -n '^\s*(import numpy|from numpy|from .*f17_c17_common_congruence_hadamard_no_go import|import f17_c17_common_congruence_hadamard_no_go([[:space:]]|$))' "${evidence_dir}/${oracle}"; then
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
  and .common_congruence_no_go.common_invariant_kernel_dimensions == [0,17]
  and .common_congruence_no_go.nonzero_proper_common_kernel_exists == false
  and .common_congruence_no_go.smallest_nonzero_common_quotient_dimension == 17
  and .common_congruence_no_go.all_coordinate_orbits_span_full_space == true
  and ([.restricted_hadamard_multiplier_search[] | .multiplier_space_dimension == 1 and .only_constant_multipliers] | all)
  and (.cases | length) == 24
  and ([.cases[] | .matches_identical_compact_classical_recurrence and .exact_cells_restored and .same_backing_restored and (.hidden_a_serialized | not)] | all)
  and ([.controls[]] | all)
  and .restoration_and_reuse.same_backing_reused == true
  and .restoration_and_reuse.exact_cells_restored_after_reuse == true
  and .restoration_and_reuse.restoration_generation == 2
  and .restoration_and_reuse.unrelated_second_boundary_matches_fresh == true
  and .restoration_and_reuse.unrelated_second_commitment_matches_fresh == true
  and .restoration_and_reuse.snapshot_used == false
  and .resource_accounting.maximum_accepted_phase_jet_carrier_field_cells == 16
  and .resource_accounting.full_nonzero_common_quotient_two_register_carrier_field_cells == 34
  and .resource_accounting.matched_compact_classical_carrier_field_cells == 16
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
  and .oracle_state_law == "FULL17_COEFFICIENT_F17_C17_GROUP_ALGEBRA"
  and .common_kernel_oracle.common_invariant_kernel_dimensions == [0,17]
  and .common_kernel_oracle.nonzero_proper_common_kernel_exists == false
  and ([.independent_monomial_multiplier_search[] | .multiplier_space_dimension == 1 and .only_constants] | all)
  and .structural_parity.common_kernel_parity == true
  and .structural_parity.multiplier_space_parity == true
  and .production_comparison.cases == 24
  and .production_comparison.field_comparisons == 120
  and .production_comparison.all_match == true
  and ([.cases[] | .all_fields_match and .full34_cells_restore_exactly] | all)
  and .quotient_law.comparison_count == 54
  and .quotient_law.all_pass == true
  and .resource_scope.oracle_full_coefficient_carrier_field_cells == 34
  and .resource_scope.accepted_maximum_jet_carrier_field_cells == 16
  and .resource_scope.oracle_dense17_by17_relation_table_cells == 0
' "${evidence_dir}/oracle.rerun.json" >/dev/null

echo "QUALIFIED_F17_C17_COMMON_CONGRUENCE_HADAMARD_NO_GO_STRICT_SCOPE"
echo "evidence=${evidence_dir}"
