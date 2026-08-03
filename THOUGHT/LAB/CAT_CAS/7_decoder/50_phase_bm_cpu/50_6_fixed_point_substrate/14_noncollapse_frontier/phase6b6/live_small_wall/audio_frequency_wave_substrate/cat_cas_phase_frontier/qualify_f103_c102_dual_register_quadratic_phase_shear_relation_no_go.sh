#!/usr/bin/env bash
set -euo pipefail

frontier_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(git -C "${frontier_dir}" rev-parse --show-toplevel)"
python_bin="${repo_dir}/.venv/bin/python"
evidence_dir="$(mktemp -d /dev/shm/ags-audio-m158-qualifier-XXXXXX)"

production="f103_c102_dual_register_quadratic_phase_shear_relation_no_go.py"
oracle="f103_c102_dual_register_quadratic_phase_shear_relation_no_go_oracle.py"
base_production="f103_unresolved_c102_group_algebra_superposition_relation_no_go.py"
base_oracle="f103_unresolved_c102_group_algebra_superposition_relation_no_go_oracle.py"
result="F103_C102_DUAL_REGISTER_QUADRATIC_PHASE_SHEAR_RELATION_NO_GO_RESULTS.json"
oracle_result="F103_C102_DUAL_REGISTER_QUADRATIC_PHASE_SHEAR_RELATION_NO_GO_ORACLE_RESULTS.json"
review="F103_C102_DUAL_REGISTER_QUADRATIC_PHASE_SHEAR_RELATION_NO_GO_INDEPENDENT_REEXECUTION_REVIEW.md"

test -x "${python_bin}"

for artifact in \
  "${production}" \
  "${oracle}" \
  "${base_production}" \
  "${base_oracle}" \
  "${result}" \
  "${oracle_result}" \
  "${review}"; do
  cp "${frontier_dir}/${artifact}" "${evidence_dir}/${artifact}"
done

test "$(sha256sum "${evidence_dir}/${production}" | cut -d' ' -f1)" = "8b4e78613c5c9afc78ffed7fe8022a70b40307e23093b0770ed7c5a09e4e0112"
test "$(sha256sum "${evidence_dir}/${oracle}" | cut -d' ' -f1)" = "cb2e6a037b02aaab2d0c81900601d9224ce189731bd39f6c6f5ee2ca03fb0a70"
test "$(sha256sum "${evidence_dir}/${base_production}" | cut -d' ' -f1)" = "961ba11f4c19841985fdfc66959ec865bea1f3b1800c9e05dccfd313fa1f1d57"
test "$(sha256sum "${evidence_dir}/${base_oracle}" | cut -d' ' -f1)" = "dc40fb0a6a3004262415a573ec5e391ecd04658bd1f3c0fd11952e2092d8f05d"
test "$(sha256sum "${evidence_dir}/${result}" | cut -d' ' -f1)" = "706f72845cf6519c0c78d64ff4b59e0b8fdab8026931a2951033708b94d5811e"
test "$(sha256sum "${evidence_dir}/${oracle_result}" | cut -d' ' -f1)" = "7d933c85583bb227e2ee4fe7215636c142c9a4d46184833afc228d4d2a78defe"
test "$(sha256sum "${evidence_dir}/${review}" | cut -d' ' -f1)" = "21b23ee3d076ba32b37bfb1e965fa73167afb0828b22dd46197d3395ebc544ea"

if rg -n '^\s*(import numpy|from numpy|from .*f103_c102_dual_register_quadratic_phase_shear_relation_no_go import|import f103_c102_dual_register_quadratic_phase_shear_relation_no_go)' "${evidence_dir}/${oracle}"; then
  echo "oracle isolation violation" >&2
  exit 1
fi

"${python_bin}" "${evidence_dir}/${production}" --output "${evidence_dir}/production.rerun.json"
cmp "${evidence_dir}/${result}" "${evidence_dir}/production.rerun.json"

"${python_bin}" "${evidence_dir}/${oracle}" "${evidence_dir}/production.rerun.json" --output "${evidence_dir}/oracle.rerun.json" >/dev/null
cmp "${evidence_dir}/${oracle_result}" "${evidence_dir}/oracle.rerun.json"

jq -e '
  .classification == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level == "PACKAGE_SELF_REVIEW"
  and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
  and .experiment.case_count == 12
  and .experiment.same_source_port_multiple_consumers_established == false
  and .experiment.final_boundary_only_evaluation == true
  and .single_character_quotient_attack.single_character_quotient_rejected == true
  and .single_character_quotient_attack.quadratic_observable_hessian_rank == 102
  and .character_coupling_diagnostic.multiplier_character_support_size == 51
  and .character_coupling_diagnostic.multiplier_character_support_is_dense == false
  and .character_coupling_diagnostic.all102_character_sectors_are_causally_coupled == true
  and ([.controls[]] | all)
  and .restoration_and_reuse.actual_inverse_on_borrowed_carrier == true
  and .restoration_and_reuse.exact_payload_restoration == true
  and .restoration_and_reuse.same_backing_restoration == true
  and .restoration_and_reuse.snapshot_used == false
  and .restoration_and_reuse.unrelated_program_reuse.second_boundary_matches_fresh == true
  and .restoration_and_reuse.repeated_reuse.cycles == 8
  and .resource_accounting.accepted_phase_and_classical_state_law == "204N2_FIELD_COORDINATES"
  and .resource_accounting.phase_to_matched_classical_payload_ratio == 1
  and .resource_accounting.physical_numpy_transient_peak_bytes_measured == false
  and .resource_accounting.no_physical_peak_or_runtime_advantage_claimed == true
  and .matched_compact_classical.strongest_accepted_path == "IDENTICAL_DUAL_REGISTER_COEFFICIENT_RECURRENCE"
  and .no_smuggle.raw_final_boundaries_serialized == false
  and .no_smuggle.resident_coefficients_or_characters_serialized == false
  and ([paths(scalars) as $p | select(($p[-1] | tostring) == "_boundary")] | length == 0)
' "${evidence_dir}/production.rerun.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "INDEPENDENT_ORACLE_REEXECUTION"
  and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
  and .imports_production_module == false
  and .imports_numpy == false
  and .reuses_prior_independent_m157_scalar_reference == true
  and .case_count == 12
  and .comparison_count == 96
  and .exact_inverse_cases == 12
  and .reconstructed_dual_register_coefficients == true
  and .reconstructed_quadratic_shear_and_inverse == true
  and .reconstructed_single_character_collision == true
  and .reconstructed_character_coupling_support == true
  and .reconstructed_identical_204n2_coordinate_law == true
  and ([.semantic_controls[]] | all)
' "${evidence_dir}/oracle.rerun.json" >/dev/null

echo "QUALIFIED_F103_C102_DUAL_REGISTER_QUADRATIC_PHASE_SHEAR_RELATION_NO_GO_STRICT_SCOPE"
echo "evidence=${evidence_dir}"
