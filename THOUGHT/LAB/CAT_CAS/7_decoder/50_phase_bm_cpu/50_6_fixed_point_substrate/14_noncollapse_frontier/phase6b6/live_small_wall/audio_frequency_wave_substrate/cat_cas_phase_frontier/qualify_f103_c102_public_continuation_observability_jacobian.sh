#!/usr/bin/env bash
set -euo pipefail

frontier_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(git -C "${frontier_dir}" rev-parse --show-toplevel)"
python_bin="${repo_dir}/.venv/bin/python"
evidence_dir="$(mktemp -d /dev/shm/ags-audio-m159-qualifier-XXXXXX)"

production="f103_c102_public_continuation_observability_jacobian.py"
oracle="f103_c102_public_continuation_observability_jacobian_oracle.py"
m158="f103_c102_dual_register_quadratic_phase_shear_relation_no_go.py"
m157="f103_unresolved_c102_group_algebra_superposition_relation_no_go.py"
m157_oracle="f103_unresolved_c102_group_algebra_superposition_relation_no_go_oracle.py"
result="F103_C102_PUBLIC_CONTINUATION_OBSERVABILITY_JACOBIAN_RESULTS.json"
oracle_result="F103_C102_PUBLIC_CONTINUATION_OBSERVABILITY_JACOBIAN_ORACLE_RESULTS.json"
review="F103_C102_PUBLIC_CONTINUATION_OBSERVABILITY_JACOBIAN_INDEPENDENT_REEXECUTION_REVIEW.md"

test -x "${python_bin}"

for artifact in \
  "${production}" \
  "${oracle}" \
  "${m158}" \
  "${m157}" \
  "${m157_oracle}" \
  "${result}" \
  "${oracle_result}" \
  "${review}"; do
  cp "${frontier_dir}/${artifact}" "${evidence_dir}/${artifact}"
done

test "$(sha256sum "${evidence_dir}/${production}" | cut -d' ' -f1)" = "c5114743934bbe9534709d7a247737e9138284652b28aad27cbfc07873166c8d"
test "$(sha256sum "${evidence_dir}/${oracle}" | cut -d' ' -f1)" = "878843c70966c55f42a4c5f513288cac53fb15fa16350aefb13d92cdb16bb7a5"
test "$(sha256sum "${evidence_dir}/${m158}" | cut -d' ' -f1)" = "29b0333cd9236236fad04fa7df8f6dc42851c47c69e89a61a57c843157e0eb69"
test "$(sha256sum "${evidence_dir}/${m157}" | cut -d' ' -f1)" = "961ba11f4c19841985fdfc66959ec865bea1f3b1800c9e05dccfd313fa1f1d57"
test "$(sha256sum "${evidence_dir}/${m157_oracle}" | cut -d' ' -f1)" = "dc40fb0a6a3004262415a573ec5e391ecd04658bd1f3c0fd11952e2092d8f05d"
test "$(sha256sum "${evidence_dir}/${result}" | cut -d' ' -f1)" = "9a5b4f1e718489e09a4cf901ed2f2dc654e0a33e4d17140a7064ca3fdec55806"
test "$(sha256sum "${evidence_dir}/${oracle_result}" | cut -d' ' -f1)" = "9ff37e011e8c9b21d05ce8e98c24fd7ce8dc8f704876b37a83d71a3e3022aca1"
test "$(sha256sum "${evidence_dir}/${review}" | cut -d' ' -f1)" = "ea8ac3f34a470c5d050a87a2009b922651efc48b9856a764e7069410cf2be97b"

if rg -n '^\s*(import numpy|from numpy|from .*f103_c102_public_continuation_observability_jacobian import|import f103_c102_public_continuation_observability_jacobian|from .*f103_c102_dual_register_quadratic_phase_shear_relation_no_go import|import f103_c102_dual_register_quadratic_phase_shear_relation_no_go)' "${evidence_dir}/${oracle}"; then
  echo "oracle isolation violation" >&2
  exit 1
fi

"${python_bin}" "${evidence_dir}/${production}" \
  --output "${evidence_dir}/production.rerun.json"
cmp "${evidence_dir}/${result}" "${evidence_dir}/production.rerun.json"

"${python_bin}" "${evidence_dir}/${oracle}" \
  --production "${evidence_dir}/production.rerun.json" \
  --output "${evidence_dir}/oracle.rerun.json"
cmp "${evidence_dir}/${oracle_result}" "${evidence_dir}/oracle.rerun.json"

jq -e '
  .classification == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level == "PACKAGE_SELF_REVIEW"
  and .underlying_carrier_restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
  and .diagnostic_tangent_restoration_classification == "NO_RESTORATION_CLAIM"
  and .experiment.interface == 5
  and .experiment.depths == [range(1;25)]
  and .experiment.raw_boundary_values_serialized == false
  and .experiment.compiler_inspects_final_answers == false
  and .experiment.all_final_b_matrix_coordinates_have_explicit_public_final_only_projection_descriptors == true
  and .full_rank102_on_every_family == true
  and (.families | length) == 2
  and ([.families[] | .final_rank == 102 and .nullity == 0 and .lawful_final_b_projection_rows == 600 and .source_directions == 102] | all)
  and ([.families[] | select(.family == "PRIMARY") | .prefix_ranks_by_maximum_depth["24"]] == [102])
  and ([.families[] | select(.family == "ALTERNATE") | .prefix_ranks_by_maximum_depth["23"]] == [102])
  and ([.controls[]] | all)
  and .restoration_and_reuse.same_backing_reused == true
  and .restoration_and_reuse.restoration_generation == 2
  and .restoration_and_reuse.unrelated_second_boundary_matches_fresh == true
  and .restoration_and_reuse.snapshot_used == false
  and .resource_accounting.accepted_phase_carrier_field_cells == 45900
  and .resource_accounting.matched_identical_classical_carrier_field_cells == 45900
  and .resource_accounting.diagnostic_tangent_field_cells == 4681800
  and .resource_accounting.tangent_is_verification_only_not_accepted_carrier_state == true
  and .resource_accounting.advantage_claimed == false
' "${evidence_dir}/production.rerun.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "INDEPENDENT_ORACLE_REEXECUTION"
  and .imports_m159_production == false
  and .imports_m158_production == false
  and .imports_numpy == false
  and .method == "EXACT_FULL_FORWARD_CENTERED_DIFFERENCES_OVER_F103"
  and .full_rank102_on_every_family == true
  and .production_comparison.all_match == true
  and .production_comparison.comparisons == 8
  and ([.families[] | .basis_direction_reexecutions == 102 and .forward_trajectories == 206 and .exact_full_recurrence_centered_difference_control == true and .final_rank == 102] | all)
  and .resource_accounting.jacobian_field_cells == 61200
  and .resource_accounting.simultaneous_plus_minus_source_node_state_field_cells == 10200
  and .resource_accounting.accepted_carrier_or_advantage_claimed == false
' "${evidence_dir}/oracle.rerun.json" >/dev/null

test "$(jq -r '.families[] | select(.family == "PRIMARY") | .jacobian_commitment' "${evidence_dir}/production.rerun.json")" = \
  "$(jq -r '.families[] | select(.family == "PRIMARY") | .jacobian_commitment' "${evidence_dir}/oracle.rerun.json")"
test "$(jq -r '.families[] | select(.family == "ALTERNATE") | .jacobian_commitment' "${evidence_dir}/production.rerun.json")" = \
  "$(jq -r '.families[] | select(.family == "ALTERNATE") | .jacobian_commitment' "${evidence_dir}/oracle.rerun.json")"

echo "QUALIFIED_F103_C102_PUBLIC_CONTINUATION_OBSERVABILITY_JACOBIAN_STRICT_SCOPE"
echo "evidence=${evidence_dir}"
