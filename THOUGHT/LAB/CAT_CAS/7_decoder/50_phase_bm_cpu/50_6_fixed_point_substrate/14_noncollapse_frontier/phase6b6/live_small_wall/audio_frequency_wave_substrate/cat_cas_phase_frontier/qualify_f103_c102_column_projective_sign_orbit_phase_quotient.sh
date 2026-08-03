#!/usr/bin/env bash
set -euo pipefail

frontier_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(git -C "${frontier_dir}" rev-parse --show-toplevel)"
python_bin="${repo_dir}/.venv/bin/python"
evidence_dir="$(mktemp -d /dev/shm/ags-audio-m160-qualifier-XXXXXX)"

production="f103_c102_column_projective_sign_orbit_phase_quotient.py"
oracle="f103_c102_column_projective_sign_orbit_phase_quotient_oracle.py"
m158="f103_c102_dual_register_quadratic_phase_shear_relation_no_go.py"
m157="f103_unresolved_c102_group_algebra_superposition_relation_no_go.py"
m158_oracle="f103_c102_dual_register_quadratic_phase_shear_relation_no_go_oracle.py"
m157_oracle="f103_unresolved_c102_group_algebra_superposition_relation_no_go_oracle.py"
result="F103_C102_COLUMN_PROJECTIVE_SIGN_ORBIT_PHASE_QUOTIENT_RESULTS.json"
oracle_result="F103_C102_COLUMN_PROJECTIVE_SIGN_ORBIT_PHASE_QUOTIENT_ORACLE_RESULTS.json"
review="F103_C102_COLUMN_PROJECTIVE_SIGN_ORBIT_PHASE_QUOTIENT_INDEPENDENT_REEXECUTION_REVIEW.md"

test -x "${python_bin}"
for artifact in \
  "${production}" "${oracle}" "${m158}" "${m157}" \
  "${m158_oracle}" "${m157_oracle}" "${result}" "${oracle_result}" \
  "${review}"; do
  cp "${frontier_dir}/${artifact}" "${evidence_dir}/${artifact}"
done

test "$(sha256sum "${evidence_dir}/${production}" | cut -d' ' -f1)" = "5cf1b787a6174a45e8e9262e8c2c3cefd28fe6c5cdc31462a54b5c7aa14fec7d"
test "$(sha256sum "${evidence_dir}/${oracle}" | cut -d' ' -f1)" = "ec86d09c43b7b0dd6d45586db9071e16776d1a24c36aa4f337b32c3717d7df7d"
test "$(sha256sum "${evidence_dir}/${m158}" | cut -d' ' -f1)" = "29b0333cd9236236fad04fa7df8f6dc42851c47c69e89a61a57c843157e0eb69"
test "$(sha256sum "${evidence_dir}/${m157}" | cut -d' ' -f1)" = "961ba11f4c19841985fdfc66959ec865bea1f3b1800c9e05dccfd313fa1f1d57"
test "$(sha256sum "${evidence_dir}/${m158_oracle}" | cut -d' ' -f1)" = "176a0f05528c52178455945982224df5aef620b39b022a16643f6f72d2b88c5f"
test "$(sha256sum "${evidence_dir}/${m157_oracle}" | cut -d' ' -f1)" = "dc40fb0a6a3004262415a573ec5e391ecd04658bd1f3c0fd11952e2092d8f05d"
test "$(sha256sum "${evidence_dir}/${result}" | cut -d' ' -f1)" = "ff60be20b21d6c6c7a03760849e8f3dab7432148a02da49224defe157788d109"
test "$(sha256sum "${evidence_dir}/${oracle_result}" | cut -d' ' -f1)" = "82a6d0f8af243295e746a8de12d07fdf3df29b9c350db90e5302d5ce836c6050"
test "$(sha256sum "${evidence_dir}/${review}" | cut -d' ' -f1)" = "18e11d90f9bcf68b3e1dcc3ee2de6dafa9b0202a1e88f6c776b7764d60b02378"

if rg -n '^\s*(import numpy|from numpy|from .*f103_c102_column_projective_sign_orbit_phase_quotient import|import f103_c102_column_projective_sign_orbit_phase_quotient([[:space:]]|$)|from .*f103_c102_dual_register_quadratic_phase_shear_relation_no_go import|import f103_c102_dual_register_quadratic_phase_shear_relation_no_go([[:space:]]|$))' "${evidence_dir}/${oracle}"; then
  echo "oracle isolation violation" >&2
  exit 1
fi

"${python_bin}" "${evidence_dir}/${production}" \
  --output "${evidence_dir}/production.rerun.json"
cmp "${evidence_dir}/${result}" "${evidence_dir}/production.rerun.json"

"${python_bin}" "${evidence_dir}/${oracle}" \
  "${evidence_dir}/production.rerun.json" \
  --output "${evidence_dir}/oracle.rerun.json"
cmp "${evidence_dir}/${oracle_result}" "${evidence_dir}/oracle.rerun.json"

jq -e '
  .classification == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level == "PACKAGE_SELF_REVIEW"
  and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
  and .experiment.interface == 5
  and .experiment.depths == [1,4,16,64]
  and .experiment.case_count == 8
  and .experiment.orientation_bits == 45
  and .experiment.final_boundary_only == true
  and .experiment.raw_boundary_values_serialized == false
  and ([.cases[] | .matches_raw_identical_coefficient_recurrence and .exact_raw_payload_restored_after_unseal and .same_backing_restored] | all)
  and ([.controls[]] | all)
  and .restoration_and_reuse.same_backing_reused == true
  and .restoration_and_reuse.restoration_generation == 2
  and .restoration_and_reuse.unrelated_second_boundary_matches_fresh == true
  and .restoration_and_reuse.exact_raw_payload_restored_after_reuse == true
  and .restoration_and_reuse.snapshot_used == false
  and .resource_accounting.canonical_representative_field_cells == 45900
  and .resource_accounting.matched_raw_coefficient_recurrence_field_cells == 45900
  and .resource_accounting.material_field_cell_reduction == 0
  and .resource_accounting.projective_orbit_cardinality_reduction_bits_when_all_columns_nonzero == 45
  and .resource_accounting.exact_restoration_orientation_ledger_bits == 45
  and .resource_accounting.net_lossless_information_reduction_bits_after_restoration_ledger == 0
  and .resource_accounting.tested_independent_sign_generators == 45
  and .resource_accounting.free_nonzero_seed_sign_orbit_order == 35184372088832
  and .resource_accounting.advantage_claimed == false
' "${evidence_dir}/production.rerun.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "INDEPENDENT_ORACLE_REEXECUTION"
  and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
  and .imports_m160_production == false
  and .imports_m158_production == false
  and .imports_numpy == false
  and .reuses_prior_independent_m158_scalar_reference == true
  and (.cases | length) == 8
  and ([.cases[] | .matches_raw_identical_coefficient_recurrence and .exact_sealed_restoration and .exact_raw_payload_restored_after_unseal] | all)
  and .generator_control.tested_generators == 45
  and .generator_control.all_collapse_preserve_boundary_and_restore == true
  and .generator_control.minimum_arbitrary_member_restoration_ledger_bits == 45
  and .production_comparison.comparisons == 32
  and .production_comparison.all_match == true
  and .resource_reconstruction.material_field_cell_reduction == 0
  and .resource_reconstruction.net_lossless_information_reduction_bits == 0
' "${evidence_dir}/oracle.rerun.json" >/dev/null

echo "QUALIFIED_F103_C102_COLUMN_PROJECTIVE_SIGN_ORBIT_PHASE_QUOTIENT_STRICT_SCOPE"
echo "evidence=${evidence_dir}"
