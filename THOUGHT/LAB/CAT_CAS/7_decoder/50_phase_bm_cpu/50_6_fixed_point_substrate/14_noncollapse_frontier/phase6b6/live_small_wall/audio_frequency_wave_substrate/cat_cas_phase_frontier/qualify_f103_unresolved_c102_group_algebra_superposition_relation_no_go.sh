#!/usr/bin/env bash
set -euo pipefail

frontier_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(git -C "${frontier_dir}" rev-parse --show-toplevel)"
python_bin="${repo_dir}/.venv/bin/python"
evidence_dir="$(mktemp -d /dev/shm/ags-audio-m157-qualifier-XXXXXX)"

production="f103_unresolved_c102_group_algebra_superposition_relation_no_go.py"
oracle="f103_unresolved_c102_group_algebra_superposition_relation_no_go_oracle.py"
result="F103_UNRESOLVED_C102_GROUP_ALGEBRA_SUPERPOSITION_RELATION_NO_GO_RESULTS.json"
oracle_result="F103_UNRESOLVED_C102_GROUP_ALGEBRA_SUPERPOSITION_RELATION_NO_GO_ORACLE_RESULTS.json"
review="F103_UNRESOLVED_C102_GROUP_ALGEBRA_SUPERPOSITION_RELATION_NO_GO_INDEPENDENT_REEXECUTION_REVIEW.md"

test -x "${python_bin}"

for artifact in "${production}" "${oracle}" "${result}" "${oracle_result}" "${review}"; do
  cp "${frontier_dir}/${artifact}" "${evidence_dir}/${artifact}"
done

test "$(sha256sum "${evidence_dir}/${production}" | cut -d' ' -f1)" = "961ba11f4c19841985fdfc66959ec865bea1f3b1800c9e05dccfd313fa1f1d57"
test "$(sha256sum "${evidence_dir}/${oracle}" | cut -d' ' -f1)" = "dc40fb0a6a3004262415a573ec5e391ecd04658bd1f3c0fd11952e2092d8f05d"
test "$(sha256sum "${evidence_dir}/${result}" | cut -d' ' -f1)" = "8e9ea0040c24c5ff64440ab39b67a8b8b7bd85b19515d750c2cb3ab414fbbe7e"
test "$(sha256sum "${evidence_dir}/${oracle_result}" | cut -d' ' -f1)" = "81067035c4be28308b09db38576b8fafe2e978e7272960829a953a0d1a658c8d"
test "$(sha256sum "${evidence_dir}/${review}" | cut -d' ' -f1)" = "6f7bb878e6405bc92154323f07fc208273559d91be1136aec59580d649191058"

if rg -n '^\s*(import numpy|from numpy|from .*f103_unresolved_c102_group_algebra_superposition_relation_no_go import|import f103_unresolved_c102_group_algebra_superposition_relation_no_go)' "${evidence_dir}/${oracle}"; then
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
  and .experiment.case_count == 24
  and .experiment.intermediate_log_or_value_rechart == false
  and .experiment.final_boundary_only_evaluation == true
  and .growth_law.fixed_rank_or_subdense_growth_established == false
  and .growth_law.expression_dag_nodes_edges_and_bytes_grow_strictly_with_declared_depths == true
  and ([.growth_law.phase_to_classical_payload_ratio_by_interface[]] | all(. == 102))
  and ([.growth_law.forward_maximum_nonzero_support_by_interface_depth[]["4"]] | all(. == 102))
  and ([.controls[]] | all)
  and .restoration_and_reuse.coefficient_carrier_actual_inverse_on_borrowed_carrier == true
  and .restoration_and_reuse.coefficient_carrier_exact_payload_restoration == true
  and .restoration_and_reuse.coefficient_carrier_same_backing_restoration == true
  and .restoration_and_reuse.snapshot_used == false
  and .expression_dag_inverse_diagnostic.coefficient_semantics_restore_after_expansion == true
  and .expression_dag_inverse_diagnostic.root_identity_restored == false
  and .expression_dag_inverse_diagnostic.restoration_classification == "NO_RESTORATION_CLAIM"
  and .matched_compact_classical.is_exact_evaluation_quotient_of_group_algebra_carrier == true
  and .no_smuggle.raw_final_boundaries_serialized == false
  and .no_smuggle.resident_group_algebra_coefficients_serialized == false
  and ([paths(scalars) as $p | select(($p[-1] | tostring) == "_boundary")] | length == 0)
' "${evidence_dir}/production.rerun.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "INDEPENDENT_ORACLE_REEXECUTION"
  and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
  and .imports_production_module == false
  and .imports_numpy == false
  and .case_count == 24
  and .comparison_count == 385
  and .exact_inverse_cases == 24
  and .reconstructed_full_group_algebra_coefficients == true
  and .reconstructed_characterwise_inverse_kernels == true
  and .reconstructed_forward_coefficient_commitments == true
  and .reconstructed_dense_evaluation_quotient == true
  and .reconstructed_expression_dag_metrics == true
  and ([.semantic_controls[]] | all)
' "${evidence_dir}/oracle.rerun.json" >/dev/null

echo "QUALIFIED_F103_UNRESOLVED_C102_GROUP_ALGEBRA_SUPERPOSITION_RELATION_NO_GO_STRICT_SCOPE"
echo "evidence=${evidence_dir}"
