#!/usr/bin/env bash
set -euo pipefail

frontier_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(git -C "${frontier_dir}" rev-parse --show-toplevel)"
python_bin="${repo_dir}/.venv/bin/python"
evidence_dir="$(mktemp -d /dev/shm/ags-audio-m156-qualifier-XXXXXX)"

production="f103_dual_multiplicative_exponent_value_moment_relation_no_go.py"
oracle="f103_dual_multiplicative_exponent_value_moment_relation_no_go_oracle.py"
result="F103_DUAL_MULTIPLICATIVE_EXPONENT_VALUE_MOMENT_RELATION_NO_GO_RESULTS.json"
oracle_result="F103_DUAL_MULTIPLICATIVE_EXPONENT_VALUE_MOMENT_RELATION_NO_GO_ORACLE_RESULTS.json"
review="F103_DUAL_MULTIPLICATIVE_EXPONENT_VALUE_MOMENT_RELATION_NO_GO_INDEPENDENT_REEXECUTION_REVIEW.md"

test -x "${python_bin}"

for artifact in "${production}" "${oracle}" "${result}" "${oracle_result}" "${review}"; do
  cp "${frontier_dir}/${artifact}" "${evidence_dir}/${artifact}"
done

test "$(sha256sum "${evidence_dir}/${production}" | cut -d' ' -f1)" = "bcbc036dc78417ce5d5c69243f85d818f33be0b58cb8b51bbd3cd42167bfa3ff"
test "$(sha256sum "${evidence_dir}/${oracle}" | cut -d' ' -f1)" = "c4353c1f16ee31792614a4805ff2c3eafb2e77ae3859e05ce1d18b11f0d5fa08"
test "$(sha256sum "${evidence_dir}/${result}" | cut -d' ' -f1)" = "5bef713e21fe5449acd3a9cd5fdd7c041938fb21f583123a527a75ebadbe2b57"
test "$(sha256sum "${evidence_dir}/${oracle_result}" | cut -d' ' -f1)" = "640ab2c1627b77dad516bc4c9b0230f58194721f0804726aa3549ce4e0eed522"

if rg -n '^\s*(import numpy|from numpy|from .*f103_dual_multiplicative_exponent_value_moment_relation_no_go import|import f103_dual_multiplicative_exponent_value_moment_relation_no_go)' "${evidence_dir}/${oracle}"; then
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
  and .experiment.case_count == 32
  and .experiment.ordinary_dense_relation_table_materialized_by_phase_path == false
  and .rank_law.all_seed_exponent_components_rank_at_most2 == true
  and .rank_law.every_converted_exponent_component_rank_at_least_n_minus2 == true
  and .rank_law.uniform_fixed_rank_across_growing_interfaces == false
  and ([.controls[]] | all)
  and .restoration_and_reuse.actual_inverse_on_borrowed_carrier == true
  and .restoration_and_reuse.exact_payload_restoration == true
  and .restoration_and_reuse.same_backing_restoration == true
  and .restoration_and_reuse.snapshot_used == false
  and .resource_accounting.phase_retained_dense_relation_table_cells == 0
  and .resource_accounting.phase_resident_bytes_by_interface["5"] > .resource_accounting.classical_resident_bytes_by_interface["5"]
  and .resource_accounting.phase_resident_bytes_by_interface["7"] > .resource_accounting.classical_resident_bytes_by_interface["7"]
  and .resource_accounting.phase_resident_bytes_by_interface["11"] > .resource_accounting.classical_resident_bytes_by_interface["11"]
  and .resource_accounting.phase_resident_bytes_by_interface["17"] > .resource_accounting.classical_resident_bytes_by_interface["17"]
  and .no_smuggle.raw_final_boundaries_serialized == false
  and .no_smuggle.resident_exponent_charts_serialized == false
' "${evidence_dir}/production.rerun.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "INDEPENDENT_ORACLE_REEXECUTION"
  and .imports_production_module == false
  and .imports_numpy == false
  and .case_count == 32
  and .comparison_count == 362
  and .exact_inverse_cases == 32
  and .reconstructed_canonical_chart_commitments == true
  and .reconstructed_crt_rank_law == true
  and .reconstructed_resident_byte_law == true
  and ([.semantic_controls[]] | all)
' "${evidence_dir}/oracle.rerun.json" >/dev/null

echo "QUALIFIED_F103_DUAL_MULTIPLICATIVE_EXPONENT_VALUE_MOMENT_RELATION_NO_GO_STRICT_SCOPE"
echo "evidence=${evidence_dir}"
