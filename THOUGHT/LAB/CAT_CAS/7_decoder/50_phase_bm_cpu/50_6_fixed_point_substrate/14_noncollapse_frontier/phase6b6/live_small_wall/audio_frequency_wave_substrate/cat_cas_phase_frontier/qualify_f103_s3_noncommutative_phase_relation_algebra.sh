#!/usr/bin/env bash
set -euo pipefail
frontier_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(git -C "${frontier_dir}" rev-parse --show-toplevel)"
python_bin="${repo_dir}/.venv/bin/python"
evidence_dir="$(mktemp -d /dev/shm/ags-audio-m166-qualifier-XXXXXX)"
files=(f103_s3_noncommutative_phase_relation_algebra.py f103_s3_noncommutative_phase_relation_oracle.py F103_S3_NONCOMMUTATIVE_PHASE_RELATION_RESULTS.json F103_S3_NONCOMMUTATIVE_PHASE_RELATION_ORACLE_RESULTS.json F103_S3_NONCOMMUTATIVE_PHASE_RELATION_INDEPENDENT_REEXECUTION_REVIEW.md)
hashes=(7a0496734464d71265c035cec01c790d02c706a999a021c184dac1955474d4e2 18d20f38900d221a4cbbb681eb0936016ea4346b6d689416fc13cf7e57abb648 6ee033a32857f8842a9eb9723bac377a66d9be1f820fd258dc84c5a68430fdbc 3011da7f697f3f121651f2aba342ebae6d8e251f0df0b2a9f0a295bcd27380e6 f5bdb7adab23ed72c7cc9a181c5b6ddb74d5273dfc9a819809815677f8069aac)
for index in "${!files[@]}"; do
    cp "${frontier_dir}/${files[$index]}" "${evidence_dir}/${files[$index]}"
    test "$(sha256sum "${evidence_dir}/${files[$index]}" | cut -d' ' -f1)" = "${hashes[$index]}"
done
if rg -n 'import f103_s3_noncommutative_phase_relation_algebra|from f103_s3_noncommutative_phase_relation_algebra' "${evidence_dir}/f103_s3_noncommutative_phase_relation_oracle.py"; then
    echo "oracle isolation violation" >&2
    exit 1
fi
PYTHONPYCACHEPREFIX=/dev/shm "${python_bin}" -m py_compile "${evidence_dir}"/*.py
"${python_bin}" "${evidence_dir}/f103_s3_noncommutative_phase_relation_algebra.py" --output "${evidence_dir}/production.rerun.json"
cmp "${evidence_dir}/F103_S3_NONCOMMUTATIVE_PHASE_RELATION_RESULTS.json" "${evidence_dir}/production.rerun.json"
"${python_bin}" "${evidence_dir}/f103_s3_noncommutative_phase_relation_oracle.py" "${evidence_dir}/production.rerun.json" --output "${evidence_dir}/oracle.rerun.json"
cmp "${evidence_dir}/F103_S3_NONCOMMUTATIVE_PHASE_RELATION_ORACLE_RESULTS.json" "${evidence_dir}/oracle.rerun.json"
jq -e '.classification=="SOURCE_AUDITED_PACKAGE_LOCAL" and .verification_level=="PACKAGE_SELF_REVIEW" and .restoration_classification=="EXACT_ALGEBRAIC_RESTORATION" and .experiment.case_count==12 and .experiment.native_composition=="NONABELIAN_S3_GROUP_CONVOLUTION" and .experiment.hidden_relation_values_serialized==false and .experiment.full6_by6_relation_materialized_on_accepted_path==false and .experiment.answer_bearing_lookup_table_materialized==false and ([.cases[]|.matches_identical_group_coordinate_classical_recurrence and .matches_irrep_convolution_classical_recurrence and .exact_canonical_state_restored and .same_backing_restored]|all) and ([.controls[]]|all) and ([.algebra_checks[]]|all) and .restoration_and_reuse.restoration_generation==2 and .restoration_and_reuse.same_backing_reused and .restoration_and_reuse.unrelated_second_boundary_matches_fresh and .resource_accounting.accepted_carrier_field_cells==12 and .resource_accounting.abstract_relation_state_working_field_cells_peak==18 and .resource_accounting.conservative_accepted_explicit_python_field_value_slots_peak==70 and .resource_accounting.matched_group_coordinate_classical_carrier_field_cells==12 and .resource_accounting.matched_group_coordinate_classical_abstract_working_field_cells_peak==18 and .resource_accounting.conservative_matched_group_coordinate_explicit_python_field_value_slots_peak==58 and .resource_accounting.irrep_decomposition_reduces_information_dimension==false and .resource_accounting.retained_inverse_history_field_cells==0 and .resource_accounting.advantage_claimed==false' "${evidence_dir}/production.rerun.json" >/dev/null
jq -e '.classification=="INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and .verification_level=="INDEPENDENT_ORACLE_REEXECUTION" and .restoration_classification=="EXACT_ALGEBRAIC_RESTORATION" and .imports_production==false and .imports_numpy==false and .algebra_oracle.composition_checks==36 and .algebra_oracle.hadamard_checks==36 and .algebra_oracle.all_outputs_translation_invariant and .algebra_oracle.noncommuting_products_differ and .algebra_oracle.irrep_convolution_checks==36 and .algebra_oracle.irrep_transform_rank==6 and .algebra_oracle.irrep_transform_compresses_below_group_order==false and .production_comparison.cases==12 and .production_comparison.all_match and ([.cases[]|.full72_cells_restore_exactly]|all) and .resource_scope.accepted_two_register_carrier_field_cells==12 and .resource_scope.oracle_full_two_register_relation_field_cells==72 and .resource_scope.answer_bearing_lookup_table_cells==0' "${evidence_dir}/oracle.rerun.json" >/dev/null
echo "QUALIFIED_F103_S3_NONCOMMUTATIVE_PHASE_RELATION_ALGEBRA_STRICT_SCOPE"
echo "evidence=${evidence_dir}"
