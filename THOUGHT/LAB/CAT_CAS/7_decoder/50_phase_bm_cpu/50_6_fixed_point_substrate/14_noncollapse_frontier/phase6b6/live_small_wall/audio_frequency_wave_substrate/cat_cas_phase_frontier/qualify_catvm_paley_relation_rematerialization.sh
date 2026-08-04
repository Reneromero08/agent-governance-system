#!/usr/bin/env bash
set -euo pipefail
frontier_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(git -C "${frontier_dir}" rev-parse --show-toplevel)"
python_bin="${repo_dir}/.venv/bin/python"
evidence_dir="$(mktemp -d /dev/shm/ags-audio-m165-qualifier-XXXXXX)"
files=(catvm_paley_relation_protocol.py catvm_paley_relation_rematerialization_service.py catvm_paley_relation_rematerialization_controller.py catvm_paley_relation_rematerialization_oracle.py CATVM_PALEY_RELATION_REMATERIALIZATION_RESULTS.json CATVM_PALEY_RELATION_REMATERIALIZATION_ORACLE_RESULTS.json CATVM_PALEY_RELATION_REMATERIALIZATION_INDEPENDENT_REEXECUTION_REVIEW.md)
hashes=(5772a5cef316dfdd04deb728bf5579ed1d2f91f6c4bd4b895355b6dfc0953844 b296e272faeb576e44e997251622bfb392e17068fbb306b994735fc1026fbfcb 3bc9918b928d46e2d8896b430ce04bce427c45008d923b071c8190120da09fd8 351cb1dc37830f872b18284ef0ab737678627e330ea10eb1170d5b558799d1fc e971e9cb36544a0d154b780861fcd5f45c3a712a8a7b13d40a72399cf3030076 34bc369e58a8641296a531ff9c648a97e56aa1ca6d33dbe1dc95d5e0efcddfff 074687d9d6b81389ef2e099cf8422abc4fb1ba891329d42715b5cceba48f74ae)
for index in "${!files[@]}"; do
    cp "${frontier_dir}/${files[$index]}" "${evidence_dir}/${files[$index]}"
    test "$(sha256sum "${evidence_dir}/${files[$index]}" | cut -d' ' -f1)" = "${hashes[$index]}"
done
if rg -n 'import catvm_paley_relation_rematerialization_service|from catvm_paley_relation_rematerialization_service' "${evidence_dir}/catvm_paley_relation_rematerialization_controller.py"; then
    echo "controller backend import" >&2
    exit 1
fi
if rg -n 'import catvm_paley_relation_rematerialization_(service|controller)|from catvm_paley_relation_rematerialization_(service|controller)' "${evidence_dir}/catvm_paley_relation_rematerialization_oracle.py"; then
    echo "oracle isolation violation" >&2
    exit 1
fi
PYTHONPYCACHEPREFIX=/dev/shm "${python_bin}" -m py_compile "${evidence_dir}"/*.py
run_dir="${evidence_dir}/run"
mkdir "${run_dir}"
(
    cd "${evidence_dir}"
    "${python_bin}" catvm_paley_relation_rematerialization_controller.py catvm_paley_relation_rematerialization_service.py "${run_dir}" --output production.rerun.json
    cmp CATVM_PALEY_RELATION_REMATERIALIZATION_RESULTS.json production.rerun.json
    "${python_bin}" catvm_paley_relation_rematerialization_oracle.py production.rerun.json --output oracle.rerun.json
    cmp CATVM_PALEY_RELATION_REMATERIALIZATION_ORACLE_RESULTS.json oracle.rerun.json
)
jq -e '.classification=="SOURCE_AUDITED_PACKAGE_LOCAL" and .controller_imports_backend==false and .controller_computes_boundary==false and .primary.forward_toggle_actions==15 and .primary.internal_relation_capacity==6 and .primary.carrier_field_cells==24 and .primary.fresh_restored_reuse_boundary_parity and .primary.response_write_after_restoration_for_each_transaction and ([.denial_controls[]]|all) and ([.inverse_controls[]|.restoration_failure_withheld_response and .mutation_executed_after_forward and (.stderr_bytes==0)]|all) and .disconnect_control.restoration_precedes_disconnected_write_attempt and (.disconnect_control.stderr_bytes==0) and .no_smuggle.response_contains_intermediate_relation_cells==false and .resource_accounting.retained_dynamic_inverse_history_records==0 and .resource_accounting.snapshot_bytes==0' "${evidence_dir}/production.rerun.json" >/dev/null
jq -e '.classification=="INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and .imports_service==false and .imports_controller==false and .imports_numpy==false and .public_topology_scheduler.minimum_clean_local_reversible_pebble_capacity==6 and .public_topology_scheduler.capacities_1_through_5_exhausted_without_clean_sink_state and .public_topology_scheduler.forward_toggle_count==15 and .public_topology_scheduler.compiler_reads_relation_VALUES_or_BOUNDARY==false and ([.full_c17_oracle[]|.full136_cells_restore_exactly]|all) and .boundary_parity.primary and .boundary_parity.alternate_and_restored_reuse and .matched_compact_classical_baselines.accepted_storage_advantage_over_strongest_executed_compact_baseline==false and ([.matched_compact_classical_baselines.occurrence_expanded_compact_recurrence[]|(.field_cells==15 and .relation_evaluations==13)]|all) and .resource_scope.accepted_compact_relation_field_cells==24 and .resource_scope.occurrence_expanded_compact_relation_field_cells==15' "${evidence_dir}/oracle.rerun.json" >/dev/null
echo "QUALIFIED_CATVM_PALEY_RELATION_REMATERIALIZATION_STRICT_SCOPE"
echo "evidence=${evidence_dir}"
