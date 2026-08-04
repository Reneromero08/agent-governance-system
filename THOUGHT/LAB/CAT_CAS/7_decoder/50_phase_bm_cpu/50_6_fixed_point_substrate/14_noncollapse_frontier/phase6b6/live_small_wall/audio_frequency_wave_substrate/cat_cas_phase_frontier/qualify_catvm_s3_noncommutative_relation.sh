#!/usr/bin/env bash
set -euo pipefail
frontier_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(git -C "${frontier_dir}" rev-parse --show-toplevel)"
python_bin="${repo_dir}/.venv/bin/python"
evidence_dir="$(mktemp -d /dev/shm/ags-audio-m167-qualifier-XXXXXX)"
production_evidence="${evidence_dir}/production-evidence"
oracle_evidence="${evidence_dir}/oracle-evidence"
mkdir -p "${production_evidence}" "${oracle_evidence}"
files=(catvm_s3_relation_protocol.py catvm_s3_relation_service.py catvm_s3_relation_controller.py catvm_s3_relation_oracle.py CATVM_S3_NONCOMMUTATIVE_RELATION_RESULTS.json CATVM_S3_NONCOMMUTATIVE_RELATION_ORACLE_RESULTS.json CATVM_S3_NONCOMMUTATIVE_RELATION_INDEPENDENT_REEXECUTION_REVIEW.md)
hashes=(b59852d800f6a7a4e793a0282354198fc06a4e1bbdf73d63278b01e389dbd217 a16eea2411b2855c5dd467a660bf6e0e33dcebe723e32bd8b3ae1d5b2a79f081 4db40e94852592454859ac9def320ff8d411d18a928e8ede0be3b97c32ae083f 447baf8c95936aec542810f279c398370106e0248680293681b6f0e64ebee2e9 1210b0339ad70bf30767a4ef823e95777b361e8ea1d35f07b853575563936699 bd94259835bc1d68ee4ace84bb1a0f6b20a9cabd264c6e256293cf70c072e1c3 989c2d4eeece8e7ed657632e541ecd4387f2040db0878943a008ffd3c7b3acaa)
for index in "${!files[@]}"; do
    cp "${frontier_dir}/${files[$index]}" "${evidence_dir}/${files[$index]}"
    test "$(sha256sum "${evidence_dir}/${files[$index]}" | cut -d' ' -f1)" = "${hashes[$index]}"
done
if rg -n 'catvm_s3_relation_(service|controller|oracle)' "${evidence_dir}/catvm_s3_relation_controller.py"; then
    echo "controller boundary import violation" >&2
    exit 1
fi
if rg -n 'catvm_s3_relation_(protocol|service|controller)' "${evidence_dir}/catvm_s3_relation_oracle.py"; then
    echo "oracle isolation violation" >&2
    exit 1
fi
PYTHONPYCACHEPREFIX=/dev/shm "${python_bin}" -m py_compile "${evidence_dir}"/*.py
"${python_bin}" "${evidence_dir}/catvm_s3_relation_controller.py" \
    "${evidence_dir}/catvm_s3_relation_service.py" \
    "${production_evidence}" \
    --output "${evidence_dir}/production.rerun.json"
cmp "${evidence_dir}/CATVM_S3_NONCOMMUTATIVE_RELATION_RESULTS.json" "${evidence_dir}/production.rerun.json"
"${python_bin}" "${evidence_dir}/catvm_s3_relation_oracle.py" \
    "${evidence_dir}/catvm_s3_relation_service.py" \
    "${evidence_dir}/production.rerun.json" \
    "${oracle_evidence}" \
    --output "${evidence_dir}/oracle.rerun.json"
cmp "${evidence_dir}/CATVM_S3_NONCOMMUTATIVE_RELATION_ORACLE_RESULTS.json" "${evidence_dir}/oracle.rerun.json"
jq -e '
    .classification=="SOURCE_AUDITED_PACKAGE_LOCAL" and
    .verification_level=="PACKAGE_SELF_REVIEW" and
    .restoration_classification=="EXACT_ALGEBRAIC_RESTORATION" and
    .machine_boundary=="SEPARATE_NON_DUMPABLE_NO_NEW_PRIVS_SAME_UID_LINUX_BINARY_PIPE_SERVICE" and
    .controller_imports_backend==false and
    .controller_computes_boundary==false and
    .inplace.carrier_field_cells==12 and
    .inplace.restoration_generation==2 and
    .inplace.same_service_process_reused and
    .inplace.response_write_after_restoration_for_each_transaction and
    .inplace.fresh_restored_reuse_boundary_parity and
    .inplace.fresh_restored_reuse_receipt_parity and
    .inplace.process_memory_open_denied and
    ([.resident_stage_controls[] | .denied_without_boundary and .restored_before_denial_response and .process_memory_open_denied] | all) and
    ([.inverse_controls[] | .mutation_executed_after_forward_residency and .restoration_failure_withheld_response and .response_bytes==0 and .stderr_bytes==0 and .process_memory_open_denied] | all) and
    .disconnect_control.restoration_precedes_disconnected_write_attempt and
    .disconnect_control.process_memory_open_denied and
    .simple_denial_controls.null_carrier_denied and
    .simple_denial_controls.wrong_generation_denied and
    .simple_denial_controls.snapshot_command_on_inplace_service_denied and
    .snapshot_sham.classification=="SNAPSHOT_RELOAD" and
    .snapshot_sham.inplace_command_on_snapshot_service_denied and
    .snapshot_sham.snapshot_reload_precedes_response and
    .no_smuggle.response_contains_intermediate_relation_cells==false and
    .no_smuggle.audit_contains_relation_values==false and
    .no_smuggle.receipt_is_one_way_64_bit_commitment and
    .resource_accounting.accepted_carrier_field_cells==12 and
    .resource_accounting.temporary_restoration_verification_baseline_field_cells==12 and
    .resource_accounting.conservative_accepted_field_value_slots_peak==37 and
    .resource_accounting.matched_direct_classical_field_value_slots_peak==25 and
    .resource_accounting.retained_inverse_history_records==0 and
    .resource_accounting.retained_compiled_plan_records==0 and
    .resource_accounting.snapshot_cells_on_accepted_path==0 and
    (.not_established | index("DISTINCT_PHASE_RESOURCE"))!=null and
    (.not_established | index("COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING"))!=null
' "${evidence_dir}/production.rerun.json" >/dev/null
jq -e '
    .classification=="INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
    .verification_level=="INDEPENDENT_ORACLE_REEXECUTION" and
    .restoration_classification=="EXACT_ALGEBRAIC_RESTORATION" and
    .imports_controller==false and
    .imports_protocol==false and
    .imports_service==false and
    .imports_numpy==false and
    .algebra.all36_group_and_full_matrix_basis_products_checked==36 and
    .algebra.noncommuting_products_differ and
    .algebra.reordered_depth64_compact_inverse_fails and
    ([.semantic_cases[] | .compact_matches_full72_cells and .compact_restores_exactly and .full72_cells_restore_exactly] | all) and
    .production_boundary_and_receipt_fields_match and
    .black_box_service_reexecution.accepted_atomic_event_order and
    .black_box_service_reexecution.accepted_first_matches_oracle and
    .black_box_service_reexecution.accepted_reuse_matches_oracle and
    .black_box_service_reexecution.projection_denied_after_forward_and_restoration and
    .black_box_service_reexecution.reordered_inverse_fails_after_forward_without_response and
    .black_box_service_reexecution.snapshot_matches_oracle_but_is_reload and
    .black_box_service_reexecution.accepted_process_memory_denied and
    .black_box_service_reexecution.projection_process_memory_denied and
    .black_box_service_reexecution.reordered_process_memory_denied and
    .black_box_service_reexecution.snapshot_process_memory_denied and
    .resource_scope.accepted_conservative_field_value_slots_peak==37 and
    .resource_scope.matched_streamed_group_classical_field_value_slots_peak==25 and
    .resource_scope.oracle_full_relation_field_cells==72 and
    .resource_scope.physical_process_peak_unmeasured
' "${evidence_dir}/oracle.rerun.json" >/dev/null
echo "QUALIFIED_CATVM_S3_NONCOMMUTATIVE_RELATION_STRICT_SCOPE"
echo "evidence=${evidence_dir}"
