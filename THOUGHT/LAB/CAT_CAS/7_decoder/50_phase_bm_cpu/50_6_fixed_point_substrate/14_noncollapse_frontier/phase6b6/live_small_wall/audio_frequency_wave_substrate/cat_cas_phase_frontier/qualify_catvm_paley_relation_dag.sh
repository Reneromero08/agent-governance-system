#!/usr/bin/env bash
set -euo pipefail
frontier_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(git -C "${frontier_dir}" rev-parse --show-toplevel)"
python_bin="${repo_dir}/.venv/bin/python"
evidence_dir="$(mktemp -d /dev/shm/ags-audio-m164-qualifier-XXXXXX)"
files=(catvm_paley_relation_protocol.py catvm_paley_relation_service.py catvm_paley_relation_controller.py catvm_paley_relation_oracle.py CATVM_PALEY_RELATION_DAG_RESULTS.json CATVM_PALEY_RELATION_DAG_ORACLE_RESULTS.json CATVM_PALEY_RELATION_DAG_INDEPENDENT_REEXECUTION_REVIEW.md)
hashes=(5772a5cef316dfdd04deb728bf5579ed1d2f91f6c4bd4b895355b6dfc0953844 b6d0c9841dc973d67ff2e26b8920f373b64119e7183fa592c65dd9ba8b43b6b5 c284e963dd51324625750e8e925643d9aba84ef090264be6810695e0d8e71097 009a1ccd753fc8402601853fa1330cf92a2f7250ef3cf88645f9ebaab7c48732 5af21c6843cdbfb033ac1be524880e1dda179c7ed63e82a6ee93e2edc2708551 91e4a6fa1a5d82defeed273f1c59859a3ef5e6e7a461c49cd2a5264a7f89d2a3 9db7ecb10093d38fa974d9c292290b638476f84a723a1806cc4c6cac172d8418)
for index in "${!files[@]}"; do cp "${frontier_dir}/${files[$index]}" "${evidence_dir}/${files[$index]}"; test "$(sha256sum "${evidence_dir}/${files[$index]}" | cut -d' ' -f1)" = "${hashes[$index]}"; done
if rg -n 'import catvm_paley_relation_service|from catvm_paley_relation_service' "${evidence_dir}/catvm_paley_relation_controller.py"; then echo "controller backend import" >&2; exit 1; fi
if rg -n 'import catvm_paley_relation_(service|controller)|from catvm_paley_relation_(service|controller)' "${evidence_dir}/catvm_paley_relation_oracle.py"; then echo "oracle isolation violation" >&2; exit 1; fi
run_dir="${evidence_dir}/run"; mkdir "${run_dir}"
"${python_bin}" "${evidence_dir}/catvm_paley_relation_controller.py" "${evidence_dir}/catvm_paley_relation_service.py" "${run_dir}" --output "${evidence_dir}/production.rerun.json"
cmp "${evidence_dir}/CATVM_PALEY_RELATION_DAG_RESULTS.json" "${evidence_dir}/production.rerun.json"
"${python_bin}" "${evidence_dir}/catvm_paley_relation_oracle.py" "${evidence_dir}/production.rerun.json" --output "${evidence_dir}/oracle.rerun.json"
cmp "${evidence_dir}/CATVM_PALEY_RELATION_DAG_ORACLE_RESULTS.json" "${evidence_dir}/oracle.rerun.json"
jq -e '.classification=="SOURCE_AUDITED_PACKAGE_LOCAL" and .controller_imports_backend==false and .controller_computes_boundary==false and .primary.restoration_generation==2 and .primary.fresh_restored_reuse_boundary_parity and .primary.response_write_after_restoration_for_each_transaction and ([.denial_controls[]]|all) and ([.inverse_controls[]|.mutation_rejected_without_response and .audit_contains_only_rejection_event and (.stderr_bytes==0)]|all) and .disconnect_control.restoration_precedes_disconnected_write_attempt and (.disconnect_control.stderr_bytes==0) and .no_smuggle.response_contains_intermediate_relation_cells==false and .resource_accounting.carrier_field_cells==27 and .resource_accounting.retained_inverse_history_field_cells==0 and .resource_accounting.snapshot_bytes==0' "${evidence_dir}/production.rerun.json" >/dev/null
jq -e '.classification=="INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and .imports_service==false and .imports_controller==false and .imports_numpy==false and .boundary_parity.primary and .boundary_parity.alternate_and_reuse and ([.topologies[]|.full153_cells_restore_exactly]|all) and .resource_scope.oracle_full_relation_cells==153 and .resource_scope.accepted_paley_relation_cells==27' "${evidence_dir}/oracle.rerun.json" >/dev/null
echo "QUALIFIED_CATVM_PALEY_RELATION_DAG_STRICT_SCOPE"
echo "evidence=${evidence_dir}"
