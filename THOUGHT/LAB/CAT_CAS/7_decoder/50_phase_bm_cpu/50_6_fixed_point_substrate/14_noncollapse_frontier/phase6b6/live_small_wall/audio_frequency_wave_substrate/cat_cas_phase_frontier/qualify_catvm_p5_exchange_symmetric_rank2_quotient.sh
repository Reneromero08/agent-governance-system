#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 || ! -d "$1" ]]; then
  echo "usage: $0 DISK_BACKED_BUILD_DIRECTORY" >&2
  exit 2
fi
build=$(realpath -e -- "$1")
case "$build" in
  /dev/shm|/dev/shm/*|/run/shm|/run/shm/*)
    echo "RAM-backed M245 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M245 filesystem forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
service="$here/catvm_p5_exchange_symmetric_rank2_quotient_service.py"
client="$here/catvm_p5_exchange_symmetric_rank2_quotient_client.py"
reference="$here/catvm_p5_exchange_symmetric_rank2_quotient_separate_reference.py"
qualifier="$here/qualify_catvm_p5_exchange_symmetric_rank2_quotient.sh"
field_dependency="$here/zeta5_normalized_cubic_fourier_coherent_port.py"
sealed_raw="$here/CATVM_P5_EXCHANGE_SYMMETRIC_RANK2_QUOTIENT_RAW_RESULTS.json"
sealed_ref="$here/CATVM_P5_EXCHANGE_SYMMETRIC_RANK2_QUOTIENT_SEPARATE_REFERENCE.json"
sealed_result="$here/CATVM_P5_EXCHANGE_SYMMETRIC_RANK2_QUOTIENT_RESULTS.json"
raw="$build/CATVM_P5_EXCHANGE_SYMMETRIC_RANK2_QUOTIENT_RAW_RESULTS.json"
ref="$build/CATVM_P5_EXCHANGE_SYMMETRIC_RANK2_QUOTIENT_SEPARATE_REFERENCE.json"
result="$build/CATVM_P5_EXCHANGE_SYMMETRIC_RANK2_QUOTIENT_RESULTS.json"
service_stdout="$build/CATVM_P5_EXCHANGE_SYMMETRIC_RANK2_QUOTIENT_SERVICE.stdout"
service_stderr="$build/CATVM_P5_EXCHANGE_SYMMETRIC_RANK2_QUOTIENT_SERVICE.stderr"

mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
run_env=(
  env
  TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp"
  XDG_CACHE_HOME="$build/xdg-cache"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache"
)

private_config=$(${run_env[@]} python3 - <<'PY'
import json

primary={
    "depth":4,
    "lambdas":[1,2,3,4],
    "quadratics":[0,1,3,2],
    "rungs":[1,2,4,3],
    "couplings":[2,3,4],
    "output_orbit":[1,3],
    "carrier_id":"shared_orbit15",
}
reuse={
    "depth":4,
    "lambdas":[2,4,1,3],
    "quadratics":[4,2,0,3],
    "rungs":[3,1,2,4],
    "couplings":[4,1,2],
    "output_orbit":[0,4],
    "carrier_id":"shared_orbit15",
}
fresh={**reuse,"carrier_id":"fresh_orbit15"}
oracles={"primary":primary,"reuse":reuse,"reuse_fresh":fresh}
disconnect={**primary,"carrier_id":"disconnect_orbit15","delay_before_inverse_ms":120}
oracles["disconnect_control"]=disconnect
exception={**reuse,"carrier_id":"exception_orbit15","inject_failure_after_projection":True}
oracles["exception_control"]=exception
partial={**primary,"carrier_id":"partial_orbit15","inject_failure_after_modules":2}
oracles["partial_exception_control"]=partial
print(json.dumps({"oracles":oracles},sort_keys=True,separators=(",",":")))
PY
)
private_config_bytes=$(( ${#private_config} + 1 ))
socket_name="@catvm-m245-$$"

service_pid=""
cleanup_service() {
  if [[ -n "$service_pid" ]] && kill -0 "$service_pid" 2>/dev/null; then
    kill "$service_pid"
    wait "$service_pid" || true
  fi
}
trap cleanup_service EXIT

printf '%s\n' "$private_config" | "${run_env[@]}" nice -n 10 ionice -c 3 \
  python3 "$service" "$socket_name" >"$service_stdout" 2>"$service_stderr" &
service_pid=$!
ready=false
for _ in $(seq 1 120); do
  if "${run_env[@]}" python3 - "$socket_name" <<'PY'
import socket,sys
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
try:
    s.connect("\0"+sys.argv[1][1:])
except OSError:
    raise SystemExit(1)
s.close()
PY
  then
    ready=true
    break
  fi
  sleep 0.01
done
if [[ "$ready" != true ]]; then
  wait "$service_pid" || true
  service_pid=""
  echo "M245 service did not become ready" >&2
  exit 2
fi

"${run_env[@]}" nice -n 10 ionice -c 3 python3 "$client" "$socket_name" >"$raw"
wait "$service_pid"
service_pid=""
[[ ! -s "$service_stdout" ]]
[[ ! -s "$service_stderr" ]]
printf '%s\n' "$private_config" | "${run_env[@]}" nice -n 10 ionice -c 3 \
  python3 "$reference" >"$ref"

"${run_env[@]}" python3 - \
  "$raw" "$ref" "$result" "$service" "$client" "$reference" "$qualifier" "$field_dependency" \
  "$private_config_bytes" <<'PY'
import hashlib,json,sys
from pathlib import Path

raw_path,ref_path,result_path=map(Path,sys.argv[1:4])
source_paths=list(map(Path,sys.argv[4:9]))
private_bytes=int(sys.argv[9])
raw=json.loads(raw_path.read_text())
ref=json.loads(ref_path.read_text())

raw_cases={case["run_kind"]:case for case in raw["cases"]}
ref_cases={case["run_kind"]:case for case in ref["cases"]}
if set(raw_cases)!=set(ref_cases):
    raise SystemExit("M245 case key mismatch")
for key in raw_cases:
    if raw_cases[key]["final_amplitude"]!=ref_cases[key]["final_amplitude"]:
        raise SystemExit(f"M245 independent boundary mismatch {key}")
    for field in ("generation","forward_character_terms","inverse_character_terms"):
        if raw_cases[key][field]!=ref_cases[key][field]:
            raise SystemExit(f"M245 independent case mismatch {key} {field}")

if not all(raw["controls"].values()) or not all(ref["controls"].values()):
    raise SystemExit("M245 control failure")
for certificate in ref["rank_certificates"].values():
    if not (
        certificate["symmetric_reachability_rank"]==15
        and certificate["symmetric_observability_rank"]==15
        and certificate["symmetric_hankel_rank"]==15
        and certificate["first_rung_labelled_matrix_rank"]==5
        and certificate["declared_exchange_broken_three_gate_reachability_rank"]==25
        and certificate["declared_exchange_broken_three_gate_observability_rank"]==25
    ):
        raise SystemExit("M245 rank certificate failure")

raw["separate_reference"]={
    "result":ref["result"],
    "factorized_endpoint_baseline":ref["factorized_endpoint_baseline"],
    "full25_labelled_parity":ref["full25_labelled_parity"],
    "rank_certificates":ref["rank_certificates"],
    "declared_exchange_broken_three_gate_alphabet":ref["declared_exchange_broken_three_gate_alphabet"],
    "controls":ref["controls"],
    "baseline":ref["baseline"],
    "imports_production_service_client_or_m237":ref["imports_production_service_client_or_m237"],
}
raw["private_configuration_accounting"]={
    "stdin_bytes_per_delivery":private_bytes,
    "deliveries":2,
    "recipients":["CATVM_SERVICE","SEPARATE_REFERENCE"],
    "controller_receives_private_configuration":False,
}
names=("service","controller","separate_reference","qualifier","exact_qzeta5_field_dependency")
raw["source_dependencies"]={
    f"{name}_sha256":hashlib.sha256(path.read_bytes()).hexdigest()
    for name,path in zip(names,source_paths)
}
raw["verification_statement"]={
    "production_and_independent15_orbit_boundaries_match":True,
    "full25_labelled_recurrence_matches15_orbit_quotient":True,
    "symmetric_reachable_observable_hankel_rank15_at_split_primes41_61":True,
    "declared_three_gate_exchange_broken_labelled_alphabet_rank25_at_split_primes41_61":True,
    "first_rung_labelled_matrix_rank5_at_split_primes41_61":True,
    "strongest_endpoint_factorized_classical_boundary_matches":True,
    "secret_dependent_intermediate_payload_metrics_not_delivered_or_sealed":True,
    "controller_imports_backend_or_field_code":False,
    "atomic_response_release_after_restoration":True,
}
with result_path.open("w") as handle:
    json.dump(raw,handle,sort_keys=True,indent=2)
    handle.write("\n")
PY

cmp -s "$raw" "$sealed_raw"
cmp -s "$ref" "$sealed_ref"
cmp -s "$result" "$sealed_result"

"${run_env[@]}" python3 - "$sealed_result" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["result"]=="PASS_CATVM_P5_EXCHANGE_SYMMETRIC_RANK2_ORBIT_QUOTIENT_STRICT_SCOPE"
assert r["classification"]=="INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
assert r["verification_level"]=="SEPARATE_REFERENCE_PARITY"
assert r["restoration_classification"]=="EXACT_ALGEBRAIC_RESTORATION"
assert all(r["controls"].values())
assert r["resource_law"]["accepted_orbit_message_field_cells"]==15
assert r["resource_law"]["accepted_orbit_scratch_field_cells"]==15
assert r["resource_law"]["accepted_total_message_plus_scratch_field_cells"]==30
assert r["resource_law"]["hidden_descriptor_residue_cells"]==17
assert r["resource_law"]["suite_service_hidden_configuration_descriptor_residue_cells"]==102
assert r["resource_law"]["suite_service_carrier_descriptor_residue_cells"]==85
assert r["resource_law"]["suite_service_configuration_plus_carrier_descriptor_residue_cells"]==187
assert r["resource_law"]["suite_service_phase_message_plus_scratch_field_cells"]==150
assert r["resource_law"]["suite_service_startup_descriptor_conversion_and_nonzero_validation_residue_reads"]==168
assert r["resource_law"]["forward_character_terms"]==1140
assert r["resource_law"]["inverse_character_terms"]==1500
assert r["resource_law"]["accepted_complete_transaction_character_terms"]==2640
assert r["resource_law"]["accepted_complete_transaction_root_field_multiplications"]==2625
assert r["resource_law"]["accepted_complete_transaction_field_accumulations"]==2625
assert r["resource_law"]["classical_endpoint_specialized_forward_character_terms"]==440
assert r["resource_law"]["classical_orbit_resident_field_cells"]==15
assert r["resource_law"]["classical_factorized_temporary_field_cells"]==25
assert r["resource_law"]["identical15_orbit_classical_forward_character_terms"]==1140
assert r["resource_law"]["identical15_orbit_classical_message_plus_scratch_field_cells"]==30
assert r["resource_law"]["time_memory_pareto_not_total_advantage"] is True
assert r["resource_law"]["fixed_bounded_width_exact_state"] is False
assert r["resource_law"]["kernel_transient_field_values_and_integer_reduction_scratch_in30_backing_cells"] is False
assert r["resource_law"]["canonicalization_scan_and_integer_division_work_instrumented"] is False
assert r["resource_law"]["secret_dependent_intermediate_payload_metrics_released"] is False
assert r["resource_law"]["secret_dependent_intermediate_payload_metrics_sealed"] is False
assert r["resource_law"]["dense25_state_materialized_on_accepted_path"] is False
assert r["resource_law"]["transfer_matrix_materialized_on_accepted_path"] is False
assert r["private_configuration_accounting"]["controller_receives_private_configuration"] is False
assert r["separate_reference"]["imports_production_service_client_or_m237"] is False
assert set(r["separate_reference"]["declared_exchange_broken_three_gate_alphabet"])=={"A","B","D"}
assert all(r["separate_reference"]["controls"].values())
assert r["verification_statement"]["controller_imports_backend_or_field_code"] is False
assert not any(r["claim_limits"].values())
assert len(r["cases"])==3
for case in r["cases"]:
    assert case["canonical_after_restoration"]
    assert case["same_cell_backing"] and case["same_scratch_backing"] and case["same_descriptor_backings"]
    assert not case["baseline_reload_used"]
    assert case["dense25_state_materializations"]==0
    assert case["transfer_matrices_materialized"]==0
    assert case["retained_dynamic_inverse_history_entries"]==0
    assert case["forward_character_terms"]==1140
    assert case["inverse_character_terms"]==1500
    assert "cells" not in case and "scratch" not in case
for certificate in r["separate_reference"]["rank_certificates"].values():
    assert certificate["symmetric_reachability_rank"]==15
    assert certificate["symmetric_observability_rank"]==15
    assert certificate["symmetric_hankel_rank"]==15
    assert certificate["first_rung_labelled_matrix_rank"]==5
    assert certificate["declared_exchange_broken_three_gate_reachability_rank"]==25
    assert certificate["declared_exchange_broken_three_gate_observability_rank"]==25
PY

"${run_env[@]}" python3 - "$client" <<'PY'
import ast,sys
tree=ast.parse(open(sys.argv[1]).read())
allowed={"__future__","json","socket","sys","time","typing"}
for node in ast.walk(tree):
    if isinstance(node,ast.Import):
        names={alias.name.split('.')[0] for alias in node.names}
    elif isinstance(node,ast.ImportFrom):
        names={str(node.module).split('.')[0]}
    else:
        continue
    if not names<=allowed:
        raise SystemExit(f"M245 controller imports nonpublic dependency: {names-allowed}")
PY

if rg -n 'catvm_p5_exchange_symmetric_rank2_quotient_service|zeta5_normalized_cubic_fourier_coherent_port|importlib|SourceFileLoader|runpy' "$client"; then
  echo "M245 controller imports or names backend code" >&2
  exit 2
fi
if rg -n 'itertools\.product|assignment_table|transfer_matrix\s*=|labelled25' "$service"; then
  echo "M245 accepted service contains forbidden dense/table construction" >&2
  exit 2
fi

printf '%s\n' QUALIFIED_CATVM_P5_EXCHANGE_SYMMETRIC_RANK2_ORBIT_QUOTIENT_STRICT_SCOPE
