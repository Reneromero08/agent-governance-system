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
    echo "RAM-backed M252 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M252 filesystem forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
service="$here/catvm_coherent_order_commutator_service.py"
client="$here/catvm_coherent_order_commutator_client.py"
reference="$here/catvm_coherent_order_commutator_separate_reference.py"
qualifier="$here/qualify_catvm_coherent_order_commutator.sh"
sealed_raw="$here/CATVM_COHERENT_ORDER_COMMUTATOR_RAW_RESULTS.json"
sealed_ref="$here/CATVM_COHERENT_ORDER_COMMUTATOR_SEPARATE_REFERENCE.json"
sealed_result="$here/CATVM_COHERENT_ORDER_COMMUTATOR_RESULTS.json"
raw="$build/CATVM_COHERENT_ORDER_COMMUTATOR_RAW_RESULTS.json"
ref="$build/CATVM_COHERENT_ORDER_COMMUTATOR_SEPARATE_REFERENCE.json"
result="$build/CATVM_COHERENT_ORDER_COMMUTATOR_RESULTS.json"
service_stdout="$build/CATVM_COHERENT_ORDER_COMMUTATOR_SERVICE.stdout"
service_stderr="$build/CATVM_COHERENT_ORDER_COMMUTATOR_SERVICE.stderr"

mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
run_env=(
  env
  TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp"
  XDG_CACHE_HOME="$build/xdg-cache"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache"
)

public_config=$(${run_env[@]} python3 - <<'PY'
import json
primary={"u":"X","v":"Z"}
reuse={"u":"H","v":"T"}
cases={
 "primary":{"carrier_id":"m252-primary","descriptor":primary},
 "reuse":{"carrier_id":"m252-primary","descriptor":reuse},
 "fresh":{"carrier_id":"m252-fresh","descriptor":reuse},
}
for name in ("disconnect","partial","postprojection","descriptor_control"):
 cases[name]={"carrier_id":f"m252-{name}","descriptor":dict(primary)}
print(json.dumps({"cases":cases},sort_keys=True,separators=(",",":")))
PY
)
public_config_bytes=$(( ${#public_config} + 1 ))
reference_config='{"suite":"M252_COHERENT_ORDER_COMMUTATOR_STRICT_SCOPE"}'
reference_config_bytes=$(( ${#reference_config} + 1 ))
socket_name="@catvm-m252-$$"

service_pid=""
cleanup_service() {
  if [[ -n "$service_pid" ]] && kill -0 "$service_pid" 2>/dev/null; then
    kill "$service_pid"
    wait "$service_pid" || true
  fi
}
trap cleanup_service EXIT

printf '%s\n' '{"service":"M252_COHERENT_ORDER_COMMUTATOR_MODE"}' | \
  "${run_env[@]}" nice -n 10 ionice -c 3 \
  python3 "$service" "$socket_name" >"$service_stdout" 2>"$service_stderr" &
service_pid=$!
ready=false
for _ in $(seq 1 160); do
  if "${run_env[@]}" python3 - "$socket_name" 2>/dev/null <<'PY'
import socket,sys
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
try: s.connect("\0"+sys.argv[1][1:])
except OSError: raise SystemExit(1)
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
  echo "M252 service did not become ready" >&2
  exit 2
fi

printf '%s\n' "$public_config" | "${run_env[@]}" nice -n 10 ionice -c 3 \
  python3 "$client" "$socket_name" >"$raw"
wait "$service_pid"
service_pid=""
[[ ! -s "$service_stdout" ]]
[[ ! -s "$service_stderr" ]]
printf '%s\n' "$reference_config" | "${run_env[@]}" nice -n 10 ionice -c 3 \
  python3 "$reference" >"$ref"

"${run_env[@]}" python3 - \
  "$raw" "$ref" "$result" "$service" "$client" "$reference" "$qualifier" \
  "$public_config_bytes" "$reference_config_bytes" <<'PY'
import ast,hashlib,json,sys
from pathlib import Path

raw_path,ref_path,result_path=map(Path,sys.argv[1:4])
source_paths=list(map(Path,sys.argv[4:8]))
public_bytes=int(sys.argv[8]); reference_bytes=int(sys.argv[9])
raw=json.loads(raw_path.read_text()); ref=json.loads(ref_path.read_text())

def key(case): return (tuple(case["pair"]),case["run_kind"])
raw_cases={key(case):case for case in raw["cases"]}
ref_cases={key(case):case for case in ref["cases"]}
if set(raw_cases)!=set(ref_cases): raise SystemExit("M252 case key mismatch")
for case_key in raw_cases:
    production=raw_cases[case_key]; oracle=ref_cases[case_key]
    for field in (
        "pair","generation","commutator_boundary","hidden_branch_field_cells",
        "hidden_scratch_field_cells","hidden_order_consumer_receipt_cells",
        "retained_final_boundary_field_cells_during_inverse",
        "same_branch_scratch_and_receipt_backings","canonical_after_restoration",
        "baseline_reload_used","work",
    ):
        if production[field]!=oracle[field]:
            raise SystemExit(f"M252 independent case mismatch {case_key} {field}")

if not all(raw["controls"].values()) or not all(ref["controls"].values()):
    raise SystemExit("M252 control failure")
if not all(ref["reuse_parity"].values()) or raw["reuse_parity"] is not True:
    raise SystemExit("M252 reuse parity failure")
expected={
 (("X","Z"),"PRIMARY"):[[-1,1],[0,1],[0,1],[0,1]],
 (("H","T"),"REUSE"):[[1,2],[1,4],[0,1],[-1,4]],
 (("H","T"),"FRESH"):[[1,2],[1,4],[0,1],[-1,4]],
}
if any(raw_cases[item]["commutator_boundary"]!=value for item,value in expected.items()):
    raise SystemExit("M252 exact boundary mismatch")
if not all(
    case["same_branch_scratch_and_receipt_backings"]
    and case["canonical_after_restoration"] and not case["baseline_reload_used"]
    for case in raw["cases"]
): raise SystemExit("M252 transaction invariant failure")

law=raw["coherent_order_law"]
if law!={
 "hidden_order_branch_count":2,
 "same_target_consumed_in_orders":["VU","UV"],
 "final_boundary_is_order_commutator_expectation":True,
 "order_port_remains_unprojected_until_final_overlap":True,
 "order_coherence_is_causally_required_for_boundary":True,
 "route_disposition":"RETIRE_AFTER_ONE_BOUNDED_GATE_PAIR_SUITE_IF_STREAMED_CLASSICAL_RECURRENCE_MATCHES",
}: raise SystemExit("M252 coherent-order law mismatch")

resource=raw["resource_law"]
required={
 "declared_public_gate_pairs":[["X","Z"],["H","T"],["H","T"]],
 "accepted_persistent_carriers":2,
 "accepted_hidden_branch_field_cells_per_carrier":4,
 "accepted_hidden_scratch_field_cells_per_carrier":2,
 "accepted_hidden_order_consumer_receipt_cells_per_carrier":2,
 "accepted_transactions":3,
 "service_static_public_gate_library_field_cells":16,
 "accepted_compiled_public_gate_plan_matrix_references":12,
 "accepted_forward_branch_gate_actions":12,
 "accepted_inverse_branch_gate_actions":12,
 "accepted_forward_field_multiply_terms":48,
 "accepted_inverse_field_multiply_terms":48,
 "accepted_forward_field_accumulations":48,
 "accepted_inverse_field_accumulations":48,
 "accepted_forward_branch_field_writes":24,
 "accepted_inverse_branch_field_writes":24,
 "accepted_forward_scratch_clear_writes":24,
 "accepted_inverse_scratch_clear_writes":24,
 "accepted_boundary_field_multiply_terms":9,
 "accepted_boundary_field_accumulations":6,
 "retained_final_boundary_field_cells_during_inverse_per_transaction":1,
 "retained_dynamic_inverse_history_entries":0,
 "strongest_fixed_fixture_classical_baseline":"PUBLIC_PAIR_VALIDATION_PLUS_FROZEN_EXACT_COMMUTATOR_BOUNDARY_IN_O1_WORK",
 "strongest_transferable_descriptor_level_classical_baseline":"DIRECT_EXACT_ONE_TWO_COMPONENT_QZETA8_VECTOR_COMMUTATOR_WORD_V_THEN_U_THEN_V_DAGGER_THEN_U_DAGGER_WITH2_RESIDENT_FIELD_CELLS_PLUS2_REUSABLE_SCRATCH_CELLS_AND_NO_CATVM_RESTORATION",
 "transferable_baseline_resident_target_field_cells":2,
 "transferable_baseline_reusable_scratch_field_cells":2,
 "transferable_baseline_gate_actions_per_case":4,
 "transferable_baseline_field_multiply_terms_per_case":16,
 "accepted_catvm_path_has_space_work_or_query_advantage":False,
}
for name,value in required.items():
    if resource.get(name)!=value: raise SystemExit(f"M252 resource mismatch {name}")
if resource["resource_verification_level"]!="PACKAGE_SELF_REVIEW":
    raise SystemExit("M252 resource verification level mismatch")
if resource["whole_transaction_live_payload_peak_complete"] is not False:
    raise SystemExit("M252 global live-payload overclaim")
if any(raw["claim_limits"].values()): raise SystemExit("M252 claim-limit overreach")
if raw["claim_ceiling"]!="EXACT_SOFTWARE_ONE_HIDDEN_TWO_BRANCH_ORDER_PORT_ONE_FIXED_ZERO_TARGET_QUBIT_PUBLIC_GATE_PAIRS_FROM_X_Z_H_T_ON_AN_ABSTRACT_UNIX_SOCKET_CATVM_ONLY":
    raise SystemExit("M252 claim ceiling mismatch")

client_tree=ast.parse(source_paths[1].read_text())
reference_tree=ast.parse(source_paths[2].read_text())
client_imports={node.module for node in ast.walk(client_tree) if isinstance(node,ast.ImportFrom)}
client_imports.update(alias.name for node in ast.walk(client_tree) if isinstance(node,ast.Import) for alias in node.names)
reference_imports={node.module for node in ast.walk(reference_tree) if isinstance(node,ast.ImportFrom)}
reference_imports.update(alias.name for node in ast.walk(reference_tree) if isinstance(node,ast.Import) for alias in node.names)
if any("coherent_order_commutator_service" in (name or "") for name in client_imports|reference_imports):
    raise SystemExit("M252 controller/reference imports backend")
for path in source_paths:
    ast.parse(path.read_text()) if path.suffix==".py" else None

dependencies={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
result=dict(raw)
result["independent_reference"]={
 "result":ref["result"],
 "oracle_law":ref["oracle_law"],
 "all_controls_pass":all(ref["controls"].values()),
 "all_reuse_parity_checks_pass":all(ref["reuse_parity"].values()),
}
result["source_dependencies"]=dependencies
result["input_provenance"]={
 "public_controller_configuration_bytes":public_bytes,
 "standalone_reference_configuration_bytes":reference_bytes,
 "controller_received_only_public_gate_pairs":True,
 "backend_private_mode_delivered_only_to_service_stdin":True,
 "standalone_oracle_mode_delivered_only_to_reference_stdin":True,
}
result_path.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
PY

if [[ "${M252_WRITE_SEALS:-0}" == 1 ]]; then
  cp -- "$raw" "$sealed_raw"
  cp -- "$ref" "$sealed_ref"
  cp -- "$result" "$sealed_result"
fi
cmp -s "$raw" "$sealed_raw"
cmp -s "$ref" "$sealed_ref"
cmp -s "$result" "$sealed_result"
printf '%s\n' "QUALIFIED_CATVM_COHERENT_ORDER_COMMUTATOR_STRICT_SCOPE"
