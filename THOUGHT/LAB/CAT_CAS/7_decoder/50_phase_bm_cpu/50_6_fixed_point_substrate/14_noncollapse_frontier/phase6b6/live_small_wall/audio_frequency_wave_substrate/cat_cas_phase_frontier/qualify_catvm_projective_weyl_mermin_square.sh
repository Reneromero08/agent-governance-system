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
    echo "RAM-backed M250 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M250 filesystem forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
service="$here/catvm_projective_weyl_mermin_square_service.py"
client="$here/catvm_projective_weyl_mermin_square_client.py"
reference="$here/catvm_projective_weyl_mermin_square_separate_reference.py"
qualifier="$here/qualify_catvm_projective_weyl_mermin_square.sh"
sealed_raw="$here/CATVM_PROJECTIVE_WEYL_MERMIN_SQUARE_RAW_RESULTS.json"
sealed_ref="$here/CATVM_PROJECTIVE_WEYL_MERMIN_SQUARE_SEPARATE_REFERENCE.json"
sealed_result="$here/CATVM_PROJECTIVE_WEYL_MERMIN_SQUARE_RESULTS.json"
raw="$build/CATVM_PROJECTIVE_WEYL_MERMIN_SQUARE_RAW_RESULTS.json"
ref="$build/CATVM_PROJECTIVE_WEYL_MERMIN_SQUARE_SEPARATE_REFERENCE.json"
result="$build/CATVM_PROJECTIVE_WEYL_MERMIN_SQUARE_RESULTS.json"
service_stdout="$build/CATVM_PROJECTIVE_WEYL_MERMIN_SQUARE_SERVICE.stdout"
service_stderr="$build/CATVM_PROJECTIVE_WEYL_MERMIN_SQUARE_SERVICE.stderr"

mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
run_env=(
  env
  TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp"
  XDG_CACHE_HOME="$build/xdg-cache"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache"
)

public_config=$(${run_env[@]} python3 - <<'PY'
import json
cases={
 "primary":{"carrier_id":"m250-primary","descriptor":{"variant":"BASE"}},
 "reuse":{"carrier_id":"m250-primary","descriptor":{"variant":"H_CONJUGATED_REORDERED"}},
 "fresh":{"carrier_id":"m250-fresh","descriptor":{"variant":"H_CONJUGATED_REORDERED"}},
}
for name in ("disconnect","partial","postprojection","descriptor_control"):
 cases[name]={"carrier_id":f"m250-{name}","descriptor":{"variant":"BASE"}}
print(json.dumps({"cases":cases},sort_keys=True,separators=(",",":")))
PY
)
public_config_bytes=$(( ${#public_config} + 1 ))
reference_config='{"suite":"M250_PROJECTIVE_WEYL_MERMIN_STRICT_SCOPE"}'
reference_config_bytes=$(( ${#reference_config} + 1 ))
socket_name="@catvm-m250-$$"

service_pid=""
cleanup_service() {
  if [[ -n "$service_pid" ]] && kill -0 "$service_pid" 2>/dev/null; then
    kill "$service_pid"
    wait "$service_pid" || true
  fi
}
trap cleanup_service EXIT

printf '%s\n' '{"service":"M250_PROJECTIVE_WEYL_MERMIN_MODE"}' | \
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
  echo "M250 service did not become ready" >&2
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

def key(case): return (case["variant"],case["run_kind"])
raw_cases={key(case):case for case in raw["cases"]}
ref_cases={key(case):case for case in ref["cases"]}
if set(raw_cases)!=set(ref_cases): raise SystemExit("M250 case key mismatch")
for case_key in raw_cases:
    production=raw_cases[case_key]; oracle=ref_cases[case_key]
    for field in (
        "variant","generation","central_phase_exponent_mod4","central_phase",
        "observable_port_count","context_count","hidden_carrier_field_cells",
        "hidden_scratch_field_cells","hidden_context_signature_cells",
        "retained_final_boundary_field_cells_during_inverse",
        "same_carrier_and_custody_backings","canonical_after_restoration",
        "baseline_reload_used","work",
    ):
        if production[field]!=oracle[field]:
            raise SystemExit(f"M250 independent case mismatch {case_key} {field}")

if not all(raw["controls"].values()) or not all(ref["controls"].values()):
    raise SystemExit("M250 control failure")
if not all(
    case["central_phase_exponent_mod4"]==2
    and case["central_phase"]=={"real":[-1,1],"imag":[0,1]}
    for case in raw["cases"]
): raise SystemExit("M250 central phase mismatch")
if raw["reuse_parity"] is not True: raise SystemExit("M250 reuse parity failure")
if raw["contextuality_law"]!={
    "native_projective_two_cocycle_causally_changes_boundary":True,
    "typed_shared_observable_ports_have_two_consumers_each":True,
    "noncontextual_assignment_parity_product":1,
    "projective_context_product":-1,
    "assignment_enumeration_used":False,
    "carrier_state_coherence_required_for_operator_cocycle":False,
    "route_disposition":"RETIRE_AFTER_ONE_MERMIN_SQUARE_IF_IDENTICAL_COMPACT_SYMPLECTIC_RECURRENCE_MATCHES",
}: raise SystemExit("M250 contextuality law mismatch")

resource=raw["resource_law"]
required={
 "accepted_persistent_carriers":2,
 "accepted_hidden_carrier_field_cells_per_carrier":4,
 "accepted_hidden_scratch_field_cells_per_carrier":4,
 "accepted_hidden_context_signature_cells_per_carrier":6,
 "accepted_typed_observable_port_receipts_per_carrier":9,
 "accepted_contexts_per_transaction":6,
 "accepted_observable_consumptions_per_transaction":18,
 "accepted_transactions":3,
 "accepted_forward_pauli_actions":54,
 "accepted_inverse_pauli_actions":54,
 "accepted_forward_vector_cell_reads":216,
 "accepted_forward_vector_cell_writes":432,
 "accepted_inverse_vector_cell_reads":216,
 "accepted_inverse_vector_cell_writes":432,
 "accepted_signature_compositions_forward_and_inverse":[54,54],
 "accepted_final_overlap_field_multiplications":12,
 "retained_final_boundary_field_cells_during_inverse_per_transaction":1,
 "retained_dynamic_inverse_history_entries":0,
 "accepted_catvm_path_has_space_or_work_advantage":False,
 "whole_transaction_live_payload_peak_complete":False,
 "resource_verification_level":"PACKAGE_SELF_REVIEW",
}
for field,value in required.items():
    if resource.get(field)!=value: raise SystemExit(f"M250 resource mismatch {field}")
if resource["strongest_classical_baseline"]!="PUBLIC_VARIANT_VALIDATION_PLUS_FIXED_MERMIN_PARITY_COCYCLE_INVARIANT_RETURNING_CENTRAL_EXPONENT2_IN_O1_WORK":
    raise SystemExit("M250 classical baseline mismatch")
if resource["strongest_transferable_descriptor_level_classical_baseline"]!="IDENTICAL_BINARY_SYMPLECTIC_PROJECTIVE_2_COCYCLE_RECURRENCE_WITH_CONSTANT_SIGNATURE_STATE_AND18_PUBLIC_COMPOSITIONS":
    raise SystemExit("M250 transferable classical baseline mismatch")
if any(raw["claim_limits"].values()): raise SystemExit("M250 claim ceiling failure")
expected_independent_oracle={
 "q_i_matrix_arithmetic_reconstructed":True,
 "binary_symplectic_cocycle_reconstructed":True,
 "dense_four_by_four_context_and_transaction_oracle_executed":True,
 "dephased_density_operator_trace_executed":True,
 "assignment_enumeration_used":False,
 "production_source_imported":False,
}
if ref["independent_oracle"]!=expected_independent_oracle:
    raise SystemExit("M250 independent oracle failure")
if ref["oracle_resource_law"]!={
 "dense_matrix_oracle_is_verifier_only":True,
 "dense_matvec_scalar_terms_per_action":16,
 "accepted_transaction_semantic_matvec_terms_per_reference_case":576,
 "strongest_declared_family_classical_baseline":"PUBLIC_VARIANT_VALIDATION_PLUS_FIXED_MERMIN_PARITY_COCYCLE_INVARIANT_RETURNING_CENTRAL_EXPONENT2_IN_O1_WORK",
 "strongest_transferable_descriptor_level_classical_baseline":"BINARY_SYMPLECTIC_PROJECTIVE_2_COCYCLE_CONSTANT_SIGNATURE_STATE_WITH18_COMPOSITIONS",
 "both_declared_variants_proved_same_fixed_central_exponent":True,
}:
    raise SystemExit("M250 oracle resource baseline mismatch")

for path in source_paths[:3]: ast.parse(path.read_text(),filename=str(path))
service_source=source_paths[0].read_text(); client_source=source_paths[1].read_text(); ref_source=source_paths[2].read_text()
if "catvm_projective_weyl_mermin_square_service" in client_source or "catvm_projective_weyl_mermin_square_service" in ref_source:
    raise SystemExit("M250 backend imported by controller/reference")
if "itertools" in service_source or "itertools" in ref_source:
    raise SystemExit("M250 assignment enumeration helper forbidden")
if "stdin.close()" not in service_source or "libc.prctl(4, 0, 0, 0, 0)" not in service_source:
    raise SystemExit("M250 backend hardening missing")
if service_source.index("carrier.release()") > service_source.index('return {\n        "variant"'):
    raise SystemExit("M250 response can precede restoration")

forbidden={"vector","scratch_values","context_signatures","context_products","observable_values","amplitude_vector","dense_matrix","assignments","intermediate","port_values"}
def contains(value):
    if isinstance(value,dict): return any(str(k).lower() in forbidden or contains(v) for k,v in value.items())
    if isinstance(value,list): return any(contains(v) for v in value)
    return False
if any(contains(case) for case in raw["cases"]): raise SystemExit("M250 intermediate smuggle")

source_dependencies={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
raw["separate_reference"]={
 "result":ref["result"],"cases":ref["cases"],"controls":ref["controls"],
 "independent_oracle":ref["independent_oracle"],
 "oracle_resource_law":ref["oracle_resource_law"],
}
raw["source_dependencies"]=source_dependencies
raw["protocol_accounting"]["public_configuration_bytes"]=public_bytes
raw["protocol_accounting"]["reference_configuration_bytes"]=reference_bytes
raw["qualification"]={
 "service_stdout_empty":True,"service_stderr_empty":True,
 "controller_imports_no_backend":True,"reference_imports_no_production":True,
 "backend_nondumpable_checked":True,"backend_stdin_closed_after_startup":True,
 "response_released_after_restoration":True,
 "sealed_outputs_byte_reproducible":True,
 "disk_backed_build_directory":True,
 "no_assignment_enumeration":True,"no_intermediate_smuggle":True,
}
json.dump(raw,result_path.open("w"),sort_keys=True,indent=2); result_path.open("a").write("\n")
PY

cmp -s "$raw" "$sealed_raw"
cmp -s "$ref" "$sealed_ref"
cmp -s "$result" "$sealed_result"
echo "QUALIFIED_CATVM_PROJECTIVE_WEYL_MERMIN_SQUARE_STRICT_SCOPE"
