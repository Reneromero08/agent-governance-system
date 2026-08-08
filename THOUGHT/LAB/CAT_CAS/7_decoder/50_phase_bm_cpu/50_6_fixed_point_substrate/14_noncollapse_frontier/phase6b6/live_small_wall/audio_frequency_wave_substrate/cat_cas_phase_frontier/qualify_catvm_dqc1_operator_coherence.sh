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
    echo "RAM-backed M251 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M251 filesystem forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
service="$here/catvm_dqc1_operator_coherence_service.py"
client="$here/catvm_dqc1_operator_coherence_client.py"
reference="$here/catvm_dqc1_operator_coherence_separate_reference.py"
qualifier="$here/qualify_catvm_dqc1_operator_coherence.sh"
sealed_raw="$here/CATVM_DQC1_OPERATOR_COHERENCE_RAW_RESULTS.json"
sealed_ref="$here/CATVM_DQC1_OPERATOR_COHERENCE_SEPARATE_REFERENCE.json"
sealed_result="$here/CATVM_DQC1_OPERATOR_COHERENCE_RESULTS.json"
raw="$build/CATVM_DQC1_OPERATOR_COHERENCE_RAW_RESULTS.json"
ref="$build/CATVM_DQC1_OPERATOR_COHERENCE_SEPARATE_REFERENCE.json"
result="$build/CATVM_DQC1_OPERATOR_COHERENCE_RESULTS.json"
service_stdout="$build/CATVM_DQC1_OPERATOR_COHERENCE_SERVICE.stdout"
service_stderr="$build/CATVM_DQC1_OPERATOR_COHERENCE_SERVICE.stderr"

mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
run_env=(
  env
  TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp"
  XDG_CACHE_HOME="$build/xdg-cache"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache"
)

public_config=$(${run_env[@]} python3 - <<'PY'
import json
primary=["H0","T0","H0","CNOT01"]
reuse=["H1","CNOT01","T0","H0","T1","CNOT10"]
cases={
 "primary":{"carrier_id":"m251-primary","descriptor":{"word":primary}},
 "reuse":{"carrier_id":"m251-primary","descriptor":{"word":reuse}},
 "fresh":{"carrier_id":"m251-fresh","descriptor":{"word":reuse}},
}
for name in ("disconnect","partial","postprojection","descriptor_control"):
 cases[name]={"carrier_id":f"m251-{name}","descriptor":{"word":primary}}
print(json.dumps({"cases":cases},sort_keys=True,separators=(",",":")))
PY
)
public_config_bytes=$(( ${#public_config} + 1 ))
reference_config='{"suite":"M251_DQC1_OPERATOR_COHERENCE_STRICT_SCOPE"}'
reference_config_bytes=$(( ${#reference_config} + 1 ))
socket_name="@catvm-m251-$$"

service_pid=""
cleanup_service() {
  if [[ -n "$service_pid" ]] && kill -0 "$service_pid" 2>/dev/null; then
    kill "$service_pid"
    wait "$service_pid" || true
  fi
}
trap cleanup_service EXIT

printf '%s\n' '{"service":"M251_DQC1_OPERATOR_COHERENCE_MODE"}' | \
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
  echo "M251 service did not become ready" >&2
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

def key(case): return (tuple(case["word"]),case["run_kind"])
raw_cases={key(case):case for case in raw["cases"]}
ref_cases={key(case):case for case in ref["cases"]}
if set(raw_cases)!=set(ref_cases): raise SystemExit("M251 case key mismatch")
for case_key in raw_cases:
    production=raw_cases[case_key]; oracle=ref_cases[case_key]
    for field in (
        "word","generation","normalized_trace","hidden_density_field_cells",
        "hidden_scratch_field_cells","retained_final_boundary_field_cells_during_inverse",
        "same_density_and_scratch_backings","canonical_after_restoration",
        "baseline_reload_used","work",
    ):
        if production[field]!=oracle[field]:
            raise SystemExit(f"M251 independent case mismatch {case_key} {field}")

if not all(raw["controls"].values()) or not all(ref["controls"].values()):
    raise SystemExit("M251 control failure")
expected_traces={
 ("H0","T0","H0","CNOT01"):[[1,4],[1,8],[0,1],[1,8]],
 ("H1","CNOT01","T0","H0","T1","CNOT10"):[[1,8],[-1,8],[1,8],[-1,8]],
}
if any(case["normalized_trace"]!=expected_traces[tuple(case["word"])] for case in raw["cases"]):
    raise SystemExit("M251 normalized trace mismatch")
if not all(
    case["same_density_and_scratch_backings"]
    and case["canonical_after_restoration"]
    and not case["baseline_reload_used"]
    for case in raw["cases"]
): raise SystemExit("M251 transaction invariant failure")
if any(
    "joint_not_product_after_forward" in case
    or "data_marginal_maximally_mixed_after_forward" in case
    for case in raw["cases"]
): raise SystemExit("M251 pre-inverse diagnostics escaped through accepted response")
if not all(
    case["joint_not_product_after_forward"]
    and case["data_marginal_maximally_mixed_after_forward"]
    for case in ref["cases"]
): raise SystemExit("M251 independent forward-invariant reconstruction failure")
if raw["reuse_parity"] is not True: raise SystemExit("M251 reuse parity failure")
if raw["operator_coherence_law"]!={
    "one_clean_control":True,
    "maximally_mixed_data_qubits":2,
    "final_boundary_is_normalized_trace":True,
    "data_marginal_unchanged_but_full_joint_state_correlated":True,
    "operator_coherence_is_causally_required_for_xy_boundary":True,
    "route_disposition":"RETIRE_AFTER_ONE_BOUNDED_GRAMMAR_IF_DIRECT_FOUR_BY_FOUR_TRACE_RECURRENCE_MATCHES",
}: raise SystemExit("M251 operator coherence law mismatch")

resource=raw["resource_law"]
required={
 "declared_public_word_lengths":[4,6,6],
 "accepted_persistent_carriers":2,
 "accepted_hidden_density_field_cells_per_carrier":64,
 "accepted_hidden_scratch_field_cells_per_carrier":4,
 "accepted_transactions":3,
 "service_static_public_gate_library_field_cells":96,
 "accepted_compiled_public_gate_plan_matrix_references":16,
 "accepted_forward_controlled_gates":16,
 "accepted_inverse_controlled_gates":16,
 "accepted_forward_field_multiply_terms":4096,
 "accepted_inverse_field_multiply_terms":4096,
 "accepted_forward_density_field_writes":1024,
 "accepted_inverse_density_field_writes":1024,
 "accepted_scratch_result_and_clear_writes_forward":[1024,1024],
 "accepted_scratch_result_and_clear_writes_inverse":[1024,1024],
 "accepted_boundary_field_multiplications":12,
 "accepted_boundary_field_accumulations":12,
 "retained_final_boundary_field_cells_during_inverse_per_transaction":1,
 "retained_dynamic_inverse_history_entries":0,
 "accepted_catvm_path_has_space_work_or_query_advantage":False,
 "resource_verification_level":"PACKAGE_SELF_REVIEW",
 "whole_transaction_live_payload_peak_complete":False,
 "exact_qzeta8_coordinate_payload_instrumented":False,
 "field_cell_counts_are_not_fixed_bit_payload_claims":True,
 "direct_transferable_baseline_resident_matrix_field_cells":16,
 "direct_transferable_baseline_transient_peak_complete":False,
}
for field,value in required.items():
    if resource.get(field)!=value: raise SystemExit(f"M251 resource mismatch {field}")
if resource["strongest_fixed_fixture_classical_baseline"]!="PUBLIC_WORD_VALIDATION_PLUS_FROZEN_EXACT_NORMALIZED_TRACE_IN_O1_WORK":
    raise SystemExit("M251 fixed classical baseline mismatch")
if resource["strongest_transferable_descriptor_level_classical_baseline"]!="DIRECT_EXACT_FOUR_BY_FOUR_QZETA8_PUBLIC_WORD_MATRIX_RECURRENCE_PLUS_TRACE_WITH16_RESIDENT_MATRIX_FIELD_CELLS_PLUS_DECLARED_TRANSIENT_MULTIPLICATION_SCRATCH_AND_NO_CATVM_INVERSE":
    raise SystemExit("M251 transferable classical baseline mismatch")
if resource["independent_verifier_only_baseline"]!="EXACT_DENSE_EIGHT_BY_EIGHT_QZETA8_DENSITY_CONJUGATION":
    raise SystemExit("M251 verifier baseline mismatch")
if any(raw["claim_limits"].values()): raise SystemExit("M251 claim ceiling failure")
if ref["independent_oracle"]!={
    "qzeta8_polynomial_quotient_arithmetic_reconstructed":True,
    "dense_eight_by_eight_density_oracle_executed":True,
    "direct_four_by_four_trace_oracle_executed":True,
    "block_identity_reconstructed":True,
    "computational_path_enumeration_used":False,
    "production_source_imported":False,
}: raise SystemExit("M251 independent oracle failure")
if ref["oracle_resource_law"]!={
    "dense_eight_by_eight_oracle_is_verifier_only":True,
    "strongest_fixed_fixture_classical_baseline":"PUBLIC_WORD_VALIDATION_PLUS_FROZEN_EXACT_NORMALIZED_TRACE_IN_O1_WORK",
    "strongest_transferable_descriptor_level_classical_baseline":"DIRECT_EXACT_FOUR_BY_FOUR_QZETA8_PUBLIC_WORD_MATRIX_RECURRENCE_PLUS_TRACE_WITH16_RESIDENT_MATRIX_FIELD_CELLS_PLUS_DECLARED_TRANSIENT_MULTIPLICATION_SCRATCH_AND_NO_CATVM_INVERSE",
    "direct_transferable_baseline_resident_matrix_field_cells":16,
    "direct_transferable_baseline_transient_peak_complete":False,
}: raise SystemExit("M251 independent resource law failure")

for path in source_paths[:3]: ast.parse(path.read_text(),filename=str(path))
service_source=source_paths[0].read_text(); client_source=source_paths[1].read_text(); ref_source=source_paths[2].read_text()
if "catvm_dqc1_operator_coherence_service" in client_source or "catvm_dqc1_operator_coherence_service" in ref_source:
    raise SystemExit("M251 backend imported by controller/reference")
if "itertools" in service_source or "itertools" in ref_source or "product("+"range" in service_source:
    raise SystemExit("M251 path enumeration helper forbidden")
if "stdin.close()" not in service_source or "libc.prctl(4, 0, 0, 0, 0)" not in service_source:
    raise SystemExit("M251 backend hardening missing")
if service_source.index("carrier.release()") > service_source.index('return {\n        "word"'):
    raise SystemExit("M251 response can precede restoration")

forbidden={"density_values","density_matrix","scratch_values","operator_blocks","gate_matrices","amplitudes","paths","assignments","intermediate"}
def contains(value):
    if isinstance(value,dict): return any(str(k).lower() in forbidden or contains(v) for k,v in value.items())
    if isinstance(value,list): return any(contains(v) for v in value)
    return False
if any(contains(case) for case in raw["cases"]): raise SystemExit("M251 hidden operator-coherence smuggle")

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
    "accepted_response_contains_only_declared_trace_boundary_and_custody_resource_receipts":True,
    "sealed_outputs_byte_reproducible":True,
    "disk_backed_build_directory":True,
    "no_path_enumeration":True,"no_hidden_operator_coherence_smuggle":True,
}
json.dump(raw,result_path.open("w"),sort_keys=True,indent=2); result_path.open("a").write("\n")
PY

cmp -s "$raw" "$sealed_raw"
cmp -s "$ref" "$sealed_ref"
cmp -s "$result" "$sealed_result"
echo "QUALIFIED_CATVM_DQC1_OPERATOR_COHERENCE_STRICT_SCOPE"
