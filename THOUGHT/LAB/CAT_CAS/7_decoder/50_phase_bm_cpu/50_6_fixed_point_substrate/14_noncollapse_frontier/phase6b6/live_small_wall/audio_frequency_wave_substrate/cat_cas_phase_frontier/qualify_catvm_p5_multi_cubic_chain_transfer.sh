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
    echo "RAM-backed M244 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M244 filesystem forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
service="$here/catvm_p5_multi_cubic_chain_transfer_service.py"
client="$here/catvm_p5_multi_cubic_chain_transfer_client.py"
reference="$here/catvm_p5_multi_cubic_chain_transfer_separate_reference.py"
qualifier="$here/qualify_catvm_p5_multi_cubic_chain_transfer.sh"
field_dependency="$here/zeta5_normalized_cubic_fourier_coherent_port.py"
sealed_raw="$here/CATVM_P5_MULTI_CUBIC_CHAIN_TRANSFER_RAW_RESULTS.json"
sealed_ref="$here/CATVM_P5_MULTI_CUBIC_CHAIN_TRANSFER_SEPARATE_REFERENCE.json"
sealed_result="$here/CATVM_P5_MULTI_CUBIC_CHAIN_TRANSFER_RESULTS.json"
raw="$build/CATVM_P5_MULTI_CUBIC_CHAIN_TRANSFER_RAW_RESULTS.json"
ref="$build/CATVM_P5_MULTI_CUBIC_CHAIN_TRANSFER_SEPARATE_REFERENCE.json"
result="$build/CATVM_P5_MULTI_CUBIC_CHAIN_TRANSFER_RESULTS.json"
service_stdout="$build/CATVM_P5_MULTI_CUBIC_CHAIN_TRANSFER_SERVICE.stdout"
service_stderr="$build/CATVM_P5_MULTI_CUBIC_CHAIN_TRANSFER_SERVICE.stderr"

mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
run_env=(
  env
  TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp"
  XDG_CACHE_HOME="$build/xdg-cache"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache"
)

private_config=$(${run_env[@]} python3 - <<'PY'
import json

DEPTHS=(2,3,4,8,16,32,64)

def item(depth,family,carrier_id):
    return {
        "depth":depth,
        "lambdas":[1+((index*(family+1)+family)%4) for index in range(depth)],
        "quadratics":[(index*index+(2*family+1)*index+family+1)%5 for index in range(depth)],
        "couplings":[1+((2*index+family)%4) for index in range(depth-1)],
        "output_index":(depth+2*family+1)%5,
        "carrier_id":carrier_id,
    }

oracles={}
for depth in DEPTHS:
    primary=item(depth,0,f"chain_k{depth}")
    reuse=item(depth,1,f"chain_k{depth}")
    fresh=item(depth,1,f"fresh_chain_k{depth}")
    oracles[f"k{depth}_primary"]=primary
    oracles[f"k{depth}_reuse"]=reuse
    oracles[f"k{depth}_reuse_fresh"]=fresh

disconnect=item(64,0,"disconnect_chain")
disconnect["delay_before_inverse_ms"]=120
oracles["disconnect_control"]=disconnect
exception=item(64,1,"exception_chain")
exception["inject_failure_after_projection"]=True
oracles["exception_control"]=exception
partial=item(64,0,"partial_chain")
partial["inject_failure_after_modules"]=7
oracles["partial_exception_control"]=partial

print(json.dumps({"oracles":oracles},sort_keys=True,separators=(",",":")))
PY
)
private_config_bytes=$(( ${#private_config} + 1 ))
socket_name="@catvm-m244-$$"

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
  echo "M244 service did not become ready" >&2
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

raw_cases={(case["depth"],case["run_kind"]):case for case in raw["cases"]}
ref_cases={(case["depth"],case["run_kind"]):case for case in ref["cases"]}
if set(raw_cases)!=set(ref_cases):
    raise SystemExit("M244 case key mismatch")
for key in raw_cases:
    if raw_cases[key]["final_amplitude"]!=ref_cases[key]["final_amplitude"]:
        raise SystemExit(f"M244 independent boundary mismatch {key}")

if not all(raw["controls"].values()) or not all(
    value for value in ref["controls"].values() if isinstance(value,bool)
):
    raise SystemExit("M244 control failure")
if ref["cross_rank_certificate"]["interface_rank_qzeta5"]!=25:
    raise SystemExit("M244 cross-rank certificate failure")

raw["separate_reference"]={
    "result":ref["result"],
    "dense_path_parity_through_depth4":ref["dense_path_parity_through_depth4"],
    "endpoint_specialized_baseline_parity":ref["endpoint_specialized_baseline_parity"],
    "endpoint_specialized_baseline_character_terms":ref["endpoint_specialized_baseline_character_terms"],
    "controls":ref["controls"],
    "cross_rank_certificate":ref["cross_rank_certificate"],
    "baseline":ref["baseline"],
    "imports_production_or_m237":ref["imports_production_or_m237"],
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
    "production_and_independent_five_vector_boundaries_match":True,
    "secret_dependent_intermediate_payload_metrics_not_delivered_or_sealed":True,
    "released_final_boundary_payload_is_controller_computed":True,
    "endpoint_specialized_forward_only_classical_baseline_matches":True,
    "dense_assignment_parity_limited_to_depths2_3_4":True,
    "cross_rank2_interface_rank25_reconstructed_at_split_primes41_61":True,
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
assert r["result"]=="PASS_CATVM_P5_MULTI_CUBIC_CHAIN_TRANSFER_STRICT_SCOPE"
assert r["classification"]=="INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
assert r["verification_level"]=="SEPARATE_REFERENCE_PARITY"
assert r["restoration_classification"]=="EXACT_ALGEBRAIC_RESTORATION"
assert all(r["controls"].values())
assert r["resource_law"]["phase_message_field_cells"]==[5]*7
assert r["resource_law"]["phase_message_scratch_field_cells"]==[5]*7
assert r["resource_law"]["catvm_hidden_configuration_residue_cells"]==[6,9,12,24,48,96,192]
assert r["resource_law"]["hidden_inter_module_coupling_residue_cells"]==[1,2,3,7,15,31,63]
assert r["resource_law"]["hidden_selected_output_index_residue_cells"]==[1]*7
assert r["resource_law"]["first_module_coupling_is_public_fixed_one"] is True
assert r["resource_law"]["first_module_coupling_is_nonmaterial_for_declared_e0_input"] is True
assert r["resource_law"]["carrier_resident_descriptor_residue_cells"]==[6,9,12,24,48,96,192]
assert r["resource_law"]["service_startup_descriptor_validation_residue_reads_per_oracle"]==[9,14,19,39,79,159,319]
assert r["resource_law"]["suite_service_hidden_configuration_descriptor_residue_cells"]==1737
assert r["resource_law"]["suite_service_peak_resident_phase_and_scratch_field_cells"]==170
assert r["resource_law"]["suite_service_peak_resident_carrier_descriptor_residue_cells"]==1350
assert r["resource_law"]["suite_service_peak_resident_configuration_plus_carrier_descriptor_residue_cells"]==3087
assert r["resource_law"]["suite_service_startup_descriptor_validation_residue_reads"]==2871
assert r["resource_law"]["suite_service_retains17_carriers_for_controls_cases_and_fresh_comparisons"] is True
assert r["resource_law"]["forward_character_terms"]==[50,75,100,200,400,800,1600]
assert r["resource_law"]["inverse_character_terms"]==[50,75,100,200,400,800,1600]
assert r["resource_law"]["accepted_transfer_descriptor_reads"]==[16,25,34,70,142,286,574]
assert r["resource_law"]["strongest_implemented_classical_baseline"]=="ENDPOINT_SPECIALIZED_EXACT_FIVE_VECTOR_INTERIOR_WITH5_TERM_FIRST_AND_FINAL_BOUNDARY_TRANSFERS"
assert r["resource_law"]["endpoint_specialized_classical_forward_character_terms"]==[10,35,60,160,360,760,1560]
assert r["resource_law"]["endpoint_specialized_classical_forward_only_no_inverse_or_restoration_work"] is True
assert r["resource_law"]["fixed_bounded_width_exact_state"] is False
assert r["resource_law"]["secret_dependent_intermediate_payload_metrics_released"] is False
assert r["resource_law"]["secret_dependent_intermediate_payload_metrics_retained_in_sealed_evidence"] is False
assert r["resource_law"]["whole_transaction_exact_payload_live_peak_measured"] is False
assert r["resource_law"]["public_forward_single_five_cell_vector_exact_payload_bit_upper_bounds"]==[226,328,431,840,1639,3256,6470]
assert r["resource_law"]["public_whole_transaction_message_scratch_and_retained_boundary_payload_bit_upper_bounds"]==[912,1342,1775,3462,6871,13652,27212]
assert r["separate_reference"]["cross_rank_certificate"]["interface_rank_qzeta5"]==25
assert r["separate_reference"]["controls"]["cross_rank2_rejects_five_cell_arbitrary_topology_transfer"] is True
assert all(r["separate_reference"]["dense_path_parity_through_depth4"].values())
assert all(r["separate_reference"]["endpoint_specialized_baseline_parity"].values())
assert r["separate_reference"]["endpoint_specialized_baseline_character_terms"]=={
    "2":10,"3":35,"4":60,"8":160,"16":360,"32":760,"64":1560
}
assert r["private_configuration_accounting"]["controller_receives_private_configuration"] is False
assert r["verification_statement"]["controller_imports_backend_or_field_code"] is False
assert not any(r["claim_limits"].values())
assert len(r["cases"])==21
for case in r["cases"]:
    assert case["canonical_after_restoration"]
    assert case["same_cell_backing"] and case["same_scratch_backing"] and case["same_descriptor_backings"]
    assert not case["baseline_reload_used"]
    assert "work" not in case
    assert "retained_final_amplitude_exact_payload_bits_during_inverse" not in case
    assert "released_final_boundary_exact_payload_bits" in case
for forbidden in (
    "peak_single_phase_message_or_scratch_denominator_exponent",
    "peak_single_phase_message_or_scratch_numerator_signed_bits",
    "peak_single_phase_message_or_scratch_exact_payload_bits",
    "canonical_common_factor_cancellations",
):
    assert forbidden not in json.dumps(r)
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
        raise SystemExit(f"M244 controller imports nonpublic dependency: {names-allowed}")
PY

if rg -n 'catvm_p5_multi_cubic_chain_transfer_service|zeta5_normalized_cubic_fourier_coherent_port|importlib|SourceFileLoader|runpy' "$client"; then
  echo "M244 controller imports or names backend code" >&2
  exit 2
fi
if rg -n 'itertools\.product|cartesian|assignment_table|transfer_matrix\s*=' "$service"; then
  echo "M244 accepted service contains forbidden assignment/table construction" >&2
  exit 2
fi

printf '%s\n' QUALIFIED_CATVM_P5_MULTI_CUBIC_CHAIN_TRANSFER_STRICT_SCOPE
