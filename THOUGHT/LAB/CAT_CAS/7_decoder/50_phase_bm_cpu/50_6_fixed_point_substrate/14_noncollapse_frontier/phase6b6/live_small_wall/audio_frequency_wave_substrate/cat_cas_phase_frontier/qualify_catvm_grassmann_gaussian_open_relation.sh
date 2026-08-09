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
    echo "RAM-backed M253 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M253 filesystem forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
service="$here/catvm_grassmann_gaussian_open_relation_service.py"
client="$here/catvm_grassmann_gaussian_open_relation_client.py"
reference="$here/catvm_grassmann_gaussian_open_relation_separate_reference.py"
qualifier="$here/qualify_catvm_grassmann_gaussian_open_relation.sh"
sealed_raw="$here/CATVM_GRASSMANN_GAUSSIAN_OPEN_RELATION_RAW_RESULTS.json"
sealed_ref="$here/CATVM_GRASSMANN_GAUSSIAN_OPEN_RELATION_SEPARATE_REFERENCE.json"
sealed_result="$here/CATVM_GRASSMANN_GAUSSIAN_OPEN_RELATION_RESULTS.json"
raw="$build/CATVM_GRASSMANN_GAUSSIAN_OPEN_RELATION_RAW_RESULTS.json"
ref="$build/CATVM_GRASSMANN_GAUSSIAN_OPEN_RELATION_SEPARATE_REFERENCE.json"
result="$build/CATVM_GRASSMANN_GAUSSIAN_OPEN_RELATION_RESULTS.json"
service_stdout="$build/CATVM_GRASSMANN_GAUSSIAN_OPEN_RELATION_SERVICE.stdout"
service_stderr="$build/CATVM_GRASSMANN_GAUSSIAN_OPEN_RELATION_SERVICE.stderr"

mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
run_env=(
  env
  TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp"
  XDG_CACHE_HOME="$build/xdg-cache"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache"
)

public_config=$("${run_env[@]}" python3 - <<'PY'
import json
zero=[[0,1],[0,1],[0,1],[0,1]]
one=[[1,1],[0,1],[0,1],[0,1]]
zeta8=[[0,1],[1,2],[0,1],[1,2]]
def intersect(mu,coefficients):
    return {"op":"INTERSECT","mu":mu,"coefficients":coefficients}
fourier={"op":"FOURIER"}
c1=intersect(zeta8,[zero,one,zero,zero,zero,zero])
c2=intersect(one,[zero,zero,one,one,zero,zero])
c3=intersect(one,[zero,zero,zero,zero,zeta8,zero])
primary={
    "ports":["THETA0","THETA1","THETA2","THETA3"],
    "modules":[c1,c2,fourier,c3,fourier],
}
reuse={
    "ports":["THETA0","THETA1","THETA2","THETA3"],
    "modules":[c3,c1,fourier,c2,fourier],
}
cases={
    "primary":{"carrier_id":"m253-primary","descriptor":primary},
    "reuse":{"carrier_id":"m253-primary","descriptor":reuse},
    "fresh":{"carrier_id":"m253-fresh","descriptor":reuse},
}
for name in ("disconnect","partial","postprojection","descriptor_control"):
    cases[name]={"carrier_id":f"m253-{name}","descriptor":primary}
print(json.dumps({"cases":cases},sort_keys=True,separators=(",",":")))
PY
)
reference_config=$(printf '%s\n' "$public_config" | "${run_env[@]}" python3 -c \
  'import json,sys; value=json.load(sys.stdin); value["suite"]="M253_GRASSMANN_GAUSSIAN_OPEN_RELATION_STRICT_SCOPE"; print(json.dumps(value,sort_keys=True,separators=(",",":")))')
public_config_bytes=$(( ${#public_config} + 1 ))
reference_config_bytes=$(( ${#reference_config} + 1 ))
socket_name="@catvm-m253-$$"

service_pid=""
cleanup_service() {
  if [[ -n "$service_pid" ]] && kill -0 "$service_pid" 2>/dev/null; then
    kill "$service_pid"
    wait "$service_pid" || true
  fi
}
trap cleanup_service EXIT

printf '%s\n' '{"service":"M253_GRASSMANN_GAUSSIAN_OPEN_RELATION_MODE"}' | \
  "${run_env[@]}" nice -n 10 ionice -c 3 \
  python3 "$service" "$socket_name" >"$service_stdout" 2>"$service_stderr" &
service_pid=$!
ready=false
for _ in $(seq 1 160); do
  if "${run_env[@]}" python3 - "$socket_name" 2>/dev/null <<'PY'
import socket,sys
connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
try: connection.connect("\0"+sys.argv[1][1:])
except OSError: raise SystemExit(1)
connection.close()
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
  echo "M253 service did not become ready" >&2
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

def key(case): return case["run_kind"]
raw_cases={key(case):case for case in raw["cases"]}
ref_cases={key(case):case for case in ref["cases"]}
if set(raw_cases)!=set(ref_cases)!={"PRIMARY","REUSE","FRESH"}:
    raise SystemExit("M253 case key mismatch")
for case_key in raw_cases:
    production=raw_cases[case_key]; oracle=ref_cases[case_key]
    for field in (
        "module_kinds","generation","top_form_boundary","hidden_relation_field_cells",
        "hidden_fourier_scratch_field_cells","hidden_module_receipt_cells",
        "retained_final_boundary_field_cells_during_inverse",
        "same_relation_scratch_and_receipt_backings","canonical_after_restoration",
        "baseline_reload_used","work",
    ):
        if production[field]!=oracle[field]:
            raise SystemExit(f"M253 independent case mismatch {case_key} {field}")

if not all(raw["controls"].values()) or not all(ref["controls"].values()):
    raise SystemExit("M253 control failure")
for name in (
    "dense_gauss_jordan_fourier_matches_inplace_seven_cell_two_scratch_formula",
    "four_port_fourier_transform_is_exact_involution",
    "inplace_comparator_two_cell_scratch_is_cleared_after_fourier",
):
    if ref["controls"].get(name) is not True:
        raise SystemExit(f"M253 independent Fourier control missing {name}")
if not all(ref["reuse_parity"].values()) or raw["reuse_parity"] is not True:
    raise SystemExit("M253 reuse parity failure")
expected={
    "PRIMARY":[[0,1],[1,1],[0,1],[1,1]],
    "REUSE":[[0,1],[1,2],[-1,1],[1,2]],
    "FRESH":[[0,1],[1,2],[-1,1],[1,2]],
}
for case_key,value in expected.items():
    if raw_cases[case_key]["top_form_boundary"]!=value:
        raise SystemExit(f"M253 exact boundary mismatch {case_key}")
if not all(
    case["same_relation_scratch_and_receipt_backings"]
    and case["canonical_after_restoration"] and not case["baseline_reload_used"]
    for case in raw["cases"]
):
    raise SystemExit("M253 transaction invariant failure")

law=raw["open_relation_law"]
if law!={
    "typed_open_grassmann_ports":["THETA0","THETA1","THETA2","THETA3"],
    "resident_gaussian_relation_field_cells":7,
    "intersection_is_native_coefficient_addition_and_scalar_multiplication":True,
    "four_port_berezin_closure_is_pfaffian_scaled_inverse_on_nonsingular_chart":True,
    "fourier_closure_is_exact_involution":True,
    "relation_coefficients_remain_unprojected_until_final_top_form_boundary":True,
    "formal_gaussian_relation_class_is_closed_for_declared_modules":True,
    "route_disposition":"RETIRE_AFTER_ONE_BOUNDED_FOUR_PORT_SUITE_BECAUSE_IDENTICAL_CLASSICAL_PFAFFIAN_RECURRENCE_BISIMULATES_IT",
}:
    raise SystemExit("M253 open relation law mismatch")

resource=raw["resource_law"]
required={
    "accepted_persistent_carriers":2,
    "accepted_hidden_relation_field_cells_per_carrier":7,
    "accepted_hidden_fourier_scratch_field_cells_per_carrier":7,
    "accepted_hidden_module_receipt_cells_per_carrier":8,
    "accepted_transactions":3,
    "accepted_compiled_public_module_plan_references":15,
    "accepted_compiled_public_intersection_field_cells":63,
    "accepted_forward_intersections":9,
    "accepted_inverse_intersections":9,
    "accepted_forward_fourier_closures":6,
    "accepted_inverse_fourier_closures":6,
    "accepted_forward_field_multiplications":69,
    "accepted_inverse_field_multiplications":69,
    "accepted_forward_field_accumulations":66,
    "accepted_inverse_field_accumulations":66,
    "accepted_forward_field_inversions":6,
    "accepted_inverse_field_inversions":15,
    "accepted_forward_carrier_field_writes":105,
    "accepted_inverse_carrier_field_writes":105,
    "accepted_forward_scratch_field_writes_and_clears":84,
    "accepted_inverse_scratch_field_writes_and_clears":84,
    "accepted_boundary_field_multiplications":12,
    "accepted_boundary_field_accumulations":6,
    "accepted_retained_dynamic_inverse_history_entries":0,
    "retained_final_boundary_field_cells_during_inverse_per_transaction":1,
    "strongest_fixed_fixture_classical_baseline":"PUBLIC_DESCRIPTOR_VALIDATION_PLUS_FROZEN_EXACT_TOP_FORM_BOUNDARY_IN_O1_WORK",
    "strongest_implemented_transferable_descriptor_level_classical_baseline":"IDENTICAL_EXACT_INPLACE_SEVEN_QZETA8_CELL_ANTISYMMETRIC_PFAFFIAN_INTERSECTION_AND_COMPLEMENTARY_PAIR_BEREZIN_RECURRENCE_WITH_TWO_REUSABLE_FIELD_SCRATCH_CELLS_NO_CATVM_RESTORATION",
    "transferable_baseline_resident_relation_field_cells":7,
    "transferable_baseline_reusable_fourier_scratch_field_cells":2,
    "transferable_baseline_field_multiplications_per_fourier":10,
    "transferable_baseline_field_accumulations_per_fourier":2,
    "transferable_baseline_field_inversions_per_fourier":1,
    "transferable_baseline_carrier_writes_per_fourier":7,
    "accepted_catvm_path_has_space_work_or_query_advantage":False,
}
for name,value in required.items():
    if resource.get(name)!=value:
        raise SystemExit(f"M253 resource mismatch {name}: {resource.get(name)!r}")
if resource["resource_verification_level"]!="PACKAGE_SELF_REVIEW":
    raise SystemExit("M253 resource verification level mismatch")
if resource["whole_transaction_live_payload_peak_complete"] is not False:
    raise SystemExit("M253 global live-payload overclaim")
if any(raw["claim_limits"].values()):
    raise SystemExit("M253 claim-limit overreach")
if raw["claim_ceiling"]!="EXACT_SOFTWARE_FOUR_TYPED_GRASSMANN_PORT_QZETA8_GAUSSIAN_RELATIONS_PUBLIC_PROGRAM_LENGTH_AT_MOST8_ON_AN_ABSTRACT_UNIX_SOCKET_CATVM_ONLY":
    raise SystemExit("M253 claim ceiling mismatch")

client_tree=ast.parse(source_paths[1].read_text())
reference_tree=ast.parse(source_paths[2].read_text())
client_imports={node.module for node in ast.walk(client_tree) if isinstance(node,ast.ImportFrom)}
client_imports.update(alias.name for node in ast.walk(client_tree) if isinstance(node,ast.Import) for alias in node.names)
reference_imports={node.module for node in ast.walk(reference_tree) if isinstance(node,ast.ImportFrom)}
reference_imports.update(alias.name for node in ast.walk(reference_tree) if isinstance(node,ast.Import) for alias in node.names)
for imported in client_imports|reference_imports:
    if "grassmann_gaussian_open_relation_service" in (imported or ""):
        raise SystemExit("M253 controller/reference imports backend")
for path in source_paths:
    if path.suffix==".py": ast.parse(path.read_text())

service_tree=ast.parse(source_paths[0].read_text())
service_imports={node.module for node in ast.walk(service_tree) if isinstance(node,ast.ImportFrom)}
service_imports.update(alias.name for node in ast.walk(service_tree) if isinstance(node,ast.Import) for alias in node.names)
if "itertools" in service_imports or any(
    isinstance(node,ast.Call) and (
        isinstance(node.func,ast.Name) and node.func.id in {"product","permutations"}
        or isinstance(node.func,ast.Attribute) and node.func.attr in {"product","permutations"}
    )
    for node in ast.walk(service_tree)
):
    raise SystemExit("M253 accepted source contains forbidden enumeration primitive")
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
    "controller_received_only_public_typed_ports_and_relation_modules":True,
    "backend_private_mode_delivered_only_to_service_stdin":True,
    "standalone_oracle_received_the_same_public_descriptors_plus_suite_selector":True,
}
result_path.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
PY

if [[ "${M253_WRITE_SEALS:-0}" == 1 ]]; then
  cp -- "$raw" "$sealed_raw"
  cp -- "$ref" "$sealed_ref"
  cp -- "$result" "$sealed_result"
fi
cmp -s "$raw" "$sealed_raw"
cmp -s "$ref" "$sealed_ref"
cmp -s "$result" "$sealed_result"
printf '%s\n' "QUALIFIED_CATVM_GRASSMANN_GAUSSIAN_OPEN_RELATION_STRICT_SCOPE"
