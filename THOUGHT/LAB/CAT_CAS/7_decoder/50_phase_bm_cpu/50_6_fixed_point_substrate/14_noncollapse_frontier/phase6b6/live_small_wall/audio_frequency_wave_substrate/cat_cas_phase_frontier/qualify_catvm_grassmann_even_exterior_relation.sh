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
    echo "RAM-backed M254 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M254 filesystem forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
service="$here/catvm_grassmann_even_exterior_relation_service.py"
client="$here/catvm_grassmann_even_exterior_relation_client.py"
reference="$here/catvm_grassmann_even_exterior_relation_separate_reference.py"
qualifier="$here/qualify_catvm_grassmann_even_exterior_relation.sh"
sealed_raw="$here/CATVM_GRASSMANN_EVEN_EXTERIOR_RELATION_RAW_RESULTS.json"
sealed_ref="$here/CATVM_GRASSMANN_EVEN_EXTERIOR_RELATION_SEPARATE_REFERENCE.json"
sealed_result="$here/CATVM_GRASSMANN_EVEN_EXTERIOR_RELATION_RESULTS.json"
raw="$build/CATVM_GRASSMANN_EVEN_EXTERIOR_RELATION_RAW_RESULTS.json"
ref="$build/CATVM_GRASSMANN_EVEN_EXTERIOR_RELATION_SEPARATE_REFERENCE.json"
result="$build/CATVM_GRASSMANN_EVEN_EXTERIOR_RELATION_RESULTS.json"
service_stdout="$build/CATVM_GRASSMANN_EVEN_EXTERIOR_RELATION_SERVICE.stdout"
service_stderr="$build/CATVM_GRASSMANN_EVEN_EXTERIOR_RELATION_SERVICE.stderr"

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
minus_one=[[-1,1],[0,1],[0,1],[0,1]]
zeta8=[[0,1],[1,2],[0,1],[1,2]]
def intersect(factor): return {"op":"INTERSECT","factor":factor}
hodge={"op":"HODGE"}
f1=[one,one,zero,zero,zero,zero,one,zeta8]
f2=[zeta8,zero,one,zero,one,zero,zero,one]
f3=[one,zero,zero,one,zero,zeta8,zero,minus_one]
primary={
    "ports":["THETA0","THETA1","THETA2","THETA3"],
    "modules":[intersect(f1),hodge,intersect(f2),intersect(f3),hodge],
}
reuse={
    "ports":["THETA0","THETA1","THETA2","THETA3"],
    "modules":[intersect(f3),hodge,intersect(f1),intersect(f2),hodge],
}
cases={
    "primary":{"carrier_id":"m254-primary","descriptor":primary},
    "reuse":{"carrier_id":"m254-primary","descriptor":reuse},
    "fresh":{"carrier_id":"m254-fresh","descriptor":reuse},
}
for name in ("disconnect","partial","postprojection","descriptor_control"):
    cases[name]={"carrier_id":f"m254-{name}","descriptor":primary}
print(json.dumps({"cases":cases},sort_keys=True,separators=(",",":")))
PY
)
reference_config=$(printf '%s\n' "$public_config" | "${run_env[@]}" python3 -c \
  'import json,sys; value=json.load(sys.stdin); value["suite"]="M254_GRASSMANN_EVEN_EXTERIOR_OPEN_RELATION_STRICT_SCOPE"; print(json.dumps(value,sort_keys=True,separators=(",",":")))')
public_config_bytes=$(( ${#public_config} + 1 ))
reference_config_bytes=$(( ${#reference_config} + 1 ))
socket_name="@catvm-m254-$$"

service_pid=""
cleanup_service() {
  if [[ -n "$service_pid" ]] && kill -0 "$service_pid" 2>/dev/null; then
    kill "$service_pid"
    wait "$service_pid" || true
  fi
}
trap cleanup_service EXIT

printf '%s\n' '{"service":"M254_GRASSMANN_EVEN_EXTERIOR_RELATION_MODE"}' | \
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
  echo "M254 service did not become ready" >&2
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
    raise SystemExit("M254 case key mismatch")
for case_key in raw_cases:
    production=raw_cases[case_key]; oracle=ref_cases[case_key]
    for field in (
        "module_kinds","generation","top_form_boundary","hidden_even_relation_field_cells",
        "hidden_hodge_scratch_field_cells","hidden_module_receipt_cells",
        "retained_final_boundary_field_cells_during_inverse",
        "same_relation_scratch_and_receipt_backings","canonical_after_restoration",
        "baseline_reload_used","work",
    ):
        if production[field]!=oracle[field]:
            raise SystemExit(f"M254 independent case mismatch {case_key} {field}")

if not all(raw["controls"].values()) or not all(ref["controls"].values()):
    raise SystemExit("M254 control failure")
for name in (
    "eight_variable_berezin_kernel_derives_expected_signed_complement_on_all_basis_masks",
    "generic_nilpotent_geometric_inverse_is_exact",
    "declared_public_linear_reachable_observable_hankel_ranks_are_exactly_eight",
):
    if ref["controls"].get(name) is not True:
        raise SystemExit(f"M254 independent control missing {name}")
if ref["rank_certificate"]!={"reachable_rank":8,"observable_rank":8,"hankel_rank":8}:
    raise SystemExit("M254 exact linear-rank certificate mismatch")
if not all(ref["reuse_parity"].values()) or raw["reuse_parity"] is not True:
    raise SystemExit("M254 reuse parity failure")
expected={
    "PRIMARY":[[0,1],[0,1],[1,1],[0,1]],
    "REUSE":[[0,1],[-1,2],[0,1],[-1,2]],
    "FRESH":[[0,1],[-1,2],[0,1],[-1,2]],
}
for case_key,value in expected.items():
    if raw_cases[case_key]["top_form_boundary"]!=value:
        raise SystemExit(f"M254 exact boundary mismatch {case_key}")
if not all(
    case["same_relation_scratch_and_receipt_backings"]
    and case["canonical_after_restoration"] and not case["baseline_reload_used"]
    for case in raw["cases"]
):
    raise SystemExit("M254 transaction invariant failure")

law=raw["open_relation_law"]
if law!={
    "typed_open_grassmann_ports":["THETA0","THETA1","THETA2","THETA3"],
    "resident_full_even_relation_field_cells":8,
    "intersection_is_native_signed_exterior_multiplication":True,
    "four_port_berezin_hodge_closure_is_exact_signed_complement_involution":True,
    "relation_coefficients_remain_unprojected_until_final_top_form_boundary":True,
    "independent_quartic_coordinate_is_strictly_broader_than_m253_gaussian_chart":True,
    "declared_public_linear_language_minimal_dimension":8,
    "route_disposition":"RETIRE_AFTER_ONE_BOUNDED_FOUR_PORT_SUITE_BECAUSE_IDENTICAL_CLASSICAL_EXTERIOR_RECURRENCE_BISIMULATES_IT_AND_EVEN_INTERFACE_DIMENSION_GROWS_EXPONENTIALLY",
}:
    raise SystemExit("M254 open relation law mismatch")

resource=raw["resource_law"]
required={
    "accepted_persistent_carriers":2,
    "accepted_hidden_even_relation_field_cells_per_carrier":8,
    "accepted_hidden_hodge_scratch_field_cells_per_carrier":1,
    "accepted_hidden_module_receipt_cells_per_carrier":8,
    "accepted_transactions":3,
    "accepted_compiled_public_module_plan_references":15,
    "accepted_compiled_public_intersection_factor_field_cells":72,
    "accepted_forward_intersections":9,
    "accepted_inverse_intersections":9,
    "accepted_forward_hodge_closures":6,
    "accepted_inverse_hodge_closures":6,
    "accepted_forward_field_multiplications":189,
    "accepted_inverse_field_multiplications":189,
    "accepted_forward_field_accumulations":126,
    "accepted_inverse_field_accumulations":126,
    "accepted_forward_field_negations":42,
    "accepted_inverse_field_negations":42,
    "accepted_forward_carrier_field_writes":120,
    "accepted_inverse_carrier_field_writes":120,
    "accepted_forward_hodge_scratch_writes_and_clears":48,
    "accepted_inverse_hodge_scratch_writes_and_clears":48,
    "accepted_inverse_factor_rematerializations":9,
    "accepted_inverse_factor_returned_field_cells_materialized":72,
    "accepted_peak_returned_inverse_factor_field_cells":8,
    "accepted_inverse_factor_field_multiplications":144,
    "accepted_inverse_factor_field_accumulations":63,
    "accepted_inverse_factor_field_negations":81,
    "accepted_inverse_factor_field_inversions":9,
    "accepted_retained_dynamic_inverse_history_entries":0,
    "retained_final_boundary_field_cells_during_inverse_per_transaction":1,
    "strongest_fixed_fixture_classical_baseline":"PUBLIC_DESCRIPTOR_VALIDATION_PLUS_FROZEN_EXACT_TOP_FORM_BOUNDARY_IN_O1_WORK",
    "strongest_implemented_transferable_descriptor_level_classical_baseline":"IDENTICAL_EXACT_INPLACE_EIGHT_QZETA8_CELL_FULL_EVEN_EXTERIOR_WEDGE_AND_HODGE_RECURRENCE_WITH_ONE_REUSABLE_FIELD_SCRATCH_CELL_NO_CATVM_RESTORATION",
    "transferable_baseline_resident_relation_field_cells":8,
    "transferable_baseline_reusable_hodge_and_intersection_scratch_field_cells":1,
    "declared_public_linear_language_exact_reachable_rank":8,
    "declared_public_linear_language_exact_observable_rank":8,
    "declared_public_linear_language_exact_hankel_rank":8,
    "full_even_interface_dimension_law":"TWO_TO_THE_POWER_PORT_COUNT_MINUS_ONE",
    "accepted_catvm_path_has_space_work_or_query_advantage":False,
}
for name,value in required.items():
    if resource.get(name)!=value:
        raise SystemExit(f"M254 resource mismatch {name}: {resource.get(name)!r}")
if resource["resource_verification_level"]!="PACKAGE_SELF_REVIEW":
    raise SystemExit("M254 resource verification level mismatch")
if resource["whole_transaction_live_payload_peak_complete"] is not False:
    raise SystemExit("M254 global live-payload overclaim")
if any(raw["claim_limits"].values()):
    raise SystemExit("M254 claim-limit overreach")
if raw["claim_ceiling"]!="EXACT_SOFTWARE_FOUR_TYPED_GRASSMANN_PORT_QZETA8_FULL_EVEN_EXTERIOR_RELATIONS_PUBLIC_PROGRAM_LENGTH_AT_MOST8_ON_AN_ABSTRACT_UNIX_SOCKET_CATVM_ONLY":
    raise SystemExit("M254 claim ceiling mismatch")

client_tree=ast.parse(source_paths[1].read_text())
reference_tree=ast.parse(source_paths[2].read_text())
client_imports={node.module for node in ast.walk(client_tree) if isinstance(node,ast.ImportFrom)}
client_imports.update(alias.name for node in ast.walk(client_tree) if isinstance(node,ast.Import) for alias in node.names)
reference_imports={node.module for node in ast.walk(reference_tree) if isinstance(node,ast.ImportFrom)}
reference_imports.update(alias.name for node in ast.walk(reference_tree) if isinstance(node,ast.Import) for alias in node.names)
for imported in client_imports|reference_imports:
    if "grassmann_even_exterior_relation_service" in (imported or ""):
        raise SystemExit("M254 controller/reference imports backend")
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
    raise SystemExit("M254 accepted source contains forbidden enumeration primitive")
dependencies={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
result=dict(raw)
result["independent_reference"]={
    "result":ref["result"],
    "oracle_law":ref["oracle_law"],
    "rank_certificate":ref["rank_certificate"],
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

if [[ "${M254_WRITE_SEALS:-0}" == 1 ]]; then
  cp -- "$raw" "$sealed_raw"
  cp -- "$ref" "$sealed_ref"
  cp -- "$result" "$sealed_result"
fi
cmp -s "$raw" "$sealed_raw"
cmp -s "$ref" "$sealed_ref"
cmp -s "$result" "$sealed_result"
printf '%s\n' "QUALIFIED_CATVM_GRASSMANN_EVEN_EXTERIOR_RELATION_STRICT_SCOPE"
