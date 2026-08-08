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
    echo "RAM-backed M247 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M247 filesystem forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
service="$here/catvm_p5_coherent_affine_cubic_signature_service.py"
client="$here/catvm_p5_coherent_affine_cubic_signature_client.py"
reference="$here/catvm_p5_coherent_affine_cubic_signature_separate_reference.py"
qualifier="$here/qualify_catvm_p5_coherent_affine_cubic_signature.sh"
field_dependency="$here/zeta5_normalized_cubic_fourier_coherent_port.py"
sealed_raw="$here/CATVM_P5_COHERENT_AFFINE_CUBIC_SIGNATURE_RAW_RESULTS.json"
sealed_ref="$here/CATVM_P5_COHERENT_AFFINE_CUBIC_SIGNATURE_SEPARATE_REFERENCE.json"
sealed_result="$here/CATVM_P5_COHERENT_AFFINE_CUBIC_SIGNATURE_RESULTS.json"
raw="$build/CATVM_P5_COHERENT_AFFINE_CUBIC_SIGNATURE_RAW_RESULTS.json"
ref="$build/CATVM_P5_COHERENT_AFFINE_CUBIC_SIGNATURE_SEPARATE_REFERENCE.json"
result="$build/CATVM_P5_COHERENT_AFFINE_CUBIC_SIGNATURE_RESULTS.json"
service_stdout="$build/CATVM_P5_COHERENT_AFFINE_CUBIC_SIGNATURE_SERVICE.stdout"
service_stderr="$build/CATVM_P5_COHERENT_AFFINE_CUBIC_SIGNATURE_SERVICE.stderr"

mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
run_env=(
  env
  TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp"
  XDG_CACHE_HOME="$build/xdg-cache"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache"
)

public_config=$(${run_env[@]} python3 - <<'PY'
import json

family0={
1:{"A":[[1]],"B":[[2]],"C":[[1]],"a":[1],"b":[1],"output":[0]},
2:{"A":[[2,2],[4,0]],"B":[[3,1],[3,2]],"C":[[4,2],[4,0]],"a":[3,4],"b":[1,1],"output":[0,4]},
3:{"A":[[4,2,3],[2,3,0],[0,0,4]],"B":[[3,1,1],[0,0,2],[1,4,3]],"C":[[1,4,3],[2,2,3],[1,2,2]],"a":[1,2,2],"b":[2,2,1],"output":[1,3,0]},
4:{"A":[[0,4,4,3],[4,2,2,2],[4,1,2,4],[2,1,2,1]],"B":[[2,2,4,3],[3,0,0,2],[3,4,1,0],[2,3,2,2]],"C":[[3,0,2,1],[2,0,4,1],[4,4,2,4],[4,2,0,1]],"a":[3,1,2,1],"b":[4,2,2,3],"output":[0,3,2,2]},
}
family1={
4:{"A":[[2,1,4,3],[0,3,3,3],[0,2,1,1],[1,4,2,0]],"B":[[4,2,4,1],[4,2,1,3],[1,2,1,0],[1,1,4,1]],"C":[[0,4,1,4],[0,3,1,3],[2,0,2,4],[4,1,2,2]],"a":[1,4,1,2],"b":[1,1,4,4],"output":[0,4,1,2]},
}
for width,descriptor in family0.items(): descriptor["width"]=width
family1[4]["width"]=4
def case(carrier,descriptor):
    return {"carrier_id":carrier,"descriptor":descriptor}
cases={
    **{f"primary_w{w}":case(f"signature_w{w}",family0[w]) for w in range(1,5)},
    "reuse_w4":case("signature_w4",family1[4]),
    "reuse_fresh_w4":case("signature_w4_fresh",family1[4]),
    "disconnect_w2":case("signature_disconnect_w2",family0[2]),
    "partial_w2":case("signature_partial_w2",family0[2]),
    "exception_w2":case("signature_exception_w2",family0[2]),
    "descriptor_control_w2":case("signature_descriptor_control_w2",family0[2]),
}
print(json.dumps({"cases":cases},sort_keys=True,separators=(",",":")))
PY
)
public_config_bytes=$(( ${#public_config} + 1 ))
socket_name="@catvm-m247-$$"

service_pid=""
cleanup_service() {
  if [[ -n "$service_pid" ]] && kill -0 "$service_pid" 2>/dev/null; then
    kill "$service_pid"
    wait "$service_pid" || true
  fi
}
trap cleanup_service EXIT

printf '%s\n' '{"service":"M247_PUBLIC_DESCRIPTOR_MODE"}' | \
  "${run_env[@]}" nice -n 10 ionice -c 3 \
  python3 "$service" "$socket_name" >"$service_stdout" 2>"$service_stderr" &
service_pid=$!
ready=false
for _ in $(seq 1 160); do
  if "${run_env[@]}" python3 - "$socket_name" <<'PY'
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
  echo "M247 service did not become ready" >&2
  exit 2
fi

printf '%s\n' "$public_config" | "${run_env[@]}" nice -n 10 ionice -c 3 \
  python3 "$client" "$socket_name" >"$raw"
wait "$service_pid"
service_pid=""
[[ ! -s "$service_stdout" ]]
[[ ! -s "$service_stderr" ]]
printf '%s\n' "$public_config" | "${run_env[@]}" nice -n 10 ionice -c 3 \
  python3 "$reference" >"$ref"

"${run_env[@]}" python3 - \
  "$raw" "$ref" "$result" "$service" "$client" "$reference" "$qualifier" \
  "$field_dependency" "$public_config_bytes" <<'PY'
import hashlib,json,sys
from pathlib import Path

raw_path,ref_path,result_path=map(Path,sys.argv[1:4])
source_paths=list(map(Path,sys.argv[4:9]))
public_bytes=int(sys.argv[9])
raw=json.loads(raw_path.read_text())
ref=json.loads(ref_path.read_text())
raw_cases={case["run_kind"]:case for case in raw["cases"]}
ref_cases={case["run_kind"]:case for case in ref["cases"]}
if set(raw_cases)!=set(ref_cases): raise SystemExit("M247 case key mismatch")
for key in raw_cases:
    if raw_cases[key]["final_amplitude"]!=ref_cases[key]["final_amplitude"]:
        raise SystemExit(f"M247 independent amplitude mismatch {key}")
    for field in ("width","generation"):
        if raw_cases[key][field]!=ref_cases[key][field]:
            raise SystemExit(f"M247 independent case mismatch {key} {field}")

expected={
 "PRIMARY_W1":([2,2,0,1],1),
 "PRIMARY_W2":([1,-3,-2,-1],2),
 "PRIMARY_W3":([0,2,-2,-2],2),
 "PRIMARY_W4":([1,3,1,5],3),
 "RESTORED_REUSE_W4":([-1,-7,-3,1],3),
 "FRESH_REUSE_REFERENCE_W4":([-1,-7,-3,1],3),
}
for key,(numerator,exponent) in expected.items():
    if raw_cases[key]["final_amplitude"]!={"numerator":numerator,"denominator_power5":exponent}:
        raise SystemExit(f"M247 frozen public fixture mismatch {key}")

if not all(raw["controls"].values()) or not all(ref["controls"].values()):
    raise SystemExit("M247 control failure")
if not all(ref["gate_semantic_oracle_parity"].values()):
    raise SystemExit("M247 gate semantic parity failure")
for width in range(1,5):
    baseline=ref["variable_elimination_baselines"][f"W{width}"]
    if baseline["final_amplitude"]!=raw_cases[f"PRIMARY_W{width}"]["final_amplitude"]:
        raise SystemExit(f"M247 variable elimination mismatch width {width}")
    sham=ref["quadratic_gauss_shams"][f"W{width}"]
    if not sham["gauss_parity"]:
        raise SystemExit(f"M247 quadratic Gauss mismatch width {width}")

raw["separate_reference"]={
    "result":ref["result"],
    "variable_elimination_baselines":ref["variable_elimination_baselines"],
    "streamed_scalar_baselines":ref["streamed_scalar_baselines"],
    "quadratic_gauss_shams":ref["quadratic_gauss_shams"],
    "gate_semantic_oracle_parity":ref["gate_semantic_oracle_parity"],
    "controls":ref["controls"],
    "imports_production_service_client_or_m237":ref[
        "imports_production_service_client_or_m237"
    ],
    "accepted_path_or_reference_stores5_to_width_amplitude_vector":ref[
        "accepted_path_or_reference_stores5_to_width_amplitude_vector"
    ],
    "implemented_classical_variable_elimination_uses_width_dependent_factor_tables":ref[
        "implemented_classical_variable_elimination_uses_width_dependent_factor_tables"
    ],
}
raw["public_configuration_accounting"]={
    "stdin_bytes_per_delivery":public_bytes,
    "deliveries":2,
    "recipients":["PUBLIC_CONTROLLER","SEPARATE_REFERENCE"],
    "service_receives_only_public_descriptors_in_per_transaction_requests":True,
    "private_or_answer_bearing_configuration_delivered":False,
}
names=("service","controller","separate_reference","qualifier","exact_qzeta5_field_dependency")
raw["source_dependencies"]={
    f"{name}_sha256":hashlib.sha256(path.read_bytes()).hexdigest()
    for name,path in zip(names,source_paths)
}
raw["verification_statement"]={
    "production_and_independent_final_amplitudes_match":True,
    "width1_2_gate_step_semantics_match_compiled_signature":True,
    "exact_min_fill_variable_elimination_matches_all_widths":True,
    "identical_streamed_scalar_classical_bisimulation_matches_all_widths":True,
    "zero_cubic_quadratic_gauss_formula_matches_all_widths":True,
    "noncommuting_shared_syndrome_consumers_change_selected_boundaries":True,
    "atomic_response_release_after_restoration":True,
    "controller_imports_backend_or_field_code":False,
    "no_amplitude_vector_or_path_assignment_list_on_accepted_path":True,
    "polynomial_size_signature_still_has_width_dependent_final_contraction_in_declared_fixtures":True,
}
with result_path.open("w") as handle:
    json.dump(raw,handle,sort_keys=True,indent=2)
    handle.write("\n")
PY

cmp -s "$raw" "$sealed_raw"
cmp -s "$ref" "$sealed_ref"
cmp -s "$result" "$sealed_result"

"${run_env[@]}" python3 - "$sealed_result" <<'PY'
import json,math,sys
r=json.load(open(sys.argv[1]))
assert r["result"]=="PASS_CATVM_P5_COHERENT_AFFINE_CUBIC_SIGNATURE_STRICT_SCOPE"
assert r["classification"]=="INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
assert r["verification_level"]=="SEPARATE_REFERENCE_PARITY"
assert r["restoration_classification"]=="EXACT_ALGEBRAIC_RESTORATION"
assert all(r["controls"].values()) and all(r["separate_reference"]["controls"].values())
assert r["resource_law"]["declared_widths"]==[1,2,3,4]
assert r["resource_law"]["degree_at_most_three_signature_residue_cells"]==[4,10,20,35]
assert r["resource_law"]["accepted_total_residue_backing_cells_excluding_projection_field_workspace"]==[12,36,74,127]
assert r["resource_law"]["suite_service_carrier_count"]==9
assert r["resource_law"]["suite_service_signature_coefficient_residue_cells"]==144
assert r["resource_law"]["suite_service_data_plus_syndrome_map_residue_cells"]==124
assert r["resource_law"]["suite_service_descriptor_residue_cells"]==252
assert r["resource_law"]["suite_service_projection_workspace_field_cells"]==9
assert r["resource_law"]["shared_public_monomial_plan_exponent_integer_cells"]==224
assert r["resource_law"]["shared_public_monomial_index_scalar_entries"]==69
assert r["resource_law"]["projection_assignment_terms"]==[5,25,125,625]
assert r["resource_law"]["polynomial_signature_size_does_not_remove_width_dependent_projection_work_or_exact_payload"]
assert not r["resource_law"]["projection_work_below_treewidth_or_stabilizer_sum_established"]
assert r["separate_reference"]["imports_production_service_client_or_m237"] is False
assert r["separate_reference"]["accepted_path_or_reference_stores5_to_width_amplitude_vector"] is False
assert not any(r["claim_limits"].values())
assert len(r["cases"])==6
for case in r["cases"]:
    assert case["canonical_after_restoration"]
    assert case["same_coefficient_backing"] and case["same_data_map_backing"]
    assert case["same_syndrome_map_backing"] and case["same_projection_workspace_backing"]
    assert case["same_descriptor_backings"] and not case["baseline_reload_used"]
    assert case["amplitude_vector_cells_materialized"]==0
    assert case["retained_path_assignment_lists_materialized"]==0
    assert case["bag_tables_materialized_on_accepted_path"]==0
    assert case["projection_assignment_terms"]==5**case["width"]
    for forbidden in ("coefficients","data_map","syndrome_map","assignment","bag_table"):
        assert forbidden not in case
for width in range(1,5):
    baseline=r["separate_reference"]["variable_elimination_baselines"][f"W{width}"]
    assert baseline["final_amplitude"]==r["cases"][width-1]["final_amplitude"]
    assert baseline["inverse_or_restoration_work"]==0
    assert not baseline["stores_amplitude_vector"]
    assert not baseline["stores_path_assignment_list"]
    streamed=r["separate_reference"]["streamed_scalar_baselines"][f"W{width}"]
    assert streamed["final_amplitude"]==r["cases"][width-1]["final_amplitude"]
    assert streamed["assignment_terms"]==5**width
    assert streamed["peak_field_accumulator_cells"]==1
    assert streamed["bag_table_field_cells"]==0
    assert streamed["amplitude_vector_field_cells"]==0
    assert streamed["inverse_or_restoration_work"]==0
    assert r["separate_reference"]["quadratic_gauss_shams"][f"W{width}"]["gauss_parity"]
PY

"${run_env[@]}" python3 - "$client" <<'PY'
import ast,sys
tree=ast.parse(open(sys.argv[1]).read())
allowed={"__future__","hashlib","json","math","socket","sys","time","typing"}
for node in ast.walk(tree):
    if isinstance(node,ast.Import): names={alias.name.split('.')[0] for alias in node.names}
    elif isinstance(node,ast.ImportFrom): names={str(node.module).split('.')[0]}
    else: continue
    if not names<=allowed: raise SystemExit(f"M247 controller imports nonpublic dependency: {names-allowed}")
PY

if rg -n 'catvm_p5_coherent_affine_cubic_signature_service|zeta5_normalized_cubic_fourier_coherent_port|importlib|SourceFileLoader|runpy' "$client"; then
  echo "M247 controller imports or names backend code" >&2
  exit 2
fi
if rg -n 'catvm_p5_coherent_affine_cubic_signature_service|catvm_p5_coherent_affine_cubic_signature_client|zeta5_normalized_cubic_fourier_coherent_port|importlib|SourceFileLoader|runpy' "$reference"; then
  echo "M247 reference imports or names production code" >&2
  exit 2
fi

printf '%s\n' QUALIFIED_CATVM_P5_COHERENT_AFFINE_CUBIC_SIGNATURE_STRICT_SCOPE
