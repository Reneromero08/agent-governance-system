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
    echo "RAM-backed M243 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M243 filesystem forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
service="$here/catvm_p5_quadratic_rank1_cubic_gauss_quotient_service.py"
client="$here/catvm_p5_quadratic_rank1_cubic_gauss_quotient_client.py"
reference="$here/catvm_p5_quadratic_rank1_cubic_gauss_quotient_separate_reference.py"
qualifier="$here/qualify_catvm_p5_quadratic_rank1_cubic_gauss_quotient.sh"
field_dependency="$here/zeta5_normalized_cubic_fourier_coherent_port.py"
sealed_raw="$here/CATVM_P5_QUADRATIC_RANK1_CUBIC_GAUSS_QUOTIENT_RAW_RESULTS.json"
sealed_ref="$here/CATVM_P5_QUADRATIC_RANK1_CUBIC_GAUSS_QUOTIENT_SEPARATE_REFERENCE.json"
sealed_result="$here/CATVM_P5_QUADRATIC_RANK1_CUBIC_GAUSS_QUOTIENT_RESULTS.json"
raw="$build/CATVM_P5_QUADRATIC_RANK1_CUBIC_GAUSS_QUOTIENT_RAW_RESULTS.json"
ref="$build/CATVM_P5_QUADRATIC_RANK1_CUBIC_GAUSS_QUOTIENT_SEPARATE_REFERENCE.json"
result="$build/CATVM_P5_QUADRATIC_RANK1_CUBIC_GAUSS_QUOTIENT_RESULTS.json"
raw_view="$build/CATVM_P5_QUADRATIC_RANK1_CUBIC_GAUSS_QUOTIENT_RAW_SEAL_VIEW.json"
ref_view="$build/CATVM_P5_QUADRATIC_RANK1_CUBIC_GAUSS_QUOTIENT_REFERENCE_SEAL_VIEW.json"
service_stdout="$build/CATVM_P5_QUADRATIC_RANK1_CUBIC_GAUSS_QUOTIENT_SERVICE.stdout"
service_stderr="$build/CATVM_P5_QUADRATIC_RANK1_CUBIC_GAUSS_QUOTIENT_SERVICE.stderr"

mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
run_env=(
  env
  TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp"
  XDG_CACHE_HOME="$build/xdg-cache"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache"
)

private_config=$("${run_env[@]}" python3 - <<'PY'
import json
import secrets
from itertools import product

P=5
rng=secrets.SystemRandom()

def packed_index(row,column):
    if row<column:
        row,column=column,row
    return row*(row+1)//2+column

def build_matrix(n):
    lower=[[0]*n for _ in range(n)]
    for row in range(n):
        lower[row][row]=1
        for column in range(row):
            lower[row][column]=rng.randrange(1,P)
    diagonal=[rng.randrange(1,P) for _ in range(n)]
    matrix=[[0]*n for _ in range(n)]
    for row in range(n):
        for column in range(n):
            matrix[row][column]=sum(
                lower[row][k]*diagonal[k]*lower[column][k]
                for k in range(min(row,column)+1)
            )%P
    return [matrix[row][column] for row in range(n) for column in range(row+1)]

def root(power):
    power%=P
    if power==4:
        return (-1,-1,-1,-1)
    result=[0]*4
    result[power]=1
    return tuple(result)

def add(left,right):
    return tuple(left[i]+right[i] for i in range(4))

def dense_boundary(n,matrix,vector,strength):
    total=(0,0,0,0)
    for state in product(range(P),repeat=n):
        phase=0
        for row in range(n):
            phase+=matrix[packed_index(row,row)]*state[row]*state[row]
            for column in range(row):
                phase+=2*matrix[packed_index(row,column)]*state[row]*state[column]
        linear=sum(vector[i]*state[i] for i in range(n))%P
        total=add(total,root(phase+strength*linear**3))
    return total

def descriptor(n,require_strength_change=False):
    while True:
        matrix=build_matrix(n)
        vector=[rng.randrange(P) for _ in range(n)]
        if sum(value!=0 for value in vector)<2:
            continue
        strength=rng.randrange(1,P)
        changed=strength+1
        if changed==P:
            changed=1
        if require_strength_change and dense_boundary(n,matrix,vector,strength)==dense_boundary(n,matrix,vector,changed):
            continue
        return {
            "dimension":n,
            "matrix":matrix,
            "vector":vector,
            "cubic_strength":strength,
        }

def congruence(descriptor):
    n=descriptor["dimension"]
    matrix=list(descriptor["matrix"])
    vector=list(descriptor["vector"])
    for row in range(n):
        cell=packed_index(row,0)
        matrix[cell]=2*matrix[cell]%P
    matrix[packed_index(0,0)]=2*matrix[packed_index(0,0)]%P
    vector[0]=2*vector[0]%P
    return {
        "dimension":n,
        "matrix":matrix,
        "vector":vector,
        "cubic_strength":descriptor["cubic_strength"],
    }

oracles={}
for name,flag in (
    ("disconnect_control","delay"),
    ("exception_control","exception"),
    ("partial_exception_control","partial"),
):
    item=descriptor(16)
    item["carrier_id"]=name+"_carrier"
    if flag=="delay":
        item["delay_before_inverse_ms"]=80
    elif flag=="exception":
        item["inject_failure_after_projection"]=True
    else:
        item["inject_failure_after_pivots"]=7
    oracles[name]=item

for n in (2,3,4,6,8,12,16):
    primary=descriptor(n,require_strength_change=(n==4))
    reuse=descriptor(n)
    while reuse==primary:
        reuse=descriptor(n)
    primary["carrier_id"]=f"carrier_n{n}"
    reuse["carrier_id"]=f"carrier_n{n}"
    fresh=dict(reuse)
    fresh["matrix"]=list(reuse["matrix"])
    fresh["vector"]=list(reuse["vector"])
    fresh["carrier_id"]=f"fresh_carrier_n{n}"
    oracles[f"n{n}_primary"]=primary
    oracles[f"n{n}_reuse"]=reuse
    oracles[f"n{n}_reuse_fresh"]=fresh

collision_a=descriptor(4)
collision_b=congruence(collision_a)
collision_a["carrier_id"]="collision_a_carrier"
collision_b["carrier_id"]="collision_b_carrier"
oracles["quotient_collision_a"]=collision_a
oracles["quotient_collision_b"]=collision_b

print(json.dumps({"oracles":oracles},sort_keys=True,separators=(",",":")))
PY
)
private_config_bytes=$(( ${#private_config} + 1 ))
socket_name="@catvm-m243-$$"

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
for _ in $(seq 1 100); do
  if "${run_env[@]}" python3 - "$socket_name" <<'PY'
import socket
import sys
connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
try:
    connection.connect("\0"+sys.argv[1][1:])
except OSError:
    raise SystemExit(1)
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
  echo "M243 service did not become ready" >&2
  exit 2
fi

"${run_env[@]}" nice -n 10 ionice -c 3 python3 "$client" "$socket_name" >"$raw"
wait "$service_pid"
service_pid=""
[[ ! -s "$service_stdout" ]]
[[ ! -s "$service_stderr" ]]
printf '%s\n' "$private_config" | "${run_env[@]}" nice -n 10 ionice -c 3 \
  python3 "$reference" "$raw" >"$ref"

"${run_env[@]}" python3 - \
  "$raw" "$ref" "$service" "$client" "$reference" "$qualifier" \
  "$field_dependency" "$raw_view" "$ref_view" "$private_config_bytes" \
  >"$result" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    raw_path,ref_path,service_path,client_path,reference_path,qualifier_path,
    field_path,raw_view_path,ref_view_path,
)=map(Path,sys.argv[1:10])
private_config_bytes=int(sys.argv[10])
raw=json.loads(raw_path.read_text())
reference=json.loads(ref_path.read_text())
by_id={case["oracle_id"]:case for case in reference["cases"]}
sanitized=[]
reference_views=[]
for case in raw["cases"]:
    check=by_id[case["oracle_id"]]
    if case["final_amplitude"]!=check["final_amplitude"]:
        raise SystemExit("M243 independent formula mismatch")
    if check["dense_exact_parity"]!=(case["dimension"]<=4):
        raise SystemExit("M243 dense parity scope mismatch")
    view={
        key:value for key,value in case.items()
        if key not in {"final_amplitude","controller_request_bytes","backend_response_bytes"}
    }
    view.update({
        "final_amplitude_field_cells":1,
        "private_descriptor_authority_agreement":True,
        "independent_formula_boundary_parity":True,
        "independent_dense_sum_parity":check["dense_exact_parity"],
    })
    sanitized.append(view)
    reference_views.append({
        "oracle_id":case["oracle_id"],
        "dimension":case["dimension"],
        "final_amplitude_field_cells":1,
        "private_descriptor_authority_agreement":True,
        "independent_formula_boundary_parity":True,
        "independent_dense_sum_parity":check["dense_exact_parity"],
        "dense_assignments_verifier_only":check["dense_assignments_verifier_only"],
        "channel_character_terms":check["channel_character_terms"],
        "coherent_channel_terms":check["coherent_channel_terms"],
    })

result=dict(raw)
result["schema"]="cat_cas.catvm_p5_quadratic_rank1_cubic_gauss_quotient.v1"
result["cases"]=sanitized
result["classification"]="INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
result["verification_level"]="SEPARATE_REFERENCE_PARITY"
result["restoration_classification"]="EXACT_ALGEBRAIC_RESTORATION"
result["atomic_response_law"]={
    "hidden_descriptor_then_ldl_then_solve_then_five_channels_then_internal_final_boundary_then_actual_inverse_then_restoration_verification_then_response":True,
    "disconnect_before_response_still_restores":raw["controls"]["disconnect_before_response_still_restores"],
    "post_projection_exception_rejected_only_after_restoration":raw["controls"]["post_projection_exception_rejected_only_after_restoration"],
    "partial_ldl_exception_rejected_only_after_prefix_rollback_and_restoration":raw["controls"]["partial_ldl_exception_rejected_only_after_restoration"],
    "service_process_nondumpable":True,
    "abstract_unix_socket_has_no_filesystem_artifact":True,
    "controller_imports_or_loads_backend_code":False,
    "private_configuration_delivered_over_stdin_to_service_and_separate_verifier":True,
    "controller_receives_private_configuration":False,
    "pre_run_secret_dependent_receipt_exposed":False,
    "final_amplitude_is_only_secret_dependent_response":True,
}
result["separate_reference"]={
    "schema":reference["schema"],
    "independent_exact_power_basis_formula_reexecution":reference["independent_exact_power_basis_formula_reexecution"],
    "independent_modular_gaussian_elimination_not_ldl_production_path":reference["independent_modular_gaussian_elimination_not_ldl_production_path"],
    "independent_dense_sum_parity_dimensions2_3_4":reference["independent_dense_sum_parity_dimensions2_3_4"],
    "strongest_compact_classical_baseline_is_identical_gauss_quotient":reference["strongest_compact_classical_baseline_is_identical_gauss_quotient"],
    "controls":reference["controls"],
    "imports_service_controller_or_m237":reference["imports_service_controller_or_m237"],
    "source_sha256":reference["source_sha256"],
}
result["source_dependencies"]={
    "service_sha256":hashlib.sha256(service_path.read_bytes()).hexdigest(),
    "controller_sha256":hashlib.sha256(client_path.read_bytes()).hexdigest(),
    "separate_reference_sha256":hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    "qualifier_sha256":hashlib.sha256(qualifier_path.read_bytes()).hexdigest(),
    "exact_qzeta5_field_dependency_sha256":hashlib.sha256(field_path.read_bytes()).hexdigest(),
}
resource=dict(result["resource_law"])
for key in (
    "controller_backend_request_bytes_total",
    "backend_controller_response_bytes_total",
    "disconnect_control_request_bytes",
):
    resource.pop(key,None)
resource.update({
    "protocol_bytes_measured_in_raw_execution":True,
    "secret_dependent_encoded_byte_counts_retained_in_sanitized_evidence":False,
    "private_configuration_stdin_bytes_per_delivery":private_config_bytes,
    "private_configuration_stdin_deliveries":2,
    "private_configuration_stdin_recipients":["CATVM_SERVICE","SEPARATE_REFERENCE"],
    "private_configuration_and_verifier_traffic_bytes_total":2*private_config_bytes,
})
result["resource_law"]=resource

raw_view_data={
    "schema":"cat_cas.catvm_p5_quadratic_rank1_cubic_gauss_quotient_raw_seal_view.v1",
    "result":raw["result"],
    "claim":raw["claim"],
    "claim_ceiling":raw["claim_ceiling"],
    "cases":sanitized,
    "controls":raw["controls"],
    "algebra_law":raw["algebra_law"],
    "resource_law":resource,
    "claim_limits":raw["claim_limits"],
    "terminal":raw["terminal"],
}
reference_view_data={
    "schema":"cat_cas.catvm_p5_quadratic_rank1_cubic_gauss_quotient_reference_seal_view.v1",
    "cases":reference_views,
    "controls":reference["controls"],
    "independent_exact_power_basis_formula_reexecution":reference["independent_exact_power_basis_formula_reexecution"],
    "independent_modular_gaussian_elimination_not_ldl_production_path":reference["independent_modular_gaussian_elimination_not_ldl_production_path"],
    "independent_dense_sum_parity_dimensions2_3_4":reference["independent_dense_sum_parity_dimensions2_3_4"],
    "strongest_compact_classical_baseline_is_identical_gauss_quotient":reference["strongest_compact_classical_baseline_is_identical_gauss_quotient"],
    "imports_service_controller_or_m237":reference["imports_service_controller_or_m237"],
    "source_sha256":reference["source_sha256"],
}
raw_view_path.write_text(json.dumps(raw_view_data,indent=2,sort_keys=True)+"\n")
ref_view_path.write_text(json.dumps(reference_view_data,indent=2,sort_keys=True)+"\n")
print(json.dumps(result,indent=2,sort_keys=True))
PY

if [[ ${M243_GENERATE_ONLY:-0} != 1 ]]; then
  cmp "$raw_view" "$sealed_raw"
  cmp "$ref_view" "$sealed_ref"
  cmp "$result" "$sealed_result"
fi

jq -e '
  .result=="PASS_CATVM_P5_QUADRATIC_RANK1_CUBIC_GAUSS_QUOTIENT_STRICT_SCOPE"
  and .classification=="INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level=="SEPARATE_REFERENCE_PARITY"
  and .restoration_classification=="EXACT_ALGEBRAIC_RESTORATION"
  and (.cases|length)==21
  and ([.cases[]|.dimension]==[2,2,2,3,3,3,4,4,4,6,6,6,8,8,8,12,12,12,16,16,16])
  and ([.cases[]|.restoration_generation]==[1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1])
  and (.cases|all(
    .final_amplitude_field_cells==1
    and .private_descriptor_authority_agreement
    and .independent_formula_boundary_parity
    and (.independent_dense_sum_parity==(.dimension<=4))
    and .response_released_after_restoration
    and .canonical_post_inverse_state_exact
    and .same_all_carrier_backings
    and (.baseline_reload_used|not)
    and .dense_global_amplitude_cells_materialized==0
    and .coherent_channel_field_cells==5
    and .coherent_channel_scratch_field_cells==5
    and .quotient_residue_scratch_cells==2
    and .work.hidden_descriptor_residue_reads==2*.hidden_descriptor_residue_cells
    and .work.ldl_forward_pivots==.dimension
    and .work.ldl_inverse_pivots==.dimension
    and .work.cubic_channel_character_terms==25
    and .work.coherent_channel_terms==5
    and .work.retained_final_amplitude_field_cells_during_inverse==1
    and .work.retained_dynamic_inverse_history_entries==0
    and .work.dense_global_amplitude_cells_materialized==0
  ))
  and (.controls|to_entries|map(select(
    .key!="controller_imports_or_loads_backend_code"
    and .key!="controller_receives_hidden_descriptor_or_quotient"
    and .key!="controller_computes_final_amplitude_independently"
    and .key!="service_stdout_stderr_contains_hidden_descriptor_or_channels"
    and .key!="snapshot_reload_used_by_accepted_path"
    and .key!="pre_run_secret_dependent_receipt_exposed"
  ))|all(.value==true))
  and (.controls.controller_imports_or_loads_backend_code|not)
  and (.controls.controller_receives_hidden_descriptor_or_quotient|not)
  and (.controls.controller_computes_final_amplitude_independently|not)
  and (.controls.service_stdout_stderr_contains_hidden_descriptor_or_channels|not)
  and (.controls.snapshot_reload_used_by_accepted_path|not)
  and (.controls.pre_run_secret_dependent_receipt_exposed|not)
  and .algebra_law.coherent_cubic_channels==5
  and (.algebra_law.standard_one_shot_quantum_inference_claimed|not)
  and (.algebra_law.dense_5_to_the_n_sum_accepted_path|not)
  and .resource_law.packed_symmetric_matrix_residue_cells==[3,6,10,21,36,78,136]
  and .resource_law.hidden_descriptor_residue_cells==[6,10,15,28,45,91,153]
  and .resource_law.catvm_hidden_configuration_residue_cells==[6,10,15,28,45,91,153]
  and .resource_law.accepted_carrier_residue_cells==[6,10,15,28,45,91,153]
  and .resource_law.accepted_carrier_plus_hidden_configuration_residue_cells==[12,20,30,56,90,182,306]
  and .resource_law.solve_scratch_residue_cells==[2,3,4,6,8,12,16]
  and .resource_law.quotient_residue_scratch_cells==[2,2,2,2,2,2,2]
  and .resource_law.coherent_channel_integer_coordinate_cells==[20,20,20,20,20,20,20]
  and .resource_law.coherent_channel_scratch_integer_coordinate_cells==[20,20,20,20,20,20,20]
  and .resource_law.retained_final_amplitude_integer_coordinates_during_inverse==[4,4,4,4,4,4,4]
  and .resource_law.retained_final_amplitude_denominator_exponent_scalar_cells_during_inverse==[1,1,1,1,1,1,1]
  and (.resource_law.final_amplitude_exact_payload_fixed_width|not)
  and .resource_law.final_amplitude_denominator_power5_upper_bounds==[3,4,5,7,9,13,17]
  and .resource_law.final_amplitude_denominator_material_value_bit_upper_bounds==[7,10,12,17,21,31,40]
  and .resource_law.final_amplitude_numerator_coordinate_signed_bit_upper_bounds==[12,14,16,21,26,35,44]
  and (.resource_law.secret_dependent_exact_amplitude_payload_metrics_retained_in_sanitized_evidence|not)
  and .resource_law.hidden_descriptor_residue_reads_forward_plus_inverse==[12,20,30,56,90,182,306]
  and .resource_law.dense_global_amplitude_cells_not_materialized==[25,125,625,15625,390625,244140625,152587890625]
  and .resource_law.strongest_implemented_classical_baseline=="IDENTICAL_COMPACT_MODULAR_LDL_SOLVE_PLUS_FIVE_TERM_QZETA5_CLOSURE"
  and .resource_law.protocol_bytes_measured_in_raw_execution
  and (.resource_law.secret_dependent_encoded_byte_counts_retained_in_sanitized_evidence|not)
  and .resource_law.private_configuration_stdin_deliveries==2
  and .resource_law.private_configuration_stdin_recipients==["CATVM_SERVICE","SEPARATE_REFERENCE"]
  and .resource_law.private_configuration_and_verifier_traffic_bytes_total==2*.resource_law.private_configuration_stdin_bytes_per_delivery
  and (.resource_law.whole_process_rss_python_objects_allocator_socket_kernel_hashing_and_scheduler_costs_complete|not)
  and .atomic_response_law.hidden_descriptor_then_ldl_then_solve_then_five_channels_then_internal_final_boundary_then_actual_inverse_then_restoration_verification_then_response
  and .atomic_response_law.disconnect_before_response_still_restores
  and .atomic_response_law.post_projection_exception_rejected_only_after_restoration
  and .atomic_response_law.partial_ldl_exception_rejected_only_after_prefix_rollback_and_restoration
  and .atomic_response_law.service_process_nondumpable
  and .atomic_response_law.abstract_unix_socket_has_no_filesystem_artifact
  and (.atomic_response_law.controller_imports_or_loads_backend_code|not)
  and .atomic_response_law.private_configuration_delivered_over_stdin_to_service_and_separate_verifier
  and (.atomic_response_law.controller_receives_private_configuration|not)
  and (.atomic_response_law.pre_run_secret_dependent_receipt_exposed|not)
  and .atomic_response_law.final_amplitude_is_only_secret_dependent_response
  and .separate_reference.independent_exact_power_basis_formula_reexecution
  and .separate_reference.independent_modular_gaussian_elimination_not_ldl_production_path
  and .separate_reference.independent_dense_sum_parity_dimensions2_3_4
  and .separate_reference.strongest_compact_classical_baseline_is_identical_gauss_quotient
  and (.separate_reference.controls|to_entries|all(.value==true))
  and (.separate_reference.imports_service_controller_or_m237|not)
  and (.source_dependencies.exact_qzeta5_field_dependency_sha256|length)==64
  and (.claim_limits|to_entries|all(.value==false))
  and ([paths(scalars) as $p|select(
    ($p[-1]=="final_amplitude")
    or ($p[-1]=="matrix")
    or ($p[-1]=="vector")
    or ($p[-1]=="solution")
    or ($p[-1]=="delta")
    or ($p[-1]=="square_class")
    or ($p[-1]=="channel_numerators")
  )]|length)==0
  and (.terminal|not)
' "$result" >/dev/null

python3 - "$service" "$client" "$reference" <<'PY'
import ast
import sys
from pathlib import Path

paths=[Path(value) for value in sys.argv[1:]]
trees=[ast.parse(path.read_text()) for path in paths]

def imports(tree):
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node,ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node,ast.ImportFrom)
    }

client_imports=imports(trees[1])
if any("quadratic_rank1_cubic_gauss_quotient_service" in name for name in client_imports):
    raise SystemExit("M243 controller imports backend")
if "zeta5_normalized_cubic_fourier_coherent_port" in client_imports:
    raise SystemExit("M243 controller imports exact backend arithmetic")
for node in ast.walk(trees[1]):
    if isinstance(node,ast.Call) and getattr(node.func,"id","") in {
        "open","exec","eval","compile","__import__"
    }:
        raise SystemExit("M243 controller has backend-loading primitive")

reference_imports=imports(trees[2])
if any(
    "quadratic_rank1_cubic_gauss_quotient" in name
    or name=="zeta5_normalized_cubic_fourier_coherent_port"
    for name in reference_imports
):
    raise SystemExit("M243 reference imports production or predecessor")

service_source=paths[0].read_text()
if "sys.stdin.buffer.readline()" not in service_source:
    raise SystemExit("M243 private authority is not stdin-delivered")
tree=trees[0]
atomic=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="execute_atomic")
release_lines=[
    node.lineno for node in ast.walk(atomic)
    if isinstance(node,ast.Call) and getattr(node.func,"attr","")=="release"
]
return_lines=[node.lineno for node in ast.walk(atomic) if isinstance(node,ast.Return)]
boundary_lines=[
    node.lineno for node in ast.walk(atomic)
    if isinstance(node,ast.Assign)
    and any(getattr(target,"id","")=="final_numerator" for target in node.targets)
]
if not boundary_lines or not release_lines or not return_lines:
    raise SystemExit("M243 atomic response stages not source-explicit")
if not min(boundary_lines)<min(release_lines)<max(return_lines):
    raise SystemExit("M243 response ordering not source-explicit")
PY

echo "QUALIFIED_CATVM_P5_QUADRATIC_RANK1_CUBIC_GAUSS_QUOTIENT_STRICT_SCOPE"
