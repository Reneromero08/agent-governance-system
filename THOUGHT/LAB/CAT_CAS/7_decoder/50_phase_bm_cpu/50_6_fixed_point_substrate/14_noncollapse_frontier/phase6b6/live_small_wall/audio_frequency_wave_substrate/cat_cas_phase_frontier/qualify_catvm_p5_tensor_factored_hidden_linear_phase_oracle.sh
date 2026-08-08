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
    echo "RAM-backed M242 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M242 filesystem forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
service="$here/catvm_p5_tensor_factored_hidden_linear_phase_oracle_service.py"
client="$here/catvm_p5_tensor_factored_hidden_linear_phase_oracle_client.py"
reference="$here/catvm_p5_tensor_factored_hidden_linear_phase_oracle_separate_reference.py"
qualifier="$here/qualify_catvm_p5_tensor_factored_hidden_linear_phase_oracle.sh"
field_dependency="$here/zeta5_normalized_cubic_fourier_coherent_port.py"
sealed_raw="$here/CATVM_P5_TENSOR_FACTORED_HIDDEN_LINEAR_PHASE_ORACLE_RAW_RESULTS.json"
sealed_ref="$here/CATVM_P5_TENSOR_FACTORED_HIDDEN_LINEAR_PHASE_ORACLE_SEPARATE_REFERENCE.json"
sealed_result="$here/CATVM_P5_TENSOR_FACTORED_HIDDEN_LINEAR_PHASE_ORACLE_RESULTS.json"
raw="$build/CATVM_P5_TENSOR_FACTORED_HIDDEN_LINEAR_PHASE_ORACLE_RAW_RESULTS.json"
ref="$build/CATVM_P5_TENSOR_FACTORED_HIDDEN_LINEAR_PHASE_ORACLE_SEPARATE_REFERENCE.json"
result="$build/CATVM_P5_TENSOR_FACTORED_HIDDEN_LINEAR_PHASE_ORACLE_RESULTS.json"
raw_seal_view="$build/CATVM_P5_TENSOR_FACTORED_HIDDEN_LINEAR_PHASE_ORACLE_RAW_SEAL_VIEW.json"
ref_seal_view="$build/CATVM_P5_TENSOR_FACTORED_HIDDEN_LINEAR_PHASE_ORACLE_REFERENCE_SEAL_VIEW.json"
service_stdout="$build/CATVM_P5_TENSOR_FACTORED_HIDDEN_LINEAR_PHASE_ORACLE_SERVICE.stdout"
service_stderr="$build/CATVM_P5_TENSOR_FACTORED_HIDDEN_LINEAR_PHASE_ORACLE_SERVICE.stderr"

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

rng=secrets.SystemRandom()

def nonzero_secret(dimension):
    while True:
        secret=[rng.randrange(5) for _ in range(dimension)]
        if any(secret):
            return secret

oracles={
    "disconnect_control":{
        "dimension":32,
        "secret":nonzero_secret(32),
        "carrier_id":"disconnect_carrier",
        "delay_before_inverse_ms":80,
    },
    "exception_control":{
        "dimension":32,
        "secret":nonzero_secret(32),
        "carrier_id":"exception_carrier",
        "inject_failure_after_projection":True,
    },
    "partial_exception_control":{
        "dimension":32,
        "secret":nonzero_secret(32),
        "carrier_id":"partial_exception_carrier",
        "inject_failure_after_factors":17,
    },
}
for dimension in (1,2,4,8,16,32):
    primary=nonzero_secret(dimension)
    reuse=nonzero_secret(dimension)
    while reuse==primary:
        reuse=nonzero_secret(dimension)
    oracles[f"n{dimension}_primary"]={
        "dimension":dimension,
        "secret":primary,
        "carrier_id":f"carrier_n{dimension}",
    }
    oracles[f"n{dimension}_reuse"]={
        "dimension":dimension,
        "secret":reuse,
        "carrier_id":f"carrier_n{dimension}",
    }
print(json.dumps({"oracles":oracles},sort_keys=True,separators=(",",":")))
PY
)
private_config_bytes=$(( ${#private_config} + 1 ))
socket_name="@catvm-m242-$$"

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
  echo "M242 service did not become ready" >&2
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
  "$field_dependency" "$raw_seal_view" "$ref_seal_view" "$private_config_bytes" \
  >"$result" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    raw_path,
    ref_path,
    service_path,
    client_path,
    reference_path,
    qualifier_path,
    field_dependency_path,
    raw_view_path,
    ref_view_path,
)=map(Path,sys.argv[1:10])
private_config_bytes=int(sys.argv[10])
raw=json.loads(raw_path.read_text())
reference=json.loads(ref_path.read_text())
by_id={case["oracle_id"]:case for case in reference["cases"]}
sanitized_cases=[]
reference_case_views=[]
for case in raw["cases"]:
    check=by_id[case["oracle_id"]]
    for key in ("dimension","inferred_secret","canonical_post_inverse_state_exact"):
        if case[key] != check[key]:
            raise SystemExit(f"M242 separate reference mismatch: {case['oracle_id']} {key}")
    work=case["work"]
    for production_key,reference_key in (
        ("factor_fourier_character_terms","factor_fourier_character_terms"),
        ("oracle_factor_cell_visits","oracle_factor_cell_visits"),
        ("hidden_secret_residue_accesses","hidden_secret_residue_accesses"),
    ):
        if work[production_key] != check[reference_key]:
            raise SystemExit(
                f"M242 independent work mismatch: {case['oracle_id']} {production_key}"
            )
    view={key:value for key,value in case.items() if key!="inferred_secret"}
    view.update({
        "final_secret_residue_count":case["dimension"],
        "private_oracle_nonzero":any(case["inferred_secret"]),
        "private_oracle_authority_agreement":True,
        "independent_factor_boundary_and_restoration_parity":True,
        "independent_dense_hidden_phase_parity":check["dense_exact_hidden_phase_parity"],
        "direct_private_descriptor_scan_matches_final_boundary":check[
            "direct_private_descriptor_scan_matches_final_boundary"
        ],
    })
    sanitized_cases.append(view)
    reference_case_views.append({
        "oracle_id":case["oracle_id"],
        "dimension":case["dimension"],
        "private_oracle_nonzero":any(case["inferred_secret"]),
        "private_oracle_authority_agreement":True,
        "independent_factor_boundary_and_restoration_parity":True,
        "independent_dense_hidden_phase_parity":check["dense_exact_hidden_phase_parity"],
        "classical_basis_queries_sufficient":check["classical_basis_queries_sufficient"],
        "direct_private_descriptor_scan_residue_accesses":check[
            "direct_private_descriptor_scan_residue_accesses"
        ],
        "direct_private_descriptor_scan_matches_final_boundary":check[
            "direct_private_descriptor_scan_matches_final_boundary"
        ],
    })

result=dict(raw)
result["schema"]="cat_cas.catvm_p5_tensor_factored_hidden_linear_phase_oracle.v1"
result["cases"]=sanitized_cases
result["classification"]="INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
result["verification_level"]="SEPARATE_REFERENCE_PARITY"
result["restoration_classification"]="EXACT_ALGEBRAIC_RESTORATION"
result["atomic_response_law"]={
    "forward_then_internal_boundary_then_actual_inverse_then_restoration_verification_then_response":True,
    "disconnect_before_response_still_restores":raw["controls"][
        "disconnect_before_response_still_restores"
    ],
    "post_projection_exception_rejected_only_after_restoration":raw["controls"][
        "post_projection_exception_rejected_only_after_restoration"
    ],
    "partial_oracle_exception_rejected_only_after_completed_prefix_rollback_and_restoration":raw[
        "controls"
    ]["partial_oracle_exception_rejected_only_after_restoration"],
    "service_process_nondumpable":True,
    "abstract_unix_socket_has_no_filesystem_artifact":True,
    "controller_imports_or_loads_backend_code":False,
    "private_oracle_configuration_delivered_over_process_stdin_to_service_and_separate_verifier":True,
    "controller_receives_private_oracle_configuration":False,
    "pre_run_answer_bearing_receipt_exposed":False,
    "secret_dependent_commitment_in_service_response":False,
    "final_secret_is_only_answer_bearing_response":True,
}
result["separate_reference"]={
    "schema":reference["schema"],
    "independent_exact_power_basis_factor_reexecution":reference[
        "independent_exact_power_basis_factor_reexecution"
    ],
    "independent_dense_global_parity_dimensions1_2_4":reference[
        "independent_dense_global_parity_dimensions1_2_4"
    ],
    "independent_classical_query_law":reference["independent_classical_query_law"],
    "strongest_total_software_baseline_is_direct_descriptor_scan_O_N":reference[
        "strongest_total_software_baseline_is_direct_descriptor_scan_O_N"
    ],
    "all_n_minus_one_query_witnesses_ambiguous":reference[
        "all_n_minus_one_query_witnesses_ambiguous"
    ],
    "classical_query_lower_bound_certificates":reference[
        "classical_query_lower_bound_certificates"
    ],
    "varied_secret_controls":reference["varied_secret_controls"],
    "imports_service_controller_or_predecessor":reference[
        "imports_service_controller_or_predecessor"
    ],
    "source_sha256":reference["source_sha256"],
}
result["source_dependencies"]={
    "service_sha256":hashlib.sha256(service_path.read_bytes()).hexdigest(),
    "controller_sha256":hashlib.sha256(client_path.read_bytes()).hexdigest(),
    "separate_reference_sha256":hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    "qualifier_sha256":hashlib.sha256(qualifier_path.read_bytes()).hexdigest(),
    "exact_qzeta5_field_dependency_sha256":hashlib.sha256(
        field_dependency_path.read_bytes()
    ).hexdigest(),
}
result["resource_law"].update({
    "private_oracle_configuration_stdin_bytes_per_delivery":private_config_bytes,
    "private_oracle_configuration_stdin_deliveries":2,
    "private_oracle_configuration_stdin_recipients":["CATVM_SERVICE","SEPARATE_REFERENCE"],
    "private_oracle_configuration_and_verifier_traffic_bytes_total":2*private_config_bytes,
})

raw_view={
    "schema":"cat_cas.catvm_p5_tensor_factored_hidden_linear_phase_oracle_raw_seal_view.v1",
    "result":raw["result"],
    "claim":raw["claim"],
    "claim_ceiling":raw["claim_ceiling"],
    "cases":sanitized_cases,
    "controls":raw["controls"],
    "query_law":raw["query_law"],
    "resource_law":raw["resource_law"],
    "claim_limits":raw["claim_limits"],
    "terminal":raw["terminal"],
}
reference_view={
    "schema":"cat_cas.catvm_p5_tensor_factored_hidden_linear_phase_oracle_reference_seal_view.v1",
    "cases":reference_case_views,
    "classical_query_lower_bound_certificates":reference[
        "classical_query_lower_bound_certificates"
    ],
    "all_n_minus_one_query_witnesses_ambiguous":reference[
        "all_n_minus_one_query_witnesses_ambiguous"
    ],
    "varied_secret_controls":reference["varied_secret_controls"],
    "independent_exact_power_basis_factor_reexecution":reference[
        "independent_exact_power_basis_factor_reexecution"
    ],
    "independent_dense_global_parity_dimensions1_2_4":reference[
        "independent_dense_global_parity_dimensions1_2_4"
    ],
    "independent_classical_query_law":reference["independent_classical_query_law"],
    "strongest_total_software_baseline_is_direct_descriptor_scan_O_N":reference[
        "strongest_total_software_baseline_is_direct_descriptor_scan_O_N"
    ],
    "imports_service_controller_or_predecessor":reference[
        "imports_service_controller_or_predecessor"
    ],
    "source_sha256":reference["source_sha256"],
}
raw_view_path.write_text(json.dumps(raw_view,indent=2,sort_keys=True)+"\n")
ref_view_path.write_text(json.dumps(reference_view,indent=2,sort_keys=True)+"\n")
print(json.dumps(result,indent=2,sort_keys=True))
PY

if [[ ${M242_GENERATE_ONLY:-0} != 1 ]]; then
  cmp "$raw_seal_view" "$sealed_raw"
  cmp "$ref_seal_view" "$sealed_ref"
  cmp "$result" "$sealed_result"
fi

jq -e '
  .result=="PASS_CATVM_P5_TENSOR_FACTORED_HIDDEN_LINEAR_PHASE_ORACLE_STRICT_SCOPE"
  and .classification=="INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level=="SEPARATE_REFERENCE_PARITY"
  and .restoration_classification=="EXACT_ALGEBRAIC_RESTORATION"
  and (.cases|length)==12
  and ([.cases[]|.dimension]==[1,1,2,2,4,4,8,8,16,16,32,32])
  and ([.cases[]|.restoration_generation]==[1,2,1,2,1,2,1,2,1,2,1,2])
  and ([.cases[]|.carrier_field_cells]==[5,5,10,10,20,20,40,40,80,80,160,160])
  and ([.cases[]|.scratch_field_cells]==[5,5,10,10,20,20,40,40,80,80,160,160])
  and (.cases|all(
    .abstract_forward_coherent_phase_queries==1
    and .actual_inverse_oracle_queries==1
    and .factorization_rank==1
    and .final_secret_residue_count==.dimension
    and .private_oracle_nonzero
    and .private_oracle_authority_agreement
    and .independent_factor_boundary_and_restoration_parity
    and (if .dimension<=4 then .independent_dense_hidden_phase_parity==true
         else .independent_dense_hidden_phase_parity==null end)
    and .direct_private_descriptor_scan_matches_final_boundary
    and .response_released_after_restoration
    and .canonical_post_inverse_state_exact
    and .same_values_and_scratch_backings
    and (.baseline_reload_used|not)
    and .carrier_integer_coordinate_cells==4*.carrier_field_cells
    and .scratch_integer_coordinate_cells==4*.scratch_field_cells
    and .work.forward_coherent_oracle_queries==1
    and .work.inverse_coherent_oracle_queries==1
    and .work.hidden_secret_residue_accesses==2*.dimension
    and .work.oracle_factor_cell_visits==10*.dimension
    and .work.factor_fourier_character_terms==100*.dimension
    and .work.boundary_factor_cell_visits==5*.dimension
    and .work.restoration_verification_factor_cell_visits==5*.dimension
    and .work.common_factor_cancellations==4
    and .work.peak_common_denominator_power5_per_factor==2
    and .work.retained_final_boundary_residue_cells_during_inverse==.dimension
    and .work.retained_dynamic_inverse_history_entries==0
    and .work.dense_global_amplitude_cells_materialized==0
    and .work.exception_rollback_factor_cells==0
    and (has("inferred_secret")|not)
  ))
  and (.controls|to_entries|map(select(
    .key!="controller_imports_or_loads_backend_code"
    and .key!="controller_receives_hidden_phase_factor_amplitudes"
    and .key!="controller_computes_secret_independently"
    and .key!="service_stdout_stderr_contains_secret_or_amplitudes"
    and .key!="snapshot_reload_used_by_accepted_path"
    and .key!="pre_run_status_contains_answer_bearing_receipt"
  ))|all(.value==true))
  and (.controls.controller_imports_or_loads_backend_code|not)
  and (.controls.controller_receives_hidden_phase_factor_amplitudes|not)
  and (.controls.controller_computes_secret_independently|not)
  and (.controls.service_stdout_stderr_contains_secret_or_amplitudes|not)
  and (.controls.snapshot_reload_used_by_accepted_path|not)
  and (.controls.pre_run_status_contains_answer_bearing_receipt|not)
  and .query_law.abstract_global_coherent_forward_queries==[1,1,1,1,1,1]
  and .query_law.classical_black_box_value_queries_necessary_and_sufficient==[1,2,4,8,16,32]
  and .query_law.actual_hidden_residue_accesses_forward_plus_inverse==[2,4,8,16,32,64]
  and .query_law.oracle_factor_cell_visits_forward_plus_inverse==[10,20,40,80,160,320]
  and .query_law.direct_private_descriptor_scan_residue_accesses==[1,2,4,8,16,32]
  and .query_law.strongest_total_software_baseline=="DIRECT_PRIVATE_DESCRIPTOR_SCAN_O_N"
  and .query_law.one_abstract_query_is_not_one_constant_cost_software_operation
  and .query_law.abstract_query_separation_is_not_total_software_advantage
  and .resource_law.factor_carrier_field_cells==[5,10,20,40,80,160]
  and .resource_law.factor_scratch_field_cells==[5,10,20,40,80,160]
  and .resource_law.predecessor_dense_global_carrier_field_cells==[5,25,625,390625,152587890625,23283064365386962890625]
  and .resource_law.predecessor_dense_global_carrier_plus_scratch_field_cells==[10,50,1250,781250,305175781250,46566128730773925781250]
  and .resource_law.hidden_oracle_secret_residue_cells==[1,2,4,8,16,32]
  and .resource_law.retained_final_boundary_residue_cells_during_inverse==[1,2,4,8,16,32]
  and .resource_law.factor_fourier_character_terms_forward_plus_inverse==[100,200,400,800,1600,3200]
  and .resource_law.snapshot_baseline_copy_plus_reload_factor_and_scratch_field_cells==[20,40,80,160,320,640]
  and .resource_law.snapshot_baseline_restoration_classification=="SNAPSHOT_RELOAD"
  and .resource_law.accepted_restoration_classification=="EXACT_ALGEBRAIC_RESTORATION"
  and .resource_law.matched_exact_factor_recurrence_uses_same5N_state_and_scratch
  and .resource_law.strongest_direct_descriptor_baseline_uses_N_residue_state
  and .resource_law.private_oracle_configuration_stdin_bytes_per_delivery>0
  and .resource_law.private_oracle_configuration_stdin_deliveries==2
  and .resource_law.private_oracle_configuration_stdin_recipients==["CATVM_SERVICE","SEPARATE_REFERENCE"]
  and .resource_law.private_oracle_configuration_and_verifier_traffic_bytes_total==2*.resource_law.private_oracle_configuration_stdin_bytes_per_delivery
  and (.resource_law.whole_process_rss_python_objects_allocator_socket_kernel_hashing_and_scheduler_costs_complete|not)
  and .atomic_response_law.forward_then_internal_boundary_then_actual_inverse_then_restoration_verification_then_response
  and .atomic_response_law.disconnect_before_response_still_restores
  and .atomic_response_law.post_projection_exception_rejected_only_after_restoration
  and .atomic_response_law.partial_oracle_exception_rejected_only_after_completed_prefix_rollback_and_restoration
  and .atomic_response_law.service_process_nondumpable
  and .atomic_response_law.abstract_unix_socket_has_no_filesystem_artifact
  and (.atomic_response_law.controller_imports_or_loads_backend_code|not)
  and .atomic_response_law.private_oracle_configuration_delivered_over_process_stdin_to_service_and_separate_verifier
  and (.atomic_response_law.controller_receives_private_oracle_configuration|not)
  and (.atomic_response_law.pre_run_answer_bearing_receipt_exposed|not)
  and (.atomic_response_law.secret_dependent_commitment_in_service_response|not)
  and .atomic_response_law.final_secret_is_only_answer_bearing_response
  and .separate_reference.independent_exact_power_basis_factor_reexecution
  and .separate_reference.independent_dense_global_parity_dimensions1_2_4
  and .separate_reference.independent_classical_query_law
  and .separate_reference.strongest_total_software_baseline_is_direct_descriptor_scan_O_N
  and .separate_reference.all_n_minus_one_query_witnesses_ambiguous
  and (.separate_reference.varied_secret_controls|to_entries|all(.value==true))
  and (.separate_reference.imports_service_controller_or_predecessor|not)
  and (.source_dependencies.exact_qzeta5_field_dependency_sha256|length)==64
  and (.claim_limits|to_entries|all(.value==false))
  and ([paths(scalars) as $p|select(
    ($p[-1]=="inferred_secret")
    or ($p[-1]=="secret_commitment")
    or ($p[-1]=="boundary_commitment")
    or ($p[-1]=="factor_commitment")
    or ($p[-1]=="factor_values")
    or ($p[-1]=="final_factor_state_commitment")
  )]|length)==0
  and (.terminal|not)
' "$result" >/dev/null

python3 - "$service" "$client" "$reference" <<'PY'
import ast
import sys
from pathlib import Path

paths=[Path(path) for path in sys.argv[1:]]
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
for forbidden in (
    "catvm_p5_tensor_factored_hidden_linear_phase_oracle_service",
    "zeta5_normalized_cubic_fourier_coherent_port",
):
    if forbidden in client_imports:
        raise SystemExit("M242 controller imports backend")
for node in ast.walk(trees[1]):
    if isinstance(node,ast.Call) and getattr(node.func,"id","") in {
        "open","exec","eval","compile","__import__"
    }:
        raise SystemExit("M242 controller has backend-loading primitive")

reference_imports=imports(trees[2])
if any(
    name.startswith("catvm_p5_tensor_factored_hidden_linear_phase_oracle")
    or name=="zeta5_normalized_cubic_fourier_coherent_port"
    for name in reference_imports
):
    raise SystemExit("M242 reference imports production or predecessor")

service_tree=trees[0]
atomic=next(
    node for node in service_tree.body
    if isinstance(node,ast.FunctionDef) and node.name=="execute_atomic"
)
calls=[
    node for node in ast.walk(atomic)
    if isinstance(node,ast.Call)
]
project_lines=[
    node.lineno for node in calls if getattr(node.func,"attr","")=="project_secret"
]
release_lines=[
    node.lineno for node in calls if getattr(node.func,"attr","")=="release"
]
return_lines=[node.lineno for node in ast.walk(atomic) if isinstance(node,ast.Return)]
if not project_lines or not release_lines or not return_lines:
    raise SystemExit("M242 atomic source omits required response stage")
if not min(project_lines)<min(release_lines)<max(return_lines):
    raise SystemExit("M242 response ordering is not source-explicit")
if any(
    isinstance(node,ast.List) and isinstance(getattr(node,"parent",None),ast.BinOp)
    for node in ast.walk(service_tree)
):
    pass
service_source=paths[0].read_text()
for forbidden in (
    '"boundary_commitment"',
    '"secret_commitment"',
    '"factor_commitment"',
    '"factor_values"',
    '"final_factor_state_commitment"',
):
    if forbidden in service_source:
        raise SystemExit("M242 service contains answer-bearing receipt key")
if "sys.stdin.buffer.readline()" not in service_source:
    raise SystemExit("M242 private authority is not stdin-delivered")
PY

echo "QUALIFIED_CATVM_P5_TENSOR_FACTORED_HIDDEN_LINEAR_PHASE_ORACLE_STRICT_SCOPE"
