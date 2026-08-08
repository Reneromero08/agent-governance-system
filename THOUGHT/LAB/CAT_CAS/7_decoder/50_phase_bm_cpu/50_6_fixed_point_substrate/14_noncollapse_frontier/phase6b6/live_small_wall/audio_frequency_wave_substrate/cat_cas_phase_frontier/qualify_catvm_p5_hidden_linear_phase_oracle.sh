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
    echo "RAM-backed M241 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M241 filesystem forbidden" >&2
    exit 2
    ;;
esac
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
service="$here/catvm_p5_hidden_linear_phase_oracle_service.py"
client="$here/catvm_p5_hidden_linear_phase_oracle_client.py"
reference="$here/catvm_p5_hidden_linear_phase_oracle_separate_reference.py"
qualifier="$here/qualify_catvm_p5_hidden_linear_phase_oracle.sh"
sealed_raw="$here/CATVM_P5_HIDDEN_LINEAR_PHASE_ORACLE_RAW_RESULTS.json"
sealed_ref="$here/CATVM_P5_HIDDEN_LINEAR_PHASE_ORACLE_SEPARATE_REFERENCE.json"
sealed_result="$here/CATVM_P5_HIDDEN_LINEAR_PHASE_ORACLE_RESULTS.json"
raw="$build/CATVM_P5_HIDDEN_LINEAR_PHASE_ORACLE_RAW_RESULTS.json"
ref="$build/CATVM_P5_HIDDEN_LINEAR_PHASE_ORACLE_SEPARATE_REFERENCE.json"
result="$build/CATVM_P5_HIDDEN_LINEAR_PHASE_ORACLE_RESULTS.json"
raw_seal_view="$build/CATVM_P5_HIDDEN_LINEAR_PHASE_ORACLE_RAW_SEAL_VIEW.json"
ref_seal_view="$build/CATVM_P5_HIDDEN_LINEAR_PHASE_ORACLE_REFERENCE_SEAL_VIEW.json"
service_stdout="$build/CATVM_P5_HIDDEN_LINEAR_PHASE_ORACLE_SERVICE.stdout"
service_stderr="$build/CATVM_P5_HIDDEN_LINEAR_PHASE_ORACLE_SERVICE.stderr"
mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
run_env=(
  env
  TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp"
  XDG_CACHE_HOME="$build/xdg-cache"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache"
)
private_config=$("${run_env[@]}" python3 - <<'PY'
import json, secrets
rng=secrets.SystemRandom()
def nonzero_secret(dimension):
    while True:
        value=[rng.randrange(5) for _ in range(dimension)]
        if any(value):
            return value
oracles={
    "disconnect_control":{"dimension":4,"carrier_id":"disconnect_carrier","delay_before_inverse_ms":80},
    "exception_control":{"dimension":4,"carrier_id":"exception_carrier","inject_failure_after_projection":True},
}
for dimension in (1,2,3,4):
    primary=nonzero_secret(dimension)
    reuse=nonzero_secret(dimension)
    while reuse==primary:
        reuse=nonzero_secret(dimension)
    oracles[f"n{dimension}_primary"]={"dimension":dimension,"secret":primary,"carrier_id":f"carrier_n{dimension}"}
    oracles[f"n{dimension}_reuse"]={"dimension":dimension,"secret":reuse,"carrier_id":f"carrier_n{dimension}"}
oracles["disconnect_control"]["secret"]=nonzero_secret(4)
oracles["exception_control"]["secret"]=nonzero_secret(4)
print(json.dumps({"oracles":oracles},sort_keys=True,separators=(",",":")))
PY
)
private_config_bytes=$(( ${#private_config} + 1 ))
socket_name="@catvm-m241-$$"
printf '%s\n' "$private_config" | "${run_env[@]}" nice -n 10 ionice -c 3 python3 "$service" "$socket_name" >"$service_stdout" 2>"$service_stderr" &
service_pid=$!
ready=false
for _ in $(seq 1 100); do
  if "${run_env[@]}" python3 - "$socket_name" <<'PY'
import socket, sys
name=sys.argv[1]
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
try:
    s.connect("\0"+name[1:])
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
  echo "M241 service did not become ready" >&2
  exit 2
fi
"${run_env[@]}" nice -n 10 ionice -c 3 python3 "$client" "$socket_name" >"$raw"
wait "$service_pid"
[[ ! -s "$service_stdout" ]]
[[ ! -s "$service_stderr" ]]
printf '%s\n' "$private_config" | "${run_env[@]}" nice -n 10 ionice -c 3 python3 "$reference" "$raw" >"$ref"
"${run_env[@]}" python3 - "$raw" "$ref" "$service" "$client" "$reference" "$qualifier" "$raw_seal_view" "$ref_seal_view" "$private_config_bytes" >"$result" <<'PY'
import hashlib, json, sys
from pathlib import Path
raw_path, ref_path, service_path, client_path, reference_path, qualifier_path, raw_view_path, ref_view_path = map(Path, sys.argv[1:9])
private_config_bytes=int(sys.argv[9])
raw=json.loads(raw_path.read_text())
reference=json.loads(ref_path.read_text())
by_id={case["oracle_id"]:case for case in reference["cases"]}
sanitized_cases=[]
reference_case_views=[]
for case in raw["cases"]:
    check=by_id[case["oracle_id"]]
    for key in ("dimension","inferred_secret","boundary_commitment","final_basis_state_commitment","canonical_post_inverse_state_exact"):
        if case[key] != check[key]:
            raise SystemExit(f"M241 separate reference mismatch: {case['oracle_id']} {key}")
    view={key:value for key,value in case.items() if key not in {"inferred_secret","boundary_commitment","final_basis_state_commitment"}}
    view.update({
        "final_secret_residue_count":case["dimension"],
        "private_oracle_nonzero":any(case["inferred_secret"]),
        "private_oracle_authority_agreement":True,
        "boundary_commitment_reference_parity":True,
        "final_basis_state_commitment_reference_parity":True,
    })
    sanitized_cases.append(view)
    reference_case_views.append({
        "oracle_id":case["oracle_id"],
        "dimension":case["dimension"],
        "private_oracle_nonzero":any(case["inferred_secret"]),
        "private_oracle_authority_agreement":True,
        "independent_boundary_and_commitment_parity":True,
        "independent_exact_restoration":check["canonical_post_inverse_state_exact"],
        "classical_basis_queries_sufficient":check["classical_basis_queries_sufficient"],
    })
result=dict(raw)
result["schema"]="cat_cas.catvm_p5_hidden_linear_phase_oracle.v1"
result["cases"]=sanitized_cases
result["classification"]="INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
result["verification_level"]="SEPARATE_REFERENCE_PARITY"
result["restoration_classification"]="EXACT_ALGEBRAIC_RESTORATION"
result["atomic_response_law"]={
    "forward_then_internal_boundary_then_actual_inverse_then_restoration_verification_then_response": True,
    "disconnect_before_response_still_restores": raw["controls"]["disconnect_before_response_still_restores"],
    "post_projection_exception_rejected_only_after_restoration": raw["controls"]["post_projection_exception_rejected_only_after_restoration"],
    "service_process_nondumpable": True,
    "abstract_unix_socket_has_no_filesystem_artifact": True,
    "controller_imports_or_loads_backend_code": False,
    "oracle_configuration_delivered_only_to_backend_stdin": True,
    "pre_run_enumerable_secret_commitment_exposed": False,
    "final_secret_is_the_only_answer_bearing_response": True,
}
result["separate_reference"]={
    "schema": reference["schema"],
    "independent_exact_power_basis_reexecution": reference["independent_exact_power_basis_reexecution"],
    "independent_classical_query_law": reference["independent_classical_query_law"],
    "all_n_minus_one_query_witnesses_ambiguous": reference["all_n_minus_one_query_witnesses_ambiguous"],
    "classical_query_lower_bound_certificates": reference["classical_query_lower_bound_certificates"],
    "imports_service_or_controller": reference["imports_service_or_controller"],
    "source_sha256": reference["source_sha256"],
}
result["source_dependencies"]={
    "service_sha256": hashlib.sha256(service_path.read_bytes()).hexdigest(),
    "controller_sha256": hashlib.sha256(client_path.read_bytes()).hexdigest(),
    "separate_reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    "qualifier_sha256": hashlib.sha256(qualifier_path.read_bytes()).hexdigest(),
}
result["resource_law"].update({
    "private_oracle_configuration_stdin_bytes_per_delivery": private_config_bytes,
    "private_oracle_configuration_stdin_deliveries": 2,
    "private_oracle_configuration_and_verifier_traffic_bytes_total": 2 * private_config_bytes,
})
raw_view={
    "schema":"cat_cas.catvm_p5_hidden_linear_phase_oracle_raw_seal_view.v1",
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
    "schema":"cat_cas.catvm_p5_hidden_linear_phase_oracle_reference_seal_view.v1",
    "cases":reference_case_views,
    "classical_query_lower_bound_certificates":reference["classical_query_lower_bound_certificates"],
    "all_n_minus_one_query_witnesses_ambiguous":reference["all_n_minus_one_query_witnesses_ambiguous"],
    "independent_exact_power_basis_reexecution":reference["independent_exact_power_basis_reexecution"],
    "independent_classical_query_law":reference["independent_classical_query_law"],
    "imports_service_or_controller":reference["imports_service_or_controller"],
    "source_sha256":reference["source_sha256"],
}
raw_view_path.write_text(json.dumps(raw_view,indent=2,sort_keys=True)+"\n")
ref_view_path.write_text(json.dumps(reference_view,indent=2,sort_keys=True)+"\n")
print(json.dumps(result,indent=2,sort_keys=True))
PY

if [[ ${M241_GENERATE_ONLY:-0} != 1 ]]; then
  cmp "$raw_seal_view" "$sealed_raw"
  cmp "$ref_seal_view" "$sealed_ref"
  cmp "$result" "$sealed_result"
fi

jq -e '
  .result=="PASS_CATVM_EXACT_P5_HIDDEN_LINEAR_PHASE_ORACLE_QUERY_DIAGNOSTIC_STRICT_SCOPE"
  and .classification=="INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level=="SEPARATE_REFERENCE_PARITY"
  and .restoration_classification=="EXACT_ALGEBRAIC_RESTORATION"
  and (.cases|length)==8
  and ([.cases[]|.dimension]==[1,1,2,2,3,3,4,4])
  and ([.cases[]|.restoration_generation]==[1,2,1,2,1,2,1,2])
  and ([.cases[]|.carrier_field_cells]==[5,5,25,25,125,125,625,625])
  and ([.cases[]|.scratch_field_cells]==[5,5,25,25,125,125,625,625])
  and (.cases|all(
    .abstract_forward_coherent_phase_queries==1
    and .actual_inverse_oracle_queries==1
    and .final_secret_residue_count==.dimension
    and .private_oracle_nonzero
    and .private_oracle_authority_agreement
    and .boundary_commitment_reference_parity
    and .final_basis_state_commitment_reference_parity
    and (has("inferred_secret")|not)
    and (has("boundary_commitment")|not)
    and (has("final_basis_state_commitment")|not)
    and .response_released_after_restoration
    and .canonical_post_inverse_state_exact
    and .same_values_and_scratch_backings
    and (.baseline_reload_used|not)
    and .work.forward_coherent_oracle_queries==1
    and .work.inverse_coherent_oracle_queries==1
    and .work.retained_dynamic_inverse_history_entries==0
  ))
  and (.controls|to_entries|map(select(
    .key!="controller_imports_or_loads_backend_code"
    and .key!="controller_receives_hidden_phase_amplitudes"
    and .key!="controller_computes_secret_independently"
    and .key!="service_stdout_stderr_contains_secret_or_amplitudes"
    and .key!="snapshot_reload_used_by_accepted_path"
    and .key!="pre_run_status_contains_enumerable_secret_commitment"
  ))|all(.value==true))
  and (.controls.controller_imports_or_loads_backend_code|not)
  and (.controls.controller_receives_hidden_phase_amplitudes|not)
  and (.controls.controller_computes_secret_independently|not)
  and (.controls.service_stdout_stderr_contains_secret_or_amplitudes|not)
  and (.controls.snapshot_reload_used_by_accepted_path|not)
  and (.controls.pre_run_status_contains_enumerable_secret_commitment|not)
  and .query_law.coherent_forward_phase_queries==[1,1,1,1]
  and .query_law.classical_deterministic_value_queries_necessary_and_sufficient==[1,2,3,4]
  and .query_law.coherent_query_acts_on_all5_TO_THE_N_BASIS_CELLS
  and .query_law.oracle_query_separation_is_not_total_software_advantage
  and .resource_law.carrier_field_cells==[5,25,125,625]
  and .resource_law.scratch_field_cells==[5,25,125,625]
  and .resource_law.hidden_oracle_secret_residue_cells==[1,2,3,4]
  and .resource_law.abstract_forward_query_count==1
  and .resource_law.actual_inverse_query_count==1
  and .resource_law.snapshot_baseline_would_copy_and_reload_carrier_cells==[10,50,250,1250]
  and .resource_law.snapshot_baseline_restoration_classification=="SNAPSHOT_RELOAD"
  and .resource_law.warm_direct_phase_software_uses_same5_TO_THE_N_amplitude_recurrence
  and .resource_law.warm_isolated_boundary_overhead_counted_in_protocol_bytes
  and .resource_law.private_oracle_configuration_stdin_bytes_per_delivery>0
  and .resource_law.private_oracle_configuration_stdin_deliveries==2
  and .resource_law.private_oracle_configuration_and_verifier_traffic_bytes_total==2*.resource_law.private_oracle_configuration_stdin_bytes_per_delivery
  and (.resource_law.whole_process_rss_allocator_socket_kernel_and_scheduler_costs_complete|not)
  and .atomic_response_law.forward_then_internal_boundary_then_actual_inverse_then_restoration_verification_then_response
  and .atomic_response_law.disconnect_before_response_still_restores
  and .atomic_response_law.post_projection_exception_rejected_only_after_restoration
  and .atomic_response_law.service_process_nondumpable
  and .atomic_response_law.abstract_unix_socket_has_no_filesystem_artifact
  and (.atomic_response_law.controller_imports_or_loads_backend_code|not)
  and .atomic_response_law.oracle_configuration_delivered_only_to_backend_stdin
  and (.atomic_response_law.pre_run_enumerable_secret_commitment_exposed|not)
  and .atomic_response_law.final_secret_is_the_only_answer_bearing_response
  and .separate_reference.independent_exact_power_basis_reexecution
  and .separate_reference.independent_classical_query_law
  and .separate_reference.all_n_minus_one_query_witnesses_ambiguous
  and (.separate_reference.imports_service_or_controller|not)
  and (.claim_limits|to_entries|all(.value==false))
  and ([paths(scalars) as $p | select(
    ($p[-1]=="inferred_secret")
    or ($p[-1]=="boundary_commitment")
    or ($p[-1]=="final_basis_state_commitment")
  )] | length)==0
  and (.terminal|not)
' "$result" >/dev/null

python3 - "$service" "$client" "$reference" <<'PY'
import ast, sys
from pathlib import Path
trees=[ast.parse(Path(path).read_text()) for path in sys.argv[1:]]
client=trees[1]
imports={
    alias.name for node in ast.walk(client) if isinstance(node,ast.Import) for alias in node.names
} | {node.module or "" for node in ast.walk(client) if isinstance(node,ast.ImportFrom)}
for forbidden in ("catvm_p5_hidden_linear_phase_oracle_service","zeta5_normalized_cubic_fourier_coherent_port"):
    if forbidden in imports:
        raise SystemExit("M241 controller imports backend")
for node in ast.walk(client):
    if isinstance(node,ast.Call) and getattr(node.func,"id","") in {"open","exec","eval","compile","__import__"}:
        raise SystemExit("M241 controller has backend-loading primitive")
reference=trees[2]
reference_imports={
    alias.name for node in ast.walk(reference) if isinstance(node,ast.Import) for alias in node.names
} | {node.module or "" for node in ast.walk(reference) if isinstance(node,ast.ImportFrom)}
if any(name.startswith("catvm_p5_hidden_linear_phase_oracle") for name in reference_imports):
    raise SystemExit("M241 reference imports production")
service=trees[0]
atomic=next(node for node in service.body if isinstance(node,ast.FunctionDef) and node.name=="execute_atomic")
calls=[getattr(node.func,"attr",getattr(node.func,"id","")) for node in ast.walk(atomic) if isinstance(node,ast.Call)]
for required in ("project_secret","release"):
    if required not in calls:
        raise SystemExit("M241 atomic source omits required stage")
PY
echo "QUALIFIED_CATVM_P5_HIDDEN_LINEAR_PHASE_ORACLE_STRICT_SCOPE"
