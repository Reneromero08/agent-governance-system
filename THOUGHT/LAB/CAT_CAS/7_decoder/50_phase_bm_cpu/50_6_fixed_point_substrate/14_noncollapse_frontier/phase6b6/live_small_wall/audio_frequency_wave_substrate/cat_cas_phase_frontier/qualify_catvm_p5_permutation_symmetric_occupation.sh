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
    echo "RAM-backed M246 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M246 filesystem forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
service="$here/catvm_p5_permutation_symmetric_occupation_service.py"
client="$here/catvm_p5_permutation_symmetric_occupation_client.py"
reference="$here/catvm_p5_permutation_symmetric_occupation_separate_reference.py"
qualifier="$here/qualify_catvm_p5_permutation_symmetric_occupation.sh"
field_dependency="$here/zeta5_normalized_cubic_fourier_coherent_port.py"
sealed_raw="$here/CATVM_P5_PERMUTATION_SYMMETRIC_OCCUPATION_RAW_RESULTS.json"
sealed_ref="$here/CATVM_P5_PERMUTATION_SYMMETRIC_OCCUPATION_SEPARATE_REFERENCE.json"
sealed_result="$here/CATVM_P5_PERMUTATION_SYMMETRIC_OCCUPATION_RESULTS.json"
raw="$build/CATVM_P5_PERMUTATION_SYMMETRIC_OCCUPATION_RAW_RESULTS.json"
ref="$build/CATVM_P5_PERMUTATION_SYMMETRIC_OCCUPATION_SEPARATE_REFERENCE.json"
result="$build/CATVM_P5_PERMUTATION_SYMMETRIC_OCCUPATION_RESULTS.json"
service_stdout="$build/CATVM_P5_PERMUTATION_SYMMETRIC_OCCUPATION_SERVICE.stdout"
service_stderr="$build/CATVM_P5_PERMUTATION_SYMMETRIC_OCCUPATION_SERVICE.stderr"

mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
run_env=(
  env
  TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp"
  XDG_CACHE_HOME="$build/xdg-cache"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache"
)

private_config=$(${run_env[@]} python3 - <<'PY'
import json

def primary(n, output):
    return {
        "rails":n, "depth":3,
        "lambdas":[1,2,3], "quadratics":[0,1,2], "rungs":[1,2,1],
        "couplings":[2,3], "output_occupation":output,
        "carrier_id":f"occupation_n{n}",
    }

oracles={
    "primary_n2":primary(2,[1,1,0,0,0]),
    "primary_n3":primary(3,[1,1,1,0,0]),
    "primary_n4":primary(4,[1,1,1,1,0]),
    "primary_n6":primary(6,[2,1,1,1,1]),
}
reuse={
    "rails":6, "depth":3,
    "lambdas":[4,1,2], "quadratics":[3,4,1], "rungs":[2,3,4],
    "couplings":[4,2], "output_occupation":[0,2,1,2,1],
    "carrier_id":"occupation_n6",
}
oracles["reuse_n6"]=reuse
oracles["reuse_fresh_n6"]={**reuse,"carrier_id":"occupation_n6_fresh"}
oracles["disconnect_n2"]={
    **oracles["primary_n2"], "carrier_id":"occupation_disconnect_n2",
    "delay_before_inverse_ms":120,
}
oracles["exception_n2"]={
    **oracles["primary_n2"], "carrier_id":"occupation_exception_n2",
    "inject_failure_after_projection":True,
}
oracles["partial_exception_n2"]={
    **oracles["primary_n2"], "carrier_id":"occupation_partial_n2",
    "inject_failure_after_modules":2,
}
print(json.dumps({"oracles":oracles},sort_keys=True,separators=(",",":")))
PY
)
private_config_bytes=$(( ${#private_config} + 1 ))
socket_name="@catvm-m246-$$"

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
for _ in $(seq 1 160); do
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
  echo "M246 service did not become ready" >&2
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
import hashlib,json,math,sys
from pathlib import Path

raw_path,ref_path,result_path=map(Path,sys.argv[1:4])
source_paths=list(map(Path,sys.argv[4:9]))
private_bytes=int(sys.argv[9])
raw=json.loads(raw_path.read_text())
ref=json.loads(ref_path.read_text())

raw_cases={case["run_kind"]:case for case in raw["cases"]}
ref_cases={case["run_kind"]:case for case in ref["cases"]}
if set(raw_cases)!=set(ref_cases):
    raise SystemExit("M246 case key mismatch")
for key in raw_cases:
    if raw_cases[key]["final_amplitude"]!=ref_cases[key]["final_amplitude"]:
        raise SystemExit(f"M246 independent boundary mismatch {key}")
    for field in ("rails","occupation_dimension","generation"):
        if raw_cases[key][field]!=ref_cases[key][field]:
            raise SystemExit(f"M246 independent case mismatch {key} {field}")

if not all(raw["controls"].values()) or not all(ref["controls"].values()):
    raise SystemExit("M246 control failure")
if not all(ref["labelled_verifier_parity"].values()):
    raise SystemExit("M246 labelled parity failure")

def dimensions(n):
    return math.comb(n+4,4)

def coefficient_terms(n):
    total=0
    def occs(rem,slots=5):
        if slots==1:
            yield (rem,); return
        for value in range(rem+1):
            for rest in occs(rem-value,slots-1):
                yield (value,)+rest
    for occupation in occs(n):
        product=1
        for count in occupation:
            product*=math.comb(count+4,4)
        total+=product
    return total

for n in (2,3,4,6):
    d=dimensions(n)
    primary=raw_cases[f"PRIMARY_N{n}"]
    terms=coefficient_terms(n)
    if primary["forward_kernel_coefficient_terms"]!=2*terms:
        raise SystemExit(f"M246 forward coefficient count mismatch n={n}")
    if primary["inverse_kernel_coefficient_terms"]!=3*terms:
        raise SystemExit(f"M246 inverse coefficient count mismatch n={n}")
    if primary["forward_orbit_dot_terms"]!=2*d*d:
        raise SystemExit(f"M246 forward dot count mismatch n={n}")
    if primary["inverse_orbit_dot_terms"]!=3*d*d:
        raise SystemExit(f"M246 inverse dot count mismatch n={n}")
    if primary["accepted_descriptor_reads"]!=38:
        raise SystemExit(f"M246 descriptor read mismatch n={n}")
    if primary["total_fixed_field_backing_cells"]!=3*d:
        raise SystemExit(f"M246 backing mismatch n={n}")
    baseline=ref["endpoint_baselines"][f"N{n}"]
    if baseline["final_amplitude"]!=primary["final_amplitude"]:
        raise SystemExit(f"M246 endpoint baseline mismatch n={n}")
    accepted_work=(primary["forward_kernel_coefficient_terms"]+
                   primary["forward_orbit_dot_terms"]+d)
    baseline_work=(baseline["first_layer_direct_phase_terms"]+
                   baseline["interior_kernel_coefficient_terms"]+
                   baseline["interior_orbit_dot_terms"]+
                   baseline["final_row_kernel_coefficient_terms"]+
                   baseline["final_row_orbit_dot_terms"])
    baseline["declared_coefficient_update_plus_dot_terms"]=baseline_work
    baseline["below_accepted_declared_coefficient_update_plus_dot_terms"]=(
        baseline_work<accepted_work
    )
    baseline["total_work_comparison_authorized"]=False
    if not baseline["below_accepted_declared_coefficient_update_plus_dot_terms"]:
        raise SystemExit(f"M246 endpoint declared-term count not smaller n={n}")

for key,certificate in ref["rank_certificates"].items():
    dimension=certificate["occupation_dimension"]
    if not (
        certificate["exact_depth_three_forward_descriptor_family_reachability_rank"]==dimension
        and certificate["observability_rank_from_all_public_final_occupation_selectors"]==dimension
        and certificate["exact_depth_three_hankel_rank"]==dimension
        and certificate["selected_program_count"]==dimension
        and len(certificate["selected_exact_depth_three_forward_programs"])==dimension
        and certificate["selected_program_depths"]==[3]
        and not certificate["inverse_gates_used_in_rank_certificate"]
        and certificate["descriptor_domain"]=={
            "lambda":[1,2,3,4], "quadratic":[0,1,2,3,4],
            "rung":[1,2,3,4], "beta":[1,2,3,4],
        }
    ):
        raise SystemExit(f"M246 rank certificate failure {key}")

raw["separate_reference"]={
    "result":ref["result"],
    "endpoint_baselines":ref["endpoint_baselines"],
    "labelled_verifier_parity":ref["labelled_verifier_parity"],
    "rank_certificates":ref["rank_certificates"],
    "controls":ref["controls"],
    "all_declared_exact_depth_three_forward_descriptor_family_ranks_equal_occupation_dimension":ref[
        "all_declared_exact_depth_three_forward_descriptor_family_ranks_equal_occupation_dimension"
    ],
    "verifier_only_labelled_assignment_counts":ref["verifier_only_labelled_assignment_counts"],
    "imports_production_service_client_or_m237":ref[
        "imports_production_service_client_or_m237"
    ],
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
    "production_and_independent_occupation_boundaries_match":True,
    "labelled25_and125_state_verifier_matches_occupation_quotient":True,
    "exact_depth_three_forward_descriptor_family_reachable_observable_hankel_ranks15_35_70_210_at_split_primes41_61":True,
    "endpoint_specialized_matrix_free_classical_boundaries_match":True,
    "selected_exact_depth_three_forward_rank_programs_persisted":True,
    "rank_certificate_uses_inverse_gates":False,
    "secret_dependent_intermediate_payload_metrics_not_delivered_or_sealed":True,
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
assert r["result"]=="PASS_CATVM_P5_PERMUTATION_SYMMETRIC_OCCUPATION_QUOTIENT_STRICT_SCOPE"
assert r["classification"]=="INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
assert r["verification_level"]=="SEPARATE_REFERENCE_PARITY"
assert r["restoration_classification"]=="EXACT_ALGEBRAIC_RESTORATION"
assert all(r["controls"].values())
assert r["resource_law"]["declared_rails"]==[2,3,4,6]
assert r["resource_law"]["occupation_dimensions"]==[15,35,70,210]
assert r["resource_law"]["accepted_total_fixed_field_backing_cells"]==[45,105,210,630]
assert r["resource_law"]["hidden_descriptor_residue_cells_per_program"]==16
assert r["resource_law"]["suite_service_configuration_descriptor_residue_cells"]==144
assert r["resource_law"]["suite_service_carrier_descriptor_residue_cells"]==128
assert r["resource_law"]["suite_service_configuration_plus_carrier_descriptor_residue_cells"]==272
assert r["resource_law"]["suite_service_total_fixed_field_backing_cells"]==1755
assert r["resource_law"]["suite_service_shared_public_occupation_and_index_plan_integer_cells"]==2640
assert r["resource_law"]["classical_optimality_claimed"] is False
assert r["resource_law"]["total_forward_work_comparison_authorized"] is False
assert "KRAWTCHOUK" in r["resource_law"]["unimplemented_stronger_exact_classical_ceiling"]
assert r["resource_law"]["secret_dependent_intermediate_payload_metrics_released"] is False
assert r["resource_law"]["secret_dependent_intermediate_payload_metrics_sealed"] is False
assert r["private_configuration_accounting"]["controller_receives_private_configuration"] is False
assert r["separate_reference"]["imports_production_service_client_or_m237"] is False
assert r["separate_reference"]["all_declared_exact_depth_three_forward_descriptor_family_ranks_equal_occupation_dimension"]
assert all(r["separate_reference"]["controls"].values())
assert all(r["separate_reference"]["labelled_verifier_parity"].values())
assert r["verification_statement"]["controller_imports_backend_or_field_code"] is False
assert not any(r["claim_limits"].values())
assert len(r["cases"])==6
for case in r["cases"]:
    assert case["canonical_after_restoration"]
    assert case["same_message_backing"] and case["same_output_scratch_backing"]
    assert case["same_coefficient_row_backing"] and case["same_descriptor_backings"]
    assert not case["baseline_reload_used"]
    assert case["labelled_assignment_materializations"]==0
    assert case["transfer_matrices_materialized"]==0
    assert case["retained_dynamic_inverse_history_entries"]==0
    assert "cells" not in case and "scratch" not in case and "row" not in case
for certificate in r["separate_reference"]["rank_certificates"].values():
    assert certificate["exact_depth_three_forward_descriptor_family_reachability_rank"]==certificate["occupation_dimension"]
    assert certificate["observability_rank_from_all_public_final_occupation_selectors"]==certificate["occupation_dimension"]
    assert certificate["exact_depth_three_hankel_rank"]==certificate["occupation_dimension"]
    assert certificate["selected_program_count"]==certificate["occupation_dimension"]
    assert len(certificate["selected_exact_depth_three_forward_programs"])==certificate["occupation_dimension"]
    assert certificate["selected_program_depths"]==[3]
    assert certificate["inverse_gates_used_in_rank_certificate"] is False
for baseline in r["separate_reference"]["endpoint_baselines"].values():
    assert baseline["below_accepted_declared_coefficient_update_plus_dot_terms"]
    assert baseline["total_work_comparison_authorized"] is False
    assert baseline["inverse_or_restoration_work"]==0
PY

"${run_env[@]}" python3 - "$client" <<'PY'
import ast,sys
tree=ast.parse(open(sys.argv[1]).read())
allowed={"__future__","json","math","socket","sys","time","typing"}
for node in ast.walk(tree):
    if isinstance(node,ast.Import):
        names={alias.name.split('.')[0] for alias in node.names}
    elif isinstance(node,ast.ImportFrom):
        names={str(node.module).split('.')[0]}
    else:
        continue
    if not names<=allowed:
        raise SystemExit(f"M246 controller imports nonpublic dependency: {names-allowed}")
PY

if rg -n 'catvm_p5_permutation_symmetric_occupation_service|zeta5_normalized_cubic_fourier_coherent_port|importlib|SourceFileLoader|runpy' "$client"; then
  echo "M246 controller imports or names backend code" >&2
  exit 2
fi
if "${run_env[@]}" python3 - "$service" <<'PY'
import ast,sys
tree=ast.parse(open(sys.argv[1]).read())
for node in ast.walk(tree):
    if isinstance(node,ast.Import) and any(alias.name=="itertools" for alias in node.names):
        raise SystemExit(1)
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="product":
        raise SystemExit(1)
PY
then
  true
else
  echo "M246 accepted service contains labelled Cartesian-product construction" >&2
  exit 2
fi

printf '%s\n' QUALIFIED_CATVM_P5_PERMUTATION_SYMMETRIC_OCCUPATION_STRICT_SCOPE
