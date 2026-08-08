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
    echo "RAM-backed M248 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M248 filesystem forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
service="$here/catvm_p5_cubic_magic_catalyst_service.py"
client="$here/catvm_p5_cubic_magic_catalyst_client.py"
reference="$here/catvm_p5_cubic_magic_catalyst_separate_reference.py"
qualifier="$here/qualify_catvm_p5_cubic_magic_catalyst.sh"
field_dependency="$here/zeta5_normalized_cubic_fourier_coherent_port.py"
sealed_raw="$here/CATVM_P5_CUBIC_MAGIC_CATALYST_RAW_RESULTS.json"
sealed_ref="$here/CATVM_P5_CUBIC_MAGIC_CATALYST_SEPARATE_REFERENCE.json"
sealed_result="$here/CATVM_P5_CUBIC_MAGIC_CATALYST_RESULTS.json"
raw="$build/CATVM_P5_CUBIC_MAGIC_CATALYST_RAW_RESULTS.json"
ref="$build/CATVM_P5_CUBIC_MAGIC_CATALYST_SEPARATE_REFERENCE.json"
result="$build/CATVM_P5_CUBIC_MAGIC_CATALYST_RESULTS.json"
service_stdout="$build/CATVM_P5_CUBIC_MAGIC_CATALYST_SERVICE.stdout"
service_stderr="$build/CATVM_P5_CUBIC_MAGIC_CATALYST_SERVICE.stderr"

mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
run_env=(
  env
  TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp"
  XDG_CACHE_HOME="$build/xdg-cache"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache"
)

public_config=$(${run_env[@]} python3 - <<'PY'
import json

commitment1="31a152396ed266a9147459d3ba9a42b422fafca171360fe5d97ead372f5b3687"
commitment2="e23fd5a968bfdf6d28190dc29d1824bbb4ffef7424c76baf318e3b113eb43a70"
def descriptor(family,width,maps,output):
    return {
        "family":family,"width":width,"syndrome_maps":maps,"output":output,
        "catalyst_strength":1,"catalyst_commitment":commitment1,
    }
def case(carrier,value): return {"carrier_id":carrier,"descriptor":value}
single=descriptor(0,1,[[1]],[0])
primary=descriptor(1,2,[[1,1],[1,2]],[0,1])
reuse=descriptor(2,2,[[2,1],[1,1]],[1,0])
cases={
    "single":case("catalyst_single",single),
    "primary":case("catalyst_shared",primary),
    "reuse":case("catalyst_shared",reuse),
    "fresh":case("catalyst_fresh",reuse),
    "disconnect":case("catalyst_disconnect",primary),
    "partial":case("catalyst_partial",primary),
    "postprojection":case("catalyst_postprojection",primary),
    "descriptor_control":case("catalyst_descriptor_control",primary),
}
print(json.dumps({
    "alternate_catalyst_commitment":commitment2,"cases":cases,
},sort_keys=True,separators=(",",":")))
PY
)
public_config_bytes=$(( ${#public_config} + 1 ))
socket_name="@catvm-m248-$$"

service_pid=""
cleanup_service() {
  if [[ -n "$service_pid" ]] && kill -0 "$service_pid" 2>/dev/null; then
    kill "$service_pid"
    wait "$service_pid" || true
  fi
}
trap cleanup_service EXIT

printf '%s\n' '{"service":"M248_CUBIC_MAGIC_CATALYST_MODE"}' | \
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
  echo "M248 service did not become ready" >&2
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
if set(raw_cases)!=set(ref_cases): raise SystemExit("M248 case key mismatch")
for key in raw_cases:
    for field in (
        "family","width","syndrome_use_count","generation","final_amplitude",
        "catalyst_commitment","same_all_backings","canonical_after_restoration",
        "baseline_reload_used",
    ):
        if raw_cases[key][field]!=ref_cases[key][field]:
            raise SystemExit(f"M248 independent case mismatch {key} {field}")
    baseline=ref["direct_non_catalytic_baselines"][key]
    if baseline["final_amplitude"]!=raw_cases[key]["final_amplitude"]:
        raise SystemExit(f"M248 direct baseline mismatch {key}")

expected={
    "SINGLE_SYNDROME":([0,0,0,0],0),
    "TWO_SYNDROME_PRIMARY":([1,0,0,0],1),
    "RESTORED_UNRELATED_REUSE":([1,0,0,0],1),
    "FRESH_UNRELATED_REFERENCE":([1,0,0,0],1),
}
for key,(numerator,exponent) in expected.items():
    if raw_cases[key]["final_amplitude"]!={
        "numerator":numerator,"denominator_power5":exponent,
    }:
        raise SystemExit(f"M248 frozen public fixture mismatch {key}")
if not all(raw["controls"].values()) or not all(ref["controls"].values()):
    raise SystemExit("M248 control failure")
if not ref["tensor_identity"]["all_strengths1_2_3_4_and_syndromes0_TO_4_exact"]:
    raise SystemExit("M248 tensor identity failure")
if not ref["magic_resource"]["expected_l1_one_plus_two_sqrt5_over5"]:
    raise SystemExit("M248 exact Wigner-l1 reconstruction failure")
if not ref["magic_resource"]["direct_target_cubic_magic_state_has_the_same_single_state_l1"]:
    raise SystemExit("M248 direct target Wigner-l1 parity failure")
if not ref["dephased_sham"]["all_offdiagonal_syndrome_channel_factors_zero"]:
    raise SystemExit("M248 exact dephased channel failure")

raw["separate_reference"]={
    "result":ref["result"],
    "cases":ref["cases"],
    "controls":ref["controls"],
    "direct_non_catalytic_baselines":ref["direct_non_catalytic_baselines"],
    "tensor_identity":ref["tensor_identity"],
    "magic_resource":ref["magic_resource"],
    "dephased_sham":ref["dephased_sham"],
    "imports_m248_service_client_or_m237":ref["imports_m248_service_client_or_m237"],
    "independent_polynomial_quotient_arithmetic":ref["independent_polynomial_quotient_arithmetic"],
    "independent_tensor_contraction":ref["independent_tensor_contraction"],
    "independent_custody_state_machine":ref["independent_custody_state_machine"],
    "independent_direct_phase_boundary":ref["independent_direct_phase_boundary"],
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
    "all_twenty_strength_syndrome_tensor_identities_exact":True,
    "joint_state_refactors_to_the_same_actual_catalyst_after_every_use":True,
    "direct_noncatalytic_phase_boundary_matches_every_accepted_case":True,
    "exact_single_catalyst_wigner_l1_reconstructed":True,
    "dephased_and_stabilizer_shams_do_not_supply_the_accepted_identity":True,
    "atomic_response_release_after_restoration":True,
    "controller_imports_backend_or_field_code":False,
    "no_catalyst_joint_or_phase_signature_values_cross_the_socket":True,
    "joint_correction_magic_cost_is_not_hidden":True,
}
result_path.write_text(json.dumps(raw,sort_keys=True,indent=2)+"\n")
PY

cmp -s "$raw" "$sealed_raw"
cmp -s "$ref" "$sealed_ref"
cmp -s "$result" "$sealed_result"

"${run_env[@]}" python3 - "$sealed_result" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["result"]=="PASS_CATVM_P5_CUBIC_MAGIC_CATALYST_RESOURCE_BALANCE_STRICT_SCOPE"
assert r["classification"]=="INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
assert r["verification_level"]=="SEPARATE_REFERENCE_PARITY"
assert r["restoration_classification"]=="EXACT_ALGEBRAIC_RESTORATION"
assert len(r["cases"])==4 and all(r["controls"].values())
assert all(r["separate_reference"]["controls"].values())
assert not any(r["claim_limits"].values())
assert r["resource_law"]["accepted_fixed_field_backings_per_carrier"]==41
assert r["resource_law"]["actual_catalyst_field_cells_per_carrier"]==5
assert r["resource_law"]["joint_interaction_scratch_field_cells_per_carrier"]==25
assert r["resource_law"]["two_hidden_phase_signature_field_cells_per_carrier"]==10
assert r["resource_law"]["accepted_transaction_catalyst_uses"]==7
assert r["resource_law"]["accepted_path_joint_correction_root_multiplications"]==700
assert not r["resource_law"]["joint_correction_magic_monotone_or_optimal_synthesis_measured"]
assert not r["resource_law"]["accepted_software_path_has_work_or_magic_advantage_over_direct_phase"]
assert r["separate_reference"]["magic_resource"]["catalyst_negative_wigner_cells"]==5
assert r["separate_reference"]["magic_resource"]["direct_target_cubic_magic_state_negative_wigner_cells"]==5
assert r["separate_reference"]["magic_resource"]["direct_target_cubic_magic_state_has_the_same_single_state_l1"]
assert r["separate_reference"]["dephased_sham"]["all_offdiagonal_syndrome_channel_factors_zero"]
assert r["separate_reference"]["dephased_sham"]["offdiagonal_syndrome_coherence_survives"] is False
assert r["resource_balance"]["route_disposition"]=="RETIRE_THIS_IDENTITY_AS_AN_ADVANTAGE_ROUTE_AFTER_THE_BOUNDED_PROOF"
for case in r["cases"]:
    assert case["canonical_after_restoration"] and case["same_all_backings"]
    assert not case["baseline_reload_used"]
    for forbidden in (
        "catalyst","joint","phase_signatures","syndrome_maps","descriptor",
        "amplitude_vector","path_assignments",
    ):
        assert forbidden not in case
PY

"${run_env[@]}" python3 - "$client" "$reference" <<'PY'
import ast,sys
allowed_client={"__future__","hashlib","json","socket","sys","time","typing"}
allowed_reference={"__future__","hashlib","itertools","json","sys","dataclasses","fractions","typing"}
for path,allowed in ((sys.argv[1],allowed_client),(sys.argv[2],allowed_reference)):
    tree=ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): names={alias.name.split('.')[0] for alias in node.names}
        elif isinstance(node,ast.ImportFrom): names={str(node.module).split('.')[0]}
        else: continue
        if not names<=allowed: raise SystemExit(f"M248 nonpublic import in {path}: {names-allowed}")
PY

if rg -n 'catvm_p5_cubic_magic_catalyst_service|zeta5_normalized_cubic_fourier_coherent_port|importlib|SourceFileLoader|runpy' "$client"; then
  echo "M248 controller imports or names backend code" >&2
  exit 2
fi
if rg -n '^import .*catvm_p5|^from .*catvm_p5|^import .*zeta5|^from .*zeta5|importlib|SourceFileLoader|runpy' "$reference"; then
  echo "M248 reference imports production code" >&2
  exit 2
fi
if ! rg -n 'prctl\(4, 0, 0, 0, 0\).*!= 0' "$service" >/dev/null; then
  echo "M248 service lacks checked nondumpability" >&2
  exit 2
fi
if ! rg -n 'sys.stdin.close\(\)' "$service" >/dev/null; then
  echo "M248 service does not close startup stdin" >&2
  exit 2
fi

printf '%s\n' QUALIFIED_CATVM_P5_CUBIC_MAGIC_CATALYST_STRICT_SCOPE
