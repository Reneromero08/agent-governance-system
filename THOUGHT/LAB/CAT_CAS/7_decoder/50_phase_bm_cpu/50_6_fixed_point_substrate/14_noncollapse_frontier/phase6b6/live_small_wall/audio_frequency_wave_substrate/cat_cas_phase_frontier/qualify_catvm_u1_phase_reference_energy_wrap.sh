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
    echo "RAM-backed M249 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M249 filesystem forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
service="$here/catvm_u1_phase_reference_energy_wrap_service.py"
client="$here/catvm_u1_phase_reference_energy_wrap_client.py"
reference="$here/catvm_u1_phase_reference_energy_wrap_separate_reference.py"
qualifier="$here/qualify_catvm_u1_phase_reference_energy_wrap.sh"
sealed_raw="$here/CATVM_U1_PHASE_REFERENCE_ENERGY_WRAP_RAW_RESULTS.json"
sealed_ref="$here/CATVM_U1_PHASE_REFERENCE_ENERGY_WRAP_SEPARATE_REFERENCE.json"
sealed_result="$here/CATVM_U1_PHASE_REFERENCE_ENERGY_WRAP_RESULTS.json"
raw="$build/CATVM_U1_PHASE_REFERENCE_ENERGY_WRAP_RAW_RESULTS.json"
ref="$build/CATVM_U1_PHASE_REFERENCE_ENERGY_WRAP_SEPARATE_REFERENCE.json"
result="$build/CATVM_U1_PHASE_REFERENCE_ENERGY_WRAP_RESULTS.json"
service_stdout="$build/CATVM_U1_PHASE_REFERENCE_ENERGY_WRAP_SERVICE.stdout"
service_stderr="$build/CATVM_U1_PHASE_REFERENCE_ENERGY_WRAP_SERVICE.stderr"

mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
run_env=(
  env
  TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp"
  XDG_CACHE_HOME="$build/xdg-cache"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache"
)

public_config=$(${run_env[@]} python3 - <<'PY'
import json

lengths=(2,4,8,16)
cases={}
for length in lengths:
    primary={"length":length,"gate":"H"}
    reuse={"length":length,"gate":"RATIONAL_3_4_5"}
    cases[f"primary_{length}"]={
        "carrier_id":f"m249-L{length}-primary","descriptor":primary,
    }
    cases[f"reuse_{length}"]={
        "carrier_id":f"m249-L{length}-primary","descriptor":reuse,
    }
    cases[f"fresh_{length}"]={
        "carrier_id":f"m249-L{length}-fresh","descriptor":reuse,
    }
for name,length in (
    ("disconnect",4),("partial",4),("postprojection",4),("descriptor_control",2),
):
    cases[name]={
        "carrier_id":f"m249-{name}","descriptor":{"length":length,"gate":"H"},
    }
print(json.dumps({"cases":cases},sort_keys=True,separators=(",",":")))
PY
)
public_config_bytes=$(( ${#public_config} + 1 ))
reference_config='{"suite":"M249_U1_PHASE_REFERENCE_STRICT_SCOPE"}'
reference_config_bytes=$(( ${#reference_config} + 1 ))
socket_name="@catvm-m249-$$"

service_pid=""
cleanup_service() {
  if [[ -n "$service_pid" ]] && kill -0 "$service_pid" 2>/dev/null; then
    kill "$service_pid"
    wait "$service_pid" || true
  fi
}
trap cleanup_service EXIT

printf '%s\n' '{"service":"M249_U1_PHASE_REFERENCE_MODE"}' | \
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
  echo "M249 service did not become ready" >&2
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
import hashlib,json,sys
from pathlib import Path

raw_path,ref_path,result_path=map(Path,sys.argv[1:4])
source_paths=list(map(Path,sys.argv[4:8]))
public_bytes=int(sys.argv[8])
reference_bytes=int(sys.argv[9])
raw=json.loads(raw_path.read_text())
ref=json.loads(ref_path.read_text())

def key(case): return (case["length"],case["gate"],case["run_kind"])
raw_cases={key(case):case for case in raw["cases"]}
ref_cases={key(case):case for case in ref["cases"]}
if set(raw_cases)!=set(ref_cases): raise SystemExit("M249 case key mismatch")
for case_key in raw_cases:
    production=raw_cases[case_key]
    oracle=ref_cases[case_key]
    for field in (
        "length","gate","generation","boundary","joint_carrier_commitment",
        "joint_field_cells","retained_final_boundary_field_cells_during_inverse",
        "same_joint_backing","canonical_after_restoration","baseline_reload_used","work",
    ):
        if production[field]!=oracle[field]:
            raise SystemExit(f"M249 independent case mismatch {case_key} {field}")

for case in raw["cases"]:
    analytic=ref["analytic_boundaries_and_minors"][f"L{case['length']}_{case['gate']}"]
    if case["boundary"]!=analytic["boundary"]:
        raise SystemExit("M249 accepted boundary does not match O(1) formula")
if raw_cases[(2,"H","PRIMARY")]["boundary"]!={
    "p0":{"rational":[3,4],"sqrt2":[0,1]},
    "p1":{"rational":[1,4],"sqrt2":[0,1]},
    "coherence":{"rational":[0,1],"sqrt2":[1,4]},
}:
    raise SystemExit("M249 frozen L2 H boundary mismatch")
if raw_cases[(16,"RATIONAL_3_4_5","FRESH")]["boundary"]!={
    "p0":{"rational":[2,5],"sqrt2":[0,1]},
    "p1":{"rational":[3,5],"sqrt2":[0,1]},
    "coherence":{"rational":[47,100],"sqrt2":[0,1]},
}:
    raise SystemExit("M249 frozen L16 rational boundary mismatch")
if not all(raw["controls"].values()) or not all(ref["controls"].values()):
    raise SystemExit("M249 control failure")
if not all(entry["minor_nonzero"] for entry in ref["analytic_boundaries_and_minors"].values()):
    raise SystemExit("M249 finite open correlation minor failure")
if not all(entry["factorizes_exactly"] and entry["nonzero"] for entry in ref["cyclic_sham"].values()):
    raise SystemExit("M249 cyclic return/wrap dichotomy failure")
if not all(entry["coherence_zero"] for entry in ref["dephased_reference"].values()):
    raise SystemExit("M249 dephased reference control failure")
if ref["bilateral_ideal"]["nonzero_shift_eigenvector_in_l2_z"]:
    raise SystemExit("M249 bilateral normalizability ceiling failure")

raw["separate_reference"]={
    "result":ref["result"],
    "cases":ref["cases"],
    "analytic_boundaries_and_minors":ref["analytic_boundaries_and_minors"],
    "cyclic_sham":ref["cyclic_sham"],
    "dephased_reference":ref["dephased_reference"],
    "bilateral_ideal":ref["bilateral_ideal"],
    "controls":ref["controls"],
    "resource_oracle":ref["resource_oracle"],
    "claim_limits":ref["claim_limits"],
}
raw["public_configuration_accounting"]={
    "controller_public_descriptor_stdin_bytes":public_bytes,
    "separate_reference_suite_stdin_bytes":reference_bytes,
    "private_or_answer_bearing_configuration_delivered":False,
    "service_receives_only_public_descriptors_in_per_transaction_requests":True,
}
names=("service","controller","separate_reference","qualifier")
raw["source_dependencies"]={
    f"{name}_sha256":hashlib.sha256(path.read_bytes()).hexdigest()
    for name,path in zip(names,source_paths)
}
raw["verification_statement"]={
    "production_and_independent_boundaries_match_all_twelve_transactions":True,
    "o1_analytic_and_streamed_open_boundaries_match":True,
    "finite_open_schmidt_minors_nonzero":True,
    "cyclic_exact_return_and_energy_wrap_both_verified":True,
    "bilateral_shift_eigenreference_nonnormalizability_derived":True,
    "dephased_reference_coherence_zero_reexecuted":True,
    "atomic_response_release_after_restoration":True,
    "controller_imports_backend_or_arithmetic_code":False,
    "no_joint_reference_or_reservoir_amplitudes_cross_the_socket":True,
    "no_space_or_work_advantage_over_o1_analytic_baseline":True,
}
result_path.write_text(json.dumps(raw,sort_keys=True,indent=2)+"\n")
PY

cmp -s "$raw" "$sealed_raw"
cmp -s "$ref" "$sealed_ref"
cmp -s "$result" "$sealed_result"

"${run_env[@]}" python3 - "$sealed_result" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["result"]=="PASS_CATVM_U1_PHASE_REFERENCE_ENERGY_WRAP_DICHOTOMY_STRICT_SCOPE"
assert r["classification"]=="INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
assert r["verification_level"]=="SEPARATE_REFERENCE_PARITY"
assert r["restoration_classification"]=="EXACT_ALGEBRAIC_RESTORATION"
assert len(r["cases"])==12 and all(r["controls"].values())
assert all(r["separate_reference"]["controls"].values())
assert not any(r["claim_limits"].values())
assert not any(r["separate_reference"]["claim_limits"].values())
assert r["resource_law"]["accepted_joint_field_cells_by_length"]==[4,8,16,32]
assert r["resource_law"]["accepted_forward_fixed_energy_pair_updates"]==78
assert r["resource_law"]["accepted_inverse_fixed_energy_pair_updates"]==78
assert r["resource_law"]["accepted_forward_field_multiplications"]==312
assert r["resource_law"]["accepted_inverse_field_multiplications"]==312
assert r["resource_law"]["accepted_boundary_field_multiplications"]==270
assert r["resource_law"]["accepted_boundary_field_accumulations"]==270
assert r["resource_law"]["control_only_persistent_carriers"]==3
assert r["resource_law"]["control_only_joint_field_cells"]==24
assert r["resource_law"]["strongest_classical_baseline"]=="O1_EXACT_ALL_L_BOUNDARY_FORMULAS_FROM_PUBLIC_L_A_B"
assert not r["resource_law"]["accepted_catvm_path_has_space_or_work_advantage"]
assert r["dichotomy"]["cyclic_reference_exact_return"]
assert not r["dichotomy"]["cyclic_reference_preserves_declared_total_number"]
assert not r["dichotomy"]["bilateral_exact_shift_eigenreference_is_normalizable"]
for case in r["cases"]:
    assert case["canonical_after_restoration"] and case["same_joint_backing"]
    assert not case["baseline_reload_used"]
    for forbidden in (
        "joint","reference","reservoir","amplitude_vector","dense_operator",
        "intermediate","eigenvector",
    ):
        assert forbidden not in case
PY

"${run_env[@]}" python3 - "$client" "$reference" <<'PY'
import ast,sys
allowed_client={"__future__","hashlib","json","socket","sys","time","typing"}
allowed_reference={"__future__","hashlib","json","sys","dataclasses","fractions","typing"}
for path,allowed in ((sys.argv[1],allowed_client),(sys.argv[2],allowed_reference)):
    tree=ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): names={alias.name.split('.')[0] for alias in node.names}
        elif isinstance(node,ast.ImportFrom): names={str(node.module).split('.')[0]}
        else: continue
        if not names<=allowed: raise SystemExit(f"M249 nonpublic import in {path}: {names-allowed}")
PY

if rg -n 'catvm_u1_phase_reference_energy_wrap_service|importlib|SourceFileLoader|runpy' "$client"; then
  echo "M249 controller imports or names backend code" >&2
  exit 2
fi
if rg -n '^import .*catvm_u1|^from .*catvm_u1|importlib|SourceFileLoader|runpy' "$reference"; then
  echo "M249 reference imports production code" >&2
  exit 2
fi
if ! rg -n 'prctl\(4, 0, 0, 0, 0\).*!= 0' "$service" >/dev/null; then
  echo "M249 service lacks checked nondumpability" >&2
  exit 2
fi
if ! rg -n 'sys.stdin.close\(\)' "$service" >/dev/null; then
  echo "M249 service does not close startup stdin" >&2
  exit 2
fi

printf '%s\n' QUALIFIED_CATVM_U1_PHASE_REFERENCE_ENERGY_WRAP_STRICT_SCOPE
