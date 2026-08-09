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
    echo "RAM-backed M256 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M256 filesystem forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
service="$here/catvm_schur_allpass_waveform_service.py"
client="$here/catvm_schur_allpass_waveform_client.py"
reference="$here/catvm_schur_allpass_waveform_separate_reference.py"
qualifier="$here/qualify_catvm_schur_allpass_waveform.sh"
sealed_raw="$here/CATVM_SCHUR_ALLPASS_WAVEFORM_RAW_RESULTS.json"
sealed_ref="$here/CATVM_SCHUR_ALLPASS_WAVEFORM_SEPARATE_REFERENCE.json"
sealed_result="$here/CATVM_SCHUR_ALLPASS_WAVEFORM_RESULTS.json"
public="$build/M256_PUBLIC.json"
raw="$build/CATVM_SCHUR_ALLPASS_WAVEFORM_RAW_RESULTS.json"
ref="$build/CATVM_SCHUR_ALLPASS_WAVEFORM_SEPARATE_REFERENCE.json"
result="$build/CATVM_SCHUR_ALLPASS_WAVEFORM_RESULTS.json"
stdout_log="$build/M256_SERVICE.stdout"
stderr_log="$build/M256_SERVICE.stderr"

mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
run_env=(
  env
  TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp"
  XDG_CACHE_HOME="$build/xdg-cache"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache"
)

"${run_env[@]}" python3 - "$public" <<'PY'
import json,sys
from pathlib import Path

output_type="QZETA8_ALLPASS_WINDING_AND_POINT_EVALUATION_V1"
def descriptor(word):
    return {"sections":word,"evaluation":"ZETA8","output_type":output_type}
a=descriptor([[1,2],[-1,3],[1,4]])
b=descriptor([[-1,4],[1,3],[1,2]])
sham=descriptor([[0,1],[0,1],[0,1]])
cases={
    "primary":{"carrier_id":"m256-shared","descriptor":a},
    "reuse":{"carrier_id":"m256-shared","descriptor":b},
    "fresh":{"carrier_id":"m256-fresh","descriptor":b},
    "sham":{"carrier_id":"m256-sham","descriptor":sham},
    "disconnect":{"carrier_id":"m256-disconnect","descriptor":a},
    "partial":{"carrier_id":"m256-partial","descriptor":a},
    "postprojection":{"carrier_id":"m256-postprojection","descriptor":a},
    "descriptor_control":{"carrier_id":"m256-control","descriptor":a},
}
Path(sys.argv[1]).write_text(json.dumps({"cases":cases},sort_keys=True,separators=(",",":"))+"\n")
PY

socket_name="@catvm-m256-qualifier-$$"
service_pid=""
cleanup_service() {
  if [[ -n "$service_pid" ]] && kill -0 "$service_pid" 2>/dev/null; then
    kill "$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
}
trap cleanup_service EXIT
printf '%s\n' '{"service":"M256_SCHUR_ALLPASS_WAVEFORM_MODE"}' |
  "${run_env[@]}" nice -n 10 ionice -c 3 python3 "$service" "$socket_name" >"$stdout_log" 2>"$stderr_log" &
service_pid=$!

ready=0
for _ in $(seq 1 200); do
  if ! kill -0 "$service_pid" 2>/dev/null; then
    break
  fi
  if "${run_env[@]}" python3 - "$socket_name" 2>/dev/null <<'PY'
import socket,sys
name=sys.argv[1]
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
try:
    s.connect("\0"+name[1:])
finally:
    s.close()
PY
  then
    ready=1
    break
  fi
  sleep 0.01
done
if [[ $ready -ne 1 ]]; then
  echo "M256 backend did not become ready" >&2
  exit 1
fi

"${run_env[@]}" nice -n 10 ionice -c 3 python3 "$client" "$socket_name" <"$public" >"$raw"
wait "$service_pid"
service_pid=""
if [[ -s "$stdout_log" || -s "$stderr_log" ]]; then
  echo "M256 backend emitted stdout/stderr" >&2
  exit 1
fi

"${run_env[@]}" nice -n 10 ionice -c 3 python3 "$reference" "$public" "$raw" >"$ref"

"${run_env[@]}" python3 - \
  "$raw" "$ref" "$result" "$service" "$client" "$reference" "$qualifier" <<'PY'
import ast,hashlib,json,sys
from pathlib import Path

raw_path,ref_path,result_path=map(Path,sys.argv[1:4])
sources=list(map(Path,sys.argv[4:8]))
raw=json.loads(raw_path.read_text())
ref=json.loads(ref_path.read_text())

if raw.get("result")!="PASS_CATVM_EXACT_SCHUR_ALLPASS_WAVEFORM_STRICT_SCOPE":
    raise SystemExit("M256 result mismatch")
if raw.get("classification")!="INDEPENDENTLY_VERIFIED_STRICT_SCOPE" or ref.get("classification")!="INDEPENDENTLY_VERIFIED_STRICT_SCOPE":
    raise SystemExit("M256 classification mismatch")
if raw.get("verification_level")!="SEPARATE_REFERENCE_PARITY" or ref.get("verification_level")!="SEPARATE_REFERENCE_PARITY":
    raise SystemExit("M256 verification mismatch")
if raw.get("restoration_classification")!="EXACT_ALGEBRAIC_RESTORATION" or ref.get("restoration_classification")!="EXACT_ALGEBRAIC_RESTORATION":
    raise SystemExit("M256 restoration classification mismatch")
if not all(raw.get("controls",{}).values()) or not all(ref.get("controls",{}).values()):
    raise SystemExit("M256 control failure")

raw_cases={case["run_kind"]:case for case in raw["cases"]}
ref_cases=ref["cases"]
if set(raw_cases)!={"PRIMARY","REUSE","FRESH","SHAM"} or set(ref_cases)!=set(raw_cases):
    raise SystemExit("M256 case set mismatch")
for key in raw_cases:
    production=raw_cases[key]; oracle=ref_cases[key]
    for field in ("generation","winding","evaluation","canonical_after_restoration","baseline_reload_used"):
        if production[field]!=oracle[field]:
            raise SystemExit(f"M256 independent mismatch {key} {field}")
    if not production["same_waveform_scratch_and_receipt_backings"]:
        raise SystemExit(f"M256 production backing mismatch {key}")
    if not oracle["same_reference_backings"]:
        raise SystemExit(f"M256 reference backing mismatch {key}")
    if production["resource_shape"]["actual_final_numerator_degree"]!=oracle["actual_final_numerator_degree"] or production["resource_shape"]["actual_final_denominator_degree"]!=oracle["actual_final_denominator_degree"]:
        raise SystemExit(f"M256 independent degree mismatch {key}")
    if not oracle["exact_formal_laurent_allpass_identity"]:
        raise SystemExit(f"M256 formal allpass identity mismatch {key}")
if raw_cases["REUSE"]["generation"]!=2 or raw_cases["FRESH"]["generation"]!=1:
    raise SystemExit("M256 reuse generation mismatch")
if raw_cases["REUSE"]["evaluation"]!=raw_cases["FRESH"]["evaluation"]:
    raise SystemExit("M256 restored/fresh boundary mismatch")
if raw_cases["PRIMARY"]["winding"]!=3 or raw_cases["SHAM"]["winding"]!=3:
    raise SystemExit("M256 winding mismatch")
if raw_cases["PRIMARY"]["evaluation"]==raw_cases["SHAM"]["evaluation"]:
    raise SystemExit("M256 feedback sham not causal")

for key,case in raw_cases.items():
    shape=case["resource_shape"]
    expected_denominator_degree=0 if key=="SHAM" else 3
    if shape!={
        "resident_waveform_field_cells":8,
        "scratch_waveform_field_cells":8,
        "receipt_rational_cells":3,
        "retained_final_boundary_field_cells_during_inverse":1,
        "retained_final_winding_integer_cells_during_inverse":1,
        "retained_final_degree_integer_cells_during_inverse":2,
        "dynamic_inverse_history_field_cells":0,
        "allocated_polynomial_capacity_each":4,
        "actual_final_numerator_degree":3,
        "actual_final_denominator_degree":expected_denominator_degree,
    }:
        raise SystemExit("M256 resource shape mismatch")
    work=case["work"]
    expected={
        "forward_sections":3,"inverse_sections":3,
        "forward_field_scalar_multiplications":24,"forward_field_additions":24,
        "inverse_field_scalar_multiplications":42,"inverse_field_subtractions":21,
        "inverse_public_rational_divisions":3,
        "inverse_public_rational_multiplications":3,
        "inverse_public_rational_subtractions":3,
        "forward_scratch_writes":24,"inverse_scratch_writes":24,
        "carrier_coefficient_writes":48,"scratch_clears":48,
        "inverse_divisibility_checks":3,
        "evaluation_field_multiplications":8,"evaluation_field_additions":8,
        "evaluation_field_inversions":1,
    }
    if work!=expected:
        raise SystemExit("M256 work law mismatch")

law=raw["resource_law"]
if law["strongest_actual_boundary_classical_baseline"]!="ONE_QZETA8_SCALAR_PLUS_ONE_WINDING_INTEGER_SCHUR_RECURRENCE_IN_O1_LIVE_ALGEBRAIC_STATE":
    raise SystemExit("M256 scalar baseline mismatch")
if law["strongest_full_formal_waveform_classical_baseline"]!="IDENTICAL_TWO_POLYNOMIAL_EXACT_SCHUR_RECURRENCE_WITHOUT_CATVM_INVERSE_RESTORATION":
    raise SystemExit("M256 full-waveform baseline mismatch")
if law["catvm_path_has_space_work_or_query_advantage"]:
    raise SystemExit("M256 advantage overclaim")
if not law["field_cells_are_not_fixed_width_payload_claims"] or law["whole_transaction_live_payload_peak_complete"]:
    raise SystemExit("M256 resource caveat mismatch")
if ref["classical_baselines"]!={
    "actual_boundary":"ONE_FIELD_SCALAR_PLUS_ONE_INTEGER_WINDING",
    "actual_boundary_scalar_field_cells":1,
    "catvm_restoration_required_for_classical_baselines":False,
    "fixed_fixture":"O1_PUBLIC_CERTIFICATE",
    "full_waveform":"TWO_POLYNOMIAL_SCHUR_RECURRENCE",
    "full_waveform_field_cells_at_depth3":8,
}:
    raise SystemExit("M256 independent comparator mismatch")
for key,case in ref_cases.items():
    payload=case["final_waveform_service_basis_payload"]
    if payload["integer_coordinate_count"]!=32 or payload["total_fraction_payload_bits"]<=0:
        raise SystemExit(f"M256 private payload mismatch {key}")
if any(value is not False for value in raw["claim_limits"].values()):
    raise SystemExit("M256 claim limit promoted")
if raw["waveform_law"]["route_disposition"]!="RETIRE_AFTER_THIS_BOUNDED_DIAGNOSTIC_IF_SCALAR_AND_FULL_FUNCTION_BISIMULATIONS_MATCH":
    raise SystemExit("M256 route disposition mismatch")

service_tree=ast.parse(sources[0].read_text())
client_tree=ast.parse(sources[1].read_text())
reference_tree=ast.parse(sources[2].read_text())
client_imports={node.module for node in ast.walk(client_tree) if isinstance(node,ast.ImportFrom) and node.module}
reference_imports={node.module for node in ast.walk(reference_tree) if isinstance(node,ast.ImportFrom) and node.module}
if any("schur" in module or "catvm" in module for module in client_imports|reference_imports):
    raise SystemExit("M256 controller/reference imported production")
service_text=sources[0].read_text()
client_text=sources[1].read_text()
for forbidden in ("itertools.product","cartesian_product","assignment_table","truth_table","path_history"):
    if forbidden in service_text:
        raise SystemExit(f"M256 backend forbidden expansion marker {forbidden}")
if "catvm_schur_allpass_waveform_service" in client_text or "fractions" in client_text:
    raise SystemExit("M256 controller loaded backend arithmetic")
if not any(isinstance(node,ast.Try) and node.finalbody for node in ast.walk(service_tree)):
    raise SystemExit("M256 atomic restoration finally missing")

source_dependencies={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
result={
    "milestone":256,
    "claim":raw["claim"],
    "claim_ceiling":raw["claim_ceiling"],
    "classification":raw["classification"],
    "verification_level":raw["verification_level"],
    "restoration_class":raw["restoration_classification"],
    "cases":raw["cases"],
    "controls":{"production":raw["controls"],"independent":ref["controls"]},
    "waveform_law":raw["waveform_law"],
    "resource_law":raw["resource_law"],
    "independent_private_resident_waveform_payload":{
        key:value["final_waveform_service_basis_payload"] for key,value in ref_cases.items()
    },
    "classical_baselines":ref["classical_baselines"],
    "protocol_accounting":raw["protocol_accounting"],
    "claim_limits":raw["claim_limits"],
    "source_dependencies":source_dependencies,
}
result_path.write_text(json.dumps(result,sort_keys=True,separators=(",",":"))+"\n")
PY

if [[ ${M256_GENERATE_ONLY:-0} == 1 ]]; then
  echo "M256 SCHUR ALLPASS WAVEFORM EVIDENCE GENERATED"
  exit 0
fi

cmp -s "$raw" "$sealed_raw" || {
  echo "M256 raw seal mismatch" >&2
  exit 1
}
cmp -s "$ref" "$sealed_ref" || {
  echo "M256 reference seal mismatch" >&2
  exit 1
}
cmp -s "$result" "$sealed_result" || {
  echo "M256 result seal mismatch" >&2
  exit 1
}

echo "M256 CATVM SCHUR ALLPASS WAVEFORM QUALIFIED"
