#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_catvm_cyclotomic_f5_tt.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
out=$1
mkdir -p "$out"
service="$here/catvm_cyclotomic_f5_tt_service.py"
controller="$here/catvm_cyclotomic_f5_tt_controller.py"
protocol="$here/catvm_cyclotomic_f5_tt_protocol.py"
phase="$here/cyclotomic_f5_cubic_tt_phase.py"
run_dir=$(mktemp -d "$out/run.XXXXXX")
inplace_socket="$run_dir/inplace.sock"
snapshot_socket="$run_dir/snapshot.sock"
inplace_pid=""
snapshot_pid=""
started_pid=""

cleanup() {
    if [[ -n "$inplace_pid" ]] && kill -0 "$inplace_pid" 2>/dev/null; then
        kill "$inplace_pid" 2>/dev/null || true
        wait "$inplace_pid" 2>/dev/null || true
    fi
    if [[ -n "$snapshot_pid" ]] && kill -0 "$snapshot_pid" 2>/dev/null; then
        kill "$snapshot_pid" 2>/dev/null || true
        wait "$snapshot_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

for tool in jq cmp rg sha256sum nice stat; do
    command -v "$tool" >/dev/null
done
[[ -x "$python" ]]
"$python" -m py_compile "$service" "$controller" "$protocol" "$phase"

env PYTHONPATH="$here" "$python" - >"$out/controller-module-isolation.json" <<'PY'
import ast
import json
import pathlib
import sys

import catvm_cyclotomic_f5_tt_controller as controller

controller_path = pathlib.Path(controller.__file__)
tree = ast.parse(controller_path.read_text(encoding="utf-8"))
imports = sorted(
    alias.name
    for node in ast.walk(tree)
    if isinstance(node, ast.Import)
    for alias in node.names
)
forbidden_modules = {
    "catvm_cyclotomic_f5_tt_service",
    "cyclotomic_f5_cubic_tt_phase",
}
loaded_forbidden = sorted(forbidden_modules.intersection(sys.modules))
source_forbidden = sorted(forbidden_modules.intersection(imports))
if loaded_forbidden or source_forbidden:
    raise RuntimeError("controller loaded or imported hidden phase backend")
print(json.dumps({
    "result": "PASS",
    "controller_imports": imports,
    "loaded_forbidden_modules": loaded_forbidden,
    "source_forbidden_imports": source_forbidden,
    "controller_service_module_loaded": False,
    "controller_phase_engine_loaded": False,
}, sort_keys=True, separators=(",", ":")))
PY

start_service() {
    local socket_path=$1
    local mode=$2
    local prefix=$3
    env PYTHONPATH="$here" nice -n 10 "$python" -X dev \
        "$service" "$socket_path" "$mode" \
        >"$out/${prefix}.stdout" 2>"$out/${prefix}.stderr" &
    started_pid=$!
    for _ in $(seq 1 200); do
        [[ -S "$socket_path" ]] && break
        sleep 0.05
    done
    [[ -S "$socket_path" ]]
    [[ "$(stat -c %a "$socket_path")" == "600" ]]
}

exchange() {
    local socket_path=$1
    local request=$2
    local destination=$3
    env PYTHONPATH="$here" "$python" - "$socket_path" \
        "$request" >"$destination" <<'PY'
import json
import sys
import catvm_cyclotomic_f5_tt_controller as controller
print(json.dumps(
    controller.exchange(sys.argv[1], json.loads(sys.argv[2])),
    sort_keys=True,
    separators=(",", ":"),
))
PY
}

start_service "$inplace_socket" IN_PLACE inplace-service
inplace_pid=$started_pid
env PYTHONPATH="$here" "$python" -X dev "$controller" \
    "$inplace_socket" >"$out/controller.json" \
    2>"$out/controller.stderr"
[[ ! -s "$out/controller.stderr" ]]

set +e
"$python" - "$inplace_pid" >"$out/process-inspection.stdout" \
    2>"$out/process-inspection.stderr" <<'PY'
import os
import sys
descriptor = os.open(f"/proc/{sys.argv[1]}/mem", os.O_RDONLY)
os.close(descriptor)
PY
inspection_rc=$?
set -e
[[ "$inspection_rc" -ne 0 ]]
[[ ! -s "$out/process-inspection.stdout" ]]

exchange "$inplace_socket" \
    '{"command":"PROJECT_INTERMEDIATE"}' \
    "$out/project-intermediate.json"
exchange "$inplace_socket" \
    '{"command":"NULL_CARRIER"}' \
    "$out/null-carrier.json"
exchange "$inplace_socket" \
    '{"command":"SNAPSHOT_PRIMARY"}' \
    "$out/snapshot-denied-inplace.json"
exchange "$inplace_socket" '{"command":"STOP"}' "$out/inplace-stop.json"
wait "$inplace_pid"
inplace_pid=""
[[ ! -s "$out/inplace-service.stdout" ]]
[[ ! -s "$out/inplace-service.stderr" ]]

start_service "$snapshot_socket" SNAPSHOT snapshot-service
snapshot_pid=$started_pid
exchange "$snapshot_socket" \
    '{"command":"SNAPSHOT_PRIMARY"}' "$out/snapshot.json"
exchange "$snapshot_socket" \
    '{"command":"RUN","program":"REUSE"}' \
    "$out/inplace-denied-snapshot.json"
exchange "$snapshot_socket" '{"command":"STOP"}' \
    "$out/snapshot-stop.json"
wait "$snapshot_pid"
snapshot_pid=""
[[ ! -s "$out/snapshot-service.stdout" ]]
[[ ! -s "$out/snapshot-service.stderr" ]]

env PYTHONPATH="$here" "$python" - >"$out/reference.json" <<'PY'
import json
import cyclotomic_f5_cubic_tt_phase as phase
primary = phase.transaction(
    phase.product_zero_state(4), 4, 4, 0
)
reuse = phase.transaction(
    phase.product_zero_state(4), 4, 3, 1
)
print(json.dumps({
    "primary_numerators": primary["boundary_numerators"],
    "primary_denominators": primary["boundary_denominators"],
    "reuse_numerators": reuse["boundary_numerators"],
    "reuse_denominators": reuse["boundary_denominators"],
}, sort_keys=True, separators=(",", ":")))
PY

jq -e --slurpfile reference "$out/reference.json" '
  .result == "PASS"
  and .claim_candidate
    == "CATVM_ENFORCED_CYCLOTOMIC_CUBIC_TT_HIDDEN_BOND_COMPOSITION_WITH_ACTUAL_RESTORATION_AND_REUSE"
  and .primary.status == "PASS"
  and .primary.boundary_numerators
    == $reference[0].primary_numerators
  and .primary.boundary_denominators
    == $reference[0].primary_denominators
  and .primary.restoration_generation == 1
  and .primary.actual_inverse_restoration
  and (.primary.snapshot_loaded | not)
  and (.primary.custody_receipt | length == 64)
  and .reuse.status == "PASS"
  and .reuse.boundary_numerators == $reference[0].reuse_numerators
  and .reuse.boundary_denominators
    == $reference[0].reuse_denominators
  and .reuse.restoration_generation == 2
  and .reuse.actual_inverse_restoration
  and (.reuse.snapshot_loaded | not)
  and (.reuse.custody_receipt | length == 64)
  and .status.transactions == 2
  and .status.restoration_generation == 2
  and .request_bytes_each == 1024
  and .response_bytes_each == 4096
  and (.controller_phase_engine_loaded | not)
  and (.controller_service_module_loaded | not)
  and (.terminal | not)
' "$out/controller.json" >/dev/null

jq -e '
  .status == "PASS"
  and .program == "PRIMARY"
  and .actual_inverse_restoration == false
  and .snapshot_loaded
  and .restoration_generation == 0
  and .snapshot_logical_payload_bytes == 160
  and .snapshot_creation_logical_copy_bytes == 160
  and .snapshot_execution_load_logical_copy_bytes == 160
  and .snapshot_restoration_reload_logical_copy_bytes == 160
  and .snapshot_total_logical_copy_bytes == 480
  and .snapshot_image_python_resident_bytes > 160
  and .snapshot_working_python_resident_bytes > 160
  and .snapshot_restored_python_resident_bytes > 160
' "$out/snapshot.json" >/dev/null
jq -e --slurpfile controller "$out/controller.json" '
  .boundary_numerators == $controller[0].primary.boundary_numerators
  and .boundary_denominators
    == $controller[0].primary.boundary_denominators
' "$out/snapshot.json" >/dev/null

jq -e '
  .status == "DENIED"
  and .error == "cyclotomic TT intermediate projection denied"
' "$out/project-intermediate.json" >/dev/null
jq -e '
  .status == "DENIED"
  and .error == "invalid cyclotomic TT carrier"
' "$out/null-carrier.json" >/dev/null
jq -e '
  .status == "DENIED"
  and .error == "snapshot command denied in in-place service"
' "$out/snapshot-denied-inplace.json" >/dev/null
jq -e '
  .status == "DENIED"
  and .error == "in-place command denied in snapshot service"
' "$out/inplace-denied-snapshot.json" >/dev/null

if {
    jq -c '{primary,reuse}' "$out/controller.json"
    jq -c . "$out/snapshot.json"
} | rg -n \
    'forward_bond_ranks|tensor|pivot|intermediate|maximum_bond_rank'
then
    echo "CATVM cyclotomic response smuggled internal state" >&2
    exit 1
fi
rg -q 'PR_SET_DUMPABLE' "$service"
rg -q 'SO_PEERCRED' "$service"
rg -q 'snapshot_image.*None' "$service"
jq -e '
  .result == "PASS"
  and (.controller_service_module_loaded | not)
  and (.controller_phase_engine_loaded | not)
  and (.loaded_forbidden_modules | length == 0)
  and (.source_forbidden_imports | length == 0)
  and .controller_imports
    == ["catvm_cyclotomic_f5_tt_protocol","json","socket","sys"]
' "$out/controller-module-isolation.json" >/dev/null

jq -n \
    --slurpfile controller "$out/controller.json" \
    --slurpfile snapshot "$out/snapshot.json" \
    --slurpfile isolation "$out/controller-module-isolation.json" '
  {
    result: "PASS",
    claim:
      "CATVM_ENFORCED_CYCLOTOMIC_CUBIC_TT_HIDDEN_BOND_COMPOSITION_WITH_ACTUAL_RESTORATION_AND_REUSE",
    claim_ceiling:
      "LINUX_USERSPACE_AF_UNIX_WIDTH4_PRIMARY_ROUNDS4_REUSE_ROUNDS3_EXACT_Q_ZETA5_SOFTWARE_ONLY",
    accepted: $controller[0],
    snapshot_sham: $snapshot[0],
    controller_module_isolation: $isolation[0],
    process_memory_inspection_denied: true,
    socket_mode: "0600",
    fixed_request_bytes: 1024,
    fixed_response_bytes: 4096,
    in_place_service_contains_snapshot_image: false,
    distinct_phase_resource_established: false,
    computational_advantage: false,
    small_wall_crossed: false,
    terminal: false
  }
' >"$out/qualification.json"

sha256sum \
    "$service" "$controller" "$protocol" "$phase" \
    "$out/controller.json" "$out/snapshot.json" \
    "$out/reference.json" "$out/controller-module-isolation.json" \
    "$out/qualification.json" \
    >"$out/SHA256SUMS"

rm -rf "$run_dir"
trap - EXIT
echo "CATVM cyclotomic F5 TT qualification: PASS"
