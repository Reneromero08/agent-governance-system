#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_catvm_four_rotor_incremental.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
out=$1
mkdir -p "$out"
run_dir=$(mktemp -d "$out/run.XXXXXX")
service="$here/catvm_four_rotor_incremental_service.py"
controller="$here/catvm_four_rotor_incremental_controller.py"
protocol="$here/catvm_four_rotor_incremental_protocol.py"
backend="$here/catvm_four_rotor_incremental_backend.py"
direct="$here/catvm_four_rotor_incremental_direct.py"
phase="$here/four_rotor_incremental_schmidt_closure.py"
qualifier="$here/qualify_catvm_four_rotor_incremental.sh"
active_pids=()

cleanup() {
    for pid in "${active_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

for tool in jq rg sha256sum nice stat; do
    command -v "$tool" >/dev/null
done
[[ -x "$python" ]]
"$python" -m py_compile \
    "$service" "$controller" "$protocol" "$backend" "$direct" "$phase"

env PYTHONPATH="$here" "$python" - >"$out/controller-isolation.json" <<'PY'
import ast
import json
import pathlib
import sys

import catvm_four_rotor_incremental_controller as controller

tree = ast.parse(
    pathlib.Path(controller.__file__).read_text(encoding="utf-8")
)
imports = sorted(
    alias.name
    for node in ast.walk(tree)
    if isinstance(node, ast.Import)
    for alias in node.names
)
forbidden = {
    "catvm_four_rotor_incremental_backend",
    "catvm_four_rotor_incremental_service",
    "four_rotor_incremental_schmidt_closure",
    "four_rotor_kicked_phase_tt",
    "numpy",
    "scipy",
}
loaded = sorted(forbidden.intersection(sys.modules))
source = sorted(forbidden.intersection(imports))
if loaded or source:
    raise RuntimeError("controller imported hidden phase backend")
print(json.dumps({
    "result": "PASS",
    "controller_imports": imports,
    "loaded_forbidden_modules": loaded,
    "source_forbidden_imports": source,
}, sort_keys=True, separators=(",", ":")))
PY

env PYTHONPATH="$here" nice -n 10 "$python" -X dev "$direct" \
    >"$out/direct-process.json" 2>"$out/direct-process.stderr"
[[ ! -s "$out/direct-process.stderr" ]]

start_service() {
    local mode=$1
    local lower
    lower=$(printf '%s' "$mode" | tr '[:upper:]' '[:lower:]')
    local socket_path="$run_dir/${lower}.sock"
    env PYTHONPATH="$here" nice -n 10 "$python" -X dev \
        "$service" "$socket_path" "$mode" \
        >"$out/${lower}-service.stdout" \
        2>"$out/${lower}-service.stderr" &
    local pid=$!
    active_pids+=("$pid")
    for _ in $(seq 1 400); do
        [[ -S "$socket_path" ]] && break
        sleep 0.05
    done
    [[ -S "$socket_path" ]]
    [[ "$(stat -c %a "$socket_path")" == "600" ]]

    set +e
    "$python" - "$pid" >"$out/${lower}-mem.stdout" \
        2>"$out/${lower}-mem.stderr" <<'PY'
import os
import sys
descriptor = os.open(f"/proc/{sys.argv[1]}/mem", os.O_RDONLY)
os.close(descriptor)
PY
    local inspection_rc=$?
    set -e
    [[ "$inspection_rc" -ne 0 ]]
    [[ ! -s "$out/${lower}-mem.stdout" ]]

    env PYTHONPATH="$here" "$python" -X dev "$controller" \
        "$socket_path" >"$out/${lower}.json" \
        2>"$out/${lower}-controller.stderr"
    wait "$pid"
    active_pids=("${active_pids[@]/$pid}")
    [[ ! -s "$out/${lower}-service.stdout" ]]
    [[ ! -s "$out/${lower}-service.stderr" ]]
    [[ ! -s "$out/${lower}-controller.stderr" ]]
}

start_service ISOLATED
start_service SNAPSHOT
start_service IN_PLACE

env PYTHONPATH="$here" "$python" - \
    "$out/direct-process.json" \
    "$out/isolated.json" \
    "$out/snapshot.json" \
    "$out/in_place.json" \
    >"$out/comparison.json" <<'PY'
import json
import sys

import catvm_four_rotor_incremental_backend as backend

direct, isolated, snapshot, in_place = [
    json.load(open(path, encoding="utf-8")) for path in sys.argv[1:]
]
arms = {
    "direct_process": direct,
    "isolated_sham": isolated,
    "snapshot_sham": snapshot,
    "in_place_catvm": in_place,
}
distances = {}
for program_key in ("primary", "reuse"):
    reference_boundary = direct[program_key]["final_boundary"]
    distances[program_key] = {
        name: backend.boundary_distance(
            reference_boundary,
            arm[program_key]["final_boundary"],
        )
        for name, arm in arms.items()
    }
def resource_row(arm):
    row = {
        key: arm["primary"]["resources"][key]
        for key in (
            "engine_execution_ns",
            "service_transaction_ns",
            "process_peak_rss_bytes",
            "engine_accounted_peak_array_bytes",
            "wrapper_accounted_peak_array_bytes",
            "verification_baseline_bytes",
            "snapshot_copy_bytes_cumulative",
            "service_init_carrier_creation_ns",
            "snapshot_creation_ns",
            "transaction_carrier_creation_ns",
            "snapshot_execution_load_ns",
            "snapshot_restoration_reload_ns",
            "native_coupling_applications",
            "native_incremental_updates",
            "public_operator_materialization_bytes_total",
            "logical_request_bytes",
            "logical_response_bytes",
        )
    }
    row["primary_controller_roundtrip_ns"] = arm.get(
        "primary_roundtrip_ns"
    )
    row["primary_total_path_ns"] = arm.get(
        "primary_roundtrip_ns",
        arm["primary"]["resources"]["service_transaction_ns"],
    )
    row["actual_controller_backend_traffic_bytes"] = arm.get(
        "actual_controller_backend_traffic_bytes", 0
    )
    return row


primary_resources = {
    name: resource_row(arm)
    for name, arm in arms.items()
}
print(json.dumps({
    "result": "PASS",
    "boundary_distances_from_direct": distances,
    "primary_resources": primary_resources,
    "deterministic_obstruction": (
        "ACTUAL_INVERSE_REQUIRES_2X_NATIVE_COUPLING_APPLICATIONS_"
        "IN_THIS_BOUNDED_IMPLEMENTATION"
    ),
    "separately_measured_components": [
        "ENGINE_EXECUTION",
        "SERVICE_TRANSACTION",
        "CONTROLLER_ROUNDTRIP",
        "CARRIER_CREATION",
        "SNAPSHOT_CREATION_LOAD_RELOAD",
        "CANONICAL_RESTORATION_AND_VERIFICATION_INSIDE_IN_PLACE_ENGINE",
    ],
    "direct_primary_coupling_applications": (
        primary_resources["direct_process"][
            "native_coupling_applications"
        ]
    ),
    "in_place_primary_coupling_applications": (
        primary_resources["in_place_catvm"][
            "native_coupling_applications"
        ]
    ),
    "matched_classical_incremental_tt_identical": True,
    "distinct_phase_resource_established": False,
    "computational_advantage": False,
    "small_wall_crossed": False,
}, sort_keys=True, separators=(",", ":")))
PY

env PYTHONPATH="$here" "$python" - >"$out/inverse-controls.json" <<'PY'
import json
import four_rotor_incremental_schmidt_closure as phase
values = phase.controls()
if min(values.values()) <= 1e-4:
    raise RuntimeError("four-rotor inverse controls failed")
print(json.dumps({
    "result": "PASS",
    "controls": values,
}, sort_keys=True, separators=(",", ":")))
PY

for arm in isolated snapshot in_place; do
    jq -e --arg arm "$(printf '%s' "$arm" | tr '[:lower:]' '[:upper:]')" '
      .result == "PASS"
      and .arm == $arm
      and .primary.status == "PASS"
      and .reuse.status == "PASS"
      and .request_count == 7
      and .response_count == 7
      and .request_bytes_each == 1024
      and .response_bytes_each == 8192
      and .actual_controller_backend_traffic_bytes == 64512
      and (.primary_roundtrip_ns > 0)
      and (.reuse_roundtrip_ns > 0)
      and (.controller_total_ns >= (
        .primary_roundtrip_ns + .reuse_roundtrip_ns
      ))
      and .projection_control.status == "DENIED"
      and .projection_control.error
        == "four-rotor intermediate projection denied"
      and .null_carrier_control.status == "DENIED"
      and .status.transactions == 2
      and (.controller_phase_engine_loaded | not)
      and (.controller_service_module_loaded | not)
      and (.terminal | not)
    ' "$out/${arm}.json" >/dev/null
done

jq -e '
  .result == "PASS"
  and .protocol_equivalent_request_count == 7
  and .protocol_equivalent_response_count == 7
  and .protocol_equivalent_logical_bytes == 64512
  and .simulated_matched_protocol_traffic
  and .actual_controller_backend_traffic_bytes == 0
  and (.primary.resources.service_transaction_ns > 0)
  and (.reuse.resources.service_transaction_ns > 0)
' "$out/direct-process.json" >/dev/null

jq -e '
  .primary.actual_inverse_restoration == false
  and .reuse.actual_inverse_restoration == false
  and (.primary.snapshot_loaded | not)
  and (.reuse.snapshot_loaded | not)
  and .primary.carrier_creation_count == 1
  and .reuse.carrier_creation_count == 2
' "$out/isolated.json" >/dev/null
jq -e '
  .primary.actual_inverse_restoration == false
  and .reuse.actual_inverse_restoration == false
  and .primary.snapshot_loaded
  and .reuse.snapshot_loaded
  and .primary.restoration_generation == 0
  and .reuse.restoration_generation == 0
  and .primary.resources.snapshot_copy_bytes_cumulative == 5568
  and .reuse.resources.snapshot_copy_bytes_cumulative == 9280
  and .primary.resources.snapshot_execution_load_ns > 0
  and .primary.resources.snapshot_restoration_reload_ns > 0
  and .primary.resources.snapshot_creation_ns > 0
' "$out/snapshot.json" >/dev/null
jq -e '
  .primary.actual_inverse_restoration
  and .reuse.actual_inverse_restoration
  and .primary.canonical_restoration
  and .reuse.canonical_restoration
  and (.primary.snapshot_loaded | not)
  and (.reuse.snapshot_loaded | not)
  and .primary.restoration_generation == 1
  and .reuse.restoration_generation == 2
  and .primary.carrier_creation_count == 1
  and .reuse.carrier_creation_count == 1
  and .primary.resources.retained_inverse_history_bytes == 0
  and .primary.resources.verification_baseline_reload_count == 0
  and (.primary.resources.verification_baseline_used_for_restoration | not)
' "$out/in_place.json" >/dev/null
jq -e '
  .boundary_distances_from_direct.primary.isolated_sham <= 1e-12
  and .boundary_distances_from_direct.primary.snapshot_sham <= 1e-12
  and .boundary_distances_from_direct.primary.in_place_catvm <= 1e-12
  and .boundary_distances_from_direct.reuse.isolated_sham <= 1e-12
  and .boundary_distances_from_direct.reuse.snapshot_sham <= 1e-12
  and .boundary_distances_from_direct.reuse.in_place_catvm <= 1e-5
  and .direct_primary_coupling_applications == 9
  and .in_place_primary_coupling_applications == 18
  and .matched_classical_incremental_tt_identical
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
' "$out/comparison.json" >/dev/null

jq -e '
  .result == "PASS"
  and (.loaded_forbidden_modules | length == 0)
  and (.source_forbidden_imports | length == 0)
  and .controller_imports
    == ["catvm_four_rotor_incremental_protocol","json","socket","sys","time"]
' "$out/controller-isolation.json" >/dev/null

if {
    jq -c '{primary,reuse}' "$out/direct-process.json"
    jq -c '{primary,reuse}' "$out/isolated.json"
    jq -c '{primary,reuse}' "$out/snapshot.json"
    jq -c '{primary,reuse}' "$out/in_place.json"
} | rg -n \
    'central_rank_history|maximum_incremental_rank|maximum_retained_rank|maximum_context|bond_ranks|tensor_values|singular_values|left_basis|right_basis|candidate_set|truth_table'
then
    echo "CATVM four-rotor response smuggled hidden state" >&2
    exit 1
fi
rg -q 'PR_SET_DUMPABLE' "$service"
rg -q 'PR_SET_PTRACER' "$service"
rg -q 'PR_SET_NO_NEW_PRIVS' "$service"
rg -q 'SO_PEERCRED' "$service"

jq -n \
    --slurpfile direct "$out/direct-process.json" \
    --slurpfile isolated "$out/isolated.json" \
    --slurpfile snapshot "$out/snapshot.json" \
    --slurpfile inplace "$out/in_place.json" \
    --slurpfile comparison "$out/comparison.json" \
    --slurpfile controls "$out/inverse-controls.json" \
    --slurpfile isolation "$out/controller-isolation.json" '
  {
    result: "PASS",
    claim_candidate:
      "BOUNDED_CATVM_ENFORCED_PROBE_FREE_INCREMENTAL_BESSEL_SCHMIDT_HIDDEN_TT_COMPOSITION_WITH_ACTUAL_INVERSE_CANONICAL_RESTORATION_AND_REUSE",
    claim_ceiling:
      "LINUX_SAME_UID_PYTHON_NUMPY_SCIPY_AF_UNIX_SOCK_SEQPACKET_PR_SET_DUMPABLE_FIXED_PRIMARY_REUSE_FOUR_OPEN_CHAIN_ROTORS_MODE_RADIUS14_DEPTH3_2_INCREMENTAL_L2_1E_MINUS6_RESTORATION_L2_5E_MINUS5_NO_SECCOMP_NO_LOCKED_MEMORY_SOFTWARE_FLOAT64",
    direct_process: $direct[0],
    isolated_sham: $isolated[0],
    snapshot_sham: $snapshot[0],
    in_place_catvm: $inplace[0],
    comparison: $comparison[0],
    inverse_controls: $controls[0],
    controller_isolation: $isolation[0],
    fixed_request_bytes: 1024,
    fixed_response_bytes: 8192,
    identical_isolated_request_response_count: true,
    direct_protocol_shape_is_simulated_not_actual_traffic: true,
    intermediate_projection_denied: true,
    null_carrier_denied: true,
    proc_pid_mem_inspection_denied: true,
    no_smuggle_scan: "PASS",
    accepted_path_snapshot_loaded: false,
    accepted_path_retained_inverse_history_bytes: 0,
    verification_baseline_counted_never_reloaded: true,
    matched_classical_incremental_tt_identical: true,
    distinct_phase_resource_established: false,
    computational_advantage: false,
    small_wall_crossed: false,
    unbounded_computation_established: false,
    physical_waveform_execution: false,
    terminal: false
  }
' >"$out/qualification.json"

sha256sum \
    "$service" "$controller" "$protocol" "$backend" "$direct" \
    "$phase" "$qualifier" \
    "$out/direct-process.json" "$out/isolated.json" \
    "$out/snapshot.json" "$out/in_place.json" \
    "$out/comparison.json" "$out/inverse-controls.json" \
    "$out/controller-isolation.json" "$out/qualification.json" \
    >"$out/SHA256SUMS"

rm -rf "$run_dir"
trap - EXIT
echo "CATVM four-rotor incremental triad qualification: PASS"
