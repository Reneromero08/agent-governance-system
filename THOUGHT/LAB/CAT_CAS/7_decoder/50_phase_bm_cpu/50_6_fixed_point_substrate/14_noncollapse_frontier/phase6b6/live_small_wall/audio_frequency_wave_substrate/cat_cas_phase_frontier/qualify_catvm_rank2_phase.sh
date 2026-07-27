#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SERVICE_SOURCE="$HERE/catvm_rank2_service.c"
CONTROLLER_SOURCE="$HERE/catvm_rank2_controller.c"
CONTROL_CLIENT_SOURCE="$HERE/catvm_rank2_control_client.c"
PHASE_SOURCE="$HERE/recursive_rematerializing_general_multi_dag_affine_phase.c"
REFERENCE_SOURCE="$HERE/general_multi_dag_affine_reference.c"
MANIFEST="$HERE/general_multi_dag_affine_topology.txt"
PHASE_QUALIFIER="$HERE/qualify_recursive_rematerializing_general_multi_dag_affine_phase.sh"
OUT=${1:-}
if [[ -z "$OUT" ]]; then
    OUT=$(mktemp -d /tmp/catvm-rank2.XXXXXX)
else
    mkdir -p "$OUT"
fi

for tool in gcc jq sha256sum cmp rg strace nm strings stat size date; do
    command -v "$tool" >/dev/null
done

COMMON=(
    -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror
    -Wconversion -Wshadow
)
SERVICE_LINK=(
    -fvisibility=hidden -ffunction-sections -fdata-sections
    -Wl,--gc-sections -lm -lseccomp
)

gcc "${COMMON[@]}" \
    "$SERVICE_SOURCE" "${SERVICE_LINK[@]}" \
    -o "$OUT/service"
gcc "${COMMON[@]}" -DCATVM_RANK2_TESTING=1 \
    "$SERVICE_SOURCE" "${SERVICE_LINK[@]}" \
    -o "$OUT/service_testing"
gcc "${COMMON[@]}" -DCATVM_RANK2_TRACE_BUILD=1 \
    "$SERVICE_SOURCE" "${SERVICE_LINK[@]}" \
    -o "$OUT/service_trace"
gcc "${COMMON[@]}" -DCATVM_RANK2_SIZE_PROBE=1 \
    "$SERVICE_SOURCE" "${SERVICE_LINK[@]}" \
    -o "$OUT/service_size_probe"
gcc "${COMMON[@]}" \
    "$CONTROLLER_SOURCE" -o "$OUT/controller"
gcc "${COMMON[@]}" \
    "$CONTROL_CLIENT_SOURCE" -o "$OUT/control_client"
gcc "${COMMON[@]}" -DRR_REUSE_CYCLES=0U \
    "$PHASE_SOURCE" -lm -o "$OUT/direct_phase"
gcc "${COMMON[@]}" \
    "$REFERENCE_SOURCE" -lm -o "$OUT/reference"

gcc "${COMMON[@]}" -fanalyzer \
    "$SERVICE_SOURCE" "${SERVICE_LINK[@]}" \
    -o "$OUT/service_analyzer"
gcc "${COMMON[@]}" -fanalyzer \
    "$CONTROLLER_SOURCE" -o "$OUT/controller_analyzer"

gcc -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror \
    -Wconversion -Wshadow \
    -DCATVM_SANITIZER_BUILD=1 \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    "$SERVICE_SOURCE" -lm -lseccomp \
    -o "$OUT/service_sanitized"
gcc -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror \
    -Wconversion -Wshadow \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    "$CONTROLLER_SOURCE" -o "$OUT/controller_sanitized"

run_production_case() {
    local name=$1
    local service=$2
    local controller=$3
    local cycles=$4
    local case_dir="$OUT/$name"
    mkdir "$case_dir"
    local socket_path="$case_dir/service.sock"
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        "$service" "$socket_path" "$MANIFEST" \
        >"$case_dir/service.stdout" \
        2>"$case_dir/service.stderr" &
    local service_pid=$!
    local ready=0
    for _ in $(seq 1 400); do
        if [[ -S "$socket_path" ]]; then
            ready=1
            break
        fi
        if ! kill -0 "$service_pid" 2>/dev/null; then
            break
        fi
        sleep 0.01
    done
    [[ "$ready" -eq 1 ]]
    set +e
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        "$controller" "$socket_path" "$cycles" \
        >"$case_dir/result.json" \
        2>"$case_dir/controller.stderr"
    local controller_rc=$?
    wait "$service_pid"
    local service_rc=$?
    set -e
    [[ "$controller_rc" -eq 0 ]]
    [[ "$service_rc" -eq 0 ]]
    [[ ! -e "$socket_path" ]]
    [[ ! -s "$case_dir/service.stdout" ]]
    [[ ! -s "$case_dir/service.stderr" ]]
    [[ ! -s "$case_dir/controller.stderr" ]]
    jq -e '.result == "PASS"' "$case_dir/result.json" >/dev/null
}

run_control_case() {
    local mode=$1
    local expected_service=$2
    local extra_cycles=${3:-0}
    local case_dir="$OUT/control_$mode"
    mkdir "$case_dir"
    local socket_path="$case_dir/service.sock"
    "$OUT/service_testing" \
        "$socket_path" "$MANIFEST" "$mode" \
        >"$case_dir/service.stdout" \
        2>"$case_dir/service.stderr" &
    local service_pid=$!
    local ready=0
    for _ in $(seq 1 400); do
        if [[ -S "$socket_path" ]]; then
            ready=1
            break
        fi
        if ! kill -0 "$service_pid" 2>/dev/null; then
            break
        fi
        sleep 0.01
    done
    [[ "$ready" -eq 1 ]]
    set +e
    "$OUT/control_client" \
        "$socket_path" "$mode" "$extra_cycles" \
        >"$case_dir/result.json" \
        2>"$case_dir/controller.stderr"
    local controller_rc=$?
    wait "$service_pid"
    local service_rc=$?
    set -e
    [[ "$controller_rc" -eq 0 ]]
    if [[ "$expected_service" == "zero" ]]; then
        [[ "$service_rc" -eq 0 ]]
    else
        [[ "$service_rc" -ne 0 ]]
    fi
    [[ ! -e "$socket_path" ]]
    [[ ! -s "$case_dir/service.stdout" ]]
    [[ ! -s "$case_dir/service.stderr" ]]
    [[ ! -s "$case_dir/controller.stderr" ]]
    jq -e \
        --arg mode "$mode" \
        '.result == "PASS"
         and .control == $mode
         and .production_protocol_selector == false' \
        "$case_dir/result.json" >/dev/null
}

run_production_case accepted \
    "$OUT/service" "$OUT/controller" 256
run_production_case accepted_replay \
    "$OUT/service" "$OUT/controller" 256
run_production_case sanitized \
    "$OUT/service_sanitized" "$OUT/controller_sanitized" 16
cmp "$OUT/accepted/result.json" "$OUT/accepted_replay/result.json"

run_control_case wrong-root zero
run_control_case missing-root zero
run_control_case snapshot zero 30
run_control_case inert zero 30
run_control_case reordered nonzero

jq -e '
  (keys | sort) == ([
    "actual_inverse",
    "all_intermediate_projection_requests_denied",
    "carrier_creation_count",
    "embedded_nul_denied",
    "oversize_packet_denied",
    "pidfd_getfd_denied",
    "plan_fnv1a64",
    "primary",
    "proc_fd_denied",
    "proc_maps_denied",
    "proc_mem_denied",
    "process_vm_readv_denied",
    "protocol",
    "ptrace_denied",
    "request_bytes",
    "request_packets",
    "response_bytes",
    "response_packets",
    "result",
    "reuse",
    "same_actual_restored_carrier",
    "same_service_process",
    "snapshot_reload",
    "topology_fnv1a64",
    "transactions",
    "unknown_command_denied"
  ] | sort)
  and .result == "PASS"
  and .protocol == "CATVM_RANK2_PHASE_1"
  and .plan_fnv1a64 == "d888309ed9ac3f32"
  and .topology_fnv1a64 == "41d917d4a3308fbe"
  and .carrier_creation_count == 1
  and .transactions == 258
  and .actual_inverse == true
  and .snapshot_reload == false
  and .same_service_process == true
  and .same_actual_restored_carrier == true
  and .all_intermediate_projection_requests_denied == true
  and .embedded_nul_denied == true
  and .oversize_packet_denied == true
  and .unknown_command_denied == true
  and .proc_mem_denied == true
  and .proc_maps_denied == true
  and .proc_fd_denied == true
  and .process_vm_readv_denied == true
  and .ptrace_denied == true
  and .pidfd_getfd_denied == true
  and .primary.variant == 0
  and .primary.generation == 1
  and .reuse.variant == 1
  and .reuse.generation == 2
  and (.primary.coefficients | length) == 49
  and (.reuse.coefficients | length) == 49
  and .primary.coefficients != .reuse.coefficients
  and all(.primary.coefficients[]; . >= 0 and . <= 2)
  and all(.reuse.coefficients[]; . >= 0 and . <= 2)
' "$OUT/accepted/result.json" >/dev/null

"$OUT/direct_phase" "$MANIFEST" >"$OUT/direct.jsonl"
"$OUT/reference" "$MANIFEST" >"$OUT/reference.jsonl"
jq -c '
  [
    .primary,
    .reuse
  ][]
  | {
      boundary_coefficients: .coefficients,
      boundary_fnv1a64
    }
' "$OUT/accepted/result.json" >"$OUT/service_boundaries.jsonl"
jq -c '
  select(
    .mode == "recursive-remat-primary"
    or .mode == "recursive-remat-reuse"
  )
  | {boundary_coefficients, boundary_fnv1a64}
' "$OUT/direct.jsonl" >"$OUT/direct_boundaries.jsonl"
jq -sc '
  .[0:2][]
  | {boundary_coefficients, boundary_fnv1a64}
' "$OUT/reference.jsonl" >"$OUT/reference_boundaries.jsonl"
cmp "$OUT/service_boundaries.jsonl" "$OUT/direct_boundaries.jsonl"
cmp "$OUT/service_boundaries.jsonl" "$OUT/reference_boundaries.jsonl"

if nm -a "$OUT/service" \
    | rg -q \
        'catvm_rank2_reviewed_standalone_main|rr_print|rr_print_receipt_plan'
then
    exit 1
fi
if nm -a "$OUT/controller" \
    | awk '{print $NF}' \
    | rg -q \
        '^(rr_|ga_|rc_|dc_|complex|cexp|root3|phase_|carrier)'
then
    exit 1
fi
if strings "$OUT/service" \
    | rg -q \
        '^wrong-root$|^missing-root$|^snapshot$|^reordered$|^inert$'
then
    exit 1
fi
if rg -q \
    '2e2200557469163d|6f46809cf9d2cb8e|boundary.*=.*\{' \
    "$CONTROLLER_SOURCE"
then
    exit 1
fi
[[ "$(rg -n '\brr_execute\(' "$SERVICE_SOURCE" | wc -l)" -eq 1 ]]
[[ "$(rg -n '\brr_print\(' "$SERVICE_SOURCE" | wc -l)" -eq 0 ]]
rg -q 'RC_RESTORE_CORRECT' "$SERVICE_SOURCE"
rg -q 'RR_FAULT_NONE' "$SERVICE_SOURCE"
rg -q 'E_INTERMEDIATE_PROJECTION_DENIED' "$SERVICE_SOURCE"
rg -q 'SCMP_ACT_KILL_PROCESS' "$SERVICE_SOURCE"
if sed -n \
    '/static int cdg_install_seccomp/,/^}/p' \
    "$SERVICE_SOURCE" \
    | rg -q \
        'SCMP_SYS\((open|openat|creat|socket|connect|accept|fork|clone|execve|ptrace|process_vm_readv|process_vm_writev|memfd_create|shmget)'
then
    exit 1
fi

trace_dir="$OUT/trace"
mkdir "$trace_dir"
trace_socket="$trace_dir/service.sock"
strace -qq -f -s 256 \
    -e trace=%file,%network,%ipc,%memory,write,writev,pwrite64,process_vm_writev,ptrace,ioctl \
    -o "$trace_dir/service.strace" \
    "$OUT/service_trace" "$trace_socket" "$MANIFEST" \
    >"$trace_dir/service.stdout" \
    2>"$trace_dir/service.stderr" &
trace_pid=$!
for _ in $(seq 1 400); do
    [[ -S "$trace_socket" ]] && break
    kill -0 "$trace_pid" 2>/dev/null || break
    sleep 0.01
done
"$OUT/controller" "$trace_socket" 8 \
    >"$trace_dir/result.json" \
    2>"$trace_dir/controller.stderr"
wait "$trace_pid"
[[ ! -s "$trace_dir/service.stdout" ]]
[[ ! -s "$trace_dir/service.stderr" ]]
[[ ! -s "$trace_dir/controller.stderr" ]]
jq -e '.result == "PASS"' "$trace_dir/result.json" >/dev/null
[[ "$(rg -c ' unlink\(' "$trace_dir/service.strace")" -eq 1 ]]
awk '
  / unlink\(/ {sealed=1; next}
  sealed {print}
' "$trace_dir/service.strace" >"$trace_dir/post_custody.strace"
[[ -s "$trace_dir/post_custody.strace" ]]
rg -q ' recvfrom\(' "$trace_dir/post_custody.strace"
rg -q ' sendto\(' "$trace_dir/post_custody.strace"
if rg -q \
    'open(at)?\(|creat\(|socket\(|connect\(|bind\(|listen\(|accept(4)?\(|MAP_SHARED|shm(get|at|dt|ctl)|msg(get|snd|rcv|ctl)|sem(get|op|ctl)|process_vm_writev|ptrace\(|ioctl\(|write(v)?\((1|2),' \
    "$trace_dir/post_custody.strace"
then
    exit 1
fi

"$OUT/service_size_probe" --size-probe >"$OUT/sizes.json"
jq -e '
  .context_bytes > 0
  and .compiled_topology_bytes == 2664
  and .plan_bytes == 13192
  and .program_table_bytes == 1960
  and .machine_counter_bytes == 24
  and .execution_summary_bytes == 1400
  and .carrier_cells == 849
  and .live_carrier_bytes == 27168
  and .verification_snapshot_bytes == 27168
  and .request_buffer_bytes == 128
  and .response_buffer_bytes == 1024
' "$OUT/sizes.json" >/dev/null

gcc "${COMMON[@]}" -DRR_REUSE_CYCLES=995U \
    "$PHASE_SOURCE" -lm -o "$OUT/warm_direct"

measure_service() {
    local name=$1
    local service=$2
    local client=$3
    local mode=$4
    local cycles=$5
    local case_dir="$OUT/measure_$name"
    mkdir "$case_dir"
    local socket_path="$case_dir/service.sock"
    if [[ "$mode" == "accepted" ]]; then
        "$service" "$socket_path" "$MANIFEST" \
            >"$case_dir/service.stdout" \
            2>"$case_dir/service.stderr" &
    else
        "$service" "$socket_path" "$MANIFEST" "$mode" \
            >"$case_dir/service.stdout" \
            2>"$case_dir/service.stderr" &
    fi
    local service_pid=$!
    for _ in $(seq 1 400); do
        [[ -S "$socket_path" ]] && break
        sleep 0.01
    done
    local start
    local end
    start=$(date +%s%N)
    if [[ "$mode" == "accepted" ]]; then
        "$client" "$socket_path" "$cycles" \
            >"$case_dir/result.json"
    else
        "$client" "$socket_path" "$mode" "$cycles" \
            >"$case_dir/result.json"
    fi
    end=$(date +%s%N)
    wait "$service_pid"
    [[ ! -s "$case_dir/service.stdout" ]]
    [[ ! -s "$case_dir/service.stderr" ]]
    printf '%s\n' "$((end - start))"
}

direct_start=$(date +%s%N)
"$OUT/warm_direct" "$MANIFEST" >"$OUT/warm_direct.jsonl"
direct_end=$(date +%s%N)
direct_ns=$((direct_end - direct_start))
in_place_ns=$(measure_service \
    in_place "$OUT/service" "$OUT/controller" accepted 998)
snapshot_ns=$(measure_service \
    snapshot "$OUT/service_testing" "$OUT/control_client" snapshot 998)
inert_ns=$(measure_service \
    inert "$OUT/service_testing" "$OUT/control_client" inert 998)

service_binary_bytes=$(stat -c%s "$OUT/service")
controller_binary_bytes=$(stat -c%s "$OUT/controller")
testing_service_binary_bytes=$(stat -c%s "$OUT/service_testing")
accepted_request_bytes=$(jq -r '.request_bytes' "$OUT/accepted/result.json")
accepted_response_bytes=$(jq -r '.response_bytes' "$OUT/accepted/result.json")

jq -n \
  --arg claim \
    "CATVM_ENFORCED_15_NODE_RANK2_AUTOMATIC_SCHEDULED_SHARED_RELATIONAL_DAG_ESTABLISHED_ON_PHASE_BACKEND" \
  --arg ceiling \
    "BOUNDED_LINUX_SAME_UID_SOFTWARE_GF2_WIDTH3_EXACT_15_NODE_ATOMIC_RUN_28_FORWARD_28_REVERSE_9_WORKING_SLOTS_REFERENCE_ONLY" \
  --argjson service_binary_bytes "$service_binary_bytes" \
  --argjson controller_binary_bytes "$controller_binary_bytes" \
  --argjson testing_service_binary_bytes "$testing_service_binary_bytes" \
  --argjson direct_ns "$direct_ns" \
  --argjson inert_ns "$inert_ns" \
  --argjson snapshot_ns "$snapshot_ns" \
  --argjson in_place_ns "$in_place_ns" \
  --argjson request_bytes "$accepted_request_bytes" \
  --argjson response_bytes "$accepted_response_bytes" \
  --slurpfile sizes "$OUT/sizes.json" \
  '{
    claim: $claim,
    claim_ceiling: $ceiling,
    result: "PASS",
    protocol: "AF_UNIX_SOCK_SEQPACKET_ATOMIC_RUN",
    exact_public_topology: true,
    automatic_rank2_schedule: true,
    controller_phase_core_linked: false,
    controller_computes_boundary_independently: false,
    final_boundary_only: true,
    unresolved_intermediate_serialized: false,
    actual_inverse: true,
    snapshot_reload: false,
    actual_restored_carrier_reuse: true,
    carrier_creation_count: 1,
    accepted_transactions: 258,
    repeated_reuse_cycles: 256,
    wrong_root_control: "PASS_TEST_BUILD_ONLY",
    missing_root_control: "PASS_TEST_BUILD_ONLY",
    reordered_inverse_control:
      "PASS_STRUCTURAL_FAIL_CLOSED_TEST_BUILD_ONLY",
    snapshot_control:
      "PASS_SEPARATE_WEAKER_TEST_BUILD_ONLY",
    inert_boundary_control:
      "PASS_TEST_BUILD_ONLY",
    intermediate_projection_control: "PASS_FIXED_DENIAL",
    no_smuggle:
      "PASS_PROTOCOL_KEY_ALLOWLIST_SECCOMP_TRACE_ZERO_STDIO_CONTROLLER_UNLINKED",
    same_uid_inspection_controls: 6,
    sizes: $sizes[0],
    binaries: {
      service_bytes: $service_binary_bytes,
      controller_bytes: $controller_binary_bytes,
      testing_service_bytes: $testing_service_binary_bytes
    },
    accepted_traffic_258_transactions: {
      request_bytes: $request_bytes,
      response_bytes: $response_bytes
    },
    uncontrolled_warm_1000_transaction_wall_ns: {
      direct_process_phase: $direct_ns,
      isolated_inert_boundary: $inert_ns,
      snapshot_catvm: $snapshot_ns,
      accepted_in_place_catvm: $in_place_ns
    },
    performance_advantage: false,
    total_memory_advantage: false,
    non_affine_relation_signature: false,
    general_holographic_relational_computation: false,
    small_wall_crossing: false,
    physical_waveform_or_silicon_computation: false,
    unlimited_catalytic_computation: false
  }' >"$OUT/result.json"

bash "$PHASE_QUALIFIER" "$OUT/scheduler_regression" \
    >"$OUT/scheduler_regression.summary.json"
jq -e '.result == "PASS"' "$OUT/result.json" >/dev/null

sha256sum \
    "$SERVICE_SOURCE" \
    "$CONTROLLER_SOURCE" \
    "$CONTROL_CLIENT_SOURCE" \
    "$PHASE_SOURCE" \
    "$MANIFEST" \
    "$OUT/service" \
    "$OUT/controller" \
    "$OUT/accepted/result.json" \
    "$OUT/trace/service.strace" \
    "$OUT/result.json" \
    >"$OUT/SHA256SUMS"

printf '%s\n' "$OUT"
