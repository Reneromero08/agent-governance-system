#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_catvm_qanf_phase.sh EVIDENCE_DIR" >&2
    exit 2
fi

frontier_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
evidence_dir=$1
mkdir -p "$evidence_dir"

service_source="$frontier_dir/catvm_qanf_service.c"
controller_source="$frontier_dir/catvm_qanf_controller.c"
control_source="$frontier_dir/catvm_qanf_control_client.c"
phase_source="$frontier_dir/quadratic_anf_chain_phase.c"
reference_source="$frontier_dir/quadratic_anf_chain_reference.c"
phase_qualifier="$frontier_dir/qualify_quadratic_anf_chain_phase.sh"
primary_fixture="$frontier_dir/quadratic_anf_chain_primary.qanf"
reuse_fixture="$frontier_dir/quadratic_anf_chain_reuse.qanf"
degree2_fixture="$frontier_dir/quadratic_anf_chain_degree2_nonaffine.qanf"
sham_fixture="$frontier_dir/quadratic_anf_chain_affine_sham.qanf"

for tool in gcc jq sha256sum cmp rg strace nm strings stat date; do
    command -v "$tool" >/dev/null
done

common=(
    -std=c11
    -O2
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wconversion
    -Wshadow
)
service_link=(
    -fvisibility=hidden
    -ffunction-sections
    -fdata-sections
    -Wl,--gc-sections
    -lm
    -lseccomp
)

gcc "${common[@]}" "$service_source" "${service_link[@]}" \
    -o "$evidence_dir/service"
gcc "${common[@]}" -DCATVM_QANF_TESTING=1 \
    "$service_source" "${service_link[@]}" \
    -o "$evidence_dir/service_testing"
gcc "${common[@]}" -DCATVM_QANF_TRACE_BUILD=1 \
    "$service_source" "${service_link[@]}" \
    -o "$evidence_dir/service_trace"
gcc "${common[@]}" -DCATVM_QANF_SIZE_PROBE=1 \
    "$service_source" "${service_link[@]}" \
    -o "$evidence_dir/service_size_probe"
gcc "${common[@]}" "$controller_source" \
    -o "$evidence_dir/controller"
gcc "${common[@]}" "$control_source" \
    -o "$evidence_dir/control_client"
gcc "${common[@]}" "$phase_source" -lm \
    -o "$evidence_dir/direct_phase"
gcc "${common[@]}" "$reference_source" \
    -o "$evidence_dir/reference"

gcc "${common[@]}" -fanalyzer \
    "$service_source" "${service_link[@]}" \
    -o "$evidence_dir/service_analyzer"
gcc "${common[@]}" -fanalyzer "$controller_source" \
    -o "$evidence_dir/controller_analyzer"

sanitizer_flags=(
    -std=c11
    -O1
    -g
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wconversion
    -Wshadow
    -fsanitize=address,undefined
    -fno-omit-frame-pointer
    -fno-sanitize-recover=all
)
gcc "${sanitizer_flags[@]}" -DCATVM_SANITIZER_BUILD=1 \
    "$service_source" "${service_link[@]}" \
    -o "$evidence_dir/service_sanitized"
gcc "${sanitizer_flags[@]}" "$controller_source" \
    -o "$evidence_dir/controller_sanitized"

service_arguments=(
    "$primary_fixture"
    "$reuse_fixture"
    "$degree2_fixture"
    "$sham_fixture"
)

wait_for_socket() {
    local socket_path=$1
    local process_id=$2
    local ready=0
    for _ in $(seq 1 400); do
        if [[ -S "$socket_path" ]]; then
            ready=1
            break
        fi
        if ! kill -0 "$process_id" 2>/dev/null; then
            break
        fi
        sleep 0.01
    done
    [[ "$ready" -eq 1 ]]
}

run_production_case() {
    local name=$1
    local service=$2
    local controller=$3
    local cycles=$4
    local case_dir="$evidence_dir/$name"
    mkdir "$case_dir"
    local socket_path="$case_dir/service.sock"
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        "$service" "$socket_path" "${service_arguments[@]}" \
        >"$case_dir/service.stdout" \
        2>"$case_dir/service.stderr" &
    local service_pid=$!
    wait_for_socket "$socket_path" "$service_pid"
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
    local extra_cycles=${2:-0}
    local case_dir="$evidence_dir/control_$mode"
    mkdir "$case_dir"
    local socket_path="$case_dir/service.sock"
    "$evidence_dir/service_testing" \
        "$socket_path" "${service_arguments[@]}" "$mode" \
        >"$case_dir/service.stdout" \
        2>"$case_dir/service.stderr" &
    local service_pid=$!
    wait_for_socket "$socket_path" "$service_pid"
    set +e
    "$evidence_dir/control_client" \
        "$socket_path" "$mode" "$extra_cycles" \
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
    jq -e \
        --arg mode "$mode" \
        '.result == "PASS"
         and .control == $mode
         and .production_protocol_selector == false' \
        "$case_dir/result.json" >/dev/null
}

run_production_case accepted \
    "$evidence_dir/service" "$evidence_dir/controller" 256
run_production_case replay \
    "$evidence_dir/service" "$evidence_dir/controller" 256
run_production_case sanitized \
    "$evidence_dir/service_sanitized" \
    "$evidence_dir/controller_sanitized" 16
cmp "$evidence_dir/accepted/result.json" \
    "$evidence_dir/replay/result.json"

run_control_case wrong-z
run_control_case missing-z
run_control_case reordered
run_control_case snapshot 30
run_control_case inert 30

set +e
"$evidence_dir/service" \
    >"$evidence_dir/null_carrier.stdout" \
    2>"$evidence_dir/null_carrier.stderr"
null_carrier_rc=$?
set -e
[[ "$null_carrier_rc" -eq 2 ]]
[[ ! -s "$evidence_dir/null_carrier.stdout" ]]
[[ ! -s "$evidence_dir/null_carrier.stderr" ]]

jq -e '
    (keys | sort) == ([
      "actual_inverse",
      "all_intermediate_projection_requests_denied",
      "carrier_creation_count",
      "controller_computes_boundary_independently",
      "controller_phase_core_linked",
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
      "transactions",
      "unknown_command_denied"
    ] | sort)
    and .result == "PASS"
    and .protocol == "CATVM_QANF_PHASE_1"
    and .plan_fnv1a64 == "f8198cf1e338bbb5"
    and .carrier_creation_count == 1
    and .transactions == 258
    and .primary.variant == 0
    and .primary.generation == 1
    and .primary.coefficients == [1,0,0,0,1]
    and .reuse.variant == 1
    and .reuse.generation == 2
    and .reuse.coefficients == [1,1,1,1,1]
    and .primary.coefficients != .reuse.coefficients
    and .actual_inverse == true
    and .snapshot_reload == false
    and .same_service_process == true
    and .same_actual_restored_carrier == true
    and .controller_phase_core_linked == false
    and .controller_computes_boundary_independently == false
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
' "$evidence_dir/accepted/result.json" >/dev/null

"$evidence_dir/direct_phase" \
    "$primary_fixture" "$reuse_fixture" \
    >"$evidence_dir/direct.json" \
    2>"$evidence_dir/direct.stderr"
"$evidence_dir/reference" "$primary_fixture" \
    >"$evidence_dir/reference_primary.json"
"$evidence_dir/reference" "$reuse_fixture" \
    >"$evidence_dir/reference_reuse.json"
[[ ! -s "$evidence_dir/direct.stderr" ]]
jq -c '[
    {boundary_coefficients: .primary.coefficients},
    {boundary_coefficients: .reuse.coefficients}
] | .[]' "$evidence_dir/accepted/result.json" \
    >"$evidence_dir/service_boundaries.jsonl"
jq -c '[
    {boundary_coefficients: .primary_boundary},
    {boundary_coefficients: .reuse_boundary}
] | .[]' "$evidence_dir/direct.json" \
    >"$evidence_dir/direct_boundaries.jsonl"
jq -c '{boundary_coefficients}' \
    "$evidence_dir/reference_primary.json" \
    "$evidence_dir/reference_reuse.json" \
    >"$evidence_dir/reference_boundaries.jsonl"
cmp "$evidence_dir/service_boundaries.jsonl" \
    "$evidence_dir/direct_boundaries.jsonl"
cmp "$evidence_dir/service_boundaries.jsonl" \
    "$evidence_dir/reference_boundaries.jsonl"

if nm -a "$evidence_dir/service" \
    | rg -q \
        'qcs_reviewed_standalone_main|qb_print_boundary'
then
    echo "standalone report machinery survived service link" >&2
    exit 1
fi
if nm -a "$evidence_dir/controller" \
    | awk '{print $NF}' \
    | rg -q '^(qb_|phase_|carrier|complex|cexp|root3)'
then
    echo "controller links phase or carrier machinery" >&2
    exit 1
fi
if strings "$evidence_dir/service" \
    | rg -q '^wrong-z$|^missing-z$|^snapshot$|^reordered$|^inert$'
then
    echo "production service contains a test selector" >&2
    exit 1
fi
if strings "$evidence_dir/service" \
    | rg -q \
        'primary_boundary|reuse_boundary|same_carrier_transactions'
then
    echo "standalone answer report survived service link" >&2
    exit 1
fi
if rg -n \
    'expected.*(boundary|coefficient)|answer.*(table|vector)|theta.*gamma|theta.*delta' \
    "$controller_source"
then
    echo "controller contains an answer-bearing evaluator seam" >&2
    exit 1
fi
[[ "$(rg -n '\bqb_execute\(' "$service_source" | wc -l)" -eq 1 ]]
rg -q 'QB_INVERSE_CORRECT' "$service_source"
rg -q 'E_INTERMEDIATE_PROJECTION_DENIED' "$service_source"
rg -q 'SCMP_ACT_KILL_PROCESS' "$service_source"
if sed -n \
    '/static int qcs_install_seccomp/,/^}/p' \
    "$service_source" \
    | rg -q \
        'SCMP_SYS\((open|openat|creat|socket|connect|accept|fork|clone|execve|ptrace|process_vm_readv|process_vm_writev|memfd_create|shmget)'
then
    echo "post-custody seccomp exposes a forbidden syscall" >&2
    exit 1
fi

trace_dir="$evidence_dir/trace"
mkdir "$trace_dir"
trace_socket="$trace_dir/service.sock"
strace -qq -f -s 256 \
    -e trace=%file,%network,%ipc,%memory,write,writev,pwrite64,process_vm_writev,ptrace,ioctl \
    -o "$trace_dir/service.strace" \
    "$evidence_dir/service_trace" \
    "$trace_socket" "${service_arguments[@]}" \
    >"$trace_dir/service.stdout" \
    2>"$trace_dir/service.stderr" &
trace_pid=$!
wait_for_socket "$trace_socket" "$trace_pid"
"$evidence_dir/controller" "$trace_socket" 8 \
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
' "$trace_dir/service.strace" \
    >"$trace_dir/post_custody.strace"
[[ -s "$trace_dir/post_custody.strace" ]]
rg -q ' recvfrom\(' "$trace_dir/post_custody.strace"
rg -q ' sendto\(' "$trace_dir/post_custody.strace"
if rg -q \
    'open(at)?\(|creat\(|socket\(|connect\(|bind\(|listen\(|accept(4)?\(|MAP_SHARED|shm(get|at|dt|ctl)|msg(get|snd|rcv|ctl)|sem(get|op|ctl)|process_vm_writev|ptrace\(|ioctl\(|write(v)?\((1|2),' \
    "$trace_dir/post_custody.strace"
then
    echo "post-custody trace exposed a forbidden output channel" >&2
    exit 1
fi

"$evidence_dir/service_size_probe" --size-probe \
    >"$evidence_dir/sizes.json"
jq -e '
    .context_bytes > 0
    and .mapped_context_bytes >= .context_bytes
    and .program_table_bytes > 0
    and .sealed_program_table_bytes == .program_table_bytes
    and .carrier_cells == 23
    and .live_carrier_bytes == 736
    and .sealed_verification_state_bytes == 736
    and .execution_snapshot_bytes == 736
    and .execution_summary_bytes > 0
    and .request_buffer_bytes == 128
    and .response_buffer_bytes == 512
' "$evidence_dir/sizes.json" >/dev/null

measure_service() {
    local name=$1
    local service=$2
    local client=$3
    local mode=$4
    local cycles=$5
    local case_dir="$evidence_dir/measure_$name"
    mkdir "$case_dir"
    local socket_path="$case_dir/service.sock"
    if [[ "$mode" == accepted ]]; then
        "$service" "$socket_path" "${service_arguments[@]}" \
            >"$case_dir/service.stdout" \
            2>"$case_dir/service.stderr" &
    else
        "$service" "$socket_path" \
            "${service_arguments[@]}" "$mode" \
            >"$case_dir/service.stdout" \
            2>"$case_dir/service.stderr" &
    fi
    local service_pid=$!
    wait_for_socket "$socket_path" "$service_pid"
    local start_ns
    local end_ns
    start_ns=$(date +%s%N)
    if [[ "$mode" == accepted ]]; then
        "$client" "$socket_path" "$cycles" \
            >"$case_dir/result.json"
    else
        "$client" "$socket_path" "$mode" "$cycles" \
            >"$case_dir/result.json"
    fi
    end_ns=$(date +%s%N)
    wait "$service_pid"
    [[ ! -s "$case_dir/service.stdout" ]]
    [[ ! -s "$case_dir/service.stderr" ]]
    printf '%s\n' "$((end_ns - start_ns))"
}

direct_start_ns=$(date +%s%N)
"$evidence_dir/direct_phase" \
    "$primary_fixture" "$reuse_fixture" \
    >"$evidence_dir/measure_direct.json"
direct_end_ns=$(date +%s%N)
direct_ns=$((direct_end_ns - direct_start_ns))
in_place_ns=$(measure_service \
    in_place "$evidence_dir/service" \
    "$evidence_dir/controller" accepted 256)
snapshot_ns=$(measure_service \
    snapshot "$evidence_dir/service_testing" \
    "$evidence_dir/control_client" snapshot 256)
inert_ns=$(measure_service \
    inert "$evidence_dir/service_testing" \
    "$evidence_dir/control_client" inert 256)

service_binary_bytes=$(stat -c%s "$evidence_dir/service")
controller_binary_bytes=$(stat -c%s "$evidence_dir/controller")
testing_service_binary_bytes=$(
    stat -c%s "$evidence_dir/service_testing"
)
direct_binary_bytes=$(stat -c%s "$evidence_dir/direct_phase")
request_bytes=$(jq -r '.request_bytes' \
    "$evidence_dir/accepted/result.json")
response_bytes=$(jq -r '.response_bytes' \
    "$evidence_dir/accepted/result.json")

jq -n \
    --arg claim \
        "CATVM_ENFORCED_FIXED_SCHEMA_QUADRATIC_ANF_TWO_HIDDEN_PORT_COMPOSITION_ESTABLISHED_ON_PHASE_BACKEND" \
    --arg ceiling \
        "BOUNDED_LINUX_SAME_UID_SOFTWARE_BOOLEAN_GF2_MONIC_QAND_CHAIN_DEGREE4_FIVE_COEFFICIENT_ATOMIC_TRANSACTION_REFERENCE_ONLY" \
    --argjson service_binary_bytes "$service_binary_bytes" \
    --argjson controller_binary_bytes "$controller_binary_bytes" \
    --argjson testing_service_binary_bytes \
        "$testing_service_binary_bytes" \
    --argjson direct_binary_bytes "$direct_binary_bytes" \
    --argjson direct_ns "$direct_ns" \
    --argjson inert_ns "$inert_ns" \
    --argjson snapshot_ns "$snapshot_ns" \
    --argjson in_place_ns "$in_place_ns" \
    --argjson request_bytes "$request_bytes" \
    --argjson response_bytes "$response_bytes" \
    --slurpfile sizes "$evidence_dir/sizes.json" \
    '{
      result: "PASS",
      claim: $claim,
      claim_ceiling: $ceiling,
      protocol: "AF_UNIX_SOCK_SEQPACKET_ATOMIC_RUN",
      fixed_public_schema: "MONIC_QAND_CHAIN_F_G_J",
      plan_fnv1a64: "f8198cf1e338bbb5",
      non_affine_boundary_signature: true,
      maximum_demonstrated_boundary_degree: 4,
      controller_phase_core_linked: false,
      controller_computes_boundary_independently: false,
      final_boundary_only: true,
      unresolved_h_serialized: false,
      actual_resident_h_consumed_by_z: true,
      actual_inverse: true,
      snapshot_reload: false,
      actual_restored_carrier_reuse: true,
      carrier_creation_count: 1,
      accepted_transactions: 258,
      repeated_reuse_cycles: 256,
      exact_operations_per_accepted_transaction: {
        phase_products: 18,
        carrier_reads: 97,
        phase_cell_updates: 46,
        final_boundary_decodes: 5,
        intermediate_decodes: 0,
        intermediate_copies: 0,
        boundary_copies: 2
      },
      controls: {
        wrong_z_inverse: "PASS_TEST_BUILD_ONLY",
        missing_z_inverse: "PASS_TEST_BUILD_ONLY",
        reordered_noncommuting_inverse: "PASS_TEST_BUILD_ONLY",
        snapshot: "PASS_SEPARATE_WEAKER_TEST_BUILD_ONLY",
        inert_boundary: "PASS_TEST_BUILD_ONLY",
        null_carrier: "PASS_REJECTED",
        intermediate_projection: "PASS_FIXED_DENIAL",
        same_uid_inspection_attempts_denied: 6,
        deterministic_replay: "PASS",
        independent_reference_parity: "PASS",
        direct_phase_parity: "PASS",
        analyzer: "PASS",
        sanitizer: "PASS",
        no_smuggle_trace:
          "PASS_ALLOWLIST_SECCOMP_ZERO_SERVICE_STDIO_POST_CUSTODY_TRACE"
      },
      resources: {
        sizes: $sizes[0],
        binaries: {
          service_bytes: $service_binary_bytes,
          controller_bytes: $controller_binary_bytes,
          testing_service_bytes: $testing_service_binary_bytes,
          direct_phase_bytes: $direct_binary_bytes
        },
        accepted_traffic_258_transactions: {
          request_bytes: $request_bytes,
          response_bytes: $response_bytes
        },
        uncontrolled_fixed_batch_wall_ns: {
          direct_process_phase_258_transactions: $direct_ns,
          isolated_inert_boundary_258_transactions: $inert_ns,
          snapshot_catvm_258_transactions: $snapshot_ns,
          accepted_in_place_catvm_258_transactions: $in_place_ns
        }
      },
      performance_advantage: false,
      total_memory_advantage: false,
      general_anf_elimination: false,
      arbitrary_non_affine_closure: false,
      general_holographic_relational_computation: false,
      small_wall_crossing: false,
      physical_waveform_or_silicon_computation: false,
      unlimited_catalytic_computation: false
    }' >"$evidence_dir/result.json"

bash "$phase_qualifier" "$evidence_dir/qanf_regression" \
    >"$evidence_dir/qanf_regression.summary.txt"
jq -e '
    .result == "PASS"
    and .claim ==
      "CATVM_ENFORCED_FIXED_SCHEMA_QUADRATIC_ANF_TWO_HIDDEN_PORT_COMPOSITION_ESTABLISHED_ON_PHASE_BACKEND"
    and .non_affine_boundary_signature == true
    and .actual_inverse == true
    and .snapshot_reload == false
    and .actual_restored_carrier_reuse == true
    and .small_wall_crossing == false
' "$evidence_dir/result.json" >/dev/null

sha256sum \
    "$service_source" \
    "$controller_source" \
    "$control_source" \
    "$frontier_dir/qualify_catvm_qanf_phase.sh" \
    "$phase_qualifier" \
    "$phase_source" \
    "$reference_source" \
    "$primary_fixture" \
    "$reuse_fixture" \
    "$degree2_fixture" \
    "$sham_fixture" \
    "$evidence_dir/service" \
    "$evidence_dir/service_testing" \
    "$evidence_dir/service_trace" \
    "$evidence_dir/service_size_probe" \
    "$evidence_dir/controller" \
    "$evidence_dir/control_client" \
    "$evidence_dir/direct_phase" \
    "$evidence_dir/reference" \
    "$evidence_dir/accepted/result.json" \
    "$evidence_dir/replay/result.json" \
    "$evidence_dir/sanitized/result.json" \
    "$evidence_dir/control_wrong-z/result.json" \
    "$evidence_dir/control_missing-z/result.json" \
    "$evidence_dir/control_reordered/result.json" \
    "$evidence_dir/control_snapshot/result.json" \
    "$evidence_dir/control_inert/result.json" \
    "$evidence_dir/direct.json" \
    "$evidence_dir/reference_primary.json" \
    "$evidence_dir/reference_reuse.json" \
    "$evidence_dir/trace/service.strace" \
    "$evidence_dir/trace/post_custody.strace" \
    "$evidence_dir/trace/result.json" \
    "$evidence_dir/sizes.json" \
    "$evidence_dir/qanf_regression/SHA256SUMS" \
    "$evidence_dir/qanf_regression/result.json" \
    "$evidence_dir/result.json" \
    >"$evidence_dir/SHA256SUMS"

printf '%s\n' "$evidence_dir"
