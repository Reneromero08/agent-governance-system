#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_catvm_nonlinear_kerr_wave.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$1
mkdir -p "$out"
service_source="$here/catvm_nonlinear_kerr_wave_service.c"
controller_source="$here/catvm_nonlinear_kerr_wave_controller.c"
raw_client_source="$here/catvm_nonlinear_kerr_wave_raw_client.c"
benchmark_client_source="$here/catvm_nonlinear_graph_benchmark_client.c"
benchmark_direct_source="$here/nonlinear_kerr_wave_warm_benchmark.c"
phase_source="$here/algebraic_nonlinear_kerr_wave.c"
reference_source="$here/algebraic_nonlinear_kerr_wave_reference.c"

for tool in gcc jq cmp rg sha256sum nm strings nice strace; do
    command -v "$tool" >/dev/null
done

common=(
    -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror
    -Wconversion -Wshadow
)
service_link=(
    -fvisibility=hidden -ffunction-sections -fdata-sections
    -Wl,--gc-sections -lm -lseccomp
)
gcc "${common[@]}" "$service_source" "${service_link[@]}" \
    -o "$out/service"
gcc "${common[@]}" -DCATVM_KERR_TESTING=1 \
    "$service_source" "${service_link[@]}" \
    -o "$out/service_testing"
gcc "${common[@]}" -DCATVM_KERR_TRACE_BUILD=1 \
    "$service_source" "${service_link[@]}" \
    -o "$out/service_trace"
gcc "${common[@]}" -DCATVM_KERR_SIZE_PROBE=1 \
    "$service_source" "${service_link[@]}" \
    -o "$out/service_size"
gcc "${common[@]}" "$controller_source" -lm -o "$out/controller"
gcc "${common[@]}" "$raw_client_source" -o "$out/raw_client"
gcc "${common[@]}" "$benchmark_client_source" \
    -o "$out/benchmark_client"
gcc "${common[@]}" "$benchmark_direct_source" -lm \
    -o "$out/benchmark_direct"
gcc "${common[@]}" -fanalyzer "$service_source" \
    "${service_link[@]}" -o "$out/service_analyzer"
gcc "${common[@]}" -fanalyzer "$controller_source" -lm \
    -o "$out/controller_analyzer"
gcc \
    -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror \
    -Wconversion -Wshadow -DCATVM_SANITIZER_BUILD=1 \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    -fno-sanitize-recover=all \
    "$service_source" -lm -lseccomp \
    -o "$out/service_sanitized"
gcc \
    -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror \
    -Wconversion -Wshadow \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    -fno-sanitize-recover=all \
    "$controller_source" -lm -o "$out/controller_sanitized"

wait_socket() {
    local socket=$1
    local service_pid=$2
    local ready=0
    for _ in $(seq 1 500); do
        if [[ -S "$socket" ]]; then
            ready=1
            break
        fi
        if ! kill -0 "$service_pid" 2>/dev/null; then
            break
        fi
        sleep 0.01
    done
    [[ "$ready" -eq 1 ]]
}

run_case() {
    local name=$1
    local service=$2
    local service_mode=$3
    local controller=$4
    local controller_mode=$5
    local cycles=$6
    local case_dir="$out/$name"
    mkdir "$case_dir"
    local socket="$case_dir/service.sock"
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        nice -n 10 "$service" "$socket" 128 "$service_mode" \
        >"$case_dir/service.stdout" \
        2>"$case_dir/service.stderr" &
    local service_pid=$!
    wait_socket "$socket" "$service_pid"
    local vm_rss_kib
    local vm_lck_kib
    vm_rss_kib=$(
        awk '/^VmRSS:/ {print $2+0}' \
            "/proc/$service_pid/status"
    )
    vm_lck_kib=$(
        awk '/^VmLck:/ {print $2+0}' \
            "/proc/$service_pid/status"
    )
    jq -n \
        --argjson vm_rss_kib "$vm_rss_kib" \
        --argjson vm_lck_kib "$vm_lck_kib" '
        {
          warm_service_vm_rss_kib: $vm_rss_kib,
          warm_service_vm_lck_kib: $vm_lck_kib
        }
    ' >"$case_dir/process_memory.json"
    set +e
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        nice -n 10 "$controller" \
        "$socket" "$cycles" "$controller_mode" \
        >"$case_dir/result.json" \
        2>"$case_dir/controller.stderr"
    local controller_rc=$?
    wait "$service_pid"
    local service_rc=$?
    set -e
    [[ "$controller_rc" -eq 0 ]]
    [[ "$service_rc" -eq 0 ]]
    [[ ! -e "$socket" ]]
    [[ ! -s "$case_dir/service.stdout" ]]
    [[ ! -s "$case_dir/service.stderr" ]]
    [[ ! -s "$case_dir/controller.stderr" ]]
    jq -e '.result == "PASS"' "$case_dir/result.json" >/dev/null
}

run_case \
    accepted "$out/service" correct "$out/controller" correct 256
run_case \
    accepted_replay "$out/service" correct "$out/controller" correct 256
run_case \
    sanitized "$out/service_sanitized" correct \
    "$out/controller_sanitized" correct 16
for mode in missing wrong reordered; do
    run_case \
        "control_$mode" "$out/service_testing" "$mode" \
        "$out/controller" restoration 1
done
run_case \
    control_snapshot "$out/service_testing" snapshot \
    "$out/controller" snapshot 8
run_case \
    control_inert "$out/service_testing" inert \
    "$out/controller" inert 8
cmp "$out/accepted/result.json" "$out/accepted_replay/result.json"

set +e
"$out/service_testing" \
    "$out/null.sock" 128 null \
    >"$out/null.stdout" 2>"$out/null.stderr"
null_rc=$?
set -e
[[ "$null_rc" -eq 2 ]]
[[ ! -e "$out/null.sock" ]]
[[ ! -s "$out/null.stdout" ]]
[[ ! -s "$out/null.stderr" ]]

jq -e '
    .result == "PASS"
    and .protocol == "CATVM_NONLINEAR_KERR_WAVE_1"
    and .cells == 4
    and .depth == 128
    and .transactions == 256
    and .carrier_creation_count == 1
    and .mode == "correct"
    and .same_service_process
    and .same_actual_restored_carrier
    and .actual_inverse
    and (.snapshot_reload | not)
    and .all_intermediate_projection_requests_denied
    and .null_carrier_request_denied
    and .unknown_command_denied
    and .proc_mem_denied
    and .proc_maps_denied
    and .proc_fd_denied
    and .process_vm_readv_denied
    and .ptrace_denied
    and .pidfd_getfd_denied
    and (
      (.continuous_boundary_tolerance - 2e-10) | abs
    ) <= 1e-25
    and .maximum_repeated_boundary_drift
      <= .continuous_boundary_tolerance
    and .maximum_restoration_error
      <= .continuous_boundary_tolerance
    and (.repeated_boundary_hash_exactness_required | not)
    and .primary.program == 0
    and .primary.generation == 1
    and .primary.restoration_error <= 2e-10
    and .reuse.program == 1
    and .reuse.generation == 2
    and .reuse.restoration_error <= 2e-10
    and .primary.boundary_fnv1a64 != .reuse.boundary_fnv1a64
' "$out/accepted/result.json" >/dev/null
jq -e '
    .warm_service_vm_rss_kib > 0
    and .warm_service_vm_lck_kib > 0
    and .warm_service_vm_lck_kib <= .warm_service_vm_rss_kib
' "$out/accepted/process_memory.json" >/dev/null

for mode in missing wrong reordered; do
    jq -e '
        .result == "PASS"
        and .restoration_failure_detected
    ' "$out/control_$mode/result.json" >/dev/null
done
jq -e '
    .result == "PASS"
    and .mode == "snapshot"
    and .primary.generation == 0
    and .reuse.generation == 0
    and .snapshot_reload
    and (.actual_inverse | not)
    and (.same_actual_restored_carrier | not)
' "$out/control_snapshot/result.json" >/dev/null
jq -e '
    .result == "PASS"
    and .mode == "inert"
    and .primary.generation == 0
    and .reuse.generation == 0
    and (.actual_inverse | not)
    and (.same_actual_restored_carrier | not)
' "$out/control_inert/result.json" >/dev/null

"$out/benchmark_direct" 128 1 compare >"$out/direct_compare.json"
jq -e \
    --slurpfile service "$out/accepted/result.json" '
    .result == "PASS"
    and .primary_boundary_fnv1a64
      == .compact_primary_boundary_fnv1a64
    and .reuse_boundary_fnv1a64
      == .compact_reuse_boundary_fnv1a64
    and .primary_boundary_fnv1a64
      == $service[0].primary.boundary_fnv1a64
    and .reuse_boundary_fnv1a64
      == $service[0].reuse.boundary_fnv1a64
    and .primary_restoration_error <= 2e-10
    and .reuse_restoration_error <= 2e-10
    and .carrier_complex_cells == 4
    and .compact_state_bytes == 64
' "$out/direct_compare.json" >/dev/null

"$out/service_size" --size-probe >"$out/service_size.json"
jq -e '
    .carrier_bytes == 136
    and .stats_bytes > 0
    and .context_mapped_bytes >= .context_bytes
    and .request_buffer_bytes == 128
    and .response_buffer_bytes == 512
' "$out/service_size.json" >/dev/null

trace_dir="$out/trace"
mkdir "$trace_dir"
trace_socket="$trace_dir/service.sock"
strace -ff -qq \
    -e trace=write,writev,pwrite64,sendto,sendmsg,sendmmsg \
    -s 512 -o "$trace_dir/service.trace" \
    nice -n 10 "$out/service_trace" \
    "$trace_socket" 128 correct \
    >"$trace_dir/service.stdout" \
    2>"$trace_dir/service.stderr" &
trace_pid=$!
wait_socket "$trace_socket" "$trace_pid"
"$out/controller" "$trace_socket" 8 correct \
    >"$trace_dir/result.json" 2>"$trace_dir/controller.stderr"
wait "$trace_pid"
[[ ! -s "$trace_dir/service.stdout" ]]
[[ ! -s "$trace_dir/service.stderr" ]]
[[ ! -s "$trace_dir/controller.stderr" ]]
if rg -n \
    'working|sealed|KERR INPUT|READ CARRIER|STATE DETAIL|nan|inf' \
    "$trace_dir"/service.trace*
then
    echo "CATVM Kerr service trace smuggles hidden wave state" >&2
    exit 1
fi
trace_bytes=$(
    wc -c "$trace_dir"/service.trace* \
        | awk 'END {print $1+0}'
)
[[ "$trace_bytes" -gt 0 ]]

raw_dir="$out/raw_response"
mkdir "$raw_dir"
raw_socket="$raw_dir/service.sock"
nice -n 10 "$out/service" \
    "$raw_socket" 128 correct \
    >"$raw_dir/service.stdout" \
    2>"$raw_dir/service.stderr" &
raw_service_pid=$!
wait_socket "$raw_socket" "$raw_service_pid"
"$out/raw_client" "$raw_socket" \
    >"$raw_dir/responses.txt" 2>"$raw_dir/client.stderr"
wait "$raw_service_pid"
[[ ! -s "$raw_dir/service.stdout" ]]
[[ ! -s "$raw_dir/service.stderr" ]]
[[ ! -s "$raw_dir/client.stderr" ]]
[[ $(wc -l <"$raw_dir/responses.txt") -eq 11 ]]
sed -n '1p' "$raw_dir/responses.txt" \
    | rg -q '^OK HELLO CATVM_NONLINEAR_KERR_WAVE_1 4 128 1$'
for line in 2 3 4 5 6; do
    sed -n "${line}p" "$raw_dir/responses.txt" \
        | rg -q '^ERR E_INTERMEDIATE_PROJECTION_DENIED$'
done
for line in 7 8; do
    sed -n "${line}p" "$raw_dir/responses.txt" \
        | rg -q '^ERR E_PROTOCOL$'
done
for line in 9 10; do
    sed -n "${line}p" "$raw_dir/responses.txt" \
        | rg -q \
          '^OK FINAL___ [01] [0-9]{20} [0-9a-f]{16} [+][-0-9.e+]+ [+][-0-9.e+]+ [+][-0-9.e+]+ [+][-0-9.e+]+ [0-9]{20}$'
done
sed -n '11p' "$raw_dir/responses.txt" | rg -q '^OK CLOSED$'
if rg -ni \
    'working|sealed|intermediate|kerr|phase|amplitude|complex|carrier' \
    <(sed -n '9,10p' "$raw_dir/responses.txt")
then
    echo "raw CATVM response payload smuggles hidden wave state" >&2
    exit 1
fi
raw_response_bytes=$(wc -c <"$raw_dir/responses.txt")
[[ "$raw_response_bytes" -gt 0 ]]

benchmark_cycles=1024
for observation in 1 2 3; do
    "$out/benchmark_direct" \
        128 "$benchmark_cycles" compact \
        >"$out/benchmark_compact_${observation}.json"
    "$out/benchmark_direct" \
        128 "$benchmark_cycles" phase \
        >"$out/benchmark_direct_phase_${observation}.json"
done

run_boundary_benchmark() {
    local mode=$1
    local expected=$2
    local observation=$3
    local case_dir="$out/benchmark_${mode}_${observation}"
    mkdir "$case_dir"
    local socket="$case_dir/service.sock"
    nice -n 10 "$out/service_testing" \
        "$socket" 128 "$mode" \
        >"$case_dir/service.stdout" \
        2>"$case_dir/service.stderr" &
    local service_pid=$!
    wait_socket "$socket" "$service_pid"
    nice -n 10 "$out/benchmark_client" \
        "$socket" "$benchmark_cycles" "$expected" \
        >"$case_dir/result.json" \
        2>"$case_dir/controller.stderr"
    wait "$service_pid"
    [[ ! -e "$socket" ]]
    [[ ! -s "$case_dir/service.stdout" ]]
    [[ ! -s "$case_dir/service.stderr" ]]
    [[ ! -s "$case_dir/controller.stderr" ]]
}

for observation in 1 2 3; do
    run_boundary_benchmark inert INERT___ "$observation"
    run_boundary_benchmark snapshot SNAPSHOT "$observation"
    run_boundary_benchmark correct FINAL___ "$observation"
done

for arm in compact direct_phase; do
    jq -s 'sort_by(.ns_per_transaction) | .[1]' \
        "$out"/benchmark_${arm}_*.json \
        >"$out/benchmark_${arm}_median.json"
done
for arm in inert snapshot correct; do
    jq -s 'sort_by(.ns_per_transaction) | .[1]' \
        "$out"/benchmark_${arm}_*/result.json \
        >"$out/benchmark_${arm}_median.json"
done
for benchmark in \
    benchmark_compact_median.json \
    benchmark_direct_phase_median.json \
    benchmark_inert_median.json \
    benchmark_snapshot_median.json \
    benchmark_correct_median.json
do
    jq -e '
        .result == "PASS"
        and .warmup_transactions == 32
        and .timed_transactions == 1024
        and .ns_per_transaction > 0
    ' "$out/$benchmark" >/dev/null
done
jq -n \
    --slurpfile inert "$out/benchmark_inert_median.json" \
    --slurpfile snapshot "$out/benchmark_snapshot_median.json" \
    --slurpfile correct "$out/benchmark_correct_median.json" '
    $inert[0].request_bytes_including_warmup
      == $snapshot[0].request_bytes_including_warmup
    and $inert[0].request_bytes_including_warmup
      == $correct[0].request_bytes_including_warmup
    and $inert[0].response_bytes_including_warmup
      == $snapshot[0].response_bytes_including_warmup
    and $inert[0].response_bytes_including_warmup
      == $correct[0].response_bytes_including_warmup
' | jq -e 'select(. == true)' >/dev/null

if nm -g --defined-only "$out/controller" | rg \
    'nw_|kr_|cexp|cabs|conj|complex|kerr|couple'
then
    echo "controller contains forbidden wave implementation" >&2
    exit 1
fi
if strings "$out/accepted/result.json" | rg \
    'working|sealed|intermediate_(amplitude|phase)|kerr_input'
then
    echo "CATVM result smuggles hidden wave state" >&2
    exit 1
fi

jq -n \
    --argjson trace_bytes "$trace_bytes" \
    --argjson raw_response_bytes "$raw_response_bytes" \
    --slurpfile accepted "$out/accepted/result.json" \
    --slurpfile direct "$out/direct_compare.json" \
    --slurpfile size "$out/service_size.json" \
    --slurpfile process_memory \
      "$out/accepted/process_memory.json" \
    --slurpfile compact "$out/benchmark_compact_median.json" \
    --slurpfile direct_phase \
      "$out/benchmark_direct_phase_median.json" \
    --slurpfile inert "$out/benchmark_inert_median.json" \
    --slurpfile snapshot "$out/benchmark_snapshot_median.json" \
    --slurpfile in_place "$out/benchmark_correct_median.json" '
    {
      result: "PASS",
      claim:
        "CATVM_ENFORCED_NONLINEAR_KERR_INTERFERENCE_WAVE_COMPOSITION_WITH_ACTUAL_RESTORATION_AND_REUSE",
      claim_ceiling:
        "LINUX_USERSPACE_AF_UNIX_FOUR_COMPLEX_CELL_DEPTH128_DOUBLE_COMPLEX_KERR_SU2_SOFTWARE_REFERENCE_ONLY",
      accepted: $accepted[0],
      direct_and_compact_reference: $direct[0],
      resource_law: {
        live_wave_component_bytes: 64,
        verification_seal_bytes: 64,
        carrier_descriptor_bytes: $size[0].carrier_bytes,
        service_context_logical_bytes: $size[0].context_bytes,
        service_context_page_rounded_locked_mapping_bytes:
          $size[0].context_mapped_bytes,
        warm_service_vm_rss_kib:
          $process_memory[0].warm_service_vm_rss_kib,
        warm_service_vm_lck_kib:
          $process_memory[0].warm_service_vm_lck_kib,
        request_buffer_bytes: $size[0].request_buffer_bytes,
        response_buffer_bytes: $size[0].response_buffer_bytes,
        matched_compact_semantic_state_bytes: 64,
        retained_inverse_history_bytes: 0,
        syscall_metadata_trace_bytes: $trace_bytes,
        raw_response_payload_bytes: $raw_response_bytes
      },
      warm_comparison: {
        metric:
          "MEDIAN_MONOTONIC_WALL_NS_PER_TRANSACTION_DESCRIPTIVE",
        observations_per_arm: 3,
        warmup_transactions_per_observation: 32,
        timed_transactions_per_observation: 1024,
        compact_scalar_direct_process_ns:
          $compact[0].ns_per_transaction,
        direct_process_wave_full_lifecycle_ns:
          $direct_phase[0].ns_per_transaction,
        isolated_inert_boundary_ns:
          $inert[0].ns_per_transaction,
        snapshot_catvm_ns:
          $snapshot[0].ns_per_transaction,
        in_place_catvm_ns:
          $in_place[0].ns_per_transaction,
        matched_request_bytes_per_observation:
          $inert[0].request_bytes_including_warmup,
        matched_response_bytes_per_observation:
          $inert[0].response_bytes_including_warmup
      },
      controls: {
        exact_initial_direct_compact_service_boundary_parity: true,
        deterministic_replay: true,
        matched_boundary_benchmark_traffic: true,
        process_memory_inspection_denied: true,
        intermediate_projection_denied: true,
        null_carrier_backend_and_request_denied: true,
        wrong_inverse_detected: true,
        missing_inverse_detected: true,
        reordered_noncommuting_inverse_detected: true,
        snapshot_separated_from_actual_inverse: true,
        actual_restored_carrier_reuse: true,
        analyzer: "PASS",
        asan_ubsan: "PASS",
        no_smuggle_scan: "PASS"
        ,
        raw_response_payload_scan: "PASS",
        response_parser_rejects_trailing_tokens: true
      },
      remaining_primary_obstruction:
        "EQUIVALENT_64_BYTE_FOUR_COMPLEX_VALUE_CLASSICAL_RECURRENCE",
      genuinely_distinct_phase_resource: false,
      computational_advantage: false,
      small_wall_crossed: false,
      physical_execution: false,
      unlimited_catalytic_computation: false,
      terminal: false
    }
' >"$out/summary.json"

sha256sum \
    "$phase_source" "$reference_source" \
    "$service_source" "$controller_source" "$raw_client_source" \
    "$benchmark_direct_source" \
    "$out/service" "$out/controller" \
    "$out/accepted/result.json" "$out/direct_compare.json" \
    "$out/summary.json" >"$out/SHA256SUMS"
