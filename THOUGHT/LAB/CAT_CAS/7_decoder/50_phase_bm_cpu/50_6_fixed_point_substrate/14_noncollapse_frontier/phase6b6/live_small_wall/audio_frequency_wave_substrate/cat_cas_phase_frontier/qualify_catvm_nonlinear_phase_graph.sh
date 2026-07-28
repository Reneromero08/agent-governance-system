#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_catvm_nonlinear_phase_graph.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$1
mkdir -p "$out"
service_source="$here/catvm_nonlinear_graph_service.c"
controller_source="$here/catvm_nonlinear_graph_controller.c"
control_source="$here/catvm_nonlinear_graph_control_client.c"
benchmark_client_source="$here/catvm_nonlinear_graph_benchmark_client.c"
benchmark_direct_source="$here/nonlinear_phase_graph_warm_benchmark.c"
phase_source="$here/topology_nonlinear_phase_graph.c"
reference_source="$here/topology_nonlinear_phase_graph_reference.c"
topology="$here/nonlinear_phase_graph_topology.txt"

for tool in gcc jq cmp rg sha256sum nm strings nice; do
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
gcc "${common[@]}" -DCATVM_NPG_TESTING=1 \
    "$service_source" "${service_link[@]}" \
    -o "$out/service_testing"
gcc "${common[@]}" -DCATVM_NPG_TRACE_BUILD=1 \
    "$service_source" "${service_link[@]}" \
    -o "$out/service_trace"
gcc "${common[@]}" -DCATVM_NPG_SIZE_PROBE=1 \
    "$service_source" "${service_link[@]}" \
    -o "$out/service_size"
gcc "${common[@]}" "$controller_source" -lm \
    -o "$out/controller"
gcc "${common[@]}" "$control_source" \
    -o "$out/control_client"
gcc "${common[@]}" "$benchmark_client_source" \
    -o "$out/benchmark_client"
gcc "${common[@]}" "$benchmark_direct_source" -lm \
    -o "$out/benchmark_direct"
gcc "${common[@]}" "$phase_source" -lm \
    -o "$out/direct_phase"
gcc "${common[@]}" "$reference_source" -lm \
    -o "$out/reference"
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

run_production() {
    local name=$1
    local service=$2
    local controller=$3
    local cycles=$4
    local case_dir="$out/$name"
    mkdir "$case_dir"
    local socket="$case_dir/service.sock"
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        nice -n 10 "$service" "$socket" "$topology" 128 \
        >"$case_dir/service.stdout" \
        2>"$case_dir/service.stderr" &
    local service_pid=$!
    local ready=0
    for _ in $(seq 1 400); do
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
    set +e
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        nice -n 10 "$controller" "$socket" "$cycles" \
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

run_control() {
    local mode=$1
    local case_dir="$out/control_$mode"
    mkdir "$case_dir"
    local socket="$case_dir/service.sock"
    nice -n 10 "$out/service_testing" \
        "$socket" "$topology" 128 "$mode" \
        >"$case_dir/service.stdout" \
        2>"$case_dir/service.stderr" &
    local service_pid=$!
    local ready=0
    for _ in $(seq 1 400); do
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
    set +e
    "$out/control_client" "$socket" "$mode" \
        >"$case_dir/result.json" \
        2>"$case_dir/controller.stderr"
    local client_rc=$?
    wait "$service_pid"
    local service_rc=$?
    set -e
    [[ "$client_rc" -eq 0 ]]
    [[ "$service_rc" -eq 0 ]]
    [[ ! -e "$socket" ]]
    [[ ! -s "$case_dir/service.stdout" ]]
    [[ ! -s "$case_dir/service.stderr" ]]
    [[ ! -s "$case_dir/controller.stderr" ]]
    jq -e \
        --arg mode "$mode" '
        .result == "PASS"
        and .control == $mode
        and (.production_protocol_selector | not)
    ' "$case_dir/result.json" >/dev/null
}

run_production accepted "$out/service" "$out/controller" 256
run_production accepted_replay "$out/service" "$out/controller" 256
run_production \
    sanitized "$out/service_sanitized" "$out/controller_sanitized" 16
cmp "$out/accepted/result.json" "$out/accepted_replay/result.json"
for mode in missing wrong reordered snapshot; do
    run_control "$mode"
done

jq -e '
    .result == "PASS"
    and .protocol == "CATVM_NONLINEAR_PHASE_GRAPH_1"
    and .width == 4
    and .edges == 6
    and .rounds == 128
    and .transactions == 256
    and .carrier_creation_count == 1
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
    and .primary.program == 0
    and .primary.generation == 1
    and .reuse.program == 1
    and .reuse.generation == 2
    and .primary.boundary_fnv1a64 != .reuse.boundary_fnv1a64
' "$out/accepted/result.json" >/dev/null

"$out/direct_phase" "$topology" 128 >"$out/direct.json"
"$out/reference" "$topology" 128 >"$out/reference.json"
jq -e \
    --slurpfile service "$out/accepted/result.json" \
    --slurpfile reference "$out/reference.json" '
    .primary_boundary_fnv1a64
      == $service[0].primary.boundary_fnv1a64
    and .reuse_boundary_fnv1a64
      == $service[0].reuse.boundary_fnv1a64
    and .primary_boundary_fnv1a64
      == $reference[0].primary_boundary_fnv1a64
    and .reuse_boundary_fnv1a64
      == $reference[0].reuse_boundary_fnv1a64
' "$out/direct.json" >/dev/null

"$out/service_size" --size-probe >"$out/service_size.json"

benchmark_cycles=1024
for observation in 1 2 3; do
    "$out/benchmark_direct" \
        "$topology" 128 "$benchmark_cycles" compact \
        >"$out/benchmark_compact_${observation}.json"
    "$out/benchmark_direct" \
        "$topology" 128 "$benchmark_cycles" phase \
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
        "$socket" "$topology" 128 "$mode" \
        >"$case_dir/service.stdout" \
        2>"$case_dir/service.stderr" &
    local service_pid=$!
    local ready=0
    for _ in $(seq 1 400); do
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

jq -s 'sort_by(.ns_per_transaction) | .[1]' \
    "$out"/benchmark_compact_*.json \
    >"$out/benchmark_compact_median.json"
jq -s 'sort_by(.ns_per_transaction) | .[1]' \
    "$out"/benchmark_direct_phase_*.json \
    >"$out/benchmark_direct_phase_median.json"
jq -s 'sort_by(.ns_per_transaction) | .[1]' \
    "$out"/benchmark_inert_*/result.json \
    >"$out/benchmark_inert_median.json"
jq -s 'sort_by(.ns_per_transaction) | .[1]' \
    "$out"/benchmark_snapshot_*/result.json \
    >"$out/benchmark_snapshot_median.json"
jq -s 'sort_by(.ns_per_transaction) | .[1]' \
    "$out"/benchmark_correct_*/result.json \
    >"$out/benchmark_in_place_median.json"

jq -e '
    .result == "PASS"
    and .warmup_transactions == 32
    and .timed_transactions == 1024
' "$out/benchmark_compact_median.json" >/dev/null
for benchmark in \
    benchmark_direct_phase_median.json \
    benchmark_inert_median.json \
    benchmark_snapshot_median.json \
    benchmark_in_place_median.json
do
    jq -e '
        .result == "PASS"
        and .warmup_transactions == 32
        and .timed_transactions == 1024
    ' "$out/$benchmark" >/dev/null
done
jq -n \
    --slurpfile inert "$out/benchmark_inert_median.json" \
    --slurpfile snapshot "$out/benchmark_snapshot_median.json" \
    --slurpfile in_place "$out/benchmark_in_place_median.json" '
    $inert[0].request_bytes_including_warmup
      == $snapshot[0].request_bytes_including_warmup
    and $inert[0].request_bytes_including_warmup
      == $in_place[0].request_bytes_including_warmup
    and $inert[0].response_bytes_including_warmup
      == $snapshot[0].response_bytes_including_warmup
    and $inert[0].response_bytes_including_warmup
      == $in_place[0].response_bytes_including_warmup
' | jq -e 'select(. == true)' >/dev/null

if nm -a "$out/controller" | rg \
    'relative|multiply_cell|npg_|carrier|initial_angle|shear'
then
    echo "controller contains forbidden phase implementation" >&2
    exit 1
fi
if strings "$out/accepted/result.json" | rg \
    'resident_phase|source_phase|working\[|baseline\[|forward_factor'
then
    echo "result smuggles hidden phase state" >&2
    exit 1
fi
for file in \
    "$out/accepted/service.stdout" \
    "$out/accepted/service.stderr" \
    "$out/accepted/controller.stderr"
do
    [[ ! -s "$file" ]]
done

jq -n \
    --slurpfile service "$out/accepted/result.json" \
    --slurpfile direct "$out/direct.json" \
    --slurpfile reference "$out/reference.json" \
    --slurpfile size "$out/service_size.json" \
    --slurpfile compact_benchmark \
      "$out/benchmark_compact_median.json" \
    --slurpfile direct_phase_benchmark \
      "$out/benchmark_direct_phase_median.json" \
    --slurpfile inert_benchmark \
      "$out/benchmark_inert_median.json" \
    --slurpfile snapshot_benchmark \
      "$out/benchmark_snapshot_median.json" \
    --slurpfile in_place_benchmark \
      "$out/benchmark_in_place_median.json" '
    {
      result: "PASS",
      claim:
        "BOUNDED_CATVM_TOPOLOGY_COMPILED_SHARED_NONLINEAR_UNIT_PHASE_GRAPH_WITH_ACTUAL_RESTORATION_AND_REUSE",
      claim_ceiling:
        "LINUX_USERSPACE_AF_UNIX_WIDTH4_EDGES6_ROUNDS128_DOUBLE_COMPLEX_PHASE_REFERENCE_ONLY",
      accepted: $service[0],
      direct_phase: $direct[0],
      compact_classical_reference: $reference[0],
      resource_law: {
        carrier_cells: $direct[0].carrier_cells,
        live_carrier_bytes: $direct[0].live_carrier_bytes,
        verification_snapshot_bytes: $direct[0].live_carrier_bytes,
        accepted_path_peak_carrier_and_snapshot_bytes:
          (2 * $direct[0].live_carrier_bytes),
        snapshot_creation_bytes:
          $direct[0].live_carrier_bytes,
        snapshot_reload_working_bytes:
          ($direct[0].live_carrier_bytes / 2),
        compiled_topology_bytes:
          $direct[0].compiled_topology_bytes,
        service_context_bytes: $size[0].context_bytes,
        service_context_page_rounded_locked_mapping_bytes:
          $size[0].context_mapped_bytes,
        service_request_buffer_bytes:
          $size[0].request_buffer_bytes,
        service_response_buffer_bytes:
          $size[0].response_buffer_bytes,
        execution_summary_bytes:
          $size[0].execution_summary_bytes,
        final_projection_bytes: $size[0].projection_bytes,
        best_matched_classical_state_bytes:
          $reference[0].classical_state_bytes,
        accepted_controller_request_packets:
          $service[0].request_packets,
        accepted_controller_response_packets:
          $service[0].response_packets,
        accepted_controller_request_bytes:
          $service[0].request_bytes,
        accepted_controller_response_bytes:
          $service[0].response_bytes,
        native_phase_shears_per_transaction:
          $direct[0].forward_shears,
        inverse_phase_shears_per_transaction:
          $direct[0].inverse_shears,
        seal_and_unseal_phase_updates_per_transaction:
          (2 * $direct[0].width),
        boundary_attach_and_detach_updates_per_transaction: 4,
        final_phase_decodes_per_transaction:
          $direct[0].final_phase_decodes,
        total_native_phase_updates_per_transaction:
          $direct[0].native_phase_updates,
        resident_phase_reads_per_transaction:
          $direct[0].resident_source_reads,
        restoration_and_integrity_complex_cells_verified:
          (2 * $direct[0].carrier_cells),
        unit_modulus_complex_cell_checks_per_transaction:
          ((2 * $direct[0].forward_shears + 4)
            * $direct[0].carrier_cells)
      },
      warm_comparison: {
        observations_per_arm: 3,
        warmup_transactions_per_observation: 32,
        timed_transactions_per_observation: 1024,
        metric:
          "MEDIAN_MONOTONIC_WALL_NS_PER_TRANSACTION_DESCRIPTIVE",
        compact_classical_direct_process:
          $compact_benchmark[0],
        direct_process_phase:
          $direct_phase_benchmark[0],
        isolated_inert_boundary:
          $inert_benchmark[0],
        snapshot_catvm:
          $snapshot_benchmark[0],
        in_place_catvm:
          $in_place_benchmark[0]
      },
      controls: {
        exact_direct_and_compact_reference_parity: true,
        matched_boundary_benchmark_traffic: true,
        deterministic_replay: true,
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
      },
      phase_owned_advance:
        "MACHINE_ENFORCED_SHARED_NONLINEAR_RESIDENT_PHASE_GRAPH",
      remaining_primary_obstruction:
        "EQUIVALENT_WIDTH_DOUBLE_ANGLE_GRAPH_RECURRENCE",
      genuinely_distinct_phase_resource: false,
      computational_advantage: false,
      small_wall_crossed: false,
      physical_execution: false,
      unlimited_catalytic_computation: false,
      terminal: false
    }
' >"$out/summary.json"

sha256sum \
    "$service_source" "$controller_source" "$control_source" \
    "$phase_source" "$reference_source" "$topology" \
    "$benchmark_client_source" "$benchmark_direct_source" \
    "$out/service" "$out/controller" \
    "$out/accepted/result.json" "$out/direct.json" \
    "$out/reference.json" "$out/service_size.json" \
    "$out/benchmark_compact_median.json" \
    "$out/benchmark_direct_phase_median.json" \
    "$out/benchmark_inert_median.json" \
    "$out/benchmark_snapshot_median.json" \
    "$out/benchmark_in_place_median.json" \
    "$out/summary.json" >"$out/SHA256SUMS"
