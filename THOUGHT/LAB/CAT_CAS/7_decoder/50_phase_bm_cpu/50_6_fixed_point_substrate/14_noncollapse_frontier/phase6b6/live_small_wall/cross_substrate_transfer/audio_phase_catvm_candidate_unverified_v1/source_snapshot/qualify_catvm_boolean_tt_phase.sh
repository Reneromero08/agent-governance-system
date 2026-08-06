#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
ulimit -c 0

if [[ $# -ne 1 ]]; then
    printf 'usage: qualify_catvm_boolean_tt_phase.sh EVIDENCE_DIR\n' >&2
    exit 2
fi

frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
evidence_dir=$1
if [[ -e "$evidence_dir" ]]; then
    printf 'evidence directory already exists: %s\n' "$evidence_dir" >&2
    exit 2
fi
mkdir -p "$evidence_dir"

service_source="$frontier_dir/catvm_boolean_tt_service.c"
controller_source="$frontier_dir/catvm_boolean_tt_controller.c"
control_source="$frontier_dir/catvm_boolean_tt_control_client.c"
phase_source="$frontier_dir/algebraic_boolean_tt_phase.c"
reference_source="$frontier_dir/algebraic_boolean_tt_reference.c"
phase_qualifier="$frontier_dir/qualify_algebraic_boolean_tt_phase.sh"
qualifier_source="$frontier_dir/qualify_catvm_boolean_tt_phase.sh"
included_source="$frontier_dir/algebraic_series_parallel_phase.c"
widths=(4 5 8 12 16)

for tool in gcc jq sha256sum cmp rg strace nm strings stat; do
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
sanitizer=(
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

gcc "${common[@]}" "$service_source" "${service_link[@]}" \
    -o "$evidence_dir/service"
gcc "${common[@]}" -DCATVM_BOOLEAN_TT_TESTING=1 \
    "$service_source" "${service_link[@]}" \
    -o "$evidence_dir/service_testing"
gcc "${common[@]}" -DCATVM_BOOLEAN_TT_TRACE_BUILD=1 \
    "$service_source" "${service_link[@]}" \
    -o "$evidence_dir/service_trace"
gcc "${common[@]}" -DCATVM_BOOLEAN_TT_SIZE_PROBE=1 \
    "$service_source" "${service_link[@]}" \
    -o "$evidence_dir/service_size_probe"
gcc "${common[@]}" "$controller_source" \
    -o "$evidence_dir/controller"
gcc "${common[@]}" "$control_source" \
    -o "$evidence_dir/control_client"

gcc "${common[@]}" -fanalyzer \
    "$service_source" "${service_link[@]}" \
    -o "$evidence_dir/service_analyzer"
gcc "${common[@]}" -fanalyzer "$controller_source" \
    -o "$evidence_dir/controller_analyzer"
gcc "${common[@]}" -fanalyzer "$control_source" \
    -o "$evidence_dir/control_analyzer"

gcc "${sanitizer[@]}" -DCATVM_SANITIZER_BUILD=1 \
    "$service_source" "${service_link[@]}" \
    -o "$evidence_dir/service_sanitized"
gcc "${sanitizer[@]}" "$controller_source" \
    -o "$evidence_dir/controller_sanitized"

wait_for_socket() {
    local socket_path=$1
    local process_id=$2
    for _ in $(seq 1 400); do
        if [[ -S "$socket_path" ]]; then
            return 0
        fi
        if ! kill -0 "$process_id" 2>/dev/null; then
            return 1
        fi
        sleep 0.01
    done
    return 1
}

run_production() {
    local name=$1
    local service=$2
    local controller=$3
    local width=$4
    local cycles=$5
    local case_dir="$evidence_dir/$name"
    mkdir "$case_dir"
    local socket_path="$case_dir/service.sock"
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        "$service" "$socket_path" "$width" \
        >"$case_dir/service.stdout" \
        2>"$case_dir/service.stderr" &
    local service_pid=$!
    wait_for_socket "$socket_path" "$service_pid"
    set +e
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        "$controller" "$socket_path" "$width" "$cycles" \
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

run_control() {
    local mode=$1
    local cycles=$2
    local case_dir="$evidence_dir/control_$mode"
    mkdir "$case_dir"
    local socket_path="$case_dir/service.sock"
    "$evidence_dir/service_testing" \
        "$socket_path" 4 "$mode" \
        >"$case_dir/service.stdout" \
        2>"$case_dir/service.stderr" &
    local service_pid=$!
    wait_for_socket "$socket_path" "$service_pid"
    set +e
    "$evidence_dir/control_client" \
        "$socket_path" "$mode" "$cycles" \
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
         and (.production_protocol_selector | not)' \
        "$case_dir/result.json" >/dev/null
}

for width in "${widths[@]}"; do
    run_production \
        "accepted_width$width" \
        "$evidence_dir/service" \
        "$evidence_dir/controller" \
        "$width" \
        32
    "$evidence_dir/service_size_probe" \
        --size-probe "$width" \
        >"$evidence_dir/size_width$width.json"
done
run_production \
    replay_width16 \
    "$evidence_dir/service" \
    "$evidence_dir/controller" \
    16 \
    32
cmp "$evidence_dir/accepted_width16/result.json" \
    "$evidence_dir/replay_width16/result.json"
run_production \
    sanitized_width4 \
    "$evidence_dir/service_sanitized" \
    "$evidence_dir/controller_sanitized" \
    4 \
    4

run_control wrong 0
run_control missing 0
run_control reordered 0
run_control snapshot 16
run_control inert 16

set +e
"$evidence_dir/service" \
    >"$evidence_dir/malformed_invocation.stdout" \
    2>"$evidence_dir/malformed_invocation.stderr"
null_rc=$?
set -e
[[ "$null_rc" -eq 2 ]]
[[ ! -s "$evidence_dir/malformed_invocation.stdout" ]]
[[ ! -s "$evidence_dir/malformed_invocation.stderr" ]]

bash "$phase_qualifier" "$evidence_dir/direct_regression" \
    >"$evidence_dir/direct_regression_path.txt"
sha256sum -c "$evidence_dir/direct_regression/SHA256SUMS" \
    >"$evidence_dir/direct_regression_sha_check.txt"

jq -s '.' \
    "$evidence_dir"/accepted_width*/result.json \
    >"$evidence_dir/accepted.json"
jq -s '.' \
    "$evidence_dir"/size_width*.json \
    >"$evidence_dir/sizes.json"

jq -e '
    sort_by(.width)
    | map(.width) == [4,5,8,12,16]
    and all(.[];
        .protocol == "CATVM_BOOLEAN_TT_PHASE_1"
        and .carrier_cells == 624 * .width - 1040
        and .leaf_cells_each == 16 * .width - 16
        and .resident_h_cells == 64 * .width - 96
        and .final_z_cells == 256 * .width - 448
        and .logical_phase_ands
            == 4 * (.resident_h_cells + .final_z_cells)
        and .logical_phase_ors
            == 2 * (.resident_h_cells + .final_z_cells)
        and .carrier_reads
            == 8 * .resident_h_cells + 11 * .final_z_cells
        and .phase_cell_updates
            == 6 * .leaf_cells_each
                + 2 * .resident_h_cells
                + 4 * .final_z_cells
        and .final_decodes == .final_z_cells
        and .carrier_creation_count == 1
        and .transactions == 34
        and .all_projection_requests_denied
        and .embedded_nul_denied
        and .oversize_packet_denied
        and .unknown_command_denied
        and .proc_mem_denied
        and .proc_maps_denied
        and .proc_fd_denied
        and .process_vm_readv_denied
        and .ptrace_denied
        and .pidfd_getfd_denied
        and .actual_inverse
        and (.snapshot_reload | not)
        and .same_service_process
        and .same_actual_restored_carrier
        and (.controller_phase_core_linked | not)
        and (.controller_computes_boundary_independently | not)
        and .primary_boundary_fnv1a64
            != .reuse_boundary_fnv1a64
    )
' "$evidence_dir/accepted.json" >"$evidence_dir/accepted_assertion.txt"

jq -n -e \
    --slurpfile accepted "$evidence_dir/accepted.json" \
    --slurpfile direct "$evidence_dir/direct_regression/native.json" '
    ($accepted[0] | sort_by(.width)) as $a
    | ($direct[0] | sort_by(.width)) as $d
    | ($a | length) == ($d | length)
    and all(range(0; $a | length);
        $a[.].width == $d[.].width
        and $a[.].primary_boundary_fnv1a64
            == $d[.].primary_boundary_fnv1a64
        and $a[.].reuse_boundary_fnv1a64
            == $d[.].reuse_boundary_fnv1a64
        and $a[.].primary_boundary_ones
            == $d[.].primary_boundary_ones
        and $a[.].reuse_boundary_ones
            == $d[.].reuse_boundary_ones
    )
' >"$evidence_dir/service_direct_parity.txt"

jq -e '
    sort_by(.width)
    | map(.width) == [4,5,8,12,16]
    and all(.[];
        .carrier_cells == 624 * .width - 1040
        and .live_carrier_bytes == 32 * .carrier_cells
        and .sealed_verification_state_bytes
            == 32 * .carrier_cells
        and .execution_snapshot_bytes
            == 32 * .carrier_cells
        and .projected_boundary_bytes
            == 256 * .width - 448
    )
' "$evidence_dir/sizes.json" >"$evidence_dir/size_assertion.txt"

nm -a "$evidence_dir/service" >"$evidence_dir/service_symbols.txt"
nm -a "$evidence_dir/controller" >"$evidence_dir/controller_symbols.txt"
nm -a "$evidence_dir/control_client" >"$evidence_dir/control_symbols.txt"
strings -a "$evidence_dir/controller" \
    >"$evidence_dir/controller_strings.txt"
rg 'btt_compose|btt_execute|symbol_product' \
    "$evidence_dir/service_symbols.txt" \
    >"$evidence_dir/service_phase_symbols.txt"
test -z "$(rg \
    'btt_compose|btt_execute|symbol_product|root3|neighbor' \
    "$evidence_dir/controller_symbols.txt" || true)"
test -z "$(rg \
    'dc5280fdcd7d3c45|5803137ff713e0bd|cfa0bb6acc66c085|4ef17937f885946d' \
    "$evidence_dir/controller_strings.txt" || true)"

trace_dir="$evidence_dir/trace_width4"
mkdir "$trace_dir"
socket_path="$trace_dir/service.sock"
strace -qq -f \
    -e trace=recvfrom,sendto,write,writev,openat,creat,rename,unlink,connect \
    -o "$trace_dir/service.trace" \
    "$evidence_dir/service_trace" "$socket_path" 4 \
    >"$trace_dir/service.stdout" \
    2>"$trace_dir/service.stderr" &
trace_pid=$!
wait_for_socket "$socket_path" "$trace_pid"
"$evidence_dir/controller" "$socket_path" 4 8 \
    >"$trace_dir/result.json" \
    2>"$trace_dir/controller.stderr"
wait "$trace_pid"
[[ ! -s "$trace_dir/service.stdout" ]]
[[ ! -s "$trace_dir/service.stderr" ]]
[[ ! -s "$trace_dir/controller.stderr" ]]
awk '
    /unlink\(.*\)[[:space:]]*=[[:space:]]*0/ { seen=1; next }
    seen { print }
' "$trace_dir/service.trace" >"$trace_dir/post_custody.trace"
test -s "$trace_dir/post_custody.trace"
test "$(rg -c 'recvfrom\(' "$trace_dir/post_custody.trace")" -gt 0
test "$(rg -c 'sendto\(' "$trace_dir/post_custody.trace")" -gt 0
test -z "$(rg \
    'write\(|writev\(|openat\(|creat\(|rename\(|connect\(' \
    "$trace_dir/post_custody.trace" || true)"

if rg -n \
    'boundary_fnv1a64.*(dc5280|cfa0bb)|expected_boundary|answer_table|truth_table|witness_list|candidate_set' \
    "$controller_source" \
    >"$evidence_dir/controller_forbidden_static.txt"; then
    printf 'controller contains an answer-bearing evaluator or table\n' >&2
    exit 1
fi
: >"$evidence_dir/controller_forbidden_static.txt"

sha256sum \
    "$service_source" \
    "$controller_source" \
    "$control_source" \
    "$phase_source" \
    "$reference_source" \
    "$included_source" \
    "$phase_qualifier" \
    "$qualifier_source" \
    "$evidence_dir/service" \
    "$evidence_dir/controller" \
    "$evidence_dir/control_client" \
    "$evidence_dir/accepted.json" \
    "$evidence_dir/sizes.json" \
    "$trace_dir/post_custody.trace" \
    "$evidence_dir/direct_regression/SHA256SUMS" \
    "$evidence_dir/direct_regression/result.json" \
    >"$evidence_dir/CORE_SHA256SUMS"

service_binary_bytes=$(stat -c%s "$evidence_dir/service")
controller_binary_bytes=$(stat -c%s "$evidence_dir/controller")
testing_service_binary_bytes=$(
    stat -c%s "$evidence_dir/service_testing"
)
direct_phase_binary_bytes=$(
    stat -c%s "$evidence_dir/direct_regression/algebraic_boolean_tt_phase"
)

jq -n \
    --slurpfile accepted "$evidence_dir/accepted.json" \
    --slurpfile sizes "$evidence_dir/sizes.json" \
    --arg evidence_dir "$evidence_dir" \
    --arg service_sha "$(sha256sum "$service_source" | cut -d' ' -f1)" \
    --arg controller_sha "$(sha256sum "$controller_source" | cut -d' ' -f1)" \
    --arg control_sha "$(sha256sum "$control_source" | cut -d' ' -f1)" \
    --arg phase_sha "$(sha256sum "$phase_source" | cut -d' ' -f1)" \
    --arg reference_sha "$(sha256sum "$reference_source" | cut -d' ' -f1)" \
    --arg qualifier_sha "$(sha256sum "$qualifier_source" | cut -d' ' -f1)" \
    --argjson service_binary_bytes "$service_binary_bytes" \
    --argjson controller_binary_bytes "$controller_binary_bytes" \
    --argjson testing_service_binary_bytes \
        "$testing_service_binary_bytes" \
    --argjson direct_phase_binary_bytes \
        "$direct_phase_binary_bytes" '
    {
        result: "PASS",
        claim:
            "CATVM_ENFORCED_WIDTH_PARAMETRIC_BOOLEAN_TT_RESIDENT_H_"
            + "COMPOSITION_ESTABLISHED_ON_PHASE_BACKEND",
        claim_ceiling:
            "BOUNDED_LINUX_SAME_UID_SOFTWARE_WIDTHS4_5_8_12_16_"
            + "BOOLEAN_TT_RANK2_TO_RANK4_TO_RANK8_ATOMIC_"
            + "TRANSACTION_REFERENCE_ONLY",
        evidence_directory: $evidence_dir,
        protocol: "CATVM_BOOLEAN_TT_PHASE_1",
        widths: ($accepted[0] | map(.width) | sort),
        service_process_per_width: 1,
        carrier_creation_count_per_service: 1,
        transactions_per_width: 34,
        controller_phase_core_linked: false,
        controller_computes_boundary_independently: false,
        actual_resident_h_consumed_by_z: true,
        intermediate_h_decoded_cells: 0,
        intermediate_h_serialized_cells: 0,
        intermediate_h_block_materializations: 0,
        intermediate_h_operand_reads_counted_in_carrier_reads: true,
        final_boundary_receipt_only: true,
        actual_inverse: true,
        actual_restored_carrier_reuse: true,
        snapshot_separate_generation_zero: true,
        same_uid_process_inspection_controls_denied: 6,
        post_custody_trace_bytes:
            ("'"$(stat -c %s "$trace_dir/post_custody.trace")"'" | tonumber),
        service_stdout_bytes: 0,
        service_stderr_bytes: 0,
        exact_direct_backend_parity: true,
        controls: {
            wrong_inverse: "PASS_DETECTED",
            missing_inverse: "PASS_DETECTED",
            reordered_noncommuting_inverse: "PASS_DETECTED",
            snapshot: "PASS_SEPARATE_WEAKER_PATH",
            inert_carrier_disabled:
                "PASS_NO_CATALYTIC_RESULT_TEST_BUILD_ONLY",
            malformed_invocation: "PASS_REJECTED_BEFORE_CARRIER_CREATION",
            intermediate_projection: "PASS_REJECTED",
            embedded_nul: "PASS_REJECTED",
            oversize_packet: "PASS_REJECTED",
            unknown_command: "PASS_REJECTED",
            analyzer: "PASS",
            asan_ubsan: "PASS",
            deterministic_replay: "PASS",
            no_smuggle: "PASS_NONVACUOUS_POST_CUSTODY_TRACE"
        },
        resource_accounting_scope:
            "PHASE_PAYLOAD_ARRAYS_BUFFERS_TRAFFIC_AND_OPERATION_LAWS_"
            + "REFERENCE_ONLY_NOT_COMPREHENSIVE_PROCESS_RSS",
        resources: ($sizes[0] | sort_by(.width)),
        resource_comparison: {
            accepted_qualifier_traffic:
                ($accepted[0]
                 | sort_by(.width)
                 | map({
                    width,
                    transactions,
                    request_packets,
                    response_packets,
                    request_bytes,
                    response_bytes
                 })),
            binaries: {
                service_bytes: $service_binary_bytes,
                controller_bytes: $controller_binary_bytes,
                testing_service_bytes: $testing_service_binary_bytes,
                direct_phase_bytes: $direct_phase_binary_bytes
            },
            warm_direct_process_phase:
                "SAME_ACCEPTED_PER_TRANSACTION_PHASE_OPERATION_LAW_"
                + "WITHOUT_IPC_34_ACTUAL_SAME_CARRIER_TRANSACTIONS",
            warm_isolated_inert_boundary:
                "ZERO_PHASE_OPERATIONS_NO_FINAL_BOUNDARY_TEST_BUILD_ONLY",
            snapshot_catvm:
                "FORWARD_ONLY_PHASE_OPERATION_LAW_PLUS_WORKING_CARRIER_"
                + "RELOAD_GENERATION_ZERO_TEST_BUILD_ONLY",
            accepted_in_place_catvm:
                "FORWARD_PLUS_ACTUAL_INVERSE_PHASE_OPERATION_LAW_"
                + "GENERATION_ADVANCES"
        },
        fixed_rank_unbounded_depth_closure: false,
        native_rank_minimization_or_recompression: false,
        computational_advantage: false,
        small_wall_crossed: false,
        physical_execution: false,
        unlimited_catalytic_computation: false,
        sha256: {
            service_source: $service_sha,
            controller_source: $controller_sha,
            control_source: $control_sha,
            phase_source: $phase_sha,
            reference_source: $reference_sha,
            qualifier: $qualifier_sha
        },
        terminal: false
    }
' >"$evidence_dir/result.json"

find "$evidence_dir" -type f \
    ! -name SHA256SUMS \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >"$evidence_dir/SHA256SUMS"

printf '%s\n' "$evidence_dir"
