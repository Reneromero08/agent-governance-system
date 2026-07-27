#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_small_wall_boolean_tt_quotient.sh EVIDENCE_DIR" >&2
    exit 2
fi

frontier_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
evidence_dir=$1
mkdir -p "$evidence_dir"

service_source="$frontier_dir/small_wall_boolean_tt_quotient_service.c"
controller_source="$frontier_dir/small_wall_boolean_tt_quotient_controller.c"
probe_source="$frontier_dir/small_wall_boolean_tt_quotient_probe.c"
phase_source="$frontier_dir/algebraic_boolean_tt_suffix_quotient_phase.c"
reference_source="$frontier_dir/algebraic_boolean_tt_suffix_quotient_reference.c"

for tool in \
    gcc jq sha256sum cmp rg strace nm strings stat taskset nice /usr/bin/time
do
    command -v "$tool" >/dev/null
done

affinity_spec=$(taskset -pc $$ | sed 's/.*: //')
first_affinity_segment=${affinity_spec%%,*}
benchmark_cpu=${first_affinity_segment%%-*}
[[ "$benchmark_cpu" =~ ^[0-9]+$ ]]

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

declare -A arm_number=(
    [baseline]=1
    [snapshot]=2
    [in_place]=3
)
arms=(baseline snapshot in_place)
cases=("4:4" "5:5" "8:8" "12:8" "16:8")

for arm in "${arms[@]}"; do
    number=${arm_number[$arm]}
    gcc "${common[@]}" -DQTW_ARM="$number" \
        "$service_source" "${service_link[@]}" \
        -o "$evidence_dir/service_$arm"
    gcc "${common[@]}" -DQTW_ARM="$number" -DQTW_TRACE_BUILD=1 \
        "$service_source" "${service_link[@]}" \
        -o "$evidence_dir/service_${arm}_trace"
    gcc "${common[@]}" -DQTW_ARM="$number" -DQTW_SIZE_PROBE=1 \
        "$service_source" "${service_link[@]}" \
        -o "$evidence_dir/service_${arm}_size"
    gcc "${common[@]}" -fanalyzer -DQTW_ARM="$number" \
        "$service_source" "${service_link[@]}" \
        -o "$evidence_dir/service_${arm}_analyzer"
    gcc "${common[@]}" -fstack-usage -DQTW_ARM="$number" \
        "$service_source" "${service_link[@]}" \
        -o "$evidence_dir/service_${arm}_stack"
    gcc -std=c11 -O1 -g \
        -Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow \
        -fsanitize=address,undefined \
        -fno-omit-frame-pointer -fno-sanitize-recover=all \
        -DQTW_ARM="$number" -DQTW_SANITIZER_BUILD=1 \
        "$service_source" "${service_link[@]}" \
        -o "$evidence_dir/service_${arm}_sanitized"
done

gcc "${common[@]}" "$controller_source" \
    -o "$evidence_dir/controller"
gcc "${common[@]}" -fanalyzer "$controller_source" \
    -o "$evidence_dir/controller_analyzer"
gcc -std=c11 -O1 -g \
    -Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow \
    -fsanitize=address,undefined \
    -fno-omit-frame-pointer -fno-sanitize-recover=all \
    "$controller_source" \
    -o "$evidence_dir/controller_sanitized"
gcc "${common[@]}" "$probe_source" \
    -o "$evidence_dir/probe"
gcc "${common[@]}" "$reference_source" \
    -o "$evidence_dir/reference"

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

run_controller_case() {
    local name=$1
    local arm=$2
    local service=$3
    local controller=$4
    local width=$5
    local depth=$6
    local warm=$7
    local transactions=$8
    local case_dir="$evidence_dir/$name"
    mkdir "$case_dir"
    local socket_path="$case_dir/service.sock"
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        /usr/bin/time -f '%M' -o "$case_dir/maxrss_kib.txt" \
        taskset -c "$benchmark_cpu" nice -n 5 \
        "$service" "$socket_path" "$width" "$depth" \
        >"$case_dir/service.stdout" \
        2>"$case_dir/service.stderr" &
    local service_pid=$!
    wait_for_socket "$socket_path" "$service_pid"
    set +e
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        taskset -c "$benchmark_cpu" nice -n 5 \
        "$controller" \
        "$socket_path" "$width" "$depth" "$warm" "$transactions" \
        >"$case_dir/raw.json" \
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
    local maxrss_kib
    maxrss_kib=$(<"$case_dir/maxrss_kib.txt")
    [[ "$maxrss_kib" =~ ^[0-9]+$ ]]
    jq \
        --arg arm "$arm" \
        --argjson maximum_resident_set_kib "$maxrss_kib" \
        '. + {
          arm: $arm,
          maximum_resident_set_kib: $maximum_resident_set_kib
        }' \
        "$case_dir/raw.json" >"$case_dir/result.json"
    jq -e '.result == "PASS"' "$case_dir/result.json" >/dev/null
}

run_probe_case() {
    local arm=$1
    local case_dir="$evidence_dir/probe_$arm"
    mkdir "$case_dir"
    local socket_path="$case_dir/service.sock"
    "$evidence_dir/service_$arm" "$socket_path" 16 8 \
        >"$case_dir/service.stdout" \
        2>"$case_dir/service.stderr" &
    local service_pid=$!
    wait_for_socket "$socket_path" "$service_pid"
    "$evidence_dir/probe" "$socket_path" 16 8 \
        >"$case_dir/result.json" \
        2>"$case_dir/probe.stderr"
    wait "$service_pid"
    [[ ! -e "$socket_path" ]]
    [[ ! -s "$case_dir/service.stdout" ]]
    [[ ! -s "$case_dir/service.stderr" ]]
    [[ ! -s "$case_dir/probe.stderr" ]]
    jq -e '
        .result == "PASS"
        and .intermediate_projection_denied
        and .carrier_read_denied
        and .debug_dump_denied
        and .embedded_nul_denied
        and .oversize_packet_denied
        and .unknown_family_denied
        and .null_carrier_request_denied
        and .final_boundary_receipt_only
    ' "$case_dir/result.json" >/dev/null
}

mkdir "$evidence_dir/semantic"
for case_spec in "${cases[@]}"; do
    width=${case_spec%%:*}
    depth=${case_spec#*:}
    label="w${width}_d${depth}"
    "$evidence_dir/reference" "$width" "$depth" \
        >"$evidence_dir/semantic/reference_$label.json"
    jq -e '.result == "PASS"' \
        "$evidence_dir/semantic/reference_$label.json" >/dev/null
    for arm in "${arms[@]}"; do
        "$evidence_dir/service_${arm}_size" \
            --size-probe "$width" "$depth" \
            >"$evidence_dir/semantic/size_${label}_$arm.json"
        run_controller_case \
            "semantic/${label}_$arm" \
            "$arm" \
            "$evidence_dir/service_$arm" \
            "$evidence_dir/controller" \
            "$width" "$depth" 4 4
    done
    jq -s '
        (map(.receipts) | unique | length) == 1
        and (map(.timed_request_packets) | unique | length) == 1
        and (map(.timed_response_packets) | unique | length) == 1
        and (map(.timed_request_bytes) | unique | length) == 1
        and (map(.timed_response_bytes) | unique | length) == 1
    ' \
        "$evidence_dir/semantic/${label}_baseline/result.json" \
        "$evidence_dir/semantic/${label}_snapshot/result.json" \
        "$evidence_dir/semantic/${label}_in_place/result.json" \
        | jq -e '. == true' >/dev/null
    jq -e \
        --slurpfile reference \
            "$evidence_dir/semantic/reference_$label.json" '
        .receipts[0].boundary_fnv1a64
            == $reference[0].and_boundary_fnv1a64
        and .receipts[1].boundary_fnv1a64
            == $reference[0].or_boundary_fnv1a64
        and .receipts[0].ones == $reference[0].and_boundary_ones
        and .receipts[1].ones == $reference[0].or_boundary_ones
        and .receipts[0].cells == $reference[0].final_quotient_cells
        and .receipts[1].cells == $reference[0].final_quotient_cells
    ' "$evidence_dir/semantic/${label}_baseline/result.json" >/dev/null

    baseline="$evidence_dir/semantic/${label}_baseline/result.json"
    snapshot="$evidence_dir/semantic/${label}_snapshot/result.json"
    in_place="$evidence_dir/semantic/${label}_in_place/result.json"
    size_baseline="$evidence_dir/semantic/size_${label}_baseline.json"
    size_snapshot="$evidence_dir/semantic/size_${label}_snapshot.json"
    size_in_place="$evidence_dir/semantic/size_${label}_in_place.json"
    jq -e \
        --slurpfile size "$size_baseline" '
        .direct_cells == ($size[0].final_cells * .timed_transactions)
        and .direct_predicates == .direct_cells
        and .logical_phase_ands == 0
        and .logical_phase_ors == 0
        and .carrier_reads == 0
        and .phase_cell_updates == 0
        and .final_decodes == 0
        and .projected_bytes == .direct_cells
        and .comparison_snapshot_bytes == 0
        and .snapshot_loads == 0
        and .actual_inverse_transactions == 0
        and .carrier_creation_count == 0
        and .carrier_seal_cpu_ns > 0
    ' "$baseline" >/dev/null
    jq -e \
        --slurpfile size "$size_snapshot" '
        .direct_cells == 0
        and .logical_phase_ands > 0
        and .logical_phase_ors > 0
        and .carrier_reads > 0
        and .phase_cell_updates > 0
        and .quotient_member_terms == .logical_phase_ands
        and .final_decodes
            == ($size[0].final_cells * .timed_transactions)
        and .projected_bytes == .final_decodes
        and .comparison_snapshot_bytes
            == (
              $size[0].comparison_snapshot_bytes_per_transaction
              * .timed_transactions
            )
        and .restoration_scan_cells
            == ($size[0].carrier_cells * .timed_transactions)
        and .snapshot_loads == .timed_transactions
        and .snapshot_reload_bytes
            == (
              $size[0].snapshot_reload_bytes_per_transaction
              * .timed_transactions
            )
        and .actual_inverse_transactions == 0
        and .restoration_generation == 0
        and .carrier_creation_count == 1
        and .carrier_seal_cpu_ns > 0
    ' "$snapshot" >/dev/null
    jq -e \
        --slurpfile size "$size_in_place" \
        --slurpfile sham "$snapshot" '
        .direct_cells == 0
        and .logical_phase_ands
            == (2 * $sham[0].logical_phase_ands)
        and .logical_phase_ors
            == (2 * $sham[0].logical_phase_ors)
        and .phase_cell_updates
            == (2 * $sham[0].phase_cell_updates)
        and .quotient_member_terms
            == (2 * $sham[0].quotient_member_terms)
        and .carrier_reads
            == (
              2 * $sham[0].carrier_reads
              - $sham[0].final_decodes
            )
        and .final_decodes
            == ($size[0].final_cells * .timed_transactions)
        and .projected_bytes == .final_decodes
        and .comparison_snapshot_bytes
            == (
              $size[0].comparison_snapshot_bytes_per_transaction
              * .timed_transactions
            )
        and .restoration_scan_cells
            == ($size[0].carrier_cells * .timed_transactions)
        and .snapshot_loads == 0
        and .snapshot_reload_bytes == 0
        and .actual_inverse_transactions == .timed_transactions
        and .restoration_generation
            == (.warm_transactions + .timed_transactions)
        and .carrier_creation_count == 1
        and .carrier_seal_cpu_ns > 0
        and .maximum_restoration_error <= 2e-12
    ' "$in_place" >/dev/null
    jq -e '
        .arm == 1
        and .carrier_creation_count == 0
        and .live_carrier_bytes == 0
        and .sealed_verification_state_bytes == 0
        and .comparison_snapshot_bytes_per_transaction == 0
        and .snapshot_reload_bytes_per_transaction == 0
        and .rematerialized_cells_per_transaction == 0
    ' "$size_baseline" >/dev/null
    jq -e '
        .arm == 2
        and .carrier_creation_count == 1
        and .live_carrier_bytes
            == (32 * .carrier_cells)
        and .sealed_verification_state_bytes
            == .live_carrier_bytes
        and .comparison_snapshot_bytes_per_transaction
            == .live_carrier_bytes
        and .snapshot_reload_bytes_per_transaction
            == (16 * .carrier_cells)
        and .rematerialized_cells_per_transaction == 0
    ' "$size_snapshot" >/dev/null
    jq -e '
        .arm == 3
        and .carrier_creation_count == 1
        and .live_carrier_bytes
            == (32 * .carrier_cells)
        and .sealed_verification_state_bytes
            == .live_carrier_bytes
        and .comparison_snapshot_bytes_per_transaction
            == .live_carrier_bytes
        and .snapshot_reload_bytes_per_transaction == 0
        and .rematerialized_cells_per_transaction == 0
    ' "$size_in_place" >/dev/null
done

for arm in "${arms[@]}"; do
    run_probe_case "$arm"
    run_controller_case \
        "sanitized_$arm" \
        "$arm" \
        "$evidence_dir/service_${arm}_sanitized" \
        "$evidence_dir/controller_sanitized" \
        16 8 4 4
done

if nm -a "$evidence_dir/service_baseline" \
    | awk '{print $NF}' \
    | rg -q \
        '^(qtt_|root3|symbol_product|multiply_cell|make_carrier|restoration)$'
then
    echo "compact baseline links phase machinery" >&2
    exit 1
fi
for arm in snapshot in_place; do
    if nm -a "$evidence_dir/service_$arm" \
        | awk '{print $NF}' \
        | rg -q '^qtw_direct_(cell|receipt)$'
    then
        echo "phase arm links the direct quotient generator" >&2
        exit 1
    fi
done
if nm -a "$evidence_dir/controller" \
    | awk '{print $NF}' \
    | rg -q 'qtt_|qtw_direct|root3|symbol_product|carrier'
then
    echo "comparison controller links a computation engine" >&2
    exit 1
fi
if rg -n \
    'expected.*(hash|boundary|receipt)|answer.*(table|vector)|qtw_direct_cell' \
    "$controller_source"
then
    echo "comparison controller contains answer-bearing material" >&2
    exit 1
fi
direct_function=$(
    sed -n \
        '/static struct qtw_receipt qtw_direct_receipt/,/^}/p' \
        "$service_source"
)
if rg -q \
    'static.*\[|malloc|calloc|realloc|memcpy|cache|lookup|truth|candidate|witness' \
    <<<"$direct_function"
then
    echo "compact baseline contains cached or extensional state" >&2
    exit 1
fi
if rg -n \
    'answer_table|expected_(hash|boundary|receipt)|\[[[:space:]]*3548[[:space:]]*\]' \
    "$service_source" "$controller_source"
then
    echo "comparison contains a finite answer vector" >&2
    exit 1
fi

mkdir "$evidence_dir/stack"
for arm in "${arms[@]}"; do
    stack_file=$(
        find "$evidence_dir" -maxdepth 1 \
            -name "service_${arm}_stack-*.su" -print -quit
    )
    [[ -n "$stack_file" ]]
    max_frame=$(
        awk -F '\t' '$2 > maximum {maximum=$2} END {print maximum+0}' \
            "$stack_file"
    )
    [[ "$max_frame" =~ ^[0-9]+$ ]]
    jq -n \
        --arg arm "$arm" \
        --argjson compiler_max_single_frame_bytes "$max_frame" \
        '{
          arm: $arm,
          compiler_max_single_frame_bytes:
            $compiler_max_single_frame_bytes
        }' >"$evidence_dir/stack/$arm.json"
done

for arm in "${arms[@]}"; do
    trace_dir="$evidence_dir/trace_$arm"
    mkdir "$trace_dir"
    trace_socket="$trace_dir/service.sock"
    taskset -c "$benchmark_cpu" \
        strace -qq -f -s 256 \
        -e trace=%file,%network,%ipc,%memory,write,writev,pwrite64,process_vm_writev,ptrace,ioctl,clock_gettime \
        -o "$trace_dir/service.strace" \
        "$evidence_dir/service_${arm}_trace" \
        "$trace_socket" 16 8 \
        >"$trace_dir/service.stdout" \
        2>"$trace_dir/service.stderr" &
    trace_pid=$!
    wait_for_socket "$trace_socket" "$trace_pid"
    taskset -c "$benchmark_cpu" \
        "$evidence_dir/controller" \
        "$trace_socket" 16 8 4 4 \
        >"$trace_dir/result.json" \
        2>"$trace_dir/controller.stderr"
    wait "$trace_pid"
    [[ ! -s "$trace_dir/service.stdout" ]]
    [[ ! -s "$trace_dir/service.stderr" ]]
    [[ ! -s "$trace_dir/controller.stderr" ]]
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
        echo "comparison arm exposed a forbidden post-custody channel" >&2
        exit 1
    fi
done

for arm in "${arms[@]}"; do
    set +e
    "$evidence_dir/service_$arm" \
        >"$evidence_dir/null_${arm}.stdout" \
        2>"$evidence_dir/null_${arm}.stderr"
    null_rc=$?
    set -e
    [[ "$null_rc" -eq 2 ]]
    [[ ! -s "$evidence_dir/null_${arm}.stdout" ]]
    [[ ! -s "$evidence_dir/null_${arm}.stderr" ]]
done

mkdir "$evidence_dir/runs"
orders=(baseline snapshot in_place in_place snapshot baseline)
for case_spec in "${cases[@]}"; do
    width=${case_spec%%:*}
    depth=${case_spec#*:}
    ordinal=0
    for arm in "${orders[@]}"; do
        name="measure_w${width}_d${depth}_${ordinal}_$arm"
        run_controller_case \
            "$name" \
            "$arm" \
            "$evidence_dir/service_$arm" \
            "$evidence_dir/controller" \
            "$width" "$depth" 32 128
        jq \
            --arg arm "$arm" \
            --argjson ordinal "$ordinal" \
            '. + {arm: $arm, ordinal: $ordinal}' \
            "$evidence_dir/$name/result.json" \
            >"$evidence_dir/runs/w${width}_d${depth}_${ordinal}_$arm.json"
        ordinal=$((ordinal + 1))
    done
done
jq -s 'sort_by(.width, .depth, .ordinal)' \
    "$evidence_dir"/runs/*.json \
    >"$evidence_dir/runs.json"
jq -s 'sort_by(.width, .depth, .arm)' \
    "$evidence_dir"/semantic/size_*.json \
    >"$evidence_dir/sizes.json"
jq -s 'sort_by(.width, .depth)' \
    "$evidence_dir"/semantic/reference_*.json \
    >"$evidence_dir/references.json"
jq -s 'sort_by(.arm)' \
    "$evidence_dir"/stack/*.json \
    >"$evidence_dir/stacks.json"

jq -e '
    length == 30
    and all(.[];
      .result == "PASS"
      and .warm_transactions == 32
      and .timed_transactions == 128
      and .same_public_palindrome_schedule
      and (.controller_computes_boundary | not)
      and .timed_request_packets == 128
      and .timed_response_packets == 128
      and .timed_request_bytes == 1152
    )
    and all(
      group_by([.width,.depth])[];
      (map(.receipts) | unique | length) == 1
      and (map(.timed_request_packets) | unique | length) == 1
      and (map(.timed_response_packets) | unique | length) == 1
      and (map(.timed_request_bytes) | unique | length) == 1
      and (map(.timed_response_bytes) | unique | length) == 1
    )
' "$evidence_dir/runs.json" >/dev/null

jq -n \
    --arg claim \
        "GROWING_NONAFFINE_BOOLEAN_TT_SUFFIX_QUOTIENT_SMALL_WALL_TRIAD_CONFIRMS_COMPACT_RECURRENCE_AND_RETAINED_HISTORY_OBSTRUCTION" \
    --arg ceiling \
        "BOUNDED_LINUX_WARM_WIDTHS4_5_8_12_16_DEPTHS4_5_8_HOMOGENEOUS_NEIGHBOR_AND_OR_REFERENCE_ONLY" \
    --argjson benchmark_cpu "$benchmark_cpu" \
    --slurpfile raw "$evidence_dir/runs.json" \
    --slurpfile sizes "$evidence_dir/sizes.json" \
    --slurpfile references "$evidence_dir/references.json" \
    --slurpfile stacks "$evidence_dir/stacks.json" '
    def mean(values): (values | add / length);
    def rows($w;$d;$arm):
      [$raw[0][]
       | select(.width==$w and .depth==$d and .arm==$arm)];
    def arm_summary($w;$d;$arm):
      (rows($w;$d;$arm)) as $rows
      | {
          arm: $arm,
          wall_ns_mean: mean([$rows[].wall_ns]),
          service_cpu_ns_mean: mean([$rows[].service_cpu_ns]),
          maximum_resident_set_kib_mean:
            mean([$rows[].maximum_resident_set_kib]),
          counters: ($rows[0]
            | del(
                .result,.protocol,.width,.depth,.warm_transactions,
                .timed_transactions,.wall_ns,.service_cpu_ns,
                .timed_request_packets,.timed_response_packets,
                .timed_request_bytes,.timed_response_bytes,
                .receipts,.same_public_palindrome_schedule,
                .controller_computes_boundary,.arm,.ordinal,
                .maximum_resident_set_kib
              ))
        };
    [4,5,8,12,16] as $widths
    | [4,5,8,8,8] as $depths
    | {
      result: "PASS",
      claim: $claim,
      claim_ceiling: $ceiling,
      adjudication:
        "COMPACT_PUBLIC_RECURRENCE_IS_PRIMARY_RESOURCE_OBSTRUCTION_RETAINED_HISTORY_IS_SECONDARY_PHASE_MACHINE_DEFECT",
      benchmark_cpu: $benchmark_cpu,
      process_nice: 5,
      protocol: "MATCHED_PERSISTENT_AF_UNIX_SOCK_SEQPACKET",
      warm_transactions_per_observation: 32,
      timed_transactions_per_observation: 128,
      arm_order:
        ["baseline","snapshot","in_place",
         "in_place","snapshot","baseline"],
      public_family_schedule: ["AND","OR","OR","AND"],
      cases: [
        range(0;5) as $index
        | $widths[$index] as $w
        | $depths[$index] as $d
        | ($sizes[0][]
            | select(.width==$w and .depth==$d and .arm==3)
          ) as $phase_size
        | {
            width: $w,
            depth: $d,
            final_cells: $phase_size.final_cells,
            retained_stage_cells: $phase_size.retained_stage_cells,
            predecessor_inverse_history_cells:
              $phase_size.predecessor_inverse_history_cells,
            carrier_cells: $phase_size.carrier_cells,
            receipts: (rows($w;$d;"baseline")[0].receipts),
            arms: [
              arm_summary($w;$d;"baseline"),
              arm_summary($w;$d;"snapshot"),
              arm_summary($w;$d;"in_place")
            ],
            baseline_to_snapshot_service_cpu_ratio:
              (
                arm_summary($w;$d;"snapshot").service_cpu_ns_mean
                / arm_summary($w;$d;"baseline").service_cpu_ns_mean
              ),
            baseline_to_in_place_service_cpu_ratio:
              (
                arm_summary($w;$d;"in_place").service_cpu_ns_mean
                / arm_summary($w;$d;"baseline").service_cpu_ns_mean
              )
          }
      ],
      sizes: $sizes[0],
      offline_raw_product_verification: $references[0],
      compiler_stack_frames: $stacks[0],
      controls: {
        matched_receipts: true,
        matched_protocol_traffic: true,
        baseline_binary_has_no_phase_engine: true,
        phase_binaries_have_no_direct_generator: true,
        controller_has_no_result_evaluator: true,
        no_intermediate_projection: true,
        no_post_custody_smuggle_channel_under_trace: true,
        snapshot_generation_zero: true,
        actual_inverse_and_reuse: true,
        carrier_seal_cpu_exposed: true,
        null_carrier_request_rejected: true,
        analyzer: "PASS",
        asan_ubsan: "PASS"
      },
      obstruction: {
        rank_growth: "LINEAR_MAXIMUM_RANK_DEPTH_PLUS_ONE",
        retained_history:
          "SUM_OF_ALL_STAGES_ADDS_ONE_DEPTH_FACTOR",
        quotient_generation:
          "PUBLIC_THRESHOLD_RECURRENCE_EMITS_FINAL_CORE_DIRECTLY",
        projection: "LINEAR_IN_FINAL_QUOTIENT_CELLS",
        restoration:
          "IN_PLACE_REPEATS_NATIVE_FORWARD_TRANSFORM_WORK_IN_REVERSE",
        snapshot:
          "WORKING_STATE_RELOAD_IS_16_BYTES_PER_CARRIER_CELL",
        machine_boundary:
          "MATCHED_AND_NONDOMINANT_RELATIVE_TO_PHASE_KERNEL",
        distinct_phase_resource:
          "NOT_PRESENT_IN_THIS_BOOLEAN_ROOT_LOCKED_FAMILY"
      },
      selected_phase_owned_repair:
        "TOPOLOGY_DERIVED_REVERSIBLE_QUOTIENT_STAGE_PEBBLING_WITH_ACTUAL_SLOT_REUSE",
      family_exhausted_for_advantage_after_repair: true,
      computational_advantage: false,
      small_wall_crossed: false,
      physical_execution: false,
      unlimited_catalytic_computation: false,
      terminal: false
    }
' >"$evidence_dir/summary.json"

jq -e '
    .result == "PASS"
    and (.cases | length) == 5
    and all(.cases[];
      .baseline_to_snapshot_service_cpu_ratio > 1
      and .baseline_to_in_place_service_cpu_ratio > 1
      and (.arms | length) == 3
    )
    and .controls.matched_receipts
    and .controls.matched_protocol_traffic
    and .controls.carrier_seal_cpu_exposed
    and (.computational_advantage | not)
    and (.small_wall_crossed | not)
    and (.terminal | not)
' "$evidence_dir/summary.json" >/dev/null

sha256sum \
    "$service_source" \
    "$controller_source" \
    "$probe_source" \
    "$phase_source" \
    "$reference_source" \
    "$evidence_dir/service_baseline" \
    "$evidence_dir/service_snapshot" \
    "$evidence_dir/service_in_place" \
    "$evidence_dir/controller" \
    "$evidence_dir/runs.json" \
    "$evidence_dir/sizes.json" \
    "$evidence_dir/references.json" \
    "$evidence_dir/summary.json" \
    >"$evidence_dir/SHA256SUMS"

printf '%s\n' \
    "GROWING_NONAFFINE_BOOLEAN_TT_SUFFIX_QUOTIENT_SMALL_WALL_TRIAD_PASS"
