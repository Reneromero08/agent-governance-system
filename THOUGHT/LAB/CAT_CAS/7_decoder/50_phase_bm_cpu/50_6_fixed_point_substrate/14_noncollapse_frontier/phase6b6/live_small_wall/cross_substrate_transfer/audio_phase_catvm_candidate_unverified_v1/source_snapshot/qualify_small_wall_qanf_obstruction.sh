#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_small_wall_qanf_obstruction.sh EVIDENCE_DIR" >&2
    exit 2
fi

frontier_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
evidence_dir=$1
mkdir -p "$evidence_dir"

service_source="$frontier_dir/small_wall_qanf_service.c"
controller_source="$frontier_dir/small_wall_qanf_controller.c"
probe_source="$frontier_dir/small_wall_qanf_probe.c"
phase_source="$frontier_dir/quadratic_anf_chain_phase.c"
reference_source="$frontier_dir/quadratic_anf_chain_reference.c"
catvm_qualifier="$frontier_dir/qualify_catvm_qanf_phase.sh"
primary_fixture="$frontier_dir/quadratic_anf_chain_primary.qanf"
reuse_fixture="$frontier_dir/quadratic_anf_chain_reuse.qanf"
degree2_fixture="$frontier_dir/quadratic_anf_chain_degree2_nonaffine.qanf"
sham_fixture="$frontier_dir/quadratic_anf_chain_affine_sham.qanf"

for tool in gcc jq sha256sum cmp rg strace nm strings stat taskset nice; do
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

for arm in "${arms[@]}"; do
    number=${arm_number[$arm]}
    gcc "${common[@]}" -DQSW_ARM="$number" \
        "$service_source" "${service_link[@]}" \
        -o "$evidence_dir/service_$arm"
    gcc "${common[@]}" -DQSW_ARM="$number" -DQSW_TRACE_BUILD=1 \
        "$service_source" "${service_link[@]}" \
        -o "$evidence_dir/service_${arm}_trace"
    gcc "${common[@]}" -DQSW_ARM="$number" -DQSW_SIZE_PROBE=1 \
        "$service_source" "${service_link[@]}" \
        -o "$evidence_dir/service_${arm}_size"
    gcc "${common[@]}" -fanalyzer -DQSW_ARM="$number" \
        "$service_source" "${service_link[@]}" \
        -o "$evidence_dir/service_${arm}_analyzer"
    gcc "${common[@]}" -fstack-usage -DQSW_ARM="$number" \
        "$service_source" "${service_link[@]}" \
        -o "$evidence_dir/service_${arm}_stack"
    gcc -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror \
        -Wconversion -Wshadow \
        -fsanitize=address,undefined \
        -fno-omit-frame-pointer -fno-sanitize-recover=all \
        -DQSW_ARM="$number" -DQSW_SANITIZER_BUILD=1 \
        "$service_source" "${service_link[@]}" \
        -o "$evidence_dir/service_${arm}_sanitized"
done

gcc "${common[@]}" "$controller_source" \
    -o "$evidence_dir/controller"
gcc "${common[@]}" -fanalyzer "$controller_source" \
    -o "$evidence_dir/controller_analyzer"
gcc -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror \
    -Wconversion -Wshadow \
    -fsanitize=address,undefined \
    -fno-omit-frame-pointer -fno-sanitize-recover=all \
    "$controller_source" \
    -o "$evidence_dir/controller_sanitized"
gcc "${common[@]}" "$probe_source" \
    -o "$evidence_dir/probe"
gcc "${common[@]}" "$reference_source" \
    -o "$evidence_dir/reference"

fixtures=(
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

run_controller_case() {
    local name=$1
    local service=$2
    local controller=$3
    local warm=$4
    local transactions=$5
    local case_dir="$evidence_dir/$name"
    mkdir "$case_dir"
    local socket_path="$case_dir/service.sock"
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        taskset -c "$benchmark_cpu" nice -n 5 \
        "$service" "$socket_path" "${fixtures[@]}" \
        >"$case_dir/service.stdout" \
        2>"$case_dir/service.stderr" &
    local service_pid=$!
    wait_for_socket "$socket_path" "$service_pid"
    set +e
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        taskset -c "$benchmark_cpu" nice -n 5 \
        "$controller" "$socket_path" "$warm" "$transactions" \
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

run_probe_case() {
    local arm=$1
    local case_dir="$evidence_dir/probe_$arm"
    mkdir "$case_dir"
    local socket_path="$case_dir/service.sock"
    "$evidence_dir/service_$arm" \
        "$socket_path" "${fixtures[@]}" \
        >"$case_dir/service.stdout" \
        2>"$case_dir/service.stderr" &
    local service_pid=$!
    wait_for_socket "$socket_path" "$service_pid"
    "$evidence_dir/probe" "$socket_path" \
        >"$case_dir/result.json" \
        2>"$case_dir/probe.stderr"
    wait "$service_pid"
    [[ ! -e "$socket_path" ]]
    [[ ! -s "$case_dir/service.stdout" ]]
    [[ ! -s "$case_dir/service.stderr" ]]
    [[ ! -s "$case_dir/probe.stderr" ]]
    jq -e '
        .result == "PASS"
        and .intermediate_projection_denied == true
        and .final_boundary_only == true
    ' "$case_dir/result.json" >/dev/null
}

for arm in "${arms[@]}"; do
    run_controller_case \
        "semantic_$arm" \
        "$evidence_dir/service_$arm" \
        "$evidence_dir/controller" \
        8 8
    run_probe_case "$arm"
    run_controller_case \
        "sanitized_$arm" \
        "$evidence_dir/service_${arm}_sanitized" \
        "$evidence_dir/controller_sanitized" \
        8 8
    cmp \
        <(jq -S 'del(.wall_ns,.service_cpu_ns,.seal_cpu_ns)' \
            "$evidence_dir/semantic_$arm/result.json") \
        <(jq -S 'del(.wall_ns,.service_cpu_ns,.seal_cpu_ns)' \
            "$evidence_dir/sanitized_$arm/result.json")
done

reference_files=()
for item in \
    "primary:$primary_fixture" \
    "reuse:$reuse_fixture" \
    "degree2:$degree2_fixture" \
    "sham:$sham_fixture"
do
    label=${item%%:*}
    fixture=${item#*:}
    "$evidence_dir/reference" "$fixture" \
        >"$evidence_dir/reference_$label.json"
    reference_files+=("$evidence_dir/reference_$label.json")
done
jq -s '[.[].boundary_coefficients]' \
    "${reference_files[@]}" \
    >"$evidence_dir/reference_boundaries.json"

for arm in "${arms[@]}"; do
    jq -e \
        --slurpfile reference "$evidence_dir/reference_boundaries.json" '
        .boundaries == $reference[0]
        and .timed_request_packets == 8
        and .timed_response_packets == 8
        and .timed_request_bytes == 72
        and .timed_response_bytes == 176
    ' "$evidence_dir/semantic_$arm/result.json" >/dev/null
done

jq -s '
    (map(.boundaries) | unique | length) == 1
    and (map(.timed_request_packets) | unique | length) == 1
    and (map(.timed_response_packets) | unique | length) == 1
    and (map(.timed_request_bytes) | unique | length) == 1
    and (map(.timed_response_bytes) | unique | length) == 1
' \
    "$evidence_dir/semantic_baseline/result.json" \
    "$evidence_dir/semantic_snapshot/result.json" \
    "$evidence_dir/semantic_in_place/result.json" >/dev/null

jq -e '
    .boolean_ands == 32
    and .phase_products == 0
    and .carrier_creation_count == 0
' "$evidence_dir/semantic_baseline/result.json" >/dev/null
jq -e '
    .boolean_ands == 0
    and .phase_products == 72
    and .carrier_reads == 408
    and .phase_cell_updates == 184
    and .final_decodes == 40
    and .snapshot_loads == 8
    and .snapshot_reload_bytes == 2944
    and .actual_inverse_transactions == 0
    and .restoration_generation == 0
    and .carrier_creation_count == 1
' "$evidence_dir/semantic_snapshot/result.json" >/dev/null
jq -e '
    .boolean_ands == 0
    and .phase_products == 144
    and .carrier_reads == 776
    and .phase_cell_updates == 368
    and .final_decodes == 40
    and .snapshot_loads == 0
    and .actual_inverse_transactions == 8
    and .restoration_generation == 16
    and .carrier_creation_count == 1
' "$evidence_dir/semantic_in_place/result.json" >/dev/null

if nm -a "$evidence_dir/service_baseline" \
    | awk '{print $NF}' \
    | rg -q '^qb_|qsw_reviewed_standalone_main'
then
    echo "compact baseline links phase machinery" >&2
    exit 1
fi
for arm in snapshot in_place; do
    if nm -a "$evidence_dir/service_$arm" \
        | rg -q \
            'qsw_reviewed_standalone_main|qb_print_boundary|qsw_compact_boundary'
    then
        echo "phase arm retains report or compact fallback machinery" >&2
        exit 1
    fi
done
if nm -a "$evidence_dir/controller" \
    | awk '{print $NF}' \
    | rg -q '^qb_|qsw_compact_boundary|phase_|carrier'
then
    echo "comparison controller links a computation engine" >&2
    exit 1
fi
if rg -n \
    'theta.*gamma|theta.*delta|expected.*(boundary|coefficient)|answer.*(table|vector)' \
    "$controller_source"
then
    echo "comparison controller contains a result evaluator" >&2
    exit 1
fi
compact_function=$(
    sed -n \
        '/static struct qsw_boundary qsw_compact_boundary/,/^}/p' \
        "$service_source"
)
if rg -q \
    'static.*\[|malloc|calloc|realloc|memcpy|cache|lookup|truth|candidate|witness' \
    <<<"$compact_function"
then
    echo "compact baseline function contains cached or extensional state" >&2
    exit 1
fi
[[ "$(rg -n '\bqsw_compact_boundary\(' "$service_source" | wc -l)" -eq 2 ]]
if rg -n \
    '\[[[:space:]]*(64|320)[[:space:]]*\]|1[UuLl]*[[:space:]]*<<[[:space:]]*6|answer_table|expected_boundary' \
    "$service_source" "$controller_source"
then
    echo "comparison contains a finite answer vector or lookup table" >&2
    exit 1
fi

for arm in "${arms[@]}"; do
    "$evidence_dir/service_${arm}_size" --size-probe \
        >"$evidence_dir/size_$arm.json"
    stack_file=$(
        find "$evidence_dir" -maxdepth 1 \
            -name "service_${arm}_stack-*.su" -print -quit
    )
    [[ -n "$stack_file" ]]
    max_frame=$(
        awk -F '\t' '
            $1 !~ /qsw_reviewed_standalone_main$/ && $2 > maximum {
                maximum=$2
            }
            END {print maximum+0}
        ' "$stack_file"
    )
    parser_frame=$(
        awk -F '\t' '$1 ~ /:qsw_read_program$/ {print $2}' \
            "$stack_file"
    )
    serve_frame=$(
        awk -F '\t' '$1 ~ /:qsw_serve(\.isra)?$/ {print $2}' \
            "$stack_file"
    )
    execute_frame=$(
        awk -F '\t' '$1 ~ /:qb_execute$/ {print $2}' \
            "$stack_file"
    )
    [[ "$max_frame" =~ ^[0-9]+$ ]]
    [[ "$parser_frame" =~ ^[0-9]+$ ]]
    [[ "$serve_frame" =~ ^[0-9]+$ ]]
    if [[ "$arm" == baseline ]]; then
        execute_frame=0
    fi
    [[ "$execute_frame" =~ ^[0-9]+$ ]]
    jq -n \
        --argjson compiler_max_single_frame_bytes "$max_frame" \
        --argjson public_program_parser_frame_bytes "$parser_frame" \
        --argjson service_loop_frame_bytes "$serve_frame" \
        --argjson phase_execute_frame_bytes "$execute_frame" \
        '{
          compiler_max_single_frame_bytes:
            $compiler_max_single_frame_bytes,
          public_program_parser_frame_bytes:
            $public_program_parser_frame_bytes,
          service_loop_frame_bytes: $service_loop_frame_bytes,
          phase_execute_frame_bytes: $phase_execute_frame_bytes
        }' >"$evidence_dir/stack_$arm.json"
done
jq -e '
    .arm == 1
    and .mapped_context_bytes >= .context_bytes
    and .program_table_bytes == .sealed_program_table_bytes
    and .carrier_creation_count == 0
    and .live_carrier_bytes == 0
    and .execution_snapshot_bytes == 0
' "$evidence_dir/size_baseline.json" >/dev/null
jq -e '
    .arm == 2
    and .mapped_context_bytes >= .context_bytes
    and .program_table_bytes == .sealed_program_table_bytes
    and .carrier_creation_count == 1
    and .live_carrier_bytes == 736
    and .sealed_carrier_bytes == 736
    and .execution_snapshot_bytes == 736
    and .working_reload_bytes == 368
' "$evidence_dir/size_snapshot.json" >/dev/null
jq -e '
    .arm == 3
    and .mapped_context_bytes >= .context_bytes
    and .program_table_bytes == .sealed_program_table_bytes
    and .carrier_creation_count == 1
    and .live_carrier_bytes == 736
    and .sealed_carrier_bytes == 736
    and .execution_snapshot_bytes == 736
    and .working_reload_bytes == 0
' "$evidence_dir/size_in_place.json" >/dev/null

for arm in "${arms[@]}"; do
    trace_dir="$evidence_dir/trace_$arm"
    mkdir "$trace_dir"
    trace_socket="$trace_dir/service.sock"
    taskset -c "$benchmark_cpu" strace -qq -f -s 256 \
        -e trace=%file,%network,%ipc,%memory,write,writev,pwrite64,process_vm_writev,ptrace,ioctl,clock_gettime \
        -o "$trace_dir/service.strace" \
        "$evidence_dir/service_${arm}_trace" \
        "$trace_socket" "${fixtures[@]}" \
        >"$trace_dir/service.stdout" \
        2>"$trace_dir/service.stderr" &
    trace_pid=$!
    wait_for_socket "$trace_socket" "$trace_pid"
    taskset -c "$benchmark_cpu" \
        "$evidence_dir/controller" "$trace_socket" 8 8 \
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
batches=(1024 4096 16384)
for batch in "${batches[@]}"; do
    ordinal=0
    for arm in "${orders[@]}"; do
        run_controller_case \
            "measure_${batch}_${ordinal}_$arm" \
            "$evidence_dir/service_$arm" \
            "$evidence_dir/controller" \
            1024 "$batch"
        jq \
            --arg arm "$arm" \
            --argjson batch "$batch" \
            --argjson ordinal "$ordinal" \
            '. + {arm: $arm, batch: $batch, ordinal: $ordinal}' \
            "$evidence_dir/measure_${batch}_${ordinal}_$arm/result.json" \
            >"$evidence_dir/runs/${batch}_${ordinal}_$arm.json"
        ordinal=$((ordinal + 1))
    done
done
jq -s 'sort_by(.batch, .ordinal)' \
    "$evidence_dir"/runs/*.json \
    >"$evidence_dir/runs.json"

jq -e \
    --slurpfile reference "$evidence_dir/reference_boundaries.json" '
    length == 18
    and all(.[];
      .result == "PASS"
      and .warm_transactions == 1024
      and .timed_transactions == .batch
      and .boundaries == $reference[0]
      and .same_public_palindrome_schedule == true
      and .controller_computes_boundary == false
      and .timed_request_packets == .batch
      and .timed_response_packets == .batch
      and .timed_request_bytes == (9 * .batch)
      and .timed_response_bytes == (22 * .batch)
      and (
        if .arm == "baseline" then
          .boolean_ands == (4 * .batch)
          and .phase_products == 0
          and .carrier_creation_count == 0
          and .restoration_generation == 0
        elif .arm == "snapshot" then
          .boolean_ands == 0
          and .phase_products == (9 * .batch)
          and .carrier_reads == (51 * .batch)
          and .phase_cell_updates == (23 * .batch)
          and .final_decodes == (5 * .batch)
          and .snapshot_loads == .batch
          and .snapshot_reload_bytes == (368 * .batch)
          and .actual_inverse_transactions == 0
          and .restoration_generation == 0
          and .carrier_creation_count == 1
        else
          .arm == "in_place"
          and .boolean_ands == 0
          and .phase_products == (18 * .batch)
          and .carrier_reads == (97 * .batch)
          and .phase_cell_updates == (46 * .batch)
          and .final_decodes == (5 * .batch)
          and .snapshot_loads == 0
          and .snapshot_reload_bytes == 0
          and .actual_inverse_transactions == .batch
          and .restoration_generation == (1024 + .batch)
          and .carrier_creation_count == 1
        end
      )
    )
    and all(
      group_by(.batch)[];
      (map(.boundaries) | unique | length) == 1
      and (map(.timed_request_bytes) | unique | length) == 1
      and (map(.timed_response_bytes) | unique | length) == 1
    )
' "$evidence_dir/runs.json" >/dev/null

baseline_bytes=$(stat -c%s "$evidence_dir/service_baseline")
snapshot_bytes=$(stat -c%s "$evidence_dir/service_snapshot")
in_place_bytes=$(stat -c%s "$evidence_dir/service_in_place")
controller_bytes=$(stat -c%s "$evidence_dir/controller")

jq -n \
    --arg claim \
        "FIXED_SCHEMA_QANF_COMPACT_BASELINE_OBSTRUCTION_CONFIRMED_UNDER_MATCHED_CATVM_BOUNDARY" \
    --arg ceiling \
        "BOUNDED_LINUX_WARM_FIXED_SCHEMA_QANF_BASELINE_SNAPSHOT_IN_PLACE_RESOURCE_COMPARISON_REFERENCE_ONLY" \
    --argjson benchmark_cpu "$benchmark_cpu" \
    --argjson baseline_bytes "$baseline_bytes" \
    --argjson snapshot_bytes "$snapshot_bytes" \
    --argjson in_place_bytes "$in_place_bytes" \
    --argjson controller_bytes "$controller_bytes" \
    --slurpfile raw "$evidence_dir/runs.json" \
    --slurpfile baseline_size "$evidence_dir/size_baseline.json" \
    --slurpfile snapshot_size "$evidence_dir/size_snapshot.json" \
    --slurpfile in_place_size "$evidence_dir/size_in_place.json" \
    --slurpfile baseline_stack "$evidence_dir/stack_baseline.json" \
    --slurpfile snapshot_stack "$evidence_dir/stack_snapshot.json" \
    --slurpfile in_place_stack "$evidence_dir/stack_in_place.json" '
    def mean(values): (values | add / length);
    def arm_mean($batch; $arm; $field):
      mean([
        $raw[0][]
        | select(.batch == $batch and .arm == $arm)
        | .[$field]
      ]);
    [1024,4096,16384] as $batches
    | {
      result: "PASS",
      claim: $claim,
      claim_ceiling: $ceiling,
      adjudication: "FIXED_SCHEMA_COMPACT_BASELINE_OBSTRUCTION_CONFIRMED",
      protocol: "MATCHED_PERSISTENT_AF_UNIX_SOCK_SEQPACKET",
      benchmark_cpu: $benchmark_cpu,
      process_nice: 5,
      warm_transactions_per_run: 1024,
      timed_batches: $batches,
      fixed_arm_order_per_batch:
        ["baseline","snapshot","in_place","in_place","snapshot","baseline"],
      public_palindrome_schedule:
        [0,3,2,1,1,2,3,0],
      matched_final_boundaries: true,
      matched_timed_request_response_bytes: true,
      compact_baseline: {
        formula:
          "[1,eta,theta*gamma,theta*delta*alpha,theta*delta*beta]",
        ordinary_boolean_ands_per_transaction: 4,
        result_cache: false,
        lookup_table: false,
        carrier: false
      },
      snapshot_sham: {
        phase_products_per_transaction: 9,
        carrier_reads_per_transaction: 51,
        phase_cell_updates_per_transaction: 23,
        final_decodes_per_transaction: 5,
        snapshot_reloads_per_transaction: 1,
        working_reload_bytes_per_transaction: 368,
        actual_inverse: false
      },
      in_place_catvm: {
        phase_products_per_transaction: 18,
        carrier_reads_per_transaction: 97,
        phase_cell_updates_per_transaction: 46,
        final_decodes_per_transaction: 5,
        snapshot_reloads_per_transaction: 0,
        actual_inverse: true,
        actual_restored_carrier_reuse: true
      },
      analytic_fixed_schema_capacity: {
        public_coefficient_bits: 6,
        possible_public_programs: 64,
        final_boundary_bits_per_program: 5,
        raw_answer_vector_upper_bound_bits_not_materialized: 320,
        constant_size_compact_formula_exists: true
      },
      aggregates: [
        $batches[] as $batch
        | {
          batch: $batch,
          baseline_wall_ns_mean:
            arm_mean($batch; "baseline"; "wall_ns"),
          snapshot_wall_ns_mean:
            arm_mean($batch; "snapshot"; "wall_ns"),
          in_place_wall_ns_mean:
            arm_mean($batch; "in_place"; "wall_ns"),
          baseline_service_cpu_ns_mean:
            arm_mean($batch; "baseline"; "service_cpu_ns"),
          snapshot_service_cpu_ns_mean:
            arm_mean($batch; "snapshot"; "service_cpu_ns"),
          in_place_service_cpu_ns_mean:
            arm_mean($batch; "in_place"; "service_cpu_ns"),
          rho_snapshot_over_in_place:
            (
              arm_mean($batch; "snapshot"; "wall_ns")
              / arm_mean($batch; "in_place"; "wall_ns")
            ),
          rho_compact_over_in_place:
            (
              arm_mean($batch; "baseline"; "wall_ns")
              / arm_mean($batch; "in_place"; "wall_ns")
            )
        }
      ],
      resources: {
        baseline:
          ($baseline_size[0] + {stack: $baseline_stack[0]}),
        snapshot:
          ($snapshot_size[0] + {stack: $snapshot_stack[0]}),
        in_place:
          ($in_place_size[0] + {stack: $in_place_stack[0]}),
        binary_bytes: {
          baseline_service: $baseline_bytes,
          snapshot_service: $snapshot_bytes,
          in_place_service: $in_place_bytes,
          controller: $controller_bytes
        }
      },
      controls: {
        all_four_public_fixture_boundaries_match_reference: "PASS",
        affine_and_degree2_applicability_controls: "PASS",
        same_protocol_and_execution_traffic: "PASS",
        attempted_intermediate_projection: "PASS_REJECTED_ALL_ARMS",
        null_carrier: "PASS_REJECTED_ALL_ARMS",
        embedded_nul_oversize_unknown: "PASS_ALL_ARMS",
        analyzer: "PASS_ALL_ARMS",
        sanitizer: "PASS_ALL_ARMS",
        post_custody_no_smuggle_trace: "PASS_ALL_ARMS",
        catvm_predecessor_regression: "PASS"
      },
      batch_repetition_is_problem_scaling: false,
      mission_gamma_measured: false,
      capacity_separation: false,
      bounded_constant_factor_catvm_advantage: false,
      performance_advantage: false,
      total_memory_advantage: false,
      c5_fixed_point_advantage: false,
      small_wall_crossed: false,
      arbitrary_non_affine_closure: false,
      physical_execution: false,
      unlimited_catalytic_computation: false,
      terminal: false
    }
' >"$evidence_dir/result.json"

bash "$catvm_qualifier" "$evidence_dir/catvm_regression" \
    >"$evidence_dir/catvm_regression.summary.txt"
jq -e '
    .result == "PASS"
    and .adjudication ==
      "FIXED_SCHEMA_COMPACT_BASELINE_OBSTRUCTION_CONFIRMED"
    and .matched_final_boundaries == true
    and .matched_timed_request_response_bytes == true
    and .compact_baseline.ordinary_boolean_ands_per_transaction == 4
    and .in_place_catvm.actual_inverse == true
    and .in_place_catvm.actual_restored_carrier_reuse == true
    and .mission_gamma_measured == false
    and .c5_fixed_point_advantage == false
    and .small_wall_crossed == false
' "$evidence_dir/result.json" >/dev/null

sha256sum \
    "$service_source" \
    "$controller_source" \
    "$probe_source" \
    "$frontier_dir/qualify_small_wall_qanf_obstruction.sh" \
    "$catvm_qualifier" \
    "$phase_source" \
    "$reference_source" \
    "${fixtures[@]}" \
    "$evidence_dir/service_baseline" \
    "$evidence_dir/service_snapshot" \
    "$evidence_dir/service_in_place" \
    "$evidence_dir/service_baseline_trace" \
    "$evidence_dir/service_snapshot_trace" \
    "$evidence_dir/service_in_place_trace" \
    "$evidence_dir/service_baseline_sanitized" \
    "$evidence_dir/service_snapshot_sanitized" \
    "$evidence_dir/service_in_place_sanitized" \
    "$evidence_dir/service_baseline_analyzer" \
    "$evidence_dir/service_snapshot_analyzer" \
    "$evidence_dir/service_in_place_analyzer" \
    "$evidence_dir/service_baseline_stack" \
    "$evidence_dir/service_snapshot_stack" \
    "$evidence_dir/service_in_place_stack" \
    "$evidence_dir/controller" \
    "$evidence_dir/controller_sanitized" \
    "$evidence_dir/controller_analyzer" \
    "$evidence_dir/probe" \
    "$evidence_dir/reference" \
    "$evidence_dir/semantic_baseline/result.json" \
    "$evidence_dir/semantic_snapshot/result.json" \
    "$evidence_dir/semantic_in_place/result.json" \
    "$evidence_dir/sanitized_baseline/result.json" \
    "$evidence_dir/sanitized_snapshot/result.json" \
    "$evidence_dir/sanitized_in_place/result.json" \
    "$evidence_dir/probe_baseline/result.json" \
    "$evidence_dir/probe_snapshot/result.json" \
    "$evidence_dir/probe_in_place/result.json" \
    "$evidence_dir/size_baseline.json" \
    "$evidence_dir/size_snapshot.json" \
    "$evidence_dir/size_in_place.json" \
    "$evidence_dir/stack_baseline.json" \
    "$evidence_dir/stack_snapshot.json" \
    "$evidence_dir/stack_in_place.json" \
    "$evidence_dir/trace_baseline/service.strace" \
    "$evidence_dir/trace_baseline/post_custody.strace" \
    "$evidence_dir/trace_snapshot/service.strace" \
    "$evidence_dir/trace_snapshot/post_custody.strace" \
    "$evidence_dir/trace_in_place/service.strace" \
    "$evidence_dir/trace_in_place/post_custody.strace" \
    "$evidence_dir/trace_baseline/result.json" \
    "$evidence_dir/trace_snapshot/result.json" \
    "$evidence_dir/trace_in_place/result.json" \
    "$evidence_dir/reference_primary.json" \
    "$evidence_dir/reference_reuse.json" \
    "$evidence_dir/reference_degree2.json" \
    "$evidence_dir/reference_sham.json" \
    "$evidence_dir/reference_boundaries.json" \
    "$evidence_dir"/runs/*.json \
    "$evidence_dir/runs.json" \
    "$evidence_dir/catvm_regression/SHA256SUMS" \
    "$evidence_dir/catvm_regression/result.json" \
    "$evidence_dir/result.json" \
    >"$evidence_dir/SHA256SUMS"

printf '%s\n' "$evidence_dir"
