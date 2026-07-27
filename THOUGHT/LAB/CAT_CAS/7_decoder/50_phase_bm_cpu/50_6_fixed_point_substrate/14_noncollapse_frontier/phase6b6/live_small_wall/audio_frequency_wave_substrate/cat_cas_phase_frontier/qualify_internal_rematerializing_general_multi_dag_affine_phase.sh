#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PHASE_SOURCE="$HERE/internal_rematerializing_general_multi_dag_affine_phase.c"
LEAF_SOURCE="$HERE/automatic_compact_general_multi_dag_affine_phase.c"
REFERENCE_SOURCE="$HERE/general_multi_dag_affine_reference.c"
MANIFEST="$HERE/general_multi_dag_affine_topology.txt"
LEAF_QUALIFIER="$HERE/qualify_automatic_compact_general_multi_dag_affine_phase.sh"
OUT=${1:-}
if [[ -z "$OUT" ]]; then
    OUT=$(mktemp -d /tmp/internal-rematerializing-dag.XXXXXX)
else
    mkdir -p "$OUT"
fi
for tool in cc jq sha256sum cmp rg strace awk stat; do
    command -v "$tool" >/dev/null
done
COMMON=(
    -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror
    -Wconversion -Wshadow
)

cc "${COMMON[@]}" -fstack-usage \
    "$PHASE_SOURCE" -lm -o "$OUT/internal_phase"
cc "${COMMON[@]}" \
    "$LEAF_SOURCE" -lm -o "$OUT/leaf_phase"
cc "${COMMON[@]}" \
    "$REFERENCE_SOURCE" -lm -o "$OUT/reference"
cc "${COMMON[@]}" -fanalyzer \
    "$PHASE_SOURCE" -lm -o "$OUT/internal_analyzer"
cc "${COMMON[@]}" -DIR_WORKING_SLOTS=7U \
    "$PHASE_SOURCE" -lm -o "$OUT/capacity7"

"$OUT/internal_phase" "$MANIFEST" >"$OUT/phase.jsonl"
"$OUT/internal_phase" "$MANIFEST" >"$OUT/replay.jsonl"
"$OUT/internal_analyzer" "$MANIFEST" >"$OUT/analyzer.jsonl"
"$OUT/leaf_phase" "$MANIFEST" >"$OUT/leaf.jsonl"
"$OUT/reference" "$MANIFEST" >"$OUT/reference.jsonl"
cmp "$OUT/phase.jsonl" "$OUT/replay.jsonl"
cmp "$OUT/phase.jsonl" "$OUT/analyzer.jsonl"
jq -e . "$OUT/phase.jsonl" >/dev/null
jq -e . "$OUT/leaf.jsonl" >/dev/null
jq -e . "$OUT/reference.jsonl" >/dev/null

jq -c '
  select(
    .mode == "internal-remat-primary"
    or .mode == "internal-remat-reuse"
    or .mode == "internal-remat-semantic"
  )
  | {boundary_coefficients, boundary_fnv1a64}
' "$OUT/phase.jsonl" >"$OUT/phase_boundaries.jsonl"
jq -c '
  select(
    .mode == "automatic-compact-primary"
    or .mode == "automatic-compact-reuse"
    or .mode == "automatic-compact-semantic"
  )
  | {boundary_coefficients, boundary_fnv1a64}
' "$OUT/leaf.jsonl" >"$OUT/leaf_boundaries.jsonl"
jq -c '
  {boundary_coefficients, boundary_fnv1a64}
' "$OUT/reference.jsonl" >"$OUT/reference_boundaries.jsonl"
cmp "$OUT/phase_boundaries.jsonl" "$OUT/leaf_boundaries.jsonl"
cmp "$OUT/phase_boundaries.jsonl" "$OUT/reference_boundaries.jsonl"

expected_hashes=(
    2e2200557469163d
    6f46809cf9d2cb8e
    eb6451231a634b0b
    7a47f3c447c8ae4f
    fa9b96ae5e69b5cd
)
mapfile -t observed_hashes < <(
    jq -r '
      select(
        .mode == "internal-remat-primary"
        or .mode == "internal-remat-reuse"
        or .mode == "internal-remat-semantic"
      )
      | .boundary_fnv1a64
    ' "$OUT/phase.jsonl"
)
[[ ${#observed_hashes[@]} -eq 5 ]]
for index in 0 1 2 3 4; do
    [[ "${observed_hashes[$index]}" == "${expected_hashes[$index]}" ]]
done

jq -se '
  [.[] | select(
    .mode == "internal-remat-primary"
    or .mode == "internal-remat-reuse"
    or .mode == "internal-remat-semantic"
  )] as $runs
  | ($runs | length) == 5
  and all($runs[];
    .claim
      == "BOUNDED_15_NODE_FOUR_OWNER_INTERNAL_OPERATOR_REMATERIALIZATION_ESTABLISHED"
    and .claim_ceiling
      == "BOUNDED_WIDTH16_SOFTWARE_GF2_EXACT_15_NODE_ONE_LAYER_FOUR_INTERNAL_OPERATOR_REMATERIALIZATION_23_STEP_REVERSIBLE_TAPE_8_WORKING_SLOTS_9_PHYSICAL_BLOCKS_MULTI_ACTIVATION_EDGE_CUSTODY_REFERENCE_ONLY"
    and .width == 3
    and .manifest_nodes == 15
    and .manifest_leaves == 4
    and .compose_nodes == 8
    and .intersect_nodes == 3
    and .dag_edges == 22
    and .dag_depth == 5
    and .fanout_nodes == 4
    and .maximum_fanout == 4
    and .compiler_generated_reversible_tape == true
    and .coefficient_oblivious_plan == true
    and .plan_inputs
      == "PUBLIC_TOPOLOGY_SIGNATURES_AND_RECIPES"
    and .bound_topology_fnv1a64 == "41d917d4a3308fbe"
    and .bound_schedule_fnv1a64 == "aa5719d149bc55e0"
    and .plan_fnv1a64 == "f0345b7ae7bfe27d"
    and .plan_steps == 23
    and .tape_cursor_after_forward == 23
    and .tape_cursor_after_inverse == 0
    and .predicted_peak_live == 8
    and .runtime_peak_live == 8
    and .projection_live == 7
    and .working_relation_slots == 8
    and .physical_relation_blocks_including_boundary == 9
    and .relation_cells == 49
    and .workspace_cells == 359
    and .carrier_cells == 800
    and .live_carrier_bytes == 25600
    and .decoded_intermediate_coefficients == 0
    and .serialized_intermediate_coefficients == 0
    and .intermediate_relation_hashes == 0
    and .pre_inverse_carrier_aggregate_outputs == 0
    and .host_selected_pivots == 0
    and .tuple_slots == 0
    and .assignment_slots == 0
    and .witness_slots == 0
    and .content_dependent_receipts == 0
    and .restoration_max_abs <= .restoration_tolerance
    and .workspace_cleared == true
    and .allocator_restored == true
    and .logical_custody_restored == true
    and .obligations_restored == true
    and .residency_restored == true
    and .tape_cursor_empty == true
    and .snapshot_loaded == false
    and .lease_allocations == 23
    and .lease_releases == 23
    and .forward_native_operations == 15
    and .inverse_native_operations == 15
    and .native_operation_calls == 30
    and .forward_leaf_encodes == 8
    and .inverse_leaf_encodes == 8
    and .leaf_encode_calls == 16
    and .early_leaf_unmaterializations == 4
    and .leaf_reconstructions == 4
    and .final_leaf_inverses == 4
    and .early_internal_unmaterializations == 4
    and .internal_node_rematerializations == 4
    and .operator_recomputations == 4
    and .physical_edge_forward_activations == 30
    and .physical_edge_inverse_activations == 30
    and .multi_activation_edges == 8
    and .second_activations == 8
    and .exact_activation_receipts == 22
    and .reconstructed_leaf_edges == 4
    and .reconstructed_operator_edges == 4
    and .shared_forward_activations == 20
    and .shared_inverse_activations == 20
    and .all_receipts_lawful == true
    and .boundary_block_copy_calls == 2
    and .intermediate_block_copy_calls == 0
    and .serial_after == (.serial_before + 23)
    and .restoration_generation_after
      == (.restoration_generation_before + 1)
    and (.activation_edge_receipts | length) == 22
    and (
      [.activation_edge_receipts[]
        | select(.policy == "EXACT_MULTI_ACTIVATION")
      ] | length
    ) == 8
    and (
      [.activation_edge_receipts[]
        | select(.policy == "EXACT_SINGLE_ACTIVATION")
      ] | length
    ) == 6
    and (
      [.activation_edge_receipts[]
        | select(.policy == "PUBLIC_LEAF_RECONSTRUCTION")
      ] | length
    ) == 4
    and (
      [.activation_edge_receipts[]
        | select(.policy == "INTERNAL_OPERATOR_RECONSTRUCTION")
      ] | length
    ) == 4
    and all(.activation_edge_receipts[];
      .receipt_count == .activation_limit
      and .closed_count == .activation_limit
      and .all_activations_lawful == true
    )
    and (.shared_owner_activation_receipts | length) == 4
    and all(.shared_owner_activation_receipts[];
      .forward_activations == .inverse_activations
      and .one_original_generation == true
      and .live_at_projection == true
      and .live_after_root_inverse == true
    )
    and (
      [.shared_owner_activation_receipts[]
        | {id: .public_node_id, n: .forward_activations}
      ] | sort_by(.id)
    ) == [
      {"id":805,"n":6},
      {"id":806,"n":4},
      {"id":807,"n":6},
      {"id":808,"n":4}
    ]
  )
  and ([$runs[].plan_fnv1a64] | unique | length) == 1
' "$OUT/phase.jsonl" >/dev/null

jq -e '
  select(.mode == "internal-remat-controls")
  | .plan_fnv1a64 == "f0345b7ae7bfe27d"
    and .primary_reuse_boundary_different == true
    and .wrong_root_inverse_detected == true
    and .missing_root_inverse_detected == true
    and .snapshot_separate_weaker_path == true
    and .snapshot_post_projection_inverse_operations == 0
    and .accepted_working_slots == 8
    and .leaf_only_working_slots == 11
    and .retain_all_working_slots == 15
    and .occurrence_working_slots == 51
    and .accepted_native_operation_calls == 30
    and .accepted_leaf_encode_calls == 16
    and .accepted_lease_allocations == 23
    and .same_carrier_transactions == 17
    and .maximum_reuse_restoration_abs <= 2e-12
    and .wrong_root_restoration_abs > 1e-6
    and .missing_root_restoration_abs > 1e-6
' "$OUT/phase.jsonl" >/dev/null
jq -e '
  select(.mode == "internal-remat-snapshot-baseline")
  | .snapshot_loaded == true
    and .restoration_max_abs == 0
    and .serial_before == .serial_after
    and .restoration_generation_before
      == .restoration_generation_after
' "$OUT/phase.jsonl" >/dev/null

run_reject() {
    local control=$1
    local expected=$2
    local name=${control#--}
    set +e
    "$OUT/internal_phase" "$MANIFEST" "$control" \
        >"$OUT/control_${name}.stdout" \
        2>"$OUT/control_${name}.stderr"
    local rc=$?
    set -e
    [[ $rc -ne 0 ]]
    [[ ! -s "$OUT/control_${name}.stdout" ]]
    [[ -s "$OUT/control_${name}.stderr" ]]
    rg -q "$expected" "$OUT/control_${name}.stderr"
}

run_reject --control-stale-consumer-activation \
    'consumer activation stale'
run_reject --control-missing-activation-close \
    'activation begin state invalid'
run_reject --control-stale-producer-activation \
    'reconstruction activation invalid'
run_reject --control-missing-operator-reconstruction \
    'reconstruction activation invalid'
run_reject --control-double-operator-reconstruction \
    'operator obligation invalid'
run_reject --control-stale-obligation-epoch \
    'operator obligation invalid'
run_reject --control-evict-noneligible \
    'operator eviction ineligible'
run_reject --control-tamper-tape 'tape hash changed'
run_reject --project-intermediate 'projection denied'
run_reject --project-tape 'projection denied'
run_reject --project-obligations 'projection denied'
run_reject --project-residency 'projection denied'
run_reject --project-activation-receipts 'projection denied'
run_reject --null-carrier 'null internal rematerialization carrier'

set +e
"$OUT/capacity7" "$MANIFEST" \
    >"$OUT/control_capacity7.stdout" \
    2>"$OUT/control_capacity7.stderr"
capacity_rc=$?
set -e
[[ $capacity_rc -ne 0 ]]
[[ ! -s "$OUT/control_capacity7.stdout" ]]
rg -q 'clean relation-slot pool exhausted' \
    "$OUT/control_capacity7.stderr"

if rg -q 'ga_copy_block\(' "$PHASE_SOURCE"; then
    exit 1
fi
strace -qq -f \
    -e trace=%file,%network,%ipc,%memory,write,writev,pwrite64,pwritev,pwritev2,memfd_create,sendfile,splice,tee,vmsplice,copy_file_range,process_vm_writev,ptrace,ioctl \
    -o "$OUT/no_smuggle.strace" \
    "$OUT/internal_phase" "$MANIFEST" \
    >"$OUT/traced.jsonl" 2>"$OUT/traced.stderr"
cmp "$OUT/phase.jsonl" "$OUT/traced.jsonl"
[[ ! -s "$OUT/traced.stderr" ]]
if rg -q \
    'O_WRONLY|O_RDWR|O_CREAT|write(v)?\((2|[3-9]|[1-9][0-9]+),|pwrite|send(to|msg|mmsg)?\(|socket\(|connect\(|MAP_SHARED|memfd_create|shm(get|at|dt|ctl)|msg(get|snd|rcv|ctl)|sem(get|op|ctl)|sendfile|splice|tee\(|vmsplice|copy_file_range|process_vm_writev|ptrace|ioctl' \
    "$OUT/no_smuggle.strace"
then
    exit 1
fi

cc -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror \
    -Wconversion -Wshadow \
    -DIR_REUSE_CYCLES=0U \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    "$PHASE_SOURCE" -lm -o "$OUT/internal_sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$OUT/internal_sanitized" "$MANIFEST" \
    >"$OUT/sanitized.jsonl" 2>"$OUT/sanitized.stderr"
[[ ! -s "$OUT/sanitized.stderr" ]]
jq -e . "$OUT/sanitized.jsonl" >/dev/null

: >"$OUT/scaling.jsonl"
for width in 3 4 8 12 16; do
    cc "${COMMON[@]}" \
        -DGM_WIDTH="${width}U" \
        -DIR_REUSE_CYCLES=0U \
        -DIR_SCALE_ONLY=1 \
        "$PHASE_SOURCE" -lm -o "$OUT/phase_w${width}"
    cc "${COMMON[@]}" \
        -DGM_REF_WIDTH="${width}U" \
        "$REFERENCE_SOURCE" -lm -o "$OUT/reference_w${width}"
    "$OUT/phase_w${width}" "$MANIFEST" \
        >"$OUT/phase_w${width}.jsonl"
    "$OUT/reference_w${width}" "$MANIFEST" \
        >"$OUT/reference_w${width}.jsonl"
    jq -c '
      select(
        .mode == "internal-remat-primary"
        or .mode == "internal-remat-reuse"
        or .mode == "internal-remat-semantic"
      )
      | {boundary_coefficients, boundary_fnv1a64}
    ' "$OUT/phase_w${width}.jsonl" \
        >"$OUT/phase_w${width}.boundaries.jsonl"
    jq -c '
      {boundary_coefficients, boundary_fnv1a64}
    ' "$OUT/reference_w${width}.jsonl" \
        >"$OUT/reference_w${width}.boundaries.jsonl"
    cmp \
        "$OUT/phase_w${width}.boundaries.jsonl" \
        "$OUT/reference_w${width}.boundaries.jsonl"
    jq -e --argjson w "$width" '
      select(
        .mode == "internal-remat-primary"
        or .mode == "internal-remat-reuse"
        or .mode == "internal-remat-semantic"
      )
      | .width == $w
        and .working_relation_slots == 8
        and .runtime_peak_live == 8
        and .physical_relation_blocks_including_boundary == 9
        and .relation_cells == (4*$w*$w + 4*$w + 1)
        and .workspace_cells == (36*$w*$w + 11*$w + 2)
        and .carrier_cells == (72*$w*$w + 47*$w + 11)
        and .native_operation_calls == 30
        and .leaf_encode_calls == 16
        and .lease_allocations == 23
        and .lease_releases == 23
        and .physical_edge_forward_activations == 30
        and .physical_edge_inverse_activations == 30
        and .restoration_max_abs <= .restoration_tolerance
    ' "$OUT/phase_w${width}.jsonl" >/dev/null
    jq -c '
      select(.mode == "internal-remat-primary")
      | {
          width,
          relation_cells,
          workspace_cells,
          carrier_cells,
          live_carrier_bytes,
          carrier_and_verification_snapshot_heap_bytes,
          compiled_topology_bytes_current_abi,
          program_storage_bytes_current_abi,
          logical_custody_state_bytes_current_abi,
          activation_receipt_state_bytes_current_abi,
          reversible_plan_storage_bytes_current_abi,
          obligation_storage_bytes_current_abi,
          execution_summary_bytes_current_abi,
          phase_ands,
          phase_xors,
          phase_cell_updates,
          restoration_max_abs,
          boundary_fnv1a64
        }
    ' "$OUT/phase_w${width}.jsonl" >>"$OUT/scaling.jsonl"
done

"$LEAF_QUALIFIER" "$OUT/leaf_regression" \
    >"$OUT/leaf_regression.stdout"
jq -e '.result == "PASS"' \
    "$OUT/leaf_regression/result.json" >/dev/null

phase_hash=$(sha256sum "$PHASE_SOURCE" | awk '{print $1}')
leaf_hash=$(sha256sum "$LEAF_SOURCE" | awk '{print $1}')
reference_hash=$(sha256sum "$REFERENCE_SOURCE" | awk '{print $1}')
manifest_hash=$(sha256sum "$MANIFEST" | awk '{print $1}')
qualifier_hash=$(sha256sum "$0" | awk '{print $1}')
output_hash=$(sha256sum "$OUT/phase.jsonl" | awk '{print $1}')
leaf_output_hash=$(sha256sum "$OUT/leaf.jsonl" | awk '{print $1}')
reference_output_hash=$(
    sha256sum "$OUT/reference.jsonl" | awk '{print $1}'
)
trace_hash=$(sha256sum "$OUT/no_smuggle.strace" | awk '{print $1}')
phase_binary_bytes=$(stat -c %s "$OUT/internal_phase")
accepted_stdout_bytes=$(stat -c %s "$OUT/phase.jsonl")
stack_file=$(
    find "$OUT" -maxdepth 1 -name \
      'internal_phase-internal_rematerializing_general_multi_dag_affine_phase.su'
)
main_stack_bytes=$(
    awk -F '\t' '$1 ~ /:main$/ {print $2}' "$stack_file"
)
execute_stack_bytes=$(
    awk -F '\t' '$1 ~ /:ir_execute$/ {print $2}' "$stack_file"
)
accepted_stack_floor_bytes=$(
    awk -v main="$main_stack_bytes" -v execute="$execute_stack_bytes" \
        'BEGIN {print main + execute}'
)
all_frame_sum_bytes=$(
    awk -F '\t' '{sum += $2} END {print sum}' "$stack_file"
)
stack_function_records=$(awk 'END {print NR}' "$stack_file")

jq -n \
    --arg evidence_directory "$OUT" \
    --arg phase_hash "$phase_hash" \
    --arg leaf_hash "$leaf_hash" \
    --arg reference_hash "$reference_hash" \
    --arg manifest_hash "$manifest_hash" \
    --arg qualifier_hash "$qualifier_hash" \
    --arg output_hash "$output_hash" \
    --arg leaf_output_hash "$leaf_output_hash" \
    --arg reference_output_hash "$reference_output_hash" \
    --arg trace_hash "$trace_hash" \
    --argjson phase_binary_bytes "$phase_binary_bytes" \
    --argjson accepted_stdout_bytes "$accepted_stdout_bytes" \
    --argjson main_stack_bytes "$main_stack_bytes" \
    --argjson execute_stack_bytes "$execute_stack_bytes" \
    --argjson accepted_stack_floor_bytes "$accepted_stack_floor_bytes" \
    --argjson all_frame_sum_bytes "$all_frame_sum_bytes" \
    --argjson stack_function_records "$stack_function_records" \
    --argjson reuse_error "$(
      jq '
        select(.mode == "internal-remat-controls")
        | .maximum_reuse_restoration_abs
      ' "$OUT/phase.jsonl"
    )" \
    --argjson width16 "$(
      jq -s 'map(select(.width == 16))[0]' "$OUT/scaling.jsonl"
    )" \
    '{
      result: "PASS",
      claim:
        "BOUNDED_15_NODE_FOUR_OWNER_INTERNAL_OPERATOR_REMATERIALIZATION_ESTABLISHED",
      claim_ceiling:
        "BOUNDED_WIDTH16_SOFTWARE_GF2_EXACT_15_NODE_ONE_LAYER_FOUR_INTERNAL_OPERATOR_REMATERIALIZATION_23_STEP_REVERSIBLE_TAPE_8_WORKING_SLOTS_9_PHYSICAL_BLOCKS_MULTI_ACTIVATION_EDGE_CUSTODY_REFERENCE_ONLY",
      evidence_directory: $evidence_directory,
      tested_widths: [3,4,8,12,16],
      exact_phase_reference_complete_boundaries: 25,
      deterministic_replay: "PASS",
      analyzer: "PASS",
      sanitizers: "PASS",
      mandatory_expanded_no_smuggle_strace: "PASS",
      leaf_only_predecessor_regression: "PASS",
      schedule: {
        compiler_generated_reversible_tape: true,
        bound_topology_fnv1a64: "41d917d4a3308fbe",
        bound_schedule_fnv1a64: "aa5719d149bc55e0",
        plan_fnv1a64: "f0345b7ae7bfe27d",
        tape_steps: 23,
        rematerialized_internal_public_ids: [809,810,811,812],
        reconstructed_public_leaf_ids: [801,802,803,804],
        predicted_and_observed_peak_live: 8,
        projection_live: 7,
        capacity_eight_passes_capacity_seven_fails: true
      },
      custody: {
        pinned_shared_owners: [805,806,807,808],
        logical_shared_fanout_histogram: [4,3,3,2],
        physical_shared_activation_histogram_by_public_id:
          [{"id":805,"n":6},{"id":806,"n":4},{"id":807,"n":6},{"id":808,"n":4}],
        logical_edges: 22,
        physical_activation_receipts: 30,
        exact_activation_receipts: 22,
        lawful_reconstructed_public_leaf_edges: 4,
        lawful_reconstructed_internal_operator_edges: 4,
        multi_activation_edges: 8,
        second_activations: 8,
        intermediate_relation_copies: 0
      },
      resources: {
        carrier_cells_law: "72*w^2 + 47*w + 11",
        leaf_only_carrier_cells_law: "84*w^2 + 59*w + 14",
        retain_all_carrier_cells_law: "100*w^2 + 75*w + 18",
        occurrence_carrier_cells_law: "244*w^2 + 219*w + 54",
        default_width3_carrier_cells: 800,
        default_width3_live_carrier_bytes: 25600,
        native_calls: 30,
        leaf_encode_calls: 16,
        lease_allocations: 23,
        same_carrier_transactions: 17,
        maximum_reuse_restoration_error: $reuse_error
      },
      maximum_width16: $width16,
      controls: {
        stale_consumer_activation_rejected: true,
        missing_activation_close_rejected: true,
        stale_producer_activation_rejected: true,
        missing_double_and_stale_operator_reconstruction_rejected: true,
        noneligible_eviction_rejected: true,
        tape_tamper_rejected: true,
        projection_and_null_rejected: true,
        wrong_and_missing_root_inverse_detected: true,
        snapshot_separate_weaker_path: true,
        no_smuggle: true
      },
      additional_material_resources: {
        phase_binary_bytes: $phase_binary_bytes,
        accepted_stdout_bytes: $accepted_stdout_bytes,
        compiler_measured_main_stack_bytes: $main_stack_bytes,
        compiler_measured_ir_execute_stack_bytes: $execute_stack_bytes,
        compiler_measured_accepted_main_plus_execute_stack_floor_bytes:
          $accepted_stack_floor_bytes,
        compiler_all_function_frames_sum_conservative_upper_bound_bytes:
          $all_frame_sum_bytes,
        compiler_stack_usage_function_records: $stack_function_records
      },
      sha256: {
        phase_source: $phase_hash,
        leaf_only_source: $leaf_hash,
        reference_source: $reference_hash,
        public_manifest: $manifest_hash,
        qualifier: $qualifier_hash,
        accepted_output: $output_hash,
        leaf_only_output: $leaf_output_hash,
        reference_output: $reference_output_hash,
        no_smuggle_trace: $trace_hash
      },
      recursive_internal_rematerialization: false,
      automatic_general_dag_pebbling: false,
      arbitrary_graph_topology: false,
      global_pebbling_optimum: false,
      catvm_enforcement: false,
      total_memory_advantage: false,
      performance_advantage: false,
      physical_execution: false,
      small_wall_crossed: false,
      unlimited_catalytic_computation: false,
      terminal: false
    }' | tee "$OUT/result.json"
