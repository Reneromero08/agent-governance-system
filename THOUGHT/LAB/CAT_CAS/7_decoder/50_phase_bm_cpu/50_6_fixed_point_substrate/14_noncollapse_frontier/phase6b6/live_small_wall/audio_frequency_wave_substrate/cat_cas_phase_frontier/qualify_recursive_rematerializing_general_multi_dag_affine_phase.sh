#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PHASE_SOURCE="$HERE/recursive_rematerializing_general_multi_dag_affine_phase.c"
PREDECESSOR_SOURCE="$HERE/internal_rematerializing_general_multi_dag_affine_phase.c"
REFERENCE_SOURCE="$HERE/general_multi_dag_affine_reference.c"
MANIFEST="$HERE/general_multi_dag_affine_topology.txt"
PREDECESSOR_QUALIFIER="$HERE/qualify_internal_rematerializing_general_multi_dag_affine_phase.sh"
OUT=${1:-}
if [[ -z "$OUT" ]]; then
    OUT=$(mktemp -d /tmp/recursive-rematerializing-dag.XXXXXX)
else
    mkdir -p "$OUT"
fi
for tool in cc jq sha256sum cmp rg strace awk stat find; do
    command -v "$tool" >/dev/null
done
COMMON=(
    -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror
    -Wconversion -Wshadow
)

cc "${COMMON[@]}" -fstack-usage \
    "$PHASE_SOURCE" -lm -o "$OUT/recursive_phase"
cc "${COMMON[@]}" \
    "$PREDECESSOR_SOURCE" -lm -o "$OUT/predecessor_phase"
cc "${COMMON[@]}" \
    "$REFERENCE_SOURCE" -lm -o "$OUT/reference"
cc "${COMMON[@]}" -fanalyzer \
    "$PHASE_SOURCE" -lm -o "$OUT/recursive_analyzer"
cc "${COMMON[@]}" -DRR_WORKING_SLOTS=8U \
    "$PHASE_SOURCE" -lm -o "$OUT/capacity8"

"$OUT/recursive_phase" "$MANIFEST" >"$OUT/phase.jsonl"
"$OUT/recursive_phase" "$MANIFEST" >"$OUT/replay.jsonl"
"$OUT/recursive_analyzer" "$MANIFEST" >"$OUT/analyzer.jsonl"
"$OUT/predecessor_phase" "$MANIFEST" >"$OUT/predecessor.jsonl"
"$OUT/reference" "$MANIFEST" >"$OUT/reference.jsonl"
cmp "$OUT/phase.jsonl" "$OUT/replay.jsonl"
cmp "$OUT/phase.jsonl" "$OUT/analyzer.jsonl"
jq -e . "$OUT/phase.jsonl" >/dev/null
jq -e . "$OUT/predecessor.jsonl" >/dev/null
jq -e . "$OUT/reference.jsonl" >/dev/null

jq -c '
  select(
    .mode == "recursive-remat-primary"
    or .mode == "recursive-remat-reuse"
    or .mode == "recursive-remat-semantic"
  )
  | {boundary_coefficients, boundary_fnv1a64}
' "$OUT/phase.jsonl" >"$OUT/phase_boundaries.jsonl"
jq -c '
  select(
    .mode == "internal-remat-primary"
    or .mode == "internal-remat-reuse"
    or .mode == "internal-remat-semantic"
  )
  | {boundary_coefficients, boundary_fnv1a64}
' "$OUT/predecessor.jsonl" >"$OUT/predecessor_boundaries.jsonl"
jq -c '
  {boundary_coefficients, boundary_fnv1a64}
' "$OUT/reference.jsonl" >"$OUT/reference_boundaries.jsonl"
cmp "$OUT/phase_boundaries.jsonl" "$OUT/predecessor_boundaries.jsonl"
cmp "$OUT/phase_boundaries.jsonl" "$OUT/reference_boundaries.jsonl"

jq -se '
  [.[] | select(
    .mode == "recursive-remat-primary"
    or .mode == "recursive-remat-reuse"
    or .mode == "recursive-remat-semantic"
  )] as $runs
  | ($runs | length) == 5
  and all($runs[];
    .claim
      == "BOUNDED_15_NODE_RANK2_RECURSIVE_OPERATOR_REMATERIALIZATION_ESTABLISHED"
    and .claim_ceiling
      == "BOUNDED_WIDTH16_SOFTWARE_GF2_EXACT_15_NODE_SINGLE_RANK2_BORROWER_28_FORWARD_28_REVERSE_ACTIONS_9_WORKING_SLOTS_10_PHYSICAL_BLOCKS_MAX_ACTIVATION_ORDINAL3_REFERENCE_ONLY"
    and .width == 3
    and .manifest_nodes == 15
    and .manifest_leaves == 4
    and .compose_nodes == 8
    and .intersect_nodes == 3
    and .dag_edges == 22
    and .dag_depth == 5
    and .fanout_nodes == 4
    and .compiler_generated_forward_and_literal_reverse_tapes == true
    and .coefficient_oblivious_plan == true
    and .rank_selection == "MAX_RANK_THEN_PUBLIC_ID_ASC"
    and .selected_rank2_public_id == 813
    and .nested_rematerialization_frame_depth == 2
    and .activation_chain_depth == 3
    and .maximum_activation_ordinal == 3
    and .bound_topology_fnv1a64 == "41d917d4a3308fbe"
    and .bound_schedule_fnv1a64 == "aa5719d149bc55e0"
    and .forward_actions == 28
    and .reverse_actions == 28
    and .forward_cursor == 28
    and .reverse_cursor == 28
    and .predicted_peak_live == 9
    and .runtime_peak_live == 9
    and .projection_live == 6
    and .working_relation_slots == 9
    and .physical_relation_blocks_including_boundary == 10
    and .relation_cells == 49
    and .workspace_cells == 359
    and .carrier_cells == 849
    and .live_carrier_bytes == 27168
    and .decoded_intermediate_coefficients == 0
    and .serialized_intermediate_coefficients == 0
    and .intermediate_relation_hashes == 0
    and .pre_inverse_carrier_aggregate_outputs == 0
    and .tuple_slots == 0
    and .assignment_slots == 0
    and .witness_slots == 0
    and .content_dependent_receipts == 0
    and .restoration_max_abs <= .restoration_tolerance
    and .workspace_cleared == true
    and .allocator_restored == true
    and .logical_custody_restored == true
    and .receipts_restored == true
    and .nodes_restored == true
    and .obligations_restored == true
    and .snapshot_loaded == false
    and .lease_allocations == 28
    and .lease_releases == 28
    and .forward_native_operations == 20
    and .inverse_native_operations == 20
    and .native_operation_calls == 40
    and .forward_leaf_encodes == 8
    and .inverse_leaf_encodes == 8
    and .leaf_encode_calls == 16
    and .operator_reconstructions == 9
    and .operator_suspensions == 4
    and .persistent_obligations_created == 9
    and .persistent_obligations_cleared == 9
    and .physical_edge_forward_activations == 40
    and .physical_edge_inverse_activations == 40
    and .multi_activation_edges == 10
    and .second_or_later_activations == 18
    and .exact_activation_receipts == 29
    and .leaf_rebind_receipts == 4
    and .operator_rebind_receipts == 7
    and .shared_forward_activations == 28
    and .shared_inverse_activations == 28
    and .boundary_block_copy_calls == 2
    and .intermediate_block_copy_calls == 0
    and .serial_after == (.serial_before + 28)
    and .restoration_generation_after
      == (.restoration_generation_before + 1)
    and (.activation_edge_plans | length) == 22
    and (
      [.activation_edge_plans[]
        | select(.activation_limit > 1)
      ] | length
    ) == 10
    and (
      [.activation_edge_plans[]
        | .receipts[]
        | select(.policy == "EXACT")
      ] | length
    ) == 29
    and (
      [.activation_edge_plans[]
        | .receipts[]
        | select(.policy == "PUBLIC_LEAF_REBIND")
      ] | length
    ) == 4
    and (
      [.activation_edge_plans[]
        | .receipts[]
        | select(.policy == "INTERNAL_OPERATOR_REBIND")
      ] | length
    ) == 7
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
      {"id":805,"n":8},
      {"id":806,"n":6},
      {"id":807,"n":10},
      {"id":808,"n":4}
    ]
  )
  and ([$runs[].plan_fnv1a64] | unique | length) == 1
' "$OUT/phase.jsonl" >/dev/null

jq -e '
  select(.mode == "recursive-remat-controls")
  | .primary_reuse_boundary_different == true
    and .wrong_root_inverse_detected == true
    and .missing_root_inverse_detected == true
    and .snapshot_separate_weaker_path == true
    and .snapshot_post_projection_inverse_operations == 0
    and .accepted_working_slots == 9
    and .one_layer_working_slots == 8
    and .leaf_only_working_slots == 11
    and .retain_all_working_slots == 15
    and .occurrence_working_slots == 51
    and .accepted_native_operation_calls == 40
    and .accepted_leaf_encode_calls == 16
    and .accepted_lease_allocations == 28
    and .same_carrier_transactions == 17
    and .maximum_reuse_restoration_abs <= 2e-12
    and .wrong_root_restoration_abs > 1e-6
    and .missing_root_restoration_abs > 1e-6
' "$OUT/phase.jsonl" >/dev/null
jq -e '
  select(.mode == "recursive-remat-snapshot-baseline")
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
    "$OUT/recursive_phase" "$MANIFEST" "$control" \
        >"$OUT/control_${name}.stdout" \
        2>"$OUT/control_${name}.stderr"
    local rc=$?
    set -e
    [[ $rc -ne 0 ]]
    [[ ! -s "$OUT/control_${name}.stdout" ]]
    [[ -s "$OUT/control_${name}.stderr" ]]
    rg -q "$expected" "$OUT/control_${name}.stderr"
}

run_reject --control-deep-stale-consumer \
    'receipt close mismatch'
run_reject --control-deep-stale-producer \
    'receipt close mismatch'
run_reject --control-rebind-swap \
    'rebind authorization swapped'
run_reject --control-activation-reuse \
    'operator generation reused'
run_reject --control-missing-nested-close \
    'prior receipt still open'
run_reject --control-missing-nested-child \
    'nested child missing'
run_reject --control-tamper-tape \
    'plan provenance changed'
run_reject --project-intermediate 'projection denied'
run_reject --project-forward-tape 'projection denied'
run_reject --project-reverse-tape 'projection denied'
run_reject --project-obligations 'projection denied'
run_reject --project-node-generations 'projection denied'
run_reject --project-activation-receipts 'projection denied'
run_reject --debug-dump 'projection denied'
run_reject --null-carrier \
    'null recursive rematerialization carrier'

set +e
"$OUT/capacity8" "$MANIFEST" \
    >"$OUT/control_capacity8.stdout" \
    2>"$OUT/control_capacity8.stderr"
capacity_rc=$?
set -e
[[ $capacity_rc -ne 0 ]]
[[ ! -s "$OUT/control_capacity8.stdout" ]]
rg -q 'clean relation-slot pool exhausted' \
    "$OUT/control_capacity8.stderr"

if rg -q 'ga_copy_block\(' "$PHASE_SOURCE"; then
    exit 1
fi
strace -qq -f \
    -e trace=%file,%network,%ipc,%memory,write,writev,pwrite64,pwritev,pwritev2,memfd_create,sendfile,splice,tee,vmsplice,copy_file_range,process_vm_writev,ptrace,ioctl \
    -o "$OUT/no_smuggle.strace" \
    "$OUT/recursive_phase" "$MANIFEST" \
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
    -DRR_REUSE_CYCLES=0U \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    "$PHASE_SOURCE" -lm -o "$OUT/recursive_sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$OUT/recursive_sanitized" "$MANIFEST" \
    >"$OUT/sanitized.jsonl" 2>"$OUT/sanitized.stderr"
[[ ! -s "$OUT/sanitized.stderr" ]]
jq -e . "$OUT/sanitized.jsonl" >/dev/null

: >"$OUT/scaling.jsonl"
for width in 3 4 8 12 16; do
    cc "${COMMON[@]}" \
        -DGM_WIDTH="${width}U" \
        -DRR_REUSE_CYCLES=0U \
        -DRR_SCALE_ONLY=1 \
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
        .mode == "recursive-remat-primary"
        or .mode == "recursive-remat-reuse"
        or .mode == "recursive-remat-semantic"
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
        .mode == "recursive-remat-primary"
        or .mode == "recursive-remat-reuse"
        or .mode == "recursive-remat-semantic"
      )
      | .width == $w
        and .working_relation_slots == 9
        and .runtime_peak_live == 9
        and .physical_relation_blocks_including_boundary == 10
        and .relation_cells == (4*$w*$w + 4*$w + 1)
        and .workspace_cells == (36*$w*$w + 11*$w + 2)
        and .carrier_cells == (76*$w*$w + 51*$w + 12)
        and .native_operation_calls == 40
        and .leaf_encode_calls == 16
        and .lease_allocations == 28
        and .lease_releases == 28
        and .physical_edge_forward_activations == 40
        and .physical_edge_inverse_activations == 40
        and .restoration_max_abs <= .restoration_tolerance
    ' "$OUT/phase_w${width}.jsonl" >/dev/null
    jq -c '
      select(.mode == "recursive-remat-primary")
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
          activation_plan_bytes_current_abi,
          activation_runtime_bytes_current_abi,
          node_generation_state_bytes_current_abi,
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

"$PREDECESSOR_QUALIFIER" "$OUT/predecessor_regression" \
    >"$OUT/predecessor_regression.stdout"
jq -e '.result == "PASS"' \
    "$OUT/predecessor_regression/result.json" >/dev/null

phase_hash=$(sha256sum "$PHASE_SOURCE" | awk '{print $1}')
predecessor_hash=$(
    sha256sum "$PREDECESSOR_SOURCE" | awk '{print $1}'
)
reference_hash=$(sha256sum "$REFERENCE_SOURCE" | awk '{print $1}')
manifest_hash=$(sha256sum "$MANIFEST" | awk '{print $1}')
qualifier_hash=$(sha256sum "$0" | awk '{print $1}')
output_hash=$(sha256sum "$OUT/phase.jsonl" | awk '{print $1}')
predecessor_output_hash=$(
    sha256sum "$OUT/predecessor.jsonl" | awk '{print $1}'
)
reference_output_hash=$(
    sha256sum "$OUT/reference.jsonl" | awk '{print $1}'
)
trace_hash=$(sha256sum "$OUT/no_smuggle.strace" | awk '{print $1}')
phase_binary_bytes=$(stat -c %s "$OUT/recursive_phase")
accepted_stdout_bytes=$(stat -c %s "$OUT/phase.jsonl")
stack_file=$(
    find "$OUT" -maxdepth 1 -name \
      'recursive_phase-recursive_rematerializing_general_multi_dag_affine_phase.su'
)
main_stack_bytes=$(
    awk -F '\t' '$1 ~ /:main$/ {print $2}' "$stack_file"
)
execute_stack_bytes=$(
    awk -F '\t' '$1 ~ /:rr_execute$/ {print $2}' "$stack_file"
)
accepted_stack_floor_bytes=$(
    awk -v main="$main_stack_bytes" -v execute="$execute_stack_bytes" \
        'BEGIN {print main + execute}'
)
all_frame_sum_bytes=$(
    awk -F '\t' '{sum += $2} END {print sum}' "$stack_file"
)
stack_function_records=$(awk 'END {print NR}' "$stack_file")
plan_hash=$(
    jq -r '
      select(.mode == "recursive-remat-primary")
      | .plan_fnv1a64
    ' "$OUT/phase.jsonl"
)

jq -n \
    --arg evidence_directory "$OUT" \
    --arg plan_hash "$plan_hash" \
    --arg phase_hash "$phase_hash" \
    --arg predecessor_hash "$predecessor_hash" \
    --arg reference_hash "$reference_hash" \
    --arg manifest_hash "$manifest_hash" \
    --arg qualifier_hash "$qualifier_hash" \
    --arg output_hash "$output_hash" \
    --arg predecessor_output_hash "$predecessor_output_hash" \
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
        select(.mode == "recursive-remat-controls")
        | .maximum_reuse_restoration_abs
      ' "$OUT/phase.jsonl"
    )" \
    --argjson width16 "$(
      jq -s 'map(select(.width == 16))[0]' "$OUT/scaling.jsonl"
    )" \
    '{
      result: "PASS",
      claim:
        "BOUNDED_15_NODE_RANK2_RECURSIVE_OPERATOR_REMATERIALIZATION_ESTABLISHED",
      claim_ceiling:
        "BOUNDED_WIDTH16_SOFTWARE_GF2_EXACT_15_NODE_SINGLE_RANK2_BORROWER_28_FORWARD_28_REVERSE_ACTIONS_9_WORKING_SLOTS_10_PHYSICAL_BLOCKS_MAX_ACTIVATION_ORDINAL3_REFERENCE_ONLY",
      evidence_directory: $evidence_directory,
      tested_widths: [3,4,8,12,16],
      exact_phase_reference_complete_boundaries: 25,
      deterministic_replay: "PASS",
      analyzer: "PASS",
      sanitizers: "PASS",
      mandatory_expanded_no_smuggle_strace: "PASS",
      one_layer_predecessor_regression: "PASS",
      schedule: {
        compiler_generated_literal_tapes: true,
        bound_topology_fnv1a64: "41d917d4a3308fbe",
        bound_schedule_fnv1a64: "aa5719d149bc55e0",
        plan_fnv1a64: $plan_hash,
        forward_actions: 28,
        reverse_actions: 28,
        selected_rank2_public_id: 813,
        nested_frame_depth: 2,
        activation_chain_depth: 3,
        maximum_activation_ordinal: 3,
        predicted_and_observed_peak_live: 9,
        projection_live: 6,
        capacity_nine_passes_capacity_eight_fails: true
      },
      custody: {
        pinned_shared_owners: [805,806,807,808],
        logical_shared_fanout_histogram: [4,3,3,2],
        physical_shared_activation_histogram_by_public_id:
          [{"id":805,"n":8},{"id":806,"n":6},{"id":807,"n":10},{"id":808,"n":4}],
        logical_edges: 22,
        physical_activation_receipts: 40,
        exact_activation_receipts: 29,
        leaf_rebind_receipts: 4,
        operator_rebind_receipts: 7,
        multi_activation_edges: 10,
        second_or_later_activations: 18,
        intermediate_relation_copies: 0
      },
      resources: {
        carrier_cells_law: "76*w^2 + 51*w + 12",
        one_layer_carrier_cells_law: "72*w^2 + 47*w + 11",
        leaf_only_carrier_cells_law: "84*w^2 + 59*w + 14",
        retain_all_carrier_cells_law: "100*w^2 + 75*w + 18",
        occurrence_carrier_cells_law: "244*w^2 + 219*w + 54",
        default_width3_carrier_cells: 849,
        default_width3_live_carrier_bytes: 27168,
        native_calls: 40,
        leaf_encode_calls: 16,
        lease_allocations: 28,
        same_carrier_transactions: 17,
        maximum_reuse_restoration_error: $reuse_error
      },
      maximum_width16: $width16,
      controls: {
        deep_stale_consumer_rejected: true,
        deep_stale_producer_rejected: true,
        rebind_authorization_swap_rejected: true,
        activation_reuse_rejected: true,
        missing_nested_close_rejected: true,
        missing_nested_child_rejected: true,
        tape_tamper_rejected: true,
        projection_debug_and_null_rejected: true,
        wrong_and_missing_root_inverse_detected: true,
        snapshot_separate_weaker_path: true,
        no_smuggle: true
      },
      additional_material_resources: {
        phase_binary_bytes: $phase_binary_bytes,
        accepted_stdout_bytes: $accepted_stdout_bytes,
        compiler_measured_main_stack_bytes: $main_stack_bytes,
        compiler_measured_rr_execute_stack_bytes: $execute_stack_bytes,
        compiler_measured_accepted_main_plus_execute_stack_floor_bytes:
          $accepted_stack_floor_bytes,
        compiler_all_function_frames_sum_conservative_upper_bound_bytes:
          $all_frame_sum_bytes,
        compiler_stack_usage_function_records: $stack_function_records
      },
      sha256: {
        phase_source: $phase_hash,
        one_layer_predecessor_source: $predecessor_hash,
        reference_source: $reference_hash,
        public_manifest: $manifest_hash,
        qualifier: $qualifier_hash,
        accepted_output: $output_hash,
        predecessor_output: $predecessor_output_hash,
        reference_output: $reference_output_hash,
        no_smuggle_trace: $trace_hash
      },
      both_rank2_branches: false,
      automatic_general_dag_pebbling: false,
      arbitrary_graph_topology: false,
      unbounded_activation_depth: false,
      catvm_enforcement: false,
      non_affine_relations: false,
      total_memory_advantage: false,
      performance_advantage: false,
      physical_execution: false,
      small_wall_crossed: false,
      unlimited_catalytic_computation: false,
      terminal: false
    }' | tee "$OUT/result.json"
