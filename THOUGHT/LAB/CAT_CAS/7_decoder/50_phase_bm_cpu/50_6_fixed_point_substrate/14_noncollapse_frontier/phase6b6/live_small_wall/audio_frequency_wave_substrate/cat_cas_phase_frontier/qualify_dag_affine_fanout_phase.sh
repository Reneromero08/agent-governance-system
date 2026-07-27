#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PHASE_SOURCE="$HERE/dag_affine_fanout_phase.c"
REFERENCE_SOURCE="$HERE/dag_affine_fanout_reference.c"
DAG_MANIFEST="$HERE/dag_affine_fanout_topology.txt"
DUPLICATE_MANIFEST="$HERE/dag_affine_duplicate_tree_topology.txt"
RECURSIVE_PHASE="$HERE/recursive_affine_module_phase.c"
RECURSIVE_REFERENCE="$HERE/recursive_affine_module_reference.c"
MODULE_SOURCE="$HERE/algebraic_affine_module_phase.c"
MODULE_REFERENCE="$HERE/affine_module_reference.c"
KERNEL_SOURCE="$HERE/algebraic_oblivious_affine_phase.c"
KERNEL_REFERENCE="$HERE/oblivious_affine_reference.c"
BASE_SOURCE="$HERE/algebraic_series_parallel_phase.c"
OUT=${1:-}
if [[ -z "$OUT" ]]; then
    OUT=$(mktemp -d /tmp/dag-affine-fanout.XXXXXX)
else
    mkdir -p "$OUT"
fi

for tool in cc jq sha256sum diff cmp awk sed stat strace rg; do
    command -v "$tool" >/dev/null
done

COMMON=(-std=c11 -O2 -Wall -Wextra -Wpedantic -Werror)

cc "${COMMON[@]}" -fstack-usage \
    "$PHASE_SOURCE" -lm -o "$OUT/dag_phase"
cc "${COMMON[@]}" -fstack-usage \
    "$REFERENCE_SOURCE" -lm -o "$OUT/dag_reference"
cc "${COMMON[@]}" -fanalyzer \
    "$PHASE_SOURCE" -lm -o "$OUT/dag_analyzer"

"$OUT/dag_phase" "$DAG_MANIFEST" "$DUPLICATE_MANIFEST" \
    >"$OUT/phase.jsonl"
"$OUT/dag_phase" "$DAG_MANIFEST" "$DUPLICATE_MANIFEST" \
    >"$OUT/phase_replay.jsonl"
"$OUT/dag_reference" "$DAG_MANIFEST" >"$OUT/reference.jsonl"
"$OUT/dag_analyzer" "$DAG_MANIFEST" "$DUPLICATE_MANIFEST" \
    >"$OUT/analyzer.jsonl"

cmp "$OUT/phase.jsonl" "$OUT/phase_replay.jsonl"
cmp "$OUT/phase.jsonl" "$OUT/analyzer.jsonl"
jq -e . "$OUT/phase.jsonl" >/dev/null
jq -e . "$OUT/reference.jsonl" >/dev/null

jq -c '
  select(.mode | startswith("affine-dag"))
  | {boundary_coefficients, boundary_fnv1a64}
' "$OUT/phase.jsonl" >"$OUT/phase_boundaries.jsonl"
jq -c '
  {boundary_coefficients, boundary_fnv1a64}
' "$OUT/reference.jsonl" >"$OUT/reference_boundaries.jsonl"
cmp "$OUT/phase_boundaries.jsonl" "$OUT/reference_boundaries.jsonl"
[[ $(wc -l <"$OUT/phase_boundaries.jsonl") -eq 5 ]]
[[ $(wc -l <"$OUT/reference_boundaries.jsonl") -eq 5 ]]

jq -se '
  [.[] | select(.mode | startswith("affine-dag"))] as $runs
  | ($runs | length) == 5
  and all($runs[];
    .claim
      == "BOUNDED_NATIVE_PRODUCED_AFFINE_DAG_SHARED_RESIDENT_MESSAGE_FANOUT"
    and .width == 3
    and .manifest_nodes == 8
    and .manifest_leaves == 4
    and .compose_nodes == 3
    and .intersect_nodes == 1
    and .dag_edges == 8
    and .dag_depth == 3
    and .fanout_nodes == 1
    and .maximum_fanout == 2
    and .compiled_postorder_valid == true
    and .derived_nominal_signatures == true
    and .address_assigned_after_validation == true
    and .live_set_scheduler == true
    and .working_relation_slots == 8
    and .maximum_live_relation_slots == 8
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
    and .snapshot_loaded == false
    and .restoration_tolerance == 2e-12
    and .restoration_max_abs <= .restoration_tolerance
    and .carrier_integrity_max_abs <= .restoration_tolerance
    and .workspace_cleared == true
    and .allocator_restored == true
    and .edge_custody_restored == true
    and .shared_receipt_reset == true
    and .pending_counts_restored == true
    and .scheduler_queue_empty == true
    and .outstanding_relation_leases == 0
    and .lease_allocations == 8
    and .lease_releases == 8
    and .dirty_control_releases == 0
    and .forward_native_operations == 4
    and .inverse_native_operations == 4
    and .native_operation_calls == 8
    and .forward_leaf_encodes == 4
    and .inverse_leaf_encodes == 4
    and .shared_forward_consumptions == 2
    and .shared_inverse_consumptions == 2
    and .shared_forward_materializations == 1
    and .shared_inverse_applications == 1
    and .shared_owner_allocations == 1
    and .shared_owner_releases == 1
    and .shared_resident_blocks == 1
    and .shared_consumer_distinct_slots == 1
    and .shared_consumer_distinct_serials == 1
    and .shared_live_at_projection == true
    and .shared_live_after_root_inverse == true
    and .boundary_block_copy_calls == 2
    and .intermediate_block_copy_calls == 0
    and .serial_after == (.serial_before + 8)
    and .restoration_generation_after
      == (.restoration_generation_before + 1)
  )
' "$OUT/phase.jsonl" >/dev/null

primary_hash=$(jq -r '
  select(.mode == "affine-dag-primary") | .boundary_fnv1a64
' "$OUT/phase.jsonl")
reuse_hash=$(jq -r '
  select(.mode == "affine-dag-reuse") | .boundary_fnv1a64
' "$OUT/phase.jsonl")
[[ "$primary_hash" == "630132cdcd942021" ]]
[[ "$reuse_hash" == "6f46809cf9d2cb8e" ]]
[[ "$primary_hash" != "$reuse_hash" ]]

jq -e '
  select(.mode == "snapshot-baseline")
  | .snapshot_loaded == true
    and .boundary_fnv1a64 == "630132cdcd942021"
    and .restoration_max_abs == 0
    and .allocator_restored == true
    and .edge_custody_restored == true
    and .shared_receipt_reset == true
    and .native_operation_calls == 4
    and .inverse_native_operations == 0
    and .serial_before == .serial_after
    and .restoration_generation_before
      == .restoration_generation_after
' "$OUT/phase.jsonl" >/dev/null

jq -e '
  select(.mode == "duplicate-tree-phase-baseline")
  | .manifest_nodes == 11
    and .manifest_leaves == 6
    and .compose_nodes == 4
    and .intersect_nodes == 1
    and .dag_edges == 10
    and .fanout_nodes == 0
    and .maximum_fanout == 1
    and .working_relation_slots == 11
    and .maximum_live_relation_slots == 11
    and .physical_relation_blocks_including_boundary == 12
    and .carrier_cells == 947
    and .live_carrier_bytes == 30304
    and .boundary_fnv1a64 == "630132cdcd942021"
    and .restoration_max_abs <= .restoration_tolerance
    and .allocator_restored == true
    and .edge_custody_restored == true
    and .shared_receipt_reset == true
    and .forward_native_operations == 5
    and .inverse_native_operations == 5
    and .native_operation_calls == 10
    and .forward_leaf_encodes == 6
    and .inverse_leaf_encodes == 6
    and .shared_forward_materializations == 0
    and .shared_owner_allocations == 0
    and .shared_owner_releases == 0
    and .shared_resident_blocks == 0
    and .intermediate_block_copy_calls == 0
' "$OUT/phase.jsonl" >/dev/null

jq -e '
  select(.mode == "controls")
  | .primary_reuse_boundary_different == true
    and .left_branch_essential == true
    and .right_branch_essential == true
    and .empty_intersection_control == true
    and .coefficient_oblivious_schedule == true
    and .wrong_root_inverse == true
    and .missing_root_inverse == true
    and .duplicate_tree_boundary_equal == true
    and .duplicate_tree_restored == true
    and .duplicate_tree_fails_shared_predicate == true
    and .snapshot_separate_weaker_path == true
    and .snapshot_post_projection_inverse_operations == 0
    and .accepted_working_slots == 8
    and .duplicate_tree_working_slots == 11
    and .relation_slot_reduction == 3
    and .accepted_native_operation_calls == 8
    and .duplicate_tree_native_operation_calls == 10
    and .accepted_leaf_encode_calls == 8
    and .duplicate_tree_leaf_encode_calls == 12
    and .wrong_root_restoration_abs > 1e-6
    and .missing_root_restoration_abs > 1e-6
    and .same_carrier_transactions == 17
    and .repeated_reuse_max_restoration_abs <= 2e-12
' "$OUT/phase.jsonl" >/dev/null

controls=(
  --project-intermediate
  --project-controls
  --null-carrier
  --control-copy-shared
  --control-cycle
  --control-type
  --control-disconnected
  --control-stale-shared
  --control-skip-consumer
  --control-double-consume
  --reordered-inverse
)
for control in "${controls[@]}"; do
    name=${control#--}
    set +e
    "$OUT/dag_phase" \
        "$DAG_MANIFEST" "$DUPLICATE_MANIFEST" "$control" \
        >"$OUT/control_${name}.stdout" \
        2>"$OUT/control_${name}.stderr"
    control_rc=$?
    set -e
    [[ $control_rc -ne 0 ]]
    [[ ! -s "$OUT/control_${name}.stdout" ]]
    [[ -s "$OUT/control_${name}.stderr" ]]
done
rg -q 'contains cycle' "$OUT/control_control-cycle.stderr"
rg -q 'nominal type mismatch' "$OUT/control_control-type.stderr"
rg -q 'disconnected' "$OUT/control_control-disconnected.stderr"
rg -q 'substituted a shared lease' \
    "$OUT/control_control-stale-shared.stderr"
rg -q 'skipped a declared consumer edge' \
    "$OUT/control_control-skip-consumer.stderr"
rg -q 'consumed more than once' \
    "$OUT/control_control-double-consume.stderr"
rg -q 'substituted a shared lease' \
    "$OUT/control_control-copy-shared.stderr"
rg -q 'live dependents' "$OUT/control_reordered-inverse.stderr"

set +e
"$OUT/dag_phase" "$DUPLICATE_MANIFEST" "$DUPLICATE_MANIFEST" \
    >"$OUT/control_duplicate_as_accepted.stdout" \
    2>"$OUT/control_duplicate_as_accepted.stderr"
duplicate_as_accepted_rc=$?
set -e
[[ $duplicate_as_accepted_rc -ne 0 ]]
[[ ! -s "$OUT/control_duplicate_as_accepted.stdout" ]]
rg -q 'exactly one bounded fanout' \
    "$OUT/control_duplicate_as_accepted.stderr"

strace -qq -f \
    -e trace=%file,%network,%ipc,%memory,write,writev,pwrite64,pwritev,pwritev2,memfd_create,sendfile,splice,tee,vmsplice,copy_file_range,process_vm_writev,ptrace,ioctl \
    -o "$OUT/no_smuggle.strace" \
    "$OUT/dag_phase" "$DAG_MANIFEST" "$DUPLICATE_MANIFEST" \
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
    -DDC_REUSE_CYCLES=0U \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    "$PHASE_SOURCE" -lm -o "$OUT/dag_sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$OUT/dag_sanitized" "$DAG_MANIFEST" "$DUPLICATE_MANIFEST" \
    >"$OUT/sanitized.jsonl" 2>"$OUT/sanitized.stderr"
[[ ! -s "$OUT/sanitized.stderr" ]]
jq -e . "$OUT/sanitized.jsonl" >/dev/null

: >"$OUT/scaling.jsonl"
compiler_builds=4
for width in 3 4 8 12 16; do
    cc "${COMMON[@]}" -fstack-usage \
        -DDC_WIDTH="${width}U" \
        -DDC_REUSE_CYCLES=0U \
        -DDC_SCALE_ONLY=1 \
        "$PHASE_SOURCE" -lm -o "$OUT/phase_w${width}"
    cc "${COMMON[@]}" \
        -DDF_WIDTH="${width}U" \
        "$REFERENCE_SOURCE" -lm -o "$OUT/reference_w${width}"
    compiler_builds=$((compiler_builds + 2))
    "$OUT/phase_w${width}" "$DAG_MANIFEST" "$DUPLICATE_MANIFEST" \
        >"$OUT/phase_w${width}.jsonl"
    "$OUT/reference_w${width}" "$DAG_MANIFEST" \
        >"$OUT/reference_w${width}.jsonl"
    jq -c '
      {boundary_coefficients, boundary_fnv1a64}
    ' "$OUT/phase_w${width}.jsonl" \
        >"$OUT/phase_w${width}.boundaries.jsonl"
    jq -c '
      {boundary_coefficients, boundary_fnv1a64}
    ' "$OUT/reference_w${width}.jsonl" \
        >"$OUT/reference_w${width}.boundaries.jsonl"
    cmp \
        "$OUT/phase_w${width}.boundaries.jsonl" \
        "$OUT/reference_w${width}.boundaries.jsonl"
    jq -e --argjson width "$width" '
      all(.
        ; .width == $width
          and .manifest_nodes == 8
          and .fanout_nodes == 1
          and .maximum_fanout == 2
          and .working_relation_slots == 8
          and .maximum_live_relation_slots == 8
          and .physical_relation_blocks_including_boundary == 9
          and .relation_cells
            == (4 * $width * $width + 4 * $width + 1)
          and .workspace_cells
            == (36 * $width * $width + 11 * $width + 2)
          and .carrier_cells
            == (72 * $width * $width + 47 * $width + 11)
          and .native_operation_calls == 8
          and .phase_ands
            == (
              4 * (
                288 * $width * $width * $width
                + 225 * $width * $width
                - 9 * $width
              )
            )
          and .phase_xors
            == (
              8 * (
                288 * $width * $width * $width
                + 96 * $width * $width
                - 42 * $width
              )
            )
          and .shared_forward_consumptions == 2
          and .shared_inverse_consumptions == 2
          and .shared_forward_materializations == 1
          and .shared_inverse_applications == 1
          and .shared_owner_allocations == 1
          and .shared_owner_releases == 1
          and .shared_resident_blocks == 1
          and .shared_consumer_distinct_slots == 1
          and .shared_consumer_distinct_serials == 1
          and .restoration_max_abs <= .restoration_tolerance
          and .allocator_restored == true
          and .edge_custody_restored == true
          and .outstanding_relation_leases == 0
      )
    ' "$OUT/phase_w${width}.jsonl" >/dev/null
    jq -c '
      select(.mode == "affine-dag-primary")
      | {
          width,
          relation_cells,
          workspace_cells,
          carrier_cells,
          live_carrier_bytes,
          carrier_and_verification_snapshot_heap_bytes,
          compiled_topology_bytes_current_abi,
          program_storage_bytes_current_abi,
          active_edge_descriptor_bytes,
          custody_state_bytes_current_abi,
          boundary_storage_bytes_current_abi,
          phase_ands,
          phase_xors,
          phase_cell_updates,
          native_phase_carrier_reads,
          logical_carrier_cell_inspections,
          restoration_max_abs,
          boundary_fnv1a64
        }
    ' "$OUT/phase_w${width}.jsonl" >>"$OUT/scaling.jsonl"
done

phase_hash=$(sha256sum "$PHASE_SOURCE" | awk '{print $1}')
reference_hash=$(sha256sum "$REFERENCE_SOURCE" | awk '{print $1}')
dag_manifest_hash=$(sha256sum "$DAG_MANIFEST" | awk '{print $1}')
duplicate_manifest_hash=$(
    sha256sum "$DUPLICATE_MANIFEST" | awk '{print $1}'
)
recursive_phase_hash=$(sha256sum "$RECURSIVE_PHASE" | awk '{print $1}')
recursive_reference_hash=$(
    sha256sum "$RECURSIVE_REFERENCE" | awk '{print $1}'
)
module_hash=$(sha256sum "$MODULE_SOURCE" | awk '{print $1}')
module_reference_hash=$(sha256sum "$MODULE_REFERENCE" | awk '{print $1}')
kernel_hash=$(sha256sum "$KERNEL_SOURCE" | awk '{print $1}')
kernel_reference_hash=$(sha256sum "$KERNEL_REFERENCE" | awk '{print $1}')
base_hash=$(sha256sum "$BASE_SOURCE" | awk '{print $1}')
qualifier_hash=$(sha256sum "$0" | awk '{print $1}')
result_hash=$(sha256sum "$OUT/phase.jsonl" | awk '{print $1}')
reference_result_hash=$(
    sha256sum "$OUT/reference.jsonl" | awk '{print $1}'
)
scaling_hash=$(sha256sum "$OUT/scaling.jsonl" | awk '{print $1}')
trace_hash=$(sha256sum "$OUT/no_smuggle.strace" | awk '{print $1}')

default_stack_file="$OUT/dag_phase-dag_affine_fanout_phase.su"
if [[ ! -f "$default_stack_file" ]]; then
    exit 1
fi
stack_frame_from() {
    local stack_file=$1
    local function_name=$2
    awk -F '\t' -v name="$function_name" '
      $1 ~ (":" name "$") {print $2; found=1; exit}
      END {if (!found) print 0}
    ' "$stack_file"
}
stack_chain_from() {
    local stack_file=$1
    local main_stack
    local execute_stack
    local forward_stack
    local reverse_stack
    local apply_stack
    local compose_stack
    local reduce_stack
    local swap_stack
    main_stack=$(stack_frame_from "$stack_file" main)
    execute_stack=$(stack_frame_from "$stack_file" dc_execute)
    forward_stack=$(stack_frame_from "$stack_file" dc_forward)
    reverse_stack=$(stack_frame_from "$stack_file" dc_reverse_one)
    apply_stack=$(stack_frame_from "$stack_file" rc_apply_node)
    compose_stack=$(
        stack_frame_from "$stack_file" ga_apply_compose.part.0
    )
    reduce_stack=$(stack_frame_from "$stack_file" ga_reduce_stage)
    swap_stack=$(
        stack_frame_from \
            "$stack_file" ga_conditional_swap_rows.part.0
    )
    printf '%s\n' "$((\
      main_stack + execute_stack + forward_stack + reverse_stack \
      + apply_stack + compose_stack + reduce_stack + swap_stack\
    ))"
}
conservative_stack=$(stack_chain_from "$default_stack_file")
w16_stack_file=$(
    find "$OUT" -maxdepth 1 \
        -name 'phase_w16-dag_affine_fanout_phase.su' | head -1
)
if [[ -z "$w16_stack_file" ]]; then
    exit 1
fi
w16_conservative_stack=$(stack_chain_from "$w16_stack_file")

phase_binary_bytes=$(stat -c '%s' "$OUT/dag_phase")
reference_binary_bytes=$(stat -c '%s' "$OUT/dag_reference")
accepted_stdout_bytes=$(stat -c '%s' "$OUT/phase.jsonl")
reference_stdout_bytes=$(stat -c '%s' "$OUT/reference.jsonl")
dag_manifest_bytes=$(stat -c '%s' "$DAG_MANIFEST")
duplicate_manifest_bytes=$(stat -c '%s' "$DUPLICATE_MANIFEST")

jq -n \
    --arg result PASS \
    --arg evidence_directory "$OUT" \
    --arg phase_hash "$phase_hash" \
    --arg reference_hash "$reference_hash" \
    --arg dag_manifest_hash "$dag_manifest_hash" \
    --arg duplicate_manifest_hash "$duplicate_manifest_hash" \
    --arg recursive_phase_hash "$recursive_phase_hash" \
    --arg recursive_reference_hash "$recursive_reference_hash" \
    --arg module_hash "$module_hash" \
    --arg module_reference_hash "$module_reference_hash" \
    --arg kernel_hash "$kernel_hash" \
    --arg kernel_reference_hash "$kernel_reference_hash" \
    --arg base_hash "$base_hash" \
    --arg qualifier_hash "$qualifier_hash" \
    --arg result_hash "$result_hash" \
    --arg reference_result_hash "$reference_result_hash" \
    --arg scaling_hash "$scaling_hash" \
    --arg trace_hash "$trace_hash" \
    --arg primary_hash "$primary_hash" \
    --arg reuse_hash "$reuse_hash" \
    --argjson compiler_builds "$compiler_builds" \
    --argjson conservative_stack "$conservative_stack" \
    --argjson w16_conservative_stack "$w16_conservative_stack" \
    --argjson phase_binary_bytes "$phase_binary_bytes" \
    --argjson reference_binary_bytes "$reference_binary_bytes" \
    --argjson accepted_stdout_bytes "$accepted_stdout_bytes" \
    --argjson reference_stdout_bytes "$reference_stdout_bytes" \
    --argjson dag_manifest_bytes "$dag_manifest_bytes" \
    --argjson duplicate_manifest_bytes "$duplicate_manifest_bytes" \
    --argjson maximum_reuse_error "$(
      jq '
        select(.mode == "controls")
        | .repeated_reuse_max_restoration_abs
      ' "$OUT/phase.jsonl"
    )" \
    --argjson width16 "$(
      jq -s 'map(select(.width == 16))[0]' "$OUT/scaling.jsonl"
    )" \
    '{
      result: $result,
      claim:
        "BOUNDED_NATIVE_PRODUCED_AFFINE_DAG_SHARED_RESIDENT_MESSAGE_FANOUT",
      claim_ceiling:
        "BOUNDED_WIDTH16_SOFTWARE_GF2_SINGLE_FANOUT_AFFINE_DAG_REFERENCE_ONLY",
      evidence_directory: $evidence_directory,
      compiler_builds: $compiler_builds,
      sanitizers: "PASS",
      analyzer: "PASS",
      deterministic_replay: "PASS",
      mandatory_expanded_no_smuggle_strace: "PASS",
      tested_widths: [3,4,8,12,16],
      exact_phase_reference_complete_boundaries: 25,
      public_dag: {
        nodes: 8,
        leaves: 4,
        compose_nodes: 3,
        intersect_nodes: 1,
        edges: 8,
        depth: 3,
        fanout_nodes: 1,
        maximum_fanout: 2,
        shared_native_producer_forward_materializations: 1,
        shared_native_producer_inverse_applications: 1,
        shared_forward_edge_consumptions: 2,
        shared_inverse_edge_consumptions: 2,
        shared_resident_blocks: 1,
        intermediate_block_copies: 0
      },
      matched_duplicate_tree: {
        nodes: 11,
        leaves: 6,
        operators: 5,
        fanout_nodes: 0,
        native_calls: 10,
        leaf_encode_calls: 12,
        same_complete_boundary: true,
        restored_by_actual_inverse: true,
        fails_shared_message_predicate: true
      },
      default_width3: {
        relation_cells: 49,
        workspace_cells: 359,
        carrier_cells: 800,
        live_carrier_bytes: 25600,
        duplicate_tree_carrier_cells: 947,
        duplicate_tree_live_carrier_bytes: 30304,
        primary_boundary_fnv1a64: $primary_hash,
        reuse_boundary_fnv1a64: $reuse_hash,
        same_carrier_transactions: 17,
        maximum_reuse_restoration_error: $maximum_reuse_error,
        conservative_compiler_measured_stack_bytes: $conservative_stack,
        conservative_accounted_execution_bytes_current_abi:
          (
            51200 + 2536 + 1960 + 216
            + $conservative_stack
            + $dag_manifest_bytes + $duplicate_manifest_bytes
          )
      },
      maximum_width16:
        (
          $width16
          + {
              conservative_compiler_measured_stack_bytes:
                $w16_conservative_stack,
              conservative_accounted_execution_bytes_current_abi:
                (
                  $width16.carrier_and_verification_snapshot_heap_bytes
                  + $width16.compiled_topology_bytes_current_abi
                  + $width16.program_storage_bytes_current_abi
                  + $width16.boundary_storage_bytes_current_abi
                  + $w16_conservative_stack
                  + $dag_manifest_bytes
                  + $duplicate_manifest_bytes
                )
            }
        ),
      state_law: {
        exact_discrete_lease_and_edge_custody_reset: true,
        serial_postcondition: "serial_after=serial_before+allocations",
        restoration_generation_postcondition:
          "generation_after=generation_before+1",
        carrier_tolerance_predeclared: 2e-12,
        snapshot_loaded_on_accepted_path: false
      },
      additional_material_resources: {
        phase_binary_bytes: $phase_binary_bytes,
        reference_binary_bytes: $reference_binary_bytes,
        accepted_default_stdout_bytes: $accepted_stdout_bytes,
        reference_default_stdout_bytes: $reference_stdout_bytes,
        public_dag_manifest_bytes: $dag_manifest_bytes,
        duplicate_tree_manifest_bytes: $duplicate_manifest_bytes
      },
      controls: {
        dependency_reordered_producer_inverse_rejected: true,
        stale_shared_serial_rejected: true,
        skipped_consumer_rejected: true,
        duplicate_consumer_rejected: true,
        intermediate_copy_rejected: true,
        projection_and_null_rejected: true,
        wrong_and_missing_root_inverse_detected: true,
        left_and_right_branch_essential: true,
        snapshot_separate_weaker_path: true,
        duplicate_tree_output_equivalence_control: true,
        no_smuggle: true
      },
      sha256: {
        phase_source: $phase_hash,
        reference_source: $reference_hash,
        public_dag_manifest: $dag_manifest_hash,
        duplicate_tree_manifest: $duplicate_manifest_hash,
        recursive_phase_source: $recursive_phase_hash,
        recursive_reference_source: $recursive_reference_hash,
        mixed_module_source: $module_hash,
        mixed_module_reference: $module_reference_hash,
        phase_kernel_source: $kernel_hash,
        phase_kernel_reference: $kernel_reference_hash,
        base_source: $base_hash,
        qualifier_source: $qualifier_hash,
        result: $result_hash,
        reference: $reference_result_hash,
        scaling: $scaling_hash,
        no_smuggle_trace: $trace_hash
      },
      catvm_enforcement: false,
      multiple_fanout_nodes: false,
      arbitrary_dag_topology: false,
      arbitrary_graph_topology: false,
      physical_execution: false,
      performance_advantage: false,
      small_wall_crossed: false,
      unlimited_catalytic_computation: false,
      terminal: false
    }' >"$OUT/summary.json"

jq -e '.result == "PASS" and .terminal == false' \
    "$OUT/summary.json" >/dev/null
cat "$OUT/summary.json"
