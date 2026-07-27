#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PHASE_SOURCE="$HERE/multi_dag_affine_phase.c"
REFERENCE_SOURCE="$HERE/multi_dag_affine_reference.c"
DAG_SOURCE="$HERE/dag_affine_fanout_phase.c"
DAG_REFERENCE="$HERE/dag_affine_fanout_reference.c"
RECURSIVE_PHASE="$HERE/recursive_affine_module_phase.c"
RECURSIVE_REFERENCE="$HERE/recursive_affine_module_reference.c"
DAG_MANIFEST="$HERE/multi_dag_affine_topology.txt"
DUPLICATE_MANIFEST="$HERE/multi_dag_affine_duplicate_tree_topology.txt"
OUT=${1:-}
if [[ -z "$OUT" ]]; then
    OUT=$(mktemp -d /tmp/multi-dag-affine.XXXXXX)
else
    mkdir -p "$OUT"
fi

for tool in cc jq sha256sum cmp rg strace stat awk; do
    command -v "$tool" >/dev/null
done
COMMON=(-std=c11 -O2 -Wall -Wextra -Wpedantic -Werror)

cc "${COMMON[@]}" -fstack-usage \
    "$PHASE_SOURCE" -lm -o "$OUT/multi_phase"
cc "${COMMON[@]}" \
    "$REFERENCE_SOURCE" -lm -o "$OUT/multi_reference"
cc "${COMMON[@]}" -fanalyzer \
    "$PHASE_SOURCE" -lm -o "$OUT/multi_analyzer"

"$OUT/multi_phase" "$DAG_MANIFEST" "$DUPLICATE_MANIFEST" \
    >"$OUT/phase.jsonl"
"$OUT/multi_phase" "$DAG_MANIFEST" "$DUPLICATE_MANIFEST" \
    >"$OUT/phase_replay.jsonl"
"$OUT/multi_reference" "$DAG_MANIFEST" 2 \
    >"$OUT/reference.jsonl"
"$OUT/multi_reference" "$DUPLICATE_MANIFEST" 0 \
    >"$OUT/duplicate_reference.jsonl"
"$OUT/multi_analyzer" "$DAG_MANIFEST" "$DUPLICATE_MANIFEST" \
    >"$OUT/analyzer.jsonl"

cmp "$OUT/phase.jsonl" "$OUT/phase_replay.jsonl"
cmp "$OUT/phase.jsonl" "$OUT/analyzer.jsonl"
jq -e . "$OUT/phase.jsonl" >/dev/null
jq -e . "$OUT/reference.jsonl" >/dev/null
jq -c '
  select(
    .mode == "multi-dag-primary"
    or .mode == "multi-dag-reuse"
    or .mode == "multi-dag-semantic"
  )
  | {boundary_coefficients, boundary_fnv1a64}
' "$OUT/phase.jsonl" >"$OUT/phase_boundaries.jsonl"
jq -c '
  {boundary_coefficients, boundary_fnv1a64}
' "$OUT/reference.jsonl" >"$OUT/reference_boundaries.jsonl"
cmp "$OUT/phase_boundaries.jsonl" "$OUT/reference_boundaries.jsonl"
jq -c '
  select(.mode == "duplicate-tree-phase-baseline")
  | {boundary_coefficients, boundary_fnv1a64}
' "$OUT/phase.jsonl" >"$OUT/duplicate_phase_primary.jsonl"
head -n 1 "$OUT/duplicate_reference.jsonl" \
    | jq -c '{boundary_coefficients, boundary_fnv1a64}' \
    >"$OUT/duplicate_reference_primary.jsonl"
cmp \
    "$OUT/duplicate_phase_primary.jsonl" \
    "$OUT/duplicate_reference_primary.jsonl"

jq -se '
  [.[] | select(
    .mode == "multi-dag-primary"
    or .mode == "multi-dag-reuse"
    or .mode == "multi-dag-semantic"
  )] as $runs
  | ($runs | length) == 5
  and all($runs[];
    .claim
      == "BOUNDED_NESTED_AFFINE_DAG_MULTI_SHARED_RESIDENT_MESSAGE_CUSTODY"
    and .width == 3
    and .manifest_nodes == 7
    and .manifest_leaves == 3
    and .compose_nodes == 3
    and .intersect_nodes == 1
    and .dag_edges == 8
    and .dag_depth == 4
    and .fanout_nodes == 2
    and .maximum_fanout == 2
    and .working_relation_slots == 7
    and .maximum_live_relation_slots == 7
    and .physical_relation_blocks_including_boundary == 8
    and .relation_cells == 49
    and .workspace_cells == 359
    and .carrier_cells == 751
    and .live_carrier_bytes == 24032
    and .snapshot_loaded == false
    and .decoded_intermediate_coefficients == 0
    and .serialized_intermediate_coefficients == 0
    and .intermediate_relation_hashes == 0
    and .pre_inverse_carrier_aggregate_outputs == 0
    and .host_selected_pivots == 0
    and .tuple_slots == 0
    and .assignment_slots == 0
    and .witness_slots == 0
    and .content_dependent_receipts == 0
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
    and .lease_allocations == 7
    and .lease_releases == 7
    and .dirty_control_releases == 0
    and .forward_native_operations == 4
    and .inverse_native_operations == 4
    and .native_operation_calls == 8
    and .forward_leaf_encodes == 3
    and .inverse_leaf_encodes == 3
    and .shared_forward_consumptions == 4
    and .shared_inverse_consumptions == 4
    and .shared_forward_materializations == 2
    and .shared_inverse_applications == 2
    and .shared_owner_allocations == 2
    and .shared_owner_releases == 2
    and .shared_resident_blocks == 2
    and .shared_consumer_distinct_slots == 1
    and .shared_consumer_distinct_serials == 1
    and .shared_live_at_projection == true
    and .shared_live_after_root_inverse == true
    and .shared_pair_nonalias_observed == true
    and .shared_overlap_observed == true
    and .shared_simultaneous_live_high_water == 2
    and .boundary_block_copy_calls == 2
    and .intermediate_block_copy_calls == 0
    and .serial_after == (.serial_before + 7)
    and .restoration_generation_after
      == (.restoration_generation_before + 1)
    and (.shared_owner_receipts | length) == 2
    and (
      [.shared_owner_receipts[].public_node_id] | sort
    ) == [704,705]
    and all(.shared_owner_receipts[];
      .declared_consumers == 2
      and .forward_consumptions == 2
      and .inverse_consumptions == 2
      and .forward_materializations == 1
      and .inverse_applications == 1
      and .owner_allocations == 1
      and .owner_releases == 1
      and .peak_live_instances == 1
      and .consumer_distinct_slots == 1
      and .consumer_distinct_serials == 1
      and .live_at_projection == true
      and .live_after_root_inverse == true
      and .all_edge_generations_exact == true
      and .receipt_exact == true
    )
  )
' "$OUT/phase.jsonl" >/dev/null

primary_hash=$(jq -r '
  select(.mode == "multi-dag-primary") | .boundary_fnv1a64
' "$OUT/phase.jsonl")
reuse_hash=$(jq -r '
  select(.mode == "multi-dag-reuse") | .boundary_fnv1a64
' "$OUT/phase.jsonl")
[[ "$primary_hash" == "59f0245207f2a0f1" ]]
[[ "$reuse_hash" == "6f46809cf9d2cb8e" ]]
[[ "$primary_hash" != "$reuse_hash" ]]

jq -e '
  select(.mode == "snapshot-baseline")
  | .snapshot_loaded == true
    and .boundary_fnv1a64 == "59f0245207f2a0f1"
    and .restoration_max_abs == 0
    and .inverse_native_operations == 0
    and .serial_before == .serial_after
    and .restoration_generation_before
      == .restoration_generation_after
' "$OUT/phase.jsonl" >/dev/null
jq -e '
  select(.mode == "duplicate-tree-phase-baseline")
  | .manifest_nodes == 15
    and .manifest_leaves == 8
    and .compose_nodes == 6
    and .intersect_nodes == 1
    and .dag_edges == 14
    and .dag_depth == 4
    and .fanout_nodes == 0
    and .maximum_fanout == 1
    and .working_relation_slots == 15
    and .physical_relation_blocks_including_boundary == 16
    and .carrier_cells == 1143
    and .live_carrier_bytes == 36576
    and .boundary_fnv1a64 == "59f0245207f2a0f1"
    and .restoration_max_abs <= .restoration_tolerance
    and .allocator_restored == true
    and .native_operation_calls == 14
    and .forward_leaf_encodes == 8
    and .inverse_leaf_encodes == 8
    and .shared_forward_materializations == 0
    and .shared_owner_allocations == 0
    and .shared_resident_blocks == 0
    and .intermediate_block_copy_calls == 0
' "$OUT/phase.jsonl" >/dev/null
jq -e '
  select(.mode == "multi-dag-controls")
  | .nested_same_typed_shared_dependency == true
    and .nested_lifetime_order_observed == true
    and .all_shared_receipts_exact == true
    and .primary_reuse_boundary_different == true
    and .h_branch_essential == true
    and .g_branch_essential == true
    and .empty_intersection_control == true
    and .coefficient_oblivious_schedule == true
    and .wrong_root_inverse == true
    and .missing_root_inverse == true
    and .duplicate_tree_boundary_equal == true
    and .duplicate_tree_restored == true
    and .duplicate_tree_fails_shared_predicate == true
    and .snapshot_separate_weaker_path == true
    and .accepted_working_slots == 7
    and .duplicate_tree_working_slots == 15
    and .relation_slot_reduction == 8
    and .accepted_native_operation_calls == 8
    and .duplicate_tree_native_operation_calls == 14
    and .accepted_leaf_encode_calls == 6
    and .duplicate_tree_leaf_encode_calls == 16
    and .wrong_root_restoration_abs > 1e-6
    and .missing_root_restoration_abs > 1e-6
    and .same_carrier_transactions == 17
    and .repeated_reuse_max_restoration_abs <= 2e-12
' "$OUT/phase.jsonl" >/dev/null

controls=(
  --project-s
  --project-t
  --project-controls
  --null-carrier
  --control-copy-s
  --control-copy-t
  --control-cross-s-t
  --control-cross-t-s
  --control-stale-s
  --control-stale-t
  --control-skip-s
  --control-skip-t
  --control-double-s
  --control-double-t
  --reordered-inverse-s
  --reordered-inverse-t
  --control-cycle
  --control-type
  --control-disconnected
)
for control in "${controls[@]}"; do
    name=${control#--}
    set +e
    "$OUT/multi_phase" \
        "$DAG_MANIFEST" "$DUPLICATE_MANIFEST" "$control" \
        >"$OUT/control_${name}.stdout" \
        2>"$OUT/control_${name}.stderr"
    control_rc=$?
    set -e
    [[ $control_rc -ne 0 ]]
    [[ ! -s "$OUT/control_${name}.stdout" ]]
    [[ -s "$OUT/control_${name}.stderr" ]]
done
rg -q 'consumer substituted a shared lease' \
    "$OUT/control_control-cross-s-t.stderr"
rg -q 'consumer substituted a shared lease' \
    "$OUT/control_control-cross-t-s.stderr"
rg -q 'skipped a declared consumer edge' \
    "$OUT/control_control-skip-s.stderr"
rg -q 'skipped a declared consumer edge' \
    "$OUT/control_control-skip-t.stderr"
rg -q 'consumed more than once' \
    "$OUT/control_control-double-s.stderr"
rg -q 'consumed more than once' \
    "$OUT/control_control-double-t.stderr"
rg -q 'live dependents' "$OUT/control_reordered-inverse-s.stderr"
rg -q 'live dependents' "$OUT/control_reordered-inverse-t.stderr"

set +e
awk '
  $1 == "compose" && $2 == "707" {
    print "compose 707 706 708"
    next
  }
  {print}
  END {print "leaf 708 A A 2"}
' "$DAG_MANIFEST" >"$OUT/one_fanout_topology.txt"
"$OUT/multi_phase" \
    "$OUT/one_fanout_topology.txt" \
    "$DUPLICATE_MANIFEST" \
    >"$OUT/control_one_fanout.stdout" \
    2>"$OUT/control_one_fanout.stderr"
one_fanout_rc=$?
"$OUT/multi_phase" "$DUPLICATE_MANIFEST" "$DUPLICATE_MANIFEST" \
    >"$OUT/control_zero_fanout.stdout" \
    2>"$OUT/control_zero_fanout.stderr"
zero_fanout_rc=$?
set -e
[[ $one_fanout_rc -ne 0 ]]
[[ $zero_fanout_rc -ne 0 ]]
[[ ! -s "$OUT/control_one_fanout.stdout" ]]
[[ ! -s "$OUT/control_zero_fanout.stdout" ]]
rg -q 'wrong bounded fanout-node count' \
    "$OUT/control_one_fanout.stderr"
rg -q 'wrong bounded fanout-node count' \
    "$OUT/control_zero_fanout.stderr"

[[ $(rg -n 'ga_copy_block\(' "$DAG_SOURCE" | wc -l) -eq 1 ]]
if rg -q 'ga_copy_block\(' "$PHASE_SOURCE"; then
    exit 1
fi
strace -qq -f \
    -e trace=%file,%network,%ipc,%memory,write,writev,pwrite64,pwritev,pwritev2,memfd_create,sendfile,splice,tee,vmsplice,copy_file_range,process_vm_writev,ptrace,ioctl \
    -o "$OUT/no_smuggle.strace" \
    "$OUT/multi_phase" "$DAG_MANIFEST" "$DUPLICATE_MANIFEST" \
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
    -DMF_REUSE_CYCLES=0U \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    "$PHASE_SOURCE" -lm -o "$OUT/multi_sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$OUT/multi_sanitized" "$DAG_MANIFEST" "$DUPLICATE_MANIFEST" \
    >"$OUT/sanitized.jsonl" 2>"$OUT/sanitized.stderr"
[[ ! -s "$OUT/sanitized.stderr" ]]
jq -e . "$OUT/sanitized.jsonl" >/dev/null

: >"$OUT/scaling.jsonl"
for width in 3 4 8 12 16; do
    cc "${COMMON[@]}" \
        -DMF_WIDTH="${width}U" \
        -DMF_REUSE_CYCLES=0U \
        -DMF_SCALE_ONLY=1 \
        "$PHASE_SOURCE" -lm -o "$OUT/phase_w${width}"
    cc "${COMMON[@]}" \
        -DMF_REF_WIDTH="${width}U" \
        "$REFERENCE_SOURCE" -lm -o "$OUT/reference_w${width}"
    "$OUT/phase_w${width}" "$DAG_MANIFEST" "$DUPLICATE_MANIFEST" \
        >"$OUT/phase_w${width}.jsonl"
    "$OUT/reference_w${width}" "$DAG_MANIFEST" 2 \
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
    jq -e --argjson w "$width" '
      all(.
        ; .width == $w
          and .manifest_nodes == 7
          and .fanout_nodes == 2
          and .working_relation_slots == 7
          and .maximum_live_relation_slots == 7
          and .physical_relation_blocks_including_boundary == 8
          and .relation_cells == (4*$w*$w + 4*$w + 1)
          and .workspace_cells == (36*$w*$w + 11*$w + 2)
          and .carrier_cells == (68*$w*$w + 43*$w + 10)
          and .native_operation_calls == 8
          and .phase_ands
            == (
              4 * (
                288*$w*$w*$w + 225*$w*$w - 9*$w
              )
            )
          and .phase_xors
            == (
              8 * (
                288*$w*$w*$w + 96*$w*$w - 42*$w
              )
            )
          and .shared_forward_consumptions == 4
          and .shared_inverse_consumptions == 4
          and .shared_forward_materializations == 2
          and .shared_inverse_applications == 2
          and .shared_resident_blocks == 2
          and .shared_simultaneous_live_high_water == 2
          and all(.shared_owner_receipts[]; .receipt_exact == true)
          and .restoration_max_abs <= .restoration_tolerance
          and .allocator_restored == true
          and .edge_custody_restored == true
      )
    ' "$OUT/phase_w${width}.jsonl" >/dev/null
    jq -c '
      select(.mode == "multi-dag-primary")
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
dag_source_hash=$(sha256sum "$DAG_SOURCE" | awk '{print $1}')
dag_reference_hash=$(sha256sum "$DAG_REFERENCE" | awk '{print $1}')
recursive_phase_hash=$(sha256sum "$RECURSIVE_PHASE" | awk '{print $1}')
recursive_reference_hash=$(
    sha256sum "$RECURSIVE_REFERENCE" | awk '{print $1}'
)
manifest_hash=$(sha256sum "$DAG_MANIFEST" | awk '{print $1}')
duplicate_hash=$(sha256sum "$DUPLICATE_MANIFEST" | awk '{print $1}')
qualifier_hash=$(sha256sum "$0" | awk '{print $1}')
result_hash=$(sha256sum "$OUT/phase.jsonl" | awk '{print $1}')
reference_result_hash=$(sha256sum "$OUT/reference.jsonl" | awk '{print $1}')
trace_hash=$(sha256sum "$OUT/no_smuggle.strace" | awk '{print $1}')

jq -n \
    --arg evidence_directory "$OUT" \
    --arg phase_hash "$phase_hash" \
    --arg reference_hash "$reference_hash" \
    --arg dag_source_hash "$dag_source_hash" \
    --arg dag_reference_hash "$dag_reference_hash" \
    --arg recursive_phase_hash "$recursive_phase_hash" \
    --arg recursive_reference_hash "$recursive_reference_hash" \
    --arg manifest_hash "$manifest_hash" \
    --arg duplicate_hash "$duplicate_hash" \
    --arg qualifier_hash "$qualifier_hash" \
    --arg result_hash "$result_hash" \
    --arg reference_result_hash "$reference_result_hash" \
    --arg trace_hash "$trace_hash" \
    --arg primary_hash "$primary_hash" \
    --arg reuse_hash "$reuse_hash" \
    --argjson maximum_reuse_error "$(
      jq '
        select(.mode == "multi-dag-controls")
        | .repeated_reuse_max_restoration_abs
      ' "$OUT/phase.jsonl"
    )" \
    --argjson width16 "$(
      jq -s 'map(select(.width == 16))[0]' "$OUT/scaling.jsonl"
    )" \
    '{
      result: "PASS",
      claim:
        "BOUNDED_NESTED_AFFINE_DAG_MULTI_SHARED_RESIDENT_MESSAGE_CUSTODY",
      claim_ceiling:
        "BOUNDED_WIDTH16_SOFTWARE_GF2_EXACTLY_TWO_NESTED_FANOUTS_REFERENCE_ONLY",
      evidence_directory: $evidence_directory,
      tested_widths: [3,4,8,12,16],
      exact_phase_reference_complete_boundaries: 25,
      deterministic_replay: "PASS",
      analyzer: "PASS",
      sanitizers: "PASS",
      mandatory_expanded_no_smuggle_strace: "PASS",
      public_dag: {
        nodes: 7,
        leaves: 3,
        operators: 4,
        edges: 8,
        depth: 4,
        fanout_nodes: 2,
        maximum_fanout: 2,
        nested_same_typed_producers: [704,705],
        per_owner_forward_and_inverse_consumers: 2,
        simultaneous_resident_blocks: 2,
        intermediate_block_copies: 0
      },
      occurrence_expanded_tree: {
        nodes: 15,
        leaves: 8,
        operators: 7,
        fanout_nodes: 0,
        native_calls: 14,
        leaf_encode_calls: 16,
        same_complete_boundary: true,
        actual_inverse_restored: true
      },
      default_width3: {
        carrier_cells: 751,
        live_carrier_bytes: 24032,
        duplicate_tree_carrier_cells: 1143,
        duplicate_tree_live_carrier_bytes: 36576,
        primary_boundary_fnv1a64: $primary_hash,
        reuse_boundary_fnv1a64: $reuse_hash,
        same_carrier_transactions: 17,
        maximum_reuse_restoration_error: $maximum_reuse_error
      },
      maximum_width16: $width16,
      state_law: {
        exact_discrete_owner_edge_pending_allocator_reset: true,
        serial_postcondition: "serial_after=serial_before+allocations",
        generation_postcondition:
          "generation_after=generation_before+1",
        carrier_tolerance: 2e-12,
        snapshot_loaded_on_accepted_path: false
      },
      controls: {
        same_typed_cross_substitution_both_directions_rejected: true,
        equal_content_clone_each_owner_rejected: true,
        stale_skip_double_each_owner_rejected: true,
        premature_inverse_each_owner_rejected: true,
        projection_and_null_rejected: true,
        wrong_and_missing_root_inverse_detected: true,
        one_and_zero_fanout_acceptance_rejected: true,
        snapshot_separate_weaker_path: true,
        occurrence_tree_equivalence: true,
        no_smuggle: true
      },
      sha256: {
        phase_source: $phase_hash,
        reference_source: $reference_hash,
        generalized_dag_source: $dag_source_hash,
        generalized_dag_reference: $dag_reference_hash,
        recursive_phase_source: $recursive_phase_hash,
        recursive_reference_source: $recursive_reference_hash,
        public_manifest: $manifest_hash,
        duplicate_manifest: $duplicate_hash,
        qualifier: $qualifier_hash,
        accepted_output: $result_hash,
        reference_output: $reference_result_hash,
        no_smuggle_trace: $trace_hash
      }
    }' | tee "$OUT/result.json"
