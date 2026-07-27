#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PHASE_SOURCE="$HERE/compact_multi_dag_affine_phase.c"
REFERENCE_SOURCE="$HERE/multi_dag_affine_reference.c"
DAG_SOURCE="$HERE/dag_affine_fanout_phase.c"
DAG_MANIFEST="$HERE/multi_dag_affine_topology.txt"
TREE_MANIFEST="$HERE/multi_dag_affine_duplicate_tree_topology.txt"
OUT=${1:-}
if [[ -z "$OUT" ]]; then
    OUT=$(mktemp -d /tmp/compact-multi-dag.XXXXXX)
else
    mkdir -p "$OUT"
fi
for tool in cc jq sha256sum cmp rg strace awk; do
    command -v "$tool" >/dev/null
done
COMMON=(-std=c11 -O2 -Wall -Wextra -Wpedantic -Werror)

cc "${COMMON[@]}" -fstack-usage \
    "$PHASE_SOURCE" -lm -o "$OUT/compact_phase"
cc "${COMMON[@]}" \
    "$REFERENCE_SOURCE" -lm -o "$OUT/reference"
cc "${COMMON[@]}" -fanalyzer \
    "$PHASE_SOURCE" -lm -o "$OUT/compact_analyzer"

"$OUT/compact_phase" "$DAG_MANIFEST" "$TREE_MANIFEST" \
    >"$OUT/phase.jsonl"
"$OUT/compact_phase" "$DAG_MANIFEST" "$TREE_MANIFEST" \
    >"$OUT/replay.jsonl"
"$OUT/compact_analyzer" "$DAG_MANIFEST" "$TREE_MANIFEST" \
    >"$OUT/analyzer.jsonl"
"$OUT/reference" "$DAG_MANIFEST" 2 >"$OUT/reference.jsonl"
cmp "$OUT/phase.jsonl" "$OUT/replay.jsonl"
cmp "$OUT/phase.jsonl" "$OUT/analyzer.jsonl"
jq -e . "$OUT/phase.jsonl" >/dev/null
jq -c '
  select(
    .mode == "compact-dag-primary"
    or .mode == "compact-dag-reuse"
    or .mode == "compact-dag-semantic"
  )
  | {boundary_coefficients, boundary_fnv1a64}
' "$OUT/phase.jsonl" >"$OUT/phase_boundaries.jsonl"
jq -c '
  {boundary_coefficients, boundary_fnv1a64}
' "$OUT/reference.jsonl" >"$OUT/reference_boundaries.jsonl"
cmp "$OUT/phase_boundaries.jsonl" "$OUT/reference_boundaries.jsonl"

jq -se '
  [.[] | select(
    .mode == "compact-dag-primary"
    or .mode == "compact-dag-reuse"
    or .mode == "compact-dag-semantic"
  )] as $runs
  | ($runs | length) == 5
  and all($runs[];
    .claim
      == "BOUNDED_NESTED_AFFINE_DAG_PINNED_SHARED_COMPACT_LEAF_PEBBLING"
    and .width == 3
    and .manifest_nodes == 7
    and .manifest_leaves == 3
    and .compose_nodes == 3
    and .intersect_nodes == 1
    and .dag_edges == 8
    and .dag_depth == 4
    and .fanout_nodes == 2
    and .maximum_fanout == 2
    and .working_relation_slots == 4
    and .maximum_live_relation_slots == 4
    and .physical_relation_blocks_including_boundary == 6
    and .relation_cells == 49
    and .workspace_cells == 359
    and .carrier_cells == 653
    and .live_carrier_bytes == 20896
    and .snapshot_loaded == false
    and .decoded_intermediate_coefficients == 0
    and .serialized_intermediate_coefficients == 0
    and .intermediate_relation_hashes == 0
    and .pre_inverse_carrier_aggregate_outputs == 0
    and .tuple_slots == 0
    and .assignment_slots == 0
    and .witness_slots == 0
    and .content_dependent_receipts == 0
    and .restoration_max_abs <= .restoration_tolerance
    and .carrier_integrity_max_abs <= .restoration_tolerance
    and .workspace_cleared == true
    and .allocator_restored == true
    and .edge_custody_restored == true
    and .shared_receipt_reset == true
    and .pending_counts_restored == true
    and .scheduler_queue_empty == true
    and .outstanding_relation_leases == 0
    and .lease_allocations == 10
    and .lease_releases == 10
    and .dirty_control_releases == 0
    and .forward_native_operations == 4
    and .inverse_native_operations == 4
    and .native_operation_calls == 8
    and .forward_leaf_encodes == 6
    and .inverse_leaf_encodes == 6
    and .shared_forward_consumptions == 4
    and .shared_inverse_consumptions == 4
    and .shared_forward_materializations == 2
    and .shared_inverse_applications == 2
    and .shared_owner_allocations == 2
    and .shared_owner_releases == 2
    and .shared_resident_blocks == 2
    and .shared_live_at_projection == true
    and .shared_live_after_root_inverse == true
    and .shared_pair_nonalias_observed == true
    and .shared_overlap_observed == true
    and .boundary_block_copy_calls == 2
    and .intermediate_block_copy_calls == 0
    and .serial_after == (.serial_before + 10)
    and .restoration_generation_after
      == (.restoration_generation_before + 1)
    and (.shared_owner_receipts | length) == 2
    and all(.shared_owner_receipts[]; .receipt_exact == true)
  )
' "$OUT/phase.jsonl" >/dev/null

jq -se '
  [.[] | select(.mode | endswith("-compact-schedule"))] as $runs
  | ($runs | length) == 5
  and all($runs[];
    .claim
      == "BOUNDED_NESTED_AFFINE_DAG_PINNED_SHARED_COMPACT_LEAF_PEBBLING"
    and .working_relation_slots == 4
    and .physical_relation_blocks_including_boundary == 6
    and .maximum_live_relation_slots == 4
    and .initial_leaf_materializations == 3
    and .early_leaf_unmaterializations == 3
    and .restoration_leaf_reconstructions == 3
    and .final_leaf_inverses == 3
    and .internal_node_rematerializations == 0
    and .operator_recomputations == 0
    and .leaf_obligations_restored == true
    and .residency_restored == true
    and .morphism_cursor_empty == true
    and .nested_lifetime_order_observed == true
  )
' "$OUT/phase.jsonl" >/dev/null

primary_hash=$(jq -r '
  select(.mode == "compact-dag-primary") | .boundary_fnv1a64
' "$OUT/phase.jsonl")
reuse_hash=$(jq -r '
  select(.mode == "compact-dag-reuse") | .boundary_fnv1a64
' "$OUT/phase.jsonl")
[[ "$primary_hash" == "59f0245207f2a0f1" ]]
[[ "$reuse_hash" == "6f46809cf9d2cb8e" ]]

jq -e '
  select(.mode == "retain-unique-phase-baseline")
  | .working_relation_slots == 7
    and .physical_relation_blocks_including_boundary == 8
    and .carrier_cells == 751
    and .native_operation_calls == 8
    and .forward_leaf_encodes == 3
    and .inverse_leaf_encodes == 3
    and .lease_allocations == 7
    and .lease_releases == 7
    and .boundary_fnv1a64 == "59f0245207f2a0f1"
    and .restoration_max_abs <= .restoration_tolerance
' "$OUT/phase.jsonl" >/dev/null
jq -e '
  select(.mode == "occurrence-tree-phase-baseline")
  | .working_relation_slots == 15
    and .physical_relation_blocks_including_boundary == 16
    and .carrier_cells == 1143
    and .native_operation_calls == 14
    and .forward_leaf_encodes == 8
    and .inverse_leaf_encodes == 8
    and .boundary_fnv1a64 == "59f0245207f2a0f1"
    and .restoration_max_abs <= .restoration_tolerance
' "$OUT/phase.jsonl" >/dev/null
jq -e '
  select(.mode == "snapshot-baseline")
  | .snapshot_loaded == true
    and .working_relation_slots == 4
    and .inverse_native_operations == 0
    and .restoration_max_abs == 0
    and .serial_before == .serial_after
    and .restoration_generation_before
      == .restoration_generation_after
' "$OUT/phase.jsonl" >/dev/null
jq -e '
  select(.mode == "compact-dag-controls")
  | .compact_retain_boundary_equal == true
    and .compact_occurrence_boundary_equal == true
    and .compact_actual_inverse_restored == true
    and .retain_actual_inverse_restored == true
    and .occurrence_actual_inverse_restored == true
    and .wrong_root_inverse == true
    and .missing_root_inverse == true
    and .snapshot_separate_weaker_path == true
    and .compact_working_slots == 4
    and .retain_working_slots == 7
    and .occurrence_working_slots == 15
    and .compact_native_operation_calls == 8
    and .retain_native_operation_calls == 8
    and .occurrence_native_operation_calls == 14
    and .compact_leaf_encode_calls == 12
    and .retain_leaf_encode_calls == 6
    and .occurrence_leaf_encode_calls == 16
    and .compact_lease_allocations == 10
    and .retain_lease_allocations == 7
    and .internal_node_rematerializations == 0
    and .operator_recomputations == 0
    and .same_carrier_transactions == 17
    and .repeated_reuse_max_restoration_abs <= 2e-12
    and .wrong_root_restoration_abs > 1e-6
    and .missing_root_restoration_abs > 1e-6
' "$OUT/phase.jsonl" >/dev/null

controls=(
  --project-intermediate
  --project-obligations
  --project-residency
  --null-carrier
  --control-evict-s
  --control-evict-t
  --control-evict-i
  --control-wrong-leaf-body
  --control-missing-reconstruction
  --control-double-reconstruction
  --control-skip-final-leaf-inverse
  --control-reorder-t
  --control-reorder-s
)
for control in "${controls[@]}"; do
    name=${control#--}
    set +e
    "$OUT/compact_phase" \
        "$DAG_MANIFEST" "$TREE_MANIFEST" "$control" \
        >"$OUT/control_${name}.stdout" \
        2>"$OUT/control_${name}.stderr"
    control_rc=$?
    set -e
    [[ $control_rc -ne 0 ]]
    [[ ! -s "$OUT/control_${name}.stdout" ]]
    [[ -s "$OUT/control_${name}.stderr" ]]
done
rg -q 'leaf-only' "$OUT/control_control-evict-s.stderr"
rg -q 'leaf-only' "$OUT/control_control-evict-t.stderr"
rg -q 'leaf-only' "$OUT/control_control-evict-i.stderr"
rg -q 'obligation invalid' \
    "$OUT/control_control-wrong-leaf-body.stderr"
rg -q 'lease outside pool' \
    "$OUT/control_control-missing-reconstruction.stderr"
rg -q 'materialization state invalid' \
    "$OUT/control_control-double-reconstruction.stderr"
rg -q 'incomplete state' \
    "$OUT/control_control-skip-final-leaf-inverse.stderr"
rg -q 'live dependents' "$OUT/control_control-reorder-t.stderr"
rg -q 'live dependents' "$OUT/control_control-reorder-s.stderr"

cc "${COMMON[@]}" \
    -DCP_WORKING_SLOTS=3U \
    "$PHASE_SOURCE" -lm -o "$OUT/compact_capacity3"
set +e
"$OUT/compact_capacity3" "$DAG_MANIFEST" "$TREE_MANIFEST" \
    >"$OUT/control_capacity3.stdout" \
    2>"$OUT/control_capacity3.stderr"
capacity3_rc=$?
set -e
[[ $capacity3_rc -ne 0 ]]
[[ ! -s "$OUT/control_capacity3.stdout" ]]
rg -q 'pool exhausted' "$OUT/control_capacity3.stderr"

[[ $(rg -n 'ga_copy_block\(' "$DAG_SOURCE" | wc -l) -eq 1 ]]
if rg -q 'ga_copy_block\(' "$PHASE_SOURCE"; then
    exit 1
fi
strace -qq -f \
    -e trace=%file,%network,%ipc,%memory,write,writev,pwrite64,pwritev,pwritev2,memfd_create,sendfile,splice,tee,vmsplice,copy_file_range,process_vm_writev,ptrace,ioctl \
    -o "$OUT/no_smuggle.strace" \
    "$OUT/compact_phase" "$DAG_MANIFEST" "$TREE_MANIFEST" \
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
    -DCP_REUSE_CYCLES=0U \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    "$PHASE_SOURCE" -lm -o "$OUT/compact_sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$OUT/compact_sanitized" "$DAG_MANIFEST" "$TREE_MANIFEST" \
    >"$OUT/sanitized.jsonl" 2>"$OUT/sanitized.stderr"
[[ ! -s "$OUT/sanitized.stderr" ]]
jq -e . "$OUT/sanitized.jsonl" >/dev/null

: >"$OUT/scaling.jsonl"
for width in 3 4 8 12 16; do
    cc "${COMMON[@]}" \
        -DCP_WIDTH="${width}U" \
        -DCP_REUSE_CYCLES=0U \
        -DCP_SCALE_ONLY=1 \
        "$PHASE_SOURCE" -lm -o "$OUT/phase_w${width}"
    cc "${COMMON[@]}" \
        -DMF_REF_WIDTH="${width}U" \
        "$REFERENCE_SOURCE" -lm -o "$OUT/reference_w${width}"
    "$OUT/phase_w${width}" "$DAG_MANIFEST" "$TREE_MANIFEST" \
        >"$OUT/phase_w${width}.jsonl"
    "$OUT/reference_w${width}" "$DAG_MANIFEST" 2 \
        >"$OUT/reference_w${width}.jsonl"
    jq -c '
      select(
        .mode == "compact-dag-primary"
        or .mode == "compact-dag-reuse"
        or .mode == "compact-dag-semantic"
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
        .mode == "compact-dag-primary"
        or .mode == "compact-dag-reuse"
        or .mode == "compact-dag-semantic"
      )
      | .width == $w
        and .working_relation_slots == 4
        and .maximum_live_relation_slots == 4
        and .physical_relation_blocks_including_boundary == 6
        and .relation_cells == (4*$w*$w + 4*$w + 1)
        and .workspace_cells == (36*$w*$w + 11*$w + 2)
        and .carrier_cells == (60*$w*$w + 35*$w + 8)
        and .native_operation_calls == 8
        and .forward_leaf_encodes == 6
        and .inverse_leaf_encodes == 6
        and .lease_allocations == 10
        and .lease_releases == 10
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
        and .restoration_max_abs <= .restoration_tolerance
        and all(.shared_owner_receipts[]; .receipt_exact == true)
    ' "$OUT/phase_w${width}.jsonl" >/dev/null
    jq -c '
      select(.mode == "compact-dag-primary")
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
dag_hash=$(sha256sum "$DAG_SOURCE" | awk '{print $1}')
manifest_hash=$(sha256sum "$DAG_MANIFEST" | awk '{print $1}')
tree_hash=$(sha256sum "$TREE_MANIFEST" | awk '{print $1}')
qualifier_hash=$(sha256sum "$0" | awk '{print $1}')
output_hash=$(sha256sum "$OUT/phase.jsonl" | awk '{print $1}')
trace_hash=$(sha256sum "$OUT/no_smuggle.strace" | awk '{print $1}')

jq -n \
    --arg evidence_directory "$OUT" \
    --arg phase_hash "$phase_hash" \
    --arg reference_hash "$reference_hash" \
    --arg dag_hash "$dag_hash" \
    --arg manifest_hash "$manifest_hash" \
    --arg tree_hash "$tree_hash" \
    --arg qualifier_hash "$qualifier_hash" \
    --arg output_hash "$output_hash" \
    --arg trace_hash "$trace_hash" \
    --arg primary_hash "$primary_hash" \
    --arg reuse_hash "$reuse_hash" \
    --argjson reuse_error "$(
      jq '
        select(.mode == "compact-dag-controls")
        | .repeated_reuse_max_restoration_abs
      ' "$OUT/phase.jsonl"
    )" \
    --argjson width16 "$(
      jq -s 'map(select(.width == 16))[0]' "$OUT/scaling.jsonl"
    )" \
    '{
      result: "PASS",
      claim:
        "BOUNDED_NESTED_AFFINE_DAG_PINNED_SHARED_COMPACT_LEAF_PEBBLING",
      claim_ceiling:
        "BOUNDED_WIDTH16_SOFTWARE_GF2_EXACT_SEVEN_NODE_DAG_FOUR_LOGICAL_LEASE_CAP_SIX_PHYSICAL_RELATION_BLOCKS_DISTINCT_COPIED_BOUNDARY_PINNED_S_T_ORIGINAL_I_ROOT_PUBLIC_IMMUTABLE_LEAF_ONLY_RECONSTRUCTION_CARRIER_COMPACTNESS_ONLY",
      evidence_directory: $evidence_directory,
      tested_widths: [3,4,8,12,16],
      exact_phase_reference_complete_boundaries: 25,
      deterministic_replay: "PASS",
      analyzer: "PASS",
      sanitizers: "PASS",
      mandatory_expanded_no_smuggle_strace: "PASS",
      schedule: {
        logical_working_slots: 4,
        physical_relation_blocks_including_boundary: 6,
        retain_unique_working_slots: 7,
        occurrence_tree_working_slots: 15,
        proven_lower_bound_under_current_boundary_law: 4,
        shared_generations_pinned: [704,705],
        retained_internal_until_consumer_inverse: [706,707],
        reconstructible_public_leaves: [701,702,703],
        internal_node_rematerializations: 0,
        operator_recomputations: 0
      },
      default_width3: {
        carrier_cells: 653,
        live_carrier_bytes: 20896,
        primary_boundary_fnv1a64: $primary_hash,
        reuse_boundary_fnv1a64: $reuse_hash,
        same_carrier_transactions: 17,
        maximum_reuse_restoration_error: $reuse_error
      },
      maximum_width16: $width16,
      controls: {
        capacity_four_passes_capacity_three_fails: true,
        shared_and_internal_eviction_rejected: true,
        wrong_body_missing_double_reconstruction_rejected: true,
        skipped_final_leaf_inverse_rejected: true,
        premature_shared_inverse_rejected: true,
        wrong_and_missing_root_inverse_detected: true,
        snapshot_separate_weaker_path: true,
        retain_and_occurrence_boundaries_equal: true,
        no_smuggle: true
      },
      sha256: {
        phase_source: $phase_hash,
        reference_source: $reference_hash,
        generalized_dag_source: $dag_hash,
        public_manifest: $manifest_hash,
        occurrence_tree_manifest: $tree_hash,
        qualifier: $qualifier_hash,
        accepted_output: $output_hash,
        no_smuggle_trace: $trace_hash
      }
    }' | tee "$OUT/result.json"
