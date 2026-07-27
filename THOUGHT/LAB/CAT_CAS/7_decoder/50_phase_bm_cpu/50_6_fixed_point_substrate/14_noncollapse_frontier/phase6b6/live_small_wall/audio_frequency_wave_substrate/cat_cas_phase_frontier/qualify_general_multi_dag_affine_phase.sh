#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PHASE_SOURCE="$HERE/general_multi_dag_affine_phase.c"
REFERENCE_SOURCE="$HERE/general_multi_dag_affine_reference.c"
BACKEND_SOURCE="$HERE/dag_affine_fanout_phase.c"
REFERENCE_BACKEND="$HERE/dag_affine_fanout_reference.c"
MANIFEST="$HERE/general_multi_dag_affine_topology.txt"
OCCURRENCE_MANIFEST="$HERE/general_multi_dag_affine_occurrence_tree_topology.txt"
OUT=${1:-}
if [[ -z "$OUT" ]]; then
    OUT=$(mktemp -d /tmp/general-multi-dag.XXXXXX)
else
    mkdir -p "$OUT"
fi
for tool in cc jq sha256sum cmp rg strace awk stat; do
    command -v "$tool" >/dev/null
done
COMMON=(-std=c11 -O2 -Wall -Wextra -Wpedantic -Werror)

cc "${COMMON[@]}" -fstack-usage \
    "$PHASE_SOURCE" -lm -o "$OUT/general_phase"
cc "${COMMON[@]}" \
    "$REFERENCE_SOURCE" -lm -o "$OUT/reference"
cc "${COMMON[@]}" -fanalyzer \
    "$PHASE_SOURCE" -lm -o "$OUT/general_analyzer"
cc "${COMMON[@]}" \
    -DGM_NODE_CAP=51U -DGM_OCCURRENCE_ONLY=1 \
    "$PHASE_SOURCE" -lm -o "$OUT/occurrence_phase"
cc "${COMMON[@]}" \
    -DGM_NODE_CAP=16U \
    "$PHASE_SOURCE" -lm -o "$OUT/clone_control"
cc "${COMMON[@]}" \
    -DGM_MAX_FANOUT=3U \
    "$PHASE_SOURCE" -lm -o "$OUT/fanout3_control"

"$OUT/general_phase" "$MANIFEST" >"$OUT/phase.jsonl"
"$OUT/general_phase" "$MANIFEST" >"$OUT/replay.jsonl"
"$OUT/general_analyzer" "$MANIFEST" >"$OUT/analyzer.jsonl"
"$OUT/reference" "$MANIFEST" >"$OUT/reference.jsonl"
"$OUT/occurrence_phase" "$OCCURRENCE_MANIFEST" \
    >"$OUT/occurrence.jsonl"
cmp "$OUT/phase.jsonl" "$OUT/replay.jsonl"
cmp "$OUT/phase.jsonl" "$OUT/analyzer.jsonl"
jq -e . "$OUT/phase.jsonl" >/dev/null
jq -e . "$OUT/reference.jsonl" >/dev/null
jq -e . "$OUT/occurrence.jsonl" >/dev/null

jq -c '
  select(
    .mode == "general-dag-primary"
    or .mode == "general-dag-reuse"
    or .mode == "general-dag-semantic"
  )
  | {boundary_coefficients, boundary_fnv1a64}
' "$OUT/phase.jsonl" >"$OUT/phase_boundaries.jsonl"
jq -c '
  {boundary_coefficients, boundary_fnv1a64}
' "$OUT/reference.jsonl" >"$OUT/reference_boundaries.jsonl"
cmp "$OUT/phase_boundaries.jsonl" "$OUT/reference_boundaries.jsonl"
jq -c '
  select(.mode == "general-dag-primary")
  | {boundary_coefficients, boundary_fnv1a64}
' "$OUT/phase.jsonl" >"$OUT/primary_boundary.json"
jq -c '
  {boundary_coefficients, boundary_fnv1a64}
' "$OUT/occurrence.jsonl" >"$OUT/occurrence_boundary.json"
cmp "$OUT/primary_boundary.json" "$OUT/occurrence_boundary.json"

mapfile -t observed_hashes < <(
    jq -r '
      select(
        .mode == "general-dag-primary"
        or .mode == "general-dag-reuse"
        or .mode == "general-dag-semantic"
      )
      | .boundary_fnv1a64
    ' "$OUT/phase.jsonl"
)
expected_hashes=(
    2e2200557469163d
    6f46809cf9d2cb8e
    eb6451231a634b0b
    7a47f3c447c8ae4f
    fa9b96ae5e69b5cd
)
[[ ${#observed_hashes[@]} -eq 5 ]]
for index in 0 1 2 3 4; do
    [[ "${observed_hashes[$index]}" == "${expected_hashes[$index]}" ]]
done

jq -se '
  [.[] | select(
    .mode == "general-dag-primary"
    or .mode == "general-dag-reuse"
    or .mode == "general-dag-semantic"
  )] as $runs
  | ($runs | length) == 5
  and all($runs[];
    .claim
      == "BOUNDED_FOUR_OWNER_HETEROGENEOUS_FANOUT_RETAIN_ALL_AFFINE_DAG_CUSTODY_ESTABLISHED"
    and .width == 3
    and .manifest_nodes == 15
    and .manifest_leaves == 4
    and .compose_nodes == 8
    and .intersect_nodes == 3
    and .dag_edges == 22
    and .dag_depth == 5
    and .fanout_nodes == 4
    and .maximum_fanout == 4
    and .working_relation_slots == 15
    and .maximum_live_relation_slots == 15
    and .physical_relation_blocks_including_boundary == 16
    and .relation_cells == 49
    and .workspace_cells == 359
    and .carrier_cells == 1143
    and .live_carrier_bytes == 36576
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
    and .lease_allocations == 15
    and .lease_releases == 15
    and .dirty_control_releases == 0
    and .forward_native_operations == 11
    and .inverse_native_operations == 11
    and .native_operation_calls == 22
    and .forward_leaf_encodes == 4
    and .inverse_leaf_encodes == 4
    and .shared_forward_consumptions == 12
    and .shared_inverse_consumptions == 12
    and .shared_forward_materializations == 4
    and .shared_inverse_applications == 4
    and .shared_owner_allocations == 4
    and .shared_owner_releases == 4
    and .shared_resident_blocks == 4
    and .shared_live_at_projection == true
    and .shared_live_after_root_inverse == true
    and .shared_all_pair_nonalias_at_projection == true
    and .shared_pair_nonalias_pairs_at_projection == 6
    and .shared_simultaneous_live_high_water == 4
    and .boundary_block_copy_calls == 2
    and .intermediate_block_copy_calls == 0
    and .serial_after == (.serial_before + 15)
    and .restoration_generation_after
      == (.restoration_generation_before + 1)
    and (.shared_owner_receipts | length) == 4
    and all(.shared_owner_receipts[];
      .receipt_exact == true
      and .all_edge_generations_exact == true
      and .forward_materializations == 1
      and .inverse_applications == 1
      and .owner_allocations == 1
      and .owner_releases == 1
      and .consumer_distinct_slots == 1
      and .consumer_distinct_serials == 1
    )
    and (
      [.shared_owner_receipts[].declared_consumers] | sort
    ) == [2,3,3,4]
    and (.shared_edge_receipts | length) == 12
    and all(.shared_edge_receipts[];
      .forward_seen == true
      and .inverse_seen == true
      and .same_owner_generation == true
    )
  )
' "$OUT/phase.jsonl" >/dev/null

jq -e '
  select(.mode == "general-dag-controls")
  | .compiler_derived_shared_owner_set == true
    and .shared_owner_degrees == [4,3,3,2]
    and .all_shared_receipts_exact == true
    and .all_twelve_shared_edges_exact == true
    and .all_shared_pair_nonalias_at_projection == true
    and .all_six_shared_owner_pairs_nonalias == true
    and .all_shared_live_after_root_inverse == true
    and .nested_lifecycle_order_observed == true
    and .all_semantic_boundary_hashes_distinct == true
    and .primary_reuse_boundary_different == true
    and .coefficient_oblivious_schedule == true
    and .wrong_root_inverse == true
    and .missing_root_inverse == true
    and .snapshot_separate_weaker_path == true
    and .accepted_working_slots == 15
    and .accepted_native_operation_calls == 22
    and .accepted_leaf_encode_calls == 8
    and .same_carrier_transactions == 17
    and .repeated_reuse_max_restoration_abs <= 2e-12
    and .wrong_root_restoration_abs > 1e-6
    and .missing_root_restoration_abs > 1e-6
' "$OUT/phase.jsonl" >/dev/null
jq -e '
  select(.mode == "snapshot-baseline")
  | .snapshot_loaded == true
    and .working_relation_slots == 15
    and .inverse_native_operations == 0
    and .restoration_max_abs == 0
    and .serial_before == .serial_after
    and .restoration_generation_before
      == .restoration_generation_after
' "$OUT/phase.jsonl" >/dev/null
jq -e '
  .mode == "occurrence-tree-phase-baseline"
  and .manifest_nodes == 51
  and .manifest_leaves == 26
  and .compose_nodes == 22
  and .intersect_nodes == 3
  and .dag_edges == 50
  and .dag_depth == 5
  and .fanout_nodes == 0
  and .working_relation_slots == 51
  and .maximum_live_relation_slots == 51
  and .physical_relation_blocks_including_boundary == 52
  and .carrier_cells == 2907
  and .live_carrier_bytes == 93024
  and .native_operation_calls == 50
  and .forward_leaf_encodes == 26
  and .inverse_leaf_encodes == 26
  and .lease_allocations == 51
  and .lease_releases == 51
  and .boundary_fnv1a64 == "2e2200557469163d"
  and .restoration_max_abs <= .restoration_tolerance
  and (.shared_owner_receipts | length) == 0
  and (.shared_edge_receipts | length) == 0
' "$OUT/occurrence.jsonl" >/dev/null

semantic_edges=(
  "805 807 L 803 307d48fe958f7ef9"
  "806 807 R 801 e2be994a1ddd5e41"
  "806 808 L 801 307d48fe958f7ef9"
  "805 808 R 803 307d48fe958f7ef9"
  "805 809 L 801 307d48fe958f7ef9"
  "807 809 R 801 66eeb733fa471b2c"
  "806 810 L 801 6994e952684e170e"
  "807 810 R 801 66eeb733fa471b2c"
  "805 811 L 801 66eeb733fa471b2c"
  "808 811 R 801 66eeb733fa471b2c"
  "807 812 L 801 66eeb733fa471b2c"
  "808 812 R 803 307d48fe958f7ef9"
)
semantic_index=0
for specification in "${semantic_edges[@]}"; do
    read -r producer consumer side replacement expected <<<"$specification"
    "$OUT/reference" "$MANIFEST" \
        --replace-edge "$producer" "$consumer" "$side" "$replacement" 0 \
        >"$OUT/semantic_edge_${semantic_index}.json"
    observed=$(jq -r .boundary_fnv1a64 \
        "$OUT/semantic_edge_${semantic_index}.json")
    [[ "$observed" == "$expected" ]]
    [[ "$observed" != "2e2200557469163d" ]]
    jq -e '
      .intermediate_boundaries_emitted == 0
      and .tuple_slots == 0
      and .assignment_slots == 0
      and .witness_slots == 0
      and ([.boundary_coefficients[]] | add) > 1
    ' "$OUT/semantic_edge_${semantic_index}.json" >/dev/null
    semantic_index=$((semantic_index + 1))
done
[[ $semantic_index -eq 12 ]]

run_reject() {
    local binary=$1
    local control=$2
    local name=$3
    set +e
    "$binary" "$MANIFEST" "$control" \
        >"$OUT/control_${name}.stdout" \
        2>"$OUT/control_${name}.stderr"
    local rc=$?
    set -e
    [[ $rc -ne 0 ]]
    [[ ! -s "$OUT/control_${name}.stdout" ]]
    [[ -s "$OUT/control_${name}.stderr" ]]
}

for owner in 805 806 807 808; do
    run_reject "$OUT/clone_control" \
        "--control-copy-${owner}" "copy_${owner}"
    run_reject "$OUT/general_phase" \
        "--control-stale-${owner}" "stale_${owner}"
    run_reject "$OUT/general_phase" \
        "--control-skip-${owner}" "skip_${owner}"
    run_reject "$OUT/general_phase" \
        "--control-double-${owner}" "double_${owner}"
    run_reject "$OUT/general_phase" \
        "--reordered-inverse-${owner}" "reorder_${owner}"
done
for source in 805 806 807 808; do
    for target in 805 806 807 808; do
        if [[ "$source" != "$target" ]]; then
            run_reject "$OUT/general_phase" \
                "--control-cross-${source}-${target}" \
                "cross_${source}_${target}"
        fi
    done
done
owner_edges=("805 4" "806 3" "807 3" "808 2")
for owner_specification in "${owner_edges[@]}"; do
    read -r owner degree <<<"$owner_specification"
    for ((ordinal=0; ordinal<degree; ++ordinal)); do
        run_reject "$OUT/general_phase" \
            "--control-edge-missing-${owner}-${ordinal}" \
            "edge_missing_${owner}_${ordinal}"
        run_reject "$OUT/general_phase" \
            "--control-edge-stale-${owner}-${ordinal}" \
            "edge_stale_${owner}_${ordinal}"
    done
done
run_reject "$OUT/general_phase" \
    "--control-edge-swap-805-808" "edge_swap_805_808"
run_reject "$OUT/general_phase" \
    "--control-edge-swap-806-807" "edge_swap_806_807"
run_reject "$OUT/general_phase" \
    "--project-intermediate" "project_intermediate"
run_reject "$OUT/general_phase" \
    "--project-custody" "project_custody"
run_reject "$OUT/general_phase" \
    "--null-carrier" "null_carrier"
set +e
"$OUT/fanout3_control" "$MANIFEST" \
    >"$OUT/control_fanout3.stdout" \
    2>"$OUT/control_fanout3.stderr"
fanout3_rc=$?
set -e
[[ $fanout3_rc -ne 0 ]]
[[ ! -s "$OUT/control_fanout3.stdout" ]]
[[ -s "$OUT/control_fanout3.stderr" ]]
rg -q 'consumer count outside bound' "$OUT/control_fanout3.stderr"

[[ $(rg -n 'ga_copy_block\(' "$BACKEND_SOURCE" | wc -l) -eq 1 ]]
if rg -q 'ga_copy_block\(' "$PHASE_SOURCE"; then
    exit 1
fi
strace -qq -f \
    -e trace=%file,%network,%ipc,%memory,write,writev,pwrite64,pwritev,pwritev2,memfd_create,sendfile,splice,tee,vmsplice,copy_file_range,process_vm_writev,ptrace,ioctl \
    -o "$OUT/no_smuggle.strace" \
    "$OUT/general_phase" "$MANIFEST" \
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
    -DGM_REUSE_CYCLES=0U \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    "$PHASE_SOURCE" -lm -o "$OUT/general_sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$OUT/general_sanitized" "$MANIFEST" \
    >"$OUT/sanitized.jsonl" 2>"$OUT/sanitized.stderr"
[[ ! -s "$OUT/sanitized.stderr" ]]
jq -e . "$OUT/sanitized.jsonl" >/dev/null

: >"$OUT/scaling.jsonl"
for width in 3 4 8 12 16; do
    cc "${COMMON[@]}" \
        -DGM_WIDTH="${width}U" \
        -DGM_REUSE_CYCLES=0U \
        -DGM_SCALE_ONLY=1 \
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
        .mode == "general-dag-primary"
        or .mode == "general-dag-reuse"
        or .mode == "general-dag-semantic"
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
        .mode == "general-dag-primary"
        or .mode == "general-dag-reuse"
        or .mode == "general-dag-semantic"
      )
      | .width == $w
        and .working_relation_slots == 15
        and .maximum_live_relation_slots == 15
        and .physical_relation_blocks_including_boundary == 16
        and .relation_cells == (4*$w*$w + 4*$w + 1)
        and .workspace_cells == (36*$w*$w + 11*$w + 2)
        and .carrier_cells == (100*$w*$w + 75*$w + 18)
        and .native_operation_calls == 22
        and .forward_leaf_encodes == 4
        and .inverse_leaf_encodes == 4
        and .lease_allocations == 15
        and .lease_releases == 15
        and .phase_ands
          == (
            11 * (
              288*$w*$w*$w + 225*$w*$w - 9*$w
            )
          )
        and .phase_xors
          == (
            22 * (
              288*$w*$w*$w + 96*$w*$w - 42*$w
            )
          )
        and .restoration_max_abs <= .restoration_tolerance
        and (.shared_owner_receipts | length) == 4
        and all(.shared_owner_receipts[]; .receipt_exact == true)
        and (.shared_edge_receipts | length) == 12
        and all(.shared_edge_receipts[];
          .same_owner_generation == true
        )
    ' "$OUT/phase_w${width}.jsonl" >/dev/null
    jq -c '
      select(.mode == "general-dag-primary")
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

cc "${COMMON[@]}" \
    -DGM_WIDTH=16U \
    -DGM_NODE_CAP=51U \
    -DGM_OCCURRENCE_ONLY=1 \
    "$PHASE_SOURCE" -lm -o "$OUT/occurrence_w16"
"$OUT/occurrence_w16" "$OCCURRENCE_MANIFEST" \
    >"$OUT/occurrence_w16.jsonl"
jq -e '
  .mode == "occurrence-tree-phase-baseline"
  and .width == 16
  and .working_relation_slots == 51
  and .physical_relation_blocks_including_boundary == 52
  and .relation_cells == 1089
  and .workspace_cells == 9394
  and .carrier_cells == 66022
  and .live_carrier_bytes == 2112704
  and .native_operation_calls == 50
  and .forward_leaf_encodes == 26
  and .inverse_leaf_encodes == 26
  and .restoration_max_abs <= .restoration_tolerance
' "$OUT/occurrence_w16.jsonl" >/dev/null
jq -c '
  select(.mode == "general-dag-primary")
  | {boundary_coefficients, boundary_fnv1a64}
' "$OUT/phase_w16.jsonl" >"$OUT/phase_w16.primary_boundary.json"
jq -c '
  {boundary_coefficients, boundary_fnv1a64}
' "$OUT/occurrence_w16.jsonl" >"$OUT/occurrence_w16.boundary.json"
cmp \
    "$OUT/phase_w16.primary_boundary.json" \
    "$OUT/occurrence_w16.boundary.json"

phase_hash=$(sha256sum "$PHASE_SOURCE" | awk '{print $1}')
reference_hash=$(sha256sum "$REFERENCE_SOURCE" | awk '{print $1}')
backend_hash=$(sha256sum "$BACKEND_SOURCE" | awk '{print $1}')
reference_backend_hash=$(sha256sum "$REFERENCE_BACKEND" | awk '{print $1}')
manifest_hash=$(sha256sum "$MANIFEST" | awk '{print $1}')
occurrence_manifest_hash=$(
    sha256sum "$OCCURRENCE_MANIFEST" | awk '{print $1}'
)
qualifier_hash=$(sha256sum "$0" | awk '{print $1}')
output_hash=$(sha256sum "$OUT/phase.jsonl" | awk '{print $1}')
reference_output_hash=$(
    sha256sum "$OUT/reference.jsonl" | awk '{print $1}'
)
occurrence_output_hash=$(
    sha256sum "$OUT/occurrence.jsonl" | awk '{print $1}'
)
trace_hash=$(sha256sum "$OUT/no_smuggle.strace" | awk '{print $1}')
phase_binary_bytes=$(stat -c %s "$OUT/general_phase")
reference_binary_bytes=$(stat -c %s "$OUT/reference")
accepted_stdout_bytes=$(stat -c %s "$OUT/phase.jsonl")
phase_stack_bytes=$(
    awk -F '\t' '$1 ~ /:main$/ {print $2}' \
        "$OUT/general_phase-general_multi_dag_affine_phase.su"
)

jq -n \
    --arg evidence_directory "$OUT" \
    --arg phase_hash "$phase_hash" \
    --arg reference_hash "$reference_hash" \
    --arg backend_hash "$backend_hash" \
    --arg reference_backend_hash "$reference_backend_hash" \
    --arg manifest_hash "$manifest_hash" \
    --arg occurrence_manifest_hash "$occurrence_manifest_hash" \
    --arg qualifier_hash "$qualifier_hash" \
    --arg output_hash "$output_hash" \
    --arg reference_output_hash "$reference_output_hash" \
    --arg occurrence_output_hash "$occurrence_output_hash" \
    --arg trace_hash "$trace_hash" \
    --argjson phase_binary_bytes "$phase_binary_bytes" \
    --argjson reference_binary_bytes "$reference_binary_bytes" \
    --argjson accepted_stdout_bytes "$accepted_stdout_bytes" \
    --argjson phase_stack_bytes "$phase_stack_bytes" \
    --argjson reuse_error "$(
      jq '
        select(.mode == "general-dag-controls")
        | .repeated_reuse_max_restoration_abs
      ' "$OUT/phase.jsonl"
    )" \
    --argjson width16 "$(
      jq -s 'map(select(.width == 16))[0]' "$OUT/scaling.jsonl"
    )" \
    --argjson occurrence16 "$(
      jq '{
        width,
        relation_cells,
        workspace_cells,
        carrier_cells,
        live_carrier_bytes,
        carrier_and_verification_snapshot_heap_bytes,
        native_operation_calls,
        forward_leaf_encodes,
        inverse_leaf_encodes,
        phase_ands,
        phase_xors,
        restoration_max_abs,
        boundary_fnv1a64
      }' "$OUT/occurrence_w16.jsonl"
    )" \
    '{
      result: "PASS",
      claim:
        "BOUNDED_FOUR_OWNER_HETEROGENEOUS_FANOUT_RETAIN_ALL_AFFINE_DAG_CUSTODY_ESTABLISHED",
      claim_ceiling:
        "BOUNDED_WIDTH16_SOFTWARE_GF2_EXACT_15_NODE_RETAIN_ALL_DAG_FOUR_NATIVE_SHARED_OWNERS_FANOUT_HISTOGRAM_4_3_3_2_REFERENCE_ONLY",
      evidence_directory: $evidence_directory,
      tested_widths: [3,4,8,12,16],
      exact_phase_reference_complete_boundaries: 25,
      deterministic_replay: "PASS",
      analyzer: "PASS",
      sanitizers: "PASS",
      mandatory_expanded_no_smuggle_strace: "PASS",
      public_dag: {
        nodes: 15,
        leaves: 4,
        compose_nodes: 8,
        intersect_nodes: 3,
        edges: 22,
        depth: 5,
        compiler_derived_shared_owners: [805,806,807,808],
        fanout_histogram: [4,3,3,2],
        maximum_fanout: 4,
        shared_edges: 12,
        all_shared_edge_generations_exact: true,
        simultaneous_live_shared_owners: 4,
        pairwise_nonalias_owner_pairs: 6,
        internal_relation_copies: 0
      },
      occurrence_expansion: {
        nodes: 51,
        leaves: 26,
        compose_nodes: 22,
        intersect_nodes: 3,
        edges: 50,
        fanout_nodes: 0,
        same_complete_boundary: true,
        actual_inverse_restored: true
      },
      default_width3: {
        carrier_cells: 1143,
        live_carrier_bytes: 36576,
        occurrence_carrier_cells: 2907,
        occurrence_live_carrier_bytes: 93024,
        primary_boundary_fnv1a64: "2e2200557469163d",
        reuse_boundary_fnv1a64: "6f46809cf9d2cb8e",
        same_carrier_transactions: 17,
        maximum_reuse_restoration_error: $reuse_error
      },
      maximum_width16: $width16,
      occurrence_width16: $occurrence16,
      operation_comparison: {
        accepted_native_calls: 22,
        occurrence_native_calls: 50,
        accepted_leaf_encode_calls: 8,
        occurrence_leaf_encode_calls: 52,
        accepted_allocations: 15,
        occurrence_allocations: 51
      },
      controls: {
        all_twelve_semantic_edges_essential: true,
        all_ordered_same_typed_owner_substitutions_rejected: true,
        equal_content_clone_each_owner_rejected: true,
        stale_skip_double_and_reordered_each_owner_rejected: true,
        every_shared_edge_missing_and_stale_receipt_rejected: true,
        swapped_cross_owner_edge_receipts_rejected: true,
        fanout_three_cap_rejects_degree_four_owner: true,
        projection_and_null_rejected: true,
        wrong_and_missing_root_inverse_detected: true,
        snapshot_separate_weaker_path: true,
        no_smuggle: true
      },
      additional_material_resources: {
        phase_binary_bytes: $phase_binary_bytes,
        reference_binary_bytes: $reference_binary_bytes,
        accepted_stdout_bytes: $accepted_stdout_bytes,
        compiler_measured_main_stack_bytes: $phase_stack_bytes
      },
      sha256: {
        phase_source: $phase_hash,
        reference_source: $reference_hash,
        generalized_dag_source: $backend_hash,
        generalized_dag_reference: $reference_backend_hash,
        public_manifest: $manifest_hash,
        occurrence_tree_manifest: $occurrence_manifest_hash,
        qualifier: $qualifier_hash,
        accepted_output: $output_hash,
        reference_output: $reference_output_hash,
        occurrence_output: $occurrence_output_hash,
        no_smuggle_trace: $trace_hash
      },
      general_dag_compiler: false,
      compact_shared_dag_pebbling: false,
      unbounded_fanout: false,
      arbitrary_graph_topology: false,
      catvm_enforcement: false,
      physical_execution: false,
      performance_advantage: false,
      small_wall_crossed: false,
      unlimited_catalytic_computation: false,
      terminal: false
    }' | tee "$OUT/result.json"
