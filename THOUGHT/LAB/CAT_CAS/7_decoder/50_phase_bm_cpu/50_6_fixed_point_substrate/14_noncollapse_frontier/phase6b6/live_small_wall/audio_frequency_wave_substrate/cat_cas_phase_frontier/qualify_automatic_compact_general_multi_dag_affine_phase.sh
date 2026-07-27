#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PHASE_SOURCE="$HERE/automatic_compact_general_multi_dag_affine_phase.c"
RETAIN_SOURCE="$HERE/general_multi_dag_affine_phase.c"
REFERENCE_SOURCE="$HERE/general_multi_dag_affine_reference.c"
BACKEND_SOURCE="$HERE/dag_affine_fanout_phase.c"
MANIFEST="$HERE/general_multi_dag_affine_topology.txt"
OCCURRENCE_MANIFEST="$HERE/general_multi_dag_affine_occurrence_tree_topology.txt"
OUT=${1:-}
if [[ -z "$OUT" ]]; then
    OUT=$(mktemp -d /tmp/automatic-compact-general-dag.XXXXXX)
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
    "$PHASE_SOURCE" -lm -o "$OUT/auto_phase"
cc "${COMMON[@]}" \
    "$RETAIN_SOURCE" -lm -o "$OUT/retain_phase"
cc "${COMMON[@]}" \
    "$REFERENCE_SOURCE" -lm -o "$OUT/reference"
cc "${COMMON[@]}" -fanalyzer \
    "$PHASE_SOURCE" -lm -o "$OUT/auto_analyzer"
cc "${COMMON[@]}" \
    -DGM_NODE_CAP=51U -DGM_OCCURRENCE_ONLY=1 \
    "$RETAIN_SOURCE" -lm -o "$OUT/occurrence_phase"
cc "${COMMON[@]}" \
    -DAC_WORKING_SLOTS=10U \
    "$PHASE_SOURCE" -lm -o "$OUT/capacity10"

"$OUT/auto_phase" "$MANIFEST" >"$OUT/phase.jsonl"
"$OUT/auto_phase" "$MANIFEST" >"$OUT/replay.jsonl"
"$OUT/auto_analyzer" "$MANIFEST" >"$OUT/analyzer.jsonl"
"$OUT/retain_phase" "$MANIFEST" >"$OUT/retain.jsonl"
"$OUT/reference" "$MANIFEST" >"$OUT/reference.jsonl"
"$OUT/occurrence_phase" "$OCCURRENCE_MANIFEST" \
    >"$OUT/occurrence.jsonl"
cmp "$OUT/phase.jsonl" "$OUT/replay.jsonl"
cmp "$OUT/phase.jsonl" "$OUT/analyzer.jsonl"
jq -e . "$OUT/phase.jsonl" >/dev/null
jq -e . "$OUT/retain.jsonl" >/dev/null
jq -e . "$OUT/reference.jsonl" >/dev/null
jq -e . "$OUT/occurrence.jsonl" >/dev/null

jq -c '
  select(
    .mode == "automatic-compact-primary"
    or .mode == "automatic-compact-reuse"
    or .mode == "automatic-compact-semantic"
  )
  | {boundary_coefficients, boundary_fnv1a64}
' "$OUT/phase.jsonl" >"$OUT/phase_boundaries.jsonl"
jq -c '
  select(
    .mode == "general-dag-primary"
    or .mode == "general-dag-reuse"
    or .mode == "general-dag-semantic"
  )
  | {boundary_coefficients, boundary_fnv1a64}
' "$OUT/retain.jsonl" >"$OUT/retain_boundaries.jsonl"
jq -c '
  {boundary_coefficients, boundary_fnv1a64}
' "$OUT/reference.jsonl" >"$OUT/reference_boundaries.jsonl"
cmp "$OUT/phase_boundaries.jsonl" "$OUT/retain_boundaries.jsonl"
cmp "$OUT/phase_boundaries.jsonl" "$OUT/reference_boundaries.jsonl"
jq -c '
  select(.mode == "automatic-compact-primary")
  | {boundary_coefficients, boundary_fnv1a64}
' "$OUT/phase.jsonl" >"$OUT/primary_boundary.json"
jq -c '
  {boundary_coefficients, boundary_fnv1a64}
' "$OUT/occurrence.jsonl" >"$OUT/occurrence_boundary.json"
cmp "$OUT/primary_boundary.json" "$OUT/occurrence_boundary.json"

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
        .mode == "automatic-compact-primary"
        or .mode == "automatic-compact-reuse"
        or .mode == "automatic-compact-semantic"
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
    .mode == "automatic-compact-primary"
    or .mode == "automatic-compact-reuse"
    or .mode == "automatic-compact-semantic"
  )] as $runs
  | ($runs | length) == 5
  and all($runs[];
    .claim
      == "BOUNDED_15_NODE_FOUR_OWNER_AUTOMATIC_PUBLIC_LEAF_PEBBLING_ESTABLISHED"
    and .width == 3
    and .manifest_nodes == 15
    and .manifest_leaves == 4
    and .compose_nodes == 8
    and .intersect_nodes == 3
    and .dag_edges == 22
    and .dag_depth == 5
    and .fanout_nodes == 4
    and .maximum_fanout == 4
    and .working_relation_slots == 11
    and .maximum_live_relation_slots == 11
    and .physical_relation_blocks_including_boundary == 12
    and .relation_cells == 49
    and .workspace_cells == 359
    and .carrier_cells == 947
    and .live_carrier_bytes == 30304
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
    and .restoration_max_abs <= .restoration_tolerance
    and .carrier_integrity_max_abs <= .restoration_tolerance
    and .workspace_cleared == true
    and .allocator_restored == true
    and .edge_custody_restored == true
    and .shared_receipt_reset == true
    and .pending_counts_restored == true
    and .scheduler_queue_empty == true
    and .outstanding_relation_leases == 0
    and .lease_allocations == 19
    and .lease_releases == 19
    and .dirty_control_releases == 0
    and .forward_native_operations == 11
    and .inverse_native_operations == 11
    and .native_operation_calls == 22
    and .forward_leaf_encodes == 8
    and .inverse_leaf_encodes == 8
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
    and .serial_after == (.serial_before + 19)
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

jq -se '
  [.[] | select(.mode == "automatic-compact-schedule")] as $runs
  | ($runs | length) == 5
  and all($runs[];
    .compiler_generated_reversible_tape == true
    and .coefficient_oblivious_plan == true
    and .plan_inputs
      == "PUBLIC_TOPOLOGY_SIGNATURES_AND_LEAF_BODIES"
    and .bound_topology_fnv1a64 == "41d917d4a3308fbe"
    and .bound_schedule_fnv1a64 == "aa5719d149bc55e0"
    and .plan_fnv1a64 == "627f298bb1d2c4e8"
    and .plan_steps == 19
    and .tape_cursor_after_forward == 19
    and .tape_cursor_after_inverse == 0
    and .predicted_peak_live == 11
    and .runtime_peak_live == 11
    and .working_relation_slots == 11
    and .physical_relation_blocks_including_boundary == 12
    and .reversible_plan_storage_bytes_current_abi > 0
    and .scheduler_obligation_storage_bytes_current_abi > 0
    and .scheduler_edge_receipt_storage_bytes_current_abi > 0
    and .scheduler_residency_storage_bytes_current_abi > 0
    and .scheduler_node_lease_storage_bytes_current_abi > 0
    and .transaction_execution_summary_bytes_current_abi > 0
    and .automatic_machine_state_bytes_current_abi > 0
    and .additional_scheduler_state_bytes_current_abi
      == (
        .reversible_plan_storage_bytes_current_abi
        + .scheduler_obligation_storage_bytes_current_abi
        + .scheduler_edge_receipt_storage_bytes_current_abi
        + .scheduler_residency_storage_bytes_current_abi
        + .scheduler_node_lease_storage_bytes_current_abi
      )
    and (
      .reconstructible_public_leaf_ids | sort
    ) == [801,802,803,804]
    and .reconstructible_public_leaves == 4
    and .initial_leaf_materializations == 4
    and .early_leaf_unmaterializations == 4
    and .restoration_leaf_reconstructions == 4
    and .final_leaf_inverses == 4
    and .internal_node_rematerializations == 0
    and .operator_recomputations == 0
    and .all_edge_generations_lawful == true
    and .exact_resident_edges == 18
    and .reconstructed_leaf_edges == 4
    and .obligations_restored == true
    and .residency_restored == true
    and .tape_cursor_empty == true
    and .program_epoch > 0
    and (.all_edge_receipts | length) == 22
    and (
      [.all_edge_receipts[]
        | select(.policy == "EXACT_RESIDENT_GENERATION")
      ] | length
    ) == 18
    and (
      [.all_edge_receipts[]
        | select(.policy == "PUBLIC_LEAF_RECONSTRUCTION")
      ] | length
    ) == 4
    and all(.all_edge_receipts[];
      .forward_seen == true
      and .inverse_seen == true
      and (
        (
          .policy == "EXACT_RESIDENT_GENERATION"
          and .exact_generation == true
          and .lawful_reconstruction == false
        )
        or (
          .policy == "PUBLIC_LEAF_RECONSTRUCTION"
          and .exact_generation == false
          and .lawful_reconstruction == true
        )
      )
    )
  )
  and ([$runs[].plan_fnv1a64] | unique | length) == 1
' "$OUT/phase.jsonl" >/dev/null

jq -e '
  select(.mode == "automatic-compact-controls")
  | .compiler_generated_plan == true
    and .plan_fnv1a64 == "627f298bb1d2c4e8"
    and .all_semantic_plan_hashes_equal == true
    and .coefficient_oblivious_schedule == true
    and .primary_reuse_boundary_different == true
    and .wrong_root_inverse == true
    and .missing_root_inverse == true
    and .snapshot_separate_weaker_path == true
    and .snapshot_post_projection_inverse_operations == 0
    and .accepted_working_slots == 11
    and .retain_all_working_slots == 15
    and .occurrence_working_slots == 51
    and .accepted_native_operation_calls == 22
    and .retain_all_native_operation_calls == 22
    and .occurrence_native_operation_calls == 50
    and .accepted_leaf_encode_calls == 16
    and .retain_all_leaf_encode_calls == 8
    and .occurrence_leaf_encode_calls == 52
    and .accepted_lease_allocations == 19
    and .retain_all_lease_allocations == 15
    and .occurrence_lease_allocations == 51
    and .same_carrier_transactions == 17
    and .maximum_reuse_restoration_abs <= 2e-12
    and .wrong_root_restoration_abs > 1e-6
    and .missing_root_restoration_abs > 1e-6
' "$OUT/phase.jsonl" >/dev/null
jq -e '
  select(.mode == "automatic-compact-snapshot-baseline")
  | .snapshot_loaded == true
    and .working_relation_slots == 11
    and .inverse_native_operations == 0
    and .restoration_max_abs == 0
    and .serial_before == .serial_after
    and .restoration_generation_before
      == .restoration_generation_after
' "$OUT/phase.jsonl" >/dev/null

jq -e '
  select(.mode == "general-dag-primary")
  | .working_relation_slots == 15
    and .carrier_cells == 1143
    and .native_operation_calls == 22
    and .forward_leaf_encodes == 4
    and .inverse_leaf_encodes == 4
    and .lease_allocations == 15
    and .boundary_fnv1a64 == "2e2200557469163d"
    and .restoration_max_abs <= .restoration_tolerance
' "$OUT/retain.jsonl" >/dev/null
jq -e '
  .mode == "occurrence-tree-phase-baseline"
  and .working_relation_slots == 51
  and .carrier_cells == 2907
  and .native_operation_calls == 50
  and .forward_leaf_encodes == 26
  and .inverse_leaf_encodes == 26
  and .lease_allocations == 51
  and .boundary_fnv1a64 == "2e2200557469163d"
  and .restoration_max_abs <= .restoration_tolerance
' "$OUT/occurrence.jsonl" >/dev/null

run_reject() {
    local control=$1
    local expected=$2
    local name=${control#--}
    set +e
    "$OUT/auto_phase" "$MANIFEST" "$control" \
        >"$OUT/control_${name}.stdout" \
        2>"$OUT/control_${name}.stderr"
    local rc=$?
    set -e
    [[ $rc -ne 0 ]]
    [[ ! -s "$OUT/control_${name}.stdout" ]]
    [[ -s "$OUT/control_${name}.stderr" ]]
    rg -q "$expected" "$OUT/control_${name}.stderr"
}

run_reject --control-evict-shared 'public-leaf-only'
run_reject --control-evict-internal 'public-leaf-only'
run_reject --control-wrong-body 'obligation invalid'
run_reject --control-stale-epoch 'obligation invalid'
run_reject --control-missing-reconstruction \
    'reconstructed edge custody invalid'
run_reject --control-double-reconstruction \
    'materialization state invalid'
run_reject --control-skip-final-leaf-inverse \
    'injected fault escaped detection'
run_reject --control-reorder-shared \
    'producer inverse has live dependents'
run_reject --control-stale-internal-edge \
    'resident edge changed generation'
run_reject --control-tamper-tape 'tape hash changed'
run_reject --project-intermediate 'internal projection denied'
run_reject --project-tape 'internal projection denied'
run_reject --project-obligations 'internal projection denied'
run_reject --project-residency 'internal projection denied'
run_reject --null-carrier 'null automatic compact carrier'

set +e
"$OUT/capacity10" "$MANIFEST" \
    >"$OUT/control_capacity10.stdout" \
    2>"$OUT/control_capacity10.stderr"
capacity_rc=$?
set -e
[[ $capacity_rc -ne 0 ]]
[[ ! -s "$OUT/control_capacity10.stdout" ]]
rg -q 'clean relation-slot pool exhausted' \
    "$OUT/control_capacity10.stderr"

[[ $(rg -n 'ga_copy_block\(' "$BACKEND_SOURCE" | wc -l) -eq 1 ]]
if rg -q 'ga_copy_block\(' "$PHASE_SOURCE"; then
    exit 1
fi
strace -qq -f \
    -e trace=%file,%network,%ipc,%memory,write,writev,pwrite64,pwritev,pwritev2,memfd_create,sendfile,splice,tee,vmsplice,copy_file_range,process_vm_writev,ptrace,ioctl \
    -o "$OUT/no_smuggle.strace" \
    "$OUT/auto_phase" "$MANIFEST" \
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
    -DAC_REUSE_CYCLES=0U \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    "$PHASE_SOURCE" -lm -o "$OUT/auto_sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$OUT/auto_sanitized" "$MANIFEST" \
    >"$OUT/sanitized.jsonl" 2>"$OUT/sanitized.stderr"
[[ ! -s "$OUT/sanitized.stderr" ]]
jq -e . "$OUT/sanitized.jsonl" >/dev/null

: >"$OUT/scaling.jsonl"
for width in 3 4 8 12 16; do
    cc "${COMMON[@]}" \
        -DGM_WIDTH="${width}U" \
        -DAC_REUSE_CYCLES=0U \
        -DAC_SCALE_ONLY=1 \
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
        .mode == "automatic-compact-primary"
        or .mode == "automatic-compact-reuse"
        or .mode == "automatic-compact-semantic"
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
        .mode == "automatic-compact-primary"
        or .mode == "automatic-compact-reuse"
        or .mode == "automatic-compact-semantic"
      )
      | .width == $w
        and .working_relation_slots == 11
        and .maximum_live_relation_slots == 11
        and .physical_relation_blocks_including_boundary == 12
        and .relation_cells == (4*$w*$w + 4*$w + 1)
        and .workspace_cells == (36*$w*$w + 11*$w + 2)
        and .carrier_cells == (84*$w*$w + 59*$w + 14)
        and .native_operation_calls == 22
        and .forward_leaf_encodes == 8
        and .inverse_leaf_encodes == 8
        and .lease_allocations == 19
        and .lease_releases == 19
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
    jq -e '
      select(.mode == "automatic-compact-schedule")
      | .bound_topology_fnv1a64 == "41d917d4a3308fbe"
        and .bound_schedule_fnv1a64 == "aa5719d149bc55e0"
        and .plan_fnv1a64 == "627f298bb1d2c4e8"
        and .predicted_peak_live == 11
        and .runtime_peak_live == 11
        and .all_edge_generations_lawful == true
        and .tape_cursor_empty == true
    ' "$OUT/phase_w${width}.jsonl" >/dev/null
    jq -c '
      select(.mode == "automatic-compact-primary")
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
retain_hash=$(sha256sum "$RETAIN_SOURCE" | awk '{print $1}')
reference_hash=$(sha256sum "$REFERENCE_SOURCE" | awk '{print $1}')
backend_hash=$(sha256sum "$BACKEND_SOURCE" | awk '{print $1}')
manifest_hash=$(sha256sum "$MANIFEST" | awk '{print $1}')
occurrence_manifest_hash=$(
    sha256sum "$OCCURRENCE_MANIFEST" | awk '{print $1}'
)
qualifier_hash=$(sha256sum "$0" | awk '{print $1}')
output_hash=$(sha256sum "$OUT/phase.jsonl" | awk '{print $1}')
retain_output_hash=$(sha256sum "$OUT/retain.jsonl" | awk '{print $1}')
reference_output_hash=$(
    sha256sum "$OUT/reference.jsonl" | awk '{print $1}'
)
occurrence_output_hash=$(
    sha256sum "$OUT/occurrence.jsonl" | awk '{print $1}'
)
trace_hash=$(sha256sum "$OUT/no_smuggle.strace" | awk '{print $1}')
phase_binary_bytes=$(stat -c %s "$OUT/auto_phase")
accepted_stdout_bytes=$(stat -c %s "$OUT/phase.jsonl")
phase_stack_bytes=$(
    awk -F '\t' '$1 ~ /:main$/ {print $2}' \
        "$OUT/auto_phase-automatic_compact_general_multi_dag_affine_phase.su"
)
execute_stack_bytes=$(
    awk -F '\t' '$1 ~ /:ac_execute$/ {print $2}' \
        "$OUT/auto_phase-automatic_compact_general_multi_dag_affine_phase.su"
)
accepted_stack_floor_bytes=$(
    awk -v main="$phase_stack_bytes" -v execute="$execute_stack_bytes" \
        'BEGIN {print main + execute}'
)
all_frame_sum_bytes=$(
    awk -F '\t' '{sum += $2} END {print sum}' \
        "$OUT/auto_phase-automatic_compact_general_multi_dag_affine_phase.su"
)
stack_function_records=$(
    awk -F '\t' 'END {print NR}' \
        "$OUT/auto_phase-automatic_compact_general_multi_dag_affine_phase.su"
)
scheduler_abi=$(
    jq -s '
      map(select(.mode == "automatic-compact-schedule"))[0]
      | {
          reversible_plan_storage_bytes_current_abi,
          scheduler_obligation_storage_bytes_current_abi,
          scheduler_edge_receipt_storage_bytes_current_abi,
          scheduler_residency_storage_bytes_current_abi,
          scheduler_node_lease_storage_bytes_current_abi,
          transaction_execution_summary_bytes_current_abi,
          automatic_machine_state_bytes_current_abi,
          additional_scheduler_state_bytes_current_abi
        }
    ' "$OUT/phase.jsonl"
)

jq -n \
    --arg evidence_directory "$OUT" \
    --arg phase_hash "$phase_hash" \
    --arg retain_hash "$retain_hash" \
    --arg reference_hash "$reference_hash" \
    --arg backend_hash "$backend_hash" \
    --arg manifest_hash "$manifest_hash" \
    --arg occurrence_manifest_hash "$occurrence_manifest_hash" \
    --arg qualifier_hash "$qualifier_hash" \
    --arg output_hash "$output_hash" \
    --arg retain_output_hash "$retain_output_hash" \
    --arg reference_output_hash "$reference_output_hash" \
    --arg occurrence_output_hash "$occurrence_output_hash" \
    --arg trace_hash "$trace_hash" \
    --argjson phase_binary_bytes "$phase_binary_bytes" \
    --argjson accepted_stdout_bytes "$accepted_stdout_bytes" \
    --argjson phase_stack_bytes "$phase_stack_bytes" \
    --argjson execute_stack_bytes "$execute_stack_bytes" \
    --argjson accepted_stack_floor_bytes "$accepted_stack_floor_bytes" \
    --argjson all_frame_sum_bytes "$all_frame_sum_bytes" \
    --argjson stack_function_records "$stack_function_records" \
    --argjson scheduler_abi "$scheduler_abi" \
    --argjson reuse_error "$(
      jq '
        select(.mode == "automatic-compact-controls")
        | .maximum_reuse_restoration_abs
      ' "$OUT/phase.jsonl"
    )" \
    --argjson width16 "$(
      jq -s 'map(select(.width == 16))[0]' "$OUT/scaling.jsonl"
    )" \
    '{
      result: "PASS",
      claim:
        "BOUNDED_15_NODE_FOUR_OWNER_AUTOMATIC_PUBLIC_LEAF_PEBBLING_ESTABLISHED",
      claim_ceiling:
        "BOUNDED_WIDTH16_SOFTWARE_GF2_EXACT_15_NODE_DAG_COMPILER_EMITTED_19_STEP_REVERSIBLE_PUBLIC_DEGREE_ONE_LEAF_ONLY_PEBBLING_11_WORKING_SLOTS_12_PHYSICAL_RELATION_BLOCKS_FOUR_PINNED_SHARED_OWNERS_REFERENCE_ONLY",
      evidence_directory: $evidence_directory,
      tested_widths: [3,4,8,12,16],
      exact_phase_reference_complete_boundaries: 25,
      deterministic_replay: "PASS",
      analyzer: "PASS",
      sanitizers: "PASS",
      mandatory_expanded_no_smuggle_strace: "PASS",
      schedule: {
        compiler_generated_reversible_tape: true,
        bound_topology_fnv1a64: "41d917d4a3308fbe",
        bound_schedule_fnv1a64: "aa5719d149bc55e0",
        plan_fnv1a64: "627f298bb1d2c4e8",
        tape_steps: 19,
        reconstructible_public_leaves: [801,802,803,804],
        predicted_and_observed_peak_live: 11,
        capacity_eleven_passes_capacity_ten_fails: true,
        internal_node_rematerializations: 0,
        operator_recomputations: 0
      },
      custody: {
        pinned_shared_owners: [805,806,807,808],
        shared_fanout_histogram: [4,3,3,2],
        exact_resident_generation_edges: 18,
        lawful_reconstructed_public_leaf_edges: 4,
        exact_shared_edges: 12,
        pairwise_nonalias_owner_pairs: 6,
        intermediate_relation_copies: 0
      },
      resources: {
        carrier_cells_law: "84*w^2 + 59*w + 14",
        retain_all_carrier_cells_law: "100*w^2 + 75*w + 18",
        occurrence_carrier_cells_law: "244*w^2 + 219*w + 54",
        default_width3_carrier_cells: 947,
        default_width3_live_carrier_bytes: 30304,
        retain_all_width3_carrier_cells: 1143,
        occurrence_width3_carrier_cells: 2907,
        native_calls: 22,
        leaf_encode_calls: 16,
        lease_allocations: 19,
        same_carrier_transactions: 17,
        maximum_reuse_restoration_error: $reuse_error
      },
      maximum_width16: $width16,
      controls: {
        shared_and_internal_eviction_rejected: true,
        wrong_body_and_stale_epoch_rejected: true,
        missing_double_and_skipped_reconstruction_rejected: true,
        shared_dependency_reorder_rejected: true,
        stale_internal_generation_rejected: true,
        tape_tamper_rejected: true,
        projection_and_null_rejected: true,
        wrong_and_missing_root_inverse_detected: true,
        snapshot_separate_weaker_path: true,
        no_smuggle: true
      },
      additional_material_resources: {
        phase_binary_bytes: $phase_binary_bytes,
        accepted_stdout_bytes: $accepted_stdout_bytes,
        compiler_measured_main_stack_bytes: $phase_stack_bytes,
        compiler_measured_ac_execute_stack_bytes: $execute_stack_bytes,
        compiler_measured_accepted_main_plus_execute_stack_floor_bytes:
          $accepted_stack_floor_bytes,
        compiler_all_function_frames_sum_conservative_upper_bound_bytes:
          $all_frame_sum_bytes,
        compiler_stack_usage_function_records: $stack_function_records,
        compiler_stack_measurement:
          "GCC_FSTACK_USAGE_O2_DYNAMIC_BOUNDED",
        scheduler_abi: $scheduler_abi
      },
      sha256: {
        phase_source: $phase_hash,
        retain_all_source: $retain_hash,
        reference_source: $reference_hash,
        generalized_dag_source: $backend_hash,
        public_manifest: $manifest_hash,
        occurrence_tree_manifest: $occurrence_manifest_hash,
        qualifier: $qualifier_hash,
        accepted_output: $output_hash,
        retain_all_output: $retain_output_hash,
        reference_output: $reference_output_hash,
        occurrence_output: $occurrence_output_hash,
        no_smuggle_trace: $trace_hash
      },
      internal_operator_rematerialization: false,
      automatic_general_dag_pebbling: false,
      arbitrary_graph_topology: false,
      catvm_enforcement: false,
      physical_execution: false,
      performance_advantage: false,
      small_wall_crossed: false,
      unlimited_catalytic_computation: false,
      terminal: false
    }' | tee "$OUT/result.json"
