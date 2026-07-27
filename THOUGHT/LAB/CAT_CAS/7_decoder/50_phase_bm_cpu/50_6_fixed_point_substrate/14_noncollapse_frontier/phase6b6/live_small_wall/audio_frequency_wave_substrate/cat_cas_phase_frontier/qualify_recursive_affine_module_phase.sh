#!/usr/bin/env bash
set -euo pipefail

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PHASE_SOURCE="$HERE/recursive_affine_module_phase.c"
REFERENCE_SOURCE="$HERE/recursive_affine_module_reference.c"
MANIFEST="$HERE/recursive_affine_topology.txt"
MODULE_SOURCE="$HERE/algebraic_affine_module_phase.c"
MODULE_REFERENCE="$HERE/affine_module_reference.c"
KERNEL_SOURCE="$HERE/algebraic_oblivious_affine_phase.c"
KERNEL_REFERENCE="$HERE/oblivious_affine_reference.c"
BASE_SOURCE="$HERE/algebraic_series_parallel_phase.c"
OUT=${1:-}
if [[ -z "$OUT" ]]; then
    OUT=$(mktemp -d /tmp/recursive-affine-phase.XXXXXX)
else
    mkdir -p "$OUT"
fi

for tool in cc jq sha256sum diff cmp awk sed stat strace; do
    command -v "$tool" >/dev/null
done

COMMON=(-std=c11 -O2 -Wall -Wextra -Wpedantic -Werror)
RC_MAX_NODES=15

cc "${COMMON[@]}" -fstack-usage \
    "$PHASE_SOURCE" -lm -o "$OUT/recursive_phase"
cc "${COMMON[@]}" -fstack-usage \
    "$REFERENCE_SOURCE" -lm -o "$OUT/recursive_reference"
cc "${COMMON[@]}" -fanalyzer \
    "$PHASE_SOURCE" -lm -o "$OUT/recursive_analyzer"

"$OUT/recursive_phase" "$MANIFEST" >"$OUT/phase.jsonl"
"$OUT/recursive_phase" "$MANIFEST" >"$OUT/phase_replay.jsonl"
"$OUT/recursive_reference" "$MANIFEST" >"$OUT/reference.jsonl"
"$OUT/recursive_analyzer" "$MANIFEST" >"$OUT/analyzer.jsonl"

cmp "$OUT/phase.jsonl" "$OUT/phase_replay.jsonl"
cmp "$OUT/phase.jsonl" "$OUT/analyzer.jsonl"
jq -e . "$OUT/phase.jsonl" >/dev/null
jq -e . "$OUT/reference.jsonl" >/dev/null

jq -c '
    select(.mode | startswith("recursive-affine"))
    | {boundary_coefficients, boundary_fnv1a64}
' "$OUT/phase.jsonl" >"$OUT/phase_boundaries.jsonl"
jq -c '
    {boundary_coefficients, boundary_fnv1a64}
' "$OUT/reference.jsonl" >"$OUT/reference_boundaries.jsonl"
cmp "$OUT/phase_boundaries.jsonl" "$OUT/reference_boundaries.jsonl"

[[ $(wc -l <"$OUT/phase_boundaries.jsonl") -eq 5 ]]
[[ $(wc -l <"$OUT/reference_boundaries.jsonl") -eq 5 ]]

jq -se '
    [
      .[] | select(.mode | startswith("recursive-affine"))
    ] as $runs
    | ($runs | length) == 5
    and all($runs[];
      .claim
        == "BOUNDED_PUBLIC_RECURSIVE_AFFINE_SERIES_INTERSECTION_TREE_COMPILATION"
      and .width == 3
      and .manifest_nodes == 15
      and .manifest_leaves == 8
      and .compose_nodes == 5
      and .intersect_nodes == 2
      and .tree_edges == 14
      and .tree_depth == 3
      and .manifest_dependency_ordered == false
      and .compiled_postorder_valid == true
      and .derived_nominal_signatures == true
      and .address_assigned_after_validation == true
      and .tree_only_no_dag_fanout == true
      and .reversible_pebbling == true
      and .working_relation_slots == 7
      and .maximum_live_relation_slots == 7
      and .physical_relation_blocks_including_boundary == 8
      and .retain_all_relation_blocks_including_boundary == 16
      and .relation_cells == 49
      and .workspace_cells == 359
      and .carrier_cells == 751
      and .live_carrier_bytes == 24032
      and .decoded_intermediate_coefficients == 0
      and .serialized_intermediate_coefficients == 0
      and .pre_inverse_carrier_aggregate_outputs == 0
      and .host_selected_pivots == 0
      and .tuple_slots == 0
      and .assignment_slots == 0
      and .witness_slots == 0
      and .snapshot_loaded == false
      and .restoration_tolerance == 2e-12
      and .restoration_max_abs <= .restoration_tolerance
      and .carrier_integrity_max_abs <= .restoration_tolerance
      and .workspace_cleared == true
      and .allocator_restored == true
      and .outstanding_relation_leases == 0
      and .dirty_control_releases == 0
      and .forward_native_operations == 21
      and .inverse_native_operations == 21
      and .native_operation_calls == 42
      and .transient_forward_recomputations == 14
      and .inverse_factor_recomputations == 21
      and .forward_leaf_encodes == 64
      and .inverse_leaf_encodes == 64
    )
' "$OUT/phase.jsonl" >/dev/null

primary_hash=$(jq -r '
    select(.mode == "recursive-affine-primary")
    | .boundary_fnv1a64
' "$OUT/phase.jsonl")
reuse_hash=$(jq -r '
    select(.mode == "recursive-affine-reuse")
    | .boundary_fnv1a64
' "$OUT/phase.jsonl")
[[ "$primary_hash" == "e9f225404bbe6d62" ]]
[[ "$reuse_hash" == "6f46809cf9d2cb8e" ]]
[[ "$primary_hash" != "$reuse_hash" ]]

jq -e '
    select(.mode == "snapshot-baseline")
    | .snapshot_loaded == true
      and .reversible_pebbling == true
      and .boundary_fnv1a64 == "e9f225404bbe6d62"
      and .restoration_max_abs == 0
      and .allocator_restored == true
      and .native_operation_calls == 21
' "$OUT/phase.jsonl" >/dev/null

jq -e '
    select(.mode == "retain-all-phase-baseline")
    | .snapshot_loaded == false
      and .reversible_pebbling == false
      and .working_relation_slots == 15
      and .maximum_live_relation_slots == 15
      and .physical_relation_blocks_including_boundary == 16
      and .carrier_cells == 1143
      and .live_carrier_bytes == 36576
      and .boundary_fnv1a64 == "e9f225404bbe6d62"
      and .restoration_max_abs <= .restoration_tolerance
      and .allocator_restored == true
      and .forward_native_operations == 7
      and .inverse_native_operations == 7
      and .native_operation_calls == 14
      and .transient_forward_recomputations == 0
' "$OUT/phase.jsonl" >/dev/null

jq -e '
    select(.mode == "controls")
    | .primary_reuse_boundary_different == true
      and .contradictory_branch_empty == true
      and .highest_input_column_restriction_nonempty == true
      and .empty_leaf_absorber == true
      and .coefficient_oblivious_schedule == true
      and .wrong_root_inverse == true
      and .missing_root_inverse == true
      and .retain_all_boundary_equal == true
      and .retain_all_restored == true
      and .snapshot_separate_weaker_path == true
      and .snapshot_post_projection_inverse_factor_recomputations == 0
      and .clean_slots == 7
      and .retain_all_working_slots == 15
      and .relation_slot_reduction == 8
      and .pebbled_native_operation_calls == 42
      and .retain_all_native_operation_calls == 14
      and .wrong_root_restoration_abs > 1e-6
      and .missing_root_restoration_abs > 1e-6
      and .same_carrier_transactions == 17
      and .repeated_reuse_max_restoration_abs <= 2e-12
' "$OUT/phase.jsonl" >/dev/null

controls=(
    --project-intermediate
    --project-controls
    --null-carrier
    --control-cycle
    --control-unknown-child
    --control-fanout
    --control-type-mismatch
    --control-duplicate-id
    --control-unreachable
    --control-multiple-root
    --control-leaf-owner
    --reordered-inverse
)
for control in "${controls[@]}"; do
    name=${control#--}
    set +e
    "$OUT/recursive_phase" "$MANIFEST" "$control" \
        >"$OUT/control_${name}.stdout" \
        2>"$OUT/control_${name}.stderr"
    control_rc=$?
    set -e
    [[ $control_rc -ne 0 ]]
    [[ ! -s "$OUT/control_${name}.stdout" ]]
    [[ -s "$OUT/control_${name}.stderr" ]]
done

rg -q 'contains cycle' "$OUT/control_control-cycle.stderr"
rg -q 'unknown node' "$OUT/control_control-unknown-child.stderr"
rg -q 'not a single tree' "$OUT/control_control-fanout.stderr"
rg -q 'nominal type mismatch' \
    "$OUT/control_control-type-mismatch.stderr"
rg -q 'stale or mistyped relation lease' \
    "$OUT/control_reordered-inverse.stderr"

strace -qq -f \
    -e trace=%file,%network,write,writev,pwrite64,pwritev,pwritev2 \
    -o "$OUT/no_smuggle.strace" \
    "$OUT/recursive_phase" "$MANIFEST" \
    >"$OUT/traced.jsonl" 2>"$OUT/traced.stderr"
cmp "$OUT/phase.jsonl" "$OUT/traced.jsonl"
[[ ! -s "$OUT/traced.stderr" ]]
if rg -q \
    'O_WRONLY|O_RDWR|O_CREAT|write(v)?\((2|[3-9]|[1-9][0-9]+),|pwrite|send(to|msg|mmsg)?\(|socket\(|connect\(' \
    "$OUT/no_smuggle.strace"
then
    exit 1
fi

cc -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror \
    -DRC_REUSE_CYCLES=0U \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    "$PHASE_SOURCE" -lm -o "$OUT/recursive_sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$OUT/recursive_sanitized" "$MANIFEST" \
    >"$OUT/sanitized.jsonl" 2>"$OUT/sanitized.stderr"
[[ ! -s "$OUT/sanitized.stderr" ]]
jq -e . "$OUT/sanitized.jsonl" >/dev/null

: >"$OUT/scaling.jsonl"
compiler_builds=4
for width in 3 4 8 12 16; do
    cc "${COMMON[@]}" -fstack-usage \
        -DRC_WIDTH="${width}U" \
        -DRC_RUN_VARIANTS=2U \
        -DRC_REUSE_CYCLES=0U \
        -DRC_SCALE_ONLY=1 \
        "$PHASE_SOURCE" -lm -o "$OUT/phase_w${width}"
    cc "${COMMON[@]}" \
        -DRR_WIDTH="${width}U" \
        -DRR_RUN_VARIANTS=2U \
        "$REFERENCE_SOURCE" -lm -o "$OUT/reference_w${width}"
    compiler_builds=$((compiler_builds + 2))
    "$OUT/phase_w${width}" "$MANIFEST" \
        >"$OUT/phase_w${width}.jsonl"
    "$OUT/reference_w${width}" "$MANIFEST" \
        >"$OUT/reference_w${width}.jsonl"
    jq -e . "$OUT/phase_w${width}.jsonl" >/dev/null
    jq -e . "$OUT/reference_w${width}.jsonl" >/dev/null
    jq -c '{boundary_coefficients, boundary_fnv1a64}' \
        "$OUT/phase_w${width}.jsonl" \
        >"$OUT/phase_w${width}.boundaries.jsonl"
    jq -c '{boundary_coefficients, boundary_fnv1a64}' \
        "$OUT/reference_w${width}.jsonl" \
        >"$OUT/reference_w${width}.boundaries.jsonl"
    cmp \
        "$OUT/phase_w${width}.boundaries.jsonl" \
        "$OUT/reference_w${width}.boundaries.jsonl"
    jq -e --argjson width "$width" '
      all(.
        ; .width == $width
          and .manifest_nodes == 15
          and .working_relation_slots == 7
          and .maximum_live_relation_slots == 7
          and .physical_relation_blocks_including_boundary == 8
          and .retain_all_relation_blocks_including_boundary == 16
          and .relation_cells
            == (4 * $width * $width + 4 * $width + 1)
          and .workspace_cells
            == (36 * $width * $width + 11 * $width + 2)
          and .carrier_cells
            == (68 * $width * $width + 43 * $width + 10)
          and .native_operation_calls == 42
          and .phase_ands
            == (
              21 * (
                288 * $width * $width * $width
                + 225 * $width * $width
                - 9 * $width
              )
            )
          and .phase_xors
            == (
              42 * (
                288 * $width * $width * $width
                + 96 * $width * $width
                - 42 * $width
              )
            )
          and .restoration_max_abs <= .restoration_tolerance
          and .allocator_restored == true
          and .outstanding_relation_leases == 0
      )
    ' "$OUT/phase_w${width}.jsonl" >/dev/null
    jq -c '
      select(.mode == "recursive-affine-primary")
      | {
          width,
          relation_cells,
          workspace_cells,
          carrier_cells,
          live_carrier_bytes,
          carrier_and_verification_snapshot_heap_bytes,
          compiled_topology_bytes_current_abi,
          program_storage_bytes_current_abi,
          boundary_storage_bytes_current_abi,
          retain_all_relation_blocks_including_boundary,
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
manifest_hash=$(sha256sum "$MANIFEST" | awk '{print $1}')
module_hash=$(sha256sum "$MODULE_SOURCE" | awk '{print $1}')
module_reference_hash=$(sha256sum "$MODULE_REFERENCE" | awk '{print $1}')
kernel_hash=$(sha256sum "$KERNEL_SOURCE" | awk '{print $1}')
kernel_reference_hash=$(sha256sum "$KERNEL_REFERENCE" | awk '{print $1}')
base_hash=$(sha256sum "$BASE_SOURCE" | awk '{print $1}')
qualifier_hash=$(sha256sum "$0" | awk '{print $1}')
result_hash=$(sha256sum "$OUT/phase.jsonl" | awk '{print $1}')
reference_result_hash=$(sha256sum "$OUT/reference.jsonl" | awk '{print $1}')
scaling_hash=$(sha256sum "$OUT/scaling.jsonl" | awk '{print $1}')
trace_hash=$(sha256sum "$OUT/no_smuggle.strace" | awk '{print $1}')

default_stack_file="$OUT/recursive_phase-recursive_affine_module_phase.su"
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
main_stack=$(stack_frame_from "$default_stack_file" main)
execute_stack=$(stack_frame_from "$default_stack_file" rc_execute)
eval_stack=$(stack_frame_from "$default_stack_file" rc_eval)
uneval_stack=$(stack_frame_from "$default_stack_file" rc_uneval)
apply_stack=$(stack_frame_from "$default_stack_file" rc_apply_node)
compose_stack=$(
    stack_frame_from "$default_stack_file" ga_apply_compose.part.0
)
reduce_stack=$(stack_frame_from "$default_stack_file" ga_reduce_stage)
swap_stack=$(
    stack_frame_from \
        "$default_stack_file" ga_conditional_swap_rows.part.0
)
conservative_stack=$((
    main_stack
    + execute_stack
    + RC_MAX_NODES * (eval_stack + uneval_stack)
    + apply_stack
    + compose_stack
    + reduce_stack
    + swap_stack
))
w16_stack_file=$(
    find "$OUT" -maxdepth 1 \
        -name 'phase_w16-recursive_affine_module_phase.su' | head -1
)
if [[ -z "$w16_stack_file" ]]; then
    exit 1
fi
w16_main_stack=$(stack_frame_from "$w16_stack_file" main)
w16_execute_stack=$(stack_frame_from "$w16_stack_file" rc_execute)
w16_eval_stack=$(stack_frame_from "$w16_stack_file" rc_eval)
w16_uneval_stack=$(stack_frame_from "$w16_stack_file" rc_uneval)
w16_apply_stack=$(stack_frame_from "$w16_stack_file" rc_apply_node)
w16_compose_stack=$(
    stack_frame_from "$w16_stack_file" ga_apply_compose.part.0
)
w16_reduce_stack=$(
    stack_frame_from "$w16_stack_file" ga_reduce_stage
)
w16_swap_stack=$(
    stack_frame_from \
        "$w16_stack_file" ga_conditional_swap_rows.part.0
)
w16_conservative_stack=$((
    w16_main_stack
    + w16_execute_stack
    + RC_MAX_NODES * (w16_eval_stack + w16_uneval_stack)
    + w16_apply_stack
    + w16_compose_stack
    + w16_reduce_stack
    + w16_swap_stack
))
phase_binary_bytes=$(stat -c '%s' "$OUT/recursive_phase")
reference_binary_bytes=$(stat -c '%s' "$OUT/recursive_reference")
accepted_stdout_bytes=$(stat -c '%s' "$OUT/phase.jsonl")
reference_stdout_bytes=$(stat -c '%s' "$OUT/reference.jsonl")

jq -n \
    --arg result PASS \
    --arg evidence_directory "$OUT" \
    --arg phase_hash "$phase_hash" \
    --arg reference_hash "$reference_hash" \
    --arg manifest_hash "$manifest_hash" \
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
        "BOUNDED_PUBLIC_RECURSIVE_AFFINE_SERIES_INTERSECTION_TREE_COMPILATION",
      claim_ceiling:
        "BOUNDED_WIDTH16_SOFTWARE_GF2_AFFINE_TREE_COMPILER_REFERENCE_ONLY",
      evidence_directory: $evidence_directory,
      compiler_builds: $compiler_builds,
      sanitizers: "PASS",
      analyzer: "PASS",
      deterministic_replay: "PASS",
      mandatory_no_smuggle_strace: "PASS",
      tested_widths: [3,4,8,12,16],
      exact_phase_reference_complete_boundaries: 10,
      default_semantic_phase_reference_boundaries: 5,
      public_tree: {
        nodes: 15,
        leaves: 8,
        compose_nodes: 5,
        intersect_nodes: 2,
        depth: 3,
        clean_working_slots: 7,
        retain_all_working_slots: 15,
        dedicated_boundary_slots: 1,
        pebbled_native_calls: 42,
        retain_all_native_calls: 14
      },
      default_width3: {
        relation_cells: 49,
        workspace_cells: 359,
        carrier_cells: 751,
        live_carrier_bytes: 24032,
        primary_boundary_fnv1a64: $primary_hash,
        reuse_boundary_fnv1a64: $reuse_hash,
        same_carrier_transactions: 17,
        maximum_reuse_restoration_error: $maximum_reuse_error,
        conservative_compiler_measured_stack_bytes: $conservative_stack,
        conservative_accounted_execution_bytes_current_abi:
          (
            48064 + 1536 + 1960 + 216
            + $conservative_stack + 472
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
                  + 472
                )
            }
        ),
      additional_material_resources: {
        phase_binary_bytes: $phase_binary_bytes,
        reference_binary_bytes: $reference_binary_bytes,
        accepted_default_stdout_bytes: $accepted_stdout_bytes,
        reference_default_stdout_bytes: $reference_stdout_bytes,
        public_manifest_bytes: 472
      },
      controls: {
        parser_and_graph_rejections: 8,
        projection_and_null_rejections: 3,
        dependency_reordered_inverse_rejected: true,
        wrong_and_missing_root_inverse_detected: true,
        snapshot_separate_weaker_path: true,
        retain_all_phase_baseline_equal: true,
        no_smuggle: true
      },
      sha256: {
        phase_source: $phase_hash,
        reference_source: $reference_hash,
        public_manifest: $manifest_hash,
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
