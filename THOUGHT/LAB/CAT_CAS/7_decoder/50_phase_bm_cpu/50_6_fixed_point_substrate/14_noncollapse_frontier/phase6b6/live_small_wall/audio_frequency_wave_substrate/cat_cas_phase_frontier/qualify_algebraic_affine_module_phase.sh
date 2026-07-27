#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 SOURCE_DIR WORK_DIR" >&2
    exit 2
fi

source_dir=$(cd "$1" && pwd)
work_dir=$2
mkdir -p "$work_dir"
work_dir=$(cd "$work_dir" && pwd)

phase_source="$source_dir/algebraic_affine_module_phase.c"
reference_source="$source_dir/affine_module_reference.c"
kernel_source="$source_dir/algebraic_oblivious_affine_phase.c"
kernel_reference="$source_dir/oblivious_affine_reference.c"
base_source="$source_dir/algebraic_series_parallel_phase.c"
qualifier_source="$source_dir/qualify_algebraic_affine_module_phase.sh"

phase="$work_dir/algebraic_affine_module_phase"
reference="$work_dir/affine_module_reference"
analyzer="$work_dir/algebraic_affine_module_phase_analyzer"
sanitized="$work_dir/algebraic_affine_module_phase_sanitized"
stack_object="$work_dir/algebraic_affine_module_phase_stack.o"

gcc \
    -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$phase_source" -lm -o "$phase"
gcc \
    -std=c11 -O2 -Wall -Wextra -Werror -pedantic -fanalyzer \
    "$phase_source" -lm -o "$analyzer"
gcc \
    -std=c11 -O1 -g -Wall -Wextra -Werror -pedantic \
    -fsanitize=address,undefined -fno-sanitize-recover=undefined \
    "$phase_source" -lm -o "$sanitized"
gcc \
    -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$reference_source" -o "$reference"
gcc \
    -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    -fstack-usage -c "$phase_source" -o "$stack_object"

compiler_builds=5
if command -v clang >/dev/null 2>&1; then
    clang \
        -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
        "$phase_source" -lm -o "$work_dir/phase_clang"
    compiler_builds=$((compiler_builds + 1))
fi

"$phase" >"$work_dir/result.jsonl"
"$phase" >"$work_dir/replay.jsonl"
cmp "$work_dir/result.jsonl" "$work_dir/replay.jsonl"
"$analyzer" >"$work_dir/analyzer.jsonl"
cmp "$work_dir/result.jsonl" "$work_dir/analyzer.jsonl"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
    "$sanitized" >"$work_dir/sanitized.jsonl"
cmp "$work_dir/result.jsonl" "$work_dir/sanitized.jsonl"
"$reference" >"$work_dir/reference.jsonl"

reject() {
    local expected=$1
    local name=$2
    shift 2
    if "$phase" "$@" >"$work_dir/$name.stdout" 2>"$work_dir/$name.stderr"
    then
        echo "expected rejection for $name" >&2
        exit 1
    fi
    test ! -s "$work_dir/$name.stdout"
    grep -Fx "$expected" "$work_dir/$name.stderr" >/dev/null
}

reject \
    "affine module intermediate projection denied" \
    intermediate_projection \
    --project-intermediate
reject \
    "affine module phase-control projection denied" \
    control_projection \
    --project-controls
reject \
    "null affine module carrier" \
    null_carrier \
    --null-carrier
reject \
    "affine module nominal type mismatch" \
    invalid_topology \
    --invalid-topology
reject \
    "usage: algebraic_affine_module_phase" \
    unknown \
    --unknown

execution_keys='[
  "assignment_slots",
  "boundary_coefficients",
  "boundary_decodes",
  "boundary_fnv1a64",
  "carrier_and_snapshot_heap_bytes",
  "carrier_cells",
  "carrier_integrity_max_abs",
  "claim",
  "coexisting_program_storage_bytes_current_abi",
  "complete_boundary_materialized",
  "compose_nodes",
  "conditional_row_adds",
  "conditional_swaps",
  "decoded_intermediate_coefficients",
  "host_selected_pivots",
  "integrity_cell_checks",
  "intersect_nodes",
  "inverse_factor_recomputations",
  "live_carrier_bytes",
  "logical_carrier_cell_inspections",
  "maximum_root_error",
  "mode",
  "native_phase_carrier_reads",
  "nominal_type_validation",
  "operation_calls",
  "phase_ands",
  "phase_cell_updates",
  "phase_nots",
  "phase_ors",
  "phase_xors",
  "public_topology",
  "relation_blocks",
  "relation_block_type_bindings",
  "relation_cells",
  "resident_child_messages",
  "restoration_cell_checks",
  "restoration_max_abs",
  "serialized_intermediate_coefficients",
  "snapshot_creation_complex_values_copied",
  "snapshot_loaded",
  "snapshot_reload_complex_values_copied",
  "tuple_slots",
  "typed_leaf_relations",
  "width",
  "witness_slots",
  "workspace_cell_checks",
  "workspace_cells",
  "workspace_cleared",
  "workspace_scan_passes"
]'
control_keys='[
  "cross_branch_contradiction_empty",
  "altered_algebra_paths_restored",
  "composition_substitution_changes_boundary",
  "different_reuse_boundary",
  "duplicate_intersection_idempotent",
  "fixed_operation_schedule",
  "input_empty_propagated",
  "intersection_left_bypass_changes_boundary",
  "intersection_right_bypass_changes_boundary",
  "missing_root_inverse",
  "missing_root_restoration_abs",
  "mode",
  "nominal_type_validation",
  "reordered_root_inverse",
  "reordered_root_restoration_abs",
  "repeated_reuse_max_restoration_abs",
  "right_operand_high_index_covered",
  "same_carrier_transactions",
  "universal_branch_identity",
  "wrong_root_inverse",
  "wrong_root_restoration_abs"
]'

jq -s -e \
    --argjson execution "$execution_keys" \
    --argjson controls "$control_keys" \
    '
    length == 8
    and all(.[0:7][]; (keys | sort) == ($execution | sort))
    and (.[7] | keys | sort) == ($controls | sort)
    and all(.[0:7][];
        .claim
            == "BOUNDED_WIDTH_PARAMETRIC_AFFINE_COMPOSITION_"
                + "AND_INTERSECTION_MODULE_CLOSURE"
        and .width == 3
        and .public_topology
            == "COMPOSE(INTERSECT(COMPOSE(F,G),P),K)"
        and .nominal_type_validation == true
        and .typed_leaf_relations == 4
        and .compose_nodes == 2
        and .intersect_nodes == 1
        and .resident_child_messages == 2
        and .relation_cells == 49
        and .relation_blocks == 8
        and .relation_block_type_bindings == 7
        and .workspace_cells == 359
        and .carrier_cells == 751
        and .live_carrier_bytes == 24032
        and .carrier_and_snapshot_heap_bytes == 48064
        and .coexisting_program_storage_bytes_current_abi == 1176
        and .decoded_intermediate_coefficients == 0
        and .serialized_intermediate_coefficients == 0
        and .host_selected_pivots == 0
        and .tuple_slots == 0
        and .assignment_slots == 0
        and .witness_slots == 0
        and (.boundary_coefficients | length) == 49
        and all(.boundary_coefficients[]; . == 0 or . == 1)
        and .boundary_decodes == 49
        and .maximum_root_error <= 4e-10
        and .restoration_max_abs <= 2e-12
        and .carrier_integrity_max_abs <= 2e-12
        and .workspace_cleared == true
        and .restoration_cell_checks == 751
        and .integrity_cell_checks == 751
        and .snapshot_creation_complex_values_copied == 1502
    )
    and all(.[0:6][];
        .snapshot_loaded == false
        and .snapshot_reload_complex_values_copied == 0
        and .inverse_factor_recomputations == 3
        and .operation_calls == 6
        and .phase_ands == 29322
        and .phase_xors == 51084
        and .phase_ors == 732
        and .phase_nots == 1428
        and .conditional_swaps == 1296
        and .conditional_row_adds == 1188
        and .phase_cell_updates == 43054
        and .native_phase_carrier_reads == 57638
        and .workspace_scan_passes == 13
        and .workspace_cell_checks == 4667
        and .logical_carrier_cell_inspections == 63856
    )
    and .[0].boundary_coefficients == [
        1,1,0,0,0,0,0,1,
        1,0,1,0,0,1,1,1,
        1,0,0,1,0,1,0,0,
        1,0,0,0,1,0,0,1,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0
    ]
    and .[1].boundary_coefficients == [
        1,1,0,0,0,1,0,0,
        1,0,1,0,0,0,1,1,
        1,0,0,1,0,0,0,0,
        1,0,0,0,1,0,0,1,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0
    ]
    and .[0].boundary_fnv1a64 == "594d12aa095dab79"
    and .[1].boundary_fnv1a64 == "4057ecbc8bf19d29"
    and .[2].boundary_coefficients[-1] == 1
    and .[3].boundary_coefficients[-1] == 0
    and .[4].boundary_coefficients[-1] == 1
    and (
        [range(0;6) as $row
            | .[5].boundary_coefficients[$row * 8]]
        | add
    ) == 2
    and .[6].snapshot_loaded == true
    and .[6].snapshot_reload_complex_values_copied == 751
    and .[6].inverse_factor_recomputations == 0
    and .[6].operation_calls == 3
    and .[6].boundary_coefficients == .[0].boundary_coefficients
    and .[6].boundary_fnv1a64 == .[0].boundary_fnv1a64
    and .[7].different_reuse_boundary == true
    and .[7].input_empty_propagated == true
    and .[7].universal_branch_identity == true
    and .[7].cross_branch_contradiction_empty == true
    and .[7].duplicate_intersection_idempotent == true
    and .[7].right_operand_high_index_covered == true
    and .[7].fixed_operation_schedule == true
    and .[7].wrong_root_inverse == true
    and .[7].missing_root_inverse == true
    and .[7].reordered_root_inverse == true
    and .[7].intersection_left_bypass_changes_boundary == true
    and .[7].intersection_right_bypass_changes_boundary == true
    and .[7].composition_substitution_changes_boundary == true
    and .[7].altered_algebra_paths_restored == true
    and .[7].nominal_type_validation == true
    and .[7].wrong_root_restoration_abs > 1e-6
    and .[7].missing_root_restoration_abs > 1e-6
    and .[7].reordered_root_restoration_abs > 1e-6
    and .[7].same_carrier_transactions == 54
    and .[7].repeated_reuse_max_restoration_abs <= 2e-12
    ' "$work_dir/result.jsonl" >/dev/null

jq -s '
    .[0:6] | map({boundary_coefficients, boundary_fnv1a64})
' "$work_dir/result.jsonl" \
    >"$work_dir/phase_boundaries.json"
jq -s '
    map({boundary_coefficients, boundary_fnv1a64})
' "$work_dir/reference.jsonl" \
    >"$work_dir/reference_boundaries.json"
cmp \
    "$work_dir/phase_boundaries.json" \
    "$work_dir/reference_boundaries.json"

scale_summary="$work_dir/scaling.jsonl"
: >"$scale_summary"
scale_stack_value() {
    local file=$1
    local pattern=$2
    awk -F '\t' -v pattern="$pattern" \
        '$1 ~ pattern {print $2; exit}' "$file"
}

for width in 3 4 8 12 16; do
    scale_phase="$work_dir/phase_w$width"
    scale_reference="$work_dir/reference_w$width"
    scale_stack_object="$work_dir/stack_w$width.o"
    gcc \
        -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
        -DAM_WIDTH="${width}U" \
        -DAM_REUSE_CYCLES=1U \
        "$phase_source" -lm -o "$scale_phase"
    gcc \
        -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
        -DAM_REF_WIDTH="${width}U" \
        "$reference_source" -o "$scale_reference"
    gcc \
        -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
        -DAM_WIDTH="${width}U" \
        -DAM_REUSE_CYCLES=1U \
        -fstack-usage -c "$phase_source" -o "$scale_stack_object"
    compiler_builds=$((compiler_builds + 3))

    scale_stack_file="${scale_stack_object%.o}.su"
    scale_main_stack=$(
        scale_stack_value "$scale_stack_file" ':main$'
    )
    scale_execute_stack=$(
        scale_stack_value "$scale_stack_file" ':am_execute$'
    )
    scale_dispatch_stack=$(
        scale_stack_value "$scale_stack_file" ':am_apply_operation$'
    )
    scale_reduce_stack=$(
        scale_stack_value "$scale_stack_file" ':ga_reduce_stage$'
    )
    scale_swap_stack=$(
        scale_stack_value \
            "$scale_stack_file" \
            ':ga_conditional_swap_rows'
    )
    scale_symbol_stack=$(
        scale_stack_value "$scale_stack_file" ':symbol_product$'
    )
    scale_lock_stack=$(
        scale_stack_value "$scale_stack_file" ':lock_f3_phase$'
    )
    scale_stack_chain=$((
        scale_main_stack
        + scale_execute_stack
        + scale_dispatch_stack
        + scale_reduce_stack
        + scale_swap_stack
        + scale_symbol_stack
        + scale_lock_stack
    ))

    "$scale_phase" >"$work_dir/scale_w$width.phase.jsonl"
    "$scale_reference" >"$work_dir/scale_w$width.reference.jsonl"
    jq -s '
        .[0:6] | map({boundary_coefficients, boundary_fnv1a64})
    ' "$work_dir/scale_w$width.phase.jsonl" \
        >"$work_dir/scale_w$width.phase-boundaries.json"
    jq -s '
        map({boundary_coefficients, boundary_fnv1a64})
    ' "$work_dir/scale_w$width.reference.jsonl" \
        >"$work_dir/scale_w$width.reference-boundaries.json"
    cmp \
        "$work_dir/scale_w$width.phase-boundaries.json" \
        "$work_dir/scale_w$width.reference-boundaries.json"

    jq -s -e --argjson width "$width" '
        length == 8
        and all(.[0:7][];
            .width == $width
            and .relation_cells
                == 4 * $width * $width + 4 * $width + 1
            and .workspace_cells
                == 36 * $width * $width + 11 * $width + 2
            and .carrier_cells
                == 68 * $width * $width + 43 * $width + 10
            and .live_carrier_bytes == 32 * .carrier_cells
            and .carrier_and_snapshot_heap_bytes
                == 64 * .carrier_cells
            and .coexisting_program_storage_bytes_current_abi
                == 24 * .relation_cells
            and .boundary_decodes == .relation_cells
            and .restoration_cell_checks == .carrier_cells
            and .integrity_cell_checks == .carrier_cells
            and .snapshot_creation_complex_values_copied
                == 2 * .carrier_cells
            and (.boundary_coefficients | length) == .relation_cells
            and all(.boundary_coefficients[]; . == 0 or . == 1)
            and .decoded_intermediate_coefficients == 0
            and .serialized_intermediate_coefficients == 0
            and .host_selected_pivots == 0
            and .tuple_slots == 0
            and .assignment_slots == 0
            and .witness_slots == 0
            and .maximum_root_error <= 4e-10
            and .restoration_max_abs <= 2e-12
            and .carrier_integrity_max_abs <= 2e-12
            and .workspace_cleared == true
        )
        and all(.[0:6][];
            .operation_calls == 6
            and .phase_ands
                == 864 * $width * $width * $width
                    + 675 * $width * $width - 27 * $width
            and .phase_xors
                == 1728 * $width * $width * $width
                    + 576 * $width * $width - 252 * $width
            and .phase_ors
                == 72 * $width * $width + 24 * $width + 12
            and .phase_nots
                == 147 * $width * $width + 33 * $width + 6
            and .conditional_swaps == 144 * $width * $width
            and .conditional_row_adds
                == 144 * $width * $width - 36 * $width
            and .phase_cell_updates
                == 1296 * $width * $width * $width
                    + 892 * $width * $width - 2 * $width + 40
            and .native_phase_carrier_reads
                == 1728 * $width * $width * $width
                    + 1235 * $width * $width - 55 * $width + 32
            and .workspace_scan_passes == 13
            and .workspace_cell_checks == 13 * .workspace_cells
            and .logical_carrier_cell_inspections
                == 1728 * $width * $width * $width
                    + 1843 * $width * $width + 178 * $width + 79
            and .inverse_factor_recomputations == 3
            and .snapshot_loaded == false
        )
        and .[6].snapshot_loaded == true
        and .[6].snapshot_reload_complex_values_copied
            == .[6].carrier_cells
        and .[7].fixed_operation_schedule == true
        and .[7].wrong_root_inverse == true
        and .[7].missing_root_inverse == true
        and .[7].reordered_root_inverse == true
        and .[7].intersection_left_bypass_changes_boundary == true
        and .[7].intersection_right_bypass_changes_boundary == true
        and .[7].composition_substitution_changes_boundary == true
        and .[7].altered_algebra_paths_restored == true
        and .[7].nominal_type_validation == true
        and .[7].right_operand_high_index_covered == true
        and .[7].same_carrier_transactions == 7
        and .[7].repeated_reuse_max_restoration_abs <= 2e-12
    ' "$work_dir/scale_w$width.phase.jsonl" >/dev/null

    jq -c -s --argjson stack_chain "$scale_stack_chain" '
        .[0] as $primary
        | .[7] as $controls
        | {
            width: $primary.width,
            relation_cells: $primary.relation_cells,
            dense_relation_entries:
                (pow(2; 2 * $primary.width) | floor),
            workspace_cells: $primary.workspace_cells,
            carrier_cells: $primary.carrier_cells,
            live_carrier_bytes: $primary.live_carrier_bytes,
            phase_ands: $primary.phase_ands,
            phase_xors: $primary.phase_xors,
            phase_cell_updates: $primary.phase_cell_updates,
            native_phase_carrier_reads:
                $primary.native_phase_carrier_reads,
            logical_carrier_cell_inspections:
                $primary.logical_carrier_cell_inspections,
            primary_boundary_fnv1a64:
                $primary.boundary_fnv1a64,
            maximum_root_error: $primary.maximum_root_error,
            restoration_max_abs: $primary.restoration_max_abs,
            repeated_reuse_max_restoration_abs:
                $controls.repeated_reuse_max_restoration_abs,
            compiler_measured_nested_stack_call_chain_bytes:
                $stack_chain,
            conservative_accounted_execution_bytes_current_abi:
                64 * $primary.carrier_cells + $stack_chain,
            reference_match: true
        }
    ' "$work_dir/scale_w$width.phase.jsonl" >>"$scale_summary"
done

jq -s -e '
    map(.width) == [3,4,8,12,16]
    and map(.compiler_measured_nested_stack_call_chain_bytes)
        == [8448,11376,29584,58576,98320]
    and all(.[];
        .relation_cells == 4 * .width * .width + 4 * .width + 1
        and .workspace_cells
            == 36 * .width * .width + 11 * .width + 2
        and .carrier_cells
            == 68 * .width * .width + 43 * .width + 10
        and .maximum_root_error <= 4e-10
        and .restoration_max_abs <= 2e-12
        and .repeated_reuse_max_restoration_abs <= 2e-12
        and .reference_match == true
    )
    and .[-1].width == 16
    and .[-1].relation_cells == 1089
    and .[-1].dense_relation_entries == 4294967296
    and .[-1].workspace_cells == 9394
    and .[-1].carrier_cells == 18106
    and .[-1].live_carrier_bytes == 579392
    and .[-1].compiler_measured_nested_stack_call_chain_bytes
        == 98320
    and .[-1].conservative_accounted_execution_bytes_current_abi
        == 1257104
' "$scale_summary" >/dev/null

test "$(grep -c 'decode_root(' "$kernel_source")" -eq 1
test "$(grep -c 'ga_latch_boundary(' "$kernel_source")" -eq 2
if rg -n \
    -e 'truth_table' \
    -e 'assignment_table' \
    -e 'assignment_expansion' \
    -e 'witness_list' \
    -e 'candidate_set' \
    -e 'dense_relation[[:space:]]*\[' \
    "$phase_source" "$reference_source"
then
    echo "forbidden expanded relation structure in module sources" >&2
    exit 1
fi
if rg -n \
    -e 'displacement_l2' \
    -e 'displacement[[:space:]]*[(][[:space:]]*carrier' \
    "$phase_source" "$kernel_source"
then
    echo "pre-inverse whole-carrier aggregate in module path" >&2
    exit 1
fi
if nm -a "$phase" | rg \
    'affine_module_reference|oblivious_affine_reference| am_ref_| intersect$'
then
    echo "independent reference linked into phase executable" >&2
    exit 1
fi
if nm -a "$phase" | rg '[[:space:]]displacement$'; then
    echo "whole-carrier displacement symbol retained" >&2
    exit 1
fi

command -v strace >/dev/null 2>&1
strace \
    -qq -f \
    -e trace=%file,%network,write,writev,pwrite64,pwritev,pwritev2 \
    -o "$work_dir/no_smuggle.strace" \
    "$phase" >"$work_dir/strace_result.jsonl"
cmp "$work_dir/result.jsonl" "$work_dir/strace_result.jsonl"
if rg \
    'O_WRONLY|O_RDWR|O_CREAT|write(v)?\((2|[3-9]|[1-9][0-9]+),|pwrite|send(to|msg|mmsg)?\(|socket\(|connect\(' \
    "$work_dir/no_smuggle.strace"
then
    echo "unexpected output or file-write channel under strace" >&2
    exit 1
fi

stack_file="${stack_object%.o}.su"
test -s "$stack_file"
main_stack=$(scale_stack_value "$stack_file" ':main$')
execute_stack=$(scale_stack_value "$stack_file" ':am_execute$')
dispatch_stack=$(
    scale_stack_value "$stack_file" ':am_apply_operation$'
)
reduce_stack=$(scale_stack_value "$stack_file" ':ga_reduce_stage$')
swap_stack=$(
    scale_stack_value "$stack_file" ':ga_conditional_swap_rows'
)
symbol_stack=$(scale_stack_value "$stack_file" ':symbol_product$')
lock_stack=$(scale_stack_value "$stack_file" ':lock_f3_phase$')
test "$main_stack" -eq 6064
test "$execute_stack" -eq 848
test "$dispatch_stack" -eq 224
test "$reduce_stack" -eq 208
test "$swap_stack" -eq 960
test "$symbol_stack" -eq 96
test "$lock_stack" -eq 32
nested_stack_chain_bytes=$((
    main_stack
    + execute_stack
    + dispatch_stack
    + reduce_stack
    + swap_stack
    + symbol_stack
    + lock_stack
))
test "$nested_stack_chain_bytes" -eq 8432
material_heap_subtotal_bytes=48064
conservative_accounted_bytes=$((
    material_heap_subtotal_bytes
    + nested_stack_chain_bytes
))
test "$conservative_accounted_bytes" -eq 56496

sha256sum \
    "$phase_source" \
    "$reference_source" \
    "$kernel_source" \
    "$kernel_reference" \
    "$base_source" \
    "$qualifier_source" \
    >"$work_dir/source_sha256.txt"
sha256sum \
    "$work_dir/result.jsonl" \
    "$work_dir/reference.jsonl" \
    "$work_dir/phase_boundaries.json" \
    "$scale_summary" \
    >"$work_dir/result_sha256.txt"

phase_source_sha256=$(sha256sum "$phase_source" | awk '{print $1}')
reference_source_sha256=$(
    sha256sum "$reference_source" | awk '{print $1}'
)
kernel_source_sha256=$(sha256sum "$kernel_source" | awk '{print $1}')
kernel_reference_sha256=$(
    sha256sum "$kernel_reference" | awk '{print $1}'
)
base_source_sha256=$(sha256sum "$base_source" | awk '{print $1}')
qualifier_source_sha256=$(
    sha256sum "$qualifier_source" | awk '{print $1}'
)
result_sha256=$(sha256sum "$work_dir/result.jsonl" | awk '{print $1}')
reference_sha256=$(
    sha256sum "$work_dir/reference.jsonl" | awk '{print $1}'
)
scaling_sha256=$(sha256sum "$scale_summary" | awk '{print $1}')

jq -n \
    --argjson compiler_builds "$compiler_builds" \
    --argjson nested_stack_chain_bytes "$nested_stack_chain_bytes" \
    --argjson material_heap_subtotal_bytes "$material_heap_subtotal_bytes" \
    --argjson conservative_accounted_bytes "$conservative_accounted_bytes" \
    --arg phase_source_sha256 "$phase_source_sha256" \
    --arg reference_source_sha256 "$reference_source_sha256" \
    --arg kernel_source_sha256 "$kernel_source_sha256" \
    --arg kernel_reference_sha256 "$kernel_reference_sha256" \
    --arg base_source_sha256 "$base_source_sha256" \
    --arg qualifier_source_sha256 "$qualifier_source_sha256" \
    --arg result_sha256 "$result_sha256" \
    --arg reference_sha256 "$reference_sha256" \
    --arg scaling_sha256 "$scaling_sha256" \
    --slurpfile results "$work_dir/result.jsonl" \
    --slurpfile scaling "$scale_summary" \
    '{
        result: "PASS",
        claim: $results[0].claim,
        claim_ceiling:
            "BOUNDED_WIDTH16_SOFTWARE_GF2_AFFINE_MIXED_MODULE_"
                + "REFERENCE_ONLY",
        compiler_builds: $compiler_builds,
        sanitizers: "PASS",
        public_topology: $results[0].public_topology,
        semantic_variants: 6,
        phase_reference_exact_matches: 6,
        primary_boundary_fnv1a64: $results[0].boundary_fnv1a64,
        reuse_boundary_fnv1a64: $results[1].boundary_fnv1a64,
        carrier_cells: $results[0].carrier_cells,
        workspace_cells: $results[0].workspace_cells,
        material_heap_subtotal_bytes_current_abi:
            $material_heap_subtotal_bytes,
        compiler_measured_nested_stack_call_chain_bytes:
            $nested_stack_chain_bytes,
        conservative_accounted_execution_bytes_current_abi:
            $conservative_accounted_bytes,
        same_carrier_transactions: $results[7].same_carrier_transactions,
        maximum_reuse_restoration_error:
            $results[7].repeated_reuse_max_restoration_abs,
        fixed_operation_schedule: $results[7].fixed_operation_schedule,
        inverse_controls: {
            wrong: $results[7].wrong_root_inverse,
            missing: $results[7].missing_root_inverse,
            reordered: $results[7].reordered_root_inverse
        },
        altered_algebra_controls: {
            left_bypass_changed:
                $results[7].intersection_left_bypass_changes_boundary,
            right_bypass_changed:
                $results[7].intersection_right_bypass_changes_boundary,
            composition_substitution_changed:
                $results[7].composition_substitution_changes_boundary,
            all_restored: $results[7].altered_algebra_paths_restored,
            nominal_type_validation:
                $results[7].nominal_type_validation
        },
        semantic_controls: {
            input_empty: $results[7].input_empty_propagated,
            universal_intersection_identity:
                $results[7].universal_branch_identity,
            contradiction_empty:
                $results[7].cross_branch_contradiction_empty,
            duplicate_intersection_idempotence:
                $results[7].duplicate_intersection_idempotent,
            right_operand_high_index_covered:
                $results[7].right_operand_high_index_covered
        },
        no_smuggle: {
            decoded_intermediate_coefficients:
                $results[0].decoded_intermediate_coefficients,
            serialized_intermediate_coefficients:
                $results[0].serialized_intermediate_coefficients,
            host_selected_pivots: $results[0].host_selected_pivots,
            tuple_slots: $results[0].tuple_slots,
            assignment_slots: $results[0].assignment_slots,
            witness_slots: $results[0].witness_slots,
            pre_inverse_carrier_aggregate_outputs: 0,
            sole_decode_is_final_boundary: true,
            phase_reference_binary_separation: true,
            strace_no_extra_write_channel: true
        },
        width_scaling: {
            tested_widths: ($scaling | map(.width)),
            exact_phase_reference_matches: ($scaling | length),
            relation_cells_law: "4*w^2 + 4*w + 1",
            workspace_cells_law: "36*w^2 + 11*w + 2",
            carrier_cells_law: "68*w^2 + 43*w + 10",
            maximum: $scaling[-1]
        },
        sha256: {
            phase_source: $phase_source_sha256,
            reference_source: $reference_source_sha256,
            kernel_source: $kernel_source_sha256,
            kernel_reference: $kernel_reference_sha256,
            base_source: $base_source_sha256,
            qualifier_source: $qualifier_source_sha256,
            result: $result_sha256,
            reference: $reference_sha256,
            scaling: $scaling_sha256
        },
        terminal: false
    }'
