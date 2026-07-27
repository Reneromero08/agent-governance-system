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

compact_source="$source_dir/algebraic_compact_module_phase.c"
typed_source="$source_dir/algebraic_typed_module_phase.c"
phase_source="$source_dir/algebraic_series_parallel_phase.c"
depth4_primary="$source_dir/compact_module_depth4_primary.atrm"
depth4_reuse="$source_dir/compact_module_depth4_reuse.atrm"
shallow_primary="$source_dir/typed_module_primary.atrm"
shallow_reuse="$source_dir/typed_module_reuse.atrm"
flat_primary="$source_dir/nested_two_cycle_discriminator.aspr"
flat_reuse="$source_dir/nested_two_cycle_reuse.aspr"
nominal_negative="$source_dir/negative_typed_module_nominal_mismatch.atrm"
projection_negative="$source_dir/negative_typed_module_nonroot_projection.atrm"

compact="$work_dir/algebraic_compact_module_phase"
compact_analyzer="$work_dir/algebraic_compact_module_phase_analyzer"
compact_sanitized="$work_dir/algebraic_compact_module_phase_sanitized"
typed="$work_dir/algebraic_typed_module_phase"
flat="$work_dir/algebraic_series_parallel_phase"

gcc \
    -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$compact_source" -lm -o "$compact"
gcc \
    -std=c11 -O2 -Wall -Wextra -Werror -pedantic -fanalyzer \
    "$compact_source" -lm -o "$compact_analyzer"
gcc \
    -std=c11 -O1 -g -Wall -Wextra -Werror -pedantic \
    -fsanitize=address,undefined \
    -fno-sanitize-recover=undefined \
    "$compact_source" -lm -o "$compact_sanitized"
gcc \
    -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$typed_source" -lm -o "$typed"
gcc \
    -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$phase_source" -lm -o "$flat"

compiler_builds=5
if command -v clang >/dev/null 2>&1; then
    clang \
        -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
        "$compact_source" -lm -o "$work_dir/algebraic_compact_module_phase_clang"
    compiler_builds=$((compiler_builds + 1))
fi

run_from_source() {
    (
        cd "$source_dir"
        "$@"
    )
}

run_from_source "$compact" \
    "$(basename "$depth4_primary")" "$(basename "$depth4_reuse")" \
    >"$work_dir/depth4.jsonl"
run_from_source "$compact" \
    "$(basename "$depth4_primary")" "$(basename "$depth4_reuse")" \
    >"$work_dir/depth4_replay.jsonl"
cmp "$work_dir/depth4.jsonl" "$work_dir/depth4_replay.jsonl"

run_from_source "$compact" \
    "$(basename "$shallow_primary")" "$(basename "$shallow_reuse")" \
    >"$work_dir/shallow.jsonl"
run_from_source "$typed" \
    "$(basename "$depth4_primary")" "$(basename "$depth4_reuse")" \
    >"$work_dir/expanded_depth4.jsonl"
run_from_source "$flat" \
    "$(basename "$flat_primary")" "$(basename "$flat_reuse")" \
    >"$work_dir/flat.jsonl"

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:abort_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
run_from_source "$compact_sanitized" \
    "$(basename "$depth4_primary")" "$(basename "$depth4_reuse")" \
    >"$work_dir/depth4_sanitized.jsonl"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:abort_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
run_from_source "$compact_sanitized" \
    "$(basename "$shallow_primary")" "$(basename "$shallow_reuse")" \
    >"$work_dir/shallow_sanitized.jsonl"

if run_from_source "$compact" "$(basename "$nominal_negative")" \
    >"$work_dir/nominal_negative.stdout" \
    2>"$work_dir/nominal_negative.stderr"
then
    echo "nominal type mismatch fixture unexpectedly passed" >&2
    exit 1
else
    test "$?" -eq 2
fi
test ! -s "$work_dir/nominal_negative.stdout"
grep -Fx \
    "nominal module interface type mismatch at line 8" \
    "$work_dir/nominal_negative.stderr" >/dev/null

if run_from_source "$compact" "$(basename "$projection_negative")" \
    >"$work_dir/projection_negative.stdout" \
    2>"$work_dir/projection_negative.stderr"
then
    echo "non-root projection fixture unexpectedly passed" >&2
    exit 1
else
    test "$?" -eq 2
fi
test ! -s "$work_dir/projection_negative.stdout"
grep -Fx \
    "typed-module source is incomplete" \
    "$work_dir/projection_negative.stderr" >/dev/null

jq -s -e '
    length == 7
    and .[0].mode == "compact-typed-module-phase"
    and .[0].claim
        == "COMPACT_COMPILED_BODY_TYPED_RELATIONAL_MODULE_EXECUTION"
    and .[0].module_tree_depth == 4
    and .[0].module_descriptors == 15
    and .[0].unique_leaf_sources == 1
    and .[0].leaf_instances == 8
    and .[0].compiled_leaf_definition_reuse
    and .[0].compact_definition_execution
    and .[0].persistent_per_instance_native_operation_descriptors == 0
    and .[0].persistent_unique_leaf_operation_descriptors == 4
    and .[0].persistent_composite_module_descriptors == 7
    and .[0].transient_operation_descriptors == 1
    and .[0].transient_operation_descriptor_bytes == 48
    and .[0].executed_native_operations == 39
    and .[0].phase_resident_operation_messages == 39
    and .[0].module_export_copy_cells == 0
    and .[0].decoded_module_coefficients == 0
    and .[0].serialized_module_coefficients == 0
    and .[0].input_relation_count == 40
    and .[0].carrier_cells == 320
    and .[0].live_carrier_complex_values == 640
    and .[0].live_carrier_bytes == 10240
    and .[0].carrier_and_snapshot_complex_values == 1280
    and .[0].maximum_native_operator_workspace_complex_values == 52
    and .[0].retained_boundary_inverse_factor_sets == 1
    and .[0].retained_boundary_inverse_factor_complex_values == 4
    and .[0].control_rotation_complex_values == 4
    and .[0].logical_phase_execution_peak_complex_values == 1340
    and .[0].logical_phase_execution_peak_bytes == 21440
    and .[0].coexisting_typed_source_storage_bytes == 16464
    and .[0].coexisting_leaf_definition_storage_bytes == 41248
    and .[0].coexisting_layout_storage_bytes == 112
    and .[0].boundary_coefficients == [0,2,2,0]
    and .[0].maximum_root_error <= 4e-10
    and .[0].restoration_max_abs <= 2e-12
    and .[1].mode
        == "actual-restored-cross-program-compact-module-reuse"
    and .[1].boundary_coefficients == [0,1,1,1]
    and .[1].restoration_max_abs <= 2e-12
    and .[2].restoration_max_abs >= 0.001
    and .[3].restoration_max_abs >= 0.001
    and .[4].boundary_coefficients != .[0].boundary_coefficients
    and .[4].restoration_max_abs <= 2e-12
    and .[6].wrong_boundary
    and .[6].omitted_parent
    and .[6].bypassed_leaf_intersection
' "$work_dir/depth4.jsonl" >/dev/null

jq -s -e '
    .[0].boundary_coefficients == [0,2,1,2]
    and .[1].boundary_coefficients == [0,1,1,1]
    and .[0].leaf_instances == 2
    and .[0].unique_leaf_sources == 2
    and (.[] | select(.mode != "compact-module-control-applicability")
        | .compiled_leaf_definition_reuse == false)
    and .[4].boundary_coefficients != .[0].boundary_coefficients
    and .[5].boundary_coefficients != .[0].boundary_coefficients
    and .[6].ordinary_sum_leaf_intersection
' "$work_dir/shallow.jsonl" >/dev/null

jq -s -e '
    .[0].boundary_coefficients == .[7].boundary_coefficients
    and .[1].boundary_coefficients == .[8].boundary_coefficients
    and .[0].executed_native_operations
        == .[7].expanded_native_operation_descriptors
    and .[7].expanded_native_operation_descriptors == 39
    and .[7].compact_definition_reuse == false
' \
    "$work_dir/depth4.jsonl" \
    "$work_dir/expanded_depth4.jsonl" >/dev/null

jq -s -e '
    .[0].boundary_coefficients == .[7].boundary_coefficients
    and .[1].boundary_coefficients == .[8].boundary_coefficients
' \
    "$work_dir/flat.jsonl" \
    "$work_dir/shallow.jsonl" >/dev/null

execution_keys='[
  "boundary_coefficients",
  "carrier_and_snapshot_complex_values",
  "carrier_cells",
  "carrier_integrity_max_abs",
  "claim",
  "coexisting_layout_storage_bytes",
  "coexisting_leaf_definition_storage_bytes",
  "coexisting_typed_source_storage_bytes",
  "compact_definition_execution",
  "compiled_leaf_definition_reuse",
  "control_rotation_complex_values",
  "decoded_module_coefficients",
  "displacement_l2",
  "executed_native_operations",
  "input_relation_count",
  "leaf_instances",
  "live_carrier_bytes",
  "live_carrier_complex_values",
  "logical_phase_execution_peak_bytes",
  "logical_phase_execution_peak_complex_values",
  "maximum_native_operator_workspace_complex_values",
  "maximum_root_error",
  "mode",
  "module_descriptors",
  "module_export_copy_cells",
  "module_tree_depth",
  "persistent_composite_module_descriptors",
  "persistent_per_instance_native_operation_descriptors",
  "persistent_unique_leaf_operation_descriptors",
  "phase_resident_operation_messages",
  "restoration_max_abs",
  "retained_boundary_inverse_factor_complex_values",
  "retained_boundary_inverse_factor_sets",
  "serialized_module_coefficients",
  "transient_operation_descriptor_bytes",
  "transient_operation_descriptors",
  "unique_leaf_sources"
]'
for output in depth4 shallow depth4_sanitized shallow_sanitized; do
    jq -s -e --argjson allowed "$execution_keys" '
        length == 7
        and all(.[0:6][]; (keys | sort) == ($allowed | sort))
        and (.[6] | keys | sort) == [
            "bypassed_leaf_intersection",
            "mode",
            "omitted_parent",
            "ordinary_sum_leaf_intersection",
            "wrong_boundary"
        ]
    ' "$work_dir/$output.jsonl" >/dev/null
done

test "$(rg -n 'decode_root' "$compact_source" | wc -l)" -eq 1
test "$(rg -n 'compact_latch_boundary' "$compact_source" | wc -l)" -eq 2
test "$(
    rg -n '^[[:space:]]*(printf|puts|fwrite)[[:space:]]*[(]' \
        "$compact_source" | wc -l
)" -eq 2
if sed -n '1,/^static void print_compact_execution(/p' "$compact_source" \
    | rg -q '^[[:space:]]*(printf|puts|fwrite)[[:space:]]*[(]'
then
    echo "compact resident execution contains an output sink" >&2
    exit 1
fi
if rg -q \
    'module_export_hash|module_export_coefficients|decoded_module_relation' \
    "$compact_source"
then
    echo "compact engine contains a module export side channel" >&2
    exit 1
fi
test "$(rg -n 'persistent_per_instance_native_operation_descriptors\\\":0' "$compact_source" | wc -l)" -eq 1
test "$(rg -n 'struct operation relocated' "$compact_source" | wc -l)" -eq 3

sha256sum \
    "$compact_source" \
    "$typed_source" \
    "$phase_source" \
    "$depth4_primary" \
    "$depth4_reuse" \
    "$shallow_primary" \
    "$shallow_reuse" \
    "$source_dir/typed_module_left_primary.aspr" \
    "$source_dir/typed_module_left_reuse.aspr" \
    "$source_dir/typed_module_right_primary.aspr" \
    "$source_dir/typed_module_right_reuse.aspr" \
    "$nominal_negative" \
    "$projection_negative" \
    >"$work_dir/source_sha256.txt"
sha256sum \
    "$work_dir/depth4.jsonl" \
    "$work_dir/shallow.jsonl" \
    "$work_dir/expanded_depth4.jsonl" \
    "$work_dir/flat.jsonl" \
    "$work_dir/depth4_sanitized.jsonl" \
    "$work_dir/shallow_sanitized.jsonl" \
    >"$work_dir/result_sha256.txt"

jq -n \
    --argjson rows "$(jq -s '.' "$work_dir/depth4.jsonl")" \
    --argjson compiler_builds "$compiler_builds" \
    '{
        result: "PASS",
        claim:
            "COMPACT_COMPILED_BODY_TYPED_RELATIONAL_MODULE_EXECUTION",
        compiler_builds: $compiler_builds,
        sanitizers: "PASS_DEPTH4_AND_SHALLOW",
        primary_boundary: $rows[0].boundary_coefficients,
        reuse_boundary: $rows[1].boundary_coefficients,
        module_tree_depth: $rows[0].module_tree_depth,
        module_descriptors: $rows[0].module_descriptors,
        unique_leaf_sources: $rows[0].unique_leaf_sources,
        persistent_per_instance_native_operation_descriptors:
            $rows[0].persistent_per_instance_native_operation_descriptors,
        persistent_unique_leaf_operation_descriptors:
            $rows[0].persistent_unique_leaf_operation_descriptors,
        persistent_composite_module_descriptors:
            $rows[0].persistent_composite_module_descriptors,
        transient_operation_descriptors:
            $rows[0].transient_operation_descriptors,
        executed_native_operations:
            $rows[0].executed_native_operations,
        phase_resident_operation_messages:
            $rows[0].phase_resident_operation_messages,
        carrier_cells: $rows[0].carrier_cells,
        live_carrier_bytes: $rows[0].live_carrier_bytes,
        restoration_max_abs:
            ([$rows[0].restoration_max_abs, $rows[1].restoration_max_abs]
                | max),
        descriptor_comparison: {
            expanded_backend_native_records: 39,
            compact_unique_leaf_records: 4,
            compact_composite_module_descriptors: 7,
            compact_transient_records: 1
        },
        controls: {
            nominal_type_mismatch: "PASS_REJECTED",
            nonroot_projection: "PASS_REJECTED",
            wrong_inverse: "PASS",
            omitted_parent_inverse: "PASS",
            bypassed_leaf_intersection: "PASS",
            ordinary_sum_leaf_intersection: "PASS_SHALLOW",
            expanded_typed_parity: "PASS",
            flat_series_parallel_parity: "PASS_SHALLOW"
        },
        terminal: false
    }'
