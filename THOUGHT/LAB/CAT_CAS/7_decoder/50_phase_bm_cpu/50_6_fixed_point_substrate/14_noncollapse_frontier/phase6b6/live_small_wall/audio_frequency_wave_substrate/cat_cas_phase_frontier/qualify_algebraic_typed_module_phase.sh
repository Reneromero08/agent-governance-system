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

typed_source="$source_dir/algebraic_typed_module_phase.c"
flat_source="$source_dir/algebraic_series_parallel_phase.c"
primary="$source_dir/typed_module_primary.atrm"
reuse="$source_dir/typed_module_reuse.atrm"
flat_primary="$source_dir/nested_two_cycle_discriminator.aspr"
flat_reuse="$source_dir/nested_two_cycle_reuse.aspr"
recursive_primary="$source_dir/typed_module_recursive_primary.atrm"
recursive_reuse="$source_dir/typed_module_recursive_reuse.atrm"
nominal_negative="$source_dir/negative_typed_module_nominal_mismatch.atrm"
projection_negative="$source_dir/negative_typed_module_nonroot_projection.atrm"

typed="$work_dir/algebraic_typed_module_phase"
typed_analyzer="$work_dir/algebraic_typed_module_phase_analyzer"
typed_sanitized="$work_dir/algebraic_typed_module_phase_sanitized"
flat="$work_dir/algebraic_series_parallel_phase"

gcc \
    -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$typed_source" -lm -o "$typed"
gcc \
    -std=c11 -O2 -Wall -Wextra -Werror -pedantic -fanalyzer \
    "$typed_source" -lm -o "$typed_analyzer"
gcc \
    -std=c11 -O1 -g -Wall -Wextra -Werror -pedantic \
    -fsanitize=address,undefined \
    -fno-sanitize-recover=undefined \
    "$typed_source" -lm -o "$typed_sanitized"
gcc \
    -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$flat_source" -lm -o "$flat"

compiler_builds=4
if command -v clang >/dev/null 2>&1; then
    clang \
        -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
        "$typed_source" -lm -o "$work_dir/algebraic_typed_module_phase_clang"
    compiler_builds=$((compiler_builds + 1))
fi

run_from_source() {
    (
        cd "$source_dir"
        "$@"
    )
}

run_from_source "$typed" \
    "$(basename "$primary")" "$(basename "$reuse")" \
    >"$work_dir/typed.jsonl"
run_from_source "$typed" \
    "$(basename "$primary")" "$(basename "$reuse")" \
    >"$work_dir/typed_replay.jsonl"
cmp "$work_dir/typed.jsonl" "$work_dir/typed_replay.jsonl"

run_from_source "$flat" \
    "$(basename "$flat_primary")" "$(basename "$flat_reuse")" \
    >"$work_dir/flat.jsonl"
run_from_source "$typed" \
    "$(basename "$recursive_primary")" "$(basename "$recursive_reuse")" \
    >"$work_dir/recursive.jsonl"
run_from_source "$typed" \
    "$(basename "$recursive_primary")" "$(basename "$recursive_reuse")" \
    >"$work_dir/recursive_replay.jsonl"
cmp "$work_dir/recursive.jsonl" "$work_dir/recursive_replay.jsonl"

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:abort_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
run_from_source "$typed_sanitized" \
    "$(basename "$primary")" "$(basename "$reuse")" \
    >"$work_dir/sanitized.jsonl"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:abort_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
run_from_source "$typed_sanitized" \
    "$(basename "$recursive_primary")" "$(basename "$recursive_reuse")" \
    >"$work_dir/recursive_sanitized.jsonl"

if run_from_source "$typed" "$(basename "$nominal_negative")" \
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

if run_from_source "$typed" "$(basename "$projection_negative")" \
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
    and .[0].mode == "typed-module-relational-phase"
    and .[0].claim
        == "BOUNDED_RECURSIVE_TYPED_RELATIONAL_MODULE_COMPOSITION"
    and .[0].nominal_domain_count == 3
    and .[0].module_descriptor_count == 3
    and .[0].leaf_module_count == 2
    and .[0].composite_module_count == 1
    and .[0].module_tree_depth == 2
    and .[0].unique_leaf_source_count == 2
    and .[0].module_handoffs_direct
    and .[0].module_export_copy_cells == 0
    and .[0].decoded_module_coefficients == 0
    and .[0].serialized_module_coefficients == 0
    and .[0].expanded_module_truth_tables == 0
    and .[0].input_relation_count == 10
    and .[0].native_composition_operations == 7
    and .[0].native_intersection_operations == 2
    and .[0].phase_resident_relation_messages == 9
    and .[0].carrier_cells == 80
    and .[0].retained_inverse_factors == 0
    and .[0].boundary_coefficients == [0,2,1,2]
    and .[0].maximum_root_error <= 4e-10
    and .[0].restoration_max_abs <= 2e-12
    and .[1].mode == "actual-restored-cross-program-module-reuse"
    and .[1].module_handoffs_direct
    and .[1].boundary_coefficients == [0,1,1,1]
    and .[1].restoration_max_abs <= 2e-12
    and .[2].mode == "wrong-boundary-inverse"
    and .[2].restoration_max_abs >= 0.001
    and .[3].mode == "omitted-resident-module-inverse"
    and .[3].restoration_max_abs >= 0.001
    and .[4].mode == "bypassed-leaf-intersection"
    and .[4].boundary_coefficients != .[0].boundary_coefficients
    and .[4].restoration_max_abs <= 2e-12
    and .[5].mode == "ordinary-sum-leaf-intersection"
    and .[5].boundary_coefficients != .[0].boundary_coefficients
    and .[5].restoration_max_abs <= 2e-12
    and .[6].mode == "module-control-applicability"
    and .[6].wrong_boundary
    and .[6].omitted_resident_module
    and .[6].bypassed_leaf_intersection
    and .[6].ordinary_sum_leaf_intersection
' "$work_dir/typed.jsonl" >/dev/null

jq -s -e '
    length == 7
    and .[0].claim
        == "BOUNDED_RECURSIVE_TYPED_RELATIONAL_MODULE_COMPOSITION"
    and .[0].nominal_domain_count == 5
    and .[0].module_descriptor_count == 7
    and .[0].leaf_module_count == 4
    and .[0].composite_module_count == 3
    and .[0].module_tree_depth == 3
    and .[0].unique_leaf_source_count == 1
    and .[0].compiled_leaf_definition_reuse
    and (.[0].compact_definition_reuse | not)
    and .[0].expanded_native_operation_descriptors == 19
    and .[0].compiled_typed_source_storage_bytes == 8232
    and .[0].compiled_leaf_definition_storage_bytes == 20624
    and .[0].native_operation_descriptor_bytes == 912
    and .[0].module_handoffs_direct
    and .[0].module_export_copy_cells == 0
    and .[0].decoded_module_coefficients == 0
    and .[0].serialized_module_coefficients == 0
    and .[0].input_relation_count == 20
    and .[0].native_composition_operations == 15
    and .[0].native_intersection_operations == 4
    and .[0].phase_resident_relation_messages == 19
    and .[0].carrier_cells == 160
    and .[0].live_carrier_complex_values == 320
    and .[0].verification_peak_complex_values == 640
    and .[0].live_carrier_bytes == 5120
    and .[0].boundary_coefficients == [0,2,2,0]
    and .[0].maximum_root_error <= 4e-10
    and .[0].restoration_max_abs <= 2e-12
    and .[1].mode == "actual-restored-cross-program-module-reuse"
    and .[1].boundary_coefficients == [0,1,1,1]
    and .[1].restoration_max_abs <= 2e-12
    and .[2].restoration_max_abs >= 0.001
    and .[3].restoration_max_abs >= 0.001
    and .[4].boundary_coefficients != .[0].boundary_coefficients
    and .[4].restoration_max_abs <= 2e-12
    and .[6].wrong_boundary
    and .[6].omitted_resident_module
    and .[6].bypassed_leaf_intersection
' "$work_dir/recursive.jsonl" >/dev/null

jq -s -e '
    length == 7
    and .[0].boundary_coefficients == [0,2,1,2]
    and .[1].boundary_coefficients == [0,1,1,1]
' "$work_dir/flat.jsonl" >/dev/null

jq -s -e '
    .[0].boundary_coefficients == .[7].boundary_coefficients
    and .[1].boundary_coefficients == .[8].boundary_coefficients
' \
    "$work_dir/flat.jsonl" \
    "$work_dir/typed.jsonl" >/dev/null

if rg -q \
    'decode_root|module_export_hash|module_export_coefficients' \
    "$typed_source"
then
    echo "typed module compiler contains an intermediate decode/export path" >&2
    exit 1
fi
test "$(rg -n 'decode_root' "$flat_source" | wc -l)" -eq 2
test "$(rg -n 'module_export_copy_cells\\\":0' "$typed_source" | wc -l)" -eq 1
test "$(rg -n 'operation->left_start' "$typed_source" | wc -l)" -ge 1
test "$(rg -n 'operation->right_start' "$typed_source" | wc -l)" -ge 1

sha256sum \
    "$typed_source" \
    "$flat_source" \
    "$primary" \
    "$reuse" \
    "$recursive_primary" \
    "$recursive_reuse" \
    "$source_dir/typed_module_left_primary.aspr" \
    "$source_dir/typed_module_right_primary.aspr" \
    "$source_dir/typed_module_left_reuse.aspr" \
    "$source_dir/typed_module_right_reuse.aspr" \
    "$nominal_negative" \
    "$projection_negative" \
    >"$work_dir/source_sha256.txt"
sha256sum \
    "$work_dir/typed.jsonl" \
    "$work_dir/recursive.jsonl" \
    "$work_dir/flat.jsonl" \
    "$work_dir/sanitized.jsonl" \
    "$work_dir/recursive_sanitized.jsonl" \
    >"$work_dir/result_sha256.txt"

jq -n \
    --argjson rows "$(jq -s '.' "$work_dir/recursive.jsonl")" \
    --argjson shallow "$(jq -s '.' "$work_dir/typed.jsonl")" \
    --argjson compiler_builds "$compiler_builds" \
    '{
        result: "PASS",
        claim: "BOUNDED_RECURSIVE_TYPED_RELATIONAL_MODULE_COMPOSITION",
        compiler_builds: $compiler_builds,
        sanitizers: "PASS",
        primary_boundary: $rows[0].boundary_coefficients,
        reuse_boundary: $rows[1].boundary_coefficients,
        nominal_domains: $rows[0].nominal_domain_count,
        module_descriptors: $rows[0].module_descriptor_count,
        leaf_modules: $rows[0].leaf_module_count,
        composite_modules: $rows[0].composite_module_count,
        module_tree_depth: $rows[0].module_tree_depth,
        unique_leaf_sources: $rows[0].unique_leaf_source_count,
        compiled_leaf_definition_reuse:
            $rows[0].compiled_leaf_definition_reuse,
        compact_definition_reuse: $rows[0].compact_definition_reuse,
        expanded_native_operation_descriptors:
            $rows[0].expanded_native_operation_descriptors,
        compiled_typed_source_storage_bytes:
            $rows[0].compiled_typed_source_storage_bytes,
        compiled_leaf_definition_storage_bytes:
            $rows[0].compiled_leaf_definition_storage_bytes,
        native_operation_descriptor_bytes:
            $rows[0].native_operation_descriptor_bytes,
        direct_phase_resident_handoffs: $rows[0].module_handoffs_direct,
        module_export_copy_cells: $rows[0].module_export_copy_cells,
        decoded_module_coefficients: $rows[0].decoded_module_coefficients,
        input_relations: $rows[0].input_relation_count,
        native_operations:
            ($rows[0].native_composition_operations
                + $rows[0].native_intersection_operations),
        carrier_cells: $rows[0].carrier_cells,
        live_carrier_bytes: $rows[0].live_carrier_bytes,
        restoration_max_abs:
            ([$rows[0].restoration_max_abs, $rows[1].restoration_max_abs]
                | max),
        controls: {
            nominal_type_mismatch: "PASS_REJECTED",
            nonroot_projection: "PASS_REJECTED",
            wrong_inverse: "PASS",
            omitted_resident_module_inverse: "PASS",
            bypassed_leaf_intersection: "PASS",
            ordinary_sum_leaf_intersection: "PASS",
            flat_series_parallel_parity:
                (if $shallow[0].boundary_coefficients == [0,2,1,2]
                    then "PASS" else "FAIL" end)
        },
        terminal: false
    }'
