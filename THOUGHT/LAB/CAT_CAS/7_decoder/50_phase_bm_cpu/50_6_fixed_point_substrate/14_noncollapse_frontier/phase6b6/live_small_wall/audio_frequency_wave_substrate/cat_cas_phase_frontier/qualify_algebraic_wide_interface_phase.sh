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

wide_source="$source_dir/algebraic_wide_interface_phase.c"
phase_source="$source_dir/algebraic_series_parallel_phase.c"
primary="$source_dir/wide_interface_primary.wr2"
reuse="$source_dir/wide_interface_reuse.wr2"
bad_coefficient="$source_dir/negative_wide_interface_coefficient.wr2"
bad_order="$source_dir/negative_wide_interface_order.wr2"

wide="$work_dir/algebraic_wide_interface_phase"
wide_analyzer="$work_dir/algebraic_wide_interface_phase_analyzer"
wide_sanitized="$work_dir/algebraic_wide_interface_phase_sanitized"

gcc \
    -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$wide_source" -lm -o "$wide"
gcc \
    -std=c11 -O2 -Wall -Wextra -Werror -pedantic -fanalyzer \
    "$wide_source" -lm -o "$wide_analyzer"
gcc \
    -std=c11 -O1 -g -Wall -Wextra -Werror -pedantic \
    -fsanitize=address,undefined \
    -fno-sanitize-recover=undefined \
    "$wide_source" -lm -o "$wide_sanitized"

compiler_builds=3
if command -v clang >/dev/null 2>&1; then
    clang \
        -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
        "$wide_source" -lm -o "$work_dir/algebraic_wide_interface_phase_clang"
    compiler_builds=$((compiler_builds + 1))
fi

run_from_source() {
    (
        cd "$source_dir"
        "$@"
    )
}

run_from_source "$wide" \
    "$(basename "$primary")" "$(basename "$reuse")" \
    >"$work_dir/result.jsonl"
run_from_source "$wide" \
    "$(basename "$primary")" "$(basename "$reuse")" \
    >"$work_dir/replay.jsonl"
cmp "$work_dir/result.jsonl" "$work_dir/replay.jsonl"

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:abort_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
run_from_source "$wide_sanitized" \
    "$(basename "$primary")" "$(basename "$reuse")" \
    >"$work_dir/sanitized.jsonl"
cmp "$work_dir/result.jsonl" "$work_dir/sanitized.jsonl"

reject_fixed() {
    local expected=$1
    shift
    local name=$1
    shift
    if run_from_source "$wide" "$@" \
        >"$work_dir/$name.stdout" \
        2>"$work_dir/$name.stderr"
    then
        echo "$name unexpectedly passed" >&2
        exit 1
    else
        test "$?" -eq 2
    fi
    test ! -s "$work_dir/$name.stdout"
    grep -Fx "$expected" "$work_dir/$name.stderr" >/dev/null
}

reject_fixed \
    "only the final width-two boundary may be projected" \
    projection --project-intermediate
reject_fixed \
    "null carrier or descriptor for CONTRACT2" \
    null_carrier --null-carrier
reject_fixed \
    "invalid BOOLEAN_F3 coefficient at line 2" \
    bad_coefficient "$(basename "$bad_coefficient")"
reject_fixed \
    "invalid ordered wide relation at line 2" \
    bad_order "$(basename "$bad_order")"

execution_keys='[
  "boundary_coefficients",
  "accounted_phase_execution_upper_bound_bytes",
  "accounted_phase_execution_upper_bound_complex_values",
  "accepted_carrier_creation_count",
  "accepted_transactions_on_same_carrier",
  "carrier_and_snapshot_complex_values",
  "carrier_cells",
  "carrier_integrity_max_abs",
  "claim",
  "coefficient_accumulation_additions_per_contract2",
  "comparison_snapshot_complex_values",
  "coexisting_program_storage_bytes",
  "contract2_body_definitions",
  "contract2_descriptor_bytes_each_current_abi",
  "contract2_descriptor_storage_bytes_current_abi",
  "contract2_instance_descriptors",
  "control_rotation_complex_values",
  "decoded_intermediate_coefficients",
  "displacement_l2",
  "forward_contract2_calls_per_transaction",
  "input_relation_encoding_cells",
  "inverse_contract2_calls_per_transaction",
  "live_carrier_bytes",
  "live_carrier_complex_values",
  "maximum_contract2_workspace_complex_values",
  "maximum_root_error",
  "mode",
  "relation_phase_cells",
  "relation_ports",
  "resident_intermediate_cells",
  "public_projection_cells",
  "restoration_max_abs",
  "restriction_and_intersection_additions_per_contract2",
  "retained_boundary_inverse_factor_complex_values",
  "serialized_intermediate_coefficients",
  "shared_assignment_loops",
  "shared_interface_boolean_ports",
  "symbol_product_calls_per_contract2",
  "tuple_witness_truth_table_slots"
]'
for output in result sanitized; do
    jq -s -e --argjson allowed "$execution_keys" '
        length == 9
        and all(.[0:8][]; (keys | sort) == ($allowed | sort))
        and (.[8] | keys | sort) == [
            "bypassed_norm",
            "mode",
            "omitted_parent",
            "ordinary_sum_norm",
            "reordered_noncommuting",
            "swapped_shared_ports",
            "wrong_boundary"
        ]
    ' "$work_dir/$output.jsonl" >/dev/null
done

jq -s -e '
    .[0].mode == "width2-contract-chain"
    and .[0].claim
        == "NATIVE_WIDTH2_TYPED_RELATIONAL_PHASE_CONTRACTION"
    and .[0].shared_interface_boolean_ports == 2
    and .[0].relation_ports == 4
    and .[0].relation_phase_cells == 16
    and .[0].contract2_body_definitions == 1
    and .[0].contract2_instance_descriptors == 2
    and .[0].contract2_descriptor_bytes_each_current_abi == 24
    and .[0].contract2_descriptor_storage_bytes_current_abi == 48
    and .[0].forward_contract2_calls_per_transaction == 2
    and .[0].inverse_contract2_calls_per_transaction == 2
    and .[0].accepted_carrier_creation_count == 1
    and .[0].accepted_transactions_on_same_carrier == 2
    and .[0].input_relation_encoding_cells == 48
    and .[0].resident_intermediate_cells == 16
    and .[0].public_projection_cells == 16
    and .[0].decoded_intermediate_coefficients == 0
    and .[0].serialized_intermediate_coefficients == 0
    and .[0].shared_assignment_loops == 0
    and .[0].tuple_witness_truth_table_slots == 0
    and .[0].carrier_cells == 96
    and .[0].live_carrier_complex_values == 192
    and .[0].live_carrier_bytes == 3072
    and .[0].comparison_snapshot_complex_values == 192
    and .[0].carrier_and_snapshot_complex_values == 384
    and .[0].maximum_contract2_workspace_complex_values == 240
    and .[0].symbol_product_calls_per_contract2 == 1792
    and .[0].coefficient_accumulation_additions_per_contract2 == 1792
    and .[0].restriction_and_intersection_additions_per_contract2 == 112
    and .[0].retained_boundary_inverse_factor_complex_values == 16
    and .[0].control_rotation_complex_values == 16
    and .[0].accounted_phase_execution_upper_bound_complex_values == 656
    and .[0].accounted_phase_execution_upper_bound_bytes == 10496
    and .[0].coexisting_program_storage_bytes == 400
    and .[0].boundary_coefficients
        == [1,0,2,1,2,1,2,1,0,0,1,1,1,1,1,1]
    and .[0].maximum_root_error <= 4e-10
    and .[0].restoration_max_abs <= 2e-12
    and .[1].mode
        == "actual-restored-cross-program-width2-reuse"
    and .[1].boundary_coefficients
        == [1,0,2,1,0,0,1,1,2,1,2,1,1,1,1,1]
    and .[1].restoration_max_abs <= 2e-12
    and .[2].restoration_max_abs >= 0.001
    and .[3].restoration_max_abs >= 0.001
    and .[4].restoration_max_abs >= 0.001
    and .[5].boundary_coefficients
        == [1,1,2,1,2,1,2,1,0,0,1,2,1,2,1,2]
    and .[5].restoration_max_abs <= 2e-12
    and .[6].boundary_coefficients
        == [0,0,2,0,2,0,2,0,0,0,0,0,0,0,0,0]
    and .[6].restoration_max_abs <= 2e-12
    and .[7].boundary_coefficients
        == [1,2,0,1,2,2,1,1,0,1,0,1,1,1,1,1]
    and .[7].restoration_max_abs <= 2e-12
    and .[8].wrong_boundary
    and .[8].omitted_parent
    and .[8].reordered_noncommuting
    and .[8].bypassed_norm
    and .[8].ordinary_sum_norm
    and .[8].swapped_shared_ports
' "$work_dir/result.jsonl" >/dev/null

test "$(rg -n 'decode_root' "$wide_source" | wc -l)" -eq 1
test "$(rg -n 'latch_wide_boundary' "$wide_source" | wc -l)" -eq 2
if sed -n '1,/^static void print_wide_execution(/p' "$wide_source" \
    | rg -q '^[[:space:]]*(printf|puts|fwrite)[[:space:]]*[(]'
then
    echo "wide-interface resident execution contains an output sink" >&2
    exit 1
fi
if rg -q \
    'intermediate_(hash|output|boundary)|decoded_intermediate_relation' \
    "$wide_source"
then
    echo "wide-interface engine contains an intermediate export" >&2
    exit 1
fi
test "$(rg -n 'WIDE_BOUNDARY_START' "$wide_source" | wc -l)" -eq 4
test "$(rg -n 'H_START, K_START, Z_START' "$wide_source" | wc -l)" -eq 1
test "$(rg -n 'F_START, G_START, H_START' "$wide_source" | wc -l)" -eq 2

sha256sum \
    "$wide_source" \
    "$phase_source" \
    "$primary" \
    "$reuse" \
    "$bad_coefficient" \
    "$bad_order" \
    >"$work_dir/source_sha256.txt"
sha256sum \
    "$work_dir/result.jsonl" \
    "$work_dir/replay.jsonl" \
    "$work_dir/sanitized.jsonl" \
    >"$work_dir/result_sha256.txt"

jq -n \
    --argjson rows "$(jq -s '.' "$work_dir/result.jsonl")" \
    --argjson compiler_builds "$compiler_builds" \
    '{
        result: "PASS",
        claim: "NATIVE_WIDTH2_TYPED_RELATIONAL_PHASE_CONTRACTION",
        compiler_builds: $compiler_builds,
        sanitizers: "PASS",
        primary_boundary: $rows[0].boundary_coefficients,
        reuse_boundary: $rows[1].boundary_coefficients,
        shared_interface_boolean_ports: 2,
        relation_phase_cells: 16,
        resident_intermediate_cells: 16,
        contract2_body_definitions: 1,
        contract2_instance_descriptors: 2,
        native_phase_operation_counts: {
            symbol_products_per_contract2: 1792,
            accumulator_additions_per_contract2: 1792,
            restriction_and_intersection_additions_per_contract2: 112
        },
        carrier_cells: 96,
        live_carrier_bytes: 3072,
        accounted_phase_execution_upper_bound_bytes: 10496,
        restoration_max_abs:
            ([$rows[0].restoration_max_abs, $rows[1].restoration_max_abs]
                | max),
        controls: {
            attempted_intermediate_projection: "PASS_REJECTED",
            null_carrier: "PASS_REJECTED",
            malformed_coefficient: "PASS_REJECTED",
            malformed_relation_order: "PASS_REJECTED",
            wrong_boundary_inverse: "PASS",
            omitted_parent_inverse: "PASS",
            reordered_noncommuting_inverse: "PASS",
            bypassed_boolean_norm: "PASS",
            ordinary_sum_boolean_norm: "PASS",
            swapped_shared_ports: "PASS",
            output_allowlist: "PASS",
            preprojection_sink_gate: "PASS"
        },
        separator_resource_law: {
            persistent_relation_cells: "2^(2w)",
            transient_intersection_coefficients: "2^(3w)",
            demonstrated_width: 2
        },
        terminal: false
    }'
