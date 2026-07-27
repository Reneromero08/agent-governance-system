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

tt_source="$source_dir/algebraic_lazy_tt_phase.c"
phase_source="$source_dir/algebraic_series_parallel_phase.c"
primary="$source_dir/lazy_tt_primary.tt3"
reuse="$source_dir/lazy_tt_reuse.tt3"
bad_coefficient="$source_dir/negative_lazy_tt_coefficient.tt3"
bad_rank="$source_dir/negative_lazy_tt_rank.tt3"

tt="$work_dir/algebraic_lazy_tt_phase"
tt_analyzer="$work_dir/algebraic_lazy_tt_phase_analyzer"
tt_sanitized="$work_dir/algebraic_lazy_tt_phase_sanitized"
tt_stack_object="$work_dir/algebraic_lazy_tt_phase_stack.o"

gcc \
    -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$tt_source" -lm -o "$tt"
gcc \
    -std=c11 -O2 -Wall -Wextra -Werror -pedantic -fanalyzer \
    "$tt_source" -lm -o "$tt_analyzer"
gcc \
    -std=c11 -O1 -g -Wall -Wextra -Werror -pedantic \
    -fsanitize=address,undefined \
    -fno-sanitize-recover=undefined \
    "$tt_source" -lm -o "$tt_sanitized"
gcc \
    -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    -fstack-usage -c \
    "$tt_source" -o "$tt_stack_object"

compiler_builds=4
if command -v clang >/dev/null 2>&1; then
    clang \
        -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
        "$tt_source" -lm -o "$work_dir/algebraic_lazy_tt_phase_clang"
    compiler_builds=$((compiler_builds + 1))
fi

run_from_source() {
    (
        cd "$source_dir"
        "$@"
    )
}

run_from_source "$tt" \
    "$(basename "$primary")" "$(basename "$reuse")" \
    >"$work_dir/result.jsonl"
run_from_source "$tt" \
    "$(basename "$primary")" "$(basename "$reuse")" \
    >"$work_dir/replay.jsonl"
cmp "$work_dir/result.jsonl" "$work_dir/replay.jsonl"

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:abort_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
run_from_source "$tt_sanitized" \
    "$(basename "$primary")" "$(basename "$reuse")" \
    >"$work_dir/sanitized.jsonl"
cmp "$work_dir/result.jsonl" "$work_dir/sanitized.jsonl"

reject_fixed() {
    local expected=$1
    shift
    local name=$1
    shift
    if run_from_source "$tt" "$@" \
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
    "only the sparse final lazy-TT boundary may be projected" \
    projection --project-intermediate
reject_fixed \
    "null carrier or lazy descriptor" \
    null_carrier --null-carrier
reject_fixed \
    "eager lazy-TT rank budget exceeded: required rank 16777216" \
    eager_materialization --eager-output-tt
reject_fixed \
    "invalid lazy-TT F3 coefficient at line 4" \
    bad_coefficient "$(basename "$bad_coefficient")"
reject_fixed \
    "invalid ordered lazy-TT rank at line 3" \
    bad_rank "$(basename "$bad_rank")"

execution_keys='[
  "accounted_material_buffer_subtotal_bytes_current_abi",
  "boundary_coefficients",
  "bounded_rank_tt_closure_claimed",
  "carrier_and_snapshot_complex_values",
  "carrier_cells",
  "carrier_integrity_max_abs",
  "carrier_reads",
  "claim",
  "coefficient_additions",
  "compiler_measured_nested_stack_call_chain_bytes",
  "coexisting_program_storage_bytes",
  "comparison_snapshot_complex_values",
  "compiled_descriptor_bytes_current_abi",
  "conservative_accounted_execution_upper_bound_bytes_current_abi",
  "decoded_intermediate_coefficients",
  "dense_phase_cells_per_relation",
  "displacement_l2",
  "eager_structural_ranks",
  "full_output_tensor_materialized",
  "genuine_rank2_minor_f",
  "genuine_rank2_minor_g",
  "input_phase_cell_saving_vs_dense",
  "input_phase_cells_per_relation",
  "input_tt_rank",
  "interface_width",
  "lazy_cache_hits",
  "lazy_cache_misses",
  "lazy_expression_leaf_custody_references",
  "lazy_expression_operator_nodes",
  "live_carrier_bytes",
  "live_carrier_complex_values",
  "maximum_live_sparse_cache_entries",
  "maximum_root_error",
  "mode",
  "or_decomposition_pairs",
  "phase_cell_updates",
  "public_projection_cells",
  "resident_sparse_message_cells",
  "restoration_max_abs",
  "restoration_path",
  "scratch_phase_values_upper_bound",
  "selector_evaluations",
  "serialized_intermediate_coefficients",
  "shared_assignment_loops",
  "snapshot_loaded",
  "sparse_cache_capacity_entries",
  "sparse_cache_entry_bytes_current_abi",
  "sparse_cache_storage_bytes_current_abi",
  "symbol_products",
  "truth_table_slots",
  "tt_coefficient_contractions",
  "tt_evaluator_bytes_current_abi",
  "width1_op_compose_parity_max_abs"
]'
control_keys='[
  "mode",
  "omitted_message_inverse",
  "rank1_bond_cut_changes_boundary",
  "repeated_alternating_reuse",
  "repeated_reuse_max_restoration_abs",
  "reordered_noncommuting_inverse",
  "same_carrier_transactions",
  "snapshot_restores",
  "wrong_g_bond_changes_boundary",
  "wrong_message_inverse"
]'
for output in result sanitized; do
    jq -s -e \
        --argjson execution "$execution_keys" \
        --argjson controls "$control_keys" '
        length == 9
        and all(.[0:8][]; (keys | sort) == ($execution | sort))
        and (.[8] | keys | sort) == ($controls | sort)
    ' "$work_dir/$output.jsonl" >/dev/null
done

jq -s -e '
    .[0].mode == "width3-rank2-lazy-tt"
    and .[0].claim
        == "BOUNDED_WIDTH3_RANK2_PHASE_RESIDENT_LAZY_TT_"
            + "RELATION_COMPOSITION_WITH_SPARSE_BOUNDARY_INVARIANT"
    and .[0].restoration_path == "actual_inverse"
    and .[0].interface_width == 3
    and .[0].input_tt_rank == 2
    and .[0].input_phase_cells_per_relation == 32
    and .[0].dense_phase_cells_per_relation == 64
    and .[0].input_phase_cell_saving_vs_dense == 32
    and .[0].genuine_rank2_minor_f != 0
    and .[0].genuine_rank2_minor_g != 0
    and .[0].lazy_expression_operator_nodes == 6
    and .[0].lazy_expression_leaf_custody_references == 2
    and .[0].full_output_tensor_materialized == false
    and .[0].bounded_rank_tt_closure_claimed == false
    and .[0].decoded_intermediate_coefficients == 0
    and .[0].serialized_intermediate_coefficients == 0
    and .[0].shared_assignment_loops == 0
    and .[0].truth_table_slots == 0
    and .[0].resident_sparse_message_cells == 8
    and .[0].public_projection_cells == 8
    and .[0].carrier_cells == 80
    and .[0].live_carrier_complex_values == 160
    and .[0].live_carrier_bytes == 2560
    and .[0].comparison_snapshot_complex_values == 160
    and .[0].carrier_and_snapshot_complex_values == 320
    and .[0].sparse_cache_capacity_entries == 1024
    and .[0].maximum_live_sparse_cache_entries == 449
    and .[0].scratch_phase_values_upper_bound == 1032
    and .[0].sparse_cache_entry_bytes_current_abi == 32
    and .[0].sparse_cache_storage_bytes_current_abi == 32768
    and .[0].tt_evaluator_bytes_current_abi == 32808
    and .[0].accounted_material_buffer_subtotal_bytes_current_abi == 38688
    and .[0].compiler_measured_nested_stack_call_chain_bytes == 37088
    and .[0].conservative_accounted_execution_upper_bound_bytes_current_abi
        == 75776
    and .[0].eager_structural_ranks
        == [2,4,8,64,4096,16777216]
    and .[0].selector_evaluations == 16
    and .[0].tt_coefficient_contractions == 69376
    and .[0].symbol_products == 668276
    and .[0].coefficient_additions == 470968
    and .[0].or_decomposition_pairs == 113268
    and .[0].carrier_reads == 832528
    and .[0].phase_cell_updates == 160
    and .[0].lazy_cache_hits == 232912
    and .[0].lazy_cache_misses == 2844
    and .[0].compiled_descriptor_bytes_current_abi == 56
    and .[0].coexisting_program_storage_bytes == 576
    and .[0].width1_op_compose_parity_max_abs <= 2e-12
    and .[0].snapshot_loaded == false
    and .[0].boundary_coefficients == [0,0,1,2,2,2,1,2]
    and .[0].maximum_root_error <= 4e-10
    and .[0].restoration_max_abs <= 2e-12
    and .[1].mode
        == "actual-restored-cross-program-width3-tt-reuse"
    and .[1].restoration_path == "actual_inverse"
    and .[1].genuine_rank2_minor_f != 0
    and .[1].genuine_rank2_minor_g != 0
    and .[1].snapshot_loaded == false
    and .[1].boundary_coefficients == [1,1,2,2,1,0,2,0]
    and .[1].restoration_max_abs <= 2e-12
    and .[2].restoration_max_abs >= 0.001
    and .[3].restoration_max_abs >= 0.001
    and .[4].restoration_max_abs >= 0.001
    and .[5].boundary_coefficients == [1,0,0,0,0,0,0,0]
    and .[5].restoration_max_abs <= 2e-12
    and .[6].boundary_coefficients == [0,0,2,1,2,1,0,2]
    and .[6].restoration_max_abs <= 2e-12
    and .[7].restoration_path == "snapshot_reload"
    and .[7].snapshot_loaded
    and .[7].restoration_max_abs <= 2e-12
    and .[8].wrong_message_inverse
    and .[8].omitted_message_inverse
    and .[8].reordered_noncommuting_inverse
    and .[8].rank1_bond_cut_changes_boundary
    and .[8].wrong_g_bond_changes_boundary
    and .[8].snapshot_restores
    and .[8].repeated_alternating_reuse
    and .[8].same_carrier_transactions == 34
    and .[8].repeated_reuse_max_restoration_abs <= 2e-12
' "$work_dir/result.jsonl" >/dev/null

test "$(rg -n 'decode_root' "$tt_source" | wc -l)" -eq 1
test "$(rg -n 'latch_tt_boundary' "$tt_source" | wc -l)" -eq 2
if sed -n '1,/^static void print_tt_execution(/p' "$tt_source" \
    | rg -q '^[[:space:]]*(printf|puts|fwrite)[[:space:]]*[(]'
then
    echo "lazy-TT resident execution contains an output sink" >&2
    exit 1
fi
if rg -q \
    'intermediate_(hash|output|boundary)|decoded_intermediate_relation' \
    "$tt_source"
then
    echo "lazy-TT engine contains an intermediate export" >&2
    exit 1
fi
if rg -q \
    'double complex [A-Za-z_][A-Za-z0-9_]*\[64(U)?\]|assignment_expansion|truth_table\[' \
    "$tt_source"
then
    echo "lazy-TT engine contains a dense relation or assignment table" >&2
    exit 1
fi
test "$(rg -n 'relative\(' "$tt_source" | wc -l)" -eq 3
test "$(rg -n 'TT_MESSAGE_START' "$tt_source" | wc -l)" -eq 2
test "$(rg -n 'TT_BOUNDARY_START' "$tt_source" | wc -l)" -eq 2

stack_usage="$work_dir/algebraic_lazy_tt_phase_stack.su"
test -s "$stack_usage"
stack_bytes() {
    local function_name=$1
    awk -F '\t' -v function_name="$function_name" '
        $1 ~ (":" function_name "(\\.constprop)?$") {
            print $2
            found = 1
        }
        END {
            if (!found) {
                exit 1
            }
        }
    ' "$stack_usage"
}
main_stack=$(stack_bytes main)
execute_stack=$(stack_bytes execute_tt)
apply_stack=$(stack_bytes apply_tt_messages)
evaluate_stack=$(stack_bytes evaluate_tt_selector)
lazy_stack=$(stack_bytes lazy_tt_coefficient)
square_stack=$(stack_bytes tt_square_coefficient)
coefficient_stack=$(stack_bytes tt_coefficient)
measured_stack_call_chain=$((
    main_stack
    + execute_stack
    + apply_stack
    + evaluate_stack
    + 4 * lazy_stack
    + square_stack
    + coefficient_stack
))
test "$measured_stack_call_chain" -eq 37088

sha256sum \
    "$tt_source" \
    "$phase_source" \
    "$primary" \
    "$reuse" \
    "$bad_coefficient" \
    "$bad_rank" \
    >"$work_dir/source_sha256.txt"
sha256sum \
    "$work_dir/result.jsonl" \
    "$work_dir/replay.jsonl" \
    "$work_dir/sanitized.jsonl" \
    >"$work_dir/result_sha256.txt"

jq -n \
    --argjson rows "$(jq -s '.' "$work_dir/result.jsonl")" \
    --argjson compiler_builds "$compiler_builds" \
    --argjson measured_stack_call_chain "$measured_stack_call_chain" \
    '{
        result: "PASS",
        claim:
            "BOUNDED_WIDTH3_RANK2_PHASE_RESIDENT_LAZY_TT_"
            + "RELATION_COMPOSITION_WITH_SPARSE_BOUNDARY_INVARIANT",
        compiler_builds: $compiler_builds,
        sanitizers: "PASS",
        primary_boundary: $rows[0].boundary_coefficients,
        reuse_boundary: $rows[1].boundary_coefficients,
        interface_width: 3,
        input_tt_rank: 2,
        input_phase_cells_per_relation: 32,
        dense_phase_cells_per_relation: 64,
        resident_sparse_message_cells: 8,
        carrier_cells: 80,
        live_carrier_bytes: 2560,
        maximum_live_sparse_cache_entries:
            $rows[0].maximum_live_sparse_cache_entries,
        accounted_material_buffer_subtotal_bytes_current_abi:
            $rows[0].accounted_material_buffer_subtotal_bytes_current_abi,
        compiler_measured_nested_stack_call_chain_bytes:
            $measured_stack_call_chain,
        conservative_accounted_execution_upper_bound_bytes_current_abi:
            $rows[0]
                .conservative_accounted_execution_upper_bound_bytes_current_abi,
        restoration_max_abs:
            ([$rows[0].restoration_max_abs, $rows[1].restoration_max_abs]
                | max),
        controls: {
            attempted_intermediate_projection: "PASS_REJECTED",
            null_carrier: "PASS_REJECTED",
            eager_output_tt: "PASS_REJECTED_AT_RANK_CAP",
            malformed_coefficient: "PASS_REJECTED",
            malformed_rank: "PASS_REJECTED",
            wrong_sparse_message_inverse: "PASS",
            omitted_sparse_message_inverse: "PASS",
            reordered_noncommuting_inverse: "PASS",
            rank1_bond_cut: "PASS_DISCRIMINATED_AND_RESTORED",
            wrong_g_virtual_bond:
                "PASS_DISCRIMINATED_AND_RESTORED",
            snapshot_backed_comparison: "PASS_SEPARATE",
            repeated_alternating_restored_carrier_reuse: "PASS_34",
            width1_op_compose_parity: "PASS",
            output_allowlist: "PASS",
            preprojection_sink_gate: "PASS",
            dense_relation_static_gate: "PASS"
        },
        rank_disclosure: {
            input: 2,
            square: 4,
            intersection: 8,
            norm_y0: 64,
            norm_y1: 4096,
            norm_y2: 16777216,
            bounded_rank_closure: false
        },
        terminal: false
    }'
