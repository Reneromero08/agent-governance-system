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

phase_source="$source_dir/algebraic_oblivious_affine_phase.c"
reference_source="$source_dir/oblivious_affine_reference.c"
base_source="$source_dir/algebraic_series_parallel_phase.c"
qualifier_source="$source_dir/qualify_algebraic_oblivious_affine_phase.sh"

phase="$work_dir/algebraic_oblivious_affine_phase"
reference="$work_dir/oblivious_affine_reference"
analyzer="$work_dir/algebraic_oblivious_affine_phase_analyzer"
sanitized="$work_dir/algebraic_oblivious_affine_phase_sanitized"
stack_object="$work_dir/algebraic_oblivious_affine_phase_stack.o"

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
        "$phase_source" -lm \
        -o "$work_dir/algebraic_oblivious_affine_phase_clang"
    compiler_builds=$((compiler_builds + 1))
fi

"$phase" >"$work_dir/result.jsonl"
"$phase" >"$work_dir/replay.jsonl"
cmp "$work_dir/result.jsonl" "$work_dir/replay.jsonl"
"$analyzer" >"$work_dir/analyzer.jsonl"
cmp "$work_dir/result.jsonl" "$work_dir/analyzer.jsonl"

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:abort_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$sanitized" >"$work_dir/sanitized.jsonl"
cmp "$work_dir/result.jsonl" "$work_dir/sanitized.jsonl"

"$reference" >"$work_dir/reference.jsonl"
jq -s '
    .[0:9] | map({boundary_coefficients, boundary_fnv1a64})
' "$work_dir/result.jsonl" >"$work_dir/phase_boundaries.json"
jq -s '
    map({boundary_coefficients, boundary_fnv1a64})
' "$work_dir/reference.jsonl" >"$work_dir/reference_boundaries.json"
cmp "$work_dir/phase_boundaries.json" "$work_dir/reference_boundaries.json"

reject_fixed() {
    local expected=$1
    local name=$2
    shift 2
    if "$phase" "$@" \
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
    "only final affine equation boundary may be projected" \
    intermediate_projection \
    --project-intermediate
reject_fixed \
    "phase pivot and elimination controls are not projectable" \
    control_projection \
    --project-controls
reject_fixed \
    "null oblivious affine carrier" \
    null_carrier \
    --null-carrier
reject_fixed \
    "usage: algebraic_oblivious_affine_phase" \
    unknown \
    --unknown

execution_keys='[
  "assignment_slots",
  "boundary_coefficients",
  "boundary_decodes",
  "boundary_fnv1a64",
  "canonical_boundary",
  "carrier_and_snapshot_heap_bytes",
  "carrier_cells",
  "carrier_integrity_max_abs",
  "carrier_reads",
  "claim",
  "coexisting_program_storage_bytes_current_abi",
  "comparison_snapshot_bytes",
  "comparison_snapshot_complex_values",
  "compiled_descriptor_bytes_current_abi",
  "complete_boundary_materialized",
  "compose_calls",
  "conditional_row_adds",
  "conditional_swaps",
  "decoded_intermediate_coefficients",
  "dense_relation_entries",
  "displacement_l2",
  "execution_result_bytes_current_abi",
  "fixed_host_loop_address_schedule",
  "host_selected_pivots",
  "inverse_factor_recomputations",
  "joint_workspace_cells",
  "live_carrier_bytes",
  "live_carrier_complex_values",
  "maximum_root_error",
  "mode",
  "phase_ands",
  "phase_cell_updates",
  "phase_control_cells",
  "phase_nots",
  "phase_ors",
  "phase_xors",
  "polynomial_descriptor_cells",
  "program_storage_bytes_current_abi",
  "public_program_coefficients",
  "relation_cells",
  "relation_family",
  "relation_rows",
  "restoration_cell_checks",
  "restoration_max_abs",
  "restoration_path",
  "serialized_intermediate_coefficients",
  "snapshot_loaded",
  "total_workspace_cells",
  "tuple_slots",
  "width",
  "witness_slots",
  "workspace_clear_tolerance",
  "workspace_cleared",
  "workspace_tolerance_checks"
]'
control_keys='[
  "canonical_full_rank_boundary",
  "different_reuse_boundary",
  "duplicate_equation_invariant",
  "fixed_operation_schedule",
  "fixture_hashes_match",
  "input_empty_propagated",
  "missing_parent_inverse",
  "mode",
  "multi_contradiction_empty",
  "rank0_universal",
  "rank1_universal",
  "reordered_inverse",
  "reordered_restoration_abs",
  "repeated_reuse_max_restoration_abs",
  "row_permutation_invariant",
  "same_carrier_transactions",
  "wrong_parent_inverse",
  "wrong_parent_restoration_abs",
  "missing_parent_restoration_abs"
]'

for output in result analyzer sanitized; do
    jq -s -e \
        --argjson execution "$execution_keys" \
        --argjson controls "$control_keys" '
        length == 11
        and all(.[0:10][]; (keys | sort) == ($execution | sort))
        and (.[10] | keys | sort) == ($controls | sort)
    ' "$work_dir/$output.jsonl" >/dev/null
done

jq -s -e '
    .[0].mode == "oblivious-affine-primary"
    and .[1].mode == "oblivious-affine-reuse"
    and .[2].mode == "oblivious-affine-permuted"
    and .[3].mode == "oblivious-affine-duplicated"
    and .[4].mode == "oblivious-affine-universal-rank0"
    and .[5].mode == "oblivious-affine-universal-rank1"
    and .[6].mode == "oblivious-affine-input-empty"
    and .[7].mode == "oblivious-affine-composed-empty"
    and .[8].mode == "oblivious-affine-full-rank"
    and .[9].mode == "oblivious-affine-snapshot"
    and .[10].mode == "controls"
    and all(.[0:10][];
        .claim
            == "BOUNDED_WIDTH2_COEFFICIENT_OBLIVIOUS_GENERAL_"
                + "AFFINE_PHASE_RELATION_COMPOSITION"
        and .relation_family
            == "GENERAL_AFFINE_F2_EQUATION_SYSTEMS_FIXED_WIDTH2"
        and .width == 2
        and .relation_rows == 4
        and .relation_cells == 25
        and .complete_boundary_materialized == true
        and .canonical_boundary == true
        and .dense_relation_entries == 16
        and .polynomial_descriptor_cells == 25
        and .joint_workspace_cells == 56
        and .phase_control_cells == 112
        and .total_workspace_cells == 168
        and .carrier_cells == 318
        and .live_carrier_complex_values == 636
        and .live_carrier_bytes == 10176
        and .comparison_snapshot_complex_values == 636
        and .comparison_snapshot_bytes == 10176
        and .carrier_and_snapshot_heap_bytes == 20352
        and .public_program_coefficients == 75
        and .program_storage_bytes_current_abi == 75
        and .coexisting_program_storage_bytes_current_abi == 150
        and .compiled_descriptor_bytes_current_abi == 48
        and .execution_result_bytes_current_abi == 240
        and .decoded_intermediate_coefficients == 0
        and .serialized_intermediate_coefficients == 0
        and .host_selected_pivots == 0
        and .fixed_host_loop_address_schedule == true
        and .tuple_slots == 0
        and .assignment_slots == 0
        and .witness_slots == 0
        and .restoration_cell_checks == 318
        and (.boundary_coefficients | length) == 25
        and all(.boundary_coefficients[]; . == 0 or . == 1)
        and .boundary_decodes == 25
        and .maximum_root_error <= 4e-10
        and .restoration_max_abs <= 2e-12
        and .carrier_integrity_max_abs <= 2e-12
        and .workspace_cleared == true
        and .workspace_clear_tolerance == 2e-12
    )
    and all(.[0:9][];
        .restoration_path == "actual_inverse"
        and .snapshot_loaded == false
        and .compose_calls == 4
        and .phase_ands == 6372
        and .phase_xors == 10416
        and .phase_ors == 232
        and .phase_nots == 440
        and .conditional_swaps == 384
        and .conditional_row_adds == 336
        and .phase_cell_updates == 9348
        and .carrier_reads == 12474
        and .workspace_tolerance_checks == 8
        and .inverse_factor_recomputations == 2
    )
    and .[9].restoration_path == "snapshot_reload"
    and .[9].snapshot_loaded == true
    and .[9].compose_calls == 2
    and .[9].inverse_factor_recomputations == 0
    and .[0].boundary_fnv1a64 == "a96f5625e4054ee6"
    and .[1].boundary_fnv1a64 == "0183d6882c23ee46"
    and .[2].boundary_fnv1a64 == .[0].boundary_fnv1a64
    and .[3].boundary_fnv1a64 == .[0].boundary_fnv1a64
    and .[4].boundary_fnv1a64 == "d4657f55662f817f"
    and .[5].boundary_fnv1a64 == .[4].boundary_fnv1a64
    and .[6].boundary_fnv1a64 == "d4657e55662f7fcc"
    and .[7].boundary_fnv1a64 == .[6].boundary_fnv1a64
    and .[8].boundary_fnv1a64 == "2a656a080ba3a76b"
    and .[9].boundary_fnv1a64 == .[0].boundary_fnv1a64
    and .[0].boundary_coefficients == .[2].boundary_coefficients
    and .[0].boundary_coefficients == .[3].boundary_coefficients
    and .[4].boundary_coefficients == ([range(0;25)] | map(0))
    and .[5].boundary_coefficients == .[4].boundary_coefficients
    and .[6].boundary_coefficients[0:24]
        == ([range(0;24)] | map(0))
    and .[6].boundary_coefficients[24] == 1
    and .[7].boundary_coefficients == .[6].boundary_coefficients
    and .[8].boundary_coefficients
        == [1,1,0,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0]
    and .[10].wrong_parent_inverse == true
    and .[10].missing_parent_inverse == true
    and .[10].reordered_inverse == true
    and .[10].wrong_parent_restoration_abs >= 1e-3
    and .[10].missing_parent_restoration_abs >= 1e-3
    and .[10].reordered_restoration_abs >= 1e-3
    and .[10].different_reuse_boundary == true
    and .[10].row_permutation_invariant == true
    and .[10].duplicate_equation_invariant == true
    and .[10].rank0_universal == true
    and .[10].rank1_universal == true
    and .[10].input_empty_propagated == true
    and .[10].multi_contradiction_empty == true
    and .[10].canonical_full_rank_boundary == true
    and .[10].fixture_hashes_match == true
    and .[10].fixed_operation_schedule == true
    and .[10].same_carrier_transactions == 73
    and .[10].repeated_reuse_max_restoration_abs <= 2e-12
' "$work_dir/result.jsonl" >/dev/null

jq -s -e '
    length == 9
    and all(.[]; .tuple_slots == 0)
    and all(.[]; .assignment_slots == 0)
    and all(.[]; .witness_slots == 0)
    and .[0].boundary_fnv1a64 == "a96f5625e4054ee6"
    and .[1].boundary_fnv1a64 == "0183d6882c23ee46"
    and .[2].boundary_fnv1a64 == .[0].boundary_fnv1a64
    and .[3].boundary_fnv1a64 == .[0].boundary_fnv1a64
    and .[4].boundary_fnv1a64 == "d4657f55662f817f"
    and .[5].boundary_fnv1a64 == .[4].boundary_fnv1a64
    and .[6].boundary_fnv1a64 == "d4657e55662f7fcc"
    and .[7].boundary_fnv1a64 == .[6].boundary_fnv1a64
    and .[8].boundary_fnv1a64 == "2a656a080ba3a76b"
' "$work_dir/reference.jsonl" >/dev/null

test "$(grep -c 'decode_root(' "$phase_source")" -eq 1
test "$(grep -c 'ga_latch_boundary(' "$phase_source")" -eq 2
if rg -n \
    -e 'truth_table' \
    -e 'assignment_table' \
    -e 'assignment_expansion' \
    -e 'witness_list' \
    -e 'candidate_set' \
    -e 'dense_relation\[' \
    "$phase_source" "$reference_source"
then
    echo "forbidden expanded relation structure in affine sources" >&2
    exit 1
fi
if nm -a "$phase" | rg \
    'oblivious_affine_reference|reference_boundaries|ref_compose'
then
    echo "independent reference linked into phase executable" >&2
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
stack_value() {
    local pattern=$1
    awk -F '\t' -v pattern="$pattern" \
        '$1 ~ pattern {print $2; exit}' "$stack_file"
}
main_stack=$(stack_value ':main$')
execute_stack=$(stack_value ':ga_execute$')
apply_stack=$(stack_value ':ga_apply_compose$')
swap_stack=$(stack_value ':ga_conditional_swap_rows')
symbol_stack=$(stack_value ':symbol_product$')
lock_stack=$(stack_value ':lock_f3_phase$')
test "$main_stack" -eq 4144
test "$execute_stack" -eq 736
test "$apply_stack" -eq 224
test "$swap_stack" -eq 768
test "$symbol_stack" -eq 96
test "$lock_stack" -eq 32
nested_stack_chain_bytes=$((
    main_stack
    + execute_stack
    + apply_stack
    + swap_stack
    + symbol_stack
    + lock_stack
))
test "$nested_stack_chain_bytes" -eq 6000
material_heap_subtotal_bytes=20352
conservative_accounted_bytes=$((
    material_heap_subtotal_bytes
    + nested_stack_chain_bytes
))

sha256sum \
    "$phase_source" \
    "$reference_source" \
    "$base_source" \
    "$qualifier_source" \
    >"$work_dir/source_sha256.txt"
sha256sum \
    "$work_dir/result.jsonl" \
    "$work_dir/reference.jsonl" \
    "$work_dir/phase_boundaries.json" \
    >"$work_dir/result_sha256.txt"

phase_source_sha256=$(sha256sum "$phase_source" | awk '{print $1}')
reference_source_sha256=$(
    sha256sum "$reference_source" | awk '{print $1}'
)
base_source_sha256=$(sha256sum "$base_source" | awk '{print $1}')
qualifier_source_sha256=$(
    sha256sum "$qualifier_source" | awk '{print $1}'
)
result_sha256=$(sha256sum "$work_dir/result.jsonl" | awk '{print $1}')
reference_sha256=$(
    sha256sum "$work_dir/reference.jsonl" | awk '{print $1}'
)

jq -n \
    --argjson compiler_builds "$compiler_builds" \
    --argjson nested_stack_chain_bytes "$nested_stack_chain_bytes" \
    --argjson material_heap_subtotal_bytes "$material_heap_subtotal_bytes" \
    --argjson conservative_accounted_bytes "$conservative_accounted_bytes" \
    --arg phase_source_sha256 "$phase_source_sha256" \
    --arg reference_source_sha256 "$reference_source_sha256" \
    --arg base_source_sha256 "$base_source_sha256" \
    --arg qualifier_source_sha256 "$qualifier_source_sha256" \
    --arg result_sha256 "$result_sha256" \
    --arg reference_sha256 "$reference_sha256" \
    --slurpfile results "$work_dir/result.jsonl" \
    '{
        result: "PASS",
        claim: $results[0].claim,
        claim_ceiling:
            "FIXED_WIDTH2_SOFTWARE_GF2_AFFINE_RELATION_SYSTEM_"
                + "REFERENCE_ONLY",
        compiler_builds: $compiler_builds,
        sanitizers: "PASS",
        semantic_variants: 9,
        phase_reference_exact_matches: 9,
        primary_boundary_fnv1a64: $results[0].boundary_fnv1a64,
        reuse_boundary_fnv1a64: $results[1].boundary_fnv1a64,
        full_rank_boundary_fnv1a64: $results[8].boundary_fnv1a64,
        carrier_cells: $results[0].carrier_cells,
        workspace_cells: $results[0].total_workspace_cells,
        material_heap_subtotal_bytes_current_abi:
            $material_heap_subtotal_bytes,
        compiler_measured_nested_stack_call_chain_bytes:
            $nested_stack_chain_bytes,
        conservative_accounted_execution_bytes_current_abi:
            $conservative_accounted_bytes,
        same_carrier_transactions: $results[10].same_carrier_transactions,
        maximum_reuse_restoration_error:
            $results[10].repeated_reuse_max_restoration_abs,
        fixed_operation_schedule: $results[10].fixed_operation_schedule,
        semantic_controls: {
            row_permutation: $results[10].row_permutation_invariant,
            duplicate_equation: $results[10].duplicate_equation_invariant,
            rank0_universal: $results[10].rank0_universal,
            rank1_universal: $results[10].rank1_universal,
            input_empty: $results[10].input_empty_propagated,
            multi_contradiction_empty:
                $results[10].multi_contradiction_empty,
            full_rank: $results[10].canonical_full_rank_boundary
        },
        inverse_controls: {
            wrong: $results[10].wrong_parent_inverse,
            missing: $results[10].missing_parent_inverse,
            reordered: $results[10].reordered_inverse
        },
        no_smuggle: {
            decoded_intermediate_coefficients:
                $results[0].decoded_intermediate_coefficients,
            serialized_intermediate_coefficients:
                $results[0].serialized_intermediate_coefficients,
            host_selected_pivots:
                $results[0].host_selected_pivots,
            fixed_host_loop_address_schedule:
                $results[0].fixed_host_loop_address_schedule,
            tuple_slots: $results[0].tuple_slots,
            assignment_slots: $results[0].assignment_slots,
            witness_slots: $results[0].witness_slots,
            sole_decode_is_final_boundary: true,
            phase_reference_binary_separation: true,
            strace_no_extra_write_channel: true
        },
        sha256: {
            phase_source: $phase_source_sha256,
            reference_source: $reference_source_sha256,
            base_source: $base_source_sha256,
            qualifier_source: $qualifier_source_sha256,
            result: $result_sha256,
            reference: $reference_sha256
        },
        terminal: false
    }'
