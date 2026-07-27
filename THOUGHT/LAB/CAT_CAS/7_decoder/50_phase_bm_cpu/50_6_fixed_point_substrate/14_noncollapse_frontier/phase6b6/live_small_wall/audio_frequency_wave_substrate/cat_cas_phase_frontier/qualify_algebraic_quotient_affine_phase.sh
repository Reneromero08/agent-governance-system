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

phase_source="$source_dir/algebraic_quotient_affine_phase.c"
reference_source="$source_dir/quotient_affine_reference.c"
base_source="$source_dir/algebraic_series_parallel_phase.c"
qualifier_source="$source_dir/qualify_algebraic_quotient_affine_phase.sh"

phase="$work_dir/algebraic_quotient_affine_phase"
reference="$work_dir/quotient_affine_reference"
analyzer="$work_dir/algebraic_quotient_affine_phase_analyzer"
sanitized="$work_dir/algebraic_quotient_affine_phase_sanitized"
stack_object="$work_dir/algebraic_quotient_affine_phase_stack.o"

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
    -fstack-usage -c \
    "$phase_source" -o "$stack_object"

compiler_builds=5
if command -v clang >/dev/null 2>&1; then
    clang \
        -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
        "$phase_source" -lm \
        -o "$work_dir/algebraic_quotient_affine_phase_clang"
    compiler_builds=$((compiler_builds + 1))
fi

"$phase" >"$work_dir/result.jsonl"
"$phase" >"$work_dir/replay.jsonl"
cmp "$work_dir/result.jsonl" "$work_dir/replay.jsonl"

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:abort_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$sanitized" >"$work_dir/sanitized.jsonl"
cmp "$work_dir/result.jsonl" "$work_dir/sanitized.jsonl"

"$reference" >"$work_dir/reference.jsonl"
jq -s '
    .[0:2]
    | map({boundary_coefficients, boundary_fnv1a64})
' "$work_dir/result.jsonl" >"$work_dir/phase_boundary.json"
jq -s '
    map({boundary_coefficients, boundary_fnv1a64})
' "$work_dir/reference.jsonl" >"$work_dir/reference_boundary.json"
cmp "$work_dir/phase_boundary.json" "$work_dir/reference_boundary.json"

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
    "only the compact full final boundary may be projected" \
    projection \
    --project-intermediate
reject_fixed \
    "null carrier or quotient-affine program" \
    null_carrier \
    --null-carrier
reject_fixed \
    "usage: algebraic_quotient_affine_phase" \
    unknown \
    --unknown

execution_keys='[
  "assignment_slots",
  "boundary_coefficients",
  "boundary_decodes",
  "boundary_fnv1a64",
  "carrier_and_snapshot_complex_values",
  "carrier_and_snapshot_heap_bytes",
  "carrier_cells",
  "carrier_integrity_max_abs",
  "carrier_reads",
  "claim",
  "comparison_snapshot_complex_values",
  "compiled_descriptor_bytes_current_abi",
  "coexisting_program_storage_bytes_current_abi",
  "decoded_intermediate_coefficients",
  "dense_multiaffine_coefficient_exponent",
  "dense_pair_table_exponent",
  "displacement_l2",
  "execution_result_bytes_current_abi",
  "factor_coefficients",
  "fibre_bits",
  "full_boundary_materialized",
  "genuinely_many_to_many",
  "inverse_factor_recomputations",
  "live_carrier_bytes",
  "live_carrier_complex_values",
  "maximum_root_error",
  "mode",
  "native_and_products",
  "native_compose_calls",
  "native_xor_accumulations",
  "phase_cell_updates",
  "port_bits",
  "program_storage_bytes_current_abi",
  "public_program_coefficients",
  "quotient_bits",
  "relation_family",
  "restoration_cell_checks",
  "restoration_max_abs",
  "restoration_path",
  "serialized_intermediate_coefficients",
  "snapshot_loaded",
  "source_fnv1a64",
  "sources_per_target",
  "sparse_boundary_query",
  "symbol_product_calls",
  "targets_per_source",
  "tuple_slots",
  "witness_slots"
]'
control_keys='[
  "actual_restored_cross_program_reuse",
  "altered_algebra_histories_restore",
  "f3_sum_changes_full_boundary",
  "missing_parent_inverse",
  "mode",
  "reordered_noncommuting_inverse",
  "repeated_reuse_max_restoration_abs",
  "reverse_order_changes_full_boundary",
  "same_carrier_transactions",
  "snapshot_restores",
  "wrong_parent_inverse"
]'

for output in result sanitized; do
    jq -s -e \
        --argjson execution "$execution_keys" \
        --argjson controls "$control_keys" '
        length == 4
        and all(.[0:3][]; (keys | sort) == ($execution | sort))
        and (.[3] | keys | sort) == ($controls | sort)
    ' "$work_dir/$output.jsonl" >/dev/null
done

jq -s -e '
    .[0].mode == "projected-affine-primary"
    and .[1].mode == "projected-affine-reuse"
    and .[2].mode == "projected-affine-snapshot"
    and .[0].claim
        == "BOUNDED_PROJECTED_AFFINE_PHASE_RELATION_COMPOSITION_"
            + "WITH_COMPACT_FULL_BOUNDARY"
    and all(.[0:3][]; .relation_family == "PROJECTED_AFFINE_F2")
    and all(.[0:3][]; .quotient_bits == 8)
    and all(.[0:3][]; .fibre_bits == 4)
    and all(.[0:3][]; .port_bits == 12)
    and all(.[0:3][]; .genuinely_many_to_many == true)
    and all(.[0:3][]; .targets_per_source == 16)
    and all(.[0:3][]; .sources_per_target == 16)
    and all(.[0:3][]; .factor_coefficients == 72)
    and all(.[0:3][]; .dense_pair_table_exponent == 24)
    and all(.[0:3][]; .dense_multiaffine_coefficient_exponent == 24)
    and all(.[0:3][]; .full_boundary_materialized == true)
    and all(.[0:3][]; .sparse_boundary_query == false)
    and all(.[0:3][]; .tuple_slots == 0)
    and all(.[0:3][]; .assignment_slots == 0)
    and all(.[0:3][]; .witness_slots == 0)
    and all(.[0:3][]; .decoded_intermediate_coefficients == 0)
    and all(.[0:3][]; .serialized_intermediate_coefficients == 0)
    and .[0].boundary_fnv1a64 == "33f962d84bbdd8db"
    and .[1].boundary_fnv1a64 == "1b6d1f9d91bbe788"
    and .[2].boundary_fnv1a64 == .[0].boundary_fnv1a64
    and .[0].boundary_coefficients == .[2].boundary_coefficients
    and .[0].boundary_coefficients != .[1].boundary_coefficients
    and all(.[0:3][]; (.boundary_coefficients | length) == 72)
    and all(.[0:3][]; all(.boundary_coefficients[]; . == 0 or . == 1))
    and all(.[0:3][]; .carrier_cells == 432)
    and all(.[0:3][]; .live_carrier_complex_values == 864)
    and all(.[0:3][]; .live_carrier_bytes == 13824)
    and all(.[0:3][]; .comparison_snapshot_complex_values == 864)
    and all(.[0:3][]; .carrier_and_snapshot_complex_values == 1728)
    and all(.[0:3][]; .carrier_and_snapshot_heap_bytes == 27648)
    and all(.[0:3][]; .public_program_coefficients == 216)
    and all(.[0:3][]; .program_storage_bytes_current_abi == 224)
    and all(.[0:3][]; .coexisting_program_storage_bytes_current_abi == 448)
    and all(.[0:3][]; .compiled_descriptor_bytes_current_abi == 48)
    and all(.[0:3][]; .execution_result_bytes_current_abi == 408)
    and all(.[0:2][]; .native_compose_calls == 4)
    and all(.[0:2][]; .native_and_products == 2304)
    and all(.[0:2][]; .native_xor_accumulations == 2304)
    and all(.[0:2][]; .symbol_product_calls == 4608)
    and all(.[0:2][]; .carrier_reads == 4784)
    and all(.[0:2][]; .phase_cell_updates == 864)
    and all(.[0:2][]; .inverse_factor_recomputations == 2)
    and all(.[0:3][]; .boundary_decodes == 72)
    and all(.[0:3][]; .restoration_cell_checks == 432)
    and all(.[0:2][]; .restoration_path == "actual_inverse")
    and all(.[0:2][]; .snapshot_loaded == false)
    and .[2].restoration_path == "snapshot_reload"
    and .[2].snapshot_loaded == true
    and all(.[0:3][]; .maximum_root_error <= 4e-10)
    and all(.[0:3][]; .restoration_max_abs <= 2e-12)
    and all(.[0:3][]; .carrier_integrity_max_abs <= 2e-12)
    and .[3].wrong_parent_inverse == true
    and .[3].missing_parent_inverse == true
    and .[3].reordered_noncommuting_inverse == true
    and .[3].f3_sum_changes_full_boundary == true
    and .[3].reverse_order_changes_full_boundary == true
    and .[3].altered_algebra_histories_restore == true
    and .[3].snapshot_restores == true
    and .[3].actual_restored_cross_program_reuse == true
    and .[3].same_carrier_transactions == 130
    and .[3].repeated_reuse_max_restoration_abs <= 2e-12
' "$work_dir/result.jsonl" >/dev/null

jq -s -e '
    length == 2
    and all(.[]; .input_ranks == [8,8,8])
    and all(.[]; .boundary_rank == 8)
    and all(.[]; .tuple_slots == 0)
    and all(.[]; .assignment_slots == 0)
    and all(.[]; .witness_slots == 0)
    and .[0].boundary_fnv1a64 == "33f962d84bbdd8db"
    and .[1].boundary_fnv1a64 == "1b6d1f9d91bbe788"
' "$work_dir/reference.jsonl" >/dev/null

scale_summary="$work_dir/scaling.jsonl"
: >"$scale_summary"
for q in 2 4 8 16 32 64; do
    fibre=$((q / 2))
    if [ "$fibre" -gt 20 ]; then
        fibre=20
    fi
    scale_phase="$work_dir/phase_q$q"
    scale_reference="$work_dir/reference_q$q"
    gcc \
        -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
        -DQA_Q="${q}U" \
        -DQA_FIBRE="${fibre}U" \
        -DQA_REPEAT_CYCLES=1U \
        "$phase_source" -lm -o "$scale_phase"
    gcc \
        -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
        -DREF_Q="${q}U" \
        "$reference_source" -o "$scale_reference"
    compiler_builds=$((compiler_builds + 2))
    "$scale_phase" >"$work_dir/scale_q$q.phase.jsonl"
    "$scale_reference" >"$work_dir/scale_q$q.reference.jsonl"
    jq -s '
        .[0:2] | map({boundary_coefficients, boundary_fnv1a64})
    ' "$work_dir/scale_q$q.phase.jsonl" \
        >"$work_dir/scale_q$q.phase-boundary.json"
    jq -s '
        map({boundary_coefficients, boundary_fnv1a64})
    ' "$work_dir/scale_q$q.reference.jsonl" \
        >"$work_dir/scale_q$q.reference-boundary.json"
    cmp \
        "$work_dir/scale_q$q.phase-boundary.json" \
        "$work_dir/scale_q$q.reference-boundary.json"
    jq -c -s --argjson q "$q" '
        .[0]
        | {
            quotient_bits,
            fibre_bits,
            port_bits,
            factor_coefficients,
            carrier_cells,
            live_carrier_bytes,
            dense_pair_table_exponent,
            native_and_products,
            native_xor_accumulations,
            symbol_product_calls,
            restoration_max_abs,
            maximum_root_error,
            reference_match: true
        }
        | select(.quotient_bits == $q)
    ' "$work_dir/scale_q$q.phase.jsonl" >>"$scale_summary"
done

jq -s -e '
    map(.quotient_bits) == [2,4,8,16,32,64]
    and all(.[];
        .factor_coefficients
            == .quotient_bits * (.quotient_bits + 1)
        and .carrier_cells == 6 * .factor_coefficients
        and .native_and_products
            == 4 * .quotient_bits * .quotient_bits
                * (.quotient_bits + 1)
        and .native_xor_accumulations == .native_and_products
        and .symbol_product_calls == 2 * .native_and_products
        and .restoration_max_abs <= 2e-12
        and .maximum_root_error <= 4e-10
        and .reference_match == true
    )
    and .[-1].port_bits == 84
    and .[-1].factor_coefficients == 4160
    and .[-1].carrier_cells == 24960
    and .[-1].dense_pair_table_exponent == 168
' "$scale_summary" >/dev/null

"$work_dir/phase_q2" --exhaustive-group \
    >"$work_dir/q2_group_calibration.json"
jq -e '
    .mode == "q2-affine-group-calibration"
    and .relation_count == 24
    and .ordered_pair_closures == 576
    and .associativity_triples == 13824
    and .retained_relation_lookup_entries == 0
    and .tuple_slots == 0
    and .assignment_slots == 0
    and .witness_slots == 0
    and .decoded_intermediate_coefficients == 0
    and .maximum_restoration_error <= 2e-12
' "$work_dir/q2_group_calibration.json" >/dev/null

test "$(grep -c 'decode_root(' "$phase_source")" -eq 1
test "$(grep -c 'qa_latch_boundary(' "$phase_source")" -eq 2
if rg -n \
    -e 'truth_table' \
    -e 'assignment_table' \
    -e 'witness_list' \
    -e 'candidate_set' \
    -e 'dense_relation\[' \
    "$phase_source"
then
    echo "forbidden expanded relation structure in phase source" >&2
    exit 1
fi
if nm -a "$phase" | rg 'quotient_affine_reference|ref_compose'; then
    echo "independent reference linked into phase executable" >&2
    exit 1
fi

stack_file="${stack_object%.o}.su"
test -s "$stack_file"
main_stack=$(awk -F '\t' '$1 ~ /:main$/ {print $2; exit}' "$stack_file")
execute_stack=$(
    awk -F '\t' '$1 ~ /:qa_execute_ordered/ {print $2; exit}' "$stack_file"
)
apply_stack=$(
    awk -F '\t' '$1 ~ /:qa_apply_compose/ {print $2; exit}' "$stack_file"
)
symbol_stack=$(
    awk -F '\t' '$1 ~ /:symbol_product$/ {print $2; exit}' "$stack_file"
)
lock_stack=$(
    awk -F '\t' '$1 ~ /:lock_f3_phase$/ {print $2; exit}' "$stack_file"
)
test "$main_stack" -eq 4048
test "$execute_stack" -eq 1376
test "$apply_stack" -eq 224
test "$symbol_stack" -eq 96
test "$lock_stack" -eq 32
nested_stack_chain_bytes=$((
    main_stack
    + execute_stack
    + apply_stack
    + symbol_stack
    + lock_stack
))
test "$nested_stack_chain_bytes" -eq 5776
material_heap_subtotal_bytes=27648
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
    "$scale_summary" \
    "$work_dir/q2_group_calibration.json" \
    >"$work_dir/result_sha256.txt"

jq -n \
    --argjson compiler_builds "$compiler_builds" \
    --argjson nested_stack_chain_bytes "$nested_stack_chain_bytes" \
    --argjson material_heap_subtotal_bytes "$material_heap_subtotal_bytes" \
    --argjson conservative_accounted_bytes "$conservative_accounted_bytes" \
    --slurpfile results "$work_dir/result.jsonl" \
    --slurpfile scaling "$scale_summary" \
    --slurpfile group "$work_dir/q2_group_calibration.json" \
    '{
        result: "PASS",
        claim: $results[0].claim,
        claim_ceiling:
            "BOUNDED_PROJECTED_AFFINE_F2_RELATION_SUBCATEGORY_REFERENCE_ONLY",
        compiler_builds: $compiler_builds,
        sanitizers: "PASS",
        primary_boundary_fnv1a64: $results[0].boundary_fnv1a64,
        reuse_boundary_fnv1a64: $results[1].boundary_fnv1a64,
        quotient_bits: $results[0].quotient_bits,
        fibre_bits: $results[0].fibre_bits,
        factor_coefficients: $results[0].factor_coefficients,
        dense_pair_table_exponent: $results[0].dense_pair_table_exponent,
        material_heap_subtotal_bytes_current_abi:
            $material_heap_subtotal_bytes,
        compiler_measured_nested_stack_call_chain_bytes:
            $nested_stack_chain_bytes,
        conservative_accounted_execution_bytes_current_abi:
            $conservative_accounted_bytes,
        same_carrier_transactions: $results[3].same_carrier_transactions,
        maximum_reuse_restoration_error:
            $results[3].repeated_reuse_max_restoration_abs,
        inverse_controls: {
            wrong: $results[3].wrong_parent_inverse,
            missing: $results[3].missing_parent_inverse,
            reordered_noncommuting:
                $results[3].reordered_noncommuting_inverse
        },
        algebra_controls: {
            f3_sum_changes_full_boundary:
                $results[3].f3_sum_changes_full_boundary,
            reverse_order_changes_full_boundary:
                $results[3].reverse_order_changes_full_boundary,
            altered_histories_restore:
                $results[3].altered_algebra_histories_restore
        },
        no_smuggle: {
            full_boundary_materialized:
                $results[0].full_boundary_materialized,
            sparse_boundary_query: $results[0].sparse_boundary_query,
            decoded_intermediate_coefficients:
                $results[0].decoded_intermediate_coefficients,
            serialized_intermediate_coefficients:
                $results[0].serialized_intermediate_coefficients,
            tuple_slots: $results[0].tuple_slots,
            assignment_slots: $results[0].assignment_slots,
            witness_slots: $results[0].witness_slots,
            phase_reference_separation: true
        },
        maximum_scale: $scaling[-1],
        exhaustive_minimum_nonabelian_group: {
            relation_count: $group[0].relation_count,
            ordered_pair_closures: $group[0].ordered_pair_closures,
            associativity_triples: $group[0].associativity_triples,
            maximum_restoration_error:
                $group[0].maximum_restoration_error
        },
        terminal: false
    }'
