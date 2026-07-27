#!/usr/bin/env bash
set -euo pipefail

source_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
phase_source="$source_dir/algebraic_boolean_tt_phase.c"
reference_source="$source_dir/algebraic_boolean_tt_reference.c"
included_source="$source_dir/algebraic_series_parallel_phase.c"
qualifier_source="$source_dir/qualify_algebraic_boolean_tt_phase.sh"
evidence_dir=${1:-}

if [[ -z "$evidence_dir" ]]; then
    evidence_dir=$(mktemp -d /tmp/boolean-tt-phase.XXXXXX)
else
    if [[ -e "$evidence_dir" ]]; then
        printf 'evidence directory already exists: %s\n' "$evidence_dir" >&2
        exit 2
    fi
    mkdir -p "$evidence_dir"
fi

phase_bin="$evidence_dir/algebraic_boolean_tt_phase"
reference_bin="$evidence_dir/algebraic_boolean_tt_reference"
phase_asan="$evidence_dir/algebraic_boolean_tt_phase_asan"
reference_asan="$evidence_dir/algebraic_boolean_tt_reference_asan"
native_jsonl="$evidence_dir/native.jsonl"
reference_jsonl="$evidence_dir/reference.jsonl"
native_replay="$evidence_dir/native_replay.jsonl"

strict_flags=(
    -std=c11
    -O2
    -Wall
    -Wextra
    -Werror
    -Wconversion
    -Wshadow
    -Wpedantic
)
sanitizer_flags=(
    -std=c11
    -O1
    -g
    -fno-omit-frame-pointer
    -fsanitize=address,undefined
    -Wall
    -Wextra
    -Werror
    -Wconversion
    -Wshadow
    -Wpedantic
)
widths=(4 5 8 12 16)

gcc "${strict_flags[@]}" -o "$phase_bin" "$phase_source" -lm
gcc "${strict_flags[@]}" -o "$reference_bin" "$reference_source"
gcc "${sanitizer_flags[@]}" -o "$phase_asan" "$phase_source" -lm
gcc "${sanitizer_flags[@]}" -o "$reference_asan" "$reference_source"

gcc -std=c11 -O0 -fanalyzer -Wall -Wextra -Werror \
    -c "$phase_source" -o "$evidence_dir/phase_analyzer.o" \
    >"$evidence_dir/phase_analyzer.stdout" \
    2>"$evidence_dir/phase_analyzer.stderr"
gcc -std=c11 -O0 -fanalyzer -Wall -Wextra -Werror \
    -c "$reference_source" -o "$evidence_dir/reference_analyzer.o" \
    >"$evidence_dir/reference_analyzer.stdout" \
    2>"$evidence_dir/reference_analyzer.stderr"

: >"$native_jsonl"
: >"$reference_jsonl"
: >"$native_replay"
for width in "${widths[@]}"; do
    "$phase_bin" "$width" >>"$native_jsonl"
    "$reference_bin" "$width" >>"$reference_jsonl"
    "$phase_bin" "$width" >>"$native_replay"
done
cmp "$native_jsonl" "$native_replay"

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$phase_asan" 4 >"$evidence_dir/asan_phase_width4.json"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$phase_asan" 16 >"$evidence_dir/asan_phase_width16.json"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$reference_asan" 4 >"$evidence_dir/asan_reference_width4.json"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$reference_asan" 16 >"$evidence_dir/asan_reference_width16.json"

expect_rejection() {
    local name=$1
    local expected=$2
    shift 2
    if "$@" >"$evidence_dir/$name.stdout" 2>"$evidence_dir/$name.stderr"; then
        printf 'negative control unexpectedly succeeded: %s\n' "$name" >&2
        exit 1
    fi
    test ! -s "$evidence_dir/$name.stdout"
    test "$(cat "$evidence_dir/$name.stderr")" = "$expected"
}

expect_rejection \
    project_hidden_h \
    "only the final rank-eight Boolean-TT boundary may be projected" \
    "$phase_bin" --project 4
expect_rejection \
    eager_dense \
    "dense 4^width relation materialization is forbidden" \
    "$phase_bin" --dense 16
expect_rejection \
    rank_cap \
    "public rank cap 7 rejects required rank-eight root" \
    "$phase_bin" --rank-cap-7 4
expect_rejection \
    null_carrier \
    "null Boolean-TT carrier" \
    "$phase_bin" --null 4
expect_rejection \
    malformed_width_low \
    "width must be a canonical decimal from 4 through 16" \
    "$phase_bin" 3
expect_rejection \
    malformed_width_high \
    "width must be a canonical decimal from 4 through 16" \
    "$phase_bin" 17
expect_rejection \
    malformed_width_noncanonical \
    "width must be a canonical decimal from 4 through 16" \
    "$phase_bin" 04

jq -s '.' "$native_jsonl" >"$evidence_dir/native.json"
jq -s '.' "$reference_jsonl" >"$evidence_dir/reference.json"

jq -e '
    length == 5
    and map(.width) == [4,5,8,12,16]
    and all(.[];
        .mode == "width-parametric-boolean-tt"
        and .claim
            == "BOUNDED_WIDTH_PARAMETRIC_BOOLEAN_TT_MANY_TO_MANY_"
             + "RELATION_COMPOSITION_WITH_PRODUCT_RANK_NATIVE_PHASE_"
             + "CLOSURE_AND_RESIDENT_INTERMEDIATE"
        and .leaf_rank == 2
        and .resident_h_rank == 4
        and .final_z_rank == 8
        and .leaf_cells_each == 16 * .width - 16
        and .resident_h_cells == 64 * .width - 96
        and .final_z_cells == 256 * .width - 448
        and .boundary_cells == .final_z_cells
        and .carrier_cells == 624 * .width - 1040
        and .live_carrier_bytes == 32 * .carrier_cells
        and .verification_snapshot_bytes == 32 * .carrier_cells
        and .projected_boundary_bytes == .final_z_cells
        and .phase_cell_updates_per_transaction
            == 1248 * .width - 2080
        and .logical_phase_ands_per_transaction
            == 4 * (.resident_h_cells + .final_z_cells)
        and .logical_phase_ors_per_transaction
            == 2 * (.resident_h_cells + .final_z_cells)
        and .phase_products_inside_or_per_transaction
            == .logical_phase_ors_per_transaction
        and .carrier_reads_per_transaction
            == 8 * .resident_h_cells + 11 * .final_z_cells
        and .final_decodes_per_transaction == .final_z_cells
        and .decoded_intermediate_cells == 0
        and .serialized_intermediate_cells == 0
        and .copied_intermediate_cells == 0
        and .shared_assignment_loops == 0
        and .local_shared_values_per_output_core == 2
        and .dense_relation_cells_materialized == 0
        and .truth_table_slots == 0
        and .full_output_tt_materialized
        and .actual_resident_h_consumed_by_z
        and .actual_inverse_restoration
        and .actual_restored_carrier_reuse
        and .same_carrier_transactions == 34
        and .restoration_generation == 34
        and .repeated_reuse_max_restoration_abs
            <= .restoration_tolerance
        and .primary_many_to_many
        and .leaf_outputs_per_input == 2
        and .root_outputs_per_input_lower_bound == 2
        and .root_inputs_for_zero_output_lower_bound > 1
        and .degree4_window_count == .width - 3
        and .analytic_fourth_boolean_derivative == 1
        and .analytic_non_affine_certificate
        and (.fixed_rank_unbounded_depth_closure | not)
        and .finite_product_rank_closure
        and (.rank_minimization_or_recompression | not)
        and .wrong_inverse_restoration_error >= 1e-3
        and .missing_inverse_restoration_error >= 1e-3
        and .reordered_inverse_restoration_error >= 1e-3
        and .snapshot_restoration_error <= .restoration_tolerance
        and .snapshot_loaded
        and .snapshot_restoration_generation == 0
        and (.terminal | not)
    )
    and .[0].root_inputs_for_zero_output_lower_bound == 8
    and .[1].root_inputs_for_zero_output_lower_bound == 13
    and .[2].root_inputs_for_zero_output_lower_bound == 55
    and .[3].root_inputs_for_zero_output_lower_bound == 377
    and .[4].root_inputs_for_zero_output_lower_bound == 2584
    and (.[0].rank8_smaller_than_dense | not)
    and all(.[1:][]; .rank8_smaller_than_dense)
' "$evidence_dir/native.json" >"$evidence_dir/native_assertion.txt"

jq -e '
    length == 5
    and map(.width) == [4,5,8,12,16]
    and all(.[];
        .mode == "compact-boolean-tt-reference"
        and .width_wide_assignments_enumerated == 0
        and .dense_relation_cells_materialized == 0
        and .truth_table_slots == 0
    )
' "$evidence_dir/reference.json" >"$evidence_dir/reference_assertion.txt"

jq -n -e \
    --slurpfile native "$evidence_dir/native.json" \
    --slurpfile reference "$evidence_dir/reference.json" '
    ($native[0] | sort_by(.width)) as $n
    | ($reference[0] | sort_by(.width)) as $r
    | ($n | length) == ($r | length)
    and all(range(0; $n | length);
        $n[.].width == $r[.].width
        and $n[.].primary_program_fnv1a64
            == $r[.].primary_program_fnv1a64
        and $n[.].reuse_program_fnv1a64
            == $r[.].reuse_program_fnv1a64
        and $n[.].primary_boundary_fnv1a64
            == $r[.].primary_boundary_fnv1a64
        and $n[.].reuse_boundary_fnv1a64
            == $r[.].reuse_boundary_fnv1a64
        and $n[.].primary_boundary_ones
            == $r[.].primary_boundary_ones
        and $n[.].reuse_boundary_ones
            == $r[.].reuse_boundary_ones
    )
' >"$evidence_dir/native_reference_parity.txt"

nm -a "$phase_bin" >"$evidence_dir/phase_symbols.txt"
nm -a "$reference_bin" >"$evidence_dir/reference_symbols.txt"
test -z "$(rg 'ref_compose|ref_fill_leaf' "$evidence_dir/phase_symbols.txt" || true)"
test -z "$(rg 'symbol_product|lock_f3_phase|make_carrier' "$evidence_dir/reference_symbols.txt" || true)"

rg -n \
    'btt_compose|actual_resident_h_consumed_by_z|shared_assignment_loops' \
    "$phase_source" >"$evidence_dir/phase_static_presence.txt"
if rg -n \
    '1(U|ULL)?[[:space:]]*<<[[:space:]]*\([[:space:]]*2.*width|accepted_pairs\[|witness(es)?\[|truth_table\[' \
    "$phase_source" "$reference_source" \
    >"$evidence_dir/forbidden_static.txt"; then
    printf 'forbidden dense or witness structure found\n' >&2
    exit 1
fi
: >"$evidence_dir/forbidden_static.txt"

strace -qq -f \
    -e trace=write,writev,openat,creat,rename,unlink,sendto,sendmsg \
    -o "$evidence_dir/output_trace.txt" \
    "$phase_bin" 4 >"$evidence_dir/traced_width4.json" \
    2>"$evidence_dir/traced_width4.stderr"
test ! -s "$evidence_dir/traced_width4.stderr"
test "$(rg -c 'write\(1,' "$evidence_dir/output_trace.txt")" -eq 1
test -z "$(rg 'write\(2,|writev\(|sendto\(|sendmsg\(' "$evidence_dir/output_trace.txt" || true)"

sha256sum \
    "$phase_source" \
    "$reference_source" \
    "$included_source" \
    "$qualifier_source" \
    "$phase_bin" \
    "$reference_bin" \
    "$native_jsonl" \
    "$reference_jsonl" \
    "$evidence_dir/native.json" \
    "$evidence_dir/reference.json" \
    "$evidence_dir/output_trace.txt" \
    >"$evidence_dir/CORE_SHA256SUMS"

jq -n \
    --slurpfile native "$evidence_dir/native.json" \
    --slurpfile reference "$evidence_dir/reference.json" \
    --arg evidence_dir "$evidence_dir" \
    --arg phase_sha "$(sha256sum "$phase_source" | cut -d' ' -f1)" \
    --arg reference_sha "$(sha256sum "$reference_source" | cut -d' ' -f1)" \
    --arg included_sha "$(sha256sum "$included_source" | cut -d' ' -f1)" \
    --arg qualifier_sha "$(sha256sum "$qualifier_source" | cut -d' ' -f1)" '
    {
        result: "PASS",
        claim:
            "BOUNDED_WIDTH_PARAMETRIC_BOOLEAN_TT_MANY_TO_MANY_"
            + "RELATION_COMPOSITION_WITH_PRODUCT_RANK_NATIVE_PHASE_"
            + "CLOSURE_AND_RESIDENT_INTERMEDIATE",
        claim_ceiling:
            "BOUNDED_LINUX_SOFTWARE_WIDTHS4_5_8_12_16_BOOLEAN_"
            + "SEMIRING_TT_RANK2_TO_RANK4_TO_RANK8_REFERENCE_ONLY",
        evidence_directory: $evidence_dir,
        widths: ($native[0] | map(.width)),
        exact_native_reference_boundary_parity: true,
        width_wide_shared_assignments_enumerated: 0,
        dense_relation_cells_materialized: 0,
        truth_table_slots: 0,
        leaf_rank: 2,
        resident_h_rank: 4,
        final_z_rank: 8,
        minimum_dense_crossover_width: 5,
        width4_fixed_qanf_degree4_embedding: true,
        analytic_fourth_boolean_derivative: 1,
        primary_many_to_many: true,
        actual_resident_h_consumed_by_z: true,
        actual_inverse_restoration: true,
        actual_restored_carrier_reuse: true,
        same_carrier_transactions_per_width: 34,
        controls: {
            wrong_inverse: "PASS",
            missing_inverse: "PASS",
            reordered_noncommuting_inverse: "PASS",
            snapshot_weaker_path: "PASS",
            attempted_h_projection: "PASS_REJECTED",
            eager_dense_request: "PASS_REJECTED",
            rank_cap_7: "PASS_REJECTED_BEFORE_MUTATION",
            null_carrier: "PASS_REJECTED",
            deterministic_replay: "PASS",
            analyzer: "PASS",
            asan_ubsan: "PASS",
            compact_reference_separation: "PASS",
            output_trace: "PASS_ONE_STDOUT_WRITE_ZERO_STDERR"
        },
        resources: ($native[0] | map({
            width,
            leaf_cells_each,
            resident_h_cells,
            final_z_cells,
            carrier_cells,
            live_carrier_bytes,
            verification_snapshot_bytes,
            projected_boundary_bytes,
            phase_cell_updates_per_transaction,
            logical_phase_ands_per_transaction,
            logical_phase_ors_per_transaction,
            carrier_reads_per_transaction,
            final_decodes_per_transaction
        })),
        best_matched_generic_classical_storage_bits:
            ($native[0] | map({
                width,
                bits: (368 * .width - 592)
            })),
        strongest_fixture_specialized_baseline:
            "DIRECT_PUBLIC_NEIGHBOR_AND_CUBED_RANK8_CORE_GENERATOR_O_N8",
        finite_product_rank_closure: true,
        fixed_rank_unbounded_depth_closure: false,
        native_rank_minimization_or_recompression: false,
        arbitrary_qanf_compactness: false,
        arbitrary_graph_topology: false,
        performance_advantage: false,
        small_wall_crossed: false,
        physical_execution: false,
        unlimited_catalytic_computation: false,
        sha256: {
            phase_source: $phase_sha,
            reference_source: $reference_sha,
            included_phase_engine: $included_sha,
            qualifier: $qualifier_sha
        },
        terminal: false
    }
' >"$evidence_dir/result.json"

find "$evidence_dir" -maxdepth 1 -type f \
    ! -name SHA256SUMS \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >"$evidence_dir/SHA256SUMS"

printf '%s\n' "$evidence_dir"
