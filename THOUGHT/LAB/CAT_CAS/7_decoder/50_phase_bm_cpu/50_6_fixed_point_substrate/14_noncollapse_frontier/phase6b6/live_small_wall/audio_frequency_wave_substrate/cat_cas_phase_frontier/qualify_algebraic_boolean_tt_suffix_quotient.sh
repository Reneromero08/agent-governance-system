#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
ulimit -c 0

if [[ $# -ne 1 ]]; then
    printf 'usage: qualify_algebraic_boolean_tt_suffix_quotient.sh EVIDENCE_DIR\n' >&2
    exit 2
fi

frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
evidence_dir=$1
if [[ -e "$evidence_dir" ]]; then
    printf 'evidence directory already exists: %s\n' "$evidence_dir" >&2
    exit 2
fi
mkdir -p "$evidence_dir"

phase_source="$frontier_dir/algebraic_boolean_tt_suffix_quotient_phase.c"
reference_source="$frontier_dir/algebraic_boolean_tt_suffix_quotient_reference.c"
qualifier_source="$frontier_dir/qualify_algebraic_boolean_tt_suffix_quotient.sh"
included_source="$frontier_dir/algebraic_series_parallel_phase.c"
predecessor_qualifier="$frontier_dir/qualify_catvm_boolean_tt_phase.sh"

for tool in gcc jq sha256sum cmp rg strace nm; do
    command -v "$tool" >/dev/null
done

common=(
    -std=c11
    -O2
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wconversion
    -Wshadow
)
sanitizer=(
    -std=c11
    -O1
    -g
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wconversion
    -Wshadow
    -fsanitize=address,undefined
    -fno-omit-frame-pointer
    -fno-sanitize-recover=all
)

gcc "${common[@]}" "$phase_source" -lm \
    -o "$evidence_dir/phase"
gcc "${common[@]}" "$reference_source" \
    -o "$evidence_dir/reference"
gcc "${common[@]}" -DQTT_BAD_HORIZON_CONTROL=1 \
    "$phase_source" -lm \
    -o "$evidence_dir/control_bad_horizon"
gcc "${common[@]}" -DQTT_BAD_OR_CONTROL=1 \
    "$phase_source" -lm \
    -o "$evidence_dir/control_bad_or"
gcc "${common[@]}" -fanalyzer -c "$phase_source" \
    -o "$evidence_dir/phase_analyzer.o"
gcc "${common[@]}" -fanalyzer -c "$reference_source" \
    -o "$evidence_dir/reference_analyzer.o"
gcc "${sanitizer[@]}" "$phase_source" -lm \
    -o "$evidence_dir/phase_sanitized"
gcc "${sanitizer[@]}" "$reference_source" \
    -o "$evidence_dir/reference_sanitized"

cases=(
    "4 2" "5 2" "8 2" "12 2" "16 2"
    "4 3" "5 3" "8 3" "12 3" "16 3"
    "4 4" "5 5" "8 8" "12 8" "16 8"
)

: >"$evidence_dir/native.jsonl"
: >"$evidence_dir/reference.jsonl"
for case in "${cases[@]}"; do
    read -r width depth <<<"$case"
    "$evidence_dir/phase" "$width" "$depth" \
        >>"$evidence_dir/native.jsonl"
    "$evidence_dir/reference" "$width" "$depth" \
        >>"$evidence_dir/reference.jsonl"
done
cp "$evidence_dir/native.jsonl" "$evidence_dir/native_first.jsonl"
for case in "${cases[@]}"; do
    read -r width depth <<<"$case"
    "$evidence_dir/phase" "$width" "$depth" \
        >>"$evidence_dir/native_replay.jsonl"
done
cmp "$evidence_dir/native_first.jsonl" \
    "$evidence_dir/native_replay.jsonl"

jq -s '.' "$evidence_dir/native.jsonl" \
    >"$evidence_dir/native.json"
jq -s '.' "$evidence_dir/reference.jsonl" \
    >"$evidence_dir/reference.json"

jq -n -e \
    --slurpfile native "$evidence_dir/native.json" \
    --slurpfile reference "$evidence_dir/reference.json" '
    ($native[0]) as $n
    | ($reference[0]) as $r
    | ($n | length) == 15
    and ($r | length) == 15
    and all(range(0; 15);
        $n[.].width == $r[.].width
        and $n[.].depth == $r[.].depth
        and $n[.].final_quotient_cells
            == $r[.].final_quotient_cells
        and $n[.].primary_boundary_fnv1a64
            == $r[.].and_boundary_fnv1a64
        and $n[.].reuse_boundary_fnv1a64
            == $r[.].or_boundary_fnv1a64
        and $n[.].primary_boundary_ones
            == $r[.].and_boundary_ones
        and $n[.].reuse_boundary_ones
            == $r[.].or_boundary_ones
        and $n[.].same_carrier_transactions == 34
        and $n[.].decoded_intermediate_cells == 0
        and $n[.].serialized_intermediate_cells == 0
        and $n[.].materialized_raw_product_cells == 0
        and $n[.].width_wide_assignments_enumerated == 0
        and $n[.].truth_table_slots == 0
        and $n[.].actual_resident_stage_consumed_by_successor
        and $n[.].actual_inverse_restoration
        and $n[.].actual_restored_carrier_reuse
        and $n[.].family_scoped_analytic_quotient
        and ($n[.].general_boolean_tt_minimization | not)
        and ($n[.].fixed_rank_unbounded_depth_closure | not)
        and ($n[.].computational_advantage | not)
        and ($n[.].small_wall_crossed | not)
        and $r[.].class_uniformity_checks > 0
        and $r[.].quotient_edge_checks > 0
        and $r[.].width_wide_assignments_enumerated == 0
        and $r[.].dense_relation_cells_materialized == 0
        and $r[.].raw_product_tt_cores_materialized_for_verification
        and $r[.].raw_product_tt_final_stage_cells
            == $n[.].raw_product_final_cells
        and $r[.].raw_product_tt_final_stage_bytes
            == $n[.].raw_product_final_cells
        and $r[.].raw_intermediate_relations_serialized == 0
    )
' >"$evidence_dir/native_reference_parity.txt"

jq -e '
    all(.[];
        .raw_product_rank == pow(2; .depth)
        and .raw_product_final_cells
            == 8 * pow(2; .depth)
                + 4 * (.width - 2) * pow(4; .depth)
        and .maximum_quotient_rank == .depth + 1
        and .final_quotient_cells < .raw_product_final_cells
        and .final_cell_reduction_ratio > 1
        and .carrier_cells
            == .retained_stage_cells + .boundary_cells
        and .logical_phase_ands_per_transaction
            == .quotient_member_terms_per_transaction
        and .carrier_reads_per_transaction
            > 2 * .logical_phase_ands_per_transaction
        and .phase_cell_updates_per_transaction
            == 2 * .carrier_cells
        and .final_decodes_per_transaction == .boundary_cells
        and .wrong_inverse_restoration_error > 1
        and .missing_inverse_restoration_error > 1
        and .reordered_inverse_restoration_error > 1
        and .maximum_repeated_restoration_error < 2e-12
        and .snapshot_loaded
        and .snapshot_generation == 0
    )
    and all(.[] | select(.depth == 2);
        .final_quotient_cells == 36 * .width - 64
    )
    and all(.[] | select(.depth == 3);
        .final_quotient_cells == 64 * .width - 136
    )
' "$evidence_dir/native.json" \
    >"$evidence_dir/resource_assertion.txt"

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$evidence_dir/phase_sanitized" 4 3 \
    >"$evidence_dir/asan_phase_width4_depth3.json"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$evidence_dir/reference_sanitized" 4 3 \
    >"$evidence_dir/asan_reference_width4_depth3.json"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$evidence_dir/phase_sanitized" 16 8 \
    >"$evidence_dir/asan_phase_width16_depth8.json"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$evidence_dir/reference_sanitized" 16 8 \
    >"$evidence_dir/asan_reference_width16_depth8.json"

set +e
"$evidence_dir/control_bad_horizon" 4 3 \
    >"$evidence_dir/control_bad_horizon.json" \
    2>"$evidence_dir/control_bad_horizon.stderr"
bad_horizon_rc=$?
"$evidence_dir/control_bad_or" 4 3 \
    >"$evidence_dir/control_bad_or.stdout" \
    2>"$evidence_dir/control_bad_or.stderr"
bad_or_rc=$?
"$evidence_dir/phase" --project-intermediate 4 3 \
    >"$evidence_dir/project_intermediate.stdout" \
    2>"$evidence_dir/project_intermediate.stderr"
project_rc=$?
"$evidence_dir/phase" --null 4 3 \
    >"$evidence_dir/null_carrier.stdout" \
    2>"$evidence_dir/null_carrier.stderr"
null_rc=$?
"$evidence_dir/phase" 4 5 \
    >"$evidence_dir/depth_exceeds_width.stdout" \
    2>"$evidence_dir/depth_exceeds_width.stderr"
depth_rc=$?
"$evidence_dir/phase" 04 3 \
    >"$evidence_dir/noncanonical_width.stdout" \
    2>"$evidence_dir/noncanonical_width.stderr"
noncanonical_rc=$?
set -e

[[ "$bad_horizon_rc" -eq 0 ]]
[[ "$bad_or_rc" -eq 2 ]]
[[ "$project_rc" -eq 2 ]]
[[ "$null_rc" -eq 2 ]]
[[ "$depth_rc" -eq 2 ]]
[[ "$noncanonical_rc" -eq 2 ]]
[[ ! -s "$evidence_dir/control_bad_horizon.stderr" ]]
[[ ! -s "$evidence_dir/control_bad_or.stdout" ]]
[[ ! -s "$evidence_dir/project_intermediate.stdout" ]]
[[ ! -s "$evidence_dir/null_carrier.stdout" ]]
[[ ! -s "$evidence_dir/depth_exceeds_width.stdout" ]]
[[ ! -s "$evidence_dir/noncanonical_width.stdout" ]]

jq -n -e \
    --slurpfile bad "$evidence_dir/control_bad_horizon.json" \
    --slurpfile good "$evidence_dir/reference.json" '
    ($good[0][] | select(.width == 4 and .depth == 3)) as $g
    | $bad[0].final_quotient_cells != $g.final_quotient_cells
    and $bad[0].primary_boundary_fnv1a64
        != $g.and_boundary_fnv1a64
' >"$evidence_dir/control_bad_horizon_detected.txt"
rg -q 'left Boolean alphabet' \
    "$evidence_dir/control_bad_or.stderr"
rg -q 'only the final quotient-TT boundary' \
    "$evidence_dir/project_intermediate.stderr"
rg -q 'null quotient-TT carrier' \
    "$evidence_dir/null_carrier.stderr"
rg -q 'depth may not exceed width' \
    "$evidence_dir/depth_exceeds_width.stderr"
rg -q 'canonical decimal' \
    "$evidence_dir/noncanonical_width.stderr"

strace -qq -f -e trace=write \
    -o "$evidence_dir/output_trace.txt" \
    "$evidence_dir/phase" 4 3 \
    >"$evidence_dir/traced_width4_depth3.json" \
    2>"$evidence_dir/traced_width4_depth3.stderr"
[[ ! -s "$evidence_dir/traced_width4_depth3.stderr" ]]
test "$(rg -c 'write\(1,' "$evidence_dir/output_trace.txt")" -eq 1
test -z "$(rg 'write\(2,' "$evidence_dir/output_trace.txt" || true)"

nm -a "$evidence_dir/phase" >"$evidence_dir/phase_symbols.txt"
nm -a "$evidence_dir/reference" >"$evidence_dir/reference_symbols.txt"
rg 'qtt_compose_quotient|qtt_execute|symbol_product' \
    "$evidence_dir/phase_symbols.txt" \
    >"$evidence_dir/phase_native_symbols.txt"
test -z "$(rg \
    'qtt_execute|symbol_product|multiply_cell|decode_root' \
    "$evidence_dir/reference_symbols.txt" || true)"

if rg -n \
    '1U[[:space:]]*<<[[:space:]]*width|4\^width|truth_table[[:space:]]*\[|candidate_set[[:space:]]*\[|witness_list[[:space:]]*\[' \
    "$phase_source" \
    >"$evidence_dir/phase_forbidden_static.txt"; then
    printf 'phase source contains forbidden assignment expansion\n' >&2
    exit 1
fi
: >"$evidence_dir/phase_forbidden_static.txt"

bash "$predecessor_qualifier" \
    "$evidence_dir/catvm_predecessor_regression" \
    >"$evidence_dir/catvm_predecessor_path.txt"
sha256sum -c \
    "$evidence_dir/catvm_predecessor_regression/SHA256SUMS" \
    >"$evidence_dir/catvm_predecessor_sha_check.txt"

sha256sum \
    "$phase_source" \
    "$reference_source" \
    "$qualifier_source" \
    "$included_source" \
    "$evidence_dir/phase" \
    "$evidence_dir/reference" \
    "$evidence_dir/native.jsonl" \
    "$evidence_dir/reference.jsonl" \
    "$evidence_dir/native_reference_parity.txt" \
    "$evidence_dir/resource_assertion.txt" \
    "$evidence_dir/control_bad_horizon.json" \
    "$evidence_dir/control_bad_horizon_detected.txt" \
    "$evidence_dir/control_bad_or.stderr" \
    "$evidence_dir/output_trace.txt" \
    "$evidence_dir/catvm_predecessor_regression/SHA256SUMS" \
    "$evidence_dir/catvm_predecessor_regression/result.json" \
    >"$evidence_dir/CORE_SHA256SUMS"

jq -n \
    --slurpfile native "$evidence_dir/native.json" \
    --slurpfile reference "$evidence_dir/reference.json" \
    --arg evidence "$evidence_dir" \
    --arg phase_sha "$(sha256sum "$phase_source" | cut -d' ' -f1)" \
    --arg reference_sha "$(sha256sum "$reference_source" | cut -d' ' -f1)" \
    --arg qualifier_sha "$(sha256sum "$qualifier_source" | cut -d' ' -f1)" '
    {
        result: "PASS",
        claim:
            "BOUNDED_BOOLEAN_TT_SUFFIX_BISIMULATION_QUOTIENT_"
            + "REDUCES_PRODUCT_RANK_GROWTH_FROM_EXPONENTIAL_TO_LINEAR_"
            + "WITH_PHASE_RESIDENT_CLOSURE",
        claim_ceiling:
            "BOUNDED_LINUX_SOFTWARE_WIDTHS4_5_8_12_16_DEPTHS2_3_4_5_8_"
            + "NEIGHBOR_AND_OR_FAMILY_SCOPED_QUOTIENT_REFERENCE_ONLY",
        evidence_directory: $evidence,
        tested_cases:
            ($native[0] | map({width, depth})),
        phase: $native[0],
        reference_certificates: $reference[0],
        product_rank_growth: "2^depth",
        quotient_maximum_rank_growth: "depth+1_for_width_at_least_depth",
        fixed_width_saturation_upper_bound: "width+1",
        h_depth2_cells: "36w-64",
        z_depth3_cells: "64w-136",
        materialized_raw_product_cells_in_accepted_phase_path: 0,
        verification_reference: {
            raw_product_tt_cores_materialized: true,
            maximum_final_stage_cells: 3672064,
            maximum_final_stage_bytes: 3672064,
            width_wide_assignments_enumerated: 0,
            dense_4_pow_width_relation_materialized: false,
            raw_intermediate_relations_serialized: false
        },
        decoded_intermediate_cells: 0,
        serialized_intermediate_cells: 0,
        width_wide_assignments_enumerated: 0,
        truth_table_slots: 0,
        actual_inverse_restoration: true,
        actual_restored_carrier_reuse: true,
        controls: {
            wrong_inverse: "PASS_DETECTED",
            missing_inverse: "PASS_DETECTED",
            reordered_noncommuting_inverse: "PASS_DETECTED",
            snapshot: "PASS_SEPARATE_GENERATION_ZERO",
            bad_horizon_overmerge: "PASS_REFERENCE_PARITY_FAILED",
            bad_boolean_or: "PASS_ROOT_ALPHABET_FAILED",
            intermediate_projection: "PASS_REJECTED",
            null_carrier: "PASS_REJECTED",
            depth_exceeds_width: "PASS_REJECTED",
            deterministic_replay: "PASS",
            analyzer: "PASS",
            asan_ubsan: "PASS",
            catvm_predecessor_regression: "PASS"
        },
        strongest_fixture_specialized_baseline:
            "DIRECT_PUBLIC_SUFFIX_QUOTIENT_CORE_GENERATOR_O_FINAL_CELLS",
        family_scoped_analytic_quotient: true,
        general_boolean_tt_minimization: false,
        globally_minimal_boolean_tt_rank: false,
        fixed_rank_unbounded_depth_closure: false,
        compact_inverse_history_recursion: false,
        computational_advantage: false,
        small_wall_crossed: false,
        physical_execution: false,
        unlimited_catalytic_computation: false,
        sha256: {
            phase_source: $phase_sha,
            reference_source: $reference_sha,
            qualifier: $qualifier_sha
        },
        terminal: false
    }
' >"$evidence_dir/result.json"

find "$evidence_dir" -type f \
    ! -name SHA256SUMS \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >"$evidence_dir/SHA256SUMS"

printf '%s\n' "$evidence_dir"
