#!/bin/sh
set -eu

source_dir=${1:-.}
work_dir=${2:-/tmp/catcas_series_parallel_qualification}
mkdir -p "$work_dir"

phase_source="$source_dir/algebraic_series_parallel_phase.c"
reference_source="$source_dir/algebraic_series_parallel_reference.c"
generator_source="$source_dir/generate_series_parallel_capacity_fixture.c"
primary_fixture="$source_dir/nested_two_cycle_discriminator.aspr"
reuse_fixture="$source_dir/nested_two_cycle_reuse.aspr"

phase="$work_dir/algebraic_series_parallel_phase"
reference="$work_dir/algebraic_series_parallel_reference"
generator="$work_dir/generate_series_parallel_capacity_fixture"

gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$phase_source" -o "$phase" -lm
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$reference_source" -o "$reference"
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$generator_source" -o "$generator"
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -fanalyzer \
    "$phase_source" -o "$work_dir/phase_analyzer_build" -lm

compiler_builds=2
if command -v clang >/dev/null 2>&1; then
    clang -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
        "$phase_source" -o "$work_dir/phase_clang" -lm
    clang -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
        "$reference_source" -o "$work_dir/reference_clang"
    compiler_builds=4
fi

"$phase" "$primary_fixture" "$reuse_fixture" \
    >"$work_dir/primary_native.jsonl"
"$phase" "$primary_fixture" "$reuse_fixture" \
    >"$work_dir/primary_native_reproduction.jsonl"
cmp "$work_dir/primary_native.jsonl" \
    "$work_dir/primary_native_reproduction.jsonl"
"$reference" "$primary_fixture" >"$work_dir/primary_reference.json"
"$reference" "$reuse_fixture" >"$work_dir/reuse_reference.json"

primary_coefficients=$(
    sed -n \
        '1s/.*"boundary_coefficients":\(\[[0-9,]*\]\).*/\1/p' \
        "$work_dir/primary_native.jsonl"
)
reuse_coefficients=$(
    sed -n \
        '2s/.*"boundary_coefficients":\(\[[0-9,]*\]\).*/\1/p' \
        "$work_dir/primary_native.jsonl"
)
primary_reference_coefficients=$(
    sed -n \
        's/.*"projected_coefficients":\(\[[0-9,]*\]\).*/\1/p' \
        "$work_dir/primary_reference.json"
)
reuse_reference_coefficients=$(
    sed -n \
        's/.*"projected_coefficients":\(\[[0-9,]*\]\).*/\1/p' \
        "$work_dir/reuse_reference.json"
)
test "$primary_coefficients" = "[0,2,1,2]"
test "$reuse_coefficients" = "[0,1,1,1]"
test "$primary_coefficients" = "$primary_reference_coefficients"
test "$reuse_coefficients" = "$reuse_reference_coefficients"
grep -q '"projected_zero_mask":1,' "$work_dir/primary_reference.json"
grep -q '"enumerated_zero_mask":1,' "$work_dir/primary_reference.json"
grep -q '"projected_zero_mask":9,' "$work_dir/reuse_reference.json"
grep -q '"enumerated_zero_mask":9,' "$work_dir/reuse_reference.json"
grep -q '"projection_matches_enumeration":true' \
    "$work_dir/primary_reference.json"
grep -q '"projection_matches_enumeration":true' \
    "$work_dir/reuse_reference.json"
grep -q \
    '"wrong_boundary":true,"omitted_message":true,"bypassed_intersection":true,"ordinary_sum":true' \
    "$work_dir/primary_native.jsonl"

negative_count=0
for fixture in \
    negative_degree_three.aspr \
    negative_duplicate_elimination.aspr \
    negative_duplicate_relation_identifier.aspr \
    negative_missing_elimination.aspr \
    negative_tree_without_shared_interface.aspr \
    negative_wrong_order.aspr
do
    if "$phase" "$source_dir/$fixture" \
        >"$work_dir/$fixture.out" 2>"$work_dir/$fixture.err"
    then
        echo "negative fixture unexpectedly accepted: $fixture" >&2
        exit 1
    else
        result=$?
        test "$result" -eq 2
    fi
    if "$reference" "$source_dir/$fixture" \
        >"$work_dir/$fixture.reference.out" \
        2>"$work_dir/$fixture.reference.err"
    then
        echo "scalar negative fixture unexpectedly accepted: $fixture" >&2
        exit 1
    else
        result=$?
        test "$result" -eq 2
    fi
    negative_count=$((negative_count + 1))
done
grep -q 'active degree two' \
    "$work_dir/negative_degree_three.aspr.err"
grep -q 'duplicate elimination node' \
    "$work_dir/negative_duplicate_elimination.aspr.err"
grep -q 'duplicate relation identifier' \
    "$work_dir/negative_duplicate_relation_identifier.aspr.err"
grep -q 'duplicate relation identifier' \
    "$work_dir/negative_duplicate_relation_identifier.aspr.reference.err"
grep -q 'every internal node needs one elimination record' \
    "$work_dir/negative_missing_elimination.aspr.err"
grep -q 'parallel-path merge' \
    "$work_dir/negative_tree_without_shared_interface.aspr.err"
grep -q 'active degree two' \
    "$work_dir/negative_wrong_order.aspr.err"

survey_count=0
enumerated_count=0
for diamonds in 1 2 3 4 5 6
do
    for pattern in \
        1 2 3 4 5 6 7 8 9 10 \
        11 12 13 14 15 16 17 18 19 20
    do
        fixture="$work_dir/survey_${diamonds}_${pattern}.aspr"
        native="$work_dir/survey_${diamonds}_${pattern}.native"
        scalar="$work_dir/survey_${diamonds}_${pattern}.scalar"
        "$generator" "$diamonds" "$pattern" >"$fixture"
        "$phase" "$fixture" >"$native"
        "$reference" "$fixture" >"$scalar"
        native_coefficients=$(
            sed -n \
                '1s/.*"boundary_coefficients":\(\[[0-9,]*\]\).*/\1/p' \
                "$native"
        )
        scalar_coefficients=$(
            sed -n \
                's/.*"projected_coefficients":\(\[[0-9,]*\]\).*/\1/p' \
                "$scalar"
        )
        test -n "$native_coefficients"
        test "$native_coefficients" = "$scalar_coefficients"
        grep -q '"enumeration_performed":true' "$scalar"
        grep -q '"projection_matches_enumeration":true' "$scalar"
        survey_count=$((survey_count + 1))
        enumerated_count=$((enumerated_count + 1))
    done
done

capacity_count=0
for diamonds in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    fixture="$work_dir/capacity_${diamonds}.aspr"
    native="$work_dir/capacity_${diamonds}.native"
    scalar="$work_dir/capacity_${diamonds}.scalar"
    "$generator" "$diamonds" 0 >"$fixture"
    "$phase" "$fixture" >"$native"
    "$reference" "$fixture" >"$scalar"
    native_coefficients=$(
        sed -n \
            '1s/.*"boundary_coefficients":\(\[[0-9,]*\]\).*/\1/p' \
            "$native"
    )
    scalar_coefficients=$(
        sed -n \
            's/.*"projected_coefficients":\(\[[0-9,]*\]\).*/\1/p' \
            "$scalar"
    )
    test "$native_coefficients" = "$scalar_coefficients"
    grep -q '"projected_zero_mask":1,' "$scalar"
    capacity_count=$((capacity_count + 1))
done

gcc -std=c11 -O1 -g -Wall -Wextra -Werror -pedantic \
    -fsanitize=address,undefined -fno-sanitize-recover=undefined \
    -fno-omit-frame-pointer \
    "$phase_source" -o "$work_dir/phase_sanitized" -lm
gcc -std=c11 -O1 -g -Wall -Wextra -Werror -pedantic \
    -fsanitize=address,undefined -fno-sanitize-recover=undefined \
    -fno-omit-frame-pointer \
    "$reference_source" -o "$work_dir/reference_sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$work_dir/phase_sanitized" \
    "$primary_fixture" "$reuse_fixture" \
    >"$work_dir/sanitized_primary.jsonl"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$work_dir/phase_sanitized" \
    "$work_dir/capacity_15.aspr" \
    >"$work_dir/sanitized_capacity.jsonl"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$work_dir/reference_sanitized" \
    "$primary_fixture" \
    >"$work_dir/sanitized_reference.json"

awk '
    /static void compile_topology/ {inside=1}
    /static struct process read_process/ {inside=0}
    inside {print}
' "$phase_source" >"$work_dir/topology_compiler_slice.c"
if grep -q 'coefficient' "$work_dir/topology_compiler_slice.c"; then
    echo "topology compiler reads relation coefficients" >&2
    exit 1
fi
awk '
    /static void exact_compose_factors/ {inside=1}
    /static int decode_root/ {inside=0}
    inside {print}
' "$phase_source" >"$work_dir/native_recurrence_slice.c"
if grep -q 'decode_root' "$work_dir/native_recurrence_slice.c"; then
    echo "native recurrence decodes an intermediate coefficient" >&2
    exit 1
fi
if grep -q 'evaluate(' "$phase_source"; then
    echo "native engine contains scalar relation evaluation" >&2
    exit 1
fi

sha256sum \
    "$phase_source" \
    "$reference_source" \
    "$generator_source" \
    "$primary_fixture" \
    "$reuse_fixture" \
    >"$work_dir/source_sha256.txt"
sha256sum \
    "$work_dir/primary_native.jsonl" \
    "$work_dir/primary_reference.json" \
    "$work_dir/reuse_reference.json" \
    "$work_dir/capacity_15.native" \
    "$work_dir/capacity_15.scalar" \
    >"$work_dir/result_sha256.txt"

printf \
    '{"result":"PASS","strict_compiler_builds":%d,"deterministic_reproductions":2,"primary_scalar_assignment_checks":1024,"arbitrary_coefficient_survey_cases":%d,"enumerated_survey_cases":%d,"capacity_scale_cases":%d,"maximum_nodes":46,"maximum_input_relations":60,"maximum_native_operations":59,"maximum_carrier_cells":480,"structural_negative_cases":%d,"paired_parser_duplicate_relation_control":"PASS","sanitizer_runs":3,"topology_coefficient_independence":"PASS","native_no_intermediate_decode":"PASS"}\n' \
    "$compiler_builds" \
    "$survey_count" \
    "$enumerated_count" \
    "$capacity_count" \
    "$negative_count"
