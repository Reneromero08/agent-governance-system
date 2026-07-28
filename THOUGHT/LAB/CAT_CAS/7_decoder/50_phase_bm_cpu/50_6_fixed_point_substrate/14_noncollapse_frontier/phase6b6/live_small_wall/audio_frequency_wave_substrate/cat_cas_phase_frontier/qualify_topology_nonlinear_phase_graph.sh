#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_topology_nonlinear_phase_graph.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$1
mkdir -p "$out"
phase_source="$here/topology_nonlinear_phase_graph.c"
reference_source="$here/topology_nonlinear_phase_graph_reference.c"
topology="$here/nonlinear_phase_graph_topology.txt"
ordered="$here/nonlinear_phase_graph_topology_ordered.txt"
unordered="$here/nonlinear_phase_graph_topology_unordered_hazard.txt"

for tool in gcc jq cmp rg sha256sum nice; do
    command -v "$tool" >/dev/null
done

common=(
    -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror
    -Wconversion -Wshadow
)
nice -n 10 gcc "${common[@]}" "$phase_source" -lm -o "$out/phase"
nice -n 10 gcc "${common[@]}" "$reference_source" -lm \
    -o "$out/reference"
nice -n 10 gcc "${common[@]}" -fanalyzer "$phase_source" -lm \
    -o "$out/phase_analyzer"
nice -n 10 gcc "${common[@]}" -fstack-usage "$phase_source" -lm \
    -o "$out/phase_stack"
nice -n 10 gcc \
    -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror \
    -Wconversion -Wshadow -fsanitize=address,undefined \
    -fno-omit-frame-pointer -fno-sanitize-recover=all \
    "$phase_source" -lm -o "$out/phase_sanitized"

rounds=(1 3 128 512 2048 4096)
for round_count in "${rounds[@]}"; do
    nice -n 10 "$out/phase" "$topology" "$round_count" \
        >"$out/phase_r${round_count}.json"
    nice -n 10 "$out/phase" "$topology" "$round_count" \
        >"$out/phase_r${round_count}_replay.json"
    cmp \
        "$out/phase_r${round_count}.json" \
        "$out/phase_r${round_count}_replay.json"
    nice -n 10 "$out/reference" "$topology" "$round_count" \
        >"$out/reference_r${round_count}.json"
    jq -e \
        --slurpfile reference "$out/reference_r${round_count}.json" '
        .claim
          == "BOUNDED_TOPOLOGY_COMPILED_SHARED_NONLINEAR_PHASE_GRAPH_WITH_ACTUAL_RESTORATION_AND_REUSE"
        and .primary_boundary_fnv1a64
          == $reference[0].primary_boundary_fnv1a64
        and .reuse_boundary_fnv1a64
          == $reference[0].reuse_boundary_fnv1a64
        and (
          .primary_interference_probability
          - $reference[0].primary_interference_probability
          | fabs
        ) <= 2e-10
        and .width == 4
        and .edges == 6
        and .shared_sources == 2
        and .maximum_source_fanout == 2
        and .compiled_precedence_pairs == 12
        and .resident_phase_cells == 4
        and .boundary_phase_cells == 2
        and .carrier_cells == 6
        and .best_matched_classical_state_bytes
          == $reference[0].classical_state_bytes
        and .forward_shears == (6 * .rounds)
        and .inverse_shears == (6 * .rounds)
        and .maximum_unit_modulus_error <= 2e-11
        and .maximum_repeated_restoration_error <= 2e-11
        and .missing_inverse_residual >= 1e-6
        and .wrong_inverse_residual >= 1e-6
        and .reordered_inverse_residual >= 1e-6
        and .snapshot_residual <= 2e-11
        and .projected_intermediate_phases == 0
        and .serialized_intermediate_phases == 0
        and .materialized_phase_paths == 0
        and .topology_only_compilation
        and .deterministic_topological_scheduler
        and .unordered_hazards_rejected
        and .shared_resident_phase_sources
        and .nonlinear_phase_coupling
        and .native_phase_updates_only
        and .actual_inverse_restoration
        and .actual_restored_carrier_reuse
        and .fixed_carrier_across_rounds
        and .snapshot_loaded_separately
        and .compact_classical_width_angle_recurrence_exists
        and (.machine_enforced_hidden_intermediate | not)
        and (.genuinely_distinct_phase_resource | not)
        and (.computational_advantage | not)
        and (.small_wall_crossed | not)
        and (.physical_execution | not)
        and (.unlimited_catalytic_computation | not)
        and (.terminal | not)
    ' "$out/phase_r${round_count}.json" >/dev/null
done

nice -n 10 "$out/phase" "$ordered" 128 \
    >"$out/ordered_r128.json"
jq -e \
    --slurpfile scrambled "$out/phase_r128.json" '
    .topology_fnv1a64 == $scrambled[0].topology_fnv1a64
    and .primary_boundary_fnv1a64
      == $scrambled[0].primary_boundary_fnv1a64
    and .reuse_boundary_fnv1a64
      == $scrambled[0].reuse_boundary_fnv1a64
' "$out/ordered_r128.json" >/dev/null

set +e
"$out/phase" "$unordered" 3 \
    >"$out/unordered.stdout" 2>"$out/unordered.stderr"
unordered_rc=$?
"$out/phase" --project-intermediate \
    >"$out/project.stdout" 2>"$out/project.stderr"
project_rc=$?
"$out/phase" --null-carrier \
    >"$out/null.stdout" 2>"$out/null.stderr"
null_rc=$?
set -e
[[ "$unordered_rc" -eq 2 ]]
[[ "$project_rc" -eq 2 ]]
[[ "$null_rc" -eq 2 ]]
[[ ! -s "$out/unordered.stdout" ]]
[[ ! -s "$out/project.stdout" ]]
[[ ! -s "$out/null.stdout" ]]
rg -q '^nonlinear phase graph has unordered hazard$' \
    "$out/unordered.stderr"
rg -q '^nonlinear graph intermediate projection denied$' \
    "$out/project.stderr"
rg -q '^invalid topology nonlinear phase transaction$' \
    "$out/null.stderr"

for round_count in 3 4096; do
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        nice -n 10 "$out/phase_sanitized" \
        "$topology" "$round_count" \
        >"$out/sanitized_r${round_count}.json" \
        2>"$out/sanitized_r${round_count}.stderr"
    [[ ! -s "$out/sanitized_r${round_count}.stderr" ]]
    cmp \
        <(jq -S . "$out/phase_r${round_count}.json") \
        <(jq -S . "$out/sanitized_r${round_count}.json")
done

if rg -n \
    'truth_table|witness_list|candidate_set|assignment_expansion|expected_(hash|boundary)' \
    "$phase_source"
then
    echo "nonlinear graph path contains forbidden extensional state" >&2
    exit 1
fi
rg -q 'relative\(' "$phase_source"
rg -q 'multiply_cell\(' "$phase_source"
if rg -q 'carrier->working\\[[^]]+\\][[:space:]]*[-+*/]?=' \
    "$phase_source"
then
    echo "nonlinear graph bypasses native phase update" >&2
    exit 1
fi

stack_file=$(
    find "$out" -maxdepth 1 -name 'phase_stack-*.su' -print -quit
)
maximum_stack_frame=$(
    awk -F '\t' '$2 > maximum {maximum=$2} END {print maximum+0}' \
        "$stack_file"
)
jq -s 'sort_by(.rounds)' "$out"/phase_r[0-9]*.json \
    >"$out/phase_results.json"
jq -s 'sort_by(.rounds)' "$out"/reference_r[0-9]*.json \
    >"$out/reference_results.json"
jq -n \
    --arg claim \
      "BOUNDED_TOPOLOGY_COMPILED_SHARED_NONLINEAR_PHASE_GRAPH_WITH_ACTUAL_RESTORATION_AND_REUSE" \
    --arg ceiling \
      "BOUNDED_LINUX_SOFTWARE_WIDTH4_EDGES6_ROUNDS1_3_128_512_2048_4096_DOUBLE_COMPLEX_PHASE_REFERENCE_ONLY" \
    --argjson maximum_stack_frame "$maximum_stack_frame" \
    --slurpfile cases "$out/phase_results.json" '
    {
      result: "PASS",
      claim: $claim,
      claim_ceiling: $ceiling,
      cases: $cases[0],
      maximum_compiler_stack_frame_bytes: $maximum_stack_frame,
      phase_owned_advance:
        "PUBLIC_DEPENDENCY_COMPILED_SHARED_NONLINEAR_PHASE_GRAPH",
      controls: {
        exact_compact_angle_reference_parity: true,
        manifest_order_independence: true,
        unordered_hazard_rejected: true,
        wrong_missing_reordered_inverse_detected: true,
        snapshot_separated: true,
        restored_carrier_reuse: true,
        intermediate_projection_rejected: true,
        null_carrier_execution_rejected: true,
        analyzer: "PASS",
        asan_ubsan: "PASS"
      },
      remaining_primary_obstruction:
        "EQUIVALENT_WIDTH_DOUBLE_ANGLE_GRAPH_RECURRENCE",
      machine_enforced_hidden_intermediate: false,
      genuinely_distinct_phase_resource: false,
      computational_advantage: false,
      small_wall_crossed: false,
      terminal: false
    }
' >"$out/summary.json"

sha256sum \
    "$phase_source" "$reference_source" \
    "$topology" "$ordered" "$unordered" \
    "$out/phase" "$out/reference" \
    "$out/phase_results.json" "$out/reference_results.json" \
    "$out/summary.json" \
    >"$out/SHA256SUMS"
