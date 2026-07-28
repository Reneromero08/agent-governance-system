#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_phase_coupler_gauge_obstruction.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$1
mkdir -p "$out"
source_file="$here/phase_coupler_gauge_obstruction.c"
common=(
    -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror
    -Wconversion -Wshadow
)
for tool in gcc jq cmp sha256sum; do
    command -v "$tool" >/dev/null
done

gcc "${common[@]}" "$source_file" -lm -o "$out/diagnostic"
gcc "${common[@]}" -fanalyzer "$source_file" -lm \
    -o "$out/diagnostic_analyzer"
gcc "${common[@]}" -fstack-usage "$source_file" -lm \
    -o "$out/diagnostic_stack"
gcc \
    -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror \
    -Wconversion -Wshadow -fsanitize=address,undefined \
    -fno-omit-frame-pointer -fno-sanitize-recover=all \
    "$source_file" -lm -o "$out/diagnostic_sanitized"

"$out/diagnostic" >"$out/diagnostic.json"
"$out/diagnostic" >"$out/replay.json"
cmp "$out/diagnostic.json" "$out/replay.json"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$out/diagnostic_sanitized" \
    >"$out/sanitized.json" 2>"$out/sanitized.stderr"
[[ ! -s "$out/sanitized.stderr" ]]
cmp <(jq -S . "$out/diagnostic.json") <(jq -S . "$out/sanitized.json")

jq -e '
    .result == "PASS"
    and .claim
      == "ORDERED_THREE_UNIT_PHASOR_MEAN_HAS_SMOOTH_ZERO_LOCAL_SECTION_BUT_NO_GLOBAL_CONTINUOUS_GAUGE_FREE_DISK_SECTION"
    and .two_phasor_zero_jacobian_rank == 1
    and .three_phasor_equilateral_zero_jacobian_rank == 2
    and (
      (.fixed_third_phase_jacobian_determinant
        - 0.09622504486493763) | abs
    ) <= 2e-13
    and .local_section_linearization_residual <= 1e-9
    and .three_phasor_boundary_jacobian_rank == 1
    and (
      (.two_phasor_zero_jacobian_eigenvalues[0] - 0) | abs
    ) <= 2e-13
    and (
      (.two_phasor_zero_jacobian_eigenvalues[1] - 0.5) | abs
    ) <= 2e-13
    and (
      (.three_phasor_equilateral_zero_jacobian_eigenvalues[0]
        - 0.16666666666666666) | abs
    ) <= 2e-13
    and (
      (.three_phasor_equilateral_zero_jacobian_eigenvalues[1]
        - 0.16666666666666666) | abs
    ) <= 2e-13
    and .canonical_zero_amplitude_residual <= 2e-13
    and .shifted_zero_amplitude_residual <= 2e-13
    and .actual_phase_cell_residual_same_amplitude > 0.3
    and .boundary_sample_count == 64
    and .boundary_component_windings == [1, 1, 1]
    and .boundary_triangle_equality_unique_preimage
    and .disk_to_circle_extension_boundary_winding_required_zero
    and (.global_continuous_gauge_free_section | not)
    and (
      (.fixed_r_chart_failure_required_pair_sum_modulus - 4) | abs
    ) <= 2e-13
    and .fixed_r_chart_maximum_pair_sum_modulus == 2
    and (.fixed_r_chart_global | not)
    and .two_phasor_zero_local_section_singular
    and .three_phasor_zero_local_section_exists
    and .three_phasor_zero_local_fiber_dimension == 1
    and (.amplitude_only_state_distinguishes_gauge_orbit | not)
    and (
      .amplitude_only_canonicalization_restores_both_actual_carriers
      | not
    )
    and .gauge_retention_or_reversible_gauge_transport_required
    and (.global_smooth_section_over_closed_disk_established | not)
    and (.all_finite_phase_couplers_impossible | not)
    and (.host_amplitude_split_eliminated | not)
    and (.genuinely_distinct_phase_resource | not)
    and (.computational_advantage | not)
    and (.small_wall_crossed | not)
    and (.terminal | not)
' "$out/diagnostic.json" >/dev/null

stack_file=$(
    find "$out" -maxdepth 1 -name 'diagnostic_stack-*.su' -print -quit
)
maximum_stack=$(
    awk -F '\t' '$2 > maximum {maximum=$2} END {print maximum+0}' \
        "$stack_file"
)
jq -n \
    --argjson maximum_stack "$maximum_stack" \
    --slurpfile diagnostic "$out/diagnostic.json" '
    {
      result: "PASS",
      claim: $diagnostic[0].claim,
      claim_ceiling:
        "FINITE_ORDERED_UNIT_PHASOR_ARITHMETIC_MEAN_ENCODING_TOPOLOGICAL_OBSTRUCTION_SOFTWARE_REFERENCE_ONLY",
      accepted: $diagnostic[0],
      maximum_individual_compiler_stack_frame_bytes: $maximum_stack,
      obstruction:
        "THREE_PHASORS_REPAIR_LOCAL_ZERO_RANK_BUT_BOUNDARY_WINDING_FORBIDS_A_GLOBAL_CONTINUOUS_MEMORYLESS_DISK_SECTION",
      next_required_resource:
        "NATIVE_CONSERVED_PHASE_GEOMETRY_OR_REVERSIBLE_GAUGE_TRANSPORT_NOT_FACTORED_THROUGH_A_MEMORYLESS_DISK_SECTION",
      general_finite_phase_coupler_impossibility: false,
      distinct_phase_resource: false,
      terminal: false
    }
' >"$out/summary.json"

sha256sum \
    "$source_file" "$out/diagnostic" \
    "$out/diagnostic.json" "$out/summary.json" \
    >"$out/SHA256SUMS"
