#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_algebraic_nonlinear_kerr_wave.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$1
mkdir -p "$out"
phase_source="$here/algebraic_nonlinear_kerr_wave.c"
reference_source="$here/algebraic_nonlinear_kerr_wave_reference.c"
common=(
    -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror
    -Wconversion -Wshadow
)
for tool in gcc jq cmp rg sha256sum; do
    command -v "$tool" >/dev/null
done

gcc "${common[@]}" "$phase_source" -lm -o "$out/phase"
gcc "${common[@]}" "$reference_source" -lm -o "$out/reference"
gcc "${common[@]}" -fanalyzer "$phase_source" -lm \
    -o "$out/phase_analyzer"
gcc "${common[@]}" -fanalyzer "$reference_source" -lm \
    -o "$out/reference_analyzer"
gcc "${common[@]}" -fstack-usage "$phase_source" -lm \
    -o "$out/phase_stack"
gcc \
    -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror \
    -Wconversion -Wshadow -fsanitize=address,undefined \
    -fno-omit-frame-pointer -fno-sanitize-recover=all \
    "$phase_source" -lm -o "$out/phase_sanitized"

"$out/phase" >"$out/phase.json"
"$out/phase" >"$out/replay.json"
"$out/reference" >"$out/reference.json"
cmp "$out/phase.json" "$out/replay.json"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$out/phase_sanitized" \
    >"$out/sanitized.json" 2>"$out/sanitized.stderr"
[[ ! -s "$out/sanitized.stderr" ]]
cmp <(jq -S . "$out/phase.json") <(jq -S . "$out/sanitized.json")

if "$out/phase" --project-intermediate \
    >"$out/project-intermediate.stdout" \
    2>"$out/project-intermediate.stderr"; then
    echo "intermediate projection was accepted" >&2
    exit 1
fi
if "$out/phase" --null-carrier \
    >"$out/null-carrier.stdout" 2>"$out/null-carrier.stderr"; then
    echo "null carrier was accepted" >&2
    exit 1
fi
[[ ! -s "$out/project-intermediate.stdout" ]]
[[ ! -s "$out/null-carrier.stdout" ]]
rg -q '^active nonlinear wave projection denied$' \
    "$out/project-intermediate.stderr"
rg -q '^nonlinear wave carrier required$' "$out/null-carrier.stderr"

jq -e '
    .result == "PASS"
    and .claim
      == "BOUNDED_FIXED_RANK_NONLINEAR_KERR_INTERFERENCE_WAVE_CARRIER_WITH_ACTUAL_RESTORATION_AND_REUSE"
    and .tested_depths == [1, 4, 32, 128, 512, 2048]
    and (.boundary_hashes | length) == 6
    and .depth2048_kerr_phase_updates == 16384
    and .depth2048_interference_couplers == 8192
    and .maximum_norm_error <= 2e-12
    and .maximum_restoration_error <= .restoration_tolerance
    and ((.restoration_tolerance - 2e-10) | abs) <= 1e-25
    and .missing_inverse_residual > 1e-7
    and .wrong_inverse_residual > 1e-7
    and .reordered_inverse_residual > 1e-7
    and .snapshot_restoration_generation == 0
    and .reuse_transactions == 16
    and .reuse_restoration_generation == 16
    and .maximum_reuse_restoration_error <= .restoration_tolerance
    and .carrier_complex_cells == 4
    and .live_carrier_bytes == 64
    and .verification_snapshot_bytes == 64
    and .compiled_inverse_history_bytes == 0
    and .best_matched_compact_complex_state_bytes == 64
    and .independent_scalar_reference_parity_depth_max == 512
    and (.depth2048_cross_implementation_parity_established | not)
    and .amplitude_resident_in_wave_carrier
    and .host_amplitude_split_count == 0
    and .gauge_canonicalization_count == 0
    and .kerr_intensity_phase_coupling
    and .su2_interference_coupling
    and .fixed_rank_across_depth
    and .topology_derived_inverse_rematerialization
    and .projected_intermediate_complex_cells == 0
    and .serialized_intermediate_complex_cells == 0
    and .actual_inverse_restoration
    and .actual_restored_carrier_reuse
    and .snapshot_separated_from_actual_inverse
    and (.machine_enforced_hidden_intermediate | not)
    and .equivalent_compact_complex_recurrence
    and (.genuinely_distinct_phase_resource | not)
    and (.computational_advantage | not)
    and (.small_wall_crossed | not)
    and (.physical_execution | not)
    and (.unlimited_catalytic_computation | not)
    and (.terminal | not)
' "$out/phase.json" >/dev/null

jq -e '
    .result == "PASS"
    and (.boundary_hashes | length) == 5
    and .reference_parity_depth_max == 512
    and .state_bytes == 64
    and .time == "O_DEPTH"
' "$out/reference.json" >/dev/null

jq -e -n \
    --slurpfile phase "$out/phase.json" \
    --slurpfile reference "$out/reference.json" '
      $phase[0].boundary_hashes[0:5] == $reference[0].boundary_hashes
    ' >/dev/null

if rg -n \
    'acos|atan2|carg|pp_encode|canonical_split' \
    "$phase_source"; then
    echo "production wave path contains an amplitude chart or split" >&2
    exit 1
fi

stack_file=$(
    find "$out" -maxdepth 1 -name 'phase_stack-*.su' -print -quit
)
maximum_stack=$(
    awk -F '\t' '$2 > maximum {maximum=$2} END {print maximum+0}' \
        "$stack_file"
)
jq -n \
    --argjson maximum_stack "$maximum_stack" \
    --slurpfile phase "$out/phase.json" \
    --slurpfile reference "$out/reference.json" '
    {
      result: "PASS",
      claim: $phase[0].claim,
      claim_ceiling:
        "BOUNDED_FOUR_COMPLEX_CELL_NORMALIZED_KERR_SU2_WAVE_MESH_DEPTHS1_4_32_128_512_2048_DOUBLE_COMPLEX_SOFTWARE_REFERENCE_ONLY",
      accepted: $phase[0],
      matched_reference: $reference[0],
      maximum_individual_compiler_stack_frame_bytes: $maximum_stack,
      phase_owned_advance:
        "AMPLITUDE_REMAINS_RESIDENT_IN_A_NORMALIZED_WAVE_CARRIER_THROUGH_NONLINEAR_KERR_PHASE_AND_SU2_INTERFERENCE_WITHOUT_REENCODING",
      obstruction_removed:
        "HOST_AMPLITUDE_SPLIT_AND_MEMORYLESS_GAUGE_CANONICALIZATION",
      remaining_primary_obstruction:
        "EXACTLY_MATCHED_FOUR_COMPLEX_VALUE_CLASSICAL_RECURRENCE_AND_NO_MACHINE_ENFORCED_CUSTODY",
      depth2048_cross_implementation_boundary_parity: false,
      distinct_phase_resource: false,
      computational_advantage: false,
      small_wall_crossed: false,
      terminal: false
    }
' >"$out/summary.json"

sha256sum \
    "$phase_source" "$reference_source" \
    "$out/phase" "$out/reference" \
    "$out/phase.json" "$out/reference.json" "$out/summary.json" \
    >"$out/SHA256SUMS"
