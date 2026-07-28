#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_algebraic_f5_conic_relation.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$1
mkdir -p "$out"
phase_source="$here/algebraic_f5_conic_relation_phase.c"
reference_source="$here/algebraic_f5_conic_relation_reference.c"
common=(
    -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror
    -Wconversion -Wshadow
)
for tool in gcc jq cmp rg sha256sum nice; do
    command -v "$tool" >/dev/null
done

gcc "${common[@]}" "$phase_source" -lm -o "$out/phase"
gcc "${common[@]}" "$reference_source" -o "$out/reference"
gcc "${common[@]}" -fanalyzer "$phase_source" -lm \
    -o "$out/phase_analyzer"
gcc "${common[@]}" -fstack-usage "$phase_source" -lm \
    -o "$out/phase_stack"
gcc \
    -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror \
    -Wconversion -Wshadow -fsanitize=address,undefined \
    -fno-omit-frame-pointer -fno-sanitize-recover=all \
    "$phase_source" -lm -o "$out/phase_sanitized"

"$out/phase" >"$out/phase.json"
"$out/phase" >"$out/phase_replay.json"
cmp "$out/phase.json" "$out/phase_replay.json"
"$out/reference" >"$out/reference.json"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$out/phase_sanitized" \
    >"$out/sanitized.json" 2>"$out/sanitized.stderr"
[[ ! -s "$out/sanitized.stderr" ]]
cmp <(jq -S . "$out/phase.json") <(jq -S . "$out/sanitized.json")

jq -e \
    --slurpfile reference "$out/reference.json" '
    .result == "PASS"
    and .claim
      == "BOUNDED_F5_MONIC_AFFINE_CONIC_TWO_HIDDEN_PORT_FIXED_RANK_PHASE_CLOSURE_WITH_RESTORATION_AND_REUSE"
    and .port_type == "FULL_F5"
    and .closed_hidden_ports == 2
    and .input_signature_cells == 10
    and .resident_intermediate_cells == 6
    and .resident_final_cells == 6
    and .carrier_cells == 22
    and .resident_phase_storage_bytes == 352
    and .hidden_baseline_storage_bytes == 352
    and .borrowed_verification_copy_bytes == 704
    and .phase_factor_temporary_bytes == 192
    and .projected_boundary_bytes >= 24
    and .signature_rank_after_each_closure == 6
    and .primary_boundary_fnv1a64
      == $reference[0].primary_boundary_fnv1a64
    and .reuse_boundary_fnv1a64
      == $reference[0].reuse_boundary_fnv1a64
    and .primary_boundary_coefficients
      == $reference[0].primary_boundary_coefficients
    and $reference[0].primary_boundary_zero_set_size == 9
    and ($reference[0].affine_f5_subset_cardinality | not)
    and $reference[0].compact_classical_live_state_bytes == 64
    and $reference[0].retain_all_classical_state_bytes == 88
    and ($reference[0] | has("primary_intermediate_coefficients") | not)
    and .hidden_coefficient_decodes == 0
    and .final_coefficient_decodes == 6
    and .maximum_unit_modulus_error <= 2e-11
    and .maximum_final_root_error <= 2e-9
    and .maximum_repeated_restoration_error <= 2e-11
    and .missing_inverse_residual >= 1e-6
    and .wrong_inverse_residual >= 1e-6
    and .reordered_inverse_residual >= 1e-6
    and .snapshot_residual <= 2e-11
    and .actual_resident_intermediate_consumed
    and .exact_monic_substitution_closure
    and .non_affine_primary_xz_boundary
    and .fixed_rank_two_closure_chain
    and .actual_inverse_restoration
    and .actual_restored_carrier_reuse
    and .snapshot_loaded_separately
    and .intermediate_bypass_changes_boundary
    and (.general_conic_conic_composition | not)
    and (.canonical_zero_set_normalization | not)
    and .compact_classical_coefficient_recurrence_exists
    and (.genuinely_distinct_phase_resource | not)
    and (.computational_advantage | not)
    and (.small_wall_crossed | not)
    and (.terminal | not)
' "$out/phase.json" >/dev/null
if rg -n \
    'intermediate_coefficients|resident_intermediate_coefficients' \
    "$out/phase.json" "$out/reference.json"
then
    echo "F5 conic evidence serialized an intermediate relation" >&2
    exit 1
fi

for denied in project-intermediate null-carrier; do
    set +e
    "$out/phase" "--${denied}" \
        >"$out/${denied}.stdout" \
        2>"$out/${denied}.stderr"
    rc=$?
    set -e
    [[ "$rc" -eq 2 ]]
    [[ ! -s "$out/${denied}.stdout" ]]
done
rg -q '^F5 resident conic projection denied$' \
    "$out/project-intermediate.stderr"
rg -q '^invalid F5 conic carrier$' "$out/null-carrier.stderr"

if rg -n \
    'witness|candidate_set|assignment_expansion|expected_(hash|boundary)|truth_table' \
    "$phase_source"
then
    echo "F5 conic accepted path contains forbidden extensional state" >&2
    exit 1
fi
rg -q 'fc_field_product' "$phase_source"
rg -q 'fc_relative' "$phase_source"
rg -q 'fc_multiply' "$phase_source"
working_assignments=$(
    rg 'working\[[^]]+\][[:space:]]*[-+*/]?=' "$phase_source"
)
if [[ $(wc -l <<<"$working_assignments") -ne 2 ]] \
    || ! rg -q \
        'carrier\.working\[cell\] = carrier\.baseline\[cell\];' \
        <<<"$working_assignments" \
    || ! rg -q \
        'carrier->working\[cell\] = fc_unit\(' \
        <<<"$working_assignments"
then
    echo "F5 conic path has an unauthorized working-cell assignment" >&2
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
        "BOUNDED_FULL_F5_MONIC_AFFINE_PORT_MAPS_TOTAL_DEGREE2_CONIC_TWO_HIDDEN_PORTS_FIXED_FIXTURES_SOFTWARE_REFERENCE_ONLY",
      accepted: $phase[0],
      compact_classical_reference: $reference[0],
      maximum_compiler_stack_frame_bytes: $maximum_stack,
      phase_owned_advance:
        "FIXED_RANK_SIX_NONAFFINE_OPEN_PHASE_RELATION_CLOSURE",
      exactness_law:
        "FULL_F5_PORTS_AND_MONIC_AFFINE_PINNING_ONLY",
      remaining_primary_obstruction:
        "EQUIVALENT_COMPACT_MODULAR_COEFFICIENT_SUBSTITUTION",
      general_conic_conic_composition: false,
      genuinely_distinct_phase_resource: false,
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
