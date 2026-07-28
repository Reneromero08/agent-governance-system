#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_algebraic_f5_gauss_kernel.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$1
mkdir -p "$out"
phase_source="$here/algebraic_f5_gauss_kernel_phase.c"
reference_source="$here/algebraic_f5_gauss_kernel_reference.c"
common=(
    -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror
    -Wconversion -Wshadow
)
for tool in gcc jq cmp rg sha256sum nice; do
    command -v "$tool" >/dev/null
done

gcc "${common[@]}" "$phase_source" -lm -o "$out/phase"
gcc "${common[@]}" "$reference_source" -lm -o "$out/reference"
gcc "${common[@]}" -fanalyzer "$phase_source" -lm \
    -o "$out/phase_analyzer"
gcc "${common[@]}" -fstack-usage "$phase_source" -lm \
    -o "$out/phase_stack"
gcc \
    -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror \
    -Wconversion -Wshadow -fsanitize=address,undefined \
    -fno-omit-frame-pointer -fno-sanitize-recover=all \
    "$phase_source" -lm -o "$out/phase_sanitized"

nice -n 10 "$out/phase" >"$out/phase.json"
nice -n 10 "$out/phase" >"$out/phase_replay.json"
cmp "$out/phase.json" "$out/phase_replay.json"
"$out/reference" >"$out/reference.json"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    nice -n 10 "$out/phase_sanitized" \
    >"$out/sanitized.json" 2>"$out/sanitized.stderr"
[[ ! -s "$out/sanitized.stderr" ]]
cmp <(jq -S . "$out/phase.json") <(jq -S . "$out/sanitized.json")

jq -e \
    --slurpfile reference "$out/reference.json" '
    .result == "PASS"
    and .claim
      == "BOUNDED_F5_QUADRATIC_PHASE_KERNEL_COHERENT_SHARED_PORT_GAUSS_CLOSURE_WITH_FIXED_TWO_BLOCK_CUSTODY_RESTORATION_AND_REUSE"
    and .tested_depths == [2,4,8,32,128,512,2048]
    and .tested_depths == $reference[0].tested_depths
    and .final_boundary_hashes == $reference[0].final_boundary_hashes
    and (
      [range(0;5) as $index |
        ((.deepest_probability[$index]
          - $reference[0].deepest_probability[$index]) | abs)
      ] | max
    ) <= 2e-11
    and .phase_kernel_side == 5
    and .hidden_phase_blocks == 2
    and .hidden_phase_cells == 50
    and .retained_inverse_kernel_cells == 0
    and .resident_phase_storage_bytes == 800
    and .hidden_baseline_storage_bytes == 800
    and .borrowed_verification_copy_bytes == 1600
    and .explicit_contraction_temporaries_bytes == 800
    and .native_phase_updates_at_depth2048 == 204850
    and .resident_phase_reads_at_depth2048 == 1024000
    and .coherent_path_terms_at_depth2048 == 1024000
    and .hidden_kernel_decodes == 0
    and .final_phase_decodes == 25
    and .maximum_raw_modulus_error <= 2e-11
    and .maximum_reconstruction_error <= 2e-10
    and .maximum_repeated_restoration_error <= 2e-10
    and .missing_inverse_residual >= 1e-6
    and .wrong_inverse_residual >= 1e-6
    and .reordered_inverse_residual >= 1e-6
    and .snapshot_residual <= 2e-10
    and .perturbed_raw_modulus_error > 2e-11
    and .generic_raw_modulus_error > 2e-11
    and .actual_coherent_shared_port_sum
    and .actual_adjoint_source_uncompute
    and .fixed_rank_across_tested_depth
    and .actual_inverse_restoration
    and .actual_restored_carrier_reuse
    and (.off_manifold_fixed_rank_closure | not)
    and .compact_weil_recurrence_exists
    and (.genuinely_distinct_phase_resource | not)
    and (.computational_advantage | not)
    and (.small_wall_crossed | not)
    and (.terminal | not)
    and $reference[0].compact_signature_state_bytes == 28
    and $reference[0].retained_intermediate_signatures == 0
    and $reference[0].hidden_kernel_rows_materialized == 0
' "$out/phase.json" >/dev/null

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
rg -q '^resident Gauss kernel projection denied$' \
    "$out/project-intermediate.stderr"
rg -q '^invalid Gauss-kernel transaction$' \
    "$out/null-carrier.stderr"

if rg -n \
    'witness|candidate_set|assignment_expansion|expected_(hash|boundary)|truth_table' \
    "$phase_source"
then
    echo "Gauss-kernel path contains forbidden extensional state" >&2
    exit 1
fi
if rg -n \
    '"(intermediate|resident)_(kernel|phase|signature)(s|_values|_coefficients|_entries)?":[[:space:]]*[\[{]' \
    "$out/phase.json" "$out/reference.json"
then
    echo "Gauss-kernel evidence serialized an intermediate" >&2
    exit 1
fi
rg -q 'gk_contract' "$phase_source"
rg -q 'gk_relative' "$phase_source"
rg -q 'gk_multiply' "$phase_source"
working_assignments=$(
    rg 'working\[[^]]+\][[:space:]]*[-+*/]?=' "$phase_source"
)
if [[ $(wc -l <<<"$working_assignments") -ne 2 ]] \
    || ! rg -q \
        'carrier\.working\[cell\] = carrier\.baseline\[cell\];' \
        <<<"$working_assignments" \
    || ! rg -q \
        'carrier->working\[cell\] = gk_unit\(' \
        <<<"$working_assignments"
then
    echo "Gauss-kernel path has an unauthorized working-cell assignment" >&2
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
        "BOUNDED_F5_NONDEGENERATE_QUADRATIC_PHASE_KERNEL_DEPTHS2_4_8_32_128_512_2048_DOUBLE_COMPLEX_SOFTWARE_REFERENCE_ONLY",
      accepted: $phase[0],
      compact_classical_reference: $reference[0],
      maximum_individual_compiler_stack_frame_bytes: $maximum_stack,
      resource_adjudication:
        "FIXED_50_HIDDEN_PHASE_CELLS_NO_RETAINED_INVERSE_KERNELS_BUT_EXACT_28_BYTE_WEIL_RECURRENCE",
      off_manifold_adjudication:
        "ONE_ENTRY_AND_GENERIC_PERTURBATIONS_FAIL_UNIT_MODULUS_PHASE_ONLY_CLOSURE",
      remaining_primary_obstruction:
        "EXACT_COMPACT_SEVEN_INTEGER_WEIL_RECURRENCE_AND_NO_OFF_MANIFOLD_PHASE_ONLY_CLOSURE",
      catvm_enforced: false,
      arbitrary_kernel_closure: false,
      genuinely_distinct_phase_resource: false,
      computational_advantage: false,
      small_wall_crossed: false,
      physical_execution: false,
      unlimited_catalytic_computation: false,
      terminal: false
    }
' >"$out/summary.json"

sha256sum \
    "$phase_source" "$reference_source" \
    "$out/phase" "$out/reference" \
    "$out/phase.json" "$out/reference.json" "$out/summary.json" \
    >"$out/SHA256SUMS"
