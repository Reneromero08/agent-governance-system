#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_algebraic_stokes_harmonic_sector.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
out=$1
mkdir -p "$out"
phase_source="$here/algebraic_stokes_harmonic_sector_phase.py"
reference_source="$here/algebraic_stokes_harmonic_sector_reference.py"
phase_base="$here/algebraic_stokes_lie_relation_phase.py"
reference_base="$here/algebraic_stokes_lie_relation_reference.py"
for tool in jq cmp rg sha256sum nice; do
    command -v "$tool" >/dev/null
done
[[ -x "$python" ]]

"$python" -m py_compile \
    "$phase_source" "$reference_source" "$phase_base" "$reference_base"
nice -n 10 "$python" -X dev "$phase_source" \
    >"$out/phase.json" 2>"$out/phase.stderr"
nice -n 10 "$python" -X dev "$phase_source" \
    >"$out/phase_replay.json" 2>"$out/replay.stderr"
[[ ! -s "$out/phase.stderr" ]]
[[ ! -s "$out/replay.stderr" ]]
cmp "$out/phase.json" "$out/phase_replay.json"
"$python" -X dev "$reference_source" \
    >"$out/reference.json" 2>"$out/reference.stderr"
[[ ! -s "$out/reference.stderr" ]]

jq -e \
    --slurpfile reference "$out/reference.json" '
    .result == "PASS"
    and .claim
      == "BOUNDED_PARITY_ADMISSIBLE_STOKES_HARMONIC_SECTOR_DUAL_PRIME_PHASE_SIGNATURE_REDUCTION_WITH_RESTORATION_AND_REUSE"
    and .port_type == "NORMALIZED_TWO_MODE_STOKES_SPHERE_WAVE_PORT"
    and .grades == 5
    and .maximum_degree == 6
    and .public_degree_parity_basis_compilation
    and .parity_admissible_basis_cells == 80
    and .resident_dual_prime_coefficient_cells == 160
    and .logical_packed_phase_payload_bytes == 2560
    and (.actual_python_object_allocation_measured | not)
    and .retained_inverse_kernels == 0
    and .hidden_coefficient_decodes == 0
    and .final_coefficient_decodes == 160
    and .maximum_unit_modulus_error <= 5e-12
    and .maximum_final_root_error <= 1e-3
    and .restoration_max_abs <= 2e-10
    and .maximum_reuse_restoration_error <= 2e-10
    and .successful_restoration_receipt == 1
    and .same_carrier_reuse_transactions == 8
    and .missing_inverse_residual >= 1e-6
    and .wrong_inverse_residual >= 1e-6
    and .reordered_inverse_residual >= 1e-6
    and .missing_inverse_modular_mismatch_cells > 0
    and .wrong_inverse_modular_mismatch_cells > 0
    and .reordered_inverse_modular_mismatch_cells > 0
    and .snapshot_residual <= 2e-10
    and .snapshot_restoration_receipt == 0
    and .identity_mixer_higher_grades_zero
    and .zero_kerr_all_grades_zero
    and .swapped_order_discriminated
    and .deterministic_replay
    and .actual_inverse_restoration
    and .actual_restored_carrier_reuse
    and .final_boundary_only_projection
    and .actual_resident_grade_consumed
    and .public_topology_inverse_rematerialization
    and .exact_norm_casimir_quotient_reduction
    and .mathematically_exact_dual_prime_phase_algebra
    and .exact_rank_reduction_established
    and (.irreducible_harmonic_decomposition_established | not)
    and (.fixed_rank_closure_established | not)
    and (.unbounded_growth_proved | not)
    and (.genuinely_distinct_phase_resource | not)
    and (.computational_advantage | not)
    and (.small_wall_crossed | not)
    and (.physical_waveform_execution | not)
    and (.terminal | not)
    and $reference[0].result == "PASS"
    and $reference[0].sphere_relation == "x^2+y^2+z^2=1"
    and $reference[0].homogeneous_harmonic_quotient_relation
      == "x^2+y^2+z^2=0"
    and $reference[0].successive_primary_term_counts == [3,4,9,9,16]
    and $reference[0].successive_parity_basis_cells == [6,10,15,21,28]
    and $reference[0].all_rational_support_parity_exact
    and $reference[0].all_leading_harmonic_quotients_nonzero
    and (
      [$reference[0].leading_harmonic_certificates[].degree]
      == [2,3,4,5,6]
    )
    and $reference[0].identity_mixer_higher_grades_zero
    and $reference[0].swapped_first_bracket_negates_primary
    and $reference[0].stokes_full_quotient_basis_cells == 135
    and $reference[0].parity_admissible_basis_cells == 80
    and $reference[0].compact_direct_wave_semantic_state_bytes == 64
    and $reference[0].exact_parity_rank_reduction_established
    and (
      $reference[0].irreducible_harmonic_decomposition_established
      | not
    )
    and ($reference[0].fixed_rank_reduction_found | not)
    and $reference[0].bounded_highest_harmonic_shell_growth_observed
    and ($reference[0].unbounded_growth_proved | not)
    and (
      [.primary_grades[]
        | {degree_limit, quotient_basis_cells, nonzero_p17,
           nonzero_p19, hash_p17, hash_p19}]
      == [$reference[0].primary_grades[]
        | {degree_limit, quotient_basis_cells, nonzero_p17,
           nonzero_p19, hash_p17, hash_p19}]
    )
    and (
      [.reuse_grades[]
        | {degree_limit, quotient_basis_cells, nonzero_p17,
           nonzero_p19, hash_p17, hash_p19}]
      == [$reference[0].reuse_grades[]
        | {degree_limit, quotient_basis_cells, nonzero_p17,
           nonzero_p19, hash_p17, hash_p19}]
    )
  ' "$out/phase.json" >/dev/null

for denied in project-shared-port null-carrier; do
    set +e
    "$python" "$phase_source" "--${denied}" \
        >"$out/${denied}.stdout" \
        2>"$out/${denied}.stderr"
    rc=$?
    set -e
    [[ "$rc" -eq 2 ]]
    [[ ! -s "$out/${denied}.stdout" ]]
done
rg -q '^resident Stokes harmonic shared wave port projection denied$' \
    "$out/project-shared-port.stderr"
rg -q '^invalid Stokes harmonic sector carrier$' \
    "$out/null-carrier.stderr"

if rg -n \
    'witness_list|candidate_set|truth_table|assignment_expansion|shared_port_coordinates|expected_(hash|boundary)' \
    "$phase_source" "$reference_source" "$phase_base" "$reference_base"
then
    echo "Stokes harmonic path contains forbidden extensional state" >&2
    exit 1
fi
rg -q 'parity_reduced' "$phase_base"
rg -q 'HOMOGENEOUS_SPHERE' "$reference_source"

jq -n \
    --slurpfile phase "$out/phase.json" \
    --slurpfile reference "$out/reference.json" '
    {
      result: "PASS",
      claim: $phase[0].claim,
      claim_ceiling:
        "BOUNDED_NORMALIZED_TWO_MODE_PARITY_ADMISSIBLE_STOKES_HARMONIC_DUAL_PRIME_LIE_GRADES2_3_4_5_6_SOFTWARE_REFERENCE_ONLY",
      accepted: $phase[0],
      exact_rational_reference: $reference[0],
      phase_owned_repair:
        "PUBLIC_DEGREE_PARITY_COMPILED_CLOSED_STOKES_HARMONIC_SECTOR",
      previous_stokes_basis_cells: 135,
      parity_admissible_basis_cells: 80,
      basis_reduction_ratio: (135 / 80),
      previous_dual_prime_phase_bytes: 4320,
      reduced_dual_prime_phase_bytes: 2560,
      bounded_leading_harmonic_shells_nonzero: true,
      remaining_primary_obstruction:
        "NONZERO_IRREDUCIBLE_HARMONIC_SHELL_GROWTH_WITH_DEGREE",
      matched_compact_direct_wave_semantic_state_bytes: 64,
      irreducible_harmonic_decomposition_established: false,
      fixed_rank_closure_established: false,
      unbounded_growth_proved: false,
      genuinely_distinct_phase_resource: false,
      computational_advantage: false,
      small_wall_crossed: false,
      physical_waveform_execution: false,
      terminal: false
    }
  ' >"$out/summary.json"

sha256sum \
    "$phase_source" "$reference_source" "$phase_base" "$reference_base" \
    "$out/phase.json" "$out/reference.json" "$out/summary.json" \
    >"$out/SHA256SUMS"
