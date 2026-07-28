#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_algebraic_stokes_factorized_shell.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
out=$1
mkdir -p "$out"
phase_source="$here/algebraic_stokes_factorized_shell_phase.py"
reference_source="$here/algebraic_stokes_factorized_shell_reference.py"
stokes_phase="$here/algebraic_stokes_lie_relation_phase.py"
stokes_reference="$here/algebraic_stokes_lie_relation_reference.py"
for tool in jq cmp rg sha256sum nice; do
    command -v "$tool" >/dev/null
done
[[ -x "$python" ]]

"$python" -m py_compile \
    "$phase_source" "$reference_source" \
    "$stokes_phase" "$stokes_reference"
nice -n 10 env PYTHONPATH="$here" "$python" -X dev "$phase_source" \
    >"$out/phase.json" 2>"$out/phase.stderr"
nice -n 10 env PYTHONPATH="$here" "$python" -X dev "$phase_source" \
    >"$out/phase_replay.json" 2>"$out/replay.stderr"
[[ ! -s "$out/phase.stderr" ]]
[[ ! -s "$out/replay.stderr" ]]
cmp "$out/phase.json" "$out/phase_replay.json"
env PYTHONPATH="$here" "$python" -X dev "$reference_source" \
    >"$out/reference.json" 2>"$out/reference.stderr"
[[ ! -s "$out/reference.stderr" ]]

jq -e \
    --slurpfile reference "$out/reference.json" '
    .result == "PASS"
    and .claim
      == "FIXED_RANK_FACTORIZED_HIGHEST_STOKES_HARMONIC_SHELL_PHASE_RECURRENCE_WITH_RESTORATION_AND_REUSE"
    and .tested_depths == [1,2,4,8,32,128,512,2048]
    and .public_factorized_signature == "L_AXIS_AND_EXPONENT_TIMES_Q4"
    and .public_factor_axis == [24,0,-7]
    and .maximum_public_factor_exponent == 2048
    and .resident_quadratic_coordinates == 4
    and .resident_dual_prime_phase_cells == 144
    and .logical_packed_phase_payload_bytes == 2304
    and .compiled_public_recurrence_logical_packed_bytes == 96
    and (.actual_python_object_allocation_measured | not)
    and .retained_inverse_history_bytes == 0
    and .maximum_hidden_phase_cells_independent_of_depth
    and .native_phase_permutations_all_primary_depths > 0
    and .hidden_coefficient_decodes == 0
    and .final_coefficient_decodes_at_maximum_depth == 8
    and .maximum_unit_modulus_error <= 5e-12
    and .maximum_final_root_error <= 1e-3
    and .restoration_max_abs_at_maximum_depth <= 2e-10
    and .maximum_same_carrier_restoration_error <= 2e-10
    and .successful_restoration_receipt == 1
    and .same_carrier_transactions == 16
    and .missing_inverse_residual >= 1e-6
    and .wrong_inverse_residual >= 1e-6
    and .missing_inverse_modular_mismatch_cells > 0
    and .wrong_inverse_modular_mismatch_cells > 0
    and .snapshot_residual <= 2e-10
    and .snapshot_restoration_receipt == 0
    and (.reordered_inverse_applicable | not)
    and .reordered_inverse_reason
      == "ALL_REPEATED_FIXED_AXIS_TRANSITIONS_ARE_IDENTICAL"
    and .deterministic_replay
    and .actual_inverse_restoration
    and .actual_restored_carrier_reuse
    and .reuse_seed_is_second_public_factorized_q4_program
    and .final_boundary_only_projection
    and .actual_resident_quadratic_consumed
    and .exact_public_factor_not_expanded
    and .factor_exponent_storage_grows_logarithmically
    and .complete_unit_phase_character_orbit_encoding
    and .coefficient_scaling_is_exact_phase_index_permutation
    and .exact_rational_oracle_required
    and .highest_shell_fixed_rank_closure_established
    and .software_execution_bounded_to_tested_depths
    and (.full_stokes_signature_fixed_rank_closure_established | not)
    and .best_matched_dual_prime_classical_residue_state_bytes == 8
    and .matched_classical_recurrence_exists
    and (.genuinely_distinct_phase_resource | not)
    and (.computational_advantage | not)
    and (.small_wall_crossed | not)
    and (.unbounded_catalytic_computation | not)
    and (.physical_waveform_execution | not)
    and (.terminal | not)
    and $reference[0].result == "PASS"
    and $reference[0].all_positive_depth_factorization_law_proved
    and $reference[0].all_symbolic_sphere_quotient_identities
    and (
      [$reference[0].symbolic_factorization_checks[].depth]
      == [1,2,3,4,5,6]
    )
    and $reference[0].tested_depths == .tested_depths
    and $reference[0].maximum_expanded_highest_shell_dimension == 4101
    and $reference[0].factorized_resident_coordinates == 4
    and $reference[0].public_factor_descriptor_does_not_expand
    and $reference[0].exact_fixed_rank_highest_shell_recurrence
    and ($reference[0].full_stokes_signature_fixed_rank_closure | not)
    and (
      .primary_boundaries == $reference[0].primary_boundaries
    )
    and .reuse_boundary == $reference[0].reuse_boundary
  ' "$out/phase.json" >/dev/null

for denied in project-resident-quadratic null-carrier; do
    set +e
    env PYTHONPATH="$here" "$python" "$phase_source" "--${denied}" \
        >"$out/${denied}.stdout" \
        2>"$out/${denied}.stderr"
    rc=$?
    set -e
    [[ "$rc" -eq 2 ]]
    [[ ! -s "$out/${denied}.stdout" ]]
done
rg -q '^resident factorized shell projection denied$' \
    "$out/project-resident-quadratic.stderr"
rg -q '^invalid factorized shell carrier$' \
    "$out/null-carrier.stderr"

if rg -n \
    'witness_list|candidate_set|truth_table|assignment_expansion|expected_(hash|boundary)|residue_shadow' \
    "$phase_source" "$reference_source"
then
    echo "factorized shell path contains forbidden extensional state" >&2
    exit 1
fi
rg -q 'source = carrier.working' "$phase_source"
rg -q 'scalar \* harmonic' "$phase_source"
rg -q 'AD_L_SQUARED_EQUALS_2_L_TIMES_AD_L' "$reference_source"

jq -n \
    --slurpfile phase "$out/phase.json" \
    --slurpfile reference "$out/reference.json" '
    {
      result: "PASS",
      claim: $phase[0].claim,
      claim_ceiling:
        "EXACT_REPEATED_SINGLE_AXIS_QUADRATIC_STOKES_KERR_HIGHEST_HOMOGENEOUS_SHELL_FACTOR_L_POWER_N_TIMES_Q4_DUAL_PRIME_SOFTWARE_DEPTHS1_TO_2048",
      accepted: $phase[0],
      exact_rational_reference: $reference[0],
      phase_owned_repair:
        "FIXED_RANK_FACTORIZED_HIGHEST_SHELL_WITH_REVERSIBLE_CHARACTER_ORBIT_PHASE_PERMUTATIONS",
      maximum_expanded_highest_shell_dimension: 4101,
      fixed_resident_phase_cells: 144,
      fixed_logical_phase_payload_bytes: 2304,
      compiled_public_recurrence_logical_packed_bytes: 96,
      matched_classical_residue_state_bytes: 8,
      algebraic_all_positive_depth_factorization_law: true,
      bounded_software_maximum_depth: 2048,
      remaining_primary_obstruction:
        "FULL_MULTI_SHELL_STOKES_SIGNATURE_CLOSURE_AND_MATCHED_CLASSICAL_EQUIVALENCE",
      full_stokes_signature_fixed_rank_closure_established: false,
      genuinely_distinct_phase_resource: false,
      computational_advantage: false,
      small_wall_crossed: false,
      unbounded_catalytic_computation: false,
      physical_waveform_execution: false,
      terminal: false
    }
  ' >"$out/summary.json"

sha256sum \
    "$phase_source" "$reference_source" \
    "$stokes_phase" "$stokes_reference" \
    "$out/phase.json" "$out/reference.json" "$out/summary.json" \
    >"$out/SHA256SUMS"
