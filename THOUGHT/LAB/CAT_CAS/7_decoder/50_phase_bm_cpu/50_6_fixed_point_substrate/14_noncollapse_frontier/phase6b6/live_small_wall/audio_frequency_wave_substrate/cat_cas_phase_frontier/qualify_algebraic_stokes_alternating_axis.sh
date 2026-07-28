#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_algebraic_stokes_alternating_axis.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
out=$1
mkdir -p "$out"
phase_source="$here/algebraic_stokes_alternating_axis_phase.py"
reference_source="$here/algebraic_stokes_alternating_axis_reference.py"
verifier_source="$here/algebraic_stokes_alternating_axis_exact_verifier.py"
phase_base="$here/algebraic_stokes_lie_relation_phase.py"
reference_base="$here/algebraic_stokes_lie_relation_reference.py"
for tool in jq cmp rg sha256sum nice; do
    command -v "$tool" >/dev/null
done
[[ -x "$python" ]]

"$python" -m py_compile \
    "$phase_source" "$reference_source" "$verifier_source" \
    "$phase_base" "$reference_base"
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
env PYTHONPATH="$here" "$python" -X dev "$verifier_source" \
    >"$out/exact_verifier.json" 2>"$out/exact_verifier.stderr"
[[ ! -s "$out/exact_verifier.stderr" ]]

jq -e \
    --slurpfile reference "$out/reference.json" '
    .result == "PASS"
    and .claim
      == "BOUNDED_NONCOMMUTING_ALTERNATING_AXIS_STOKES_CHARACTER_PHASE_HARMONIC_CATALECTICANT_RANK_GROWTH_WITH_RESTORATION_AND_REUSE"
    and .generator_schedule
      == "TILTED_AXIS_SQUARED_THEN_Z_AXIS_SQUARED_ALTERNATING"
    and .declared_boundary == "FULL_13_GRADE_COEFFICIENT_JET_SUMMARIES"
    and .all_projected_grades_are_declared_final_jet
    and .grades == 13
    and .maximum_degree == 14
    and .parity_admissible_basis_cells == 676
    and .resident_character_phase_cells == 24336
    and .logical_packed_phase_payload_bytes == 389376
    and (.actual_python_object_allocation_measured | not)
    and .retained_inverse_history_bytes == 0
    and .complete_unit_phase_character_orbit_encoding
    and .coefficient_addition_is_native_phase_multiplication
    and .public_scalar_action_is_phase_index_permutation
    and .native_phase_updates > 0
    and .resident_phase_reads > 0
    and .zero_scalar_contributions_skipped > 0
    and .nonzero_scalar_phase_permutations > 0
    and .hidden_coefficient_decodes == 0
    and .final_coefficient_decodes == 1352
    and .maximum_unit_modulus_error <= 5e-12
    and .maximum_final_root_error <= 1e-3
    and .restoration_max_abs <= 2e-10
    and .maximum_reuse_restoration_error <= 2e-10
    and .successful_restoration_receipt == 1
    and .same_carrier_transactions == 8
    and .missing_inverse_residual >= 1e-6
    and .wrong_inverse_residual >= 1e-6
    and .reordered_inverse_residual >= 1e-6
    and .missing_inverse_modular_mismatch_cells > 0
    and .wrong_inverse_modular_mismatch_cells > 0
    and .reordered_inverse_modular_mismatch_cells > 0
    and .snapshot_residual <= 2e-10
    and .snapshot_restoration_receipt == 0
    and .deterministic_replay
    and .actual_inverse_restoration
    and .actual_restored_carrier_reuse
    and .final_boundary_only_projection
    and .actual_resident_prior_grade_consumed
    and .noncommuting_generator_order_material
    and .exact_rational_oracle_required
    and (.fixed_rank_closure_established | not)
    and (.unbounded_rank_growth_proved | not)
    and .same_output_dual_prime_classical_semantic_state_bytes == 1352
    and (.same_output_classical_actual_allocation_measured | not)
    and .compact_point_evaluation_semantic_state_bytes == 64
    and (.point_evaluation_has_same_boundary_semantics | not)
    and (.genuinely_distinct_phase_resource | not)
    and (.computational_advantage | not)
    and (.small_wall_crossed | not)
    and (.physical_waveform_execution | not)
    and (.terminal | not)
    and $reference[0].result == "PASS"
    and $reference[0].generator_schedule == .generator_schedule
    and $reference[0].middle_catalecticant_ranks
      == [3,3,5,5,7,7,9,9,11,11,13,13,15]
    and $reference[0].strict_rank_increases_observed == 6
    and $reference[0].maximum_middle_catalecticant_rank == 15
    and $reference[0].rank_computed_after_unique_harmonic_projection
    and $reference[0].catalecticant_is_separable_waring_rank_lower_bound
    and $reference[0].bounded_separable_factor_rank_growth_established
    and ($reference[0].arbitrary_compact_representation_lower_bound | not)
    and ($reference[0].unbounded_rank_growth_proved | not)
    and $reference[0].parity_support_exact
    and $reference[0].same_output_dual_prime_classical_semantic_state_bytes
      == 1352
    and ($reference[0].same_output_classical_actual_allocation_measured | not)
    and $reference[0].compact_point_evaluation_semantic_state_bytes == 64
    and ($reference[0].point_evaluation_has_same_boundary_semantics | not)
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

jq -e '
    .result == "PASS"
    and .exact_per_cell_dual_prime_comparison
    and (.coefficient_values_emitted | not)
    and .primary_comparisons == 1352
    and .reuse_comparisons == 1352
    and .total_comparisons == 2704
    and .maximum_verifier_restoration_error <= 2e-10
  ' "$out/exact_verifier.json" >/dev/null

for denied in project-internal-grade null-carrier; do
    set +e
    env PYTHONPATH="$here" "$python" "$phase_source" "--${denied}" \
        >"$out/${denied}.stdout" 2>"$out/${denied}.stderr"
    rc=$?
    set -e
    [[ "$rc" -eq 2 ]]
    [[ ! -s "$out/${denied}.stdout" ]]
done
rg -q '^resident alternating-axis grade projection denied$' \
    "$out/project-internal-grade.stderr"
rg -q '^invalid alternating-axis phase carrier$' \
    "$out/null-carrier.stderr"

if rg -n \
    'witness_list|candidate_set|truth_table|assignment_expansion|expected_(hash|boundary)|residue_shadow' \
    "$phase_source" "$reference_source"
then
    echo "alternating-axis path contains forbidden extensional state" >&2
    exit 1
fi
rg -q 'source_character' "$phase_source"
rg -q 'harmonic_projection' "$reference_source"
rg -q 'middle_catalecticant_rank' "$reference_source"

jq -n \
    --slurpfile phase "$out/phase.json" \
    --slurpfile reference "$out/reference.json" '
    {
      result: "PASS",
      claim: $phase[0].claim,
      claim_ceiling:
        "BOUNDED_NORMALIZED_TWO_MODE_RATIONAL_ALTERNATING_NONCOMMUTING_QUADRATIC_STOKES_AXES_DUAL_PRIME_CHARACTER_PHASE_GRADES2_TO14_SOFTWARE_REFERENCE_ONLY",
      accepted: $phase[0],
      exact_rational_reference: $reference[0],
      phase_owned_repair:
        "GENERAL_SPARSE_LIE_COEFFICIENT_ADDITION_BY_REVERSIBLE_CHARACTER_PHASE_MULTIPLICATION",
      harmonic_catalecticant_ranks:
        $reference[0].middle_catalecticant_ranks,
      maximum_harmonic_catalecticant_rank: 15,
      bounded_separable_factor_rank_growth: true,
      arbitrary_compact_representation_lower_bound: false,
      unbounded_rank_growth_proved: false,
      same_output_dual_prime_classical_semantic_state_bytes: 1352,
      same_output_classical_actual_allocation_measured: false,
      different_output_point_evaluation_semantic_state_bytes: 64,
      remaining_primary_obstruction:
        "NONCOMMUTING_PHASE_RELATION_NEEDS_COMPACT_NONSEPARABLE_CLOSURE_OR_DISTINCT_RESOURCE",
      fixed_rank_closure_established: false,
      genuinely_distinct_phase_resource: false,
      computational_advantage: false,
      small_wall_crossed: false,
      physical_waveform_execution: false,
      terminal: false
    }
  ' >"$out/summary.json"

sha256sum \
    "$phase_source" "$reference_source" "$verifier_source" \
    "$phase_base" "$reference_base" \
    "$out/phase.json" "$out/reference.json" \
    "$out/exact_verifier.json" "$out/summary.json" \
    >"$out/SHA256SUMS"
