#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_algebraic_stokes_bch_rematerialized.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
out=$1
mkdir -p "$out"
phase="$here/algebraic_stokes_bch_rematerialized_phase.py"
reference="$here/algebraic_stokes_bch_signature_reference.py"
verifier="$here/algebraic_stokes_bch_exact_verifier.py"

for tool in jq cmp rg sha256sum nice; do
    command -v "$tool" >/dev/null
done
[[ -x "$python" ]]
"$python" -m py_compile "$phase" "$reference" "$verifier"

nice -n 10 env PYTHONPATH="$here" "$python" -X dev "$phase" \
    >"$out/phase.json" 2>"$out/phase.stderr"
nice -n 10 env PYTHONPATH="$here" "$python" -X dev "$phase" \
    >"$out/phase_replay.json" 2>"$out/replay.stderr"
cmp "$out/phase.json" "$out/phase_replay.json"
[[ ! -s "$out/phase.stderr" ]]
[[ ! -s "$out/replay.stderr" ]]

nice -n 10 env PYTHONPATH="$here" "$python" -X dev "$reference" \
    >"$out/reference.json" 2>"$out/reference.stderr"
nice -n 10 env PYTHONPATH="$here" "$python" -X dev "$verifier" \
    >"$out/verifier.json" 2>"$out/verifier.stderr"
[[ ! -s "$out/reference.stderr" ]]
[[ ! -s "$out/verifier.stderr" ]]

jq -e --slurpfile reference "$out/reference.json" '
  .result == "PASS"
  and .claim == "BOUNDED_TOPOLOGY_REMATERIALIZED_NONCOMMUTING_STOKES_BCH_CHARACTER_PHASE_CLOSURE_WITH_RESTORATION_AND_REUSE"
  and .maximum_word_grade == 6
  and .maximum_polynomial_degree == 7
  and .declared_boundary == "FULL_GRADE1_TO6_BCH_COEFFICIENT_SIGNATURE"
  and .compiled_nonzero_lie_words == 72
  and .compiled_public_word_letter_bytes > 0
  and .compiled_public_coefficient_logical_bytes == 1152
  and (.compiled_public_topology_actual_allocation_measured | not)
  and .maximum_live_scratch_blocks == 6
  and .retained_word_history_blocks == 0
  and .final_coefficient_cells == 116
  and .reusable_scratch_coefficient_cells == 116
  and .resident_character_phase_cells == 8352
  and .logical_packed_phase_payload_bytes == 133632
  and .same_output_dual_prime_classical_semantic_state_bytes == 232
  and (.same_output_classical_actual_allocation_measured | not)
  and .native_phase_updates > 0
  and .resident_phase_reads > 0
  and .hidden_decodes == 0
  and .final_decodes == 232
  and .maximum_final_root_error <= 1e-3
  and .restoration_max_abs <= 2e-10
  and .reuse_restoration_max_abs <= 2e-10
  and .missing_inverse_residual >= 1e-6
  and .wrong_inverse_residual >= 1e-6
  and .swapped_order_boundary_differs
  and (.reordered_inverse_applicable | not)
  and .snapshot_creation_traffic_bytes == 133632
  and .snapshot_reload_traffic_bytes == 133632
  and .snapshot_sham_loaded
  and .snapshot_sham_residual == 0
  and .snapshot_sham_reuse_boundary_matches
  and .snapshot_sham_reuse_residual <= 2e-10
  and .snapshot_restoration_receipt == 0
  and .actual_inverse_restoration
  and .actual_restored_carrier_reuse
  and .topology_derived_rematerialization
  and .final_boundary_only_projection
  and (.snapshot_loaded | not)
  and (.fixed_rank_nonseparable_closure_established | not)
  and (.genuinely_distinct_phase_resource | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.physical_waveform_execution | not)
  and (.terminal | not)
  and $reference[0].result == "PASS"
  and $reference[0].maximum_word_grade == 10
  and $reference[0].maximum_polynomial_degree == 11
  and $reference[0].middle_catalecticant_ranks
    == [3,3,5,5,7,7,9,9,11,11]
  and $reference[0].commuting_higher_components_zero
  and $reference[0].swapped_order_material
  and ($reference[0].fixed_rank_nonseparable_closure_established | not)
  and ($reference[0].arbitrary_compact_representation_lower_bound | not)
  and ($reference[0].unbounded_rank_growth_proved | not)
  and (
    [.primary_components[]
      | {polynomial_degree,quotient_basis_cells,nonzero_p17,
         nonzero_p19,hash_p17,hash_p19}]
    == [$reference[0].primary_components[:6][]
      | {polynomial_degree,quotient_basis_cells,nonzero_p17,
         nonzero_p19,hash_p17,hash_p19}]
  )
' "$out/phase.json" >/dev/null

jq -e '
  .result == "PASS"
  and .exact_per_cell_dual_prime_comparison
  and (.coefficient_values_emitted | not)
  and .primary_comparisons == 232
  and .reuse_comparisons == 232
  and .total_comparisons == 464
  and .maximum_verifier_restoration_error <= 2e-10
' "$out/verifier.json" >/dev/null

for denied in project-scratch null-carrier; do
    set +e
    env PYTHONPATH="$here" "$python" "$phase" "--${denied}" \
        >"$out/${denied}.stdout" \
        2>"$out/${denied}.stderr"
    rc=$?
    set -e
    [[ "$rc" -eq 2 ]]
    [[ ! -s "$out/${denied}.stdout" ]]
done
rg -q '^resident BCH scratch projection denied$' \
    "$out/project-scratch.stderr"
rg -q '^invalid BCH phase carrier$' \
    "$out/null-carrier.stderr"

if rg -n \
    'witness_list|candidate_set|truth_table|assignment_expansion|residue_shadow' \
    "$phase" "$reference" "$verifier"
then
    echo "BCH path contains forbidden extensional state" >&2
    exit 1
fi

jq -n \
  --slurpfile phase "$out/phase.json" \
  --slurpfile reference "$out/reference.json" \
  --slurpfile verifier "$out/verifier.json" '
  {
    result: "PASS",
    claim: $phase[0].claim,
    claim_ceiling:
      "BOUNDED_NORMALIZED_TWO_MODULE_NONCOMMUTING_STOKES_BCH_GRADES1_TO6_PHASE_AND_EXACT_RATIONAL_DIAGNOSTIC_THROUGH_GRADE10_SOFTWARE_ONLY",
    accepted: $phase[0],
    exact_reference: $reference[0],
    exact_verifier: $verifier[0],
    terminal: false
  }
' >"$out/qualification.json"

sha256sum \
    "$phase" "$reference" "$verifier" \
    "$out/phase.json" "$out/reference.json" \
    "$out/verifier.json" "$out/qualification.json" \
    >"$out/SHA256SUMS"

echo "algebraic Stokes BCH rematerialized qualification: PASS"
