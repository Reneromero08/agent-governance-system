#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank.py"
OPEN_ACTION_DEPENDENCY="$HERE/growing_prime_cubic_weil_open_interface_action_span.py"
COMPONENT_DEPENDENCY="$HERE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py"
GAUSSIAN_DEPENDENCY="$HERE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py"
ORACLE="$HERE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank_independent_oracle.py"
SEALED="$HERE/GROWING_PRIME_RESIDENT_CUBIC_STRENGTH_PHASE_PORT_WEIL_ACTION_RANK_RESULTS.json"
SEALED_ORACLE="$HERE/GROWING_PRIME_RESIDENT_CUBIC_STRENGTH_PHASE_PORT_WEIL_ACTION_RANK_INDEPENDENT_ORACLE.json"
EVIDENCE="$(mktemp -d /dev/shm/ags-audio-m174-qualifier-XXXXXX)"

cp "$PRODUCTION" "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank.py"
cp "$OPEN_ACTION_DEPENDENCY" "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span.py"
cp "$COMPONENT_DEPENDENCY" "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py"
cp "$GAUSSIAN_DEPENDENCY" "$EVIDENCE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py"
cp "$ORACLE" "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank_independent_oracle.py"
cp "$SEALED" "$EVIDENCE/sealed.json"
cp "$SEALED_ORACLE" "$EVIDENCE/sealed_oracle.json"
PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" -m py_compile \
  "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank.py" \
  "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span.py" \
  "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py" \
  "$EVIDENCE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py" \
  "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank_independent_oracle.py"
PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank.py" \
  --output "$EVIDENCE/reexecuted.json"
cmp "$EVIDENCE/sealed.json" "$EVIDENCE/reexecuted.json"
PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank_independent_oracle.py" \
  --production "$EVIDENCE/reexecuted.json" \
  --output "$EVIDENCE/reexecuted_oracle.json"
cmp "$EVIDENCE/sealed_oracle.json" "$EVIDENCE/reexecuted_oracle.json"

jq -e '
  (.cases | length) == 11 and
  ([.cases[].q] | unique) == [5,11,23,29,41,53,83,89,113] and
  ([.cases[] |
    .initial_latent_data_separation_rank == 1 and
    .final_latent_data_separation_rank == .q and
    .all_controlled_cubic_separation_ranks_full_q and
    ([.controlled_cubic_separation_ranks[] == .q] | all) and
    .shared_latent_port_consumers >= 2 and
    .cube_map_bijective_on_fq and
    .exact_canonical_payload_restored and
    .same_backing_restored and
    (.snapshot_used | not) and
    (.resident_latent_amplitudes_serialized | not) and
    (.intermediate_joint_state_serialized | not) and
    .accepted_resident_field_cells == (2*.q*.q) and
    .accepted_resident_plus_temporary_peak_field_cells == (2*.q*.q+.q) and
    .matched_identical_matrix_free_classical.boundary_matches and
    .matched_identical_matrix_free_classical.full_state_commitment_matches and
    .matched_identical_matrix_free_classical.forward_work_matches and
    .matched_identical_matrix_free_classical.resident_field_cells == (2*.q*.q) and
    .matched_exact_rader_ntt_classical.boundary_matches and
    .matched_exact_rader_ntt_classical.full_state_commitment_matches and
    .matched_exact_rader_ntt_classical.single_auxiliary_modulus_exactness_bound_checked and
    .matched_exact_rader_ntt_classical.retained_ntt_kernel_cache_cells == 0 and
    .matched_exact_rader_ntt_classical.work.exact_convolution_coefficient_bound_peak < .matched_exact_rader_ntt_classical.auxiliary_ntt_prime and
    .matched_exact_rader_ntt_classical.scratch_logical_payload_cells_peak == (4*.q-2+2*.matched_exact_rader_ntt_classical.work.ntt_max_length) and
    .matched_exact_rader_ntt_classical.resident_plus_scratch_logical_payload_cells_peak == (2*.q*.q+.matched_exact_rader_ntt_classical.scratch_logical_payload_cells_peak) and
    .matched_exact_rader_ntt_classical.combined_payload_bit_capacity_upper_bound_at_peak == (.matched_exact_rader_ntt_classical.field_cells_at_combined_peak*.matched_exact_rader_ntt_classical.field_cell_bit_capacity + .matched_exact_rader_ntt_classical.auxiliary_integer_cells_at_combined_peak*.matched_exact_rader_ntt_classical.auxiliary_integer_cell_bit_capacity)
  ] | all) and
  ([.controls[]] | all) and
  .restoration_and_reuse.restoration_generation == 2 and
  .restoration_and_reuse.same_backing_reused and
  .restoration_and_reuse.second_boundary_matches_fresh and
  .restoration_and_reuse.second_commitment_matches_fresh and
  (.restoration_and_reuse.snapshot_used | not) and
  .observed_resource_law.initial_product_separation_rank == 1 and
  .observed_resource_law.all_first_controlled_cubic_separation_ranks == "Q" and
  .observed_resource_law.all_declared_final_separation_ranks == "Q" and
  (.observed_resource_law.fixed_rank_across_growing_q_established | not) and
  .matched_baseline.all_case_boundaries_match and
  .matched_baseline.all_case_full_state_commitments_match and
  .matched_baseline.all_rader_ntt_boundaries_match and
  .matched_baseline.same_leading_2q2_resident_field_cell_law and
  (.claim_boundaries[] | not) and
  (.claim_ceiling | contains("AUXILIARY_NTT_MODULUS998244353"))
' "$EVIDENCE/reexecuted.json" >/dev/null

jq -e '
  .qualified and
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.independence.imports_production_module | not) and
  (.independence.imports_predecessor_module | not) and
  (.independence.production_forward_inverse_projection_rank_or_baseline_called | not) and
  .independence.production_result_used_only_as_comparison_target and
  .independence.separate_field_plan_rank_inverse_dense_and_rader_recurrences and
  .all_11_cases_reconstructed and
  (.case_comparisons | length) == 11 and
  ([.case_comparisons[].checks[]] | all) and
  ([.rader_direct_semantic_checks[]] | all) and
  ([.controls[]] | all) and
  ([.restoration_and_reuse[]] | all) and
  .observed_resource_law.initial_rank == 1 and
  .observed_resource_law.first_controlled_rank == "Q" and
  .observed_resource_law.final_rank == "Q" and
  (.observed_resource_law.fixed_rank_across_growing_q_established | not) and
  (.claim_ceiling | contains("AUXILIARY_NTT_MODULUS998244353"))
' "$EVIDENCE/reexecuted_oracle.json" >/dev/null

if rg -n 'import[[:space:]]+growing_prime|from[[:space:]]+growing_prime' \
  "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank_independent_oracle.py"; then
  printf '%s\n' 'oracle dependency isolation failed' >&2
  exit 1
fi

sha256sum \
  "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank.py" \
  "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span.py" \
  "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py" \
  "$EVIDENCE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py" \
  "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank_independent_oracle.py" \
  "$EVIDENCE/reexecuted.json" \
  "$EVIDENCE/reexecuted_oracle.json" > "$EVIDENCE/SHA256SUMS"
printf '%s\n' 'QUALIFIED_GROWING_PRIME_RESIDENT_CUBIC_STRENGTH_PHASE_PORT_WEIL_ACTION_RANK_STRICT_SCOPE'
printf 'evidence=%s\n' "$EVIDENCE"
