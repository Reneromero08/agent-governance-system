#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py"
DEPENDENCY="$HERE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py"
ORACLE="$HERE/growing_prime_two_fiber_cubic_deformed_weil_component_rank_independent_oracle.py"
SEALED="$HERE/GROWING_PRIME_TWO_FIBER_CUBIC_DEFORMED_WEIL_COMPONENT_RANK_RESULTS.json"
SEALED_ORACLE="$HERE/GROWING_PRIME_TWO_FIBER_CUBIC_DEFORMED_WEIL_COMPONENT_RANK_INDEPENDENT_ORACLE.json"
EVIDENCE="$(mktemp -d /dev/shm/ags-audio-m172-qualifier-XXXXXX)"

cp "$PRODUCTION" "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py"
cp "$DEPENDENCY" "$EVIDENCE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py"
cp "$ORACLE" "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank_independent_oracle.py"
cp "$SEALED" "$EVIDENCE/sealed.json"
cp "$SEALED_ORACLE" "$EVIDENCE/sealed_oracle.json"
PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" -m py_compile \
  "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py" \
  "$EVIDENCE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py" \
  "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank_independent_oracle.py"
PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py" \
  --output "$EVIDENCE/reexecuted.json"
cmp "$EVIDENCE/sealed.json" "$EVIDENCE/reexecuted.json"
PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank_independent_oracle.py" \
  --production "$EVIDENCE/reexecuted.json" \
  --output "$EVIDENCE/reexecuted_oracle.json"
cmp "$EVIDENCE/sealed_oracle.json" "$EVIDENCE/reexecuted_oracle.json"

jq -e '
  (.cases | length) == 11 and
  ([.cases[] |
    .exact_canonical_state_restored and
    .same_backing_restored and
    (.snapshot_used | not) and
    .matched_q_vector_classical.boundary_matches and
    (.hidden_relation_entries_serialized | not) and
    (.checkpoints[0].deformed_port_gaussian_component_support >= ((.q - 3) * (.q - 3)))] | all) and
  ([.controls[]] | all) and
  ([.algebra_checks[]] | all) and
  .restoration_and_reuse.restoration_generation == 2 and
  .restoration_and_reuse.same_backing_reused and
  .restoration_and_reuse.unrelated_second_boundary_matches_fresh and
  .matched_baselines.linear_q_dynamic_state and
  .matched_baselines.all_case_boundaries_match and
  .resource_accounting.matched_q_vector_live_field_cells == "2*Q" and
  (.resource_accounting.advantage_claimed | not)
' "$EVIDENCE/reexecuted.json" >/dev/null

jq -e '
  .qualified and
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .independence.imports_production_module == false and
  .independence.imports_gaussian_predecessor_module == false and
  .independence.production_projection_or_inverse_called == false and
  .all_11_cases_reconstructed and
  (.case_comparisons | length) == 11 and
  ([.case_comparisons[].checks[]] | all) and
  .dense_semantic_checks.all_3000_q5_gaussian_displacement_products_match_dense_semantics and
  .dense_semantic_checks.all_q5_q11_cubic_spectra_reconstruct_exact_diagonals and
  .dense_semantic_checks.q5_depth2_full_program_component_dense_parity and
  .dense_semantic_checks.q11_depth1_full_program_component_dense_parity and
  ([.controls[]] | all) and
  ([.restoration_and_reuse[]] | all) and
  .observed_resource_law.all_first_supports_at_least_q_minus_3_squared and
  .observed_resource_law.accepted_component_coordinates_are_quadratic_in_q and
  .observed_resource_law.matched_q_vector_dynamic_state == "2*Q_FIELD_CELLS" and
  (.observed_resource_law.fixed_bit_width_across_unbounded_q | not)
' "$EVIDENCE/reexecuted_oracle.json" >/dev/null

if rg -n 'import[[:space:]]+growing_prime_two_fiber|from[[:space:]]+growing_prime_two_fiber' \
  "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank_independent_oracle.py"; then
  printf '%s\n' 'oracle dependency isolation failed' >&2
  exit 1
fi

sha256sum \
  "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py" \
  "$EVIDENCE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py" \
  "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank_independent_oracle.py" \
  "$EVIDENCE/reexecuted.json" \
  "$EVIDENCE/reexecuted_oracle.json" > "$EVIDENCE/SHA256SUMS"
printf '%s\n' 'QUALIFIED_GROWING_PRIME_TWO_FIBER_CUBIC_DEFORMED_WEIL_COMPONENT_RANK_STRICT_SCOPE'
printf 'evidence=%s\n' "$EVIDENCE"
