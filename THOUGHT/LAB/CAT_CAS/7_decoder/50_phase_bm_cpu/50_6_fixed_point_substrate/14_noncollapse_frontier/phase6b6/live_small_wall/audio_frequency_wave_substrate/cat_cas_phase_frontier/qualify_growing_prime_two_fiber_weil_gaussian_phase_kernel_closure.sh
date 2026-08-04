#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py"
ORACLE="$HERE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure_independent_oracle.py"
SEALED="$HERE/GROWING_PRIME_TWO_FIBER_WEIL_GAUSSIAN_PHASE_KERNEL_CLOSURE_RESULTS.json"
SEALED_ORACLE="$HERE/GROWING_PRIME_TWO_FIBER_WEIL_GAUSSIAN_PHASE_KERNEL_CLOSURE_INDEPENDENT_ORACLE.json"
EVIDENCE="$(mktemp -d /dev/shm/ags-audio-m171-qualifier-XXXXXX)"

cp "$PRODUCTION" "$EVIDENCE/production.py"
cp "$ORACLE" "$EVIDENCE/oracle.py"
cp "$SEALED" "$EVIDENCE/sealed.json"
cp "$SEALED_ORACLE" "$EVIDENCE/sealed_oracle.json"
PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" -m py_compile "$EVIDENCE/production.py" "$EVIDENCE/oracle.py"
PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" "$EVIDENCE/production.py" --output "$EVIDENCE/reexecuted.json"
cmp "$EVIDENCE/sealed.json" "$EVIDENCE/reexecuted.json"
PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" "$EVIDENCE/oracle.py" \
  --production "$EVIDENCE/reexecuted.json" \
  --output "$EVIDENCE/reexecuted_oracle.json"
cmp "$EVIDENCE/sealed_oracle.json" "$EVIDENCE/reexecuted_oracle.json"

jq -e '
  .fixed16_two_port_cells_across_all_orders_and_depths and
  (.cases | length) == 11 and
  ([.cases[] |
    .exact_canonical_state_restored and
    .same_backing_restored and
    (.snapshot_used | not) and
    .matched_classical.boundary_matches and
    .matched_classical.semantic_commitment_matches and
    .matched_classical.checkpoints_match and
    .matched_classical.restoration_matches] | all) and
  ([.controls[]] | all) and
  ([.algebra_checks[]] | all) and
  .restoration_and_reuse.restoration_generation == 2 and
  .restoration_and_reuse.same_backing_reused and
  .restoration_and_reuse.unrelated_second_boundary_matches_fresh and
  .matched_baselines.state_law_identical_to_accepted and
  .matched_baselines.classical_composition_work_is_lower_after_one_scalar_gauss_cache and
  (.resource_accounting.advantage_claimed | not)
' "$EVIDENCE/reexecuted.json" >/dev/null

jq -e '
  .qualified and
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .independence.imports_production_module == false and
  .independence.production_projection_function_called == false and
  .independence.production_inverse_function_called == false and
  .all_11_public_cases_reconstructed and
  (.case_comparisons | length) == 11 and
  ([.case_comparisons[].checks[]] | all) and
  .trajectory_cocycle_parity_checks == 21648 and
  .exhaustive_q5_dense_oracle.sl2_elements == 120 and
  .exhaustive_q5_dense_oracle.ordered_pairs == 14400 and
  .exhaustive_q5_dense_oracle.all_streamed_closed_dense_equal and
  ([.controls[]] | all) and
  ([.restoration_and_reuse[]] | all) and
  .observed_resource_law.resident_two_port_field_cells == 16 and
  .observed_resource_law.logical_resident_payload_bit_values == [56,72,88,104,120] and
  (.observed_resource_law.fixed_bit_width_across_unbounded_q | not)
' "$EVIDENCE/reexecuted_oracle.json" >/dev/null

if rg -n 'import[[:space:]]+growing_prime_two_fiber|from[[:space:]]+growing_prime_two_fiber' "$EVIDENCE/oracle.py"; then
  printf '%s\n' 'oracle dependency isolation failed' >&2
  exit 1
fi

sha256sum "$EVIDENCE/production.py" "$EVIDENCE/oracle.py" "$EVIDENCE/reexecuted.json" "$EVIDENCE/reexecuted_oracle.json" > "$EVIDENCE/SHA256SUMS"
printf '%s\n' 'QUALIFIED_GROWING_PRIME_TWO_FIBER_WEIL_GAUSSIAN_PHASE_KERNEL_CLOSURE_STRICT_SCOPE'
printf 'evidence=%s\n' "$EVIDENCE"
