#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/growing_prime_cubic_weil_open_interface_action_span.py"
COMPONENT_DEPENDENCY="$HERE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py"
GAUSSIAN_DEPENDENCY="$HERE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py"
ORACLE="$HERE/growing_prime_cubic_weil_open_interface_action_span_independent_oracle.py"
SEALED="$HERE/GROWING_PRIME_CUBIC_WEIL_OPEN_INTERFACE_ACTION_SPAN_RESULTS.json"
SEALED_ORACLE="$HERE/GROWING_PRIME_CUBIC_WEIL_OPEN_INTERFACE_ACTION_SPAN_INDEPENDENT_ORACLE.json"
EVIDENCE="$(mktemp -d /dev/shm/ags-audio-m173-qualifier-XXXXXX)"

cp "$PRODUCTION" "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span.py"
cp "$COMPONENT_DEPENDENCY" "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py"
cp "$GAUSSIAN_DEPENDENCY" "$EVIDENCE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py"
cp "$ORACLE" "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span_independent_oracle.py"
cp "$SEALED" "$EVIDENCE/sealed.json"
cp "$SEALED_ORACLE" "$EVIDENCE/sealed_oracle.json"
PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" -m py_compile \
  "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span.py" \
  "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py" \
  "$EVIDENCE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py" \
  "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span_independent_oracle.py"
PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span.py" \
  --output "$EVIDENCE/reexecuted.json"
cmp "$EVIDENCE/sealed.json" "$EVIDENCE/reexecuted.json"
PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span_independent_oracle.py" \
  --production "$EVIDENCE/reexecuted.json" \
  --output "$EVIDENCE/reexecuted_oracle.json"
cmp "$EVIDENCE/sealed_oracle.json" "$EVIDENCE/reexecuted_oracle.json"

jq -e '
  (.cases | length) == 11 and
  ([.cases[] |
    .action_span_equals_declared_source_rank and
    .exact_canonical_state_restored and
    .same_backing_restored and
    (.snapshot_used | not) and
    .matched_source_streamed_classical.boundary_matches and
    (.intermediate_columns_serialized | not)] | all) and
  ([.controls[]] | all) and
  ([.dense_small_order_parity[]] | all) and
  .restoration_and_reuse.restoration_generation == 2 and
  .restoration_and_reuse.same_backing_reused and
  .restoration_and_reuse.second_boundary_matches_fresh and
  .observed_resource_law.all_declared_action_spans_equal_source_rank and
  .observed_resource_law.all_full_two_fiber_operators_certified_invertible and
  (.observed_resource_law.fixed_rank_open_interface_across_growing_q_established | not) and
  .matched_baselines.public_operator_word_is_a_complete_rematerialization_descriptor and
  (.resource_accounting.advantage_claimed | not)
' "$EVIDENCE/reexecuted.json" >/dev/null

jq -e '
  .qualified and
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .independence.imports_production_module == false and
  .independence.imports_predecessor_module == false and
  .independence.production_forward_inverse_or_projection_called == false and
  .all_11_cases_reconstructed and
  (.case_comparisons | length) == 11 and
  ([.case_comparisons[].checks[]] | all) and
  ([.dense_semantic_checks[]] | all) and
  ([.controls[]] | all) and
  ([.restoration_and_reuse[]] | all) and
  .observed_resource_law.executed_full_source_bundles == ["Q5_R10","Q11_R22"] and
  (.observed_resource_law.fixed_rank_general_open_interface_established | not)
' "$EVIDENCE/reexecuted_oracle.json" >/dev/null

if rg -n 'import[[:space:]]+growing_prime|from[[:space:]]+growing_prime' \
  "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span_independent_oracle.py"; then
  printf '%s\n' 'oracle dependency isolation failed' >&2
  exit 1
fi

sha256sum \
  "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span.py" \
  "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py" \
  "$EVIDENCE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py" \
  "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span_independent_oracle.py" \
  "$EVIDENCE/reexecuted.json" \
  "$EVIDENCE/reexecuted_oracle.json" > "$EVIDENCE/SHA256SUMS"
printf '%s\n' 'QUALIFIED_GROWING_PRIME_CUBIC_WEIL_OPEN_INTERFACE_ACTION_SPAN_STRICT_SCOPE'
printf 'evidence=%s\n' "$EVIDENCE"
