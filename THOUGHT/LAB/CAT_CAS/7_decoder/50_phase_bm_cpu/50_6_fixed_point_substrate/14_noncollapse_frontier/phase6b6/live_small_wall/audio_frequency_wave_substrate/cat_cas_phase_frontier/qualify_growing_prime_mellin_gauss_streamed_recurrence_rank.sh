#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/growing_prime_mellin_gauss_streamed_recurrence_rank.py"
ORACLE="$HERE/growing_prime_mellin_gauss_streamed_recurrence_rank_independent_oracle.py"
SEALED="$HERE/GROWING_PRIME_MELLIN_GAUSS_STREAMED_RECURRENCE_RANK_RESULTS.json"
SEALED_ORACLE="$HERE/GROWING_PRIME_MELLIN_GAUSS_STREAMED_RECURRENCE_RANK_INDEPENDENT_ORACLE.json"
REVIEW="$HERE/GROWING_PRIME_MELLIN_GAUSS_STREAMED_RECURRENCE_RANK_INDEPENDENT_REEXECUTION_REVIEW.md"
EVIDENCE="$(mktemp -d /dev/shm/ags-audio-m181-qualifier-XXXXXX)"

cp "$PRODUCTION" "$EVIDENCE/production.py"
cp "$ORACLE" "$EVIDENCE/oracle.py"
cp "$SEALED" "$EVIDENCE/sealed.json"
cp "$SEALED_ORACLE" "$EVIDENCE/sealed_oracle.json"
cp "$REVIEW" "$EVIDENCE/review.md"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" -m py_compile \
  "$EVIDENCE/production.py" \
  "$EVIDENCE/oracle.py"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/production.py" \
  --output "$EVIDENCE/reexecuted.json"
cmp "$EVIDENCE/sealed.json" "$EVIDENCE/reexecuted.json"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/oracle.py" \
  --production-source "$EVIDENCE/production.py" \
  --production-result "$EVIDENCE/reexecuted.json" \
  --output "$EVIDENCE/reexecuted_oracle.json"
cmp "$EVIDENCE/sealed_oracle.json" "$EVIDENCE/reexecuted_oracle.json"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.declared_fields | length) == 14 and
  (.rank_diagnostics | length) == 14 and
  (.transaction_cases | length) == 16 and
  ([.rank_diagnostics[] |
    .source_coefficient_cyclic_hankel_rank == (.q - 1) and
    .source_coefficient_periodic_linear_recurrence_order == (.q - 1) and
    .gamma_cyclic_hankel_rank == (.q - 1) and
    .zero_one_spectral_mode_mutation_rank == (.q - 2)] | all) and
  ([.transaction_cases[] |
    .workspace_field_cells == 10 and
    .primary.exactly_restored and
    .primary.projected_result_survives_inverse and
    .same_backing_primary and
    .same_backing_reused and
    (.generation_or_lease_metadata_enforced | not) and
    .unrelated_second_program.actual_restored_backing_consumed and
    .unrelated_second_program.fresh_restored_resource_signature_agrees and
    .unrelated_second_program.exactly_restored_again and
    .controls.missing_inverse_fails and
    .controls.wrong_inverse_fails and
    .controls.omitted_channel_changes_boundary and
    .controls.ascending_inverse_also_restores and
    (.controls.snapshot_used | not)] | all) and
  .observed_resource_law.accepted_workspace_carrier_field_cells == 10 and
  .observed_resource_law.accepted_full_lifecycle_named_exact_field_peak_including_output == 16 and
  .observed_resource_law.resident_gauss_or_coefficient_tables == 0 and
  (.observed_resource_law.accepted_workspace_exact_bit_width_independent_of_q | not) and
  .matched_baselines.identical_classical_stream.workspace_field_cells == 10 and
  (.matched_baselines.computational_advantage_established | not) and
  ([.claim_boundaries[]] | all(. == false))
' "$EVIDENCE/reexecuted.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.oracle_imports_production | not) and
  (.oracle_imports_predecessor | not) and
  (.production_source.imports_production_or_predecessor | not) and
  .production_source.declares_ten_cell_workspace and
  (.independent_rank_checks | length) == 14 and
  (.independent_transactions | length) == 16 and
  (.direct_fourier_checks | length) == 4 and
  ([.independent_rank_checks[] |
    .source_coefficient_fourier_support == (.q - 1) and
    .source_coefficient_cyclic_hankel_rank == (.q - 1) and
    .source_coefficient_periodic_recurrence_order == (.q - 1) and
    .matches_production] | all) and
  ([.independent_transactions[] |
    .boundary_matches_table and
    .ten_cell_same_backing_primary_restoration and
    .projected_scalar_persists_after_inverse and
    .actual_restored_backing_reuse_matches_fresh and
    .ten_cell_same_backing_second_restoration and
    .wrong_inverse_fails and
    .omitted_channel_changes_boundary and
    (.snapshot_used | not)] | all) and
  ([.direct_fourier_checks[] | .matches_streamed_boundary] | all) and
  .mutations.wrong_inverse_fails_all_transactions and
  .mutations.omitted_channel_changes_all_boundaries and
  .mutations.wrong_gamma_shift_changes_all_applicable_boundaries and
  .observed_resource_law.accepted_workspace_field_cells == 10 and
  .observed_resource_law.resident_tables == 0 and
  (.observed_resource_law.fixed_exact_bit_width | not) and
  .observed_resource_law.identical_classical_stream_preserved
' "$EVIDENCE/reexecuted_oracle.json" >/dev/null

if rg -n 'import[[:space:]]+growing_prime_mellin_gauss|from[[:space:]]+growing_prime_mellin_gauss' \
  "$EVIDENCE/oracle.py"; then
  printf '%s\n' 'oracle dependency isolation failed' >&2
  exit 1
fi

sha256sum \
  "$EVIDENCE/production.py" \
  "$EVIDENCE/oracle.py" \
  "$EVIDENCE/reexecuted.json" \
  "$EVIDENCE/reexecuted_oracle.json" \
  "$EVIDENCE/review.md" > "$EVIDENCE/SHA256SUMS"

printf '%s\n' 'QUALIFIED_GROWING_PRIME_MELLIN_GAUSS_STREAMED_RECURRENCE_RANK_STRICT_SCOPE'
printf 'evidence=%s\n' "$EVIDENCE"
