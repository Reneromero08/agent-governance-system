#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/growing_prime_cubic_determinant_mellin_gauss_phase_descriptor_closure.py"
PREDECESSOR="$HERE/growing_safe_prime_cubic_determinant_generating_relation_closure.py"
ORACLE="$HERE/growing_prime_cubic_determinant_mellin_gauss_phase_descriptor_closure_independent_oracle.py"
SEALED="$HERE/GROWING_PRIME_CUBIC_DETERMINANT_MELLIN_GAUSS_PHASE_DESCRIPTOR_CLOSURE_RESULTS.json"
SEALED_ORACLE="$HERE/GROWING_PRIME_CUBIC_DETERMINANT_MELLIN_GAUSS_PHASE_DESCRIPTOR_CLOSURE_INDEPENDENT_ORACLE.json"
EVIDENCE="$(mktemp -d /dev/shm/ags-audio-m180-qualifier-XXXXXX)"

cp "$PRODUCTION" "$EVIDENCE/growing_prime_cubic_determinant_mellin_gauss_phase_descriptor_closure.py"
cp "$PREDECESSOR" "$EVIDENCE/growing_safe_prime_cubic_determinant_generating_relation_closure.py"
cp "$ORACLE" "$EVIDENCE/growing_prime_cubic_determinant_mellin_gauss_phase_descriptor_closure_independent_oracle.py"
cp "$SEALED" "$EVIDENCE/sealed.json"
cp "$SEALED_ORACLE" "$EVIDENCE/sealed_oracle.json"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" -m py_compile \
  "$EVIDENCE/growing_prime_cubic_determinant_mellin_gauss_phase_descriptor_closure.py" \
  "$EVIDENCE/growing_safe_prime_cubic_determinant_generating_relation_closure.py" \
  "$EVIDENCE/growing_prime_cubic_determinant_mellin_gauss_phase_descriptor_closure_independent_oracle.py"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/growing_prime_cubic_determinant_mellin_gauss_phase_descriptor_closure.py" \
  --output "$EVIDENCE/reexecuted.json"
cmp "$EVIDENCE/sealed.json" "$EVIDENCE/reexecuted.json"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/growing_prime_cubic_determinant_mellin_gauss_phase_descriptor_closure_independent_oracle.py" \
  --production-source "$EVIDENCE/growing_prime_cubic_determinant_mellin_gauss_phase_descriptor_closure.py" \
  --production-result "$EVIDENCE/reexecuted.json" \
  --output "$EVIDENCE/reexecuted_oracle.json"
cmp "$EVIDENCE/sealed_oracle.json" "$EVIDENCE/reexecuted_oracle.json"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .descriptor.singular_conductor_character_count == 2 and
  (.descriptor.middle_extension_sheaf_claimed | not) and
  ([.determinant_gamma_law[] | .all_pass] | all) and
  ([.determinant_gamma_law[] | .singular_support_character_exponents == .expected_exceptional_exponents] | all) and
  ([.full_descriptor_cases[] | .full_boundary_exact] | all) and
  ([.full_descriptor_cases[] | .accepted_peak_materialized_field_cells == (4 * .q - 3)] | all) and
  ([.full_descriptor_cases[] | .restoration_and_reuse.primary_exactly_restored] | all) and
  ([.full_descriptor_cases[] | .restoration_and_reuse.same_backing_restored] | all) and
  ([.full_descriptor_cases[] | .restoration_and_reuse.projected_result_survives_inverse] | all) and
  ([.full_descriptor_cases[] | .restoration_and_reuse.second_program_consumed_actual_restored_backing] | all) and
  ([.full_descriptor_cases[] | .restoration_and_reuse.second_matches_fresh] | all) and
  ([.full_descriptor_cases[] | .restoration_and_reuse.second_exactly_restored] | all) and
  ([.full_descriptor_cases[] | .restoration_and_reuse.restoration_generation_after_reuse == 2] | all) and
  ([.full_descriptor_cases[] | .restoration_and_reuse.snapshot_used | not] | all) and
  .observed_resource_law.resident_descriptor_cells == "q-1" and
  .observed_resource_law.accepted_peak_field_cells_including_compiler_tables == "4*q-3" and
  (.observed_resource_law.resident_width_is_fixed | not) and
  .matched_baseline.identical_mellin_gauss_character_sum_recurrence and
  (.matched_baseline.computational_advantage_established | not) and
  ([.claim_boundaries[]] | all(. == false))
' "$EVIDENCE/reexecuted.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.production_source.oracle_imports_production | not) and
  ([.representative_character_sum_attacks[] | .all_pass] | all) and
  ([.representative_character_sum_attacks[] | .omitting_either_exceptional_term_changes_a_direct_sum] | all) and
  ([.full_boundary_reexecutions[] | .full_boundary_exact] | all) and
  ([.full_boundary_reexecutions[] | .missing_one_mellin_channel_fails] | all) and
  ([.full_boundary_reexecutions[] | .wrong_one_gamma_factor_fails] | all) and
  ([.full_boundary_reexecutions[] | .exact_descriptor_restoration] | all) and
  ([.full_boundary_reexecutions[] | .same_backing_restored] | all) and
  ([.full_boundary_reexecutions[] | .unrelated_second_program_matches_fresh] | all) and
  ([.full_boundary_reexecutions[] | .second_program_exactly_restored] | all) and
  ([.full_boundary_reexecutions[] | .restoration_generation_after_reuse == 2] | all) and
  .mutations.missing_one_mellin_channel_fails_all_cases and
  .mutations.wrong_one_gamma_factor_fails_all_cases and
  .mutations.q7_scalar_stratum_failure_preserved and
  .observed_resource_law.matches_production and
  .observed_resource_law.resident_width == "q-1" and
  .observed_resource_law.accepted_peak == "4*q-3" and
  .observed_resource_law.identical_classical_recurrence_preserved
' "$EVIDENCE/reexecuted_oracle.json" >/dev/null

if rg -n 'import[[:space:]]+growing_prime_cubic_determinant_mellin|from[[:space:]]+growing_prime_cubic_determinant_mellin' \
  "$EVIDENCE/growing_prime_cubic_determinant_mellin_gauss_phase_descriptor_closure_independent_oracle.py"; then
  printf '%s\n' 'oracle dependency isolation failed' >&2
  exit 1
fi

sha256sum \
  "$EVIDENCE/growing_prime_cubic_determinant_mellin_gauss_phase_descriptor_closure.py" \
  "$EVIDENCE/growing_safe_prime_cubic_determinant_generating_relation_closure.py" \
  "$EVIDENCE/growing_prime_cubic_determinant_mellin_gauss_phase_descriptor_closure_independent_oracle.py" \
  "$EVIDENCE/reexecuted.json" \
  "$EVIDENCE/reexecuted_oracle.json" > "$EVIDENCE/SHA256SUMS"

printf '%s\n' 'QUALIFIED_GROWING_PRIME_CUBIC_DETERMINANT_MELLIN_GAUSS_PHASE_DESCRIPTOR_CLOSURE_STRICT_SCOPE'
printf 'evidence=%s\n' "$EVIDENCE"
