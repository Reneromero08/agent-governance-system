#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/growing_safe_prime_cubic_determinant_generating_relation_closure.py"
ORACLE="$HERE/growing_safe_prime_cubic_determinant_generating_relation_closure_independent_oracle.py"
SEALED="$HERE/GROWING_SAFE_PRIME_CUBIC_DETERMINANT_GENERATING_RELATION_CLOSURE_RESULTS.json"
SEALED_ORACLE="$HERE/GROWING_SAFE_PRIME_CUBIC_DETERMINANT_GENERATING_RELATION_CLOSURE_INDEPENDENT_ORACLE.json"
EVIDENCE="$(mktemp -d /dev/shm/ags-audio-m179-qualifier-XXXXXX)"

cp "$PRODUCTION" "$EVIDENCE/growing_safe_prime_cubic_determinant_generating_relation_closure.py"
cp "$ORACLE" "$EVIDENCE/growing_safe_prime_cubic_determinant_generating_relation_closure_independent_oracle.py"
cp "$SEALED" "$EVIDENCE/sealed.json"
cp "$SEALED_ORACLE" "$EVIDENCE/sealed_oracle.json"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" -m py_compile \
  "$EVIDENCE/growing_safe_prime_cubic_determinant_generating_relation_closure.py" \
  "$EVIDENCE/growing_safe_prime_cubic_determinant_generating_relation_closure_independent_oracle.py"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/growing_safe_prime_cubic_determinant_generating_relation_closure.py" \
  --output "$EVIDENCE/reexecuted.json"
cmp "$EVIDENCE/sealed.json" "$EVIDENCE/reexecuted.json"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/growing_safe_prime_cubic_determinant_generating_relation_closure_independent_oracle.py" \
  --production-source "$EVIDENCE/growing_safe_prime_cubic_determinant_generating_relation_closure.py" \
  --production-result "$EVIDENCE/reexecuted.json" \
  --output "$EVIDENCE/reexecuted_oracle.json"
cmp "$EVIDENCE/sealed_oracle.json" "$EVIDENCE/reexecuted_oracle.json"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .execution == "DIRECT_PROCESS_SOFTWARE_EXACT_FINITE_FIELD_RESIDUE_PHASE_DIAGNOSTIC" and
  .q5_full_boundary_closure.full_closure_match_count == 5 and
  .q5_full_boundary_closure.character_pairs_tested == 256 and
  .q5_full_boundary_closure.full_boundary_points_per_pair == 78125 and
  .q5_full_boundary_closure.singular_completion_coordinate_count == 27 and
  .q5_full_boundary_closure.joint_boundary_column_rank == 35 and
  ([.q5_full_boundary_closure.full_closure_matches[] |
    [.source_character_signature,.target_character_signature]]) ==
    [[[0,0],[2,2]],[[0,2],[0,2]],[[1,3],[3,1]],[[2,2],[0,0]],[[3,1],[1,3]]] and
  ([.q5_full_boundary_closure.full_closure_matches[] | .full_boundary_exact] | all) and
  ([.q5_full_boundary_closure.controls | to_entries[] |
    select(.key != "reordered_inverse_applicable" and .key != "reordered_inverse_reason") |
    .value] | all) and
  (.q5_full_boundary_closure.controls.reordered_inverse_applicable | not) and
  .q5_full_boundary_closure.restoration_and_reuse.primary_exactly_restored and
  .q5_full_boundary_closure.restoration_and_reuse.same_backing_restored and
  .q5_full_boundary_closure.restoration_and_reuse.second_program_consumed_actual_restored_backing and
  .q5_full_boundary_closure.restoration_and_reuse.second_matches_fresh and
  .q5_full_boundary_closure.restoration_and_reuse.second_exactly_restored and
  .q5_full_boundary_closure.restoration_and_reuse.same_backing_reused and
  .q5_full_boundary_closure.restoration_and_reuse.restoration_generation_after_reuse == 2 and
  (.q5_full_boundary_closure.restoration_and_reuse.snapshot_used | not) and
  .q7_transfer_attack.logical_phase_cells == 823543 and
  .q7_transfer_attack.singular_completion_coordinate_count == 37 and
  .q7_transfer_attack.open_boundary_sample_points == 576 and
  .q7_transfer_attack.character_pairs_tested == 1296 and
  .q7_transfer_attack.surviving_character_pair_count == 0 and
  .matched_baseline.identical_seven_axis_separable_classical_dft and
  .matched_baseline.identical_public_congruence_stratum_descriptor and
  .matched_baseline.identical_character_sum_recurrence and
  (.matched_baseline.computational_advantage_established | not) and
  ([.claim_boundaries[]] | all(. == false))
' "$EVIDENCE/reexecuted.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.production_source.imports_oracle | not) and
  .production_source.uses_exact_modular_equality_not_digest_for_closure and
  .q5_full_boundary_reexecution.full_match_count == 5 and
  .q5_full_boundary_reexecution.independently_verified_completion_vectors == 5 and
  .q5_full_boundary_reexecution.full_boundary_points_per_match == 78125 and
  .q5_full_boundary_reexecution.exact_forward_inverse_restoration and
  .q7_direct_character_sum_transfer_attack.independent_algorithm ==
    "DIRECT_SUM_GROUPED_BY_DETERMINANT_TRACE_AND_CONGRUENCE_CLASS" and
  .q7_direct_character_sum_transfer_attack.sample_points == 576 and
  .q7_direct_character_sum_transfer_attack.character_pairs_tested == 1296 and
  .q7_direct_character_sum_transfer_attack.surviving_pairs == [] and
  .controls.q7_different_transform_algorithm and
  .observed_resource_law.fixed_rank_transferable_closure == false
' "$EVIDENCE/reexecuted_oracle.json" >/dev/null

if rg -n 'import[[:space:]]+growing_safe_prime|from[[:space:]]+growing_safe_prime' \
  "$EVIDENCE/growing_safe_prime_cubic_determinant_generating_relation_closure_independent_oracle.py"; then
  printf '%s\n' 'oracle dependency isolation failed' >&2
  exit 1
fi

sha256sum \
  "$EVIDENCE/growing_safe_prime_cubic_determinant_generating_relation_closure.py" \
  "$EVIDENCE/growing_safe_prime_cubic_determinant_generating_relation_closure_independent_oracle.py" \
  "$EVIDENCE/reexecuted.json" \
  "$EVIDENCE/reexecuted_oracle.json" > "$EVIDENCE/SHA256SUMS"

printf '%s\n' 'QUALIFIED_GROWING_SAFE_PRIME_CUBIC_DETERMINANT_GENERATING_RELATION_CLOSURE_STRICT_SCOPE'
printf 'evidence=%s\n' "$EVIDENCE"
