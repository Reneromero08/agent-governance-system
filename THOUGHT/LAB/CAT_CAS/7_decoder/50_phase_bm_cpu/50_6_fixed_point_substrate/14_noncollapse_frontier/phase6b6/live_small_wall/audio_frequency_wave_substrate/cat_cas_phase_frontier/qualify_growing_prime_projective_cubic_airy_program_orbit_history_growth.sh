#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/growing_prime_projective_cubic_airy_program_orbit_history_growth.py"
RESIDENT_DEPENDENCY="$HERE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank.py"
OPEN_ACTION_DEPENDENCY="$HERE/growing_prime_cubic_weil_open_interface_action_span.py"
COMPONENT_DEPENDENCY="$HERE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py"
GAUSSIAN_DEPENDENCY="$HERE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py"
ORACLE="$HERE/growing_prime_projective_cubic_airy_program_orbit_history_growth_independent_oracle.py"
SEALED="$HERE/GROWING_PRIME_PROJECTIVE_CUBIC_AIRY_PROGRAM_ORBIT_HISTORY_GROWTH_RESULTS.json"
SEALED_ORACLE="$HERE/GROWING_PRIME_PROJECTIVE_CUBIC_AIRY_PROGRAM_ORBIT_HISTORY_GROWTH_INDEPENDENT_ORACLE.json"
EVIDENCE="$(mktemp -d /dev/shm/ags-audio-m178-qualifier-XXXXXX)"

cp "$PRODUCTION" "$EVIDENCE/growing_prime_projective_cubic_airy_program_orbit_history_growth.py"
cp "$RESIDENT_DEPENDENCY" "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank.py"
cp "$OPEN_ACTION_DEPENDENCY" "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span.py"
cp "$COMPONENT_DEPENDENCY" "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py"
cp "$GAUSSIAN_DEPENDENCY" "$EVIDENCE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py"
cp "$ORACLE" "$EVIDENCE/growing_prime_projective_cubic_airy_program_orbit_history_growth_independent_oracle.py"
cp "$SEALED" "$EVIDENCE/sealed.json"
cp "$SEALED_ORACLE" "$EVIDENCE/sealed_oracle.json"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" -m py_compile \
  "$EVIDENCE/growing_prime_projective_cubic_airy_program_orbit_history_growth.py" \
  "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank.py" \
  "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span.py" \
  "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py" \
  "$EVIDENCE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py" \
  "$EVIDENCE/growing_prime_projective_cubic_airy_program_orbit_history_growth_independent_oracle.py"

PYTHONPATH="$EVIDENCE" PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/growing_prime_projective_cubic_airy_program_orbit_history_growth.py" \
  --output "$EVIDENCE/reexecuted.json"
cmp "$EVIDENCE/sealed.json" "$EVIDENCE/reexecuted.json"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/growing_prime_projective_cubic_airy_program_orbit_history_growth_independent_oracle.py" \
  --production-source "$EVIDENCE/growing_prime_projective_cubic_airy_program_orbit_history_growth.py" \
  --production-result "$EVIDENCE/reexecuted.json" \
  --output "$EVIDENCE/reexecuted_oracle.json"
cmp "$EVIDENCE/sealed_oracle.json" "$EVIDENCE/reexecuted_oracle.json"

jq -e '
  ([.cases[] | [.q,.depth]]) == [[5,1],[5,2],[5,3],[5,4],[11,1],[11,2]] and
  ([.cases[0:3][] | .projective_collisions] | all(. == 0)) and
  .cases[3].public_program_histories == 390625 and
  .cases[3].distinct_projective_full_states == 388125 and
  .cases[3].projective_collisions == 2500 and
  .cases[3].non_singleton_projective_classes == 625 and
  .cases[3].projective_collision_class_size_histogram == {"1":387500,"5":625} and
  .cases[3].histories_in_non_singleton_projective_classes == 3125 and
  .cases[3].minimum_exact_projective_state_identifier_bits == 19 and
  .cases[3].minimum_exact_public_history_identifier_bits == 19 and
  .cases[3].collision_relation_diagnostic.colliding_history_span_rank == 5 and
  .cases[3].collision_relation_diagnostic.colliding_histories_form_complete_linear_subspace and
  .cases[3].collision_relation_diagnostic.single_collision_difference_line and
  .cases[3].collision_relation_diagnostic.normalized_collision_difference_line_generators == [[0,0,0,1,0,0,0,4]] and
  .cases[3].collision_relation_diagnostic.all_collisions_are_raw_state_equal_not_only_projectively_equal and
  ([.cases[4:6][] | .projective_collisions] | all(. == 0)) and
  ([.cases[] | .exact_dense_carrier_restored and .same_backing_restored and (.snapshot_used | not)] | all) and
  ([.controls[]] | all) and
  .restoration_and_reuse.restoration_generation == 2 and
  .restoration_and_reuse.second_matches_fresh and
  .restoration_and_reuse.exact_payload_restored_after_reuse and
  .restoration_and_reuse.same_backing_reused and
  (.restoration_and_reuse.snapshot_used | not) and
  .matched_baseline.identical_classical_factor_graph_with_2d_public_coefficients and
  .matched_baseline.identical_classical_in_place_dense_traversal and
  (.claim_boundaries[] | not)
' "$EVIDENCE/reexecuted.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.case_certificates | length) == 6 and
  .case_certificates[3].projective_collisions == 2500 and
  .case_certificates[3].non_singleton_projective_classes == 625 and
  .case_certificates[3].collision_relation_diagnostic.single_collision_difference_line and
  ([.case_certificates[] | .exact_root_restored] | all) and
  ([.controls[]] | all) and
  ([.restoration_and_reuse | to_entries[] | select(.key!="snapshot_used") | .value] | all) and
  (.restoration_and_reuse.snapshot_used | not) and
  (.production_source.production_source_imports_oracle | not) and
  .production_source.exact_projective_traversal_present and
  (.production_source.cryptographic_digest_used_as_state_equality | not) and
  .observed_resource_law.q5_depth4_identifier_bits_before_and_after_quotient == 19
' "$EVIDENCE/reexecuted_oracle.json" >/dev/null

if rg -n 'import[[:space:]]+growing_prime|from[[:space:]]+growing_prime' \
  "$EVIDENCE/growing_prime_projective_cubic_airy_program_orbit_history_growth_independent_oracle.py"; then
  printf '%s\n' 'oracle dependency isolation failed' >&2
  exit 1
fi

sha256sum \
  "$EVIDENCE/growing_prime_projective_cubic_airy_program_orbit_history_growth.py" \
  "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank.py" \
  "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span.py" \
  "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py" \
  "$EVIDENCE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py" \
  "$EVIDENCE/growing_prime_projective_cubic_airy_program_orbit_history_growth_independent_oracle.py" \
  "$EVIDENCE/reexecuted.json" \
  "$EVIDENCE/reexecuted_oracle.json" > "$EVIDENCE/SHA256SUMS"

printf '%s\n' 'QUALIFIED_GROWING_PRIME_PROJECTIVE_CUBIC_AIRY_PROGRAM_ORBIT_HISTORY_GROWTH_STRICT_SCOPE'
printf 'evidence=%s\n' "$EVIDENCE"
