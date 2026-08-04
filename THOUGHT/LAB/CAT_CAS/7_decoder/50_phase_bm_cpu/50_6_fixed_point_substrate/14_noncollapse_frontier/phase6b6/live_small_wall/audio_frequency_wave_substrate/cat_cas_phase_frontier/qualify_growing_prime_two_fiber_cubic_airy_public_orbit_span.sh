#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/growing_prime_two_fiber_cubic_airy_public_orbit_span.py"
RESIDENT_DEPENDENCY="$HERE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank.py"
OPEN_ACTION_DEPENDENCY="$HERE/growing_prime_cubic_weil_open_interface_action_span.py"
COMPONENT_DEPENDENCY="$HERE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py"
GAUSSIAN_DEPENDENCY="$HERE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py"
ORACLE="$HERE/growing_prime_two_fiber_cubic_airy_public_orbit_span_independent_oracle.py"
SEALED="$HERE/GROWING_PRIME_TWO_FIBER_CUBIC_AIRY_PUBLIC_ORBIT_SPAN_RESULTS.json"
SEALED_ORACLE="$HERE/GROWING_PRIME_TWO_FIBER_CUBIC_AIRY_PUBLIC_ORBIT_SPAN_INDEPENDENT_ORACLE.json"
EVIDENCE="$(mktemp -d /dev/shm/ags-audio-m177-qualifier-XXXXXX)"

cp "$PRODUCTION" "$EVIDENCE/growing_prime_two_fiber_cubic_airy_public_orbit_span.py"
cp "$RESIDENT_DEPENDENCY" "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank.py"
cp "$OPEN_ACTION_DEPENDENCY" "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span.py"
cp "$COMPONENT_DEPENDENCY" "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py"
cp "$GAUSSIAN_DEPENDENCY" "$EVIDENCE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py"
cp "$ORACLE" "$EVIDENCE/growing_prime_two_fiber_cubic_airy_public_orbit_span_independent_oracle.py"
cp "$SEALED" "$EVIDENCE/sealed.json"
cp "$SEALED_ORACLE" "$EVIDENCE/sealed_oracle.json"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" -m py_compile \
  "$EVIDENCE/growing_prime_two_fiber_cubic_airy_public_orbit_span.py" \
  "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank.py" \
  "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span.py" \
  "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py" \
  "$EVIDENCE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py" \
  "$EVIDENCE/growing_prime_two_fiber_cubic_airy_public_orbit_span_independent_oracle.py"

PYTHONPATH="$EVIDENCE" PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/growing_prime_two_fiber_cubic_airy_public_orbit_span.py" \
  --output "$EVIDENCE/reexecuted.json"
cmp "$EVIDENCE/sealed.json" "$EVIDENCE/reexecuted.json"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/growing_prime_two_fiber_cubic_airy_public_orbit_span_independent_oracle.py" \
  --production-source "$EVIDENCE/growing_prime_two_fiber_cubic_airy_public_orbit_span.py" \
  --production-result "$EVIDENCE/reexecuted.json" \
  --output "$EVIDENCE/reexecuted_oracle.json"
cmp "$EVIDENCE/sealed_oracle.json" "$EVIDENCE/reexecuted_oracle.json"

jq -e '
  (.cases | length) == 17 and
  ([.cases[] | select(.program_family=="PRIMARY") | .q] | sort) == [5,11,23,29,41,53,83,89,113] and
  ([.cases[] | select(.program_family=="ALTERNATE") | .q] | sort) == [11,23,29,41,53,83,89,113] and
  ([.cases[] |
    .resident_factor_field_cells == (3*.q) and
    .public_morphism_node_records == 5 and
    .certificate.public_two_shear_program_count == (.q*.q) and
    (.certificate.full_q2_by_q2_public_orbit_matrix_materialized | not) and
    .certificate.all_nonzero_levels_equivalent and
    (if .program_family=="PRIMARY" then
       .certificate.zero_level_rank == .q and
       .certificate.representative_nonzero_level_rank == .q and
       .certificate.public_orbit_linear_span == (.q*.q) and
       .certificate.alternate_quadratic_phase_chart_coefficients == null
     else
       .certificate.zero_level_rank == .q and
       .certificate.representative_nonzero_level_rank == (.q-1) and
       .certificate.public_orbit_linear_span == (.q*.q-.q+1) and
       (.certificate.alternate_quadratic_phase_chart_coefficients | length) == 5
     end) and
    (.q as $q | [.sampled_dense_boundaries[] |
      .expanded_verification_state_field_cells == (2*$q*$q)
    ] | all) and
    .expanded_dense_state_used_only_for_sampled_verification and
    .exact_graph_payload_restored and
    .same_backing_restored and
    (.snapshot_used | not)
  ] | all) and
  ([.controls[]] | all) and
  .restoration_and_reuse.restoration_generation == 2 and
  .restoration_and_reuse.first_primary_span == 529 and
  .restoration_and_reuse.second_alternate_span == 507 and
  .restoration_and_reuse.second_matches_fresh and
  .restoration_and_reuse.exact_payload_restored_after_reuse and
  .restoration_and_reuse.same_backing_reused and
  (.restoration_and_reuse.snapshot_used | not) and
  .matched_baseline.identical_classical_factor_graph_recurrence and
  .matched_baseline.identical_classical_block_fourier_rank_recurrence and
  .matched_baseline.identical_dense_q2_joint_state_recurrence and
  (.claim_boundaries[] | not)
' "$EVIDENCE/reexecuted.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.certificates | length) == 17 and
  ([.certificates[] |
    .all_nonzero_level_ranks_equal and
    (.full_orbit_matrix_materialized | not) and
    (if .program_family=="PRIMARY" then
       .public_orbit_linear_span == (.q*.q)
     else
       .public_orbit_linear_span == (.q*.q-.q+1)
     end)
  ] | all) and
  .direct_full_orbit_checks.q5_primary_direct_full_orbit_rank == 25 and
  .direct_full_orbit_checks.q11_primary_direct_full_orbit_rank == 121 and
  .direct_full_orbit_checks.q11_alternate_direct_full_orbit_rank == 111 and
  ([.controls[]] | all) and
  ([.restoration_and_reuse | to_entries[] | select(.key!="snapshot_used") | .value] | all) and
  (.restoration_and_reuse.snapshot_used | not) and
  (.production_source.production_source_imports_oracle | not) and
  .production_source.orbit_carrier_present
' "$EVIDENCE/reexecuted_oracle.json" >/dev/null

if rg -n 'import[[:space:]]+growing_prime|from[[:space:]]+growing_prime' \
  "$EVIDENCE/growing_prime_two_fiber_cubic_airy_public_orbit_span_independent_oracle.py"; then
  printf '%s\n' 'oracle dependency isolation failed' >&2
  exit 1
fi

sha256sum \
  "$EVIDENCE/growing_prime_two_fiber_cubic_airy_public_orbit_span.py" \
  "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank.py" \
  "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span.py" \
  "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py" \
  "$EVIDENCE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py" \
  "$EVIDENCE/growing_prime_two_fiber_cubic_airy_public_orbit_span_independent_oracle.py" \
  "$EVIDENCE/reexecuted.json" \
  "$EVIDENCE/reexecuted_oracle.json" > "$EVIDENCE/SHA256SUMS"

printf '%s\n' 'QUALIFIED_GROWING_PRIME_TWO_FIBER_CUBIC_AIRY_PUBLIC_ORBIT_SPAN_STRICT_SCOPE'
printf 'evidence=%s\n' "$EVIDENCE"
