#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/growing_prime_fused_two_segment_bluestein_boundary_compiler.py"
ORACLE="$HERE/growing_prime_fused_two_segment_bluestein_boundary_compiler_independent_oracle.py"
SEALED="$HERE/GROWING_PRIME_FUSED_TWO_SEGMENT_BLUESTEIN_BOUNDARY_COMPILER_RESULTS.json"
SEALED_ORACLE="$HERE/GROWING_PRIME_FUSED_TWO_SEGMENT_BLUESTEIN_BOUNDARY_COMPILER_INDEPENDENT_ORACLE.json"
REVIEW="$HERE/GROWING_PRIME_FUSED_TWO_SEGMENT_BLUESTEIN_BOUNDARY_COMPILER_INDEPENDENT_REEXECUTION_REVIEW.md"

export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX

# Reexecute directly in the assigned worktree.  Process substitution keeps the
# comparison ephemeral without a temp directory, extra checkout, or RAM path.
cmp "$SEALED" <("$PYTHON" "$PRODUCTION")
cmp "$SEALED_ORACLE" <("$PYTHON" "$ORACLE")

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.transaction_cases | length) == 14 and
  ([.transaction_cases[] |
    .layout.resident_scalar_cells == 1 and
    .layout.total_carrier_field_cells == (1 + 2 * .layout.scratch_segment_cells) and
    .m183_carrier_field_cells == ((.q - 1) + 3 * .layout.scratch_segment_cells) and
    .primary.forward_counts.transform.ntt_calls == 6 and
    .primary.forward_counts.transform.ntt_butterflies ==
      (3 * .layout.scratch_segment_cells * (.layout.scratch_segment_cells | log2 | floor)) and
    .primary.forward_counts.in_place_spectral_multiplications ==
      (2 * .layout.scratch_segment_cells) and
    .primary.forward_counts.public_kernel_spectrum_inversions ==
      .layout.scratch_segment_cells and
    .primary.forward_counts == .primary.inverse_counts and
    .primary.exactly_restored and
    .primary.same_backing and
    .primary.projected_result_survives_inverse and
    .primary.projection_counts.channels == (.q - 1) and
    .primary.projection_counts.character_calls == 3 and
    .primary.projection_counts.character_orbit_visits <= (3 * (.q - 1)) and
    .verification_only.executed_before_carrier_allocation and
    .verification_only.direct_gauss_table_peak_field_cells == (.q - 1) and
    .verification_only.direct_gauss_generation_field_terms == ((.q - 1) * (.q - 1)) and
    (.verification_only.included_in_accepted_transaction_carrier | not) and
    .unrelated_reuse.actual_restored_backing_consumed and
    .unrelated_reuse.fresh_restored_boundary_agrees and
    .unrelated_reuse.fresh_restored_resource_signature_agrees and
    .unrelated_reuse.exactly_restored_again and
    .unrelated_reuse.same_backing_after_reuse and
    .controls.missing_inverse_fails and
    .controls.wrong_program_inverse_fails and
    .controls.frequency_zero_omission_changes_boundary and
    .controls.singular_kernel_spectrum_rejected and
    (.controls.singular_rejection_atomic_rollback_claimed | not) and
    .controls.null_carrier_rejected and
    (.controls.snapshot_used | not) and
    (.controls.generation_or_lease_metadata_enforced | not)] | all) and
  .controls.all_boundaries_match_m183_descriptor_reference and
  .controls.all_public_kernel_spectra_nonzero and
  .controls.all_two_segment_scratch_clears and
  .controls.all_missing_inverses_fail and
  .controls.all_wrong_program_inverses_fail and
  .controls.frequency_zero_omission_changes_all_declared_boundaries and
  .controls.all_forced_singular_kernel_spectra_rejected and
  .controls.all_null_carriers_rejected and
  .controls.all_same_backing_restorations_pass and
  .controls.all_unrelated_reuses_match_fresh and
  (.matched_baselines.state_advantage | not) and
  (.matched_baselines.work_advantage | not) and
  ([.strict_boundaries[]] | all(. == false))
' "$SEALED" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.oracle_independence.imports_production | not) and
  (.oracle_independence.imports_predecessor | not) and
  .oracle_independence.field_roots_reconstructed and
  .oracle_independence.uses_out_of_place_recursive_ntt and
  (.oracle_independence.uses_production_iterative_ntt | not) and
  .oracle_independence.direct_gauss_values_reconstructed and
  .oracle_independence.fused_two_segment_transaction_reconstructed and
  .oracle_independence.direct_original_relation_checks == 2 and
  (.oracle_cases | length) == 14 and
  ([.oracle_cases[] |
    .logical_carrier_field_cells == (1 + 2 * .convolution_width) and
    .fused_gauss_matches_direct and
    .projected_result_survives_inverse and
    .exact_same_backing_restoration and
    .actual_restored_backing_reuse_matches_fresh and
    .reuse_resource_signature_matches_fresh and
    .missing_inverse_fails and
    .wrong_program_inverse_fails and
    .frequency_zero_omission_changes_boundary and
    .singular_kernel_spectrum_rejected and
    (.singular_rejection_atomic_rollback_claimed | not) and
    .null_carrier_rejected and
    .expected_ntt_calls_per_compiler == 6 and
    .expected_butterflies_per_compiler ==
      (3 * .convolution_width * (.convolution_width | log2 | floor))] | all) and
  .controls.q5_q7_direct_original_relation_boundaries_match and
  .observed_resource_law.oracle_uses_out_of_place_recursive_transform_buffers and
  .observed_resource_law.oracle_derives_production_logical_schedule_not_oracle_peak_memory and
  ([.strict_boundaries[]] | all(. == false))
' "$SEALED_ORACLE" >/dev/null

"$PYTHON" - "$SEALED" "$SEALED_ORACLE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    production = json.load(stream)
with open(sys.argv[2], encoding="utf-8") as stream:
    oracle = json.load(stream)
assert production["claim"] == oracle["claim"]
assert len(production["transaction_cases"]) == len(oracle["oracle_cases"]) == 14
for left, right in zip(production["transaction_cases"], oracle["oracle_cases"]):
    assert left["q"] == right["q"]
    assert left["auxiliary_prime"] == right["auxiliary_prime"]
    assert left["layout"]["scratch_segment_cells"] == right["convolution_width"]
    assert left["layout"]["total_carrier_field_cells"] == right["logical_carrier_field_cells"]
    assert left["carrier_capacity_bits"] == right["carrier_capacity_bits"]
    assert left["primary"]["public_kernel_spectrum_commitment"] == right["kernel_spectrum_commitment"]
    assert left["primary"]["resident_scalar_commitment"] == right["resident_scalar_commitment"]
    assert left["primary"]["projected_final_boundary_scalar"] == right["projected_boundary_scalar"]
    assert left["unrelated_reuse"]["restored_boundary"] == right["reused_boundary_scalar"]
    assert left["unrelated_reuse"]["fresh_boundary"] == right["fresh_boundary_scalar"]
    assert left["primary"]["forward_counts"]["transform"]["ntt_calls"] == right["expected_ntt_calls_per_compiler"]
    assert left["primary"]["forward_counts"]["transform"]["ntt_butterflies"] == right["expected_butterflies_per_compiler"]
    if right["direct_relation_check"] is not None:
        assert right["direct_relation_check"]["matches_fused_boundary"]
PY

if rg -n \
  'import[[:space:]]+growing_prime_|from[[:space:]]+growing_prime_' \
  "$ORACLE"; then
  printf '%s\n' 'oracle dependency isolation failed' >&2
  exit 1
fi

sha256sum "$PRODUCTION" "$ORACLE" "$SEALED" "$SEALED_ORACLE" "$REVIEW"
printf '%s\n' 'QUALIFIED_GROWING_PRIME_FUSED_TWO_SEGMENT_BLUESTEIN_BOUNDARY_COMPILER_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$HERE"
