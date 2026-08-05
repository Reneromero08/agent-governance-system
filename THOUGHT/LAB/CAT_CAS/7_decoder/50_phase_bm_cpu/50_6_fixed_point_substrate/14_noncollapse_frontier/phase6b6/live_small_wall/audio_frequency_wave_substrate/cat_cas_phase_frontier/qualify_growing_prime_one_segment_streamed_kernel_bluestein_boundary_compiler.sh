#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/growing_prime_one_segment_streamed_kernel_bluestein_boundary_compiler.py"
ORACLE="$HERE/growing_prime_one_segment_streamed_kernel_bluestein_boundary_compiler_independent_oracle.py"
SEALED="$HERE/GROWING_PRIME_ONE_SEGMENT_STREAMED_KERNEL_BLUESTEIN_BOUNDARY_COMPILER_RESULTS.json"
SEALED_ORACLE="$HERE/GROWING_PRIME_ONE_SEGMENT_STREAMED_KERNEL_BLUESTEIN_BOUNDARY_COMPILER_INDEPENDENT_ORACLE.json"
REVIEW="$HERE/GROWING_PRIME_ONE_SEGMENT_STREAMED_KERNEL_BLUESTEIN_BOUNDARY_COMPILER_INDEPENDENT_REEXECUTION_REVIEW.md"

export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX

cmp "$SEALED" <("$PYTHON" "$PRODUCTION")
cmp "$SEALED_ORACLE" <("$PYTHON" "$ORACLE")

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.transaction_cases | length) == 14 and
  ([.transaction_cases[] |
    .layout.resident_scalar_cells == 1 and
    .layout.total_carrier_field_cells == (1 + .layout.transform_segment_cells) and
    .m184_carrier_field_cells == (1 + 2 * .layout.transform_segment_cells) and
    .primary.forward_counts.transform.ntt_calls == 4 and
    .primary.forward_counts.transform.ntt_butterflies ==
      (2 * .layout.transform_segment_cells * (.layout.transform_segment_cells | log2 | floor)) and
    .primary.forward_counts.streamed_kernel_spectrum_values ==
      (2 * .layout.transform_segment_cells) and
    .primary.forward_counts.streamed_kernel_nonzero_terms ==
      (2 * .layout.transform_segment_cells * (2 * (.q - 1) - 1)) and
    .primary.forward_counts == .primary.inverse_counts and
    .primary.exactly_restored and
    .primary.same_backing and
    .primary.projected_result_survives_inverse and
    .verification_only.executed_before_carrier_allocation and
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
  .controls.all_boundaries_match_descriptor_reference and
  .controls.all_streamed_kernel_spectra_nonzero and
  .controls.all_one_segment_workspaces_clear and
  .controls.all_missing_inverses_fail and
  .controls.all_wrong_program_inverses_fail and
  .controls.frequency_zero_omission_changes_all_declared_boundaries and
  .controls.all_forced_singular_kernel_spectra_rejected and
  .controls.all_null_carriers_rejected and
  .controls.all_same_backing_restorations_pass and
  .controls.all_unrelated_reuses_match_fresh and
  (.matched_baselines.state_advantage | not) and
  (.matched_baselines.work_advantage | not) and
  (.matched_baselines.new_asymptotic_pareto_point | not) and
  ([.strict_boundaries[]] | all(. == false))
' "$SEALED" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.oracle_cases | length) == 14 and
  ([.oracle_cases[] |
    .logical_carrier_field_cells == (1 + .convolution_width) and
    .one_segment_gauss_matches_direct and
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
    .expected_ntt_calls_per_compiler == 4 and
    .expected_butterflies_per_compiler ==
      (2 * .convolution_width * (.convolution_width | log2 | floor)) and
    .expected_streamed_kernel_values_per_compiler == (2 * .convolution_width) and
    .expected_streamed_kernel_terms_per_compiler ==
      (2 * .convolution_width * (2 * (.q - 1) - 1))] | all) and
  .controls.q5_q7_direct_original_relation_boundaries_match and
  .observed_resource_law.oracle_materializes_kernel_seed_for_verification_only and
  .observed_resource_law.oracle_uses_recursive_out_of_place_transform_buffers and
  .observed_resource_law.oracle_reports_production_logical_schedule_not_oracle_peak_memory and
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
    assert left["layout"]["transform_segment_cells"] == right["convolution_width"]
    assert left["layout"]["total_carrier_field_cells"] == right["logical_carrier_field_cells"]
    assert left["carrier_capacity_bits"] == right["carrier_capacity_bits"]
    assert left["primary"]["streamed_kernel_spectrum_commitment"] == right["streamed_kernel_spectrum_commitment"]
    assert left["primary"]["resident_scalar_commitment"] == right["resident_scalar_commitment"]
    assert left["primary"]["projected_final_boundary_scalar"] == right["projected_boundary_scalar"]
    assert left["unrelated_reuse"]["restored_boundary"] == right["reused_boundary_scalar"]
    assert left["unrelated_reuse"]["fresh_boundary"] == right["fresh_boundary_scalar"]
    assert left["primary"]["forward_counts"]["transform"]["ntt_calls"] == right["expected_ntt_calls_per_compiler"]
    assert left["primary"]["forward_counts"]["transform"]["ntt_butterflies"] == right["expected_butterflies_per_compiler"]
    assert left["primary"]["forward_counts"]["streamed_kernel_spectrum_values"] == right["expected_streamed_kernel_values_per_compiler"]
    assert left["primary"]["forward_counts"]["streamed_kernel_nonzero_terms"] == right["expected_streamed_kernel_terms_per_compiler"]
    if right["direct_relation_check"] is not None:
        assert right["direct_relation_check"]["matches_one_segment_boundary"]
PY

if rg -n \
  'import[[:space:]]+growing_prime_one_segment|from[[:space:]]+growing_prime_one_segment' \
  "$ORACLE"; then
  printf '%s\n' 'oracle production-dependency isolation failed' >&2
  exit 1
fi

sha256sum "$PRODUCTION" "$ORACLE" "$SEALED" "$SEALED_ORACLE" "$REVIEW"
printf '%s\n' 'QUALIFIED_GROWING_PRIME_ONE_SEGMENT_STREAMED_KERNEL_BLUESTEIN_BOUNDARY_COMPILER_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$HERE"
