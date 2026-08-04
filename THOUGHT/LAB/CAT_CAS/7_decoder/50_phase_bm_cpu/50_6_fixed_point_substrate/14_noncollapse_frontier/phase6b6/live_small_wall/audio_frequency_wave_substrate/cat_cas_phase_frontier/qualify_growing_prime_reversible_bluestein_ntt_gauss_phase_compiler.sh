#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/growing_prime_reversible_bluestein_ntt_gauss_phase_compiler.py"
PREDECESSOR="$HERE/growing_prime_mellin_gauss_streamed_recurrence_rank.py"
ORACLE="$HERE/growing_prime_reversible_bluestein_ntt_gauss_phase_compiler_independent_oracle.py"
SEALED="$HERE/GROWING_PRIME_REVERSIBLE_BLUESTEIN_NTT_GAUSS_PHASE_COMPILER_RESULTS.json"
SEALED_ORACLE="$HERE/GROWING_PRIME_REVERSIBLE_BLUESTEIN_NTT_GAUSS_PHASE_COMPILER_INDEPENDENT_ORACLE.json"
REVIEW="$HERE/GROWING_PRIME_REVERSIBLE_BLUESTEIN_NTT_GAUSS_PHASE_COMPILER_INDEPENDENT_REEXECUTION_REVIEW.md"
EVIDENCE="$(mktemp -d /dev/shm/ags-audio-m183-qualifier-XXXXXX)"

cp "$PRODUCTION" "$EVIDENCE/production.py"
cp "$PREDECESSOR" "$EVIDENCE/growing_prime_mellin_gauss_streamed_recurrence_rank.py"
cp "$ORACLE" "$EVIDENCE/oracle.py"
cp "$SEALED" "$EVIDENCE/sealed.json"
cp "$SEALED_ORACLE" "$EVIDENCE/sealed_oracle.json"
cp "$REVIEW" "$EVIDENCE/review.md"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" -m py_compile \
  "$EVIDENCE/production.py" \
  "$EVIDENCE/oracle.py"

PYTHONPATH="$EVIDENCE" PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/production.py" --output "$EVIDENCE/reexecuted.json"
cmp "$EVIDENCE/sealed.json" "$EVIDENCE/reexecuted.json"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/oracle.py" --output "$EVIDENCE/reexecuted_oracle.json"
cmp "$EVIDENCE/sealed_oracle.json" "$EVIDENCE/reexecuted_oracle.json"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.declared_fields | length) == 14 and
  (.transaction_cases | length) == 14 and
  ([.transaction_cases[] |
    .layout.descriptor_cells == (.q - 1) and
    .layout.total_carrier_field_cells == ((.q - 1) + 3 * .layout.scratch_segment_cells) and
    .primary.forward_counts.ntt_calls == 6 and
    .primary.forward_counts.ntt_butterflies ==
      (3 * .layout.scratch_segment_cells * (.layout.scratch_segment_cells | log2 | floor)) and
    .primary.exactly_restored and
    .primary.same_backing and
    .primary.projected_result_survives_inverse and
    .primary.projection_counts.channels == (.q - 1) and
    .primary.projection_counts.character_calls == 3 and
    .primary.projection_counts.character_orbit_visits <= (3 * (.q - 1)) and
    .unrelated_reuse.actual_restored_backing_consumed and
    .unrelated_reuse.fresh_restored_boundary_agrees and
    .unrelated_reuse.fresh_restored_resource_signature_agrees and
    .unrelated_reuse.exactly_restored_again and
    .unrelated_reuse.same_backing_after_reuse and
    .controls.missing_inverse_fails and
    .controls.wrong_additive_phase_inverse_fails and
    .controls.omitted_frequency_inverse_fails and
    .controls.null_carrier_rejected and
    (.controls.snapshot_used | not) and
    (.controls.generation_or_lease_metadata_enforced | not)] | all) and
  .controls.all_direct_gauss_tables_match and
  .controls.all_scratch_segments_zero_after_forward and
  .controls.all_projection_character_scans_are_linear and
  .controls.all_same_backing_restorations_pass and
  .controls.all_unrelated_reuses_match_fresh and
  (.matched_baselines.computational_advantage_established | not) and
  ([.strict_boundaries[]] | all(. == false))
' "$EVIDENCE/reexecuted.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.oracle_independence.imports_production | not) and
  (.oracle_independence.imports_predecessor | not) and
  .oracle_independence.field_roots_reconstructed and
  .oracle_independence.uses_out_of_place_recursive_ntt and
  (.oracle_independence.uses_production_iterative_ntt | not) and
  .oracle_independence.direct_gauss_tables_reconstructed and
  .oracle_independence.direct_original_relation_checks == 2 and
  (.oracle_cases | length) == 14 and
  ([.oracle_cases[] |
    .descriptor_matches_direct_gauss and
    .projected_result_survives_inverse and
    .exact_same_backing_restoration and
    .actual_restored_backing_reuse_matches_fresh and
    .missing_inverse_fails and
    .wrong_additive_phase_inverse_fails and
    .omitted_frequency_inverse_fails and
    .expected_ntt_calls_per_compiler == 6] | all) and
  .controls.q5_q7_direct_original_relation_boundaries_match
' "$EVIDENCE/reexecuted_oracle.json" >/dev/null

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" - \
  "$EVIDENCE/reexecuted.json" "$EVIDENCE/reexecuted_oracle.json" <<'PY'
import json
import sys

production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
assert production["claim"] == oracle["claim"]
assert len(production["transaction_cases"]) == len(oracle["oracle_cases"]) == 14
for left, right in zip(production["transaction_cases"], oracle["oracle_cases"]):
    assert left["q"] == right["q"]
    assert left["auxiliary_prime"] == right["auxiliary_prime"]
    assert left["layout"]["scratch_segment_cells"] == right["convolution_width"]
    assert left["layout"]["total_carrier_field_cells"] == right["carrier_field_cells"]
    assert left["primary"]["resident_descriptor_commitment"] == right["resident_commitment"]
    assert left["primary"]["projected_final_boundary_scalar"] == right["projected_boundary_scalar"]
    assert left["unrelated_reuse"]["restored_boundary"] == right["reused_boundary_scalar"]
    assert left["unrelated_reuse"]["fresh_boundary"] == right["fresh_boundary_scalar"]
    assert left["primary"]["forward_counts"]["ntt_calls"] == right["expected_ntt_calls_per_compiler"]
    assert left["primary"]["forward_counts"]["ntt_butterflies"] == right["expected_butterflies_per_compiler"]
    if right["direct_relation_check"] is not None:
        assert right["direct_relation_check"]["matches_compiled_boundary"]
PY

if rg -n \
  'import[[:space:]]+growing_prime_|from[[:space:]]+growing_prime_' \
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

printf '%s\n' 'QUALIFIED_GROWING_PRIME_REVERSIBLE_BLUESTEIN_NTT_GAUSS_PHASE_COMPILER_STRICT_SCOPE'
printf 'evidence=%s\n' "$EVIDENCE"
