#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/growing_prime_hasse_davenport_jacobi_relation_rank.py"
PREDECESSOR="$HERE/growing_prime_mellin_gauss_streamed_recurrence_rank.py"
ORACLE="$HERE/growing_prime_hasse_davenport_jacobi_relation_rank_independent_oracle.py"
SEALED="$HERE/GROWING_PRIME_HASSE_DAVENPORT_JACOBI_RELATION_RANK_RESULTS.json"
SEALED_ORACLE="$HERE/GROWING_PRIME_HASSE_DAVENPORT_JACOBI_RELATION_RANK_INDEPENDENT_ORACLE.json"
REVIEW="$HERE/GROWING_PRIME_HASSE_DAVENPORT_JACOBI_RELATION_RANK_INDEPENDENT_REEXECUTION_REVIEW.md"
EVIDENCE="$(mktemp -d /dev/shm/ags-audio-m182-qualifier-XXXXXX)"

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
  .restoration_classification == "NO_RESTORATION_CLAIM" and
  (.rank_diagnostics | length) == 14 and
  (.residue_identity_diagnostics | length) == 14 and
  ([.rank_diagnostics[] |
    .hasse_free_monomial_generator_rank == .euler_phi_q_minus_1 and
    .norm_augmented_free_monomial_generator_rank == ((.euler_phi_q_minus_1 / 2) | floor) and
    .boundary_products_span_entire_hasse_quotient and
    .boundary_products_span_entire_norm_augmented_quotient and
    .missing_quadratic_relation_increases_free_rank and
    .jacobi_definitions_add_no_constraint_on_gauss_projection and
    .jacobi_augmented_free_rank == .euler_phi_q_minus_1] | all) and
  ([.residue_identity_diagnostics[] |
    .hasse_davenport_identity_checks > 0 and
    .gauss_norm_identity_checks == (.q - 2) and
    .applicable_jacobi_identity_checks == ((.q - 2) * (.q - 3)) and
    .false_overmerge_g1_equals_one_rejected] | all) and
  .controls.every_declared_hasse_davenport_identity_checked and
  .controls.every_declared_nontrivial_gauss_norm_identity_checked and
  .controls.every_applicable_declared_two_character_jacobi_identity_checked and
  (.declared_scope.answer_bearing_gauss_or_jacobi_constants_admitted | not) and
  (.matched_baseline.state_advantage | not) and
  (.matched_baseline.work_advantage | not)
' "$EVIDENCE/reexecuted.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "NO_RESTORATION_CLAIM" and
  (.oracle_independence.imports_production | not) and
  (.oracle_independence.imports_predecessor | not) and
  .oracle_independence.finite_fields_reconstructed and
  .oracle_independence.dense_fraction_rank_instead_of_sparse_basis and
  .oracle_independence.sign_reversed_relation_rows and
  .oracle_independence.character_values_scanned_without_log_table and
  (.rank_diagnostics | length) == 14 and
  (.residue_identity_diagnostics | length) == 14 and
  ([.rank_diagnostics[] |
    .hasse_free_monomial_generator_rank == .euler_phi_q_minus_1 and
    .norm_augmented_free_monomial_generator_rank == ((.euler_phi_q_minus_1 / 2) | floor) and
    .boundary_product_span_mod_hasse == .hasse_free_monomial_generator_rank and
    .boundary_product_span_mod_hasse_and_norm == .norm_augmented_free_monomial_generator_rank and
    .without_quadratic_hasse_relation_free_rank > .hasse_free_monomial_generator_rank and
    .jacobi_augmented_free_rank == .euler_phi_q_minus_1] | all) and
  ([.residue_identity_diagnostics[] |
    .hasse_davenport_identity_checks > 0 and
    .gauss_norm_identity_checks == (.q - 2) and
    .applicable_jacobi_identity_checks == ((.q - 2) * (.q - 3)) and
    .false_overmerge_rejected] | all) and
  .controls.missing_quadratic_relation_rejected_every_case and
  .controls.false_overmerge_rejected_every_case and
  .controls.boundary_products_span_every_declared_quotient
' "$EVIDENCE/reexecuted_oracle.json" >/dev/null

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" - \
  "$EVIDENCE/reexecuted.json" "$EVIDENCE/reexecuted_oracle.json" <<'PY'
import json
import sys

production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
assert production["claim"] == oracle["claim"]
for left, right in zip(production["rank_diagnostics"], oracle["rank_diagnostics"]):
    assert left["q"] == right["q"]
    for key in (
        "hasse_exact_rational_rank",
        "hasse_free_monomial_generator_rank",
        "norm_augmented_exact_rational_rank",
        "norm_augmented_free_monomial_generator_rank",
        "boundary_product_span_mod_hasse",
        "boundary_product_span_mod_hasse_and_norm",
        "without_quadratic_hasse_relation_free_rank",
        "euler_phi_q_minus_1",
        "jacobi_augmented_free_rank",
    ):
        assert left[key] == right[key], (left["q"], key)
for left, right in zip(
    production["residue_identity_diagnostics"],
    oracle["residue_identity_diagnostics"],
):
    assert left["q"] == right["q"]
    for key in (
        "hasse_davenport_identity_checks",
        "hasse_davenport_value_commitment",
        "gauss_norm_identity_checks",
        "applicable_jacobi_identity_checks",
        "streamed_jacobi_character_terms",
        "jacobi_value_commitment",
    ):
        assert left[key] == right[key], (left["q"], key)
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

printf '%s\n' 'QUALIFIED_GROWING_PRIME_HASSE_DAVENPORT_JACOBI_RELATION_RANK_STRICT_SCOPE'
printf 'evidence=%s\n' "$EVIDENCE"
