#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/f103_growing_dihedral_irrep_sparse_relation_algebra.py"
ORACLE="$HERE/f103_growing_dihedral_irrep_sparse_relation_oracle.py"
SEALED="$HERE/F103_GROWING_DIHEDRAL_IRREP_SPARSE_RELATION_RESULTS.json"
SEALED_ORACLE="$HERE/F103_GROWING_DIHEDRAL_IRREP_SPARSE_RELATION_ORACLE_RESULTS.json"
EVIDENCE="$(mktemp -d /dev/shm/ags-audio-m168-qualifier-XXXXXX)"

cp "$PRODUCTION" "$EVIDENCE/production.py"
cp "$ORACLE" "$EVIDENCE/oracle.py"
cp "$SEALED" "$EVIDENCE/sealed.json"
cp "$SEALED_ORACLE" "$EVIDENCE/sealed_oracle.json"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" -m py_compile "$EVIDENCE/production.py" "$EVIDENCE/oracle.py"
"$PYTHON" "$EVIDENCE/production.py" --output "$EVIDENCE/reexecuted.json"
cmp "$EVIDENCE/sealed.json" "$EVIDENCE/reexecuted.json"
"$PYTHON" "$EVIDENCE/oracle.py" \
  --production-results "$EVIDENCE/reexecuted.json" \
  --output "$EVIDENCE/reexecuted_oracle.json"
cmp "$EVIDENCE/sealed_oracle.json" "$EVIDENCE/reexecuted_oracle.json"

jq -e '
  .classification == "SOURCE_AUDITED_PACKAGE_LOCAL" and
  .verification_level == "PACKAGE_SELF_REVIEW" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .all_primary_cases_reach_full_two_port_irrep_block_capacity_by_depth16 and
  (.cases | length) == 6 and
  ([.cases[] |
    .matches_streamed_group_coordinate_boundary and
    .matches_streamed_group_coordinate_commitment and
    .group_coordinate_reference_restores_exactly and
    .exact_canonical_state_restored and
    .same_backing_restored and
    (.snapshot_used | not) and
    (.hidden_relation_values_serialized | not)
  ] | all) and
  ([.controls[]] | all) and
  ([.algebra_checks[]] | all) and
  .restoration_and_reuse.same_backing_reused and
  .restoration_and_reuse.exact_canonical_state_restored_after_reuse and
  .restoration_and_reuse.unrelated_second_boundary_matches_fresh and
  .restoration_and_reuse.unrelated_second_commitment_matches_fresh and
  .restoration_and_reuse.restoration_generation == 2 and
  (.restoration_and_reuse.snapshot_used | not) and
  .matched_baselines.strongest_law_identical_to_accepted and
  .matched_baselines.executed_group_coordinate_matches_all_cases and
  (.resource_accounting.advantage_claimed | not)
' "$EVIDENCE/reexecuted.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.production_module_imported | not) and
  (.production_projection_called | not) and
  (.case_comparisons | length) == 6 and
  ([.case_comparisons[] |
    .boundary_matches and
    .commitment_matches and
    .support_history_matches and
    .final_support_matches and
    .independent_restoration_exact
  ] | all) and
  ([.checks[]] | all)
' "$EVIDENCE/reexecuted_oracle.json" >/dev/null

if rg -n 'import[[:space:]]+f103_growing_dihedral|from[[:space:]]+f103_growing_dihedral' "$EVIDENCE/oracle.py"; then
  printf '%s\n' 'oracle dependency isolation failed' >&2
  exit 1
fi

sha256sum \
  "$EVIDENCE/production.py" \
  "$EVIDENCE/oracle.py" \
  "$EVIDENCE/reexecuted.json" \
  "$EVIDENCE/reexecuted_oracle.json" \
  > "$EVIDENCE/SHA256SUMS"

printf '%s\n' 'QUALIFIED_F103_GROWING_DIHEDRAL_IRREP_SPARSE_RELATION_NO_GO_STRICT_SCOPE'
printf 'evidence=%s\n' "$EVIDENCE"
