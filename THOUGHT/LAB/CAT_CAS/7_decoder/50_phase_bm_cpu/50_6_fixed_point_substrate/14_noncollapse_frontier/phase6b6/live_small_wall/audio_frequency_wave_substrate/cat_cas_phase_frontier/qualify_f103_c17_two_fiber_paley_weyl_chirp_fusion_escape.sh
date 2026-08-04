#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/f103_c17_two_fiber_paley_weyl_chirp_fusion_escape.py"
ORACLE="$HERE/f103_c17_two_fiber_paley_weyl_chirp_fusion_escape_independent_oracle.py"
SEALED="$HERE/F103_C17_TWO_FIBER_PALEY_WEYL_CHIRP_FUSION_ESCAPE_RESULTS.json"
SEALED_ORACLE="$HERE/F103_C17_TWO_FIBER_PALEY_WEYL_CHIRP_FUSION_ESCAPE_INDEPENDENT_ORACLE.json"
EVIDENCE="$(mktemp -d /dev/shm/ags-audio-m170-qualifier-XXXXXX)"

cp "$PRODUCTION" "$EVIDENCE/production.py"
cp "$ORACLE" "$EVIDENCE/oracle.py"
cp "$SEALED" "$EVIDENCE/sealed.json"
cp "$SEALED_ORACLE" "$EVIDENCE/sealed_oracle.json"
PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" -m py_compile "$EVIDENCE/production.py" "$EVIDENCE/oracle.py"
"$PYTHON" "$EVIDENCE/production.py" --output "$EVIDENCE/reexecuted.json"
cmp "$EVIDENCE/sealed.json" "$EVIDENCE/reexecuted.json"
"$PYTHON" "$EVIDENCE/oracle.py" \
  --production-result "$EVIDENCE/reexecuted.json" \
  --output "$EVIDENCE/reexecuted_oracle.json"
cmp "$EVIDENCE/sealed_oracle.json" "$EVIDENCE/reexecuted_oracle.json"

jq -e '
  .all_observed_post_update_states_use_at_least_2278_of_2312_cells and
  .observed_two_port_active_weyl_cell_range_after_first_update ==
    {"capacity":2312,"maximum":2308,"minimum":2278} and
  (.cases | length) == 10 and
  ([.cases[] | .exact_canonical_state_restored and .same_backing_restored and (.snapshot_used | not)] | all) and
  ([.controls[]] | all) and
  ([.algebra_checks[]] | all) and
  .restoration_and_reuse.restoration_generation == 2 and
  .restoration_and_reuse.same_backing_reused and
  .restoration_and_reuse.unrelated_second_boundary_matches_fresh and
  .matched_baselines.strongest_law_identical_to_accepted and
  (.resource_accounting.advantage_claimed | not)
' "$EVIDENCE/reexecuted.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  (.production_imported | not) and
  .all_ten_case_boundaries_commitments_support_histories_and_restoration_match and
  .controls_match_production and
  .reuse_matches_production and
  .support_range_matches_production and
  .fusion_escape_attack.fixed12_fusion_representation_rejected_by_x_dependence and
  (.case_comparisons | length) == 10 and
  ([.case_comparisons[] |
    .boundary_matches and
    .semantic_commitment_matches and
    .support_history_matches and
    .final_support_matches and
    .exact_restoration_matches and
    .same_backing_matches and
    .generation_matches] | all)
' "$EVIDENCE/reexecuted_oracle.json" >/dev/null

if rg -n 'import[[:space:]]+f103_c17_two_fiber|from[[:space:]]+f103_c17_two_fiber' "$EVIDENCE/oracle.py"; then
  printf '%s\n' 'oracle dependency isolation failed' >&2
  exit 1
fi

sha256sum "$EVIDENCE/production.py" "$EVIDENCE/oracle.py" "$EVIDENCE/reexecuted.json" "$EVIDENCE/reexecuted_oracle.json" > "$EVIDENCE/SHA256SUMS"
printf '%s\n' 'QUALIFIED_F103_C17_TWO_FIBER_PALEY_WEYL_CHIRP_FUSION_ESCAPE_STRICT_SCOPE'
printf 'evidence=%s\n' "$EVIDENCE"
