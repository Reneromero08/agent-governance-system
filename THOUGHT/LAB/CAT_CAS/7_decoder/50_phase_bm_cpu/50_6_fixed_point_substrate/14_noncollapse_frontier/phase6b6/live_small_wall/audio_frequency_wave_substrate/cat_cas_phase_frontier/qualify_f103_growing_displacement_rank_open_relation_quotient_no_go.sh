#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/f103_growing_displacement_rank_open_relation_quotient_no_go.py"
ORACLE="$HERE/f103_growing_displacement_rank_open_relation_quotient_no_go_oracle.py"
SEALED="$HERE/F103_GROWING_DISPLACEMENT_RANK_OPEN_RELATION_QUOTIENT_NO_GO_RESULTS.json"
SEALED_ORACLE="$HERE/F103_GROWING_DISPLACEMENT_RANK_OPEN_RELATION_QUOTIENT_NO_GO_ORACLE_RESULTS.json"
EVIDENCE="$(mktemp -d /dev/shm/ags-audio-m155-qualifier-XXXXXX)"

cp "$PRODUCTION" "$EVIDENCE/production.py"
cp "$ORACLE" "$EVIDENCE/oracle.py"
cp "$SEALED" "$EVIDENCE/sealed.json"
cp "$SEALED_ORACLE" "$EVIDENCE/sealed_oracle.json"

"$PYTHON" -m py_compile "$EVIDENCE/production.py" "$EVIDENCE/oracle.py"
"$PYTHON" "$EVIDENCE/production.py" --output "$EVIDENCE/reexecuted.json"
cmp "$EVIDENCE/sealed.json" "$EVIDENCE/reexecuted.json"
"$PYTHON" "$EVIDENCE/oracle.py" \
  --production "$EVIDENCE/reexecuted.json" \
  --output "$EVIDENCE/reexecuted_oracle.json"

test "$(sha256sum "$EVIDENCE/production.py" | awk '{print $1}')" = \
  "$(jq -r .source_sha256 "$EVIDENCE/reexecuted.json")"
test "$(sha256sum "$EVIDENCE/oracle.py" | awk '{print $1}')" = \
  "$(jq -r .oracle_source_sha256 "$EVIDENCE/reexecuted_oracle.json")"
test "$(sha256sum "$EVIDENCE/reexecuted.json" | awk '{print $1}')" = \
  "$(jq -r .production_result_sha256 "$EVIDENCE/reexecuted_oracle.json")"

jq -e '
  .execution_scope.case_count == 40 and
  .execution_scope.interfaces == [5,7,11,17] and
  .relation_law.maximum_displacement_rank_by_interface == {"5":5,"7":7,"11":11,"17":17} and
  (.relation_law.uniform_interface_independent_rank_bound_observed | not) and
  .relation_law.full_displacement_rank_reached_after_one_layer_at_every_interface_and_family and
  .relation_law.ordinary_relation_tables_materialized_on_phase_path == 0 and
  .matched_classical_recurrence.full_relation_and_boundary_match_every_case and
  .restoration.carrier_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .restoration.same_backing and
  (.restoration.snapshot_reload_used | not) and
  .restoration.inverse_history_cells == 0 and
  .restoration.retained_restoration_baseline_cells == 0 and
  .restoration.unrelated_program_reuse.second_boundary_matches_fresh and
  .restoration.unrelated_program_reuse.resource_signature_matches_fresh and
  .restoration.repeated_reuse.exact_restoration and
  ([.controls[]] | all)
' "$EVIDENCE/reexecuted.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  (.imports_production_module | not) and
  (.imports_numpy | not) and
  .case_count == 40 and
  .comparison_count == 1628 and
  ([.controls[]] | all) and
  (.package_local_fields_not_independently_recounted | length) == 3
' "$EVIDENCE/reexecuted_oracle.json" >/dev/null

if rg -n 'import[[:space:]]+f103_growing_displacement|from[[:space:]]+f103_growing_displacement|import[[:space:]]+numpy|from[[:space:]]+numpy' "$EVIDENCE/oracle.py"; then
  printf '%s\n' 'oracle dependency isolation failed' >&2
  exit 1
fi

if jq -e '.cases[] | has("boundary") or has("final_charts")' "$EVIDENCE/reexecuted.json" >/dev/null; then
  printf '%s\n' 'sealed production output smuggles boundary or resident chart' >&2
  exit 1
fi

sha256sum \
  "$EVIDENCE/production.py" \
  "$EVIDENCE/oracle.py" \
  "$EVIDENCE/reexecuted.json" \
  "$EVIDENCE/reexecuted_oracle.json" \
  > "$EVIDENCE/SHA256SUMS"

printf '%s\n' 'QUALIFIED_F103_GROWING_DISPLACEMENT_RANK_OPEN_RELATION_QUOTIENT_NO_GO_STRICT_SCOPE'
printf 'evidence=%s\n' "$EVIDENCE"
