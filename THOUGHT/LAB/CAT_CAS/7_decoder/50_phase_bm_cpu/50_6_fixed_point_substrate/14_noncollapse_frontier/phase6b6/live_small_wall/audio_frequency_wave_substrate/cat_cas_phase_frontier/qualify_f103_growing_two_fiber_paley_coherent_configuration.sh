#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/f103_growing_two_fiber_paley_coherent_configuration.py"
ORACLE="$HERE/f103_growing_two_fiber_paley_coherent_configuration_oracle.py"
SEALED="$HERE/F103_GROWING_TWO_FIBER_PALEY_COHERENT_CONFIGURATION_RESULTS.json"
SEALED_ORACLE="$HERE/F103_GROWING_TWO_FIBER_PALEY_COHERENT_CONFIGURATION_ORACLE_RESULTS.json"
EVIDENCE="$(mktemp -d /dev/shm/ags-audio-m169-qualifier-XXXXXX)"

cp "$PRODUCTION" "$EVIDENCE/production.py"
cp "$ORACLE" "$EVIDENCE/oracle.py"
cp "$SEALED" "$EVIDENCE/sealed.json"
cp "$SEALED_ORACLE" "$EVIDENCE/sealed_oracle.json"
PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" -m py_compile "$EVIDENCE/production.py" "$EVIDENCE/oracle.py"
"$PYTHON" "$EVIDENCE/production.py" --output "$EVIDENCE/reexecuted.json"
cmp "$EVIDENCE/sealed.json" "$EVIDENCE/reexecuted.json"
"$PYTHON" "$EVIDENCE/oracle.py" --production-results "$EVIDENCE/reexecuted.json" --output "$EVIDENCE/reexecuted_oracle.json"
cmp "$EVIDENCE/sealed_oracle.json" "$EVIDENCE/reexecuted_oracle.json"

jq -e '
  .fixed_rank_law.fixed12_across_all_declared_orders_and_depths and
  .fixed_rank_law.resident_coefficients_per_port == 12 and
  .fixed_rank_law.maximum_represented_vertices == 194 and
  (.cases | length) == 10 and
  ([.cases[] | .exact_canonical_state_restored and .same_backing_restored and (.snapshot_used | not)] | all) and
  ([.controls[]] | all) and
  .restoration_and_reuse.restoration_generation == 2 and
  .restoration_and_reuse.same_backing_reused and
  .restoration_and_reuse.unrelated_second_boundary_matches_fresh and
  .matched_baselines.strongest_law_identical_to_accepted and
  (.resource_accounting.advantage_claimed | not)
' "$EVIDENCE/reexecuted.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  (.production_module_imported | not) and
  (.production_projection_called | not) and
  (.case_comparisons | length) == 10 and
  ([.case_comparisons[] | .boundary_matches and .commitment_matches and .checkpoints_match and .independent_restoration_exact] | all) and
  ([.checks[]] | all)
' "$EVIDENCE/reexecuted_oracle.json" >/dev/null

if rg -n 'import[[:space:]]+f103_growing_two_fiber|from[[:space:]]+f103_growing_two_fiber' "$EVIDENCE/oracle.py"; then
  printf '%s\n' 'oracle dependency isolation failed' >&2
  exit 1
fi

sha256sum "$EVIDENCE/production.py" "$EVIDENCE/oracle.py" "$EVIDENCE/reexecuted.json" "$EVIDENCE/reexecuted_oracle.json" > "$EVIDENCE/SHA256SUMS"
printf '%s\n' 'QUALIFIED_F103_GROWING_TWO_FIBER_PALEY_COHERENT_CONFIGURATION_STRICT_SCOPE'
printf 'evidence=%s\n' "$EVIDENCE"
