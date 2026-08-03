#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
DIR="$ROOT/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/audio_frequency_wave_substrate/cat_cas_phase_frontier"
PYTHON="$ROOT/.venv/bin/python"
RUN_DIR="$(mktemp -d /dev/shm/ags-audio-m153-qualifier-XXXXXX)"
PRODUCTION="$DIR/f103_c17_rank1_open_relation_closure_no_go.py"
ORACLE="$DIR/f103_c17_rank1_open_relation_closure_no_go_oracle.py"
SEALED_PRODUCTION="$DIR/F103_C17_RANK1_OPEN_RELATION_CLOSURE_NO_GO_RESULTS.json"
SEALED_ORACLE="$DIR/F103_C17_RANK1_OPEN_RELATION_CLOSURE_NO_GO_ORACLE_RESULTS.json"

test "$(git branch --show-current)" = "codex/audio-frequency-wave-substrate"
printf '%s  %s\n' 'fc10cd9f86c8f63459d1c470b04d2dffbf937d662373dd0eaf7bf6d42907713d' "$PRODUCTION" | sha256sum -c -
printf '%s  %s\n' '706731942bc640c4e9c562012a36a5a4c75c8b131d940dbf3144f28592fa5470' "$ORACLE" | sha256sum -c -
if rg -n '(^|[[:space:]])(import|from)[[:space:]].*(f103_c17_rank1_open_relation_closure_no_go|numpy)' "$ORACLE"; then
  echo 'oracle imports production or NumPy' >&2
  exit 1
fi

nice -n 10 "$PYTHON" -X dev "$PRODUCTION" --output "$RUN_DIR/production.json"
cmp "$RUN_DIR/production.json" "$SEALED_PRODUCTION"
nice -n 10 "$PYTHON" -X dev "$ORACLE" --production "$RUN_DIR/production.json" --output "$RUN_DIR/oracle.json"
cmp "$RUN_DIR/oracle.json" "$SEALED_ORACLE"

jq -e '
  .execution_scope.case_count == 18 and .execution_scope.all_cases_exact and
  .relation_law.translation_invariant == false and
  .relation_law.rank1_factor_coordinates_per_relation == 34 and
  .relation_law.materialized_dense_relation_table_cells == 0 and
  .relation_law.rank2_escape_certificate.exact_rank == 2 and
  .carrier_law.resident_field_coordinates == 612 and
  .carrier_law.machine_enforced_generation_or_lease_custody == false and
  .matched_classical_recurrence.resident_target_field_coordinates == 306 and
  .matched_classical_recurrence.phase_to_classical_resident_dimension_ratio == 2 and
  .matched_classical_recurrence.maximum_relation_field_multiplications_each == 278528 and
  .restoration.carrier_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .restoration.inverse_history_cells == 0 and .restoration.snapshot_reload_used == false and
  ([.controls[]] | all) and
  (.not_established | index("CATVM_CUSTODY")) != null and
  (.not_established | index("DISTINCT_PHASE_RESOURCE")) != null
' "$RUN_DIR/production.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .independence.imports_production == false and .independence.imports_numpy == false and
  .exact_case_reexecutions == 18 and .case_field_comparisons == 414 and
  .all_target_and_boundary_matches and .all_exact_restorations and
  .rank2_escape_certificate.exact_rank == 2 and
  .resource_law.phase_resident_coordinates == 612 and
  .resource_law.rematerialized_classical_resident_coordinates == 306 and
  .decision == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
' "$RUN_DIR/oracle.json" >/dev/null

echo "QUALIFIED_F103_C17_RANK1_OPEN_RELATION_CLOSURE_NO_GO_STRICT_SCOPE"
echo "evidence=$RUN_DIR"
