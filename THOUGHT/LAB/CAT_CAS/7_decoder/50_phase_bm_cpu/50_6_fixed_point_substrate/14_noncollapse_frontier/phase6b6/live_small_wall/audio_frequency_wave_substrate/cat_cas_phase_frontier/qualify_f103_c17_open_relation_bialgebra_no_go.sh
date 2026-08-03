#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
CAT_DIR="$ROOT/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/audio_frequency_wave_substrate/cat_cas_phase_frontier"
PYTHON="$ROOT/.venv/bin/python"
RUN_DIR="$(mktemp -d /dev/shm/ags-audio-m152-qualifier-XXXXXX)"
PRODUCTION="$CAT_DIR/f103_c17_open_relation_bialgebra_no_go.py"
ORACLE="$CAT_DIR/f103_c17_open_relation_bialgebra_no_go_oracle.py"
SEALED_PRODUCTION="$CAT_DIR/F103_C17_OPEN_RELATION_BIALGEBRA_NO_GO_RESULTS.json"
SEALED_ORACLE="$CAT_DIR/F103_C17_OPEN_RELATION_BIALGEBRA_NO_GO_ORACLE_RESULTS.json"

test "$(git branch --show-current)" = "codex/audio-frequency-wave-substrate"
test -x "$PYTHON"

printf '%s  %s\n' \
  '49d4b39e04b6fcc5fca35cba4f6303b8a009ce8a42083fe5069066f3df2e4896' \
  "$PRODUCTION" | sha256sum -c -
printf '%s  %s\n' \
  '6c3a85dff1d6441b1f0317af3545f9e725c4e6db710e4ee19d7920f98e400f7c' \
  "$ORACLE" | sha256sum -c -

if rg -n '(^|[[:space:]])(import|from)[[:space:]].*(f103_c17_open_relation_bialgebra_no_go|numpy)' "$ORACLE"; then
  echo 'oracle imports production or NumPy' >&2
  exit 1
fi

nice -n 10 "$PYTHON" -X dev "$PRODUCTION" \
  --output "$RUN_DIR/production.json"
cmp "$RUN_DIR/production.json" "$SEALED_PRODUCTION"

nice -n 10 "$PYTHON" -X dev "$ORACLE" \
  --production "$RUN_DIR/production.json" \
  --output "$RUN_DIR/oracle.json"
cmp "$RUN_DIR/oracle.json" "$SEALED_ORACLE"

jq -e '
  .execution_scope.case_count == 18 and
  .execution_scope.all_cases_exact == true and
  .relation_law.signature_coordinates_per_relation == 17 and
  .relation_law.implicit_dense_relation_cells_per_signature == 289 and
  .relation_law.materialized_dense_relation_table_cells == 0 and
  .relation_law.materialized_assignment_expansion_cells == 0 and
  .relation_law.shared_unresolved_port_consumers_per_out_layer == 8 and
  .relation_law.resident_relation_projection_before_boundary == false and
  .matched_classical_recurrences.full_state_and_boundary_match_every_case == true and
  .matched_classical_recurrences.resident_field_coordinates_each == 459 and
  .matched_classical_recurrences.maximum_phase_bialgebra_multiplications == 2506752 and
  .matched_classical_recurrences.maximum_dual_spectral_bialgebra_multiplications == 2506752 and
  .matched_classical_recurrences.optimal_compact_classical_recurrence_claimed == false and
  .restoration.carrier_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .restoration.same_backing == true and
  .restoration.inverse_history_cells == 0 and
  .restoration.retained_restoration_baseline_cells == 0 and
  .restoration.snapshot_reload_used == false and
  .restoration.unrelated_program_reuse.second_boundary_matches_fresh == true and
  .restoration.repeated_reuse.cycles == 64 and
  ([.controls[]] | all) and
  (.not_established | index("CATVM_CUSTODY")) != null and
  (.not_established | index("DISTINCT_PHASE_RESOURCE")) != null and
  (.not_established | index("COMPUTATIONAL_ADVANTAGE")) != null and
  (.not_established | index("SMALL_WALL_CROSSING")) != null
' "$RUN_DIR/production.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .independence.imports_production == false and
  .independence.imports_numpy == false and
  .exact_case_reexecutions == 18 and
  .case_field_comparisons == 360 and
  .all_459_coordinate_coefficient_dual_matches == true and
  .all_boundary_matches == true and
  .all_exact_restorations == true and
  .resource_law.resident_field_coordinates_each == 459 and
  .resource_law.maximum_phase_bialgebra_multiplications == 2506752 and
  .resource_law.maximum_dual_spectral_bialgebra_multiplications == 2506752 and
  .resource_law.dense_relation_table_cells_materialized == 0 and
  .restoration_class == "EXACT_ALGEBRAIC_RESTORATION" and
  ([.controls[]] | all) and
  .decision == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
' "$RUN_DIR/oracle.json" >/dev/null

echo "QUALIFIED_F103_C17_OPEN_RELATION_BIALGEBRA_NO_GO_STRICT_SCOPE"
echo "evidence=$RUN_DIR"
