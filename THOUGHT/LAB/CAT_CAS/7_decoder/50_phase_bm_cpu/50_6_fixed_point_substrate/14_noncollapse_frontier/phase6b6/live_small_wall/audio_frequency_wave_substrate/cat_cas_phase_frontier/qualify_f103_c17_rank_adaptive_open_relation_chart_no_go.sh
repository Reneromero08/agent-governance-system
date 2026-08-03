#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
DIR="$ROOT/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/audio_frequency_wave_substrate/cat_cas_phase_frontier"
PYTHON="$ROOT/.venv/bin/python"
RUN_DIR="$(mktemp -d /dev/shm/ags-audio-m154-qualifier-XXXXXX)"
PRODUCTION="$DIR/f103_c17_rank_adaptive_open_relation_chart_no_go.py"
ORACLE="$DIR/f103_c17_rank_adaptive_open_relation_chart_no_go_oracle.py"
SEALED_PRODUCTION="$DIR/F103_C17_RANK_ADAPTIVE_OPEN_RELATION_CHART_NO_GO_RESULTS.json"
SEALED_ORACLE="$DIR/F103_C17_RANK_ADAPTIVE_OPEN_RELATION_CHART_NO_GO_ORACLE_RESULTS.json"

test "$(git branch --show-current)" = "codex/audio-frequency-wave-substrate"
printf '%s  %s\n' 'e0fd159e64d6e3f127ffb83b2e904ec8d7dc25ca28851d6ee46c305b7da44cbf' "$PRODUCTION" | sha256sum -c -
printf '%s  %s\n' 'd5732914c184a5f53c5518f4898556706ab324a2f2aa896eaff7b625566a2728' "$ORACLE" | sha256sum -c -
if rg -n '(^|[[:space:]])(import|from)[[:space:]].*(f103_c17_rank_adaptive_open_relation_chart_no_go|numpy)' "$ORACLE"; then
  echo 'oracle imports production or NumPy' >&2
  exit 1
fi

nice -n 10 "$PYTHON" -X dev "$PRODUCTION" --output "$RUN_DIR/production.json"
cmp "$RUN_DIR/production.json" "$SEALED_PRODUCTION"
nice -n 10 "$PYTHON" -X dev "$ORACLE" \
  --production-result "$RUN_DIR/production.json" \
  --output "$RUN_DIR/oracle.json"
cmp "$RUN_DIR/oracle.json" "$SEALED_ORACLE"

jq -e '
  .execution_scope.case_count == 18 and
  .execution_scope.catvm_machine_boundary_used == false and
  .relation_law.translation_invariant == false and
  .relation_law.control_rank == 2 and
  .relation_law.reciprocal_control_rank == 2 and
  .relation_law.maximum_rank_observed == 17 and
  .relation_law.dense_equivalent_at_rank17 and
  .relation_law.rank17_resident_payload_is_relation_transpose_and_dense_equivalent and
  .relation_law.separate_dense_entry_table_materialized_on_phase_path == false and
  ([.cases[] | select(.depth == 32) | .final_ranks[]] | all(. == 17)) and
  .carrier_law.resident_total_bytes == 3375 and
  .carrier_law.machine_enforced_generation_or_lease_custody == false and
  .matched_classical_recurrence.maximum_resident_bytes == 2672 and
  .matched_classical_recurrence.phase_to_classical_resident_byte_ratio > 1 and
  .resource_accounting.maximum_phase_named_warm_bytes >
    .resource_accounting.maximum_classical_named_warm_bytes and
  .restoration.carrier_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .restoration.inverse_history_cells == 0 and
  .restoration.snapshot_reload_used == false and
  .restoration.unrelated_program_reuse.second_boundary_matches_fresh and
  .restoration.repeated_reuse.cycles == 32 and
  .restoration.repeated_reuse.exact_restoration and
  ([.controls[]] | all) and
  (.not_established | index("CATVM_CUSTODY")) != null and
  (.not_established | index("DISTINCT_PHASE_RESOURCE")) != null
' "$RUN_DIR/production.json" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .oracle_imports_production == false and
  .oracle_imports_numpy == false and
  .independent_case_count == 18 and
  .independent_comparisons == 531 and
  .maximum_rank_observed == 17 and
  .independent_dense_resident_field_coordinates == 2601 and
  ([.controls[]] | all)
' "$RUN_DIR/oracle.json" >/dev/null

echo "QUALIFIED_F103_C17_RANK_ADAPTIVE_OPEN_RELATION_CHART_NO_GO_STRICT_SCOPE"
echo "evidence=$RUN_DIR"
