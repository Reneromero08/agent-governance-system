#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/controlled_virtual_phase_wave_superposition_small_wall_triad.py"
ORACLE="$HERE/controlled_virtual_phase_wave_superposition_small_wall_triad_independent_oracle.py"
SEALED="$HERE/CONTROLLED_VIRTUAL_PHASE_WAVE_SUPERPOSITION_SMALL_WALL_TRIAD_RESULTS.json"
SEALED_ORACLE="$HERE/CONTROLLED_VIRTUAL_PHASE_WAVE_SUPERPOSITION_SMALL_WALL_TRIAD_INDEPENDENT_ORACLE.json"
REVIEW="$HERE/CONTROLLED_VIRTUAL_PHASE_WAVE_SUPERPOSITION_SMALL_WALL_TRIAD_INDEPENDENT_REEXECUTION_REVIEW.md"

export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX

# Warm timing is environment-specific.  Reexecute and compare every
# deterministic field after replacing sealed timing objects with null.  Parse
# JSON before comparison so exponent-letter formatting is immaterial.
"$PYTHON" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    sealed = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    fresh = json.load(handle)
for case in sealed["triad_cases"]:
    case["warm_timing"] = None
if sealed != fresh:
    raise SystemExit("deterministic production reexecution differs from sealed result")
' "$SEALED" <("$PYTHON" "$PRODUCTION")
cmp "$SEALED_ORACLE" <("$PYTHON" "$ORACLE")

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .in_place_restoration_classification == "NUMERICAL_PHYSICAL_STATE_RESTORATION" and
  .sham_restoration_classification == "SNAPSHOT_RELOAD" and
  .direct_restoration_classification == "NO_RESTORATION_CLAIM" and
  (.triad_cases | length) == 14 and
  ([.triad_cases[] |
    .width == (.q - 1) and
    .identical_public_boundary and
    .identical_request_response_traffic and
    .compact_direct.request_bytes == .snapshot_sham.request_bytes and
    .compact_direct.request_bytes == .in_place_virtual_phase.request_bytes and
    .compact_direct.response_bytes == .snapshot_sham.response_bytes and
    .compact_direct.response_bytes == .in_place_virtual_phase.response_bytes and
    .compact_direct.software_fft_calls_primary_and_reuse == 2 and
    .snapshot_sham.software_fft_calls_primary_and_reuse == 2 and
    .snapshot_sham.snapshot_bytes_primary_and_reuse == (3 * 16 * .width) and
    .snapshot_sham.restoration_classification == "SNAPSHOT_RELOAD" and
    .snapshot_sham.reuse_matches_fresh and
    .snapshot_sham.baseline_reload_used and
    .in_place_virtual_phase.logical_complex_cells == (3 * .width) and
    .in_place_virtual_phase.carrier_complex_cells == .width and
    .in_place_virtual_phase.declared_chirp_and_transform_temporary_complex_cells == (2 * .width) and
    .in_place_virtual_phase.carrier_bits == (128 * .width) and
    .in_place_virtual_phase.mode_count == .width and
    .in_place_virtual_phase.simultaneous_mode_bandwidth_units == .width and
    .in_place_virtual_phase.software_fft_calls_primary_and_reuse == 4 and
    .in_place_virtual_phase.verification_and_controls_software_fft_calls == 517 and
    .in_place_virtual_phase.restoration_error <= .predeclared_state_tolerance and
    .in_place_virtual_phase.reuse_restoration_error <= .predeclared_state_tolerance and
    .in_place_virtual_phase.repeated_reuse_max_error <= .predeclared_state_tolerance and
    .in_place_virtual_phase.repeated_reuse_cycles == 256 and
    .in_place_virtual_phase.boundary_survives_inverse and
    .in_place_virtual_phase.same_backing and
    .in_place_virtual_phase.reuse_matches_fresh and
    (.in_place_virtual_phase.baseline_reload_used | not) and
    .in_place_virtual_phase.controls.missing_inverse_fails and
    .in_place_virtual_phase.controls.wrong_inverse_fails and
    .in_place_virtual_phase.controls.reordered_inverse_fails_for_noncommuting_chirp_and_dft and
    .in_place_virtual_phase.controls.dephasing_changes_boundary and
    .in_place_virtual_phase.controls.null_carrier_rejected and
    (.in_place_virtual_phase.controls.premature_projection_machine_enforced | not) and
    .warm_timing.repetitions == 17 and
    .warm_timing.timing_is_environment_specific_not_claim_authority] | all) and
  .controls.all_boundaries_match and
  .controls.all_public_traffic_matches and
  .controls.all_actual_inverse_restorations_within_tolerance and
  .controls.all_actual_restored_reuses_match_fresh and
  .controls.all_256_cycle_reuses_within_tolerance and
  .controls.all_missing_inverses_fail and
  .controls.all_wrong_inverses_fail and
  .controls.all_applicable_reordered_inverses_fail and
  .controls.all_dephased_boundaries_change and
  .controls.all_null_carriers_rejected and
  .controls.all_shams_use_snapshot_reload and
  (.small_wall_decision.crossing_established | not) and
  (.small_wall_decision.distinct_phase_resource_established | not) and
  .small_wall_decision.coherence_causally_relevant and
  (.small_wall_decision.coherence_advantage_over_compact_classical_software | not) and
  ([.strict_boundaries[]] | all(. == false))
' "$SEALED" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "NUMERICAL_PHYSICAL_STATE_RESTORATION" and
  (.oracle_independence.imports_production | not) and
  (.oracle_independence.uses_numpy_fft | not) and
  .oracle_independence.uses_standalone_dense_unitary_dft and
  (.oracle_cases | length) == 14 and
  ([.oracle_cases[] |
    .width == (.q - 1) and
    .restoration_error <= 2e-12 and
    .reuse_restoration_error <= 2e-12 and
    .reuse_matches_fresh and
    .missing_inverse_fails and
    .wrong_inverse_fails and
    .reordered_inverse_fails and
    .dephasing_changes_boundary and
    .dense_dft_terms_primary_and_reuse_with_inverses == (4 * .width * .width)] | all) and
  .controls.all_boundaries_match_production_within_predeclared_tolerance and
  .controls.all_inverse_restorations_within_predeclared_tolerance and
  .controls.all_reuses_match_fresh and
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
assert len(production["triad_cases"]) == len(oracle["oracle_cases"]) == 14
for left, right in zip(production["triad_cases"], oracle["oracle_cases"]):
    assert left["q"] == right["q"]
    assert left["request"]["chirp_strength"] == right["chirp_strength"]
    assert left["request"]["weight_shift"] == right["weight_shift"]
    assert left["second_program"]["chirp_strength"] == right["second_chirp_strength"]
    assert left["second_program"]["weight_shift"] == right["second_weight_shift"]
    produced = complex(*left["in_place_virtual_phase"]["boundary"])
    independently = complex(*right["projected_boundary"])
    assert abs(produced - independently) <= 2e-12
    produced_reuse = complex(*left["compact_direct"]["fresh_reuse_boundary"])
    independent_reuse = complex(*right["reused_boundary"])
    assert abs(produced_reuse - independent_reuse) <= 2e-12
PY

if rg -n \
  'import[[:space:]]+controlled_virtual_phase|from[[:space:]]+controlled_virtual_phase' \
  "$ORACLE"; then
  printf '%s\n' 'oracle production-dependency isolation failed' >&2
  exit 1
fi

sha256sum "$PRODUCTION" "$ORACLE" "$SEALED" "$SEALED_ORACLE" "$REVIEW"
printf '%s\n' 'QUALIFIED_CONTROLLED_VIRTUAL_PHASE_WAVE_SUPERPOSITION_SMALL_WALL_TRIAD_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$HERE"
