#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/controlled_local_interference_feedback_phase_coupling.py"
ORACLE="$HERE/controlled_local_interference_feedback_phase_coupling_independent_oracle.py"
SEALED="$HERE/CONTROLLED_LOCAL_INTERFERENCE_FEEDBACK_PHASE_COUPLING_RESULTS.json"
SEALED_ORACLE="$HERE/CONTROLLED_LOCAL_INTERFERENCE_FEEDBACK_PHASE_COUPLING_INDEPENDENT_ORACLE.json"
REVIEW="$HERE/CONTROLLED_LOCAL_INTERFERENCE_FEEDBACK_PHASE_COUPLING_INDEPENDENT_REEXECUTION_REVIEW.md"

export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX

# Reexecute deterministic production fields. Warm timing is deliberately
# excluded from authority and normalized to null before semantic comparison.
"$PYTHON" -c '
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    sealed = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    fresh = json.load(handle)
for case in sealed["cases"]:
    case["warm_timing"] = None
if sealed != fresh:
    raise SystemExit("deterministic production reexecution differs from seal")
' "$SEALED" <("$PYTHON" "$PRODUCTION")

"$PYTHON" -c '
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    sealed = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    fresh = json.load(handle)
if sealed != fresh:
    raise SystemExit("independent oracle reexecution differs from seal")
' "$SEALED_ORACLE" <(/usr/bin/python3 "$ORACLE")

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .in_place_restoration_classification == "NUMERICAL_PHYSICAL_STATE_RESTORATION" and
  .sham_restoration_classification == "SNAPSHOT_RELOAD" and
  .direct_restoration_classification == "NO_RESTORATION_CLAIM" and
  (.cases | length) == 4 and
  ([.cases[] |
    .compact_direct.logical_complex_cells == (.width + 2) and
    .compact_direct.local_couplers == (.width * .depth) and
    .compact_direct.local_feedback_operations == (.width * .depth) and
    .compact_direct.projection_terms == .width and
    .snapshot_sham.logical_complex_cells == (2 * .width + 2) and
    .snapshot_sham.accepted_transaction_logical_complex_cells == (2 * .width + 2) and
    .snapshot_sham.verification_peak_logical_complex_cells == (3 * .width + 2) and
    .snapshot_sham.local_couplers_primary_and_reuse == (2 * .width * .depth) and
    .snapshot_sham.local_feedback_operations_primary_and_reuse == (2 * .width * .depth) and
    .snapshot_sham.snapshot_bytes_primary_and_reuse == (3 * 16 * .width) and
    .snapshot_sham.restoration_classification == "SNAPSHOT_RELOAD" and
    .snapshot_sham.baseline_reload_used and
    .snapshot_sham.reuse_matches_fresh and
    .in_place_virtual_phase.logical_complex_cells == (.width + 2) and
    .in_place_virtual_phase.accepted_transaction_logical_complex_cells == (.width + 2) and
    .in_place_virtual_phase.verification_and_controls_peak_logical_complex_cells == (4 * .width + 2) and
    .in_place_virtual_phase.mode_count == .width and
    .in_place_virtual_phase.simultaneous_mode_bandwidth_units == .width and
    .in_place_virtual_phase.precision_bits_per_mode == 128 and
    .in_place_virtual_phase.local_couplers_primary_and_reuse == (4 * .width * .depth) and
    .in_place_virtual_phase.local_feedback_operations_primary_and_reuse == (4 * .width * .depth) and
    .in_place_virtual_phase.abstract_parallel_sublayer_depth_primary_and_reuse == (12 * .depth) and
    .in_place_virtual_phase.projection_terms_primary_and_reuse == (2 * .width) and
    .in_place_virtual_phase.verification_and_controls_projection_terms == (4 * .width) and
    .in_place_virtual_phase.commitment_serialization_bytes == (3 * 16 * .width) and
    .in_place_virtual_phase.same_backing and
    .in_place_virtual_phase.reuse_matches_fresh and
    .in_place_virtual_phase.boundary_survives_inverse and
    (.in_place_virtual_phase.baseline_reload_used | not) and
    .in_place_virtual_phase.restoration_error <= .predeclared_state_tolerance and
    .in_place_virtual_phase.reuse_restoration_error <= .predeclared_state_tolerance and
    .in_place_virtual_phase.repeated_reuse_max_error <= .predeclared_state_tolerance and
    .in_place_virtual_phase.controls.missing_inverse_fails and
    .in_place_virtual_phase.controls.wrong_inverse_fails and
    .in_place_virtual_phase.controls.reordered_inverse_fails_for_noncommuting_feedback_and_couplers and
    .in_place_virtual_phase.controls.dephasing_changes_boundary and
    .in_place_virtual_phase.controls.zero_feedback_changes_boundary and
    .in_place_virtual_phase.controls.swapped_coupler_order_changes_boundary and
    .in_place_virtual_phase.controls.null_carrier_rejected and
    (.in_place_virtual_phase.controls.premature_projection_machine_enforced | not) and
    .identical_public_boundary and
    .identical_request_response_traffic and
    .warm_timing.repetitions == 9 and
    .warm_timing.timing_is_environment_specific_not_claim_authority] | all) and
  .controls.all_boundaries_match and
  .controls.all_controls_discriminate and
  .controls.all_public_traffic_matches and
  .controls.all_restorations_within_tolerance and
  .resource_conclusion.free_global_dft_removed and
  .resource_conclusion.local_nonlinear_feedback_present and
  (.resource_conclusion.distinct_phase_resource_established | not) and
  (.resource_conclusion.small_wall_crossing_established | not) and
  (.claim_boundaries.catvm_custody | not) and
  (.claim_boundaries.machine_enforced_hidden_intermediate | not) and
  (.claim_boundaries.physical_waveform_execution | not) and
  (.claim_boundaries.computational_advantage | not) and
  (.claim_boundaries.unbounded_catalytic_computation | not)
' "$SEALED" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  (.cases | length) == 4 and
  .all_controls_discriminate and
  .all_restorations_within_tolerance and
  ([.cases[] |
    .compiled_public_operation_count == (2 * .width * .depth) and
    .local_couplers_per_forward == (.width * .depth) and
    .local_feedback_operations_per_forward == (.width * .depth) and
    .repeated_reuse_cycles == 128 and
    .restoration_error <= 2e-10 and
    .reuse_restoration_error <= 2e-10 and
    .repeated_reuse_max_error <= 2e-10 and
    .controls.missing_inverse_fails and
    .controls.wrong_inverse_fails and
    .controls.reordered_inverse_fails_for_noncommuting_feedback_and_couplers and
    .controls.dephasing_changes_boundary and
    .controls.zero_feedback_changes_boundary and
    .controls.swapped_coupler_order_changes_boundary] | all)
' "$SEALED_ORACLE" >/dev/null

"$PYTHON" -c '
import json
import sys
production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
if [case["width"] for case in production["cases"]] != [case["width"] for case in oracle["cases"]]:
    raise SystemExit("oracle case topology differs")
for left, right in zip(production["cases"], oracle["cases"], strict=True):
    p = complex(*left["in_place_virtual_phase"]["boundary"])
    o = complex(*right["boundary"])
    if abs(p - o) > production["predeclared_boundary_tolerance"]:
        raise SystemExit("oracle boundary differs")
    if right["local_couplers_per_forward"] != left["width"] * left["depth"]:
        raise SystemExit("oracle coupler accounting differs")
' "$SEALED" "$SEALED_ORACLE"

"$PYTHON" -c '
import ast
import pathlib
import sys
tree = ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        names = [node.module or ""]
    else:
        continue
    if any(name == "numpy" or "controlled_local_interference_feedback" in name for name in names):
        raise SystemExit("oracle production-dependency isolation failed")
' "$ORACLE"

sha256sum "$PRODUCTION" "$ORACLE" "$SEALED" "$SEALED_ORACLE" "$REVIEW"
printf '%s\n' 'QUALIFIED_CONTROLLED_LOCAL_INTERFERENCE_FEEDBACK_PHASE_COUPLING_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$HERE"
