#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/local_nonlinear_phase_continuation_observability_rank.py"
ORACLE="$HERE/local_nonlinear_phase_continuation_observability_rank_independent_oracle.py"
DEPENDENCY="$HERE/controlled_local_interference_feedback_phase_coupling.py"
SEALED="$HERE/LOCAL_NONLINEAR_PHASE_CONTINUATION_OBSERVABILITY_RANK_RESULTS.json"
SEALED_ORACLE="$HERE/LOCAL_NONLINEAR_PHASE_CONTINUATION_OBSERVABILITY_RANK_INDEPENDENT_ORACLE.json"
REVIEW="$HERE/LOCAL_NONLINEAR_PHASE_CONTINUATION_OBSERVABILITY_RANK_INDEPENDENT_REEXECUTION_REVIEW.md"

export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX

"$PYTHON" -c '
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    sealed = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    fresh = json.load(handle)
if sealed != fresh:
    raise SystemExit("analytic observability reexecution differs from seal")
' "$SEALED" <("$PYTHON" "$PRODUCTION")

"$PYTHON" -c '
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    sealed = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    fresh = json.load(handle)
if sealed != fresh:
    raise SystemExit("finite-difference oracle reexecution differs from seal")
' "$SEALED_ORACLE" <("$PYTHON" "$ORACLE")

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "NUMERICAL_PHYSICAL_STATE_RESTORATION" and
  .predeclared_state_tolerance == 2e-10 and
  .predeclared_rank_relative_tolerance == 1e-9 and
  (.cases | length) == 3 and
  ([.cases[] |
    .public_suffix_count == .width and
    .certificate.public_suffix_count == .width and
    .certificate.tangent_dimension == (2 * .width - 1) and
    .certificate.observability_matrix_shape == [2 * .width, 2 * .width - 1] and
    .certificate.observed_rank == (2 * .width - 1) and
    .certificate.full_tangent_rank and
    .certificate.minimum_retained_singular_value > .certificate.rank_threshold and
    .certificate.remove_one_suffix_rank == (2 * .width - 2) and
    .certificate.remove_one_suffix_forces_rank_below_tangent_dimension and
    .certificate.duplicate_one_suffix_rank == 2 and
    .certificate.duplicate_suffix_underobserves and
    .certificate.zero_feedback_changes_analytic_matrix and
    .certificate.accepted_carrier_logical_complex_cells == (.width + 2) and
    .certificate.analytic_verification_conservative_named_real_scalars == (6 * .width * (2 * .width - 1) + 10 * .width - 1) and
    .certificate.public_suffix_descriptor_scalars == (7 * .width) and
    .certificate.public_suffix_descriptor_json_bytes > 0 and
    .certificate.analytic_state_coupler_operations == (2 * .width * .width) and
    .certificate.analytic_state_feedback_operations == (2 * .width * .width) and
    .certificate.analytic_tangent_coupler_operations == (.width * .width * (2 * .width - 1)) and
    .certificate.analytic_tangent_feedback_operations == (.width * .width * (2 * .width - 1)) and
    .certificate.analytic_boundary_projection_terms == (.width * .width) and
    .certificate.analytic_tangent_projection_terms == (.width * .width * (2 * .width - 1)) and
    .accepted_carrier_logical_complex_cells == (.width + 2) and
    .accepted_primary_reuse_coupler_operations == (4 * (.width * .width + .width)) and
    .accepted_primary_reuse_feedback_operations == (4 * (.width * .width + .width)) and
    .accepted_primary_reuse_projection_terms == (2 * .width) and
    .verification_reuse_and_inverse_control_coupler_operations == ((2 * 64 + 4) * (.width * .width + .width)) and
    .verification_reuse_and_inverse_control_feedback_operations == ((2 * 64 + 4) * (.width * .width + .width)) and
    .verification_fresh_projection_terms == .width and
    .commitment_serialization_bytes == (2 * 16 * .width) and
    .public_prefix_descriptor_scalars == 14 and
    .same_backing and
    .reuse_matches_fresh and
    .restoration_error <= 2e-10 and
    .reuse_restoration_error <= 2e-10 and
    .repeated_reuse_cycles == 64 and
    .repeated_reuse_max_error <= 2e-10 and
    .controls.missing_inverse_fails and
    .controls.wrong_inverse_fails and
    .controls.reordered_prefix_suffix_inverse_fails and
    .controls.remove_one_suffix_underobserves and
    .controls.duplicate_suffix_underobserves and
    .controls.zero_feedback_changes_analytic_matrix and
    (.response_order_machine_enforced | not)] | all) and
  .controls.all_full_tangent_rank and
  .controls.all_controls_discriminate and
  .controls.all_restorations_within_tolerance and
  (.claim_ceiling.global_nonlinear_quotient_rejected | not) and
  (.claim_ceiling.arbitrary_continuation_family_rejected | not) and
  (.claim_ceiling.algorithmic_state_lower_bound_established | not) and
  (.claim_ceiling.distinct_phase_resource_established | not) and
  (.claim_ceiling.small_wall_crossing_established | not) and
  (.claim_ceiling.catvm_custody | not) and
  (.claim_ceiling.physical_waveform_execution | not)
' "$SEALED" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .all_full_tangent_rank and
  .all_controls_discriminate and
  .all_restorations_within_tolerance and
  (.cases | length) == 3 and
  ([.cases[] |
    .public_suffix_count == .width and
    .tangent_dimension == (2 * .width - 1) and
    .finite_difference_step == 1e-6 and
    .rank_relative_tolerance == 1e-9 and
    .finite_difference_rank == (2 * .width - 1) and
    .full_tangent_rank and
    .minimum_retained_singular_value > .rank_threshold and
    .remove_one_suffix_rank == (2 * .width - 2) and
    .remove_one_suffix_underobserves and
    .duplicate_one_suffix_rank == 2 and
    .duplicate_suffix_underobserves and
    .zero_feedback_changes_finite_difference_matrix and
    .finite_difference_forward_evaluations == (2 * .width * (2 * .width - 1)) and
    .zero_feedback_control_forward_evaluations == (2 * .width * (2 * .width - 1)) and
    .restoration_error <= 2e-10 and
    .reuse_restoration_error <= 2e-10 and
    .reuse_matches_fresh and
    .repeated_reuse_cycles == 64 and
    .repeated_reuse_max_error <= 2e-10] | all)
' "$SEALED_ORACLE" >/dev/null

"$PYTHON" -c '
import json
import sys
production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
for left, right in zip(production["cases"], oracle["cases"], strict=True):
    if left["width"] != right["width"]:
        raise SystemExit("oracle chart differs")
    if left["certificate"]["observed_rank"] != right["finite_difference_rank"]:
        raise SystemExit("oracle rank differs")
    p = complex(*left["final_boundary"])
    o = complex(*right["final_boundary"])
    if abs(p - o) > 2e-11:
        raise SystemExit("oracle selected boundary differs")
    analytic = left["certificate"]["singular_values"]
    finite = right["singular_values"]
    relative = max(abs(a - b) / max(abs(a), 1e-30) for a, b in zip(analytic, finite, strict=True))
    if relative > 1e-3:
        raise SystemExit("oracle singular spectrum differs")
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
    if any("controlled_local_interference" in name or "local_nonlinear_phase_continuation" in name for name in names):
        raise SystemExit("finite-difference oracle production-dependency isolation failed")
' "$ORACLE"

sha256sum "$PRODUCTION" "$ORACLE" "$DEPENDENCY" "$SEALED" "$SEALED_ORACLE" "$REVIEW"
printf '%s\n' 'QUALIFIED_LOCAL_NONLINEAR_PHASE_CONTINUATION_OBSERVABILITY_RANK_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$HERE"
