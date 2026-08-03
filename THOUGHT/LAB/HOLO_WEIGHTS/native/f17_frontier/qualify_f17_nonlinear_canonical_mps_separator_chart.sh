#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_nonlinear_canonical_mps_separator_chart.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python_bin="$repo_root/.venv/bin/python"
production="$frontier_dir/f17_nonlinear_canonical_mps_separator_chart.py"
oracle="$frontier_dir/f17_nonlinear_canonical_mps_separator_chart_oracle.py"
sealed="$frontier_dir/F17_NONLINEAR_CANONICAL_MPS_SEPARATOR_CHART_RESULTS.json"
sealed_oracle="$frontier_dir/F17_NONLINEAR_CANONICAL_MPS_SEPARATOR_CHART_ORACLE_RESULTS.json"

mkdir -p "$evidence_dir"
production_result="$evidence_dir/F17_NONLINEAR_CANONICAL_MPS_SEPARATOR_CHART_RESULTS.json"
oracle_result="$evidence_dir/F17_NONLINEAR_CANONICAL_MPS_SEPARATOR_CHART_ORACLE_RESULTS.json"

if [[ ! -x "$python_bin" ]]; then
  echo "repository virtual environment is unavailable" >&2
  exit 1
fi

if rg -n '^[[:space:]]*(from|import)[[:space:]]+(f17_nonlinear_canonical_mps_separator_chart|f17_runtime_weighted_grid|f17_grid_linear_separator|catvm)([[:space:]]|$)' "$oracle"; then
  echo "independent M122 oracle imports production or a phase/CATVM backend" >&2
  exit 1
fi

"$python_bin" - "$production" <<'PY'
import ast
import sys

tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())
transaction = next(
    node for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name == "execute_transaction"
)
projection_calls = [
    node for node in ast.walk(transaction)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "project_boundary"
]
if len(projection_calls) != 1:
    raise SystemExit("accepted M122 transaction does not have exactly one boundary projection")
top_level_for_lines = [node.lineno for node in transaction.body if isinstance(node, ast.For)]
if len(top_level_for_lines) < 2 or not (top_level_for_lines[0] < projection_calls[0].lineno < top_level_for_lines[-1]):
    raise SystemExit("accepted M122 transaction projection is not between forward and inverse row schedules")
PY

PYTHONHASHSEED=0 LC_ALL=C nice -n 10 "$python_bin" -X dev "$production" > "$production_result"
PYTHONHASHSEED=0 LC_ALL=C nice -n 10 "$python_bin" -X dev "$oracle" > "$oracle_result"

cmp "$production_result" "$sealed"
cmp "$oracle_result" "$sealed_oracle"

jq -e '
  .schema == "CAT_CAS_F17_NONLINEAR_CANONICAL_MPS_SEPARATOR_CHART_V1"
  and .classification == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level == "PACKAGE_SELF_REVIEW"
  and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
  and .transient_chart_and_projection_buffers_restoration_classification == "NO_RESTORATION_CLAIM"
  and .accepted_path_dense_two_to_the_n_row_vector_materialized == false
  and .canonicalization_can_require_two_to_the_n_scratch_cells_at_rank_saturation == true
  and (.scratch_exact_coefficient_payload_measured|not)
  and .scratch_payload_number_is_resident_height_estimate_only
  and .exact_zero_division_pivot_and_gauge_operations_are_resident_phase_primitives == false
  and (.exact_transactions | length) == 6
  and (.dual_field_structural_transactions | length) == 28
  and ([.exact_transactions[] | .restored_exactly and .same_backing and .restoration_generation_correct and .lease_cleared_after_restoration and (.retained_inverse_history_bytes == 0) and (.snapshot_reload_used|not)] | all)
  and ([.dual_field_structural_transactions[] | .restored_exactly and .same_backing and .restoration_generation_correct and .lease_cleared_after_restoration and (.retained_inverse_history_bytes == 0) and (.snapshot_reload_used|not)] | all)
  and .generic_n4_discriminator.final_rank_profile == [2,4,2]
  and .generic_n4_discriminator.raw_core_field_cells == 40
  and .generic_n4_discriminator.effective_chart_field_coordinates == 16
  and .generic_n4_discriminator.dense_row_message_field_coordinates == 16
  and .generic_n4_discriminator.projective_ratios_plus_scale == 16
  and (.generic_n4_discriminator.separator_compaction_observed|not)
  and ([.projective_controls[] | .amplitude_bearing_projective_chart_coordinates == .row_message_field_coordinates and (.separator_compaction|not)] | all)
  and ([.controls.missing_inverse_detected,.controls.wrong_inverse_detected,.controls.prospectively_noncommuting_reordered_inverse_detected,.controls.zero_phase_weight_rejected,.controls.null_carrier_rejected,.controls.wrong_chart_metadata_detected] | all)
  and .matchgate_controls.positive_control_all_zero
  and .matchgate_controls.negative_control_nonzero_sites == [13]
  and .matchgate_controls.generic_discriminator_field_residues == [7,13,1,11,1,3,16,10,1,7,3,8,13,2,6,16]
  and .matchgate_controls.arbitrary_local_basis_or_holographic_matchgate_reductions_exhausted == false
  and (.catvm_custody_claimed|not)
  and (.distinct_phase_resource_established|not)
  and (.computational_advantage_established|not)
  and (.small_wall_crossing_established|not)
  and (.physical_waveform_execution_established|not)
  and (.physical_bits_replaced_with_pi|not)
  and (.unbounded_catalytic_computation_established|not)
' "$production_result" >/dev/null

jq -e '
  .schema == "CAT_CAS_F17_NONLINEAR_CANONICAL_MPS_SEPARATOR_CHART_ORACLE_V1"
  and .independent_of_production
  and (.production_imported|not)
  and (.phase_backend_imported|not)
  and (.exact | length) == 6
  and (.structural | length) == 28
  and .generic_n4_discriminator.final_natural_rank_profile == [2,4,2]
  and .generic_n4_discriminator.every_separator_bit_order_has_maximal_tt_ranks
  and .generic_n4_discriminator.effective_tt_chart_coordinates == 16
  and .generic_n4_discriminator.best_fully_reduced_projective_evdd_nodes == 15
  and .generic_n4_discriminator.full_tree_nodes == 15
  and .generic_n4_discriminator.all_entries_nonzero
  and ([.controls.all_exact_inverse_restorations_pass,.controls.all_dual_field_inverse_restorations_pass,.controls.all_exact_message_entries_nonzero,.controls.dense_materialization_is_oracle_only] | all)
  and (.interpretation.projectivization_reduces_amplitude_bearing_coordinates|not)
  and (.interpretation.generic_n4_tt_effective_coordinates_below_dense_width|not)
  and (.interpretation.generic_n4_projective_evdd_below_full_tree|not)
  and (.interpretation.dual_field_growth_is_exact_q_zeta17_proof_beyond_n4|not)
  and (.interpretation.broader_holographic_or_global_algorithms_exhausted|not)
  and (.interpretation.distinct_phase_resource_established|not)
  and (.interpretation.computational_advantage_established|not)
' "$oracle_result" >/dev/null

"$python_bin" - "$production_result" "$oracle_result" <<'PY'
import json
import sys

production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))

for actual, reference in zip(production["exact_transactions"], oracle["exact"], strict=True):
    if (actual["n"], actual["family"], actual["boundary"]) != (reference["n"], reference["family"], reference["boundary"]):
        raise SystemExit("M122 exact boundary parity failed")
    if [row["ranks"] for row in actual["rank_trace"]] != reference["rank_trace"]:
        raise SystemExit("M122 exact rank-trace parity failed")

for actual, reference in zip(production["dual_field_structural_transactions"], oracle["structural"], strict=True):
    if (actual["n"], actual["family"], actual["algebra"], actual["boundary"]) != (reference["n"], reference["family"], reference["field"], reference["boundary"]):
        raise SystemExit("M122 dual-field boundary parity failed")
    if [row["ranks"] for row in actual["rank_trace"]] != reference["rank_trace"]:
        raise SystemExit("M122 dual-field rank-trace parity failed")

generic = [item for item in oracle["structural"] if item["family"] == "GENERIC"]
for n in range(2, 9):
    fixtures = [item for item in generic if item["n"] == n]
    if len(fixtures) != 2:
        raise SystemExit("M122 dual-field fixture count changed")
    if not any(item["every_separator_bit_order_has_maximal_tt_ranks"] for item in fixtures):
        raise SystemExit(f"M122 lacks a full modular all-order rank certificate at n={n}")
    if any(item["final_effective_tt_chart_coordinates"] != 2 ** n for item in fixtures):
        raise SystemExit(f"M122 effective TT dimension unexpectedly compressed at n={n}")

print("QUALIFIED_F17_NONLINEAR_CANONICAL_MPS_SEPARATOR_CHART_NO_GO")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle" "$0" > "$evidence_dir/SHA256SUMS"
