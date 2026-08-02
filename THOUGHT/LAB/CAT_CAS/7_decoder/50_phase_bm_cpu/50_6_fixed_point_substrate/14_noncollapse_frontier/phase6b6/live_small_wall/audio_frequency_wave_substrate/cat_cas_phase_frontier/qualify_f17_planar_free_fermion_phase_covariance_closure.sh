#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_planar_free_fermion_phase_covariance_closure.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python_bin="$repo_root/.venv/bin/python"
production="$frontier_dir/f17_planar_free_fermion_phase_covariance_closure.py"
oracle="$frontier_dir/f17_planar_free_fermion_phase_covariance_closure_oracle.py"
sealed="$frontier_dir/F17_PLANAR_FREE_FERMION_PHASE_COVARIANCE_CLOSURE_RESULTS.json"
sealed_oracle="$frontier_dir/F17_PLANAR_FREE_FERMION_PHASE_COVARIANCE_CLOSURE_ORACLE_RESULTS.json"
production_result="$evidence_dir/F17_PLANAR_FREE_FERMION_PHASE_COVARIANCE_CLOSURE_RESULTS.json"
oracle_result="$evidence_dir/F17_PLANAR_FREE_FERMION_PHASE_COVARIANCE_CLOSURE_ORACLE_RESULTS.json"

mkdir -p "$evidence_dir"
if [[ ! -x "$python_bin" ]]; then
  echo "repository virtual environment is unavailable" >&2
  exit 1
fi

if rg -n '^[[:space:]]*(from|import)[[:space:]]+f17_planar_free_fermion_phase_covariance_closure([[:space:]]|$)' "$oracle"; then
  echo "independent M123 oracle imports the terminal-Pfaffian implementation" >&2
  exit 1
fi

PYTHONDONTWRITEBYTECODE=1 "$python_bin" - "$production" <<'PY'
import ast
import sys

tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())
transaction = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "execute_transaction"
)
calls = {
    name: [
        node for node in ast.walk(transaction)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]
    for name in ("forward", "project_boundary", "inverse")
}
if any(len(calls[name]) != 1 for name in calls):
    raise SystemExit("M123 transaction must call forward, final projection, and inverse exactly once")
if not (calls["forward"][0].lineno < calls["project_boundary"][0].lineno < calls["inverse"][0].lineno):
    raise SystemExit("M123 response ordering is not forward -> final projection -> inverse")
for function_name in ("forward", "_sector_value", "inverse"):
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "project_boundary"
        for node in ast.walk(function)
    ):
        raise SystemExit(f"M123 internal stage {function_name} projects a boundary")
PY

PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 LC_ALL=C nice -n 10 "$python_bin" -X dev "$production" > "$production_result"
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 LC_ALL=C nice -n 10 "$python_bin" -X dev "$oracle" > "$oracle_result"

cmp "$production_result" "$sealed"
cmp "$oracle_result" "$sealed_oracle"

PYTHONDONTWRITEBYTECODE=1 "$python_bin" - "$production_result" "$oracle_result" <<'PY'
import json
import sys

production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))

def require(condition, message):
    if not condition:
        raise SystemExit(message)

require(production["schema"] == "CAT_CAS_F17_PLANAR_FREE_FERMION_PHASE_COVARIANCE_CLOSURE_V1", "production schema changed")
require(production["classification"] == "SOURCE_AUDITED_PACKAGE_LOCAL", "production self-classification changed")
require(production["verification_level"] == "PACKAGE_SELF_REVIEW", "production verification level changed")
require(len(production["exact_transactions"]) == 6, "exact transaction count changed")
require(len(production["dual_field_structural_transactions"]) == 14, "dual-field transaction count changed")
for item in production["exact_transactions"] + production["dual_field_structural_transactions"]:
    require(item["exact_phase_cells_restored"], "phase cells were not restored")
    require(item["same_backing"], "carrier backing identity changed")
    require(item["restoration_generation_increment"], "restoration generation changed")
    require(not item["snapshot_reload_used"], "snapshot reload entered the accepted path")
    require(not item["inverse_history_retained"], "inverse history entered the accepted path")
for item in production["exact_transactions"]:
    require(item["carrier_restoration_class"] == "EXACT_ALGEBRAIC_RESTORATION", "carrier restoration class changed")
    require(item["projection_buffer_restoration_class"] == "NO_RESTORATION_CLAIM", "projection buffer class changed")
    require(item["intermediate_boundary_projection_calls"] == 0 and item["final_boundary_projection_calls"] == 1, "boundary projection law changed")
    expected = 1 if item["defect_count"] <= 1 else 1 << (item["defect_count"] - 1)
    require(item["even_sector_count"] == expected, "executed defect-sector law changed")
    require(item["internal_sector_pfaffian_evaluations"] == 2 * expected, "internal Pfaffian count changed")
    require(item["dense_projection_work_field_cells"] == item["pfaffian_stats"]["maximum_dense_work_field_cells"], "dense work-cell accounting diverged")
    require(item["pfaffian_stats"]["maximum_named_dense_work_payload_bits_upper_bound_from_observed_field_value"] > 0, "dense named-logical payload bound is absent")
    require(item["maximum_resident_phase_payload_bits"] > 0 and item["boundary_payload_bits"] > 0, "phase payload accounting is absent")

reuse = production["reuse"]
require(reuse["fresh_restored_boundary_agreement"], "fresh/restored reuse parity failed")
require(reuse["same_actual_backing_across_unrelated_programs"], "reuse backing identity changed")
require(reuse["generation_after_two_transactions"] == 2 and not reuse["baseline_reload_used"], "reuse generation or reload law changed")

controls = production["controls"]
for name in (
    "pfaffian_square_equals_determinant_f103_n3",
    "missing_inverse_detected",
    "wrong_inverse_ownership_detected",
    "premature_projection_rejected",
    "null_carrier_rejected",
    "odd_defect_sector_rejected",
    "alternate_path_boundary_agreement_f103_n4_k4",
    "undeclared_external_field_rejected_by_zero_field_gate",
    "single_orientation_edge_flip_changes_pfaffian",
    "intermediate_boundary_projection_api_absent",
):
    require(controls[name], f"control failed: {name}")
require(not controls["reordered_inverse_control_applicable"], "commuting inverse was misclassified")

ceiling = production["claim_ceiling"]
require(ceiling["implemented_sparse_external_field_path_uses_even_sector_expansion"], "implemented defect-path classification changed")
require("sparse_external_fields_require_even_sector_expansion" not in ceiling, "defect upper bound was promoted to a requirement")
for name in (
    "catvm_custody_established",
    "distinct_phase_resource_established",
    "computational_advantage_established",
    "small_wall_crossing_established",
    "physical_waveform_execution_established",
    "physical_bits_replaced_with_pi",
    "unbounded_catalytic_computation_established",
):
    require(not ceiling[name], f"forbidden promotion appeared: {name}")
require(not production["matched_baselines"]["phase_advantage_over_matched_classical"], "matched baseline boundary changed")

require(oracle["schema"] == "CAT_CAS_F17_PLANAR_FREE_FERMION_PHASE_COVARIANCE_CLOSURE_ORACLE_V1", "oracle schema changed")
require(oracle["implementation"] == "INDEPENDENT_BINARY_RESIDUE_HISTOGRAM_AND_BOUNDED_FACE_CYCLE_SPACE", "oracle implementation changed")
require(not oracle["imports_terminal_pfaffian_implementation"], "oracle imported production")
require(len(oracle["exact_cases"]) == 6 and len(oracle["modular_controls"]) == 12, "oracle case count changed")
require(all(item["dense_cycle_exact_agreement"] for item in oracle["exact_cases"]), "oracle exact recurrences disagree")
require(all(item["dense_cycle_agreement"] for item in oracle["modular_controls"]), "oracle modular recurrences disagree")

actual = {(item["n"], item["defect_count"]): item["boundary"] for item in production["exact_transactions"]}
reference = {(item["n"], item["defect_count"]): item["boundary"] for item in oracle["exact_cases"]}
require(actual == reference, "terminal-Pfaffian boundaries disagree with independent exact oracle")
require(
    [(item["defect_count"], item["even_sector_count"]) for item in production["defect_growth"]]
    == [(0, 1), (1, 1), (2, 2), (3, 4), (4, 8), (5, 16), (6, 32), (7, 64), (8, 128)],
    "declared sparse-defect sector law changed",
)
print("QUALIFIED_F17_PLANAR_FREE_FERMION_PHASE_COVARIANCE_CLOSURE_STRICT_SCOPE")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle" "$0" > "$evidence_dir/SHA256SUMS"
