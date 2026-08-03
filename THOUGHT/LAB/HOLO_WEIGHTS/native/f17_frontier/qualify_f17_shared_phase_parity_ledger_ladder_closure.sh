#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_shared_phase_parity_ledger_ladder_closure.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python_bin="$repo_root/.venv/bin/python"
production="$frontier_dir/f17_shared_phase_parity_ledger_ladder_closure.py"
oracle="$frontier_dir/f17_shared_phase_parity_ledger_ladder_closure_oracle.py"
sealed="$frontier_dir/F17_SHARED_PHASE_PARITY_LEDGER_LADDER_CLOSURE_RESULTS.json"
sealed_oracle="$frontier_dir/F17_SHARED_PHASE_PARITY_LEDGER_LADDER_CLOSURE_ORACLE_RESULTS.json"
production_result="$evidence_dir/F17_SHARED_PHASE_PARITY_LEDGER_LADDER_CLOSURE_RESULTS.json"
oracle_result="$evidence_dir/F17_SHARED_PHASE_PARITY_LEDGER_LADDER_CLOSURE_ORACLE_RESULTS.json"

mkdir -p "$evidence_dir"
if [[ ! -x "$python_bin" ]]; then
  echo "repository virtual environment is unavailable" >&2
  exit 1
fi

if rg -n '^[[:space:]]*(from|import)[[:space:]]+(f17_shared_phase_parity_ledger_ladder_closure|f17_planar_free_fermion_phase_covariance_closure)([[:space:]]|$)' "$oracle"; then
  echo "independent M124 oracle imports production or its predecessor" >&2
  exit 1
fi

PYTHONDONTWRITEBYTECODE=1 "$python_bin" - "$production" <<'PY'
import ast
import sys

tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())
functions = {
    node.name: node
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}
transaction = functions["execute_transaction"]
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
    raise SystemExit("M124 transaction must call forward, final projection, and inverse exactly once")
if not (calls["forward"][0].lineno < calls["project_boundary"][0].lineno < calls["inverse"][0].lineno):
    raise SystemExit("M124 response ordering is not forward -> final projection -> inverse")

for function_name in ("forward", "_contract_resident_ledger", "inverse"):
    function = functions[function_name]
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "project_boundary"
        for node in ast.walk(function)
    ):
        raise SystemExit(f"M124 internal stage {function_name} projects a boundary")

forbidden_names = {
    "combinations",
    "permutations",
    "product",
    "even_subsets",
    "zero_field_signature",
    "occurrence_expanded_boundary",
}
accepted = functions["_contract_resident_ledger"]
for node in ast.walk(accepted):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in forbidden_names:
            raise SystemExit(f"accepted M124 contraction calls forbidden expansion: {node.func.id}")
        if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_names:
            raise SystemExit(f"accepted M124 contraction calls forbidden expansion: {node.func.attr}")

inverse = functions["inverse"]
inverse_calls = [
    node.func.id
    for node in ast.walk(inverse)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
]
if "_contract_resident_ledger" not in inverse_calls or "_load" not in inverse_calls:
    raise SystemExit("M124 inverse does not rematerialize and unload the resident factors")
PY

PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 LC_ALL=C nice -n 10 "$python_bin" -X dev "$production" > "$production_result"
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 LC_ALL=C nice -n 10 "$python_bin" -X dev "$oracle" --production "$production_result" > "$oracle_result"

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

require(production["schema"] == "CAT_CAS_F17_SHARED_PHASE_PARITY_LEDGER_LADDER_CLOSURE_V1", "production schema changed")
require(production["classification"] == "SOURCE_AUDITED_PACKAGE_LOCAL", "production classification changed")
require(production["verification_level"] == "PACKAGE_SELF_REVIEW", "production verification level changed")
require(len(production["exact_transactions"]) == 14, "exact transaction count changed")
require(len(production["dual_field_structural_transactions"]) == 32, "structural transaction count changed")

rows = production["exact_transactions"] + production["dual_field_structural_transactions"]
for item in rows:
    require(item["resident_factor_field_cells"] == 10 * item["width"] - 4, "resident factor-cell law changed")
    require(item["maximum_frontier_field_cells"] == 4, "four-state frontier law changed")
    if item["width"] > 1:
        require(item["maximum_named_transient_field_cells"] == 19, "named transient field-cell accounting changed")
    require(item["exact_factor_carrier_restored"], "factor carrier was not restored")
    require(item["same_backing"], "carrier backing identity changed")
    require(item["restoration_generation_increment"], "restoration generation changed")
    require(item["response_released_after_restoration"], "response ordering receipt changed")
    require(not item["snapshot_reload_used"], "snapshot reload entered the in-place path")
    require(not item["inverse_history_retained"], "inverse history entered the accepted path")
    require(not item["accepted_path_even_sector_enumeration"], "accepted path enumerated sectors")
    require(not item["accepted_path_dense_signature_materialized"], "accepted path materialized a signature")
    require(item["intermediate_boundary_projection_calls"] == 0 and item["final_boundary_projection_calls"] == 1, "projection law changed")
    require(item["carrier_restoration_class"] == "EXACT_ALGEBRAIC_RESTORATION", "carrier restoration class changed")
    require(item["transient_frontier_restoration_class"] == "NO_RESTORATION_CLAIM", "transient restoration class changed")

reuse = production["reuse"]
require(reuse["fresh_restored_boundary_agreement"], "fresh/restored reuse parity failed")
require(reuse["same_actual_backing_across_unrelated_programs"], "restored reuse backing changed")
require(reuse["generation_after_two_transactions"] == 2 and not reuse["baseline_reload_used"], "reuse generation or reload law changed")

controls = production["controls"]
for name in (
    "missing_inverse_detected",
    "wrong_inverse_ownership_detected",
    "premature_projection_rejected",
    "null_carrier_rejected",
    "wrong_field_port_type_rejected",
    "resident_factor_mutation_changes_boundary",
    "snapshot_command_absent",
):
    require(controls[name], f"control failed: {name}")
require(not controls["reordered_inverse_control_applicable"], "commuting inverse was misclassified")
require(production["matched_baselines"]["strongest_implemented"] == "IDENTICAL_EXACT_FOUR_STATE_COLUMN_TRANSFER", "strongest matched baseline changed")
require(not production["matched_baselines"]["phase_advantage_over_matched_classical"], "phase advantage was promoted")
require(production["resource_law"]["named_transient_field_cells_upper_bound"] == 19, "declared transient field-cell ceiling changed")
require(production["claim_ceiling"]["native_non_gaussian_signature_scope"] == "PRIMARY_WIDTH2_EXACT_GRASSMANN_PLUCKER_WITNESS_ONLY", "production non-Gaussian scope changed")
require(not production["claim_ceiling"]["other_exact_families_or_widths_non_gaussian_established"], "production non-Gaussian scope was broadened")

ceiling = production["claim_ceiling"]
for name in (
    "arbitrary_width_or_treewidth_compaction_established",
    "catvm_custody_established",
    "distinct_phase_resource_established",
    "computational_advantage_established",
    "small_wall_crossing_established",
    "physical_waveform_execution_established",
    "physical_bits_replaced_with_pi",
    "unbounded_catalytic_computation_established",
):
    require(not ceiling[name], f"forbidden promotion appeared: {name}")

require(oracle["schema"] == "CAT_CAS_F17_SHARED_PHASE_PARITY_LEDGER_LADDER_CLOSURE_ORACLE_V1", "oracle schema changed")
require(oracle["classification"] == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE", "oracle classification changed")
require(oracle["verification_level"] == "INDEPENDENT_ORACLE_REEXECUTION", "oracle level changed")
require(not oracle["independence"]["imports_production_or_predecessor"], "oracle independence changed")
require(len(oracle["production_parity"]["exact"]) == 14, "oracle exact parity count changed")
require(len(oracle["production_parity"]["dual_field_structural"]) == 32, "oracle modular parity count changed")
require(all(item["agreement"] for item in oracle["production_parity"]["exact"]), "exact oracle parity failed")
require(all(item["agreement"] for item in oracle["production_parity"]["dual_field_structural"]), "modular oracle parity failed")
require(len(oracle["bounded_dense_checks"]) == 6 and all(item["agreement"] for item in oracle["bounded_dense_checks"]), "bounded dense oracle failed")

delta = oracle["exact_non_gaussian_signature"]
require(delta["nonzero"], "exact Grassmann-Plucker violation vanished")
require(delta["plucker_delta"] == [[64, 1], [0, 1], [0, 1], [0, 1], [0, 1], [-32, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [-32, 1], [0, 1], [0, 1], [0, 1]], "exact Grassmann-Plucker witness changed")
require(oracle["claim_ceiling"]["native_non_gaussian_signature_scope"] == "PRIMARY_WIDTH2_EXACT_GRASSMANN_PLUCKER_WITNESS_ONLY", "oracle non-Gaussian scope changed")
require(not oracle["claim_ceiling"]["other_exact_families_or_widths_non_gaussian_established"], "oracle non-Gaussian scope was broadened")

expected_grouped = {
    1: [2],
    2: [2, 4, 2],
    3: [2, 4, 8, 4, 2],
    4: [2, 4, 8, 16, 8, 4, 2],
}
expected_interleaved = {
    1: [2],
    2: [2, 4, 2],
    3: [2, 4, 4, 4, 2],
    4: [2, 4, 4, 4, 4, 4, 2],
}
expected_violations = {1: 0, 2: 1, 3: 9, 4: 36}
require(len(oracle["residual_rank_order_gauge"]) == 8, "rank diagnostic count changed")
for item in oracle["residual_rank_order_gauge"]:
    width = item["width"]
    require(item["odd_signature_entries_zero"], "odd signature entry survived")
    require(item["grouped_cut_rank_profile"] == expected_grouped[width], "grouped rank profile changed")
    require(item["column_interleaved_cut_rank_profile"] == expected_interleaved[width], "interleaved rank profile changed")
    require(item["all_order_optimal_maximum_cut_rank"] == (2 if width == 1 else 4), "all-order optimum changed")
    require(item["plucker_violation_count"] == expected_violations[width], "Plucker violation count changed")

for name in (
    "rank3_overmerge_changes_boundary",
    "semantic_field_port_mutation_changes_boundary",
    "grouped_order_is_width_exposing_not_a_semantic_mutation",
    "oracle_dense_enumeration_is_bounded_to_width4",
):
    require(oracle["controls"][name], f"oracle control failed: {name}")
require(not oracle["controls"]["accepted_recurrence_enumerates_even_sectors"], "oracle promoted sector enumeration")
require(not oracle["controls"]["accepted_recurrence_materializes_dense_signature"], "oracle promoted signature materialization")
require(oracle["lifecycle_receipt_consistency"]["limitation"].startswith("LIFECYCLE_FIELDS_REQUIRE_SEPARATE_SOURCE_AUDIT"), "lifecycle evidence ceiling changed")

for name in (
    "catvm_custody",
    "distinct_phase_resource",
    "computational_advantage",
    "small_wall_crossing",
    "physical_waveform_execution",
    "physical_bits_replaced_with_pi",
    "unbounded_catalytic_computation",
):
    require(not oracle["claim_ceiling"][name], f"oracle promoted forbidden claim: {name}")

print("QUALIFIED_F17_SHARED_PHASE_PARITY_LEDGER_LADDER_CLOSURE_STRICT_SCOPE")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle" "$0" > "$evidence_dir/SHA256SUMS"
