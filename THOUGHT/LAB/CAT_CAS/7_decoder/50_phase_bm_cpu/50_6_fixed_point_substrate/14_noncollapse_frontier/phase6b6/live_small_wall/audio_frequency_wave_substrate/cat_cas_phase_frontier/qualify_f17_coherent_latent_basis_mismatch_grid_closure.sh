#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_coherent_latent_basis_mismatch_grid_closure.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python_bin="$repo_root/.venv/bin/python"
production="$frontier_dir/f17_coherent_latent_basis_mismatch_grid_closure.py"
oracle="$frontier_dir/f17_coherent_latent_basis_mismatch_grid_closure_oracle.py"
sealed="$frontier_dir/F17_COHERENT_LATENT_BASIS_MISMATCH_GRID_CLOSURE_RESULTS.json"
sealed_oracle="$frontier_dir/F17_COHERENT_LATENT_BASIS_MISMATCH_GRID_CLOSURE_ORACLE_RESULTS.json"
production_result="$evidence_dir/F17_COHERENT_LATENT_BASIS_MISMATCH_GRID_CLOSURE_RESULTS.json"
oracle_result="$evidence_dir/F17_COHERENT_LATENT_BASIS_MISMATCH_GRID_CLOSURE_ORACLE_RESULTS.json"

mkdir -p "$evidence_dir"
if [[ ! -x "$python_bin" ]]; then
  echo "repository virtual environment is unavailable" >&2
  exit 1
fi

if rg -n '^[[:space:]]*(from|import)[[:space:]]+(f17_coherent_latent_basis_mismatch_grid_closure|f17_paired_phase_basis_holographic_matchgate_closure)([[:space:]]|$)' "$oracle"; then
  echo "independent M126 oracle imports production or its matchgate predecessor" >&2
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
    raise SystemExit("M126 transaction must call forward, projection, and inverse exactly once")
if not (calls["forward"][0].lineno < calls["project_boundary"][0].lineno < calls["inverse"][0].lineno):
    raise SystemExit("M126 response order is not forward -> final projection -> inverse")

for function_name in (
    "forward",
    "inverse",
    "module_boundary",
    "apply_controlled_grid_shear",
    "apply_fourier",
    "apply_chirp",
):
    function = functions[function_name]
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in {"product", "combinations", "permutations", "native_local_value", "bounded_native_holant"}:
            raise SystemExit(f"accepted M126 path calls forbidden expansion helper: {name}")
        if name == "project_boundary" and function_name != "execute_transaction":
            raise SystemExit(f"M126 internal stage projects a boundary: {function_name}")

inverse = functions["inverse"]
inverse_calls = [
    node.func.id
    for node in ast.walk(inverse)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
]
required = {"apply_controlled_grid_shear", "apply_fourier", "apply_chirp", "load_program"}
if not required.issubset(inverse_calls):
    raise SystemExit("M126 inverse does not rematerialize and reverse every resident operation class")
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

claim = "BOUNDED_EXACT_COHERENT_SHARED_F17_LATENT_PHASE_PORT_CONTROLS_TWO_GRID_WIDE_BASIS_MISMATCH_MODULES_SEPARATED_BY_NONCOMMUTING_FOURIER_AND_CHIRP_UPDATES_WITH_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_IDENTICAL_34_COORDINATE_CLASSICAL_RECURRENCE"
require(production["schema"] == "CAT_CAS_F17_COHERENT_LATENT_BASIS_MISMATCH_GRID_CLOSURE_V1", "production schema changed")
require(production["claim"] == claim, "production claim changed")
require(production["classification"] == "SOURCE_AUDITED_PACKAGE_LOCAL", "production classification changed")
require(production["verification_level"] == "PACKAGE_SELF_REVIEW", "production verification level changed")
require(len(production["exact_transactions"]) == 6, "exact transaction count changed")
require(len(production["dual_field_structural_transactions"]) == 20, "structural transaction count changed")

rows = production["exact_transactions"] + production["dual_field_structural_transactions"]
for item in rows:
    n = item["n"]
    edges = 2 * n * (n - 1)
    dimension = n * n // 2
    require(item["edge_count"] == edges, "edge count changed")
    require(item["resident_grid_weight_field_cells"] == 2 * edges, "grid weight residency changed")
    require(item["resident_latent_field_cells"] == 34, "latent residency changed")
    require(item["resident_phase_field_cells"] == 2 * edges + 34, "total residency changed")
    require(item["unresolved_latent_dimension"] == 17, "latent dimension changed")
    require(item["coherent_grid_modules"] == 2, "module count changed")
    require(item["module_boundary_evaluations"] == 68, "module rematerialization count changed")
    require(item["basis_mismatch_edge_contractions"] == 68 * edges, "basis mismatch accounting changed")
    require(item["fourier_transforms"] == 8, "Fourier transform count changed")
    require(item["chirp_field_multiplications"] == 68, "chirp accounting changed")
    require(item["coherent_shear_field_multiplications"] == 68, "shear multiplication accounting changed")
    require(item["coherent_shear_field_additions"] == 68, "shear addition accounting changed")
    require(item["determinant_matrix_dimension"] == dimension, "determinant dimension changed")
    require(item["determinant_matrix_field_cells"] == dimension * dimension, "determinant matrix-cell law changed")
    require(item["determinant_stats"]["determinant_calls"] == 68, "determinant call count changed")
    require(item["determinant_stats"]["maximum_named_dense_work_field_cells"] == 2 * dimension * dimension + 5, "named determinant work changed")
    require(item["maximum_named_transaction_transient_field_cells"] == max(2 * dimension * dimension + 5, 34), "full named transaction transient accounting changed")
    require(item["factor_load_additions"] == 2 * edges + 34, "factor load count changed")
    require(item["factor_unload_additions"] == 2 * edges + 34, "factor unload count changed")
    require(item["exact_phase_carrier_restored"], "resident carrier was not restored")
    require(item["same_backing"], "carrier backing identity changed")
    require(item["restoration_generation_increment"], "restoration generation changed")
    require(item["response_released_after_restoration"], "response ordering receipt changed")
    require(not item["snapshot_reload_used"] and not item["inverse_history_retained"], "snapshot or history entered accepted path")
    require(item["intermediate_projection_calls"] == 0 and item["final_projection_calls"] == 1, "projection law changed")
    require(not item["accepted_path_edge_assignment_enumeration"], "accepted path enumerated edges")
    require(not item["accepted_path_edge_local_native_signature_table_materialized"], "accepted path materialized edge-local signatures")
    require(not item["accepted_path_latent_diagonal_boundary_vector_materialized"], "accepted path materialized the latent diagonal boundary vector")
    require(item["final_boundary_payload_bits"] >= 1, "final boundary payload accounting changed")
    require(item["final_boundary_json_bytes"] >= 1, "final boundary serialization accounting changed")
    require(item["resident_carrier_restoration_class"] == "EXACT_ALGEBRAIC_RESTORATION", "resident restoration class changed")
    require(item["transient_determinant_buffer_restoration_class"] == "NO_RESTORATION_CLAIM", "transient restoration class changed")

reuse = production["reuse"]
require(reuse["fresh_restored_boundary_agreement"], "fresh/restored reuse parity failed")
require(reuse["fresh_restored_resource_signature_agreement"], "fresh/restored resource signature parity failed")
require(reuse["reused_resource_signature"] == reuse["fresh_resource_signature"], "fresh/restored resource signatures differ")
require(reuse["same_actual_backing_across_unrelated_programs"], "restored backing reuse failed")
require(reuse["generation_after_two_transactions"] == 2 and not reuse["baseline_reload_used"], "reuse generation or reload law changed")

controls = production["controls"]
for name in (
    "missing_inverse_detected",
    "wrong_inverse_ownership_detected",
    "premature_projection_rejected",
    "null_carrier_rejected",
    "reordered_inverse_detected",
    "semantic_basis_control_mutation_changes_boundary",
    "coherent_chirp_mutation_changes_boundary",
    "fourier_chirp_order_changes_latent_state",
    "snapshot_command_absent",
):
    require(controls[name], f"control failed: {name}")
require(not controls["intermediate_latent_vector_serialized"], "latent vector was serialized")
require(not controls["accepted_path_edge_assignment_enumeration"], "control promoted edge enumeration")
require(not controls["accepted_path_edge_local_native_signature_table_materialized"], "control promoted edge-local signature tables")
require(not controls["accepted_path_latent_diagonal_boundary_vector_materialized"], "control promoted latent diagonal boundary vectors")
require(production["matched_baselines"]["strongest_implemented"].startswith("IDENTICAL_34_COORDINATE"), "matched baseline changed")
require(not production["matched_baselines"]["phase_advantage_over_matched_classical"], "phase advantage was promoted")
require(production["resource_law"]["control_carrier_instances"] == 7, "control carrier accounting changed")
require(production["resource_law"]["control_full_or_partial_forward_executions"] == 6, "control forward accounting changed")
require(production["resource_law"]["control_and_verification_runs_are_sequential_not_added_to_accepted_peak"], "control peak classification changed")
require(production["resource_law"]["native_edge_local_signature_tables_materialized"] == 0, "edge-local signature resource law changed")
require(production["resource_law"]["latent_diagonal_boundary_vectors_materialized"] == 0, "latent vector resource law changed")
require(production["resource_law"]["compiled_load_or_unload_vectors_materialized"] == 0, "compiled payload vector reappeared")
require("FULL_EXACT_BIT_COMPLEXITY_NOT_ESTABLISHED" in production["resource_law"]["accepted_field_operation_work"], "exact bit-complexity caveat changed")

for name in (
    "arbitrary_latent_dimension_or_planar_holant_closure",
    "catvm_custody_established",
    "distinct_phase_resource_established",
    "computational_advantage_established",
    "small_wall_crossing_established",
    "physical_waveform_execution_established",
    "physical_bits_replaced_with_pi",
    "unbounded_catalytic_computation_established",
):
    require(not production["claim_ceiling"][name], f"forbidden production promotion appeared: {name}")

require(oracle["schema"] == "CAT_CAS_F17_COHERENT_LATENT_BASIS_MISMATCH_GRID_CLOSURE_ORACLE_V1", "oracle schema changed")
require(oracle["classification"] == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE", "oracle classification changed")
require(oracle["verification_level"] == "INDEPENDENT_ORACLE_REEXECUTION", "oracle level changed")
require(not oracle["independence"]["imports_production_or_matchgate_predecessor"], "oracle independence changed")
require(len(oracle["exact_production_parity"]) == 6 and all(item["agreement"] for item in oracle["exact_production_parity"]), "exact oracle parity failed")
require(len(oracle["dual_field_structural_parity"]) == 20 and all(item["agreement"] for item in oracle["dual_field_structural_parity"]), "structural oracle parity failed")
require(len(oracle["bounded_native_holant_parity"]) == 12 and all(item["agreement"] for item in oracle["bounded_native_holant_parity"]), "bounded native Holant parity failed")
require(len(oracle["basis_mismatch_identity_checks"]) == 4 and all(item["agreement"] for item in oracle["basis_mismatch_identity_checks"]), "basis identity checks failed")
require(oracle["control_receipt_consistency"]["all_required_controls_true"], "oracle control receipt check failed")
require(oracle["observed_resource_law"]["fixed_latent_direct_sum_rank"] == 17, "classical direct-sum rank changed")
require(oracle["observed_resource_law"]["oracle_materializes_fixed_latent_boundary_vector"], "oracle vector accounting changed")
require(oracle["observed_resource_law"]["oracle_latent_boundary_vector_field_cells"] == 17, "oracle vector size changed")
require(oracle["observed_resource_law"]["production_streams_each_latent_boundary_directly_into_resident_state"], "production streaming distinction changed")
require(oracle["claim_ceiling"]["identical_compact_classical_direct_sum_recurrence"], "classical collapse was removed")
for name in (
    "arbitrary_latent_geometry_or_planar_holant_closure",
    "catvm_custody",
    "distinct_phase_resource",
    "computational_advantage",
    "small_wall_crossing",
    "physical_waveform_execution",
    "physical_bits_replaced_with_pi",
    "unbounded_catalytic_computation",
):
    require(not oracle["claim_ceiling"][name], f"forbidden oracle promotion appeared: {name}")
PY

echo QUALIFIED_F17_COHERENT_LATENT_BASIS_MISMATCH_GRID_CLOSURE_STRICT_SCOPE
