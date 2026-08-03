#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_paired_phase_basis_holographic_matchgate_closure.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python_bin="$repo_root/.venv/bin/python"
production="$frontier_dir/f17_paired_phase_basis_holographic_matchgate_closure.py"
oracle="$frontier_dir/f17_paired_phase_basis_holographic_matchgate_closure_oracle.py"
sealed="$frontier_dir/F17_PAIRED_PHASE_BASIS_HOLOGRAPHIC_MATCHGATE_CLOSURE_RESULTS.json"
sealed_oracle="$frontier_dir/F17_PAIRED_PHASE_BASIS_HOLOGRAPHIC_MATCHGATE_CLOSURE_ORACLE_RESULTS.json"
production_result="$evidence_dir/F17_PAIRED_PHASE_BASIS_HOLOGRAPHIC_MATCHGATE_CLOSURE_RESULTS.json"
oracle_result="$evidence_dir/F17_PAIRED_PHASE_BASIS_HOLOGRAPHIC_MATCHGATE_CLOSURE_ORACLE_RESULTS.json"

mkdir -p "$evidence_dir"
if [[ ! -x "$python_bin" ]]; then
  echo "repository virtual environment is unavailable" >&2
  exit 1
fi

if rg -n '^[[:space:]]*(from|import)[[:space:]]+(f17_paired_phase_basis_holographic_matchgate_closure|f17_nonlinear_canonical_mps_separator_chart)([[:space:]]|$)' "$oracle"; then
  echo "independent M125 oracle imports production or its predecessor" >&2
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
    raise SystemExit("M125 transaction must call forward, final projection, and inverse exactly once")
if not (calls["forward"][0].lineno < calls["project_boundary"][0].lineno < calls["inverse"][0].lineno):
    raise SystemExit("M125 response ordering is not forward -> final projection -> inverse")

for function_name in (
    "paired_basis_contraction",
    "compile_kasteleyn_matrix",
    "contract_resident_hologram",
    "forward",
    "inverse",
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
        if name in {
            "product",
            "combinations",
            "permutations",
            "bounded_native_holant",
            "native_local_value",
        }:
            raise SystemExit(f"accepted M125 path calls bounded expansion helper: {name}")

inverse = functions["inverse"]
inverse_calls = [
    node for node in ast.walk(inverse)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
]
rematerialize = [node for node in inverse_calls if node.func.id == "contract_resident_hologram"]
unload = [node for node in inverse_calls if node.func.id == "load_factors"]
if len(rematerialize) != 1 or len(unload) != 1 or rematerialize[0].lineno >= unload[0].lineno:
    raise SystemExit("M125 inverse must rematerialize before unloading the actual resident factors")
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

require(production["schema"] == "CAT_CAS_F17_PAIRED_PHASE_BASIS_HOLOGRAPHIC_MATCHGATE_CLOSURE_V1", "production schema changed")
require(production["classification"] == "SOURCE_AUDITED_PACKAGE_LOCAL", "production classification changed")
require(production["verification_level"] == "PACKAGE_SELF_REVIEW", "production verification level changed")
require(len(production["exact_transactions"]) == 6, "exact transaction count changed")
require(len(production["dual_field_structural_transactions"]) == 24, "structural transaction count changed")

rows = production["exact_transactions"] + production["dual_field_structural_transactions"]
for item in rows:
    n = item["n"]
    edge_count = 2 * n * (n - 1)
    dimension = n * n // 2
    require(item["edge_count"] == edge_count, "edge-count law changed")
    require(item["public_program_integer_cells"] == edge_count + 3, "public program accounting changed")
    require(item["compiled_topology_edge_records"] == edge_count, "topology record accounting changed")
    require(item["resident_phase_field_cells"] == edge_count + 8, "resident phase-cell law changed")
    require(item["basis_contraction_evaluations"] == 2, "basis contraction evaluation count changed")
    require(item["basis_contraction_field_multiplications"] == 16, "basis contraction multiplication count changed")
    require(item["edge_identity_substitutions"] == 2 * edge_count, "edge identity-substitution count changed")
    require(item["kasteleyn_matrix_dimension"] == dimension, "Kasteleyn dimension changed")
    require(item["kasteleyn_matrix_field_cells"] == dimension * dimension, "Kasteleyn matrix-cell law changed")
    require(item["final_boundary_field_cells"] == 1, "final boundary accounting changed")
    require(item["restoration_verification_field_cells"] == edge_count + 9, "restoration verification accounting changed")
    require(item["controller_backend_traffic_bytes"] == 0, "direct-process traffic classification changed")
    stats = item["determinant_stats"]
    require(stats["determinant_calls"] == 2, "forward/inverse determinant count changed")
    require(stats["maximum_caller_matrix_field_cells"] == dimension * dimension, "caller matrix accounting changed")
    require(stats["maximum_elimination_copy_field_cells"] == dimension * dimension, "elimination-copy accounting changed")
    require(stats["maximum_named_dense_work_field_cells"] == 2 * dimension * dimension + 5, "named dense work-cell accounting changed")
    require(stats["maximum_named_dense_work_payload_bits_upper_bound"] > 0, "named dense payload upper bound is absent")
    require(item["exact_phase_carrier_restored"], "resident phase carrier was not restored")
    require(item["same_backing"], "carrier backing identity changed")
    require(item["restoration_generation_increment"], "restoration generation changed")
    require(item["response_released_after_restoration"], "response ordering receipt changed")
    require(not item["snapshot_reload_used"] and not item["inverse_history_retained"], "reload or retained history entered accepted path")
    require(not item["accepted_path_native_signature_table_materialized"], "accepted path materialized native signatures")
    require(not item["accepted_path_edge_assignment_enumeration"], "accepted path enumerated edge assignments")
    require(item["intermediate_boundary_projection_calls"] == 0 and item["final_boundary_projection_calls"] == 1, "projection law changed")
    require(item["resident_carrier_restoration_class"] == "EXACT_ALGEBRAIC_RESTORATION", "resident restoration class changed")
    require(item["transient_determinant_buffer_restoration_class"] == "NO_RESTORATION_CLAIM", "transient restoration class changed")

reuse = production["reuse"]
require(reuse["fresh_restored_boundary_agreement"], "fresh/restored reuse parity failed")
require(reuse["same_actual_backing_across_unrelated_programs"], "restored reuse backing changed")
require(reuse["generation_after_two_transactions"] == 2 and not reuse["baseline_reload_used"], "reuse generation or reload law changed")

controls = production["controls"]
for name in (
    "bounded_n2_native_holant_matches_holographic_determinant",
    "degree4_left_and_right_native_witnesses_have_both_parities_nonzero",
    "paired_basis_exact_identity",
    "public_grid_face_signing_valid",
    "missing_inverse_detected",
    "wrong_inverse_ownership_detected",
    "premature_projection_rejected",
    "null_carrier_rejected",
    "basis_mutation_rejected",
    "reordered_inverse_detected",
    "snapshot_command_absent",
):
    require(controls[name], f"control failed: {name}")
require(not controls["accepted_path_native_signature_table_materialized"], "control promoted signature table")
require(not controls["accepted_path_edge_assignment_enumeration"], "control promoted edge enumeration")
require(production["matched_baselines"]["strongest_implemented"] == "IDENTICAL_PAIRED_BASIS_COMPILATION_AND_KASTELEYN_DETERMINANT", "strongest matched baseline changed")
require(not production["matched_baselines"]["phase_advantage_over_matched_classical"], "phase advantage was promoted")

ceiling = production["claim_ceiling"]
for name in (
    "arbitrary_native_signature_or_planar_holant_closure",
    "catvm_custody_established",
    "distinct_phase_resource_established",
    "computational_advantage_established",
    "small_wall_crossing_established",
    "physical_waveform_execution_established",
    "physical_bits_replaced_with_pi",
    "unbounded_catalytic_computation_established",
):
    require(not ceiling[name], f"forbidden production promotion appeared: {name}")

require(oracle["schema"] == "CAT_CAS_F17_PAIRED_PHASE_BASIS_HOLOGRAPHIC_MATCHGATE_CLOSURE_ORACLE_V1", "oracle schema changed")
require(oracle["classification"] == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE", "oracle classification changed")
require(oracle["verification_level"] == "INDEPENDENT_ORACLE_REEXECUTION", "oracle verification level changed")
require(not oracle["independence"]["imports_production_or_predecessor"], "oracle independence changed")
require(len(oracle["exact_matching_parity"]) == 6, "exact matching oracle count changed")
require(len(oracle["dual_field_row_dp_parity"]) == 24, "row-DP oracle count changed")
require(all(item["agreement"] for item in oracle["exact_matching_parity"]), "exact matching parity failed")
require(all(item["agreement"] for item in oracle["dual_field_row_dp_parity"]), "row-DP parity failed")
require([item["perfect_matching_count"] for item in oracle["exact_matching_parity"][:3]] == [2, 36, 6728], "exact matching counts changed")
require(len(oracle["exact_native_holant_parity"]) == 2, "native Holant oracle count changed")
require(all(item["paired_basis_identity"] and item["native_matching_agreement"] for item in oracle["exact_native_holant_parity"]), "native paired-basis parity failed")
require(len(oracle["native_mixed_parity_witnesses"]) == 2, "native parity-witness count changed")
for item in oracle["native_mixed_parity_witnesses"]:
    require(all(item[name] for name in ("left_even_nonzero", "left_odd_nonzero", "right_even_nonzero", "right_odd_nonzero")), "native mixed-parity witness failed")

for name in (
    "paired_basis_mutation_changes_native_boundary",
    "semantic_edge_phase_mutation_changes_boundary",
    "public_grid_face_signing_valid",
    "direct_native_assignment_control_bounded_to_n2",
):
    require(oracle["controls"][name], f"oracle control failed: {name}")
require(not oracle["controls"]["production_accepted_path_signature_tables_required"], "oracle promoted signature tables")
require(not oracle["controls"]["production_accepted_path_edge_assignments_required"], "oracle promoted edge enumeration")
require(oracle["lifecycle_receipt_consistency"]["limitation"].startswith("LIFECYCLE_RECEIPTS_REQUIRE_SEPARATE_SOURCE_AUDIT"), "lifecycle evidence ceiling changed")

for name in (
    "arbitrary_planar_holant_closure",
    "catvm_custody",
    "distinct_phase_resource",
    "computational_advantage",
    "small_wall_crossing",
    "physical_waveform_execution",
    "physical_bits_replaced_with_pi",
    "unbounded_catalytic_computation",
):
    require(not oracle["claim_ceiling"][name], f"forbidden oracle promotion appeared: {name}")

print("QUALIFIED_F17_PAIRED_PHASE_BASIS_HOLOGRAPHIC_MATCHGATE_CLOSURE_STRICT_SCOPE")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle" "$0" > "$evidence_dir/SHA256SUMS"
