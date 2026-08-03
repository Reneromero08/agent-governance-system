#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_exchange_symmetric_latent_geometry_closure.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python_bin="$repo_root/.venv/bin/python"
production="$frontier_dir/f17_exchange_symmetric_latent_geometry_closure.py"
oracle="$frontier_dir/f17_exchange_symmetric_latent_geometry_closure_oracle.py"
sealed="$frontier_dir/F17_EXCHANGE_SYMMETRIC_LATENT_GEOMETRY_CLOSURE_RESULTS.json"
sealed_oracle="$frontier_dir/F17_EXCHANGE_SYMMETRIC_LATENT_GEOMETRY_CLOSURE_ORACLE_RESULTS.json"
production_result="$evidence_dir/F17_EXCHANGE_SYMMETRIC_LATENT_GEOMETRY_CLOSURE_RESULTS.json"
oracle_result="$evidence_dir/F17_EXCHANGE_SYMMETRIC_LATENT_GEOMETRY_CLOSURE_ORACLE_RESULTS.json"

mkdir -p "$evidence_dir"
if [[ ! -x "$python_bin" ]]; then
  echo "repository virtual environment is unavailable" >&2
  exit 1
fi

if rg -n '^[[:space:]]*(from|import)[[:space:]]+(f17_exchange_symmetric_latent_geometry_closure|f17_paired_phase_basis_holographic_matchgate_closure)([[:space:]]|$)' "$oracle"; then
  echo "independent M127 oracle imports production or its matchgate predecessor" >&2
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
    raise SystemExit("M127 transaction must call forward, projection, and inverse exactly once")
if not (calls["forward"][0].lineno < calls["project_boundary"][0].lineno < calls["inverse"][0].lineno):
    raise SystemExit("M127 response order is not forward -> final projection -> inverse")

accepted = (
    "forward",
    "inverse",
    "apply_lifted_fourier",
    "apply_lifted_fourier_segment",
    "apply_lifted_scale",
    "apply_lifted_swap",
    "apply_lifted_shear",
    "apply_symmetric_chirp",
    "apply_orbit_shear",
    "module_boundary",
    "kasteleyn_matrix",
)
for function_name in accepted:
    for node in ast.walk(functions[function_name]):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in {
            "product", "permutations", "combinations", "independent_transaction",
            "labelled_fourier", "enumerate_assignments", "native_truth_table",
        }:
            raise SystemExit(f"accepted M127 path calls forbidden expansion helper: {name}")
        if name == "project_boundary":
            raise SystemExit(f"M127 internal stage projects a boundary: {function_name}")

orbit_shear = functions["apply_orbit_shear"]
boundary_calls = [
    node for node in ast.walk(orbit_shear)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "module_boundary"
]
if len(boundary_calls) != 1:
    raise SystemExit("M127 orbit shear does not stream exactly one boundary call site")
if any(
    isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
    and any(call is boundary_calls[0] for call in ast.walk(node))
    for node in ast.walk(orbit_shear)
):
    raise SystemExit("M127 orbit shear materializes a boundary vector")

inverse_calls = [
    node.func.id
    for node in ast.walk(functions["inverse"])
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
]
required = {"apply_orbit_shear", "apply_lifted_fourier", "apply_symmetric_chirp", "load_program"}
if not required.issubset(inverse_calls):
    raise SystemExit("M127 inverse does not reverse every resident operation class")
PY

PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 LC_ALL=C nice -n 10 \
  "$python_bin" -X dev "$production" --output "$production_result"
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 LC_ALL=C nice -n 10 \
  "$python_bin" -X dev "$oracle" --production "$production_result" --output "$oracle_result"

cmp "$production_result" "$sealed"
cmp "$oracle_result" "$sealed_oracle"

PYTHONDONTWRITEBYTECODE=1 "$python_bin" - "$production_result" "$oracle_result" <<'PY'
import json
import math
import sys

production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))

def require(condition, message):
    if not condition:
        raise SystemExit(message)

claim = "BOUNDED_EXACT_GROWING_EXCHANGE_SYMMETRIC_F17_LATENT_GEOMETRY_USES_DEGREE_K_OCCUPATION_ORBITS_WITH_EXACT_LIFTED_FOURIER_NON_SUM_ONLY_POWER_SUM_GRID_CONTROLS_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_THE_IDENTICAL_CLASSICAL_ORBIT_RECURRENCE"
require(production["schema"] == "CAT_CAS_F17_EXCHANGE_SYMMETRIC_LATENT_GEOMETRY_CLOSURE_V1", "production schema changed")
require(production["claim"] == claim, "production claim changed")
require(production["classification"] == "SOURCE_AUDITED_PACKAGE_LOCAL", "production classification changed")
require(production["verification_level"] == "PACKAGE_SELF_REVIEW", "production verification level changed")
require(len(production["exact_transactions"]) == 3, "exact transaction count changed")
require(len(production["dual_field_structural_transactions"]) == 9, "structural transaction count changed")

rows = production["exact_transactions"] + production["dual_field_structural_transactions"]
for item in rows:
    k = item["k"]
    h = math.comb(k + 16, 16)
    require(item["occupation_dimension"] == h, "occupation dimension changed")
    require(item["labelled_tensor_dimension_not_materialized"] == 17 ** k, "labelled dimension receipt changed")
    require(item["resident_grid_weight_field_cells"] == 48, "grid residency changed")
    require(item["resident_orbit_field_cells"] == 2 * h, "orbit residency changed")
    require(item["resident_phase_field_cells"] == 48 + 2 * h, "total residency changed")
    require(item["public_occupation_topology_integer_cells"] == 18 * h, "topology accounting changed")
    require(item["public_program_integer_cells"] == 146 and item["public_program_json_bytes"] >= 146, "public program accounting changed")
    require(item["public_grid_edge_coordinate_integer_cells"] == 96, "public grid edge accounting changed")
    require(item["public_grid_vertex_coordinate_integer_cells"] == 32, "public grid vertex accounting changed")
    require(item["public_fourier_plan_operations"] == 578, "retained Fourier plan accounting changed")
    require(item["public_fourier_plan_field_cells"] == 578, "retained Fourier coefficient accounting changed")
    require(item["public_fourier_plan_compile_maximum_named_field_cells"] == 1496, "Fourier compile field accounting changed")
    require(item["public_fourier_plan_fingerprint_maximum_record_json_bytes"] >= 1, "Fourier fingerprint accounting absent")
    require(item["module_boundary_evaluations"] == 4 * h, "module rematerialization count changed")
    require(item["basis_mismatch_edge_contractions"] == 96 * h, "grid edge accounting changed")
    require(item["power_sum_evaluations"] == 4 * h * k, "power-sum accounting changed")
    require(item["lifted_fourier_vector_transforms"] == 8, "lifted Fourier count changed")
    require(item["lifted_elementary_operations"] == 8 * 289, "lifted elementary operation count changed")
    require(item["symmetric_chirp_multiplications"] == 4 * h, "chirp accounting changed")
    require(item["orbit_shear_multiplications"] == 4 * h, "orbit shear multiplication count changed")
    require(item["orbit_shear_additions"] == 4 * h, "orbit shear addition count changed")
    require(item["determinant_matrix_dimension"] == 8, "determinant dimension changed")
    require(item["determinant_stats"]["determinant_calls"] == 4 * h, "determinant call count changed")
    require(item["maximum_named_transaction_transient_field_cells"] == 133, "named transient field-cell ceiling changed")
    require(item["maximum_named_transaction_transient_integer_cells"] == 48, "named transient integer-cell ceiling changed")
    require(item["maximum_named_transaction_transient_payload_bits_upper_bound"] >= 133, "transient payload accounting absent")
    require(item["factor_load_additions"] == 49 and item["factor_unload_additions"] == 49, "load/unload accounting changed")
    require(item["maximum_resident_payload_bits"] >= item["resident_phase_field_cells"], "resident payload accounting absent")
    require(item["maximum_resident_field_value_payload_bits"] >= 1, "resident value-height accounting absent")
    require(item["public_fourier_plan_maximum_coefficient_payload_bits"] >= 1, "plan value-height accounting absent")
    require(item["exact_phase_carrier_restored"], "resident carrier was not restored")
    require(item["same_backing"], "carrier backing identity changed")
    require(item["restoration_generation_increment"], "restoration generation changed")
    require(item["response_released_after_restoration"], "response ordering receipt changed")
    require(item["intermediate_projection_calls"] == 0 and item["final_projection_calls"] == 1, "projection law changed")
    require(not item["snapshot_reload_used"] and not item["inverse_history_retained"], "snapshot or history entered accepted path")
    for name in (
        "accepted_path_labelled_tensor_materialized",
        "accepted_path_dense_occupation_operator_materialized",
        "accepted_path_occupation_boundary_vector_materialized",
        "accepted_path_assignment_or_relation_table_materialized",
    ):
        require(not item[name], f"accepted path expanded forbidden state: {name}")
    require(item["resident_carrier_restoration_class"] == "EXACT_ALGEBRAIC_RESTORATION", "resident restoration class changed")
    require(item["transient_buffers_restoration_class"] == "NO_RESTORATION_CLAIM", "transient restoration class changed")

reuse = production["reuse"]
require(reuse["fresh_restored_boundary_agreement"], "fresh/restored boundary parity failed")
require(reuse["fresh_restored_resource_signature_agreement"], "fresh/restored resource parity failed")
require(reuse["same_actual_backing_across_unrelated_programs"], "restored backing reuse failed")
require(reuse["generation_after_two_transactions"] == 2 and not reuse["baseline_reload_used"], "reuse generation or reload law changed")

controls = production["controls"]
for name in (
    "missing_inverse_detected",
    "wrong_inverse_ownership_detected",
    "premature_projection_rejected",
    "null_carrier_rejected",
    "reordered_inverse_detected",
    "symmetry_breaking_descriptor_rejected",
    "particle_permutation_preserves_histogram_invariants",
    "total_sum_overmerge_changes_boundary",
    "missing_power_sum_changes_boundary",
    "semantic_control_mutation_changes_boundary",
):
    require(controls[name], f"control failed: {name}")
require(not controls["intermediate_orbit_state_serialized"], "intermediate orbit state was serialized")
require(not controls["catvm_boundary_claimed"], "direct process was promoted to CATVM")

require(production["source_scope"]["occupation_dimensions"] == {"1": 17, "2": 153, "3": 969, "4": 4845}, "observed dimension law changed")
require(production["matched_baselines"]["strongest_implemented"].startswith("IDENTICAL_H_K_COORDINATE"), "matched baseline changed")
require(not production["matched_baselines"]["phase_advantage_over_matched_classical"], "phase advantage was promoted")
ceiling = production["claim_ceiling"]
require(ceiling["exchange_symmetry_is_required_not_inferred"], "exchange-symmetry premise was weakened")
require(not ceiling["original_labelled_open_chain_family_compressed"], "labelled family was promoted")
for name in (
    "fixed_rank_across_growing_k",
    "catvm_custody_established",
    "distinct_phase_resource_established",
    "computational_advantage_established",
    "small_wall_crossing_established",
    "physical_waveform_execution_established",
    "physical_bits_replaced_with_pi",
    "unbounded_catalytic_computation_established",
):
    require(not ceiling[name], f"strict ceiling was promoted: {name}")

require(oracle["schema"] == "CAT_CAS_F17_EXCHANGE_SYMMETRIC_LATENT_GEOMETRY_CLOSURE_ORACLE_V1", "oracle schema changed")
require(oracle["classification"] == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE", "oracle classification changed")
require(oracle["verification_level"] == "INDEPENDENT_ORACLE_REEXECUTION", "oracle verification level changed")
require(all(item["agreement"] and item["independent_labelled_forward_inverse_restored"] for item in oracle["exact_production_parity"]), "exact labelled oracle parity failed")
require(all(item["agreement"] and item["independent_labelled_forward_inverse_restored"] for item in oracle["dual_field_structural_parity"]), "dual-field labelled oracle parity failed")
require(all(item["full_power_sums_separate_all_histograms"] for item in oracle["power_sum_separation"]), "power sums did not separate the declared histograms")
require(all(item["total_sum_alone_overmerges"] for item in oracle["power_sum_separation"] if item["k"] >= 2), "total-sum overmerge was not independently reproduced")
require(all(item["exchange_invariant"] for item in oracle["particle_permutation_equivariance"]), "particle permutation equivariance failed")
require(oracle["control_receipt_consistency"]["all_required_controls_true"], "control receipt audit failed")
require(oracle["observed_resource_law"]["strongest_matched_classical_baseline"].startswith("IDENTICAL_OCCUPATION_ORBIT"), "oracle baseline classification changed")
PY

echo "QUALIFIED_F17_EXCHANGE_SYMMETRIC_LATENT_GEOMETRY_CLOSURE_STRICT_SCOPE"
