#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 EVIDENCE_DIR" >&2
    exit 2
fi

evidence_dir=$1
mkdir -p "$evidence_dir"
frontier_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python_bin="$repo_root/.venv/bin/python"
source_path="$frontier_dir/f17_anisotropic_quartic_radial_phase_quotient_closure.py"
oracle_path="$frontier_dir/f17_anisotropic_quartic_radial_phase_quotient_closure_oracle.py"
production_seal="$frontier_dir/F17_ANISOTROPIC_QUARTIC_RADIAL_PHASE_QUOTIENT_CLOSURE_RESULTS.json"
oracle_seal="$frontier_dir/F17_ANISOTROPIC_QUARTIC_RADIAL_PHASE_QUOTIENT_CLOSURE_ORACLE_RESULTS.json"
production_replay="$evidence_dir/production.json"
oracle_replay="$evidence_dir/oracle.json"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export LC_ALL=C
export PYTHONPATH="$frontier_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

"$python_bin" - "$source_path" "$oracle_path" <<'PY'
import ast
import pathlib
import sys

source_path, oracle_path = map(pathlib.Path, sys.argv[1:])
source_tree = ast.parse(source_path.read_text(encoding="utf-8"), str(source_path))
oracle_tree = ast.parse(oracle_path.read_text(encoding="utf-8"), str(oracle_path))


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SystemExit(f"missing function {name}")


accepted = function(source_tree, "execute_transaction")
calls = []
for node in ast.walk(accepted):
    if not isinstance(node, ast.Call):
        continue
    if isinstance(node.func, ast.Name):
        calls.append((node.lineno, node.func.id))
    elif isinstance(node.func, ast.Attribute):
        calls.append((node.lineno, node.func.attr))
ordered = [name for _, name in sorted(calls)]
cursor = 0
for required in ("begin_forward", "forward", "state_commitment", "project", "inverse"):
    try:
        cursor = ordered.index(required, cursor) + 1
    except ValueError as exc:
        raise SystemExit(f"accepted transaction lacks ordered {required}") from exc

for name in ("begin_forward", "forward", "apply_phase", "apply_fourier", "project", "inverse"):
    text = ast.unparse(function(source_tree, name))
    for forbidden in (
        "dense_coordinate_boundary",
        "verify_public_quotient_invariance",
        "coordinates()",
        "assignments",
        "truth_table",
        "snapshot",
    ):
        if forbidden in text:
            raise SystemExit(f"accepted {name} contains forbidden {forbidden}")

for node in oracle_tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        imports = [alias.name for alias in node.names]
        if any(name.startswith("f17_anisotropic_quartic_radial_phase_quotient_closure") for name in imports):
            raise SystemExit("independent oracle imports production")

required_oracle = {
    "matrix",
    "matrix_vector",
    "gate",
    "execute",
    "verify_transform",
    "dense",
    "commitment",
}
present = {node.name for node in ast.walk(oracle_tree) if isinstance(node, ast.FunctionDef)}
missing = required_oracle - present
if missing:
    raise SystemExit(f"independent oracle lacks {sorted(missing)}")
PY

"$python_bin" -m py_compile "$source_path" "$oracle_path"
nice -n 10 "$python_bin" -X dev "$source_path" --output "$production_replay"
cmp "$production_seal" "$production_replay"
nice -n 10 "$python_bin" -X dev "$oracle_path" \
    --production "$production_replay" \
    --production-source "$source_path" \
    --output "$oracle_replay"
cmp "$oracle_seal" "$oracle_replay"

"$python_bin" - "$production_replay" "$oracle_replay" <<'PY'
import json
import pathlib
import sys

production = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
oracle = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))

assert production["classification"] == "SOURCE_AUDITED_PACKAGE_LOCAL"
assert production["verification_level"] == "PACKAGE_SELF_REVIEW"
assert oracle["classification"] == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
assert oracle["verification_level"] == "INDEPENDENT_ORACLE_REEXECUTION"
assert oracle["passed"]

scope = production["source_scope"]
assert scope["coordinate_field"] == "F17_SQUARED"
assert scope["anisotropic_norm"] == "X_SQUARED_MINUS_3_Y_SQUARED"
assert scope["norm_shell_count"] == 17
assert scope["norm_shell_sizes"] == [1, *([18] * 16)]
assert scope["exact_depths"] == [1, 2, 4, 8, 16, 32, 64]
assert scope["structural_depths"] == [1, 2, 4, 8, 16, 32, 64, 128]

transactions = production["exact_transactions"] + production["structural_transactions"]
assert len(production["exact_transactions"]) == 7
assert len(production["structural_transactions"]) == 48
assert len(transactions) == 55
for item in transactions:
    assert item["resident_relation_quotient_field_cells"] == 17
    assert item["represented_coordinate_cells"] == 289
    assert item["maximum_update_scratch_field_cells"] == 20
    assert item["maximum_live_resident_plus_update_scratch_field_cells"] == 37
    assert item["maximum_projection_scratch_field_cells"] == 6
    assert item["maximum_live_resident_plus_projection_scratch_field_cells"] == 23
    assert item["compiled_public_fourier_kernel_field_cells"] == 289
    assert item["public_geometry_coordinate_visits_at_compile"] == 5202
    assert item["public_fourier_involution_check_multiply_accumulates_at_compile"] == 4913
    assert item["maximum_live_public_compile_exact_field_cells_excluding_expression_temporaries"] == 306
    assert item["accepted_path_assignment_or_truth_table_cells"] == 0
    assert item["accepted_path_public_coordinate_expansion_cells"] == 0
    assert not item["intermediate_relation_quotient_exposed"]
    assert item["final_projection_calls"] == 1
    assert item["response_released_after_restoration"]
    assert item["same_backing"]
    assert item["restored_exact_zero"]
    assert item["initial_restored_digest_equal"]
    assert item["package_local_restoration_count_after"] == item["package_local_restoration_count_before"] + 1
    assert item["inverse_history_cells"] == 0
    assert item["inverse_operations_rematerialized_from_public_topology"]
    assert not item["snapshot_reload_used"]
    assert item["resident_carrier_restoration_class"] == "EXACT_ALGEBRAIC_RESTORATION"

false_controls = {
    "snapshot_command_available",
    "intermediate_projection_available",
    "nonradial_phase_descriptor_available",
}
for key, value in production["controls"].items():
    assert (not value) if key in false_controls else value, key

verification = production["quotient_invariance_verification_outside_accepted_transaction"]
assert len(verification) == 3
for item in verification:
    assert item["all_289_target_rows_factor_through_norm"]
    assert item["quotient_row_checks"] == 4913
    assert item["target_coordinate_visits"] == 289
    assert item["source_coordinate_visits"] == 83521
    assert item["total_coordinate_visits"] == 83810

reuse = production["reuse"]
assert reuse["same_original_backing"]
assert reuse["fresh_restored_boundary_equal"]
assert reuse["fresh_restored_resource_signature_equal"]
assert reuse["package_local_restoration_count"] == 2
assert reuse["restored_exact_zero"]
assert not reuse["snapshot_reload_used"]
assert reuse["inverse_history_cells"] == 0

payloads = [item["maximum_resident_payload_bits"] for item in production["exact_transactions"]]
assert payloads == [2203, 2243, 4364, 8709, 17552, 35394, 70992]
assert production["resource_law"]["public_topology_compile_coordinate_visits_per_algebra"] == 5202
assert production["resource_law"]["exhaustive_invariance_verification_total_coordinate_visits_per_algebra"] == 83810
assert production["resource_law"]["maximum_live_one_compiled_kernel_plus_dense_verification_buffers_exact_field_cells_per_algebra_excluding_arithmetic_temporaries"] == 1156
assert production["resource_law"]["maximum_live_all_three_compiled_kernels_plus_dense_verification_buffers_exact_field_cells_excluding_arithmetic_temporaries"] == 1734
assert len(production["matched_classical_baselines"]) == 55
for item in production["matched_classical_baselines"]:
    assert item["boundary_equal"]
    assert item["resident_exact_field_cells"] == 17
    assert item["maximum_update_scratch_exact_field_cells"] == 20
    assert item["maximum_live_state_plus_scratch_exact_field_cells"] == 37
    assert item["maximum_projection_scratch_exact_field_cells"] == 6
    assert item["maximum_live_state_plus_projection_scratch_exact_field_cells"] == 23
    assert item["compiled_public_kernel_exact_field_cells"] == 289
    assert not item["universal_optimality_or_lower_bound_claimed"]
assert not production["matched_baseline"]["universal_strongest_classical_or_lower_bound_claimed"]
assert not production["matched_baseline"]["comparison_establishes_distinct_phase_resource"]
assert not production["matched_baseline"]["comparison_establishes_computational_advantage"]

exact = oracle["exact_reexecution"]
assert exact["transaction_count"] == 7
assert exact["all_boundaries_equal"]
assert exact["all_final_state_commitments_equal"]
assert exact["all_resident_payload_maxima_equal"]
assert exact["all_restore_seed_exactly"]
assert exact["resident_payload_tuple"] == payloads
assert not exact["update_scratch_payload_tuple_independently_reproduced"]
assert len(oracle["dense_coordinate_controls"]) == 4
for item in oracle["dense_coordinate_controls"]:
    assert item["boundary_equal"]
    assert item["dense_state_remains_radial"]
    assert item["all_17_shell_values_equal"]
for item in oracle["structural_reexecution"]:
    assert item["all_structural_boundaries_equal"]
    assert all(item["transform_checks"].values())
assert all(oracle["mutations"].values())
assert oracle["restoration_class"] == "EXACT_ALGEBRAIC_RESTORATION"

for claim in production["claim_boundary"]["not_established"]:
    assert claim
PY

echo QUALIFIED_F17_ANISOTROPIC_QUARTIC_RADIAL_PHASE_QUOTIENT_CLOSURE_STRICT_SCOPE
