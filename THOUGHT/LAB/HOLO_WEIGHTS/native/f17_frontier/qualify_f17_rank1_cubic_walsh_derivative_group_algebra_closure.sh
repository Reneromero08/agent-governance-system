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
source_path="$frontier_dir/f17_rank1_cubic_walsh_derivative_group_algebra_closure.py"
oracle_path="$frontier_dir/f17_rank1_cubic_walsh_derivative_group_algebra_closure_oracle.py"
production_seal="$frontier_dir/F17_RANK1_CUBIC_WALSH_DERIVATIVE_GROUP_ALGEBRA_CLOSURE_RESULTS.json"
oracle_seal="$frontier_dir/F17_RANK1_CUBIC_WALSH_DERIVATIVE_GROUP_ALGEBRA_CLOSURE_ORACLE_RESULTS.json"
production_replay="$evidence_dir/production.json"
oracle_replay="$evidence_dir/oracle.json"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export LC_ALL=C
export PYTHONPATH="$frontier_dir"

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
for required in ("begin_forward", "forward", "stream_state_commitment", "projected_boundary", "inverse"):
    try:
        cursor = ordered.index(required, cursor) + 1
    except ValueError as exc:
        raise SystemExit(f"accepted transaction lacks ordered {required}") from exc

for name in ("forward", "projected_boundary", "inverse"):
    node = function(source_tree, name)
    text = ast.unparse(node)
    for forbidden in ("product(", "assignments", "truth_table", "permanent", "snapshot"):
        if forbidden in text:
            raise SystemExit(f"accepted {name} contains forbidden {forbidden}")

for node in oracle_tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        imports = [alias.name for alias in node.names]
        if any(name.startswith("f17_rank1_cubic_walsh_derivative_group_algebra_closure") for name in imports):
            raise SystemExit("independent oracle imports production")

required_oracle = {
    "descriptor",
    "canonical_rows",
    "rank_mod17",
    "forward_rows",
    "inverse_rows",
    "project_rows",
    "residue_class_boundary",
    "direct_assignment_boundary",
    "serialize_payload_bits",
}
present = {node.name for node in ast.walk(oracle_tree) if isinstance(node, ast.FunctionDef)}
missing = required_oracle - present
if missing:
    raise SystemExit(f"independent oracle lacks {sorted(missing)}")
PY

"$python_bin" -m py_compile "$source_path" "$oracle_path"
nice -n 10 "$python_bin" -X dev "$source_path" --output "$production_replay"
cmp "$production_seal" "$production_replay"
nice -n 10 "$python_bin" -X dev "$oracle_path" --production "$production_replay" --output "$oracle_replay"
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
assert production["source_scope"]["exact_branch_pair_round_cases"] == [
    [1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128], [128, 256]
]
transactions = production["exact_transactions"] + production["dual_field_structural_transactions"]
assert len(transactions) == 20
for item in transactions:
    certificate = item["signature_certificate"]
    assert certificate["signature_rank_over_f17"] == 1
    assert certificate["canonical_group_algebra_chart_field_cells_for_signature_rank"] == 34
    assert certificate["rank_one_34_cell_chart_accepted"]
    assert not certificate["sampling_or_truth_table_equivalence_used"]
    assert not certificate["assignment_expansion_used"]
    assert item["resident_group_algebra_field_cells"] == 34
    assert item["maximum_update_scratch_field_cells"] == 34
    assert item["accepted_path_branch_assignment_cells"] == 0
    assert item["accepted_path_truth_table_cells"] == 0
    assert item["accepted_path_projected_binomial_factor_list"] == 0
    assert not item["intermediate_group_algebra_coefficients_exposed"]
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

reuse = production["reuse"]
assert reuse["same_original_backing"]
assert reuse["fresh_restored_boundary_equal"]
assert reuse["fresh_restored_resource_signature_equal"]
assert reuse["package_local_restoration_count"] == 2
assert reuse["restored_exact_zero"]
assert not reuse["baseline_reload"]
assert reuse["inverse_history_cells"] == 0

false_controls = {"snapshot_command_available", "intermediate_projection_available"}
for key, value in production["controls"].items():
    assert (not value) if key in false_controls else value, key

assert len(oracle["transaction_parity"]) == 20
for item in oracle["transaction_parity"]:
    assert item["program_fingerprint_equal"]
    assert item["signature_rank"] == 1
    assert item["boundary_equal"]
    assert item["residue_class_boundary_equal"]
    assert item["state_commitment_equal"]
    assert item["payload_tuple_equal"]
    assert item["exact_inverse_restores_seed"]
    assert item["logical_cells"] == 34
assert len(oracle["direct_assignment_checks"]) == 12
for item in oracle["direct_assignment_checks"]:
    assert item["chart_matches_direct_assignments"]
    assert item["chart_matches_residue_class_dynamic_program"]
for key, value in oracle["independent_mutation_checks"].items():
    assert (not value) if key == "snapshot_or_baseline_reload_used" else value, key

payloads = [item["maximum_resident_payload_bits"] for item in production["exact_transactions"]]
assert payloads == [1096, 1120, 1616, 3313, 7607, 16305, 33763, 68588]
assert production["resource_law"]["resident_exact_field_cells"] == 34
assert production["resource_law"]["rank_two_canonical_group_algebra_chart_field_cells"] == 578
assert production["matched_baseline"]["same_asymptotic_arithmetic_and_payload_law"]
assert not production["matched_baseline"]["distinct_phase_resource"]
assert not production["matched_baseline"]["computational_advantage"]
for key in (
    "rank_two_fixed_34_cell_closure",
    "general_rank_r_or_arbitrary_cubic_hypergraph_closure",
    "catvm_custody",
    "distinct_phase_resource",
    "computational_advantage",
    "small_wall_crossing",
    "physical_waveform_execution",
    "physical_bits_replaced_with_pi",
    "unbounded_catalytic_computation",
):
    assert production["claim_ceiling"][key] is False
PY

echo QUALIFIED_F17_RANK1_CUBIC_WALSH_DERIVATIVE_GROUP_ALGEBRA_CLOSURE_STRICT_SCOPE
