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
source_path="$frontier_dir/f17_two_signature_cubic_walsh_separation_rank_diagnostic.py"
oracle_path="$frontier_dir/f17_two_signature_cubic_walsh_separation_rank_diagnostic_oracle.py"
production_seal="$frontier_dir/F17_TWO_SIGNATURE_CUBIC_WALSH_SEPARATION_RANK_DIAGNOSTIC_RESULTS.json"
oracle_seal="$frontier_dir/F17_TWO_SIGNATURE_CUBIC_WALSH_SEPARATION_RANK_DIAGNOSTIC_ORACLE_RESULTS.json"
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
for required in ("begin_forward", "forward", "state_commitment", "project_diagnostic", "inverse"):
    try:
        cursor = ordered.index(required, cursor) + 1
    except ValueError as exc:
        raise SystemExit(f"accepted transaction lacks ordered {required}") from exc

for name in ("forward", "project_diagnostic", "inverse"):
    node = function(source_tree, name)
    text = ast.unparse(node)
    for forbidden in ("product(", "assignments", "truth_table", "permanent", "snapshot"):
        if forbidden in text:
            raise SystemExit(f"accepted {name} contains forbidden {forbidden}")

for node in oracle_tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        imports = [alias.name for alias in node.names]
        if any(name.startswith("f17_two_signature_cubic_walsh_separation_rank_diagnostic") for name in imports):
            raise SystemExit("independent oracle imports production")

required_oracle = {
    "descriptor",
    "forward_rows",
    "inverse_rows",
    "matrix_rank",
    "project",
    "residue_boundary",
    "direct_assignment_boundary",
    "state_payload",
    "rank_mutations",
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
assert production["source_scope"]["derivative_signature_rank"] == 2
assert production["source_scope"]["unresolved_typed_port_count"] == 1
assert production["source_scope"]["exact_branch_pair_phase_step_cases"] == [
    [1, 2], [2, 4], [4, 6], [8, 8], [16, 10], [32, 12], [64, 16], [128, 32]
]

transactions = production["exact_transactions"] + production["structural_transactions"]
assert len(production["exact_transactions"]) == 8
assert len(production["structural_transactions"]) == 54
assert len(transactions) == 62
for item in transactions:
    assert item["derivative_signature_rank"] == 2
    assert item["resident_group_algebra_field_cells"] == 578
    assert item["maximum_update_scratch_field_cells"] == 578
    assert item["maximum_live_resident_plus_update_scratch_field_cells"] == 1156
    assert item["maximum_live_resident_plus_update_scratch_payload_bits"] >= item["maximum_resident_payload_bits"]
    assert item["maximum_rank_verification_dense_field_cells"] == 289
    assert item["maximum_live_resident_plus_rank_dense_buffer_field_cells"] == 867
    assert item["projection_persistently_named_field_cells_excluding_rank_work_and_expression_temporaries"] == 20
    assert item["accepted_path_branch_assignment_or_truth_table_cells"] == 0
    assert not item["intermediate_coefficient_surfaces_exposed"]
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
assert not reuse["snapshot_reload_used"]
assert reuse["inverse_history_cells"] == 0

false_controls = {"snapshot_command_available", "intermediate_projection_available"}
for key, value in production["controls"].items():
    assert (not value) if key in false_controls else value, key

assert len(oracle["transaction_parity"]) == 62
for item in oracle["transaction_parity"]:
    assert item["program_fingerprint_equal"]
    assert item["separation_ranks_equal"]
    assert item["boundary_equal"]
    assert item["residue_boundary_equal"]
    assert item["state_commitment_equal"]
    assert item["payload_equal"]
    assert item["declared_resource_shape_consistent"]
    assert item["exact_inverse_restores_seed"]
    assert item["logical_cells"] == 578
assert len(oracle["direct_assignment_checks"]) == 6
for item in oracle["direct_assignment_checks"]:
    assert item["chart_matches_direct_assignments"]
    assert item["chart_matches_289_class_residue_recurrence"]

false_mutations = {"canonical_578_cells_universal_minimum_claimed", "snapshot_or_baseline_reload_used"}
for key, value in oracle["independent_rank_mutations"].items():
    assert (not value) if key in false_mutations else value, key
assert oracle["observed_resource_law"]["payload_tuples_reproduced_independently"]
assert not oracle["observed_resource_law"]["update_scratch_payload_tuples_reproduced_independently"]
assert oracle["observed_resource_law"]["phase_update_scratch_shape_source_audited"]
assert not oracle["observed_resource_law"]["strongest_streamed_payload_tuple_claimed"]

payloads = [item["maximum_resident_payload_bits"] for item in production["exact_transactions"]]
assert payloads == [18504, 18528, 18744, 19369, 21392, 25842, 40028, 110637]
assert production["resource_law"]["canonical_resident_exact_field_cells"] == 578
assert not production["resource_law"]["canonical_578_cells_universal_representation_minimum"]
assert not production["resource_law"]["uniform_low_separation_rank_below17_for_declared_families"]
assert len(production["matched_classical_baselines"]) == 62
for item in production["matched_classical_baselines"]:
    retained = item["retain_both_surfaces_full_diagnostic"]
    streamed = item["streamed_final_scalar"]
    rematerialized = item["streamed_rematerialized_full_diagnostic"]
    assert retained["reproduces_final_scalar_ranks_and_canonical_commitment"]
    assert retained["resident_exact_field_cells"] == 578
    assert retained["maximum_update_scratch_field_cells"] == 578
    assert retained["maximum_live_resident_plus_update_scratch_field_cells"] == 1156
    assert streamed["executed"] and streamed["boundary_equal"]
    assert streamed["dynamic_exact_field_cells_upper_bound"] == 8
    assert streamed["public_residue_count_integer_cells"] == 17
    assert not streamed["reproduces_rank_and_commitment_diagnostics"]
    assert not rematerialized["executed"]
    assert rematerialized["reproduces_final_scalar_ranks_and_canonical_commitment"]
    assert rematerialized["dynamic_exact_field_cells_conservative_upper_bound"] == 320
    assert rematerialized["payload_tuple_not_claimed_or_measured"]
    assert item["no_single_classical_point_dominates_memory_and_work"]
assert production["matched_baseline"]["all_executed_streamed_final_boundaries_equal"]
assert production["matched_baseline"]["no_single_classical_point_dominates_memory_and_work"]
assert production["matched_baseline"]["same_public_inputs_instances_and_final_scalar_boundary"]
assert not production["matched_baseline"]["same_payload_law_claimed_for_strongest_streamed_path"]
assert not production["matched_baseline"]["distinct_phase_resource"]
assert not production["matched_baseline"]["computational_advantage"]

for key in (
    "canonical_578_cell_chart_is_universal_minimum",
    "general_rank_r_or_arbitrary_cubic_hypergraph_no_go",
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

echo QUALIFIED_F17_TWO_SIGNATURE_CUBIC_WALSH_SEPARATION_RANK_DIAGNOSTIC_STRICT_SCOPE
