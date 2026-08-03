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
source_path="$frontier_dir/f17_growing_shared_latent_cubic_port_separator_rank_no_go.py"
oracle_path="$frontier_dir/f17_growing_shared_latent_cubic_port_separator_rank_no_go_oracle.py"
production_seal="$frontier_dir/F17_GROWING_SHARED_LATENT_CUBIC_PORT_SEPARATOR_RANK_NO_GO_RESULTS.json"
oracle_seal="$frontier_dir/F17_GROWING_SHARED_LATENT_CUBIC_PORT_SEPARATOR_RANK_NO_GO_ORACLE_RESULTS.json"
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
for required in ("begin_forward", "project_certificate", "inverse"):
    try:
        cursor = ordered.index(required, cursor) + 1
    except ValueError as exc:
        raise SystemExit(f"accepted transaction lacks ordered {required}") from exc

certificate = function(source_tree, "rank_certificate_from_resident")
for node in ast.walk(certificate):
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else ""
        )
        if name in {"matrix_rank", "branch_minor", "walsh_matrix", "matmul"}:
            raise SystemExit("accepted symbolic certificate calls dense oracle machinery")
        if name == "range" and any(
            isinstance(arg, ast.Name) and arg.id in {"port_coordinates", "width"}
            for arg in node.args
        ):
            raise SystemExit("accepted symbolic certificate enumerates the port width")

for node in oracle_tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        names = [alias.name for alias in node.names]
        if any(
            name.startswith("f17_growing_shared_latent_cubic_port_separator_rank_no_go")
            or name.startswith("f17_overlapping_cubic_bond3_phase_factor_closure")
            for name in names
        ):
            raise SystemExit("independent oracle imports production or factor carrier")

required_oracle = {
    "branch_minor",
    "walsh_matrix",
    "matrix_rank",
    "matmul",
    "direct_case",
    "local_factorized_boundary",
    "inverse_pair",
    "transaction_parity",
}
present = {
    node.name for node in ast.walk(oracle_tree) if isinstance(node, ast.FunctionDef)
}
missing = required_oracle - present
if missing:
    raise SystemExit(f"independent oracle lacks {sorted(missing)}")
PY

"$python_bin" -m py_compile "$source_path" "$oracle_path"
nice -n 10 "$python_bin" -X dev "$source_path" --output "$production_replay"
cmp "$production_seal" "$production_replay"
nice -n 10 "$python_bin" -X dev "$oracle_path" \
    --production "$production_replay" --output "$oracle_replay"
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
assert production["source_scope"]["exact_arities"] == [1, 2, 4, 8, 16, 32, 64]
assert production["source_scope"]["dual_field_structural_arities"] == [1, 2, 4, 8, 16]
assert production["source_scope"]["formula_arities"] == [1, 64]

transactions = production["exact_transactions"] + production["dual_field_structural_transactions"]
assert len(transactions) == 17
for item in transactions:
    k = item["latent_arity"]
    certificate = item["rank_certificate"]
    assert certificate["left_kronecker_minor_rank"] == 1 << k
    assert certificate["right_kronecker_minor_rank"] == 1 << k
    assert certificate["full_two_sided_separator_rank"] == 1 << k
    assert certificate["typed_configuration_bisimulation_classes"] == 1 << k
    assert certificate["individual_typed_continuation_separates_distinct_configurations"]
    assert not certificate["typed_port_overmerge_exact_relation_preserving"]
    assert certificate["separator_transport_invertible"]
    assert not certificate["dense_port_vector_materialized"]
    assert not certificate["dense_minor_materialized"]
    assert not certificate["local_assignment_family_enumerated"]
    assert item["resident_phase_factor_field_cells"] == 4 * k
    assert item["resident_nontrivial_theta_field_cells"] == 2 * k
    assert item["accepted_path_port_field_cells"] == 0
    assert item["accepted_path_assignment_or_dense_minor_cells"] == 0
    assert not item["intermediate_factor_or_port_payload_exposed"]
    assert item["one_way_factor_commitment_emitted"]
    assert item["projection_calls"] == 1
    assert item["response_released_after_restoration"]
    assert item["restored_exact_zero"]
    assert item["same_backing"]
    assert item["inverse_history_cells"] == 0
    assert not item["snapshot_reload_used"]
    assert item["resident_carrier_restoration_class"] == "EXACT_ALGEBRAIC_RESTORATION"
    assert item["compiler_commitment_and_verification_buffer_restoration_class"] == "NO_RESTORATION_CLAIM"

formulas = production["formula_certificates_k1_through_k64"]
assert len(formulas) == 64
for k, item in enumerate(formulas, 1):
    assert item["latent_arity"] == k
    assert item["full_two_sided_separator_rank"] == 1 << k
    assert item["uniform_exact_linear_port_quotient_minimum_field_coordinates"] == 1 << k
    assert item["formula_certificate_work_scalars"] == 6 * k + 12
    assert not item["dense_port_vector_materialized"]
    assert not item["dense_minor_materialized"]

for item in production["compiled_classical_baselines"]:
    k = item["latent_arity"]
    assert item["full_public_signature_field_cells"] == 2 * k
    assert item["uniform_arbitrary_message_port_field_cells"] == 1 << k
    assert item["same_rank_lower_bound"]
    assert item["strictly_local_sealed_contraction"] == "O_K_TWO_BY_TWO_KRONECKER_FACTORS"
    assert item["general_descriptor_contraction"] == "POLY_DESCRIPTOR_SIZE_TIMES_TWO_TO_THE_TREEWIDTH"
    assert item["accepted_dense_port_or_minor_cells"] == 0
    assert not item["phase_carrier_or_snapshot_used"]
    assert not item["comparison_establishes_advantage"]

reuse = production["reuse"]
assert reuse["primary_arity"] == 8
assert reuse["reuse_arity"] == 16
assert reuse["same_original_backing"]
assert reuse["fresh_restored_reuse_signature_equal"]
assert reuse["package_local_restoration_count"] == 2
assert reuse["restored_exact_zero"]
assert not reuse["baseline_reload"]
assert reuse["inverse_history_cells"] == 0

false_controls = {
    "accepted_formula_materializes_port",
    "accepted_formula_enumerates_assignments",
    "snapshot_command_available",
}
for key, value in production["controls"].items():
    assert (not value) if key in false_controls else value, key

for item in oracle["transaction_parity"]:
    assert item["all_core_fields_match"], item["mismatches"]
    assert item["factor_pairs_restore_exact_seed"]
assert len(oracle["direct_rank_checks"]) == 18
for item in oracle["direct_rank_checks"]:
    width = 1 << item["latent_arity"]
    assert item["left_minor_rank"] == width
    assert item["right_minor_rank"] == width
    assert item["walsh_transport_rank"] == width
    assert item["two_sided_boundary_minor_rank"] == width
    assert item["identity_left_phase_drops_rank"]
    assert item["identity_right_phase_drops_rank"]
    assert item["singular_transport_drops_rank"]
    assert item["all_typed_configurations_have_distinct_right_observation_columns"]
    assert item["direct_boundary_matches_o_k_factorized_contraction"]
    assert item["dense_matrices_are_verification_only"]
    assert item["accepted_path_dense_cells"] == 0

for mutation in oracle["independent_mutation_checks"]:
    assert all(
        value for key, value in mutation.items() if key != "snapshot_command_available"
    )
    assert not mutation["snapshot_command_available"]

assert production["matched_baseline"]["phase_advantage_over_matched_classical"] is False
assert production["matched_baseline"]["nonlinear_program_dependent_or_global_contraction_routes_exhausted"] is False
assert production["resource_law"][
    "uniform_exact_linear_arbitrary_message_port_minimum_field_coordinates"
] == "TWO_TO_THE_K"
for key in (
    "rank_alone_general_storage_or_advantage_lower_bound",
    "nonlinear_or_program_dependent_quotient_no_go",
    "operational_two_to_the_k_port_phase_closure",
    "arbitrary_cubic_hypergraph_closure",
    "catvm_custody",
    "distinct_phase_resource",
    "computational_advantage",
    "small_wall_crossing",
    "physical_execution",
    "physical_bits_replaced_with_pi",
    "unbounded_catalytic_computation",
):
    assert production["claim_ceiling"][key] is False
PY

echo QUALIFIED_F17_GROWING_SHARED_LATENT_CUBIC_PORT_SEPARATOR_RANK_NO_GO_STRICT_SCOPE
