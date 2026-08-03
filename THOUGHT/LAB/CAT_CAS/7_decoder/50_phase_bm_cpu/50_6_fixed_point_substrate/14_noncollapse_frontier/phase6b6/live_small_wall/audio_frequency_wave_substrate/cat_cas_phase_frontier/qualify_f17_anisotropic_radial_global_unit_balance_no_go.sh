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
source_path="$frontier_dir/f17_anisotropic_radial_global_unit_balance_no_go.py"
oracle_path="$frontier_dir/f17_anisotropic_radial_global_unit_balance_no_go_oracle.py"
production_seal="$frontier_dir/F17_ANISOTROPIC_RADIAL_GLOBAL_UNIT_BALANCE_NO_GO_RESULTS.json"
oracle_seal="$frontier_dir/F17_ANISOTROPIC_RADIAL_GLOBAL_UNIT_BALANCE_NO_GO_ORACLE_RESULTS.json"
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
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SystemExit(f"missing function {name}")


accepted = function(source_tree, "execute_case")
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
for required in (
    "begin_forward",
    "forward",
    "state_commitment",
    "compile_shared_scale",
    "balance_signature",
    "reconstruct_balanced",
    "project",
    "inverse",
):
    try:
        cursor = ordered.index(required, cursor) + 1
    except ValueError as exc:
        raise SystemExit(f"accepted transaction lacks ordered {required}") from exc

balancer = ast.unparse(function(source_tree, "balance_signature"))
for required in (
    "descent.balance_vector",
    "lattice.balance_vector",
    "exact_residual != proposal_residual",
    "exact_ledger != proposal_ledger",
):
    if required not in balancer:
        raise SystemExit(f"accepted balancer lacks {required}")

oracle_imports = []
for node in oracle_tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        oracle_imports.extend(alias.name for alias in node.names)
for forbidden in (
    "f17_anisotropic_radial_global_unit_balance_no_go",
    "f17_matrix_free_anisotropic_radial_phase_fourier",
    "f17_anisotropic_radial_shared_scale_height_diagnostic",
):
    if forbidden in oracle_imports:
        raise SystemExit(f"independent oracle imports {forbidden}")

required_oracle = {
    "shared_scale",
    "ledger_scale",
    "vector_energy",
    "verify_selected_ledger",
    "execute_case",
    "run",
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
    --package "$production_replay" \
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
assert oracle["restoration_class"] == "EXACT_ALGEBRAIC_RESTORATION"
assert oracle["all_cases_equal"]
assert oracle["all_exact_unit_reconstructions_equal"]
assert oracle["all_exact_inverse_seed_checks_pass"]
assert oracle["unit_ledger_mutation_detected"]
assert oracle["case_count"] == 21
assert oracle["metric_and_boundary_comparisons"] == 231
assert oracle["exact_adjacent_direction_checks"] == 2058

scope = production["source_scope"]
assert scope["depths"] == [1, 2, 4, 8, 16, 32, 64]
assert scope["families"] == ["PRIMARY", "REUSE", "ALTERNATE"]
assert scope["case_count"] == 21
assert len(production["cases"]) == 21
assert production["all_three_depth1_cases_repaired"]
assert production["all_18_depth2_plus_cases_select_identity_unit"]
assert production["all_21_balancer_outputs_equal"]
assert production["all_21_exact_reconstructions_equal"]

for case in production["cases"]:
    assert case["exact_balanced_reconstruction_equal"]
    assert case["balancer_residuals_and_ledgers_equal"]
    assert case["exact_coordinatewise_certified"]
    assert case["exact_sweep_cap_hits"] == 0
    assert case["exact_bracket_cap_hits"] == 0
    assert case["fixed_precision_nonpositive_embedding_rejections"] == 0
    assert case["same_backing"]
    assert case["restored_exact_zero"]
    assert case["package_local_restoration_count"] == 1
    assert not case["snapshot_reload_used"]
    assert case["inverse_history_cells"] == 0
    assert case["resident_carrier_restoration_class"] == "EXACT_ALGEBRAIC_RESTORATION"
    assert case["unit_balance_buffer_restoration_class"] == "NO_RESTORATION_CLAIM"
    if case["depth"] == 1:
        assert case["unit_ledger"] == [1, 1, 1, 1, 1, 1, 1]
        assert case["balanced_all_ledgers_payload_bits"] < case["m141_common_pi_normalized_payload_bits"]
    else:
        assert case["unit_ledger"] == [0, 0, 0, 0, 0, 0, 0]

reuse = production["reuse"]
assert reuse["same_original_backing"]
assert reuse["fresh_restored_boundary_equal"]
assert reuse["fresh_restored_balance_signature_equal"]
assert reuse["restored_exact_zero"]
assert reuse["package_local_restoration_count"] == 2
assert not reuse["snapshot_reload_used"]
assert reuse["inverse_history_cells"] == 0

resources = production["resource_law"]
assert resources["resident_radial_exact_field_cells"] == 17
assert resources["represented_cyclotomic_integer_coefficients"] == 272
assert resources["unit_ledger_integers"] == 7
assert resources["compiled_exact_direction_count"] == 49
assert resources["compiled_exact_unit_table_payload_bits"] == 8177
assert resources["fixed_precision_proposal_bits"] == 65536
assert resources["fixed_precision_path_is_control_not_accepted_balancer"]

matched = production["matched_classical_baseline"]
assert matched["method"] == "IDENTICAL_SHARED_SCALE_AND_EXACT49_DIRECTION_GLOBAL_UNIT_BALANCING_RECURRENCE"
assert matched["same_representation_search_ledgers_and_payload_law"]
assert not matched["universal_optimality_or_lower_bound_claimed"]
assert not matched["comparison_establishes_distinct_phase_resource"]
assert not matched["comparison_establishes_computational_advantage"]

assert production["restoration"]["resident_carrier"] == "EXACT_ALGEBRAIC_RESTORATION"
assert production["restoration"]["unit_search_and_scale_buffers"] == "NO_RESTORATION_CLAIM"
assert not production["restoration"]["snapshot_reload_used"]
for claim in production["claim_boundary"]["not_established"]:
    assert claim
PY

echo QUALIFIED_F17_ANISOTROPIC_RADIAL_GLOBAL_UNIT_BALANCE_NO_GO_STRICT_SCOPE
