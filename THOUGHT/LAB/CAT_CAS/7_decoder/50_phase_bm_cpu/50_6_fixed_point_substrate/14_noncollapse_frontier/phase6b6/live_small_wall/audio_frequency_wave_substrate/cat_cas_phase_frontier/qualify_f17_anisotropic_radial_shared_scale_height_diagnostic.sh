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
source_path="$frontier_dir/f17_anisotropic_radial_shared_scale_height_diagnostic.py"
oracle_path="$frontier_dir/f17_anisotropic_radial_shared_scale_height_diagnostic_oracle.py"
production_seal="$frontier_dir/F17_ANISOTROPIC_RADIAL_SHARED_SCALE_HEIGHT_DIAGNOSTIC_RESULTS.json"
oracle_seal="$frontier_dir/F17_ANISOTROPIC_RADIAL_SHARED_SCALE_HEIGHT_DIAGNOSTIC_ORACLE_RESULTS.json"
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
    "project",
    "inverse",
):
    try:
        cursor = ordered.index(required, cursor) + 1
    except ValueError as exc:
        raise SystemExit(f"accepted transaction lacks ordered {required}") from exc

normalizer = ast.unparse(function(source_tree, "compile_shared_scale"))
for required in (
    "math.lcm",
    "math.gcd",
    "pi_valuation",
    "divide_pi_exact",
    "ring_multiply",
    "reconstructed != cells",
):
    if required not in normalizer:
        raise SystemExit(f"accepted normalizer lacks {required}")
for forbidden in ("float(", "complex(", "numpy", "snapshot"):
    if forbidden in normalizer:
        raise SystemExit(f"accepted normalizer contains forbidden {forbidden}")

oracle_imports = []
for node in oracle_tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        oracle_imports.extend(alias.name for alias in node.names)
for forbidden in (
    "f17_matrix_free_anisotropic_radial_phase_fourier",
    "f17_anisotropic_radial_shared_scale_height_diagnostic",
    "f17_cubic_chain_period17_height_lower_bound",
):
    if any(name.startswith(forbidden) for name in oracle_imports):
        raise SystemExit(f"independent oracle imports {forbidden}")

required_oracle = {
    "compile_program",
    "dense_fourier",
    "forward",
    "inverse",
    "boundary",
    "divide_pi",
    "multiply_pi",
    "pi_valuation",
    "metrics",
    "execute_case",
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
assert oracle["all_exact_inverse_seed_checks_pass"]
assert oracle["payload_mutation_detected"]
assert oracle["case_count"] == 21
assert oracle["metric_and_boundary_comparisons"] == 252

scope = production["source_scope"]
assert scope["depths"] == [1, 2, 4, 8, 16, 32, 64]
assert scope["families"] == ["PRIMARY", "REUSE", "ALTERNATE"]
assert len(production["cases"]) == 21
assert production["all_depth2_and_above_have_zero_common_pi_content"]
assert production["all_shared_denominators_are_power17"]
assert production["all_exact_reconstructions_equal"]

for case in production["cases"]:
    assert case["shared_denominator_payload_bits"] * 2 < case["raw_repeated_denominator_payload_bits"]
    assert case["shared_scale_resident_integer_coefficients"] == 272
    assert case["exact_reconstruction_equal"]
    assert case["same_backing"]
    assert case["restored_exact_zero"]
    assert case["package_local_restoration_count"] == 1
    assert not case["snapshot_reload_used"]
    assert case["inverse_history_cells"] == 0
    assert case["resident_carrier_restoration_class"] == "EXACT_ALGEBRAIC_RESTORATION"
    assert case["scale_ledger_restoration_class"] == "NO_RESTORATION_CLAIM"
    if case["depth"] >= 2:
        assert case["common_pi_exponent"] == 0

assert all(
    all(checks.values())
    for checks in production["endpoint_growth"].values()
)
reuse = production["reuse"]
assert reuse["same_original_backing"]
assert reuse["fresh_restored_boundary_equal"]
assert reuse["fresh_restored_scale_signature_equal"]
assert reuse["restored_exact_zero"]
assert reuse["package_local_restoration_count"] == 2
assert not reuse["snapshot_reload_used"]
assert reuse["inverse_history_cells"] == 0

resources = production["resource_law"]
assert resources["resident_radial_field_cells"] == 17
assert resources["represented_cyclotomic_integer_coefficients"] == 272
assert resources["shared_denominator_ledger_integers"] == 1
assert resources["common_pi_ledger_integers"] == 1
assert resources["matrix_free_generator_exact_field_cells"] == 1
assert resources["retained_public_kernel_exact_field_cells"] == 0
assert resources["accepted_assignment_truth_table_coordinate_or_kernel_cells"] == 0

matched = production["matched_classical_baseline"]
assert matched["method"] == "IDENTICAL_SHARED_SCALE_NORMALIZED_MATRIX_FREE_17_COORDINATE_RECURRENCE"
assert matched["all_case_boundaries_equal"]
assert matched["same_normalization_and_payload_law"]
assert not matched["universal_optimality_or_lower_bound_claimed"]
assert not matched["comparison_establishes_distinct_phase_resource"]
assert not matched["comparison_establishes_computational_advantage"]

assert production["restoration"]["class"] == "EXACT_ALGEBRAIC_RESTORATION"
assert production["restoration"]["scale_metric_buffers"] == "NO_RESTORATION_CLAIM"
assert not production["restoration"]["snapshot_reload_used"]
for claim in production["claim_boundary"]["not_established"]:
    assert claim
PY

echo QUALIFIED_F17_ANISOTROPIC_RADIAL_SHARED_SCALE_HEIGHT_DIAGNOSTIC_STRICT_SCOPE
