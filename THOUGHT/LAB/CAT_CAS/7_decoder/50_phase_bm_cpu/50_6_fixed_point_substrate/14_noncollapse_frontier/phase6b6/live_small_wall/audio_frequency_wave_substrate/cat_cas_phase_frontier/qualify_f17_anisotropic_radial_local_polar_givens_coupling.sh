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
source_path="$frontier_dir/f17_anisotropic_radial_local_polar_givens_coupling.py"
oracle_path="$frontier_dir/f17_anisotropic_radial_local_polar_givens_coupling_oracle.py"
production_seal="$frontier_dir/F17_ANISOTROPIC_RADIAL_LOCAL_POLAR_GIVENS_COUPLING_RESULTS.json"
oracle_seal="$frontier_dir/F17_ANISOTROPIC_RADIAL_LOCAL_POLAR_GIVENS_COUPLING_ORACLE_RESULTS.json"
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
source_text = source_path.read_text(encoding="utf-8")
oracle_text = oracle_path.read_text(encoding="utf-8")
source_tree = ast.parse(source_text, str(source_path))
oracle_tree = ast.parse(oracle_text, str(oracle_path))


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SystemExit(f"missing function {name}")


accepted = ast.unparse(function(source_tree, "apply_givens_fourier"))
for required in (
    "apply_local_coupler",
    "elimination_pairs",
    "reverse_elimination_pairs",
    "apply_signs",
    "plan.cosine_sine",
):
    if required not in accepted:
        raise SystemExit(f"accepted local Fourier lacks {required}")
for forbidden in (
    "complex(",
    "np.dot",
    "np.matmul",
    "np.empty",
    "m144",
):
    if forbidden in accepted:
        raise SystemExit(f"accepted local Fourier contains {forbidden}")

coupler = ast.unparse(function(source_tree, "apply_local_coupler"))
for required in (
    "phasor_pair_value",
    "next_upper_real",
    "next_lower_real",
    "chart_pair",
    "carrier.angles",
):
    if required not in coupler:
        raise SystemExit(f"local coupler lacks {required}")
for forbidden in ("np.complex", "np.dot", "np.matmul"):
    if forbidden in coupler:
        raise SystemExit(f"local coupler contains {forbidden}")

forward_source = ast.unparse(function(source_tree, "forward"))
if "apply_phase" not in forward_source or "apply_givens_fourier" not in forward_source:
    raise SystemExit("forward path does not use the local phase law")
if "matrix_free_boundary" in forward_source or "m144" in forward_source:
    raise SystemExit("forward path calls a verification baseline")

inverse_source = ast.unparse(function(source_tree, "inverse"))
for forbidden in (
    "carrier.angles = seed",
    "carrier.angles[:] = seed",
    "np.copyto(carrier.angles, seed",
):
    if forbidden in inverse_source:
        raise SystemExit(f"inverse contains post-hoc reset {forbidden}")

restored_error = ast.unparse(function(source_tree, "physical_error_from_seed"))
if "seed_angles" in restored_error or "np.empty" in restored_error:
    raise SystemExit("restoration verification materializes an uncounted seed")

commitment = ast.unparse(function(source_tree, "state_commitment"))
if "memoryview(carrier.angles).cast('B')" not in commitment:
    raise SystemExit("commitment lacks zero-copy carrier view")
if ".tobytes(" in commitment:
    raise SystemExit("commitment creates an uncounted input copy")

plan_fields = {
    field.target.id
    for node in source_tree.body
    if isinstance(node, ast.ClassDef) and node.name == "GivensPlan"
    for field in node.body
    if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)
}
if "cosine_sine" not in plan_fields or "diagonal_signs" not in plan_fields:
    raise SystemExit("public plan does not retain coefficients and signs")
if any("kernel" in field or "matrix" in field for field in plan_fields):
    raise SystemExit("public plan retains a dense kernel or matrix")

oracle_imports = []
for node in oracle_tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        oracle_imports.extend(alias.name for alias in node.names)
for forbidden in (
    "f17_anisotropic_radial_local_polar_givens_coupling",
    "f17_anisotropic_radial_native_angle_interference",
    "f17_anisotropic_radial_unit_phasor_pair_chart",
):
    if forbidden in oracle_imports:
        raise SystemExit(f"independent oracle imports {forbidden}")
for required in (
    "dense_weighted_long",
    "compile_float_plan",
    "local_coupler",
    "local_forward",
    "local_inverse",
    "dense_case",
):
    function(oracle_tree, required)
PY

"$python_bin" -m py_compile "$source_path" "$oracle_path"
nice -n 10 "$python_bin" -X dev "$source_path" --output "$production_replay" >/dev/null
cmp "$production_seal" "$production_replay"
nice -n 10 "$python_bin" -X dev "$oracle_path" \
    --package "$production_replay" \
    --output "$oracle_replay" >/dev/null
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
assert oracle["restoration_class"] == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
assert oracle["case_count"] == 21
assert oracle["metric_field_and_control_comparisons"] == 289
assert oracle["production_restoration_reuse_and_resource_fields_verified"]
assert oracle["all_cases_within_predeclared_tolerances"]
assert oracle["dense_weighted_orthogonality_error"] < 1.0e-15
assert oracle["dense_weighted_symmetry_error"] < 1.0e-15
assert oracle["dense_weighted_involution_error"] < 1.0e-15
assert all(oracle["mutation_controls"].values())

scope = production["execution_scope"]
assert scope["depths"] == [1, 4, 16, 64, 256, 1024, 4096]
assert scope["families"] == ["PRIMARY", "REUSE", "ALTERNATE"]
assert scope["case_count"] == 21
assert not scope["public_topology_compilation_reads_final_answers"]

tolerances = production["predeclared_tolerances"]
for case in production["cases"]:
    assert case["resident_phase_angle_cells"] == 34
    assert case["resident_phase_angle_bytes"] == 272
    assert case["maximum_state_error_against_matched_complex_givens"] <= tolerances["state_max_abs"]
    assert case["boundary_error_against_matched_complex_givens"] <= tolerances["boundary_max_abs"]
    assert case["matched_frontier_boundary_error"] <= tolerances["boundary_max_abs"]
    assert case["matched_frontier_state_error"] <= tolerances["state_max_abs"]
    assert case["restoration_error"] <= tolerances["restoration_phasor_max_abs"]
    assert case["same_backing"]
    assert case["restoration_generation_before"] == 0
    assert case["restoration_generation_after"] == 1
    assert not case["snapshot_reload_used"]
    assert case["inverse_history_cells"] == 0
    assert case["resident_restoration_class"] == "NUMERICAL_PHYSICAL_STATE_RESTORATION"

plan = production["plan"]
assert plan["rotation_count"] == 136
assert plan["cosine_sine_float64_cells"] == 272
assert plan["diagonal_sign_float64_cells"] == 17
assert plan["retained_plan_bytes"] == 2312
assert plan["stored_index_cells"] == 0
assert plan["compilation_maximum_named_bytes"] == 4960

resources = production["resource_law"]
assert resources["resident_phase_angle_bytes"] == 272
assert resources["retained_public_givens_plan_bytes"] == 2312
assert resources["maximum_named_update_bytes"] == 96
assert resources["maximum_named_restoration_verification_bytes"] == 64
assert resources["maximum_named_warm_execution_live_bytes_including_program_json"] == 3055
assert resources["maximum_named_full_lifecycle_live_bytes"] == 4960
assert resources["retained_complex_state_cells"] == 0
assert resources["retained_dense_kernel_cells"] == 0
assert resources["inverse_history_cells"] == 0
assert resources["commitment_input_copy_bytes"] == 0

matched = production["matched_classical_baseline"]
assert matched["method"] == "IDENTICAL_PUBLIC_PLAN17_COMPLEX_IN_PLACE_GIVENS_RECURRENCE"
assert matched["maximum_named_warm_execution_live_bytes_including_program_json"] == 2959
assert matched["maximum_named_full_lifecycle_live_bytes"] == 4960
assert matched["fourier_input_phasor_or_chart_trigonometry"] == 0
assert not matched["comparison_establishes_distinct_phase_resource"]
assert not matched["comparison_establishes_computational_advantage"]

assert all(production["controls"].values())
assert production["reuse"]["restoration_generation"] == 2
assert production["reuse"]["same_original_backing"]
assert not production["reuse"]["snapshot_reload_used"]
assert production["repeated_reuse"]["restoration_generation"] == 100
assert production["repeated_reuse"]["same_backing"]
assert production["repeated_reuse"]["maximum_restoration_error"] <= tolerances["restoration_phasor_max_abs"]
assert production["restoration"]["class"] == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
assert not production["restoration"]["post_inverse_state_reset_or_canonical_reload_used"]
PY

echo QUALIFIED_F17_ANISOTROPIC_RADIAL_LOCAL_POLAR_GIVENS_COUPLING_STRICT_SCOPE
