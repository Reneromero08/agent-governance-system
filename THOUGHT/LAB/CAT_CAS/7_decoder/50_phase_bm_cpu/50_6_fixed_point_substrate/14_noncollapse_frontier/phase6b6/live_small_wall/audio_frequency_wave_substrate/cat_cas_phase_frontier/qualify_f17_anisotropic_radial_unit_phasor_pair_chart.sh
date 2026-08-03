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
source_path="$frontier_dir/f17_anisotropic_radial_unit_phasor_pair_chart.py"
oracle_path="$frontier_dir/f17_anisotropic_radial_unit_phasor_pair_chart_oracle.py"
production_seal="$frontier_dir/F17_ANISOTROPIC_RADIAL_UNIT_PHASOR_PAIR_CHART_RESULTS.json"
oracle_seal="$frontier_dir/F17_ANISOTROPIC_RADIAL_UNIT_PHASOR_PAIR_CHART_ORACLE_RESULTS.json"
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
    "project",
    "classical_boundary",
    "inverse",
):
    try:
        cursor = ordered.index(required, cursor) + 1
    except ValueError as exc:
        raise SystemExit(f"accepted transaction lacks ordered {required}") from exc

phase_source = ast.unparse(function(source_tree, "apply_phase"))
for required in ("phase_exponents", "angles", "wrap_phase"):
    if required not in phase_source:
        raise SystemExit(f"phase update lacks {required}")
fourier_source = ast.unparse(function(source_tree, "apply_fourier"))
for required in ("decode", "matrix_free_fourier", "encode", "carrier.angles"):
    if required not in fourier_source:
        raise SystemExit(f"Fourier chart update lacks {required}")
inverse_source = ast.unparse(function(source_tree, "inverse"))
for forbidden in (
    "carrier.angles[:] = seed",
    "carrier.angles = seed",
    "np.copyto(carrier.angles, seed",
):
    if forbidden in inverse_source:
        raise SystemExit(f"inverse contains post-hoc seed reset {forbidden}")
for forbidden in ("np.zeros((P, P", "np.empty((P, P", "np.eye(P"):
    if forbidden in source_text:
        raise SystemExit(f"production retains dense matrix expression {forbidden}")

oracle_imports = []
for node in oracle_tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        oracle_imports.extend(alias.name for alias in node.names)
for forbidden in (
    "f17_anisotropic_radial_unit_phasor_pair_chart",
    "f17_matrix_free_anisotropic_radial_phase_fourier",
    "f17_anisotropic_quartic_radial_phase_quotient_closure",
):
    if forbidden in oracle_imports:
        raise SystemExit(f"independent oracle imports {forbidden}")
for required in (
    "compile_dense_fourier",
    "gate_parameters",
    "encode",
    "decode",
    "execute_case",
    "sequential_reuse",
    "repeated_reuse",
):
    function(oracle_tree, required)
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
assert oracle["restoration_class"] == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
assert oracle["all_cases_within_predeclared_tolerances"]
assert oracle["boundary_mutation_detected"]
assert oracle["case_count"] == 21
assert oracle["metric_and_boundary_comparisons"] == 147
assert oracle["dense_public_coordinate_visits"] == 4913
assert oracle["dense_fourier_involution_max_abs"] < 1.0e-16

scope = production["source_scope"]
assert scope["depths"] == [1, 4, 16, 64, 256, 1024, 4096]
assert scope["families"] == ["PRIMARY", "REUSE", "ALTERNATE"]
assert scope["case_count"] == 21
assert scope["repeated_reuse_cycles"] == 100
assert scope["unrelated_reuse_depth"] == 1537
assert len(production["cases"]) == 21

tolerances = production["predeclared_tolerances"]
for case in production["cases"]:
    assert case["phase_angle_resident_cells"] == 34
    assert case["phase_angle_resident_bytes"] == 272
    assert case["boundary_error_against_matched_classical"] <= tolerances["boundary_max_abs"]
    assert case["restoration_error"] <= tolerances["restoration_phasor_max_abs"]
    assert case["same_backing"]
    assert case["restoration_generation_before"] == 0
    assert case["restoration_generation_after"] == 1
    assert not case["snapshot_reload_used"]
    assert case["inverse_history_cells"] == 0
    assert case["resident_restoration_class"] == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
    assert case["transient_buffer_restoration_class"] == "NO_RESTORATION_CLAIM"

reuse = production["reuse"]
assert reuse["same_original_backing"]
assert reuse["fresh_restored_boundary_error"] <= tolerances["boundary_max_abs"]
assert reuse["restoration_error"] <= tolerances["restoration_phasor_max_abs"]
assert reuse["restoration_generation"] == 2
assert not reuse["snapshot_reload_used"]
assert reuse["inverse_history_cells"] == 0

repeated = production["repeated_reuse"]
assert repeated["cycles"] == 100
assert repeated["same_backing"]
assert repeated["maximum_restoration_error"] <= tolerances["restoration_phasor_max_abs"]
assert repeated["restoration_generation"] == 100
assert not repeated["snapshot_reload_used"]
assert repeated["inverse_history_cells"] == 0
assert all(production["controls"].values())

resources = production["resource_law"]
assert resources["resident_phase_angle_float64_cells"] == 34
assert resources["resident_phase_angle_bytes"] == 272
assert resources["matched_classical_resident_complex128_cells"] == 17
assert resources["matched_classical_resident_bytes"] == 272
assert resources["retained_dense_fourier_kernel_cells"] == 0
assert resources["compiled_gate_sequence_cells"] == 0
assert resources["inverse_history_cells"] == 0
assert resources["maximum_named_update_bytes"] == 1056
assert resources["maximum_named_projection_bytes"] == 816
assert resources["maximum_public_program_json_bytes"] == 184
assert resources["maximum_named_accepted_live_bytes_including_program_json"] == 2184
assert resources["verification_baseline_is_not_accepted_phase_state"]

matched = production["matched_classical_baseline"]
assert matched["method"] == "IDENTICAL_MATRIX_FREE_NORMALIZED17_COMPLEX_RADIAL_RECURRENCE_WITHOUT_PHASE_PAIR_CHART"
assert matched["resident_bytes_equal"]
assert matched["phase_chart_requires_additional_transcendental_work"]
assert matched["all_case_boundaries_within_predeclared_tolerance"]
assert not matched["comparison_establishes_distinct_phase_resource"]
assert not matched["comparison_establishes_computational_advantage"]

restoration = production["restoration"]
assert restoration["class"] == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
assert not restoration["post_inverse_state_reset_or_canonical_reload_used"]
assert restoration["transient_buffers"] == "NO_RESTORATION_CLAIM"
assert restoration["same_backing"]
assert restoration["repeated_reuse_tested"]
assert not restoration["snapshot_reload_used"]
for claim in production["claim_boundary"]["not_established"]:
    assert claim
PY

echo QUALIFIED_F17_ANISOTROPIC_RADIAL_UNIT_PHASOR_PAIR_CHART_STRICT_SCOPE
