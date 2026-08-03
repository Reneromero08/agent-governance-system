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
source_path="$frontier_dir/f17_anisotropic_radial_native_angle_interference.py"
oracle_path="$frontier_dir/f17_anisotropic_radial_native_angle_interference_oracle.py"
production_seal="$frontier_dir/F17_ANISOTROPIC_RADIAL_NATIVE_ANGLE_INTERFERENCE_RESULTS.json"
oracle_seal="$frontier_dir/F17_ANISOTROPIC_RADIAL_NATIVE_ANGLE_INTERFERENCE_ORACLE_RESULTS.json"
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

fourier_source = ast.unparse(function(source_tree, "apply_angle_fourier"))
for required in (
    "carrier.angles",
    "real_radial_kernel",
    "phasor_pair_value",
    "accumulator_real",
    "accumulator_imag",
    "chart_pair",
    "output_angles",
):
    if required not in fourier_source:
        raise SystemExit(f"angle Fourier lacks {required}")
for forbidden in (
    "decode(",
    "complex(",
    "np.complex",
    "classical_fourier",
    "classical_boundary",
    "np.dot",
    "np.matmul",
):
    if forbidden in fourier_source:
        raise SystemExit(f"angle Fourier contains forbidden {forbidden}")

kernel_source = ast.unparse(function(source_tree, "real_radial_kernel"))
for required in ("range(stop)", "math.cos", "inverse_four_parameters_half"):
    if required not in kernel_source:
        raise SystemExit(f"streamed real kernel lacks {required}")

projection_source = ast.unparse(function(source_tree, "project"))
for required in ("boundary_real", "boundary_imag", "math.cos", "math.sin"):
    if required not in projection_source:
        raise SystemExit(f"streamed projection lacks {required}")
for forbidden in ("decode(", "np.complex", "np.empty((P", "np.zeros((P"):
    if forbidden in projection_source:
        raise SystemExit(f"streamed projection contains {forbidden}")

inverse_source = ast.unparse(function(source_tree, "inverse"))
for forbidden in (
    "carrier.angles[:] = seed",
    "carrier.angles = seed",
    "np.copyto(carrier.angles, seed",
):
    if forbidden in inverse_source:
        raise SystemExit(f"inverse contains post-hoc seed reset {forbidden}")

commitment_source = ast.unparse(function(source_tree, "state_commitment"))
if "memoryview(carrier.angles).cast('B')" not in commitment_source:
    raise SystemExit("commitment does not use the declared zero-copy byte view")
if ".tobytes(" in commitment_source:
    raise SystemExit("commitment materializes an uncounted state byte buffer")

for forbidden in (
    "np.empty((P, P",
    "np.zeros((P, P",
    "np.eye(P",
):
    if forbidden in source_text:
        raise SystemExit(f"production retains dense kernel expression {forbidden}")

oracle_imports = []
for node in oracle_tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        oracle_imports.extend(alias.name for alias in node.names)
for forbidden in (
    "f17_anisotropic_radial_native_angle_interference",
    "f17_anisotropic_radial_unit_phasor_pair_chart",
    "f17_matrix_free_anisotropic_radial_phase_fourier",
):
    if forbidden in oracle_imports:
        raise SystemExit(f"independent oracle imports {forbidden}")
for required in (
    "compile_dense_kernel",
    "gate_parameters",
    "decode",
    "encode",
    "execute_case",
    "reuse_control",
    "repeated_reuse_control",
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
assert oracle["case_count"] == 21
assert oracle["metric_and_boundary_comparisons"] == 474
assert oracle["production_restoration_and_reuse_fields_verified"]
assert oracle["dense_public_coordinate_visits"] == 2312
assert oracle["public_character_pairing_violations"] == 0
assert oracle["dense_kernel_involution_max_abs"] < 1.0e-15
assert oracle["all_cases_within_predeclared_tolerances"]
assert oracle["kernel_mutation_detected"]

scope = production["source_scope"]
assert scope["depths"] == [1, 4, 16, 64, 256, 1024, 4096]
assert scope["families"] == ["PRIMARY", "REUSE", "ALTERNATE"]
assert scope["case_count"] == 21
assert scope["unrelated_reuse_depth"] == 1537
assert scope["repeated_reuse_cycles"] == 100

tolerances = production["predeclared_tolerances"]
for case in production["cases"]:
    assert case["resident_phase_angle_cells"] == 34
    assert case["resident_phase_angle_bytes"] == 272
    assert case["boundary_error_against_matched_classical"] <= tolerances[
        "boundary_max_abs"
    ]
    assert case["boundary_error_against_streamed_classical"] <= tolerances[
        "boundary_max_abs"
    ]
    assert case["matched_classical_frontier_boundary_error"] <= tolerances[
        "boundary_max_abs"
    ]
    assert case["matched_classical_frontier_state_error"] <= tolerances[
        "boundary_max_abs"
    ]
    assert case["restoration_error"] <= tolerances[
        "restoration_phasor_max_abs"
    ]
    assert case["same_backing"]
    assert case["restoration_generation_before"] == 0
    assert case["restoration_generation_after"] == 1
    assert not case["snapshot_reload_used"]
    assert case["inverse_history_cells"] == 0
    assert case["resident_restoration_class"] == (
        "NUMERICAL_PHYSICAL_STATE_RESTORATION"
    )

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
assert repeated["maximum_restoration_error"] <= tolerances[
    "restoration_phasor_max_abs"
]
assert repeated["restoration_generation"] == 100
assert not repeated["snapshot_reload_used"]
assert repeated["inverse_history_cells"] == 0
assert all(production["controls"].values())

resources = production["resource_law"]
assert resources["resident_phase_angle_float64_cells"] == 34
assert resources["resident_phase_angle_bytes"] == 272
assert resources["retained_inverse_half_parameter_int64_cells"] == 8
assert resources["retained_complex_state_cells"] == 0
assert resources["retained_dense_kernel_cells"] == 0
assert resources["compiled_gate_sequence_cells"] == 0
assert resources["inverse_history_cells"] == 0
assert resources["cartesian_accumulator_float64_cells_per_output"] == 2
assert resources["output_angle_buffer_float64_cells"] == 34
assert resources["maximum_named_update_bytes"] == 368
assert resources["maximum_named_projection_bytes"] == 80
assert resources["maximum_named_accepted_live_bytes_including_program_json"] == 1018
assert resources["commitment_input_copy_bytes"] == 0
assert resources["commitment_public_hexdigest_bytes"] == 64
assert resources["commitment_logical_sha256_state_and_block_bytes"] == 96
assert resources["maximum_named_commitment_live_bytes_including_program_json"] == 810
assert resources["kernel_cosine_evaluations_per_fourier"] == 2312
assert resources["input_phasor_cosine_evaluations_per_fourier"] == 578
assert resources["input_phasor_sine_evaluations_per_fourier"] == 578

matched = production["matched_classical_baseline"]
assert matched["work_minimizing_method"] == (
    "IDENTICAL_MATRIX_FREE_NORMALIZED17_COMPLEX_RADIAL_RECURRENCE"
)
assert matched["equal_memory_method"] == (
    "STREAMED_REAL_KERNEL_NORMALIZED17_COMPLEX_RADIAL_RECURRENCE"
)
assert matched["resident_bytes_equal"]
assert matched["complex_character_products_per_fourier"] == 544
assert matched["matrix_free_retained_geometry_bytes"] == 536
assert matched["matrix_free_maximum_live_bytes_including_program_json"] == 1770
assert matched["streamed_complex_maximum_live_bytes_including_program_json"] == 986
assert matched["native_angle_maximum_live_bytes_including_program_json"] == 1018
assert matched[
    "all_case_frontier_boundaries_and_states_within_predeclared_tolerance"
]
assert matched["native_angle_path_has_strictly_more_trigonometric_work"]
assert matched[
    "equal_memory_streamed_complex_uses_no_input_phasor_or_chart_trigonometry"
]
assert not matched["comparison_establishes_distinct_phase_resource"]
assert not matched["comparison_establishes_computational_advantage"]

restoration = production["restoration"]
assert restoration["class"] == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
assert not restoration["post_inverse_state_reset_or_canonical_reload_used"]
assert restoration["transient_buffers"] == "NO_RESTORATION_CLAIM"
assert restoration["same_backing"]
assert restoration["repeated_reuse_tested"]
assert not restoration["snapshot_reload_used"]
PY

echo QUALIFIED_F17_ANISOTROPIC_RADIAL_NATIVE_ANGLE_INTERFERENCE_STRICT_SCOPE
