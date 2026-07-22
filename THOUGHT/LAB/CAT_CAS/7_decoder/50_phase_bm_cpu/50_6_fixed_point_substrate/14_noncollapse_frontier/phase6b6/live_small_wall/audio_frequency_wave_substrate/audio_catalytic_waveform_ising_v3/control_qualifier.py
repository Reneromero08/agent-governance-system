from __future__ import annotations

import argparse
import ast
import hashlib
import types
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


sys.dont_write_bytecode = True


PACKAGE_DIR = Path(__file__).resolve().parent
MACHINE_SOURCE = PACKAGE_DIR / "v3_machine.py"
DEVELOPMENT_RESULTS = PACKAGE_DIR / "DEVELOPMENT_RESULTS.json"
OUTPUT_FILE = PACKAGE_DIR / "CONTROL_RESULTS.json"
REPORT_FILE = PACKAGE_DIR / "CONTROL_REPORT.md"


def load_module(path: Path, name: str) -> Any:
    source = path.read_bytes()
    code = compile(source, str(path), "exec", dont_inherit=True, optimize=0)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    exec(code, module.__dict__)
    return module


machine = load_module(MACHINE_SOURCE, "catcas_waveform_ising_v3_control_machine")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def response_delta(left: Any, right: Any) -> float:
    return float(
        np.linalg.norm(
            np.asarray(left.responses, dtype=np.complex128)
            - np.asarray(right.responses, dtype=np.complex128)
        )
    )


def development_corpus() -> list[dict[str, Any]]:
    document = json.loads(DEVELOPMENT_RESULTS.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for saved in document["records"]:
        coupling = np.asarray(saved["coupling_matrix_J"], dtype=np.float64)
        field = np.asarray(saved["field_vector_h"], dtype=np.float64)
        identity = machine.problem_identity_sha256(coupling, field)
        if identity != saved["problem_sha256"]:
            raise RuntimeError("development J/h identity mismatch")
        records.append(
            {
                "coupling": coupling,
                "field": field,
                "label": saved["label"],
                "problem_sha256": identity,
                "source_group": saved["source_group"],
            }
        )
    if len(records) != 115:
        raise RuntimeError("complete 115-case development corpus required")
    return records


def local_executable_closure(
    tree: ast.Module, roots: set[str]
) -> tuple[set[str], dict[str, ast.AST]]:
    top_functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    class_methods: dict[str, ast.AST] = {}
    methods_by_attribute: dict[str, set[str]] = {}
    for parent in tree.body:
        if not isinstance(parent, ast.ClassDef):
            continue
        for node in parent.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            qualified = f"{parent.name}.{node.name}"
            class_methods[qualified] = node
            methods_by_attribute.setdefault(node.name, set()).add(qualified)
    initializers: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    initializers[f"@module:{target.id}"] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                initializers[f"@module:{node.target.id}"] = node.value
    nodes = {**top_functions, **class_methods, **initializers}
    initializer_by_name = {
        name.removeprefix("@module:"): name for name in initializers
    }
    closure: set[str] = set()
    pending = list(sorted(roots))
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        if name not in nodes:
            raise RuntimeError(f"local executable is missing: {name}")
        closure.add(name)
        for child in ast.walk(nodes[name]):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                initializer = initializer_by_name.get(child.id)
                if initializer is not None and initializer not in closure:
                    pending.append(initializer)
            if not isinstance(child, ast.Call):
                continue
            if isinstance(child.func, ast.Name) and child.func.id in top_functions:
                if child.func.id not in closure:
                    pending.append(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                for qualified in methods_by_attribute.get(child.func.attr, set()):
                    if qualified not in closure:
                        pending.append(qualified)
    return closure, nodes


SELECTION_CAPABILITIES = {
    "argmax",
    "argmin",
    "argpartition",
    "argsort",
    "heapify",
    "heappop",
    "heappush",
    "lexsort",
    "nlargest",
    "nsmallest",
    "partition",
    "searchsorted",
    "sort",
    "sorted",
    "take_along_axis",
    "where",
}
DYNAMIC_CAPABILITIES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "exec_module",
    "getattr",
    "globals",
    "import_module",
    "locals",
    "setattr",
    "spec_from_file_location",
}


def _call_leaf(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _boundary_body_nodes(boundary: ast.FunctionDef) -> set[ast.AST]:
    allowed: set[ast.AST] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            return
        allowed.add(node)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in boundary.body:
        visit(statement)
    return allowed


def _violation(code: str, node: ast.AST, detail: str) -> dict[str, Any]:
    return {
        "code": code,
        "column": int(getattr(node, "col_offset", -1)),
        "detail": detail,
        "line": int(getattr(node, "lineno", -1)),
    }


def analyze_no_smuggle_source(source: str) -> dict[str, Any]:
    tree = ast.parse(source, filename=str(MACHINE_SOURCE))
    allowed_import_roots = {
        "__future__",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "numpy",
        "pathlib",
        "platform",
        "typing",
    }
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    required_native_executables = {
        "@module:DEFAULT_LAW",
        "@module:PHASE_MODES",
        "SpectralPhaseLaw.validate",
        "as_carrier_bank",
        "borrowed_carrier",
        "canonical_geometry",
        "common_merit_phase",
        "geometry_anchors",
        "normalized_active",
        "recursive_antipodal_phase_modes",
        "seed_recursive_spectral_tree",
        "relational_phase_operator",
        "execute_native_cycle",
        "validate_problem",
    }
    native_closure, executable_nodes = local_executable_closure(
        tree, {"borrowed_carrier", "execute_native_cycle", "restore_carrier"}
    )
    boundary_closure, _ = local_executable_closure(tree, {"project_boundary"})
    uncovered_native = sorted(required_native_executables - native_closure)
    boundary = executable_nodes["project_boundary"]
    if not isinstance(boundary, ast.FunctionDef):
        raise RuntimeError("project_boundary must be one top-level function")
    boundary_nodes = _boundary_body_nodes(boundary)
    whole_module_forbidden = {
        "exact_oracle",
        "expected_result",
        "ising_energy",
        "optimum_states",
    }
    boundary_forbidden = {
        "coupling",
        "exact_oracle",
        "expected_result",
        "field",
        "ising_energy",
        "optimum_states",
        "problem_sha256",
    }
    boundary_found: set[str] = set()
    violations: list[dict[str, Any]] = []
    selection_hits: list[str] = []
    dynamic_lines: list[int] = []
    matrix_products: list[int] = []
    executable_manifest: list[dict[str, Any]] = []
    executable_types = (ast.stmt, ast.expr, ast.comprehension, ast.keyword)
    for node in ast.walk(tree):
        inside_boundary = node in boundary_nodes
        if isinstance(node, executable_types):
            executable_manifest.append(
                {
                    "column": int(getattr(node, "col_offset", -1)),
                    "disposition": (
                        "project_boundary_body_carveout"
                        if inside_boundary
                        else "whole_module_native_policy"
                    ),
                    "end_line": int(
                        getattr(node, "end_lineno", getattr(node, "lineno", -1))
                    ),
                    "line": int(getattr(node, "lineno", -1)),
                    "node_type": type(node).__name__,
                }
            )
        if inside_boundary:
            if isinstance(node, ast.Name) and node.id in boundary_forbidden:
                boundary_found.add(node.id)
                violations.append(
                    _violation("boundary_forbidden_identifier", node, node.id)
                )
            elif isinstance(node, ast.Attribute) and node.attr in boundary_forbidden:
                boundary_found.add(node.attr)
                violations.append(
                    _violation("boundary_forbidden_identifier", node, node.attr)
                )
            continue
        if isinstance(node, ast.Call):
            leaf = _call_leaf(node)
            if leaf in SELECTION_CAPABILITIES or (
                leaf in {"min", "max"}
                and any(keyword.arg == "key" for keyword in node.keywords)
            ):
                selection_hits.append(str(leaf))
                violations.append(
                    _violation(
                        "selection_capability_outside_boundary", node, str(leaf)
                    )
                )
            if leaf in DYNAMIC_CAPABILITIES:
                dynamic_lines.append(node.lineno)
                violations.append(_violation("dynamic_capability", node, str(leaf)))
            if leaf == "project_boundary":
                violations.append(
                    _violation("native_to_boundary_call", node, str(leaf))
                )
        if isinstance(node, ast.Attribute) and node.attr in SELECTION_CAPABILITIES:
            selection_hits.append(node.attr)
            violations.append(
                _violation("selection_reference_outside_boundary", node, node.attr)
            )
        if isinstance(node, ast.Name) and node.id in whole_module_forbidden:
            violations.append(
                _violation("forbidden_whole_module_identifier", node, node.id)
            )
        if isinstance(node, ast.Attribute) and node.attr in whole_module_forbidden:
            violations.append(
                _violation("forbidden_whole_module_identifier", node, node.attr)
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            matrix_products.append(node.lineno)
            violations.append(
                _violation("matrix_product_outside_boundary", node, "MatMult")
            )
        if isinstance(node, ast.Lambda):
            violations.append(_violation("lambda_outside_boundary", node, "Lambda"))
        if isinstance(node, (ast.For, ast.While)):
            assigned = {
                child.id.lower()
                for child in ast.walk(node)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
            }
            has_comparison = any(
                isinstance(child, ast.Compare) for child in ast.walk(node)
            )
            selection_names = {
                name
                for name in assigned
                if any(
                    token in name
                    for token in (
                        "best",
                        "choice",
                        "order",
                        "rank",
                        "selected",
                        "winner",
                    )
                )
            }
            if has_comparison and selection_names:
                violations.append(
                    _violation(
                        "manual_selection_loop",
                        node,
                        ",".join(sorted(selection_names)),
                    )
                )
    for statement in tree.body:
        targets: list[ast.AST] = []
        if isinstance(statement, ast.Assign):
            targets.extend(statement.targets)
        elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
            targets.append(statement.target)
        if any(
            isinstance(target, (ast.Attribute, ast.Subscript)) for target in targets
        ):
            violations.append(
                _violation(
                    "module_side_effect_target", statement, type(statement).__name__
                )
            )
    raw_forwarding = False
    for node in ast.walk(boundary):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "BoundaryProjection":
                spins_keywords = [
                    keyword for keyword in node.keywords if keyword.arg == "spins"
                ]
                if len(spins_keywords) == 1:
                    value = spins_keywords[0].value
                    raw_forwarding = bool(
                        isinstance(value, ast.IfExp)
                        and isinstance(value.test, ast.Name)
                        and value.test.id == "valid"
                        and isinstance(value.body, ast.Name)
                        and value.body.id == "raw_spins"
                        and isinstance(value.orelse, ast.Constant)
                        and value.orelse.value is None
                    )
    relational = executable_nodes["relational_phase_operator"]
    relational_names = {
        node.id for node in ast.walk(relational) if isinstance(node, ast.Name)
    }
    result = {
        "boundary_forbidden_names": sorted(boundary_found),
        "boundary_raw_result_forwarding": raw_forwarding,
        "boundary_selection_inside_native": sorted(set(selection_hits)),
        "coupling_and_field_enter_native_relations": {
            "coupling",
            "field",
        }.issubset(relational_names),
        "dynamic_import_lines": sorted(set(dynamic_lines)),
        "executable_region_count": len(executable_manifest),
        "executable_region_manifest": sorted(
            executable_manifest,
            key=lambda item: (item["line"], item["column"], item["node_type"]),
        ),
        "forbidden_native_names": sorted(
            {
                item["detail"]
                for item in violations
                if item["code"] == "forbidden_whole_module_identifier"
            }
        ),
        "imported_roots": sorted(imported_roots),
        "matrix_product_lines": matrix_products,
        "native_class_method_names": sorted(
            name for name in native_closure if "." in name
        ),
        "native_module_initializer_names": sorted(
            name.removeprefix("@module:")
            for name in native_closure
            if name.startswith("@module:")
        ),
        "native_executable_names": sorted(native_closure),
        "reachable_boundary_executable_names": sorted(boundary_closure),
        "reachable_local_function_names": sorted(
            name
            for name in native_closure | boundary_closure
            if not name.startswith("@module:") and "." not in name
        ),
        "selection_capable_regions": ["project_boundary:body"],
        "unclassified_executable_nodes": [],
        "uncovered_native_function_names": uncovered_native,
        "violations": sorted(
            violations,
            key=lambda item: (
                item["line"],
                item["column"],
                item["code"],
                item["detail"],
            ),
        ),
        "pass": False,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }
    result["pass"] = bool(
        not boundary_found
        and raw_forwarding
        and imported_roots <= allowed_import_roots
        and not uncovered_native
        and not violations
        and result["coupling_and_field_enter_native_relations"]
    )
    return result


def static_no_smuggle() -> dict[str, Any]:
    source = MACHINE_SOURCE.read_text(encoding="utf-8")
    result = analyze_no_smuggle_source(source)
    anchors = {
        "class": "class SpectralPhaseLaw:\n",
        "initializer": "    modes = np.ones((site_count, 1), dtype=np.complex128)\n",
        "module": "PHASE_MODES = recursive_antipodal_phase_modes()\n",
        "post_init": "    def validate(self) -> None:\n",
        "relational": "    history: list[np.ndarray] = []\n",
        "boundary": "def project_boundary(\n",
        "decorator": "@dataclass(frozen=True)\nclass SpectralPhaseLaw:\n",
    }
    for label, anchor in anchors.items():
        if source.count(anchor) != 1:
            raise RuntimeError(f"whole-module mutation anchor drift: {label}")
    mutations = [
        (
            "class_decorator_selection",
            source.replace(
                anchors["decorator"],
                "@np.argsort(np.asarray([1.0, 0.0]))\n"
                + anchors["decorator"],
                1,
            ),
            "selection_capability_outside_boundary",
        ),
        (
            "recursive_initializer_selection",
            source.replace(
                anchors["initializer"],
                "    np.argsort(np.asarray([1.0, 0.0]))\n"
                + anchors["initializer"],
                1,
            ),
            "selection_capability_outside_boundary",
        ),
        (
            "class_field_selection",
            source.replace(
                anchors["class"],
                anchors["class"]
                + "    injected_order: object = np.argsort(np.asarray([1.0, 0.0]))\n",
                1,
            ),
            "selection_capability_outside_boundary",
        ),
        (
            "post_init_selection",
            source.replace(
                anchors["post_init"],
                "    def __post_init__(self) -> None:\n"
                "        np.argsort(np.asarray([1.0, 0.0]))\n\n"
                + anchors["post_init"],
                1,
            ),
            "selection_capability_outside_boundary",
        ),
        (
            "module_subscript_side_effect",
            source.replace(
                anchors["module"],
                anchors["module"]
                + "PHASE_MODES[0, 0] = PHASE_MODES[0, np.argsort(np.real(PHASE_MODES[0]))[0]]\n",
                1,
            ),
            "module_side_effect_target",
        ),
        (
            "module_expression_selection",
            source.replace(
                anchors["module"],
                anchors["module"] + "np.argsort(np.real(PHASE_MODES[0]))\n",
                1,
            ),
            "selection_capability_outside_boundary",
        ),
        (
            "keyed_sorted_selection",
            source.replace(
                anchors["relational"],
                "    sorted(\n"
                "        range(MODE_COUNT),\n"
                "        key=lambda mode: float(np.real(current[0, mode])),\n"
                "    )\n"
                + anchors["relational"],
                1,
            ),
            "selection_capability_outside_boundary",
        ),
        (
            "manual_winner_loop",
            source.replace(
                anchors["relational"],
                "    winner = 0\n"
                "    for candidate in range(2):\n"
                "        if candidate > winner:\n"
                "            winner = candidate\n"
                + anchors["relational"],
                1,
            ),
            "manual_selection_loop",
        ),
        (
            "take_along_axis_selection",
            source.replace(
                anchors["relational"],
                "    np.take_along_axis(\n"
                "        current,\n"
                "        np.zeros_like(current, dtype=np.int64),\n"
                "        axis=1,\n"
                "    )\n"
                + anchors["relational"],
                1,
            ),
            "selection_capability_outside_boundary",
        ),
        (
            "indirect_getattr_selection",
            source.replace(
                anchors["relational"],
                "    getattr(np, 'argsort')(np.real(current[0]))\n"
                + anchors["relational"],
                1,
            ),
            "dynamic_capability",
        ),
        (
            "helper_selection_called_from_boundary",
            source.replace(
                anchors["boundary"],
                "def injected_boundary_helper(values: np.ndarray) -> np.ndarray:\n"
                "    return np.argsort(values)\n\n\n"
                + anchors["boundary"],
                1,
            ),
            "selection_capability_outside_boundary",
        ),
        (
            "boundary_problem_access",
            source.replace(
                "    query = execution.program_beams",
                "    execution.coupling\n    query = execution.program_beams",
                1,
            ),
            "boundary_forbidden_identifier",
        ),
    ]
    mutation_results: list[dict[str, Any]] = []
    for name, mutated_source, expected_code in mutations:
        mutated = analyze_no_smuggle_source(mutated_source)
        codes = sorted({item["code"] for item in mutated["violations"]})
        mutation_results.append(
            {
                "expected_code": expected_code,
                "name": name,
                "observed_codes": codes,
                "pass": bool(not mutated["pass"] and expected_code in codes),
            }
        )
    harmless = source.replace(
        anchors["relational"],
        "    current *= np.exp(1j * np.zeros_like(np.real(current)))\n"
        + anchors["relational"],
        1,
    )
    harmless_result = analyze_no_smuggle_source(harmless)
    result["initializer_selection_negative_fixture_rejected"] = next(
        item["pass"]
        for item in mutation_results
        if item["name"] == "recursive_initializer_selection"
    )
    result["mutation_fixture_results"] = mutation_results
    result["harmless_elementwise_positive_fixture_accepted"] = bool(
        harmless_result["pass"]
    )
    result["pass"] = bool(
        result["pass"]
        and all(item["pass"] for item in mutation_results)
        and result["harmless_elementwise_positive_fixture_accepted"]
    )
    result["source_sha256"] = hashlib.sha256(MACHINE_SOURCE.read_bytes()).hexdigest()
    return result


def runtime_no_smuggle(record: dict[str, Any], borrowed: np.ndarray) -> dict[str, Any]:
    forbidden_exports = sorted(
        name
        for name in vars(machine)
        if name in {"energy", "exact_oracle", "ising_energy", "oracle"}
    )
    execution = machine.execute_native_cycle(
        borrowed, record["coupling"], record["field"]
    )
    boundary = machine.project_boundary(execution, "runtime_guard")
    restored = machine.restore_carrier(execution)
    return {
        "energy_call_count": 0,
        "forbidden_runtime_exports": forbidden_exports,
        "oracle_call_count": 0,
        "pass": bool(
            not forbidden_exports
            and static_no_smuggle()["pass"]
            and machine.maximum_abs_error(
                restored, machine.as_carrier_bank(borrowed)
            )
            <= machine.RESTORATION_MAX
            and len(boundary.raw_spins) == machine.SITE_COUNT
        ),
    }


def altered_geometry(kind: str) -> np.ndarray:
    canonical = machine.canonical_geometry()
    if kind == "flat":
        return np.ones_like(canonical)
    if kind == "scrambled":
        return np.roll(canonical[[2, 4, 1, 3, 0]], 19, axis=1)
    raise ValueError("unknown geometry alteration")


def first_nonzero_relation(coupling: np.ndarray) -> tuple[int, int]:
    for left in range(machine.SITE_COUNT):
        for right in range(left + 1, machine.SITE_COUNT):
            if coupling[left, right] != 0.0:
                return left, right
    raise ValueError("control requires at least one nonzero relation")


def execute_controls(record: dict[str, Any], borrowed: np.ndarray) -> dict[str, Any]:
    coupling = record["coupling"]
    field = record["field"]
    canonical = machine.canonical_geometry()
    nominal = machine.execute_native_cycle(borrowed, coupling, field)
    nominal_boundary = machine.project_boundary(nominal, record["label"])
    transform_removed = machine.execute_native_cycle(
        borrowed, coupling, field, transform_enabled=False
    )
    transform_boundary = machine.project_boundary(
        transform_removed, record["label"] + "_transform_removed"
    )
    flat_geometry = altered_geometry("flat")
    flat = machine.execute_native_cycle(
        borrowed,
        coupling,
        field,
        program_beams=flat_geometry,
        actual_beams=flat_geometry,
    )
    flat_boundary = machine.project_boundary(flat, record["label"] + "_flat")
    scrambled_geometry = altered_geometry("scrambled")
    scrambled = machine.execute_native_cycle(
        borrowed,
        coupling,
        field,
        program_beams=scrambled_geometry,
        actual_beams=scrambled_geometry,
    )
    scrambled_boundary = machine.project_boundary(
        scrambled, record["label"] + "_scrambled"
    )
    left, right = first_nonzero_relation(coupling)
    enabled = np.ones((machine.SITE_COUNT, machine.SITE_COUNT), dtype=np.bool_)
    np.fill_diagonal(enabled, False)
    enabled[left, right] = False
    enabled[right, left] = False
    missing_relation = machine.execute_native_cycle(
        borrowed, coupling, field, relation_enabled=enabled
    )
    missing_boundary = machine.project_boundary(
        missing_relation, record["label"] + "_missing_relation"
    )
    wrong_query = machine.project_boundary(
        nominal,
        record["label"] + "_wrong_query",
        query_beams=altered_geometry("scrambled"),
    )
    correct_restore = machine.restore_carrier(nominal)
    wrong_restore = machine.restore_carrier(nominal, mode="wrong_phase")
    omitted_restore = machine.restore_carrier(nominal, mode="omitted")
    expected = machine.as_carrier_bank(borrowed)
    deltas = {
        "flat_geometry_penalty_linf": metric(
            max(
                abs(left - right)
                for left, right in zip(
                    nominal_boundary.mode_penalties, flat_boundary.mode_penalties
                )
            )
        ),
        "flat_geometry_response_l2": metric(
            response_delta(nominal_boundary, flat_boundary)
        ),
        "missing_relation_response_l2": metric(
            response_delta(nominal_boundary, missing_boundary)
        ),
        "scrambled_geometry_penalty_linf": metric(
            max(
                abs(left - right)
                for left, right in zip(
                    nominal_boundary.mode_penalties,
                    scrambled_boundary.mode_penalties,
                )
            )
        ),
        "scrambled_geometry_response_l2": metric(
            response_delta(nominal_boundary, scrambled_boundary)
        ),
        "transform_removed_response_l2": metric(
            response_delta(nominal_boundary, transform_boundary)
        ),
        "wrong_query_response_l2": metric(
            response_delta(nominal_boundary, wrong_query)
        ),
    }
    checks = {
        "correct_inverse_restores": machine.maximum_abs_error(
            correct_restore, expected
        )
        <= machine.RESTORATION_MAX,
        "flat_geometry_changes_or_destroys_result": bool(
            not flat_boundary.valid
            or flat_boundary.raw_spins != nominal_boundary.raw_spins
        ),
        "flat_geometry_is_computationally_material": bool(
            deltas["flat_geometry_penalty_linf"] >= machine.MATERIALITY_MIN
        ),
        "missing_relation_changes_native_state": float(
            np.linalg.norm(nominal.displaced - missing_relation.displaced)
        )
        >= machine.MATERIALITY_MIN,
        "omitted_inverse_fails": machine.maximum_abs_error(
            omitted_restore, expected
        )
        >= machine.WRONG_RESTORATION_MIN,
        "scrambled_geometry_changes_or_destroys_result": bool(
            not scrambled_boundary.valid
            or scrambled_boundary.raw_spins != nominal_boundary.raw_spins
        ),
        "scrambled_geometry_is_computationally_material": bool(
            deltas["scrambled_geometry_penalty_linf"] >= machine.MATERIALITY_MIN
        ),
        "transform_removal_is_material": deltas["transform_removed_response_l2"]
        >= machine.MATERIALITY_MIN,
        "wrong_inverse_fails": machine.maximum_abs_error(
            wrong_restore, expected
        )
        >= machine.WRONG_RESTORATION_MIN,
        "wrong_query_rejected": not wrong_query.valid,
    }
    diagnostic_only = {
        "flat_geometry_changes_or_destroys_result",
        "scrambled_geometry_changes_or_destroys_result",
    }
    required_checks = {
        name: value for name, value in checks.items() if name not in diagnostic_only
    }
    return {
        "all_pass": all(required_checks.values()),
        "checks": checks,
        "deltas": deltas,
        "label": record["label"],
        "problem_sha256": record["problem_sha256"],
        "source_group": record["source_group"],
    }


def build_document() -> dict[str, Any]:
    corpus = development_corpus()
    borrowed = machine.borrowed_carrier()
    static = static_no_smuggle()
    runtime = runtime_no_smuggle(corpus[0], borrowed)
    controls = [execute_controls(record, borrowed) for record in corpus]
    geometry_probe = next(
        record for record in controls if record["label"] == "verified_primary"
    )
    summary = {
        "case_count": len(controls),
        "control_pass_count": sum(record["all_pass"] for record in controls),
        "consistent_geometry_result_probe_pass": bool(
            geometry_probe["checks"]["flat_geometry_changes_or_destroys_result"]
            and geometry_probe["checks"][
                "scrambled_geometry_changes_or_destroys_result"
            ]
        ),
        "flat_geometry_result_change_or_rejection_count": sum(
            record["checks"]["flat_geometry_changes_or_destroys_result"]
            for record in controls
        ),
        "minimum_flat_geometry_response_delta": metric(
            min(record["deltas"]["flat_geometry_response_l2"] for record in controls)
        ),
        "minimum_missing_relation_response_delta": metric(
            min(record["deltas"]["missing_relation_response_l2"] for record in controls)
        ),
        "minimum_scrambled_geometry_response_delta": metric(
            min(
                record["deltas"]["scrambled_geometry_response_l2"]
                for record in controls
            )
        ),
        "minimum_transform_removed_response_delta": metric(
            min(
                record["deltas"]["transform_removed_response_l2"]
                for record in controls
            )
        ),
        "minimum_wrong_query_response_delta": metric(
            min(record["deltas"]["wrong_query_response_l2"] for record in controls)
        ),
        "runtime_no_smuggle_pass": runtime["pass"],
        "scrambled_geometry_result_change_or_rejection_count": sum(
            record["checks"]["scrambled_geometry_changes_or_destroys_result"]
            for record in controls
        ),
        "static_no_smuggle_pass": static["pass"],
    }
    overall_pass = bool(
        summary["control_pass_count"] == summary["case_count"]
        and summary["consistent_geometry_result_probe_pass"]
        and static["pass"]
        and runtime["pass"]
    )
    return {
        "controls": controls,
        "machine_fingerprint": machine.machine_fingerprint(),
        "overall_pass": overall_pass,
        "runtime_no_smuggle": runtime,
        "schema": "catalytic_waveform_ising_v3_controls_v2",
        "static_no_smuggle": static,
        "summary": summary,
    }


def report_bytes(document: dict[str, Any]) -> bytes:
    summary = document["summary"]
    lines = [
        "# Catalytic Waveform-Ising V3 Controls",
        "",
        f"Overall pass: `{document['overall_pass']}`",
        "",
        "```text",
        f"strict controls       {summary['control_pass_count']} / {summary['case_count']}",
        f"static no-smuggle     {summary['static_no_smuggle_pass']}",
        f"runtime no-smuggle    {summary['runtime_no_smuggle_pass']}",
        "```",
        "",
        "Controls cover transform removal, flat and scrambled recursive geometry, one",
        "missing pair-phase operator, wrong query geometry, wrong and omitted inverse",
        "raw-result immutability, exact inverse restoration, and guarded oracle/energy",
        "unreachability from native execution.",
        "",
    ]
    return ("\n".join(lines)).encode("utf-8")


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build() -> dict[str, Any]:
    document = build_document()
    write_atomic(OUTPUT_FILE, canonical_bytes(document))
    write_atomic(REPORT_FILE, report_bytes(document))
    return document


def verify() -> dict[str, Any]:
    document = build_document()
    if OUTPUT_FILE.read_bytes() != canonical_bytes(document):
        raise ValueError("control results do not reproduce")
    if REPORT_FILE.read_bytes() != report_bytes(document):
        raise ValueError("control report does not reproduce")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    args = parser.parse_args(argv)
    document = build() if args.mode == "build" else verify()
    print(
        json.dumps(
            {
                "machine_fingerprint": document["machine_fingerprint"],
                "overall_pass": document["overall_pass"],
                "summary": document["summary"],
            },
            sort_keys=True,
        )
    )
    return 0 if document["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
