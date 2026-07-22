from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
MACHINE_SOURCE = PACKAGE_DIR / "v3_machine.py"
DEVELOPMENT_RESULTS = PACKAGE_DIR / "DEVELOPMENT_RESULTS.json"
OUTPUT_FILE = PACKAGE_DIR / "CONTROL_RESULTS.json"
REPORT_FILE = PACKAGE_DIR / "CONTROL_REPORT.md"


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
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


def function_source_nodes(tree: ast.Module, names: set[str]) -> list[ast.AST]:
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]


def static_no_smuggle() -> dict[str, Any]:
    source = MACHINE_SOURCE.read_text(encoding="utf-8")
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
    dynamic_import_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {
                "__import__",
                "compile",
                "eval",
                "exec",
            }:
                dynamic_import_lines.append(node.lineno)
            elif isinstance(node.func, ast.Attribute) and node.func.attr in {
                "exec_module",
                "import_module",
                "spec_from_file_location",
            }:
                dynamic_import_lines.append(node.lineno)
    native_names = {
        "as_carrier_bank",
        "borrowed_carrier",
        "canonical_geometry",
        "common_merit_phase",
        "geometry_anchors",
        "normalized_active",
        "seed_recursive_spectral_tree",
        "relational_phase_operator",
        "execute_native_cycle",
        "validate_problem",
    }
    native_nodes = function_source_nodes(tree, native_names)
    if {node.name for node in native_nodes} != native_names:
        raise RuntimeError("native function set is incomplete")
    local_functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    closure: set[str] = set()
    pending = ["borrowed_carrier", "execute_native_cycle", "project_boundary", "restore_carrier"]
    while pending:
        name = pending.pop()
        if name in closure or name not in local_functions:
            continue
        closure.add(name)
        for node in ast.walk(local_functions[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in local_functions and node.func.id not in closure:
                    pending.append(node.func.id)
    uncovered_native = sorted(native_names - closure)
    boundary_names = {"project_boundary"}
    boundary_nodes = function_source_nodes(tree, boundary_names)
    if {node.name for node in boundary_nodes} != boundary_names:
        raise RuntimeError("boundary function set is incomplete")
    forbidden_names = {
        "exact_oracle",
        "ising_energy",
        "optimum_states",
        "problem_sha256",
        "raw_spins",
        "spins",
    }
    found_names: set[str] = set()
    matrix_products: list[int] = []
    boundary_selection_inside_native: list[str] = []
    for root in native_nodes:
        for node in ast.walk(root):
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                found_names.add(node.id)
            if isinstance(node, ast.Attribute) and node.attr in forbidden_names:
                found_names.add(node.attr)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                matrix_products.append(node.lineno)
            if isinstance(node, ast.Attribute) and node.attr in {
                "argsort",
                "argmin",
                "argmax",
            }:
                boundary_selection_inside_native.append(node.attr)
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
    raw_forwarding = False
    for root in boundary_nodes:
        for node in ast.walk(root):
            if isinstance(node, ast.Name) and node.id in boundary_forbidden:
                boundary_found.add(node.id)
            if isinstance(node, ast.Attribute) and node.attr in boundary_forbidden:
                boundary_found.add(node.attr)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "BoundaryProjection":
                    spins_keywords = [
                        keyword
                        for keyword in node.keywords
                        if keyword.arg == "spins"
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
    relational = next(
        node for node in native_nodes if node.name == "relational_phase_operator"
    )
    relational_names = {
        node.id for node in ast.walk(relational) if isinstance(node, ast.Name)
    }
    result = {
        "boundary_forbidden_names": sorted(boundary_found),
        "boundary_raw_result_forwarding": raw_forwarding,
        "boundary_selection_inside_native": sorted(boundary_selection_inside_native),
        "coupling_and_field_enter_native_relations": {
            "coupling",
            "field",
        }.issubset(relational_names),
        "dynamic_import_lines": dynamic_import_lines,
        "forbidden_native_names": sorted(found_names),
        "imported_roots": sorted(imported_roots),
        "matrix_product_lines": matrix_products,
        "native_function_names": sorted(native_names),
        "reachable_local_function_names": sorted(closure),
        "uncovered_native_function_names": uncovered_native,
        "pass": False,
        "source_sha256": hashlib.sha256(MACHINE_SOURCE.read_bytes()).hexdigest(),
    }
    result["pass"] = bool(
        not boundary_found
        and raw_forwarding
        and not found_names
        and imported_roots <= allowed_import_roots
        and not dynamic_import_lines
        and not uncovered_native
        and not matrix_products
        and not boundary_selection_inside_native
        and result["coupling_and_field_enter_native_relations"]
    )
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
