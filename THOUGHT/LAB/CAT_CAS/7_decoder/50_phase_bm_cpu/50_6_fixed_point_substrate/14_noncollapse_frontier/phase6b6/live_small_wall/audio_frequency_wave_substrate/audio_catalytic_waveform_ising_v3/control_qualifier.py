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
DEVELOPMENT_SOURCE = PACKAGE_DIR / "development_qualifier.py"
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
development = load_module(
    DEVELOPMENT_SOURCE, "catcas_waveform_ising_v3_control_development"
)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def function_source_nodes(tree: ast.Module, names: set[str]) -> list[ast.AST]:
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]


def static_no_smuggle() -> dict[str, Any]:
    source = MACHINE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MACHINE_SOURCE))
    native_names = {
        "seed_recursive_spectral_tree",
        "relational_phase_operator",
        "execute_native_cycle",
    }
    native_nodes = function_source_nodes(tree, native_names)
    if {node.name for node in native_nodes} != native_names:
        raise RuntimeError("native function set is incomplete")
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
    relational = next(
        node for node in native_nodes if node.name == "relational_phase_operator"
    )
    relational_names = {
        node.id for node in ast.walk(relational) if isinstance(node, ast.Name)
    }
    result = {
        "boundary_selection_inside_native": sorted(boundary_selection_inside_native),
        "coupling_and_field_enter_native_relations": {
            "coupling",
            "field",
        }.issubset(relational_names),
        "forbidden_native_names": sorted(found_names),
        "matrix_product_lines": matrix_products,
        "native_function_names": sorted(native_names),
        "pass": False,
        "source_sha256": hashlib.sha256(MACHINE_SOURCE.read_bytes()).hexdigest(),
    }
    result["pass"] = bool(
        not found_names
        and not matrix_products
        and not boundary_selection_inside_native
        and result["coupling_and_field_enter_native_relations"]
    )
    return result


def runtime_no_smuggle(record: dict[str, Any], borrowed: np.ndarray) -> dict[str, Any]:
    calls = {"energy": 0, "oracle": 0}
    original_energy = machine.v2.ising_energy
    original_oracle = machine.v2.exact_oracle

    def blocked_energy(*args: Any, **kwargs: Any) -> Any:
        calls["energy"] += 1
        raise RuntimeError("energy is unreachable from V3 native execution")

    def blocked_oracle(*args: Any, **kwargs: Any) -> Any:
        calls["oracle"] += 1
        raise RuntimeError("oracle is unreachable from V3 native execution")

    machine.v2.ising_energy = blocked_energy
    machine.v2.exact_oracle = blocked_oracle
    try:
        execution = machine.execute_native_cycle(
            borrowed, record["coupling"], record["field"]
        )
        boundary = machine.project_boundary(execution, "runtime_guard")
        restored = machine.restore_carrier(execution)
    finally:
        machine.v2.ising_energy = original_energy
        machine.v2.exact_oracle = original_oracle
    return {
        "energy_call_count": calls["energy"],
        "oracle_call_count": calls["oracle"],
        "pass": bool(
            calls == {"energy": 0, "oracle": 0}
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
    flat = machine.execute_native_cycle(
        borrowed,
        coupling,
        field,
        program_beams=canonical,
        actual_beams=altered_geometry("flat"),
    )
    flat_boundary = machine.project_boundary(flat, record["label"] + "_flat")
    scrambled = machine.execute_native_cycle(
        borrowed,
        coupling,
        field,
        program_beams=canonical,
        actual_beams=altered_geometry("scrambled"),
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
        "flat_geometry_response_l2": metric(
            development.response_delta(nominal_boundary, flat_boundary)
        ),
        "missing_relation_response_l2": metric(
            development.response_delta(nominal_boundary, missing_boundary)
        ),
        "scrambled_geometry_response_l2": metric(
            development.response_delta(nominal_boundary, scrambled_boundary)
        ),
        "transform_removed_response_l2": metric(
            development.response_delta(nominal_boundary, transform_boundary)
        ),
        "wrong_query_response_l2": metric(
            development.response_delta(nominal_boundary, wrong_query)
        ),
    }
    checks = {
        "correct_inverse_restores": machine.maximum_abs_error(
            correct_restore, expected
        )
        <= machine.RESTORATION_MAX,
        "flat_geometry_rejected": not flat_boundary.valid,
        "missing_relation_changes_native_state": float(
            np.linalg.norm(nominal.displaced - missing_relation.displaced)
        )
        >= machine.MATERIALITY_MIN,
        "omitted_inverse_fails": machine.maximum_abs_error(
            omitted_restore, expected
        )
        >= machine.WRONG_RESTORATION_MIN,
        "raw_result_not_corrected": nominal_boundary.spins
        in (None, nominal_boundary.raw_spins),
        "scrambled_geometry_rejected": not scrambled_boundary.valid,
        "transform_removal_is_material": deltas["transform_removed_response_l2"]
        >= machine.MATERIALITY_MIN,
        "wrong_inverse_fails": machine.maximum_abs_error(
            wrong_restore, expected
        )
        >= machine.WRONG_RESTORATION_MIN,
        "wrong_query_rejected": not wrong_query.valid,
    }
    return {
        "all_pass": all(checks.values()),
        "checks": checks,
        "deltas": deltas,
        "label": record["label"],
        "problem_sha256": record["problem_sha256"],
        "source_group": record["source_group"],
    }


def build_document() -> dict[str, Any]:
    corpus = development.development_corpus()
    borrowed = development.v2.r4.borrowed_carrier()
    static = static_no_smuggle()
    runtime = runtime_no_smuggle(corpus[0], borrowed)
    controls = [execute_controls(record, borrowed) for record in corpus]
    summary = {
        "case_count": len(controls),
        "control_pass_count": sum(record["all_pass"] for record in controls),
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
        "static_no_smuggle_pass": static["pass"],
    }
    overall_pass = bool(
        summary["control_pass_count"] == summary["case_count"]
        and static["pass"]
        and runtime["pass"]
    )
    return {
        "controls": controls,
        "machine_fingerprint": machine.machine_fingerprint(),
        "overall_pass": overall_pass,
        "runtime_no_smuggle": runtime,
        "schema": "catalytic_waveform_ising_v3_controls_v1",
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
