from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
MACHINE_SOURCE = PACKAGE_DIR / "successor_machine.py"
PROOF_FILE = PACKAGE_DIR / "NO_SMUGGLE_PROOF.json"


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


v2 = load_module(MACHINE_SOURCE, "catcas_v2_no_smuggle_machine")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


NATIVE_FUNCTIONS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("execute_native_cycle", v2.execute_native_cycle),
    ("transform_channel_bank", v2.transform_channel_bank),
    ("channel_calibration", v2.channel_calibration),
    ("evolve_native_waveforms", v2.evolve_native_waveforms),
    ("native_wave_velocity", v2.native_wave_velocity),
    ("schedule", v2.schedule),
    ("transport_shift", v2.transport_shift),
    ("r4.render_trees", v2.r4.render_trees),
    ("r4.geometry_channel_bank", v2.r4.geometry_channel_bank),
    ("r4.transport_mask_bank", v2.r4.transport_mask_bank),
    ("r4.validate_problem", v2.r4.validate_problem),
)


FORBIDDEN_IDENTIFIERS = {
    "accepted",
    "answer",
    "cached_answer",
    "decoded_spins",
    "energy",
    "exact_oracle",
    "expected_result",
    "optimum",
    "oracle_rows",
    "prior_outcome",
    "raw_signs",
    "raw_spins",
    "result_latch",
    "score",
    "scalar_js",
    "spins",
    "winner",
}


def function_record(label: str, function: Callable[..., Any]) -> dict[str, Any]:
    source = inspect.getsource(function)
    tree = ast.parse(source)
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    arguments = {
        argument.arg
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for argument in (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        )
    }
    local_stores = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    builtin_names = set(dir(builtins))
    global_reads = sorted(
        name
        for name in identifiers
        if name not in arguments and name not in local_stores and name not in builtin_names
    )
    forbidden = sorted(identifiers & FORBIDDEN_IDENTIFIERS)
    matrix_multiply_count = sum(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult)
        for node in ast.walk(tree)
    )
    dynamic_calls = sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"eval", "exec"}
        for node in ast.walk(tree)
    )
    closure = inspect.getclosurevars(function)
    return {
        "argument_names": sorted(arguments),
        "closure_nonlocals": sorted(closure.nonlocals),
        "dynamic_evaluation_call_count": dynamic_calls,
        "forbidden_identifier_findings": forbidden,
        "global_reads": global_reads,
        "label": label,
        "matrix_multiply_count": matrix_multiply_count,
        "source_sha256": sha256_bytes(source.encode("utf-8")),
    }


def guarded_runtime_probe() -> dict[str, Any]:
    def forbidden_boundary(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("a forbidden scalar/adjudication boundary was reached")

    guarded_names = (
        "development_instances",
        "exact_oracle",
        "ising_energy",
        "project_boundary",
    )
    originals = {name: getattr(v2, name) for name in guarded_names}
    try:
        for name in guarded_names:
            setattr(v2, name, forbidden_boundary)
        carrier = v2.r4.borrowed_carrier()
        borrowed = np.repeat(carrier[np.newaxis, :], v2.SITE_COUNT, axis=0)
        first = v2.execute_native_cycle(
            borrowed, v2.r4.COUPLING, v2.r4.PRIMARY_FIELD
        )
        second = v2.execute_native_cycle(
            borrowed, v2.r4.COUPLING, v2.r4.PRIMARY_FIELD
        )
    finally:
        for name, function in originals.items():
            setattr(v2, name, function)
    repeat_displaced_delta = float(np.max(np.abs(first.displaced - second.displaced)))
    repeat_history_delta = float(
        np.max(np.abs(first.operator_history - second.operator_history))
    )
    return {
        "guarded_boundary_names": list(guarded_names),
        "native_execution_completed_with_all_boundaries_guarded": True,
        "repeat_displaced_max_delta": metric(repeat_displaced_delta),
        "repeat_history_max_delta": metric(repeat_history_delta),
        "runtime_inputs": [
            "borrowed_complex_carrier",
            "coupling_matrix_J",
            "field_vector_h",
            "recursive_complex_geometry",
            "frozen_machine_law",
        ],
    }


def build_document() -> dict[str, Any]:
    functions = [function_record(label, function) for label, function in NATIVE_FUNCTIONS]
    runtime = guarded_runtime_probe()
    forbidden_findings = [
        {"function": record["label"], "identifiers": record["forbidden_identifier_findings"]}
        for record in functions
        if record["forbidden_identifier_findings"]
    ]
    matrix_multiply_count = sum(record["matrix_multiply_count"] for record in functions)
    dynamic_count = sum(
        record["dynamic_evaluation_call_count"] for record in functions
    )
    mutable_answer_globals = sorted(
        name
        for name, value in vars(v2).items()
        if name.lower() in FORBIDDEN_IDENTIFIERS
        and isinstance(value, (dict, list, set, np.ndarray))
    )
    pass_conditions = {
        "deterministic_repeat": (
            runtime["repeat_displaced_max_delta"] == 0.0
            and runtime["repeat_history_max_delta"] == 0.0
        ),
        "dynamic_evaluation_absent": dynamic_count == 0,
        "forbidden_identifiers_absent_from_native_path": not forbidden_findings,
        "matrix_multiply_absent": matrix_multiply_count == 0,
        "mutable_answer_globals_absent": not mutable_answer_globals,
        "scalar_boundaries_unreachable_during_guarded_runtime": runtime[
            "native_execution_completed_with_all_boundaries_guarded"
        ],
    }
    return {
        "actual_transitive_native_path": [label for label, _ in NATIVE_FUNCTIONS],
        "function_records": functions,
        "mutable_answer_global_findings": mutable_answer_globals,
        "pass": all(pass_conditions.values()),
        "pass_conditions": pass_conditions,
        "runtime_probe": runtime,
        "schema": "catalytic_waveform_ising_v2_no_smuggle_proof_v1",
        "scope": (
            "actual native call path, arguments, closures, global reads, guarded runtime, "
            "and deterministic repeated execution"
        ),
    }


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def build() -> dict[str, Any]:
    document = build_document()
    write_atomic(PROOF_FILE, canonical_bytes(document))
    return document


def verify() -> dict[str, Any]:
    document = build_document()
    if PROOF_FILE.read_bytes() != canonical_bytes(document):
        raise ValueError("no-smuggle proof does not reproduce")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    document = build() if args.command == "build" else verify()
    print(
        json.dumps(
            {
                "pass": document["pass"],
                "runtime_probe": document["runtime_probe"],
            },
            sort_keys=True,
        )
    )
    return 0 if document["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
