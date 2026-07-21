from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
GATE_SOURCE = PACKAGE_DIR / "stability_gate.py"
OUTPUT_FILE = PACKAGE_DIR / "NO_SMUGGLE_PROOF.json"

FORBIDDEN_NAMES = {
    "coupling",
    "development_instances",
    "energy",
    "exact_oracle",
    "field",
    "ising_energy",
    "optimum",
    "oracle",
    "problem_hash",
    "problem_index",
    "project_boundary",
    "raw_spins",
}


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


gate = load_module(GATE_SOURCE, "catcas_stability_gate_no_smuggle")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ast_proof() -> dict[str, Any]:
    source = GATE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(GATE_SOURCE))
    observed_names: set[str] = set()
    forbidden_calls: list[str] = []
    forbidden_attributes: list[str] = []
    matmul_nodes = 0
    long_hex_literals: list[str] = []
    conditional_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            observed_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            observed_names.add(node.attr)
            if node.attr in FORBIDDEN_NAMES:
                forbidden_attributes.append(node.attr)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
                forbidden_calls.append(node.func.id)
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_NAMES:
                forbidden_calls.append(node.func.attr)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            matmul_nodes += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            long_hex_literals.extend(re.findall(r"\b[0-9a-fA-F]{64}\b", node.value))
        elif isinstance(node, (ast.If, ast.IfExp)):
            conditional_count += 1
    forbidden_present = sorted(observed_names & FORBIDDEN_NAMES)
    contract = gate.gate_contract()
    pass_result = not any(
        (
            forbidden_present,
            forbidden_calls,
            forbidden_attributes,
            long_hex_literals,
            matmul_nodes,
        )
    )
    return {
        "conditional_count": conditional_count,
        "forbidden_attributes": sorted(forbidden_attributes),
        "forbidden_calls": sorted(forbidden_calls),
        "forbidden_names_present": forbidden_present,
        "gate_contract": contract,
        "long_hex_identity_literals": long_hex_literals,
        "matmul_node_count": matmul_nodes,
        "pass": pass_result,
        "source_sha256": sha256_file(GATE_SOURCE),
    }


def runtime_probe() -> dict[str, Any]:
    v2 = gate.v2
    carrier = v2.r4.borrowed_carrier()
    borrowed = np.repeat(carrier[np.newaxis, :], v2.SITE_COUNT, axis=0)
    coupling = np.asarray(v2.r4.COUPLING, dtype=np.float64)
    field = np.asarray(v2.r4.PRIMARY_FIELD, dtype=np.float64)
    execution = v2.execute_native_cycle(borrowed, coupling, field, law=v2.DEFAULT_LAW)

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("forbidden scalar or adjudication boundary reached")

    guarded_names = (
        "development_instances",
        "exact_oracle",
        "ising_energy",
        "project_boundary",
    )
    originals = {name: getattr(v2, name) for name in guarded_names}
    try:
        for name in guarded_names:
            setattr(v2, name, blocked)
        first = gate.evaluate_stability(execution)
        second = gate.evaluate_stability(execution)
    finally:
        for name, value in originals.items():
            setattr(v2, name, value)
    first_document = first.document()
    second_document = second.document()
    return {
        "deterministic_repeat": first_document == second_document,
        "diagnostic_restoration_pass": (
            first.diagnostic_restoration_max_abs_error
            <= gate.DIAGNOSTIC_RESTORATION_MAX
        ),
        "gate_returns_only_reject_accept_and_observables": (
            set(first_document)
            == {
                "checkpoint_response_sha256",
                "diagnostic_replay_max_abs_delta",
                "diagnostic_restoration_max_abs_error",
                "gate_pass",
                "joint_instability_score",
                "late_max_phase_velocity_rad_per_step",
                "late_mean_response_drift_l2",
            }
        ),
        "guarded_names": list(guarded_names),
        "native_gate_execution_completed_with_all_boundaries_guarded": True,
        "result": first_document,
        "zero_perturbation_replay_pass": (
            first.diagnostic_replay_max_abs_delta <= gate.REPLAY_TOLERANCE
        ),
    }


def build_document() -> dict[str, Any]:
    source = ast_proof()
    runtime = runtime_probe()
    pass_conditions = {
        "ast_no_forbidden_identity_or_scalar_path": source["pass"],
        "diagnostic_restoration": runtime["diagnostic_restoration_pass"],
        "deterministic_repeat": runtime["deterministic_repeat"],
        "reject_only_output_surface": runtime[
            "gate_returns_only_reject_accept_and_observables"
        ],
        "runtime_boundaries_blocked": runtime[
            "native_gate_execution_completed_with_all_boundaries_guarded"
        ],
        "zero_perturbation_replay": runtime["zero_perturbation_replay_pass"],
    }
    return {
        "null_baseline": (
            "gate disabled leaves the nominal V2 acceptance and raw result unchanged"
        ),
        "pass": all(pass_conditions.values()),
        "pass_conditions": pass_conditions,
        "runtime_probe": runtime,
        "schema": "catalytic_waveform_ising_v2_stability_no_smuggle_v1",
        "source_inspection": source,
    }


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build() -> dict[str, Any]:
    document = build_document()
    write_atomic(OUTPUT_FILE, canonical_bytes(document))
    return {"pass": document["pass"], "sha256": sha256_file(OUTPUT_FILE)}


def verify() -> dict[str, Any]:
    expected = canonical_bytes(build_document())
    if OUTPUT_FILE.read_bytes() != expected:
        raise RuntimeError("no-smuggle proof does not reproduce")
    document = json.loads(expected)
    if not document["pass"]:
        raise RuntimeError("no-smuggle proof failed")
    return {"pass": True, "sha256": hashlib.sha256(expected).hexdigest()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    args = parser.parse_args(argv)
    result = build() if args.mode == "build" else verify()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
