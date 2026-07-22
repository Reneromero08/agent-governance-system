from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
SUBSTRATE_DIR = PACKAGE_DIR.parent
MACHINE_SOURCE = PACKAGE_DIR / "v3_machine.py"
CONTROL_SOURCE = PACKAGE_DIR / "control_qualifier.py"
FREEZE_FILE = PACKAGE_DIR / "V3_FREEZE.json"
BATCH_FILE = PACKAGE_DIR / "V3_BATCH_CUSTODY.json"
EVIDENCE_FILE = PACKAGE_DIR / "V3_PREORACLE_EVIDENCE.json"
TRACE_FILE = PACKAGE_DIR / "V3_PREORACLE_TRACE.json"
SEAL_FILE = PACKAGE_DIR / "V3_PREORACLE_SEAL.json"
RESULTS_FILE = PACKAGE_DIR / "V3_BATCH_RESULTS.json"
VERIFICATION_FILE = PACKAGE_DIR / "V3_INDEPENDENT_VERIFICATION.json"
REVIEW_FILE = PACKAGE_DIR / "V3_INDEPENDENT_REVIEW.md"

DECISION = "CATALYTIC_WAVEFORM_ISING_V3_VERIFIED"
REVIEWER_ID = "V3-INDEPENDENT-REEXECUTION-VERIFIER-02"


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


machine = load_module(MACHINE_SOURCE, "catcas_waveform_ising_v3_independent_machine_v2")
controls = load_module(CONTROL_SOURCE, "catcas_waveform_ising_v3_independent_controls_v2")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def problem_sha256(coupling: np.ndarray, field: np.ndarray) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "coupling_matrix_J": np.asarray(
                    coupling, dtype=np.float64
                ).tolist(),
                "field_vector_h": np.asarray(field, dtype=np.float64).tolist(),
            }
        )
    )


def response_delta(left: Any, right: Any) -> float:
    return float(
        np.linalg.norm(
            np.asarray(left.responses, dtype=np.complex128)
            - np.asarray(right.responses, dtype=np.complex128)
        )
    )


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
    raise ValueError("control requires a nonzero relation")


def static_integrity() -> dict[str, Any]:
    tree = ast.parse(MACHINE_SOURCE.read_text(encoding="utf-8"))
    native_names = {
        "as_carrier_bank",
        "borrowed_carrier",
        "canonical_geometry",
        "common_merit_phase",
        "execute_native_cycle",
        "geometry_anchors",
        "normalized_active",
        "relational_phase_operator",
        "seed_recursive_spectral_tree",
        "validate_problem",
    }
    boundary_names = {"project_boundary"}
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = sorted((native_names | boundary_names) - functions.keys())
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
    closure: set[str] = set()
    pending = ["borrowed_carrier", "execute_native_cycle", "project_boundary", "restore_carrier"]
    while pending:
        name = pending.pop()
        if name in closure or name not in functions:
            continue
        closure.add(name)
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in functions and node.func.id not in closure:
                    pending.append(node.func.id)
    uncovered_native = sorted(native_names - closure)
    native_forbidden = {
        "argmax",
        "argmin",
        "argsort",
        "exact_oracle",
        "ising_energy",
        "optimum_states",
        "problem_sha256",
        "raw_spins",
        "spins",
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
    native_hits: set[str] = set()
    boundary_hits: set[str] = set()
    matrix_product_lines: list[int] = []
    for name in native_names:
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Name) and node.id in native_forbidden:
                native_hits.add(node.id)
            if isinstance(node, ast.Attribute) and node.attr in native_forbidden:
                native_hits.add(node.attr)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                matrix_product_lines.append(node.lineno)
    boundary = functions["project_boundary"]
    raw_forwarding = False
    for node in ast.walk(boundary):
        if isinstance(node, ast.Name) and node.id in boundary_forbidden:
            boundary_hits.add(node.id)
        if isinstance(node, ast.Attribute) and node.attr in boundary_forbidden:
            boundary_hits.add(node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "BoundaryProjection":
                keywords = [item for item in node.keywords if item.arg == "spins"]
                if len(keywords) == 1:
                    value = keywords[0].value
                    raw_forwarding = bool(
                        isinstance(value, ast.IfExp)
                        and isinstance(value.test, ast.Name)
                        and value.test.id == "valid"
                        and isinstance(value.body, ast.Name)
                        and value.body.id == "raw_spins"
                        and isinstance(value.orelse, ast.Constant)
                        and value.orelse.value is None
                    )
    passed = bool(
        not missing
        and not native_hits
        and not boundary_hits
        and imported_roots <= allowed_import_roots
        and not dynamic_import_lines
        and not uncovered_native
        and not matrix_product_lines
        and raw_forwarding
    )
    return {
        "boundary_forbidden_names": sorted(boundary_hits),
        "boundary_raw_result_forwarding": raw_forwarding,
        "dynamic_import_lines": dynamic_import_lines,
        "imported_roots": sorted(imported_roots),
        "matrix_product_lines_in_native": matrix_product_lines,
        "missing_functions": missing,
        "native_forbidden_names": sorted(native_hits),
        "pass": passed,
        "reachable_local_function_names": sorted(closure),
        "uncovered_native_function_names": uncovered_native,
    }


def independent_energy(
    state: Sequence[int], coupling: np.ndarray, field: np.ndarray
) -> float:
    total = 0.0
    for left in range(machine.SITE_COUNT):
        total -= float(field[left]) * int(state[left])
        for right in range(left + 1, machine.SITE_COUNT):
            total -= (
                float(coupling[left, right])
                * int(state[left])
                * int(state[right])
            )
    return total


def independent_oracle(
    coupling: np.ndarray, field: np.ndarray
) -> tuple[float, tuple[tuple[int, ...], ...], tuple[tuple[float, tuple[int, ...]], ...]]:
    rows = tuple(
        sorted(
            (
                independent_energy(state, coupling, field),
                tuple(int(value) for value in state),
            )
            for state in itertools.product((-1, 1), repeat=machine.SITE_COUNT)
        )
    )
    optimum = rows[0][0]
    states = tuple(state for energy, state in rows if abs(energy - optimum) <= 1.0e-12)
    return optimum, states, rows


def local_controls(
    borrowed: np.ndarray, coupling: np.ndarray, field: np.ndarray
) -> dict[str, Any]:
    nominal = machine.execute_native_cycle(borrowed, coupling, field)
    boundary = machine.project_boundary(nominal, "independent_nominal")
    transform = machine.execute_native_cycle(
        borrowed, coupling, field, transform_enabled=False
    )
    transform_boundary = machine.project_boundary(transform, "independent_transform")
    flat_geometry = altered_geometry("flat")
    flat = machine.execute_native_cycle(
        borrowed,
        coupling,
        field,
        program_beams=flat_geometry,
        actual_beams=flat_geometry,
    )
    flat_boundary = machine.project_boundary(flat, "independent_flat")
    scrambled_geometry = altered_geometry("scrambled")
    scrambled = machine.execute_native_cycle(
        borrowed,
        coupling,
        field,
        program_beams=scrambled_geometry,
        actual_beams=scrambled_geometry,
    )
    scrambled_boundary = machine.project_boundary(scrambled, "independent_scrambled")
    left, right = first_nonzero_relation(coupling)
    enabled = np.ones((machine.SITE_COUNT, machine.SITE_COUNT), dtype=np.bool_)
    np.fill_diagonal(enabled, False)
    enabled[left, right] = False
    enabled[right, left] = False
    missing = machine.execute_native_cycle(
        borrowed, coupling, field, relation_enabled=enabled
    )
    wrong_query = machine.project_boundary(
        nominal, "independent_wrong_query", query_beams=scrambled_geometry
    )
    expected = machine.as_carrier_bank(borrowed)
    restored = machine.restore_carrier(nominal)
    wrong = machine.restore_carrier(nominal, mode="wrong_phase")
    omitted = machine.restore_carrier(nominal, mode="omitted")
    flat_penalty = max(
        abs(left_value - right_value)
        for left_value, right_value in zip(
            boundary.mode_penalties, flat_boundary.mode_penalties
        )
    )
    scrambled_penalty = max(
        abs(left_value - right_value)
        for left_value, right_value in zip(
            boundary.mode_penalties, scrambled_boundary.mode_penalties
        )
    )
    checks = {
        "correct_inverse_restores": machine.maximum_abs_error(restored, expected)
        <= machine.RESTORATION_MAX,
        "flat_geometry_material": flat_penalty >= machine.MATERIALITY_MIN,
        "missing_relation_material": float(np.linalg.norm(nominal.displaced - missing.displaced))
        >= machine.MATERIALITY_MIN,
        "omitted_inverse_fails": machine.maximum_abs_error(omitted, expected)
        >= machine.WRONG_RESTORATION_MIN,
        "scrambled_geometry_material": scrambled_penalty >= machine.MATERIALITY_MIN,
        "transform_removed_material": response_delta(boundary, transform_boundary)
        >= machine.MATERIALITY_MIN,
        "wrong_inverse_fails": machine.maximum_abs_error(wrong, expected)
        >= machine.WRONG_RESTORATION_MIN,
        "wrong_query_rejected": not wrong_query.valid,
    }
    return {"checks": checks, "pass": all(checks.values())}


def verify_custody(
    freeze: dict[str, Any],
    batch: dict[str, Any],
    evidence: dict[str, Any],
    seal: dict[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    if freeze["batch_file_sha256"] != sha256_file(BATCH_FILE):
        mismatches.append("batch_file_sha256")
    if machine.machine_fingerprint() != freeze["machine_fingerprint"]:
        mismatches.append("machine_fingerprint")
    if sha256_file(EVIDENCE_FILE) != seal["evidence_sha256"]:
        mismatches.append("evidence_sha256")
    if sha256_file(TRACE_FILE) != seal["trace_sha256"]:
        mismatches.append("preoracle_trace_sha256")
    if evidence["freeze_commit"] != seal["freeze_commit"]:
        mismatches.append("freeze_commit")
    if evidence["oracle_opened"] or seal["oracle_call_count"] != 0:
        mismatches.append("oracle_order")
    if seal["energy_call_count"] != 0:
        mismatches.append("energy_order")
    identities: list[str] = []
    for index, record in enumerate(batch["ordered_instances"]):
        coupling = np.asarray(record["coupling_matrix_J"], dtype=np.float64)
        field = np.asarray(record["field_vector_h"], dtype=np.float64)
        identity = problem_sha256(coupling, field)
        if int(record["index"]) != index or identity != record["problem_sha256"]:
            mismatches.append(f"problem_identity_{index:03d}")
        if evidence["instances"][index]["problem_sha256"] != identity:
            mismatches.append(f"evidence_identity_{index:03d}")
        identities.append(identity)
    if sha256_bytes(canonical_bytes(identities)) != freeze["batch_ordered_sha256"]:
        mismatches.append("ordered_batch_sha256")
    for name, expected_sha in freeze["execution_source_sha256"].items():
        if sha256_file(PACKAGE_DIR / name) != expected_sha:
            mismatches.append(f"execution_source_{name}")
    for name, expected_sha in freeze["transitive_dependency_sha256"].items():
        if sha256_file(SUBSTRATE_DIR / name) != expected_sha:
            mismatches.append(f"transitive_dependency_{name}")
    return mismatches


def build_document() -> dict[str, Any]:
    freeze = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    batch = json.loads(BATCH_FILE.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE_FILE.read_text(encoding="utf-8"))
    seal = json.loads(SEAL_FILE.read_text(encoding="utf-8"))
    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    mismatches = verify_custody(freeze, batch, evidence, seal)
    static = static_integrity()
    if not static["pass"]:
        mismatches.append("static_integrity")
    borrowed = machine.borrowed_carrier()
    classification_counts = {
        name: 0
        for name in (
            "UNIQUE_ACCEPTED_CORRECT",
            "UNIQUE_ACCEPTED_INCORRECT",
            "UNIQUE_REJECTED_CORRECT",
            "UNIQUE_REJECTED_INCORRECT",
            "NON_UNIQUE_ACCEPTED",
            "NON_UNIQUE_REJECTED",
        )
    }
    all_energy_rows: list[dict[str, Any]] = []
    restoration_errors: list[float] = []
    reuse_errors: list[float] = []
    reuse_response_deltas: list[float] = []
    strict_pass_count = 0
    state_count = 0
    for index, (problem, sealed, published) in enumerate(
        zip(
            batch["ordered_instances"],
            evidence["instances"],
            results["outcomes"],
        )
    ):
        coupling = np.asarray(problem["coupling_matrix_J"], dtype=np.float64)
        field = np.asarray(problem["field_vector_h"], dtype=np.float64)
        execution = machine.execute_native_cycle(borrowed, coupling, field)
        boundary = machine.project_boundary(execution, f"independent_{index:03d}")
        if boundary.valid != sealed["native"]["valid"]:
            mismatches.append(f"native_valid_{index:03d}")
        if list(boundary.raw_spins) != sealed["native"]["raw_spins"]:
            mismatches.append(f"native_spins_{index:03d}")
        penalty_delta = max(
            abs(left - right)
            for left, right in zip(
                boundary.mode_penalties, sealed["native"]["mode_penalties"]
            )
        )
        if penalty_delta > 1.0e-9:
            mismatches.append(f"native_penalty_{index:03d}")
        expected = machine.as_carrier_bank(borrowed)
        restored = machine.restore_carrier(execution)
        restoration_error = machine.maximum_abs_error(restored, expected)
        reuse = machine.execute_native_cycle(restored, coupling, field)
        reuse_boundary = machine.project_boundary(reuse, f"independent_reuse_{index:03d}")
        reuse_restored = machine.restore_carrier(reuse)
        reuse_error = machine.maximum_abs_error(reuse_restored, expected)
        reuse_delta = response_delta(boundary, reuse_boundary)
        restoration_errors.append(restoration_error)
        reuse_errors.append(reuse_error)
        reuse_response_deltas.append(reuse_delta)
        if not (
            restoration_error <= machine.RESTORATION_MAX
            and reuse_error <= machine.RESTORATION_MAX
            and reuse_delta <= machine.REUSE_RESPONSE_MAX
            and boundary.valid == reuse_boundary.valid
            and (
                not boundary.valid
                or boundary.raw_spins == reuse_boundary.raw_spins
            )
        ):
            mismatches.append(f"restoration_reuse_{index:03d}")
        local = local_controls(borrowed, coupling, field)
        if local["pass"]:
            strict_pass_count += 1
        else:
            mismatches.append(f"strict_controls_{index:03d}")
        optimum_energy, optima, rows = independent_oracle(coupling, field)
        state_count += len(rows)
        raw = tuple(int(value) for value in boundary.raw_spins)
        raw_energy = independent_energy(raw, coupling, field)
        raw_correct = raw in optima
        if len(optima) != 1:
            classification = "NON_UNIQUE_ACCEPTED" if boundary.valid else "NON_UNIQUE_REJECTED"
        elif boundary.valid:
            classification = "UNIQUE_ACCEPTED_CORRECT" if raw_correct else "UNIQUE_ACCEPTED_INCORRECT"
        else:
            classification = "UNIQUE_REJECTED_CORRECT" if raw_correct else "UNIQUE_REJECTED_INCORRECT"
        classification_counts[classification] += 1
        if (
            published["classification"] != classification
            or published["raw_spins"] != list(raw)
            or abs(float(published["raw_energy"]) - raw_energy) > 1.0e-9
            or abs(float(published["optimum_energy"]) - optimum_energy) > 1.0e-9
            or published["optimum_states"] != [list(state) for state in optima]
        ):
            mismatches.append(f"oracle_outcome_{index:03d}")
        all_energy_rows.extend(
            {"energy": metric(value), "index": index, "state": list(state)}
            for value, state in rows
        )
    development_probe = next(
        record
        for record in controls.development_corpus()
        if record["label"] == "verified_primary"
    )
    nominal = machine.execute_native_cycle(
        borrowed, development_probe["coupling"], development_probe["field"]
    )
    nominal_boundary = machine.project_boundary(nominal, "geometry_probe_nominal")
    probe_checks: dict[str, bool] = {}
    for kind in ("flat", "scrambled"):
        geometry = altered_geometry(kind)
        altered = machine.execute_native_cycle(
            borrowed,
            development_probe["coupling"],
            development_probe["field"],
            program_beams=geometry,
            actual_beams=geometry,
        )
        altered_boundary = machine.project_boundary(altered, f"geometry_probe_{kind}")
        probe_checks[kind] = bool(
            not altered_boundary.valid
            or altered_boundary.raw_spins != nominal_boundary.raw_spins
        )
    if not all(probe_checks.values()):
        mismatches.append("consistent_geometry_result_probe")
    expected_counts = results["summary"]["classification_counts"]
    if classification_counts != expected_counts:
        mismatches.append("classification_counts")
    promotion_pass = bool(
        classification_counts["UNIQUE_ACCEPTED_INCORRECT"] == 0
        and classification_counts["UNIQUE_REJECTED_CORRECT"] == 0
        and classification_counts["NON_UNIQUE_ACCEPTED"] == 0
        and classification_counts["UNIQUE_ACCEPTED_CORRECT"] >= 160
        and strict_pass_count == 256
        and state_count == 8192
        and all(probe_checks.values())
        and not mismatches
    )
    verdict = "PASS" if promotion_pass else "FAIL"
    return {
        "all_state_energies_sha256": sha256_bytes(canonical_bytes(all_energy_rows)),
        "authorized_claim_ceiling": machine.CLAIM_CEILING,
        "authorized_decision": DECISION if promotion_pass else None,
        "classification_counts": classification_counts,
        "consistent_geometry_result_probe": probe_checks,
        "findings": [] if promotion_pass else sorted(set(mismatches)),
        "machine_fingerprint": freeze["machine_fingerprint"],
        "maximum_restoration_error": metric(max(restoration_errors)),
        "maximum_reuse_response_delta_l2": metric(max(reuse_response_deltas)),
        "maximum_reuse_restoration_error": metric(max(reuse_errors)),
        "mismatches": sorted(set(mismatches)),
        "reviewer_id": REVIEWER_ID,
        "schema": "catalytic_waveform_ising_v3_independent_reexecution_v2",
        "state_count_enumerated": state_count,
        "static_integrity": static,
        "strict_control_pass_count": strict_pass_count,
        "verdict": verdict,
    }


def review_bytes(document: dict[str, Any]) -> bytes:
    text = f"""# Catalytic Waveform-Ising V3 Independent Re-execution Review

Reviewer: `{document['reviewer_id']}`
Verdict: `{document['verdict']}`
Findings: `{len(document['findings'])}`

This verifier does not import the oracle adjudicator. It recomputes every batch
identity, all 8,192 scalar adjudication states, every published classification,
all 256 native waveform executions, restoration and reuse, eight strict
counterfactual controls per instance, the consistent-geometry result probe, and
the transitive native/boundary AST gate.

All-state energy hash: `{document['all_state_energies_sha256']}`

Authorized decision: `{document['authorized_decision']}`
Authorized claim ceiling: `{document['authorized_claim_ceiling']}`
"""
    return text.encode("utf-8")


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build() -> dict[str, Any]:
    document = build_document()
    write_atomic(VERIFICATION_FILE, canonical_bytes(document))
    write_atomic(REVIEW_FILE, review_bytes(document))
    return document


def verify() -> dict[str, Any]:
    document = build_document()
    if VERIFICATION_FILE.read_bytes() != canonical_bytes(document):
        raise ValueError("independent verification does not reproduce")
    if REVIEW_FILE.read_bytes() != review_bytes(document):
        raise ValueError("independent review does not reproduce")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    arguments = parser.parse_args(argv)
    document = build() if arguments.mode == "build" else verify()
    print(
        json.dumps(
            {
                "all_state_energies_sha256": document[
                    "all_state_energies_sha256"
                ],
                "state_count_enumerated": document["state_count_enumerated"],
                "strict_control_pass_count": document["strict_control_pass_count"],
                "verdict": document["verdict"],
            },
            sort_keys=True,
        )
    )
    return 0 if document["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
