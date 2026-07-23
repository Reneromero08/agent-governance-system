from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
ENGINE_SOURCE = PACKAGE_DIR / "phase_native_engine.py"
SUITE_SOURCE = PACKAGE_DIR / "prospective_suite.py"
DEVELOPMENT_SOURCE = PACKAGE_DIR / "development_qualifier.py"
RUNNER_SOURCE = PACKAGE_DIR / "prospective_runner.py"
ADJUDICATOR_SOURCE = PACKAGE_DIR / "adjudicator.py"
RESOURCE_SOURCE = PACKAGE_DIR / "resource_profiler.py"
RAW_FILE = PACKAGE_DIR / "PROSPECTIVE_RAW_RESULTS.json"
SEAL_FILE = PACKAGE_DIR / "PROSPECTIVE_RAW_SEAL.json"
CONTRACT_FILE = PACKAGE_DIR / "PROSPECTIVE_CONTRACT.json"
RESULTS_FILE = PACKAGE_DIR / "FINAL_RESULTS.json"
OUTPUT_FILE = PACKAGE_DIR / "INDEPENDENT_VERIFICATION.json"
REVIEW_FILE = PACKAGE_DIR / "INDEPENDENT_MECHANICAL_REVIEW.md"
PRE_ADJUDICATION_COMMIT = "6e9112f78ef5b18dfa4ed8c646b80202a68a2d4b"
START_COMMIT = "3ea54416dd42b6dec3cc4d43af52fc774c91fd1e"
REVIEWER_ID = "PHASE-COMPUTER-INDEPENDENT-REEXECUTION-01"


def repository_root() -> Path:
    for parent in (PACKAGE_DIR, *PACKAGE_DIR.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("cannot locate worktree root")


REPOSITORY_ROOT = repository_root()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = load_module(ENGINE_SOURCE, "phase_native_engine_independent")
sys.modules["phase_native_engine_shared"] = engine
suite = load_module(SUITE_SOURCE, "phase_native_suite_independent")


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def independent_functions() -> dict[str, Callable[[tuple[int, ...]], tuple[int, ...]]]:
    def affine(values: tuple[int, ...]) -> tuple[int, ...]:
        return ((3 * values[0] + 2 * values[1] + 4) % 7,)

    def add3(values: tuple[int, ...]) -> tuple[int, ...]:
        total = (
            values[0]
            + 2 * values[1]
            + 4 * values[2]
            + values[3]
            + 2 * values[4]
            + 4 * values[5]
        )
        return tuple((total // (1 << bit)) % 2 for bit in range(4))

    def pipeline(values: tuple[int, ...]) -> tuple[int, ...]:
        chosen = values[1] + values[0] * (values[2] - values[1])
        return (chosen, (chosen + values[3]) % 2)

    def routed(values: tuple[int, ...]) -> tuple[int, ...]:
        return ((values[2] + 2) % 7, values[1], (values[0] + values[1]) % 7)

    return {
        "prospective_affine_mod7": affine,
        "prospective_binary_add3": add3,
        "prospective_mux_xor_pipeline": pipeline,
        "prospective_route_compose_mod7": routed,
    }


def git_text(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def pre_adjudication_order_proof() -> dict[str, Any]:
    raw_relative = RAW_FILE.relative_to(REPOSITORY_ROOT).as_posix()
    adjudicator_relative = ADJUDICATOR_SOURCE.relative_to(
        REPOSITORY_ROOT
    ).as_posix()
    committed_raw = subprocess.run(
        ["git", "cat-file", "-e", f"{PRE_ADJUDICATION_COMMIT}:{raw_relative}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
    ).returncode == 0
    committed_adjudicator = subprocess.run(
        [
            "git",
            "cat-file",
            "-e",
            f"{PRE_ADJUDICATION_COMMIT}:{adjudicator_relative}",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
    ).returncode == 0
    return {
        "adjudicator_absent": not committed_adjudicator,
        "passed": committed_raw and not committed_adjudicator,
        "pre_adjudication_commit": PRE_ADJUDICATION_COMMIT,
        "raw_present": committed_raw,
    }


def static_no_smuggle() -> dict[str, Any]:
    source = ENGINE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            if any(
                marker in name
                for marker in (
                    "adjudicator",
                    "development_qualifier",
                    "resource_profiler",
                    "prospective_runner",
                )
            ):
                forbidden_imports.append(name)

    native_functions = {
        "compile_program",
        "seed_phase_registers",
        "phase_registers",
        "_load_input",
        "_execute_instruction",
        "execute_phase_program",
    }
    forbidden_calls: list[str] = []
    forbidden_names: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in native_functions:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    call = child.func.id
                elif isinstance(child.func, ast.Attribute):
                    call = child.func.attr
                else:
                    call = ""
                if call in {
                    "argmax",
                    "argsort",
                    "round",
                    "sign",
                    "comparison_functions",
                }:
                    forbidden_calls.append(f"{node.name}:{call}")
            if isinstance(child, ast.Name) and child.id.lower() in {
                "expected",
                "answer",
                "winner",
                "oracle",
            }:
                forbidden_names.append(f"{node.name}:{child.id}")
    source_guard = engine.source_no_smuggle()
    return {
        "engine_source_guard": source_guard,
        "forbidden_calls": forbidden_calls,
        "forbidden_imports": forbidden_imports,
        "forbidden_names": forbidden_names,
        "passed": bool(
            source_guard["passed"]
            and not forbidden_calls
            and not forbidden_imports
            and not forbidden_names
        ),
    }


def changed_path_scope() -> dict[str, Any]:
    names = [
        line
        for line in git_text("diff", "--name-only", START_COMMIT, "--").splitlines()
        if line
    ]
    package_relative = PACKAGE_DIR.relative_to(REPOSITORY_ROOT).as_posix()
    outside = [
        name
        for name in names
        if not name.startswith(package_relative + "/")
    ]
    return {
        "changed_path_count": len(names),
        "outside_package": outside,
        "passed": not outside,
        "start_commit": START_COMMIT,
    }


def reexecute() -> dict[str, Any]:
    raw_payload = RAW_FILE.read_bytes()
    raw = json.loads(raw_payload)
    seal = json.loads(SEAL_FILE.read_text(encoding="utf-8"))
    if sha256_bytes(raw_payload) != seal["raw_sha256"]:
        raise RuntimeError("independent raw seal mismatch")
    functions = independent_functions()
    cases = suite.ordered_cases()
    if len(cases) != len(raw["records"]):
        raise RuntimeError("independent case count mismatch")
    exact = 0
    byte_exact_boundaries = 0
    restoration_passed = 0
    reuse_passed = 0
    max_restoration = 0.0
    max_reuse_restoration = 0.0
    max_reuse_delta = 0.0
    per_program: dict[str, dict[str, int]] = {}
    for (program, values), stored in zip(cases, raw["records"]):
        phase_input = engine.PhaseInput(values)
        borrowed = engine.borrowed_carrier(program.register_count)
        execution = engine.execute_phase_program(program, phase_input, borrowed)
        boundary = engine.extract_boundary(execution)
        expected = functions[program.name](values)
        correct = bool(
            boundary.valid and boundary.output_symbols == expected
        )
        exact += int(correct)
        byte_exact_boundaries += int(
            canonical_bytes(boundary.document())
            == canonical_bytes(stored["boundary"])
        )
        trace_sha = sha256_bytes(
            np.asarray(execution.phase_trace, dtype="<c16").tobytes()
        )
        if trace_sha != stored["phase_trace_sha256"]:
            raise RuntimeError("independent phase trace mismatch")
        restored = engine.restore_carrier(execution)
        restoration_error = engine.maximum_abs_error(borrowed, restored)
        max_restoration = max(max_restoration, restoration_error)
        restoration_passed += int(
            restoration_error <= engine.RESTORATION_MAX
        )
        reuse_execution = engine.execute_phase_program(
            program, phase_input, restored
        )
        reuse_boundary = engine.extract_boundary(reuse_execution)
        reuse_restored = engine.restore_carrier(reuse_execution)
        reuse_restoration = engine.maximum_abs_error(restored, reuse_restored)
        reuse_delta = engine.phase_response_delta(boundary, reuse_boundary)
        max_reuse_restoration = max(
            max_reuse_restoration, reuse_restoration
        )
        max_reuse_delta = max(max_reuse_delta, reuse_delta)
        reuse_passed += int(
            reuse_boundary.output_symbols == expected
            and reuse_restoration <= engine.RESTORATION_MAX
            and reuse_delta <= engine.REUSE_RESPONSE_MAX
        )
        result = per_program.setdefault(
            program.name, {"correct": 0, "count": 0}
        )
        result["count"] += 1
        result["correct"] += int(correct)
    count = len(cases)
    return {
        "boundary_byte_exact": byte_exact_boundaries,
        "case_count": count,
        "exact": exact,
        "max_restoration_error": float(f"{max_restoration:.12g}"),
        "max_reuse_response_delta": float(f"{max_reuse_delta:.12g}"),
        "max_reuse_restoration_error": float(
            f"{max_reuse_restoration:.12g}"
        ),
        "passed": bool(
            exact == count
            and byte_exact_boundaries == count
            and restoration_passed == count
            and reuse_passed == count
        ),
        "per_program": per_program,
        "restoration_passed": restoration_passed,
        "reuse_passed": reuse_passed,
    }


def build_document() -> dict[str, Any]:
    final = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
    reexecution = reexecute()
    static = static_no_smuggle()
    order = pre_adjudication_order_proof()
    scope = changed_path_scope()
    checks = {
        "adjudication_decision": final["decision"]
        == "PHASE_NATIVE_COMPUTER_REFERENCE_VERIFIED",
        "changed_path_scope": scope["passed"],
        "engine_fingerprint": final["engine_fingerprint"]
        == engine.engine_fingerprint()
        == contract["engine_fingerprint"],
        "pre_adjudication_order": order["passed"],
        "reexecution": reexecution["passed"],
        "static_no_smuggle": static["passed"],
    }
    document = {
        "checks": checks,
        "engine_fingerprint": engine.engine_fingerprint(),
        "finding_count": 0 if all(checks.values()) else 1,
        "order_proof": order,
        "reexecution": reexecution,
        "reviewer_id": REVIEWER_ID,
        "schema": "phase_native_computer_independent_verification_v1",
        "scope": scope,
        "static_no_smuggle": static,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }
    document["document_sha256"] = sha256_bytes(
        canonical_bytes(
            {key: value for key, value in document.items() if key != "document_sha256"}
        )
    )
    return document


def review_bytes(document: dict[str, Any]) -> bytes:
    lines = [
        "# Independent mechanical review",
        "",
        f"- reviewer: `{document['reviewer_id']}`",
        f"- verdict: **{document['verdict']}**",
        f"- findings: {document['finding_count']}",
        f"- independent re-execution: "
        f"{document['reexecution']['exact']}/{document['reexecution']['case_count']}",
        f"- restoration: "
        f"{document['reexecution']['restoration_passed']}/"
        f"{document['reexecution']['case_count']}",
        f"- reuse: "
        f"{document['reexecution']['reuse_passed']}/"
        f"{document['reexecution']['case_count']}",
        "",
        "The verifier independently re-executed every prospective program/input, "
        "used separately written comparison functions, checked raw byte custody and "
        "pre-adjudication Git order, inspected the native source, and confirmed that "
        "all changes remain inside the new package.",
        "",
    ]
    return ("\n".join(lines)).encode("utf-8")


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def build() -> dict[str, Any]:
    document = build_document()
    write_atomic(OUTPUT_FILE, canonical_bytes(document))
    write_atomic(REVIEW_FILE, review_bytes(document))
    return document


def verify() -> dict[str, Any]:
    stored = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    rebuilt = build_document()
    if canonical_bytes(stored) != canonical_bytes(rebuilt):
        raise RuntimeError("independent verification reproduction mismatch")
    if REVIEW_FILE.read_bytes() != review_bytes(rebuilt):
        raise RuntimeError("independent mechanical review mismatch")
    if rebuilt["verdict"] != "PASS":
        raise RuntimeError("independent mechanical verification failed")
    return rebuilt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    document = build() if args.command == "build" else verify()
    print(
        json.dumps(
            {
                "case_count": document["reexecution"]["case_count"],
                "finding_count": document["finding_count"],
                "reviewer_id": document["reviewer_id"],
                "verdict": document["verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
