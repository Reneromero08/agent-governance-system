from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence


PACKAGE_DIR = Path(__file__).resolve().parent
RUNNER_SOURCE = PACKAGE_DIR / "prospective_runner.py"
SUITE_SOURCE = PACKAGE_DIR / "prospective_suite.py"
RAW_FILE = PACKAGE_DIR / "PROSPECTIVE_RAW_RESULTS.json"
SEAL_FILE = PACKAGE_DIR / "PROSPECTIVE_RAW_SEAL.json"
CONTRACT_FILE = PACKAGE_DIR / "PROSPECTIVE_CONTRACT.json"
RESULTS_FILE = PACKAGE_DIR / "FINAL_RESULTS.json"
REPORT_FILE = PACKAGE_DIR / "FINAL_REPORT.md"
DECISION = "PHASE_NATIVE_COMPUTER_REFERENCE_VERIFIED"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module(RUNNER_SOURCE, "phase_native_runner_adjudication")
suite = runner.suite
engine = runner.engine


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def affine_mod7(values: tuple[int, ...]) -> tuple[int, ...]:
    return ((3 * values[0] + 2 * values[1] + 4) % 7,)


def binary_add3(values: tuple[int, ...]) -> tuple[int, ...]:
    left = values[0] + 2 * values[1] + 4 * values[2]
    right = values[3] + 2 * values[4] + 4 * values[5]
    total = left + right
    return tuple((total >> bit) & 1 for bit in range(4))


def mux_xor_pipeline(values: tuple[int, ...]) -> tuple[int, ...]:
    selected = values[2] if values[0] else values[1]
    return (selected, selected ^ values[3])


def route_compose_mod7(values: tuple[int, ...]) -> tuple[int, ...]:
    return ((values[2] + 2) % 7, values[1], (values[0] + values[1]) % 7)


def comparison_functions() -> dict[str, Callable[[tuple[int, ...]], tuple[int, ...]]]:
    return {
        "prospective_affine_mod7": affine_mod7,
        "prospective_binary_add3": binary_add3,
        "prospective_mux_xor_pipeline": mux_xor_pipeline,
        "prospective_route_compose_mod7": route_compose_mod7,
    }


def build_document() -> dict[str, Any]:
    raw = runner.verify_raw()
    contract = runner.verify_contract()
    seal = json.loads(SEAL_FILE.read_text(encoding="utf-8"))
    functions = comparison_functions()
    if set(functions) != {program.name for program in suite.programs()}:
        raise RuntimeError("comparison program set mismatch")
    classifications: list[dict[str, Any]] = []
    per_program: dict[str, dict[str, Any]] = {}
    for record in raw["records"]:
        name = record["program"]
        values = tuple(int(value) for value in record["input"])
        expected = functions[name](values)
        observed_value = record["boundary"]["output_symbols"]
        observed = (
            None
            if observed_value is None
            else tuple(int(value) for value in observed_value)
        )
        reuse_value = record["reuse_boundary"]["output_symbols"]
        reuse = (
            None
            if reuse_value is None
            else tuple(int(value) for value in reuse_value)
        )
        exact = bool(
            record["boundary"]["valid"]
            and observed == expected
            and reuse == expected
        )
        classifications.append(
            {
                "classification": "EXACT" if exact else "INCORRECT",
                "expected": list(expected),
                "input_identity": record["input_identity"],
                "observed": None if observed is None else list(observed),
                "program": name,
                "reuse_observed": None if reuse is None else list(reuse),
            }
        )
        program_result = per_program.setdefault(
            name, {"correct": 0, "count": 0, "incorrect": 0}
        )
        program_result["count"] += 1
        program_result["correct"] += int(exact)
        program_result["incorrect"] += int(not exact)

    exact_count = sum(
        item["classification"] == "EXACT" for item in classifications
    )
    criterion = contract["promotion_criterion"]
    promotion_checks = {
        "all_raw_boundaries_valid": raw["summary"]["all_boundaries_valid"]
        == criterion["all_raw_boundaries_valid"],
        "all_results_exact_after_separate_comparison": exact_count
        == len(classifications),
        "all_restoration_and_reuse_pass": raw["summary"][
            "all_restoration_passed"
        ]
        == criterion["all_restoration_and_reuse_pass"],
        "cross_program_reuse_pass": raw["controls"]["cross_program_reuse"][
            "passed"
        ]
        == criterion["cross_program_reuse_pass"],
        "native_no_smuggle_pass": raw["controls"]["native_no_smuggle"]["passed"]
        == criterion["native_no_smuggle_pass"],
        "strict_controls_pass": raw["controls"]["all_passed"]
        == criterion["strict_controls_pass"],
        "uninterpretable_maximum": raw["summary"]["uninterpretable"]
        <= criterion["uninterpretable_maximum"],
    }
    decision = DECISION if all(promotion_checks.values()) else (
        "PHASE_NATIVE_PROGRAM_EXECUTION_PARTIAL"
    )
    document = {
        "case_count": len(classifications),
        "claim_ceiling": engine.CLAIM_CEILING,
        "classifications": classifications,
        "contract_sha256": contract["contract_sha256"],
        "decision": decision,
        "engine_fingerprint": engine.engine_fingerprint(),
        "exact_count": exact_count,
        "incorrect_count": len(classifications) - exact_count,
        "per_program": per_program,
        "pre_adjudication_raw_sha256": seal["raw_sha256"],
        "promotion_checks": promotion_checks,
        "schema": "phase_native_computer_adjudication_v1",
    }
    document["document_sha256"] = sha256_bytes(
        canonical_bytes(
            {key: value for key, value in document.items() if key != "document_sha256"}
        )
    )
    return document


def report_bytes(document: dict[str, Any]) -> bytes:
    lines = [
        "# Phase-native computer prospective result",
        "",
        f"- result: `{document['decision']}`",
        f"- engine fingerprint: `{document['engine_fingerprint']}`",
        f"- pre-adjudication raw SHA-256: `{document['pre_adjudication_raw_sha256']}`",
        f"- exact results: {document['exact_count']}/{document['case_count']}",
        f"- incorrect results: {document['incorrect_count']}",
        "",
        "## Programs",
        "",
    ]
    for name, result in document["per_program"].items():
        lines.append(
            f"- `{name}`: {result['correct']}/{result['count']} exact"
        )
    lines.extend(
        [
            "",
            "All output symbols were sealed before this comparison layer was loaded. "
            "The comparison layer did not rerun or alter native execution.",
            "",
            "The result establishes a bounded reusable software phase computer "
            "reference. It does not establish physical execution or computational "
            "advantage.",
            "",
        ]
    )
    return ("\n".join(lines)).encode("utf-8")


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def build() -> dict[str, Any]:
    document = build_document()
    write_atomic(RESULTS_FILE, canonical_bytes(document))
    write_atomic(REPORT_FILE, report_bytes(document))
    return document


def verify() -> dict[str, Any]:
    stored = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    rebuilt = build_document()
    if canonical_bytes(stored) != canonical_bytes(rebuilt):
        raise RuntimeError("adjudication reproduction mismatch")
    if REPORT_FILE.read_bytes() != report_bytes(rebuilt):
        raise RuntimeError("adjudication report reproduction mismatch")
    if rebuilt["decision"] != DECISION:
        raise RuntimeError("phase-native computer criterion did not pass")
    return rebuilt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    document = build() if args.command == "build" else verify()
    print(
        json.dumps(
            {
                "case_count": document["case_count"],
                "decision": document["decision"],
                "exact_count": document["exact_count"],
                "incorrect_count": document["incorrect_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
