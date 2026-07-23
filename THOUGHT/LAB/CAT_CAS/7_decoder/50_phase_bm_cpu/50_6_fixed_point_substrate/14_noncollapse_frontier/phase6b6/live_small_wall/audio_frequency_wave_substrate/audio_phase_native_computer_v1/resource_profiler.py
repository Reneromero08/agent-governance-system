from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
ADJUDICATOR_SOURCE = PACKAGE_DIR / "adjudicator.py"
OUTPUT_FILE = PACKAGE_DIR / "RESOURCE_RESULTS.json"
REPORT_FILE = PACKAGE_DIR / "RESOURCE_REPORT.md"
REPETITIONS = 41
SCALING_REPETITIONS = 21


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


adjudicator = load_module(ADJUDICATOR_SOURCE, "phase_native_adjudicator_resource")
engine = adjudicator.engine
suite = adjudicator.suite


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def distribution(values: Sequence[int]) -> dict[str, int]:
    return {
        "maximum_ns": max(values),
        "median_ns": int(statistics.median(values)),
        "minimum_ns": min(values),
    }


def execution_array_bytes(execution: Any) -> dict[str, int]:
    factor_bytes = sum(
        0 if step.factors is None else int(step.factors.nbytes)
        for step in execution.history
    )
    values = {
        "borrowed_carrier": int(execution.borrowed.nbytes),
        "displaced_carrier": int(execution.displaced.nbytes),
        "history_phase_factors": factor_bytes,
        "phase_trace": int(execution.phase_trace.nbytes),
        "seed_operator": int(execution.seed_operator.nbytes),
    }
    values["total_instantiated_execution_arrays"] = sum(values.values())
    return values


def benchmark_program(
    program: Any,
    values: tuple[int, ...],
    comparison: Callable[[tuple[int, ...]], tuple[int, ...]],
) -> dict[str, Any]:
    phase_input = engine.PhaseInput(values)
    borrowed = engine.borrowed_carrier(program.register_count)
    warm = engine.execute_phase_program(program, phase_input, borrowed)
    warm_boundary = engine.extract_boundary(warm)
    if warm_boundary.output_symbols != comparison(values):
        raise RuntimeError("resource benchmark output mismatch")
    execution_times: list[int] = []
    restoration_times: list[int] = []
    compile_times: list[int] = []
    comparison_times: list[int] = []
    last_execution = warm
    for _ in range(REPETITIONS):
        start = time.perf_counter_ns()
        compiled = engine.compile_program(program)
        compile_times.append(time.perf_counter_ns() - start)
        if not compiled:
            raise RuntimeError("resource benchmark program compiled empty")

        start = time.perf_counter_ns()
        execution = engine.execute_phase_program(program, phase_input, borrowed)
        boundary = engine.extract_boundary(execution)
        execution_times.append(time.perf_counter_ns() - start)
        if not boundary.valid:
            raise RuntimeError("resource benchmark boundary invalid")

        start = time.perf_counter_ns()
        restored = engine.restore_carrier(execution)
        restoration_times.append(time.perf_counter_ns() - start)
        if engine.maximum_abs_error(borrowed, restored) > engine.RESTORATION_MAX:
            raise RuntimeError("resource benchmark restoration failed")

        start = time.perf_counter_ns()
        comparison(values)
        comparison_times.append(time.perf_counter_ns() - start)
        last_execution = execution

    phase_median = int(statistics.median(execution_times))
    comparison_median = int(statistics.median(comparison_times))
    return {
        "comparison": distribution(comparison_times),
        "compiled_instruction_count": len(engine.compile_program(program)),
        "compiler": distribution(compile_times),
        "execution_arrays": execution_array_bytes(last_execution),
        "phase_execution_and_boundary": distribution(execution_times),
        "phase_to_conventional_median_ratio": metric(
            phase_median / max(1, comparison_median)
        ),
        "program_identity": engine.program_identity(program),
        "program_ir_bytes": len(
            canonical_bytes(
                {
                    "compiled": [
                        instruction.document()
                        for instruction in engine.compile_program(program)
                    ],
                    "source": program.source_document(),
                }
            )
        ),
        "restoration": distribution(restoration_times),
    }


def scaling_program(register_count: int) -> Any:
    return engine.PhaseProgram(
        name=f"resource_scale_{register_count}",
        radix=3,
        register_count=register_count,
        input_registers=(),
        output_registers=(register_count - 1,),
        statements=tuple(
            engine.PhaseInstruction("ROT", (register, 1))
            for register in range(register_count)
        ),
        computational_class="resource scaling probe",
    )


def scaling_record(register_count: int) -> dict[str, Any]:
    program = scaling_program(register_count)
    phase_input = engine.PhaseInput(())
    borrowed = engine.borrowed_carrier(register_count)
    execution_times: list[int] = []
    restoration_times: list[int] = []
    last_execution = None
    for _ in range(SCALING_REPETITIONS):
        start = time.perf_counter_ns()
        execution = engine.execute_phase_program(program, phase_input, borrowed)
        boundary = engine.extract_boundary(execution)
        execution_times.append(time.perf_counter_ns() - start)
        if not boundary.valid:
            raise RuntimeError("scaling boundary invalid")
        start = time.perf_counter_ns()
        restored = engine.restore_carrier(execution)
        restoration_times.append(time.perf_counter_ns() - start)
        if engine.maximum_abs_error(borrowed, restored) > engine.RESTORATION_MAX:
            raise RuntimeError("scaling restoration failed")
        last_execution = execution
    if last_execution is None:
        raise RuntimeError("scaling probe did not execute")
    return {
        "active_phase_components": 2 * register_count,
        "complete_configuration_modes": 0,
        "execution_arrays": execution_array_bytes(last_execution),
        "instruction_count": register_count,
        "logical_phase_state_bytes": 16 * register_count,
        "phase_execution_and_boundary": distribution(execution_times),
        "register_count": register_count,
        "restoration": distribution(restoration_times),
    }


def build_document() -> dict[str, Any]:
    programs = {program.name: program for program in suite.programs()}
    comparisons = adjudicator.comparison_functions()
    selected_inputs = {
        "prospective_affine_mod7": (5, 2),
        "prospective_binary_add3": (1, 0, 1, 1, 1, 0),
        "prospective_mux_xor_pipeline": (1, 0, 1, 1),
        "prospective_route_compose_mod7": (6, 1, 3),
    }
    benchmarks = {
        name: benchmark_program(programs[name], values, comparisons[name])
        for name, values in selected_inputs.items()
    }
    scaling = [
        scaling_record(register_count)
        for register_count in (2, 4, 8, 12, 16)
    ]
    document = {
        "benchmark_repetitions": REPETITIONS,
        "engine_fingerprint": engine.engine_fingerprint(),
        "honesty": {
            "computational_advantage_claimed": False,
            "complete_mode_materialization": False,
            "timing_is_environment_specific": True,
        },
        "program_benchmarks": benchmarks,
        "scaling": scaling,
        "scaling_law": {
            "carrier": "O(register_count * sample_count)",
            "complete_configuration_modes": 0,
            "logical_phase_state": "O(register_count)",
            "reversible_history": "O(register_count * instruction_count)",
            "time": (
                "O(register_count * sample_count * log(sample_count) "
                "+ register_count * instruction_count "
                "+ swap_instruction_count * sample_count)"
            ),
        },
        "scaling_repetitions": SCALING_REPETITIONS,
        "schema": "phase_native_computer_resource_profile_v1",
        "v3_six_site_reference": {
            "active_complete_modes": 64,
            "native_carrier_bytes": 49_152,
            "site_count": 6,
        },
    }
    document["document_sha256"] = sha256_bytes(
        canonical_bytes(
            {key: value for key, value in document.items() if key != "document_sha256"}
        )
    )
    return document


def validate_document(document: dict[str, Any]) -> None:
    if document["engine_fingerprint"] != engine.engine_fingerprint():
        raise RuntimeError("resource profile engine drift")
    if document["benchmark_repetitions"] != REPETITIONS:
        raise RuntimeError("resource repetition drift")
    scaling = document["scaling"]
    if [row["register_count"] for row in scaling] != [2, 4, 8, 12, 16]:
        raise RuntimeError("resource scaling grid drift")
    for row in scaling:
        register_count = row["register_count"]
        if row["logical_phase_state_bytes"] != 16 * register_count:
            raise RuntimeError("logical phase state byte law mismatch")
        if row["active_phase_components"] != 2 * register_count:
            raise RuntimeError("active phase component law mismatch")
        if row["complete_configuration_modes"] != 0:
            raise RuntimeError("resource profile secretly materialized complete modes")
        arrays = row["execution_arrays"]
        subtotal = sum(
            value
            for key, value in arrays.items()
            if key != "total_instantiated_execution_arrays"
        )
        if subtotal != arrays["total_instantiated_execution_arrays"]:
            raise RuntimeError("execution array byte accounting mismatch")
    expected_sha = sha256_bytes(
        canonical_bytes(
            {key: value for key, value in document.items() if key != "document_sha256"}
        )
    )
    if document["document_sha256"] != expected_sha:
        raise RuntimeError("resource profile hash mismatch")


def report_bytes(document: dict[str, Any]) -> bytes:
    lines = [
        "# Phase-native computer resources",
        "",
        "Timings are environment-specific medians. No performance advantage is claimed.",
        "",
        "## Program benchmarks",
        "",
    ]
    for name, result in document["program_benchmarks"].items():
        lines.append(
            f"- `{name}`: phase {result['phase_execution_and_boundary']['median_ns']} ns, "
            f"conventional comparison {result['comparison']['median_ns']} ns, "
            f"ratio {result['phase_to_conventional_median_ratio']:.3g}x"
        )
    lines.extend(["", "## Scaling", ""])
    for row in document["scaling"]:
        lines.append(
            f"- {row['register_count']} registers: "
            f"{row['execution_arrays']['borrowed_carrier']} carrier bytes, "
            f"{row['execution_arrays']['total_instantiated_execution_arrays']} "
            f"total execution-array bytes, "
            f"{row['phase_execution_and_boundary']['median_ns']} ns"
        )
    lines.extend(
        [
            "",
            "The logical phase state is 16 bytes per register, with two active spectral "
            "components per register and zero complete configuration modes.",
            "The retained reversible history is O(registers * instructions); it becomes "
            "quadratic only in the scaling probe because that probe deliberately uses "
            "one instruction per register.",
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
    validate_document(document)
    write_atomic(OUTPUT_FILE, canonical_bytes(document))
    write_atomic(REPORT_FILE, report_bytes(document))
    return document


def verify() -> dict[str, Any]:
    document = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    validate_document(document)
    if REPORT_FILE.read_bytes() != report_bytes(document):
        raise RuntimeError("resource report mismatch")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    document = build() if args.command == "build" else verify()
    print(
        json.dumps(
            {
                "engine_fingerprint": document["engine_fingerprint"],
                "program_count": len(document["program_benchmarks"]),
                "scaling_points": len(document["scaling"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
