from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
ENGINE_SOURCE = PACKAGE_DIR / "phase_native_engine.py"
RESULTS_FILE = PACKAGE_DIR / "DEVELOPMENT_RESULTS.json"
REPORT_FILE = PACKAGE_DIR / "DEVELOPMENT_REPORT.md"


def load_engine() -> Any:
    spec = importlib.util.spec_from_file_location("phase_native_engine", ENGINE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load phase-native engine")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = load_engine()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def reference_functions() -> dict[str, Callable[[tuple[int, ...]], tuple[int, ...]]]:
    def binary_add2(values: tuple[int, ...]) -> tuple[int, ...]:
        left = values[0] + 2 * values[1]
        right = values[2] + 2 * values[3]
        total = left + right
        return (total & 1, (total >> 1) & 1, (total >> 2) & 1)

    return {
        "affine_mod5": lambda values: ((2 * values[0] + values[1] + 3) % 5,),
        "binary_mux": lambda values: (values[2] if values[0] else values[1],),
        "binary_add2": binary_add2,
        "reverse_rotate_mod5": lambda values: (
            (values[2] + 1) % 5,
            values[1],
            values[0],
        ),
        "accumulate_mod3": lambda values: (sum(values) % 3,),
    }


def all_inputs(program: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(
        itertools.product(
            range(program.radix),
            repeat=len(program.input_registers),
        )
    )


def execute_case(
    program: Any,
    values: tuple[int, ...],
) -> tuple[dict[str, Any], Any, Any]:
    phase_input = engine.PhaseInput(values)
    borrowed = engine.borrowed_carrier(program.register_count)
    execution = engine.execute_phase_program(program, phase_input, borrowed)
    boundary = engine.extract_boundary(execution)
    restored = engine.restore_carrier(execution)
    restoration_error = engine.maximum_abs_error(borrowed, restored)
    reuse_execution = engine.execute_phase_program(program, phase_input, restored)
    reuse_boundary = engine.extract_boundary(reuse_execution)
    reuse_restored = engine.restore_carrier(reuse_execution)
    reuse_restoration_error = engine.maximum_abs_error(restored, reuse_restored)
    response_delta = engine.phase_response_delta(boundary, reuse_boundary)
    record = {
        "boundary": boundary.document(),
        "displacement_l2": metric(execution.displacement_l2),
        "input": list(values),
        "input_identity": engine.input_identity(program, phase_input),
        "instruction_count": len(execution.compiled),
        "phase_trace_sha256": sha256_bytes(
            np.asarray(execution.phase_trace, dtype="<c16").tobytes()
        ),
        "restoration_error": metric(restoration_error),
        "reuse_response_delta": metric(response_delta),
        "reuse_restoration_error": metric(reuse_restoration_error),
    }
    return record, execution, boundary


def replacement_program(program: Any, statements: tuple[Any, ...]) -> Any:
    return engine.PhaseProgram(
        name=program.name,
        radix=program.radix,
        register_count=program.register_count,
        input_registers=program.input_registers,
        output_registers=program.output_registers,
        statements=statements,
        computational_class=program.computational_class,
    )


def remove_primitive(program: Any, op: str) -> Any:
    compiled = tuple(
        instruction
        for instruction in engine.compile_program(program)
        if instruction.op != op
    )
    if not compiled:
        compiled = (engine.PhaseInstruction("ROT", (0, 1)),)
    return replacement_program(program, compiled)


def run_controls(programs: dict[str, Any]) -> dict[str, Any]:
    controls: dict[str, Any] = {}

    affine = programs["affine_mod5"]
    values = (4, 3)
    nominal = engine.extract_boundary(
        engine.execute_phase_program(
            affine,
            engine.PhaseInput(values),
            engine.borrowed_carrier(affine.register_count),
        )
    )
    no_transform = replacement_program(
        affine, (engine.PhaseInstruction("ROT", (2, 3)),)
    )
    altered = engine.extract_boundary(
        engine.execute_phase_program(
            no_transform,
            engine.PhaseInput(values),
            engine.borrowed_carrier(no_transform.register_count),
        )
    )
    controls["remove_relational_transform"] = {
        "nominal": list(nominal.output_symbols or ()),
        "altered": list(altered.output_symbols or ()),
        "passed": nominal.output_symbols != altered.output_symbols,
    }

    mux = programs["binary_mux"]
    values = (1, 0, 1)
    nominal = engine.extract_boundary(
        engine.execute_phase_program(
            mux,
            engine.PhaseInput(values),
            engine.borrowed_carrier(mux.register_count),
        )
    )
    no_ccx = remove_primitive(mux, "CCX")
    altered = engine.extract_boundary(
        engine.execute_phase_program(
            no_ccx,
            engine.PhaseInput(values),
            engine.borrowed_carrier(no_ccx.register_count),
        )
    )
    controls["remove_interference_control"] = {
        "nominal": list(nominal.output_symbols or ()),
        "altered": list(altered.output_symbols or ()),
        "passed": nominal.output_symbols != altered.output_symbols,
    }

    routed = programs["reverse_rotate_mod5"]
    values = (1, 2, 4)
    borrowed = engine.borrowed_carrier(routed.register_count)
    execution = engine.execute_phase_program(
        routed, engine.PhaseInput(values), borrowed
    )
    nominal = engine.extract_boundary(execution)
    no_swap = remove_primitive(routed, "SWAP")
    altered = engine.extract_boundary(
        engine.execute_phase_program(
            no_swap,
            engine.PhaseInput(values),
            engine.borrowed_carrier(no_swap.register_count),
        )
    )
    correct = engine.maximum_abs_error(
        borrowed, engine.restore_carrier(execution, "correct")
    )
    wrong_order = engine.maximum_abs_error(
        borrowed, engine.restore_carrier(execution, "wrong_order")
    )
    wrong_phase = engine.maximum_abs_error(
        borrowed, engine.restore_carrier(execution, "wrong_phase")
    )
    omitted = engine.maximum_abs_error(
        borrowed, engine.restore_carrier(execution, "omitted")
    )
    controls["remove_waveform_routing"] = {
        "nominal": list(nominal.output_symbols or ()),
        "altered": list(altered.output_symbols or ()),
        "passed": nominal.output_symbols != altered.output_symbols,
    }
    controls["inverse_traversal"] = {
        "correct": metric(correct),
        "omitted": metric(omitted),
        "passed": bool(
            correct <= engine.RESTORATION_MAX
            and wrong_order >= engine.WRONG_RESTORATION_MIN
            and wrong_phase >= engine.WRONG_RESTORATION_MIN
            and omitted >= engine.WRONG_RESTORATION_MIN
        ),
        "wrong_order": metric(wrong_order),
        "wrong_phase": metric(wrong_phase),
    }

    scrambled = replacement_program(
        routed,
        tuple(reversed(engine.compile_program(routed))),
    )
    altered = engine.extract_boundary(
        engine.execute_phase_program(
            scrambled,
            engine.PhaseInput(values),
            engine.borrowed_carrier(scrambled.register_count),
        )
    )
    controls["instruction_order"] = {
        "nominal": list(nominal.output_symbols or ()),
        "altered": list(altered.output_symbols or ()),
        "passed": nominal.output_symbols != altered.output_symbols,
    }

    accumulator = programs["accumulate_mod3"]
    nominal_input = engine.PhaseInput((1, 2, 2))
    nominal = engine.extract_boundary(
        engine.execute_phase_program(
            accumulator,
            nominal_input,
            engine.borrowed_carrier(accumulator.register_count),
        )
    )
    wrong_input = engine.extract_boundary(
        engine.execute_phase_program(
            accumulator,
            engine.PhaseInput((0, 0, 0)),
            engine.borrowed_carrier(accumulator.register_count),
        )
    )
    controls["input_causality"] = {
        "nominal": list(nominal.output_symbols or ()),
        "zero_input": list(wrong_input.output_symbols or ()),
        "passed": nominal.output_symbols != wrong_input.output_symbols,
    }

    adder = programs["binary_add2"]
    carry_values = (1, 1, 1, 0)
    carry_execution = engine.execute_phase_program(
        adder,
        engine.PhaseInput(carry_values),
        engine.borrowed_carrier(adder.register_count),
    )
    carry_nominal = engine.extract_boundary(carry_execution)
    without_carry_transport = replacement_program(
        adder,
        tuple(
            instruction
            for instruction in engine.compile_program(adder)
            if not (instruction.op == "ADD" and instruction.args == (5, 6))
        ),
    )
    carry_altered = engine.extract_boundary(
        engine.execute_phase_program(
            without_carry_transport,
            engine.PhaseInput(carry_values),
            engine.borrowed_carrier(without_carry_transport.register_count),
        )
    )
    controls["intermediate_phase_carry"] = {
        "altered": list(carry_altered.output_symbols or ()),
        "nominal": list(carry_nominal.output_symbols or ()),
        "passed": carry_nominal.output_symbols != carry_altered.output_symbols,
        "transported_register": 5,
        "used_by_later_register": 6,
    }

    mux_input = engine.PhaseInput((0, 1, 0))
    mux_carrier = engine.borrowed_carrier(mux.register_count)
    mux_execution = engine.execute_phase_program(mux, mux_input, mux_carrier)
    mux_boundary = engine.extract_boundary(mux_execution)
    cross_restored = engine.restore_carrier(mux_execution)
    cross_program = programs["accumulate_mod3"]
    cross_input = engine.PhaseInput((2, 1, 2))
    cross_execution = engine.execute_phase_program(
        cross_program, cross_input, cross_restored
    )
    cross_boundary = engine.extract_boundary(cross_execution)
    cross_restored_twice = engine.restore_carrier(cross_execution)
    controls["cross_program_carrier_reuse"] = {
        "first_program": mux.name,
        "first_result": list(mux_boundary.output_symbols or ()),
        "passed": bool(
            mux_boundary.valid
            and cross_boundary.valid
            and cross_boundary.output_symbols == (2,)
            and engine.maximum_abs_error(mux_carrier, cross_restored_twice)
            <= engine.RESTORATION_MAX
        ),
        "restoration_after_second_program": metric(
            engine.maximum_abs_error(mux_carrier, cross_restored_twice)
        ),
        "second_program": cross_program.name,
        "second_result": list(cross_boundary.output_symbols or ()),
    }

    boundary_before_restore = canonical_bytes(carry_nominal.document())
    restored_after_boundary = engine.restore_carrier(carry_execution)
    boundary_after_restore = canonical_bytes(carry_nominal.document())
    controls["result_outside_inverse_history"] = {
        "borrowed_restored": engine.maximum_abs_error(
            carry_execution.borrowed, restored_after_boundary
        )
        <= engine.RESTORATION_MAX,
        "passed": boundary_before_restore == boundary_after_restore,
        "result_sha256": sha256_bytes(boundary_before_restore),
    }

    carrier_rejected = False
    invalid_carrier = engine.borrowed_carrier(affine.register_count)
    invalid_spectrum = np.fft.fft(invalid_carrier, axis=1, norm="ortho")
    invalid_spectrum[:, engine.REFERENCE_BIN] = 0.0
    invalid_spectrum[:, engine.SIGNAL_BIN] = 0.0
    invalid_carrier = np.fft.ifft(invalid_spectrum, axis=1, norm="ortho")
    try:
        engine.execute_phase_program(
            affine, engine.PhaseInput((1, 2)), invalid_carrier
        )
    except ValueError as error:
        carrier_rejected = "cannot support" in str(error)
    controls["borrowed_carrier_required"] = {
        "passed": carrier_rejected,
    }

    ambiguous_execution = engine.execute_phase_program(
        affine,
        engine.PhaseInput((1, 2)),
        engine.borrowed_carrier(affine.register_count),
    )
    ambiguous_spectrum = np.fft.fft(
        ambiguous_execution.displaced, axis=1, norm="ortho"
    )
    target = affine.output_registers[0]
    ambiguous_spectrum[target, engine.SIGNAL_BIN] *= np.exp(1j * math.pi / 5.0)
    ambiguous_displaced = np.fft.ifft(
        ambiguous_spectrum, axis=1, norm="ortho"
    )
    ambiguous_execution = engine.NativeExecution(
        program=ambiguous_execution.program,
        phase_input=ambiguous_execution.phase_input,
        compiled=ambiguous_execution.compiled,
        borrowed=ambiguous_execution.borrowed,
        displaced=ambiguous_displaced,
        seed_operator=ambiguous_execution.seed_operator,
        history=ambiguous_execution.history,
        phase_trace=ambiguous_execution.phase_trace,
        displacement_l2=ambiguous_execution.displacement_l2,
    )
    ambiguous = engine.extract_boundary(ambiguous_execution)
    controls["ambiguous_phase_rejection"] = {
        "output": (
            None if ambiguous.output_symbols is None else list(ambiguous.output_symbols)
        ),
        "passed": not ambiguous.valid,
        "root_distances": [metric(value) for value in ambiguous.root_distances],
    }

    source_guard = engine.source_no_smuggle()
    controls["native_no_smuggle"] = source_guard
    controls["program_input_identity_separation"] = {
        "input_identity": engine.input_identity(
            affine, engine.PhaseInput((1, 2))
        ),
        "passed": engine.program_identity(affine)
        != engine.input_identity(affine, engine.PhaseInput((1, 2))),
        "program_identity": engine.program_identity(affine),
    }
    controls["all_passed"] = all(
        bool(value.get("passed"))
        for key, value in controls.items()
        if key != "all_passed"
    )
    return controls


def build_document() -> dict[str, Any]:
    references = reference_functions()
    programs = {program.name: program for program in engine.phase_programs()}
    if set(programs) != set(references):
        raise RuntimeError("program and independent reference sets differ")
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    all_passed = True
    for name, program in programs.items():
        program.validate()
        reference = references[name]
        program_records: list[dict[str, Any]] = []
        correct = 0
        for values in all_inputs(program):
            record, execution, boundary = execute_case(program, values)
            expected = reference(values)
            record["expected"] = list(expected)
            record["correct"] = bool(
                boundary.valid and boundary.output_symbols == expected
            )
            record["program"] = name
            record["radix"] = program.radix
            record["register_count"] = program.register_count
            record["state_changed_each_instruction"] = bool(
                all(
                    not np.allclose(
                        execution.phase_trace[index],
                        execution.phase_trace[index + 1],
                        atol=1.0e-12,
                        rtol=0.0,
                    )
                    for index in range(1, execution.phase_trace.shape[0] - 1)
                )
            )
            correct += int(record["correct"])
            program_records.append(record)
            records.append(record)
        count = len(program_records)
        program_passed = bool(
            correct == count
            and all(
                item["restoration_error"] <= engine.RESTORATION_MAX
                and item["reuse_restoration_error"] <= engine.RESTORATION_MAX
                and item["reuse_response_delta"] <= engine.REUSE_RESPONSE_MAX
                and item["displacement_l2"] >= engine.DISPLACEMENT_MIN
                for item in program_records
            )
        )
        all_passed = all_passed and program_passed
        summary[name] = {
            "computational_class": program.computational_class,
            "correct": correct,
            "input_count": count,
            "instruction_count": len(engine.compile_program(program)),
            "max_restoration_error": metric(
                max(item["restoration_error"] for item in program_records)
            ),
            "max_reuse_response_delta": metric(
                max(item["reuse_response_delta"] for item in program_records)
            ),
            "max_reuse_restoration_error": metric(
                max(item["reuse_restoration_error"] for item in program_records)
            ),
            "min_displacement_l2": metric(
                min(item["displacement_l2"] for item in program_records)
            ),
            "passed": program_passed,
            "program_identity": engine.program_identity(program),
            "radix": program.radix,
            "register_count": program.register_count,
        }
    controls = run_controls(programs)
    all_passed = all_passed and controls["all_passed"]
    state_bytes = {
        str(registers): registers * engine.SAMPLE_COUNT * 16
        for registers in sorted({program.register_count for program in programs.values()})
    }
    document = {
        "all_passed": all_passed,
        "case_count": len(records),
        "claim_ceiling": engine.CLAIM_CEILING,
        "controls": controls,
        "engine_contract": engine.engine_contract(),
        "engine_fingerprint": engine.engine_fingerprint(),
        "phase_state_bytes_by_register_count": state_bytes,
        "program_summary": summary,
        "records": records,
        "schema": "phase_native_computer_development_v1",
    }
    document["document_sha256"] = sha256_bytes(
        canonical_bytes(
            {key: value for key, value in document.items() if key != "document_sha256"}
        )
    )
    return document


def report_bytes(document: dict[str, Any]) -> bytes:
    lines = [
        "# Phase-native computer development",
        "",
        f"- result: {'PASS' if document['all_passed'] else 'FAIL'}",
        f"- shared engine fingerprint: `{document['engine_fingerprint']}`",
        f"- exhaustive development cases: {document['case_count']}",
        f"- controls: {'PASS' if document['controls']['all_passed'] else 'FAIL'}",
        "",
        "## Programs",
        "",
    ]
    for name, result in document["program_summary"].items():
        lines.append(
            f"- `{name}` ({result['computational_class']}): "
            f"{result['correct']}/{result['input_count']} correct, "
            f"restoration max {result['max_restoration_error']:.3g}"
        )
    lines.extend(
        [
            "",
            "The engine carries intermediate state only as relative spectral phase. "
            "Discrete symbols appear only at input loading and the final boundary.",
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
        raise RuntimeError("development result reproduction mismatch")
    if REPORT_FILE.read_bytes() != report_bytes(rebuilt):
        raise RuntimeError("development report reproduction mismatch")
    if not rebuilt["all_passed"]:
        raise RuntimeError("phase-native development qualification failed")
    return rebuilt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    document = build() if args.command == "build" else verify()
    print(
        json.dumps(
            {
                "all_passed": document["all_passed"],
                "case_count": document["case_count"],
                "controls_passed": document["controls"]["all_passed"],
                "engine_fingerprint": document["engine_fingerprint"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
