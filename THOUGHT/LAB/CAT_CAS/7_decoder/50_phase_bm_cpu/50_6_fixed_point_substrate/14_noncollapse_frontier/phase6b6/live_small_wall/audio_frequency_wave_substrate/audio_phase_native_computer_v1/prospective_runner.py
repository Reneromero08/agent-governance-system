from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
ENGINE_SOURCE = PACKAGE_DIR / "phase_native_engine.py"
SUITE_SOURCE = PACKAGE_DIR / "prospective_suite.py"
CONTRACT_FILE = PACKAGE_DIR / "PROSPECTIVE_CONTRACT.json"
RAW_FILE = PACKAGE_DIR / "PROSPECTIVE_RAW_RESULTS.json"
SEAL_FILE = PACKAGE_DIR / "PROSPECTIVE_RAW_SEAL.json"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = load_module(ENGINE_SOURCE, "phase_native_engine_shared")
suite = load_module(SUITE_SOURCE, "phase_native_suite_prospective")


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def ordered_case_document() -> list[dict[str, Any]]:
    return [
        {
            "input": list(values),
            "input_identity": engine.input_identity(
                program, engine.PhaseInput(values)
            ),
            "program": program.name,
            "program_identity": engine.program_identity(program),
        }
        for program, values in suite.ordered_cases()
    ]


def build_contract_document() -> dict[str, Any]:
    phase_programs = suite.programs()
    for program in phase_programs:
        program.validate()
    case_document = ordered_case_document()
    contract = {
        "case_count": len(case_document),
        "claim_target": "PHASE_NATIVE_COMPUTER_REFERENCE_VERIFIED",
        "engine_fingerprint": engine.engine_fingerprint(),
        "engine_sha256": sha256_file(ENGINE_SOURCE),
        "execution_order": [
            "commit engine, program declarations, and complete input identities",
            "execute native phase programs without post-execution comparison code",
            "seal raw outputs, carrier restoration, reuse, and controls",
            "commit and push raw seal",
            "load separate comparison functions and adjudicate",
        ],
        "input_set_sha256": sha256_bytes(canonical_bytes(case_document)),
        "programs": [
            {
                "compiled": [
                    instruction.document()
                    for instruction in engine.compile_program(program)
                ],
                "input_count": len(suite.all_inputs(program)),
                "program_identity": engine.program_identity(program),
                "source": program.source_document(),
            }
            for program in phase_programs
        ],
        "promotion_criterion": {
            "all_raw_boundaries_valid": True,
            "all_results_exact_after_separate_comparison": True,
            "all_restoration_and_reuse_pass": True,
            "cross_program_reuse_pass": True,
            "maximum_restoration_error": engine.RESTORATION_MAX,
            "maximum_reuse_response_delta": engine.REUSE_RESPONSE_MAX,
            "native_no_smuggle_pass": True,
            "strict_controls_pass": True,
            "uninterpretable_maximum": 0,
        },
        "schema": "phase_native_computer_prospective_contract_v1",
        "suite_sha256": sha256_file(SUITE_SOURCE),
    }
    contract["contract_sha256"] = sha256_bytes(
        canonical_bytes(
            {key: value for key, value in contract.items() if key != "contract_sha256"}
        )
    )
    return contract


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def build_contract() -> dict[str, Any]:
    document = build_contract_document()
    write_atomic(CONTRACT_FILE, canonical_bytes(document))
    return document


def verify_contract() -> dict[str, Any]:
    stored = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
    rebuilt = build_contract_document()
    if canonical_bytes(stored) != canonical_bytes(rebuilt):
        raise RuntimeError("prospective contract reproduction mismatch")
    return rebuilt


def case_record(program: Any, values: tuple[int, ...]) -> dict[str, Any]:
    phase_input = engine.PhaseInput(values)
    borrowed = engine.borrowed_carrier(program.register_count)
    execution = engine.execute_phase_program(program, phase_input, borrowed)
    boundary = engine.extract_boundary(execution)
    result_bytes = canonical_bytes(boundary.document())
    restored = engine.restore_carrier(execution)
    reuse_execution = engine.execute_phase_program(program, phase_input, restored)
    reuse_boundary = engine.extract_boundary(reuse_execution)
    reuse_restored = engine.restore_carrier(reuse_execution)
    return {
        "boundary": boundary.document(),
        "displacement_l2": metric(execution.displacement_l2),
        "input": list(values),
        "input_identity": engine.input_identity(program, phase_input),
        "instruction_count": len(execution.compiled),
        "phase_trace_sha256": sha256_bytes(
            np.asarray(execution.phase_trace, dtype="<c16").tobytes()
        ),
        "program": program.name,
        "program_identity": engine.program_identity(program),
        "restoration_error": metric(
            engine.maximum_abs_error(borrowed, restored)
        ),
        "result_outside_inverse_history_sha256": sha256_bytes(result_bytes),
        "reuse_boundary": reuse_boundary.document(),
        "reuse_response_delta": metric(
            engine.phase_response_delta(boundary, reuse_boundary)
        ),
        "reuse_restoration_error": metric(
            engine.maximum_abs_error(restored, reuse_restored)
        ),
    }


def replacement_program(program: Any, instructions: tuple[Any, ...]) -> Any:
    return engine.PhaseProgram(
        name=program.name,
        radix=program.radix,
        register_count=program.register_count,
        input_registers=program.input_registers,
        output_registers=program.output_registers,
        statements=instructions,
        computational_class=program.computational_class,
    )


def raw_controls(programs: dict[str, Any]) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    routed = programs["prospective_route_compose_mod7"]
    values = (1, 3, 5)
    carrier = engine.borrowed_carrier(routed.register_count)
    execution = engine.execute_phase_program(
        routed, engine.PhaseInput(values), carrier
    )
    boundary = engine.extract_boundary(execution)
    no_route = replacement_program(
        routed,
        tuple(
            instruction
            for instruction in engine.compile_program(routed)
            if instruction.op != "SWAP"
        ),
    )
    altered = engine.extract_boundary(
        engine.execute_phase_program(
            no_route, engine.PhaseInput(values), carrier
        )
    )
    correct_error = engine.maximum_abs_error(
        carrier, engine.restore_carrier(execution, "correct")
    )
    wrong_order_error = engine.maximum_abs_error(
        carrier, engine.restore_carrier(execution, "wrong_order")
    )
    wrong_phase_error = engine.maximum_abs_error(
        carrier, engine.restore_carrier(execution, "wrong_phase")
    )
    omitted_error = engine.maximum_abs_error(
        carrier, engine.restore_carrier(execution, "omitted")
    )
    controls["routing_materiality"] = {
        "altered_output": list(altered.output_symbols or ()),
        "nominal_output": list(boundary.output_symbols or ()),
        "passed": altered.output_symbols != boundary.output_symbols,
    }
    controls["inverse_controls"] = {
        "correct": metric(correct_error),
        "omitted": metric(omitted_error),
        "passed": bool(
            correct_error <= engine.RESTORATION_MAX
            and min(wrong_order_error, wrong_phase_error, omitted_error)
            >= engine.WRONG_RESTORATION_MIN
        ),
        "wrong_order": metric(wrong_order_error),
        "wrong_phase": metric(wrong_phase_error),
    }

    pipeline = programs["prospective_mux_xor_pipeline"]
    pipeline_values = (1, 0, 1, 0)
    pipeline_carrier = engine.borrowed_carrier(pipeline.register_count)
    nominal = engine.extract_boundary(
        engine.execute_phase_program(
            pipeline, engine.PhaseInput(pipeline_values), pipeline_carrier
        )
    )
    no_control = replacement_program(
        pipeline,
        tuple(
            instruction
            for instruction in engine.compile_program(pipeline)
            if instruction.op != "CCX"
        ),
    )
    altered = engine.extract_boundary(
        engine.execute_phase_program(
            no_control,
            engine.PhaseInput(pipeline_values),
            pipeline_carrier,
        )
    )
    controls["conditional_materiality"] = {
        "altered_output": list(altered.output_symbols or ()),
        "nominal_output": list(nominal.output_symbols or ()),
        "passed": altered.output_symbols != nominal.output_symbols,
    }

    adder = programs["prospective_binary_add3"]
    carry_values = (1, 1, 1, 1, 0, 0)
    adder_carrier = engine.borrowed_carrier(adder.register_count)
    nominal = engine.extract_boundary(
        engine.execute_phase_program(
            adder, engine.PhaseInput(carry_values), adder_carrier
        )
    )
    no_second_carry = replacement_program(
        adder,
        tuple(
            instruction
            for instruction in engine.compile_program(adder)
            if not (instruction.op == "ADD" and instruction.args == (9, 10))
        ),
    )
    altered = engine.extract_boundary(
        engine.execute_phase_program(
            no_second_carry,
            engine.PhaseInput(carry_values),
            adder_carrier,
        )
    )
    controls["intermediate_phase_materiality"] = {
        "altered_output": list(altered.output_symbols or ()),
        "nominal_output": list(nominal.output_symbols or ()),
        "passed": altered.output_symbols != nominal.output_symbols,
    }

    affine = programs["prospective_affine_mod7"]
    first_carrier = engine.borrowed_carrier(affine.register_count)
    first_execution = engine.execute_phase_program(
        affine, engine.PhaseInput((5, 2)), first_carrier
    )
    first_boundary = engine.extract_boundary(first_execution)
    restored = engine.restore_carrier(first_execution)
    second_execution = engine.execute_phase_program(
        routed, engine.PhaseInput((6, 1, 3)), restored
    )
    second_boundary = engine.extract_boundary(second_execution)
    restored_again = engine.restore_carrier(second_execution)
    controls["cross_program_reuse"] = {
        "first_program": affine.name,
        "first_result": list(first_boundary.output_symbols or ()),
        "passed": bool(
            first_boundary.valid
            and second_boundary.valid
            and engine.maximum_abs_error(first_carrier, restored_again)
            <= engine.RESTORATION_MAX
        ),
        "restoration_error": metric(
            engine.maximum_abs_error(first_carrier, restored_again)
        ),
        "second_program": routed.name,
        "second_result": list(second_boundary.output_symbols or ()),
    }

    controls["native_no_smuggle"] = engine.source_no_smuggle()
    controls["all_passed"] = all(
        bool(value["passed"])
        for key, value in controls.items()
        if key != "all_passed"
    )
    return controls


def execute_raw() -> dict[str, Any]:
    contract = verify_contract()
    phase_programs = {program.name: program for program in suite.programs()}
    records = [
        case_record(program, values)
        for program, values in suite.ordered_cases()
    ]
    controls = raw_controls(phase_programs)
    document = {
        "case_count": len(records),
        "contract_sha256": contract["contract_sha256"],
        "controls": controls,
        "engine_fingerprint": engine.engine_fingerprint(),
        "records": records,
        "schema": "phase_native_computer_raw_execution_v1",
        "summary": {
            "all_boundaries_valid": all(
                record["boundary"]["valid"] for record in records
            ),
            "all_restoration_passed": all(
                record["restoration_error"] <= engine.RESTORATION_MAX
                and record["reuse_restoration_error"] <= engine.RESTORATION_MAX
                and record["reuse_response_delta"] <= engine.REUSE_RESPONSE_MAX
                for record in records
            ),
            "control_passed": controls["all_passed"],
            "max_restoration_error": metric(
                max(record["restoration_error"] for record in records)
            ),
            "max_reuse_response_delta": metric(
                max(record["reuse_response_delta"] for record in records)
            ),
            "max_reuse_restoration_error": metric(
                max(record["reuse_restoration_error"] for record in records)
            ),
            "minimum_displacement_l2": metric(
                min(record["displacement_l2"] for record in records)
            ),
            "uninterpretable": sum(
                not record["boundary"]["valid"] for record in records
            ),
        },
    }
    raw_payload = canonical_bytes(document)
    write_atomic(RAW_FILE, raw_payload)
    seal = {
        "case_count": len(records),
        "contract_sha256": contract["contract_sha256"],
        "engine_fingerprint": engine.engine_fingerprint(),
        "raw_bytes": len(raw_payload),
        "raw_sha256": sha256_bytes(raw_payload),
        "schema": "phase_native_computer_raw_seal_v1",
    }
    write_atomic(SEAL_FILE, canonical_bytes(seal))
    return document


def verify_raw() -> dict[str, Any]:
    raw_payload = RAW_FILE.read_bytes()
    document = json.loads(raw_payload)
    seal = json.loads(SEAL_FILE.read_text(encoding="utf-8"))
    contract = verify_contract()
    if seal["raw_sha256"] != sha256_bytes(raw_payload):
        raise RuntimeError("raw execution seal mismatch")
    if seal["raw_bytes"] != len(raw_payload):
        raise RuntimeError("raw execution byte count mismatch")
    if document["contract_sha256"] != contract["contract_sha256"]:
        raise RuntimeError("raw execution contract mismatch")
    if document["engine_fingerprint"] != engine.engine_fingerprint():
        raise RuntimeError("raw execution engine drift")
    if document["case_count"] != contract["case_count"]:
        raise RuntimeError("raw execution case count mismatch")
    identities = [
        {
            "input": record["input"],
            "input_identity": record["input_identity"],
            "program": record["program"],
            "program_identity": record["program_identity"],
        }
        for record in document["records"]
    ]
    if sha256_bytes(canonical_bytes(identities)) != contract["input_set_sha256"]:
        raise RuntimeError("raw execution input order mismatch")
    summary = document["summary"]
    if not (
        summary["all_boundaries_valid"]
        and summary["all_restoration_passed"]
        and summary["control_passed"]
        and summary["uninterpretable"] == 0
    ):
        raise RuntimeError("raw phase execution failed")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("contract", "verify-contract", "run", "verify-raw"),
    )
    args = parser.parse_args(argv)
    if args.command == "contract":
        document = build_contract()
        output = {
            "case_count": document["case_count"],
            "contract_sha256": document["contract_sha256"],
            "engine_fingerprint": document["engine_fingerprint"],
        }
    elif args.command == "verify-contract":
        document = verify_contract()
        output = {
            "case_count": document["case_count"],
            "contract_sha256": document["contract_sha256"],
            "engine_fingerprint": document["engine_fingerprint"],
        }
    elif args.command == "run":
        document = execute_raw()
        output = document["summary"]
    else:
        document = verify_raw()
        output = document["summary"]
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
