from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SAMPLE_RATE_HZ = 48_000
SAMPLE_COUNT = 256
REFERENCE_BIN = 7
SIGNAL_BIN = 19
RESTORATION_MAX = 2.0e-12
REUSE_RESPONSE_MAX = 2.0e-12
WRONG_RESTORATION_MIN = 1.0e-3
DISPLACEMENT_MIN = 1.0
ROOT_DISTANCE_MAX = 2.0e-10
ROOT_MARGIN_FRACTION_MIN = 0.75
MAX_REGISTERS = 16
MAX_INSTRUCTIONS = 256
MAX_RADIX = 17
CLAIM_CEILING = "BOUNDED_SOFTWARE_PHASE_NATIVE_COMPUTER_REFERENCE_ONLY"

PRIMITIVE_OPS = frozenset({"ROT", "ADD", "CCX", "SWAP"})
NATIVE_FORBIDDEN_WORDS = (
    "expected_output",
    "oracle",
    "winner",
    "correct_result",
    "reference_evaluator",
    "cached_answer",
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def unit_roots(radix: int) -> np.ndarray:
    if not 2 <= int(radix) <= MAX_RADIX:
        raise ValueError("radix outside supported phase alphabet")
    return np.exp(2j * math.pi * np.arange(radix, dtype=np.float64) / radix)


def ideal_root_margin(radix: int) -> float:
    unit_roots(radix)
    return float(1.0 - math.cos(2.0 * math.pi / radix))


@dataclass(frozen=True)
class PhaseInstruction:
    op: str
    args: tuple[int, ...]

    def validate(self, register_count: int, radix: int) -> None:
        if self.op not in PRIMITIVE_OPS:
            raise ValueError(f"unknown phase primitive {self.op!r}")
        expected_arity = {"ROT": 2, "ADD": 2, "CCX": 3, "SWAP": 2}[self.op]
        if len(self.args) != expected_arity:
            raise ValueError(f"{self.op} requires {expected_arity} operands")
        register_args = self.args[:1] if self.op == "ROT" else self.args
        for register in register_args:
            if not 0 <= int(register) < register_count:
                raise ValueError(f"{self.op} register outside program bank")
        if self.op == "ROT":
            if self.args[1] == 0:
                raise ValueError("zero phase rotation is not an instruction")
        elif len(set(self.args)) != len(self.args):
            raise ValueError(f"{self.op} requires distinct registers")
        if self.op == "CCX" and radix != 2:
            raise ValueError("CCX is the binary interference primitive")

    def document(self) -> dict[str, Any]:
        return {"args": list(self.args), "op": self.op}


@dataclass(frozen=True)
class PhaseCall:
    name: str
    registers: tuple[int, ...]

    def document(self) -> dict[str, Any]:
        return {"call": self.name, "registers": list(self.registers)}


ProgramStatement = PhaseInstruction | PhaseCall


@dataclass(frozen=True)
class PhaseProgram:
    name: str
    radix: int
    register_count: int
    input_registers: tuple[int, ...]
    output_registers: tuple[int, ...]
    statements: tuple[ProgramStatement, ...]
    computational_class: str

    def validate(self) -> None:
        unit_roots(self.radix)
        if not 1 <= self.register_count <= MAX_REGISTERS:
            raise ValueError("register count outside bounded machine")
        if not self.name or not self.computational_class:
            raise ValueError("program identity is incomplete")
        for group in (self.input_registers, self.output_registers):
            if len(set(group)) != len(group):
                raise ValueError("register interface contains duplicates")
            if any(not 0 <= item < self.register_count for item in group):
                raise ValueError("register interface outside bank")
        if not self.output_registers:
            raise ValueError("program has no output boundary")
        compiled = compile_program(self)
        if not 1 <= len(compiled) <= MAX_INSTRUCTIONS:
            raise ValueError("compiled program outside instruction bound")

    def source_document(self) -> dict[str, Any]:
        return {
            "computational_class": self.computational_class,
            "input_registers": list(self.input_registers),
            "name": self.name,
            "output_registers": list(self.output_registers),
            "radix": self.radix,
            "register_count": self.register_count,
            "statements": [statement.document() for statement in self.statements],
        }


@dataclass(frozen=True)
class PhaseInput:
    values: tuple[int, ...]

    def document(self) -> dict[str, Any]:
        return {"values": list(self.values)}


@dataclass(frozen=True)
class SubroutineTemplate:
    name: str
    parameter_count: int
    body: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]


# Each operand is ("r", parameter_index) or ("i", integer literal).
SUBROUTINES: Mapping[str, SubroutineTemplate] = {
    "DOUBLE_ADD": SubroutineTemplate(
        name="DOUBLE_ADD",
        parameter_count=2,
        body=(
            ("ADD", (("r", 0), ("r", 1))),
            ("ADD", (("r", 0), ("r", 1))),
        ),
    ),
    "MUX_INTO": SubroutineTemplate(
        name="MUX_INTO",
        parameter_count=4,
        body=(
            ("ROT", (("r", 0), ("i", 1))),
            ("CCX", (("r", 0), ("r", 1), ("r", 3))),
            ("ROT", (("r", 0), ("i", 1))),
            ("CCX", (("r", 0), ("r", 2), ("r", 3))),
        ),
    ),
    "REVERSE3": SubroutineTemplate(
        name="REVERSE3",
        parameter_count=3,
        body=(("SWAP", (("r", 0), ("r", 2))),),
    ),
    "ACCUMULATE3": SubroutineTemplate(
        name="ACCUMULATE3",
        parameter_count=4,
        body=(
            ("ADD", (("r", 0), ("r", 3))),
            ("ADD", (("r", 1), ("r", 3))),
            ("ADD", (("r", 2), ("r", 3))),
        ),
    ),
    "XOR2_INTO": SubroutineTemplate(
        name="XOR2_INTO",
        parameter_count=3,
        body=(
            ("ADD", (("r", 0), ("r", 2))),
            ("ADD", (("r", 1), ("r", 2))),
        ),
    ),
    "AND_INTO": SubroutineTemplate(
        name="AND_INTO",
        parameter_count=3,
        body=(("CCX", (("r", 0), ("r", 1), ("r", 2))),),
    ),
    "XOR3_INTO": SubroutineTemplate(
        name="XOR3_INTO",
        parameter_count=4,
        body=(
            ("ADD", (("r", 0), ("r", 3))),
            ("ADD", (("r", 1), ("r", 3))),
            ("ADD", (("r", 2), ("r", 3))),
        ),
    ),
    "MAJORITY3_INTO": SubroutineTemplate(
        name="MAJORITY3_INTO",
        parameter_count=4,
        body=(
            ("CCX", (("r", 0), ("r", 1), ("r", 3))),
            ("CCX", (("r", 0), ("r", 2), ("r", 3))),
            ("CCX", (("r", 1), ("r", 2), ("r", 3))),
        ),
    ),
}


def compile_program(program: PhaseProgram) -> tuple[PhaseInstruction, ...]:
    compiled: list[PhaseInstruction] = []
    for statement in program.statements:
        if isinstance(statement, PhaseInstruction):
            compiled.append(statement)
            continue
        template = SUBROUTINES.get(statement.name)
        if template is None:
            raise ValueError(f"unknown phase subroutine {statement.name!r}")
        if len(statement.registers) != template.parameter_count:
            raise ValueError(f"{statement.name} parameter count mismatch")
        for op, operands in template.body:
            args: list[int] = []
            for kind, value in operands:
                if kind == "r":
                    args.append(statement.registers[value])
                elif kind == "i":
                    args.append(value)
                else:
                    raise RuntimeError("invalid subroutine operand kind")
            compiled.append(PhaseInstruction(op, tuple(args)))
    if len(compiled) > MAX_INSTRUCTIONS:
        raise ValueError("compiled program exceeds instruction bound")
    for instruction in compiled:
        instruction.validate(program.register_count, program.radix)
    return tuple(compiled)


def program_identity(program: PhaseProgram) -> str:
    compiled = compile_program(program)
    return sha256_bytes(
        canonical_bytes(
            {
                "compiled": [instruction.document() for instruction in compiled],
                "source": program.source_document(),
            }
        )
    )


def input_identity(program: PhaseProgram, phase_input: PhaseInput) -> str:
    validate_input(program, phase_input)
    return sha256_bytes(
        canonical_bytes(
            {
                "input": phase_input.document(),
                "program_identity": program_identity(program),
            }
        )
    )


def validate_input(program: PhaseProgram, phase_input: PhaseInput) -> None:
    if len(phase_input.values) != len(program.input_registers):
        raise ValueError("input arity does not match phase program")
    if any(
        not isinstance(value, int) or not 0 <= value < program.radix
        for value in phase_input.values
    ):
        raise ValueError("input symbol outside program phase alphabet")


def borrowed_carrier(register_count: int) -> np.ndarray:
    if not 1 <= register_count <= MAX_REGISTERS:
        raise ValueError("register count outside carrier bank bound")
    sample_index = np.arange(SAMPLE_COUNT, dtype=np.float64)
    times = sample_index / SAMPLE_RATE_HZ
    rows: list[np.ndarray] = []
    for register in range(register_count):
        phase = (
            (0.119 + 0.009 * register) * sample_index
            + (0.19 + 0.007 * register)
            * np.sin(
                2.0 * math.pi * (37.0 + 3.0 * register) * times
                + 0.13 * register
            )
        )
        amplitude = (
            0.64
            + 0.21
            * np.cos(
                2.0 * math.pi * (23.0 + 2.0 * register) * times
                - 0.08 * register
            )
        )
        rows.append(amplitude * np.exp(1j * phase))
    carrier = np.asarray(rows, dtype=np.complex128)
    if (
        carrier.shape != (register_count, SAMPLE_COUNT)
        or not np.all(np.isfinite(carrier))
        or float(np.min(np.abs(carrier))) <= 0.0
    ):
        raise RuntimeError("borrowed phase carrier is invalid")
    return carrier


def as_carrier_bank(carrier: np.ndarray, register_count: int) -> np.ndarray:
    values = np.asarray(carrier, dtype=np.complex128)
    if values.shape != (register_count, SAMPLE_COUNT):
        raise ValueError("borrowed carrier bank has the wrong shape")
    if not np.all(np.isfinite(values)) or float(np.min(np.abs(values))) <= 0.0:
        raise ValueError("borrowed carrier bank must be finite and nonzero")
    return np.array(values, copy=True)


def seed_phase_registers(
    spectrum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.array(spectrum, dtype=np.complex128, copy=True)
    operator = np.ones_like(current)
    original = current[:, (REFERENCE_BIN, SIGNAL_BIN)]
    if float(np.min(np.abs(original))) <= 1.0e-12:
        raise ValueError("borrowed carrier cannot support the phase register bins")
    target_amplitude = np.sqrt(np.mean(np.abs(original) ** 2, axis=1))
    target = np.column_stack((target_amplitude, target_amplitude)).astype(
        np.complex128
    )
    operator[:, REFERENCE_BIN] = target[:, 0] / original[:, 0]
    operator[:, SIGNAL_BIN] = target[:, 1] / original[:, 1]
    current *= operator
    return current, operator


def phase_registers(spectrum: np.ndarray) -> np.ndarray:
    reference = np.asarray(spectrum[:, REFERENCE_BIN], dtype=np.complex128)
    signal = np.asarray(spectrum[:, SIGNAL_BIN], dtype=np.complex128)
    if float(np.min(np.abs(reference))) <= 1.0e-15:
        raise ValueError("phase reference collapsed")
    if float(np.min(np.abs(signal))) <= 1.0e-15:
        raise ValueError("phase signal collapsed")
    relation = signal * np.conjugate(reference)
    return relation / np.abs(relation)


@dataclass(frozen=True)
class NativeStep:
    op: str
    args: tuple[int, ...]
    factors: np.ndarray | None


@dataclass(frozen=True)
class NativeExecution:
    program: PhaseProgram
    phase_input: PhaseInput
    compiled: tuple[PhaseInstruction, ...]
    borrowed: np.ndarray
    displaced: np.ndarray
    seed_operator: np.ndarray
    history: tuple[NativeStep, ...]
    phase_trace: np.ndarray
    displacement_l2: float


@dataclass(frozen=True)
class BoundaryResult:
    program_name: str
    program_identity: str
    input_identity: str
    output_phasors: tuple[complex, ...]
    output_symbols: tuple[int, ...] | None
    root_distances: tuple[float, ...]
    root_margins: tuple[float, ...]
    valid: bool

    def document(self) -> dict[str, Any]:
        return {
            "input_identity": self.input_identity,
            "output_phasors": [
                {"imag": metric(value.imag), "real": metric(value.real)}
                for value in self.output_phasors
            ],
            "output_symbols": (
                None if self.output_symbols is None else list(self.output_symbols)
            ),
            "program_identity": self.program_identity,
            "program_name": self.program_name,
            "root_distances": [metric(value) for value in self.root_distances],
            "root_margins": [metric(value) for value in self.root_margins],
            "valid": self.valid,
        }


def _phase_factors(register_count: int) -> np.ndarray:
    return np.ones(register_count, dtype=np.complex128)


def _apply_factors(spectrum: np.ndarray, factors: np.ndarray) -> None:
    if factors.shape != (spectrum.shape[0],):
        raise ValueError("phase factor bank has the wrong shape")
    magnitudes = np.abs(factors)
    if not np.all(np.isfinite(factors)) or not np.allclose(
        magnitudes, 1.0, atol=2.0e-12, rtol=0.0
    ):
        raise ValueError("phase primitive produced a non-unit rotation")
    spectrum[:, SIGNAL_BIN] *= factors


def _load_input(
    spectrum: np.ndarray,
    program: PhaseProgram,
    phase_input: PhaseInput,
) -> NativeStep:
    roots = unit_roots(program.radix)
    factors = _phase_factors(program.register_count)
    for register, value in zip(program.input_registers, phase_input.values):
        factors[register] = roots[value]
    _apply_factors(spectrum, factors)
    return NativeStep("LOAD", program.input_registers, factors)


def _execute_instruction(
    spectrum: np.ndarray,
    instruction: PhaseInstruction,
    radix: int,
) -> NativeStep:
    register_count = spectrum.shape[0]
    if instruction.op == "SWAP":
        left, right = instruction.args
        spectrum[[left, right], :] = spectrum[[right, left], :]
        return NativeStep("SWAP", instruction.args, None)

    relations = phase_registers(spectrum)
    factors = _phase_factors(register_count)
    if instruction.op == "ROT":
        target, amount = instruction.args
        factors[target] = np.exp(2j * math.pi * amount / radix)
    elif instruction.op == "ADD":
        source, target = instruction.args
        factors[target] = relations[source]
    elif instruction.op == "CCX":
        control_left, control_right, target = instruction.args
        left_null = 1.0 - relations[control_left]
        right_null = 1.0 - relations[control_right]
        interference = 0.25 * left_null * right_null
        factors[target] = np.exp(1j * math.pi * float(np.real(interference)))
    else:
        raise RuntimeError("unreachable phase primitive")
    _apply_factors(spectrum, factors)
    return NativeStep(instruction.op, instruction.args, factors)


def execute_phase_program(
    program: PhaseProgram,
    phase_input: PhaseInput,
    carrier: np.ndarray,
) -> NativeExecution:
    program.validate()
    validate_input(program, phase_input)
    compiled = compile_program(program)
    borrowed = as_carrier_bank(carrier, program.register_count)
    spectrum = np.fft.fft(borrowed, axis=1, norm="ortho")
    spectrum, seed_operator = seed_phase_registers(spectrum)
    history: list[NativeStep] = []
    trace: list[np.ndarray] = [phase_registers(spectrum)]
    history.append(_load_input(spectrum, program, phase_input))
    trace.append(phase_registers(spectrum))
    for instruction in compiled:
        history.append(
            _execute_instruction(spectrum, instruction, program.radix)
        )
        trace.append(phase_registers(spectrum))
    displaced = np.fft.ifft(spectrum, axis=1, norm="ortho")
    return NativeExecution(
        program=program,
        phase_input=phase_input,
        compiled=compiled,
        borrowed=borrowed,
        displaced=displaced,
        seed_operator=seed_operator,
        history=tuple(history),
        phase_trace=np.asarray(trace, dtype=np.complex128),
        displacement_l2=float(np.linalg.norm(displaced - borrowed)),
    )


def extract_boundary(execution: NativeExecution) -> BoundaryResult:
    spectrum = np.fft.fft(execution.displaced, axis=1, norm="ortho")
    relations = phase_registers(spectrum)
    roots = unit_roots(execution.program.radix)
    phasors: list[complex] = []
    symbols: list[int] = []
    distances: list[float] = []
    margins: list[float] = []
    for register in execution.program.output_registers:
        phasor = complex(relations[register])
        root_distances = np.abs(roots - phasor)
        order = np.argsort(root_distances, kind="stable")
        best = int(order[0])
        phasors.append(phasor)
        symbols.append(best)
        distances.append(float(root_distances[best]))
        correlations = np.real(np.conjugate(roots) * phasor)
        ranked = np.sort(correlations)
        margins.append(float(ranked[-1] - ranked[-2]))
    valid = bool(
        max(distances, default=math.inf) <= ROOT_DISTANCE_MAX
        and min(margins, default=-math.inf)
        >= ROOT_MARGIN_FRACTION_MIN * ideal_root_margin(execution.program.radix)
    )
    output = tuple(symbols) if valid else None
    return BoundaryResult(
        program_name=execution.program.name,
        program_identity=program_identity(execution.program),
        input_identity=input_identity(execution.program, execution.phase_input),
        output_phasors=tuple(phasors),
        output_symbols=output,
        root_distances=tuple(distances),
        root_margins=tuple(margins),
        valid=valid,
    )


def _inverse_step(spectrum: np.ndarray, step: NativeStep) -> None:
    if step.op == "SWAP":
        left, right = step.args
        spectrum[[left, right], :] = spectrum[[right, left], :]
        return
    if step.factors is None:
        raise RuntimeError("phase step is missing its inverse factor")
    _apply_factors(spectrum, np.conjugate(step.factors))


def restore_carrier(
    execution: NativeExecution,
    mode: str = "correct",
) -> np.ndarray:
    if mode == "omitted":
        return np.array(execution.displaced, copy=True)
    spectrum = np.fft.fft(execution.displaced, axis=1, norm="ortho")
    if mode == "correct":
        steps: Iterable[NativeStep] = reversed(execution.history)
    elif mode == "wrong_order":
        steps = execution.history
    elif mode == "wrong_phase":
        corrupted: list[NativeStep] = []
        for step in reversed(execution.history):
            if step.factors is None:
                corrupted.append(step)
                continue
            wrong = np.roll(np.conjugate(step.factors), 1)
            corrupted.append(NativeStep("DIRECT", step.args, wrong))
        for step in corrupted:
            if step.op == "SWAP":
                _inverse_step(spectrum, step)
            elif step.factors is not None:
                _apply_factors(spectrum, step.factors)
        spectrum /= execution.seed_operator
        return np.fft.ifft(spectrum, axis=1, norm="ortho")
    else:
        raise ValueError("unknown restoration mode")
    for step in steps:
        _inverse_step(spectrum, step)
    spectrum /= execution.seed_operator
    return np.fft.ifft(spectrum, axis=1, norm="ortho")


def maximum_abs_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.complex128)
                - np.asarray(right, dtype=np.complex128)
            )
        )
    )


def phase_response_delta(left: BoundaryResult, right: BoundaryResult) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(left.output_phasors, dtype=np.complex128)
                - np.asarray(right.output_phasors, dtype=np.complex128)
            )
        )
    )


def source_no_smuggle() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    native_start = source.index("def _execute_instruction(")
    native_end = source.index("def extract_boundary(")
    native = source[native_start:native_end].lower()
    found = [word for word in NATIVE_FORBIDDEN_WORDS if word in native]
    forbidden_selection = [
        word for word in ("argmax", "argsort", "round(", "sign(", "energy")
        if word in native
    ]
    return {
        "forbidden_answer_words": found,
        "forbidden_native_selection": forbidden_selection,
        "native_receives_only": [
            "phase program",
            "phase input symbols",
            "borrowed complex carrier",
        ],
        "passed": not found and not forbidden_selection,
    }


def engine_contract() -> dict[str, Any]:
    return {
        "claim_ceiling": CLAIM_CEILING,
        "execution_environment": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
        "factorization": {
            "active_phase_components_per_register": 2,
            "complete_configuration_modes": 0,
            "logical_phase_state_growth": "O(register_count)",
            "sample_count": SAMPLE_COUNT,
        },
        "instruction_set": sorted(PRIMITIVE_OPS),
        "root_acceptance": {
            "distance_maximum": ROOT_DISTANCE_MAX,
            "ideal_neighbor_margin_fraction_minimum": ROOT_MARGIN_FRACTION_MIN,
            "margin_law": (
                "fraction * (1 - cos(2*pi/radix)); derived before output decoding"
            ),
        },
        "native_state": (
            "relative S1 phase between reference and signal spectral components"
        ),
        "restoration": (
            "reverse instruction history, reverse input phase load, divide seed operator"
        ),
        "subroutines": sorted(SUBROUTINES),
    }


def engine_fingerprint() -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "contract": engine_contract(),
                "source_sha256": sha256_bytes(Path(__file__).read_bytes()),
            }
        )
    )


def phase_programs() -> tuple[PhaseProgram, ...]:
    return (
        PhaseProgram(
            name="affine_mod5",
            radix=5,
            register_count=3,
            input_registers=(0, 1),
            output_registers=(2,),
            statements=(
                PhaseCall("DOUBLE_ADD", (0, 2)),
                PhaseInstruction("ADD", (1, 2)),
                PhaseInstruction("ROT", (2, 3)),
            ),
            computational_class="modular arithmetic",
        ),
        PhaseProgram(
            name="binary_mux",
            radix=2,
            register_count=4,
            input_registers=(0, 1, 2),
            output_registers=(3,),
            statements=(PhaseCall("MUX_INTO", (0, 1, 2, 3)),),
            computational_class="phase-conditioned control",
        ),
        PhaseProgram(
            name="binary_add2",
            radix=2,
            register_count=8,
            input_registers=(0, 1, 2, 3),
            output_registers=(4, 6, 7),
            statements=(
                PhaseCall("XOR2_INTO", (0, 2, 4)),
                PhaseCall("AND_INTO", (0, 2, 5)),
                PhaseCall("XOR3_INTO", (1, 3, 5, 6)),
                PhaseCall("MAJORITY3_INTO", (1, 3, 5, 7)),
            ),
            computational_class="multi-stage binary arithmetic",
        ),
        PhaseProgram(
            name="reverse_rotate_mod5",
            radix=5,
            register_count=3,
            input_registers=(0, 1, 2),
            output_registers=(0, 1, 2),
            statements=(
                PhaseCall("REVERSE3", (0, 1, 2)),
                PhaseInstruction("ROT", (0, 1)),
            ),
            computational_class="sequence transformation and routing",
        ),
        PhaseProgram(
            name="accumulate_mod3",
            radix=3,
            register_count=4,
            input_registers=(0, 1, 2),
            output_registers=(3,),
            statements=(PhaseCall("ACCUMULATE3", (0, 1, 2, 3)),),
            computational_class="finite-state accumulation",
        ),
    )


if not source_no_smuggle()["passed"]:
    raise RuntimeError("native phase engine contains a forbidden answer path")
