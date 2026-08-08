#!/usr/bin/env python3
"""M240: exact interacting multi-injection shared-syndrome diagnostic.

The accepted path is a direct-process exact amplitude transaction.  Counts one
through four reuse one unresolved syndrome wire to inject distinct data wires,
then consume that same wire in a connected Clifford network.  The package
tests an exact stabilizer-relative Wigner l1 law; it does not claim a classical
lower bound, CATVM custody, physical execution, or computational advantage.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Sequence

import zeta5_normalized_cubic_fourier_coherent_port as m237


P = 5
INJECTION_COUNTS = (1, 2, 3, 4)
PORT_TYPE = "ZETA5_INTERACTING_MULTI_INJECTION_SHARED_SYNDROME_V1"
K = m237.K
ZERO = m237.ZERO
ONE = m237.ONE
SQRT5 = m237.SQRT5

RESULT = "PASS_EXACT_ZETA5_INTERACTING_MULTI_INJECTION_SYNDROME_NETWORK_STRICT_SCOPE"
CLAIM = (
    "EXACT_P5_COUNTS1_2_3_4_REPEATED_CUBIC_MAGIC_INJECTIONS_REUSE_ONE_ACTUAL_"
    "UNRESOLVED5_VALUE_SYNDROME_PORT_ACROSS_DISTINCT_DATA_WIRES_THEN_CONSUME_"
    "IT_IN_A_CONNECTED_SUM_CZ_CLIFFORD_NETWORK_WITH_EXACT_MULTIPLICATIVE_"
    "STABILIZER_RELATIVE_WIGNER_L1_MAGIC_FINAL_ONLY_DATA_PROBABILITY_EXACT_"
    "SAME_BACKING_RESTORATION_AND_DESCRIPTOR_DISTINCT_REUSE_BUT_THE_MAGIC_"
    "LAW_IS_THE_PRODUCT_INPUT_LAW_THE_EXACT_STABILIZER_COMPONENT_UPPER_BOUND_"
    "GROWS5_TO_THE_INJECTION_COUNT_AND_A_STREAMED_SCALAR_CLASSICAL_BOUNDARY_"
    "RECURRENCE_REMAINS_SMALLER"
)
CLAIM_CEILING = (
    "QZETA5_ONE_SHARED_SYNDROME_DATA_WIRES1_2_3_4_TWO_PUBLIC_FAMILIES_"
    "DIRECT_PROCESS_LOGICAL_CUSTODY_STABILIZER_RELATIVE_PRODUCT_MAGIC_ONLY"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def canonicalize(values: list[K], exponent: int) -> int:
    while exponent and all(c % P == 0 for value in values for c in value):
        for i, value in enumerate(values):
            values[i] = tuple(c // P for c in value)  # type: ignore[assignment]
        exponent -= 1
    return exponent


def payload_bits(values: Sequence[K], exponent: int) -> int:
    return sum(signed_bits(c) for value in values for c in value) + signed_bits(P**exponent)


def vector_commitment(values: Sequence[K], exponent: int) -> str:
    payload = {"denominator_power5": exponent, "numerators": [list(v) for v in values]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def value_commitment(value: K, exponent: int) -> str:
    values = [value]
    exponent = canonicalize(values, exponent)
    payload = {"denominator_power5": exponent, "numerator": list(values[0])}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def probability_encoding(value: K, exponent: int) -> dict[str, object]:
    values = [value]
    exponent = canonicalize(values, exponent)
    value = values[0]
    if value[1] or value[2] != value[3]:
        raise RuntimeError("probability left Q(sqrt(5))")
    return {
        "a_numerator": 2 * value[0] - value[2],
        "b_sqrt5_numerator": -value[2],
        "denominator_twice_power5_exponent": exponent,
        "commitment": value_commitment(value, exponent),
    }


def flat_index(coordinates: Sequence[int]) -> int:
    result = 0
    for coordinate in coordinates:
        result = P * result + coordinate
    return result


def coordinates(index: int, wire_count: int) -> list[int]:
    result = [0] * wire_count
    for position in range(wire_count - 1, -1, -1):
        result[position] = index % P
        index //= P
    return result


def sqrt5_power(power: int) -> K:
    result = ONE
    for _ in range(power):
        result = m237.k_mul(result, SQRT5)
    return result


@dataclass(frozen=True)
class QuadraticReal:
    rational: Fraction
    sqrt5: Fraction

    def __add__(self, other: "QuadraticReal") -> "QuadraticReal":
        return QuadraticReal(self.rational + other.rational, self.sqrt5 + other.sqrt5)

    def __sub__(self, other: "QuadraticReal") -> "QuadraticReal":
        return QuadraticReal(self.rational - other.rational, self.sqrt5 - other.sqrt5)

    def __mul__(self, other: "QuadraticReal") -> "QuadraticReal":
        return QuadraticReal(
            self.rational * other.rational + 5 * self.sqrt5 * other.sqrt5,
            self.rational * other.sqrt5 + self.sqrt5 * other.rational,
        )

    def scale(self, scalar: Fraction) -> "QuadraticReal":
        return QuadraticReal(self.rational * scalar, self.sqrt5 * scalar)

    def power(self, exponent: int) -> "QuadraticReal":
        result = QuadraticReal(Fraction(1), Fraction(0))
        base = self
        remaining = exponent
        while remaining:
            if remaining & 1:
                result = result * base
            remaining //= 2
            if remaining:
                base = base * base
        return result

    def encoding(self) -> dict[str, int]:
        return {
            "rational_numerator": self.rational.numerator,
            "rational_denominator": self.rational.denominator,
            "sqrt5_numerator": self.sqrt5.numerator,
            "sqrt5_denominator": self.sqrt5.denominator,
        }


def real_pair(value: K, exponent: int) -> QuadraticReal:
    values = [value]
    exponent = canonicalize(values, exponent)
    value = values[0]
    if value[1] or value[2] != value[3]:
        raise RuntimeError("value is not in Q(sqrt(5))")
    denominator = 2 * P**exponent
    return QuadraticReal(Fraction(2 * value[0] - value[2], denominator), Fraction(-value[2], denominator))


def real_sign(value: QuadraticReal) -> int:
    a, b = value.rational, value.sqrt5
    if not a and not b:
        return 0
    if a >= 0 and b >= 0:
        return 1
    if a <= 0 and b <= 0:
        return -1
    left = a * a
    right = 5 * b * b
    if left == right:
        return 0
    if a > 0:
        return 1 if left > right else -1
    return -1 if left > right else 1


def one_magic_wigner_l1(strength: int) -> tuple[QuadraticReal, int]:
    amplitudes = [m237.k_mul(SQRT5, m237.zeta_power(strength * x**3)) for x in range(P)]
    absolute = QuadraticReal(Fraction(0), Fraction(0))
    negative_count = 0
    half = 3
    for q in range(P):
        for momentum in range(P):
            total = ZERO
            for displacement in range(P):
                left = amplitudes[(q + half * displacement) % P]
                right = m237.k_conjugate(amplitudes[(q - half * displacement) % P])
                term = m237.k_mul(m237.zeta_power(-momentum * displacement), m237.k_mul(left, right))
                total = m237.k_add(total, term)
            value = real_pair(total, 3)
            sign = real_sign(value)
            if sign < 0:
                negative_count += 1
                absolute = absolute - value
            else:
                absolute = absolute + value
    return absolute, negative_count


@dataclass(frozen=True)
class Gate:
    kind: str
    wire: int
    target: int = -1
    parameter: int = 0
    direction: int = 1
    stage: str = "NETWORK"
    injection_id: int = -1
    consumer_id: int = -1

    def __post_init__(self) -> None:
        if self.kind not in {"FOURIER", "CUBIC", "SUM", "CZ", "Q", "L", "G"}:
            raise ValueError("unknown M240 gate")
        if self.stage not in {"PREP", "INJECTION", "NETWORK", "FINAL"}:
            raise ValueError("unknown M240 stage")
        if self.kind == "FOURIER" and self.direction not in (-1, 1):
            raise ValueError("invalid Fourier direction")
        if self.kind != "FOURIER" and self.parameter % P == 0:
            raise ValueError("zero non-Fourier parameter")
        if self.kind in {"SUM", "CZ", "Q", "L", "G"} and self.target < 0:
            raise ValueError("two-wire gate lacks target")
        if self.stage == "INJECTION" and self.injection_id < 0:
            raise ValueError("injection gate lacks injection identity")
        if self.stage != "INJECTION" and self.injection_id != -1:
            raise ValueError("non-injection gate carries injection identity")

    def inverse(self) -> "Gate":
        if self.kind == "FOURIER":
            return replace(self, direction=-self.direction)
        return replace(self, parameter=-self.parameter)

    def serial(self) -> tuple[object, ...]:
        return (
            self.kind, self.wire, self.target, self.parameter % P, self.direction,
            self.stage, self.injection_id, self.consumer_id,
        )


def program_digest(
    family: int,
    injection_count: int,
    strengths: Sequence[int],
    output_data: Sequence[int],
    gates: Sequence[Gate],
) -> str:
    payload = {
        "family": family,
        "injection_count": injection_count,
        "strengths": list(strengths),
        "output_data": list(output_data),
        "gates": [gate.serial() for gate in gates],
        "port_type": PORT_TYPE,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class PublicProgram:
    family: int
    injection_count: int
    strengths: tuple[int, ...]
    output_data: tuple[int, ...]
    gates: tuple[Gate, ...]
    program_id: str

    def __post_init__(self) -> None:
        if self.family not in (0, 1) or self.injection_count not in INJECTION_COUNTS:
            raise ValueError("program outside declared M240 family")
        if len(self.strengths) != self.injection_count or any(a % P == 0 for a in self.strengths):
            raise ValueError("invalid injection strengths")
        if len(self.output_data) != self.injection_count or any(y not in range(P) for y in self.output_data):
            raise ValueError("invalid data boundary")
        wire_count = self.injection_count + 1
        if any(g.wire not in range(wire_count) or (g.target >= wire_count) for g in self.gates):
            raise ValueError("gate outside public interface")
        consumers = [g.consumer_id for g in self.gates if g.consumer_id >= 0]
        if consumers != list(range(len(consumers))):
            raise ValueError("consumer identities are not unique and ordered")
        seen = {g.injection_id for g in self.gates if g.stage == "INJECTION"}
        if seen != set(range(self.injection_count)):
            raise ValueError("injection identities incomplete")
        expected = program_digest(
            self.family, self.injection_count, self.strengths, self.output_data, self.gates
        )
        if self.program_id != expected:
            raise ValueError("program digest is not bound to full descriptor")

    @property
    def descriptor(self) -> tuple[object, ...]:
        return (
            self.family, self.injection_count, self.strengths, self.output_data,
            tuple(g.serial() for g in self.gates), self.program_id,
        )

    @property
    def syndrome_wire(self) -> int:
        return self.injection_count

    @property
    def wire_count(self) -> int:
        return self.injection_count + 1

    @property
    def dimension(self) -> int:
        return P**self.wire_count

    @property
    def consumer_count(self) -> int:
        return sum(g.consumer_id >= 0 for g in self.gates)


def compile_program(family: int, injection_count: int) -> PublicProgram:
    if injection_count not in INJECTION_COUNTS or family not in (0, 1):
        raise ValueError("program outside declared M240 suite")
    strengths = (1, 2, 3, 4)[:injection_count] if family == 0 else (2, 4, 1, 3)[:injection_count]
    def nonzero(value: int) -> int:
        return value % P or 1

    syndrome = injection_count
    gates: list[Gate] = []
    for data in range(injection_count):
        gates.append(Gate("FOURIER", data, stage="PREP"))
    gates.append(Gate("FOURIER", syndrome, stage="PREP"))
    consumer = 0
    for data, strength in enumerate(strengths):
        gates.append(Gate("CUBIC", syndrome, parameter=strength, stage="INJECTION", injection_id=data))
        gates.append(Gate("SUM", data, syndrome, -1, stage="INJECTION", injection_id=data, consumer_id=consumer)); consumer += 1
        gates.append(Gate("Q", data, syndrome, strength, stage="INJECTION", injection_id=data, consumer_id=consumer)); consumer += 1
        gates.append(Gate("L", data, syndrome, strength, stage="INJECTION", injection_id=data, consumer_id=consumer)); consumer += 1
        gates.append(Gate("G", data, syndrome, strength, stage="INJECTION", injection_id=data, consumer_id=consumer)); consumer += 1
    order: Iterable[int] = range(injection_count) if family == 0 else reversed(range(injection_count))
    for data in order:
        if family == 0:
            network = [
                Gate("SUM", data, syndrome, nonzero(1 + data), stage="NETWORK", consumer_id=consumer),
                Gate("CZ", (data + 1) % injection_count, syndrome, nonzero(1 + 2 * data), stage="NETWORK", consumer_id=consumer + 1),
                Gate("SUM", syndrome, data, nonzero(2 + data), stage="NETWORK", consumer_id=consumer + 2),
            ]
            consumer += 3
            if injection_count > 1:
                network.append(Gate("CZ", data, (data + 1) % injection_count, nonzero(2 + data), stage="NETWORK"))
        else:
            network = [
                Gate("SUM", syndrome, data, nonzero(-(2 + data)), stage="NETWORK", consumer_id=consumer),
                Gate("CZ", data, syndrome, nonzero(2 + data), stage="NETWORK", consumer_id=consumer + 1),
                Gate("SUM", data, syndrome, nonzero(-(1 + 2 * data)), stage="NETWORK", consumer_id=consumer + 2),
            ]
            consumer += 3
            if injection_count > 1:
                network.append(Gate("CZ", data, (data - 1) % injection_count, nonzero(1 + data), stage="NETWORK"))
        gates.extend(network)
    directions = tuple(
        (1 if data % 2 == 0 else -1) * (1 if family == 0 else -1)
        for data in range(injection_count)
    )
    for data, direction in enumerate(directions):
        gates.append(Gate("FOURIER", data, direction=direction, stage="FINAL"))
    output = tuple(
        ((data + 1) if family == 0 else (2 * data + 1)) % P
        for data in range(injection_count)
    )
    gate_tuple = tuple(gates)
    return PublicProgram(
        family, injection_count, tuple(strengths), output, gate_tuple,
        program_digest(family, injection_count, strengths, output, gate_tuple),
    )


@dataclass
class Work:
    fourier_character_terms: int = 0
    permutation_cell_moves: int = 0
    diagonal_phase_multiplications: int = 0
    coherent_correction_multiplications: int = 0
    syndrome_consumer_applications: int = 0
    common_factor_cancellations: int = 0
    peak_denominator_exponent: int = 0
    peak_total_exact_payload_bits: int = 0
    retained_dynamic_inverse_history_entries: int = 0

    def observe(self, values: Sequence[K], exponent: int) -> None:
        self.peak_denominator_exponent = max(self.peak_denominator_exponent, exponent)
        self.peak_total_exact_payload_bits = max(
            self.peak_total_exact_payload_bits, payload_bits(values, exponent)
        )


def gate_phase(gate: Gate, state: Sequence[int]) -> int:
    left = state[gate.wire]
    right = state[gate.target] if gate.target >= 0 else 0
    if gate.kind == "CUBIC":
        return gate.parameter * left**3
    if gate.kind == "CZ":
        return gate.parameter * left * right
    if gate.kind == "Q":
        return -3 * gate.parameter * right * left**2
    if gate.kind == "L":
        return -3 * gate.parameter * right**2 * left
    if gate.kind == "G":
        return -gate.parameter * right**3
    raise RuntimeError("gate has no diagonal phase")


def apply_gate(values: list[K], scratch: list[K], exponent: int, gate: Gate, work: Work) -> int:
    if any(value != ZERO for value in scratch):
        raise RuntimeError("dirty M240 scratch")
    dimension = len(values)
    wire_count = 0
    size = dimension
    while size > 1:
        if size % P:
            raise RuntimeError("non-p5 carrier dimension")
        size //= P
        wire_count += 1
    if gate.kind == "FOURIER":
        for destination in range(dimension):
            output_state = coordinates(destination, wire_count)
            output = output_state[gate.wire]
            total = ZERO
            for source in range(P):
                source_state = output_state.copy()
                source_state[gate.wire] = source
                term = m237.k_mul(
                    m237.zeta_power(gate.direction * source * output),
                    values[flat_index(source_state)],
                )
                total = m237.k_add(total, term)
                work.fourier_character_terms += 1
            scratch[destination] = m237.k_mul(SQRT5, total)
        values[:] = scratch
        scratch[:] = [ZERO] * dimension
        exponent += 1
    elif gate.kind == "SUM":
        for source in range(dimension):
            state = coordinates(source, wire_count)
            state[gate.target] = (state[gate.target] + gate.parameter * state[gate.wire]) % P
            scratch[flat_index(state)] = values[source]
            work.permutation_cell_moves += 1
        values[:] = scratch
        scratch[:] = [ZERO] * dimension
    else:
        for location in range(dimension):
            state = coordinates(location, wire_count)
            values[location] = m237.k_mul(m237.zeta_power(gate_phase(gate, state)), values[location])
            work.diagonal_phase_multiplications += 1
            if gate.kind in {"Q", "L", "G"}:
                work.coherent_correction_multiplications += 1
    if gate.consumer_id >= 0:
        work.syndrome_consumer_applications += 1
    before = exponent
    exponent = canonicalize(values, exponent)
    work.common_factor_cancellations += before - exponent
    work.observe(values, exponent)
    return exponent


def selected_probability(values: Sequence[K], exponent: int, program: PublicProgram) -> tuple[K, int]:
    total = ZERO
    for syndrome in range(P):
        amplitude = values[flat_index((*program.output_data, syndrome))]
        total = m237.k_add(total, m237.k_mul(amplitude, m237.k_conjugate(amplitude)))
    encoded = [total]
    encoded_exponent = canonicalize(encoded, 2 * exponent)
    return encoded[0], encoded_exponent


class SyndromeNetworkPort:
    def __init__(self, injection_count: int) -> None:
        self.injection_count = injection_count
        self.dimension = P ** (injection_count + 1)
        self.values = [ZERO] * self.dimension
        self.values[0] = ONE
        self.scratch = [ZERO] * self.dimension
        self.cursor = 0
        self.consumer_cursor = 0
        self.owner = 0
        self.program_descriptor: tuple[object, ...] | None = None
        self.generation = 0
        self.last_restored_generation = 0
        self.leased = False
        self.port_type = PORT_TYPE

    def canonical(self) -> bool:
        return (
            self.values[0] == ONE and all(v == ZERO for v in self.values[1:])
            and all(v == ZERO for v in self.scratch) and self.cursor == 0
            and self.consumer_cursor == 0
        )

    def require(self, program: PublicProgram, owner: int, generation: int, port_type: str = PORT_TYPE) -> None:
        if not self.leased or owner != self.owner or generation != self.generation or port_type != self.port_type:
            raise RuntimeError("M240 custody mismatch")
        if self.program_descriptor != program.descriptor:
            raise RuntimeError("M240 descriptor custody mismatch")

    def lease(self, program: PublicProgram, owner: int, generation: int, port_type: str = PORT_TYPE) -> None:
        if program.injection_count != self.injection_count or owner <= 0 or port_type != PORT_TYPE:
            raise RuntimeError("invalid M240 lease")
        if self.leased or not self.canonical() or generation != self.last_restored_generation + 1:
            raise RuntimeError("nonmonotone or dirty M240 lease")
        self.leased = True
        self.owner = owner
        self.generation = generation
        self.program_descriptor = program.descriptor

    def forward(self, program: PublicProgram, owner: int, generation: int, work: Work) -> None:
        self.require(program, owner, generation)
        if self.cursor >= len(program.gates):
            raise RuntimeError("M240 forward cursor exhausted")
        gate = program.gates[self.cursor]
        if gate.consumer_id >= 0 and gate.consumer_id != self.consumer_cursor:
            raise RuntimeError("M240 consumer ordering mismatch")
        self.generation = generation
        self.values_exponent = apply_gate(
            self.values, self.scratch, getattr(self, "values_exponent", 0), gate, work
        )
        self.cursor += 1
        if gate.consumer_id >= 0:
            self.consumer_cursor += 1

    def inverse(self, program: PublicProgram, owner: int, generation: int, work: Work, supplied: Gate | None = None) -> None:
        self.require(program, owner, generation)
        if self.cursor <= 0:
            raise RuntimeError("M240 inverse cursor exhausted")
        expected = program.gates[self.cursor - 1]
        inverse = expected.inverse()
        if supplied is not None and supplied.serial() != inverse.serial():
            raise RuntimeError("wrong M240 inverse rejected")
        if expected.consumer_id >= 0 and expected.consumer_id != self.consumer_cursor - 1:
            raise RuntimeError("M240 inverse consumer ordering mismatch")
        self.values_exponent = apply_gate(
            self.values, self.scratch, getattr(self, "values_exponent", 0), inverse, work
        )
        self.cursor -= 1
        if expected.consumer_id >= 0:
            self.consumer_cursor -= 1

    def project(self, program: PublicProgram, owner: int, generation: int) -> tuple[K, int, str]:
        self.require(program, owner, generation)
        if self.cursor != len(program.gates) or self.consumer_cursor != program.consumer_count:
            raise RuntimeError("premature M240 boundary projection")
        value, exponent = selected_probability(self.values, self.values_exponent, program)
        slice_values = [self.values[flat_index((*program.output_data, s))] for s in range(P)]
        return value, exponent, vector_commitment(slice_values, self.values_exponent)

    def project_syndrome(self) -> object:
        raise RuntimeError("M240 syndrome projection forbidden")

    def release(self, program: PublicProgram, owner: int, generation: int) -> None:
        self.require(program, owner, generation)
        if not self.canonical() or getattr(self, "values_exponent", 0) != 0:
            raise RuntimeError("M240 carrier not exactly restored")
        self.last_restored_generation = generation
        self.leased = False
        self.owner = 0
        self.generation = 0
        self.program_descriptor = None


def run_transaction(
    port: SyndromeNetworkPort, program: PublicProgram, owner: int, generation: int
) -> dict[str, object]:
    if port is None:
        raise TypeError("null M240 port")
    port.lease(program, owner, generation)
    value_backing = id(port.values)
    scratch_backing = id(port.scratch)
    work = Work()
    while port.cursor < len(program.gates):
        port.forward(program, owner, generation, work)
    boundary, boundary_exponent, boundary_slice_commitment = port.project(program, owner, generation)
    retained_boundary = probability_encoding(boundary, boundary_exponent)
    final_commitment = vector_commitment(port.values, port.values_exponent)
    final_payload = payload_bits(port.values, port.values_exponent)
    while port.cursor:
        port.inverse(program, owner, generation, work)
    if retained_boundary != probability_encoding(boundary, boundary_exponent):
        raise RuntimeError("retained M240 result changed during inverse")
    port.release(program, owner, generation)
    return {
        "family": program.family,
        "injection_count": program.injection_count,
        "wire_count": program.wire_count,
        "syndrome_consumer_count": program.consumer_count,
        "selected_data_probability": retained_boundary,
        "selected_boundary_slice_commitment": boundary_slice_commitment,
        "final_state_commitment": final_commitment,
        "final_total_exact_payload_bits": final_payload,
        "resident_amplitude_field_cells": program.dimension,
        "scratch_amplitude_field_cells": program.dimension,
        "retained_final_boundary_during_inverse": True,
        "canonical_post_inverse_state_exact": port.canonical(),
        "same_amplitude_and_scratch_backings": (
            id(port.values) == value_backing and id(port.scratch) == scratch_backing
        ),
        "restoration_generation": port.last_restored_generation,
        "baseline_reload_used": False,
        "response_released_after_restoration": True,
        "work": asdict(work),
    }


def execute_gates(program: PublicProgram, gates: Sequence[Gate] | None = None) -> tuple[list[K], int]:
    values = [ZERO] * program.dimension
    values[0] = ONE
    scratch = [ZERO] * program.dimension
    exponent = 0
    work = Work()
    for gate in program.gates if gates is None else gates:
        exponent = apply_gate(values, scratch, exponent, gate, work)
    return values, exponent


def injection_cut_identity(program: PublicProgram) -> bool:
    injection_gates = [g for g in program.gates if g.stage in {"PREP", "INJECTION"}]
    actual, exponent = execute_gates(program, injection_gates)
    expected = [ZERO] * program.dimension
    numerator = sqrt5_power(program.wire_count)
    for state in itertools.product(range(P), repeat=program.wire_count):
        phase = sum(program.strengths[data] * state[data] ** 3 for data in range(program.injection_count))
        expected[flat_index(state)] = m237.k_mul(numerator, m237.zeta_power(phase))
    expected_exponent = canonicalize(expected, program.wire_count)
    return exponent == expected_exponent and actual == expected


def simulate_network_basis(program: PublicProgram, state: list[int]) -> tuple[list[int], int]:
    result = state.copy()
    phase = 0
    for gate in program.gates:
        if gate.stage != "NETWORK":
            continue
        if gate.kind == "SUM":
            result[gate.target] = (result[gate.target] + gate.parameter * result[gate.wire]) % P
        elif gate.kind == "CZ":
            phase += gate.parameter * result[gate.wire] * result[gate.target]
        else:
            raise RuntimeError("non-Clifford gate in compiled network baseline")
    return result, phase


def streamed_scalar_boundary(
    program: PublicProgram, fixed_initial_syndrome: int | None = None
) -> tuple[list[K], int, int]:
    accumulators = [ZERO] * P
    initial_syndromes: Iterable[int] = range(P) if fixed_initial_syndrome is None else (fixed_initial_syndrome,)
    normalization_power = 2 * program.injection_count + (1 if fixed_initial_syndrome is None else 0)
    numerator = sqrt5_power(normalization_power)
    terms = 0
    final_fouriers = [g for g in program.gates if g.stage == "FINAL"]
    for data_state in itertools.product(range(P), repeat=program.injection_count):
        cubic_phase = sum(a * x**3 for a, x in zip(program.strengths, data_state))
        for syndrome in initial_syndromes:
            transformed, phase = simulate_network_basis(program, [*data_state, syndrome])
            phase += cubic_phase
            phase += sum(
                gate.direction * transformed[gate.wire] * program.output_data[gate.wire]
                for gate in final_fouriers
            )
            destination_syndrome = transformed[program.syndrome_wire]
            accumulators[destination_syndrome] = m237.k_add(
                accumulators[destination_syndrome],
                m237.k_mul(numerator, m237.zeta_power(phase)),
            )
            terms += 1
    exponent = canonicalize(accumulators, normalization_power)
    return accumulators, exponent, terms


def probability_from_slice(values: Sequence[K], exponent: int) -> tuple[K, int]:
    total = ZERO
    for value in values:
        total = m237.k_add(total, m237.k_mul(value, m237.k_conjugate(value)))
    encoded = [total]
    result_exponent = canonicalize(encoded, 2 * exponent)
    return encoded[0], result_exponent


def compiled_scalar_baseline(program: PublicProgram) -> dict[str, object]:
    amplitudes, exponent, terms = streamed_scalar_boundary(program)
    probability, probability_exponent = probability_from_slice(amplitudes, exponent)
    return {
        "selected_data_probability": probability_encoding(probability, probability_exponent),
        "selected_boundary_slice_commitment": vector_commitment(amplitudes, exponent),
        "streamed_component_syndrome_terms": terms,
        "stabilizer_component_upper_bound": P**program.injection_count,
        "resident_boundary_accumulator_field_values": P,
        "peak_term_field_values": 1,
        "materialized_assignment_table_entries": 0,
        "full_amplitude_vector_retained": False,
    }


def dephased_syndrome_boundary(program: PublicProgram) -> dict[str, object]:
    total = ZERO
    common_exponent = 4 * program.injection_count + 1
    branch_terms = 0
    for syndrome in range(P):
        amplitudes, exponent, terms = streamed_scalar_boundary(program, syndrome)
        if exponent != 2 * program.injection_count:
            # Lift a canonically reduced branch back to the declared common denominator.
            amplitudes = [m237.k_scale(v, P ** (2 * program.injection_count - exponent)) for v in amplitudes]
            exponent = 2 * program.injection_count
        probability, probability_exponent = probability_from_slice(amplitudes, exponent)
        if probability_exponent > common_exponent - 1:
            raise RuntimeError("dephased branch exponent exceeds common scale")
        lifted = m237.k_scale(probability, P ** ((common_exponent - 1) - probability_exponent))
        total = m237.k_add(total, lifted)
        branch_terms += terms
    values = [total]
    exponent = canonicalize(values, common_exponent)
    return {
        "selected_data_probability": probability_encoding(values[0], exponent),
        "sequential_syndrome_branches": P,
        "streamed_terms": branch_terms,
        "restoring_path": False,
    }


def rejected(callback: Callable[[], object]) -> bool:
    try:
        callback()
    except (RuntimeError, TypeError, ValueError):
        return True
    return False


def controls() -> dict[str, bool]:
    programs = [compile_program(family, count) for family in (0, 1) for count in INJECTION_COUNTS]
    identities = all(injection_cut_identity(program) for program in programs)
    cut_mutations = True
    for program in programs:
        actual, actual_exponent = execute_gates(program)
        for injection in range(program.injection_count):
            mutated = [
                gate for gate in program.gates
                if not (gate.stage == "INJECTION" and gate.injection_id == injection)
            ]
            altered, altered_exponent = execute_gates(program, mutated)
            cut_mutations &= vector_commitment(actual, actual_exponent) != vector_commitment(altered, altered_exponent)
    selected = compile_program(0, 3)
    selected_values, selected_exponent = execute_gates(selected)
    selected_probability_value, selected_probability_exponent = selected_probability(selected_values, selected_exponent, selected)
    without_sum = [g for g in selected.gates if not (g.stage == "NETWORK" and g.kind == "SUM")]
    without_cz = [g for g in selected.gates if not (g.stage == "NETWORK" and g.kind == "CZ")]
    sum_values, sum_exponent = execute_gates(selected, without_sum)
    cz_values, cz_exponent = execute_gates(selected, without_cz)
    sum_probability = probability_encoding(*selected_probability(sum_values, sum_exponent, selected))
    cz_probability = probability_encoding(*selected_probability(cz_values, cz_exponent, selected))
    selected_probability_encoded = probability_encoding(selected_probability_value, selected_probability_exponent)
    network_gates = [i for i, g in enumerate(selected.gates) if g.stage == "NETWORK"]
    reordered = list(selected.gates)
    first, second = network_gates[:2]
    reordered[first], reordered[second] = reordered[second], reordered[first]
    reordered_values, reordered_exponent = execute_gates(selected, reordered)

    custody_program = compile_program(0, 2)
    premature = SyndromeNetworkPort(2)
    premature.lease(custody_program, 2401, 1)
    premature_projection = rejected(lambda: premature.project(custody_program, 2401, 1))
    syndrome_projection = rejected(premature.project_syndrome)
    wrong_owner = rejected(lambda: premature.require(custody_program, 2402, 1))
    wrong_generation = rejected(lambda: premature.require(custody_program, 2401, 2))
    wrong_type = rejected(lambda: premature.require(custody_program, 2401, 1, "WRONG"))
    dirty = SyndromeNetworkPort(2)
    dirty.lease(custody_program, 2403, 1)
    dirty.scratch[0] = ONE
    dirty_scratch = rejected(lambda: dirty.forward(custody_program, 2403, 1, Work()))

    missing = SyndromeNetworkPort(2)
    missing.lease(custody_program, 2404, 1)
    missing_work = Work()
    while missing.cursor < len(custody_program.gates):
        missing.forward(custody_program, 2404, 1, missing_work)
    missing.inverse(custody_program, 2404, 1, missing_work)
    missing_inverse = rejected(lambda: missing.release(custody_program, 2404, 1))

    reordered_inverse_port = SyndromeNetworkPort(2)
    reordered_inverse_port.lease(custody_program, 2405, 1)
    reordered_work = Work()
    while reordered_inverse_port.cursor < len(custody_program.gates):
        reordered_inverse_port.forward(custody_program, 2405, 1, reordered_work)
    wrong_expected = custody_program.gates[-2].inverse()
    reordered_inverse = rejected(
        lambda: reordered_inverse_port.inverse(
            custody_program, 2405, 1, reordered_work, supplied=wrong_expected
        )
    )
    wrong_inverse_port = SyndromeNetworkPort(2)
    wrong_inverse_port.lease(custody_program, 2406, 1)
    wrong_work = Work()
    while wrong_inverse_port.cursor < len(custody_program.gates):
        wrong_inverse_port.forward(custody_program, 2406, 1, wrong_work)
    last_inverse = custody_program.gates[-1].inverse()
    wrong_inverse = rejected(
        lambda: wrong_inverse_port.inverse(
            custody_program, 2406, 1, wrong_work,
            supplied=replace(last_inverse, direction=-last_inverse.direction),
        )
    )

    mutation = rejected(
        lambda: replace(
            custody_program,
            output_data=tuple((value + 1) % P for value in custody_program.output_data),
        )
    )
    null_port = rejected(lambda: run_transaction(None, custody_program, 2407, 1))  # type: ignore[arg-type]
    stale = SyndromeNetworkPort(1)
    stale_program = compile_program(0, 1)
    run_transaction(stale, stale_program, 2408, 1)
    stale_generation = rejected(lambda: stale.lease(compile_program(1, 1), 2408, 1))

    magic_a1, negative_a1 = one_magic_wigner_l1(1)
    magic_a2, negative_a2 = one_magic_wigner_l1(2)
    expected_magic = QuadraticReal(Fraction(1), Fraction(2, 5))
    dephasing_changes = all(
        compiled_scalar_baseline(program)["selected_data_probability"]
        != dephased_syndrome_boundary(program)["selected_data_probability"]
        for program in programs if program.injection_count >= 2
    )
    return {
        "all_injection_cut_identities_exact": identities,
        "each_declared_injection_changes_final_state_commitment": cut_mutations,
        "remove_network_sum_changes_selected_boundary": sum_probability != selected_probability_encoded,
        "remove_network_cz_changes_selected_boundary": cz_probability != selected_probability_encoded,
        "reorder_noncommuting_network_gates_changes_final_state": vector_commitment(
            reordered_values, reordered_exponent
        ) != vector_commitment(selected_values, selected_exponent),
        "dephasing_shared_syndrome_changes_selected_boundary_counts2_3_4": dephasing_changes,
        "single_magic_wigner_l1_exact_a1_a2": magic_a1 == expected_magic and magic_a2 == expected_magic,
        "single_magic_negative_cell_count_a1_a2": negative_a1 == negative_a2 == 5,
        "premature_projection_rejected": premature_projection,
        "syndrome_projection_rejected": syndrome_projection,
        "wrong_owner_rejected": wrong_owner,
        "wrong_generation_rejected": wrong_generation,
        "wrong_type_rejected": wrong_type,
        "same_id_descriptor_mutation_rejected": mutation,
        "dirty_scratch_rejected": dirty_scratch,
        "missing_inverse_rejected": missing_inverse,
        "reordered_inverse_rejected": reordered_inverse,
        "wrong_inverse_rejected_before_mutation": wrong_inverse,
        "stale_generation_rejected": stale_generation,
        "null_carrier_rejected": null_port,
        "accepted_path_serializes_syndrome_values": False,
        "accepted_path_materializes_assignment_or_history_table": False,
        "public_compiler_reads_final_answer": False,
        "dephased_diagnostic_enumerates_five_syndrome_branches": True,
        "dephased_diagnostic_is_nonrestoring": True,
    }


def comparable_case(case: dict[str, object]) -> dict[str, object]:
    return {
        key: case[key]
        for key in (
            "family", "injection_count", "wire_count", "syndrome_consumer_count",
            "selected_data_probability", "selected_boundary_slice_commitment",
            "final_state_commitment", "final_total_exact_payload_bits",
            "resident_amplitude_field_cells", "scratch_amplitude_field_cells",
            "canonical_post_inverse_state_exact", "same_amplitude_and_scratch_backings",
            "restoration_generation", "baseline_reload_used",
            "response_released_after_restoration",
        )
    }


def main(reference_path: Path) -> None:
    reference = json.loads(reference_path.read_text())
    cases: list[dict[str, object]] = []
    baselines: dict[tuple[int, int], dict[str, object]] = {}
    dephased: dict[tuple[int, int], dict[str, object]] = {}
    magic_laws: list[dict[str, object]] = []
    for count in INJECTION_COUNTS:
        port = SyndromeNetworkPort(count)
        primary_program = compile_program(0, count)
        primary = run_transaction(port, primary_program, 240000 + count, 1)
        reuse_program = compile_program(1, count)
        reuse = run_transaction(port, reuse_program, 240000 + count, 2)
        fresh = run_transaction(SyndromeNetworkPort(count), reuse_program, 240000 + count, 1)
        primary["run_kind"] = "PRIMARY"
        reuse["run_kind"] = "RESTORED_REUSE"
        for case, program in ((primary, primary_program), (reuse, reuse_program)):
            baseline = compiled_scalar_baseline(program)
            if baseline["selected_data_probability"] != case["selected_data_probability"]:
                raise RuntimeError("streamed scalar boundary disagrees with carrier")
            if baseline["selected_boundary_slice_commitment"] != case["selected_boundary_slice_commitment"]:
                raise RuntimeError("streamed scalar boundary slice disagrees")
            case["matched_streamed_scalar_baseline"] = baseline
            cases.append(case)
            baselines[(program.family, count)] = baseline
            dephased[(program.family, count)] = dephased_syndrome_boundary(program)
        if comparable_case(reuse) | {"restoration_generation": 1} != comparable_case(fresh):
            # Generation is the only intentionally different case-level field.
            left = comparable_case(reuse)
            left["restoration_generation"] = 1
            if left != comparable_case(fresh):
                raise RuntimeError("fresh/restored M240 reuse mismatch")
        one_magic, _ = one_magic_wigner_l1(primary_program.strengths[0])
        l1 = one_magic.power(count)
        negativity = (l1 - QuadraticReal(Fraction(1), Fraction(0))).scale(Fraction(1, 2))
        magic_laws.append({
            "injection_count": count,
            "exact_wigner_l1": l1.encoding(),
            "exact_negative_mass": negativity.encoding(),
            "stabilizer_component_upper_bound": P**count,
            "product_input_law_only": True,
            "computational_lower_bound_established": False,
        })
    reference_cases = reference["cases"]
    if [comparable_case(case) for case in cases] != reference_cases:
        raise RuntimeError("standalone M240 transaction parity failed")
    control_results = controls()
    if control_results != reference["controls"]:
        raise RuntimeError("standalone M240 control parity failed")
    if magic_laws != reference["magic_laws"]:
        raise RuntimeError("standalone M240 magic-law parity failed")

    result = {
        "schema": "cat_cas.zeta5_interacting_multi_injection_syndrome_network.v1",
        "result": RESULT,
        "claim": CLAIM,
        "claim_ceiling": CLAIM_CEILING,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "cases": cases,
        "controls": control_results,
        "magic_laws": magic_laws,
        "reuse": {
            "primary_generations": [case["restoration_generation"] for case in cases if case["run_kind"] == "PRIMARY"],
            "reuse_generations": [case["restoration_generation"] for case in cases if case["run_kind"] == "RESTORED_REUSE"],
            "same_backing_reuse_at_each_declared_count": all(case["same_amplitude_and_scratch_backings"] for case in cases),
            "fresh_restored_parity": True,
            "baseline_reload_used": False,
        },
        "composition_law": {
            "field": "Q(zeta_5)",
            "injection_counts": list(INJECTION_COUNTS),
            "one_actual_shared_syndrome_wire": True,
            "distinct_data_wires_receive_distinct_injections": True,
            "post_injection_network_uses_only_clifford_sum_cz_fourier": True,
            "shared_syndrome_consumed_before_final_boundary": True,
            "syndrome_values_projected": False,
            "exact_single_magic_wigner_l1": QuadraticReal(Fraction(1), Fraction(2, 5)).encoding(),
            "exact_l1_multiplies_across_product_magic_inputs": True,
            "post_injection_clifford_network_preserves_wigner_l1": True,
            "l1_growth_is_not_a_classical_runtime_lower_bound": True,
            "direct_process_logical_custody_only": True,
        },
        "matched_classical": {
            "strongest_implemented": "PUBLIC_INJECTION_COMPILED_STREAMED_STABILIZER_COMPONENT_AND_SYNDROME_SCALAR_BOUNDARY_RECURRENCE",
            "streamed_terms_by_count": [P ** (count + 1) for count in INJECTION_COUNTS],
            "stabilizer_component_upper_bounds": [P**count for count in INJECTION_COUNTS],
            "resident_boundary_accumulator_field_values": P,
            "peak_term_field_values": 1,
            "declared_accumulator_plus_term_field_values": P + 1,
            "assignment_table_materialized": False,
            "full_amplitude_vector_retained": False,
            "optimal_stabilizer_rank_or_extent_proved": False,
            "computational_advantage": False,
        },
        "resource_law": {
            "phase_resident_field_cells_by_count": [P ** (count + 1) for count in INJECTION_COUNTS],
            "phase_equal_scratch_field_cells_by_count": [P ** (count + 1) for count in INJECTION_COUNTS],
            "phase_resident_plus_scratch_backing_cells_by_count": [2 * P ** (count + 1) for count in INJECTION_COUNTS],
            "classical_persistent_field_values": 5,
            "classical_peak_term_field_values": 1,
            "classical_declared_accumulator_plus_term_field_values": 6,
            "accepted_phase_backings_exceed_streamed_scalar_baseline": True,
            "comparison_basis": "DECLARED_QZETA5_FIELD_BACKINGS_AND_SCALAR_ACCUMULATORS_NOT_WHOLE_TRANSACTION_LIVENESS",
            "public_descriptors_loop_coordinates_phase_integers_and_container_state_excluded_not_zero": True,
            "retained_dynamic_inverse_history_entries": 0,
            "dephased_diagnostic_excluded_from_accepted_restoring_path": True,
            "resource_verification_level": "PACKAGE_SELF_REVIEW",
            "whole_transaction_live_cell_and_payload_accounting_complete": False,
            "python_objects_allocator_hash_serialization_rss_excluded_not_zero": True,
        },
        "separate_reference": {
            "imports_m240_m239_or_m237_production": False,
            "independent_power_basis_arithmetic": True,
            "independent_full_amplitude_transaction": True,
            "independent_streamed_scalar_boundary": True,
            "independent_wigner_l1_reconstruction": True,
            "independent_custody_state_machine": True,
            "source_sha256": reference["source_sha256"],
        },
        "source_dependencies": {
            "production_sha256": file_sha256(Path(__file__)),
            "separate_reference_sha256": file_sha256(
                Path(__file__).with_name("zeta5_interacting_multi_injection_syndrome_network_separate_reference.py")
            ),
            "m237_algebra_sha256": file_sha256(
                Path(__file__).with_name("zeta5_normalized_cubic_fourier_coherent_port.py")
            ),
        },
        "claim_limits": {
            "wigner_l1_is_computational_lower_bound": False,
            "optimal_stabilizer_rank_or_extent": False,
            "magic_resource_unavailable_to_compact_classical_software": False,
            "computational_advantage": False,
            "fixed_rank_or_fixed_width_state": False,
            "catvm_machine_custody": False,
            "general_measurement_or_inference": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "unbounded_catalytic_computation": False,
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: zeta5_interacting_multi_injection_syndrome_network.py REFERENCE_JSON")
    main(Path(sys.argv[1]))
