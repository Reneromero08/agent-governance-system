#!/usr/bin/env python3
"""M239: exact coherent p=5 magic injection through one unresolved syndrome.

The accepted mechanism is a bounded two-qudit direct-process transaction.  It
does not model a physical measurement and does not claim a resource beyond the
matched five-component stabilizer-sum or exact 25-amplitude recurrences.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import zeta5_normalized_cubic_fourier_coherent_port as m237


P = 5
PORT_TYPE = "ZETA5_COHERENT_MAGIC_INJECTION_SYNDROME_V1"
PAIR_COUNTS = (1, 2, 4)
K = m237.K
ZERO = m237.ZERO
ONE = m237.ONE
SQRT5 = m237.SQRT5

RESULT = "PASS_EXACT_ZETA5_COHERENT_MAGIC_INJECTION_SYNDROME_PORT_STRICT_SCOPE"
CLAIM = (
    "EXACT_P5_CUBIC_MAGIC_INJECTION_RETAINS_ONE_COHERENT_UNRESOLVED5_VALUE_"
    "SYNDROME_PORT_CONSUMED_BY_THREE_FIBERWISE_STABILIZER_CORRECTIONS_AND_"
    "DECLARED_NONCOMMUTING_SUM_CZ_WORDS_ON_ONE_ACTUAL25_AMPLITUDE_CARRIER_"
    "WITH_FINAL_ONLY_DATA_PROBABILITY_EXACT_SAME_BACKING_RESTORATION_AND_"
    "DESCRIPTOR_DISTINCT_REUSE_BUT_THE_COMPLETE_INJECTION_HAS_AN_EXACT_FIVE_"
    "STABILIZER_COMPONENT_UPPER_BOUND_AND_THE_SMALLER_COMPILED_CLASSICAL_"
    "RECURRENCE_REMAINS"
)
CLAIM_CEILING = (
    "QZETA5_TWO_QUDIT_ONE_CUBIC_INJECTION_FAMILIES_A1_2_CONSUMER_PAIR_COUNTS1_2_4_"
    "COHERENT_SYNDROME_DIRECT_PROCESS_LOGICAL_CUSTODY_ONLY"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def canonicalize(values: list[K], exponent: int) -> int:
    while exponent and all(c % 5 == 0 for value in values for c in value):
        for index, value in enumerate(values):
            values[index] = tuple(c // 5 for c in value)  # type: ignore[assignment]
        exponent -= 1
    return exponent


def payload_bits(values: Sequence[K], exponent: int) -> int:
    return sum(signed_bits(c) for value in values for c in value) + signed_bits(5**exponent)


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
    if value[1] != 0 or value[2] != value[3]:
        raise RuntimeError("probability left the real cyclotomic subfield")
    a = 2 * value[0] - value[2]
    b = -value[2]
    return {
        "a_numerator": a,
        "b_sqrt5_numerator": b,
        "denominator_twice_power5_exponent": exponent,
        "commitment": value_commitment(value, exponent),
    }


def index(data: int, syndrome: int) -> int:
    return P * data + syndrome


@dataclass(frozen=True)
class Gate:
    kind: str
    wire: int = -1
    target: int = -1
    parameter: int = 0
    direction: int = 1
    consumer_id: int = -1

    def __post_init__(self) -> None:
        if self.kind not in {"FOURIER", "CUBIC", "SUM", "CZ", "Q", "L", "G"}:
            raise ValueError("unknown M239 gate")
        if self.kind == "FOURIER" and (self.wire not in (0, 1) or self.direction not in (-1, 1)):
            raise ValueError("invalid Fourier gate")
        if self.kind in {"CUBIC", "Q", "L", "G", "SUM", "CZ"} and self.parameter % P == 0:
            raise ValueError("zero gate parameter outside declared grammar")
        if self.kind in {"SUM", "CZ"} and {self.wire, self.target} != {0, 1}:
            raise ValueError("two-wire gate must address data and syndrome")
        if self.kind in {"Q", "L", "G", "SUM", "CZ"} and self.consumer_id < 0:
            raise ValueError("syndrome consumer lacks public identity")
        if self.kind not in {"Q", "L", "G", "SUM", "CZ"} and self.consumer_id != -1:
            raise ValueError("nonconsumer gate carries consumer identity")

    def inverse(self) -> "Gate":
        if self.kind == "FOURIER":
            return Gate(self.kind, self.wire, self.target, self.parameter, -self.direction, self.consumer_id)
        return Gate(self.kind, self.wire, self.target, -self.parameter, self.direction, self.consumer_id)

    def serial(self) -> tuple[object, ...]:
        return (
            self.kind, self.wire, self.target, self.parameter % P,
            self.direction, self.consumer_id,
        )


def program_digest(
    family: int, pair_count: int, output_data: int, gates: Sequence[Gate]
) -> str:
    payload = {
        "family": family,
        "pair_count": pair_count,
        "output_data": output_data,
        "gates": [gate.serial() for gate in gates],
        "port_type": PORT_TYPE,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class PublicProgram:
    family: int
    pair_count: int
    output_data: int
    gates: tuple[Gate, ...]
    program_id: str

    def __post_init__(self) -> None:
        if self.family not in (0, 1) or self.pair_count not in PAIR_COUNTS:
            raise ValueError("program outside declared M239 family")
        if not 0 <= self.output_data < P:
            raise ValueError("invalid public output")
        consumer_ids = [gate.consumer_id for gate in self.gates if gate.consumer_id >= 0]
        if consumer_ids != list(range(len(consumer_ids))):
            raise ValueError("consumer identities are not unique and ordered")
        expected = program_digest(self.family, self.pair_count, self.output_data, self.gates)
        if self.program_id != expected:
            raise ValueError("program digest is not bound to the full descriptor")

    @property
    def descriptor(self) -> tuple[object, ...]:
        return (
            self.family, self.pair_count, self.output_data,
            tuple(gate.serial() for gate in self.gates), self.program_id,
        )

    @property
    def consumer_count(self) -> int:
        return sum(gate.consumer_id >= 0 for gate in self.gates)

    @property
    def cubic_strength(self) -> int:
        return self.family + 1


def compile_program(family: int, pair_count: int) -> PublicProgram:
    public_words: dict[tuple[int, int], tuple[tuple[str, int], ...]] = {
        (0, 1): (("F", -1), ("SUM", 1), ("CZ", 2), ("F", 1), ("SUM", 1)),
        (0, 2): (("SUM", 1), ("F", 1), ("CZ", 2), ("SUM", 1), ("CZ", 4),
                 ("F", 1), ("F", -1), ("SUM", 3)),
        (0, 4): (("F", -1), ("SUM", 1), ("CZ", 2), ("F", 1), ("SUM", 3),
                 ("CZ", 4), ("SUM", 3), ("CZ", 4), ("F", 1), ("CZ", 2),
                 ("SUM", 1), ("F", -1), ("F", -1), ("SUM", 3)),
        (1, 1): (("F", -1), ("SUM", 3), ("CZ", 3), ("F", -1), ("SUM", 3)),
        (1, 2): (("SUM", 4), ("F", 1), ("CZ", 1), ("CZ", 2), ("SUM", 2),
                 ("F", 1), ("F", 1), ("SUM", 4)),
        (1, 4): (("SUM", 3), ("F", -1), ("CZ", 2), ("CZ", 2), ("F", 1),
                 ("SUM", 1), ("F", -1), ("SUM", 3), ("CZ", 4), ("CZ", 2),
                 ("SUM", 2), ("F", 1), ("F", -1), ("SUM", 2)),
    }
    if (family, pair_count) not in public_words:
        raise ValueError("program outside declared M239 family")
    a = family + 1
    gates: list[Gate] = [
        Gate("FOURIER", wire=0),
        Gate("FOURIER", wire=1),
        Gate("CUBIC", wire=1, parameter=a),
        Gate("SUM", wire=0, target=1, parameter=-1, consumer_id=0),
        Gate("Q", parameter=a, consumer_id=1),
        Gate("L", parameter=a, consumer_id=2),
        Gate("G", parameter=a, consumer_id=3),
    ]
    consumer_id = 4
    for kind, parameter in public_words[(family, pair_count)]:
        if kind == "F":
            gates.append(Gate("FOURIER", wire=1, direction=parameter))
        else:
            gates.append(
                Gate(kind, wire=1, target=0, parameter=parameter, consumer_id=consumer_id)
            )
            consumer_id += 1
    gates.append(Gate("FOURIER", wire=0, direction=1 if family == 0 else -1))
    gate_tuple = tuple(gates)
    output_data = (pair_count + 2 * family) % P
    return PublicProgram(
        family, pair_count, output_data, gate_tuple,
        program_digest(family, pair_count, output_data, gate_tuple),
    )


@dataclass
class Work:
    fourier_character_terms: int = 0
    permutation_cell_moves: int = 0
    diagonal_phase_multiplications: int = 0
    coherent_nonclifford_correction_multiplications: int = 0
    syndrome_consumer_applications: int = 0
    common_factor_cancellations: int = 0
    peak_denominator_exponent: int = 0
    peak_numerator_payload_bits: int = 0
    peak_total_exact_payload_bits: int = 0
    retained_dynamic_inverse_history_entries: int = 0

    def observe(self, values: Sequence[K], exponent: int) -> None:
        numerator = sum(signed_bits(c) for value in values for c in value)
        self.peak_denominator_exponent = max(self.peak_denominator_exponent, exponent)
        self.peak_numerator_payload_bits = max(self.peak_numerator_payload_bits, numerator)
        self.peak_total_exact_payload_bits = max(
            self.peak_total_exact_payload_bits, numerator + signed_bits(5**exponent)
        )


def apply_gate(
    values: list[K], scratch: list[K], exponent: int, gate: Gate, work: Work
) -> int:
    if any(value != ZERO for value in scratch):
        raise RuntimeError("dirty M239 scratch")
    if gate.kind == "FOURIER":
        for fixed in range(P):
            for output in range(P):
                total = ZERO
                for source in range(P):
                    data, syndrome = (source, fixed) if gate.wire == 0 else (fixed, source)
                    phase = gate.direction * source * output
                    total = m237.k_add(
                        total, m237.k_mul(m237.zeta_power(phase), values[index(data, syndrome)])
                    )
                    work.fourier_character_terms += 1
                data, syndrome = (output, fixed) if gate.wire == 0 else (fixed, output)
                scratch[index(data, syndrome)] = m237.k_mul(SQRT5, total)
        values[:] = scratch
        scratch[:] = [ZERO] * len(scratch)
        exponent += 1
    elif gate.kind == "SUM":
        for data in range(P):
            for syndrome in range(P):
                destination = [data, syndrome]
                destination[gate.target] = (
                    destination[gate.target] + gate.parameter * destination[gate.wire]
                ) % P
                scratch[index(destination[0], destination[1])] = values[index(data, syndrome)]
                work.permutation_cell_moves += 1
        values[:] = scratch
        scratch[:] = [ZERO] * len(scratch)
    else:
        for data in range(P):
            for syndrome in range(P):
                if gate.kind == "CUBIC":
                    coordinate = data if gate.wire == 0 else syndrome
                    phase = gate.parameter * coordinate**3
                elif gate.kind == "CZ":
                    phase = gate.parameter * data * syndrome
                elif gate.kind == "Q":
                    phase = -3 * gate.parameter * syndrome * data**2
                elif gate.kind == "L":
                    phase = -3 * gate.parameter * syndrome**2 * data
                elif gate.kind == "G":
                    phase = -gate.parameter * syndrome**3
                else:
                    raise RuntimeError("unknown diagonal M239 gate")
                location = index(data, syndrome)
                values[location] = m237.k_mul(m237.zeta_power(phase), values[location])
                work.diagonal_phase_multiplications += 1
                if gate.kind in {"Q", "L", "G"}:
                    work.coherent_nonclifford_correction_multiplications += 1
    if gate.consumer_id >= 0:
        work.syndrome_consumer_applications += 1
    before = exponent
    exponent = canonicalize(values, exponent)
    work.common_factor_cancellations += before - exponent
    work.observe(values, exponent)
    return exponent


def selected_probability(
    values: Sequence[K], exponent: int, output_data: int
) -> tuple[K, int]:
    total = ZERO
    for syndrome in range(P):
        amplitude = values[index(output_data, syndrome)]
        total = m237.k_add(total, m237.k_mul(amplitude, m237.k_conjugate(amplitude)))
    encoded = [total]
    encoded_exponent = canonicalize(encoded, 2 * exponent)
    return encoded[0], encoded_exponent


class SyndromePort:
    def __init__(self) -> None:
        self.values = [ZERO] * 25
        self.values[index(0, 0)] = ONE
        self.scratch = [ZERO] * 25
        self.exponent = 0
        self.cursor = 0
        self.consumer_cursor = 0
        self.leased = False
        self.owner = 0
        self.descriptor: tuple[object, ...] | None = None
        self.generation = 0
        self.last_restored_generation = 0

    def canonical(self) -> bool:
        return (
            self.values == [ONE] + [ZERO] * 24
            and self.scratch == [ZERO] * 25
            and self.exponent == 0
            and self.cursor == 0
            and self.consumer_cursor == 0
        )

    def require(self, owner: int, program: PublicProgram, generation: int) -> None:
        if not self.leased or owner != self.owner or generation != self.generation:
            raise RuntimeError("syndrome custody mismatch")
        if program.descriptor != self.descriptor:
            raise RuntimeError("syndrome program descriptor mismatch")

    def lease(
        self, owner: int, program: PublicProgram, generation: int,
        port_type: str = PORT_TYPE,
    ) -> None:
        if port_type != PORT_TYPE or self.leased:
            raise RuntimeError("invalid syndrome lease")
        if generation != self.last_restored_generation + 1 or not self.canonical():
            raise RuntimeError("nonmonotone or dirty syndrome lease")
        self.leased = True
        self.owner = owner
        self.descriptor = program.descriptor
        self.generation = generation

    def forward(
        self, owner: int, program: PublicProgram, generation: int,
        gate_index: int, work: Work,
    ) -> None:
        self.require(owner, program, generation)
        if gate_index != self.cursor:
            raise RuntimeError("forward gate ordering mismatch")
        gate = program.gates[gate_index]
        if gate.consumer_id >= 0:
            if gate.consumer_id != self.consumer_cursor:
                raise RuntimeError("syndrome consumer ordering mismatch")
            self.consumer_cursor += 1
        self.exponent = apply_gate(self.values, self.scratch, self.exponent, gate, work)
        self.cursor += 1

    def inverse(
        self, owner: int, program: PublicProgram, generation: int,
        gate_index: int, work: Work,
    ) -> None:
        self.require(owner, program, generation)
        if gate_index != self.cursor - 1:
            raise RuntimeError("inverse gate ordering mismatch")
        gate = program.gates[gate_index]
        if gate.consumer_id >= 0 and gate.consumer_id != self.consumer_cursor - 1:
            raise RuntimeError("inverse syndrome consumer ordering mismatch")
        self.exponent = apply_gate(self.values, self.scratch, self.exponent, gate.inverse(), work)
        if gate.consumer_id >= 0:
            self.consumer_cursor -= 1
        self.cursor -= 1

    def project(
        self, owner: int, program: PublicProgram, generation: int
    ) -> dict[str, object]:
        self.require(owner, program, generation)
        if self.cursor != len(program.gates) or self.consumer_cursor != program.consumer_count:
            raise RuntimeError("premature syndrome boundary projection")
        if any(value != ZERO for value in self.scratch):
            raise RuntimeError("dirty scratch at projection")
        probability, probability_exponent = selected_probability(
            self.values, self.exponent, program.output_data
        )
        norm = ZERO
        for amplitude in self.values:
            norm = m237.k_add(norm, m237.k_mul(amplitude, m237.k_conjugate(amplitude)))
        norm_values = [norm]
        norm_exponent = canonicalize(norm_values, 2 * self.exponent)
        if norm_values[0] != ONE or norm_exponent != 0:
            raise RuntimeError("M239 normalization failed")
        return {
            "selected_data_probability": probability_encoding(probability, probability_exponent),
            "selected_data_probability_nonzero": probability != ZERO,
            "normalization_exact": True,
            "final_state_commitment": vector_commitment(self.values, self.exponent),
        }

    def project_syndrome(self) -> object:
        raise RuntimeError("syndrome projection forbidden")

    def release(self, owner: int, program: PublicProgram, generation: int) -> None:
        self.require(owner, program, generation)
        if not self.canonical():
            raise RuntimeError("syndrome carrier not exactly restored")
        self.leased = False
        self.last_restored_generation = generation
        self.owner = 0
        self.descriptor = None
        self.generation = 0


def run_transaction(
    port: SyndromePort, program: PublicProgram, generation: int,
    owner: int = 23901,
) -> dict[str, object]:
    if port is None:
        raise TypeError("null syndrome port")
    backings = (id(port.values), id(port.scratch))
    work = Work()
    work.observe(port.values, port.exponent)
    port.lease(owner, program, generation)
    for gate_index in range(len(program.gates)):
        port.forward(owner, program, generation, gate_index, work)
    boundary = port.project(owner, program, generation)
    retained = json.dumps(boundary, sort_keys=True)
    final_exponent = port.exponent
    final_payload = payload_bits(port.values, port.exponent)
    for gate_index in range(len(program.gates) - 1, -1, -1):
        port.inverse(owner, program, generation, gate_index, work)
    exact = port.canonical()
    same = backings == (id(port.values), id(port.scratch))
    port.release(owner, program, generation)
    if retained != json.dumps(boundary, sort_keys=True):
        raise RuntimeError("retained M239 boundary was consumed by inverse")
    return {
        "family": program.family,
        "cubic_strength": program.cubic_strength,
        "consumer_pair_count": program.pair_count,
        "syndrome_consumer_count": program.consumer_count,
        "resident_amplitude_field_cells": 25,
        "scratch_amplitude_field_cells": 25,
        "final_denominator_exponent": final_exponent,
        "final_total_exact_payload_bits": final_payload,
        **boundary,
        "retained_final_boundary_during_inverse": True,
        "canonical_post_inverse_state_exact": exact,
        "same_amplitude_and_scratch_backings": same,
        "restoration_generation": generation,
        "baseline_reload_used": False,
        "work": asdict(work),
    }


def execute_gates(gates: Sequence[Gate]) -> tuple[list[K], int]:
    values = [ZERO] * 25
    values[0] = ONE
    scratch = [ZERO] * 25
    exponent = 0
    work = Work()
    for gate in gates:
        exponent = apply_gate(values, scratch, exponent, gate, work)
    return values, exponent


def injection_identity_exact(a: int, data_input: int) -> bool:
    values = [ZERO] * 25
    values[index(data_input, 0)] = ONE
    scratch = [ZERO] * 25
    exponent = 0
    gates = (
        Gate("FOURIER", wire=1), Gate("CUBIC", wire=1, parameter=a),
        Gate("SUM", wire=0, target=1, parameter=-1, consumer_id=0),
        Gate("Q", parameter=a, consumer_id=1),
        Gate("L", parameter=a, consumer_id=2),
        Gate("G", parameter=a, consumer_id=3),
    )
    work = Work()
    for gate in gates:
        exponent = apply_gate(values, scratch, exponent, gate, work)
    expected = [ZERO] * 25
    phase = m237.zeta_power(a * data_input**3)
    for syndrome in range(P):
        expected[index(data_input, syndrome)] = m237.k_mul(SQRT5, phase)
    expected_exponent = 1
    return values == expected and exponent == expected_exponent


def compact_apply_gate(values: list[K], exponent: int, gate: Gate) -> int:
    """Strong exact baseline: 25 values and at most five saved Fourier inputs."""
    if gate.kind == "FOURIER":
        for fixed in range(P):
            old = [
                values[index(source, fixed) if gate.wire == 0 else index(fixed, source)]
                for source in range(P)
            ]
            for output in range(P):
                total = ZERO
                for source, amplitude in enumerate(old):
                    total = m237.k_add(
                        total,
                        m237.k_mul(
                            m237.zeta_power(gate.direction * source * output), amplitude
                        ),
                    )
                location = index(output, fixed) if gate.wire == 0 else index(fixed, output)
                values[location] = m237.k_mul(SQRT5, total)
        exponent += 1
    elif gate.kind == "SUM":
        permutation: list[int] = []
        for data in range(P):
            for syndrome in range(P):
                destination = [data, syndrome]
                destination[gate.target] = (
                    destination[gate.target] + gate.parameter * destination[gate.wire]
                ) % P
                permutation.append(index(destination[0], destination[1]))
        visited = [False] * 25
        for start in range(25):
            if visited[start]:
                continue
            current = start
            carried = values[current]
            while True:
                visited[current] = True
                destination = permutation[current]
                carried, values[destination] = values[destination], carried
                current = destination
                if current == start:
                    break
    elif gate.kind == "CZ":
        for data in range(P):
            for syndrome in range(P):
                location = index(data, syndrome)
                values[location] = m237.k_mul(
                    values[location],
                    m237.zeta_power(gate.parameter * data * syndrome),
                )
    else:
        raise RuntimeError("compact baseline received a non-Clifford post-injection gate")
    return canonicalize(values, exponent)


def compiled_injection_identity_baseline(program: PublicProgram) -> dict[str, object]:
    """Compile the injection identity, then use the compact 25-value recurrence."""
    a = program.cubic_strength
    values = [ZERO] * 25
    # Complete public injection identity: D_a|+>_D tensor |+>_S.
    for data in range(P):
        coefficient = m237.zeta_power(a * data**3)
        for syndrome in range(P):
            values[index(data, syndrome)] = coefficient
    exponent = 1
    for gate in program.gates[7:]:
        exponent = compact_apply_gate(values, exponent, gate)
    probability, probability_exponent = selected_probability(values, exponent, program.output_data)
    return {
        "stabilizer_component_upper_bound": 5,
        "resident_field_cells": 25,
        "peak_fourier_saved_input_field_cells": 5,
        "peak_accumulator_field_cells": 1,
        "compiled_injection_gates_removed": 7,
        "selected_data_probability": probability_encoding(probability, probability_exponent),
        "final_state_commitment": vector_commitment(values, exponent),
        "final_denominator_exponent": exponent,
        "final_total_exact_payload_bits": payload_bits(values, exponent),
    }


def dephased_syndrome_probability(program: PublicProgram) -> dict[str, object]:
    """Non-restoring diagnostic: remove coherence between syndrome values."""
    a = program.cubic_strength
    probability = ZERO
    probability_exponent = 0
    # After complete injection, each dephased syndrome branch retains the
    # normalized data state D_a|+>; the classical branch weight is 1/5.
    for syndrome_seed in range(P):
        values = [ZERO] * 25
        for data_seed in range(P):
            values[index(data_seed, syndrome_seed)] = m237.k_mul(
                SQRT5, m237.zeta_power(a * data_seed**3)
            )
        scratch = [ZERO] * 25
        exponent = 1
        work = Work()
        for gate in program.gates[7:]:
            exponent = apply_gate(values, scratch, exponent, gate, work)
        branch_probability, branch_exponent = selected_probability(
            values, exponent, program.output_data
        )
        common_exponent = max(probability_exponent, branch_exponent)
        probability = m237.k_add(
            m237.k_scale(probability, 5 ** (common_exponent - probability_exponent)),
            m237.k_scale(branch_probability, 5 ** (common_exponent - branch_exponent)),
        )
        probability_exponent = common_exponent
        encoded = [probability]
        probability_exponent = canonicalize(encoded, probability_exponent)
        probability = encoded[0]
    encoded = [probability]
    probability_exponent = canonicalize(encoded, probability_exponent + 1)
    return probability_encoding(encoded[0], probability_exponent)


def rejected(callback: Callable[[], object]) -> bool:
    try:
        callback()
    except (RuntimeError, TypeError, ValueError):
        return True
    return False


def controls() -> dict[str, bool]:
    program = compile_program(0, 2)
    owner = 23911
    identity = all(injection_identity_exact(a, data) for a in (1, 2) for data in range(P))

    full_values, full_exponent = execute_gates(program.gates)
    omission_changes: dict[str, bool] = {}
    for omitted in ("Q", "L", "G"):
        mutated = tuple(gate for gate in program.gates if gate.kind != omitted)
        values, exponent = execute_gates(mutated)
        omission_changes[omitted] = vector_commitment(values, exponent) != vector_commitment(full_values, full_exponent)

    omitted_g_values, omitted_g_exponent = execute_gates(program.gates[:6])
    expected_omitted_g = [ZERO] * 25
    for data in range(P):
        for syndrome in range(P):
            expected_omitted_g[index(data, syndrome)] = m237.zeta_power(
                program.cubic_strength * (data**3 + syndrome**3)
            )

    baseline = compiled_injection_identity_baseline(program)
    coherent_probability, coherent_exponent = selected_probability(
        full_values, full_exponent, program.output_data
    )
    dephased = dephased_syndrome_probability(program)
    coherent = probability_encoding(coherent_probability, coherent_exponent)

    premature = SyndromePort()
    premature.lease(owner, program, 1)
    premature_projection = rejected(lambda: premature.project(owner, program, 1))
    syndrome_projection = rejected(premature.project_syndrome)
    wrong_owner = rejected(lambda: premature.forward(owner + 1, program, 1, 0, Work()))
    wrong_program = rejected(lambda: premature.forward(owner, compile_program(1, 2), 1, 0, Work()))
    wrong_generation = rejected(lambda: premature.forward(owner, program, 2, 0, Work()))
    wrong_type = rejected(lambda: SyndromePort().lease(owner, program, 1, "WRONG"))
    forged_gates = program.gates[:-1] + (Gate("FOURIER", wire=0, direction=-1),)
    forged = rejected(
        lambda: PublicProgram(
            program.family, program.pair_count, program.output_data,
            forged_gates, program.program_id,
        )
    )
    duplicate_gates = list(program.gates)
    duplicate_gates[5] = Gate("L", parameter=1, consumer_id=1)
    duplicate_consumer = rejected(
        lambda: PublicProgram(
            program.family, program.pair_count, program.output_data,
            tuple(duplicate_gates),
            program_digest(program.family, program.pair_count, program.output_data, duplicate_gates),
        )
    )
    null = rejected(lambda: run_transaction(None, program, 1))  # type: ignore[arg-type]
    dirty = SyndromePort()
    dirty.scratch[0] = ONE
    dirty_rejected = rejected(lambda: dirty.lease(owner, program, 1))

    missing = SyndromePort()
    missing.lease(owner, program, 1)
    for gate_index in range(len(program.gates)):
        missing.forward(owner, program, 1, gate_index, Work())
    missing_inverse = rejected(lambda: missing.release(owner, program, 1))

    reordered = SyndromePort()
    reordered.lease(owner, program, 1)
    for gate_index in range(len(program.gates)):
        reordered.forward(owner, program, 1, gate_index, Work())
    reordered_inverse = rejected(
        lambda: reordered.inverse(owner, program, 1, len(program.gates) - 2, Work())
    )

    wrong = SyndromePort()
    wrong.lease(owner, program, 1)
    for gate_index in range(len(program.gates)):
        wrong.forward(owner, program, 1, gate_index, Work())
    last_index = len(program.gates) - 1
    wrong.exponent = apply_gate(
        wrong.values, wrong.scratch, wrong.exponent,
        Gate("FOURIER", wire=0, direction=program.gates[-1].direction), Work(),
    )
    wrong.cursor -= 1
    for gate_index in range(last_index - 1, -1, -1):
        wrong.inverse(owner, program, 1, gate_index, Work())
    wrong_inverse = rejected(lambda: wrong.release(owner, program, 1))

    stale = SyndromePort()
    run_transaction(stale, compile_program(0, 1), 1, owner)
    run_transaction(stale, compile_program(1, 1), 2, owner)
    stale_rejected = rejected(lambda: stale.lease(owner, program, 2))

    removed_x = tuple(gate for gate in program.gates if not (gate.kind == "SUM" and gate.wire == 1))
    removed_z = tuple(gate for gate in program.gates if gate.kind != "CZ")
    x_values, x_exp = execute_gates(removed_x)
    z_values, z_exp = execute_gates(removed_z)
    x_probability = probability_encoding(
        *selected_probability(x_values, x_exp, program.output_data)
    )
    z_probability = probability_encoding(
        *selected_probability(z_values, z_exp, program.output_data)
    )
    order_program = program
    order_values, order_exp = execute_gates(order_program.gates)
    swapped_consumers = list(order_program.gates)
    # In this declared control word, indices 9 and 10 are adjacent CZ and SUM
    # consumers of the same resident syndrome port.
    swapped_consumers[9], swapped_consumers[10] = (
        swapped_consumers[10], swapped_consumers[9]
    )
    swapped_values, swapped_exp = execute_gates(swapped_consumers)
    sham_gates = tuple(
        gate for gate in program.gates if gate.kind not in {"CUBIC", "Q", "L", "G"}
    )
    sham_values, sham_exp = execute_gates(sham_gates)
    sham_norm = ZERO
    for amplitude in sham_values:
        sham_norm = m237.k_add(
            sham_norm, m237.k_mul(amplitude, m237.k_conjugate(amplitude))
        )
    sham_norm_values = [sham_norm]
    sham_norm_exp = canonicalize(sham_norm_values, 2 * sham_exp)
    return {
        "injection_identity_all_data_inputs_a1_a2": identity,
        "q_correction_omission_changes_full_state": omission_changes["Q"],
        "l_correction_omission_changes_full_state": omission_changes["L"],
        "g_correction_omission_changes_full_state": omission_changes["G"],
        "g_omission_exactly_factorizes_data_magic_and_syndrome_magic": (
            omitted_g_values == expected_omitted_g and omitted_g_exponent == 1
        ),
        "complete_injection_compact_state_parity": baseline["final_state_commitment"] == vector_commitment(full_values, full_exponent),
        "complete_injection_compact_boundary_parity": baseline["selected_data_probability"] == coherent,
        "dephased_syndrome_changes_selected_boundary": dephased != coherent,
        "dephased_measurement_is_nonrestoring_diagnostic_only": True,
        "remove_sum_consumer_changes_full_state": vector_commitment(x_values, x_exp) != vector_commitment(full_values, full_exponent),
        "remove_cz_consumer_changes_full_state": vector_commitment(z_values, z_exp) != vector_commitment(full_values, full_exponent),
        "remove_sum_consumer_changes_selected_boundary": x_probability != coherent,
        "remove_cz_consumer_changes_selected_boundary": z_probability != coherent,
        "adjacent_sum_cz_consumer_order_changes_full_state": vector_commitment(
            swapped_values, swapped_exp
        ) != vector_commitment(order_values, order_exp),
        "stabilizer_sham_contains_only_clifford_gates": all(
            gate.kind in {"FOURIER", "SUM", "CZ"} for gate in sham_gates
        ),
        "stabilizer_sham_normalization_exact": (
            sham_norm_values[0] == ONE and sham_norm_exp == 0
        ),
        "premature_final_projection_rejected": premature_projection,
        "syndrome_projection_rejected": syndrome_projection,
        "wrong_owner_rejected": wrong_owner,
        "wrong_program_rejected": wrong_program,
        "wrong_generation_rejected": wrong_generation,
        "wrong_type_rejected": wrong_type,
        "same_id_changed_descriptor_rejected": forged,
        "duplicate_consumer_rejected": duplicate_consumer,
        "null_carrier_rejected": null,
        "dirty_scratch_rejected": dirty_rejected,
        "missing_inverse_rejected": missing_inverse,
        "reordered_inverse_rejected": reordered_inverse,
        "wrong_inverse_after_remaining_schedule_rejected": wrong_inverse,
        "stale_generation_rejected": stale_rejected,
        "syndrome_values_serialized": False,
        "accepted_in_place_path_retains_branch_assignment_expansion": False,
        "dephased_diagnostic_enumerates_five_syndrome_branches": True,
        "public_compiler_reads_final_answer": False,
    }


def comparable_case(case: dict[str, object]) -> dict[str, object]:
    return {
        key: case[key]
        for key in (
            "family", "cubic_strength", "consumer_pair_count", "syndrome_consumer_count",
            "selected_data_probability", "selected_data_probability_nonzero",
            "normalization_exact", "final_state_commitment",
            "canonical_post_inverse_state_exact", "same_amplitude_and_scratch_backings",
            "restoration_generation", "baseline_reload_used",
        )
    }


def main(reference_path: Path) -> None:
    reference = json.loads(reference_path.read_text())
    cases = [
        run_transaction(SyndromePort(), compile_program(family, pairs), 1)
        for family in (0, 1) for pairs in PAIR_COUNTS
    ]
    baselines = {
        (family, pairs): compiled_injection_identity_baseline(compile_program(family, pairs))
        for family in (0, 1) for pairs in PAIR_COUNTS
    }
    for case in cases:
        baseline = baselines[(case["family"], case["consumer_pair_count"])]
        if baseline["selected_data_probability"] != case["selected_data_probability"]:
            raise RuntimeError("compiled-injection boundary baseline disagrees")
        if baseline["final_state_commitment"] != case["final_state_commitment"]:
            raise RuntimeError("compiled-injection full-state baseline disagrees")
        case["matched_compiled_injection_baseline"] = baseline

    shared = SyndromePort()
    shared_backings = (id(shared.values), id(shared.scratch))
    primary = run_transaction(shared, compile_program(0, 4), 1)
    reuse = run_transaction(shared, compile_program(1, 2), 2)
    fresh = run_transaction(SyndromePort(), compile_program(1, 2), 1)
    control_values = controls()
    if [comparable_case(case) for case in cases] != reference["cases"]:
        raise RuntimeError("standalone 25-amplitude case parity failed")
    if control_values != reference["controls"]:
        raise RuntimeError("standalone controls differ")
    for key, case in (("primary", primary), ("reuse", reuse), ("fresh_reuse", fresh)):
        if comparable_case(case) != reference["reuse"][key]:
            raise RuntimeError("standalone reuse parity failed")

    output = {
        "schema": "cat_cas.zeta5_coherent_magic_injection_syndrome_port.v1",
        "result": RESULT,
        "claim": CLAIM,
        "claim_ceiling": CLAIM_CEILING,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "cases": cases,
        "controls": control_values,
        "reuse": {
            "primary": primary,
            "reuse": reuse,
            "fresh_reuse": fresh,
            "restoration_generation_after_reuse": shared.last_restored_generation,
            "fresh_restored_boundary_agreement": reuse["selected_data_probability"] == fresh["selected_data_probability"],
            "fresh_restored_full_state_commitment_agreement": (
                reuse["final_state_commitment"] == fresh["final_state_commitment"]
            ),
            "fresh_restored_resource_signature_agreement": (
                reuse["final_denominator_exponent"], reuse["final_total_exact_payload_bits"]
            ) == (
                fresh["final_denominator_exponent"], fresh["final_total_exact_payload_bits"]
            ),
            "same_backing_across_primary_and_reuse": (
                primary["same_amplitude_and_scratch_backings"]
                and reuse["same_amplitude_and_scratch_backings"]
                and shared_backings == (id(shared.values), id(shared.scratch))
            ),
        },
        "injection_law": {
            "field": "Q(zeta_5)",
            "coherent_identity": "G_a L_a Q_a SUM_D_TO_S^-1 (psi_D tensor M_a_S) = D3_a psi_D tensor plus_S",
            "syndrome_values": 5,
            "same_resident_syndrome_consumers": ["Q", "L", "G", "SUM", "CZ"],
            "fiberwise_corrections_are_stabilizer_for_fixed_syndrome": True,
            "coherent_q_l_g_maps_are_not_claimed_clifford": True,
            "physical_measurement_performed": False,
            "direct_process_logical_custody_only": True,
        },
        "matched_classical": {
            "strongest_implemented": "COMPLETE_INJECTION_IDENTITY_PLUS_EXACT25_AMPLITUDE_FIVE_INPUT_FOURIER_SCRATCH_RECURRENCE",
            "identical_full_state": "EXACT25_AMPLITUDE_RECURRENCE",
            "stabilizer_component_upper_bound": 5,
            "pairwise_overlap_upper_bound_for_ONE_BOUNDARY": 25,
            "stabilizer_component_resource_implementation": "NOT_INSTRUMENTED",
            "consumer_depth_increases_component_bound": False,
            "computational_advantage": False,
            "distinct_phase_resource": False,
        },
        "resource_law": {
            "resident_amplitude_field_cells": 25,
            "scratch_amplitude_field_cells": 25,
            "strongest_implemented_classical_resident_field_cells": 25,
            "strongest_implemented_classical_peak_fourier_saved_inputs": 5,
            "strongest_implemented_classical_peak_accumulator_field_cells": 1,
            "accepted_phase_backings_exceed_matched_classical_field_values": True,
            "retained_dynamic_inverse_history_entries": 0,
            "public_program_supplies_inverse_schedule": True,
            "all_coherent_nonclifford_corrections_counted": True,
            "dephased_diagnostic_sequential_branch_count": 5,
            "dephased_diagnostic_field_cells_per_branch": 25,
            "dephased_diagnostic_excluded_from_accepted_restoring_path": True,
            "whole_transaction_live_cell_and_payload_accounting_complete": False,
            "python_objects_allocator_hash_serialization_rss_excluded_not_zero": True,
            "resource_verification_level": "PACKAGE_SELF_REVIEW",
        },
        "separate_reference": {
            "imports_m239_or_m237_production": False,
            "independent_25_amplitude_recurrence": True,
            "independent_injection_identity": True,
            "independent_compiled_identity_25_amplitude_boundary": True,
            "independent_custody_state_machine": True,
        },
        "source_dependencies": {
            "production_sha256": file_sha256(Path(__file__)),
            "m237_algebra_sha256": file_sha256(Path(m237.__file__)),
            "separate_reference_sha256": file_sha256(
                Path(__file__).with_name("zeta5_coherent_magic_injection_syndrome_port_separate_reference.py")
            ),
        },
        "claim_limits": {
            "physical_measurement_or_general_measurement_theory": False,
            "optimal_magic_monotone_or_stabilizer_rank": False,
            "growing_compositional_magic_cost": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "catvm_machine_custody": False,
            "small_wall_crossed": False,
            "unbounded_catalytic_computation": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference": False,
        },
        "terminal": False,
    }
    print(json.dumps(output, sort_keys=True, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: zeta5_coherent_magic_injection_syndrome_port.py REFERENCE_JSON")
    main(Path(sys.argv[1]))
