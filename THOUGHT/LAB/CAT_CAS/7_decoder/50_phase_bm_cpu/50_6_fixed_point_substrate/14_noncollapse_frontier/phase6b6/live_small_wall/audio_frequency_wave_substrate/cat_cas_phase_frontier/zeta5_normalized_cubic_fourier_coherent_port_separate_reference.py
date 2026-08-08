#!/usr/bin/env python3
"""Standalone exact matrix oracle for M237; imports no production source."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Sequence


E = tuple[int, int, int, int]
Z: E = (0, 0, 0, 0)
O: E = (1, 0, 0, 0)
S: E = (-1, 0, -2, -2)
TYPE = "ZETA5_NORMALIZED_COHERENT_PORT_V1"
DEPTHS = (2, 4, 8, 16, 32, 64)


def eadd(left: E, right: E) -> E:
    return tuple(left[i] + right[i] for i in range(4))  # type: ignore[return-value]


def emul(left: E, right: E) -> E:
    raw = [0] * 7
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            raw[i + j] += a * b
    for power in range(6, 3, -1):
        value = raw[power]
        for lower in range(power - 4, power):
            raw[lower] -= value
        raw[power] = 0
    return tuple(raw[:4])  # type: ignore[return-value]


def root(power: int) -> E:
    power %= 5
    if power == 4:
        return (-1, -1, -1, -1)
    result = [0] * 4
    result[power] = 1
    return tuple(result)  # type: ignore[return-value]


def conjugate(value: E) -> E:
    result = Z
    for power, coefficient in enumerate(value):
        term = tuple(coefficient * x for x in root(-power))
        result = eadd(result, term)  # type: ignore[arg-type]
    return result


def bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def normalize(vector: list[E], exponent: int) -> int:
    while exponent and all(c % 5 == 0 for value in vector for c in value):
        for i, value in enumerate(vector):
            vector[i] = tuple(c // 5 for c in value)  # type: ignore[assignment]
        exponent -= 1
    return exponent


def normalize_element(value: E, exponent: int) -> tuple[E, int]:
    result = value
    while exponent and all(c % 5 == 0 for c in result):
        result = tuple(c // 5 for c in result)  # type: ignore[assignment]
        exponent -= 1
    return result, exponent


def commitment(value: E, exponent: int) -> str:
    return hashlib.sha256(json.dumps([list(value), exponent], separators=(",", ":")).encode()).hexdigest()


def vector_commitment(vector: Sequence[E], exponent: int) -> str:
    return hashlib.sha256(json.dumps([[list(x) for x in vector], exponent], separators=(",", ":")).encode()).hexdigest()


def vector_payload(vector: Sequence[E], exponent: int) -> int:
    return sum(bits(c) for value in vector for c in value) + bits(5**exponent)


def probability(value: E, exponent: int) -> tuple[E, int]:
    return normalize_element(emul(value, conjugate(value)), 2 * exponent)


def rational_probability(value: E, exponent: int) -> list[int] | None:
    product, power = probability(value, exponent)
    if any(product[i] for i in range(1, 4)):
        return None
    return [product[0], 5**power]


def norm(vector: Sequence[E], exponent: int) -> tuple[int, int]:
    result = Z
    for value in vector:
        result = eadd(result, emul(value, conjugate(value)))
    if any(result[i] for i in range(1, 4)):
        raise RuntimeError("reference norm is not rational")
    numerator, denominator = result[0], 5 ** (2 * exponent)
    while denominator > 1 and numerator % 5 == 0:
        numerator //= 5
        denominator //= 5
    return numerator, denominator


@dataclass(frozen=True)
class RGate:
    kind: str
    a: int = 0
    b: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("FOURIER", "CUBIC_PHASE"):
            raise ValueError("bad reference gate")
        if self.kind == "FOURIER" and (self.a or self.b):
            raise ValueError("bad Fourier parameters")
        if self.kind == "CUBIC_PHASE" and self.a % 5 == 0:
            raise ValueError("zero cubic coefficient")


def digest(family: int, depth: int, output: int, gates: Sequence[RGate]) -> str:
    data = [[gate.kind, gate.a, gate.b] for gate in gates]
    return hashlib.sha256(json.dumps([family, depth, output, data], separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class RProgram:
    family: int
    depth: int
    output: int
    gates: tuple[RGate, ...]
    program_id: str

    def __post_init__(self) -> None:
        if self.family not in (0, 1) or self.depth <= 0 or self.output not in range(5):
            raise ValueError("bad reference program")
        if self.program_id != digest(self.family, self.depth, self.output, self.gates):
            raise ValueError("reference program identity is not bound")


def make_program(family: int, depth: int, declared: bool = True) -> RProgram:
    if family not in (0, 1) or depth <= 0 or (declared and depth not in DEPTHS):
        raise ValueError("bad reference family/depth")
    gates = [RGate("FOURIER")]
    for i in range(depth):
        if family == 0:
            a, b = 1 + i % 4, (i * i + i + 1) % 5
        else:
            a, b = 1 + ((2 * i + 1) % 4), (3 * i * i + 2 * i + 2) % 5
        gates += [RGate("CUBIC_PHASE", a, b), RGate("FOURIER")]
    result = tuple(gates)
    output = 0 if family == 0 else 2
    return RProgram(family, depth, output, result, digest(family, depth, output, result))


def matrix_for(gate: RGate, inverse: bool) -> tuple[list[list[E]], int]:
    matrix = [[Z for _ in range(5)] for _ in range(5)]
    if gate.kind == "CUBIC_PHASE":
        for x in range(5):
            phase = (gate.a * x**3 + gate.b * x) * (-1 if inverse else 1)
            matrix[x][x] = root(phase)
        return matrix, 0
    sign = -1 if inverse else 1
    for y in range(5):
        for x in range(5):
            matrix[y][x] = emul(S, root(sign * x * y))
    return matrix, 1


def matrix_apply(vector: list[E], exponent: int, gate: RGate, inverse: bool) -> int:
    matrix, matrix_exponent = matrix_for(gate, inverse)
    result = [Z] * 5
    for row in range(5):
        value = Z
        for column in range(5):
            value = eadd(value, emul(matrix[row][column], vector[column]))
        result[row] = value
    for index in range(5):
        vector[index] = result[index]
    exponent = normalize(vector, exponent + matrix_exponent)
    if norm(vector, exponent) != (1, 1):
        raise RuntimeError("reference matrix changed norm")
    return exponent


class RPort:
    def __init__(self) -> None:
        self.vector = [O, Z, Z, Z, Z]
        self.scratch = [Z] * 5
        self.exponent = 0
        self.owner: int | None = None
        self.program_id: str | None = None
        self.descriptor: tuple[object, ...] | None = None
        self.port_type: str | None = None
        self.generation = 0
        self.last = 0
        self.cursor = 0
        self.leased = False

    @staticmethod
    def describe(program: RProgram) -> tuple[object, ...]:
        return (program.family, program.depth, program.output, tuple((g.kind, g.a, g.b) for g in program.gates))

    def canonical(self) -> bool:
        return self.vector == [O, Z, Z, Z, Z] and self.scratch == [Z] * 5 and self.exponent == 0 and self.cursor == 0

    def lease(self, owner: int, program: RProgram, generation: int, port_type: str = TYPE) -> None:
        if self.leased or not self.canonical() or owner <= 0 or port_type != TYPE or generation != self.last + 1:
            raise RuntimeError("reference lease rejected")
        self.owner, self.program_id, self.descriptor = owner, program.program_id, self.describe(program)
        self.port_type, self.generation, self.leased = port_type, generation, True

    def require(self, owner: int, program: RProgram, generation: int) -> None:
        if not self.leased or (owner, program.program_id, self.describe(program), TYPE, generation) != (
            self.owner, self.program_id, self.descriptor, self.port_type, self.generation
        ):
            raise RuntimeError("reference custody rejected")

    def forward(self, owner: int, program: RProgram, generation: int, index: int) -> None:
        self.require(owner, program, generation)
        if index != self.cursor or self.scratch != [Z] * 5:
            raise RuntimeError("reference forward order")
        self.exponent = matrix_apply(self.vector, self.exponent, program.gates[index], False)
        self.cursor += 1

    def inverse(self, owner: int, program: RProgram, generation: int, index: int) -> None:
        self.require(owner, program, generation)
        if index != self.cursor - 1 or self.scratch != [Z] * 5:
            raise RuntimeError("reference inverse order")
        self.exponent = matrix_apply(self.vector, self.exponent, program.gates[index], True)
        self.cursor -= 1

    def project(self, owner: int, program: RProgram, generation: int) -> tuple[E, int]:
        self.require(owner, program, generation)
        if self.cursor != len(program.gates) or self.scratch != [Z] * 5:
            raise RuntimeError("reference premature projection")
        return normalize_element(self.vector[program.output], self.exponent)

    def full(self) -> None:
        raise RuntimeError("reference full vector forbidden")

    def release(self, owner: int, program: RProgram, generation: int) -> None:
        self.require(owner, program, generation)
        if not self.canonical():
            raise RuntimeError("reference not restored")
        self.last = generation
        self.owner = self.program_id = self.descriptor = self.port_type = None
        self.generation, self.leased = 0, False


def direct(program: RProgram) -> dict[str, object]:
    vector = [O, Z, Z, Z, Z]
    exponent = 0
    for gate in program.gates:
        exponent = matrix_apply(vector, exponent, gate, False)
    amplitude, amp_exp = normalize_element(vector[program.output], exponent)
    prob, prob_exp = probability(amplitude, amp_exp)
    return {
        "amplitude_commitment": commitment(amplitude, amp_exp),
        "probability_commitment": commitment(prob, prob_exp),
        "probability_rational": rational_probability(amplitude, amp_exp),
        "final_vector_commitment": vector_commitment(vector, exponent),
        "final_denominator_exponent": exponent,
        "final_numerator_maximum_signed_bits": max(bits(c) for v in vector for c in v),
        "final_total_exact_payload_bits": vector_payload(vector, exponent),
    }


def run_case(port: RPort, program: RProgram, generation: int, owner: int = 23701) -> dict[str, object]:
    if port is None:
        raise TypeError("null reference port")
    backings = (id(port.vector), id(port.scratch))
    port.lease(owner, program, generation)
    for index in range(len(program.gates)):
        port.forward(owner, program, generation, index)
    final_vector = vector_commitment(port.vector, port.exponent)
    final_exp = port.exponent
    final_bits = max(bits(c) for v in port.vector for c in v)
    final_payload = vector_payload(port.vector, port.exponent)
    amplitude, amp_exp = port.project(owner, program, generation)
    retained_boundary = (amplitude, amp_exp)
    for index in range(len(program.gates) - 1, -1, -1):
        port.inverse(owner, program, generation, index)
    exact = port.canonical()
    same = backings == (id(port.vector), id(port.scratch))
    port.release(owner, program, generation)
    if retained_boundary != (amplitude, amp_exp):
        raise RuntimeError("reference final result was consumed by inverse")
    prob, prob_exp = probability(amplitude, amp_exp)
    amp_commit = commitment(amplitude, amp_exp)
    prob_commit = commitment(prob, prob_exp)
    rational = rational_probability(amplitude, amp_exp)
    baseline = direct(program)
    if baseline["final_vector_commitment"] != final_vector or baseline["amplitude_commitment"] != amp_commit:
        raise RuntimeError("reference post-release baseline mismatch")
    return {
        "family": program.family,
        "depth": program.depth,
        "gate_count": len(program.gates),
        "logical_amplitude_cells": 5,
        "resident_integer_numerator_coordinates": 20,
        "scratch_integer_numerator_coordinates": 20,
        "final_denominator_exponent": final_exp,
        "final_denominator_material_bits": bits(5**final_exp),
        "final_numerator_maximum_signed_bits": final_bits,
        "final_total_exact_payload_bits": final_payload,
        "final_vector_commitment": final_vector,
        "amplitude_commitment": amp_commit,
        "probability_commitment": prob_commit,
        "probability_rational": rational,
        "retained_final_amplitude_integer_coordinates_during_inverse": 4,
        "retained_final_amplitude_denominator_exponents_during_inverse": 1,
        "retained_final_amplitude_denominator_material_bits_during_inverse": bits(5**amp_exp),
        "retained_one_way_vector_commitment_digest_bytes_during_inverse": 32,
        "canonical_post_inverse_state_exact": exact,
        "same_psi_and_scratch_backings": same,
        "restoration_generation": generation,
        "baseline_reload_used": False,
    }


def rejected(callback: Callable[[], object]) -> bool:
    try:
        callback()
    except (RuntimeError, TypeError, ValueError):
        return True
    return False


def basis_identity(index: int) -> bool:
    vector = [Z] * 5
    vector[index] = O
    exponent = matrix_apply(vector, 0, RGate("FOURIER"), False)
    exponent = matrix_apply(vector, exponent, RGate("FOURIER"), True)
    return exponent == 0 and vector == [O if i == index else Z for i in range(5)]


def dephased_probability() -> Fraction:
    vector = [O, Z, Z, Z, Z]
    exponent = matrix_apply(vector, 0, RGate("FOURIER"), False)
    weights = []
    for value in vector:
        exact = rational_probability(value, exponent)
        if exact is None:
            raise RuntimeError("reference dephased weight is not rational")
        weights.append(Fraction(exact[0], exact[1]))
    transition = rational_probability(S, 1)
    if transition is None:
        raise RuntimeError("reference Fourier transition is not rational")
    transition_weight = Fraction(transition[0], transition[1])
    return sum((weight * transition_weight for weight in weights), Fraction(0))


def controls() -> dict[str, bool]:
    witness_gates = (RGate("FOURIER"), RGate("CUBIC_PHASE", 1, 0), RGate("FOURIER"))
    witness = RProgram(0, 1, 0, witness_gates, digest(0, 1, 0, witness_gates))
    identity_gates = (RGate("FOURIER"), RGate("FOURIER"))
    identity = RProgram(0, 1, 0, identity_gates, digest(0, 1, 0, identity_gates))
    generic = [Z, O, Z, Z, Z]
    left = list(generic)
    eleft = matrix_apply(left, 0, RGate("FOURIER"), False)
    eleft = matrix_apply(left, eleft, RGate("CUBIC_PHASE", 1, 0), False)
    right = list(generic)
    eright = matrix_apply(right, 0, RGate("CUBIC_PHASE", 1, 0), False)
    eright = matrix_apply(right, eright, RGate("FOURIER"), False)
    program = make_program(0, 2)
    owner = 23711
    port = RPort(); port.lease(owner, program, 1)
    premature = rejected(lambda: port.project(owner, program, 1))
    full = rejected(port.full)
    wrong_owner = rejected(lambda: port.forward(owner + 1, program, 1, 0))
    wrong_program = rejected(lambda: port.forward(owner, make_program(1, 2), 1, 0))
    wrong_generation = rejected(lambda: port.forward(owner, program, 2, 0))
    wrong_type = rejected(lambda: RPort().lease(owner, program, 1, "WRONG"))
    forged = rejected(lambda: RProgram(program.family, program.depth, program.output, program.gates[:-1] + (RGate("CUBIC_PHASE", 2, 1),), program.program_id))
    null = rejected(lambda: run_case(None, program, 1))  # type: ignore[arg-type]
    dirty = RPort(); dirty.scratch[0] = O
    dirty_rejected = rejected(lambda: dirty.lease(owner, program, 1))
    missing = RPort(); missing.lease(owner, program, 1)
    for i in range(len(program.gates)): missing.forward(owner, program, 1, i)
    missing_rejected = rejected(lambda: missing.release(owner, program, 1))
    reordered = RPort(); reordered.lease(owner, program, 1)
    for i in range(len(program.gates)): reordered.forward(owner, program, 1, i)
    reorder_rejected = rejected(lambda: reordered.inverse(owner, program, 1, len(program.gates) - 2))
    wrong = RPort(); wrong.lease(owner, program, 1)
    for i in range(len(program.gates)): wrong.forward(owner, program, 1, i)
    wrong.exponent = matrix_apply(wrong.vector, wrong.exponent, RGate("CUBIC_PHASE", 1, 0), True); wrong.cursor -= 1
    for i in range(len(program.gates) - 2, -1, -1):
        wrong.inverse(owner, program, 1, i)
    wrong_rejected = rejected(lambda: wrong.release(owner, program, 1))
    stale = RPort(); run_case(stale, make_program(0, 2), 1, owner); run_case(stale, make_program(1, 2), 2, owner)
    stale_rejected = rejected(lambda: stale.lease(owner, program, 2))
    phase = emul(root(1), O)
    return {
        "cyclotomic_relation": eadd(eadd(eadd(eadd(O, root(1)), root(2)), root(3)), root(4)) == Z,
        "zeta_fifth_power": emul(root(4), root(1)) == O,
        "sqrt5_square": emul(S, S) == (5, 0, 0, 0),
        "conjugation": emul(root(1), conjugate(root(1))) == O,
        "fourier_dagger_fourier_identity": all(basis_identity(i) for i in range(5)),
        "cubic_phase_unitary_identity": all(emul(root(x**3), root(-x**3)) == O for x in range(5)),
        "cubic_third_finite_difference_nonzero": 6 % 5 != 0,
        "fourier_and_cubic_phase_noncommute": (left, eleft) != (right, eright),
        "exact_cubic_destructive_interference_zero": direct(witness)["probability_rational"] == [0, 1],
        "dephased_control_probability_one_over5": dephased_probability() == Fraction(1, 5),
        "identity_phase_control_probability_one": direct(identity)["probability_rational"] == [1, 1],
        "global_phase_retained_in_amplitude": phase != O,
        "global_phase_cancels_only_in_probability": rational_probability(phase, 0) == [1, 1],
        "premature_final_projection_rejected": premature,
        "full_vector_projection_rejected": full,
        "wrong_owner_rejected": wrong_owner,
        "wrong_program_rejected": wrong_program,
        "wrong_generation_rejected": wrong_generation,
        "wrong_type_rejected": wrong_type,
        "same_id_changed_descriptor_rejected": forged,
        "null_carrier_rejected": null,
        "dirty_scratch_rejected": dirty_rejected,
        "missing_inverse_rejected": missing_rejected,
        "reordered_inverse_rejected": reorder_rejected,
        "wrong_inverse_rejected": wrong_rejected,
        "stale_generation_rejected": stale_rejected,
        "intermediate_or_full_vectors_serialized": False,
        "path_histories_or_assignments_enumerated": False,
        "public_compiler_reads_final_answer": False,
    }


def main() -> None:
    cases = [run_case(RPort(), make_program(family, depth), 1) for family in (0, 1) for depth in DEPTHS]
    primary_program, reuse_program = make_program(0, 32), make_program(1, 16)
    shared = RPort(); primary = run_case(shared, primary_program, 1); reuse = run_case(shared, reuse_program, 2)
    fresh = run_case(RPort(), reuse_program, 1)
    output = {
        "schema": "cat_cas.zeta5_normalized_cubic_fourier_coherent_port_reference.v1",
        "cases": cases,
        "controls": controls(),
        "reuse": {
            "primary": primary, "reuse": reuse, "fresh_reuse": fresh,
            "restoration_generation_after_reuse": shared.last,
            "fresh_restored_boundary_agreement": reuse["amplitude_commitment"] == fresh["amplitude_commitment"] and reuse["probability_commitment"] == fresh["probability_commitment"],
            "fresh_restored_resource_signature_agreement": reuse["final_denominator_exponent"] == fresh["final_denominator_exponent"] and reuse["final_total_exact_payload_bits"] == fresh["final_total_exact_payload_bits"],
            "same_backing_across_primary_and_reuse": primary["same_psi_and_scratch_backings"] and reuse["same_psi_and_scratch_backings"],
        },
        "imports_m237_production": False,
        "uses_independent_exact_matrix_recurrence": True,
        "implements_independent_custody_state_machine": True,
        "source_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
    }
    print(json.dumps(output, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
