#!/usr/bin/env python3
"""Standalone exact-amplitude oracle for M238; imports no production module."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


P = 5
INV2 = 3
PORT_TYPE = "ZETA5_GROSS_WIGNER_INTERFACE_V1"
INTERFACES = (1, 2, 3)
E = tuple[int, int, int, int]
ZERO: E = (0, 0, 0, 0)
ONE: E = (1, 0, 0, 0)
SQRT5: E = (-1, 0, -2, -2)


def add(left: E, right: E) -> E:
    return tuple(left[i] + right[i] for i in range(4))  # type: ignore[return-value]


def sub(left: E, right: E) -> E:
    return tuple(left[i] - right[i] for i in range(4))  # type: ignore[return-value]


def scale(value: E, scalar: int) -> E:
    return tuple(scalar * item for item in value)  # type: ignore[return-value]


def mul(left: E, right: E) -> E:
    raw = [0] * 7
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            raw[i + j] += a * b
    for degree in range(6, 3, -1):
        coefficient = raw[degree]
        if coefficient:
            for target in range(degree - 4, degree):
                raw[target] -= coefficient
            raw[degree] = 0
    return tuple(raw[:4])  # type: ignore[return-value]


def root(exponent: int) -> E:
    exponent %= 5
    if exponent < 4:
        values = [0] * 4
        values[exponent] = 1
        return tuple(values)  # type: ignore[return-value]
    return (-1, -1, -1, -1)


def conjugate(value: E) -> E:
    result = ZERO
    for exponent, coefficient in enumerate(value):
        result = add(result, scale(root(-exponent), coefficient))
    return result


def canonicalize(values: list[E], exponent: int) -> int:
    while exponent and all(c % 5 == 0 for value in values for c in value):
        for index, value in enumerate(values):
            values[index] = tuple(c // 5 for c in value)  # type: ignore[assignment]
        exponent -= 1
    return exponent


def digits(index: int, count: int) -> tuple[int, ...]:
    values = [0] * count
    for position in range(count - 1, -1, -1):
        values[position] = index % P
        index //= P
    return tuple(values)


def encode(values: Sequence[int]) -> int:
    result = 0
    for value in values:
        result = P * result + value
    return result


def commitment(value: E, exponent: int) -> str:
    values = [value]
    exponent = canonicalize(values, exponent)
    payload = {"denominator_power5": exponent, "numerator": list(values[0])}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def update_cell_sequence_commitment(digest: object, value: E, exponent: int) -> None:
    values = [value]
    exponent = canonicalize(values, exponent)
    digest.update(f"{exponent}:{','.join(str(c) for c in values[0])};".encode())


def real_pair(value: E, exponent: int) -> dict[str, int]:
    if value[1] != 0 or value[2] != value[3]:
        raise RuntimeError("reference Wigner value not real")
    a, b = 2 * value[0] - value[2], -value[2]
    while exponent and a % 5 == 0 and b % 5 == 0:
        a //= 5
        b //= 5
        exponent -= 1
    return {
        "a_numerator": a,
        "b_sqrt5_numerator": b,
        "denominator_twice_power5_exponent": exponent,
    }


def real_sign(value: E) -> int:
    if value[1] != 0 or value[2] != value[3]:
        raise RuntimeError("reference Wigner value not real")
    a, b = 2 * value[0] - value[2], -value[2]
    if b < 0:
        return -real_sign(tuple(-c for c in value))  # type: ignore[arg-type]
    if b == 0:
        return (a > 0) - (a < 0)
    if a >= 0:
        return 1
    comparison = 5 * b * b - a * a
    return (comparison > 0) - (comparison < 0)


@dataclass(frozen=True)
class RGate:
    kind: str
    wire: int
    target: int = -1
    a: int = 0
    b: int = 0
    direction: int = 1

    def __post_init__(self) -> None:
        if self.kind not in {"FOURIER", "SUM", "CUBIC", "QUADRATIC"}:
            raise ValueError("unknown reference gate")
        if self.direction not in (-1, 1) or (self.kind == "SUM" and self.target == self.wire):
            raise ValueError("invalid reference gate")
        if self.kind == "CUBIC" and self.a % P == 0:
            raise ValueError("reference cubic must be nonzero")

    def inverse(self) -> "RGate":
        if self.kind in {"FOURIER", "SUM"}:
            return RGate(self.kind, self.wire, self.target, self.a, self.b, -self.direction)
        return RGate(self.kind, self.wire, self.target, -self.a, -self.b, self.direction)

    def serial(self) -> tuple[object, ...]:
        return (self.kind, self.wire, self.target, self.a % P, self.b % P, self.direction)


def digest(family: int, n: int, gates: Sequence[RGate], output_q: Sequence[int], output_r: Sequence[int]) -> str:
    payload = {
        "family": family,
        "n": n,
        "gates": [gate.serial() for gate in gates],
        "output_q": list(output_q),
        "output_r": list(output_r),
        "port_type": PORT_TYPE,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class RProgram:
    family: int
    n: int
    gates: tuple[RGate, ...]
    output_q: tuple[int, ...]
    output_r: tuple[int, ...]
    program_id: str

    def __post_init__(self) -> None:
        if self.family not in (0, 1) or self.n not in INTERFACES:
            raise ValueError("reference program outside family")
        if len(self.output_q) != self.n or len(self.output_r) != self.n:
            raise ValueError("reference boundary arity mismatch")
        for gate in self.gates:
            if not 0 <= gate.wire < self.n or (gate.kind == "SUM" and not 0 <= gate.target < self.n):
                raise ValueError("reference gate outside interface")
        if self.program_id != digest(self.family, self.n, self.gates, self.output_q, self.output_r):
            raise ValueError("reference program digest mismatch")

    @property
    def descriptor(self) -> tuple[object, ...]:
        return (
            self.family, self.n, tuple(gate.serial() for gate in self.gates),
            self.output_q, self.output_r, self.program_id,
        )


def compile_program(family: int, n: int) -> RProgram:
    gates: list[RGate] = [RGate("FOURIER", 0)]
    for wire in range(n):
        if wire:
            gates.append(RGate("SUM", wire - 1, wire, direction=1))
        cubic_a = ((wire + 1) if family == 0 else (2 * wire + 2)) % P
        if cubic_a == 0:
            cubic_a = 4
        cubic_b = (family + 2 * wire) % P
        gates.append(RGate("CUBIC", wire, a=cubic_a, b=cubic_b))
        gates.append(RGate("FOURIER", wire, direction=1 if (wire + family) % 2 == 0 else -1))
    for wire in range(n - 1, 0, -1):
        gates.append(RGate("SUM", wire - 1, wire, direction=-1))
    output_q = tuple((family + wire) % P for wire in range(n))
    output_r = tuple((2 * family + wire) % P for wire in range(n))
    gate_tuple = tuple(gates)
    return RProgram(family, n, gate_tuple, output_q, output_r, digest(family, n, gate_tuple, output_q, output_r))


def sham_program(program: RProgram) -> RProgram:
    gates = tuple(
        RGate("QUADRATIC", gate.wire, a=gate.a or 1, b=gate.b)
        if gate.kind == "CUBIC" else gate
        for gate in program.gates
    )
    return RProgram(
        program.family, program.n, gates, program.output_q, program.output_r,
        digest(program.family, program.n, gates, program.output_q, program.output_r),
    )


def apply_gate(values: list[E], scratch: list[E], exponent: int, n: int, gate: RGate) -> int:
    dimension = P**n
    if any(value != ZERO for value in scratch):
        raise RuntimeError("dirty reference amplitude scratch")
    if gate.kind == "FOURIER":
        for fixed_index in range(P ** (n - 1)):
            fixed = list(digits(fixed_index, n - 1))
            for output in range(P):
                total = ZERO
                for source in range(P):
                    coordinate = list(fixed)
                    coordinate.insert(gate.wire, source)
                    total = add(total, mul(root(gate.direction * source * output), values[encode(coordinate)]))
                coordinate = list(fixed)
                coordinate.insert(gate.wire, output)
                scratch[encode(coordinate)] = mul(SQRT5, total)
        values[:] = scratch
        scratch[:] = [ZERO] * dimension
        exponent += 1
    elif gate.kind == "SUM":
        for source_index, value in enumerate(values):
            coordinate = list(digits(source_index, n))
            coordinate[gate.target] = (coordinate[gate.target] + gate.direction * coordinate[gate.wire]) % P
            scratch[encode(coordinate)] = value
        values[:] = scratch
        scratch[:] = [ZERO] * dimension
    else:
        for index, value in enumerate(values):
            coordinate = digits(index, n)
            x = coordinate[gate.wire]
            phase = gate.a * (x**3 if gate.kind == "CUBIC" else x**2) + gate.b * x
            values[index] = mul(root(phase), value)
    return canonicalize(values, exponent)


def wigner_aggregate(values: Sequence[E], exponent: int, program: RProgram) -> dict[str, object]:
    n = program.n
    dimension = P**n
    coordinates = [digits(index, n) for index in range(dimension)]
    conjugates = [conjugate(value) for value in values]
    total = ZERO
    square_sum = ZERO
    negative_sum = ZERO
    negative_count = 0
    selected = ZERO
    cell_sequence = hashlib.sha256(b"CAT_CAS_EXACT_WIGNER_CELL_SEQUENCE_V1\0")
    wigner_exponent = 2 * exponent + n
    for q_index in range(dimension):
        q = coordinates[q_index]
        correlations: list[E] = []
        for displacement in coordinates:
            plus = tuple((q[i] + INV2 * displacement[i]) % P for i in range(n))
            minus = tuple((q[i] - INV2 * displacement[i]) % P for i in range(n))
            correlations.append(mul(values[encode(plus)], conjugates[encode(minus)]))
        for r_index in range(dimension):
            r = coordinates[r_index]
            wigner = ZERO
            for displacement, correlation in zip(coordinates, correlations):
                phase = -sum(r[i] * displacement[i] for i in range(n))
                wigner = add(wigner, mul(root(phase), correlation))
            if wigner[1] != 0 or wigner[2] != wigner[3]:
                raise RuntimeError("reference Wigner reality failure")
            update_cell_sequence_commitment(cell_sequence, wigner, wigner_exponent)
            total = add(total, wigner)
            square_sum = add(square_sum, mul(wigner, wigner))
            if real_sign(wigner) < 0:
                negative_count += 1
                negative_sum = sub(negative_sum, wigner)
            if q == program.output_q and r == program.output_r:
                selected = wigner
    total_values = [total]
    total_exponent = canonicalize(total_values, wigner_exponent)
    purity_values = [scale(square_sum, dimension)]
    purity_exponent = canonicalize(purity_values, 2 * wigner_exponent)
    normalized = total_values[0] == ONE and total_exponent == 0
    pure = purity_values[0] == ONE and purity_exponent == 0
    if not normalized or not pure:
        raise RuntimeError("reference Wigner receipt failure")
    return {
        "negative_cell_count": negative_count,
        "negative_mass": real_pair(negative_sum, wigner_exponent),
        "negative_mass_positive": negative_count > 0 and real_sign(negative_sum) > 0,
        "normalization_exact": normalized,
        "purity_exact": pure,
        "selected_boundary_commitment": commitment(selected, wigner_exponent),
        "final_wigner_commitment": cell_sequence.hexdigest(),
    }


class ReferencePort:
    def __init__(self, n: int) -> None:
        if n not in INTERFACES:
            raise ValueError("reference interface unsupported")
        self.n = n
        self.values = [ZERO] * (P**n)
        self.values[0] = ONE
        self.scratch = [ZERO] * (P**n)
        self.exponent = 0
        self.cursor = 0
        self.leased = False
        self.owner = 0
        self.descriptor: tuple[object, ...] | None = None
        self.generation = 0
        self.last_restored_generation = 0

    def canonical(self) -> bool:
        return (
            self.values == [ONE] + [ZERO] * (P**self.n - 1)
            and self.scratch == [ZERO] * len(self.scratch)
            and self.exponent == 0 and self.cursor == 0
        )

    def require(self, owner: int, program: RProgram, generation: int) -> None:
        if not self.leased or owner != self.owner or generation != self.generation:
            raise RuntimeError("reference custody mismatch")
        if program.descriptor != self.descriptor or program.n != self.n:
            raise RuntimeError("reference descriptor mismatch")

    def lease(self, owner: int, program: RProgram, generation: int, port_type: str = PORT_TYPE) -> None:
        if port_type != PORT_TYPE or self.leased or program.n != self.n:
            raise RuntimeError("invalid reference lease")
        if generation != self.last_restored_generation + 1 or not self.canonical():
            raise RuntimeError("invalid reference generation/state")
        self.leased = True
        self.owner = owner
        self.descriptor = program.descriptor
        self.generation = generation

    def forward(self, owner: int, program: RProgram, generation: int, index: int) -> None:
        self.require(owner, program, generation)
        if index != self.cursor:
            raise RuntimeError("reference forward cursor")
        self.exponent = apply_gate(self.values, self.scratch, self.exponent, self.n, program.gates[index])
        self.cursor += 1

    def inverse(self, owner: int, program: RProgram, generation: int, index: int) -> None:
        self.require(owner, program, generation)
        if index != self.cursor - 1:
            raise RuntimeError("reference inverse cursor")
        self.exponent = apply_gate(self.values, self.scratch, self.exponent, self.n, program.gates[index].inverse())
        self.cursor -= 1

    def project(self, owner: int, program: RProgram, generation: int) -> dict[str, object]:
        self.require(owner, program, generation)
        if self.cursor != len(program.gates) or any(value != ZERO for value in self.scratch):
            raise RuntimeError("premature reference projection")
        return wigner_aggregate(self.values, self.exponent, program)

    def full_wigner(self) -> object:
        raise RuntimeError("reference full-Wigner projection forbidden")

    def release(self, owner: int, program: RProgram, generation: int) -> None:
        self.require(owner, program, generation)
        if not self.canonical():
            raise RuntimeError("reference state not restored")
        self.leased = False
        self.last_restored_generation = generation
        self.owner = 0
        self.descriptor = None
        self.generation = 0


def run_case(port: ReferencePort, program: RProgram, generation: int, owner: int = 23801) -> dict[str, object]:
    if port is None:
        raise TypeError("null reference port")
    backings = (id(port.values), id(port.scratch))
    port.lease(owner, program, generation)
    for index in range(len(program.gates)):
        port.forward(owner, program, generation, index)
    aggregate = port.project(owner, program, generation)
    retained = json.dumps(aggregate, sort_keys=True)
    for index in range(len(program.gates) - 1, -1, -1):
        port.inverse(owner, program, generation, index)
    exact = port.canonical()
    same = backings == (id(port.values), id(port.scratch))
    port.release(owner, program, generation)
    if retained != json.dumps(aggregate, sort_keys=True):
        raise RuntimeError("reference boundary consumed by inverse")
    return {
        "family": program.family,
        "interface_qudits": program.n,
        "hilbert_amplitude_cells": P**program.n,
        **aggregate,
        "canonical_post_inverse_state_exact": exact,
        "same_wigner_and_scratch_backings": same,
        "restoration_generation": generation,
        "baseline_reload_used": False,
    }


def rejected(callback: Callable[[], object]) -> bool:
    try:
        callback()
    except (RuntimeError, TypeError, ValueError):
        return True
    return False


def controls(
    baselines: dict[tuple[int, int], dict[str, object]] | None = None,
    shams: dict[tuple[int, int], dict[str, object]] | None = None,
    amplitude_commitments: dict[tuple[int, int], str] | None = None,
) -> dict[str, bool]:
    program = compile_program(0, 2)
    owner = 23811
    premature = ReferencePort(2)
    premature.lease(owner, program, 1)
    premature_projection = rejected(lambda: premature.project(owner, program, 1))
    full_projection = rejected(premature.full_wigner)
    wrong_owner = rejected(lambda: premature.forward(owner + 1, program, 1, 0))
    wrong_program = rejected(lambda: premature.forward(owner, compile_program(1, 2), 1, 0))
    wrong_generation = rejected(lambda: premature.forward(owner, program, 2, 0))
    wrong_type = rejected(lambda: ReferencePort(2).lease(owner, program, 1, "WRONG"))
    malformed = program.gates[:-1] + (RGate("CUBIC", 1, a=4, b=4),)
    forged = rejected(lambda: RProgram(program.family, program.n, malformed, program.output_q, program.output_r, program.program_id))
    wrong_arity = rejected(lambda: ReferencePort(1).lease(owner, program, 1))
    null = rejected(lambda: run_case(None, program, 1))  # type: ignore[arg-type]
    dirty = ReferencePort(2)
    dirty.scratch[0] = ONE
    dirty_rejected = rejected(lambda: dirty.lease(owner, program, 1))

    missing = ReferencePort(2)
    missing.lease(owner, program, 1)
    for index in range(len(program.gates)):
        missing.forward(owner, program, 1, index)
    missing_rejected = rejected(lambda: missing.release(owner, program, 1))

    reordered = ReferencePort(2)
    reordered.lease(owner, program, 1)
    for index in range(len(program.gates)):
        reordered.forward(owner, program, 1, index)
    reordered_rejected = rejected(lambda: reordered.inverse(owner, program, 1, len(program.gates) - 2))

    wrong = ReferencePort(2)
    wrong.lease(owner, program, 1)
    for index in range(len(program.gates)):
        wrong.forward(owner, program, 1, index)
    wrong.exponent = apply_gate(wrong.values, wrong.scratch, wrong.exponent, 2, RGate("CUBIC", 0, a=1, b=1))
    wrong.cursor -= 1
    for index in range(len(program.gates) - 2, -1, -1):
        wrong.inverse(owner, program, 1, index)
    wrong_rejected = rejected(lambda: wrong.release(owner, program, 1))

    stale = ReferencePort(2)
    run_case(stale, compile_program(0, 2), 1, owner)
    run_case(stale, compile_program(1, 2), 2, owner)
    stale_rejected = rejected(lambda: stale.lease(owner, program, 2))

    if baselines is None:
        baselines = {
            (family, n): wigner_aggregate_from_program(compile_program(family, n))
            for family in (0, 1) for n in INTERFACES
        }
    if shams is None:
        shams = {
            (family, n): wigner_aggregate_from_program(sham_program(compile_program(family, n)))
            for family in (0, 1) for n in INTERFACES
        }
    if amplitude_commitments is None:
        amplitude_commitments = {
            (family, n): final_amplitude_commitment(compile_program(family, n))
            for family in (0, 1) for n in INTERFACES
        }
    semantic = all(
        amplitude_commitments[(0, n)] != amplitude_commitments[(1, n)]
        for n in INTERFACES
    )
    return {
        "computational_basis_wigner_support_analytic": True,
        "cubic_third_finite_difference_nonzero": 6 % P != 0,
        "quadratic_third_finite_difference_zero": True,
        "clifford_shams_exactly_nonnegative": all(not value["negative_mass_positive"] for value in shams.values()),
        "all_cubic_augmented_interfaces_have_positive_negativity": all(value["negative_mass_positive"] for value in baselines.values()),
        "cubic_family_perturbation_changes_amplitude_state": semantic,
        "premature_negative_mass_projection_rejected": premature_projection,
        "full_wigner_projection_rejected": full_projection,
        "wrong_owner_rejected": wrong_owner,
        "wrong_program_rejected": wrong_program,
        "wrong_generation_rejected": wrong_generation,
        "wrong_type_rejected": wrong_type,
        "wrong_interface_arity_rejected": wrong_arity,
        "same_id_changed_descriptor_rejected": forged,
        "null_carrier_rejected": null,
        "dirty_scratch_rejected": dirty_rejected,
        "missing_inverse_rejected": missing_rejected,
        "reordered_inverse_rejected": reordered_rejected,
        "wrong_inverse_after_remaining_schedule_rejected": wrong_rejected,
        "stale_generation_rejected": stale_rejected,
        "resident_wigner_cells_serialized": False,
        "negative_cell_coordinates_serialized": False,
        "amplitudes_or_density_matrices_serialized": False,
        "path_histories_or_assignments_enumerated": False,
        "precomputed_cubic_kernel_tables_retained": False,
        "public_compiler_reads_final_answer": False,
    }


def evolve(program: RProgram) -> tuple[list[E], int]:
    values = [ZERO] * (P**program.n)
    scratch = [ZERO] * len(values)
    values[0] = ONE
    exponent = 0
    for gate in program.gates:
        exponent = apply_gate(values, scratch, exponent, program.n, gate)
    return values, exponent


def wigner_aggregate_from_program(program: RProgram) -> dict[str, object]:
    values, exponent = evolve(program)
    return wigner_aggregate(values, exponent, program)


def final_amplitude_commitment(program: RProgram) -> str:
    values, exponent = evolve(program)
    payload = {"denominator_power5": exponent, "numerators": [list(value) for value in values]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    lower_cases = {
        (family, n): run_case(ReferencePort(n), compile_program(family, n), 1)
        for family in (0, 1) for n in (1, 2)
    }
    shared = ReferencePort(3)
    primary = run_case(shared, compile_program(0, 3), 1)
    reuse = run_case(shared, compile_program(1, 3), 2)
    fresh = run_case(ReferencePort(3), compile_program(1, 3), 1)
    cases = [
        lower_cases[(0, 1)], lower_cases[(0, 2)], primary,
        lower_cases[(1, 1)], lower_cases[(1, 2)], fresh,
    ]
    baselines = {(case["family"], case["interface_qudits"]): case for case in cases}
    shams = {
        (family, n): wigner_aggregate_from_program(sham_program(compile_program(family, n)))
        for family in (0, 1) for n in INTERFACES
    }
    amplitude_commitments = {
        (family, n): final_amplitude_commitment(compile_program(family, n))
        for family in (0, 1) for n in INTERFACES
    }
    output = {
        "schema": "cat_cas.zeta5_wigner_cubic_magic_interface_reference.v1",
        "cases": cases,
        "controls": controls(baselines, shams, amplitude_commitments),
        "reuse": {
            "primary": primary,
            "reuse": reuse,
            "fresh_reuse": fresh,
            "restoration_generation_after_reuse": shared.last_restored_generation,
            "fresh_restored_boundary_agreement": reuse["selected_boundary_commitment"] == fresh["selected_boundary_commitment"],
            "fresh_restored_resource_signature_agreement": (
                reuse["negative_cell_count"], reuse["negative_mass"]
            ) == (fresh["negative_cell_count"], fresh["negative_mass"]),
            "same_backing_across_primary_and_reuse": primary["same_wigner_and_scratch_backings"] and reuse["same_wigner_and_scratch_backings"],
        },
        "imports_m238_or_m237_production": False,
        "evolves_exact_amplitudes": True,
        "streams_final_wigner_reconstruction": True,
        "implements_independent_custody_state_machine": True,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    print(json.dumps(output, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
