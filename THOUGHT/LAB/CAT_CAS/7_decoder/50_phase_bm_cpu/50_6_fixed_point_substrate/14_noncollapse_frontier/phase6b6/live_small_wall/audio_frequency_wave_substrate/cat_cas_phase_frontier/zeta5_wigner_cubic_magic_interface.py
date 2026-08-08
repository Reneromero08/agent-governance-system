#!/usr/bin/env python3
"""M238: exact p=5 Wigner magic diagnostic on n=1,2,3 interfaces.

The accepted carrier is an exact Gross odd-prime Wigner representation.  The
result is bounded direct-process software and does not claim an advantage over
the smaller exact amplitude/stabilizer-aware classical recurrence.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import zeta5_normalized_cubic_fourier_coherent_port as m237


P = 5
INV2 = 3
PORT_TYPE = "ZETA5_GROSS_WIGNER_INTERFACE_V1"
INTERFACES = (1, 2, 3)
ZERO = m237.ZERO
ONE = m237.ONE
SQRT5 = m237.SQRT5
K = m237.K

RESULT = "PASS_EXACT_ZETA5_WIGNER_CUBIC_MAGIC_INTERFACE_STRICT_SCOPE"
CLAIM = (
    "EXACT_GROSS_P5_WIGNER_PHASE_CARRIERS_ON_DECLARED_ONE_TWO_THREE_QUDIT_"
    "INTERFACES_PROPAGATE_CLIFFORD_PERMUTATIONS_AND_FORMULA_GENERATED_CUBIC_"
    "KERNELS_WITH_EXACT_POSITIVE_STABILIZER_SHAMS_AND_CUBIC_CAUSED_NEGATIVITY_"
    "FINAL_ONLY_AGGREGATE_PROJECTION_EXACT_SAME_BACKING_RESTORATION_AND_"
    "DESCRIPTOR_DISTINCT_REUSE_BUT_THE_SMALLER_EXACT5_TO_THE_N_AMPLITUDE_PLUS_"
    "STREAMED_WIGNER_CLASSICAL_RECURRENCE_REMAINS"
)
CLAIM_CEILING = (
    "EXACT_QZETA5_GROSS_WIGNER_P5_INTERFACES_N1_2_3_TWO_PUBLIC_CIRCUIT_"
    "FAMILIES_DIRECT_PROCESS_LOGICAL_CUSTODY_STABILIZER_RELATIVE_MAGIC_ONLY"
)


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonicalize_vector(values: list[K], exponent: int) -> int:
    while exponent and all(coordinate % 5 == 0 for value in values for coordinate in value):
        for index, value in enumerate(values):
            values[index] = tuple(coordinate // 5 for coordinate in value)  # type: ignore[assignment]
        exponent -= 1
    return exponent


def digits(index: int, count: int) -> tuple[int, ...]:
    output = [0] * count
    for position in range(count - 1, -1, -1):
        output[position] = index % P
        index //= P
    return tuple(output)


def encode(values: Sequence[int]) -> int:
    index = 0
    for value in values:
        index = P * index + value
    return index


def phase_index(q: Sequence[int], r: Sequence[int]) -> int:
    dimension = P ** len(q)
    return encode(q) * dimension + encode(r)


def vector_payload_bits(values: Sequence[K], exponent: int) -> int:
    return sum(signed_bits(c) for value in values for c in value) + signed_bits(5**exponent)


def vector_commitment(values: Sequence[K], exponent: int) -> str:
    payload = {"denominator_power5": exponent, "numerators": [list(value) for value in values]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def value_commitment(value: K, exponent: int) -> str:
    canonical = [value]
    exponent = canonicalize_vector(canonical, exponent)
    return hashlib.sha256(
        json.dumps({"denominator_power5": exponent, "numerator": list(canonical[0])}, sort_keys=True).encode()
    ).hexdigest()


def exact_cell_sequence_commitment(values: Iterable[K], exponent: int) -> str:
    digest = hashlib.sha256(b"CAT_CAS_EXACT_WIGNER_CELL_SEQUENCE_V1\0")
    for value in values:
        canonical = [value]
        cell_exponent = canonicalize_vector(canonical, exponent)
        digest.update(
            f"{cell_exponent}:{','.join(str(coordinate) for coordinate in canonical[0])};".encode()
        )
    return digest.hexdigest()


def real_pair(value: K, exponent: int) -> dict[str, int]:
    if value[1] != 0 or value[2] != value[3]:
        raise RuntimeError("Wigner aggregate left the real subfield")
    a = 2 * value[0] - value[2]
    b = -value[2]
    while exponent and a % 5 == 0 and b % 5 == 0:
        a //= 5
        b //= 5
        exponent -= 1
    return {
        "a_numerator": a,
        "b_sqrt5_numerator": b,
        "denominator_twice_power5_exponent": exponent,
    }


def real_sign(value: K) -> int:
    if value[1] != 0 or value[2] != value[3]:
        raise RuntimeError("Wigner cell is not real")
    # value = (A + B*sqrt(5))/2 in the physical embedding.
    a = 2 * value[0] - value[2]
    b = -value[2]
    if b < 0:
        return -real_sign(tuple(-coordinate for coordinate in value))  # type: ignore[arg-type]
    if b == 0:
        return (a > 0) - (a < 0)
    if a >= 0:
        return 1
    comparison = 5 * b * b - a * a
    return (comparison > 0) - (comparison < 0)


@dataclass(frozen=True)
class Gate:
    kind: str
    wire: int
    target: int = -1
    a: int = 0
    b: int = 0
    direction: int = 1

    def __post_init__(self) -> None:
        if self.kind not in {"FOURIER", "SUM", "CUBIC", "QUADRATIC"}:
            raise ValueError("unknown Wigner gate")
        if self.direction not in {-1, 1}:
            raise ValueError("gate direction must be signed")
        if self.kind == "SUM" and self.target == self.wire:
            raise ValueError("SUM needs distinct wires")
        if self.kind == "CUBIC" and self.a % P == 0:
            raise ValueError("cubic strength must be nonzero")

    def inverse(self) -> "Gate":
        if self.kind in {"FOURIER", "SUM"}:
            return Gate(self.kind, self.wire, self.target, self.a, self.b, -self.direction)
        return Gate(self.kind, self.wire, self.target, -self.a, -self.b, self.direction)

    def serial(self) -> tuple[object, ...]:
        return (self.kind, self.wire, self.target, self.a % P, self.b % P, self.direction)


def program_digest(family: int, n: int, gates: Sequence[Gate], output_q: Sequence[int], output_r: Sequence[int]) -> str:
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
class PublicProgram:
    family: int
    n: int
    gates: tuple[Gate, ...]
    output_q: tuple[int, ...]
    output_r: tuple[int, ...]
    program_id: str

    def __post_init__(self) -> None:
        if self.family not in (0, 1) or self.n not in INTERFACES:
            raise ValueError("program outside declared family")
        if len(self.output_q) != self.n or len(self.output_r) != self.n:
            raise ValueError("boundary arity mismatch")
        for gate in self.gates:
            if not 0 <= gate.wire < self.n or (gate.kind == "SUM" and not 0 <= gate.target < self.n):
                raise ValueError("gate wire outside interface")
        expected = program_digest(self.family, self.n, self.gates, self.output_q, self.output_r)
        if self.program_id != expected:
            raise ValueError("program digest is not bound to full descriptor")

    @property
    def descriptor(self) -> tuple[object, ...]:
        return (
            self.family,
            self.n,
            tuple(gate.serial() for gate in self.gates),
            self.output_q,
            self.output_r,
            self.program_id,
        )


def compile_program(family: int, n: int) -> PublicProgram:
    gates: list[Gate] = [Gate("FOURIER", 0)]
    for wire in range(n):
        if wire:
            gates.append(Gate("SUM", wire - 1, wire, direction=1))
        cubic_a = ((wire + 1) if family == 0 else (2 * wire + 2)) % P
        if cubic_a == 0:
            cubic_a = 4
        cubic_b = (family + 2 * wire) % P
        gates.append(Gate("CUBIC", wire, a=cubic_a, b=cubic_b))
        gates.append(Gate("FOURIER", wire, direction=1 if (wire + family) % 2 == 0 else -1))
    for wire in range(n - 1, 0, -1):
        gates.append(Gate("SUM", wire - 1, wire, direction=-1))
    output_q = tuple((family + wire) % P for wire in range(n))
    output_r = tuple((2 * family + wire) % P for wire in range(n))
    gate_tuple = tuple(gates)
    return PublicProgram(
        family, n, gate_tuple, output_q, output_r,
        program_digest(family, n, gate_tuple, output_q, output_r),
    )


def sham_program(program: PublicProgram) -> PublicProgram:
    gates = tuple(
        Gate("QUADRATIC", gate.wire, a=gate.a or 1, b=gate.b)
        if gate.kind == "CUBIC" else gate
        for gate in program.gates
    )
    return PublicProgram(
        program.family, program.n, gates, program.output_q, program.output_r,
        program_digest(program.family, program.n, gates, program.output_q, program.output_r),
    )


@dataclass
class Work:
    clifford_permutation_cell_moves: int = 0
    cubic_kernel_character_terms: int = 0
    cubic_kernel_field_multiplications: int = 0
    cubic_kernel_field_additions: int = 0
    canonical_common_factor_cancellations: int = 0
    exact_reality_checks: int = 0
    exact_normalization_checks: int = 0
    exact_purity_checks: int = 0
    peak_denominator_exponent: int = 0
    peak_resident_numerator_payload_bits: int = 0
    peak_resident_total_exact_payload_bits: int = 0
    retained_dynamic_inverse_history_entries: int = 0

    def observe(self, values: Sequence[K], exponent: int) -> None:
        self.peak_denominator_exponent = max(self.peak_denominator_exponent, exponent)
        numerator = sum(signed_bits(c) for value in values for c in value)
        self.peak_resident_numerator_payload_bits = max(self.peak_resident_numerator_payload_bits, numerator)
        self.peak_resident_total_exact_payload_bits = max(
            self.peak_resident_total_exact_payload_bits, numerator + signed_bits(5**exponent)
        )


def initial_wigner(n: int) -> list[K]:
    dimension = P**n
    values = [ZERO] * (dimension * dimension)
    q = (0,) * n
    for r_index in range(dimension):
        values[phase_index(q, digits(r_index, n))] = ONE
    return values


def apply_permutation(values: list[K], scratch: list[K], n: int, gate: Gate, work: Work) -> None:
    if any(value != ZERO for value in scratch):
        raise RuntimeError("dirty Wigner permutation scratch")
    dimension = P**n
    for q_index in range(dimension):
        q_new = list(digits(q_index, n))
        for r_index in range(dimension):
            r_new = list(digits(r_index, n))
            q_old = list(q_new)
            r_old = list(r_new)
            if gate.kind == "FOURIER":
                wire = gate.wire
                if gate.direction == 1:
                    q_old[wire], r_old[wire] = r_new[wire], (-q_new[wire]) % P
                else:
                    q_old[wire], r_old[wire] = (-r_new[wire]) % P, q_new[wire]
            elif gate.kind == "SUM":
                control, target, direction = gate.wire, gate.target, gate.direction
                q_old[target] = (q_new[target] - direction * q_new[control]) % P
                r_old[control] = (r_new[control] + direction * r_new[target]) % P
            elif gate.kind == "QUADRATIC":
                wire = gate.wire
                derivative = (2 * gate.a * q_new[wire] + gate.b) % P
                r_old[wire] = (r_new[wire] - derivative) % P
            else:
                raise RuntimeError("non-permutation gate entered permutation path")
            destination = phase_index(q_new, r_new)
            scratch[destination] = values[phase_index(q_old, r_old)]
            work.clifford_permutation_cell_moves += 1
    values[:] = scratch
    scratch[:] = [ZERO] * len(scratch)


def apply_cubic(values: list[K], scratch: list[K], n: int, gate: Gate, exponent: int, work: Work) -> int:
    if any(value != ZERO for value in scratch):
        raise RuntimeError("dirty Wigner cubic scratch")
    dimension = P**n
    wire = gate.wire
    a, b = gate.a % P, gate.b % P
    # Formula-generated, gate-local 5 x 5 x 5 kernel.  It is neither retained
    # by the carrier nor serialized; accounting reports its 125 field cells.
    kernels: list[list[list[K]]] = []
    for q_wire in range(P):
        q_rows: list[list[K]] = []
        for output_momentum in range(P):
            row: list[K] = []
            for source_momentum in range(P):
                kernel = ZERO
                for displacement in range(P):
                    phase = (
                        (source_momentum - output_momentum + 3 * a * q_wire * q_wire + b) * displacement
                        + 4 * a * displacement**3
                    ) % P
                    kernel = m237.k_add(kernel, m237.zeta_power(phase))
                    work.cubic_kernel_character_terms += 1
                row.append(kernel)
            q_rows.append(row)
        kernels.append(q_rows)
    for q_index in range(dimension):
        q = digits(q_index, n)
        for r_index in range(dimension):
            r = list(digits(r_index, n))
            total = ZERO
            for source_momentum in range(P):
                source_r = list(r)
                source_r[wire] = source_momentum
                total = m237.k_add(
                    total,
                    m237.k_mul(kernels[q[wire]][r[wire]][source_momentum], values[phase_index(q, source_r)]),
                )
                work.cubic_kernel_field_multiplications += 1
                work.cubic_kernel_field_additions += 1
            scratch[phase_index(q, r)] = total
    values[:] = scratch
    scratch[:] = [ZERO] * len(scratch)
    exponent += 1
    before = exponent
    exponent = canonicalize_vector(values, exponent)
    work.canonical_common_factor_cancellations += before - exponent
    work.observe(values, exponent)
    return exponent


def apply_gate(values: list[K], scratch: list[K], n: int, gate: Gate, exponent: int, work: Work) -> int:
    if gate.kind == "CUBIC":
        return apply_cubic(values, scratch, n, gate, exponent, work)
    apply_permutation(values, scratch, n, gate, work)
    work.observe(values, exponent)
    return exponent


def exact_receipts(values: Sequence[K], exponent: int, n: int, work: Work) -> dict[str, object]:
    total = ZERO
    square_sum = ZERO
    negative_sum = ZERO
    negative_count = 0
    for value in values:
        if value[1] != 0 or value[2] != value[3]:
            raise RuntimeError("non-real Wigner cell")
        work.exact_reality_checks += 1
        total = m237.k_add(total, value)
        square_sum = m237.k_add(square_sum, m237.k_mul(value, value))
        if real_sign(value) < 0:
            negative_count += 1
            negative_sum = m237.k_sub(negative_sum, value)
    total_values = [total]
    total_exponent = canonicalize_vector(total_values, exponent)
    normalized = total_values[0] == ONE and total_exponent == 0
    work.exact_normalization_checks += 1
    purity_values = [m237.k_scale(square_sum, P**n)]
    purity_exponent = canonicalize_vector(purity_values, 2 * exponent)
    pure = purity_values[0] == ONE and purity_exponent == 0
    work.exact_purity_checks += 1
    if not normalized or not pure:
        raise RuntimeError("Wigner normalization or purity failure")
    return {
        "negative_cell_count": negative_count,
        "negative_mass": real_pair(negative_sum, exponent),
        "negative_mass_positive": negative_count > 0 and real_sign(negative_sum) > 0,
        "normalization_exact": normalized,
        "purity_exact": pure,
    }


class WignerPort:
    def __init__(self, n: int) -> None:
        if n not in INTERFACES:
            raise ValueError("unsupported interface")
        self.n = n
        self.values = initial_wigner(n)
        self.scratch = [ZERO] * len(self.values)
        self.exponent = n
        self.leased = False
        self.owner = 0
        self.descriptor: tuple[object, ...] | None = None
        self.generation = 0
        self.last_restored_generation = 0
        self.cursor = 0

    def canonical(self) -> bool:
        return (
            self.values == initial_wigner(self.n)
            and self.scratch == [ZERO] * len(self.scratch)
            and self.exponent == self.n
            and self.cursor == 0
        )

    def require(self, owner: int, program: PublicProgram, generation: int) -> None:
        if not self.leased or owner != self.owner or generation != self.generation:
            raise RuntimeError("Wigner custody mismatch")
        if program.descriptor != self.descriptor or program.n != self.n:
            raise RuntimeError("Wigner program descriptor mismatch")

    def lease(self, owner: int, program: PublicProgram, generation: int, port_type: str = PORT_TYPE) -> None:
        if port_type != PORT_TYPE or self.leased or program.n != self.n:
            raise RuntimeError("invalid Wigner lease")
        if generation != self.last_restored_generation + 1 or not self.canonical():
            raise RuntimeError("nonmonotone or noncanonical Wigner lease")
        self.leased = True
        self.owner = owner
        self.descriptor = program.descriptor
        self.generation = generation

    def forward(self, owner: int, program: PublicProgram, generation: int, index: int, work: Work) -> None:
        self.require(owner, program, generation)
        if index != self.cursor or index >= len(program.gates):
            raise RuntimeError("forward cursor mismatch")
        self.exponent = apply_gate(self.values, self.scratch, self.n, program.gates[index], self.exponent, work)
        self.cursor += 1

    def inverse(self, owner: int, program: PublicProgram, generation: int, index: int, work: Work) -> None:
        self.require(owner, program, generation)
        if index != self.cursor - 1:
            raise RuntimeError("inverse cursor mismatch")
        self.exponent = apply_gate(
            self.values, self.scratch, self.n, program.gates[index].inverse(), self.exponent, work
        )
        self.cursor -= 1

    def project(self, owner: int, program: PublicProgram, generation: int, work: Work) -> dict[str, object]:
        self.require(owner, program, generation)
        if self.cursor != len(program.gates) or any(value != ZERO for value in self.scratch):
            raise RuntimeError("premature Wigner projection")
        receipts = exact_receipts(self.values, self.exponent, self.n, work)
        selected = self.values[phase_index(program.output_q, program.output_r)]
        return {
            **receipts,
            "selected_boundary_commitment": value_commitment(selected, self.exponent),
            "final_wigner_commitment": exact_cell_sequence_commitment(self.values, self.exponent),
        }

    def full_vector(self) -> list[K]:
        raise RuntimeError("full Wigner projection forbidden")

    def release(self, owner: int, program: PublicProgram, generation: int) -> None:
        self.require(owner, program, generation)
        if not self.canonical():
            raise RuntimeError("Wigner carrier not exactly restored")
        self.leased = False
        self.last_restored_generation = generation
        self.owner = 0
        self.descriptor = None
        self.generation = 0


def amplitude_apply_gate(values: list[K], scratch: list[K], exponent: int, n: int, gate: Gate) -> int:
    dimension = P**n
    if any(value != ZERO for value in scratch):
        raise RuntimeError("dirty amplitude scratch")
    if gate.kind == "FOURIER":
        for fixed_index in range(P ** (n - 1)):
            fixed = list(digits(fixed_index, n - 1))
            for output in range(P):
                total = ZERO
                for source in range(P):
                    coordinate = list(fixed)
                    coordinate.insert(gate.wire, source)
                    phase = gate.direction * source * output
                    total = m237.k_add(total, m237.k_mul(m237.zeta_power(phase), values[encode(coordinate)]))
                coordinate = list(fixed)
                coordinate.insert(gate.wire, output)
                scratch[encode(coordinate)] = m237.k_mul(SQRT5, total)
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
    elif gate.kind in {"CUBIC", "QUADRATIC"}:
        for index, value in enumerate(values):
            coordinate = digits(index, n)
            x = coordinate[gate.wire]
            phase = gate.a * (x**3 if gate.kind == "CUBIC" else x**2) + gate.b * x
            values[index] = m237.k_mul(m237.zeta_power(phase), value)
    else:
        raise RuntimeError("unknown amplitude gate")
    return canonicalize_vector(values, exponent)


def streamed_wigner_from_amplitudes(
    amplitudes: Sequence[K], exponent: int, program: PublicProgram
) -> tuple[dict[str, object], dict[str, int]]:
    n = program.n
    dimension = P**n
    negative_sum = ZERO
    total = ZERO
    square_sum = ZERO
    negative_count = 0
    correlation_terms = 0
    character_multiplications = 0
    coordinates = [digits(index, n) for index in range(dimension)]
    conjugates = [m237.k_conjugate(value) for value in amplitudes]
    selected: K | None = None
    wigner_commitment = hashlib.sha256(b"CAT_CAS_EXACT_WIGNER_CELL_SEQUENCE_V1\0")
    for q_index in range(dimension):
        q = coordinates[q_index]
        correlations: list[K] = []
        for displacement in coordinates:
            plus = tuple((q[wire] + INV2 * displacement[wire]) % P for wire in range(n))
            minus = tuple((q[wire] - INV2 * displacement[wire]) % P for wire in range(n))
            correlations.append(m237.k_mul(amplitudes[encode(plus)], conjugates[encode(minus)]))
            correlation_terms += 1
        for r_index in range(dimension):
            r = coordinates[r_index]
            value = ZERO
            for displacement, correlation in zip(coordinates, correlations):
                phase = -sum(r[wire] * displacement[wire] for wire in range(n))
                value = m237.k_add(value, m237.k_mul(m237.zeta_power(phase), correlation))
                character_multiplications += 1
            if value[1] != 0 or value[2] != value[3]:
                raise RuntimeError("amplitude oracle produced non-real Wigner value")
            canonical_cell = [value]
            cell_exponent = canonicalize_vector(canonical_cell, 2 * exponent + n)
            wigner_commitment.update(
                f"{cell_exponent}:{','.join(str(coordinate) for coordinate in canonical_cell[0])};".encode()
            )
            total = m237.k_add(total, value)
            square_sum = m237.k_add(square_sum, m237.k_mul(value, value))
            if real_sign(value) < 0:
                negative_count += 1
                negative_sum = m237.k_sub(negative_sum, value)
            if q == program.output_q and r == program.output_r:
                selected = value
    wigner_exponent = 2 * exponent + n
    total_values = [total]
    total_exponent = canonicalize_vector(total_values, wigner_exponent)
    purity_values = [m237.k_scale(square_sum, dimension)]
    purity_exponent = canonicalize_vector(purity_values, 2 * wigner_exponent)
    if total_values[0] != ONE or total_exponent != 0 or purity_values[0] != ONE or purity_exponent != 0:
        raise RuntimeError("amplitude oracle Wigner receipts failed")
    if selected is None:
        raise RuntimeError("selected Wigner boundary was not visited")
    aggregate = {
        "negative_cell_count": negative_count,
        "negative_mass": real_pair(negative_sum, wigner_exponent),
        "negative_mass_positive": negative_count > 0 and real_sign(negative_sum) > 0,
        "normalization_exact": True,
        "purity_exact": True,
        "selected_boundary_commitment": value_commitment(selected, wigner_exponent),
        "final_wigner_commitment": wigner_commitment.hexdigest(),
    }
    work = {
        "resident_amplitude_field_cells": dimension,
        "amplitude_scratch_field_cells": dimension,
        "streamed_wigner_correlation_products": correlation_terms,
        "streamed_wigner_character_multiplications": character_multiplications,
        "resident_wigner_field_cells": 0,
    }
    return aggregate, work


def amplitude_baseline(program: PublicProgram) -> dict[str, object]:
    dimension = P**program.n
    values = [ZERO] * dimension
    scratch = [ZERO] * dimension
    values[0] = ONE
    exponent = 0
    for gate in program.gates:
        exponent = amplitude_apply_gate(values, scratch, exponent, program.n, gate)
    aggregate, work = streamed_wigner_from_amplitudes(values, exponent, program)
    return {
        **aggregate,
        "final_amplitude_commitment": vector_commitment(values, exponent),
        "final_amplitude_denominator_exponent": exponent,
        "final_amplitude_payload_bits": vector_payload_bits(values, exponent),
        "work": work,
    }


def run_transaction(
    port: WignerPort,
    program: PublicProgram,
    generation: int,
    owner: int = 23801,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    if port is None:
        raise TypeError("null Wigner port")
    backings = (id(port.values), id(port.scratch))
    work = Work()
    work.observe(port.values, port.exponent)
    port.lease(owner, program, generation)
    for index in range(len(program.gates)):
        port.forward(owner, program, generation, index, work)
    aggregate = port.project(owner, program, generation, work)
    retained_boundary = json.dumps(aggregate, sort_keys=True)
    final_exponent = port.exponent
    final_payload = vector_payload_bits(port.values, port.exponent)
    for index in range(len(program.gates) - 1, -1, -1):
        port.inverse(owner, program, generation, index, work)
    exact = port.canonical()
    same_backings = backings == (id(port.values), id(port.scratch))
    port.release(owner, program, generation)
    if retained_boundary != json.dumps(aggregate, sort_keys=True):
        raise RuntimeError("Wigner boundary was consumed by inverse")
    if baseline is None:
        baseline = amplitude_baseline(program)
    comparable = (
        "negative_cell_count", "negative_mass", "negative_mass_positive",
        "normalization_exact", "purity_exact", "selected_boundary_commitment",
        "final_wigner_commitment",
    )
    if any(aggregate[key] != baseline[key] for key in comparable):
        raise RuntimeError("smaller amplitude baseline disagrees with Wigner carrier")
    dimension = P**program.n
    return {
        "family": program.family,
        "interface_qudits": program.n,
        "hilbert_amplitude_cells": dimension,
        "resident_wigner_field_cells": dimension * dimension,
        "resident_wigner_integer_coordinates": 4 * dimension * dimension,
        "scratch_wigner_field_cells": dimension * dimension,
        "scratch_wigner_integer_coordinates": 4 * dimension * dimension,
        "gate_count": len(program.gates),
        "cubic_gate_count": sum(gate.kind == "CUBIC" for gate in program.gates),
        "final_denominator_exponent": final_exponent,
        "final_total_exact_payload_bits": final_payload,
        **aggregate,
        "retained_final_aggregate_during_inverse": True,
        "canonical_post_inverse_state_exact": exact,
        "same_wigner_and_scratch_backings": same_backings,
        "restoration_generation": generation,
        "baseline_reload_used": False,
        "work": asdict(work),
        "matched_classical": baseline,
    }


def expect_rejected(callback: Callable[[], object]) -> bool:
    try:
        callback()
    except (RuntimeError, TypeError, ValueError):
        return True
    return False


def controls(
    baselines: dict[tuple[int, int], dict[str, object]] | None = None,
    shams: dict[tuple[int, int], dict[str, object]] | None = None,
) -> dict[str, bool]:
    program = compile_program(0, 2)
    owner = 23811
    premature = WignerPort(2)
    premature.lease(owner, program, 1)
    premature_projection = expect_rejected(lambda: premature.project(owner, program, 1, Work()))
    full_projection = expect_rejected(premature.full_vector)
    wrong_owner = expect_rejected(lambda: premature.forward(owner + 1, program, 1, 0, Work()))
    wrong_program = expect_rejected(lambda: premature.forward(owner, compile_program(1, 2), 1, 0, Work()))
    wrong_generation = expect_rejected(lambda: premature.forward(owner, program, 2, 0, Work()))
    wrong_type = expect_rejected(lambda: WignerPort(2).lease(owner, program, 1, "WRONG"))
    malformed_gates = program.gates[:-1] + (Gate("CUBIC", 1, a=4, b=4),)
    forged_descriptor = expect_rejected(
        lambda: PublicProgram(program.family, program.n, malformed_gates, program.output_q, program.output_r, program.program_id)
    )
    wrong_arity = expect_rejected(lambda: WignerPort(1).lease(owner, program, 1))
    null_carrier = expect_rejected(lambda: run_transaction(None, program, 1))  # type: ignore[arg-type]
    dirty = WignerPort(2)
    dirty.scratch[0] = ONE
    dirty_scratch = expect_rejected(lambda: dirty.lease(owner, program, 1))

    missing = WignerPort(2)
    missing.lease(owner, program, 1)
    for index in range(len(program.gates)):
        missing.forward(owner, program, 1, index, Work())
    missing_inverse = expect_rejected(lambda: missing.release(owner, program, 1))

    reordered = WignerPort(2)
    reordered.lease(owner, program, 1)
    for index in range(len(program.gates)):
        reordered.forward(owner, program, 1, index, Work())
    reordered_inverse = expect_rejected(
        lambda: reordered.inverse(owner, program, 1, len(program.gates) - 2, Work())
    )

    wrong = WignerPort(2)
    wrong.lease(owner, program, 1)
    for index in range(len(program.gates)):
        wrong.forward(owner, program, 1, index, Work())
    last_index = len(program.gates) - 1
    wrong.exponent = apply_gate(
        wrong.values, wrong.scratch, wrong.n, Gate("CUBIC", 0, a=1, b=1), wrong.exponent, Work()
    )
    wrong.cursor -= 1
    for index in range(last_index - 1, -1, -1):
        wrong.inverse(owner, program, 1, index, Work())
    wrong_inverse = expect_rejected(lambda: wrong.release(owner, program, 1))

    stale = WignerPort(2)
    run_transaction(stale, compile_program(0, 2), 1, owner)
    run_transaction(stale, compile_program(1, 2), 2, owner)
    stale_generation = expect_rejected(lambda: stale.lease(owner, program, 2))

    if baselines is None:
        baselines = {
            (family, n): amplitude_baseline(compile_program(family, n))
            for family in (0, 1) for n in INTERFACES
        }
    if shams is None:
        shams = {
            (family, n): amplitude_baseline(sham_program(compile_program(family, n)))
            for family in (0, 1) for n in INTERFACES
        }
    sham_nonnegative = all(not value["negative_mass_positive"] for value in shams.values())
    cubic_positive = all(value["negative_mass_positive"] for value in baselines.values())
    semantic_perturbation = all(
        baselines[(0, n)]["final_amplitude_commitment"]
        != baselines[(1, n)]["final_amplitude_commitment"]
        for n in INTERFACES
    )
    return {
        "computational_basis_wigner_support_analytic": initial_wigner(1).count(ONE) == 5,
        "cubic_third_finite_difference_nonzero": 6 % P != 0,
        "quadratic_third_finite_difference_zero": True,
        "clifford_shams_exactly_nonnegative": sham_nonnegative,
        "all_cubic_augmented_interfaces_have_positive_negativity": cubic_positive,
        "cubic_family_perturbation_changes_amplitude_state": semantic_perturbation,
        "premature_negative_mass_projection_rejected": premature_projection,
        "full_wigner_projection_rejected": full_projection,
        "wrong_owner_rejected": wrong_owner,
        "wrong_program_rejected": wrong_program,
        "wrong_generation_rejected": wrong_generation,
        "wrong_type_rejected": wrong_type,
        "wrong_interface_arity_rejected": wrong_arity,
        "same_id_changed_descriptor_rejected": forged_descriptor,
        "null_carrier_rejected": null_carrier,
        "dirty_scratch_rejected": dirty_scratch,
        "missing_inverse_rejected": missing_inverse,
        "reordered_inverse_rejected": reordered_inverse,
        "wrong_inverse_after_remaining_schedule_rejected": wrong_inverse,
        "stale_generation_rejected": stale_generation,
        "resident_wigner_cells_serialized": False,
        "negative_cell_coordinates_serialized": False,
        "amplitudes_or_density_matrices_serialized": False,
        "path_histories_or_assignments_enumerated": False,
        "precomputed_cubic_kernel_tables_retained": False,
        "public_compiler_reads_final_answer": False,
    }


def comparable_case(case: dict[str, object]) -> dict[str, object]:
    return {
        key: case[key]
        for key in (
            "family", "interface_qudits", "hilbert_amplitude_cells",
            "negative_cell_count", "negative_mass", "negative_mass_positive",
            "normalization_exact", "purity_exact", "selected_boundary_commitment",
            "final_wigner_commitment",
            "canonical_post_inverse_state_exact", "same_wigner_and_scratch_backings",
            "restoration_generation", "baseline_reload_used",
        )
    }


def main(reference_path: Path) -> None:
    reference = json.loads(reference_path.read_text())
    programs = {(family, n): compile_program(family, n) for family in (0, 1) for n in INTERFACES}
    baselines = {key: amplitude_baseline(program) for key, program in programs.items()}
    shams = {key: amplitude_baseline(sham_program(program)) for key, program in programs.items()}
    lower_cases = {
        (family, n): run_transaction(WignerPort(n), programs[(family, n)], 1, baseline=baselines[(family, n)])
        for family in (0, 1) for n in (1, 2)
    }
    primary_program = compile_program(0, 3)
    reuse_program = compile_program(1, 3)
    shared = WignerPort(3)
    primary = run_transaction(shared, primary_program, 1, baseline=baselines[(0, 3)])
    reuse = run_transaction(shared, reuse_program, 2, baseline=baselines[(1, 3)])
    fresh = run_transaction(WignerPort(3), reuse_program, 1, baseline=baselines[(1, 3)])
    cases = [
        lower_cases[(0, 1)], lower_cases[(0, 2)], primary,
        lower_cases[(1, 1)], lower_cases[(1, 2)], fresh,
    ]
    control_values = controls(baselines, shams)
    if [comparable_case(case) for case in cases] != reference["cases"]:
        raise RuntimeError("independent amplitude reference case parity failed")
    if control_values != reference["controls"]:
        raise RuntimeError("independent amplitude reference controls differ")
    reference_reuse = reference["reuse"]
    for key, value in {
        "primary": comparable_case(primary),
        "reuse": comparable_case(reuse),
        "fresh_reuse": comparable_case(fresh),
    }.items():
        if value != reference_reuse[key]:
            raise RuntimeError("independent amplitude reference reuse parity failed")
    output = {
        "schema": "cat_cas.zeta5_wigner_cubic_magic_interface.v1",
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
            "fresh_restored_boundary_agreement": comparable_case(reuse)["selected_boundary_commitment"] == comparable_case(fresh)["selected_boundary_commitment"],
            "fresh_restored_resource_signature_agreement": (
                reuse["final_denominator_exponent"], reuse["final_total_exact_payload_bits"], reuse["negative_cell_count"]
            ) == (
                fresh["final_denominator_exponent"], fresh["final_total_exact_payload_bits"], fresh["negative_cell_count"]
            ),
            "same_backing_across_primary_and_reuse": primary["same_wigner_and_scratch_backings"] and reuse["same_wigner_and_scratch_backings"],
        },
        "phase_resource_diagnostic": {
            "formalism": "GROSS_ODD_PRIME_DISCRETE_WIGNER_P5",
            "interfaces": list(INTERFACES),
            "resident_wigner_cells": [P ** (2 * n) for n in INTERFACES],
            "clifford_sham_nonnegative": True,
            "cubic_causes_exact_negative_mass": True,
            "stabilizer_relative_magic_witness": True,
            "distinct_resource_unavailable_to_compact_classical_software": False,
        },
        "resource_law": {
            "resident_wigner_field_cells": [25, 625, 15625],
            "resident_wigner_integer_coordinates": [100, 2500, 62500],
            "same_sized_wigner_scratch_retained": True,
            "stronger_amplitude_persistent_field_cells": [5, 25, 125],
            "stronger_amplitude_scratch_field_cells": [5, 25, 125],
            "streamed_reference_retains_wigner_grid": False,
            "peak_formula_generated_cubic_kernel_field_cells": 125,
            "interface_synergy_established": False,
            "tensor_product_factorization_excluded": False,
            "whole_transaction_live_cell_and_payload_accounting_complete": False,
            "python_objects_allocator_hash_serialization_rss_excluded_not_zero": True,
            "resource_verification_level": "PACKAGE_SELF_REVIEW",
        },
        "matched_classical": {
            "clifford_sham": "AFFINE_LAGRANGIAN_STABILIZER_WIGNER_SUPPORT_O_N2_F5_STATE",
            "cubic_circuit": "EXACT5_TO_THE_N_AMPLITUDE_RECURRENCE_PLUS_STREAMED_FINAL_WIGNER_AGGREGATE",
            "phase_carrier_resident_cells_at_n3": 15625,
            "amplitude_baseline_resident_cells_at_n3": 125,
            "amplitude_baseline_is_smaller": True,
            "explicit_gate_depth_path_sum_used": False,
            "computational_advantage": False,
            "distinct_phase_resource": False,
        },
        "separate_reference": {
            "imports_m238_or_m237_production": False,
            "evolves_exact_amplitudes": True,
            "streams_final_wigner_reconstruction": True,
            "implements_independent_custody_state_machine": True,
        },
        "source_dependencies": {
            "production_sha256": file_sha256(Path(__file__)),
            "m237_algebra_sha256": file_sha256(Path(m237.__file__)),
            "separate_reference_sha256": file_sha256(
                Path(__file__).with_name("zeta5_wigner_cubic_magic_interface_separate_reference.py")
            ),
        },
        "claim_limits": {
            "magic_resource_unavailable_to_compact_classical_software": False,
            "computational_advantage": False,
            "general_n_scaling": False,
            "compact_fixed_width_growth": False,
            "general_tensor_network_contraction": False,
            "catvm_machine_enforced_custody": False,
            "small_wall_crossed": False,
            "unbounded_catalytic_computation_established": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference_established": False,
        },
        "terminal": False,
    }
    print(json.dumps(output, sort_keys=True, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: zeta5_wigner_cubic_magic_interface.py REFERENCE_JSON")
    main(Path(sys.argv[1]))
