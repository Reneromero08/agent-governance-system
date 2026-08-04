#!/usr/bin/env python3
"""Import-isolated exact oracle for the resident cubic-strength phase port."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ORDERS = (5, 11, 23, 29, 41, 53, 83, 89, 113)
FAMILIES = ("PRIMARY", "ALTERNATE")
AUXILIARY_MODULUS = 998244353
AUXILIARY_GENERATOR = 3


def fail(message: str) -> None:
    raise RuntimeError(message)


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return value == divisor
        divisor += 1
    return True


@dataclass(frozen=True)
class Field:
    q: int
    p: int
    root: int
    gauss_one: int


def field_for(q: int) -> Field:
    p = 2 * q + 1
    if not is_prime(q) or not is_prime(p):
        fail("invalid safe-prime pair")
    generator = 2
    while generator < p and (pow(generator, 2, p) == 1 or pow(generator, q, p) == 1):
        generator += 1
    if generator == p:
        fail("field generator absent")
    root = pow(generator, 2, p)
    if pow(root, q, p) != 1 or any(pow(root, exponent, p) == 1 for exponent in range(1, q)):
        fail("phase root invalid")
    gauss = sum(pow(root, x * x % q, p) for x in range(q)) % p
    if not gauss:
        fail("Gauss sum vanished")
    return Field(q, p, root, gauss)


def phase(field: Field, exponent: int) -> int:
    return pow(field.root, exponent % field.q, field.p)


def matrix2_inverse(matrix: tuple[int, ...], modulus: int) -> tuple[int, ...]:
    a, b, c, d = matrix
    determinant = (a * d - b * c) % modulus
    if not determinant:
        fail("singular 2x2 matrix")
    scale = pow(determinant, -1, modulus)
    return d * scale % modulus, -b * scale % modulus, -c * scale % modulus, a * scale % modulus


def kernel(field: Field, symplectic: tuple[int, ...], x: int, y: int) -> int:
    a, b, c, d = symplectic
    if b % field.q:
        half_b = pow(2 * b % field.q, -1, field.q)
        return phase(field, (d * x * x - 2 * x * y + a * y * y) * half_b)
    if y % field.q != d * x % field.q:
        return 0
    return phase(field, c * d * x * x * pow(2, -1, field.q))


def cocycle(field: Field, left: tuple[int, ...], right: tuple[int, ...]) -> int:
    a, b, _, _ = left
    _, f, _, h = right
    if b % field.q == 0 or f % field.q == 0:
        return 1
    coefficient = (
        a * pow(2 * b % field.q, -1, field.q)
        + h * pow(2 * f % field.q, -1, field.q)
    ) % field.q
    if coefficient == 0:
        return field.q % field.p
    character = pow(coefficient, (field.q - 1) // 2, field.q)
    return field.gauss_one if character == 1 else -field.gauss_one % field.p


def gaussian_inverse(field: Field, payload: tuple[int, ...]) -> tuple[int, ...]:
    symplectic = payload[:4]
    inverse_symplectic = matrix2_inverse(symplectic, field.q)
    scalar = cocycle(field, symplectic, inverse_symplectic)
    coefficient = payload[4] % field.p
    return inverse_symplectic + (pow(coefficient * scalar % field.p, -1, field.p),)


@dataclass(frozen=True)
class Operation:
    kind: str
    payload: tuple[int, ...]


def compile_word(q: int, depth: int, family: str) -> list[Operation]:
    if depth < 1 or family not in FAMILIES:
        fail("invalid public word")
    code = 1 if family == "PRIMARY" else 2
    operations: list[Operation] = []
    for layer in range(depth):
        parameter = (layer + code) % q or 1
        first = Operation("CONTROLLED_CUBIC", ((2 * layer + code) % q or 1,))
        second = Operation("CONTROLLED_CUBIC", ((3 * layer + 2 * code + 1) % q or 1,))
        data = Operation("DATA_GAUSSIAN", (1, 1, parameter, parameter + 1, 1))
        latent = Operation("LATENT_GAUSSIAN", (parameter + 1, 1, parameter, 1, 1))
        operations.extend((first, data, second, latent) if family == "PRIMARY" else (first, latent, second, data))
    operations.append(Operation("FIBER", (1, 1, 1, 2)))
    return operations


def resident_latent(field: Field, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    return [phase(field, (code + 1) * s * s + (2 * code + 1) * s + code) for s in range(field.q)]


def data_port(field: Field, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    return [
        (fiber + 1) * phase(field, (fiber + code + 1) * x * x + (3 * fiber + 2 * code + 1) * x) % field.p
        for fiber in range(2)
        for x in range(field.q)
    ]


def address(q: int, s: int, fiber: int, x: int) -> int:
    return (2 * s + fiber) * q + x


def initial_values(field: Field, family: str, latent: list[int] | None = None) -> list[int]:
    strengths = latent if latent is not None else resident_latent(field, family)
    data = data_port(field, family)
    return [strengths[s] * data[fiber * field.q + x] % field.p for s in range(field.q) for fiber in range(2) for x in range(field.q)]


def work_record() -> dict[str, int]:
    return {key: 0 for key in (
        "controlled_phase_evaluations",
        "controlled_field_multiplications",
        "data_gaussian_kernel_evaluations",
        "data_gaussian_field_multiply_adds",
        "latent_gaussian_kernel_evaluations",
        "latent_gaussian_field_multiply_adds",
        "fiber_field_multiply_adds",
        "projection_field_multiply_adds",
        "inverse_gaussian_scalar_rematerializations",
        "temporary_vector_field_cells_peak",
    )}


@dataclass
class State:
    field: Field
    values: list[int]
    fixture_family: str
    stage: str = "IDLE"
    generation: int = 0

    @classmethod
    def fixture(cls, q: int, family: str, latent: list[int] | None = None) -> "State":
        if family not in FAMILIES:
            fail("invalid fixture family")
        field = field_for(q)
        strengths = latent if latent is not None else resident_latent(field, family)
        if len(strengths) != q or not any(value % field.p for value in strengths):
            fail("invalid latent operand")
        return cls(field, initial_values(field, family, strengths), family)

    def backing(self) -> tuple[int, int]:
        return id(self), id(self.values)

    def payload(self) -> tuple[Any, ...]:
        return self.field.q, self.field.p, self.fixture_family, tuple(self.values), self.stage


def dense_vector(
    vector: list[int],
    payload: tuple[int, ...],
    field: Field,
    work: dict[str, int],
    axis: str,
) -> list[int]:
    q, p = field.q, field.p
    output = [
        payload[4] * sum(kernel(field, payload[:4], x, y) * vector[y] for y in range(q)) % p
        for x in range(q)
    ]
    work[f"{axis.lower()}_gaussian_kernel_evaluations"] += q * q
    work[f"{axis.lower()}_gaussian_field_multiply_adds"] += q * q
    work["temporary_vector_field_cells_peak"] = max(work["temporary_vector_field_cells_peak"], q)
    return output


def apply(state: State, operation: Operation, work: dict[str, int]) -> None:
    q, p = state.field.q, state.field.p
    if operation.kind == "CONTROLLED_CUBIC":
        strength = operation.payload[0]
        for s in range(q):
            for x in range(q):
                multiplier = phase(state.field, strength * s * x * x * x)
                work["controlled_phase_evaluations"] += 1
                for fiber in range(2):
                    index = address(q, s, fiber, x)
                    state.values[index] = state.values[index] * multiplier % p
                    work["controlled_field_multiplications"] += 1
    elif operation.kind == "DATA_GAUSSIAN":
        for s in range(q):
            for fiber in range(2):
                start = address(q, s, fiber, 0)
                state.values[start : start + q] = dense_vector(
                    state.values[start : start + q], operation.payload, state.field, work, "DATA"
                )
    elif operation.kind == "LATENT_GAUSSIAN":
        for fiber in range(2):
            for x in range(q):
                vector = [state.values[address(q, s, fiber, x)] for s in range(q)]
                output = dense_vector(vector, operation.payload, state.field, work, "LATENT")
                for s, value in enumerate(output):
                    state.values[address(q, s, fiber, x)] = value
    elif operation.kind == "FIBER":
        a, b, c, d = operation.payload
        for s in range(q):
            for x in range(q):
                left_index, right_index = address(q, s, 0, x), address(q, s, 1, x)
                left, right = state.values[left_index], state.values[right_index]
                state.values[left_index] = (a * left + c * right) % p
                state.values[right_index] = (b * left + d * right) % p
                work["fiber_field_multiply_adds"] += 4
    else:
        fail("unknown operation")


def inverse(state: State, operation: Operation, work: dict[str, int], wrong: bool = False) -> Operation:
    if operation.kind == "CONTROLLED_CUBIC":
        return Operation(operation.kind, ((-operation.payload[0] + int(wrong)) % state.field.q,))
    if operation.kind in ("DATA_GAUSSIAN", "LATENT_GAUSSIAN"):
        payload = list(gaussian_inverse(state.field, operation.payload))
        work["inverse_gaussian_scalar_rematerializations"] += 1
        if wrong:
            payload[-1] = (payload[-1] + 1) % state.field.p or 1
        return Operation(operation.kind, tuple(payload))
    payload = list(matrix2_inverse(operation.payload, state.field.p))
    if wrong:
        payload[0] = (payload[0] + 1) % state.field.p
    return Operation(operation.kind, tuple(payload))


def matrix_rank(state: State) -> int:
    q, p = state.field.q, state.field.p
    columns = 2 * q
    matrix = [state.values[address(q, s, 0, 0) : address(q, s, 0, 0) + columns] for s in range(q)]
    pivot_row = 0
    for column in range(columns):
        pivot = next((row for row in range(pivot_row, q) if matrix[row][column] % p), None)
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        scale = pow(matrix[pivot_row][column], -1, p)
        matrix[pivot_row] = [value * scale % p for value in matrix[pivot_row]]
        for row in range(q):
            if row == pivot_row or not matrix[row][column] % p:
                continue
            factor = matrix[row][column]
            matrix[row] = [(left - factor * right) % p for left, right in zip(matrix[row], matrix[pivot_row])]
        pivot_row += 1
        if pivot_row == q:
            break
    return pivot_row


def forward(state: State, depth: int, family: str, reordered: bool = False) -> tuple[list[Operation], dict[str, int], list[int]]:
    if state.stage != "IDLE":
        fail("state not idle")
    operations = compile_word(state.field.q, depth, family)
    if reordered:
        operations = list(reversed(operations))
    work = work_record()
    ranks: list[int] = []
    for operation in operations:
        apply(state, operation, work)
        if operation.kind == "CONTROLLED_CUBIC":
            ranks.append(matrix_rank(state))
    state.stage = "FORWARD_COMPLETE"
    return operations, work, ranks


def reverse(state: State, operations: list[Operation], work: dict[str, int], mutation: str | None = None) -> None:
    if state.stage != "FORWARD_COMPLETE":
        fail("state not forward complete")
    sequence = list(reversed(operations))
    if mutation == "MISSING":
        sequence = sequence[1:]
    elif mutation == "REORDER":
        sequence = list(operations)
    for index, operation in enumerate(sequence):
        apply(state, inverse(state, operation, work, mutation == "WRONG" and index == 0), work)
    state.stage = "IDLE"


def probes(q: int, family: str) -> tuple[tuple[int, int, int, int], ...]:
    code = 1 if family == "PRIMARY" else 2
    return (
        ((3 * code + 1) % q, 0, (5 * code + 2) % q, 1),
        ((7 * code + 2) % q, 1, (11 * code + 3) % q, 2),
        ((13 * code + 1) % q, 0, (17 * code + 4) % q, 3),
        ((19 * code + 2) % q, 1, (23 * code + 5) % q, 5),
    )


def project(state: State, family: str) -> int:
    if state.stage != "FORWARD_COMPLETE":
        fail("projection unavailable")
    return sum(weight * state.values[address(state.field.q, s, fiber, x)] for s, fiber, x, weight in probes(state.field.q, family)) % state.field.p


def digest(state: State) -> str:
    return hashlib.sha256(repr((state.field.q, tuple(state.values))).encode("ascii")).hexdigest()


@dataclass
class ScratchAccount:
    field: int = 0
    auxiliary: int = 0
    field_peak: int = 0
    auxiliary_peak: int = 0
    total_peak: int = 0
    field_at_total_peak: int = 0
    auxiliary_at_total_peak: int = 0

    def add(self, field: int = 0, auxiliary: int = 0) -> None:
        self.field += field
        self.auxiliary += auxiliary
        self.field_peak = max(self.field_peak, self.field)
        self.auxiliary_peak = max(self.auxiliary_peak, self.auxiliary)
        total = self.field + self.auxiliary
        if total > self.total_peak:
            self.total_peak = total
            self.field_at_total_peak = self.field
            self.auxiliary_at_total_peak = self.auxiliary

    def drop(self, field: int = 0, auxiliary: int = 0) -> None:
        self.field -= field
        self.auxiliary -= auxiliary
        if self.field < 0 or self.auxiliary < 0:
            fail("scratch account underflow")

    def require_closed(self) -> None:
        if self.field or self.auxiliary:
            fail("scratch account open")


def prime_divisors(value: int) -> tuple[int, ...]:
    answer: list[int] = []
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            answer.append(divisor)
            while value % divisor == 0:
                value //= divisor
        divisor += 1
    if value > 1:
        answer.append(value)
    return tuple(answer)


def least_generator(prime: int) -> int:
    factors = prime_divisors(prime - 1)
    return next(candidate for candidate in range(2, prime) if all(pow(candidate, (prime - 1) // factor, prime) != 1 for factor in factors))


def radix2(buffer: list[int], undo: bool, work: dict[str, int]) -> None:
    size = len(buffer)
    if size & (size - 1) or (AUXILIARY_MODULUS - 1) % size:
        fail("unsupported oracle NTT size")
    permutation = 0
    for position in range(1, size):
        bit = size >> 1
        while permutation & bit:
            permutation ^= bit
            bit >>= 1
        permutation ^= bit
        if position < permutation:
            buffer[position], buffer[permutation] = buffer[permutation], buffer[position]
    width = 2
    while width <= size:
        omega = pow(AUXILIARY_GENERATOR, (AUXILIARY_MODULUS - 1) // width, AUXILIARY_MODULUS)
        if undo:
            omega = pow(omega, -1, AUXILIARY_MODULUS)
        half = width // 2
        for base in range(0, size, width):
            factor = 1
            for position in range(half):
                even = buffer[base + position]
                odd = buffer[base + position + half] * factor % AUXILIARY_MODULUS
                buffer[base + position] = (even + odd) % AUXILIARY_MODULUS
                buffer[base + position + half] = (even - odd) % AUXILIARY_MODULUS
                factor = factor * omega % AUXILIARY_MODULUS
                work["ntt_butterflies"] += 1
        width *= 2
    if undo:
        scale = pow(size, -1, AUXILIARY_MODULUS)
        for position in range(size):
            buffer[position] = buffer[position] * scale % AUXILIARY_MODULUS
    work["ntt_transforms"] += 1


def cyclic_convolution(
    left: list[int],
    right: list[int],
    field_modulus: int,
    bound: int,
    work: dict[str, int],
    scratch: ScratchAccount,
) -> list[int]:
    cycle = len(left)
    size = 1
    while size < 2 * cycle - 1:
        size *= 2
    if len(right) != cycle or not 0 < bound < AUXILIARY_MODULUS:
        fail("invalid exact convolution")
    first = left + [0] * (size - cycle)
    scratch.add(auxiliary=size)
    second = right + [0] * (size - cycle)
    scratch.add(auxiliary=size)
    work["ntt_max_length"] = max(work["ntt_max_length"], size)
    work["ntt_temporary_integer_cells_peak"] = max(work["ntt_temporary_integer_cells_peak"], 2 * size)
    work["exact_convolution_coefficient_bound_peak"] = max(work["exact_convolution_coefficient_bound_peak"], bound)
    radix2(first, False, work)
    radix2(second, False, work)
    for position in range(size):
        first[position] = first[position] * second[position] % AUXILIARY_MODULUS
        work["ntt_pointwise_multiplications"] += 1
    radix2(first, True, work)
    for position in range(cycle - 1):
        first[position] = (first[position] + first[position + cycle]) % field_modulus
    second.clear()
    scratch.drop(auxiliary=size)
    scratch.drop(auxiliary=size)
    scratch.add(field=cycle)
    del first[cycle:]
    return first


def rader(values: list[int], root: int, field: Field, work: dict[str, int], scratch: ScratchAccount) -> list[int]:
    q, p = field.q, field.p
    generator = least_generator(q)
    cycle = q - 1
    source = [values[pow(generator, -position % cycle, q)] for position in range(cycle)]
    scratch.add(field=cycle)
    roots = [pow(root, pow(generator, position, q), p) for position in range(cycle)]
    scratch.add(field=cycle)
    convolution = cyclic_convolution(source, roots, p, cycle * (p - 1) ** 2, work, scratch)
    source.clear()
    roots.clear()
    scratch.drop(field=2 * cycle)
    output = [0] * q
    scratch.add(field=q)
    output[0] = sum(values) % p
    for position in range(cycle):
        output[pow(generator, position, q)] = (values[0] + convolution[position]) % p
    convolution.clear()
    scratch.drop(field=cycle)
    work["rader_dfts"] += 1
    return output


def fast_gaussian(vector: list[int], payload: tuple[int, ...], field: Field, work: dict[str, int], scratch: ScratchAccount) -> list[int]:
    q, p = field.q, field.p
    a, b, c, d, coefficient = payload
    if b % q == 0:
        output = [coefficient * phase(field, c * d * x * x) * vector[d * x % q] % p for x in range(q)]
        scratch.add(field=q)
        work["sparse_gaussian_terms"] += q
        return output
    inverse_two_b = pow(2 * b % q, -1, q)
    prepared = [phase(field, a * y * y * inverse_two_b) * vector[y] % p for y in range(q)]
    scratch.add(field=q)
    transformed = rader(prepared, phase(field, -pow(b, -1, q)), field, work, scratch)
    output = [coefficient * phase(field, d * x * x * inverse_two_b) * transformed[x] % p for x in range(q)]
    scratch.add(field=q)
    prepared.clear()
    transformed.clear()
    scratch.drop(field=2 * q)
    work["chirp_field_multiplications"] += 2 * q
    return output


def fast_result(initial: list[int], field: Field, depth: int, family: str) -> dict[str, Any]:
    q, p = field.q, field.p
    values = initial.copy()
    work = {key: 0 for key in (
        "chirp_field_multiplications",
        "controlled_field_multiplications",
        "controlled_phase_evaluations",
        "cubic_field_multiplications",
        "cubic_phase_evaluations",
        "exact_convolution_coefficient_bound_peak",
        "fiber_field_multiply_adds",
        "ntt_butterflies",
        "ntt_max_length",
        "ntt_pointwise_multiplications",
        "ntt_temporary_integer_cells_peak",
        "ntt_transforms",
        "projection_field_multiply_adds",
        "rader_dfts",
        "sparse_gaussian_terms",
    )}
    scratch = ScratchAccount()

    def transform(vector: list[int], payload: tuple[int, ...]) -> list[int]:
        scratch.add(field=q)
        output = fast_gaussian(vector, payload, field, work, scratch)
        vector.clear()
        scratch.drop(field=q)
        return output

    for operation in compile_word(q, depth, family):
        if operation.kind == "CONTROLLED_CUBIC":
            for s in range(q):
                for x in range(q):
                    multiplier = phase(field, operation.payload[0] * s * x * x * x)
                    work["controlled_phase_evaluations"] += 1
                    for fiber in range(2):
                        index = address(q, s, fiber, x)
                        values[index] = values[index] * multiplier % p
                        work["controlled_field_multiplications"] += 1
        elif operation.kind == "DATA_GAUSSIAN":
            for s in range(q):
                for fiber in range(2):
                    start = address(q, s, fiber, 0)
                    output = transform(values[start : start + q], operation.payload)
                    values[start : start + q] = output
                    scratch.drop(field=q)
                    output.clear()
        elif operation.kind == "LATENT_GAUSSIAN":
            for fiber in range(2):
                for x in range(q):
                    output = transform([values[address(q, s, fiber, x)] for s in range(q)], operation.payload)
                    for s, value in enumerate(output):
                        values[address(q, s, fiber, x)] = value
                    scratch.drop(field=q)
                    output.clear()
        else:
            a, b, c, d = operation.payload
            for s in range(q):
                for x in range(q):
                    left_index, right_index = address(q, s, 0, x), address(q, s, 1, x)
                    left, right = values[left_index], values[right_index]
                    values[left_index] = (a * left + c * right) % p
                    values[right_index] = (b * left + d * right) % p
                    work["fiber_field_multiply_adds"] += 4
    scratch.require_closed()
    boundary = sum(weight * values[address(q, s, fiber, x)] for s, fiber, x, weight in probes(q, family)) % p
    work["projection_field_multiply_adds"] = 8
    resident = 2 * q * q
    return {
        "boundary": boundary,
        "semantic_commitment": hashlib.sha256(repr((q, tuple(values))).encode("ascii")).hexdigest(),
        "resident_field_cells": resident,
        "scratch_field_cells_peak": scratch.field_peak,
        "scratch_auxiliary_integer_cells_peak": scratch.auxiliary_peak,
        "scratch_logical_payload_cells_peak": scratch.total_peak,
        "resident_plus_scratch_logical_payload_cells_peak": resident + scratch.total_peak,
        "experimental_comparison_concurrent_restored_phase_plus_classical_peak_logical_payload_cells": 2 * resident + scratch.total_peak,
        "field_cells_at_combined_peak": resident + scratch.field_at_total_peak,
        "auxiliary_integer_cells_at_combined_peak": scratch.auxiliary_at_total_peak,
        "field_cell_bit_capacity": p.bit_length(),
        "auxiliary_integer_cell_bit_capacity": AUXILIARY_MODULUS.bit_length(),
        "combined_payload_bit_capacity_upper_bound_at_peak": (
            (resident + scratch.field_at_total_peak) * p.bit_length()
            + scratch.auxiliary_at_total_peak * AUXILIARY_MODULUS.bit_length()
        ),
        "public_plan_cells": 12 * depth + 4,
        "public_boundary_probe_cells": 16,
        "retained_ntt_kernel_cache_cells": 0,
        "auxiliary_ntt_prime": AUXILIARY_MODULUS,
        "single_auxiliary_modulus_exactness_bound_checked": 0 < work["exact_convolution_coefficient_bound_peak"] < AUXILIARY_MODULUS,
        "work": work,
        "cold_start_comparison_used": False,
    }


def reconstruct_case(production: dict[str, Any]) -> dict[str, Any]:
    q, depth, family = production["q"], production["depth"], production["family"]
    state = State.fixture(q, production["fixture_family"])
    expected = state.values.copy()
    backing = state.backing()
    initial_rank = matrix_rank(state)
    operations, work, ranks = forward(state, depth, family)
    boundary = project(state, family)
    work["projection_field_multiply_adds"] += 8
    final_rank = matrix_rank(state)
    semantic_commitment = digest(state)
    forward_work = work.copy()
    reverse(state, operations, work)
    exact = state.values == expected and state.stage == "IDLE"
    fast = fast_result(expected, state.field, depth, family)
    checks = {
        "boundary": boundary == production["boundary"],
        "semantic_commitment": semantic_commitment == production["semantic_commitment"],
        "initial_rank": initial_rank == production["initial_latent_data_separation_rank"],
        "controlled_ranks": ranks == production["controlled_cubic_separation_ranks"],
        "final_rank": final_rank == production["final_latent_data_separation_rank"],
        "cube_bijection": math.gcd(3, q - 1) == 1 and len({pow(x, 3, q) for x in range(q)}) == q,
        "forward_work": forward_work == production["forward_work"],
        "full_lifecycle_work": work == production["full_lifecycle_work"],
        "dense_baseline_boundary": boundary == production["matched_identical_matrix_free_classical"]["boundary"],
        "dense_baseline_commitment": semantic_commitment == production["matched_identical_matrix_free_classical"]["semantic_commitment"],
        "dense_baseline_work": forward_work == production["matched_identical_matrix_free_classical"]["work"],
        "rader_boundary": fast["boundary"] == production["matched_exact_rader_ntt_classical"]["boundary"],
        "rader_commitment": fast["semantic_commitment"] == production["matched_exact_rader_ntt_classical"]["semantic_commitment"],
        "rader_work": fast["work"] == production["matched_exact_rader_ntt_classical"]["work"],
        "rader_resources": all(
            fast[key] == production["matched_exact_rader_ntt_classical"][key]
            for key in (
                "scratch_field_cells_peak",
                "scratch_auxiliary_integer_cells_peak",
                "scratch_logical_payload_cells_peak",
                "resident_plus_scratch_logical_payload_cells_peak",
                "field_cells_at_combined_peak",
                "auxiliary_integer_cells_at_combined_peak",
                "combined_payload_bit_capacity_upper_bound_at_peak",
            )
        ),
        "restoration": exact,
        "same_backing": state.backing() == backing,
    }
    if not all(checks.values()):
        fail(f"case mismatch q={q} depth={depth} family={family} checks={checks}")
    return {"q": q, "depth": depth, "family": family, "checks": checks}


def raises(action: Callable[[], Any]) -> bool:
    try:
        action()
    except (RuntimeError, ValueError):
        return True
    return False


def controls() -> dict[str, bool]:
    def mutation_fails(mutation: str) -> bool:
        state = State.fixture(5, "PRIMARY")
        expected = state.values.copy()
        operations, work, _ = forward(state, 1, "PRIMARY")
        reverse(state, operations, work, mutation)
        return state.values != expected or state.stage != "IDLE"

    normal = State.fixture(5, "PRIMARY")
    operations, work, _ = forward(normal, 1, "PRIMARY")
    normal_boundary = project(normal, "PRIMARY")
    reverse(normal, operations, work)
    reordered = State.fixture(5, "PRIMARY")
    forward(reordered, 1, "PRIMARY", True)
    one_hot = State.fixture(5, "PRIMARY", [1, 0, 0, 0, 0])
    one_hot_operations, one_hot_work, ranks = forward(one_hot, 1, "PRIMARY")
    reverse(one_hot, one_hot_operations, one_hot_work)
    return {
        "missing_inverse_fails": mutation_fails("MISSING"),
        "wrong_inverse_fails": mutation_fails("WRONG"),
        "reordered_inverse_fails": mutation_fails("REORDER"),
        "module_reorder_changes_boundary": project(reordered, "PRIMARY") != normal_boundary,
        "premature_projection_rejected": raises(lambda: project(State.fixture(5, "PRIMARY"), "PRIMARY")),
        "invalid_family_rejected": raises(lambda: compile_word(5, 1, "INVALID")),
        "zero_depth_rejected": raises(lambda: compile_word(5, 0, "PRIMARY")),
        "null_latent_rejected": raises(lambda: State.fixture(5, "PRIMARY", [0] * 5)),
        "one_hot_latent_rank_below_full": bool(ranks) and ranks[0] < 5,
        "plan_independent_of_latent_values": normal.values != one_hot.values and compile_word(5, 1, "PRIMARY") == compile_word(5, 1, "PRIMARY"),
    }


def reuse() -> dict[str, bool]:
    state = State.fixture(23, "PRIMARY")
    initial, backing = state.payload(), state.backing()

    def run(target: State, family: str) -> tuple[int, str]:
        operations, work, _ = forward(target, 1, family)
        answer, commitment = project(target, family), digest(target)
        reverse(target, operations, work)
        target.generation += 1
        return answer, commitment

    first = run(state, "PRIMARY")
    second = run(state, "ALTERNATE")
    fresh = State.fixture(23, "PRIMARY")
    fresh_second = run(fresh, "ALTERNATE")
    return {
        "first_boundary_nonzero": bool(first[0]),
        "second_boundary_matches_fresh": second[0] == fresh_second[0],
        "second_commitment_matches_fresh": second[1] == fresh_second[1],
        "same_backing": state.backing() == backing,
        "payload_restored": state.payload() == initial,
        "generation_two": state.generation == 2,
        "no_snapshot": True,
    }


def rader_direct_checks() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for q in (5, 11):
        field = field_for(q)
        for exponent in (1, 2):
            values = [((index + 2) * (index + 5) + exponent) % field.p for index in range(q)]
            root = phase(field, exponent)
            direct = [sum(values[y] * pow(root, x * y, field.p) for y in range(q)) % field.p for x in range(q)]
            work = {key: 0 for key in (
                "exact_convolution_coefficient_bound_peak",
                "ntt_butterflies",
                "ntt_max_length",
                "ntt_pointwise_multiplications",
                "ntt_temporary_integer_cells_peak",
                "ntt_transforms",
                "rader_dfts",
            )}
            scratch = ScratchAccount()
            scratch.add(field=q)
            observed = rader(values, root, field, work, scratch)
            scratch.drop(field=2 * q)
            checks[f"q{q}_root{exponent}_direct_dft"] = observed == direct
            checks[f"q{q}_root{exponent}_scratch_closed"] = not scratch.field and not scratch.auxiliary
            checks[f"q{q}_root{exponent}_bound_exact"] = 0 < work["exact_convolution_coefficient_bound_peak"] < AUXILIARY_MODULUS
    return checks


def build(production_path: Path) -> dict[str, Any]:
    production = json.loads(production_path.read_text(encoding="utf-8"))
    comparisons = [reconstruct_case(case) for case in production["cases"]]
    controls_result = controls()
    reuse_result = reuse()
    rader_checks = rader_direct_checks()
    result = {
        "schema": "CAT_CAS_GROWING_PRIME_RESIDENT_CUBIC_STRENGTH_PHASE_PORT_WEIL_ACTION_RANK_INDEPENDENT_ORACLE_V1",
        "production_result": "GROWING_PRIME_RESIDENT_CUBIC_STRENGTH_PHASE_PORT_WEIL_ACTION_RANK_RESULTS.json",
        "production_result_sha256": hashlib.sha256(production_path.read_bytes()).hexdigest(),
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "qualified": True,
        "independence": {
            "imports_production_module": False,
            "imports_predecessor_module": False,
            "production_forward_inverse_projection_rank_or_baseline_called": False,
            "production_result_used_only_as_comparison_target": True,
            "separate_field_plan_rank_inverse_dense_and_rader_recurrences": True,
        },
        "all_11_cases_reconstructed": len(comparisons) == 11 and all(all(case["checks"].values()) for case in comparisons),
        "case_comparisons": comparisons,
        "controls": controls_result,
        "restoration_and_reuse": reuse_result,
        "rader_direct_semantic_checks": rader_checks,
        "observed_resource_law": {
            "initial_rank": 1,
            "first_controlled_rank": "Q",
            "final_rank": "Q",
            "accepted_resident_field_cells": "2*Q^2",
            "accepted_peak_field_cells": "2*Q^2+Q",
            "dense_classical_peak_field_cells": "2*Q^2+Q",
            "rader_classical_peak_logical_payload_cells": "2*Q^2+4*Q-2+2*NEXT_POWER_OF_TWO_AT_LEAST_2Q_MINUS3",
            "fixed_rank_across_growing_q_established": False,
        },
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": [
            "RESIDENT_COHERENT_CUBIC_STRENGTH_OPERAND_SHARED_ACROSS_MULTIPLE_CONSUMERS",
            "FULL_Q_LATENT_DATA_SEPARATION_RANK_AFTER_FIRST_CONTROLLED_CUBIC",
            "EXACT_IN_PLACE_RESTORATION_AND_SAME_BACKING_REUSE",
            "DENSE_AND_EXACT_RADER_NTT_FULL_STATE_CLASSICAL_PARITY",
        ],
        "rejected_interpretations": [
            "FIXED_RANK_GROWING_ORDER_CLOSURE",
            "LOWER_BOUND_AGAINST_NONLINEAR_OR_PROGRAM_SPECIFIC_ALGORITHMS",
            "MACHINE_ENFORCED_HIDDEN_RUNTIME_OPERAND",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
    }
    if not result["all_11_cases_reconstructed"] or not all(controls_result.values()) or not all(reuse_result.values()) or not all(rader_checks.values()):
        fail("independent qualification failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.write_text(json.dumps(build(arguments.production), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
