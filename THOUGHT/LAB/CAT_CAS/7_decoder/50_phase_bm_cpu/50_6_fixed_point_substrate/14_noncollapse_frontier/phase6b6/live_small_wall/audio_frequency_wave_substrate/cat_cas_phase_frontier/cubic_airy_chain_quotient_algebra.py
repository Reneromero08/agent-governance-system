#!/usr/bin/env python3
"""M236: exact cubic Airy-chain stationary quotient-algebra carrier.

The experiment is formal algebraic stationary composition.  It does not
evaluate a convergent oscillatory integral or enumerate stationary roots.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence


PORT_TYPE = "CUBIC_AIRY_CHAIN_QUOTIENT_ALGEBRA_V1"
ORIENTATION = "LEFT_TO_RIGHT"
DEPTHS = (2, 3, 4, 5, 6)
BOUNDARIES = ((-3, 1), (-2, 1))
ALPHA_FAMILIES = (
    (1, -1, 1, 1, -1, 1),
    (-1, 1, 1, -1, -1, 1),
)

RESULT = "PASS_EXACT_CUBIC_AIRY_CHAIN_QUOTIENT_ALGEBRA_STRICT_SCOPE"
CLAIM = (
    "EXACT_CONNECTED_CUBIC_AIRY_CHAIN_PHASE_SIGNATURES_RETAIN_ALL_"
    "STATIONARY_SHARED_PORT_BRANCHES_IMPLICITLY_IN_Q_OF_T_MOD_PN_AND_"
    "ACCUMULATE_THE_SCALED_CRITICAL_VALUE_ON_THREE_ACTUAL_QUOTIENT_"
    "ALGEBRA_BACKINGS_WITH_FINAL_ONLY_CHARACTERISTIC_NORM_RELATION_"
    "EXACT_SAME_BACKING_RESTORATION_AND_UNRELATED_SENTINEL_REUSE_"
    "THROUGH_DECLARED_LENGTHS2_3_4_5_6_BUT_QUOTIENT_DIMENSION_AND_FINAL_"
    "RELATION_DEGREE_GROW2_4_8_16_32_AND_THE_IDENTICAL_EXACT_POLYNOMIAL_"
    "CLASSICAL_RECURRENCE_REMAINS"
)
CLAIM_CEILING = (
    "FORMAL_PROJECTIVE_CUBIC_AIRY_CHAIN_STATIONARY_RELATIONS_OVER_Q_"
    "TWO_RATIONAL_BOUNDARY_FAMILIES_LENGTHS2_3_4_5_6_SIMPLE_ENDPOINT_AND_"
    "CRITICAL_VALUE_POLYNOMIALS_SCALED_PHASE_W_EQUALS3PHI_DIRECT_PROCESS_ONLY"
)


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def payload_bits(values: Iterable[int]) -> int:
    return sum(signed_bits(value) for value in values)


def trim(values: Sequence[int]) -> list[int]:
    result = list(values)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def add(left: Sequence[int], right: Sequence[int]) -> list[int]:
    size = max(len(left), len(right))
    result = [0] * size
    for index in range(size):
        result[index] = (left[index] if index < len(left) else 0) + (
            right[index] if index < len(right) else 0
        )
    return trim(result)


def subtract(left: Sequence[int], right: Sequence[int]) -> list[int]:
    size = max(len(left), len(right))
    result = [0] * size
    for index in range(size):
        result[index] = (left[index] if index < len(left) else 0) - (
            right[index] if index < len(right) else 0
        )
    return trim(result)


def scale(values: Sequence[int], scalar: int) -> list[int]:
    return trim([scalar * value for value in values])


def multiply(left: Sequence[int], right: Sequence[int]) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return trim(result)


def monic(values: Sequence[int]) -> list[int]:
    result = trim(values)
    if result[-1] == -1:
        return [-value for value in result]
    if result[-1] != 1:
        raise RuntimeError("accepted endpoint polynomial must be integral monic")
    return result


def reduce_mod(values: Sequence[int], modulus: Sequence[int]) -> list[int]:
    modulus = monic(modulus)
    result = trim(values)
    degree = len(modulus) - 1
    while len(result) - 1 >= degree:
        coefficient = result[-1]
        offset = len(result) - len(modulus)
        if coefficient:
            for index, value in enumerate(modulus):
                result[offset + index] -= coefficient * value
        result = trim(result)
    result.extend([0] * (degree - len(result)))
    return result


def quotient_add(left: Sequence[int], right: Sequence[int], modulus: Sequence[int]) -> list[int]:
    return reduce_mod(add(left, right), modulus)


def quotient_subtract(
    left: Sequence[int], right: Sequence[int], modulus: Sequence[int]
) -> list[int]:
    return reduce_mod(subtract(left, right), modulus)


def quotient_multiply(
    left: Sequence[int], right: Sequence[int], modulus: Sequence[int]
) -> list[int]:
    return reduce_mod(multiply(left, right), modulus)


def quotient_scale(values: Sequence[int], scalar: int, modulus: Sequence[int]) -> list[int]:
    return reduce_mod(scale(values, scalar), modulus)


def polynomial_commitment(values: Sequence[int]) -> str:
    encoded = json.dumps(list(values), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix_commitment(values: Sequence[Sequence[int]]) -> str:
    encoded = json.dumps([list(row) for row in values], separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def polynomial_divmod_fraction(
    numerator: Sequence[int], denominator: Sequence[int]
) -> tuple[list[Fraction], list[Fraction]]:
    num = [Fraction(value) for value in trim(numerator)]
    den = [Fraction(value) for value in trim(denominator)]
    quotient = [Fraction(0)] * max(1, len(num) - len(den) + 1)
    while len(num) >= len(den) and any(num):
        coefficient = num[-1] / den[-1]
        offset = len(num) - len(den)
        quotient[offset] += coefficient
        for index, value in enumerate(den):
            num[offset + index] -= coefficient * value
        while len(num) > 1 and num[-1] == 0:
            num.pop()
    return quotient, num


def square_free(values: Sequence[int]) -> bool:
    values = trim(values)
    derivative = [index * values[index] for index in range(1, len(values))]
    left = [Fraction(value) for value in values]
    right = [Fraction(value) for value in trim(derivative)]
    while len(right) > 1 or right[0] != 0:
        _, remainder = polynomial_divmod_fraction(left, right)
        left, right = right, remainder
    return len(left) == 1


def program_digest(
    family: int, length: int, x: int, z: int, alphas: Sequence[int]
) -> str:
    encoded = json.dumps(
        [family, length, x, z, list(alphas)], separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PublicProgram:
    family: int
    length: int
    x: int
    z: int
    alphas: tuple[int, ...]
    program_id: str

    def __post_init__(self) -> None:
        if self.family not in (0, 1) or self.length not in DEPTHS:
            raise ValueError("invalid public cubic Airy-chain descriptor")
        if len(self.alphas) != self.length or any(
            not isinstance(alpha, int) or alpha == 0 for alpha in self.alphas
        ):
            raise ValueError("cubic coefficients must be nonzero integers")
        if self.program_id != program_digest(
            self.family, self.length, self.x, self.z, self.alphas
        ):
            raise ValueError("public program identity is not bound to its descriptor")

    @staticmethod
    def compile(family: int, length: int) -> "PublicProgram":
        if family not in (0, 1) or length not in DEPTHS:
            raise ValueError("invalid public cubic Airy-chain program")
        x, z = BOUNDARIES[family]
        alphas = ALPHA_FAMILIES[family][:length]
        if len(alphas) != length or any(alpha == 0 for alpha in alphas):
            raise ValueError("cubic coefficients must be nonzero")
        return PublicProgram(
            family, length, x, z, alphas,
            program_digest(family, length, x, z, alphas),
        )


def coordinate_polynomials(program: PublicProgram) -> list[list[int]]:
    coordinates = [[program.x], [0, 1]]
    for index in range(1, program.length):
        next_coordinate = subtract(
            scale(coordinates[index], 0),
            add(coordinates[index - 1], scale(multiply(coordinates[index], coordinates[index]), program.alphas[index - 1])),
        )
        coordinates.append(next_coordinate)
    return coordinates


def endpoint_modulus(program: PublicProgram) -> tuple[list[int], list[list[int]]]:
    coordinates = coordinate_polynomials(program)
    endpoint = subtract(coordinates[-1], [program.z])
    modulus = monic(endpoint)
    expected = 1 << (program.length - 1)
    if len(modulus) - 1 != expected:
        raise RuntimeError("endpoint degree law failed")
    return modulus, coordinates


def phase_term(
    left: Sequence[int], right: Sequence[int], alpha: int, modulus: Sequence[int]
) -> list[int]:
    # W=3*Phi, so one term is 3*left*right + alpha*right^3.
    product = quotient_multiply(left, right, modulus)
    square = quotient_multiply(right, right, modulus)
    cube = quotient_multiply(square, right, modulus)
    return quotient_add(quotient_scale(product, 3, modulus), quotient_scale(cube, alpha, modulus), modulus)


@dataclass
class Work:
    forward_phase_terms: int = 0
    forward_maps: int = 0
    inverse_maps: int = 0
    inverse_phase_terms: int = 0
    characteristic_matrix_products: int = 0
    retained_dynamic_inverse_history_entries: int = 0


def multiplication_matrix(element: Sequence[int], modulus: Sequence[int]) -> list[list[int]]:
    degree = len(modulus) - 1
    matrix = [[0] * degree for _ in range(degree)]
    basis = [1] + [0] * (degree - 1)
    for column in range(degree):
        product = quotient_multiply(element, basis, modulus)
        for row, value in enumerate(product):
            matrix[row][column] = value
        basis = quotient_multiply(basis, [0, 1], modulus)
    return matrix


def matrix_multiply(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]) -> list[list[int]]:
    size = len(left)
    result = [[0] * size for _ in range(size)]
    for i in range(size):
        for k in range(size):
            if left[i][k] == 0:
                continue
            factor = left[i][k]
            for j in range(size):
                result[i][j] += factor * right[k][j]
    return result


def characteristic_polynomial(matrix: Sequence[Sequence[int]], work: Work | None = None) -> list[int]:
    """Exact Faddeev-LeVerrier characteristic polynomial, low order first."""
    size = len(matrix)
    identity = [[int(i == j) for j in range(size)] for i in range(size)]
    current = identity
    del identity
    high_order = [1]
    for order in range(1, size + 1):
        product = matrix_multiply(matrix, current)
        numerator = -sum(product[index][index] for index in range(size))
        if numerator % order:
            raise RuntimeError("nonintegral characteristic coefficient")
        coefficient = numerator // order
        high_order.append(coefficient)
        for index in range(size):
            product[index][index] += coefficient
        current = product
        if work is not None:
            work.characteristic_matrix_products += 1
    return list(reversed(high_order))


def evaluate_quotient_polynomial(
    coefficients: Sequence[int], value: Sequence[int], modulus: Sequence[int]
) -> list[int]:
    result = [0] * (len(modulus) - 1)
    for coefficient in reversed(coefficients):
        result = quotient_multiply(result, value, modulus)
        result[0] += coefficient
    return reduce_mod(result, modulus)


def direct_phase_and_norm(program: PublicProgram, work: Work | None = None) -> dict[str, object]:
    modulus, coordinates = endpoint_modulus(program)
    degree = len(modulus) - 1
    reduced = [reduce_mod(item, modulus) for item in coordinates]
    critical = [0] * degree
    for index in range(1, program.length):
        critical = quotient_add(
            critical,
            phase_term(reduced[index - 1], reduced[index], program.alphas[index - 1], modulus),
            modulus,
        )
    final_coordinate = [program.z] + [0] * (degree - 1)
    critical = quotient_add(
        critical,
        phase_term(reduced[-2], final_coordinate, program.alphas[-1], modulus),
        modulus,
    )
    matrix = multiplication_matrix(critical, modulus)
    relation = characteristic_polynomial(matrix, work)
    if not square_free(modulus) or not square_free(relation):
        raise RuntimeError("accepted endpoint and critical-value relations must be square-free")
    if any(evaluate_quotient_polynomial(relation, critical, modulus)):
        raise RuntimeError("critical-value relation does not annihilate the resident action")
    return {
        "modulus": modulus,
        "critical": critical,
        "relation": relation,
        "matrix": matrix,
        "coordinates": reduced,
    }


class CubicAiryPort:
    def __init__(self, program: PublicProgram) -> None:
        self.modulus, _ = endpoint_modulus(program)
        degree = len(self.modulus) - 1
        self.r = [program.x] + [0] * (degree - 1)
        self.s = [0, 1] + [0] * (degree - 2)
        self.h = [0] * degree
        self.initial_x = program.x
        self.owner: int | None = None
        self.program_id: str | None = None
        self.program_descriptor: tuple[object, ...] | None = None
        self.port_type: str | None = None
        self.generation = 0
        self.last_restored_generation = 0
        self.cursor = 0
        self.final_phase_resident = False
        self.leased = False

    def canonical_r(self) -> list[int]:
        return [self.initial_x] + [0] * (len(self.modulus) - 2)

    def canonical_s(self) -> list[int]:
        return [0, 1] + [0] * (len(self.modulus) - 3)

    def canonical_state(self) -> tuple[object, ...]:
        return (
            tuple(self.modulus), tuple(self.r), tuple(self.s), tuple(self.h),
            self.owner, self.program_id, self.program_descriptor, self.port_type, self.generation,
            self.last_restored_generation, self.cursor,
            self.final_phase_resident, self.leased,
        )

    def lease(self, owner: int, program: PublicProgram, generation: int, port_type: str = PORT_TYPE) -> None:
        expected_modulus, _ = endpoint_modulus(program)
        if self.leased or self.modulus != expected_modulus:
            raise RuntimeError("carrier is unavailable or has the wrong quotient modulus")
        if self.r != self.canonical_r() or self.s != self.canonical_s() or any(self.h):
            raise RuntimeError("carrier is not canonical")
        if owner <= 0 or port_type != PORT_TYPE or generation != self.last_restored_generation + 1:
            raise RuntimeError("invalid carrier lease")
        self.owner = owner
        self.program_id = program.program_id
        self.program_descriptor = (
            program.family, program.length, program.x, program.z, program.alphas
        )
        self.port_type = port_type
        self.generation = generation
        self.leased = True

    def require(self, owner: int, program: PublicProgram, generation: int) -> None:
        if not self.leased:
            raise RuntimeError("carrier is not leased")
        if (
            self.owner != owner or self.program_id != program.program_id
            or self.program_descriptor != (
                program.family, program.length, program.x, program.z, program.alphas
            )
            or self.port_type != PORT_TYPE or self.generation != generation
        ):
            raise RuntimeError("quotient carrier custody mismatch")

    def forward_module(self, owner: int, program: PublicProgram, generation: int, index: int, work: Work) -> None:
        self.require(owner, program, generation)
        if index != self.cursor or index >= program.length - 1 or self.final_phase_resident:
            raise RuntimeError("forward module dependency failure")
        alpha = program.alphas[index]
        phase = phase_term(self.r, self.s, alpha, self.modulus)
        self.h[:] = quotient_add(self.h, phase, self.modulus)
        self.r[:], self.s[:] = self.s[:], self.r[:]
        square = quotient_multiply(self.r, self.r, self.modulus)
        self.s[:] = quotient_subtract(quotient_scale(self.s, -1, self.modulus), quotient_scale(square, alpha, self.modulus), self.modulus)
        self.cursor += 1
        work.forward_phase_terms += 1
        work.forward_maps += 1

    def add_final_phase(self, owner: int, program: PublicProgram, generation: int, work: Work) -> None:
        self.require(owner, program, generation)
        if self.cursor != program.length - 1 or self.final_phase_resident:
            raise RuntimeError("final phase dependency failure")
        endpoint = [program.z] + [0] * (len(self.modulus) - 2)
        if self.s != endpoint:
            raise RuntimeError("endpoint relation is not resident")
        phase = phase_term(self.r, endpoint, program.alphas[-1], self.modulus)
        self.h[:] = quotient_add(self.h, phase, self.modulus)
        self.final_phase_resident = True
        work.forward_phase_terms += 1

    def project_final(self, owner: int, program: PublicProgram, generation: int, work: Work) -> list[int]:
        self.require(owner, program, generation)
        if self.cursor != program.length - 1 or not self.final_phase_resident:
            raise RuntimeError("premature critical-value projection")
        relation = characteristic_polynomial(multiplication_matrix(self.h, self.modulus), work)
        if any(evaluate_quotient_polynomial(relation, self.h, self.modulus)):
            raise RuntimeError("projected relation does not annihilate resident critical value")
        return relation

    def project_stationary_port(self) -> None:
        raise RuntimeError("stationary branch projection is forbidden")

    def remove_final_phase(self, owner: int, program: PublicProgram, generation: int, work: Work) -> None:
        self.require(owner, program, generation)
        if not self.final_phase_resident or self.cursor != program.length - 1:
            raise RuntimeError("final inverse dependency failure")
        endpoint = [program.z] + [0] * (len(self.modulus) - 2)
        phase = phase_term(self.r, endpoint, program.alphas[-1], self.modulus)
        self.h[:] = quotient_subtract(self.h, phase, self.modulus)
        self.final_phase_resident = False
        work.inverse_phase_terms += 1

    def inverse_module(self, owner: int, program: PublicProgram, generation: int, index: int, work: Work) -> None:
        self.require(owner, program, generation)
        if index != self.cursor - 1 or self.final_phase_resident:
            raise RuntimeError("inverse module dependency failure")
        alpha = program.alphas[index]
        square = quotient_multiply(self.r, self.r, self.modulus)
        self.s[:] = quotient_subtract(quotient_scale(self.s, -1, self.modulus), quotient_scale(square, alpha, self.modulus), self.modulus)
        self.r[:], self.s[:] = self.s[:], self.r[:]
        phase = phase_term(self.r, self.s, alpha, self.modulus)
        self.h[:] = quotient_subtract(self.h, phase, self.modulus)
        self.cursor -= 1
        work.inverse_maps += 1
        work.inverse_phase_terms += 1

    def release(self, owner: int, program: PublicProgram, generation: int) -> None:
        self.require(owner, program, generation)
        if (
            self.cursor != 0 or self.final_phase_resident or any(self.h)
            or self.r != self.canonical_r() or self.s != self.canonical_s()
        ):
            raise RuntimeError("quotient carrier is not exactly restored")
        self.last_restored_generation = generation
        self.owner = None
        self.program_id = None
        self.program_descriptor = None
        self.port_type = None
        self.generation = 0
        self.leased = False


def run_transaction(port: CubicAiryPort, program: PublicProgram, generation: int, owner: int = 23601) -> dict[str, object]:
    if port is None:
        raise TypeError("null cubic Airy-chain carrier")
    backings = (id(port.r), id(port.s), id(port.h), id(port.modulus))
    work = Work()
    port.lease(owner, program, generation)
    for index in range(program.length - 1):
        port.forward_module(owner, program, generation, index, work)
    port.add_final_phase(owner, program, generation, work)
    resident_commitment = polynomial_commitment(port.h)
    relation = port.project_final(owner, program, generation, work)
    retained_relation = tuple(relation)
    port.remove_final_phase(owner, program, generation, work)
    for index in range(program.length - 2, -1, -1):
        port.inverse_module(owner, program, generation, index, work)
    restored_before_release = (
        port.r == port.canonical_r() and port.s == port.canonical_s()
        and not any(port.h) and port.cursor == 0 and not port.final_phase_resident
    )
    same_backings = backings == (id(port.r), id(port.s), id(port.h), id(port.modulus))
    port.release(owner, program, generation)
    if tuple(relation) != retained_relation:
        raise RuntimeError("final relation was consumed by inverse")
    baseline = direct_phase_and_norm(program)
    if (
        relation != baseline["relation"]
        or resident_commitment != polynomial_commitment(baseline["critical"])
    ):
        raise RuntimeError("post-restoration identical polynomial classical baseline mismatch")
    degree = len(port.modulus) - 1
    return {
        "family": program.family,
        "length": program.length,
        "quotient_dimension": degree,
        "endpoint_polynomial_degree": degree,
        "critical_value_relation_degree": len(relation) - 1,
        "endpoint_polynomial_square_free": square_free(port.modulus),
        "critical_value_relation_square_free": square_free(relation),
        "critical_value_relation_annihilates_resident_action": not any(
            evaluate_quotient_polynomial(relation, baseline["critical"], port.modulus)
        ),
        "modulus_commitment": polynomial_commitment(port.modulus),
        "resident_critical_value_commitment": resident_commitment,
        "critical_value_relation_commitment": polynomial_commitment(relation),
        "modulus_coefficient_cells": len(port.modulus),
        "carrier_quotient_coefficient_cells": 3 * degree,
        "retained_final_relation_coefficient_cells_during_inverse": len(relation),
        "modulus_payload_bits": payload_bits(port.modulus),
        "resident_critical_value_payload_bits": payload_bits(baseline["critical"]),
        "critical_value_relation_payload_bits": payload_bits(relation),
        "canonical_post_inverse_state_exact": restored_before_release,
        "same_r_s_h_modulus_backings": same_backings,
        "restoration_generation": generation,
        "baseline_reload_used": False,
        "retained_dynamic_inverse_history_entries": work.retained_dynamic_inverse_history_entries,
        "work": asdict(work),
        "matched_classical_relation_commitment": polynomial_commitment(baseline["relation"]),
    }


def run_sentinel(port: CubicAiryPort, program: PublicProgram, generation: int, gamma: int = 2, owner: int = 23602) -> dict[str, object]:
    backings = (id(port.r), id(port.s), id(port.h), id(port.modulus))
    port.lease(owner, program, generation)
    product = quotient_multiply(port.r, port.s, port.modulus)
    port.h[:] = quotient_add(port.h, product, port.modulus)
    square = quotient_multiply(port.s, port.s, port.modulus)
    port.r[:] = quotient_add(port.r, quotient_scale(square, gamma, port.modulus), port.modulus)
    boundary = characteristic_polynomial(multiplication_matrix(port.h, port.modulus))
    retained = tuple(boundary)
    port.r[:] = quotient_subtract(port.r, quotient_scale(square, gamma, port.modulus), port.modulus)
    port.h[:] = quotient_subtract(port.h, product, port.modulus)
    exact = port.r == port.canonical_r() and port.s == port.canonical_s() and not any(port.h)
    same = backings == (id(port.r), id(port.s), id(port.h), id(port.modulus))
    port.release(owner, program, generation)
    if tuple(boundary) != retained:
        raise RuntimeError("sentinel output was consumed by inverse")
    return {
        "boundary_commitment": polynomial_commitment(boundary),
        "boundary_payload_bits": payload_bits(boundary),
        "canonical_post_inverse_state_exact": exact,
        "same_r_s_h_modulus_backings": same,
        "restoration_generation": generation,
        "baseline_reload_used": False,
    }


def expect_rejected(callback) -> bool:
    try:
        callback()
    except (RuntimeError, TypeError, ValueError):
        return True
    return False


def controls() -> dict[str, bool]:
    program = PublicProgram.compile(0, 3)
    owner = 23611
    direct = direct_phase_and_norm(PublicProgram.compile(0, 2))
    relation = direct["relation"]
    alpha1, alpha2 = PublicProgram.compile(0, 2).alphas
    x, z = PublicProgram.compile(0, 2).x, PublicProgram.compile(0, 2).z
    # alpha1*(W-alpha2*z^3)^2 + 4*(x+z)^3, normalized to monic.
    expected_n2 = [
        alpha1 * alpha2 * alpha2 * z**6 + 4 * (x + z) ** 3,
        -2 * alpha1 * alpha2 * z**3,
        alpha1,
    ]
    if expected_n2[-1] == -1:
        expected_n2 = [-value for value in expected_n2]
    n2_formula = relation == expected_n2

    premature = CubicAiryPort(program)
    premature.lease(owner, program, 1)
    premature_projection = expect_rejected(lambda: premature.project_final(owner, program, 1, Work()))
    stationary_projection = expect_rejected(premature.project_stationary_port)
    wrong_owner = expect_rejected(lambda: premature.forward_module(owner + 1, program, 1, 0, Work()))
    wrong_generation = expect_rejected(lambda: premature.forward_module(owner, program, 2, 0, Work()))
    wrong_program = expect_rejected(lambda: premature.forward_module(owner, PublicProgram.compile(1, 3), 1, 0, Work()))
    forged_program_identity = expect_rejected(
        lambda: PublicProgram(
            program.family, program.length, program.x, program.z,
            program.alphas[:-1] + (-program.alphas[-1],), program.program_id,
        )
    )
    wrong_type = expect_rejected(lambda: CubicAiryPort(program).lease(owner, program, 1, "WRONG"))
    null_carrier = expect_rejected(lambda: run_transaction(None, program, 1))  # type: ignore[arg-type]

    all_stationary_equations = True
    for family in (0, 1):
        for length in DEPTHS:
            declared = PublicProgram.compile(family, length)
            declared_modulus, declared_coordinates = endpoint_modulus(declared)
            for index in range(1, declared.length):
                stationary = add(
                    declared_coordinates[index - 1],
                    add(
                        scale(multiply(declared_coordinates[index], declared_coordinates[index]), declared.alphas[index - 1]),
                        declared_coordinates[index + 1],
                    ),
                )
                all_stationary_equations &= not any(reduce_mod(stationary, declared_modulus))

    missing = CubicAiryPort(program)
    missing.lease(owner, program, 1)
    for index in range(program.length - 1):
        missing.forward_module(owner, program, 1, index, Work())
    missing.add_final_phase(owner, program, 1, Work())
    missing_inverse = expect_rejected(lambda: missing.release(owner, program, 1))
    reordered_port = CubicAiryPort(program)
    reordered_port.lease(owner, program, 1)
    for index in range(program.length - 1):
        reordered_port.forward_module(owner, program, 1, index, Work())
    reordered_port.add_final_phase(owner, program, 1, Work())
    reordered_port.remove_final_phase(owner, program, 1, Work())
    reordered_inverse = expect_rejected(
        lambda: reordered_port.inverse_module(owner, program, 1, 0, Work())
    )

    wrong_inverse_port = CubicAiryPort(program)
    wrong_inverse_port.lease(owner, program, 1)
    for index in range(program.length - 1):
        wrong_inverse_port.forward_module(owner, program, 1, index, Work())
    wrong_inverse_port.add_final_phase(owner, program, 1, Work())
    endpoint_value = [program.z] + [0] * (len(wrong_inverse_port.modulus) - 2)
    wrong_final = phase_term(
        wrong_inverse_port.r, endpoint_value, -program.alphas[-1], wrong_inverse_port.modulus
    )
    wrong_inverse_port.h[:] = quotient_subtract(
        wrong_inverse_port.h, wrong_final, wrong_inverse_port.modulus
    )
    wrong_inverse_port.final_phase_resident = False
    for index in range(program.length - 2, -1, -1):
        wrong_inverse_port.inverse_module(owner, program, 1, index, Work())
    wrong_inverse = expect_rejected(
        lambda: wrong_inverse_port.release(owner, program, 1)
    )

    dirty_port = CubicAiryPort(program)
    dirty_port.h[0] = 1
    dirty_carrier = expect_rejected(lambda: dirty_port.lease(owner, program, 1))

    stale = CubicAiryPort(program)
    run_transaction(stale, program, 1, owner)
    run_sentinel(stale, program, 2, owner=owner)
    stale_generation = expect_rejected(lambda: stale.lease(owner, program, 2))
    zero_alpha = expect_rejected(
        lambda: PublicProgram(0, 2, -3, 1, (0, 1), "zero")
    )

    original = direct_phase_and_norm(PublicProgram.compile(0, 4))
    reordered_program = PublicProgram.compile(0, 4)
    swapped = PublicProgram(
        0, 4, reordered_program.x, reordered_program.z,
        (reordered_program.alphas[1], reordered_program.alphas[0]) + reordered_program.alphas[2:],
        program_digest(
            0, 4, reordered_program.x, reordered_program.z,
            (reordered_program.alphas[1], reordered_program.alphas[0]) + reordered_program.alphas[2:],
        ),
    )
    reordered_changes = (
        polynomial_commitment(direct_phase_and_norm(swapped)["relation"])
        != polynomial_commitment(original["relation"])
    )
    deleted = PublicProgram(
        0, 4, reordered_program.x, reordered_program.z,
        (reordered_program.alphas[0], reordered_program.alphas[1], -reordered_program.alphas[2], reordered_program.alphas[3]),
        program_digest(
            0, 4, reordered_program.x, reordered_program.z,
            (reordered_program.alphas[0], reordered_program.alphas[1], -reordered_program.alphas[2], reordered_program.alphas[3]),
        ),
    )
    perturbation_changes = (
        polynomial_commitment(direct_phase_and_norm(deleted)["relation"])
        != polynomial_commitment(original["relation"])
    )
    return {
        "direct_n2_eliminated_critical_value_formula": n2_formula,
        "all_declared_stationary_equations_vanish_in_endpoint_quotient": all_stationary_equations,
        "premature_final_projection_rejected": premature_projection,
        "stationary_port_projection_rejected": stationary_projection,
        "wrong_owner_rejected": wrong_owner,
        "wrong_program_rejected": wrong_program,
        "same_id_changed_descriptor_rejected": forged_program_identity,
        "wrong_type_rejected": wrong_type,
        "wrong_generation_rejected": wrong_generation,
        "stale_generation_rejected": stale_generation,
        "null_carrier_rejected": null_carrier,
        "missing_inverse_rejected": missing_inverse,
        "reordered_inverse_rejected": reordered_inverse,
        "wrong_alpha_inverse_rejected": wrong_inverse,
        "dirty_carrier_rejected": dirty_carrier,
        "zero_cubic_coefficient_rejected": zero_alpha,
        "module_reorder_changes_critical_value_relation": reordered_changes,
        "semantic_alpha_perturbation_changes_critical_value_relation": perturbation_changes,
        "stationary_roots_or_branch_lists_materialized": False,
        "intermediate_quotient_elements_serialized": False,
        "public_compiler_reads_final_answer": False,
    }


def comparable_case(case: dict[str, object]) -> dict[str, object]:
    keys = (
        "family", "length", "quotient_dimension", "endpoint_polynomial_degree",
        "critical_value_relation_degree", "endpoint_polynomial_square_free",
        "critical_value_relation_square_free",
        "critical_value_relation_annihilates_resident_action", "modulus_commitment",
        "resident_critical_value_commitment", "critical_value_relation_commitment",
        "modulus_coefficient_cells", "carrier_quotient_coefficient_cells",
        "retained_final_relation_coefficient_cells_during_inverse",
        "modulus_payload_bits", "resident_critical_value_payload_bits",
        "critical_value_relation_payload_bits", "canonical_post_inverse_state_exact",
        "same_r_s_h_modulus_backings", "restoration_generation",
        "baseline_reload_used", "retained_dynamic_inverse_history_entries",
        "matched_classical_relation_commitment",
    )
    return {key: case[key] for key in keys}


def main(reference_path: Path) -> None:
    reference = json.loads(reference_path.read_text())
    production_controls = controls()
    if production_controls != reference["controls"]:
        raise RuntimeError("independent control parity failed")
    cases = []
    for family in (0, 1):
        for length in DEPTHS:
            program = PublicProgram.compile(family, length)
            cases.append(run_transaction(CubicAiryPort(program), program, 1))
    if [comparable_case(case) for case in cases] != reference["cases"]:
        raise RuntimeError("independent case parity failed")

    primary_program = PublicProgram.compile(0, 5)
    shared = CubicAiryPort(primary_program)
    primary = run_transaction(shared, primary_program, 1)
    reuse = run_sentinel(shared, primary_program, 2)
    fresh = CubicAiryPort(primary_program)
    fresh_reuse = run_sentinel(fresh, primary_program, 1)
    if reuse["boundary_commitment"] != fresh_reuse["boundary_commitment"]:
        raise RuntimeError("fresh/restored sentinel boundary mismatch")
    if reference["reuse"]["reuse"]["boundary_commitment"] != reuse["boundary_commitment"]:
        raise RuntimeError("independent reuse parity failed")

    dimensions = [case["quotient_dimension"] for case in cases if case["family"] == 0]
    payloads = [case["critical_value_relation_payload_bits"] for case in cases if case["family"] == 0]
    output = {
        "schema": "cat_cas.cubic_airy_chain_quotient_algebra_result.v1",
        "result": RESULT,
        "claim": CLAIM,
        "claim_ceiling": CLAIM_CEILING,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "cases": cases,
        "controls": production_controls,
        "reuse": {
            "primary": primary,
            "reuse": reuse,
            "fresh_reuse": fresh_reuse,
            "restoration_generation_after_reuse": shared.last_restored_generation,
            "fresh_restored_boundary_agreement": reuse["boundary_commitment"] == fresh_reuse["boundary_commitment"],
            "same_backing_across_primary_and_reuse": primary["same_r_s_h_modulus_backings"] and reuse["same_r_s_h_modulus_backings"],
        },
        "phase_relation_law": {
            "formal_cubic_signature": "S_alpha(u,v)=u*v+alpha*v^3/3",
            "scaled_action_coordinate": "W=3*Phi",
            "stationary_recurrence": "q_(i+1)=-q_(i-1)-alpha_i*q_i^2",
            "endpoint_quotient": "Q[t]/(q_n(t)-z)",
            "native_reversible_map": "T_alpha(r,s)=(s,-r-alpha*s^2)",
            "final_relation": "Norm_(Q[t]/P_n)(W-H_n)",
            "stationary_branches_unresolved": True,
            "stationary_roots_enumerated": False,
            "resident_critical_value_exported": False,
            "final_boundary_only": True,
            "direct_process_logical_custody_only": True,
        },
        "resource_law": {
            "quotient_dimensions_by_length": dimensions,
            "critical_value_relation_payload_bits_family0": payloads,
            "carrier_quotient_elements": 3,
            "carrier_coefficient_cells_by_length": [3 * value for value in dimensions],
            "immutable_modulus_coefficient_cells_by_length": [value + 1 for value in dimensions],
            "final_relation_output_coefficient_cells_by_length": [value + 1 for value in dimensions],
            "retained_dynamic_inverse_history_entries": 0,
            "carrier_initial_state_snapshot_retained": False,
            "quotient_arithmetic_operation_counts_instrumented": False,
            "characteristic_matrix_is_projection_and_verification_scratch": True,
            "characteristic_matrix_cells_by_length": [value * value for value in dimensions],
            "characteristic_polynomial_component_peak_matrix_cells_by_length": [3 * value * value for value in dimensions],
            "whole_transaction_live_cell_and_payload_accounting_complete": False,
            "python_object_allocator_interpreter_serialization_rss_excluded": True,
            "resource_verification_level": "PACKAGE_SELF_REVIEW",
        },
        "matched_classical": {
            "strongest_implemented": "IDENTICAL_ENDPOINT_POLYNOMIAL_QUOTIENT_RECURRENCE_AND_EXACT_CHARACTERISTIC_NORM",
            "strongest_known_compact_projection": "UNIVARIATE_CHAIN_SPECIFIC_EXACT_SUBRESULTANT_PRS_RESULTANT",
            "subresultant_prs_projection_implemented": False,
            "projection_resource_comparison_authorized": False,
            "same_exact_final_relation": True,
            "public_arithmetic_circuit_retained": True,
            "executed_only_after_carrier_release": True,
            "root_enumeration_used": False,
            "computational_advantage": False,
            "distinct_phase_resource": False,
        },
        "separate_reference": {
            "imports_m236_production": False,
            "uses_independent_sparse_dictionary_quotient_arithmetic": True,
            "implements_independent_custody_state_machine": True,
            "case_control_reuse_parity": True,
        },
        "source_dependencies": {
            "production_sha256": file_sha256(Path(__file__).resolve()),
            "separate_reference_sha256": file_sha256(
                Path(__file__).resolve().with_name(
                    "cubic_airy_chain_quotient_algebra_separate_reference.py"
                )
            ),
        },
        "claim_limits": {
            "all_stationary_branches_real": False,
            "normalized_or_convergent_airy_integral_established": False,
            "fresnel_maslov_or_full_amplitude_preserved": False,
            "arbitrary_cubic_relation_closure": False,
            "bounded_quotient_dimension": False,
            "unbounded_degree_theorem_beyond_declared_lengths": False,
            "machine_enforced_catvm_custody": False,
            "distinct_phase_resource_established": False,
            "computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference_established": False,
            "unbounded_catalytic_computation_established": False,
        },
        "terminal": False,
    }
    print(json.dumps(output, sort_keys=True, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: cubic_airy_chain_quotient_algebra.py REFERENCE_JSON")
    main(Path(sys.argv[1]))
