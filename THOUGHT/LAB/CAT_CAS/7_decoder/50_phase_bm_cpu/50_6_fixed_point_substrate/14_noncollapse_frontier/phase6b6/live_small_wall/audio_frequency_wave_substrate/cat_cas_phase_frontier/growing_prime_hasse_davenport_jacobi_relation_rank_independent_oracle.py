#!/usr/bin/env python3
"""No-import oracle for the M182 nonlinear Gauss relation diagnostic.

This file imports neither M182 nor the M180/M181 implementation.  It rebuilds
the finite fields and phase sums, uses dense exact rational elimination rather
than the production sparse basis, reverses the sign convention of relation
rows, and reconstructs all boundary-product exponent families directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


CLAIM = (
    "BOUNDED_EXACT_FOURTEEN_DECLARED_PRIME_ALL_DIVISOR_HASSE_DAVENPORT_"
    "GAUSS_PHASE_RELATIONS_HAVE_FORMAL_FREE_MONOMIAL_GENERATOR_RANK_"
    "PHI_OF_QMINUS1_AND_GAUSS_NORM_AUGMENTATION_HAS_RANK_HALF_PHI_WHILE_"
    "THE_ACTUAL_MELLIN_DETERMINANT_BOUNDARY_CHANNEL_PRODUCTS_SPAN_THE_"
    "ENTIRE_REMAINING_QUOTIENT_AND_STREAMED_JACOBI_DEFINITIONS_ADD_NO_"
    "GAUSS_CONSTRAINT_WITHOUT_RETAINING_ANSWER_BEARING_VALUES_OR_"
    "REMATERIALIZING_Q_TERM_SUMS_SO_THE_DECLARED_SAFE_PRIME_CASES_RETAIN_"
    "GROWING_RANK_AND_THE_IDENTICAL_CLASSICAL_RELATION_ALGEBRA_REMAINS"
)


CASES = (
    (5, 41, 1, 2),
    (7, 43, 3, 5),
    (11, 331, 5, 8),
    (13, 157, 7, 11),
    (17, 1361, 9, 14),
    (19, 2053, 11, 17),
    (23, 1013, 13, 20),
    (29, 2437, 15, 23),
    (31, 1861, 17, 26),
    (37, 6661, 19, 29),
    (41, 13121, 21, 32),
    (43, 3613, 23, 35),
    (47, 12973, 25, 38),
    (53, 8269, 27, 41),
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def factors(value: int) -> list[int]:
    answer: list[int] = []
    probe = 2
    while probe * probe <= value:
        if value % probe == 0:
            answer.append(probe)
            while value % probe == 0:
                value //= probe
        probe += 1
    if value != 1:
        answer.append(value)
    return answer


def primitive_root(prime: int) -> int:
    prime_divisors = factors(prime - 1)
    for candidate in range(2, prime):
        if all(
            pow(candidate, (prime - 1) // divisor, prime) != 1
            for divisor in prime_divisors
        ):
            return candidate
    fail("primitive-root search failed")


def phi(value: int) -> int:
    answer = value
    for divisor in factors(value):
        answer = answer // divisor * (divisor - 1)
    return answer


def divisors(width: int) -> list[int]:
    return [candidate for candidate in range(2, width + 1) if width % candidate == 0]


def exact_dense_rank(integer_rows: list[list[int]]) -> int:
    """Independent dense forward elimination over the rational numbers."""
    if not integer_rows:
        return 0
    matrix = [[Fraction(value) for value in row] for row in integer_rows]
    row_count = len(matrix)
    column_count = len(matrix[0])
    rank = 0
    for column in range(column_count):
        pivot = next(
            (row for row in range(rank, row_count) if matrix[row][column]),
            None,
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        pivot_value = matrix[rank][column]
        for row in range(rank + 1, row_count):
            if not matrix[row][column]:
                continue
            multiplier = matrix[row][column] / pivot_value
            for target in range(column, column_count):
                matrix[row][target] -= multiplier * matrix[rank][target]
        rank += 1
        if rank == row_count:
            break
    return rank


def multiplication_rows(width: int, skip_two: bool = False) -> list[list[int]]:
    """Build sign-reversed Hasse--Davenport exponent equations."""
    answer: list[list[int]] = []
    for multiplier in divisors(width):
        if skip_two and multiplier == 2:
            continue
        stride = width // multiplier
        for character in range(width - 1, -1, -1):
            row = [0] * width
            row[multiplier * character % width] -= 1
            for offset in range(multiplier):
                row[(character + offset * stride) % width] += 1
                row[offset * stride] -= 1
            if any(row):
                answer.append(row)
    return answer


def reflection_rows(width: int) -> list[list[int]]:
    answer: list[list[int]] = []
    for character in range(width - 1, 0, -1):
        row = [0] * width
        row[character] = 1
        row[-character % width] += 1
        answer.append(row)
    return answer


def requested_product_rows(width: int, determinant_index: int, scale_index: int) -> list[list[int]]:
    quadratic = width // 2
    answer: list[list[int]] = []
    for channel in range(width - 1, -1, -1):
        row = [0] * width
        row[(determinant_index - channel) % width] += 1
        row[channel] += 2
        row[(channel + quadratic) % width] += 1
        row[quadratic] += 3
        row[(determinant_index + scale_index - channel) % width] += 1
        answer.append(row)
    return answer


def build_field(q: int, p: int) -> dict[str, int]:
    generator = primitive_root(p)
    return {
        "q": q,
        "p": p,
        "q_generator": primitive_root(q),
        "additive_root": pow(generator, (p - 1) // q, p),
        "multiplicative_root": pow(generator, (p - 1) // (q - 1), p),
    }


def character(field: dict[str, int], exponent: int, value: int) -> int:
    q = field["q"]
    reduced = value % q
    if reduced == 0:
        return 0
    point = 1
    phase = 1
    step = pow(field["multiplicative_root"], exponent % (q - 1), field["p"])
    for _ in range(q - 1):
        if point == reduced:
            return phase
        point = point * field["q_generator"] % q
        phase = phase * step % field["p"]
    fail("independent character orbit failed")


def gauss(field: dict[str, int], exponent: int) -> int:
    q, p = field["q"], field["p"]
    return sum(
        character(field, exponent, value) * pow(field["additive_root"], value, p)
        for value in range(q)
    ) % p


def jacobi(field: dict[str, int], left: int, right: int) -> int:
    return sum(
        character(field, left, value) * character(field, right, 1 - value)
        for value in range(field["q"])
    ) % field["p"]


def numerical_checks(q: int, p: int) -> dict[str, Any]:
    field = build_field(q, p)
    width = q - 1
    values = [gauss(field, exponent) for exponent in range(width)]
    multiplication_count = 0
    multiplication_hash = hashlib.sha256()
    for multiplier in divisors(width):
        stride = width // multiplier
        denominator = 1
        for offset in range(multiplier):
            denominator = denominator * values[offset * stride] % p
        for index in range(width):
            numerator = 1
            for offset in range(multiplier):
                numerator = numerator * values[(index + offset * stride) % width] % p
            scalar = character(field, multiplier * index, multiplier)
            recovered = -scalar * numerator * pow(denominator, -1, p) % p
            if recovered != values[multiplier * index % width]:
                fail("independent Hasse--Davenport check failed")
            multiplication_hash.update(
                f"{multiplier}:{index}:{recovered};".encode("ascii")
            )
            multiplication_count += 1

    norm_count = 0
    for index in range(width - 1, 0, -1):
        right = character(field, index, -1) * q % p
        if values[index] * values[-index % width] % p != right:
            fail("independent norm check failed")
        norm_count += 1

    jacobi_count = 0
    jacobi_terms = 0
    jacobi_hash = hashlib.sha256()
    # Canonical ascending serialization makes the independently obtained
    # values directly commitment-comparable with production.
    for left in range(1, width):
        for right in range(1, width):
            combined = (left + right) % width
            if combined == 0:
                continue
            value = jacobi(field, left, right)
            if values[left] * values[right] % p != value * values[combined] % p:
                fail("independent Jacobi check failed")
            jacobi_hash.update(f"{left}:{right}:{value};".encode("ascii"))
            jacobi_count += 1
            jacobi_terms += q

    if values[1] == 1:
        fail("independent false overmerge control passed")
    return {
        "q": q,
        "auxiliary_prime": p,
        "hasse_davenport_identity_checks": multiplication_count,
        "hasse_davenport_value_commitment": multiplication_hash.hexdigest(),
        "gauss_norm_identity_checks": norm_count,
        "applicable_jacobi_identity_checks": jacobi_count,
        "streamed_jacobi_character_terms": jacobi_terms,
        "jacobi_value_commitment": jacobi_hash.hexdigest(),
        "false_overmerge_rejected": True,
    }


def rank_checks(q: int, determinant_index: int, scale_index: int) -> dict[str, Any]:
    width = q - 1
    hd = multiplication_rows(width)
    reflected = reflection_rows(width)
    products = requested_product_rows(width, determinant_index, scale_index)
    hd_rank = exact_dense_rank(hd)
    reflected_rank = exact_dense_rank(hd + reflected)
    hd_free = width - hd_rank
    reflected_free = width - reflected_rank
    product_hd_span = exact_dense_rank(hd + products) - hd_rank
    product_reflected_span = exact_dense_rank(hd + reflected + products) - reflected_rank
    expected_phi = phi(width)
    missing_two_free = width - exact_dense_rank(multiplication_rows(width, skip_two=True))
    if (
        hd_free != expected_phi
        or reflected_free != expected_phi // 2
        or product_hd_span != hd_free
        or product_reflected_span != reflected_free
        or missing_two_free <= hd_free
    ):
        fail(f"independent exact relation-rank check failed at q={q}")
    jacobi_count = (width - 1) * (width - 2)
    return {
        "q": q,
        "hasse_exact_rational_rank": hd_rank,
        "hasse_free_monomial_generator_rank": hd_free,
        "norm_augmented_exact_rational_rank": reflected_rank,
        "norm_augmented_free_monomial_generator_rank": reflected_free,
        "boundary_product_span_mod_hasse": product_hd_span,
        "boundary_product_span_mod_hasse_and_norm": product_reflected_span,
        "without_quadratic_hasse_relation_free_rank": missing_two_free,
        "euler_phi_q_minus_1": expected_phi,
        "formal_jacobi_generator_count": jacobi_count,
        "jacobi_augmented_free_rank": (
            width + jacobi_count - (hd_rank + jacobi_count)
        ),
    }


def build_result() -> dict[str, Any]:
    ranks = [rank_checks(q, determinant, scale) for q, _, determinant, scale in CASES]
    numerical = [numerical_checks(q, p) for q, p, _, _ in CASES]
    return {
        "claim": CLAIM,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "NO_RESTORATION_CLAIM",
        "oracle_independence": {
            "imports_production": False,
            "imports_predecessor": False,
            "finite_fields_reconstructed": True,
            "dense_fraction_rank_instead_of_sparse_basis": True,
            "sign_reversed_relation_rows": True,
            "character_values_scanned_without_log_table": True,
        },
        "rank_diagnostics": ranks,
        "residue_identity_diagnostics": numerical,
        "controls": {
            "missing_quadratic_relation_rejected_every_case": True,
            "false_overmerge_rejected_every_case": True,
            "boundary_products_span_every_declared_quotient": True,
        },
        "claim_ceiling": (
            "FOURTEEN_DECLARED_M181_FIELD_AND_PROGRAM_FAMILIES_FORMAL_"
            "MONOMIAL_RELATION_RANK_WITH_DIRECT_RESIDUE_IDENTITY_REEXECUTION"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    text = json.dumps(build_result(), indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
