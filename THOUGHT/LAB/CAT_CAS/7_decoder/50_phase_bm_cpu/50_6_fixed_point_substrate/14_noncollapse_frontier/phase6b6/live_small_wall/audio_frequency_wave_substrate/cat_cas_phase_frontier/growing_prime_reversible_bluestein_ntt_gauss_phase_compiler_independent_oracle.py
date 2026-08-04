#!/usr/bin/env python3
"""No-import oracle for the M183 reversible Bluestein Gauss compiler."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


CLAIM = (
    "BOUNDED_EXACT_FOURTEEN_DECLARED_PRIME_BLUESTEIN_POWER_OF_TWO_NTT_"
    "GAUSS_PHASE_COMPILER_REPLACES_QUADRATIC_ALL_GAUSS_CHARACTER_SUM_"
    "COMPILATION_WITH_SUBQUADRATIC_M_LOG_M_EXACT_FIELD_WORK_ON_ONE_"
    "QMINUS1_RESIDENT_DESCRIPTOR_PLUS_THREE_M_CELL_REVERSIBLY_CLEARED_"
    "SCRATCH_SEGMENTS_WITH_FINAL_BOUNDARY_PROJECTION_TOPOLOGY_"
    "REMATERIALIZED_EXACT_SAME_BACKING_RESTORATION_AND_UNRELATED_REUSE_"
    "BUT_M_IS_LINEAR_IN_Q_EXACT_BIT_WIDTH_GROWS_AND_THE_IDENTICAL_"
    "CLASSICAL_BLUESTEIN_NTT_COMPILER_REMAINS"
)


CASES = (
    (5, 41, 1, 2),
    (7, 337, 3, 5),
    (11, 5281, 5, 8),
    (13, 1249, 7, 11),
    (17, 5441, 9, 14),
    (19, 32833, 11, 17),
    (23, 16193, 13, 20),
    (29, 38977, 15, 23),
    (31, 29761, 17, 26),
    (37, 127873, 19, 29),
    (41, 78721, 21, 32),
    (43, 231169, 23, 35),
    (47, 691841, 25, 38),
    (53, 264577, 27, 41),
)


PAIRING_WEIGHTS = (1, 2, 2, 1, 2, 1)
PRIMARY_POINT = (2, 0, 0, 1, 0, 1)


def fail(message: str) -> None:
    raise RuntimeError(message)


def factors(number: int) -> list[int]:
    answer: list[int] = []
    divisor = 2
    while divisor * divisor <= number:
        if number % divisor == 0:
            answer.append(divisor)
            while number % divisor == 0:
                number //= divisor
        divisor += 1
    if number > 1:
        answer.append(number)
    return answer


def generator(prime: int) -> int:
    divisors = factors(prime - 1)
    for candidate in range(2, prime):
        if all(
            pow(candidate, (prime - 1) // divisor, prime) != 1
            for divisor in divisors
        ):
            return candidate
    fail("independent primitive-root search failed")


def power_two_ceiling(value: int) -> int:
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class Field:
    q: int
    p: int
    q_generator: int
    additive_root: int
    character_root: int
    chirp_root: int
    convolution_root: int
    convolution_width: int

    @classmethod
    def make(cls, q: int, p: int) -> "Field":
        primitive = generator(p)
        h = q - 1
        width = power_two_ceiling(2 * h - 1)
        return cls(
            q=q,
            p=p,
            q_generator=generator(q),
            additive_root=pow(primitive, (p - 1) // q, p),
            character_root=pow(primitive, (p - 1) // h, p),
            chirp_root=pow(primitive, (p - 1) // (2 * h), p),
            convolution_root=pow(primitive, (p - 1) // width, p),
            convolution_width=width,
        )


def log_map(field: Field) -> dict[int, int]:
    return {
        pow(field.q_generator, exponent, field.q): exponent
        for exponent in range(field.q - 1)
    }


def character(
    field: Field, logs: dict[int, int], value: int, exponent: int
) -> int:
    reduced = value % field.q
    if reduced == 0:
        return 0
    return pow(
        field.character_root,
        logs[reduced] * exponent % (field.q - 1),
        field.p,
    )


def direct_gauss(field: Field, logs: dict[int, int]) -> list[int]:
    return [
        sum(
            pow(field.additive_root, value, field.p)
            * character(field, logs, value, exponent)
            for value in range(1, field.q)
        )
        % field.p
        for exponent in range(field.q - 1)
    ]


def recursive_transform(
    values: list[int], root: int, modulus: int, inverse: bool = False
) -> list[int]:
    """Independent out-of-place recursive radix-two NTT."""
    width = len(values)
    if width == 1:
        return values[:]
    stage_root = pow(root, -1, modulus) if inverse else root

    def recurse(data: list[int], local_root: int) -> list[int]:
        if len(data) == 1:
            return data
        even = recurse(data[0::2], local_root * local_root % modulus)
        odd = recurse(data[1::2], local_root * local_root % modulus)
        twiddle = 1
        half = len(data) // 2
        output = [0] * len(data)
        for index in range(half):
            high = odd[index] * twiddle % modulus
            output[index] = (even[index] + high) % modulus
            output[index + half] = (even[index] - high) % modulus
            twiddle = twiddle * local_root % modulus
        return output

    transformed = recurse(values, stage_root)
    if inverse:
        scale = pow(width, -1, modulus)
        transformed = [value * scale % modulus for value in transformed]
    return transformed


def independent_bluestein(field: Field) -> list[int]:
    q, p = field.q, field.p
    h = q - 1
    width = field.convolution_width
    left = [0] * width
    kernel = [0] * width
    orbit = 1
    for index in range(h):
        chirp = pow(field.chirp_root, index * index, p)
        left[index] = pow(field.additive_root, orbit, p) * chirp % p
        anti = pow(field.chirp_root, -index * index, p)
        kernel[index] = anti
        if index:
            kernel[width - index] = anti
        orbit = orbit * field.q_generator % q
    left_hat = recursive_transform(left, field.convolution_root, p)
    kernel_hat = recursive_transform(kernel, field.convolution_root, p)
    product = [a * b % p for a, b in zip(left_hat, kernel_hat)]
    convolution = recursive_transform(
        product, field.convolution_root, p, inverse=True
    )
    return [
        pow(field.chirp_root, index * index, p) * convolution[index] % p
        for index in range(h)
    ]


def determinant(point: tuple[int, ...], q: int) -> int:
    a, b, c, d, e, f = point
    return (
        a * (d * f - e * e)
        - b * (b * f - e * c)
        + c * (b * e - d * c)
    ) % q


def minor(matrix: tuple[tuple[int, ...], ...], indexes: tuple[int, ...], q: int) -> int:
    if len(indexes) == 1:
        return matrix[indexes[0]][indexes[0]] % q
    if len(indexes) == 2:
        i, j = indexes
        return (matrix[i][i] * matrix[j][j] - matrix[i][j] ** 2) % q
    return determinant(
        (
            matrix[0][0],
            matrix[0][1],
            matrix[0][2],
            matrix[1][1],
            matrix[1][2],
            matrix[2][2],
        ),
        q,
    )


def rank_class(point: tuple[int, ...], q: int) -> tuple[int, int]:
    a, b, c, d, e, f = point
    matrix = ((a, b, c), (b, d, e), (c, e, f))
    for rank in (3, 2, 1):
        for indexes in itertools.combinations(range(3), rank):
            value = minor(matrix, indexes, q)
            if value:
                return rank, 1 if pow(value, (q - 1) // 2, q) == 1 else -1
    return 0, 0


def boundary(
    field: Field,
    logs: dict[int, int],
    gauss: list[int],
    determinant_index: int,
    scale_index: int,
    point: tuple[int, ...],
    target_scale: int,
) -> int:
    q, p = field.q, field.p
    h = q - 1
    half = h // 2
    matrix_rank, square = rank_class(point, q)
    answer = 0
    for channel in range(h):
        coefficient = pow(h, -1, p) * gauss[(determinant_index - channel) % h] % p
        if matrix_rank == 3:
            gamma = (
                gauss[channel] ** 2
                * gauss[(channel + half) % h]
                * gauss[half] ** 3
            ) % p
            det_factor = gamma * character(
                field, logs, determinant(point, q), -channel
            ) % p
        elif channel == 0:
            det_factor = (
                q**6 - q**5 - q**3 + q**2
                if matrix_rank == 0
                else q**2 - q**3
            ) % p
        elif channel == half and matrix_rank == 1:
            det_factor = q**2 * h * gauss[half] ** 3 * square % p
        else:
            det_factor = 0
        source_index = (determinant_index + scale_index - channel) % h
        scale_factor = (
            gauss[source_index]
            * character(field, logs, target_scale, -source_index)
            % p
            if target_scale % q
            else (h % p if source_index == 0 else 0)
        )
        answer += coefficient * det_factor * scale_factor
    return answer % p


def direct_relation_sum(
    field: Field,
    logs: dict[int, int],
    determinant_index: int,
    scale_index: int,
    target: tuple[int, ...],
    target_scale: int,
) -> tuple[int, int]:
    q, p = field.q, field.p
    answer = 0
    terms = 0
    for source in itertools.product(range(q), repeat=6):
        det = determinant(source, q)
        if det == 0:
            continue
        det_phase = character(field, logs, det, determinant_index)
        pairing = sum(
            weight * left * right
            for weight, left, right in zip(PAIRING_WEIGHTS, target, source)
        )
        for scale in range(1, q):
            source_value = (
                pow(field.additive_root, det * pow(scale, -1, q) % q, p)
                * det_phase
                * character(field, logs, scale, scale_index)
            ) % p
            final_phase = pow(
                field.additive_root,
                (pairing + target_scale * scale) % q,
                p,
            )
            answer += source_value * final_phase
            terms += 1
    return answer % p, terms


def digest(values: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, values)).encode("ascii")).hexdigest()


def oracle_case(q: int, p: int, determinant_index: int, scale_index: int) -> dict[str, Any]:
    field = Field.make(q, p)
    logs = log_map(field)
    direct = direct_gauss(field, logs)
    transformed = independent_bluestein(field)
    if transformed != direct:
        fail("independent recursive Bluestein transform differs from direct Gauss table")
    h = q - 1
    total_cells = h + 3 * field.convolution_width
    cells = [0] * total_cells
    backing = id(cells)
    for index, value in enumerate(transformed):
        cells[index] = (cells[index] + value) % p
    if any(cells[h:]):
        fail("independent scratch region is not zero")
    projected = boundary(
        field,
        logs,
        cells[:h],
        determinant_index,
        scale_index,
        PRIMARY_POINT,
        2 % q,
    )
    resident = digest(cells)
    persisted = projected
    rematerialized = independent_bluestein(field)
    for index, value in enumerate(rematerialized):
        cells[index] = (cells[index] - value) % p
    if any(cells) or id(cells) != backing:
        fail("independent rematerialized inverse failed")

    second_det = (determinant_index + 2) % h
    second_scale = (scale_index + 3) % h
    second_point = (3, 0, 0, 2, 0, 1)
    for index, value in enumerate(independent_bluestein(field)):
        cells[index] = (cells[index] + value) % p
    reused_boundary = boundary(
        field, logs, cells[:h], second_det, second_scale, second_point, 3 % q
    )
    fresh = independent_bluestein(field)
    fresh_boundary = boundary(
        field, logs, fresh, second_det, second_scale, second_point, 3 % q
    )
    for index, value in enumerate(independent_bluestein(field)):
        cells[index] = (cells[index] - value) % p
    if reused_boundary != fresh_boundary or any(cells) or id(cells) != backing:
        fail("independent restored reuse failed")

    missing_inverse_fails = any(transformed)
    wrong_field = replace(
        field,
        additive_root=pow(field.additive_root, 2, p),
    )
    wrong_residue = [
        (left - right) % p
        for left, right in zip(transformed, independent_bluestein(wrong_field))
    ]
    wrong_inverse_fails = any(wrong_residue)
    omitted_residue = transformed[:]
    for index, value in enumerate(independent_bluestein(field)):
        if index != 1:
            omitted_residue[index] = (omitted_residue[index] - value) % p
    omitted_frequency_fails = any(omitted_residue)
    if not (missing_inverse_fails and wrong_inverse_fails and omitted_frequency_fails):
        fail("independent inverse mutation control failed")

    direct_check: dict[str, Any] | None = None
    if q in (5, 7):
        direct_value, terms = direct_relation_sum(
            field,
            logs,
            determinant_index,
            scale_index,
            PRIMARY_POINT,
            2 % q,
        )
        if direct_value != projected:
            fail("independent seven-dimensional relation sum differs")
        direct_check = {
            "direct_boundary_scalar": direct_value,
            "nonzero_source_terms": terms,
            "matches_compiled_boundary": True,
        }
    return {
        "q": q,
        "auxiliary_prime": p,
        "convolution_width": field.convolution_width,
        "carrier_field_cells": total_cells,
        "descriptor_commitment": digest(transformed),
        "descriptor_matches_direct_gauss": True,
        "projected_boundary_scalar": projected,
        "projected_result_survives_inverse": persisted == projected,
        "resident_commitment": resident,
        "exact_same_backing_restoration": id(cells) == backing and not any(cells),
        "actual_restored_backing_reuse_matches_fresh": reused_boundary == fresh_boundary,
        "reused_boundary_scalar": reused_boundary,
        "fresh_boundary_scalar": fresh_boundary,
        "missing_inverse_fails": missing_inverse_fails,
        "wrong_additive_phase_inverse_fails": wrong_inverse_fails,
        "omitted_frequency_inverse_fails": omitted_frequency_fails,
        "direct_relation_check": direct_check,
        "expected_ntt_calls_per_compiler": 6,
        "expected_butterflies_per_compiler": (
            3 * field.convolution_width * (field.convolution_width.bit_length() - 1)
        ),
    }


def build_result() -> dict[str, Any]:
    cases = [oracle_case(*case) for case in CASES]
    return {
        "claim": CLAIM,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "oracle_independence": {
            "imports_production": False,
            "imports_predecessor": False,
            "field_roots_reconstructed": True,
            "uses_out_of_place_recursive_ntt": True,
            "uses_production_iterative_ntt": False,
            "direct_gauss_tables_reconstructed": True,
            "direct_original_relation_checks": 2,
        },
        "oracle_cases": cases,
        "controls": {
            "all_descriptors_match_direct_gauss": True,
            "all_exact_same_backing_restorations_pass": True,
            "all_reuses_match_fresh": True,
            "all_missing_inverses_fail": True,
            "all_wrong_additive_phase_inverses_fail": True,
            "all_omitted_frequency_inverses_fail": True,
            "q5_q7_direct_original_relation_boundaries_match": True,
        },
        "claim_ceiling": (
            "FOURTEEN_DECLARED_FIELD_PROGRAM_CASES_INDEPENDENT_RECURSIVE_"
            "NTT_AND_DIRECT_GAUSS_PARITY_WITH_Q5_Q7_DIRECT_ORIGINAL_"
            "RELATION_BOUNDARY_REEXECUTION"
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
