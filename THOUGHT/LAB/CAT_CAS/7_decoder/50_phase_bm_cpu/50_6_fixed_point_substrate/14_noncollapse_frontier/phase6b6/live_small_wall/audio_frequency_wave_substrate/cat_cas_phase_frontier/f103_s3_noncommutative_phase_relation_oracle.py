#!/usr/bin/env python3
"""Independent full-relation oracle for the S3 phase relation algebra."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FIELD = 103
ZETA6 = 57
DEPTHS = (1, 4, 16, 64, 256, 1024)
FAMILIES = ("PRIMARY", "ALTERNATE")
ELEMENTS = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)
INDEX = {element: index for index, element in enumerate(ELEMENTS)}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def multiply(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(left[right[position]] for position in range(3))


def inverse(element: tuple[int, ...]) -> tuple[int, ...]:
    result = [0, 0, 0]
    for source, target in enumerate(element):
        result[target] = source
    return tuple(result)


def sign(element: tuple[int, ...]) -> int:
    inversions = sum(element[left] > element[right] for left in range(3) for right in range(left + 1, 3))
    return -1 if inversions % 2 else 1


def phase(exponent: int) -> int:
    return pow(ZETA6, exponent % 6, FIELD)


def public_function(index: int, family: str, kind: int) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    return [phase((kind + 1) * index + (2 * kind + code) * position + position * position + code) for position in range(6)]


def parameters(index: int, family: str) -> tuple[int, int, int, int]:
    code = 1 if family == "PRIMARY" else 2
    return tuple(phase((offset + 1) * index + offset * code + 1) for offset in (1, 2, 4, 5))


def seed_function(family: str, register: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    offset = 1 if register == "A" else 4
    return [phase(offset + code * (position + 1) + position * position) for position in range(6)]


def relation(function: list[int]) -> list[int]:
    matrix = []
    for left in ELEMENTS:
        for right in ELEMENTS:
            difference = multiply(inverse(left), right)
            matrix.append(function[INDEX[difference]])
    return matrix


def extract_function(matrix: list[int]) -> list[int]:
    function = matrix[:6]
    if relation(function) != matrix:
        raise RuntimeError("oracle relation escaped S3 translation invariance")
    return function


def add(left: list[int], right: list[int]) -> list[int]:
    return [(x + y) % FIELD for x, y in zip(left, right, strict=True)]


def subtract(left: list[int], right: list[int]) -> list[int]:
    return [(x - y) % FIELD for x, y in zip(left, right, strict=True)]


def scale(value: list[int], scalar: int) -> list[int]:
    return [(scalar * item) % FIELD for item in value]


def hadamard(left: list[int], right: list[int]) -> list[int]:
    return [(x * y) % FIELD for x, y in zip(left, right, strict=True)]


def matrix_product(left: list[int], right: list[int]) -> list[int]:
    result = [0] * 36
    for row in range(6):
        for column in range(6):
            result[6 * row + column] = sum(left[6 * row + middle] * right[6 * middle + column] for middle in range(6)) % FIELD
    return result


def descriptors(index: int, family: str) -> tuple[tuple[str, str, int, list[int]], ...]:
    alpha, beta, gamma, delta = parameters(index, family)
    stages = (
        ("B", "RIGHT_COMPOSE", alpha, relation(public_function(index, family, 1))),
        ("A", "INTERSECT", beta, relation(public_function(index, family, 2))),
        ("B", "LEFT_COMPOSE", gamma, relation(public_function(index, family, 3))),
        ("A", "INTERSECT", delta, relation(public_function(index, family, 4))),
    )
    return stages if family == "PRIMARY" else tuple(reversed(stages))


def apply(a: list[int], b: list[int], stage: tuple[str, str, int, list[int]], subtracting: bool = False) -> tuple[list[int], list[int]]:
    target, operation, scalar, public = stage
    source = a if target == "B" else b
    if operation == "RIGHT_COMPOSE":
        value = matrix_product(source, public)
    elif operation == "LEFT_COMPOSE":
        value = matrix_product(public, source)
    elif operation == "INTERSECT":
        value = hadamard(source, public)
    else:
        raise RuntimeError("unknown oracle stage")
    value = scale(value, scalar)
    combine = subtract if subtracting else add
    return (combine(a, value), b) if target == "A" else (a, combine(b, value))


def boundary(b: list[int], family: str) -> int:
    code = 1 if family == "PRIMARY" else 2
    weights = [phase(code + 2 * position + position * position) for position in range(6)]
    return sum(weight * value for weight, value in zip(weights, extract_function(b), strict=True)) % FIELD


def execute(depth: int, family: str) -> dict[str, Any]:
    a = relation(seed_function(family, "A"))
    b = relation(seed_function(family, "B"))
    sealed = (a.copy(), b.copy())
    for index in range(depth):
        for stage in descriptors(index, family):
            a, b = apply(a, b, stage)
    compact_a, compact_b = extract_function(a), extract_function(b)
    translation_invariance_preserved = relation(compact_a) == a and relation(compact_b) == b
    result_boundary = boundary(b, family)
    commitment = digest_json([compact_a, compact_b])
    for index in reversed(range(depth)):
        for stage in reversed(descriptors(index, family)):
            a, b = apply(a, b, stage, subtracting=True)
    return {
        "depth": depth,
        "family": family,
        "boundary": result_boundary,
        "forward_commitment": commitment,
        "full72_cells_restore_exactly": a == sealed[0] and b == sealed[1],
        "translation_invariance_preserved": translation_invariance_preserved,
    }


def representation(element: tuple[int, ...]) -> list[int]:
    output = []
    for basis in ((1, 0, -1), (0, 1, -1)):
        image = [0, 0, 0]
        for source, value in enumerate(basis):
            image[element[source]] = value
        output.append((image[0] % FIELD, image[1] % FIELD))
    return [output[0][0], output[1][0], output[0][1], output[1][1]]


def multiply_2x2(left: list[int], right: list[int]) -> list[int]:
    return [
        (left[0] * right[0] + left[1] * right[2]) % FIELD,
        (left[0] * right[1] + left[1] * right[3]) % FIELD,
        (left[2] * right[0] + left[3] * right[2]) % FIELD,
        (left[2] * right[1] + left[3] * right[3]) % FIELD,
    ]


def transform(function: list[int]) -> tuple[int, int, list[int]]:
    trivial = sum(function) % FIELD
    signed = sum(value * sign(element) for value, element in zip(function, ELEMENTS, strict=True)) % FIELD
    standard = [0, 0, 0, 0]
    for value, element in zip(function, ELEMENTS, strict=True):
        standard = [(entry + value * coefficient) % FIELD for entry, coefficient in zip(standard, representation(element), strict=True)]
    return trivial, signed, standard


def inverse_transform(value: tuple[int, int, list[int]]) -> list[int]:
    trivial, signed, standard = value
    inverse_six = pow(6, -1, FIELD)
    function = []
    for element in ELEMENTS:
        product = multiply_2x2(representation(inverse(element)), standard)
        trace = (product[0] + product[3]) % FIELD
        function.append(inverse_six * (trivial + sign(element) * signed + 2 * trace) % FIELD)
    return function


def compact_convolution(left: list[int], right: list[int]) -> list[int]:
    result = [0] * 6
    for target, element in enumerate(ELEMENTS):
        for source, source_element in enumerate(ELEMENTS):
            residual = multiply(inverse(source_element), element)
            result[target] = (result[target] + left[source] * right[INDEX[residual]]) % FIELD
    return result


def irrep_convolution(left: list[int], right: list[int]) -> list[int]:
    lt, ls, lm = transform(left)
    rt, rs, rm = transform(right)
    return inverse_transform((lt * rt % FIELD, ls * rs % FIELD, multiply_2x2(lm, rm)))


def delta(position: int) -> list[int]:
    value = [0] * 6
    value[position] = 1
    return value


def algebra_oracle() -> dict[str, Any]:
    basis = [delta(position) for position in range(6)]
    composition_checks = 0
    hadamard_checks = 0
    all_translation_invariant = True
    for left in range(6):
        for right in range(6):
            product = matrix_product(relation(basis[left]), relation(basis[right]))
            expected = relation(basis[INDEX[multiply(ELEMENTS[left], ELEMENTS[right])]])
            if product != expected:
                raise RuntimeError("independent full relation composition mismatch")
            composition_checks += 1
            if hadamard(relation(basis[left]), relation(basis[right])) != relation(basis[left] if left == right else [0] * 6):
                raise RuntimeError("independent full relation intersection mismatch")
            hadamard_checks += 1
            all_translation_invariant = all_translation_invariant and relation(extract_function(product)) == product
    irrep_checks = sum(irrep_convolution(basis[left], basis[right]) == compact_convolution(basis[left], basis[right]) for left in range(6) for right in range(6))
    pair = next((left, right) for left in range(6) for right in range(6) if multiply(ELEMENTS[left], ELEMENTS[right]) != multiply(ELEMENTS[right], ELEMENTS[left]))
    transform_rows = []
    for basis_value in basis:
        trivial, signed, standard = transform(basis_value)
        transform_rows.append([trivial, signed, *standard])
    rank = matrix_rank(transform_rows)
    return {
        "composition_checks": composition_checks,
        "hadamard_checks": hadamard_checks,
        "all_outputs_translation_invariant": all_translation_invariant,
        "noncommuting_basis_pair": list(pair),
        "noncommuting_products_differ": compact_convolution(basis[pair[0]], basis[pair[1]]) != compact_convolution(basis[pair[1]], basis[pair[0]]),
        "irrep_convolution_checks": irrep_checks,
        "irrep_transform_rank": rank,
        "irrep_coordinate_count": 6,
        "irrep_transform_compresses_below_group_order": rank < 6,
        "all_basis_fourier_roundtrips": all(inverse_transform(transform(item)) == item for item in basis),
    }


def matrix_rank(rows: list[list[int]]) -> int:
    matrix = [row.copy() for row in rows]
    rank = 0
    for column in range(len(matrix[0])):
        pivot = next((row for row in range(rank, len(matrix)) if matrix[row][column] % FIELD), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        inverse_pivot = pow(matrix[rank][column], -1, FIELD)
        matrix[rank] = [value * inverse_pivot % FIELD for value in matrix[rank]]
        for row in range(len(matrix)):
            if row != rank and matrix[row][column]:
                factor = matrix[row][column]
                matrix[row] = [(left - factor * right) % FIELD for left, right in zip(matrix[row], matrix[rank], strict=True)]
        rank += 1
    return rank


def build_result(production: dict[str, Any]) -> dict[str, Any]:
    cases = [execute(depth, family) for family in FAMILIES for depth in DEPTHS]
    production_cases = {(item["family"], item["depth"]): item for item in production["cases"]}
    comparisons = []
    for case in cases:
        expected = production_cases[(case["family"], case["depth"])]
        comparisons.append(case["boundary"] == expected["boundary"] and case["forward_commitment"] == expected["forward_commitment"])
    algebra = algebra_oracle()
    if not all(comparisons) or any(not case["full72_cells_restore_exactly"] for case in cases):
        raise RuntimeError("independent S3 production comparison failed")
    if algebra["composition_checks"] != 36 or algebra["hadamard_checks"] != 36 or algebra["irrep_convolution_checks"] != 36 or algebra["irrep_transform_rank"] != 6:
        raise RuntimeError("independent S3 algebra certificate failed")
    return {
        "schema": "CAT_CAS_F103_S3_NONCOMMUTATIVE_PHASE_RELATION_ORACLE_RESULTS_V1",
        "claim": production["claim"],
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "imports_production": False,
        "imports_numpy": False,
        "oracle_state_law": "TWO_FULL6_BY6_F103_S3_TRANSLATION_RELATION_MATRICES",
        "algebra_oracle": algebra,
        "cases": cases,
        "production_comparison": {
            "cases": len(cases),
            "boundary_and_commitment_comparisons": 2 * len(cases),
            "all_match": all(comparisons),
        },
        "resource_scope": {
            "accepted_two_register_carrier_field_cells": 12,
            "oracle_full_two_register_relation_field_cells": 72,
            "matched_group_coordinate_carrier_field_cells": 12,
            "matched_irrep_information_coordinates": 6,
            "dense6_by6_relations_are_oracle_only": True,
            "answer_bearing_lookup_table_cells": 0,
            "python_allocator_and_interpreter_overhead_excluded": True,
        },
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": ["NONABELIAN_S3_TRANSLATION_RELATIONS_CLOSE_UNDER_FULL_MATRIX_COMPOSITION", "S3_TRANSLATION_RELATIONS_CLOSE_UNDER_FULL_MATRIX_HADAMARD_INTERSECTION", "LEFT_AND_RIGHT_COMPOSITION_ARE_DISTINCT", "FULL72_CELL_ORACLE_MATCHES_ALL12_FINAL_BOUNDARIES_AND_COMMITMENTS", "EXACT_CARRIER_RESTORATION_AND_CROSS_FAMILY_REUSE", "S3_IRREP_TRANSFORM_HAS_FULL6_INFORMATION_RANK"],
        "negative_result": "THE_SMALLEST_NONCOMMUTATIVE_TRANSLATION_RELATION_ALGEBRA_BROADENS_THE_PHASE_COMPOSITION_LAW_BUT_HAS_IDENTICAL_SIX_COORDINATE_GROUP_AND_FULL_RANK_IRREP_CLASSICAL_RECURRENCES",
        "rejected_interpretations": production["not_established"] + ["GENERAL_FINITE_GROUP_RELATION_COMPILER", "GENERAL_SIX_LABEL_RELATIONS", "CATVM_CUSTODY"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("production_result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    production = json.loads(args.production_result.read_text(encoding="utf-8"))
    payload = json.dumps(build_result(production), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
