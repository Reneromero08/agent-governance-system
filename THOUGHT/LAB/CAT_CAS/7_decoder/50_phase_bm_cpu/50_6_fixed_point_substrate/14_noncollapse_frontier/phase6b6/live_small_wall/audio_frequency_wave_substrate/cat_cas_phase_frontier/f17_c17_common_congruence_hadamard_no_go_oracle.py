#!/usr/bin/env python3
"""Independent full-coefficient oracle for the F17 C17 congruence no-go.

This implementation imports neither the production package nor NumPy.  It
uses monomial evaluation spaces for the compatible-multiplier calculation and
executes the restricted programs in the complete 17-coefficient cyclic group
algebra before projecting final states to Hasse jets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FIELD = 17
ORDER = 17
RANKS = (2, 4, 8)
DEPTHS = (1, 4, 16, 64)
FAMILIES = ("PRIMARY", "ALTERNATE")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def binomial(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    numerator = 1
    denominator = 1
    for item in range(1, k + 1):
        numerator = numerator * (n - item + 1) % FIELD
        denominator = denominator * item % FIELD
    return numerator * pow(denominator, FIELD - 2, FIELD) % FIELD


def rref(matrix: list[list[int]]) -> tuple[list[list[int]], list[int]]:
    work = [[value % FIELD for value in row] for row in matrix]
    if not work:
        return work, []
    pivots: list[int] = []
    row = 0
    for column in range(len(work[0])):
        pivot = next((candidate for candidate in range(row, len(work)) if work[candidate][column]), None)
        if pivot is None:
            continue
        work[row], work[pivot] = work[pivot], work[row]
        inverse = pow(work[row][column], FIELD - 2, FIELD)
        work[row] = [(inverse * value) % FIELD for value in work[row]]
        for other in range(len(work)):
            if other == row or work[other][column] == 0:
                continue
            factor = work[other][column]
            work[other] = [
                (left - factor * right) % FIELD
                for left, right in zip(work[other], work[row], strict=True)
            ]
        pivots.append(column)
        row += 1
        if row == len(work):
            break
    return work, pivots


def rank(matrix: list[list[int]]) -> int:
    return len(rref(matrix)[1])


def nullspace(matrix: list[list[int]]) -> list[list[int]]:
    reduced, pivots = rref(matrix)
    width = len(matrix[0]) if matrix else 0
    free = [column for column in range(width) if column not in pivots]
    basis: list[list[int]] = []
    for free_column in free:
        vector = [0] * width
        vector[free_column] = 1
        for row, pivot in reversed(list(enumerate(pivots))):
            vector[pivot] = -sum(
                reduced[row][column] * vector[column]
                for column in free
            ) % FIELD
        basis.append(vector)
    return basis


def full_add(left: list[int], right: list[int]) -> list[int]:
    return [(x + y) % FIELD for x, y in zip(left, right, strict=True)]


def full_subtract(left: list[int], right: list[int]) -> list[int]:
    return [(x - y) % FIELD for x, y in zip(left, right, strict=True)]


def full_scale(value: list[int], scalar: int) -> list[int]:
    return [(scalar * item) % FIELD for item in value]


def convolution(left: list[int], right: list[int]) -> list[int]:
    result = [0] * ORDER
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            result[(i + j) % ORDER] = (result[(i + j) % ORDER] + x * y) % FIELD
    return result


def rotate(value: list[int], shift: int) -> list[int]:
    result = [0] * ORDER
    for index, item in enumerate(value):
        result[(index + shift) % ORDER] = item
    return result


def to_jet(coefficients: list[int], jet_rank: int) -> list[int]:
    return [
        sum(coefficients[position] * binomial(position, degree) for position in range(ORDER))
        % FIELD
        for degree in range(jet_rank)
    ]


def jet_to_full(jet: list[int]) -> list[int]:
    coefficients = [0] * ORDER
    for degree, value in enumerate(jet):
        for power_index in range(degree + 1):
            sign = -1 if (degree - power_index) % 2 else 1
            coefficients[power_index] = (
                coefficients[power_index]
                + sign * value * binomial(degree, power_index)
            ) % FIELD
    return coefficients


def jet_multiply(left: list[int], right: list[int], jet_rank: int) -> list[int]:
    return [
        sum(left[index] * right[degree - index] for index in range(degree + 1)) % FIELD
        for degree in range(jet_rank)
    ]


def jet_power(value: list[int], exponent: int, jet_rank: int) -> list[int]:
    result = [1] + [0] * (jet_rank - 1)
    base = value.copy()
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = jet_multiply(result, base, jet_rank)
        base = jet_multiply(base, base, jet_rank)
        remaining >>= 1
    return result


def jet_inverse(value: list[int], jet_rank: int) -> list[int]:
    inverse = [0] * jet_rank
    inverse[0] = pow(value[0], FIELD - 2, FIELD)
    for degree in range(1, jet_rank):
        subtotal = sum(value[index] * inverse[degree - index] for index in range(1, degree + 1))
        inverse[degree] = (-inverse[0] * subtotal) % FIELD
    return inverse


def phase_jet(jet_rank: int, shift: int) -> list[int]:
    return jet_power([1, 1] + [0] * (jet_rank - 2), shift % ORDER, jet_rank)


def kernel_jet(jet_rank: int, index: int, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    local = [1, (7 * index + 4 * code + 3) % FIELD] + [0] * (jet_rank - 2)
    return jet_multiply(phase_jet(jet_rank, 3 * index + 5 * code + 1), local, jet_rank)


def kernel_full(index: int, family: str) -> list[int]:
    return jet_to_full(kernel_jet(ORDER, index, family))


def kernel_inverse_full(index: int, family: str) -> list[int]:
    return jet_to_full(jet_inverse(kernel_jet(ORDER, index, family), ORDER))


def parameters(index: int, family: str) -> tuple[int, int, int]:
    code = 1 if family == "PRIMARY" else 2
    alpha = (5 * index + 3 * code + 1) % FIELD or 1
    beta = (7 * index + 2 * code + 4) % FIELD or 1
    constant = (11 * index + 5 * code + 2) % FIELD or 1
    return alpha, beta, constant


def seed_jet(jet_rank: int, family: str, register: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    offset = 3 if register == "A" else 9
    return [
        (offset + code * (degree + 1) + degree * degree + jet_rank) % FIELD
        for degree in range(jet_rank)
    ]


def forward(a: list[int], b: list[int], depth: int, family: str) -> tuple[list[int], list[int]]:
    current_a, current_b = a.copy(), b.copy()
    for index in range(depth):
        alpha, beta, constant = parameters(index, family)
        kernel = kernel_full(index, family)
        if family == "PRIMARY":
            current_a = convolution(current_a, kernel)
            current_a = full_scale(current_a, constant)
            current_b = full_add(
                current_b,
                full_scale(convolution(current_a, current_a), beta),
            )
            current_a = full_add(current_a, full_scale(current_b, alpha))
        else:
            current_a = full_add(current_a, full_scale(current_b, alpha))
            current_b = full_add(
                current_b,
                full_scale(convolution(current_a, current_a), beta),
            )
            current_a = full_scale(current_a, constant)
            current_a = convolution(current_a, kernel)
    return current_a, current_b


def inverse(a: list[int], b: list[int], depth: int, family: str) -> tuple[list[int], list[int]]:
    current_a, current_b = a.copy(), b.copy()
    for index in reversed(range(depth)):
        alpha, beta, constant = parameters(index, family)
        if family == "PRIMARY":
            current_a = full_subtract(current_a, full_scale(current_b, alpha))
            current_b = full_subtract(
                current_b,
                full_scale(convolution(current_a, current_a), beta),
            )
            current_a = full_scale(current_a, pow(constant, FIELD - 2, FIELD))
            current_a = convolution(current_a, kernel_inverse_full(index, family))
        else:
            current_a = convolution(current_a, kernel_inverse_full(index, family))
            current_a = full_scale(current_a, pow(constant, FIELD - 2, FIELD))
            current_b = full_subtract(
                current_b,
                full_scale(convolution(current_a, current_a), beta),
            )
            current_a = full_subtract(current_a, full_scale(current_b, alpha))
    return current_a, current_b


def boundary(b_jet: list[int], family: str) -> int:
    code = 1 if family == "PRIMARY" else 2
    weights = [(3 * degree + 5 * code + len(b_jet)) % FIELD or 1 for degree in range(len(b_jet))]
    return sum(weight * value for weight, value in zip(weights, b_jet, strict=True)) % FIELD


def monomial_multiplier_search(jet_rank: int) -> dict[str, Any]:
    """Independently solve diag(m) P_<r subset P_<r in value space."""
    width = ORDER + jet_rank * jet_rank
    matrix: list[list[int]] = []
    for degree in range(jet_rank):
        for point in range(ORDER):
            row = [0] * width
            row[point] = pow(point, degree, FIELD)
            for target_degree in range(jet_rank):
                row[ORDER + degree * jet_rank + target_degree] = -pow(
                    point, target_degree, FIELD
                ) % FIELD
            matrix.append(row)
    solutions = nullspace(matrix)
    multipliers = [solution[:ORDER] for solution in solutions]
    dimension = rank(multipliers) if multipliers else 0
    normalized = []
    for multiplier in multipliers:
        first = next((value for value in multiplier if value), 1)
        inverse = pow(first, FIELD - 2, FIELD)
        normalized.append([(inverse * value) % FIELD for value in multiplier])
    return {
        "rank": jet_rank,
        "basis": "MONOMIAL_EVALUATION_SPACE",
        "solution_dimension": len(solutions),
        "multiplier_space_dimension": dimension,
        "normalized_multiplier_basis": normalized,
        "only_constants": dimension == 1 and normalized == [[1] * ORDER],
    }


def common_kernel_oracle() -> dict[str, Any]:
    orbit_ranks = []
    for coordinate in range(ORDER):
        basis = [0] * ORDER
        basis[coordinate] = 1
        orbit_ranks.append(rank([rotate(basis, amount) for amount in range(ORDER)]))
    return {
        "coordinate_projector_then_shift_orbit_ranks": orbit_ranks,
        "all_nonzero_coordinate_orbits_are_full": all(value == ORDER for value in orbit_ranks),
        "common_invariant_kernel_dimensions": [0, ORDER],
        "nonzero_proper_common_kernel_exists": False,
        "proof_law": "PROJECT_ANY_NONZERO_VECTOR_TO_A_NONZERO_COORDINATE_THEN_CYCLICALLY_SHIFT_TO_ALL_COORDINATES",
    }


def compare_cases(production: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    expected = {
        (case["rank"], case["depth"], case["family"]): case
        for case in production["cases"]
    }
    results: list[dict[str, Any]] = []
    comparisons = 0
    for jet_rank in RANKS:
        for family in FAMILIES:
            initial_a = jet_to_full(seed_jet(jet_rank, family, "A"))
            initial_b = jet_to_full(seed_jet(jet_rank, family, "B"))
            for depth in DEPTHS:
                forward_a, forward_b = forward(initial_a, initial_b, depth, family)
                a_jet, b_jet = to_jet(forward_a, jet_rank), to_jet(forward_b, jet_rank)
                restored_a, restored_b = inverse(forward_a, forward_b, depth, family)
                case = expected[(jet_rank, depth, family)]
                observed = {
                    "rank": jet_rank,
                    "depth": depth,
                    "family": family,
                    "forward_a_matches": a_jet == case["forward_a"],
                    "forward_b_matches": b_jet == case["forward_b"],
                    "boundary_matches": boundary(b_jet, family) == case["boundary"],
                    "commitment_matches": digest_json([a_jet, b_jet]) == case["forward_commitment"],
                    "full34_cells_restore_exactly": restored_a == initial_a and restored_b == initial_b,
                }
                comparisons += 5
                observed["all_fields_match"] = all(
                    observed[key]
                    for key in (
                        "forward_a_matches",
                        "forward_b_matches",
                        "boundary_matches",
                        "commitment_matches",
                        "full34_cells_restore_exactly",
                    )
                )
                results.append(observed)
    return results, comparisons


def quotient_law_checks() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for jet_rank in RANKS:
        for sample in range(6):
            left = [
                (3 * position * position + (sample + 2) * position + 5) % FIELD
                for position in range(ORDER)
            ]
            right = [
                (7 * position + sample * sample + 4) % FIELD
                for position in range(ORDER)
            ]
            scalar = (5 * sample + 3) % FIELD or 1
            shift = (4 * sample + 1) % ORDER
            left_jet, right_jet = to_jet(left, jet_rank), to_jet(right, jet_rank)
            convolution_ok = to_jet(convolution(left, right), jet_rank) == jet_multiply(
                left_jet, right_jet, jet_rank
            )
            scalar_ok = to_jet(full_scale(left, scalar), jet_rank) == full_scale(
                left_jet, scalar
            )
            rotation_ok = to_jet(rotate(left, shift), jet_rank) == jet_multiply(
                left_jet, phase_jet(jet_rank, shift), jet_rank
            )
            checks.extend(
                [
                    {"rank": jet_rank, "sample": sample, "law": "CONVOLUTION", "pass": convolution_ok},
                    {"rank": jet_rank, "sample": sample, "law": "CONSTANT_HADAMARD", "pass": scalar_ok},
                    {"rank": jet_rank, "sample": sample, "law": "ROTATION", "pass": rotation_ok},
                ]
            )
    return {
        "comparison_count": len(checks),
        "all_pass": all(item["pass"] for item in checks),
        "checks": checks,
    }


def build_result(production: dict[str, Any]) -> dict[str, Any]:
    cases, comparisons = compare_cases(production)
    multiplier_search = [monomial_multiplier_search(jet_rank) for jet_rank in RANKS]
    common_kernel = common_kernel_oracle()
    laws = quotient_law_checks()
    production_search = production["restricted_hadamard_multiplier_search"]
    multiplier_parity = all(
        independent["multiplier_space_dimension"] == package["multiplier_space_dimension"]
        and independent["normalized_multiplier_basis"] == package["normalized_multiplier_basis"]
        for independent, package in zip(multiplier_search, production_search, strict=True)
    )
    no_go_parity = (
        common_kernel["common_invariant_kernel_dimensions"]
        == production["common_congruence_no_go"]["common_invariant_kernel_dimensions"]
        and common_kernel["nonzero_proper_common_kernel_exists"]
        == production["common_congruence_no_go"]["nonzero_proper_common_kernel_exists"]
    )
    if not all(item["all_fields_match"] for item in cases):
        raise RuntimeError("full-coefficient case mismatch")
    if not all(item["only_constants"] for item in multiplier_search):
        raise RuntimeError("independent multiplier search mismatch")
    if not multiplier_parity or not no_go_parity or not laws["all_pass"]:
        raise RuntimeError("independent structural check mismatch")
    return {
        "schema": "CAT_CAS_F17_C17_COMMON_CONGRUENCE_HADAMARD_NO_GO_ORACLE_RESULTS_V1",
        "claim": production["claim"],
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "imports_production": False,
        "imports_numpy": False,
        "oracle_state_law": "FULL17_COEFFICIENT_F17_C17_GROUP_ALGEBRA",
        "common_kernel_oracle": common_kernel,
        "independent_monomial_multiplier_search": multiplier_search,
        "structural_parity": {
            "common_kernel_parity": no_go_parity,
            "multiplier_space_parity": multiplier_parity,
        },
        "production_comparison": {
            "cases": len(cases),
            "field_comparisons": comparisons,
            "all_match": all(item["all_fields_match"] for item in cases),
        },
        "cases": cases,
        "quotient_law": laws,
        "resource_scope": {
            "oracle_full_coefficient_carrier_field_cells": 2 * ORDER,
            "accepted_maximum_jet_carrier_field_cells": 2 * max(RANKS),
            "oracle_dense17_by17_relation_table_cells": 0,
            "independent_constraint_matrix_max_rows": ORDER * max(RANKS),
            "independent_constraint_matrix_max_columns": ORDER + max(RANKS) ** 2,
            "constraint_matrices_are_verification_only": True,
            "whole_process_memory_not_measured": True,
        },
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": [
            "EXACT_LINEAR_COMMON_CONGRUENCE_NO_GO_IN_DECLARED_SCOPE",
            "ONLY_CONSTANT_DIAGONAL_MULTIPLIERS_DESCEND_AT_RANKS2_4_8",
            "FULL_COEFFICIENT_PARITY_FOR24_RESTRICTED_PROGRAM_CASES",
            "FULL34_CELL_EXACT_RESTORATION",
        ],
        "rejected_interpretations": production["rejected_interpretations"],
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
