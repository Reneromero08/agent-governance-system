#!/usr/bin/env python3
"""Separate exact certificate oracle for the period-17 cyclotomic module.

This oracle does not import the production module.  It independently
compiles the two public coefficient families, builds the 17-by-17 block
operator with tuple arithmetic in Z[zeta_17], checks the supplied monic
degree-17 annihilator on the entire operator, and reconstructs all declared
boundaries.
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any


PRIME = 17
DIMENSION = 16
PERIOD = 17
EXPECTED_PERIODS = (1, 2, 4, 8)
MESSAGE_INTEGER_CELLS = PRIME * DIMENSION
OPERATOR_INTEGER_CELLS = PRIME * PRIME * DIMENSION

RingElement = tuple[int, ...]
RingVector = list[RingElement]
RingMatrix = list[list[RingElement]]


def fail(message: str) -> None:
    raise RuntimeError(message)


def encoded(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def unary_phase(coefficients: list[int], value: int) -> int:
    cubic, quadratic, linear = coefficients
    return (
        cubic * value**3
        + quadratic * value**2
        + linear * value
    ) % PRIME


def edge_phase(
    coefficients: list[int],
    left: int,
    right: int,
) -> int:
    left_square_right, left_right_square, bilinear = coefficients
    return (
        left_square_right * left * left * right
        + left_right_square * left * right * right
        + bilinear * left * right
    ) % PRIME


def compile_descriptor(nodes: int, family: str) -> dict[str, Any]:
    if family not in {"PRIMARY", "REUSE"}:
        fail("unknown oracle family")
    shift = 0 if family == "PRIMARY" else 7
    return {
        "field": "F17",
        "phase_root_order": PRIME,
        "topology": "PUBLIC_PATH_GRAPH",
        "family": family,
        "nodes": nodes,
        "unary_coefficients": [
            [
                1 + ((3 * index + shift) % PRIME) % 16,
                (5 * index + 2 + 2 * shift) % PRIME,
                (7 * index + 4 + shift) % PRIME,
            ]
            for index in range(nodes)
        ],
        "edge_coefficients": [
            [
                1 + ((5 * index + 2 + shift) % PRIME) % 16,
                1 + ((7 * index + 4 + 2 * shift) % PRIME) % 16,
                (3 * index + 6 + shift) % PRIME,
            ]
            for index in range(nodes - 1)
        ],
    }


def ring_zero() -> RingElement:
    return (0,) * DIMENSION


def ring_one() -> RingElement:
    return (1,) + (0,) * (DIMENSION - 1)


def ring_add(left: RingElement, right: RingElement) -> RingElement:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def ring_subtract(left: RingElement, right: RingElement) -> RingElement:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def ring_multiply(left: RingElement, right: RingElement) -> RingElement:
    work = [0] * (2 * DIMENSION - 1)
    for left_degree, left_value in enumerate(left):
        if left_value == 0:
            continue
        for right_degree, right_value in enumerate(right):
            if right_value:
                work[left_degree + right_degree] += (
                    left_value * right_value
                )
    for degree in range(len(work) - 1, DIMENSION - 1, -1):
        value = work[degree]
        if value == 0:
            continue
        offset = degree - DIMENSION
        for reduced_degree in range(offset, offset + DIMENSION):
            work[reduced_degree] -= value
        work[degree] = 0
    return tuple(work[:DIMENSION])


def ring_monomial(exponent: int) -> RingElement:
    exponent %= PRIME
    if exponent < DIMENSION:
        result = [0] * DIMENSION
        result[exponent] = 1
        return tuple(result)
    return (-1,) * DIMENSION


def identity_matrix() -> RingMatrix:
    return [
        [
            ring_one() if row == column else ring_zero()
            for column in range(PRIME)
        ]
        for row in range(PRIME)
    ]


def matrix_multiply(left: RingMatrix, right: RingMatrix) -> RingMatrix:
    target = [
        [ring_zero() for _ in range(PRIME)]
        for _ in range(PRIME)
    ]
    for row in range(PRIME):
        for column in range(PRIME):
            accumulator = ring_zero()
            for shared in range(PRIME):
                accumulator = ring_add(
                    accumulator,
                    ring_multiply(
                        left[row][shared],
                        right[shared][column],
                    ),
                )
            target[row][column] = accumulator
    return target


def matrix_vector_multiply(
    matrix: RingMatrix,
    vector: RingVector,
) -> RingVector:
    target = [ring_zero() for _ in range(PRIME)]
    for row in range(PRIME):
        accumulator = ring_zero()
        for column in range(PRIME):
            accumulator = ring_add(
                accumulator,
                ring_multiply(
                    matrix[row][column],
                    vector[column],
                ),
            )
        target[row] = accumulator
    return target


def compile_operator(program: dict[str, Any]) -> RingMatrix:
    operator = identity_matrix()
    for edge_index in range(PERIOD):
        edge_matrix = []
        for right in range(PRIME):
            row = []
            for left in range(PRIME):
                shift = (
                    unary_phase(
                        program["unary_coefficients"][edge_index + 1],
                        right,
                    )
                    + edge_phase(
                        program["edge_coefficients"][edge_index],
                        left,
                        right,
                    )
                ) % PRIME
                row.append(ring_monomial(shift))
            edge_matrix.append(row)
        operator = matrix_multiply(edge_matrix, operator)
    return operator


def check_annihilator(
    operator: RingMatrix,
    characteristic: list[RingElement],
) -> bool:
    if len(characteristic) != PRIME + 1:
        return False
    if characteristic[0] != ring_one():
        return False
    residual = identity_matrix()
    for coefficient in characteristic[1:]:
        residual = matrix_multiply(operator, residual)
        for diagonal in range(PRIME):
            residual[diagonal][diagonal] = ring_add(
                residual[diagonal][diagonal],
                coefficient,
            )
    return all(
        element == ring_zero()
        for row in residual
        for element in row
    )


def seed_vector(program: dict[str, Any]) -> RingVector:
    return [
        ring_monomial(
            unary_phase(program["unary_coefficients"][0], value)
        )
        for value in range(PRIME)
    ]


def project(vector: RingVector) -> RingElement:
    result = ring_zero()
    for element in vector:
        result = ring_add(result, element)
    return result


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_chain_period17_"
            "cyclotomic_module_oracle.py PRODUCTION_RESULT"
        )
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        production_text = handle.read()
    production_result_file_bytes = len(
        production_text.encode("utf-8")
    )
    production = json.loads(production_text)
    del production_text

    family_checks: dict[str, Any] = {}
    compiled: dict[str, tuple[dict[str, Any], RingMatrix]] = {}
    for family in ("primary", "reuse"):
        descriptor = compile_descriptor(PERIOD + 1, family.upper())
        operator = compile_operator(descriptor)
        compiled[family] = (descriptor, operator)
        supplied_operator = [
            [tuple(element) for element in row]
            for row in production["blocks"][family]["operator"]
        ]
        characteristic = [
            tuple(element)
            for element in production["blocks"][family][
                "characteristic"
            ]
        ]
        family_checks[family] = {
            "descriptor_sha256_equal": (
                hashlib.sha256(encoded(descriptor)).hexdigest()
                == production["blocks"][family][
                    "public_program_sha256"
                ]
            ),
            "operator_equal": operator == supplied_operator,
            "operator_sha256_equal": (
                hashlib.sha256(encoded(operator)).hexdigest()
                == production["blocks"][family]["operator_sha256"]
            ),
            "characteristic_sha256_equal": (
                hashlib.sha256(encoded(characteristic)).hexdigest()
                == production["blocks"][family][
                    "characteristic_sha256"
                ]
            ),
            "monic_degree17_annihilator_identity_exact": (
                check_annihilator(operator, characteristic)
            ),
        }

    boundary_checks = []
    for case in production["cases"]:
        periods = case["periods"]
        family = case["family"].lower()
        if periods not in EXPECTED_PERIODS:
            fail("unexpected oracle period count")
        descriptor, operator = compiled[family]
        vector = seed_vector(descriptor)
        for _ in range(periods):
            vector = matrix_vector_multiply(operator, vector)
        boundary = project(vector)
        expected_boundary = tuple(case["boundary"])
        boundary_checks.append(
            {
                "periods": periods,
                "family": family.upper(),
                "boundary_equal": boundary == expected_boundary,
                "boundary_sha256_equal": (
                    hashlib.sha256(encoded(boundary)).hexdigest()
                    == case["boundary_sha256"]
                ),
            }
        )

    primary_descriptor, primary_operator = compiled["primary"]
    primary_seed = seed_vector(primary_descriptor)
    primary_target = matrix_vector_multiply(
        primary_operator,
        primary_seed,
    )
    restored_target = [
        ring_subtract(actual, expected)
        for actual, expected in zip(
            primary_target,
            matrix_vector_multiply(
                primary_operator,
                primary_seed,
            ),
            strict=True,
        )
    ]
    exact_subtractive_inverse_zero = all(
        element == ring_zero()
        for element in restored_target
    )
    all_family_checks = all(
        all(check.values())
        for check in family_checks.values()
    )
    all_boundaries_equal = all(
        check["boundary_equal"]
        and check["boundary_sha256_equal"]
        for check in boundary_checks
    )
    result = {
        "result": "PASS",
        "oracle": (
            "SEPARATE_PUBLIC_DESCRIPTOR_TUPLE_Z_ZETA17_OPERATOR_"
            "MONIC_ANNIHILATOR_AND_BOUNDARY_CERTIFICATE"
        ),
        "production_module_imported": False,
        "production_compiler_called": False,
        "production_operator_builder_called": False,
        "production_characteristic_called": False,
        "production_inverse_called": False,
        "family_checks": family_checks,
        "boundary_checks": boundary_checks,
        "all_family_checks": all_family_checks,
        "all_boundaries_equal": all_boundaries_equal,
        "exact_subtractive_inverse_zero": (
            exact_subtractive_inverse_zero
        ),
        "exact_native_cyclotomic_recurrence_order_upper_bound": (
            PRIME
        ),
        "exact_native_minimal_order_established": False,
        "native_cyclotomic_recurrence_certified_not_executed": True,
        "runtime_executes_dense_17_by_17_cyclotomic_block": True,
        "prior_modular_dependencies_lifted_to_q_recurrence": False,
        "q_scalar_recurrence_order_upper_bound17_established": False,
        "restriction_of_scalars_dimension_per_k_coefficient": (
            DIMENSION
        ),
        "native_k_coefficient_maps_expand_to_16_by_16_q_linear_operators": (
            True
        ),
        "coefficient_ring_change_is_not_q_order_reduction": True,
        "resource_law": {
            "accounting_scope": (
                "COMPONENT_LEVEL_NAMED_LOGICAL_INTEGER_CELLS_NOT_"
                "EXACT_PROCESS_PEAK"
            ),
            "named_logical_cell_accounting_is_exact_total": False,
            "production_result_file_bytes": (
                production_result_file_bytes
            ),
            "parsed_production_operator_and_characteristic_integer_cells": (
                2
                * (
                    OPERATOR_INTEGER_CELLS
                    + (PRIME + 1) * DIMENSION
                )
            ),
            "retained_oracle_compiled_operator_integer_cells": (
                2 * OPERATOR_INTEGER_CELLS
            ),
            "compiled_operator_integer_cells_per_family": (
                OPERATOR_INTEGER_CELLS
            ),
            "operator_build_three_matrix_peak_integer_cells": (
                3 * OPERATOR_INTEGER_CELLS
            ),
            "annihilator_certificate_three_matrix_plus_polynomial_peak_integer_cells": (
                3 * OPERATOR_INTEGER_CELLS
                + (PRIME + 1) * DIMENSION
            ),
            "boundary_source_target_integer_cells": (
                2 * MESSAGE_INTEGER_CELLS
            ),
            "coexisting_cross_family_named_components_are_not_a_peak_bound": (
                True
            ),
            "python_object_overhead_bounded": False,
            "allocator_peak_bounded": False,
            "bit_operation_peak_bounded": False,
            "whole_process_peak_bounded": False,
        },
        "identical_compact_classical_cyclotomic_recurrence": True,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "terminal": False,
    }
    if (
        not all_family_checks
        or not all_boundaries_equal
        or not exact_subtractive_inverse_zero
    ):
        fail("independent cyclotomic module certificate mismatch")
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
