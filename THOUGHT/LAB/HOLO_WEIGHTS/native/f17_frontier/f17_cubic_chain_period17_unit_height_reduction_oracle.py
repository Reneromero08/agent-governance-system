#!/usr/bin/env python3
"""Separate exact oracle for bounded cyclotomic-unit height reduction.

This file imports no production module.  It independently compiles the two
public descriptors and period operators with tuple Z[zeta_17] arithmetic,
checks supplied whole-operator annihilators, advances coefficients through a
sequential x-mod-q law, and independently reexecutes the declared seven-generator
unit normalization.  It reproduces exact boundaries, ledger-inclusive carrier
payloads, coefficient widths, search counts, and subtractive restoration.
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any


PRIME = 17
DIMENSION = 16
PERIOD = 17
EXPECTED_PERIODS = (1, 4, 16, 64, 256)
MESSAGE_INTEGER_CELLS = PRIME * DIMENSION
OPERATOR_INTEGER_CELLS = PRIME * PRIME * DIMENSION
MESSAGE_SLOTS = 18
OUTPUT_SLOT = MESSAGE_SLOTS - 1
UNIT_GENERATOR_INDICES = tuple(range(2, 9))
UNIT_RANK = len(UNIT_GENERATOR_INDICES)
MAX_NORMALIZATION_STEPS = 128

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


def ring_subtract(
    left: RingElement,
    right: RingElement,
) -> RingElement:
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


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def unit_generator(index: int) -> RingElement:
    result = ring_zero()
    for exponent in range(index):
        result = ring_add(result, ring_monomial(exponent))
    return result


def unit_generator_inverse(index: int) -> RingElement:
    inverse_index = pow(index, -1, PRIME)
    result = ring_zero()
    for multiplier in range(inverse_index):
        result = ring_add(
            result,
            ring_monomial((index * multiplier) % PRIME),
        )
    return result


UNIT_GENERATORS = tuple(
    unit_generator(index)
    for index in UNIT_GENERATOR_INDICES
)
UNIT_GENERATOR_INVERSES = tuple(
    unit_generator_inverse(index)
    for index in UNIT_GENERATOR_INDICES
)


def unit_identities_exact() -> bool:
    return all(
        ring_multiply(generator, inverse) == ring_one()
        for generator, inverse in zip(
            UNIT_GENERATORS,
            UNIT_GENERATOR_INVERSES,
            strict=True,
        )
    )


def vector_payload_bits(vector: RingVector) -> int:
    return sum(
        signed_bits(coefficient)
        for element in vector
        for coefficient in element
    )


def vector_maximum_signed_bits(vector: RingVector) -> int:
    return max(
        signed_bits(coefficient)
        for element in vector
        for coefficient in element
    )


def ledger_payload_bits(ledger: list[int]) -> int:
    return sum(signed_bits(exponent) for exponent in ledger)


def multiply_vector_by_scalar(
    scalar: RingElement,
    vector: RingVector,
) -> RingVector:
    return [ring_multiply(scalar, value) for value in vector]


def normalized_vector(
    vector: RingVector,
    base_ledger: list[int],
) -> tuple[RingVector, list[int], dict[str, int]]:
    current = list(vector)
    ledger = list(base_ledger)
    current_cost = vector_payload_bits(current) + ledger_payload_bits(ledger)
    candidate_evaluations = 0
    selected_steps = 0
    for _ in range(MAX_NORMALIZATION_STEPS):
        best: tuple[
            int,
            int,
            tuple[int, ...],
            int,
            int,
        ] | None = None
        for generator_index in range(UNIT_RANK):
            for delta, multiplier in (
                (1, UNIT_GENERATOR_INVERSES[generator_index]),
                (-1, UNIT_GENERATORS[generator_index]),
            ):
                trial = multiply_vector_by_scalar(multiplier, current)
                candidate_evaluations += 1
                trial_ledger = list(ledger)
                trial_ledger[generator_index] += delta
                choice = (
                    vector_payload_bits(trial)
                    + ledger_payload_bits(trial_ledger),
                    vector_maximum_signed_bits(trial),
                    tuple(trial_ledger),
                    generator_index,
                    delta,
                )
                if best is None or choice < best:
                    best = choice
        if best is None or best[0] >= current_cost:
            break
        generator_index = best[3]
        delta = best[4]
        multiplier = (
            UNIT_GENERATOR_INVERSES[generator_index]
            if delta == 1
            else UNIT_GENERATORS[generator_index]
        )
        current = multiply_vector_by_scalar(multiplier, current)
        candidate_evaluations += 1
        ledger[generator_index] += delta
        current_cost = best[0]
        selected_steps += 1
    return current, ledger, {
        "normalization_calls": 1,
        "normalization_step_cap_hits": int(
            selected_steps == MAX_NORMALIZATION_STEPS
        ),
        "normalization_candidate_vector_evaluations": (
            candidate_evaluations
        ),
        "normalization_selected_steps": selected_steps,
        "normalization_vector_ring_multiplications": (
            candidate_evaluations * len(vector)
        ),
    }


def ledger_scale(ledger: list[int]) -> RingElement:
    result = ring_one()
    for exponent, generator, inverse in zip(
        ledger,
        UNIT_GENERATORS,
        UNIT_GENERATOR_INVERSES,
        strict=True,
    ):
        result = ring_multiply(
            result,
            ring_power(
                generator if exponent >= 0 else inverse,
                abs(exponent),
            ),
        )
    return result


def ring_power(base: RingElement, exponent: int) -> RingElement:
    result = ring_one()
    factor = base
    while exponent:
        if exponent & 1:
            result = ring_multiply(result, factor)
        exponent >>= 1
        if exponent:
            factor = ring_multiply(factor, factor)
    return result


def maximum_vector_signed_bits(vectors: list[RingVector]) -> int:
    return max(
        signed_bits(coefficient)
        for vector in vectors
        for element in vector
        for coefficient in element
    )


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
                ring_multiply(matrix[row][column], vector[column]),
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


def sequential_coefficients(
    periods: int,
    characteristic: list[RingElement],
) -> list[RingElement]:
    if periods < 1:
        fail("oracle recurrence requires at least one period")
    if characteristic[-1] != ring_zero():
        fail("oracle expected a singular characteristic")
    coefficients = [ring_zero() for _ in range(DIMENSION)]
    coefficients[0] = ring_one()
    for _ in range(periods - 1):
        highest = coefficients[-1]
        advanced = [ring_zero() for _ in range(DIMENSION)]
        for degree in range(DIMENSION):
            shifted = (
                coefficients[degree - 1]
                if degree > 0
                else ring_zero()
            )
            advanced[degree] = ring_subtract(
                shifted,
                ring_multiply(
                    highest,
                    characteristic[DIMENSION - degree],
                ),
            )
        coefficients = advanced
    return coefficients


def build_basis(
    operator: RingMatrix,
    seed: RingVector,
) -> list[RingVector]:
    basis = []
    current = seed
    for _ in range(DIMENSION):
        current = matrix_vector_multiply(operator, current)
        basis.append(current)
    return basis


def linear_combination(
    coefficients: list[RingElement],
    basis: list[RingVector],
) -> RingVector:
    target = [ring_zero() for _ in range(PRIME)]
    for scalar, vector in zip(coefficients, basis, strict=True):
        for phase_index, value in enumerate(vector):
            target[phase_index] = ring_add(
                target[phase_index],
                ring_multiply(scalar, value),
            )
    return target


def project(vector: RingVector) -> RingElement:
    result = ring_zero()
    for element in vector:
        result = ring_add(result, element)
    return result


def direct_boundary_with_metrics(
    operator: RingMatrix,
    seed: RingVector,
    periods: int,
) -> tuple[RingElement, dict[str, int]]:
    current = seed
    maximum_two_message_payload_bits = 0
    maximum_coefficient_signed_bits = 1
    for _ in range(periods):
        target = matrix_vector_multiply(operator, current)
        maximum_two_message_payload_bits = max(
            maximum_two_message_payload_bits,
            vector_payload_bits(current) + vector_payload_bits(target),
        )
        maximum_coefficient_signed_bits = max(
            maximum_coefficient_signed_bits,
            maximum_vector_signed_bits([current, target]),
        )
        current = target
    return project(current), {
        "maximum_two_message_payload_bits": (
            maximum_two_message_payload_bits
        ),
        "maximum_coefficient_signed_bits": (
            maximum_coefficient_signed_bits
        ),
    }


def recurrence_carrier_metrics(
    seed: RingVector,
    basis: list[RingVector],
    coefficients: list[RingElement],
    output: RingVector,
) -> dict[str, int]:
    zero_message = [ring_zero() for _ in range(PRIME)]
    zero_coefficients = [ring_zero() for _ in range(DIMENSION)]
    basis_stage_messages = [seed, *basis, zero_message]
    output_stage_messages = [seed, *basis, output]

    def payload(
        messages: list[RingVector],
        registers: list[RingElement],
    ) -> int:
        return (
            sum(vector_payload_bits(message) for message in messages)
            + sum(
                signed_bits(coefficient)
                for element in registers
                for coefficient in element
            )
        )

    return {
        "maximum_carrier_payload_bits": max(
            payload(basis_stage_messages, zero_coefficients),
            payload(output_stage_messages, coefficients),
        ),
        "maximum_coefficient_signed_bits": max(
            maximum_vector_signed_bits(basis_stage_messages),
            maximum_vector_signed_bits(output_stage_messages),
            max(
                signed_bits(coefficient)
                for element in coefficients
                for coefficient in element
            ),
        ),
        "maximum_nonzero_message_slots": max(
            sum(
                any(element != ring_zero() for element in message)
                for message in basis_stage_messages
            ),
            sum(
                any(element != ring_zero() for element in message)
                for message in output_stage_messages
            ),
        ),
        "maximum_nonzero_coefficient_registers": sum(
            element != ring_zero()
            for element in coefficients
        ),
    }


def create_carrier() -> dict[str, Any]:
    return {
        "messages": [
            [ring_zero() for _ in range(PRIME)]
            for _ in range(MESSAGE_SLOTS)
        ],
        "coefficients": [
            ring_zero()
            for _ in range(DIMENSION)
        ],
        "generation": 0,
        "lease": 0,
        "active": False,
    }


def carrier_is_zero(carrier: dict[str, Any]) -> bool:
    return (
        all(
            element == ring_zero()
            for message in carrier["messages"]
            for element in message
        )
        and all(
            value == ring_zero()
            for value in carrier["coefficients"]
        )
    )


def carrier_backing_identity(carrier: dict[str, Any]) -> tuple[int, ...]:
    return (
        id(carrier["messages"]),
        *(id(message) for message in carrier["messages"]),
        id(carrier["coefficients"]),
    )


def execute_on_carrier(
    carrier: dict[str, Any],
    operator: RingMatrix,
    seed: RingVector,
    characteristic: list[RingElement],
    periods: int,
) -> tuple[RingElement, dict[str, Any]]:
    if carrier["active"] or not carrier_is_zero(carrier):
        fail("oracle carrier was not restored")
    backing = carrier_backing_identity(carrier)
    carrier["active"] = True
    carrier["lease"] += 1
    messages = carrier["messages"]
    coefficients = carrier["coefficients"]
    messages[0][:] = seed
    for index in range(DIMENSION):
        messages[index + 1][:] = matrix_vector_multiply(
            operator,
            messages[index],
        )
    coefficients[:] = sequential_coefficients(periods, characteristic)
    messages[-1][:] = linear_combination(
        coefficients,
        messages[1:DIMENSION + 1],
    )
    boundary = project(messages[-1])
    expected_output = linear_combination(
        coefficients,
        messages[1:DIMENSION + 1],
    )
    messages[-1][:] = [
        ring_subtract(value, expected)
        for value, expected in zip(
            messages[-1],
            expected_output,
            strict=True,
        )
    ]
    coefficients[:] = [
        ring_subtract(value, expected)
        for value, expected in zip(
            coefficients,
            sequential_coefficients(periods, characteristic),
            strict=True,
        )
    ]
    for index in range(DIMENSION, 0, -1):
        expected = matrix_vector_multiply(
            operator,
            messages[index - 1],
        )
        messages[index][:] = [
            ring_subtract(value, expected_value)
            for value, expected_value in zip(
                messages[index],
                expected,
                strict=True,
            )
        ]
    messages[0][:] = [
        ring_subtract(value, expected)
        for value, expected in zip(
            messages[0],
            seed,
            strict=True,
        )
    ]
    carrier["active"] = False
    carrier["generation"] += 1
    restored = carrier_is_zero(carrier)
    same_backing = carrier_backing_identity(carrier) == backing
    return boundary, {
        "restored_exactly": restored,
        "same_message_and_coefficient_backing": same_backing,
        "generation": carrier["generation"],
        "lease": carrier["lease"],
        "message_slots": MESSAGE_SLOTS,
        "message_integer_cells": (
            MESSAGE_SLOTS * MESSAGE_INTEGER_CELLS
        ),
        "coefficient_register_integer_cells": (
            DIMENSION * DIMENSION
        ),
    }


def exact_restoration_witness(
    operator: RingMatrix,
    seed: RingVector,
    characteristic: list[RingElement],
    periods: int,
) -> dict[str, Any]:
    _, witness = execute_on_carrier(
        create_carrier(),
        operator,
        seed,
        characteristic,
        periods,
    )
    return witness


def legacy_executed_recurrence_main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_chain_period17_"
            "executed_recurrence_oracle.py PRODUCTION_RESULT"
        )
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        production_text = handle.read()
    production_result_file_bytes = len(
        production_text.encode("utf-8")
    )
    production = json.loads(production_text)
    del production_text

    compiled: dict[
        str,
        tuple[
            dict[str, Any],
            RingMatrix,
            list[RingElement],
            RingVector,
            list[RingVector],
        ],
    ] = {}
    family_checks: dict[str, Any] = {}
    for family in ("primary", "reuse"):
        descriptor = compile_descriptor(PERIOD + 1, family.upper())
        operator = compile_operator(descriptor)
        certificate = production["block_certificates"][family]
        characteristic = [
            tuple(value)
            for value in certificate["characteristic"]
        ]
        seed = seed_vector(descriptor)
        basis = build_basis(operator, seed)
        compiled[family] = (
            descriptor,
            operator,
            characteristic,
            seed,
            basis,
        )
        family_checks[family] = {
            "descriptor_sha256_equal": (
                hashlib.sha256(encoded(descriptor)).hexdigest()
                == certificate["public_program_sha256"]
            ),
            "operator_sha256_equal": (
                hashlib.sha256(encoded(operator)).hexdigest()
                == certificate["operator_sha256"]
            ),
            "characteristic_sha256_equal": (
                hashlib.sha256(encoded(characteristic)).hexdigest()
                == certificate["characteristic_sha256"]
            ),
            "characteristic_constant_zero": (
                characteristic[-1] == ring_zero()
            ),
            "whole_operator_annihilator_identity_exact": (
                check_annihilator(operator, characteristic)
            ),
        }

    case_checks = []
    for case in production["cases"]:
        periods = case["periods"]
        family = case["family"].lower()
        if periods not in EXPECTED_PERIODS:
            fail("unexpected oracle recurrence period")
        _, operator, characteristic, seed, basis = compiled[family]
        coefficients = sequential_coefficients(
            periods,
            characteristic,
        )
        recurrence_boundary = project(
            linear_combination(coefficients, basis)
        )
        recurrence_output = linear_combination(coefficients, basis)
        recurrence_metrics = recurrence_carrier_metrics(
            seed,
            basis,
            coefficients,
            recurrence_output,
        )
        direct, direct_metrics = direct_boundary_with_metrics(
            operator,
            seed,
            periods,
        )
        expected = tuple(case["boundary"])
        case_checks.append(
            {
                "periods": periods,
                "family": family.upper(),
                "sequential_recurrence_boundary_equal": (
                    recurrence_boundary == expected
                ),
                "direct_dense_boundary_equal": direct == expected,
                "boundary_sha256_equal": (
                    hashlib.sha256(encoded(recurrence_boundary)).hexdigest()
                    == case["boundary_sha256"]
                ),
                "boundary_payload_bits_equal": (
                    sum(signed_bits(value) for value in recurrence_boundary)
                    == case["boundary_payload_bits"]
                ),
                "recurrence_maximum_carrier_payload_bits_equal": (
                    recurrence_metrics["maximum_carrier_payload_bits"]
                    == case["stats"]["maximum_carrier_payload_bits"]
                ),
                "recurrence_maximum_coefficient_signed_bits_equal": (
                    recurrence_metrics["maximum_coefficient_signed_bits"]
                    == case["stats"]["maximum_coefficient_signed_bits"]
                ),
                "recurrence_maximum_nonzero_message_slots_equal": (
                    recurrence_metrics["maximum_nonzero_message_slots"]
                    == case["stats"]["maximum_nonzero_message_slots"]
                ),
                "recurrence_maximum_nonzero_coefficient_registers_equal": (
                    recurrence_metrics[
                        "maximum_nonzero_coefficient_registers"
                    ]
                    == case["stats"][
                        "maximum_nonzero_coefficient_registers"
                    ]
                ),
                "direct_maximum_two_message_payload_bits_equal": (
                    direct_metrics["maximum_two_message_payload_bits"]
                    == case["dense_direct_stats"][
                        "maximum_two_message_payload_bits"
                    ]
                ),
                "direct_maximum_coefficient_signed_bits_equal": (
                    direct_metrics["maximum_coefficient_signed_bits"]
                    == case["dense_direct_stats"][
                        "maximum_coefficient_signed_bits"
                    ]
                ),
            }
        )

    restoration = {
        family: exact_restoration_witness(
            compiled[family][1],
            compiled[family][3],
            compiled[family][2],
            max(EXPECTED_PERIODS),
        )
        for family in ("primary", "reuse")
    }
    all_family_checks = all(
        all(check.values())
        for check in family_checks.values()
    )
    all_case_checks = all(
        all(
            value
            for key, value in check.items()
            if key not in {"periods", "family"}
        )
        for check in case_checks
    )
    all_restored = all(
        check["restored_exactly"]
        and check["same_message_and_coefficient_backing"]
        for check in restoration.values()
    )
    shared_carrier = create_carrier()
    shared_backing = carrier_backing_identity(shared_carrier)
    primary_boundary, primary_reuse_witness = execute_on_carrier(
        shared_carrier,
        compiled["primary"][1],
        compiled["primary"][3],
        compiled["primary"][2],
        max(EXPECTED_PERIODS),
    )
    reuse_boundary, reuse_witness = execute_on_carrier(
        shared_carrier,
        compiled["reuse"][1],
        compiled["reuse"][3],
        compiled["reuse"][2],
        max(EXPECTED_PERIODS),
    )
    fresh_reuse_boundary, fresh_reuse_witness = execute_on_carrier(
        create_carrier(),
        compiled["reuse"][1],
        compiled["reuse"][3],
        compiled["reuse"][2],
        max(EXPECTED_PERIODS),
    )
    expected_primary_sha256 = next(
        case["boundary_sha256"]
        for case in production["cases"]
        if case["family"] == "PRIMARY"
        and case["periods"] == max(EXPECTED_PERIODS)
    )
    expected_reuse_sha256 = next(
        case["boundary_sha256"]
        for case in production["cases"]
        if case["family"] == "REUSE"
        and case["periods"] == max(EXPECTED_PERIODS)
    )
    production_reuse = production["restoration_reuse_case"]
    cross_family_reuse = {
        "primary_restored_exactly": (
            primary_reuse_witness["restored_exactly"]
        ),
        "reuse_restored_exactly": reuse_witness["restored_exactly"],
        "same_original_message_and_coefficient_backing": (
            carrier_backing_identity(shared_carrier) == shared_backing
            and primary_reuse_witness[
                "same_message_and_coefficient_backing"
            ]
            and reuse_witness[
                "same_message_and_coefficient_backing"
            ]
        ),
        "primary_boundary_sha256_equal": (
            hashlib.sha256(encoded(primary_boundary)).hexdigest()
            == expected_primary_sha256
        ),
        "reuse_boundary_sha256_equal": (
            hashlib.sha256(encoded(reuse_boundary)).hexdigest()
            == expected_reuse_sha256
        ),
        "fresh_restored_reuse_boundary_equal": (
            reuse_boundary == fresh_reuse_boundary
        ),
        "fresh_reuse_restored_exactly": (
            fresh_reuse_witness["restored_exactly"]
        ),
        "generation": shared_carrier["generation"],
        "lease": shared_carrier["lease"],
        "production_generation_equal": (
            shared_carrier["generation"]
            == production_reuse["generation"]
        ),
        "production_lease_equal": (
            shared_carrier["lease"] == production_reuse["lease"]
        ),
        "all_state_zero": carrier_is_zero(shared_carrier),
        "separate_inverse_operation_log_bytes": 0,
        "baseline_reload_bytes": 0,
    }
    all_cross_family_reuse = all(
        value == 2 if key in {"generation", "lease"} else value is True
        for key, value in cross_family_reuse.items()
        if key not in {
            "separate_inverse_operation_log_bytes",
            "baseline_reload_bytes",
        }
    )
    result = {
        "result": "PASS",
        "oracle": (
            "SEPARATE_PUBLIC_DESCRIPTOR_TUPLE_Z_ZETA17_OPERATOR_"
            "WHOLE_ANNIHILATOR_SEQUENTIAL_RECURRENCE_DIRECT_"
            "BOUNDARY_EXACT_RESOURCE_TUPLES_AND_SUBTRACTIVE_"
            "RESTORATION"
        ),
        "production_module_imported": False,
        "production_compiler_called": False,
        "production_binary_polynomial_power_called": False,
        "production_basis_builder_called": False,
        "production_inverse_called": False,
        "family_checks": family_checks,
        "case_checks": case_checks,
        "restoration_checks": restoration,
        "cross_family_restored_reuse_check": cross_family_reuse,
        "all_family_checks": all_family_checks,
        "all_case_checks": all_case_checks,
        "all_restored_exactly": all_restored,
        "all_cross_family_reuse_checks": all_cross_family_reuse,
        "independent_recurrence_method": (
            "SEQUENTIAL_MULTIPLICATION_BY_X_MOD_MONIC_Q16"
        ),
        "independent_exact_integer_payload_and_width_reexecution": True,
        "fixed_18_resident_message_slots_confirmed": True,
        "fixed_16_cyclotomic_coefficient_registers_confirmed": True,
        "fixed_slots_are_not_fixed_total_footprint": True,
        "reversible_integral_rolling_window_established": False,
        "resource_law": {
            "accounting_scope": (
                "COMPONENT_LEVEL_NAMED_LOGICAL_INTEGER_CELLS_NOT_"
                "EXACT_PROCESS_PEAK"
            ),
            "production_result_file_bytes": (
                production_result_file_bytes
            ),
            "compiled_operator_integer_cells": (
                2 * OPERATOR_INTEGER_CELLS
            ),
            "parsed_characteristic_integer_cells": (
                2 * (PRIME + 1) * DIMENSION
            ),
            "verification_prebuilt_basis_integer_cells_two_families": (
                2 * DIMENSION * MESSAGE_INTEGER_CELLS
            ),
            "restoration_message_integer_cells_per_family": (
                MESSAGE_SLOTS * MESSAGE_INTEGER_CELLS
            ),
            "coefficient_register_integer_cells_per_family": (
                DIMENSION * DIMENSION
            ),
            "named_logical_cell_accounting_is_exact_total": False,
            "coexisting_cross_family_components_are_not_a_peak_bound": (
                True
            ),
            "python_object_overhead_bounded": False,
            "allocator_peak_bounded": False,
            "bit_operation_peak_bounded": False,
            "whole_process_peak_bounded": False,
        },
        "identical_compact_classical_recurrence": True,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "terminal": False,
    }
    if (
        not all_family_checks
        or not all_case_checks
        or not all_restored
        or not all_cross_family_reuse
    ):
        fail("independent executed recurrence certificate mismatch")
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def empty_unit_stats() -> dict[str, int]:
    return {
        "normalization_calls": 0,
        "normalization_step_cap_hits": 0,
        "normalization_candidate_vector_evaluations": 0,
        "normalization_selected_steps": 0,
        "normalization_vector_ring_multiplications": 0,
        "scale_ring_multiplications": 0,
        "output_ring_multiplications": 0,
        "basis_forward_block_applications": 0,
        "basis_inverse_block_applications": 0,
        "basis_ring_multiply_accumulations": 0,
        "maximum_carrier_payload_bits": 0,
        "maximum_reduced_coefficient_signed_bits": 1,
        "maximum_ledger_exponent_signed_bits": 1,
        "maximum_nonzero_message_slots": 0,
        "maximum_nonzero_message_ledgers": 0,
        "maximum_nonzero_coefficient_registers": 0,
        "maximum_transient_raw_message_payload_bits": 0,
        "maximum_transient_raw_message_signed_bits": 1,
    }


def merge_normalization_stats(
    stats: dict[str, int],
    update: dict[str, int],
) -> None:
    for key, value in update.items():
        stats[key] += value


def ring_power_with_count(
    base: RingElement,
    exponent: int,
) -> tuple[RingElement, int]:
    result = ring_one()
    factor = base
    count = 0
    while exponent:
        if exponent & 1:
            result = ring_multiply(result, factor)
            count += 1
        exponent >>= 1
        if exponent:
            factor = ring_multiply(factor, factor)
            count += 1
    return result, count


def ledger_scale_with_stats(
    ledger: list[int],
    stats: dict[str, int],
) -> RingElement:
    result = ring_one()
    for exponent, generator, inverse in zip(
        ledger,
        UNIT_GENERATORS,
        UNIT_GENERATOR_INVERSES,
        strict=True,
    ):
        factor, count = ring_power_with_count(
            generator if exponent >= 0 else inverse,
            abs(exponent),
        )
        result = ring_multiply(result, factor)
        stats["scale_ring_multiplications"] += count + 1
    return result


def unit_carrier() -> dict[str, Any]:
    return {
        "messages": [
            [ring_zero() for _ in range(PRIME)]
            for _ in range(MESSAGE_SLOTS)
        ],
        "message_ledgers": [
            [0 for _ in range(UNIT_RANK)]
            for _ in range(MESSAGE_SLOTS)
        ],
        "coefficients": [ring_zero() for _ in range(DIMENSION)],
        "coefficient_ledger": [0 for _ in range(UNIT_RANK)],
        "generation": 0,
        "lease": 0,
        "active": False,
    }


def unit_carrier_is_zero(carrier: dict[str, Any]) -> bool:
    return (
        all(
            element == ring_zero()
            for message in carrier["messages"]
            for element in message
        )
        and all(
            not any(ledger)
            for ledger in carrier["message_ledgers"]
        )
        and all(
            value == ring_zero()
            for value in carrier["coefficients"]
        )
        and not any(carrier["coefficient_ledger"])
    )


def unit_backing_identity(carrier: dict[str, Any]) -> tuple[int, ...]:
    return (
        id(carrier["messages"]),
        *(id(message) for message in carrier["messages"]),
        id(carrier["message_ledgers"]),
        *(id(ledger) for ledger in carrier["message_ledgers"]),
        id(carrier["coefficients"]),
        id(carrier["coefficient_ledger"]),
    )


def observe_unit_transient(
    vector: RingVector,
    stats: dict[str, int],
) -> None:
    stats["maximum_transient_raw_message_payload_bits"] = max(
        stats["maximum_transient_raw_message_payload_bits"],
        vector_payload_bits(vector),
    )
    stats["maximum_transient_raw_message_signed_bits"] = max(
        stats["maximum_transient_raw_message_signed_bits"],
        vector_maximum_signed_bits(vector),
    )


def record_unit_peak(
    carrier: dict[str, Any],
    stats: dict[str, int],
) -> None:
    payload = 0
    maximum_reduced = 1
    maximum_ledger = 1
    nonzero_messages = 0
    nonzero_message_ledgers = 0
    nonzero_coefficients = 0
    for message, ledger in zip(
        carrier["messages"],
        carrier["message_ledgers"],
        strict=True,
    ):
        nonzero_messages += int(
            any(element != ring_zero() for element in message)
        )
        nonzero_message_ledgers += int(any(ledger))
        payload += vector_payload_bits(message)
        maximum_reduced = max(
            maximum_reduced,
            vector_maximum_signed_bits(message),
        )
        payload += ledger_payload_bits(ledger)
        maximum_ledger = max(
            maximum_ledger,
            *(signed_bits(value) for value in ledger),
        )
    for element in carrier["coefficients"]:
        nonzero_coefficients += int(element != ring_zero())
        for coefficient in element:
            bits = signed_bits(coefficient)
            payload += bits
            maximum_reduced = max(maximum_reduced, bits)
    payload += ledger_payload_bits(carrier["coefficient_ledger"])
    maximum_ledger = max(
        maximum_ledger,
        *(
            signed_bits(value)
            for value in carrier["coefficient_ledger"]
        ),
    )
    stats["maximum_carrier_payload_bits"] = max(
        stats["maximum_carrier_payload_bits"],
        payload,
    )
    stats["maximum_reduced_coefficient_signed_bits"] = max(
        stats["maximum_reduced_coefficient_signed_bits"],
        maximum_reduced,
    )
    stats["maximum_ledger_exponent_signed_bits"] = max(
        stats["maximum_ledger_exponent_signed_bits"],
        maximum_ledger,
    )
    stats["maximum_nonzero_message_slots"] = max(
        stats["maximum_nonzero_message_slots"],
        nonzero_messages,
    )
    stats["maximum_nonzero_message_ledgers"] = max(
        stats["maximum_nonzero_message_ledgers"],
        nonzero_message_ledgers,
    )
    stats["maximum_nonzero_coefficient_registers"] = max(
        stats["maximum_nonzero_coefficient_registers"],
        nonzero_coefficients,
    )


def independently_normalize(
    vector: RingVector,
    ledger: list[int],
    stats: dict[str, int],
) -> tuple[RingVector, list[int]]:
    reduced, reduced_ledger, update = normalized_vector(vector, ledger)
    merge_normalization_stats(stats, update)
    return reduced, reduced_ledger


def unit_raw_output(
    carrier: dict[str, Any],
    stats: dict[str, int],
) -> RingVector:
    target = [ring_zero() for _ in range(PRIME)]
    coefficient_scale = ledger_scale_with_stats(
        carrier["coefficient_ledger"],
        stats,
    )
    for basis_index, coefficient in enumerate(
        carrier["coefficients"],
        start=1,
    ):
        basis_scale = ledger_scale_with_stats(
            carrier["message_ledgers"][basis_index],
            stats,
        )
        effective = ring_multiply(
            ring_multiply(coefficient_scale, coefficient),
            basis_scale,
        )
        stats["output_ring_multiplications"] += 2
        for phase_index, value in enumerate(
            carrier["messages"][basis_index]
        ):
            target[phase_index] = ring_add(
                target[phase_index],
                ring_multiply(effective, value),
            )
            stats["output_ring_multiplications"] += 1
    observe_unit_transient(target, stats)
    return target


def unit_semantic_boundary(
    vector: RingVector,
    ledger: list[int],
    stats: dict[str, int],
) -> RingElement:
    semantic = multiply_vector_by_scalar(
        ledger_scale_with_stats(ledger, stats),
        vector,
    )
    stats["scale_ring_multiplications"] += len(vector)
    return project(semantic)


def unit_execute_on_carrier(
    carrier: dict[str, Any],
    operator: RingMatrix,
    seed: RingVector,
    characteristic: list[RingElement],
    periods: int,
) -> tuple[RingElement, dict[str, int], dict[str, Any]]:
    if carrier["active"] or not unit_carrier_is_zero(carrier):
        fail("unit oracle carrier was not restored")
    backing = unit_backing_identity(carrier)
    stats = empty_unit_stats()
    carrier["active"] = True
    carrier["lease"] += 1
    messages = carrier["messages"]
    ledgers = carrier["message_ledgers"]
    messages[0][:] = seed
    for index in range(1, DIMENSION + 1):
        raw = matrix_vector_multiply(operator, messages[index - 1])
        stats["basis_forward_block_applications"] += 1
        stats["basis_ring_multiply_accumulations"] += PRIME * PRIME
        observe_unit_transient(raw, stats)
        reduced, ledger = independently_normalize(
            raw,
            ledgers[index - 1],
            stats,
        )
        messages[index][:] = reduced
        ledgers[index][:] = ledger
    record_unit_peak(carrier, stats)

    raw_coefficients = sequential_coefficients(periods, characteristic)
    reduced_coefficients, coefficient_ledger = independently_normalize(
        raw_coefficients,
        [0 for _ in range(UNIT_RANK)],
        stats,
    )
    carrier["coefficients"][:] = reduced_coefficients
    carrier["coefficient_ledger"][:] = coefficient_ledger
    raw_output = unit_raw_output(carrier, stats)
    reduced_output, output_ledger = independently_normalize(
        raw_output,
        [0 for _ in range(UNIT_RANK)],
        stats,
    )
    messages[OUTPUT_SLOT][:] = reduced_output
    ledgers[OUTPUT_SLOT][:] = output_ledger
    record_unit_peak(carrier, stats)
    boundary = unit_semantic_boundary(
        reduced_output,
        output_ledger,
        stats,
    )

    raw_expected_output = unit_raw_output(carrier, stats)
    expected_output, expected_output_ledger = independently_normalize(
        raw_expected_output,
        [0 for _ in range(UNIT_RANK)],
        stats,
    )
    if (
        messages[OUTPUT_SLOT] != expected_output
        or ledgers[OUTPUT_SLOT] != expected_output_ledger
    ):
        fail("unit oracle output inverse mismatch")
    messages[OUTPUT_SLOT][:] = [
        ring_subtract(value, expected)
        for value, expected in zip(
            messages[OUTPUT_SLOT],
            expected_output,
            strict=True,
        )
    ]
    ledgers[OUTPUT_SLOT][:] = [
        value - expected
        for value, expected in zip(
            ledgers[OUTPUT_SLOT],
            expected_output_ledger,
            strict=True,
        )
    ]

    expected_raw_coefficients = sequential_coefficients(
        periods,
        characteristic,
    )
    expected_coefficients, expected_coefficient_ledger = (
        independently_normalize(
            expected_raw_coefficients,
            [0 for _ in range(UNIT_RANK)],
            stats,
        )
    )
    if (
        carrier["coefficients"] != expected_coefficients
        or carrier["coefficient_ledger"]
        != expected_coefficient_ledger
    ):
        fail("unit oracle coefficient inverse mismatch")
    carrier["coefficients"][:] = [
        ring_subtract(value, expected)
        for value, expected in zip(
            carrier["coefficients"],
            expected_coefficients,
            strict=True,
        )
    ]
    carrier["coefficient_ledger"][:] = [
        value - expected
        for value, expected in zip(
            carrier["coefficient_ledger"],
            expected_coefficient_ledger,
            strict=True,
        )
    ]
    for index in range(DIMENSION, 0, -1):
        raw = matrix_vector_multiply(operator, messages[index - 1])
        stats["basis_inverse_block_applications"] += 1
        stats["basis_ring_multiply_accumulations"] += PRIME * PRIME
        observe_unit_transient(raw, stats)
        expected, expected_ledger = independently_normalize(
            raw,
            ledgers[index - 1],
            stats,
        )
        if ledgers[index] != expected_ledger:
            fail("unit oracle basis ledger inverse mismatch")
        messages[index][:] = [
            ring_subtract(value, expected_value)
            for value, expected_value in zip(
                messages[index],
                expected,
                strict=True,
            )
        ]
        ledgers[index][:] = [
            value - expected_value
            for value, expected_value in zip(
                ledgers[index],
                expected_ledger,
                strict=True,
            )
        ]
    messages[0][:] = [
        ring_subtract(value, expected)
        for value, expected in zip(messages[0], seed, strict=True)
    ]
    carrier["active"] = False
    carrier["generation"] += 1
    record_unit_peak(carrier, stats)
    return boundary, stats, {
        "restored_exactly": unit_carrier_is_zero(carrier),
        "same_backing": unit_backing_identity(carrier) == backing,
        "generation": carrier["generation"],
        "lease": carrier["lease"],
    }


def unnormalized_resource_tuple(
    seed: RingVector,
    operator: RingMatrix,
    characteristic: list[RingElement],
    periods: int,
) -> tuple[RingElement, dict[str, int]]:
    basis = build_basis(operator, seed)
    coefficients = sequential_coefficients(periods, characteristic)
    output = linear_combination(coefficients, basis)
    return project(output), recurrence_carrier_metrics(
        seed,
        basis,
        coefficients,
        output,
    )


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_chain_period17_"
            "unit_height_reduction_oracle.py PRODUCTION_RESULT"
        )
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        production_text = handle.read()
    production_result_file_bytes = len(
        production_text.encode("utf-8")
    )
    production = json.loads(production_text)
    del production_text

    compiled: dict[
        str,
        tuple[
            RingMatrix,
            list[RingElement],
            RingVector,
        ],
    ] = {}
    family_checks: dict[str, dict[str, bool]] = {}
    for family in ("primary", "reuse"):
        descriptor = compile_descriptor(PERIOD + 1, family.upper())
        operator = compile_operator(descriptor)
        certificate = production["block_certificates"][family]
        characteristic = [
            tuple(element)
            for element in certificate["characteristic"]
        ]
        seed = seed_vector(descriptor)
        compiled[family] = (operator, characteristic, seed)
        family_checks[family] = {
            "descriptor_sha256_equal": (
                hashlib.sha256(encoded(descriptor)).hexdigest()
                == certificate["public_program_sha256"]
            ),
            "operator_sha256_equal": (
                hashlib.sha256(encoded(operator)).hexdigest()
                == certificate["operator_sha256"]
            ),
            "characteristic_sha256_equal": (
                hashlib.sha256(encoded(characteristic)).hexdigest()
                == certificate["characteristic_sha256"]
            ),
            "whole_operator_annihilator_identity_exact": (
                check_annihilator(operator, characteristic)
            ),
            "whole_characteristic_constant_zero": (
                characteristic[-1] == ring_zero()
            ),
        }

    case_checks: list[dict[str, Any]] = []
    for production_case in production["cases"]:
        family = production_case["family"].lower()
        periods = production_case["periods"]
        operator, characteristic, seed = compiled[family]
        boundary, stats, witness = unit_execute_on_carrier(
            unit_carrier(),
            operator,
            seed,
            characteristic,
            periods,
        )
        raw_boundary, raw_metrics = unnormalized_resource_tuple(
            seed,
            operator,
            characteristic,
            periods,
        )
        dense_boundary = None
        if periods <= 64:
            dense_boundary, _ = direct_boundary_with_metrics(
                operator,
                seed,
                periods,
            )
        production_raw = production_case[
            "unnormalized_recurrence_stats"
        ]
        check = {
            "family": production_case["family"],
            "periods": periods,
            "boundary_equal": (
                list(boundary) == production_case["boundary"]
            ),
            "boundary_sha256_equal": (
                hashlib.sha256(encoded(boundary)).hexdigest()
                == production_case["boundary_sha256"]
            ),
            "normalized_stats_exact": (
                stats == production_case["stats"]
            ),
            "raw_boundary_equal": raw_boundary == boundary,
            "raw_carrier_payload_bits_equal": (
                raw_metrics["maximum_carrier_payload_bits"]
                == production_raw["maximum_carrier_payload_bits"]
            ),
            "raw_coefficient_width_equal": (
                raw_metrics["maximum_coefficient_signed_bits"]
                == production_raw["maximum_coefficient_signed_bits"]
            ),
            "raw_nonzero_message_slots_equal": (
                raw_metrics["maximum_nonzero_message_slots"]
                == production_raw["maximum_nonzero_message_slots"]
            ),
            "raw_nonzero_coefficient_registers_equal": (
                raw_metrics[
                    "maximum_nonzero_coefficient_registers"
                ]
                == production_raw[
                    "maximum_nonzero_coefficient_registers"
                ]
            ),
            "payload_reduction_bits_equal": (
                raw_metrics["maximum_carrier_payload_bits"]
                - stats["maximum_carrier_payload_bits"]
                == production_case["carrier_payload_reduction_bits"]
            ),
            "dense_boundary_equal_when_applicable": (
                dense_boundary == boundary
                if dense_boundary is not None
                else not production_case["dense_direct_applicable"]
            ),
            "restored_exactly": witness["restored_exactly"],
            "same_backing": witness["same_backing"],
            "production_restored_exactly": (
                production_case["restored_exactly"]
            ),
            "production_same_backing": production_case["same_backing"],
            "exact_resource_tuple": {
                "normalized_carrier_payload_bits": (
                    stats["maximum_carrier_payload_bits"]
                ),
                "normalized_coefficient_signed_bits": (
                    stats[
                        "maximum_reduced_coefficient_signed_bits"
                    ]
                ),
                "ledger_exponent_signed_bits": (
                    stats[
                        "maximum_ledger_exponent_signed_bits"
                    ]
                ),
                "raw_carrier_payload_bits": (
                    raw_metrics["maximum_carrier_payload_bits"]
                ),
                "raw_coefficient_signed_bits": (
                    raw_metrics["maximum_coefficient_signed_bits"]
                ),
                "normalization_calls": stats["normalization_calls"],
                "normalization_step_cap_hits": (
                    stats["normalization_step_cap_hits"]
                ),
                "normalization_candidate_vector_evaluations": (
                    stats[
                        "normalization_candidate_vector_evaluations"
                    ]
                ),
                "normalization_selected_steps": (
                    stats["normalization_selected_steps"]
                ),
            },
        }
        check["all_checks"] = all(
            value
            for key, value in check.items()
            if key not in {
                "family",
                "periods",
                "exact_resource_tuple",
                "all_checks",
            }
        )
        case_checks.append(check)

    shared = unit_carrier()
    shared_backing = unit_backing_identity(shared)
    primary_operator, primary_characteristic, primary_seed = (
        compiled["primary"]
    )
    reuse_operator, reuse_characteristic, reuse_seed = compiled["reuse"]
    primary_boundary, _, primary_witness = unit_execute_on_carrier(
        shared,
        primary_operator,
        primary_seed,
        primary_characteristic,
        max(EXPECTED_PERIODS),
    )
    reuse_boundary, _, reuse_witness = unit_execute_on_carrier(
        shared,
        reuse_operator,
        reuse_seed,
        reuse_characteristic,
        max(EXPECTED_PERIODS),
    )
    fresh_boundary, _, fresh_witness = unit_execute_on_carrier(
        unit_carrier(),
        reuse_operator,
        reuse_seed,
        reuse_characteristic,
        max(EXPECTED_PERIODS),
    )
    production_reuse = production["restoration_reuse_case"]
    reuse_checks = {
        "primary_boundary_sha256_equal": (
            hashlib.sha256(encoded(primary_boundary)).hexdigest()
            == next(
                case["boundary_sha256"]
                for case in production["cases"]
                if case["family"] == "PRIMARY"
                and case["periods"] == max(EXPECTED_PERIODS)
            )
        ),
        "reuse_boundary_sha256_equal": (
            hashlib.sha256(encoded(reuse_boundary)).hexdigest()
            == next(
                case["boundary_sha256"]
                for case in production["cases"]
                if case["family"] == "REUSE"
                and case["periods"] == max(EXPECTED_PERIODS)
            )
        ),
        "primary_restored_exactly": primary_witness["restored_exactly"],
        "reuse_restored_exactly": reuse_witness["restored_exactly"],
        "fresh_reuse_restored_exactly": (
            fresh_witness["restored_exactly"]
        ),
        "fresh_restored_reuse_boundary_equal": (
            reuse_boundary == fresh_boundary
        ),
        "same_original_backing": (
            unit_backing_identity(shared) == shared_backing
            and primary_witness["same_backing"]
            and reuse_witness["same_backing"]
        ),
        "generation_equal": (
            shared["generation"] == production_reuse["generation"] == 2
        ),
        "lease_equal": (
            shared["lease"] == production_reuse["lease"] == 2
        ),
        "all_state_zero": unit_carrier_is_zero(shared),
    }
    all_family_checks = all(
        all(checks.values())
        for checks in family_checks.values()
    )
    all_case_checks = all(
        check["all_checks"]
        for check in case_checks
    )
    all_reuse_checks = all(reuse_checks.values())
    result = {
        "result": "PASS",
        "oracle": (
            "SEPARATE_PUBLIC_DESCRIPTOR_TUPLE_Z_ZETA17_"
            "SEQUENTIAL_X_MOD_Q_SEVEN_GENERATOR_UNIT_NORMALIZATION_"
            "EXACT_RESOURCE_TUPLE_AND_RESTORATION_REEXECUTION"
        ),
        "classification_candidate": (
            "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
        ),
        "verification_level_candidate": (
            "SEPARATE_REFERENCE_PARITY"
        ),
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "production_module_imported": False,
        "production_compiler_called": False,
        "production_normalizer_called": False,
        "production_recurrence_called": False,
        "production_inverse_called": False,
        "unit_generator_inverse_identities_exact": (
            unit_identities_exact()
        ),
        "family_checks": family_checks,
        "case_checks": case_checks,
        "restoration_reuse_checks": reuse_checks,
        "all_family_checks": all_family_checks,
        "all_case_checks": all_case_checks,
        "all_restoration_reuse_checks": all_reuse_checks,
        "independent_coefficient_method": (
            "SEQUENTIAL_MULTIPLICATION_BY_X_MOD_MONIC_Q16"
        ),
        "independent_exact_unit_search_reexecution": True,
        "independent_exact_payload_and_width_reexecution": True,
        "unit_generator_multiplicative_independence_certified": False,
        "fixed_slots_are_not_fixed_total_footprint": True,
        "strict_local_minimum_for_every_call_established": False,
        "identical_compact_classical_normalizer": True,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "failure_atomic_restoration_established": False,
        "resource_law": {
            "production_result_file_bytes": (
                production_result_file_bytes
            ),
            "compiled_operator_integer_cells": (
                2 * OPERATOR_INTEGER_CELLS
            ),
            "parsed_characteristic_integer_cells": (
                2 * (PRIME + 1) * DIMENSION
            ),
            "oracle_unit_and_inverse_integer_cells": (
                2 * UNIT_RANK * DIMENSION
            ),
            "oracle_reexecutes_each_case_separately": True,
            "named_logical_cell_accounting_is_exact_total": False,
            "python_object_overhead_bounded": False,
            "allocator_peak_bounded": False,
            "whole_process_peak_bounded": False,
        },
        "terminal": False,
    }
    if (
        not result["unit_generator_inverse_identities_exact"]
        or not all_family_checks
        or not all_case_checks
        or not all_reuse_checks
    ):
        fail("independent unit-height certificate mismatch")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
