#!/usr/bin/env python3
"""Separate fixed-basis oracle for the period-17 F17 chain diagnostic.

This file does not import the production adaptive carrier or period-orbit
module.  It consumes public descriptors from the production result, uses an
opposite-pivot modular elimination, and independently reconstructs the exact
integer content quotient and final adaptive omitted-root chart.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from typing import Any


PRIME = 17
DIMENSION = 16
MESSAGE_CELLS = PRIME * DIMENSION
EXPECTED_PERIOD = 17
EXPECTED_MODULI = (41, 73)
EXPECTED_PERIODS = (1, 2, 4, 8)


def fail(message: str) -> None:
    raise RuntimeError(message)


def signed_bits(value: int) -> int:
    if value == 0:
        return 1
    return abs(value).bit_length() + 1


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


def encoded_descriptor(program: dict[str, Any]) -> bytes:
    return json.dumps(
        program,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def zero_message() -> list[list[int]]:
    return [
        [0 for _ in range(DIMENSION)]
        for _ in range(PRIME)
    ]


def seed(program: dict[str, Any]) -> list[list[int]]:
    message = zero_message()
    unary = program["unary_coefficients"][0]
    for value in range(PRIME):
        exponent = unary_phase(unary, value)
        if exponent < DIMENSION:
            message[value][exponent] = 1
        else:
            for basis in range(DIMENSION):
                message[value][basis] = -1
    return message


def transfer_integer(
    source: list[list[int]],
    program: dict[str, Any],
    edge_index: int,
) -> list[list[int]]:
    target = zero_message()
    unary = program["unary_coefficients"][edge_index + 1]
    edge = program["edge_coefficients"][edge_index]
    for right in range(PRIME):
        target_row = target[right]
        for left in range(PRIME):
            shift = (
                unary_phase(unary, right)
                + edge_phase(edge, left, right)
            ) % PRIME
            for basis, coefficient in enumerate(source[left]):
                if coefficient == 0:
                    continue
                exponent = (basis + shift) % PRIME
                if exponent < DIMENSION:
                    target_row[exponent] += coefficient
                else:
                    for output_basis in range(DIMENSION):
                        target_row[output_basis] -= coefficient
    return target


def extract_17_content(
    message: list[list[int]],
) -> tuple[list[list[int]], int]:
    gcd_value = 0
    for row in message:
        for value in row:
            gcd_value = math.gcd(gcd_value, abs(value))
    exponent = 0
    while gcd_value and gcd_value % PRIME == 0:
        gcd_value //= PRIME
        exponent += 1
    divisor = PRIME**exponent
    if any(
        value % divisor
        for row in message
        for value in row
    ):
        fail("oracle content divisor was not exact")
    return (
        [
            [value // divisor for value in row]
            for row in message
        ],
        exponent,
    )


def choose_gauge(row: list[int]) -> tuple[int, list[int]]:
    redundant = row + [0]
    candidates: list[tuple[int, int]] = []
    for pivot in range(PRIME):
        reference = redundant[pivot]
        coefficients = [
            redundant[index] - reference
            for index in range(PRIME)
            if index != pivot
        ]
        candidates.append(
            (
                sum(signed_bits(value) for value in coefficients),
                pivot,
            )
        )
    _, pivot = min(candidates)
    reference = redundant[pivot]
    return (
        pivot,
        [
            redundant[index] - reference
            for index in range(PRIME)
            if index != pivot
        ],
    )


def exact_projective_message(
    program: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    current = seed(program)
    total_exponent = 0
    for edge_index in range(program["nodes"] - 1):
        current = transfer_integer(current, program, edge_index)
        current, gained = extract_17_content(current)
        total_exponent += gained
    pivots: list[int] = []
    coefficients: list[list[int]] = []
    for row in current:
        pivot, adaptive_row = choose_gauge(row)
        pivots.append(pivot)
        coefficients.append(adaptive_row)
    quotient_gcd = 0
    for row in coefficients:
        for value in row:
            quotient_gcd = math.gcd(quotient_gcd, abs(value))
    scale = PRIME**total_exponent
    semantic = [
        [scale * value for value in row]
        for row in current
    ]
    encoded_message = json.dumps(
        {
            "pivots": pivots,
            "coefficients": coefficients,
            "scale_17_exponent": total_exponent,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        {
            "adaptive_message_sha256": hashlib.sha256(
                encoded_message
            ).hexdigest(),
            "adaptive_total_payload_bits": (
                PRIME * 5
                + max(1, total_exponent.bit_length())
                + sum(
                    signed_bits(value)
                    for row in coefficients
                    for value in row
                )
            ),
            "fixed_basis_semantic_payload_bits": sum(
                signed_bits(value)
                for row in semantic
                for value in row
            ),
            "maximum_quotient_coefficient_signed_bits": max(
                signed_bits(value)
                for row in coefficients
                for value in row
            ),
            "maximum_semantic_coefficient_signed_bits": max(
                signed_bits(value)
                for row in semantic
                for value in row
            ),
            "stored_17_content_exponent": total_exponent,
            "stored_exponent_payload_bits": max(
                1,
                total_exponent.bit_length(),
            ),
            "semantic_scale_integer_bits_if_materialized": (
                scale.bit_length()
            ),
            "quotient_coefficient_gcd": quotient_gcd,
            "residual_common_integer_content_removed": (
                quotient_gcd == 1
            ),
        },
        {
            "encoded_message_bytes": len(encoded_message),
            "final_fixed_adaptive_semantic_integer_cells": (
                3 * MESSAGE_CELLS
            ),
        },
    )


def flatten_seed(
    program: dict[str, Any],
    modulus: int,
) -> list[int]:
    return [
        value % modulus
        for row in seed(program)
        for value in row
    ]


def transfer_modular(
    source: list[int],
    program: dict[str, Any],
    edge_index: int,
    modulus: int,
) -> list[int]:
    rows = [
        source[index * DIMENSION:(index + 1) * DIMENSION]
        for index in range(PRIME)
    ]
    target = transfer_integer(rows, program, edge_index)
    return [
        value % modulus
        for row in target
        for value in row
    ]


def apply_block(
    source: list[int],
    program: dict[str, Any],
    modulus: int,
) -> list[int]:
    current = source
    for edge_index in range(EXPECTED_PERIOD):
        current = transfer_modular(
            current,
            program,
            edge_index,
            modulus,
        )
    return current


def reverse_pivot_krylov_dimension(
    program: dict[str, Any],
    modulus: int,
) -> int:
    basis: dict[int, list[int]] = {}
    current = flatten_seed(program, modulus)
    for step in range(MESSAGE_CELLS):
        work = current[:]
        for pivot in sorted(basis, reverse=True):
            factor = work[pivot]
            if factor:
                row = basis[pivot]
                for index in range(pivot + 1):
                    work[index] = (
                        work[index] - factor * row[index]
                    ) % modulus
        pivot = next(
            (
                index
                for index in range(MESSAGE_CELLS - 1, -1, -1)
                if work[index]
            ),
            None,
        )
        if pivot is None:
            return step
        inverse = pow(work[pivot], modulus - 2, modulus)
        for index in range(pivot + 1):
            work[index] = work[index] * inverse % modulus
        basis[pivot] = work
        current = apply_block(current, program, modulus)
    return MESSAGE_CELLS


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_chain_period17_krylov_oracle.py "
            "PRODUCTION_RESULT"
        )
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        production_text = handle.read()
    production_result_file_bytes = len(
        production_text.encode("utf-8")
    )
    production = json.loads(production_text)
    del production_text
    if production["period"] != EXPECTED_PERIOD:
        fail("unexpected public period")
    if tuple(production["krylov_moduli"]) != EXPECTED_MODULI:
        fail("unexpected modular fields")

    descriptor_checks: dict[str, Any] = {}
    rank_checks: dict[str, Any] = {}
    for family in ("primary", "reuse"):
        family_upper = family.upper()
        descriptor = compile_descriptor(
            EXPECTED_PERIOD + 1,
            family_upper,
        )
        descriptor_checks[family] = {
            "compiled_equal_public_descriptor": (
                descriptor == production["block_programs"][family]
            ),
            "descriptor_sha256_equal": (
                hashlib.sha256(
                    encoded_descriptor(descriptor)
                ).hexdigest()
                == production["block_program_sha256"][family]
            ),
        }
        rank_checks[family] = {
            str(modulus): reverse_pivot_krylov_dimension(
                descriptor,
                modulus,
            )
            for modulus in EXPECTED_MODULI
        }

    projective_checks: list[dict[str, Any]] = []
    projective_resource_observations: list[dict[str, int]] = []
    for case in production["projective_cases"]:
        periods = case["periods"]
        if periods not in EXPECTED_PERIODS:
            fail("unexpected projective period count")
        family_checks: dict[str, Any] = {}
        for family in ("primary", "reuse"):
            descriptor = compile_descriptor(
                periods * EXPECTED_PERIOD + 1,
                family.upper(),
            )
            descriptor_sha256 = hashlib.sha256(
                encoded_descriptor(descriptor)
            ).hexdigest()
            independent, resource_observation = (
                exact_projective_message(descriptor)
            )
            projective_resource_observations.append(
                resource_observation
            )
            expected = {
                key: case[family][key]
                for key in independent
            }
            family_checks[family] = {
                "descriptor_sha256_equal": (
                    descriptor_sha256
                    == case[family]["public_program_sha256"]
                ),
                "exact_projective_metrics_equal": (
                    independent == expected
                ),
                "independent": independent,
            }
        projective_checks.append(
            {
                "periods": periods,
                **family_checks,
            }
        )

    all_descriptors_equal = all(
        value["compiled_equal_public_descriptor"]
        and value["descriptor_sha256_equal"]
        for value in descriptor_checks.values()
    )
    rank_values_equal = all(
        rank_checks[family][str(modulus)]
        == production["krylov"][family][str(modulus)][
            "dimension"
        ]
        for family in ("primary", "reuse")
        for modulus in EXPECTED_MODULI
    )
    all_projective_equal = all(
        case[family]["descriptor_sha256_equal"]
        and case[family]["exact_projective_metrics_equal"]
        for case in projective_checks
        for family in ("primary", "reuse")
    )
    result = {
        "result": "PASS",
        "oracle": (
            "SEPARATE_PUBLIC_DESCRIPTOR_FIXED_BASIS_INTEGER_CONTENT_"
            "ADAPTIVE_GAUGE_AND_REVERSE_PIVOT_MODULAR_KRYLOV"
        ),
        "production_module_imported": False,
        "production_compiler_called": False,
        "production_transfer_called": False,
        "production_gauge_selector_called": False,
        "production_inverse_called": False,
        "descriptor_checks": descriptor_checks,
        "rank_checks": rank_checks,
        "projective_checks": projective_checks,
        "all_public_descriptors_equal": all_descriptors_equal,
        "all_modular_krylov_dimensions_equal": rank_values_equal,
        "all_exact_projective_metrics_equal": all_projective_equal,
        "exact_rational_krylov_dimensions_established": False,
        "identical_compact_classical_block_map": True,
        "resource_law": {
            "production_result_file_bytes": (
                production_result_file_bytes
            ),
            "modular_seed_field_cells": MESSAGE_CELLS,
            "modular_krylov_basis_peak_field_cells": (
                max(
                    rank_checks[family][str(modulus)]
                    for family in ("primary", "reuse")
                    for modulus in EXPECTED_MODULI
                )
                * MESSAGE_CELLS
            ),
            "modular_current_and_reduced_field_cells": (
                2 * MESSAGE_CELLS
            ),
            "modular_transfer_source_rows_target_and_flat_peak_field_cells": (
                4 * MESSAGE_CELLS
            ),
            "modular_combined_explicit_peak_field_cells": (
                max(
                    rank_checks[family][str(modulus)]
                    for family in ("primary", "reuse")
                    for modulus in EXPECTED_MODULI
                )
                * MESSAGE_CELLS
                + 4 * MESSAGE_CELLS
            ),
            "exact_projective_transfer_two_message_peak_integer_cells": (
                2 * MESSAGE_CELLS
            ),
            "exact_projective_final_fixed_adaptive_semantic_peak_integer_cells": (
                max(
                    observation[
                        "final_fixed_adaptive_semantic_integer_cells"
                    ]
                    for observation in projective_resource_observations
                )
            ),
            "maximum_exact_projective_encoded_message_bytes": (
                max(
                    observation["encoded_message_bytes"]
                    for observation in projective_resource_observations
                )
            ),
            "gauge_candidate_metadata_integer_fields": 2 * PRIME,
            "gauge_candidate_retained_coefficient_integer_cells": (
                2 * DIMENSION
            ),
            "gauge_candidate_redundant_row_integer_cells": PRIME,
            "python_object_overhead_bounded": False,
            "allocator_peak_bounded": False,
            "bit_operation_peak_bounded": False,
            "whole_process_peak_bounded": False,
        },
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "terminal": False,
    }
    if (
        not all_descriptors_equal
        or not rank_values_equal
        or not all_projective_equal
    ):
        fail("independent period-17 oracle mismatch")
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
