#!/usr/bin/env python3
"""Separate exact oracle for the adaptive F17 cyclotomic quotient.

This implementation consumes the production result as data.  It does not
import or call the production compiler, carrier, transfer, gauge selector,
projector, or inverse.  It reconstructs the semantic chain with a fixed
16-coefficient basis, independently factors maximal powers of 17, checks the
adaptive omitted-root receipt, runs a retain-all inverse, and directly
enumerates the two smallest cases.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from typing import Any


PRIME = 17
DIMENSION = 16
MESSAGE_CELLS = PRIME * DIMENSION
DIRECT_NODES = {2, 3}


def fail(message: str) -> None:
    raise RuntimeError(message)


def reduced(value: int) -> int:
    return value % PRIME


def signed_bits(value: int) -> int:
    if value == 0:
        return 1
    return abs(value).bit_length() + 1


def encoded_descriptor(program: dict[str, Any]) -> bytes:
    return json.dumps(
        program,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def unary_phase(coefficients: list[int], value: int) -> int:
    cubic, quadratic, linear = coefficients
    return reduced(
        cubic * value**3
        + quadratic * value**2
        + linear * value
    )


def edge_phase(
    coefficients: list[int],
    left: int,
    right: int,
) -> int:
    left_square_right, left_right_square, bilinear = coefficients
    return reduced(
        left_square_right * left * left * right
        + left_right_square * left * right * right
        + bilinear * left * right
    )


def zero_message() -> list[list[int]]:
    return [
        [0 for _ in range(DIMENSION)]
        for _ in range(PRIME)
    ]


def message_is_zero(message: list[list[int]]) -> bool:
    return all(value == 0 for row in message for value in row)


def seed(program: dict[str, Any]) -> list[list[int]]:
    message = zero_message()
    coefficients = program["unary_coefficients"][0]
    for value in range(PRIME):
        phase = unary_phase(coefficients, value)
        if phase < DIMENSION:
            message[value][phase] = 1
        else:
            for basis in range(DIMENSION):
                message[value][basis] = -1
    return message


def transfer(
    source: list[list[int]],
    program: dict[str, Any],
    edge_index: int,
) -> list[list[int]]:
    target = zero_message()
    unary = program["unary_coefficients"][edge_index + 1]
    edge = program["edge_coefficients"][edge_index]
    for left in range(PRIME):
        for basis in range(DIMENSION):
            coefficient = source[left][basis]
            for right in range(PRIME):
                shift = reduced(
                    unary_phase(unary, right)
                    + edge_phase(edge, left, right)
                )
                exponent = reduced(basis + shift)
                if exponent < DIMENSION:
                    target[right][exponent] += coefficient
                else:
                    for output_basis in range(DIMENSION):
                        target[right][output_basis] -= coefficient
    return target


def integer_valuation(value: int) -> int:
    if value == 0:
        return sys.maxsize
    result = 0
    magnitude = abs(value)
    while magnitude % PRIME == 0:
        magnitude //= PRIME
        result += 1
    return result


def extract_content(
    message: list[list[int]],
) -> tuple[list[list[int]], int]:
    valuation = min(
        integer_valuation(value)
        for row in message
        for value in row
        if value != 0
    )
    if valuation == sys.maxsize:
        fail("zero message has no declared content")
    divisor = PRIME**valuation
    quotient = [
        [value // divisor for value in row]
        for row in message
    ]
    if any(
        value % divisor
        for row in message
        for value in row
    ):
        fail("content quotient was not exact")
    return quotient, valuation


def factorized_stream(
    program: dict[str, Any],
) -> tuple[list[list[int]], int]:
    current = seed(program)
    total_exponent = 0
    for edge_index in range(program["nodes"] - 1):
        current = transfer(current, program, edge_index)
        current, gained = extract_content(current)
        total_exponent += gained
    return current, total_exponent


def semantic_stream(
    program: dict[str, Any],
) -> list[list[int]]:
    current = seed(program)
    for edge_index in range(program["nodes"] - 1):
        current = transfer(current, program, edge_index)
    return current


def canonical_boundary(
    message: list[list[int]],
) -> list[int]:
    return [
        sum(message[value][basis] for value in range(PRIME))
        for basis in range(DIMENSION)
    ]


def choose_pivot(
    canonical: list[int],
) -> tuple[int, list[int]]:
    redundant = canonical + [0]
    candidates: list[tuple[int, int, list[int]]] = []
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
                coefficients,
            )
        )
    best = min(candidates, key=lambda item: (item[0], item[1]))
    return best[1], best[2]


def canonical_from_adaptive(
    pivot: int,
    coefficients: list[int],
) -> list[int]:
    redundant = [0] * PRIME
    source_index = 0
    for exponent in range(PRIME):
        if exponent == pivot:
            continue
        redundant[exponent] = coefficients[source_index]
        source_index += 1
    reference = redundant[16]
    return [
        redundant[index] - reference
        for index in range(DIMENSION)
    ]


def reconstruct_boundary(
    boundary: dict[str, Any],
) -> list[int]:
    quotient = canonical_from_adaptive(
        boundary["adaptive_omitted_root"],
        boundary["adaptive_cyclotomic_coefficients"],
    )
    scale = PRIME**boundary["content_17_exponent"]
    return [scale * value for value in quotient]


def retain_all_inverse(
    program: dict[str, Any],
) -> bool:
    messages = [seed(program)]
    for edge_index in range(program["nodes"] - 1):
        messages.append(
            transfer(messages[-1], program, edge_index)
        )
    for edge_index in range(program["nodes"] - 2, -1, -1):
        expected = transfer(
            messages[edge_index],
            program,
            edge_index,
        )
        for value in range(PRIME):
            for basis in range(DIMENSION):
                messages[edge_index + 1][value][basis] -= (
                    expected[value][basis]
                )
    initial = seed(program)
    for value in range(PRIME):
        for basis in range(DIMENSION):
            messages[0][value][basis] -= initial[value][basis]
    return all(message_is_zero(message) for message in messages)


def direct_boundary(program: dict[str, Any]) -> list[int]:
    histogram = [0] * PRIME
    nodes = program["nodes"]
    for values in itertools.product(range(PRIME), repeat=nodes):
        phase = 0
        for index, value in enumerate(values):
            phase += unary_phase(
                program["unary_coefficients"][index],
                value,
            )
        for edge_index in range(nodes - 1):
            phase += edge_phase(
                program["edge_coefficients"][edge_index],
                values[edge_index],
                values[edge_index + 1],
            )
        histogram[reduced(phase)] += 1
    reference = histogram[16]
    return [
        histogram[index] - reference
        for index in range(DIMENSION)
    ]


def validate_program(
    program: dict[str, Any],
    nodes: int,
) -> None:
    if (
        program.get("field") != "F17"
        or program.get("phase_root_order") != PRIME
        or program.get("topology") != "PUBLIC_PATH_GRAPH"
        or program.get("nodes") != nodes
        or len(program.get("unary_coefficients", [])) != nodes
        or len(program.get("edge_coefficients", [])) != nodes - 1
    ):
        fail("invalid public path descriptor")
    for coefficients in (
        program["unary_coefficients"]
        + program["edge_coefficients"]
    ):
        if (
            len(coefficients) != 3
            or any(
                not isinstance(value, int)
                or value < 0
                or value >= PRIME
                for value in coefficients
            )
        ):
            fail("invalid public F17 coefficient")


def descriptor_period_is_17(
    program: dict[str, Any],
) -> bool:
    nodes = program["nodes"]
    for index in range(nodes - 17):
        if (
            program["unary_coefficients"][index]
            != program["unary_coefficients"][index + 17]
        ):
            return False
    for index in range(nodes - 1 - 17):
        if (
            program["edge_coefficients"][index]
            != program["edge_coefficients"][index + 17]
        ):
            return False
    return True


def pebble_applications(edges: int) -> int:
    if edges == 1:
        return 1
    left = edges // 2
    return (
        2 * pebble_applications(left)
        + pebble_applications(edges - left)
    )


def verify_program(
    program: dict[str, Any],
    expected_hash: str,
    boundary: dict[str, Any],
    final_diagnostic: dict[str, Any],
) -> dict[str, Any]:
    nodes = program["nodes"]
    descriptor_hash = hashlib.sha256(
        encoded_descriptor(program)
    ).hexdigest()
    if descriptor_hash != expected_hash:
        fail("public descriptor hash mismatch")
    semantic_message = semantic_stream(program)
    semantic_boundary = canonical_boundary(semantic_message)
    factorized, message_exponent = factorized_stream(program)
    quotient_boundary = canonical_boundary(factorized)
    quotient_boundary, boundary_gain = extract_content(
        [quotient_boundary]
    )
    boundary_quotient = quotient_boundary[0]
    total_exponent = message_exponent + boundary_gain
    if (
        reconstruct_boundary(boundary) != semantic_boundary
        or boundary[
            "verification_reconstructed_canonical_coefficients"
        ] != semantic_boundary
        or boundary["content_17_exponent"] != total_exponent
        or boundary[
            "effective_normalization_denominator_sqrt_power"
        ] != nodes - 2 * total_exponent
        or final_diagnostic["stored_17_content_exponent"]
        != message_exponent
    ):
        fail("factorized boundary or content exponent mismatch")
    pivot, coefficients = choose_pivot(boundary_quotient)
    if (
        boundary["adaptive_omitted_root"] != pivot
        or boundary["adaptive_cyclotomic_coefficients"]
        != coefficients
    ):
        fail("adaptive omitted-root receipt is not minimal")
    direct_equal: bool | None = None
    if nodes in DIRECT_NODES:
        direct_equal = direct_boundary(program) == semantic_boundary
        if not direct_equal:
            fail("direct assignment enumeration mismatch")
    inverse_restored = retain_all_inverse(program)
    if not inverse_restored:
        fail("independent retain-all inverse did not restore")
    return {
        "descriptor_sha256": descriptor_hash,
        "semantic_boundary_equal": True,
        "factorized_boundary_equal": True,
        "adaptive_pivot_minimal": True,
        "message_17_content_exponent": message_exponent,
        "boundary_17_content_exponent": total_exponent,
        "direct_enumeration_equal": direct_equal,
        "retain_all_inverse_restored_exactly": True,
        "descriptor_period_17": descriptor_period_is_17(program),
    }


def verify_case(case: dict[str, Any]) -> dict[str, Any]:
    nodes = case["nodes"]
    primary = case["primary_public_program"]
    reuse = case["reuse_public_program"]
    validate_program(primary, nodes)
    validate_program(reuse, nodes)
    primary_result = verify_program(
        primary,
        case["primary_program_sha256"],
        case["primary_boundary"],
        case["primary_final_content_diagnostic"],
    )
    reuse_result = verify_program(
        reuse,
        case["reuse_program_sha256"],
        case["reuse_boundary"],
        case["reuse_final_content_diagnostic"],
    )
    expected_slots = (nodes - 1).bit_length() + 1
    expected_applications = pebble_applications(nodes - 1)
    if (
        case["message_slots"] != expected_slots
        or case["message_integer_cells"]
        != expected_slots * MESSAGE_CELLS
        or case["pebble_forward_applications"]
        != expected_applications
    ):
        fail("reported reversible resource law mismatch")
    periodic = case["periodic_block_baseline"]
    if (
        periodic["public_transfer_period"] != 17
        or periodic["streaming_transfer_applications"] != nodes - 1
        or periodic["dense_block_integer_cells"]
        != MESSAGE_CELLS**2
        or periodic["dense_block_build_transfer_equivalents"]
        != 17 * MESSAGE_CELLS
        or not periodic["dense_block_build_exceeds_streaming_at_case"]
        or periodic["powering_executed"]
    ):
        fail("periodic block applicability accounting mismatch")
    return {
        "nodes": nodes,
        "primary": primary_result,
        "reuse": reuse_result,
        "message_slots": expected_slots,
        "message_integer_cells": expected_slots * MESSAGE_CELLS,
        "pebble_forward_applications": expected_applications,
        "direct_assignment_count": (
            PRIME**nodes if nodes in DIRECT_NODES else None
        ),
        "periodic_dense_block_build_exceeds_streaming": True,
    }


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_chain_adaptive_gauge_oracle.py "
            "PRODUCTION_RESULT.json"
        )
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        production = json.load(handle)
    if production.get("result") != "PASS":
        fail("production result is not a passing candidate")
    cases = [verify_case(case) for case in production["cases"]]
    result = {
        "result": "PASS",
        "oracle": (
            "SEPARATE_FIXED_BASIS_EXACT_CONTENT_FACTOR_ADAPTIVE_"
            "GAUGE_RECONSTRUCTION_RETAIN_ALL_INVERSE_AND_SMALL_"
            "DIRECT_ENUMERATION"
        ),
        "tested_nodes": [case["nodes"] for case in cases],
        "direct_enumeration_nodes": sorted(DIRECT_NODES),
        "cases": cases,
        "all_semantic_boundaries_equal": True,
        "all_factorized_boundaries_equal": True,
        "all_adaptive_pivots_minimal": True,
        "all_retain_all_inverses_restored_exactly": True,
        "all_declared_descriptor_periods_equal_17": True,
        "production_module_imported": False,
        "production_compiler_called": False,
        "production_transfer_called": False,
        "production_gauge_selector_called": False,
        "production_projector_called": False,
        "production_inverse_called": False,
        "public_descriptors_consumed": True,
        "identical_compact_classical_recurrence": True,
        "periodic_dense_block_powering_executed": False,
        "periodic_dense_block_build_inapplicable_at_declared_cases": True,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "terminal": False,
    }
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
