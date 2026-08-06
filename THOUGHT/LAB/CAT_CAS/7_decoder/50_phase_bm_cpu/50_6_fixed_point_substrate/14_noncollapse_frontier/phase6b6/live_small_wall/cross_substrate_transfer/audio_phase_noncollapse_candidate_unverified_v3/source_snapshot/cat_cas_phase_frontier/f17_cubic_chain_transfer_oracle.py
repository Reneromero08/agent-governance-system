#!/usr/bin/env python3
"""Separate exact oracle for the F17 cubic-chain transfer package.

The oracle consumes only the committed public descriptors and result payload.
It does not import the production package.  It uses a conventional two-message
dynamic program, a retain-all reversible parity construction, and direct
assignment enumeration at the two smallest declared depths.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from typing import Any


PRIME = 17
CYCLOTOMIC_DIMENSION = PRIME - 1
MESSAGE_CELLS = PRIME * CYCLOTOMIC_DIMENSION
ENUMERATION_NODES = {2, 3}


def fail(message: str) -> None:
    raise RuntimeError(message)


def reduced(value: int) -> int:
    return value % PRIME


def empty_message() -> list[list[int]]:
    return [
        [0 for _ in range(CYCLOTOMIC_DIMENSION)]
        for _ in range(PRIME)
    ]


def is_zero(message: list[list[int]]) -> bool:
    return all(value == 0 for row in message for value in row)


def unary_phase(coefficients: list[int], value: int) -> int:
    cubic, quadratic, linear = coefficients
    return reduced(
        cubic * value**3
        + quadratic * value**2
        + linear * value
    )


def interaction_phase(
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


def encoded_descriptor(program: dict[str, Any]) -> bytes:
    return json.dumps(
        program,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def seed(program: dict[str, Any]) -> list[list[int]]:
    message = empty_message()
    coefficients = program["unary_coefficients"][0]
    for value in range(PRIME):
        phase = unary_phase(coefficients, value)
        if phase < CYCLOTOMIC_DIMENSION:
            message[value][phase] += 1
        else:
            for basis in range(CYCLOTOMIC_DIMENSION):
                message[value][basis] -= 1
    return message


def transfer(
    source: list[list[int]],
    program: dict[str, Any],
    edge_index: int,
) -> list[list[int]]:
    target = empty_message()
    unary = program["unary_coefficients"][edge_index + 1]
    edge = program["edge_coefficients"][edge_index]
    for left in range(PRIME):
        for basis in range(CYCLOTOMIC_DIMENSION):
            coefficient = source[left][basis]
            for right in range(PRIME):
                shift = reduced(
                    unary_phase(unary, right)
                    + interaction_phase(edge, left, right)
                )
                exponent = reduced(basis + shift)
                if exponent < CYCLOTOMIC_DIMENSION:
                    target[right][exponent] += coefficient
                else:
                    for output_basis in range(CYCLOTOMIC_DIMENSION):
                        target[right][output_basis] -= coefficient
    return target


def add_transfer(
    source: list[list[int]],
    target: list[list[int]],
    program: dict[str, Any],
    edge_index: int,
    direction: int,
) -> None:
    expected = transfer(source, program, edge_index)
    for value in range(PRIME):
        for basis in range(CYCLOTOMIC_DIMENSION):
            target[value][basis] += (
                direction * expected[value][basis]
            )


def boundary(
    message: list[list[int]],
    nodes: int,
) -> dict[str, Any]:
    canonical = [0] * CYCLOTOMIC_DIMENSION
    for value in range(PRIME):
        for basis in range(CYCLOTOMIC_DIMENSION):
            canonical[basis] += message[value][basis]
    signed_bits = lambda value: (
        1 if value == 0 else abs(value).bit_length() + 1
    )
    return {
        "root_order": PRIME,
        "normalization_denominator_base": PRIME,
        "normalization_denominator_sqrt_power": nodes,
        "canonical_cyclotomic_coefficients": canonical,
        "canonical_nonzero_coefficients": sum(
            value != 0 for value in canonical
        ),
        "canonical_l1_coefficient_weight": sum(
            abs(value) for value in canonical
        ),
        "canonical_signed_bit_width": max(
            (signed_bits(value) for value in canonical),
            default=1,
        ),
        "final_canonical_message_integer_payload_bits": sum(
            signed_bits(value) for value in canonical
        ),
        "canonical_cyclotomic_integer_payload_bits": sum(
            signed_bits(value) for value in canonical
        ),
        "preprojection_value_cyclotomic_message_cells": MESSAGE_CELLS,
        "projected_boundary_cyclotomic_coefficients": (
            CYCLOTOMIC_DIMENSION
        ),
    }


def two_message_dynamic_program(
    program: dict[str, Any],
) -> dict[str, Any]:
    current = seed(program)
    for edge_index in range(program["nodes"] - 1):
        current = transfer(current, program, edge_index)
    return boundary(current, program["nodes"])


def retain_all_inverse_parity(
    program: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    messages = [seed(program)]
    for edge_index in range(program["nodes"] - 1):
        messages.append(
            transfer(messages[-1], program, edge_index)
        )
    final_boundary = boundary(messages[-1], program["nodes"])
    for edge_index in range(program["nodes"] - 2, -1, -1):
        add_transfer(
            messages[edge_index],
            messages[edge_index + 1],
            program,
            edge_index,
            -1,
        )
    for value in range(PRIME):
        phase = unary_phase(
            program["unary_coefficients"][0],
            value,
        )
        if phase < CYCLOTOMIC_DIMENSION:
            messages[0][value][phase] -= 1
        else:
            for basis in range(CYCLOTOMIC_DIMENSION):
                messages[0][value][basis] += 1
    return final_boundary, all(is_zero(message) for message in messages)


def direct_enumeration(
    program: dict[str, Any],
) -> dict[str, Any]:
    nodes = program["nodes"]
    histogram = [0] * PRIME
    for values in itertools.product(range(PRIME), repeat=nodes):
        phase = 0
        for index, value in enumerate(values):
            phase += unary_phase(
                program["unary_coefficients"][index],
                value,
            )
        for index in range(nodes - 1):
            phase += interaction_phase(
                program["edge_coefficients"][index],
                values[index],
                values[index + 1],
            )
        histogram[reduced(phase)] += 1
    message = empty_message()
    for phase, count in enumerate(histogram):
        if phase < CYCLOTOMIC_DIMENSION:
            message[0][phase] += count
        else:
            for basis in range(CYCLOTOMIC_DIMENSION):
                message[0][basis] -= count
    return boundary(message, nodes)


def pebble_applications(edges: int) -> int:
    if edges == 1:
        return 1
    left = edges // 2
    right = edges - left
    return (
        2 * pebble_applications(left)
        + pebble_applications(right)
    )


def validate_program(program: dict[str, Any], nodes: int) -> None:
    if (
        program.get("field") != "F17"
        or program.get("phase_root_order") != PRIME
        or program.get("topology") != "PUBLIC_PATH_GRAPH"
        or program.get("nodes") != nodes
        or len(program.get("unary_coefficients", [])) != nodes
        or len(program.get("edge_coefficients", [])) != nodes - 1
    ):
        fail("public chain descriptor has invalid topology")
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
            fail("public factor coefficient is outside F17")


def verify_case(case: dict[str, Any]) -> dict[str, Any]:
    nodes = case["nodes"]
    programs = [
        (
            "primary",
            case["public_primary_program"],
            case["primary_program_sha256"],
            case["primary_boundary"],
        ),
        (
            "reuse",
            case["public_reuse_program"],
            case["reuse_program_sha256"],
            case["reuse_boundary"],
        ),
    ]
    program_results: list[dict[str, Any]] = []
    for name, program, expected_hash, expected_boundary in programs:
        validate_program(program, nodes)
        descriptor_hash = hashlib.sha256(
            encoded_descriptor(program)
        ).hexdigest()
        if descriptor_hash != expected_hash:
            fail("public program descriptor hash mismatch")
        dynamic_boundary = two_message_dynamic_program(program)
        retained_boundary, restored = retain_all_inverse_parity(program)
        if (
            dynamic_boundary != expected_boundary
            or retained_boundary != expected_boundary
            or not restored
        ):
            fail("separate exact chain recurrence disagrees")
        enumeration_equal: bool | None = None
        if nodes in ENUMERATION_NODES:
            enumeration_equal = (
                direct_enumeration(program) == expected_boundary
            )
            if not enumeration_equal:
                fail("direct assignment enumeration disagrees")
        program_results.append(
            {
                "family": name,
                "descriptor_sha256": descriptor_hash,
                "two_message_boundary_equal": True,
                "retain_all_boundary_equal": True,
                "retain_all_inverse_restored_exactly": True,
                "direct_enumeration_boundary_equal": (
                    enumeration_equal
                ),
            }
        )
    expected_slots = (nodes - 1).bit_length() + 1
    if (nodes - 1) & (nodes - 2) == 0:
        expected_slots = (nodes - 1).bit_length() + 1
    if case["reversible_pebble_message_slots"] != expected_slots:
        fail("reported pebble slot law disagrees")
    expected_applications = pebble_applications(nodes - 1)
    if (
        case["pebble_forward_step_applications"]
        != expected_applications
        or case[
            "pebble_full_forward_inverse_step_applications"
        ] != 2 * expected_applications
    ):
        fail("reported pebble work law disagrees")
    return {
        "nodes": nodes,
        "programs": program_results,
        "two_message_integer_cells": 2 * MESSAGE_CELLS,
        "retain_all_integer_cells": nodes * MESSAGE_CELLS,
        "reported_reversible_pebble_integer_cells": (
            case["reversible_pebble_message_integer_cells"]
        ),
        "independent_pebble_forward_applications": (
            expected_applications
        ),
        "direct_enumeration_assignments": (
            PRIME**nodes if nodes in ENUMERATION_NODES else None
        ),
    }


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_chain_transfer_oracle.py "
            "PRODUCTION_RESULT.json"
        )
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        production = json.load(handle)
    cases = [verify_case(case) for case in production["cases"]]
    result = {
        "result": "PASS",
        "oracle": (
            "SEPARATE_PYTHON_LIST_TWO_MESSAGE_DP_RETAIN_ALL_INVERSE_"
            "AND_SMALL_DIRECT_ASSIGNMENT_ENUMERATION"
        ),
        "tested_nodes": [case["nodes"] for case in cases],
        "direct_enumeration_nodes": sorted(ENUMERATION_NODES),
        "cases": cases,
        "production_module_imported": False,
        "production_compiler_called": False,
        "production_transfer_called": False,
        "production_projection_called": False,
        "public_descriptors_consumed": True,
        "all_two_message_boundaries_equal": True,
        "all_retain_all_boundaries_equal": True,
        "all_retain_all_inverse_restorations_exact": True,
        "small_direct_enumerations_equal": True,
        "independent_pebble_resource_recurrence_equal": True,
        "strongest_compact_classical_message_integer_cells": (
            2 * MESSAGE_CELLS
        ),
        "python_integer_object_overhead_bounded": False,
        "python_container_allocator_peak_bounded": False,
        "whole_process_peak_bounded": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
