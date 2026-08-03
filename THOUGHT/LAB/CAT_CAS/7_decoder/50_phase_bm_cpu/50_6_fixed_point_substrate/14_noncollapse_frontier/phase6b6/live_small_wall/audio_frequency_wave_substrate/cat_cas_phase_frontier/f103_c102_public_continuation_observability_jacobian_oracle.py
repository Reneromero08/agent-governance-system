#!/usr/bin/env python3
"""Independent exact finite-difference oracle for the M159 Jacobian.

This oracle imports neither M159 production, M158 production, nor NumPy.  It
reuses the separately qualified scalar M157 linear group-algebra reference,
reconstructs the M158 shear formulas, and obtains every Jacobian column from
full plus/minus forward executions.  The recurrence is quadratic in the chosen
source A polynomial, so centered differences are exact over F103.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import f103_unresolved_c102_group_algebra_superposition_relation_no_go_oracle as base


INTERFACE = 5
DEPTHS = tuple(range(1, 25))
FAMILIES = base.FAMILIES
SOURCE_ROW = 0
SOURCE_COLUMN = 0
INVERSE_TWO = pow(2, -1, base.FIELD)


def fail(message: str) -> None:
    raise RuntimeError(message)


def source_node(family: str) -> int:
    return base.peer_order(base.hub_index(0, family), family)[0]


def zero_polynomial() -> list[int]:
    return [0] * base.ORDER


def seed_matrix(node: int, family: str, register: int) -> list[list[list[int]]]:
    code = base.family_code(family)
    matrix: list[list[list[int]]] = []
    for row in range(INTERFACE):
        row_values: list[list[int]] = []
        for column in range(INTERFACE):
            polynomial = zero_polynomial()
            exponent = base.seed_exponent(node, family, row, column)
            if register == 1:
                exponent = (
                    exponent
                    + 29
                    + 7 * node
                    + 3 * row
                    + 5 * column
                    + code
                ) % base.ORDER
            polynomial[exponent] = 1
            row_values.append(polynomial)
        matrix.append(row_values)
    return matrix


def clone_matrix(matrix: list[list[list[int]]]) -> list[list[list[int]]]:
    return [[polynomial[:] for polynomial in row] for row in matrix]


def gamma(index: int, hub: int, peer: int, family: str) -> int:
    exponent = (
        31
        + 5 * index
        + 7 * hub
        + 11 * peer
        + base.family_code(family)
    ) % base.ORDER
    return base.POWERS[exponent]


def multipliers(index: int, hub: int, peer: int, family: str) -> list[int]:
    code = base.family_code(family)
    return [
        base.POWERS[
            (
                13
                + code
                + 3 * index
                + 5 * hub
                + 7 * peer
                + 2 * exponent * exponent
                + 3 * exponent
            )
            % base.ORDER
        ]
        for exponent in range(base.ORDER)
    ]


def shear(
    register_a: list[list[list[int]]],
    register_b: list[list[list[int]]],
    index: int,
    hub: int,
    peer: int,
    family: str,
) -> list[list[list[int]]]:
    factor = gamma(index, hub, peer, family)
    weights = multipliers(index, hub, peer, family)
    return [
        [
            [
                (
                    register_b[row][column][exponent]
                    + factor
                    * weights[exponent]
                    * register_a[row][column][exponent]
                    * register_a[row][column][exponent]
                )
                % base.FIELD
                for exponent in range(base.ORDER)
            ]
            for column in range(INTERFACE)
        ]
        for row in range(INTERFACE)
    ]


def advance(
    registers: list[list[list[list[int]]]],
    index: int,
    family: str,
) -> None:
    node = source_node(family)
    hub = base.hub_index(index, family)
    if node == hub:
        return
    if node not in base.peer_order(hub, family):
        fail("source node absent from public peer order")
    amount = base.rotation_shift(INTERFACE, node, index, family)
    current = [base.rotate_matrix(matrix, amount) for matrix in registers]
    left, right, coupling, inverse_kernel = base.composition_plan(
        INTERFACE, hub, node, index, family, 0
    )
    current = [
        base.compose_matrix(
            matrix, left, right, coupling, inverse_kernel, False
        )
        for matrix in current
    ]
    current[1] = shear(current[0], current[1], index, hub, node, family)
    current = [
        base.intersect_matrix(matrix, hub, node, index, family, False, 0)
        for matrix in current
    ]
    registers[:] = current


def evaluate(polynomial: list[int]) -> int:
    return sum(
        coefficient * base.POWERS[exponent]
        for exponent, coefficient in enumerate(polynomial)
    ) % base.FIELD


def project(registers: list[list[list[list[int]]]]) -> list[int]:
    return [
        evaluate(registers[1][row][column])
        for row in range(INTERFACE)
        for column in range(INTERFACE)
    ]


def perturbed_registers(
    family: str, direction: list[int], sign: int
) -> list[list[list[list[int]]]]:
    node = source_node(family)
    registers = [seed_matrix(node, family, register) for register in range(2)]
    polynomial = registers[0][SOURCE_ROW][SOURCE_COLUMN]
    for exponent, value in enumerate(direction):
        polynomial[exponent] = (polynomial[exponent] + sign * value) % base.FIELD
    return registers


def directional_boundaries(family: str, direction: list[int]) -> list[list[int]]:
    plus = perturbed_registers(family, direction, 1)
    minus = perturbed_registers(family, direction, -1)
    result: list[list[int]] = []
    for index in range(len(DEPTHS)):
        advance(plus, index, family)
        advance(minus, index, family)
        plus_boundary = project(plus)
        minus_boundary = project(minus)
        result.append(
            [
                (positive - negative) * INVERSE_TWO % base.FIELD
                for positive, negative in zip(plus_boundary, minus_boundary)
            ]
        )
    return result


def rank_mod(matrix: list[list[int]]) -> int:
    work = [[value % base.FIELD for value in row] for row in matrix]
    pivot_row = 0
    for column in range(len(work[0])):
        candidate = next(
            (row for row in range(pivot_row, len(work)) if work[row][column]),
            None,
        )
        if candidate is None:
            continue
        work[pivot_row], work[candidate] = work[candidate], work[pivot_row]
        inverse = pow(work[pivot_row][column], -1, base.FIELD)
        work[pivot_row] = [value * inverse % base.FIELD for value in work[pivot_row]]
        for row in range(len(work)):
            if row == pivot_row or not work[row][column]:
                continue
            factor = work[row][column]
            work[row] = [
                (left - factor * right) % base.FIELD
                for left, right in zip(work[row], work[pivot_row])
            ]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return pivot_row


def family_result(family: str) -> dict[str, Any]:
    rows = [[0] * base.ORDER for _ in range(len(DEPTHS) * INTERFACE * INTERFACE)]
    for exponent in range(base.ORDER):
        direction = [0] * base.ORDER
        direction[exponent] = 1
        boundaries = directional_boundaries(family, direction)
        for depth_index, boundary in enumerate(boundaries):
            for coordinate, value in enumerate(boundary):
                rows[depth_index * INTERFACE * INTERFACE + coordinate][exponent] = value
    prefix_ranks = {
        str(depth): rank_mod(rows[: depth * INTERFACE * INTERFACE])
        for depth in DEPTHS
    }
    combined = [
        (base.POWERS[(7 * exponent) % base.ORDER] + 3 * exponent + 11)
        % base.FIELD
        for exponent in range(base.ORDER)
    ]
    observed = directional_boundaries(family, combined)
    predicted = [
        sum(rows[depth_index * 25 + coordinate][exponent] * combined[exponent]
            for exponent in range(base.ORDER)) % base.FIELD
        for depth_index in range(len(DEPTHS))
        for coordinate in range(25)
    ]
    flattened_observed = [value for boundary in observed for value in boundary]
    if flattened_observed != predicted:
        fail("whole-recurrence centered-difference linearity control failed")
    flat = bytes(value for row in rows for value in row)
    return {
        "family": family,
        "source_node": source_node(family),
        "prefix_ranks_by_maximum_depth": prefix_ranks,
        "final_rank": rank_mod(rows),
        "jacobian_commitment": hashlib.sha256(flat).hexdigest(),
        "exact_full_recurrence_centered_difference_control": True,
        "basis_direction_reexecutions": base.ORDER,
        "forward_trajectories": 2 * (base.ORDER + 1),
    }


def compare_production(path: Path, families: list[dict[str, Any]]) -> dict[str, Any]:
    production = json.loads(path.read_text(encoding="utf-8"))
    expected = {item["family"]: item for item in production["families"]}
    comparisons = 0
    for observed in families:
        wanted = expected[observed["family"]]
        for field in (
            "source_node",
            "prefix_ranks_by_maximum_depth",
            "final_rank",
            "jacobian_commitment",
        ):
            comparisons += 1
            if observed[field] != wanted[field]:
                fail(f"production mismatch for {observed['family']} {field}")
    return {
        "production_artifact": "F103_C102_PUBLIC_CONTINUATION_OBSERVABILITY_JACOBIAN_RESULTS.json",
        "comparisons": comparisons,
        "all_match": True,
    }


def run(production: Path | None) -> dict[str, Any]:
    families = [family_result(family) for family in FAMILIES]
    if not all(item["final_rank"] == base.ORDER for item in families):
        fail("independent finite-difference Jacobian did not reach rank102")
    result: dict[str, Any] = {
        "schema": "CAT_CAS_F103_C102_PUBLIC_CONTINUATION_OBSERVABILITY_JACOBIAN_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "imports_m159_production": False,
        "imports_m158_production": False,
        "imports_numpy": False,
        "method": "EXACT_FULL_FORWARD_CENTERED_DIFFERENCES_OVER_F103",
        "families": families,
        "full_rank102_on_every_family": True,
        "claim_ceiling": "INDEPENDENT_SCALAR_PARITY_FOR_THE_EXACT_M159_C5_DEPTHS1_TO24_SOURCE_CHART_JACOBIAN_ONLY",
        "resource_accounting": {
            "jacobian_field_cells": len(DEPTHS) * INTERFACE * INTERFACE * base.ORDER,
            "simultaneous_plus_minus_source_node_state_field_cells": 4
            * INTERFACE
            * INTERFACE
            * base.ORDER,
            "accepted_carrier_or_advantage_claimed": False,
        },
    }
    if production is not None:
        result["production_comparison"] = compare_production(production, families)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.production)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(payload, encoding="utf-8")
    if arguments.summary:
        print(
            json.dumps(
                {
                    "ranks": {
                        item["family"]: item["final_rank"]
                        for item in result["families"]
                    },
                    "production_comparison": result.get("production_comparison"),
                },
                sort_keys=True,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
