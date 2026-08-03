#!/usr/bin/env python3
"""Independent scalar oracle for the growing displacement-rank diagnostic.

This file imports neither the production module nor NumPy.  It reconstructs
the public programs, controls, dense relation semantics, cyclic-displacement
ranks, canonical chart commitments, boundaries, inverse execution, mutations,
and resident-coordinate formulas from the sealed production JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODULUS = 103
INTERFACES = (5, 7, 11, 17)
DEPTHS = (1, 2, 4, 8, 16)
FAMILIES = ("PRIMARY", "ALTERNATE")
NODE_COUNT = 9
CONTROL_RANK = 2


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def family_code(family: str) -> int:
    return {"PRIMARY": 5, "ALTERNATE": 17}[family]


def program(interface: int, depth: int, family: str) -> dict[str, Any]:
    code = family_code(family)
    owner = (0xD15A0000 + 257 * interface + 131 * depth + code) & 0xFFFFFFFF
    return {
        "schema": "CAT_CAS_F103_GROWING_DISPLACEMENT_RELATION_PROGRAM_V1",
        "interface": interface,
        "depth": depth,
        "family": family,
        "owner": owner,
        "node_count": NODE_COUNT,
        "port_type": f"F103_CYCLIC_DISPLACEMENT_C{interface}_TO_C{interface}",
        "topology": "PUBLIC_ROTATING_CONTROL_HUB8",
        "composition": "LEFT_ACTION_BY_IDENTITY_PLUS_RANK2",
        "intersection": "HADAMARD_WITH_RECIPROCAL_RANK2_CONTROL",
        "observation": [
            (7 * depth + 3 * code + interface) % interface,
            (11 * depth + 5 * code + 2 * interface) % interface,
        ],
    }


def matrix_copy(matrix: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in matrix]


def bank_copy(bank: list[list[list[int]]]) -> list[list[list[int]]]:
    return [matrix_copy(matrix) for matrix in bank]


def rank_mod(matrix: list[list[int]]) -> int:
    if not matrix:
        return 0
    work = [[value % MODULUS for value in row] for row in matrix]
    rows = len(work)
    columns = len(work[0])
    pivot_row = 0
    for column in range(columns):
        pivot = next(
            (row for row in range(pivot_row, rows) if work[row][column]), None
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = pow(work[pivot_row][column], -1, MODULUS)
        work[pivot_row] = [value * scale % MODULUS for value in work[pivot_row]]
        for row in range(rows):
            if row == pivot_row or not work[row][column]:
                continue
            factor = work[row][column]
            work[row] = [
                (work[row][index] - factor * work[pivot_row][index]) % MODULUS
                for index in range(columns)
            ]
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


def inverse_mod(matrix: list[list[int]]) -> list[list[int]]:
    size = len(matrix)
    work = [
        [value % MODULUS for value in row]
        + [1 if row_index == column else 0 for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if work[row][column]), None
        )
        if pivot is None:
            fail("oracle singular matrix")
        work[column], work[pivot] = work[pivot], work[column]
        scale = pow(work[column][column], -1, MODULUS)
        work[column] = [value * scale % MODULUS for value in work[column]]
        for row in range(size):
            if row == column or not work[row][column]:
                continue
            factor = work[row][column]
            work[row] = [
                (work[row][index] - factor * work[column][index]) % MODULUS
                for index in range(2 * size)
            ]
    return [row[size:] for row in work]


def matmul(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    rows = len(left)
    inner = len(right)
    columns = len(right[0])
    return [
        [
            sum(left[row][middle] * right[middle][column] for middle in range(inner))
            % MODULUS
            for column in range(columns)
        ]
        for row in range(rows)
    ]


def transpose(matrix: list[list[int]]) -> list[list[int]]:
    return [list(column) for column in zip(*matrix)]


def target(interface: int, node: int) -> list[list[int]]:
    result: list[list[int]] = []
    for x in range(interface):
        row: list[int] = []
        for y in range(interface):
            diagonal = (y - x) % interface
            base = (
                3 + 5 * node + 7 * diagonal + 11 * diagonal * diagonal
            ) % MODULUS
            left = (2 + 7 * node + 5 * x + node * x) % MODULUS
            right = (5 + 13 * node + 3 * y + 2 * node * y) % MODULUS
            row.append((base + left * right) % MODULUS)
        result.append(row)
    return result


def seed(interface: int) -> list[list[list[int]]]:
    return [target(interface, node) for node in range(NODE_COUNT)]


def control(interface: int, node: int) -> tuple[list[list[int]], list[list[int]]]:
    q = (7 + 9 * node) % MODULUS
    if q in (0, MODULUS - 1):
        q = 3
    a = [
        1 + ((11 * node + 7 * x + 3 * node * x) % 101)
        for x in range(interface)
    ]
    b = [
        1 + ((13 * node + 5 * y + 4 * node * y) % 101)
        for y in range(interface)
    ]
    selector_left = [
        1 if ((x + 2 * node) % 5) < 2 else 0 for x in range(interface)
    ]
    selector_right = [
        1 if ((3 * y + node) % 7) < 3 else 0 for y in range(interface)
    ]
    left = [[a[x], a[x] * selector_left[x] % MODULUS] for x in range(interface)]
    right = [
        [b[y], b[y] * q * selector_right[y] % MODULUS]
        for y in range(interface)
    ]
    return left, right


def factor_entry(
    factors: tuple[list[list[int]], list[list[int]]], x: int, y: int
) -> int:
    left, right = factors
    return sum(left[x][index] * right[y][index] for index in range(CONTROL_RANK)) % MODULUS


def reciprocal(
    factors: tuple[list[list[int]], list[list[int]]]
) -> tuple[list[list[int]], list[list[int]]]:
    left, right = factors
    inverse_a = [pow(row[0], -1, MODULUS) for row in left]
    inverse_b = [pow(row[0], -1, MODULUS) for row in right]
    selector_left = [left[x][1] * inverse_a[x] % MODULUS for x in range(len(left))]
    selector_scaled = [right[y][1] * inverse_b[y] % MODULUS for y in range(len(right))]
    nonzero = sorted({value for value in selector_scaled if value})
    if len(nonzero) != 1:
        fail("oracle reciprocal structure failed")
    coefficient = -pow(1 + nonzero[0], -1, MODULUS) % MODULUS
    return (
        [
            [inverse_a[x], inverse_a[x] * selector_left[x] % MODULUS]
            for x in range(len(left))
        ],
        [
            [
                inverse_b[y],
                inverse_b[y] * coefficient * selector_scaled[y] % MODULUS,
            ]
            for y in range(len(right))
        ],
    )


def control_view(
    interface: int,
    hub: int,
    peer: int,
    index: int,
    family: str,
    mutation: int = 0,
) -> tuple[list[list[int]], list[list[int]]]:
    left, right = control(interface, hub)
    offset = (
        7 * hub + 11 * peer + 3 * index + family_code(family) + mutation
    ) % interface
    return (
        [left[(x - offset) % interface][:] for x in range(interface)],
        [right[(y + offset) % interface][:] for y in range(interface)],
    )


def coupling(factors: tuple[list[list[int]], list[list[int]]]) -> int:
    left, right = factors
    pairing = matmul(transpose(right), left)
    for value in range(1, MODULUS):
        kernel = [
            [
                ((1 if row == column else 0) + value * pairing[row][column])
                % MODULUS
                for column in range(CONTROL_RANK)
            ]
            for row in range(CONTROL_RANK)
        ]
        determinant = (
            kernel[0][0] * kernel[1][1] - kernel[0][1] * kernel[1][0]
        ) % MODULUS
        if determinant:
            return value
    fail("oracle coupling unavailable")


def compose(
    relation: list[list[int]],
    factors: tuple[list[list[int]], list[list[int]]],
    inverse: bool,
) -> list[list[int]]:
    left, right = factors
    scale = coupling(factors)
    contraction = matmul(transpose(right), relation)
    if inverse:
        pairing = matmul(transpose(right), left)
        kernel = [
            [
                ((1 if row == column else 0) + scale * pairing[row][column])
                % MODULUS
                for column in range(CONTROL_RANK)
            ]
            for row in range(CONTROL_RANK)
        ]
        effective = matmul(left, inverse_mod(kernel))
        effective = [
            [-scale * value % MODULUS for value in row] for row in effective
        ]
    else:
        effective = [[scale * value % MODULUS for value in row] for row in left]
    correction = matmul(effective, contraction)
    return [
        [
            (relation[x][y] + correction[x][y]) % MODULUS
            for y in range(len(relation))
        ]
        for x in range(len(relation))
    ]


def intersect(
    relation: list[list[int]],
    factors: tuple[list[list[int]], list[list[int]]],
    inverse: bool,
) -> list[list[int]]:
    actual = reciprocal(factors) if inverse else factors
    interface = len(relation)
    return [
        [
            relation[x][y] * factor_entry(actual, x, y) % MODULUS
            for y in range(interface)
        ]
        for x in range(interface)
    ]


def rotate(relation: list[list[int]], shift: int) -> list[list[int]]:
    interface = len(relation)
    return [
        [
            relation[(x - shift) % interface][(y - shift) % interface]
            for y in range(interface)
        ]
        for x in range(interface)
    ]


def hub_index(index: int, family: str, mutation: int = 0) -> int:
    return (5 * index + family_code(family) + mutation) % NODE_COUNT


def peer_order(hub: int) -> list[int]:
    return [(hub + offset) % NODE_COUNT for offset in range(1, NODE_COUNT)]


def rotation_shift(interface: int, node: int, index: int, family: str) -> int:
    return (
        3 * node * node
        + 5 * index
        + family_code(family) * (1 + index.bit_count())
    ) % interface


def forward(
    interface: int,
    depth: int,
    family: str,
    *,
    action_order: str = "COMPOSE_INTERSECT",
    port_enabled: bool = True,
    hub_mutation: int = 0,
    offset_mutation: int = 0,
) -> list[list[list[int]]]:
    state = seed(interface)
    actions = ("COMPOSE", "INTERSECT") if action_order == "COMPOSE_INTERSECT" else ("INTERSECT", "COMPOSE")
    for index in range(depth):
        state = [
            rotate(
                state[node],
                rotation_shift(interface, node, index, family),
            )
            for node in range(NODE_COUNT)
        ]
        hub = hub_index(index, family, hub_mutation)
        for peer in peer_order(hub):
            if not port_enabled:
                continue
            factors = control_view(
                interface, hub, peer, index, family, offset_mutation
            )
            for action in actions:
                if action == "COMPOSE":
                    state[peer] = compose(state[peer], factors, False)
                else:
                    state[peer] = intersect(state[peer], factors, False)
    return state


def reverse(
    state: list[list[list[int]]],
    interface: int,
    depth: int,
    family: str,
    *,
    assumed_action_order: str = "COMPOSE_INTERSECT",
    offset_mutation: int = 0,
) -> list[list[list[int]]]:
    restored = bank_copy(state)
    actions = ("INTERSECT", "COMPOSE") if assumed_action_order == "COMPOSE_INTERSECT" else ("COMPOSE", "INTERSECT")
    for index in reversed(range(depth)):
        hub = hub_index(index, family)
        for peer in reversed(peer_order(hub)):
            factors = control_view(
                interface, hub, peer, index, family, offset_mutation
            )
            for action in actions:
                if action == "INTERSECT":
                    restored[peer] = intersect(restored[peer], factors, True)
                else:
                    restored[peer] = compose(restored[peer], factors, True)
        restored = [
            rotate(
                restored[node],
                -rotation_shift(interface, node, index, family),
            )
            for node in range(NODE_COUNT)
        ]
    return restored


def boundary(
    state: list[list[list[int]]], interface: int, depth: int, family: str
) -> list[int]:
    descriptor = program(interface, depth, family)
    observation_left, observation_right = descriptor["observation"]
    values: list[int] = []
    for coordinate in range(interface):
        value = 0
        for node in range(NODE_COUNT):
            x = (observation_left + coordinate + 3 * node) % interface
            y = (observation_right + 2 * coordinate + 5 * node) % interface
            value += (1 + node + coordinate * coordinate) * state[node][x][y]
        values.append(value % MODULUS)
    return values


def displacement(relation: list[list[int]]) -> list[list[int]]:
    interface = len(relation)
    return [
        [
            (
                relation[x][y]
                - relation[(x - 1) % interface][(y - 1) % interface]
            )
            % MODULUS
            for y in range(interface)
        ]
        for x in range(interface)
    ]


def rref_factor(matrix: list[list[int]]) -> tuple[list[list[int]], list[list[int]], list[int]]:
    interface = len(matrix)
    basis: dict[int, list[int]] = {}
    for column in range(interface):
        vector = [matrix[row][column] % MODULUS for row in range(interface)]
        for pivot in sorted(basis):
            factor = vector[pivot]
            if factor:
                vector = [
                    (vector[row] - factor * basis[pivot][row]) % MODULUS
                    for row in range(interface)
                ]
        nonzero = [row for row, value in enumerate(vector) if value]
        if not nonzero:
            continue
        pivot = nonzero[0]
        scale = pow(vector[pivot], -1, MODULUS)
        vector = [value * scale % MODULUS for value in vector]
        for existing in list(basis):
            factor = basis[existing][pivot]
            if factor:
                basis[existing] = [
                    (basis[existing][row] - factor * vector[row]) % MODULUS
                    for row in range(interface)
                ]
        basis[pivot] = vector
    pivots = sorted(basis)
    left = [
        [basis[pivot][row] for pivot in pivots] for row in range(interface)
    ]
    right = [
        [matrix[pivot][column] for pivot in pivots]
        for column in range(interface)
    ]
    return left, right, pivots


def chart_bytes(relation: list[list[int]]) -> tuple[bytes, bytes, int]:
    interface = len(relation)
    delta = displacement(relation)
    left, right, pivots = rref_factor(delta)
    rank = len(pivots)
    nonpivots = [row for row in range(interface) if row not in pivots]
    values = [relation[0][column] for column in range(interface)]
    values.extend(left[row][column] for row in nonpivots for column in range(rank))
    values.extend(right[row][column] for row in range(interface) for column in range(rank))
    capacity = interface + interface * interface
    values.extend([0] * (capacity - len(values)))
    pivot_values = pivots + [255] * (interface - rank)
    return bytes(values), bytes(pivot_values), rank


def chart_commitment(state: list[list[list[int]]]) -> str:
    digest = hashlib.sha256()
    for relation in state:
        payload, pivots, rank = chart_bytes(relation)
        digest.update(payload)
        digest.update(pivots)
        digest.update(bytes([rank]))
    return digest.hexdigest()


def controls() -> dict[str, bool]:
    interface = 7
    depth = 2
    family = "PRIMARY"
    normal = forward(interface, depth, family)
    missing = normal != seed(interface)
    wrong = reverse(normal, interface, depth, family, offset_mutation=1)
    reordered = reverse(
        normal,
        interface,
        depth,
        family,
        assumed_action_order="INTERSECT_COMPOSE",
    )
    disabled = forward(interface, depth, family, port_enabled=False)
    swapped = forward(
        interface, depth, family, action_order="INTERSECT_COMPOSE"
    )
    mutated = forward(interface, depth, family, hub_mutation=1)
    reciprocal_exact = True
    for size in INTERFACES:
        for node in range(NODE_COUNT):
            factors = control(size, node)
            inverse_factors = reciprocal(factors)
            for x in range(size):
                for y in range(size):
                    reciprocal_exact &= (
                        factor_entry(factors, x, y)
                        * factor_entry(inverse_factors, x, y)
                    ) % MODULUS == 1
    return {
        "missing_inverse_changes_state": missing,
        "wrong_inverse_changes_state": wrong != seed(interface),
        "reordered_inverse_changes_state": reordered != seed(interface),
        "null_port_changes_boundary": boundary(normal, interface, depth, family)
        != boundary(disabled, interface, depth, family),
        "composition_intersection_order_changes_boundary": boundary(normal, interface, depth, family)
        != boundary(swapped, interface, depth, family),
        "topology_mutation_changes_boundary": boundary(normal, interface, depth, family)
        != boundary(mutated, interface, depth, family),
        "reciprocal_rank2_controls_exact": reciprocal_exact,
    }


def run(production_path: Path) -> dict[str, Any]:
    production = json.loads(production_path.read_text(encoding="utf-8"))
    case_map = {
        (case["interface"], case["depth"], case["family"]): case
        for case in production["cases"]
    }
    comparisons = 0
    cases: list[dict[str, Any]] = []
    for interface in INTERFACES:
        seed_state = seed(interface)
        seed_displacement = [rank_mod(displacement(relation)) for relation in seed_state]
        seed_ordinary = [rank_mod(relation) for relation in seed_state]
        if seed_displacement != [2] * NODE_COUNT or seed_ordinary != [interface] * NODE_COUNT:
            fail("oracle seed signature mismatch")
        for family in FAMILIES:
            for depth in DEPTHS:
                key = (interface, depth, family)
                production_case = case_map[key]
                descriptor = program(interface, depth, family)
                state = forward(interface, depth, family)
                restored = reverse(state, interface, depth, family)
                ranks = [rank_mod(displacement(relation)) for relation in state]
                ordinary = [rank_mod(relation) for relation in state]
                observed_boundary = boundary(state, interface, depth, family)
                observed_commitment = chart_commitment(state)
                checks = [
                    production_case["program_fingerprint"] == digest_json(descriptor),
                    production_case["boundary_commitment"] == digest_json(observed_boundary),
                    production_case["final_relation_commitment"] == observed_commitment,
                    production_case["final_displacement_ranks"] == ranks,
                    production_case["final_ordinary_ranks"] == ordinary,
                    production_case["maximum_final_displacement_rank"] == max(ranks),
                    production_case["minimum_final_displacement_rank"] == min(ranks),
                    production_case["all_final_relations_full_ordinary_rank"]
                    == all(rank == interface for rank in ordinary),
                    production_case["exact_restoration"],
                    restored == seed_state,
                    production_case["same_backing"],
                    production_case["boundary_identical_to_hybrid_classical_recurrence"],
                ]
                if not all(checks):
                    fail(f"oracle case mismatch: {key}")
                comparisons += len(checks) + NODE_COUNT * 2 + interface
                cases.append(
                    {
                        "interface": interface,
                        "depth": depth,
                        "family": family,
                        "displacement_ranks": ranks,
                        "ordinary_ranks": ordinary,
                        "boundary_commitment": digest_json(observed_boundary),
                        "chart_commitment": observed_commitment,
                        "exact_dense_inverse_restoration": True,
                    }
                )
    independent_controls = controls()
    for key, value in independent_controls.items():
        if not value or not production["controls"][key]:
            fail(f"oracle control mismatch: {key}")
        comparisons += 2
    resident_checks: dict[str, Any] = {}
    for interface in INTERFACES:
        phase = NODE_COUNT * 4 * interface + NODE_COUNT * (
            interface + interface * interface + interface + 1
        )
        classical = NODE_COUNT * interface * interface + NODE_COUNT * 3
        if production["carrier_law"]["resident_bytes_by_interface"][str(interface)] != phase:
            fail("phase resident formula mismatch")
        if production["matched_classical_recurrence"]["maximum_resident_bytes_by_interface"][str(interface)] != classical:
            fail("classical resident formula mismatch")
        ratio = production["matched_classical_recurrence"]["phase_to_classical_resident_byte_ratio_by_interface"][str(interface)]
        if abs(ratio - phase / classical) > 1e-15:
            fail("resident ratio mismatch")
        resident_checks[str(interface)] = {
            "phase": phase,
            "classical": classical,
            "phase_to_classical_ratio": ratio,
        }
        comparisons += 3
    if production["relation_law"]["uniform_interface_independent_rank_bound_observed"]:
        fail("production incorrectly claims uniform displacement-rank bound")
    if not production["relation_law"]["full_displacement_rank_reached_after_one_layer_at_every_interface_and_family"]:
        fail("production missed one-layer saturation")
    comparisons += 2
    return {
        "schema": "CAT_CAS_F103_GROWING_DISPLACEMENT_RELATION_QUOTIENT_ORACLE_RESULT_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "oracle_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_result_sha256": hashlib.sha256(production_path.read_bytes()).hexdigest(),
        "imports_production_module": False,
        "imports_numpy": False,
        "case_count": len(cases),
        "comparison_count": comparisons,
        "cases": cases,
        "controls": independent_controls,
        "resident_formula_checks": resident_checks,
        "preserved_subclaims": [
            "FULL_ORDINARY_RANK_SEEDS_HAVE_EXACT_DISPLACEMENT_RANK2",
            "ONE_LAYER_SATURATES_DISPLACEMENT_RANK_TO_INTERFACE_SIZE",
            "DENSE_SEMANTICS_BOUNDARIES_CHART_COMMITMENTS_AND_INVERSE_RECONSTRUCT",
            "PHASE_RESIDENT_CHART_IS_LARGER_THAN_EXECUTED_CLASSICAL_DENSE_FALLBACK",
        ],
        "package_local_fields_not_independently_recounted": [
            "STREAMED_PHASE_FIELD_MULTIPLICATION_TOTALS",
            "DECLARED_TRANSIENT_SCRATCH_MAXIMA",
            "PYTHON_AND_NUMPY_RUNTIME_EXCLUSIONS",
        ],
        "claim_ceiling": production["claim_ceiling"],
        "next_obstruction": production["next_obstruction"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(arguments.production)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
