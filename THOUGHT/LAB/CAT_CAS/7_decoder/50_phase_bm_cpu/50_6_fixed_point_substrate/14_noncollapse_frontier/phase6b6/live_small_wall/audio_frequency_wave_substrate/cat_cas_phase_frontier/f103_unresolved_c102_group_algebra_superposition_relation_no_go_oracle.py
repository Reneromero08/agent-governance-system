#!/usr/bin/env python3
"""Independent scalar oracle for the F103[C102] superposition no-go.

This file imports neither the production module nor NumPy.  It independently
reconstructs every public descriptor, the full 102-coefficient group-algebra
recurrence, characterwise inverse kernels, exact forward/inverse restoration,
the dense F103 evaluation quotient, and the structural hash-consed expression
DAG metrics used by the strict bounded claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FIELD = 103
GENERATOR = 5
ORDER = 102
INTERFACES = (5, 7, 11, 17)
DEPTHS = (1, 2, 4)
FAMILIES = ("PRIMARY", "ALTERNATE")
NODE_COUNT = 9
POWERS = tuple(pow(GENERATOR, exponent, FIELD) for exponent in range(ORDER))


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


def descriptor(interface: int, depth: int, family: str) -> dict[str, Any]:
    code = family_code(family)
    return {
        "schema": "CAT_CAS_F103_C102_GROUP_ALGEBRA_RELATION_PROGRAM_V1",
        "interface": interface,
        "depth": depth,
        "family": family,
        "owner": (0xC1020000 + 257 * interface + 131 * depth + code)
        & 0xFFFFFFFF,
        "node_count": NODE_COUNT,
        "port_type": f"F103_C102_GROUP_ALGEBRA_C{interface}_TO_C{interface}",
        "topology": "PUBLIC_ROTATING_CONTROL_HUB8",
        "composition": "RANK1_LEFT_ACTION_BY_UNRESOLVED_GROUP_ALGEBRA_MOMENT",
        "intersection": "NATIVE_C102_SUPPORT_SHIFT",
        "projection": "FINAL_BOUNDARY_EVALUATION_T_TO_5",
        "observation": [
            (7 * depth + 3 * code + interface) % interface,
            (11 * depth + 5 * code + 2 * interface) % interface,
        ],
    }


def seed_exponent(node: int, family: str, row: int, column: int) -> int:
    code = family_code(family)
    left1 = (code + 3 * node + 2 * row + row * row) % ORDER
    right1 = (7 + node + 5 * column + column**3) % ORDER
    left2 = (11 + 5 * node + 3 * row + row**3) % ORDER
    right2 = (13 + 2 * node + 7 * column + column * column) % ORDER
    return (left1 * right1 + left2 * right2) % ORDER


def hub_index(index: int, family: str, mutation: int = 0) -> int:
    return (5 * index + family_code(family) + mutation) % NODE_COUNT


def peer_order(hub: int, family: str) -> list[int]:
    peers = [(hub + offset) % NODE_COUNT for offset in range(1, NODE_COUNT)]
    return peers if family == "PRIMARY" else list(reversed(peers))


def rotation_shift(interface: int, node: int, index: int, family: str) -> int:
    return (
        3 * node * node
        + 5 * index
        + family_code(family) * (1 + index.bit_count())
    ) % interface


def intersection_exponent(
    hub: int,
    peer: int,
    index: int,
    family: str,
    row: int,
    column: int,
    mutation: int = 0,
) -> int:
    code = family_code(family) + mutation
    left1 = (3 + code + 2 * hub + row + row * row) % ORDER
    right1 = (5 + peer + 3 * index + 2 * column + column**3) % ORDER
    left2 = (7 + peer + index + 3 * row + row**3) % ORDER
    right2 = (11 + hub + code + column + column * column) % ORDER
    return (left1 * right1 + left2 * right2) % ORDER


def evaluations(polynomial: list[int]) -> list[int]:
    return [
        sum(
            coefficient * POWERS[(character * exponent) % ORDER]
            for exponent, coefficient in enumerate(polynomial)
        )
        % FIELD
        for character in range(ORDER)
    ]


def interpolate(values: list[int]) -> list[int]:
    inverse_order = pow(ORDER, -1, FIELD)
    return [
        inverse_order
        * sum(
            value * POWERS[(-character * exponent) % ORDER]
            for character, value in enumerate(values)
        )
        % FIELD
        for exponent in range(ORDER)
    ]


def shift_polynomial(polynomial: list[int], amount: int) -> list[int]:
    amount %= ORDER
    return [polynomial[(exponent - amount) % ORDER] for exponent in range(ORDER)]


def convolution(left: list[int], right: list[int]) -> list[int]:
    result = [0] * ORDER
    for exponent, coefficient in enumerate(right):
        if coefficient:
            shifted = shift_polynomial(left, exponent)
            for position in range(ORDER):
                result[position] = (
                    result[position] + coefficient * shifted[position]
                ) % FIELD
    return result


def composition_plan(
    interface: int,
    hub: int,
    peer: int,
    index: int,
    family: str,
    mutation: int = 0,
) -> tuple[list[int], list[int], int, list[int]]:
    code = family_code(family) + mutation
    left = [
        (17 + code + hub + 3 * x + 2 * x * x) % ORDER
        for x in range(interface)
    ]
    right = [
        (23 + 2 * peer + 5 * x + 3 * x**3 + index) % ORDER
        for x in range(interface)
    ]
    pairing = [0] * ORDER
    for left_value, right_value in zip(left, right, strict=True):
        exponent = (left_value + right_value) % ORDER
        pairing[exponent] = (pairing[exponent] + 1) % FIELD
    excluded = {
        (-pow(value, -1, FIELD)) % FIELD
        for value in evaluations(pairing)
        if value
    }
    coupling = next(value for value in range(1, FIELD) if value not in excluded)
    denominator = pairing[:]
    denominator = [(coupling * value) % FIELD for value in denominator]
    denominator[0] = (denominator[0] + 1) % FIELD
    denominator_values = evaluations(denominator)
    if any(value == 0 for value in denominator_values):
        fail("oracle found noninvertible composition denominator")
    inverse_kernel = interpolate(
        [pow(value, -1, FIELD) for value in denominator_values]
    )
    identity = convolution(denominator, inverse_kernel)
    if identity != [1] + [0] * (ORDER - 1):
        fail("oracle inverse kernel failed")
    return left, right, coupling, inverse_kernel


def seed_coefficients(
    interface: int, family: str
) -> list[list[list[list[int]]]]:
    matrices: list[list[list[list[int]]]] = []
    for node in range(NODE_COUNT):
        matrix: list[list[list[int]]] = []
        for row in range(interface):
            row_values: list[list[int]] = []
            for column in range(interface):
                polynomial = [0] * ORDER
                polynomial[seed_exponent(node, family, row, column)] = 1
                row_values.append(polynomial)
            matrix.append(row_values)
        matrices.append(matrix)
    return matrices


def rotate_matrix(
    matrix: list[list[list[int]]], amount: int
) -> list[list[list[int]]]:
    interface = len(matrix)
    return [
        [
            matrix[(row - amount) % interface][
                (column - amount) % interface
            ][:]
            for column in range(interface)
        ]
        for row in range(interface)
    ]


def compose_matrix(
    matrix: list[list[list[int]]],
    left: list[int],
    right: list[int],
    coupling: int,
    inverse_kernel: list[int],
    inverse: bool,
) -> list[list[list[int]]]:
    interface = len(matrix)
    moments: list[list[int]] = []
    for column in range(interface):
        moment = [0] * ORDER
        for row in range(interface):
            shifted = shift_polynomial(matrix[row][column], right[row])
            moment = [
                (old + value) % FIELD
                for old, value in zip(moment, shifted, strict=True)
            ]
        moments.append(moment)
    if inverse:
        moments = [convolution(moment, inverse_kernel) for moment in moments]
        coupling = -coupling
    return [
        [
            [
                (old + coupling * correction) % FIELD
                for old, correction in zip(
                    matrix[row][column],
                    shift_polynomial(moments[column], left[row]),
                    strict=True,
                )
            ]
            for column in range(interface)
        ]
        for row in range(interface)
    ]


def intersect_matrix(
    matrix: list[list[list[int]]],
    hub: int,
    peer: int,
    index: int,
    family: str,
    inverse: bool,
    mutation: int,
) -> list[list[list[int]]]:
    interface = len(matrix)
    sign = -1 if inverse else 1
    return [
        [
            shift_polynomial(
                matrix[row][column],
                sign
                * intersection_exponent(
                    hub, peer, index, family, row, column, mutation
                ),
            )
            for column in range(interface)
        ]
        for row in range(interface)
    ]


def coefficient_forward(
    interface: int,
    depth: int,
    family: str,
    *,
    action_order: str = "COMPOSE_INTERSECT",
    mutation: int = 0,
    enabled: bool = True,
) -> tuple[list[list[list[list[int]]]], int]:
    matrices = seed_coefficients(interface, family)
    maximum_support = 1
    if not enabled:
        return matrices, maximum_support
    for index in range(depth):
        hub = hub_index(index, family, mutation)
        for peer in peer_order(hub, family):
            current = rotate_matrix(
                matrices[peer], rotation_shift(interface, peer, index, family)
            )
            left, right, coupling, inverse_kernel = composition_plan(
                interface, hub, peer, index, family, mutation
            )
            actions = ("COMPOSE", "INTERSECT")
            if action_order == "INTERSECT_COMPOSE":
                actions = tuple(reversed(actions))
            for action in actions:
                if action == "COMPOSE":
                    current = compose_matrix(
                        current,
                        left,
                        right,
                        coupling,
                        inverse_kernel,
                        False,
                    )
                else:
                    current = intersect_matrix(
                        current, hub, peer, index, family, False, mutation
                    )
            matrices[peer] = current
            maximum_support = max(
                maximum_support,
                max(
                    sum(value != 0 for value in polynomial)
                    for matrix in matrices
                    for row_values in matrix
                    for polynomial in row_values
                ),
            )
    return matrices, maximum_support


def coefficient_inverse(
    matrices: list[list[list[list[int]]]],
    interface: int,
    depth: int,
    family: str,
    *,
    inverse_order: str = "INTERSECT_COMPOSE",
    mutation: int = 0,
) -> None:
    for index in reversed(range(depth)):
        hub = hub_index(index, family, mutation)
        for peer in reversed(peer_order(hub, family)):
            current = matrices[peer]
            left, right, coupling, inverse_kernel = composition_plan(
                interface, hub, peer, index, family, mutation
            )
            actions = ("INTERSECT", "COMPOSE")
            if inverse_order == "COMPOSE_INTERSECT":
                actions = tuple(reversed(actions))
            for action in actions:
                if action == "INTERSECT":
                    current = intersect_matrix(
                        current, hub, peer, index, family, True, mutation
                    )
                else:
                    current = compose_matrix(
                        current,
                        left,
                        right,
                        coupling,
                        inverse_kernel,
                        True,
                    )
            matrices[peer] = rotate_matrix(
                current, -rotation_shift(interface, peer, index, family)
            )


def coefficient_commitment(matrices: list[list[list[list[int]]]]) -> str:
    digest = hashlib.sha256()
    for matrix in matrices:
        for row in matrix:
            for polynomial in row:
                digest.update(bytes(polynomial))
    return digest.hexdigest()


def polynomial_value(polynomial: list[int]) -> int:
    return sum(
        coefficient * POWERS[exponent]
        for exponent, coefficient in enumerate(polynomial)
    ) % FIELD


def coefficient_boundary(
    matrices: list[list[list[list[int]]]],
    program: dict[str, Any],
) -> tuple[int, ...]:
    interface = int(program["interface"])
    left, right = program["observation"]
    return tuple(
        polynomial_value(
            matrices[node][(left + node) % interface][
                (right + 2 * node) % interface
            ]
        )
        for node in range(NODE_COUNT)
    )


def dense_forward(
    interface: int, depth: int, family: str
) -> list[list[list[int]]]:
    matrices = [
        [
            [
                POWERS[seed_exponent(node, family, row, column)]
                for column in range(interface)
            ]
            for row in range(interface)
        ]
        for node in range(NODE_COUNT)
    ]
    for index in range(depth):
        hub = hub_index(index, family)
        for peer in peer_order(hub, family):
            amount = rotation_shift(interface, peer, index, family)
            matrix = [
                [
                    matrices[peer][(row - amount) % interface][
                        (column - amount) % interface
                    ]
                    for column in range(interface)
                ]
                for row in range(interface)
            ]
            left, right, coupling, _inverse_kernel = composition_plan(
                interface, hub, peer, index, family
            )
            moments = [
                sum(
                    POWERS[right[row]] * matrix[row][column]
                    for row in range(interface)
                )
                % FIELD
                for column in range(interface)
            ]
            matrix = [
                [
                    (
                        matrix[row][column]
                        + coupling * POWERS[left[row]] * moments[column]
                    )
                    % FIELD
                    for column in range(interface)
                ]
                for row in range(interface)
            ]
            matrices[peer] = [
                [
                    matrix[row][column]
                    * POWERS[
                        intersection_exponent(
                            hub, peer, index, family, row, column
                        )
                    ]
                    % FIELD
                    for column in range(interface)
                ]
                for row in range(interface)
            ]
    return matrices


def dense_boundary(
    matrices: list[list[list[int]]], program: dict[str, Any]
) -> tuple[int, ...]:
    interface = int(program["interface"])
    left, right = program["observation"]
    return tuple(
        matrices[node][(left + node) % interface][
            (right + 2 * node) % interface
        ]
        for node in range(NODE_COUNT)
    )


class DAG:
    def __init__(self) -> None:
        self.nodes: list[tuple[Any, ...]] = [("ZERO",)]
        self.index: dict[tuple[Any, ...], int] = {self.nodes[0]: 0}

    def intern(self, value: tuple[Any, ...]) -> int:
        if value in self.index:
            return self.index[value]
        identifier = len(self.nodes)
        self.nodes.append(value)
        self.index[value] = identifier
        return identifier

    def monomial(self, coefficient: int, exponent: int) -> int:
        coefficient %= FIELD
        return 0 if not coefficient else self.intern(
            ("MONOMIAL", coefficient, exponent % ORDER)
        )

    def scale_shift(self, node: int, scalar: int, amount: int) -> int:
        scalar %= FIELD
        amount %= ORDER
        if not node or not scalar:
            return 0
        value = self.nodes[node]
        if value[0] == "MONOMIAL":
            return self.monomial(
                scalar * int(value[1]), amount + int(value[2])
            )
        if value[0] == "SCALE_SHIFT":
            return self.scale_shift(
                int(value[3]), scalar * int(value[1]), amount + int(value[2])
            )
        if scalar == 1 and amount == 0:
            return node
        return self.intern(("SCALE_SHIFT", scalar, amount, node))

    def add(self, children: list[int]) -> int:
        counts: dict[int, int] = {}
        for child in children:
            if child:
                counts[child] = (counts.get(child, 0) + 1) % FIELD
        normalized = tuple(
            sorted(
                self.scale_shift(child, count, 0)
                for child, count in counts.items()
                if count
            )
        )
        if not normalized:
            return 0
        if len(normalized) == 1:
            return normalized[0]
        return self.intern(("ADD", normalized))

    def evaluate(self, node: int, memo: dict[int, int]) -> int:
        if node in memo:
            return memo[node]
        value = self.nodes[node]
        if value[0] == "ZERO":
            result = 0
        elif value[0] == "MONOMIAL":
            result = int(value[1]) * POWERS[int(value[2])] % FIELD
        elif value[0] == "SCALE_SHIFT":
            result = (
                int(value[1])
                * POWERS[int(value[2])]
                * self.evaluate(int(value[3]), memo)
            ) % FIELD
        elif value[0] == "ADD":
            result = sum(
                self.evaluate(int(child), memo) for child in value[1]
            ) % FIELD
        else:
            fail("oracle DAG node invalid")
        memo[node] = result
        return result

    def reachable(self, roots: list[list[list[int]]]) -> set[int]:
        pending = [
            node
            for matrix in roots
            for row in matrix
            for node in row
        ]
        seen: set[int] = set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            value = self.nodes[node]
            if value[0] == "SCALE_SHIFT":
                pending.append(int(value[3]))
            elif value[0] == "ADD":
                pending.extend(int(child) for child in value[1])
        return seen

    def logical_bytes(self, roots: list[list[list[int]]]) -> int:
        total = 4 * NODE_COUNT * len(roots[0]) * len(roots[0])
        for node in self.reachable(roots):
            value = self.nodes[node]
            if value[0] == "ZERO":
                total += 1
            elif value[0] == "MONOMIAL":
                total += 3
            elif value[0] == "SCALE_SHIFT":
                total += 7
            elif value[0] == "ADD":
                total += 3 + 4 * len(value[1])
        return total

    def edges(self, roots: list[list[list[int]]]) -> int:
        total = 0
        for node in self.reachable(roots):
            value = self.nodes[node]
            if value[0] == "SCALE_SHIFT":
                total += 1
            elif value[0] == "ADD":
                total += len(value[1])
        return total


def dag_measurement(
    interface: int, depth: int, family: str, program: dict[str, Any]
) -> dict[str, Any]:
    dag = DAG()
    roots = [
        [
            [dag.monomial(1, seed_exponent(node, family, row, column)) for column in range(interface)]
            for row in range(interface)
        ]
        for node in range(NODE_COUNT)
    ]
    seed_nodes = len(dag.reachable(roots))
    seed_bytes = dag.logical_bytes(roots)
    for index in range(depth):
        hub = hub_index(index, family)
        for peer in peer_order(hub, family):
            amount = rotation_shift(interface, peer, index, family)
            matrix = [
                [
                    roots[peer][(row - amount) % interface][
                        (column - amount) % interface
                    ]
                    for column in range(interface)
                ]
                for row in range(interface)
            ]
            left, right, coupling, _inverse_kernel = composition_plan(
                interface, hub, peer, index, family
            )
            moments = [
                dag.add(
                    [
                        dag.scale_shift(matrix[row][column], 1, right[row])
                        for row in range(interface)
                    ]
                )
                for column in range(interface)
            ]
            matrix = [
                [
                    dag.add(
                        [
                            matrix[row][column],
                            dag.scale_shift(
                                moments[column], coupling, left[row]
                            ),
                        ]
                    )
                    for column in range(interface)
                ]
                for row in range(interface)
            ]
            roots[peer] = [
                [
                    dag.scale_shift(
                        matrix[row][column],
                        1,
                        intersection_exponent(
                            hub, peer, index, family, row, column
                        ),
                    )
                    for column in range(interface)
                ]
                for row in range(interface)
            ]
    reachable = dag.reachable(roots)
    left_observation, right_observation = program["observation"]
    memo: dict[int, int] = {}
    boundary = tuple(
        dag.evaluate(
            roots[node][(left_observation + node) % interface][
                (right_observation + 2 * node) % interface
            ],
            memo,
        )
        for node in range(NODE_COUNT)
    )
    return {
        "boundary": boundary,
        "seed_nodes": seed_nodes,
        "forward_nodes": len(reachable),
        "total_nodes": len(dag.nodes),
        "seed_bytes": seed_bytes,
        "forward_bytes": dag.logical_bytes(roots),
        "edges": dag.edges(roots),
    }


def compare(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        fail(f"{label}: production={actual!r} oracle={expected!r}")


def run(production_path: Path) -> dict[str, Any]:
    production = json.loads(production_path.read_text(encoding="utf-8"))
    compare(
        production["schema"],
        "CAT_CAS_F103_C102_GROUP_ALGEBRA_SUPERPOSITION_RELATION_NO_GO_RESULT_V1",
        "schema",
    )
    production_cases = {
        (int(case["interface"]), int(case["depth"]), str(case["family"])): case
        for case in production["cases"]
    }
    comparisons = 1
    exact_inverse_cases = 0
    oracle_cases: list[dict[str, Any]] = []
    for interface in INTERFACES:
        for family in FAMILIES:
            for depth in DEPTHS:
                key = (interface, depth, family)
                case = production_cases[key]
                program = descriptor(interface, depth, family)
                coefficients, maximum_support = coefficient_forward(
                    interface, depth, family
                )
                forward_commitment = coefficient_commitment(coefficients)
                group_boundary = coefficient_boundary(coefficients, program)
                dense = dense_forward(interface, depth, family)
                quotient_boundary = dense_boundary(dense, program)
                dag = dag_measurement(interface, depth, family, program)
                seed_commitment = coefficient_commitment(
                    seed_coefficients(interface, family)
                )
                coefficient_inverse(
                    coefficients, interface, depth, family
                )
                restored = coefficient_commitment(coefficients) == seed_commitment
                if restored:
                    exact_inverse_cases += 1
                expected_values = {
                    "program_fingerprint": digest_json(program),
                    "boundary_commitment": digest_json(list(group_boundary)),
                    "forward_coefficient_commitment": forward_commitment,
                    "phase_payload_bytes": NODE_COUNT * interface * interface * ORDER,
                    "classical_payload_bytes": NODE_COUNT * interface * interface,
                    "phase_to_classical_payload_ratio": 102.0,
                    "support": maximum_support,
                    "dag_seed_nodes": dag["seed_nodes"],
                    "dag_forward_nodes": dag["forward_nodes"],
                    "dag_total_nodes": dag["total_nodes"],
                    "dag_seed_bytes": dag["seed_bytes"],
                    "dag_forward_bytes": dag["forward_bytes"],
                    "dag_edges": dag["edges"],
                }
                production_values = {
                    "program_fingerprint": case["program_fingerprint"],
                    "boundary_commitment": case["boundary_commitment"],
                    "forward_coefficient_commitment": case[
                        "forward_coefficient_commitment"
                    ],
                    "phase_payload_bytes": case["phase_payload_bytes"],
                    "classical_payload_bytes": case["classical_payload_bytes"],
                    "phase_to_classical_payload_ratio": case[
                        "phase_to_classical_payload_ratio"
                    ],
                    "support": case["phase_forward_stats"][
                        "maximum_nonzero_support_per_entry"
                    ],
                    "dag_seed_nodes": case["dag_seed_reachable_nodes"],
                    "dag_forward_nodes": case["dag_forward_reachable_nodes"],
                    "dag_total_nodes": case[
                        "dag_forward_total_interned_nodes"
                    ],
                    "dag_seed_bytes": case["dag_seed_logical_bytes"],
                    "dag_forward_bytes": case["dag_forward_logical_bytes"],
                    "dag_edges": case["dag_forward_predecessor_edges"],
                }
                for label, expected in expected_values.items():
                    compare(production_values[label], expected, f"{key}:{label}")
                    comparisons += 1
                compare(group_boundary, quotient_boundary, f"{key}:dense quotient")
                compare(group_boundary, dag["boundary"], f"{key}:DAG boundary")
                compare(restored, True, f"{key}:inverse")
                comparisons += 3
                oracle_cases.append(
                    {
                        "interface": interface,
                        "depth": depth,
                        "family": family,
                        "program_fingerprint": digest_json(program),
                        "boundary_commitment": digest_json(list(group_boundary)),
                        "forward_coefficient_commitment": forward_commitment,
                        "maximum_support": maximum_support,
                        "dag_forward_nodes": dag["forward_nodes"],
                        "dag_forward_logical_bytes": dag["forward_bytes"],
                        "dag_predecessor_edges": dag["edges"],
                        "exact_inverse_restored": restored,
                    }
                )

    normal, _ = coefficient_forward(5, 2, "PRIMARY")
    normal_program = descriptor(5, 2, "PRIMARY")
    normal_boundary = coefficient_boundary(normal, normal_program)
    missing_changes = coefficient_commitment(normal) != coefficient_commitment(
        seed_coefficients(5, "PRIMARY")
    )
    reordered, _ = coefficient_forward(5, 2, "PRIMARY")
    coefficient_inverse(
        reordered, 5, 2, "PRIMARY", inverse_order="COMPOSE_INTERSECT"
    )
    reordered_changes = coefficient_commitment(reordered) != coefficient_commitment(
        seed_coefficients(5, "PRIMARY")
    )
    swapped, _ = coefficient_forward(
        5, 2, "PRIMARY", action_order="INTERSECT_COMPOSE"
    )
    mutated, _ = coefficient_forward(5, 2, "PRIMARY", mutation=1)
    disabled, _ = coefficient_forward(5, 2, "PRIMARY", enabled=False)
    semantic_controls = {
        "missing_inverse_changes_payload": missing_changes,
        "reordered_inverse_changes_payload": reordered_changes,
        "action_order_changes_boundary": coefficient_boundary(
            swapped, normal_program
        )
        != normal_boundary,
        "topology_mutation_changes_boundary": coefficient_boundary(
            mutated, normal_program
        )
        != normal_boundary,
        "disabled_port_changes_boundary": coefficient_boundary(
            disabled, normal_program
        )
        != normal_boundary,
        "character_table_complete": len(set(POWERS)) == ORDER,
    }
    if not all(semantic_controls.values()):
        fail("oracle semantic controls failed")

    return {
        "schema": "CAT_CAS_F103_C102_GROUP_ALGEBRA_SUPERPOSITION_RELATION_NO_GO_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "imports_production_module": False,
        "imports_numpy": False,
        "case_count": len(oracle_cases),
        "comparison_count": comparisons,
        "exact_inverse_cases": exact_inverse_cases,
        "reconstructed_public_programs": True,
        "reconstructed_full_group_algebra_coefficients": True,
        "reconstructed_characterwise_inverse_kernels": True,
        "reconstructed_forward_coefficient_commitments": True,
        "reconstructed_dense_evaluation_quotient": True,
        "reconstructed_expression_dag_metrics": True,
        "semantic_controls": semantic_controls,
        "cases": oracle_cases,
        "package_local_only": [
            "NATIVE_NUMPY_BACKING_IDENTITY",
            "NATIVE_NUMPY_OPERATION_COUNTERS_AND_TRANSIENT_MAXIMA",
            "PYTHON_ALLOCATOR_AND_NATIVE_LIBRARY_EXCLUSIONS",
            "DIRECT_PROCESS_CUSTODY_CHECKS",
        ],
        "claim_ceiling": production["claim_ceiling"],
        "not_established": production["not_established"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("production_result", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(arguments.production_result)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(payload, encoding="utf-8")
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "case_count": result["case_count"],
                "comparison_count": result["comparison_count"],
                "exact_inverse_cases": result["exact_inverse_cases"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
