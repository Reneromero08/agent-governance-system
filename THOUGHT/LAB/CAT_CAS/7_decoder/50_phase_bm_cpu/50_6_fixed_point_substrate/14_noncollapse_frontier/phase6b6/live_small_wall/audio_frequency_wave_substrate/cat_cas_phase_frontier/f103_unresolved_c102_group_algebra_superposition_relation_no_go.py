#!/usr/bin/env python3
"""Exact unresolved C102 group-algebra phase-superposition diagnostic.

This successor removes M156's intermediate discrete-log rechart.  Each F103
relation entry is an unresolved element of F103[C102].  Hadamard phase factors
act by cyclic support shift; hidden-interface composition adds shifted phase
superpositions.  Only the final public boundary is evaluated by t -> 5.

Two representations are measured.  A 102-coefficient carrier executes actual
forward/inverse algebra with exact same-backing restoration.  A hash-consed
expression DAG defers coefficient expansion but retains predecessor structure;
its algebraic inverse is semantically restoring yet not syntactically restoring
without coefficient canonicalization or the smaller F103 evaluation quotient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


FIELD = 103
GENERATOR = 5
GROUP_ORDER = 102
INTERFACES = (5, 7, 11, 17)
DEPTHS = (1, 2, 4)
FAMILIES = ("PRIMARY", "ALTERNATE")
NODE_COUNT = 9
CLAIM = (
    "BOUNDED_EXACT_F103_UNRESOLVED_C102_GROUP_ALGEBRA_PHASE_"
    "SUPERPOSITION_OPEN_RELATION_COMPOSITION_ELIMINATES_INTERMEDIATE_"
    "LOG_RECHARTING_AND_DEFERS_EVALUATION_TO_THE_FINAL_BOUNDARY_ON_"
    "C5_C7_C11_C17_WITH_EXACT_COEFFICIENT_CARRIER_RESTORATION_AND_"
    "REUSE_BUT_THE_CANONICAL_CARRIER_IS102X_THE_DENSE_F103_QUOTIENT_"
    "AND_THE_COMPACT_EXPRESSION_DAG_RETAINS_GROWING_PREDECESSOR_"
    "HISTORY_WHILE_THE_EVALUATION_QUOTIENT_IS_THE_IDENTICAL_CLASSICAL_"
    "RECURRENCE"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


POWERS = tuple(pow(GENERATOR, exponent, FIELD) for exponent in range(GROUP_ORDER))
if len(set(POWERS)) != GROUP_ORDER:
    fail("declared generator is not primitive")


def family_code(family: str) -> int:
    return {"PRIMARY": 5, "ALTERNATE": 17}[family]


@dataclass(frozen=True)
class Program:
    interface: int
    depth: int
    family: str
    owner: int
    observation_left: int
    observation_right: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "CAT_CAS_F103_C102_GROUP_ALGEBRA_RELATION_PROGRAM_V1",
            "interface": self.interface,
            "depth": self.depth,
            "family": self.family,
            "owner": self.owner,
            "node_count": NODE_COUNT,
            "port_type": f"F103_C102_GROUP_ALGEBRA_C{self.interface}_TO_C{self.interface}",
            "topology": "PUBLIC_ROTATING_CONTROL_HUB8",
            "composition": "RANK1_LEFT_ACTION_BY_UNRESOLVED_GROUP_ALGEBRA_MOMENT",
            "intersection": "NATIVE_C102_SUPPORT_SHIFT",
            "projection": "FINAL_BOUNDARY_EVALUATION_T_TO_5",
            "observation": [self.observation_left, self.observation_right],
        }

    def fingerprint(self) -> str:
        return digest_json(self.descriptor())


def compile_program(interface: int, depth: int, family: str) -> Program:
    if interface not in INTERFACES:
        fail("interface outside declared set")
    if not isinstance(depth, int) or not 1 <= depth <= max(DEPTHS):
        fail("depth outside declared ceiling")
    if family not in FAMILIES:
        fail("family outside declared set")
    code = family_code(family)
    return Program(
        interface,
        depth,
        family,
        (0xC1020000 + 257 * interface + 131 * depth + code) & 0xFFFFFFFF,
        (7 * depth + 3 * code + interface) % interface,
        (11 * depth + 5 * code + 2 * interface) % interface,
    )


def seed_exponent(node: int, family: str, row: int, column: int) -> int:
    code = family_code(family)
    left1 = (code + 3 * node + 2 * row + row * row) % GROUP_ORDER
    right1 = (7 + node + 5 * column + column**3) % GROUP_ORDER
    left2 = (11 + 5 * node + 3 * row + row**3) % GROUP_ORDER
    right2 = (13 + 2 * node + 7 * column + column * column) % GROUP_ORDER
    return (left1 * right1 + left2 * right2) % GROUP_ORDER


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
    left1 = (3 + code + 2 * hub + row + row * row) % GROUP_ORDER
    right1 = (5 + peer + 3 * index + 2 * column + column**3) % GROUP_ORDER
    left2 = (7 + peer + index + 3 * row + row**3) % GROUP_ORDER
    right2 = (11 + hub + code + column + column * column) % GROUP_ORDER
    return (left1 * right1 + left2 * right2) % GROUP_ORDER


def composition_exponents(
    interface: int,
    hub: int,
    peer: int,
    index: int,
    family: str,
    mutation: int = 0,
) -> tuple[list[int], list[int], int]:
    code = family_code(family) + mutation
    left_exponents = [
        (17 + code + hub + 3 * x + 2 * x * x) % GROUP_ORDER
        for x in range(interface)
    ]
    right_exponents = [
        (23 + 2 * peer + 5 * x + 3 * x**3 + index) % GROUP_ORDER
        for x in range(interface)
    ]
    pairing_polynomial = [0] * GROUP_ORDER
    for left, right in zip(left_exponents, right_exponents, strict=True):
        exponent = (left + right) % GROUP_ORDER
        pairing_polynomial[exponent] = (pairing_polynomial[exponent] + 1) % FIELD
    excluded: set[int] = set()
    for value in polynomial_evaluations(pairing_polynomial):
        if value:
            excluded.add((-pow(value, -1, FIELD)) % FIELD)
    coupling = next(value for value in range(1, FIELD) if value not in excluded)
    return left_exponents, right_exponents, coupling


def polynomial_evaluations(polynomial: list[int] | np.ndarray) -> list[int]:
    return [
        sum(
            int(coefficient) * POWERS[(character * exponent) % GROUP_ORDER]
            for exponent, coefficient in enumerate(polynomial)
        )
        % FIELD
        for character in range(GROUP_ORDER)
    ]


def polynomial_from_evaluations(values: list[int]) -> np.ndarray:
    if len(values) != GROUP_ORDER:
        fail("wrong character-evaluation count")
    inverse_order = pow(GROUP_ORDER, -1, FIELD)
    return np.array(
        [
            inverse_order
            * sum(
                int(value) * POWERS[(-character * exponent) % GROUP_ORDER]
                for character, value in enumerate(values)
            )
            % FIELD
            for exponent in range(GROUP_ORDER)
        ],
        dtype=np.uint8,
    )


def cyclic_convolution(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.zeros(GROUP_ORDER, dtype=np.int64)
    for exponent, coefficient in enumerate(right):
        if int(coefficient):
            result += int(coefficient) * np.roll(left.astype(np.int64), exponent)
    return np.asarray(result % FIELD, dtype=np.uint8)


def composition_inverse_kernel(
    left_exponents: list[int], right_exponents: list[int], coupling: int
) -> np.ndarray:
    denominator = [0] * GROUP_ORDER
    denominator[0] = 1
    for left, right in zip(left_exponents, right_exponents, strict=True):
        exponent = (left + right) % GROUP_ORDER
        denominator[exponent] = (denominator[exponent] + coupling) % FIELD
    evaluations = polynomial_evaluations(denominator)
    if any(value == 0 for value in evaluations):
        fail("composition denominator is not invertible in F103[C102]")
    inverse = polynomial_from_evaluations(
        [pow(value, -1, FIELD) for value in evaluations]
    )
    identity = cyclic_convolution(np.asarray(denominator, dtype=np.uint8), inverse)
    expected = np.zeros(GROUP_ORDER, dtype=np.uint8)
    expected[0] = 1
    if not np.array_equal(identity, expected):
        fail("character-derived group-algebra inverse kernel failed")
    return inverse


@dataclass
class WorkStats:
    coefficient_additions: int = 0
    coefficient_scalar_multiplications: int = 0
    cyclic_support_shifts: int = 0
    compositions: int = 0
    intersections: int = 0
    rotations: int = 0
    consumers: int = 0
    maximum_moment_coefficient_cells: int = 0
    maximum_output_buffer_cells: int = 0
    maximum_nonzero_support_per_entry: int = 0
    inverse_convolution_terms: int = 0
    inverse_kernel_maximum_support: int = 0
    maximum_plan_cells: int = 0

    def descriptor(self) -> dict[str, int]:
        return dict(vars(self))


def seed_coefficients(interface: int, family: str) -> np.ndarray:
    result = np.zeros(
        (NODE_COUNT, interface, interface, GROUP_ORDER), dtype=np.uint8
    )
    for node in range(NODE_COUNT):
        for row in range(interface):
            for column in range(interface):
                result[
                    node, row, column, seed_exponent(node, family, row, column)
                ] = 1
    return result


def evaluate_polynomial(polynomial: np.ndarray) -> int:
    return sum(int(polynomial[e]) * POWERS[e] for e in range(GROUP_ORDER)) % FIELD


def evaluate_coefficients(coefficients: np.ndarray) -> np.ndarray:
    interface = coefficients.shape[1]
    result = np.empty((NODE_COUNT, interface, interface), dtype=np.uint8)
    for node in range(NODE_COUNT):
        for row in range(interface):
            for column in range(interface):
                result[node, row, column] = evaluate_polynomial(
                    coefficients[node, row, column]
                )
    return result


def coefficient_state_commitment(coefficients: np.ndarray) -> str:
    return hashlib.sha256(coefficients.tobytes()).hexdigest()


def observe_support(coefficients: np.ndarray, stats: WorkStats | None) -> None:
    if stats is not None:
        support = int(np.count_nonzero(coefficients, axis=3).max())
        stats.maximum_nonzero_support_per_entry = max(
            stats.maximum_nonzero_support_per_entry, support
        )


def compose_coefficients(
    matrix: np.ndarray,
    left_exponents: list[int],
    right_exponents: list[int],
    coupling: int,
    stats: WorkStats | None,
) -> np.ndarray:
    interface = matrix.shape[0]
    moments = np.zeros((interface, GROUP_ORDER), dtype=np.int64)
    for column in range(interface):
        for row in range(interface):
            moments[column] += np.roll(matrix[row, column].astype(np.int64), right_exponents[row])
        moments[column] %= FIELD
    result = np.empty_like(matrix)
    for row in range(interface):
        for column in range(interface):
            correction = coupling * np.roll(moments[column], left_exponents[row])
            result[row, column] = (
                matrix[row, column].astype(np.int64) + correction
            ) % FIELD
    if stats is not None:
        stats.coefficient_additions += 2 * interface * interface * GROUP_ORDER
        stats.coefficient_scalar_multiplications += interface * interface * GROUP_ORDER
        stats.cyclic_support_shifts += 2 * interface * interface
        stats.compositions += 1
        stats.maximum_moment_coefficient_cells = max(
            stats.maximum_moment_coefficient_cells, interface * GROUP_ORDER
        )
        stats.maximum_output_buffer_cells = max(
            stats.maximum_output_buffer_cells, interface * interface * GROUP_ORDER
        )
        stats.maximum_plan_cells = max(
            stats.maximum_plan_cells, 2 * interface + GROUP_ORDER + 1
        )
    return result


def intersect_coefficients(
    matrix: np.ndarray,
    hub: int,
    peer: int,
    index: int,
    family: str,
    *,
    inverse: bool,
    mutation: int,
    stats: WorkStats | None,
) -> np.ndarray:
    interface = matrix.shape[0]
    sign = -1 if inverse else 1
    result = np.empty_like(matrix)
    for row in range(interface):
        for column in range(interface):
            amount = sign * intersection_exponent(
                hub, peer, index, family, row, column, mutation
            )
            result[row, column] = np.roll(matrix[row, column], amount)
    if stats is not None:
        stats.cyclic_support_shifts += interface * interface
        stats.intersections += 1
    return result


def inverse_compose_coefficients(
    matrix: np.ndarray,
    left_exponents: list[int],
    right_exponents: list[int],
    coupling: int,
    stats: WorkStats | None,
) -> np.ndarray:
    interface = matrix.shape[0]
    inverse_kernel = composition_inverse_kernel(
        left_exponents, right_exponents, coupling
    )
    kernel_support = int(np.count_nonzero(inverse_kernel))
    moments = np.zeros((interface, GROUP_ORDER), dtype=np.int64)
    for column in range(interface):
        for row in range(interface):
            moments[column] += np.roll(
                matrix[row, column].astype(np.int64), right_exponents[row]
            )
        moments[column] %= FIELD
    filtered = np.empty((interface, GROUP_ORDER), dtype=np.uint8)
    for column in range(interface):
        filtered[column] = cyclic_convolution(
            np.asarray(moments[column], dtype=np.uint8), inverse_kernel
        )
    result = np.empty_like(matrix)
    for row in range(interface):
        for column in range(interface):
            correction = -coupling * np.roll(
                filtered[column].astype(np.int64), left_exponents[row]
            )
            result[row, column] = (
                matrix[row, column].astype(np.int64) + correction
            ) % FIELD
    if stats is not None:
        stats.coefficient_additions += (
            interface * interface * GROUP_ORDER
            + interface * kernel_support * GROUP_ORDER
            + interface * interface * GROUP_ORDER
        )
        stats.coefficient_scalar_multiplications += (
            interface * kernel_support * GROUP_ORDER
            + interface * interface * GROUP_ORDER
        )
        stats.cyclic_support_shifts += (
            interface * interface
            + interface * kernel_support
            + interface * interface
        )
        stats.compositions += 1
        stats.inverse_convolution_terms += interface * kernel_support
        stats.inverse_kernel_maximum_support = max(
            stats.inverse_kernel_maximum_support, kernel_support
        )
        stats.maximum_moment_coefficient_cells = max(
            stats.maximum_moment_coefficient_cells, 2 * interface * GROUP_ORDER
        )
        stats.maximum_output_buffer_cells = max(
            stats.maximum_output_buffer_cells, interface * interface * GROUP_ORDER
        )
        stats.maximum_plan_cells = max(
            stats.maximum_plan_cells, 2 * interface + GROUP_ORDER + 1
        )
    return result


def overwrite_coefficients(target: np.ndarray, source: np.ndarray) -> None:
    if target.shape != source.shape or target.dtype != source.dtype:
        fail("coefficient carrier backing shape changed")
    np.copyto(target, source)


def rotate_coefficients(matrix: np.ndarray, shift: int, stats: WorkStats | None) -> np.ndarray:
    if stats is not None:
        stats.rotations += 1
    return np.roll(matrix, (shift, shift), axis=(0, 1))


def raw_forward_coefficients(
    coefficients: np.ndarray,
    program: Program,
    stats: WorkStats | None = None,
    *,
    action_order: str = "COMPOSE_INTERSECT",
    topology_mutation: int = 0,
    port_enabled: bool = True,
) -> None:
    if not port_enabled:
        return
    interface = program.interface
    observe_support(coefficients, stats)
    for index in range(program.depth):
        hub = hub_index(index, program.family, topology_mutation)
        for peer in peer_order(hub, program.family):
            current = rotate_coefficients(
                coefficients[peer],
                rotation_shift(interface, peer, index, program.family),
                stats,
            )
            actions = ("COMPOSE", "INTERSECT")
            if action_order == "INTERSECT_COMPOSE":
                actions = tuple(reversed(actions))
            elif action_order != "COMPOSE_INTERSECT":
                fail("unknown action order")
            left, right, coupling = composition_exponents(
                interface, hub, peer, index, program.family, topology_mutation
            )
            for action in actions:
                if action == "COMPOSE":
                    current = compose_coefficients(
                        current, left, right, coupling, stats
                    )
                else:
                    current = intersect_coefficients(
                        current,
                        hub,
                        peer,
                        index,
                        program.family,
                        inverse=False,
                        mutation=topology_mutation,
                        stats=stats,
                    )
            overwrite_coefficients(coefficients[peer], current)
            observe_support(coefficients, stats)
            if stats is not None:
                stats.consumers += 1


def raw_inverse_coefficients(
    coefficients: np.ndarray,
    program: Program,
    stats: WorkStats | None = None,
    *,
    inverse_order: str = "INTERSECT_COMPOSE",
    topology_mutation: int = 0,
) -> None:
    interface = program.interface
    for index in reversed(range(program.depth)):
        hub = hub_index(index, program.family, topology_mutation)
        for peer in reversed(peer_order(hub, program.family)):
            current = coefficients[peer]
            actions = ("INTERSECT", "COMPOSE")
            if inverse_order == "COMPOSE_INTERSECT":
                actions = tuple(reversed(actions))
            elif inverse_order != "INTERSECT_COMPOSE":
                fail("unknown inverse order")
            left, right, coupling = composition_exponents(
                interface, hub, peer, index, program.family, topology_mutation
            )
            for action in actions:
                if action == "INTERSECT":
                    current = intersect_coefficients(
                        current,
                        hub,
                        peer,
                        index,
                        program.family,
                        inverse=True,
                        mutation=topology_mutation,
                        stats=stats,
                    )
                else:
                    current = inverse_compose_coefficients(
                        current, left, right, coupling, stats
                    )
            current = rotate_coefficients(
                current,
                -rotation_shift(interface, peer, index, program.family),
                stats,
            )
            overwrite_coefficients(coefficients[peer], current)
            observe_support(coefficients, stats)
            if stats is not None:
                stats.consumers += 1


def boundary_from_coefficients(
    coefficients: np.ndarray, program: Program
) -> tuple[int, ...]:
    interface = program.interface
    return tuple(
        evaluate_polynomial(
            coefficients[
                node,
                (program.observation_left + node) % interface,
                (program.observation_right + 2 * node) % interface,
            ]
        )
        for node in range(NODE_COUNT)
    )


@dataclass
class ClassicalStats:
    field_multiplications: int = 0
    field_additions: int = 0
    rotations: int = 0
    compositions: int = 0
    intersections: int = 0
    consumers: int = 0
    maximum_control_rematerialization_cells: int = 0

    def descriptor(self) -> dict[str, int]:
        return dict(vars(self))


def dense_seeds(interface: int, family: str) -> np.ndarray:
    return np.array(
        [
            [
                [
                    POWERS[seed_exponent(node, family, row, column)]
                    for column in range(interface)
                ]
                for row in range(interface)
            ]
            for node in range(NODE_COUNT)
        ],
        dtype=np.uint8,
    )


def dense_compose(
    matrix: np.ndarray,
    left_exponents: list[int],
    right_exponents: list[int],
    coupling: int,
    stats: ClassicalStats | None,
) -> np.ndarray:
    interface = matrix.shape[0]
    left = [POWERS[exponent] for exponent in left_exponents]
    right = [POWERS[exponent] for exponent in right_exponents]
    moments = [
        sum(right[row] * int(matrix[row, column]) for row in range(interface))
        % FIELD
        for column in range(interface)
    ]
    result = np.empty_like(matrix)
    for row in range(interface):
        for column in range(interface):
            result[row, column] = (
                int(matrix[row, column])
                + coupling * left[row] * moments[column]
            ) % FIELD
    if stats is not None:
        stats.field_multiplications += 2 * interface * interface
        stats.field_additions += 2 * interface * interface
        stats.compositions += 1
        stats.maximum_control_rematerialization_cells = max(
            stats.maximum_control_rematerialization_cells,
            2 * interface + GROUP_ORDER + 1,
        )
    return result


def dense_intersect(
    matrix: np.ndarray,
    hub: int,
    peer: int,
    index: int,
    family: str,
    *,
    mutation: int,
    stats: ClassicalStats | None,
) -> np.ndarray:
    interface = matrix.shape[0]
    result = np.empty_like(matrix)
    for row in range(interface):
        for column in range(interface):
            exponent = intersection_exponent(
                hub, peer, index, family, row, column, mutation
            )
            result[row, column] = (
                int(matrix[row, column]) * POWERS[exponent]
            ) % FIELD
    if stats is not None:
        stats.field_multiplications += interface * interface
        stats.intersections += 1
    return result


def classical_forward(program: Program) -> tuple[np.ndarray, ClassicalStats]:
    matrices = dense_seeds(program.interface, program.family)
    stats = ClassicalStats()
    for index in range(program.depth):
        hub = hub_index(index, program.family)
        for peer in peer_order(hub, program.family):
            shift = rotation_shift(
                program.interface, peer, index, program.family
            )
            matrices[peer] = np.roll(
                matrices[peer], (shift, shift), axis=(0, 1)
            )
            stats.rotations += 1
            left, right, coupling = composition_exponents(
                program.interface, hub, peer, index, program.family
            )
            matrices[peer] = dense_compose(
                matrices[peer], left, right, coupling, stats
            )
            matrices[peer] = dense_intersect(
                matrices[peer],
                hub,
                peer,
                index,
                program.family,
                mutation=0,
                stats=stats,
            )
            stats.consumers += 1
    return matrices, stats


def boundary_from_dense(matrices: np.ndarray, program: Program) -> tuple[int, ...]:
    interface = program.interface
    return tuple(
        int(
            matrices[
                node,
                (program.observation_left + node) % interface,
                (program.observation_right + 2 * node) % interface,
            ]
        )
        for node in range(NODE_COUNT)
    )


class ExpressionDAG:
    """Hash-consed unresolved group-algebra expressions.

    This is a deliberately structural representation: it performs local
    monomial/scalar normalization and commutative add hash-consing, but it does
    not expand expressions into 102 coefficient cells.  Consequently actual
    predecessor structure remains part of the carrier state.
    """

    def __init__(self) -> None:
        self.nodes: list[tuple[Any, ...]] = [("ZERO",)]
        self.index: dict[tuple[Any, ...], int] = {self.nodes[0]: 0}

    def intern(self, descriptor: tuple[Any, ...]) -> int:
        existing = self.index.get(descriptor)
        if existing is not None:
            return existing
        identifier = len(self.nodes)
        self.nodes.append(descriptor)
        self.index[descriptor] = identifier
        return identifier

    def monomial(self, coefficient: int, exponent: int) -> int:
        coefficient %= FIELD
        if coefficient == 0:
            return 0
        return self.intern(("MONOMIAL", coefficient, exponent % GROUP_ORDER))

    def scale_shift(self, node: int, scalar: int, shift: int) -> int:
        scalar %= FIELD
        shift %= GROUP_ORDER
        if node == 0 or scalar == 0:
            return 0
        descriptor = self.nodes[node]
        if descriptor[0] == "MONOMIAL":
            return self.monomial(
                scalar * int(descriptor[1]), shift + int(descriptor[2])
            )
        if descriptor[0] == "SCALE_SHIFT":
            return self.scale_shift(
                int(descriptor[3]),
                scalar * int(descriptor[1]),
                shift + int(descriptor[2]),
            )
        if scalar == 1 and shift == 0:
            return node
        return self.intern(("SCALE_SHIFT", scalar, shift, node))

    def add(self, children: list[int] | tuple[int, ...]) -> int:
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

    def evaluate(self, node: int, memo: dict[int, int] | None = None) -> int:
        if memo is None:
            memo = {}
        if node in memo:
            return memo[node]
        descriptor = self.nodes[node]
        kind = descriptor[0]
        if kind == "ZERO":
            value = 0
        elif kind == "MONOMIAL":
            value = int(descriptor[1]) * POWERS[int(descriptor[2])] % FIELD
        elif kind == "SCALE_SHIFT":
            value = (
                int(descriptor[1])
                * POWERS[int(descriptor[2])]
                * self.evaluate(int(descriptor[3]), memo)
            ) % FIELD
        elif kind == "ADD":
            value = sum(self.evaluate(int(child), memo) for child in descriptor[1]) % FIELD
        else:
            fail("unknown expression DAG node")
        memo[node] = value
        return value

    def expand(
        self, node: int, memo: dict[int, np.ndarray] | None = None
    ) -> np.ndarray:
        if memo is None:
            memo = {}
        if node in memo:
            return memo[node]
        descriptor = self.nodes[node]
        kind = descriptor[0]
        if kind == "ZERO":
            value = np.zeros(GROUP_ORDER, dtype=np.uint8)
        elif kind == "MONOMIAL":
            value = np.zeros(GROUP_ORDER, dtype=np.uint8)
            value[int(descriptor[2])] = int(descriptor[1])
        elif kind == "SCALE_SHIFT":
            value = np.asarray(
                int(descriptor[1])
                * np.roll(
                    self.expand(int(descriptor[3]), memo).astype(np.int64),
                    int(descriptor[2]),
                )
                % FIELD,
                dtype=np.uint8,
            )
        elif kind == "ADD":
            accumulator = np.zeros(GROUP_ORDER, dtype=np.int64)
            for child in descriptor[1]:
                accumulator += self.expand(int(child), memo).astype(np.int64)
            value = np.asarray(accumulator % FIELD, dtype=np.uint8)
        else:
            fail("unknown expression DAG node")
        memo[node] = value
        return value

    def reachable(self, roots: np.ndarray) -> set[int]:
        pending = [int(value) for value in roots.flat]
        seen: set[int] = set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            descriptor = self.nodes[node]
            if descriptor[0] == "SCALE_SHIFT":
                pending.append(int(descriptor[3]))
            elif descriptor[0] == "ADD":
                pending.extend(int(child) for child in descriptor[1])
        return seen

    def logical_bytes(self, roots: np.ndarray) -> int:
        total = 4 * int(roots.size)
        for node in self.reachable(roots):
            descriptor = self.nodes[node]
            if descriptor[0] == "ZERO":
                total += 1
            elif descriptor[0] == "MONOMIAL":
                total += 3
            elif descriptor[0] == "SCALE_SHIFT":
                total += 7
            elif descriptor[0] == "ADD":
                total += 3 + 4 * len(descriptor[1])
        return total

    def predecessor_edges(self, roots: np.ndarray) -> int:
        total = 0
        for node in self.reachable(roots):
            descriptor = self.nodes[node]
            if descriptor[0] == "SCALE_SHIFT":
                total += 1
            elif descriptor[0] == "ADD":
                total += len(descriptor[1])
        return total


def dag_seeds(interface: int, family: str, dag: ExpressionDAG) -> np.ndarray:
    roots = np.empty((NODE_COUNT, interface, interface), dtype=np.int64)
    for node in range(NODE_COUNT):
        for row in range(interface):
            for column in range(interface):
                roots[node, row, column] = dag.monomial(
                    1, seed_exponent(node, family, row, column)
                )
    return roots


def dag_compose(
    matrix: np.ndarray,
    dag: ExpressionDAG,
    left_exponents: list[int],
    right_exponents: list[int],
    coupling: int,
    *,
    inverse: bool,
) -> np.ndarray:
    interface = matrix.shape[0]
    moments = [
        dag.add(
            [
                dag.scale_shift(int(matrix[row, column]), 1, right_exponents[row])
                for row in range(interface)
            ]
        )
        for column in range(interface)
    ]
    if inverse:
        kernel = composition_inverse_kernel(
            left_exponents, right_exponents, coupling
        )
        moments = [
            dag.add(
                [
                    dag.scale_shift(moment, int(coefficient), exponent)
                    for exponent, coefficient in enumerate(kernel)
                    if int(coefficient)
                ]
            )
            for moment in moments
        ]
        coupling = -coupling
    result = np.empty_like(matrix)
    for row in range(interface):
        for column in range(interface):
            result[row, column] = dag.add(
                [
                    int(matrix[row, column]),
                    dag.scale_shift(
                        moments[column], coupling, left_exponents[row]
                    ),
                ]
            )
    return result


def dag_intersect(
    matrix: np.ndarray,
    dag: ExpressionDAG,
    hub: int,
    peer: int,
    index: int,
    family: str,
    *,
    inverse: bool,
) -> np.ndarray:
    interface = matrix.shape[0]
    sign = -1 if inverse else 1
    result = np.empty_like(matrix)
    for row in range(interface):
        for column in range(interface):
            result[row, column] = dag.scale_shift(
                int(matrix[row, column]),
                1,
                sign
                * intersection_exponent(
                    hub, peer, index, family, row, column
                ),
            )
    return result


def dag_forward(roots: np.ndarray, dag: ExpressionDAG, program: Program) -> None:
    interface = program.interface
    for index in range(program.depth):
        hub = hub_index(index, program.family)
        for peer in peer_order(hub, program.family):
            shift = rotation_shift(interface, peer, index, program.family)
            current = np.roll(roots[peer], (shift, shift), axis=(0, 1))
            left, right, coupling = composition_exponents(
                interface, hub, peer, index, program.family
            )
            current = dag_compose(
                current, dag, left, right, coupling, inverse=False
            )
            current = dag_intersect(
                current,
                dag,
                hub,
                peer,
                index,
                program.family,
                inverse=False,
            )
            np.copyto(roots[peer], current)


def dag_inverse(roots: np.ndarray, dag: ExpressionDAG, program: Program) -> None:
    interface = program.interface
    for index in reversed(range(program.depth)):
        hub = hub_index(index, program.family)
        for peer in reversed(peer_order(hub, program.family)):
            current = dag_intersect(
                roots[peer],
                dag,
                hub,
                peer,
                index,
                program.family,
                inverse=True,
            )
            left, right, coupling = composition_exponents(
                interface, hub, peer, index, program.family
            )
            current = dag_compose(
                current, dag, left, right, coupling, inverse=True
            )
            shift = rotation_shift(interface, peer, index, program.family)
            current = np.roll(current, (-shift, -shift), axis=(0, 1))
            np.copyto(roots[peer], current)


def boundary_from_dag(
    roots: np.ndarray, dag: ExpressionDAG, program: Program
) -> tuple[int, ...]:
    interface = program.interface
    memo: dict[int, int] = {}
    return tuple(
        dag.evaluate(
            int(
                roots[
                    node,
                    (program.observation_left + node) % interface,
                    (program.observation_right + 2 * node) % interface,
                ]
            ),
            memo,
        )
        for node in range(NODE_COUNT)
    )


def dag_case(program: Program) -> dict[str, Any]:
    dag = ExpressionDAG()
    roots = dag_seeds(program.interface, program.family, dag)
    seed_node_count = len(dag.reachable(roots))
    seed_bytes = dag.logical_bytes(roots)
    dag_forward(roots, dag, program)
    reachable = dag.reachable(roots)
    return {
        "boundary": boundary_from_dag(roots, dag, program),
        "seed_reachable_nodes": seed_node_count,
        "forward_reachable_nodes": len(reachable),
        "forward_total_interned_nodes": len(dag.nodes),
        "seed_logical_bytes": seed_bytes,
        "forward_logical_bytes": dag.logical_bytes(roots),
        "forward_predecessor_edges": dag.predecessor_edges(roots),
        "predecessor_graph_retained": dag.predecessor_edges(roots) > 0,
        "coefficient_expansion_used_for_forward_or_projection": False,
    }


def dag_inverse_diagnostic() -> dict[str, Any]:
    program = compile_program(5, 1, "PRIMARY")
    dag = ExpressionDAG()
    roots = dag_seeds(5, "PRIMARY", dag)
    original_roots = roots.copy()
    seed_coefficients_value = seed_coefficients(5, "PRIMARY")
    dag_forward(roots, dag, program)
    forward_nodes = len(dag.reachable(roots))
    dag_inverse(roots, dag, program)
    expanded_memo: dict[int, np.ndarray] = {}
    semantic_restoration = True
    for node in range(NODE_COUNT):
        for row in range(5):
            for column in range(5):
                semantic_restoration &= np.array_equal(
                    dag.expand(int(roots[node, row, column]), expanded_memo),
                    seed_coefficients_value[node, row, column],
                )
    return {
        "scope": "C5_PRIMARY_DEPTH1_ONLY",
        "coefficient_semantics_restore_after_expansion": semantic_restoration,
        "root_identity_restored": bool(np.array_equal(roots, original_roots)),
        "forward_reachable_nodes": forward_nodes,
        "post_inverse_reachable_nodes": len(dag.reachable(roots)),
        "post_inverse_total_interned_nodes": len(dag.nodes),
        "post_inverse_logical_bytes": dag.logical_bytes(roots),
        "post_inverse_predecessor_edges": dag.predecessor_edges(roots),
        "coefficient_expansion_required_to_verify_semantic_restoration": True,
        "actual_expression_carrier_restoration_claimed": False,
        "restoration_classification": "NO_RESTORATION_CLAIM",
    }


@dataclass
class Carrier:
    interface: int
    family: str
    port_type: str
    coefficients: np.ndarray
    generation: int = 0
    restoration_generation: int = 0
    state: str = "RESTORED"
    active_owner: int | None = None
    active_program: str | None = None
    stats: WorkStats = field(default_factory=WorkStats)

    @classmethod
    def fresh(cls, interface: int, family: str) -> "Carrier":
        return cls(
            interface=interface,
            family=family,
            port_type=f"F103_C102_GROUP_ALGEBRA_C{interface}_TO_C{interface}",
            coefficients=seed_coefficients(interface, family),
        )

    def backing_id(self) -> int:
        return int(self.coefficients.__array_interface__["data"][0])

    def payload_bytes(self) -> int:
        return int(self.coefficients.nbytes)

    def resident_bytes(self) -> int:
        return self.payload_bytes() + 16


def carrier_payload_commitment(carrier: Carrier) -> str:
    return digest_json(
        {
            "interface": carrier.interface,
            "family": carrier.family,
            "port_type": carrier.port_type,
            "coefficient_commitment": coefficient_state_commitment(
                carrier.coefficients
            ),
        }
    )


def begin_forward(carrier: Carrier | None, program: Program, owner: int) -> None:
    if carrier is None:
        fail("null carrier")
    if carrier.state != "RESTORED":
        fail("carrier is not restored")
    if carrier.interface != program.interface or carrier.family != program.family:
        fail("carrier/program type mismatch")
    if carrier.port_type != program.descriptor()["port_type"]:
        fail("typed port mismatch")
    if owner != program.owner:
        fail("owner mismatch")
    carrier.state = "FORWARD_ACTIVE"
    carrier.active_owner = owner
    carrier.active_program = program.fingerprint()
    carrier.stats = WorkStats()


def forward(carrier: Carrier, program: Program, owner: int) -> None:
    if carrier.state != "FORWARD_ACTIVE" or carrier.active_owner != owner:
        fail("forward custody mismatch")
    if carrier.active_program != program.fingerprint():
        fail("forward program mismatch")
    raw_forward_coefficients(carrier.coefficients, program, carrier.stats)
    carrier.state = "FORWARD_COMPLETE"


def project(carrier: Carrier, program: Program, owner: int) -> tuple[int, ...]:
    if carrier.state != "FORWARD_COMPLETE":
        fail("projection outside final-boundary stage")
    if carrier.active_owner != owner or carrier.active_program != program.fingerprint():
        fail("projection custody mismatch")
    return boundary_from_coefficients(carrier.coefficients, program)


def project_resident_port(_carrier: Carrier, _node: int) -> None:
    fail("resident group-algebra coefficient projection forbidden")


def inverse(carrier: Carrier, program: Program, owner: int) -> None:
    if carrier.state != "FORWARD_COMPLETE":
        fail("inverse outside forward-complete stage")
    if carrier.active_owner != owner or carrier.active_program != program.fingerprint():
        fail("inverse custody mismatch")
    raw_inverse_coefficients(carrier.coefficients, program, carrier.stats)
    carrier.state = "RESTORED"
    carrier.active_owner = None
    carrier.active_program = None
    carrier.generation += 1
    carrier.restoration_generation = carrier.generation


def transaction(carrier: Carrier, program: Program) -> dict[str, Any]:
    before = carrier_payload_commitment(carrier)
    backing = carrier.backing_id()
    begin_generation = carrier.generation
    begin_forward(carrier, program, program.owner)
    forward(carrier, program, program.owner)
    forward_commitment = coefficient_state_commitment(carrier.coefficients)
    forward_stats = carrier.stats.descriptor()
    boundary = project(carrier, program, program.owner)
    inverse(carrier, program, program.owner)
    restored = carrier_payload_commitment(carrier)
    if restored != before:
        fail("actual group-algebra inverse did not restore exact carrier payload")
    if carrier.backing_id() != backing:
        fail("carrier backing changed across transaction")
    if carrier.generation != begin_generation + 1:
        fail("restoration generation did not advance exactly once")
    return {
        "_boundary": boundary,
        "program_fingerprint": program.fingerprint(),
        "boundary_commitment": digest_json(list(boundary)),
        "forward_coefficient_commitment": forward_commitment,
        "forward_stats": forward_stats,
        "transaction_stats": carrier.stats.descriptor(),
        "payload_restored_exactly": True,
        "same_backing_restored": True,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
        "retained_inverse_history_cells": 0,
        "retained_restoration_baseline_cells": 0,
    }


def execute_case(interface: int, depth: int, family: str) -> dict[str, Any]:
    program = compile_program(interface, depth, family)
    carrier = Carrier.fresh(interface, family)
    receipt = transaction(carrier, program)
    dense, classical_stats = classical_forward(program)
    dense_boundary = boundary_from_dense(dense, program)
    dag_measurement = dag_case(program)
    if dense_boundary != receipt["_boundary"]:
        fail("group-algebra evaluation and matched dense recurrence disagree")
    if dag_measurement["boundary"] != dense_boundary:
        fail("expression DAG and matched dense recurrence disagree")
    phase_payload = carrier.payload_bytes()
    classical_payload = NODE_COUNT * interface * interface
    phase_resident = carrier.resident_bytes()
    classical_resident = classical_payload + 16
    plan_cells = 2 * interface + GROUP_ORDER + 1
    phase_named_peak = (
        phase_resident
        + interface * interface * GROUP_ORDER
        + 8 * interface * GROUP_ORDER
        + interface * GROUP_ORDER
        + 8 * GROUP_ORDER
        + plan_cells
        + len(canonical_json(program.descriptor()))
    )
    classical_named_peak = (
        classical_resident
        + interface * interface
        + 8 * interface
        + plan_cells
        + len(canonical_json(program.descriptor()))
    )
    return {
        "interface": interface,
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "boundary_commitment": receipt["boundary_commitment"],
        "forward_coefficient_commitment": receipt[
            "forward_coefficient_commitment"
        ],
        "phase_forward_stats": receipt["forward_stats"],
        "phase_transaction_stats": receipt["transaction_stats"],
        "classical_forward_stats": classical_stats.descriptor(),
        "phase_payload_bytes": phase_payload,
        "classical_payload_bytes": classical_payload,
        "phase_to_classical_payload_ratio": phase_payload / classical_payload,
        "phase_resident_bytes": phase_resident,
        "classical_resident_bytes": classical_resident,
        "phase_named_warm_peak_bytes": phase_named_peak,
        "classical_named_warm_peak_bytes": classical_named_peak,
        "dag_seed_reachable_nodes": dag_measurement["seed_reachable_nodes"],
        "dag_forward_reachable_nodes": dag_measurement[
            "forward_reachable_nodes"
        ],
        "dag_forward_total_interned_nodes": dag_measurement[
            "forward_total_interned_nodes"
        ],
        "dag_seed_logical_bytes": dag_measurement["seed_logical_bytes"],
        "dag_forward_logical_bytes": dag_measurement["forward_logical_bytes"],
        "dag_forward_predecessor_edges": dag_measurement[
            "forward_predecessor_edges"
        ],
        "dag_predecessor_graph_retained": dag_measurement[
            "predecessor_graph_retained"
        ],
        "dag_coefficient_expansion_used_for_forward_or_projection": dag_measurement[
            "coefficient_expansion_used_for_forward_or_projection"
        ],
        "payload_restored_exactly": receipt["payload_restored_exactly"],
        "same_backing_restored": receipt["same_backing_restored"],
        "restoration_generation": receipt["restoration_generation"],
        "snapshot_used": False,
        "retained_inverse_history_cells": 0,
        "retained_restoration_baseline_cells": 0,
    }


def expect_failure(action: Callable[[], Any]) -> bool:
    try:
        action()
    except RuntimeError:
        return True
    return False


def control_results() -> dict[str, bool]:
    program = compile_program(5, 2, "PRIMARY")
    seed = Carrier.fresh(5, "PRIMARY")
    seed_commitment = carrier_payload_commitment(seed)

    missing = Carrier.fresh(5, "PRIMARY")
    raw_forward_coefficients(missing.coefficients, program)

    wrong = Carrier.fresh(5, "PRIMARY")
    raw_forward_coefficients(wrong.coefficients, program)
    raw_inverse_coefficients(
        wrong.coefficients, program, topology_mutation=1
    )

    reordered = Carrier.fresh(5, "PRIMARY")
    raw_forward_coefficients(reordered.coefficients, program)
    raw_inverse_coefficients(
        reordered.coefficients, program, inverse_order="COMPOSE_INTERSECT"
    )

    normal = Carrier.fresh(5, "PRIMARY")
    raw_forward_coefficients(normal.coefficients, program)
    normal_boundary = boundary_from_coefficients(normal.coefficients, program)

    disabled = Carrier.fresh(5, "PRIMARY")
    raw_forward_coefficients(
        disabled.coefficients, program, port_enabled=False
    )

    swapped = Carrier.fresh(5, "PRIMARY")
    raw_forward_coefficients(
        swapped.coefficients, program, action_order="INTERSECT_COMPOSE"
    )

    mutated = Carrier.fresh(5, "PRIMARY")
    raw_forward_coefficients(
        mutated.coefficients, program, topology_mutation=1
    )

    wrong_owner_rejected = expect_failure(
        lambda: begin_forward(
            Carrier.fresh(5, "PRIMARY"), program, program.owner + 1
        )
    )
    wrong_type_rejected = expect_failure(
        lambda: begin_forward(
            Carrier.fresh(7, "PRIMARY"), program, program.owner
        )
    )
    null_rejected = expect_failure(
        lambda: begin_forward(None, program, program.owner)
    )
    premature_rejected = expect_failure(
        lambda: project(Carrier.fresh(5, "PRIMARY"), program, program.owner)
    )
    resident_rejected = expect_failure(
        lambda: project_resident_port(normal, 0)
    )

    every_kernel_exact = True
    every_character_invertible = True
    for interface in INTERFACES:
        for family in FAMILIES:
            for index in range(max(DEPTHS)):
                hub = hub_index(index, family)
                for peer in peer_order(hub, family):
                    left, right, coupling = composition_exponents(
                        interface, hub, peer, index, family
                    )
                    denominator = np.zeros(GROUP_ORDER, dtype=np.uint8)
                    denominator[0] = 1
                    for left_value, right_value in zip(left, right, strict=True):
                        exponent = (left_value + right_value) % GROUP_ORDER
                        denominator[exponent] = (
                            int(denominator[exponent]) + coupling
                        ) % FIELD
                    every_character_invertible &= all(
                        value != 0
                        for value in polynomial_evaluations(denominator)
                    )
                    kernel = composition_inverse_kernel(
                        left, right, coupling
                    )
                    identity = cyclic_convolution(denominator, kernel)
                    every_kernel_exact &= bool(
                        int(identity[0]) == 1
                        and np.count_nonzero(identity[1:]) == 0
                    )

    return {
        "missing_inverse_changes_payload": carrier_payload_commitment(missing)
        != seed_commitment,
        "wrong_inverse_changes_payload": carrier_payload_commitment(wrong)
        != seed_commitment,
        "reordered_inverse_changes_payload": carrier_payload_commitment(reordered)
        != seed_commitment,
        "null_carrier_rejected": null_rejected,
        "wrong_owner_rejected": wrong_owner_rejected,
        "wrong_type_rejected": wrong_type_rejected,
        "premature_projection_rejected": premature_rejected,
        "resident_coefficient_projection_rejected": resident_rejected,
        "disabled_port_changes_boundary": boundary_from_coefficients(
            disabled.coefficients, program
        )
        != normal_boundary,
        "composition_intersection_order_changes_boundary": boundary_from_coefficients(
            swapped.coefficients, program
        )
        != normal_boundary,
        "topology_mutation_changes_boundary": boundary_from_coefficients(
            mutated.coefficients, program
        )
        != normal_boundary,
        "all_composition_denominators_characterwise_invertible": every_character_invertible,
        "all_group_algebra_inverse_kernels_exact": every_kernel_exact,
        "primitive_character_table_complete": len(set(POWERS)) == GROUP_ORDER,
    }


def reuse_results() -> tuple[dict[str, Any], dict[str, Any]]:
    first = compile_program(11, 2, "PRIMARY")
    second = compile_program(11, 4, "PRIMARY")
    reused = Carrier.fresh(11, "PRIMARY")
    backing = reused.backing_id()
    first_receipt = transaction(reused, first)
    second_receipt = transaction(reused, second)
    fresh = Carrier.fresh(11, "PRIMARY")
    fresh_receipt = transaction(fresh, second)
    unrelated = {
        "first_boundary_commitment": first_receipt["boundary_commitment"],
        "second_boundary_matches_fresh": second_receipt["boundary_commitment"]
        == fresh_receipt["boundary_commitment"],
        "second_resource_signature_matches_fresh": second_receipt[
            "transaction_stats"
        ]
        == fresh_receipt["transaction_stats"],
        "same_backing_consumed": reused.backing_id() == backing,
        "restoration_generation": reused.restoration_generation,
        "snapshot_used": False,
    }
    repeated_carrier = Carrier.fresh(5, "PRIMARY")
    repeated_program = compile_program(5, 2, "PRIMARY")
    repeated_backing = repeated_carrier.backing_id()
    reference: str | None = None
    stable = True
    for _ in range(8):
        receipt = transaction(repeated_carrier, repeated_program)
        if reference is None:
            reference = receipt["boundary_commitment"]
        stable &= receipt["boundary_commitment"] == reference
        stable &= repeated_carrier.backing_id() == repeated_backing
    repeated = {
        "cycles": 8,
        "boundary_stable": stable,
        "same_backing_stable": repeated_carrier.backing_id()
        == repeated_backing,
        "restoration_generation": repeated_carrier.restoration_generation,
        "snapshot_used": False,
    }
    return unrelated, repeated


def run() -> dict[str, Any]:
    cases = [
        execute_case(interface, depth, family)
        for interface in INTERFACES
        for family in FAMILIES
        for depth in DEPTHS
    ]
    controls = control_results()
    if not all(controls.values()):
        fail(
            "group-algebra controls failed: "
            + repr([key for key, value in controls.items() if not value])
        )
    unrelated, repeated = reuse_results()
    if not all(
        (
            unrelated["second_boundary_matches_fresh"],
            unrelated["second_resource_signature_matches_fresh"],
            unrelated["same_backing_consumed"],
            repeated["boundary_stable"],
            repeated["same_backing_stable"],
        )
    ):
        fail("group-algebra carrier reuse failed")

    dag_growth_strict = True
    for interface in INTERFACES:
        for family in FAMILIES:
            selected = sorted(
                (
                    case
                    for case in cases
                    if case["interface"] == interface
                    and case["family"] == family
                ),
                key=lambda case: int(case["depth"]),
            )
            for field_name in (
                "dag_forward_reachable_nodes",
                "dag_forward_logical_bytes",
                "dag_forward_predecessor_edges",
            ):
                values = [int(case[field_name]) for case in selected]
                dag_growth_strict &= all(
                    left < right for left, right in zip(values, values[1:])
                )
    if not dag_growth_strict:
        fail("declared expression DAG depth-growth law did not hold")

    inverse_dag = dag_inverse_diagnostic()
    if not inverse_dag["coefficient_semantics_restore_after_expansion"]:
        fail("expression DAG inverse did not restore coefficient semantics")
    if inverse_dag["root_identity_restored"]:
        fail("expression DAG unexpectedly restored syntactic carrier identity")

    payload_ratio = {
        str(interface): next(
            case["phase_to_classical_payload_ratio"]
            for case in cases
            if case["interface"] == interface
        )
        for interface in INTERFACES
    }
    forward_support_by_interface_depth = {
        str(interface): {
            str(depth): max(
                int(case["phase_forward_stats"]["maximum_nonzero_support_per_entry"])
                for case in cases
                if case["interface"] == interface and case["depth"] == depth
            )
            for depth in DEPTHS
        }
        for interface in INTERFACES
    }
    dag_bytes_by_interface_depth = {
        str(interface): {
            str(depth): max(
                int(case["dag_forward_logical_bytes"])
                for case in cases
                if case["interface"] == interface and case["depth"] == depth
            )
            for depth in DEPTHS
        }
        for interface in INTERFACES
    }

    return {
        "schema": "CAT_CAS_F103_C102_GROUP_ALGEBRA_SUPERPOSITION_RELATION_NO_GO_RESULT_V1",
        "claim": CLAIM,
        "platform": "LINUX_DIRECT_PROCESS_SOFTWARE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "experiment": {
            "field": "F103",
            "unresolved_phase_group_algebra": "F103[C102]",
            "final_evaluation_homomorphism": "T_MAPS_TO_5",
            "interfaces": list(INTERFACES),
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "node_count": NODE_COUNT,
            "shared_unresolved_port": True,
            "consumers_per_layer": NODE_COUNT - 1,
            "native_intersection": "CYCLIC_C102_SUPPORT_SHIFT",
            "native_composition": "RANK1_GROUP_ALGEBRA_LEFT_ACTION",
            "intermediate_log_or_value_rechart": False,
            "final_boundary_only_evaluation": True,
            "ordinary_dense_relation_table_materialized_by_phase_path": False,
            "compiler_inspects_final_answers": False,
        },
        "cases": cases,
        "growth_law": {
            "coefficient_carrier_payload_cells_per_relation": "102N2",
            "matched_dense_quotient_payload_cells_per_relation": "N2",
            "phase_to_classical_payload_ratio_by_interface": payload_ratio,
            "forward_maximum_nonzero_support_by_interface_depth": forward_support_by_interface_depth,
            "expression_dag_logical_bytes_by_interface_depth": dag_bytes_by_interface_depth,
            "expression_dag_nodes_edges_and_bytes_grow_strictly_with_declared_depths": dag_growth_strict,
            "fixed_rank_or_subdense_growth_established": False,
        },
        "expression_dag_inverse_diagnostic": inverse_dag,
        "controls": controls,
        "restoration_and_reuse": {
            "coefficient_carrier_actual_inverse_on_borrowed_carrier": True,
            "coefficient_carrier_exact_payload_restoration": True,
            "coefficient_carrier_same_backing_restoration": True,
            "coefficient_carrier_restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
            "expression_dag_restoration_classification": "NO_RESTORATION_CLAIM",
            "snapshot_used": False,
            "retained_inverse_history_cells": 0,
            "retained_restoration_baseline_cells": 0,
            "unrelated_program_reuse": unrelated,
            "repeated_reuse": repeated,
        },
        "resource_accounting": {
            "all_coefficient_carrier_and_dense_quotient_payloads_counted": True,
            "forward_and_inverse_group_algebra_work_counted": True,
            "inverse_kernel_and_convolution_work_counted": True,
            "maximum_moment_output_and_plan_buffers_counted": True,
            "compiled_program_descriptor_counted": True,
            "expression_dag_roots_nodes_edges_and_logical_bytes_counted": True,
            "controller_backend_traffic_bytes": 0,
            "snapshot_traffic_bytes": 0,
            "python_object_container_allocator_native_library_internal_workspace_and_whole_process_peaks_excluded": True,
            "optimal_classical_recurrence_claimed": False,
        },
        "matched_compact_classical": {
            "implementation": "EXECUTED_DENSE_UINT8_F103_RECURRENCE",
            "relation_state_cells_per_relation": "N2",
            "identical_public_programs_and_boundaries": True,
            "is_exact_evaluation_quotient_of_group_algebra_carrier": True,
            "coherence_or_phase_superposition_advantage_assumed": False,
        },
        "no_smuggle": {
            "raw_final_boundaries_serialized": False,
            "resident_group_algebra_coefficients_serialized": False,
            "expression_dag_serialized": False,
            "ordinary_relation_tables_serialized": False,
            "assignment_or_truth_table_expansion": False,
            "boundary_and_state_commitments_only": True,
        },
        "claim_ceiling": "F103_C102_GROUP_ALGEBRA_AND_HASH_CONSED_EXPRESSION_DAG_ON_DECLARED_C5_C7_C11_C17_NINE_NODE_ROTATING_HUB_FAMILIES_THROUGH_DEPTH4_IN_LINUX_DIRECT_PROCESS_SOFTWARE",
        "preserved_subclaims": [
            "INTERMEDIATE_DISCRETE_LOG_RECHARTING_IS_ELIMINATED",
            "HADAMARD_INTERSECTION_IS_NATIVE_C102_SUPPORT_SHIFT",
            "RANK1_COMPOSITION_CLOSES_IN_F103_C102_GROUP_ALGEBRA",
            "ONLY_FINAL_BOUNDARY_IS_EVALUATED_AT_T_EQUALS5",
            "EXACT_COEFFICIENT_CARRIER_RESTORATION_AND_SAME_BACKING_REUSE",
            "EXECUTED_DENSE_F103_EVALUATION_QUOTIENT_BOUNDARY_PARITY",
        ],
        "obstruction": "THE_CANONICAL_UNRESOLVED_GROUP_ALGEBRA_REQUIRES102_COEFFICIENT_CELLS_PER_RELATION_ENTRY_WHILE_HASH_CONSED_UNEXPANDED_EXPRESSIONS_RETAIN_A_STRICTLY_GROWING_PREDECESSOR_GRAPH_AND_FAIL_ACTUAL_SYNTACTIC_CARRIER_RESTORATION;EVALUATION_AT_T_EQUALS5_IS_AN_EXACT_N2_CLASSICAL_QUOTIENT_WITH_IDENTICAL_BOUNDARY_RECURRENCE",
        "not_established": [
            "COMPACT_OR_FIXED_RANK_UNRESOLVED_GROUP_ALGEBRA_CLOSURE",
            "ACTUAL_EXPRESSION_DAG_CARRIER_RESTORATION",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_COMPUTATION",
        ],
        "next_obstruction": "THE_NEXT_PHASE_MACHINE_CHANGE_MUST_QUOTIENT_UNRESOLVED_SUPERPOSITIONS_WITHOUT_COLLAPSING_TO_THE_IDENTICAL_F103_VALUE_RECURRENCE_OR_RETAINING_A_GROWING_EXPRESSION_HISTORY;A_CANDIDATE_MUST_PRESERVE_A_SHARED_UNRESOLVED_PORT_AND_EXACT_RESTORATION_WHILE BEATING_THE102N2_COEFFICIENT_BACKING_AND_THE_N2_CLASSICAL_EVALUATION_QUOTIENT",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    arguments = parser.parse_args()
    result = run()
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(payload, encoding="utf-8")
    if arguments.summary:
        print(
            json.dumps(
                {
                    "claim": result["claim"],
                    "case_count": result["experiment"]["case_count"],
                    "payload_ratio": result["growth_law"][
                        "phase_to_classical_payload_ratio_by_interface"
                    ],
                    "support": result["growth_law"][
                        "forward_maximum_nonzero_support_by_interface_depth"
                    ],
                    "dag_bytes": result["growth_law"][
                        "expression_dag_logical_bytes_by_interface_depth"
                    ],
                    "obstruction": result["obstruction"],
                },
                sort_keys=True,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
