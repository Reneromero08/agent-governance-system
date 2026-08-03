#!/usr/bin/env python3
"""Exact dual multiplicative-exponent/value-moment relation diagnostic.

Nonzero F103 relation entries are represented by their exponent in the
primitive C102 phase group.  The exponent is stored exactly as three canonical
rank charts over the CRT factors F2, F3, and F17; a fourth F2 chart stores the
zero mask.  Hadamard intersection is native exponent addition.  A reversible
rank-one left composition is evaluated through one streamed value moment per
output column and converted back into the exponent charts without retaining
an ordinary dense relation table.

The experiment asks whether the two native laws coexist in a uniformly compact
chart on C5, C7, C11, and C17.  A matched dense F103 recurrence executes the
same public programs.  This is bounded Linux direct-process software: it is
not CATVM custody, physical waveform execution, or evidence of an advantage.
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
EXPONENT_MODULUS = FIELD - 1
CRT_MODULI = (2, 3, 17)
INTERFACES = (5, 7, 11, 17)
DEPTHS = (1, 2, 4, 8)
FAMILIES = ("PRIMARY", "ALTERNATE")
NODE_COUNT = 9
CLAIM = (
    "BOUNDED_EXACT_F103_DUAL_MULTIPLICATIVE_C102_EXPONENT_CRT_RANK_"
    "AND_VALUE_MOMENT_OPEN_RELATION_CHART_MAKES_HADAMARD_INTERSECTION_"
    "NATIVE_WITHOUT_DENSE_RELATION_TABLES_ON_C5_C7_C11_C17_BUT_ONE_"
    "REVERSIBLE_COMPOSITION_CONVERSION_RAISES_THE_EXPONENT_CHART_TO_"
    "NEAR_FULL_OR_FULL_RANK_IN_EVERY_CRT_COMPONENT_THE_FIXED_BACKING_"
    "IS_AT_LEAST4N2_AND_AN_EXECUTED_DENSE_F103_RECURRENCE_IS_SMALLER_"
    "WITH_EXACT_RESTORATION_AND_REUSE"
)


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


def build_log_table() -> tuple[int, ...]:
    table = [-1] * FIELD
    value = 1
    for exponent in range(EXPONENT_MODULUS):
        if table[value] != -1:
            fail("generator repeats before exhausting F103 multiplicative group")
        table[value] = exponent
        value = value * GENERATOR % FIELD
    if value != 1 or any(table[value] < 0 for value in range(1, FIELD)):
        fail("declared F103 generator is not primitive")
    return tuple(table)


LOG_TABLE = build_log_table()
POWERS = tuple(pow(GENERATOR, exponent, FIELD) for exponent in range(EXPONENT_MODULUS))


def crt_exponent(residues: tuple[int, int, int]) -> int:
    value = 0
    for residue, modulus in zip(residues, CRT_MODULI, strict=True):
        partial = EXPONENT_MODULUS // modulus
        value += residue * partial * pow(partial, -1, modulus)
    return value % EXPONENT_MODULUS


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
            "schema": "CAT_CAS_F103_DUAL_EXPONENT_MOMENT_RELATION_PROGRAM_V1",
            "interface": self.interface,
            "depth": self.depth,
            "family": self.family,
            "owner": self.owner,
            "node_count": NODE_COUNT,
            "port_type": f"F103_DUAL_EXPONENT_MOMENT_C{self.interface}_TO_C{self.interface}",
            "topology": "PUBLIC_ROTATING_CONTROL_HUB8",
            "composition": "REVERSIBLE_RANK1_LEFT_ACTION_THROUGH_STREAMED_VALUE_MOMENT",
            "intersection": "NATIVE_C102_EXPONENT_ADDITION_WITH_ZERO_MASK",
            "observation": [self.observation_left, self.observation_right],
        }

    def fingerprint(self) -> str:
        return digest_json(self.descriptor())


def compile_program(interface: int, depth: int, family: str) -> Program:
    if interface not in INTERFACES:
        fail("interface outside declared growing set")
    if not isinstance(depth, int) or not 1 <= depth <= max(DEPTHS):
        fail("depth outside declared ceiling")
    if family not in FAMILIES:
        fail("family outside declared set")
    code = family_code(family)
    return Program(
        interface=interface,
        depth=depth,
        family=family,
        owner=(0xE11D0000 + 257 * interface + 131 * depth + code) & 0xFFFFFFFF,
        observation_left=(7 * depth + 3 * code + interface) % interface,
        observation_right=(11 * depth + 5 * code + 2 * interface) % interface,
    )


@dataclass
class WorkStats:
    canonicalizations: int = 0
    factor_entry_evaluations: int = 0
    phase_value_entry_evaluations: int = 0
    discrete_log_lookups: int = 0
    exponent_additions: int = 0
    field_multiplications: int = 0
    field_additions: int = 0
    field_inversions: int = 0
    elimination_field_multiplications: int = 0
    composition_conversions: int = 0
    native_intersections: int = 0
    rotations: int = 0
    consumers: int = 0
    maximum_value_moment_cells: int = 0
    maximum_factorization_scratch_cells: int = 0
    maximum_active_value_coordinates: int = 0
    maximum_component_ranks: dict[str, int] = field(
        default_factory=lambda: {"ZERO_F2": 0, "EXP_F2": 0, "EXP_F3": 0, "EXP_F17": 0}
    )

    def observe(self, interface: int, chart: "DualChart") -> None:
        ranks = chart.ranks()
        active = 0
        for name, rank in ranks.items():
            self.maximum_component_ranks[name] = max(
                self.maximum_component_ranks[name], rank
            )
            active += rank * (2 * interface - rank)
        self.maximum_active_value_coordinates = max(
            self.maximum_active_value_coordinates, active
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "canonicalizations": self.canonicalizations,
            "factor_entry_evaluations": self.factor_entry_evaluations,
            "phase_value_entry_evaluations": self.phase_value_entry_evaluations,
            "discrete_log_lookups": self.discrete_log_lookups,
            "exponent_additions": self.exponent_additions,
            "field_multiplications": self.field_multiplications,
            "field_additions": self.field_additions,
            "field_inversions": self.field_inversions,
            "elimination_field_multiplications": self.elimination_field_multiplications,
            "composition_conversions": self.composition_conversions,
            "native_intersections": self.native_intersections,
            "rotations": self.rotations,
            "consumers": self.consumers,
            "maximum_value_moment_cells": self.maximum_value_moment_cells,
            "maximum_factorization_scratch_cells": self.maximum_factorization_scratch_cells,
            "maximum_active_value_coordinates": self.maximum_active_value_coordinates,
            "maximum_component_ranks": dict(self.maximum_component_ranks),
        }


@dataclass
class FactorChart:
    modulus: int
    payload: np.ndarray
    pivots: np.ndarray
    rank: int

    def copy(self) -> "FactorChart":
        return FactorChart(self.modulus, self.payload.copy(), self.pivots.copy(), self.rank)


@dataclass
class DualChart:
    zero: FactorChart
    exponent: tuple[FactorChart, FactorChart, FactorChart]

    def copy(self) -> "DualChart":
        return DualChart(self.zero.copy(), tuple(chart.copy() for chart in self.exponent))

    def parts(self) -> tuple[FactorChart, ...]:
        return (self.zero, *self.exponent)

    def ranks(self) -> dict[str, int]:
        return {
            "ZERO_F2": self.zero.rank,
            "EXP_F2": self.exponent[0].rank,
            "EXP_F3": self.exponent[1].rank,
            "EXP_F17": self.exponent[2].rank,
        }


def factor_coordinates(interface: int, rank: int) -> int:
    return rank * (2 * interface - rank)


def factor_from_entry(
    interface: int,
    modulus: int,
    entry: Callable[[int, int], int],
    stats: WorkStats | None,
) -> FactorChart:
    basis: dict[int, np.ndarray] = {}
    for column in range(interface):
        vector = np.array(
            [entry(row, column) % modulus for row in range(interface)],
            dtype=np.int64,
        )
        if stats is not None:
            stats.factor_entry_evaluations += interface
        for pivot in sorted(basis):
            scale = int(vector[pivot])
            if scale:
                vector = (vector - scale * basis[pivot]) % modulus
                if stats is not None:
                    stats.elimination_field_multiplications += interface
        nonzero = np.flatnonzero(vector)
        if not nonzero.size:
            continue
        pivot = int(nonzero[0])
        scale = pow(int(vector[pivot]), -1, modulus)
        vector = vector * scale % modulus
        if stats is not None:
            stats.field_inversions += 1
            stats.elimination_field_multiplications += interface
        for old_pivot in list(basis):
            scale = int(basis[old_pivot][pivot])
            if scale:
                basis[old_pivot] = (basis[old_pivot] - scale * vector) % modulus
                if stats is not None:
                    stats.elimination_field_multiplications += interface
        basis[pivot] = vector
    pivots = sorted(basis)
    rank = len(pivots)
    payload = np.zeros(interface * interface, dtype=np.uint8)
    pivot_payload = np.full(interface, 255, dtype=np.uint8)
    if rank:
        left = np.stack([basis[pivot] for pivot in pivots], axis=1)
        right = np.empty((interface, rank), dtype=np.int64)
        for column in range(interface):
            right[column] = np.array(
                [entry(pivot, column) % modulus for pivot in pivots], dtype=np.int64
            )
        if stats is not None:
            stats.factor_entry_evaluations += interface * rank
        nonpivots = [row for row in range(interface) if row not in pivots]
        values = np.concatenate((left[nonpivots].reshape(-1), right.reshape(-1)))
        payload[: values.size] = values.astype(np.uint8)
        pivot_payload[:rank] = np.array(pivots, dtype=np.uint8)
        for row in range(interface):
            for column in range(interface):
                actual = int(np.dot(left[row], right[column]) % modulus)
                if actual != entry(row, column) % modulus:
                    fail("canonical rank chart reconstruction failed")
        if stats is not None:
            stats.factor_entry_evaluations += interface * interface
            stats.maximum_factorization_scratch_cells = max(
                stats.maximum_factorization_scratch_cells,
                3 * interface * rank + 3 * interface,
            )
    if stats is not None:
        stats.canonicalizations += 1
    return FactorChart(modulus, payload, pivot_payload, rank)


def unpack_factor(interface: int, chart: FactorChart) -> tuple[np.ndarray, np.ndarray]:
    rank = int(chart.rank)
    if not 0 <= rank <= interface:
        fail("factor rank outside interface")
    if rank == 0:
        return (
            np.empty((interface, 0), dtype=np.int64),
            np.empty((interface, 0), dtype=np.int64),
        )
    pivots = [int(value) for value in chart.pivots[:rank]]
    if len(set(pivots)) != rank or any(not 0 <= value < interface for value in pivots):
        fail("invalid factor pivot metadata")
    nonpivots = [row for row in range(interface) if row not in pivots]
    left_count = (interface - rank) * rank
    count = factor_coordinates(interface, rank)
    values = chart.payload[:count].astype(np.int64)
    left = np.zeros((interface, rank), dtype=np.int64)
    left[pivots] = np.eye(rank, dtype=np.int64)
    left[nonpivots] = values[:left_count].reshape(interface - rank, rank)
    right = values[left_count:].reshape(interface, rank)
    return left, right


def factor_entry(interface: int, chart: FactorChart, row: int, column: int) -> int:
    left, right = unpack_factor(interface, chart)
    return int(np.dot(left[row], right[column]) % chart.modulus)


def dual_code(interface: int, chart: DualChart, row: int, column: int) -> tuple[bool, int]:
    zero = bool(factor_entry(interface, chart.zero, row, column))
    if zero:
        return True, 0
    residues = tuple(
        factor_entry(interface, part, row, column) for part in chart.exponent
    )
    return False, crt_exponent(residues)


def dual_value(interface: int, chart: DualChart, row: int, column: int) -> int:
    zero, exponent = dual_code(interface, chart, row, column)
    return 0 if zero else POWERS[exponent]


def dual_reader(
    interface: int, chart: DualChart
) -> Callable[[int, int], tuple[bool, int]]:
    """Pre-unpack immutable factors while preserving streamed entry access."""
    unpacked = [unpack_factor(interface, part) for part in chart.parts()]

    def read(row: int, column: int) -> tuple[bool, int]:
        zero_left, zero_right = unpacked[0]
        zero = bool(np.dot(zero_left[row], zero_right[column]) % 2)
        if zero:
            return True, 0
        residues = tuple(
            int(np.dot(left[row], right[column]) % modulus)
            for (left, right), modulus in zip(
                unpacked[1:], CRT_MODULI, strict=True
            )
        )
        return False, crt_exponent(residues)

    return read


def dual_from_code_entry(
    interface: int,
    entry: Callable[[int, int], tuple[bool, int]],
    stats: WorkStats | None,
) -> DualChart:
    zero = factor_from_entry(
        interface, 2, lambda row, column: int(entry(row, column)[0]), stats
    )
    def residue(row: int, column: int, modulus: int) -> int:
        zero_flag, exponent_value = entry(row, column)
        return 0 if zero_flag else exponent_value % modulus

    exponent = tuple(
        factor_from_entry(
            interface,
            modulus,
            lambda row, column, modulus=modulus: residue(row, column, modulus),
            stats,
        )
        for modulus in CRT_MODULI
    )
    result = DualChart(zero, exponent)
    if stats is not None:
        stats.observe(interface, result)
    return result


def dual_from_value_entry(
    interface: int,
    entry: Callable[[int, int], int],
    stats: WorkStats | None,
) -> DualChart:
    def code(row: int, column: int) -> tuple[bool, int]:
        value = entry(row, column) % FIELD
        if stats is not None:
            stats.phase_value_entry_evaluations += 1
        if value == 0:
            return True, 0
        if stats is not None:
            stats.discrete_log_lookups += 1
        exponent = LOG_TABLE[value]
        if exponent < 0:
            fail("nonzero F103 value has no declared phase exponent")
        return False, exponent

    return dual_from_code_entry(interface, code, stats)


def overwrite_dual(target: DualChart, source: DualChart) -> None:
    for old, new in zip(target.parts(), source.parts(), strict=True):
        if old.modulus != new.modulus or old.payload.shape != new.payload.shape:
            fail("dual chart backing shape changed")
        np.copyto(old.payload, new.payload)
        np.copyto(old.pivots, new.pivots)
        old.rank = new.rank


def chart_commitment(chart: DualChart) -> str:
    digest = hashlib.sha256()
    for part in chart.parts():
        digest.update(bytes((part.modulus, part.rank)))
        digest.update(part.payload.tobytes())
        digest.update(part.pivots.tobytes())
    return digest.hexdigest()


def charts_commitment(charts: list[DualChart]) -> str:
    return digest_json([chart_commitment(chart) for chart in charts])


def seed_exponent(interface: int, node: int, family: str, row: int, column: int) -> int:
    code = family_code(family)
    left1 = (code + 3 * node + 2 * row + row * row) % EXPONENT_MODULUS
    right1 = (7 + node + 5 * column + column**3) % EXPONENT_MODULUS
    left2 = (11 + 5 * node + 3 * row + row**3) % EXPONENT_MODULUS
    right2 = (13 + 2 * node + 7 * column + column * column) % EXPONENT_MODULUS
    return (left1 * right1 + left2 * right2) % EXPONENT_MODULUS


def seed_charts(interface: int, family: str) -> list[DualChart]:
    return [
        dual_from_code_entry(
            interface,
            lambda row, column, node=node: (
                False,
                seed_exponent(interface, node, family, row, column),
            ),
            None,
        )
        for node in range(NODE_COUNT)
    ]


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
    interface: int,
    hub: int,
    peer: int,
    index: int,
    family: str,
    row: int,
    column: int,
    mutation: int = 0,
) -> int:
    code = family_code(family) + mutation
    left1 = (3 + code + 2 * hub + row + row * row) % EXPONENT_MODULUS
    right1 = (5 + peer + 3 * index + 2 * column + column**3) % EXPONENT_MODULUS
    left2 = (7 + peer + index + 3 * row + row**3) % EXPONENT_MODULUS
    right2 = (11 + hub + code + column + column * column) % EXPONENT_MODULUS
    return (left1 * right1 + left2 * right2) % EXPONENT_MODULUS


def composition_vectors(
    interface: int,
    hub: int,
    peer: int,
    index: int,
    family: str,
    mutation: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    code = family_code(family) + mutation
    positions = np.arange(interface, dtype=np.int64)
    left_exponents = (
        17 + code + hub + 3 * positions + 2 * positions * positions
    ) % EXPONENT_MODULUS
    right_exponents = (
        23 + 2 * peer + 5 * positions + 3 * positions**3 + index
    ) % EXPONENT_MODULUS
    left = np.array([POWERS[int(value)] for value in left_exponents], dtype=np.int64)
    right = np.array([POWERS[int(value)] for value in right_exponents], dtype=np.int64)
    pairing = int(np.dot(right, left) % FIELD)
    coupling = next(
        value for value in range(1, FIELD) if (1 + value * pairing) % FIELD
    )
    return left, right, coupling


def rotate_dual(
    interface: int, chart: DualChart, shift: int, stats: WorkStats | None
) -> DualChart:
    old = dual_reader(interface, chart)
    result = dual_from_code_entry(
        interface,
        lambda row, column: old(
            (row - shift) % interface,
            (column - shift) % interface,
        ),
        stats,
    )
    if stats is not None:
        stats.rotations += 1
    return result


def intersect_dual(
    interface: int,
    chart: DualChart,
    hub: int,
    peer: int,
    index: int,
    family: str,
    *,
    inverse: bool,
    mutation: int,
    stats: WorkStats | None,
) -> DualChart:
    sign = -1 if inverse else 1
    old = dual_reader(interface, chart)

    def updated(row: int, column: int) -> tuple[bool, int]:
        zero, exponent = old(row, column)
        if zero:
            return True, 0
        control = intersection_exponent(
            interface, hub, peer, index, family, row, column, mutation
        )
        if stats is not None:
            stats.exponent_additions += 1
        return False, (exponent + sign * control) % EXPONENT_MODULUS

    result = dual_from_code_entry(interface, updated, stats)
    if stats is not None:
        stats.native_intersections += 1
    return result


def compose_dual(
    interface: int,
    chart: DualChart,
    hub: int,
    peer: int,
    index: int,
    family: str,
    *,
    inverse: bool,
    mutation: int,
    stats: WorkStats | None,
) -> DualChart:
    left, right, coupling = composition_vectors(
        interface, hub, peer, index, family, mutation
    )
    pairing = int(np.dot(right, left) % FIELD)
    effective = (
        -coupling * pow((1 + coupling * pairing) % FIELD, -1, FIELD)
        if inverse
        else coupling
    ) % FIELD
    if stats is not None and inverse:
        stats.field_inversions += 1
    old = dual_reader(interface, chart)

    def old_value(row: int, column: int) -> int:
        zero, exponent = old(row, column)
        return 0 if zero else POWERS[exponent]

    moments = np.empty(interface, dtype=np.int64)
    for column in range(interface):
        total = 0
        for row in range(interface):
            total += int(right[row]) * old_value(row, column)
        moments[column] = total % FIELD
    if stats is not None:
        stats.field_multiplications += interface * interface
        stats.field_additions += interface * interface
        stats.maximum_value_moment_cells = max(
            stats.maximum_value_moment_cells, interface
        )

    def updated(row: int, column: int) -> int:
        old_entry = old_value(row, column)
        correction = effective * int(left[row]) * int(moments[column])
        if stats is not None:
            stats.field_multiplications += 2
            stats.field_additions += 1
        return (old_entry + correction) % FIELD

    result = dual_from_value_entry(interface, updated, stats)
    if stats is not None:
        stats.composition_conversions += 1
    return result


def raw_forward(
    charts: list[DualChart],
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
    for index in range(program.depth):
        hub = hub_index(index, program.family, topology_mutation)
        for peer in peer_order(hub, program.family):
            current = rotate_dual(
                interface,
                charts[peer],
                rotation_shift(interface, peer, index, program.family),
                stats,
            )
            actions = ("COMPOSE", "INTERSECT")
            if action_order == "INTERSECT_COMPOSE":
                actions = tuple(reversed(actions))
            elif action_order != "COMPOSE_INTERSECT":
                fail("unknown action order")
            for action in actions:
                if action == "COMPOSE":
                    current = compose_dual(
                        interface,
                        current,
                        hub,
                        peer,
                        index,
                        program.family,
                        inverse=False,
                        mutation=topology_mutation,
                        stats=stats,
                    )
                else:
                    current = intersect_dual(
                        interface,
                        current,
                        hub,
                        peer,
                        index,
                        program.family,
                        inverse=False,
                        mutation=topology_mutation,
                        stats=stats,
                    )
            overwrite_dual(charts[peer], current)
            if stats is not None:
                stats.consumers += 1


def raw_inverse(
    charts: list[DualChart],
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
            current = charts[peer]
            actions = ("INTERSECT", "COMPOSE")
            if inverse_order == "COMPOSE_INTERSECT":
                actions = tuple(reversed(actions))
            elif inverse_order != "INTERSECT_COMPOSE":
                fail("unknown inverse order")
            for action in actions:
                if action == "INTERSECT":
                    current = intersect_dual(
                        interface,
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
                    current = compose_dual(
                        interface,
                        current,
                        hub,
                        peer,
                        index,
                        program.family,
                        inverse=True,
                        mutation=topology_mutation,
                        stats=stats,
                    )
            current = rotate_dual(
                interface,
                current,
                -rotation_shift(interface, peer, index, program.family),
                stats,
            )
            overwrite_dual(charts[peer], current)
            if stats is not None:
                stats.consumers += 1


def boundary_from_dual(charts: list[DualChart], program: Program) -> tuple[int, ...]:
    interface = program.interface
    return tuple(
        dual_value(
            interface,
            chart,
            (program.observation_left + node) % interface,
            (program.observation_right + 2 * node) % interface,
        )
        for node, chart in enumerate(charts)
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
        return {
            "field_multiplications": self.field_multiplications,
            "field_additions": self.field_additions,
            "rotations": self.rotations,
            "compositions": self.compositions,
            "intersections": self.intersections,
            "consumers": self.consumers,
            "maximum_control_rematerialization_cells": self.maximum_control_rematerialization_cells,
        }


def dense_seeds(interface: int, family: str) -> list[np.ndarray]:
    return [
        np.array(
            [
                [POWERS[seed_exponent(interface, node, family, row, column)] for column in range(interface)]
                for row in range(interface)
            ],
            dtype=np.uint8,
        )
        for node in range(NODE_COUNT)
    ]


def dense_compose(
    matrix: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    coupling: int,
    stats: ClassicalStats | None,
) -> np.ndarray:
    interface = matrix.shape[0]
    moment = np.empty(interface, dtype=np.int64)
    for column in range(interface):
        moment[column] = sum(
            int(right[row]) * int(matrix[row, column])
            for row in range(interface)
        ) % FIELD
    result = np.empty_like(matrix)
    for row in range(interface):
        for column in range(interface):
            result[row, column] = (
                int(matrix[row, column])
                + coupling * int(left[row]) * int(moment[column])
            ) % FIELD
    if stats is not None:
        stats.field_multiplications += 2 * interface * interface
        stats.field_additions += 2 * interface * interface
        stats.compositions += 1
        stats.maximum_control_rematerialization_cells = max(
            stats.maximum_control_rematerialization_cells, 2 * interface
        )
    return result


def dense_intersect(
    matrix: np.ndarray,
    interface: int,
    hub: int,
    peer: int,
    index: int,
    family: str,
    stats: ClassicalStats | None,
) -> np.ndarray:
    result = np.empty_like(matrix)
    for row in range(interface):
        for column in range(interface):
            exponent = intersection_exponent(
                interface, hub, peer, index, family, row, column
            )
            result[row, column] = int(matrix[row, column]) * POWERS[exponent] % FIELD
    if stats is not None:
        stats.field_multiplications += interface * interface
        stats.intersections += 1
    return result


def classical_forward(program: Program) -> tuple[list[np.ndarray], ClassicalStats]:
    interface = program.interface
    matrices = dense_seeds(interface, program.family)
    stats = ClassicalStats()
    for index in range(program.depth):
        hub = hub_index(index, program.family)
        for peer in peer_order(hub, program.family):
            shift = rotation_shift(interface, peer, index, program.family)
            matrices[peer] = np.roll(matrices[peer], (shift, shift), axis=(0, 1))
            stats.rotations += 1
            left, right, coupling = composition_vectors(
                interface, hub, peer, index, program.family
            )
            matrices[peer] = dense_compose(
                matrices[peer], left, right, coupling, stats
            )
            matrices[peer] = dense_intersect(
                matrices[peer], interface, hub, peer, index, program.family, stats
            )
            stats.consumers += 1
    return matrices, stats


def boundary_from_dense(matrices: list[np.ndarray], program: Program) -> tuple[int, ...]:
    interface = program.interface
    return tuple(
        int(
            matrix[
                (program.observation_left + node) % interface,
                (program.observation_right + 2 * node) % interface,
            ]
        )
        for node, matrix in enumerate(matrices)
    )


@dataclass
class Carrier:
    interface: int
    family: str
    port_type: str
    charts: list[DualChart]
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
            port_type=f"F103_DUAL_EXPONENT_MOMENT_C{interface}_TO_C{interface}",
            charts=seed_charts(interface, family),
        )

    def backing_ids(self) -> tuple[int, ...]:
        ids: list[int] = []
        for chart in self.charts:
            for part in chart.parts():
                ids.extend(
                    (
                        int(part.payload.__array_interface__["data"][0]),
                        int(part.pivots.__array_interface__["data"][0]),
                    )
                )
        return tuple(ids)

    def resident_bytes(self) -> int:
        array_bytes = sum(
            part.payload.nbytes + part.pivots.nbytes
            for chart in self.charts
            for part in chart.parts()
        )
        rank_metadata_bytes = NODE_COUNT * 4 * 2
        generation_metadata_bytes = 16
        return array_bytes + rank_metadata_bytes + generation_metadata_bytes

    def active_value_coordinates(self) -> int:
        return sum(
            factor_coordinates(self.interface, part.rank)
            for chart in self.charts
            for part in chart.parts()
        )


def carrier_payload_commitment(carrier: Carrier) -> str:
    return digest_json(
        {
            "interface": carrier.interface,
            "family": carrier.family,
            "port_type": carrier.port_type,
            "charts": [chart_commitment(chart) for chart in carrier.charts],
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
    raw_forward(carrier.charts, program, carrier.stats)
    carrier.state = "FORWARD_COMPLETE"


def project(carrier: Carrier, program: Program, owner: int) -> tuple[int, ...]:
    if carrier.state != "FORWARD_COMPLETE":
        fail("projection outside final-boundary stage")
    if carrier.active_owner != owner or carrier.active_program != program.fingerprint():
        fail("projection custody mismatch")
    return boundary_from_dual(carrier.charts, program)


def project_resident_port(_carrier: Carrier, _node: int) -> None:
    fail("resident exponent chart projection forbidden")


def inverse(carrier: Carrier, program: Program, owner: int) -> None:
    if carrier.state != "FORWARD_COMPLETE":
        fail("inverse outside forward-complete stage")
    if carrier.active_owner != owner or carrier.active_program != program.fingerprint():
        fail("inverse custody mismatch")
    raw_inverse(carrier.charts, program, carrier.stats)
    carrier.state = "RESTORED"
    carrier.active_owner = None
    carrier.active_program = None
    carrier.generation += 1
    carrier.restoration_generation = carrier.generation


def transaction(carrier: Carrier, program: Program) -> dict[str, Any]:
    before = carrier_payload_commitment(carrier)
    backing = carrier.backing_ids()
    begin_generation = carrier.generation
    begin_forward(carrier, program, program.owner)
    forward(carrier, program, program.owner)
    forward_commitment = charts_commitment(carrier.charts)
    active_coordinates = carrier.active_value_coordinates()
    forward_ranks = [chart.ranks() for chart in carrier.charts]
    boundary = project(carrier, program, program.owner)
    forward_stats = carrier.stats.descriptor()
    inverse(carrier, program, program.owner)
    restored = carrier_payload_commitment(carrier)
    if restored != before:
        fail("actual inverse did not restore exact carrier payload")
    if carrier.backing_ids() != backing:
        fail("carrier backing changed across transaction")
    if carrier.generation != begin_generation + 1:
        fail("restoration generation did not advance exactly once")
    return {
        "program_fingerprint": program.fingerprint(),
        "boundary_commitment": digest_json(list(boundary)),
        "forward_chart_commitment": forward_commitment,
        "forward_component_ranks": forward_ranks,
        "forward_active_value_coordinates": active_coordinates,
        "forward_stats": forward_stats,
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
    seed_ranks = [chart.ranks() for chart in carrier.charts]
    receipt = transaction(carrier, program)
    matrices, classical_stats = classical_forward(program)
    classical_boundary = boundary_from_dense(matrices, program)
    if digest_json(list(classical_boundary)) != receipt["boundary_commitment"]:
        fail("dual chart and matched dense recurrence disagree")
    phase_resident = carrier.resident_bytes() + len(LOG_TABLE) + len(POWERS)
    classical_resident = NODE_COUNT * interface * interface
    phase_output_buffer = 4 * (interface * interface + interface)
    classical_output_buffer = interface * interface
    phase_named_peak = (
        phase_resident
        + 8 * int(receipt["forward_stats"]["maximum_value_moment_cells"])
        + 8 * int(receipt["forward_stats"]["maximum_factorization_scratch_cells"])
        + 16 * interface
        + phase_output_buffer
        + len(canonical_json(program.descriptor()))
    )
    classical_named_peak = (
        classical_resident
        + 8 * classical_stats.maximum_control_rematerialization_cells
        + 8 * interface
        + classical_output_buffer
        + len(canonical_json(program.descriptor()))
    )
    return {
        "interface": interface,
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "boundary_commitment": receipt["boundary_commitment"],
        "forward_chart_commitment": receipt["forward_chart_commitment"],
        "seed_component_ranks": seed_ranks,
        "forward_component_ranks": receipt["forward_component_ranks"],
        "forward_active_value_coordinates": receipt["forward_active_value_coordinates"],
        "phase_forward_stats": receipt["forward_stats"],
        "classical_forward_stats": classical_stats.descriptor(),
        "phase_resident_bytes": phase_resident,
        "classical_resident_bytes": classical_resident,
        "phase_named_warm_peak_bytes": phase_named_peak,
        "classical_named_warm_peak_bytes": classical_named_peak,
        "phase_maximum_output_buffer_bytes": phase_output_buffer,
        "classical_maximum_output_buffer_bytes": classical_output_buffer,
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
    raw_forward(missing.charts, program)

    wrong = Carrier.fresh(5, "PRIMARY")
    raw_forward(wrong.charts, program)
    raw_inverse(wrong.charts, program, topology_mutation=1)

    reordered = Carrier.fresh(5, "PRIMARY")
    raw_forward(reordered.charts, program)
    raw_inverse(reordered.charts, program, inverse_order="COMPOSE_INTERSECT")

    normal = Carrier.fresh(5, "PRIMARY")
    raw_forward(normal.charts, program)
    normal_boundary = boundary_from_dual(normal.charts, program)

    disabled = Carrier.fresh(5, "PRIMARY")
    raw_forward(disabled.charts, program, port_enabled=False)

    swapped = Carrier.fresh(5, "PRIMARY")
    raw_forward(swapped.charts, program, action_order="INTERSECT_COMPOSE")

    mutated = Carrier.fresh(5, "PRIMARY")
    raw_forward(mutated.charts, program, topology_mutation=1)

    owner_carrier = Carrier.fresh(5, "PRIMARY")
    wrong_owner_rejected = expect_failure(
        lambda: begin_forward(owner_carrier, program, program.owner + 1)
    )
    type_rejected = expect_failure(
        lambda: begin_forward(
            Carrier.fresh(7, "PRIMARY"), program, program.owner
        )
    )
    null_rejected = expect_failure(lambda: begin_forward(None, program, program.owner))
    premature_rejected = expect_failure(
        lambda: project(Carrier.fresh(5, "PRIMARY"), program, program.owner)
    )
    resident_rejected = expect_failure(lambda: project_resident_port(normal, 0))

    reciprocal_exact = True
    for interface in INTERFACES:
        for family in FAMILIES:
            for hub in range(NODE_COUNT):
                peer = (hub + 1) % NODE_COUNT
                for row in range(interface):
                    for column in range(interface):
                        exponent = intersection_exponent(
                            interface, hub, peer, 0, family, row, column
                        )
                        reciprocal_exact &= (
                            POWERS[exponent]
                            * POWERS[(-exponent) % EXPONENT_MODULUS]
                            % FIELD
                            == 1
                        )
                left, right, coupling = composition_vectors(
                    interface, hub, peer, 0, family
                )
                pairing = int(np.dot(right, left) % FIELD)
                inverse_coupling = (
                    -coupling * pow((1 + coupling * pairing) % FIELD, -1, FIELD)
                ) % FIELD
                reciprocal_exact &= (
                    (coupling + inverse_coupling + coupling * inverse_coupling * pairing)
                    % FIELD
                    == 0
                )

    return {
        "missing_inverse_changes_payload": carrier_payload_commitment(missing) != seed_commitment,
        "wrong_inverse_changes_payload": carrier_payload_commitment(wrong) != seed_commitment,
        "reordered_inverse_changes_payload": carrier_payload_commitment(reordered) != seed_commitment,
        "null_carrier_rejected": null_rejected,
        "wrong_owner_rejected": wrong_owner_rejected,
        "wrong_type_rejected": type_rejected,
        "premature_projection_rejected": premature_rejected,
        "resident_chart_projection_rejected": resident_rejected,
        "disabled_port_changes_boundary": boundary_from_dual(disabled.charts, program) != normal_boundary,
        "composition_intersection_order_changes_boundary": boundary_from_dual(swapped.charts, program) != normal_boundary,
        "topology_mutation_changes_boundary": boundary_from_dual(mutated.charts, program) != normal_boundary,
        "intersection_and_composition_inverses_exact": reciprocal_exact,
        "generic_log_table_is_complete_and_zero_excluded": (
            LOG_TABLE[0] == -1
            and len({LOG_TABLE[value] for value in range(1, FIELD)}) == EXPONENT_MODULUS
        ),
    }


def reuse_results() -> tuple[dict[str, Any], dict[str, Any]]:
    first = compile_program(11, 2, "PRIMARY")
    second = compile_program(11, 7, "PRIMARY")
    reused = Carrier.fresh(11, "PRIMARY")
    backing = reused.backing_ids()
    first_receipt = transaction(reused, first)
    second_receipt = transaction(reused, second)
    fresh = Carrier.fresh(11, "PRIMARY")
    fresh_receipt = transaction(fresh, second)
    unrelated = {
        "first_boundary_commitment": first_receipt["boundary_commitment"],
        "second_boundary_matches_fresh": second_receipt["boundary_commitment"] == fresh_receipt["boundary_commitment"],
        "second_resource_signature_matches_fresh": second_receipt["forward_stats"] == fresh_receipt["forward_stats"],
        "same_backing_consumed": reused.backing_ids() == backing,
        "restoration_generation": reused.restoration_generation,
        "snapshot_used": False,
    }
    repeated_carrier = Carrier.fresh(5, "PRIMARY")
    repeated_program = compile_program(5, 3, "PRIMARY")
    repeated_backing = repeated_carrier.backing_ids()
    reference: str | None = None
    stable = True
    for _ in range(8):
        receipt = transaction(repeated_carrier, repeated_program)
        if reference is None:
            reference = receipt["boundary_commitment"]
        stable &= receipt["boundary_commitment"] == reference
        stable &= repeated_carrier.backing_ids() == repeated_backing
    repeated = {
        "cycles": 8,
        "boundary_stable": stable,
        "same_backing_stable": repeated_carrier.backing_ids() == repeated_backing,
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
            "dual exponent/value-moment controls failed: "
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
        fail("dual exponent/value-moment reuse failed")

    depth_one = [case for case in cases if case["depth"] == 1]
    rank_floor: dict[str, dict[str, int]] = {}
    rank_ceiling: dict[str, dict[str, int]] = {}
    seed_ceiling: dict[str, dict[str, int]] = {}
    for interface in INTERFACES:
        selected = [case for case in depth_one if case["interface"] == interface]
        converted_ranks: list[dict[str, int]] = []
        seed_ranks: list[dict[str, int]] = []
        for case in selected:
            hub = hub_index(0, case["family"])
            converted_ranks.extend(
                ranks
                for node, ranks in enumerate(case["forward_component_ranks"])
                if node != hub
            )
            seed_ranks.extend(case["seed_component_ranks"])
        rank_floor[str(interface)] = {
            component: min(ranks[component] for ranks in converted_ranks)
            for component in ("EXP_F2", "EXP_F3", "EXP_F17", "ZERO_F2")
        }
        rank_ceiling[str(interface)] = {
            component: max(ranks[component] for ranks in converted_ranks)
            for component in ("EXP_F2", "EXP_F3", "EXP_F17", "ZERO_F2")
        }
        seed_ceiling[str(interface)] = {
            component: max(ranks[component] for ranks in seed_ranks)
            for component in ("EXP_F2", "EXP_F3", "EXP_F17", "ZERO_F2")
        }

    near_full = all(
        rank_floor[str(interface)][component] >= interface - 2
        for interface in INTERFACES
        for component in ("EXP_F2", "EXP_F3", "EXP_F17")
    )
    seeds_compact = all(
        seed_ceiling[str(interface)][component] <= 2
        for interface in INTERFACES
        for component in ("EXP_F2", "EXP_F3", "EXP_F17")
    ) and all(seed_ceiling[str(interface)]["ZERO_F2"] == 0 for interface in INTERFACES)
    if not near_full or not seeds_compact:
        fail("declared seed/one-conversion rank law did not hold")

    phase_resident = {
        str(interface): next(
            case["phase_resident_bytes"]
            for case in cases
            if case["interface"] == interface
        )
        for interface in INTERFACES
    }
    classical_resident = {
        str(interface): next(
            case["classical_resident_bytes"]
            for case in cases
            if case["interface"] == interface
        )
        for interface in INTERFACES
    }
    resident_ratio = {
        str(interface): phase_resident[str(interface)] / classical_resident[str(interface)]
        for interface in INTERFACES
    }
    maximum_active_coordinates = {
        str(interface): max(
            case["forward_active_value_coordinates"]
            for case in cases
            if case["interface"] == interface
        )
        for interface in INTERFACES
    }
    depth8_conversion_ratio = {
        str(interface): {
            family: (
                next(
                    case["phase_forward_stats"]["phase_value_entry_evaluations"]
                    for case in cases
                    if case["interface"] == interface
                    and case["depth"] == 8
                    and case["family"] == family
                )
                / next(
                    case["classical_forward_stats"]["field_multiplications"]
                    for case in cases
                    if case["interface"] == interface
                    and case["depth"] == 8
                    and case["family"] == family
                )
            )
            for family in FAMILIES
        }
        for interface in INTERFACES
    }

    return {
        "schema": "CAT_CAS_F103_DUAL_EXPONENT_MOMENT_RELATION_NO_GO_RESULT_V1",
        "claim": CLAIM,
        "platform": "LINUX_DIRECT_PROCESS_SOFTWARE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "experiment": {
            "field": "F103",
            "primitive_phase_group": "C102",
            "exponent_crt_factors": list(CRT_MODULI),
            "interfaces": list(INTERFACES),
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "node_count": NODE_COUNT,
            "shared_unresolved_port": True,
            "consumers_per_layer": NODE_COUNT - 1,
            "native_intersection": "C102_EXPONENT_ADDITION",
            "composition_bridge": "ONE_STREAMED_F103_VALUE_MOMENT_PER_OUTPUT_COLUMN",
            "ordinary_dense_relation_table_materialized_by_phase_path": False,
            "generic_public_log_table_field_cells": len(LOG_TABLE),
            "generic_public_power_table_field_cells": len(POWERS),
            "compiler_inspects_final_answers": False,
        },
        "rank_law": {
            "seed_component_rank_ceiling": seed_ceiling,
            "depth1_converted_node_rank_floor": rank_floor,
            "depth1_converted_node_rank_ceiling": rank_ceiling,
            "all_seed_exponent_components_rank_at_most2": seeds_compact,
            "every_converted_exponent_component_rank_at_least_n_minus2": near_full,
            "uniform_fixed_rank_across_growing_interfaces": False,
            "fixed_backing_value_cell_law_per_relation": "4N2_PLUS4N",
            "maximum_active_factor_value_coordinates_by_interface": maximum_active_coordinates,
        },
        "cases": cases,
        "controls": controls,
        "restoration_and_reuse": {
            "actual_inverse_on_borrowed_carrier": True,
            "exact_payload_restoration": True,
            "same_backing_restoration": True,
            "snapshot_used": False,
            "retained_inverse_history_cells": 0,
            "retained_restoration_baseline_cells": 0,
            "unrelated_program_reuse": unrelated,
            "repeated_reuse": repeated,
        },
        "resource_accounting": {
            "phase_resident_bytes_by_interface": phase_resident,
            "classical_resident_bytes_by_interface": classical_resident,
            "phase_to_classical_resident_ratio_by_interface": resident_ratio,
            "depth8_phase_value_entry_evaluations_over_classical_field_multiplications": depth8_conversion_ratio,
            "phase_retained_dense_relation_table_cells": 0,
            "phase_retained_inverse_history_cells": 0,
            "phase_retained_restoration_baseline_cells": 0,
            "controller_backend_traffic_bytes": 0,
            "snapshot_traffic_bytes": 0,
            "public_algebra_tables_counted": True,
            "compiled_program_descriptor_counted": True,
            "moment_scratch_counted": True,
            "factorization_scratch_counted": True,
            "double_buffer_output_counted": True,
            "declared_numpy_int64_transient_arrays_counted_at8_bytes_per_cell": True,
            "python_object_container_allocator_native_library_internal_workspace_and_whole_process_peaks_excluded": True,
            "optimal_classical_recurrence_claimed": False,
        },
        "no_smuggle": {
            "raw_final_boundaries_serialized": False,
            "resident_exponent_charts_serialized": False,
            "ordinary_relation_tables_serialized": False,
            "assignment_or_truth_table_expansion": False,
            "boundary_commitments_only": True,
        },
        "claim_ceiling": "F103_DUAL_C102_EXPONENT_CRT_RANK_AND_STREAMED_VALUE_MOMENT_CHARTS_ON_DECLARED_C5_C7_C11_C17_NINE_NODE_ROTATING_HUB_FAMILIES_THROUGH_DEPTH8_IN_LINUX_DIRECT_PROCESS_SOFTWARE",
        "preserved_subclaims": [
            "HADAMARD_INTERSECTION_IS_NATIVE_C102_EXPONENT_ADDITION",
            "RANK1_LEFT_COMPOSITION_USES_ONE_STREAMED_VALUE_MOMENT_PER_COLUMN",
            "PHASE_PATH_RETAINS_NO_ORDINARY_DENSE_RELATION_TABLE",
            "EXACT_ALGEBRAIC_RESTORATION_AND_SAME_BACKING_REUSE",
            "EXECUTED_DENSE_F103_CLASSICAL_BOUNDARY_PARITY",
        ],
        "obstruction": "ONE_VALUE_SPACE_COMPOSITION_AND_LOG_CONVERSION_RAISES_EVERY_CRT_EXPONENT_COMPONENT_TO_AT_LEAST_N_MINUS2_ON_ALL_CONVERTED_DEPTH1_RELATIONS_WHILE_THE_FOUR_COMPONENT_FIXED_BACKING_IS_AT_LEAST4N2_AND_THE_EXECUTED_DENSE_F103_RECURRENCE_IS_SMALLER",
        "not_established": [
            "UNIFORM_FIXED_RANK_DUAL_EXPONENT_VALUE_MOMENT_CLOSURE",
            "SUB_DENSE_PHASE_CARRIER",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_COMPUTATION",
        ],
        "next_obstruction": "ADDITIVE_VALUE_MOMENT_TO_MULTIPLICATIVE_PHASE_EXPONENT_CONVERSION_GENERICALLY_DESTROYS_LOW_CRT_RANK_AND_REQUIRES_DENSE_EQUIVALENT_BACKING_SO_THE_NEXT_MACHINE_CHANGE_MUST_AVOID_GLOBAL_LOG_RECHARTING_OR_INTRODUCE_A_NONCLASSICAL_PHASE_NATIVE_COMPOSITION_LAW",
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
                    "rank_floor": result["rank_law"]["depth1_converted_node_rank_floor"],
                    "resident_ratio": result["resource_accounting"]["phase_to_classical_resident_ratio_by_interface"],
                    "obstruction": result["obstruction"],
                },
                sort_keys=True,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
