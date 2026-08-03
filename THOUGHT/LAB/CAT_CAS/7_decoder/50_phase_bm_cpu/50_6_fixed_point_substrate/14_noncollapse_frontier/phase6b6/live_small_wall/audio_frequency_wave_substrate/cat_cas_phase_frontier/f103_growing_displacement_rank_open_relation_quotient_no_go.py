#!/usr/bin/env python3
"""Exact growing-interface cyclic-displacement relation quotient diagnostic.

For a relation A on C_n define the cyclic displacement

    D(A)[x,y] = A[x,y] - A[x-1,y-1].

A is represented by one boundary row and an exact canonical rank factorization
of D(A).  Entry access, composition, intersection, rotation, projection, and
inverse execution stream through that chart; the accepted phase path never
constructs an n by n relation table.  The experiment asks whether displacement
rank stays uniformly bounded for n in {5, 7, 11, 17}.  An executed compact
classical comparison uses the same chart while it is smaller and switches to a
dense recurrence when the chart ceases to be compact.

This is bounded Linux direct-process software.  It is not CATVM custody, a
physical waveform experiment, or evidence of a distinct phase resource.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


MODULUS = 103
INTERFACES = (5, 7, 11, 17)
DEPTHS = (1, 2, 4, 8, 16)
FAMILIES = ("PRIMARY", "ALTERNATE")
NODE_COUNT = 9
CONTROL_RANK = 2
CLAIM = (
    "BOUNDED_EXACT_F103_GROWING_C5_C7_C11_C17_CYCLIC_DISPLACEMENT_"
    "RANK_OPEN_RELATION_CHART_STREAMS_FULL_ORDINARY_RANK_SEEDS_FROM_"
    "RANK2_GENERATORS_WITHOUT_DENSE_RELATION_TABLES_BUT_ONE_"
    "NONCOMMUTING_COMPOSITION_INTERSECTION_LAYER_SATURATES_"
    "DISPLACEMENT_RANK_N_AT_EVERY_INTERFACE_THE_CHART_BECOMES_"
    "N2_PLUS_N_DENSE_EQUIVALENT_AND_AN_EXECUTED_DENSE_FALLBACK_"
    "CLASSICAL_RECURRENCE_IS_SMALLER_WITH_EXACT_RESTORATION_AND_REUSE"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def mod(value: Any) -> Any:
    return np.mod(value, MODULUS).astype(np.int64, copy=False)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


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
            "schema": "CAT_CAS_F103_GROWING_DISPLACEMENT_RELATION_PROGRAM_V1",
            "interface": self.interface,
            "depth": self.depth,
            "family": self.family,
            "owner": self.owner,
            "node_count": NODE_COUNT,
            "port_type": f"F103_CYCLIC_DISPLACEMENT_C{self.interface}_TO_C{self.interface}",
            "topology": "PUBLIC_ROTATING_CONTROL_HUB8",
            "composition": "LEFT_ACTION_BY_IDENTITY_PLUS_RANK2",
            "intersection": "HADAMARD_WITH_RECIPROCAL_RANK2_CONTROL",
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
        owner=(0xD15A0000 + 257 * interface + 131 * depth + code) & 0xFFFFFFFF,
        observation_left=(7 * depth + 3 * code + interface) % interface,
        observation_right=(11 * depth + 5 * code + 2 * interface) % interface,
    )


@dataclass
class WorkStats:
    entry_evaluations: int = 0
    canonicalizations: int = 0
    composition_actions: int = 0
    intersection_actions: int = 0
    rotations: int = 0
    consumers: int = 0
    field_multiplications: int = 0
    field_inversions: int = 0
    elimination_field_multiplications: int = 0
    maximum_displacement_rank: int = 0
    maximum_raw_chart_value_coordinates: int = 0
    maximum_declared_scratch_field_coordinates: int = 0
    rank_histogram: dict[int, int] = field(default_factory=dict)

    def observe_rank(self, interface: int, rank: int) -> None:
        self.maximum_displacement_rank = max(self.maximum_displacement_rank, rank)
        self.maximum_raw_chart_value_coordinates = max(
            self.maximum_raw_chart_value_coordinates,
            interface + rank * (2 * interface - rank),
        )
        self.rank_histogram[rank] = self.rank_histogram.get(rank, 0) + 1

    def descriptor(self) -> dict[str, Any]:
        return {
            "entry_evaluations": self.entry_evaluations,
            "canonicalizations": self.canonicalizations,
            "composition_actions": self.composition_actions,
            "intersection_actions": self.intersection_actions,
            "rotations": self.rotations,
            "consumers": self.consumers,
            "field_multiplications": self.field_multiplications,
            "field_inversions": self.field_inversions,
            "elimination_field_multiplications": self.elimination_field_multiplications,
            "maximum_displacement_rank": self.maximum_displacement_rank,
            "maximum_raw_chart_value_coordinates": self.maximum_raw_chart_value_coordinates,
            "maximum_declared_scratch_field_coordinates": self.maximum_declared_scratch_field_coordinates,
            "rank_histogram": {
                str(key): self.rank_histogram[key]
                for key in sorted(self.rank_histogram)
            },
        }


def inverse_mod(matrix: np.ndarray, stats: WorkStats | None = None) -> np.ndarray:
    size = matrix.shape[0]
    if matrix.shape != (size, size):
        fail("inverse requires square matrix")
    work = np.concatenate(
        (mod(matrix.copy()), np.eye(size, dtype=np.int64)), axis=1
    )
    width = 2 * size
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if int(work[row, column])), None
        )
        if pivot is None:
            fail("singular modular matrix")
        if pivot != column:
            work[[column, pivot]] = work[[pivot, column]]
        scale = pow(int(work[column, column]), -1, MODULUS)
        work[column] = mod(work[column] * scale)
        if stats is not None:
            stats.field_inversions += 1
            stats.elimination_field_multiplications += width
        for row in range(size):
            if row == column or not int(work[row, column]):
                continue
            factor = int(work[row, column])
            work[row] = mod(work[row] - factor * work[column])
            if stats is not None:
                stats.elimination_field_multiplications += width
    return work[:, size:]


def streamed_rref_basis(
    interface: int,
    entry: Callable[[int, int], int],
    stats: WorkStats | None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return canonical U,V with entry(x,y) == U[x] dot V[y]."""
    basis: dict[int, np.ndarray] = {}
    for column in range(interface):
        vector = np.array(
            [entry(row, column) % MODULUS for row in range(interface)],
            dtype=np.int64,
        )
        for pivot in sorted(basis):
            factor = int(vector[pivot])
            if factor:
                vector = mod(vector - factor * basis[pivot])
                if stats is not None:
                    stats.elimination_field_multiplications += interface
        nonzero = np.flatnonzero(vector)
        if not nonzero.size:
            continue
        pivot = int(nonzero[0])
        scale = pow(int(vector[pivot]), -1, MODULUS)
        vector = mod(vector * scale)
        if stats is not None:
            stats.field_inversions += 1
            stats.elimination_field_multiplications += interface
        for existing in list(basis):
            factor = int(basis[existing][pivot])
            if factor:
                basis[existing] = mod(basis[existing] - factor * vector)
                if stats is not None:
                    stats.elimination_field_multiplications += interface
        basis[pivot] = vector
    pivots = sorted(basis)
    rank = len(pivots)
    if rank == 0:
        return (
            np.empty((interface, 0), dtype=np.int64),
            np.empty((interface, 0), dtype=np.int64),
            [],
        )
    left = np.stack([basis[pivot] for pivot in pivots], axis=1)
    right = np.empty((interface, rank), dtype=np.int64)
    for column in range(interface):
        right[column] = np.array(
            [entry(pivot, column) % MODULUS for pivot in pivots], dtype=np.int64
        )
    if not np.array_equal(left[pivots, :], np.eye(rank, dtype=np.int64)):
        fail("streamed canonical displacement gauge failed")
    for column in range(interface):
        for row in range(interface):
            if int(np.dot(left[row], right[column]) % MODULUS) != entry(row, column) % MODULUS:
                fail("streamed displacement reconstruction failed")
    if stats is not None:
        stats.maximum_declared_scratch_field_coordinates = max(
            stats.maximum_declared_scratch_field_coordinates,
            3 * interface * rank + 3 * interface,
        )
    return left, right, pivots


@dataclass
class Chart:
    payload: np.ndarray
    pivots: np.ndarray
    rank: int

    def copy(self) -> "Chart":
        return Chart(self.payload.copy(), self.pivots.copy(), self.rank)


def overwrite_chart(target: Chart, source: Chart) -> None:
    """Replace logical chart state without replacing borrowed backing arrays."""
    if target.payload.shape != source.payload.shape or target.pivots.shape != source.pivots.shape:
        fail("chart backing shape changed during in-place transaction")
    np.copyto(target.payload, source.payload)
    np.copyto(target.pivots, source.pivots)
    target.rank = source.rank


def payload_capacity(interface: int) -> int:
    return interface + interface * interface


def chart_value_coordinates(interface: int, rank: int) -> int:
    return interface + rank * (2 * interface - rank)


def pack_chart(
    interface: int,
    boundary: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    pivots: list[int],
) -> Chart:
    rank = left.shape[1]
    if boundary.shape != (interface,) or left.shape != right.shape:
        fail("invalid displacement chart shape")
    payload = np.zeros(payload_capacity(interface), dtype=np.uint8)
    pivot_payload = np.full(interface, 255, dtype=np.uint8)
    payload[:interface] = mod(boundary).astype(np.uint8)
    if rank:
        nonpivots = [row for row in range(interface) if row not in pivots]
        values = np.concatenate((left[nonpivots].reshape(-1), right.reshape(-1)))
        expected = rank * (2 * interface - rank)
        if values.size != expected:
            fail("displacement payload coordinate mismatch")
        payload[interface : interface + expected] = mod(values).astype(np.uint8)
        pivot_payload[:rank] = np.array(pivots, dtype=np.uint8)
    return Chart(payload, pivot_payload, rank)


def unpack_chart(interface: int, chart: Chart) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rank = int(chart.rank)
    if not 0 <= rank <= interface:
        fail("displacement rank outside interface")
    boundary = chart.payload[:interface].astype(np.int64)
    if rank == 0:
        return (
            boundary,
            np.empty((interface, 0), dtype=np.int64),
            np.empty((interface, 0), dtype=np.int64),
        )
    pivots = [int(value) for value in chart.pivots[:rank]]
    if len(set(pivots)) != rank or any(not 0 <= value < interface for value in pivots):
        fail("invalid displacement pivot metadata")
    nonpivots = [row for row in range(interface) if row not in pivots]
    left_count = (interface - rank) * rank
    total = rank * (2 * interface - rank)
    values = chart.payload[interface : interface + total].astype(np.int64)
    left = np.zeros((interface, rank), dtype=np.int64)
    left[pivots] = np.eye(rank, dtype=np.int64)
    left[nonpivots] = values[:left_count].reshape(interface - rank, rank)
    right = values[left_count:].reshape(interface, rank)
    return boundary, left, right


def factor_entry(left: np.ndarray, right: np.ndarray, x: int, y: int) -> int:
    return int(np.dot(left[x], right[y]) % MODULUS)


def chart_entry_from_parts(
    interface: int,
    boundary: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    x: int,
    y: int,
    stats: WorkStats | None = None,
) -> int:
    if stats is not None:
        stats.entry_evaluations += 1
    diagonal = (y - x) % interface
    value = int(boundary[diagonal])
    for row in range(1, x + 1):
        column = (diagonal + row) % interface
        value += factor_entry(left, right, row, column)
        if stats is not None:
            stats.field_multiplications += left.shape[1]
    return value % MODULUS


def chart_entry(
    interface: int, chart: Chart, x: int, y: int, stats: WorkStats | None = None
) -> int:
    return chart_entry_from_parts(interface, *unpack_chart(interface, chart), x, y, stats)


def chart_from_entry(
    interface: int,
    entry: Callable[[int, int], int],
    stats: WorkStats | None,
) -> Chart:
    boundary = np.array([entry(0, column) for column in range(interface)], dtype=np.int64)

    def displacement(row: int, column: int) -> int:
        return (entry(row, column) - entry((row - 1) % interface, (column - 1) % interface)) % MODULUS

    left, right, pivots = streamed_rref_basis(interface, displacement, stats)
    if stats is not None:
        stats.canonicalizations += 1
        stats.observe_rank(interface, left.shape[1])
    return pack_chart(interface, boundary, left, right, pivots)


def streamed_rank(interface: int, entry: Callable[[int, int], int]) -> int:
    left, _right, _pivots = streamed_rref_basis(interface, entry, None)
    return left.shape[1]


def target_entry(interface: int, node: int, x: int, y: int) -> int:
    diagonal = (y - x) % interface
    base = (3 + 5 * node + 7 * diagonal + 11 * diagonal * diagonal) % MODULUS
    left = (2 + 7 * node + 5 * x + node * x) % MODULUS
    right = (5 + 13 * node + 3 * y + 2 * node * y) % MODULUS
    return (base + left * right) % MODULUS


def seed_charts(interface: int) -> list[Chart]:
    return [
        chart_from_entry(
            interface,
            lambda x, y, node=node: target_entry(interface, node, x, y),
            None,
        )
        for node in range(NODE_COUNT)
    ]


def control_factors(interface: int, node: int) -> np.ndarray:
    q = (7 + 9 * node) % MODULUS
    if q in (0, MODULUS - 1):
        q = 3
    a = np.array(
        [1 + ((11 * node + 7 * x + 3 * node * x) % 101) for x in range(interface)],
        dtype=np.int64,
    )
    b = np.array(
        [1 + ((13 * node + 5 * y + 4 * node * y) % 101) for y in range(interface)],
        dtype=np.int64,
    )
    selector_left = np.array(
        [1 if ((x + 2 * node) % 5) < 2 else 0 for x in range(interface)],
        dtype=np.int64,
    )
    selector_right = np.array(
        [1 if ((3 * y + node) % 7) < 3 else 0 for y in range(interface)],
        dtype=np.int64,
    )
    left = np.stack((a, mod(a * selector_left)), axis=1)
    right = np.stack((b, mod(b * q * selector_right)), axis=1)
    for x in range(interface):
        for y in range(interface):
            if factor_entry(left, right, x, y) == 0:
                fail("structured rank-two control contains zero")
    return np.stack((left, right), axis=0)


def seed_controls(interface: int) -> np.ndarray:
    return np.stack(
        [control_factors(interface, node) for node in range(NODE_COUNT)], axis=0
    ).astype(np.uint8)


def reciprocal_control(control: np.ndarray) -> np.ndarray:
    left = control[0].astype(np.int64)
    right = control[1].astype(np.int64)
    a = left[:, 0]
    b = right[:, 0]
    inverse_a = np.array([pow(int(value), -1, MODULUS) for value in a], dtype=np.int64)
    inverse_b = np.array([pow(int(value), -1, MODULUS) for value in b], dtype=np.int64)
    selector_left = mod(left[:, 1] * inverse_a)
    selector_scaled = mod(right[:, 1] * inverse_b)
    nonzero = sorted({int(value) for value in selector_scaled if int(value)})
    if len(nonzero) != 1:
        fail("control lost reciprocal rank-two structure")
    q = nonzero[0]
    coefficient = (-pow(1 + q, -1, MODULUS)) % MODULUS
    reciprocal_left = np.stack((inverse_a, mod(inverse_a * selector_left)), axis=1)
    reciprocal_right = np.stack(
        (inverse_b, mod(inverse_b * coefficient * selector_scaled)), axis=1
    )
    return np.stack((reciprocal_left, reciprocal_right), axis=0)


def nonsingular_coupling(control: np.ndarray) -> int:
    left = control[0].astype(np.int64)
    right = control[1].astype(np.int64)
    pairing = mod(right.T @ left)
    for coupling in range(1, MODULUS):
        kernel = mod(np.eye(CONTROL_RANK, dtype=np.int64) + coupling * pairing)
        determinant = (
            int(kernel[0, 0]) * int(kernel[1, 1])
            - int(kernel[0, 1]) * int(kernel[1, 0])
        ) % MODULUS
        if determinant:
            return coupling
    fail("no nonsingular identity-plus-rank-two coupling")


def hub_index(index: int, family: str, mutation: int = 0) -> int:
    return (5 * index + family_code(family) + mutation) % NODE_COUNT


def peer_order(hub: int) -> list[int]:
    return [(hub + offset) % NODE_COUNT for offset in range(1, NODE_COUNT)]


def relation_offset(
    interface: int,
    hub: int,
    peer: int,
    index: int,
    family: str,
    mutation: int = 0,
) -> int:
    return (
        7 * hub + 11 * peer + 3 * index + family_code(family) + mutation
    ) % interface


def rotation_shift(interface: int, node: int, index: int, family: str) -> int:
    return (
        3 * node * node
        + 5 * index
        + family_code(family) * (1 + index.bit_count())
    ) % interface


def control_view(
    controls: np.ndarray,
    interface: int,
    hub: int,
    peer: int,
    index: int,
    family: str,
    mutation: int = 0,
) -> np.ndarray:
    offset = relation_offset(interface, hub, peer, index, family, mutation)
    result = controls[hub].astype(np.int64).copy()
    result[0] = np.roll(result[0], offset, axis=0)
    result[1] = np.roll(result[1], -offset, axis=0)
    return result


def rotate_chart(
    interface: int, chart: Chart, shift: int, stats: WorkStats | None
) -> Chart:
    boundary, left, right = unpack_chart(interface, chart)

    def old(x: int, y: int) -> int:
        return chart_entry_from_parts(interface, boundary, left, right, x, y, stats)

    result = chart_from_entry(
        interface,
        lambda x, y: old((x - shift) % interface, (y - shift) % interface),
        stats,
    )
    if stats is not None:
        stats.rotations += 1
    return result


def compose_chart(
    interface: int,
    chart: Chart,
    control: np.ndarray,
    *,
    inverse: bool,
    stats: WorkStats | None,
) -> Chart:
    boundary, left, right = unpack_chart(interface, chart)
    control_left = control[0].astype(np.int64)
    control_right = control[1].astype(np.int64)
    coupling = nonsingular_coupling(control)

    def old(x: int, y: int) -> int:
        return chart_entry_from_parts(interface, boundary, left, right, x, y, stats)

    contraction = np.empty((CONTROL_RANK, interface), dtype=np.int64)
    for component in range(CONTROL_RANK):
        for y in range(interface):
            value = 0
            for x in range(interface):
                value += int(control_right[x, component]) * old(x, y)
            contraction[component, y] = value % MODULUS
            if stats is not None:
                stats.field_multiplications += interface
    if inverse:
        pairing = mod(control_right.T @ control_left)
        kernel = mod(np.eye(CONTROL_RANK, dtype=np.int64) + coupling * pairing)
        effective_left = mod(
            -coupling * control_left @ inverse_mod(kernel, stats)
        )
        if stats is not None:
            stats.field_multiplications += (
                CONTROL_RANK * CONTROL_RANK * interface
                + CONTROL_RANK * CONTROL_RANK * CONTROL_RANK
            )
    else:
        effective_left = mod(coupling * control_left)

    def updated(x: int, y: int) -> int:
        correction = int(np.dot(effective_left[x], contraction[:, y]))
        if stats is not None:
            stats.field_multiplications += CONTROL_RANK
        return (old(x, y) + correction) % MODULUS

    result = chart_from_entry(interface, updated, stats)
    if stats is not None:
        stats.composition_actions += 1
        stats.maximum_declared_scratch_field_coordinates = max(
            stats.maximum_declared_scratch_field_coordinates,
            2 * interface + 2 * CONTROL_RANK * interface,
        )
    return result


def intersect_chart(
    interface: int,
    chart: Chart,
    control: np.ndarray,
    *,
    inverse: bool,
    stats: WorkStats | None,
) -> Chart:
    factor = reciprocal_control(control) if inverse else control
    boundary, left, right = unpack_chart(interface, chart)
    control_left = factor[0].astype(np.int64)
    control_right = factor[1].astype(np.int64)

    def updated(x: int, y: int) -> int:
        old = chart_entry_from_parts(interface, boundary, left, right, x, y, stats)
        value = factor_entry(control_left, control_right, x, y)
        if stats is not None:
            stats.field_multiplications += CONTROL_RANK
        return old * value % MODULUS

    result = chart_from_entry(interface, updated, stats)
    if stats is not None:
        stats.intersection_actions += 1
    return result


def raw_forward(
    controls: np.ndarray,
    charts: list[Chart],
    program: Program,
    *,
    action_order: str = "COMPOSE_INTERSECT",
    port_enabled: bool = True,
    hub_mutation: int = 0,
    offset_mutation: int = 0,
    stats: WorkStats | None = None,
) -> None:
    actions = ("COMPOSE", "INTERSECT") if action_order == "COMPOSE_INTERSECT" else ("INTERSECT", "COMPOSE")
    interface = program.interface
    for index in range(program.depth):
        for node in range(NODE_COUNT):
            overwrite_chart(
                charts[node],
                rotate_chart(
                    interface,
                    charts[node],
                    rotation_shift(interface, node, index, program.family),
                    stats,
                ),
            )
        hub = hub_index(index, program.family, hub_mutation)
        for peer in peer_order(hub):
            if not port_enabled:
                continue
            control = control_view(
                controls,
                interface,
                hub,
                peer,
                index,
                program.family,
                offset_mutation,
            )
            for action in actions:
                if action == "COMPOSE":
                    overwrite_chart(
                        charts[peer],
                        compose_chart(
                            interface,
                            charts[peer],
                            control,
                            inverse=False,
                            stats=stats,
                        ),
                    )
                else:
                    overwrite_chart(
                        charts[peer],
                        intersect_chart(
                            interface,
                            charts[peer],
                            control,
                            inverse=False,
                            stats=stats,
                        ),
                    )
            if stats is not None:
                stats.consumers += 1


def raw_inverse(
    controls: np.ndarray,
    charts: list[Chart],
    program: Program,
    *,
    assumed_action_order: str = "COMPOSE_INTERSECT",
    offset_mutation: int = 0,
    stats: WorkStats | None = None,
) -> None:
    actions = ("INTERSECT", "COMPOSE") if assumed_action_order == "COMPOSE_INTERSECT" else ("COMPOSE", "INTERSECT")
    interface = program.interface
    for index in reversed(range(program.depth)):
        hub = hub_index(index, program.family)
        for peer in reversed(peer_order(hub)):
            control = control_view(
                controls,
                interface,
                hub,
                peer,
                index,
                program.family,
                offset_mutation,
            )
            for action in actions:
                if action == "INTERSECT":
                    overwrite_chart(
                        charts[peer],
                        intersect_chart(
                            interface,
                            charts[peer],
                            control,
                            inverse=True,
                            stats=stats,
                        ),
                    )
                else:
                    overwrite_chart(
                        charts[peer],
                        compose_chart(
                            interface,
                            charts[peer],
                            control,
                            inverse=True,
                            stats=stats,
                        ),
                    )
        for node in reversed(range(NODE_COUNT)):
            overwrite_chart(
                charts[node],
                rotate_chart(
                    interface,
                    charts[node],
                    -rotation_shift(interface, node, index, program.family),
                    stats,
                ),
            )


def boundary_from_charts(charts: list[Chart], program: Program) -> tuple[int, ...]:
    interface = program.interface
    values: list[int] = []
    for coordinate in range(interface):
        value = 0
        for node, chart in enumerate(charts):
            x = (program.observation_left + coordinate + 3 * node) % interface
            y = (program.observation_right + 2 * coordinate + 5 * node) % interface
            value += (1 + node + coordinate * coordinate) * chart_entry(
                interface, chart, x, y
            )
        values.append(value % MODULUS)
    return tuple(values)


def charts_commitment(charts: list[Chart]) -> str:
    digest = hashlib.sha256()
    for chart in charts:
        digest.update(chart.payload.tobytes())
        digest.update(chart.pivots.tobytes())
        digest.update(bytes([chart.rank]))
    return digest.hexdigest()


@dataclass
class HybridNode:
    mode: str
    chart: Chart
    dense: np.ndarray


@dataclass
class ClassicalStats:
    chart_stats: WorkStats = field(default_factory=WorkStats)
    dense_composition_multiplications: int = 0
    dense_intersection_multiplications: int = 0
    chart_to_dense_switches: int = 0
    maximum_resident_value_coordinates: int = 0
    maximum_resident_metadata_bytes: int = 0
    maximum_control_rematerialization_coordinates: int = 0
    maximum_dense_scratch_coordinates: int = 0

    def descriptor(self) -> dict[str, Any]:
        return {
            "chart_stats": self.chart_stats.descriptor(),
            "dense_composition_multiplications": self.dense_composition_multiplications,
            "dense_intersection_multiplications": self.dense_intersection_multiplications,
            "chart_to_dense_switches": self.chart_to_dense_switches,
            "maximum_resident_value_coordinates": self.maximum_resident_value_coordinates,
            "maximum_resident_metadata_bytes": self.maximum_resident_metadata_bytes,
            "maximum_control_rematerialization_coordinates": self.maximum_control_rematerialization_coordinates,
            "maximum_dense_scratch_coordinates": self.maximum_dense_scratch_coordinates,
        }


def hybrid_seed(interface: int) -> list[HybridNode]:
    return [
        HybridNode("CHART", chart, np.zeros((interface, interface), dtype=np.uint8))
        for chart in seed_charts(interface)
    ]


def hybrid_account(interface: int, nodes: list[HybridNode], stats: ClassicalStats) -> None:
    values = 0
    metadata = 0
    for node in nodes:
        if node.mode == "CHART":
            values += chart_value_coordinates(interface, node.chart.rank)
            metadata += node.chart.rank + 1
        else:
            values += interface * interface
            metadata += 1
    stats.maximum_resident_value_coordinates = max(
        stats.maximum_resident_value_coordinates, values
    )
    stats.maximum_resident_metadata_bytes = max(
        stats.maximum_resident_metadata_bytes, metadata
    )


def hybrid_maybe_switch(interface: int, node: HybridNode, stats: ClassicalStats) -> None:
    if node.mode != "CHART":
        return
    if chart_value_coordinates(interface, node.chart.rank) < interface * interface:
        return
    dense = np.empty((interface, interface), dtype=np.uint8)
    for x in range(interface):
        for y in range(interface):
            dense[x, y] = chart_entry(interface, node.chart, x, y)
    node.mode = "DENSE"
    node.dense = dense
    node.chart = Chart(
        np.zeros(payload_capacity(interface), dtype=np.uint8),
        np.full(interface, 255, dtype=np.uint8),
        0,
    )
    stats.chart_to_dense_switches += 1
    stats.maximum_dense_scratch_coordinates = max(
        stats.maximum_dense_scratch_coordinates, interface * interface
    )


def hybrid_rotate(
    interface: int, node: HybridNode, shift: int, stats: ClassicalStats
) -> None:
    if node.mode == "CHART":
        node.chart = rotate_chart(interface, node.chart, shift, stats.chart_stats)
        hybrid_maybe_switch(interface, node, stats)
    else:
        node.dense = np.roll(np.roll(node.dense, shift, axis=0), shift, axis=1)


def hybrid_apply(
    interface: int,
    node: HybridNode,
    control: np.ndarray,
    action: str,
    stats: ClassicalStats,
) -> None:
    if node.mode == "CHART":
        if action == "COMPOSE":
            node.chart = compose_chart(
                interface, node.chart, control, inverse=False, stats=stats.chart_stats
            )
        else:
            node.chart = intersect_chart(
                interface, node.chart, control, inverse=False, stats=stats.chart_stats
            )
        hybrid_maybe_switch(interface, node, stats)
        return
    dense = node.dense.astype(np.int64)
    if action == "COMPOSE":
        coupling = nonsingular_coupling(control)
        contraction = mod(control[1].T.astype(np.int64) @ dense)
        node.dense = mod(
            dense + coupling * control[0].astype(np.int64) @ contraction
        ).astype(np.uint8)
        stats.dense_composition_multiplications += (
            2 * CONTROL_RANK * interface * interface
        )
        stats.maximum_dense_scratch_coordinates = max(
            stats.maximum_dense_scratch_coordinates,
            2 * interface * interface + CONTROL_RANK * interface,
        )
    else:
        updated = np.empty_like(dense)
        for x in range(interface):
            for y in range(interface):
                updated[x, y] = (
                    dense[x, y]
                    * factor_entry(control[0], control[1], x, y)
                ) % MODULUS
        node.dense = updated.astype(np.uint8)
        stats.dense_intersection_multiplications += (
            (CONTROL_RANK + 1) * interface * interface
        )
        stats.maximum_dense_scratch_coordinates = max(
            stats.maximum_dense_scratch_coordinates, 2 * interface * interface
        )


def hybrid_forward(program: Program) -> tuple[list[HybridNode], ClassicalStats]:
    interface = program.interface
    nodes = hybrid_seed(interface)
    stats = ClassicalStats()
    hybrid_account(interface, nodes, stats)
    for index in range(program.depth):
        for node_index, node in enumerate(nodes):
            hybrid_rotate(
                interface,
                node,
                rotation_shift(interface, node_index, index, program.family),
                stats,
            )
        hub = hub_index(index, program.family)
        for peer in peer_order(hub):
            control = control_view(
                seed_controls(interface),
                interface,
                hub,
                peer,
                index,
                program.family,
            )
            stats.maximum_control_rematerialization_coordinates = max(
                stats.maximum_control_rematerialization_coordinates,
                4 * interface,
            )
            hybrid_apply(interface, nodes[peer], control, "COMPOSE", stats)
            hybrid_apply(interface, nodes[peer], control, "INTERSECT", stats)
        hybrid_account(interface, nodes, stats)
    return nodes, stats


def hybrid_entry(interface: int, node: HybridNode, x: int, y: int) -> int:
    if node.mode == "DENSE":
        return int(node.dense[x, y])
    return chart_entry(interface, node.chart, x, y)


def boundary_from_hybrid(nodes: list[HybridNode], program: Program) -> tuple[int, ...]:
    interface = program.interface
    values: list[int] = []
    for coordinate in range(interface):
        value = 0
        for node_index, node in enumerate(nodes):
            x = (program.observation_left + coordinate + 3 * node_index) % interface
            y = (program.observation_right + 2 * coordinate + 5 * node_index) % interface
            value += (1 + node_index + coordinate * coordinate) * hybrid_entry(
                interface, node, x, y
            )
        values.append(value % MODULUS)
    return tuple(values)


def streamed_parity(
    interface: int, charts: list[Chart], hybrid: list[HybridNode]
) -> int:
    checks = 0
    for node in range(NODE_COUNT):
        for x in range(interface):
            for y in range(interface):
                if chart_entry(interface, charts[node], x, y) != hybrid_entry(
                    interface, hybrid[node], x, y
                ):
                    fail("displacement chart differs from matched recurrence")
                checks += 1
    return checks


@dataclass
class Carrier:
    interface: int
    controls: np.ndarray
    charts: list[Chart]
    port_type: str
    leased: bool = False
    lease_owner: int | None = None
    lease_program: str | None = None
    stage: str = "RESTORED"
    restoration_generation: int = 0
    projection_calls: int = 0
    snapshot_reload_used: bool = False
    stats: WorkStats = field(default_factory=WorkStats)
    inverse_stats: WorkStats = field(default_factory=WorkStats)

    @classmethod
    def seal(cls, interface: int) -> "Carrier":
        return cls(
            interface=interface,
            controls=seed_controls(interface),
            charts=seed_charts(interface),
            port_type=f"F103_CYCLIC_DISPLACEMENT_C{interface}_TO_C{interface}",
        )

    @property
    def backing_identity(self) -> tuple[int, ...]:
        return (
            int(self.controls.__array_interface__["data"][0]),
            *(
                value
                for chart in self.charts
                for value in (
                    int(chart.payload.__array_interface__["data"][0]),
                    int(chart.pivots.__array_interface__["data"][0]),
                )
            ),
        )


def carrier_commitment(carrier: Carrier) -> str:
    digest = hashlib.sha256()
    digest.update(carrier.controls.tobytes())
    digest.update(charts_commitment(carrier.charts).encode("ascii"))
    digest.update(carrier.port_type.encode("ascii"))
    digest.update(carrier.stage.encode("ascii"))
    digest.update(bytes([1 if carrier.leased else 0]))
    return digest.hexdigest()


def begin_forward(carrier: Carrier | None, program: Program, owner: int) -> None:
    if carrier is None:
        fail("null carrier")
    if carrier.interface != program.interface:
        fail("program interface does not match carrier")
    expected = f"F103_CYCLIC_DISPLACEMENT_C{program.interface}_TO_C{program.interface}"
    if carrier.port_type != expected:
        fail("wrong relation type")
    if carrier.leased or carrier.stage != "RESTORED":
        fail("carrier is not restored")
    if owner != program.owner:
        fail("wrong lease owner")
    carrier.leased = True
    carrier.lease_owner = owner
    carrier.lease_program = program.fingerprint()
    carrier.stage = "FORWARD_ACTIVE"
    carrier.stats = WorkStats()
    carrier.inverse_stats = WorkStats()


def forward(carrier: Carrier, program: Program, owner: int) -> None:
    if (
        not carrier.leased
        or carrier.lease_owner != owner
        or carrier.lease_program != program.fingerprint()
        or carrier.stage != "FORWARD_ACTIVE"
    ):
        fail("forward lease mismatch")
    raw_forward(carrier.controls, carrier.charts, program, stats=carrier.stats)
    carrier.stage = "FORWARD_COMPLETE"


def project(carrier: Carrier, program: Program, owner: int) -> tuple[int, ...]:
    if (
        carrier.stage != "FORWARD_COMPLETE"
        or not carrier.leased
        or carrier.lease_owner != owner
    ):
        fail("projection before final boundary")
    carrier.projection_calls += 1
    return boundary_from_charts(carrier.charts, program)


def project_resident_port(_carrier: Carrier, _node: int) -> None:
    fail("resident relation projection forbidden")


def inverse(carrier: Carrier, program: Program, owner: int) -> None:
    if (
        carrier.stage != "FORWARD_COMPLETE"
        or not carrier.leased
        or carrier.lease_owner != owner
        or carrier.lease_program != program.fingerprint()
    ):
        fail("inverse lease mismatch")
    raw_inverse(
        carrier.controls,
        carrier.charts,
        program,
        stats=carrier.inverse_stats,
    )
    carrier.leased = False
    carrier.lease_owner = None
    carrier.lease_program = None
    carrier.stage = "RESTORED"
    carrier.restoration_generation += 1


def transaction(carrier: Carrier, program: Program) -> dict[str, Any]:
    initial = carrier_commitment(carrier)
    backing = carrier.backing_identity
    generation = carrier.restoration_generation
    begin_forward(carrier, program, program.owner)
    forward(carrier, program, program.owner)
    final_charts = [chart.copy() for chart in carrier.charts]
    boundary = project(carrier, program, program.owner)
    final_commitment = charts_commitment(carrier.charts)
    forward_stats = carrier.stats.descriptor()
    final_ranks = [chart.rank for chart in carrier.charts]
    inverse(carrier, program, program.owner)
    return {
        "boundary": list(boundary),
        "final_charts": final_charts,
        "final_commitment": final_commitment,
        "final_displacement_ranks": final_ranks,
        "exact_restoration": carrier_commitment(carrier) == initial,
        "same_backing": carrier.backing_identity == backing,
        "generation_before": generation,
        "generation_after": carrier.restoration_generation,
        "projection_calls": carrier.projection_calls,
        "forward_stats": forward_stats,
        "inverse_stats": carrier.inverse_stats.descriptor(),
    }


def execute_case(interface: int, depth: int, family: str) -> dict[str, Any]:
    program = compile_program(interface, depth, family)
    carrier = Carrier.seal(interface)
    result = transaction(carrier, program)
    hybrid, hybrid_stats = hybrid_forward(program)
    checks = streamed_parity(interface, result["final_charts"], hybrid)
    hybrid_boundary = boundary_from_hybrid(hybrid, program)
    if tuple(result["boundary"]) != hybrid_boundary:
        fail("final boundary differs from matched recurrence")
    ordinary_ranks = [
        streamed_rank(
            interface,
            lambda x, y, chart=chart: chart_entry(interface, chart, x, y),
        )
        for chart in result["final_charts"]
    ]
    program_bytes = len(canonical_json(program.descriptor()))
    return {
        "interface": interface,
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "public_program_json_bytes": program_bytes,
        "boundary_commitment": digest_json(result["boundary"]),
        "final_relation_commitment": result["final_commitment"],
        "final_displacement_ranks": result["final_displacement_ranks"],
        "maximum_final_displacement_rank": max(result["final_displacement_ranks"]),
        "minimum_final_displacement_rank": min(result["final_displacement_ranks"]),
        "final_ordinary_ranks": ordinary_ranks,
        "all_final_relations_full_ordinary_rank": all(rank == interface for rank in ordinary_ranks),
        "streamed_relation_entry_parity_checks": checks,
        "boundary_identical_to_hybrid_classical_recurrence": True,
        "phase_stats": result["forward_stats"],
        "inverse_stats": result["inverse_stats"],
        "hybrid_classical_stats": hybrid_stats.descriptor(),
        "hybrid_final_modes": [node.mode for node in hybrid],
        "exact_restoration": result["exact_restoration"],
        "same_backing": result["same_backing"],
        "restoration_generation_before": result["generation_before"],
        "restoration_generation_after": result["generation_after"],
        "projection_calls": result["projection_calls"],
        "snapshot_reload_used": carrier.snapshot_reload_used,
        "inverse_history_cells": 0,
        "retained_restoration_baseline_cells": 0,
    }


def controls() -> dict[str, bool]:
    program = compile_program(7, 2, "PRIMARY")
    seed = Carrier.seal(7)
    initial = carrier_commitment(seed)
    missing = Carrier.seal(7)
    raw_forward(missing.controls, missing.charts, program)
    wrong = Carrier.seal(7)
    raw_forward(wrong.controls, wrong.charts, program)
    raw_inverse(wrong.controls, wrong.charts, program, offset_mutation=1)
    reordered = Carrier.seal(7)
    raw_forward(reordered.controls, reordered.charts, program)
    raw_inverse(
        reordered.controls,
        reordered.charts,
        program,
        assumed_action_order="INTERSECT_COMPOSE",
    )
    normal = Carrier.seal(7)
    raw_forward(normal.controls, normal.charts, program)
    disabled = Carrier.seal(7)
    raw_forward(disabled.controls, disabled.charts, program, port_enabled=False)
    swapped = Carrier.seal(7)
    raw_forward(
        swapped.controls, swapped.charts, program, action_order="INTERSECT_COMPOSE"
    )
    mutated = Carrier.seal(7)
    raw_forward(mutated.controls, mutated.charts, program, hub_mutation=1)
    null_rejected = False
    try:
        begin_forward(None, program, program.owner)
    except RuntimeError:
        null_rejected = True
    wrong_type = Carrier.seal(7)
    wrong_type.port_type = "F103_DENSE_RELATION"
    wrong_type_rejected = False
    try:
        begin_forward(wrong_type, program, program.owner)
    except RuntimeError:
        wrong_type_rejected = True
    wrong_owner_rejected = False
    try:
        begin_forward(Carrier.seal(7), program, program.owner ^ 1)
    except RuntimeError:
        wrong_owner_rejected = True
    leased = Carrier.seal(7)
    begin_forward(leased, program, program.owner)
    premature_rejected = False
    try:
        project(leased, program, program.owner)
    except RuntimeError:
        premature_rejected = True
    resident_rejected = False
    try:
        project_resident_port(leased, 0)
    except RuntimeError:
        resident_rejected = True
    reciprocal_exact = True
    for interface in INTERFACES:
        for node in range(NODE_COUNT):
            control = control_factors(interface, node)
            reciprocal = reciprocal_control(control)
            for x in range(interface):
                for y in range(interface):
                    reciprocal_exact &= (
                        factor_entry(control[0], control[1], x, y)
                        * factor_entry(reciprocal[0], reciprocal[1], x, y)
                    ) % MODULUS == 1
    return {
        "missing_inverse_changes_state": charts_commitment(missing.charts) != charts_commitment(seed.charts),
        "wrong_inverse_changes_state": charts_commitment(wrong.charts) != charts_commitment(seed.charts),
        "reordered_inverse_changes_state": charts_commitment(reordered.charts) != charts_commitment(seed.charts),
        "null_carrier_rejected": null_rejected,
        "wrong_relation_type_rejected": wrong_type_rejected,
        "wrong_owner_rejected": wrong_owner_rejected,
        "premature_projection_rejected": premature_rejected,
        "resident_port_projection_rejected": resident_rejected,
        "null_port_changes_boundary": boundary_from_charts(normal.charts, program) != boundary_from_charts(disabled.charts, program),
        "composition_intersection_order_changes_boundary": boundary_from_charts(normal.charts, program) != boundary_from_charts(swapped.charts, program),
        "topology_mutation_changes_boundary": boundary_from_charts(normal.charts, program) != boundary_from_charts(mutated.charts, program),
        "resident_controls_remain_unmodified": np.array_equal(normal.controls, seed.controls),
        "reciprocal_rank2_controls_exact": reciprocal_exact,
        "seed_commitment_is_canonical": carrier_commitment(seed) == initial,
    }


def reuse_controls() -> tuple[dict[str, Any], dict[str, Any]]:
    carrier = Carrier.seal(11)
    first = transaction(carrier, compile_program(11, 3, "PRIMARY"))
    backing = carrier.backing_identity
    second_program = compile_program(11, 13, "ALTERNATE")
    second = transaction(carrier, second_program)
    fresh = transaction(Carrier.seal(11), second_program)
    unrelated = {
        "first_exact_restoration": first["exact_restoration"],
        "second_exact_restoration": second["exact_restoration"],
        "same_backing_across_programs": carrier.backing_identity == backing,
        "second_boundary_matches_fresh": second["boundary"] == fresh["boundary"],
        "second_final_commitment_matches_fresh": second["final_commitment"] == fresh["final_commitment"],
        "resource_signature_matches_fresh": (
            second["forward_stats"] == fresh["forward_stats"]
            and second["inverse_stats"] == fresh["inverse_stats"]
        ),
        "restoration_generation": carrier.restoration_generation,
        "snapshot_reload_used": carrier.snapshot_reload_used,
    }
    repeated_carrier = Carrier.seal(5)
    initial = carrier_commitment(repeated_carrier)
    backing = repeated_carrier.backing_identity
    boundaries: set[tuple[int, ...]] = set()
    for _ in range(16):
        result = transaction(
            repeated_carrier, compile_program(5, 4, "PRIMARY")
        )
        boundaries.add(tuple(result["boundary"]))
    repeated = {
        "cycles": 16,
        "exact_restoration": carrier_commitment(repeated_carrier) == initial,
        "same_backing": repeated_carrier.backing_identity == backing,
        "restoration_generation": repeated_carrier.restoration_generation,
        "stable_boundary_count": len(boundaries),
        "snapshot_reload_used": repeated_carrier.snapshot_reload_used,
    }
    return unrelated, repeated


def run() -> dict[str, Any]:
    cases = [
        execute_case(interface, depth, family)
        for interface in INTERFACES
        for family in FAMILIES
        for depth in DEPTHS
    ]
    if not all(
        case["boundary_identical_to_hybrid_classical_recurrence"]
        and case["streamed_relation_entry_parity_checks"]
        == NODE_COUNT * case["interface"] * case["interface"]
        and case["exact_restoration"]
        and case["same_backing"]
        and case["restoration_generation_after"]
        == case["restoration_generation_before"] + 1
        and case["projection_calls"] == 1
        and not case["snapshot_reload_used"]
        for case in cases
    ):
        fail("one or more displacement-rank cases failed")
    control_results = controls()
    if not all(control_results.values()):
        fail(
            "displacement controls failed: "
            + repr([key for key, value in control_results.items() if not value])
        )
    unrelated, repeated = reuse_controls()
    if not all(
        (
            unrelated["first_exact_restoration"],
            unrelated["second_exact_restoration"],
            unrelated["same_backing_across_programs"],
            unrelated["second_boundary_matches_fresh"],
            unrelated["second_final_commitment_matches_fresh"],
            unrelated["resource_signature_matches_fresh"],
            not unrelated["snapshot_reload_used"],
            repeated["exact_restoration"],
            repeated["same_backing"],
            repeated["restoration_generation"] == repeated["cycles"],
            repeated["stable_boundary_count"] == 1,
            not repeated["snapshot_reload_used"],
        )
    ):
        fail("displacement-rank reuse failed")
    rank_law = {
        str(interface): {
            family: {
                str(depth): next(
                    case["maximum_final_displacement_rank"]
                    for case in cases
                    if case["interface"] == interface
                    and case["family"] == family
                    and case["depth"] == depth
                )
                for depth in DEPTHS
            }
            for family in FAMILIES
        }
        for interface in INTERFACES
    }
    maximum_rank_by_interface = {
        str(interface): max(
            case["maximum_final_displacement_rank"]
            for case in cases
            if case["interface"] == interface
        )
        for interface in INTERFACES
    }
    seed_signatures = {
        str(interface): {
            "displacement_ranks": [chart.rank for chart in seed_charts(interface)],
            "ordinary_ranks": [
                streamed_rank(
                    interface,
                    lambda x, y, chart=chart: chart_entry(
                        interface, chart, x, y
                    ),
                )
                for chart in seed_charts(interface)
            ],
        }
        for interface in INTERFACES
    }
    uniform_bound_observed = all(
        maximum_rank_by_interface[str(interface)]
        == maximum_rank_by_interface[str(INTERFACES[0])]
        for interface in INTERFACES
    )
    phase_resident_by_interface = {
        str(interface): (
            NODE_COUNT * 4 * interface
            + NODE_COUNT * (payload_capacity(interface) + interface + 1)
        )
        for interface in INTERFACES
    }
    classical_resident_by_interface = {
        str(interface): max(
            case["hybrid_classical_stats"]["maximum_resident_value_coordinates"]
            + case["hybrid_classical_stats"]["maximum_resident_metadata_bytes"]
            for case in cases
            if case["interface"] == interface
        )
        for interface in INTERFACES
    }
    maximum_phase_scratch_by_interface = {
        str(interface): max(
            max(
                case["phase_stats"]["maximum_declared_scratch_field_coordinates"],
                case["inverse_stats"]["maximum_declared_scratch_field_coordinates"],
            )
            for case in cases
            if case["interface"] == interface
        )
        for interface in INTERFACES
    }
    maximum_classical_scratch_by_interface = {
        str(interface): max(
            case["hybrid_classical_stats"]["chart_stats"]["maximum_declared_scratch_field_coordinates"]
            + case["hybrid_classical_stats"]["maximum_dense_scratch_coordinates"]
            + case["hybrid_classical_stats"]["maximum_control_rematerialization_coordinates"]
            for case in cases
            if case["interface"] == interface
        )
        for interface in INTERFACES
    }
    maximum_program_bytes = max(case["public_program_json_bytes"] for case in cases)
    depth16_work = {
        str(interface): {
            family: {
                "phase_forward_counted_field_multiplications": (
                    case["phase_stats"]["field_multiplications"]
                    + case["phase_stats"]["elimination_field_multiplications"]
                ),
                "phase_inverse_counted_field_multiplications": (
                    case["inverse_stats"]["field_multiplications"]
                    + case["inverse_stats"]["elimination_field_multiplications"]
                ),
                "classical_forward_counted_field_multiplications": (
                    case["hybrid_classical_stats"]["chart_stats"]["field_multiplications"]
                    + case["hybrid_classical_stats"]["chart_stats"]["elimination_field_multiplications"]
                    + case["hybrid_classical_stats"]["dense_composition_multiplications"]
                    + case["hybrid_classical_stats"]["dense_intersection_multiplications"]
                ),
            }
            for family in FAMILIES
            for case in cases
            if case["interface"] == interface
            and case["family"] == family
            and case["depth"] == 16
        }
        for interface in INTERFACES
    }
    for interface in INTERFACES:
        for family in FAMILIES:
            work = depth16_work[str(interface)][family]
            work["phase_forward_to_classical_forward_ratio"] = (
                work["phase_forward_counted_field_multiplications"]
                / work["classical_forward_counted_field_multiplications"]
            )
    phase_to_classical_resident_ratio = {
        str(interface): (
            phase_resident_by_interface[str(interface)]
            / classical_resident_by_interface[str(interface)]
        )
        for interface in INTERFACES
    }
    return {
        "schema": "CAT_CAS_F103_GROWING_DISPLACEMENT_RELATION_QUOTIENT_RESULT_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_scope": "LINUX_DIRECT_PROCESS_EXACT_FINITE_FIELD_STREAMED_DISPLACEMENT_CHART_SOFTWARE",
        "execution_scope": {
            "interfaces": list(INTERFACES),
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "public_topology_compilation_reads_final_answers": False,
            "catvm_machine_boundary_used": False,
        },
        "relation_law": {
            "field_modulus": MODULUS,
            "cyclic_displacement": "A_XY_MINUS_A_X_MINUS1_Y_MINUS1",
            "chart_value_coordinates": "N_PLUS_R_TIMES_2N_MINUS_R",
            "ordinary_relation_tables_materialized_on_phase_path": 0,
            "streamed_entry_composition": True,
            "streamed_entry_intersection": True,
            "exact_rank_canonicalization": True,
            "seed_signatures": seed_signatures,
            "rank_growth_by_interface_family_depth": rank_law,
            "maximum_displacement_rank_by_interface": maximum_rank_by_interface,
            "uniform_interface_independent_rank_bound_observed": uniform_bound_observed,
            "full_displacement_rank_reached_after_one_layer_at_every_interface_and_family": all(
                next(
                    case["maximum_final_displacement_rank"]
                    for case in cases
                    if case["interface"] == interface
                    and case["family"] == family
                    and case["depth"] == 1
                )
                == interface
                for interface in INTERFACES
                for family in FAMILIES
            ),
            "full_rank_chart_is_dense_equivalent_displacement_transpose_plus_boundary": True,
            "shared_unresolved_port_consumers_per_layer": 8,
            "resident_port_projection_before_final_boundary": False,
        },
        "carrier_law": {
            "resident_bytes_by_interface": phase_resident_by_interface,
            "fixed_backing_capacity_across_depth_at_each_interface": True,
            "backing_growth_law": "QUADRATIC_IN_INTERFACE_PLUS_LINEAR_CONTROL_FACTORS",
            "machine_enforced_generation_or_lease_custody": False,
            "retained_public_plan_cells": 0,
        },
        "matched_classical_recurrence": {
            "implementation": "EXECUTED_REMATERIALIZED_CONTROL_DISPLACEMENT_CHART_TO_DENSE_HYBRID",
            "switch_law": "USE_CHART_ONLY_WHILE_N_PLUS_R_TIMES_2N_MINUS_R_IS_LESS_THAN_N_SQUARED",
            "maximum_resident_bytes_by_interface": classical_resident_by_interface,
            "phase_to_classical_resident_byte_ratio_by_interface": phase_to_classical_resident_ratio,
            "depth16_counted_work_by_interface_and_family": depth16_work,
            "full_relation_and_boundary_match_every_case": True,
            "optimal_compact_classical_recurrence_claimed": False,
        },
        "restoration": {
            "carrier_classification": "EXACT_ALGEBRAIC_RESTORATION",
            "transient_buffers_classification": "NO_RESTORATION_CLAIM",
            "same_backing": True,
            "inverse_history_cells": 0,
            "retained_restoration_baseline_cells": 0,
            "snapshot_reload_used": False,
            "unrelated_program_reuse": unrelated,
            "repeated_reuse": repeated,
        },
        "controls": control_results,
        "resource_accounting": {
            "phase_resident_bytes_by_interface": phase_resident_by_interface,
            "classical_resident_bytes_by_interface": classical_resident_by_interface,
            "maximum_phase_declared_scratch_field_coordinates_by_interface": maximum_phase_scratch_by_interface,
            "maximum_classical_declared_scratch_field_coordinates_by_interface": maximum_classical_scratch_by_interface,
            "maximum_public_program_bytes": maximum_program_bytes,
            "phase_relation_table_scratch_coordinates": 0,
            "classical_dense_fallback_counted": True,
            "exact_rank_reduction_counted": True,
            "controller_backend_traffic_bytes": 0,
            "excluded": [
                "PYTHON_CONTAINER_OVERHEAD",
                "PYTHON_OBJECT_ALLOCATOR",
                "NUMPY_AND_NATIVE_LIBRARY_INTERNAL_STORAGE",
                "WHOLE_PROCESS_PEAK",
            ],
        },
        "cases": cases,
        "claim_ceiling": "F103_CYCLIC_DISPLACEMENT_CHARTS_ON_THE_DECLARED_C5_C7_C11_C17_NINE_NODE_ROTATING_HUB_FAMILIES_THROUGH_DEPTH16_IN_LINUX_DIRECT_PROCESS_SOFTWARE",
        "preserved_subclaims": [
            "FULL_ORDINARY_RANK_RELATIONS_HAVE_EXACT_RANK2_CYCLIC_DISPLACEMENT_SEED_CHARTS",
            "STREAMED_CHART_OPERATIONS_AVOID_PHASE_PATH_DENSE_RELATION_TABLE_MATERIALIZATION",
            "FINAL_ONLY_BOUNDARY_EXACT_ALGEBRAIC_RESTORATION_AND_SAME_BACKING_REUSE",
        ],
        "rejected_interpretations": [
            "UNIFORM_SUB_DENSE_DISPLACEMENT_RANK_CLOSURE_ACROSS_GROWING_INTERFACES",
            "DISPLACEMENT_CHART_RESOURCE_ADVANTAGE_OVER_MATCHED_COMPACT_CLASSICAL_SOFTWARE",
        ],
        "not_established": [
            "UNIFORM_SUB_DENSE_DISPLACEMENT_RANK_CLOSURE_ACROSS_GROWING_INTERFACES",
            "ARBITRARY_INTERFACE_CARDINALITY",
            "ARBITRARY_PORT_ARITY_OR_GRAPH_TOPOLOGY",
            "MACHINE_ENFORCED_GENERATION_OR_LEASE_CUSTODY",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "PHYSICAL_BIT_REPLACEMENT",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
        "next_obstruction": "CYCLIC_DISPLACEMENT_RANK2_SEEDS_BECOME_FULL_DISPLACEMENT_RANK_AFTER_ONE_NONCOMMUTING_COMPOSITION_INTERSECTION_LAYER_AT_EVERY_GROWING_INTERFACE_SO_THE_N2_PLUS_N_PHASE_CHART_IS_DENSE_EQUIVALENT_AND_THE_EXECUTED_DENSE_FALLBACK_CLASSICAL_RECURRENCE_IS_SMALLER_AND_CHEAPER",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run()
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
