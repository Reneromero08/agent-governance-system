#!/usr/bin/env python3
"""Exact rank-adaptive non-translation-invariant F103 open relations.

The accepted path stores each C17-to-C17 relation in a deterministic
rank-factor chart with r(34-r) field coordinates and pivot metadata.  Native
left composition by I plus a rank-two relation preserves rank.  Parallel
Hadamard intersection with a structured reciprocal-rank-two relation is
reversible but can double rank; exact factor-space canonicalization closes it
at the C17 rank ceiling without constructing a 17 by 17 entry table.

The comparison path is an executed adaptive classical recurrence: it uses the
same compact chart through rank eight, then switches to a 289-coordinate dense
relation because that is smaller than a rank-r factor chart.  This is bounded
direct-process software, not CATVM custody or a distinct phase resource.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


MODULUS = 103
CARDINALITY = 17
NODE_COUNT = 9
CONTROL_RANK = 2
MAX_RANK = 17
TARGET_PAYLOAD_CELLS = CARDINALITY * CARDINALITY
CONTROL_FACTOR_CELLS = NODE_COUNT * 2 * CARDINALITY * CONTROL_RANK
DEPTHS = (1, 2, 4, 8, 32, 128)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
PORT_TYPE = "F103_RANK_ADAPTIVE_NONTRANSLATION_C17_TO_C17"
CLAIM = (
    "BOUNDED_EXACT_RANK_ADAPTIVE_NON_TRANSLATION_INVARIANT_F103_C17_"
    "OPEN_RELATION_CHART_CLOSES_IDENTITY_PLUS_RANK2_COMPOSITION_AND_"
    "RECIPROCAL_RANK2_INTERSECTION_ON_ONE_SHARED_UNRESOLVED_PORT_"
    "ACROSS8_NONCOMMUTING_CONSUMERS_WITH_EXACT_RANK_ADAPTIVE_"
    "CANONICALIZATION_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE_"
    "THROUGH_DEPTH128_BUT_RANK_SATURATES17_THE_CHART_BECOMES_DENSE_"
    "EQUIVALENT_AND_AN_EXECUTED_REMATERIALIZED_CONTROL_HYBRID_CLASSICAL_"
    "RECURRENCE_RETAINS_FEWER_RESIDENT_COORDINATES"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def mod_array(value: np.ndarray) -> np.ndarray:
    return np.mod(value, MODULUS).astype(np.int64, copy=False)


def family_code(family: str) -> int:
    return {"PRIMARY": 5, "REUSE": 11, "ALTERNATE": 17}[family]


@dataclass(frozen=True)
class Program:
    depth: int
    family: str
    owner: int
    observation_left: int
    observation_right: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "CAT_CAS_F103_C17_RANK_ADAPTIVE_RELATION_PROGRAM_V1",
            "depth": self.depth,
            "family": self.family,
            "owner": self.owner,
            "node_count": NODE_COUNT,
            "port_type": PORT_TYPE,
            "topology": "PUBLIC_ROTATING_RANK2_CONTROL_HUB8",
            "composition": "LEFT_ACTION_BY_IDENTITY_PLUS_RANK2",
            "intersection": "HADAMARD_WITH_RECIPROCAL_RANK2_CONTROL",
            "observation": [self.observation_left, self.observation_right],
        }

    def fingerprint(self) -> str:
        return digest_json(self.descriptor())


def compile_program(depth: int, family: str) -> Program:
    if not isinstance(depth, int) or not 1 <= depth <= max(DEPTHS):
        fail("rank-adaptive relation depth outside declared ceiling")
    if family not in FAMILIES:
        fail("rank-adaptive relation family outside declared set")
    code = family_code(family)
    return Program(
        depth=depth,
        family=family,
        owner=(0xC1720000 + 131 * depth + code) & 0xFFFFFFFF,
        observation_left=(7 * depth + 3 * code + 2) % MODULUS,
        observation_right=(11 * depth + 5 * code + 1) % MODULUS,
    )


@dataclass
class WorkStats:
    composition_actions: int = 0
    intersection_actions: int = 0
    canonicalizations: int = 0
    matrix_field_multiplications: int = 0
    elimination_field_multiplications: int = 0
    field_inversions: int = 0
    factor_rotations: int = 0
    consumers: int = 0
    maximum_raw_factor_components: int = 0
    maximum_rank: int = 0
    maximum_declared_scratch_field_coordinates: int = 0
    rank_histogram: dict[int, int] = field(default_factory=dict)

    def observe_rank(self, rank: int) -> None:
        self.maximum_rank = max(self.maximum_rank, rank)
        self.rank_histogram[rank] = self.rank_histogram.get(rank, 0) + 1

    def descriptor(self) -> dict[str, Any]:
        return {
            "composition_actions": self.composition_actions,
            "intersection_actions": self.intersection_actions,
            "canonicalizations": self.canonicalizations,
            "matrix_field_multiplications": self.matrix_field_multiplications,
            "elimination_field_multiplications": self.elimination_field_multiplications,
            "field_inversions": self.field_inversions,
            "factor_rotations": self.factor_rotations,
            "consumers": self.consumers,
            "maximum_raw_factor_components": self.maximum_raw_factor_components,
            "maximum_rank": self.maximum_rank,
            "maximum_declared_scratch_field_coordinates": self.maximum_declared_scratch_field_coordinates,
            "rank_histogram": {
                str(key): self.rank_histogram[key]
                for key in sorted(self.rank_histogram)
            },
        }


def mmul(
    left: np.ndarray, right: np.ndarray, stats: WorkStats | None
) -> np.ndarray:
    if left.shape[1] != right.shape[0]:
        fail("matrix product shape mismatch")
    if stats is not None:
        stats.matrix_field_multiplications += (
            left.shape[0] * left.shape[1] * right.shape[1]
        )
    return mod_array(left @ right)


def rank_mod(matrix: np.ndarray, stats: WorkStats | None = None) -> int:
    work = mod_array(matrix.copy())
    rows, columns = work.shape
    pivot_row = 0
    for column in range(columns):
        candidate = next(
            (row for row in range(pivot_row, rows) if work[row, column]),
            None,
        )
        if candidate is None:
            continue
        if candidate != pivot_row:
            work[[pivot_row, candidate]] = work[[candidate, pivot_row]]
        inverse = pow(int(work[pivot_row, column]), -1, MODULUS)
        work[pivot_row] = mod_array(work[pivot_row] * inverse)
        if stats is not None:
            stats.field_inversions += 1
            stats.elimination_field_multiplications += columns
        for row in range(rows):
            if row == pivot_row or not work[row, column]:
                continue
            factor = int(work[row, column])
            work[row] = mod_array(work[row] - factor * work[pivot_row])
            if stats is not None:
                stats.elimination_field_multiplications += columns
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


def inverse_mod(matrix: np.ndarray, stats: WorkStats | None = None) -> np.ndarray:
    size = matrix.shape[0]
    if matrix.shape != (size, size):
        fail("inverse requires square matrix")
    work = np.concatenate(
        (mod_array(matrix.copy()), np.eye(size, dtype=np.int64)), axis=1
    )
    width = 2 * size
    for column in range(size):
        candidate = next(
            (row for row in range(column, size) if work[row, column]),
            None,
        )
        if candidate is None:
            fail("singular modular matrix")
        if candidate != column:
            work[[column, candidate]] = work[[candidate, column]]
        inverse = pow(int(work[column, column]), -1, MODULUS)
        work[column] = mod_array(work[column] * inverse)
        if stats is not None:
            stats.field_inversions += 1
            stats.elimination_field_multiplications += width
        for row in range(size):
            if row == column or not work[row, column]:
                continue
            factor = int(work[row, column])
            work[row] = mod_array(work[row] - factor * work[column])
            if stats is not None:
                stats.elimination_field_multiplications += width
    return work[:, size:]


def independent_rows(matrix: np.ndarray, stats: WorkStats | None) -> list[int]:
    selected: list[int] = []
    current = np.empty((0, matrix.shape[1]), dtype=np.int64)
    for row in range(matrix.shape[0]):
        candidate = np.concatenate((current, matrix[row : row + 1]), axis=0)
        if rank_mod(candidate, stats) > len(selected):
            selected.append(row)
            current = candidate
        if len(selected) == matrix.shape[1]:
            break
    if len(selected) != matrix.shape[1]:
        fail("full-column-rank factor lacks pivot rows")
    return selected


def column_basis_with_coefficients(
    matrix: np.ndarray, stats: WorkStats | None
) -> tuple[np.ndarray, np.ndarray]:
    columns = matrix.shape[1]
    if columns == 0:
        return (
            np.empty((CARDINALITY, 0), dtype=np.int64),
            np.empty((0, 0), dtype=np.int64),
        )
    selected: list[int] = []
    basis = np.empty((CARDINALITY, 0), dtype=np.int64)
    for column in range(columns):
        candidate = np.concatenate((basis, matrix[:, column : column + 1]), axis=1)
        if rank_mod(candidate, stats) > len(selected):
            selected.append(column)
            basis = candidate
        if len(selected) == CARDINALITY:
            break
    rank = len(selected)
    if rank == 0:
        return (
            np.empty((CARDINALITY, 0), dtype=np.int64),
            np.empty((0, columns), dtype=np.int64),
        )
    pivots = independent_rows(basis, stats)
    inverse = inverse_mod(basis[pivots, :], stats)
    coefficients = mmul(inverse, matrix[pivots, :], stats)
    if not np.array_equal(mmul(basis, coefficients, None), mod_array(matrix)):
        fail("column basis reconstruction failed")
    return basis, coefficients


def canonicalize_factors(
    left: np.ndarray,
    right: np.ndarray,
    stats: WorkStats | None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if left.shape != right.shape or left.shape[0] != CARDINALITY:
        fail("factor shape outside C17 relation law")
    components = left.shape[1]
    if stats is not None:
        stats.canonicalizations += 1
        stats.maximum_raw_factor_components = max(
            stats.maximum_raw_factor_components, components
        )
        # Conservative simultaneous named-array bound for the implemented
        # two-stage basis reduction, pivot inversion, and product buffers.
        stats.maximum_declared_scratch_field_coordinates = max(
            stats.maximum_declared_scratch_field_coordinates,
            2 * CARDINALITY * components
            + 10 * CARDINALITY * CARDINALITY
            + 2 * components * MAX_RANK,
        )
    left_basis, left_coefficients = column_basis_with_coefficients(left, stats)
    right_reduced = mmul(right, left_coefficients.T, stats)
    right_basis, right_coefficients = column_basis_with_coefficients(
        right_reduced, stats
    )
    rank = right_basis.shape[1]
    if rank == 0:
        if stats is not None:
            stats.observe_rank(0)
        return (
            np.empty((CARDINALITY, 0), dtype=np.int64),
            np.empty((CARDINALITY, 0), dtype=np.int64),
            [],
        )
    left_reduced = mmul(left_basis, right_coefficients.T, stats)
    pivots = independent_rows(left_reduced, stats)
    pivot_matrix = left_reduced[pivots, :]
    pivot_inverse = inverse_mod(pivot_matrix, stats)
    canonical_left = mmul(left_reduced, pivot_inverse, stats)
    canonical_right = mmul(right_basis, pivot_matrix.T, stats)
    if not np.array_equal(
        canonical_left[pivots, :], np.eye(rank, dtype=np.int64)
    ):
        fail("canonical pivot gauge failed")
    if stats is not None:
        stats.observe_rank(rank)
    return canonical_left, canonical_right, pivots


def chart_coordinate_count(rank: int) -> int:
    return rank * (2 * CARDINALITY - rank)


def pack_chart(
    left: np.ndarray,
    right: np.ndarray,
    pivots: list[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    rank = left.shape[1]
    if right.shape != left.shape or len(pivots) != rank:
        fail("invalid canonical factor chart")
    payload = np.zeros(TARGET_PAYLOAD_CELLS, dtype=np.uint8)
    pivot_payload = np.full(MAX_RANK, 255, dtype=np.uint8)
    if rank == 0:
        return payload, pivot_payload, 0
    pivot_payload[:rank] = np.array(pivots, dtype=np.uint8)
    nonpivots = [row for row in range(CARDINALITY) if row not in pivots]
    values = np.concatenate(
        (left[nonpivots, :].reshape(-1), right.reshape(-1))
    )
    expected = chart_coordinate_count(rank)
    if values.size != expected or expected > TARGET_PAYLOAD_CELLS:
        fail("rank chart payload size mismatch")
    payload[:expected] = mod_array(values).astype(np.uint8)
    return payload, pivot_payload, rank


def unpack_chart(
    payload: np.ndarray, pivot_payload: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= rank <= MAX_RANK:
        fail("rank metadata outside C17 ceiling")
    if rank == 0:
        return (
            np.empty((CARDINALITY, 0), dtype=np.int64),
            np.empty((CARDINALITY, 0), dtype=np.int64),
        )
    pivots = [int(value) for value in pivot_payload[:rank]]
    if len(set(pivots)) != rank or any(not 0 <= value < CARDINALITY for value in pivots):
        fail("invalid chart pivot metadata")
    nonpivots = [row for row in range(CARDINALITY) if row not in pivots]
    left_count = (CARDINALITY - rank) * rank
    total = chart_coordinate_count(rank)
    values = payload[:total].astype(np.int64)
    left = np.zeros((CARDINALITY, rank), dtype=np.int64)
    left[pivots, :] = np.eye(rank, dtype=np.int64)
    left[nonpivots, :] = values[:left_count].reshape(
        CARDINALITY - rank, rank
    )
    right = values[left_count:].reshape(CARDINALITY, rank)
    return left, right


def factor_entry(left: np.ndarray, right: np.ndarray, x: int, y: int) -> int:
    return int(np.dot(left[x], right[y]) % MODULUS)


def factors_to_dense(
    left: np.ndarray, right: np.ndarray, stats: WorkStats | None
) -> np.ndarray:
    return mmul(left, right.T, stats)


def control_relation(node: int) -> np.ndarray:
    q = (7 + 9 * node) % MODULUS
    if q in (0, MODULUS - 1):
        q = 3
    a = np.array(
        [1 + ((11 * node + 7 * x + 3 * node * x) % 101) for x in range(CARDINALITY)],
        dtype=np.int64,
    )
    b = np.array(
        [1 + ((13 * node + 5 * y + 4 * node * y) % 101) for y in range(CARDINALITY)],
        dtype=np.int64,
    )
    s = np.array([1 if ((x + 2 * node) % 5) < 2 else 0 for x in range(CARDINALITY)], dtype=np.int64)
    t = np.array([1 if ((3 * y + node) % 7) < 3 else 0 for y in range(CARDINALITY)], dtype=np.int64)
    left = np.stack((a, mod_array(a * s)), axis=1)
    right = np.stack((b, mod_array(b * q * t)), axis=1)
    if rank_mod(left) != 2 or rank_mod(right) != 2:
        fail("structured control failed exact rank-two requirement")
    for x in range(CARDINALITY):
        for y in range(CARDINALITY):
            if factor_entry(left, right, x, y) == 0:
                fail("structured rank-two control contains zero entry")
    return np.stack((left, right), axis=0)


def reciprocal_control(control: np.ndarray) -> np.ndarray:
    left = control[0].astype(np.int64)
    right = control[1].astype(np.int64)
    a = left[:, 0]
    b = right[:, 0]
    a_inverse = np.array([pow(int(value), -1, MODULUS) for value in a], dtype=np.int64)
    b_inverse = np.array([pow(int(value), -1, MODULUS) for value in b], dtype=np.int64)
    s = mod_array(left[:, 1] * a_inverse)
    z = mod_array(right[:, 1] * b_inverse)
    nonzero = sorted({int(value) for value in z if value})
    if len(nonzero) != 1 or any(int(value) not in (0, nonzero[0]) for value in z):
        fail("actual resident control lost reciprocal-rank-two structure")
    q = nonzero[0]
    coefficient = (-pow(1 + q, -1, MODULUS)) % MODULUS
    inverse_left = np.stack((a_inverse, mod_array(a_inverse * s)), axis=1)
    inverse_right = np.stack(
        (b_inverse, mod_array(b_inverse * coefficient * z)), axis=1
    )
    for x in range(CARDINALITY):
        for y in range(CARDINALITY):
            original = factor_entry(left, right, x, y)
            reciprocal = factor_entry(inverse_left, inverse_right, x, y)
            if original * reciprocal % MODULUS != 1:
                fail("rank-two reciprocal factorization failed")
    return np.stack((inverse_left, inverse_right), axis=0)


def target_relation(node: int) -> tuple[np.ndarray, np.ndarray]:
    left = np.empty((CARDINALITY, 2), dtype=np.int64)
    right = np.empty((CARDINALITY, 2), dtype=np.int64)
    for coordinate in range(CARDINALITY):
        left[coordinate, 0] = (2 + 7 * node + 5 * coordinate + node * coordinate) % MODULUS
        left[coordinate, 1] = (3 + 11 * node + 9 * coordinate * coordinate) % MODULUS
        right[coordinate, 0] = (5 + 13 * node + 7 * coordinate * coordinate) % MODULUS
        right[coordinate, 1] = (8 + 17 * node + 3 * coordinate + 2 * node * coordinate) % MODULUS
    return left, right


def seed_targets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payloads = np.zeros((NODE_COUNT, TARGET_PAYLOAD_CELLS), dtype=np.uint8)
    pivots = np.full((NODE_COUNT, MAX_RANK), 255, dtype=np.uint8)
    ranks = np.zeros(NODE_COUNT, dtype=np.uint8)
    for node in range(NODE_COUNT):
        left, right = target_relation(node)
        canonical_left, canonical_right, pivot_rows = canonicalize_factors(
            left, right, None
        )
        payloads[node], pivots[node], rank = pack_chart(
            canonical_left, canonical_right, pivot_rows
        )
        ranks[node] = rank
    return payloads, pivots, ranks


def seed_controls() -> np.ndarray:
    return np.stack([control_relation(node) for node in range(NODE_COUNT)], axis=0).astype(np.uint8)


def hub_index(index: int, family: str, mutation: int = 0) -> int:
    return (5 * index + family_code(family) + mutation) % NODE_COUNT


def peer_order(hub: int) -> list[int]:
    return [(hub + offset) % NODE_COUNT for offset in range(1, NODE_COUNT)]


def relation_offset(hub: int, peer: int, index: int, family: str, mutation: int = 0) -> int:
    return (7 * hub + 11 * peer + 3 * index + family_code(family) + mutation) % CARDINALITY


def rotation_shift(node: int, index: int, family: str) -> int:
    code = family_code(family)
    return (3 * node * node + 5 * index + code * (1 + index.bit_count())) % CARDINALITY


def control_view(controls: np.ndarray, hub: int, peer: int, index: int, family: str, mutation: int = 0) -> np.ndarray:
    offset = relation_offset(hub, peer, index, family, mutation)
    view = controls[hub].astype(np.int64).copy()
    view[0] = np.roll(view[0], offset, axis=0)
    view[1] = np.roll(view[1], -offset, axis=0)
    return view


def nonsingular_coupling(control: np.ndarray) -> int:
    left = control[0].astype(np.int64)
    right = control[1].astype(np.int64)
    pairing = mod_array(right.T @ left)
    for coupling in range(1, MODULUS):
        kernel = mod_array(
            np.eye(CONTROL_RANK, dtype=np.int64) + coupling * pairing
        )
        determinant = (
            int(kernel[0, 0]) * int(kernel[1, 1])
            - int(kernel[0, 1]) * int(kernel[1, 0])
        ) % MODULUS
        if determinant:
            return coupling
    fail("no invertible identity-plus-rank-two coupling")


def store_chart(
    payloads: np.ndarray,
    pivots: np.ndarray,
    ranks: np.ndarray,
    node: int,
    left: np.ndarray,
    right: np.ndarray,
    stats: WorkStats | None,
) -> None:
    canonical_left, canonical_right, pivot_rows = canonicalize_factors(
        left, right, stats
    )
    payload, pivot_payload, rank = pack_chart(
        canonical_left, canonical_right, pivot_rows
    )
    payloads[node] = payload
    pivots[node] = pivot_payload
    ranks[node] = rank


def rotate_chart(payloads: np.ndarray, pivots: np.ndarray, ranks: np.ndarray, node: int, shift: int, stats: WorkStats | None) -> None:
    if int(ranks[node]) == MAX_RANK:
        right = payloads[node, :TARGET_PAYLOAD_CELLS].astype(np.int64).reshape(
            CARDINALITY, CARDINALITY
        )
        right = np.roll(np.roll(right, -shift, axis=0), shift, axis=1)
        payloads[node, :TARGET_PAYLOAD_CELLS] = right.reshape(-1).astype(np.uint8)
        if stats is not None:
            stats.factor_rotations += 2
            stats.observe_rank(MAX_RANK)
        return
    left, right = unpack_chart(payloads[node], pivots[node], int(ranks[node]))
    store_chart(
        payloads,
        pivots,
        ranks,
        node,
        np.roll(left, shift, axis=0),
        np.roll(right, -shift, axis=0),
        stats,
    )
    if stats is not None:
        stats.factor_rotations += 2


def compose_chart(payloads: np.ndarray, pivots: np.ndarray, ranks: np.ndarray, node: int, control: np.ndarray, *, inverse: bool, stats: WorkStats | None) -> None:
    control_left = control[0].astype(np.int64)
    control_right = control[1].astype(np.int64)
    coupling = nonsingular_coupling(control)
    if int(ranks[node]) == MAX_RANK:
        right = payloads[node, :TARGET_PAYLOAD_CELLS].astype(np.int64).reshape(
            CARDINALITY, CARDINALITY
        )
        contraction = mmul(right, control_right, stats)
        if inverse:
            kernel = mod_array(
                np.eye(CONTROL_RANK, dtype=np.int64)
                + coupling * mmul(control_right.T, control_left, stats)
            )
            kernel_inverse_transpose = inverse_mod(kernel, stats).T
            correction = mmul(
                mmul(contraction, kernel_inverse_transpose, stats),
                control_left.T,
                stats,
            )
            updated_right = mod_array(right - coupling * correction)
        else:
            correction = mmul(contraction, control_left.T, stats)
            updated_right = mod_array(right + coupling * correction)
        payloads[node, :TARGET_PAYLOAD_CELLS] = updated_right.reshape(-1).astype(np.uint8)
        if stats is not None:
            stats.composition_actions += 1
            stats.observe_rank(MAX_RANK)
            stats.maximum_declared_scratch_field_coordinates = max(
                stats.maximum_declared_scratch_field_coordinates,
                2 * TARGET_PAYLOAD_CELLS + 4 * CARDINALITY,
            )
        return
    left, right = unpack_chart(payloads[node], pivots[node], int(ranks[node]))
    if inverse:
        kernel = mod_array(
            np.eye(CONTROL_RANK, dtype=np.int64)
            + coupling * mmul(control_right.T, control_left, stats)
        )
        kernel_inverse = inverse_mod(kernel, stats)
        contraction = mmul(control_right.T, left, stats)
        correction = mmul(
            control_left, mmul(kernel_inverse, contraction, stats), stats
        )
        updated_left = mod_array(left - coupling * correction)
    else:
        contraction = mmul(control_right.T, left, stats)
        updated_left = mod_array(
            left + coupling * mmul(control_left, contraction, stats)
        )
    store_chart(payloads, pivots, ranks, node, updated_left, right, stats)
    if stats is not None:
        stats.composition_actions += 1


def intersect_chart(payloads: np.ndarray, pivots: np.ndarray, ranks: np.ndarray, node: int, control: np.ndarray, *, inverse: bool, stats: WorkStats | None) -> None:
    factor = reciprocal_control(control) if inverse else control
    control_left = factor[0].astype(np.int64)
    control_right = factor[1].astype(np.int64)
    if int(ranks[node]) == MAX_RANK:
        right = payloads[node, :TARGET_PAYLOAD_CELLS].astype(np.int64).reshape(
            CARDINALITY, CARDINALITY
        )
        updated = np.empty_like(right)
        for y in range(CARDINALITY):
            for x in range(CARDINALITY):
                updated[y, x] = (
                    int(right[y, x])
                    * factor_entry(control_left, control_right, x, y)
                ) % MODULUS
        payloads[node, :TARGET_PAYLOAD_CELLS] = updated.reshape(-1).astype(np.uint8)
        if stats is not None:
            stats.intersection_actions += 1
            stats.matrix_field_multiplications += 3 * TARGET_PAYLOAD_CELLS
            stats.observe_rank(MAX_RANK)
            stats.maximum_declared_scratch_field_coordinates = max(
                stats.maximum_declared_scratch_field_coordinates,
                2 * TARGET_PAYLOAD_CELLS,
            )
        return
    left, right = unpack_chart(payloads[node], pivots[node], int(ranks[node]))
    rank = left.shape[1]
    product_left = mod_array(
        (left[:, :, None] * control_left[:, None, :]).reshape(
            CARDINALITY, rank * CONTROL_RANK
        )
    )
    product_right = mod_array(
        (right[:, :, None] * control_right[:, None, :]).reshape(
            CARDINALITY, rank * CONTROL_RANK
        )
    )
    if stats is not None:
        stats.matrix_field_multiplications += 2 * CARDINALITY * rank * CONTROL_RANK
    store_chart(payloads, pivots, ranks, node, product_left, product_right, stats)
    if stats is not None:
        stats.intersection_actions += 1


def raw_forward(controls: np.ndarray, payloads: np.ndarray, pivots: np.ndarray, ranks: np.ndarray, program: Program, *, action_order: str = "COMPOSE_INTERSECT", port_enabled: bool = True, hub_mutation: int = 0, offset_mutation: int = 0, stats: WorkStats | None = None) -> None:
    actions = (compose_chart, intersect_chart) if action_order == "COMPOSE_INTERSECT" else (intersect_chart, compose_chart)
    for index in range(program.depth):
        for node in range(NODE_COUNT):
            rotate_chart(payloads, pivots, ranks, node, rotation_shift(node, index, program.family), stats)
        hub = hub_index(index, program.family, hub_mutation)
        for peer in peer_order(hub):
            if not port_enabled:
                continue
            control = control_view(controls, hub, peer, index, program.family, offset_mutation)
            for action in actions:
                action(payloads, pivots, ranks, peer, control, inverse=False, stats=stats)
            if stats is not None:
                stats.consumers += 1


def canonicalize_target_bank(
    payloads: np.ndarray,
    pivots: np.ndarray,
    ranks: np.ndarray,
    stats: WorkStats | None,
) -> None:
    for node in range(NODE_COUNT):
        left, right = unpack_chart(
            payloads[node], pivots[node], int(ranks[node])
        )
        store_chart(payloads, pivots, ranks, node, left, right, stats)


def raw_inverse(controls: np.ndarray, payloads: np.ndarray, pivots: np.ndarray, ranks: np.ndarray, program: Program, *, assumed_action_order: str = "COMPOSE_INTERSECT", offset_mutation: int = 0, stats: WorkStats | None = None) -> None:
    actions = (intersect_chart, compose_chart) if assumed_action_order == "COMPOSE_INTERSECT" else (compose_chart, intersect_chart)
    for index in reversed(range(program.depth)):
        hub = hub_index(index, program.family)
        for peer in reversed(peer_order(hub)):
            control = control_view(controls, hub, peer, index, program.family, offset_mutation)
            for action in actions:
                action(payloads, pivots, ranks, peer, control, inverse=True, stats=stats)
        for node in reversed(range(NODE_COUNT)):
            rotate_chart(payloads, pivots, ranks, node, -rotation_shift(node, index, program.family), stats)
    # Full-rank execution uses the exact ambient chart whose payload is the
    # relation transpose.  One history-free exact factor reduction after the
    # complete inverse restores the unique canonical machine representation.
    canonicalize_target_bank(payloads, pivots, ranks, stats)


def chart_entry(payload: np.ndarray, pivots: np.ndarray, rank: int, x: int, y: int) -> int:
    left, right = unpack_chart(payload, pivots, rank)
    return factor_entry(left, right, x, y)


def chart_actual_rank(payload: np.ndarray, pivots: np.ndarray, rank: int) -> int:
    if rank < MAX_RANK:
        return rank
    _left, right = unpack_chart(payload, pivots, rank)
    return rank_mod(right)


def boundary_from_charts(payloads: np.ndarray, pivots: np.ndarray, ranks: np.ndarray, program: Program) -> tuple[int, ...]:
    boundary: list[int] = []
    for coordinate in range(CARDINALITY):
        value = 0
        for node in range(NODE_COUNT):
            x = (program.observation_left + coordinate + 3 * node) % CARDINALITY
            y = (program.observation_right + 2 * coordinate + 5 * node) % CARDINALITY
            weight = (1 + node + coordinate * coordinate) % MODULUS
            value += weight * chart_entry(
                payloads[node], pivots[node], int(ranks[node]), x, y
            )
        boundary.append(value % MODULUS)
    return tuple(boundary)


def charts_commitment(payloads: np.ndarray, pivots: np.ndarray, ranks: np.ndarray) -> str:
    return hashlib.sha256(payloads.tobytes() + pivots.tobytes() + ranks.tobytes()).hexdigest()


@dataclass
class HybridNode:
    mode: str
    payload: np.ndarray
    pivots: np.ndarray
    rank: int
    dense: np.ndarray


@dataclass
class ClassicalStats:
    chart_stats: WorkStats = field(default_factory=WorkStats)
    dense_composition_multiplications: int = 0
    dense_intersection_multiplications: int = 0
    chart_to_dense_switches: int = 0
    maximum_resident_value_coordinates: int = 0
    maximum_resident_metadata_bytes: int = 0
    maximum_dense_control_scratch_coordinates: int = 0
    maximum_control_rematerialization_scratch_coordinates: int = 0

    def descriptor(self) -> dict[str, Any]:
        return {
            "chart_stats": self.chart_stats.descriptor(),
            "dense_composition_multiplications": self.dense_composition_multiplications,
            "dense_intersection_multiplications": self.dense_intersection_multiplications,
            "chart_to_dense_switches": self.chart_to_dense_switches,
            "maximum_resident_value_coordinates": self.maximum_resident_value_coordinates,
            "maximum_resident_metadata_bytes": self.maximum_resident_metadata_bytes,
            "maximum_dense_control_scratch_coordinates": self.maximum_dense_control_scratch_coordinates,
            "maximum_control_rematerialization_scratch_coordinates": self.maximum_control_rematerialization_scratch_coordinates,
        }


def seed_hybrid() -> list[HybridNode]:
    payloads, pivots, ranks = seed_targets()
    return [
        HybridNode(
            mode="CHART",
            payload=payloads[node].copy(),
            pivots=pivots[node].copy(),
            rank=int(ranks[node]),
            dense=np.zeros((CARDINALITY, CARDINALITY), dtype=np.uint8),
        )
        for node in range(NODE_COUNT)
    ]


def hybrid_account(nodes: list[HybridNode], stats: ClassicalStats) -> None:
    value_coordinates = 0
    metadata = 0
    for node in nodes:
        if node.mode == "CHART":
            value_coordinates += chart_coordinate_count(node.rank)
            metadata += 1 + node.rank
        else:
            value_coordinates += TARGET_PAYLOAD_CELLS
            metadata += 2
    stats.maximum_resident_value_coordinates = max(
        stats.maximum_resident_value_coordinates, value_coordinates
    )
    stats.maximum_resident_metadata_bytes = max(
        stats.maximum_resident_metadata_bytes, metadata
    )


def hybrid_chart_arrays(node: HybridNode) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        node.payload.reshape(1, -1).copy(),
        node.pivots.reshape(1, -1).copy(),
        np.array([node.rank], dtype=np.uint8),
    )


def hybrid_store_chart(node: HybridNode, payloads: np.ndarray, pivots: np.ndarray, ranks: np.ndarray, stats: ClassicalStats) -> None:
    rank = int(ranks[0])
    if rank <= 8:
        node.mode = "CHART"
        node.payload = payloads[0].copy()
        node.pivots = pivots[0].copy()
        node.rank = rank
        node.dense.fill(0)
        return
    left, right = unpack_chart(payloads[0], pivots[0], rank)
    node.mode = "DENSE"
    node.dense = factors_to_dense(left, right, stats.chart_stats).astype(np.uint8)
    node.rank = rank
    node.payload.fill(0)
    node.pivots.fill(255)
    stats.chart_to_dense_switches += 1


def dense_control_entry(control: np.ndarray, x: int, y: int) -> int:
    return factor_entry(control[0], control[1], x, y)


def hybrid_apply(node: HybridNode, control: np.ndarray, action: str, stats: ClassicalStats) -> None:
    if node.mode == "CHART":
        payloads, pivots, ranks = hybrid_chart_arrays(node)
        if action == "COMPOSE":
            compose_chart(payloads, pivots, ranks, 0, control, inverse=False, stats=stats.chart_stats)
        else:
            intersect_chart(payloads, pivots, ranks, 0, control, inverse=False, stats=stats.chart_stats)
        hybrid_store_chart(node, payloads, pivots, ranks, stats)
        return
    dense = node.dense.astype(np.int64)
    if action == "COMPOSE":
        coupling = nonsingular_coupling(control)
        contraction = mod_array(control[1].T.astype(np.int64) @ dense)
        node.dense = mod_array(
            dense
            + coupling * control[0].astype(np.int64) @ contraction
        ).astype(np.uint8)
        stats.dense_composition_multiplications += 2 * CONTROL_RANK * CARDINALITY * CARDINALITY
    else:
        updated = np.empty_like(dense)
        for x in range(CARDINALITY):
            for y in range(CARDINALITY):
                updated[x, y] = dense[x, y] * dense_control_entry(control, x, y) % MODULUS
        node.dense = updated.astype(np.uint8)
        stats.dense_intersection_multiplications += 3 * TARGET_PAYLOAD_CELLS
        stats.maximum_dense_control_scratch_coordinates = max(
            stats.maximum_dense_control_scratch_coordinates, 1
        )


def hybrid_rotate(node: HybridNode, shift: int, stats: ClassicalStats) -> None:
    if node.mode == "DENSE":
        node.dense = np.roll(np.roll(node.dense, shift, axis=0), -shift, axis=1)
        return
    payloads, pivots, ranks = hybrid_chart_arrays(node)
    rotate_chart(payloads, pivots, ranks, 0, shift, stats.chart_stats)
    hybrid_store_chart(node, payloads, pivots, ranks, stats)


def hybrid_forward(program: Program) -> tuple[list[HybridNode], ClassicalStats]:
    nodes = seed_hybrid()
    stats = ClassicalStats()
    hybrid_account(nodes, stats)
    for index in range(program.depth):
        for node_index, node in enumerate(nodes):
            hybrid_rotate(node, rotation_shift(node_index, index, program.family), stats)
        hub = hub_index(index, program.family)
        for peer in peer_order(hub):
            control = control_relation(hub).astype(np.int64)
            offset = relation_offset(hub, peer, index, program.family)
            control[0] = np.roll(control[0], offset, axis=0)
            control[1] = np.roll(control[1], -offset, axis=0)
            stats.maximum_control_rematerialization_scratch_coordinates = max(
                stats.maximum_control_rematerialization_scratch_coordinates,
                8 * CARDINALITY,
            )
            hybrid_apply(nodes[peer], control, "COMPOSE", stats)
            hybrid_apply(nodes[peer], control, "INTERSECT", stats)
        hybrid_account(nodes, stats)
    return nodes, stats


def hybrid_entry(node: HybridNode, x: int, y: int) -> int:
    if node.mode == "DENSE":
        return int(node.dense[x, y])
    return chart_entry(node.payload, node.pivots, node.rank, x, y)


def boundary_from_hybrid(nodes: list[HybridNode], program: Program) -> tuple[int, ...]:
    values: list[int] = []
    for coordinate in range(CARDINALITY):
        value = 0
        for node_index, node in enumerate(nodes):
            x = (program.observation_left + coordinate + 3 * node_index) % CARDINALITY
            y = (program.observation_right + 2 * coordinate + 5 * node_index) % CARDINALITY
            value += (1 + node_index + coordinate * coordinate) * hybrid_entry(node, x, y)
        values.append(value % MODULUS)
    return tuple(values)


def streamed_relation_parity(payloads: np.ndarray, pivots: np.ndarray, ranks: np.ndarray, hybrid: list[HybridNode]) -> int:
    checks = 0
    for node in range(NODE_COUNT):
        for x in range(CARDINALITY):
            for y in range(CARDINALITY):
                phase_value = chart_entry(payloads[node], pivots[node], int(ranks[node]), x, y)
                if phase_value != hybrid_entry(hybrid[node], x, y):
                    fail("rank-adaptive relation differs from hybrid classical recurrence")
                checks += 1
    return checks


@dataclass
class Carrier:
    controls: np.ndarray
    payloads: np.ndarray
    pivots: np.ndarray
    ranks: np.ndarray
    port_type: str = PORT_TYPE
    leased: bool = False
    lease_owner: int | None = None
    lease_program: str | None = None
    forward_complete: bool = False
    restoration_generation: int = 0
    projection_calls: int = 0
    snapshot_reload_used: bool = False
    stats: WorkStats = field(default_factory=WorkStats)
    inverse_stats: WorkStats = field(default_factory=WorkStats)

    @classmethod
    def seal(cls) -> "Carrier":
        payloads, pivots, ranks = seed_targets()
        return cls(seed_controls(), payloads, pivots, ranks)

    @property
    def backing_identity(self) -> tuple[int, int, int, int]:
        return (
            int(self.controls.__array_interface__["data"][0]),
            int(self.payloads.__array_interface__["data"][0]),
            int(self.pivots.__array_interface__["data"][0]),
            int(self.ranks.__array_interface__["data"][0]),
        )


def state_commitment(carrier: Carrier) -> str:
    return hashlib.sha256(
        carrier.controls.tobytes()
        + carrier.payloads.tobytes()
        + carrier.pivots.tobytes()
        + carrier.ranks.tobytes()
    ).hexdigest()


def begin_forward(carrier: Carrier | None, program: Program, owner: int) -> None:
    if carrier is None:
        fail("null carrier")
    if carrier.port_type != PORT_TYPE:
        fail("wrong relation port type")
    if carrier.leased:
        fail("carrier already leased")
    if owner != program.owner:
        fail("wrong direct-process owner")
    carrier.leased = True
    carrier.lease_owner = owner
    carrier.lease_program = program.fingerprint()
    carrier.forward_complete = False
    carrier.projection_calls = 0
    carrier.stats = WorkStats()


def require_owned(carrier: Carrier, program: Program, owner: int) -> None:
    if not carrier.leased or carrier.lease_owner != owner or carrier.lease_program != program.fingerprint():
        fail("direct-process relation lease mismatch")


def forward(carrier: Carrier, program: Program, owner: int) -> None:
    require_owned(carrier, program, owner)
    raw_forward(carrier.controls, carrier.payloads, carrier.pivots, carrier.ranks, program, stats=carrier.stats)
    carrier.forward_complete = True


def project(carrier: Carrier, program: Program, owner: int) -> tuple[int, ...]:
    require_owned(carrier, program, owner)
    if not carrier.forward_complete:
        fail("projection before final boundary")
    carrier.projection_calls += 1
    return boundary_from_charts(carrier.payloads, carrier.pivots, carrier.ranks, program)


def project_resident_port(_carrier: Carrier, _node: int) -> None:
    fail("resident rank-adaptive relation projection is forbidden")


def inverse(carrier: Carrier, program: Program, owner: int) -> None:
    require_owned(carrier, program, owner)
    carrier.inverse_stats = WorkStats()
    raw_inverse(
        carrier.controls,
        carrier.payloads,
        carrier.pivots,
        carrier.ranks,
        program,
        stats=carrier.inverse_stats,
    )
    carrier.leased = False
    carrier.lease_owner = None
    carrier.lease_program = None
    carrier.forward_complete = False
    carrier.restoration_generation += 1


def execute_case(depth: int, family: str) -> dict[str, Any]:
    program = compile_program(depth, family)
    carrier = Carrier.seal()
    initial = state_commitment(carrier)
    backing = carrier.backing_identity
    initial_controls = carrier.controls.copy()
    generation = carrier.restoration_generation
    begin_forward(carrier, program, program.owner)
    forward(carrier, program, program.owner)
    final_commitment = state_commitment(carrier)
    boundary = project(carrier, program, program.owner)
    final_ranks = [
        chart_actual_rank(
            carrier.payloads[node],
            carrier.pivots[node],
            int(carrier.ranks[node]),
        )
        for node in range(NODE_COUNT)
    ]
    hybrid, classical_stats = hybrid_forward(program)
    relation_checks = streamed_relation_parity(carrier.payloads, carrier.pivots, carrier.ranks, hybrid)
    classical_boundary = boundary_from_hybrid(hybrid, program)
    stats = carrier.stats.descriptor()
    controls_unchanged = np.array_equal(carrier.controls, initial_controls)
    inverse(carrier, program, program.owner)
    inverse_stats = carrier.inverse_stats.descriptor()
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "public_program_json_bytes": len(canonical_json(program.descriptor())),
        "initial_commitment": initial,
        "final_commitment": final_commitment,
        "boundary": list(boundary),
        "hybrid_classical_boundary": list(classical_boundary),
        "boundary_identical_to_hybrid_classical_recurrence": boundary == classical_boundary,
        "streamed_final_relation_entry_checks": relation_checks,
        "final_ranks": final_ranks,
        "maximum_final_rank": max(final_ranks),
        "all_controls_unchanged": controls_unchanged,
        "phase_stats": stats,
        "inverse_stats": inverse_stats,
        "hybrid_classical_stats": classical_stats.descriptor(),
        "exact_restoration": state_commitment(carrier) == initial,
        "same_backing": carrier.backing_identity == backing,
        "restoration_generation_before": generation,
        "restoration_generation_after": carrier.restoration_generation,
        "projection_calls": carrier.projection_calls,
        "snapshot_reload_used": carrier.snapshot_reload_used,
        "inverse_history_cells": 0,
        "retained_restoration_baseline_cells": 0,
    }


def transaction(carrier: Carrier, program: Program) -> dict[str, Any]:
    initial = state_commitment(carrier)
    backing = carrier.backing_identity
    generation = carrier.restoration_generation
    begin_forward(carrier, program, program.owner)
    forward(carrier, program, program.owner)
    boundary = project(carrier, program, program.owner)
    final = state_commitment(carrier)
    forward_resource = carrier.stats.descriptor()
    inverse(carrier, program, program.owner)
    return {
        "boundary": list(boundary),
        "final_commitment": final,
        "exact_restoration": state_commitment(carrier) == initial,
        "same_backing": carrier.backing_identity == backing,
        "generation_before": generation,
        "generation_after": carrier.restoration_generation,
        "resource_signature": {
            "forward": forward_resource,
            "inverse": carrier.inverse_stats.descriptor(),
        },
    }


def reuse_controls() -> tuple[dict[str, Any], dict[str, Any]]:
    carrier = Carrier.seal()
    first = transaction(carrier, compile_program(7, "PRIMARY"))
    backing = carrier.backing_identity
    second_program = compile_program(37, "ALTERNATE")
    second = transaction(carrier, second_program)
    fresh = transaction(Carrier.seal(), second_program)
    unrelated = {
        "first_exact_restoration": first["exact_restoration"],
        "second_exact_restoration": second["exact_restoration"],
        "same_backing_across_programs": carrier.backing_identity == backing,
        "second_boundary_matches_fresh": second["boundary"] == fresh["boundary"],
        "second_final_commitment_matches_fresh": second["final_commitment"] == fresh["final_commitment"],
        "resource_signature_matches_fresh": second["resource_signature"] == fresh["resource_signature"],
        "restoration_generation": carrier.restoration_generation,
        "snapshot_reload_used": carrier.snapshot_reload_used,
    }
    repeated_carrier = Carrier.seal()
    repeated_initial = state_commitment(repeated_carrier)
    repeated_backing = repeated_carrier.backing_identity
    boundaries: set[tuple[int, ...]] = set()
    for _ in range(32):
        result = transaction(repeated_carrier, compile_program(4, "REUSE"))
        boundaries.add(tuple(result["boundary"]))
    repeated = {
        "cycles": 32,
        "exact_restoration": state_commitment(repeated_carrier) == repeated_initial,
        "same_backing": repeated_carrier.backing_identity == repeated_backing,
        "restoration_generation": repeated_carrier.restoration_generation,
        "stable_boundary_count": len(boundaries),
        "snapshot_reload_used": repeated_carrier.snapshot_reload_used,
    }
    return unrelated, repeated


def control_rank_and_reciprocal_certificate() -> dict[str, Any]:
    checks = 0
    for node in range(NODE_COUNT):
        control = seed_controls()[node].astype(np.int64)
        reciprocal = reciprocal_control(control)
        if rank_mod(control[0]) != 2 or rank_mod(control[1]) != 2:
            fail("rank-two control certificate failed")
        if rank_mod(reciprocal[0]) != 2 or rank_mod(reciprocal[1]) != 2:
            fail("rank-two reciprocal certificate failed")
        for x in range(CARDINALITY):
            for y in range(CARDINALITY):
                if factor_entry(control[0], control[1], x, y) * factor_entry(reciprocal[0], reciprocal[1], x, y) % MODULUS != 1:
                    fail("control reciprocal entry certificate failed")
                checks += 1
    return {
        "controls_checked": NODE_COUNT,
        "control_rank": 2,
        "reciprocal_control_rank": 2,
        "entrywise_inverse_checks": checks,
        "dense_relation_tables_materialized": 0,
    }


def controls() -> dict[str, bool]:
    program = compile_program(2, "PRIMARY")
    seed = Carrier.seal()
    missing = Carrier.seal()
    raw_forward(missing.controls, missing.payloads, missing.pivots, missing.ranks, program)
    wrong = Carrier.seal()
    raw_forward(wrong.controls, wrong.payloads, wrong.pivots, wrong.ranks, program)
    raw_inverse(wrong.controls, wrong.payloads, wrong.pivots, wrong.ranks, program, offset_mutation=1)
    reordered = Carrier.seal()
    raw_forward(reordered.controls, reordered.payloads, reordered.pivots, reordered.ranks, program)
    raw_inverse(reordered.controls, reordered.payloads, reordered.pivots, reordered.ranks, program, assumed_action_order="INTERSECT_COMPOSE")
    normal = Carrier.seal()
    raw_forward(normal.controls, normal.payloads, normal.pivots, normal.ranks, program)
    disabled = Carrier.seal()
    raw_forward(disabled.controls, disabled.payloads, disabled.pivots, disabled.ranks, program, port_enabled=False)
    swapped = Carrier.seal()
    raw_forward(swapped.controls, swapped.payloads, swapped.pivots, swapped.ranks, program, action_order="INTERSECT_COMPOSE")
    mutated = Carrier.seal()
    raw_forward(mutated.controls, mutated.payloads, mutated.pivots, mutated.ranks, program, hub_mutation=1)
    null_rejected = False
    try:
        begin_forward(None, program, program.owner)
    except RuntimeError:
        null_rejected = True
    wrong_type = Carrier.seal()
    wrong_type.port_type = "F103_DENSE_RELATION"
    wrong_type_rejected = False
    try:
        begin_forward(wrong_type, program, program.owner)
    except RuntimeError:
        wrong_type_rejected = True
    wrong_owner_rejected = False
    try:
        begin_forward(Carrier.seal(), program, program.owner ^ 1)
    except RuntimeError:
        wrong_owner_rejected = True
    leased = Carrier.seal()
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
    return {
        "missing_inverse_changes_state": state_commitment(missing) != state_commitment(seed),
        "wrong_inverse_changes_state": state_commitment(wrong) != state_commitment(seed),
        "reordered_inverse_changes_state": state_commitment(reordered) != state_commitment(seed),
        "null_carrier_rejected": null_rejected,
        "wrong_relation_type_rejected": wrong_type_rejected,
        "wrong_owner_rejected": wrong_owner_rejected,
        "premature_projection_rejected": premature_rejected,
        "resident_port_projection_rejected": resident_rejected,
        "null_port_changes_boundary": boundary_from_charts(normal.payloads, normal.pivots, normal.ranks, program) != boundary_from_charts(disabled.payloads, disabled.pivots, disabled.ranks, program),
        "composition_intersection_order_changes_boundary": boundary_from_charts(normal.payloads, normal.pivots, normal.ranks, program) != boundary_from_charts(swapped.payloads, swapped.pivots, swapped.ranks, program),
        "topology_mutation_changes_boundary": boundary_from_charts(normal.payloads, normal.pivots, normal.ranks, program) != boundary_from_charts(mutated.payloads, mutated.pivots, mutated.ranks, program),
        "resident_controls_remain_unmodified": np.array_equal(normal.controls, seed.controls),
    }


def run() -> dict[str, Any]:
    cases = [execute_case(depth, family) for family in FAMILIES for depth in DEPTHS]
    if not all(
        case["boundary_identical_to_hybrid_classical_recurrence"]
        and case["streamed_final_relation_entry_checks"] == NODE_COUNT * CARDINALITY * CARDINALITY
        and case["all_controls_unchanged"]
        and case["exact_restoration"]
        and case["same_backing"]
        and case["restoration_generation_after"] == case["restoration_generation_before"] + 1
        and case["projection_calls"] == 1
        and not case["snapshot_reload_used"]
        and case["inverse_history_cells"] == 0
        and case["retained_restoration_baseline_cells"] == 0
        for case in cases
    ):
        fail("one or more rank-adaptive relation cases failed")
    control_results = controls()
    if not all(control_results.values()):
        fail("rank-adaptive controls failed: " + repr([key for key, value in control_results.items() if not value]))
    reciprocal = control_rank_and_reciprocal_certificate()
    unrelated, repeated = reuse_controls()
    if not all((
        unrelated["first_exact_restoration"],
        unrelated["second_exact_restoration"],
        unrelated["same_backing_across_programs"],
        unrelated["second_boundary_matches_fresh"],
        unrelated["second_final_commitment_matches_fresh"],
        unrelated["resource_signature_matches_fresh"],
        not unrelated["snapshot_reload_used"],
        repeated["exact_restoration"],
        repeated["same_backing"],
        repeated["restoration_generation"] == 32,
        repeated["stable_boundary_count"] == 1,
        not repeated["snapshot_reload_used"],
    )):
        fail("rank-adaptive relation reuse failed")
    rank_growth = {
        family: {
            str(depth): next(case["maximum_final_rank"] for case in cases if case["family"] == family and case["depth"] == depth)
            for depth in DEPTHS
        }
        for family in FAMILIES
    }
    maximum_rank = max(case["maximum_final_rank"] for case in cases)
    if maximum_rank != MAX_RANK:
        fail("declared rank-seventeen saturation was not observed")
    target_payload_cells = NODE_COUNT * TARGET_PAYLOAD_CELLS
    target_pivot_bytes = NODE_COUNT * MAX_RANK
    target_rank_bytes = NODE_COUNT
    phase_resident_bytes = CONTROL_FACTOR_CELLS + target_payload_cells + target_pivot_bytes + target_rank_bytes
    maximum_program_bytes = max(case["public_program_json_bytes"] for case in cases)
    maximum_phase_scratch_coordinates = max(
        max(
            case["phase_stats"]["maximum_declared_scratch_field_coordinates"],
            case["inverse_stats"]["maximum_declared_scratch_field_coordinates"],
        )
        for case in cases
    )
    maximum_classical_value_coordinates = max(case["hybrid_classical_stats"]["maximum_resident_value_coordinates"] for case in cases)
    maximum_classical_metadata_bytes = max(case["hybrid_classical_stats"]["maximum_resident_metadata_bytes"] for case in cases)
    maximum_classical_chart_scratch_coordinates = max(
        case["hybrid_classical_stats"]["chart_stats"]["maximum_declared_scratch_field_coordinates"]
        for case in cases
    )
    maximum_classical_control_scratch_coordinates = max(
        case["hybrid_classical_stats"]["maximum_control_rematerialization_scratch_coordinates"]
        for case in cases
    )
    maximum_classical_resident_bytes = (
        maximum_classical_value_coordinates + maximum_classical_metadata_bytes
    )
    maximum_phase_named_warm_bytes = (
        phase_resident_bytes
        + 8 * maximum_phase_scratch_coordinates
        + maximum_program_bytes
    )
    maximum_classical_named_warm_bytes = (
        maximum_classical_resident_bytes
        + 8 * (
            maximum_classical_chart_scratch_coordinates
            + maximum_classical_control_scratch_coordinates
        )
        + maximum_program_bytes
    )
    depth128_work = {
        case["family"]: {
            "phase_forward_counted_field_multiplications": (
                case["phase_stats"]["matrix_field_multiplications"]
                + case["phase_stats"]["elimination_field_multiplications"]
            ),
            "phase_inverse_counted_field_multiplications": (
                case["inverse_stats"]["matrix_field_multiplications"]
                + case["inverse_stats"]["elimination_field_multiplications"]
            ),
            "classical_forward_counted_field_multiplications": (
                case["hybrid_classical_stats"]["chart_stats"]["matrix_field_multiplications"]
                + case["hybrid_classical_stats"]["chart_stats"]["elimination_field_multiplications"]
                + case["hybrid_classical_stats"]["dense_composition_multiplications"]
                + case["hybrid_classical_stats"]["dense_intersection_multiplications"]
            ),
            "phase_forward_to_classical_forward_multiplication_ratio": (
                (
                    case["phase_stats"]["matrix_field_multiplications"]
                    + case["phase_stats"]["elimination_field_multiplications"]
                )
                /
                (
                    case["hybrid_classical_stats"]["chart_stats"]["matrix_field_multiplications"]
                    + case["hybrid_classical_stats"]["chart_stats"]["elimination_field_multiplications"]
                    + case["hybrid_classical_stats"]["dense_composition_multiplications"]
                    + case["hybrid_classical_stats"]["dense_intersection_multiplications"]
                )
            ),
        }
        for case in cases
        if case["depth"] == 128
    }
    return {
        "schema": "CAT_CAS_F103_C17_RANK_ADAPTIVE_RELATION_CHART_NO_GO_RESULT_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_scope": "LINUX_DIRECT_PROCESS_EXACT_FINITE_FIELD_RANK_ADAPTIVE_OPEN_RELATION_SOFTWARE",
        "execution_scope": {
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "public_topology_compilation_reads_final_answers": False,
            "catvm_machine_boundary_used": False,
        },
        "relation_law": {
            "field_modulus": MODULUS,
            "boundary_cardinality": CARDINALITY,
            "port_type": PORT_TYPE,
            "translation_invariant": False,
            "control_rank": CONTROL_RANK,
            "reciprocal_control_rank": CONTROL_RANK,
            "maximum_relation_rank": MAX_RANK,
            "rank_r_chart_value_coordinates": "R_TIMES_34_MINUS_R",
            "maximum_chart_value_coordinates": TARGET_PAYLOAD_CELLS,
            "identity_plus_rank2_composition_preserves_rank": True,
            "rank2_intersection_can_double_rank": True,
            "rank_growth_by_family_and_depth": rank_growth,
            "maximum_rank_observed": maximum_rank,
            "dense_equivalent_at_rank17": True,
            "separate_dense_entry_table_materialized_on_phase_path": False,
            "rank17_resident_payload_is_relation_transpose_and_dense_equivalent": True,
            "factor_canonicalization_exact": True,
            "factor_canonicalization_streamed_without_assignment_expansion": True,
            "shared_unresolved_port_consumers_per_layer": 8,
            "resident_port_projection_before_boundary": False,
            "control_and_reciprocal_certificate": reciprocal,
        },
        "carrier_law": {
            "resident_control_factor_coordinates": CONTROL_FACTOR_CELLS,
            "resident_target_payload_coordinates": target_payload_cells,
            "resident_target_pivot_metadata_bytes": target_pivot_bytes,
            "resident_target_rank_metadata_bytes": target_rank_bytes,
            "resident_total_bytes": phase_resident_bytes,
            "fixed_backing_capacity_across_depth": True,
            "direct_process_type_and_owner_checks_observed": True,
            "machine_enforced_generation_or_lease_custody": False,
            "retained_public_plan_cells": 0,
        },
        "matched_classical_recurrence": {
            "implementation": "EXECUTED_REMATERIALIZED_CONTROL_RANK_ADAPTIVE_CHART_TO_DENSE_HYBRID",
            "chart_used_through_rank": 8,
            "dense_used_from_rank": 9,
            "maximum_resident_value_coordinates": maximum_classical_value_coordinates,
            "maximum_resident_metadata_bytes": maximum_classical_metadata_bytes,
            "maximum_resident_bytes": maximum_classical_resident_bytes,
            "phase_resident_bytes": phase_resident_bytes,
            "phase_to_classical_resident_byte_ratio": phase_resident_bytes / maximum_classical_resident_bytes,
            "depth128_counted_work_by_family": depth128_work,
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
            "canonical_relation_chart_is_machine_relevant_state": True,
            "unrelated_program_reuse": unrelated,
            "repeated_reuse": repeated,
        },
        "controls": control_results,
        "resource_accounting": {
            "phase_resident_bytes": phase_resident_bytes,
            "maximum_phase_declared_named_scratch_field_coordinates": maximum_phase_scratch_coordinates,
            "maximum_phase_declared_named_scratch_bytes_at_int64": 8 * maximum_phase_scratch_coordinates,
            "maximum_public_program_bytes": maximum_program_bytes,
            "maximum_classical_resident_value_coordinates": maximum_classical_value_coordinates,
            "maximum_classical_resident_metadata_bytes": maximum_classical_metadata_bytes,
            "maximum_classical_control_rematerialization_scratch_coordinates": maximum_classical_control_scratch_coordinates,
            "maximum_classical_chart_scratch_field_coordinates": maximum_classical_chart_scratch_coordinates,
            "maximum_phase_named_warm_bytes": maximum_phase_named_warm_bytes,
            "maximum_classical_named_warm_bytes": maximum_classical_named_warm_bytes,
            "factor_canonicalization_arithmetic_counted": True,
            "dense_hybrid_switch_and_dense_operations_counted": True,
            "excluded": [
                "PYTHON_CONTAINER_OVERHEAD",
                "PYTHON_OBJECT_ALLOCATOR",
                "NUMPY_AND_NATIVE_LIBRARY_INTERNAL_STORAGE",
                "WHOLE_PROCESS_PEAK",
            ],
        },
        "cases": cases,
        "claim_ceiling": "F103_NON_TRANSLATION_INVARIANT_C17_TO_C17_RELATIONS_REACHED_BY_RECIPROCAL_RANK2_CONTROLS_ON_THE_DECLARED9_NODE_ROTATING_HUB_ACROSS18_CASES_THROUGH_DEPTH128_IN_LINUX_DIRECT_PROCESS_SOFTWARE",
        "not_established": [
            "SUB_DENSE_FIXED_RANK_CLOSURE_ACROSS_GENERAL_RANK2_INTERSECTION",
            "INTERFACE_CARDINALITY_GROWTH_BEYOND17",
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
        "next_obstruction": "RANK_ADAPTIVE_CANONICALIZATION_CLOSES_THE_DECLARED_NONTRANSLATION_RELATION_ALGEBRA_BUT_INTERSECTION_SATURATES_RANK17_THE_CHART_BECOMES_DENSE_EQUIVALENT_AND_THE_EXECUTED_REMATERIALIZED_CONTROL_CLASSICAL_HYBRID_IS_SMALLER",
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
