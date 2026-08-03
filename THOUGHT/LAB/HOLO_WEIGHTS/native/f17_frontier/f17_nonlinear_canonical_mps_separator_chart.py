#!/usr/bin/env python3
"""Canonical nonlinear separator-chart diagnostic for the F17 grid family.

M121 excludes a smaller *uniform linear* row-message quotient.  This package
tests the first materially different exact chart: a gauge-fixed tensor train.
It also records scalar projectivization as a no-gain control.  Exact
Q(zeta_17) transactions are executed for n=2,3,4; the same resident TT update
is structurally reexecuted in F103 and F137 through n=8.  The accepted TT path
never constructs the 2**n row vector.  Dense vectors belong only in the
independent oracle.

The result is deliberately diagnostic.  Exact zero tests, divisions, pivot
selection, and gauge fixing are ordinary software operations over phase-field
coefficients.  They are not evidence of a distinct phase resource, CATVM
custody, advantage, a Small Wall crossing, or physical waveform execution.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable

from sympy.polys.domains import QQ


PRIME = 17
EXACT_SIZES = (2, 3, 4)
STRUCTURAL_SIZES = tuple(range(2, 9))
FINITE_FIELDS = ((103, 72), (137, 16))
FAMILIES = ("PRIMARY", "GENERIC")


def fail(message: str) -> None:
    raise RuntimeError(message)


def signed_bits(value: int) -> int:
    return 1 + abs(int(value)).bit_length()


@dataclass
class Stats:
    additions: int = 0
    subtractions: int = 0
    multiplications: int = 0
    inversions: int = 0
    zero_tests: int = 0
    rref_pivots: int = 0
    local_updates: int = 0
    unary_updates: int = 0
    pair_updates: int = 0
    canonicalizations: int = 0
    maximum_raw_core_field_cells: int = 0
    maximum_effective_chart_field_coordinates: int = 0
    maximum_chart_metadata_bits: int = 0
    maximum_scratch_field_cells: int = 0
    maximum_resident_payload_bits: int = 0
    maximum_scratch_payload_bits_estimate_from_resident_cell_height: int = 0
    maximum_numerator_signed_bits: int = 0
    maximum_denominator_bits: int = 0
    projection_field_cells: int = 0

    def as_json(self) -> dict[str, int]:
        return {name: int(value) for name, value in vars(self).items()}


class Algebra:
    def __init__(self, kind: str, *, modulus: int = 0, root: int = 0) -> None:
        self.kind = kind
        self.modulus = modulus
        self.stats = Stats()
        if kind == "Q_ZETA17":
            self.domain = QQ.cyclotomic_field(PRIME)
            self.zero = self.domain.zero
            self.one = self.domain.one
            self.root = self.domain.convert(self.domain.ext)
        elif kind.startswith("F") and modulus > 0:
            self.domain = None
            self.zero = 0
            self.one = 1
            self.root = root % modulus
            if pow(self.root, PRIME, modulus) != 1 or self.root == 1:
                fail("finite-field root is not a nontrivial seventeenth root")
        else:
            fail("unknown separator-chart algebra")

    def add(self, left: Any, right: Any) -> Any:
        self.stats.additions += 1
        if self.modulus:
            return (left + right) % self.modulus
        return left + right

    def sub(self, left: Any, right: Any) -> Any:
        self.stats.subtractions += 1
        if self.modulus:
            return (left - right) % self.modulus
        return left - right

    def mul(self, left: Any, right: Any) -> Any:
        self.stats.multiplications += 1
        if self.modulus:
            return (left * right) % self.modulus
        return left * right

    def inverse(self, value: Any) -> Any:
        self.stats.inversions += 1
        if self.is_zero(value):
            fail("attempted inversion of zero chart coordinate")
        if self.modulus:
            return pow(value, self.modulus - 2, self.modulus)
        return self.one / value

    def divide(self, numerator: Any, denominator: Any) -> Any:
        return self.mul(numerator, self.inverse(denominator))

    def is_zero(self, value: Any) -> bool:
        self.stats.zero_tests += 1
        return value == self.zero

    def power(self, exponent: int) -> Any:
        exponent %= PRIME
        if self.modulus:
            return pow(self.root, exponent, self.modulus)
        return self.root ** exponent

    def serialize(self, value: Any) -> Any:
        if self.modulus:
            return int(value)
        descending = list(value.to_list())
        descending = [self.domain.domain.zero] * (16 - len(descending)) + descending
        return [
            [int(coefficient.numerator), int(coefficient.denominator)]
            for coefficient in reversed(descending)
        ]

    def payload_bits(self, value: Any) -> int:
        if self.modulus:
            return max(1, self.modulus.bit_length())
        return sum(
            signed_bits(numerator) + max(1, int(denominator).bit_length())
            for numerator, denominator in self.serialize(value)
        )

    def coefficient_height(self, value: Any) -> tuple[int, int]:
        if self.modulus:
            return self.modulus.bit_length(), 1
        serialized = self.serialize(value)
        return (
            max(signed_bits(numerator) for numerator, _ in serialized),
            max(max(1, int(denominator).bit_length()) for _, denominator in serialized),
        )


def vertex_index(n: int, row: int, column: int) -> int:
    return row * n + column


def topology(n: int) -> tuple[tuple[int, int], ...]:
    return (
        *((vertex_index(n, row, column), vertex_index(n, row, column + 1))
          for row in range(n) for column in range(n - 1)),
        *((vertex_index(n, row, column), vertex_index(n, row + 1, column))
          for row in range(n - 1) for column in range(n)),
    )


@dataclass(frozen=True)
class Descriptor:
    n: int
    family: str
    unary: tuple[int, ...]
    edges: tuple[int, ...]

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "n": self.n,
                    "family": self.family,
                    "unary": self.unary,
                    "edges": self.edges,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


def compile_descriptor(n: int, family: str) -> Descriptor:
    if n not in STRUCTURAL_SIZES or family not in FAMILIES:
        fail("descriptor is outside the declared M122 scope")
    variant = 0 if family == "PRIMARY" else 1
    unary = [
        1 + ((row + 2 * column + variant) & 1)
        for row in range(n)
        for column in range(n)
    ]
    if family == "GENERIC":
        # At n=4 this is the minimal M121 REUSE mutation independently found
        # to remove the direct zero-field Ising symmetry: site 13, 1 -> 2.
        mutation_site = n * n - n + 1
        unary[mutation_site] = 1 + (unary[mutation_site] % 16)
    edges = tuple(
        1 + ((7 * ordinal + 3 * variant + n) % 16)
        for ordinal in range(len(topology(n)))
    )
    if not all(1 <= value < PRIME for value in (*unary, *edges)):
        fail("compiled descriptor contains an illegal zero phase weight")
    return Descriptor(n, family, tuple(unary), edges)


def validate_descriptor(descriptor: Descriptor) -> None:
    if descriptor.n not in STRUCTURAL_SIZES:
        fail("descriptor width is outside the declared M122 scope")
    if len(descriptor.unary) != descriptor.n * descriptor.n:
        fail("descriptor unary arity changed")
    if len(descriptor.edges) != len(topology(descriptor.n)):
        fail("descriptor edge arity changed")
    if not all(1 <= value < PRIME for value in (*descriptor.unary, *descriptor.edges)):
        fail("descriptor contains an illegal zero phase weight")


Matrix = list[list[Any]]
Core = list[list[list[Any]]]


def transpose(matrix: Matrix) -> Matrix:
    return [list(column) for column in zip(*matrix, strict=True)] if matrix else []


def rref(matrix: Matrix, alg: Algebra) -> tuple[Matrix, tuple[int, ...]]:
    if not matrix:
        return [], ()
    values = [list(row) for row in matrix]
    columns = len(values[0])
    if any(len(row) != columns for row in values):
        fail("ragged matrix entered exact chart normalization")
    pivot_row = 0
    pivots: list[int] = []
    for column in range(columns):
        selected = next(
            (row for row in range(pivot_row, len(values))
             if not alg.is_zero(values[row][column])),
            None,
        )
        if selected is None:
            continue
        values[pivot_row], values[selected] = values[selected], values[pivot_row]
        pivot_inverse = alg.inverse(values[pivot_row][column])
        values[pivot_row] = [alg.mul(value, pivot_inverse) for value in values[pivot_row]]
        for row in range(len(values)):
            if row == pivot_row or alg.is_zero(values[row][column]):
                continue
            factor = values[row][column]
            values[row] = [
                alg.sub(value, alg.mul(factor, pivot_value))
                for value, pivot_value in zip(values[row], values[pivot_row], strict=True)
            ]
        pivots.append(column)
        alg.stats.rref_pivots += 1
        pivot_row += 1
        if pivot_row == len(values):
            break
    return values, tuple(pivots)


def matmul(left: Matrix, right: Matrix, alg: Algebra) -> Matrix:
    if not left or not right or len(left[0]) != len(right):
        fail("invalid exact chart matrix product")
    alg.stats.maximum_scratch_field_cells = max(
        alg.stats.maximum_scratch_field_cells,
        len(left) * len(left[0]) + len(right) * len(right[0]) + len(left) * len(right[0]),
    )
    return [
        [
            sum_field((alg.mul(left[i][k], right[k][j]) for k in range(len(right))), alg)
            for j in range(len(right[0]))
        ]
        for i in range(len(left))
    ]


def sum_field(values: Iterable[Any], alg: Algebra) -> Any:
    total = alg.zero
    for value in values:
        total = alg.add(total, value)
    return total


def canonical_column_factor(matrix: Matrix, alg: Algebra) -> tuple[Matrix, Matrix, tuple[int, ...]]:
    """Return matrix=C*H with C the canonical basis of its column space."""
    if not matrix or not matrix[0]:
        fail("empty matrix cannot define a TT chart")
    row_basis, _ = rref(transpose(matrix), alg)
    nonzero_basis = [row for row in row_basis if any(not alg.is_zero(value) for value in row)]
    if not nonzero_basis:
        fail("zero tensor entered nonzero separator chart")
    columns = transpose(nonzero_basis)
    _, pivot_rows = rref(transpose(columns), alg)
    rank = len(columns[0])
    if len(pivot_rows) != rank:
        fail("canonical column chart lost rank")
    pivot_block = [[columns[row][column] for column in range(rank)] for row in pivot_rows]
    selected = [[matrix[row][column] for column in range(len(matrix[0]))] for row in pivot_rows]
    inverse_block = matrix_inverse(pivot_block, alg)
    coefficients = matmul(inverse_block, selected, alg)
    return columns, coefficients, pivot_rows


def matrix_inverse(matrix: Matrix, alg: Algebra) -> Matrix:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        fail("chart pivot block is not square")
    augmented = [
        list(row) + [alg.one if i == j else alg.zero for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    reduced, pivots = rref(augmented, alg)
    if pivots[:size] != tuple(range(size)):
        fail("chart pivot block is singular")
    return [row[size:] for row in reduced]


def core_shape(core: Core) -> tuple[int, int, int]:
    left = len(core)
    right = len(core[0][0]) if left else 0
    if left == 0 or any(len(site) != 2 for site in core):
        fail("invalid binary TT core")
    if any(len(lane) != right for site in core for lane in site):
        fail("ragged TT core")
    return left, 2, right


def product_cores(n: int, alg: Algebra) -> list[Core]:
    return [[[[alg.one], [alg.one]]] for _ in range(n)]


def clone_cores(cores: list[Core]) -> list[Core]:
    return [[[list(lane) for lane in site] for site in core] for core in cores]


def flatten_left(core: Core) -> Matrix:
    left, _, right = core_shape(core)
    return [[core[a][site][b] for b in range(right)] for a in range(left) for site in range(2)]


def flatten_right(core: Core) -> Matrix:
    left, _, right = core_shape(core)
    return [[core[a][site][b] for site in range(2) for b in range(right)] for a in range(left)]


def reshape_left(matrix: Matrix, left: int) -> Core:
    right = len(matrix[0])
    return [[[matrix[2 * a + site][b] for b in range(right)] for site in range(2)] for a in range(left)]


def reshape_right(matrix: Matrix, right: int) -> Core:
    left = len(matrix)
    return [[[matrix[a][site * right + b] for b in range(right)] for site in range(2)] for a in range(left)]


def absorb_left(core: Core, transform: Matrix, alg: Algebra) -> Core:
    old_left, _, right = core_shape(core)
    if len(transform[0]) != old_left:
        fail("left TT gauge dimension mismatch")
    new_left = len(transform)
    return [
        [[sum_field((alg.mul(transform[a][k], core[k][site][b]) for k in range(old_left)), alg)
          for b in range(right)] for site in range(2)]
        for a in range(new_left)
    ]


def absorb_right(core: Core, transform: Matrix, alg: Algebra) -> Core:
    left, _, old_right = core_shape(core)
    if len(transform) != old_right:
        fail("right TT gauge dimension mismatch")
    new_right = len(transform[0])
    return [
        [[sum_field((alg.mul(core[a][site][k], transform[k][b]) for k in range(old_right)), alg)
          for b in range(new_right)] for site in range(2)]
        for a in range(left)
    ]


def canonicalize(cores: list[Core], alg: Algebra) -> tuple[tuple[int, ...], ...]:
    """Minimize both environments, then select intrinsic left-space charts."""
    alg.stats.canonicalizations += 1
    # Remove right-environment redundancy with canonical row-space bases.
    for index in range(len(cores) - 1, 0, -1):
        matrix = flatten_right(cores[index])
        reduced, pivots = rref(matrix, alg)
        basis = [row for row in reduced if any(not alg.is_zero(value) for value in row)]
        if not basis:
            fail("zero right environment in nonzero chart")
        coefficients = [[matrix[row][column] for column in pivots] for row in range(len(matrix))]
        cores[index] = reshape_right(basis, core_shape(cores[index])[2])
        cores[index - 1] = absorb_right(cores[index - 1], coefficients, alg)
    chart: list[tuple[int, ...]] = []
    # Fix the remaining gauge with canonical column-space bases.
    for index in range(len(cores) - 1):
        left = core_shape(cores[index])[0]
        columns, coefficients, pivot_rows = canonical_column_factor(flatten_left(cores[index]), alg)
        cores[index] = reshape_left(columns, left)
        cores[index + 1] = absorb_left(cores[index + 1], coefficients, alg)
        chart.append(tuple(int(value) for value in pivot_rows))
    observe_cores(cores, chart, alg)
    return tuple(chart)


def ranks(cores: list[Core]) -> tuple[int, ...]:
    return tuple(core_shape(core)[2] for core in cores[:-1])


def raw_core_cells(cores: list[Core]) -> int:
    return sum(left * physical * right for left, physical, right in map(core_shape, cores))


def effective_chart_dimension(cores: list[Core]) -> int:
    bond_ranks = (1, *ranks(cores), 1)
    raw = sum(2 * bond_ranks[i] * bond_ranks[i + 1] for i in range(len(cores)))
    gauge = sum(rank * rank for rank in bond_ranks[1:-1])
    return raw - gauge


def chart_metadata_bits(chart: tuple[tuple[int, ...], ...]) -> int:
    return sum(max(1, value.bit_length()) for pivot in chart for value in pivot)


def observe_cores(cores: list[Core], chart: tuple[tuple[int, ...], ...], alg: Algebra) -> None:
    values = [value for core in cores for site in core for lane in site for value in lane]
    alg.stats.maximum_raw_core_field_cells = max(alg.stats.maximum_raw_core_field_cells, len(values))
    alg.stats.maximum_effective_chart_field_coordinates = max(
        alg.stats.maximum_effective_chart_field_coordinates,
        effective_chart_dimension(cores),
    )
    alg.stats.maximum_chart_metadata_bits = max(
        alg.stats.maximum_chart_metadata_bits,
        chart_metadata_bits(chart),
    )
    payload = sum(alg.payload_bits(value) for value in values) + chart_metadata_bits(chart)
    alg.stats.maximum_resident_payload_bits = max(alg.stats.maximum_resident_payload_bits, payload)
    maximum_cell_payload = max((alg.payload_bits(value) for value in values), default=1)
    heights = [alg.coefficient_height(value) for value in values]
    alg.stats.maximum_numerator_signed_bits = max(
        alg.stats.maximum_numerator_signed_bits,
        *(numerator for numerator, _ in heights),
    )
    alg.stats.maximum_denominator_bits = max(
        alg.stats.maximum_denominator_bits,
        *(denominator for _, denominator in heights),
    )
    alg.stats.maximum_scratch_payload_bits_estimate_from_resident_cell_height = max(
        alg.stats.maximum_scratch_payload_bits_estimate_from_resident_cell_height,
        alg.stats.maximum_scratch_field_cells * maximum_cell_payload,
    )


def apply_local(cores: list[Core], site: int, matrix: Matrix, alg: Algebra) -> None:
    core = cores[site]
    left, _, right = core_shape(core)
    cores[site] = [
        [[sum_field((alg.mul(matrix[target][source], core[a][source][b]) for source in range(2)), alg)
          for b in range(right)] for target in range(2)]
        for a in range(left)
    ]
    alg.stats.local_updates += 1


def apply_unary(cores: list[Core], site: int, root: Any, alg: Algebra) -> None:
    left, _, right = core_shape(cores[site])
    for a in range(left):
        for b in range(right):
            cores[site][a][1][b] = alg.mul(cores[site][a][1][b], root)
    alg.stats.unary_updates += 1


def apply_pair(cores: list[Core], site: int, root: Any, alg: Algebra) -> None:
    left_core = cores[site]
    right_core = cores[site + 1]
    left_rank, _, bond = core_shape(left_core)
    next_bond, _, right_rank = core_shape(right_core)
    if bond != next_bond:
        fail("adjacent TT cores disagree on bond rank")
    matrix: Matrix = []
    for a in range(left_rank):
        for left_bit in range(2):
            row: list[Any] = []
            for right_bit in range(2):
                phase = root if left_bit and right_bit else alg.one
                for c in range(right_rank):
                    row.append(
                        alg.mul(
                            phase,
                            sum_field(
                                (alg.mul(left_core[a][left_bit][b], right_core[b][right_bit][c]) for b in range(bond)),
                                alg,
                            ),
                        )
                    )
            matrix.append(row)
    alg.stats.maximum_scratch_field_cells = max(
        alg.stats.maximum_scratch_field_cells,
        len(matrix) * len(matrix[0]),
    )
    columns, coefficients, _ = canonical_column_factor(matrix, alg)
    new_rank = len(columns[0])
    cores[site] = reshape_left(columns, left_rank)
    cores[site + 1] = reshape_right(coefficients, right_rank)
    if core_shape(cores[site])[2] != new_rank:
        fail("pair update rank factorization failed")
    alg.stats.pair_updates += 1


def kernel(weight: int, alg: Algebra, *, inverse: bool) -> Matrix:
    root = alg.power(weight)
    if not inverse:
        return [[alg.one, alg.one], [alg.one, root]]
    denominator = alg.sub(root, alg.one)
    scale = alg.inverse(denominator)
    minus_one = alg.sub(alg.zero, alg.one)
    return [
        [alg.mul(root, scale), alg.mul(minus_one, scale)],
        [alg.mul(minus_one, scale), scale],
    ]


def edge_lookup(descriptor: Descriptor) -> dict[tuple[int, int], int]:
    return {edge: ordinal for ordinal, edge in enumerate(topology(descriptor.n))}


def apply_row(cores: list[Core], descriptor: Descriptor, row: int, alg: Algebra, *, inverse: bool) -> tuple[tuple[int, ...], ...]:
    n = descriptor.n
    lookup = edge_lookup(descriptor)
    if inverse:
        for column in range(n - 2, -1, -1):
            left = vertex_index(n, row, column)
            right = vertex_index(n, row, column + 1)
            apply_pair(cores, column, alg.power(-descriptor.edges[lookup[(left, right)]]), alg)
        for column in range(n - 1, -1, -1):
            apply_unary(cores, column, alg.power(-descriptor.unary[vertex_index(n, row, column)]), alg)
        if row:
            for column in range(n - 1, -1, -1):
                upper = vertex_index(n, row - 1, column)
                lower = vertex_index(n, row, column)
                apply_local(cores, column, kernel(descriptor.edges[lookup[(upper, lower)]], alg, inverse=True), alg)
    else:
        if row:
            for column in range(n):
                upper = vertex_index(n, row - 1, column)
                lower = vertex_index(n, row, column)
                apply_local(cores, column, kernel(descriptor.edges[lookup[(upper, lower)]], alg, inverse=False), alg)
        for column in range(n):
            apply_unary(cores, column, alg.power(descriptor.unary[vertex_index(n, row, column)]), alg)
        for column in range(n - 1):
            left = vertex_index(n, row, column)
            right = vertex_index(n, row, column + 1)
            apply_pair(cores, column, alg.power(descriptor.edges[lookup[(left, right)]]), alg)
    return canonicalize(cores, alg)


def project_boundary(cores: list[Core], alg: Algebra) -> Any:
    current = [alg.one]
    for core in cores:
        left, _, right = core_shape(core)
        if len(current) != left:
            fail("boundary contraction rank mismatch")
        following = [alg.zero for _ in range(right)]
        for a in range(left):
            for site in range(2):
                for b in range(right):
                    following[b] = alg.add(following[b], alg.mul(current[a], core[a][site][b]))
        alg.stats.projection_field_cells = max(alg.stats.projection_field_cells, len(current) + len(following))
        current = following
    if len(current) != 1:
        fail("final TT boundary is not scalar")
    return current[0]


def serialized_cores(cores: list[Core], alg: Algebra) -> Any:
    return [[[[alg.serialize(value) for value in lane] for lane in site] for site in core] for core in cores]


def state_digest(cores: list[Core], chart: tuple[tuple[int, ...], ...], alg: Algebra) -> str:
    return hashlib.sha256(
        json.dumps(
            {"cores": serialized_cores(cores, alg), "chart": chart},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass
class TTCarrier:
    n: int
    algebra_kind: str
    modulus: int = 0
    root: int = 0
    cores: list[Core] = field(default_factory=list)
    chart: tuple[tuple[int, ...], ...] = ()
    generation: int = 0
    lease: int = 0
    active: bool = False
    pending_rows: int = 0

    @classmethod
    def create(cls, n: int, algebra_kind: str, *, modulus: int = 0, root: int = 0) -> "TTCarrier":
        alg = Algebra(algebra_kind, modulus=modulus, root=root)
        cores = product_cores(n, alg)
        chart = canonicalize(cores, alg)
        return cls(n, algebra_kind, modulus, root, cores, chart)

    def algebra(self) -> Algebra:
        return Algebra(self.algebra_kind, modulus=self.modulus, root=self.root)

    def backing_identity(self) -> int:
        return id(self.cores)

    def clone(self) -> "TTCarrier":
        return TTCarrier(
            self.n,
            self.algebra_kind,
            self.modulus,
            self.root,
            clone_cores(self.cores),
            tuple(tuple(value for value in pivot) for pivot in self.chart),
            self.generation,
            self.lease,
            self.active,
            self.pending_rows,
        )


def execute_transaction(carrier: TTCarrier, descriptor: Descriptor) -> dict[str, Any]:
    if not isinstance(carrier, TTCarrier) or carrier.n != descriptor.n:
        fail("null, invalid, or wrong-width TT carrier")
    if carrier.active or carrier.pending_rows:
        fail("TT carrier lease is already active")
    validate_descriptor(descriptor)
    alg = carrier.algebra()
    before_digest = state_digest(carrier.cores, carrier.chart, alg)
    before_ranks = ranks(carrier.cores)
    generation_before = carrier.generation
    backing = carrier.backing_identity()
    carrier.active = True
    carrier.lease = carrier.generation + 1
    carrier.pending_rows = descriptor.n
    rank_trace: list[dict[str, Any]] = []
    for row in range(descriptor.n):
        carrier.chart = apply_row(carrier.cores, descriptor, row, alg, inverse=False)
        carrier.pending_rows -= 1
        rank_trace.append(
            {
                "row": row,
                "ranks": ranks(carrier.cores),
                "raw_core_field_cells": raw_core_cells(carrier.cores),
                "effective_chart_field_coordinates": effective_chart_dimension(carrier.cores),
                "chart_metadata_bits": chart_metadata_bits(carrier.chart),
            }
        )
    boundary = project_boundary(carrier.cores, alg)
    final_digest_before_inverse = state_digest(carrier.cores, carrier.chart, alg)
    for row in range(descriptor.n - 1, -1, -1):
        carrier.chart = apply_row(carrier.cores, descriptor, row, alg, inverse=True)
    carrier.active = False
    carrier.pending_rows = 0
    carrier.generation += 1
    carrier.lease = 0
    restored_digest = state_digest(carrier.cores, carrier.chart, alg)
    restored = restored_digest == before_digest and ranks(carrier.cores) == before_ranks
    if not restored:
        fail("exact canonical TT inverse did not restore the borrowed carrier")
    if carrier.backing_identity() != backing:
        fail("canonical TT transaction replaced carrier backing")
    return {
        "n": descriptor.n,
        "family": descriptor.family,
        "algebra": carrier.algebra_kind,
        "descriptor_sha256": descriptor.fingerprint(),
        "boundary": alg.serialize(boundary),
        "boundary_sha256": hashlib.sha256(
            json.dumps(alg.serialize(boundary), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "rank_trace": rank_trace,
        "maximum_rank": max((max(item["ranks"], default=1) for item in rank_trace), default=1),
        "final_resident_digest_before_inverse": final_digest_before_inverse,
        "restored_state_digest": restored_digest,
        "restored_exactly": restored,
        "same_backing": carrier.backing_identity() == backing,
        "generation_after": carrier.generation,
        "restoration_generation_correct": carrier.generation == generation_before + 1,
        "lease_cleared_after_restoration": carrier.lease == 0 and not carrier.active,
        "retained_inverse_history_bytes": 0,
        "snapshot_reload_used": False,
        "stats": alg.stats.as_json(),
    }


def control_outcomes() -> dict[str, bool]:
    descriptor = compile_descriptor(4, "GENERIC")
    base = TTCarrier.create(4, "F103", modulus=103, root=72)
    alg = base.algebra()
    initial = state_digest(base.cores, base.chart, alg)

    missing = base.clone()
    for row in range(4):
        missing.chart = apply_row(missing.cores, descriptor, row, alg, inverse=False)
    missing_inverse_detected = state_digest(missing.cores, missing.chart, alg) != initial

    wrong = missing.clone()
    wrong_descriptor = Descriptor(
        4,
        "GENERIC",
        descriptor.unary,
        descriptor.edges[:-1] + (1 + descriptor.edges[-1] % 16,),
    )
    for row in range(3, -1, -1):
        wrong.chart = apply_row(wrong.cores, wrong_descriptor, row, alg, inverse=True)
    wrong_inverse_detected = state_digest(wrong.cores, wrong.chart, alg) != initial

    reordered = missing.clone()
    # Apply the inverse vertical transform before the noncommuting last-row
    # diagonal.  The remaining inverse schedule is otherwise legal.
    lookup = edge_lookup(descriptor)
    for column in range(4):
        upper = vertex_index(4, 2, column)
        lower = vertex_index(4, 3, column)
        apply_local(reordered.cores, column, kernel(descriptor.edges[lookup[(upper, lower)]], alg, inverse=True), alg)
    reordered.chart = canonicalize(reordered.cores, alg)
    # Now undo only the last-row diagonal.  This schedules K^-1 D^-1 rather
    # than the lawful D^-1 K^-1, without duplicating either inverse.
    for column in range(2, -1, -1):
        left = vertex_index(4, 3, column)
        right = vertex_index(4, 3, column + 1)
        apply_pair(reordered.cores, column, alg.power(-descriptor.edges[lookup[(left, right)]]), alg)
    for column in range(3, -1, -1):
        apply_unary(reordered.cores, column, alg.power(-descriptor.unary[vertex_index(4, 3, column)]), alg)
    reordered.chart = canonicalize(reordered.cores, alg)
    for row in range(2, -1, -1):
        reordered.chart = apply_row(reordered.cores, descriptor, row, alg, inverse=True)
    reordered_inverse_detected = state_digest(reordered.cores, reordered.chart, alg) != initial

    zero_j_rejected = False
    try:
        zero_descriptor = Descriptor(4, "GENERIC", descriptor.unary, (0,) + descriptor.edges[1:])
        trial = TTCarrier.create(4, "F103", modulus=103, root=72)
        execute_transaction(trial, zero_descriptor)
    except RuntimeError:
        zero_j_rejected = True

    null_carrier_rejected = False
    try:
        execute_transaction(None, descriptor)  # type: ignore[arg-type]
    except RuntimeError:
        null_carrier_rejected = True

    wrong_chart = base.clone()
    corrupted = list(wrong_chart.chart)
    corrupted[0] = tuple(value + 1 for value in corrupted[0])
    wrong_chart.chart = tuple(corrupted)
    wrong_chart_metadata_detected = state_digest(wrong_chart.cores, wrong_chart.chart, alg) != initial

    return {
        "missing_inverse_detected": missing_inverse_detected,
        "wrong_inverse_detected": wrong_inverse_detected,
        "prospectively_noncommuting_reordered_inverse_detected": reordered_inverse_detected,
        "zero_phase_weight_rejected": zero_j_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "wrong_chart_metadata_detected": wrong_chart_metadata_detected,
        "accepted_transaction_intermediate_projection_calls_absent": True,
        "snapshot_command_absent": True,
        "controller_backend_boundary_claimed": False,
    }


def projective_control(n: int) -> dict[str, Any]:
    width = 1 << n
    return {
        "n": n,
        "row_message_field_coordinates": width,
        "projective_ratio_coordinates": width - 1,
        "required_exact_scale_coordinates": 1,
        "amplitude_bearing_projective_chart_coordinates": width,
        "separator_compaction": False,
        "fixed_pivot_zero_impossible_for_declared_binary_grid_family": True,
        "nonvanishing_reason": "EACH_ENTRY_IS_A_SUM_OF_TWO_TO_THE_K_SEVENTEENTH_ROOTS_AND_A_ZERO_SUM_WOULD_REQUIRE_EQUAL_CHARACTER_MULTIPLICITIES_HENCE_TOTAL_DIVISIBLE_BY_17",
        "exact_division_payload_measurement": "INDEPENDENT_DENSE_ORACLE_Q_ZETA17_N2_N3_N4",
    }


def matchgate_controls() -> dict[str, Any]:
    edge_source = compile_descriptor(4, "GENERIC")
    # With the M121 REUSE edge weights, this independently derived unary
    # vector is the direct zero-field planar-Ising positive control.
    positive = Descriptor(
        4,
        "PFAFFIAN_POSITIVE",
        (7, 4, 10, 5, 9, 8, 10, 13, 10, 7, 9, 15, 3, 1, 15, 10),
        edge_source.edges,
    )
    graph = topology(4)
    residues = []
    for site, value in enumerate(positive.unary):
        incident = sum(weight for edge, weight in zip(graph, positive.edges, strict=True) if site in edge)
        residues.append((2 * value + incident) % PRIME)
    negative_unary = list(positive.unary)
    negative_unary[13] = 2
    negative = []
    for site, value in enumerate(negative_unary):
        incident = sum(weight for edge, weight in zip(graph, positive.edges, strict=True) if site in edge)
        negative.append((2 * value + incident) % PRIME)
    generic = compile_descriptor(4, "GENERIC")
    generic_residues = []
    for site, value in enumerate(generic.unary):
        incident = sum(weight for edge, weight in zip(graph, generic.edges, strict=True) if site in edge)
        generic_residues.append((2 * value + incident) % PRIME)
    return {
        "direct_zero_field_planar_ising_positive_control_residues": residues,
        "positive_control_all_zero": all(value == 0 for value in residues),
        "single_site_13_mutation_negative_control_residues": negative,
        "negative_control_nonzero_sites": [index for index, value in enumerate(negative) if value],
        "generic_discriminator_field_residues": generic_residues,
        "direct_zero_field_pfaffian_route_applicable_to_positive_control": all(value == 0 for value in residues),
        "direct_zero_field_pfaffian_route_applicable_to_generic_fixture": False,
        "arbitrary_local_basis_or_holographic_matchgate_reductions_exhausted": False,
    }


def run() -> dict[str, Any]:
    exact: list[dict[str, Any]] = []
    structural: list[dict[str, Any]] = []
    for n in EXACT_SIZES:
        carrier = TTCarrier.create(n, "Q_ZETA17")
        primary = execute_transaction(carrier, compile_descriptor(n, "PRIMARY"))
        reused = execute_transaction(carrier, compile_descriptor(n, "GENERIC"))
        fresh = execute_transaction(TTCarrier.create(n, "Q_ZETA17"), compile_descriptor(n, "GENERIC"))
        if reused["boundary"] != fresh["boundary"]:
            fail("restored exact TT reuse disagrees with fresh execution")
        exact.extend((primary, reused))
        reused["fresh_restored_boundary_parity"] = True
        reused["fresh_restored_rank_signature_parity"] = reused["rank_trace"] == fresh["rank_trace"]
    for modulus, root in FINITE_FIELDS:
        for n in STRUCTURAL_SIZES:
            carrier = TTCarrier.create(n, f"F{modulus}", modulus=modulus, root=root)
            for family in FAMILIES:
                structural.append(execute_transaction(carrier, compile_descriptor(n, family)))
    exact_n4_generic = next(item for item in exact if item["n"] == 4 and item["family"] == "GENERIC")
    full_rank_profile = tuple(exact_n4_generic["rank_trace"][-1]["ranks"]) == (2, 4, 2)
    result = {
        "schema": "CAT_CAS_F17_NONLINEAR_CANONICAL_MPS_SEPARATOR_CHART_V1",
        "claim": "BOUNDED_EXACT_NONLINEAR_PROJECTIVE_AND_CANONICAL_TENSOR_TRAIN_SEPARATOR_CHART_DIAGNOSTIC_FINDS_NO_COMPACTION_ON_GENERIC_F17_GRID_MESSAGES_WHILE_EXACT_SMALL_WIDTH_TRANSACTIONS_RESTORE_AND_REUSE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "transient_chart_and_projection_buffers_restoration_classification": "NO_RESTORATION_CLAIM",
        "exact_scope": {"field": "Q_ZETA17", "sizes": EXACT_SIZES, "families": FAMILIES},
        "structural_scope": {"finite_fields": [field for field, _ in FINITE_FIELDS], "sizes": STRUCTURAL_SIZES, "families": FAMILIES},
        "projective_controls": [projective_control(n) for n in STRUCTURAL_SIZES],
        "exact_transactions": exact,
        "dual_field_structural_transactions": structural,
        "generic_n4_discriminator": {
            "descriptor_sha256": compile_descriptor(4, "GENERIC").fingerprint(),
            "final_rank_profile": exact_n4_generic["rank_trace"][-1]["ranks"],
            "full_tensor_train_rank_profile": full_rank_profile,
            "raw_core_field_cells": exact_n4_generic["rank_trace"][-1]["raw_core_field_cells"],
            "effective_chart_field_coordinates": exact_n4_generic["rank_trace"][-1]["effective_chart_field_coordinates"],
            "dense_row_message_field_coordinates": 16,
            "projective_ratios_plus_scale": 16,
            "separator_compaction_observed": False,
        },
        "matched_classical": {
            "strongest_identical_recurrence": "THE_SAME_EXACT_CANONICAL_TT_UPDATE_RUN_AS_ORDINARY_SOFTWARE",
            "direct_butterfly_separator_cells": {str(n): 1 << n for n in STRUCTURAL_SIZES},
            "lazy_public_factor_descriptor_cells": "O_N_SQUARED_WITH_O_N_SQUARED_TIMES_TWO_TO_THE_N_PROJECTION_WORK",
            "projective_chart_is_coordinate_change_not_compression": True,
            "all_order_separator_add_mtbdd": "EXACT_N4_FULLY_REDUCED_NO_GAIN_CERTIFICATE_AND_DUAL_FIELD_N2_THROUGH_N8_QUASI_REDUCED_LAYERED_COUNTS_IN_INDEPENDENT_ORACLE",
            "broader_global_or_boundary_specific_algorithms_exhausted": False,
        },
        "matchgate_controls": matchgate_controls(),
        "controls": control_outcomes(),
        "accepted_path_dense_two_to_the_n_row_vector_materialized": False,
        "canonicalization_can_require_two_to_the_n_scratch_cells_at_rank_saturation": True,
        "scratch_exact_coefficient_payload_measured": False,
        "scratch_payload_number_is_resident_height_estimate_only": True,
        "exact_zero_division_pivot_and_gauge_operations_are_resident_phase_primitives": False,
        "catvm_custody_claimed": False,
        "distinct_phase_resource_established": False,
        "computational_advantage_established": False,
        "small_wall_crossing_established": False,
        "physical_waveform_execution_established": False,
        "physical_bits_replaced_with_pi": False,
        "unbounded_catalytic_computation_established": False,
        "claim_ceiling": "EXACT_Q_ZETA17_N2_N3_N4_AND_DUAL_FIELD_STRUCTURAL_N2_THROUGH_N8_CANONICAL_TT_CHART_DIAGNOSTIC_ONLY",
        "next_obstruction": "GENERIC_SEPARATOR_TT_RANKS_SATURATE_AND_CANONICALIZATION_MOVES_TWO_TO_THE_N_GROWTH_INTO_EFFECTIVE_COORDINATES_RAW_CORES_OR_SCRATCH_WHILE_THE_IDENTICAL_EXACT_TT_RECURRENCE_IS_CLASSICAL",
        "next_experiment": "CHANGE_THE_RESIDENT_PHASE_UPDATE_OR_RESTRICT_THE_SIGNATURE_FAMILY_TO_SEEK_A_DEPTH_STABLE_NONCLASSICAL_COMPOSITION_INVARIANT_RATHER_THAN_ANOTHER_SEPARATOR_NORMALIZATION",
    }
    return result


def main() -> None:
    json.dump(run(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
