#!/usr/bin/env python3
"""Independent dense and planar cycle-space oracle for M123.

This file does not import the terminal-Pfaffian implementation.  It rebuilds
the public programs, evaluates their binary phase sums by residue histograms,
and separately evaluates the spin transform using the bounded-face cycle
basis of the open square grid.  The latter enumerates the oracle cycle space;
it is verification work, not an accepted compact carrier path.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any, Iterable

from sympy.polys.domains import QQ


PRIME = 17
EXACT_CASES = ((2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 4))
DEFECT_SITES_N4 = (5, 6, 9, 10)
DEFECT_DELTAS_N4 = (1, 2, 3, 4)


def fail(message: str) -> None:
    raise RuntimeError(message)


class Field:
    def __init__(self, modulus: int = 0, root: int = 0) -> None:
        self.modulus = modulus
        if modulus:
            self.zero = 0
            self.one = 1
            self.root = root % modulus
            if pow(self.root, PRIME, modulus) != 1 or self.root == 1:
                fail("oracle finite-field root is invalid")
            self.domain = None
        else:
            self.domain = QQ.cyclotomic_field(PRIME)
            self.zero = self.domain.zero
            self.one = self.domain.one
            self.root = self.domain.convert(self.domain.ext)

    def add(self, left: Any, right: Any) -> Any:
        if self.modulus:
            return (left + right) % self.modulus
        return left + right

    def mul(self, left: Any, right: Any) -> Any:
        if self.modulus:
            return (left * right) % self.modulus
        return left * right

    def sub(self, left: Any, right: Any) -> Any:
        if self.modulus:
            return (left - right) % self.modulus
        return left - right

    def integer(self, value: int) -> Any:
        if self.modulus:
            return value % self.modulus
        return self.domain.convert(value)

    def half(self) -> Any:
        if self.modulus:
            return pow(2, self.modulus - 2, self.modulus)
        return self.domain.convert(1) / self.domain.convert(2)

    def phase(self, exponent: int) -> Any:
        if self.modulus:
            return pow(self.root, exponent % PRIME, self.modulus)
        return self.root ** (exponent % PRIME)

    def serialize(self, value: Any) -> Any:
        if self.modulus:
            return int(value)
        descending = list(value.to_list())
        descending = [self.domain.domain.zero] * (16 - len(descending)) + descending
        return [
            [int(coefficient.numerator), int(coefficient.denominator)]
            for coefficient in reversed(descending)
        ]


def product(field: Field, values: Iterable[Any]) -> Any:
    result = field.one
    for value in values:
        result = field.mul(result, value)
    return result


def vertex(n: int, row: int, column: int) -> int:
    return row * n + column


def grid_edges(n: int) -> tuple[tuple[int, int], ...]:
    horizontal = tuple(
        (vertex(n, row, column), vertex(n, row, column + 1))
        for row in range(n)
        for column in range(n - 1)
    )
    vertical = tuple(
        (vertex(n, row, column), vertex(n, row + 1, column))
        for row in range(n - 1)
        for column in range(n)
    )
    return horizontal + vertical


@dataclass(frozen=True)
class Program:
    n: int
    unary: tuple[int, ...]
    edges: tuple[int, ...]
    defects: tuple[int, ...]


def compile_program(n: int, defect_count: int) -> Program:
    edges = tuple(1 + ((7 * ordinal + n) % 16) for ordinal in range(len(grid_edges(n))))
    unary = []
    for site in range(n * n):
        incident = sum(
            weight
            for edge, weight in zip(grid_edges(n), edges, strict=True)
            if site in edge
        )
        unary.append((-9 * incident) % PRIME)
    sites: tuple[int, ...] = ()
    if n == 4:
        if not 0 <= defect_count <= len(DEFECT_SITES_N4):
            fail("oracle defect count is outside the declared exact cases")
        sites = DEFECT_SITES_N4[:defect_count]
        for site, delta in zip(sites, DEFECT_DELTAS_N4[:defect_count], strict=True):
            unary[site] = (unary[site] + delta) % PRIME
    elif defect_count:
        fail("oracle has no defects for this width")
    return Program(n, tuple(unary), edges, sites)


def dense_histogram(program: Program) -> tuple[int, ...]:
    counts = [0] * PRIME
    graph = grid_edges(program.n)
    for assignment in range(1 << (program.n * program.n)):
        exponent = 0
        for site, unary in enumerate(program.unary):
            if (assignment >> site) & 1:
                exponent += unary
        for (left, right), edge_weight in zip(graph, program.edges, strict=True):
            if ((assignment >> left) & 1) and ((assignment >> right) & 1):
                exponent += edge_weight
        counts[exponent % PRIME] += 1
    return tuple(counts)


def histogram_boundary(histogram: tuple[int, ...], field: Field) -> Any:
    result = field.zero
    for exponent, multiplicity in enumerate(histogram):
        result = field.add(result, field.mul(field.integer(multiplicity), field.phase(exponent)))
    return result


def phase_pair(field: Field, exponent: int) -> tuple[Any, Any]:
    forward = field.phase(exponent)
    backward = field.phase(-exponent)
    return (
        field.mul(field.half(), field.add(forward, backward)),
        field.mul(field.half(), field.sub(forward, backward)),
    )


def bounded_face_masks(n: int) -> tuple[int, ...]:
    lookup = {tuple(sorted(edge)): index for index, edge in enumerate(grid_edges(n))}
    masks = []
    for row in range(n - 1):
        for column in range(n - 1):
            corners = (
                vertex(n, row, column),
                vertex(n, row, column + 1),
                vertex(n, row + 1, column + 1),
                vertex(n, row + 1, column),
            )
            mask = 0
            for left, right in zip(corners, (*corners[1:], corners[0]), strict=True):
                mask ^= 1 << lookup[tuple(sorted((left, right)))]
            masks.append(mask)
    return tuple(masks)


def path_mask(n: int, left: int, right: int, *, alternate: bool = True) -> int:
    lookup = {tuple(sorted(edge)): index for index, edge in enumerate(grid_edges(n))}
    row, column = divmod(left, n)
    target_row, target_column = divmod(right, n)
    vertices = [left]
    if alternate:
        while row != target_row:
            row += 1 if target_row > row else -1
            vertices.append(vertex(n, row, column))
        while column != target_column:
            column += 1 if target_column > column else -1
            vertices.append(vertex(n, row, column))
    else:
        while column != target_column:
            column += 1 if target_column > column else -1
            vertices.append(vertex(n, row, column))
        while row != target_row:
            row += 1 if target_row > row else -1
            vertices.append(vertex(n, row, column))
    result = 0
    for a, b in zip(vertices, vertices[1:]):
        result ^= 1 << lookup[tuple(sorted((a, b)))]
    return result


def chain_mask(n: int, subset: tuple[int, ...]) -> int:
    if len(subset) % 2:
        fail("oracle received an odd defect subset")
    result = 0
    for left, right in zip(subset[::2], subset[1::2], strict=True):
        result ^= path_mask(n, left, right)
    return result


def cycle_space_masks(n: int) -> Iterable[int]:
    faces = bounded_face_masks(n)
    for choice in range(1 << len(faces)):
        mask = 0
        for index, face in enumerate(faces):
            if (choice >> index) & 1:
                mask ^= face
        yield mask


def spin_cycle_boundary(program: Program, field: Field) -> tuple[Any, dict[str, int]]:
    graph = grid_edges(program.n)
    residues = tuple(
        (
            2 * program.unary[site]
            + sum(
                weight
                for edge, weight in zip(graph, program.edges, strict=True)
                if site in edge
            )
        )
        % PRIME
        for site in range(program.n * program.n)
    )
    defects = tuple(index for index, residue in enumerate(residues) if residue)
    if defects != program.defects:
        fail("oracle independently found different defect ownership")
    constant = (9 * sum(program.unary) + 13 * sum(program.edges)) % PRIME
    edge_pairs = tuple(phase_pair(field, 13 * weight) for weight in program.edges)
    edge_c = tuple(pair[0] for pair in edge_pairs)
    edge_d = tuple(pair[1] for pair in edge_pairs)
    fields = tuple(phase_pair(field, -13 * residues[site]) for site in defects)
    accumulator = field.zero
    sector_count = 0
    cycle_terms = 0
    for size in range(0, len(defects) + 1, 2):
        for subset in itertools.combinations(defects, size):
            insertion = set(subset)
            coefficient = product(
                field,
                (
                    fields[index][1] if site in insertion else fields[index][0]
                    for index, site in enumerate(defects)
                ),
            )
            chain = chain_mask(program.n, subset)
            sector = field.zero
            for even_mask in cycle_space_masks(program.n):
                selected = even_mask ^ chain
                weight = product(
                    field,
                    (
                        edge_d[edge_index]
                        if (selected >> edge_index) & 1
                        else edge_c[edge_index]
                        for edge_index in range(len(graph))
                    ),
                )
                sector = field.add(sector, weight)
                cycle_terms += 1
            accumulator = field.add(accumulator, field.mul(coefficient, sector))
            sector_count += 1
    boundary = field.mul(
        field.phase(constant),
        field.mul(field.integer(1 << (program.n * program.n)), accumulator),
    )
    return boundary, {
        "defect_count": len(defects),
        "even_sector_count": sector_count,
        "bounded_face_cycle_rank": (program.n - 1) ** 2,
        "cycle_space_terms_evaluated": cycle_terms,
    }


def exact_case(n: int, defect_count: int) -> dict[str, Any]:
    program = compile_program(n, defect_count)
    histogram = dense_histogram(program)
    dense_field = Field()
    cycle_field = Field()
    dense = histogram_boundary(histogram, dense_field)
    cycle, metadata = spin_cycle_boundary(program, cycle_field)
    if dense_field.serialize(dense) != cycle_field.serialize(cycle):
        fail("independent dense and cycle-space exact oracles disagree")
    return {
        "n": n,
        "defect_count": defect_count,
        "boundary": dense_field.serialize(dense),
        "residue_histogram": histogram,
        "dense_assignment_count": 1 << (n * n),
        "dense_cycle_exact_agreement": True,
        **metadata,
    }


def modular_controls() -> list[dict[str, Any]]:
    results = []
    for modulus, root in ((103, 72), (137, 16)):
        for n, defect_count in EXACT_CASES:
            program = compile_program(n, defect_count)
            histogram = dense_histogram(program)
            dense_field = Field(modulus, root)
            cycle_field = Field(modulus, root)
            dense = histogram_boundary(histogram, dense_field)
            cycle, metadata = spin_cycle_boundary(program, cycle_field)
            if dense != cycle:
                fail("independent modular dense and cycle-space oracles disagree")
            results.append(
                {
                    "field": f"F{modulus}",
                    "n": n,
                    "defect_count": defect_count,
                    "boundary": dense,
                    "dense_cycle_agreement": True,
                    **metadata,
                }
            )
    return results


def run() -> dict[str, Any]:
    exact = [exact_case(n, defect_count) for n, defect_count in EXACT_CASES]
    payload = {
        "schema": "CAT_CAS_F17_PLANAR_FREE_FERMION_PHASE_COVARIANCE_CLOSURE_ORACLE_V1",
        "implementation": "INDEPENDENT_BINARY_RESIDUE_HISTOGRAM_AND_BOUNDED_FACE_CYCLE_SPACE",
        "imports_terminal_pfaffian_implementation": False,
        "exact_cases": exact,
        "modular_controls": modular_controls(),
        "claim_ceiling": {
            "cycle_space_enumeration_is_accepted_path": False,
            "oracle_enumeration_proves_compact_execution": False,
            "independently_checks_spin_transform_and_sparse_defect_boundaries": True,
        },
    }
    payload["oracle_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
