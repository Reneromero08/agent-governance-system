#!/usr/bin/env python3
"""Exact bounded free-fermion closure for the planar F17 grid sector.

The accepted path maps the declared binary phase program to an Ising spin
program.  A public zero-field gate selects the free-fermion sector.  Its
even-subgraph sum is evaluated by a square-root-free terminal-graph
Pfaffian over Q(zeta_17).  Sparse external fields are expanded only over
even insertion sets and therefore expose the exact 2**(k-1) defect law.

The resident object is an antisymmetric phase-field carrier plus one scalar
accumulator.  Terminal entries are loaded by exact reversible additions,
the final scalar is projected once, and the commuting contributions are
rematerialized deterministically to restore the same backing.  Dense
Pfaffian elimination buffers are transient projection work and carry
NO_RESTORATION_CLAIM.

This is a bounded restricted-family diagnostic.  Topology compilation,
orientation, pivoting, division, and Pfaffian elimination are ordinary
software.  The identical Pfaffian is the matched classical recurrence.  No
CATVM custody, distinct phase resource, advantage, Small Wall crossing,
physical waveform execution, bit replacement, or unbounded claim follows.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import networkx as nx

import f17_nonlinear_canonical_mps_separator_chart as m122


PRIME = 17
EXACT_WIDTHS = (2, 3, 4)
MODULAR_FIELDS = ((103, 72), (137, 16))
STRUCTURAL_WIDTHS = tuple(range(2, 9))
DEFECT_SITES_N4 = (5, 6, 9, 10)
DEFECT_DELTAS_N4 = (1, 2, 3, 4)
DEFECT_SITES_N6 = (7, 9, 10, 14, 16, 19, 21, 26)
DEFECT_DELTAS_N6 = (1, 2, 3, 4, 5, 6, 7, 8)


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def scalar(alg: m122.Algebra, value: int) -> Any:
    if alg.modulus:
        return value % alg.modulus
    return alg.domain.convert(value)


def rational_half(alg: m122.Algebra) -> Any:
    if alg.modulus:
        return pow(2, alg.modulus - 2, alg.modulus)
    return alg.domain.convert(1) / alg.domain.convert(2)


def neg(alg: m122.Algebra, value: Any) -> Any:
    return alg.sub(alg.zero, value)


def product(alg: m122.Algebra, values: Iterable[Any]) -> Any:
    result = alg.one
    for value in values:
        result = alg.mul(result, value)
    return result


@dataclass(frozen=True)
class Program:
    n: int
    family: str
    unary: tuple[int, ...]
    edges: tuple[int, ...]
    declared_defect_sites: tuple[int, ...]

    def fingerprint(self) -> str:
        return sha256_json(
            {
                "n": self.n,
                "family": self.family,
                "unary": self.unary,
                "edges": self.edges,
                "declared_defect_sites": self.declared_defect_sites,
            }
        )


def zero_field_unary(n: int, edges: tuple[int, ...]) -> tuple[int, ...]:
    graph = m122.topology(n)
    result = []
    for vertex in range(n * n):
        incident = sum(
            weight
            for edge, weight in zip(graph, edges, strict=True)
            if vertex in edge
        )
        result.append((-9 * incident) % PRIME)
    return tuple(result)


def compile_program(n: int, family: str, defect_count: int = 0) -> Program:
    if n not in STRUCTURAL_WIDTHS:
        fail("width is outside the declared M123 grid scope")
    if family not in {"PRIMARY", "REUSE"}:
        fail("unknown M123 program family")
    descriptor_family = "PRIMARY" if family == "PRIMARY" else "GENERIC"
    edges = m122.compile_descriptor(n, descriptor_family).edges
    unary = list(zero_field_unary(n, edges))
    if n == 4:
        sites, deltas = DEFECT_SITES_N4, DEFECT_DELTAS_N4
    elif n == 6:
        sites, deltas = DEFECT_SITES_N6, DEFECT_DELTAS_N6
    else:
        sites, deltas = (), ()
    if not 0 <= defect_count <= len(sites):
        fail("defect count is outside the declared fixture")
    for site, delta in zip(sites[:defect_count], deltas[:defect_count], strict=True):
        unary[site] = (unary[site] + delta) % PRIME
    return Program(n, family, tuple(unary), edges, tuple(sites[:defect_count]))


def validate_program(program: Program) -> None:
    if program.n not in STRUCTURAL_WIDTHS:
        fail("program width changed")
    if len(program.unary) != program.n * program.n:
        fail("program unary arity changed")
    if len(program.edges) != len(m122.topology(program.n)):
        fail("program edge arity changed")
    if not all(0 <= item < PRIME for item in (*program.unary, *program.edges)):
        fail("program residue is outside F17")
    if any(item < 0 or item >= program.n * program.n for item in program.declared_defect_sites):
        fail("program defect site is outside the grid")


def field_residues(program: Program) -> tuple[int, ...]:
    graph = m122.topology(program.n)
    return tuple(
        (
            2 * program.unary[vertex]
            + sum(
                weight
                for edge, weight in zip(graph, program.edges, strict=True)
                if vertex in edge
            )
        )
        % PRIME
        for vertex in range(program.n * program.n)
    )


@dataclass(frozen=True)
class SpinParameters:
    constant_exponent: int
    edge_c: tuple[Any, ...]
    edge_d: tuple[Any, ...]
    defect_sites: tuple[int, ...]
    field_alpha: tuple[Any, ...]
    field_beta: tuple[Any, ...]
    field_residue_vector: tuple[int, ...]


def phase_pair(alg: m122.Algebra, exponent: int) -> tuple[Any, Any]:
    forward = alg.power(exponent)
    backward = alg.power(-exponent)
    half = rational_half(alg)
    return (
        alg.mul(half, alg.add(forward, backward)),
        alg.mul(half, alg.sub(forward, backward)),
    )


def spin_parameters(program: Program, alg: m122.Algebra) -> SpinParameters:
    validate_program(program)
    graph = m122.topology(program.n)
    residues = field_residues(program)
    defects = tuple(index for index, residue in enumerate(residues) if residue)
    if defects != program.declared_defect_sites:
        fail("declared defect ownership disagrees with the public zero-field gate")
    constant = (
        9 * sum(program.unary) + 13 * sum(program.edges)
    ) % PRIME
    edge_pairs = tuple(phase_pair(alg, 13 * weight) for weight in program.edges)
    # h_v = -r_v/4 in F17.
    field_pairs = tuple(phase_pair(alg, -13 * residues[site]) for site in defects)
    return SpinParameters(
        constant,
        tuple(item[0] for item in edge_pairs),
        tuple(item[1] for item in edge_pairs),
        defects,
        tuple(item[0] for item in field_pairs),
        tuple(item[1] for item in field_pairs),
        residues,
    )


@dataclass(frozen=True)
class TrivalentEdge:
    left: int
    right: int
    original_edge: int | None


@dataclass(frozen=True)
class TerminalEdge:
    left: int
    right: int
    kind: str
    trivalent_edge: int | None
    left_incident_edge: int | None = None
    right_incident_edge: int | None = None
    city_vertex: int | None = None


@dataclass(frozen=True)
class TerminalPlan:
    n: int
    trivalent_vertices: int
    trivalent_edges: tuple[TrivalentEdge, ...]
    terminal_vertices: int
    terminal_edges: tuple[TerminalEdge, ...]
    orientation_signs: tuple[int, ...]
    constrained_faces: tuple[tuple[int, ...], ...]
    omitted_face: tuple[int, ...]
    standard_matching_sign: int
    original_edge_to_trivalent: tuple[int, ...]
    topology_fingerprint: str


def _clockwise_incident_order(n: int, vertex: int, graph: tuple[tuple[int, int], ...]) -> list[int]:
    row, column = divmod(vertex, n)
    direction_order = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    incident = []
    for edge_index, edge in enumerate(graph):
        if vertex not in edge:
            continue
        other = edge[0] if edge[1] == vertex else edge[1]
        other_row, other_column = divmod(other, n)
        direction = (other_row - row, other_column - column)
        incident.append((direction_order[direction], edge_index))
    return [edge_index for _, edge_index in sorted(incident)]


def _solve_gf2(rows: list[tuple[int, int]], variable_count: int) -> int:
    pivots: dict[int, tuple[int, int]] = {}
    for mask, rhs in rows:
        while mask:
            pivot = mask.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = (mask, rhs)
                break
            prior_mask, prior_rhs = pivots[pivot]
            mask ^= prior_mask
            rhs ^= prior_rhs
        if not mask and rhs:
            fail("public terminal-face orientation equations are inconsistent")
    solution = 0
    for pivot in sorted(pivots):
        mask, rhs = pivots[pivot]
        lower = mask & ((1 << pivot) - 1)
        value = rhs ^ ((lower & solution).bit_count() & 1)
        if value:
            solution |= 1 << pivot
    if solution >> variable_count:
        fail("orientation solver escaped its public variable set")
    return solution


def compile_terminal_plan(n: int) -> TerminalPlan:
    graph = m122.topology(n)
    incidence_node: dict[tuple[int, int], int] = {}
    next_vertex = 0
    decorations: list[TrivalentEdge] = []
    for vertex in range(n * n):
        ordered = _clockwise_incident_order(n, vertex, graph)
        if len(ordered) <= 3:
            node = next_vertex
            next_vertex += 1
            for edge_index in ordered:
                incidence_node[(vertex, edge_index)] = node
        else:
            city_nodes = list(range(next_vertex, next_vertex + len(ordered)))
            next_vertex += len(ordered)
            for edge_index, node in zip(ordered, city_nodes, strict=True):
                incidence_node[(vertex, edge_index)] = node
            decorations.extend(
                TrivalentEdge(left, right, None)
                for left, right in zip(city_nodes, city_nodes[1:])
            )
    transformed: list[TrivalentEdge] = []
    original_map = []
    for edge_index, (left, right) in enumerate(graph):
        transformed_left = incidence_node[(left, edge_index)]
        transformed_right = incidence_node[(right, edge_index)]
        transformed.append(TrivalentEdge(transformed_left, transformed_right, edge_index))
        original_map.append(len(transformed) - 1)
    transformed.extend(decorations)

    terminal: list[TerminalEdge] = []
    half_at: dict[tuple[int, int], int] = {}
    cities: dict[int, list[tuple[int, int]]] = {vertex: [] for vertex in range(next_vertex)}
    for edge_index, edge in enumerate(transformed):
        left_half = 2 * edge_index
        right_half = left_half + 1
        half_at[(edge_index, edge.left)] = left_half
        half_at[(edge_index, edge.right)] = right_half
        cities[edge.left].append((edge_index, left_half))
        cities[edge.right].append((edge_index, right_half))
        terminal.append(TerminalEdge(left_half, right_half, "LONG", edge_index))
    for city_vertex in sorted(cities):
        members = sorted(cities[city_vertex])
        if len(members) > 3:
            fail("trivalent decoration left a nonplanar terminal city")
        for (edge_a, half_a), (edge_b, half_b) in itertools.combinations(members, 2):
            left, right = sorted((half_a, half_b))
            terminal.append(
                TerminalEdge(left, right, "SHORT", None, edge_a, edge_b, city_vertex)
            )

    graph_terminal = nx.Graph()
    graph_terminal.add_nodes_from(range(2 * len(transformed)))
    graph_terminal.add_edges_from((edge.left, edge.right) for edge in terminal)
    planar, embedding = nx.check_planarity(graph_terminal, counterexample=False)
    if not planar:
        fail("trivalent terminal graph failed its public planarity gate")
    seen_half_edges: set[tuple[int, int]] = set()
    faces: list[tuple[int, ...]] = []
    for left in sorted(embedding):
        for right in embedding.neighbors_cw_order(left):
            if (left, right) in seen_half_edges:
                continue
            faces.append(tuple(embedding.traverse_face(left, right, seen_half_edges)))
    if len(faces) < 2:
        fail("terminal embedding did not expose bounded faces")
    omitted_index = max(range(len(faces)), key=lambda index: (len(faces[index]), faces[index]))
    omitted = faces[omitted_index]
    constrained = tuple(face for index, face in enumerate(faces) if index != omitted_index)
    edge_lookup = {
        tuple(sorted((edge.left, edge.right))): index
        for index, edge in enumerate(terminal)
    }
    if len(edge_lookup) != len(terminal):
        fail("terminal graph unexpectedly contains parallel edges")
    equations: list[tuple[int, int]] = []
    for face in constrained:
        mask = 0
        aligned_reference = 0
        for left, right in zip(face, (*face[1:], face[0]), strict=True):
            edge_index = edge_lookup[tuple(sorted((left, right)))]
            mask ^= 1 << edge_index
            aligned_reference ^= int(left < right)
        equations.append((mask, 1 ^ aligned_reference))
    toggles = _solve_gf2(equations, len(terminal))
    signs = tuple(
        1 if ((edge.left < edge.right) ^ bool((toggles >> index) & 1)) else -1
        for index, edge in enumerate(terminal)
    )
    for face in constrained:
        clockwise = 0
        for left, right in zip(face, (*face[1:], face[0]), strict=True):
            edge_index = edge_lookup[tuple(sorted((left, right)))]
            edge = terminal[edge_index]
            oriented_left_to_right = signs[edge_index] == 1
            clockwise ^= int(
                (left == edge.left and oriented_left_to_right)
                or (left == edge.right and not oriented_left_to_right)
            )
        if clockwise != 1:
            fail("compiled terminal orientation violates a face-parity equation")
    standard_sign = product_int(signs[index] for index in range(len(transformed)))
    payload = {
        "n": n,
        "trivalent_vertices": next_vertex,
        "trivalent_edges": [vars(edge) for edge in transformed],
        "terminal_edges": [vars(edge) for edge in terminal],
        "orientation_signs": signs,
        "constrained_faces": constrained,
        "omitted_face": omitted,
        "standard_matching_sign": standard_sign,
    }
    return TerminalPlan(
        n,
        next_vertex,
        tuple(transformed),
        2 * len(transformed),
        tuple(terminal),
        signs,
        constrained,
        omitted,
        standard_sign,
        tuple(original_map),
        sha256_json(payload),
    )


def product_int(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def tree_path_edges(n: int, left: int, right: int, *, alternate: bool = False) -> set[int]:
    graph = m122.topology(n)
    edge_lookup = {tuple(sorted(edge)): index for index, edge in enumerate(graph)}
    left_row, left_column = divmod(left, n)
    right_row, right_column = divmod(right, n)
    vertices = [left]
    row, column = left_row, left_column
    if alternate:
        while row != right_row:
            row += 1 if right_row > row else -1
            vertices.append(row * n + column)
        while column != right_column:
            column += 1 if right_column > column else -1
            vertices.append(row * n + column)
    else:
        while column != right_column:
            column += 1 if right_column > column else -1
            vertices.append(row * n + column)
        while row != right_row:
            row += 1 if right_row > row else -1
            vertices.append(row * n + column)
    return {
        edge_lookup[tuple(sorted((a, b)))]
        for a, b in zip(vertices, vertices[1:])
    }


def defect_chain(n: int, subset: tuple[int, ...], *, alternate: bool = False) -> frozenset[int]:
    if len(subset) % 2:
        fail("odd defect insertion cannot enter the even free-fermion closure")
    chain: set[int] = set()
    for left, right in zip(subset[::2], subset[1::2], strict=True):
        chain.symmetric_difference_update(tree_path_edges(n, left, right, alternate=alternate))
    boundary = [0] * (n * n)
    graph = m122.topology(n)
    for edge_index in chain:
        left, right = graph[edge_index]
        boundary[left] ^= 1
        boundary[right] ^= 1
    if tuple(index for index, odd in enumerate(boundary) if odd) != tuple(sorted(subset)):
        fail("public defect chain has the wrong boundary")
    return frozenset(chain)


def even_subsets(items: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    for size in range(0, len(items) + 1, 2):
        yield from itertools.combinations(items, size)


@dataclass
class PfaffianStats:
    pivots: int = 0
    swaps: int = 0
    zero_tests: int = 0
    additions: int = 0
    subtractions: int = 0
    multiplications: int = 0
    divisions: int = 0
    maximum_dense_work_field_cells: int = 0
    maximum_observed_field_element_payload_bits: int = 0
    maximum_numerator_signed_bits: int = 0
    maximum_denominator_bits: int = 0
    maximum_named_dense_work_payload_bits_upper_bound_from_observed_field_value: int = 0

    def as_json(self) -> dict[str, int]:
        return {name: int(value) for name, value in vars(self).items()}


def observe_pfaffian_value(stats: PfaffianStats, alg: m122.Algebra, value: Any) -> None:
    if alg.modulus:
        payload = max(1, alg.modulus.bit_length())
        numerator_bits, denominator_bits = payload, 1
    else:
        payload = alg.payload_bits(value)
        numerator_bits, denominator_bits = alg.coefficient_height(value)
    stats.maximum_observed_field_element_payload_bits = max(
        stats.maximum_observed_field_element_payload_bits, payload
    )
    stats.maximum_numerator_signed_bits = max(
        stats.maximum_numerator_signed_bits, numerator_bits
    )
    stats.maximum_denominator_bits = max(
        stats.maximum_denominator_bits, denominator_bits
    )


def pfaffian(matrix: list[list[Any]], alg: m122.Algebra, stats: PfaffianStats) -> Any:
    dimension = len(matrix)
    if dimension % 2 or any(len(row) != dimension for row in matrix):
        fail("Pfaffian requires an even square matrix")
    values = [list(row) for row in matrix]
    # The caller's dense input and this elimination copy coexist.  Count both
    # matrices plus ten simultaneously live named scalar work slots; Python container/native
    # allocator overhead remains excluded.
    stats.maximum_dense_work_field_cells = max(
        stats.maximum_dense_work_field_cells, 2 * dimension * dimension + 10
    )
    if alg.modulus:
        observe_pfaffian_value(stats, alg, alg.zero)
    else:
        for row in values:
            for value in row:
                observe_pfaffian_value(stats, alg, value)
    result = alg.one
    for start in range(0, dimension, 2):
        pivot_column = None
        for column in range(start + 1, dimension):
            stats.zero_tests += 1
            if values[start][column] != alg.zero:
                pivot_column = column
                break
        if pivot_column is None:
            return alg.zero
        if pivot_column != start + 1:
            for row in range(dimension):
                values[row][start + 1], values[row][pivot_column] = (
                    values[row][pivot_column], values[row][start + 1]
                )
            values[start + 1], values[pivot_column] = values[pivot_column], values[start + 1]
            result = neg(alg, result)
            stats.swaps += 1
        pivot = values[start][start + 1]
        result = alg.mul(result, pivot)
        stats.multiplications += 1
        inverse = alg.inverse(pivot)
        stats.divisions += 1
        if not alg.modulus:
            observe_pfaffian_value(stats, alg, pivot)
            observe_pfaffian_value(stats, alg, inverse)
            observe_pfaffian_value(stats, alg, result)
        for left in range(start + 2, dimension):
            u_left = values[start][left]
            v_left = values[start + 1][left]
            for right in range(left + 1, dimension):
                # Schur complement preserving Pf(A)=pivot*Pf(remainder).
                left_product = alg.mul(v_left, values[start][right])
                right_product = alg.mul(u_left, values[start + 1][right])
                difference = alg.sub(left_product, right_product)
                correction = alg.mul(difference, inverse)
                updated = alg.add(values[left][right], correction)
                values[left][right] = updated
                values[right][left] = neg(alg, updated)
                if not alg.modulus:
                    observe_pfaffian_value(stats, alg, left_product)
                    observe_pfaffian_value(stats, alg, right_product)
                    observe_pfaffian_value(stats, alg, difference)
                    observe_pfaffian_value(stats, alg, correction)
                    observe_pfaffian_value(stats, alg, updated)
                stats.multiplications += 3
                stats.subtractions += 1
                stats.additions += 1
        stats.pivots += 1
    stats.maximum_named_dense_work_payload_bits_upper_bound_from_observed_field_value = (
        stats.maximum_dense_work_field_cells
        * stats.maximum_observed_field_element_payload_bits
    )
    return result


def matrix_determinant(matrix: list[list[Any]], alg: m122.Algebra) -> Any:
    values = [list(row) for row in matrix]
    result = alg.one
    for column in range(len(values)):
        pivot = next(
            (row for row in range(column, len(values)) if values[row][column] != alg.zero),
            None,
        )
        if pivot is None:
            return alg.zero
        if pivot != column:
            values[pivot], values[column] = values[column], values[pivot]
            result = neg(alg, result)
        value = values[column][column]
        result = alg.mul(result, value)
        inverse = alg.inverse(value)
        for row in range(column + 1, len(values)):
            if values[row][column] == alg.zero:
                continue
            factor = alg.mul(values[row][column], inverse)
            for target in range(column + 1, len(values)):
                values[row][target] = alg.sub(
                    values[row][target], alg.mul(factor, values[column][target])
                )
    return result


def sector_terminal_values(
    plan: TerminalPlan,
    parameters: SpinParameters,
    path_chain: frozenset[int],
    alg: m122.Algebra,
) -> list[Any]:
    c_values = list(parameters.edge_c)
    d_values = list(parameters.edge_d)
    for edge_index in path_chain:
        c_values[edge_index], d_values[edge_index] = d_values[edge_index], c_values[edge_index]
    transformed_c: list[Any] = []
    transformed_d: list[Any] = []
    for edge in plan.trivalent_edges:
        if edge.original_edge is None:
            transformed_c.append(alg.one)
            transformed_d.append(alg.one)
        else:
            transformed_c.append(c_values[edge.original_edge])
            transformed_d.append(d_values[edge.original_edge])
    result = []
    for edge, sign in zip(plan.terminal_edges, plan.orientation_signs, strict=True):
        if edge.kind == "LONG":
            assert edge.trivalent_edge is not None
            value = transformed_c[edge.trivalent_edge]
        else:
            assert edge.left_incident_edge is not None and edge.right_incident_edge is not None
            incident_values = []
            for incident_edge in (edge.left_incident_edge, edge.right_incident_edge):
                transformed_edge = plan.trivalent_edges[incident_edge]
                is_tail = edge.city_vertex == transformed_edge.left
                incident_values.append(transformed_d[incident_edge] if is_tail else alg.one)
            value = alg.mul(incident_values[0], incident_values[1])
        result.append(value if sign == 1 else neg(alg, value))
    return result


def dense_matrix(plan: TerminalPlan, values: list[Any], alg: m122.Algebra) -> list[list[Any]]:
    if len(values) != len(plan.terminal_edges):
        fail("terminal carrier arity changed")
    matrix = [[alg.zero for _ in range(plan.terminal_vertices)] for _ in range(plan.terminal_vertices)]
    for edge, value in zip(plan.terminal_edges, values, strict=True):
        matrix[edge.left][edge.right] = value
        matrix[edge.right][edge.left] = neg(alg, value)
    return matrix


@dataclass
class CovarianceCarrier:
    plan: TerminalPlan
    alg: m122.Algebra
    cells: list[Any]
    accumulator: Any
    generation: int = 0
    lease: str | None = None
    stage: str = "RESTORED"
    loads: int = 0
    unloads: int = 0
    accumulator_updates: int = 0
    projection_calls: int = 0
    maximum_resident_phase_payload_bits: int = 0
    pfaffian_stats: PfaffianStats = field(default_factory=PfaffianStats)

    @classmethod
    def create(cls, n: int, alg: m122.Algebra) -> "CovarianceCarrier":
        plan = compile_terminal_plan(n)
        return cls(plan, alg, [alg.zero for _ in plan.terminal_edges], alg.zero)

    def backing_identity(self) -> tuple[int, int]:
        return id(self.cells), id(self)

    def exact_zero(self) -> bool:
        return (
            all(value == self.alg.zero for value in self.cells)
            and self.accumulator == self.alg.zero
            and self.lease is None
            and self.stage == "RESTORED"
        )

    def digest(self) -> str:
        return sha256_json(
            {
                "topology": self.plan.topology_fingerprint,
                "cells": [self.alg.serialize(value) for value in self.cells],
                "accumulator": self.alg.serialize(self.accumulator),
                "generation": self.generation,
                "lease": self.lease,
                "stage": self.stage,
            }
        )

    def observe_resident_payload(self) -> None:
        self.maximum_resident_phase_payload_bits = max(
            self.maximum_resident_phase_payload_bits,
            sum(self.alg.payload_bits(value) for value in self.cells)
            + self.alg.payload_bits(self.accumulator),
        )


def _load_values(carrier: CovarianceCarrier, values: list[Any], inverse: bool = False) -> None:
    for index, value in enumerate(values):
        delta = neg(carrier.alg, value) if inverse else value
        carrier.cells[index] = carrier.alg.add(carrier.cells[index], delta)
    if inverse:
        carrier.unloads += len(values)
    else:
        carrier.loads += len(values)
    carrier.observe_resident_payload()


def _sector_value(
    carrier: CovarianceCarrier,
    parameters: SpinParameters,
    subset: tuple[int, ...],
    *,
    alternate_path: bool = False,
) -> Any:
    chain = defect_chain(carrier.plan.n, subset, alternate=alternate_path)
    values = sector_terminal_values(carrier.plan, parameters, chain, carrier.alg)
    _load_values(carrier, values)
    matrix = dense_matrix(carrier.plan, carrier.cells, carrier.alg)
    value = pfaffian(matrix, carrier.alg, carrier.pfaffian_stats)
    if carrier.plan.standard_matching_sign == -1:
        value = neg(carrier.alg, value)
    _load_values(carrier, values, inverse=True)
    if any(item != carrier.alg.zero for item in carrier.cells):
        fail("sector closure did not restore terminal cells")
    return value


def _field_coefficient(
    parameters: SpinParameters,
    subset: tuple[int, ...],
    alg: m122.Algebra,
) -> Any:
    chosen = set(subset)
    return product(
        alg,
        (
            parameters.field_beta[index]
            if site in chosen
            else parameters.field_alpha[index]
            for index, site in enumerate(parameters.defect_sites)
        ),
    )


def forward(carrier: CovarianceCarrier, program: Program, *, alternate_path: bool = False) -> dict[str, Any]:
    if not isinstance(carrier, CovarianceCarrier) or not carrier.exact_zero():
        fail("null, leased, or unrestored covariance carrier")
    if carrier.plan.n != program.n:
        fail("program width does not own the covariance carrier")
    parameters = spin_parameters(program, carrier.alg)
    carrier.lease = program.fingerprint()
    carrier.stage = "FORWARD_ACTIVE"
    expected = 1 if len(parameters.defect_sites) <= 1 else 1 << (len(parameters.defect_sites) - 1)
    sector_count = 0
    for subset in even_subsets(parameters.defect_sites):
        value = _sector_value(
            carrier, parameters, subset, alternate_path=alternate_path
        )
        coefficient = _field_coefficient(parameters, subset, carrier.alg)
        contribution = carrier.alg.mul(coefficient, value)
        carrier.accumulator = carrier.alg.add(carrier.accumulator, contribution)
        carrier.accumulator_updates += 1
        carrier.observe_resident_payload()
        sector_count += 1
    if sector_count != expected:
        fail("even defect-sector law changed")
    carrier.stage = "FORWARD_COMPLETE"
    return {
        "defect_count": len(parameters.defect_sites),
        "even_sector_count": sector_count,
        "constant_exponent": parameters.constant_exponent,
        "field_residue_vector": parameters.field_residue_vector,
    }


def project_boundary(carrier: CovarianceCarrier, program: Program) -> Any:
    if carrier.stage != "FORWARD_COMPLETE" or carrier.lease != program.fingerprint():
        fail("only the completed aggregate boundary may be projected")
    if any(value != carrier.alg.zero for value in carrier.cells):
        fail("terminal work cells remained live at final projection")
    carrier.projection_calls += 1
    phase = carrier.alg.power(spin_parameters(program, carrier.alg).constant_exponent)
    spin_count = scalar(carrier.alg, 1 << (program.n * program.n))
    return carrier.alg.mul(phase, carrier.alg.mul(spin_count, carrier.accumulator))


def inverse(carrier: CovarianceCarrier, program: Program, *, alternate_path: bool = False) -> None:
    if carrier.stage != "FORWARD_COMPLETE" or carrier.lease != program.fingerprint():
        fail("inverse program does not own the live covariance lease")
    parameters = spin_parameters(program, carrier.alg)
    carrier.stage = "INVERSE_ACTIVE"
    # These accumulator and entry additions commute, so rematerializing the
    # generator in its public forward order is the exact inverse without a
    # retained subset list or inverse history.
    for subset in even_subsets(parameters.defect_sites):
        value = _sector_value(
            carrier, parameters, subset, alternate_path=alternate_path
        )
        coefficient = _field_coefficient(parameters, subset, carrier.alg)
        contribution = carrier.alg.mul(coefficient, value)
        carrier.accumulator = carrier.alg.sub(carrier.accumulator, contribution)
        carrier.accumulator_updates += 1
        carrier.observe_resident_payload()
    carrier.lease = None
    carrier.stage = "RESTORED"
    carrier.generation += 1
    if not carrier.exact_zero():
        fail("actual inverse failed exact covariance restoration")


def execute_transaction(
    carrier: CovarianceCarrier,
    program: Program,
    *,
    alternate_path: bool = False,
) -> dict[str, Any]:
    backing = carrier.backing_identity()
    generation = carrier.generation
    initial = carrier.digest()
    metadata = forward(carrier, program, alternate_path=alternate_path)
    boundary = project_boundary(carrier, program)
    inverse(carrier, program, alternate_path=alternate_path)
    restored_digest = carrier.digest()
    # Generation is a monotone custody counter and is intentionally excluded
    # from the phase-cell equality below.
    return {
        "program_fingerprint": program.fingerprint(),
        "boundary": carrier.alg.serialize(boundary),
        **metadata,
        "terminal_plan_fingerprint": carrier.plan.topology_fingerprint,
        "trivalent_vertex_count": carrier.plan.trivalent_vertices,
        "trivalent_edge_count": len(carrier.plan.trivalent_edges),
        "terminal_vertex_count": carrier.plan.terminal_vertices,
        "sparse_resident_terminal_field_cells": len(carrier.cells),
        "dense_projection_work_field_cells": 2 * carrier.plan.terminal_vertices ** 2 + 10,
        "maximum_resident_phase_payload_bits": carrier.maximum_resident_phase_payload_bits,
        "orientation_constrained_face_count": len(carrier.plan.constrained_faces),
        "orientation_face_parity_verified": True,
        "intermediate_boundary_projection_calls": 0,
        "final_boundary_projection_calls": 1,
        "internal_sector_pfaffian_evaluations": 2 * metadata["even_sector_count"],
        "generation_before": generation,
        "generation_after": carrier.generation,
        "restoration_generation_increment": carrier.generation == generation + 1,
        "same_backing": carrier.backing_identity() == backing,
        "exact_phase_cells_restored": carrier.exact_zero(),
        "initial_digest": initial,
        "restored_digest_with_generation": restored_digest,
        "snapshot_reload_used": False,
        "inverse_history_retained": False,
        "carrier_restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "projection_buffer_restoration_class": "NO_RESTORATION_CLAIM",
        "pfaffian_stats": carrier.pfaffian_stats.as_json(),
        "terminal_entry_load_additions": carrier.loads,
        "terminal_entry_unload_additions": carrier.unloads,
        "scalar_accumulator_forward_and_inverse_updates": carrier.accumulator_updates,
        "boundary_payload_bits": carrier.alg.payload_bits(boundary),
    }


def identical_classical_boundary(program: Program, alg: m122.Algebra) -> tuple[Any, dict[str, Any]]:
    carrier = CovarianceCarrier.create(program.n, alg)
    parameters = spin_parameters(program, alg)
    accumulator = alg.zero
    stats = PfaffianStats()
    for subset in even_subsets(parameters.defect_sites):
        chain = defect_chain(program.n, subset)
        values = sector_terminal_values(carrier.plan, parameters, chain, alg)
        matrix = dense_matrix(carrier.plan, values, alg)
        value = pfaffian(matrix, alg, stats)
        if carrier.plan.standard_matching_sign == -1:
            value = neg(alg, value)
        accumulator = alg.add(
            accumulator,
            alg.mul(_field_coefficient(parameters, subset, alg), value),
        )
    boundary = alg.mul(
        alg.power(parameters.constant_exponent),
        alg.mul(scalar(alg, 1 << (program.n * program.n)), accumulator),
    )
    return boundary, {
        "algorithm": "IDENTICAL_SQUARE_ROOT_FREE_TERMINAL_PFAFFIAN",
        "pfaffian_stats": stats.as_json(),
        "restoration_class": "NO_RESTORATION_CLAIM",
    }


def exact_case(n: int, family: str, defect_count: int) -> dict[str, Any]:
    program = compile_program(n, family, defect_count)
    alg = m122.Algebra("Q_ZETA17")
    carrier = CovarianceCarrier.create(n, alg)
    transaction = execute_transaction(carrier, program)
    baseline, baseline_meta = identical_classical_boundary(program, m122.Algebra("Q_ZETA17"))
    if transaction["boundary"] != alg.serialize(baseline):
        fail("phase carrier disagrees with the identical classical Pfaffian")
    transaction["n"] = n
    transaction["family"] = family
    transaction["matched_classical_boundary_agreement"] = True
    transaction["matched_classical"] = baseline_meta
    return transaction


def modular_case(n: int, modulus: int, root: int, defect_count: int = 0) -> dict[str, Any]:
    program = compile_program(n, "PRIMARY", defect_count)
    alg = m122.Algebra(f"F{modulus}", modulus=modulus, root=root)
    carrier = CovarianceCarrier.create(n, alg)
    result = execute_transaction(carrier, program)
    result.update({"n": n, "field": f"F{modulus}", "family": "PRIMARY"})
    return result


def controls() -> dict[str, Any]:
    program = compile_program(3, "PRIMARY", 0)
    alg = m122.Algebra("F103", modulus=103, root=72)
    carrier = CovarianceCarrier.create(3, alg)
    parameters = spin_parameters(program, alg)
    values = sector_terminal_values(carrier.plan, parameters, frozenset(), alg)
    matrix = dense_matrix(carrier.plan, values, alg)
    pf_stats = PfaffianStats()
    pf = pfaffian(matrix, alg, pf_stats)
    determinant = matrix_determinant(matrix, alg)
    pfaffian_square_agreement = alg.mul(pf, pf) == determinant

    missing = CovarianceCarrier.create(3, m122.Algebra("F103", modulus=103, root=72))
    forward(missing, program)
    missing_inverse_detected = not missing.exact_zero()

    wrong = CovarianceCarrier.create(4, m122.Algebra("F103", modulus=103, root=72))
    live_program = compile_program(4, "PRIMARY", 2)
    forward(wrong, live_program)
    wrong_inverse_detected = False
    try:
        inverse(wrong, compile_program(4, "REUSE", 2))
    except RuntimeError:
        wrong_inverse_detected = True

    premature = CovarianceCarrier.create(3, m122.Algebra("F103", modulus=103, root=72))
    premature_projection_rejected = False
    try:
        project_boundary(premature, program)
    except RuntimeError:
        premature_projection_rejected = True

    null_carrier_rejected = False
    try:
        forward(None, program)  # type: ignore[arg-type]
    except (RuntimeError, AttributeError):
        null_carrier_rejected = True

    odd_sector_rejected = False
    try:
        defect_chain(4, (5,))
    except RuntimeError:
        odd_sector_rejected = True

    path_program = compile_program(4, "PRIMARY", 4)
    path_alg_a = m122.Algebra("F103", modulus=103, root=72)
    path_alg_b = m122.Algebra("F103", modulus=103, root=72)
    path_a = execute_transaction(CovarianceCarrier.create(4, path_alg_a), path_program)
    path_b = execute_transaction(
        CovarianceCarrier.create(4, path_alg_b), path_program, alternate_path=True
    )

    zero_gate_mutation = Program(
        program.n,
        "MUTATED",
        tuple((value + 1) % PRIME if index == 4 else value for index, value in enumerate(program.unary)),
        program.edges,
        (),
    )
    undeclared_field_rejected = False
    try:
        spin_parameters(zero_gate_mutation, alg)
    except RuntimeError:
        undeclared_field_rejected = True

    flipped_values = list(values)
    flipped_values[0] = neg(alg, flipped_values[0])
    flipped_pf = pfaffian(dense_matrix(carrier.plan, flipped_values, alg), alg, PfaffianStats())

    return {
        "pfaffian_square_equals_determinant_f103_n3": pfaffian_square_agreement,
        "missing_inverse_detected": missing_inverse_detected,
        "wrong_inverse_ownership_detected": wrong_inverse_detected,
        "premature_projection_rejected": premature_projection_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "odd_defect_sector_rejected": odd_sector_rejected,
        "alternate_path_boundary_agreement_f103_n4_k4": path_a["boundary"] == path_b["boundary"],
        "undeclared_external_field_rejected_by_zero_field_gate": undeclared_field_rejected,
        "single_orientation_edge_flip_changes_pfaffian": flipped_pf != pf,
        "reordered_inverse_control_applicable": False,
        "reordered_inverse_reason": "TERMINAL_ENTRY_ADDITIONS_AND_SCALAR_SECTOR_ADDITIONS_COMMUTE",
        "snapshot_command_absent": True,
        "controller_backend_boundary_claimed": False,
        "intermediate_boundary_projection_api_absent": True,
    }


def run() -> dict[str, Any]:
    exact = [
        exact_case(2, "PRIMARY", 0),
        exact_case(3, "PRIMARY", 0),
        exact_case(4, "PRIMARY", 0),
        exact_case(4, "PRIMARY", 1),
        exact_case(4, "PRIMARY", 2),
        exact_case(4, "PRIMARY", 4),
    ]
    reuse_alg = m122.Algebra("Q_ZETA17")
    reuse_carrier = CovarianceCarrier.create(4, reuse_alg)
    first = execute_transaction(reuse_carrier, compile_program(4, "PRIMARY", 0))
    reused = execute_transaction(reuse_carrier, compile_program(4, "REUSE", 2))
    fresh = execute_transaction(
        CovarianceCarrier.create(4, m122.Algebra("Q_ZETA17")),
        compile_program(4, "REUSE", 2),
    )
    if reused["boundary"] != fresh["boundary"]:
        fail("restored covariance carrier reuse disagrees with fresh execution")

    structural = [
        modular_case(n, modulus, root)
        for modulus, root in MODULAR_FIELDS
        for n in STRUCTURAL_WIDTHS
    ]
    defect_growth = [
        {
            "n": 6,
            "defect_count": count,
            "even_sector_count": 1 if count <= 1 else 1 << (count - 1),
            "executed": False,
            "status": "PUBLIC_EXACT_SECTOR_LAW_ONLY_NOT_A_LOWER_BOUND",
        }
        for count in range(9)
    ]
    return {
        "schema": "CAT_CAS_F17_PLANAR_FREE_FERMION_PHASE_COVARIANCE_CLOSURE_V1",
        "claim": "BOUNDED_EXACT_PLANAR_FREE_FERMION_TERMINAL_PFAFFIAN_PHASE_CLOSURE_FOR_DECLARED_ZERO_FIELD_F17_GRIDS_WITH_EXACT_SPARSE_DEFECT_SECTOR_GROWTH_RESTORATION_AND_REUSE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_scope": {
            "exact_q_zeta17_widths": EXACT_WIDTHS,
            "exact_defect_execution": {"n": 4, "k": [0, 1, 2, 4]},
            "dual_field_structural_widths": STRUCTURAL_WIDTHS,
            "defect_growth_law_only": {"n": 6, "k": list(range(9))},
            "topology": "OPEN_PLANAR_SQUARE_GRIDS_ONLY",
        },
        "exact_transactions": exact,
        "dual_field_structural_transactions": structural,
        "defect_growth": defect_growth,
        "reuse": {
            "first_boundary": first["boundary"],
            "reused_boundary": reused["boundary"],
            "fresh_boundary": fresh["boundary"],
            "fresh_restored_boundary_agreement": reused["boundary"] == fresh["boundary"],
            "same_actual_backing_across_unrelated_programs": first["same_backing"] and reused["same_backing"],
            "generation_after_two_transactions": reuse_carrier.generation,
            "baseline_reload_used": False,
        },
        "controls": controls(),
        "resource_accounting": {
            "public_topology_orientation_compilation_counted": True,
            "sparse_terminal_phase_cells_counted": True,
            "dense_pfaffian_projection_buffers_counted": True,
            "exact_phase_coefficient_height_observed": True,
            "dense_named_logical_payload_is_an_upper_bound_from_every_observed_field_temporary": True,
            "python_container_sympy_and_native_internal_payload_excluded": True,
            "pfaffian_pivots_and_exact_divisions_counted": True,
            "forward_and_inverse_rematerialization_counted": True,
            "defect_sector_count_counted": True,
            "controller_backend_traffic": "DIRECT_PROCESS_NO_CATVM_TRAFFIC",
            "allocator_and_native_library_overhead_excluded": True,
            "nested_dissection_not_implemented": True,
        },
        "matched_baselines": {
            "strongest_implemented": "IDENTICAL_SQUARE_ROOT_FREE_TERMINAL_PFAFFIAN",
            "strongest_known_compact_zero_field": "IDENTICAL_EVEN_GAUSSIAN_BOUNDARY_COVARIANCE_OR_SPARSE_PLANAR_PFAFFIAN",
            "general_field_fallback": "M120_ROW_TRANSFER_O_N2_2N_WORK_AND_2N_FIELD_CELLS_INDEPENDENT_OF_DEFECT_COUNT",
            "defect_expansion": "STREAMED_2_TO_THE_K_MINUS_1_PFAFFIANS_IS_AN_UPPER_BOUND_NOT_A_LOWER_BOUND",
            "phase_advantage_over_matched_classical": False,
        },
        "restoration": {
            "resident_terminal_and_scalar_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "dense_pfaffian_projection_buffers": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "direct_single_pfaffian_only_in_declared_zero_field_sector": True,
            "implemented_sparse_external_field_path_uses_even_sector_expansion": True,
            "dense_external_field_compaction_established": False,
            "arbitrary_matchgate_or_holographic_reduction_established": False,
            "phase_native_zero_tests_pivots_divisions_or_orientation_established": False,
            "catvm_custody_established": False,
            "distinct_phase_resource_established": False,
            "computational_advantage_established": False,
            "small_wall_crossing_established": False,
            "physical_waveform_execution_established": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation_established": False,
        },
        "next_obstruction": "ZERO_FIELD_FREE_FERMION_CLOSURE_IS_IDENTICAL_TO_COMPACT_CLASSICAL_GAUSSIAN_SOFTWARE_WHILE_TRUE_EXTERNAL_FIELDS SPLIT_INTO_2_TO_THE_K_MINUS_1_EVEN_GAUSSIAN_SECTORS_AND_NO_PHASE_NATIVE_OPERATION_CURRENTLY_CLOSES_THEIR_SUM_WITHOUT_MOVING_THAT_GROWTH",
        "next_experiment": "EXACT_SHARED_PHASE_PARITY_LEDGER_FOR_SPARSE_NON_GAUSSIAN_FIELD_INSERTIONS_WITHOUT_EVEN_SECTOR_ENUMERATION_OR_A_PROOF_THAT_THE_LEDGER_RANK_MUST_GROW",
    }


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
