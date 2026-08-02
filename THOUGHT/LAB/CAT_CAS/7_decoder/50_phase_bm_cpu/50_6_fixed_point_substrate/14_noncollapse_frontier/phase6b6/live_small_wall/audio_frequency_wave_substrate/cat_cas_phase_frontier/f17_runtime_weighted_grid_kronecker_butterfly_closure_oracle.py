#!/usr/bin/env python3
"""Independent exact-integer oracle for the M120 butterfly repair.

This file imports neither M120/M119 nor their cyclotomic backend.  It rebuilds
the public programs, exact Q(zeta17) power-basis arithmetic, the Kronecker
butterfly recurrence, a Gray-code global histogram, finite-field rank checks,
and factor-cell forward/inverse/unload semantics.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any


PRIME = 17
MODULUS = 103
ROOT_MOD103 = 72
SIZES = (2, 3, 4)
FAMILIES = ("PRIMARY", "REUSE")
ZERO = (0,) * 16
ONE = (1,) + (0,) * 15


def fail(message: str) -> None:
    raise RuntimeError(message)


def nonzero_weight(value: int) -> int:
    return value % 16 + 1


def vertex_index(n: int, row: int, column: int) -> int:
    return row * n + column


def topology(n: int) -> tuple[tuple[int, int], ...]:
    return (
        *(
            (vertex_index(n, row, column), vertex_index(n, row, column + 1))
            for row in range(n)
            for column in range(n - 1)
        ),
        *(
            (vertex_index(n, row, column), vertex_index(n, row + 1, column))
            for row in range(n - 1)
            for column in range(n)
        ),
    )


def plan_fingerprint(n: int) -> str:
    vertices = [[row, column] for row in range(n) for column in range(n)]
    edges = [list(edge) for edge in topology(n)]
    operations = [
        *({"kind": "PREPARE", "index": site} for site in range(n * n)),
        *({"kind": "UNARY", "index": site} for site in range(n * n)),
        *({"kind": "EDGE", "index": edge} for edge in range(len(edges))),
    ]
    record = {
        "schema": "F17_GRID_TOPOLOGY_PLAN_V1",
        "n": n,
        "vertices": vertices,
        "edges": edges,
        "operations": operations,
    }
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def runtime_weights(n: int, family: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if family not in FAMILIES:
        fail("unknown oracle family")
    offset = 1 if family == "PRIMARY" else 7
    unary = tuple(nonzero_weight(7 * site + 3 * n + offset) for site in range(n * n))
    edges = tuple(
        nonzero_weight(11 * ordinal + 5 * n + 2 * offset)
        for ordinal in range(len(topology(n)))
    )
    return unary, edges


def add(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def subtract(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def root_action(value: tuple[int, ...], exponent: int) -> tuple[int, ...]:
    characters = [0 for _ in range(PRIME)]
    for index, coefficient in enumerate(value):
        characters[(index + exponent) % PRIME] += coefficient
    omitted = characters[16]
    return tuple(characters[index] - omitted for index in range(16))


def canonical_histogram(histogram: tuple[int, ...]) -> tuple[int, ...]:
    omitted = histogram[16]
    return tuple(histogram[index] - omitted for index in range(16))


def row_bits(index: int, n: int) -> tuple[int, ...]:
    return tuple((index >> (n - 1 - column)) & 1 for column in range(n))


def energy(
    assignment: int,
    unary: tuple[int, ...],
    edges: tuple[int, ...],
    graph: tuple[tuple[int, int], ...],
) -> int:
    total = sum(weight for site, weight in enumerate(unary) if (assignment >> site) & 1)
    total += sum(
        weight
        for (left, right), weight in zip(graph, edges, strict=True)
        if ((assignment >> left) & 1) and ((assignment >> right) & 1)
    )
    return total % PRIME


@dataclass
class GrayStats:
    assignments_streamed: int = 0
    changed_bit_updates: int = 0
    incident_edge_checks: int = 0
    maximum_live_bins: int = PRIME

    def as_json(self) -> dict[str, int]:
        return {name: int(value) for name, value in vars(self).items()}


def gray_histogram(
    n: int,
    unary: tuple[int, ...],
    edge_weights: tuple[int, ...],
) -> tuple[tuple[int, ...], GrayStats]:
    graph = topology(n)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in unary]
    for (left, right), weight in zip(graph, edge_weights, strict=True):
        adjacency[left].append((right, weight))
        adjacency[right].append((left, weight))
    histogram = [0 for _ in range(PRIME)]
    histogram[0] = 1
    stats = GrayStats(assignments_streamed=1)
    assignment = 0
    exponent = 0
    for ordinal in range(1, 1 << (n * n)):
        changed = (ordinal & -ordinal).bit_length() - 1
        following = ordinal ^ (ordinal >> 1)
        turning_on = (following >> changed) & 1
        local = unary[changed]
        for neighbor, weight in adjacency[changed]:
            stats.incident_edge_checks += 1
            if (assignment >> neighbor) & 1:
                local += weight
        exponent = (exponent + local if turning_on else exponent - local) % PRIME
        assignment = following
        histogram[exponent] += 1
        stats.assignments_streamed += 1
        stats.changed_bit_updates += 1
    return tuple(histogram), stats


@dataclass
class ButterflyStats:
    butterfly_root_actions: int = 0
    butterfly_additions: int = 0
    diagonal_root_actions: int = 0
    diagonal_multiplications: int = 0
    final_additions: int = 0
    maximum_frontier_cells: int = 0
    maximum_coordinate_signed_bits: int = 0
    maximum_frontier_payload_bits: int = 0

    def observe(self, values: list[tuple[int, ...]]) -> None:
        self.maximum_frontier_cells = max(self.maximum_frontier_cells, len(values))
        self.maximum_coordinate_signed_bits = max(
            self.maximum_coordinate_signed_bits,
            *(max((abs(coordinate).bit_length() + 1) for coordinate in value) for value in values),
        )
        self.maximum_frontier_payload_bits = max(
            self.maximum_frontier_payload_bits,
            sum(sum(abs(coordinate).bit_length() + 1 for coordinate in value) for value in values),
        )

    def as_json(self) -> dict[str, int]:
        return {name: int(value) for name, value in vars(self).items()}


def row_exponent(
    n: int,
    row: int,
    assignment: int,
    unary: tuple[int, ...],
    edge_weights: tuple[int, ...],
    lookup: dict[tuple[int, int], int],
) -> int:
    values = row_bits(assignment, n)
    total = sum(
        unary[vertex_index(n, row, column)]
        for column, bit in enumerate(values)
        if bit
    )
    total += sum(
        edge_weights[
            lookup[(vertex_index(n, row, column), vertex_index(n, row, column + 1))]
        ]
        for column in range(n - 1)
        if values[column] and values[column + 1]
    )
    return total % PRIME


def butterfly_boundary(
    n: int,
    unary: tuple[int, ...],
    edge_weights: tuple[int, ...],
    *,
    column_orders: tuple[tuple[int, ...], ...] | None = None,
    skipped_stage: tuple[int, int] | None = None,
) -> tuple[tuple[int, ...], ButterflyStats]:
    graph = topology(n)
    lookup = {edge: ordinal for ordinal, edge in enumerate(graph)}
    width = 1 << n
    stats = ButterflyStats()
    current = [root_action(ONE, row_exponent(n, 0, value, unary, edge_weights, lookup)) for value in range(width)]
    stats.diagonal_root_actions += width
    stats.observe(current)
    if column_orders is None:
        column_orders = tuple(tuple(range(n)) for _ in range(n - 1))
    if len(column_orders) != n - 1:
        fail("oracle butterfly order count changed")
    for row in range(1, n):
        for column in column_orders[row - 1]:
            if skipped_stage == (row, column):
                continue
            edge = lookup[(vertex_index(n, row - 1, column), vertex_index(n, row, column))]
            weight = edge_weights[edge]
            mask = 1 << (n - 1 - column)
            following = list(current)
            for low_index in range(width):
                if low_index & mask:
                    continue
                high_index = low_index | mask
                low = current[low_index]
                high = current[high_index]
                following[low_index] = add(low, high)
                following[high_index] = add(low, root_action(high, weight))
                stats.butterfly_root_actions += 1
                stats.butterfly_additions += 2
            current = following
            stats.observe(current)
        following = []
        for assignment, value in enumerate(current):
            exponent = row_exponent(n, row, assignment, unary, edge_weights, lookup)
            following.append(root_action(value, exponent))
            stats.diagonal_root_actions += 1
            stats.diagonal_multiplications += 1
        current = following
        stats.observe(current)
    boundary = ZERO
    for value in current:
        boundary = add(boundary, value)
        stats.final_additions += 1
    return boundary, stats


def independent_factor_restoration(
    n: int,
    unary: tuple[int, ...],
    edge_weights: tuple[int, ...],
) -> dict[str, Any]:
    sites = [[ONE, ZERO] for _ in range(n * n)]
    edges = [[ONE, ONE, ONE, ONE] for _ in topology(n)]
    backing = (id(sites), id(edges), tuple(map(id, sites)), tuple(map(id, edges)))
    for site, weight in enumerate(unary):
        sites[site][1] = add(sites[site][1], sites[site][0])
        sites[site][1] = root_action(sites[site][1], weight)
    for edge, weight in enumerate(edge_weights):
        edges[edge][3] = root_action(edges[edge][3], weight)
    forward_changed = any(row != [ONE, ZERO] for row in sites) or any(row != [ONE] * 4 for row in edges)
    for edge in reversed(range(len(edge_weights))):
        edges[edge][3] = root_action(edges[edge][3], -edge_weights[edge])
    for site in reversed(range(len(unary))):
        sites[site][1] = root_action(sites[site][1], -unary[site])
        sites[site][1] = subtract(sites[site][1], sites[site][0])
    seed_restored = all(row == [ONE, ZERO] for row in sites) and all(row == [ONE] * 4 for row in edges)
    for row in sites:
        row[0] = subtract(row[0], ONE)
    for row in edges:
        for index in range(4):
            row[index] = subtract(row[index], ONE)
    zero_backing = all(cell == ZERO for row in sites for cell in row) and all(cell == ZERO for row in edges for cell in row)
    return {
        "forward_changed_seed": forward_changed,
        "seed_restored_exactly": seed_restored,
        "unload_restored_zero_backing": zero_backing,
        "same_backing": backing == (id(sites), id(edges), tuple(map(id, sites)), tuple(map(id, edges))),
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
    }


def rank_mod103(matrix: list[list[int]]) -> int:
    rows = [row[:] for row in matrix]
    rank = 0
    for column in range(len(rows[0])):
        pivot = next((index for index in range(rank, len(rows)) if rows[index][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inverse = pow(rows[rank][column], -1, MODULUS)
        rows[rank] = [(value * inverse) % MODULUS for value in rows[rank]]
        for index in range(len(rows)):
            if index == rank or not rows[index][column]:
                continue
            factor = rows[index][column]
            rows[index] = [
                (left - factor * right) % MODULUS
                for left, right in zip(rows[index], rows[rank], strict=True)
            ]
        rank += 1
    return rank


def interface_rank_certificate(n: int, edge_weights: tuple[int, ...]) -> dict[str, Any]:
    graph = topology(n)
    lookup = {edge: ordinal for ordinal, edge in enumerate(graph)}
    row = n // 2
    weights = tuple(
        edge_weights[lookup[(vertex_index(n, row - 1, column), vertex_index(n, row, column))]]
        for column in range(n)
    )
    width = 1 << n
    matrix = [
        [
            pow(
                ROOT_MOD103,
                sum(weight * a * b for weight, a, b in zip(weights, row_bits(source, n), row_bits(target, n), strict=True)),
                MODULUS,
            )
            for target in range(width)
        ]
        for source in range(width)
    ]
    zeroed = [list(row_values) for row_values in matrix]
    first_weight = weights[0]
    first_mask = 1 << (n - 1)
    for source in range(width):
        for target in range(width):
            if (source & first_mask) and (target & first_mask):
                zeroed[source][target] = (
                    zeroed[source][target] * pow(pow(ROOT_MOD103, first_weight, MODULUS), -1, MODULUS)
                ) % MODULUS
    determinant = 1
    for weight in weights:
        determinant = determinant * pow((pow(ROOT_MOD103, weight, MODULUS) - 1) % MODULUS, 1 << (n - 1), MODULUS) % MODULUS
    return {
        "separator_weights": list(weights),
        "explicit_matrix_rank_mod103": rank_mod103(matrix),
        "one_zero_weight_matrix_rank_mod103": rank_mod103(zeroed),
        "exact_rank_over_q_zeta17": width,
        "one_zero_weight_exact_rank": width // 2,
        "kronecker_determinant_mod103": determinant,
        "kronecker_determinant_nonzero": determinant != 0,
    }


def case(n: int, family: str) -> dict[str, Any]:
    unary, edges = runtime_weights(n, family)
    boundary, stats = butterfly_boundary(n, unary, edges)
    histogram, gray_stats = gray_histogram(n, unary, edges)
    reversed_orders = tuple(tuple(reversed(range(n))) for _ in range(n - 1))
    reordered, _ = butterfly_boundary(n, unary, edges, column_orders=reversed_orders)
    omitted, _ = butterfly_boundary(n, unary, edges, skipped_stage=(1, 0))
    changed_edges = list(edges)
    changed_edges[n * (n - 1)] = nonzero_weight(changed_edges[n * (n - 1)] + 1)
    if changed_edges[n * (n - 1)] == edges[n * (n - 1)]:
        changed_edges[n * (n - 1)] = nonzero_weight(changed_edges[n * (n - 1)] + 1)
    changed, _ = butterfly_boundary(n, unary, tuple(changed_edges))
    expected_roots = (n - 1) * n * (1 << (n - 1))
    return {
        "n": n,
        "family": family,
        "plan_fingerprint": plan_fingerprint(n),
        "unary_weights": list(unary),
        "edge_weights": list(edges),
        "canonical_boundary": list(boundary),
        "gray_histogram_boundary": list(canonical_histogram(histogram)),
        "butterfly_gray_boundary_agreement": boundary == canonical_histogram(histogram),
        "butterfly_stats": stats.as_json(),
        "exact_operation_counts": (
            stats.butterfly_root_actions == expected_roots
            and stats.butterfly_additions == 2 * expected_roots
            and stats.final_additions == (1 << n)
        ),
        "column_stage_reorder_agrees": reordered == boundary,
        "omitted_stage_changes_boundary": omitted != boundary,
        "vertical_weight_mutation_changes_boundary": changed != boundary,
        "gray_stats": gray_stats.as_json(),
        "histogram_total_assignments": sum(histogram),
        "factor_restoration": independent_factor_restoration(n, unary, edges),
        "rank_certificate": interface_rank_certificate(n, edges),
    }


def main() -> int:
    if pow(ROOT_MOD103, PRIME, MODULUS) != 1 or ROOT_MOD103 == 1:
        fail("F103 root does not have declared exact order 17")
    cases = [case(n, family) for n in SIZES for family in FAMILIES]
    if not all(
        item["butterfly_gray_boundary_agreement"]
        and item["exact_operation_counts"]
        and item["column_stage_reorder_agrees"]
        and item["omitted_stage_changes_boundary"]
        and item["vertical_weight_mutation_changes_boundary"]
        and item["histogram_total_assignments"] == 1 << (item["n"] * item["n"])
        and item["factor_restoration"]["forward_changed_seed"]
        and item["factor_restoration"]["seed_restored_exactly"]
        and item["factor_restoration"]["unload_restored_zero_backing"]
        and item["factor_restoration"]["same_backing"]
        and item["rank_certificate"]["explicit_matrix_rank_mod103"] == 1 << item["n"]
        and item["rank_certificate"]["one_zero_weight_matrix_rank_mod103"] == 1 << (item["n"] - 1)
        for item in cases
    ):
        fail("independent butterfly oracle failed")
    result = {
        "experiment": "INDEPENDENT_RUNTIME_WEIGHTED_F17_GRID_KRONECKER_BUTTERFLY_ORACLE",
        "result": "PASS",
        "imports_production_m120_or_m119": False,
        "imports_phase_backend": False,
        "representation": "CANONICAL_16_INTEGER_POWER_BASIS_OF_Q_ZETA17",
        "cases": cases,
        "all_boundaries_reconstructed_from_gray_histograms": True,
        "all_factor_cells_restore_exactly_on_same_backing": True,
        "all_interface_ranks_reexecuted_from_explicit_f103_matrices": True,
        "all_zero_weight_rank_halving_controls_pass": True,
        "all_operation_counts_reexecuted": True,
        "butterfly_recurrence_role": "STRONGEST_EVALUATED_GENERIC_ROW_INTERFACE_RECURRENCE_NOT_PROVEN_PARETO_OPTIMAL",
        "gray_histogram_role": "LOWER_MEMORY_TWO_TO_THE_N_SQUARED_ASSIGNMENT_BASELINE",
        "broader_matchgate_holographic_add_mps_or_boundary_specific_reduction_ruled_out": False,
        "strict_scope": "TWO_PUBLIC_RUNTIME_WEIGHT_FAMILIES_ON_BINARY_GRIDS_N2_N3_N4_IN_DIRECT_PROCESS_SOFTWARE",
    }
    json.dump(result, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
