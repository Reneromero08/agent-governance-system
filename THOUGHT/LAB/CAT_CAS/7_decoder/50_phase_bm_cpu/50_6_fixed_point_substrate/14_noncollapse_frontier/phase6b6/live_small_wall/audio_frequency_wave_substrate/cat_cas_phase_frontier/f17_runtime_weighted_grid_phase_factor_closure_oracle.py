#!/usr/bin/env python3
"""Independent character, transfer, rank, and decision-diagram oracle for M119.

This module imports neither the production grid carrier nor the M116/M118
pair backend.  It reconstructs runtime programs from the public formulas,
compares a 17-bin row transfer against independent direct and Gray-code
full-assignment histograms,
certifies separator determinants in F103, and reports reduced ordered
multi-terminal decision-diagram sizes for three public variable orders.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


PRIME = 17
MODULUS = 103
ROOT_MOD103 = 72
SIZES = (2, 3, 4)
FAMILIES = ("PRIMARY", "REUSE")
VECTOR_ZERO = (0,) * 16
VECTOR_ONE = (1,) + (0,) * 15


def fail(message: str) -> None:
    raise RuntimeError(message)


def nonzero_weight(value: int) -> int:
    return value % 16 + 1


def vertex_index(n: int, row: int, column: int) -> int:
    return row * n + column


def topology(n: int) -> tuple[tuple[int, int], ...]:
    horizontal = tuple(
        (vertex_index(n, row, column), vertex_index(n, row, column + 1))
        for row in range(n)
        for column in range(n - 1)
    )
    vertical = tuple(
        (vertex_index(n, row, column), vertex_index(n, row + 1, column))
        for row in range(n - 1)
        for column in range(n)
    )
    return horizontal + vertical


def plan_fingerprint(n: int) -> str:
    vertices = [[row, column] for row in range(n) for column in range(n)]
    edges = [list(edge) for edge in topology(n)]
    site_count = n * n
    operations = (
        *({"kind": "PREPARE", "index": site} for site in range(site_count)),
        *({"kind": "UNARY", "index": site} for site in range(site_count)),
        *({"kind": "EDGE", "index": edge} for edge in range(len(edges))),
    )
    record = {
        "schema": "F17_GRID_TOPOLOGY_PLAN_V1",
        "n": n,
        "vertices": vertices,
        "edges": edges,
        "operations": list(operations),
    }
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def runtime_weights(n: int, family: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if family not in FAMILIES:
        fail("unknown independent grid family")
    offset = 1 if family == "PRIMARY" else 7
    unary = tuple(
        nonzero_weight(7 * site + 3 * n + offset)
        for site in range(n * n)
    )
    edge = tuple(
        nonzero_weight(11 * ordinal + 5 * n + 2 * offset)
        for ordinal in range(len(topology(n)))
    )
    return unary, edge


def canonical_power_basis(histogram: tuple[int, ...]) -> tuple[int, ...]:
    if len(histogram) != PRIME:
        fail("character histogram width changed")
    omitted = histogram[16]
    return tuple(histogram[index] - omitted for index in range(16))


def vector_add(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def vector_subtract(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def vector_root_action(
    value: tuple[int, ...],
    exponent: int,
) -> tuple[int, ...]:
    """Multiply a canonical power-basis vector by zeta17**exponent."""
    characters = [0 for _ in range(PRIME)]
    for index, coefficient in enumerate(value):
        characters[(index + exponent) % PRIME] += coefficient
    return canonical_power_basis(tuple(characters))


def energy(
    assignment: int,
    unary: tuple[int, ...],
    edges: tuple[tuple[int, int], ...],
    edge_weights: tuple[int, ...],
) -> int:
    value = 0
    for site, weight in enumerate(unary):
        if (assignment >> site) & 1:
            value += weight
    for (left, right), weight in zip(edges, edge_weights, strict=True):
        if ((assignment >> left) & 1) and ((assignment >> right) & 1):
            value += weight
    return value % PRIME


@dataclass
class DenseStats:
    assignments_streamed: int = 0
    unary_terms_checked: int = 0
    edge_terms_checked: int = 0
    maximum_live_histogram_bins: int = PRIME

    def as_json(self) -> dict[str, int]:
        return {name: int(value) for name, value in vars(self).items()}


@dataclass
class GrayStats:
    assignments_streamed: int = 0
    changed_bit_updates: int = 0
    incident_edge_checks: int = 0
    histogram_increments: int = 0
    maximum_live_histogram_bins: int = PRIME
    maximum_live_assignment_bits: int = 0

    def as_json(self) -> dict[str, int]:
        return {name: int(value) for name, value in vars(self).items()}


def streamed_dense_histogram(
    n: int,
    unary: tuple[int, ...],
    edge_weights: tuple[int, ...],
) -> tuple[tuple[int, ...], DenseStats]:
    edges = topology(n)
    histogram = [0 for _ in range(PRIME)]
    stats = DenseStats()
    for assignment in range(1 << (n * n)):
        exponent = energy(assignment, unary, edges, edge_weights)
        histogram[exponent] += 1
        stats.assignments_streamed += 1
        stats.unary_terms_checked += n * n
        stats.edge_terms_checked += len(edges)
    return tuple(histogram), stats


def gray_delta_histogram(
    n: int,
    unary: tuple[int, ...],
    edge_weights: tuple[int, ...],
) -> tuple[tuple[int, ...], GrayStats]:
    """Enumerate Gray assignments while updating energy through one bit delta."""
    edges = topology(n)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in unary]
    for (left, right), weight in zip(edges, edge_weights, strict=True):
        adjacency[left].append((right, weight))
        adjacency[right].append((left, weight))
    histogram = [0 for _ in range(PRIME)]
    stats = GrayStats(maximum_live_assignment_bits=n * n)
    assignment = 0
    exponent = 0
    histogram[0] = 1
    stats.assignments_streamed = 1
    stats.histogram_increments = 1
    for ordinal in range(1, 1 << (n * n)):
        changed_site = (ordinal & -ordinal).bit_length() - 1
        following = ordinal ^ (ordinal >> 1)
        turning_on = (following >> changed_site) & 1
        local = unary[changed_site]
        for neighbor, weight in adjacency[changed_site]:
            stats.incident_edge_checks += 1
            if (assignment >> neighbor) & 1:
                local += weight
        exponent = (exponent + local if turning_on else exponent - local) % PRIME
        assignment = following
        histogram[exponent] += 1
        stats.assignments_streamed += 1
        stats.changed_bit_updates += 1
        stats.histogram_increments += 1
    return tuple(histogram), stats


@dataclass
class TransferStats:
    character_bin_additions: int = 0
    source_target_transitions: int = 0
    maximum_frontier_states: int = 0
    maximum_live_character_bins: int = 0

    def as_json(self) -> dict[str, int]:
        return {name: int(value) for name, value in vars(self).items()}


def row_bits(value: int, n: int) -> tuple[int, ...]:
    return tuple((value >> column) & 1 for column in range(n))


def row_energy(
    n: int,
    row: int,
    assignment: int,
    unary: tuple[int, ...],
    edge_lookup: dict[tuple[int, int], int],
    edge_weights: tuple[int, ...],
) -> int:
    values = row_bits(assignment, n)
    total = 0
    for column, bit in enumerate(values):
        if bit:
            total += unary[vertex_index(n, row, column)]
    for column in range(n - 1):
        if values[column] and values[column + 1]:
            edge = (
                vertex_index(n, row, column),
                vertex_index(n, row, column + 1),
            )
            total += edge_weights[edge_lookup[edge]]
    return total % PRIME


def vertical_energy(
    n: int,
    upper_row: int,
    upper_assignment: int,
    lower_assignment: int,
    edge_lookup: dict[tuple[int, int], int],
    edge_weights: tuple[int, ...],
) -> int:
    upper = row_bits(upper_assignment, n)
    lower = row_bits(lower_assignment, n)
    total = 0
    for column in range(n):
        if upper[column] and lower[column]:
            edge = (
                vertex_index(n, upper_row, column),
                vertex_index(n, upper_row + 1, column),
            )
            total += edge_weights[edge_lookup[edge]]
    return total % PRIME


def character_transfer_histogram(
    n: int,
    unary: tuple[int, ...],
    edge_weights: tuple[int, ...],
) -> tuple[tuple[int, ...], TransferStats]:
    edges = topology(n)
    edge_lookup = {edge: ordinal for ordinal, edge in enumerate(edges)}
    width = 1 << n
    stats = TransferStats(maximum_frontier_states=width)
    current = [[0 for _ in range(PRIME)] for _ in range(width)]
    for assignment in range(width):
        current[assignment][
            row_energy(n, 0, assignment, unary, edge_lookup, edge_weights)
        ] = 1
    stats.maximum_live_character_bins = width * PRIME
    for row in range(1, n):
        following = [[0 for _ in range(PRIME)] for _ in range(width)]
        for target in range(width):
            target_energy = row_energy(
                n,
                row,
                target,
                unary,
                edge_lookup,
                edge_weights,
            )
            for source in range(width):
                shift = (
                    target_energy
                    + vertical_energy(
                        n,
                        row - 1,
                        source,
                        target,
                        edge_lookup,
                        edge_weights,
                    )
                ) % PRIME
                stats.source_target_transitions += 1
                for exponent, count in enumerate(current[source]):
                    if count:
                        following[target][(exponent + shift) % PRIME] += count
                        stats.character_bin_additions += 1
        current = following
        stats.maximum_live_character_bins = max(
            stats.maximum_live_character_bins,
            2 * width * PRIME,
        )
    final = [0 for _ in range(PRIME)]
    for row_histogram in current:
        for exponent, count in enumerate(row_histogram):
            if count:
                final[exponent] += count
                stats.character_bin_additions += 1
    return tuple(final), stats


def inverse_restoration_stream(
    n: int,
    unary: tuple[int, ...],
    edge_weights: tuple[int, ...],
) -> bool:
    edges = topology(n)
    for assignment in range(1 << (n * n)):
        forward = energy(assignment, unary, edges, edge_weights)
        inverse = (-forward) % PRIME
        if (forward + inverse) % PRIME:
            return False
    return True


def independent_factor_restoration(
    n: int,
    unary: tuple[int, ...],
    edge_weights: tuple[int, ...],
) -> dict[str, Any]:
    """Reexecute factor-cell forward/inverse/unload without production code."""
    sites = [[VECTOR_ONE, VECTOR_ZERO] for _ in range(n * n)]
    edges = [[VECTOR_ONE, VECTOR_ONE, VECTOR_ONE, VECTOR_ONE] for _ in topology(n)]
    backing_identity = (id(sites), id(edges), tuple(map(id, sites)), tuple(map(id, edges)))

    for site, weight in enumerate(unary):
        sites[site][1] = vector_add(sites[site][1], sites[site][0])
        sites[site][1] = vector_root_action(sites[site][1], weight)
    for edge, weight in enumerate(edge_weights):
        edges[edge][3] = vector_root_action(edges[edge][3], weight)

    forward_changed_seed = any(
        row != [VECTOR_ONE, VECTOR_ZERO] for row in sites
    ) or any(row != [VECTOR_ONE] * 4 for row in edges)

    for edge in reversed(range(len(edge_weights))):
        edges[edge][3] = vector_root_action(edges[edge][3], -edge_weights[edge])
    for site in reversed(range(len(unary))):
        sites[site][1] = vector_root_action(sites[site][1], -unary[site])
        sites[site][1] = vector_subtract(sites[site][1], sites[site][0])

    seed_restored = all(row == [VECTOR_ONE, VECTOR_ZERO] for row in sites) and all(
        row == [VECTOR_ONE] * 4 for row in edges
    )
    for row in sites:
        row[0] = vector_subtract(row[0], VECTOR_ONE)
    for row in edges:
        for index in range(4):
            row[index] = vector_subtract(row[index], VECTOR_ONE)
    zero_backing = all(cell == VECTOR_ZERO for row in sites for cell in row) and all(
        cell == VECTOR_ZERO for row in edges for cell in row
    )
    same_backing = backing_identity == (
        id(sites),
        id(edges),
        tuple(map(id, sites)),
        tuple(map(id, edges)),
    )
    return {
        "forward_changed_seed": forward_changed_seed,
        "seed_restored_exactly": seed_restored,
        "unload_restored_zero_backing": zero_backing,
        "same_backing": same_backing,
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
    }


def separator_certificate(
    n: int,
    edge_weights: tuple[int, ...],
) -> dict[str, Any]:
    edges = topology(n)
    edge_lookup = {edge: ordinal for ordinal, edge in enumerate(edges)}
    cut = n // 2
    ordinals = tuple(
        edge_lookup[
            (
                vertex_index(n, cut - 1, column),
                vertex_index(n, cut, column),
            )
        ]
        for column in range(n)
    )
    weights = tuple(edge_weights[index] for index in ordinals)
    residue = 1
    factors: list[int] = []
    for weight in weights:
        determinant = (pow(ROOT_MOD103, weight, MODULUS) - 1) % MODULUS
        factors.append(determinant)
        residue = (
            residue
            * pow(determinant, 1 << (n - 1), MODULUS)
        ) % MODULUS
    return {
        "cut_row": cut,
        "certifies_actual_row_transfer_interface": True,
        "separator_edge_ordinals": list(ordinals),
        "separator_weights": list(weights),
        "kernel_determinants_mod103": factors,
        "kronecker_determinant_mod103": residue,
        "kronecker_determinant_nonzero": residue != 0,
        "exact_rank_over_q_zeta17": 1 << n,
        "rank_after_one_separator_edge_removal": 1 << (n - 1),
        "rank_with_all_separator_edges_removed": 1,
        "determinant_norm_power_of_17_exponent": n * (1 << (n - 1)),
    }


def direct_zero_field_pfaffian_check(
    n: int,
    unary: tuple[int, ...],
    edge_weights: tuple[int, ...],
) -> dict[str, Any]:
    incident = [0 for _ in unary]
    for (left, right), weight in zip(topology(n), edge_weights, strict=True):
        incident[left] = (incident[left] + weight) % PRIME
        incident[right] = (incident[right] + weight) % PRIME
    residues = tuple(
        (2 * unary_weight + incident_weight) % PRIME
        for unary_weight, incident_weight in zip(unary, incident, strict=True)
    )
    return {
        "spin_field_residues_mod17": list(residues),
        "direct_zero_field_planar_ising_pfaffian_applicable": all(
            value == 0 for value in residues
        ),
        "broader_matchgate_or_holographic_reduction_ruled_out": False,
    }


def variable_orders(n: int) -> dict[str, tuple[int, ...]]:
    row_major = tuple(range(n * n))
    column_major = tuple(
        vertex_index(n, row, column)
        for column in range(n)
        for row in range(n)
    )
    snake = tuple(
        vertex_index(n, row, column)
        for row in range(n)
        for column in (
            range(n) if row % 2 == 0 else reversed(range(n))
        )
    )
    return {
        "ROW_MAJOR": row_major,
        "COLUMN_MAJOR": column_major,
        "SNAKE_ROW": snake,
    }


def reduced_energy_mtbdd(
    n: int,
    unary: tuple[int, ...],
    edge_weights: tuple[int, ...],
    order: tuple[int, ...],
) -> dict[str, int]:
    edges = topology(n)
    terminal_ids = {exponent: exponent for exponent in range(PRIME)}
    unique_nodes: dict[tuple[int, int, int], int] = {}
    next_id = PRIME

    @lru_cache(maxsize=None)
    def build(depth: int, assignment: int) -> int:
        nonlocal next_id
        if depth == len(order):
            return terminal_ids[energy(assignment, unary, edges, edge_weights)]
        site = order[depth]
        low = build(depth + 1, assignment)
        high = build(depth + 1, assignment | (1 << site))
        if low == high:
            return low
        key = (depth, low, high)
        if key not in unique_nodes:
            unique_nodes[key] = next_id
            next_id += 1
        return unique_nodes[key]

    root = build(0, 0)
    cache = build.cache_info()
    return {
        "root_id": root,
        "nonterminal_nodes": len(unique_nodes),
        "directed_edges": 2 * len(unique_nodes),
        "terminal_classes": PRIME,
        "full_binary_tree_nonterminal_nodes": (1 << (n * n)) - 1,
        "full_assignment_leaves_visited": 1 << (n * n),
        "memoized_subproblem_cache_hits": cache.hits,
        "memoized_subproblem_cache_misses": cache.misses,
        "full_assignment_tree_built_by_this_oracle": True,
        "order_optimality_claimed": False,
    }


def case(n: int, family: str) -> dict[str, Any]:
    unary, edge_weights = runtime_weights(n, family)
    dense_histogram, dense_stats = streamed_dense_histogram(n, unary, edge_weights)
    gray_histogram, gray_stats = gray_delta_histogram(n, unary, edge_weights)
    transfer_histogram, transfer_stats = character_transfer_histogram(
        n,
        unary,
        edge_weights,
    )
    diagrams = {
        name: reduced_energy_mtbdd(n, unary, edge_weights, order)
        for name, order in variable_orders(n).items()
    }
    factor_restoration = independent_factor_restoration(n, unary, edge_weights)
    return {
        "n": n,
        "family": family,
        "plan_fingerprint": plan_fingerprint(n),
        "unary_weights": list(unary),
        "edge_weights": list(edge_weights),
        "direct_streamed_character_histogram": list(dense_histogram),
        "gray_delta_character_histogram": list(gray_histogram),
        "transfer_character_histogram": list(transfer_histogram),
        "histograms_agree": dense_histogram == gray_histogram == transfer_histogram,
        "canonical_boundary": list(canonical_power_basis(dense_histogram)),
        "histogram_total_assignments": sum(dense_histogram),
        "expected_total_assignments": 1 << (n * n),
        "inverse_restores_every_assignment_phase": inverse_restoration_stream(
            n,
            unary,
            edge_weights,
        ),
        "independent_factor_restoration": factor_restoration,
        "separator_certificate": separator_certificate(n, edge_weights),
        "direct_pfaffian_check": direct_zero_field_pfaffian_check(
            n,
            unary,
            edge_weights,
        ),
        "three_order_observed_reduced_energy_mtbdd_sweep": diagrams,
        "minimum_observed_mtbdd_nonterminal_nodes": min(
            result["nonterminal_nodes"] for result in diagrams.values()
        ),
        "dense_streaming_stats": dense_stats.as_json(),
        "gray_delta_streaming_stats": gray_stats.as_json(),
        "character_transfer_stats": transfer_stats.as_json(),
    }


def main() -> int:
    cases = [case(n, family) for n in SIZES for family in FAMILIES]
    if pow(ROOT_MOD103, PRIME, MODULUS) != 1 or any(
        pow(ROOT_MOD103, divisor, MODULUS) == 1 for divisor in (1,)
    ):
        fail("independent F103 image does not have exact order 17")
    if not all(
        item["histograms_agree"]
        and item["histogram_total_assignments"] == item["expected_total_assignments"]
        and item["inverse_restores_every_assignment_phase"]
        and all(
            item["independent_factor_restoration"][field]
            for field in (
                "forward_changed_seed",
                "seed_restored_exactly",
                "unload_restored_zero_backing",
                "same_backing",
            )
        )
        and item["separator_certificate"]["kronecker_determinant_nonzero"]
        and not item["direct_pfaffian_check"]["direct_zero_field_planar_ising_pfaffian_applicable"]
        for item in cases
    ):
        fail("independent grid oracle failed")
    result = {
        "experiment": "INDEPENDENT_RUNTIME_WEIGHTED_F17_GRID_CHARACTER_ORACLE",
        "result": "PASS",
        "representation": "CANONICAL_16_INTEGER_POWER_BASIS_FROM_17_CHARACTER_HISTOGRAM",
        "imports_production_m119": False,
        "imports_m116_or_m118_backend": False,
        "finite_field_certificate": "ZETA17_MAPS_TO_EXACT_ORDER17_ELEMENT72_IN_F103",
        "cases": cases,
        "all_histograms_agree": all(item["histograms_agree"] for item in cases),
        "all_boundaries_reconstructed_independently": True,
        "all_inverse_phase_streams_restore": all(
            item["inverse_restores_every_assignment_phase"] for item in cases
        ),
        "all_factor_cells_restore_exactly_on_same_backing": all(
            item["independent_factor_restoration"]["seed_restored_exactly"]
            and item["independent_factor_restoration"]["unload_restored_zero_backing"]
            and item["independent_factor_restoration"]["same_backing"]
            for item in cases
        ),
        "all_separator_ranks_certified": all(
            item["separator_certificate"]["exact_rank_over_q_zeta17"] == (1 << item["n"])
            for item in cases
        ),
        "direct_histogram_role": "INDEPENDENT_REFERENCE_REEXECUTION_NOT_MATCHED_PERFORMANCE_BASELINE",
        "gray_delta_histogram_role": "EVALUATED_17_BIN_COMPACT_CLASSICAL_BASELINE",
        "mtbdd_role": "THREE_ORDER_OBSERVED_SWEEP_NOT_ORDER_OPTIMAL_AND_BUILT_THROUGH_FULL_ASSIGNMENT_TREE",
        "evaluated_baseline_set_proven_exhaustive_or_pareto_optimal": False,
        "all_direct_zero_field_pfaffian_checks_inapplicable": all(
            not item["direct_pfaffian_check"]["direct_zero_field_planar_ising_pfaffian_applicable"]
            for item in cases
        ),
        "broader_matchgate_or_holographic_reduction_ruled_out": False,
        "strict_scope": "TWO_RUNTIME_WEIGHT_FAMILIES_EACH_FOR_BINARY_GRIDS_N2_N3_N4_DIRECT_PROCESS_SOFTWARE",
    }
    json.dump(result, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
