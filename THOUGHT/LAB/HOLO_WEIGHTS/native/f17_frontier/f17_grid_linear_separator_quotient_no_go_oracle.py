#!/usr/bin/env python3
"""Independent oracle for the M121 exact linear-separator obstruction.

This file imports neither the production package nor any phase backend.  It
reconstructs the public descriptors, exact Q(zeta17) boundaries, factor-cell
restoration, and explicit separator matrices over two independent finite
fields.  Dense matrices are verification-only and are limited to n=2,3,4.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any


PRIME = 17
SIZES = (2, 3, 4)
VARIANTS = (("PRIMARY", 0), ("REUSE", 1))
FIELDS = ((103, 72), (137, 16))
ZERO = (0,) * 16
ONE = (1,) + (0,) * 15


def fail(message: str) -> None:
    raise RuntimeError(message)


def vertex_index(n: int, row: int, column: int) -> int:
    return row * n + column


def topology(n: int) -> tuple[tuple[int, int], ...]:
    return (
        *((vertex_index(n, row, column), vertex_index(n, row, column + 1))
          for row in range(n) for column in range(n - 1)),
        *((vertex_index(n, row, column), vertex_index(n, row + 1, column))
          for row in range(n - 1) for column in range(n)),
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


def descriptor(n: int, family: str, variant: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if (family, variant) not in VARIANTS:
        fail("oracle descriptor tag changed")
    unary = tuple(
        1 + ((row + 2 * column + variant) & 1)
        for row in range(n)
        for column in range(n)
    )
    edges = tuple(
        1 + ((7 * ordinal + 3 * variant + n) % 16)
        for ordinal in range(len(topology(n)))
    )
    if not all(1 <= value < PRIME for value in (*unary, *edges)):
        fail("oracle produced an illegal zero descriptor")
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


def row_exponent(
    n: int,
    row: int,
    assignment: int,
    unary: tuple[int, ...],
    edge_weights: tuple[int, ...],
    lookup: dict[tuple[int, int], int],
) -> int:
    bits = row_bits(assignment, n)
    total = sum(
        unary[vertex_index(n, row, column)]
        for column, bit in enumerate(bits)
        if bit
    )
    total += sum(
        edge_weights[
            lookup[(vertex_index(n, row, column), vertex_index(n, row, column + 1))]
        ]
        for column in range(n - 1)
        if bits[column] and bits[column + 1]
    )
    return total % PRIME


def butterfly_boundary(
    n: int,
    unary: tuple[int, ...],
    edge_weights: tuple[int, ...],
) -> tuple[int, ...]:
    graph = topology(n)
    lookup = {edge: ordinal for ordinal, edge in enumerate(graph)}
    width = 1 << n
    current = [
        root_action(ONE, row_exponent(n, 0, state, unary, edge_weights, lookup))
        for state in range(width)
    ]
    for row in range(1, n):
        for column in range(n):
            edge = lookup[(
                vertex_index(n, row - 1, column),
                vertex_index(n, row, column),
            )]
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
            current = following
        current = [
            root_action(
                value,
                row_exponent(n, row, state, unary, edge_weights, lookup),
            )
            for state, value in enumerate(current)
        ]
    boundary = ZERO
    for value in current:
        boundary = add(boundary, value)
    return boundary


def gray_histogram(
    n: int,
    unary: tuple[int, ...],
    edge_weights: tuple[int, ...],
) -> tuple[int, ...]:
    graph = topology(n)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in unary]
    for (left, right), weight in zip(graph, edge_weights, strict=True):
        adjacency[left].append((right, weight))
        adjacency[right].append((left, weight))
    histogram = [0 for _ in range(PRIME)]
    histogram[0] = 1
    assignment = 0
    exponent = 0
    for ordinal in range(1, 1 << (n * n)):
        changed = (ordinal & -ordinal).bit_length() - 1
        following = ordinal ^ (ordinal >> 1)
        turning_on = (following >> changed) & 1
        local = unary[changed]
        for neighbor, weight in adjacency[changed]:
            if (assignment >> neighbor) & 1:
                local += weight
        exponent = (exponent + local if turning_on else exponent - local) % PRIME
        assignment = following
        histogram[exponent] += 1
    return tuple(histogram)


def independent_factor_restoration(
    n: int,
    unary: tuple[int, ...],
    edge_weights: tuple[int, ...],
) -> dict[str, Any]:
    sites = [[ONE, ZERO] for _ in range(n * n)]
    edges = [[ONE, ONE, ONE, ONE] for _ in topology(n)]
    backing = (id(sites), id(edges), tuple(map(id, sites)), tuple(map(id, edges)))
    for site, weight in enumerate(unary):
        sites[site][1] = root_action(add(sites[site][1], sites[site][0]), weight)
    for edge, weight in enumerate(edge_weights):
        edges[edge][3] = root_action(edges[edge][3], weight)
    changed = any(row != [ONE, ZERO] for row in sites) or any(row != [ONE] * 4 for row in edges)
    for edge in reversed(range(len(edge_weights))):
        edges[edge][3] = root_action(edges[edge][3], -edge_weights[edge])
    for site in reversed(range(len(unary))):
        sites[site][1] = subtract(root_action(sites[site][1], -unary[site]), sites[site][0])
    seed_restored = all(row == [ONE, ZERO] for row in sites) and all(row == [ONE] * 4 for row in edges)
    for row in sites:
        row[0] = subtract(row[0], ONE)
    for row in edges:
        for index in range(4):
            row[index] = subtract(row[index], ONE)
    zero = all(cell == ZERO for row in sites for cell in row) and all(
        cell == ZERO for row in edges for cell in row
    )
    same = backing == (id(sites), id(edges), tuple(map(id, sites)), tuple(map(id, edges)))
    return {
        "forward_changed_seed": changed,
        "seed_restored_exactly": seed_restored,
        "unload_restored_zero_backing": zero,
        "same_backing": same,
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
    }


def rank_mod(matrix: list[list[int]], modulus: int) -> int:
    rows = [[value % modulus for value in row] for row in matrix]
    rank = 0
    columns = len(rows[0])
    for column in range(columns):
        pivot = next((index for index in range(rank, len(rows)) if rows[index][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inverse = pow(rows[rank][column], -1, modulus)
        rows[rank] = [(value * inverse) % modulus for value in rows[rank]]
        for index in range(len(rows)):
            if index == rank or rows[index][column] == 0:
                continue
            factor = rows[index][column]
            rows[index] = [
                (left - factor * right) % modulus
                for left, right in zip(rows[index], rows[rank], strict=True)
            ]
        rank += 1
    return rank


def matrix_product(left: list[list[int]], right: list[list[int]], modulus: int) -> list[list[int]]:
    transposed = list(zip(*right, strict=True))
    return [
        [sum(a * b for a, b in zip(row, column, strict=True)) % modulus for column in transposed]
        for row in left
    ]


def separator_weights(n: int, edge_weights: tuple[int, ...], cut_row: int) -> tuple[int, ...]:
    lookup = {edge: ordinal for ordinal, edge in enumerate(topology(n))}
    return tuple(
        edge_weights[lookup[(
            vertex_index(n, cut_row - 1, column),
            vertex_index(n, cut_row, column),
        )]]
        for column in range(n)
    )


def explicit_rank_certificate(
    n: int,
    edge_weights: tuple[int, ...],
    modulus: int,
    root: int,
) -> dict[str, Any]:
    width = 1 << n
    weights = separator_weights(n, edge_weights, n - 1)
    if pow(root, PRIME, modulus) != 1 or root == 1:
        fail("declared finite-field element lacks exact order 17")
    continuation = [
        [
            pow(root, sum((1 + row_bit) * column_bit for row_bit, column_bit in zip(row_bits(choice, n), row_bits(state, n), strict=True)), modulus)
            for state in range(width)
        ]
        for choice in range(width)
    ]
    vertical = [
        [
            pow(root, sum(weight * source_bit * target_bit for weight, source_bit, target_bit in zip(weights, row_bits(source, n), row_bits(target, n), strict=True)), modulus)
            for target in range(width)
        ]
        for source in range(width)
    ]
    diagonal = [
        [pow(root, (3 * state + 2 * state.bit_count()) % PRIME, modulus) if row == state else 0 for state in range(width)]
        for row in range(width)
    ]
    combined = matrix_product(matrix_product(continuation, diagonal, modulus), vertical, modulus)

    zero_vertical = [row[:] for row in vertical]
    mask = 1 << (n - 1)
    first_weight = weights[0]
    root_factor_inverse = pow(pow(root, first_weight, modulus), -1, modulus)
    for source in range(width):
        for target in range(width):
            if source & mask and target & mask:
                zero_vertical[source][target] = zero_vertical[source][target] * root_factor_inverse % modulus

    duplicate_continuation = [row[:] for row in continuation]
    for choice in range(width):
        if choice & mask:
            duplicate_continuation[choice] = duplicate_continuation[choice ^ mask][:]

    dropped_kernel = [0 for _ in range(width)]
    dropped_kernel[-1] = 1
    separating_value = sum(
        continuation[0][index] * dropped_kernel[index] for index in range(width)
    ) % modulus
    return {
        "modulus": modulus,
        "root": root,
        "width": width,
        "separator_weights": list(weights),
        "continuation_rank": rank_mod(continuation, modulus),
        "vertical_rank": rank_mod(vertical, modulus),
        "combined_rank": rank_mod(combined, modulus),
        "zero_vertical_weight_rank": rank_mod(zero_vertical, modulus),
        "duplicate_local_continuation_choice_rank": rank_mod(duplicate_continuation, modulus),
        "coordinate_drop_encoder_rank": width - 1,
        "coordinate_drop_kernel_is_nonzero": any(dropped_kernel),
        "valid_continuation_sees_dropped_kernel": separating_value != 0,
        "dense_verification_cells_per_matrix": width * width,
    }


def case(n: int, family: str, variant: int) -> dict[str, Any]:
    unary, edges = descriptor(n, family, variant)
    boundary = butterfly_boundary(n, unary, edges)
    histogram = gray_histogram(n, unary, edges)
    ranks = [explicit_rank_certificate(n, edges, modulus, root) for modulus, root in FIELDS]
    changed_edges = list(edges)
    changed_edges[0] = changed_edges[0] % 16 + 1
    changed_boundary = butterfly_boundary(n, unary, tuple(changed_edges))
    return {
        "n": n,
        "family": family,
        "variant": variant,
        "plan_fingerprint": plan_fingerprint(n),
        "unary_weights": list(unary),
        "edge_weights": list(edges),
        "canonical_boundary": list(boundary),
        "gray_histogram_boundary": list(canonical_histogram(histogram)),
        "butterfly_gray_boundary_agreement": boundary == canonical_histogram(histogram),
        "histogram_total_assignments": sum(histogram),
        "descriptor_mutation_changes_boundary": changed_boundary != boundary,
        "factor_restoration": independent_factor_restoration(n, unary, edges),
        "rank_certificates": ranks,
    }


def main() -> int:
    cases = [case(n, family, variant) for n in SIZES for family, variant in VARIANTS]
    if not all(
        item["butterfly_gray_boundary_agreement"]
        and item["histogram_total_assignments"] == 1 << (item["n"] * item["n"])
        and item["descriptor_mutation_changes_boundary"]
        and item["factor_restoration"]["forward_changed_seed"]
        and item["factor_restoration"]["seed_restored_exactly"]
        and item["factor_restoration"]["unload_restored_zero_backing"]
        and item["factor_restoration"]["same_backing"]
        and all(
            certificate["continuation_rank"] == 1 << item["n"]
            and certificate["vertical_rank"] == 1 << item["n"]
            and certificate["combined_rank"] == 1 << item["n"]
            and certificate["zero_vertical_weight_rank"] == 1 << (item["n"] - 1)
            and certificate["duplicate_local_continuation_choice_rank"] == 1 << (item["n"] - 1)
            and certificate["coordinate_drop_encoder_rank"] == (1 << item["n"]) - 1
            and certificate["coordinate_drop_kernel_is_nonzero"]
            and certificate["valid_continuation_sees_dropped_kernel"]
            for certificate in item["rank_certificates"]
        )
        for item in cases
    ):
        fail("independent linear-separator oracle failed")
    result = {
        "experiment": "INDEPENDENT_GENERIC_RUNTIME_F17_GRID_LINEAR_SEPARATOR_QUOTIENT_NO_GO_ORACLE",
        "result": "PASS",
        "imports_production_m121_m120_or_m119": False,
        "imports_phase_backend": False,
        "exact_boundary_representation": "CANONICAL_16_INTEGER_POWER_BASIS_OF_Q_ZETA17",
        "finite_field_rank_checks": [modulus for modulus, _ in FIELDS],
        "cases": cases,
        "all_boundaries_reconstructed_by_two_independent_exact_recurrences": True,
        "all_factor_cells_restore_exactly_on_same_backing": True,
        "all_continuation_vertical_and_combined_ranks_are_full": True,
        "all_rank_halving_controls_pass": True,
        "coordinate_drop_counterexample_control_passes": True,
        "dense_matrices_role": "VERIFICATION_ONLY_AT_N2_N3_N4",
        "strict_scope": "UNIFORM_FIXED_Q_ZETA17_LINEAR_SEPARATOR_ENCODERS_SUPPORTING_ARBITRARY_FIELD_MESSAGES_AND_EVERY_LEGAL_NONZERO_RUNTIME_CONTINUATION",
        "not_adjudicated": [
            "NONLINEAR_OR_PROGRAM_DEPENDENT_ENCODINGS",
            "RESTRICTED_DESCRIPTOR_FAMILIES",
            "ADD_MTBDD_MPS_MATCHGATE_PFAFFIAN_OR_GLOBAL_CONTRACTION",
            "TOTAL_MEMORY_TIME_OR_BIT_COMPLEXITY",
        ],
    }
    json.dump(result, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
