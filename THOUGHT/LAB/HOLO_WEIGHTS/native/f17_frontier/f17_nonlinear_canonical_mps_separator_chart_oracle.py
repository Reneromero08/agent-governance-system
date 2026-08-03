#!/usr/bin/env python3
"""Independent dense oracle for the M122 nonlinear separator diagnostic.

This file imports neither the production package nor any CAT_CAS phase
backend.  It reconstructs the public grid descriptors, row recurrence,
boundaries, inverse restoration, every separator-subset matricization rank,
and exact best static ordinary/projective decision-diagram order.  Dense row
messages are verification-only.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from functools import lru_cache
from typing import Any, Iterable

from sympy.polys.domains import QQ


PRIME = 17
EXACT_SIZES = (2, 3, 4)
STRUCTURAL_SIZES = tuple(range(2, 9))
FIELDS = ((103, 72), (137, 16))
FAMILIES = ("PRIMARY", "GENERIC")


def fail(message: str) -> None:
    raise RuntimeError(message)


class Field:
    def __init__(self, name: str, modulus: int = 0, root: int = 0) -> None:
        self.name = name
        self.modulus = modulus
        if modulus:
            self.domain = None
            self.zero = 0
            self.one = 1
            self.root = root % modulus
        else:
            self.domain = QQ.cyclotomic_field(PRIME)
            self.zero = self.domain.zero
            self.one = self.domain.one
            self.root = self.domain.convert(self.domain.ext)

    def add(self, left: Any, right: Any) -> Any:
        return (left + right) % self.modulus if self.modulus else left + right

    def sub(self, left: Any, right: Any) -> Any:
        return (left - right) % self.modulus if self.modulus else left - right

    def mul(self, left: Any, right: Any) -> Any:
        return (left * right) % self.modulus if self.modulus else left * right

    def inv(self, value: Any) -> Any:
        if value == self.zero:
            fail("oracle attempted zero inversion")
        return pow(value, self.modulus - 2, self.modulus) if self.modulus else self.one / value

    def div(self, numerator: Any, denominator: Any) -> Any:
        return self.mul(numerator, self.inv(denominator))

    def power(self, exponent: int) -> Any:
        exponent %= PRIME
        return pow(self.root, exponent, self.modulus) if self.modulus else self.root ** exponent

    def serial(self, value: Any) -> Any:
        if self.modulus:
            return int(value)
        coefficients = list(value.to_list())
        coefficients = [self.domain.domain.zero] * (16 - len(coefficients)) + coefficients
        return [
            [int(coefficient.numerator), int(coefficient.denominator)]
            for coefficient in reversed(coefficients)
        ]

    def key(self, value: Any) -> Any:
        if self.modulus:
            return int(value)
        return tuple(tuple(pair) for pair in self.serial(value))

    def payload_metrics(self, values: Iterable[Any]) -> dict[str, int]:
        values = list(values)
        if self.modulus:
            return {
                "payload_bits": len(values) * self.modulus.bit_length(),
                "maximum_numerator_signed_bits": self.modulus.bit_length(),
                "maximum_denominator_bits": 1,
            }
        serialized = [coefficient for value in values for coefficient in self.serial(value)]
        return {
            "payload_bits": sum(
                1 + abs(int(numerator)).bit_length() + max(1, int(denominator).bit_length())
                for numerator, denominator in serialized
            ),
            "maximum_numerator_signed_bits": max(
                1 + abs(int(numerator)).bit_length() for numerator, _ in serialized
            ),
            "maximum_denominator_bits": max(
                max(1, int(denominator).bit_length()) for _, denominator in serialized
            ),
        }


def summation(values: Iterable[Any], field: Field) -> Any:
    total = field.zero
    for value in values:
        total = field.add(total, value)
    return total


def vertex(n: int, row: int, column: int) -> int:
    return row * n + column


def graph(n: int) -> tuple[tuple[int, int], ...]:
    return (
        *((vertex(n, row, column), vertex(n, row, column + 1))
          for row in range(n) for column in range(n - 1)),
        *((vertex(n, row, column), vertex(n, row + 1, column))
          for row in range(n - 1) for column in range(n)),
    )


def descriptor(n: int, family: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    variant = 0 if family == "PRIMARY" else 1
    unary = [
        1 + ((row + 2 * column + variant) & 1)
        for row in range(n)
        for column in range(n)
    ]
    if family == "GENERIC":
        site = n * n - n + 1
        unary[site] = 1 + (unary[site] % 16)
    edges = tuple(1 + ((7 * ordinal + 3 * variant + n) % 16) for ordinal in range(len(graph(n))))
    return tuple(unary), edges


def descriptor_sha256(n: int, family: str) -> str:
    unary, edges = descriptor(n, family)
    return hashlib.sha256(
        json.dumps(
            {"n": n, "family": family, "unary": unary, "edges": edges},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def bits(index: int, n: int) -> tuple[int, ...]:
    return tuple((index >> (n - 1 - site)) & 1 for site in range(n))


def row_phase(
    n: int,
    row: int,
    assignment: int,
    unary: tuple[int, ...],
    edges: tuple[int, ...],
    lookup: dict[tuple[int, int], int],
) -> int:
    state = bits(assignment, n)
    total = sum(unary[vertex(n, row, site)] for site, bit in enumerate(state) if bit)
    total += sum(
        edges[lookup[(vertex(n, row, site), vertex(n, row, site + 1))]]
        for site in range(n - 1)
        if state[site] and state[site + 1]
    )
    return total % PRIME


def apply_vertical(
    values: list[Any],
    n: int,
    weights: tuple[int, ...],
    field: Field,
    *,
    inverse: bool,
) -> list[Any]:
    current = list(values)
    columns = range(n - 1, -1, -1) if inverse else range(n)
    for column in columns:
        root = field.power(weights[column])
        if inverse:
            scale = field.inv(field.sub(root, field.one))
            matrix = (
                (field.mul(root, scale), field.sub(field.zero, scale)),
                (field.sub(field.zero, scale), scale),
            )
        else:
            matrix = ((field.one, field.one), (field.one, root))
        mask = 1 << (n - 1 - column)
        following = list(current)
        for low in range(1 << n):
            if low & mask:
                continue
            high = low | mask
            low_value, high_value = current[low], current[high]
            following[low] = field.add(field.mul(matrix[0][0], low_value), field.mul(matrix[0][1], high_value))
            following[high] = field.add(field.mul(matrix[1][0], low_value), field.mul(matrix[1][1], high_value))
        current = following
    return current


def forward_trace(n: int, family: str, field: Field) -> tuple[list[list[Any]], list[Any]]:
    unary, edges = descriptor(n, family)
    topology = graph(n)
    lookup = {edge: ordinal for ordinal, edge in enumerate(topology)}
    values = [field.one for _ in range(1 << n)]
    trace: list[list[Any]] = []
    for row in range(n):
        if row:
            vertical = tuple(
                edges[lookup[(vertex(n, row - 1, column), vertex(n, row, column))]]
                for column in range(n)
            )
            values = apply_vertical(values, n, vertical, field, inverse=False)
        values = [
            field.mul(value, field.power(row_phase(n, row, index, unary, edges, lookup)))
            for index, value in enumerate(values)
        ]
        trace.append(list(values))
    return trace, values


def inverse_restore(n: int, family: str, values: list[Any], field: Field) -> list[Any]:
    unary, edges = descriptor(n, family)
    topology = graph(n)
    lookup = {edge: ordinal for ordinal, edge in enumerate(topology)}
    current = list(values)
    for row in range(n - 1, -1, -1):
        current = [
            field.mul(value, field.power(-row_phase(n, row, index, unary, edges, lookup)))
            for index, value in enumerate(current)
        ]
        if row:
            vertical = tuple(
                edges[lookup[(vertex(n, row - 1, column), vertex(n, row, column))]]
                for column in range(n)
            )
            current = apply_vertical(current, n, vertical, field, inverse=True)
    return current


def matrix_rank(matrix: list[list[Any]], field: Field) -> int:
    values = [list(row) for row in matrix]
    rank = 0
    for column in range(len(values[0]) if values else 0):
        pivot = next((row for row in range(rank, len(values)) if values[row][column] != field.zero), None)
        if pivot is None:
            continue
        values[rank], values[pivot] = values[pivot], values[rank]
        scale = field.inv(values[rank][column])
        values[rank] = [field.mul(value, scale) for value in values[rank]]
        for row in range(len(values)):
            if row == rank or values[row][column] == field.zero:
                continue
            coefficient = values[row][column]
            values[row] = [
                field.sub(value, field.mul(coefficient, pivot_value))
                for value, pivot_value in zip(values[row], values[rank], strict=True)
            ]
        rank += 1
        if rank == len(values):
            break
    return rank


def subset_matrix(values: list[Any], n: int, subset: int) -> list[list[Any]]:
    left_sites = [site for site in range(n) if subset & (1 << site)]
    right_sites = [site for site in range(n) if not subset & (1 << site)]
    matrix = [[None for _ in range(1 << len(right_sites))] for _ in range(1 << len(left_sites))]
    for index, value in enumerate(values):
        state = bits(index, n)
        left = sum(state[site] << (len(left_sites) - 1 - ordinal) for ordinal, site in enumerate(left_sites))
        right = sum(state[site] << (len(right_sites) - 1 - ordinal) for ordinal, site in enumerate(right_sites))
        matrix[left][right] = value
    return matrix  # type: ignore[return-value]


def all_subset_ranks(values: list[Any], n: int, field: Field) -> dict[str, int]:
    return {
        str(subset): matrix_rank(subset_matrix(values, n, subset), field)
        for subset in range(1, (1 << n) - 1)
    }


def natural_ranks(values: list[Any], n: int, field: Field) -> tuple[int, ...]:
    return tuple(
        matrix_rank(subset_matrix(values, n, (1 << cut) - 1), field)
        for cut in range(1, n)
    )


def residual_vector(values: list[Any], n: int, subset: int, assignment: int) -> tuple[Any, ...]:
    selected = [site for site in range(n) if subset & (1 << site)]
    remaining = [site for site in range(n) if not subset & (1 << site)]
    output: list[Any] = []
    for rest in range(1 << len(remaining)):
        state = [0] * n
        for ordinal, site in enumerate(selected):
            state[site] = (assignment >> (len(selected) - 1 - ordinal)) & 1
        for ordinal, site in enumerate(remaining):
            state[site] = (rest >> (len(remaining) - 1 - ordinal)) & 1
        index = sum(bit << (n - 1 - site) for site, bit in enumerate(state))
        output.append(values[index])
    return tuple(output)


def decision_diagram_subset_counts(values: list[Any], n: int, field: Field) -> tuple[dict[int, int], dict[int, int]]:
    ordinary: dict[int, int] = {}
    projective: dict[int, int] = {}
    for subset in range(1 << n):
        assigned = subset.bit_count()
        ordinary_classes: set[Any] = set()
        projective_classes: set[Any] = set()
        for assignment in range(1 << assigned):
            residual = residual_vector(values, n, subset, assignment)
            ordinary_classes.add(tuple(field.key(value) for value in residual))
            pivot = next((value for value in residual if value != field.zero), None)
            if pivot is None:
                normalized = tuple(field.key(value) for value in residual)
            else:
                normalized = tuple(field.key(field.div(value, pivot)) for value in residual)
            projective_classes.add(normalized)
        ordinary[subset] = len(ordinary_classes)
        projective[subset] = len(projective_classes)
    return ordinary, projective


def best_static_order(counts: dict[int, int], n: int) -> tuple[int, tuple[int, ...]]:
    full = (1 << n) - 1

    @lru_cache(maxsize=None)
    def solve(subset: int) -> tuple[int, tuple[int, ...]]:
        if subset == full:
            return 0, ()
        choices = []
        for variable in range(n):
            if subset & (1 << variable):
                continue
            suffix_cost, suffix = solve(subset | (1 << variable))
            choices.append((counts[subset] + suffix_cost, (variable,) + suffix))
        return min(choices)

    return solve(0)


def raw_tt_cells(rank_profile: tuple[int, ...]) -> int:
    ranks = (1, *rank_profile, 1)
    return sum(2 * ranks[index] * ranks[index + 1] for index in range(len(ranks) - 1))


def effective_tt_coordinates(rank_profile: tuple[int, ...]) -> int:
    return raw_tt_cells(rank_profile) - sum(rank * rank for rank in rank_profile)


def analyze(n: int, family: str, field: Field, *, decision_diagram: bool) -> dict[str, Any]:
    trace, final = forward_trace(n, family, field)
    restored = inverse_restore(n, family, final, field)
    rank_trace = [natural_ranks(values, n, field) for values in trace]
    subset_ranks = all_subset_ranks(final, n, field)
    all_orders_maximal = all(
        rank == min(1 << subset.bit_count(), 1 << (n - subset.bit_count()))
        for subset_text, rank in subset_ranks.items()
        for subset in (int(subset_text),)
    )
    result: dict[str, Any] = {
        "n": n,
        "family": family,
        "field": field.name,
        "descriptor_sha256": descriptor_sha256(n, family),
        "boundary": field.serial(summation(final, field)),
        "rank_trace": rank_trace,
        "final_all_subset_ranks": subset_ranks,
        "every_separator_bit_order_has_maximal_tt_ranks": all_orders_maximal,
        "final_natural_rank_profile": rank_trace[-1],
        "final_raw_tt_core_field_cells": raw_tt_cells(rank_trace[-1]),
        "final_effective_tt_chart_coordinates": effective_tt_coordinates(rank_trace[-1]),
        "row_message_field_coordinates": 1 << n,
        "all_final_message_entries_nonzero": all(value != field.zero for value in final),
        "all_final_message_entries_distinct": len({field.key(value) for value in final}) == len(final),
        "exact_inverse_restored_product_state": restored == [field.one for _ in range(1 << n)],
        "dense_row_message_cells_verification_only": 1 << n,
    }
    if not field.modulus:
        scale = final[0]
        ratios = [field.div(value, scale) for value in final[1:]]
        result["amplitude_bearing_projective_control"] = {
            "fixed_pivot_index": 0,
            "ratio_field_coordinates": (1 << n) - 1,
            "scale_field_coordinates": 1,
            "total_field_coordinates": 1 << n,
            "ratio_payload": field.payload_metrics(ratios),
            "scale_payload": field.payload_metrics([scale]),
            "combined_payload": field.payload_metrics([scale, *ratios]),
            "separator_compaction": False,
        }
    if decision_diagram:
        ordinary, projective = decision_diagram_subset_counts(final, n, field)
        ordinary_cost, ordinary_order = best_static_order(ordinary, n)
        projective_cost, projective_order = best_static_order(projective, n)
        ordinary_full = all(
            ordinary[subset] == 1 << subset.bit_count()
            for subset in range((1 << n) - 1)
        )
        projective_full = all(
            projective[subset] == 1 << subset.bit_count()
            for subset in range((1 << n) - 1)
        )
        result["all_order_decision_diagram"] = {
            "ordinary_subset_class_counts": {str(key): value for key, value in ordinary.items()},
            "projective_subset_class_counts": {str(key): value for key, value in projective.items()},
            "exact_best_static_quasi_reduced_layered_ordinary_nodes": ordinary_cost,
            "exact_best_static_quasi_reduced_layered_ordinary_order": ordinary_order,
            "exact_best_static_quasi_reduced_layered_projective_nodes": projective_cost,
            "exact_best_static_quasi_reduced_layered_projective_order": projective_order,
            "redundant_test_deletion_applied": False,
            "all_proper_prefix_ordinary_classes_full": ordinary_full,
            "all_proper_prefix_projective_classes_full": projective_full,
            "exact_best_static_fully_reduced_ordinary_mtbdd_nodes_when_certified": ordinary_cost if ordinary_full else None,
            "exact_best_static_fully_reduced_projective_evdd_nodes_when_certified": projective_cost if projective_full else None,
            "all_static_orders_optimized_by_subset_dynamic_program": True,
            "full_unreduced_binary_tree_nonterminal_nodes": (1 << n) - 1,
            "best_quasi_reduced_projective_arcs": 2 * projective_cost,
            "best_quasi_reduced_projective_scalar_edge_labels": 2 * projective_cost,
            "best_quasi_reduced_projective_unique_table_entries": projective_cost,
            "subset_dynamic_program_states": 1 << n,
            "residual_vectors_classified": 3 ** n,
            "residual_field_cells_examined": 4 ** n,
            "row_message_production_required_and_counted_separately": True,
        }
    return result


def run() -> dict[str, Any]:
    exact: list[dict[str, Any]] = []
    exact_field = Field("Q_ZETA17")
    for n in EXACT_SIZES:
        for family in FAMILIES:
            exact.append(analyze(n, family, exact_field, decision_diagram=n in (3, 4)))
    structural: list[dict[str, Any]] = []
    for modulus, root in FIELDS:
        field = Field(f"F{modulus}", modulus, root)
        for n in STRUCTURAL_SIZES:
            for family in FAMILIES:
                structural.append(analyze(n, family, field, decision_diagram=family == "GENERIC"))
    generic_n4 = next(item for item in exact if item["n"] == 4 and item["family"] == "GENERIC")
    return {
        "schema": "CAT_CAS_F17_NONLINEAR_CANONICAL_MPS_SEPARATOR_CHART_ORACLE_V1",
        "independent_of_production": True,
        "production_imported": False,
        "phase_backend_imported": False,
        "exact_dense_oracle_scope": {"field": "Q_ZETA17", "sizes": EXACT_SIZES, "families": FAMILIES},
        "dual_field_dense_oracle_scope": {"fields": [field for field, _ in FIELDS], "sizes": STRUCTURAL_SIZES, "families": FAMILIES},
        "exact": exact,
        "structural": structural,
        "generic_n4_discriminator": {
            "descriptor_sha256": generic_n4["descriptor_sha256"],
            "final_natural_rank_profile": generic_n4["final_natural_rank_profile"],
            "every_separator_bit_order_has_maximal_tt_ranks": generic_n4["every_separator_bit_order_has_maximal_tt_ranks"],
            "effective_tt_chart_coordinates": generic_n4["final_effective_tt_chart_coordinates"],
            "best_fully_reduced_projective_evdd_nodes": generic_n4["all_order_decision_diagram"]["exact_best_static_fully_reduced_projective_evdd_nodes_when_certified"],
            "full_tree_nodes": 15,
            "all_entries_nonzero": generic_n4["all_final_message_entries_nonzero"],
        },
        "controls": {
            "mutated_descriptor_changes_boundary": next(item for item in exact if item["n"] == 4 and item["family"] == "PRIMARY")["boundary"] != generic_n4["boundary"],
            "all_exact_inverse_restorations_pass": all(item["exact_inverse_restored_product_state"] for item in exact),
            "all_dual_field_inverse_restorations_pass": all(item["exact_inverse_restored_product_state"] for item in structural),
            "all_exact_message_entries_nonzero": all(item["all_final_message_entries_nonzero"] for item in exact),
            "dense_materialization_is_oracle_only": True,
        },
        "interpretation": {
            "projectivization_reduces_amplitude_bearing_coordinates": False,
            "generic_n4_tt_effective_coordinates_below_dense_width": False,
            "generic_n4_projective_evdd_below_full_tree": generic_n4["all_order_decision_diagram"]["exact_best_static_fully_reduced_projective_evdd_nodes_when_certified"] < 15,
            "dual_field_growth_is_exact_q_zeta17_proof_beyond_n4": False,
            "broader_holographic_or_global_algorithms_exhausted": False,
            "distinct_phase_resource_established": False,
            "computational_advantage_established": False,
        },
    }


def main() -> None:
    json.dump(run(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
