#!/usr/bin/env python3
"""Independent exact oracle for the growing even-mode Phase-QEMU V2 family.

This file deliberately does not import the production package or Phase-QEMU
V1.  It constructs fixed-number homogeneous Fock sectors from compositions,
uses dense integer coefficient arrays with a shared sqrt(2)-denominator
exponent, derives matching substitutions directly from the declared creation
operator law, and computes bipartite ranks with independent rational and
finite-field eliminations.

No floating-point value participates in a scientific decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence


Occupation = tuple[int, ...]
Edge = tuple[int, int]


def fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def compositions(total: int, slots: int) -> list[Occupation]:
    """Return a deterministic descending occupation basis."""

    result: list[Occupation] = []

    def visit(prefix: tuple[int, ...], remaining: int, count: int) -> None:
        if count == 1:
            result.append(prefix + (remaining,))
            return
        for value in range(remaining, -1, -1):
            visit(prefix + (value,), remaining - value, count - 1)

    visit((), total, slots)
    return result


def canonical_edge(first: int, second: int) -> Edge:
    return (first, second) if first < second else (second, first)


def matching_a(modes: int) -> tuple[Edge, ...]:
    return tuple((mode, mode + 1) for mode in range(0, modes, 2))


def matching_b(modes: int) -> tuple[Edge, ...]:
    edges = [(0, modes - 1)]
    edges.extend((mode, mode + 1) for mode in range(1, modes - 1, 2))
    return tuple(edges)


def all_edges(modes: int) -> tuple[Edge, ...]:
    return tuple((left, right) for left in range(modes)
                 for right in range(left + 1, modes))


def graph_components(modes: int, edges: Iterable[Edge]) -> list[list[int]]:
    adjacency = [set() for _ in range(modes)]
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(range(modes))
    components: list[list[int]] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in sorted(adjacency[node], reverse=True):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    return components


def pair_expansion(left_count: int, right_count: int,
                   adjoint: bool) -> dict[tuple[int, int], int]:
    """Expand one directed exchange in an unnormalized monomial basis.

    Forward:
        x_i -> (x_i-x_j)/sqrt(2)
        x_j -> (x_i+x_j)/sqrt(2)

    Adjoint (the transpose real map):
        x_i -> (x_i+x_j)/sqrt(2)
        x_j -> (-x_i+x_j)/sqrt(2)
    """

    result: dict[tuple[int, int], int] = {}
    for left_to_i in range(left_count + 1):
        left_factor = math.comb(left_count, left_to_i)
        if not adjoint and (left_count - left_to_i) & 1:
            left_factor = -left_factor
        for right_to_i in range(right_count + 1):
            right_factor = math.comb(right_count, right_to_i)
            if adjoint and right_to_i & 1:
                right_factor = -right_factor
            destination_i = left_to_i + right_to_i
            destination_j = left_count + right_count - destination_i
            key = (destination_i, destination_j)
            result[key] = result.get(key, 0) + left_factor * right_factor
    return {key: value for key, value in result.items() if value}


@dataclass(frozen=True)
class Sector:
    modes: int
    particles: int
    basis: tuple[Occupation, ...]
    index: dict[Occupation, int]
    factorial_weight: tuple[int, ...]
    occupation_parity_mask: tuple[int, ...]

    @classmethod
    def create(cls, modes: int) -> "Sector":
        if modes < 4 or modes % 2:
            raise ValueError("the V2 family requires even modes >= 4")
        particles = modes // 2
        basis = tuple(compositions(particles, modes))
        return cls(
            modes=modes,
            particles=particles,
            basis=basis,
            index={occupation: offset for offset, occupation in enumerate(basis)},
            factorial_weight=tuple(
                math.prod(math.factorial(value) for value in occupation)
                for occupation in basis
            ),
            occupation_parity_mask=tuple(
                sum((value & 1) << mode
                    for mode, value in enumerate(occupation))
                for occupation in basis
            ),
        )


@dataclass(frozen=True)
class ExactState:
    coefficients: tuple[int, ...]
    sqrt2_denominator_exponent: int


@dataclass(frozen=True)
class MatchingPlan:
    transitions: tuple[tuple[tuple[int, int], ...], ...]
    active_particle_counts: tuple[int, ...]


def build_matching_plan(sector: Sector, edges: Sequence[Edge],
                        adjoint: bool = False) -> MatchingPlan:
    used: set[int] = set()
    for left, right in edges:
        if left == right or left in used or right in used:
            raise ValueError("matching edges must be disjoint")
        if not (0 <= left < sector.modes and 0 <= right < sector.modes):
            raise ValueError("matching edge is outside the sector")
        used.update((left, right))

    transitions: list[tuple[tuple[int, int], ...]] = []
    active_counts: list[int] = []
    for source in sector.basis:
        partial: dict[Occupation, int] = {
            tuple(source[mode] if mode not in used else 0
                  for mode in range(sector.modes)): 1
        }
        for left, right in edges:
            expanded = pair_expansion(source[left], source[right], adjoint)
            next_partial: dict[Occupation, int] = {}
            for occupation, coefficient in partial.items():
                for (left_value, right_value), factor in expanded.items():
                    destination = list(occupation)
                    destination[left] = left_value
                    destination[right] = right_value
                    key = tuple(destination)
                    next_partial[key] = next_partial.get(key, 0) + coefficient * factor
            partial = next_partial
        transitions.append(tuple(sorted(
            ((sector.index[destination], coefficient)
             for destination, coefficient in partial.items() if coefficient),
            key=lambda item: item[0],
        )))
        active_counts.append(sum(source[mode] for mode in used))
    return MatchingPlan(tuple(transitions), tuple(active_counts))


def canonical_state(coefficients: Sequence[int], exponent: int) -> ExactState:
    values = list(coefficients)
    while exponent >= 2 and values and all(value % 2 == 0 for value in values):
        values = [value // 2 for value in values]
        exponent -= 2
    return ExactState(tuple(values), exponent)


def initial_state(sector: Sector) -> ExactState:
    occupation = tuple(1 if mode % 2 == 0 else 0
                       for mode in range(sector.modes))
    coefficients = [0] * len(sector.basis)
    coefficients[sector.index[occupation]] = 1
    return ExactState(tuple(coefficients), 0)


def apply_matching(state: ExactState, plan: MatchingPlan) -> ExactState:
    increments = {
        plan.active_particle_counts[source_index]
        for source_index, coefficient in enumerate(state.coefficients)
        if coefficient
    }
    if len(increments) != 1:
        raise AssertionError("matching support lacks a common denominator exponent")
    increment = next(iter(increments))
    result = [0] * len(state.coefficients)
    for source_index, source_value in enumerate(state.coefficients):
        if source_value == 0:
            continue
        for destination_index, factor in plan.transitions[source_index]:
            result[destination_index] += source_value * factor
    return canonical_state(result, state.sqrt2_denominator_exponent + increment)


def kerr_edge_masks(sector: Sector, edges: Sequence[Edge]) -> tuple[int, ...]:
    masks: list[int] = []
    for occupation in sector.basis:
        mask = 0
        for edge_index, (left, right) in enumerate(edges):
            if (occupation[left] * occupation[right]) & 1:
                mask |= 1 << edge_index
        masks.append(mask)
    return tuple(masks)


def apply_kerr_mask(state: ExactState, occupation_edge_masks: Sequence[int],
                    selected_edge_mask: int) -> ExactState:
    return ExactState(
        tuple(
            -coefficient
            if (occupation_edge_masks[index] & selected_edge_mask).bit_count() & 1
            else coefficient
            for index, coefficient in enumerate(state.coefficients)
        ),
        state.sqrt2_denominator_exponent,
    )


def norm_squared(sector: Sector, state: ExactState) -> Fraction:
    numerator = sum(
        weight * coefficient * coefficient
        for weight, coefficient in zip(sector.factorial_weight,
                                       state.coefficients)
    )
    return Fraction(numerator, 1 << state.sqrt2_denominator_exponent)


def support_size(state: ExactState) -> int:
    return sum(coefficient != 0 for coefficient in state.coefficients)


def state_commitment(state: ExactState) -> str:
    payload = json.dumps(
        {
            "coefficients": state.coefficients,
            "sqrt2_denominator_exponent": state.sqrt2_denominator_exponent,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def projectively_equal(left: ExactState, right: ExactState) -> bool:
    """Compare rays without assuming equal denominator normalization."""

    left_pivot = next((value for value in left.coefficients if value), None)
    right_pivot = next((value for value in right.coefficients if value), None)
    if left_pivot is None or right_pivot is None:
        return left_pivot is right_pivot
    return all(
        left_value * right_pivot == right_value * left_pivot
        for left_value, right_value in zip(left.coefficients,
                                           right.coefficients)
    )


def parity_weights(sector: Sector, state: ExactState,
                   selector_mask: int) -> tuple[Fraction, Fraction]:
    even_numerator = 0
    odd_numerator = 0
    for mask, weight, coefficient in zip(
        sector.occupation_parity_mask,
        sector.factorial_weight,
        state.coefficients,
    ):
        term = weight * coefficient * coefficient
        if (mask & selector_mask).bit_count() & 1:
            odd_numerator += term
        else:
            even_numerator += term
    denominator = 1 << state.sqrt2_denominator_exponent
    return Fraction(even_numerator, denominator), Fraction(odd_numerator, denominator)


def coefficient_matrix(sector: Sector, state: ExactState,
                       cut: int) -> list[list[int]]:
    left_basis = sorted({occupation[:cut] for occupation in sector.basis},
                        reverse=True)
    right_basis = sorted({occupation[cut:] for occupation in sector.basis},
                         reverse=True)
    left_index = {occupation: index for index, occupation in enumerate(left_basis)}
    right_index = {occupation: index for index, occupation in enumerate(right_basis)}
    matrix = [[0] * len(right_basis) for _ in left_basis]
    for occupation, coefficient in zip(sector.basis, state.coefficients):
        matrix[left_index[occupation[:cut]]][right_index[occupation[cut:]]] = coefficient
    return matrix


def rational_rank(matrix: Sequence[Sequence[int]]) -> int:
    work = [[Fraction(value) for value in row] for row in matrix]
    if not work:
        return 0
    row_count = len(work)
    column_count = len(work[0])
    rank = 0
    for column in range(column_count):
        pivot = next((row for row in range(rank, row_count)
                      if work[row][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][column]
        for row in range(rank + 1, row_count):
            if not work[row][column]:
                continue
            factor = work[row][column] / pivot_value
            for offset in range(column, column_count):
                work[row][offset] -= factor * work[rank][offset]
        rank += 1
        if rank == row_count:
            break
    return rank


def modular_rank(matrix: Sequence[Sequence[int]], prime: int) -> int:
    work = [[value % prime for value in row] for row in matrix]
    if not work:
        return 0
    row_count = len(work)
    column_count = len(work[0])
    rank = 0
    for column in range(column_count):
        pivot = next((row for row in range(rank, row_count)
                      if work[row][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = pow(work[rank][column], prime - 2, prime)
        for row in range(rank + 1, row_count):
            if not work[row][column]:
                continue
            factor = work[row][column] * inverse % prime
            for offset in range(column, column_count):
                work[row][offset] = (
                    work[row][offset] - factor * work[rank][offset]
                ) % prime
        rank += 1
        if rank == row_count:
            break
    return rank


def certified_rank(sector: Sector, state: ExactState) -> dict[str, object]:
    matrix = coefficient_matrix(sector, state, sector.modes // 2)
    exact = rational_rank(matrix)
    residues = {
        str(prime): modular_rank(matrix, prime)
        for prime in (1_000_000_007, 1_000_000_009)
    }
    if any(rank != exact for rank in residues.values()):
        raise AssertionError("modular and exact Schmidt ranks disagree")
    return {
        "exact_rational_rank": exact,
        "independent_modular_ranks": residues,
    }


def gf2_rank(vectors: Iterable[int], width: int) -> int:
    pivots = [0] * width
    rank = 0
    for original in vectors:
        value = original
        while value:
            column = value.bit_length() - 1
            if pivots[column]:
                value ^= pivots[column]
            else:
                pivots[column] = value
                rank += 1
                break
    return rank


def deterministic_selector_record(sector: Sector,
                                  state: ExactState) -> dict[str, int]:
    support_masks = [
        mask for mask, coefficient in zip(sector.occupation_parity_mask,
                                          state.coefficients)
        if coefficient
    ]
    if not support_masks:
        raise AssertionError("unitary evolution produced empty support")
    anchor = support_masks[0]
    affine_rank = gf2_rank((mask ^ anchor for mask in support_masks[1:]),
                          sector.modes)
    all_nonempty = (1 << (sector.modes - affine_rank)) - 1
    full_selector = (1 << sector.modes) - 1
    full_is_deterministic = all(
        ((mask & full_selector).bit_count() & 1)
        == ((anchor & full_selector).bit_count() & 1)
        for mask in support_masks
    )
    proper_nonempty = all_nonempty - int(full_is_deterministic)
    return {
        "support_affine_gf2_rank": affine_rank,
        "all_nonempty_deterministic_selectors": all_nonempty,
        "conserved_full_system_selectors": int(full_is_deterministic),
        "proper_nonempty_deterministic_selectors": proper_nonempty,
    }


def count_parity_flip_selectors(sector: Sector, state: ExactState,
                                initial: ExactState) -> int:
    final_support = [
        mask for mask, coefficient in zip(sector.occupation_parity_mask,
                                          state.coefficients)
        if coefficient
    ]
    initial_mask = next(
        mask for mask, coefficient in zip(sector.occupation_parity_mask,
                                          initial.coefficients)
        if coefficient
    )
    hits = 0
    full_selector = (1 << sector.modes) - 1
    for selector in range(1, full_selector):
        parity = (final_support[0] & selector).bit_count() & 1
        if all(((mask & selector).bit_count() & 1) == parity
               for mask in final_support):
            initial_parity = (initial_mask & selector).bit_count() & 1
            hits += parity != initial_parity
    return hits


@dataclass(frozen=True)
class FamilyPlans:
    a: MatchingPlan
    a_dag: MatchingPlan
    b: MatchingPlan
    b_dag: MatchingPlan


def family_plans(sector: Sector) -> FamilyPlans:
    return FamilyPlans(
        a=build_matching_plan(sector, matching_a(sector.modes)),
        a_dag=build_matching_plan(sector, matching_a(sector.modes), True),
        b=build_matching_plan(sector, matching_b(sector.modes)),
        b_dag=build_matching_plan(sector, matching_b(sector.modes), True),
    )


def apply_named(state: ExactState, name: str, plans: FamilyPlans,
                main_kerr_masks: Sequence[int]) -> ExactState:
    if name == "A":
        return apply_matching(state, plans.a)
    if name == "A_DAG":
        return apply_matching(state, plans.a_dag)
    if name == "B":
        return apply_matching(state, plans.b)
    if name == "B_DAG":
        return apply_matching(state, plans.b_dag)
    if name == "K01":
        return apply_kerr_mask(state, main_kerr_masks, 1)
    raise ValueError(f"unknown operation {name}")


def fixture_record(modes: int) -> dict[str, object]:
    sector = Sector.create(modes)
    plans = family_plans(sector)
    main_edges = ((0, 1),)
    main_kerr_masks = kerr_edge_masks(sector, main_edges)
    initial = initial_state(sector)
    state = initial
    trace: list[dict[str, object]] = []
    for operation in ("A", "B", "K01", "A", "B"):
        state = apply_named(state, operation, plans, main_kerr_masks)
        if norm_squared(sector, state) != 1:
            raise AssertionError(f"n={modes} {operation} broke exact norm")
        rank = certified_rank(sector, state)
        trace.append({
            "operation": operation,
            "support_cells": support_size(state),
            "central_schmidt_rank": rank,
        })
    final = state
    restored = final
    for operation in ("B_DAG", "A_DAG", "K01", "B_DAG", "A_DAG"):
        restored = apply_named(restored, operation, plans, main_kerr_masks)
    if restored != initial or norm_squared(sector, restored) != 1:
        raise AssertionError(f"n={modes} public adjoints failed exact restoration")

    last_even, last_odd = parity_weights(sector, final, 1 << (modes - 1))
    # "Central-half" is the parity of the left half at the central bond, not
    # the geometrically middle block of modes.
    middle_count = modes // 2
    middle_start = 0
    middle_selector = sum(1 << mode for mode in range(middle_count))
    middle_even, middle_odd = parity_weights(sector, final, middle_selector)
    components = graph_components(
        modes, (*matching_a(modes), *matching_b(modes), *main_edges)
    )
    return {
        "modes": modes,
        "particles": sector.particles,
        "dimension": len(sector.basis),
        "exchange_kerr_graph_components": components,
        "connected_public_geometry": len(components) == 1,
        "forward_execution_order": ["A", "B", "K01", "A", "B"],
        "inverse_execution_order": [
            "B_DAG", "A_DAG", "K01", "B_DAG", "A_DAG"
        ],
        "final_support_cells": support_size(final),
        "final_state_commitment": state_commitment(final),
        "central_schmidt_trace": trace,
        "peak_central_schmidt_rank": max(
            item["central_schmidt_rank"]["exact_rational_rank"]
            for item in trace
        ),
        "last_mode_parity": {
            "selector_modes": [modes - 1],
            "even_weight": fraction_text(last_even),
            "odd_weight": fraction_text(last_odd),
        },
        "central_half_parity": {
            "selector_modes": list(range(middle_start,
                                         middle_start + middle_count)),
            "even_weight": fraction_text(middle_even),
            "odd_weight": fraction_text(middle_odd),
        },
        "restored_exactly": restored == initial,
        "restored_state_commitment": state_commitment(restored),
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
    }


def fixed_core_record(modes: int) -> dict[str, object]:
    sector = Sector.create(modes)
    a_edges = matching_a(modes)
    kerr_edges = ((1, 2), (3, 4))
    a = build_matching_plan(sector, a_edges)
    a_dag = build_matching_plan(sector, a_edges, True)
    kerr_masks = kerr_edge_masks(sector, kerr_edges)
    initial = initial_state(sector)
    after_a = apply_matching(initial, a)

    def close(selected_edges: int) -> ExactState:
        return apply_matching(
            apply_kerr_mask(after_a, kerr_masks, selected_edges),
            a_dag,
        )

    final = close(0b11)
    no_first_edge = close(0b10)
    no_second_edge = close(0b01)
    no_kerr = close(0)
    trace = [after_a, apply_kerr_mask(after_a, kerr_masks, 0b11), final]
    rank_records = [certified_rank(sector, item) for item in trace]
    peak_rank = max(record["exact_rational_rank"] for record in rank_records)
    final_rank = certified_rank(sector, final)

    selector_mask = (1 << 0) | (1 << 4)
    full_boundary = parity_weights(sector, final, selector_mask)

    def ablation_record(state: ExactState) -> dict[str, object]:
        boundary = parity_weights(sector, state, selector_mask)
        return {
            "projective_state_changed": not projectively_equal(final, state),
            "declared_boundary_changed": boundary != full_boundary,
            "support_cells": support_size(state),
            "boundary_even_weight": fraction_text(boundary[0]),
            "boundary_odd_weight": fraction_text(boundary[1]),
            "state_commitment": state_commitment(state),
        }

    restored = apply_matching(
        apply_kerr_mask(apply_matching(final, a), kerr_masks, 0b11),
        a_dag,
    )
    components = graph_components(modes, (*a_edges, *kerr_edges))
    edge_ablations = {
        "remove_K_12": ablation_record(no_first_edge),
        "remove_K_34": ablation_record(no_second_edge),
    }
    return {
        "modes": modes,
        "public_word": ["A_n", "K_{(1,2),(3,4)}", "A_n_DAG"],
        "full_matching_edges": [list(edge) for edge in a_edges],
        "fixed_kerr_core_edges": [list(edge) for edge in kerr_edges],
        "declared_boundary_selector_modes": [0, 4],
        "graph_components": components,
        "graph_component_sizes": [len(component) for component in components],
        "connected_public_geometry": len(components) == 1,
        "final_support_cells": support_size(final),
        "central_schmidt_trace": rank_records,
        "peak_central_schmidt_rank": peak_rank,
        "final_central_schmidt_rank": final_rank,
        "boundary_even_weight": fraction_text(full_boundary[0]),
        "boundary_odd_weight": fraction_text(full_boundary[1]),
        "final_state_commitment": state_commitment(final),
        "edge_ablation_controls": edge_ablations,
        "both_kerr_edges_change_projective_state_and_boundary": all(
            item["projective_state_changed"]
            and item["declared_boundary_changed"]
            for item in edge_ablations.values()
        ),
        "no_kerr_control": ablation_record(no_kerr),
        "restored_exactly": restored == initial,
        "restored_state_commitment": state_commitment(restored),
    }


def exhaustive_n6_record() -> dict[str, object]:
    sector = Sector.create(6)
    plans = family_plans(sector)
    edges = all_edges(6)
    edge_masks = kerr_edge_masks(sector, edges)
    initial = initial_state(sector)
    prefix = apply_matching(apply_matching(initial, plans.a), plans.b)
    all_nonempty_word1_hits = 0
    all_nonempty_word2_hits = 0
    conserved_full_word1_hits = 0
    conserved_full_word2_hits = 0
    proper_word1_hits = 0
    proper_word2_hits = 0
    word2_parity_flip_hits = 0
    minimum_word1_affine_rank = sector.modes
    minimum_word2_affine_rank = sector.modes

    for selected_edges in range(1, 1 << len(edges)):
        after_kerr = apply_kerr_mask(prefix, edge_masks, selected_edges)
        word1 = apply_matching(apply_matching(after_kerr, plans.a), plans.b)
        word2 = apply_matching(
            apply_matching(after_kerr, plans.b_dag), plans.a_dag
        )
        first = deterministic_selector_record(sector, word1)
        second = deterministic_selector_record(sector, word2)
        all_nonempty_word1_hits += first["all_nonempty_deterministic_selectors"]
        all_nonempty_word2_hits += second["all_nonempty_deterministic_selectors"]
        conserved_full_word1_hits += first["conserved_full_system_selectors"]
        conserved_full_word2_hits += second["conserved_full_system_selectors"]
        proper_word1_hits += first["proper_nonempty_deterministic_selectors"]
        proper_word2_hits += second["proper_nonempty_deterministic_selectors"]
        minimum_word1_affine_rank = min(
            minimum_word1_affine_rank, first["support_affine_gf2_rank"]
        )
        minimum_word2_affine_rank = min(
            minimum_word2_affine_rank, second["support_affine_gf2_rank"]
        )
        if second["proper_nonempty_deterministic_selectors"]:
            word2_parity_flip_hits += count_parity_flip_selectors(
                sector, word2, initial
            )

    edge_subsets = (1 << len(edges)) - 1
    if conserved_full_word1_hits != edge_subsets:
        raise AssertionError("fixed-N global parity was not conserved in word 1")
    if conserved_full_word2_hits != edge_subsets:
        raise AssertionError("fixed-N global parity was not conserved in word 2")
    if proper_word1_hits or word2_parity_flip_hits:
        raise AssertionError("n=6 exhaustive route-kill theorem failed")
    return {
        "modes": 6,
        "particles": 3,
        "edge_count": len(edges),
        "nonempty_kerr_edge_subsets": edge_subsets,
        "searched_selector_scope": "ALL_NONEMPTY_PROPER_MODE_SUBSETS",
        "excluded_trivial_selector": {
            "modes": [0, 1, 2, 3, 4, 5],
            "reason": "FIXED_N_GLOBAL_PARTICLE_PARITY_IS_CONSERVED",
        },
        "word1": {
            "form": "A B K_E A B",
            "proper_nonempty_deterministic_selector_hits": proper_word1_hits,
            "conserved_full_system_selector_hits": conserved_full_word1_hits,
            "all_nonempty_deterministic_selector_hits_including_global": (
                all_nonempty_word1_hits
            ),
            "minimum_support_affine_gf2_rank": minimum_word1_affine_rank,
        },
        "word2": {
            "form": "A B K_E B_DAG A_DAG",
            "proper_nonempty_deterministic_selector_closures": proper_word2_hits,
            "proper_nonempty_parity_flip_closures": word2_parity_flip_hits,
            "conserved_full_system_selector_hits": conserved_full_word2_hits,
            "all_nonempty_deterministic_selector_hits_including_global": (
                all_nonempty_word2_hits
            ),
            "minimum_support_affine_gf2_rank": minimum_word2_affine_rank,
        },
        "theorem": (
            "ZERO_WORD1_NONTRIVIAL_DETERMINISTIC_BOUNDARIES_AND_"
            "ZERO_WORD2_NONTRIVIAL_PARITY_FLIP_CLOSURES_AT_N6"
        ),
    }


def verify_expected_laws(fixtures: dict[str, dict[str, object]],
                         fixed_core: dict[str, dict[str, object]]) -> None:
    expected = {
        "4": {
            "dimension": 10,
            "final_support_cells": 1,
            "peak_central_schmidt_rank": 2,
        },
        "6": {
            "dimension": 56,
            "final_support_cells": 54,
            "peak_central_schmidt_rank": 8,
            "last_mode_parity": ("95/128", "33/128"),
            "central_half_parity": ("125/256", "131/256"),
        },
        "8": {
            "dimension": 330,
            "final_support_cells": 292,
            "peak_central_schmidt_rank": 18,
            "last_mode_parity": ("767/1024", "257/1024"),
            "central_half_parity": ("273/512", "239/512"),
        },
    }
    for key, required in expected.items():
        observed = fixtures[key]
        for field in ("dimension", "final_support_cells",
                      "peak_central_schmidt_rank"):
            if observed[field] != required[field]:
                raise AssertionError(
                    f"n={key} {field}: {observed[field]} != {required[field]}"
                )
        for field in ("last_mode_parity", "central_half_parity"):
            if field not in required:
                continue
            pair = (observed[field]["even_weight"],
                    observed[field]["odd_weight"])
            if pair != required[field]:
                raise AssertionError(
                    f"n={key} {field}: {pair} != {required[field]}"
                )
        if not observed["connected_public_geometry"] or not observed["restored_exactly"]:
            raise AssertionError(f"n={key} lacks connectedness or restoration")
    fixed_expected = {
        "6": {"component_sizes": [6], "rank": 4},
        "8": {"component_sizes": [6, 2], "rank": 2},
    }
    for key, observed in fixed_core.items():
        required = fixed_expected[key]
        if (observed["graph_component_sizes"] != required["component_sizes"]
                or observed["final_support_cells"] != 4
                or observed["peak_central_schmidt_rank"] != required["rank"]
                or observed["final_central_schmidt_rank"]["exact_rational_rank"]
                != required["rank"]
                or not observed[
                    "both_kerr_edges_change_projective_state_and_boundary"
                ]
                or not observed["restored_exactly"]):
            raise AssertionError(f"n={key} fixed-core disconnected control failed")


def result() -> dict[str, object]:
    fixtures = {str(modes): fixture_record(modes) for modes in (4, 6, 8)}
    fixed_core = {str(modes): fixed_core_record(modes) for modes in (6, 8)}
    verify_expected_laws(fixtures, fixed_core)
    exhaustive = exhaustive_n6_record()
    return {
        "schema": "phase-qemu-v2-growing-even-mode-qnd-bond-separate-reference-v1",
        "implementation_independence": {
            "imports_production": False,
            "imports_phase_qemu_v1": False,
            "state_representation": (
                "DENSE_INTEGER_HOMOGENEOUS_COEFFICIENT_ARRAY_WITH_SHARED_SQRT2_EXPONENT"
            ),
            "operator_construction": "DIRECT_BINOMIAL_CREATION_OPERATOR_SUBSTITUTION",
            "rank_oracles": ["EXACT_RATIONAL_ELIMINATION", "TWO_PRIME_MODULAR_ELIMINATION"],
            "floating_point_scientific_decisions": False,
        },
        "family": {
            "modes": "n=2m",
            "particles": "N=m",
            "initial": "|1010...10>",
            "dimension_formula": "binomial(n+N-1,N)",
            "A": "product_j R_(2j,2j+1)",
            "B": "R_(0,n-1) product_j R_(2j+1,2j+2)",
            "R_ij": {
                "creation_i": "(creation_i-creation_j)/sqrt(2)",
                "creation_j": "(creation_i+creation_j)/sqrt(2)",
            },
            "K_E": "(-1)^(sum_(i,j in E) n_i*n_j)",
            "boundary": "P_S=(-1)^(sum_(i in S) n_i)",
        },
        "fixtures": fixtures,
        "fixed_core_disconnected_controls": fixed_core,
        "exhaustive_n6": exhaustive,
        "controls": {
            "dimensions_10_56_330": [fixtures[key]["dimension"]
                                      for key in ("4", "6", "8")] == [10, 56, 330],
            "supports_1_54_292": [fixtures[key]["final_support_cells"]
                                   for key in ("4", "6", "8")] == [1, 54, 292],
            "peak_central_ranks_2_8_18": [
                fixtures[key]["peak_central_schmidt_rank"]
                for key in ("4", "6", "8")
            ] == [2, 8, 18],
            "all_connected_fixtures_restore": all(
                fixture["connected_public_geometry"]
                and fixture["restored_exactly"]
                for fixture in fixtures.values()
            ),
            "fixed_core_public_control_matches_declared_geometry": all(
                control["final_support_cells"] == 4
                and control[
                    "both_kerr_edges_change_projective_state_and_boundary"
                ]
                and control["restored_exactly"]
                for control in fixed_core.values()
            ),
            "n6_nontrivial_exhaustive_zero_hit": (
                exhaustive["word1"]["proper_nonempty_deterministic_selector_hits"] == 0
                and exhaustive["word2"]["proper_nonempty_parity_flip_closures"] == 0
            ),
        },
        "verification_classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "EXACT_IDEAL_SOFTWARE_GROWING_EVEN_MODE_DIAGNOSTIC_ONLY",
        "claim_limits": {
            "phase_qemu_device_execution": False,
            "machine_enforced_custody": False,
            "physical_bosons_or_phonons": False,
            "physical_qnd_detector": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "m257_escape": False,
            "small_wall_crossing": False,
            "unbounded_compute": False,
            "replacement_of_physical_bits_with_pi": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    encoded = json.dumps(result(), indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
