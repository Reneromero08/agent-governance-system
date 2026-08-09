#!/usr/bin/env python3
"""Exact M260 growing-family QND-versus-bond diagnostic.

The accepted calculations use integer coefficients of homogeneous creation
polynomials.  A complete 50:50 matching contributes one common
``2**(N/2)`` denominator, so probabilities are exact dyadic rationals even
when normalized Fock amplitudes contain square roots.  Scientific decisions
never use floating point.

The exhaustive n=6 search evaluates every nonempty pi cross-Kerr graph on the
complete six-mode graph.  A 15-bit Walsh-Hadamard transform generates a
verifier-only 56 by 32768 coefficient table for each declared word form; this
table is counted and is not an accepted carrier or compiler.  All nonempty
proper subset-parity selectors are checked.  The full-system selector is
reported separately because fixed total particle number makes it a conserved
parity sham for every word.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence


Occupation = tuple[int, ...]
Pair = tuple[int, int]


def fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def signed_bits(value: int) -> int:
    """Material signed width for one exact integer coefficient."""

    return 1 if value == 0 else abs(value).bit_length() + 1


def weak_compositions(parts: int, total: int) -> list[Occupation]:
    """Return the public fixed-number Fock basis in deterministic order."""

    def generate(prefix: Occupation, remaining_parts: int, remaining: int):
        if remaining_parts == 1:
            yield prefix + (remaining,)
            return
        for value in range(remaining + 1):
            yield from generate(prefix + (value,), remaining_parts - 1, remaining - value)

    return list(generate((), parts, total))


def occupation_factorial(occupation: Occupation) -> int:
    return math.prod(math.factorial(value) for value in occupation)


def occupation_parity_mask(occupation: Occupation) -> int:
    mask = 0
    for mode, value in enumerate(occupation):
        if value & 1:
            mask |= 1 << mode
    return mask


def pair_image(particles_i: int, particles_j: int, adjoint: bool) -> tuple[tuple[int, int, int], ...]:
    """Integer numerator of a directed 50:50 R_ij Fock-polynomial image.

    Forward uses
      a_i^dagger -> (a_i^dagger - a_j^dagger)/sqrt(2)
      a_j^dagger -> (a_i^dagger + a_j^dagger)/sqrt(2).
    The adjoint is its public transpose.  The omitted common denominator is
    2**((particles_i + particles_j)/2).
    """

    total = particles_i + particles_j
    coefficients: dict[int, int] = {}
    for chosen_i_from_i in range(particles_i + 1):
        for chosen_i_from_j in range(particles_j + 1):
            output_i = chosen_i_from_i + chosen_i_from_j
            if adjoint:
                sign = -1 if chosen_i_from_j & 1 else 1
            else:
                sign = -1 if (particles_i - chosen_i_from_i) & 1 else 1
            coefficient = (
                sign
                * math.comb(particles_i, chosen_i_from_i)
                * math.comb(particles_j, chosen_i_from_j)
            )
            coefficients[output_i] = coefficients.get(output_i, 0) + coefficient
    return tuple(
        (output_i, total - output_i, coefficient)
        for output_i, coefficient in sorted(coefficients.items())
        if coefficient
    )


@dataclass(frozen=True)
class ExactState:
    coefficients: tuple[int, ...]
    matching_layers: int


class FixedNumberSpace:
    def __init__(self, modes: int, particles: int):
        self.modes = modes
        self.particles = particles
        self.basis = weak_compositions(modes, particles)
        self.index = {occupation: index for index, occupation in enumerate(self.basis)}
        self.factorials = tuple(occupation_factorial(occupation) for occupation in self.basis)
        self.parity_masks = tuple(occupation_parity_mask(occupation) for occupation in self.basis)
        self.complete_edges = tuple(itertools.combinations(range(modes), 2))
        self.edge_index = {edge: index for index, edge in enumerate(self.complete_edges)}
        self.edge_signatures = tuple(self._edge_signature(occupation) for occupation in self.basis)
        self._matching_cache: dict[
            tuple[tuple[Pair, ...], bool], tuple[tuple[tuple[int, int], ...], ...]
        ] = {}

    def _edge_signature(self, occupation: Occupation) -> int:
        signature = 0
        for edge_index, (first, second) in enumerate(self.complete_edges):
            if (occupation[first] * occupation[second]) & 1:
                signature |= 1 << edge_index
        return signature

    def basis_state(self, occupation: Occupation) -> ExactState:
        coefficients = [0] * len(self.basis)
        coefficients[self.index[occupation]] = 1
        return ExactState(tuple(coefficients), 0)

    def matching_images(
        self, pairs: Sequence[Pair], adjoint: bool = False
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        key = (tuple(pairs), adjoint)
        cached = self._matching_cache.get(key)
        if cached is not None:
            return cached

        covered = sorted(mode for pair in pairs for mode in pair)
        if covered != list(range(self.modes)):
            raise ValueError("a matching must cover every mode exactly once")

        images: list[tuple[tuple[int, int], ...]] = []
        for occupation in self.basis:
            partial: dict[Occupation, int] = {(0,) * self.modes: 1}
            for first, second in pairs:
                local = pair_image(occupation[first], occupation[second], adjoint)
                updated: dict[Occupation, int] = {}
                for output, outer_coefficient in partial.items():
                    for output_i, output_j, inner_coefficient in local:
                        candidate = list(output)
                        candidate[first] = output_i
                        candidate[second] = output_j
                        candidate_tuple = tuple(candidate)
                        updated[candidate_tuple] = updated.get(candidate_tuple, 0) + (
                            outer_coefficient * inner_coefficient
                        )
                partial = updated
            images.append(
                tuple(
                    sorted(
                        (
                            (self.index[output], coefficient)
                            for output, coefficient in partial.items()
                            if coefficient
                        ),
                        key=lambda item: item[0],
                    )
                )
            )
        result = tuple(images)
        self._matching_cache[key] = result
        return result

    def apply_images(
        self,
        state: ExactState,
        images: tuple[tuple[tuple[int, int], ...], ...],
    ) -> ExactState:
        output = [0] * len(self.basis)
        for source, amplitude in enumerate(state.coefficients):
            if not amplitude:
                continue
            for destination, coefficient in images[source]:
                output[destination] += amplitude * coefficient
        return ExactState(tuple(output), state.matching_layers + 1)

    def apply_matching(
        self, state: ExactState, pairs: Sequence[Pair], adjoint: bool = False
    ) -> ExactState:
        return self.apply_images(state, self.matching_images(pairs, adjoint))

    def edge_mask(self, edges: Iterable[Pair]) -> int:
        mask = 0
        for first, second in edges:
            edge = (first, second) if first < second else (second, first)
            mask |= 1 << self.edge_index[edge]
        return mask

    def apply_kerr_mask(self, state: ExactState, edge_mask: int) -> ExactState:
        output = tuple(
            -amplitude
            if (edge_mask & signature).bit_count() & 1
            else amplitude
            for amplitude, signature in zip(state.coefficients, self.edge_signatures)
        )
        return ExactState(output, state.matching_layers)

    def norm_squared(self, state: ExactState) -> Fraction:
        numerator = sum(
            coefficient * coefficient * factorial
            for coefficient, factorial in zip(state.coefficients, self.factorials)
        )
        denominator = 1 << (state.matching_layers * self.particles)
        return Fraction(numerator, denominator)

    def parity_weights(self, state: ExactState, selector: int) -> tuple[Fraction, Fraction]:
        even_numerator = 0
        odd_numerator = 0
        for coefficient, factorial, parity_mask in zip(
            state.coefficients, self.factorials, self.parity_masks
        ):
            contribution = coefficient * coefficient * factorial
            if (selector & parity_mask).bit_count() & 1:
                odd_numerator += contribution
            else:
                even_numerator += contribution
        denominator = 1 << (state.matching_layers * self.particles)
        return Fraction(even_numerator, denominator), Fraction(odd_numerator, denominator)

    def central_schmidt_rank(self, state: ExactState) -> int:
        split = self.modes // 2
        total_rank = 0
        for left_particles in range(self.particles + 1):
            left_basis = weak_compositions(split, left_particles)
            right_basis = weak_compositions(self.modes - split, self.particles - left_particles)
            matrix = [
                [
                    state.coefficients[self.index[left + right]]
                    for right in right_basis
                ]
                for left in left_basis
            ]
            total_rank += rational_matrix_rank(matrix)
        return total_rank

    def state_commitment(self, state: ExactState) -> str:
        payload = json.dumps(
            {
                "basis": self.basis,
                "coefficients": state.coefficients,
                "matching_layers": state.matching_layers,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def exact_initial_restored(self, state: ExactState, occupation: Occupation) -> bool:
        exponent = state.matching_layers * self.particles
        if exponent & 1:
            return False
        expected_scale = 1 << (exponent // 2)
        expected = [0] * len(self.basis)
        expected[self.index[occupation]] = expected_scale
        return state.coefficients == tuple(expected)


def rational_matrix_rank(matrix: list[list[int]]) -> int:
    if not matrix or not matrix[0]:
        return 0
    work = [[Fraction(value) for value in row] for row in matrix]
    rows = len(work)
    columns = len(work[0])
    rank = 0
    for column in range(columns):
        pivot = next((row for row in range(rank, rows) if work[row][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][column]
        work[rank] = [value / pivot_value for value in work[rank]]
        for row in range(rows):
            if row == rank or not work[row][column]:
                continue
            factor = work[row][column]
            work[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(work[row], work[rank])
            ]
        rank += 1
        if rank == rows:
            break
    return rank


def matching_a(modes: int) -> tuple[Pair, ...]:
    return tuple((mode, mode + 1) for mode in range(0, modes, 2))


def matching_b(modes: int) -> tuple[Pair, ...]:
    return ((0, modes - 1),) + tuple((mode, mode + 1) for mode in range(1, modes - 1, 2))


def alternating_initial(modes: int) -> Occupation:
    return tuple(1 if mode % 2 == 0 else 0 for mode in range(modes))


def compose_images(
    dimension: int,
    first: tuple[tuple[tuple[int, int], ...], ...],
    second: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    result: list[tuple[tuple[int, int], ...]] = []
    for source in range(dimension):
        accumulated: dict[int, int] = {}
        for middle, first_coefficient in first[source]:
            for destination, second_coefficient in second[middle]:
                accumulated[destination] = accumulated.get(destination, 0) + (
                    first_coefficient * second_coefficient
                )
        result.append(
            tuple(sorted((index, value) for index, value in accumulated.items() if value))
        )
    return tuple(result)


def projectively_equal(left: Sequence[int], right: Sequence[int]) -> bool:
    relation: int | None = None
    for left_value, right_value in zip(left, right):
        if left_value == 0 and right_value == 0:
            continue
        if left_value == right_value:
            candidate = 1
        elif left_value == -right_value:
            candidate = -1
        else:
            return False
        if relation is None:
            relation = candidate
        elif relation != candidate:
            return False
    return relation is not None


def component_sizes(modes: int, edges: Iterable[Pair]) -> list[int]:
    adjacency = [set() for _ in range(modes)]
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    unseen = set(range(modes))
    sizes = []
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def parity_record(space: FixedNumberSpace, state: ExactState, selector: int) -> dict[str, object]:
    even, odd = space.parity_weights(state, selector)
    return {
        "selector_modes": [mode for mode in range(space.modes) if selector & (1 << mode)],
        "even_weight": fraction_text(even),
        "odd_weight": fraction_text(odd),
        "factorized": (even == 1 and odd == 0) or (even == 0 and odd == 1),
        "pointer_purity": fraction_text(even * even + odd * odd),
        "pointer_schmidt_rank": int(bool(even)) + int(bool(odd)),
    }


def state_record(space: FixedNumberSpace, state: ExactState) -> dict[str, object]:
    norm = space.norm_squared(state)
    if norm != 1:
        raise AssertionError(f"state norm is {fraction_text(norm)}, not one")
    return {
        "support": sum(bool(value) for value in state.coefficients),
        "central_schmidt_rank": space.central_schmidt_rank(state),
        "matching_layers": state.matching_layers,
        "maximum_integer_coefficient_signed_bits": max(
            signed_bits(value) for value in state.coefficients
        ),
        "common_sqrt2_denominator_exponent": state.matching_layers * space.particles,
        "commitment": space.state_commitment(state),
    }


def execute_primary(modes: int) -> dict[str, object]:
    particles = modes // 2
    space = FixedNumberSpace(modes, particles)
    initial_occupation = alternating_initial(modes)
    state = space.basis_state(initial_occupation)
    pairs_a = matching_a(modes)
    pairs_b = matching_b(modes)
    kerr_edges = ((0, 1),)
    kerr_mask = space.edge_mask(kerr_edges)

    stages = []
    state = space.apply_matching(state, pairs_a)
    stages.append({"operation": "A", **state_record(space, state)})
    state = space.apply_matching(state, pairs_b)
    stages.append({"operation": "B", **state_record(space, state)})
    state = space.apply_kerr_mask(state, kerr_mask)
    stages.append({"operation": "K_01", **state_record(space, state)})
    state = space.apply_matching(state, pairs_a)
    stages.append({"operation": "A", **state_record(space, state)})
    state = space.apply_matching(state, pairs_b)
    stages.append({"operation": "B", **state_record(space, state)})
    final_state = state

    disabled = space.basis_state(initial_occupation)
    for pairs in (pairs_a, pairs_b, pairs_a, pairs_b):
        disabled = space.apply_matching(disabled, pairs)

    restored = space.apply_matching(final_state, pairs_b, adjoint=True)
    restored = space.apply_matching(restored, pairs_a, adjoint=True)
    restored = space.apply_kerr_mask(restored, kerr_mask)
    restored = space.apply_matching(restored, pairs_b, adjoint=True)
    restored = space.apply_matching(restored, pairs_a, adjoint=True)

    wrong = space.apply_matching(final_state, pairs_b, adjoint=True)
    wrong = space.apply_matching(wrong, pairs_a, adjoint=True)
    wrong = space.apply_kerr_mask(wrong, space.edge_mask(((0, modes - 1),)))
    wrong = space.apply_matching(wrong, pairs_b, adjoint=True)
    wrong = space.apply_matching(wrong, pairs_a, adjoint=True)

    # A descriptor-distinct word consumes the returned restored value directly.
    # Exact normalized equality is established, but this immutable functional
    # diagnostic does not preserve a QEMU or same-backing allocation identity.
    reuse_kerr_edges = ((1, 2),)
    reuse_kerr_mask = space.edge_mask(reuse_kerr_edges)
    reused = space.apply_matching(restored, pairs_a)
    reused = space.apply_kerr_mask(reused, reuse_kerr_mask)
    reused = space.apply_matching(reused, pairs_a, adjoint=True)
    reuse_boundary = parity_record(space, reused, 1 << 0)
    reused_restored = space.apply_matching(reused, pairs_a)
    reused_restored = space.apply_kerr_mask(reused_restored, reuse_kerr_mask)
    reused_restored = space.apply_matching(reused_restored, pairs_a, adjoint=True)

    last_selector = 1 << (modes - 1)
    central_selector = sum(1 << mode for mode in range(modes // 2))
    return {
        "modes": modes,
        "particles": particles,
        "sector_dimension": len(space.basis),
        "public_descriptor": {
            "initial": "alternating unit Fock occupancy",
            "A": [list(pair) for pair in pairs_a],
            "B": [list(pair) for pair in pairs_b],
            "K": [list(pair) for pair in kerr_edges],
            "word": ["A", "B", "K", "A", "B"],
        },
        "stages": stages,
        "final": {
            **state_record(space, final_state),
            "last_mode_parity": parity_record(space, final_state, last_selector),
            "central_half_parity": parity_record(space, final_state, central_selector),
        },
        "controls": {
            "kerr_disabled_final_commitment": space.state_commitment(disabled),
            "kerr_changes_projective_state": not projectively_equal(
                final_state.coefficients, disabled.coefficients
            ),
            "kerr_disabled_last_mode_parity": parity_record(
                space, disabled, last_selector
            ),
            "kerr_disabled_central_half_parity": parity_record(
                space, disabled, central_selector
            ),
            "kerr_changes_last_mode_boundary": space.parity_weights(
                final_state, last_selector
            )
            != space.parity_weights(disabled, last_selector),
            "kerr_changes_central_half_boundary": space.parity_weights(
                final_state, central_selector
            )
            != space.parity_weights(disabled, central_selector),
            "public_adjoint_exact_restore": space.exact_initial_restored(
                restored, initial_occupation
            ),
            "wrong_kerr_inverse_rejected": not space.exact_initial_restored(
                wrong, initial_occupation
            ),
            "descriptor_distinct_reuse": {
                "word": ["A", "K_12", "A_DAGGER"],
                "consumes_returned_restored_value": True,
                "same_backing_reuse_established": False,
                "boundary": reuse_boundary,
                "forward_commitment": space.state_commitment(reused),
                "public_adjoint_exact_restore": space.exact_initial_restored(
                    reused_restored, initial_occupation
                ),
            },
            "no_baseline_reload": True,
        },
        "resource_signature": {
            "logical_modes": modes,
            "bosons": particles,
            "exact_sparse_sector_dimension": len(space.basis),
            "resident_state_integer_coefficient_cells": len(space.basis),
            "common_sqrt2_denominator_exponent_after_forward": (
                final_state.matching_layers * particles
            ),
            "forward_matching_count": 4,
            "forward_exchange_count": 2 * modes,
            "forward_kerr_edge_count": 1,
            "retained_inverse_history": 0,
            "inverse_derived_from_public_word": True,
        },
    }


def fwht(values: list[int]) -> None:
    span = 1
    length = len(values)
    while span < length:
        block = span * 2
        for start in range(0, length, block):
            stop = start + span
            for index in range(start, stop):
                left = values[index]
                right = values[index + span]
                values[index] = left + right
                values[index + span] = left - right
        span = block


def exhaustive_tables(
    space: FixedNumberSpace,
    prefix: ExactState,
    suffix_images: tuple[tuple[tuple[int, int], ...], ...],
) -> list[list[int]]:
    graph_count = 1 << len(space.complete_edges)
    rows = [[0] * graph_count for _ in space.basis]
    for source, source_coefficient in enumerate(prefix.coefficients):
        if not source_coefficient:
            continue
        signature = space.edge_signatures[source]
        for destination, suffix_coefficient in suffix_images[source]:
            rows[destination][signature] += source_coefficient * suffix_coefficient
    for row in rows:
        fwht(row)
    return rows


def table_column(rows: Sequence[Sequence[int]], graph_mask: int) -> tuple[int, ...]:
    return tuple(row[graph_mask] for row in rows)


def table_parity_weights(
    space: FixedNumberSpace,
    rows: Sequence[Sequence[int]],
    graph_mask: int,
    selector: int,
    matching_layers: int,
) -> tuple[Fraction, Fraction]:
    even_numerator = 0
    odd_numerator = 0
    for row, factorial, parity_mask in zip(rows, space.factorials, space.parity_masks):
        coefficient = row[graph_mask]
        contribution = coefficient * coefficient * factorial
        if (selector & parity_mask).bit_count() & 1:
            odd_numerator += contribution
        else:
            even_numerator += contribution
    denominator = 1 << (matching_layers * space.particles)
    return Fraction(even_numerator, denominator), Fraction(odd_numerator, denominator)


def exhaustive_search(word_form: str) -> dict[str, object]:
    modes = 6
    particles = 3
    space = FixedNumberSpace(modes, particles)
    pairs_a = matching_a(modes)
    pairs_b = matching_b(modes)
    initial = space.basis_state(alternating_initial(modes))
    prefix = space.apply_matching(initial, pairs_a)
    prefix = space.apply_matching(prefix, pairs_b)

    if word_form == "TRANSFER_ABKAB":
        suffix = compose_images(
            len(space.basis),
            space.matching_images(pairs_a),
            space.matching_images(pairs_b),
        )
        public_word = ["A", "B", "K_E", "A", "B"]
    elif word_form == "ECHO_ABKBdAd":
        suffix = compose_images(
            len(space.basis),
            space.matching_images(pairs_b, adjoint=True),
            space.matching_images(pairs_a, adjoint=True),
        )
        public_word = ["A", "B", "K_E", "B_DAGGER", "A_DAGGER"]
    else:
        raise ValueError(word_form)

    rows = exhaustive_tables(space, prefix, suffix)
    graph_limit = 1 << len(space.complete_edges)
    full_selector = (1 << modes) - 1
    proper_selectors = tuple(range(1, full_selector))
    baseline_weights = {
        selector: table_parity_weights(space, rows, 0, selector, 4)
        for selector in proper_selectors
    }

    deterministic_pairs = 0
    graphs_with_deterministic = 0
    kerr_distinguishing_pairs = 0
    all_edge_state_causal_graphs = 0
    all_edge_boundary_causal_pairs = 0
    first_hit: dict[str, object] | None = None

    for graph_mask in range(1, graph_limit):
        support_parities = {
            parity_mask
            for row, parity_mask in zip(rows, space.parity_masks)
            if row[graph_mask]
        }
        if not support_parities:
            raise AssertionError("unitary output has empty support")
        reference_parity = next(iter(support_parities))
        differences = tuple(value ^ reference_parity for value in support_parities)
        valid_selectors = [
            selector
            for selector in proper_selectors
            if all(((selector & difference).bit_count() & 1) == 0 for difference in differences)
        ]
        if valid_selectors:
            graphs_with_deterministic += 1
            deterministic_pairs += len(valid_selectors)
            full_column = table_column(rows, graph_mask)
            state_causal = all(
                not projectively_equal(
                    full_column, table_column(rows, graph_mask ^ (1 << edge_index))
                )
                for edge_index in range(len(space.complete_edges))
                if graph_mask & (1 << edge_index)
            )
            if state_causal:
                all_edge_state_causal_graphs += 1
            for selector in valid_selectors:
                weights = table_parity_weights(space, rows, graph_mask, selector, 4)
                if weights != baseline_weights[selector]:
                    kerr_distinguishing_pairs += 1
                boundary_causal = all(
                    table_parity_weights(
                        space, rows, graph_mask ^ (1 << edge_index), selector, 4
                    )
                    != weights
                    for edge_index in range(len(space.complete_edges))
                    if graph_mask & (1 << edge_index)
                )
                if boundary_causal:
                    all_edge_boundary_causal_pairs += 1
                if first_hit is None:
                    first_hit = {
                        "edge_count": graph_mask.bit_count(),
                        "selector_modes": [
                            mode for mode in range(modes) if selector & (1 << mode)
                        ],
                        "even_weight": fraction_text(weights[0]),
                        "odd_weight": fraction_text(weights[1]),
                        "differs_from_kerr_disabled": weights
                        != baseline_weights[selector],
                    }

        # The excluded full-system selector is a required conserved sham.
        full_weights = table_parity_weights(space, rows, graph_mask, full_selector, 4)
        if full_weights != (Fraction(0), Fraction(1)):
            raise AssertionError("fixed N=3 full-system parity was not conserved")

    return {
        "word_form": word_form,
        "public_word": public_word,
        "modes": modes,
        "particles": particles,
        "complete_graph_kerr_edges": len(space.complete_edges),
        "nonempty_kerr_graphs": graph_limit - 1,
        "nonempty_proper_parity_selectors": len(proper_selectors),
        "graph_selector_pairs_checked": (graph_limit - 1) * len(proper_selectors),
        "deterministic_proper_selector_pairs": deterministic_pairs,
        "graphs_with_any_deterministic_proper_selector": graphs_with_deterministic,
        "kerr_distinguishing_deterministic_pairs": kerr_distinguishing_pairs,
        "all_edge_state_causal_deterministic_graphs": all_edge_state_causal_graphs,
        "all_edge_boundary_causal_deterministic_pairs": all_edge_boundary_causal_pairs,
        "first_hit": first_hit,
        "conserved_total_parity_sham": {
            "selector_modes": list(range(modes)),
            "checked_for_every_nonempty_kerr_graph": True,
            "even_weight": "0",
            "odd_weight": "1",
            "excluded_from_nontrivial_search": True,
        },
        "algorithm": {
            "kerr_graph_enumeration": "exact 15-bit Walsh-Hadamard transform",
            "selector_enumeration": "all nonempty proper six-mode subsets",
            "scientific_numeric_type": "integer coefficients and Fraction weights",
            "floating_point_decisions": 0,
            "generated_fwht_table_is_verifier_only": True,
            "generated_fwht_table_integer_cells": len(space.basis) * graph_limit,
            "generated_fwht_table_rows": len(space.basis),
            "generated_fwht_table_columns": graph_limit,
            "generated_fwht_table_peak_integer_signed_bits": max(
                signed_bits(value) for row in rows for value in row
            ),
            "fwht_integer_add_subtract_operations": (
                len(space.basis) * graph_limit * len(space.complete_edges)
            ),
            "generated_table_is_not_an_accepted_carrier_or_compiler": True,
        },
    }


def fixed_core_control(modes: int) -> dict[str, object]:
    particles = modes // 2
    space = FixedNumberSpace(modes, particles)
    pairs_a = matching_a(modes)
    kerr_edges = tuple(edge for edge in ((1, 2), (3, 4)) if edge[1] < modes)
    selector_modes = tuple(mode for mode in (0, 4) if mode < modes)
    selector = sum(1 << mode for mode in selector_modes)
    initial = space.basis_state(alternating_initial(modes))
    before_kerr = space.apply_matching(initial, pairs_a)
    final = space.apply_kerr_mask(before_kerr, space.edge_mask(kerr_edges))
    final = space.apply_matching(final, pairs_a, adjoint=True)
    disabled = space.apply_matching(before_kerr, pairs_a, adjoint=True)
    enabled_weights = space.parity_weights(final, selector)
    restored = space.apply_matching(final, pairs_a)
    restored = space.apply_kerr_mask(restored, space.edge_mask(kerr_edges))
    restored = space.apply_matching(restored, pairs_a, adjoint=True)

    edge_controls = []
    for edge in kerr_edges:
        ablated_edges = tuple(candidate for candidate in kerr_edges if candidate != edge)
        ablated = space.apply_kerr_mask(before_kerr, space.edge_mask(ablated_edges))
        ablated = space.apply_matching(ablated, pairs_a, adjoint=True)
        edge_controls.append(
            {
                "edge": list(edge),
                "changes_projective_state": not projectively_equal(
                    final.coefficients, ablated.coefficients
                ),
                "changes_declared_boundary": space.parity_weights(ablated, selector)
                != enabled_weights,
            }
        )

    geometry_edges = tuple(pairs_a) + kerr_edges
    components = component_sizes(modes, geometry_edges)
    return {
        "modes": modes,
        "particles": particles,
        "public_word": ["A", "K_FIXED_CORE", "A_DAGGER"],
        "kerr_edges": [list(edge) for edge in kerr_edges],
        "selector_modes": list(selector_modes),
        "geometry_component_sizes": components,
        "geometry_connected": components == [modes],
        "final": {
            **state_record(space, final),
            "boundary": parity_record(space, final, selector),
        },
        "kerr_disabled_boundary": parity_record(space, disabled, selector),
        "per_edge_causality": edge_controls,
        "all_edges_change_projective_state": all(
            control["changes_projective_state"] for control in edge_controls
        ),
        "all_edges_change_declared_boundary": all(
            control["changes_declared_boundary"] for control in edge_controls
        ),
        "public_word_exact_result_free_restore": space.exact_initial_restored(
            restored, alternating_initial(modes)
        ),
        "bounded_active_core_certificate": modes > 6 and components != [modes],
    }


def build_result() -> dict[str, object]:
    primary = [execute_primary(modes) for modes in (4, 6, 8)]
    exhaustive = [
        exhaustive_search("TRANSFER_ABKAB"),
        exhaustive_search("ECHO_ABKBdAd"),
    ]
    fixed_core = [fixed_core_control(modes) for modes in (6, 8)]

    zero_causal_boundary_hit = all(
        result["kerr_distinguishing_deterministic_pairs"] == 0
        for result in exhaustive
    )
    final_rank_growth = [result["final"]["central_schmidt_rank"] for result in primary]
    maximum_rank_growth = [
        max(stage["central_schmidt_rank"] for stage in result["stages"])
        for result in primary
    ]
    qnd_factorized = [result["final"]["last_mode_parity"]["factorized"] for result in primary]

    return {
        "schema": "phase_qemu_v2_growing_even_mode_qnd_bond_diagnostic_v1",
        "milestone": "M260",
        "experiment": "GROWING_EVEN_MODE_ALTERNATING_MATCHING_CROSS_KERR_PARITY_QND_TRANSFER_AND_BOND_GROWTH_DIAGNOSTIC",
        "claim": (
            "EXACT_GROWING_EVEN_MODE_FIXED_NUMBER_ALTERNATING_MATCHING_PI_CROSS_KERR_"
            "DIAGNOSTIC_HAS_SECTOR_DIMENSIONS10_56_330_AND_PEAK_CENTRAL_SCHMIDT_RANKS2_8_18_"
            "BUT_PRIMARY_QND_PARITY_FACTORIZATION_FAILS_AT_N6_N8_EXHAUSTIVE_N6_ALL_EDGESET_"
            "PROPER_SELECTOR_SEARCH_FINDS_ZERO_KERR_DISTINGUISHING_DETERMINISTIC_BOUNDARIES_"
            "AND_THE_DECLARED_DETERMINISTIC_N8_FIXED_CORE_ECHO_CONTROL_IS_DISCONNECTED_WITH_"
            "FUNCTIONAL_EXACT_PUBLIC_ADJOINT_RESTORATION_AND_NO_ADVANTAGE"
        ),
        "claim_ceiling": (
            "EXACT_SOFTWARE_FIXED_NUMBER_BOSONIC_ALTERNATING_MATCHING_SINGLE_PI_CROSS_KERR_"
            "QND_PARITY_DIAGNOSTIC_AT_N4_N6_N8_ONLY"
        ),
        "exactness": {
            "state_representation": "integer homogeneous-polynomial coefficients",
            "probabilities": "exact dyadic Fraction",
            "schmidt_rank": "exact rational matrix rank after invertible Fock normalization factors are removed",
            "floating_point_scientific_decisions": 0,
        },
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "restoration_scope": (
            "FUNCTIONAL_NORMALIZED_EXACT_STATE_EQUALITY_AND_RETURNED_VALUE_REUSE_"
            "WITHOUT_SAME_BACKING"
        ),
        "package_verification_level": "PACKAGE_SELF_REVIEW",
        "primary_family": primary,
        "n6_exhaustive_searches": exhaustive,
        "fixed_core_controls": fixed_core,
        "observed_disposition": {
            "classification": (
                "DETERMINISTIC_QND_PARITY_CLOSURE_AND_CONNECTED_BOND_GROWTH_DO_NOT_COEXIST_"
                "IN_THE_TESTED_PI_CROSS_KERR_ALTERNATING_MATCHING_FAMILY"
                if zero_causal_boundary_hit
                else "INCONCLUSIVE_NONZERO_CAUSAL_DETERMINISTIC_BOUNDARY_HITS_REQUIRE_REVIEW"
            ),
            "primary_final_central_schmidt_ranks_n4_n6_n8": final_rank_growth,
            "primary_maximum_central_schmidt_ranks_n4_n6_n8": maximum_rank_growth,
            "primary_last_mode_qnd_factorized_n4_n6_n8": qnd_factorized,
            "exact_exhaustive_zero_kerr_distinguishing_deterministic_boundary_hit": (
                zero_causal_boundary_hit
            ),
            "raw_echo_deterministic_parities_are_not_promoted": True,
            "scope": (
                "declared n=4,6,8 primary descriptor; exhaustive n=6 single-pi-Kerr graph search "
                "for two declared word forms; declared n=6,8 fixed-core controls"
            ),
        },
        "strongest_honest_comparator": {
            "required": [
                "exact sparse fixed-number propagation",
                "adaptive U(1)-symmetric MPS using measured minimal bond dimensions",
                "TTN or boundary-only contraction of the requested parity projector",
                "fixed-number linear-optical recurrence for the Kerr-disabled path",
                "period-two, reflection, translation, involution, and finite-depth symmetry reductions",
                "O(1) active-component certificate when causal support remains bounded",
            ],
            "dense_sector_only_comparison_forbidden": True,
            "constant_depth_growth_is_not_an_asymptotic_separation": True,
            "adaptive_tensor_network_comparator_implemented": False,
            "comparator_optimality_established": False,
            "resource_advantage_comparison_authorized": False,
        },
        "resource_accounting": {
            "primary_resident_integer_coefficient_cells_n4_n6_n8": [10, 56, 330],
            "exhaustive_n6_generated_fwht_integer_cells_per_word": 1835008,
            "exhaustive_n6_generated_fwht_tables_sequential_not_simultaneous": True,
            "matching_transition_cache_and_compiler_entries_instrumented": False,
            "python_object_allocator_hashing_and_serialization_accounted": False,
            "whole_process_live_payload_peak_complete": False,
            "physical_energy_noise_precision_bandwidth_and_latency_modeled": False,
            "resource_verification_level": "PACKAGE_SELF_REVIEW",
        },
        "strict_promote_rule": [
            "one formula-generated public descriptor works at n=4,6,8 without answer search",
            "all-mode causal interaction geometry is connected",
            "every declared Kerr edge changes the declared boundary under ablation",
            "a nontrivial parity is exactly deterministic and differs from the Kerr-disabled path",
            "QND copy, unlatch, public-adjoint restoration, and reuse are exact",
            "minimal MPS or TTN bond rank grows with no bounded-component or symmetry certificate",
        ],
        "strict_kill_rule": [
            "deterministic parity requires n-specific answer-bearing selector search",
            "connectedness depends on inactive edges",
            "the causal active component remains bounded",
            "parity becomes mixed precisely when Schmidt rank grows",
            "the strongest adaptive boundary-only comparator remains bounded or compact",
        ],
        "claim_ceilings": [
            "software exact-arithmetic diagnostic only",
            "no physical Phase-QEMU execution",
            "no QEMU or CATVM custody enforcement",
            "no same-backing restoration or reuse claim",
            "no phase-native resource advantage",
            "no complexity lower bound",
            "no general cross-Kerr no-go",
            "no Small Wall crossing",
            "no physical-bit replacement",
            "no unbounded catalytic computation",
        ],
        "next_materially_different_backend_if_killed": (
            "CONTROLLED_MANY_BODY_EIGENPHASE_HOLONOMY_SCATTERING_PHASE_QEMU_BACKEND"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional deterministic JSON output path")
    arguments = parser.parse_args()
    payload = json.dumps(build_result(), indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
