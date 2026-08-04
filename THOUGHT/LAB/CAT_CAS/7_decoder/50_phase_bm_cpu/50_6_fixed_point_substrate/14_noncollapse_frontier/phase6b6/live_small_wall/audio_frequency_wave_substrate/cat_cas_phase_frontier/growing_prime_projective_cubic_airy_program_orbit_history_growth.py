#!/usr/bin/env python3
"""Exact projective collision search for deeper cubic Airy public words.

M177 rejected smaller regular linear quotients at one two-shear layer.  This
successor tests the most direct nonlinear repair: canonicalize the complete
joint state up to a global field scalar and merge equal projective states.
The traversal applies every public coefficient pair in place and executes the
actual inverse after each branch, so it needs one dense carrier rather than a
retained tree of amplitude states.  The exact canonical-code set is an honest
diagnostic resource and has an identical classical implementation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import growing_prime_resident_cubic_strength_phase_port_weil_action_rank as resident


CLAIM = (
    "BOUNDED_EXACT_PRIMARY_TWO_SHEAR_CUBIC_AIRY_PROGRAM_WORD_PROJECTIVE_"
    "CANONICALIZATION_FIRST_COLLIDES_AT_Q5_DEPTH4_AFTER_COLLISION_FREE_DEPTHS1_"
    "THROUGH3_IN625_RAW_EQUALITY_CLASSES_OF_SIZE5_ON_A_RANK5_PROGRAM_SUBSPACE_"
    "BUT_THE_COLLISION_DIRECTION_FAILS_Q11_AND_Q23_TRANSFER_WHILE_Q11_DEPTHS1_"
    "AND2_REMAIN_COLLISION_FREE_WITH_EXACT_RESTORATION_AND_REUSE_SO_NO_FIXED_"
    "HISTORY_FREE_CLOSURE_OR_ADVANTAGE_IS_ESTABLISHED_AND_THE_IDENTICAL_"
    "CLASSICAL_FACTOR_GRAPH_REMAINS"
)
CASES = ((5, 1), (5, 2), (5, 3), (5, 4), (11, 1), (11, 2))


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class TraversalWork:
    forward_branches: int = 0
    inverse_branches: int = 0
    controlled_field_multiplications: int = 0
    gaussian_field_multiply_adds: int = 0
    exact_restoration_comparisons: int = 0
    leaf_projective_canonicalizations: int = 0
    exact_projective_code_entries_peak: int = 0
    exact_projective_code_payload_bits_sum: int = 0
    exact_projective_code_max_bits: int = 0
    temporary_axis_vector_field_cells_peak: int = 0
    full_history_state_tree_materialized: bool = False


@dataclass(frozen=True)
class LayerChart:
    data_forward: tuple[tuple[int, ...], ...]
    data_inverse: tuple[tuple[int, ...], ...]
    latent_forward: tuple[tuple[int, ...], ...]
    latent_inverse: tuple[tuple[int, ...], ...]


def gaussian_matrix(
    field: resident.open_action.cubic.gaussian.Field,
    payload: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    coefficient = payload[4] % field.p
    symplectic = list(payload[:4])
    return tuple(
        tuple(
            coefficient
            * resident.open_action.cubic.gaussian.kernel_value(
                symplectic, target, source, field
            )
            % field.p
            for source in range(field.q)
        )
        for target in range(field.q)
    )


def layer_payloads(q: int, layer: int, family: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    code = 1 if family == "PRIMARY" else 2
    parameter = (layer + code) % q or 1
    return (
        (1, 1, parameter, parameter + 1, 1),
        (parameter + 1, 1, parameter, 1, 1),
    )


def inverse_payload(
    field: resident.open_action.cubic.gaussian.Field,
    payload: tuple[int, ...],
) -> tuple[int, ...]:
    return resident.open_action.gaussian_payload_inverse(
        field,
        payload,
        resident.open_action.Work(),
    )


def layer_chart(
    field: resident.open_action.cubic.gaussian.Field,
    layer: int,
    family: str,
) -> LayerChart:
    data, latent = layer_payloads(field.q, layer, family)
    return LayerChart(
        gaussian_matrix(field, data),
        gaussian_matrix(field, inverse_payload(field, data)),
        gaussian_matrix(field, latent),
        gaussian_matrix(field, inverse_payload(field, latent)),
    )


def controlled(
    carrier: resident.Carrier,
    coefficient: int,
    work: TraversalWork,
) -> None:
    q, p = carrier.q, carrier.field.p
    for strength in range(q):
        for coordinate in range(q):
            multiplier = resident.phase(carrier.field, coefficient * strength * coordinate**3)
            for fiber in range(2):
                index = resident.offset(q, strength, fiber, coordinate)
                carrier.cells[index] = carrier.cells[index] * multiplier % p
                work.controlled_field_multiplications += 1


def apply_axis_matrix(
    carrier: resident.Carrier,
    matrix: tuple[tuple[int, ...], ...],
    axis: str,
    work: TraversalWork,
) -> None:
    q, p = carrier.q, carrier.field.p
    work.temporary_axis_vector_field_cells_peak = max(
        work.temporary_axis_vector_field_cells_peak, q
    )
    if axis == "DATA":
        for strength in range(q):
            for fiber in range(2):
                start = resident.offset(q, strength, fiber, 0)
                source = carrier.cells[start : start + q]
                carrier.cells[start : start + q] = [
                    sum(left * right for left, right in zip(row, source)) % p
                    for row in matrix
                ]
                work.gaussian_field_multiply_adds += q * q
        return
    if axis == "LATENT":
        for fiber in range(2):
            for coordinate in range(q):
                source = [
                    carrier.cells[resident.offset(q, strength, fiber, coordinate)]
                    for strength in range(q)
                ]
                output = [
                    sum(left * right for left, right in zip(row, source)) % p
                    for row in matrix
                ]
                for strength, value in enumerate(output):
                    carrier.cells[resident.offset(q, strength, fiber, coordinate)] = value
                work.gaussian_field_multiply_adds += q * q
        return
    fail("invalid axis")


def apply_layer(
    carrier: resident.Carrier,
    chart: LayerChart,
    first: int,
    second: int,
    family: str,
    work: TraversalWork,
) -> None:
    controlled(carrier, first, work)
    if family == "PRIMARY":
        apply_axis_matrix(carrier, chart.data_forward, "DATA", work)
        controlled(carrier, second, work)
        apply_axis_matrix(carrier, chart.latent_forward, "LATENT", work)
    else:
        apply_axis_matrix(carrier, chart.latent_forward, "LATENT", work)
        controlled(carrier, second, work)
        apply_axis_matrix(carrier, chart.data_forward, "DATA", work)
    work.forward_branches += 1


def inverse_layer(
    carrier: resident.Carrier,
    chart: LayerChart,
    first: int,
    second: int,
    family: str,
    work: TraversalWork,
    mutation: str | None = None,
) -> None:
    inverse_first = (-first + int(mutation == "WRONG")) % carrier.q
    inverse_second = -second % carrier.q
    if family == "PRIMARY":
        if mutation == "REORDER":
            controlled(carrier, inverse_first, work)
        apply_axis_matrix(carrier, chart.latent_inverse, "LATENT", work)
        controlled(carrier, inverse_second, work)
        apply_axis_matrix(carrier, chart.data_inverse, "DATA", work)
        if mutation not in ("MISSING", "REORDER"):
            controlled(carrier, inverse_first, work)
    else:
        if mutation == "REORDER":
            controlled(carrier, inverse_first, work)
        apply_axis_matrix(carrier, chart.data_inverse, "DATA", work)
        controlled(carrier, inverse_second, work)
        apply_axis_matrix(carrier, chart.latent_inverse, "LATENT", work)
        if mutation not in ("MISSING", "REORDER"):
            controlled(carrier, inverse_first, work)
    work.inverse_branches += 1


def raw_code(values: list[int], base: int) -> int:
    code = 0
    for value in reversed(values):
        code = code * base + value
    return code


def projective_code(values: list[int], p: int) -> tuple[int, int]:
    pivot = next((value for value in values if value % p), None)
    if pivot is None:
        fail("zero state has no projective code")
    inverse = pow(pivot, -1, p)
    normalized = [(value * inverse) % p for value in values]
    return raw_code(normalized, p), pivot % p


def row_reduce(rows: list[list[int]], modulus: int) -> tuple[list[list[int]], list[int]]:
    reduced = [[value % modulus for value in row] for row in rows if any(row)]
    pivots: list[int] = []
    pivot_row = 0
    if not reduced:
        return [], pivots
    for column in range(len(reduced[0])):
        candidate = next(
            (index for index in range(pivot_row, len(reduced)) if reduced[index][column]),
            None,
        )
        if candidate is None:
            continue
        reduced[pivot_row], reduced[candidate] = reduced[candidate], reduced[pivot_row]
        inverse = pow(reduced[pivot_row][column], -1, modulus)
        reduced[pivot_row] = [value * inverse % modulus for value in reduced[pivot_row]]
        for index, row in enumerate(reduced):
            if index == pivot_row or not row[column]:
                continue
            factor = row[column]
            reduced[index] = [
                (left - factor * right) % modulus
                for left, right in zip(row, reduced[pivot_row])
            ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(reduced):
            break
    return reduced[:pivot_row], pivots


def nullspace_basis(
    rows: list[list[int]], modulus: int, columns: int
) -> list[list[int]]:
    reduced, pivots = row_reduce(rows, modulus)
    free_columns = [column for column in range(columns) if column not in pivots]
    basis: list[list[int]] = []
    for free in free_columns:
        vector = [0] * columns
        vector[free] = 1
        for row, pivot in zip(reduced, pivots):
            vector[pivot] = -row[free] % modulus
        basis.append(vector)
    return basis


def normalized_line_vector(vector: tuple[int, ...], modulus: int) -> tuple[int, ...]:
    pivot = next(value for value in vector if value)
    inverse = pow(pivot, -1, modulus)
    return tuple(value * inverse % modulus for value in vector)


def traverse_projective_orbit(
    carrier: resident.Carrier,
    depth: int,
    family: str,
) -> tuple[dict[str, Any], TraversalWork]:
    if depth < 1:
        fail("depth must be positive")
    charts = [layer_chart(carrier.field, layer, family) for layer in range(depth)]
    first_history_and_pivot_by_code: dict[int, int] = {}
    collision_count_by_code: dict[int, int] = {}
    colliding_histories: set[int] = set()
    collision_difference_lines: set[tuple[int, ...]] = set()
    collision_raw_equalities = 0
    collision_witnesses: list[dict[str, Any]] = []
    work = TraversalWork()

    def decode_history(encoded: int) -> list[tuple[int, int]]:
        pairs = []
        for _ in range(depth):
            second = encoded % carrier.q
            encoded //= carrier.q
            first = encoded % carrier.q
            encoded //= carrier.q
            pairs.append((first, second))
        return list(reversed(pairs))

    def visit(layer: int, history: int) -> None:
        nonlocal collision_raw_equalities
        if layer == depth:
            code, pivot = projective_code(carrier.cells, carrier.field.p)
            if code in first_history_and_pivot_by_code:
                first_history, first_pivot = divmod(
                    first_history_and_pivot_by_code[code], carrier.field.p
                )
                collision_count_by_code[code] = collision_count_by_code.get(code, 0) + 1
                first_pairs = decode_history(first_history)
                second_pairs = decode_history(history)
                scalar = pivot * pow(first_pivot, -1, carrier.field.p) % carrier.field.p
                if scalar == 1:
                    collision_raw_equalities += 1
                colliding_histories.update((first_history, history))
                difference = tuple(
                    (right - left) % carrier.q
                    for first_pair, second_pair in zip(first_pairs, second_pairs)
                    for left, right in zip(first_pair, second_pair)
                )
                collision_difference_lines.add(
                    normalized_line_vector(difference, carrier.q)
                )
                if len(collision_witnesses) < 16:
                    collision_witnesses.append(
                        {
                            "first_history": first_pairs,
                            "second_history": second_pairs,
                            "second_state_global_scalar_over_first": scalar,
                        }
                    )
            else:
                first_history_and_pivot_by_code[code] = history * carrier.field.p + pivot
            work.leaf_projective_canonicalizations += 1
            bits = code.bit_length()
            work.exact_projective_code_payload_bits_sum += bits
            work.exact_projective_code_max_bits = max(
                work.exact_projective_code_max_bits, bits
            )
            work.exact_projective_code_entries_peak = max(
                work.exact_projective_code_entries_peak,
                len(first_history_and_pivot_by_code),
            )
            return
        chart = charts[layer]
        for first in range(carrier.q):
            for second in range(carrier.q):
                before = raw_code(carrier.cells, carrier.field.p)
                apply_layer(carrier, chart, first, second, family, work)
                visit(layer + 1, (history * carrier.q + first) * carrier.q + second)
                inverse_layer(carrier, chart, first, second, family, work)
                work.exact_restoration_comparisons += 1
                if raw_code(carrier.cells, carrier.field.p) != before:
                    fail("in-place branch inverse failed exact restoration")

    initial = raw_code(carrier.cells, carrier.field.p)
    visit(0, 0)
    expected = carrier.q ** (2 * depth)
    if raw_code(carrier.cells, carrier.field.p) != initial:
        fail("projective orbit root restoration failed")
    distinct = len(first_history_and_pivot_by_code)
    collision_class_histogram: dict[str, int] = {}
    for collision_count in collision_count_by_code.values():
        class_size = collision_count + 1
        key = str(class_size)
        collision_class_histogram[key] = collision_class_histogram.get(key, 0) + 1
    singleton_classes = distinct - len(collision_count_by_code)
    if singleton_classes:
        collision_class_histogram["1"] = singleton_classes
    colliding_vectors = [
        [value for pair in decode_history(encoded) for value in pair]
        for encoded in colliding_histories
    ]
    colliding_basis, _ = row_reduce(colliding_vectors, carrier.q)
    colliding_constraints = nullspace_basis(
        colliding_basis, carrier.q, 2 * depth
    )
    return {
        "q": carrier.q,
        "p": carrier.field.p,
        "depth": depth,
        "program_family": family,
        "public_program_histories": expected,
        "distinct_projective_full_states": distinct,
        "projective_collisions": expected - distinct,
        "non_singleton_projective_classes": len(collision_count_by_code),
        "projective_collision_class_size_histogram": collision_class_histogram,
        "maximum_projective_collision_class_size": max(
            (count + 1 for count in collision_count_by_code.values()), default=1
        ),
        "histories_in_non_singleton_projective_classes": sum(
            count + 1 for count in collision_count_by_code.values()
        ),
        "collision_relation_diagnostic": {
            "collision_events_total": expected - distinct,
            "all_collisions_are_raw_state_equal_not_only_projectively_equal": (
                collision_raw_equalities == expected - distinct
            ),
            "colliding_history_vectors": len(colliding_histories),
            "colliding_history_span_rank": len(colliding_basis),
            "colliding_histories_form_complete_linear_subspace": bool(
                colliding_histories
            )
            and len(colliding_histories) == carrier.q ** len(colliding_basis),
            "colliding_history_subspace_rref_basis": colliding_basis,
            "colliding_history_subspace_constraint_basis": colliding_constraints,
            "normalized_collision_difference_line_generators": sorted(
                collision_difference_lines
            ),
            "single_collision_difference_line": len(collision_difference_lines) == 1,
        },
        "collision_witnesses": collision_witnesses,
        "minimum_exact_projective_state_identifier_bits": (distinct - 1).bit_length(),
        "minimum_exact_public_history_identifier_bits": (expected - 1).bit_length(),
        "public_coefficient_history_field_elements": 2 * depth,
        "resident_dense_state_field_elements": 2 * carrier.q**2,
        "retained_dense_history_state_field_elements": 0,
        "global_scalar_ledger_field_elements_for_lawful_projective_state": 1,
        "exact_projective_code_is_collision_free_base_p_encoding": True,
        "cryptographic_hash_used_as_equality_oracle": False,
        "assignment_or_state_tree_materialized": False,
        "work": work.__dict__,
    }, work


def transaction(q: int, depth: int) -> dict[str, Any]:
    carrier = resident.Carrier.seal(q, "PRIMARY")
    initial = raw_code(carrier.cells, carrier.field.p)
    backing = (id(carrier), id(carrier.cells))
    carrier.stage = "PROJECTIVE_TRAVERSAL"
    result, _ = traverse_projective_orbit(carrier, depth, "PRIMARY")
    carrier.stage = "IDLE"
    restored = raw_code(carrier.cells, carrier.field.p) == initial
    same_backing = (id(carrier), id(carrier.cells)) == backing
    if not restored or not same_backing:
        fail("transaction restoration failed")
    carrier.restoration_generation += 1
    return {
        **result,
        "exact_dense_carrier_restored": restored,
        "same_backing_restored": same_backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
    }


def inverse_mutation_fails(mutation: str) -> bool:
    carrier = resident.Carrier.seal(5, "PRIMARY")
    initial = raw_code(carrier.cells, carrier.field.p)
    chart = layer_chart(carrier.field, 0, "PRIMARY")
    apply_layer(carrier, chart, 1, 2, "PRIMARY", TraversalWork())
    inverse_layer(
        carrier,
        chart,
        1,
        2,
        "PRIMARY",
        TraversalWork(),
        mutation,
    )
    return raw_code(carrier.cells, carrier.field.p) != initial


def state_for_history(q: int, history: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    carrier = resident.Carrier.seal(q, "PRIMARY")
    work = TraversalWork()
    for layer, (first, second) in enumerate(history):
        apply_layer(
            carrier,
            layer_chart(carrier.field, layer, "PRIMARY"),
            first % q,
            second % q,
            "PRIMARY",
            work,
        )
    return tuple(carrier.cells)


def collision_direction_comparisons(q: int) -> list[bool]:
    comparisons = []
    for first0, second0, first1, total, shift in (
        (0, 0, 0, 1, 1),
        (1, 2, 3, 4, 2),
        (2, 4, 1, 0, 3),
    ):
        left = (
            (first0, second0),
            (first1, total),
            (0, 0),
            (0, 0),
        )
        right = (
            (first0, second0),
            (first1, total - shift),
            (0, 0),
            (0, shift),
        )
        comparisons.append(state_for_history(q, left) == state_for_history(q, right))
    return comparisons


def controls() -> dict[str, bool]:
    original = resident.Carrier.seal(5, "PRIMARY")
    chart = layer_chart(original.field, 0, "PRIMARY")
    left = resident.Carrier.seal(5, "PRIMARY")
    right = resident.Carrier.seal(5, "PRIMARY")
    apply_layer(left, chart, 1, 1, "PRIMARY", TraversalWork())
    apply_layer(right, chart, 1, 2, "PRIMARY", TraversalWork())
    left_code, _ = projective_code(left.cells, left.field.p)
    right_code, _ = projective_code(right.cells, right.field.p)

    scaled = list(left.cells)
    for index, value in enumerate(scaled):
        scaled[index] = 2 * value % left.field.p
    scaled_code, _ = projective_code(scaled, left.field.p)
    q5_transfer = collision_direction_comparisons(5)
    q11_transfer = collision_direction_comparisons(11)
    q23_transfer = collision_direction_comparisons(23)
    return {
        "missing_inverse_fails": inverse_mutation_fails("MISSING"),
        "wrong_inverse_fails": inverse_mutation_fails("WRONG"),
        "reordered_inverse_fails": inverse_mutation_fails("REORDER"),
        "second_public_coefficient_mutation_changes_projective_state": left_code != right_code,
        "global_scalar_mutation_preserves_projective_code": scaled_code == left_code,
        "global_scalar_ledger_remains_required": raw_code(
            scaled, left.field.p
        )
        != raw_code(left.cells, left.field.p),
        "projective_code_uses_exact_base_p_integer_not_hash": True,
        "q5_depth4_collision_direction_reproduces_on_three_exact_samples": all(
            q5_transfer
        ),
        "q5_depth4_collision_direction_fails_q11_transfer_on_all_three_samples": not any(
            q11_transfer
        ),
        "q5_depth4_collision_direction_fails_q23_transfer_on_all_three_samples": not any(
            q23_transfer
        ),
    }


def reuse() -> dict[str, Any]:
    carrier = resident.Carrier.seal(5, "PRIMARY")
    initial = raw_code(carrier.cells, carrier.field.p)
    backing = (id(carrier), id(carrier.cells))
    carrier.stage = "PROJECTIVE_TRAVERSAL"
    first, _ = traverse_projective_orbit(carrier, 2, "PRIMARY")
    carrier.stage = "IDLE"
    carrier.restoration_generation += 1

    chart = layer_chart(carrier.field, 0, "ALTERNATE")
    apply_layer(carrier, chart, 2, 3, "ALTERNATE", TraversalWork())
    carrier.stage = "FORWARD_COMPLETE"
    second_boundary = resident.boundary(carrier, "PRIMARY")
    carrier.stage = "IDLE"
    inverse_layer(carrier, chart, 2, 3, "ALTERNATE", TraversalWork())
    carrier.restoration_generation += 1

    fresh = resident.Carrier.seal(5, "PRIMARY")
    apply_layer(fresh, chart, 2, 3, "ALTERNATE", TraversalWork())
    fresh.stage = "FORWARD_COMPLETE"
    fresh_boundary = resident.boundary(fresh, "PRIMARY")
    return {
        "first_projective_orbit_size": first["distinct_projective_full_states"],
        "second_unrelated_alternate_boundary": second_boundary,
        "fresh_alternate_boundary": fresh_boundary,
        "second_matches_fresh": second_boundary == fresh_boundary,
        "exact_payload_restored_after_reuse": raw_code(carrier.cells, carrier.field.p) == initial,
        "same_backing_reused": (id(carrier), id(carrier.cells)) == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
    }


def run() -> dict[str, Any]:
    cases = [transaction(q, depth) for q, depth in CASES]
    control_result, reuse_result = controls(), reuse()
    if not all(control_result.values()) or not all(
        (
            reuse_result["second_matches_fresh"],
            reuse_result["exact_payload_restored_after_reuse"],
            reuse_result["same_backing_reused"],
            reuse_result["restoration_generation"] == 2,
            not reuse_result["snapshot_used"],
        )
    ):
        fail("controls or reuse failed")
    return {
        "schema": "CAT_CAS_GROWING_PRIME_PROJECTIVE_CUBIC_AIRY_PROGRAM_ORBIT_HISTORY_GROWTH_V1",
        "claim_candidate": CLAIM,
        "claim_ceiling": "PRIMARY_TWO_SHEAR_PUBLIC_COEFFICIENT_WORDS_Q5_DEPTHS1_THROUGH4_Q11_DEPTHS1_AND2_FIXED_DECLARED_GAUSSIAN_LAYER_SEQUENCE_TWO_FIBER_DENSE_FULL_STATE_PROJECTIVE_CANONICALIZATION_DIRECT_PROCESS_SOFTWARE",
        "cases": cases,
        "controls": control_result,
        "restoration_and_reuse": reuse_result,
        "observed_resource_law": {
            "tested_projective_orbit_entries": "Q^(2*DEPTH)",
            "minimum_exact_horizon_identifier_bits": "CEIL_LOG2_Q^(2*DEPTH)",
            "resident_dense_state_field_elements": "2*Q^2",
            "compact_factor_graph_public_history_field_elements": "2*DEPTH",
            "retained_dense_history_state_field_elements": 0,
            "exact_code_set_entries": "Q^(2*DEPTH)",
            "fixed_history_free_projective_closure_established": False,
        },
        "matched_baseline": {
            "identical_classical_in_place_dense_traversal": True,
            "identical_classical_factor_graph_with_2d_public_coefficients": True,
            "factor_graph_rematerialization_avoids_dense_history_tree": True,
            "cold_start_comparison_used": False,
        },
        "resource_accounting": {
            "dense_carrier_axis_temporaries_public_branches_inverse_work_exact_code_set_restoration_and_reuse_counted": True,
            "python_set_and_integer_object_headers_allocator_interpreter_and_whole_process_peak_excluded": True,
            "diagnostic_enumerates_all_declared_public_program_histories": True,
            "advantage_claimed": False,
        },
        "claim_boundaries": {
            "arbitrary_q_or_depth": False,
            "all_nonlinear_or_singular_quotients_rejected": False,
            "fixed_history_free_nonlinear_closure": False,
            "universal_state_or_information_lower_bound": False,
            "polynomial_work_subquadratic_state_closure": False,
            "machine_enforced_hidden_intermediates": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage_or_small_wall_crossing": False,
            "physical_waveform_or_silicon_execution": False,
            "replacement_of_physical_bits_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
        "next_obstruction": "THE_FIRST_Q5_PROJECTIVE_COLLISIONS_ARE_A_Q5_LOCAL_RAW_EQUALITY_DIRECTION_ON_A_RANK5_DEPTH4_PROGRAM_SUBSPACE_THAT_DOES_NOT_REDUCE_FIXED_HORIZON_IDENTIFIER_BIT_WIDTH_AND_FAILS_Q11_AND_Q23_TRANSFER_WHILE_THE_IDENTICAL_FACTOR_GRAPH_STORES_ONLY2D_PUBLIC_COEFFICIENTS_SO_THE_NEXT_PHASE_REPAIR_MUST_CHANGE_THE_NATIVE_COUPLING_LAW_OR_FIND_A_TRANSFERABLE_SEMANTIC_QUOTIENT_WITHOUT_MOVING_HISTORY_INTO_A_CODE_SET",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(run(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
