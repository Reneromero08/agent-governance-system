#!/usr/bin/env python3
"""Exact open-interface action bundle for the cubic-Weil program family.

This diagnostic replaces the quadratic Weyl-component operator carrier with
actual source-state columns acted on factorwise by the public cubic/Weil word.
It tests the boundary between the compact single-source q-vector recurrence
and a resident many-source open interface.  It does not claim CATVM custody or
a resource unavailable to the identical classical vector recurrence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import growing_prime_two_fiber_cubic_deformed_weil_component_rank as cubic


ORDERS = (5, 11, 23, 29, 41, 53, 83, 89, 113)
SOURCE_RANK_CAP = 22
BASE_DEPTH = 2
DEEPEST_ORDER = 113
DEEPEST_DEPTH = 4
ALTERNATE_ORDER = 41
ALTERNATE_DEPTH = 3
FAMILIES = ("PRIMARY", "ALTERNATE")
CLAIM = (
    "BOUNDED_EXACT_GROWING_SAFE_PRIME_CUBIC_WEIL_OPEN_INTERFACE_PHASE_STATE_"
    "BUNDLE_EXECUTES_FACTORWISE_WITH_ACTION_SPAN_EQUAL_TO_EACH_DECLARED_SOURCE_"
    "RANK_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE_BUT_THE_FULL_TWO_"
    "FIBER_BASIS_ACTION_BUNDLE_USES4Q2_EXPLICIT_FIELD_CELLS_WHILE_AN_EXECUTED_"
    "SOURCE_STREAMED_Q_VECTOR_CLASSICAL_RECURRENCE_USES2Q_DYNAMIC_CELLS"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass(frozen=True)
class Operation:
    kind: str
    payload: tuple[int, ...]


@dataclass
class Work:
    gaussian_kernel_phase_evaluations: int = 0
    gaussian_field_multiply_adds: int = 0
    cubic_phase_evaluations: int = 0
    cubic_field_multiplications: int = 0
    fiber_field_multiply_adds: int = 0
    inverse_gaussian_scalar_rematerializations: int = 0
    temporary_vector_field_cells_peak: int = 0


@dataclass
class Carrier:
    field: cubic.gaussian.Field
    source_rank: int
    cells: list[int]
    stage: str = "IDLE"
    restoration_generation: int = 0

    @classmethod
    def seal(cls, q: int, source_rank: int) -> "Carrier":
        field = cubic.gaussian.make_field(q)
        dimension = 2 * q
        if not 1 <= source_rank <= dimension:
            fail("invalid source rank")
        cells = [0] * (dimension * source_rank)
        for column in range(source_rank):
            cells[column * source_rank + column] = 1
        return cls(field, source_rank, cells)

    @property
    def dimension(self) -> int:
        return 2 * self.field.q

    def canonical_state(self) -> tuple[Any, ...]:
        return self.field.q, self.field.p, self.source_rank, tuple(self.cells), self.stage, self.restoration_generation

    def payload_state(self) -> tuple[Any, ...]:
        return self.field.q, self.field.p, self.source_rank, tuple(self.cells), self.stage

    def backing_ids(self) -> tuple[int, int]:
        return id(self), id(self.cells)


def gaussian_payload_inverse(field: cubic.gaussian.Field, payload: tuple[int, ...], work: Work) -> tuple[int, ...]:
    symplectic = list(payload[:4])
    coefficient = payload[4] % field.p
    inverse_symplectic = cubic.gaussian.symplectic_inverse(symplectic, field.q)
    scalar = cubic.gaussian.cocycle_closed(symplectic, inverse_symplectic, field, cubic.gaussian.Work())
    inverse_coefficient = pow(coefficient * scalar % field.p, -1, field.p)
    work.inverse_gaussian_scalar_rematerializations += 1
    return tuple(inverse_symplectic) + (inverse_coefficient,)


def compile_public_plan(q: int, depth: int, family: str) -> tuple[list[Operation], list[int]]:
    if depth < 1 or family not in FAMILIES:
        fail("invalid public plan")
    seed, left_ops, right_ops, fiber = cubic.public_word_plan(q, depth, family)
    operations: list[Operation] = []
    for kind, payload in reversed(right_ops):
        if kind == "GAUSSIAN":
            operations.append(Operation(kind, tuple(payload[0]) + (payload[1],)))
        else:
            operations.append(Operation(kind, (int(payload),)))
    operations.append(Operation("GAUSSIAN", tuple(seed.deformed.symplectic) + (1,)))
    for kind, payload in left_ops:
        if kind == "GAUSSIAN":
            operations.append(Operation(kind, tuple(payload[0]) + (payload[1],)))
        else:
            operations.append(Operation(kind, (int(payload),)))
    return operations, fiber


def apply_gaussian(carrier: Carrier, payload: tuple[int, ...], work: Work) -> None:
    q, p, rank = carrier.field.q, carrier.field.p, carrier.source_rank
    symplectic = list(payload[:4])
    coefficient = payload[4] % p
    output = [0] * len(carrier.cells)
    work.temporary_vector_field_cells_peak = max(work.temporary_vector_field_cells_peak, len(output))
    for fiber in range(2):
        base = fiber * q
        for x in range(q):
            output_offset = (base + x) * rank
            for y in range(q):
                kernel = coefficient * cubic.gaussian.kernel_value(symplectic, x, y, carrier.field) % p
                work.gaussian_kernel_phase_evaluations += 1
                input_offset = (base + y) * rank
                for column in range(rank):
                    output[output_offset + column] += kernel * carrier.cells[input_offset + column]
                    work.gaussian_field_multiply_adds += 1
            for column in range(rank):
                output[output_offset + column] %= p
    carrier.cells[:] = output


def apply_cubic(carrier: Carrier, strength: int, work: Work) -> None:
    q, p, rank = carrier.field.q, carrier.field.p, carrier.source_rank
    for fiber in range(2):
        for x in range(q):
            multiplier = cubic.gaussian.phase(carrier.field, strength * x * x * x)
            work.cubic_phase_evaluations += 1
            offset = (fiber * q + x) * rank
            for column in range(rank):
                carrier.cells[offset + column] = carrier.cells[offset + column] * multiplier % p
                work.cubic_field_multiplications += 1


def apply_fiber(carrier: Carrier, fiber: list[int], work: Work) -> None:
    q, p, rank = carrier.field.q, carrier.field.p, carrier.source_rank
    a, b, c, d = fiber
    output = [0] * len(carrier.cells)
    work.temporary_vector_field_cells_peak = max(work.temporary_vector_field_cells_peak, len(output))
    for x in range(q):
        first = x * rank
        second = (q + x) * rank
        for column in range(rank):
            left = carrier.cells[first + column]
            right = carrier.cells[second + column]
            # Relation fiber coordinates are [source,target], so state action
            # uses the transpose of the stored 2-by-2 matrix.
            output[first + column] = (a * left + c * right) % p
            output[second + column] = (b * left + d * right) % p
            work.fiber_field_multiply_adds += 4
    carrier.cells[:] = output


def apply_operation(carrier: Carrier, operation: Operation, work: Work) -> None:
    if operation.kind == "GAUSSIAN":
        apply_gaussian(carrier, operation.payload, work)
    elif operation.kind == "CUBIC":
        apply_cubic(carrier, operation.payload[0], work)
    elif operation.kind == "FIBER":
        apply_fiber(carrier, list(operation.payload), work)
    else:
        fail("unknown operation")


def inverse_operation(carrier: Carrier, operation: Operation, work: Work, wrong: bool = False) -> Operation:
    if operation.kind == "GAUSSIAN":
        payload = list(gaussian_payload_inverse(carrier.field, operation.payload, work))
        if wrong:
            payload[-1] = (payload[-1] + 1) % carrier.field.p or 1
        return Operation("GAUSSIAN", tuple(payload))
    if operation.kind == "CUBIC":
        return Operation("CUBIC", ((-operation.payload[0] + int(wrong)) % carrier.field.q,))
    if operation.kind == "FIBER":
        payload = cubic.gaussian.fiber_inverse(list(operation.payload), carrier.field.p)
        if wrong:
            payload[0] = (payload[0] + 1) % carrier.field.p
        return Operation("FIBER", tuple(payload))
    fail("unknown inverse operation")


def forward(carrier: Carrier, depth: int, family: str, reorder: bool = False) -> tuple[list[Operation], Work]:
    if carrier.stage != "IDLE":
        fail("carrier is not idle")
    kernel_operations, fiber = compile_public_plan(carrier.field.q, depth, family)
    operations = kernel_operations + [Operation("FIBER", tuple(fiber))]
    if reorder:
        operations = list(reversed(operations))
    work = Work()
    for operation in operations:
        apply_operation(carrier, operation, work)
    carrier.stage = "FORWARD_COMPLETE"
    return operations, work


def reverse(carrier: Carrier, operations: list[Operation], work: Work, mutation: str | None = None) -> None:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("carrier lacks forward state")
    sequence = list(reversed(operations))
    if mutation == "MISSING":
        sequence = sequence[1:]
    elif mutation == "REORDER":
        sequence = list(operations)
    for index, operation in enumerate(sequence):
        apply_operation(carrier, inverse_operation(carrier, operation, work, mutation == "WRONG" and index == 0), work)
    carrier.stage = "IDLE"


def matrix_rank(carrier: Carrier) -> int:
    p, rows, columns = carrier.field.p, carrier.dimension, carrier.source_rank
    matrix = [carrier.cells[row * columns : (row + 1) * columns] for row in range(rows)]
    pivot_row = 0
    for column in range(columns):
        pivot = next((row for row in range(pivot_row, rows) if matrix[row][column] % p), None)
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        inverse = pow(matrix[pivot_row][column] % p, -1, p)
        matrix[pivot_row] = [value * inverse % p for value in matrix[pivot_row]]
        for row in range(rows):
            if row == pivot_row or not matrix[row][column] % p:
                continue
            scale = matrix[row][column] % p
            matrix[row] = [(left - scale * right) % p for left, right in zip(matrix[row], matrix[pivot_row])]
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


def probes(q: int, source_rank: int, family: str) -> tuple[tuple[int, int, int], ...]:
    code = 1 if family == "PRIMARY" else 2
    return (
        ((3 * code + 1) % source_rank, (5 * code + 2) % (2 * q), 1),
        ((7 * code + 2) % source_rank, (11 * code + 3) % (2 * q), 2),
        ((13 * code + 1) % source_rank, (17 * code + 4) % (2 * q), 3),
        ((19 * code + 2) % source_rank, (23 * code + 5) % (2 * q), 5),
    )


def boundary(carrier: Carrier, family: str) -> int:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("boundary unavailable")
    return sum(
        weight * carrier.cells[output_coordinate * carrier.source_rank + source_column]
        for source_column, output_coordinate, weight in probes(carrier.field.q, carrier.source_rank, family)
    ) % carrier.field.p


def semantic_digest(carrier: Carrier) -> str:
    return hashlib.sha256(repr((carrier.field.q, carrier.source_rank, tuple(carrier.cells))).encode("ascii")).hexdigest()


def apply_single_gaussian(vector: list[int], payload: tuple[int, ...], field: cubic.gaussian.Field, work: dict[str, int]) -> list[int]:
    q, p = field.q, field.p
    symplectic = list(payload[:4])
    coefficient = payload[4] % p
    output = []
    for x in range(q):
        output.append(coefficient * sum(cubic.gaussian.kernel_value(symplectic, x, y, field) * vector[y] for y in range(q)) % p)
    work["gaussian_field_multiply_adds"] += q * q
    work["gaussian_kernel_phase_evaluations"] += q * q
    return output


def streamed_boundary(q: int, depth: int, family: str, source_rank: int) -> dict[str, Any]:
    field = cubic.gaussian.make_field(q)
    kernel_operations, fiber = compile_public_plan(q, depth, family)
    work = {
        "gaussian_field_multiply_adds": 0,
        "gaussian_kernel_phase_evaluations": 0,
        "cubic_field_multiplications": 0,
        "cubic_phase_evaluations": 0,
    }
    total = 0
    for source_column, output_coordinate, weight in probes(q, source_rank, family):
        input_fiber, y = divmod(source_column, q)
        target_fiber, x = divmod(output_coordinate, q)
        vector = [0] * q
        vector[y] = 1
        for operation in kernel_operations:
            if operation.kind == "GAUSSIAN":
                vector = apply_single_gaussian(vector, operation.payload, field, work)
            else:
                multiplier_values = [cubic.gaussian.phase(field, operation.payload[0] * index * index * index) for index in range(q)]
                vector = [value * multiplier_values[index] % field.p for index, value in enumerate(vector)]
                work["cubic_phase_evaluations"] += q
                work["cubic_field_multiplications"] += q
        total = (total + weight * fiber[2 * input_fiber + target_fiber] * vector[x]) % field.p
    return {
        "boundary": total,
        "live_dynamic_field_cells": 2 * q,
        "public_word_plan_cells": 12 * depth,
        "resident_dynamic_relation_cells": 0,
        "source_columns_streamed": len(probes(q, source_rank, family)),
        "work": work,
        "cold_start_comparison_used": False,
    }


def factor_invertibility_certificate(q: int, depth: int, family: str) -> dict[str, Any]:
    field = cubic.gaussian.make_field(q)
    operations, fiber = compile_public_plan(q, depth, family)
    gaussian_nonzero = all(operation.payload[4] % field.p for operation in operations if operation.kind == "GAUSSIAN")
    symplectic_determinants = [
        (operation.payload[0] * operation.payload[3] - operation.payload[1] * operation.payload[2]) % q
        for operation in operations if operation.kind == "GAUSSIAN"
    ]
    fiber_determinant = (fiber[0] * fiber[3] - fiber[1] * fiber[2]) % field.p
    return {
        "all_symplectic_determinants_one": all(value == 1 for value in symplectic_determinants),
        "all_gaussian_scalars_nonzero": gaussian_nonzero,
        "all_cubic_diagonal_entries_nonzero_by_root_of_unity_law": True,
        "fiber_determinant_nonzero": bool(fiber_determinant),
        "full_two_fiber_operator_invertible": all(value == 1 for value in symplectic_determinants) and gaussian_nonzero and bool(fiber_determinant),
        "derived_full_source_action_span": 2 * q,
        "derived_explicit_full_open_interface_field_cells": 4 * q * q,
    }


def transaction(carrier: Carrier, depth: int, family: str) -> dict[str, Any]:
    before = carrier.canonical_state()
    backing = carrier.backing_ids()
    generation = carrier.restoration_generation
    operations, work = forward(carrier, depth, family)
    projected = boundary(carrier, family)
    action_rank = matrix_rank(carrier)
    commitment = semantic_digest(carrier)
    baseline = streamed_boundary(carrier.field.q, depth, family, carrier.source_rank)
    certificate = factor_invertibility_certificate(carrier.field.q, depth, family)
    reverse(carrier, operations, work)
    exact = carrier.canonical_state() == before
    same_backing = carrier.backing_ids() == backing
    generation_stable = carrier.restoration_generation == generation
    if not exact or not same_backing or not generation_stable or projected != baseline["boundary"] or action_rank != carrier.source_rank:
        fail(
            f"action-bundle qualification failed q={carrier.field.q} depth={depth} family={family} "
            f"exact={exact} backing={same_backing} generation={generation_stable} "
            f"boundary={projected} baseline={baseline['boundary']} rank={action_rank}/{carrier.source_rank}"
        )
    carrier.restoration_generation += 1
    q, rank = carrier.field.q, carrier.source_rank
    return {
        "q": q,
        "p": carrier.field.p,
        "family": family,
        "depth": depth,
        "source_rank": rank,
        "boundary": projected,
        "semantic_commitment": commitment,
        "observed_action_span_rank": action_rank,
        "action_span_equals_declared_source_rank": action_rank == rank,
        "factor_invertibility_certificate": certificate,
        "accepted_resident_field_cells": 2 * q * rank,
        "accepted_temporary_field_cells_peak": work.temporary_vector_field_cells_peak,
        "accepted_resident_plus_temporary_peak_field_cells": 4 * q * rank,
        "retained_inverse_history_field_cells": 0,
        "public_word_plan_cells": 12 * depth,
        "work": work.__dict__,
        "matched_source_streamed_classical": {**baseline, "boundary_matches": True},
        "exact_canonical_state_restored": exact,
        "same_backing_restored": same_backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
        "intermediate_columns_serialized": False,
    }


def raises(action: Callable[[], Any]) -> bool:
    try:
        action()
    except (RuntimeError, ValueError):
        return True
    return False


def controls() -> dict[str, bool]:
    q, depth, family, rank = 5, 2, "PRIMARY", 10

    def restoration_mutation(mutation: str) -> bool:
        carrier = Carrier.seal(q, rank)
        before = carrier.canonical_state()
        operations, work = forward(carrier, depth, family)
        reverse(carrier, operations, work, mutation)
        return carrier.canonical_state() != before

    normal = Carrier.seal(q, rank)
    normal_operations, normal_work = forward(normal, depth, family)
    normal_boundary = boundary(normal, family)
    reverse(normal, normal_operations, normal_work)
    reordered = Carrier.seal(q, rank)
    forward(reordered, depth, family, reorder=True)
    reordered_boundary = boundary(reordered, family)
    return {
        "missing_inverse_fails_restoration": restoration_mutation("MISSING"),
        "wrong_inverse_fails_restoration": restoration_mutation("WRONG"),
        "reordered_inverse_fails_restoration": restoration_mutation("REORDER"),
        "module_reordering_changes_final_boundary": reordered_boundary != normal_boundary,
        "premature_boundary_projection_rejected": raises(lambda: boundary(Carrier.seal(q, rank), family)),
        "null_source_rank_rejected": raises(lambda: Carrier.seal(q, 0)),
        "oversize_source_rank_rejected": raises(lambda: Carrier.seal(q, 2 * q + 1)),
        "invalid_family_rejected": raises(lambda: compile_public_plan(q, depth, "INVALID")),
        "zero_depth_rejected": raises(lambda: compile_public_plan(q, 0, family)),
    }


def dense_small_order_parity() -> dict[str, bool]:
    q, depth, family, rank = 5, 2, "PRIMARY", 10
    carrier = Carrier.seal(q, rank)
    operations, _ = forward(carrier, depth, family)
    relation_carrier = cubic.Carrier.seal(q)
    cubic.forward(relation_carrier, depth, family, "STREAMED_PHASE_SUM")
    expected = []
    for output_coordinate in range(2 * q):
        target, x = divmod(output_coordinate, q)
        row = []
        for source_coordinate in range(2 * q):
            source, y = divmod(source_coordinate, q)
            row.append(cubic.relation_value(relation_carrier.deformed, source, target, x, y))
        expected.extend(row)
    return {
        "q5_depth2_full_two_fiber_action_matches_component_relation": carrier.cells == expected,
        "q5_depth2_action_rank_is10": matrix_rank(carrier) == 10,
        "compiled_operation_count_matches_word_law": len(operations) == 4 * depth + 2,
    }


def reuse_check() -> dict[str, Any]:
    carrier = Carrier.seal(23, 22)
    backing = carrier.backing_ids()
    initial = carrier.payload_state()
    first = transaction(carrier, 1, "PRIMARY")
    second = transaction(carrier, 2, "ALTERNATE")
    fresh = transaction(Carrier.seal(23, 22), 2, "ALTERNATE")
    return {
        "first_boundary": first["boundary"],
        "second_boundary": second["boundary"],
        "fresh_second_boundary": fresh["boundary"],
        "second_boundary_matches_fresh": second["boundary"] == fresh["boundary"],
        "second_commitment_matches_fresh": second["semantic_commitment"] == fresh["semantic_commitment"],
        "same_backing_reused": carrier.backing_ids() == backing,
        "exact_payload_state_restored_after_reuse": carrier.payload_state() == initial,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
    }


def case_specs() -> list[tuple[int, int, str, int]]:
    cases = [(q, BASE_DEPTH, "PRIMARY", min(2 * q, SOURCE_RANK_CAP)) for q in ORDERS]
    cases.append((DEEPEST_ORDER, DEEPEST_DEPTH, "PRIMARY", SOURCE_RANK_CAP))
    cases.append((ALTERNATE_ORDER, ALTERNATE_DEPTH, "ALTERNATE", SOURCE_RANK_CAP))
    return cases


def build_result() -> dict[str, Any]:
    cases = [transaction(Carrier.seal(q, rank), depth, family) for q, depth, family, rank in case_specs()]
    all_certified = all(case["factor_invertibility_certificate"]["full_two_fiber_operator_invertible"] for case in cases)
    result = {
        "schema": "CAT_CAS_GROWING_PRIME_CUBIC_WEIL_OPEN_INTERFACE_ACTION_SPAN_RESULTS_V1",
        "claim_candidate": CLAIM,
        "claim_ceiling": "SAFE_PRIME_PAIRS_Q5_11_23_29_41_53_83_89_113_P11_23_47_59_83_107_167_179_227_PRIMARY_DEPTH2_ALL_PRIMARY_DEPTH4_Q113_ALTERNATE_DEPTH3_Q41_SOURCE_RANKS10_OR22_DIRECT_PROCESS_SOFTWARE",
        "cases": cases,
        "controls": controls(),
        "dense_small_order_parity": dense_small_order_parity(),
        "restoration_and_reuse": reuse_check(),
        "observed_resource_law": {
            "all_declared_action_spans_equal_source_rank": all(case["action_span_equals_declared_source_rank"] for case in cases),
            "all_full_two_fiber_operators_certified_invertible": all_certified,
            "full_explicit_open_interface_field_cells": "4*Q^2",
            "accepted_declared_bundle_resident_field_cells": "2*Q*R",
            "accepted_declared_bundle_resident_plus_temporary_peak_field_cells": "4*Q*R",
            "matched_final_boundary_dynamic_field_cells": "2*Q",
            "matched_public_word_plan_cells": "12*DEPTH",
            "retained_inverse_history_field_cells": 0,
            "fixed_rank_open_interface_across_growing_q_established": False,
        },
        "matched_baselines": {
            "strongest_executed": "EXACT_SOURCE_STREAMED_Q_VECTOR_PUBLIC_OPERATOR_WORD_RECURRENCE_FOR_THE_IDENTICAL_FINAL_BOUNDARY",
            "all_case_boundaries_match": all(case["matched_source_streamed_classical"]["boundary_matches"] for case in cases),
            "public_operator_word_is_a_complete_rematerialization_descriptor": True,
            "dense_relation_matrix_is_QUALIFICATION_ORACLE_NOT_MATCHED_BASELINE": True,
            "cold_start_comparison_used": False,
        },
        "resource_accounting": {
            "maximum_q": max(case["q"] for case in cases),
            "maximum_depth": max(case["depth"] for case in cases),
            "maximum_executed_source_rank": max(case["source_rank"] for case in cases),
            "accepted_carrier_and_temporary_cells_counted": True,
            "public_plan_and_matched_streaming_work_counted": True,
            "controller_backend_traffic_bytes": 0,
            "snapshot_cells": 0,
            "python_objects_allocator_interpreter_native_libraries_and_whole_process_peak_excluded": True,
            "advantage_claimed": False,
        },
        "next_obstruction": "THE_Q2_COMPONENT_CHART_CAN_BE_REPLACED_BY_A_2QR_ACTION_BUNDLE_FOR_R_SOURCE_COLUMNS_BUT_INVERTIBILITY_PRESERVES_ALL_R_DIRECTIONS_THE_EXPLICIT_FULL_TWO_FIBER_BASIS_BUNDLE_USES4Q2_CELLS_AND_THE_IDENTICAL_FINAL_BOUNDARY_CAN_BE_STREAMED_IN2Q_DYNAMIC_CELLS_FROM_THE_PUBLIC_WORD_DESCRIPTOR",
        "claim_boundaries": {
            "compact_general_open_relation_closure": False,
            "full_source_bundle_executed_at_every_declared_q": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage_or_small_wall_crossing": False,
            "physical_waveform_or_silicon_execution": False,
            "replacement_of_physical_bits_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
    }
    if not all(result["controls"].values()) or not all(result["dense_small_order_parity"].values()):
        fail("controls or dense parity failed")
    if not result["observed_resource_law"]["all_declared_action_spans_equal_source_rank"] or not all_certified:
        fail("action-span law failed")
    if not all(case["exact_canonical_state_restored"] and case["same_backing_restored"] for case in cases):
        fail("restoration failed")
    if not all([
        result["restoration_and_reuse"]["second_boundary_matches_fresh"],
        result["restoration_and_reuse"]["second_commitment_matches_fresh"],
        result["restoration_and_reuse"]["same_backing_reused"],
        result["restoration_and_reuse"]["exact_payload_state_restored_after_reuse"],
    ]):
        fail("reuse failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = json.dumps(build_result(), indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
