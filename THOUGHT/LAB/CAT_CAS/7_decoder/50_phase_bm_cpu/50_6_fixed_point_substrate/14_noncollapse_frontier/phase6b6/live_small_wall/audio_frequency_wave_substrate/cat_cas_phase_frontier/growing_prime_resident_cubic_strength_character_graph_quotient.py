#!/usr/bin/env python3
"""Exact character-graph quotient diagnostic for the resident cubic port.

The carrier keeps the actual q-cell cubic-strength factor and two q-cell data
factors.  Public controlled-cubic and Weil-Gaussian morphisms are attached as
typed graph nodes rather than expanding a q by 2q amplitude state.  Final
boundary amplitudes are evaluated by a cache-free recursive character sum.

This is direct-process finite-field software.  The recursive projection is
explicitly counted and is identical to a compact classical factor-graph
recurrence.  No custody, resource advantage, physical-phase, or unbounded-
computation claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import growing_prime_resident_cubic_strength_phase_port_weil_action_rank as resident


ORDERS = (5, 11, 23, 29, 41, 53, 83, 89, 113)
FAMILIES = ("PRIMARY", "ALTERNATE")
CLAIM = (
    "BOUNDED_EXACT_GROWING_SAFE_PRIME_RESIDENT_CUBIC_STRENGTH_CHARACTER_GRAPH_"
    "STORES3Q_PHASE_FACTOR_CELLS_PLUS12D4_PUBLIC_MORPHISM_INTEGERS_AND_STREAMS_"
    "FINAL_ONLY_BOUNDARIES_WITHOUT_Q2_AMPLITUDE_STATE_WITH_EXACT_GRAPH_"
    "RESTORATION_AND_REUSE_BUT_PROJECTION_ENUMERATES8Q_TO_THE_2D_BASE_PATHS_"
    "AND_IDENTICAL_CLASSICAL_GRAPH_AND_Q2_RADER_NTT_TRANSFER_PARETO_BASELINES_REMAIN"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class RecursiveWork:
    base_path_evaluations: int = 0
    base_field_multiplications: int = 0
    controlled_phase_evaluations: int = 0
    controlled_field_multiplications: int = 0
    gaussian_kernel_evaluations: int = 0
    gaussian_field_multiply_adds: int = 0
    fiber_field_multiply_adds: int = 0
    projection_field_multiply_adds: int = 0
    recursion_calls: int = 0
    recursion_stack_frames_peak: int = 0


@dataclass
class GraphCarrier:
    field: resident.open_action.cubic.gaussian.Field
    latent: list[int]
    data: list[int]
    nodes: list[resident.Operation]
    fixture_family: str
    stage: str = "IDLE"
    restoration_generation: int = 0

    @classmethod
    def seal(
        cls,
        q: int,
        fixture_family: str,
        latent_override: list[int] | None = None,
        data_override: list[int] | None = None,
    ) -> "GraphCarrier":
        if fixture_family not in FAMILIES:
            fail("invalid fixture family")
        field = resident.open_action.cubic.gaussian.make_field(q)
        latent = list(latent_override) if latent_override is not None else resident.latent_fixture(field, fixture_family)
        data = list(data_override) if data_override is not None else resident.data_fixture(field, fixture_family)
        if len(latent) != q or not any(value % field.p for value in latent):
            fail("invalid latent factor")
        if len(data) != 2 * q or not any(value % field.p for value in data):
            fail("invalid data factor")
        return cls(field, [value % field.p for value in latent], [value % field.p for value in data], [], fixture_family)

    @property
    def q(self) -> int:
        return self.field.q

    def backing_ids(self) -> tuple[int, int, int, int]:
        return id(self), id(self.latent), id(self.data), id(self.nodes)

    def canonical_payload(self) -> tuple[Any, ...]:
        return (
            self.q,
            self.field.p,
            self.fixture_family,
            tuple(self.latent),
            tuple(self.data),
            tuple(self.nodes),
            self.stage,
        )


def plan_payload_integer_cells(operations: list[resident.Operation]) -> int:
    return sum(len(operation.payload) for operation in operations)


def graph_commitment(carrier: GraphCarrier) -> str:
    payload = (
        carrier.q,
        tuple(carrier.latent),
        tuple(carrier.data),
        tuple((node.kind, node.payload) for node in carrier.nodes),
    )
    return hashlib.sha256(repr(payload).encode("ascii")).hexdigest()


def forward(carrier: GraphCarrier, depth: int, family: str) -> list[resident.Operation]:
    if carrier.stage != "IDLE" or carrier.nodes:
        fail("graph carrier is not idle")
    operations = resident.public_plan(carrier.q, depth, family)
    carrier.nodes.extend(operations)
    carrier.stage = "GRAPH_FORWARD_COMPLETE"
    return operations


def inverse_pair_exact(
    carrier: GraphCarrier,
    operation: resident.Operation,
    inverse: resident.Operation,
) -> bool:
    q, p = carrier.q, carrier.field.p
    if operation.kind != inverse.kind:
        return False
    if operation.kind == "CONTROLLED_CUBIC":
        return (operation.payload[0] + inverse.payload[0]) % q == 0
    if operation.kind in ("DATA_GAUSSIAN", "LATENT_GAUSSIAN"):
        a, b, c, d = operation.payload[:4]
        e, f, g, h = inverse.payload[:4]
        product = [
            (a * e + b * g) % q,
            (a * f + b * h) % q,
            (c * e + d * g) % q,
            (c * f + d * h) % q,
        ]
        if product != [1, 0, 0, 1]:
            return False
        vector = [(index * index + 3 * index + 1) % p for index in range(q)]
        first = resident.gaussian_vector(vector, operation.payload, carrier.field, resident.Work(), "DATA")
        second = resident.gaussian_vector(first, inverse.payload, carrier.field, resident.Work(), "DATA")
        return second == vector
    if operation.kind == "FIBER":
        a, b, c, d = operation.payload
        e, f, g, h = inverse.payload
        return (
            (a * e + c * f) % p,
            (b * e + d * f) % p,
            (a * g + c * h) % p,
            (b * g + d * h) % p,
        ) == (1, 0, 0, 1)
    return False


def reverse(
    carrier: GraphCarrier,
    operations: list[resident.Operation],
    mutation: str | None = None,
) -> None:
    if carrier.stage != "GRAPH_FORWARD_COMPLETE":
        fail("graph carrier lacks forward state")
    sequence = list(reversed(operations))
    if mutation == "MISSING":
        sequence = sequence[1:]
    elif mutation == "REORDER":
        sequence = list(operations)
    for index, operation in enumerate(sequence):
        if not carrier.nodes or carrier.nodes[-1] != operation:
            fail("inverse order does not match resident graph")
        inverse = resident.inverse_operation(
            resident.Carrier(carrier.field, [], carrier.fixture_family),
            operation,
            resident.Work(),
            wrong=mutation == "WRONG" and index == 0,
        )
        if not inverse_pair_exact(carrier, operation, inverse):
            fail("inverse certificate failed")
        carrier.nodes.pop()
    if carrier.nodes:
        fail("graph inverse left resident morphisms")
    carrier.stage = "IDLE"


def graph_amplitude(
    carrier: GraphCarrier,
    operation_count: int,
    s: int,
    fiber: int,
    x: int,
    work: RecursiveWork,
    stack_depth: int = 1,
) -> int:
    q, p = carrier.q, carrier.field.p
    work.recursion_calls += 1
    work.recursion_stack_frames_peak = max(work.recursion_stack_frames_peak, stack_depth)
    if operation_count == 0:
        work.base_path_evaluations += 1
        work.base_field_multiplications += 1
        return carrier.latent[s] * carrier.data[fiber * q + x] % p
    operation = carrier.nodes[operation_count - 1]
    if operation.kind == "CONTROLLED_CUBIC":
        work.controlled_phase_evaluations += 1
        work.controlled_field_multiplications += 1
        multiplier = resident.phase(carrier.field, operation.payload[0] * s * x * x * x)
        return multiplier * graph_amplitude(
            carrier, operation_count - 1, s, fiber, x, work, stack_depth + 1
        ) % p
    if operation.kind == "DATA_GAUSSIAN":
        total = 0
        for y in range(q):
            kernel = resident.open_action.cubic.gaussian.kernel_value(
                list(operation.payload[:4]), x, y, carrier.field
            )
            work.gaussian_kernel_evaluations += 1
            work.gaussian_field_multiply_adds += 1
            total += operation.payload[4] * kernel * graph_amplitude(
                carrier, operation_count - 1, s, fiber, y, work, stack_depth + 1
            )
        return total % p
    if operation.kind == "LATENT_GAUSSIAN":
        total = 0
        for source_s in range(q):
            kernel = resident.open_action.cubic.gaussian.kernel_value(
                list(operation.payload[:4]), s, source_s, carrier.field
            )
            work.gaussian_kernel_evaluations += 1
            work.gaussian_field_multiply_adds += 1
            total += operation.payload[4] * kernel * graph_amplitude(
                carrier, operation_count - 1, source_s, fiber, x, work, stack_depth + 1
            )
        return total % p
    if operation.kind == "FIBER":
        a, b, c, d = operation.payload
        if fiber == 0:
            coefficients = (a, c)
        else:
            coefficients = (b, d)
        work.fiber_field_multiply_adds += 2
        return sum(
            coefficient
            * graph_amplitude(
                carrier,
                operation_count - 1,
                s,
                source_fiber,
                x,
                work,
                stack_depth + 1,
            )
            for source_fiber, coefficient in enumerate(coefficients)
        ) % p
    fail("unknown graph operation")


def boundary(carrier: GraphCarrier, family: str) -> tuple[int, RecursiveWork]:
    if carrier.stage != "GRAPH_FORWARD_COMPLETE":
        fail("boundary unavailable")
    work = RecursiveWork()
    total = 0
    for s, fiber, x, weight in resident.probes(carrier.q, family):
        total += weight * graph_amplitude(carrier, len(carrier.nodes), s, fiber, x, work)
        work.projection_field_multiply_adds += 2
    return total % carrier.field.p, work


def classical_graph_amplitude(
    field: resident.open_action.cubic.gaussian.Field,
    latent: tuple[int, ...],
    data: tuple[int, ...],
    operations: tuple[resident.Operation, ...],
    operation_count: int,
    s: int,
    fiber: int,
    x: int,
    work: RecursiveWork,
    stack_depth: int = 1,
) -> int:
    q, p = field.q, field.p
    work.recursion_calls += 1
    work.recursion_stack_frames_peak = max(work.recursion_stack_frames_peak, stack_depth)
    if operation_count == 0:
        work.base_path_evaluations += 1
        work.base_field_multiplications += 1
        return latent[s] * data[fiber * q + x] % p
    operation = operations[operation_count - 1]
    if operation.kind == "CONTROLLED_CUBIC":
        work.controlled_phase_evaluations += 1
        work.controlled_field_multiplications += 1
        multiplier = pow(field.root, operation.payload[0] * s * x * x * x % q, p)
        return multiplier * classical_graph_amplitude(
            field, latent, data, operations, operation_count - 1, s, fiber, x, work, stack_depth + 1
        ) % p
    if operation.kind in ("DATA_GAUSSIAN", "LATENT_GAUSSIAN"):
        total = 0
        for source in range(q):
            out_coordinate = x if operation.kind == "DATA_GAUSSIAN" else s
            kernel = resident.open_action.cubic.gaussian.kernel_value(
                list(operation.payload[:4]), out_coordinate, source, field
            )
            work.gaussian_kernel_evaluations += 1
            work.gaussian_field_multiply_adds += 1
            next_s = s if operation.kind == "DATA_GAUSSIAN" else source
            next_x = source if operation.kind == "DATA_GAUSSIAN" else x
            total += operation.payload[4] * kernel * classical_graph_amplitude(
                field,
                latent,
                data,
                operations,
                operation_count - 1,
                next_s,
                fiber,
                next_x,
                work,
                stack_depth + 1,
            )
        return total % p
    if operation.kind == "FIBER":
        a, b, c, d = operation.payload
        coefficients = (a, c) if fiber == 0 else (b, d)
        work.fiber_field_multiply_adds += 2
        return sum(
            coefficient
            * classical_graph_amplitude(
                field,
                latent,
                data,
                operations,
                operation_count - 1,
                s,
                source_fiber,
                x,
                work,
                stack_depth + 1,
            )
            for source_fiber, coefficient in enumerate(coefficients)
        ) % p
    fail("unknown classical graph operation")


def matched_classical_graph(carrier: GraphCarrier, family: str) -> dict[str, Any]:
    operations = tuple(carrier.nodes)
    latent, data = tuple(carrier.latent), tuple(carrier.data)
    work = RecursiveWork()
    total = 0
    for s, fiber, x, weight in resident.probes(carrier.q, family):
        total += weight * classical_graph_amplitude(
            carrier.field, latent, data, operations, len(operations), s, fiber, x, work
        )
        work.projection_field_multiply_adds += 2
    depth = (len(operations) - 1) // 4
    return {
        "boundary": total % carrier.field.p,
        "phase_factor_field_cells": 3 * carrier.q,
        "retained_public_morphism_payload_integer_cells": plan_payload_integer_cells(list(operations)),
        "retained_public_morphism_node_records": len(operations),
        "recursive_field_accumulator_slots_peak": work.recursion_stack_frames_peak + 1,
        "work": work.__dict__,
        "expected_base_path_evaluations": 8 * carrier.q ** (2 * depth),
        "cold_start_comparison_used": False,
    }


def expanded_initial_cells(carrier: GraphCarrier) -> list[int]:
    q, p = carrier.q, carrier.field.p
    return [
        carrier.latent[s] * carrier.data[fiber * q + x] % p
        for s in range(q)
        for fiber in range(2)
        for x in range(q)
    ]


def expanded_small_order_parity(carrier: GraphCarrier, family: str) -> bool:
    if carrier.q not in (5, 11):
        fail("small-order parity is restricted")
    graph_values = [
        graph_amplitude(carrier, len(carrier.nodes), s, fiber, x, RecursiveWork())
        for s in range(carrier.q)
        for fiber in range(2)
        for x in range(carrier.q)
    ]
    expanded = resident.Carrier(carrier.field, expanded_initial_cells(carrier), carrier.fixture_family)
    resident.forward(expanded, (len(carrier.nodes) - 1) // 4, family)
    return graph_values == expanded.cells


def transaction(carrier: GraphCarrier, depth: int, family: str) -> dict[str, Any]:
    initial = carrier.canonical_payload()
    backing = carrier.backing_ids()
    operations = forward(carrier, depth, family)
    graph_state_commitment = graph_commitment(carrier)
    projected, work = boundary(carrier, family)
    classical = matched_classical_graph(carrier, family)
    rader = resident.matched_rader_ntt_classical_boundary(
        expanded_initial_cells(carrier), carrier.field, depth, family
    )
    small_parity = expanded_small_order_parity(carrier, family) if carrier.q in (5, 11) and depth == 1 else None
    expected_paths = 8 * carrier.q ** (2 * depth)
    if not all((
        work.base_path_evaluations == expected_paths,
        classical["work"]["base_path_evaluations"] == expected_paths,
        work.__dict__ == classical["work"],
        projected == classical["boundary"],
        projected == rader["boundary"],
        small_parity is not False,
    )):
        fail("character graph semantic comparison failed")
    reverse(carrier, operations)
    restored = carrier.canonical_payload() == initial
    same_backing = carrier.backing_ids() == backing
    if not restored or not same_backing:
        fail("character graph restoration failed")
    carrier.restoration_generation += 1
    return {
        "q": carrier.q,
        "p": carrier.field.p,
        "depth": depth,
        "family": family,
        "boundary": projected,
        "graph_state_commitment": graph_state_commitment,
        "phase_factor_field_cells": 3 * carrier.q,
        "retained_public_morphism_payload_integer_cells": plan_payload_integer_cells(operations),
        "retained_public_morphism_node_records": len(operations),
        "recursive_field_accumulator_slots_peak": work.recursion_stack_frames_peak + 1,
        "accepted_named_field_value_slots_peak": 3 * carrier.q + work.recursion_stack_frames_peak + 1,
        "expected_base_path_evaluations": expected_paths,
        "actual_base_path_evaluations": work.base_path_evaluations,
        "recursive_work": work.__dict__,
        "q2_amplitude_cells_on_accepted_graph_path": 0,
        "assignment_or_path_list_materialized": False,
        "recursive_cache_entries": 0,
        "runtime_factor_amplitudes_serialized": False,
        "intermediate_amplitudes_serialized": False,
        "small_order_full_state_parity": small_parity,
        "matched_identical_classical_graph": classical,
        "matched_exact_rader_ntt_transfer": rader,
        "exact_graph_payload_restored": restored,
        "same_backing_restored": same_backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
    }


def mutation_fails(mutation: str) -> bool:
    carrier = GraphCarrier.seal(5, "PRIMARY")
    initial = carrier.canonical_payload()
    operations = forward(carrier, 1, "PRIMARY")
    try:
        reverse(carrier, operations, mutation)
    except RuntimeError:
        return carrier.canonical_payload() != initial
    return carrier.canonical_payload() != initial


def controls() -> dict[str, bool]:
    premature = GraphCarrier.seal(5, "PRIMARY")
    premature_rejected = False
    try:
        boundary(premature, "PRIMARY")
    except RuntimeError:
        premature_rejected = True

    normal = GraphCarrier.seal(5, "PRIMARY")
    reordered = GraphCarrier.seal(5, "PRIMARY")
    forward(normal, 1, "PRIMARY")
    normal_boundary, _ = boundary(normal, "PRIMARY")
    reordered_nodes = resident.public_plan(5, 1, "PRIMARY")
    reordered.nodes.extend(reversed(reordered_nodes))
    reordered.stage = "GRAPH_FORWARD_COMPLETE"
    reordered_boundary, _ = boundary(reordered, "PRIMARY")

    null_latent_rejected = False
    null_data_rejected = False
    invalid_family_rejected = False
    zero_depth_rejected = False
    try:
        GraphCarrier.seal(5, "PRIMARY", latent_override=[0] * 5)
    except RuntimeError:
        null_latent_rejected = True
    try:
        GraphCarrier.seal(5, "PRIMARY", data_override=[0] * 10)
    except RuntimeError:
        null_data_rejected = True
    try:
        GraphCarrier.seal(5, "INVALID")
    except RuntimeError:
        invalid_family_rejected = True
    try:
        resident.public_plan(5, 0, "PRIMARY")
    except RuntimeError:
        zero_depth_rejected = True

    field = resident.open_action.cubic.gaussian.make_field(5)
    first = GraphCarrier.seal(5, "PRIMARY")
    second = GraphCarrier.seal(
        5,
        "PRIMARY",
        latent_override=[resident.phase(field, index + 1) for index in range(5)],
    )
    first_plan = resident.public_plan(first.q, 1, "PRIMARY")
    second_plan = resident.public_plan(second.q, 1, "PRIMARY")
    return {
        "missing_inverse_fails": mutation_fails("MISSING"),
        "wrong_inverse_fails": mutation_fails("WRONG"),
        "reordered_inverse_fails": mutation_fails("REORDER"),
        "module_reordering_changes_boundary": normal_boundary != reordered_boundary,
        "premature_projection_rejected": premature_rejected,
        "null_latent_rejected": null_latent_rejected,
        "null_data_rejected": null_data_rejected,
        "invalid_family_rejected": invalid_family_rejected,
        "zero_depth_rejected": zero_depth_rejected,
        "public_plan_independent_of_runtime_factor_amplitudes": first_plan == second_plan,
        "accepted_graph_has_no_q2_amplitude_array": not hasattr(first, "cells"),
        "accepted_projection_has_no_recursive_cache": True,
    }


def reuse() -> dict[str, Any]:
    carrier = GraphCarrier.seal(23, "PRIMARY")
    backing = carrier.backing_ids()
    initial = carrier.canonical_payload()
    first = transaction(carrier, 1, "PRIMARY")
    second = transaction(carrier, 1, "ALTERNATE")
    fresh = GraphCarrier.seal(23, "PRIMARY")
    fresh_second = transaction(fresh, 1, "ALTERNATE")
    return {
        "first_boundary": first["boundary"],
        "second_boundary": second["boundary"],
        "fresh_second_boundary": fresh_second["boundary"],
        "second_matches_fresh": second["boundary"] == fresh_second["boundary"],
        "second_graph_commitment_matches_fresh": second["graph_state_commitment"] == fresh_second["graph_state_commitment"],
        "exact_payload_restored_after_reuse": carrier.canonical_payload() == initial,
        "same_backing_reused": carrier.backing_ids() == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
    }


def run() -> dict[str, Any]:
    case_specs = [(q, 1, "PRIMARY") for q in ORDERS]
    case_specs.extend(((11, 2, "PRIMARY"), (5, 3, "PRIMARY"), (5, 2, "ALTERNATE")))
    cases = [transaction(GraphCarrier.seal(q, family), depth, family) for q, depth, family in case_specs]
    control_results = controls()
    reuse_result = reuse()
    if not all(control_results.values()) or not all((
        reuse_result["second_matches_fresh"],
        reuse_result["second_graph_commitment_matches_fresh"],
        reuse_result["exact_payload_restored_after_reuse"],
        reuse_result["same_backing_reused"],
        reuse_result["restoration_generation"] == 2,
    )):
        fail("controls or reuse failed")
    return {
        "schema": "CAT_CAS_GROWING_PRIME_RESIDENT_CUBIC_STRENGTH_CHARACTER_GRAPH_QUOTIENT_V1",
        "claim_candidate": CLAIM,
        "claim_ceiling": "SAFE_PRIME_PAIRS_Q5_11_23_29_41_53_83_89_113_P11_23_47_59_83_107_167_179_227_PRIMARY_DEPTH1_ALL_PRIMARY_DEPTH2_Q11_PRIMARY_DEPTH3_Q5_ALTERNATE_DEPTH2_Q5_TWO_FIBER_DIRECT_PROCESS_SOFTWARE_AUXILIARY_NTT_MODULUS998244353",
        "cases": cases,
        "controls": control_results,
        "restoration_and_reuse": reuse_result,
        "observed_resource_law": {
            "resident_runtime_phase_factor_field_cells": "3*Q",
            "retained_public_morphism_payload_integer_cells": "12*DEPTH+4",
            "retained_public_morphism_node_records": "4*DEPTH+1",
            "q2_amplitude_cells_on_accepted_graph_path": 0,
            "cache_free_base_path_evaluations_for_four_two_fiber_probes": "8*Q^(2*DEPTH)",
            "fixed_work_across_growing_depth_established": False,
            "matched_exact_rader_transfer_resident_field_cells": "2*Q^2",
        },
        "matched_baseline": {
            "identical_classical_character_graph_all_boundaries_and_work_match": True,
            "exact_rader_ntt_q2_transfer_all_boundaries_match": True,
            "pareto_points": [
                "CACHE_FREE_3Q_FACTOR_PLUS_PUBLIC_MORPHISM_GRAPH_WITH_EXPONENTIAL_RECOMPUTATION",
                "EXACT_RADER_NTT_2Q2_TRANSFER_WITH_POLYNOMIAL_WORK_AND_COUNTED_LINEAR_SCRATCH",
            ],
            "cold_start_comparison_used": False,
        },
        "resource_accounting": {
            "runtime_factor_cells_public_morphism_payloads_node_records_recursive_stack_and_projection_work_counted": True,
            "rader_ntt_state_scratch_mixed_width_capacity_and_work_counted": True,
            "python_frames_objects_allocator_interpreter_native_libraries_and_whole_process_peak_excluded": True,
            "advantage_claimed": False,
        },
        "claim_boundaries": {
            "amplitude_state_closure_in_3q_cells": False,
            "fixed_work_or_fixed_total_cost_across_depth": False,
            "machine_enforced_hidden_runtime_factors": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage_or_small_wall_crossing": False,
            "physical_waveform_or_silicon_execution": False,
            "replacement_of_physical_bits_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
        "next_obstruction": "THE_NONLINEAR_CHARACTER_GRAPH_REMOVES_Q2_RESIDENT_AMPLITUDE_MATERIALIZATION_BUT_CACHE_FREE_FINAL_CLOSURE_ENUMERATES8Q_TO_THE_2D_PATHS_WHILE_THE_WORK_REDUCED_TRANSFER_USES_Q2_STATE_AND_IDENTICAL_CLASSICAL_RECURRENCES_IMPLEMENT_BOTH_PARETO_POINTS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
