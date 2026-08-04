#!/usr/bin/env python3
"""Exact latent Gaussian-message closure for the cubic character graph.

Conditioned on a data-coordinate path, the resident cubic-strength factor
remains one Gaussian-or-delta phase message through every declared latent
Gaussian transform.  The accepted path therefore sums q**depth data histories
instead of q**(2*depth) joint histories.  This is direct-process finite-field
software and has an identical classical message recurrence.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import growing_prime_resident_cubic_strength_character_graph_quotient as graph


resident = graph.resident
ORDERS = graph.ORDERS
FAMILIES = graph.FAMILIES
CLAIM = (
    "BOUNDED_EXACT_GROWING_SAFE_PRIME_LATENT_GAUSSIAN_OR_DELTA_PHASE_SEPARATOR_"
    "MESSAGE_ELIMINATES_ALL_D_CUBIC_STRENGTH_HISTORY_COORDINATES_STORES2QPLUS4_"
    "RUNTIME_FIELD_ELEMENTS_AND_STREAMS_FINAL_BOUNDARY_IN8Q_TO_THE_D_DATA_PATHS_WITH_"
    "EXACT_GRAPH_RESTORATION_AND_REUSE_BUT_DATA_AIRY_CLOSURE_REMAINS_OPEN_AND_"
    "IDENTICAL_CLASSICAL_MESSAGE_AND2Q2_RADER_TRANSFER_BASELINES_REMAIN"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


Message = tuple[Any, ...]


@dataclass
class MessageWork:
    data_history_evaluations: int = 0
    data_kernel_evaluations: int = 0
    controlled_message_phase_updates: int = 0
    gaussian_message_transforms: int = 0
    gaussian_to_gaussian_transforms: int = 0
    gaussian_to_delta_transforms: int = 0
    delta_to_gaussian_transforms: int = 0
    message_evaluations: int = 0
    data_field_multiplications: int = 0
    final_field_multiply_adds: int = 0
    data_history_tuple_cells_peak: int = 0
    latent_message_field_cells_peak: int = 0


@dataclass
class MessageCarrier:
    field: resident.open_action.cubic.gaussian.Field
    latent_message: Message
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
        message_override: Message | None = None,
        data_override: list[int] | None = None,
    ) -> "MessageCarrier":
        if fixture_family not in FAMILIES:
            fail("invalid fixture family")
        field = resident.open_action.cubic.gaussian.make_field(q)
        code = 1 if fixture_family == "PRIMARY" else 2
        message = message_override or (
            "GAUSSIAN",
            1,
            (code + 1) % q,
            (2 * code + 1) % q,
            code % q,
        )
        validate_message(message, field)
        data = list(data_override) if data_override is not None else resident.data_fixture(field, fixture_family)
        if len(data) != 2 * q or not any(value % field.p for value in data):
            fail("invalid data factor")
        return cls(field, message, [value % field.p for value in data], [], fixture_family)

    @property
    def q(self) -> int:
        return self.field.q

    def backing_ids(self) -> tuple[int, int, int, int]:
        return id(self), id(self.latent_message), id(self.data), id(self.nodes)

    def canonical_payload(self) -> tuple[Any, ...]:
        return (
            self.q,
            self.field.p,
            self.fixture_family,
            self.latent_message,
            tuple(self.data),
            tuple(self.nodes),
            self.stage,
        )


def validate_message(message: Message, field: resident.open_action.cubic.gaussian.Field) -> None:
    if message[0] == "GAUSSIAN" and len(message) == 5:
        if not message[1] % field.p:
            fail("zero Gaussian message scalar")
        return
    if message[0] == "DELTA" and len(message) == 3:
        if not message[1] % field.p or not 0 <= message[2] < field.q:
            fail("invalid delta message")
        return
    fail("invalid latent message")


def legendre(value: int, q: int) -> int:
    symbol = pow(value % q, (q - 1) // 2, q)
    if symbol == 1:
        return 1
    if symbol == q - 1:
        return -1
    fail("zero Legendre argument")


def message_field_cells(message: Message) -> int:
    return 4 if message[0] == "GAUSSIAN" else 2


def message_phase_multiply(
    message: Message,
    coefficient: int,
    field: resident.open_action.cubic.gaussian.Field,
    work: MessageWork,
) -> Message:
    work.controlled_message_phase_updates += 1
    if message[0] == "GAUSSIAN":
        _, scalar, quadratic, linear, constant = message
        output = ("GAUSSIAN", scalar, quadratic, (linear + coefficient) % field.q, constant)
    else:
        _, scalar, center = message
        output = ("DELTA", scalar * resident.phase(field, coefficient * center) % field.p, center)
    work.latent_message_field_cells_peak = max(work.latent_message_field_cells_peak, message_field_cells(output))
    return output


def gaussian_message_transform(
    message: Message,
    payload: tuple[int, ...],
    field: resident.open_action.cubic.gaussian.Field,
    work: MessageWork,
) -> Message:
    q, p = field.q, field.p
    a, b, _, d, coefficient = payload
    if b % q == 0:
        fail("declared message closure requires nonzero Gaussian b")
    inverse_b = pow(b, -1, q)
    half_b = pow(2 * b % q, -1, q)
    work.gaussian_message_transforms += 1
    if message[0] == "DELTA":
        _, scalar, center = message
        output = (
            "GAUSSIAN",
            scalar * coefficient % p,
            d * half_b % q,
            -center * inverse_b % q,
            a * center * center * half_b % q,
        )
        work.delta_to_gaussian_transforms += 1
    else:
        _, scalar, quadratic, linear, constant = message
        alpha = (quadratic + a * half_b) % q
        if alpha == 0:
            center = b * linear % q
            value = scalar * coefficient * q % p
            value = value * resident.phase(field, constant + d * center * center * half_b) % p
            output = ("DELTA", value, center)
            work.gaussian_to_delta_transforms += 1
        else:
            inverse_four_alpha = pow(4 * alpha % q, -1, q)
            output = (
                "GAUSSIAN",
                scalar * coefficient * field.gauss_one * legendre(alpha, q) % p,
                (d * half_b - pow(b, -2, q) * inverse_four_alpha) % q,
                linear * pow(2 * alpha * b % q, -1, q) % q,
                (constant - linear * linear * inverse_four_alpha) % q,
            )
            work.gaussian_to_gaussian_transforms += 1
    work.latent_message_field_cells_peak = max(work.latent_message_field_cells_peak, message_field_cells(output))
    return output


def evaluate_message(
    message: Message,
    coordinate: int,
    field: resident.open_action.cubic.gaussian.Field,
    work: MessageWork,
) -> int:
    work.message_evaluations += 1
    if message[0] == "DELTA":
        return message[1] if coordinate == message[2] else 0
    _, scalar, quadratic, linear, constant = message
    return scalar * resident.phase(field, quadratic * coordinate * coordinate + linear * coordinate + constant) % field.p


def forward(carrier: MessageCarrier, depth: int, family: str) -> list[resident.Operation]:
    if carrier.stage != "IDLE" or carrier.nodes:
        fail("message carrier is not idle")
    operations = resident.public_plan(carrier.q, depth, family)
    carrier.nodes.extend(operations)
    carrier.stage = "MESSAGE_FORWARD_COMPLETE"
    return operations


def reverse(
    carrier: MessageCarrier,
    operations: list[resident.Operation],
    mutation: str | None = None,
) -> None:
    if carrier.stage != "MESSAGE_FORWARD_COMPLETE":
        fail("message carrier lacks forward state")
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
        if not graph.inverse_pair_exact(carrier, operation, inverse):
            fail("inverse certificate failed")
        carrier.nodes.pop()
    if carrier.nodes:
        fail("message inverse left resident morphisms")
    carrier.stage = "IDLE"


def layer_operations(carrier: MessageCarrier) -> list[list[resident.Operation]]:
    depth = (len(carrier.nodes) - 1) // 4
    if depth < 1 or len(carrier.nodes) != 4 * depth + 1:
        fail("invalid resident public word")
    return [carrier.nodes[4 * layer : 4 * layer + 4] for layer in range(depth)]


def message_boundary(carrier: MessageCarrier, family: str) -> tuple[int, MessageWork]:
    if carrier.stage != "MESSAGE_FORWARD_COMPLETE":
        fail("boundary unavailable")
    layers = layer_operations(carrier)
    depth, q, p = len(layers), carrier.q, carrier.field.p
    fiber_matrix = carrier.nodes[-1].payload
    work = MessageWork(latent_message_field_cells_peak=message_field_cells(carrier.latent_message))
    total = 0
    for final_s, final_fiber, final_x, weight in resident.probes(q, family):
        fiber_coefficients = (fiber_matrix[0], fiber_matrix[2]) if final_fiber == 0 else (fiber_matrix[1], fiber_matrix[3])
        for source_fiber, fiber_coefficient in enumerate(fiber_coefficients):
            for source_path in itertools.product(range(q), repeat=depth):
                work.data_history_evaluations += 1
                work.data_history_tuple_cells_peak = max(work.data_history_tuple_cells_peak, depth + 1)
                coordinates = source_path + (final_x,)
                message = carrier.latent_message
                scalar = carrier.data[source_fiber * q + coordinates[0]]
                for layer_index, operations in enumerate(layers):
                    source_x, target_x = coordinates[layer_index], coordinates[layer_index + 1]
                    if family == "PRIMARY":
                        message = message_phase_multiply(
                            message, operations[0].payload[0] * source_x**3, carrier.field, work
                        )
                        scalar = scalar * operations[1].payload[4] % p
                        scalar = scalar * resident.open_action.cubic.gaussian.kernel_value(
                            list(operations[1].payload[:4]), target_x, source_x, carrier.field
                        ) % p
                        work.data_kernel_evaluations += 1
                        work.data_field_multiplications += 2
                        message = message_phase_multiply(
                            message, operations[2].payload[0] * target_x**3, carrier.field, work
                        )
                        message = gaussian_message_transform(message, operations[3].payload, carrier.field, work)
                    else:
                        message = message_phase_multiply(
                            message, operations[0].payload[0] * source_x**3, carrier.field, work
                        )
                        message = gaussian_message_transform(message, operations[1].payload, carrier.field, work)
                        message = message_phase_multiply(
                            message, operations[2].payload[0] * source_x**3, carrier.field, work
                        )
                        scalar = scalar * operations[3].payload[4] % p
                        scalar = scalar * resident.open_action.cubic.gaussian.kernel_value(
                            list(operations[3].payload[:4]), target_x, source_x, carrier.field
                        ) % p
                        work.data_kernel_evaluations += 1
                        work.data_field_multiplications += 2
                total += weight * fiber_coefficient * scalar * evaluate_message(
                    message, final_s, carrier.field, work
                )
                work.final_field_multiply_adds += 3
    return total % p, work


def classical_boundary(carrier: MessageCarrier, family: str) -> dict[str, Any]:
    # Separate copies make the comparison state explicit; the algebra is
    # deliberately identical because this experiment tests resource novelty.
    clone = MessageCarrier(
        carrier.field,
        tuple(carrier.latent_message),
        list(carrier.data),
        list(carrier.nodes),
        carrier.fixture_family,
        carrier.stage,
    )
    boundary, work = message_boundary(clone, family)
    depth = (len(clone.nodes) - 1) // 4
    return {
        "boundary": boundary,
        "runtime_field_elements": 2 * carrier.q + 4,
        "public_morphism_node_records": len(clone.nodes),
        "public_morphism_payload_integer_cells": graph.plan_payload_integer_cells(clone.nodes),
        "data_history_evaluations": 8 * carrier.q**depth,
        "work": work.__dict__,
        "cold_start_comparison_used": False,
    }


def expanded_initial(carrier: MessageCarrier) -> list[int]:
    latent = [evaluate_message(carrier.latent_message, s, carrier.field, MessageWork()) for s in range(carrier.q)]
    return [
        latent[s] * carrier.data[fiber * carrier.q + x] % carrier.field.p
        for s in range(carrier.q)
        for fiber in range(2)
        for x in range(carrier.q)
    ]


def message_commitment(carrier: MessageCarrier) -> str:
    return hashlib.sha256(
        repr((carrier.q, carrier.latent_message, tuple(carrier.data), tuple(carrier.nodes))).encode("ascii")
    ).hexdigest()


def transaction(carrier: MessageCarrier, depth: int, family: str) -> dict[str, Any]:
    initial, backing = carrier.canonical_payload(), carrier.backing_ids()
    operations = forward(carrier, depth, family)
    commitment = message_commitment(carrier)
    boundary, work = message_boundary(carrier, family)
    classical = classical_boundary(carrier, family)
    rader = resident.matched_rader_ntt_classical_boundary(expanded_initial(carrier), carrier.field, depth, family)
    expected_histories = 8 * carrier.q**depth
    q_bits, p_bits = carrier.q.bit_length(), carrier.field.p.bit_length()
    if not all((
        boundary == classical["boundary"] == rader["boundary"],
        work.__dict__ == classical["work"],
        work.data_history_evaluations == expected_histories,
        work.gaussian_message_transforms == depth * expected_histories,
        work.controlled_message_phase_updates == 2 * depth * expected_histories,
    )):
        fail("message semantic or work comparison failed")
    reverse(carrier, operations)
    restored = carrier.canonical_payload() == initial
    same_backing = carrier.backing_ids() == backing
    if not restored or not same_backing:
        fail("message carrier restoration failed")
    carrier.restoration_generation += 1
    return {
        "q": carrier.q,
        "p": carrier.field.p,
        "depth": depth,
        "family": family,
        "boundary": boundary,
        "message_graph_commitment": commitment,
        "runtime_data_field_cells": 2 * carrier.q,
        "runtime_latent_message_field_cells": 4,
        "accepted_runtime_field_elements": 2 * carrier.q + 4,
        "accepted_runtime_bit_capacity_upper_bound": (2 * carrier.q + 1) * p_bits + 3 * q_bits,
        "data_and_message_scalar_field_cell_bit_capacity": p_bits,
        "message_exponent_field_cell_bit_capacity": q_bits,
        "public_morphism_node_records": len(operations),
        "public_morphism_payload_integer_cells": graph.plan_payload_integer_cells(operations),
        "public_morphism_payload_bit_capacity_upper_bound": 10 * depth * q_bits + (2 * depth + 4) * p_bits,
        "expected_data_history_evaluations": expected_histories,
        "actual_data_history_evaluations": work.data_history_evaluations,
        "joint_strength_data_history_evaluations": 0,
        "q2_amplitude_cells_on_accepted_message_path": 0,
        "data_history_or_assignment_list_materialized": False,
        "recursive_or_dynamic_cache_entries": 0,
        "latent_message_serialized": False,
        "intermediate_amplitudes_serialized": False,
        "message_work": work.__dict__,
        "matched_identical_classical_message": classical,
        "matched_exact_rader_ntt_transfer": rader,
        "exact_graph_payload_restored": restored,
        "same_backing_restored": same_backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
    }


def mutation_fails(mutation: str) -> bool:
    carrier = MessageCarrier.seal(5, "PRIMARY")
    initial = carrier.canonical_payload()
    operations = forward(carrier, 1, "PRIMARY")
    try:
        reverse(carrier, operations, mutation)
    except RuntimeError:
        return carrier.canonical_payload() != initial
    return carrier.canonical_payload() != initial


def controls() -> dict[str, bool]:
    premature = MessageCarrier.seal(5, "PRIMARY")
    premature_rejected = False
    try:
        message_boundary(premature, "PRIMARY")
    except RuntimeError:
        premature_rejected = True

    original = MessageCarrier.seal(5, "PRIMARY")
    mutated = MessageCarrier.seal(5, "PRIMARY", message_override=("GAUSSIAN", 1, 2, 5, 1))
    original_plan = forward(original, 1, "PRIMARY")
    mutated_plan = forward(mutated, 1, "PRIMARY")
    original_boundary, _ = message_boundary(original, "PRIMARY")
    mutated_boundary, _ = message_boundary(mutated, "PRIMARY")

    invalid_message_rejected = False
    null_data_rejected = False
    try:
        MessageCarrier.seal(5, "PRIMARY", message_override=("INVALID", 1))
    except RuntimeError:
        invalid_message_rejected = True
    try:
        MessageCarrier.seal(5, "PRIMARY", data_override=[0] * 10)
    except RuntimeError:
        null_data_rejected = True
    return {
        "missing_inverse_fails": mutation_fails("MISSING"),
        "wrong_inverse_fails": mutation_fails("WRONG"),
        "reordered_inverse_fails": mutation_fails("REORDER"),
        "premature_projection_rejected": premature_rejected,
        "invalid_message_rejected": invalid_message_rejected,
        "null_data_rejected": null_data_rejected,
        "latent_message_mutation_changes_boundary": original_boundary != mutated_boundary,
        "public_plan_independent_of_latent_message": original_plan == mutated_plan,
        "accepted_message_carrier_has_no_q2_amplitude_array": not hasattr(original, "cells"),
        "accepted_projection_has_no_cache": True,
    }


def reuse() -> dict[str, Any]:
    carrier = MessageCarrier.seal(23, "PRIMARY")
    initial, backing = carrier.canonical_payload(), carrier.backing_ids()
    first = transaction(carrier, 2, "PRIMARY")
    second = transaction(carrier, 2, "ALTERNATE")
    fresh = MessageCarrier.seal(23, "PRIMARY")
    fresh_second = transaction(fresh, 2, "ALTERNATE")
    return {
        "first_boundary": first["boundary"],
        "second_boundary": second["boundary"],
        "fresh_second_boundary": fresh_second["boundary"],
        "second_matches_fresh": second["boundary"] == fresh_second["boundary"],
        "second_commitment_matches_fresh": second["message_graph_commitment"] == fresh_second["message_graph_commitment"],
        "exact_payload_restored_after_reuse": carrier.canonical_payload() == initial,
        "same_backing_reused": carrier.backing_ids() == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
    }


def run() -> dict[str, Any]:
    case_specs = [(q, 1, "PRIMARY") for q in ORDERS]
    case_specs.extend(((23, 3, "PRIMARY"), (11, 4, "PRIMARY"), (5, 6, "PRIMARY"), (5, 5, "ALTERNATE")))
    cases = [transaction(MessageCarrier.seal(q, family), depth, family) for q, depth, family in case_specs]
    control_result, reuse_result = controls(), reuse()
    if not all(control_result.values()) or not all((
        reuse_result["second_matches_fresh"],
        reuse_result["second_commitment_matches_fresh"],
        reuse_result["exact_payload_restored_after_reuse"],
        reuse_result["same_backing_reused"],
        reuse_result["restoration_generation"] == 2,
    )):
        fail("controls or reuse failed")
    return {
        "schema": "CAT_CAS_GROWING_PRIME_CUBIC_CHARACTER_SUM_SEPARATOR_MESSAGE_CLOSURE_V1",
        "claim_candidate": CLAIM,
        "claim_ceiling": "SAFE_PRIME_PAIRS_Q5_11_23_29_41_53_83_89_113_P11_23_47_59_83_107_167_179_227_PRIMARY_DEPTH1_ALL_PRIMARY_DEPTH3_Q23_PRIMARY_DEPTH4_Q11_PRIMARY_DEPTH6_Q5_ALTERNATE_DEPTH5_Q5_TWO_FIBER_GAUSSIAN_LATENT_FIXTURE_DIRECT_PROCESS_SOFTWARE_AUXILIARY_NTT_MODULUS998244353",
        "cases": cases,
        "controls": control_result,
        "restoration_and_reuse": reuse_result,
        "observed_resource_law": {
            "accepted_runtime_field_elements": "2*Q+4_MIXED_FQ_AND_FP_ELEMENTS",
            "public_morphism_node_records": "4*DEPTH+1",
            "public_morphism_payload_integer_cells": "12*DEPTH+4",
            "data_history_evaluations": "8*Q^DEPTH",
            "joint_strength_data_history_evaluations": 0,
            "q2_amplitude_cells_on_accepted_message_path": 0,
            "fixed_work_across_growing_depth_established": False,
            "matched_exact_rader_transfer_resident_field_cells": "2*Q^2",
        },
        "matched_baseline": {
            "identical_classical_message_all_boundaries_and_work_match": True,
            "exact_rader_ntt_q2_transfer_all_boundaries_match": True,
            "pareto_points": [
                "EXACT_2Q_PLUS4_LATENT_GAUSSIAN_MESSAGE_WITH_8Q_TO_THE_D_DATA_HISTORY_WORK",
                "EXACT_RADER_NTT_2Q2_TRANSFER_WITH_POLYNOMIAL_WORK_AND_COUNTED_LINEAR_SCRATCH",
            ],
            "cold_start_comparison_used": False,
        },
        "resource_accounting": {
            "message_data_public_plan_history_tuple_and_projection_work_counted": True,
            "rader_ntt_state_scratch_mixed_width_capacity_and_work_counted": True,
            "verification_expansion_excluded_from_accepted_message_path_but_reported_as_baseline": True,
            "python_iterators_frames_objects_allocator_interpreter_native_libraries_and_whole_process_peak_excluded": True,
            "advantage_claimed": False,
        },
        "claim_boundaries": {
            "arbitrary_latent_factor_closure": False,
            "data_airy_message_closure": False,
            "subquadratic_state_polynomial_work_closure": False,
            "fixed_work_or_fixed_total_cost_across_depth": False,
            "machine_enforced_hidden_runtime_factors": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage_or_small_wall_crossing": False,
            "physical_waveform_or_silicon_execution": False,
            "replacement_of_physical_bits_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
        "next_obstruction": "LATENT_GAUSSIAN_PHASE_MESSAGES_REMOVE_ALL_STRENGTH_HISTORY_SUMS_BUT_THE_DATA_AXIS_EXITS_GAUSSIAN_CLOSURE_UNDER_CUBIC_PHASES_SO_FINAL_CONTRACTION_STILL_ENUMERATES8Q_TO_THE_D_DATA_PATHS_WHILE_IDENTICAL_CLASSICAL_RECURRENCES_REALIZE_BOTH_TIME_STATE_PARETO_POINTS",
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
