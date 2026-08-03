#!/usr/bin/env python3
"""Exact rank-two coherent secant closure for one nonintegrable coupling."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import f17_coherent_veronese_phase_chart_closure as rank1
import f17_nonlinear_canonical_mps_separator_chart as backend


MODE_COUNT = 17
DECLARED_K = (4, 8, 16, 32, 64, 128)
EXACT_K = (4, 8, 16, 32)
FINITE_FIELDS = ((103, 72), (137, 16))
FAMILIES = ("PRIMARY", "REUSE")
FINAL_BOUNDARY = "K_MINUS_1_MODE0_ONE_MODE1_OCCUPATION"


def fail(message: str) -> None:
    raise RuntimeError(message)


def negative(alg: backend.Algebra, value: Any) -> Any:
    return alg.sub(alg.zero, value)


def reflection(vector: list[Any]) -> list[Any]:
    result = list(vector)
    result[0], result[1] = result[1], result[0]
    return result


@dataclass(frozen=True)
class SecantProgram:
    k: int
    family: str
    eta_exponent: int
    module_a: tuple[rank1.Primitive, ...]
    module_b: tuple[rank1.Primitive, ...]
    final_boundary: str = FINAL_BOUNDARY

    def fingerprint(self) -> str:
        return rank1.digest_json(public_program_descriptor(self))


def compile_program(k: int, family: str) -> SecantProgram:
    base = rank1.compile_program(k, family)
    eta_exponent = 3 if family == "PRIMARY" else 5
    program = SecantProgram(
        k=k,
        family=family,
        eta_exponent=eta_exponent,
        module_a=base.primitives[:20],
        module_b=base.primitives[20:],
    )
    validate_program(program)
    return program


def validate_program(program: SecantProgram) -> None:
    if program.k not in DECLARED_K or program.family not in FAMILIES:
        fail("secant program identity changed")
    if program.eta_exponent not in (3, 5):
        fail("nonintegrable coupling coefficient changed")
    if len(program.module_a) != 20 or len(program.module_b) != 20:
        fail("two-consumer module split changed")
    for primitive in (*program.module_a, *program.module_b):
        rank1.validate_primitive(primitive)
    if program.final_boundary != FINAL_BOUNDARY:
        fail("secant boundary type changed")


def public_program_descriptor(program: SecantProgram) -> dict[str, Any]:
    return {
        "k": program.k,
        "family": program.family,
        "chart": "RANK2_COHERENT_VERONESE_SECANT",
        "mode_count": MODE_COUNT,
        "seed": "MODE0_RAISED_TO_K",
        "coupling": {
            "kind": "INVOLUTIVE_COHERENT_SUPERPOSITION",
            "law": "I_PLUS_ETA_R",
            "eta_exponent": program.eta_exponent,
            "reflection": "SWAP_MODE0_MODE1",
        },
        "module_a": [item.as_json() for item in program.module_a],
        "module_b": [item.as_json() for item in program.module_b],
        "final_boundary": program.final_boundary,
    }


def algebra_signature(alg: backend.Algebra) -> str:
    return rank1.algebra_signature(alg)


def lease(program: SecantProgram, alg: backend.Algebra) -> str:
    return rank1.digest_json(
        {
            "program": program.fingerprint(),
            "algebra": algebra_signature(alg),
            "carrier": "TWO_SLOT_RANK2_COHERENT_SECANT",
        }
    )


@dataclass
class SecantCarrier:
    alg: backend.Algebra
    weights: list[Any]
    vectors: list[list[Any]]
    active_rank: int = 0
    generation: int = 0
    lease: str | None = None
    stage: str = "RESTORED"
    active_k: int | None = None
    projection_calls: int = 0
    current_resident_payload_bits: int = 0
    maximum_resident_payload_bits: int = 0
    maximum_resident_value_payload_bits: int = 0
    maximum_resident_numerator_signed_bits: int = 0
    maximum_resident_denominator_bits: int = 0
    maximum_transient_value_payload_bits: int = 0
    maximum_transient_numerator_signed_bits: int = 0
    maximum_transient_denominator_bits: int = 0
    maximum_commitment_record_json_bytes: int = 0
    maximum_active_rank: int = 0
    maximum_inverse_coupling_transient_components: int = 0
    character_cell_multiplications: int = 0
    shear_cell_updates: int = 0

    @classmethod
    def create(cls, alg: backend.Algebra) -> "SecantCarrier":
        carrier = cls(
            alg=alg,
            weights=[alg.zero, alg.zero],
            vectors=[
                [alg.zero for _ in range(MODE_COUNT)],
                [alg.zero for _ in range(MODE_COUNT)],
            ],
        )
        carrier.observe_resident()
        return carrier

    def backing_identity(self) -> tuple[int, int, int, int]:
        return (id(self), id(self.weights), id(self.vectors[0]), id(self.vectors[1]))

    def flat_values(self) -> list[Any]:
        return [
            self.weights[0],
            *self.vectors[0],
            self.weights[1],
            *self.vectors[1],
        ]

    def exact_zero(self) -> bool:
        return (
            self.active_rank == 0
            and all(value == self.alg.zero for value in self.flat_values())
            and self.lease is None
            and self.stage == "RESTORED"
            and self.active_k is None
        )

    def observe_value(self, value: Any, *, resident: bool = False) -> None:
        payload = self.alg.payload_bits(value)
        numerator, denominator = self.alg.coefficient_height(value)
        if resident:
            self.maximum_resident_value_payload_bits = max(
                self.maximum_resident_value_payload_bits, payload
            )
            self.maximum_resident_numerator_signed_bits = max(
                self.maximum_resident_numerator_signed_bits, numerator
            )
            self.maximum_resident_denominator_bits = max(
                self.maximum_resident_denominator_bits, denominator
            )
        else:
            self.maximum_transient_value_payload_bits = max(
                self.maximum_transient_value_payload_bits, payload
            )
            self.maximum_transient_numerator_signed_bits = max(
                self.maximum_transient_numerator_signed_bits, numerator
            )
            self.maximum_transient_denominator_bits = max(
                self.maximum_transient_denominator_bits, denominator
            )

    def observe_resident(self) -> None:
        self.current_resident_payload_bits = sum(
            self.alg.payload_bits(value) for value in self.flat_values()
        )
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits, self.current_resident_payload_bits
        )
        self.maximum_active_rank = max(self.maximum_active_rank, self.active_rank)
        for value in self.flat_values():
            self.observe_value(value, resident=True)

    def write_weight(self, slot: int, value: Any) -> None:
        self.weights[slot] = value
        self.observe_resident()

    def write_cell(self, slot: int, mode: int, value: Any) -> None:
        self.vectors[slot][mode] = value
        self.observe_resident()

    def digest(self) -> str:
        return rank1.digest_json(
            {
                "weights": [self.alg.serialize(value) for value in self.weights],
                "vectors": [
                    [self.alg.serialize(value) for value in vector]
                    for vector in self.vectors
                ],
                "active_rank": self.active_rank,
                "generation": self.generation,
                "lease": self.lease,
                "stage": self.stage,
                "active_k": self.active_k,
            }
        )


def reset_observation(carrier: SecantCarrier) -> None:
    carrier.projection_calls = 0
    carrier.current_resident_payload_bits = 0
    carrier.maximum_resident_payload_bits = 0
    carrier.maximum_resident_value_payload_bits = 0
    carrier.maximum_resident_numerator_signed_bits = 0
    carrier.maximum_resident_denominator_bits = 0
    carrier.maximum_transient_value_payload_bits = 0
    carrier.maximum_transient_numerator_signed_bits = 0
    carrier.maximum_transient_denominator_bits = 0
    carrier.maximum_commitment_record_json_bytes = 0
    carrier.maximum_active_rank = carrier.active_rank
    carrier.maximum_inverse_coupling_transient_components = 0
    carrier.character_cell_multiplications = 0
    carrier.shear_cell_updates = 0
    carrier.observe_resident()


def clear_slots(carrier: SecantCarrier) -> None:
    for slot in range(2):
        carrier.write_weight(slot, carrier.alg.zero)
        for mode in range(MODE_COUNT):
            carrier.write_cell(slot, mode, carrier.alg.zero)
    carrier.active_rank = 0
    carrier.observe_resident()


def load_seed(carrier: SecantCarrier) -> None:
    if carrier.active_rank != 0:
        fail("seed load requires an empty secant carrier")
    carrier.write_weight(0, carrier.alg.one)
    carrier.write_cell(0, 0, carrier.alg.one)
    carrier.active_rank = 1
    carrier.observe_resident()


def unload_seed(carrier: SecantCarrier) -> None:
    expected = [carrier.alg.one] + [carrier.alg.zero] * (MODE_COUNT - 1)
    if (
        carrier.active_rank != 1
        or carrier.weights[0] != carrier.alg.one
        or carrier.vectors[0] != expected
        or carrier.weights[1] != carrier.alg.zero
        or any(value != carrier.alg.zero for value in carrier.vectors[1])
    ):
        fail("inverse path did not return the actual seed coordinates")
    carrier.write_weight(0, carrier.alg.sub(carrier.weights[0], carrier.alg.one))
    carrier.write_cell(0, 0, carrier.alg.sub(carrier.vectors[0][0], carrier.alg.one))
    carrier.active_rank = 0
    carrier.observe_resident()


def assign_components(
    carrier: SecantCarrier,
    components: list[tuple[Any, list[Any]]],
) -> None:
    if len(components) > 2:
        fail("rank-two carrier capacity exceeded")
    clear_slots(carrier)
    for slot, (weight, vector) in enumerate(components):
        carrier.write_weight(slot, weight)
        for mode, value in enumerate(vector):
            carrier.write_cell(slot, mode, value)
    carrier.active_rank = len(components)
    carrier.observe_resident()


def combine_exact_components(
    terms: list[tuple[Any, list[Any]]], alg: backend.Algebra
) -> list[tuple[Any, list[Any]]]:
    combined: list[tuple[Any, list[Any]]] = []
    for weight, vector in terms:
        if weight == alg.zero:
            continue
        for index, (prior_weight, prior_vector) in enumerate(combined):
            if vector == prior_vector:
                merged = alg.add(prior_weight, weight)
                combined[index] = (merged, prior_vector)
                break
        else:
            combined.append((weight, list(vector)))
    return [(weight, vector) for weight, vector in combined if weight != alg.zero]


def apply_coupling(
    carrier: SecantCarrier,
    program: SecantProgram,
    *,
    inverse: bool = False,
    canonical_merge: bool = True,
) -> None:
    eta = carrier.alg.power(program.eta_exponent)
    eta_squared = carrier.alg.mul(eta, eta)
    denominator = carrier.alg.sub(carrier.alg.one, eta_squared)
    if denominator == carrier.alg.zero:
        fail("coupling inverse denominator vanished")
    components = [
        (carrier.weights[slot], list(carrier.vectors[slot]))
        for slot in range(carrier.active_rank)
    ]
    terms: list[tuple[Any, list[Any]]] = []
    if not inverse:
        if carrier.active_rank != 1:
            fail("declared forward coupling requires the rank-one seed chart")
        weight, vector = components[0]
        terms = [
            (weight, vector),
            (carrier.alg.mul(weight, eta), reflection(vector)),
        ]
    else:
        scale = carrier.alg.inverse(denominator)
        carrier.maximum_inverse_coupling_transient_components = max(
            carrier.maximum_inverse_coupling_transient_components,
            2 * len(components),
        )
        for weight, vector in components:
            scaled = carrier.alg.mul(weight, scale)
            reflected_weight = negative(
                carrier.alg, carrier.alg.mul(carrier.alg.mul(weight, eta), scale)
            )
            terms.append((scaled, vector))
            terms.append((reflected_weight, reflection(vector)))
        if not canonical_merge:
            fail("inverse coupling requires exact duplicate cancellation")
        terms = combine_exact_components(terms, carrier.alg)
    for weight, vector in terms:
        carrier.observe_value(weight)
        for value in vector:
            carrier.observe_value(value)
    assign_components(carrier, terms)


def apply_primitive(
    carrier: SecantCarrier,
    primitive: rank1.Primitive,
    *,
    inverse: bool = False,
) -> None:
    rank1.validate_primitive(primitive)
    sign = -1 if inverse else 1
    for slot in range(carrier.active_rank):
        if primitive.kind == "CHARACTER":
            for mode in range(MODE_COUNT):
                phase = carrier.alg.power(
                    sign
                    * primitive.coefficient_exponent
                    * pow(mode, primitive.first, rank1.PRIME)
                )
                carrier.observe_value(phase)
                carrier.write_cell(
                    slot,
                    mode,
                    carrier.alg.mul(carrier.vectors[slot][mode], phase),
                )
                carrier.character_cell_multiplications += 1
        else:
            coefficient = carrier.alg.power(primitive.coefficient_exponent)
            if inverse:
                coefficient = negative(carrier.alg, coefficient)
            product = carrier.alg.mul(
                coefficient, carrier.vectors[slot][primitive.second]
            )
            carrier.observe_value(coefficient)
            carrier.observe_value(product)
            carrier.write_cell(
                slot,
                primitive.first,
                carrier.alg.add(carrier.vectors[slot][primitive.first], product),
            )
            carrier.shear_cell_updates += 1


def apply_sequence(
    carrier: SecantCarrier,
    primitives: Iterable[rank1.Primitive],
    *,
    inverse: bool = False,
) -> None:
    sequence = tuple(primitives)
    if inverse:
        sequence = tuple(reversed(sequence))
    for primitive in sequence:
        apply_primitive(carrier, primitive, inverse=inverse)


def forward(carrier: SecantCarrier, program: SecantProgram) -> None:
    if not isinstance(carrier, SecantCarrier) or not carrier.exact_zero():
        fail("null, leased, or unrestored secant carrier")
    validate_program(program)
    carrier.lease = lease(program, carrier.alg)
    carrier.active_k = program.k
    carrier.stage = "FORWARD_ACTIVE"
    load_seed(carrier)
    apply_coupling(carrier, program)
    carrier.stage = "COUPLED_HIDDEN"
    apply_sequence(carrier, program.module_a)
    carrier.stage = "AFTER_MODULE_A_HIDDEN"
    apply_sequence(carrier, program.module_b)
    carrier.stage = "FORWARD_COMPLETE"


def component_boundary(
    alg: backend.Algebra, weight: Any, vector: list[Any], k: int
) -> Any:
    power = rank1.scalar_power(alg, vector[0], k - 1)
    product = alg.mul(power, vector[1])
    coefficient = alg.mul(rank1.field_integer(alg, k), product)
    return alg.mul(weight, coefficient)


def project_boundary(carrier: SecantCarrier, program: SecantProgram) -> Any:
    if (
        carrier.stage != "FORWARD_COMPLETE"
        or carrier.lease != lease(program, carrier.alg)
        or carrier.active_k != program.k
    ):
        fail("only the final owned secant boundary may be projected")
    carrier.projection_calls += 1
    boundary = carrier.alg.zero
    for slot in range(carrier.active_rank):
        contribution = component_boundary(
            carrier.alg, carrier.weights[slot], carrier.vectors[slot], program.k
        )
        carrier.observe_value(contribution)
        boundary = carrier.alg.add(boundary, contribution)
        carrier.observe_value(boundary)
    return boundary


def inverse(carrier: SecantCarrier, program: SecantProgram) -> None:
    if (
        carrier.stage != "FORWARD_COMPLETE"
        or carrier.lease != lease(program, carrier.alg)
        or carrier.active_k != program.k
    ):
        fail("inverse program does not own the secant lease")
    carrier.stage = "INVERSE_ACTIVE"
    apply_sequence(carrier, program.module_b, inverse=True)
    apply_sequence(carrier, program.module_a, inverse=True)
    apply_coupling(carrier, program, inverse=True)
    unload_seed(carrier)
    carrier.lease = None
    carrier.active_k = None
    carrier.stage = "RESTORED"
    carrier.generation += 1
    carrier.observe_resident()
    if not carrier.exact_zero():
        fail("actual inverse failed exact secant-carrier restoration")


def state_commitment(carrier: SecantCarrier) -> tuple[str, int]:
    return rank1.stream_vector_commitment(carrier.flat_values(), carrier.alg)


def rank_two_certificate(carrier: SecantCarrier) -> dict[str, Any]:
    if carrier.active_rank != 2:
        fail("rank-two certificate requires two resident components")
    for left in range(MODE_COUNT):
        for right in range(left + 1, MODE_COUNT):
            wedge = carrier.alg.sub(
                carrier.alg.mul(
                    carrier.vectors[0][left], carrier.vectors[1][right]
                ),
                carrier.alg.mul(
                    carrier.vectors[0][right], carrier.vectors[1][left]
                ),
            )
            if wedge != carrier.alg.zero:
                return {
                    "component_vectors_projectively_distinct": True,
                    "first_nonzero_wedge_mode_pair": [left, right],
                    "nonzero_component_weights": all(
                        weight != carrier.alg.zero for weight in carrier.weights
                    ),
                    "rank_one_chart_rejected": True,
                }
    fail("declared secant state collapsed to rank one")


def public_program_integer_cells(program: SecantProgram) -> int:
    return 8 + 4 * (len(program.module_a) + len(program.module_b))


def execute_transaction(
    carrier: SecantCarrier, program: SecantProgram
) -> dict[str, Any]:
    reset_observation(carrier)
    initial_digest = carrier.digest()
    backing_before = carrier.backing_identity()
    generation_before = carrier.generation
    stats_before = rank1.stats_snapshot(carrier.alg)
    forward(carrier, program)
    forward_commitment, record_bytes = state_commitment(carrier)
    carrier.maximum_commitment_record_json_bytes = max(
        carrier.maximum_commitment_record_json_bytes, record_bytes
    )
    certificate = rank_two_certificate(carrier)
    boundary = project_boundary(carrier, program)
    inverse(carrier, program)
    stats_after = rank1.stats_snapshot(carrier.alg)
    descriptor = public_program_descriptor(program)
    serialized_boundary = carrier.alg.serialize(boundary)
    return {
        "k": program.k,
        "family": program.family,
        "algebra": algebra_signature(carrier.alg),
        "program_fingerprint": program.fingerprint(),
        "boundary": serialized_boundary,
        "forward_secant_commitment": forward_commitment,
        "rank_two_certificate": certificate,
        "restored_exact_zero": carrier.exact_zero(),
        "same_backing": backing_before == carrier.backing_identity(),
        "generation_before": generation_before,
        "generation_after": carrier.generation,
        "implicit_occupation_dimension_h_k": math.comb(
            program.k + MODE_COUNT - 1, MODE_COUNT - 1
        ),
        "resident_phase_field_cells": 2 * (MODE_COUNT + 1),
        "resident_chart_rank": 2,
        "maximum_active_rank": carrier.maximum_active_rank,
        "maximum_inverse_coupling_transient_components": carrier.maximum_inverse_coupling_transient_components,
        "inverse_coupling_transient_field_cells": 4 * (MODE_COUNT + 1),
        "public_mode_topology_integer_cells": 2 * (MODE_COUNT - 1) + 4,
        "public_program_integer_cells": public_program_integer_cells(program),
        "public_program_json_bytes": len(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ),
        "maximum_resident_payload_bits": carrier.maximum_resident_payload_bits,
        "maximum_resident_value_payload_bits": carrier.maximum_resident_value_payload_bits,
        "maximum_resident_numerator_signed_bits": carrier.maximum_resident_numerator_signed_bits,
        "maximum_resident_denominator_bits": carrier.maximum_resident_denominator_bits,
        "maximum_transient_value_payload_bits": carrier.maximum_transient_value_payload_bits,
        "maximum_transient_numerator_signed_bits": carrier.maximum_transient_numerator_signed_bits,
        "maximum_transient_denominator_bits": carrier.maximum_transient_denominator_bits,
        "maximum_named_transaction_transient_field_cells": 4 * (MODE_COUNT + 1),
        "maximum_commitment_record_json_bytes": carrier.maximum_commitment_record_json_bytes,
        "field_operation_counts": rank1.stats_delta(stats_before, stats_after),
        "character_cell_multiplications": carrier.character_cell_multiplications,
        "shear_cell_updates": carrier.shear_cell_updates,
        "final_boundary_field_cells": 1,
        "final_boundary_payload_bits": carrier.alg.payload_bits(boundary),
        "final_boundary_json_bytes": len(
            json.dumps(serialized_boundary, separators=(",", ":")).encode("utf-8")
        ),
        "intermediate_projection_calls": 0,
        "final_projection_calls": 1,
        "inverse_history_cells": 0,
        "inverse_history_retained": False,
        "snapshot_reload_used": False,
        "response_released_after_restoration": True,
        "resident_carrier_restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "transient_buffer_restoration_class": "NO_RESTORATION_CLAIM",
        "initial_digest": initial_digest,
        "restored_digest_with_generation": carrier.digest(),
        "accepted_path_occupation_vector_materialized": False,
        "accepted_path_occupation_topology_materialized": False,
        "accepted_path_matching_or_assignment_expansion_materialized": False,
        "accepted_path_dense_operator_materialized": False,
        "intermediate_secant_components_serialized_to_controller": False,
    }


def resource_signature(transaction: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "implicit_occupation_dimension_h_k",
        "resident_phase_field_cells",
        "resident_chart_rank",
        "maximum_active_rank",
        "maximum_inverse_coupling_transient_components",
        "inverse_coupling_transient_field_cells",
        "public_mode_topology_integer_cells",
        "public_program_integer_cells",
        "public_program_json_bytes",
        "maximum_resident_payload_bits",
        "maximum_resident_value_payload_bits",
        "maximum_resident_numerator_signed_bits",
        "maximum_resident_denominator_bits",
        "maximum_transient_value_payload_bits",
        "maximum_transient_numerator_signed_bits",
        "maximum_transient_denominator_bits",
        "maximum_named_transaction_transient_field_cells",
        "maximum_commitment_record_json_bytes",
        "field_operation_counts",
        "character_cell_multiplications",
        "shear_cell_updates",
        "final_boundary_payload_bits",
        "final_boundary_json_bytes",
    )
    return {key: transaction[key] for key in keys}


def run_case(k: int, family: str, alg: backend.Algebra) -> dict[str, Any]:
    return execute_transaction(SecantCarrier.create(alg), compile_program(k, family))


def compiled_four_coordinate_baseline(
    transaction: dict[str, Any], program: SecantProgram, alg: backend.Algebra
) -> dict[str, Any]:
    carrier = SecantCarrier.create(alg)
    load_seed(carrier)
    apply_coupling(carrier, program)
    apply_sequence(carrier, program.module_a)
    apply_sequence(carrier, program.module_b)
    retained = []
    for slot in range(carrier.active_rank):
        retained.extend(
            (
                carrier.vectors[slot][0],
                alg.mul(carrier.weights[slot], carrier.vectors[slot][1]),
            )
        )
    commitment, maximum_record_bytes = rank1.stream_vector_commitment(
        retained, alg
    )
    boundary = alg.zero
    k_value = rank1.field_integer(alg, program.k)
    for slot in range(carrier.active_rank):
        mode0, weighted_mode1 = retained[2 * slot : 2 * slot + 2]
        boundary = alg.add(
            boundary,
            alg.mul(
                k_value,
                alg.mul(rank1.scalar_power(alg, mode0, program.k - 1), weighted_mode1),
            ),
        )
    return {
        "k": program.k,
        "family": program.family,
        "algebra": algebra_signature(alg),
        "boundary_agreement": alg.serialize(boundary) == transaction["boundary"],
        "compiler_working_field_cells": 2 * (MODE_COUNT + 1),
        "retained_folded_endpoint_field_cells": 4,
        "total_compiled_warm_field_cells": 4,
        "warm_named_transient_field_cells": 4,
        "retained_folded_endpoint_commitment": commitment,
        "maximum_commitment_record_json_bytes": maximum_record_bytes,
        "snapshot_or_phase_carrier_used_by_warm_evaluator": False,
    }


def control_boundary(
    program: SecantProgram,
    *,
    omit_coupling: bool = False,
    swap_modules: bool = False,
) -> Any:
    carrier = SecantCarrier.create(backend.Algebra("F103", modulus=103, root=72))
    load_seed(carrier)
    if not omit_coupling:
        apply_coupling(carrier, program)
    first, second = (
        (program.module_b, program.module_a)
        if swap_modules
        else (program.module_a, program.module_b)
    )
    apply_sequence(carrier, first)
    apply_sequence(carrier, second)
    carrier.stage = "FORWARD_COMPLETE"
    carrier.lease = lease(program, carrier.alg)
    carrier.active_k = program.k
    return project_boundary(carrier, program)


def controls() -> dict[str, Any]:
    program = compile_program(4, "PRIMARY")
    reference = control_boundary(program)

    missing = SecantCarrier.create(backend.Algebra("F103", modulus=103, root=72))
    forward(missing, program)
    missing_inverse_detected = not missing.exact_zero()

    premature = SecantCarrier.create(backend.Algebra("F103", modulus=103, root=72))
    premature.lease = lease(program, premature.alg)
    premature.active_k = program.k
    premature.stage = "FORWARD_ACTIVE"
    load_seed(premature)
    apply_coupling(premature, program)
    premature.stage = "COUPLED_HIDDEN"
    try:
        project_boundary(premature, program)
    except RuntimeError:
        premature_projection_rejected = True
    else:
        premature_projection_rejected = False

    wrong_order = SecantCarrier.create(backend.Algebra("F103", modulus=103, root=72))
    forward(wrong_order, program)
    apply_sequence(wrong_order, program.module_a, inverse=True)
    apply_sequence(wrong_order, program.module_b, inverse=True)
    try:
        apply_coupling(wrong_order, program, inverse=True)
        unload_seed(wrong_order)
    except RuntimeError:
        reordered_inverse_detected = True
    else:
        reordered_inverse_detected = False

    wrong_coupling = SecantCarrier.create(
        backend.Algebra("F103", modulus=103, root=72)
    )
    forward(wrong_coupling, program)
    apply_sequence(wrong_coupling, program.module_b, inverse=True)
    apply_sequence(wrong_coupling, program.module_a, inverse=True)
    try:
        eta = wrong_coupling.alg.power(program.eta_exponent)
        components = [
            (wrong_coupling.weights[slot], wrong_coupling.vectors[slot])
            for slot in range(wrong_coupling.active_rank)
        ]
        wrong_terms = []
        for weight, vector in components:
            wrong_terms.append((weight, list(vector)))
            wrong_terms.append(
                (wrong_coupling.alg.mul(weight, eta), reflection(vector))
            )
        assign_components(
            wrong_coupling,
            combine_exact_components(wrong_terms, wrong_coupling.alg),
        )
        unload_seed(wrong_coupling)
    except RuntimeError:
        wrong_coupling_inverse_detected = True
    else:
        wrong_coupling_inverse_detected = False

    owner = SecantCarrier.create(backend.Algebra("F103", modulus=103, root=72))
    forward(owner, program)
    try:
        inverse(owner, compile_program(4, "REUSE"))
    except RuntimeError:
        wrong_owner_rejected = True
    else:
        wrong_owner_rejected = False

    try:
        execute_transaction(None, program)  # type: ignore[arg-type]
    except (AttributeError, RuntimeError):
        null_carrier_rejected = True
    else:
        null_carrier_rejected = False

    undermerge = SecantCarrier.create(backend.Algebra("F103", modulus=103, root=72))
    forward(undermerge, program)
    apply_sequence(undermerge, program.module_b, inverse=True)
    apply_sequence(undermerge, program.module_a, inverse=True)
    try:
        apply_coupling(undermerge, program, inverse=True, canonical_merge=False)
    except RuntimeError:
        missing_exact_merge_rejected = True
    else:
        missing_exact_merge_rejected = False

    rank2 = SecantCarrier.create(backend.Algebra("F103", modulus=103, root=72))
    load_seed(rank2)
    apply_coupling(rank2, program)
    certificate = rank_two_certificate(rank2)

    return {
        "missing_inverse_detected": missing_inverse_detected,
        "wrong_coupling_inverse_detected": wrong_coupling_inverse_detected,
        "reordered_inverse_detected": reordered_inverse_detected,
        "wrong_program_ownership_rejected": wrong_owner_rejected,
        "premature_projection_rejected": premature_projection_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "missing_exact_merge_rejected": missing_exact_merge_rejected,
        "coupling_omission_changes_boundary": control_boundary(
            program, omit_coupling=True
        )
        != reference,
        "module_order_changes_boundary": control_boundary(
            program, swap_modules=True
        )
        != reference,
        "rank_one_chart_rejected_after_coupling": certificate[
            "rank_one_chart_rejected"
        ],
        "snapshot_command_available": False,
    }


def run() -> dict[str, Any]:
    exact = [run_case(k, "PRIMARY", backend.Algebra("Q_ZETA17")) for k in EXACT_K]
    structural = []
    for modulus, root in FINITE_FIELDS:
        for k in DECLARED_K:
            item = run_case(
                k,
                "PRIMARY",
                backend.Algebra(f"F{modulus}", modulus=modulus, root=root),
            )
            item["field"] = f"F{modulus}"
            structural.append(item)

    baselines = [
        compiled_four_coordinate_baseline(
            item, compile_program(item["k"], item["family"]), backend.Algebra("Q_ZETA17")
        )
        for item in exact
    ]
    for item in structural:
        modulus, root = next(
            pair for pair in FINITE_FIELDS if item["field"] == f"F{pair[0]}"
        )
        baselines.append(
            compiled_four_coordinate_baseline(
                item,
                compile_program(item["k"], item["family"]),
                backend.Algebra(item["field"], modulus=modulus, root=root),
            )
        )
    if not all(item["boundary_agreement"] for item in baselines):
        fail("compiled four-coordinate classical baseline disagrees")

    reuse_carrier = SecantCarrier.create(backend.Algebra("Q_ZETA17"))
    first = execute_transaction(reuse_carrier, compile_program(8, "PRIMARY"))
    backing = reuse_carrier.backing_identity()
    reused = execute_transaction(reuse_carrier, compile_program(16, "REUSE"))
    fresh = run_case(16, "REUSE", backend.Algebra("Q_ZETA17"))
    if reused["boundary"] != fresh["boundary"]:
        fail("restored secant carrier disagrees with fresh unrelated program")
    if resource_signature(reused) != resource_signature(fresh):
        fail("restored secant carrier changed the resource signature")

    control_results = controls()
    if not all(
        value for key, value in control_results.items() if key != "snapshot_command_available"
    ) or control_results["snapshot_command_available"]:
        fail("one or more secant controls failed")

    return {
        "schema": "CAT_CAS_F17_COHERENT_RANK2_SECANT_PHASE_COUPLING_CLOSURE_V1",
        "claim": "BOUNDED_EXACT_F17_INVOLUTIVE_COHERENT_SUPERPOSITION_COUPLING_EXPANDS_ONE_RANK1_VERONESE_PHASE_STATE_TO_A_RANK2_SECANT_SHARED_BY_TWO_NONCOMMUTING_MODULES_WITH_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_A_COMPILED_FOUR_DYNAMIC_SCALAR_CLASSICAL_RECURRENCE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_scope": {
            "base_obstruction": "M129_RANK1_CHART_EXCLUDES_NONINTEGRABLE_SUPERPOSITION",
            "chart": "RANK2_COHERENT_VERONESE_SECANT",
            "coupling": "I_PLUS_ETA_SWAP_MODE0_MODE1",
            "inverse": "I_MINUS_ETA_R_DIVIDED_BY_ONE_MINUS_ETA_SQUARED",
            "declared_degrees": DECLARED_K,
            "exact_q_zeta17_degrees": EXACT_K,
            "dual_field_degrees": DECLARED_K,
            "consumer_modules": 2,
            "primitives_per_consumer": 20,
        },
        "exact_transactions": exact,
        "dual_field_structural_transactions": structural,
        "compiled_four_dynamic_scalar_classical_baselines": baselines,
        "reuse": {
            "first_k": 8,
            "reused_k": 16,
            "first_family": "PRIMARY",
            "reused_family": "REUSE",
            "first_boundary": first["boundary"],
            "reused_boundary": reused["boundary"],
            "fresh_boundary": fresh["boundary"],
            "fresh_restored_boundary_agreement": reused["boundary"]
            == fresh["boundary"],
            "fresh_restored_resource_signature_agreement": resource_signature(reused)
            == resource_signature(fresh),
            "same_actual_backing_across_unrelated_programs": (
                first["same_backing"]
                and reused["same_backing"]
                and reuse_carrier.backing_identity() == backing
            ),
            "generation_after_two_transactions": reuse_carrier.generation,
            "baseline_reload_used": False,
        },
        "controls": control_results,
        "resource_law": {
            "accepted_resident_phase_field_cells": 36,
            "resident_chart_rank": 2,
            "maximum_inverse_coupling_transient_components": 4,
            "inverse_coupling_transient_field_cells": 72,
            "accepted_path_occupation_coordinates": 0,
            "accepted_path_matching_or_assignment_expansion": 0,
            "accepted_path_dense_operator_cells": 0,
            "inverse_history_cells": 0,
            "hashlib_internal_state_excluded": True,
            "python_container_native_bigint_and_whole_process_memory_excluded": True,
            "full_exact_bit_complexity_established": False,
        },
        "matched_baseline": {
            "strongest_declared_warm": "COMPILED_FOUR_TOTAL_FOLDED_ENDPOINT_SCALARS",
            "descriptor_runtime": "IDENTICAL_TWO_COMPONENT_36_FIELD_CELL_SECANT_RECURRENCE",
            "phase_advantage_over_matched_classical": False,
        },
        "restoration": {
            "resident_rank2_secant_phase_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "transient_four_component_inverse_merge": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "single_involutive_superposition_coupling_only": True,
            "fixed_two_consumer_public_word_only": True,
            "declared_k_values_only": DECLARED_K,
            "repeated_coupling_rank_law_established": False,
            "m127_grid_orbit_shear_closed": False,
            "general_secant_rank_reduction_established": False,
            "gaussian_chart_closure_established": False,
            "catvm_custody_established": False,
            "distinct_phase_resource_established": False,
            "computational_advantage_established": False,
            "small_wall_crossing_established": False,
            "physical_waveform_execution_established": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation_established": False,
        },
        "next_obstruction": "ONE_INVOLUTIVE_SUPERPOSITION_BROADENS_RANK1_TO_RANK2_BUT_THE_SECANT_RECURRENCE_IS_CLASSICALLY_IDENTICAL_FIXED_WORDS_COMPILE_TO_FOUR_TOTAL_FOLDED_ENDPOINT_SCALARS_AND_A_SECOND_COUPLING_GENERATES_FOUR_DISTINCT_TERMS_WITHOUT_YET_PROVING_MINIMAL_SECANT_RANK_ABOVE_TWO",
        "next_experiment": "EXACT_SECOND_NONCOMMUTING_COUPLING_CATALECTICANT_RANK_LOWER_BOUND_OR_GAUSSIAN_CLOSURE_DIAGNOSTIC_WITHOUT_COMPONENT_ASSIGNMENT_ENUMERATION",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    arguments = parser.parse_args()
    result = run()
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
