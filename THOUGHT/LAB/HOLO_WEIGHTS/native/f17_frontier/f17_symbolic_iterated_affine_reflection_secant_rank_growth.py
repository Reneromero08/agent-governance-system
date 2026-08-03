#!/usr/bin/env python3
"""Exact iterated affine-reflection secant-rank growth diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import f17_coherent_veronese_phase_chart_closure as rank1
import f17_nonlinear_canonical_mps_separator_chart as backend


MODE_COUNT = 17
DECLARED_M = (1, 2, 3, 4, 5, 6)
FAMILIES = ("PRIMARY", "REUSE")
FINITE_FIELDS = ((103, 72), (137, 16))
FINAL_BOUNDARY = "K_MINUS_1_MODE0_ONE_MODE1_OCCUPATION"


def fail(message: str) -> None:
    raise RuntimeError(message)


def negative(alg: backend.Algebra, value: Any) -> Any:
    return alg.sub(alg.zero, value)


def algebra_signature(alg: backend.Algebra) -> str:
    return rank1.algebra_signature(alg)


def eta_exponents(m: int, family: str) -> tuple[int, ...]:
    if family == "PRIMARY":
        return tuple(2 * level + 1 for level in range(1, m + 1))
    if family == "REUSE":
        return tuple(2 * level + 2 for level in range(1, m + 1))
    fail("iterated coupling family changed")


@dataclass(frozen=True)
class IteratedProgram:
    m: int
    k: int
    family: str
    centers_twice: tuple[int, ...]
    eta_exponents: tuple[int, ...]
    final_boundary: str = FINAL_BOUNDARY

    @property
    def target_rank(self) -> int:
        return 1 << self.m

    def fingerprint(self) -> str:
        return rank1.digest_json(public_program_descriptor(self))


def compile_program(m: int, family: str) -> IteratedProgram:
    if m not in DECLARED_M or family not in FAMILIES:
        fail("iterated coupling program identity changed")
    rank = 1 << m
    program = IteratedProgram(
        m=m,
        k=2 * rank - 2,
        family=family,
        centers_twice=tuple((1 << level) - 1 for level in range(1, m + 1)),
        eta_exponents=eta_exponents(m, family),
    )
    validate_program(program)
    return program


def validate_program(program: IteratedProgram) -> None:
    if program.m not in DECLARED_M or program.family not in FAMILIES:
        fail("iterated coupling program domain changed")
    if program.k != 2 * (1 << program.m) - 2:
        fail("catalecticant degree threshold changed")
    if program.centers_twice != tuple(
        (1 << level) - 1 for level in range(1, program.m + 1)
    ):
        fail("public affine-reflection centers changed")
    if program.eta_exponents != eta_exponents(program.m, program.family):
        fail("public coupling exponents changed")
    if any(exponent % 17 == 0 for exponent in program.eta_exponents):
        fail("coupling exponent became singular")
    if program.final_boundary != FINAL_BOUNDARY:
        fail("iterated coupling boundary changed")


def public_program_descriptor(program: IteratedProgram) -> dict[str, Any]:
    return {
        "m": program.m,
        "k": program.k,
        "family": program.family,
        "chart": "ITERATED_AFFINE_REFLECTION_COHERENT_SECANT",
        "mode_count": MODE_COUNT,
        "seed": "MODE0_RAISED_TO_K",
        "couplings": [
            {
                "level": level,
                "kind": "INVOLUTIVE_AFFINE_REFLECTION_SUPERPOSITION",
                "law": "I_PLUS_ETA_LEVEL_TIMES_SYM_K_R_LEVEL",
                "one_particle_action": "V0_FIXED_V1_TO_A_LEVEL_V0_MINUS_V1",
                "a_level": center,
                "eta_exponent": exponent,
            }
            for level, (center, exponent) in enumerate(
                zip(program.centers_twice, program.eta_exponents, strict=True),
                start=1,
            )
        ],
        "final_boundary": program.final_boundary,
    }


def lease(program: IteratedProgram, alg: backend.Algebra, capacity: int) -> str:
    return rank1.digest_json(
        {
            "program": program.fingerprint(),
            "algebra": algebra_signature(alg),
            "capacity": capacity,
            "carrier": "ITERATED_AFFINE_REFLECTION_SECANT",
        }
    )


@dataclass
class SecantCarrier:
    alg: backend.Algebra
    capacity: int
    weights: list[Any]
    vectors: list[list[Any]]
    active_rank: int = 0
    generation: int = 0
    lease: str | None = None
    active_m: int | None = None
    active_k: int | None = None
    stage: str = "RESTORED"
    projection_calls: int = 0
    maximum_active_rank: int = 0
    maximum_coupling_transient_components: int = 0
    maximum_resident_payload_bits: int = 0
    maximum_commitment_record_json_bytes: int = 0

    @classmethod
    def create(cls, alg: backend.Algebra, capacity: int) -> "SecantCarrier":
        if capacity < 1 or capacity & (capacity - 1):
            fail("carrier capacity must be a positive power of two")
        carrier = cls(
            alg=alg,
            capacity=capacity,
            weights=[alg.zero for _ in range(capacity)],
            vectors=[
                [alg.zero for _ in range(MODE_COUNT)] for _ in range(capacity)
            ],
        )
        carrier.observe()
        return carrier

    def backing_identity(self) -> tuple[int, ...]:
        return (id(self), id(self.weights), *(id(vector) for vector in self.vectors))

    def flat_values(self) -> list[Any]:
        values: list[Any] = []
        for slot in range(self.capacity):
            values.extend((self.weights[slot], *self.vectors[slot]))
        return values

    def exact_zero(self) -> bool:
        return (
            self.active_rank == 0
            and all(value == self.alg.zero for value in self.flat_values())
            and self.lease is None
            and self.active_m is None
            and self.active_k is None
            and self.stage == "RESTORED"
        )

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
                "active_m": self.active_m,
                "active_k": self.active_k,
                "stage": self.stage,
            }
        )

    def observe(self) -> None:
        self.maximum_active_rank = max(self.maximum_active_rank, self.active_rank)
        payload = sum(self.alg.payload_bits(value) for value in self.flat_values())
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits, payload
        )


def reset_observation(carrier: SecantCarrier) -> None:
    carrier.projection_calls = 0
    carrier.maximum_active_rank = carrier.active_rank
    carrier.maximum_coupling_transient_components = 0
    carrier.maximum_resident_payload_bits = 0
    carrier.maximum_commitment_record_json_bytes = 0
    carrier.observe()


def clear_slots(carrier: SecantCarrier) -> None:
    for slot in range(carrier.capacity):
        carrier.weights[slot] = carrier.alg.zero
        for mode in range(MODE_COUNT):
            carrier.vectors[slot][mode] = carrier.alg.zero
    carrier.active_rank = 0
    carrier.observe()


def load_seed(carrier: SecantCarrier) -> None:
    if carrier.active_rank != 0:
        fail("seed load requires an empty carrier")
    carrier.weights[0] = carrier.alg.one
    carrier.vectors[0][0] = carrier.alg.one
    carrier.active_rank = 1
    carrier.observe()


def unload_seed(carrier: SecantCarrier) -> None:
    expected = [carrier.alg.one, *([carrier.alg.zero] * (MODE_COUNT - 1))]
    if (
        carrier.active_rank != 1
        or carrier.weights[0] != carrier.alg.one
        or carrier.vectors[0] != expected
        or any(carrier.weights[slot] != carrier.alg.zero for slot in range(1, carrier.capacity))
        or any(
            value != carrier.alg.zero
            for slot in range(1, carrier.capacity)
            for value in carrier.vectors[slot]
        )
    ):
        fail("inverse path did not return the exact seed")
    carrier.weights[0] = carrier.alg.zero
    carrier.vectors[0][0] = carrier.alg.zero
    carrier.active_rank = 0
    carrier.observe()


def affine_reflection(
    vector: list[Any], center_twice: int, alg: backend.Algebra
) -> list[Any]:
    result = list(vector)
    center = rank1.field_integer(alg, center_twice)
    result[1] = alg.sub(alg.mul(center, vector[0]), vector[1])
    return result


def combine_components(
    terms: list[tuple[Any, list[Any]]], alg: backend.Algebra
) -> list[tuple[Any, list[Any]]]:
    combined: list[tuple[Any, list[Any]]] = []
    for weight, vector in terms:
        if weight == alg.zero:
            continue
        for index, (prior_weight, prior_vector) in enumerate(combined):
            if vector == prior_vector:
                combined[index] = (alg.add(prior_weight, weight), prior_vector)
                break
        else:
            combined.append((weight, list(vector)))
    return [(weight, vector) for weight, vector in combined if weight != alg.zero]


def assign_components(
    carrier: SecantCarrier, components: list[tuple[Any, list[Any]]]
) -> None:
    if len(components) > carrier.capacity:
        fail("secant carrier capacity exceeded")
    clear_slots(carrier)
    for slot, (weight, vector) in enumerate(components):
        carrier.weights[slot] = weight
        carrier.vectors[slot][:] = vector
    carrier.active_rank = len(components)
    carrier.observe()


def apply_coupling(
    carrier: SecantCarrier,
    center_twice: int,
    eta_exponent: int,
    *,
    expected_input_rank: int,
    expected_output_rank: int,
    inverse: bool = False,
) -> None:
    if carrier.active_rank != expected_input_rank:
        fail("iterated coupling input rank changed")
    eta = carrier.alg.power(eta_exponent)
    denominator = carrier.alg.sub(carrier.alg.one, carrier.alg.mul(eta, eta))
    if denominator == carrier.alg.zero:
        fail("iterated coupling inverse denominator vanished")
    components = [
        (carrier.weights[slot], list(carrier.vectors[slot]))
        for slot in range(carrier.active_rank)
    ]
    terms: list[tuple[Any, list[Any]]] = []
    if inverse:
        scale = carrier.alg.inverse(denominator)
        for weight, vector in components:
            terms.append((carrier.alg.mul(weight, scale), vector))
            terms.append(
                (
                    negative(
                        carrier.alg,
                        carrier.alg.mul(carrier.alg.mul(weight, eta), scale),
                    ),
                    affine_reflection(vector, center_twice, carrier.alg),
                )
            )
    else:
        for weight, vector in components:
            terms.append((weight, vector))
            terms.append(
                (
                    carrier.alg.mul(weight, eta),
                    affine_reflection(vector, center_twice, carrier.alg),
                )
            )
    carrier.maximum_coupling_transient_components = max(
        carrier.maximum_coupling_transient_components, len(terms)
    )
    components = combine_components(terms, carrier.alg)
    if len(components) != expected_output_rank:
        fail("iterated coupling output rank changed")
    assign_components(carrier, components)


def forward(carrier: SecantCarrier, program: IteratedProgram) -> None:
    if not isinstance(carrier, SecantCarrier) or not carrier.exact_zero():
        fail("null, leased, or unrestored iterated carrier")
    if carrier.capacity < program.target_rank:
        fail("borrowed carrier is too small")
    validate_program(program)
    carrier.lease = lease(program, carrier.alg, carrier.capacity)
    carrier.active_m = program.m
    carrier.active_k = program.k
    carrier.stage = "FORWARD_ACTIVE"
    load_seed(carrier)
    for level, (center, exponent) in enumerate(
        zip(program.centers_twice, program.eta_exponents, strict=True), start=1
    ):
        apply_coupling(
            carrier,
            center,
            exponent,
            expected_input_rank=1 << (level - 1),
            expected_output_rank=1 << level,
        )
        carrier.stage = f"AFTER_COUPLING_{level}_HIDDEN"
    carrier.stage = "FORWARD_COMPLETE"


def project_boundary(carrier: SecantCarrier, program: IteratedProgram) -> Any:
    if (
        carrier.stage != "FORWARD_COMPLETE"
        or carrier.lease != lease(program, carrier.alg, carrier.capacity)
        or carrier.active_rank != program.target_rank
    ):
        fail("only the final owned iterated boundary may be projected")
    carrier.projection_calls += 1
    return raw_boundary(carrier, program.k)


def raw_boundary(carrier: SecantCarrier, k: int) -> Any:
    result = carrier.alg.zero
    k_value = rank1.field_integer(carrier.alg, k)
    for slot in range(carrier.active_rank):
        vector = carrier.vectors[slot]
        contribution = carrier.alg.mul(
            carrier.weights[slot],
            carrier.alg.mul(
                k_value,
                carrier.alg.mul(
                    rank1.scalar_power(carrier.alg, vector[0], k - 1),
                    vector[1],
                ),
            ),
        )
        result = carrier.alg.add(result, contribution)
    return result


def inverse(carrier: SecantCarrier, program: IteratedProgram) -> None:
    if (
        carrier.stage != "FORWARD_COMPLETE"
        or carrier.lease != lease(program, carrier.alg, carrier.capacity)
        or carrier.active_rank != program.target_rank
    ):
        fail("inverse program does not own the iterated carrier")
    carrier.stage = "INVERSE_ACTIVE"
    for level in range(program.m, 0, -1):
        apply_coupling(
            carrier,
            program.centers_twice[level - 1],
            program.eta_exponents[level - 1],
            expected_input_rank=1 << level,
            expected_output_rank=1 << (level - 1),
            inverse=True,
        )
    unload_seed(carrier)
    carrier.lease = None
    carrier.active_m = None
    carrier.active_k = None
    carrier.stage = "RESTORED"
    carrier.generation += 1
    carrier.observe()
    if not carrier.exact_zero():
        fail("iterated inverse failed exact restoration")


def state_commitment(carrier: SecantCarrier) -> tuple[str, int]:
    active: list[Any] = []
    for slot in range(carrier.active_rank):
        active.extend((carrier.weights[slot], *carrier.vectors[slot]))
    return rank1.stream_vector_commitment(active, carrier.alg)


def point_coordinate(vector: list[Any], alg: backend.Algebra) -> Any:
    if vector[0] != alg.one or any(value != alg.zero for value in vector[2:]):
        fail("declared binary moment-curve chart changed")
    return vector[1]


def symbolic_rank_certificate(
    carrier: SecantCarrier, program: IteratedProgram
) -> dict[str, Any]:
    if carrier.active_rank != program.target_rank:
        fail("rank certificate requires the final hidden secant")
    expected = [
        rank1.field_integer(carrier.alg, value)
        for value in range(program.target_rank)
    ]
    observed = [
        point_coordinate(carrier.vectors[slot], carrier.alg)
        for slot in range(carrier.active_rank)
    ]
    observed_keys = {
        json.dumps(carrier.alg.serialize(value), separators=(",", ":"))
        for value in observed
    }
    expected_keys = {
        json.dumps(carrier.alg.serialize(value), separators=(",", ":"))
        for value in expected
    }
    distinct = len(observed_keys) == program.target_rank
    public_range = observed_keys == expected_keys
    weights_nonzero = all(
        carrier.weights[slot] != carrier.alg.zero
        for slot in range(carrier.active_rank)
    )
    field_supports_distinct_points = (
        carrier.alg.modulus == 0 or program.target_rank <= carrier.alg.modulus
    )
    nonzero = distinct and public_range and weights_nonzero and field_supports_distinct_points
    return {
        "certificate": "MOMENT_HANKEL_EQUALS_V_DIAG_W_V_TRANSPOSE",
        "rank": program.target_rank if nonzero else None,
        "catalecticant_size": program.target_rank,
        "degree_threshold": 2 * program.target_rank - 2,
        "declared_degree_meets_threshold": program.k
        >= 2 * program.target_rank - 2,
        "public_point_set_is_integer_range_zero_to_rank_minus_one": public_range,
        "public_points_distinct_in_declared_algebra": distinct,
        "all_component_weights_nonzero": weights_nonzero,
        "factorization": "DET_EQUALS_PRODUCT_WEIGHTS_TIMES_VANDERMONDE_SQUARED",
        "factor_nonzero": nonzero,
        "exact_normalized_divided_power_secant_rank": (
            program.target_rank if nonzero else None
        ),
        "ordinary_symmetric_waring_rank_interpretation": (
            carrier.alg.kind == "Q_ZETA17"
        ),
        "finite_field_applicability_requires_rank_at_most_modulus": True,
        "accepted_transaction_materializes_catalecticant": False,
        "determinant_value_serialized": False,
        "intermediate_components_serialized": False,
    }


def execute_transaction(
    carrier: SecantCarrier, program: IteratedProgram
) -> dict[str, Any]:
    reset_observation(carrier)
    initial_digest = carrier.digest()
    backing_before = carrier.backing_identity()
    generation_before = carrier.generation
    forward(carrier, program)
    commitment, record_bytes = state_commitment(carrier)
    carrier.maximum_commitment_record_json_bytes = record_bytes
    boundary = project_boundary(carrier, program)
    inverse(carrier, program)
    descriptor = public_program_descriptor(program)
    return {
        "m": program.m,
        "k": program.k,
        "family": program.family,
        "algebra": algebra_signature(carrier.alg),
        "program_fingerprint": program.fingerprint(),
        "boundary": carrier.alg.serialize(boundary),
        "forward_secant_commitment": commitment,
        "restored_exact_zero": carrier.exact_zero(),
        "same_backing": backing_before == carrier.backing_identity(),
        "generation_before": generation_before,
        "generation_after": carrier.generation,
        "target_rank": program.target_rank,
        "allocated_carrier_capacity": carrier.capacity,
        "allocated_phase_field_cells": carrier.capacity * (MODE_COUNT + 1),
        "active_phase_field_cells": program.target_rank * (MODE_COUNT + 1),
        "maximum_active_rank": carrier.maximum_active_rank,
        "maximum_coupling_transient_components": carrier.maximum_coupling_transient_components,
        "maximum_coupling_transient_field_cells": (
            carrier.maximum_coupling_transient_components * (MODE_COUNT + 1)
        ),
        "public_program_integer_cells": 4 + 3 * program.m,
        "public_program_json_bytes": len(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        ),
        "maximum_resident_payload_bits": carrier.maximum_resident_payload_bits,
        "maximum_commitment_record_json_bytes": carrier.maximum_commitment_record_json_bytes,
        "final_boundary_field_cells": 1,
        "final_boundary_payload_bits": carrier.alg.payload_bits(boundary),
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
        "accepted_path_catalecticant_materialized": False,
        "accepted_path_occupation_vector_materialized": False,
        "accepted_path_explicit_coherent_component_enumeration": True,
        "accepted_path_resident_coherent_components": program.target_rank,
        "accepted_path_separate_truth_table_or_assignment_buffer_materialized": False,
        "accepted_path_dense_operator_materialized": False,
        "intermediate_components_serialized_to_controller": False,
    }


def run_case(m: int, family: str, alg: backend.Algebra) -> dict[str, Any]:
    program = compile_program(m, family)
    return execute_transaction(SecantCarrier.create(alg, program.target_rank), program)


def certificate_case(m: int, alg: backend.Algebra) -> dict[str, Any]:
    program = compile_program(m, "PRIMARY")
    carrier = SecantCarrier.create(alg, program.target_rank)
    forward(carrier, program)
    certificate = symbolic_rank_certificate(carrier, program)
    inverse(carrier, program)
    return {
        "m": m,
        "k": program.k,
        "algebra": algebra_signature(alg),
        **certificate,
        "verification_carrier_restored_exact_zero": carrier.exact_zero(),
        "verification_only_not_accepted_transaction_path": True,
    }


def compiled_two_moment_baseline(
    transaction: dict[str, Any], program: IteratedProgram, alg: backend.Algebra
) -> dict[str, Any]:
    total_weight = alg.one
    first_moment = alg.zero
    for center_twice, exponent in zip(
        program.centers_twice, program.eta_exponents, strict=True
    ):
        eta = alg.power(exponent)
        prior_weight = total_weight
        prior_moment = first_moment
        total_weight = alg.mul(alg.add(alg.one, eta), prior_weight)
        first_moment = alg.add(
            alg.mul(alg.sub(alg.one, eta), prior_moment),
            alg.mul(
                alg.mul(eta, rank1.field_integer(alg, center_twice)),
                prior_weight,
            ),
        )
    boundary = alg.mul(rank1.field_integer(alg, program.k), first_moment)
    retained = [total_weight, first_moment]
    commitment, record_bytes = rank1.stream_vector_commitment(retained, alg)
    return {
        "m": program.m,
        "k": program.k,
        "family": program.family,
        "algebra": algebra_signature(alg),
        "boundary_agreement": alg.serialize(boundary) == transaction["boundary"],
        "descriptor_runtime_dynamic_field_cells": 2,
        "full_binary_state_triangular_moment_field_cells": program.k + 1,
        "sealed_word_final_boundary_field_cells": 1,
        "warm_named_transient_field_cells": 4,
        "moment_commitment": commitment,
        "maximum_commitment_record_json_bytes": record_bytes,
        "phase_carrier_or_snapshot_used": False,
    }


def compiled_atomic_weight_baseline(
    transaction: dict[str, Any], program: IteratedProgram, alg: backend.Algebra
) -> dict[str, Any]:
    """Strong compact full-state recurrence on public integer support."""

    weights = [alg.one]
    maximum_named_field_cells = 1
    for center, exponent in zip(
        program.centers_twice, program.eta_exponents, strict=True
    ):
        eta = alg.power(exponent)
        previous = weights
        updated = [alg.zero for _ in range(2 * len(previous))]
        for point, weight in enumerate(previous):
            updated[point] = weight
            updated[center - point] = alg.mul(eta, weight)
        maximum_named_field_cells = max(
            maximum_named_field_cells, len(previous) + len(updated)
        )
        weights = updated
    first_moment = alg.zero
    for point, weight in enumerate(weights):
        first_moment = alg.add(
            first_moment,
            alg.mul(rank1.field_integer(alg, point), weight),
        )
    boundary = alg.mul(rank1.field_integer(alg, program.k), first_moment)
    commitment, record_bytes = rank1.stream_vector_commitment(weights, alg)
    return {
        "m": program.m,
        "k": program.k,
        "family": program.family,
        "algebra": algebra_signature(alg),
        "boundary_agreement": alg.serialize(boundary) == transaction["boundary"],
        "resident_atomic_weight_field_cells": len(weights),
        "public_support": "INTEGER_RANGE_ZERO_TO_TWO_TO_THE_M_MINUS_ONE",
        "public_support_field_cells_retained": 0,
        "maximum_named_field_cells_including_update_buffer": maximum_named_field_cells,
        "explicit_weight_enumeration": True,
        "weight_commitment": commitment,
        "maximum_commitment_record_json_bytes": record_bytes,
        "phase_carrier_or_snapshot_used": False,
    }


def resource_signature(transaction: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "boundary",
        "forward_secant_commitment",
        "generation_before",
        "generation_after",
        "initial_digest",
        "restored_digest_with_generation",
        "program_fingerprint",
        "family",
    }
    return {key: value for key, value in transaction.items() if key not in excluded}


def controls() -> dict[str, Any]:
    alg = backend.Algebra("F103", modulus=103, root=72)
    program = compile_program(4, "PRIMARY")
    reference_carrier = SecantCarrier.create(alg, program.target_rank)
    forward(reference_carrier, program)
    reference_boundary = project_boundary(reference_carrier, program)

    omitted_carrier = SecantCarrier.create(
        backend.Algebra("F103", modulus=103, root=72), program.target_rank
    )
    load_seed(omitted_carrier)
    for level in range(1, program.m):
        apply_coupling(
            omitted_carrier,
            program.centers_twice[level - 1],
            program.eta_exponents[level - 1],
            expected_input_rank=1 << (level - 1),
            expected_output_rank=1 << level,
        )
    omitted_boundary = raw_boundary(omitted_carrier, program.k)

    premature = SecantCarrier.create(
        backend.Algebra("F103", modulus=103, root=72), program.target_rank
    )
    premature.lease = lease(program, premature.alg, premature.capacity)
    premature.active_m = program.m
    premature.active_k = program.k
    premature.stage = "FORWARD_ACTIVE"
    load_seed(premature)
    try:
        project_boundary(premature, program)
    except RuntimeError:
        premature_rejected = True
    else:
        premature_rejected = False

    wrong = SecantCarrier.create(
        backend.Algebra("F103", modulus=103, root=72), program.target_rank
    )
    forward(wrong, program)
    try:
        apply_coupling(
            wrong,
            program.centers_twice[-1],
            program.eta_exponents[-1] + 1,
            expected_input_rank=program.target_rank,
            expected_output_rank=program.target_rank // 2,
            inverse=True,
        )
    except RuntimeError:
        wrong_inverse_detected = True
    else:
        wrong_inverse_detected = wrong.active_rank != program.target_rank // 2

    reordered = SecantCarrier.create(
        backend.Algebra("F103", modulus=103, root=72), program.target_rank
    )
    forward(reordered, program)
    try:
        apply_coupling(
            reordered,
            program.centers_twice[0],
            program.eta_exponents[0],
            expected_input_rank=program.target_rank,
            expected_output_rank=program.target_rank // 2,
            inverse=True,
        )
    except RuntimeError:
        reordered_inverse_detected = True
    else:
        reordered_inverse_detected = reordered.active_rank != program.target_rank // 2

    owner = SecantCarrier.create(
        backend.Algebra("F103", modulus=103, root=72), program.target_rank
    )
    forward(owner, program)
    try:
        inverse(owner, compile_program(4, "REUSE"))
    except RuntimeError:
        wrong_owner_rejected = True
    else:
        wrong_owner_rejected = False

    first_vector = [alg.one, alg.zero, *([alg.zero] * (MODE_COUNT - 2))]
    r1r2 = affine_reflection(affine_reflection(first_vector, 3, alg), 1, alg)
    r2r1 = affine_reflection(affine_reflection(first_vector, 1, alg), 3, alg)
    certificate = symbolic_rank_certificate(reference_carrier, program)
    try:
        execute_transaction(None, program)  # type: ignore[arg-type]
    except (AttributeError, RuntimeError):
        null_carrier_rejected = True
    else:
        null_carrier_rejected = False
    return {
        "missing_inverse_detected": not reference_carrier.exact_zero(),
        "wrong_inverse_detected": wrong_inverse_detected,
        "reordered_inverse_detected": reordered_inverse_detected,
        "wrong_program_ownership_rejected": wrong_owner_rejected,
        "premature_projection_rejected": premature_rejected,
        "last_coupling_omission_changes_boundary": omitted_boundary
        != reference_boundary,
        "distinct_affine_reflections_noncommute": r1r2 != r2r1,
        "rank16_certificate_nonzero": certificate["factor_nonzero"],
        "rank16_capacity_required": certificate[
            "exact_normalized_divided_power_secant_rank"
        ]
        == 16,
        "f103_m7_distinct_point_applicability_gate_rejects_rank128": (1 << 7)
        > 103,
        "null_carrier_rejected": null_carrier_rejected,
        "snapshot_command_available": False,
    }


def run() -> dict[str, Any]:
    exact = [run_case(m, "PRIMARY", backend.Algebra("Q_ZETA17")) for m in DECLARED_M]
    structural = []
    for modulus, root in FINITE_FIELDS:
        for m in DECLARED_M:
            item = run_case(
                m,
                "PRIMARY",
                backend.Algebra(f"F{modulus}", modulus=modulus, root=root),
            )
            item["field"] = f"F{modulus}"
            structural.append(item)

    certificates = [
        certificate_case(m, backend.Algebra("Q_ZETA17")) for m in DECLARED_M
    ]
    for modulus, root in FINITE_FIELDS:
        certificates.extend(
            certificate_case(
                m, backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
            )
            for m in DECLARED_M
        )
    if not all(
        item["factor_nonzero"]
        and item["exact_normalized_divided_power_secant_rank"] == 1 << item["m"]
        and item["verification_carrier_restored_exact_zero"]
        for item in certificates
    ):
        fail("one or more symbolic rank-growth certificates failed")

    moment_baselines = []
    weight_baselines = []
    for item in exact:
        program = compile_program(item["m"], item["family"])
        alg = backend.Algebra("Q_ZETA17")
        moment_baselines.append(
            compiled_two_moment_baseline(
                item, program, alg,
            )
        )
        weight_baselines.append(compiled_atomic_weight_baseline(item, program, alg))
    for item in structural:
        modulus, root = next(
            pair for pair in FINITE_FIELDS if item["field"] == f"F{pair[0]}"
        )
        program = compile_program(item["m"], item["family"])
        alg = backend.Algebra(item["field"], modulus=modulus, root=root)
        moment_baselines.append(
            compiled_two_moment_baseline(
                item, program, alg,
            )
        )
        weight_baselines.append(compiled_atomic_weight_baseline(item, program, alg))
    if not all(item["boundary_agreement"] for item in moment_baselines):
        fail("compiled two-moment baseline disagrees")
    if not all(item["boundary_agreement"] for item in weight_baselines):
        fail("compiled atomic-weight baseline disagrees")

    reuse_carrier = SecantCarrier.create(backend.Algebra("Q_ZETA17"), 16)
    first = execute_transaction(reuse_carrier, compile_program(3, "PRIMARY"))
    backing = reuse_carrier.backing_identity()
    reused = execute_transaction(reuse_carrier, compile_program(4, "REUSE"))
    fresh = run_case(4, "REUSE", backend.Algebra("Q_ZETA17"))
    if reused["boundary"] != fresh["boundary"]:
        fail("restored iterated carrier disagrees with fresh reuse")
    if resource_signature(reused) != resource_signature(fresh):
        fail("restored iterated carrier changed resource signature")

    control_results = controls()
    if not all(
        value for key, value in control_results.items() if key != "snapshot_command_available"
    ) or control_results["snapshot_command_available"]:
        fail("one or more iterated coupling controls failed")

    return {
        "schema": "CAT_CAS_F17_SYMBOLIC_ITERATED_AFFINE_REFLECTION_SECANT_RANK_GROWTH_V1",
        "claim": "BOUNDED_EXECUTION_AND_EXACT_SYMBOLIC_VANDERMONDE_CERTIFICATE_FOR_ITERATED_NONCOMMUTING_AFFINE_REFLECTION_SUPERPOSITION_COUPLING_SECANT_RANK_TWO_TO_THE_M_GROWTH_WITH_EXACT_RESTORATION_AND_REUSE_BUT_EXPLICIT_EXPONENTIAL_PHASE_COMPONENT_ENUMERATION_AND_MATCHED_R_WEIGHT_FULL_STATE_OR_TWO_SCALAR_FINAL_BOUNDARY_CLASSICAL_RECURRENCES",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_scope": {
            "executed_m": DECLARED_M,
            "executed_ranks": [1 << m for m in DECLARED_M],
            "executed_degrees": [2 * (1 << m) - 2 for m in DECLARED_M],
            "analytic_family": "A_LEVEL_EQUALS_TWO_TO_THE_LEVEL_MINUS_ONE",
            "analytic_point_set": "ZERO_THROUGH_TWO_TO_THE_M_MINUS_ONE",
            "analytic_catalecticant_factorization": "V_DIAG_W_V_TRANSPOSE",
            "ordinary_waring_interpretation": "Q_ZETA17_ONLY",
            "finite_field_applicability": "TWO_TO_THE_M_AT_MOST_FIELD_MODULUS",
        },
        "exact_transactions": exact,
        "dual_field_structural_transactions": structural,
        "symbolic_rank_certificates": certificates,
        "compiled_two_moment_classical_baselines": moment_baselines,
        "compiled_atomic_weight_classical_baselines": weight_baselines,
        "reuse": {
            "first_m": 3,
            "reused_m": 4,
            "first_family": "PRIMARY",
            "reused_family": "REUSE",
            "fresh_restored_boundary_agreement": reused["boundary"] == fresh["boundary"],
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
            "rank_at_m": "TWO_TO_THE_M",
            "active_phase_field_cells_at_m": "18_TIMES_TWO_TO_THE_M",
            "maximum_inverse_transient_field_cells_at_m": "36_TIMES_TWO_TO_THE_M",
            "accepted_path_explicit_coherent_components_at_m": "TWO_TO_THE_M",
            "strongest_compact_classical_atomic_weight_field_cells_at_m": "TWO_TO_THE_M",
            "matched_full_binary_moment_field_cells_at_m": "TWO_TIMES_TWO_TO_THE_M_MINUS_ONE",
            "accepted_path_occupation_coordinates": 0,
            "accepted_path_catalecticant_cells": 0,
            "accepted_path_separate_truth_table_or_assignment_buffer_cells": 0,
            "inverse_history_cells": 0,
            "full_exact_bit_complexity_established": False,
            "python_container_allocator_native_bigint_hashlib_and_whole_process_excluded": True,
        },
        "matched_baseline": {
            "strongest_descriptor_runtime": "TWO_DYNAMIC_MOMENT_SCALARS",
            "strongest_sealed_word_warm": "ONE_FINAL_BOUNDARY_SCALAR",
            "strongest_compact_full_state": "TWO_TO_THE_M_ATOMIC_WEIGHTS_ON_PUBLIC_SUPPORT",
            "independent_dense_moment_full_state": "TWO_TIMES_TWO_TO_THE_M_MINUS_ONE_TRIANGULAR_MOMENTS",
            "phase_advantage_over_matched_classical": False,
        },
        "restoration": {
            "resident_secant_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "transient_inverse_merge": "NO_RESTORATION_CLAIM",
            "verification_only_certificate_work": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "bounded_execution_m1_to6": True,
            "analytic_rank_law_for_declared_affine_reflection_family": True,
            "arbitrary_interleaved_coupling_rank_law": False,
            "fixed_rank_closure": False,
            "general_gaussian_closure": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_execution": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
        "next_obstruction": "THE_DECLARED_NONCOMMUTING_AFFINE_REFLECTION_COUPLING_FAMILY_HAS_AN_EXACT_TWO_TO_THE_M_SECANT_RANK_LAW_AND_REQUIRES_EXPONENTIAL_EXPLICIT_PHASE_COMPONENTS_WHILE_ITS_FINAL_BOUNDARY_COLLAPSES_TO_TWO_CLASSICAL_MOMENTS",
        "next_experiment": "EXACT_FIXED_RANK_GAUSSIAN_OR_STABILIZER_PHASE_CHART_FOR_NONCOMMUTING_SUPERPOSITION_COUPLINGS_OR_A_TRANSFERABLE_NO_GO_THAT_FORCES_A_DIFFERENT_NATIVE_PHASE_RESOURCE",
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
