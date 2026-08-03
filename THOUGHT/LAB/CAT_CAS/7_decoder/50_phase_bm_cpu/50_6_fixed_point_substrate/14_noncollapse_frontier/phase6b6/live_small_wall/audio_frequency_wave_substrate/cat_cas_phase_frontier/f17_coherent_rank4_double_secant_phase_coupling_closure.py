#!/usr/bin/env python3
"""Exact four-component closure for two interleaved coherent couplings."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import f17_coherent_rank2_secant_phase_coupling_closure as rank2
import f17_coherent_veronese_phase_chart_closure as rank1
import f17_nonlinear_canonical_mps_separator_chart as backend


MODE_COUNT = 17
SLOT_COUNT = 4
DECLARED_K = (4, 8, 16, 32, 64, 128)
EXACT_K = (4, 8, 16, 32)
FINITE_FIELDS = ((103, 72), (137, 16))
FAMILIES = ("PRIMARY", "REUSE")
FINAL_BOUNDARY = "K_MINUS_1_MODE0_ONE_MODE1_OCCUPATION"
CATALYTICANT_COLUMN_PAIRS = ((16, 16), (1, 16), (1, 1), (0, 16))


def fail(message: str) -> None:
    raise RuntimeError(message)


def negative(alg: backend.Algebra, value: Any) -> Any:
    return alg.sub(alg.zero, value)


def reflection(vector: list[Any]) -> list[Any]:
    result = list(vector)
    result[0], result[1] = result[1], result[0]
    return result


@dataclass(frozen=True)
class RankFourProgram:
    k: int
    family: str
    first_eta_exponent: int
    second_eta_exponent: int
    module_a: tuple[rank1.Primitive, ...]
    module_b: tuple[rank1.Primitive, ...]
    final_boundary: str = FINAL_BOUNDARY

    def fingerprint(self) -> str:
        return rank1.digest_json(public_program_descriptor(self))


def compile_program(k: int, family: str) -> RankFourProgram:
    predecessor = rank2.compile_program(k, family)
    program = RankFourProgram(
        k=k,
        family=family,
        first_eta_exponent=predecessor.eta_exponent,
        second_eta_exponent=7 if family == "PRIMARY" else 9,
        module_a=predecessor.module_a,
        module_b=predecessor.module_b,
    )
    validate_program(program)
    return program


def validate_program(program: RankFourProgram) -> None:
    if program.k not in DECLARED_K or program.family not in FAMILIES:
        fail("rank-four program identity changed")
    expected = (3, 7) if program.family == "PRIMARY" else (5, 9)
    if (program.first_eta_exponent, program.second_eta_exponent) != expected:
        fail("declared coupling exponents changed")
    if len(program.module_a) != 20 or len(program.module_b) != 20:
        fail("two-module split changed")
    for primitive in (*program.module_a, *program.module_b):
        rank1.validate_primitive(primitive)
    if program.final_boundary != FINAL_BOUNDARY:
        fail("rank-four boundary type changed")


def public_program_descriptor(program: RankFourProgram) -> dict[str, Any]:
    return {
        "k": program.k,
        "family": program.family,
        "chart": "RANK4_DOUBLE_COHERENT_VERONESE_SECANT",
        "mode_count": MODE_COUNT,
        "seed": "MODE0_RAISED_TO_K",
        "first_coupling": {
            "kind": "INVOLUTIVE_COHERENT_SUPERPOSITION",
            "law": "I_PLUS_ETA1_R",
            "eta_exponent": program.first_eta_exponent,
            "reflection": "SWAP_MODE0_MODE1",
        },
        "module_a": [item.as_json() for item in program.module_a],
        "second_coupling": {
            "kind": "INVOLUTIVE_COHERENT_SUPERPOSITION",
            "law": "I_PLUS_ETA2_R",
            "eta_exponent": program.second_eta_exponent,
            "reflection": "SWAP_MODE0_MODE1",
        },
        "module_b": [item.as_json() for item in program.module_b],
        "final_boundary": program.final_boundary,
    }


def algebra_signature(alg: backend.Algebra) -> str:
    return rank1.algebra_signature(alg)


def lease(program: RankFourProgram, alg: backend.Algebra) -> str:
    return rank1.digest_json(
        {
            "program": program.fingerprint(),
            "algebra": algebra_signature(alg),
            "carrier": "FOUR_SLOT_RANK4_DOUBLE_COHERENT_SECANT",
        }
    )


@dataclass
class RankFourCarrier:
    alg: backend.Algebra
    weights: list[Any]
    vectors: list[list[Any]]
    active_rank: int = 0
    generation: int = 0
    lease: str | None = None
    stage: str = "RESTORED"
    active_k: int | None = None
    projection_calls: int = 0
    maximum_resident_payload_bits: int = 0
    maximum_resident_value_payload_bits: int = 0
    maximum_resident_numerator_signed_bits: int = 0
    maximum_resident_denominator_bits: int = 0
    maximum_transient_value_payload_bits: int = 0
    maximum_transient_numerator_signed_bits: int = 0
    maximum_transient_denominator_bits: int = 0
    maximum_commitment_record_json_bytes: int = 0
    maximum_active_rank: int = 0
    maximum_coupling_transient_components: int = 0
    character_cell_multiplications: int = 0
    shear_cell_updates: int = 0

    @classmethod
    def create(cls, alg: backend.Algebra) -> "RankFourCarrier":
        carrier = cls(
            alg=alg,
            weights=[alg.zero for _ in range(SLOT_COUNT)],
            vectors=[
                [alg.zero for _ in range(MODE_COUNT)] for _ in range(SLOT_COUNT)
            ],
        )
        carrier.observe_resident()
        return carrier

    def backing_identity(self) -> tuple[int, ...]:
        return (id(self), id(self.weights), *(id(vector) for vector in self.vectors))

    def flat_values(self) -> list[Any]:
        result: list[Any] = []
        for slot in range(SLOT_COUNT):
            result.extend((self.weights[slot], *self.vectors[slot]))
        return result

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
        payload = sum(self.alg.payload_bits(value) for value in self.flat_values())
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits, payload
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


def reset_observation(carrier: RankFourCarrier) -> None:
    carrier.projection_calls = 0
    carrier.maximum_resident_payload_bits = 0
    carrier.maximum_resident_value_payload_bits = 0
    carrier.maximum_resident_numerator_signed_bits = 0
    carrier.maximum_resident_denominator_bits = 0
    carrier.maximum_transient_value_payload_bits = 0
    carrier.maximum_transient_numerator_signed_bits = 0
    carrier.maximum_transient_denominator_bits = 0
    carrier.maximum_commitment_record_json_bytes = 0
    carrier.maximum_active_rank = carrier.active_rank
    carrier.maximum_coupling_transient_components = 0
    carrier.character_cell_multiplications = 0
    carrier.shear_cell_updates = 0
    carrier.observe_resident()


def clear_slots(carrier: RankFourCarrier) -> None:
    for slot in range(SLOT_COUNT):
        carrier.write_weight(slot, carrier.alg.zero)
        for mode in range(MODE_COUNT):
            carrier.write_cell(slot, mode, carrier.alg.zero)
    carrier.active_rank = 0
    carrier.observe_resident()


def load_seed(carrier: RankFourCarrier) -> None:
    if carrier.active_rank != 0:
        fail("seed load requires an empty rank-four carrier")
    carrier.write_weight(0, carrier.alg.one)
    carrier.write_cell(0, 0, carrier.alg.one)
    carrier.active_rank = 1
    carrier.observe_resident()


def unload_seed(carrier: RankFourCarrier) -> None:
    expected = [carrier.alg.one, *([carrier.alg.zero] * (MODE_COUNT - 1))]
    if (
        carrier.active_rank != 1
        or carrier.weights[0] != carrier.alg.one
        or carrier.vectors[0] != expected
        or any(carrier.weights[slot] != carrier.alg.zero for slot in range(1, SLOT_COUNT))
        or any(
            value != carrier.alg.zero
            for slot in range(1, SLOT_COUNT)
            for value in carrier.vectors[slot]
        )
    ):
        fail("inverse path did not return the actual rank-four seed coordinates")
    carrier.write_weight(0, carrier.alg.sub(carrier.weights[0], carrier.alg.one))
    carrier.write_cell(0, 0, carrier.alg.sub(carrier.vectors[0][0], carrier.alg.one))
    carrier.active_rank = 0
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
                combined[index] = (alg.add(prior_weight, weight), prior_vector)
                break
        else:
            combined.append((weight, list(vector)))
    return [(weight, vector) for weight, vector in combined if weight != alg.zero]


def assign_components(
    carrier: RankFourCarrier, components: list[tuple[Any, list[Any]]]
) -> None:
    if len(components) > SLOT_COUNT:
        fail("rank-four carrier capacity exceeded")
    clear_slots(carrier)
    for slot, (weight, vector) in enumerate(components):
        carrier.write_weight(slot, weight)
        for mode, value in enumerate(vector):
            carrier.write_cell(slot, mode, value)
    carrier.active_rank = len(components)
    carrier.observe_resident()


def apply_coupling(
    carrier: RankFourCarrier,
    exponent: int,
    *,
    expected_input_rank: int,
    expected_output_rank: int,
    inverse: bool = False,
    canonical_merge: bool = True,
) -> None:
    if carrier.active_rank != expected_input_rank:
        fail("coupling input rank changed")
    eta = carrier.alg.power(exponent)
    denominator = carrier.alg.sub(carrier.alg.one, carrier.alg.mul(eta, eta))
    if denominator == carrier.alg.zero:
        fail("coupling inverse denominator vanished")
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
                    reflection(vector),
                )
            )
        if not canonical_merge:
            fail("inverse coupling requires exact duplicate cancellation")
        carrier.maximum_coupling_transient_components = max(
            carrier.maximum_coupling_transient_components, len(terms)
        )
        terms = combine_exact_components(terms, carrier.alg)
    else:
        for weight, vector in components:
            terms.extend(
                (
                    (weight, vector),
                    (carrier.alg.mul(weight, eta), reflection(vector)),
                )
            )
        carrier.maximum_coupling_transient_components = max(
            carrier.maximum_coupling_transient_components, len(terms)
        )
        terms = combine_exact_components(terms, carrier.alg)
    if len(terms) != expected_output_rank:
        fail("coupling output rank changed")
    for weight, vector in terms:
        carrier.observe_value(weight)
        for value in vector:
            carrier.observe_value(value)
    assign_components(carrier, terms)


def apply_sequence(
    carrier: RankFourCarrier,
    primitives: Iterable[rank1.Primitive],
    *,
    inverse: bool = False,
) -> None:
    rank2.apply_sequence(carrier, primitives, inverse=inverse)  # type: ignore[arg-type]


def forward(carrier: RankFourCarrier, program: RankFourProgram) -> None:
    if not isinstance(carrier, RankFourCarrier) or not carrier.exact_zero():
        fail("null, leased, or unrestored rank-four carrier")
    validate_program(program)
    carrier.lease = lease(program, carrier.alg)
    carrier.active_k = program.k
    carrier.stage = "FORWARD_ACTIVE"
    load_seed(carrier)
    apply_coupling(
        carrier,
        program.first_eta_exponent,
        expected_input_rank=1,
        expected_output_rank=2,
    )
    carrier.stage = "AFTER_FIRST_COUPLING_HIDDEN"
    apply_sequence(carrier, program.module_a)
    carrier.stage = "AFTER_MODULE_A_HIDDEN"
    apply_coupling(
        carrier,
        program.second_eta_exponent,
        expected_input_rank=2,
        expected_output_rank=4,
    )
    carrier.stage = "AFTER_SECOND_COUPLING_HIDDEN"
    apply_sequence(carrier, program.module_b)
    carrier.stage = "FORWARD_COMPLETE"


def component_boundary(
    alg: backend.Algebra, weight: Any, vector: list[Any], k: int
) -> Any:
    return alg.mul(
        weight,
        alg.mul(
            rank1.field_integer(alg, k),
            alg.mul(rank1.scalar_power(alg, vector[0], k - 1), vector[1]),
        ),
    )


def project_boundary(carrier: RankFourCarrier, program: RankFourProgram) -> Any:
    if (
        carrier.stage != "FORWARD_COMPLETE"
        or carrier.lease != lease(program, carrier.alg)
        or carrier.active_k != program.k
    ):
        fail("only the final owned rank-four boundary may be projected")
    carrier.projection_calls += 1
    result = carrier.alg.zero
    for slot in range(carrier.active_rank):
        contribution = component_boundary(
            carrier.alg, carrier.weights[slot], carrier.vectors[slot], program.k
        )
        carrier.observe_value(contribution)
        result = carrier.alg.add(result, contribution)
    return result


def inverse(carrier: RankFourCarrier, program: RankFourProgram) -> None:
    if (
        carrier.stage != "FORWARD_COMPLETE"
        or carrier.lease != lease(program, carrier.alg)
        or carrier.active_k != program.k
    ):
        fail("inverse program does not own the rank-four lease")
    carrier.stage = "INVERSE_ACTIVE"
    apply_sequence(carrier, program.module_b, inverse=True)
    apply_coupling(
        carrier,
        program.second_eta_exponent,
        expected_input_rank=4,
        expected_output_rank=2,
        inverse=True,
    )
    apply_sequence(carrier, program.module_a, inverse=True)
    apply_coupling(
        carrier,
        program.first_eta_exponent,
        expected_input_rank=2,
        expected_output_rank=1,
        inverse=True,
    )
    unload_seed(carrier)
    carrier.lease = None
    carrier.active_k = None
    carrier.stage = "RESTORED"
    carrier.generation += 1
    carrier.observe_resident()
    if not carrier.exact_zero():
        fail("actual inverse failed exact rank-four-carrier restoration")


def state_commitment(carrier: RankFourCarrier) -> tuple[str, int]:
    return rank1.stream_vector_commitment(carrier.flat_values(), carrier.alg)


def degree_two_histograms() -> tuple[tuple[int, ...], ...]:
    result = []
    for left, right in CATALYTICANT_COLUMN_PAIRS:
        histogram = [0] * MODE_COUNT
        histogram[left] += 1
        histogram[right] += 1
        result.append(tuple(histogram))
    return tuple(result)


def monomial_value(
    vector: list[Any], histogram: tuple[int, ...], alg: backend.Algebra
) -> Any:
    value = alg.one
    for mode, exponent in enumerate(histogram):
        if exponent:
            value = alg.mul(value, rank1.scalar_power(alg, vector[mode], exponent))
    return value


def determinant4(matrix: list[list[Any]], alg: backend.Algebra) -> Any:
    work = [list(row) for row in matrix]
    determinant = alg.one
    for column in range(4):
        pivot = next(
            (row for row in range(column, 4) if work[row][column] != alg.zero),
            None,
        )
        if pivot is None:
            return alg.zero
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant = negative(alg, determinant)
        pivot_value = work[column][column]
        determinant = alg.mul(determinant, pivot_value)
        inverse_pivot = alg.inverse(pivot_value)
        for row in range(column + 1, 4):
            factor = alg.mul(work[row][column], inverse_pivot)
            for index in range(column, 4):
                work[row][index] = alg.sub(
                    work[row][index], alg.mul(factor, work[column][index])
                )
    return determinant


def catalecticant_rank_four_certificate(
    carrier: RankFourCarrier, program: RankFourProgram
) -> dict[str, Any]:
    if carrier.active_rank != 4 or carrier.stage != "AFTER_SECOND_COUPLING_HIDDEN":
        fail("rank-four certificate requires the second-coupling hidden stage")
    columns = degree_two_histograms()
    rows = []
    for column in columns:
        row = list(column)
        row[2] += program.k - 4
        rows.append(tuple(row))
    matrix: list[list[Any]] = []
    for row_histogram in rows:
        matrix_row = []
        for column_histogram in columns:
            joined = tuple(
                left + right
                for left, right in zip(row_histogram, column_histogram, strict=True)
            )
            value = carrier.alg.zero
            for slot in range(carrier.active_rank):
                value = carrier.alg.add(
                    value,
                    carrier.alg.mul(
                        carrier.weights[slot],
                        monomial_value(
                            carrier.vectors[slot], joined, carrier.alg
                        ),
                    ),
                )
            matrix_row.append(value)
        matrix.append(matrix_row)
    determinant = determinant4(matrix, carrier.alg)
    if determinant == carrier.alg.zero:
        fail("declared catalecticant rank-four minor vanished")
    return {
        "certificate": "NORMALIZED_CATALECTICANT_FOUR_BY_FOUR_MINOR",
        "minor_nonzero": True,
        "lower_bound": 4,
        "generated_component_upper_bound": 4,
        "exact_normalized_divided_power_secant_rank": 4,
        "ordinary_symmetric_waring_rank_interpretation": (
            carrier.alg.kind == "Q_ZETA17"
        ),
        "column_degree": 2,
        "row_degree": program.k - 2,
        "column_mode_pairs": [list(pair) for pair in CATALYTICANT_COLUMN_PAIRS],
        "row_common_mode": 2,
        "row_common_power": program.k - 4,
        "intermediate_amplitudes_serialized": False,
        "minor_value_serialized": False,
    }


def certificate_case(
    program: RankFourProgram, alg: backend.Algebra
) -> dict[str, Any]:
    carrier = RankFourCarrier.create(alg)
    carrier.lease = lease(program, alg)
    carrier.active_k = program.k
    carrier.stage = "FORWARD_ACTIVE"
    load_seed(carrier)
    apply_coupling(
        carrier,
        program.first_eta_exponent,
        expected_input_rank=1,
        expected_output_rank=2,
    )
    apply_sequence(carrier, program.module_a)
    carrier.stage = "AFTER_SECOND_COUPLING_HIDDEN"
    apply_coupling(
        carrier,
        program.second_eta_exponent,
        expected_input_rank=2,
        expected_output_rank=4,
    )
    carrier.stage = "AFTER_SECOND_COUPLING_HIDDEN"
    certificate = catalecticant_rank_four_certificate(carrier, program)
    apply_coupling(
        carrier,
        program.second_eta_exponent,
        expected_input_rank=4,
        expected_output_rank=2,
        inverse=True,
    )
    apply_sequence(carrier, program.module_a, inverse=True)
    apply_coupling(
        carrier,
        program.first_eta_exponent,
        expected_input_rank=2,
        expected_output_rank=1,
        inverse=True,
    )
    unload_seed(carrier)
    return {
        "k": program.k,
        "family": program.family,
        "algebra": algebra_signature(alg),
        **certificate,
        "verification_carrier_restored_to_empty": carrier.active_rank == 0
        and all(value == alg.zero for value in carrier.flat_values()),
        "verification_only_not_accepted_transaction_path": True,
    }


def public_program_integer_cells(program: RankFourProgram) -> int:
    return 10 + 4 * (len(program.module_a) + len(program.module_b))


def execute_transaction(
    carrier: RankFourCarrier, program: RankFourProgram
) -> dict[str, Any]:
    reset_observation(carrier)
    initial_digest = carrier.digest()
    backing_before = carrier.backing_identity()
    generation_before = carrier.generation
    stats_before = rank1.stats_snapshot(carrier.alg)
    forward(carrier, program)
    commitment, record_bytes = state_commitment(carrier)
    carrier.maximum_commitment_record_json_bytes = max(
        carrier.maximum_commitment_record_json_bytes, record_bytes
    )
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
        "forward_rank4_commitment": commitment,
        "restored_exact_zero": carrier.exact_zero(),
        "same_backing": backing_before == carrier.backing_identity(),
        "generation_before": generation_before,
        "generation_after": carrier.generation,
        "implicit_occupation_dimension_h_k": math.comb(program.k + 16, 16),
        "resident_phase_field_cells": SLOT_COUNT * (MODE_COUNT + 1),
        "resident_chart_rank": 4,
        "maximum_active_rank": carrier.maximum_active_rank,
        "maximum_coupling_transient_components": carrier.maximum_coupling_transient_components,
        "maximum_coupling_transient_field_cells": 8 * (MODE_COUNT + 1),
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
        "maximum_named_transaction_transient_field_cells": 8 * (MODE_COUNT + 1),
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
        "accepted_path_catalecticant_materialized": False,
        "accepted_path_occupation_vector_materialized": False,
        "accepted_path_occupation_topology_materialized": False,
        "accepted_path_matching_or_assignment_expansion_materialized": False,
        "accepted_path_dense_operator_materialized": False,
        "intermediate_rank4_components_serialized_to_controller": False,
    }


def resource_signature(transaction: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "boundary",
        "forward_rank4_commitment",
        "generation_before",
        "generation_after",
        "initial_digest",
        "restored_digest_with_generation",
        "program_fingerprint",
        "family",
    }
    return {key: value for key, value in transaction.items() if key not in excluded}


def run_case(k: int, family: str, alg: backend.Algebra) -> dict[str, Any]:
    return execute_transaction(RankFourCarrier.create(alg), compile_program(k, family))


def compiled_eight_scalar_baseline(
    transaction: dict[str, Any], program: RankFourProgram, alg: backend.Algebra
) -> dict[str, Any]:
    carrier = RankFourCarrier.create(alg)
    forward(carrier, program)
    retained: list[Any] = []
    for slot in range(carrier.active_rank):
        retained.extend(
            (
                carrier.vectors[slot][0],
                alg.mul(carrier.weights[slot], carrier.vectors[slot][1]),
            )
        )
    commitment, record_bytes = rank1.stream_vector_commitment(retained, alg)
    k_value = rank1.field_integer(alg, program.k)
    boundary = alg.zero
    for slot in range(SLOT_COUNT):
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
        "compiler_working_field_cells": SLOT_COUNT * (MODE_COUNT + 1),
        "retained_folded_endpoint_field_cells": 8,
        "total_compiled_warm_field_cells": 8,
        "warm_named_transient_field_cells": 4,
        "retained_folded_endpoint_commitment": commitment,
        "maximum_commitment_record_json_bytes": record_bytes,
        "snapshot_or_phase_carrier_used_by_warm_evaluator": False,
    }


def control_boundary(
    program: RankFourProgram,
    *,
    omit_second_coupling: bool = False,
    second_before_module_a: bool = False,
) -> Any:
    carrier = RankFourCarrier.create(backend.Algebra("F103", modulus=103, root=72))
    carrier.lease = lease(program, carrier.alg)
    carrier.active_k = program.k
    carrier.stage = "FORWARD_ACTIVE"
    load_seed(carrier)
    apply_coupling(
        carrier,
        program.first_eta_exponent,
        expected_input_rank=1,
        expected_output_rank=2,
    )
    if second_before_module_a:
        apply_coupling(
            carrier,
            program.second_eta_exponent,
            expected_input_rank=2,
            expected_output_rank=4,
        )
        apply_sequence(carrier, program.module_a)
    else:
        apply_sequence(carrier, program.module_a)
        if not omit_second_coupling:
            apply_coupling(
                carrier,
                program.second_eta_exponent,
                expected_input_rank=2,
                expected_output_rank=4,
            )
    apply_sequence(carrier, program.module_b)
    carrier.stage = "FORWARD_COMPLETE"
    return project_boundary(carrier, program)


def controls() -> dict[str, Any]:
    program = compile_program(4, "PRIMARY")
    reference = control_boundary(program)

    missing = RankFourCarrier.create(backend.Algebra("F103", modulus=103, root=72))
    forward(missing, program)

    premature = RankFourCarrier.create(backend.Algebra("F103", modulus=103, root=72))
    premature.lease = lease(program, premature.alg)
    premature.active_k = program.k
    premature.stage = "FORWARD_ACTIVE"
    load_seed(premature)
    apply_coupling(
        premature,
        program.first_eta_exponent,
        expected_input_rank=1,
        expected_output_rank=2,
    )
    premature.stage = "AFTER_FIRST_COUPLING_HIDDEN"
    try:
        project_boundary(premature, program)
    except RuntimeError:
        premature_projection_rejected = True
    else:
        premature_projection_rejected = False

    wrong_order = RankFourCarrier.create(backend.Algebra("F103", modulus=103, root=72))
    forward(wrong_order, program)
    try:
        apply_coupling(
            wrong_order,
            program.second_eta_exponent,
            expected_input_rank=4,
            expected_output_rank=2,
            inverse=True,
        )
        apply_sequence(wrong_order, program.module_b, inverse=True)
        apply_sequence(wrong_order, program.module_a, inverse=True)
        apply_coupling(
            wrong_order,
            program.first_eta_exponent,
            expected_input_rank=2,
            expected_output_rank=1,
            inverse=True,
        )
        unload_seed(wrong_order)
    except RuntimeError:
        reordered_inverse_detected = True
    else:
        reordered_inverse_detected = False

    wrong_coupling = RankFourCarrier.create(
        backend.Algebra("F103", modulus=103, root=72)
    )
    forward(wrong_coupling, program)
    apply_sequence(wrong_coupling, program.module_b, inverse=True)
    try:
        apply_coupling(
            wrong_coupling,
            program.second_eta_exponent + 1,
            expected_input_rank=4,
            expected_output_rank=2,
            inverse=True,
        )
    except RuntimeError:
        wrong_second_coupling_inverse_detected = True
    else:
        wrong_second_coupling_inverse_detected = wrong_coupling.active_rank != 2

    owner = RankFourCarrier.create(backend.Algebra("F103", modulus=103, root=72))
    forward(owner, program)
    try:
        inverse(owner, compile_program(4, "REUSE"))
    except RuntimeError:
        wrong_owner_rejected = True
    else:
        wrong_owner_rejected = False

    undermerge = RankFourCarrier.create(backend.Algebra("F103", modulus=103, root=72))
    forward(undermerge, program)
    apply_sequence(undermerge, program.module_b, inverse=True)
    try:
        apply_coupling(
            undermerge,
            program.second_eta_exponent,
            expected_input_rank=4,
            expected_output_rank=2,
            inverse=True,
            canonical_merge=False,
        )
    except RuntimeError:
        missing_exact_merge_rejected = True
    else:
        missing_exact_merge_rejected = False

    try:
        execute_transaction(None, program)  # type: ignore[arg-type]
    except (AttributeError, RuntimeError):
        null_carrier_rejected = True
    else:
        null_carrier_rejected = False

    certificate = certificate_case(
        program, backend.Algebra("F103", modulus=103, root=72)
    )
    try:
        early_coupling_boundary = control_boundary(
            program, second_before_module_a=True
        )
    except RuntimeError:
        second_coupling_module_order_detected = True
    else:
        second_coupling_module_order_detected = early_coupling_boundary != reference
    return {
        "missing_inverse_detected": not missing.exact_zero(),
        "wrong_second_coupling_inverse_detected": wrong_second_coupling_inverse_detected,
        "reordered_inverse_detected": reordered_inverse_detected,
        "wrong_program_ownership_rejected": wrong_owner_rejected,
        "premature_projection_rejected": premature_projection_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "missing_exact_merge_rejected": missing_exact_merge_rejected,
        "second_coupling_omission_changes_boundary": control_boundary(
            program, omit_second_coupling=True
        )
        != reference,
        "second_coupling_module_order_detected": second_coupling_module_order_detected,
        "rank_two_capacity_insufficient_after_second_coupling": certificate[
            "exact_normalized_divided_power_secant_rank"
        ]
        == 4,
        "catalecticant_rank_four_minor_nonzero": certificate["minor_nonzero"],
        "generated_term_count_not_used_without_lower_bound": True,
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

    certificates = [
        certificate_case(compile_program(k, "PRIMARY"), backend.Algebra("Q_ZETA17"))
        for k in EXACT_K
    ]
    for modulus, root in FINITE_FIELDS:
        certificates.extend(
            certificate_case(
                compile_program(k, "PRIMARY"),
                backend.Algebra(f"F{modulus}", modulus=modulus, root=root),
            )
            for k in DECLARED_K
        )
    if not all(
        item["minor_nonzero"]
        and item["exact_normalized_divided_power_secant_rank"] == 4
        and item["verification_carrier_restored_to_empty"]
        for item in certificates
    ):
        fail("one or more catalecticant certificates failed")

    baselines = []
    for item in exact:
        baselines.append(
            compiled_eight_scalar_baseline(
                item, compile_program(item["k"], item["family"]), backend.Algebra("Q_ZETA17")
            )
        )
    for item in structural:
        modulus, root = next(
            pair for pair in FINITE_FIELDS if item["field"] == f"F{pair[0]}"
        )
        baselines.append(
            compiled_eight_scalar_baseline(
                item,
                compile_program(item["k"], item["family"]),
                backend.Algebra(item["field"], modulus=modulus, root=root),
            )
        )
    if not all(item["boundary_agreement"] for item in baselines):
        fail("compiled eight-scalar classical baseline disagrees")

    reuse_carrier = RankFourCarrier.create(backend.Algebra("Q_ZETA17"))
    first = execute_transaction(reuse_carrier, compile_program(8, "PRIMARY"))
    backing = reuse_carrier.backing_identity()
    reused = execute_transaction(reuse_carrier, compile_program(16, "REUSE"))
    fresh = run_case(16, "REUSE", backend.Algebra("Q_ZETA17"))
    if reused["boundary"] != fresh["boundary"]:
        fail("restored rank-four carrier disagrees with fresh unrelated program")
    if resource_signature(reused) != resource_signature(fresh):
        fail("restored rank-four carrier changed the resource signature")

    control_results = controls()
    if not all(
        value for key, value in control_results.items() if key != "snapshot_command_available"
    ) or control_results["snapshot_command_available"]:
        fail("one or more rank-four controls failed")

    return {
        "schema": "CAT_CAS_F17_COHERENT_RANK4_DOUBLE_SECANT_PHASE_COUPLING_CLOSURE_V1",
        "claim": "BOUNDED_EXACT_SECOND_INTERLEAVED_INVOLUTIVE_COHERENT_SUPERPOSITION_COUPLING_WITH_MODULE_ORDER_NONCOMMUTATION_FORCES_CATALECTICANT_RANK4_ON_THE_DECLARED_F17_SECANT_PROGRAM_AND_CLOSES_ON_A_FOUR_COMPONENT_72_CELL_PHASE_CARRIER_WITH_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_AN_EIGHT_TOTAL_FOLDED_SCALAR_WARM_CLASSICAL_RECURRENCE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_scope": {
            "base_obstruction": "M130_SECOND_COUPLING_FOUR_TERMS_WITHOUT_MINIMAL_RANK_PROOF",
            "chart": "RANK4_DOUBLE_COHERENT_VERONESE_SECANT",
            "coupling_count": 2,
            "coupling_law": "I_PLUS_ETA_SWAP_MODE0_MODE1",
            "two_couplings_commute_with_each_other": True,
            "interleaved_coupling_module_order_noncommutation_detected": True,
            "declared_degrees": DECLARED_K,
            "exact_q_zeta17_degrees": EXACT_K,
            "dual_field_degrees": DECLARED_K,
            "consumer_modules": 2,
            "primitives_per_consumer": 20,
        },
        "exact_transactions": exact,
        "dual_field_structural_transactions": structural,
        "catalecticant_rank_certificates": certificates,
        "compiled_eight_total_scalar_classical_baselines": baselines,
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
            "accepted_resident_phase_field_cells": 72,
            "resident_chart_rank": 4,
            "maximum_coupling_transient_components": 8,
            "maximum_coupling_transient_field_cells": 144,
            "accepted_path_occupation_coordinates": 0,
            "accepted_path_catalecticant_cells": 0,
            "accepted_path_matching_or_assignment_expansion": 0,
            "accepted_path_dense_operator_cells": 0,
            "inverse_history_cells": 0,
            "verification_only_catalecticant_scalar_cells_per_certificate": 16,
            "hashlib_internal_state_excluded": True,
            "python_container_native_bigint_and_whole_process_memory_excluded": True,
            "full_exact_bit_complexity_established": False,
        },
        "matched_baseline": {
            "strongest_declared_warm": "COMPILED_EIGHT_TOTAL_FOLDED_ENDPOINT_SCALARS",
            "descriptor_runtime": "IDENTICAL_FOUR_COMPONENT_72_FIELD_CELL_RECURRENCE",
            "phase_advantage_over_matched_classical": False,
        },
        "restoration": {
            "resident_rank4_double_secant_phase_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "transient_eight_component_inverse_merge": "NO_RESTORATION_CLAIM",
            "verification_only_catalecticant_buffers": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "two_declared_involutive_couplings_only": True,
            "fixed_two_consumer_public_word_only": True,
            "declared_k_values_only": DECLARED_K,
            "third_or_unbounded_coupling_rank_law_established": False,
            "fixed_rank_unbounded_depth_closure_established": False,
            "gaussian_chart_closure_established": False,
            "catvm_custody_established": False,
            "distinct_phase_resource_established": False,
            "computational_advantage_established": False,
            "small_wall_crossing_established": False,
            "physical_waveform_execution_established": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation_established": False,
        },
        "next_obstruction": "THE_SECOND_COUPLING_NOW_HAS_AN_EXACT_RANK4_LOWER_BOUND_AND_A72_CELL_CLOSED_TRANSACTION_BUT_EACH_ADDITIONAL_GENERIC_INVOLUTIVE_COUPLING_CAN_DOUBLE_THE_GENERATED_COHERENT_TERMS_AND_THE_FIXED_WORD_ALREADY_COMPILES_TO_AN_EIGHT_TOTAL_SCALAR_CLASSICAL_BOUNDARY_RECURRENCE",
        "next_experiment": "EXACT_SYMBOLIC_ITERATED_COHERENT_COUPLING_SECANT_RANK_GROWTH_LAW_OR_FIXED_RANK_GAUSSIAN_PHASE_CHART_NO_GO_WITHOUT_COMPONENT_ENUMERATION",
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
