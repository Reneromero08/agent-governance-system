#!/usr/bin/env python3
"""Separate exact oracle for pi-plus-unit embedding balance.

The oracle imports only previously sealed independent kernels.  It recompiles
the public operators, uses sequential x-mod-q coefficients, and implements
the exact trace-energy balance without importing the production successor.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_pi_content_recurrence_oracle as pi_reference
import f17_cubic_chain_period17_unit_height_reduction_oracle as reference


PRIME = 17
DIMENSION = 16
MESSAGE_SLOTS = 18
OUTPUT_SLOT = 17
UNIT_RANK = 7
MAX_BALANCE_STEPS = 128
EXPECTED_PERIODS = (1, 64)

RingElement = tuple[int, ...]
RingVector = list[RingElement]


def fail(message: str) -> None:
    raise RuntimeError(message)


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def element_payload_bits(element: RingElement) -> int:
    return sum(signed_bits(value) for value in element)


def vector_payload_bits(vector: RingVector) -> int:
    return sum(element_payload_bits(element) for element in vector)


def element_width(element: RingElement) -> int:
    return max(signed_bits(value) for value in element)


def vector_width(vector: RingVector) -> int:
    return max(element_width(element) for element in vector)


def ledger_payload_bits(ledger: tuple[int, ...] | list[int]) -> int:
    return sum(signed_bits(value) for value in ledger)


def conjugate(element: RingElement) -> RingElement:
    result = reference.ring_zero()
    for exponent, coefficient in enumerate(element):
        monomial = reference.ring_monomial((-exponent) % PRIME)
        result = reference.ring_add(
            result,
            tuple(coefficient * value for value in monomial),
        )
    return result


def field_trace(element: RingElement) -> int:
    return DIMENSION * element[0] - sum(element[1:])


def norm_element(
    vector: RingVector,
    stats: "OracleStats | None" = None,
) -> RingElement:
    result = reference.ring_zero()
    for element in vector:
        result = reference.ring_add(
            result,
            reference.ring_multiply(element, conjugate(element)),
        )
        if stats is not None:
            stats.initial_norm_element_ring_multiplications += 1
    return result


def direct_energy(vector: RingVector) -> int:
    return sum(
        field_trace(
            reference.ring_multiply(element, conjugate(element))
        )
        for element in vector
    )


def multiply_vector(
    scalar: RingElement,
    vector: RingVector,
) -> RingVector:
    return [
        reference.ring_multiply(scalar, element)
        for element in vector
    ]


UNIT_MOVES = tuple(
    (
        generator_index,
        delta,
        multiplier,
        reference.ring_multiply(multiplier, conjugate(multiplier)),
    )
    for generator_index in range(UNIT_RANK)
    for delta, multiplier in (
        (1, reference.UNIT_GENERATOR_INVERSES[generator_index]),
        (-1, reference.UNIT_GENERATORS[generator_index]),
    )
)


@dataclass
class OracleStats:
    balance_calls: int = 0
    balance_candidate_evaluations: int = 0
    balance_selected_steps: int = 0
    balance_step_cap_hits: int = 0
    exact_embedding_energy_evaluations: int = 0
    initial_norm_element_ring_multiplications: int = 0
    candidate_norm_ring_multiplications: int = 0
    unit_vector_ring_multiplications: int = 0
    balanced_scalar_vector_ring_multiplications: int = 0
    unit_scale_ring_multiplications: int = 0
    unit_scale_materializations: int = 0
    basis_operator_applications: int = 0
    basis_operator_ring_multiply_accumulations: int = 0
    maximum_trace_energy_bits: int = 1
    maximum_resident_payload_bits: int = 0
    maximum_duplicate_state_payload_bits: int = 0
    maximum_declared_live_state_payload_bits: int = 0
    maximum_residual_signed_bits: int = 1
    maximum_pi_ledger_signed_bits: int = 1
    maximum_unit_ledger_signed_bits: int = 1
    maximum_nonzero_message_slots: int = 0
    maximum_nonzero_message_pi_ledgers: int = 0
    maximum_nonzero_message_unit_ledgers: int = 0
    maximum_nonzero_coefficient_registers: int = 0
    maximum_nonzero_coefficient_pi_ledgers: int = 0
    maximum_nonzero_coefficient_unit_ledgers: int = 0
    maximum_unit_materialization_payload_bits: int = 0


def observe_energy(value: int, stats: OracleStats) -> int:
    if value < 0:
        fail("oracle trace energy became negative")
    stats.exact_embedding_energy_evaluations += 1
    stats.maximum_trace_energy_bits = max(
        stats.maximum_trace_energy_bits,
        max(1, value.bit_length()),
    )
    return value


def balance(
    vector: RingVector,
    base_ledger: tuple[int, ...] | list[int],
    stats: OracleStats,
) -> tuple[RingVector, tuple[int, ...]]:
    if len(base_ledger) != UNIT_RANK:
        fail("oracle unit ledger width changed")
    if all(element == reference.ring_zero() for element in vector):
        return (
            [reference.ring_zero() for _ in vector],
            tuple(0 for _ in range(UNIT_RANK)),
        )
    stats.balance_calls += 1
    current = list(vector)
    ledger = list(base_ledger)
    current_norm = norm_element(current, stats)
    current_energy = observe_energy(field_trace(current_norm), stats)
    selected = 0
    for _ in range(MAX_BALANCE_STEPS):
        best: tuple[
            int,
            tuple[int, ...],
            int,
            int,
            RingElement,
        ] | None = None
        for generator_index, delta, _, factor in UNIT_MOVES:
            stats.balance_candidate_evaluations += 1
            trial_norm = reference.ring_multiply(factor, current_norm)
            stats.candidate_norm_ring_multiplications += 1
            trial_ledger = list(ledger)
            trial_ledger[generator_index] += delta
            choice = (
                observe_energy(field_trace(trial_norm), stats),
                tuple(trial_ledger),
                generator_index,
                delta,
                trial_norm,
            )
            if best is None or choice[:-1] < best[:-1]:
                best = choice
        if best is None or best[0] >= current_energy:
            break
        current_energy = best[0]
        ledger = list(best[1])
        generator_index = best[2]
        delta = best[3]
        multiplier = (
            reference.UNIT_GENERATOR_INVERSES[generator_index]
            if delta == 1
            else reference.UNIT_GENERATORS[generator_index]
        )
        current = multiply_vector(multiplier, current)
        stats.unit_vector_ring_multiplications += len(current)
        current_norm = best[4]
        selected += 1
        stats.balance_selected_steps += 1
    if selected == MAX_BALANCE_STEPS:
        stats.balance_step_cap_hits += 1
    return current, tuple(ledger)


def ring_power(
    base: RingElement,
    exponent: int,
    stats: OracleStats,
) -> RingElement:
    if exponent < 0:
        fail("oracle unit power exponent became negative")
    result = reference.ring_one()
    factor = base
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = reference.ring_multiply(result, factor)
            stats.unit_scale_ring_multiplications += 1
        remaining >>= 1
        if remaining:
            factor = reference.ring_multiply(factor, factor)
            stats.unit_scale_ring_multiplications += 1
    return result


def ledger_scale(
    ledger: tuple[int, ...] | list[int],
    stats: OracleStats,
) -> RingElement:
    if len(ledger) != UNIT_RANK:
        fail("oracle unit ledger width changed")
    result = reference.ring_one()
    for exponent, generator, inverse in zip(
        ledger,
        reference.UNIT_GENERATORS,
        reference.UNIT_GENERATOR_INVERSES,
        strict=True,
    ):
        factor = (
            ring_power(generator, exponent, stats)
            if exponent >= 0
            else ring_power(inverse, -exponent, stats)
        )
        result = reference.ring_multiply(result, factor)
        stats.unit_scale_ring_multiplications += 1
    return result


@dataclass(frozen=True)
class BalancedElement:
    residual: RingElement
    pi_exponent: int
    ledger: tuple[int, ...]


@dataclass(frozen=True)
class BalancedVector:
    residual: RingVector
    pi_exponent: int
    ledger: tuple[int, ...]


def zero_vector() -> BalancedVector:
    return BalancedVector(
        [reference.ring_zero() for _ in range(PRIME)],
        0,
        tuple(0 for _ in range(UNIT_RANK)),
    )


def normalize_element(
    element: RingElement,
    exponent: int,
    ledger: tuple[int, ...] | list[int],
    stats: OracleStats,
) -> BalancedElement:
    scaled = pi_reference.normalize_element(element, exponent)
    if scaled.residual == reference.ring_zero():
        return BalancedElement(
            reference.ring_zero(),
            0,
            tuple(0 for _ in range(UNIT_RANK)),
        )
    residual, balanced_ledger = balance(
        [scaled.residual],
        ledger,
        stats,
    )
    return BalancedElement(
        residual[0],
        scaled.exponent,
        balanced_ledger,
    )


def normalize_vector(
    vector: RingVector,
    exponent: int,
    ledger: tuple[int, ...] | list[int],
    stats: OracleStats,
) -> BalancedVector:
    scaled = pi_reference.normalize_vector(vector, exponent)
    if all(
        element == reference.ring_zero()
        for element in scaled.residual
    ):
        return zero_vector()
    residual, balanced_ledger = balance(
        scaled.residual,
        ledger,
        stats,
    )
    return BalancedVector(
        residual,
        scaled.exponent,
        balanced_ledger,
    )


def materialize_unit_vector(
    value: BalancedVector,
    stats: OracleStats,
) -> RingVector:
    if all(
        element == reference.ring_zero()
        for element in value.residual
    ):
        return [reference.ring_zero() for _ in range(PRIME)]
    scale = ledger_scale(value.ledger, stats)
    materialized = multiply_vector(scale, value.residual)
    stats.unit_scale_materializations += 1
    stats.unit_vector_ring_multiplications += len(value.residual)
    stats.maximum_unit_materialization_payload_bits = max(
        stats.maximum_unit_materialization_payload_bits,
        vector_payload_bits(materialized),
    )
    return materialized


def add_vectors(
    left: BalancedVector,
    right: BalancedVector,
    stats: OracleStats,
) -> BalancedVector:
    if all(
        element == reference.ring_zero()
        for element in left.residual
    ):
        return right
    if all(
        element == reference.ring_zero()
        for element in right.residual
    ):
        return left
    left_scaled = pi_reference.ScaledVector(
        materialize_unit_vector(left, stats),
        left.pi_exponent,
    )
    right_scaled = pi_reference.ScaledVector(
        materialize_unit_vector(right, stats),
        right.pi_exponent,
    )
    combined = pi_reference.scaled_vector_add(
        left_scaled,
        right_scaled,
    )
    return normalize_vector(
        combined.residual,
        combined.exponent,
        tuple(0 for _ in range(UNIT_RANK)),
        stats,
    )


def multiply(
    scalar: BalancedElement,
    vector: BalancedVector,
    stats: OracleStats,
) -> BalancedVector:
    if (
        scalar.residual == reference.ring_zero()
        or all(
            element == reference.ring_zero()
            for element in vector.residual
        )
    ):
        return zero_vector()
    residual = [
        reference.ring_multiply(scalar.residual, element)
        for element in vector.residual
    ]
    stats.balanced_scalar_vector_ring_multiplications += len(residual)
    return normalize_vector(
        residual,
        scalar.pi_exponent + vector.pi_exponent,
        tuple(
            left + right
            for left, right in zip(
                scalar.ledger,
                vector.ledger,
                strict=True,
            )
        ),
        stats,
    )


def project(
    output: BalancedVector,
    stats: OracleStats,
) -> RingElement:
    unit_value = materialize_unit_vector(output, stats)
    boundary = reference.project(unit_value)
    scaled = pi_reference.normalize_element(
        boundary,
        output.pi_exponent,
    )
    return pi_reference.materialize_element(scaled)


def build_state(
    operator: list[list[RingElement]],
    seed: RingVector,
    characteristic: list[RingElement],
    periods: int,
    stats: OracleStats,
) -> tuple[list[BalancedVector], list[BalancedElement]]:
    current = normalize_vector(
        seed,
        0,
        tuple(0 for _ in range(UNIT_RANK)),
        stats,
    )
    messages = [current]
    for _ in range(DIMENSION):
        current = normalize_vector(
            reference.matrix_vector_multiply(
                operator,
                current.residual,
            ),
            current.pi_exponent,
            current.ledger,
            stats,
        )
        messages.append(current)
        stats.basis_operator_applications += 1
        stats.basis_operator_ring_multiply_accumulations += (
            PRIME * PRIME
        )
    scaled_coefficients = pi_reference.sequential_scaled_coefficients(
        periods,
        characteristic,
    )
    coefficients = [
        normalize_element(
            value.residual,
            value.exponent,
            tuple(0 for _ in range(UNIT_RANK)),
            stats,
        )
        for value in scaled_coefficients
    ]
    output = zero_vector()
    for coefficient, basis in zip(
        coefficients,
        messages[1:],
        strict=True,
    ):
        output = add_vectors(
            output,
            multiply(coefficient, basis, stats),
            stats,
        )
    messages.append(output)
    return messages, coefficients


def record_metrics(
    messages: list[BalancedVector],
    coefficients: list[BalancedElement],
    stats: OracleStats,
) -> dict[str, int]:
    payload = 0
    residual_width = 1
    pi_width = 1
    unit_width = 1
    nonzero_messages = 0
    nonzero_message_pi = 0
    nonzero_message_units = 0
    nonzero_coefficients = 0
    nonzero_coefficient_pi = 0
    nonzero_coefficient_units = 0
    for value in messages:
        payload += vector_payload_bits(value.residual)
        payload += signed_bits(value.pi_exponent)
        payload += ledger_payload_bits(value.ledger)
        residual_width = max(
            residual_width,
            vector_width(value.residual),
        )
        pi_width = max(pi_width, signed_bits(value.pi_exponent))
        unit_width = max(
            unit_width,
            *(signed_bits(item) for item in value.ledger),
        )
        nonzero_messages += int(
            any(
                element != reference.ring_zero()
                for element in value.residual
            )
        )
        nonzero_message_pi += int(value.pi_exponent != 0)
        nonzero_message_units += int(any(value.ledger))
    for value in coefficients:
        payload += element_payload_bits(value.residual)
        payload += signed_bits(value.pi_exponent)
        payload += ledger_payload_bits(value.ledger)
        residual_width = max(
            residual_width,
            element_width(value.residual),
        )
        pi_width = max(pi_width, signed_bits(value.pi_exponent))
        unit_width = max(
            unit_width,
            *(signed_bits(item) for item in value.ledger),
        )
        nonzero_coefficients += int(
            value.residual != reference.ring_zero()
        )
        nonzero_coefficient_pi += int(value.pi_exponent != 0)
        nonzero_coefficient_units += int(any(value.ledger))
    stats.maximum_resident_payload_bits = max(
        stats.maximum_resident_payload_bits,
        payload,
    )
    stats.maximum_duplicate_state_payload_bits = max(
        stats.maximum_duplicate_state_payload_bits,
        payload,
    )
    stats.maximum_declared_live_state_payload_bits = max(
        stats.maximum_declared_live_state_payload_bits,
        2 * payload,
    )
    stats.maximum_residual_signed_bits = max(
        stats.maximum_residual_signed_bits,
        residual_width,
    )
    stats.maximum_pi_ledger_signed_bits = max(
        stats.maximum_pi_ledger_signed_bits,
        pi_width,
    )
    stats.maximum_unit_ledger_signed_bits = max(
        stats.maximum_unit_ledger_signed_bits,
        unit_width,
    )
    stats.maximum_nonzero_message_slots = max(
        stats.maximum_nonzero_message_slots,
        nonzero_messages,
    )
    stats.maximum_nonzero_message_pi_ledgers = max(
        stats.maximum_nonzero_message_pi_ledgers,
        nonzero_message_pi,
    )
    stats.maximum_nonzero_message_unit_ledgers = max(
        stats.maximum_nonzero_message_unit_ledgers,
        nonzero_message_units,
    )
    stats.maximum_nonzero_coefficient_registers = max(
        stats.maximum_nonzero_coefficient_registers,
        nonzero_coefficients,
    )
    stats.maximum_nonzero_coefficient_pi_ledgers = max(
        stats.maximum_nonzero_coefficient_pi_ledgers,
        nonzero_coefficient_pi,
    )
    stats.maximum_nonzero_coefficient_unit_ledgers = max(
        stats.maximum_nonzero_coefficient_unit_ledgers,
        nonzero_coefficient_units,
    )
    return {
        "balance_calls": stats.balance_calls,
        "balance_candidate_evaluations": (
            stats.balance_candidate_evaluations
        ),
        "balance_selected_steps": stats.balance_selected_steps,
        "balance_step_cap_hits": stats.balance_step_cap_hits,
        "exact_embedding_energy_evaluations": (
            stats.exact_embedding_energy_evaluations
        ),
        "initial_norm_element_ring_multiplications": (
            stats.initial_norm_element_ring_multiplications
        ),
        "candidate_norm_ring_multiplications": (
            stats.candidate_norm_ring_multiplications
        ),
        "unit_vector_ring_multiplications": (
            stats.unit_vector_ring_multiplications
        ),
        "balanced_scalar_vector_ring_multiplications": (
            stats.balanced_scalar_vector_ring_multiplications
        ),
        "unit_scale_ring_multiplications": (
            stats.unit_scale_ring_multiplications
        ),
        "unit_scale_materializations": (
            stats.unit_scale_materializations
        ),
        "basis_operator_applications": (
            stats.basis_operator_applications
        ),
        "basis_operator_ring_multiply_accumulations": (
            stats.basis_operator_ring_multiply_accumulations
        ),
        "maximum_trace_energy_bits": stats.maximum_trace_energy_bits,
        "maximum_resident_payload_bits": (
            stats.maximum_resident_payload_bits
        ),
        "maximum_duplicate_state_payload_bits": (
            stats.maximum_duplicate_state_payload_bits
        ),
        "maximum_declared_live_state_payload_bits": (
            stats.maximum_declared_live_state_payload_bits
        ),
        "maximum_residual_signed_bits": (
            stats.maximum_residual_signed_bits
        ),
        "maximum_pi_ledger_signed_bits": (
            stats.maximum_pi_ledger_signed_bits
        ),
        "maximum_unit_ledger_signed_bits": (
            stats.maximum_unit_ledger_signed_bits
        ),
        "maximum_nonzero_message_slots": (
            stats.maximum_nonzero_message_slots
        ),
        "maximum_nonzero_message_pi_ledgers": (
            stats.maximum_nonzero_message_pi_ledgers
        ),
        "maximum_nonzero_message_unit_ledgers": (
            stats.maximum_nonzero_message_unit_ledgers
        ),
        "maximum_nonzero_coefficient_registers": (
            stats.maximum_nonzero_coefficient_registers
        ),
        "maximum_nonzero_coefficient_pi_ledgers": (
            stats.maximum_nonzero_coefficient_pi_ledgers
        ),
        "maximum_nonzero_coefficient_unit_ledgers": (
            stats.maximum_nonzero_coefficient_unit_ledgers
        ),
        "maximum_unit_materialization_payload_bits": (
            stats.maximum_unit_materialization_payload_bits
        ),
    }


def family_context(
    family: str,
    certificate: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    descriptor = reference.compile_descriptor(
        PRIME + 1,
        family.upper(),
    )
    operator = reference.compile_operator(descriptor)
    characteristic = [
        tuple(element)
        for element in certificate["characteristic"]
    ]
    checks = {
        "descriptor_sha256_equal": (
            hashlib.sha256(reference.encoded(descriptor)).hexdigest()
            == certificate["public_program_sha256"]
        ),
        "operator_sha256_equal": (
            hashlib.sha256(reference.encoded(operator)).hexdigest()
            == certificate["operator_sha256"]
        ),
        "characteristic_sha256_equal": (
            hashlib.sha256(
                reference.encoded(characteristic)
            ).hexdigest()
            == certificate["characteristic_sha256"]
        ),
        "whole_operator_annihilator_identity_exact": (
            reference.check_annihilator(operator, characteristic)
        ),
        "production_characteristic_identity_asserted": (
            certificate["characteristic_identity_exact"]
        ),
    }
    return checks, {
        "operator": operator,
        "characteristic": characteristic,
        "seed": reference.seed_vector(descriptor),
    }


def case_check(
    case: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    stats = OracleStats()
    messages, coefficients = build_state(
        context["operator"],
        context["seed"],
        context["characteristic"],
        case["periods"],
        stats,
    )
    boundary = project(messages[-1], stats)
    metrics = record_metrics(messages, coefficients, stats)
    inverse_stats = OracleStats()
    inverse_messages, inverse_coefficients = build_state(
        context["operator"],
        context["seed"],
        context["characteristic"],
        case["periods"],
        inverse_stats,
    )
    inverse_metrics = record_metrics(
        inverse_messages,
        inverse_coefficients,
        inverse_stats,
    )
    production = case["balanced_stats"]
    comparable_keys = tuple(metrics)
    production_inverse = case["inverse_rematerialization_stats"]
    return {
        "periods": case["periods"],
        "family": case["family"],
        "boundary_sha256_equal": (
            hashlib.sha256(reference.encoded(boundary)).hexdigest()
            == case["boundary_sha256"]
        ),
        "all_exact_balance_and_resource_metrics_equal": all(
            metrics[key] == production[key]
            for key in comparable_keys
        ),
        "inverse_rematerialization_exact_resource_metrics_equal": all(
            inverse_metrics[key] == production_inverse[key]
            for key in inverse_metrics
        ),
        "pi_payload_reduction_equal": (
            metrics["maximum_resident_payload_bits"]
            < case["pi_content_payload_bits"]
        ),
        "raw_payload_relation_equal": (
            (
                metrics["maximum_resident_payload_bits"]
                < case["raw_recurrence_payload_bits"]
            )
            == case["balanced_beats_raw_recurrence_payload"]
        ),
        "declared_live_pi_payload_relation_equal": (
            (
                metrics["maximum_declared_live_state_payload_bits"]
                < case["pi_content_payload_bits"]
            )
            == case[
                "balanced_declared_live_reduces_pi_content_payload"
            ]
        ),
        "declared_live_raw_payload_relation_equal": (
            (
                metrics["maximum_declared_live_state_payload_bits"]
                < case["raw_recurrence_payload_bits"]
            )
            == case[
                "balanced_declared_live_beats_raw_recurrence_payload"
            ]
        ),
        "exact_resource_tuple": metrics,
        "exact_inverse_resource_tuple": inverse_metrics,
    }


@dataclass
class OracleCarrier:
    message_residuals: list[RingVector]
    message_pi: list[int]
    message_units: list[list[int]]
    coefficient_residuals: list[RingElement]
    coefficient_pi: list[int]
    coefficient_units: list[list[int]]
    generation: int = 0
    lease: int = 0

    @classmethod
    def create(cls) -> "OracleCarrier":
        return cls(
            [
                [reference.ring_zero() for _ in range(PRIME)]
                for _ in range(MESSAGE_SLOTS)
            ],
            [0 for _ in range(MESSAGE_SLOTS)],
            [
                [0 for _ in range(UNIT_RANK)]
                for _ in range(MESSAGE_SLOTS)
            ],
            [reference.ring_zero() for _ in range(DIMENSION)],
            [0 for _ in range(DIMENSION)],
            [
                [0 for _ in range(UNIT_RANK)]
                for _ in range(DIMENSION)
            ],
        )

    def backing(self) -> tuple[int, ...]:
        return (
            id(self.message_residuals),
            *(id(row) for row in self.message_residuals),
            id(self.message_pi),
            id(self.message_units),
            *(id(row) for row in self.message_units),
            id(self.coefficient_residuals),
            id(self.coefficient_pi),
            id(self.coefficient_units),
            *(id(row) for row in self.coefficient_units),
        )

    def all_zero(self) -> bool:
        return (
            all(
                all(
                    element == reference.ring_zero()
                    for element in row
                )
                for row in self.message_residuals
            )
            and not any(self.message_pi)
            and all(not any(row) for row in self.message_units)
            and all(
                element == reference.ring_zero()
                for element in self.coefficient_residuals
            )
            and not any(self.coefficient_pi)
            and all(not any(row) for row in self.coefficient_units)
        )


def execute_carrier(
    carrier: OracleCarrier,
    context: dict[str, Any],
    periods: int,
) -> tuple[RingElement, bool]:
    if not carrier.all_zero():
        fail("oracle carrier was not restored")
    carrier.lease += 1
    stats = OracleStats()
    messages, coefficients = build_state(
        context["operator"],
        context["seed"],
        context["characteristic"],
        periods,
        stats,
    )
    for index, value in enumerate(messages):
        carrier.message_residuals[index][:] = value.residual
        carrier.message_pi[index] = value.pi_exponent
        carrier.message_units[index][:] = value.ledger
    for index, value in enumerate(coefficients):
        carrier.coefficient_residuals[index] = value.residual
        carrier.coefficient_pi[index] = value.pi_exponent
        carrier.coefficient_units[index][:] = value.ledger
    boundary = project(messages[-1], stats)
    inverse_stats = OracleStats()
    inverse_messages, inverse_coefficients = build_state(
        context["operator"],
        context["seed"],
        context["characteristic"],
        periods,
        inverse_stats,
    )

    def subtract_message(index: int, expected: BalancedVector) -> None:
        if (
            carrier.message_residuals[index] != expected.residual
            or carrier.message_pi[index] != expected.pi_exponent
            or tuple(carrier.message_units[index]) != expected.ledger
        ):
            fail("oracle inverse message mismatch")
        carrier.message_residuals[index][:] = [
            reference.ring_subtract(actual, value)
            for actual, value in zip(
                carrier.message_residuals[index],
                expected.residual,
                strict=True,
            )
        ]
        carrier.message_pi[index] -= expected.pi_exponent
        carrier.message_units[index][:] = [
            actual - value
            for actual, value in zip(
                carrier.message_units[index],
                expected.ledger,
                strict=True,
            )
        ]

    subtract_message(OUTPUT_SLOT, inverse_messages[-1])
    for index, expected in enumerate(inverse_coefficients):
        if (
            carrier.coefficient_residuals[index] != expected.residual
            or carrier.coefficient_pi[index] != expected.pi_exponent
            or tuple(carrier.coefficient_units[index])
            != expected.ledger
        ):
            fail("oracle inverse coefficient mismatch")
        carrier.coefficient_residuals[index] = reference.ring_subtract(
            carrier.coefficient_residuals[index],
            expected.residual,
        )
        carrier.coefficient_pi[index] -= expected.pi_exponent
        carrier.coefficient_units[index][:] = [
            actual - value
            for actual, value in zip(
                carrier.coefficient_units[index],
                expected.ledger,
                strict=True,
            )
        ]
    for index in range(DIMENSION, 0, -1):
        subtract_message(index, inverse_messages[index])
    subtract_message(0, inverse_messages[0])
    carrier.generation += 1
    return boundary, carrier.all_zero()


def restoration_check(
    contexts: dict[str, dict[str, Any]],
    production: dict[str, Any],
) -> dict[str, bool]:
    carrier = OracleCarrier.create()
    backing = carrier.backing()
    _, primary_restored = execute_carrier(
        carrier,
        contexts["primary"],
        1,
    )
    reuse_boundary, reuse_restored = execute_carrier(
        carrier,
        contexts["reuse"],
        1,
    )
    fresh_boundary, fresh_restored = execute_carrier(
        OracleCarrier.create(),
        contexts["reuse"],
        1,
    )
    expected = production["restoration_reuse_case"]
    return {
        "primary_restored_exactly": primary_restored,
        "reuse_restored_exactly": reuse_restored,
        "fresh_restored_exactly": fresh_restored,
        "same_original_backing": carrier.backing() == backing,
        "fresh_reuse_boundary_equal": (
            reuse_boundary == fresh_boundary
        ),
        "all_payload_and_ledgers_zero": carrier.all_zero(),
        "generation_equal": carrier.generation == expected["generation"],
        "lease_equal": carrier.lease == expected["lease"],
        "no_inverse_history": (
            expected["retained_inverse_history_bytes"] == 0
        ),
        "no_baseline_reload": expected["baseline_reload_bytes"] == 0,
        "full_object_equality_not_claimed": (
            not expected["full_carrier_object_state_equal"]
        ),
        "metadata_width_not_claimed_bounded": (
            not expected["repeated_use_metadata_width_bounded"]
        ),
    }


def mutation_check(context: dict[str, Any]) -> dict[str, bool]:
    stats = OracleStats()
    messages, coefficients = build_state(
        context["operator"],
        context["seed"],
        context["characteristic"],
        64,
        stats,
    )
    boundary = project(messages[-1], stats)
    index = next(
        index
        for index, value in enumerate(coefficients)
        if value.residual != reference.ring_zero()
    )
    mutated = list(coefficients)
    value = mutated[index]
    ledger = list(value.ledger)
    ledger[0] += 1
    mutated[index] = BalancedElement(
        value.residual,
        value.pi_exponent,
        tuple(ledger),
    )
    output = zero_vector()
    mutation_stats = OracleStats()
    for coefficient, basis in zip(
        mutated,
        messages[1:17],
        strict=True,
    ):
        output = add_vectors(
            output,
            multiply(coefficient, basis, mutation_stats),
            mutation_stats,
        )
    mutated_boundary = project(output, mutation_stats)
    return {
        "nonzero_coefficient_unit_ledger_mutation_changes_boundary": (
            mutated_boundary != boundary
        ),
        "norm_element_trace_matches_direct_energy": (
            field_trace(norm_element(context["seed"]))
            == direct_energy(context["seed"])
        ),
    }


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_chain_period17_"
            "pi_unit_embedding_balance_oracle.py PRODUCTION_RESULT"
        )
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        production = json.load(handle)
    if tuple(production["tested_periods"]) != EXPECTED_PERIODS:
        fail("oracle tested periods changed")

    contexts = {}
    family_checks = {}
    for family in ("primary", "reuse"):
        checks, context = family_context(
            family,
            production["block_certificates"][family],
        )
        family_checks[family] = checks
        contexts[family] = context
    case_checks = [
        case_check(case, contexts[case["family"].lower()])
        for case in production["cases"]
    ]
    restoration = restoration_check(contexts, production)
    mutations = mutation_check(contexts["primary"])
    scope = {
        "production_result_pass": production["result"] == "PASS",
        "all_pi_payload_reductions": (
            production["all_cases_reduce_pi_content_payload"]
        ),
        "all_declared_live_pi_payload_reductions": (
            production[
                "all_declared_live_cases_reduce_pi_content_payload"
            ]
        ),
        "period64_raw_obstruction_retained": (
            production[
                "period64_both_families_remain_above_raw_recurrence"
            ]
        ),
        "period64_declared_live_raw_obstruction_retained": (
            production[
                "period64_declared_live_both_families_"
                "remain_above_raw"
            ]
        ),
        "all_cases_do_not_beat_raw": (
            not production["all_cases_beat_raw_recurrence_payload"]
        ),
        "identical_classical_balancer_retained": (
            production["matched_classical"][
                "identical_pi_and_unit_balanced_recurrence_available"
            ]
            and not production["matched_classical"][
                "comparison_establishes_advantage"
            ]
        ),
        "global_optimality_not_claimed": (
            "GLOBAL_CYCLOTOMIC_UNIT_OPTIMALITY"
            in production["not_established"]
        ),
        "distinct_phase_resource_not_claimed": (
            "DISTINCT_PHASE_RESOURCE"
            in production["not_established"]
        ),
    }
    result_pass = (
        all(all(values.values()) for values in family_checks.values())
        and all(
            all(
                value
                for key, value in check.items()
                if key not in {
                    "periods",
                    "family",
                    "exact_resource_tuple",
                    "exact_inverse_resource_tuple",
                }
            )
            for check in case_checks
        )
        and all(restoration.values())
        and all(mutations.values())
        and all(scope.values())
    )
    result = {
        "result": "PASS" if result_pass else "FAIL",
        "experiment": (
            "SEPARATE_EXACT_MULTI_EMBEDDING_PI_UNIT_BALANCE_ORACLE"
        ),
        "oracle_imports_production_module": False,
        "oracle_coefficient_method": (
            "SEQUENTIAL_MULTIPLICATION_BY_X_MOD_Q"
        ),
        "production_coefficient_method": (
            "BINARY_POLYNOMIAL_POWERING_MOD_Q"
        ),
        "family_checks": family_checks,
        "case_checks": case_checks,
        "restoration_checks": restoration,
        "mutation_checks": mutations,
        "production_scope_checks": scope,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": (
            "LINUX_X86_64_PYTHON_EXACT_TWO_PUBLIC_F17_PERIOD17_"
            "CUBIC_PATH_FAMILIES_Q_ZETA17_PI_CONTENT_PLUS_SEVEN_"
            "DECLARED_UNIT_LEDGER_EXACT_TRACE_ENERGY_BALANCE_"
            "128_STEP_CAP_PERIODS1_AND64_RESIDENT_AND_DECLARED_"
            "DUPLICATE_REMATERIALIZATION_LIVE_PAYLOAD_"
            "DIAGNOSTIC_EXACT_SUBTRACTIVE_RESTORATION_SOFTWARE_ONLY"
        ),
        "not_established": production["not_established"],
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
