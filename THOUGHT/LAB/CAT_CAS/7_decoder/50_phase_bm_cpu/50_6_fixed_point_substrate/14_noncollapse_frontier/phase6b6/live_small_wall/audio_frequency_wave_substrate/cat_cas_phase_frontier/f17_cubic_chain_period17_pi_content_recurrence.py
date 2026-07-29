#!/usr/bin/env python3
"""Execute the native-K recurrence with exact reversible pi-content ledgers.

The predecessor proves that exact boundary pi-valuation grows linearly while
its exponent needs only a logarithmic ledger.  This successor factors powers
of pi = 1-zeta_17 from every resident message and recurrence coefficient.
Polynomial multiplication, reduction, basis contraction, projection, and
subtractive restoration operate on residual-plus-exponent pairs.

No raw growing recurrence coefficient is part of the accepted carrier path.
Exact raw recurrence execution is retained only as a separately counted
verification baseline.  The identical normalized recurrence is available to
ordinary compact classical software, so reduced payload is not an advantage.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_cyclotomic_module as cyclo
import f17_cubic_chain_period17_executed_recurrence as recurrence
import f17_cubic_chain_period17_height_lower_bound as height


PRIME = cyclo.PRIME
DIMENSION = cyclo.DIMENSION
MESSAGE_INTEGER_CELLS = cyclo.MESSAGE_INTEGER_CELLS
MESSAGE_SLOTS = recurrence.MESSAGE_SLOTS
BASIS_MESSAGES = recurrence.BASIS_MESSAGES
OUTPUT_SLOT = recurrence.OUTPUT_SLOT
COEFFICIENT_REGISTERS = recurrence.COEFFICIENT_REGISTERS
TESTED_PERIODS = (1, 4, 16, 64, 256)
DENSE_DIRECT_PERIOD_CAP = 64

RingElement = cyclo.RingElement
RingVector = cyclo.RingVector


def fail(message: str) -> None:
    raise RuntimeError(message)


def signed_bits(value: int) -> int:
    return cyclo.signed_bits(value)


def element_payload_bits(element: RingElement) -> int:
    return sum(signed_bits(value) for value in element)


def vector_payload_bits(vector: RingVector) -> int:
    return sum(element_payload_bits(element) for element in vector)


def element_maximum_signed_bits(element: RingElement) -> int:
    return max(signed_bits(value) for value in element)


def vector_maximum_signed_bits(vector: RingVector) -> int:
    return max(element_maximum_signed_bits(element) for element in vector)


@dataclass
class PiStats:
    scaled_element_additions: int = 0
    scaled_element_multiplications: int = 0
    scaled_vector_additions: int = 0
    scaled_vector_scalar_multiplications: int = 0
    exact_pi_divisions: int = 0
    pi_power_ring_multiplications: int = 0
    polynomial_multiplications: int = 0
    polynomial_reduction_updates: int = 0
    basis_forward_block_applications: int = 0
    basis_inverse_block_applications: int = 0
    basis_ring_multiply_accumulations: int = 0
    output_term_ring_multiplications: int = 0
    projection_materializations: int = 0
    maximum_carrier_payload_bits: int = 0
    maximum_residual_signed_bits: int = 1
    maximum_ledger_exponent_signed_bits: int = 1
    maximum_nonzero_message_slots: int = 0
    maximum_nonzero_message_ledgers: int = 0
    maximum_nonzero_coefficient_registers: int = 0
    maximum_nonzero_coefficient_ledgers: int = 0
    maximum_named_transient_ring_element_payload_bits: int = 0
    maximum_named_transient_vector_payload_bits: int = 0
    maximum_pi_integral_basis_carrier_payload_bits: int = 0
    maximum_pi_integral_basis_residual_signed_bits: int = 1


def observe_element(element: RingElement, stats: PiStats) -> None:
    stats.maximum_named_transient_ring_element_payload_bits = max(
        stats.maximum_named_transient_ring_element_payload_bits,
        element_payload_bits(element),
    )


def observe_vector(vector: RingVector, stats: PiStats) -> None:
    stats.maximum_named_transient_vector_payload_bits = max(
        stats.maximum_named_transient_vector_payload_bits,
        vector_payload_bits(vector),
    )


def pi_power(exponent: int, stats: PiStats) -> RingElement:
    if exponent < 0:
        fail("negative pi exponent is outside the integral carrier")
    result = cyclo.ring_one()
    factor = height.PI
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = cyclo.ring_multiply(result, factor)
            stats.pi_power_ring_multiplications += 1
            observe_element(result, stats)
        remaining >>= 1
        if remaining:
            factor = cyclo.ring_multiply(factor, factor)
            stats.pi_power_ring_multiplications += 1
            observe_element(factor, stats)
    return result


def zeta_to_pi_basis(element: RingElement) -> RingElement:
    return tuple(
        (-1) ** degree
        * sum(
            element[source_degree]
            * math.comb(source_degree, degree)
            for source_degree in range(degree, DIMENSION)
        )
        for degree in range(DIMENSION)
    )


def pi_to_zeta_basis(element: RingElement) -> RingElement:
    return tuple(
        (-1) ** degree
        * sum(
            element[source_degree]
            * math.comb(source_degree, degree)
            for source_degree in range(degree, DIMENSION)
        )
        for degree in range(DIMENSION)
    )


@dataclass(frozen=True)
class ScaledElement:
    residual: RingElement
    exponent: int


def scaled_zero() -> ScaledElement:
    return ScaledElement(cyclo.ring_zero(), 0)


def normalize_element(
    element: RingElement,
    base_exponent: int,
    stats: PiStats,
) -> ScaledElement:
    if element == cyclo.ring_zero():
        return scaled_zero()
    residual = element
    valuation = 0
    while sum(residual) % PRIME == 0:
        residual = height.divide_pi_exact(residual)
        valuation += 1
        stats.exact_pi_divisions += 1
    observe_element(residual, stats)
    return ScaledElement(residual, base_exponent + valuation)


def promote_element(
    value: ScaledElement,
    target_exponent: int,
    stats: PiStats,
) -> RingElement:
    if value.residual == cyclo.ring_zero():
        return cyclo.ring_zero()
    if value.exponent < target_exponent:
        fail("scaled element cannot be promoted to a larger base")
    delta = value.exponent - target_exponent
    if delta == 0:
        return value.residual
    promoted = cyclo.ring_multiply(
        pi_power(delta, stats),
        value.residual,
    )
    stats.scaled_element_multiplications += 1
    observe_element(promoted, stats)
    return promoted


def scaled_add(
    left: ScaledElement,
    right: ScaledElement,
    stats: PiStats,
    subtract: bool = False,
) -> ScaledElement:
    if left.residual == cyclo.ring_zero():
        if not subtract:
            return right
        return ScaledElement(
            tuple(-value for value in right.residual),
            right.exponent,
        )
    if right.residual == cyclo.ring_zero():
        return left
    base = min(left.exponent, right.exponent)
    left_value = promote_element(left, base, stats)
    right_value = promote_element(right, base, stats)
    combined = (
        cyclo.ring_subtract(left_value, right_value)
        if subtract
        else cyclo.ring_add(left_value, right_value)
    )
    stats.scaled_element_additions += 1
    observe_element(combined, stats)
    return normalize_element(combined, base, stats)


def scaled_multiply(
    left: ScaledElement,
    right: ScaledElement,
    stats: PiStats,
) -> ScaledElement:
    if (
        left.residual == cyclo.ring_zero()
        or right.residual == cyclo.ring_zero()
    ):
        return scaled_zero()
    product = cyclo.ring_multiply(left.residual, right.residual)
    stats.scaled_element_multiplications += 1
    observe_element(product, stats)
    return normalize_element(
        product,
        left.exponent + right.exponent,
        stats,
    )


def materialize_element(
    value: ScaledElement,
    stats: PiStats,
) -> RingElement:
    if value.residual == cyclo.ring_zero():
        return cyclo.ring_zero()
    materialized = cyclo.ring_multiply(
        pi_power(value.exponent, stats),
        value.residual,
    )
    stats.scaled_element_multiplications += 1
    observe_element(materialized, stats)
    return materialized


@dataclass
class ScaledVector:
    residual: RingVector
    exponent: int


def scaled_zero_vector() -> ScaledVector:
    return ScaledVector(cyclo.zero_vector(), 0)


def normalize_vector(
    vector: RingVector,
    base_exponent: int,
    stats: PiStats,
) -> ScaledVector:
    if cyclo.vector_is_zero(vector):
        return scaled_zero_vector()
    residual = list(vector)
    common = 0
    while all(
        sum(element) % PRIME == 0
        for element in residual
        if element != cyclo.ring_zero()
    ):
        residual = [
            (
                height.divide_pi_exact(element)
                if element != cyclo.ring_zero()
                else cyclo.ring_zero()
            )
            for element in residual
        ]
        stats.exact_pi_divisions += sum(
            element != cyclo.ring_zero()
            for element in residual
        )
        common += 1
    observe_vector(residual, stats)
    return ScaledVector(residual, base_exponent + common)


def promote_vector(
    vector: ScaledVector,
    target_exponent: int,
    stats: PiStats,
) -> RingVector:
    if cyclo.vector_is_zero(vector.residual):
        return cyclo.zero_vector()
    if vector.exponent < target_exponent:
        fail("scaled vector cannot be promoted to a larger base")
    delta = vector.exponent - target_exponent
    if delta == 0:
        return list(vector.residual)
    scalar = pi_power(delta, stats)
    promoted = [
        cyclo.ring_multiply(scalar, element)
        for element in vector.residual
    ]
    stats.scaled_vector_scalar_multiplications += len(vector.residual)
    observe_vector(promoted, stats)
    return promoted


def scaled_vector_add(
    left: ScaledVector,
    right: ScaledVector,
    stats: PiStats,
) -> ScaledVector:
    if cyclo.vector_is_zero(left.residual):
        return right
    if cyclo.vector_is_zero(right.residual):
        return left
    base = min(left.exponent, right.exponent)
    left_value = promote_vector(left, base, stats)
    right_value = promote_vector(right, base, stats)
    combined = [
        cyclo.ring_add(a, b)
        for a, b in zip(left_value, right_value, strict=True)
    ]
    stats.scaled_vector_additions += 1
    observe_vector(combined, stats)
    return normalize_vector(combined, base, stats)


def scaled_vector_scalar_multiply(
    scalar: ScaledElement,
    vector: ScaledVector,
    stats: PiStats,
) -> ScaledVector:
    if (
        scalar.residual == cyclo.ring_zero()
        or cyclo.vector_is_zero(vector.residual)
    ):
        return scaled_zero_vector()
    residual = [
        cyclo.ring_multiply(scalar.residual, element)
        for element in vector.residual
    ]
    stats.output_term_ring_multiplications += len(vector.residual)
    observe_vector(residual, stats)
    return normalize_vector(
        residual,
        scalar.exponent + vector.exponent,
        stats,
    )


def scaled_polynomial_zero() -> list[ScaledElement]:
    return [scaled_zero() for _ in range(DIMENSION)]


def scaled_polynomial_one() -> list[ScaledElement]:
    result = scaled_polynomial_zero()
    result[0] = ScaledElement(cyclo.ring_one(), 0)
    return result


def scaled_polynomial_x() -> list[ScaledElement]:
    result = scaled_polynomial_zero()
    result[1] = ScaledElement(cyclo.ring_one(), 0)
    return result


def scaled_characteristic_factor(
    characteristic: list[RingElement],
    stats: PiStats,
) -> list[ScaledElement]:
    if len(characteristic) != PRIME + 1:
        fail("scaled recurrence characteristic width changed")
    if characteristic[-1] != cyclo.ring_zero():
        fail("scaled recurrence requires zero whole constant")
    return [
        normalize_element(
            characteristic[DIMENSION - degree],
            0,
            stats,
        )
        for degree in range(DIMENSION)
    ]


def multiply_mod_q_scaled(
    left: list[ScaledElement],
    right: list[ScaledElement],
    q_low_to_high: list[ScaledElement],
    stats: PiStats,
) -> list[ScaledElement]:
    if (
        len(left) != DIMENSION
        or len(right) != DIMENSION
        or len(q_low_to_high) != DIMENSION
    ):
        fail("scaled polynomial width changed")
    product = [scaled_zero() for _ in range(2 * DIMENSION - 1)]
    for left_degree, left_value in enumerate(left):
        for right_degree, right_value in enumerate(right):
            term = scaled_multiply(left_value, right_value, stats)
            product[left_degree + right_degree] = scaled_add(
                product[left_degree + right_degree],
                term,
                stats,
            )
    for degree in range(2 * DIMENSION - 2, DIMENSION - 1, -1):
        factor = product[degree]
        if factor.residual == cyclo.ring_zero():
            continue
        for q_degree, q_coefficient in enumerate(q_low_to_high):
            target = degree - DIMENSION + q_degree
            term = scaled_multiply(factor, q_coefficient, stats)
            product[target] = scaled_add(
                product[target],
                term,
                stats,
                subtract=True,
            )
            stats.polynomial_reduction_updates += 1
        product[degree] = scaled_zero()
    stats.polynomial_multiplications += 1
    return product[:DIMENSION]


def scaled_recurrence_coefficients(
    periods: int,
    characteristic: list[RingElement],
    stats: PiStats,
) -> list[ScaledElement]:
    if periods < 1:
        fail("scaled recurrence requires a positive period")
    q_low_to_high = scaled_characteristic_factor(
        characteristic,
        stats,
    )
    exponent = periods - 1
    result = scaled_polynomial_one()
    base = scaled_polynomial_x()
    while exponent:
        if exponent & 1:
            result = multiply_mod_q_scaled(
                result,
                base,
                q_low_to_high,
                stats,
            )
        exponent >>= 1
        if exponent:
            base = multiply_mod_q_scaled(
                base,
                base,
                q_low_to_high,
                stats,
            )
    return result


@dataclass
class PiCarrier:
    messages: list[RingVector]
    message_pi_exponents: list[int]
    coefficients: list[RingElement]
    coefficient_pi_exponents: list[int]
    generation: int = 0
    lease: int = 0
    active: bool = False
    pending_operations: int = 0
    phase: str = "RESTORED"

    @classmethod
    def create(cls) -> "PiCarrier":
        return cls(
            messages=[
                cyclo.zero_vector()
                for _ in range(MESSAGE_SLOTS)
            ],
            message_pi_exponents=[0 for _ in range(MESSAGE_SLOTS)],
            coefficients=recurrence.polynomial_zero(),
            coefficient_pi_exponents=[
                0 for _ in range(COEFFICIENT_REGISTERS)
            ],
        )

    def all_zero(self) -> bool:
        return (
            all(cyclo.vector_is_zero(row) for row in self.messages)
            and not any(self.message_pi_exponents)
            and all(
                value == cyclo.ring_zero()
                for value in self.coefficients
            )
            and not any(self.coefficient_pi_exponents)
        )

    def backing_identity(self) -> tuple[int, ...]:
        return (
            id(self.messages),
            *(id(message) for message in self.messages),
            id(self.message_pi_exponents),
            id(self.coefficients),
            id(self.coefficient_pi_exponents),
        )

    def canonical_state(self) -> dict[str, Any]:
        return {
            "message_slots": len(self.messages),
            "message_pi_exponent_ledgers": len(
                self.message_pi_exponents
            ),
            "coefficient_registers": len(self.coefficients),
            "coefficient_pi_exponent_ledgers": len(
                self.coefficient_pi_exponents
            ),
            "all_payload_and_ledgers_zero": self.all_zero(),
            "generation": self.generation,
            "lease": self.lease,
            "active": self.active,
            "pending_operations": self.pending_operations,
            "phase": self.phase,
        }


def record_peak(carrier: PiCarrier, stats: PiStats) -> None:
    payload = 0
    pi_basis_payload = 0
    pi_basis_maximum_bits = 1
    nonzero_messages = 0
    nonzero_message_ledgers = 0
    nonzero_coefficients = 0
    nonzero_coefficient_ledgers = 0
    for message, exponent in zip(
        carrier.messages,
        carrier.message_pi_exponents,
        strict=True,
    ):
        payload += vector_payload_bits(message)
        payload += signed_bits(exponent)
        pi_basis_message = [
            zeta_to_pi_basis(element)
            for element in message
        ]
        pi_basis_payload += vector_payload_bits(pi_basis_message)
        pi_basis_payload += signed_bits(exponent)
        pi_basis_maximum_bits = max(
            pi_basis_maximum_bits,
            vector_maximum_signed_bits(pi_basis_message),
        )
        nonzero_messages += int(not cyclo.vector_is_zero(message))
        nonzero_message_ledgers += int(exponent != 0)
        stats.maximum_residual_signed_bits = max(
            stats.maximum_residual_signed_bits,
            vector_maximum_signed_bits(message),
        )
        stats.maximum_ledger_exponent_signed_bits = max(
            stats.maximum_ledger_exponent_signed_bits,
            signed_bits(exponent),
        )
    for element, exponent in zip(
        carrier.coefficients,
        carrier.coefficient_pi_exponents,
        strict=True,
    ):
        payload += element_payload_bits(element)
        payload += signed_bits(exponent)
        pi_basis_element = zeta_to_pi_basis(element)
        pi_basis_payload += element_payload_bits(pi_basis_element)
        pi_basis_payload += signed_bits(exponent)
        pi_basis_maximum_bits = max(
            pi_basis_maximum_bits,
            element_maximum_signed_bits(pi_basis_element),
        )
        nonzero_coefficients += int(element != cyclo.ring_zero())
        nonzero_coefficient_ledgers += int(exponent != 0)
        stats.maximum_residual_signed_bits = max(
            stats.maximum_residual_signed_bits,
            element_maximum_signed_bits(element),
        )
        stats.maximum_ledger_exponent_signed_bits = max(
            stats.maximum_ledger_exponent_signed_bits,
            signed_bits(exponent),
        )
    stats.maximum_carrier_payload_bits = max(
        stats.maximum_carrier_payload_bits,
        payload,
    )
    stats.maximum_pi_integral_basis_carrier_payload_bits = max(
        stats.maximum_pi_integral_basis_carrier_payload_bits,
        pi_basis_payload,
    )
    stats.maximum_pi_integral_basis_residual_signed_bits = max(
        stats.maximum_pi_integral_basis_residual_signed_bits,
        pi_basis_maximum_bits,
    )
    stats.maximum_nonzero_message_slots = max(
        stats.maximum_nonzero_message_slots,
        nonzero_messages,
    )
    stats.maximum_nonzero_message_ledgers = max(
        stats.maximum_nonzero_message_ledgers,
        nonzero_message_ledgers,
    )
    stats.maximum_nonzero_coefficient_registers = max(
        stats.maximum_nonzero_coefficient_registers,
        nonzero_coefficients,
    )
    stats.maximum_nonzero_coefficient_ledgers = max(
        stats.maximum_nonzero_coefficient_ledgers,
        nonzero_coefficient_ledgers,
    )


def store_scaled_vector(
    carrier: PiCarrier,
    index: int,
    value: ScaledVector,
) -> None:
    if not cyclo.vector_is_zero(carrier.messages[index]):
        fail("message target was not clean")
    if carrier.message_pi_exponents[index] != 0:
        fail("message exponent target was not clean")
    cyclo.copy_vector_into(carrier.messages[index], value.residual)
    carrier.message_pi_exponents[index] = value.exponent


def carrier_scaled_vector(
    carrier: PiCarrier,
    index: int,
) -> ScaledVector:
    return ScaledVector(
        list(carrier.messages[index]),
        carrier.message_pi_exponents[index],
    )


def build_scaled_basis(
    carrier: PiCarrier,
    block: cyclo.CompiledBlock,
    stats: PiStats,
) -> None:
    seed = normalize_vector(
        cyclo.seed_vector(block.public_program),
        0,
        stats,
    )
    store_scaled_vector(carrier, 0, seed)
    block_stats = cyclo.Stats()
    for index in range(1, BASIS_MESSAGES + 1):
        previous = carrier_scaled_vector(carrier, index - 1)
        raw_residual = cyclo.apply_operator(
            block.operator,
            previous.residual,
            block_stats,
        )
        next_value = normalize_vector(
            raw_residual,
            previous.exponent,
            stats,
        )
        store_scaled_vector(carrier, index, next_value)
    stats.basis_forward_block_applications += BASIS_MESSAGES
    stats.basis_ring_multiply_accumulations += (
        block_stats.ring_multiply_accumulations
    )
    record_peak(carrier, stats)


def load_scaled_coefficients(
    carrier: PiCarrier,
    values: list[ScaledElement],
) -> None:
    if len(values) != COEFFICIENT_REGISTERS:
        fail("scaled coefficient register width changed")
    if any(value != cyclo.ring_zero() for value in carrier.coefficients):
        fail("scaled coefficient residual targets were not clean")
    if any(carrier.coefficient_pi_exponents):
        fail("scaled coefficient exponent targets were not clean")
    for index, value in enumerate(values):
        carrier.coefficients[index] = value.residual
        carrier.coefficient_pi_exponents[index] = value.exponent


def output_from_scaled_basis(
    carrier: PiCarrier,
    stats: PiStats,
) -> ScaledVector:
    output = scaled_zero_vector()
    for index in range(COEFFICIENT_REGISTERS):
        scalar = ScaledElement(
            carrier.coefficients[index],
            carrier.coefficient_pi_exponents[index],
        )
        basis = carrier_scaled_vector(carrier, index + 1)
        term = scaled_vector_scalar_multiply(scalar, basis, stats)
        output = scaled_vector_add(output, term, stats)
    return output


def project_scaled_boundary(
    output: ScaledVector,
    stats: PiStats,
) -> RingElement:
    projected = cyclo.project_boundary(output.residual)
    scaled = normalize_element(projected, output.exponent, stats)
    stats.projection_materializations += 1
    return materialize_element(scaled, stats)


def populate_forward(
    carrier: PiCarrier,
    block: cyclo.CompiledBlock,
    periods: int,
) -> tuple[RingElement, PiStats]:
    if carrier.active or carrier.pending_operations or not carrier.all_zero():
        fail("pi-content carrier was not restored")
    carrier.active = True
    carrier.lease += 1
    carrier.pending_operations = 1
    carrier.phase = "BUILD_SCALED_BASIS"
    stats = PiStats()
    build_scaled_basis(carrier, block, stats)
    coefficients = scaled_recurrence_coefficients(
        periods,
        block.characteristic,
        stats,
    )
    load_scaled_coefficients(carrier, coefficients)
    carrier.phase = "SCALED_COEFFICIENTS_RESIDENT"
    output = output_from_scaled_basis(carrier, stats)
    store_scaled_vector(carrier, OUTPUT_SLOT, output)
    carrier.phase = "SCALED_OUTPUT_RESIDENT"
    record_peak(carrier, stats)
    return project_scaled_boundary(output, stats), stats


def subtract_scaled_vector(
    carrier: PiCarrier,
    index: int,
    expected: ScaledVector,
) -> None:
    if carrier.message_pi_exponents[index] != expected.exponent:
        fail("scaled message exponent did not match rematerialization")
    cyclo.subtract_vector_exact(
        carrier.messages[index],
        expected.residual,
    )
    carrier.message_pi_exponents[index] -= expected.exponent


def restore_forward(
    carrier: PiCarrier,
    block: cyclo.CompiledBlock,
    periods: int,
    stats: PiStats,
) -> None:
    if carrier.phase != "SCALED_OUTPUT_RESIDENT":
        fail("pi-content inverse was reordered")
    expected_output = output_from_scaled_basis(carrier, stats)
    subtract_scaled_vector(carrier, OUTPUT_SLOT, expected_output)
    carrier.phase = "SCALED_COEFFICIENTS_RESIDENT"

    expected_coefficients = scaled_recurrence_coefficients(
        periods,
        block.characteristic,
        stats,
    )
    for index, expected in enumerate(expected_coefficients):
        if (
            carrier.coefficient_pi_exponents[index]
            != expected.exponent
        ):
            fail("scaled coefficient exponent did not match inverse")
        if carrier.coefficients[index] != expected.residual:
            fail("scaled coefficient residual did not match inverse")
        carrier.coefficients[index] = cyclo.ring_subtract(
            carrier.coefficients[index],
            expected.residual,
        )
        carrier.coefficient_pi_exponents[index] -= expected.exponent
    carrier.phase = "SCALED_BASIS_RESIDENT"

    block_stats = cyclo.Stats()
    for index in range(BASIS_MESSAGES, 0, -1):
        previous = carrier_scaled_vector(carrier, index - 1)
        raw_residual = cyclo.apply_operator(
            block.operator,
            previous.residual,
            block_stats,
        )
        expected = normalize_vector(
            raw_residual,
            previous.exponent,
            stats,
        )
        subtract_scaled_vector(carrier, index, expected)
    stats.basis_inverse_block_applications += BASIS_MESSAGES
    stats.basis_ring_multiply_accumulations += (
        block_stats.ring_multiply_accumulations
    )
    seed = normalize_vector(
        cyclo.seed_vector(block.public_program),
        0,
        stats,
    )
    subtract_scaled_vector(carrier, 0, seed)
    carrier.pending_operations = 0
    carrier.active = False
    carrier.phase = "RESTORED"
    carrier.generation += 1
    record_peak(carrier, stats)
    if not carrier.all_zero():
        fail("pi-content carrier did not restore exactly")


@dataclass
class Transaction:
    boundary: RingElement
    stats: PiStats
    restored_exactly: bool
    same_backing: bool


def execute_transaction(
    carrier: PiCarrier,
    block: cyclo.CompiledBlock,
    periods: int,
) -> Transaction:
    if not isinstance(carrier, PiCarrier):
        fail("null or invalid pi-content carrier")
    backing = carrier.backing_identity()
    boundary, stats = populate_forward(carrier, block, periods)
    restore_forward(carrier, block, periods, stats)
    return Transaction(
        boundary=boundary,
        stats=stats,
        restored_exactly=carrier.all_zero(),
        same_backing=carrier.backing_identity() == backing,
    )


def raw_recurrence_baseline(
    block: cyclo.CompiledBlock,
    periods: int,
) -> dict[str, Any]:
    carrier = recurrence.Carrier.create()
    boundary, stats = recurrence.populate_forward(
        carrier,
        block,
        periods,
    )
    recurrence.restore_forward(
        carrier,
        block,
        periods,
        stats,
    )
    return {
        "boundary": boundary,
        "maximum_carrier_payload_bits": (
            stats.maximum_carrier_payload_bits
        ),
        "maximum_coefficient_signed_bits": (
            stats.maximum_coefficient_signed_bits
        ),
        "polynomial_multiplications": stats.polynomial_multiplications,
        "restored_exactly": carrier.all_zero(),
        "message_integer_cells": (
            MESSAGE_SLOTS * MESSAGE_INTEGER_CELLS
        ),
        "coefficient_integer_cells": (
            COEFFICIENT_REGISTERS * DIMENSION
        ),
    }


def stats_json(stats: PiStats) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in vars(stats).items()
    }


def case_result(
    periods: int,
    block: cyclo.CompiledBlock,
) -> dict[str, Any]:
    carrier = PiCarrier.create()
    transaction = execute_transaction(carrier, block, periods)
    baseline = raw_recurrence_baseline(block, periods)
    boundary = transaction.boundary
    boundary_valuation = height.pi_valuation(boundary)
    if boundary_valuation is None:
        fail("pi-content boundary unexpectedly vanished")
    return {
        "periods": periods,
        "equivalent_edges": periods * cyclo.PERIOD,
        "family": block.family,
        "boundary_sha256": hashlib.sha256(
            cyclo.encoded_ring_object(boundary)
        ).hexdigest(),
        "boundary_payload_bits": element_payload_bits(boundary),
        "boundary_pi_valuation": boundary_valuation,
        "raw_recurrence_boundary_equal": (
            boundary == baseline["boundary"]
        ),
        "raw_recurrence_baseline": {
            key: value
            for key, value in baseline.items()
            if key != "boundary"
        },
        "pi_content_stats": stats_json(transaction.stats),
        "carrier_payload_reduction_bits": (
            baseline["maximum_carrier_payload_bits"]
            - transaction.stats.maximum_carrier_payload_bits
        ),
        "carrier_payload_reduction_positive": (
            transaction.stats.maximum_carrier_payload_bits
            < baseline["maximum_carrier_payload_bits"]
        ),
        "pi_integral_basis_carrier_payload_reduction_positive": (
            transaction.stats
            .maximum_pi_integral_basis_carrier_payload_bits
            < baseline["maximum_carrier_payload_bits"]
        ),
        "message_slots": MESSAGE_SLOTS,
        "message_residual_integer_cells": (
            MESSAGE_SLOTS * MESSAGE_INTEGER_CELLS
        ),
        "message_pi_exponent_ledger_cells": MESSAGE_SLOTS,
        "coefficient_residual_integer_cells": (
            COEFFICIENT_REGISTERS * DIMENSION
        ),
        "coefficient_pi_exponent_ledger_cells": (
            COEFFICIENT_REGISTERS
        ),
        "restored_exactly": transaction.restored_exactly,
        "same_backing": transaction.same_backing,
        "canonical_restored_state": carrier.canonical_state(),
    }


def restoration_reuse_case(
    primary: cyclo.CompiledBlock,
    reuse: cyclo.CompiledBlock,
) -> dict[str, Any]:
    periods = max(TESTED_PERIODS)
    carrier = PiCarrier.create()
    backing = carrier.backing_identity()
    primary_transaction = execute_transaction(
        carrier,
        primary,
        periods,
    )
    reuse_transaction = execute_transaction(
        carrier,
        reuse,
        periods,
    )
    fresh_transaction = execute_transaction(
        PiCarrier.create(),
        reuse,
        periods,
    )
    return {
        "periods": periods,
        "restoration_scope": (
            "EXACT_PAYLOAD_AND_PI_LEDGER_ZERO_ON_SAME_BACKING_"
            "WITH_MONOTONE_GENERATION_AND_LEASE_METADATA"
        ),
        "primary_restored_exactly": (
            primary_transaction.restored_exactly
        ),
        "reuse_restored_exactly": reuse_transaction.restored_exactly,
        "same_original_backing": (
            primary_transaction.same_backing
            and reuse_transaction.same_backing
            and carrier.backing_identity() == backing
        ),
        "fresh_restored_reuse_boundary_equal": (
            reuse_transaction.boundary == fresh_transaction.boundary
        ),
        "generation": carrier.generation,
        "lease": carrier.lease,
        "canonical_restored_state": carrier.canonical_state(),
        "full_carrier_object_state_equal": False,
        "repeated_use_metadata_width_bounded": False,
        "retained_inverse_history_bytes": 0,
        "baseline_reload_bytes": 0,
    }


def controls(
    primary: cyclo.CompiledBlock,
    reuse: cyclo.CompiledBlock,
) -> dict[str, bool]:
    missing = PiCarrier.create()
    populate_forward(missing, primary, 4)
    missing_inverse_leaves_nonzero = not missing.all_zero()

    wrong = PiCarrier.create()
    _, wrong_stats = populate_forward(wrong, primary, 4)
    wrong_inverse_rejected = False
    try:
        restore_forward(wrong, reuse, 4, wrong_stats)
    except RuntimeError:
        wrong_inverse_rejected = True

    reordered = PiCarrier.create()
    _, reordered_stats = populate_forward(reordered, primary, 4)
    reordered.phase = "SCALED_BASIS_RESIDENT"
    reordered_inverse_rejected = False
    try:
        restore_forward(reordered, primary, 4, reordered_stats)
    except RuntimeError:
        reordered_inverse_rejected = True

    null_carrier_rejected = False
    try:
        execute_transaction(None, primary, 4)  # type: ignore[arg-type]
    except RuntimeError:
        null_carrier_rejected = True

    primary_boundary = execute_transaction(
        PiCarrier.create(),
        primary,
        4,
    ).boundary
    reuse_boundary = execute_transaction(
        PiCarrier.create(),
        reuse,
        4,
    ).boundary
    return {
        "missing_inverse_leaves_nonzero": (
            missing_inverse_leaves_nonzero
        ),
        "wrong_inverse_rejected": wrong_inverse_rejected,
        "reordered_inverse_rejected": reordered_inverse_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "semantic_family_perturbation_changes_boundary": (
            primary_boundary != reuse_boundary
        ),
    }


def pi_basis_transform_control(
    blocks: dict[str, cyclo.CompiledBlock],
) -> bool:
    elements = [
        element
        for block in blocks.values()
        for row in block.operator
        for element in row
    ]
    elements.extend(
        element
        for block in blocks.values()
        for element in block.characteristic
    )
    return all(
        pi_to_zeta_basis(zeta_to_pi_basis(element)) == element
        for element in elements
    )


def main() -> int:
    if len(sys.argv) != 1:
        fail(
            "usage: f17_cubic_chain_period17_"
            "pi_content_recurrence.py"
        )
    blocks = {
        family.lower(): cyclo.build_compiled_block(family)
        for family in ("PRIMARY", "REUSE")
    }
    cases = [
        case_result(periods, blocks[family])
        for periods in TESTED_PERIODS
        for family in ("primary", "reuse")
    ]
    restored = restoration_reuse_case(
        blocks["primary"],
        blocks["reuse"],
    )
    control_results = controls(
        blocks["primary"],
        blocks["reuse"],
    )
    basis_transform_roundtrip = pi_basis_transform_control(blocks)
    result = {
        "result": "PASS",
        "experiment": (
            "EXACT_REVERSIBLE_PI_CONTENT_LEDGER_NORMALIZED_"
            "NATIVE_K_RECURRENCE_WITH_RESIDUAL_HEIGHT_ACCOUNTING"
        ),
        "claim_candidate": (
            "BOUNDED_EXACT_PI_CONTENT_LEDGER_NORMALIZED_NATIVE_K_"
            "RECURRENCE_WORSENS_CARRIER_PAYLOAD_IN_ZETA_AND_PI_"
            "INTEGRAL_BASES_FOR_TWO_PUBLIC_F17_PERIOD17_FAMILIES_"
            "ACROSS_PERIODS1_4_16_64_256_WITH_EXACT_PAYLOAD_AND_"
            "LEDGER_RESTORATION_AND_REUSE"
        ),
        "classification_candidate": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level_candidate": "PACKAGE_SELF_REVIEW",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "coefficient_field": "Q_ZETA17",
        "integral_carrier_ring": "Z_ZETA17",
        "uniformizer": "PI_EQUALS_1_MINUS_ZETA17",
        "tested_periods": list(TESTED_PERIODS),
        "block_certificates": {
            family: {
                "public_program_sha256": hashlib.sha256(
                    cyclo.adaptive.encoded_program(block.public_program)
                ).hexdigest(),
                "operator_sha256": block.operator_sha256,
                "characteristic_sha256": (
                    block.characteristic_sha256
                ),
                "characteristic_identity_exact": (
                    block.characteristic_identity_exact
                ),
                "characteristic": block.characteristic,
            }
            for family, block in blocks.items()
        },
        "accepted_path_materializes_raw_recurrence_coefficients": False,
        "accepted_path_factors_pi_content_during_polynomial_arithmetic": (
            True
        ),
        "cases": cases,
        "all_raw_recurrence_boundaries_equal": all(
            case["raw_recurrence_boundary_equal"]
            for case in cases
        ),
        "all_cases_restore_exactly": all(
            case["restored_exactly"]
            and case["same_backing"]
            and case["canonical_restored_state"][
                "all_payload_and_ledgers_zero"
            ]
            for case in cases
        ),
        "all_cases_reduce_carrier_payload": all(
            case["carrier_payload_reduction_positive"]
            for case in cases
        ),
        "all_cases_worsen_carrier_payload": all(
            case["carrier_payload_reduction_bits"] < 0
            for case in cases
        ),
        "all_cases_worsen_carrier_payload_in_pi_integral_basis": all(
            case["pi_content_stats"][
                "maximum_pi_integral_basis_carrier_payload_bits"
            ]
            > case["raw_recurrence_baseline"][
                "maximum_carrier_payload_bits"
            ]
            for case in cases
        ),
        "restoration_reuse_case": restored,
        "controls": control_results,
        "pi_integral_basis_transform_roundtrip_exact": (
            basis_transform_roundtrip
        ),
        "failure_atomic_restoration_established": False,
        "matched_classical": {
            "identical_pi_content_normalized_recurrence_available": (
                True
            ),
            "identical_residual_and_exponent_ledger_available": True,
            "raw_recurrence_retained_as_counted_verification_baseline": (
                True
            ),
            "comparison_establishes_advantage": False,
        },
        "observation": (
            "PI_CONTENT_LEDGER_WIDTH_STAYS_SMALL_BUT_RESIDUAL_"
            "HEIGHT_AND_CARRIER_PAYLOAD_INCREASE_IN_BOTH_THE_"
            "OMITTED_ROOT_ZETA_BASIS_AND_THE_INTEGRAL_PI_BASIS"
        ),
        "resource_law": {
            "accepted_carrier_message_residual_integer_cells": (
                MESSAGE_SLOTS * MESSAGE_INTEGER_CELLS
            ),
            "accepted_carrier_message_pi_exponent_ledger_cells": (
                MESSAGE_SLOTS
            ),
            "accepted_carrier_coefficient_residual_integer_cells": (
                COEFFICIENT_REGISTERS * DIMENSION
            ),
            "accepted_carrier_coefficient_pi_exponent_ledger_cells": (
                COEFFICIENT_REGISTERS
            ),
            "compiled_operator_integer_cells_two_families": (
                2 * cyclo.OPERATOR_INTEGER_CELLS
            ),
            "compiled_characteristic_integer_cells_two_families": (
                2 * (PRIME + 1) * DIMENSION
            ),
            "final_projection_materializes_exact_boundary": True,
            "exact_triangular_zeta_to_pi_integral_basis_control": True,
            "verification_baseline_materializes_raw_coefficients": True,
            "named_transient_pi_powers_counted_in_stats": True,
            "python_object_overhead_bounded": False,
            "sympy_internal_temporaries_bounded": False,
            "allocator_peak_bounded": False,
            "whole_process_peak_bounded": False,
        },
        "not_established": [
            "INTRINSIC_LOWER_BOUND_ACROSS_ALL_INTEGRAL_BASES",
            "OPTIMAL_CYCLOTOMIC_UNIT_BALANCING_AFTER_PI_FACTORING",
            "FIXED_RESIDUAL_INTEGER_WIDTH",
            "FIXED_TOTAL_BIT_FOOTPRINT",
            "BOUNDED_PI_EXPONENT_LEDGER_WIDTH_ACROSS_UNBOUNDED_DEPTH",
            "FULL_CARRIER_OBJECT_STATE_EQUALITY_AFTER_RESTORATION",
            "BOUNDED_REPEATED_USE_GENERATION_AND_LEASE_METADATA",
            "FAILURE_ATOMIC_ROLLBACK_AFTER_REJECTED_INVERSE",
            "MACHINE_ENFORCED_NO_SMUGGLE_OR_CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "CATALYTIC_INFERENCE",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_COMPUTATION",
        ],
        "next_experiment": (
            "EXACT_MULTI_EMBEDDING_CYCLOTOMIC_UNIT_BALANCING_"
            "AFTER_PI_CONTENT_FACTORIZATION"
        ),
        "next_obstruction": (
            "COMPULSORY_PI_SCALE_HAS_A_COMPACT_LEDGER_BUT_"
            "DIVIDING_IT_EXPANDS_RESIDUAL_HEIGHT_IN_BOTH_TESTED_"
            "INTEGRAL_BASES_AND_THE_IDENTICAL_CLASSICAL_"
            "NORMALIZATION_REMAINS"
        ),
        "terminal": False,
    }
    if (
        not result["all_raw_recurrence_boundaries_equal"]
        or not result["all_cases_restore_exactly"]
        or not result["all_cases_worsen_carrier_payload"]
        or not result[
            "all_cases_worsen_carrier_payload_in_pi_integral_basis"
        ]
        or not all(control_results.values())
        or not basis_transform_roundtrip
        or not restored["primary_restored_exactly"]
        or not restored["reuse_restored_exactly"]
        or not restored["same_original_backing"]
        or not restored["fresh_restored_reuse_boundary_equal"]
    ):
        fail("pi-content recurrence qualification failed")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
