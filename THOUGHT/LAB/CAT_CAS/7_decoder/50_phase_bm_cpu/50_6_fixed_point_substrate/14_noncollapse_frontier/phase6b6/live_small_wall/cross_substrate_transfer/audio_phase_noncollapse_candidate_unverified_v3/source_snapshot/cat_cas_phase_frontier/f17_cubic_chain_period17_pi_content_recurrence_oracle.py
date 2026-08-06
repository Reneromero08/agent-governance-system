#!/usr/bin/env python3
"""Separate exact oracle for the pi-content recurrence diagnostic.

This oracle imports only previously sealed independent reference kernels.
It recompiles both public period operators, checks their annihilators, and
uses sequential x-mod-q advancement rather than production binary polynomial
powering.  Its scaled arithmetic implementation is separate from production.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_height_lower_bound_oracle as height_reference
import f17_cubic_chain_period17_unit_height_reduction_oracle as reference


PRIME = 17
DIMENSION = 16
MESSAGE_SLOTS = 18
OUTPUT_SLOT = MESSAGE_SLOTS - 1
EXPECTED_PERIODS = (1, 4, 16, 64, 256)

RingElement = tuple[int, ...]
RingVector = list[RingElement]
RingMatrix = list[list[RingElement]]


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


def pi_power(exponent: int) -> RingElement:
    if exponent < 0:
        fail("oracle pi exponent became negative")
    result = reference.ring_one()
    factor = height_reference.PI
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = reference.ring_multiply(result, factor)
        remaining >>= 1
        if remaining:
            factor = reference.ring_multiply(factor, factor)
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
    return ScaledElement(reference.ring_zero(), 0)


def normalize_element(
    element: RingElement,
    base_exponent: int = 0,
) -> ScaledElement:
    if element == reference.ring_zero():
        return scaled_zero()
    current = element
    exponent = base_exponent
    while sum(current) % PRIME == 0:
        current = height_reference.divide_pi_exact(current)
        exponent += 1
    return ScaledElement(current, exponent)


def promote_element(
    value: ScaledElement,
    base: int,
) -> RingElement:
    if value.residual == reference.ring_zero():
        return reference.ring_zero()
    if value.exponent < base:
        fail("oracle scaled promotion reversed")
    if value.exponent == base:
        return value.residual
    return reference.ring_multiply(
        pi_power(value.exponent - base),
        value.residual,
    )


def scaled_add(
    left: ScaledElement,
    right: ScaledElement,
    subtract: bool = False,
) -> ScaledElement:
    if left.residual == reference.ring_zero():
        if not subtract:
            return right
        return ScaledElement(
            tuple(-value for value in right.residual),
            right.exponent,
        )
    if right.residual == reference.ring_zero():
        return left
    base = min(left.exponent, right.exponent)
    left_value = promote_element(left, base)
    right_value = promote_element(right, base)
    combined = (
        reference.ring_subtract(left_value, right_value)
        if subtract
        else reference.ring_add(left_value, right_value)
    )
    return normalize_element(combined, base)


def scaled_multiply(
    left: ScaledElement,
    right: ScaledElement,
) -> ScaledElement:
    if (
        left.residual == reference.ring_zero()
        or right.residual == reference.ring_zero()
    ):
        return scaled_zero()
    return normalize_element(
        reference.ring_multiply(left.residual, right.residual),
        left.exponent + right.exponent,
    )


def materialize_element(value: ScaledElement) -> RingElement:
    if value.residual == reference.ring_zero():
        return reference.ring_zero()
    return reference.ring_multiply(
        pi_power(value.exponent),
        value.residual,
    )


@dataclass
class ScaledVector:
    residual: RingVector
    exponent: int


def scaled_zero_vector() -> ScaledVector:
    return ScaledVector(
        [reference.ring_zero() for _ in range(PRIME)],
        0,
    )


def vector_is_zero(vector: RingVector) -> bool:
    return all(element == reference.ring_zero() for element in vector)


def normalize_vector(
    vector: RingVector,
    base_exponent: int = 0,
) -> ScaledVector:
    if vector_is_zero(vector):
        return scaled_zero_vector()
    current = list(vector)
    exponent = base_exponent
    while all(
        sum(element) % PRIME == 0
        for element in current
        if element != reference.ring_zero()
    ):
        current = [
            (
                height_reference.divide_pi_exact(element)
                if element != reference.ring_zero()
                else reference.ring_zero()
            )
            for element in current
        ]
        exponent += 1
    return ScaledVector(current, exponent)


def promote_vector(
    vector: ScaledVector,
    base: int,
) -> RingVector:
    if vector_is_zero(vector.residual):
        return [reference.ring_zero() for _ in range(PRIME)]
    if vector.exponent < base:
        fail("oracle vector promotion reversed")
    if vector.exponent == base:
        return list(vector.residual)
    scalar = pi_power(vector.exponent - base)
    return [
        reference.ring_multiply(scalar, element)
        for element in vector.residual
    ]


def scaled_vector_add(
    left: ScaledVector,
    right: ScaledVector,
) -> ScaledVector:
    if vector_is_zero(left.residual):
        return right
    if vector_is_zero(right.residual):
        return left
    base = min(left.exponent, right.exponent)
    return normalize_vector(
        [
            reference.ring_add(a, b)
            for a, b in zip(
                promote_vector(left, base),
                promote_vector(right, base),
                strict=True,
            )
        ],
        base,
    )


def scaled_vector_scalar_multiply(
    scalar: ScaledElement,
    vector: ScaledVector,
) -> ScaledVector:
    if (
        scalar.residual == reference.ring_zero()
        or vector_is_zero(vector.residual)
    ):
        return scaled_zero_vector()
    return normalize_vector(
        [
            reference.ring_multiply(scalar.residual, element)
            for element in vector.residual
        ],
        scalar.exponent + vector.exponent,
    )


def scaled_characteristic(
    characteristic: list[RingElement],
) -> list[ScaledElement]:
    return [
        normalize_element(characteristic[DIMENSION - degree])
        for degree in range(DIMENSION)
    ]


def sequential_scaled_coefficients(
    periods: int,
    characteristic: list[RingElement],
) -> list[ScaledElement]:
    if periods < 1:
        fail("oracle sequential recurrence requires a positive period")
    q_low = scaled_characteristic(characteristic)
    coefficients = [scaled_zero() for _ in range(DIMENSION)]
    coefficients[0] = ScaledElement(reference.ring_one(), 0)
    for _ in range(periods - 1):
        highest = coefficients[-1]
        advanced = [scaled_zero() for _ in range(DIMENSION)]
        for degree in range(DIMENSION):
            shifted = (
                coefficients[degree - 1]
                if degree > 0
                else scaled_zero()
            )
            advanced[degree] = scaled_add(
                shifted,
                scaled_multiply(highest, q_low[degree]),
                subtract=True,
            )
        coefficients = advanced
    return coefficients


def build_scaled_basis(
    operator: RingMatrix,
    seed: RingVector,
) -> list[ScaledVector]:
    basis = []
    current = normalize_vector(seed)
    for _ in range(DIMENSION):
        current = normalize_vector(
            reference.matrix_vector_multiply(
                operator,
                current.residual,
            ),
            current.exponent,
        )
        basis.append(current)
    return basis


def scaled_output(
    coefficients: list[ScaledElement],
    basis: list[ScaledVector],
) -> ScaledVector:
    output = scaled_zero_vector()
    for coefficient, vector in zip(
        coefficients,
        basis,
        strict=True,
    ):
        output = scaled_vector_add(
            output,
            scaled_vector_scalar_multiply(coefficient, vector),
        )
    return output


def scaled_boundary(output: ScaledVector) -> RingElement:
    projected = reference.project(output.residual)
    return materialize_element(
        normalize_element(projected, output.exponent)
    )


def carrier_metrics(
    seed: ScaledVector,
    basis: list[ScaledVector],
    coefficients: list[ScaledElement],
    output: ScaledVector,
) -> dict[str, int]:
    zero_vector = scaled_zero_vector()
    zero_coefficients = [scaled_zero() for _ in range(DIMENSION)]
    stages = [
        ([seed, *basis, zero_vector], zero_coefficients),
        ([seed, *basis, output], coefficients),
    ]
    maximum_payload = 0
    maximum_pi_payload = 0
    maximum_width = 1
    maximum_pi_width = 1
    maximum_ledger_width = 1
    maximum_nonzero_messages = 0
    maximum_nonzero_message_ledgers = 0
    maximum_nonzero_coefficients = 0
    maximum_nonzero_coefficient_ledgers = 0
    for messages, registers in stages:
        payload = 0
        pi_payload = 0
        nonzero_messages = 0
        nonzero_message_ledgers = 0
        for message in messages:
            payload += vector_payload_bits(message.residual)
            payload += signed_bits(message.exponent)
            pi_vector = [
                zeta_to_pi_basis(element)
                for element in message.residual
            ]
            pi_payload += vector_payload_bits(pi_vector)
            pi_payload += signed_bits(message.exponent)
            maximum_width = max(
                maximum_width,
                vector_width(message.residual),
            )
            maximum_pi_width = max(
                maximum_pi_width,
                vector_width(pi_vector),
            )
            maximum_ledger_width = max(
                maximum_ledger_width,
                signed_bits(message.exponent),
            )
            nonzero_messages += int(
                not vector_is_zero(message.residual)
            )
            nonzero_message_ledgers += int(message.exponent != 0)
        nonzero_coefficients = 0
        nonzero_coefficient_ledgers = 0
        for register in registers:
            payload += element_payload_bits(register.residual)
            payload += signed_bits(register.exponent)
            pi_element = zeta_to_pi_basis(register.residual)
            pi_payload += element_payload_bits(pi_element)
            pi_payload += signed_bits(register.exponent)
            maximum_width = max(
                maximum_width,
                element_width(register.residual),
            )
            maximum_pi_width = max(
                maximum_pi_width,
                element_width(pi_element),
            )
            maximum_ledger_width = max(
                maximum_ledger_width,
                signed_bits(register.exponent),
            )
            nonzero_coefficients += int(
                register.residual != reference.ring_zero()
            )
            nonzero_coefficient_ledgers += int(
                register.exponent != 0
            )
        maximum_payload = max(maximum_payload, payload)
        maximum_pi_payload = max(maximum_pi_payload, pi_payload)
        maximum_nonzero_messages = max(
            maximum_nonzero_messages,
            nonzero_messages,
        )
        maximum_nonzero_message_ledgers = max(
            maximum_nonzero_message_ledgers,
            nonzero_message_ledgers,
        )
        maximum_nonzero_coefficients = max(
            maximum_nonzero_coefficients,
            nonzero_coefficients,
        )
        maximum_nonzero_coefficient_ledgers = max(
            maximum_nonzero_coefficient_ledgers,
            nonzero_coefficient_ledgers,
        )
    return {
        "maximum_carrier_payload_bits": maximum_payload,
        "maximum_pi_integral_basis_carrier_payload_bits": (
            maximum_pi_payload
        ),
        "maximum_residual_signed_bits": maximum_width,
        "maximum_pi_integral_basis_residual_signed_bits": (
            maximum_pi_width
        ),
        "maximum_ledger_exponent_signed_bits": maximum_ledger_width,
        "maximum_nonzero_message_slots": maximum_nonzero_messages,
        "maximum_nonzero_message_ledgers": (
            maximum_nonzero_message_ledgers
        ),
        "maximum_nonzero_coefficient_registers": (
            maximum_nonzero_coefficients
        ),
        "maximum_nonzero_coefficient_ledgers": (
            maximum_nonzero_coefficient_ledgers
        ),
    }


def family_context(
    family: str,
    certificate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = reference.compile_descriptor(PRIME + 1, family.upper())
    operator = reference.compile_operator(descriptor)
    characteristic = [
        tuple(element)
        for element in certificate["characteristic"]
    ]
    seed = reference.seed_vector(descriptor)
    basis = build_scaled_basis(operator, seed)
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
        "pi_basis_roundtrip_exact": all(
            pi_to_zeta_basis(zeta_to_pi_basis(element)) == element
            for element in (
                [
                    entry
                    for row in operator
                    for entry in row
                ]
                + characteristic
            )
        ),
    }
    context = {
        "descriptor": descriptor,
        "operator": operator,
        "characteristic": characteristic,
        "seed": seed,
        "scaled_seed": normalize_vector(seed),
        "scaled_basis": basis,
        "raw_basis": [
            [
                materialize_element(
                    ScaledElement(element, vector.exponent)
                )
                for element in vector.residual
            ]
            for vector in basis
        ],
    }
    return checks, context


def case_check(
    case: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    periods = case["periods"]
    scaled_coefficients = sequential_scaled_coefficients(
        periods,
        context["characteristic"],
    )
    raw_coefficients = reference.sequential_coefficients(
        periods,
        context["characteristic"],
    )
    decompositions_equal = all(
        normalize_element(raw) == scaled
        for raw, scaled in zip(
            raw_coefficients,
            scaled_coefficients,
            strict=True,
        )
    )
    output = scaled_output(
        scaled_coefficients,
        context["scaled_basis"],
    )
    boundary = scaled_boundary(output)
    metrics = carrier_metrics(
        context["scaled_seed"],
        context["scaled_basis"],
        scaled_coefficients,
        output,
    )
    raw_output = reference.linear_combination(
        raw_coefficients,
        context["raw_basis"],
    )
    raw_metrics = reference.recurrence_carrier_metrics(
        context["seed"],
        context["raw_basis"],
        raw_coefficients,
        raw_output,
    )
    production_metrics = case["pi_content_stats"]
    metric_keys = tuple(metrics)
    return {
        "periods": periods,
        "family": case["family"],
        "sequential_scaled_coefficient_decompositions_equal": (
            decompositions_equal
        ),
        "boundary_sha256_equal": (
            hashlib.sha256(reference.encoded(boundary)).hexdigest()
            == case["boundary_sha256"]
        ),
        "boundary_payload_bits_equal": (
            element_payload_bits(boundary)
            == case["boundary_payload_bits"]
        ),
        "boundary_pi_valuation_equal": (
            height_reference.pi_valuation(boundary)
            == case["boundary_pi_valuation"]
        ),
        "all_carrier_metrics_equal": all(
            metrics[key] == production_metrics[key]
            for key in metric_keys
        ),
        "exact_carrier_metrics": metrics,
        "raw_baseline_payload_equal": (
            raw_metrics["maximum_carrier_payload_bits"]
            == case["raw_recurrence_baseline"][
                "maximum_carrier_payload_bits"
            ]
        ),
        "raw_baseline_width_equal": (
            raw_metrics["maximum_coefficient_signed_bits"]
            == case["raw_recurrence_baseline"][
                "maximum_coefficient_signed_bits"
            ]
        ),
        "zeta_basis_payload_worsens": (
            metrics["maximum_carrier_payload_bits"]
            > raw_metrics["maximum_carrier_payload_bits"]
        ),
        "pi_basis_payload_worsens": (
            metrics["maximum_pi_integral_basis_carrier_payload_bits"]
            > raw_metrics["maximum_carrier_payload_bits"]
        ),
    }


@dataclass
class OracleCarrier:
    message_residuals: list[RingVector]
    message_exponents: list[int]
    coefficient_residuals: list[RingElement]
    coefficient_exponents: list[int]
    generation: int = 0
    lease: int = 0

    @classmethod
    def create(cls) -> "OracleCarrier":
        return cls(
            message_residuals=[
                [reference.ring_zero() for _ in range(PRIME)]
                for _ in range(MESSAGE_SLOTS)
            ],
            message_exponents=[0 for _ in range(MESSAGE_SLOTS)],
            coefficient_residuals=[
                reference.ring_zero()
                for _ in range(DIMENSION)
            ],
            coefficient_exponents=[0 for _ in range(DIMENSION)],
        )

    def backing(self) -> tuple[int, ...]:
        return (
            id(self.message_residuals),
            *(id(row) for row in self.message_residuals),
            id(self.message_exponents),
            id(self.coefficient_residuals),
            id(self.coefficient_exponents),
        )

    def all_zero(self) -> bool:
        return (
            all(vector_is_zero(row) for row in self.message_residuals)
            and not any(self.message_exponents)
            and all(
                element == reference.ring_zero()
                for element in self.coefficient_residuals
            )
            and not any(self.coefficient_exponents)
        )


def execute_oracle_transaction(
    carrier: OracleCarrier,
    context: dict[str, Any],
    periods: int,
) -> tuple[RingElement, bool]:
    if not carrier.all_zero():
        fail("oracle carrier was not restored")
    carrier.lease += 1
    values = [
        context["scaled_seed"],
        *context["scaled_basis"],
    ]
    coefficients = sequential_scaled_coefficients(
        periods,
        context["characteristic"],
    )
    output = scaled_output(coefficients, context["scaled_basis"])
    values.append(output)
    for index, value in enumerate(values):
        carrier.message_residuals[index][:] = value.residual
        carrier.message_exponents[index] = value.exponent
    for index, value in enumerate(coefficients):
        carrier.coefficient_residuals[index] = value.residual
        carrier.coefficient_exponents[index] = value.exponent
    boundary = scaled_boundary(output)

    def subtract_message(index: int, value: ScaledVector) -> None:
        if (
            carrier.message_exponents[index] != value.exponent
            or carrier.message_residuals[index] != value.residual
        ):
            fail("oracle message rematerialization mismatch")
        carrier.message_residuals[index][:] = [
            reference.ring_subtract(actual, expected)
            for actual, expected in zip(
                carrier.message_residuals[index],
                value.residual,
                strict=True,
            )
        ]
        carrier.message_exponents[index] -= value.exponent

    subtract_message(OUTPUT_SLOT, output)
    for index, value in enumerate(coefficients):
        if (
            carrier.coefficient_exponents[index] != value.exponent
            or carrier.coefficient_residuals[index] != value.residual
        ):
            fail("oracle coefficient rematerialization mismatch")
        carrier.coefficient_residuals[index] = reference.ring_subtract(
            carrier.coefficient_residuals[index],
            value.residual,
        )
        carrier.coefficient_exponents[index] -= value.exponent
    for index in range(DIMENSION, 0, -1):
        subtract_message(index, context["scaled_basis"][index - 1])
    subtract_message(0, context["scaled_seed"])
    carrier.generation += 1
    return boundary, carrier.all_zero()


def restoration_check(
    contexts: dict[str, dict[str, Any]],
    production: dict[str, Any],
) -> dict[str, bool]:
    carrier = OracleCarrier.create()
    backing = carrier.backing()
    primary_boundary, primary_restored = execute_oracle_transaction(
        carrier,
        contexts["primary"],
        max(EXPECTED_PERIODS),
    )
    reuse_boundary, reuse_restored = execute_oracle_transaction(
        carrier,
        contexts["reuse"],
        max(EXPECTED_PERIODS),
    )
    fresh_boundary, fresh_restored = execute_oracle_transaction(
        OracleCarrier.create(),
        contexts["reuse"],
        max(EXPECTED_PERIODS),
    )
    expected = production["restoration_reuse_case"]
    return {
        "primary_restored_exactly": primary_restored,
        "reuse_restored_exactly": reuse_restored,
        "fresh_restored_exactly": fresh_restored,
        "same_original_backing": carrier.backing() == backing,
        "fresh_reuse_boundary_equal": reuse_boundary == fresh_boundary,
        "primary_boundary_nonzero": (
            primary_boundary != reference.ring_zero()
        ),
        "generation_equal": carrier.generation == expected["generation"],
        "lease_equal": carrier.lease == expected["lease"],
        "all_payload_and_ledgers_zero": carrier.all_zero(),
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


def mutation_checks(
    context: dict[str, Any],
) -> dict[str, bool]:
    coefficients = sequential_scaled_coefficients(
        64,
        context["characteristic"],
    )
    output = scaled_output(coefficients, context["scaled_basis"])
    boundary = scaled_boundary(output)
    mutated_coefficients = list(coefficients)
    mutated_index = next(
        index
        for index, coefficient in enumerate(mutated_coefficients)
        if coefficient.residual != reference.ring_zero()
    )
    original = mutated_coefficients[mutated_index]
    mutated_coefficients[mutated_index] = ScaledElement(
        original.residual,
        original.exponent + 1,
    )
    mutated_boundary = scaled_boundary(
        scaled_output(
            mutated_coefficients,
            context["scaled_basis"],
        )
    )
    return {
        "coefficient_pi_ledger_perturbation_changes_boundary": (
            mutated_boundary != boundary
        ),
        "pi_basis_nontrivial_roundtrip_exact": (
            pi_to_zeta_basis(
                zeta_to_pi_basis(context["characteristic"][3])
            )
            == context["characteristic"][3]
        ),
    }


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_chain_period17_"
            "pi_content_recurrence_oracle.py PRODUCTION_RESULT"
        )
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        production = json.load(handle)
    if production["tested_periods"] != list(EXPECTED_PERIODS):
        fail("oracle tested periods changed")

    family_checks = {}
    contexts = {}
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
    mutations = mutation_checks(contexts["primary"])
    all_family_checks = all(
        all(check.values())
        for check in family_checks.values()
    )
    all_case_checks = all(
        all(
            value
            for key, value in check.items()
            if key not in {
                "periods",
                "family",
                "exact_carrier_metrics",
            }
        )
        for check in case_checks
    )
    all_restoration_checks = all(restoration.values())
    all_mutations_detected = all(mutations.values())
    production_scope_checks = {
        "production_result_pass": production["result"] == "PASS",
        "raw_coefficients_absent_from_accepted_path": (
            not production[
                "accepted_path_materializes_raw_recurrence_coefficients"
            ]
        ),
        "pi_factored_during_arithmetic": (
            production[
                "accepted_path_factors_pi_content_"
                "during_polynomial_arithmetic"
            ]
        ),
        "all_cases_worsen_zeta_payload": (
            production["all_cases_worsen_carrier_payload"]
        ),
        "all_cases_worsen_pi_basis_payload": (
            production[
                "all_cases_worsen_carrier_payload_in_pi_integral_basis"
            ]
        ),
        "identical_classical_normalizer_retained": (
            production["matched_classical"][
                "identical_pi_content_normalized_recurrence_available"
            ]
            and not production["matched_classical"][
                "comparison_establishes_advantage"
            ]
        ),
        "all_basis_impossibility_not_claimed": (
            "INTRINSIC_LOWER_BOUND_ACROSS_ALL_INTEGRAL_BASES"
            in production["not_established"]
        ),
        "distinct_phase_resource_not_claimed": (
            "DISTINCT_PHASE_RESOURCE"
            in production["not_established"]
        ),
    }
    all_scope_checks = all(production_scope_checks.values())
    result = {
        "result": (
            "PASS"
            if (
                all_family_checks
                and all_case_checks
                and all_restoration_checks
                and all_mutations_detected
                and all_scope_checks
            )
            else "FAIL"
        ),
        "experiment": (
            "SEPARATE_EXACT_PI_CONTENT_LEDGER_NORMALIZED_"
            "NATIVE_K_RECURRENCE_ORACLE"
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
        "production_scope_checks": production_scope_checks,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": (
            "LINUX_X86_64_PYTHON_EXACT_TWO_PUBLIC_F17_PERIOD17_"
            "CUBIC_PATH_FAMILIES_Q_ZETA17_PI_CONTENT_LEDGER_"
            "NORMALIZED_RECURRENCE_PERIODS1_4_16_64_256_ZETA_"
            "AND_PI_INTEGRAL_BASIS_PAYLOAD_DIAGNOSTIC_EXACT_"
            "SUBTRACTIVE_PAYLOAD_AND_LEDGER_RESTORATION_"
            "SOFTWARE_ONLY"
        ),
        "not_established": production["not_established"],
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
