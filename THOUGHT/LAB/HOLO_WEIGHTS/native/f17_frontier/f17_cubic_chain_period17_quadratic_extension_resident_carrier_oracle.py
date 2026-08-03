#!/usr/bin/env python3
"""Separate power-basis oracle for the quadratic-extension resident carrier.

This oracle does not import the production M116 module.  It reconstructs the
public F17 Horner programs through the existing separate-reference chain and
uses power-basis quadratic-extension semantics.  For exact declared resource
parity, it independently converts those pairs to the integral real basis and
reexecutes the public streaming carrier and projection schedule.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_direct_real_hermitian_oracle as direct


streamed = direct.prior
real = streamed.prior
horner = real.horner
phase = real.phase
reference = horner.reference
ring = horner.ring
prior = horner.prior
pi_reference = horner.pi_reference

PRIME = horner.PRIME
UNIT_RANK = horner.UNIT_RANK
EXPECTED_PERIODS = horner.EXPECTED_PERIODS

RingElement = horner.RingElement
RingVector = horner.RingVector
PowerElement = real.PowerElement
SElement = PowerElement
SPair = tuple[SElement, SElement]
SVector = list[SPair]

S_ZERO: SElement = (0, 0, 0, 0, 0, 0, 0, 0)
S_ONE: SElement = (1, 0, 0, 0, 0, 0, 0, 0)
S_S1: SElement = (0, 1, 0, 0, 0, 0, 0, 0)


def fail(message: str) -> None:
    raise RuntimeError(message)


def s_add(left: SElement, right: SElement) -> SElement:
    return tuple(
        left_value + right_value
        for left_value, right_value in zip(left, right, strict=True)
    )  # type: ignore[return-value]


def s_subtract(left: SElement, right: SElement) -> SElement:
    return tuple(
        left_value - right_value
        for left_value, right_value in zip(left, right, strict=True)
    )  # type: ignore[return-value]


def s_negate(value: SElement) -> SElement:
    return tuple(-coefficient for coefficient in value)  # type: ignore[return-value]


def s_scale(value: SElement, scalar: int) -> SElement:
    return tuple(
        scalar * coefficient for coefficient in value
    )  # type: ignore[return-value]


def s_s1_multiply(value: SElement) -> SElement:
    return (
        2 * value[1] - value[7],
        value[0] + value[2] - value[7],
        value[1] + value[3] - value[7],
        value[2] + value[4] - value[7],
        value[3] + value[5] - value[7],
        value[4] + value[6] - value[7],
        value[5],
        value[6] - value[7],
    )


def s_multiply(
    left: SElement,
    right: SElement,
    stats: "OracleStats | None" = None,
) -> SElement:
    """Independent multiply through the power basis, reported in S basis."""

    result = real.power_to_s(
        real.power_multiply_untracked(
            real.s_to_power(left),
            real.s_to_power(right),
        )
    )
    if stats is not None:
        stats.real_subfield_ring_multiplications += 1
        stats.real_subfield_coefficient_multiplications += 64
    return result


def pair_zero() -> SPair:
    return S_ZERO, S_ZERO


def pair_one() -> SPair:
    return S_ONE, S_ZERO


def pair_add(left: SPair, right: SPair) -> SPair:
    return s_add(left[0], right[0]), s_add(left[1], right[1])


def pair_subtract(left: SPair, right: SPair) -> SPair:
    return s_subtract(left[0], right[0]), s_subtract(left[1], right[1])


def pair_payload_bits(value: SPair) -> int:
    return real.s_payload_bits(value[0]) + real.s_payload_bits(value[1])


def vector_payload_bits(value: SVector) -> int:
    return sum(pair_payload_bits(element) for element in value)


POWER_PAIR_TABLE = direct.QUADRATIC_EXTENSION_TABLE
S_PAIR_TABLE = tuple(
    (real.power_to_s(a_value), real.power_to_s(b_value))
    for a_value, b_value in POWER_PAIR_TABLE
)


def full_to_power_pair(value: RingElement) -> tuple[PowerElement, PowerElement]:
    a_value = direct.ZERO
    b_value = direct.ZERO
    for coefficient, (a_basis, b_basis) in zip(
        value,
        POWER_PAIR_TABLE,
        strict=True,
    ):
        a_value = direct.power_add(a_value, direct.power_scale(a_basis, coefficient))
        b_value = direct.power_add(b_value, direct.power_scale(b_basis, coefficient))
    return a_value, b_value


def full_to_s_pair(
    value: RingElement,
    stats: "OracleStats | None" = None,
) -> SPair:
    a_value = S_ZERO
    b_value = S_ZERO
    for coefficient, (a_basis, b_basis) in zip(
        value,
        S_PAIR_TABLE,
        strict=True,
    ):
        scaled_a = s_scale(a_basis, coefficient)
        scaled_b = s_scale(b_basis, coefficient)
        next_a = s_add(a_value, scaled_a)
        next_b = s_add(b_value, scaled_b)
        if stats is not None:
            stats.full_to_split_coefficient_multiplications += 16
            stats.full_to_split_coefficient_additions += 16
            stats.maximum_full_to_split_element_work_payload_bits = max(
                stats.maximum_full_to_split_element_work_payload_bits,
                real.s_payload_bits(a_value)
                + real.s_payload_bits(b_value)
                + real.s_payload_bits(scaled_a)
                + real.s_payload_bits(scaled_b)
                + real.s_payload_bits(next_a)
                + real.s_payload_bits(next_b),
            )
        a_value, b_value = next_a, next_b
    result = a_value, b_value
    if stats is not None:
        stats.full_to_split_element_conversions += 1
        stats.maximum_full_to_split_input_payload_bits = max(
            stats.maximum_full_to_split_input_payload_bits,
            prior.element_payload_bits(value),
        )
        stats.maximum_full_to_split_output_payload_bits = max(
            stats.maximum_full_to_split_output_payload_bits,
            pair_payload_bits(result),
        )
    return result


def s_pair_to_power_pair(value: SPair) -> tuple[PowerElement, PowerElement]:
    return real.s_to_power(value[0]), real.s_to_power(value[1])


def zeta_multiply_full(value: RingElement) -> RingElement:
    high = value[-1]
    return tuple(
        -high if index == 0 else value[index - 1] - high
        for index in range(16)
    )


def s_pair_to_full(value: SPair) -> RingElement:
    a_power, b_power = s_pair_to_power_pair(value)
    full_a = real.power_to_full(a_power)
    full_b = zeta_multiply_full(real.power_to_full(b_power))
    return ring.ring_add(full_a, full_b)


def pair_multiply(
    left: SPair,
    right: SPair,
    stats: "OracleStats | None" = None,
) -> SPair:
    ac_value = s_multiply(left[0], right[0], stats)
    bd_value = s_multiply(left[1], right[1], stats)
    cross_value = s_multiply(
        s_add(left[0], left[1]),
        s_add(right[0], right[1]),
        stats,
    )
    constant = s_subtract(ac_value, bd_value)
    zeta_coefficient = s_add(
        s_subtract(s_subtract(cross_value, ac_value), bd_value),
        s_s1_multiply(bd_value),
    )
    if stats is not None:
        stats.split_pair_multiplications += 1
        stats.split_real_subfield_multiplications += 3
        stats.maximum_split_multiply_live_payload_bits = max(
            stats.maximum_split_multiply_live_payload_bits,
            pair_payload_bits(left)
            + pair_payload_bits(right)
            + real.s_payload_bits(ac_value)
            + real.s_payload_bits(bd_value)
            + real.s_payload_bits(cross_value)
            + real.s_payload_bits(constant)
            + real.s_payload_bits(zeta_coefficient),
        )
    return constant, zeta_coefficient


def pair_power(value: SPair, exponent: int, stats: "OracleStats") -> SPair:
    if exponent < 0:
        fail("oracle pair power received a negative exponent")
    result = pair_one()
    factor = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = pair_multiply(result, factor, stats)
        remaining >>= 1
        if remaining:
            factor = pair_multiply(factor, factor, stats)
    return result


FULL_UNIT_GENERATORS = real.exact.ring.UNIT_GENERATORS
FULL_UNIT_GENERATOR_INVERSES = real.exact.ring.UNIT_GENERATOR_INVERSES
S_UNIT_GENERATORS = tuple(full_to_s_pair(value) for value in FULL_UNIT_GENERATORS)
S_UNIT_GENERATOR_INVERSES = tuple(
    full_to_s_pair(value) for value in FULL_UNIT_GENERATOR_INVERSES
)
PUBLIC_CONVERSION_TABLE_BITS = sum(pair_payload_bits(value) for value in S_PAIR_TABLE)
PUBLIC_UNIT_TABLE_BITS = sum(
    pair_payload_bits(value)
    for value in (*S_UNIT_GENERATORS, *S_UNIT_GENERATOR_INVERSES)
)


@dataclass
class OracleStats(direct.DirectPowerOracleStats):
    full_to_split_element_conversions: int = 0
    full_to_split_coefficient_multiplications: int = 0
    full_to_split_coefficient_additions: int = 0
    split_to_full_boundary_lifts: int = 0
    split_pair_multiplications: int = 0
    split_real_subfield_multiplications: int = 0
    split_projection_additions: int = 0
    maximum_full_to_split_input_payload_bits: int = 0
    maximum_full_to_split_output_payload_bits: int = 0
    maximum_full_to_split_element_work_payload_bits: int = 0
    maximum_full_vector_to_split_vector_live_payload_bits: int = 0
    maximum_split_carrier_resident_payload_bits: int = 0
    maximum_predecessor_full_resident_payload_bits: int = 0
    maximum_split_projection_accumulator_payload_bits: int = 0
    maximum_split_projection_scale_payload_bits: int = 0
    maximum_split_projection_product_payload_bits: int = 0
    maximum_split_multiply_live_payload_bits: int = 0
    maximum_boundary_full_lift_payload_bits: int = 0
    maximum_split_projection_normalized_payload_bits: int = 0
    maximum_split_projection_live_payload_bits: int = 0


SPLIT_FIELDS = (
    "full_to_split_element_conversions",
    "full_to_split_coefficient_multiplications",
    "full_to_split_coefficient_additions",
    "split_to_full_boundary_lifts",
    "split_pair_multiplications",
    "split_real_subfield_multiplications",
    "split_projection_additions",
    "maximum_full_to_split_input_payload_bits",
    "maximum_full_to_split_output_payload_bits",
    "maximum_full_to_split_element_work_payload_bits",
    "maximum_full_vector_to_split_vector_live_payload_bits",
    "maximum_split_carrier_resident_payload_bits",
    "maximum_predecessor_full_resident_payload_bits",
    "maximum_split_projection_accumulator_payload_bits",
    "maximum_split_projection_scale_payload_bits",
    "maximum_split_projection_product_payload_bits",
    "maximum_split_multiply_live_payload_bits",
    "maximum_boundary_full_lift_payload_bits",
    "maximum_split_projection_normalized_payload_bits",
    "maximum_split_projection_live_payload_bits",
)


def metrics_json(stats: OracleStats) -> dict[str, int]:
    result = direct.metrics_json(stats)
    for name in SPLIT_FIELDS:
        result[name] = getattr(stats, name)
    return result


def convert_vector(
    value: RingVector,
    stats: OracleStats,
    *,
    pi_exponent: int = 0,
    ledger: tuple[int, ...] | list[int] = (),
    additional_live_payload_bits: int = 0,
) -> SVector:
    full_bits = prior.vector_payload_bits(value)
    metadata_bits = prior.signed_bits(pi_exponent) + prior.ledger_payload_bits(
        tuple(ledger)
    )
    full_resident_bits = full_bits + metadata_bits
    result: SVector = []
    for element in value:
        partial_before = vector_payload_bits(result)
        converted = full_to_s_pair(element, stats)
        partial_after = partial_before + pair_payload_bits(converted)
        stats.maximum_full_vector_to_split_vector_live_payload_bits = max(
            stats.maximum_full_vector_to_split_vector_live_payload_bits,
            additional_live_payload_bits
            + full_resident_bits
            + partial_before
            + stats.maximum_full_to_split_element_work_payload_bits,
            additional_live_payload_bits + full_resident_bits + partial_after,
        )
        result.append(converted)
    split_bits = vector_payload_bits(result)
    stats.maximum_predecessor_full_resident_payload_bits = max(
        stats.maximum_predecessor_full_resident_payload_bits,
        full_resident_bits,
    )
    stats.maximum_full_vector_to_split_vector_live_payload_bits = max(
        stats.maximum_full_vector_to_split_vector_live_payload_bits,
        additional_live_payload_bits + full_resident_bits + split_bits,
    )
    return result


def carrier_payload_bits(
    value: SVector,
    pi_exponent: int,
    ledger: tuple[int, ...] | list[int],
) -> int:
    return (
        vector_payload_bits(value)
        + prior.signed_bits(pi_exponent)
        + prior.ledger_payload_bits(tuple(ledger))
    )


def ledger_scale(ledger: tuple[int, ...], stats: OracleStats) -> SPair:
    result = pair_one()
    for exponent, generator, inverse in zip(
        ledger,
        S_UNIT_GENERATORS,
        S_UNIT_GENERATOR_INVERSES,
        strict=True,
    ):
        factor = (
            pair_power(generator, exponent, stats)
            if exponent >= 0
            else pair_power(inverse, -exponent, stats)
        )
        result = pair_multiply(result, factor, stats)
    return result


def project(
    value: SVector,
    pi_exponent: int,
    ledger: tuple[int, ...],
    stats: OracleStats,
) -> RingElement:
    projected = pair_zero()
    for element in value:
        projected = pair_add(projected, element)
        stats.split_projection_additions += 1
        stats.maximum_split_projection_accumulator_payload_bits = max(
            stats.maximum_split_projection_accumulator_payload_bits,
            pair_payload_bits(projected),
        )
    scale = ledger_scale(ledger, stats)
    product = pair_multiply(scale, projected, stats)
    lifted = s_pair_to_full(product)
    stats.split_to_full_boundary_lifts += 1
    scaled = pi_reference.normalize_element(lifted, pi_exponent)
    boundary = pi_reference.materialize_element(scaled)
    scale_bits = pair_payload_bits(scale)
    projected_bits = pair_payload_bits(projected)
    product_bits = pair_payload_bits(product)
    lifted_bits = prior.element_payload_bits(lifted)
    scaled_bits = (
        prior.element_payload_bits(scaled.residual)
        + prior.signed_bits(scaled.exponent)
    )
    boundary_bits = prior.element_payload_bits(boundary)
    stats.maximum_split_projection_scale_payload_bits = max(
        stats.maximum_split_projection_scale_payload_bits,
        scale_bits,
    )
    stats.maximum_split_projection_product_payload_bits = max(
        stats.maximum_split_projection_product_payload_bits,
        product_bits,
    )
    stats.maximum_boundary_full_lift_payload_bits = max(
        stats.maximum_boundary_full_lift_payload_bits,
        lifted_bits,
        boundary_bits,
    )
    stats.maximum_split_projection_normalized_payload_bits = max(
        stats.maximum_split_projection_normalized_payload_bits,
        scaled_bits,
    )
    stats.maximum_split_projection_live_payload_bits = max(
        stats.maximum_split_projection_live_payload_bits,
        projected_bits
        + scale_bits
        + product_bits
        + lifted_bits
        + scaled_bits
        + boundary_bits,
        projected_bits + stats.maximum_split_multiply_live_payload_bits,
    )
    return boundary


def install_independent_stack() -> None:
    streamed.streamed_power_vector_norm = direct.direct_power_vector_norm
    real.balance = streamed.streamed_balance
    horner.HornerOracleStats = OracleStats
    prior.OracleStats = OracleStats
    prior.balance = streamed.streamed_balance
    prior.add_vectors = phase.add_vectors
    prior.project = phase.project
    prior.record_metrics = phase.record_metrics
    prior.ledger_scale = real.exact.tracked_ledger_scale


def named_search_temporary_maxima_sum(metrics: dict[str, int]) -> int:
    return (
        horner.named_search_temporary_maxima_sum(metrics)
        + metrics["maximum_streamed_norm_named_live_payload_bits"]
        + metrics["maximum_real_current_norm_payload_bits"]
        + metrics["maximum_real_current_energy_bits"]
    )


def case_check(case: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    stats = OracleStats()
    output = horner.build_horner_output(context, case["periods"], stats)
    empty_carrier_bits = carrier_payload_bits(
        [pair_zero() for _ in range(PRIME)],
        0,
        tuple(0 for _ in range(UNIT_RANK)),
    )
    resident = convert_vector(
        output.residual,
        stats,
        pi_exponent=output.pi_exponent,
        ledger=output.ledger,
        additional_live_payload_bits=empty_carrier_bits,
    )
    resident_bits = carrier_payload_bits(resident, output.pi_exponent, output.ledger)
    stats.maximum_split_carrier_resident_payload_bits = resident_bits
    stats.maximum_carrier_resident_payload_bits = resident_bits
    stats.maximum_resident_payload_bits = resident_bits
    boundary = project(resident, output.pi_exponent, output.ledger, stats)
    stats.maximum_projection_resident_plus_work_payload_bits = (
        resident_bits + stats.maximum_split_projection_live_payload_bits
    )
    metrics = metrics_json(stats)

    inverse_stats = OracleStats()
    inverse_output = horner.build_horner_output(
        context,
        case["periods"],
        inverse_stats,
    )
    inverse_resident = convert_vector(
        inverse_output.residual,
        inverse_stats,
        pi_exponent=inverse_output.pi_exponent,
        ledger=inverse_output.ledger,
    )
    inverse_stats.maximum_inverse_resident_plus_work_payload_bits = (
        resident_bits
        + max(
            inverse_stats.maximum_horner_named_checkpoint_payload_bits,
            inverse_stats.maximum_full_vector_to_split_vector_live_payload_bits,
        )
    )
    restored = all(
        pair_subtract(actual, expected) == pair_zero()
        for actual, expected in zip(resident, inverse_resident, strict=True)
    )
    restored_payload = carrier_payload_bits(
        [pair_zero() for _ in range(PRIME)],
        0,
        tuple(0 for _ in range(UNIT_RANK)),
    )
    inverse_stats.maximum_split_carrier_resident_payload_bits = restored_payload
    inverse_stats.maximum_carrier_resident_payload_bits = restored_payload
    inverse_stats.maximum_resident_payload_bits = restored_payload
    inverse_metrics = metrics_json(inverse_stats)
    raw_boundary, raw_stats = horner.raw_horner_boundary(context, case["periods"])
    raw_values = horner.raw_metrics(raw_stats)
    checkpoint = max(
        metrics["maximum_horner_named_checkpoint_payload_bits"],
        metrics["maximum_full_vector_to_split_vector_live_payload_bits"],
        metrics["maximum_projection_resident_plus_work_payload_bits"],
        inverse_metrics["maximum_inverse_resident_plus_work_payload_bits"],
    )
    search_temporary = named_search_temporary_maxima_sum(metrics)
    retained_tables = (
        real.ACCEPTED_TABLE_PAYLOAD_BITS
        + PUBLIC_CONVERSION_TABLE_BITS
        + PUBLIC_UNIT_TABLE_BITS
    )
    named_total = checkpoint + retained_tables + search_temporary
    return {
        "periods": case["periods"],
        "family": case["family"],
        "boundary_sha256_equal": (
            hashlib.sha256(reference.encoded(boundary)).hexdigest()
            == case["boundary_sha256"]
        ),
        "raw_boundary_sha256_equal": (
            hashlib.sha256(reference.encoded(raw_boundary)).hexdigest()
            == case["raw_horner_boundary_sha256"]
        ),
        "phase_resource_tuple_equal": all(
            case["phase_stats"][key] == value
            for key, value in metrics.items()
        ),
        "inverse_resource_tuple_equal": (
            all(
                case["inverse_rematerialization_stats"][key] == value
                for key, value in inverse_metrics.items()
            )
        ),
        "raw_resource_tuple_equal": raw_values == case["raw_horner_stats"],
        "checkpoint_equal": checkpoint == case["phase_named_checkpoint_payload_bits"],
        "search_temporary_equal": (
            search_temporary == case["named_search_temporary_maxima_sum_bits"]
        ),
        "retained_tables_equal": retained_tables == case["retained_public_table_payload_bits"],
        "named_total_equal": named_total == case["phase_named_component_maxima_sum_bits"],
        "comparable_resident_delta_equal": (
            metrics["maximum_split_carrier_resident_payload_bits"]
            - metrics["maximum_predecessor_full_resident_payload_bits"]
            == case[
                "split_resident_minus_comparable_full_resident_payload_bits"
            ]
        ),
        "inverse_output_equal": inverse_output == output,
        "pair_restoration_exact": restored,
        "semantic_power_pair_matches_s_pair": all(
            tuple(real.s_to_power(component) for component in s_pair)
            == direct_pair
            for s_pair, direct_pair in zip(
                resident,
                (full_to_power_pair(element) for element in output.residual),
                strict=True,
            )
        ),
        "exact_phase_resource_tuple": metrics,
        "exact_inverse_resource_tuple": inverse_metrics,
    }


def restoration_check(
    contexts: dict[str, dict[str, Any]],
    production: dict[str, Any],
) -> dict[str, bool]:
    def transaction(context: dict[str, Any]) -> tuple[RingElement, bool]:
        stats = OracleStats()
        output = horner.build_horner_output(context, 1, stats)
        resident = convert_vector(output.residual, stats)
        boundary = project(resident, output.pi_exponent, output.ledger, stats)
        expected = horner.build_horner_output(context, 1, OracleStats())
        expected_resident = convert_vector(expected.residual, OracleStats())
        restored = all(
            pair_subtract(actual, value) == pair_zero()
            for actual, value in zip(resident, expected_resident, strict=True)
        )
        return boundary, restored

    _, primary_restored = transaction(contexts["primary"])
    reuse_boundary, reuse_restored = transaction(contexts["reuse"])
    fresh_boundary, fresh_restored = transaction(contexts["reuse"])
    expected = production["restoration_reuse_case"]
    return {
        "primary_restored_exactly": primary_restored,
        "reuse_restored_exactly": reuse_restored,
        "fresh_restored_exactly": fresh_restored,
        "fresh_reuse_boundary_equal": reuse_boundary == fresh_boundary,
        "production_same_backing": expected["same_original_backing"],
        "production_generation_correct": expected["generation"] == 2,
        "production_lease_correct": expected["lease"] == 2,
        "production_no_baseline_reload": not expected["baseline_reload"],
    }


def algebra_checks() -> dict[str, bool]:
    basis = []
    for index in range(16):
        value = [0 for _ in range(16)]
        value[index] = 1
        basis.append(tuple(value))
    roundtrip = all(s_pair_to_full(full_to_s_pair(value)) == value for value in basis)
    products = all(
        s_pair_to_full(pair_multiply(full_to_s_pair(left), full_to_s_pair(right)))
        == ring.ring_multiply(left, right)
        for left in basis
        for right in basis
    )
    determinant_law = all(
        tuple(real.s_to_power(component) for component in full_to_s_pair(value))
        == direct_pair
        for value, direct_pair in zip(
            basis,
            (full_to_power_pair(value) for value in basis),
            strict=True,
        )
    )
    return {
        "all_16_basis_roundtrips_exact": roundtrip,
        "all_256_basis_products_exact": products,
        "s_and_power_pair_derivations_match_on_basis": determinant_law,
        "coordinate_count_remains_16": len(pair_one()[0]) + len(pair_one()[1]) == 16,
    }


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_chain_period17_quadratic_extension_"
            "resident_carrier_oracle.py PRODUCTION_RESULT"
        )
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        production = json.load(handle)
    if tuple(production["tested_periods"]) != EXPECTED_PERIODS:
        fail("oracle tested periods changed")
    install_independent_stack()
    contexts: dict[str, dict[str, Any]] = {}
    family_checks: dict[str, dict[str, bool]] = {}
    for family in ("primary", "reuse"):
        checks, context = prior.family_context(
            family,
            production["block_certificates"][family],
        )
        family_checks[family] = checks
        contexts[family] = context
    cases = [
        case_check(case, contexts[case["family"].lower()])
        for case in production["cases"]
    ]
    restoration = restoration_check(contexts, production)
    algebra = algebra_checks()
    scope = {
        "production_pass": production["result"] == "PASS",
        "representation_isomorphism_only": (
            production["representation_isomorphism"]
            and not production["dimension_reducing_quotient"]
        ),
        "full_forward_inverse_work_declared": (
            production["forward_horner_construction_remains_full_cyclotomic"]
            and production["inverse_rematerialization_remains_full_cyclotomic"]
        ),
        "one_boundary_lift": production["all_cases_use_one_boundary_full_lift"],
        "identical_classical_pair_path_retained": (
            production["matched_classical"][
                "identical_two_by_eight_quadratic_extension_carrier_available"
            ]
            and not production["matched_classical"]["comparison_establishes_advantage"]
        ),
        "all_named_totals_above_raw": all(
            case["phase_minus_raw_horner_named_payload_bits"] > 0
            for case in production["cases"]
        ),
        "no_distinct_resource_claim": (
            "DISTINCT_PHASE_RESOURCE" in production["not_established"]
        ),
    }
    result_pass = (
        all(all(values.values()) for values in family_checks.values())
        and all(
            all(
                value
                for key, value in case.items()
                if key not in {
                    "periods",
                    "family",
                    "exact_phase_resource_tuple",
                    "exact_inverse_resource_tuple",
                }
            )
            for case in cases
        )
        and all(restoration.values())
        and all(algebra.values())
        and all(scope.values())
    )
    result = {
        "result": "PASS" if result_pass else "FAIL",
        "experiment": (
            "SEPARATE_POWER_BASIS_QUADRATIC_EXTENSION_RESIDENT_"
            "CARRIER_AND_INTEGRAL_SCHEDULE_ORACLE"
        ),
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "oracle_imports_production_m116_module": False,
        "semantic_oracle_basis": "POWER_BASIS_QUADRATIC_EXTENSION",
        "resource_oracle_basis": "INTEGRAL_S_BASIS_REEXECUTION",
        "family_checks": family_checks,
        "case_checks": cases,
        "restoration_checks": restoration,
        "algebra_checks": algebra,
        "scope_checks": scope,
        "public_table_checks": {
            "conversion_table_bits_equal": (
                PUBLIC_CONVERSION_TABLE_BITS
                == production["public_table_accounting"][
                    "quadratic_conversion_table_payload_bits"
                ]
            ),
            "unit_table_bits_equal": (
                PUBLIC_UNIT_TABLE_BITS
                == production["public_table_accounting"][
                    "split_unit_table_payload_bits"
                ]
            ),
            "real_search_table_bits_equal": (
                real.ACCEPTED_TABLE_PAYLOAD_BITS
                == production["public_table_accounting"][
                    "real_search_table_payload_bits"
                ]
            ),
        },
        "claim_ceiling": (
            "LINUX_X86_64_PYTHON_TWO_PUBLIC_F17_PERIOD17_FAMILIES_"
            "PERIODS1AND64_EXACT_POST_FORWARD_TWO_BY_EIGHT_REAL_"
            "SUBFIELD_QUADRATIC_EXTENSION_STORED_CARRIER_PAIR_NATIVE_"
            "PROJECTION_AND_LEDGER_MATERIALIZATION_ONE_SPLIT_TO_FULL_"
            "FINAL_SCALAR_BOUNDARY_LIFT_FULL_CYCLOTOMIC_FORWARD_AND_"
            "INVERSE_REMATERIALIZATION_EXACT_ALGEBRAIC_ORIGINAL_"
            "BACKING_RESTORATION_AND_PERIOD1_CROSS_FAMILY_REUSE_"
            "SEPARATE_REFERENCE_PARITY_SOFTWARE_ONLY"
        ),
        "preserved_subclaims": [
            "EXACT_INTEGRAL_TWO_BY_EIGHT_RESIDENT_PAIR_ISOMORPHISM",
            "PAIR_NATIVE_RESIDENT_PROJECTION_AND_UNIT_LEDGER_ACTION",
            "ONE_FINAL_SCALAR_FULL_LIFT_PER_FORWARD_TRANSACTION",
            "RAW_HORNER_BOUNDARY_PARITY",
            "EXACT_ORIGINAL_BACKING_RESTORATION_AND_PERIOD1_REUSE",
        ],
        "rejected_interpretations": production["not_established"],
        "terminal": False,
    }
    if not all(result["public_table_checks"].values()):
        result["result"] = "FAIL"
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    if result["result"] != "PASS":
        fail("independent M116 oracle gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
