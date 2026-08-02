#!/usr/bin/env python3
"""Separate quadratic-extension oracle for direct Hermitian terms.

This oracle does not import the production direct-Hermitian successor.  It
reconstructs each semantic norm through the quadratic presentation

    Q(zeta_17) = Q(y)[zeta] / (zeta^2-y*zeta+1),
    y = zeta + zeta^-1.

Writing x=A(y)+zeta*B(y), it obtains

    x*conjugate(x) = A^2 + y*A*B + B^2

in the degree-eight power basis.  A separate cyclic schedule reexecution is
used only to check and reproduce the production resource tuple.  Neither
path constructs a degree-sixteen Hermitian product.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_streamed_real_autocorrelation_oracle as prior


power = prior.prior
full = prior.full

RingElement = power.RingElement
RingVector = prior.RingVector
PowerElement = prior.PowerElement

CYCLIC_WIDTH = power.reference.PRIME
REAL_WIDTH = 8
DIRECT_PRODUCTS_PER_TERM = 136
DIRECT_ADDITIONS_PER_TERM = 127
DIRECT_SUBTRACTIONS_PER_TERM = 8

ORIGINAL_METRICS_JSON = prior.metrics_json
ORACLE_QUADRATIC_COUNTS = {
    "semantic_calls": 0,
    "semantic_power_multiplications": 0,
    "schedule_parity_checks": 0,
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def power_add(
    left: PowerElement,
    right: PowerElement,
) -> PowerElement:
    return tuple(
        left_value + right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


def power_negate(value: PowerElement) -> PowerElement:
    return tuple(-coefficient for coefficient in value)


def power_scale(value: PowerElement, scalar: int) -> PowerElement:
    return tuple(scalar * coefficient for coefficient in value)


Y: PowerElement = (0, 1, 0, 0, 0, 0, 0, 0)
ZERO: PowerElement = (0, 0, 0, 0, 0, 0, 0, 0)
ONE: PowerElement = (1, 0, 0, 0, 0, 0, 0, 0)


def build_quadratic_extension_table() -> tuple[
    tuple[PowerElement, PowerElement], ...
]:
    """Return public A_n,B_n with zeta^n=A_n+zeta*B_n."""

    entries: list[tuple[PowerElement, PowerElement]] = []
    a_value = ONE
    b_value = ZERO
    for _ in range(CYCLIC_WIDTH - 1):
        entries.append((a_value, b_value))
        a_value, b_value = (
            power_negate(b_value),
            power_add(
                a_value,
                power.power_multiply_untracked(Y, b_value),
            ),
        )
    return tuple(entries)


QUADRATIC_EXTENSION_TABLE = build_quadratic_extension_table()


def quadratic_extension_norm(element: RingElement) -> PowerElement:
    """Compute A^2+yAB+B^2 in the independent power basis."""

    if len(element) != CYCLIC_WIDTH - 1:
        fail("oracle canonical cyclotomic width changed")
    a_value = ZERO
    b_value = ZERO
    for coefficient, (a_basis, b_basis) in zip(
        element,
        QUADRATIC_EXTENSION_TABLE,
        strict=True,
    ):
        a_value = power_add(a_value, power_scale(a_basis, coefficient))
        b_value = power_add(b_value, power_scale(b_basis, coefficient))

    a_squared = power.power_multiply_untracked(a_value, a_value)
    ab_value = power.power_multiply_untracked(a_value, b_value)
    b_squared = power.power_multiply_untracked(b_value, b_value)
    yab_value = power.power_multiply_untracked(Y, ab_value)
    ORACLE_QUADRATIC_COUNTS["semantic_calls"] += 1
    ORACLE_QUADRATIC_COUNTS["semantic_power_multiplications"] += 4
    return power_add(power_add(a_squared, yab_value), b_squared)


@dataclass
class DirectPowerOracleStats(prior.StreamedPowerOracleStats):
    direct_real_hermitian_calls: int = 0
    direct_real_hermitian_full_cyclotomic_product_calls: int = 0
    direct_real_hermitian_materialized_conjugate_calls: int = 0
    direct_real_hermitian_accumulation_additions: int = 0
    direct_real_hermitian_coefficient_subtractions: int = 0
    direct_real_hermitian_coefficient_multiplications: int = 0
    maximum_direct_real_product_payload_bits: int = 0
    maximum_direct_real_coordinate_accumulator_payload_bits: int = 0
    maximum_direct_real_retained_p8_payload_bits: int = 0
    maximum_direct_real_partial_output_payload_bits: int = 0
    maximum_direct_real_term_payload_bits: int = 0
    maximum_direct_real_generator_work_payload_bits: int = 0
    maximum_direct_real_generation_live_payload_bits: int = 0
    maximum_direct_real_degree16_scratch_payload_bits: int = 0


DIRECT_FIELDS = (
    "direct_real_hermitian_calls",
    "direct_real_hermitian_full_cyclotomic_product_calls",
    "direct_real_hermitian_materialized_conjugate_calls",
    "direct_real_hermitian_accumulation_additions",
    "direct_real_hermitian_coefficient_subtractions",
    "direct_real_hermitian_coefficient_multiplications",
    "maximum_direct_real_product_payload_bits",
    "maximum_direct_real_coordinate_accumulator_payload_bits",
    "maximum_direct_real_retained_p8_payload_bits",
    "maximum_direct_real_partial_output_payload_bits",
    "maximum_direct_real_term_payload_bits",
    "maximum_direct_real_generator_work_payload_bits",
    "maximum_direct_real_generation_live_payload_bits",
    "maximum_direct_real_degree16_scratch_payload_bits",
)


def metrics_json(stats: DirectPowerOracleStats) -> dict[str, int]:
    result = ORIGINAL_METRICS_JSON(stats)
    for name in DIRECT_FIELDS:
        result[name] = getattr(stats, name)
    return result


def account_direct_schedule(
    element: RingElement,
    stats: DirectPowerOracleStats,
    resident_accumulator_bits: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    """Reexecute the public P8,P0..P7 schedule for resource parity."""

    output: list[int] = []
    retained_p8: int | None = None
    stats.direct_real_hermitian_calls += 1
    for shift in (REAL_WIDTH, *range(REAL_WIDTH)):
        accumulator: int | None = None
        for index in range(CYCLIC_WIDTH):
            shifted = (index + shift) % CYCLIC_WIDTH
            if index == CYCLIC_WIDTH - 1 or shifted == CYCLIC_WIDTH - 1:
                continue
            product = element[index] * element[shifted]
            product_bits = full.signed_bits(product)
            accumulator_bits = (
                0 if accumulator is None else full.signed_bits(accumulator)
            )
            next_accumulator = (
                product if accumulator is None else accumulator + product
            )
            next_bits = full.signed_bits(next_accumulator)
            partial_bits = sum(full.signed_bits(value) for value in output)
            retained_bits = (
                0 if retained_p8 is None else full.signed_bits(retained_p8)
            )
            working_bits = (
                partial_bits
                + retained_bits
                + product_bits
                + accumulator_bits
                + next_bits
            )
            stats.direct_real_hermitian_coefficient_multiplications += 1
            if accumulator is not None:
                stats.direct_real_hermitian_accumulation_additions += 1
            stats.maximum_direct_real_product_payload_bits = max(
                stats.maximum_direct_real_product_payload_bits,
                product_bits,
            )
            stats.maximum_direct_real_coordinate_accumulator_payload_bits = max(
                stats.maximum_direct_real_coordinate_accumulator_payload_bits,
                accumulator_bits,
                next_bits,
            )
            stats.maximum_direct_real_retained_p8_payload_bits = max(
                stats.maximum_direct_real_retained_p8_payload_bits,
                retained_bits,
            )
            stats.maximum_direct_real_partial_output_payload_bits = max(
                stats.maximum_direct_real_partial_output_payload_bits,
                partial_bits,
            )
            stats.maximum_direct_real_generator_work_payload_bits = max(
                stats.maximum_direct_real_generator_work_payload_bits,
                working_bits,
            )
            stats.maximum_direct_real_generation_live_payload_bits = max(
                stats.maximum_direct_real_generation_live_payload_bits,
                resident_accumulator_bits + working_bits,
            )
            accumulator = next_accumulator
        if accumulator is None:
            fail("oracle direct schedule unexpectedly had no products")
        if shift == REAL_WIDTH:
            retained_p8 = accumulator
            continue
        if retained_p8 is None:
            fail("oracle direct schedule lost P8")
        output.append(accumulator - retained_p8)
        stats.direct_real_hermitian_coefficient_subtractions += 1

    result = tuple(output)
    result_bits = power.s_payload_bits(result)
    stats.maximum_direct_real_partial_output_payload_bits = max(
        stats.maximum_direct_real_partial_output_payload_bits,
        result_bits,
    )
    stats.maximum_direct_real_term_payload_bits = max(
        stats.maximum_direct_real_term_payload_bits,
        result_bits,
    )
    stats.maximum_direct_real_generation_live_payload_bits = max(
        stats.maximum_direct_real_generation_live_payload_bits,
        resident_accumulator_bits + result_bits,
    )
    return result  # type: ignore[return-value]


def direct_power_vector_norm(
    vector: RingVector,
    stats: DirectPowerOracleStats,
) -> PowerElement:
    accumulator = power.power_zero()
    stats.streamed_real_norm_calls += 1
    stats.streamed_real_norm_input_cells += len(vector)
    if len(vector) == 1:
        stats.streamed_real_norm_singleton_calls += 1
    elif len(vector) == CYCLIC_WIDTH:
        stats.streamed_real_norm_full_carrier_calls += 1
    else:
        stats.streamed_real_norm_unexpected_width_calls += 1

    for element in vector:
        accumulator_bits = power.power_payload_bits(accumulator)
        stats.maximum_streamed_norm_real_accumulator_payload_bits = max(
            stats.maximum_streamed_norm_real_accumulator_payload_bits,
            accumulator_bits,
        )
        scheduled_s = account_direct_schedule(
            element,
            stats,
            accumulator_bits,
        )
        semantic_term = quadratic_extension_norm(element)
        if power.s_to_power(scheduled_s) != semantic_term:
            fail("quadratic-extension norm disagrees with direct schedule")
        ORACLE_QUADRATIC_COUNTS["schedule_parity_checks"] += 1
        term_bits = power.power_payload_bits(semantic_term)
        stats.streamed_real_norm_terms += 1
        stats.maximum_streamed_norm_real_term_payload_bits = max(
            stats.maximum_streamed_norm_real_term_payload_bits,
            term_bits,
        )
        next_accumulator = power_add(accumulator, semantic_term)
        next_bits = power.power_payload_bits(next_accumulator)
        addition_live = accumulator_bits + term_bits + next_bits
        stats.streamed_real_norm_real_additions += 1
        stats.maximum_streamed_norm_addition_live_payload_bits = max(
            stats.maximum_streamed_norm_addition_live_payload_bits,
            addition_live,
        )
        stats.maximum_streamed_norm_named_live_payload_bits = max(
            stats.maximum_streamed_norm_named_live_payload_bits,
            stats.maximum_direct_real_generation_live_payload_bits,
            addition_live,
        )
        accumulator = next_accumulator

    stats.maximum_streamed_norm_real_accumulator_payload_bits = max(
        stats.maximum_streamed_norm_real_accumulator_payload_bits,
        power.power_payload_bits(accumulator),
    )
    return accumulator


def independent_controls() -> dict[str, bool]:
    controls: list[RingElement] = []
    for left in range(CYCLIC_WIDTH - 1):
        basis = [0 for _ in range(CYCLIC_WIDTH - 1)]
        basis[left] = 1
        controls.append(tuple(basis))
        for right in range(left + 1, CYCLIC_WIDTH - 1):
            pair = list(basis)
            pair[right] = -2
            controls.append(tuple(pair))

    parity = True
    homogeneity = True
    for element in controls:
        semantic = quadratic_extension_norm(element)
        full_product = prior.ring.ring_multiply(
            element,
            prior.full.conjugate(element),
        )
        parity = parity and semantic == power.full_to_power(full_product)
        doubled = tuple(2 * coefficient for coefficient in element)
        homogeneity = homogeneity and quadratic_extension_norm(doubled) == tuple(
            4 * coefficient for coefficient in semantic
        )
    return {
        "all_136_signed_basis_and_pair_controls_match_full_product": parity,
        "all_136_homogeneity_controls_exact": homogeneity,
        "quadratic_extension_control_count_is_136": len(controls) == 136,
        "minimal_polynomial_matches_declared_degree8_field": (
            power.MINIMAL_POLYNOMIAL
            == (1, -4, -10, 10, 15, -6, -7, 1, 1)
        ),
    }


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_chain_period17_direct_real_hermitian_"
            "oracle.py PRODUCTION_RESULT"
        )

    prior.StreamedPowerOracleStats = DirectPowerOracleStats
    prior.metrics_json = metrics_json
    prior.streamed_power_vector_norm = direct_power_vector_norm

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        predecessor_rc = prior.main()
    if predecessor_rc != 0:
        fail("streamed oracle rejected direct Hermitian successor")
    result = json.loads(captured.getvalue())

    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        production = json.load(handle)
    controls = independent_controls()
    case_checks = {
        "all_case_resource_tuples_match": all(
            case["phase_resource_tuple_equal"]
            and case["inverse_resource_tuple_equal"]
            and case["named_search_temporary_sum_equal"]
            and case["named_component_total_equal"]
            for case in result["case_checks"]
        ),
        "all_case_direct_counts_match": all(
            case["exact_phase_resource_tuple"][
                "direct_real_hermitian_coefficient_multiplications"
            ]
            == case["exact_phase_resource_tuple"][
                "direct_real_hermitian_calls"
            ]
            * DIRECT_PRODUCTS_PER_TERM
            and case["exact_phase_resource_tuple"][
                "direct_real_hermitian_accumulation_additions"
            ]
            == case["exact_phase_resource_tuple"][
                "direct_real_hermitian_calls"
            ]
            * DIRECT_ADDITIONS_PER_TERM
            and case["exact_phase_resource_tuple"][
                "direct_real_hermitian_coefficient_subtractions"
            ]
            == case["exact_phase_resource_tuple"][
                "direct_real_hermitian_calls"
            ]
            * DIRECT_SUBTRACTIONS_PER_TERM
            for case in result["case_checks"]
        ),
        "no_case_degree16_norm_product_or_conversion": all(
            case["exact_phase_resource_tuple"][
                "streamed_real_norm_full_cyclotomic_multiplications"
            ]
            == 0
            and case["exact_phase_resource_tuple"][
                "direct_real_hermitian_full_cyclotomic_product_calls"
            ]
            == 0
            and case["exact_phase_resource_tuple"][
                "direct_real_hermitian_materialized_conjugate_calls"
            ]
            == 0
            and case["exact_phase_resource_tuple"][
                "maximum_direct_real_degree16_scratch_payload_bits"
            ]
            == 0
            and case["exact_phase_resource_tuple"][
                "full_to_real_conversions"
            ]
            == 0
            for case in result["case_checks"]
        ),
        "production_declares_identical_direct_classical_map": production[
            "matched_classical"
        ]["identical_direct_real_hermitian_map_available"],
        "production_does_not_claim_advantage": not production[
            "matched_classical"
        ]["comparison_establishes_advantage"],
    }
    result.update(
        {
            "result": (
                "PASS"
                if result["result"] == "PASS"
                and all(controls.values())
                and all(case_checks.values())
                else "FAIL"
            ),
            "experiment": (
                "SEPARATE_QUADRATIC_EXTENSION_EXACT_DIRECT_REAL_"
                "HERMITIAN_ORACLE"
            ),
            "oracle_imports_production_direct_module": False,
            "oracle_semantic_path_uses_cyclic_autocorrelation": False,
            "oracle_semantic_path_constructs_degree16_product": False,
            "oracle_resource_schedule_constructs_degree16_product": False,
            "quadratic_extension_oracle_counts": dict(
                ORACLE_QUADRATIC_COUNTS
            ),
            "independent_controls": controls,
            "direct_case_checks": case_checks,
            "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
            "verification_level": "SEPARATE_REFERENCE_PARITY",
            "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
            "claim_ceiling": (
                "LINUX_X86_64_PYTHON_TWO_PUBLIC_F17_PERIOD17_"
                "FAMILIES_PERIODS1AND64_DIRECT_8_COORDINATE_"
                "HERMITIAN_TERM_GENERATION_NO_ACCEPTED_DEGREE16_NORM_"
                "PRODUCT_ONE_FULL_CYCLOTOMIC_HORNER_CARRIER_ONE_FULL_"
                "CERTIFIED_ACTION_EXACT_BOUNDARY_RESOURCE_RESTORATION_"
                "AND_PERIOD1_CROSS_FAMILY_REUSE_PARITY_SOFTWARE_ONLY"
            ),
            "not_established": production["not_established"],
            "terminal": False,
        }
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
