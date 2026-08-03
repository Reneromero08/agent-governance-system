#!/usr/bin/env python3
"""Generate exact real-subfield Hermitian terms without a full product.

For a canonical element ``x=sum(a_j*zeta^j, j=0..15)`` this successor
uses the missing seventeenth coefficient ``a_16=0`` and emits the integral
real-basis coordinates of ``x*conjugate(x)`` directly:

    P_k = sum_j a_j*a_(j+k),  indices mod 17,
    r_k = P_k-P_8,  k=0..7.

These are exactly the coefficients of ``(1,s_1,...,s_7)`` after cyclotomic
reduction.  The accepted norm path therefore constructs neither a summed
degree-sixteen norm nor a per-element degree-sixteen Hermitian product.

The phase carrier and the certified unit action remain full cyclotomic, and
compact classical software can execute the identical bilinear map.  This is
a bounded representation repair, not evidence of a distinct phase resource
or computational advantage.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_streamed_real_autocorrelation as prior


base = prior.base
cyclo = prior.cyclo

RingElement = prior.prior.RingElement
RingVector = prior.RingVector
RealElement = prior.RealElement

ORIGINAL_STATS_JSON = prior.stats_json
ORIGINAL_CASE_RESULT = prior.streamed_case_result

CYCLIC_WIDTH = cyclo.PRIME
REAL_WIDTH = 8
DIRECT_PRODUCTS_PER_TERM = CYCLIC_WIDTH * REAL_WIDTH
DIRECT_ADDITIONS_PER_TERM = 127
DIRECT_SUBTRACTIONS_PER_TERM = REAL_WIDTH


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class DirectHermitianStats(prior.StreamedRealStats):
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


def stats_json(
    stats: DirectHermitianStats,
    pi_stats: prior.pi_content.PiStats,
) -> dict[str, Any]:
    result = ORIGINAL_STATS_JSON(stats, pi_stats)
    for name in DIRECT_FIELDS:
        result[name] = getattr(stats, name)
    return result


def direct_real_hermitian_term(
    element: RingElement,
    stats: DirectHermitianStats,
    resident_accumulator_bits: int = 0,
) -> RealElement:
    """Return x*conjugate(x) in the integral real basis directly."""

    if len(element) != CYCLIC_WIDTH - 1:
        fail("canonical cyclotomic element width changed")
    output: list[int] = []
    stats.direct_real_hermitian_calls += 1

    correlations: list[int] = []
    for coordinate in (REAL_WIDTH, *range(REAL_WIDTH)):
        accumulator: int | None = None
        for index in range(CYCLIC_WIDTH):
            shifted = (index + coordinate) % CYCLIC_WIDTH
            if index == CYCLIC_WIDTH - 1 or shifted == CYCLIC_WIDTH - 1:
                continue
            product = element[index] * element[shifted]
            product_bits = base.signed_bits(product)
            accumulator_bits = (
                0 if accumulator is None else base.signed_bits(accumulator)
            )
            next_accumulator = (
                product if accumulator is None else accumulator + product
            )
            next_accumulator_bits = base.signed_bits(next_accumulator)
            partial_output_bits = sum(
                base.signed_bits(value) for value in output
            )
            retained_p8_bits = (
                0 if not correlations else base.signed_bits(correlations[0])
            )
            generator_work = (
                partial_output_bits
                + retained_p8_bits
                + product_bits
                + accumulator_bits
                + next_accumulator_bits
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
                next_accumulator_bits,
            )
            stats.maximum_direct_real_partial_output_payload_bits = max(
                stats.maximum_direct_real_partial_output_payload_bits,
                partial_output_bits,
            )
            stats.maximum_direct_real_retained_p8_payload_bits = max(
                stats.maximum_direct_real_retained_p8_payload_bits,
                retained_p8_bits,
            )
            stats.maximum_direct_real_generator_work_payload_bits = max(
                stats.maximum_direct_real_generator_work_payload_bits,
                generator_work,
            )
            stats.maximum_direct_real_generation_live_payload_bits = max(
                stats.maximum_direct_real_generation_live_payload_bits,
                resident_accumulator_bits + generator_work,
            )
            accumulator = next_accumulator
        if accumulator is None:
            fail("direct correlation unexpectedly had no terms")
        if coordinate == REAL_WIDTH:
            correlations.append(accumulator)
            continue
        real_coordinate = accumulator - correlations[0]
        stats.direct_real_hermitian_coefficient_subtractions += 1
        output.append(real_coordinate)

    result: RealElement = tuple(output)  # type: ignore[assignment]
    result_bits = prior.prior.real_payload_bits(result)
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
    return result


def direct_real_vector_norm(
    vector: RingVector,
    stats: DirectHermitianStats,
) -> RealElement:
    """Stream direct real Hermitian terms into one real accumulator."""

    accumulator = prior.prior.real_zero()
    stats.streamed_real_norm_calls += 1
    stats.streamed_real_norm_input_cells += len(vector)
    if len(vector) == 1:
        stats.streamed_real_norm_singleton_calls += 1
    elif len(vector) == CYCLIC_WIDTH:
        stats.streamed_real_norm_full_carrier_calls += 1
    else:
        stats.streamed_real_norm_unexpected_width_calls += 1

    for element in vector:
        accumulator_bits = prior.prior.real_payload_bits(accumulator)
        stats.maximum_streamed_norm_real_accumulator_payload_bits = max(
            stats.maximum_streamed_norm_real_accumulator_payload_bits,
            accumulator_bits,
        )
        real_term = direct_real_hermitian_term(
            element,
            stats,
            accumulator_bits,
        )
        real_term_bits = prior.prior.real_payload_bits(real_term)
        stats.streamed_real_norm_terms += 1
        stats.maximum_streamed_norm_real_term_payload_bits = max(
            stats.maximum_streamed_norm_real_term_payload_bits,
            real_term_bits,
        )

        next_accumulator = prior.prior.real_add(accumulator, real_term)
        next_bits = prior.prior.real_payload_bits(next_accumulator)
        addition_live = accumulator_bits + real_term_bits + next_bits
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
        prior.prior.real_payload_bits(accumulator),
    )
    return accumulator


def direct_case_result(
    periods: int,
    block: cyclo.CompiledBlock,
) -> dict[str, Any]:
    result = ORIGINAL_CASE_RESULT(periods, block)
    phase_metrics = result["phase_stats"]
    expected_products = (
        phase_metrics["streamed_real_norm_terms"]
        * DIRECT_PRODUCTS_PER_TERM
    )
    result.update(
        {
            "full_cyclotomic_per_element_norm_products_remain": False,
            "direct_real_hermitian_terms_generated": True,
            "direct_real_hermitian_operation_count_exact": (
                phase_metrics[
                    "direct_real_hermitian_coefficient_multiplications"
                ]
                == expected_products
                and phase_metrics[
                    "direct_real_hermitian_coefficient_subtractions"
                ]
                == phase_metrics["streamed_real_norm_terms"]
                * DIRECT_SUBTRACTIONS_PER_TERM
                and phase_metrics[
                    "direct_real_hermitian_accumulation_additions"
                ]
                == phase_metrics["streamed_real_norm_terms"]
                * DIRECT_ADDITIONS_PER_TERM
            ),
        }
    )
    return result


def direct_formula_controls() -> dict[str, bool]:
    """Check the quadratic map on a basis spanning set outside the path."""

    vectors = []
    for left in range(CYCLIC_WIDTH - 1):
        basis = [0 for _ in range(CYCLIC_WIDTH - 1)]
        basis[left] = 1
        vectors.append(tuple(basis))
        for right in range(left + 1, CYCLIC_WIDTH - 1):
            pair = list(basis)
            pair[right] = 1
            vectors.append(tuple(pair))

    parity = True
    for element in vectors:
        control_stats = DirectHermitianStats()
        direct = direct_real_hermitian_term(element, control_stats)
        full = cyclo.ring_multiply(
            element,
            prior.base.ring_conjugate(element),
        )
        parity = parity and direct == prior.prior.full_to_real(full)
    return {
        "all_136_basis_and_pair_sum_controls_equal_full_product": parity,
        "quadratic_spanning_control_count_is_136": len(vectors) == 136,
        "direct_products_per_term_is_136": (
            DIRECT_PRODUCTS_PER_TERM == 136
        ),
    }


def main() -> int:
    if len(sys.argv) != 1:
        fail("usage: f17_cubic_chain_period17_direct_real_hermitian.py")

    prior.StreamedRealStats = DirectHermitianStats
    prior.stats_json = stats_json
    prior.streamed_real_vector_norm = direct_real_vector_norm
    prior.streamed_case_result = direct_case_result

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        predecessor_rc = prior.main()
    if predecessor_rc != 0:
        fail("streamed predecessor rejected direct Hermitian successor")
    result = json.loads(captured.getvalue())

    formula_controls = direct_formula_controls()
    all_direct_counts = all(
        case["direct_real_hermitian_operation_count_exact"]
        and case["phase_stats"]["direct_real_hermitian_calls"]
        == case["phase_stats"]["streamed_real_norm_terms"]
        and case["phase_stats"][
            "streamed_real_norm_full_cyclotomic_multiplications"
        ]
        == 0
        and case["phase_stats"][
            "direct_real_hermitian_full_cyclotomic_product_calls"
        ]
        == 0
        and case["phase_stats"][
            "direct_real_hermitian_materialized_conjugate_calls"
        ]
        == 0
        and case["phase_stats"][
            "maximum_direct_real_degree16_scratch_payload_bits"
        ]
        == 0
        and case["phase_stats"]["full_to_real_conversions"] == 0
        for case in result["cases"]
    )

    result.update(
        {
            "experiment": (
                "EXACT_REAL_SUBFIELD_HERMITIAN_TERM_GENERATOR_WITHOUT_"
                "PER_ELEMENT_FULL_CYCLOTOMIC_PRODUCT"
            ),
            "claim_candidate": (
                "BOUNDED_EXACT_DIRECT_EIGHT_COORDINATE_HERMITIAN_"
                "GENERATOR_ELIMINATES_PER_ELEMENT_DEGREE16_PRODUCTS_"
                "FROM_THE_ACCEPTED_NORM_PATH_WHILE_PRESERVING_"
                "BOUNDARIES_EXACT_RESTORATION_AND_PERIOD1_REUSE"
            ),
            "classification_candidate": "SOURCE_AUDITED_PACKAGE_LOCAL",
            "verification_level_candidate": "PACKAGE_SELF_REVIEW",
            "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
            "full_cyclotomic_aggregate_norm_constructed": False,
            "full_cyclotomic_per_element_norm_products_remain": False,
            "direct_real_hermitian_generator_used": True,
            "direct_real_formula": (
                "P_k=sum_j a_j*a_(j+k) mod17; r_k=P_k-P_8 "
                "for k=0..7; a_16=0"
            ),
            "all_direct_operation_counts_exact": all_direct_counts,
            "direct_formula_controls": formula_controls,
            "verification_baseline": {
                "outside_accepted_path": True,
                "full_cyclotomic_products": 136,
                "full_to_real_conversions": 136,
                "purpose": "quadratic_basis_and_pair_sum_parity_only",
            },
            "direct_generator_public_plan_payload_bits": 0,
            "matched_classical": {
                **result["matched_classical"],
                "identical_direct_real_hermitian_map_available": True,
                "same_136_integer_products_per_term": True,
                "same_127_accumulation_additions_and_8_subtractions_per_term": True,
                "comparison_establishes_advantage": False,
            },
            "resource_law": {
                **result["resource_law"],
                "accepted_per_element_full_products_counted": False,
                "direct_integer_products_counted": True,
                "direct_coordinate_accumulations_counted": True,
                "direct_integer_subtractions_counted": True,
                "direct_partial_output_counted": True,
                "direct_generation_live_payload_counted": True,
                "source_carrier_payload_counted_once": True,
                "source_element_accessed_by_reference": True,
                "shallow_python_container_and_reference_bytes_excluded": True,
                "python_bigint_allocator_and_native_library_bytes_excluded": True,
                "verification_full_products_excluded_and_reported": True,
                "whole_process_peak_bounded": False,
            },
            "observation": (
                "DIRECT_AUTOCORRELATION_REMOVES_EACH_DEGREE16_"
                "HERMITIAN_PRODUCT_FROM_THE_ACCEPTED_NORM_PATH_BUT_"
                "THE_RESIDENT_CARRIER_AND_CERTIFIED_ACTION_REMAIN_"
                "FULL_CYCLOTOMIC_AND_THE_IDENTICAL_COMPACT_CLASSICAL_"
                "BILINEAR_MAP_IS_AVAILABLE"
            ),
            "not_established": sorted(
                set(result["not_established"])
                - {"ELIMINATION_OF_PER_ELEMENT_FULL_CYCLOTOMIC_PRODUCTS"}
                | {
                    "FULL_REAL_SUBFIELD_CARRIER",
                    "REAL_SUBFIELD_CERTIFIED_UNIT_ACTION",
                    "DISTINCT_PHASE_RESOURCE",
                    "COMPUTATIONAL_ADVANTAGE",
                    "SMALL_WALL_CROSSING",
                    "MACHINE_ENFORCED_NO_SMUGGLE_OR_CATVM_CUSTODY",
                    "PHYSICAL_WAVEFORM_EXECUTION",
                    "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
                    "UNBOUNDED_COMPUTATION",
                }
            ),
            "next_experiment": (
                "PHASE_NATIVE_NONCLASSICAL_TRACE_COUPLING_OR_EXACT_"
                "FULL_CARRIER_PHASE_QUOTIENT_WITH_BOUNDARY_LIFT"
            ),
            "next_obstruction": (
                "THE_NORM_SEARCH_NO_LONGER_NEEDS_ANY_DEGREE16_"
                "HERMITIAN_PRODUCT_BUT_THE_PHASE_CARRIER_AND_CERTIFIED_"
                "ACTION_REMAIN_FULL_CYCLOTOMIC_AND_COMPACT_CLASSICAL_"
                "SOFTWARE_EXECUTES_THE_IDENTICAL_DIRECT_BILINEAR_MAP"
            ),
            "terminal": False,
        }
    )

    hard_gate = {
        "predecessor_result": result["result"] == "PASS",
        "direct_counts": all_direct_counts,
        "formula_controls": all(formula_controls.values()),
        "no_full_aggregate": (
            not result["full_cyclotomic_aggregate_norm_constructed"]
        ),
        "no_per_element_full_products": (
            not result["full_cyclotomic_per_element_norm_products_remain"]
        ),
        "boundaries": (
            result["all_raw_horner_boundaries_equal"]
            and result["all_prior_raw_recurrence_boundaries_equal"]
        ),
        "restoration": result["all_cases_restore_exactly"],
        "same_backing": result["restoration_reuse_case"][
            "same_original_backing"
        ],
        "fresh_restored_reuse": result["restoration_reuse_case"][
            "fresh_restored_reuse_boundary_equal"
        ],
    }
    if not all(hard_gate.values()):
        fail(
            "direct real Hermitian qualification failed: "
            + json.dumps(hard_gate, sort_keys=True)
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
