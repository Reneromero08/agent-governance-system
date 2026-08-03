#!/usr/bin/env python3
"""Exact log-free coordinate reduction for the pi-factored F17 carrier.

The preceding global unit-lattice proposal reduced the tested resident
payload, but its accepted path included a 65,536-bit cosine table and a
float64 least-squares solve.  This successor removes both.  For each of the
seven declared cyclotomic-unit coordinates it minimizes the exact integer
trace energy along that coordinate by:

1. exact one-step direction selection;
2. exponential bracketing of the first nonnegative discrete difference;
3. binary search for the exact integer coordinate minimum; and
4. exact residual and ledger mutation only when energy decreases.

For a fixed unit-lattice direction the trace energy is a sum of positive
real embedding exponentials.  Its discrete differences are monotone, so the
bracketed search returns an exact line minimum without logarithms.  The
search frame contains the seven generator directions and all public pair
sum/difference directions.  Repeated full sweeps establish only a local
minimum over those 49 declared directions, not a global unit-lattice closest
vector.

The implementation reuses the established exact carrier, recurrence,
inverse, restoration, and unrelated-reuse machinery.  It is a bounded
phase-machine repair diagnostic; the identical exact method remains
available to compact classical software.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_pi_unit_embedding_balance as base


cyclo = base.cyclo
pi_content = base.pi_content
unit_reference = base.unit_reference

PRIME = base.PRIME
DIMENSION = base.DIMENSION
MESSAGE_SLOTS = base.MESSAGE_SLOTS
COEFFICIENT_REGISTERS = base.COEFFICIENT_REGISTERS
UNIT_RANK = base.UNIT_RANK
UNIT_GENERATORS = base.UNIT_GENERATORS
UNIT_GENERATOR_INVERSES = base.UNIT_GENERATOR_INVERSES
TESTED_PERIODS = (1, 64)
MAX_COORDINATE_SWEEPS = 64
MAX_BRACKET_MAGNITUDE = 1 << 20

RingElement = base.RingElement
RingVector = base.RingVector


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class ExactSearchStats(base.BalanceStats):
    coordinate_line_searches: int = 0
    coordinate_energy_probes: int = 0
    coordinate_bracket_expansions: int = 0
    coordinate_binary_search_steps: int = 0
    coordinate_moves_accepted: int = 0
    coordinate_sweeps_completed: int = 0
    coordinatewise_certified_calls: int = 0
    coordinate_sweep_cap_hits: int = 0
    coordinate_bracket_cap_hits: int = 0
    search_factor_ring_multiplications: int = 0
    maximum_coordinate_move_abs: int = 0
    maximum_bracket_magnitude: int = 0
    maximum_search_power_result_payload_bits: int = 0
    maximum_search_power_factor_payload_bits: int = 0
    maximum_search_power_live_pair_payload_bits: int = 0
    maximum_search_trial_norm_payload_bits: int = 0
    maximum_search_energy_scalar_pair_bits: int = 0
    maximum_accepted_move_scale_payload_bits: int = 0
    maximum_accepted_coordinate_vector_payload_bits: int = 0
    maximum_unit_scale_payload_bits: int = 0


ORIGINAL_STATS_JSON = base.stats_json
ORIGINAL_LEDGER_SCALE = base.ledger_scale


def stats_json(
    stats: ExactSearchStats,
    pi_stats: pi_content.PiStats,
) -> dict[str, Any]:
    result = ORIGINAL_STATS_JSON(stats, pi_stats)
    for name in (
        "coordinate_line_searches",
        "coordinate_energy_probes",
        "coordinate_bracket_expansions",
        "coordinate_binary_search_steps",
        "coordinate_moves_accepted",
        "coordinate_sweeps_completed",
        "coordinatewise_certified_calls",
        "coordinate_sweep_cap_hits",
        "coordinate_bracket_cap_hits",
        "search_factor_ring_multiplications",
        "maximum_coordinate_move_abs",
        "maximum_bracket_magnitude",
        "maximum_search_power_result_payload_bits",
        "maximum_search_power_factor_payload_bits",
        "maximum_search_power_live_pair_payload_bits",
        "maximum_search_trial_norm_payload_bits",
        "maximum_search_energy_scalar_pair_bits",
        "maximum_accepted_move_scale_payload_bits",
        "maximum_accepted_coordinate_vector_payload_bits",
        "maximum_unit_scale_payload_bits",
    ):
        result[name] = getattr(stats, name)
    return result


def tracked_ledger_scale(
    ledger: list[int],
    stats: base.BalanceStats | None = None,
) -> RingElement:
    result = ORIGINAL_LEDGER_SCALE(ledger, stats)
    if isinstance(stats, ExactSearchStats):
        stats.maximum_unit_scale_payload_bits = max(
            stats.maximum_unit_scale_payload_bits,
            base.element_payload_bits(result),
        )
    return result


SEARCH_DIRECTIONS = (
    tuple(
        1 if coordinate == index else 0
        for coordinate in range(UNIT_RANK)
    )
    for index in range(UNIT_RANK)
)
SEARCH_DIRECTIONS = tuple(SEARCH_DIRECTIONS) + tuple(
    direction
    for left in range(UNIT_RANK)
    for right in range(left + 1, UNIT_RANK)
    for sign in (1, -1)
    for direction in (
        tuple(
            (
                1
                if coordinate == left
                else sign
                if coordinate == right
                else 0
            )
            for coordinate in range(UNIT_RANK)
        ),
    )
)


def residual_multiplier_for_direction(
    direction: tuple[int, ...],
) -> RingElement:
    return base.ledger_scale([-value for value in direction])


DIRECTION_TABLE = tuple(
    (
        direction,
        positive_multiplier,
        negative_multiplier,
        cyclo.ring_multiply(
            positive_multiplier,
            base.ring_conjugate(positive_multiplier),
        ),
        cyclo.ring_multiply(
            negative_multiplier,
            base.ring_conjugate(negative_multiplier),
        ),
    )
    for direction in SEARCH_DIRECTIONS
    for positive_multiplier, negative_multiplier in (
        (
            residual_multiplier_for_direction(direction),
            base.ledger_scale(list(direction)),
        ),
    )
)


def exact_ring_power(
    value: RingElement,
    exponent: int,
    stats: ExactSearchStats,
) -> RingElement:
    if exponent < 0:
        fail("negative exact search exponent")
    result = cyclo.ring_one()
    factor = value
    remaining = exponent
    record_search_power_live_pair(result, factor, stats)
    while remaining:
        if remaining & 1:
            result = cyclo.ring_multiply(result, factor)
            stats.search_factor_ring_multiplications += 1
            record_search_power_live_pair(result, factor, stats)
        remaining >>= 1
        if remaining:
            factor = cyclo.ring_multiply(factor, factor)
            stats.search_factor_ring_multiplications += 1
            record_search_power_live_pair(result, factor, stats)
    return result


def record_search_power_live_pair(
    result: RingElement,
    factor: RingElement,
    stats: ExactSearchStats,
) -> None:
    result_bits = base.element_payload_bits(result)
    factor_bits = base.element_payload_bits(factor)
    stats.maximum_search_power_result_payload_bits = max(
        stats.maximum_search_power_result_payload_bits,
        result_bits,
    )
    stats.maximum_search_power_factor_payload_bits = max(
        stats.maximum_search_power_factor_payload_bits,
        factor_bits,
    )
    stats.maximum_search_power_live_pair_payload_bits = max(
        stats.maximum_search_power_live_pair_payload_bits,
        result_bits + factor_bits,
    )


def exact_coordinate_probe(
    current_norm: RingElement,
    direction_index: int,
    signed_delta: int,
    stats: ExactSearchStats,
) -> tuple[int, RingElement]:
    if signed_delta == 0:
        energy = base.exact_norm_energy(current_norm, stats)
        return energy, current_norm
    positive_factor = DIRECTION_TABLE[direction_index][3]
    negative_factor = DIRECTION_TABLE[direction_index][4]
    factor = exact_ring_power(
        positive_factor if signed_delta > 0 else negative_factor,
        abs(signed_delta),
        stats,
    )
    trial_norm = cyclo.ring_multiply(factor, current_norm)
    stats.candidate_norm_ring_multiplications += 1
    stats.coordinate_energy_probes += 1
    stats.balance_candidate_evaluations += 1
    stats.maximum_search_trial_norm_payload_bits = max(
        stats.maximum_search_trial_norm_payload_bits,
        base.element_payload_bits(trial_norm),
    )
    energy = base.exact_norm_energy(trial_norm, stats)
    return energy, trial_norm


def record_energy_pair(
    left: int,
    right: int,
    stats: ExactSearchStats,
) -> None:
    stats.maximum_search_energy_scalar_pair_bits = max(
        stats.maximum_search_energy_scalar_pair_bits,
        max(1, left.bit_length()) + max(1, right.bit_length()),
    )


def exact_coordinate_minimum(
    current_norm: RingElement,
    current_energy: int,
    direction_index: int,
    stats: ExactSearchStats,
) -> tuple[int, int, RingElement]:
    """Return exact signed coordinate move, energy, and resulting norm."""

    stats.coordinate_line_searches += 1
    positive_energy, positive_norm = exact_coordinate_probe(
        current_norm,
        direction_index,
        1,
        stats,
    )
    negative_energy, negative_norm = exact_coordinate_probe(
        current_norm,
        direction_index,
        -1,
        stats,
    )
    record_energy_pair(positive_energy, negative_energy, stats)
    directional = min(
        (
            (positive_energy, 1, positive_norm),
            (negative_energy, -1, negative_norm),
        ),
        key=lambda item: (item[0], item[1]),
    )
    if directional[0] >= current_energy:
        return 0, current_energy, current_norm

    direction = directional[1]
    high = 1
    high_energy = directional[0]
    high_norm = directional[2]
    low = 0
    while True:
        next_energy, _ = exact_coordinate_probe(
            current_norm,
            direction_index,
            direction * (high + 1),
            stats,
        )
        record_energy_pair(high_energy, next_energy, stats)
        if next_energy >= high_energy:
            break
        low = high
        high *= 2
        stats.coordinate_bracket_expansions += 1
        stats.maximum_bracket_magnitude = max(
            stats.maximum_bracket_magnitude,
            high,
        )
        if high > MAX_BRACKET_MAGNITUDE:
            stats.coordinate_bracket_cap_hits += 1
            return 0, current_energy, current_norm
        high_energy, high_norm = exact_coordinate_probe(
            current_norm,
            direction_index,
            direction * high,
            stats,
        )

    while low + 1 < high:
        midpoint = (low + high) // 2
        midpoint_energy, _ = exact_coordinate_probe(
            current_norm,
            direction_index,
            direction * midpoint,
            stats,
        )
        successor_energy, _ = exact_coordinate_probe(
            current_norm,
            direction_index,
            direction * (midpoint + 1),
            stats,
        )
        record_energy_pair(
            midpoint_energy,
            successor_energy,
            stats,
        )
        stats.coordinate_binary_search_steps += 1
        if successor_energy < midpoint_energy:
            low = midpoint
        else:
            high = midpoint

    optimum_energy, optimum_norm = exact_coordinate_probe(
        current_norm,
        direction_index,
        direction * high,
        stats,
    )
    if optimum_energy >= current_energy:
        return 0, current_energy, current_norm
    return direction * high, optimum_energy, optimum_norm


def balance_vector(
    vector: RingVector,
    base_ledger: list[int],
    stats: ExactSearchStats,
) -> tuple[RingVector, list[int]]:
    if len(base_ledger) != UNIT_RANK:
        fail("unit ledger width changed")
    if cyclo.vector_is_zero(vector):
        return cyclo.zero_vector(), [0 for _ in range(UNIT_RANK)]
    stats.balance_calls += 1
    current = list(vector)
    ledger = list(base_ledger)
    current_norm = base.vector_norm_element(current, stats)
    current_energy = base.exact_norm_energy(current_norm, stats)

    for _ in range(MAX_COORDINATE_SWEEPS):
        sweep_changed = False
        for direction_index, direction_entry in enumerate(DIRECTION_TABLE):
            move, trial_energy, trial_norm = exact_coordinate_minimum(
                current_norm,
                current_energy,
                direction_index,
                stats,
            )
            if move == 0:
                continue
            multiplier = base.ring_power(
                (
                    direction_entry[1]
                    if move > 0
                    else direction_entry[2]
                ),
                abs(move),
                stats,
            )
            stats.maximum_accepted_move_scale_payload_bits = max(
                stats.maximum_accepted_move_scale_payload_bits,
                base.element_payload_bits(multiplier),
            )
            current = base.multiply_vector(multiplier, current)
            stats.unit_vector_ring_multiplications += len(current)
            stats.maximum_accepted_coordinate_vector_payload_bits = max(
                stats.maximum_accepted_coordinate_vector_payload_bits,
                base.vector_payload_bits(current),
            )
            for generator_index, direction_coordinate in enumerate(
                direction_entry[0]
            ):
                ledger[generator_index] += (
                    move * direction_coordinate
                )
            current_norm = trial_norm
            current_energy = trial_energy
            stats.coordinate_moves_accepted += 1
            stats.balance_selected_steps += 1
            stats.maximum_coordinate_move_abs = max(
                stats.maximum_coordinate_move_abs,
                abs(move),
            )
            sweep_changed = True
        stats.coordinate_sweeps_completed += 1
        if not sweep_changed:
            stats.coordinatewise_certified_calls += 1
            return current, ledger

    stats.coordinate_sweep_cap_hits += 1
    stats.balance_step_cap_hits += 1
    return current, ledger


def compiled_unit_table_payload_bits() -> int:
    return sum(
        base.element_payload_bits(element)
        for element in (
            *UNIT_GENERATORS,
            *UNIT_GENERATOR_INVERSES,
            *(
                element
                for direction_entry in DIRECTION_TABLE
                for element in direction_entry[1:]
            ),
        )
    )


def exact_case_result(
    periods: int,
    block: cyclo.CompiledBlock,
) -> dict[str, Any]:
    result = base.case_result(periods, block)
    balanced_stats = result["balanced_stats"]
    named_search_temporary_maxima_sum_bits = sum(
        balanced_stats[name]
        for name in (
            "maximum_search_power_live_pair_payload_bits",
            "maximum_search_trial_norm_payload_bits",
            "maximum_search_energy_scalar_pair_bits",
            "maximum_accepted_move_scale_payload_bits",
            "maximum_accepted_coordinate_vector_payload_bits",
            "maximum_unit_scale_payload_bits",
            "maximum_unit_materialization_payload_bits",
        )
    )
    table_payload = compiled_unit_table_payload_bits()
    named_component_maxima_sum_bits = (
        result["balanced_declared_live_state_payload_bits"]
        + table_payload
        + named_search_temporary_maxima_sum_bits
    )
    raw_payload = result["raw_recurrence_payload_bits"]
    pi_payload = result["pi_content_payload_bits"]
    result.update(
        {
            "compiled_unit_table_payload_bits": table_payload,
            "named_exact_search_temporary_maxima_sum_bits": (
                named_search_temporary_maxima_sum_bits
            ),
            "named_component_maxima_sum_bits": (
                named_component_maxima_sum_bits
            ),
            "named_component_maxima_sum_minus_pi_content_payload_bits": (
                named_component_maxima_sum_bits - pi_payload
            ),
            "named_component_maxima_sum_minus_raw_payload_bits": (
                named_component_maxima_sum_bits - raw_payload
            ),
            "named_component_maxima_sum_beats_pi_content_payload": (
                named_component_maxima_sum_bits < pi_payload
            ),
            "named_component_maxima_sum_beats_raw_recurrence_payload": (
                named_component_maxima_sum_bits < raw_payload
            ),
            "all_nonzero_balance_calls_coordinatewise_certified": (
                balanced_stats["coordinatewise_certified_calls"]
                == balanced_stats["balance_calls"]
                and balanced_stats["coordinate_sweep_cap_hits"] == 0
                and balanced_stats["coordinate_bracket_cap_hits"] == 0
            ),
        }
    )
    return result


def exact_controls() -> dict[str, bool]:
    sample_vector = [
        cyclo.ring_add(cyclo.ring_one(), cyclo.ring_monomial(3)),
        cyclo.ring_add(
            cyclo.ring_monomial(2),
            cyclo.ring_monomial(7),
        ),
    ]
    sample_norm = base.vector_norm_element(sample_vector)
    sample_stats = ExactSearchStats()
    original_energy = base.exact_norm_energy(sample_norm, sample_stats)
    move, reduced_energy, reduced_norm = exact_coordinate_minimum(
        sample_norm,
        original_energy,
        0,
        sample_stats,
    )
    if move:
        multiplier = base.ring_power(
            UNIT_GENERATOR_INVERSES[0]
            if move > 0
            else UNIT_GENERATORS[0],
            abs(move),
        )
        moved = base.multiply_vector(multiplier, sample_vector)
    else:
        moved = sample_vector
    return {
        **base.exact_embedding_control(),
        "sample_line_minimum_energy_not_increased": (
            reduced_energy <= original_energy
        ),
        "sample_line_minimum_norm_matches_materialized_vector": (
            reduced_norm == base.vector_norm_element(moved)
        ),
        "sample_search_uses_no_bracket_cap": (
            sample_stats.coordinate_bracket_cap_hits == 0
        ),
        "compiled_unit_direction_table_exact": all(
            positive_multiplier
            == residual_multiplier_for_direction(direction)
            and negative_multiplier
            == base.ledger_scale(list(direction))
            and positive_factor
            == cyclo.ring_multiply(
                positive_multiplier,
                base.ring_conjugate(positive_multiplier),
            )
            and negative_factor
            == cyclo.ring_multiply(
                negative_multiplier,
                base.ring_conjugate(negative_multiplier),
            )
            for (
                direction,
                positive_multiplier,
                negative_multiplier,
                positive_factor,
                negative_factor,
            ) in DIRECTION_TABLE
        ),
    }


def main() -> int:
    if len(sys.argv) != 1:
        fail(
            "usage: f17_cubic_chain_period17_"
            "pi_unit_exact_coordinate_descent.py"
        )

    base.BalanceStats = ExactSearchStats
    base.stats_json = stats_json
    base.balance_vector = balance_vector
    base.ledger_scale = tracked_ledger_scale

    blocks = {
        family.lower(): cyclo.build_compiled_block(family)
        for family in ("PRIMARY", "REUSE")
    }
    cases = [
        exact_case_result(periods, blocks[family])
        for periods in TESTED_PERIODS
        for family in ("primary", "reuse")
    ]
    restored = base.restoration_reuse_case(
        blocks["primary"],
        blocks["reuse"],
    )
    control_results = base.controls(
        blocks["primary"],
        blocks["reuse"],
    )
    exact_search_controls = exact_controls()

    result = {
        "result": "PASS",
        "experiment": (
            "EXACT_LOG_FREE_COORDINATEWISE_CYCLOTOMIC_UNIT_"
            "REDUCTION_AFTER_PI_CONTENT_FACTORIZATION"
        ),
        "claim_candidate": (
            "BOUNDED_EXACT_LOG_FREE_49_DIRECTION_CYCLOTOMIC_UNIT_"
            "LINE_DESCENT_AFTER_PI_FACTORIZATION_REDUCES_RESIDENT_"
            "AND_DUPLICATE_REMATERIALIZATION_LIVE_PAYLOAD_BELOW_RAW_"
            "FOR_TWO_PUBLIC_F17_PERIOD17_FAMILIES_AT_PERIODS1AND64_"
            "WITH_EXACT_RESTORATION_AT_PERIODS1AND64_AND_CROSS_"
            "FAMILY_RESTORED_CARRIER_REUSE_AT_PERIOD1_BUT_PERIOD64_"
            "CONSERVATIVE_NAMED_COMPONENT_MAXIMA_SUM_REMAINS_ABOVE_"
            "RAW_AND_THE_IDENTICAL_CLASSICAL_EXECUTION_REMAINS"
        ),
        "classification_candidate": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level_candidate": "PACKAGE_SELF_REVIEW",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "coefficient_field": "Q_ZETA17",
        "integral_carrier_ring": "Z_ZETA17",
        "uniformizer": "PI_EQUALS_1_MINUS_ZETA17",
        "unit_generator_indices": list(
            unit_reference.UNIT_GENERATOR_INDICES
        ),
        "tested_periods": list(TESTED_PERIODS),
        "maximum_coordinate_sweeps": MAX_COORDINATE_SWEEPS,
        "maximum_bracket_magnitude": MAX_BRACKET_MAGNITUDE,
        "declared_exact_search_directions": [
            list(direction) for direction in SEARCH_DIRECTIONS
        ],
        "declared_exact_search_direction_count": len(SEARCH_DIRECTIONS),
        "search_numeric_type": "EXACT_Z_ZETA17_AND_INTEGER_TRACE_ONLY",
        "logarithms_used": False,
        "floating_point_used": False,
        "embedding_table_used": False,
        "coordinate_minimum_law": (
            "MONOTONE_DISCRETE_DIFFERENCE_OF_A_SUM_OF_POSITIVE_"
            "EMBEDDING_EXPONENTIALS_ALONG_EACH_OF_49_PUBLIC_"
            "GENERATOR_AND_PAIR_SUM_DIFFERENCE_DIRECTIONS"
        ),
        "global_unit_lattice_optimum_established": False,
        "block_certificates": {
            family: {
                "public_program_sha256": hashlib.sha256(
                    cyclo.adaptive.encoded_program(block.public_program)
                ).hexdigest(),
                "operator_sha256": block.operator_sha256,
                "characteristic_sha256": block.characteristic_sha256,
                "characteristic_identity_exact": (
                    block.characteristic_identity_exact
                ),
                "characteristic": block.characteristic,
            }
            for family, block in blocks.items()
        },
        "exact_controls": exact_search_controls,
        "cases": cases,
        "all_raw_recurrence_boundaries_equal": all(
            case["raw_recurrence_boundary_equal"] for case in cases
        ),
        "all_pi_content_boundaries_equal": all(
            case["pi_content_boundary_equal"] for case in cases
        ),
        "all_cases_restore_exactly": all(
            case["restored_exactly"]
            and case["same_backing"]
            and case["canonical_restored_state"][
                "all_payload_and_ledgers_zero"
            ]
            for case in cases
        ),
        "all_cases_coordinatewise_certified": all(
            case["all_nonzero_balance_calls_coordinatewise_certified"]
            for case in cases
        ),
        "all_cases_reduce_pi_content_payload": all(
            case["balanced_reduces_pi_content_payload"]
            for case in cases
        ),
        "all_cases_beat_raw_recurrence_payload": all(
            case["balanced_beats_raw_recurrence_payload"]
            for case in cases
        ),
        "all_declared_live_cases_beat_raw_recurrence_payload": all(
            case["balanced_declared_live_beats_raw_recurrence_payload"]
            for case in cases
        ),
        "all_named_component_maxima_sums_beat_raw": all(
            case["named_component_maxima_sum_beats_raw_recurrence_payload"]
            for case in cases
        ),
        "period1_named_component_maxima_sums_beat_raw": all(
            case["named_component_maxima_sum_beats_raw_recurrence_payload"]
            for case in cases
            if case["periods"] == 1
        ),
        "period64_named_component_maxima_sums_remain_above_raw": all(
            not case[
                "named_component_maxima_sum_beats_raw_recurrence_payload"
            ]
            for case in cases
            if case["periods"] == 64
        ),
        "restoration_reuse_case": restored,
        "controls": control_results,
        "matched_classical": {
            "identical_exact_coordinate_search_available": True,
            "identical_exact_recurrence_available": True,
            "raw_recurrence_retained": True,
            "pi_content_recurrence_retained": True,
            "comparison_establishes_advantage": False,
        },
        "resource_law": {
            "compiled_unit_table_payload_bits": (
                compiled_unit_table_payload_bits()
            ),
            "exact_trace_energy_probes_counted": True,
            "exponential_bracket_steps_counted": True,
            "binary_search_steps_counted": True,
            "search_factor_ring_multiplications_counted": True,
            "search_power_result_factor_live_pairs_counted": True,
            "accepted_move_scale_payload_counted": True,
            "unit_scale_payload_counted": True,
            "unit_materialization_counted": True,
            "duplicate_public_topology_state_payload_counted": True,
            "named_component_maxima_sum_is_simultaneous_peak": False,
            "python_object_overhead_bounded": False,
            "allocator_peak_bounded": False,
            "internal_ring_multiplication_peak_bounded": False,
            "whole_process_peak_bounded": False,
        },
        "observation": (
            "EXACT_LOG_FREE_49_DIRECTION_LINE_DESCENT_REMOVES_THE_"
            "65536_BIT_EMBEDDING_TABLE_AND_FLOATING_SOLVE_AND_"
            "REDUCES_TESTED_RESIDENT_AND_DUPLICATE_LIVE_PAYLOAD_"
            "BELOW_RAW_BUT_EXACT_SEARCH_FACTOR_TRIAL_NORM_ACCEPTED_"
            "VECTOR_AND_UNIT_MATERIALIZATION_TEMPORARIES_LEAVE_THE_"
            "PERIOD64_CONSERVATIVE_NAMED_MAXIMA_SUM_ABOVE_RAW"
        ),
        "not_established": [
            "GLOBAL_CYCLOTOMIC_UNIT_OPTIMALITY",
            "MULTIPLICATIVE_INDEPENDENCE_OF_DECLARED_UNIT_GENERATORS",
            "FIXED_RESIDUAL_INTEGER_WIDTH",
            "FIXED_TOTAL_BIT_FOOTPRINT",
            "ASYMPTOTIC_RESIDUAL_HEIGHT_BOUND",
            "SIMULTANEOUS_PROCESS_PEAK_FROM_NAMED_COMPONENT_MAXIMA",
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
            "DEFERRED_EXACT_UNIT_LEDGER_SEARCH_WITH_SINGLE_NET_"
            "RESIDUAL_ACTION_AND_STREAMED_UNIT_MATERIALIZATION"
        ),
        "next_obstruction": (
            "THE_LARGE_PRECISION_TABLE_AND_FLOATING_SOLVE_ARE_REMOVED_"
            "BUT_PERIOD64_EXACT_SEARCH_AND_UNIT_MATERIALIZATION_"
            "TEMPORARIES_EXCEED_THE_RAW_RECURRENCE_THE_IDENTICAL_"
            "CLASSICAL_EXECUTION_REMAINS_AND_NO_ADVANTAGE_IS_"
            "ESTABLISHED"
        ),
        "generation_and_lease_are_observed_bookkeeping_only": True,
        "generation_or_lease_enforcement_established": False,
        "terminal": False,
    }

    hard_gate_details = {
        "exact_search_controls": all(exact_search_controls.values()),
        "raw_boundaries": result["all_raw_recurrence_boundaries_equal"],
        "pi_boundaries": result["all_pi_content_boundaries_equal"],
        "restoration": result["all_cases_restore_exactly"],
        "coordinatewise_certified": (
            result["all_cases_coordinatewise_certified"]
        ),
        "resident_payload_reduced": (
            result["all_cases_reduce_pi_content_payload"]
        ),
        "resident_payload_below_raw": (
            result["all_cases_beat_raw_recurrence_payload"]
        ),
        "declared_live_payload_below_raw": (
            result["all_declared_live_cases_beat_raw_recurrence_payload"]
        ),
        "period1_named_maxima_sum_below_raw": (
            result["period1_named_component_maxima_sums_beat_raw"]
        ),
        "period64_named_maxima_sum_above_raw": (
            result["period64_named_component_maxima_sums_remain_above_raw"]
        ),
        "controls": all(control_results.values()),
        "primary_reuse_restored": restored["primary_restored_exactly"],
        "unrelated_reuse_restored": restored["reuse_restored_exactly"],
        "same_original_backing": restored["same_original_backing"],
        "fresh_restored_reuse_boundary": (
            restored["fresh_restored_reuse_boundary_equal"]
        ),
        "case_caps": [
            {
                "periods": case["periods"],
                "family": case["family"],
                "sweep_cap_hits": case["balanced_stats"][
                    "coordinate_sweep_cap_hits"
                ],
                "bracket_cap_hits": case["balanced_stats"][
                    "coordinate_bracket_cap_hits"
                ],
                "balance_calls": case["balanced_stats"]["balance_calls"],
                "certified_calls": case["balanced_stats"][
                    "coordinatewise_certified_calls"
                ],
            }
            for case in cases
        ],
    }
    if not all(
        value
        for key, value in hard_gate_details.items()
        if key != "case_caps"
    ):
        fail(
            "exact coordinate reduction qualification failed: "
            + json.dumps(hard_gate_details, sort_keys=True)
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
