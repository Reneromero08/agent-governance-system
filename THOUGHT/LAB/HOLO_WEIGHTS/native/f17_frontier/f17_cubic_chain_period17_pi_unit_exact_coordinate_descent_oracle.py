#!/usr/bin/env python3
"""Separate exact oracle for the log-free 49-direction unit search.

The oracle imports only the independently implemented recurrence and
cyclotomic-ring reference kernels.  It does not import the production
successor.  It recompiles both public families, advances recurrence
coefficients sequentially, reimplements the exact direction-line search,
and compares boundaries, exact state/resource tuples, restoration, reuse,
and semantic mutation behavior with the production result.

Agreement establishes strict bounded parity only.  It does not establish a
global unit-lattice optimum, a distinct phase resource, or an advantage over
the identical compact classical recurrence.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_pi_unit_embedding_balance_oracle as prior
import f17_cubic_chain_period17_unit_height_reduction_oracle as ring


PRIME = 17
UNIT_RANK = 7
EXPECTED_PERIODS = (1, 64)
MAX_DIRECTION_SWEEPS = 64
MAX_BRACKET_MAGNITUDE = 1 << 20

RingElement = tuple[int, ...]
RingVector = list[RingElement]


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class SearchStats(prior.OracleStats):
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


ORIGINAL_LEDGER_SCALE = prior.ledger_scale


def tracked_ledger_scale(
    ledger: tuple[int, ...] | list[int],
    stats: prior.OracleStats,
) -> RingElement:
    result = ORIGINAL_LEDGER_SCALE(ledger, stats)
    if isinstance(stats, SearchStats):
        stats.maximum_unit_scale_payload_bits = max(
            stats.maximum_unit_scale_payload_bits,
            prior.element_payload_bits(result),
        )
    return result


SEARCH_DIRECTIONS = tuple(
    tuple(
        1 if coordinate == index else 0
        for coordinate in range(UNIT_RANK)
    )
    for index in range(UNIT_RANK)
) + tuple(
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


def ledger_scale_without_stats(
    ledger: tuple[int, ...] | list[int],
) -> RingElement:
    result = ring.ring_one()
    for exponent, generator, inverse in zip(
        ledger,
        ring.UNIT_GENERATORS,
        ring.UNIT_GENERATOR_INVERSES,
        strict=True,
    ):
        result = ring.ring_multiply(
            result,
            ring.ring_power(
                generator if exponent >= 0 else inverse,
                abs(exponent),
            ),
        )
    return result


DIRECTION_TABLE = tuple(
    (
        direction,
        positive_multiplier,
        negative_multiplier,
        ring.ring_multiply(
            positive_multiplier,
            prior.conjugate(positive_multiplier),
        ),
        ring.ring_multiply(
            negative_multiplier,
            prior.conjugate(negative_multiplier),
        ),
    )
    for direction in SEARCH_DIRECTIONS
    for positive_multiplier, negative_multiplier in (
        (
            ledger_scale_without_stats(
                tuple(-value for value in direction)
            ),
            ledger_scale_without_stats(direction),
        ),
    )
)


def exact_power(
    value: RingElement,
    exponent: int,
    stats: SearchStats,
) -> RingElement:
    if exponent < 0:
        fail("oracle negative search exponent")
    result = ring.ring_one()
    factor = value
    remaining = exponent
    record_search_power_live_pair(result, factor, stats)
    while remaining:
        if remaining & 1:
            result = ring.ring_multiply(result, factor)
            stats.search_factor_ring_multiplications += 1
            record_search_power_live_pair(result, factor, stats)
        remaining >>= 1
        if remaining:
            factor = ring.ring_multiply(factor, factor)
            stats.search_factor_ring_multiplications += 1
            record_search_power_live_pair(result, factor, stats)
    return result


def record_search_power_live_pair(
    result: RingElement,
    factor: RingElement,
    stats: SearchStats,
) -> None:
    result_bits = prior.element_payload_bits(result)
    factor_bits = prior.element_payload_bits(factor)
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


def probe(
    current_norm: RingElement,
    direction_index: int,
    signed_delta: int,
    stats: SearchStats,
) -> tuple[int, RingElement]:
    positive_factor = DIRECTION_TABLE[direction_index][3]
    negative_factor = DIRECTION_TABLE[direction_index][4]
    factor = exact_power(
        positive_factor if signed_delta > 0 else negative_factor,
        abs(signed_delta),
        stats,
    )
    trial_norm = ring.ring_multiply(factor, current_norm)
    stats.candidate_norm_ring_multiplications += 1
    stats.coordinate_energy_probes += 1
    stats.balance_candidate_evaluations += 1
    stats.maximum_search_trial_norm_payload_bits = max(
        stats.maximum_search_trial_norm_payload_bits,
        prior.element_payload_bits(trial_norm),
    )
    energy = prior.observe_energy(
        prior.field_trace(trial_norm),
        stats,
    )
    return energy, trial_norm


def record_energy_pair(
    left: int,
    right: int,
    stats: SearchStats,
) -> None:
    stats.maximum_search_energy_scalar_pair_bits = max(
        stats.maximum_search_energy_scalar_pair_bits,
        max(1, left.bit_length()) + max(1, right.bit_length()),
    )


def line_minimum(
    current_norm: RingElement,
    current_energy: int,
    direction_index: int,
    stats: SearchStats,
) -> tuple[int, int, RingElement]:
    stats.coordinate_line_searches += 1
    positive_energy, positive_norm = probe(
        current_norm,
        direction_index,
        1,
        stats,
    )
    negative_energy, negative_norm = probe(
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
    low = 0
    while True:
        successor_energy, _ = probe(
            current_norm,
            direction_index,
            direction * (high + 1),
            stats,
        )
        record_energy_pair(high_energy, successor_energy, stats)
        if successor_energy >= high_energy:
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
        high_energy, _ = probe(
            current_norm,
            direction_index,
            direction * high,
            stats,
        )

    while low + 1 < high:
        midpoint = (low + high) // 2
        midpoint_energy, _ = probe(
            current_norm,
            direction_index,
            direction * midpoint,
            stats,
        )
        successor_energy, _ = probe(
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

    optimum_energy, optimum_norm = probe(
        current_norm,
        direction_index,
        direction * high,
        stats,
    )
    if optimum_energy >= current_energy:
        return 0, current_energy, current_norm
    return direction * high, optimum_energy, optimum_norm


def balance(
    vector: RingVector,
    base_ledger: tuple[int, ...] | list[int],
    stats: SearchStats,
) -> tuple[RingVector, tuple[int, ...]]:
    if len(base_ledger) != UNIT_RANK:
        fail("oracle unit ledger width changed")
    if all(element == ring.ring_zero() for element in vector):
        return (
            [ring.ring_zero() for _ in vector],
            tuple(0 for _ in range(UNIT_RANK)),
        )
    stats.balance_calls += 1
    current = list(vector)
    ledger = list(base_ledger)
    current_norm = prior.norm_element(current, stats)
    current_energy = prior.observe_energy(
        prior.field_trace(current_norm),
        stats,
    )
    for _ in range(MAX_DIRECTION_SWEEPS):
        sweep_changed = False
        for direction_index, direction_entry in enumerate(DIRECTION_TABLE):
            move, trial_energy, trial_norm = line_minimum(
                current_norm,
                current_energy,
                direction_index,
                stats,
            )
            if move == 0:
                continue
            multiplier = prior.ring_power(
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
                prior.element_payload_bits(multiplier),
            )
            current = prior.multiply_vector(multiplier, current)
            stats.unit_vector_ring_multiplications += len(current)
            stats.maximum_accepted_coordinate_vector_payload_bits = max(
                stats.maximum_accepted_coordinate_vector_payload_bits,
                prior.vector_payload_bits(current),
            )
            for generator_index, coordinate in enumerate(
                direction_entry[0]
            ):
                ledger[generator_index] += move * coordinate
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
            return current, tuple(ledger)
    stats.coordinate_sweep_cap_hits += 1
    stats.balance_step_cap_hits += 1
    return current, tuple(ledger)


ORIGINAL_RECORD_METRICS = prior.record_metrics


def record_metrics(
    messages: list[prior.BalancedVector],
    coefficients: list[prior.BalancedElement],
    stats: SearchStats,
) -> dict[str, int]:
    result = ORIGINAL_RECORD_METRICS(messages, coefficients, stats)
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


def compiled_unit_table_payload_bits() -> int:
    return sum(
        prior.element_payload_bits(element)
        for element in (
            *ring.UNIT_GENERATORS,
            *ring.UNIT_GENERATOR_INVERSES,
            *(
                element
                for direction_entry in DIRECTION_TABLE
                for element in direction_entry[1:]
            ),
        )
    )


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_chain_period17_"
            "pi_unit_exact_coordinate_descent_oracle.py "
            "PRODUCTION_RESULT"
        )
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        production = json.load(handle)
    if tuple(production["tested_periods"]) != EXPECTED_PERIODS:
        fail("oracle tested periods changed")

    prior.OracleStats = SearchStats
    prior.balance = balance
    prior.record_metrics = record_metrics
    prior.ledger_scale = tracked_ledger_scale

    contexts: dict[str, dict[str, Any]] = {}
    family_checks: dict[str, dict[str, bool]] = {}
    for family in ("primary", "reuse"):
        checks, context = prior.family_context(
            family,
            production["block_certificates"][family],
        )
        family_checks[family] = checks
        contexts[family] = context

    case_checks = []
    for case in production["cases"]:
        checked = prior.case_check(
            case,
            contexts[case["family"].lower()],
        )
        metrics = checked["exact_resource_tuple"]
        named_temporary = sum(
            metrics[name]
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
        named_total = (
            metrics["maximum_declared_live_state_payload_bits"]
            + compiled_unit_table_payload_bits()
            + named_temporary
        )
        checked.update(
            {
                "compiled_unit_table_payload_equal": (
                    compiled_unit_table_payload_bits()
                    == case["compiled_unit_table_payload_bits"]
                ),
                "named_exact_search_temporary_maxima_sum_equal": (
                    named_temporary
                    == case[
                        "named_exact_search_temporary_maxima_sum_bits"
                    ]
                ),
                "named_component_maxima_sum_equal": (
                    named_total
                    == case["named_component_maxima_sum_bits"]
                ),
                "declared_direction_local_minimum_reexecuted": (
                    metrics["coordinatewise_certified_calls"]
                    == metrics["balance_calls"]
                    and metrics["coordinate_sweep_cap_hits"] == 0
                    and metrics["coordinate_bracket_cap_hits"] == 0
                ),
            }
        )
        case_checks.append(checked)

    restoration = prior.restoration_check(contexts, production)
    mutations = prior.mutation_check(contexts["primary"])
    scope = {
        "production_result_pass": production["result"] == "PASS",
        "exact_log_free_search_asserted": (
            not production["logarithms_used"]
            and not production["floating_point_used"]
            and not production["embedding_table_used"]
        ),
        "all_boundaries_equal": (
            production["all_raw_recurrence_boundaries_equal"]
            and production["all_pi_content_boundaries_equal"]
        ),
        "all_restored": production["all_cases_restore_exactly"],
        "all_declared_direction_local_minima": (
            production["all_cases_coordinatewise_certified"]
        ),
        "all_resident_payloads_below_raw": (
            production["all_cases_beat_raw_recurrence_payload"]
        ),
        "all_duplicate_live_payloads_below_raw": (
            production["all_declared_live_cases_beat_raw_recurrence_payload"]
        ),
        "period1_named_sums_below_raw": (
            production["period1_named_component_maxima_sums_beat_raw"]
        ),
        "period64_named_sums_above_raw": (
            production[
                "period64_named_component_maxima_sums_remain_above_raw"
            ]
        ),
        "identical_classical_execution_retained": (
            production["matched_classical"][
                "identical_exact_coordinate_search_available"
            ]
            and not production["matched_classical"][
                "comparison_establishes_advantage"
            ]
        ),
        "global_optimality_not_claimed": (
            not production["global_unit_lattice_optimum_established"]
            and "GLOBAL_CYCLOTOMIC_UNIT_OPTIMALITY"
            in production["not_established"]
        ),
        "distinct_phase_resource_not_claimed": (
            "DISTINCT_PHASE_RESOURCE" in production["not_established"]
        ),
        "advantage_not_claimed": (
            "COMPUTATIONAL_ADVANTAGE" in production["not_established"]
        ),
    }
    result_pass = (
        all(all(values.values()) for values in family_checks.values())
        and all(
            all(
                value
                for key, value in checked.items()
                if key not in {
                    "periods",
                    "family",
                    "exact_resource_tuple",
                    "exact_inverse_resource_tuple",
                }
            )
            for checked in case_checks
        )
        and all(restoration.values())
        and all(mutations.values())
        and all(scope.values())
    )
    result = {
        "result": "PASS" if result_pass else "FAIL",
        "experiment": (
            "SEPARATE_EXACT_LOG_FREE_49_DIRECTION_UNIT_SEARCH_ORACLE"
        ),
        "oracle_imports_production_module": False,
        "oracle_coefficient_method": (
            "SEQUENTIAL_MULTIPLICATION_BY_X_MOD_Q"
        ),
        "production_coefficient_method": (
            "BINARY_POLYNOMIAL_POWERING_MOD_Q"
        ),
        "oracle_search_method": (
            "INDEPENDENT_EXACT_DIRECTION_BRACKET_AND_DISCRETE_"
            "DIFFERENCE_BINARY_SEARCH"
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
            "LINUX_X86_64_PYTHON_TWO_PUBLIC_F17_PERIOD17_CUBIC_"
            "PATH_FAMILIES_PERIODS1AND64_EXACT_LOG_FREE_49_DIRECTION_"
            "UNIT_LINE_DESCENT_EXACT_BOUNDARY_RESOURCE_INVERSE_AND_"
            "RESTORATION_PARITY_WITH_CROSS_FAMILY_RESTORED_CARRIER_"
            "REUSE_PARITY_AT_PERIOD1_COMPONENT_LEVEL_ACCOUNTING_"
            "SOFTWARE_ONLY"
        ),
        "not_established": production["not_established"],
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
