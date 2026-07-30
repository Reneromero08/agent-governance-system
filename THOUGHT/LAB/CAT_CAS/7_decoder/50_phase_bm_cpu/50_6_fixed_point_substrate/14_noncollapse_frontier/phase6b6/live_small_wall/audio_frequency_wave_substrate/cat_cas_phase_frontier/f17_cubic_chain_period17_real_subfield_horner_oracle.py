#!/usr/bin/env python3
"""Separate power-basis oracle for the maximal-real-subfield Horner search.

This oracle does not import the production real-subfield successor.  It uses
the independently reconstructed public operators and Horner carrier from the
predecessor oracle, but represents the real search field as Q[y] modulo

    y^8 + y^7 - 7y^6 - 6y^5 + 15y^4 + 10y^3 - 10y^2 - 4y + 1,

where y = zeta_17 + zeta_17^-1.  Production instead multiplies in the integral
basis (1, s_1, ..., s_7).  Payload is converted exactly to that declared basis
only for parity with the production accounting convention.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_pi_unit_horner_stream_oracle as horner


phase = horner.phase
exact = phase.exact
prior = horner.prior
ring = horner.ring
reference = horner.reference

UNIT_RANK = horner.UNIT_RANK
EXPECTED_PERIODS = horner.EXPECTED_PERIODS
MAX_DIRECTION_SWEEPS = exact.MAX_DIRECTION_SWEEPS
MAX_BRACKET_MAGNITUDE = exact.MAX_BRACKET_MAGNITUDE

RingElement = horner.RingElement
RingVector = horner.RingVector
PowerElement = tuple[int, int, int, int, int, int, int, int]

MINIMAL_POLYNOMIAL = (1, -4, -10, 10, 15, -6, -7, 1, 1)
POWER_REDUCTION = (-1, 4, 10, -10, -15, 6, 7, -1)
POWER_TRACE_WEIGHTS = (8, -1, 15, -4, 43, -16, 138, -64)

ORIGINAL_METRICS_JSON = horner.metrics_json


def fail(message: str) -> None:
    raise RuntimeError(message)


def s_multiply_by_y(
    value: PowerElement,
) -> PowerElement:
    """Multiply an integral (1,s1,...,s7) vector by y=s1."""

    result = [0 for _ in range(8)]
    result[1] += value[0]
    for index in range(1, 8):
        coefficient = value[index]
        if index == 1:
            result[0] += 2 * coefficient
            result[2] += coefficient
        elif index < 7:
            result[index - 1] += coefficient
            result[index + 1] += coefficient
        else:
            result[6] += coefficient
            for coordinate in range(8):
                result[coordinate] -= coefficient
    return tuple(result)


POWER_TO_S = []
_power_in_s: PowerElement = (1, 0, 0, 0, 0, 0, 0, 0)
for _degree in range(8):
    POWER_TO_S.append(_power_in_s)
    _power_in_s = s_multiply_by_y(_power_in_s)
POWER_TO_S = tuple(POWER_TO_S)
del _power_in_s, _degree


def power_multiply_untracked(
    left: PowerElement,
    right: PowerElement,
) -> PowerElement:
    coefficients = [0 for _ in range(15)]
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            coefficients[left_index + right_index] += (
                left_value * right_value
            )
    for degree in range(14, 7, -1):
        leading = coefficients[degree]
        for index, reduction in enumerate(POWER_REDUCTION):
            coefficients[degree - 8 + index] += leading * reduction
    return tuple(coefficients[:8])


S_TO_POWER = [(1, 0, 0, 0, 0, 0, 0, 0)]
S_TO_POWER.append((0, 1, 0, 0, 0, 0, 0, 0))
for index in range(1, 7):
    advanced = power_multiply_untracked(
        (0, 1, 0, 0, 0, 0, 0, 0),
        S_TO_POWER[index],
    )
    predecessor = (
        (2, 0, 0, 0, 0, 0, 0, 0)
        if index == 1
        else S_TO_POWER[index - 1]
    )
    S_TO_POWER.append(
        tuple(
            actual - prior_value
            for actual, prior_value in zip(
                advanced,
                predecessor,
                strict=True,
            )
        )
    )
S_TO_POWER = tuple(S_TO_POWER)
del index, advanced, predecessor


def power_to_s(value: PowerElement) -> PowerElement:
    return tuple(
        sum(
            value[power] * POWER_TO_S[power][coordinate]
            for power in range(8)
        )
        for coordinate in range(8)
    )


def s_to_power(value: PowerElement) -> PowerElement:
    return tuple(
        sum(
            value[s_index] * S_TO_POWER[s_index][power]
            for s_index in range(8)
        )
        for power in range(8)
    )


def s_payload_bits(value: PowerElement) -> int:
    return sum(prior.signed_bits(coefficient) for coefficient in value)


def power_payload_bits(value: PowerElement) -> int:
    return s_payload_bits(power_to_s(value))


def full_to_s(value: RingElement) -> PowerElement:
    if (
        value[1] != 0
        or value[8] != value[9]
        or any(value[index] != value[17 - index] for index in range(2, 8))
    ):
        fail("oracle full element is outside the declared real subfield")
    high = value[8]
    return (
        value[0] - high,
        -high,
        *(value[index] - high for index in range(2, 8)),
    )


def s_to_full(value: PowerElement) -> RingElement:
    result = [0 for _ in range(16)]
    result[0] = value[0] - value[1]
    result[1] = 0
    result[8] = result[9] = -value[1]
    for index in range(2, 8):
        result[index] = result[17 - index] = (
            value[index] - value[1]
        )
    return tuple(result)


@dataclass
class RealPowerOracleStats(horner.HornerOracleStats):
    real_subfield_ring_multiplications: int = 0
    real_subfield_coefficient_multiplications: int = 0
    real_subfield_trace_evaluations: int = 0
    full_to_real_conversions: int = 0
    maximum_full_to_real_input_payload_bits: int = 0
    maximum_full_to_real_output_payload_bits: int = 0
    maximum_full_to_real_live_pair_payload_bits: int = 0
    maximum_real_initial_norm_payload_bits: int = 0
    maximum_real_current_norm_payload_bits: int = 0
    maximum_real_current_energy_bits: int = 0
    maximum_real_search_power_result_payload_bits: int = 0
    maximum_real_search_power_factor_payload_bits: int = 0
    maximum_real_search_power_pair_payload_bits: int = 0
    maximum_real_trial_norm_payload_bits: int = 0


REAL_FIELDS = (
    "real_subfield_ring_multiplications",
    "real_subfield_coefficient_multiplications",
    "real_subfield_trace_evaluations",
    "full_to_real_conversions",
    "maximum_full_to_real_input_payload_bits",
    "maximum_full_to_real_output_payload_bits",
    "maximum_full_to_real_live_pair_payload_bits",
    "maximum_real_initial_norm_payload_bits",
    "maximum_real_current_norm_payload_bits",
    "maximum_real_current_energy_bits",
    "maximum_real_search_power_result_payload_bits",
    "maximum_real_search_power_factor_payload_bits",
    "maximum_real_search_power_pair_payload_bits",
    "maximum_real_trial_norm_payload_bits",
)


def metrics_json(stats: RealPowerOracleStats) -> dict[str, int]:
    result = ORIGINAL_METRICS_JSON(stats)
    for name in REAL_FIELDS:
        result[name] = getattr(stats, name)
    return result


def full_to_power(
    value: RingElement,
    stats: RealPowerOracleStats | None = None,
) -> PowerElement:
    result = s_to_power(full_to_s(value))
    if stats is not None:
        input_bits = prior.element_payload_bits(value)
        output_bits = power_payload_bits(result)
        stats.full_to_real_conversions += 1
        stats.maximum_full_to_real_input_payload_bits = max(
            stats.maximum_full_to_real_input_payload_bits,
            input_bits,
        )
        stats.maximum_full_to_real_output_payload_bits = max(
            stats.maximum_full_to_real_output_payload_bits,
            output_bits,
        )
        stats.maximum_full_to_real_live_pair_payload_bits = max(
            stats.maximum_full_to_real_live_pair_payload_bits,
            input_bits + output_bits,
        )
    return result


def power_to_full(value: PowerElement) -> RingElement:
    return s_to_full(power_to_s(value))


def power_zero() -> PowerElement:
    return (0, 0, 0, 0, 0, 0, 0, 0)


def power_one() -> PowerElement:
    return (1, 0, 0, 0, 0, 0, 0, 0)


def power_multiply(
    left: PowerElement,
    right: PowerElement,
    stats: RealPowerOracleStats,
) -> PowerElement:
    result = power_multiply_untracked(left, right)
    stats.real_subfield_ring_multiplications += 1
    stats.real_subfield_coefficient_multiplications += 64
    return result


def power_trace(value: PowerElement) -> int:
    return sum(
        coefficient * weight
        for coefficient, weight in zip(
            value,
            POWER_TRACE_WEIGHTS,
            strict=True,
        )
    )


def power_energy(
    value: PowerElement,
    stats: RealPowerOracleStats,
) -> int:
    stats.exact_embedding_energy_evaluations += 1
    stats.real_subfield_trace_evaluations += 1
    energy = 2 * power_trace(value)
    if energy < 0:
        fail("oracle real-subfield trace energy became negative")
    stats.maximum_trace_energy_bits = max(
        stats.maximum_trace_energy_bits,
        max(1, energy.bit_length()),
    )
    return energy


FULL_DIRECTION_TABLE = exact.DIRECTION_TABLE
REAL_DIRECTION_TABLE = tuple(
    (
        entry[0],
        full_to_power(entry[3]),
        full_to_power(entry[4]),
    )
    for entry in FULL_DIRECTION_TABLE
)

FULL_PREDECESSOR_TABLE_PAYLOAD_BITS = (
    exact.compiled_unit_table_payload_bits()
)
FULL_NORM_FACTOR_TABLE_PAYLOAD_BITS = sum(
    prior.element_payload_bits(value)
    for entry in FULL_DIRECTION_TABLE
    for value in entry[3:5]
)
REAL_NORM_FACTOR_TABLE_PAYLOAD_BITS = sum(
    power_payload_bits(value)
    for entry in REAL_DIRECTION_TABLE
    for value in entry[1:3]
)
FULL_UNIT_GENERATOR_TABLE_PAYLOAD_BITS = sum(
    prior.element_payload_bits(value)
    for value in (
        *ring.UNIT_GENERATORS,
        *ring.UNIT_GENERATOR_INVERSES,
    )
)
DIRECTION_DESCRIPTOR_PAYLOAD_BITS = sum(
    sum(prior.signed_bits(value) for value in direction)
    for direction in exact.SEARCH_DIRECTIONS
)
ACCEPTED_TABLE_PAYLOAD_BITS = (
    FULL_UNIT_GENERATOR_TABLE_PAYLOAD_BITS
    + DIRECTION_DESCRIPTOR_PAYLOAD_BITS
    + REAL_NORM_FACTOR_TABLE_PAYLOAD_BITS
)
FULL_DIRECTION_MULTIPLIER_TABLE_PAYLOAD_BITS = (
    FULL_PREDECESSOR_TABLE_PAYLOAD_BITS
    - FULL_UNIT_GENERATOR_TABLE_PAYLOAD_BITS
    - FULL_NORM_FACTOR_TABLE_PAYLOAD_BITS
)
BASE_UNIT_MOVE_NORM_FACTOR_PAYLOAD_BITS = sum(
    prior.element_payload_bits(entry[3])
    for entry in prior.UNIT_MOVES
)
COMPILER_TRANSITION_NAMED_PAYLOAD_BITS = (
    FULL_PREDECESSOR_TABLE_PAYLOAD_BITS
    + BASE_UNIT_MOVE_NORM_FACTOR_PAYLOAD_BITS
    + DIRECTION_DESCRIPTOR_PAYLOAD_BITS
    + REAL_NORM_FACTOR_TABLE_PAYLOAD_BITS
)


def record_power_pair(
    result: PowerElement,
    factor: PowerElement,
    stats: RealPowerOracleStats,
) -> None:
    result_bits = power_payload_bits(result)
    factor_bits = power_payload_bits(factor)
    pair_bits = result_bits + factor_bits
    stats.maximum_real_search_power_result_payload_bits = max(
        stats.maximum_real_search_power_result_payload_bits,
        result_bits,
    )
    stats.maximum_real_search_power_factor_payload_bits = max(
        stats.maximum_real_search_power_factor_payload_bits,
        factor_bits,
    )
    stats.maximum_real_search_power_pair_payload_bits = max(
        stats.maximum_real_search_power_pair_payload_bits,
        pair_bits,
    )
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
        pair_bits,
    )


def power(
    value: PowerElement,
    exponent: int,
    stats: RealPowerOracleStats,
) -> PowerElement:
    if exponent < 0:
        fail("negative oracle real-subfield exponent")
    result = power_one()
    factor = value
    remaining = exponent
    record_power_pair(result, factor, stats)
    while remaining:
        if remaining & 1:
            result = power_multiply(result, factor, stats)
            stats.search_factor_ring_multiplications += 1
            record_power_pair(result, factor, stats)
        remaining >>= 1
        if remaining:
            factor = power_multiply(factor, factor, stats)
            stats.search_factor_ring_multiplications += 1
            record_power_pair(result, factor, stats)
    return result


def coordinate_probe(
    current_norm: PowerElement,
    direction_index: int,
    signed_delta: int,
    stats: RealPowerOracleStats,
) -> tuple[int, PowerElement]:
    if signed_delta == 0:
        return power_energy(current_norm, stats), current_norm
    direction = REAL_DIRECTION_TABLE[direction_index]
    factor = power(
        direction[1] if signed_delta > 0 else direction[2],
        abs(signed_delta),
        stats,
    )
    trial_norm = power_multiply(factor, current_norm, stats)
    stats.candidate_norm_ring_multiplications += 1
    stats.coordinate_energy_probes += 1
    stats.balance_candidate_evaluations += 1
    trial_bits = power_payload_bits(trial_norm)
    stats.maximum_real_trial_norm_payload_bits = max(
        stats.maximum_real_trial_norm_payload_bits,
        trial_bits,
    )
    stats.maximum_search_trial_norm_payload_bits = max(
        stats.maximum_search_trial_norm_payload_bits,
        trial_bits,
    )
    return power_energy(trial_norm, stats), trial_norm


def record_energy_pair(
    left: int,
    right: int,
    stats: RealPowerOracleStats,
) -> None:
    stats.maximum_search_energy_scalar_pair_bits = max(
        stats.maximum_search_energy_scalar_pair_bits,
        max(1, left.bit_length()) + max(1, right.bit_length()),
    )


def coordinate_minimum(
    current_norm: PowerElement,
    current_energy: int,
    direction_index: int,
    stats: RealPowerOracleStats,
) -> tuple[int, int, PowerElement]:
    stats.coordinate_line_searches += 1
    positive_energy = coordinate_probe(
        current_norm, direction_index, 1, stats
    )[0]
    negative_energy = coordinate_probe(
        current_norm, direction_index, -1, stats
    )[0]
    record_energy_pair(positive_energy, negative_energy, stats)
    directional = min(
        (
            (positive_energy, 1),
            (negative_energy, -1),
        ),
        key=lambda item: (item[0], item[1]),
    )
    record_energy_pair(current_energy, directional[0], stats)
    if directional[0] >= current_energy:
        return 0, current_energy, current_norm
    direction = directional[1]
    high = 1
    high_energy = directional[0]
    low = 0
    del positive_energy, negative_energy, directional
    while True:
        next_energy = coordinate_probe(
            current_norm,
            direction_index,
            direction * (high + 1),
            stats,
        )[0]
        record_energy_pair(high_energy, next_energy, stats)
        if next_energy >= high_energy:
            del next_energy
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
        del next_energy
        high_energy = coordinate_probe(
            current_norm,
            direction_index,
            direction * high,
            stats,
        )[0]
    del high_energy
    while low + 1 < high:
        midpoint = (low + high) // 2
        midpoint_energy = coordinate_probe(
            current_norm,
            direction_index,
            direction * midpoint,
            stats,
        )[0]
        successor_energy = coordinate_probe(
            current_norm,
            direction_index,
            direction * (midpoint + 1),
            stats,
        )[0]
        record_energy_pair(midpoint_energy, successor_energy, stats)
        stats.coordinate_binary_search_steps += 1
        if successor_energy < midpoint_energy:
            low = midpoint
        else:
            high = midpoint
        del midpoint_energy, successor_energy
    optimum_energy, optimum_norm = coordinate_probe(
        current_norm,
        direction_index,
        direction * high,
        stats,
    )
    record_energy_pair(current_energy, optimum_energy, stats)
    if optimum_energy >= current_energy:
        return 0, current_energy, current_norm
    return direction * high, optimum_energy, optimum_norm


def balance(
    vector: RingVector,
    base_ledger: tuple[int, ...] | list[int],
    stats: RealPowerOracleStats,
) -> tuple[RingVector, tuple[int, ...]]:
    if len(base_ledger) != UNIT_RANK:
        fail("oracle unit ledger width changed")
    if all(element == ring.ring_zero() for element in vector):
        return (
            [ring.ring_zero() for _ in vector],
            tuple(0 for _ in range(UNIT_RANK)),
        )
    stats.balance_calls += 1
    stats.deferred_balance_calls += 1
    original = list(vector)
    initial_ledger = list(base_ledger)
    ledger = list(base_ledger)
    full_norm = prior.norm_element(original, stats)
    current_norm = full_to_power(full_norm, stats)
    del full_norm
    stats.maximum_real_initial_norm_payload_bits = max(
        stats.maximum_real_initial_norm_payload_bits,
        power_payload_bits(current_norm),
    )
    stats.maximum_real_current_norm_payload_bits = max(
        stats.maximum_real_current_norm_payload_bits,
        power_payload_bits(current_norm),
    )
    current_energy = power_energy(current_norm, stats)
    stats.maximum_real_current_energy_bits = max(
        stats.maximum_real_current_energy_bits,
        max(1, current_energy.bit_length()),
    )
    certified = False
    for _ in range(MAX_DIRECTION_SWEEPS):
        sweep_changed = False
        for direction_index, direction_entry in enumerate(
            REAL_DIRECTION_TABLE
        ):
            move, trial_energy, trial_norm = coordinate_minimum(
                current_norm,
                current_energy,
                direction_index,
                stats,
            )
            if move == 0:
                continue
            for generator_index, coordinate in enumerate(
                direction_entry[0]
            ):
                ledger[generator_index] += move * coordinate
            current_norm = trial_norm
            stats.maximum_real_current_norm_payload_bits = max(
                stats.maximum_real_current_norm_payload_bits,
                power_payload_bits(current_norm),
            )
            current_energy = trial_energy
            stats.maximum_real_current_energy_bits = max(
                stats.maximum_real_current_energy_bits,
                max(1, current_energy.bit_length()),
            )
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
            certified = True
            break
    if not certified:
        stats.coordinate_sweep_cap_hits += 1
        stats.balance_step_cap_hits += 1
    delta = [
        actual - initial
        for actual, initial in zip(
            ledger,
            initial_ledger,
            strict=True,
        )
    ]
    if not any(delta):
        return original, tuple(ledger)
    scale = prior.ledger_scale(
        tuple(-value for value in delta),
        stats,
    )
    result = prior.multiply_vector(scale, original)
    stats.deferred_net_residual_actions += 1
    stats.deferred_net_residual_ring_multiplications += len(result)
    stats.unit_vector_ring_multiplications += len(result)
    scale_bits = prior.element_payload_bits(scale)
    input_bits = prior.vector_payload_bits(original)
    result_bits = prior.vector_payload_bits(result)
    stats.maximum_deferred_net_scale_payload_bits = max(
        stats.maximum_deferred_net_scale_payload_bits,
        scale_bits,
    )
    stats.maximum_deferred_net_vector_payload_bits = max(
        stats.maximum_deferred_net_vector_payload_bits,
        result_bits,
    )
    stats.maximum_deferred_net_live_payload_bits = max(
        stats.maximum_deferred_net_live_payload_bits,
        scale_bits + input_bits + result_bits,
    )
    observed_norm = prior.norm_element(result, stats)
    stats.deferred_norm_verifications += 1
    if full_to_power(observed_norm, stats) != current_norm:
        fail("oracle full action changed the certified real norm")
    return result, tuple(ledger)


def named_search_temporary_maxima_sum(metrics: dict[str, int]) -> int:
    return (
        horner.named_search_temporary_maxima_sum(metrics)
        + metrics["maximum_full_to_real_live_pair_payload_bits"]
        + metrics["maximum_real_current_norm_payload_bits"]
        + metrics["maximum_real_current_energy_bits"]
    )


def case_check(
    case: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    stats = RealPowerOracleStats()
    output = horner.build_horner_output(context, case["periods"], stats)
    resident_payload = horner.carrier_payload_bits(output)
    stats.maximum_carrier_resident_payload_bits = resident_payload
    stats.maximum_resident_payload_bits = resident_payload
    boundary = prior.project(output, stats)
    stats.maximum_projection_resident_plus_work_payload_bits = (
        resident_payload
        + stats.maximum_streamed_projection_live_payload_bits
    )
    metrics = metrics_json(stats)
    inverse_stats = RealPowerOracleStats()
    inverse_output = horner.build_horner_output(
        context,
        case["periods"],
        inverse_stats,
    )
    inverse_stats.maximum_inverse_resident_plus_work_payload_bits = (
        resident_payload
        + inverse_stats.maximum_horner_named_checkpoint_payload_bits
    )
    restored_zero_payload = (
        prior.vector_payload_bits(
            [ring.ring_zero() for _ in range(horner.PRIME)]
        )
        + prior.signed_bits(0)
        + prior.ledger_payload_bits(
            tuple(0 for _ in range(UNIT_RANK))
        )
    )
    inverse_stats.maximum_carrier_resident_payload_bits = (
        restored_zero_payload
    )
    inverse_stats.maximum_resident_payload_bits = restored_zero_payload
    inverse_metrics = metrics_json(inverse_stats)
    raw_boundary, raw_stats = horner.raw_horner_boundary(
        context,
        case["periods"],
    )
    raw_values = horner.raw_metrics(raw_stats)
    phase_named_checkpoint = max(
        metrics["maximum_horner_named_checkpoint_payload_bits"],
        metrics["maximum_projection_resident_plus_work_payload_bits"],
        inverse_metrics["maximum_inverse_resident_plus_work_payload_bits"],
    )
    search_temporary = named_search_temporary_maxima_sum(metrics)
    named_total = (
        phase_named_checkpoint
        + ACCEPTED_TABLE_PAYLOAD_BITS
        + search_temporary
    )
    production_metrics = case["phase_stats"]
    production_inverse = case["inverse_rematerialization_stats"]
    return {
        "periods": case["periods"],
        "family": case["family"],
        "boundary_sha256_equal": (
            hashlib.sha256(reference.encoded(boundary)).hexdigest()
            == case["boundary_sha256"]
        ),
        "raw_horner_boundary_sha256_equal": (
            hashlib.sha256(reference.encoded(raw_boundary)).hexdigest()
            == case["raw_horner_boundary_sha256"]
        ),
        "phase_resource_tuple_equal": all(
            production_metrics[key] == value
            for key, value in metrics.items()
        ),
        "inverse_resource_tuple_equal": all(
            production_inverse[key] == value
            for key, value in inverse_metrics.items()
        ),
        "raw_horner_resource_tuple_equal": (
            raw_values == case["raw_horner_stats"]
        ),
        "phase_named_checkpoint_equal": (
            phase_named_checkpoint
            == case["phase_named_checkpoint_payload_bits"]
        ),
        "named_search_temporary_sum_equal": (
            search_temporary
            == case["named_search_temporary_maxima_sum_bits"]
        ),
        "named_component_total_equal": (
            named_total
            == case["phase_named_component_maxima_sum_bits"]
        ),
        "inverse_output_exactly_equal": inverse_output == output,
        "exact_phase_resource_tuple": metrics,
        "exact_inverse_resource_tuple": inverse_metrics,
        "exact_raw_horner_resource_tuple": raw_values,
    }


def algebra_checks() -> dict[str, bool]:
    full_factors = [
        value
        for entry in FULL_DIRECTION_TABLE
        for value in entry[3:5]
    ]
    power_factors = [
        value
        for entry in REAL_DIRECTION_TABLE
        for value in entry[1:3]
    ]
    return {
        "minimal_polynomial_declared": (
            MINIMAL_POLYNOMIAL
            == (1, -4, -10, 10, 15, -6, -7, 1, 1)
        ),
        "all_98_factor_roundtrips_exact": all(
            power_to_full(power_value) == full_value
            for power_value, full_value in zip(
                power_factors,
                full_factors,
                strict=True,
            )
        ),
        "all_98_trace_values_exact": all(
            2 * power_trace(power_value)
            == prior.field_trace(full_value)
            for power_value, full_value in zip(
                power_factors,
                full_factors,
                strict=True,
            )
        ),
        "all_9604_products_exact": all(
            power_to_full(power_multiply_untracked(left, right))
            == ring.ring_multiply(full_left, full_right)
            for left, full_left in zip(
                power_factors,
                full_factors,
                strict=True,
            )
            for right, full_right in zip(
                power_factors,
                full_factors,
                strict=True,
            )
        ),
    }


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_chain_period17_"
            "real_subfield_horner_oracle.py PRODUCTION_RESULT"
        )
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        production = json.load(handle)
    if tuple(production["tested_periods"]) != EXPECTED_PERIODS:
        fail("oracle tested periods changed")

    horner.HornerOracleStats = RealPowerOracleStats
    prior.OracleStats = RealPowerOracleStats
    prior.balance = balance
    prior.add_vectors = phase.add_vectors
    prior.project = phase.project
    prior.record_metrics = phase.record_metrics
    prior.ledger_scale = exact.tracked_ledger_scale

    contexts: dict[str, dict[str, Any]] = {}
    family_checks: dict[str, dict[str, bool]] = {}
    for family in ("primary", "reuse"):
        checks, context = prior.family_context(
            family,
            production["block_certificates"][family],
        )
        family_checks[family] = checks
        contexts[family] = context

    case_checks = [
        case_check(case, contexts[case["family"].lower()])
        for case in production["cases"]
    ]
    restoration = horner.restoration_check(contexts, production)
    mutations = horner.mutation_check(contexts["primary"])
    algebra = algebra_checks()
    expected_table = {
        "predecessor_full_table_payload_bits": (
            FULL_PREDECESSOR_TABLE_PAYLOAD_BITS
        ),
        "predecessor_full_norm_factor_payload_bits": (
            FULL_NORM_FACTOR_TABLE_PAYLOAD_BITS
        ),
        "predecessor_full_direction_multiplier_payload_bits": (
            FULL_DIRECTION_MULTIPLIER_TABLE_PAYLOAD_BITS
        ),
        "predecessor_base_unit_move_norm_factor_payload_bits": (
            BASE_UNIT_MOVE_NORM_FACTOR_PAYLOAD_BITS
        ),
        "real_norm_factor_payload_bits": (
            REAL_NORM_FACTOR_TABLE_PAYLOAD_BITS
        ),
        "retained_full_unit_generator_payload_bits": (
            FULL_UNIT_GENERATOR_TABLE_PAYLOAD_BITS
        ),
        "direction_descriptor_payload_bits": (
            DIRECTION_DESCRIPTOR_PAYLOAD_BITS
        ),
        "accepted_hybrid_table_payload_bits": (
            ACCEPTED_TABLE_PAYLOAD_BITS
        ),
        "compiler_transition_named_payload_bits": (
            COMPILER_TRANSITION_NAMED_PAYLOAD_BITS
        ),
        "full_direction_multiplier_and_norm_table_retained": False,
    }
    table_checks = {
        "compiled_table_tuple_equal": (
            production["compiled_table_accounting"] == expected_table
        ),
        "real_norm_factor_table_is_2679_bits": (
            REAL_NORM_FACTOR_TABLE_PAYLOAD_BITS == 2679
        ),
        "accepted_hybrid_table_is_3407_bits": (
            ACCEPTED_TABLE_PAYLOAD_BITS == 3407
        ),
        "compiler_transition_is_11861_bits": (
            COMPILER_TRANSITION_NAMED_PAYLOAD_BITS == 11861
        ),
    }
    scope = {
        "production_result_pass": production["result"] == "PASS",
        "one_resident_phase_vector_asserted": (
            production["carrier_resident_phase_vector_count"] == 1
        ),
        "all_boundaries_equal": (
            production["all_raw_horner_boundaries_equal"]
            and production["all_prior_raw_recurrence_boundaries_equal"]
        ),
        "all_restored": production["all_cases_restore_exactly"],
        "matched_raw_horner_retained": (
            production["matched_classical"][
                "matched_raw_horner_named_checkpoint_implemented"
            ]
        ),
        "identical_real_subfield_execution_retained": (
            production["matched_classical"][
                "identical_normalized_real_subfield_horner_available"
            ]
            and not production["matched_classical"][
                "comparison_establishes_advantage"
            ]
        ),
        "all_phase_named_totals_remain_above_raw": (
            not production["all_phase_named_payloads_beat_raw_horner"]
            and all(
                case["phase_minus_raw_horner_named_payload_bits"] > 0
                for case in production["cases"]
            )
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
                    "exact_phase_resource_tuple",
                    "exact_inverse_resource_tuple",
                    "exact_raw_horner_resource_tuple",
                }
            )
            for checked in case_checks
        )
        and all(restoration.values())
        and all(mutations.values())
        and all(algebra.values())
        and all(table_checks.values())
        and all(scope.values())
    )
    result = {
        "result": "PASS" if result_pass else "FAIL",
        "experiment": (
            "SEPARATE_EXACT_POWER_BASIS_MAXIMAL_REAL_SUBFIELD_"
            "HORNER_SEARCH_ORACLE"
        ),
        "oracle_imports_production_module": False,
        "oracle_real_field_representation": (
            "POWER_BASIS_QY_MOD_DEGREE8_MINIMAL_POLYNOMIAL"
        ),
        "production_real_field_representation": (
            "INTEGRAL_BASIS_1_S1_THROUGH_S7"
        ),
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
        "algebra_checks": algebra,
        "compiled_table_checks": table_checks,
        "production_scope_checks": scope,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": (
            "LINUX_X86_64_PYTHON_TWO_PUBLIC_F17_PERIOD17_FAMILIES_"
            "PERIODS1AND64_DEGREE8_REAL_SUBFIELD_UNIT_NORM_SEARCH_"
            "ONE_FULL_CERTIFIED_ACTION_ONE_RESIDENT_HORNER_CARRIER_"
            "EXACT_BOUNDARY_NAMED_RESOURCE_RESTORATION_AND_PERIOD1_"
            "CROSS_FAMILY_REUSE_PARITY_SOFTWARE_ONLY"
        ),
        "not_established": production["not_established"],
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
