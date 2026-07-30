#!/usr/bin/env python3
"""Exact maximal-real-subfield search for the period-17 Horner carrier.

The predecessor removed retained multi-vector inverse history, but its exact
unit search represented every real norm and every real direction norm factor
in the full 16-coordinate cyclotomic field.  This successor moves only that
search layer into the maximal real subfield

    Q(zeta_17 + zeta_17^-1),

using the integral basis

    1, s_1, ..., s_7,  where s_k = zeta_17^k + zeta_17^-k.

The phase carrier remains a full cyclotomic vector.  Its initial Hermitian
norm is streamed in the full field and converted exactly to eight real
coordinates.  Search powers, trial norms, and trace energies remain in the
real subfield.  Only one certified net unit action is materialized on the
full carrier.  The actual inverse rematerializes the same public Horner
program and restores the same backing exactly.

The identical eight-coordinate search is available to compact classical
software.  This bounded quotient diagnostic therefore does not establish a
distinct phase resource or computational advantage.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_pi_unit_horner_stream as horner


deferred = horner.prior
exact = deferred.prior
base = horner.base
cyclo = horner.cyclo
pi_content = horner.pi_content
recurrence = horner.recurrence

UNIT_RANK = horner.UNIT_RANK
UNIT_GENERATORS = exact.UNIT_GENERATORS
UNIT_GENERATOR_INVERSES = exact.UNIT_GENERATOR_INVERSES
SEARCH_DIRECTIONS = exact.SEARCH_DIRECTIONS
TESTED_PERIODS = horner.TESTED_PERIODS
MAX_COORDINATE_SWEEPS = exact.MAX_COORDINATE_SWEEPS
MAX_BRACKET_MAGNITUDE = exact.MAX_BRACKET_MAGNITUDE

RingElement = horner.RingElement
RingVector = horner.RingVector
RealElement = tuple[int, int, int, int, int, int, int, int]

ORIGINAL_HORNER_STATS_JSON = horner.stats_json


def fail(message: str) -> None:
    raise RuntimeError(message)


def real_payload_bits(value: RealElement) -> int:
    return sum(base.signed_bits(coefficient) for coefficient in value)


def full_to_real(
    value: RingElement,
    stats: "RealSubfieldStats | None" = None,
) -> RealElement:
    """Convert a canonical conjugation-fixed cyclotomic element exactly."""

    if (
        value[1] != 0
        or value[8] != value[9]
        or any(value[index] != value[17 - index] for index in range(2, 8))
    ):
        fail("cyclotomic element is not in the declared real subfield")
    high = value[8]
    result: RealElement = (
        value[0] - high,
        -high,
        *(value[index] - high for index in range(2, 8)),
    )
    if stats is not None:
        stats.full_to_real_conversions += 1
        stats.maximum_full_to_real_input_payload_bits = max(
            stats.maximum_full_to_real_input_payload_bits,
            base.element_payload_bits(value),
        )
        stats.maximum_full_to_real_output_payload_bits = max(
            stats.maximum_full_to_real_output_payload_bits,
            real_payload_bits(result),
        )
        stats.maximum_full_to_real_live_pair_payload_bits = max(
            stats.maximum_full_to_real_live_pair_payload_bits,
            base.element_payload_bits(value) + real_payload_bits(result),
        )
    return result


def real_to_full(value: RealElement) -> RingElement:
    """Embed an integral real-basis element in the canonical full basis."""

    result = [0 for _ in range(16)]
    result[0] = value[0] - value[1]
    result[1] = 0
    result[8] = result[9] = -value[1]
    for index in range(2, 8):
        result[index] = result[17 - index] = (
            value[index] - value[1]
        )
    return tuple(result)


def real_zero() -> RealElement:
    return (0, 0, 0, 0, 0, 0, 0, 0)


def real_one() -> RealElement:
    return (1, 0, 0, 0, 0, 0, 0, 0)


def real_add(left: RealElement, right: RealElement) -> RealElement:
    return tuple(
        left_value + right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


def real_s_vector(index: int) -> RealElement:
    """Return s_index in the integral real basis without a lookup table."""

    reduced = index % 17
    if reduced > 8:
        reduced = 17 - reduced
    if reduced == 0:
        return (2, 0, 0, 0, 0, 0, 0, 0)
    if reduced < 8:
        return tuple(
            1 if coordinate == reduced else 0
            for coordinate in range(8)
        )
    # 1 + s_1 + ... + s_8 = 0.
    return (-1, -1, -1, -1, -1, -1, -1, -1)


def real_multiply(
    left: RealElement,
    right: RealElement,
    stats: "RealSubfieldStats | None" = None,
) -> RealElement:
    """Multiply with s_i s_j = s_(i+j) + s_|i-j| exactly."""

    result = [0 for _ in range(8)]
    result[0] = left[0] * right[0]
    for index in range(1, 8):
        result[index] += (
            left[0] * right[index]
            + right[0] * left[index]
        )
    for left_index in range(1, 8):
        for right_index in range(1, 8):
            product = left[left_index] * right[right_index]
            if product == 0:
                continue
            for coordinate, coefficient in enumerate(
                real_s_vector(left_index + right_index)
            ):
                result[coordinate] += product * coefficient
            for coordinate, coefficient in enumerate(
                real_s_vector(abs(left_index - right_index))
            ):
                result[coordinate] += product * coefficient
    if stats is not None:
        stats.real_subfield_ring_multiplications += 1
        stats.real_subfield_coefficient_multiplications += 64
    return tuple(result)


def real_trace(value: RealElement) -> int:
    """Trace from the degree-eight real subfield to Q."""

    return 8 * value[0] - sum(value[1:])


def real_energy(
    value: RealElement,
    stats: "RealSubfieldStats",
) -> int:
    """Return the full cyclotomic trace of a real norm element."""

    stats.exact_embedding_energy_evaluations += 1
    stats.real_subfield_trace_evaluations += 1
    energy = 2 * real_trace(value)
    if energy < 0:
        fail("real-subfield trace energy became negative")
    stats.maximum_trace_energy_bits = max(
        stats.maximum_trace_energy_bits,
        max(1, energy.bit_length()),
    )
    return energy


_FULL_PREDECESSOR_TABLE_PAYLOAD_BITS = (
    exact.compiled_unit_table_payload_bits()
)
_source_direction_table = exact.DIRECTION_TABLE
_FULL_NORM_FACTOR_TABLE_PAYLOAD_BITS = sum(
    base.element_payload_bits(value)
    for entry in _source_direction_table
    for value in entry[3:5]
)
REAL_DIRECTION_TABLE = tuple(
    (
        entry[0],
        full_to_real(entry[3]),
        full_to_real(entry[4]),
    )
    for entry in _source_direction_table
)
_REAL_NORM_FACTOR_TABLE_PAYLOAD_BITS = sum(
    real_payload_bits(value)
    for entry in REAL_DIRECTION_TABLE
    for value in entry[1:3]
)
_FULL_UNIT_GENERATOR_TABLE_PAYLOAD_BITS = sum(
    base.element_payload_bits(value)
    for value in (*UNIT_GENERATORS, *UNIT_GENERATOR_INVERSES)
)
_DIRECTION_DESCRIPTOR_PAYLOAD_BITS = sum(
    sum(base.signed_bits(value) for value in direction)
    for direction in SEARCH_DIRECTIONS
)
_REAL_ACCEPTED_TABLE_PAYLOAD_BITS = (
    _FULL_UNIT_GENERATOR_TABLE_PAYLOAD_BITS
    + _DIRECTION_DESCRIPTOR_PAYLOAD_BITS
    + _REAL_NORM_FACTOR_TABLE_PAYLOAD_BITS
)
_FULL_DIRECTION_MULTIPLIER_TABLE_PAYLOAD_BITS = (
    _FULL_PREDECESSOR_TABLE_PAYLOAD_BITS
    - _FULL_UNIT_GENERATOR_TABLE_PAYLOAD_BITS
    - _FULL_NORM_FACTOR_TABLE_PAYLOAD_BITS
)
_BASE_UNIT_MOVE_NORM_FACTOR_PAYLOAD_BITS = sum(
    base.element_payload_bits(entry[3])
    for entry in base.UNIT_MOVE_TABLE
)
_COMPILER_TRANSITION_NAMED_PAYLOAD_BITS = (
    _FULL_PREDECESSOR_TABLE_PAYLOAD_BITS
    + _BASE_UNIT_MOVE_NORM_FACTOR_PAYLOAD_BITS
    + _DIRECTION_DESCRIPTOR_PAYLOAD_BITS
    + _REAL_NORM_FACTOR_TABLE_PAYLOAD_BITS
)

# The accepted search does not retain predecessor full direction multipliers
# or full norm factors.  Full unit generators remain because the final
# certified ledger action is still applied to the full phase carrier.
exact.DIRECTION_TABLE = ()
deferred.DIRECTION_TABLE = ()
base.UNIT_MOVE_TABLE = ()
del _source_direction_table


@dataclass
class RealSubfieldStats(horner.HornerStats):
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


def stats_json(
    stats: RealSubfieldStats,
    pi_stats: pi_content.PiStats,
) -> dict[str, Any]:
    result = ORIGINAL_HORNER_STATS_JSON(stats, pi_stats)
    for name in REAL_FIELDS:
        result[name] = getattr(stats, name)
    return result


def record_real_power_pair(
    result: RealElement,
    factor: RealElement,
    stats: RealSubfieldStats,
) -> None:
    result_bits = real_payload_bits(result)
    factor_bits = real_payload_bits(factor)
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
    # Preserve the inherited named-resource interface, now measured in the
    # exact eight-coordinate representation.
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


def real_power(
    value: RealElement,
    exponent: int,
    stats: RealSubfieldStats,
) -> RealElement:
    if exponent < 0:
        fail("negative real-subfield search exponent")
    result = real_one()
    factor = value
    remaining = exponent
    record_real_power_pair(result, factor, stats)
    while remaining:
        if remaining & 1:
            result = real_multiply(result, factor, stats)
            stats.search_factor_ring_multiplications += 1
            record_real_power_pair(result, factor, stats)
        remaining >>= 1
        if remaining:
            factor = real_multiply(factor, factor, stats)
            stats.search_factor_ring_multiplications += 1
            record_real_power_pair(result, factor, stats)
    return result


def real_coordinate_probe(
    current_norm: RealElement,
    direction_index: int,
    signed_delta: int,
    stats: RealSubfieldStats,
) -> tuple[int, RealElement]:
    if signed_delta == 0:
        return real_energy(current_norm, stats), current_norm
    direction = REAL_DIRECTION_TABLE[direction_index]
    factor = real_power(
        direction[1] if signed_delta > 0 else direction[2],
        abs(signed_delta),
        stats,
    )
    trial_norm = real_multiply(factor, current_norm, stats)
    stats.candidate_norm_ring_multiplications += 1
    stats.coordinate_energy_probes += 1
    stats.balance_candidate_evaluations += 1
    trial_bits = real_payload_bits(trial_norm)
    stats.maximum_real_trial_norm_payload_bits = max(
        stats.maximum_real_trial_norm_payload_bits,
        trial_bits,
    )
    stats.maximum_search_trial_norm_payload_bits = max(
        stats.maximum_search_trial_norm_payload_bits,
        trial_bits,
    )
    return real_energy(trial_norm, stats), trial_norm


def record_energy_pair(
    left: int,
    right: int,
    stats: RealSubfieldStats,
) -> None:
    stats.maximum_search_energy_scalar_pair_bits = max(
        stats.maximum_search_energy_scalar_pair_bits,
        max(1, left.bit_length()) + max(1, right.bit_length()),
    )


def real_coordinate_minimum(
    current_norm: RealElement,
    current_energy: int,
    direction_index: int,
    stats: RealSubfieldStats,
) -> tuple[int, int, RealElement]:
    """Return the exact line minimum in the real norm coordinate."""

    stats.coordinate_line_searches += 1
    positive_energy = real_coordinate_probe(
        current_norm,
        direction_index,
        1,
        stats,
    )[0]
    negative_energy = real_coordinate_probe(
        current_norm,
        direction_index,
        -1,
        stats,
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
        next_energy = real_coordinate_probe(
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
        high_energy = real_coordinate_probe(
            current_norm,
            direction_index,
            direction * high,
            stats,
        )[0]

    del high_energy
    while low + 1 < high:
        midpoint = (low + high) // 2
        midpoint_energy = real_coordinate_probe(
            current_norm,
            direction_index,
            direction * midpoint,
            stats,
        )[0]
        successor_energy = real_coordinate_probe(
            current_norm,
            direction_index,
            direction * (midpoint + 1),
            stats,
        )[0]
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
        del midpoint_energy, successor_energy

    optimum_energy, optimum_norm = real_coordinate_probe(
        current_norm,
        direction_index,
        direction * high,
        stats,
    )
    record_energy_pair(current_energy, optimum_energy, stats)
    if optimum_energy >= current_energy:
        return 0, current_energy, current_norm
    return direction * high, optimum_energy, optimum_norm


def real_deferred_balance_vector(
    vector: RingVector,
    base_ledger: list[int],
    stats: RealSubfieldStats,
) -> tuple[RingVector, list[int]]:
    """Search in eight real coordinates, then act on the full vector once."""

    if len(base_ledger) != UNIT_RANK:
        fail("unit ledger width changed")
    if cyclo.vector_is_zero(vector):
        return cyclo.zero_vector(), [0 for _ in range(UNIT_RANK)]

    stats.balance_calls += 1
    stats.deferred_balance_calls += 1
    original = list(vector)
    initial_ledger = list(base_ledger)
    ledger = list(base_ledger)
    full_norm = base.vector_norm_element(original, stats)
    current_norm = full_to_real(full_norm, stats)
    del full_norm
    stats.maximum_real_initial_norm_payload_bits = max(
        stats.maximum_real_initial_norm_payload_bits,
        real_payload_bits(current_norm),
    )
    stats.maximum_real_current_norm_payload_bits = max(
        stats.maximum_real_current_norm_payload_bits,
        real_payload_bits(current_norm),
    )
    current_energy = real_energy(current_norm, stats)
    stats.maximum_real_current_energy_bits = max(
        stats.maximum_real_current_energy_bits,
        max(1, current_energy.bit_length()),
    )

    certified = False
    for _ in range(MAX_COORDINATE_SWEEPS):
        sweep_changed = False
        for direction_index, direction_entry in enumerate(
            REAL_DIRECTION_TABLE
        ):
            move, trial_energy, trial_norm = real_coordinate_minimum(
                current_norm,
                current_energy,
                direction_index,
                stats,
            )
            if move == 0:
                continue
            for generator_index, direction_coordinate in enumerate(
                direction_entry[0]
            ):
                ledger[generator_index] += (
                    move * direction_coordinate
                )
            current_norm = trial_norm
            stats.maximum_real_current_norm_payload_bits = max(
                stats.maximum_real_current_norm_payload_bits,
                real_payload_bits(current_norm),
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
        return original, ledger

    residual_scale = base.ledger_scale(
        [-value for value in delta],
        stats,
    )
    result = base.multiply_vector(residual_scale, original)
    stats.deferred_net_residual_actions += 1
    stats.deferred_net_residual_ring_multiplications += len(result)
    stats.unit_vector_ring_multiplications += len(result)

    scale_bits = base.element_payload_bits(residual_scale)
    input_bits = base.vector_payload_bits(original)
    result_bits = base.vector_payload_bits(result)
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

    materialized_full_norm = base.vector_norm_element(result, stats)
    stats.deferred_norm_verifications += 1
    if full_to_real(materialized_full_norm, stats) != current_norm:
        fail("full carrier action changed the certified real norm")
    return result, ledger


def compiled_real_search_table_payload_bits() -> int:
    return _REAL_ACCEPTED_TABLE_PAYLOAD_BITS


def real_case_result(
    periods: int,
    block: cyclo.CompiledBlock,
) -> dict[str, Any]:
    result = horner.case_result(periods, block)
    phase_metrics = result["phase_stats"]
    search_temporary = horner.named_search_temporary_maxima_sum(
        phase_metrics
    ) + sum(
        phase_metrics[name]
        for name in (
            "maximum_full_to_real_live_pair_payload_bits",
            "maximum_real_current_norm_payload_bits",
            "maximum_real_current_energy_bits",
        )
    )
    named_total = (
        result["phase_named_checkpoint_payload_bits"]
        + compiled_real_search_table_payload_bits()
        + search_temporary
    )
    raw_payload = result["raw_horner_named_checkpoint_payload_bits"]
    result.update(
        {
            "compiled_unit_table_payload_bits": (
                compiled_real_search_table_payload_bits()
            ),
            "predecessor_full_compiled_table_payload_bits": (
                _FULL_PREDECESSOR_TABLE_PAYLOAD_BITS
            ),
            "full_norm_factor_table_payload_bits": (
                _FULL_NORM_FACTOR_TABLE_PAYLOAD_BITS
            ),
            "real_norm_factor_table_payload_bits": (
                _REAL_NORM_FACTOR_TABLE_PAYLOAD_BITS
            ),
            "compiled_table_payload_reduction_bits": (
                _FULL_PREDECESSOR_TABLE_PAYLOAD_BITS
                - compiled_real_search_table_payload_bits()
            ),
            "named_search_temporary_maxima_sum_bits": search_temporary,
            "phase_named_component_maxima_sum_bits": named_total,
            "phase_minus_raw_horner_named_payload_bits": (
                named_total - raw_payload
            ),
            "phase_named_payload_beats_raw_horner": (
                named_total < raw_payload
            ),
            "full_direction_multiplier_and_norm_table_retained": False,
            "initial_full_cyclotomic_norm_streamed": True,
            "search_norm_remains_real_until_certified_action": True,
        }
    )
    return result


def algebra_controls() -> tuple[dict[str, bool], dict[str, int]]:
    factors = [
        value
        for entry in REAL_DIRECTION_TABLE
        for value in entry[1:3]
    ]
    factor_roundtrips = all(
        full_to_real(real_to_full(value)) == value
        for value in factors
    )
    product_parity = True
    maximum_product_check_payload_bits = 0
    for left in factors:
        for right in factors:
            real_product = real_multiply(left, right)
            full_left = real_to_full(left)
            full_right = real_to_full(right)
            embedded_product = real_to_full(real_product)
            full_product = cyclo.ring_multiply(
                full_left,
                full_right,
            )
            product_parity = (
                product_parity
                and embedded_product == full_product
            )
            maximum_product_check_payload_bits = max(
                maximum_product_check_payload_bits,
                real_payload_bits(left)
                + real_payload_bits(right)
                + real_payload_bits(real_product)
                + base.element_payload_bits(full_left)
                + base.element_payload_bits(full_right)
                + base.element_payload_bits(embedded_product)
                + base.element_payload_bits(full_product),
            )
    trace_parity = all(
        2 * real_trace(value)
        == base.field_trace(real_to_full(value))
        for value in factors
    )
    controls = {
        "all_98_direction_norm_factors_in_real_subfield": (
            len(factors) == 98
        ),
        "all_direction_norm_factor_roundtrips_exact": (
            factor_roundtrips
        ),
        "all_9604_direction_norm_factor_products_exact": (
            product_parity
        ),
        "full_and_real_trace_agree_on_all_factors": trace_parity,
        "real_norm_factor_table_smaller_than_full": (
            _REAL_NORM_FACTOR_TABLE_PAYLOAD_BITS
            < _FULL_NORM_FACTOR_TABLE_PAYLOAD_BITS
        ),
        "predecessor_full_direction_table_released": (
            exact.DIRECTION_TABLE == ()
            and deferred.DIRECTION_TABLE == ()
            and base.UNIT_MOVE_TABLE == ()
        ),
        "full_unit_generators_retained_for_certified_action": (
            len(UNIT_GENERATORS) == UNIT_RANK
            and len(UNIT_GENERATOR_INVERSES) == UNIT_RANK
        ),
    }
    accounting = {
        "direction_norm_factor_roundtrip_checks": len(factors),
        "direction_norm_factor_trace_checks": len(factors),
        "all_pair_product_checks": len(factors) * len(factors),
        "real_subfield_verification_multiplications": (
            len(factors) * len(factors)
        ),
        "full_cyclotomic_verification_multiplications": (
            len(factors) * len(factors)
        ),
        "maximum_named_product_check_payload_bits": (
            maximum_product_check_payload_bits
        ),
    }
    return controls, accounting


def main() -> int:
    if len(sys.argv) != 1:
        fail(
            "usage: f17_cubic_chain_period17_"
            "real_subfield_horner.py"
        )

    horner.HornerStats = RealSubfieldStats
    horner.stats_json = stats_json
    base.BalanceStats = RealSubfieldStats
    base.stats_json = stats_json
    base.balance_vector = real_deferred_balance_vector
    base.ledger_scale = exact.tracked_ledger_scale
    base.add_balanced_vectors = deferred.relative_add_balanced_vectors
    base.project_boundary = deferred.streamed_project_boundary

    blocks = {
        family.lower(): cyclo.build_compiled_block(family)
        for family in ("PRIMARY", "REUSE")
    }
    cases = [
        real_case_result(periods, blocks[family])
        for periods in TESTED_PERIODS
        for family in ("primary", "reuse")
    ]
    restored = horner.restoration_reuse_case(
        blocks["primary"],
        blocks["reuse"],
    )
    carrier_controls = horner.controls(
        blocks["primary"],
        blocks["reuse"],
    )
    algebra, algebra_accounting = algebra_controls()
    all_phase_beats_raw = all(
        case["phase_named_payload_beats_raw_horner"]
        for case in cases
    )

    result = {
        "result": "PASS",
        "experiment": (
            "EXACT_MAXIMAL_REAL_SUBFIELD_UNIT_NORM_SEARCH_QUOTIENT_"
            "WITH_ONE_FULL_CERTIFIED_CARRIER_ACTION_AND_MATCHED_"
            "CYCLOTOMIC_AND_CLASSICAL_COST"
        ),
        "claim_candidate": (
            "BOUNDED_EXACT_MAXIMAL_REAL_SUBFIELD_QUOTIENT_REDUCES_"
            "98_UNIT_NORM_SEARCH_FACTORS_FROM16_TO8_COORDINATES_"
            "RELEASES_FULL_DIRECTION_FACTOR_RETENTION_AND_PRESERVES_"
            "SINGLE_RESIDENT_HORNER_BOUNDARIES_EXACT_RESTORATION_"
            "AND_PERIOD1_CROSS_FAMILY_REUSE"
        ),
        "classification_candidate": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level_candidate": "PACKAGE_SELF_REVIEW",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "tested_periods": list(TESTED_PERIODS),
        "declared_exact_search_direction_count": len(
            SEARCH_DIRECTIONS
        ),
        "full_cyclotomic_dimension": 16,
        "maximal_real_subfield_dimension": 8,
        "real_norm_factor_count": 2 * len(SEARCH_DIRECTIONS),
        "carrier_resident_phase_vector_count": 1,
        "retained_inverse_history_bytes": 0,
        "public_topology_compilation_answer_independent": True,
        "compiled_table_accounting": {
            "predecessor_full_table_payload_bits": (
                _FULL_PREDECESSOR_TABLE_PAYLOAD_BITS
            ),
            "predecessor_full_norm_factor_payload_bits": (
                _FULL_NORM_FACTOR_TABLE_PAYLOAD_BITS
            ),
            "predecessor_full_direction_multiplier_payload_bits": (
                _FULL_DIRECTION_MULTIPLIER_TABLE_PAYLOAD_BITS
            ),
            "predecessor_base_unit_move_norm_factor_payload_bits": (
                _BASE_UNIT_MOVE_NORM_FACTOR_PAYLOAD_BITS
            ),
            "real_norm_factor_payload_bits": (
                _REAL_NORM_FACTOR_TABLE_PAYLOAD_BITS
            ),
            "retained_full_unit_generator_payload_bits": (
                _FULL_UNIT_GENERATOR_TABLE_PAYLOAD_BITS
            ),
            "direction_descriptor_payload_bits": (
                _DIRECTION_DESCRIPTOR_PAYLOAD_BITS
            ),
            "accepted_hybrid_table_payload_bits": (
                _REAL_ACCEPTED_TABLE_PAYLOAD_BITS
            ),
            "compiler_transition_named_payload_bits": (
                _COMPILER_TRANSITION_NAMED_PAYLOAD_BITS
            ),
            "full_direction_multiplier_and_norm_table_retained": False,
        },
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
        "cases": cases,
        "all_raw_horner_boundaries_equal": all(
            case["raw_horner_boundary_equal"] for case in cases
        ),
        "all_prior_raw_recurrence_boundaries_equal": all(
            case["prior_raw_recurrence_boundary_equal"]
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
        "all_cases_coordinatewise_certified": all(
            case["phase_stats"]["coordinatewise_certified_calls"]
            == case["phase_stats"]["balance_calls"]
            and case["phase_stats"]["coordinate_sweep_cap_hits"] == 0
            and case["phase_stats"]["coordinate_bracket_cap_hits"] == 0
            for case in cases
        ),
        "all_phase_named_payloads_beat_raw_horner": (
            all_phase_beats_raw
        ),
        "restoration_reuse_case": restored,
        "carrier_controls": carrier_controls,
        "algebra_controls": algebra,
        "verification_accounting": {
            **algebra_accounting,
            "carrier_control_forward_populations": 3,
            "carrier_control_failed_inverse_attempts": 2,
            "carrier_control_reordered_inverse_attempts": 1,
            "carrier_control_raw_boundary_evaluations": 2,
            "restoration_reuse_transactions": 3,
        },
        "matched_classical": {
            "matched_raw_horner_named_checkpoint_implemented": True,
            "identical_normalized_real_subfield_horner_available": True,
            "same_public_coefficients_operator_boundary_and_search": True,
            "comparison_establishes_advantage": False,
        },
        "resource_law": {
            "initial_full_cyclotomic_autocorrelation_counted": True,
            "full_to_real_conversion_input_and_output_counted": True,
            "full_to_real_conversion_live_pair_counted": True,
            "real_direction_factors_and_descriptors_counted": True,
            "full_unit_generators_for_certified_action_counted": True,
            "compiler_full_to_real_table_transition_counted": True,
            "compiler_transition_in_warm_execution_total": False,
            "real_power_trial_norm_trace_and_energy_pairs_counted": True,
            "persistent_real_current_norm_counted": True,
            "persistent_real_current_energy_counted": True,
            "full_certified_net_action_and_norm_verification_counted": True,
            "horner_named_checkpoints_projection_and_inverse_counted": True,
            "named_component_maxima_sum_is_simultaneous_peak": False,
            "python_object_overhead_bounded": False,
            "allocator_peak_bounded": False,
            "internal_integer_multiplication_peak_bounded": False,
            "whole_process_peak_bounded": False,
            "verification_work_in_accepted_execution_total": False,
        },
        "observation": (
            "REAL_SUBFIELD_SEARCH_REMOVES_CONJUGATION_REDUNDANCY_"
            "FROM_NORM_POWERS_AND_DIRECTION_FACTORS_BUT_"
            + (
                "THE_PHASE_NAMED_TOTAL_FALLS_BELOW_RAW_WITHOUT_"
                "BEATING_THE_IDENTICAL_CLASSICAL_REAL_SUBFIELD_PATH"
                if all_phase_beats_raw
                else
                "MATCHED_RAW_OR_IDENTICAL_CLASSICAL_REAL_SUBFIELD_"
                "EXECUTION_REMAINS_THE_RESOURCE_OBSTRUCTION"
            )
        ),
        "not_established": [
            "FULL_CARRIER_STATE_IN_THE_REAL_SUBFIELD",
            "ELIMINATION_OF_INITIAL_FULL_CYCLOTOMIC_AUTOCORRELATION",
            "ELIMINATION_OF_THE_FINAL_FULL_CERTIFIED_UNIT_ACTION",
            "GLOBAL_CYCLOTOMIC_UNIT_OPTIMALITY",
            "FIXED_RESIDUAL_INTEGER_WIDTH",
            "FIXED_TOTAL_BIT_FOOTPRINT",
            "ASYMPTOTIC_RESIDUAL_HEIGHT_BOUND",
            "SIMULTANEOUS_PROCESS_PEAK_FROM_NAMED_COMPONENT_MAXIMA",
            "PERIOD64_CROSS_FAMILY_REUSE",
            "BOUNDED_REPEATED_USE_GENERATION_AND_LEASE_METADATA",
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
            "EXACT_REAL_SUBFIELD_AUTOCORRELATION_AND_ACCEPTED_ACTION_"
            "CLOSURE_OR_PHASE_NATIVE_NONCLASSICAL_TRACE_COUPLING"
        ),
        "next_obstruction": (
            "SEARCH_NORM_CONJUGATION_REDUNDANCY_IS_REMOVED_BUT_THE_"
            "INITIAL_AUTOCORRELATION_AND_FINAL_CERTIFIED_ACTION_"
            "REMAIN_FULL_CYCLOTOMIC_AND_THE_IDENTICAL_CLASSICAL_"
            "REAL_SUBFIELD_EXECUTION_REMAINS"
        ),
        "generation_and_lease_are_observed_bookkeeping_only": True,
        "generation_or_lease_enforcement_established": False,
        "terminal": False,
    }
    hard_gate = {
        "algebra": all(algebra.values()),
        "carrier_controls": all(carrier_controls.values()),
        "raw_horner_boundaries": result[
            "all_raw_horner_boundaries_equal"
        ],
        "prior_raw_boundaries": result[
            "all_prior_raw_recurrence_boundaries_equal"
        ],
        "restoration": result["all_cases_restore_exactly"],
        "coordinatewise_certified": result[
            "all_cases_coordinatewise_certified"
        ],
        "primary_reuse_restored": restored[
            "primary_restored_exactly"
        ],
        "unrelated_reuse_restored": restored[
            "reuse_restored_exactly"
        ],
        "same_original_backing": restored["same_original_backing"],
        "fresh_restored_reuse_boundary": restored[
            "fresh_restored_reuse_boundary_equal"
        ],
    }
    if not all(hard_gate.values()):
        fail(
            "real-subfield Horner qualification failed: "
            + json.dumps(hard_gate, sort_keys=True)
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
