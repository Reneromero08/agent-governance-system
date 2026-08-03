#!/usr/bin/env python3
"""Separate power-basis oracle for streamed real autocorrelation.

This oracle does not import the production streamed successor.  It extends
the predecessor's independently reconstructed degree-eight power-basis
implementation.  Each full cyclotomic Hermitian product is converted
immediately into that power basis and accumulated there, without constructing
a summed degree-sixteen norm.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_real_subfield_horner_oracle as prior


horner = prior.horner
ring = prior.ring
full = prior.prior

UNIT_RANK = prior.UNIT_RANK
MAX_DIRECTION_SWEEPS = prior.MAX_DIRECTION_SWEEPS
REAL_DIRECTION_TABLE = prior.REAL_DIRECTION_TABLE

RingVector = prior.RingVector
PowerElement = prior.PowerElement

ORIGINAL_METRICS_JSON = prior.metrics_json
ORIGINAL_NAMED_SEARCH_TEMPORARY = (
    prior.named_search_temporary_maxima_sum
)


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class StreamedPowerOracleStats(prior.RealPowerOracleStats):
    streamed_real_norm_calls: int = 0
    streamed_real_norm_input_cells: int = 0
    streamed_real_norm_terms: int = 0
    streamed_real_norm_singleton_calls: int = 0
    streamed_real_norm_full_carrier_calls: int = 0
    streamed_real_norm_unexpected_width_calls: int = 0
    streamed_real_norm_full_cyclotomic_multiplications: int = 0
    streamed_real_norm_real_additions: int = 0
    maximum_streamed_norm_full_product_payload_bits: int = 0
    maximum_streamed_norm_real_term_payload_bits: int = 0
    maximum_streamed_norm_real_accumulator_payload_bits: int = 0
    maximum_streamed_norm_conversion_live_pair_payload_bits: int = 0
    maximum_streamed_norm_addition_live_payload_bits: int = 0
    maximum_streamed_norm_named_live_payload_bits: int = 0


STREAMED_FIELDS = (
    "streamed_real_norm_calls",
    "streamed_real_norm_input_cells",
    "streamed_real_norm_terms",
    "streamed_real_norm_singleton_calls",
    "streamed_real_norm_full_carrier_calls",
    "streamed_real_norm_unexpected_width_calls",
    "streamed_real_norm_full_cyclotomic_multiplications",
    "streamed_real_norm_real_additions",
    "maximum_streamed_norm_full_product_payload_bits",
    "maximum_streamed_norm_real_term_payload_bits",
    "maximum_streamed_norm_real_accumulator_payload_bits",
    "maximum_streamed_norm_conversion_live_pair_payload_bits",
    "maximum_streamed_norm_addition_live_payload_bits",
    "maximum_streamed_norm_named_live_payload_bits",
)


def metrics_json(
    stats: StreamedPowerOracleStats,
) -> dict[str, int]:
    result = ORIGINAL_METRICS_JSON(stats)
    for name in STREAMED_FIELDS:
        result[name] = getattr(stats, name)
    return result


def power_add(
    left: PowerElement,
    right: PowerElement,
) -> PowerElement:
    return tuple(
        left_value + right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


def streamed_power_vector_norm(
    vector: RingVector,
    stats: StreamedPowerOracleStats,
) -> PowerElement:
    """Accumulate independent full products directly in the power basis."""

    accumulator = prior.power_zero()
    stats.streamed_real_norm_calls += 1
    stats.streamed_real_norm_input_cells += len(vector)
    if len(vector) == 1:
        stats.streamed_real_norm_singleton_calls += 1
    elif len(vector) == prior.reference.PRIME:
        stats.streamed_real_norm_full_carrier_calls += 1
    else:
        stats.streamed_real_norm_unexpected_width_calls += 1

    for element in vector:
        full_product = ring.ring_multiply(
            element,
            full.conjugate(element),
        )
        stats.initial_norm_element_ring_multiplications += 1
        stats.streamed_real_norm_full_cyclotomic_multiplications += 1
        full_bits = full.element_payload_bits(full_product)
        accumulator_bits = prior.power_payload_bits(accumulator)
        stats.maximum_streamed_norm_full_product_payload_bits = max(
            stats.maximum_streamed_norm_full_product_payload_bits,
            full_bits,
        )
        stats.maximum_streamed_norm_real_accumulator_payload_bits = max(
            stats.maximum_streamed_norm_real_accumulator_payload_bits,
            accumulator_bits,
        )

        real_term = prior.full_to_power(full_product, stats)
        real_term_bits = prior.power_payload_bits(real_term)
        stats.streamed_real_norm_terms += 1
        stats.maximum_streamed_norm_real_term_payload_bits = max(
            stats.maximum_streamed_norm_real_term_payload_bits,
            real_term_bits,
        )
        stats.maximum_streamed_norm_conversion_live_pair_payload_bits = max(
            stats.maximum_streamed_norm_conversion_live_pair_payload_bits,
            full_bits + real_term_bits,
        )
        conversion_live = accumulator_bits + full_bits + real_term_bits
        del full_product

        next_accumulator = power_add(accumulator, real_term)
        next_bits = prior.power_payload_bits(next_accumulator)
        addition_live = accumulator_bits + real_term_bits + next_bits
        stats.streamed_real_norm_real_additions += 1
        stats.maximum_streamed_norm_addition_live_payload_bits = max(
            stats.maximum_streamed_norm_addition_live_payload_bits,
            addition_live,
        )
        stats.maximum_streamed_norm_named_live_payload_bits = max(
            stats.maximum_streamed_norm_named_live_payload_bits,
            conversion_live,
            addition_live,
        )
        del accumulator, real_term
        accumulator = next_accumulator

    stats.maximum_streamed_norm_real_accumulator_payload_bits = max(
        stats.maximum_streamed_norm_real_accumulator_payload_bits,
        prior.power_payload_bits(accumulator),
    )
    return accumulator


def streamed_balance(
    vector: RingVector,
    base_ledger: tuple[int, ...] | list[int],
    stats: StreamedPowerOracleStats,
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
    current_norm = streamed_power_vector_norm(original, stats)
    stats.maximum_real_initial_norm_payload_bits = max(
        stats.maximum_real_initial_norm_payload_bits,
        prior.power_payload_bits(current_norm),
    )
    stats.maximum_real_current_norm_payload_bits = max(
        stats.maximum_real_current_norm_payload_bits,
        prior.power_payload_bits(current_norm),
    )
    current_energy = prior.power_energy(current_norm, stats)
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
            move, trial_energy, trial_norm = prior.coordinate_minimum(
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
                prior.power_payload_bits(current_norm),
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

    scale = full.ledger_scale(
        tuple(-value for value in delta),
        stats,
    )
    result = full.multiply_vector(scale, original)
    stats.deferred_net_residual_actions += 1
    stats.deferred_net_residual_ring_multiplications += len(result)
    stats.unit_vector_ring_multiplications += len(result)
    scale_bits = full.element_payload_bits(scale)
    input_bits = full.vector_payload_bits(original)
    result_bits = full.vector_payload_bits(result)
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

    observed_norm = streamed_power_vector_norm(result, stats)
    stats.deferred_norm_verifications += 1
    if observed_norm != current_norm:
        fail("oracle streamed norm changed after certified action")
    return result, tuple(ledger)


def named_search_temporary_maxima_sum(
    metrics: dict[str, int],
) -> int:
    return (
        horner.named_search_temporary_maxima_sum(metrics)
        + metrics["maximum_streamed_norm_named_live_payload_bits"]
        + metrics["maximum_real_current_norm_payload_bits"]
        + metrics["maximum_real_current_energy_bits"]
    )


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_chain_period17_"
            "streamed_real_autocorrelation_oracle.py "
            "PRODUCTION_RESULT"
        )

    prior.RealPowerOracleStats = StreamedPowerOracleStats
    prior.metrics_json = metrics_json
    prior.balance = streamed_balance
    prior.named_search_temporary_maxima_sum = (
        named_search_temporary_maxima_sum
    )

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        predecessor_rc = prior.main()
    if predecessor_rc != 0:
        fail("predecessor oracle rejected streamed successor")
    result = json.loads(captured.getvalue())

    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        production = json.load(handle)
    shape_checks = {
        "production_reports_no_full_aggregate_norm": (
            not production["full_cyclotomic_aggregate_norm_constructed"]
            and production["all_full_aggregate_norms_absent"]
        ),
        "production_term_counts_exact": production[
            "all_streamed_norm_term_counts_exact"
        ],
        "oracle_case_metrics_match_production": all(
            case["phase_resource_tuple_equal"]
            and case["inverse_resource_tuple_equal"]
            and case["named_search_temporary_sum_equal"]
            and case["named_component_total_equal"]
            for case in result["case_checks"]
        ),
        "only_singleton_and_17_cell_norm_shapes": all(
            case["exact_phase_resource_tuple"][
                "streamed_real_norm_unexpected_width_calls"
            ]
            == 0
            and case["exact_phase_resource_tuple"][
                "streamed_real_norm_terms"
            ]
            == case["exact_phase_resource_tuple"][
                "streamed_real_norm_input_cells"
            ]
            for case in result["case_checks"]
        ),
        "identical_streamed_classical_path_retained": (
            production["matched_classical"][
                "identical_streamed_real_subfield_norm_available"
            ]
            and not production["matched_classical"][
                "comparison_establishes_advantage"
            ]
        ),
    }
    result.update(
        {
            "result": (
                "PASS"
                if result["result"] == "PASS"
                and all(shape_checks.values())
                else "FAIL"
            ),
            "experiment": (
                "SEPARATE_POWER_BASIS_STREAMED_EXACT_REAL_SUBFIELD_"
                "AUTOCORRELATION_ORACLE"
            ),
            "oracle_imports_production_streamed_module": False,
            "oracle_full_cyclotomic_aggregate_norm_constructed": False,
            "streamed_shape_checks": shape_checks,
            "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
            "verification_level": "SEPARATE_REFERENCE_PARITY",
            "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
            "claim_ceiling": (
                "LINUX_X86_64_PYTHON_TWO_PUBLIC_F17_PERIOD17_"
                "FAMILIES_PERIODS1AND64_TERM_STREAMED_FULL_PRODUCT_"
                "TO_DEGREE8_REAL_NORM_ACCUMULATOR_NO_DEGREE16_"
                "AGGREGATE_ONE_FULL_CERTIFIED_ACTION_ONE_RESIDENT_"
                "HORNER_CARRIER_EXACT_BOUNDARY_RESOURCE_RESTORATION_"
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
