#!/usr/bin/env python3
"""Stream exact Hermitian norm terms into the period-17 real subfield.

The predecessor moved unit-norm search into the degree-eight maximal real
subfield, but it first summed all seventeen Hermitian products in the full
degree-sixteen cyclotomic representation.  This successor converts each
temporary full product exactly and accumulates it directly in the real
subfield.  No full cyclotomic aggregate norm is constructed.

The phase carrier and the one certified unit action remain full cyclotomic.
The identical streamed real-subfield recurrence is available to compact
classical software, so this bounded repair does not establish a distinct
phase resource or computational advantage.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_real_subfield_horner as prior


horner = prior.horner
base = prior.base
cyclo = prior.cyclo
pi_content = prior.pi_content

UNIT_RANK = prior.UNIT_RANK
REAL_DIRECTION_TABLE = prior.REAL_DIRECTION_TABLE
MAX_COORDINATE_SWEEPS = prior.MAX_COORDINATE_SWEEPS

RingVector = prior.RingVector
RealElement = prior.RealElement

ORIGINAL_STATS_JSON = prior.stats_json
ORIGINAL_CASE_RESULT = prior.real_case_result


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class StreamedRealStats(prior.RealSubfieldStats):
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


def stats_json(
    stats: StreamedRealStats,
    pi_stats: pi_content.PiStats,
) -> dict[str, Any]:
    result = ORIGINAL_STATS_JSON(stats, pi_stats)
    for name in STREAMED_FIELDS:
        result[name] = getattr(stats, name)
    return result


def streamed_real_vector_norm(
    vector: RingVector,
    stats: StreamedRealStats,
) -> RealElement:
    """Accumulate x*conjugate(x) directly in eight real coordinates."""

    accumulator = prior.real_zero()
    stats.streamed_real_norm_calls += 1
    stats.streamed_real_norm_input_cells += len(vector)
    if len(vector) == 1:
        stats.streamed_real_norm_singleton_calls += 1
    elif len(vector) == cyclo.PRIME:
        stats.streamed_real_norm_full_carrier_calls += 1
    else:
        stats.streamed_real_norm_unexpected_width_calls += 1
    for element in vector:
        full_product = cyclo.ring_multiply(
            element,
            base.ring_conjugate(element),
        )
        stats.initial_norm_element_ring_multiplications += 1
        stats.streamed_real_norm_full_cyclotomic_multiplications += 1
        full_bits = base.element_payload_bits(full_product)
        accumulator_bits = prior.real_payload_bits(accumulator)
        stats.maximum_streamed_norm_full_product_payload_bits = max(
            stats.maximum_streamed_norm_full_product_payload_bits,
            full_bits,
        )
        stats.maximum_streamed_norm_real_accumulator_payload_bits = max(
            stats.maximum_streamed_norm_real_accumulator_payload_bits,
            accumulator_bits,
        )

        real_term = prior.full_to_real(full_product, stats)
        real_term_bits = prior.real_payload_bits(real_term)
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

        next_accumulator = prior.real_add(accumulator, real_term)
        next_bits = prior.real_payload_bits(next_accumulator)
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
        prior.real_payload_bits(accumulator),
    )
    return accumulator


def streamed_real_balance_vector(
    vector: RingVector,
    base_ledger: list[int],
    stats: StreamedRealStats,
) -> tuple[RingVector, list[int]]:
    """Search a streamed real norm, then act on the full vector once."""

    if len(base_ledger) != UNIT_RANK:
        fail("unit ledger width changed")
    if cyclo.vector_is_zero(vector):
        return cyclo.zero_vector(), [0 for _ in range(UNIT_RANK)]

    stats.balance_calls += 1
    stats.deferred_balance_calls += 1
    original = list(vector)
    initial_ledger = list(base_ledger)
    ledger = list(base_ledger)
    current_norm = streamed_real_vector_norm(original, stats)
    stats.maximum_real_initial_norm_payload_bits = max(
        stats.maximum_real_initial_norm_payload_bits,
        prior.real_payload_bits(current_norm),
    )
    stats.maximum_real_current_norm_payload_bits = max(
        stats.maximum_real_current_norm_payload_bits,
        prior.real_payload_bits(current_norm),
    )
    current_energy = prior.real_energy(current_norm, stats)
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
            move, trial_energy, trial_norm = prior.real_coordinate_minimum(
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
                ledger[generator_index] += move * direction_coordinate
            current_norm = trial_norm
            stats.maximum_real_current_norm_payload_bits = max(
                stats.maximum_real_current_norm_payload_bits,
                prior.real_payload_bits(current_norm),
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

    observed_norm = streamed_real_vector_norm(result, stats)
    stats.deferred_norm_verifications += 1
    if observed_norm != current_norm:
        fail("streamed real norm changed after the certified action")
    return result, ledger


def streamed_case_result(
    periods: int,
    block: cyclo.CompiledBlock,
) -> dict[str, Any]:
    result = ORIGINAL_CASE_RESULT(periods, block)
    phase_metrics = result["phase_stats"]
    search_temporary = (
        horner.named_search_temporary_maxima_sum(phase_metrics)
        + phase_metrics["maximum_streamed_norm_named_live_payload_bits"]
        + phase_metrics["maximum_real_current_norm_payload_bits"]
        + phase_metrics["maximum_real_current_energy_bits"]
    )
    named_total = (
        result["phase_named_checkpoint_payload_bits"]
        + prior.compiled_real_search_table_payload_bits()
        + search_temporary
    )
    raw_payload = result["raw_horner_named_checkpoint_payload_bits"]
    result.update(
        {
            "named_search_temporary_maxima_sum_bits": search_temporary,
            "phase_named_component_maxima_sum_bits": named_total,
            "phase_minus_raw_horner_named_payload_bits": (
                named_total - raw_payload
            ),
            "phase_named_payload_beats_raw_horner": (
                named_total < raw_payload
            ),
            "initial_full_cyclotomic_norm_streamed": False,
            "full_cyclotomic_aggregate_norm_constructed": False,
            "full_cyclotomic_norm_terms_streamed_to_real": True,
            "post_action_full_cyclotomic_aggregate_norm_constructed": (
                False
            ),
        }
    )
    return result


def main() -> int:
    if len(sys.argv) != 1:
        fail(
            "usage: f17_cubic_chain_period17_"
            "streamed_real_autocorrelation.py"
        )

    prior.RealSubfieldStats = StreamedRealStats
    prior.stats_json = stats_json
    prior.real_deferred_balance_vector = streamed_real_balance_vector
    prior.real_case_result = streamed_case_result

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        predecessor_rc = prior.main()
    if predecessor_rc != 0:
        fail("predecessor driver rejected streamed successor")
    result = json.loads(captured.getvalue())

    all_phase_beats_raw = all(
        case["phase_named_payload_beats_raw_horner"]
        for case in result["cases"]
    )
    all_aggregates_absent = all(
        not case["full_cyclotomic_aggregate_norm_constructed"]
        and not case[
            "post_action_full_cyclotomic_aggregate_norm_constructed"
        ]
        and case["full_cyclotomic_norm_terms_streamed_to_real"]
        for case in result["cases"]
    )
    all_expected_term_counts = all(
        case["phase_stats"]["streamed_real_norm_terms"]
        == case["phase_stats"]["streamed_real_norm_input_cells"]
        and case["phase_stats"]["streamed_real_norm_calls"]
        == (
            case["phase_stats"]["streamed_real_norm_singleton_calls"]
            + case["phase_stats"][
                "streamed_real_norm_full_carrier_calls"
            ]
        )
        and case["phase_stats"][
            "streamed_real_norm_unexpected_width_calls"
        ]
        == 0
        for case in result["cases"]
    )

    result.update(
        {
            "experiment": (
                "STREAMED_EXACT_REAL_SUBFIELD_AUTOCORRELATION_WITHOUT_"
                "FULL_AGGREGATE_NORM_AND_MATCHED_FULL_CYCLOTOMIC_"
                "BOUNDARY"
            ),
            "claim_candidate": (
                "BOUNDED_EXACT_TERM_STREAMING_ELIMINATES_THE_FULL_"
                "CYCLOTOMIC_AGGREGATE_NORM_FROM_INITIAL_SEARCH_AND_"
                "CERTIFIED_ACTION_VERIFICATION_WHILE_PRESERVING_"
                "BOUNDARIES_EXACT_RESTORATION_AND_PERIOD1_REUSE"
            ),
            "classification_candidate": "SOURCE_AUDITED_PACKAGE_LOCAL",
            "verification_level_candidate": "PACKAGE_SELF_REVIEW",
            "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
            "full_cyclotomic_aggregate_norm_constructed": False,
            "full_cyclotomic_per_element_norm_products_remain": True,
            "all_streamed_norm_term_counts_exact": (
                all_expected_term_counts
            ),
            "all_full_aggregate_norms_absent": all_aggregates_absent,
            "all_phase_named_payloads_beat_raw_horner": (
                all_phase_beats_raw
            ),
            "matched_classical": {
                **result["matched_classical"],
                "identical_streamed_real_subfield_norm_available": True,
                "same_term_order_conversion_accumulator_and_search": True,
                "comparison_establishes_advantage": False,
            },
            "resource_law": {
                **result["resource_law"],
                "initial_full_cyclotomic_autocorrelation_counted": False,
                "full_aggregate_norm_payload_counted": False,
                "streamed_full_products_counted": True,
                "streamed_real_terms_counted": True,
                "streamed_real_accumulator_counted": True,
                "streamed_conversion_live_payload_counted": True,
                "streamed_addition_live_payload_counted": True,
                "full_to_real_conversion_live_pair_counted": True,
                "persistent_real_current_norm_counted": True,
                "whole_process_peak_bounded": False,
            },
            "observation": (
                "TERM_STREAMING_REMOVES_THE_FULL_DEGREE16_AGGREGATE_"
                "NORM_BUT_PER_ELEMENT_FULL_PRODUCTS_THE_FULL_CARRIER_"
                "AND_THE_FINAL_CERTIFIED_ACTION_REMAIN_AND_THE_"
                "IDENTICAL_COMPACT_CLASSICAL_STREAM_IS_AVAILABLE"
            ),
            "not_established": sorted(
                set(result["not_established"])
                | {
                    "ELIMINATION_OF_PER_ELEMENT_FULL_CYCLOTOMIC_PRODUCTS",
                    "FULL_REAL_SUBFIELD_CARRIER",
                    "DISTINCT_PHASE_RESOURCE",
                    "COMPUTATIONAL_ADVANTAGE",
                    "SMALL_WALL_CROSSING",
                    "MACHINE_ENFORCED_NO_SMUGGLE_OR_CATVM_CUSTODY",
                    "PHYSICAL_WAVEFORM_EXECUTION",
                    "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
                    "UNBOUNDED_COMPUTATION",
                }
                - {
                    "ELIMINATION_OF_INITIAL_FULL_CYCLOTOMIC_AUTOCORRELATION"
                }
            ),
            "next_experiment": (
                "EXACT_REAL_SUBFIELD_HERMITIAN_TERM_GENERATOR_WITHOUT_"
                "PER_ELEMENT_FULL_CYCLOTOMIC_PRODUCT_OR_PHASE_NATIVE_"
                "NONCLASSICAL_TRACE_COUPLING"
            ),
            "next_obstruction": (
                "THE_FULL_AGGREGATE_NORM_IS_GONE_BUT_EACH_HERMITIAN_"
                "TERM_THE_RESIDENT_CARRIER_AND_THE_CERTIFIED_ACTION_"
                "STILL_USE_THE_FULL_CYCLOTOMIC_REPRESENTATION_AND_"
                "COMPACT_CLASSICAL_SOFTWARE_CAN_USE_THE_SAME_STREAM"
            ),
            "terminal": False,
        }
    )

    hard_gate = {
        "predecessor_result": result["result"] == "PASS",
        "all_aggregates_absent": all_aggregates_absent,
        "term_counts": all_expected_term_counts,
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
            "streamed real autocorrelation qualification failed: "
            + json.dumps(hard_gate, sort_keys=True)
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
