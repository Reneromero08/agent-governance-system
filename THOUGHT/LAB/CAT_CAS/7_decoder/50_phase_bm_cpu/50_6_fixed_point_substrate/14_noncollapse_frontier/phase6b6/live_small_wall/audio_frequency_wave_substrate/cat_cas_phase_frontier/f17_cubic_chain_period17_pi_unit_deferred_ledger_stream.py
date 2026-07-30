#!/usr/bin/env python3
"""Deferred exact unit-ledger action for the pi-factored F17 carrier.

The predecessor's exact 49-direction search mutated every 17-entry residual
vector after every accepted coordinate move.  It also materialized both
absolute unit ledgers before every vector addition and materialized the whole
output vector before projecting one boundary element.  Those choices made
search history part of the measured period-64 work even though unit actions
commute and projection is linear.

This successor leaves the exact search and its local-minimum certificate
unchanged while:

* accumulating accepted moves only in the exact unit ledger and norm;
* applying one net residual multiplier after search convergence;
* aligning additions with one relative unit multiplier; and
* projecting the residual first, then materializing one scalar unit action.

The mechanism is exact and descriptor-driven for the same two public F17
period-17 families.  It is a bounded materialization-law diagnostic.  The
identical exact recurrence and deferred-ledger implementation remain
available to compact classical software.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_pi_unit_exact_coordinate_descent as prior


base = prior.base
cyclo = prior.cyclo
pi_content = prior.pi_content

UNIT_RANK = prior.UNIT_RANK
TESTED_PERIODS = prior.TESTED_PERIODS
MAX_COORDINATE_SWEEPS = prior.MAX_COORDINATE_SWEEPS
MAX_BRACKET_MAGNITUDE = prior.MAX_BRACKET_MAGNITUDE
DIRECTION_TABLE = prior.DIRECTION_TABLE
SEARCH_DIRECTIONS = prior.SEARCH_DIRECTIONS

RingElement = prior.RingElement
RingVector = prior.RingVector

ORIGINAL_ADD_BALANCED_VECTORS = base.add_balanced_vectors
ORIGINAL_PROJECT_BOUNDARY = base.project_boundary

PREDECESSOR_NAMED_COMPONENT_MAXIMA_SUM_BITS = {
    ("PRIMARY", 1): 1_163_011,
    ("REUSE", 1): 1_109_835,
    ("PRIMARY", 64): 8_479_798,
    ("REUSE", 64): 8_512_822,
}


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class DeferredStats(prior.ExactSearchStats):
    deferred_balance_calls: int = 0
    deferred_net_residual_actions: int = 0
    deferred_net_residual_ring_multiplications: int = 0
    deferred_norm_verifications: int = 0
    relative_unit_alignment_calls: int = 0
    relative_unit_alignment_ring_multiplications: int = 0
    streamed_projection_calls: int = 0
    streamed_projection_ring_multiplications: int = 0
    maximum_deferred_net_scale_payload_bits: int = 0
    maximum_deferred_net_vector_payload_bits: int = 0
    maximum_deferred_net_live_payload_bits: int = 0
    maximum_relative_unit_scale_payload_bits: int = 0
    maximum_relative_aligned_vector_payload_bits: int = 0
    maximum_relative_alignment_live_payload_bits: int = 0
    maximum_streamed_projection_scale_payload_bits: int = 0
    maximum_streamed_projection_element_payload_bits: int = 0
    maximum_streamed_projection_live_payload_bits: int = 0


DEFERRED_FIELDS = (
    "deferred_balance_calls",
    "deferred_net_residual_actions",
    "deferred_net_residual_ring_multiplications",
    "deferred_norm_verifications",
    "relative_unit_alignment_calls",
    "relative_unit_alignment_ring_multiplications",
    "streamed_projection_calls",
    "streamed_projection_ring_multiplications",
    "maximum_deferred_net_scale_payload_bits",
    "maximum_deferred_net_vector_payload_bits",
    "maximum_deferred_net_live_payload_bits",
    "maximum_relative_unit_scale_payload_bits",
    "maximum_relative_aligned_vector_payload_bits",
    "maximum_relative_alignment_live_payload_bits",
    "maximum_streamed_projection_scale_payload_bits",
    "maximum_streamed_projection_element_payload_bits",
    "maximum_streamed_projection_live_payload_bits",
)


def stats_json(
    stats: DeferredStats,
    pi_stats: pi_content.PiStats,
) -> dict[str, Any]:
    result = prior.stats_json(stats, pi_stats)
    for name in DEFERRED_FIELDS:
        result[name] = getattr(stats, name)
    return result


def deferred_balance_vector(
    vector: RingVector,
    base_ledger: list[int],
    stats: DeferredStats,
) -> tuple[RingVector, list[int]]:
    """Run exact line descent on norm/ledger, then act on residual once."""

    if len(base_ledger) != UNIT_RANK:
        fail("unit ledger width changed")
    if cyclo.vector_is_zero(vector):
        return cyclo.zero_vector(), [0 for _ in range(UNIT_RANK)]

    stats.balance_calls += 1
    stats.deferred_balance_calls += 1
    original = list(vector)
    initial_ledger = list(base_ledger)
    ledger = list(base_ledger)
    current_norm = base.vector_norm_element(original, stats)
    current_energy = base.exact_norm_energy(current_norm, stats)

    certified = False
    for _ in range(MAX_COORDINATE_SWEEPS):
        sweep_changed = False
        for direction_index, direction_entry in enumerate(DIRECTION_TABLE):
            move, trial_energy, trial_norm = prior.exact_coordinate_minimum(
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

    materialized_norm = base.vector_norm_element(result, stats)
    stats.deferred_norm_verifications += 1
    if materialized_norm != current_norm:
        fail("deferred net unit action changed the certified norm")
    return result, ledger


def relative_add_balanced_vectors(
    left: base.BalancedVector,
    right: base.BalancedVector,
    pi_stats: pi_content.PiStats,
    stats: DeferredStats,
) -> base.BalancedVector:
    """Add through one relative unit action under the left ledger."""

    if cyclo.vector_is_zero(left.residual):
        return right
    if cyclo.vector_is_zero(right.residual):
        return left

    relative_ledger = [
        right_value - left_value
        for left_value, right_value in zip(
            left.unit_ledger,
            right.unit_ledger,
            strict=True,
        )
    ]
    relative_scale = base.ledger_scale(relative_ledger, stats)
    aligned_right = base.multiply_vector(
        relative_scale,
        right.residual,
    )
    stats.relative_unit_alignment_calls += 1
    stats.relative_unit_alignment_ring_multiplications += len(aligned_right)
    stats.unit_vector_ring_multiplications += len(aligned_right)

    scale_bits = base.element_payload_bits(relative_scale)
    input_bits = base.vector_payload_bits(right.residual)
    aligned_bits = base.vector_payload_bits(aligned_right)
    stats.maximum_relative_unit_scale_payload_bits = max(
        stats.maximum_relative_unit_scale_payload_bits,
        scale_bits,
    )
    stats.maximum_relative_aligned_vector_payload_bits = max(
        stats.maximum_relative_aligned_vector_payload_bits,
        aligned_bits,
    )
    stats.maximum_relative_alignment_live_payload_bits = max(
        stats.maximum_relative_alignment_live_payload_bits,
        scale_bits + input_bits + aligned_bits,
    )

    combined = pi_content.scaled_vector_add(
        pi_content.ScaledVector(
            left.residual,
            left.pi_exponent,
        ),
        pi_content.ScaledVector(
            aligned_right,
            right.pi_exponent,
        ),
        pi_stats,
    )
    return base.normalize_balanced_vector(
        combined.residual,
        combined.exponent,
        list(left.unit_ledger),
        pi_stats,
        stats,
    )


def streamed_project_boundary(
    output: base.BalancedVector,
    pi_stats: pi_content.PiStats,
    stats: DeferredStats,
) -> RingElement:
    """Project residual first and materialize one scalar unit action."""

    projected_residual = cyclo.project_boundary(output.residual)
    unit_scale = base.ledger_scale(
        list(output.unit_ledger),
        stats,
    )
    unit_projected = cyclo.ring_multiply(
        unit_scale,
        projected_residual,
    )
    stats.streamed_projection_calls += 1
    stats.streamed_projection_ring_multiplications += 1
    stats.unit_scale_materializations += 1
    stats.unit_vector_ring_multiplications += 1

    scaled = pi_content.normalize_element(
        unit_projected,
        output.pi_exponent,
        pi_stats,
    )
    boundary = pi_content.materialize_element(scaled, pi_stats)
    scale_bits = base.element_payload_bits(unit_scale)
    projected_bits = base.element_payload_bits(projected_residual)
    unit_projected_bits = base.element_payload_bits(unit_projected)
    boundary_bits = base.element_payload_bits(boundary)
    stats.maximum_streamed_projection_scale_payload_bits = max(
        stats.maximum_streamed_projection_scale_payload_bits,
        scale_bits,
    )
    stats.maximum_streamed_projection_element_payload_bits = max(
        stats.maximum_streamed_projection_element_payload_bits,
        projected_bits,
        unit_projected_bits,
        boundary_bits,
    )
    stats.maximum_streamed_projection_live_payload_bits = max(
        stats.maximum_streamed_projection_live_payload_bits,
        scale_bits + projected_bits + unit_projected_bits + boundary_bits,
    )
    stats.maximum_unit_materialization_payload_bits = max(
        stats.maximum_unit_materialization_payload_bits,
        boundary_bits,
    )
    return boundary


def named_temporary_maxima_sum(metrics: dict[str, Any]) -> int:
    return sum(
        metrics[name]
        for name in (
            "maximum_search_power_live_pair_payload_bits",
            "maximum_search_trial_norm_payload_bits",
            "maximum_search_energy_scalar_pair_bits",
            "maximum_deferred_net_live_payload_bits",
            "maximum_relative_alignment_live_payload_bits",
            "maximum_streamed_projection_live_payload_bits",
        )
    )


def deferred_case_result(
    periods: int,
    block: cyclo.CompiledBlock,
) -> dict[str, Any]:
    result = base.case_result(periods, block)
    metrics = result["balanced_stats"]
    named_temporary = named_temporary_maxima_sum(metrics)
    table_payload = prior.compiled_unit_table_payload_bits()
    named_total = (
        result["balanced_declared_live_state_payload_bits"]
        + table_payload
        + named_temporary
    )
    predecessor_total = PREDECESSOR_NAMED_COMPONENT_MAXIMA_SUM_BITS[
        (result["family"], periods)
    ]
    raw_payload = result["raw_recurrence_payload_bits"]
    result.update(
        {
            "compiled_unit_table_payload_bits": table_payload,
            "named_deferred_temporary_maxima_sum_bits": named_temporary,
            "named_component_maxima_sum_bits": named_total,
            "named_component_maxima_sum_minus_raw_payload_bits": (
                named_total - raw_payload
            ),
            "named_component_maxima_sum_beats_raw_recurrence_payload": (
                named_total < raw_payload
            ),
            "predecessor_named_component_maxima_sum_bits": (
                predecessor_total
            ),
            "named_component_maxima_sum_change_from_predecessor_bits": (
                named_total - predecessor_total
            ),
            "named_component_maxima_sum_improves_predecessor": (
                named_total < predecessor_total
            ),
            "all_nonzero_balance_calls_coordinatewise_certified": (
                metrics["coordinatewise_certified_calls"]
                == metrics["balance_calls"]
                and metrics["coordinate_sweep_cap_hits"] == 0
                and metrics["coordinate_bracket_cap_hits"] == 0
            ),
            "one_or_zero_net_actions_per_balance_call": (
                metrics["deferred_net_residual_actions"]
                <= metrics["balance_calls"]
            ),
            "no_per_move_vector_materialization": (
                metrics["maximum_accepted_move_scale_payload_bits"] == 0
                and metrics[
                    "maximum_accepted_coordinate_vector_payload_bits"
                ]
                == 0
            ),
        }
    )
    return result


def deferred_controls() -> dict[str, bool]:
    sample_vector = [
        cyclo.ring_add(cyclo.ring_one(), cyclo.ring_monomial(3)),
        cyclo.ring_add(
            cyclo.ring_monomial(2),
            cyclo.ring_monomial(7),
        ),
    ]
    immediate_stats = prior.ExactSearchStats()
    immediate_vector, immediate_ledger = prior.balance_vector(
        sample_vector,
        [0 for _ in range(UNIT_RANK)],
        immediate_stats,
    )
    deferred_stats = DeferredStats()
    deferred_vector, deferred_ledger = deferred_balance_vector(
        sample_vector,
        [0 for _ in range(UNIT_RANK)],
        deferred_stats,
    )

    pi_stats = pi_content.PiStats()
    left = base.normalize_balanced_vector(
        sample_vector,
        1,
        [1, 0, -1, 0, 0, 0, 0],
        pi_stats,
        DeferredStats(),
    )
    right = base.normalize_balanced_vector(
        list(reversed(sample_vector)),
        2,
        [0, 1, 0, -1, 0, 0, 0],
        pi_stats,
        DeferredStats(),
    )
    reference_stats = DeferredStats()
    reference_sum = ORIGINAL_ADD_BALANCED_VECTORS(
        left,
        right,
        pi_content.PiStats(),
        reference_stats,
    )
    relative_stats = DeferredStats()
    relative_sum = relative_add_balanced_vectors(
        left,
        right,
        pi_content.PiStats(),
        relative_stats,
    )
    reference_materialized = base.materialize_unit_vector(
        reference_sum,
        DeferredStats(),
    )
    relative_materialized = base.materialize_unit_vector(
        relative_sum,
        DeferredStats(),
    )
    reference_boundary = ORIGINAL_PROJECT_BOUNDARY(
        relative_sum,
        pi_content.PiStats(),
        DeferredStats(),
    )
    streamed_stats = DeferredStats()
    streamed_boundary = streamed_project_boundary(
        relative_sum,
        pi_content.PiStats(),
        streamed_stats,
    )
    return {
        "deferred_balance_residual_equal_to_immediate": (
            deferred_vector == immediate_vector
        ),
        "deferred_balance_ledger_equal_to_immediate": (
            deferred_ledger == immediate_ledger
        ),
        "deferred_balance_norm_equal_to_immediate": (
            base.vector_norm_element(deferred_vector)
            == base.vector_norm_element(immediate_vector)
        ),
        "deferred_balance_uses_at_most_one_net_action": (
            deferred_stats.deferred_net_residual_actions <= 1
        ),
        "relative_addition_materialized_value_equal": (
            relative_sum.pi_exponent == reference_sum.pi_exponent
            and relative_materialized == reference_materialized
        ),
        "relative_addition_materializes_one_operand": (
            relative_stats.relative_unit_alignment_calls == 1
        ),
        "streamed_projection_boundary_equal": (
            streamed_boundary == reference_boundary
        ),
        "streamed_projection_materializes_one_element": (
            streamed_stats.streamed_projection_calls == 1
            and streamed_stats.streamed_projection_ring_multiplications
            == 1
        ),
    }


def main() -> int:
    if len(sys.argv) != 1:
        fail(
            "usage: f17_cubic_chain_period17_"
            "pi_unit_deferred_ledger_stream.py"
        )

    base.BalanceStats = DeferredStats
    base.stats_json = stats_json
    base.balance_vector = deferred_balance_vector
    base.ledger_scale = prior.tracked_ledger_scale
    base.add_balanced_vectors = relative_add_balanced_vectors
    base.project_boundary = streamed_project_boundary

    controls = deferred_controls()
    blocks = {
        family.lower(): cyclo.build_compiled_block(family)
        for family in ("PRIMARY", "REUSE")
    }
    cases = [
        deferred_case_result(periods, blocks[family])
        for periods in TESTED_PERIODS
        for family in ("primary", "reuse")
    ]
    restored = base.restoration_reuse_case(
        blocks["primary"],
        blocks["reuse"],
    )
    inherited_controls = base.controls(
        blocks["primary"],
        blocks["reuse"],
    )

    period64_improves_predecessor = all(
        case["named_component_maxima_sum_improves_predecessor"]
        for case in cases
        if case["periods"] == 64
    )
    period64_beats_raw = all(
        case["named_component_maxima_sum_beats_raw_recurrence_payload"]
        for case in cases
        if case["periods"] == 64
    )
    result = {
        "result": "PASS",
        "experiment": (
            "DEFERRED_EXACT_UNIT_LEDGER_SEARCH_WITH_SINGLE_NET_"
            "RESIDUAL_ACTION_AND_STREAMED_UNIT_MATERIALIZATION"
        ),
        "claim_candidate": (
            "BOUNDED_EXACT_49_DIRECTION_UNIT_SEARCH_DEFERS_RESIDUAL_"
            "MUTATION_TO_ONE_NET_ACTION_ALIGNS_ADDITIONS_BY_ONE_"
            "RELATIVE_UNIT_ACTION_AND_PROJECTS_BEFORE_ONE_SCALAR_"
            "UNIT_MATERIALIZATION_FOR_TWO_PUBLIC_F17_PERIOD17_"
            "FAMILIES_AT_PERIODS1AND64_WITH_EXACT_BOUNDARY_"
            "RESTORATION_AND_PERIOD1_CROSS_FAMILY_REUSE"
        ),
        "classification_candidate": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level_candidate": "PACKAGE_SELF_REVIEW",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "tested_periods": list(TESTED_PERIODS),
        "declared_exact_search_direction_count": len(SEARCH_DIRECTIONS),
        "maximum_coordinate_sweeps": MAX_COORDINATE_SWEEPS,
        "maximum_bracket_magnitude": MAX_BRACKET_MAGNITUDE,
        "search_numeric_type": "EXACT_Z_ZETA17_AND_INTEGER_TRACE_ONLY",
        "residual_action_schedule": (
            "LEDGER_AND_NORM_DURING_SEARCH_THEN_ONE_NET_RESIDUAL_ACTION"
        ),
        "addition_materialization_schedule": (
            "ONE_RELATIVE_UNIT_ACTION_UNDER_LEFT_OPERAND_LEDGER"
        ),
        "projection_materialization_schedule": (
            "PROJECT_RESIDUAL_THEN_ONE_SCALAR_UNIT_ACTION"
        ),
        "logarithms_used": False,
        "floating_point_used": False,
        "embedding_table_used": False,
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
        "deferred_controls": controls,
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
        "all_cases_one_or_zero_net_actions_per_balance": all(
            case["one_or_zero_net_actions_per_balance_call"]
            for case in cases
        ),
        "all_cases_avoid_per_move_vector_materialization": all(
            case["no_per_move_vector_materialization"]
            for case in cases
        ),
        "all_cases_reduce_pi_content_payload": all(
            case["balanced_reduces_pi_content_payload"]
            for case in cases
        ),
        "all_cases_beat_raw_recurrence_resident_payload": all(
            case["balanced_beats_raw_recurrence_payload"]
            for case in cases
        ),
        "period64_named_component_maxima_improve_predecessor": (
            period64_improves_predecessor
        ),
        "period64_named_component_maxima_beat_raw": (
            period64_beats_raw
        ),
        "restoration_reuse_case": restored,
        "controls": inherited_controls,
        "matched_classical": {
            "identical_exact_deferred_ledger_search_available": True,
            "identical_exact_recurrence_available": True,
            "raw_recurrence_retained": True,
            "comparison_establishes_advantage": False,
        },
        "resource_law": {
            "compiled_unit_table_payload_bits_counted": True,
            "search_power_result_factor_live_pairs_counted": True,
            "search_trial_norm_counted": True,
            "search_energy_scalar_pairs_counted": True,
            "net_scale_input_and_output_vector_live_payload_counted": True,
            "relative_scale_input_and_output_vector_live_payload_counted": (
                True
            ),
            "streamed_projection_observed_elements_counted": True,
            "duplicate_public_topology_state_payload_counted": True,
            "verification_norm_recomputation_counted": True,
            "named_component_maxima_sum_is_simultaneous_peak": False,
            "python_object_overhead_bounded": False,
            "allocator_peak_bounded": False,
            "internal_ring_multiplication_peak_bounded": False,
            "pi_materialization_internal_peak_bounded": False,
            "whole_process_peak_bounded": False,
        },
        "observation": (
            "DEFERRED_LEDGER_SEARCH_REMOVES_PER_ACCEPTED_MOVE_FULL_"
            "VECTOR_MUTATION_AND_REPLACES_ABSOLUTE_PAIR_"
            "MATERIALIZATION_WITH_ONE_RELATIVE_ACTION_WHILE_"
            + (
                "PERIOD64_NAMED_COMPONENT_MAXIMA_FALL_BELOW_RAW"
                if period64_beats_raw
                else "PERIOD64_NAMED_COMPONENT_MAXIMA_REMAIN_ABOVE_RAW"
            )
        ),
        "not_established": [
            "GLOBAL_CYCLOTOMIC_UNIT_OPTIMALITY",
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
            "EXACT_STREAMED_RELATIVE_LEDGER_ADDITION_TREE_WITH_"
            "LIVE_RANGE_RELEASE"
            if not period64_beats_raw
            else
            "GROWING_DEPTH_RESIDUAL_HEIGHT_LAW_AFTER_DEFERRED_"
            "UNIT_MATERIALIZATION"
        ),
        "next_obstruction": (
            "DEFERRED_ACTION_REDUCES_HISTORY_DEPENDENT_WORK_BUT_THE_"
            "IDENTICAL_COMPACT_CLASSICAL_EXECUTION_REMAINS_AND_NO_"
            "DISTINCT_PHASE_RESOURCE_IS_ESTABLISHED"
        ),
        "generation_and_lease_are_observed_bookkeeping_only": True,
        "generation_or_lease_enforcement_established": False,
        "terminal": False,
    }

    hard_gate = {
        "deferred_controls": all(controls.values()),
        "raw_boundaries": result["all_raw_recurrence_boundaries_equal"],
        "pi_boundaries": result["all_pi_content_boundaries_equal"],
        "restoration": result["all_cases_restore_exactly"],
        "coordinatewise_certified": (
            result["all_cases_coordinatewise_certified"]
        ),
        "one_net_action": (
            result["all_cases_one_or_zero_net_actions_per_balance"]
        ),
        "no_per_move_materialization": (
            result["all_cases_avoid_per_move_vector_materialization"]
        ),
        "resident_below_pi": result["all_cases_reduce_pi_content_payload"],
        "resident_below_raw": (
            result["all_cases_beat_raw_recurrence_resident_payload"]
        ),
        "inherited_controls": all(inherited_controls.values()),
        "primary_reuse_restored": restored["primary_restored_exactly"],
        "unrelated_reuse_restored": restored["reuse_restored_exactly"],
        "same_original_backing": restored["same_original_backing"],
        "fresh_restored_reuse_boundary": (
            restored["fresh_restored_reuse_boundary_equal"]
        ),
    }
    if not all(hard_gate.values()):
        fail(
            "deferred unit ledger qualification failed: "
            + json.dumps(hard_gate, sort_keys=True)
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
