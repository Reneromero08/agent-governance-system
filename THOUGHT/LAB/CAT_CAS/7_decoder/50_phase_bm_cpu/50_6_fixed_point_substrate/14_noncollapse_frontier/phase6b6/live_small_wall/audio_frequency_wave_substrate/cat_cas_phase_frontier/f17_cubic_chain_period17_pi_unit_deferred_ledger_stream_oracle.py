#!/usr/bin/env python3
"""Separate exact oracle for deferred unit-ledger materialization.

This oracle uses the independently implemented cyclotomic-ring, recurrence,
pi-content, and exact line-search kernels.  It does not import the production
successor.  It independently accumulates the unit search in norm/ledger
coordinates, applies one net residual action, aligns additions through a
relative ledger, projects before scalar materialization, and reconstructs
the production boundary and resource tuples.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_pi_unit_exact_coordinate_descent_oracle as exact


prior = exact.prior
ring = exact.ring
reference = prior.reference
pi_reference = prior.pi_reference

UNIT_RANK = exact.UNIT_RANK
EXPECTED_PERIODS = exact.EXPECTED_PERIODS
MAX_DIRECTION_SWEEPS = exact.MAX_DIRECTION_SWEEPS
DIRECTION_TABLE = exact.DIRECTION_TABLE

RingElement = exact.RingElement
RingVector = exact.RingVector


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class DeferredOracleStats(exact.SearchStats):
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


def balance(
    vector: RingVector,
    base_ledger: tuple[int, ...] | list[int],
    stats: DeferredOracleStats,
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
    current_norm = prior.norm_element(original, stats)
    current_energy = prior.observe_energy(
        prior.field_trace(current_norm),
        stats,
    )
    certified = False
    for _ in range(MAX_DIRECTION_SWEEPS):
        sweep_changed = False
        for direction_index, direction_entry in enumerate(DIRECTION_TABLE):
            move, trial_energy, trial_norm = exact.line_minimum(
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
    if observed_norm != current_norm:
        fail("oracle deferred net action changed the certified norm")
    return result, tuple(ledger)


def add_vectors(
    left: prior.BalancedVector,
    right: prior.BalancedVector,
    stats: DeferredOracleStats,
) -> prior.BalancedVector:
    if all(
        element == reference.ring_zero()
        for element in left.residual
    ):
        return right
    if all(
        element == reference.ring_zero()
        for element in right.residual
    ):
        return left

    relative_ledger = tuple(
        right_value - left_value
        for left_value, right_value in zip(
            left.ledger,
            right.ledger,
            strict=True,
        )
    )
    scale = prior.ledger_scale(relative_ledger, stats)
    aligned_right = prior.multiply_vector(scale, right.residual)
    stats.relative_unit_alignment_calls += 1
    stats.relative_unit_alignment_ring_multiplications += len(aligned_right)
    stats.unit_vector_ring_multiplications += len(aligned_right)

    scale_bits = prior.element_payload_bits(scale)
    input_bits = prior.vector_payload_bits(right.residual)
    aligned_bits = prior.vector_payload_bits(aligned_right)
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
    combined = pi_reference.scaled_vector_add(
        pi_reference.ScaledVector(
            left.residual,
            left.pi_exponent,
        ),
        pi_reference.ScaledVector(
            aligned_right,
            right.pi_exponent,
        ),
    )
    return prior.normalize_vector(
        combined.residual,
        combined.exponent,
        left.ledger,
        stats,
    )


def project(
    output: prior.BalancedVector,
    stats: DeferredOracleStats,
) -> RingElement:
    projected_residual = reference.project(output.residual)
    scale = prior.ledger_scale(output.ledger, stats)
    unit_projected = reference.ring_multiply(
        scale,
        projected_residual,
    )
    stats.streamed_projection_calls += 1
    stats.streamed_projection_ring_multiplications += 1
    stats.unit_scale_materializations += 1
    stats.unit_vector_ring_multiplications += 1
    scaled = pi_reference.normalize_element(
        unit_projected,
        output.pi_exponent,
    )
    boundary = pi_reference.materialize_element(scaled)

    scale_bits = prior.element_payload_bits(scale)
    projected_bits = prior.element_payload_bits(projected_residual)
    unit_projected_bits = prior.element_payload_bits(unit_projected)
    boundary_bits = prior.element_payload_bits(boundary)
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


def record_metrics(
    messages: list[prior.BalancedVector],
    coefficients: list[prior.BalancedElement],
    stats: DeferredOracleStats,
) -> dict[str, int]:
    result = exact.record_metrics(messages, coefficients, stats)
    for name in DEFERRED_FIELDS:
        result[name] = getattr(stats, name)
    return result


def named_temporary_maxima_sum(metrics: dict[str, int]) -> int:
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


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_chain_period17_"
            "pi_unit_deferred_ledger_stream_oracle.py PRODUCTION_RESULT"
        )
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        production = json.load(handle)
    if tuple(production["tested_periods"]) != EXPECTED_PERIODS:
        fail("oracle tested periods changed")

    prior.OracleStats = DeferredOracleStats
    prior.balance = balance
    prior.add_vectors = add_vectors
    prior.project = project
    prior.record_metrics = record_metrics
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

    case_checks: list[dict[str, Any]] = []
    for case in production["cases"]:
        checked = prior.case_check(
            case,
            contexts[case["family"].lower()],
        )
        metrics = checked["exact_resource_tuple"]
        named_temporary = named_temporary_maxima_sum(metrics)
        named_total = (
            metrics["maximum_declared_live_state_payload_bits"]
            + exact.compiled_unit_table_payload_bits()
            + named_temporary
        )
        checked.update(
            {
                "compiled_unit_table_payload_equal": (
                    exact.compiled_unit_table_payload_bits()
                    == case["compiled_unit_table_payload_bits"]
                ),
                "named_deferred_temporary_maxima_sum_equal": (
                    named_temporary
                    == case["named_deferred_temporary_maxima_sum_bits"]
                ),
                "named_component_maxima_sum_equal": (
                    named_total == case["named_component_maxima_sum_bits"]
                ),
                "declared_direction_local_minimum_reexecuted": (
                    metrics["coordinatewise_certified_calls"]
                    == metrics["balance_calls"]
                    and metrics["coordinate_sweep_cap_hits"] == 0
                    and metrics["coordinate_bracket_cap_hits"] == 0
                ),
                "one_or_zero_net_actions_per_balance_reexecuted": (
                    metrics["deferred_net_residual_actions"]
                    <= metrics["balance_calls"]
                ),
                "per_move_vector_materialization_absent": (
                    metrics[
                        "maximum_accepted_move_scale_payload_bits"
                    ]
                    == 0
                    and metrics[
                        "maximum_accepted_coordinate_vector_payload_bits"
                    ]
                    == 0
                ),
            }
        )
        case_checks.append(checked)

    restoration = prior.restoration_check(contexts, production)
    mutations = prior.mutation_check(contexts["primary"])
    scope = {
        "production_result_pass": production["result"] == "PASS",
        "deferred_schedule_asserted": (
            production["residual_action_schedule"]
            == (
                "LEDGER_AND_NORM_DURING_SEARCH_THEN_ONE_NET_"
                "RESIDUAL_ACTION"
            )
        ),
        "all_boundaries_equal": (
            production["all_raw_recurrence_boundaries_equal"]
            and production["all_pi_content_boundaries_equal"]
        ),
        "all_restored": production["all_cases_restore_exactly"],
        "all_declared_direction_local_minima": (
            production["all_cases_coordinatewise_certified"]
        ),
        "all_cases_one_or_zero_net_actions": (
            production[
                "all_cases_one_or_zero_net_actions_per_balance"
            ]
        ),
        "per_move_materialization_absent": (
            production[
                "all_cases_avoid_per_move_vector_materialization"
            ]
        ),
        "identical_classical_execution_retained": (
            production["matched_classical"][
                "identical_exact_deferred_ledger_search_available"
            ]
            and production["matched_classical"][
                "identical_exact_recurrence_available"
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
            "SEPARATE_EXACT_DEFERRED_UNIT_LEDGER_STREAM_ORACLE"
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
            "DIFFERENCE_BINARY_SEARCH_WITH_DEFERRED_NET_ACTION"
        ),
        "oracle_materialization_method": (
            "RELATIVE_LEDGER_ADDITION_AND_PROJECT_THEN_SCALAR_UNIT_ACTION"
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
            "PATH_FAMILIES_PERIODS1AND64_EXACT_49_DIRECTION_SEARCH_"
            "WITH_ONE_NET_RESIDUAL_ACTION_RELATIVE_LEDGER_ADDITION_"
            "STREAMED_SCALAR_PROJECTION_EXACT_BOUNDARY_RESOURCE_"
            "INVERSE_AND_RESTORATION_PARITY_AND_PERIOD1_CROSS_"
            "FAMILY_REUSE_SOFTWARE_ONLY"
        ),
        "not_established": production["not_established"],
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
