#!/usr/bin/env python3
"""Separate exact oracle for the single-resident Horner phase carrier.

The oracle imports no production Horner successor. It compiles both public
operators independently, advances recurrence coefficients sequentially by
x modulo q, evaluates the normalized Horner schedule, evaluates a matched raw
Horner schedule, and reconstructs forward, inverse, resource, restoration,
reuse, and mutation results.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_pi_unit_deferred_ledger_stream_oracle as phase


exact = phase.exact
prior = phase.prior
ring = phase.ring
reference = phase.reference
pi_reference = phase.pi_reference

PRIME = prior.PRIME
DIMENSION = prior.DIMENSION
UNIT_RANK = phase.UNIT_RANK
EXPECTED_PERIODS = phase.EXPECTED_PERIODS

RingElement = phase.RingElement
RingVector = phase.RingVector


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class HornerOracleStats(phase.DeferredOracleStats):
    horner_operator_applications: int = 0
    horner_operator_ring_multiply_accumulations: int = 0
    horner_scalar_terms: int = 0
    horner_additions: int = 0
    maximum_horner_coefficient_program_payload_bits: int = 0
    maximum_horner_named_checkpoint_payload_bits: int = 0
    maximum_carrier_resident_payload_bits: int = 0
    maximum_projection_resident_plus_work_payload_bits: int = 0
    maximum_inverse_resident_plus_work_payload_bits: int = 0
    maximum_horner_named_checkpoint_vector_count: int = 0
    maximum_horner_coefficient_program_elements: int = 0


HORNER_FIELDS = (
    "horner_operator_applications",
    "horner_operator_ring_multiply_accumulations",
    "horner_scalar_terms",
    "horner_additions",
    "maximum_horner_coefficient_program_payload_bits",
    "maximum_horner_named_checkpoint_payload_bits",
    "maximum_carrier_resident_payload_bits",
    "maximum_projection_resident_plus_work_payload_bits",
    "maximum_inverse_resident_plus_work_payload_bits",
    "maximum_horner_named_checkpoint_vector_count",
    "maximum_horner_coefficient_program_elements",
)


def balanced_vector_payload_bits(value: prior.BalancedVector) -> int:
    return (
        prior.vector_payload_bits(value.residual)
        + prior.signed_bits(value.pi_exponent)
        + prior.ledger_payload_bits(value.ledger)
    )


def balanced_element_payload_bits(value: prior.BalancedElement) -> int:
    return (
        prior.element_payload_bits(value.residual)
        + prior.signed_bits(value.pi_exponent)
        + prior.ledger_payload_bits(value.ledger)
    )


def record_horner_named_checkpoint(
    stats: HornerOracleStats,
    coefficient_payload: int,
    *vectors: prior.BalancedVector | RingVector,
) -> None:
    payload = coefficient_payload
    for value in vectors:
        if isinstance(value, prior.BalancedVector):
            payload += balanced_vector_payload_bits(value)
        else:
            payload += prior.vector_payload_bits(value)
    stats.maximum_horner_named_checkpoint_payload_bits = max(
        stats.maximum_horner_named_checkpoint_payload_bits,
        payload,
    )
    stats.maximum_horner_named_checkpoint_vector_count = max(
        stats.maximum_horner_named_checkpoint_vector_count,
        len(vectors),
    )


def apply_operator(
    operator: list[list[RingElement]],
    value: prior.BalancedVector,
    stats: HornerOracleStats,
) -> tuple[prior.BalancedVector, RingVector]:
    raw = reference.matrix_vector_multiply(
        operator,
        value.residual,
    )
    advanced = prior.normalize_vector(
        raw,
        value.pi_exponent,
        value.ledger,
        stats,
    )
    stats.horner_operator_applications += 1
    stats.horner_operator_ring_multiply_accumulations += (
        PRIME * PRIME
    )
    return advanced, raw


def build_horner_output(
    context: dict[str, Any],
    periods: int,
    stats: HornerOracleStats,
) -> prior.BalancedVector:
    seed = prior.normalize_vector(
        context["seed"],
        0,
        tuple(0 for _ in range(UNIT_RANK)),
        stats,
    )
    scaled_coefficients = pi_reference.sequential_scaled_coefficients(
        periods,
        context["characteristic"],
    )
    coefficients = [
        prior.normalize_element(
            value.residual,
            value.exponent,
            tuple(0 for _ in range(UNIT_RANK)),
            stats,
        )
        for value in scaled_coefficients
    ]
    del scaled_coefficients
    coefficient_payload = sum(
        balanced_element_payload_bits(value)
        for value in coefficients
    )
    stats.maximum_horner_coefficient_program_payload_bits = max(
        stats.maximum_horner_coefficient_program_payload_bits,
        coefficient_payload,
    )
    stats.maximum_horner_coefficient_program_elements = max(
        stats.maximum_horner_coefficient_program_elements,
        len(coefficients),
    )

    accumulator = prior.multiply(
        coefficients[-1],
        seed,
        stats,
    )
    stats.horner_scalar_terms += 1
    record_horner_named_checkpoint(
        stats,
        coefficient_payload,
        seed,
        accumulator,
    )
    for coefficient in reversed(coefficients[:-1]):
        advanced, raw_advanced = apply_operator(
            context["operator"],
            accumulator,
            stats,
        )
        term = prior.multiply(coefficient, seed, stats)
        stats.horner_scalar_terms += 1
        next_accumulator = prior.add_vectors(
            advanced,
            term,
            stats,
        )
        stats.horner_additions += 1
        record_horner_named_checkpoint(
            stats,
            coefficient_payload,
            seed,
            accumulator,
            raw_advanced,
            advanced,
            term,
            next_accumulator,
        )
        accumulator = next_accumulator
        del raw_advanced, advanced, term, next_accumulator
    output, raw_output = apply_operator(
        context["operator"],
        accumulator,
        stats,
    )
    record_horner_named_checkpoint(
        stats,
        coefficient_payload,
        seed,
        accumulator,
        raw_output,
        output,
    )
    if stats.horner_operator_applications != DIMENSION:
        fail("oracle Horner operator count changed")
    return output


def carrier_payload_bits(value: prior.BalancedVector) -> int:
    return balanced_vector_payload_bits(value)


def metrics_json(stats: HornerOracleStats) -> dict[str, int]:
    result = phase.record_metrics([], [], stats)
    for name in HORNER_FIELDS:
        result[name] = getattr(stats, name)
    return result


@dataclass
class RawHornerStats:
    operator_applications: int = 0
    operator_ring_multiply_accumulations: int = 0
    scalar_vector_ring_multiplications: int = 0
    vector_additions: int = 0
    maximum_coefficient_program_payload_bits: int = 0
    maximum_named_checkpoint_payload_bits: int = 0
    maximum_named_checkpoint_vector_count: int = 0


def sequential_raw_coefficients(
    periods: int,
    characteristic: list[RingElement],
) -> list[RingElement]:
    q_low = [
        characteristic[DIMENSION - degree]
        for degree in range(DIMENSION)
    ]
    coefficients = [
        ring.ring_zero()
        for _ in range(DIMENSION)
    ]
    coefficients[0] = ring.ring_one()
    for _ in range(periods - 1):
        highest = coefficients[-1]
        advanced = [
            ring.ring_zero()
            for _ in range(DIMENSION)
        ]
        for degree in range(DIMENSION):
            shifted = (
                coefficients[degree - 1]
                if degree > 0
                else ring.ring_zero()
            )
            advanced[degree] = ring.ring_subtract(
                shifted,
                ring.ring_multiply(highest, q_low[degree]),
            )
        coefficients = advanced
    return coefficients


def raw_scale(
    scalar: RingElement,
    vector: RingVector,
    stats: RawHornerStats,
) -> RingVector:
    result = [
        ring.ring_multiply(scalar, value)
        for value in vector
    ]
    stats.scalar_vector_ring_multiplications += len(result)
    return result


def record_raw_named_checkpoint(
    stats: RawHornerStats,
    coefficient_payload: int,
    *vectors: RingVector,
) -> None:
    stats.maximum_named_checkpoint_payload_bits = max(
        stats.maximum_named_checkpoint_payload_bits,
        coefficient_payload
        + sum(prior.vector_payload_bits(value) for value in vectors),
    )
    stats.maximum_named_checkpoint_vector_count = max(
        stats.maximum_named_checkpoint_vector_count,
        len(vectors),
    )


def raw_horner_boundary(
    context: dict[str, Any],
    periods: int,
) -> tuple[RingElement, RawHornerStats]:
    coefficients = sequential_raw_coefficients(
        periods,
        context["characteristic"],
    )
    coefficient_payload = sum(
        prior.element_payload_bits(value)
        for value in coefficients
    )
    stats = RawHornerStats(
        maximum_coefficient_program_payload_bits=coefficient_payload,
    )
    seed = context["seed"]
    accumulator = raw_scale(coefficients[-1], seed, stats)
    record_raw_named_checkpoint(
        stats,
        coefficient_payload,
        seed,
        accumulator,
    )
    for coefficient in reversed(coefficients[:-1]):
        advanced = reference.matrix_vector_multiply(
            context["operator"],
            accumulator,
        )
        term = raw_scale(coefficient, seed, stats)
        next_accumulator = [
            ring.ring_add(left, right)
            for left, right in zip(advanced, term, strict=True)
        ]
        stats.operator_applications += 1
        stats.operator_ring_multiply_accumulations += PRIME * PRIME
        stats.vector_additions += len(next_accumulator)
        record_raw_named_checkpoint(
            stats,
            coefficient_payload,
            seed,
            accumulator,
            advanced,
            term,
            next_accumulator,
        )
        accumulator = next_accumulator
        del advanced, term, next_accumulator
    output = reference.matrix_vector_multiply(
        context["operator"],
        accumulator,
    )
    stats.operator_applications += 1
    stats.operator_ring_multiply_accumulations += PRIME * PRIME
    record_raw_named_checkpoint(
        stats,
        coefficient_payload,
        seed,
        accumulator,
        output,
    )
    return reference.project(output), stats


def raw_metrics(stats: RawHornerStats) -> dict[str, int]:
    return {
        "operator_applications": stats.operator_applications,
        "operator_ring_multiply_accumulations": (
            stats.operator_ring_multiply_accumulations
        ),
        "scalar_vector_ring_multiplications": (
            stats.scalar_vector_ring_multiplications
        ),
        "vector_additions": stats.vector_additions,
        "maximum_coefficient_program_payload_bits": (
            stats.maximum_coefficient_program_payload_bits
        ),
        "maximum_named_checkpoint_payload_bits": (
            stats.maximum_named_checkpoint_payload_bits
        ),
        "maximum_named_checkpoint_vector_count": (
            stats.maximum_named_checkpoint_vector_count
        ),
    }


def named_search_temporary_maxima_sum(metrics: dict[str, int]) -> int:
    return sum(
        metrics[name]
        for name in (
            "maximum_search_power_live_pair_payload_bits",
            "maximum_search_trial_norm_payload_bits",
            "maximum_search_energy_scalar_pair_bits",
            "maximum_deferred_net_live_payload_bits",
            "maximum_relative_alignment_live_payload_bits",
        )
    )


def case_check(
    case: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    stats = HornerOracleStats()
    output = build_horner_output(context, case["periods"], stats)
    resident_payload = carrier_payload_bits(output)
    stats.maximum_carrier_resident_payload_bits = resident_payload
    stats.maximum_resident_payload_bits = resident_payload
    boundary = prior.project(output, stats)
    stats.maximum_projection_resident_plus_work_payload_bits = (
        resident_payload
        + stats.maximum_streamed_projection_live_payload_bits
    )
    metrics = metrics_json(stats)

    inverse_stats = HornerOracleStats()
    inverse_output = build_horner_output(
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
            [ring.ring_zero() for _ in range(PRIME)]
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
    raw_boundary, raw_stats = raw_horner_boundary(
        context,
        case["periods"],
    )
    raw_values = raw_metrics(raw_stats)
    phase_named_checkpoint = max(
        metrics["maximum_horner_named_checkpoint_payload_bits"],
        metrics["maximum_projection_resident_plus_work_payload_bits"],
        inverse_metrics[
            "maximum_inverse_resident_plus_work_payload_bits"
        ],
    )
    search_temporary = named_search_temporary_maxima_sum(metrics)
    named_total = (
        phase_named_checkpoint
        + exact.compiled_unit_table_payload_bits()
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


@dataclass
class OracleCarrier:
    output: RingVector
    pi_exponent: int
    unit_ledger: list[int]
    generation: int = 0
    lease: int = 0

    @classmethod
    def create(cls) -> "OracleCarrier":
        return cls(
            [ring.ring_zero() for _ in range(PRIME)],
            0,
            [0 for _ in range(UNIT_RANK)],
        )

    def backing(self) -> tuple[int, ...]:
        return (id(self.output), id(self.unit_ledger))

    def all_zero(self) -> bool:
        return (
            all(value == ring.ring_zero() for value in self.output)
            and self.pi_exponent == 0
            and not any(self.unit_ledger)
        )


def execute_carrier(
    carrier: OracleCarrier,
    context: dict[str, Any],
    periods: int,
) -> tuple[RingElement, bool]:
    if not carrier.all_zero():
        fail("oracle Horner carrier was not restored")
    carrier.lease += 1
    stats = HornerOracleStats()
    output = build_horner_output(context, periods, stats)
    carrier.output[:] = output.residual
    carrier.pi_exponent = output.pi_exponent
    carrier.unit_ledger[:] = output.ledger
    boundary = prior.project(output, stats)
    expected = build_horner_output(
        context,
        periods,
        HornerOracleStats(),
    )
    if (
        carrier.output != expected.residual
        or carrier.pi_exponent != expected.pi_exponent
        or tuple(carrier.unit_ledger) != expected.ledger
    ):
        fail("oracle Horner inverse mismatch")
    carrier.output[:] = [
        ring.ring_subtract(actual, value)
        for actual, value in zip(
            carrier.output,
            expected.residual,
            strict=True,
        )
    ]
    carrier.pi_exponent -= expected.pi_exponent
    carrier.unit_ledger[:] = [
        actual - value
        for actual, value in zip(
            carrier.unit_ledger,
            expected.ledger,
            strict=True,
        )
    ]
    carrier.generation += 1
    return boundary, carrier.all_zero()


def restoration_check(
    contexts: dict[str, dict[str, Any]],
    production: dict[str, Any],
) -> dict[str, bool]:
    carrier = OracleCarrier.create()
    backing = carrier.backing()
    _, primary_restored = execute_carrier(
        carrier,
        contexts["primary"],
        1,
    )
    reuse_boundary, reuse_restored = execute_carrier(
        carrier,
        contexts["reuse"],
        1,
    )
    fresh_boundary, fresh_restored = execute_carrier(
        OracleCarrier.create(),
        contexts["reuse"],
        1,
    )
    period64_checks: dict[str, bool] = {}
    for family in ("primary", "reuse"):
        period64_carrier = OracleCarrier.create()
        period64_backing = period64_carrier.backing()
        _, period64_restored = execute_carrier(
            period64_carrier,
            contexts[family],
            64,
        )
        period64_checks[
            f"period64_{family}_restored_exactly"
        ] = period64_restored
        period64_checks[
            f"period64_{family}_same_backing"
        ] = period64_carrier.backing() == period64_backing
        period64_checks[
            f"period64_{family}_payload_and_ledgers_zero"
        ] = period64_carrier.all_zero()
        period64_checks[
            f"period64_{family}_generation_correct"
        ] = period64_carrier.generation == 1
        period64_checks[
            f"period64_{family}_lease_correct"
        ] = period64_carrier.lease == 1
    expected = production["restoration_reuse_case"]
    result = {
        "primary_restored_exactly": primary_restored,
        "reuse_restored_exactly": reuse_restored,
        "fresh_restored_exactly": fresh_restored,
        "same_original_backing": carrier.backing() == backing,
        "fresh_reuse_boundary_equal": reuse_boundary == fresh_boundary,
        "all_payload_and_ledgers_zero": carrier.all_zero(),
        "generation_equal": carrier.generation == expected["generation"],
        "lease_equal": carrier.lease == expected["lease"],
        "no_inverse_history": (
            expected["retained_inverse_history_bytes"] == 0
        ),
        "no_baseline_reload": expected["baseline_reload_bytes"] == 0,
    }
    result.update(period64_checks)
    return result


def mutation_check(context: dict[str, Any]) -> dict[str, bool]:
    output = build_horner_output(
        context,
        64,
        HornerOracleStats(),
    )
    original = prior.project(output, HornerOracleStats())
    mutated_residual = list(output.residual)
    mutated_residual[0] = ring.ring_add(
        mutated_residual[0],
        ring.ring_one(),
    )
    mutated = prior.project(
        prior.BalancedVector(
            mutated_residual,
            output.pi_exponent,
            output.ledger,
        ),
        HornerOracleStats(),
    )
    return {
        "nonzero_resident_mutation_changes_boundary": (
            mutated != original
        ),
        "mutation_is_not_silently_canonicalized": (
            mutated_residual != output.residual
        ),
    }


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_chain_period17_"
            "pi_unit_horner_stream_oracle.py PRODUCTION_RESULT"
        )
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        production = json.load(handle)
    if tuple(production["tested_periods"]) != EXPECTED_PERIODS:
        fail("oracle tested periods changed")

    prior.OracleStats = HornerOracleStats
    prior.balance = phase.balance
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
    restoration = restoration_check(contexts, production)
    mutations = mutation_check(contexts["primary"])
    scope = {
        "production_result_pass": production["result"] == "PASS",
        "one_resident_phase_vector_asserted": (
            production["carrier_resident_phase_vector_count"] == 1
        ),
        "named_vector_checkpoints_reported": all(
            case["phase_stats"][
                "maximum_horner_named_checkpoint_vector_count"
            ]
            == 6
            and case["raw_horner_stats"][
                "maximum_named_checkpoint_vector_count"
            ]
            == 5
            for case in production["cases"]
        ),
        "all_boundaries_equal": (
            production["all_raw_horner_boundaries_equal"]
            and production[
                "all_prior_raw_recurrence_boundaries_equal"
            ]
        ),
        "all_restored": production["all_cases_restore_exactly"],
        "matched_raw_horner_retained": (
            production["matched_classical"][
                "matched_raw_horner_named_checkpoint_implemented"
            ]
        ),
        "identical_normalized_execution_retained": (
            production["matched_classical"][
                "identical_normalized_horner_available"
            ]
            and not production["matched_classical"][
                "comparison_establishes_advantage"
            ]
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
        and all(scope.values())
    )
    result = {
        "result": "PASS" if result_pass else "FAIL",
        "experiment": (
            "SEPARATE_EXACT_SINGLE_RESIDENT_HORNER_PHASE_ORACLE"
        ),
        "oracle_imports_production_module": False,
        "oracle_coefficient_method": (
            "SEQUENTIAL_MULTIPLICATION_BY_X_MOD_Q"
        ),
        "production_coefficient_method": (
            "BINARY_POLYNOMIAL_POWERING_MOD_Q"
        ),
        "oracle_horner_method": (
            "INDEPENDENT_PUBLIC_OPERATOR_SINGLE_RESIDENT_VECTOR_"
            "HORNER_WITH_EXPLICIT_NAMED_IMMUTABLE_VECTOR_CHECKPOINTS"
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
            "LINUX_X86_64_PYTHON_TWO_PUBLIC_F17_PERIOD17_FAMILIES_"
            "PERIODS1AND64_ONE_RESIDENT_CARRIER_VECTOR_SIX_NAMED_"
            "PHASE_VECTOR_CHECKPOINT_FIVE_NAMED_RAW_VECTOR_"
            "CHECKPOINT_EXACT_BOUNDARY_NAMED_RESOURCE_INVERSE_"
            "RESTORATION_AND_PERIOD1_CROSS_FAMILY_REUSE_PARITY_"
            "SOFTWARE_ONLY"
        ),
        "not_established": production["not_established"],
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
