#!/usr/bin/env python3
"""Single-resident-vector Horner phase recurrence with exact uncomputation.

The predecessor reduced history-dependent unit work, but retained the seed,
sixteen basis vectors, sixteen coefficient registers, and the final output.
Its inverse reconstructed the same full state beside the resident carrier.

This successor compiles the same public characteristic polynomial into a
Horner schedule. It stores only the final normalized phase vector in the
borrowed carrier and measures explicitly named immutable-vector checkpoints
during operator, normalization, scalar-term, and addition transitions. It
projects only the resident output, rematerializes the Horner output from
public topology, and subtracts it exactly from the same carrier backing.

An exact raw Z[zeta_17] Horner implementation uses the same schedule and is
the matched compact representation baseline.  The identical normalized
Horner implementation remains available to classical software, so this
bounded diagnostic does not establish a distinct phase resource.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_pi_unit_deferred_ledger_stream as prior


base = prior.base
cyclo = prior.cyclo
pi_content = prior.pi_content
recurrence = base.recurrence

UNIT_RANK = prior.UNIT_RANK
TESTED_PERIODS = prior.TESTED_PERIODS
SEARCH_DIRECTIONS = prior.SEARCH_DIRECTIONS

RingElement = prior.RingElement
RingVector = prior.RingVector


def fail(message: str) -> None:
    raise RuntimeError(message)


def balanced_vector_payload_bits(value: base.BalancedVector) -> int:
    return (
        base.vector_payload_bits(value.residual)
        + base.signed_bits(value.pi_exponent)
        + base.ledger_payload_bits(list(value.unit_ledger))
    )


def balanced_element_payload_bits(value: base.BalancedElement) -> int:
    return (
        base.element_payload_bits(value.residual)
        + base.signed_bits(value.pi_exponent)
        + base.ledger_payload_bits(list(value.unit_ledger))
    )


@dataclass
class HornerStats(prior.DeferredStats):
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


def stats_json(
    stats: HornerStats,
    pi_stats: pi_content.PiStats,
) -> dict[str, Any]:
    result = prior.stats_json(stats, pi_stats)
    for name in HORNER_FIELDS:
        result[name] = getattr(stats, name)
    return result


def coefficient_program_payload_bits(
    coefficients: list[base.BalancedElement],
) -> int:
    return sum(
        balanced_element_payload_bits(value)
        for value in coefficients
    )


def record_horner_named_checkpoint(
    stats: HornerStats,
    coefficient_payload: int,
    *vectors: base.BalancedVector | RingVector,
) -> None:
    payload = coefficient_payload
    count = 0
    for value in vectors:
        count += 1
        if isinstance(value, base.BalancedVector):
            payload += balanced_vector_payload_bits(value)
        else:
            payload += base.vector_payload_bits(value)
    stats.maximum_horner_named_checkpoint_payload_bits = max(
        stats.maximum_horner_named_checkpoint_payload_bits,
        payload,
    )
    stats.maximum_horner_named_checkpoint_vector_count = max(
        stats.maximum_horner_named_checkpoint_vector_count,
        count,
    )


def apply_balanced_operator(
    block: cyclo.CompiledBlock,
    value: base.BalancedVector,
    pi_stats: pi_content.PiStats,
    stats: HornerStats,
) -> tuple[base.BalancedVector, RingVector]:
    block_stats = cyclo.Stats()
    raw = cyclo.apply_operator(
        block.operator,
        value.residual,
        block_stats,
    )
    advanced = base.normalize_balanced_vector(
        raw,
        value.pi_exponent,
        list(value.unit_ledger),
        pi_stats,
        stats,
    )
    stats.horner_operator_applications += 1
    stats.horner_operator_ring_multiply_accumulations += (
        block_stats.ring_multiply_accumulations
    )
    return advanced, raw


def build_horner_output(
    block: cyclo.CompiledBlock,
    periods: int,
    pi_stats: pi_content.PiStats,
    stats: HornerStats,
) -> base.BalancedVector:
    """Evaluate A * p(A) * seed with one Horner accumulator."""

    seed = base.normalize_balanced_vector(
        cyclo.seed_vector(block.public_program),
        0,
        [0 for _ in range(UNIT_RANK)],
        pi_stats,
        stats,
    )
    scaled_coefficients = pi_content.scaled_recurrence_coefficients(
        periods,
        block.characteristic,
        pi_stats,
    )
    coefficients = [
        base.normalize_balanced_element(
            value.residual,
            value.exponent,
            [0 for _ in range(UNIT_RANK)],
            pi_stats,
            stats,
        )
        for value in scaled_coefficients
    ]
    del scaled_coefficients
    if len(coefficients) != recurrence.COEFFICIENT_REGISTERS:
        fail("Horner coefficient program width changed")
    coefficient_payload = coefficient_program_payload_bits(coefficients)
    stats.maximum_horner_coefficient_program_payload_bits = max(
        stats.maximum_horner_coefficient_program_payload_bits,
        coefficient_payload,
    )
    stats.maximum_horner_coefficient_program_elements = max(
        stats.maximum_horner_coefficient_program_elements,
        len(coefficients),
    )

    accumulator = base.multiply_balanced(
        coefficients[-1],
        seed,
        pi_stats,
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
        advanced, raw_advanced = apply_balanced_operator(
            block,
            accumulator,
            pi_stats,
            stats,
        )
        term = base.multiply_balanced(
            coefficient,
            seed,
            pi_stats,
            stats,
        )
        stats.horner_scalar_terms += 1
        next_accumulator = base.add_balanced_vectors(
            advanced,
            term,
            pi_stats,
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

    output, raw_output = apply_balanced_operator(
        block,
        accumulator,
        pi_stats,
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
    if stats.horner_operator_applications != recurrence.BASIS_MESSAGES:
        fail("Horner operator application count changed")
    return output


@dataclass
class HornerCarrier:
    output: RingVector
    output_pi_exponent: int
    output_unit_ledger: list[int]
    generation: int = 0
    lease: int = 0
    active: bool = False
    pending_operations: int = 0
    phase: str = "RESTORED"

    @classmethod
    def create(cls) -> "HornerCarrier":
        return cls(
            cyclo.zero_vector(),
            0,
            [0 for _ in range(UNIT_RANK)],
        )

    def all_zero(self) -> bool:
        return (
            cyclo.vector_is_zero(self.output)
            and self.output_pi_exponent == 0
            and not any(self.output_unit_ledger)
            and not self.active
            and self.pending_operations == 0
            and self.phase == "RESTORED"
        )

    def backing_identity(self) -> tuple[int, ...]:
        return (
            id(self.output),
            id(self.output_unit_ledger),
        )

    def resident_value(self) -> base.BalancedVector:
        return base.BalancedVector(
            self.output,
            self.output_pi_exponent,
            tuple(self.output_unit_ledger),
        )

    def payload_bits(self) -> int:
        return (
            base.vector_payload_bits(self.output)
            + base.signed_bits(self.output_pi_exponent)
            + base.ledger_payload_bits(self.output_unit_ledger)
        )

    def canonical_state(self) -> dict[str, Any]:
        return {
            "output_zero": cyclo.vector_is_zero(self.output),
            "pi_ledger_zero": self.output_pi_exponent == 0,
            "unit_ledger_zero": not any(self.output_unit_ledger),
            "all_payload_and_ledgers_zero": (
                cyclo.vector_is_zero(self.output)
                and self.output_pi_exponent == 0
                and not any(self.output_unit_ledger)
            ),
            "generation": self.generation,
            "lease": self.lease,
            "active": self.active,
            "pending_operations": self.pending_operations,
            "phase": self.phase,
        }


def record_carrier(
    carrier: HornerCarrier,
    stats: HornerStats,
) -> int:
    payload = carrier.payload_bits()
    stats.maximum_carrier_resident_payload_bits = max(
        stats.maximum_carrier_resident_payload_bits,
        payload,
    )
    stats.maximum_resident_payload_bits = max(
        stats.maximum_resident_payload_bits,
        payload,
    )
    return payload


def populate_forward(
    carrier: HornerCarrier,
    block: cyclo.CompiledBlock,
    periods: int,
) -> tuple[RingElement, pi_content.PiStats, HornerStats]:
    if not carrier.all_zero():
        fail("Horner carrier was not restored")
    carrier.active = True
    carrier.lease += 1
    carrier.pending_operations = 1
    carrier.phase = "BUILD_HORNER_OUTPUT"
    pi_stats = pi_content.PiStats()
    stats = HornerStats()
    output = build_horner_output(
        block,
        periods,
        pi_stats,
        stats,
    )
    carrier.output[:] = output.residual
    carrier.output_pi_exponent = output.pi_exponent
    carrier.output_unit_ledger[:] = output.unit_ledger
    carrier.phase = "HORNER_OUTPUT_RESIDENT"
    resident_payload = record_carrier(carrier, stats)
    del output

    boundary = base.project_boundary(
        carrier.resident_value(),
        pi_stats,
        stats,
    )
    stats.maximum_projection_resident_plus_work_payload_bits = max(
        stats.maximum_projection_resident_plus_work_payload_bits,
        resident_payload
        + stats.maximum_streamed_projection_live_payload_bits,
    )
    return boundary, pi_stats, stats


def restore_forward(
    carrier: HornerCarrier,
    block: cyclo.CompiledBlock,
    periods: int,
) -> tuple[pi_content.PiStats, HornerStats]:
    if carrier.phase != "HORNER_OUTPUT_RESIDENT":
        fail("Horner inverse was reordered")
    inverse_pi_stats = pi_content.PiStats()
    inverse_stats = HornerStats()
    expected = build_horner_output(
        block,
        periods,
        inverse_pi_stats,
        inverse_stats,
    )
    inverse_stats.maximum_inverse_resident_plus_work_payload_bits = max(
        inverse_stats.maximum_inverse_resident_plus_work_payload_bits,
        carrier.payload_bits()
        + inverse_stats.maximum_horner_named_checkpoint_payload_bits,
    )
    if (
        carrier.output != expected.residual
        or carrier.output_pi_exponent != expected.pi_exponent
        or tuple(carrier.output_unit_ledger)
        != expected.unit_ledger
    ):
        fail("Horner inverse rematerialization mismatch")
    cyclo.subtract_vector_exact(carrier.output, expected.residual)
    carrier.output_pi_exponent -= expected.pi_exponent
    carrier.output_unit_ledger[:] = [
        actual - value
        for actual, value in zip(
            carrier.output_unit_ledger,
            expected.unit_ledger,
            strict=True,
        )
    ]
    carrier.pending_operations = 0
    carrier.active = False
    carrier.phase = "RESTORED"
    carrier.generation += 1
    record_carrier(carrier, inverse_stats)
    if not carrier.all_zero():
        fail("Horner carrier did not restore exactly")
    return inverse_pi_stats, inverse_stats


@dataclass
class Transaction:
    boundary: RingElement
    pi_stats: pi_content.PiStats
    stats: HornerStats
    inverse_pi_stats: pi_content.PiStats
    inverse_stats: HornerStats
    restored_exactly: bool
    same_backing: bool


def execute_transaction(
    carrier: HornerCarrier,
    block: cyclo.CompiledBlock,
    periods: int,
) -> Transaction:
    if not isinstance(carrier, HornerCarrier):
        fail("null or invalid Horner carrier")
    backing = carrier.backing_identity()
    boundary, pi_stats, stats = populate_forward(
        carrier,
        block,
        periods,
    )
    inverse_pi_stats, inverse_stats = restore_forward(
        carrier,
        block,
        periods,
    )
    return Transaction(
        boundary,
        pi_stats,
        stats,
        inverse_pi_stats,
        inverse_stats,
        carrier.all_zero(),
        carrier.backing_identity() == backing,
    )


@dataclass
class RawHornerStats:
    operator_applications: int = 0
    operator_ring_multiply_accumulations: int = 0
    scalar_vector_ring_multiplications: int = 0
    vector_additions: int = 0
    maximum_coefficient_program_payload_bits: int = 0
    maximum_named_checkpoint_payload_bits: int = 0
    maximum_named_checkpoint_vector_count: int = 0


def raw_vector_scale(
    scalar: RingElement,
    vector: RingVector,
    stats: RawHornerStats,
) -> RingVector:
    result = [
        cyclo.ring_multiply(scalar, value)
        for value in vector
    ]
    stats.scalar_vector_ring_multiplications += len(result)
    return result


def record_raw_horner_named_checkpoint(
    stats: RawHornerStats,
    coefficient_payload: int,
    *vectors: RingVector,
) -> None:
    stats.maximum_named_checkpoint_payload_bits = max(
        stats.maximum_named_checkpoint_payload_bits,
        coefficient_payload
        + sum(base.vector_payload_bits(value) for value in vectors),
    )
    stats.maximum_named_checkpoint_vector_count = max(
        stats.maximum_named_checkpoint_vector_count,
        len(vectors),
    )


def raw_horner_boundary(
    block: cyclo.CompiledBlock,
    periods: int,
) -> tuple[RingElement, RawHornerStats]:
    recurrence_stats = recurrence.RecurrenceStats()
    coefficients = recurrence.recurrence_coefficients(
        periods,
        block.characteristic,
        recurrence_stats,
    )
    coefficient_payload = sum(
        base.element_payload_bits(value)
        for value in coefficients
    )
    stats = RawHornerStats(
        maximum_coefficient_program_payload_bits=coefficient_payload,
    )
    seed = cyclo.seed_vector(block.public_program)
    accumulator = raw_vector_scale(
        coefficients[-1],
        seed,
        stats,
    )
    record_raw_horner_named_checkpoint(
        stats,
        coefficient_payload,
        seed,
        accumulator,
    )
    for coefficient in reversed(coefficients[:-1]):
        block_stats = cyclo.Stats()
        advanced = cyclo.apply_operator(
            block.operator,
            accumulator,
            block_stats,
        )
        term = raw_vector_scale(coefficient, seed, stats)
        next_accumulator = [
            cyclo.ring_add(left, right)
            for left, right in zip(advanced, term, strict=True)
        ]
        stats.vector_additions += len(next_accumulator)
        stats.operator_applications += 1
        stats.operator_ring_multiply_accumulations += (
            block_stats.ring_multiply_accumulations
        )
        record_raw_horner_named_checkpoint(
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
    block_stats = cyclo.Stats()
    output = cyclo.apply_operator(
        block.operator,
        accumulator,
        block_stats,
    )
    stats.operator_applications += 1
    stats.operator_ring_multiply_accumulations += (
        block_stats.ring_multiply_accumulations
    )
    record_raw_horner_named_checkpoint(
        stats,
        coefficient_payload,
        seed,
        accumulator,
        output,
    )
    if stats.operator_applications != recurrence.BASIS_MESSAGES:
        fail("raw Horner operator application count changed")
    return cyclo.project_boundary(output), stats


def raw_horner_stats_json(stats: RawHornerStats) -> dict[str, int]:
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


def named_search_temporary_maxima_sum(metrics: dict[str, Any]) -> int:
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


def case_result(
    periods: int,
    block: cyclo.CompiledBlock,
) -> dict[str, Any]:
    carrier = HornerCarrier.create()
    transaction = execute_transaction(carrier, block, periods)
    phase_metrics = stats_json(
        transaction.stats,
        transaction.pi_stats,
    )
    inverse_metrics = stats_json(
        transaction.inverse_stats,
        transaction.inverse_pi_stats,
    )
    raw_boundary, raw_stats = raw_horner_boundary(block, periods)
    pi_case = pi_content.case_result(periods, block)
    boundary_sha256 = hashlib.sha256(
        cyclo.encoded_ring_object(transaction.boundary)
    ).hexdigest()
    search_temporary = named_search_temporary_maxima_sum(phase_metrics)
    phase_named_checkpoint = max(
        phase_metrics["maximum_horner_named_checkpoint_payload_bits"],
        phase_metrics[
            "maximum_projection_resident_plus_work_payload_bits"
        ],
        inverse_metrics[
            "maximum_inverse_resident_plus_work_payload_bits"
        ],
    )
    named_total = (
        phase_named_checkpoint
        + prior.prior.compiled_unit_table_payload_bits()
        + search_temporary
    )
    raw_payload = raw_stats.maximum_named_checkpoint_payload_bits
    return {
        "periods": periods,
        "family": block.family,
        "equivalent_edges": periods * cyclo.PERIOD,
        "boundary": transaction.boundary,
        "boundary_sha256": boundary_sha256,
        "raw_horner_boundary_sha256": hashlib.sha256(
            cyclo.encoded_ring_object(raw_boundary)
        ).hexdigest(),
        "raw_horner_boundary_equal": (
            transaction.boundary == raw_boundary
        ),
        "prior_raw_recurrence_boundary_equal": (
            boundary_sha256 == pi_case["boundary_sha256"]
        ),
        "phase_stats": phase_metrics,
        "inverse_rematerialization_stats": inverse_metrics,
        "raw_horner_stats": raw_horner_stats_json(raw_stats),
        "phase_named_checkpoint_payload_bits": phase_named_checkpoint,
        "named_search_temporary_maxima_sum_bits": search_temporary,
        "compiled_unit_table_payload_bits": (
            prior.prior.compiled_unit_table_payload_bits()
        ),
        "phase_named_component_maxima_sum_bits": named_total,
        "raw_horner_named_checkpoint_payload_bits": raw_payload,
        "phase_minus_raw_horner_named_payload_bits": (
            named_total - raw_payload
        ),
        "phase_named_payload_beats_raw_horner": (
            named_total < raw_payload
        ),
        "raw_horner_minus_prior_raw_carrier_payload_bits": (
            raw_payload
            - pi_case["raw_recurrence_baseline"][
                "maximum_carrier_payload_bits"
            ]
        ),
        "restored_exactly": transaction.restored_exactly,
        "same_backing": transaction.same_backing,
        "canonical_restored_state": carrier.canonical_state(),
    }


def restoration_reuse_case(
    primary: cyclo.CompiledBlock,
    reuse: cyclo.CompiledBlock,
) -> dict[str, Any]:
    carrier = HornerCarrier.create()
    backing = carrier.backing_identity()
    primary_transaction = execute_transaction(carrier, primary, 1)
    reuse_transaction = execute_transaction(carrier, reuse, 1)
    fresh = execute_transaction(
        HornerCarrier.create(),
        reuse,
        1,
    )
    return {
        "periods": 1,
        "primary_restored_exactly": (
            primary_transaction.restored_exactly
        ),
        "reuse_restored_exactly": reuse_transaction.restored_exactly,
        "same_original_backing": carrier.backing_identity() == backing,
        "fresh_restored_reuse_boundary_equal": (
            reuse_transaction.boundary == fresh.boundary
        ),
        "retained_inverse_history_bytes": 0,
        "baseline_reload_bytes": 0,
        "generation": carrier.generation,
        "lease": carrier.lease,
        "full_carrier_object_state_equal": False,
        "repeated_use_metadata_width_bounded": False,
        "canonical_restored_state": carrier.canonical_state(),
    }


def controls(
    primary: cyclo.CompiledBlock,
    reuse: cyclo.CompiledBlock,
) -> dict[str, bool]:
    missing = HornerCarrier.create()
    populate_forward(missing, primary, 1)
    missing_inverse_leaves_resident_state = not missing.all_zero()

    reordered = HornerCarrier.create()
    reordered_rejected = False
    try:
        restore_forward(reordered, primary, 1)
    except RuntimeError:
        reordered_rejected = True

    wrong = HornerCarrier.create()
    populate_forward(wrong, primary, 1)
    wrong_state = wrong.canonical_state()
    wrong_rejected = False
    try:
        restore_forward(wrong, reuse, 1)
    except RuntimeError:
        wrong_rejected = True
    wrong_failure_atomic = wrong.canonical_state() == wrong_state

    mutation = HornerCarrier.create()
    populate_forward(mutation, primary, 1)
    mutation.output[0] = cyclo.ring_add(
        mutation.output[0],
        cyclo.ring_one(),
    )
    mutation_rejected = False
    try:
        restore_forward(mutation, primary, 1)
    except RuntimeError:
        mutation_rejected = True

    null_rejected = False
    try:
        execute_transaction(None, primary, 1)  # type: ignore[arg-type]
    except RuntimeError:
        null_rejected = True

    primary_boundary, _ = raw_horner_boundary(primary, 1)
    reuse_boundary, _ = raw_horner_boundary(reuse, 1)
    return {
        "missing_inverse_leaves_resident_state": (
            missing_inverse_leaves_resident_state
        ),
        "reordered_inverse_rejected": reordered_rejected,
        "wrong_inverse_rejected": wrong_rejected,
        "wrong_inverse_failure_atomic_before_carrier_mutation": (
            wrong_failure_atomic
        ),
        "resident_mutation_rejected": mutation_rejected,
        "null_carrier_rejected": null_rejected,
        "public_family_boundaries_differ": (
            primary_boundary != reuse_boundary
        ),
        "no_snapshot_interface": not any(
            "snapshot" in name.lower()
            for name in dir(HornerCarrier)
        ),
    }


def main() -> int:
    if len(sys.argv) != 1:
        fail(
            "usage: f17_cubic_chain_period17_"
            "pi_unit_horner_stream.py"
        )

    base.BalanceStats = HornerStats
    base.stats_json = stats_json
    base.balance_vector = prior.deferred_balance_vector
    base.ledger_scale = prior.prior.tracked_ledger_scale
    base.add_balanced_vectors = prior.relative_add_balanced_vectors
    base.project_boundary = prior.streamed_project_boundary

    blocks = {
        family.lower(): cyclo.build_compiled_block(family)
        for family in ("PRIMARY", "REUSE")
    }
    cases = [
        case_result(periods, blocks[family])
        for periods in TESTED_PERIODS
        for family in ("primary", "reuse")
    ]
    restored = restoration_reuse_case(
        blocks["primary"],
        blocks["reuse"],
    )
    control_results = controls(
        blocks["primary"],
        blocks["reuse"],
    )
    phase_beats_raw = all(
        case["phase_named_payload_beats_raw_horner"]
        for case in cases
    )

    result = {
        "result": "PASS",
        "experiment": (
            "TOPOLOGY_DERIVED_SINGLE_RESIDENT_VECTOR_HORNER_PHASE_"
            "RECURRENCE_WITH_STREAMED_INVERSE_REMATERIALIZATION_"
            "NAMED_IMMUTABLE_VECTOR_CHECKPOINTS_AND_MATCHED_RAW_"
            "HORNER_BASELINE"
        ),
        "claim_candidate": (
            "BOUNDED_PUBLIC_TOPOLOGY_HORNER_PHASE_RECURRENCE_STORES_"
            "ONE_FINAL_17_ELEMENT_CYCLOTOMIC_VECTOR_REMATERIALIZES_"
            "THE_INVERSE_WITHOUT_RETAINED_HISTORY_RESTORES_TWO_F17_"
            "PERIOD17_FAMILIES_AT_PERIODS1AND64_REUSES_THE_SAME_"
            "CARRIER_CROSS_FAMILY_AT_PERIOD1_AND_RETAINS_A_MATCHED_"
            "RAW_HORNER_NAMED_VECTOR_CHECKPOINT_BASELINE"
        ),
        "classification_candidate": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level_candidate": "PACKAGE_SELF_REVIEW",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "tested_periods": list(TESTED_PERIODS),
        "declared_exact_search_direction_count": len(SEARCH_DIRECTIONS),
        "carrier_resident_phase_vector_count": 1,
        "retained_inverse_history_bytes": 0,
        "public_topology_compilation_answer_independent": True,
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
        "all_phase_named_payloads_beat_raw_horner": phase_beats_raw,
        "restoration_reuse_case": restored,
        "controls": control_results,
        "matched_classical": {
            "matched_raw_horner_named_checkpoint_implemented": True,
            "identical_normalized_horner_available": True,
            "same_public_coefficients_operator_and_boundary": True,
            "comparison_establishes_advantage": False,
        },
        "resource_law": {
            "public_coefficient_program_counted": True,
            "named_seed_accumulator_raw_operator_term_and_next_"
            "checkpoints_counted": True,
            "resident_plus_inverse_rematerialization_counted": True,
            "search_power_trial_norm_energy_net_and_alignment_counted": True,
            "fixed_unit_table_counted": True,
            "raw_horner_named_vectors_and_coefficients_counted": True,
            "named_component_maxima_sum_is_simultaneous_peak": False,
            "python_object_overhead_bounded": False,
            "allocator_peak_bounded": False,
            "internal_ring_multiplication_peak_bounded": False,
            "whole_process_peak_bounded": False,
        },
        "observation": (
            "SINGLE_RESIDENT_VECTOR_HORNER_REMOVES_THE_RETAINED_17_"
            "VECTOR_BASIS_AND_FULL_STATE_INVERSE_DUPLICATE_BUT_"
            + (
                "THE_NORMALIZED_NAMED_TOTAL_IS_BELOW_THE_RAW_HORNER_"
                "BASELINE_WITHOUT_BEATING_THE_IDENTICAL_NORMALIZED_"
                "CLASSICAL_EXECUTION"
                if phase_beats_raw
                else
                "THE_MATCHED_RAW_OR_IDENTICAL_NORMALIZED_CLASSICAL_"
                "EXECUTION_REMAINS_THE_RESOURCE_OBSTRUCTION"
            )
        ),
        "not_established": [
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
            "EXACT_SEARCH_POWER_HEIGHT_QUOTIENT_OR_STREAMED_"
            "MULTI_MODULUS_TRACE_CERTIFICATE_WITH_MATCHED_CLASSICAL_COST"
        ),
        "next_obstruction": (
            "HORNER_HISTORY_RELEASE_REMOVES_THE_RETAINED_MULTI_VECTOR_"
            "OBSTRUCTION_BUT_EXACT_ARITHMETIC_WIDTH_AND_THE_IDENTICAL_"
            "NORMALIZED_CLASSICAL_HORNER_EXECUTION_REMAIN"
        ),
        "generation_and_lease_are_observed_bookkeeping_only": True,
        "generation_or_lease_enforcement_established": False,
        "terminal": False,
    }
    hard_gate = {
        "raw_horner_boundary": result["all_raw_horner_boundaries_equal"],
        "prior_raw_boundary": (
            result["all_prior_raw_recurrence_boundaries_equal"]
        ),
        "restoration": result["all_cases_restore_exactly"],
        "controls": all(control_results.values()),
        "primary_reuse_restored": restored["primary_restored_exactly"],
        "unrelated_reuse_restored": restored["reuse_restored_exactly"],
        "same_original_backing": restored["same_original_backing"],
        "fresh_restored_reuse_boundary": (
            restored["fresh_restored_reuse_boundary_equal"]
        ),
        "operator_count": all(
            case["phase_stats"]["horner_operator_applications"]
            == recurrence.BASIS_MESSAGES
            and case["raw_horner_stats"]["operator_applications"]
            == recurrence.BASIS_MESSAGES
            for case in cases
        ),
    }
    if not all(hard_gate.values()):
        fail(
            "Horner stream qualification failed: "
            + json.dumps(hard_gate, sort_keys=True)
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
