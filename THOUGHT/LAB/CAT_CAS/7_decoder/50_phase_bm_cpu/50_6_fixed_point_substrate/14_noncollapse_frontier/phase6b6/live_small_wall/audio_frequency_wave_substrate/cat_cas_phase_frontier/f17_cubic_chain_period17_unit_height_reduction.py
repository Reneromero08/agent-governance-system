#!/usr/bin/env python3
"""Execute a reversible cyclotomic-unit gauge on the native-K recurrence.

The prior exact recurrence fixes resident slot and register counts but not
integer width.  This successor treats the seven-generator cyclotomic-unit family

    u_a = (1 - zeta**a) / (1 - zeta)
        = 1 + zeta + ... + zeta**(a - 1),  a = 2,...,8

as a multiplicative gauge.  Each resident message is stored as a reduced
Z[zeta_17] vector plus a seven-integer exponent ledger.  Unit scalars commute
with the K-linear period block, so the next message can be advanced directly
from the reduced predecessor.  A deterministic exact steepest-descent rule
chooses only transformations that reduce reduced-payload-plus-ledger bits.

The final contraction reconstructs every basis scale exactly.  Output,
coefficient registers, basis messages, and all ledgers are then subtracted
through deterministic rematerialization on the same backing.  The result
tests whether this declared unit gauge reduces actual exact height without
hiding the scale in uncounted state.  It does not claim globally optimal
unit reduction.  The search has a declared per-call step cap; a cap hit is
reported as an optimization ceiling rather than treated as an algebraic
failure.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_cyclotomic_module as cyclo
import f17_cubic_chain_period17_executed_recurrence as recurrence


PRIME = cyclo.PRIME
DIMENSION = cyclo.DIMENSION
MESSAGE_INTEGER_CELLS = cyclo.MESSAGE_INTEGER_CELLS
MESSAGE_SLOTS = recurrence.MESSAGE_SLOTS
BASIS_MESSAGES = recurrence.BASIS_MESSAGES
OUTPUT_SLOT = recurrence.OUTPUT_SLOT
COEFFICIENT_REGISTERS = recurrence.COEFFICIENT_REGISTERS
UNIT_GENERATOR_INDICES = tuple(range(2, 9))
UNIT_RANK = len(UNIT_GENERATOR_INDICES)
MAX_NORMALIZATION_STEPS = 128
TESTED_PERIODS = (1, 4, 16, 64, 256)
DENSE_DIRECT_PERIOD_CAP = 64

RingElement = cyclo.RingElement
RingVector = cyclo.RingVector


def fail(message: str) -> None:
    raise RuntimeError(message)


def signed_bits(value: int) -> int:
    return recurrence.signed_bits(value)


def unit_generator(index: int) -> RingElement:
    result = cyclo.ring_zero()
    for exponent in range(index):
        result = cyclo.ring_add(
            result,
            cyclo.ring_monomial(exponent),
        )
    return result


def unit_generator_inverse(index: int) -> RingElement:
    inverse_index = pow(index, -1, PRIME)
    result = cyclo.ring_zero()
    for multiplier in range(inverse_index):
        result = cyclo.ring_add(
            result,
            cyclo.ring_monomial((index * multiplier) % PRIME),
        )
    return result


UNIT_GENERATORS = tuple(
    unit_generator(index)
    for index in UNIT_GENERATOR_INDICES
)
UNIT_GENERATOR_INVERSES = tuple(
    unit_generator_inverse(index)
    for index in UNIT_GENERATOR_INDICES
)


def unit_identities_exact() -> bool:
    return all(
        cyclo.ring_multiply(generator, inverse)
        == cyclo.ring_one()
        for generator, inverse in zip(
            UNIT_GENERATORS,
            UNIT_GENERATOR_INVERSES,
            strict=True,
        )
    )


def vector_payload_bits(vector: RingVector) -> int:
    return sum(
        signed_bits(coefficient)
        for element in vector
        for coefficient in element
    )


def vector_maximum_signed_bits(vector: RingVector) -> int:
    return max(
        signed_bits(coefficient)
        for element in vector
        for coefficient in element
    )


def ledger_payload_bits(ledger: list[int]) -> int:
    return sum(signed_bits(exponent) for exponent in ledger)


def multiply_vector_by_scalar(
    scalar: RingElement,
    vector: RingVector,
) -> RingVector:
    return [
        cyclo.ring_multiply(scalar, element)
        for element in vector
    ]


@dataclass
class UnitStats:
    normalization_calls: int = 0
    normalization_step_cap_hits: int = 0
    normalization_candidate_vector_evaluations: int = 0
    normalization_selected_steps: int = 0
    normalization_vector_ring_multiplications: int = 0
    scale_ring_multiplications: int = 0
    output_ring_multiplications: int = 0
    basis_forward_block_applications: int = 0
    basis_inverse_block_applications: int = 0
    basis_ring_multiply_accumulations: int = 0
    maximum_carrier_payload_bits: int = 0
    maximum_reduced_coefficient_signed_bits: int = 1
    maximum_ledger_exponent_signed_bits: int = 1
    maximum_nonzero_message_slots: int = 0
    maximum_nonzero_message_ledgers: int = 0
    maximum_nonzero_coefficient_registers: int = 0
    maximum_transient_raw_message_payload_bits: int = 0
    maximum_transient_raw_message_signed_bits: int = 1


def candidate_vector(
    vector: RingVector,
    multiplier: RingElement,
    stats: UnitStats,
) -> RingVector:
    stats.normalization_candidate_vector_evaluations += 1
    stats.normalization_vector_ring_multiplications += len(vector)
    return multiply_vector_by_scalar(multiplier, vector)


def normalized_vector(
    vector: RingVector,
    base_ledger: list[int],
    stats: UnitStats,
) -> tuple[RingVector, list[int]]:
    if len(base_ledger) != UNIT_RANK:
        fail("invalid cyclotomic-unit ledger width")
    stats.normalization_calls += 1
    current = vector
    ledger = list(base_ledger)
    current_cost = (
        vector_payload_bits(current) + ledger_payload_bits(ledger)
    )
    selected_for_call = 0
    for _ in range(MAX_NORMALIZATION_STEPS):
        best: tuple[
            int,
            int,
            tuple[int, ...],
            int,
            int,
        ] | None = None
        for generator_index in range(UNIT_RANK):
            for delta, multiplier in (
                (1, UNIT_GENERATOR_INVERSES[generator_index]),
                (-1, UNIT_GENERATORS[generator_index]),
            ):
                trial = candidate_vector(current, multiplier, stats)
                trial_ledger = list(ledger)
                trial_ledger[generator_index] += delta
                choice = (
                    vector_payload_bits(trial)
                    + ledger_payload_bits(trial_ledger),
                    vector_maximum_signed_bits(trial),
                    tuple(trial_ledger),
                    generator_index,
                    delta,
                )
                if best is None or choice < best:
                    best = choice
        if best is None or best[0] >= current_cost:
            break
        generator_index = best[3]
        delta = best[4]
        multiplier = (
            UNIT_GENERATOR_INVERSES[generator_index]
            if delta == 1
            else UNIT_GENERATORS[generator_index]
        )
        current = candidate_vector(current, multiplier, stats)
        ledger[generator_index] += delta
        current_cost = best[0]
        stats.normalization_selected_steps += 1
        selected_for_call += 1
    if selected_for_call == MAX_NORMALIZATION_STEPS:
        stats.normalization_step_cap_hits += 1
    return current, ledger


def ring_power(
    base: RingElement,
    exponent: int,
    stats: UnitStats,
) -> RingElement:
    result = cyclo.ring_one()
    factor = base
    while exponent:
        if exponent & 1:
            result = cyclo.ring_multiply(result, factor)
            stats.scale_ring_multiplications += 1
        exponent >>= 1
        if exponent:
            factor = cyclo.ring_multiply(factor, factor)
            stats.scale_ring_multiplications += 1
    return result


def ledger_scale(
    ledger: list[int],
    stats: UnitStats,
) -> RingElement:
    result = cyclo.ring_one()
    for exponent, generator, inverse in zip(
        ledger,
        UNIT_GENERATORS,
        UNIT_GENERATOR_INVERSES,
        strict=True,
    ):
        factor = (
            ring_power(generator, exponent, stats)
            if exponent >= 0
            else ring_power(inverse, -exponent, stats)
        )
        result = cyclo.ring_multiply(result, factor)
        stats.scale_ring_multiplications += 1
    return result


@dataclass
class UnitCarrier:
    messages: list[RingVector]
    message_ledgers: list[list[int]]
    coefficients: list[RingElement]
    coefficient_ledger: list[int]
    generation: int = 0
    lease: int = 0
    active: bool = False
    pending_operations: int = 0
    phase: str = "RESTORED"

    @classmethod
    def create(cls) -> "UnitCarrier":
        return cls(
            messages=[
                cyclo.zero_vector()
                for _ in range(MESSAGE_SLOTS)
            ],
            message_ledgers=[
                [0 for _ in range(UNIT_RANK)]
                for _ in range(MESSAGE_SLOTS)
            ],
            coefficients=recurrence.polynomial_zero(),
            coefficient_ledger=[0 for _ in range(UNIT_RANK)],
        )

    def all_zero(self) -> bool:
        return (
            all(cyclo.vector_is_zero(row) for row in self.messages)
            and all(
                not any(ledger)
                for ledger in self.message_ledgers
            )
            and all(
                value == cyclo.ring_zero()
                for value in self.coefficients
            )
            and not any(self.coefficient_ledger)
        )

    def backing_identity(self) -> tuple[int, ...]:
        return (
            id(self.messages),
            *(id(message) for message in self.messages),
            id(self.message_ledgers),
            *(id(ledger) for ledger in self.message_ledgers),
            id(self.coefficients),
            id(self.coefficient_ledger),
        )

    def canonical_state(self) -> dict[str, Any]:
        return {
            "message_slots": len(self.messages),
            "message_ledger_width": UNIT_RANK,
            "coefficient_registers": len(self.coefficients),
            "coefficient_ledger_width": len(self.coefficient_ledger),
            "all_state_zero": self.all_zero(),
            "generation": self.generation,
            "lease": self.lease,
            "active": self.active,
            "pending_operations": self.pending_operations,
            "phase": self.phase,
        }


def record_peak(carrier: UnitCarrier, stats: UnitStats) -> None:
    payload = 0
    maximum_reduced_bits = 1
    maximum_ledger_bits = 1
    nonzero_messages = 0
    nonzero_message_ledgers = 0
    nonzero_coefficients = 0
    for message, ledger in zip(
        carrier.messages,
        carrier.message_ledgers,
        strict=True,
    ):
        if not cyclo.vector_is_zero(message):
            nonzero_messages += 1
        if any(ledger):
            nonzero_message_ledgers += 1
        payload += vector_payload_bits(message)
        maximum_reduced_bits = max(
            maximum_reduced_bits,
            vector_maximum_signed_bits(message),
        )
        payload += ledger_payload_bits(ledger)
        maximum_ledger_bits = max(
            maximum_ledger_bits,
            *(signed_bits(value) for value in ledger),
        )
    for element in carrier.coefficients:
        if element != cyclo.ring_zero():
            nonzero_coefficients += 1
        for coefficient in element:
            bits = signed_bits(coefficient)
            payload += bits
            maximum_reduced_bits = max(maximum_reduced_bits, bits)
    payload += ledger_payload_bits(carrier.coefficient_ledger)
    maximum_ledger_bits = max(
        maximum_ledger_bits,
        *(
            signed_bits(value)
            for value in carrier.coefficient_ledger
        ),
    )
    stats.maximum_carrier_payload_bits = max(
        stats.maximum_carrier_payload_bits,
        payload,
    )
    stats.maximum_reduced_coefficient_signed_bits = max(
        stats.maximum_reduced_coefficient_signed_bits,
        maximum_reduced_bits,
    )
    stats.maximum_ledger_exponent_signed_bits = max(
        stats.maximum_ledger_exponent_signed_bits,
        maximum_ledger_bits,
    )
    stats.maximum_nonzero_message_slots = max(
        stats.maximum_nonzero_message_slots,
        nonzero_messages,
    )
    stats.maximum_nonzero_message_ledgers = max(
        stats.maximum_nonzero_message_ledgers,
        nonzero_message_ledgers,
    )
    stats.maximum_nonzero_coefficient_registers = max(
        stats.maximum_nonzero_coefficient_registers,
        nonzero_coefficients,
    )


def observe_transient(vector: RingVector, stats: UnitStats) -> None:
    stats.maximum_transient_raw_message_payload_bits = max(
        stats.maximum_transient_raw_message_payload_bits,
        vector_payload_bits(vector),
    )
    stats.maximum_transient_raw_message_signed_bits = max(
        stats.maximum_transient_raw_message_signed_bits,
        vector_maximum_signed_bits(vector),
    )


def build_basis(
    carrier: UnitCarrier,
    block: cyclo.CompiledBlock,
    stats: UnitStats,
) -> None:
    cyclo.copy_vector_into(
        carrier.messages[0],
        cyclo.seed_vector(block.public_program),
    )
    block_stats = cyclo.Stats()
    for basis_index in range(1, BASIS_MESSAGES + 1):
        raw = cyclo.apply_operator(
            block.operator,
            carrier.messages[basis_index - 1],
            block_stats,
        )
        observe_transient(raw, stats)
        reduced, ledger = normalized_vector(
            raw,
            carrier.message_ledgers[basis_index - 1],
            stats,
        )
        cyclo.copy_vector_into(
            carrier.messages[basis_index],
            reduced,
        )
        carrier.message_ledgers[basis_index][:] = ledger
    stats.basis_forward_block_applications += BASIS_MESSAGES
    stats.basis_ring_multiply_accumulations += (
        block_stats.ring_multiply_accumulations
    )
    record_peak(carrier, stats)


def load_coefficients(
    carrier: UnitCarrier,
    block: cyclo.CompiledBlock,
    periods: int,
    stats: UnitStats,
) -> None:
    recurrence_stats = recurrence.RecurrenceStats()
    coefficients = recurrence.recurrence_coefficients(
        periods,
        block.characteristic,
        recurrence_stats,
    )
    reduced, ledger = normalized_vector(
        coefficients,
        [0 for _ in range(UNIT_RANK)],
        stats,
    )
    carrier.coefficients[:] = reduced
    carrier.coefficient_ledger[:] = ledger


def raw_output_from_resident_basis(
    carrier: UnitCarrier,
    stats: UnitStats,
) -> RingVector:
    target = cyclo.zero_vector()
    coefficient_scale = ledger_scale(
        carrier.coefficient_ledger,
        stats,
    )
    for basis_index, coefficient in enumerate(
        carrier.coefficients,
        start=1,
    ):
        basis_scale = ledger_scale(
            carrier.message_ledgers[basis_index],
            stats,
        )
        effective = cyclo.ring_multiply(
            cyclo.ring_multiply(coefficient_scale, coefficient),
            basis_scale,
        )
        stats.output_ring_multiplications += 2
        for phase_index, value in enumerate(
            carrier.messages[basis_index]
        ):
            target[phase_index] = cyclo.ring_add(
                target[phase_index],
                cyclo.ring_multiply(effective, value),
            )
            stats.output_ring_multiplications += 1
    observe_transient(target, stats)
    return target


def expected_reduced_output(
    carrier: UnitCarrier,
    stats: UnitStats,
) -> tuple[RingVector, list[int], RingVector]:
    raw = raw_output_from_resident_basis(carrier, stats)
    reduced, ledger = normalized_vector(
        raw,
        [0 for _ in range(UNIT_RANK)],
        stats,
    )
    return reduced, ledger, raw


def project_semantic_output(
    reduced: RingVector,
    ledger: list[int],
    stats: UnitStats,
) -> RingElement:
    semantic = multiply_vector_by_scalar(
        ledger_scale(ledger, stats),
        reduced,
    )
    stats.scale_ring_multiplications += len(reduced)
    return cyclo.project_boundary(semantic)


def populate_forward(
    carrier: UnitCarrier,
    block: cyclo.CompiledBlock,
    periods: int,
) -> tuple[RingElement, UnitStats]:
    if carrier.active or carrier.pending_operations or not carrier.all_zero():
        fail("unit-gauge carrier was not restored")
    carrier.active = True
    carrier.lease += 1
    carrier.pending_operations = 1
    carrier.phase = "BUILD_BASIS"
    stats = UnitStats()
    build_basis(carrier, block, stats)
    load_coefficients(carrier, block, periods, stats)
    carrier.phase = "COEFFICIENTS_RESIDENT"
    reduced, ledger, _ = expected_reduced_output(carrier, stats)
    cyclo.copy_vector_into(carrier.messages[OUTPUT_SLOT], reduced)
    carrier.message_ledgers[OUTPUT_SLOT][:] = ledger
    carrier.phase = "OUTPUT_RESIDENT"
    record_peak(carrier, stats)
    return project_semantic_output(reduced, ledger, stats), stats


def subtract_ledger_exact(
    target: list[int],
    expected: list[int],
) -> None:
    if len(target) != len(expected):
        fail("unit ledger width mismatch")
    for index, value in enumerate(expected):
        target[index] -= value


def restore_forward(
    carrier: UnitCarrier,
    block: cyclo.CompiledBlock,
    periods: int,
    stats: UnitStats,
) -> None:
    if carrier.phase != "OUTPUT_RESIDENT":
        fail("unit-gauge inverse was reordered")
    expected_output, expected_output_ledger, _ = (
        expected_reduced_output(carrier, stats)
    )
    if (
        carrier.messages[OUTPUT_SLOT] != expected_output
        or carrier.message_ledgers[OUTPUT_SLOT]
        != expected_output_ledger
    ):
        fail("unit-gauge inverse output did not match")
    cyclo.subtract_vector_exact(
        carrier.messages[OUTPUT_SLOT],
        expected_output,
    )
    subtract_ledger_exact(
        carrier.message_ledgers[OUTPUT_SLOT],
        expected_output_ledger,
    )
    carrier.phase = "COEFFICIENTS_RESIDENT"

    recurrence_stats = recurrence.RecurrenceStats()
    raw_coefficients = recurrence.recurrence_coefficients(
        periods,
        block.characteristic,
        recurrence_stats,
    )
    expected_coefficients, expected_coefficient_ledger = normalized_vector(
        raw_coefficients,
        [0 for _ in range(UNIT_RANK)],
        stats,
    )
    if (
        carrier.coefficients != expected_coefficients
        or carrier.coefficient_ledger != expected_coefficient_ledger
    ):
        fail("unit-gauge inverse coefficients did not match")
    for index, value in enumerate(expected_coefficients):
        carrier.coefficients[index] = cyclo.ring_subtract(
            carrier.coefficients[index],
            value,
        )
    subtract_ledger_exact(
        carrier.coefficient_ledger,
        expected_coefficient_ledger,
    )
    carrier.phase = "BASIS_RESIDENT"

    block_stats = cyclo.Stats()
    for basis_index in range(BASIS_MESSAGES, 0, -1):
        raw = cyclo.apply_operator(
            block.operator,
            carrier.messages[basis_index - 1],
            block_stats,
        )
        observe_transient(raw, stats)
        expected, expected_ledger = normalized_vector(
            raw,
            carrier.message_ledgers[basis_index - 1],
            stats,
        )
        if carrier.message_ledgers[basis_index] != expected_ledger:
            fail("unit-gauge inverse basis ledger did not match")
        cyclo.subtract_vector_exact(
            carrier.messages[basis_index],
            expected,
        )
        subtract_ledger_exact(
            carrier.message_ledgers[basis_index],
            expected_ledger,
        )
    stats.basis_inverse_block_applications += BASIS_MESSAGES
    stats.basis_ring_multiply_accumulations += (
        block_stats.ring_multiply_accumulations
    )
    cyclo.subtract_vector_exact(
        carrier.messages[0],
        cyclo.seed_vector(block.public_program),
    )
    carrier.pending_operations = 0
    carrier.active = False
    carrier.phase = "RESTORED"
    carrier.generation += 1
    record_peak(carrier, stats)
    if not carrier.all_zero():
        fail("unit-gauge carrier did not restore exactly")


@dataclass
class Transaction:
    boundary: RingElement
    stats: UnitStats
    restored_exactly: bool
    same_backing: bool


def execute_transaction(
    carrier: UnitCarrier,
    block: cyclo.CompiledBlock,
    periods: int,
) -> Transaction:
    if not isinstance(carrier, UnitCarrier):
        fail("null or invalid unit-gauge carrier")
    backing = carrier.backing_identity()
    boundary, stats = populate_forward(carrier, block, periods)
    restore_forward(carrier, block, periods, stats)
    return Transaction(
        boundary=boundary,
        stats=stats,
        restored_exactly=carrier.all_zero(),
        same_backing=carrier.backing_identity() == backing,
    )


def stats_json(stats: UnitStats) -> dict[str, int]:
    return {
        field: getattr(stats, field)
        for field in UnitStats.__dataclass_fields__
    }


def case_result(
    periods: int,
    block: cyclo.CompiledBlock,
) -> dict[str, Any]:
    carrier = UnitCarrier.create()
    transaction = execute_transaction(carrier, block, periods)
    baseline = recurrence.execute_transaction(
        recurrence.Carrier.create(),
        block,
        periods,
    )
    dense_boundary = None
    dense_stats = None
    if periods <= DENSE_DIRECT_PERIOD_CAP:
        dense_boundary, dense_stats = recurrence.dense_boundary(
            block,
            periods,
        )
    return {
        "periods": periods,
        "equivalent_edges": periods * cyclo.PERIOD,
        "family": block.family,
        "boundary": transaction.boundary,
        "boundary_sha256": hashlib.sha256(
            cyclo.encoded_ring_object(transaction.boundary)
        ).hexdigest(),
        "unnormalized_recurrence_boundary_equal": (
            transaction.boundary == baseline.boundary
        ),
        "dense_direct_applicable": dense_boundary is not None,
        "dense_direct_boundary_equal": (
            transaction.boundary == dense_boundary
            if dense_boundary is not None
            else None
        ),
        "dense_direct_stats": dense_stats,
        "message_slots": MESSAGE_SLOTS,
        "message_integer_cells": (
            MESSAGE_SLOTS * MESSAGE_INTEGER_CELLS
        ),
        "message_ledger_integer_cells": MESSAGE_SLOTS * UNIT_RANK,
        "coefficient_register_integer_cells": DIMENSION * DIMENSION,
        "coefficient_ledger_integer_cells": UNIT_RANK,
        "stats": stats_json(transaction.stats),
        "unnormalized_recurrence_stats": recurrence.stats_json(
            baseline.stats
        ),
        "carrier_payload_reduction_bits": (
            baseline.stats.maximum_carrier_payload_bits
            - transaction.stats.maximum_carrier_payload_bits
        ),
        "carrier_payload_reduction_positive": (
            transaction.stats.maximum_carrier_payload_bits
            < baseline.stats.maximum_carrier_payload_bits
        ),
        "restored_exactly": transaction.restored_exactly,
        "same_backing": transaction.same_backing,
        "canonical_restored_state": carrier.canonical_state(),
    }


def restoration_reuse_case(
    primary: cyclo.CompiledBlock,
    reuse: cyclo.CompiledBlock,
) -> dict[str, Any]:
    periods = max(TESTED_PERIODS)
    carrier = UnitCarrier.create()
    backing = carrier.backing_identity()
    primary_transaction = execute_transaction(carrier, primary, periods)
    reuse_transaction = execute_transaction(carrier, reuse, periods)
    fresh_transaction = execute_transaction(
        UnitCarrier.create(),
        reuse,
        periods,
    )
    return {
        "periods": periods,
        "primary_restored_exactly": (
            primary_transaction.restored_exactly
        ),
        "reuse_restored_exactly": reuse_transaction.restored_exactly,
        "same_original_backing": (
            primary_transaction.same_backing
            and reuse_transaction.same_backing
            and carrier.backing_identity() == backing
        ),
        "fresh_restored_reuse_boundary_equal": (
            reuse_transaction.boundary == fresh_transaction.boundary
        ),
        "generation": carrier.generation,
        "lease": carrier.lease,
        "canonical_restored_state": carrier.canonical_state(),
        "retained_forward_inverse_enabling_basis_integer_cells": (
            BASIS_MESSAGES * MESSAGE_INTEGER_CELLS
        ),
        "retained_forward_inverse_enabling_basis_ledger_integer_cells": (
            BASIS_MESSAGES * UNIT_RANK
        ),
        "separate_inverse_operation_log_bytes": 0,
        "baseline_reload_bytes": 0,
    }


def controls(
    primary: cyclo.CompiledBlock,
    reuse: cyclo.CompiledBlock,
) -> dict[str, bool]:
    missing = UnitCarrier.create()
    populate_forward(missing, primary, 4)

    wrong = UnitCarrier.create()
    _, wrong_stats = populate_forward(wrong, primary, 4)
    wrong_inverse_rejected = False
    try:
        restore_forward(wrong, reuse, 4, wrong_stats)
    except RuntimeError:
        wrong_inverse_rejected = True

    reordered = UnitCarrier.create()
    _, reordered_stats = populate_forward(reordered, primary, 4)
    reordered.phase = "BASIS_RESIDENT"
    reordered_inverse_rejected = False
    try:
        restore_forward(reordered, primary, 4, reordered_stats)
    except RuntimeError:
        reordered_inverse_rejected = True

    null_carrier_rejected = False
    try:
        execute_transaction(None, primary, 4)  # type: ignore[arg-type]
    except RuntimeError:
        null_carrier_rejected = True

    corrupted = UnitCarrier.create()
    boundary, corrupted_stats = populate_forward(
        corrupted,
        primary,
        4,
    )
    corrupted.message_ledgers[OUTPUT_SLOT][0] += 1
    corrupted_boundary = project_semantic_output(
        corrupted.messages[OUTPUT_SLOT],
        corrupted.message_ledgers[OUTPUT_SLOT],
        corrupted_stats,
    )
    corrupt_ledger_changes_boundary = corrupted_boundary != boundary
    corrupt_ledger_inverse_rejected = False
    try:
        restore_forward(corrupted, primary, 4, corrupted_stats)
    except RuntimeError:
        corrupt_ledger_inverse_rejected = True

    primary_boundary = execute_transaction(
        UnitCarrier.create(),
        primary,
        4,
    ).boundary
    reuse_boundary = execute_transaction(
        UnitCarrier.create(),
        reuse,
        4,
    ).boundary
    return {
        "unit_generator_inverse_identities_exact": (
            unit_identities_exact()
        ),
        "missing_inverse_leaves_nonzero": not missing.all_zero(),
        "wrong_inverse_rejected": wrong_inverse_rejected,
        "reordered_inverse_rejected": reordered_inverse_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "corrupt_output_ledger_changes_boundary": (
            corrupt_ledger_changes_boundary
        ),
        "corrupt_output_ledger_inverse_rejected": (
            corrupt_ledger_inverse_rejected
        ),
        "semantic_family_perturbation_changes_boundary": (
            primary_boundary != reuse_boundary
        ),
    }


def main() -> int:
    if len(sys.argv) != 1:
        fail(
            "usage: f17_cubic_chain_period17_"
            "unit_height_reduction.py"
        )
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
    control_results = controls(blocks["primary"], blocks["reuse"])
    result = {
        "result": "PASS",
        "experiment": (
            "EXACT_REVERSIBLE_CYCLOTOMIC_UNIT_HEIGHT_REDUCTION_"
            "FOR_NATIVE_K_RECURRENCE_WITH_SCALE_LEDGER_ACCOUNTING"
        ),
        "classification_candidate": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level_candidate": "PACKAGE_SELF_REVIEW",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "coefficient_field": "Q_ZETA17",
        "integral_carrier_ring": "Z_ZETA17",
        "unit_generators": list(UNIT_GENERATOR_INDICES),
        "unit_generator_count": UNIT_RANK,
        "unit_generator_multiplicative_independence_certified": False,
        "normalization_rule": (
            "EXACT_DETERMINISTIC_STRICT_STEEPEST_DESCENT_OVER_"
            "PLUS_OR_MINUS_ONE_GENERATOR_EXPONENT_MOVES"
        ),
        "normalization_step_cap": MAX_NORMALIZATION_STEPS,
        "global_unit_optimum_established": False,
        "block_certificates": {
            family: {
                "public_program_sha256": hashlib.sha256(
                    cyclo.adaptive.encoded_program(block.public_program)
                ).hexdigest(),
                "operator_sha256": block.operator_sha256,
                "characteristic_sha256": (
                    block.characteristic_sha256
                ),
                "characteristic": block.characteristic,
            }
            for family, block in blocks.items()
        },
        "tested_periods": list(TESTED_PERIODS),
        "dense_direct_period_cap": DENSE_DIRECT_PERIOD_CAP,
        "cases": cases,
        "all_recurrence_boundaries_equal": all(
            case["unnormalized_recurrence_boundary_equal"]
            for case in cases
        ),
        "all_applicable_dense_boundaries_equal": all(
            case["dense_direct_boundary_equal"]
            for case in cases
            if case["dense_direct_applicable"]
        ),
        "all_cases_restore_exactly": all(
            case["restored_exactly"]
            and case["same_backing"]
            and case["canonical_restored_state"]["all_state_zero"]
            for case in cases
        ),
        "all_cases_reduce_carrier_payload": all(
            case["carrier_payload_reduction_positive"]
            for case in cases
        ),
        "all_normalization_calls_below_step_cap": all(
            case["stats"]["normalization_step_cap_hits"] == 0
            for case in cases
        ),
        "normalization_step_cap_hits_observed": sum(
            case["stats"]["normalization_step_cap_hits"]
            for case in cases
        ),
        "restoration_reuse_case": restored,
        "controls": control_results,
        "failure_atomic_restoration_established": False,
        "matched_classical": {
            "identical_unit_normalized_recurrence_available": True,
            "identical_unit_ledger_available": True,
            "unnormalized_native_k_recurrence_executed": True,
            "dense_direct_block_execution_through_period64": True,
            "comparison_establishes_advantage": False,
            "strongest_global_unit_reduction_established": False,
            "strongest_family_specific_method_established": False,
        },
        "resource_law": {
            "accounting_scope": (
                "COMPONENT_LEVEL_NAMED_LOGICAL_INTEGER_CELLS_"
                "AND_EXACT_CARRIER_PAYLOAD_NOT_EXACT_PROCESS_PEAK"
            ),
            "resident_message_integer_cells": (
                MESSAGE_SLOTS * MESSAGE_INTEGER_CELLS
            ),
            "resident_message_ledger_integer_cells": (
                MESSAGE_SLOTS * UNIT_RANK
            ),
            "coefficient_register_integer_cells": (
                DIMENSION * DIMENSION
            ),
            "coefficient_ledger_integer_cells": UNIT_RANK,
            "compiled_unit_and_inverse_integer_cells": (
                2 * UNIT_RANK * DIMENSION
            ),
            "compiled_operator_integer_cells_two_families": (
                2 * cyclo.OPERATOR_INTEGER_CELLS
            ),
            "compiled_characteristic_integer_cells_two_families": (
                2 * (PRIME + 1) * DIMENSION
            ),
            "retained_forward_inverse_enabling_basis_integer_cells": (
                BASIS_MESSAGES * MESSAGE_INTEGER_CELLS
            ),
            "retained_forward_inverse_enabling_basis_ledger_integer_cells": (
                BASIS_MESSAGES * UNIT_RANK
            ),
            "normalization_live_named_message_vectors_minimum": 3,
            "normalization_live_message_vector_exact_peak_bounded": (
                False
            ),
            "normalization_candidate_temporary_integer_cells_maximum": (
                MESSAGE_INTEGER_CELLS
            ),
            "scale_temporary_integer_cells": DIMENSION,
            "effective_scalar_temporary_integer_cells": DIMENSION,
            "raw_output_temporary_integer_cells": MESSAGE_INTEGER_CELLS,
            "separate_inverse_operation_log_bytes": 0,
            "baseline_reload_bytes": 0,
            "integer_width_counted": True,
            "ledger_width_counted": True,
            "named_logical_cell_accounting_is_exact_total": False,
            "python_object_overhead_bounded": False,
            "sympy_internal_temporaries_bounded": False,
            "allocator_peak_bounded": False,
            "bit_operation_peak_bounded": False,
            "whole_process_peak_bounded": False,
        },
        "not_established": [
            "GLOBAL_OPTIMAL_CYCLOTOMIC_UNIT_REDUCTION",
            "MULTIPLICATIVE_INDEPENDENCE_OF_THE_SEVEN_UNIT_GENERATORS",
            "STRICT_LOCAL_UNIT_MINIMUM_FOR_EVERY_NORMALIZATION_CALL",
            "FAILURE_ATOMIC_ROLLBACK_AFTER_REJECTED_INVERSE",
            "FIXED_INTEGER_WIDTH",
            "CONSTANT_TOTAL_BIT_STORAGE",
            "INTEGRALLY_REVERSIBLE_ROLLING_WINDOW",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "CATALYTIC_INFERENCE",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_COMPUTATION",
        ],
        "terminal": False,
    }
    if (
        not result["all_recurrence_boundaries_equal"]
        or not result["all_applicable_dense_boundaries_equal"]
        or not result["all_cases_restore_exactly"]
        or not result["all_cases_reduce_carrier_payload"]
        or not all(control_results.values())
        or not restored["primary_restored_exactly"]
        or not restored["reuse_restored_exactly"]
        or not restored["same_original_backing"]
        or not restored["fresh_restored_reuse_boundary_equal"]
    ):
        fail("cyclotomic-unit height-reduction condition failed")
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
