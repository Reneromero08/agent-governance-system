#!/usr/bin/env python3
"""Propose global unit-lattice centers for exact pi-factored recurrence.

The preceding bounded greedy balance stores each message as

    pi**e * residual,  pi = 1 - zeta_17.

It reduces the pi-only residual, but many period-64 calls exhaust 128
single-generator moves and remain wider than the identical raw recurrence.
This successor uses a fixed-precision log-embedding calculation to propose
one global seven-dimensional unit-lattice center per normalization call.
The proposal is not treated as exact.  It is accepted only when the exact
integer trace energy

    sum_j Trace(a_j * conjugate(a_j))
      = sum_embeddings sum_j |sigma(a_j)|**2,

strictly improves.  Accepted residual and ledger updates remain exact in
Z[zeta_17], so proposal approximation cannot change boundary semantics.

The fixed 65,536-bit cosine table, floating solve, exact norm work, unit
materialization, duplicate inverse rematerialization, and matched identical
classical algorithm are all declared.  This is a bounded repair diagnostic,
not a closest-vector proof, global optimum, asymptotic bound, or distinct
phase resource.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from typing import Any

import mpmath as mp
import numpy as np

import f17_cubic_chain_period17_cyclotomic_module as cyclo
import f17_cubic_chain_period17_executed_recurrence as recurrence
import f17_cubic_chain_period17_pi_content_recurrence as pi_content
import f17_cubic_chain_period17_unit_height_reduction as unit_reference


PRIME = cyclo.PRIME
DIMENSION = cyclo.DIMENSION
MESSAGE_SLOTS = recurrence.MESSAGE_SLOTS
BASIS_MESSAGES = recurrence.BASIS_MESSAGES
OUTPUT_SLOT = recurrence.OUTPUT_SLOT
COEFFICIENT_REGISTERS = recurrence.COEFFICIENT_REGISTERS
UNIT_RANK = unit_reference.UNIT_RANK
UNIT_GENERATOR_INDICES = unit_reference.UNIT_GENERATOR_INDICES
UNIT_GENERATORS = unit_reference.UNIT_GENERATORS
UNIT_GENERATOR_INVERSES = unit_reference.UNIT_GENERATOR_INVERSES
TESTED_PERIODS = (1, 64)
EMBEDDING_PRECISION_BITS = 65536
EMBEDDING_CLASSES = tuple(range(1, 9))
EMBEDDING_COSINE_CELLS = 9
EMBEDDING_LOG_MATRIX_FLOAT64_CELLS = 8 * UNIT_RANK
EMBEDDING_SOLVE_FLOAT64_CELLS = 8 + 8 + UNIT_RANK

RingElement = cyclo.RingElement
RingVector = cyclo.RingVector


with mp.workprec(EMBEDDING_PRECISION_BITS):
    EMBEDDING_COSINES = tuple(
        mp.cos(2 * mp.pi * index / PRIME)
        for index in range(EMBEDDING_COSINE_CELLS)
    )


EMBEDDING_LOG_MATRIX = np.array(
    [
        [
            2.0
            * math.log(
                abs(
                    math.sin(math.pi * generator * embedding / PRIME)
                    / math.sin(math.pi * embedding / PRIME)
                )
            )
            for generator in UNIT_GENERATOR_INDICES
        ]
        for embedding in EMBEDDING_CLASSES
    ],
    dtype=np.float64,
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def signed_bits(value: int) -> int:
    return recurrence.signed_bits(value)


def element_payload_bits(element: RingElement) -> int:
    return sum(signed_bits(value) for value in element)


def vector_payload_bits(vector: RingVector) -> int:
    return sum(element_payload_bits(element) for element in vector)


def element_width(element: RingElement) -> int:
    return max(signed_bits(value) for value in element)


def vector_width(vector: RingVector) -> int:
    return max(element_width(element) for element in vector)


def ledger_payload_bits(ledger: list[int]) -> int:
    return sum(signed_bits(value) for value in ledger)


def ring_conjugate(element: RingElement) -> RingElement:
    result = cyclo.ring_zero()
    for exponent, coefficient in enumerate(element):
        monomial = cyclo.ring_monomial((-exponent) % PRIME)
        result = cyclo.ring_add(
            result,
            tuple(coefficient * value for value in monomial),
        )
    return result


def field_trace(element: RingElement) -> int:
    return DIMENSION * element[0] - sum(element[1:])


def element_embedding_energy(element: RingElement) -> int:
    energy = field_trace(
        cyclo.ring_multiply(element, ring_conjugate(element))
    )
    if energy < 0:
        fail("exact embedding energy became negative")
    return energy


def vector_embedding_energy(vector: RingVector) -> int:
    return sum(element_embedding_energy(element) for element in vector)


def vector_norm_element(
    vector: RingVector,
    stats: "BalanceStats | None" = None,
) -> RingElement:
    result = cyclo.ring_zero()
    for element in vector:
        result = cyclo.ring_add(
            result,
            cyclo.ring_multiply(element, ring_conjugate(element)),
        )
        if stats is not None:
            stats.initial_norm_element_ring_multiplications += 1
    return result


def multiply_vector(
    scalar: RingElement,
    vector: RingVector,
) -> RingVector:
    return [
        cyclo.ring_multiply(scalar, element)
        for element in vector
    ]


def ring_power(
    base: RingElement,
    exponent: int,
    stats: "BalanceStats | None" = None,
) -> RingElement:
    if exponent < 0:
        fail("unit power exponent became negative")
    result = cyclo.ring_one()
    factor = base
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = cyclo.ring_multiply(result, factor)
            if stats is not None:
                stats.unit_scale_ring_multiplications += 1
        remaining >>= 1
        if remaining:
            factor = cyclo.ring_multiply(factor, factor)
            if stats is not None:
                stats.unit_scale_ring_multiplications += 1
    return result


def ledger_scale(
    ledger: list[int],
    stats: "BalanceStats | None" = None,
) -> RingElement:
    if len(ledger) != UNIT_RANK:
        fail("unit ledger width changed")
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
        if stats is not None:
            stats.unit_scale_ring_multiplications += 1
    return result


@dataclass
class BalanceStats:
    balance_calls: int = 0
    balance_candidate_evaluations: int = 0
    balance_selected_steps: int = 0
    balance_step_cap_hits: int = 0
    exact_embedding_energy_evaluations: int = 0
    initial_norm_element_ring_multiplications: int = 0
    candidate_norm_ring_multiplications: int = 0
    unit_vector_ring_multiplications: int = 0
    balanced_scalar_vector_ring_multiplications: int = 0
    unit_scale_ring_multiplications: int = 0
    unit_scale_materializations: int = 0
    basis_operator_applications: int = 0
    basis_operator_ring_multiply_accumulations: int = 0
    lattice_center_proposals: int = 0
    lattice_center_proposals_accepted: int = 0
    lattice_center_proposals_rejected: int = 0
    fixed_precision_nonpositive_embedding_rejections: int = 0
    embedding_dot_multiply_accumulations: int = 0
    embedding_log_evaluations: int = 0
    embedding_linear_solves: int = 0
    proposal_commitment_updates: int = 0
    proposal_transcript_commitment: bytes = bytes(32)
    maximum_trace_energy_bits: int = 1
    maximum_norm_element_payload_bits: int = 0
    maximum_proposal_scale_payload_bits: int = 0
    maximum_proposal_norm_factor_payload_bits: int = 0
    maximum_trial_norm_element_payload_bits: int = 0
    maximum_accepted_proposal_vector_payload_bits: int = 0
    maximum_lattice_move_abs: int = 0
    minimum_embedding_log2_floor: int = 0
    maximum_embedding_log2_ceiling: int = 0
    maximum_linear_solve_residual: float = 0.0
    maximum_resident_payload_bits: int = 0
    maximum_duplicate_state_payload_bits: int = 0
    maximum_declared_live_state_payload_bits: int = 0
    maximum_residual_signed_bits: int = 1
    maximum_pi_ledger_signed_bits: int = 1
    maximum_unit_ledger_signed_bits: int = 1
    maximum_nonzero_message_slots: int = 0
    maximum_nonzero_message_pi_ledgers: int = 0
    maximum_nonzero_message_unit_ledgers: int = 0
    maximum_nonzero_coefficient_registers: int = 0
    maximum_nonzero_coefficient_pi_ledgers: int = 0
    maximum_nonzero_coefficient_unit_ledgers: int = 0
    maximum_unit_materialization_payload_bits: int = 0


def exact_vector_energy(
    vector: RingVector,
    stats: BalanceStats,
) -> int:
    stats.exact_embedding_energy_evaluations += 1
    energy = vector_embedding_energy(vector)
    stats.maximum_trace_energy_bits = max(
        stats.maximum_trace_energy_bits,
        max(1, energy.bit_length()),
    )
    return energy


def exact_norm_energy(
    norm_element: RingElement,
    stats: BalanceStats,
) -> int:
    stats.exact_embedding_energy_evaluations += 1
    energy = field_trace(norm_element)
    if energy < 0:
        fail("exact embedding energy became negative")
    stats.maximum_trace_energy_bits = max(
        stats.maximum_trace_energy_bits,
        max(1, energy.bit_length()),
    )
    return energy


def embedding_cosine_index(exponent: int) -> int:
    index = exponent % PRIME
    return index if index <= PRIME // 2 else PRIME - index


def lattice_center_proposal(
    norm_element: RingElement,
    stats: BalanceStats,
) -> list[int] | None:
    stats.lattice_center_proposals += 1
    logs: list[float] = []
    with mp.workprec(EMBEDDING_PRECISION_BITS):
        for embedding in EMBEDDING_CLASSES:
            value = mp.mpf(0)
            for exponent, coefficient in enumerate(norm_element):
                value += (
                    mp.mpf(coefficient)
                    * EMBEDDING_COSINES[
                        embedding_cosine_index(embedding * exponent)
                    ]
                )
                stats.embedding_dot_multiply_accumulations += 1
            if value <= 0:
                stats.fixed_precision_nonpositive_embedding_rejections += 1
                return None
            log_value = mp.log(value)
            stats.embedding_log_evaluations += 1
            log2_value = log_value / mp.log(2)
            stats.minimum_embedding_log2_floor = min(
                stats.minimum_embedding_log2_floor,
                math.floor(float(log2_value)),
            )
            stats.maximum_embedding_log2_ceiling = max(
                stats.maximum_embedding_log2_ceiling,
                math.ceil(float(log2_value)),
            )
            logs.append(float(log_value))
    log_array = np.array(logs, dtype=np.float64)
    target = -log_array + np.mean(log_array)
    solution, _, _, _ = np.linalg.lstsq(
        EMBEDDING_LOG_MATRIX,
        target,
        rcond=None,
    )
    stats.embedding_linear_solves += 1
    residual = float(
        np.max(
            np.abs(
                EMBEDDING_LOG_MATRIX @ solution - target
            )
        )
    )
    stats.maximum_linear_solve_residual = max(
        stats.maximum_linear_solve_residual,
        residual,
    )
    move = [int(value) for value in np.rint(solution)]
    encoded_move = ",".join(str(value) for value in move).encode("ascii")
    stats.proposal_transcript_commitment = hashlib.sha256(
        stats.proposal_transcript_commitment
        + b"|"
        + encoded_move
    ).digest()
    stats.proposal_commitment_updates += 1
    stats.maximum_lattice_move_abs = max(
        stats.maximum_lattice_move_abs,
        *(abs(value) for value in move),
    )
    return move


def balance_vector(
    vector: RingVector,
    base_ledger: list[int],
    stats: BalanceStats,
) -> tuple[RingVector, list[int]]:
    if len(base_ledger) != UNIT_RANK:
        fail("unit ledger width changed")
    if cyclo.vector_is_zero(vector):
        return cyclo.zero_vector(), [0 for _ in range(UNIT_RANK)]
    stats.balance_calls += 1
    current = list(vector)
    current_norm = vector_norm_element(current, stats)
    stats.maximum_norm_element_payload_bits = max(
        stats.maximum_norm_element_payload_bits,
        element_payload_bits(current_norm),
    )
    current_energy = exact_norm_energy(current_norm, stats)
    move = lattice_center_proposal(current_norm, stats)
    stats.balance_candidate_evaluations += 1
    if move is None or not any(move):
        stats.lattice_center_proposals_rejected += 1
        return current, list(base_ledger)
    scale = ledger_scale(move, stats)
    stats.maximum_proposal_scale_payload_bits = max(
        stats.maximum_proposal_scale_payload_bits,
        element_payload_bits(scale),
    )
    norm_factor = cyclo.ring_multiply(
        scale,
        ring_conjugate(scale),
    )
    stats.maximum_proposal_norm_factor_payload_bits = max(
        stats.maximum_proposal_norm_factor_payload_bits,
        element_payload_bits(norm_factor),
    )
    trial_norm = cyclo.ring_multiply(norm_factor, current_norm)
    stats.candidate_norm_ring_multiplications += 2
    stats.maximum_trial_norm_element_payload_bits = max(
        stats.maximum_trial_norm_element_payload_bits,
        element_payload_bits(trial_norm),
    )
    trial_energy = exact_norm_energy(trial_norm, stats)
    if trial_energy >= current_energy:
        stats.lattice_center_proposals_rejected += 1
        return current, list(base_ledger)
    accepted = multiply_vector(scale, current)
    stats.unit_vector_ring_multiplications += len(accepted)
    stats.maximum_accepted_proposal_vector_payload_bits = max(
        stats.maximum_accepted_proposal_vector_payload_bits,
        vector_payload_bits(accepted),
    )
    stats.balance_selected_steps += 1
    stats.lattice_center_proposals_accepted += 1
    return (
        accepted,
        [
            base - delta
            for base, delta in zip(
                base_ledger,
                move,
                strict=True,
            )
        ],
    )


@dataclass(frozen=True)
class BalancedElement:
    residual: RingElement
    pi_exponent: int
    unit_ledger: tuple[int, ...]


@dataclass(frozen=True)
class BalancedVector:
    residual: RingVector
    pi_exponent: int
    unit_ledger: tuple[int, ...]


def zero_element() -> BalancedElement:
    return BalancedElement(
        cyclo.ring_zero(),
        0,
        tuple(0 for _ in range(UNIT_RANK)),
    )


def zero_vector() -> BalancedVector:
    return BalancedVector(
        cyclo.zero_vector(),
        0,
        tuple(0 for _ in range(UNIT_RANK)),
    )


def normalize_balanced_element(
    element: RingElement,
    pi_exponent: int,
    unit_ledger: list[int],
    pi_stats: pi_content.PiStats,
    stats: BalanceStats,
) -> BalancedElement:
    scaled = pi_content.normalize_element(
        element,
        pi_exponent,
        pi_stats,
    )
    if scaled.residual == cyclo.ring_zero():
        return zero_element()
    residual, ledger = balance_vector(
        [scaled.residual],
        unit_ledger,
        stats,
    )
    return BalancedElement(
        residual[0],
        scaled.exponent,
        tuple(ledger),
    )


def normalize_balanced_vector(
    vector: RingVector,
    pi_exponent: int,
    unit_ledger: list[int],
    pi_stats: pi_content.PiStats,
    stats: BalanceStats,
) -> BalancedVector:
    scaled = pi_content.normalize_vector(
        vector,
        pi_exponent,
        pi_stats,
    )
    if cyclo.vector_is_zero(scaled.residual):
        return zero_vector()
    residual, ledger = balance_vector(
        scaled.residual,
        unit_ledger,
        stats,
    )
    return BalancedVector(
        residual,
        scaled.exponent,
        tuple(ledger),
    )


def materialize_unit_vector(
    value: BalancedVector,
    stats: BalanceStats,
) -> RingVector:
    if cyclo.vector_is_zero(value.residual):
        return cyclo.zero_vector()
    scale = ledger_scale(list(value.unit_ledger), stats)
    materialized = multiply_vector(scale, value.residual)
    stats.unit_scale_materializations += 1
    stats.unit_vector_ring_multiplications += len(value.residual)
    stats.maximum_unit_materialization_payload_bits = max(
        stats.maximum_unit_materialization_payload_bits,
        vector_payload_bits(materialized),
    )
    return materialized


def materialize_unit_element(
    value: BalancedElement,
    stats: BalanceStats,
) -> RingElement:
    if value.residual == cyclo.ring_zero():
        return cyclo.ring_zero()
    scale = ledger_scale(list(value.unit_ledger), stats)
    materialized = cyclo.ring_multiply(scale, value.residual)
    stats.unit_scale_materializations += 1
    stats.unit_vector_ring_multiplications += 1
    stats.maximum_unit_materialization_payload_bits = max(
        stats.maximum_unit_materialization_payload_bits,
        element_payload_bits(materialized),
    )
    return materialized


def add_balanced_vectors(
    left: BalancedVector,
    right: BalancedVector,
    pi_stats: pi_content.PiStats,
    stats: BalanceStats,
) -> BalancedVector:
    if cyclo.vector_is_zero(left.residual):
        return right
    if cyclo.vector_is_zero(right.residual):
        return left
    left_scaled = pi_content.ScaledVector(
        materialize_unit_vector(left, stats),
        left.pi_exponent,
    )
    right_scaled = pi_content.ScaledVector(
        materialize_unit_vector(right, stats),
        right.pi_exponent,
    )
    combined = pi_content.scaled_vector_add(
        left_scaled,
        right_scaled,
        pi_stats,
    )
    return normalize_balanced_vector(
        combined.residual,
        combined.exponent,
        [0 for _ in range(UNIT_RANK)],
        pi_stats,
        stats,
    )


def multiply_balanced(
    scalar: BalancedElement,
    vector: BalancedVector,
    pi_stats: pi_content.PiStats,
    stats: BalanceStats,
) -> BalancedVector:
    if (
        scalar.residual == cyclo.ring_zero()
        or cyclo.vector_is_zero(vector.residual)
    ):
        return zero_vector()
    raw = [
        cyclo.ring_multiply(scalar.residual, element)
        for element in vector.residual
    ]
    stats.balanced_scalar_vector_ring_multiplications += len(raw)
    return normalize_balanced_vector(
        raw,
        scalar.pi_exponent + vector.pi_exponent,
        [
            left + right
            for left, right in zip(
                scalar.unit_ledger,
                vector.unit_ledger,
                strict=True,
            )
        ],
        pi_stats,
        stats,
    )


def project_boundary(
    output: BalancedVector,
    pi_stats: pi_content.PiStats,
    stats: BalanceStats,
) -> RingElement:
    unit_materialized = materialize_unit_vector(output, stats)
    projected = cyclo.project_boundary(unit_materialized)
    scaled = pi_content.normalize_element(
        projected,
        output.pi_exponent,
        pi_stats,
    )
    return pi_content.materialize_element(scaled, pi_stats)


@dataclass
class BalancedCarrier:
    messages: list[RingVector]
    message_pi_exponents: list[int]
    message_unit_ledgers: list[list[int]]
    coefficients: list[RingElement]
    coefficient_pi_exponents: list[int]
    coefficient_unit_ledgers: list[list[int]]
    generation: int = 0
    lease: int = 0
    active: bool = False
    pending_operations: int = 0
    phase: str = "RESTORED"

    @classmethod
    def create(cls) -> "BalancedCarrier":
        return cls(
            messages=[
                cyclo.zero_vector()
                for _ in range(MESSAGE_SLOTS)
            ],
            message_pi_exponents=[0 for _ in range(MESSAGE_SLOTS)],
            message_unit_ledgers=[
                [0 for _ in range(UNIT_RANK)]
                for _ in range(MESSAGE_SLOTS)
            ],
            coefficients=[
                cyclo.ring_zero()
                for _ in range(COEFFICIENT_REGISTERS)
            ],
            coefficient_pi_exponents=[
                0 for _ in range(COEFFICIENT_REGISTERS)
            ],
            coefficient_unit_ledgers=[
                [0 for _ in range(UNIT_RANK)]
                for _ in range(COEFFICIENT_REGISTERS)
            ],
        )

    def all_zero(self) -> bool:
        return (
            all(cyclo.vector_is_zero(row) for row in self.messages)
            and not any(self.message_pi_exponents)
            and all(
                not any(ledger)
                for ledger in self.message_unit_ledgers
            )
            and all(
                element == cyclo.ring_zero()
                for element in self.coefficients
            )
            and not any(self.coefficient_pi_exponents)
            and all(
                not any(ledger)
                for ledger in self.coefficient_unit_ledgers
            )
        )

    def backing_identity(self) -> tuple[int, ...]:
        return (
            id(self.messages),
            *(id(row) for row in self.messages),
            id(self.message_pi_exponents),
            id(self.message_unit_ledgers),
            *(id(row) for row in self.message_unit_ledgers),
            id(self.coefficients),
            id(self.coefficient_pi_exponents),
            id(self.coefficient_unit_ledgers),
            *(id(row) for row in self.coefficient_unit_ledgers),
        )

    def canonical_state(self) -> dict[str, Any]:
        return {
            "message_slots": len(self.messages),
            "message_pi_ledger_cells": len(self.message_pi_exponents),
            "message_unit_ledger_cells": (
                len(self.message_unit_ledgers) * UNIT_RANK
            ),
            "coefficient_registers": len(self.coefficients),
            "coefficient_pi_ledger_cells": (
                len(self.coefficient_pi_exponents)
            ),
            "coefficient_unit_ledger_cells": (
                len(self.coefficient_unit_ledgers) * UNIT_RANK
            ),
            "all_payload_and_ledgers_zero": self.all_zero(),
            "generation": self.generation,
            "lease": self.lease,
            "active": self.active,
            "pending_operations": self.pending_operations,
            "phase": self.phase,
        }


def carrier_message(
    carrier: BalancedCarrier,
    index: int,
) -> BalancedVector:
    return BalancedVector(
        list(carrier.messages[index]),
        carrier.message_pi_exponents[index],
        tuple(carrier.message_unit_ledgers[index]),
    )


def carrier_coefficient(
    carrier: BalancedCarrier,
    index: int,
) -> BalancedElement:
    return BalancedElement(
        carrier.coefficients[index],
        carrier.coefficient_pi_exponents[index],
        tuple(carrier.coefficient_unit_ledgers[index]),
    )


def store_message(
    carrier: BalancedCarrier,
    index: int,
    value: BalancedVector,
) -> None:
    cyclo.copy_vector_into(carrier.messages[index], value.residual)
    carrier.message_pi_exponents[index] = value.pi_exponent
    carrier.message_unit_ledgers[index][:] = value.unit_ledger


def store_coefficient(
    carrier: BalancedCarrier,
    index: int,
    value: BalancedElement,
) -> None:
    carrier.coefficients[index] = value.residual
    carrier.coefficient_pi_exponents[index] = value.pi_exponent
    carrier.coefficient_unit_ledgers[index][:] = value.unit_ledger


def record_peak(
    carrier: BalancedCarrier,
    stats: BalanceStats,
) -> None:
    payload = 0
    residual_width = 1
    pi_width = 1
    unit_width = 1
    nonzero_messages = 0
    nonzero_message_pi = 0
    nonzero_message_units = 0
    nonzero_coefficients = 0
    nonzero_coefficient_pi = 0
    nonzero_coefficient_units = 0
    for residual, pi_exponent, ledger in zip(
        carrier.messages,
        carrier.message_pi_exponents,
        carrier.message_unit_ledgers,
        strict=True,
    ):
        payload += vector_payload_bits(residual)
        payload += signed_bits(pi_exponent)
        payload += ledger_payload_bits(ledger)
        residual_width = max(residual_width, vector_width(residual))
        pi_width = max(pi_width, signed_bits(pi_exponent))
        unit_width = max(
            unit_width,
            *(signed_bits(value) for value in ledger),
        )
        nonzero_messages += int(not cyclo.vector_is_zero(residual))
        nonzero_message_pi += int(pi_exponent != 0)
        nonzero_message_units += int(any(ledger))
    for residual, pi_exponent, ledger in zip(
        carrier.coefficients,
        carrier.coefficient_pi_exponents,
        carrier.coefficient_unit_ledgers,
        strict=True,
    ):
        payload += element_payload_bits(residual)
        payload += signed_bits(pi_exponent)
        payload += ledger_payload_bits(ledger)
        residual_width = max(residual_width, element_width(residual))
        pi_width = max(pi_width, signed_bits(pi_exponent))
        unit_width = max(
            unit_width,
            *(signed_bits(value) for value in ledger),
        )
        nonzero_coefficients += int(residual != cyclo.ring_zero())
        nonzero_coefficient_pi += int(pi_exponent != 0)
        nonzero_coefficient_units += int(any(ledger))
    stats.maximum_resident_payload_bits = max(
        stats.maximum_resident_payload_bits,
        payload,
    )
    stats.maximum_residual_signed_bits = max(
        stats.maximum_residual_signed_bits,
        residual_width,
    )
    stats.maximum_pi_ledger_signed_bits = max(
        stats.maximum_pi_ledger_signed_bits,
        pi_width,
    )
    stats.maximum_unit_ledger_signed_bits = max(
        stats.maximum_unit_ledger_signed_bits,
        unit_width,
    )
    stats.maximum_nonzero_message_slots = max(
        stats.maximum_nonzero_message_slots,
        nonzero_messages,
    )
    stats.maximum_nonzero_message_pi_ledgers = max(
        stats.maximum_nonzero_message_pi_ledgers,
        nonzero_message_pi,
    )
    stats.maximum_nonzero_message_unit_ledgers = max(
        stats.maximum_nonzero_message_unit_ledgers,
        nonzero_message_units,
    )
    stats.maximum_nonzero_coefficient_registers = max(
        stats.maximum_nonzero_coefficient_registers,
        nonzero_coefficients,
    )
    stats.maximum_nonzero_coefficient_pi_ledgers = max(
        stats.maximum_nonzero_coefficient_pi_ledgers,
        nonzero_coefficient_pi,
    )
    stats.maximum_nonzero_coefficient_unit_ledgers = max(
        stats.maximum_nonzero_coefficient_unit_ledgers,
        nonzero_coefficient_units,
    )


def expected_state_payload_bits(
    messages: list[BalancedVector],
    coefficients: list[BalancedElement],
) -> int:
    return sum(
        vector_payload_bits(value.residual)
        + signed_bits(value.pi_exponent)
        + ledger_payload_bits(list(value.unit_ledger))
        for value in messages
    ) + sum(
        element_payload_bits(value.residual)
        + signed_bits(value.pi_exponent)
        + ledger_payload_bits(list(value.unit_ledger))
        for value in coefficients
    )


def record_expected_state_peak(
    messages: list[BalancedVector],
    coefficients: list[BalancedElement],
    carrier: BalancedCarrier,
    stats: BalanceStats,
) -> None:
    duplicate_payload = expected_state_payload_bits(
        messages,
        coefficients,
    )
    stats.maximum_duplicate_state_payload_bits = max(
        stats.maximum_duplicate_state_payload_bits,
        duplicate_payload,
    )
    record_peak(carrier, stats)
    stats.maximum_declared_live_state_payload_bits = max(
        stats.maximum_declared_live_state_payload_bits,
        duplicate_payload + stats.maximum_resident_payload_bits,
    )


def build_expected_state(
    block: cyclo.CompiledBlock,
    periods: int,
    pi_stats: pi_content.PiStats,
    stats: BalanceStats,
) -> tuple[list[BalancedVector], list[BalancedElement]]:
    seed = normalize_balanced_vector(
        cyclo.seed_vector(block.public_program),
        0,
        [0 for _ in range(UNIT_RANK)],
        pi_stats,
        stats,
    )
    messages = [seed]
    block_stats = cyclo.Stats()
    for _ in range(BASIS_MESSAGES):
        previous = messages[-1]
        raw = cyclo.apply_operator(
            block.operator,
            previous.residual,
            block_stats,
        )
        messages.append(
            normalize_balanced_vector(
                raw,
                previous.pi_exponent,
                list(previous.unit_ledger),
                pi_stats,
                stats,
            )
        )
    scaled_coefficients = pi_content.scaled_recurrence_coefficients(
        periods,
        block.characteristic,
        pi_stats,
    )
    coefficients = [
        normalize_balanced_element(
            value.residual,
            value.exponent,
            [0 for _ in range(UNIT_RANK)],
            pi_stats,
            stats,
        )
        for value in scaled_coefficients
    ]
    output = zero_vector()
    for coefficient, basis in zip(
        coefficients,
        messages[1:],
        strict=True,
    ):
        output = add_balanced_vectors(
            output,
            multiply_balanced(
                coefficient,
                basis,
                pi_stats,
                stats,
            ),
            pi_stats,
            stats,
        )
    messages.append(output)
    stats.basis_operator_applications += BASIS_MESSAGES
    stats.basis_operator_ring_multiply_accumulations += (
        block_stats.ring_multiply_accumulations
    )
    return messages, coefficients


def populate_forward(
    carrier: BalancedCarrier,
    block: cyclo.CompiledBlock,
    periods: int,
) -> tuple[RingElement, pi_content.PiStats, BalanceStats]:
    if carrier.active or carrier.pending_operations or not carrier.all_zero():
        fail("balanced carrier was not restored")
    carrier.active = True
    carrier.lease += 1
    carrier.pending_operations = 1
    carrier.phase = "BUILD_BALANCED_STATE"
    pi_stats = pi_content.PiStats()
    stats = BalanceStats()
    messages, coefficients = build_expected_state(
        block,
        periods,
        pi_stats,
        stats,
    )
    for index, value in enumerate(messages):
        store_message(carrier, index, value)
    for index, value in enumerate(coefficients):
        store_coefficient(carrier, index, value)
    carrier.phase = "BALANCED_OUTPUT_RESIDENT"
    record_expected_state_peak(
        messages,
        coefficients,
        carrier,
        stats,
    )
    return (
        project_boundary(messages[-1], pi_stats, stats),
        pi_stats,
        stats,
    )


def subtract_message(
    carrier: BalancedCarrier,
    index: int,
    expected: BalancedVector,
) -> None:
    if (
        carrier.message_pi_exponents[index] != expected.pi_exponent
        or tuple(carrier.message_unit_ledgers[index])
        != expected.unit_ledger
        or carrier.messages[index] != expected.residual
    ):
        fail("balanced message inverse rematerialization mismatch")
    cyclo.subtract_vector_exact(
        carrier.messages[index],
        expected.residual,
    )
    carrier.message_pi_exponents[index] -= expected.pi_exponent
    carrier.message_unit_ledgers[index][:] = [
        actual - value
        for actual, value in zip(
            carrier.message_unit_ledgers[index],
            expected.unit_ledger,
            strict=True,
        )
    ]


def restore_forward(
    carrier: BalancedCarrier,
    block: cyclo.CompiledBlock,
    periods: int,
    pi_stats: pi_content.PiStats,
    stats: BalanceStats,
) -> tuple[pi_content.PiStats, BalanceStats]:
    if carrier.phase != "BALANCED_OUTPUT_RESIDENT":
        fail("balanced inverse was reordered")
    inverse_pi_stats = pi_content.PiStats()
    inverse_stats = BalanceStats()
    messages, coefficients = build_expected_state(
        block,
        periods,
        inverse_pi_stats,
        inverse_stats,
    )
    record_expected_state_peak(
        messages,
        coefficients,
        carrier,
        inverse_stats,
    )
    subtract_message(carrier, OUTPUT_SLOT, messages[-1])
    carrier.phase = "BALANCED_COEFFICIENTS_RESIDENT"
    for index, expected in enumerate(coefficients):
        if (
            carrier.coefficient_pi_exponents[index]
            != expected.pi_exponent
            or tuple(carrier.coefficient_unit_ledgers[index])
            != expected.unit_ledger
            or carrier.coefficients[index] != expected.residual
        ):
            fail("balanced coefficient inverse rematerialization mismatch")
        carrier.coefficients[index] = cyclo.ring_subtract(
            carrier.coefficients[index],
            expected.residual,
        )
        carrier.coefficient_pi_exponents[index] -= expected.pi_exponent
        carrier.coefficient_unit_ledgers[index][:] = [
            actual - value
            for actual, value in zip(
                carrier.coefficient_unit_ledgers[index],
                expected.unit_ledger,
                strict=True,
            )
        ]
    carrier.phase = "BALANCED_BASIS_RESIDENT"
    for index in range(BASIS_MESSAGES, 0, -1):
        subtract_message(carrier, index, messages[index])
    subtract_message(carrier, 0, messages[0])
    carrier.pending_operations = 0
    carrier.active = False
    carrier.phase = "RESTORED"
    carrier.generation += 1
    record_peak(carrier, stats)
    if not carrier.all_zero():
        fail("balanced carrier did not restore exactly")
    return inverse_pi_stats, inverse_stats


@dataclass
class Transaction:
    boundary: RingElement
    pi_stats: pi_content.PiStats
    stats: BalanceStats
    inverse_pi_stats: pi_content.PiStats
    inverse_stats: BalanceStats
    restored_exactly: bool
    same_backing: bool


def execute_transaction(
    carrier: BalancedCarrier,
    block: cyclo.CompiledBlock,
    periods: int,
) -> Transaction:
    if not isinstance(carrier, BalancedCarrier):
        fail("null or invalid balanced carrier")
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
        pi_stats,
        stats,
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


def stats_json(
    stats: BalanceStats,
    pi_stats: pi_content.PiStats,
) -> dict[str, int | float | str]:
    return {
        "balance_calls": stats.balance_calls,
        "balance_candidate_evaluations": (
            stats.balance_candidate_evaluations
        ),
        "balance_selected_steps": stats.balance_selected_steps,
        "balance_step_cap_hits": stats.balance_step_cap_hits,
        "exact_embedding_energy_evaluations": (
            stats.exact_embedding_energy_evaluations
        ),
        "initial_norm_element_ring_multiplications": (
            stats.initial_norm_element_ring_multiplications
        ),
        "candidate_norm_ring_multiplications": (
            stats.candidate_norm_ring_multiplications
        ),
        "unit_vector_ring_multiplications": (
            stats.unit_vector_ring_multiplications
        ),
        "balanced_scalar_vector_ring_multiplications": (
            stats.balanced_scalar_vector_ring_multiplications
        ),
        "unit_scale_ring_multiplications": (
            stats.unit_scale_ring_multiplications
        ),
        "unit_scale_materializations": (
            stats.unit_scale_materializations
        ),
        "basis_operator_applications": (
            stats.basis_operator_applications
        ),
        "basis_operator_ring_multiply_accumulations": (
            stats.basis_operator_ring_multiply_accumulations
        ),
        "lattice_center_proposals": stats.lattice_center_proposals,
        "lattice_center_proposals_accepted": (
            stats.lattice_center_proposals_accepted
        ),
        "lattice_center_proposals_rejected": (
            stats.lattice_center_proposals_rejected
        ),
        "fixed_precision_nonpositive_embedding_rejections": (
            stats.fixed_precision_nonpositive_embedding_rejections
        ),
        "embedding_dot_multiply_accumulations": (
            stats.embedding_dot_multiply_accumulations
        ),
        "embedding_log_evaluations": (
            stats.embedding_log_evaluations
        ),
        "embedding_linear_solves": stats.embedding_linear_solves,
        "proposal_commitment_updates": (
            stats.proposal_commitment_updates
        ),
        "proposal_transcript_sha256": (
            stats.proposal_transcript_commitment.hex()
        ),
        "maximum_trace_energy_bits": stats.maximum_trace_energy_bits,
        "maximum_norm_element_payload_bits": (
            stats.maximum_norm_element_payload_bits
        ),
        "maximum_proposal_scale_payload_bits": (
            stats.maximum_proposal_scale_payload_bits
        ),
        "maximum_proposal_norm_factor_payload_bits": (
            stats.maximum_proposal_norm_factor_payload_bits
        ),
        "maximum_trial_norm_element_payload_bits": (
            stats.maximum_trial_norm_element_payload_bits
        ),
        "maximum_accepted_proposal_vector_payload_bits": (
            stats.maximum_accepted_proposal_vector_payload_bits
        ),
        "maximum_lattice_move_abs": stats.maximum_lattice_move_abs,
        "minimum_embedding_log2_floor": (
            stats.minimum_embedding_log2_floor
        ),
        "maximum_embedding_log2_ceiling": (
            stats.maximum_embedding_log2_ceiling
        ),
        "maximum_linear_solve_residual": (
            stats.maximum_linear_solve_residual
        ),
        "maximum_resident_payload_bits": (
            stats.maximum_resident_payload_bits
        ),
        "maximum_duplicate_state_payload_bits": (
            stats.maximum_duplicate_state_payload_bits
        ),
        "maximum_declared_live_state_payload_bits": (
            stats.maximum_declared_live_state_payload_bits
        ),
        "maximum_residual_signed_bits": (
            stats.maximum_residual_signed_bits
        ),
        "maximum_pi_ledger_signed_bits": (
            stats.maximum_pi_ledger_signed_bits
        ),
        "maximum_unit_ledger_signed_bits": (
            stats.maximum_unit_ledger_signed_bits
        ),
        "maximum_nonzero_message_slots": (
            stats.maximum_nonzero_message_slots
        ),
        "maximum_nonzero_message_pi_ledgers": (
            stats.maximum_nonzero_message_pi_ledgers
        ),
        "maximum_nonzero_message_unit_ledgers": (
            stats.maximum_nonzero_message_unit_ledgers
        ),
        "maximum_nonzero_coefficient_registers": (
            stats.maximum_nonzero_coefficient_registers
        ),
        "maximum_nonzero_coefficient_pi_ledgers": (
            stats.maximum_nonzero_coefficient_pi_ledgers
        ),
        "maximum_nonzero_coefficient_unit_ledgers": (
            stats.maximum_nonzero_coefficient_unit_ledgers
        ),
        "maximum_unit_materialization_payload_bits": (
            stats.maximum_unit_materialization_payload_bits
        ),
        "exact_pi_divisions": pi_stats.exact_pi_divisions,
        "polynomial_multiplications": pi_stats.polynomial_multiplications,
        "scaled_element_multiplications": (
            pi_stats.scaled_element_multiplications
        ),
        "polynomial_reduction_updates": (
            pi_stats.polynomial_reduction_updates
        ),
    }


def case_result(
    periods: int,
    block: cyclo.CompiledBlock,
) -> dict[str, Any]:
    carrier = BalancedCarrier.create()
    transaction = execute_transaction(carrier, block, periods)
    pi_case = pi_content.case_result(periods, block)
    balanced_stats = stats_json(
        transaction.stats,
        transaction.pi_stats,
    )
    inverse_stats = stats_json(
        transaction.inverse_stats,
        transaction.inverse_pi_stats,
    )
    raw_payload = pi_case["raw_recurrence_baseline"][
        "maximum_carrier_payload_bits"
    ]
    pi_payload = pi_case["pi_content_stats"][
        "maximum_carrier_payload_bits"
    ]
    balanced_payload = balanced_stats["maximum_resident_payload_bits"]
    declared_live_payload = balanced_stats[
        "maximum_declared_live_state_payload_bits"
    ]
    compiled_cosine_nominal_mantissa_bits = (
        EMBEDDING_COSINE_CELLS * EMBEDDING_PRECISION_BITS
    )
    compiled_log_matrix_bits = (
        EMBEDDING_LOG_MATRIX_FLOAT64_CELLS * 64
    )
    solve_scratch_bits = EMBEDDING_SOLVE_FLOAT64_CELLS * 64
    fixed_precision_scalar_working_bits = (
        2 * EMBEDDING_PRECISION_BITS
    )
    named_exact_temporary_maxima_sum_bits = sum(
        balanced_stats[field]
        for field in (
            "maximum_norm_element_payload_bits",
            "maximum_proposal_scale_payload_bits",
            "maximum_proposal_norm_factor_payload_bits",
            "maximum_trial_norm_element_payload_bits",
            "maximum_accepted_proposal_vector_payload_bits",
            "maximum_unit_materialization_payload_bits",
        )
    )
    named_component_maxima_sum_bits = (
        declared_live_payload
        + compiled_cosine_nominal_mantissa_bits
        + compiled_log_matrix_bits
        + solve_scratch_bits
        + fixed_precision_scalar_working_bits
        + 256
        + named_exact_temporary_maxima_sum_bits
    )
    boundary_sha256 = hashlib.sha256(
        cyclo.encoded_ring_object(transaction.boundary)
    ).hexdigest()
    return {
        "periods": periods,
        "family": block.family,
        "equivalent_edges": periods * cyclo.PERIOD,
        "boundary": transaction.boundary,
        "boundary_sha256": boundary_sha256,
        "raw_recurrence_boundary_equal": (
            pi_case["raw_recurrence_boundary_equal"]
            and boundary_sha256 == pi_case["boundary_sha256"]
        ),
        "pi_content_boundary_equal": (
            boundary_sha256 == pi_case["boundary_sha256"]
        ),
        "raw_recurrence_payload_bits": raw_payload,
        "pi_content_payload_bits": pi_payload,
        "balanced_payload_bits": balanced_payload,
        "balanced_declared_live_state_payload_bits": (
            declared_live_payload
        ),
        "compiled_cosine_nominal_mantissa_bits": (
            compiled_cosine_nominal_mantissa_bits
        ),
        "compiled_log_matrix_float64_bits": compiled_log_matrix_bits,
        "embedding_solve_scratch_float64_bits": solve_scratch_bits,
        "fixed_precision_scalar_working_bits": (
            fixed_precision_scalar_working_bits
        ),
        "proposal_transcript_commitment_bits": 256,
        "named_exact_temporary_maxima_sum_bits": (
            named_exact_temporary_maxima_sum_bits
        ),
        "named_component_maxima_sum_bits": (
            named_component_maxima_sum_bits
        ),
        "balanced_minus_pi_content_payload_bits": (
            balanced_payload - pi_payload
        ),
        "balanced_minus_raw_payload_bits": (
            balanced_payload - raw_payload
        ),
        "balanced_declared_live_minus_pi_content_payload_bits": (
            declared_live_payload - pi_payload
        ),
        "balanced_declared_live_minus_raw_payload_bits": (
            declared_live_payload - raw_payload
        ),
        "named_component_maxima_sum_minus_pi_content_payload_bits": (
            named_component_maxima_sum_bits - pi_payload
        ),
        "named_component_maxima_sum_minus_raw_payload_bits": (
            named_component_maxima_sum_bits - raw_payload
        ),
        "balanced_reduces_pi_content_payload": (
            balanced_payload < pi_payload
        ),
        "balanced_beats_raw_recurrence_payload": (
            balanced_payload < raw_payload
        ),
        "balanced_declared_live_reduces_pi_content_payload": (
            declared_live_payload < pi_payload
        ),
        "balanced_declared_live_beats_raw_recurrence_payload": (
            declared_live_payload < raw_payload
        ),
        "named_component_maxima_sum_beats_pi_content_payload": (
            named_component_maxima_sum_bits < pi_payload
        ),
        "named_component_maxima_sum_beats_raw_recurrence_payload": (
            named_component_maxima_sum_bits < raw_payload
        ),
        "balanced_stats": balanced_stats,
        "inverse_rematerialization_stats": inverse_stats,
        "restored_exactly": transaction.restored_exactly,
        "same_backing": transaction.same_backing,
        "canonical_restored_state": carrier.canonical_state(),
    }


def restoration_reuse_case(
    primary: cyclo.CompiledBlock,
    reuse: cyclo.CompiledBlock,
) -> dict[str, Any]:
    carrier = BalancedCarrier.create()
    backing = carrier.backing_identity()
    primary_transaction = execute_transaction(carrier, primary, 1)
    reuse_transaction = execute_transaction(carrier, reuse, 1)
    fresh = execute_transaction(
        BalancedCarrier.create(),
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
    missing = BalancedCarrier.create()
    populate_forward(missing, primary, 1)
    missing_inverse = not missing.all_zero()

    reordered = BalancedCarrier.create()
    reordered_rejected = False
    try:
        restore_forward(
            reordered,
            primary,
            1,
            pi_content.PiStats(),
            BalanceStats(),
        )
    except RuntimeError:
        reordered_rejected = True

    wrong = BalancedCarrier.create()
    _, pi_stats, stats = populate_forward(wrong, primary, 1)
    wrong_rejected = False
    try:
        restore_forward(wrong, reuse, 1, pi_stats, stats)
    except RuntimeError:
        wrong_rejected = True

    null_rejected = False
    try:
        execute_transaction(None, primary, 1)  # type: ignore[arg-type]
    except RuntimeError:
        null_rejected = True

    primary_boundary = execute_transaction(
        BalancedCarrier.create(),
        primary,
        1,
    ).boundary
    reuse_boundary = execute_transaction(
        BalancedCarrier.create(),
        reuse,
        1,
    ).boundary
    return {
        "missing_inverse_leaves_nonzero": missing_inverse,
        "reordered_inverse_rejected": reordered_rejected,
        "wrong_inverse_rejected": wrong_rejected,
        "null_carrier_rejected": null_rejected,
        "semantic_family_perturbation_changes_boundary": (
            primary_boundary != reuse_boundary
        ),
    }


def exact_embedding_control() -> dict[str, bool]:
    samples = [
        cyclo.ring_one(),
        *UNIT_GENERATORS,
        *UNIT_GENERATOR_INVERSES,
    ]
    sample_vector = [
        cyclo.ring_add(
            cyclo.ring_one(),
            cyclo.ring_monomial(3),
        ),
        cyclo.ring_add(
            cyclo.ring_monomial(2),
            cyclo.ring_monomial(7),
        ),
    ]
    sample_norm = vector_norm_element(sample_vector)
    moved_sample = multiply_vector(
        UNIT_GENERATOR_INVERSES[0],
        sample_vector,
    )
    reconstructed_sample = multiply_vector(
        ledger_scale([1, 0, 0, 0, 0, 0, 0]),
        moved_sample,
    )
    proposal_stats = BalanceStats()
    sample_move = lattice_center_proposal(
        sample_norm,
        proposal_stats,
    )
    return {
        "unit_identities_exact": unit_reference.unit_identities_exact(),
        "conjugation_involution_exact": all(
            ring_conjugate(ring_conjugate(element)) == element
            for element in samples
        ),
        "trace_energy_positive_on_nonzero_samples": all(
            element_embedding_energy(element) > 0
            for element in samples
        ),
        "trace_energy_conjugation_invariant": all(
            element_embedding_energy(ring_conjugate(element))
            == element_embedding_energy(element)
            for element in samples
        ),
        "norm_element_trace_matches_direct_energy": (
            field_trace(sample_norm)
            == vector_embedding_energy(sample_vector)
        ),
        "norm_element_unit_update_exact": (
            vector_norm_element(moved_sample)
            == cyclo.ring_multiply(
                cyclo.ring_multiply(
                    UNIT_GENERATOR_INVERSES[0],
                    ring_conjugate(UNIT_GENERATOR_INVERSES[0]),
                ),
                sample_norm,
            )
        ),
        "unit_move_and_ledger_reconstruct_vector_exactly": (
            reconstructed_sample == sample_vector
        ),
        "embedding_log_matrix_has_declared_rank": (
            int(np.linalg.matrix_rank(EMBEDDING_LOG_MATRIX))
            == UNIT_RANK
        ),
        "compiled_cosine_table_is_finite": all(
            mp.isfinite(value)
            for value in EMBEDDING_COSINES
        ),
        "fixed_precision_proposal_sample_is_defined": (
            sample_move is not None
            and not proposal_stats.fixed_precision_nonpositive_embedding_rejections
        ),
    }


def main() -> int:
    if len(sys.argv) != 1:
        fail(
            "usage: f17_cubic_chain_period17_"
            "pi_unit_lattice_center.py"
        )
    blocks = {
        family.lower(): cyclo.build_compiled_block(family)
        for family in ("PRIMARY", "REUSE")
    }
    embedding_controls = exact_embedding_control()
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
    result = {
        "result": "PASS",
        "experiment": (
            "FIXED_PRECISION_LOG_EMBEDDING_UNIT_LATTICE_CENTER_"
            "PROPOSAL_WITH_EXACT_TRACE_ACCEPTANCE"
        ),
        "claim_candidate": (
            "BOUNDED_65536_BIT_LOG_EMBEDDING_UNIT_LATTICE_CENTER_"
            "PROPOSAL_WITH_EXACT_TRACE_ACCEPTANCE_REDUCES_"
            "PI_FACTORED_RESIDENT_AND_DUPLICATE_REMATERIALIZATION_"
            "LIVE_PAYLOAD_"
            "BELOW_THE_IDENTICAL_RAW_RECURRENCE_FOR_TWO_PUBLIC_F17_"
            "PERIOD17_FAMILIES_AT_PERIODS1_AND64_WITH_EXACT_"
            "RESTORATION_AND_REUSE_BUT_THE_CONSERVATIVE_DECLARED_"
            "NAMED_COMPONENT_MAXIMA_SUM_REMAINS_ABOVE_RAW_AND_THE_"
            "IDENTICAL_CLASSICAL_EXECUTION_REMAINS"
        ),
        "classification_candidate": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level_candidate": "PACKAGE_SELF_REVIEW",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "coefficient_field": "Q_ZETA17",
        "integral_carrier_ring": "Z_ZETA17",
        "uniformizer": "PI_EQUALS_1_MINUS_ZETA17",
        "unit_generator_indices": list(
            unit_reference.UNIT_GENERATOR_INDICES
        ),
        "unit_generator_multiplicative_independence_certified": False,
        "exact_multi_embedding_objective": (
            "SUM_OVER_ALL_16_EMBEDDINGS_OF_VECTOR_SQUARED_MAGNITUDE_"
            "EQUALS_FIELD_TRACE_OF_A_TIMES_CONJUGATE_A"
        ),
        "embedding_proposal_precision_bits": EMBEDDING_PRECISION_BITS,
        "embedding_proposal_numeric_type": (
            "MPMATH_FIXED_PRECISION_DOT_AND_LOG_THEN_FLOAT64_LEAST_SQUARES"
        ),
        "embedding_proposal_is_exact": False,
        "embedding_proposal_is_certified_closest_vector": False,
        "proposal_count_per_nonzero_normalization_call": 1,
        "exact_trace_acceptance_controls_carrier_update": True,
        "tested_periods": list(TESTED_PERIODS),
        "block_certificates": {
            family: {
                "public_program_sha256": hashlib.sha256(
                    cyclo.adaptive.encoded_program(block.public_program)
                ).hexdigest(),
                "operator_sha256": block.operator_sha256,
                "characteristic_sha256": (
                    block.characteristic_sha256
                ),
                "characteristic_identity_exact": (
                    block.characteristic_identity_exact
                ),
                "characteristic": block.characteristic,
            }
            for family, block in blocks.items()
        },
        "exact_embedding_controls": embedding_controls,
        "cases": cases,
        "all_raw_recurrence_boundaries_equal": all(
            case["raw_recurrence_boundary_equal"]
            for case in cases
        ),
        "all_pi_content_boundaries_equal": all(
            case["pi_content_boundary_equal"]
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
        "all_cases_reduce_pi_content_payload": all(
            case["balanced_reduces_pi_content_payload"]
            for case in cases
        ),
        "all_cases_beat_raw_recurrence_payload": all(
            case["balanced_beats_raw_recurrence_payload"]
            for case in cases
        ),
        "all_declared_live_cases_reduce_pi_content_payload": all(
            case["balanced_declared_live_reduces_pi_content_payload"]
            for case in cases
        ),
        "all_declared_live_cases_beat_raw_recurrence_payload": all(
            case["balanced_declared_live_beats_raw_recurrence_payload"]
            for case in cases
        ),
        "all_fixed_precision_embedding_evaluations_positive": all(
            case["balanced_stats"][
                "fixed_precision_nonpositive_embedding_rejections"
            ] == 0
            and case["inverse_rematerialization_stats"][
                "fixed_precision_nonpositive_embedding_rejections"
            ] == 0
            for case in cases
        ),
        "all_named_component_maxima_sums_remain_above_raw": all(
            not case[
                "named_component_maxima_sum_beats_raw_recurrence_payload"
            ]
            for case in cases
        ),
        "restoration_reuse_case": restored,
        "controls": control_results,
        "matched_classical": {
            "identical_fixed_precision_proposal_recurrence_available": True,
            "identical_exact_trace_energy_objective_available": True,
            "raw_recurrence_retained": True,
            "pi_content_recurrence_retained": True,
            "comparison_establishes_advantage": False,
        },
        "observation": (
            "A_FIXED_PRECISION_GLOBAL_LOG_EMBEDDING_PROPOSAL_WITH_"
            "EXACT_INTEGER_ACCEPTANCE_REDUCES_THE_PI_FACTORED_"
            "RESIDENT_AND_DECLARED_LIVE_PAYLOAD_BELOW_THE_RAW_"
            "RECURRENCE_IN_THE_TESTED_CASES_BUT_THE_CONSERVATIVE_"
            "SUM_OF_SEPARATELY_OBSERVED_NAMED_COMPONENT_MAXIMA_"
            "REMAINS_ABOVE_RAW_THE_IDENTICAL_ALGORITHM_IS_AVAILABLE_"
            "TO_COMPACT_CLASSICAL_SOFTWARE_AND_PROCESS_LEVEL_"
            "RESOURCE_USE_REMAINS_UNBOUNDED"
        ),
        "resource_law": {
            "resident_message_residual_integer_cells": (
                MESSAGE_SLOTS * cyclo.MESSAGE_INTEGER_CELLS
            ),
            "resident_message_pi_ledger_cells": MESSAGE_SLOTS,
            "resident_message_unit_ledger_cells": (
                MESSAGE_SLOTS * UNIT_RANK
            ),
            "resident_coefficient_residual_integer_cells": (
                COEFFICIENT_REGISTERS * DIMENSION
            ),
            "resident_coefficient_pi_ledger_cells": (
                COEFFICIENT_REGISTERS
            ),
            "resident_coefficient_unit_ledger_cells": (
                COEFFICIENT_REGISTERS * UNIT_RANK
            ),
            "fixed_precision_embedding_dot_and_log_work_counted": True,
            "float64_least_squares_work_counted": True,
            "exact_trace_energy_acceptance_work_counted": True,
            "unit_scale_materialization_counted": True,
            "initial_norm_element_ring_multiplications_counted": True,
            "candidate_norm_ring_multiplications_counted": True,
            "unit_scale_ring_multiplications_counted": True,
            "basis_operator_ring_multiply_accumulations_counted": True,
            "balanced_scalar_vector_ring_multiplications_counted": True,
            "compiled_embedding_cosine_cells": (
                EMBEDDING_COSINE_CELLS
            ),
            "compiled_embedding_cosine_nominal_mantissa_bits": (
                EMBEDDING_COSINE_CELLS * EMBEDDING_PRECISION_BITS
            ),
            "compiled_log_matrix_float64_cells": (
                EMBEDDING_LOG_MATRIX_FLOAT64_CELLS
            ),
            "embedding_solve_scratch_float64_cells": (
                EMBEDDING_SOLVE_FLOAT64_CELLS
            ),
            "fixed_precision_scalar_working_cells": 2,
            "proposal_transcript_commitment_bits": 256,
            "unit_generators_and_inverses_ring_elements": (
                2 * UNIT_RANK
            ),
            "named_exact_temporary_component_maxima_reported": True,
            "named_component_maxima_sum_is_simultaneous_peak": False,
            "named_component_maxima_sum_includes_python_overhead": False,
            "duplicate_public_topology_state_payload_counted": True,
            "declared_live_state_payload_uses_no_alias_discount": True,
            "python_object_overhead_bounded": False,
            "allocator_peak_bounded": False,
            "internal_ring_multiplication_peak_bounded": False,
            "whole_process_peak_bounded": False,
        },
        "not_established": [
            "CERTIFIED_LOG_EMBEDDING_VALUES",
            "EXACT_LOG_EMBEDDING_LATTICE_CENTER",
            "EXACT_CLOSEST_VECTOR_SOLUTION",
            "GLOBAL_CYCLOTOMIC_UNIT_OPTIMALITY",
            "LOCAL_CYCLOTOMIC_UNIT_OPTIMALITY",
            "MULTIPLICATIVE_INDEPENDENCE_OF_DECLARED_UNIT_GENERATORS",
            "FIXED_RESIDUAL_INTEGER_WIDTH",
            "FIXED_TOTAL_BIT_FOOTPRINT",
            "ASYMPTOTIC_RESIDUAL_HEIGHT_BOUND",
            "FULL_CARRIER_OBJECT_STATE_EQUALITY_AFTER_RESTORATION",
            "BOUNDED_REPEATED_USE_GENERATION_AND_LEASE_METADATA",
            "FAILURE_ATOMIC_ROLLBACK_AFTER_REJECTED_INVERSE",
            "WHOLE_PROCESS_OR_ALLOCATOR_PEAK_MEMORY_BOUND",
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
            "COMPACT_CERTIFIED_UNIT_REDUCTION_WITHOUT_A_65536_BIT_"
            "EMBEDDING_TABLE_OR_EQUIVALENT_HIDDEN_PRECISION_COST"
        ),
        "next_obstruction": (
            "THE_GLOBAL_PROPOSAL_REDUCES_TESTED_CARRIER_PAYLOAD_"
            "BUT_REQUIRES_A_LARGE_FIXED_PRECISION_EMBEDDING_TABLE_"
            "AND_FLOATING_SOLVE_THE_IDENTICAL_COMPACT_CLASSICAL_"
            "METHOD_REMAINS_AVAILABLE_AND_NO_TOTAL_RESOURCE_"
            "ADVANTAGE_IS_ESTABLISHED"
        ),
        "terminal": False,
    }
    if (
        not all(embedding_controls.values())
        or not result["all_raw_recurrence_boundaries_equal"]
        or not result["all_pi_content_boundaries_equal"]
        or not result["all_cases_restore_exactly"]
        or not result["all_cases_reduce_pi_content_payload"]
        or not result["all_declared_live_cases_reduce_pi_content_payload"]
        or not result["all_cases_beat_raw_recurrence_payload"]
        or not result[
            "all_declared_live_cases_beat_raw_recurrence_payload"
        ]
        or not result[
            "all_fixed_precision_embedding_evaluations_positive"
        ]
        or not result[
            "all_named_component_maxima_sums_remain_above_raw"
        ]
        or not all(control_results.values())
        or not restored["primary_restored_exactly"]
        or not restored["reuse_restored_exactly"]
        or not restored["same_original_backing"]
        or not restored["fresh_restored_reuse_boundary_equal"]
    ):
        fail("pi-unit embedding balance qualification failed")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
