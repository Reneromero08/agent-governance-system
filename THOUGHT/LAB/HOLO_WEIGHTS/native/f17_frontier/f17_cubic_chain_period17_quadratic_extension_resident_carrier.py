#!/usr/bin/env python3
"""Exact two-by-eight real-subfield resident carrier for the F17 chain.

The M115 predecessor removed full degree-sixteen products from Hermitian norm
generation, but its resident Horner output and boundary unit action still used
the canonical sixteen-coordinate cyclotomic representation.

This successor uses the exact quadratic presentation

    Q(zeta_17) = Q(s_1)[zeta] / (zeta^2-s_1*zeta+1)

and stores each resident element as ``A + zeta*B``, where ``A`` and ``B`` use
the integral ``(1,s_1,...,s_7)`` real-subfield basis.  Projection sums and the
unit-ledger action execute on these pairs.  Only the final scalar is lifted to
the full basis at the public boundary.

This is an exact representation isomorphism, not an eight-coordinate quotient:
the logical integer-coordinate count remains sixteen per element.  Forward
Horner construction and inverse rematerialization also remain full cyclotomic
and are counted.  Compact classical software can execute the identical pair
recurrence, so this bounded carrier repair does not establish an advantage or
a distinct phase resource.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_direct_real_hermitian as prior


streamed = prior.prior
real = streamed.prior
horner = real.horner
base = real.base
cyclo = real.cyclo
pi_content = real.pi_content
recurrence = real.recurrence

UNIT_RANK = real.UNIT_RANK
UNIT_GENERATORS = real.UNIT_GENERATORS
UNIT_GENERATOR_INVERSES = real.UNIT_GENERATOR_INVERSES
TESTED_PERIODS = horner.TESTED_PERIODS

RingElement = prior.RingElement
RingVector = prior.RingVector
RealElement = real.RealElement
SplitElement = tuple[RealElement, RealElement]
SplitVector = list[SplitElement]

REAL_ZERO = real.real_zero()
REAL_ONE = real.real_one()
S_ONE = real.real_s_vector(1)


def fail(message: str) -> None:
    raise RuntimeError(message)


def real_negate(value: RealElement) -> RealElement:
    return tuple(-coefficient for coefficient in value)  # type: ignore[return-value]


def real_subtract(left: RealElement, right: RealElement) -> RealElement:
    return tuple(
        left_value - right_value
        for left_value, right_value in zip(left, right, strict=True)
    )  # type: ignore[return-value]


def real_scale(value: RealElement, scalar: int) -> RealElement:
    return tuple(
        scalar * coefficient for coefficient in value
    )  # type: ignore[return-value]


def real_s1_multiply(value: RealElement) -> RealElement:
    """Multiply by s_1 as the declared integral-basis linear map."""

    return (
        2 * value[1] - value[7],
        value[0] + value[2] - value[7],
        value[1] + value[3] - value[7],
        value[2] + value[4] - value[7],
        value[3] + value[5] - value[7],
        value[4] + value[6] - value[7],
        value[5],
        value[6] - value[7],
    )


def split_zero() -> SplitElement:
    return REAL_ZERO, REAL_ZERO


def split_one() -> SplitElement:
    return REAL_ONE, REAL_ZERO


def split_add(left: SplitElement, right: SplitElement) -> SplitElement:
    return real.real_add(left[0], right[0]), real.real_add(left[1], right[1])


def split_subtract(left: SplitElement, right: SplitElement) -> SplitElement:
    return real_subtract(left[0], right[0]), real_subtract(left[1], right[1])


def split_payload_bits(value: SplitElement) -> int:
    return real.real_payload_bits(value[0]) + real.real_payload_bits(value[1])


def split_vector_payload_bits(value: SplitVector) -> int:
    return sum(split_payload_bits(element) for element in value)


def build_quadratic_extension_table() -> tuple[SplitElement, ...]:
    """Return public pairs for zeta^0 through zeta^15."""

    result: list[SplitElement] = []
    a_value, b_value = split_one()
    for _ in range(16):
        result.append((a_value, b_value))
        a_value, b_value = (
            real_negate(b_value),
            real.real_add(a_value, real_s1_multiply(b_value)),
        )
    return tuple(result)


QUADRATIC_EXTENSION_TABLE = build_quadratic_extension_table()


def full_to_split(
    value: RingElement,
    stats: "SplitCarrierStats | None" = None,
) -> SplitElement:
    if len(value) != 16:
        fail("canonical cyclotomic width changed")
    a_value = REAL_ZERO
    b_value = REAL_ZERO
    for coefficient, (a_basis, b_basis) in zip(
        value,
        QUADRATIC_EXTENSION_TABLE,
        strict=True,
    ):
        scaled_a = real_scale(a_basis, coefficient)
        scaled_b = real_scale(b_basis, coefficient)
        next_a = real.real_add(a_value, scaled_a)
        next_b = real.real_add(b_value, scaled_b)
        if stats is not None:
            stats.full_to_split_coefficient_multiplications += 16
            stats.full_to_split_coefficient_additions += 16
            stats.maximum_full_to_split_element_work_payload_bits = max(
                stats.maximum_full_to_split_element_work_payload_bits,
                real.real_payload_bits(a_value)
                + real.real_payload_bits(b_value)
                + real.real_payload_bits(scaled_a)
                + real.real_payload_bits(scaled_b)
                + real.real_payload_bits(next_a)
                + real.real_payload_bits(next_b),
            )
        a_value, b_value = next_a, next_b
    result = a_value, b_value
    if stats is not None:
        stats.full_to_split_element_conversions += 1
        stats.maximum_full_to_split_input_payload_bits = max(
            stats.maximum_full_to_split_input_payload_bits,
            base.element_payload_bits(value),
        )
        stats.maximum_full_to_split_output_payload_bits = max(
            stats.maximum_full_to_split_output_payload_bits,
            split_payload_bits(result),
        )
    return result


def zeta_multiply_full(value: RingElement) -> RingElement:
    """Multiply a canonical full element by zeta without ring multiplication."""

    high = value[-1]
    return tuple(
        -high if index == 0 else value[index - 1] - high
        for index in range(16)
    )


def split_to_full(value: SplitElement) -> RingElement:
    full_a = real.real_to_full(value[0])
    full_b = zeta_multiply_full(real.real_to_full(value[1]))
    return cyclo.ring_add(full_a, full_b)


def split_multiply(
    left: SplitElement,
    right: SplitElement,
    stats: "SplitCarrierStats | None" = None,
) -> SplitElement:
    """Karatsuba multiplication with zeta^2=s_1*zeta-1."""

    ac_value = real.real_multiply(left[0], right[0], stats)
    bd_value = real.real_multiply(left[1], right[1], stats)
    cross_value = real.real_multiply(
        real.real_add(left[0], left[1]),
        real.real_add(right[0], right[1]),
        stats,
    )
    constant = real_subtract(ac_value, bd_value)
    cross_without_ac = real_subtract(cross_value, ac_value)
    zeta_coefficient = real.real_add(
        real_subtract(cross_without_ac, bd_value),
        real_s1_multiply(bd_value),
    )
    if stats is not None:
        stats.split_pair_multiplications += 1
        stats.split_real_subfield_multiplications += 3
        stats.maximum_split_multiply_live_payload_bits = max(
            stats.maximum_split_multiply_live_payload_bits,
            split_payload_bits(left)
            + split_payload_bits(right)
            + real.real_payload_bits(ac_value)
            + real.real_payload_bits(bd_value)
            + real.real_payload_bits(cross_value)
            + real.real_payload_bits(constant)
            + real.real_payload_bits(zeta_coefficient),
        )
    return constant, zeta_coefficient


def split_conjugate(value: SplitElement) -> SplitElement:
    return (
        real.real_add(value[0], real_s1_multiply(value[1])),
        real_negate(value[1]),
    )


def split_power(
    value: SplitElement,
    exponent: int,
    stats: "SplitCarrierStats",
) -> SplitElement:
    if exponent < 0:
        fail("split power received a negative exponent")
    result = split_one()
    factor = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = split_multiply(result, factor, stats)
        remaining >>= 1
        if remaining:
            factor = split_multiply(factor, factor, stats)
    return result


SPLIT_UNIT_GENERATORS = tuple(full_to_split(value) for value in UNIT_GENERATORS)
SPLIT_UNIT_GENERATOR_INVERSES = tuple(
    full_to_split(value) for value in UNIT_GENERATOR_INVERSES
)
PUBLIC_SPLIT_CONVERSION_TABLE_PAYLOAD_BITS = sum(
    split_payload_bits(value) for value in QUADRATIC_EXTENSION_TABLE
)
PUBLIC_SPLIT_UNIT_TABLE_PAYLOAD_BITS = sum(
    split_payload_bits(value)
    for value in (*SPLIT_UNIT_GENERATORS, *SPLIT_UNIT_GENERATOR_INVERSES)
)


@dataclass
class SplitCarrierStats(prior.DirectHermitianStats):
    full_to_split_element_conversions: int = 0
    full_to_split_coefficient_multiplications: int = 0
    full_to_split_coefficient_additions: int = 0
    split_to_full_boundary_lifts: int = 0
    split_pair_multiplications: int = 0
    split_real_subfield_multiplications: int = 0
    split_projection_additions: int = 0
    maximum_full_to_split_input_payload_bits: int = 0
    maximum_full_to_split_output_payload_bits: int = 0
    maximum_full_to_split_element_work_payload_bits: int = 0
    maximum_full_vector_to_split_vector_live_payload_bits: int = 0
    maximum_split_carrier_resident_payload_bits: int = 0
    maximum_predecessor_full_resident_payload_bits: int = 0
    maximum_split_projection_accumulator_payload_bits: int = 0
    maximum_split_projection_scale_payload_bits: int = 0
    maximum_split_projection_product_payload_bits: int = 0
    maximum_split_multiply_live_payload_bits: int = 0
    maximum_boundary_full_lift_payload_bits: int = 0
    maximum_split_projection_normalized_payload_bits: int = 0
    maximum_split_projection_live_payload_bits: int = 0


SPLIT_FIELDS = (
    "full_to_split_element_conversions",
    "full_to_split_coefficient_multiplications",
    "full_to_split_coefficient_additions",
    "split_to_full_boundary_lifts",
    "split_pair_multiplications",
    "split_real_subfield_multiplications",
    "split_projection_additions",
    "maximum_full_to_split_input_payload_bits",
    "maximum_full_to_split_output_payload_bits",
    "maximum_full_to_split_element_work_payload_bits",
    "maximum_full_vector_to_split_vector_live_payload_bits",
    "maximum_split_carrier_resident_payload_bits",
    "maximum_predecessor_full_resident_payload_bits",
    "maximum_split_projection_accumulator_payload_bits",
    "maximum_split_projection_scale_payload_bits",
    "maximum_split_projection_product_payload_bits",
    "maximum_split_multiply_live_payload_bits",
    "maximum_boundary_full_lift_payload_bits",
    "maximum_split_projection_normalized_payload_bits",
    "maximum_split_projection_live_payload_bits",
)


def stats_json(
    stats: SplitCarrierStats,
    pi_stats: pi_content.PiStats,
) -> dict[str, Any]:
    result = prior.stats_json(stats, pi_stats)
    for name in SPLIT_FIELDS:
        result[name] = getattr(stats, name)
    return result


def split_vector(
    value: RingVector,
    stats: SplitCarrierStats,
    *,
    pi_exponent: int = 0,
    unit_ledger: tuple[int, ...] | list[int] = (),
    additional_live_payload_bits: int = 0,
) -> SplitVector:
    full_bits = base.vector_payload_bits(value)
    metadata_bits = base.signed_bits(pi_exponent) + base.ledger_payload_bits(
        unit_ledger
    )
    full_resident_bits = full_bits + metadata_bits
    result: SplitVector = []
    for element in value:
        partial_before = split_vector_payload_bits(result)
        converted = full_to_split(element, stats)
        partial_after = partial_before + split_payload_bits(converted)
        stats.maximum_full_vector_to_split_vector_live_payload_bits = max(
            stats.maximum_full_vector_to_split_vector_live_payload_bits,
            additional_live_payload_bits
            + full_resident_bits
            + partial_before
            + stats.maximum_full_to_split_element_work_payload_bits,
            additional_live_payload_bits + full_resident_bits + partial_after,
        )
        result.append(converted)
    split_bits = split_vector_payload_bits(result)
    stats.maximum_predecessor_full_resident_payload_bits = max(
        stats.maximum_predecessor_full_resident_payload_bits,
        full_resident_bits,
    )
    stats.maximum_full_vector_to_split_vector_live_payload_bits = max(
        stats.maximum_full_vector_to_split_vector_live_payload_bits,
        additional_live_payload_bits + full_resident_bits + split_bits,
    )
    return result


def split_ledger_scale(
    ledger: list[int],
    stats: SplitCarrierStats,
) -> SplitElement:
    if len(ledger) != UNIT_RANK:
        fail("split unit ledger width changed")
    result = split_one()
    for exponent, generator, inverse in zip(
        ledger,
        SPLIT_UNIT_GENERATORS,
        SPLIT_UNIT_GENERATOR_INVERSES,
        strict=True,
    ):
        factor = (
            split_power(generator, exponent, stats)
            if exponent >= 0
            else split_power(inverse, -exponent, stats)
        )
        result = split_multiply(result, factor, stats)
    return result


@dataclass
class SplitCarrier:
    output: SplitVector
    output_pi_exponent: int
    output_unit_ledger: list[int]
    generation: int = 0
    lease: int = 0
    active: bool = False
    pending_operations: int = 0
    phase: str = "RESTORED"

    @classmethod
    def create(cls) -> "SplitCarrier":
        return cls(
            [split_zero() for _ in range(cyclo.PRIME)],
            0,
            [0 for _ in range(UNIT_RANK)],
        )

    def all_zero(self) -> bool:
        return (
            all(element == split_zero() for element in self.output)
            and self.output_pi_exponent == 0
            and not any(self.output_unit_ledger)
            and not self.active
            and self.pending_operations == 0
            and self.phase == "RESTORED"
        )

    def backing_identity(self) -> tuple[int, int]:
        return id(self.output), id(self.output_unit_ledger)

    def payload_bits(self) -> int:
        return (
            split_vector_payload_bits(self.output)
            + base.signed_bits(self.output_pi_exponent)
            + base.ledger_payload_bits(self.output_unit_ledger)
        )

    def canonical_state(self) -> dict[str, Any]:
        return {
            "split_output_zero": all(
                element == split_zero() for element in self.output
            ),
            "pi_ledger_zero": self.output_pi_exponent == 0,
            "unit_ledger_zero": not any(self.output_unit_ledger),
            "all_payload_and_ledgers_zero": (
                all(element == split_zero() for element in self.output)
                and self.output_pi_exponent == 0
                and not any(self.output_unit_ledger)
            ),
            "generation": self.generation,
            "lease": self.lease,
            "active": self.active,
            "pending_operations": self.pending_operations,
            "phase": self.phase,
        }


def record_carrier(carrier: SplitCarrier, stats: SplitCarrierStats) -> int:
    payload = carrier.payload_bits()
    stats.maximum_split_carrier_resident_payload_bits = max(
        stats.maximum_split_carrier_resident_payload_bits,
        payload,
    )
    stats.maximum_carrier_resident_payload_bits = max(
        stats.maximum_carrier_resident_payload_bits,
        payload,
    )
    stats.maximum_resident_payload_bits = max(
        stats.maximum_resident_payload_bits,
        payload,
    )
    return payload


def split_project_boundary(
    carrier: SplitCarrier,
    pi_stats: pi_content.PiStats,
    stats: SplitCarrierStats,
) -> RingElement:
    projected = split_zero()
    for element in carrier.output:
        projected = split_add(projected, element)
        stats.split_projection_additions += 1
        stats.maximum_split_projection_accumulator_payload_bits = max(
            stats.maximum_split_projection_accumulator_payload_bits,
            split_payload_bits(projected),
        )
    unit_scale = split_ledger_scale(carrier.output_unit_ledger, stats)
    unit_projected = split_multiply(unit_scale, projected, stats)
    lifted = split_to_full(unit_projected)
    stats.split_to_full_boundary_lifts += 1

    scaled = pi_content.normalize_element(
        lifted,
        carrier.output_pi_exponent,
        pi_stats,
    )
    boundary = pi_content.materialize_element(scaled, pi_stats)
    scale_bits = split_payload_bits(unit_scale)
    projected_bits = split_payload_bits(projected)
    product_bits = split_payload_bits(unit_projected)
    lifted_bits = base.element_payload_bits(lifted)
    scaled_bits = (
        base.element_payload_bits(scaled.residual)
        + base.signed_bits(scaled.exponent)
    )
    boundary_bits = base.element_payload_bits(boundary)
    stats.maximum_split_projection_scale_payload_bits = max(
        stats.maximum_split_projection_scale_payload_bits,
        scale_bits,
    )
    stats.maximum_split_projection_product_payload_bits = max(
        stats.maximum_split_projection_product_payload_bits,
        product_bits,
    )
    stats.maximum_boundary_full_lift_payload_bits = max(
        stats.maximum_boundary_full_lift_payload_bits,
        lifted_bits,
        boundary_bits,
    )
    stats.maximum_split_projection_normalized_payload_bits = max(
        stats.maximum_split_projection_normalized_payload_bits,
        scaled_bits,
    )
    stats.maximum_split_projection_live_payload_bits = max(
        stats.maximum_split_projection_live_payload_bits,
        projected_bits
        + scale_bits
        + product_bits
        + lifted_bits
        + scaled_bits
        + boundary_bits,
        projected_bits + stats.maximum_split_multiply_live_payload_bits,
    )
    return boundary


def install_m115_stack() -> None:
    streamed.streamed_real_vector_norm = prior.direct_real_vector_norm
    real.real_deferred_balance_vector = streamed.streamed_real_balance_vector
    horner.HornerStats = SplitCarrierStats
    base.BalanceStats = SplitCarrierStats
    base.balance_vector = streamed.streamed_real_balance_vector
    base.ledger_scale = real.exact.tracked_ledger_scale
    base.add_balanced_vectors = real.deferred.relative_add_balanced_vectors


def populate_forward(
    carrier: SplitCarrier,
    block: cyclo.CompiledBlock,
    periods: int,
) -> tuple[RingElement, pi_content.PiStats, SplitCarrierStats]:
    if not carrier.all_zero():
        fail("split carrier was not restored")
    carrier.active = True
    carrier.lease += 1
    carrier.pending_operations = 1
    carrier.phase = "BUILD_FULL_THEN_SPLIT_RESIDENT"
    pi_stats = pi_content.PiStats()
    stats = SplitCarrierStats()
    output = horner.build_horner_output(block, periods, pi_stats, stats)
    resident = split_vector(
        output.residual,
        stats,
        pi_exponent=output.pi_exponent,
        unit_ledger=output.unit_ledger,
        additional_live_payload_bits=carrier.payload_bits(),
    )
    carrier.output[:] = resident
    carrier.output_pi_exponent = output.pi_exponent
    carrier.output_unit_ledger[:] = output.unit_ledger
    carrier.phase = "SPLIT_OUTPUT_RESIDENT"
    resident_bits = record_carrier(carrier, stats)
    del output, resident

    boundary = split_project_boundary(carrier, pi_stats, stats)
    stats.maximum_projection_resident_plus_work_payload_bits = max(
        stats.maximum_projection_resident_plus_work_payload_bits,
        resident_bits + stats.maximum_split_projection_live_payload_bits,
    )
    return boundary, pi_stats, stats


def restore_forward(
    carrier: SplitCarrier,
    block: cyclo.CompiledBlock,
    periods: int,
) -> tuple[pi_content.PiStats, SplitCarrierStats]:
    if carrier.phase != "SPLIT_OUTPUT_RESIDENT":
        fail("split inverse was reordered")
    inverse_pi_stats = pi_content.PiStats()
    inverse_stats = SplitCarrierStats()
    expected = horner.build_horner_output(
        block,
        periods,
        inverse_pi_stats,
        inverse_stats,
    )
    expected_split = split_vector(
        expected.residual,
        inverse_stats,
        pi_exponent=expected.pi_exponent,
        unit_ledger=expected.unit_ledger,
    )
    inverse_stats.maximum_inverse_resident_plus_work_payload_bits = max(
        inverse_stats.maximum_inverse_resident_plus_work_payload_bits,
        carrier.payload_bits()
        + max(
            inverse_stats.maximum_horner_named_checkpoint_payload_bits,
            inverse_stats.maximum_full_vector_to_split_vector_live_payload_bits,
        ),
    )
    if (
        carrier.output != expected_split
        or carrier.output_pi_exponent != expected.pi_exponent
        or tuple(carrier.output_unit_ledger) != expected.unit_ledger
    ):
        fail("split inverse rematerialization mismatch")
    carrier.output[:] = [
        split_subtract(actual, value)
        for actual, value in zip(carrier.output, expected_split, strict=True)
    ]
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
        fail("split carrier did not restore exactly")
    return inverse_pi_stats, inverse_stats


@dataclass
class Transaction:
    boundary: RingElement
    pi_stats: pi_content.PiStats
    stats: SplitCarrierStats
    inverse_pi_stats: pi_content.PiStats
    inverse_stats: SplitCarrierStats
    restored_exactly: bool
    same_backing: bool


def execute_transaction(
    carrier: SplitCarrier,
    block: cyclo.CompiledBlock,
    periods: int,
) -> Transaction:
    if not isinstance(carrier, SplitCarrier):
        fail("null or invalid split carrier")
    backing = carrier.backing_identity()
    boundary, pi_stats, stats = populate_forward(carrier, block, periods)
    inverse_pi_stats, inverse_stats = restore_forward(carrier, block, periods)
    return Transaction(
        boundary,
        pi_stats,
        stats,
        inverse_pi_stats,
        inverse_stats,
        carrier.all_zero(),
        carrier.backing_identity() == backing,
    )


def named_search_temporary_maxima_sum(metrics: dict[str, Any]) -> int:
    return (
        horner.named_search_temporary_maxima_sum(metrics)
        + metrics["maximum_streamed_norm_named_live_payload_bits"]
        + metrics["maximum_real_current_norm_payload_bits"]
        + metrics["maximum_real_current_energy_bits"]
    )


def case_result(periods: int, block: cyclo.CompiledBlock) -> dict[str, Any]:
    carrier = SplitCarrier.create()
    transaction = execute_transaction(carrier, block, periods)
    phase_metrics = stats_json(transaction.stats, transaction.pi_stats)
    inverse_metrics = stats_json(
        transaction.inverse_stats,
        transaction.inverse_pi_stats,
    )
    raw_boundary, raw_stats = horner.raw_horner_boundary(block, periods)
    boundary_sha256 = hashlib.sha256(
        cyclo.encoded_ring_object(transaction.boundary)
    ).hexdigest()
    search_temporary = named_search_temporary_maxima_sum(phase_metrics)
    phase_checkpoint = max(
        phase_metrics["maximum_horner_named_checkpoint_payload_bits"],
        phase_metrics["maximum_full_vector_to_split_vector_live_payload_bits"],
        phase_metrics["maximum_projection_resident_plus_work_payload_bits"],
        inverse_metrics["maximum_inverse_resident_plus_work_payload_bits"],
    )
    retained_tables = (
        real.compiled_real_search_table_payload_bits()
        + PUBLIC_SPLIT_CONVERSION_TABLE_PAYLOAD_BITS
        + PUBLIC_SPLIT_UNIT_TABLE_PAYLOAD_BITS
    )
    named_total = phase_checkpoint + retained_tables + search_temporary
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
        "raw_horner_boundary_equal": transaction.boundary == raw_boundary,
        "phase_stats": phase_metrics,
        "inverse_rematerialization_stats": inverse_metrics,
        "raw_horner_stats": horner.raw_horner_stats_json(raw_stats),
        "phase_named_checkpoint_payload_bits": phase_checkpoint,
        "named_search_temporary_maxima_sum_bits": search_temporary,
        "retained_public_table_payload_bits": retained_tables,
        "phase_named_component_maxima_sum_bits": named_total,
        "raw_horner_named_checkpoint_payload_bits": raw_payload,
        "phase_minus_raw_horner_named_payload_bits": named_total - raw_payload,
        "phase_named_payload_beats_raw_horner": named_total < raw_payload,
        "split_resident_minus_comparable_full_resident_payload_bits": (
            phase_metrics["maximum_split_carrier_resident_payload_bits"]
            - phase_metrics["maximum_predecessor_full_resident_payload_bits"]
        ),
        "restored_exactly": transaction.restored_exactly,
        "same_backing": transaction.same_backing,
        "canonical_restored_state": carrier.canonical_state(),
    }


def restoration_reuse_case(
    primary: cyclo.CompiledBlock,
    reuse: cyclo.CompiledBlock,
) -> dict[str, Any]:
    carrier = SplitCarrier.create()
    backing = carrier.backing_identity()
    primary_transaction = execute_transaction(carrier, primary, 1)
    reuse_transaction = execute_transaction(carrier, reuse, 1)
    fresh = execute_transaction(SplitCarrier.create(), reuse, 1)
    return {
        "primary_restored_exactly": primary_transaction.restored_exactly,
        "reuse_restored_exactly": reuse_transaction.restored_exactly,
        "same_original_backing": carrier.backing_identity() == backing,
        "fresh_restored_reuse_boundary_equal": (
            reuse_transaction.boundary == fresh.boundary
        ),
        "fresh_restored_reuse_phase_signature_equal": (
            stats_json(
                reuse_transaction.stats,
                reuse_transaction.pi_stats,
            )
            == stats_json(fresh.stats, fresh.pi_stats)
        ),
        "generation": carrier.generation,
        "lease": carrier.lease,
        "baseline_reload": False,
        "full_carrier_object_state_equal": False,
        "canonical_restored_state": carrier.canonical_state(),
    }


def controls(
    primary: cyclo.CompiledBlock,
    reuse: cyclo.CompiledBlock,
) -> dict[str, bool]:
    reordered = SplitCarrier.create()
    reordered_rejected = False
    try:
        restore_forward(reordered, primary, 1)
    except RuntimeError:
        reordered_rejected = reordered.all_zero()

    missing = SplitCarrier.create()
    populate_forward(missing, primary, 1)
    missing_detected = not missing.all_zero() and missing.pending_operations == 1

    wrong = SplitCarrier.create()
    populate_forward(wrong, primary, 1)
    wrong_before = list(wrong.output)
    wrong_rejected = False
    try:
        restore_forward(wrong, reuse, 1)
    except RuntimeError:
        wrong_rejected = wrong.output == wrong_before

    mutation = SplitCarrier.create()
    populate_forward(mutation, primary, 1)
    mutation.output[0] = split_add(mutation.output[0], split_one())
    mutation_before = list(mutation.output)
    mutation_rejected = False
    try:
        restore_forward(mutation, primary, 1)
    except RuntimeError:
        mutation_rejected = mutation.output == mutation_before

    null_rejected = False
    try:
        execute_transaction(None, primary, 1)  # type: ignore[arg-type]
    except RuntimeError:
        null_rejected = True
    return {
        "reordered_inverse_rejected_before_mutation": reordered_rejected,
        "missing_inverse_leaves_detectable_resident_state": missing_detected,
        "wrong_inverse_rejected_before_mutation": wrong_rejected,
        "resident_mutation_rejected_before_inverse_mutation": mutation_rejected,
        "null_carrier_rejected": null_rejected,
        "snapshot_reload_absent": True,
        "boundary_only_full_lift_declared": True,
    }


def algebra_controls() -> dict[str, bool]:
    basis = []
    for index in range(16):
        value = [0 for _ in range(16)]
        value[index] = 1
        basis.append(tuple(value))
    roundtrip = all(split_to_full(full_to_split(value)) == value for value in basis)
    multiplication = all(
        split_to_full(
            split_multiply(full_to_split(left), full_to_split(right))
        )
        == cyclo.ring_multiply(left, right)
        for left in basis
        for right in basis
    )
    conjugation = all(
        split_to_full(split_conjugate(full_to_split(value)))
        == base.ring_conjugate(value)
        for value in basis
    )
    return {
        "all_16_basis_roundtrips_exact": roundtrip,
        "all_256_basis_products_exact": multiplication,
        "all_16_basis_conjugates_exact": conjugation,
        "quadratic_relation_exact": (
            split_multiply((REAL_ZERO, REAL_ONE), (REAL_ZERO, REAL_ONE))
            == (real_negate(REAL_ONE), S_ONE)
        ),
        "coordinate_count_remains_16": (
            len(split_one()[0]) + len(split_one()[1]) == 16
        ),
    }


def main() -> int:
    if len(sys.argv) != 1:
        fail(
            "usage: f17_cubic_chain_period17_"
            "quadratic_extension_resident_carrier.py"
        )
    install_m115_stack()
    blocks = {
        family.lower(): cyclo.build_compiled_block(family)
        for family in ("PRIMARY", "REUSE")
    }
    cases = [
        case_result(periods, blocks[family])
        for periods in TESTED_PERIODS
        for family in ("primary", "reuse")
    ]
    restored = restoration_reuse_case(blocks["primary"], blocks["reuse"])
    carrier_controls = controls(blocks["primary"], blocks["reuse"])
    algebra = algebra_controls()
    all_restored = all(
        case["restored_exactly"]
        and case["same_backing"]
        and case["canonical_restored_state"]["all_payload_and_ledgers_zero"]
        for case in cases
    )
    all_boundaries = all(case["raw_horner_boundary_equal"] for case in cases)
    all_boundary_lifts_one = all(
        case["phase_stats"]["split_to_full_boundary_lifts"] == 1
        and case["inverse_rematerialization_stats"][
            "split_to_full_boundary_lifts"
        ]
        == 0
        for case in cases
    )
    result = {
        "result": "PASS",
        "experiment": (
            "EXACT_QUADRATIC_EXTENSION_REAL_SUBFIELD_RESIDENT_"
            "CARRIER_WITH_BOUNDARY_ONLY_FULL_LIFT"
        ),
        "claim_candidate": (
            "BOUNDED_EXACT_POST_FORWARD_TWO_BY_EIGHT_REAL_SUBFIELD_"
            "QUADRATIC_EXTENSION_STORED_CARRIER_REPLACES_FULL_"
            "CYCLOTOMIC_RESIDENCY_AND_PROJECTS_THROUGH_ONE_FINAL_"
            "SCALAR_FULL_LIFT_WITH_EXACT_RESTORATION_AND_PERIOD1_REUSE"
        ),
        "classification_candidate": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level_candidate": "PACKAGE_SELF_REVIEW",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "tested_periods": list(TESTED_PERIODS),
        "full_cyclotomic_dimension": 16,
        "real_subfield_dimension": 8,
        "quadratic_extension_lanes": 2,
        "resident_integer_coordinate_count_per_element": 16,
        "dimension_reducing_quotient": False,
        "representation_isomorphism": True,
        "forward_horner_construction_remains_full_cyclotomic": True,
        "inverse_rematerialization_remains_full_cyclotomic": True,
        "resident_full_cyclotomic_vector_retained_after_conversion": False,
        "accepted_projection_full_vector_materialized": False,
        "boundary_full_scalar_lifts_per_transaction": 1,
        "public_topology_compilation_answer_independent": True,
        "retained_inverse_history_bytes": 0,
        "public_table_accounting": {
            "real_search_table_payload_bits": (
                real.compiled_real_search_table_payload_bits()
            ),
            "quadratic_conversion_table_payload_bits": (
                PUBLIC_SPLIT_CONVERSION_TABLE_PAYLOAD_BITS
            ),
            "split_unit_table_payload_bits": PUBLIC_SPLIT_UNIT_TABLE_PAYLOAD_BITS,
            "predecessor_full_unit_generators_still_loaded_and_counted_in_real_search_table": True,
        },
        "block_certificates": {
            family: {
                "public_program_sha256": hashlib.sha256(
                    cyclo.adaptive.encoded_program(block.public_program)
                ).hexdigest(),
                "operator_sha256": block.operator_sha256,
                "characteristic_sha256": block.characteristic_sha256,
                "characteristic_identity_exact": block.characteristic_identity_exact,
                "characteristic": block.characteristic,
            }
            for family, block in blocks.items()
        },
        "cases": cases,
        "all_boundaries_equal_raw_horner": all_boundaries,
        "all_cases_restore_exactly": all_restored,
        "all_cases_use_one_boundary_full_lift": all_boundary_lifts_one,
        "restoration_reuse_case": restored,
        "carrier_controls": carrier_controls,
        "algebra_controls": algebra,
        "verification_accounting": {
            "carrier_control_forward_populations": 3,
            "carrier_control_failed_inverse_attempts": 3,
            "restoration_reuse_transactions": 3,
            "basis_roundtrip_controls": 16,
            "basis_product_controls": 256,
            "basis_conjugation_controls": 16,
            "verification_work_in_accepted_execution_total": False,
        },
        "matched_classical": {
            "matched_raw_horner_named_checkpoint_implemented": True,
            "identical_two_by_eight_quadratic_extension_carrier_available": True,
            "same_public_coefficients_operator_boundary_and_pair_algebra": True,
            "comparison_establishes_advantage": False,
        },
        "resource_law": {
            "full_forward_horner_checkpoints_counted": True,
            "full_inverse_rematerialization_checkpoints_counted": True,
            "full_to_split_conversion_input_and_output_counted": True,
            "full_to_split_conversion_scalar_multiplications_additions_and_live_work_counted": True,
            "empty_carrier_backing_counted_during_forward_conversion": True,
            "split_resident_payload_counted": True,
            "split_projection_accumulator_scale_product_normalized_state_and_pair_multiplication_live_work_counted": True,
            "one_boundary_full_lift_and_materialized_boundary_counted": True,
            "real_search_and_split_public_tables_counted": True,
            "matched_raw_horner_counted": True,
            "named_component_maxima_sum_is_simultaneous_peak": False,
            "python_object_allocator_native_library_and_bigint_internal_bytes_bounded": False,
            "whole_process_peak_bounded": False,
        },
        "observation": (
            "THE_RESIDENT_FULL_CYCLOTOMIC_VECTOR_CAN_BE_REPLACED_"
            "EXACTLY_BY_TWO_REAL_SUBFIELD_LANES_WITH_ONE_BOUNDARY_"
            "LIFT_BUT_COORDINATE_COUNT_REMAINS16_FORWARD_AND_INVERSE_"
            "CONSTRUCTION_REMAIN_FULL_AND_IDENTICAL_CLASSICAL_PAIR_"
            "EXECUTION_IS_AVAILABLE"
        ),
        "not_established": [
            "DIMENSION_OR_RANK_REDUCTION",
            "FULL_RECURRENCE_IN_THE_QUADRATIC_EXTENSION_CARRIER",
            "ELIMINATION_OF_FULL_FORWARD_OR_INVERSE_HORNER_WORK",
            "ELIMINATION_OF_FULL_CERTIFIED_ACTION_DURING_CONSTRUCTION",
            "LOWER_RESOURCE_TOTAL_THAN_MATCHED_RAW_HORNER",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "MACHINE_ENFORCED_NO_SMUGGLE_OR_CATVM_CUSTODY",
            "CATALYTIC_INFERENCE",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_COMPUTATION",
        ],
        "next_experiment": (
            "FULL_QUADRATIC_EXTENSION_HORNER_AND_CERTIFIED_ACTION_"
            "OR_PHASE_NATIVE_NONCLASSICAL_TRACE_COUPLING"
        ),
        "next_obstruction": (
            "RESIDENCY_AND_PROJECTION_NOW_USE_THE_EXACT_TWO_BY_EIGHT_"
            "PAIR_BUT_THE_MAP_IS_AN_ISOMORPHISM_FORWARD_AND_INVERSE_"
            "CONSTRUCTION_REMAIN_FULL_AND_COMPACT_CLASSICAL_SOFTWARE_"
            "USES_THE_IDENTICAL_PAIR_ALGEBRA"
        ),
        "terminal": False,
    }
    hard_gate = {
        "boundaries": all_boundaries,
        "restoration": all_restored,
        "one_boundary_lift": all_boundary_lifts_one,
        "reuse": (
            restored["primary_restored_exactly"]
            and restored["reuse_restored_exactly"]
            and restored["same_original_backing"]
            and restored["fresh_restored_reuse_boundary_equal"]
            and restored["fresh_restored_reuse_phase_signature_equal"]
        ),
        "controls": all(carrier_controls.values()),
        "algebra": all(algebra.values()),
        "no_dimension_claim": not result["dimension_reducing_quotient"],
    }
    if not all(hard_gate.values()):
        fail("quadratic extension carrier gate failed: " + json.dumps(hard_gate, sort_keys=True))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
