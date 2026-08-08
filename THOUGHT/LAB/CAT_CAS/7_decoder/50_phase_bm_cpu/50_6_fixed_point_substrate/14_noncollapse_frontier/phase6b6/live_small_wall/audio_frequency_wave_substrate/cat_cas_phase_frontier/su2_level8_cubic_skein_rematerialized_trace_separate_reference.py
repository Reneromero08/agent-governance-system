#!/usr/bin/env python3
"""Standalone M224 ledger-rematerialized direct-trace reference."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

import su2_level8_cubic_skein_streamed_embedding_energy_separate_reference as m222


sys.set_int_max_str_digits(0)
prior = m222.prior
base = m222.base
CASES = m222.CASES


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def fraction_payload(value: Fraction) -> int:
    return signed_bits(value.numerator) + value.denominator.bit_length()


def mobius(value: int) -> int:
    parity = 0
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            value //= divisor
            parity += 1
            if value % divisor == 0:
                return 0
            while value % divisor == 0:
                value //= divisor
        divisor += 1
    if value > 1:
        parity += 1
    return -1 if parity % 2 else 1


def phi(value: int) -> int:
    result = value
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            while value % divisor == 0:
                value //= divisor
            result -= result // divisor
        divisor += 1
    if value > 1:
        result -= result // value
    return result


def ramanujan_trace(power: int) -> int:
    divisor = math.gcd(40, power)
    quotient = 40 // divisor
    return mobius(quotient) * phi(40) // phi(quotient)


@dataclass
class Work(m222.Work):
    direct_trace_energy_evaluations: int = 0
    direct_trace_cells_scanned: int = 0
    direct_trace_nonzero_coordinate_pairs: int = 0
    direct_trace_nonzero_ramanujan_terms: int = 0
    direct_trace_zero_ramanujan_terms_skipped: int = 0
    direct_trace_rational_multiplications: int = 0
    direct_trace_rational_additions: int = 0
    weighted_cell_field_products_materialized: int = 0
    norm_cell_field_products_materialized: int = 0
    direct_trace_conjugate_field_values_materialized: int = 0
    lifecycle_scale_conjugate_field_values_materialized: int = 0
    coordinate_transient_observations: int = 0
    maximum_coordinate_rational_payload_bits: int = 0
    maximum_coordinate_integer_payload_bits: int = 0
    maximum_coordinate_rational_cells: int = 0
    ledger_norm_rematerializations: int = 0
    ledger_scale_rematerializations: int = 0
    direct_trace_weight_rematerializations: int = 0
    candidate_multiplier_rematerializations: int = 0
    selected_multiplier_rematerializations: int = 0
    rematerialization_field_multiplications: int = 0
    maximum_direct_trace_caller_scalar_cells: int = 0

    def set_ambient_liveness(
        self,
        *,
        scalars: tuple[base.E, ...],
        integers: tuple[int, ...],
    ) -> None:
        if getattr(self, "_ambient_scalars", ()) or getattr(
            self, "_ambient_integers", ()
        ):
            raise RuntimeError("nested ambient exact liveness is forbidden")
        self._ambient_scalars = scalars
        self._ambient_integers = integers

    def clear_ambient_liveness(self) -> None:
        self._ambient_scalars = ()
        self._ambient_integers = ()

    def observe(
        self,
        residual: list[base.E],
        ledger: list[int],
        scratch: list[base.E],
        *,
        scalars: Iterable[base.E] = (),
        integers: Iterable[int] = (),
        context: str = "",
    ) -> None:
        super().observe(
            residual,
            ledger,
            scratch,
            scalars=getattr(self, "_ambient_scalars", ()) + tuple(scalars),
            integers=getattr(self, "_ambient_integers", ()) + tuple(integers),
            context=context,
        )

    def observe_coordinate_term(
        self,
        *,
        resident_bits: int,
        scratch_bits: int,
        weight_bits: int,
        caller_scalar_bits: int,
        base_field_cells: int,
        caller_scalar_cells: int,
        accumulator: Fraction,
        left: Fraction,
        triple: Fraction,
        scaled: Fraction,
        updated: Fraction,
        integers: tuple[int, ...],
        context_prefix: str,
    ) -> None:
        self.maximum_direct_trace_caller_scalar_cells = max(
            self.maximum_direct_trace_caller_scalar_cells,
            caller_scalar_cells,
        )
        integer_bits = sum(signed_bits(value) for value in integers)
        accumulator_bits = fraction_payload(accumulator)
        left_bits = fraction_payload(left)
        triple_bits = fraction_payload(triple)
        scaled_bits = fraction_payload(scaled)
        updated_bits = fraction_payload(updated)
        stages = (
            (accumulator_bits + left_bits, 2, f"{context_prefix}_COORDINATE_PAIR"),
            (
                accumulator_bits + left_bits + triple_bits,
                3,
                f"{context_prefix}_COORDINATE_TRIPLE",
            ),
            (
                accumulator_bits + left_bits + triple_bits + scaled_bits,
                4,
                f"{context_prefix}_RAMANUJAN_SCALE",
            ),
            (
                accumulator_bits
                + left_bits
                + triple_bits
                + scaled_bits
                + updated_bits,
                5,
                f"{context_prefix}_RATIONAL_ACCUMULATE",
            ),
        )
        rational_bits, rational_cells, context = max(stages)
        payload = (
            resident_bits
            + weight_bits
            + caller_scalar_bits
            + rational_bits
            + integer_bits
        )
        self.coordinate_transient_observations += 4
        self.transient_observations += 4
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits, resident_bits
        )
        self.maximum_scratch_payload_bits = max(
            self.maximum_scratch_payload_bits, scratch_bits
        )
        self.maximum_scalar_payload_bits = max(
            self.maximum_scalar_payload_bits, weight_bits + caller_scalar_bits
        )
        self.maximum_line_integer_payload_bits = max(
            self.maximum_line_integer_payload_bits, integer_bits
        )
        self.maximum_coordinate_rational_payload_bits = max(
            self.maximum_coordinate_rational_payload_bits, rational_bits
        )
        self.maximum_coordinate_integer_payload_bits = max(
            self.maximum_coordinate_integer_payload_bits, integer_bits
        )
        self.maximum_coordinate_rational_cells = max(
            self.maximum_coordinate_rational_cells, rational_cells
        )
        if payload > self.maximum_declared_live_payload_bits:
            self.maximum_declared_live_payload_bits = payload
            self.maximum_declared_live_context = context
        self.maximum_declared_live_field_cells = max(
            self.maximum_declared_live_field_cells,
            base_field_cells + caller_scalar_cells,
        )


def direct_trace_energy(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    weight: base.E,
    work: Work,
    *,
    live_integers: tuple[int, ...] = (),
    caller_live_scalars: tuple[base.E, ...] = (),
    context_prefix: str,
) -> int:
    work.exact_trace_energy_evaluations += 1
    work.streamed_embedding_energy_evaluations += 1
    work.direct_trace_energy_evaluations += 1
    accumulator = Fraction(0)
    scratch_bits = base.payload_bits(scratch)
    resident_bits = (
        base.payload_bits(residual)
        + prior.m220.ledger_bits(ledger)
        + scratch_bits
    )
    weight_bits = prior.scalar_payload(weight)
    caller_scalar_bits = sum(
        prior.scalar_payload(value) for value in caller_live_scalars
    )
    base_field_cells = len(residual) + len(scratch) + 1
    for cell_index, value in enumerate(residual):
        work.streamed_embedding_cells_scanned += 1
        work.direct_trace_cells_scanned += 1
        for a, weight_coordinate in enumerate(weight.coordinates):
            if not weight_coordinate:
                continue
            for b, left_coordinate in enumerate(value.coordinates):
                if not left_coordinate:
                    continue
                left = weight_coordinate * left_coordinate
                work.direct_trace_nonzero_coordinate_pairs += 1
                work.direct_trace_rational_multiplications += 1
                for c, right_coordinate in enumerate(value.coordinates):
                    if not right_coordinate:
                        continue
                    trace_coefficient = ramanujan_trace(a + b - c)
                    if not trace_coefficient:
                        work.direct_trace_zero_ramanujan_terms_skipped += 1
                        continue
                    triple = left * right_coordinate
                    work.direct_trace_rational_multiplications += 1
                    scaled = triple * trace_coefficient
                    work.direct_trace_rational_multiplications += 1
                    updated = accumulator + scaled
                    work.direct_trace_rational_additions += 1
                    work.direct_trace_nonzero_ramanujan_terms += 1
                    work.observe_coordinate_term(
                        resident_bits=resident_bits,
                        scratch_bits=scratch_bits,
                        weight_bits=weight_bits,
                        caller_scalar_bits=caller_scalar_bits,
                        base_field_cells=base_field_cells,
                        caller_scalar_cells=len(caller_live_scalars),
                        accumulator=accumulator,
                        left=left,
                        triple=triple,
                        scaled=scaled,
                        updated=updated,
                        integers=live_integers
                        + (cell_index, a, b, c, trace_coefficient),
                        context_prefix=context_prefix,
                    )
                    accumulator = updated
    if accumulator.denominator != 1:
        raise RuntimeError("standalone direct trace is not integral")
    if accumulator.numerator < 0:
        raise RuntimeError("standalone direct energy became negative")
    return accumulator.numerator


def line_minimum(
    scale_norm: base.E,
    unit: prior.m220.Unit,
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
    *,
    outer_live_scalars: tuple[base.E, ...] = (),
    outer_live_integers: tuple[int, ...] = (),
) -> tuple[int, int]:
    work.line_searches += 1

    def energy(exponent: int, integers: tuple[int, ...]) -> int:
        retained_integers = outer_live_integers + integers + (exponent,)
        work.set_ambient_liveness(
            scalars=outer_live_scalars + (scale_norm,),
            integers=retained_integers,
        )
        try:
            factor = prior.norm_factor(
                unit, exponent, residual, ledger, scratch, work
            )
        finally:
            work.clear_ambient_liveness()
        combined = factor * scale_norm
        work.unit_norm_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=outer_live_scalars + (scale_norm, factor, combined),
            integers=retained_integers,
            context="DIRECT_TRACE_LINE_COMBINED_WEIGHT",
        )
        return direct_trace_energy(
            residual,
            ledger,
            scratch,
            combined,
            work,
            live_integers=retained_integers,
            caller_live_scalars=outer_live_scalars + (scale_norm, factor),
            context_prefix="DIRECT_TRACE_LINE",
        )

    zero = energy(0, (0,))
    positive = energy(1, (0, 1, -1, zero))
    negative = energy(-1, (0, 1, -1, zero, positive))
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=outer_live_scalars + (scale_norm,),
        integers=outer_live_integers + (zero, positive, negative),
        context="DIRECT_TRACE_INITIAL_DIRECTION_ENERGIES",
    )
    if min(positive, negative) >= zero:
        return 0, zero
    direction_sign = 1 if positive < negative else -1
    previous = 0
    current = direction_sign
    current_energy = positive if direction_sign > 0 else negative
    del zero, positive, negative
    for bracket_index in range(m222.MAX_BRACKET_DOUBLINGS):
        following = 2 * current
        work.line_bracket_doublings += 1
        following_energy = energy(
            following,
            (
                bracket_index,
                direction_sign,
                previous,
                current,
                following,
                current_energy,
            ),
        )
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=outer_live_scalars + (scale_norm,),
            integers=outer_live_integers
            + (
                bracket_index,
                direction_sign,
                previous,
                current,
                following,
                current_energy,
                following_energy,
            ),
            context="DIRECT_TRACE_LINE_BRACKET_RESULT",
        )
        if following_energy >= current_energy:
            low, high = sorted((previous, following))
            work.observe(
                residual,
                ledger,
                scratch,
                scalars=outer_live_scalars + (scale_norm,),
                integers=outer_live_integers
                + (
                    bracket_index,
                    direction_sign,
                    previous,
                    current,
                    following,
                    current_energy,
                    following_energy,
                    low,
                    high,
                ),
                context="DIRECT_TRACE_LINE_BRACKET_RELEASE",
            )
            break
        previous, current, current_energy = current, following, following_energy
    else:
        raise RuntimeError("standalone unit line minimum was not bracketed")
    del (
        bracket_index,
        direction_sign,
        previous,
        current,
        current_energy,
        following,
        following_energy,
    )
    while high - low > 8:
        work.line_ternary_steps += 1
        first = low + (high - low) // 3
        second = high - (high - low) // 3
        first_energy = energy(first, (low, high, first, second))
        second_energy = energy(second, (low, high, first, second, first_energy))
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=outer_live_scalars + (scale_norm,),
            integers=outer_live_integers
            + (low, high, first, second, first_energy, second_energy),
            context="DIRECT_TRACE_LINE_TERNARY_RESULT",
        )
        if first_energy <= second_energy:
            high = second - 1
        else:
            low = first + 1
        del first, second, first_energy, second_energy
    selected = low
    selected_energy = energy(low, (low, high, selected))
    for exponent in range(low + 1, high + 1):
        candidate_energy = energy(
            exponent, (low, high, selected, selected_energy)
        )
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=outer_live_scalars + (scale_norm,),
            integers=outer_live_integers
            + (
                low,
                high,
                selected,
                selected_energy,
                exponent,
                candidate_energy,
            ),
            context="DIRECT_TRACE_LINE_SCAN_RESULT",
        )
        if (candidate_energy, exponent) < (selected_energy, selected):
            selected, selected_energy = exponent, candidate_energy
        del candidate_energy
    if high > low:
        del exponent
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=outer_live_scalars + (scale_norm,),
        integers=outer_live_integers + (low, high, selected, selected_energy),
        context="DIRECT_TRACE_LINE_SELECTED_ENERGY",
    )
    return selected, selected_energy


def balance(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
    scale: base.E,
) -> dict[str, object]:
    work.balance_calls += 1
    conjugated_scale = prior.m220.conjugate(scale)
    work.lifecycle_scale_conjugate_field_values_materialized += 1
    scale_norm = scale * conjugated_scale
    work.norm_field_multiplications += 1
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=(scale, conjugated_scale, scale_norm),
        context="DIRECT_TRACE_LEDGER_SCALE_NORM",
    )
    del conjugated_scale
    zero_ledger = [0] * m222.RANK
    raw_payload = prior.stream_payload(
        residual,
        scale,
        residual,
        ledger,
        scratch,
        work,
        candidate=False,
        live_scalars=(scale_norm,),
        live_integers=tuple(zero_ledger),
    )
    identity_total = raw_payload + prior.m220.ledger_bits(zero_ledger)
    identity_energy = direct_trace_energy(
        residual,
        ledger,
        scratch,
        scale_norm,
        work,
        live_integers=(raw_payload, identity_total, *zero_ledger),
        caller_live_scalars=(scale,),
        context_prefix="DIRECT_TRACE_IDENTITY_ENERGY",
    )
    best_key = (
        identity_total,
        raw_payload,
        identity_energy,
        tuple(zero_ledger),
    )
    best_multiplier = scale
    best_ledger = zero_ledger
    exponents: list[int] = []
    del zero_ledger
    for index, unit in enumerate(m222.UNITS):
        retained_integers = (
            raw_payload,
            identity_total,
            best_key[0],
            best_key[1],
            best_key[2],
            *best_key[3],
            *best_ledger,
            *exponents,
            index,
        )
        exponent, energy = line_minimum(
            scale_norm,
            unit,
            residual,
            ledger,
            scratch,
            work,
            outer_live_scalars=(scale, best_multiplier),
            outer_live_integers=retained_integers,
        )
        exponents.append(exponent)
        if exponent == 0:
            work.observe(
                residual,
                ledger,
                scratch,
                scalars=(scale_norm, best_multiplier, scale),
                integers=retained_integers + (exponent, energy, *exponents),
                context="DIRECT_TRACE_ZERO_CANDIDATE_APPEND",
            )
            del exponent, energy, retained_integers
            continue
        candidate_live_integers = retained_integers + (
            exponent,
            energy,
            *exponents,
        )
        work.set_ambient_liveness(
            scalars=(scale_norm, best_multiplier, scale),
            integers=candidate_live_integers,
        )
        try:
            factor = prior.residual_factor(
                unit, exponent, residual, ledger, scratch, work
            )
        finally:
            work.clear_ambient_liveness()
        multiplier = factor * scale
        work.ledger_scale_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(scale_norm, best_multiplier, scale, factor, multiplier),
            integers=candidate_live_integers,
            context="DIRECT_TRACE_CANDIDATE_NET_UNIT_MULTIPLIER",
        )
        candidate_ledger = [0] * m222.RANK
        candidate_ledger[index] = exponent
        candidate_payload = prior.stream_payload(
            residual,
            multiplier,
            residual,
            ledger,
            scratch,
            work,
            candidate=True,
            live_scalars=(scale_norm, best_multiplier, scale, factor),
            live_integers=candidate_live_integers + tuple(candidate_ledger),
        )
        key = (
            candidate_payload + prior.m220.ledger_bits(candidate_ledger),
            candidate_payload,
            energy,
            tuple(candidate_ledger),
        )
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(scale_norm, best_multiplier, scale, factor, multiplier),
            integers=candidate_live_integers
            + (
                *candidate_ledger,
                candidate_payload,
                key[0],
                key[1],
                key[2],
                *key[3],
            ),
            context="DIRECT_TRACE_CANDIDATE_KEY",
        )
        if key < best_key:
            best_key = key
            best_multiplier = multiplier
            best_ledger = candidate_ledger
            work.observe(
                residual,
                ledger,
                scratch,
                scalars=(
                    scale_norm,
                    best_multiplier,
                    scale,
                    factor,
                    multiplier,
                ),
                integers=candidate_live_integers
                + (
                    *candidate_ledger,
                    candidate_payload,
                    key[0],
                    key[1],
                    key[2],
                    *key[3],
                    best_key[0],
                    best_key[1],
                    best_key[2],
                    *best_key[3],
                    *best_ledger,
                ),
                context="DIRECT_TRACE_WINNING_KEY_TRANSFER",
            )
        del (
            exponent,
            energy,
            retained_integers,
            candidate_live_integers,
            factor,
            multiplier,
            candidate_ledger,
            candidate_payload,
            key,
        )
    del index, unit
    work.set_ambient_liveness(
        scalars=(),
        integers=(
            raw_payload,
            identity_total,
            best_key[0],
            best_key[1],
            best_key[2],
            *best_key[3],
            *best_ledger,
            *exponents,
        ),
    )
    try:
        prior.mutate_selected(
            residual,
            ledger,
            scratch,
            best_multiplier,
            work,
            (scale, scale_norm),
        )
    finally:
        work.clear_ambient_liveness()
    final_live_integers = (
        raw_payload,
        identity_total,
        best_key[0],
        best_key[1],
        best_key[2],
        *best_key[3],
        *best_ledger,
        *exponents,
    )
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=(scale, scale_norm, best_multiplier),
        integers=final_live_integers,
        context="DIRECT_TRACE_LEDGER_COMMIT_BEFORE",
    )
    ledger[:] = best_ledger
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=(scale, scale_norm, best_multiplier),
        integers=final_live_integers,
        context="DIRECT_TRACE_LEDGER_COMMIT_AFTER",
    )
    return {
        "raw_payload_bits": raw_payload,
        "balanced_residual_payload_bits": base.payload_bits(residual),
        "unit_ledger_payload_bits": prior.m220.ledger_bits(ledger),
        "balanced_residual_plus_ledger_payload_bits": best_key[0],
        "resident_payload_reduction_bits_before_constant_scratch": (
            identity_total - best_key[0]
        ),
        "selected_unit_ledger": list(ledger),
        "per_direction_trace_energy_minimizing_exponents": exponents,
        "selected_exact_embedding_energy_bits": prior.signed_bits(best_key[2]),
        "selected_exact_embedding_energy_sha256": hashlib.sha256(
            str(best_key[2]).encode("ascii")
        ).hexdigest(),
        "identity_selected": not any(ledger),
    }


def rematerialize_ledger_norm(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
    live_integers: tuple[int, ...],
) -> base.E:
    work.ledger_norm_rematerializations += 1
    result = base.ONE
    for index, (exponent, unit) in enumerate(zip(ledger, m222.UNITS)):
        if exponent == 0:
            continue
        value = unit.norm if exponent > 0 else unit.reciprocal_norm
        work.set_ambient_liveness(
            scalars=(result,),
            integers=live_integers + (index, exponent),
        )
        try:
            factor = prior.counted_power(
                value,
                abs(exponent),
                residual,
                ledger,
                scratch,
                work,
            )
        finally:
            work.clear_ambient_liveness()
        updated = result * factor
        work.rematerialization_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(result, factor, updated),
            integers=live_integers + (index, exponent),
            context="REMATERIALIZED_LEDGER_NORM_ACCUMULATE",
        )
        result = updated
        del value, factor, updated
    return result


def rematerialized_trace_energy(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
    unit: m222.prior.m220.Unit | None,
    exponent: int,
    live_integers: tuple[int, ...],
    context_prefix: str,
) -> int:
    weight = rematerialize_ledger_norm(
        residual,
        ledger,
        scratch,
        work,
        live_integers + (exponent,),
    )
    if unit is not None and exponent != 0:
        work.set_ambient_liveness(
            scalars=(weight,),
            integers=live_integers + (exponent,),
        )
        try:
            factor = prior.norm_factor(
                unit, exponent, residual, ledger, scratch, work
            )
        finally:
            work.clear_ambient_liveness()
        updated = weight * factor
        work.rematerialization_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(weight, factor, updated),
            integers=live_integers + (exponent,),
            context="REMATERIALIZED_DIRECT_TRACE_WEIGHT",
        )
        weight = updated
        del factor, updated
    work.direct_trace_weight_rematerializations += 1
    return direct_trace_energy(
        residual,
        ledger,
        scratch,
        weight,
        work,
        live_integers=live_integers + (exponent,),
        caller_live_scalars=(),
        context_prefix=context_prefix,
    )


def rematerialized_line_minimum(
    unit: m222.prior.m220.Unit,
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
    outer_live_integers: tuple[int, ...],
) -> tuple[int, int]:
    work.line_searches += 1

    def energy(exponent: int, integers: tuple[int, ...]) -> int:
        return rematerialized_trace_energy(
            residual,
            ledger,
            scratch,
            work,
            unit,
            exponent,
            outer_live_integers + integers,
            "REMATERIALIZED_DIRECT_TRACE_LINE",
        )

    zero = energy(0, (0,))
    positive = energy(1, (0, 1, -1, zero))
    negative = energy(-1, (0, 1, -1, zero, positive))
    work.observe(
        residual,
        ledger,
        scratch,
        integers=outer_live_integers + (zero, positive, negative),
        context="REMATERIALIZED_DIRECT_TRACE_INITIAL_DIRECTION_ENERGIES",
    )
    if min(positive, negative) >= zero:
        return 0, zero
    direction_sign = 1 if positive < negative else -1
    previous = 0
    current = direction_sign
    current_energy = positive if direction_sign > 0 else negative
    del zero, positive, negative
    for bracket_index in range(m222.MAX_BRACKET_DOUBLINGS):
        following = 2 * current
        work.line_bracket_doublings += 1
        following_energy = energy(
            following,
            (
                bracket_index,
                direction_sign,
                previous,
                current,
                following,
                current_energy,
            ),
        )
        work.observe(
            residual,
            ledger,
            scratch,
            integers=outer_live_integers
            + (
                bracket_index,
                direction_sign,
                previous,
                current,
                following,
                current_energy,
                following_energy,
            ),
            context="REMATERIALIZED_DIRECT_TRACE_LINE_BRACKET_RESULT",
        )
        if following_energy >= current_energy:
            low, high = sorted((previous, following))
            work.observe(
                residual,
                ledger,
                scratch,
                integers=outer_live_integers
                + (
                    bracket_index,
                    direction_sign,
                    previous,
                    current,
                    following,
                    current_energy,
                    following_energy,
                    low,
                    high,
                ),
                context="REMATERIALIZED_DIRECT_TRACE_LINE_BRACKET_RELEASE",
            )
            break
        previous, current, current_energy = current, following, following_energy
    else:
        raise RuntimeError("standalone rematerialized line minimum was not bracketed")
    del (
        bracket_index,
        direction_sign,
        previous,
        current,
        current_energy,
        following,
        following_energy,
    )
    while high - low > 8:
        work.line_ternary_steps += 1
        first = low + (high - low) // 3
        second = high - (high - low) // 3
        first_energy = energy(first, (low, high, first, second))
        second_energy = energy(second, (low, high, first, second, first_energy))
        work.observe(
            residual,
            ledger,
            scratch,
            integers=outer_live_integers
            + (low, high, first, second, first_energy, second_energy),
            context="REMATERIALIZED_DIRECT_TRACE_LINE_TERNARY_RESULT",
        )
        if first_energy <= second_energy:
            high = second - 1
        else:
            low = first + 1
        del first, second, first_energy, second_energy
    selected = low
    selected_energy = energy(low, (low, high, selected))
    for exponent in range(low + 1, high + 1):
        candidate_energy = energy(
            exponent, (low, high, selected, selected_energy)
        )
        work.observe(
            residual,
            ledger,
            scratch,
            integers=outer_live_integers
            + (
                low,
                high,
                selected,
                selected_energy,
                exponent,
                candidate_energy,
            ),
            context="REMATERIALIZED_DIRECT_TRACE_LINE_SCAN_RESULT",
        )
        if (candidate_energy, exponent) < (selected_energy, selected):
            selected, selected_energy = exponent, candidate_energy
        del candidate_energy
    if high > low:
        del exponent
    work.observe(
        residual,
        ledger,
        scratch,
        integers=outer_live_integers + (low, high, selected, selected_energy),
        context="REMATERIALIZED_DIRECT_TRACE_LINE_SELECTED_ENERGY",
    )
    return selected, selected_energy


def rematerialized_scaled_payload(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
    candidate_unit: m222.prior.m220.Unit | None,
    candidate_exponent: int,
    live_integers: tuple[int, ...],
) -> int:
    work.set_ambient_liveness(
        scalars=(), integers=live_integers + (candidate_exponent,)
    )
    try:
        multiplier = prior.ledger_scale(residual, ledger, scratch, work)
    finally:
        work.clear_ambient_liveness()
    work.ledger_scale_rematerializations += 1
    candidate = candidate_unit is not None and candidate_exponent != 0
    if candidate:
        work.set_ambient_liveness(
            scalars=(multiplier,),
            integers=live_integers + (candidate_exponent,),
        )
        try:
            factor = prior.residual_factor(
                candidate_unit,
                candidate_exponent,
                residual,
                ledger,
                scratch,
                work,
            )
        finally:
            work.clear_ambient_liveness()
        updated = factor * multiplier
        work.rematerialization_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(multiplier, factor, updated),
            integers=live_integers + (candidate_exponent,),
            context="REMATERIALIZED_CANDIDATE_MULTIPLIER",
        )
        multiplier = updated
        work.candidate_multiplier_rematerializations += 1
        del factor, updated
    return prior.stream_payload(
        residual,
        multiplier,
        residual,
        ledger,
        scratch,
        work,
        candidate=candidate,
        live_scalars=(),
        live_integers=live_integers + (candidate_exponent,),
    )


def rematerialized_apply_selected(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    selected_ledger: list[int],
    work: Work,
    live_integers: tuple[int, ...],
) -> None:
    work.set_ambient_liveness(scalars=(), integers=live_integers)
    try:
        multiplier = prior.ledger_scale(residual, ledger, scratch, work)
    finally:
        work.clear_ambient_liveness()
    work.ledger_scale_rematerializations += 1
    for index, (exponent, unit) in enumerate(zip(selected_ledger, m222.UNITS)):
        if exponent == 0:
            continue
        work.set_ambient_liveness(
            scalars=(multiplier,),
            integers=live_integers + (index, exponent, *selected_ledger),
        )
        try:
            factor = prior.residual_factor(
                unit, exponent, residual, ledger, scratch, work
            )
        finally:
            work.clear_ambient_liveness()
        updated = factor * multiplier
        work.rematerialization_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(multiplier, factor, updated),
            integers=live_integers + (index, exponent, *selected_ledger),
            context="REMATERIALIZED_SELECTED_MULTIPLIER",
        )
        multiplier = updated
        del factor, updated
    work.selected_multiplier_rematerializations += 1
    work.set_ambient_liveness(scalars=(), integers=live_integers)
    try:
        prior.mutate_selected(
            residual,
            ledger,
            scratch,
            multiplier,
            work,
            (),
        )
    finally:
        work.clear_ambient_liveness()


def rematerialized_balance(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
) -> dict[str, object]:
    work.balance_calls += 1
    zero_ledger = [0] * m222.RANK
    raw_payload = rematerialized_scaled_payload(
        residual, ledger, scratch, work, None, 0, tuple(zero_ledger)
    )
    identity_total = raw_payload + prior.m220.ledger_bits(zero_ledger)
    identity_energy = rematerialized_trace_energy(
        residual,
        ledger,
        scratch,
        work,
        None,
        0,
        (raw_payload, identity_total, *zero_ledger),
        "REMATERIALIZED_DIRECT_TRACE_IDENTITY_ENERGY",
    )
    best_key = (
        identity_total,
        raw_payload,
        identity_energy,
        tuple(zero_ledger),
    )
    best_ledger = zero_ledger
    exponents: list[int] = []
    del zero_ledger
    for index, unit in enumerate(m222.UNITS):
        retained_integers = (
            raw_payload,
            identity_total,
            best_key[0],
            best_key[1],
            best_key[2],
            *best_key[3],
            *best_ledger,
            *exponents,
            index,
        )
        exponent, energy = rematerialized_line_minimum(
            unit,
            residual,
            ledger,
            scratch,
            work,
            retained_integers,
        )
        exponents.append(exponent)
        candidate_live_integers = retained_integers + (
            exponent,
            energy,
            *exponents,
        )
        if exponent == 0:
            work.observe(
                residual,
                ledger,
                scratch,
                integers=candidate_live_integers,
                context="REMATERIALIZED_ZERO_CANDIDATE_APPEND",
            )
            del exponent, energy, retained_integers, candidate_live_integers
            continue
        candidate_ledger = [0] * m222.RANK
        candidate_ledger[index] = exponent
        candidate_payload = rematerialized_scaled_payload(
            residual,
            ledger,
            scratch,
            work,
            unit,
            exponent,
            candidate_live_integers + tuple(candidate_ledger),
        )
        key = (
            candidate_payload + prior.m220.ledger_bits(candidate_ledger),
            candidate_payload,
            energy,
            tuple(candidate_ledger),
        )
        work.observe(
            residual,
            ledger,
            scratch,
            integers=candidate_live_integers
            + (
                *candidate_ledger,
                candidate_payload,
                key[0],
                key[1],
                key[2],
                *key[3],
            ),
            context="REMATERIALIZED_CANDIDATE_KEY",
        )
        if key < best_key:
            best_key = key
            best_ledger = candidate_ledger
            work.observe(
                residual,
                ledger,
                scratch,
                integers=candidate_live_integers
                + (
                    *candidate_ledger,
                    candidate_payload,
                    key[0],
                    key[1],
                    key[2],
                    *key[3],
                    best_key[0],
                    best_key[1],
                    best_key[2],
                    *best_key[3],
                    *best_ledger,
                ),
                context="REMATERIALIZED_WINNING_KEY_TRANSFER",
            )
        del (
            exponent,
            energy,
            retained_integers,
            candidate_live_integers,
            candidate_ledger,
            candidate_payload,
            key,
        )
    del index, unit
    final_live_integers = (
        raw_payload,
        identity_total,
        best_key[0],
        best_key[1],
        best_key[2],
        *best_key[3],
        *best_ledger,
        *exponents,
    )
    rematerialized_apply_selected(
        residual, ledger, scratch, best_ledger, work, final_live_integers
    )
    work.observe(
        residual,
        ledger,
        scratch,
        integers=final_live_integers,
        context="REMATERIALIZED_LEDGER_COMMIT_BEFORE",
    )
    ledger[:] = best_ledger
    work.observe(
        residual,
        ledger,
        scratch,
        integers=final_live_integers,
        context="REMATERIALIZED_LEDGER_COMMIT_AFTER",
    )
    return {
        "raw_payload_bits": raw_payload,
        "balanced_residual_payload_bits": base.payload_bits(residual),
        "unit_ledger_payload_bits": prior.m220.ledger_bits(ledger),
        "balanced_residual_plus_ledger_payload_bits": best_key[0],
        "resident_payload_reduction_bits_before_constant_scratch": (
            identity_total - best_key[0]
        ),
        "selected_unit_ledger": list(ledger),
        "per_direction_trace_energy_minimizing_exponents": exponents,
        "selected_exact_embedding_energy_bits": prior.signed_bits(best_key[2]),
        "selected_exact_embedding_energy_sha256": hashlib.sha256(
            str(best_key[2]).encode("ascii")
        ).hexdigest(),
        "identity_selected": not any(ledger),
    }


def rematerialized_apply_operation(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    topology: base.Topology,
    operation: base.Operation,
    work: Work,
    *,
    inverse: bool,
) -> dict[str, object]:
    scale = prior.ledger_scale(residual, ledger, scratch, work)
    work.ledger_scale_rematerializations += 1
    if inverse:
        prior.apply_shear(
            residual,
            ledger,
            scratch,
            topology,
            operation,
            work,
            scale,
            inverse=True,
        )
        base.apply_gate(
            residual,
            scratch,
            topology,
            base.Operation(operation.generator, -operation.exponent),
        )
    else:
        base.apply_gate(residual, scratch, topology, operation)
        prior.apply_shear(
            residual,
            ledger,
            scratch,
            topology,
            operation,
            work,
            scale,
            inverse=False,
        )
    del scale
    metrics = rematerialized_balance(residual, ledger, scratch, work)
    scratch[:] = [base.ZERO] * topology.dimension
    work.observe(residual, ledger, scratch)
    return metrics


def main() -> None:
    m222.Work = Work
    m222.streamed_energy = direct_trace_energy
    m222.line_minimum = rematerialized_line_minimum
    m222.balance = rematerialized_balance
    prior.Work = Work
    prior.balance = rematerialized_balance
    prior.apply_operation = rematerialized_apply_operation
    print(
        json.dumps(
            {
                "schema": "cat_cas.su2_level8_cubic_skein_rematerialized_trace_reference.v1",
                "imports_m224_production": False,
                "imports_m223_production": False,
                "imports_m222_production": False,
                "imports_m221_production": False,
                "uses_prior_standalone_m222_reference_substrate": True,
                "cases": [prior.case(*item) for item in CASES],
                "reuse": prior.reuse(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
