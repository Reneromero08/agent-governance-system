#!/usr/bin/env python3
"""M224 ledger-rematerialized direct coordinate trace.

M222 streams one weighted Hermitian cell at a time, but still materializes the
cyclotomic field product ``weight * value * conjugate(value)``.  M223 expands
only the scalar trace in the public power basis:

    Trace(w v conjugate(v))
      = sum[a,b,c] w[a] v[b] v[c] RamanujanTrace(a+b-c).

M224 keeps the M223 coordinate formula but removes derived caller field values
from the direct-trace live interval.  Exact line weights and residual rebasing
multipliers are rematerialized from the resident unit ledger and public unit
directions, consumed, and uncomputed by release.  The identical recurrence
remains classical and all rematerialization work is counted.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import su2_level8_cubic_skein_streamed_embedding_energy as m222


sys.set_int_max_str_digits(0)
m221 = m222.m221
braid = m222.braid
CASES = m222.CASES


def fraction_payload(value: Fraction) -> int:
    return m221.signed_bits(value.numerator) + value.denominator.bit_length()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        scalars: tuple[braid.K, ...],
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
        residual: list[braid.K],
        ledger: list[int],
        scratch: list[braid.K],
        *,
        scalars: Iterable[braid.K] = (),
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
        integers: Iterable[int],
        context_prefix: str,
    ) -> None:
        integer_values = tuple(integers)
        self.maximum_direct_trace_caller_scalar_cells = max(
            self.maximum_direct_trace_caller_scalar_cells,
            caller_scalar_cells,
        )
        integer_bits = sum(m221.signed_bits(value) for value in integer_values)
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
            self.maximum_scalar_payload_bits, weight_bits
            + caller_scalar_bits
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
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    weight: braid.K,
    work: Work,
    *,
    live_integers: tuple[int, ...] = (),
    caller_live_scalars: tuple[braid.K, ...] = (),
    context_prefix: str,
) -> int:
    work.exact_trace_energy_evaluations += 1
    work.streamed_embedding_energy_evaluations += 1
    work.direct_trace_energy_evaluations += 1
    accumulator = Fraction(0)
    scratch_bits = braid.field_payload_bits(scratch)
    resident_bits = (
        braid.field_payload_bits(residual)
        + m221.m220.ledger_payload_bits(ledger)
        + scratch_bits
    )
    weight_bits = m221.scalar_payload(weight)
    caller_scalar_bits = sum(
        m221.scalar_payload(value) for value in caller_live_scalars
    )
    base_field_cells = len(residual) + len(scratch) + 1
    for cell_index, value in enumerate(residual):
        work.streamed_embedding_cells_scanned += 1
        work.direct_trace_cells_scanned += 1
        for a, weight_coordinate in enumerate(weight.coefficients):
            if not weight_coordinate:
                continue
            for b, left_coordinate in enumerate(value.coefficients):
                if not left_coordinate:
                    continue
                left = weight_coordinate * left_coordinate
                work.direct_trace_nonzero_coordinate_pairs += 1
                work.direct_trace_rational_multiplications += 1
                for c, right_coordinate in enumerate(value.coefficients):
                    if not right_coordinate:
                        continue
                    trace_coefficient = m221.m220.ramanujan_trace(a + b - c)
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
        raise RuntimeError("direct coordinate trace is not integral")
    if accumulator.numerator < 0:
        raise RuntimeError("direct coordinate energy became negative")
    return accumulator.numerator


def line_minimum(
    scale_norm: braid.K,
    direction: m221.m220.UnitDirection,
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
    *,
    outer_live_scalars: tuple[braid.K, ...] = (),
    outer_live_integers: tuple[int, ...] = (),
) -> tuple[int, int]:
    """M222 line search with every caller-retained exact scalar accounted."""

    work.line_searches += 1

    def energy(exponent: int, integers: tuple[int, ...]) -> int:
        retained_integers = outer_live_integers + integers + (exponent,)
        work.set_ambient_liveness(
            scalars=outer_live_scalars + (scale_norm,),
            integers=retained_integers,
        )
        try:
            factor = m221.norm_factor(
                direction, exponent, residual, ledger, scratch, work
            )
        finally:
            work.clear_ambient_liveness()
        combined_weight = factor * scale_norm
        work.unit_norm_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=outer_live_scalars
            + (scale_norm, factor, combined_weight),
            integers=retained_integers,
            context="DIRECT_TRACE_LINE_COMBINED_WEIGHT",
        )
        return direct_trace_energy(
            residual,
            ledger,
            scratch,
            combined_weight,
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
        raise RuntimeError("unit line minimum was not bracketed")
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


def balance_resident(
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
    *,
    scale: braid.K,
) -> dict[str, Any]:
    """M222 ledger balance with direct-trace caller liveness made explicit."""

    work.balance_calls += 1
    scale_conjugate = m221.m220.conjugate(scale)
    work.lifecycle_scale_conjugate_field_values_materialized += 1
    scale_norm = scale * scale_conjugate
    work.norm_field_multiplications += 1
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=(scale, scale_conjugate, scale_norm),
        context="DIRECT_TRACE_LEDGER_SCALE_NORM",
    )
    del scale_conjugate
    zero_ledger = [0] * m222.UNIT_RANK
    raw_payload = m221.stream_scaled_payload(
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
    identity_total = raw_payload + m221.m220.ledger_payload_bits(zero_ledger)
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
    candidate_exponents: list[int] = []
    del zero_ledger
    for index, direction in enumerate(m222.UNITS):
        retained_integers = (
            raw_payload,
            identity_total,
            best_key[0],
            best_key[1],
            best_key[2],
            *best_key[3],
            *best_ledger,
            *candidate_exponents,
            index,
        )
        exponent, energy = line_minimum(
            scale_norm,
            direction,
            residual,
            ledger,
            scratch,
            work,
            outer_live_scalars=(scale, best_multiplier),
            outer_live_integers=retained_integers,
        )
        candidate_exponents.append(exponent)
        if exponent == 0:
            work.observe(
                residual,
                ledger,
                scratch,
                scalars=(scale_norm, best_multiplier, scale),
                integers=retained_integers
                + (exponent, energy, *candidate_exponents),
                context="DIRECT_TRACE_ZERO_CANDIDATE_APPEND",
            )
            del exponent, energy, retained_integers
            continue
        candidate_live_integers = retained_integers + (
            exponent,
            energy,
            *candidate_exponents,
        )
        work.set_ambient_liveness(
            scalars=(scale_norm, best_multiplier, scale),
            integers=candidate_live_integers,
        )
        try:
            factor = m221.residual_factor(
                direction, exponent, residual, ledger, scratch, work
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
        candidate_ledger = [0] * m222.UNIT_RANK
        candidate_ledger[index] = exponent
        candidate_payload = m221.stream_scaled_payload(
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
            candidate_payload
            + m221.m220.ledger_payload_bits(candidate_ledger),
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
    del index, direction
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
            *candidate_exponents,
        ),
    )
    try:
        m221.apply_selected_net(
            residual,
            ledger,
            scratch,
            best_multiplier,
            work,
            live_scalars=(scale, scale_norm),
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
        *candidate_exponents,
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
        "balanced_residual_payload_bits": braid.field_payload_bits(residual),
        "unit_ledger_payload_bits": m221.m220.ledger_payload_bits(ledger),
        "balanced_residual_plus_ledger_payload_bits": best_key[0],
        "resident_payload_reduction_bits_before_constant_scratch": (
            identity_total - best_key[0]
        ),
        "selected_unit_ledger": list(ledger),
        "per_direction_trace_energy_minimizing_exponents": candidate_exponents,
        "selected_exact_embedding_energy_bits": m221.signed_bits(best_key[2]),
        "selected_exact_embedding_energy_sha256": hashlib.sha256(
            str(best_key[2]).encode("ascii")
        ).hexdigest(),
        "identity_selected": not any(ledger),
    }


def rematerialize_ledger_norm(
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
    *,
    live_integers: tuple[int, ...],
) -> braid.K:
    """Rebuild represented-scale norm from the resident ledger only."""

    work.ledger_norm_rematerializations += 1
    result = braid.ONE
    for index, (exponent, direction) in enumerate(
        zip(ledger, m222.UNITS, strict=True)
    ):
        if exponent == 0:
            continue
        base = direction.norm if exponent > 0 else direction.inverse_norm
        work.set_ambient_liveness(
            scalars=(result,),
            integers=live_integers + (index, exponent),
        )
        try:
            factor = m221.counted_power(
                base,
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
        del base, factor, updated
    return result


def rematerialized_trace_energy(
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
    *,
    direction: m221.m220.UnitDirection | None,
    exponent: int,
    live_integers: tuple[int, ...],
    context_prefix: str,
) -> int:
    """Build one trace weight, release its factors, then consume it."""

    weight = rematerialize_ledger_norm(
        residual,
        ledger,
        scratch,
        work,
        live_integers=live_integers + (exponent,),
    )
    if direction is not None and exponent != 0:
        work.set_ambient_liveness(
            scalars=(weight,),
            integers=live_integers + (exponent,),
        )
        try:
            factor = m221.norm_factor(
                direction,
                exponent,
                residual,
                ledger,
                scratch,
                work,
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
    direction: m221.m220.UnitDirection,
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
    *,
    outer_live_integers: tuple[int, ...],
) -> tuple[int, int]:
    """Exact line search with no retained cyclotomic caller scalar."""

    work.line_searches += 1

    def energy(exponent: int, integers: tuple[int, ...]) -> int:
        return rematerialized_trace_energy(
            residual,
            ledger,
            scratch,
            work,
            direction=direction,
            exponent=exponent,
            live_integers=outer_live_integers + integers,
            context_prefix="REMATERIALIZED_DIRECT_TRACE_LINE",
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
        raise RuntimeError("rematerialized unit line minimum was not bracketed")
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
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
    *,
    candidate_direction: m221.m220.UnitDirection | None,
    candidate_exponent: int,
    live_integers: tuple[int, ...],
) -> int:
    """Stream raw or candidate cells from one rematerialized multiplier."""

    work.set_ambient_liveness(
        scalars=(), integers=live_integers + (candidate_exponent,)
    )
    try:
        multiplier = m221.ledger_scale(residual, ledger, scratch, work)
    finally:
        work.clear_ambient_liveness()
    work.ledger_scale_rematerializations += 1
    candidate = candidate_direction is not None and candidate_exponent != 0
    if candidate:
        work.set_ambient_liveness(
            scalars=(multiplier,),
            integers=live_integers + (candidate_exponent,),
        )
        try:
            factor = m221.residual_factor(
                candidate_direction,
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
    return m221.stream_scaled_payload(
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
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    selected_ledger: list[int],
    work: Work,
    *,
    live_integers: tuple[int, ...],
) -> None:
    """Rebuild and consume the selected residual-rebasing multiplier."""

    work.set_ambient_liveness(scalars=(), integers=live_integers)
    try:
        multiplier = m221.ledger_scale(residual, ledger, scratch, work)
    finally:
        work.clear_ambient_liveness()
    work.ledger_scale_rematerializations += 1
    for index, (exponent, direction) in enumerate(
        zip(selected_ledger, m222.UNITS, strict=True)
    ):
        if exponent == 0:
            continue
        work.set_ambient_liveness(
            scalars=(multiplier,),
            integers=live_integers + (index, exponent, *selected_ledger),
        )
        try:
            factor = m221.residual_factor(
                direction,
                exponent,
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
            integers=live_integers + (index, exponent, *selected_ledger),
            context="REMATERIALIZED_SELECTED_MULTIPLIER",
        )
        multiplier = updated
        del factor, updated
    work.selected_multiplier_rematerializations += 1
    work.set_ambient_liveness(scalars=(), integers=live_integers)
    try:
        m221.apply_selected_net(
            residual,
            ledger,
            scratch,
            multiplier,
            work,
            live_scalars=(),
        )
    finally:
        work.clear_ambient_liveness()


def rematerialized_balance_resident(
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
) -> dict[str, Any]:
    """Balance using ledger/public-descriptor rematerialization only."""

    work.balance_calls += 1
    zero_ledger = [0] * m222.UNIT_RANK
    raw_payload = rematerialized_scaled_payload(
        residual,
        ledger,
        scratch,
        work,
        candidate_direction=None,
        candidate_exponent=0,
        live_integers=tuple(zero_ledger),
    )
    identity_total = raw_payload + m221.m220.ledger_payload_bits(zero_ledger)
    identity_energy = rematerialized_trace_energy(
        residual,
        ledger,
        scratch,
        work,
        direction=None,
        exponent=0,
        live_integers=(raw_payload, identity_total, *zero_ledger),
        context_prefix="REMATERIALIZED_DIRECT_TRACE_IDENTITY_ENERGY",
    )
    best_key = (
        identity_total,
        raw_payload,
        identity_energy,
        tuple(zero_ledger),
    )
    best_ledger = zero_ledger
    candidate_exponents: list[int] = []
    del zero_ledger
    for index, direction in enumerate(m222.UNITS):
        retained_integers = (
            raw_payload,
            identity_total,
            best_key[0],
            best_key[1],
            best_key[2],
            *best_key[3],
            *best_ledger,
            *candidate_exponents,
            index,
        )
        exponent, energy = rematerialized_line_minimum(
            direction,
            residual,
            ledger,
            scratch,
            work,
            outer_live_integers=retained_integers,
        )
        candidate_exponents.append(exponent)
        candidate_live_integers = retained_integers + (
            exponent,
            energy,
            *candidate_exponents,
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
        candidate_ledger = [0] * m222.UNIT_RANK
        candidate_ledger[index] = exponent
        candidate_payload = rematerialized_scaled_payload(
            residual,
            ledger,
            scratch,
            work,
            candidate_direction=direction,
            candidate_exponent=exponent,
            live_integers=candidate_live_integers + tuple(candidate_ledger),
        )
        key = (
            candidate_payload
            + m221.m220.ledger_payload_bits(candidate_ledger),
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
    del index, direction
    final_live_integers = (
        raw_payload,
        identity_total,
        best_key[0],
        best_key[1],
        best_key[2],
        *best_key[3],
        *best_ledger,
        *candidate_exponents,
    )
    rematerialized_apply_selected(
        residual,
        ledger,
        scratch,
        best_ledger,
        work,
        live_integers=final_live_integers,
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
        "balanced_residual_payload_bits": braid.field_payload_bits(residual),
        "unit_ledger_payload_bits": m221.m220.ledger_payload_bits(ledger),
        "balanced_residual_plus_ledger_payload_bits": best_key[0],
        "resident_payload_reduction_bits_before_constant_scratch": (
            identity_total - best_key[0]
        ),
        "selected_unit_ledger": list(ledger),
        "per_direction_trace_energy_minimizing_exponents": candidate_exponents,
        "selected_exact_embedding_energy_bits": m221.signed_bits(best_key[2]),
        "selected_exact_embedding_energy_sha256": hashlib.sha256(
            str(best_key[2]).encode("ascii")
        ).hexdigest(),
        "identity_selected": not any(ledger),
    }


def rematerialized_apply_operation(
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    topology: m221.skein.DiagramTopology,
    operation: braid.BraidOperation,
    work: Work,
    *,
    inverse: bool,
) -> dict[str, Any]:
    """Release the operation scale before the balancing transaction."""

    scale = m221.ledger_scale(residual, ledger, scratch, work)
    work.ledger_scale_rematerializations += 1
    if inverse:
        m221.apply_ledger_cubic_shear(
            residual,
            ledger,
            scratch,
            topology,
            operation,
            work,
            scale=scale,
            inverse=True,
        )
        m221.skein.apply_gate(
            residual,
            scratch,
            topology,
            braid.BraidOperation(operation.generator, -operation.exponent),
            work.cubic.linear,
        )
        work.cubic.inverse_operations += 1
    else:
        m221.skein.apply_gate(
            residual, scratch, topology, operation, work.cubic.linear
        )
        m221.apply_ledger_cubic_shear(
            residual,
            ledger,
            scratch,
            topology,
            operation,
            work,
            scale=scale,
            inverse=False,
        )
        work.cubic.forward_operations += 1
    del scale
    metrics = rematerialized_balance_resident(residual, ledger, scratch, work)
    scratch[:] = [braid.ZERO] * topology.dimension
    work.observe(residual, ledger, scratch)
    return metrics


def install_direct_trace_path() -> None:
    m222.Work = Work
    m222.streamed_embedding_energy = direct_trace_energy
    m222.line_minimum = rematerialized_line_minimum
    m222.balance_resident = rematerialized_balance_resident
    m221.Work = Work
    m221.balance_resident = rematerialized_balance_resident
    m221.apply_operation = rematerialized_apply_operation


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: su2_level8_cubic_skein_rematerialized_trace.py "
            "SEPARATE_REFERENCE_JSON"
        )
    here = Path(__file__).resolve().parent
    reference_path = Path(sys.argv[1]).resolve()
    if str(reference_path).startswith(("/dev/shm/", "/run/shm/")):
        raise ValueError("RAM-backed M224 reference is forbidden")
    reference = json.loads(reference_path.read_text())
    if reference.get("schema") != "cat_cas.su2_level8_cubic_skein_rematerialized_trace_reference.v1":
        raise RuntimeError("M224 separate-reference schema changed")
    install_direct_trace_path()
    cases = [m221.execute_case(*case) for case in CASES]
    if [m221.reference_case_view(case) for case in cases] != reference.get("cases"):
        raise RuntimeError("M224 independent case and resource parity failed")
    reuse = m221.reuse_result()
    for section in ("primary", "reuse", "fresh_reuse"):
        for key in (
            "boundary_commitment",
            "forward_state_commitment",
            "forward_raw_payload_bits",
            "final_balance",
            "restoration_error_field_cells",
            "canonical_post_restoration_state_exact",
            "declared_live_payload_reduction_vs_raw_bits",
            "same_residual_backing",
            "same_unit_ledger_backing",
            "same_scratch_backing",
            "baseline_reload_used",
            "restoration_generation",
            "matched_raw_recurrence",
        ):
            if reuse[section][key] != reference["reuse"][section][key]:
                raise RuntimeError(
                    f"M224 independent reuse parity failed: {section}.{key}"
                )
        normalized_work = m221.reference_case_view(reuse[section])["work"]
        if normalized_work != reference["reuse"][section]["work"]:
            raise RuntimeError(
                f"M224 independent normalized work parity failed: {section}"
            )
    for key in (
        "fresh_restored_reuse_boundary_agreement",
        "fresh_restored_reuse_state_agreement",
        "restoration_generation_after_reuse",
    ):
        if reuse[key] != reference["reuse"][key]:
            raise RuntimeError(f"M224 independent reuse parity failed: {key}")
    controls = m221.controls()
    positive_controls = {
        key: value
        for key, value in controls.items()
        if key
        not in {
            "raw_actual_vector_materialized",
            "candidate_residual_vector_materialized",
            "intermediate_actual_vector_projected",
            "snapshot_command_available",
        }
    }
    if (
        not all(positive_controls.values())
        or controls["raw_actual_vector_materialized"]
        or controls["candidate_residual_vector_materialized"]
        or controls["intermediate_actual_vector_projected"]
        or controls["snapshot_command_available"]
    ):
        raise RuntimeError("M224 control failed")
    selected = [
        {
            "strands": case["strands"],
            "rounds": case["rounds"],
            "forward_raw_payload_bits": case["forward_raw_payload_bits"],
            "balanced_residual_plus_ledger_payload_bits": case["final_balance"][
                "balanced_residual_plus_ledger_payload_bits"
            ],
            "rematerialized_trace_maximum_declared_live_payload_bits": case["work"][
                "maximum_declared_live_payload_bits"
            ],
            "raw_maximum_declared_live_payload_bits": case[
                "matched_raw_recurrence"
            ]["maximum_declared_live_payload_bits"],
            "declared_live_payload_reduction_vs_raw_bits": case[
                "declared_live_payload_reduction_vs_raw_bits"
            ],
            "rematerialized_trace_maximum_context": case["work"][
                "maximum_declared_live_context"
            ],
        }
        for case in cases
        if (case["strands"], case["rounds"]) in ((4, 4), (6, 2), (8, 1))
    ]
    all_nontrivial_smaller = all(
        case["declared_live_payload_reduction_vs_raw_bits"] > 0
        for case in cases
        if case["rounds"] > 1
    )
    every_case_above = all(
        case["declared_live_payload_reduction_vs_raw_bits"] < 0 for case in cases
    )
    contexts = sorted({case["work"]["maximum_declared_live_context"] for case in cases})
    current_primary = next(
        case
        for case in cases
        if (case["strands"], case["rounds"], case["family"]) == (4, 4, 0)
    )
    m223_result_path = here / "SU2_LEVEL8_CUBIC_SKEIN_DIRECT_TRACE_FORM_RESULTS.json"
    m223_result = json.loads(m223_result_path.read_text())
    m223_primary = next(
        case
        for case in m223_result["cases"]
        if (case["strands"], case["rounds"], case["family"]) == (4, 4, 0)
    )
    predecessor_primary_bits = m223_primary["work"][
        "maximum_declared_live_payload_bits"
    ]
    current_primary_bits = current_primary["work"][
        "maximum_declared_live_payload_bits"
    ]
    if all_nontrivial_smaller:
        result_name = "PASS_BOUNDED_EXACT_REMATERIALIZED_TRACE_LIFECYCLE_REDUCTION"
        claim = "BOUNDED_EXACT_LEDGER_AND_PUBLIC_UNIT_DESCRIPTOR_REMATERIALIZATION_RELEASES_ALL_DIRECT_TRACE_CALLER_FIELD_SCALARS_AND_ELIMINATES_ACCEPTED_PATH_LIFECYCLE_SCALE_CONJUGATES_WHILE_REDUCING_DECLARED_EXACT_LIVE_PAYLOAD_BELOW_THE_MATCHED_RAW_RECURRENCE_ON_EVERY_DECLARED_DEPTH_ABOVE_ONE_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_REUSE_BUT_THE_IDENTICAL_CLASSICAL_REMATERIALIZED_COORDINATE_RECURRENCE_REMAINS"
    elif every_case_above:
        result_name = "PASS_BOUNDED_EXACT_REMATERIALIZED_TRACE_PERSISTING_HEIGHT_NO_GO"
        claim = (
            "BOUNDED_EXACT_LEDGER_AND_PUBLIC_UNIT_DESCRIPTOR_REMATERIALIZATION_"
            "RELEASES_ALL_DIRECT_TRACE_CALLER_FIELD_SCALARS_ELIMINATES_ACCEPTED_"
            "PATH_LIFECYCLE_SCALE_CONJUGATES_AND_REDUCES_PRIMARY_DECLARED_EXACT_"
            f"LIVE_PAYLOAD_FROM{predecessor_primary_bits}_TO"
            f"{current_primary_bits}_BITS_WITH_FINAL_ONLY_BOUNDARY_EXACT_"
            "RESTORATION_REUSE_BUT_REMATERIALIZED_UNIT_POWER_ACCUMULATION_"
            "DOMINATES_AND_ALL_SEVEN_DECLARED_CASES_REMAIN_ABOVE_THE_MATCHED_"
            "RAW_RECURRENCE_WHILE_THE_IDENTICAL_CLASSICAL_REMATERIALIZED_"
            "COORDINATE_RECURRENCE_REMAINS"
        )
    else:
        result_name = "PASS_BOUNDED_EXACT_REMATERIALIZED_TRACE_MIXED_LIFECYCLE_RESULT"
        claim = "BOUNDED_EXACT_LEDGER_AND_PUBLIC_UNIT_DESCRIPTOR_REMATERIALIZATION_RELEASES_ALL_DIRECT_TRACE_CALLER_FIELD_SCALARS_AND_ELIMINATES_ACCEPTED_PATH_LIFECYCLE_SCALE_CONJUGATES_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_REUSE_AND_A_MIXED_DECLARED_LIVE_PAYLOAD_RESULT_AGAINST_THE_MATCHED_RAW_RECURRENCE_WHILE_THE_IDENTICAL_CLASSICAL_REMATERIALIZED_COORDINATE_RECURRENCE_REMAINS"
    result = {
        "schema": "cat_cas.su2_level8_cubic_skein_rematerialized_trace.v1",
        "result": result_name,
        "claim": claim,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "FORMAL_PRETRUNCATION_QZETA40_M219_CUBIC_SKEIN_M220_UNIT_M221_LEDGER_NATIVE_M222_STREAMED_M223_DIRECT_PARAMETERS3_7_9_11_13_17_19_LEDGER_NORM_AND_SCALE_REMATERIALIZED_POWER_BASIS_RAMANUJAN_TRACE_LINE_SEARCH_FAMILY0_STRANDS4_DEPTH1TO4_STRANDS6_DEPTH1TO2_STRANDS8_DEPTH1_PRIMARY4_DEPTH4_REUSE4_DEPTH2_FAMILY1_DIRECT_PROCESS_ONLY",
        "mechanism": {
            "trace_formula": "SUM_ABC_WA_VB_VC_RAMANUJAN_TRACE_A_PLUS_B_MINUS_C",
            "direct_trace_residual_cell_conjugates_materialized": False,
            "direct_trace_norm_cell_field_products_materialized": False,
            "direct_trace_weighted_cell_field_products_materialized": False,
            "lifecycle_scale_conjugates_materialized": False,
            "lifecycle_scale_conjugates_per_balance": 0,
            "maximum_direct_trace_caller_scalar_cells": 0,
            "line_weights_rematerialized_from_resident_ledger": True,
            "selected_multipliers_rematerialized_from_resident_ledger": True,
            "aggregate_norm_fields_materialized": False,
            "raw_or_candidate_vectors_materialized": False,
            "retained_inverse_value_history": 0,
        },
        "cases": cases,
        "selected_cases": selected,
        "predecessor_comparison": {
            "m223_primary_declared_live_payload_bits": predecessor_primary_bits,
            "m224_primary_declared_live_payload_bits": current_primary_bits,
            "reduction_bits": predecessor_primary_bits - current_primary_bits,
        },
        "lifecycle_law": {
            "all_declared_depth_above_one_smaller_than_matched_raw": all_nontrivial_smaller,
            "every_declared_case_above_matched_raw": every_case_above,
            "dominant_rematerialized_contexts": contexts,
            "direct_trace_residual_cell_product_materialization_eliminated": all(
                case["work"]["weighted_cell_field_products_materialized"] == 0
                and case["work"]["norm_cell_field_products_materialized"] == 0
                and case["work"][
                    "direct_trace_conjugate_field_values_materialized"
                ]
                == 0
                for case in cases
            ),
            "lifecycle_scale_conjugates_eliminated": all(
                case["work"][
                    "lifecycle_scale_conjugate_field_values_materialized"
                ]
                == 0
                for case in cases
            ),
            "all_direct_trace_caller_field_scalars_released": all(
                case["work"]["maximum_direct_trace_caller_scalar_cells"] == 0
                for case in cases
            ),
            "logical_exact_live_intervals_not_process_rss": True,
        },
        "separate_reference": {
            "imports_m224_production": reference.get("imports_m224_production"),
            "imports_m223_production": reference.get("imports_m223_production"),
            "imports_m222_production": reference.get("imports_m222_production"),
            "imports_m221_production": reference.get("imports_m221_production"),
            "uses_prior_standalone_m222_reference_substrate": reference.get(
                "uses_prior_standalone_m222_reference_substrate"
            ),
            "case_state_boundary_balance_resource_restoration_parity": True,
            "reuse_parity": True,
        },
        "reuse": reuse,
        "controls": controls,
        "matched_classical_baselines": {
            "strongest_compact": "IDENTICAL_LEDGER_AND_PUBLIC_UNIT_DESCRIPTOR_REMATERIALIZED_DIRECT_POWER_BASIS_RAMANUJAN_TRACE_COORDINATE_RECURRENCE",
            "matched_raw": "IDENTICAL_RAW_LINK_PATTERN_CUBIC_SKEIN_RECURRENCE_WITH_THE_SAME_DECLARED_FIELD_TEMPORARY_LAW",
            "phase_specific_reduction": False,
            "computational_advantage": False,
        },
        "resource_law": {
            "all_rational_products_sums_accumulators_and_loop_integers_counted": True,
            "all_ledger_norm_scale_and_multiplier_rematerialization_counted": True,
            "public_ramanujan_trace_formula_not_answer_bearing": True,
            "whole_process_and_python_object_overhead_bounded": False,
            "excluded_not_zero": "PYTHON_FRACTION_OBJECT_CONTAINER_CAPACITY_ALLOCATOR_INTERPRETER_PROCESS_IMAGE_JSON_SERIALIZATION_TIMING_AND_WHOLE_PROCESS_RSS",
        },
        "claim_limits": {
            "asymptotic_height_bound": False,
            "catvm_custody": False,
            "distinct_phase_resource_established": False,
            "computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference_established": False,
            "unbounded_computation_established": False,
        },
        "source_dependencies": {
            "m222_production_sha256": sha256_file(
                here / "su2_level8_cubic_skein_streamed_embedding_energy.py"
            ),
            "m223_production_sha256": sha256_file(
                here / "su2_level8_cubic_skein_direct_trace_form.py"
            ),
            "m223_result_sha256": sha256_file(m223_result_path),
            "m224_production_sha256": sha256_file(Path(__file__).resolve()),
            "m224_separate_reference_code_sha256": sha256_file(
                here / "su2_level8_cubic_skein_rematerialized_trace_separate_reference.py"
            ),
            "m224_separate_reference_result_sha256": sha256_file(reference_path),
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
