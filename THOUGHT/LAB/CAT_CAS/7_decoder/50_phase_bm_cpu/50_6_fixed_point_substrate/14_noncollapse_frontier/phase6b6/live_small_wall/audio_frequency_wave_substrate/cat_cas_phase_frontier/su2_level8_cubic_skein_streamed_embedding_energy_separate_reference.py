#!/usr/bin/env python3
"""Standalone M222 streamed-embedding-energy reference.

This file imports only the standalone M221 reference substrate.  It does not
import M221 or M222 production.  The exact trace-linear energy stream and its
resource counters are reconstructed independently.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass

import su2_level8_cubic_skein_ledger_native_gauge_separate_reference as prior


sys.set_int_max_str_digits(0)
base = prior.base
UNITS = prior.UNITS
RANK = prior.RANK
CASES = prior.CASES
MAX_BRACKET_DOUBLINGS = 32


@dataclass
class Work(prior.Work):
    streamed_embedding_energy_evaluations: int = 0
    streamed_embedding_cells_scanned: int = 0
    streamed_embedding_norm_field_multiplications: int = 0
    streamed_embedding_weight_field_multiplications: int = 0
    streamed_embedding_trace_terms: int = 0
    aggregate_residual_norm_fields_materialized: int = 0
    aggregate_actual_norm_fields_materialized: int = 0
    aggregate_candidate_norm_fields_materialized: int = 0


def streamed_energy(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    weight: base.E,
    work: Work,
    *,
    live_integers: tuple[int, ...] = (),
    context_prefix: str,
) -> int:
    work.exact_trace_energy_evaluations += 1
    work.streamed_embedding_energy_evaluations += 1
    total = 0
    for index, value in enumerate(residual):
        work.streamed_embedding_cells_scanned += 1
        conjugated = prior.m220.conjugate(value)
        norm_cell = value * conjugated
        work.norm_field_multiplications += 1
        work.streamed_embedding_norm_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(weight, conjugated, norm_cell),
            integers=live_integers + (index, total),
            context=f"{context_prefix}_CELL_NORM",
        )
        weighted = norm_cell if weight == base.ONE else weight * norm_cell
        if weight != base.ONE:
            work.unit_norm_field_multiplications += 1
            work.streamed_embedding_weight_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(weight, norm_cell, weighted),
            integers=live_integers + (index, total),
            context=f"{context_prefix}_WEIGHTED_CELL",
        )
        term = prior.m220.trace(weighted)
        updated = total + term
        work.streamed_embedding_trace_terms += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(weight, weighted),
            integers=live_integers + (index, total, term, updated),
            context=f"{context_prefix}_INTEGER_ACCUMULATE",
        )
        total = updated
    if total < 0:
        raise RuntimeError("standalone streamed embedding energy became negative")
    return total


def line_minimum(
    scale_norm: base.E,
    unit: prior.m220.Unit,
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
) -> tuple[int, int]:
    work.line_searches += 1

    def energy(exponent: int, integers: tuple[int, ...]) -> int:
        factor = prior.norm_factor(
            unit, exponent, residual, ledger, scratch, work
        )
        combined = factor * scale_norm
        work.unit_norm_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(scale_norm, factor, combined),
            integers=integers + (exponent,),
            context="STREAMED_TRACE_LINE_COMBINED_WEIGHT",
        )
        return streamed_energy(
            residual,
            ledger,
            scratch,
            combined,
            work,
            live_integers=integers + (exponent,),
            context_prefix="STREAMED_TRACE_LINE",
        )

    zero = energy(0, (0,))
    positive = energy(1, (0, 1, -1, zero))
    negative = energy(-1, (0, 1, -1, zero, positive))
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=(scale_norm,),
        integers=(zero, positive, negative),
        context="STREAMED_TRACE_INITIAL_DIRECTION_ENERGIES",
    )
    if min(positive, negative) >= zero:
        return 0, zero
    sign = 1 if positive < negative else -1
    previous = 0
    current = sign
    current_energy = positive if sign > 0 else negative
    del zero, positive, negative
    for _ in range(MAX_BRACKET_DOUBLINGS):
        following = 2 * current
        work.line_bracket_doublings += 1
        following_energy = energy(
            following, (previous, current, following, current_energy)
        )
        if following_energy >= current_energy:
            low, high = sorted((previous, following))
            break
        previous, current, current_energy = current, following, following_energy
    else:
        raise RuntimeError("standalone unit line minimum was not bracketed")
    while high - low > 8:
        work.line_ternary_steps += 1
        first = low + (high - low) // 3
        second = high - (high - low) // 3
        first_energy = energy(first, (low, high, first, second))
        second_energy = energy(second, (low, high, first, second, first_energy))
        if first_energy <= second_energy:
            high = second - 1
        else:
            low = first + 1
    selected = low
    selected_energy = energy(low, (low, high))
    for exponent in range(low + 1, high + 1):
        candidate = energy(exponent, (low, high, selected, selected_energy))
        if (candidate, exponent) < (selected_energy, selected):
            selected, selected_energy = exponent, candidate
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=(scale_norm,),
        integers=(low, high, selected, selected_energy),
        context="STREAMED_TRACE_LINE_SELECTED_ENERGY",
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
    scale_norm = scale * conjugated_scale
    work.norm_field_multiplications += 1
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=(scale, conjugated_scale, scale_norm),
        context="STREAMED_ENERGY_LEDGER_SCALE_NORM",
    )
    zero_ledger = [0] * RANK
    raw_payload = prior.stream_payload(
        residual,
        scale,
        residual,
        ledger,
        scratch,
        work,
        candidate=False,
        live_scalars=(scale_norm,),
    )
    identity_total = raw_payload + prior.m220.ledger_bits(zero_ledger)
    identity_energy = streamed_energy(
        residual,
        ledger,
        scratch,
        scale_norm,
        work,
        context_prefix="STREAMED_IDENTITY_ENERGY",
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
    for index, unit in enumerate(UNITS):
        exponent, energy = line_minimum(
            scale_norm, unit, residual, ledger, scratch, work
        )
        exponents.append(exponent)
        if exponent == 0:
            continue
        factor = prior.residual_factor(
            unit, exponent, residual, ledger, scratch, work
        )
        multiplier = factor * scale
        work.ledger_scale_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(scale_norm, best_multiplier, scale, factor, multiplier),
            integers=(index, exponent, energy),
            context="STREAMED_ENERGY_CANDIDATE_NET_UNIT_MULTIPLIER",
        )
        candidate_ledger = [0] * RANK
        candidate_ledger[index] = exponent
        candidate_payload = prior.stream_payload(
            residual,
            multiplier,
            residual,
            ledger,
            scratch,
            work,
            candidate=True,
            live_scalars=(scale_norm, best_multiplier),
            live_integers=(index, exponent, energy),
        )
        key = (
            candidate_payload + prior.m220.ledger_bits(candidate_ledger),
            candidate_payload,
            energy,
            tuple(candidate_ledger),
        )
        if key < best_key:
            best_key = key
            best_multiplier = multiplier
            best_ledger = candidate_ledger
    prior.mutate_selected(
        residual, ledger, scratch, best_multiplier, work, ()
    )
    ledger[:] = best_ledger
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


def main() -> None:
    prior.Work = Work
    prior.balance = balance
    print(
        json.dumps(
            {
                "schema": "cat_cas.su2_level8_cubic_skein_streamed_embedding_energy_reference.v1",
                "imports_m222_production": False,
                "imports_m221_production": False,
                "uses_prior_standalone_m221_reference_substrate": True,
                "cases": [prior.case(*item) for item in CASES],
                "reuse": prior.reuse(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
