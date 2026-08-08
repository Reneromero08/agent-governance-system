#!/usr/bin/env python3
"""Standalone M221 ledger-native gauge reference.

This implementation imports the standalone M220 reference substrate, never
M221 production. It independently reconstructs residual/ledger updates,
streamed gauge scoring, exact live-state accounting, the matched raw
recurrence, forward/inverse execution, and restored reuse.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, fields

import su2_level8_cubic_skein_unit_gauge_separate_reference as m220


sys.set_int_max_str_digits(0)
base = m220.prior
UNITS = m220.UNITS
RANK = m220.RANK
CASES = m220.CASES


def scalar_payload(value: base.E) -> int:
    return base.payload_bits([value])


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


@dataclass
class Work:
    balance_calls: int = 0
    exact_trace_energy_evaluations: int = 0
    line_searches: int = 0
    line_bracket_doublings: int = 0
    line_ternary_steps: int = 0
    unit_power_field_multiplications: int = 0
    unit_norm_field_multiplications: int = 0
    ledger_scale_field_multiplications: int = 0
    ledger_square_field_multiplications: int = 0
    relative_cubic_scale_field_multiplications: int = 0
    norm_field_multiplications: int = 0
    norm_field_additions: int = 0
    streamed_raw_cells_scanned: int = 0
    streamed_raw_cell_multiplications: int = 0
    streamed_candidate_cells_scanned: int = 0
    streamed_candidate_cell_multiplications: int = 0
    selected_net_mutation_cells: int = 0
    selected_net_mutation_multiplications: int = 0
    streamed_commitment_cells: int = 0
    streamed_commitment_multiplications: int = 0
    streamed_projection_scalar_multiplications: int = 0
    raw_actual_vectors_materialized: int = 0
    candidate_residual_vectors_materialized: int = 0
    retained_inverse_value_history: int = 0
    carrier_observations: int = 0
    transient_observations: int = 0
    maximum_resident_payload_bits: int = 0
    maximum_scratch_payload_bits: int = 0
    maximum_scalar_payload_bits: int = 0
    maximum_line_integer_payload_bits: int = 0
    maximum_declared_live_payload_bits: int = 0
    maximum_declared_live_field_cells: int = 0
    maximum_declared_live_context: str = ""

    def trace_energy(self, norm: base.E) -> int:
        self.exact_trace_energy_evaluations += 1
        result = m220.trace(norm)
        if result < 0:
            raise RuntimeError("standalone exact embedding energy became negative")
        return result

    def observe(
        self,
        residual: list[base.E],
        ledger: list[int],
        scratch: list[base.E],
        *,
        scalars: tuple[base.E, ...] = (),
        integers: tuple[int, ...] = (),
        context: str = "",
    ) -> None:
        residual_payload = base.payload_bits(residual)
        scratch_payload = base.payload_bits(scratch)
        scalar_bits = sum(scalar_payload(value) for value in scalars)
        integer_bits = sum(signed_bits(value) for value in integers)
        payload = (
            residual_payload
            + m220.ledger_bits(ledger)
            + scratch_payload
            + scalar_bits
            + integer_bits
        )
        self.transient_observations += bool(scalars or integers)
        self.carrier_observations += not bool(scalars or integers)
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits,
            residual_payload + m220.ledger_bits(ledger) + scratch_payload,
        )
        self.maximum_scratch_payload_bits = max(
            self.maximum_scratch_payload_bits, scratch_payload
        )
        self.maximum_scalar_payload_bits = max(
            self.maximum_scalar_payload_bits,
            max((scalar_payload(value) for value in scalars), default=0),
        )
        self.maximum_line_integer_payload_bits = max(
            self.maximum_line_integer_payload_bits, integer_bits
        )
        if payload > self.maximum_declared_live_payload_bits:
            self.maximum_declared_live_payload_bits = payload
            self.maximum_declared_live_context = context
        self.maximum_declared_live_field_cells = max(
            self.maximum_declared_live_field_cells,
            len(residual) + len(scratch) + len(scalars),
        )

    def as_dict(self) -> dict[str, int]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


def counted_power(
    value: base.E,
    exponent: int,
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
) -> base.E:
    result = base.ONE
    factor = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            product = result * factor
            work.unit_power_field_multiplications += 1
            work.observe(
                residual,
                ledger,
                scratch,
                scalars=(result, factor, product),
                integers=(remaining,),
                context="UNIT_POWER_ACCUMULATE",
            )
            result = product
        remaining >>= 1
        if remaining:
            squared = factor * factor
            work.unit_power_field_multiplications += 1
            work.observe(
                residual,
                ledger,
                scratch,
                scalars=(result, factor, squared),
                integers=(remaining,),
                context="UNIT_POWER_SQUARE",
            )
            factor = squared
    return result


def represented_factor(
    unit: m220.Unit,
    exponent: int,
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
) -> base.E:
    return (
        counted_power(unit.value, exponent, residual, ledger, scratch, work)
        if exponent >= 0
        else counted_power(
            unit.reciprocal, -exponent, residual, ledger, scratch, work
        )
    )


def residual_factor(
    unit: m220.Unit,
    exponent: int,
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
) -> base.E:
    return (
        counted_power(unit.reciprocal, exponent, residual, ledger, scratch, work)
        if exponent >= 0
        else counted_power(unit.value, -exponent, residual, ledger, scratch, work)
    )


def norm_factor(
    unit: m220.Unit,
    exponent: int,
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
) -> base.E:
    return (
        counted_power(
            unit.reciprocal_norm, exponent, residual, ledger, scratch, work
        )
        if exponent >= 0
        else counted_power(unit.norm, -exponent, residual, ledger, scratch, work)
    )


def ledger_scale(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
) -> base.E:
    result = base.ONE
    for exponent, unit in zip(ledger, UNITS):
        factor = represented_factor(
            unit, exponent, residual, ledger, scratch, work
        )
        updated = result * factor
        work.ledger_scale_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(result, factor, updated),
            integers=(exponent,),
            context="LEDGER_SCALE_ACCUMULATE",
        )
        result = updated
    return result


def vector_norm(
    values: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
) -> base.E:
    result = base.ZERO
    for value in values:
        conjugated = m220.conjugate(value)
        product = value * conjugated
        work.norm_field_multiplications += 1
        work.observe(
            values,
            ledger,
            scratch,
            scalars=(result, conjugated, product),
            context="RESIDUAL_NORM_PRODUCT",
        )
        updated = result + product
        work.norm_field_additions += 1
        work.observe(
            values,
            ledger,
            scratch,
            scalars=(result, product, updated),
            context="RESIDUAL_NORM_ACCUMULATE",
        )
        result = updated
    return result


def line_minimum(
    norm: base.E,
    unit: m220.Unit,
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
) -> tuple[int, int]:
    work.line_searches += 1

    def energy(exponent: int, integers: tuple[int, ...]) -> int:
        factor = norm_factor(
            unit, exponent, residual, ledger, scratch, work
        )
        candidate_norm = factor * norm
        work.unit_norm_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(norm, factor, candidate_norm),
            integers=integers + (exponent,),
            context="EXACT_TRACE_LINE_NORM_FACTOR_AND_CANDIDATE_NORM",
        )
        return work.trace_energy(candidate_norm)

    zero = energy(0, (0,))
    positive = energy(1, (0, 1, -1, zero))
    negative = energy(-1, (0, 1, -1, zero, positive))
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=(norm,),
        integers=(zero, positive, negative),
        context="EXACT_TRACE_INITIAL_DIRECTION_ENERGIES",
    )
    if min(positive, negative) >= zero:
        return 0, zero
    sign = 1 if positive < negative else -1
    previous = 0
    current = sign
    current_energy = positive if sign > 0 else negative
    del zero, positive, negative
    for _ in range(32):
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
        raise RuntimeError("standalone unit line minimum not bracketed")
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
        candidate_energy = energy(
            exponent, (low, high, selected, selected_energy)
        )
        if (candidate_energy, exponent) < (selected_energy, selected):
            selected, selected_energy = exponent, candidate_energy
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=(norm,),
        integers=(low, high, selected, selected_energy),
        context="EXACT_TRACE_LINE_SELECTED_ENERGY",
    )
    return selected, selected_energy


def stream_payload(
    values: list[base.E],
    scale: base.E,
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
    *,
    candidate: bool,
    live_scalars: tuple[base.E, ...],
    live_integers: tuple[int, ...] = (),
) -> int:
    payload = 0
    for value in values:
        if candidate:
            work.streamed_candidate_cells_scanned += 1
        else:
            work.streamed_raw_cells_scanned += 1
        product = value if scale == base.ONE else scale * value
        if scale != base.ONE:
            if candidate:
                work.streamed_candidate_cell_multiplications += 1
            else:
                work.streamed_raw_cell_multiplications += 1
        payload += scalar_payload(product)
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=live_scalars + (scale, product),
            integers=live_integers + (payload,),
            context=(
                "STREAMED_CANDIDATE_CELL_PRODUCT"
                if candidate
                else "STREAMED_RAW_CELL_PRODUCT"
            ),
        )
    return payload


def mutate_selected(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    multiplier: base.E,
    work: Work,
    live_scalars: tuple[base.E, ...],
) -> None:
    for index, value in enumerate(residual):
        work.selected_net_mutation_cells += 1
        product = value if multiplier == base.ONE else multiplier * value
        if multiplier != base.ONE:
            work.selected_net_mutation_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=live_scalars + (multiplier, product),
            integers=(index,),
            context="SELECTED_NET_UNIT_MUTATION",
        )
        residual[index] = product


def balance(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
    scale: base.E,
) -> dict[str, object]:
    work.balance_calls += 1
    residual_norm = vector_norm(residual, ledger, scratch, work)
    scale_conjugate = m220.conjugate(scale)
    scale_norm = scale * scale_conjugate
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=(scale, scale_conjugate, scale_norm),
        context="LEDGER_SCALE_NORM",
    )
    actual_norm = scale_norm * residual_norm
    work.norm_field_multiplications += 2
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=(scale, residual_norm, scale_norm, actual_norm),
        context="AGGREGATE_ACTUAL_NORM_CONSTRUCTION",
    )
    zero_ledger = [0] * RANK
    raw_payload = stream_payload(
        residual,
        scale,
        residual,
        ledger,
        scratch,
        work,
        candidate=False,
        live_scalars=(actual_norm,),
    )
    identity_total = raw_payload + m220.ledger_bits(zero_ledger)
    identity_energy = work.trace_energy(actual_norm)
    best_key = (
        identity_total,
        raw_payload,
        identity_energy,
        tuple(zero_ledger),
    )
    best_multiplier = scale
    best_ledger = zero_ledger
    exponents = []
    for index, unit in enumerate(UNITS):
        exponent, energy = line_minimum(
            actual_norm, unit, residual, ledger, scratch, work
        )
        exponents.append(exponent)
        if exponent == 0:
            continue
        factor = residual_factor(
            unit, exponent, residual, ledger, scratch, work
        )
        multiplier = factor * scale
        work.ledger_scale_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(actual_norm, best_multiplier, scale, factor, multiplier),
            integers=(index, exponent, energy),
            context="CANDIDATE_NET_UNIT_MULTIPLIER",
        )
        candidate_ledger = [0] * RANK
        candidate_ledger[index] = exponent
        candidate_payload = stream_payload(
            residual,
            multiplier,
            residual,
            ledger,
            scratch,
            work,
            candidate=True,
            live_scalars=(actual_norm, best_multiplier),
            live_integers=(index, exponent, energy),
        )
        key = (
            candidate_payload + m220.ledger_bits(candidate_ledger),
            candidate_payload,
            energy,
            tuple(candidate_ledger),
        )
        if key < best_key:
            best_key = key
            best_multiplier = multiplier
            best_ledger = candidate_ledger
    mutate_selected(
        residual,
        ledger,
        scratch,
        best_multiplier,
        work,
        (),
    )
    ledger[:] = best_ledger
    return {
        "raw_payload_bits": raw_payload,
        "balanced_residual_payload_bits": base.payload_bits(residual),
        "unit_ledger_payload_bits": m220.ledger_bits(ledger),
        "balanced_residual_plus_ledger_payload_bits": best_key[0],
        "resident_payload_reduction_bits_before_constant_scratch": (
            identity_total - best_key[0]
        ),
        "selected_unit_ledger": list(ledger),
        "per_direction_trace_energy_minimizing_exponents": exponents,
        "selected_exact_embedding_energy_bits": signed_bits(best_key[2]),
        "selected_exact_embedding_energy_sha256": hashlib.sha256(
            str(best_key[2]).encode("ascii")
        ).hexdigest(),
        "identity_selected": not any(ledger),
    }


def apply_shear(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    topology: base.Topology,
    operation: base.Operation,
    work: Work,
    scale: base.E,
    *,
    inverse: bool,
) -> None:
    scale_squared = scale * scale
    work.ledger_square_field_multiplications += 1
    phase = base.E.root(base.shear_power(operation))
    if inverse:
        phase = base.ZERO - phase
    relative_phase = phase * scale_squared
    work.relative_cubic_scale_field_multiplications += 1
    targets = topology.targets[operation.generator]
    flags = topology.cup_flags[operation.generator]
    sources = (
        range(topology.dimension - 1, -1, -1)
        if inverse
        else range(topology.dimension)
    )
    for source in sources:
        if flags[source]:
            continue
        target = targets[source]
        value = residual[source]
        square = value * value
        cube = square * value
        injected = relative_phase * cube
        updated = residual[target] + injected
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(relative_phase, square),
            integers=(source, target),
            context="LEDGER_NATIVE_CUBIC_SQUARE",
        )
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(relative_phase, cube),
            integers=(source, target),
            context="LEDGER_NATIVE_CUBIC_CUBE",
        )
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(relative_phase, injected),
            integers=(source, target),
            context="LEDGER_NATIVE_CUBIC_INJECTION",
        )
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(relative_phase, updated),
            integers=(source, target),
            context="LEDGER_NATIVE_CUBIC_UPDATE",
        )
        residual[target] = updated


def apply_operation(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    topology: base.Topology,
    operation: base.Operation,
    work: Work,
    *,
    inverse: bool,
) -> dict[str, object]:
    scale = ledger_scale(residual, ledger, scratch, work)
    if inverse:
        apply_shear(
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
        apply_shear(
            residual,
            ledger,
            scratch,
            topology,
            operation,
            work,
            scale,
            inverse=False,
        )
    metrics = balance(residual, ledger, scratch, work, scale)
    scratch[:] = [base.ZERO] * topology.dimension
    work.observe(residual, ledger, scratch)
    return metrics


def commitment_and_payload(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
) -> tuple[str, int]:
    scale = ledger_scale(residual, ledger, scratch, work)
    digest = hashlib.sha256()
    payload = 0
    for index, value in enumerate(residual):
        represented = value if scale == base.ONE else scale * value
        if scale != base.ONE:
            work.streamed_commitment_multiplications += 1
        work.streamed_commitment_cells += 1
        if index:
            digest.update(b"|")
        digest.update(represented.token().encode("ascii"))
        payload += scalar_payload(represented)
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(scale, represented),
            integers=(index, payload),
            context="STREAMED_ACTUAL_COMMITMENT_CELL",
        )
    return digest.hexdigest(), payload


def project(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    topology: base.Topology,
    work: Work,
) -> base.E:
    residual_boundary = base.markov_boundary(residual, topology)
    scale = ledger_scale(residual, ledger, scratch, work)
    represented = (
        residual_boundary
        if scale == base.ONE
        else scale * residual_boundary
    )
    if scale != base.ONE:
        work.streamed_projection_scalar_multiplications += 1
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=(scale, residual_boundary, represented),
        context="FINAL_LINEAR_BOUNDARY_SCALE",
    )
    return represented


@dataclass
class RawPeak:
    maximum_declared_live_payload_bits: int = 0
    maximum_declared_live_field_cells: int = 0
    observations: int = 0
    maximum_declared_live_context: str = ""

    def observe(
        self,
        state: list[base.E],
        scratch: list[base.E],
        scalars: tuple[base.E, ...] = (),
        context: str = "",
    ) -> None:
        self.observations += 1
        payload = (
            base.payload_bits(state)
            + base.payload_bits(scratch)
            + sum(scalar_payload(value) for value in scalars)
        )
        if payload > self.maximum_declared_live_payload_bits:
            self.maximum_declared_live_payload_bits = payload
            self.maximum_declared_live_context = context
        self.maximum_declared_live_field_cells = max(
            self.maximum_declared_live_field_cells,
            len(state) + len(scratch) + len(scalars),
        )


def raw_shear(
    state: list[base.E],
    scratch: list[base.E],
    topology: base.Topology,
    operation: base.Operation,
    peak: RawPeak,
    *,
    inverse: bool,
) -> None:
    phase = base.E.root(base.shear_power(operation))
    if inverse:
        phase = base.ZERO - phase
    targets = topology.targets[operation.generator]
    flags = topology.cup_flags[operation.generator]
    sources = (
        range(topology.dimension - 1, -1, -1)
        if inverse
        else range(topology.dimension)
    )
    for source in sources:
        if flags[source]:
            continue
        target = targets[source]
        value = state[source]
        square = value * value
        cube = square * value
        injected = phase * cube
        updated = state[target] + injected
        peak.observe(state, scratch, (phase, square), "RAW_CUBIC_SQUARE")
        peak.observe(state, scratch, (phase, cube), "RAW_CUBIC_CUBE")
        peak.observe(state, scratch, (phase, injected), "RAW_CUBIC_INJECTION")
        peak.observe(state, scratch, (phase, updated), "RAW_CUBIC_UPDATE")
        state[target] = updated


def raw_baseline(
    topology: base.Topology, operations: tuple[base.Operation, ...]
) -> dict[str, int]:
    state = m220.source(topology)
    scratch = [base.ZERO] * topology.dimension
    peak = RawPeak()
    peak.observe(state, scratch)
    for operation in operations:
        base.apply_gate(state, scratch, topology, operation)
        peak.observe(state, scratch, context="RAW_LINEAR_GATE")
        raw_shear(state, scratch, topology, operation, peak, inverse=False)
    boundary = base.markov_boundary(state, topology)
    peak.observe(state, scratch, (boundary,), "RAW_FINAL_BOUNDARY")
    for operation in reversed(operations):
        raw_shear(state, scratch, topology, operation, peak, inverse=True)
        base.apply_gate(
            state,
            scratch,
            topology,
            base.Operation(operation.generator, -operation.exponent),
        )
        peak.observe(state, scratch, context="RAW_INVERSE_LINEAR_GATE")
    if state != m220.source(topology):
        raise RuntimeError("standalone matched raw recurrence failed restoration")
    return {
        "maximum_declared_live_payload_bits": peak.maximum_declared_live_payload_bits,
        "maximum_declared_live_field_cells": peak.maximum_declared_live_field_cells,
        "observations": peak.observations,
        "maximum_declared_live_context": peak.maximum_declared_live_context,
    }


def execute(
    strands: int,
    rounds: int,
    family: int,
    *,
    generation: int = 1,
    residual: list[base.E] | None = None,
    ledger: list[int] | None = None,
    scratch: list[base.E] | None = None,
) -> tuple[dict[str, object], list[base.E], list[int], list[base.E]]:
    topology = base.Topology.compile(strands)
    expected = m220.source(topology)
    residual = expected.copy() if residual is None else residual
    ledger = [0] * RANK if ledger is None else ledger
    scratch = [base.ZERO] * topology.dimension if scratch is None else scratch
    backings = id(residual), id(ledger), id(scratch)
    operations = base.operations(strands, rounds, family)
    work = Work()
    work.observe(residual, ledger, scratch)
    final_balance: dict[str, object] = {}
    for operation in operations:
        final_balance = apply_operation(
            residual, ledger, scratch, topology, operation, work, inverse=False
        )
    forward_commitment, raw_payload = commitment_and_payload(
        residual, ledger, scratch, work
    )
    boundary = project(residual, ledger, scratch, topology, work)
    for operation in reversed(operations):
        apply_operation(
            residual, ledger, scratch, topology, operation, work, inverse=True
        )
    raw = raw_baseline(topology, operations)
    result = {
        "boundary_commitment": base.boundary_commitment(boundary),
        "forward_state_commitment": forward_commitment,
        "forward_raw_payload_bits": raw_payload,
        "final_balance": final_balance,
        "same_residual_backing": id(residual) == backings[0],
        "same_unit_ledger_backing": id(ledger) == backings[1],
        "same_scratch_backing": id(scratch) == backings[2],
        "restoration_error_field_cells": sum(
            left != right for left, right in zip(residual, expected)
        ),
        "canonical_post_restoration_state_exact": (
            residual == expected
            and not any(ledger)
            and all(value == base.ZERO for value in scratch)
        ),
        "restoration_generation": generation,
        "baseline_reload_used": False,
        "work": work.as_dict(),
        "matched_raw_recurrence": raw,
        "declared_live_payload_reduction_vs_raw_bits": (
            raw["maximum_declared_live_payload_bits"]
            - work.maximum_declared_live_payload_bits
        ),
    }
    return result, residual, ledger, scratch


def case(strands: int, rounds: int, family: int) -> dict[str, object]:
    result, _, _, _ = execute(strands, rounds, family)
    topology, direct, _ = base.forward(strands, rounds, family)
    if result["forward_state_commitment"] != base.state_commitment(direct):
        raise RuntimeError("standalone ledger-native path changed M219 state")
    return {
        "strands": strands,
        "rounds": rounds,
        "family": family,
        "link_pattern_cells": topology.dimension,
        **result,
        "direct_m219_state_commitment_agreement": True,
    }


def reuse() -> dict[str, object]:
    topology = base.Topology.compile(4)
    residual = m220.source(topology)
    ledger = [0] * RANK
    scratch = [base.ZERO] * topology.dimension
    primary, residual, ledger, scratch = execute(
        4, 4, 0, generation=1, residual=residual, ledger=ledger, scratch=scratch
    )
    reused, residual, ledger, scratch = execute(
        4, 2, 1, generation=2, residual=residual, ledger=ledger, scratch=scratch
    )
    fresh, _, _, _ = execute(4, 2, 1)
    return {
        "primary": primary,
        "reuse": reused,
        "fresh_reuse": fresh,
        "fresh_restored_reuse_boundary_agreement": (
            reused["boundary_commitment"] == fresh["boundary_commitment"]
        ),
        "fresh_restored_reuse_state_agreement": (
            reused["forward_state_commitment"] == fresh["forward_state_commitment"]
        ),
        "restoration_generation_after_reuse": 2,
    }


def main() -> None:
    print(
        json.dumps(
            {
                "schema": "cat_cas.su2_level8_cubic_skein_ledger_native_gauge_reference.v1",
                "imports_m221_production": False,
                "uses_prior_standalone_m220_reference_substrate": True,
                "cases": [case(*item) for item in CASES],
                "reuse": reuse(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
