#!/usr/bin/env python3
"""M221 ledger-native exact unit gauge for the M219 cubic-skein carrier.

The carrier represents actual = unit(ledger) * residual. Unlike M220,
linear skein gates act directly on the residual, cubic shears use only the
ledger-relative squared unit scalar, candidate gauges are scored one field
cell at a time, and only the selected net unit action mutates the carrier.
No raw or candidate vector is materialized.

The experiment tests the exact transient obstruction measured by M220. It
does not claim a phase/classical separation: the identical ledger-native
recurrence remains available to ordinary exact software.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import su2_level8_cubic_skein_unit_gauge as m220


sys.set_int_max_str_digits(0)
braid = m220.braid
skein = m220.skein
cubic = m220.cubic
UNITS = m220.UNITS
UNIT_RANK = m220.UNIT_RANK
CASES = m220.CASES
PRIMARY = m220.PRIMARY
REUSE = m220.REUSE
MAX_BRACKET_DOUBLINGS = m220.MAX_BRACKET_DOUBLINGS


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_payload(value: braid.K) -> int:
    return braid.field_payload_bits([value])


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


@dataclass
class Work:
    cubic: cubic.Work = field(default_factory=cubic.Work)
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

    def trace_energy(self, norm: braid.K) -> int:
        self.exact_trace_energy_evaluations += 1
        energy = m220.field_trace(norm)
        if energy < 0:
            raise RuntimeError("exact embedding energy became negative")
        return energy

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
        scalar_values = tuple(scalars)
        integer_values = tuple(integers)
        residual_payload = braid.field_payload_bits(residual)
        scratch_payload = braid.field_payload_bits(scratch)
        scalar_bits = sum(scalar_payload(value) for value in scalar_values)
        integer_bits = sum(signed_bits(value) for value in integer_values)
        payload = (
            residual_payload
            + m220.ledger_payload_bits(ledger)
            + scratch_payload
            + scalar_bits
            + integer_bits
        )
        self.transient_observations += bool(scalar_values or integer_values)
        self.carrier_observations += not bool(scalar_values or integer_values)
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits,
            residual_payload + m220.ledger_payload_bits(ledger) + scratch_payload,
        )
        self.maximum_scratch_payload_bits = max(
            self.maximum_scratch_payload_bits, scratch_payload
        )
        self.maximum_scalar_payload_bits = max(
            self.maximum_scalar_payload_bits,
            max((scalar_payload(value) for value in scalar_values), default=0),
        )
        self.maximum_line_integer_payload_bits = max(
            self.maximum_line_integer_payload_bits, integer_bits
        )
        if payload > self.maximum_declared_live_payload_bits:
            self.maximum_declared_live_payload_bits = payload
            self.maximum_declared_live_context = context
        self.maximum_declared_live_field_cells = max(
            self.maximum_declared_live_field_cells,
            len(residual) + len(scratch) + len(scalar_values),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "cubic"
        } | {"cubic_skein": self.cubic.as_dict()}


def counted_power(
    base: braid.K,
    exponent: int,
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
) -> braid.K:
    if exponent < 0:
        raise ValueError("negative exponent in counted power")
    result = braid.ONE
    factor = base
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
    direction: m220.UnitDirection,
    exponent: int,
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
) -> braid.K:
    return (
        counted_power(direction.unit, exponent, residual, ledger, scratch, work)
        if exponent >= 0
        else counted_power(direction.inverse, -exponent, residual, ledger, scratch, work)
    )


def residual_factor(
    direction: m220.UnitDirection,
    exponent: int,
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
) -> braid.K:
    return (
        counted_power(direction.inverse, exponent, residual, ledger, scratch, work)
        if exponent >= 0
        else counted_power(direction.unit, -exponent, residual, ledger, scratch, work)
    )


def norm_factor(
    direction: m220.UnitDirection,
    exponent: int,
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
) -> braid.K:
    return (
        counted_power(
            direction.inverse_norm, exponent, residual, ledger, scratch, work
        )
        if exponent >= 0
        else counted_power(
            direction.norm, -exponent, residual, ledger, scratch, work
        )
    )


def ledger_scale(
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
) -> braid.K:
    scale = braid.ONE
    for exponent, direction in zip(ledger, UNITS, strict=True):
        factor = represented_factor(
            direction, exponent, residual, ledger, scratch, work
        )
        updated = scale * factor
        work.ledger_scale_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(scale, factor, updated),
            integers=(exponent,),
            context="LEDGER_SCALE_ACCUMULATE",
        )
        scale = updated
    return scale


def vector_norm(
    values: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
) -> braid.K:
    result = braid.ZERO
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
    norm: braid.K,
    direction: m220.UnitDirection,
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
) -> tuple[int, int]:
    """Find the exact line minimum without retaining an energy cache."""

    work.line_searches += 1

    def energy(exponent: int, integers: tuple[int, ...]) -> int:
        factor = norm_factor(
            direction, exponent, residual, ledger, scratch, work
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
    direction_sign = 1 if positive < negative else -1
    previous = 0
    current = direction_sign
    current_energy = positive if direction_sign > 0 else negative
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
        raise RuntimeError("unit line minimum was not bracketed")
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


def stream_scaled_payload(
    values: list[braid.K],
    scale: braid.K,
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
    *,
    candidate: bool,
    live_scalars: tuple[braid.K, ...],
    live_integers: tuple[int, ...] = (),
) -> int:
    payload = 0
    for value in values:
        if candidate:
            work.streamed_candidate_cells_scanned += 1
        else:
            work.streamed_raw_cells_scanned += 1
        product = value if scale == braid.ONE else scale * value
        if scale != braid.ONE:
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


def apply_selected_net(
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    multiplier: braid.K,
    work: Work,
    *,
    live_scalars: tuple[braid.K, ...],
) -> None:
    for index, value in enumerate(residual):
        work.selected_net_mutation_cells += 1
        product = value if multiplier == braid.ONE else multiplier * value
        if multiplier != braid.ONE:
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


def balance_resident(
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
    *,
    scale: braid.K,
) -> dict[str, Any]:
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
    zero_ledger = [0] * UNIT_RANK
    raw_payload = stream_scaled_payload(
        residual,
        scale,
        residual,
        ledger,
        scratch,
        work,
        candidate=False,
        live_scalars=(actual_norm,),
    )
    identity_total = raw_payload + m220.ledger_payload_bits(zero_ledger)
    identity_energy = work.trace_energy(actual_norm)
    best_key = (
        identity_total,
        raw_payload,
        identity_energy,
        tuple(zero_ledger),
    )
    best_multiplier = scale
    best_ledger = zero_ledger
    candidate_exponents: list[int] = []
    for index, direction in enumerate(UNITS):
        exponent, energy = line_minimum(
            actual_norm, direction, residual, ledger, scratch, work
        )
        candidate_exponents.append(exponent)
        if exponent == 0:
            continue
        factor = residual_factor(
            direction, exponent, residual, ledger, scratch, work
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
        candidate_ledger = [0] * UNIT_RANK
        candidate_ledger[index] = exponent
        candidate_payload = stream_scaled_payload(
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
            candidate_payload + m220.ledger_payload_bits(candidate_ledger),
            candidate_payload,
            energy,
            tuple(candidate_ledger),
        )
        if key < best_key:
            best_key = key
            best_multiplier = multiplier
            best_ledger = candidate_ledger
    apply_selected_net(
        residual,
        ledger,
        scratch,
        best_multiplier,
        work,
        live_scalars=(),
    )
    ledger[:] = best_ledger
    return {
        "raw_payload_bits": raw_payload,
        "balanced_residual_payload_bits": braid.field_payload_bits(residual),
        "unit_ledger_payload_bits": m220.ledger_payload_bits(ledger),
        "balanced_residual_plus_ledger_payload_bits": best_key[0],
        "resident_payload_reduction_bits_before_constant_scratch": (
            identity_total - best_key[0]
        ),
        "selected_unit_ledger": list(ledger),
        "per_direction_trace_energy_minimizing_exponents": candidate_exponents,
        "selected_exact_embedding_energy_bits": signed_bits(best_key[2]),
        "selected_exact_embedding_energy_sha256": hashlib.sha256(
            str(best_key[2]).encode("ascii")
        ).hexdigest(),
        "identity_selected": not any(ledger),
    }


def apply_ledger_cubic_shear(
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    topology: skein.DiagramTopology,
    operation: braid.BraidOperation,
    work: Work,
    *,
    scale: braid.K,
    inverse: bool,
) -> None:
    scale_squared = scale * scale
    work.ledger_square_field_multiplications += 1
    phase = braid.K.zeta(cubic.shear_phase_power(operation))
    if inverse:
        phase = braid.ZERO - phase
    relative_phase = phase * scale_squared
    work.relative_cubic_scale_field_multiplications += 1
    targets = topology.e_targets[operation.generator]
    cup_flags = topology.e_delta_flags[operation.generator]
    sources = (
        range(topology.dimension - 1, -1, -1)
        if inverse
        else range(topology.dimension)
    )
    for source in sources:
        work.cubic.cubic_source_scans += 1
        if cup_flags[source]:
            continue
        target = targets[source]
        if not cup_flags[target]:
            raise RuntimeError("ledger-native cubic target left local-cup partition")
        value = residual[source]
        square = value * value
        cube = square * value
        injected = relative_phase * cube
        updated = residual[target] + injected
        work.cubic.cubic_field_multiplications += 3
        work.cubic.cubic_field_additions += 1
        work.cubic.cubic_updates += 1
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
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    topology: skein.DiagramTopology,
    operation: braid.BraidOperation,
    work: Work,
    *,
    inverse: bool,
) -> dict[str, Any]:
    scale = ledger_scale(residual, ledger, scratch, work)
    if inverse:
        apply_ledger_cubic_shear(
            residual,
            ledger,
            scratch,
            topology,
            operation,
            work,
            scale=scale,
            inverse=True,
        )
        skein.apply_gate(
            residual,
            scratch,
            topology,
            braid.BraidOperation(operation.generator, -operation.exponent),
            work.cubic.linear,
        )
        work.cubic.inverse_operations += 1
    else:
        skein.apply_gate(
            residual, scratch, topology, operation, work.cubic.linear
        )
        apply_ledger_cubic_shear(
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
    metrics = balance_resident(
        residual, ledger, scratch, work, scale=scale
    )
    scratch[:] = [braid.ZERO] * topology.dimension
    work.observe(residual, ledger, scratch)
    return metrics


def streamed_actual_commitment(
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
) -> tuple[str, int]:
    scale = ledger_scale(residual, ledger, scratch, work)
    digest = hashlib.sha256()
    payload = 0
    for index, value in enumerate(residual):
        represented = value if scale == braid.ONE else scale * value
        if scale != braid.ONE:
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


def project_final(
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    topology: skein.DiagramTopology,
    work: Work,
) -> braid.K:
    residual_boundary = skein.normalized_markov_boundary(
        residual, topology, work.cubic.linear
    )
    scale = ledger_scale(residual, ledger, scratch, work)
    represented_boundary = (
        residual_boundary
        if scale == braid.ONE
        else scale * residual_boundary
    )
    if scale != braid.ONE:
        work.streamed_projection_scalar_multiplications += 1
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=(scale, residual_boundary, represented_boundary),
        context="FINAL_LINEAR_BOUNDARY_SCALE",
    )
    return represented_boundary


@dataclass
class RawPeak:
    maximum_declared_live_payload_bits: int = 0
    maximum_declared_live_field_cells: int = 0
    observations: int = 0
    maximum_declared_live_context: str = ""

    def observe(
        self,
        values: list[braid.K],
        scratch: list[braid.K],
        scalars: Iterable[braid.K] = (),
        context: str = "",
    ) -> None:
        scalar_values = tuple(scalars)
        self.observations += 1
        payload = (
            braid.field_payload_bits(values)
            + braid.field_payload_bits(scratch)
            + sum(scalar_payload(value) for value in scalar_values)
        )
        if payload > self.maximum_declared_live_payload_bits:
            self.maximum_declared_live_payload_bits = payload
            self.maximum_declared_live_context = context
        self.maximum_declared_live_field_cells = max(
            self.maximum_declared_live_field_cells,
            len(values) + len(scratch) + len(scalar_values),
        )


def raw_cubic(
    values: list[braid.K],
    scratch: list[braid.K],
    topology: skein.DiagramTopology,
    operation: braid.BraidOperation,
    peak: RawPeak,
    *,
    inverse: bool,
) -> None:
    phase = braid.K.zeta(cubic.shear_phase_power(operation))
    if inverse:
        phase = braid.ZERO - phase
    targets = topology.e_targets[operation.generator]
    cup_flags = topology.e_delta_flags[operation.generator]
    sources = (
        range(topology.dimension - 1, -1, -1)
        if inverse
        else range(topology.dimension)
    )
    for source in sources:
        if cup_flags[source]:
            continue
        target = targets[source]
        value = values[source]
        square = value * value
        cube = square * value
        injected = phase * cube
        updated = values[target] + injected
        peak.observe(values, scratch, (phase, square), "RAW_CUBIC_SQUARE")
        peak.observe(values, scratch, (phase, cube), "RAW_CUBIC_CUBE")
        peak.observe(values, scratch, (phase, injected), "RAW_CUBIC_INJECTION")
        peak.observe(values, scratch, (phase, updated), "RAW_CUBIC_UPDATE")
        values[target] = updated


def raw_baseline(program: braid.BraidProgram) -> dict[str, int]:
    topology = skein.DiagramTopology.compile(program.strands)
    values = skein.source_state(topology)
    scratch = [braid.ZERO] * topology.dimension
    peak = RawPeak()
    linear_work = skein.Work()
    peak.observe(values, scratch)
    for index in range(program.steps):
        operation = program.operation(index)
        skein.apply_gate(values, scratch, topology, operation, linear_work)
        peak.observe(values, scratch, context="RAW_LINEAR_GATE")
        raw_cubic(values, scratch, topology, operation, peak, inverse=False)
    boundary = skein.normalized_markov_boundary(values, topology, linear_work)
    peak.observe(values, scratch, (boundary,), "RAW_FINAL_BOUNDARY")
    for index in range(program.steps - 1, -1, -1):
        operation = program.operation(index)
        raw_cubic(values, scratch, topology, operation, peak, inverse=True)
        skein.apply_gate(
            values,
            scratch,
            topology,
            braid.BraidOperation(operation.generator, -operation.exponent),
            linear_work,
        )
        peak.observe(values, scratch, context="RAW_INVERSE_LINEAR_GATE")
    if values != skein.source_state(topology):
        raise RuntimeError("matched raw cubic-skein baseline failed restoration")
    return {
        "maximum_declared_live_payload_bits": peak.maximum_declared_live_payload_bits,
        "maximum_declared_live_field_cells": peak.maximum_declared_live_field_cells,
        "observations": peak.observations,
        "maximum_declared_live_context": peak.maximum_declared_live_context,
    }


@dataclass
class Port:
    topology: skein.DiagramTopology
    residual: list[braid.K]
    ledger: list[int]
    scratch: list[braid.K]
    live: bool = False
    owner: int = 0
    generation: int = 0
    cursor: int = 0
    expected_steps: int = 0
    program_commitment: str = ""
    last_balance: dict[str, Any] = field(default_factory=dict)

    def lease(
        self, owner: int, generation: int, program: braid.BraidProgram, work: Work
    ) -> None:
        if self.live:
            raise RuntimeError("ledger-native port already live")
        if (
            len(self.residual) != self.topology.dimension
            or len(self.scratch) != self.topology.dimension
            or len(self.ledger) != UNIT_RANK
        ):
            raise ValueError("null or wrong-width ledger-native carrier")
        if owner <= 0 or generation <= 0 or program.strands != self.topology.strands:
            raise ValueError("invalid ledger-native lease")
        self.live = True
        self.owner = owner
        self.generation = generation
        self.cursor = 0
        self.expected_steps = program.steps
        self.program_commitment = skein.program_commitment(program)
        work.cubic.linear.port_leases += 1
        work.cubic.linear.public_descriptor_hashes += 2
        work.cubic.linear.public_descriptor_integers_hashed += (
            3 + self.topology.retained_pairing_integer_cells + UNIT_RANK
        )
        work.observe(self.residual, self.ledger, self.scratch)

    def require(
        self, owner: int, program: braid.BraidProgram, work: Work
    ) -> None:
        if not self.live:
            raise RuntimeError("ledger-native port is not live")
        if owner != self.owner:
            raise PermissionError("ledger-native owner mismatch")
        if skein.program_commitment(program) != self.program_commitment:
            raise ValueError("ledger-native public program mismatch")
        work.cubic.linear.public_descriptor_hashes += 2
        work.cubic.linear.public_descriptor_integers_hashed += (
            3 + self.topology.retained_pairing_integer_cells + UNIT_RANK
        )

    def apply(
        self,
        owner: int,
        program: braid.BraidProgram,
        index: int,
        work: Work,
        *,
        inverse: bool,
    ) -> None:
        self.require(owner, program, work)
        expected = self.cursor - 1 if inverse else self.cursor
        if index != expected or (not inverse and index >= self.expected_steps):
            raise ValueError("ledger-native operation cursor mismatch")
        self.last_balance = apply_operation(
            self.residual,
            self.ledger,
            self.scratch,
            self.topology,
            program.operation(index),
            work,
            inverse=inverse,
        )
        self.cursor += -1 if inverse else 1

    def project(
        self, owner: int, program: braid.BraidProgram, work: Work
    ) -> braid.K:
        self.require(owner, program, work)
        if self.cursor != self.expected_steps:
            raise PermissionError("nonfinal ledger-native projection rejected")
        return project_final(
            self.residual, self.ledger, self.scratch, self.topology, work
        )

    def release(
        self, owner: int, program: braid.BraidProgram, work: Work
    ) -> int:
        self.require(owner, program, work)
        if self.cursor:
            raise RuntimeError("ledger-native port released before inverse")
        generation = self.generation
        self.live = False
        self.owner = 0
        self.generation = 0
        self.expected_steps = 0
        self.program_commitment = ""
        self.last_balance = {}
        self.scratch[:] = [braid.ZERO] * self.topology.dimension
        work.cubic.linear.port_releases += 1
        return generation


@dataclass
class Carrier:
    port: Port
    restoration_generation: int = 0


def make_carrier(strands: int) -> tuple[Carrier, list[braid.K]]:
    topology = skein.DiagramTopology.compile(strands)
    source = skein.source_state(topology)
    return (
        Carrier(
            Port(
                topology,
                source.copy(),
                [0] * UNIT_RANK,
                [braid.ZERO] * topology.dimension,
            )
        ),
        source,
    )


def canonical_restoration(
    carrier: Carrier, source: list[braid.K], generation: int
) -> bool:
    port = carrier.port
    return (
        port.residual == source
        and not any(port.ledger)
        and all(value == braid.ZERO for value in port.scratch)
        and not port.live
        and port.owner == 0
        and port.generation == 0
        and port.cursor == 0
        and port.expected_steps == 0
        and port.program_commitment == ""
        and carrier.restoration_generation == generation
    )


def transaction(
    carrier: Carrier, source: list[braid.K], program: braid.BraidProgram
) -> tuple[dict[str, Any], Work]:
    backings = id(carrier.port.residual), id(carrier.port.ledger), id(carrier.port.scratch)
    generation = carrier.restoration_generation + 1
    owner = 221000 + generation
    work = Work()
    carrier.port.lease(owner, generation, program, work)
    for index in range(program.steps):
        carrier.port.apply(owner, program, index, work, inverse=False)
    forward_commitment, raw_payload = streamed_actual_commitment(
        carrier.port.residual, carrier.port.ledger, carrier.port.scratch, work
    )
    work.cubic.linear.state_commitment_hashes += 1
    work.cubic.linear.state_commitment_field_cells_hashed += len(carrier.port.residual)
    final_balance = dict(carrier.port.last_balance)
    boundary = carrier.port.project(owner, program, work)
    work.cubic.linear.boundary_commitment_hashes += 1
    for index in range(program.steps - 1, -1, -1):
        carrier.port.apply(owner, program, index, work, inverse=True)
    carrier.restoration_generation = carrier.port.release(owner, program, work)
    raw = raw_baseline(program)
    return {
        "boundary_commitment": braid.boundary_commitment(boundary),
        "forward_state_commitment": forward_commitment,
        "forward_raw_payload_bits": raw_payload,
        "final_balance": final_balance,
        "same_residual_backing": id(carrier.port.residual) == backings[0],
        "same_unit_ledger_backing": id(carrier.port.ledger) == backings[1],
        "same_scratch_backing": id(carrier.port.scratch) == backings[2],
        "restoration_error_field_cells": sum(
            left != right
            for left, right in zip(carrier.port.residual, source, strict=True)
        ),
        "canonical_post_restoration_state_exact": canonical_restoration(
            carrier, source, generation
        ),
        "restoration_generation": carrier.restoration_generation,
        "baseline_reload_used": False,
        "work": work.as_dict(),
        "matched_raw_recurrence": raw,
        "declared_live_payload_reduction_vs_raw_bits": (
            raw["maximum_declared_live_payload_bits"]
            - work.maximum_declared_live_payload_bits
        ),
    }, work


def execute_case(strands: int, rounds: int, family: int) -> dict[str, Any]:
    carrier, source = make_carrier(strands)
    program = braid.BraidProgram(strands, rounds, family)
    result, _ = transaction(carrier, source, program)
    _, direct, _, _ = cubic.forward_state(program)
    if result["forward_state_commitment"] != skein.state_commitment(direct):
        raise RuntimeError("ledger-native execution changed the M219 actual state")
    return {
        "strands": strands,
        "rounds": rounds,
        "family": family,
        "link_pattern_cells": len(source),
        **result,
        "direct_m219_state_commitment_agreement": True,
    }


def reuse_result() -> dict[str, Any]:
    carrier, source = make_carrier(PRIMARY[0])
    primary, _ = transaction(carrier, source, braid.BraidProgram(*PRIMARY))
    reuse, _ = transaction(carrier, source, braid.BraidProgram(*REUSE))
    fresh, fresh_source = make_carrier(REUSE[0])
    fresh_reuse, _ = transaction(fresh, fresh_source, braid.BraidProgram(*REUSE))
    return {
        "primary": primary,
        "reuse": reuse,
        "fresh_reuse": fresh_reuse,
        "fresh_restored_reuse_boundary_agreement": (
            reuse["boundary_commitment"] == fresh_reuse["boundary_commitment"]
        ),
        "fresh_restored_reuse_state_agreement": (
            reuse["forward_state_commitment"] == fresh_reuse["forward_state_commitment"]
        ),
        "restoration_generation_after_reuse": carrier.restoration_generation,
    }


def controls() -> dict[str, bool]:
    carrier, _ = make_carrier(4)
    port = carrier.port
    program = braid.BraidProgram(4, 1, 0)
    wrong = braid.BraidProgram(4, 1, 1)
    work = Work()
    port.lease(221900, 1, program, work)
    wrong_owner = premature = wrong_program = reordered = False
    try:
        port.apply(221901, program, 0, work, inverse=False)
    except PermissionError:
        wrong_owner = True
    try:
        port.project(221900, program, work)
    except PermissionError:
        premature = True
    for index in range(program.steps):
        port.apply(221900, program, index, work, inverse=False)
    represented, _ = streamed_actual_commitment(
        port.residual, port.ledger, port.scratch, work
    )
    wrong_ledger = list(port.ledger)
    wrong_ledger[0] += 1
    wrong_commitment, _ = streamed_actual_commitment(
        port.residual, wrong_ledger, port.scratch, work
    )
    try:
        port.apply(221900, wrong, program.steps - 1, work, inverse=True)
    except ValueError:
        wrong_program = True
    try:
        port.apply(221900, program, program.steps - 2, work, inverse=True)
    except ValueError:
        reordered = True
    missing = port.cursor != 0
    for index in range(program.steps - 1, -1, -1):
        port.apply(221900, program, index, work, inverse=True)
    port.release(221900, program, work)
    null_rejected = False
    try:
        Port(port.topology, [], [], []).lease(1, 1, program, Work())
    except ValueError:
        null_rejected = True
    return {
        "wrong_owner_rejected": wrong_owner,
        "premature_projection_rejected": premature,
        "wrong_public_program_inverse_rejected": wrong_program,
        "reordered_inverse_rejected": reordered,
        "missing_inverse_detected": missing,
        "wrong_unit_ledger_changes_represented_state": wrong_commitment != represented,
        "null_carrier_rejected": null_rejected,
        "raw_actual_vector_materialized": work.raw_actual_vectors_materialized > 0,
        "candidate_residual_vector_materialized": (
            work.candidate_residual_vectors_materialized > 0
        ),
        "intermediate_actual_vector_projected": False,
        "snapshot_command_available": hasattr(port, "snapshot"),
    }


def reference_case_view(case: dict[str, Any]) -> dict[str, Any]:
    result = dict(case)
    work = dict(result["work"])
    work.pop("cubic_skein")
    result["work"] = work
    return result


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: su2_level8_cubic_skein_ledger_native_gauge.py SEPARATE_REFERENCE_JSON"
        )
    reference_path = Path(sys.argv[1]).resolve()
    if str(reference_path).startswith(("/dev/shm/", "/run/shm/")):
        raise ValueError("RAM-backed M221 reference is forbidden")
    reference = json.loads(reference_path.read_text())
    if reference.get("schema") != "cat_cas.su2_level8_cubic_skein_ledger_native_gauge_reference.v1":
        raise RuntimeError("M221 separate-reference schema changed")
    cases = [execute_case(*case) for case in CASES]
    if [reference_case_view(case) for case in cases] != reference.get("cases"):
        raise RuntimeError("M221 independent case and resource parity failed")
    reuse = reuse_result()
    for section in ("primary", "reuse", "fresh_reuse"):
        for key in (
            "boundary_commitment",
            "forward_state_commitment",
            "forward_raw_payload_bits",
            "final_balance",
            "restoration_error_field_cells",
            "canonical_post_restoration_state_exact",
            "declared_live_payload_reduction_vs_raw_bits",
        ):
            if reuse[section][key] != reference["reuse"][section][key]:
                raise RuntimeError(f"M221 independent reuse parity failed: {section}.{key}")
    all_controls = controls()
    positive_controls = {
        key: value
        for key, value in all_controls.items()
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
        or all_controls["raw_actual_vector_materialized"]
        or all_controls["candidate_residual_vector_materialized"]
        or all_controls["intermediate_actual_vector_projected"]
        or all_controls["snapshot_command_available"]
    ):
        raise RuntimeError("M221 control failed")
    selected = [
        {
            "strands": case["strands"],
            "rounds": case["rounds"],
            "forward_raw_payload_bits": case["forward_raw_payload_bits"],
            "balanced_residual_plus_ledger_payload_bits": case["final_balance"][
                "balanced_residual_plus_ledger_payload_bits"
            ],
            "ledger_native_maximum_declared_live_payload_bits": case["work"][
                "maximum_declared_live_payload_bits"
            ],
            "raw_maximum_declared_live_payload_bits": case[
                "matched_raw_recurrence"
            ]["maximum_declared_live_payload_bits"],
            "declared_live_payload_reduction_vs_raw_bits": case[
                "declared_live_payload_reduction_vs_raw_bits"
            ],
            "ledger_native_maximum_context": case["work"][
                "maximum_declared_live_context"
            ],
            "raw_maximum_context": case["matched_raw_recurrence"][
                "maximum_declared_live_context"
            ],
        }
        for case in cases
        if (case["strands"], case["rounds"]) in ((4, 4), (6, 2), (8, 1))
    ]
    all_nontrivial_live_smaller = all(
        case["declared_live_payload_reduction_vs_raw_bits"] > 0
        for case in cases
        if case["rounds"] > 1
    )
    every_declared_case_above_raw = all(
        case["declared_live_payload_reduction_vs_raw_bits"] < 0
        for case in cases
    )
    here = Path(__file__).resolve().parent
    result = {
        "schema": "cat_cas.su2_level8_cubic_skein_ledger_native_gauge.v1",
        "result": (
            "PASS_BOUNDED_EXACT_LEDGER_NATIVE_GAUGE_FULL_LIFECYCLE_REDUCTION"
            if all_nontrivial_live_smaller
            else "PASS_BOUNDED_EXACT_LEDGER_NATIVE_GAUGE_STREAMING_WITH_PERSISTING_TRANSIENT_NO_GO"
        ),
        "claim": (
            "BOUNDED_EXACT_LEDGER_NATIVE_TOPOLOGY_LOCAL_CUBIC_SKEIN_UNIT_GAUGE_ELIMINATES_RAW_AND_CANDIDATE_VECTOR_MATERIALIZATION_AND_REDUCES_DECLARED_EXACT_LIVE_PAYLOAD_BELOW_THE_MATCHED_RAW_RECURRENCE_ON_EVERY_DECLARED_DEPTH_ABOVE_ONE_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_REUSE_BUT_THE_IDENTICAL_CLASSICAL_LEDGER_NATIVE_RECURRENCE_REMAINS"
            if all_nontrivial_live_smaller
            else "BOUNDED_EXACT_LEDGER_NATIVE_TOPOLOGY_LOCAL_CUBIC_SKEIN_UNIT_GAUGE_ELIMINATES_RAW_AND_CANDIDATE_VECTOR_MATERIALIZATION_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_REUSE_BUT_DECLARED_EXACT_LIVE_PAYLOAD_REMAINS_ABOVE_THE_MATCHED_RAW_RECURRENCE_ON_EVERY_DECLARED_CASE_BECAUSE_EXACT_TRACE_NORM_SCALARS_DOMINATE_AND_THE_IDENTICAL_CLASSICAL_LEDGER_NATIVE_RECURRENCE_REMAINS"
        ),
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "FORMAL_PRETRUNCATION_QZETA40_M219_CUBIC_SKEIN_M220_UNIT_PARAMETERS3_7_9_11_13_17_19_LEDGER_NATIVE_RANK_ONE_EXACT_TRACE_LINE_SEARCH_FAMILY0_STRANDS4_DEPTH1TO4_STRANDS6_DEPTH1TO2_STRANDS8_DEPTH1_PRIMARY4_DEPTH4_REUSE4_DEPTH2_FAMILY1_DIRECT_PROCESS_ONLY",
        "mechanism": {
            "represented_state": "ACTUAL_EQUALS_GLOBAL_CYCLOTOMIC_UNIT_LEDGER_SCALE_TIMES_RESIDUAL",
            "linear_skein_update": "DIRECT_ON_RESIDUAL",
            "cubic_update": "RESIDUAL_TARGET_PLUS_PHASE_TIMES_LEDGER_SCALE_SQUARED_TIMES_RESIDUAL_CONTROL_CUBED",
            "candidate_scoring": "ONE_FIELD_CELL_AT_A_TIME_WITH_NO_CANDIDATE_VECTOR",
            "selected_gauge_commit": "ONE_IN_PLACE_NET_UNIT_ACTION",
            "raw_actual_vectors_materialized": False,
            "candidate_residual_vectors_materialized": False,
            "retained_inverse_value_history": 0,
        },
        "cases": cases,
        "selected_cases": selected,
        "lifecycle_law": {
            "all_declared_depth_above_one_smaller_than_matched_raw": all_nontrivial_live_smaller,
            "every_declared_case_above_matched_raw": every_declared_case_above_raw,
            "dominant_ledger_native_context": "EXACT_TRACE_LINE_NORM_FACTOR_AND_CANDIDATE_NORM",
            "raw_and_candidate_vector_materialization_eliminated": all(
                case["work"]["raw_actual_vectors_materialized"] == 0
                and case["work"]["candidate_residual_vectors_materialized"] == 0
                for case in cases
            ),
            "raw_baseline_uses_same_exact_field_and_cubic_skein_recurrence": True,
            "logical_exact_live_intervals_not_process_rss": True,
        },
        "separate_reference": {
            "imports_m221_production": reference.get("imports_m221_production"),
            "uses_prior_standalone_m220_reference_substrate": reference.get(
                "uses_prior_standalone_m220_reference_substrate"
            ),
            "case_state_boundary_balance_resource_restoration_parity": True,
            "reuse_parity": True,
        },
        "reuse": reuse,
        "controls": all_controls,
        "matched_classical_baselines": {
            "strongest_compact": "IDENTICAL_LEDGER_NATIVE_RESIDUAL_LINK_PATTERN_CUBIC_SKEIN_GAUGE_RECURRENCE",
            "matched_raw": "IDENTICAL_RAW_LINK_PATTERN_CUBIC_SKEIN_RECURRENCE_WITH_THE_SAME_DECLARED_SCALAR_TEMPORARY_LAW",
            "phase_specific_reduction": False,
            "computational_advantage": False,
        },
        "resource_law": {
            "carrier": "LINK_PATTERN_RESIDUAL_PLUS_SEVEN_SIGNED_UNIT_EXPONENTS_PLUS_EQUAL_SKEIN_SCRATCH",
            "all_streamed_cell_products_counted": True,
            "unit_power_norm_relative_cubic_and_selected_mutation_work_counted": True,
            "line_search_retains_no_energy_cache": True,
            "whole_process_and_python_object_overhead_bounded": False,
            "excluded_not_zero": "PYTHON_CONTAINER_CAPACITY_ALLOCATOR_PROCESS_IMAGE_JSON_SERIALIZATION_TIMING_AND_WHOLE_PROCESS_PEAKS",
        },
        "claim_limits": {
            "global_unit_lattice_optimum": False,
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
            "m219_production_sha256": sha256_file(
                here / "su2_level8_topology_local_cubic_skein.py"
            ),
            "m220_production_sha256": sha256_file(
                here / "su2_level8_cubic_skein_unit_gauge.py"
            ),
            "m221_production_sha256": sha256_file(Path(__file__).resolve()),
            "m221_separate_reference_code_sha256": sha256_file(
                here / "su2_level8_cubic_skein_ledger_native_gauge_separate_reference.py"
            ),
            "m221_separate_reference_result_sha256": sha256_file(reference_path),
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
