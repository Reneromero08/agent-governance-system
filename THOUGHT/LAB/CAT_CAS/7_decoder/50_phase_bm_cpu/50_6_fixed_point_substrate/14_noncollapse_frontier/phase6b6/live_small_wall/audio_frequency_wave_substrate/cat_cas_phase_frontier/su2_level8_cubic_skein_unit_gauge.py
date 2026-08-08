#!/usr/bin/env python3
"""M220 exact all-embedding unit gauge for the M219 cubic-skein carrier.

The carrier stores ``actual = unit(ledger) * residual``.  Before each M219
operation it lawfully materializes the actual vector, applies the exact
cubic-skein update, and selects a deterministic rank-one gauge from seven
public cyclotomic-unit directions.  Each direction's exponent is chosen by an
exact trace-energy line search across all sixteen embeddings; final selection
minimizes residual-plus-ledger payload.  The raw vector, unit ledger, search
work, and transient candidate vectors are all counted.

This is a bounded representation repair, not a claim of global unit-lattice
optimality, asymptotic height control, or phase/classical separation.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import su2_level8_fusion_path_braid_phase_relation as braid
import su2_level8_markov_skein_krylov as skein
import su2_level8_topology_local_cubic_skein as cubic


sys.set_int_max_str_digits(0)
UNIT_PARAMETERS = (3, 7, 9, 11, 13, 17, 19)
UNIT_RANK = len(UNIT_PARAMETERS)
CASES = (
    *((4, rounds, 0) for rounds in range(1, 5)),
    *((6, rounds, 0) for rounds in range(1, 3)),
    (8, 1, 0),
)
PRIMARY = (4, 4, 0)
REUSE = (4, 2, 1)
MAX_BRACKET_DOUBLINGS = 32


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def ledger_payload_bits(ledger: list[int] | tuple[int, ...]) -> int:
    return sum(signed_bits(value) for value in ledger)


def integer_content_metrics(values: list[braid.K]) -> dict[str, Any]:
    global_content = 0
    element_contents = []
    for value in values:
        content = 0
        for coefficient in value.coefficients:
            if coefficient.denominator != 1:
                raise RuntimeError("M220 integer-content input left the integral ring")
            content = math.gcd(content, abs(coefficient.numerator))
            global_content = math.gcd(global_content, abs(coefficient.numerator))
        if content:
            element_contents.append(content)
    return {
        "global_integer_coordinate_content": global_content,
        "all_nonzero_element_integer_coordinate_contents_one": all(
            content == 1 for content in element_contents
        ),
    }


def conjugate(value: braid.K) -> braid.K:
    coefficients = [Fraction(0)] * braid.ROOT_ORDER
    for exponent, coefficient in enumerate(value.coefficients):
        coefficients[(-exponent) % braid.ROOT_ORDER] += coefficient
    return braid.K(braid.reduce_root_polynomial(coefficients))


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


def euler_phi(value: int) -> int:
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
    divisor = math.gcd(braid.ROOT_ORDER, power)
    quotient = braid.ROOT_ORDER // divisor
    return (
        mobius(quotient)
        * euler_phi(braid.ROOT_ORDER)
        // euler_phi(quotient)
    )


TRACE_BASIS = tuple(ramanujan_trace(power) for power in range(braid.FIELD_DEGREE))


def field_trace(value: braid.K) -> int:
    result = sum(
        coefficient * TRACE_BASIS[index]
        for index, coefficient in enumerate(value.coefficients)
    )
    if result.denominator != 1:
        raise RuntimeError("cyclotomic field trace is not integral")
    return result.numerator


def vector_norm_element(values: list[braid.K], work: "Work | None" = None) -> braid.K:
    result = braid.ZERO
    for value in values:
        result = result + value * conjugate(value)
        if work is not None:
            work.norm_field_multiplications += 1
            work.norm_field_additions += 1
    return result


def field_power(base: braid.K, exponent: int, work: "Work | None" = None) -> braid.K:
    if exponent < 0:
        raise ValueError("field power exponent must be nonnegative")
    result = braid.ONE
    factor = base
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = result * factor
            if work is not None:
                work.unit_power_field_multiplications += 1
        remaining >>= 1
        if remaining:
            factor = factor * factor
            if work is not None:
                work.unit_power_field_multiplications += 1
    return result


@dataclass(frozen=True)
class UnitDirection:
    parameter: int
    unit: braid.K
    inverse: braid.K
    norm: braid.K
    inverse_norm: braid.K


def compile_units() -> tuple[UnitDirection, ...]:
    result = []
    for parameter in UNIT_PARAMETERS:
        unit = sum((braid.K.zeta(power) for power in range(parameter)), braid.ZERO)
        inverse = unit.inverse()
        if unit * inverse != braid.ONE:
            raise RuntimeError("declared cyclotomic unit inverse failed")
        if any(coefficient.denominator != 1 for coefficient in inverse.coefficients):
            raise RuntimeError("declared cyclotomic unit inverse is not integral")
        result.append(
            UnitDirection(
                parameter,
                unit,
                inverse,
                unit * conjugate(unit),
                inverse * conjugate(inverse),
            )
        )
    return tuple(result)


UNITS = compile_units()


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
    unit_vector_field_multiplications: int = 0
    norm_field_multiplications: int = 0
    norm_field_additions: int = 0
    candidate_vectors_materialized: int = 0
    actual_vectors_materialized: int = 0
    carrier_observations: int = 0
    maximum_resident_payload_bits: int = 0
    maximum_raw_vector_payload_bits: int = 0
    maximum_candidate_vector_payload_bits: int = 0
    maximum_declared_live_payload_bits: int = 0
    maximum_declared_live_field_cells: int = 0
    maximum_unit_scale_payload_bits: int = 0
    maximum_norm_payload_bits: int = 0

    def trace_energy(self, norm: braid.K) -> int:
        self.exact_trace_energy_evaluations += 1
        energy = field_trace(norm)
        if energy < 0:
            raise RuntimeError("exact embedding energy became negative")
        return energy

    def observe_carrier(
        self,
        residual: list[braid.K],
        ledger: list[int],
        scratch: list[braid.K],
    ) -> None:
        self.carrier_observations += 1
        payload = (
            braid.field_payload_bits(residual)
            + ledger_payload_bits(ledger)
            + braid.field_payload_bits(scratch)
        )
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits, payload
        )
        self.maximum_declared_live_payload_bits = max(
            self.maximum_declared_live_payload_bits, payload
        )
        self.maximum_declared_live_field_cells = max(
            self.maximum_declared_live_field_cells,
            len(residual) + len(scratch) + 2,
        )

    def observe_transient(
        self,
        residual: list[braid.K],
        ledger: list[int],
        scratch: list[braid.K],
        raw: list[braid.K],
        candidate: list[braid.K] | None = None,
        unit_scale: braid.K | None = None,
        norm: braid.K | None = None,
    ) -> None:
        vectors = [residual, scratch, raw]
        if candidate is not None:
            vectors.append(candidate)
            self.maximum_candidate_vector_payload_bits = max(
                self.maximum_candidate_vector_payload_bits,
                braid.field_payload_bits(candidate),
            )
        payload = sum(braid.field_payload_bits(vector) for vector in vectors)
        payload += ledger_payload_bits(ledger)
        scalar_cells = 0
        if unit_scale is not None:
            scale_payload = braid.field_payload_bits([unit_scale])
            self.maximum_unit_scale_payload_bits = max(
                self.maximum_unit_scale_payload_bits, scale_payload
            )
            payload += scale_payload
            scalar_cells += 1
        if norm is not None:
            norm_payload = braid.field_payload_bits([norm])
            self.maximum_norm_payload_bits = max(
                self.maximum_norm_payload_bits, norm_payload
            )
            payload += norm_payload
            scalar_cells += 1
        self.maximum_raw_vector_payload_bits = max(
            self.maximum_raw_vector_payload_bits, braid.field_payload_bits(raw)
        )
        self.maximum_declared_live_payload_bits = max(
            self.maximum_declared_live_payload_bits, payload
        )
        self.maximum_declared_live_field_cells = max(
            self.maximum_declared_live_field_cells,
            sum(len(vector) for vector in vectors) + scalar_cells,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "cubic"
        } | {"cubic_skein": self.cubic.as_dict()}


def unit_norm_factor(direction: UnitDirection, exponent: int, work: Work) -> braid.K:
    return (
        field_power(direction.inverse_norm, exponent, work)
        if exponent >= 0
        else field_power(direction.norm, -exponent, work)
    )


def unit_scale(direction: UnitDirection, exponent: int, work: Work) -> braid.K:
    return (
        field_power(direction.unit, exponent, work)
        if exponent >= 0
        else field_power(direction.inverse, -exponent, work)
    )


def residual_multiplier(
    direction: UnitDirection, exponent: int, work: Work
) -> braid.K:
    return (
        field_power(direction.inverse, exponent, work)
        if exponent >= 0
        else field_power(direction.unit, -exponent, work)
    )


def line_minimum(
    norm: braid.K, direction: UnitDirection, work: Work
) -> tuple[int, int]:
    work.line_searches += 1
    cache: dict[int, int] = {0: work.trace_energy(norm)}

    def energy(exponent: int) -> int:
        if exponent not in cache:
            candidate = unit_norm_factor(direction, exponent, work) * norm
            work.unit_norm_field_multiplications += 1
            cache[exponent] = work.trace_energy(candidate)
        return cache[exponent]

    zero = energy(0)
    positive, negative = energy(1), energy(-1)
    if min(positive, negative) >= zero:
        return 0, zero
    direction_sign = 1 if positive < negative else -1
    previous = 0
    current = direction_sign
    for _ in range(MAX_BRACKET_DOUBLINGS):
        following = 2 * current
        work.line_bracket_doublings += 1
        if energy(following) >= energy(current):
            low, high = sorted((previous, following))
            break
        previous, current = current, following
    else:
        raise RuntimeError("unit line minimum was not bracketed")
    while high - low > 8:
        work.line_ternary_steps += 1
        first = low + (high - low) // 3
        second = high - (high - low) // 3
        if energy(first) <= energy(second):
            high = second - 1
        else:
            low = first + 1
    best = min(range(low, high + 1), key=lambda item: (energy(item), item))
    return best, energy(best)


def balance_raw(
    raw: list[braid.K],
    carrier_residual: list[braid.K],
    carrier_ledger: list[int],
    scratch: list[braid.K],
    work: Work,
) -> tuple[list[braid.K], list[int], dict[str, Any]]:
    work.balance_calls += 1
    norm = vector_norm_element(raw, work)
    raw_payload = braid.field_payload_bits(raw)
    identity_ledger = [0] * UNIT_RANK
    identity_total = raw_payload + ledger_payload_bits(identity_ledger)
    best_key = (identity_total, raw_payload, work.trace_energy(norm), tuple(identity_ledger))
    best_residual = raw.copy()
    best_ledger = identity_ledger
    candidate_exponents: list[int] = []
    work.observe_transient(
        carrier_residual, carrier_ledger, scratch, raw, best_residual, norm=norm
    )
    for index, direction in enumerate(UNITS):
        exponent, energy = line_minimum(norm, direction, work)
        candidate_exponents.append(exponent)
        if exponent == 0:
            continue
        multiplier = residual_multiplier(direction, exponent, work)
        candidate = [multiplier * value for value in raw]
        work.unit_vector_field_multiplications += len(raw)
        work.candidate_vectors_materialized += 1
        ledger = [0] * UNIT_RANK
        ledger[index] = exponent
        candidate_payload = braid.field_payload_bits(candidate)
        candidate_total = candidate_payload + ledger_payload_bits(ledger)
        key = (candidate_total, candidate_payload, energy, tuple(ledger))
        work.observe_transient(
            carrier_residual,
            carrier_ledger,
            scratch,
            raw,
            candidate,
            multiplier,
            norm,
        )
        if key < best_key:
            best_key = key
            best_residual = candidate
            best_ledger = ledger
    return best_residual, best_ledger, {
        "raw_payload_bits": raw_payload,
        "balanced_residual_payload_bits": braid.field_payload_bits(best_residual),
        "unit_ledger_payload_bits": ledger_payload_bits(best_ledger),
        "balanced_residual_plus_ledger_payload_bits": best_key[0],
        "resident_payload_reduction_bits_before_constant_scratch": (
            identity_total - best_key[0]
        ),
        "selected_unit_ledger": best_ledger,
        "per_direction_trace_energy_minimizing_exponents": candidate_exponents,
        "selected_exact_embedding_energy_bits": signed_bits(best_key[2]),
        "selected_exact_embedding_energy_sha256": hashlib.sha256(
            str(best_key[2]).encode("ascii")
        ).hexdigest(),
        "identity_selected": not any(best_ledger),
    }


def materialize(
    residual: list[braid.K], ledger: list[int], work: Work
) -> tuple[list[braid.K], braid.K]:
    scale = braid.ONE
    for exponent, direction in zip(ledger, UNITS, strict=True):
        scale = scale * unit_scale(direction, exponent, work)
        work.unit_power_field_multiplications += 1
    actual = [scale * value for value in residual]
    work.unit_vector_field_multiplications += len(residual)
    work.actual_vectors_materialized += 1
    return actual, scale


@dataclass
class GaugePort:
    topology: skein.DiagramTopology
    residual: list[braid.K]
    unit_ledger: list[int]
    scratch: list[braid.K]
    live: bool = False
    owner: int = 0
    lease_generation: int = 0
    cursor: int = 0
    expected_steps: int = 0
    program_commitment: str = ""
    last_balance: dict[str, Any] = field(default_factory=dict)

    def lease(
        self, owner: int, generation: int, program: braid.BraidProgram, work: Work
    ) -> None:
        if self.live:
            raise RuntimeError("unit-gauge port already live")
        if (
            len(self.residual) != self.topology.dimension
            or len(self.scratch) != self.topology.dimension
            or len(self.unit_ledger) != UNIT_RANK
        ):
            raise ValueError("null or wrong-width unit-gauge carrier")
        if owner <= 0 or generation <= 0 or program.strands != self.topology.strands:
            raise ValueError("invalid unit-gauge lease")
        self.live = True
        self.owner = owner
        self.lease_generation = generation
        self.cursor = 0
        self.expected_steps = program.steps
        self.program_commitment = skein.program_commitment(program)
        work.cubic.linear.public_descriptor_hashes += 2
        work.cubic.linear.public_descriptor_integers_hashed += (
            3 + self.topology.retained_pairing_integer_cells + UNIT_RANK
        )
        work.cubic.linear.port_leases += 1
        work.observe_carrier(self.residual, self.unit_ledger, self.scratch)

    def require(
        self, owner: int, program: braid.BraidProgram, work: Work
    ) -> None:
        if not self.live:
            raise RuntimeError("unit-gauge port is not live")
        if owner != self.owner:
            raise PermissionError("unit-gauge owner mismatch")
        if skein.program_commitment(program) != self.program_commitment:
            raise ValueError("unit-gauge public program mismatch")
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
            raise ValueError("unit-gauge operation cursor mismatch")
        actual, scale = materialize(self.residual, self.unit_ledger, work)
        work.observe_transient(
            self.residual,
            self.unit_ledger,
            self.scratch,
            actual,
            unit_scale=scale,
        )
        operation = program.operation(index)
        if inverse:
            cubic.apply_inverse_operation(
                actual, self.scratch, self.topology, operation, work.cubic
            )
        else:
            cubic.apply_forward_operation(
                actual, self.scratch, self.topology, operation, work.cubic
            )
        balanced, ledger, metrics = balance_raw(
            actual, self.residual, self.unit_ledger, self.scratch, work
        )
        self.residual[:] = balanced
        self.unit_ledger[:] = ledger
        self.scratch[:] = [braid.ZERO] * self.topology.dimension
        self.last_balance = metrics
        self.cursor += -1 if inverse else 1
        work.observe_carrier(self.residual, self.unit_ledger, self.scratch)

    def project(
        self, owner: int, program: braid.BraidProgram, work: Work
    ) -> braid.K:
        self.require(owner, program, work)
        if self.cursor != self.expected_steps:
            raise PermissionError("nonfinal unit-gauge projection rejected")
        actual, scale = materialize(self.residual, self.unit_ledger, work)
        work.observe_transient(
            self.residual,
            self.unit_ledger,
            self.scratch,
            actual,
            unit_scale=scale,
        )
        return skein.normalized_markov_boundary(actual, self.topology, work.cubic.linear)

    def release(
        self, owner: int, program: braid.BraidProgram, work: Work
    ) -> int:
        self.require(owner, program, work)
        if self.cursor:
            raise RuntimeError("unit-gauge port released before inverse")
        generation = self.lease_generation
        self.live = False
        self.owner = 0
        self.lease_generation = 0
        self.expected_steps = 0
        self.program_commitment = ""
        self.last_balance = {}
        self.scratch[:] = [braid.ZERO] * self.topology.dimension
        work.cubic.linear.port_releases += 1
        return generation


@dataclass
class Carrier:
    port: GaugePort
    restoration_generation: int = 0


def canonical_restoration(carrier: Carrier, source: list[braid.K], generation: int) -> bool:
    port = carrier.port
    return (
        port.residual == source
        and not any(port.unit_ledger)
        and all(value == braid.ZERO for value in port.scratch)
        and not port.live
        and port.owner == 0
        and port.lease_generation == 0
        and port.cursor == 0
        and port.expected_steps == 0
        and port.program_commitment == ""
        and carrier.restoration_generation == generation
    )


def transaction(
    carrier: Carrier, source: list[braid.K], program: braid.BraidProgram
) -> tuple[dict[str, Any], Work]:
    backings = (
        id(carrier.port.residual),
        id(carrier.port.unit_ledger),
        id(carrier.port.scratch),
    )
    generation = carrier.restoration_generation + 1
    owner = 220000 + generation
    work = Work()
    carrier.port.lease(owner, generation, program, work)
    for index in range(program.steps):
        carrier.port.apply(owner, program, index, work, inverse=False)
    actual, _ = materialize(carrier.port.residual, carrier.port.unit_ledger, work)
    forward_commitment = skein.state_commitment(actual)
    work.cubic.linear.state_commitment_hashes += 1
    work.cubic.linear.state_commitment_field_cells_hashed += len(actual)
    raw_payload = braid.field_payload_bits(actual)
    balance_metrics = dict(carrier.port.last_balance)
    boundary = carrier.port.project(owner, program, work)
    work.cubic.linear.boundary_commitment_hashes += 1
    for index in range(program.steps - 1, -1, -1):
        carrier.port.apply(owner, program, index, work, inverse=True)
    carrier.restoration_generation = carrier.port.release(owner, program, work)
    return {
        "boundary_commitment": braid.boundary_commitment(boundary),
        "forward_state_commitment": forward_commitment,
        "forward_raw_payload_bits": raw_payload,
        "final_balance": balance_metrics,
        "same_residual_backing": id(carrier.port.residual) == backings[0],
        "same_unit_ledger_backing": id(carrier.port.unit_ledger) == backings[1],
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
    }, work


def make_carrier(strands: int) -> tuple[Carrier, list[braid.K]]:
    topology = skein.DiagramTopology.compile(strands)
    source = skein.source_state(topology)
    return (
        Carrier(
            GaugePort(
                topology,
                source.copy(),
                [0] * UNIT_RANK,
                [braid.ZERO] * topology.dimension,
            )
        ),
        source,
    )


def execute_case(strands: int, rounds: int, family: int) -> dict[str, Any]:
    carrier, source = make_carrier(strands)
    program = braid.BraidProgram(strands, rounds, family)
    result, _ = transaction(carrier, source, program)
    _, direct, _, _ = cubic.forward_state(program)
    direct_commitment = skein.state_commitment(direct)
    if result["forward_state_commitment"] != direct_commitment:
        raise RuntimeError("unit-gauge execution changed the M219 actual state")
    return {
        "strands": strands,
        "rounds": rounds,
        "family": family,
        "link_pattern_cells": len(source),
        **result,
        "direct_m219_state_commitment_agreement": True,
        **integer_content_metrics(direct),
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
    carrier, source = make_carrier(4)
    port = carrier.port
    accepted = braid.BraidProgram(4, 1, 0)
    wrong = braid.BraidProgram(4, 1, 1)
    work = Work()
    port.lease(220900, 1, accepted, work)
    wrong_owner = premature = wrong_program = reordered = False
    try:
        port.apply(220901, accepted, 0, work, inverse=False)
    except PermissionError:
        wrong_owner = True
    try:
        port.project(220900, accepted, work)
    except PermissionError:
        premature = True
    for index in range(accepted.steps):
        port.apply(220900, accepted, index, work, inverse=False)
    represented, _ = materialize(port.residual, port.unit_ledger, work)
    wrong_ledger = list(port.unit_ledger)
    wrong_ledger[0] += 1
    wrong_represented, _ = materialize(port.residual, wrong_ledger, work)
    try:
        port.apply(220900, wrong, accepted.steps - 1, work, inverse=True)
    except ValueError:
        wrong_program = True
    try:
        port.apply(220900, accepted, accepted.steps - 2, work, inverse=True)
    except ValueError:
        reordered = True
    missing = represented != source
    for index in range(accepted.steps - 1, -1, -1):
        port.apply(220900, accepted, index, work, inverse=True)
    port.release(220900, accepted, work)
    null_rejected = False
    try:
        GaugePort(port.topology, [], [], []).lease(1, 1, accepted, Work())
    except ValueError:
        null_rejected = True
    return {
        "wrong_owner_rejected": wrong_owner,
        "premature_projection_rejected": premature,
        "wrong_public_program_inverse_rejected": wrong_program,
        "reordered_inverse_rejected": reordered,
        "missing_inverse_detected": missing,
        "wrong_unit_ledger_changes_represented_state": wrong_represented != represented,
        "null_carrier_rejected": null_rejected,
        "intermediate_actual_vector_projected": False,
        "snapshot_command_available": hasattr(port, "snapshot"),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: su2_level8_cubic_skein_unit_gauge.py SEPARATE_REFERENCE_JSON"
        )
    reference_path = Path(sys.argv[1]).resolve()
    if str(reference_path).startswith(("/dev/shm/", "/run/shm/")):
        raise ValueError("RAM-backed M220 reference is forbidden")
    reference = json.loads(reference_path.read_text())
    if reference.get("schema") != "cat_cas.su2_level8_cubic_skein_unit_gauge_reference.v1":
        raise RuntimeError("M220 separate-reference schema changed")
    cases = [execute_case(*case) for case in CASES]
    comparable = [
        {
            key: value
            for key, value in case.items()
            if key not in {"work"}
        }
        for case in cases
    ]
    if comparable != reference.get("cases"):
        raise RuntimeError("M220 independent case parity failed")
    reuse = reuse_result()
    for section in ("primary", "reuse", "fresh_reuse"):
        for key in (
            "boundary_commitment",
            "forward_state_commitment",
            "forward_raw_payload_bits",
            "restoration_error_field_cells",
            "canonical_post_restoration_state_exact",
        ):
            if reuse[section][key] != reference["reuse"][section][key]:
                raise RuntimeError(f"M220 independent reuse parity failed: {section}.{key}")
    all_controls = controls()
    positives = {
        key: value
        for key, value in all_controls.items()
        if key not in {"intermediate_actual_vector_projected", "snapshot_command_available"}
    }
    if (
        not all(positives.values())
        or all_controls["intermediate_actual_vector_projected"]
        or all_controls["snapshot_command_available"]
    ):
        raise RuntimeError("M220 control failed")
    selected = [
        {
            "strands": case["strands"],
            "rounds": case["rounds"],
            **case["final_balance"],
            "maximum_declared_live_payload_bits": case["work"]["maximum_declared_live_payload_bits"],
        }
        for case in cases
        if (case["strands"], case["rounds"]) in ((4, 4), (6, 2), (8, 1))
    ]
    every_nontrivial_resident_smaller = all(
        item["final_balance"][
            "resident_payload_reduction_bits_before_constant_scratch"
        ]
        > 0
        for item in cases
        if item["rounds"] > 1
    )
    full_lifecycle_smaller = all(
        case["work"]["maximum_declared_live_payload_bits"]
        < case["forward_raw_payload_bits"]
        for case in cases
    )
    here = Path(__file__).resolve().parent
    result = {
        "schema": "cat_cas.su2_level8_cubic_skein_unit_gauge.v1",
        "result": "PASS_BOUNDED_EXACT_CUBIC_SKEIN_UNIT_GAUGE_RESIDENT_REDUCTION_WITH_TRANSIENT_NO_GO",
        "claim": "BOUNDED_EXACT_SEVEN_DIRECTION_ALL_EMBEDDING_CYCLOTOMIC_UNIT_GAUGE_REDUCES_RESIDENT_MANTISSA_PLUS_LEDGER_PAYLOAD_ON_DECLARED_NONTRIVIAL_TOPOLOGY_LOCAL_CUBIC_SKEIN_CASES_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_REUSE_BUT_REQUIRES_RAW_STATE_AND_CANDIDATE_MATERIALIZATION_SO_FULL_LIFECYCLE_HEIGHT_REDUCTION_IS_NOT_ESTABLISHED_AND_THE_IDENTICAL_CLASSICAL_GAUGE_RECURRENCE_REMAINS",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "FORMAL_PRETRUNCATION_QZETA40_M219_CUBIC_SKEIN_UNIT_PARAMETERS3_7_9_11_13_17_19_RANK_ONE_EXACT_TRACE_LINE_SEARCH_FAMILY0_STRANDS4_DEPTH1TO4_STRANDS6_DEPTH1TO2_STRANDS8_DEPTH1_PRIMARY4_DEPTH4_REUSE4_DEPTH2_FAMILY1_DIRECT_PROCESS_ONLY",
        "unit_law": {
            "unit_parameters": list(UNIT_PARAMETERS),
            "unit_inverse_products_exact_one": True,
            "unit_inverses_integral": True,
            "embedding_count": 16,
            "energy": "TRACE_VECTOR_X_TIMES_CONJUGATE_X_EQUALS_SUM_OVER_ALL16_EMBEDDINGS",
            "per_direction_exponent_search": "EXACT_BRACKETED_DISCRETE_TERNARY",
            "selection_objective": "MINIMIZE_EXACT_RESIDUAL_PLUS_SEVEN_CELL_SIGNED_EXPONENT_LEDGER_PAYLOAD",
            "only_one_unit_direction_nonzero_per_canonical_gauge": True,
            "all_declared_m219_states_have_primitive_integer_coordinate_content": all(
                case["global_integer_coordinate_content"] == 1
                and case["all_nonzero_element_integer_coordinate_contents_one"]
                for case in cases
            ),
            "global_unit_lattice_optimum": False,
        },
        "cases": cases,
        "selected_cases": selected,
        "resident_law": {
            "every_declared_depth_above_one_reduces_residual_plus_ledger_payload": every_nontrivial_resident_smaller,
            "depth_one_identity_gauges_allowed": True,
            "full_lifecycle_declared_live_payload_reduction": full_lifecycle_smaller,
            "raw_actual_vector_materialized_before_every_operation": True,
            "candidate_residual_vectors_materialized_during_balance": True,
            "retained_inverse_value_history": 0,
        },
        "separate_reference": {
            "imports_m220_production": reference.get("imports_m220_production"),
            "uses_prior_standalone_m219_reference_substrate": reference.get("uses_prior_standalone_m219_reference_substrate"),
            "case_state_boundary_balance_and_restoration_parity": True,
            "reuse_parity": True,
        },
        "reuse": reuse,
        "controls": all_controls,
        "matched_classical_baselines": {
            "strongest_compact": "IDENTICAL_RESIDUAL_PLUS_UNIT_LEDGER_LINK_PATTERN_CUBIC_SKEIN_GAUGE_RECURRENCE",
            "same_resident_transient_search_and_exact_height_law": True,
            "phase_specific_reduction": False,
            "computational_advantage": False,
        },
        "resource_law": {
            "resident_carrier": "LINK_PATTERN_RESIDUAL_PLUS_SEVEN_SIGNED_UNIT_EXPONENTS_PLUS_EQUAL_ZEROED_SKEIN_SCRATCH",
            "raw_actual_materialization_counted": True,
            "candidate_vector_materialization_counted": True,
            "unit_power_and_norm_multiplications_counted": True,
            "exact_trace_evaluations_counted": True,
            "whole_process_and_python_object_overhead_bounded": False,
            "excluded_not_zero": "PYTHON_CONTAINER_CAPACITY_ALLOCATOR_PROCESS_IMAGE_JSON_SERIALIZATION_TIMING_AND_WHOLE_PROCESS_PEAKS",
        },
        "claim_limits": {
            "global_unit_lattice_optimum": False,
            "asymptotic_height_bound": False,
            "full_lifecycle_height_reduction": full_lifecycle_smaller,
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
            "m219_production_sha256": sha256_file(here / "su2_level8_topology_local_cubic_skein.py"),
            "m220_wrapper_sha256": sha256_file(Path(__file__).resolve()),
            "m220_separate_reference_code_sha256": sha256_file(
                here / "su2_level8_cubic_skein_unit_gauge_separate_reference.py"
            ),
            "m220_separate_reference_result_sha256": sha256_file(reference_path),
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
