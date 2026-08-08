#!/usr/bin/env python3
"""M232 exact trivalent SU(2)_8 shared-channel relation contraction.

This is a representation change from M231.  Two analytic trivalent fusion
vertices share the unresolved channel e in {0, 2} for four fundamental
charges.  The standard non-diagonal four-fundamental associator is applied,
the middle-pair braid phase acts in its natural fusion basis, and the
associator returns the message before the right vertex contracts it to the
only projectable scalar.  No fusion table or assignment expansion is stored.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import su2_level8_fusion_path_braid_phase_relation as exact


sys.set_int_max_str_digits(0)

K = exact.K
ZERO = exact.ZERO
ONE = exact.ONE
LEVEL = exact.LEVEL
CHANNELS = (0, 2)
FAMILIES = (0, 1)
PRIMARY_FAMILY = 0
REUSE_FAMILY = 1
LEFT_PORT_TYPE = 23201
CHANNEL_PORT_TYPE = 23202
RIGHT_PORT_TYPE = 23203
BOUNDARY_PORT_TYPE = 23204
DELTA = exact.QUANTUM_DIMENSIONS[1]
INVERSE_DELTA = exact.INVERSE_DIMENSIONS[1]
PHI = K.zeta(4) + K.zeta(-4)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def admissible(a: int, b: int, c: int) -> bool:
    if not all(0 <= label <= LEVEL for label in (a, b, c)):
        return False
    return (
        (a + b + c) % 2 == 0
        and abs(a - b) <= c
        and c <= min(a + b, 2 * LEVEL - a - b)
    )


def trivalent_value(a: int, b: int, c: int) -> K:
    return ONE if admissible(a, b, c) else ZERO


@dataclass(frozen=True)
class CompiledTopology:
    left_a: int
    left_b: int
    right_c: int
    right_d: int
    braid_exponent: int
    left_type: int = LEFT_PORT_TYPE
    channel_type: int = CHANNEL_PORT_TYPE
    right_type: int = RIGHT_PORT_TYPE
    boundary_type: int = BOUNDARY_PORT_TYPE

    def __post_init__(self) -> None:
        if (self.left_a, self.left_b, self.right_c, self.right_d) != (1, 1, 1, 1):
            raise ValueError("M232 bounded topology requires four fundamentals")
        if self.braid_exponent not in (-1, 1):
            raise ValueError("M232 braid exponent must be plus or minus one")
        if (
            self.left_type != LEFT_PORT_TYPE
            or self.channel_type != CHANNEL_PORT_TYPE
            or self.right_type != RIGHT_PORT_TYPE
            or self.boundary_type != BOUNDARY_PORT_TYPE
        ):
            raise TypeError("M232 typed trivalent topology mismatch")
        left_channels = tuple(
            channel
            for channel in range(LEVEL + 1)
            if admissible(self.left_a, self.left_b, channel)
            and admissible(channel, self.right_c, self.right_d)
        )
        if left_channels != CHANNELS:
            raise RuntimeError("M232 analytic shared-channel law changed")

    def integers(self) -> tuple[int, ...]:
        return (
            self.left_a,
            self.left_b,
            self.right_c,
            self.right_d,
            self.braid_exponent,
            self.left_type,
            self.channel_type,
            self.right_type,
            self.boundary_type,
            *CHANNELS,
        )


@dataclass(frozen=True)
class PublicProgram:
    family: int

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError("M232 public family mismatch")

    def compile(self) -> CompiledTopology:
        return CompiledTopology(1, 1, 1, 1, 1 if self.family == 0 else -1)

    def token(self) -> str:
        return f"family:{self.family}|" + ":".join(
            str(value) for value in self.compile().integers()
        )


def program_commitment(program: PublicProgram) -> str:
    return hashlib.sha256(program.token().encode("ascii")).hexdigest()


def f_coefficient(row: int, column: int, *, offdiagonal_zero: bool = False) -> K:
    if row not in CHANNELS or column not in CHANNELS:
        raise ValueError("M232 associator channel outside analytic support")
    if row != column:
        return ZERO if offdiagonal_zero else PHI * INVERSE_DELTA
    return INVERSE_DELTA if row == 0 else ZERO - INVERSE_DELTA


def r_symbol(channel: int, exponent: int, *, phase_shift: int = 0) -> K:
    if channel not in CHANNELS or exponent not in (-1, 1):
        raise ValueError("M232 braid symbol outside declared support")
    a = K.zeta(11 + phase_shift)
    a_inverse = a.inverse()
    if exponent == 1:
        return a + a_inverse * DELTA if channel == 0 else a
    return a_inverse + a * DELTA if channel == 0 else a_inverse


@dataclass
class Work:
    field_additions: int = 0
    field_subtractions: int = 0
    field_multiplications: int = 0
    field_inversions: int = 0
    trivalent_admissibility_evaluations: int = 0
    left_vertex_message_productions: int = 0
    left_vertex_message_clears: int = 0
    associator_applications: int = 0
    braid_phase_applications: int = 0
    right_vertex_contractions: int = 0
    right_vertex_contraction_clears: int = 0
    port_leases: int = 0
    port_releases: int = 0
    owner_checks: int = 0
    generation_checks: int = 0
    program_checks: int = 0
    type_checks: int = 0
    relation_table_cells_materialized: int = 0
    assignment_expansions_materialized: int = 0
    intermediate_commitments_emitted: int = 0
    maximum_declared_live_field_cells: int = 0
    maximum_declared_live_payload_bits: int = 0
    maximum_declared_live_context: str = ""
    retained_topology_integers: tuple[int, ...] = field(default_factory=tuple)
    retained_result_values: tuple[K, ...] = field(default_factory=tuple, repr=False)

    def add(self, left: K, right: K) -> K:
        self.field_additions += 1
        return left + right

    def subtract(self, left: K, right: K) -> K:
        self.field_subtractions += 1
        return left - right

    def multiply(self, left: K, right: K) -> K:
        self.field_multiplications += 1
        return left * right

    def invert(self, value: K) -> K:
        self.field_inversions += 1
        return value.inverse()

    def observe(
        self,
        rails: Sequence[list[K]],
        *,
        transients: Sequence[K] = (),
        context: str,
    ) -> None:
        fields = (
            sum(len(rail) for rail in rails)
            + len(transients)
            + len(self.retained_result_values)
        )
        payload = sum(exact.field_payload_bits(rail) for rail in rails)
        payload += exact.field_payload_bits(list(transients))
        payload += exact.field_payload_bits(list(self.retained_result_values))
        payload += sum(signed_bits(value) for value in self.retained_topology_integers)
        self.maximum_declared_live_field_cells = max(
            self.maximum_declared_live_field_cells, fields
        )
        if payload > self.maximum_declared_live_payload_bits:
            self.maximum_declared_live_payload_bits = payload
            self.maximum_declared_live_context = context

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"retained_topology_integers", "retained_result_values"}
        }


def analytic_f_move(channel: list[K], work: Work, rails: Sequence[list[K]]) -> None:
    old_zero, old_two = channel
    c00 = INVERSE_DELTA
    offdiagonal = work.multiply(PHI, INVERSE_DELTA)
    c22 = work.subtract(ZERO, INVERSE_DELTA)
    p00 = work.multiply(c00, old_zero)
    p02 = work.multiply(offdiagonal, old_two)
    new_zero = work.add(p00, p02)
    p20 = work.multiply(offdiagonal, old_zero)
    p22 = work.multiply(c22, old_two)
    new_two = work.add(p20, p22)
    work.observe(
        rails,
        transients=(c00, offdiagonal, c22, old_zero, old_two, p00, p02, new_zero, p20, p22, new_two),
        context="NONDIAGONAL_F_MOVE",
    )
    channel[0] = new_zero
    channel[1] = new_two
    work.associator_applications += 1


def analytic_braid_phase(
    channel: list[K], exponent: int, work: Work, rails: Sequence[list[K]]
) -> None:
    old_zero, old_two = channel
    a = K.zeta(11)
    a_inverse = work.invert(a)
    if exponent == 1:
        r_zero = work.add(a, work.multiply(a_inverse, DELTA))
        r_two = a
        inverse_operand = a_inverse
    else:
        r_zero = work.add(a_inverse, work.multiply(a, DELTA))
        r_two = a_inverse
        inverse_operand = a
    del a, a_inverse
    new_zero = work.multiply(r_zero, old_zero)
    new_two = work.multiply(r_two, old_two)
    work.observe(
        rails,
        transients=(old_zero, old_two, r_zero, r_two, inverse_operand, new_zero, new_two),
        context="DIAGONAL_BRAID_PHASE",
    )
    channel[0] = new_zero
    channel[1] = new_two
    work.braid_phase_applications += 1


@dataclass
class TrivalentCarrier:
    channel: list[K]
    boundary: list[K]
    enabled: bool = True
    live: bool = False
    owner: int = 0
    generation: int = 0
    last_restored_generation: int = 0
    stage: int = 0
    program_token: str = ""

    def rails(self) -> tuple[list[K], list[K]]:
        return self.channel, self.boundary

    def lease(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> CompiledTopology:
        if self.live:
            raise RuntimeError("M232 trivalent port already live")
        if not self.enabled or len(self.channel) != 2 or len(self.boundary) != 1:
            raise ValueError("M232 null or wrong-width carrier")
        if any(value != ZERO for value in (*self.channel, *self.boundary)):
            raise ValueError("M232 carrier dirty at lease")
        if owner <= 0 or generation != self.last_restored_generation + 1:
            raise PermissionError("M232 invalid or stale generation")
        topology = program.compile()
        self.live = True
        self.owner = owner
        self.generation = generation
        self.stage = 0
        self.program_token = program_commitment(program)
        work.port_leases += 1
        work.retained_topology_integers = topology.integers()
        work.observe(self.rails(), context="LEASED_ZERO_CHANNEL")
        return topology

    def require(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> CompiledTopology:
        if not self.live:
            raise RuntimeError("M232 trivalent port not live")
        work.owner_checks += 1
        work.generation_checks += 1
        work.program_checks += 1
        work.type_checks += 1
        if owner != self.owner:
            raise PermissionError("M232 owner mismatch")
        if generation != self.generation:
            raise PermissionError("M232 generation mismatch")
        if program_commitment(program) != self.program_token:
            raise ValueError("M232 public program mismatch")
        return program.compile()

    def produce_left(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 0 or any(value != ZERO for value in self.channel):
            raise ValueError("M232 left vertex production order")
        for index, channel in enumerate(CHANNELS):
            value = trivalent_value(topology.left_a, topology.left_b, channel)
            work.trivalent_admissibility_evaluations += 1
            updated = work.add(self.channel[index], value)
            work.observe(
                self.rails(),
                transients=(value, updated),
                context="LEFT_TRIVALENT_MESSAGE_PRODUCTION",
            )
            self.channel[index] = updated
        self.stage = 1
        work.left_vertex_message_productions += 1

    def f_forward(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> None:
        self.require(owner, generation, program, work)
        if self.stage not in (1, 3):
            raise ValueError("M232 forward associator order")
        analytic_f_move(self.channel, work, self.rails())
        self.stage += 1

    def braid_forward(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 2:
            raise ValueError("M232 forward braid order")
        analytic_braid_phase(self.channel, topology.braid_exponent, work, self.rails())
        self.stage = 3

    def contract_right(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 4 or self.boundary[0] != ZERO:
            raise ValueError("M232 right contraction order")
        accumulator = ZERO
        for index, channel in enumerate(CHANNELS):
            vertex = trivalent_value(channel, topology.right_c, topology.right_d)
            work.trivalent_admissibility_evaluations += 1
            term = work.multiply(vertex, self.channel[index])
            updated = work.add(accumulator, term)
            work.observe(
                self.rails(),
                transients=(accumulator, vertex, term, updated),
                context="RIGHT_TRIVALENT_NATIVE_CONTRACTION",
            )
            accumulator = updated
        self.boundary[0] = work.add(self.boundary[0], accumulator)
        self.stage = 5
        work.right_vertex_contractions += 1

    def project(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> K:
        self.require(owner, generation, program, work)
        if self.stage != 5:
            raise PermissionError("M232 final boundary is not ready")
        return self.boundary[0]

    def project_channel(self) -> None:
        raise PermissionError("M232 shared latent channel is not projectable")

    def clear_right(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 5:
            raise ValueError("M232 right contraction inverse order")
        accumulator = ZERO
        for index, channel in enumerate(CHANNELS):
            vertex = trivalent_value(channel, topology.right_c, topology.right_d)
            work.trivalent_admissibility_evaluations += 1
            term = work.multiply(vertex, self.channel[index])
            accumulator = work.add(accumulator, term)
        self.boundary[0] = work.subtract(self.boundary[0], accumulator)
        if self.boundary[0] != ZERO:
            raise RuntimeError("M232 boundary did not clear")
        self.stage = 4
        work.right_vertex_contraction_clears += 1
        work.observe(self.rails(), transients=(accumulator,), context="RIGHT_CONTRACTION_CLEAR")

    def f_inverse(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> None:
        self.require(owner, generation, program, work)
        if self.stage not in (4, 2):
            raise ValueError("M232 inverse associator order")
        analytic_f_move(self.channel, work, self.rails())
        self.stage -= 1

    def braid_inverse(
        self,
        owner: int,
        generation: int,
        program: PublicProgram,
        exponent: int,
        work: Work,
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 3:
            raise ValueError("M232 inverse braid order")
        if exponent != -topology.braid_exponent:
            raise ValueError("M232 wrong inverse braid exponent")
        analytic_braid_phase(self.channel, exponent, work, self.rails())
        self.stage = 2

    def clear_left(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 1 or self.boundary[0] != ZERO:
            raise ValueError("M232 left vertex inverse dependency")
        for index, channel in enumerate(CHANNELS):
            value = trivalent_value(topology.left_a, topology.left_b, channel)
            work.trivalent_admissibility_evaluations += 1
            self.channel[index] = work.subtract(self.channel[index], value)
        if any(value != ZERO for value in self.channel):
            raise RuntimeError("M232 left message did not clear")
        self.stage = 0
        work.left_vertex_message_clears += 1
        work.observe(self.rails(), context="LEFT_MESSAGE_CLEAR")

    def release(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> int:
        self.require(owner, generation, program, work)
        if self.stage or any(value != ZERO for value in (*self.channel, *self.boundary)):
            raise RuntimeError("M232 release before exact restoration")
        restored = self.generation
        self.live = False
        self.owner = 0
        self.generation = 0
        self.program_token = ""
        self.last_restored_generation = restored
        work.port_releases += 1
        return restored


def make_carrier() -> TrivalentCarrier:
    return TrivalentCarrier([ZERO, ZERO], [ZERO])


def restored(carrier: TrivalentCarrier, generation: int) -> bool:
    return (
        not carrier.live
        and carrier.owner == 0
        and carrier.generation == 0
        and carrier.stage == 0
        and carrier.program_token == ""
        and carrier.channel == [ZERO, ZERO]
        and carrier.boundary == [ZERO]
        and carrier.last_restored_generation == generation
    )


@dataclass
class ClassicalWork:
    field_additions: int = 0
    field_multiplications: int = 0
    field_inversions: int = 0
    maximum_declared_live_field_cells: int = 0
    maximum_declared_live_payload_bits: int = 0
    maximum_declared_live_context: str = ""

    def observe(self, boundary: list[K], values: Sequence[K], context: str) -> None:
        cells = len(boundary) + len(values)
        payload = exact.field_payload_bits(boundary) + exact.field_payload_bits(list(values))
        self.maximum_declared_live_field_cells = max(self.maximum_declared_live_field_cells, cells)
        if payload > self.maximum_declared_live_payload_bits:
            self.maximum_declared_live_payload_bits = payload
            self.maximum_declared_live_context = context


def matched_sparse_factor_graph(
    program: PublicProgram,
    *,
    offdiagonal_zero: bool = False,
    phase_shift: int = 0,
    omit_channel: int | None = None,
) -> dict[str, Any]:
    topology = program.compile()
    boundary = [ZERO]
    work = ClassicalWork()
    if offdiagonal_zero or phase_shift or omit_channel is not None:
        # Controls use the transparent unfused tensor sum.  The accepted path
        # below uses the stronger public-row precontraction.
        for right_channel in CHANNELS:
            if right_channel == omit_channel:
                continue
            right_vertex = trivalent_value(
                right_channel, topology.right_c, topology.right_d
            )
            for fusion_channel in CHANNELS:
                phase = r_symbol(
                    fusion_channel,
                    topology.braid_exponent,
                    phase_shift=phase_shift,
                )
                right_f = f_coefficient(
                    right_channel,
                    fusion_channel,
                    offdiagonal_zero=offdiagonal_zero,
                )
                for left_channel in CHANNELS:
                    if left_channel == omit_channel:
                        continue
                    left_vertex = trivalent_value(
                        topology.left_a, topology.left_b, left_channel
                    )
                    left_f = f_coefficient(
                        fusion_channel,
                        left_channel,
                        offdiagonal_zero=offdiagonal_zero,
                    )
                    product = right_vertex * right_f * phase * left_f * left_vertex
                    boundary[0] = boundary[0] + product
        work.observe(
            boundary,
            (),
            "CLASSICAL_MUTATION_TRANSPARENT_TENSOR_SUM",
        )
    else:
        # Strongest warm compact comparator: rematerialize the three distinct
        # entries of the public symmetric B=F R F block, contract its public
        # right row, and stream the two left-channel terms.  No two-cell
        # channel or retained B row exists.
        a = K.zeta(11)
        a_inverse = a.inverse()
        work.field_inversions += 1
        if topology.braid_exponent == 1:
            r_zero = a + a_inverse * DELTA
            r_two = a
        else:
            r_zero = a_inverse + a * DELTA
            r_two = a_inverse
        work.field_multiplications += 1
        work.field_additions += 1
        del a, a_inverse
        phi_squared = PHI * PHI
        inverse_delta_squared = INVERSE_DELTA * INVERSE_DELTA
        offdiagonal = PHI * (r_zero - r_two) * inverse_delta_squared
        b_zero_zero = (r_zero + phi_squared * r_two) * inverse_delta_squared
        b_two_two = (phi_squared * r_zero + r_two) * inverse_delta_squared
        work.field_multiplications += 8
        work.field_additions += 3
        for left_channel in CHANNELS:
            left_vertex = trivalent_value(
                topology.left_a, topology.left_b, left_channel
            )
            if left_channel == 0:
                row_coefficient = b_zero_zero + offdiagonal
            else:
                row_coefficient = offdiagonal + b_two_two
            contribution = row_coefficient * left_vertex
            updated = boundary[0] + contribution
            work.field_additions += 2
            work.field_multiplications += 1
            work.observe(
                boundary,
                (
                    r_zero,
                    r_two,
                    phi_squared,
                    inverse_delta_squared,
                    offdiagonal,
                    b_zero_zero,
                    b_two_two,
                    left_vertex,
                    row_coefficient,
                    contribution,
                    updated,
                ),
                "CLASSICAL_PRECONTRACTED_PUBLIC_BRAID_ROW_STREAM",
            )
            boundary[0] = updated
    return {
        "boundary_commitment": exact.boundary_commitment(boundary[0]),
        "working_backing_field_cells": 1,
        "maximum_declared_live_field_cells": work.maximum_declared_live_field_cells,
        "maximum_declared_live_payload_bits": work.maximum_declared_live_payload_bits,
        "maximum_declared_live_context": work.maximum_declared_live_context,
        "field_additions": work.field_additions,
        "field_multiplications": work.field_multiplications,
        "field_inversions": work.field_inversions,
        "retained_precontracted_row_field_cells": 0,
        "recurrence": "ANALYTIC_PRECONTRACTED_PUBLIC_B_EQUALS_F_R_F_BOUNDARY_ROW_STREAM",
    }


def transaction(carrier: TrivalentCarrier, program: PublicProgram) -> dict[str, Any]:
    backings = tuple(id(rail) for rail in carrier.rails())
    generation = carrier.last_restored_generation + 1
    owner = 232000 + generation
    work = Work()
    topology = carrier.lease(owner, generation, program, work)
    carrier.produce_left(owner, generation, program, work)
    carrier.f_forward(owner, generation, program, work)
    carrier.braid_forward(owner, generation, program, work)
    carrier.f_forward(owner, generation, program, work)
    carrier.contract_right(owner, generation, program, work)
    boundary = carrier.project(owner, generation, program, work)
    boundary_commitment = exact.boundary_commitment(boundary)
    classical = matched_sparse_factor_graph(program)
    if boundary_commitment != classical["boundary_commitment"]:
        raise RuntimeError("M232 phase and sparse factor graph boundaries differ")
    work.retained_result_values = (boundary,)
    carrier.clear_right(owner, generation, program, work)
    carrier.f_inverse(owner, generation, program, work)
    carrier.braid_inverse(owner, generation, program, -topology.braid_exponent, work)
    carrier.f_inverse(owner, generation, program, work)
    carrier.clear_left(owner, generation, program, work)
    restored_generation = carrier.release(owner, generation, program, work)
    return {
        "family": program.family,
        "program_commitment": program_commitment(program),
        "topology_descriptor_integers": len(topology.integers()),
        "analytic_shared_channels": list(CHANNELS),
        "resident_unresolved_channel_cells": len(carrier.channel),
        "final_boundary_backing_cells": len(carrier.boundary),
        "phase_work_backing_cells": sum(len(rail) for rail in carrier.rails()),
        "same_channel_and_boundary_backings": tuple(id(rail) for rail in carrier.rails()) == backings,
        "boundary_commitment": boundary_commitment,
        "canonical_post_restoration_state_exact": restored(carrier, restored_generation),
        "restoration_generation": restored_generation,
        "baseline_reload_used": False,
        "work": work.as_dict(),
        "matched_sparse_classical": classical,
    }


def normalized_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: case[key]
        for key in (
            "family",
            "program_commitment",
            "topology_descriptor_integers",
            "analytic_shared_channels",
            "resident_unresolved_channel_cells",
            "final_boundary_backing_cells",
            "phase_work_backing_cells",
            "boundary_commitment",
            "canonical_post_restoration_state_exact",
            "restoration_generation",
            "baseline_reload_used",
        )
    }


def controls() -> dict[str, bool]:
    program = PublicProgram(0)
    carrier = make_carrier()
    work = Work()
    owner, generation = 232900, 1
    carrier.lease(owner, generation, program, work)
    wrong_owner = wrong_generation = wrong_program = premature = False
    try:
        carrier.produce_left(owner + 1, generation, program, work)
    except PermissionError:
        wrong_owner = True
    try:
        carrier.produce_left(owner, generation + 1, program, work)
    except PermissionError:
        wrong_generation = True
    try:
        carrier.produce_left(owner, generation, PublicProgram(1), work)
    except ValueError:
        wrong_program = True
    try:
        carrier.project(owner, generation, program, work)
    except PermissionError:
        premature = True
    hidden = False
    try:
        carrier.project_channel()
    except PermissionError:
        hidden = True
    carrier.produce_left(owner, generation, program, work)
    carrier.f_forward(owner, generation, program, work)
    carrier.braid_forward(owner, generation, program, work)
    carrier.f_forward(owner, generation, program, work)
    carrier.contract_right(owner, generation, program, work)
    missing_inverse = carrier.stage == 5 and carrier.boundary[0] != ZERO
    reordered = False
    try:
        carrier.f_inverse(owner, generation, program, work)
    except ValueError:
        reordered = True
    carrier.clear_right(owner, generation, program, work)
    carrier.f_inverse(owner, generation, program, work)
    wrong_inverse = False
    try:
        carrier.braid_inverse(owner, generation, program, 1, work)
    except ValueError:
        wrong_inverse = True
    carrier.braid_inverse(owner, generation, program, -1, work)
    carrier.f_inverse(owner, generation, program, work)
    carrier.clear_left(owner, generation, program, work)
    carrier.release(owner, generation, program, work)
    stale = False
    try:
        carrier.lease(owner + 1, generation, program, work)
    except PermissionError:
        stale = True
    null = False
    try:
        TrivalentCarrier([], [], enabled=False).lease(1, 1, program, Work())
    except ValueError:
        null = True
    wrong_type = False
    try:
        CompiledTopology(1, 1, 1, 1, 1, boundary_type=CHANNEL_PORT_TYPE)
    except TypeError:
        wrong_type = True
    f_squared = True
    for row in CHANNELS:
        for column in CHANNELS:
            value = ZERO
            for middle in CHANNELS:
                value = value + f_coefficient(row, middle) * f_coefficient(middle, column)
            f_squared &= value == (ONE if row == column else ZERO)
    braid_inverse_exact = all(
        r_symbol(channel, 1) * r_symbol(channel, -1) == ONE
        for channel in CHANNELS
    )
    def matrix_multiply(
        left: tuple[tuple[K, K], tuple[K, K]],
        right: tuple[tuple[K, K], tuple[K, K]],
    ) -> tuple[tuple[K, K], tuple[K, K]]:
        return tuple(
            tuple(
                sum(
                    (left[row][middle] * right[middle][column] for middle in range(2)),
                    ZERO,
                )
                for column in range(2)
            )
            for row in range(2)
        )  # type: ignore[return-value]

    f_matrix = tuple(
        tuple(f_coefficient(row, column) for column in CHANNELS)
        for row in CHANNELS
    )
    r_matrix = ((r_symbol(0, 1), ZERO), (ZERO, r_symbol(2, 1)))
    b_matrix = matrix_multiply(matrix_multiply(f_matrix, r_matrix), f_matrix)
    yang_baxter_exact = matrix_multiply(
        matrix_multiply(r_matrix, b_matrix), r_matrix
    ) == matrix_multiply(matrix_multiply(b_matrix, r_matrix), b_matrix)
    perturb_f = perturb_braid = omit_each = families_differ = True
    boundaries = []
    for family in FAMILIES:
        family_program = PublicProgram(family)
        base = matched_sparse_factor_graph(family_program)["boundary_commitment"]
        boundaries.append(base)
        perturb_f &= base != matched_sparse_factor_graph(
            family_program, offdiagonal_zero=True
        )["boundary_commitment"]
        perturb_braid &= base != matched_sparse_factor_graph(
            family_program, phase_shift=2
        )["boundary_commitment"]
        for channel in CHANNELS:
            omit_each &= base != matched_sparse_factor_graph(
                family_program, omit_channel=channel
            )["boundary_commitment"]
    families_differ = boundaries[0] != boundaries[1]
    both_restore = all(
        transaction(make_carrier(), PublicProgram(family))[
            "canonical_post_restoration_state_exact"
        ]
        for family in FAMILIES
    )
    return {
        "analytic_trivalent_admissibility_has_channels0_and2": tuple(
            channel
            for channel in range(LEVEL + 1)
            if admissible(1, 1, channel) and admissible(channel, 1, 1)
        )
        == CHANNELS,
        "invalid_trivalent_signature_rejected": not admissible(1, 1, 1),
        "non_diagonal_f_move_offdiagonal_nonzero": f_coefficient(0, 2) != ZERO and f_coefficient(2, 0) != ZERO,
        "f_move_involution_exact": f_squared,
        "braid_phase_inverse_exact": braid_inverse_exact,
        "yang_baxter_relation_exact": yang_baxter_exact,
        "wrong_owner_rejected": wrong_owner,
        "wrong_generation_rejected": wrong_generation,
        "wrong_public_program_rejected": wrong_program,
        "wrong_port_type_rejected": wrong_type,
        "premature_projection_rejected": premature,
        "shared_channel_projection_rejected": hidden,
        "missing_inverse_detected": missing_inverse,
        "reordered_dependent_inverse_rejected": reordered,
        "wrong_inverse_braid_rejected": wrong_inverse,
        "stale_generation_rejected": stale,
        "null_carrier_rejected": null,
        "f_move_perturbation_changes_boundary_all_families": perturb_f,
        "braid_phase_perturbation_changes_boundary_all_families": perturb_braid,
        "omitting_either_shared_channel_changes_boundary_all_families": omit_each,
        "public_families_have_distinct_boundaries": families_differ,
        "both_public_families_restore": both_restore,
        "public_topology_compilation_reads_final_answer": False,
        "fusion_relation_tables_materialized": False,
        "assignment_expansions_materialized": False,
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: su2_level8_trivalent_shared_channel_fmove.py REFERENCE_JSON")
    here = Path(__file__).resolve().parent
    reference_path = Path(sys.argv[1]).resolve()
    if str(reference_path).startswith(("/dev/shm/", "/run/shm/")):
        raise ValueError("RAM-backed M232 reference forbidden")
    reference = json.loads(reference_path.read_text())
    if reference.get("schema") != "cat_cas.su2_level8_trivalent_shared_channel_reference.v1":
        raise RuntimeError("M232 reference schema mismatch")
    current_controls = controls()
    if current_controls != reference.get("controls"):
        raise RuntimeError("M232 independent control parity failed")
    cases = [transaction(make_carrier(), PublicProgram(family)) for family in FAMILIES]
    if [normalized_case(case) for case in cases] != reference.get("cases"):
        raise RuntimeError("M232 independent case parity failed")
    carrier = make_carrier()
    primary = transaction(carrier, PublicProgram(PRIMARY_FAMILY))
    reuse = transaction(carrier, PublicProgram(REUSE_FAMILY))
    fresh = transaction(make_carrier(), PublicProgram(REUSE_FAMILY))
    reuse_result = {
        "primary": primary,
        "reuse": reuse,
        "fresh_reuse": fresh,
        "fresh_restored_reuse_boundary_agreement": reuse["boundary_commitment"] == fresh["boundary_commitment"],
        "restoration_generation_after_reuse": carrier.last_restored_generation,
    }
    for key in ("primary", "reuse", "fresh_reuse"):
        if normalized_case(reuse_result[key]) != reference["reuse"][key]:
            raise RuntimeError(f"M232 reuse parity failed: {key}")
    for key in ("fresh_restored_reuse_boundary_agreement", "restoration_generation_after_reuse"):
        if reuse_result[key] != reference["reuse"][key]:
            raise RuntimeError(f"M232 top-level reuse parity failed: {key}")
    primary_case = cases[0]
    result = {
        "schema": "cat_cas.su2_level8_trivalent_shared_channel_fmove.v1",
        "result": "PASS_BOUNDED_EXACT_TRIVALENT_SHARED_CHANNEL_F_MOVE_BRAID_CONTRACTION_WITH_IDENTICAL_SMALLER_CLASSICAL_FACTOR_GRAPH",
        "claim": "BOUNDED_EXACT_ANALYTIC_SU2_LEVEL8_TWO_TRIVALENT_FUSION_VERTICES_SHARE_ONE_ACTUAL_UNRESOLVED_TWO_CELL_CHANNEL_RELATION_MESSAGE_TRANSFORMED_BY_A_NONDIAGONAL_F_MOVE_DIAGONAL_BRAID_PHASE_AND_INVERSE_F_MOVE_BEFORE_NATIVE_RIGHT_VERTEX_CONTRACTION_WITHOUT_FUSION_TABLE_OR_ASSIGNMENT_EXPANSION_WITH_FINAL_ONLY_BOUNDARY_EXACT_SAME_BACKING_RESTORATION_AND_REUSE_BUT_THE_IDENTICAL_ONE_BACKING_SPARSE_CLASSICAL_FACTOR_GRAPH_REMAINS_SMALLER",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "QZETA40_SU2_LEVEL8_FOUR_FUNDAMENTAL_CHARGES_CHANNELS0_2_TWO_PUBLIC_BRAID_EXPONENT_FAMILIES_ONE2_CELL_SHARED_CHANNEL_ONE1_CELL_BOUNDARY_PRIMARY0_REUSE1_DIRECT_PROCESS_ONLY",
        "controls": current_controls,
        "cases": cases,
        "reuse": reuse_result,
        "relation_law": {
            "analytic_trivalent_fusion_signatures": True,
            "shared_unresolved_channel_labels": list(CHANNELS),
            "shared_unresolved_channel_cells": 2,
            "two_vertices_share_actual_channel": True,
            "non_diagonal_associator": True,
            "diagonal_braid_phase_in_associated_basis": True,
            "native_right_vertex_contraction": True,
            "fusion_table_materialized": False,
            "assignment_expansion_materialized": False,
            "shared_channel_projected": False,
            "final_boundary_only": True,
            "direct_process_logical_custody_only": True,
        },
        "resource_law": {
            "phase_shared_channel_backing_cells": 2,
            "phase_final_boundary_backing_cells": 1,
            "phase_work_backing_cells": primary_case["phase_work_backing_cells"],
            "matched_sparse_classical_work_backing_cells": primary_case["matched_sparse_classical"]["working_backing_field_cells"],
            "phase_maximum_declared_live_field_cells": primary_case["work"]["maximum_declared_live_field_cells"],
            "matched_classical_maximum_declared_live_field_cells": primary_case["matched_sparse_classical"]["maximum_declared_live_field_cells"],
            "retained_projected_boundary_during_inverse_counted": True,
            "resource_measurement_verification_level": "PACKAGE_SELF_REVIEW",
            "whole_transaction_live_payload_complete": False,
            "shared_imported_exact_kernel_constants_not_counted_as_dynamic_backings": True,
            "excluded_not_zero": "PYTHON_OBJECT_CONTAINER_ALLOCATOR_INTERPRETER_JSON_SERIALIZATION_TIMING_PROCESS_RSS_AND_IMPORTED_PUBLIC_EXACT_FIELD_KERNEL_CONSTANT_STORAGE",
        },
        "matched_sparse_classical": {
            "strongest": "ANALYTIC_PRECONTRACTED_PUBLIC_B_EQUALS_F_R_F_BOUNDARY_ROW_STREAM",
            "boundary_agreement_all_cases": True,
            "classical_work_backings_are_smaller": True,
            "identical_algebraic_contraction": True,
            "phase_specific_reduction": False,
            "computational_advantage": False,
        },
        "separate_reference": {
            "imports_m232_production": reference.get("imports_m232_production"),
            "imports_m231_production": reference.get("imports_m231_production"),
            "imports_m214_production": reference.get("imports_m214_production"),
            "uses_independent_cyclotomic_polynomial_oracle": reference.get("uses_independent_cyclotomic_polynomial_oracle"),
            "case_control_restoration_reuse_parity": True,
        },
        "claim_limits": {
            "general_su2_fusion_category": False,
            "pentagon_or_hexagon_family_verified": False,
            "growing_interface_width": False,
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
            "m214_exact_field_source_sha256": sha256_file(here / "su2_level8_fusion_path_braid_phase_relation.py"),
            "m232_production_sha256": sha256_file(Path(__file__).resolve()),
            "m232_reference_code_sha256": sha256_file(here / "su2_level8_trivalent_shared_channel_fmove_separate_reference.py"),
            "m232_reference_result_sha256": sha256_file(reference_path),
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
