#!/usr/bin/env python3
"""Standalone cyclotomic oracle for the bounded M232 trivalent network."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass

import root_of_unity_su2_level8_fusion_independent_oracle as oracle


sys.set_int_max_str_digits(0)

E = oracle.E
ZERO = oracle.ZERO
ONE = oracle.ONE
LEVEL = 8
CHANNELS = (0, 2)
FAMILIES = (0, 1)
LEFT_PORT_TYPE = 23201
CHANNEL_PORT_TYPE = 23202
RIGHT_PORT_TYPE = 23203
BOUNDARY_PORT_TYPE = 23204
DELTA = E.root(2) + E.root(-2)
INVERSE_DELTA = DELTA.inverse()
PHI = E.root(4) + E.root(-4)


def admissible(a: int, b: int, c: int) -> bool:
    return (
        all(0 <= label <= LEVEL for label in (a, b, c))
        and (a + b + c) % 2 == 0
        and abs(a - b) <= c
        and c <= min(a + b, 2 * LEVEL - a - b)
    )


def vertex(a: int, b: int, c: int) -> E:
    return ONE if admissible(a, b, c) else ZERO


@dataclass(frozen=True)
class Topology:
    family: int
    left_type: int = LEFT_PORT_TYPE
    channel_type: int = CHANNEL_PORT_TYPE
    right_type: int = RIGHT_PORT_TYPE
    boundary_type: int = BOUNDARY_PORT_TYPE

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError("reference family mismatch")
        if (
            self.left_type != LEFT_PORT_TYPE
            or self.channel_type != CHANNEL_PORT_TYPE
            or self.right_type != RIGHT_PORT_TYPE
            or self.boundary_type != BOUNDARY_PORT_TYPE
        ):
            raise TypeError("reference trivalent port type mismatch")
        channels = tuple(
            label
            for label in range(LEVEL + 1)
            if admissible(1, 1, label) and admissible(label, 1, 1)
        )
        if channels != CHANNELS:
            raise RuntimeError("reference analytic channel set changed")

    @property
    def exponent(self) -> int:
        return 1 if self.family == 0 else -1

    def integers(self) -> tuple[int, ...]:
        return (
            1,
            1,
            1,
            1,
            self.exponent,
            self.left_type,
            self.channel_type,
            self.right_type,
            self.boundary_type,
            *CHANNELS,
        )


@dataclass(frozen=True)
class Program:
    family: int

    def compile(self) -> Topology:
        return Topology(self.family)

    def token(self) -> str:
        return f"family:{self.family}|" + ":".join(
            str(value) for value in self.compile().integers()
        )


def program_commitment(program: Program) -> str:
    return hashlib.sha256(program.token().encode("ascii")).hexdigest()


def f_value(row: int, column: int, offdiagonal_zero: bool = False) -> E:
    if row not in CHANNELS or column not in CHANNELS:
        raise ValueError("reference F channel mismatch")
    if row != column:
        return ZERO if offdiagonal_zero else PHI * INVERSE_DELTA
    return INVERSE_DELTA if row == 0 else ZERO - INVERSE_DELTA


def r_value(channel: int, exponent: int, phase_shift: int = 0) -> E:
    if channel not in CHANNELS or exponent not in (-1, 1):
        raise ValueError("reference R symbol mismatch")
    a = E.root(11 + phase_shift)
    a_inverse = a.inverse()
    if exponent == 1:
        return a + a_inverse * DELTA if channel == 0 else a
    return a_inverse + a * DELTA if channel == 0 else a_inverse


def transform_f(values: list[E]) -> list[E]:
    return [
        sum(
            (
                f_value(row, column) * values[index]
                for index, column in enumerate(CHANNELS)
            ),
            ZERO,
        )
        for row in CHANNELS
    ]


def direct_network_boundary(
    program: Program,
    *,
    offdiagonal_zero: bool = False,
    phase_shift: int = 0,
    omit_channel: int | None = None,
) -> E:
    topology = program.compile()
    result = ZERO
    for right_channel in CHANNELS:
        if right_channel == omit_channel:
            continue
        right_vertex = vertex(right_channel, 1, 1)
        for braid_channel in CHANNELS:
            right_f = f_value(
                right_channel, braid_channel, offdiagonal_zero
            )
            phase = r_value(braid_channel, topology.exponent, phase_shift)
            for left_channel in CHANNELS:
                if left_channel == omit_channel:
                    continue
                left_vertex = vertex(1, 1, left_channel)
                left_f = f_value(
                    braid_channel, left_channel, offdiagonal_zero
                )
                result = (
                    result
                    + right_vertex
                    * right_f
                    * phase
                    * left_f
                    * left_vertex
                )
    return result


@dataclass
class Port:
    channel: list[E]
    boundary: list[E]
    enabled: bool = True
    live: bool = False
    owner: int = 0
    generation: int = 0
    last_restored_generation: int = 0
    stage: int = 0
    program_hash: str = ""

    def rails(self) -> tuple[list[E], list[E]]:
        return self.channel, self.boundary

    def lease(self, owner: int, generation: int, program: Program) -> Topology:
        if self.live:
            raise RuntimeError("reference port already live")
        if not self.enabled or len(self.channel) != 2 or len(self.boundary) != 1:
            raise ValueError("reference null or wrong-width port")
        if any(value != ZERO for value in (*self.channel, *self.boundary)):
            raise ValueError("reference dirty port")
        if owner <= 0 or generation != self.last_restored_generation + 1:
            raise PermissionError("reference invalid or stale generation")
        topology = program.compile()
        self.live = True
        self.owner = owner
        self.generation = generation
        self.stage = 0
        self.program_hash = program_commitment(program)
        return topology

    def require(self, owner: int, generation: int, program: Program) -> Topology:
        if not self.live:
            raise RuntimeError("reference port not live")
        if owner != self.owner:
            raise PermissionError("reference owner mismatch")
        if generation != self.generation:
            raise PermissionError("reference generation mismatch")
        if program_commitment(program) != self.program_hash:
            raise ValueError("reference program mismatch")
        return program.compile()

    def produce_left(self, owner: int, generation: int, program: Program) -> None:
        self.require(owner, generation, program)
        if self.stage != 0 or any(value != ZERO for value in self.channel):
            raise ValueError("reference left production order")
        for index, label in enumerate(CHANNELS):
            self.channel[index] = self.channel[index] + vertex(1, 1, label)
        self.stage = 1

    def f_forward(self, owner: int, generation: int, program: Program) -> None:
        self.require(owner, generation, program)
        if self.stage not in (1, 3):
            raise ValueError("reference forward F order")
        transformed = transform_f(self.channel)
        self.channel[:] = transformed
        self.stage += 1

    def braid_forward(self, owner: int, generation: int, program: Program) -> None:
        topology = self.require(owner, generation, program)
        if self.stage != 2:
            raise ValueError("reference forward R order")
        for index, label in enumerate(CHANNELS):
            self.channel[index] = r_value(label, topology.exponent) * self.channel[index]
        self.stage = 3

    def contract_right(self, owner: int, generation: int, program: Program) -> None:
        self.require(owner, generation, program)
        if self.stage != 4 or self.boundary != [ZERO]:
            raise ValueError("reference right contraction order")
        self.boundary[0] = sum(
            (
                vertex(label, 1, 1) * self.channel[index]
                for index, label in enumerate(CHANNELS)
            ),
            ZERO,
        )
        self.stage = 5

    def project(self, owner: int, generation: int, program: Program) -> E:
        self.require(owner, generation, program)
        if self.stage != 5:
            raise PermissionError("reference nonfinal projection")
        return self.boundary[0]

    def project_channel(self) -> None:
        raise PermissionError("reference shared channel hidden")

    def clear_right(self, owner: int, generation: int, program: Program) -> None:
        self.require(owner, generation, program)
        if self.stage != 5:
            raise ValueError("reference right inverse order")
        value = sum(
            (
                vertex(label, 1, 1) * self.channel[index]
                for index, label in enumerate(CHANNELS)
            ),
            ZERO,
        )
        self.boundary[0] = self.boundary[0] - value
        if self.boundary != [ZERO]:
            raise RuntimeError("reference boundary did not clear")
        self.stage = 4

    def f_inverse(self, owner: int, generation: int, program: Program) -> None:
        self.require(owner, generation, program)
        if self.stage not in (4, 2):
            raise ValueError("reference inverse F order")
        transformed = transform_f(self.channel)
        self.channel[:] = transformed
        self.stage -= 1

    def braid_inverse(
        self,
        owner: int,
        generation: int,
        program: Program,
        exponent: int,
    ) -> None:
        topology = self.require(owner, generation, program)
        if self.stage != 3:
            raise ValueError("reference inverse R order")
        if exponent != -topology.exponent:
            raise ValueError("reference wrong inverse R exponent")
        for index, label in enumerate(CHANNELS):
            self.channel[index] = r_value(label, exponent) * self.channel[index]
        self.stage = 2

    def clear_left(self, owner: int, generation: int, program: Program) -> None:
        self.require(owner, generation, program)
        if self.stage != 1 or self.boundary != [ZERO]:
            raise ValueError("reference left inverse dependency")
        for index, label in enumerate(CHANNELS):
            self.channel[index] = self.channel[index] - vertex(1, 1, label)
        if self.channel != [ZERO, ZERO]:
            raise RuntimeError("reference left message did not clear")
        self.stage = 0

    def release(self, owner: int, generation: int, program: Program) -> int:
        self.require(owner, generation, program)
        if self.stage or self.channel != [ZERO, ZERO] or self.boundary != [ZERO]:
            raise RuntimeError("reference release before restoration")
        restored = self.generation
        self.live = False
        self.owner = 0
        self.generation = 0
        self.program_hash = ""
        self.last_restored_generation = restored
        return restored


def make_port() -> Port:
    return Port([ZERO, ZERO], [ZERO])


def restored(port: Port, generation: int) -> bool:
    return (
        not port.live
        and port.owner == 0
        and port.generation == 0
        and port.stage == 0
        and port.program_hash == ""
        and port.channel == [ZERO, ZERO]
        and port.boundary == [ZERO]
        and port.last_restored_generation == generation
    )


def execute(port: Port, program: Program) -> dict[str, object]:
    backings = tuple(id(rail) for rail in port.rails())
    generation = port.last_restored_generation + 1
    owner = 232000 + generation
    topology = port.lease(owner, generation, program)
    port.produce_left(owner, generation, program)
    port.f_forward(owner, generation, program)
    port.braid_forward(owner, generation, program)
    port.f_forward(owner, generation, program)
    port.contract_right(owner, generation, program)
    boundary = port.project(owner, generation, program)
    if boundary != direct_network_boundary(program):
        raise RuntimeError("reference carrier differs from expanded tensor network")
    commitment = oracle.boundary_commitment(boundary)
    port.clear_right(owner, generation, program)
    port.f_inverse(owner, generation, program)
    port.braid_inverse(owner, generation, program, -topology.exponent)
    port.f_inverse(owner, generation, program)
    port.clear_left(owner, generation, program)
    restored_generation = port.release(owner, generation, program)
    return {
        "family": program.family,
        "program_commitment": program_commitment(program),
        "topology_descriptor_integers": len(topology.integers()),
        "analytic_shared_channels": list(CHANNELS),
        "resident_unresolved_channel_cells": len(port.channel),
        "final_boundary_backing_cells": len(port.boundary),
        "phase_work_backing_cells": sum(len(rail) for rail in port.rails()),
        "boundary_commitment": commitment,
        "canonical_post_restoration_state_exact": restored(port, restored_generation)
        and tuple(id(rail) for rail in port.rails()) == backings,
        "restoration_generation": restored_generation,
        "baseline_reload_used": False,
    }


def controls() -> dict[str, bool]:
    program = Program(0)
    port = make_port()
    owner, generation = 232900, 1
    port.lease(owner, generation, program)
    wrong_owner = wrong_generation = wrong_program = premature = False
    try:
        port.produce_left(owner + 1, generation, program)
    except PermissionError:
        wrong_owner = True
    try:
        port.produce_left(owner, generation + 1, program)
    except PermissionError:
        wrong_generation = True
    try:
        port.produce_left(owner, generation, Program(1))
    except ValueError:
        wrong_program = True
    try:
        port.project(owner, generation, program)
    except PermissionError:
        premature = True
    hidden = False
    try:
        port.project_channel()
    except PermissionError:
        hidden = True
    port.produce_left(owner, generation, program)
    port.f_forward(owner, generation, program)
    port.braid_forward(owner, generation, program)
    port.f_forward(owner, generation, program)
    port.contract_right(owner, generation, program)
    missing_inverse = port.stage == 5 and port.boundary != [ZERO]
    reordered = False
    try:
        port.f_inverse(owner, generation, program)
    except ValueError:
        reordered = True
    port.clear_right(owner, generation, program)
    port.f_inverse(owner, generation, program)
    wrong_inverse = False
    try:
        port.braid_inverse(owner, generation, program, 1)
    except ValueError:
        wrong_inverse = True
    port.braid_inverse(owner, generation, program, -1)
    port.f_inverse(owner, generation, program)
    port.clear_left(owner, generation, program)
    port.release(owner, generation, program)
    stale = False
    try:
        port.lease(owner + 1, generation, program)
    except PermissionError:
        stale = True
    null = False
    try:
        Port([], [], enabled=False).lease(1, 1, program)
    except ValueError:
        null = True
    wrong_type = False
    try:
        Topology(0, boundary_type=CHANNEL_PORT_TYPE)
    except TypeError:
        wrong_type = True
    f_squared = all(
        sum(
            (
                f_value(row, middle) * f_value(middle, column)
                for middle in CHANNELS
            ),
            ZERO,
        )
        == (ONE if row == column else ZERO)
        for row in CHANNELS
        for column in CHANNELS
    )
    braid_inverse = all(
        r_value(channel, 1) * r_value(channel, -1) == ONE
        for channel in CHANNELS
    )
    def matrix_multiply(left: tuple[tuple[E, E], tuple[E, E]], right: tuple[tuple[E, E], tuple[E, E]]) -> tuple[tuple[E, E], tuple[E, E]]:
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
        tuple(f_value(row, column) for column in CHANNELS)
        for row in CHANNELS
    )
    r_matrix = ((r_value(0, 1), ZERO), (ZERO, r_value(2, 1)))
    b_matrix = matrix_multiply(matrix_multiply(f_matrix, r_matrix), f_matrix)
    yang_baxter_exact = matrix_multiply(
        matrix_multiply(r_matrix, b_matrix), r_matrix
    ) == matrix_multiply(matrix_multiply(b_matrix, r_matrix), b_matrix)
    perturb_f = perturb_r = omit = True
    boundaries = []
    for family in FAMILIES:
        family_program = Program(family)
        base = oracle.boundary_commitment(direct_network_boundary(family_program))
        boundaries.append(base)
        perturb_f &= base != oracle.boundary_commitment(
            direct_network_boundary(family_program, offdiagonal_zero=True)
        )
        perturb_r &= base != oracle.boundary_commitment(
            direct_network_boundary(family_program, phase_shift=2)
        )
        for channel in CHANNELS:
            omit &= base != oracle.boundary_commitment(
                direct_network_boundary(family_program, omit_channel=channel)
            )
    return {
        "analytic_trivalent_admissibility_has_channels0_and2": tuple(
            label
            for label in range(LEVEL + 1)
            if admissible(1, 1, label) and admissible(label, 1, 1)
        )
        == CHANNELS,
        "invalid_trivalent_signature_rejected": not admissible(1, 1, 1),
        "non_diagonal_f_move_offdiagonal_nonzero": f_value(0, 2) != ZERO and f_value(2, 0) != ZERO,
        "f_move_involution_exact": f_squared,
        "braid_phase_inverse_exact": braid_inverse,
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
        "braid_phase_perturbation_changes_boundary_all_families": perturb_r,
        "omitting_either_shared_channel_changes_boundary_all_families": omit,
        "public_families_have_distinct_boundaries": boundaries[0] != boundaries[1],
        "both_public_families_restore": all(
            bool(execute(make_port(), Program(family))["canonical_post_restoration_state_exact"])
            for family in FAMILIES
        ),
        "public_topology_compilation_reads_final_answer": False,
        "fusion_relation_tables_materialized": False,
        "assignment_expansions_materialized": False,
    }


def main() -> None:
    cases = [execute(make_port(), Program(family)) for family in FAMILIES]
    port = make_port()
    primary = execute(port, Program(0))
    reuse = execute(port, Program(1))
    fresh = execute(make_port(), Program(1))
    result = {
        "schema": "cat_cas.su2_level8_trivalent_shared_channel_reference.v1",
        "controls": controls(),
        "cases": cases,
        "reuse": {
            "primary": primary,
            "reuse": reuse,
            "fresh_reuse": fresh,
            "fresh_restored_reuse_boundary_agreement": reuse["boundary_commitment"] == fresh["boundary_commitment"],
            "restoration_generation_after_reuse": port.last_restored_generation,
        },
        "imports_m232_production": False,
        "imports_m231_production": False,
        "imports_m214_production": False,
        "uses_independent_cyclotomic_polynomial_oracle": True,
    }
    positive = {
        key: value
        for key, value in result["controls"].items()
        if key
        not in {
            "public_topology_compilation_reads_final_answer",
            "fusion_relation_tables_materialized",
            "assignment_expansions_materialized",
        }
    }
    if not all(positive.values()) or any(
        result["controls"][key]
        for key in (
            "public_topology_compilation_reads_final_answer",
            "fusion_relation_tables_materialized",
            "assignment_expansions_materialized",
        )
    ):
        raise RuntimeError("M232 reference controls failed")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
