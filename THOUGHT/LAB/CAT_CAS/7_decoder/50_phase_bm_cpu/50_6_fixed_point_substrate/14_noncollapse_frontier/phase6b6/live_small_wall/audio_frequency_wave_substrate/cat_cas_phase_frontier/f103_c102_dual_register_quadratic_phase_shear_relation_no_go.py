#!/usr/bin/env python3
"""Reversible cross-character phase shear on the M157 group-algebra carrier.

M157 proved that every linear F103[C102] relation operation descends through
evaluation at t=5.  This successor changes the machine: a second unresolved
register receives a coefficientwise quadratic shear from the first.  In the
character chart this is a convolution across all 102 sectors, so a single
character is no longer a sufficient quotient.  The experiment measures the
result without treating the full 102-coordinate classical recurrence as an
advantage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

import f103_unresolved_c102_group_algebra_superposition_relation_no_go as base


INTERFACES = (5, 7)
DEPTHS = (1, 2, 4)
FAMILIES = base.FAMILIES
REGISTERS = 2
CLAIM = (
    "BOUNDED_EXACT_F103_C102_DUAL_REGISTER_COEFFICIENTWISE_QUADRATIC_"
    "PHASE_SHEAR_OPEN_RELATION_COMPOSITION_BREAKS_THE_SINGLE_CHARACTER_"
    "EVALUATION_QUOTIENT_COUPLES_ALL102_CHARACTER_SECTORS_AND_HAS_"
    "FULL_RANK102_ONE_SHEAR_QUADRATIC_OBSERVABLE_ON_C5_C7_WITH_EXACT_"
    "RESTORATION_AND_REUSE_BUT_THE_ACCEPTED_CARRIER_AND_STRONGEST_"
    "MATCHED_CLASSICAL_RECURRENCE_BOTH_RETAIN_THE_IDENTICAL204N2_FIELD_"
    "COORDINATES"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class Program:
    interface: int
    depth: int
    family: str
    owner: int
    observation_left: int
    observation_right: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "CAT_CAS_F103_C102_DUAL_REGISTER_QUADRATIC_SHEAR_RELATION_PROGRAM_V1",
            "interface": self.interface,
            "depth": self.depth,
            "family": self.family,
            "owner": self.owner,
            "node_count": base.NODE_COUNT,
            "register_count": REGISTERS,
            "port_type": f"F103_C102_DUAL_REGISTER_C{self.interface}_TO_C{self.interface}",
            "topology": "PUBLIC_ROTATING_CONTROL_HUB8",
            "linear_composition": "RANK1_F103_C102_GROUP_ALGEBRA_LEFT_ACTION",
            "linear_intersection": "NATIVE_C102_SUPPORT_SHIFT",
            "cross_character_operation": "REVERSIBLE_COEFFICIENTWISE_QUADRATIC_SHEAR_A_TO_B",
            "projection": "FINAL_REGISTER_B_BOUNDARY_EVALUATION_T_TO_5",
            "observation": [self.observation_left, self.observation_right],
        }

    def fingerprint(self) -> str:
        return digest_json(self.descriptor())

    def base_program(self) -> base.Program:
        return base.compile_program(self.interface, self.depth, self.family)


def compile_program(interface: int, depth: int, family: str) -> Program:
    if interface not in INTERFACES:
        fail("interface outside declared set")
    if not isinstance(depth, int) or not 1 <= depth <= max(DEPTHS):
        fail("depth outside declared ceiling")
    if family not in FAMILIES:
        fail("family outside declared set")
    code = base.family_code(family)
    return Program(
        interface,
        depth,
        family,
        (0xC1580000 + 257 * interface + 131 * depth + code) & 0xFFFFFFFF,
        (13 * depth + 3 * code + interface) % interface,
        (19 * depth + 7 * code + 2 * interface) % interface,
    )


def register_b_seed(interface: int, family: str) -> np.ndarray:
    result = np.zeros(
        (base.NODE_COUNT, interface, interface, base.GROUP_ORDER),
        dtype=np.uint8,
    )
    code = base.family_code(family)
    for node in range(base.NODE_COUNT):
        for row in range(interface):
            for column in range(interface):
                exponent = (
                    base.seed_exponent(node, family, row, column)
                    + 29
                    + 7 * node
                    + 3 * row
                    + 5 * column
                    + code
                ) % base.GROUP_ORDER
                result[node, row, column, exponent] = 1
    return result


def seed_registers(interface: int, family: str) -> np.ndarray:
    return np.stack(
        (
            base.seed_coefficients(interface, family),
            register_b_seed(interface, family),
        ),
        axis=0,
    )


def shear_gamma(index: int, hub: int, peer: int, family: str, mutation: int = 0) -> int:
    exponent = (
        31
        + 5 * index
        + 7 * hub
        + 11 * peer
        + base.family_code(family)
        + mutation
    ) % base.GROUP_ORDER
    return base.POWERS[exponent]


def shear_multipliers(
    index: int, hub: int, peer: int, family: str, mutation: int = 0
) -> np.ndarray:
    code = base.family_code(family) + mutation
    return np.array(
        [
            base.POWERS[
                (
                    13
                    + code
                    + 3 * index
                    + 5 * hub
                    + 7 * peer
                    + 2 * exponent * exponent
                    + 3 * exponent
                )
                % base.GROUP_ORDER
            ]
            for exponent in range(base.GROUP_ORDER)
        ],
        dtype=np.uint8,
    )


@dataclass
class ShearStats:
    linear: base.WorkStats = field(default_factory=base.WorkStats)
    coefficient_squares: int = 0
    coefficient_multiplications: int = 0
    coefficient_additions: int = 0
    shears: int = 0
    maximum_shear_buffer_cells: int = 0

    def descriptor(self) -> dict[str, Any]:
        return {
            "linear": self.linear.descriptor(),
            "coefficient_squares": self.coefficient_squares,
            "coefficient_multiplications": self.coefficient_multiplications,
            "coefficient_additions": self.coefficient_additions,
            "shears": self.shears,
            "maximum_shear_buffer_cells": self.maximum_shear_buffer_cells,
        }


def apply_shear(
    register_a: np.ndarray,
    register_b: np.ndarray,
    index: int,
    hub: int,
    peer: int,
    family: str,
    *,
    inverse: bool,
    mutation: int,
    stats: ShearStats | None,
) -> np.ndarray:
    interface = register_a.shape[0]
    gamma = shear_gamma(index, hub, peer, family, mutation)
    if inverse:
        gamma = -gamma
    multipliers = shear_multipliers(index, hub, peer, family, mutation).astype(
        np.int64
    )
    a = register_a.astype(np.int64)
    correction = gamma * multipliers[None, None, :] * a * a
    result = np.asarray(
        (register_b.astype(np.int64) + correction) % base.FIELD,
        dtype=np.uint8,
    )
    if stats is not None:
        cells = interface * interface * base.GROUP_ORDER
        stats.coefficient_squares += cells
        stats.coefficient_multiplications += 2 * cells
        stats.coefficient_additions += cells
        stats.shears += 1
        stats.maximum_shear_buffer_cells = max(
            stats.maximum_shear_buffer_cells, cells
        )
    return result


def raw_forward(
    registers: np.ndarray,
    program: Program,
    stats: ShearStats | None = None,
    *,
    topology_mutation: int = 0,
    port_enabled: bool = True,
) -> None:
    if not port_enabled:
        return
    topology = program.base_program()
    interface = program.interface
    for index in range(program.depth):
        hub = base.hub_index(index, program.family, topology_mutation)
        for peer in base.peer_order(hub, program.family):
            amount = base.rotation_shift(interface, peer, index, program.family)
            current_a = base.rotate_coefficients(
                registers[0, peer], amount, None if stats is None else stats.linear
            )
            current_b = base.rotate_coefficients(
                registers[1, peer], amount, None if stats is None else stats.linear
            )
            left, right, coupling = base.composition_exponents(
                interface, hub, peer, index, program.family, topology_mutation
            )
            current_a = base.compose_coefficients(
                current_a, left, right, coupling, None if stats is None else stats.linear
            )
            current_b = base.compose_coefficients(
                current_b, left, right, coupling, None if stats is None else stats.linear
            )
            current_b = apply_shear(
                current_a,
                current_b,
                index,
                hub,
                peer,
                program.family,
                inverse=False,
                mutation=topology_mutation,
                stats=stats,
            )
            current_a = base.intersect_coefficients(
                current_a,
                hub,
                peer,
                index,
                program.family,
                inverse=False,
                mutation=topology_mutation,
                stats=None if stats is None else stats.linear,
            )
            current_b = base.intersect_coefficients(
                current_b,
                hub,
                peer,
                index,
                program.family,
                inverse=False,
                mutation=topology_mutation,
                stats=None if stats is None else stats.linear,
            )
            np.copyto(registers[0, peer], current_a)
            np.copyto(registers[1, peer], current_b)
            if stats is not None:
                stats.linear.consumers += 1


def raw_inverse(
    registers: np.ndarray,
    program: Program,
    stats: ShearStats | None = None,
    *,
    inverse_order: str = "INTERSECT_SHEAR_COMPOSE",
    topology_mutation: int = 0,
) -> None:
    interface = program.interface
    for index in reversed(range(program.depth)):
        hub = base.hub_index(index, program.family, topology_mutation)
        for peer in reversed(base.peer_order(hub, program.family)):
            left, right, coupling = base.composition_exponents(
                interface, hub, peer, index, program.family, topology_mutation
            )
            current_a = registers[0, peer]
            current_b = registers[1, peer]
            if inverse_order == "INTERSECT_SHEAR_COMPOSE":
                current_a = base.intersect_coefficients(
                    current_a,
                    hub,
                    peer,
                    index,
                    program.family,
                    inverse=True,
                    mutation=topology_mutation,
                    stats=None if stats is None else stats.linear,
                )
                current_b = base.intersect_coefficients(
                    current_b,
                    hub,
                    peer,
                    index,
                    program.family,
                    inverse=True,
                    mutation=topology_mutation,
                    stats=None if stats is None else stats.linear,
                )
                current_b = apply_shear(
                    current_a,
                    current_b,
                    index,
                    hub,
                    peer,
                    program.family,
                    inverse=True,
                    mutation=topology_mutation,
                    stats=stats,
                )
                current_a = base.inverse_compose_coefficients(
                    current_a, left, right, coupling, None if stats is None else stats.linear
                )
                current_b = base.inverse_compose_coefficients(
                    current_b, left, right, coupling, None if stats is None else stats.linear
                )
            elif inverse_order == "INTERSECT_COMPOSE_SHEAR":
                current_a = base.intersect_coefficients(
                    current_a,
                    hub,
                    peer,
                    index,
                    program.family,
                    inverse=True,
                    mutation=topology_mutation,
                    stats=None,
                )
                current_b = base.intersect_coefficients(
                    current_b,
                    hub,
                    peer,
                    index,
                    program.family,
                    inverse=True,
                    mutation=topology_mutation,
                    stats=None,
                )
                current_a = base.inverse_compose_coefficients(
                    current_a, left, right, coupling, None
                )
                current_b = base.inverse_compose_coefficients(
                    current_b, left, right, coupling, None
                )
                current_b = apply_shear(
                    current_a,
                    current_b,
                    index,
                    hub,
                    peer,
                    program.family,
                    inverse=True,
                    mutation=topology_mutation,
                    stats=None,
                )
            else:
                fail("unknown inverse order")
            amount = base.rotation_shift(interface, peer, index, program.family)
            current_a = base.rotate_coefficients(
                current_a, -amount, None if stats is None else stats.linear
            )
            current_b = base.rotate_coefficients(
                current_b, -amount, None if stats is None else stats.linear
            )
            np.copyto(registers[0, peer], current_a)
            np.copyto(registers[1, peer], current_b)
            if stats is not None:
                stats.linear.consumers += 1


def registers_commitment(registers: np.ndarray) -> str:
    return hashlib.sha256(registers.tobytes()).hexdigest()


def boundary(registers: np.ndarray, program: Program) -> tuple[int, ...]:
    interface = program.interface
    return tuple(
        base.evaluate_polynomial(
            registers[
                1,
                node,
                (program.observation_left + node) % interface,
                (program.observation_right + 2 * node) % interface,
            ]
        )
        for node in range(base.NODE_COUNT)
    )


def dft_matrices() -> tuple[np.ndarray, np.ndarray]:
    forward = np.array(
        [
            [
                base.POWERS[(character * exponent) % base.GROUP_ORDER]
                for exponent in range(base.GROUP_ORDER)
            ]
            for character in range(base.GROUP_ORDER)
        ],
        dtype=np.int64,
    )
    inverse_order = pow(base.GROUP_ORDER, -1, base.FIELD)
    inverse = np.array(
        [
            [
                inverse_order
                * base.POWERS[(-exponent * character) % base.GROUP_ORDER]
                % base.FIELD
                for character in range(base.GROUP_ORDER)
            ]
            for exponent in range(base.GROUP_ORDER)
        ],
        dtype=np.int64,
    )
    return forward, inverse


DFT, INVERSE_DFT = dft_matrices()


def coefficient_to_character(registers: np.ndarray) -> np.ndarray:
    flat = registers.astype(np.int64).reshape(-1, base.GROUP_ORDER)
    transformed = flat @ DFT.T % base.FIELD
    return np.asarray(transformed, dtype=np.uint8).reshape(registers.shape)


def character_to_coefficient(registers: np.ndarray) -> np.ndarray:
    flat = registers.astype(np.int64).reshape(-1, base.GROUP_ORDER)
    transformed = flat @ INVERSE_DFT.T % base.FIELD
    return np.asarray(transformed, dtype=np.uint8).reshape(registers.shape)


def character_forward(program: Program) -> np.ndarray:
    """Separate full-character classical recurrence.

    Linear group-algebra operations are executed characterwise.  The nonlinear
    shear explicitly returns to coefficients and then to characters, making
    all-sector coupling visible rather than calling the production shear.
    """

    registers = coefficient_to_character(
        seed_registers(program.interface, program.family)
    )
    interface = program.interface
    for index in range(program.depth):
        hub = base.hub_index(index, program.family)
        for peer in base.peer_order(hub, program.family):
            amount = base.rotation_shift(interface, peer, index, program.family)
            for register in range(REGISTERS):
                registers[register, peer] = np.roll(
                    registers[register, peer],
                    (amount, amount),
                    axis=(0, 1),
                )
            left, right, coupling = base.composition_exponents(
                interface, hub, peer, index, program.family
            )
            for register in range(REGISTERS):
                current = registers[register, peer].astype(np.int64)
                result = np.empty_like(current)
                for character in range(base.GROUP_ORDER):
                    left_values = np.array(
                        [
                            base.POWERS[(character * value) % base.GROUP_ORDER]
                            for value in left
                        ],
                        dtype=np.int64,
                    )
                    right_values = np.array(
                        [
                            base.POWERS[(character * value) % base.GROUP_ORDER]
                            for value in right
                        ],
                        dtype=np.int64,
                    )
                    moment = right_values @ current[:, :, character] % base.FIELD
                    result[:, :, character] = (
                        current[:, :, character]
                        + coupling * left_values[:, None] * moment[None, :]
                    ) % base.FIELD
                registers[register, peer] = np.asarray(result, dtype=np.uint8)
            coefficients = character_to_coefficient(registers[:, peer])
            gamma = shear_gamma(index, hub, peer, program.family)
            multipliers = shear_multipliers(
                index, hub, peer, program.family
            ).astype(np.int64)
            coefficients[1] = np.asarray(
                (
                    coefficients[1].astype(np.int64)
                    + gamma
                    * multipliers[None, None, :]
                    * coefficients[0].astype(np.int64) ** 2
                )
                % base.FIELD,
                dtype=np.uint8,
            )
            registers[:, peer] = coefficient_to_character(coefficients)
            for register in range(REGISTERS):
                current = registers[register, peer].astype(np.int64)
                for row in range(interface):
                    for column in range(interface):
                        exponent = base.intersection_exponent(
                            hub,
                            peer,
                            index,
                            program.family,
                            row,
                            column,
                        )
                        current[row, column] = (
                            current[row, column]
                            * np.array(
                                [
                                    base.POWERS[
                                        (character * exponent)
                                        % base.GROUP_ORDER
                                    ]
                                    for character in range(base.GROUP_ORDER)
                                ],
                                dtype=np.int64,
                            )
                        ) % base.FIELD
                registers[register, peer] = np.asarray(current, dtype=np.uint8)
    return registers


def character_boundary(registers: np.ndarray, program: Program) -> tuple[int, ...]:
    interface = program.interface
    return tuple(
        int(
            registers[
                1,
                node,
                (program.observation_left + node) % interface,
                (program.observation_right + 2 * node) % interface,
                1,
            ]
        )
        for node in range(base.NODE_COUNT)
    )


@dataclass
class Carrier:
    interface: int
    family: str
    port_type: str
    registers: np.ndarray
    generation: int = 0
    restoration_generation: int = 0
    state: str = "RESTORED"
    active_owner: int | None = None
    active_program: str | None = None
    stats: ShearStats = field(default_factory=ShearStats)

    @classmethod
    def fresh(cls, interface: int, family: str) -> "Carrier":
        return cls(
            interface,
            family,
            f"F103_C102_DUAL_REGISTER_C{interface}_TO_C{interface}",
            seed_registers(interface, family),
        )

    def backing_id(self) -> int:
        return int(self.registers.__array_interface__["data"][0])


def payload_commitment(carrier: Carrier) -> str:
    return digest_json(
        {
            "interface": carrier.interface,
            "family": carrier.family,
            "port_type": carrier.port_type,
            "register_commitment": registers_commitment(carrier.registers),
        }
    )


def begin(carrier: Carrier | None, program: Program, owner: int) -> None:
    if carrier is None:
        fail("null carrier")
    if carrier.state != "RESTORED":
        fail("carrier not restored")
    if carrier.interface != program.interface or carrier.family != program.family:
        fail("carrier/program type mismatch")
    if carrier.port_type != program.descriptor()["port_type"]:
        fail("typed port mismatch")
    if owner != program.owner:
        fail("owner mismatch")
    carrier.state = "FORWARD_ACTIVE"
    carrier.active_owner = owner
    carrier.active_program = program.fingerprint()
    carrier.stats = ShearStats()


def forward(carrier: Carrier, program: Program, owner: int) -> None:
    if carrier.state != "FORWARD_ACTIVE" or carrier.active_owner != owner:
        fail("forward custody mismatch")
    if carrier.active_program != program.fingerprint():
        fail("forward program mismatch")
    raw_forward(carrier.registers, program, carrier.stats)
    carrier.state = "FORWARD_COMPLETE"


def project(carrier: Carrier, program: Program, owner: int) -> tuple[int, ...]:
    if carrier.state != "FORWARD_COMPLETE":
        fail("projection outside final-boundary stage")
    if carrier.active_owner != owner or carrier.active_program != program.fingerprint():
        fail("projection custody mismatch")
    return boundary(carrier.registers, program)


def project_resident(_carrier: Carrier, _register: int, _node: int) -> None:
    fail("resident group-algebra register projection forbidden")


def inverse(carrier: Carrier, program: Program, owner: int) -> None:
    if carrier.state != "FORWARD_COMPLETE":
        fail("inverse outside forward-complete stage")
    if carrier.active_owner != owner or carrier.active_program != program.fingerprint():
        fail("inverse custody mismatch")
    raw_inverse(carrier.registers, program, carrier.stats)
    carrier.state = "RESTORED"
    carrier.active_owner = None
    carrier.active_program = None
    carrier.generation += 1
    carrier.restoration_generation = carrier.generation


def transaction(carrier: Carrier, program: Program) -> dict[str, Any]:
    before = payload_commitment(carrier)
    backing = carrier.backing_id()
    generation = carrier.generation
    begin(carrier, program, program.owner)
    forward(carrier, program, program.owner)
    forward_commitment = registers_commitment(carrier.registers)
    result = project(carrier, program, program.owner)
    forward_stats = carrier.stats.descriptor()
    inverse(carrier, program, program.owner)
    if payload_commitment(carrier) != before:
        fail("dual-register inverse failed exact restoration")
    if carrier.backing_id() != backing:
        fail("dual-register carrier backing changed")
    if carrier.generation != generation + 1:
        fail("restoration generation did not advance once")
    return {
        "_boundary": result,
        "program_fingerprint": program.fingerprint(),
        "boundary_commitment": digest_json(list(result)),
        "forward_register_commitment": forward_commitment,
        "forward_stats": forward_stats,
        "transaction_stats": carrier.stats.descriptor(),
        "payload_restored_exactly": True,
        "same_backing_restored": True,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
        "retained_inverse_history_cells": 0,
        "retained_restoration_baseline_cells": 0,
    }


def single_character_collision() -> dict[str, Any]:
    program = compile_program(5, 1, "PRIMARY")
    index = 0
    hub = base.hub_index(index, program.family)
    peer = base.peer_order(hub, program.family)[0]
    first_a = np.zeros(base.GROUP_ORDER, dtype=np.uint8)
    first_a[0] = 1
    second_a = first_a.copy()
    second_a[0] = (int(second_a[0]) - base.GENERATOR) % base.FIELD
    second_a[1] = (int(second_a[1]) + 1) % base.FIELD
    first_b = np.zeros(base.GROUP_ORDER, dtype=np.uint8)
    second_b = first_b.copy()
    first_out = apply_shear(
        first_a.reshape(1, 1, -1),
        first_b.reshape(1, 1, -1),
        index,
        hub,
        peer,
        program.family,
        inverse=False,
        mutation=0,
        stats=None,
    )[0, 0]
    second_out = apply_shear(
        second_a.reshape(1, 1, -1),
        second_b.reshape(1, 1, -1),
        index,
        hub,
        peer,
        program.family,
        inverse=False,
        mutation=0,
        stats=None,
    )[0, 0]
    before_equal = base.evaluate_polynomial(first_a) == base.evaluate_polynomial(
        second_a
    ) and base.evaluate_polynomial(first_b) == base.evaluate_polynomial(second_b)
    after_different = base.evaluate_polynomial(first_out) != base.evaluate_polynomial(
        second_out
    )
    multipliers = shear_multipliers(index, hub, peer, program.family)
    gamma = shear_gamma(index, hub, peer, program.family)
    hessian_diagonal = [
        2
        * gamma
        * int(multipliers[exponent])
        * base.POWERS[exponent]
        % base.FIELD
        for exponent in range(base.GROUP_ORDER)
    ]
    return {
        "scope": "ONE_C5_PRIMARY_PUBLIC_SHEAR_ON_ARBITRARY_GROUP_ALGEBRA_INPUTS",
        "equal_pre_shear_t_to5_evaluations": before_equal,
        "different_post_shear_register_b_t_to5_evaluations": after_different,
        "single_character_quotient_rejected": before_equal and after_different,
        "quadratic_observable_hessian_rank": sum(value != 0 for value in hessian_diagonal),
        "linear_sketch_dimension_lower_bound_for_arbitrary_source_polynomial": base.GROUP_ORDER,
        "all_hessian_diagonal_entries_nonzero": all(value != 0 for value in hessian_diagonal),
    }


def character_coupling_diagnostic() -> dict[str, Any]:
    """Measure the exact character-space support of one public shear.

    If M is the character transform of the coefficient multiplier, the
    quadratic coefficient of input characters (j,l) in output character k is
    proportional to M[k-j-l].  A nonempty support therefore lets every input
    character influence every output character through at least one companion
    character.  We report the actual support rather than calling the
    multiplier transform dense (it is not).
    """

    program = compile_program(5, 1, "PRIMARY")
    hub = base.hub_index(0, program.family)
    peer = base.peer_order(hub, program.family)[0]
    multipliers = shear_multipliers(0, hub, peer, program.family).astype(np.int64)
    transformed = DFT @ multipliers % base.FIELD
    support = tuple(int(index) for index in np.flatnonzero(transformed))
    every_input_reaches_every_output = all(
        any(
            int(transformed[(output - source - companion) % base.GROUP_ORDER])
            != 0
            for companion in range(base.GROUP_ORDER)
        )
        for output in range(base.GROUP_ORDER)
        for source in range(base.GROUP_ORDER)
    )
    return {
        "scope": "ONE_C5_PRIMARY_PUBLIC_SHEAR",
        "multiplier_character_support_size": len(support),
        "multiplier_character_support_is_dense": len(support) == base.GROUP_ORDER,
        "each_output_has_a_quadratic_dependency_on_every_input_character": every_input_reaches_every_output,
        "all102_character_sectors_are_causally_coupled": every_input_reaches_every_output,
    }


def expect_failure(action: Callable[[], Any]) -> bool:
    try:
        action()
    except RuntimeError:
        return True
    return False


def controls() -> dict[str, bool]:
    program = compile_program(5, 2, "PRIMARY")
    seed = Carrier.fresh(5, "PRIMARY")
    seed_commitment = payload_commitment(seed)
    missing = Carrier.fresh(5, "PRIMARY")
    raw_forward(missing.registers, program)
    wrong = Carrier.fresh(5, "PRIMARY")
    raw_forward(wrong.registers, program)
    raw_inverse(wrong.registers, program, topology_mutation=1)
    reordered = Carrier.fresh(5, "PRIMARY")
    raw_forward(reordered.registers, program)
    raw_inverse(
        reordered.registers,
        program,
        inverse_order="INTERSECT_COMPOSE_SHEAR",
    )
    normal = Carrier.fresh(5, "PRIMARY")
    raw_forward(normal.registers, program)
    disabled = Carrier.fresh(5, "PRIMARY")
    raw_forward(disabled.registers, program, port_enabled=False)
    collision = single_character_collision()
    coupling = character_coupling_diagnostic()
    return {
        "missing_inverse_changes_payload": payload_commitment(missing) != seed_commitment,
        "wrong_inverse_changes_payload": payload_commitment(wrong) != seed_commitment,
        "reordered_inverse_changes_payload": payload_commitment(reordered) != seed_commitment,
        "null_carrier_rejected": expect_failure(
            lambda: begin(None, program, program.owner)
        ),
        "wrong_owner_rejected": expect_failure(
            lambda: begin(Carrier.fresh(5, "PRIMARY"), program, program.owner + 1)
        ),
        "wrong_type_rejected": expect_failure(
            lambda: begin(Carrier.fresh(7, "PRIMARY"), program, program.owner)
        ),
        "premature_projection_rejected": expect_failure(
            lambda: project(Carrier.fresh(5, "PRIMARY"), program, program.owner)
        ),
        "resident_projection_rejected": expect_failure(
            lambda: project_resident(normal, 0, 0)
        ),
        "disabled_port_changes_boundary": boundary(disabled.registers, program)
        != boundary(normal.registers, program),
        "single_character_collision_changes_later_boundary": collision[
            "single_character_quotient_rejected"
        ],
        "one_shear_quadratic_hessian_has_rank102": collision[
            "quadratic_observable_hessian_rank"
        ]
        == base.GROUP_ORDER,
        "every_character_sector_can_influence_every_output_sector": coupling[
            "each_output_has_a_quadratic_dependency_on_every_input_character"
        ],
    }


def reuse_results() -> tuple[dict[str, Any], dict[str, Any]]:
    first = compile_program(7, 1, "PRIMARY")
    second = compile_program(7, 4, "PRIMARY")
    reused = Carrier.fresh(7, "PRIMARY")
    backing = reused.backing_id()
    first_receipt = transaction(reused, first)
    second_receipt = transaction(reused, second)
    fresh = Carrier.fresh(7, "PRIMARY")
    fresh_receipt = transaction(fresh, second)
    unrelated = {
        "first_boundary_commitment": first_receipt["boundary_commitment"],
        "second_boundary_matches_fresh": second_receipt["boundary_commitment"]
        == fresh_receipt["boundary_commitment"],
        "second_resource_signature_matches_fresh": second_receipt[
            "transaction_stats"
        ]
        == fresh_receipt["transaction_stats"],
        "same_backing_consumed": reused.backing_id() == backing,
        "restoration_generation": reused.restoration_generation,
        "snapshot_used": False,
    }
    repeated_carrier = Carrier.fresh(5, "PRIMARY")
    repeated_program = compile_program(5, 2, "PRIMARY")
    repeated_backing = repeated_carrier.backing_id()
    reference: str | None = None
    stable = True
    for _ in range(8):
        receipt = transaction(repeated_carrier, repeated_program)
        if reference is None:
            reference = receipt["boundary_commitment"]
        stable &= receipt["boundary_commitment"] == reference
        stable &= repeated_carrier.backing_id() == repeated_backing
    repeated = {
        "cycles": 8,
        "boundary_stable": stable,
        "same_backing_stable": repeated_carrier.backing_id() == repeated_backing,
        "restoration_generation": repeated_carrier.restoration_generation,
        "snapshot_used": False,
    }
    return unrelated, repeated


def execute_case(interface: int, depth: int, family: str) -> dict[str, Any]:
    program = compile_program(interface, depth, family)
    carrier = Carrier.fresh(interface, family)
    receipt = transaction(carrier, program)
    characters = character_forward(program)
    character_result = character_boundary(characters, program)
    if character_result != receipt["_boundary"]:
        fail("coefficient carrier and full-character recurrence disagree")
    payload = REGISTERS * base.NODE_COUNT * interface * interface * base.GROUP_ORDER
    descriptor_bytes = len(canonical_json(program.descriptor()))
    return {
        "interface": interface,
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "boundary_commitment": receipt["boundary_commitment"],
        "forward_register_commitment": receipt["forward_register_commitment"],
        "phase_forward_stats": receipt["forward_stats"],
        "phase_transaction_stats": receipt["transaction_stats"],
        "phase_payload_field_cells": payload,
        "matched_classical_payload_field_cells": payload,
        "phase_to_matched_classical_payload_ratio": 1.0,
        "compiled_program_descriptor_bytes": descriptor_bytes,
        "maximum_named_shear_buffer_field_cells": receipt["forward_stats"][
            "maximum_shear_buffer_cells"
        ],
        "maximum_named_linear_moment_field_cells": receipt["forward_stats"][
            "linear"
        ]["maximum_moment_coefficient_cells"],
        "maximum_named_linear_output_field_cells": receipt["forward_stats"][
            "linear"
        ]["maximum_output_buffer_cells"],
        "physical_numpy_transient_peak_bytes_measured": False,
        "separate_full_character_recurrence_boundary_matches": True,
        "payload_restored_exactly": receipt["payload_restored_exactly"],
        "same_backing_restored": receipt["same_backing_restored"],
        "restoration_generation": receipt["restoration_generation"],
        "snapshot_used": False,
        "retained_inverse_history_cells": 0,
        "retained_restoration_baseline_cells": 0,
    }


def run() -> dict[str, Any]:
    cases = [
        execute_case(interface, depth, family)
        for interface in INTERFACES
        for family in FAMILIES
        for depth in DEPTHS
    ]
    control_results = controls()
    if not all(control_results.values()):
        fail(
            "quadratic shear controls failed: "
            + repr([key for key, value in control_results.items() if not value])
        )
    collision = single_character_collision()
    coupling = character_coupling_diagnostic()
    unrelated, repeated = reuse_results()
    if not all(
        (
            unrelated["second_boundary_matches_fresh"],
            unrelated["second_resource_signature_matches_fresh"],
            unrelated["same_backing_consumed"],
            repeated["boundary_stable"],
            repeated["same_backing_stable"],
        )
    ):
        fail("quadratic shear restored-carrier reuse failed")
    payload_by_interface = {
        str(interface): next(
            int(case["phase_payload_field_cells"])
            for case in cases
            if case["interface"] == interface
        )
        for interface in INTERFACES
    }
    return {
        "schema": "CAT_CAS_F103_C102_DUAL_REGISTER_QUADRATIC_PHASE_SHEAR_RELATION_NO_GO_RESULT_V1",
        "claim": CLAIM,
        "platform": "LINUX_DIRECT_PROCESS_SOFTWARE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "experiment": {
            "field": "F103",
            "phase_group": "C102",
            "interfaces": list(INTERFACES),
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "node_count": base.NODE_COUNT,
            "register_count": REGISTERS,
            "unresolved_source_register": True,
            "shears_per_layer": base.NODE_COUNT - 1,
            "same_source_port_multiple_consumers_established": False,
            "coefficientwise_quadratic_shear_is_reversible": True,
            "character_chart_operation": "QUADRATIC_ALL102_SECTOR_COUPLING_WITH_51_SUPPORT_MULTIPLIER_TRANSFORM",
            "final_boundary_only_evaluation": True,
            "compiler_inspects_final_answers": False,
        },
        "cases": cases,
        "single_character_quotient_attack": collision,
        "character_coupling_diagnostic": coupling,
        "controls": control_results,
        "restoration_and_reuse": {
            "actual_inverse_on_borrowed_carrier": True,
            "exact_payload_restoration": True,
            "same_backing_restoration": True,
            "snapshot_used": False,
            "retained_inverse_history_cells": 0,
            "retained_restoration_baseline_cells": 0,
            "unrelated_program_reuse": unrelated,
            "repeated_reuse": repeated,
        },
        "resource_accounting": {
            "phase_payload_field_cells_by_interface": payload_by_interface,
            "matched_classical_payload_field_cells_by_interface": payload_by_interface,
            "phase_to_matched_classical_payload_ratio": 1.0,
            "accepted_phase_and_classical_state_law": "204N2_FIELD_COORDINATES",
            "declared_forward_inverse_and_shear_arithmetic_counts_reported": True,
            "named_moment_output_shear_and_plan_buffer_cells_reported": True,
            "compiled_program_descriptor_counted": True,
            "controller_backend_traffic_bytes": 0,
            "snapshot_traffic_bytes": 0,
            "python_object_container_allocator_native_library_internal_workspace_and_whole_process_peaks_excluded": True,
            "physical_numpy_transient_peak_bytes_measured": False,
            "no_physical_peak_or_runtime_advantage_claimed": True,
            "optimal_classical_recurrence_claimed": False,
        },
        "matched_compact_classical": {
            "strongest_accepted_path": "IDENTICAL_DUAL_REGISTER_COEFFICIENT_RECURRENCE",
            "separate_parity_path": "FULL102_CHARACTER_RECURRENCE_WITH_EXPLICIT_COEFFICIENT_SHEAR_TRANSFORMS",
            "identical_state_coordinates": True,
            "identical_public_programs_and_boundaries": True,
            "single_character_recurrence_rejected": True,
            "linear_sketch_lower_bound_scope": "ONE_ARBITRARY_SOURCE_POLYNOMIAL_ONE_PUBLIC_SHEAR",
            "linear_sketch_dimension_lower_bound": base.GROUP_ORDER,
        },
        "no_smuggle": {
            "raw_final_boundaries_serialized": False,
            "resident_coefficients_or_characters_serialized": False,
            "ordinary_relation_tables_serialized": False,
            "assignment_or_truth_table_expansion": False,
            "boundary_and_state_commitments_only": True,
        },
        "claim_ceiling": "F103_C102_DUAL_REGISTER_QUADRATIC_SHEAR_ON_DECLARED_C5_C7_NINE_NODE_ROTATING_HUB_FAMILIES_THROUGH_DEPTH4_IN_LINUX_DIRECT_PROCESS_SOFTWARE",
        "preserved_subclaims": [
            "SINGLE_CHARACTER_EVALUATION_QUOTIENT_IS_REJECTED_BY_AN_EXPLICIT_COLLISION",
            "ONE_SHEAR_QUADRATIC_OBSERVABLE_HAS_EXACT_HESSIAN_RANK102",
            "COEFFICIENTWISE_SHEAR_COUPLES_ALL102_CHARACTER_SECTORS",
            "EXACT_ALGEBRAIC_RESTORATION_AND_SAME_BACKING_REUSE",
            "SEPARATE_FULL_CHARACTER_CLASSICAL_BOUNDARY_PARITY",
        ],
        "obstruction": "THE_NEW_REVERSIBLE_SHEAR_BREAKS_THE_SINGLE_CHARACTER_QUOTIENT_BUT_ONE_ARBITRARY_SOURCE_POLYNOMIAL_ALREADY_HAS_A_FULL_RANK102_QUADRATIC_OBSERVABLE_AND_THE_ACCEPTED_PHASE_CARRIER_IS_COORDINATE_IDENTICAL_TO_THE_STRONGEST_MATCHED_CLASSICAL_RECURRENCE",
        "not_established": [
            "SUB102_LINEAR_OBSERVABLE_QUOTIENT_FOR_ARBITRARY_SOURCE_POLYNOMIALS",
            "SHARED_LATENT_PORT_WITH_MULTIPLE_CONSUMERS",
            "COMPACT_OR_FIXED_RANK_GROWING_INTERFACE_CLOSURE",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_COMPUTATION",
        ],
        "next_obstruction": "CROSS_CHARACTER_NONLINEARITY_IS_CAUSALLY_RELEVANT_BUT_ITS_ONE_SHEAR_QUADRATIC_FORM_IS_ALREADY_FULL_RANK102_AND_THE_PHASE_AND_CLASSICAL_CARRIERS_ARE_IDENTICAL_SO_THE_NEXT_REPAIR_MUST_FIND_A_STRUCTURED_LOW_COMPLEXITY_NONLINEAR_PHASE_MANIFOLD_OR_MOVE_TO_MACHINE_ENFORCED_CUSTODY_WITHOUT_CALLING_EQUAL_COORDINATE_SIMULATION_A_RESOURCE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    arguments = parser.parse_args()
    result = run()
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(payload, encoding="utf-8")
    if arguments.summary:
        print(
            json.dumps(
                {
                    "claim": result["claim"],
                    "case_count": result["experiment"]["case_count"],
                    "collision": result["single_character_quotient_attack"],
                    "payload": result["resource_accounting"][
                        "phase_payload_field_cells_by_interface"
                    ],
                    "obstruction": result["obstruction"],
                },
                sort_keys=True,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
