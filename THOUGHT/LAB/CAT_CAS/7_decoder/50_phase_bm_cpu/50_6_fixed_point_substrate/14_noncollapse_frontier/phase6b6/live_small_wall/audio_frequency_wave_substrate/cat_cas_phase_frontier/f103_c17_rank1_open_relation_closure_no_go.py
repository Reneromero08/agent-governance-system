#!/usr/bin/env python3
"""Non-translation-invariant factored F103 open-relation diagnostic.

Each relation R(x,y)=a[x]b[y] is stored as two C17 factor vectors.  A shared
resident rank-one control C=u v^T acts on eight targets by reversible native
composition with I+lambda*C and by reversible native intersection with C.
The path never materializes a 17x17 relation table.  It restores and reuses
the same carrier, then compares against an executed classical recurrence that
retains only target factors and rematerializes the immutable control port.

This is bounded direct-process software, not CATVM custody or a distinct
phase resource.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


MODULUS = 103
CYCLE = 17
NODE_COUNT = 9
FACTOR_SIDES = 2
CONTROL_BANK = 0
TARGET_BANK = 1
DEPTHS = (1, 4, 16, 64, 256, 512)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
PORT_TYPE = "F103_FACTORED_RANK1_NON_TRANSLATION_INVARIANT_C17_TO_C17"
CLAIM = (
    "BOUNDED_EXACT_FACTORED_NON_TRANSLATION_INVARIANT_F103_C17_OPEN_"
    "RELATIONS_CLOSE_UNDER_IDENTITY_PLUS_RANK1_HIDDEN_INTERFACE_"
    "COMPOSITION_AND_RANK1_PARALLEL_INTERSECTION_ON_ONE_SHARED_"
    "UNRESOLVED_PORT_ACROSS8_NONCOMMUTING_CONSUMERS_WITHOUT_DENSE_"
    "RELATION_TABLES_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_"
    "REUSE_THROUGH_DEPTH512_BUT_COLLAPSE_TO_AN_EXECUTED306_COORDINATE_"
    "REMATERIALIZED_CONTROL_CLASSICAL_RECURRENCE_AND_RANK2_"
    "INTERSECTION_EXITS_THE_CLOSED_FAMILY"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def mod_array(value: np.ndarray) -> np.ndarray:
    return np.mod(value, MODULUS).astype(np.uint8)


def family_code(family: str) -> int:
    return {"PRIMARY": 3, "REUSE": 8, "ALTERNATE": 13}[family]


@dataclass(frozen=True)
class Program:
    depth: int
    family: str
    owner: int
    observation_linear: int
    observation_quadratic: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "CAT_CAS_F103_C17_RANK1_OPEN_RELATION_PROGRAM_V1",
            "depth": self.depth,
            "family": self.family,
            "owner": self.owner,
            "node_count": NODE_COUNT,
            "port_type": PORT_TYPE,
            "relation_semantics": "R_XY_EQUALS_LEFT_X_TIMES_RIGHT_Y",
            "topology": "PUBLIC_ROTATING_CONTROL_HUB8_COMPOSE_THEN_INTERSECT",
            "composition": "LEFT_MULTIPLY_BY_IDENTITY_PLUS_LAMBDA_U_V_TRANSPOSE",
            "intersection": "PARALLEL_HADAMARD_WITH_U_V_TRANSPOSE",
            "observation": [
                self.observation_linear,
                self.observation_quadratic,
            ],
        }

    def fingerprint(self) -> str:
        return digest_json(self.descriptor())


def compile_program(depth: int, family: str) -> Program:
    if not isinstance(depth, int) or not 1 <= depth <= 512:
        fail("rank-one relation program depth outside declared ceiling")
    if family not in FAMILIES:
        fail("rank-one relation program family outside declared set")
    return Program(
        depth=depth,
        family=family,
        owner=(0xC1710000 + 101 * depth + family_code(family)) & 0xFFFFFFFF,
        observation_linear=(9 * depth + 7 * len(family) + 2) % MODULUS,
        observation_quadratic=(
            (13 * depth + 5 * len(family) + 11) % MODULUS or 1
        ),
    )


def hub_index(index: int, family: str, mutation: int = 0) -> int:
    return (4 * index + family_code(family) + mutation) % NODE_COUNT


def peer_order(hub: int) -> list[int]:
    return [(hub + offset) % NODE_COUNT for offset in range(1, NODE_COUNT)]


def relation_offset(
    hub: int,
    peer: int,
    index: int,
    family: str,
    mutation: int = 0,
) -> int:
    return (
        5 * hub
        + 7 * peer
        + 3 * index
        + family_code(family)
        + mutation
    ) % CYCLE


def coupling_scalar(
    hub: int, peer: int, index: int, family: str
) -> int:
    del hub, peer, index, family
    return 1


def rotation_shift(node: int, index: int, family: str) -> int:
    if family == "PRIMARY":
        return (3 * node * node + 5 * index + index.bit_count() + 1) % CYCLE
    if family == "REUSE":
        return (7 * node + 2 * index + 3 * (index % 7) + 4) % CYCLE
    return (
        11 * node * node
        + 6 * index
        + 2 * (index ^ (index >> 1)).bit_count()
        + 5
    ) % CYCLE


def control_relation(node: int) -> np.ndarray:
    relation = np.empty((FACTOR_SIDES, CYCLE), dtype=np.uint8)
    for coordinate in range(CYCLE):
        relation[0, coordinate] = (
            1 + (7 * node * node + 11 * coordinate + 3 * node * coordinate) % 101
        )
        relation[1, coordinate] = 1
    while int(np.sum(relation[0], dtype=np.int64) % MODULUS) == MODULUS - 1:
        relation[0, -1] = int(relation[0, -1]) % 101 + 1
    return relation


def target_relation(node: int) -> np.ndarray:
    relation = np.empty((FACTOR_SIDES, CYCLE), dtype=np.uint8)
    for coordinate in range(CYCLE):
        relation[0, coordinate] = (
            2 + 17 * node + 7 * coordinate + 3 * node * coordinate
        ) % MODULUS
        relation[1, coordinate] = (
            5 + 19 * node + 11 * coordinate * coordinate + node * coordinate
        ) % MODULUS
    return relation


def seed_carrier() -> np.ndarray:
    state = np.empty(
        (2, NODE_COUNT, FACTOR_SIDES, CYCLE), dtype=np.uint8
    )
    for node in range(NODE_COUNT):
        state[CONTROL_BANK, node] = control_relation(node)
        state[TARGET_BANK, node] = target_relation(node)
    return state


def rotate(vector: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(vector, shift % CYCLE).astype(np.uint8, copy=False)


def dot(left: np.ndarray, right: np.ndarray) -> int:
    return int(
        np.dot(left.astype(np.int64), right.astype(np.int64)) % MODULUS
    )


@dataclass
class WorkStats:
    composition_actions: int = 0
    intersection_actions: int = 0
    composition_field_multiplications: int = 0
    intersection_field_multiplications: int = 0
    composition_accumulation_additions: int = 0
    factor_rotations: int = 0
    consumers: int = 0
    exact_cancellations: int = 0
    maximum_named_scratch_bytes: int = 0

    def descriptor(self) -> dict[str, int]:
        return {
            "composition_actions": self.composition_actions,
            "intersection_actions": self.intersection_actions,
            "composition_field_multiplications": self.composition_field_multiplications,
            "intersection_field_multiplications": self.intersection_field_multiplications,
            "total_relation_field_multiplications": (
                self.composition_field_multiplications
                + self.intersection_field_multiplications
            ),
            "composition_accumulation_additions": self.composition_accumulation_additions,
            "factor_rotations": self.factor_rotations,
            "consumers": self.consumers,
            "exact_cancellations": self.exact_cancellations,
            "maximum_named_scratch_bytes": self.maximum_named_scratch_bytes,
        }


def compose_identity_plus_rank1(
    target: np.ndarray,
    control: np.ndarray,
    coupling: int,
    *,
    inverse: bool,
    stats: WorkStats | None,
) -> None:
    self_pairing = dot(control[1], control[0])
    denominator = (1 + coupling * self_pairing) % MODULUS
    if denominator == 0:
        fail("identity-plus-rank-one composition is singular")
    coefficient = (
        -coupling * pow(denominator, -1, MODULUS)
        if inverse
        else coupling
    ) % MODULUS
    contraction = dot(control[1], target[0])
    term = mod_array(
        coefficient
        * contraction
        * control[0].astype(np.int64)
    )
    before = target[0].copy()
    target[0] = mod_array(target[0].astype(np.int64) + term.astype(np.int64))
    if stats is not None:
        stats.composition_actions += 1
        stats.composition_field_multiplications += 2 * CYCLE
        stats.composition_accumulation_additions += CYCLE - 1
        stats.exact_cancellations += int(
            np.count_nonzero((before != 0) & (term != 0) & (target[0] == 0))
        )
        stats.maximum_named_scratch_bytes = max(
            stats.maximum_named_scratch_bytes, 3 * CYCLE
        )


def intersect_rank1(
    target: np.ndarray,
    control: np.ndarray,
    *,
    inverse: bool,
    stats: WorkStats | None,
) -> None:
    before = target.copy()
    for side in range(FACTOR_SIDES):
        if inverse:
            factors = np.array(
                [pow(int(value), -1, MODULUS) for value in control[side]],
                dtype=np.uint8,
            )
        else:
            factors = control[side]
        target[side] = mod_array(
            target[side].astype(np.int64) * factors.astype(np.int64)
        )
    if stats is not None:
        stats.intersection_actions += 1
        stats.intersection_field_multiplications += 2 * CYCLE
        stats.exact_cancellations += int(
            np.count_nonzero((before != 0) & (target == 0))
        )
        stats.maximum_named_scratch_bytes = max(
            stats.maximum_named_scratch_bytes, 2 * CYCLE
        )


def control_view(
    controls: np.ndarray,
    hub: int,
    peer: int,
    index: int,
    family: str,
    offset_mutation: int = 0,
) -> np.ndarray:
    offset = relation_offset(
        hub, peer, index, family, offset_mutation
    )
    return np.stack(
        (rotate(controls[hub, 0], offset), rotate(controls[hub, 1], -offset)),
        axis=0,
    )


def raw_forward(
    controls: np.ndarray,
    targets: np.ndarray,
    program: Program,
    *,
    action_order: str = "COMPOSE_INTERSECT",
    port_enabled: bool = True,
    hub_mutation: int = 0,
    offset_mutation: int = 0,
    stats: WorkStats | None = None,
) -> None:
    actions = (
        (compose_identity_plus_rank1, intersect_rank1)
        if action_order == "COMPOSE_INTERSECT"
        else (intersect_rank1, compose_identity_plus_rank1)
    )
    for index in range(program.depth):
        for node in range(NODE_COUNT):
            shift = rotation_shift(node, index, program.family)
            targets[node, 0] = rotate(targets[node, 0], shift)
            targets[node, 1] = rotate(targets[node, 1], -shift)
            if stats is not None:
                stats.factor_rotations += 2
        hub = hub_index(index, program.family, hub_mutation)
        for peer in peer_order(hub):
            if not port_enabled:
                continue
            control = control_view(
                controls,
                hub,
                peer,
                index,
                program.family,
                offset_mutation,
            )
            coupling = coupling_scalar(hub, peer, index, program.family)
            for action in actions:
                if action is compose_identity_plus_rank1:
                    action(
                        targets[peer],
                        control,
                        coupling,
                        inverse=False,
                        stats=stats,
                    )
                else:
                    action(
                        targets[peer],
                        control,
                        inverse=False,
                        stats=stats,
                    )
            if stats is not None:
                stats.consumers += 1


def raw_inverse(
    controls: np.ndarray,
    targets: np.ndarray,
    program: Program,
    *,
    assumed_action_order: str = "COMPOSE_INTERSECT",
    hub_mutation: int = 0,
    offset_mutation: int = 0,
) -> None:
    actions = (
        (intersect_rank1, compose_identity_plus_rank1)
        if assumed_action_order == "COMPOSE_INTERSECT"
        else (compose_identity_plus_rank1, intersect_rank1)
    )
    for index in reversed(range(program.depth)):
        hub = hub_index(index, program.family, hub_mutation)
        peers = peer_order(hub)
        peers.reverse()
        for peer in peers:
            control = control_view(
                controls,
                hub,
                peer,
                index,
                program.family,
                offset_mutation,
            )
            coupling = coupling_scalar(hub, peer, index, program.family)
            for action in actions:
                if action is compose_identity_plus_rank1:
                    action(
                        targets[peer],
                        control,
                        coupling,
                        inverse=True,
                        stats=None,
                    )
                else:
                    action(
                        targets[peer],
                        control,
                        inverse=True,
                        stats=None,
                    )
        for node in range(NODE_COUNT):
            shift = rotation_shift(node, index, program.family)
            targets[node, 0] = rotate(targets[node, 0], -shift)
            targets[node, 1] = rotate(targets[node, 1], shift)


@dataclass
class Carrier:
    state: np.ndarray
    port_type: str = PORT_TYPE
    stage: str = "RESTORED"
    active_program: str | None = None
    active_owner: int | None = None
    restoration_generation: int = 0
    projection_calls: int = 0
    snapshot_reload_used: bool = False
    stats: WorkStats = field(default_factory=WorkStats)

    @classmethod
    def seal(cls) -> "Carrier":
        return cls(seed_carrier())

    @property
    def controls(self) -> np.ndarray:
        return self.state[CONTROL_BANK]

    @property
    def targets(self) -> np.ndarray:
        return self.state[TARGET_BANK]

    @property
    def backing_identity(self) -> int:
        return int(self.state.__array_interface__["data"][0])


def state_commitment(carrier: Carrier) -> str:
    return hashlib.sha256(carrier.state.tobytes()).hexdigest()


def begin_forward(carrier: Carrier, program: Program, owner: int) -> None:
    if not isinstance(carrier, Carrier):
        fail("null or wrong rank-one relation carrier")
    if carrier.port_type != PORT_TYPE:
        fail("rank-one relation port type mismatch")
    if owner != program.owner:
        fail("rank-one relation owner mismatch")
    if carrier.stage != "RESTORED" or carrier.active_program is not None:
        fail("rank-one relation carrier is already leased")
    carrier.stage = "FORWARD_RANK1_RELATION"
    carrier.active_program = program.fingerprint()
    carrier.active_owner = owner
    carrier.projection_calls = 0
    carrier.stats = WorkStats()


def require_owned(
    carrier: Carrier, program: Program, owner: int, stage: str
) -> None:
    if not isinstance(carrier, Carrier):
        fail("null or wrong rank-one relation carrier")
    if (
        carrier.port_type != PORT_TYPE
        or carrier.stage != stage
        or carrier.active_program != program.fingerprint()
        or carrier.active_owner != owner
        or owner != program.owner
    ):
        fail("rank-one relation owner, type, program, or stage changed")


def forward(carrier: Carrier, program: Program, owner: int) -> None:
    require_owned(carrier, program, owner, "FORWARD_RANK1_RELATION")
    raw_forward(carrier.controls, carrier.targets, program, stats=carrier.stats)
    carrier.stage = "FORWARDED_RANK1_RELATION"


def inverse(carrier: Carrier, program: Program, owner: int) -> None:
    require_owned(carrier, program, owner, "FORWARDED_RANK1_RELATION")
    raw_inverse(carrier.controls, carrier.targets, program)
    carrier.stage = "RESTORED"
    carrier.active_program = None
    carrier.active_owner = None
    carrier.restoration_generation += 1


def boundary_from_targets(
    targets: np.ndarray, program: Program
) -> tuple[int, ...]:
    output = np.zeros(CYCLE, dtype=np.int64)
    for node in range(NODE_COUNT):
        right_sum = int(np.sum(targets[node, 1], dtype=np.int64) % MODULUS)
        weight = (
            program.observation_quadratic * node * node
            + program.observation_linear * (node + 1)
            + 1
        ) % MODULUS
        output += (
            weight
            * right_sum
            * targets[node, 0].astype(np.int64)
        )
    return tuple(int(value) for value in np.mod(output, MODULUS))


def project(
    carrier: Carrier, program: Program, owner: int
) -> tuple[int, ...]:
    require_owned(carrier, program, owner, "FORWARDED_RANK1_RELATION")
    carrier.projection_calls += 1
    return boundary_from_targets(carrier.targets, program)


def project_resident_port(_carrier: Carrier, _node: int) -> None:
    fail("resident rank-one relation port projection is forbidden")


@dataclass
class ClassicalStats:
    control_coordinate_generation_steps: int = 0
    relation_stats: WorkStats = field(default_factory=WorkStats)

    def descriptor(self) -> dict[str, Any]:
        return {
            "control_coordinate_generation_steps": self.control_coordinate_generation_steps,
            "retained_control_coordinates": 0,
            "relation_stats": self.relation_stats.descriptor(),
        }


def rematerialized_classical_forward(
    program: Program,
) -> tuple[np.ndarray, ClassicalStats]:
    targets = np.stack(
        [target_relation(node) for node in range(NODE_COUNT)], axis=0
    )
    stats = ClassicalStats()
    for index in range(program.depth):
        for node in range(NODE_COUNT):
            shift = rotation_shift(node, index, program.family)
            targets[node, 0] = rotate(targets[node, 0], shift)
            targets[node, 1] = rotate(targets[node, 1], -shift)
            stats.relation_stats.factor_rotations += 2
        hub = hub_index(index, program.family)
        base_control = control_relation(hub)
        stats.control_coordinate_generation_steps += 2 * CYCLE
        for peer in peer_order(hub):
            offset = relation_offset(hub, peer, index, program.family)
            control = np.stack(
                (
                    rotate(base_control[0], offset),
                    rotate(base_control[1], -offset),
                ),
                axis=0,
            )
            coupling = coupling_scalar(hub, peer, index, program.family)
            compose_identity_plus_rank1(
                targets[peer],
                control,
                coupling,
                inverse=False,
                stats=stats.relation_stats,
            )
            intersect_rank1(
                targets[peer], control, inverse=False, stats=stats.relation_stats
            )
            stats.relation_stats.consumers += 1
    return targets, stats


def execute_case(depth: int, family: str) -> dict[str, Any]:
    program = compile_program(depth, family)
    carrier = Carrier.seal()
    backing = carrier.backing_identity
    initial = state_commitment(carrier)
    initial_controls = carrier.controls.copy()
    generation = carrier.restoration_generation
    begin_forward(carrier, program, program.owner)
    forward(carrier, program, program.owner)
    final = state_commitment(carrier)
    boundary = project(carrier, program, program.owner)
    classical_targets, classical_stats = rematerialized_classical_forward(program)
    target_match = np.array_equal(carrier.targets, classical_targets)
    classical_boundary = boundary_from_targets(classical_targets, program)
    controls_unchanged = np.array_equal(carrier.controls, initial_controls)
    support = np.count_nonzero(carrier.targets, axis=3 if carrier.targets.ndim == 4 else 2)
    inverse(carrier, program, program.owner)
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "public_program_json_bytes": len(canonical_json(program.descriptor())),
        "initial_commitment": initial,
        "final_commitment": final,
        "boundary": list(boundary),
        "rematerialized_classical_boundary": list(classical_boundary),
        "target_factors_identical_to_rematerialized_classical_recurrence": target_match,
        "boundary_identical_to_rematerialized_classical_recurrence": boundary == classical_boundary,
        "resident_controls_unchanged_during_forward": controls_unchanged,
        "minimum_final_factor_support": int(np.min(support)),
        "maximum_final_factor_support": int(np.max(support)),
        "exact_restoration": state_commitment(carrier) == initial,
        "same_backing": carrier.backing_identity == backing,
        "restoration_generation_before": generation,
        "restoration_generation_after": carrier.restoration_generation,
        "projection_calls": carrier.projection_calls,
        "snapshot_reload_used": carrier.snapshot_reload_used,
        "inverse_history_cells": 0,
        "retained_restoration_baseline_cells": 0,
        "phase_stats": carrier.stats.descriptor(),
        "rematerialized_classical_stats": classical_stats.descriptor(),
    }


def transaction(carrier: Carrier, program: Program) -> dict[str, Any]:
    backing = carrier.backing_identity
    initial = state_commitment(carrier)
    generation = carrier.restoration_generation
    begin_forward(carrier, program, program.owner)
    forward(carrier, program, program.owner)
    boundary = project(carrier, program, program.owner)
    final = state_commitment(carrier)
    inverse(carrier, program, program.owner)
    return {
        "boundary": list(boundary),
        "final_commitment": final,
        "exact_restoration": state_commitment(carrier) == initial,
        "same_backing": carrier.backing_identity == backing,
        "generation_before": generation,
        "generation_after": carrier.restoration_generation,
        "resource_signature": {
            "resident_field_coordinates": int(carrier.state.size),
            "resident_bytes": int(carrier.state.nbytes),
            "phase_stats": carrier.stats.descriptor(),
        },
    }


def reuse_controls() -> tuple[dict[str, Any], dict[str, Any]]:
    carrier = Carrier.seal()
    first = transaction(carrier, compile_program(37, "PRIMARY"))
    backing = carrier.backing_identity
    second_program = compile_program(311, "ALTERNATE")
    second = transaction(carrier, second_program)
    fresh = transaction(Carrier.seal(), second_program)
    unrelated = {
        "first_exact_restoration": first["exact_restoration"],
        "second_exact_restoration": second["exact_restoration"],
        "same_backing_across_programs": carrier.backing_identity == backing,
        "second_boundary_matches_fresh": second["boundary"] == fresh["boundary"],
        "second_final_commitment_matches_fresh": second["final_commitment"] == fresh["final_commitment"],
        "resource_signature_matches_fresh": second["resource_signature"] == fresh["resource_signature"],
        "restoration_generation": carrier.restoration_generation,
        "snapshot_reload_used": carrier.snapshot_reload_used,
    }
    repeated_carrier = Carrier.seal()
    repeated_backing = repeated_carrier.backing_identity
    initial = state_commitment(repeated_carrier)
    boundaries: set[tuple[int, ...]] = set()
    for _ in range(64):
        result = transaction(repeated_carrier, compile_program(8, "REUSE"))
        boundaries.add(tuple(result["boundary"]))
    repeated = {
        "cycles": 64,
        "exact_restoration": state_commitment(repeated_carrier) == initial,
        "same_backing": repeated_carrier.backing_identity == repeated_backing,
        "restoration_generation": repeated_carrier.restoration_generation,
        "stable_boundary_count": len(boundaries),
        "snapshot_reload_used": repeated_carrier.snapshot_reload_used,
    }
    return unrelated, repeated


def streamed_semantic_controls() -> dict[str, Any]:
    control = control_relation(2)
    target = target_relation(5)
    coupling = 7
    composed = target.copy()
    compose_identity_plus_rank1(
        composed, control, coupling, inverse=False, stats=None
    )
    intersected = composed.copy()
    intersect_rank1(intersected, control, inverse=False, stats=None)
    composition_checks = 0
    intersection_checks = 0
    contraction = dot(control[1], target[0])
    for x_value in range(CYCLE):
        for y_value in range(CYCLE):
            original = int(target[0, x_value]) * int(target[1, y_value])
            streamed_composition = (
                original
                + coupling
                * int(control[0, x_value])
                * contraction
                * int(target[1, y_value])
            ) % MODULUS
            factored_composition = (
                int(composed[0, x_value]) * int(composed[1, y_value])
            ) % MODULUS
            if streamed_composition != factored_composition:
                fail("streamed rank-one composition semantics failed")
            composition_checks += 1
            streamed_intersection = (
                streamed_composition
                * int(control[0, x_value])
                * int(control[1, y_value])
            ) % MODULUS
            factored_intersection = (
                int(intersected[0, x_value])
                * int(intersected[1, y_value])
            ) % MODULUS
            if streamed_intersection != factored_intersection:
                fail("streamed rank-one intersection semantics failed")
            intersection_checks += 1
    return {
        "composition_scalar_checks": composition_checks,
        "intersection_scalar_checks": intersection_checks,
        "dense_relation_tables_materialized": 0,
        "assignment_expansions_materialized": 0,
    }


def rank2_escape_certificate() -> dict[str, Any]:
    target = target_relation(4)
    first = control_relation(1)
    second = target_relation(7)

    def entry(x_value: int, y_value: int) -> int:
        base = int(target[0, x_value]) * int(target[1, y_value])
        control_sum = (
            int(first[0, x_value]) * int(first[1, y_value])
            + int(second[0, x_value]) * int(second[1, y_value])
        )
        return base * control_sum % MODULUS

    witness: dict[str, int] | None = None
    scalar_entries_checked = 0
    for x0 in range(CYCLE):
        for x1 in range(x0 + 1, CYCLE):
            for y0 in range(CYCLE):
                for y1 in range(y0 + 1, CYCLE):
                    determinant = (
                        entry(x0, y0) * entry(x1, y1)
                        - entry(x0, y1) * entry(x1, y0)
                    ) % MODULUS
                    scalar_entries_checked += 4
                    if determinant != 0:
                        witness = {
                            "x0": x0,
                            "x1": x1,
                            "y0": y0,
                            "y1": y1,
                            "minor_determinant": determinant,
                        }
                        break
                if witness is not None:
                    break
            if witness is not None:
                break
        if witness is not None:
            break
    if witness is None:
        fail("rank-two escape witness not found")
    return {
        "rank_upper_bound": 2,
        "nonzero_two_by_two_minor": witness,
        "exact_rank": 2,
        "rank1_closed_family_exited": True,
        "scalar_entries_streamed": scalar_entries_checked,
        "dense_relation_tables_materialized": 0,
    }


def controls() -> dict[str, bool]:
    program = compile_program(4, "PRIMARY")
    seed = seed_carrier()
    missing = seed.copy()
    raw_forward(missing[CONTROL_BANK], missing[TARGET_BANK], program)
    wrong = seed.copy()
    raw_forward(wrong[CONTROL_BANK], wrong[TARGET_BANK], program)
    raw_inverse(
        wrong[CONTROL_BANK], wrong[TARGET_BANK], program, offset_mutation=1
    )
    reordered = seed.copy()
    raw_forward(reordered[CONTROL_BANK], reordered[TARGET_BANK], program)
    raw_inverse(
        reordered[CONTROL_BANK],
        reordered[TARGET_BANK],
        program,
        assumed_action_order="INTERSECT_COMPOSE",
    )
    normal = seed.copy()
    raw_forward(normal[CONTROL_BANK], normal[TARGET_BANK], program)
    disabled = seed.copy()
    raw_forward(
        disabled[CONTROL_BANK],
        disabled[TARGET_BANK],
        program,
        port_enabled=False,
    )
    swapped = seed.copy()
    raw_forward(
        swapped[CONTROL_BANK],
        swapped[TARGET_BANK],
        program,
        action_order="INTERSECT_COMPOSE",
    )
    mutated = seed.copy()
    raw_forward(
        mutated[CONTROL_BANK],
        mutated[TARGET_BANK],
        program,
        hub_mutation=1,
    )
    null_rejected = False
    try:
        begin_forward(None, program, program.owner)  # type: ignore[arg-type]
    except RuntimeError:
        null_rejected = True
    wrong_type = Carrier.seal()
    wrong_type.port_type = "F103_DENSE_RELATION"
    wrong_type_rejected = False
    try:
        begin_forward(wrong_type, program, program.owner)
    except RuntimeError:
        wrong_type_rejected = True
    wrong_owner_rejected = False
    try:
        begin_forward(Carrier.seal(), program, program.owner ^ 1)
    except RuntimeError:
        wrong_owner_rejected = True
    leased = Carrier.seal()
    begin_forward(leased, program, program.owner)
    premature_rejected = False
    try:
        project(leased, program, program.owner)
    except RuntimeError:
        premature_rejected = True
    resident_rejected = False
    try:
        project_resident_port(leased, 0)
    except RuntimeError:
        resident_rejected = True
    return {
        "missing_inverse_changes_state": not np.array_equal(missing, seed),
        "wrong_inverse_changes_state": not np.array_equal(wrong, seed),
        "reordered_inverse_changes_state": not np.array_equal(reordered, seed),
        "null_carrier_rejected": null_rejected,
        "wrong_relation_type_rejected": wrong_type_rejected,
        "wrong_owner_rejected": wrong_owner_rejected,
        "premature_projection_rejected": premature_rejected,
        "resident_port_projection_rejected": resident_rejected,
        "null_port_changes_boundary": boundary_from_targets(normal[TARGET_BANK], program)
        != boundary_from_targets(disabled[TARGET_BANK], program),
        "composition_intersection_order_changes_boundary": boundary_from_targets(normal[TARGET_BANK], program)
        != boundary_from_targets(swapped[TARGET_BANK], program),
        "topology_mutation_changes_boundary": boundary_from_targets(normal[TARGET_BANK], program)
        != boundary_from_targets(mutated[TARGET_BANK], program),
        "resident_controls_remain_unmodified": np.array_equal(
            normal[CONTROL_BANK], seed[CONTROL_BANK]
        ),
    }


def run() -> dict[str, Any]:
    cases = [
        execute_case(depth, family)
        for family in FAMILIES
        for depth in DEPTHS
    ]
    all_exact = all(
        case["target_factors_identical_to_rematerialized_classical_recurrence"]
        and case["boundary_identical_to_rematerialized_classical_recurrence"]
        and case["resident_controls_unchanged_during_forward"]
        and case["exact_restoration"]
        and case["same_backing"]
        and case["restoration_generation_after"]
        == case["restoration_generation_before"] + 1
        and case["projection_calls"] == 1
        and not case["snapshot_reload_used"]
        and case["inverse_history_cells"] == 0
        and case["retained_restoration_baseline_cells"] == 0
        for case in cases
    )
    if not all_exact:
        fail("one or more rank-one open-relation cases failed")
    control_results = controls()
    if not all(control_results.values()):
        fail(
            "rank-one relation controls failed: "
            + repr([name for name, value in control_results.items() if not value])
        )
    semantic = streamed_semantic_controls()
    rank2 = rank2_escape_certificate()
    unrelated, repeated = reuse_controls()
    if not all(
        (
            unrelated["first_exact_restoration"],
            unrelated["second_exact_restoration"],
            unrelated["same_backing_across_programs"],
            unrelated["second_boundary_matches_fresh"],
            unrelated["second_final_commitment_matches_fresh"],
            unrelated["resource_signature_matches_fresh"],
            not unrelated["snapshot_reload_used"],
            repeated["exact_restoration"],
            repeated["same_backing"],
            repeated["restoration_generation"] == 64,
            repeated["stable_boundary_count"] == 1,
            not repeated["snapshot_reload_used"],
        )
    ):
        fail("rank-one relation reuse failed")
    resident_cells = 2 * NODE_COUNT * FACTOR_SIDES * CYCLE
    target_cells = NODE_COUNT * FACTOR_SIDES * CYCLE
    maximum_program_bytes = max(
        case["public_program_json_bytes"] for case in cases
    )
    maximum_scratch = max(
        case["phase_stats"]["maximum_named_scratch_bytes"]
        for case in cases
    )
    maximum_work = max(
        case["phase_stats"]["total_relation_field_multiplications"]
        for case in cases
    )
    cancellations = sum(
        case["phase_stats"]["exact_cancellations"] for case in cases
    )
    return {
        "schema": "CAT_CAS_F103_C17_RANK1_OPEN_RELATION_CLOSURE_NO_GO_RESULT_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_scope": "LINUX_DIRECT_PROCESS_EXACT_FINITE_FIELD_FACTORED_OPEN_RELATION_SOFTWARE",
        "execution_scope": {
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "all_cases_exact": all_exact,
            "public_topology_compilation_reads_final_answers": False,
            "catvm_machine_boundary_used": False,
        },
        "relation_law": {
            "field_modulus": MODULUS,
            "boundary_cardinality": CYCLE,
            "port_type": PORT_TYPE,
            "translation_invariant": False,
            "rank1_factor_coordinates_per_relation": 2 * CYCLE,
            "implicit_dense_relation_cells_per_relation": CYCLE * CYCLE,
            "materialized_dense_relation_table_cells": 0,
            "materialized_assignment_expansion_cells": 0,
            "hidden_interface_composition": "LEFT_ACTION_BY_IDENTITY_PLUS_RANK1_RELATION",
            "parallel_intersection": "HADAMARD_PRODUCT_WITH_RANK1_RELATION",
            "shared_unresolved_port_consumers_per_layer": 8,
            "resident_port_projection_before_boundary": False,
            "streamed_semantic_controls": semantic,
            "rank2_escape_certificate": rank2,
        },
        "carrier_law": {
            "resident_control_relations": NODE_COUNT,
            "resident_target_relations": NODE_COUNT,
            "resident_field_coordinates": resident_cells,
            "resident_bytes": resident_cells,
            "direct_process_type_and_owner_checks_observed": True,
            "restoration_generation_sequence_observed": True,
            "machine_enforced_generation_or_lease_custody": False,
            "retained_public_plan_cells": 0,
        },
        "matched_classical_recurrence": {
            "implementation": "EXECUTED_REMATERIALIZED_IMMUTABLE_CONTROL_RANK1_FACTOR_RECURRENCE",
            "full_target_state_and_boundary_match_every_case": True,
            "resident_target_field_coordinates": target_cells,
            "resident_target_bytes": target_cells,
            "maximum_control_rematerialization_scratch_coordinates": 2 * CYCLE,
            "maximum_phase_resident_coordinates": resident_cells,
            "maximum_relation_field_multiplications_each": maximum_work,
            "phase_to_classical_resident_dimension_ratio": 2,
            "optimal_compact_classical_recurrence_claimed": False,
        },
        "restoration": {
            "carrier_classification": "EXACT_ALGEBRAIC_RESTORATION",
            "transient_buffers_classification": "NO_RESTORATION_CLAIM",
            "same_backing": True,
            "inverse_history_cells": 0,
            "retained_restoration_baseline_cells": 0,
            "snapshot_reload_used": False,
            "unrelated_program_reuse": unrelated,
            "repeated_reuse": repeated,
        },
        "controls": control_results,
        "resource_accounting": {
            "phase_resident_uint8_cells": resident_cells,
            "phase_resident_bytes": resident_cells,
            "rematerialized_classical_resident_uint8_cells": target_cells,
            "rematerialized_classical_resident_bytes": target_cells,
            "maximum_named_phase_warm_live_bytes": resident_cells + maximum_scratch + maximum_program_bytes,
            "maximum_named_rematerialized_classical_warm_live_bytes": target_cells + 2 * CYCLE + maximum_scratch + maximum_program_bytes,
            "exact_cancellations_observed": cancellations,
            "excluded": [
                "PYTHON_CONTAINER_OVERHEAD",
                "PYTHON_OBJECT_ALLOCATOR",
                "NUMPY_AND_NATIVE_LIBRARY_INTERNAL_STORAGE",
                "WHOLE_PROCESS_PEAK",
            ],
        },
        "cases": cases,
        "claim_ceiling": (
            "F103_NON_TRANSLATION_INVARIANT_RANK1_FACTORED_C17_TO_C17_"
            "RELATIONS_WITH_IDENTITY_PLUS_RANK1_COMPOSITION_AND_RANK1_"
            "INTERSECTION_ON_THE_DECLARED9_CONTROL9_TARGET_ROTATING_HUB_"
            "TOPOLOGY_ACROSS18_CASES_THROUGH_DEPTH512_IN_LINUX_DIRECT_"
            "PROCESS_SOFTWARE"
        ),
        "not_established": [
            "RANK2_OR_GENERAL_RELATION_CLOSURE",
            "ARBITRARY_PORT_ARITY_OR_GRAPH_TOPOLOGY",
            "MACHINE_ENFORCED_GENERATION_OR_LEASE_CUSTODY",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "PHYSICAL_BIT_REPLACEMENT",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
        "next_obstruction": (
            "NON_TRANSLATION_INVARIANT_RANK1_RELATIONS_CLOSE_NATIVELY_BUT_"
            "A_TWO_TERM_CONTROL_FORCES_EXACT_RANK2_WHILE_THE_ACCEPTED_"
            "FAMILY_HAS_A_STRICTLY_SMALLER306_COORDINATE_REMATERIALIZED_"
            "CONTROL_CLASSICAL_RECURRENCE"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run()
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
